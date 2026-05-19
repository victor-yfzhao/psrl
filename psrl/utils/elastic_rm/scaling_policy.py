import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.logger import DualOutputHandler, FileOnlyHandler

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

@dataclass
class InstanceSignal:
    role_name: PSRL_Role
    model_name: str
    instance_id: int
    is_awaken: bool
    kv_cache_utilization: float
    running_queue_num: int
    waiting_queue_num: int
    generation_throughput: float
    total_token_num: int
    snapshot_timestamp: str | None = None
    # Global placement-group bundle indices [start, end) occupied by this instance (half-open).
    # Populated by ElasticExecutor from SubRayResourcePool bundle_range; None if unknown.
    bundle_keys: frozenset[int] | None = None


@dataclass
class ScalingAction:
    action_type: str  # "scale_up" or "scale_down"
    role_name: PSRL_Role
    model_name: str
    num_instances: int = 1
    preferred_instance_ids: list[int] | None = None
    reason: str = ""
    # When scale_up must evict another role first: preferred SLEEP targets (same dict shape
    # as executor pre_sleep). Only force_wake colocated path sets this today.
    pre_sleep_other_preferred: list[dict[str, Any]] | None = None
    # Optional same-role migration targets to wake before sleeping pre_sleep_other_preferred.
    # Used when a larger-parallelism instance can be opened by moving smaller instances
    # onto completely free devices first.
    pre_wake_other_preferred: list[dict[str, Any]] | None = None


@dataclass
class ScalingDecision:
    actions: list[ScalingAction]
    reason: str
    estimated_lambda: float
    role_to_total_mu: dict[PSRL_Role, float]


class ThroughputProfileLoader:
    """
    Loader for fitted throughput formulas in throughput_model/*_token.json.
    The formula is evaluated by running-queue length x:
        mu(x) = A * (1 - (B * x + 1)^(-k))
    """

    def __init__(
        self,
        profile_paths: dict[str, str] | None = None,
        throughput_model_dir: str | None = None,
        preferred_output_len: int = 1024,
    ):
        self.profile_paths = profile_paths or {}
        self.throughput_model_dir = throughput_model_dir
        self.preferred_output_len = int(preferred_output_len)
        self._profile_cache: dict[str, dict] = {}
        

    def _load_json(self, path: str) -> dict:
        if path in self._profile_cache:
            return self._profile_cache[path]
        if not path or not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        self._profile_cache[path] = payload
        return payload

    def _load_throughput_formula_params(self, model_name: str) -> dict | None:
        if not self.throughput_model_dir:
            return None
        token_profile_path = os.path.join(self.throughput_model_dir, f"{model_name}_token.json")
        payload = self._load_json(token_profile_path)
        if not payload:
            return None
        if payload.get("metric") != "tok":
            return None
        fit_by_output_len = payload.get("fit_by_output_len", {})
        preferred_key = str(self.preferred_output_len)
        if preferred_key in fit_by_output_len:
            return fit_by_output_len[preferred_key]
        if fit_by_output_len:
            # Deterministic fallback: use smallest output-len bucket.
            sorted_keys = sorted(fit_by_output_len.keys(), key=lambda x: int(x))
            return fit_by_output_len[sorted_keys[0]]
        return None

    def estimate_mu_by_running_queue(
        self,
        model_name: str,
        running_queue_num: float,
        fallback_mu: float | None = None,
    ) -> float | None:
        running_queue_num = max(float(running_queue_num), 0.0)
        # Priority 1: throughput_model fitted formula (token throughput, output-len=1024).
        formula_params = self._load_throughput_formula_params(model_name)
        if formula_params is not None:
            required = {"A", "B", "k"}
            if required.issubset(set(formula_params.keys())):
                A = float(formula_params["A"])
                B = float(formula_params["B"])
                k = float(formula_params["k"])
                # Formula: A*(1-(B*x+1)^(-k)), x = running queue num.
                mu = A * (1.0 - math.pow(B * running_queue_num + 1.0, -k))
                return max(mu, 0.0)

        # Priority 2: fallback to provided scalar (typically runtime throughput).
        if fallback_mu is None:
            return None
        return max(float(fallback_mu), 0.0)

    @staticmethod
    def _resolve_new_schema_entry(payload: dict, signal: InstanceSignal, role_name: PSRL_Role) -> dict | None:
        role_section = payload.get("roles", {}).get(role_name, {})
        model_section = role_section.get(signal.model_name, {})
        if not model_section:
            return None
        # Optional per-instance override in new schema
        instance_section = model_section.get("instances", {}).get(str(signal.instance_id), {})
        if instance_section:
            return instance_section
        return model_section

    @staticmethod
    def _lookup_throughput_from_table(entry: dict, signal: InstanceSignal) -> float | None:
        # Priority: explicit scalar -> queue-based map -> threshold table.
        if "default_mu" in entry:
            return float(entry["default_mu"])

        running_queue_mu = entry.get("mu_by_running_queue", {})
        if running_queue_mu:
            key = str(signal.running_queue_num)
            if key in running_queue_mu:
                return float(running_queue_mu[key])
            # nearest lower key as a stable fallback
            numeric_keys = sorted(int(k) for k in running_queue_mu.keys() if str(k).isdigit())
            lower_keys = [k for k in numeric_keys if k <= signal.running_queue_num]
            if lower_keys:
                return float(running_queue_mu[str(lower_keys[-1])])

        mu_table = entry.get("mu_table", [])
        if not mu_table:
            return None
        for row in mu_table:
            max_q = int(row.get("max_running_queue_num", 10**9))
            max_tokens = int(row.get("max_total_token_num", 10**18))
            if signal.running_queue_num <= max_q and signal.total_token_num <= max_tokens:
                if "throughput" in row:
                    return float(row["throughput"])
                if "mu" in row:
                    return float(row["mu"])
        return None

    def estimate_instance_mu_with_source(self, signal: InstanceSignal) -> tuple[float, str]:
        formula_mu = self.estimate_mu_by_running_queue(
            model_name=signal.model_name,
            running_queue_num=float(signal.running_queue_num),
            fallback_mu=None,
        )
        if formula_mu is not None:
            return formula_mu, "formula"

        # Final fallback: trust runtime throughput if no formula is found.
        return max(0.0, float(signal.generation_throughput)), "fallback_runtime"

    def estimate_instance_mu(self, signal: InstanceSignal) -> float:
        mu, _ = self.estimate_instance_mu_with_source(signal)
        return mu


class ScalingPolicy:
    def __init__(self, config: dict, policy_config: dict | None = None):
        self.config = config
        cfg = policy_config or {}
        self.enable = bool(cfg.get("enable_policy", cfg.get("enable", False)))
        self.monitor_interval_ms = int(cfg.get("monitor_interval_ms", 1000))
        self.theta_low = float(cfg.get("theta_low", 0.3))
        self.theta_max = float(cfg.get("theta_max", 0.85))
        self.cooldown_ms = int(cfg.get("cooldown_ms", 3000))
        self.hysteresis = float(cfg.get("hysteresis", 0.05))
        self.min_awake_per_role = max(0, int(cfg.get("min_awake_per_role", 0)))
        self.full_load_mode = str(cfg.get("full_load_mode", "any")).lower()
        self.load_threshold_metric = str(cfg.get("load_threshold_metric", "kv_cache")).lower()
        valid_load_metrics = {"kv_cache", "running_request_num", "waiting_request_num", "total_request_num"}
        if self.load_threshold_metric not in valid_load_metrics:
            psrl_logger.warning(
                "Invalid load_threshold_metric=%s, fallback to kv_cache. Valid options: %s",
                self.load_threshold_metric,
                sorted(valid_load_metrics),
            )
            self.load_threshold_metric = "kv_cache"
        # Extra guard for Priority-3 spontaneous shrink:
        # even if KV cache is low, do not shrink when waiting queue is still high.
        # This is computed per-role as total waiting queues across awake instances.
        self.max_waiting_queue_for_scale_down = int(cfg.get("max_waiting_queue_for_scale_down", 0))

        profile_paths = cfg.get("profile_paths", {})
        if not isinstance(profile_paths, dict):
            profile_paths = {}
        throughput_model_dir = cfg.get("throughput_model_dir")
        preferred_output_len = int(cfg.get("throughput_model_output_len", 1024))
        self.profile_loader = ThroughputProfileLoader(
            profile_paths=profile_paths,
            throughput_model_dir=throughput_model_dir,
            preferred_output_len=preferred_output_len,
        )

        self.last_action_time_ms: float = 0.0
        self._last_total_queue = 0
        self._last_lambda_time = time.time()
        self._lambda_ewma = 0.0
        self._lambda_ewma_alpha = float(cfg.get("lambda_ewma_alpha", 0.2))
        # Per-tick decision trace (reasons when scale conditions are not met).
        self.log_scaling_decisions = bool(cfg.get("log_scaling_decisions", True))

        self.log_prefix = "ScalingPolicy"
        psrl_logger.propagate = False
        psrl_logger.addHandler(FileOnlyHandler(self.config.psrl.logging_path, self.log_prefix))

    def _policy_log(self, event: str, **kwargs: Any) -> None:
        if not self.log_scaling_decisions:
            return
        parts = [f"{k}={kwargs[k]!r}" for k in sorted(kwargs.keys())]
        psrl_logger.info("elastic_rm_policy %s | %s", event, " ".join(parts))

    def _policy_log_no_action(self, final_reason: str, diagnostics: list[str], detail: list[str]) -> None:
        if not self.log_scaling_decisions:
            return
        psrl_logger.info("elastic_rm_policy decision | outcome=no_action reason=%s", final_reason)
        if diagnostics:
            psrl_logger.info(
                "elastic_rm_policy no_action | diagnostics=%s",
                "|".join(sorted(diagnostics)),
            )
        for ln in detail:
            psrl_logger.info("elastic_rm_policy no_action_detail | %s", ln)

    @staticmethod
    def _is_snapshot_staled(snapshot: dict, max_staleness_seconds: float = 5.0) -> bool:
        ts = snapshot.get("timestamp")
        if not ts:
            return True
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            return True
        return (datetime.now() - dt).total_seconds() > max_staleness_seconds

    @staticmethod
    def _pre_sleep_other_if_colocated(
        wake: InstanceSignal, victim: InstanceSignal
    ) -> list[dict[str, Any]] | None:
        """If wake target shares GPUs with ``victim``, hint executor to pre-sleep ``victim``.

        When either side lacks ``bundle_keys``, returns None so ElasticExecutor keeps KV ordering.
        """
        if not wake.bundle_keys or not victim.bundle_keys:
            return None
        if not wake.bundle_keys.intersection(victim.bundle_keys):
            return None
        return [
            {
                "role_name": victim.role_name,
                "model_name": victim.model_name,
                "instance_id": int(victim.instance_id),
            }
        ]

    @staticmethod
    def _bundle_size(signal: InstanceSignal) -> int | None:
        if not signal.bundle_keys:
            return None
        return len(signal.bundle_keys)

    @staticmethod
    def _signal_entry(signal: InstanceSignal) -> dict[str, Any]:
        return {
            "role_name": signal.role_name,
            "model_name": signal.model_name,
            "instance_id": int(signal.instance_id),
        }

    @staticmethod
    def _awake_bundle_keys(signals: list[InstanceSignal], exclude: set[tuple[PSRL_Role, str, int]] | None = None) -> set[int]:
        excluded = exclude or set()
        out: set[int] = set()
        for signal in signals:
            key = (signal.role_name, signal.model_name, int(signal.instance_id))
            if not signal.is_awaken or key in excluded or not signal.bundle_keys:
                continue
            out.update(signal.bundle_keys)
        return out

    def _pick_migration_targets_for_conflicts(
        self,
        *,
        wake: InstanceSignal,
        target_role_signals: list[InstanceSignal],
        other_role_signals: list[InstanceSignal],
        conflicts: list[InstanceSignal],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
    ) -> list[InstanceSignal] | None:
        """Pick free smaller-role targets to wake before opening a larger target."""
        if not wake.bundle_keys or not conflicts:
            return None
        conflict_keys = {
            (s.role_name, s.model_name, int(s.instance_id))
            for s in conflicts
        }
        occupied = self._awake_bundle_keys(target_role_signals)
        occupied.update(self._awake_bundle_keys(other_role_signals, exclude=conflict_keys))
        picked: list[InstanceSignal] = []
        picked_bundle_keys: set[int] = set()

        for conflict in conflicts:
            candidates = [
                s
                for s in other_role_signals
                if (
                    not s.is_awaken
                    and s.model_name == conflict.model_name
                    and s.bundle_keys
                    and not s.bundle_keys.intersection(wake.bundle_keys)
                    and not s.bundle_keys.intersection(occupied)
                    and not s.bundle_keys.intersection(picked_bundle_keys)
                )
            ]
            if not candidates:
                return None
            candidates.sort(
                key=lambda s: instance_mu.get((s.role_name, s.model_name, s.instance_id), 0.0),
                reverse=True,
            )
            target = candidates[0]
            picked.append(target)
            picked_bundle_keys.update(target.bundle_keys or [])
        return picked

    def _build_scale_up_action_with_elastic_placement(
        self,
        *,
        wake: InstanceSignal,
        target_role_signals: list[InstanceSignal],
        other_role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
        reason: str,
    ) -> ScalingAction:
        preferred_ids = [int(wake.instance_id)]
        num_instances = 1
        pre_sleep: list[dict[str, Any]] | None = None
        pre_wake: list[dict[str, Any]] | None = None

        if wake.bundle_keys:
            conflicts = [
                s
                for s in other_role_signals
                if s.is_awaken and s.bundle_keys and s.bundle_keys.intersection(wake.bundle_keys)
            ]
            if conflicts:
                wake_size = self._bundle_size(wake)
                conflict_sizes = [self._bundle_size(s) for s in conflicts]
                max_conflict_size = max((s or 0) for s in conflict_sizes)

                if wake_size is not None and wake_size < max_conflict_size:
                    # Small-model expansion: release the colocated large instance and wake
                    # all asleep small instances whose bundles are contained in that large range.
                    large_victims = [
                        s
                        for s in conflicts
                        if self._bundle_size(s) == max_conflict_size and s.bundle_keys
                    ]
                    large_victims.sort(key=lambda s: (s.kv_cache_utilization, s.instance_id))
                    victim = large_victims[0]
                    pre_sleep = [self._signal_entry(victim)]
                    contained = [
                        s
                        for s in target_role_signals
                        if (
                            not s.is_awaken
                            and s.bundle_keys
                            and s.bundle_keys.issubset(victim.bundle_keys or frozenset())
                        )
                    ]
                    contained.sort(key=lambda s: min(s.bundle_keys or frozenset({10**9})))
                    preferred_ids = [int(s.instance_id) for s in contained] or preferred_ids
                    num_instances = max(1, len(preferred_ids))
                else:
                    # Large-model expansion: if possible, migrate every overlapping smaller
                    # instance to an already free placement before opening the large one.
                    conflicts.sort(key=lambda s: (s.kv_cache_utilization, s.instance_id))
                    migration_targets = self._pick_migration_targets_for_conflicts(
                        wake=wake,
                        target_role_signals=target_role_signals,
                        other_role_signals=other_role_signals,
                        conflicts=conflicts,
                        instance_mu=instance_mu,
                    )
                    if migration_targets is not None:
                        pre_wake = [self._signal_entry(s) for s in migration_targets]
                    pre_sleep = [self._signal_entry(s) for s in conflicts]

        return ScalingAction(
            action_type="scale_up",
            role_name=wake.role_name,
            model_name=wake.model_name,
            num_instances=num_instances,
            preferred_instance_ids=preferred_ids,
            reason=reason,
            pre_sleep_other_preferred=pre_sleep,
            pre_wake_other_preferred=pre_wake,
        )

    @staticmethod
    def _group_by_role(signals: list[InstanceSignal]) -> dict[PSRL_Role, list[InstanceSignal]]:
        """
        Group the signals by role.
        Args:
            signals: list[InstanceSignal]
        Returns:
            dict[PSRL_Role, list[InstanceSignal]]:
                - grouped: {role_name: [signal]}
        """
        grouped: dict[PSRL_Role, list[InstanceSignal]] = {}
        for signal in signals:
            grouped.setdefault(signal.role_name, []).append(signal)
        return grouped

    def _estimate_lambda(self, signals: list[InstanceSignal], role_to_total_mu: dict[PSRL_Role, float]) -> float:
        """
        Estimate the lambda for the given signals and role_to_total_mu.
        Args:
            signals: list[InstanceSignal]
            role_to_total_mu: dict[PSRL_Role, float]
        Returns:
            float: the estimated lambda
        """
        now = time.time()
        elapsed = max(now - self._last_lambda_time, 1e-6)
        current_total_queue = sum(s.running_queue_num + s.waiting_queue_num for s in signals if s.is_awaken)
        d_queue = current_total_queue - self._last_total_queue
        total_mu = sum(role_to_total_mu.values())
        raw_lambda = max(total_mu + d_queue / elapsed, 0.0)
        self._lambda_ewma = self._lambda_ewma_alpha * raw_lambda + (1 - self._lambda_ewma_alpha) * self._lambda_ewma
        self._last_total_queue = current_total_queue
        self._last_lambda_time = now
        return self._lambda_ewma

    def _build_mu_maps(
        self, signals: list[InstanceSignal]
    ) -> tuple[
        dict[tuple[PSRL_Role, str, int], float],
        dict[PSRL_Role, float],
        dict[tuple[PSRL_Role, str, int], str],
    ]:
        """
        Build the mu maps for the given signals.
        Args:
            signals: list[InstanceSignal]
        Returns:
            tuple[dict[tuple[PSRL_Role, str, int], float], dict[PSRL_Role, float]]:
                - instance_mu: {(role, model_name, instance_id): mu}
                - role_total_mu: {role: total_mu}
        """
        instance_mu: dict[tuple[PSRL_Role, str, int], float] = {}
        instance_mu_source: dict[tuple[PSRL_Role, str, int], str] = {}
        role_total_mu: dict[PSRL_Role, float] = {}
        for signal in signals:
            mu, mu_source = self.profile_loader.estimate_instance_mu_with_source(signal)
            role = signal.role_name
            key = (role, signal.model_name, signal.instance_id)
            instance_mu[key] = mu
            instance_mu_source[key] = mu_source
            if signal.is_awaken:
                role_total_mu[role] = role_total_mu.get(role, 0.0) + mu
        return instance_mu, role_total_mu, instance_mu_source

    def _estimate_role_total_mu_with_rebalance(
        self,
        role_signals: list[InstanceSignal],
        scale_up_signal: InstanceSignal | None = None,
        scale_down_signal: InstanceSignal | None = None,
    ) -> float:
        # This helper simulates "after-action" throughput for one role.
        # We use it to compare candidate actions before issuing real SLEEP/WAKE_UP.
        awaken_signals = [s for s in role_signals if s.is_awaken]
        if not awaken_signals:
            return 0.0

        if scale_up_signal is not None:
            # Expansion: average (running + waiting) across all active instances.
            total_queue = sum(float(s.running_queue_num + s.waiting_queue_num) for s in awaken_signals)
            new_active = awaken_signals + [scale_up_signal]
            avg_running = total_queue / max(len(new_active), 1)
            avg_awake_throughput = sum(float(s.generation_throughput) for s in awaken_signals) / max(
                len(awaken_signals), 1
            )
            return sum(
                self.profile_loader.estimate_mu_by_running_queue(
                    model_name=s.model_name,
                    running_queue_num=avg_running,
                    # For newly added asleep instance, use current role-average throughput
                    # as fallback instead of stale 0.0 snapshot throughput.
                    fallback_mu=(
                        avg_awake_throughput
                        if (not s.is_awaken and s.instance_id == scale_up_signal.instance_id)
                        else s.generation_throughput
                    ),
                )
                or 0.0
                for s in new_active
            )

        if scale_down_signal is not None:
            # Shrink: redistribute removed running queue to retained active instances.
            retained = [s for s in awaken_signals if s.instance_id != scale_down_signal.instance_id]
            if len(retained) < self.min_awake_per_role:
                return -1.0
            extra_running = float(scale_down_signal.running_queue_num) / max(len(retained), 1)
            return sum(
                self.profile_loader.estimate_mu_by_running_queue(
                    model_name=s.model_name,
                    running_queue_num=float(s.running_queue_num) + extra_running,
                    fallback_mu=s.generation_throughput,
                )
                or 0.0
                for s in retained
            )

        return sum(
            self.profile_loader.estimate_mu_by_running_queue(
                model_name=s.model_name,
                running_queue_num=float(s.running_queue_num),
                fallback_mu=s.generation_throughput,
            )
            or 0.0
            for s in awaken_signals
        )

    def _pick_scale_down_candidate(
        self,
        role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
    ) -> InstanceSignal | None:
        """Spontaneous shrink: only cede when selected load metric is low (below ``theta_low``).

        Used when a role has spare capacity (Priority 1/3), not for forced rebalance
        under mutual full load — see ``_pick_scale_down_candidate_for_bottleneck_transfer``.
        """
        awaken = [s for s in role_signals if s.is_awaken]
        if len(awaken) <= self.min_awake_per_role:
            return None
        cede_candidates = [s for s in awaken if self._get_signal_load_value(s) <= self.theta_low]
        if not cede_candidates:
            return None
        cede_candidates.sort(
            key=lambda s: (
                self._get_signal_load_value(s),
                instance_mu.get((s.role_name, s.model_name, s.instance_id), 0.0),
            )
        )
        return cede_candidates[0]

    def _pick_scale_down_candidate_for_bottleneck_transfer(
        self,
        role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
    ) -> InstanceSignal | None:
        """Pick which awake instance to sleep when *both* roles are full-load (Priority 2).

        Does **not** require ``kv_cache_utilization <= theta_low``: mutual full load is exactly
        when we may need to sacrifice a replica to wake the other role. We pick the instance
        with the smallest estimated per-instance throughput (mu) first, then lower KV as tie-break.
        """
        awaken = [s for s in role_signals if s.is_awaken]
        if len(awaken) <= self.min_awake_per_role:
            return None
        awaken_sorted = sorted(
            awaken,
            key=lambda s: (
                instance_mu.get((s.role_name, s.model_name, s.instance_id), 0.0),
                s.kv_cache_utilization,
            ),
        )
        return awaken_sorted[0]

    def _pick_scale_up_candidate(
        self,
        role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
    ) -> InstanceSignal | None:
        asleep = [s for s in role_signals if not s.is_awaken]
        if not asleep:
            return None
        asleep.sort(
            key=lambda s: instance_mu.get((s.role_name, s.model_name, s.instance_id), 0.0),
            reverse=True,
        )
        return asleep[0]

    def _pick_scale_up_candidate_by_free_devices(
        self,
        role_signals: list[InstanceSignal],
        other_role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
    ) -> InstanceSignal | None:
        """Pick an asleep target, preferring placements with more currently free devices."""
        asleep = [s for s in role_signals if not s.is_awaken]
        if not asleep:
            return None
        other_awake_bundle_keys = self._awake_bundle_keys(other_role_signals)

        def _free_bundle_count(signal: InstanceSignal) -> int:
            if not signal.bundle_keys:
                return -1
            return len(signal.bundle_keys.difference(other_awake_bundle_keys))

        asleep.sort(
            key=lambda s: (
                _free_bundle_count(s),
                instance_mu.get((s.role_name, s.model_name, s.instance_id), 0.0),
            ),
            reverse=True,
        )
        return asleep[0]

    def _pick_scale_up_candidate_for_victim(
        self,
        role_signals: list[InstanceSignal],
        victim: InstanceSignal,
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
    ) -> InstanceSignal | None:
        """Pick an asleep target whose placement corresponds to the policy-selected victim."""
        if not victim.bundle_keys:
            return None
        candidates = [
            s
            for s in role_signals
            if (
                not s.is_awaken
                and s.bundle_keys
                and s.bundle_keys.intersection(victim.bundle_keys)
            )
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda s: instance_mu.get((s.role_name, s.model_name, s.instance_id), 0.0),
            reverse=True,
        )
        return candidates[0]

    def _pick_scale_up_candidate_by_force(
        self,
        role_signals: list[InstanceSignal],
        other_role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
    ) -> tuple[InstanceSignal, InstanceSignal | None] | None:
        """Pick (wake_target, optional other_role_sleep_victim) for router force_wake.

        Prefer a GPU where the other role has no awake instance (reuse free-GPU logic);
        then ``other_role_sleep_victim`` is None and the executor keeps its KV-based pick.

        Else walk other-role awake replicas in ascending ``kv_cache_utilization`` and pick
        the highest-``mu`` asleep backlog instance that shares ``bundle_keys`` with that victim;
        return that victim as ``other_role_sleep_victim`` so the executor pre-sleeps the
        same instance the policy assumed.

        If no GPU mapping / no colocation, fall back to global highest-``mu`` asleep
        candidate with victim None.
        """
        asleep = [s for s in role_signals if not s.is_awaken]
        if not asleep:
            return None

        free = self._pick_scale_up_candidate_on_free_gpu(
            role_signals, other_role_signals, instance_mu
        )
        if free is not None:
            return (free, None)

        other_awaken = [s for s in other_role_signals if s.is_awaken]
        if len(other_awaken) <= self.min_awake_per_role:
            wake = self._pick_scale_up_candidate(role_signals, instance_mu)
            return (wake, None) if wake is not None else None

        victims_sorted = sorted(
            other_awaken,
            key=lambda s: (s.kv_cache_utilization, s.instance_id),
        )

        def _mu(s: InstanceSignal) -> float:
            return instance_mu.get(
                (s.role_name, s.model_name, s.instance_id), 0.0
            )

        for victim in victims_sorted:
            if not victim.bundle_keys:
                continue
            sharing = [
                s
                for s in asleep
                if s.bundle_keys and s.bundle_keys.intersection(victim.bundle_keys)
            ]
            if not sharing:
                continue
            sharing.sort(key=_mu, reverse=True)
            return (sharing[0], victim)

        wake_fb = self._pick_scale_up_candidate(role_signals, instance_mu)
        return (wake_fb, None) if wake_fb is not None else None

    def _pick_scale_up_candidate_on_free_gpu(
        self,
        full_role_signals: list[InstanceSignal],
        other_role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
    ) -> InstanceSignal | None:
        """Among asleep instances of the full-load role, return the best one whose
        GPUs are completely free of any awake instance from the other role.

        "Free" means: the candidate's bundle_keys have no intersection with the bundle_keys
        of any currently awake other-role instance.  Candidates with unknown bundle
        mapping (bundle_keys is None or empty) are skipped — we can't guarantee they are
        free, so we won't take the risk of double-occupancy.

        Returns the highest-mu free candidate, or None if no such instance exists.
        """
        other_awake_bundle_keys: set[int] = set()
        for s in other_role_signals:
            if s.is_awaken and s.bundle_keys:
                other_awake_bundle_keys.update(s.bundle_keys)

        free_candidates: list[InstanceSignal] = []
        for s in full_role_signals:
            if s.is_awaken:
                continue
            if not s.bundle_keys:
                continue
            if not s.bundle_keys.intersection(other_awake_bundle_keys):
                free_candidates.append(s)

        if not free_candidates:
            return None
        free_candidates.sort(
            key=lambda s: instance_mu.get(
                (s.role_name, s.model_name, s.instance_id), 0.0
            ),
            reverse=True,
        )
        return free_candidates[0]

    def _pick_scale_up_candidates_on_free_gpu(
        self,
        full_role_signals: list[InstanceSignal],
        other_role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
    ) -> list[InstanceSignal]:
        """Return all free-GPU asleep candidates sorted by descending mu."""
        other_awake_bundle_keys: set[int] = set()
        for s in other_role_signals:
            if s.is_awaken and s.bundle_keys:
                other_awake_bundle_keys.update(s.bundle_keys)

        free_candidates: list[InstanceSignal] = []
        for s in full_role_signals:
            if s.is_awaken:
                continue
            if not s.bundle_keys:
                continue
            if not s.bundle_keys.intersection(other_awake_bundle_keys):
                free_candidates.append(s)

        free_candidates.sort(
            key=lambda s: instance_mu.get(
                (s.role_name, s.model_name, s.instance_id), 0.0
            ),
            reverse=True,
        )
        return free_candidates

    @staticmethod
    def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _resolve_role_request_cap(self, role_name: PSRL_Role, model_name: str) -> int:
        """Resolve per-instance request cap for backlog-based batch scale-up."""
        if role_name == PSRL_Role.Rollout:
            routing_cfg = self._cfg_get(self._cfg_get(self.config, "psrl"), "routing_strategy")
            rollout_cap = self._cfg_get(routing_cfg, "max_concurrent_seqs_per_instance")
            try:
                rollout_cap_i = int(rollout_cap)
                if rollout_cap_i > 0:
                    return rollout_cap_i
            except (TypeError, ValueError):
                pass
            return 1

        if role_name == PSRL_Role.RewardModel:
            reward_cfg = self._cfg_get(self.config, "reward_models_config")
            reward_models = self._cfg_get(reward_cfg, "reward_models", [])
            if reward_models:
                for item in reward_models:
                    item_model_name = self._cfg_get(item, "reward_model_name")
                    if item_model_name != model_name:
                        continue
                    cap = self._cfg_get(item, "max_concurrent_requests_per_instance")
                    try:
                        cap_i = int(cap)
                        if cap_i > 0:
                            return cap_i
                    except (TypeError, ValueError):
                        continue
            return 1

        return 1

    def _build_backlog_batch_scale_up_action(
        self,
        *,
        role_name: PSRL_Role,
        model_name: str,
        backlog_cnt: int,
        free_candidates: list[InstanceSignal],
        reason: str,
    ) -> ScalingAction:
        """Build free-resource scale-up action with batch size from backlog/cap."""
        if not free_candidates:
            raise ValueError("free_candidates must be non-empty")

        same_model_candidates = [s for s in free_candidates if s.model_name == model_name]
        if not same_model_candidates:
            same_model_candidates = free_candidates
        cap_per_instance = self._resolve_role_request_cap(role_name, model_name)
        required_instances = max(1, math.ceil(max(0, int(backlog_cnt)) / max(cap_per_instance, 1)))
        num_instances = min(len(same_model_candidates), required_instances)
        preferred_ids = [int(s.instance_id) for s in same_model_candidates[:num_instances]]
        return ScalingAction(
            action_type="scale_up",
            role_name=role_name,
            model_name=model_name,
            num_instances=num_instances,
            preferred_instance_ids=preferred_ids,
            reason=reason,
        )

    def _get_signal_load_value(self, signal: InstanceSignal) -> float:
        if self.load_threshold_metric == "running_request_num":
            return float(signal.running_queue_num)
        if self.load_threshold_metric == "waiting_request_num":
            return float(signal.waiting_queue_num)
        if self.load_threshold_metric == "total_request_num":
            return float(signal.running_queue_num + signal.waiting_queue_num)
        # default: kv_cache
        return float(signal.kv_cache_utilization)

    def _role_full_load(self, role_signals: list[InstanceSignal], theta_max: float, mode: str = "all") -> bool:
        awaken = [s for s in role_signals if s.is_awaken]
        if not awaken:
            return False
        if mode == "any":
            return any(self._get_signal_load_value(s) >= theta_max for s in awaken)
        return all(self._get_signal_load_value(s) >= theta_max for s in awaken)

    def _role_has_low_load(self, role_signals: list[InstanceSignal], theta_low: float) -> bool:
        awaken = [s for s in role_signals if s.is_awaken]
        if not awaken:
            return False
        return any(self._get_signal_load_value(s) <= theta_low for s in awaken)

    def _no_action_detail_strings(
        self,
        *,
        trainer_busy: bool,
        pending_total: int,
        waiting_on: str,
        rollout_full: bool,
        rm_full: bool,
        rollout_low: bool,
        rm_low: bool,
        rollout_down_waiting: int,
        rm_down_waiting: int,
        rollout_up: InstanceSignal | None,
        rm_up: InstanceSignal | None,
        rollout_down: InstanceSignal | None,
        rm_down: InstanceSignal | None,
        rollout_free_up: InstanceSignal | None,
        rm_free_up: InstanceSignal | None,
        rollout_down_xfer: InstanceSignal | None,
        rm_down_xfer: InstanceSignal | None,
        p2_gain_rejected: float | None,
        p2_branch_notes: list[str],
    ) -> list[str]:
        """Human-readable reasons why lower-priority branches did not scale (for logs)."""
        out: list[str] = []
        out.append(
            f"p-1_skip trainer_busy={trainer_busy} pending_total={pending_total} waiting_on={waiting_on!r} "
            f"(needs idle trainer + pending>0 + waiting_on rollout|reward)"
        )
        # Priority 1: RM-only-full branch
        if rm_full and not rollout_full:
            out.append(
                f"p1_rm_full_only rm_free_up={rm_free_up is not None} "
                f"rm_up={rm_up is not None} rollout_down_theta_low={rollout_down is not None} "
                f"(free_gpu tried first, fallback needs rm_up+rollout_down)"
            )
        # Priority 1: Rollout-only-full branch
        if rollout_full and not rm_full:
            out.append(
                f"p1_rollout_full_only rollout_free_up={rollout_free_up is not None} "
                f"rollout_up={rollout_up is not None} rm_down_theta_low={rm_down is not None} "
                f"(free_gpu tried first, fallback needs rollout_up+rm_down)"
            )
        if not rm_full and not rollout_full:
            out.append("p1_skip neither_side_full")
        # Priority 2
        out.append(
            f"p2_context both_full={rollout_full and rm_full} "
            f"rollout_down_xfer={rollout_down_xfer is not None} rm_down_xfer={rm_down_xfer is not None} "
            f"rm_up={rm_up is not None} rollout_up={rollout_up is not None}"
        )
        out.extend(p2_branch_notes)
        if p2_gain_rejected is not None:
            out.append(f"p2_reject best_gain={p2_gain_rejected:.6f} hysteresis={self.hysteresis}")
        out.append(
            f"p3_rollout_self_down_unmet rollout_low={rollout_low} "
            f"rollout_down_theta_low={rollout_down is not None} rm_full={rm_full} "
            f"down_waiting={rollout_down_waiting} waiting_max={self.max_waiting_queue_for_scale_down}"
        )
        out.append(
            f"p3_rm_self_down_unmet rm_low={rm_low} rm_down_theta_low={rm_down is not None} "
            f"rollout_full={rollout_full} down_waiting={rm_down_waiting} "
            f"waiting_max={self.max_waiting_queue_for_scale_down}"
        )
        return out

    def _make_stepwise_decision(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
        role_total_mu: dict[PSRL_Role, float],
        router_backlog_by_role: dict[PSRL_Role, int] | None = None,
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> tuple[list[ScalingAction], str]:
        rollout_role = PSRL_Role.Rollout
        rm_role = PSRL_Role.RewardModel
        rollout_signals = grouped.get(rollout_role, [])
        rm_signals = grouped.get(rm_role, [])
        if not rollout_signals or not rm_signals:
            self._policy_log(
                "decision",
                outcome="no_action",
                reason="skip_decision_missing_rollout_or_rm_signals",
                has_rm_signals=bool(rm_signals),
                has_rollout_signals=bool(rollout_signals),
            )
            return [], "skip_decision_missing_rollout_or_rm_signals"

        rollout_full = self._role_full_load(rollout_signals, self.theta_max, mode=self.full_load_mode)
        rm_full = self._role_full_load(rm_signals, self.theta_max, mode=self.full_load_mode)
        rollout_low = self._role_has_low_load(rollout_signals, self.theta_low)
        rm_low = self._role_has_low_load(rm_signals, self.theta_low)

        rollout_up = self._pick_scale_up_candidate_by_free_devices(
            rollout_signals, rm_signals, instance_mu
        )
        rm_up = self._pick_scale_up_candidate_by_free_devices(
            rm_signals, rollout_signals, instance_mu
        )
        rollout_free_up = self._pick_scale_up_candidate_on_free_gpu(
            rollout_signals, rm_signals, instance_mu
        )
        rm_free_up = self._pick_scale_up_candidate_on_free_gpu(
            rm_signals, rollout_signals, instance_mu
        )
        rollout_free_candidates = self._pick_scale_up_candidates_on_free_gpu(
            rollout_signals, rm_signals, instance_mu
        )
        rm_free_candidates = self._pick_scale_up_candidates_on_free_gpu(
            rm_signals, rollout_signals, instance_mu
        )
        rollout_down = self._pick_scale_down_candidate(rollout_signals, instance_mu)
        rm_down = self._pick_scale_down_candidate(rm_signals, instance_mu)

        actions: list[ScalingAction] = []
        diagnostics: list[str] = []
        rollout_down_waiting = rollout_down.waiting_queue_num if rollout_down is not None else -1
        rm_down_waiting = rm_down.waiting_queue_num if rm_down is not None else -1
        n_rollout_awaken = sum(1 for s in rollout_signals if s.is_awaken)
        n_rollout_asleep = sum(1 for s in rollout_signals if not s.is_awaken)
        n_rm_awaken = sum(1 for s in rm_signals if s.is_awaken)
        n_rm_asleep = sum(1 for s in rm_signals if not s.is_awaken)
        self._policy_log(
            "tick_context",
            full_load_mode=self.full_load_mode,
            load_threshold_metric=self.load_threshold_metric,
            n_rm_asleep=n_rm_asleep,
            n_rm_awaken=n_rm_awaken,
            n_rollout_asleep=n_rollout_asleep,
            n_rollout_awaken=n_rollout_awaken,
            rm_full=rm_full,
            rm_low=rm_low,
            rollout_full=rollout_full,
            rollout_low=rollout_low,
            theta_low=self.theta_low,
            theta_max=self.theta_max,
        )
        self._policy_log(
            "tick_context_waiting_guard",
            max_waiting_queue_for_scale_down=self.max_waiting_queue_for_scale_down,
            rm_down_waiting=rm_down_waiting,
            rollout_down_waiting=rollout_down_waiting,
        )

        # Priority -1: keep trainer continuously training.
        # When trainer is idle and blocked by current batch, directly bias scale-up
        # towards the bottleneck stage (rollout/reward).
        hint = trainer_waiting_hint or {}
        trainer_busy = bool(hint.get("trainer_busy", True))
        waiting_on = str(hint.get("waiting_on", "none")).lower()
        pending_total = int((hint.get("breakdown") or {}).get("pending_total", 0))
        if not trainer_busy and pending_total > 0 and waiting_on in {"rollout", "reward"}:
            if waiting_on == "rollout":
                if rollout_up is None:
                    self._policy_log(
                        "decision",
                        outcome="no_action",
                        reason="trainer_idle_waiting_rollout_but_no_scaleup_candidate",
                        pending_total=pending_total,
                        waiting_on=waiting_on,
                    )
                    return [], "trainer_idle_waiting_rollout_but_no_scaleup_candidate"
                actions.append(
                    self._build_scale_up_action_with_elastic_placement(
                        wake=rollout_up,
                        target_role_signals=rollout_signals,
                        other_role_signals=rm_signals,
                        instance_mu=instance_mu,
                        reason="trainer_idle_waiting_rollout_scale_up",
                    )
                )
                self._policy_log(
                    "decision",
                    action="scale_up",
                    instance_id=rollout_up.instance_id,
                    model_name=rollout_up.model_name,
                    outcome="action",
                    policy_branch="p-1_trainer_idle_rollout",
                    reason="trainer_idle_waiting_rollout",
                )
                return actions, "trainer_idle_waiting_rollout"
            if rm_up is None:
                self._policy_log(
                    "decision",
                    outcome="no_action",
                    reason="trainer_idle_waiting_reward_but_no_scaleup_candidate",
                    pending_total=pending_total,
                    waiting_on=waiting_on,
                )
                return [], "trainer_idle_waiting_reward_but_no_scaleup_candidate"
            actions.append(
                self._build_scale_up_action_with_elastic_placement(
                    wake=rm_up,
                    target_role_signals=rm_signals,
                    other_role_signals=rollout_signals,
                    instance_mu=instance_mu,
                    reason="trainer_idle_waiting_reward_scale_up",
                )
            )
            self._policy_log(
                "decision",
                action="scale_up",
                instance_id=rm_up.instance_id,
                model_name=rm_up.model_name,
                outcome="action",
                policy_branch="p-1_trainer_idle_reward",
                reason="trainer_idle_waiting_reward",
            )
            return actions, "trainer_idle_waiting_reward"

        # ── Priority 1 (new) ─────────────────────────────────────────────────────
        # If there are free devices and at least one side can wake without ceding
        # the other side, prefer the side with larger router backlog.
        # ──────────────────────────────────────────────────────────────────────────
        backlog_map = router_backlog_by_role or {}
        rollout_backlog = max(0, int(backlog_map.get(rollout_role, 0)))
        rm_backlog = max(0, int(backlog_map.get(rm_role, 0)))

        if rollout_free_up is not None and rm_free_up is not None and (rollout_backlog > 0 or rm_backlog > 0):
            pick_rollout = rollout_backlog >= rm_backlog
            chosen = rollout_free_up if pick_rollout else rm_free_up
            chosen_role = rollout_role if pick_rollout else rm_role
            chosen_backlog = rollout_backlog if pick_rollout else rm_backlog
            chosen_free_candidates = rollout_free_candidates if pick_rollout else rm_free_candidates
            actions.append(
                self._build_backlog_batch_scale_up_action(
                    role_name=chosen_role,
                    model_name=chosen.model_name,
                    backlog_cnt=chosen_backlog,
                    free_candidates=chosen_free_candidates,
                    reason="free_resource_backlog_priority_scale_up",
                )
            )
            self._policy_log(
                "decision",
                action="scale_up",
                instance_id=chosen.instance_id,
                model_name=chosen.model_name,
                num_instances=actions[-1].num_instances,
                rollout_backlog=rollout_backlog,
                rm_backlog=rm_backlog,
                outcome="action",
                policy_branch="p1_free_resource_backlog_priority",
                reason=f"free_resource_backlog_priority_{chosen_role.name}",
            )
            return actions, f"free_resource_backlog_priority_{chosen_role.name}"

        if rollout_free_up is not None and rm_free_up is None and rollout_backlog > 0 and rollout_backlog >= rm_backlog:
            actions.append(
                self._build_backlog_batch_scale_up_action(
                    role_name=rollout_role,
                    model_name=rollout_free_up.model_name,
                    backlog_cnt=rollout_backlog,
                    free_candidates=rollout_free_candidates,
                    reason="free_resource_backlog_priority_scale_up_rollout",
                )
            )
            self._policy_log(
                "decision",
                action="scale_up",
                instance_id=rollout_free_up.instance_id,
                model_name=rollout_free_up.model_name,
                num_instances=actions[-1].num_instances,
                rollout_backlog=rollout_backlog,
                rm_backlog=rm_backlog,
                outcome="action",
                policy_branch="p1_free_resource_backlog_priority",
                reason="free_resource_backlog_priority_rollout",
            )
            return actions, "free_resource_backlog_priority_rollout"

        if rm_free_up is not None and rollout_free_up is None and rm_backlog > 0 and rm_backlog >= rollout_backlog:
            actions.append(
                self._build_backlog_batch_scale_up_action(
                    role_name=rm_role,
                    model_name=rm_free_up.model_name,
                    backlog_cnt=rm_backlog,
                    free_candidates=rm_free_candidates,
                    reason="free_resource_backlog_priority_scale_up_rm",
                )
            )
            self._policy_log(
                "decision",
                action="scale_up",
                instance_id=rm_free_up.instance_id,
                model_name=rm_free_up.model_name,
                num_instances=actions[-1].num_instances,
                rollout_backlog=rollout_backlog,
                rm_backlog=rm_backlog,
                outcome="action",
                policy_branch="p1_free_resource_backlog_priority",
                reason="free_resource_backlog_priority_rm",
            )
            return actions, "free_resource_backlog_priority_rm"

        # ── Priority 2  ──────────────────────────────────────────────────────────
        # Applies when EXACTLY one side is full.
        # Step A: try to wake an instance of the full side on a GPU that is
        #         completely idle for the other side (no scale_down needed).
        # Step B: fall back to ceding a low-load instance from the other side.
        # Both-full case is handled by Priority 3 below.
        # ─────────────────────────────────────────────────────────────────────────

        # P1a: only RM is full
        if rm_full and not rollout_full:
            # Step A: free-GPU scale-up — no resource taken from Rollout
            if rm_free_up is not None:
                actions.append(
                    self._build_scale_up_action_with_elastic_placement(
                        wake=rm_free_up,
                        target_role_signals=rm_signals,
                        other_role_signals=rollout_signals,
                        instance_mu=instance_mu,
                        reason="rm_full_free_gpu_scale_up",
                    )
                )
                self._policy_log(
                    "decision",
                    action="scale_up",
                    free_gpu=True,
                    instance_id=rm_free_up.instance_id,
                    model_name=rm_free_up.model_name,
                    outcome="action",
                    policy_branch="p1_rm_full_free_gpu",
                    reason="rm_full_free_gpu_scale_up",
                )
                return actions, "rm_full_free_gpu_scale_up"
            # Step B: fall back — cede a low-load Rollout instance
            if rm_up is not None and rollout_down is not None:
                rm_transfer_up = self._pick_scale_up_candidate_for_victim(
                    rm_signals, rollout_down, instance_mu
                ) or rm_up
                actions.append(
                    self._build_scale_up_action_with_elastic_placement(
                        wake=rm_transfer_up,
                        target_role_signals=rm_signals,
                        other_role_signals=rollout_signals,
                        instance_mu=instance_mu,
                        reason="rm_full_rollout_cede_transfer",
                    )
                )
                self._policy_log(
                    "decision",
                    action="scale_up",
                    free_gpu=False,
                    instance_id=rm_transfer_up.instance_id,
                    model_name=rm_transfer_up.model_name,
                    outcome="action",
                    policy_branch="p1_transfer_rollout_to_rm",
                    reason="transfer_rollout_to_rm",
                    pre_sleep_other=actions[-1].pre_sleep_other_preferred,
                )
                return actions, "transfer_rollout_to_rm"

        # P1b: only Rollout is full
        if rollout_full and not rm_full:
            # Step A: free-GPU scale-up — no resource taken from RM
            if rollout_free_up is not None:
                actions.append(
                    self._build_scale_up_action_with_elastic_placement(
                        wake=rollout_free_up,
                        target_role_signals=rollout_signals,
                        other_role_signals=rm_signals,
                        instance_mu=instance_mu,
                        reason="rollout_full_free_gpu_scale_up",
                    )
                )
                self._policy_log(
                    "decision",
                    action="scale_up",
                    free_gpu=True,
                    instance_id=rollout_free_up.instance_id,
                    model_name=rollout_free_up.model_name,
                    outcome="action",
                    policy_branch="p1_rollout_full_free_gpu",
                    reason="rollout_full_free_gpu_scale_up",
                )
                return actions, "rollout_full_free_gpu_scale_up"
            # Step B: fall back — cede a low-load RM instance
            if rollout_up is not None and rm_down is not None:
                rollout_transfer_up = self._pick_scale_up_candidate_for_victim(
                    rollout_signals, rm_down, instance_mu
                ) or rollout_up
                actions.append(
                    self._build_scale_up_action_with_elastic_placement(
                        wake=rollout_transfer_up,
                        target_role_signals=rollout_signals,
                        other_role_signals=rm_signals,
                        instance_mu=instance_mu,
                        reason="rollout_full_rm_cede_transfer",
                    )
                )
                self._policy_log(
                    "decision",
                    action="scale_up",
                    free_gpu=False,
                    instance_id=rollout_transfer_up.instance_id,
                    model_name=rollout_transfer_up.model_name,
                    outcome="action",
                    policy_branch="p1_transfer_rm_to_rollout",
                    reason="transfer_rm_to_rollout",
                    pre_sleep_other=actions[-1].pre_sleep_other_preferred,
                )
                return actions, "transfer_rm_to_rollout"

        # ── Priority 3 ────────────────────────────────────────────────────────────
        # Both sides full → optimize bottleneck throughput by one-step transfer.
        # We compare candidate gains with queue-rebalance simulation instead of naive
        # add/subtract current mu, because queue pressure changes after scaling.
        # Sleep-side candidates ignore theta_low: full KV on both sides is normal
        # here; theta_low remains the gate for *spontaneous* shrink in Priority 4.
        # ─────────────────────────────────────────────────────────────────────────
        bottleneck_before = min(
            self._estimate_role_total_mu_with_rebalance(rollout_signals),
            self._estimate_role_total_mu_with_rebalance(rm_signals),
        )
        transfer_candidates: list[tuple[float, ScalingAction, str]] = []
        rollout_down_xfer: InstanceSignal | None = None
        rm_down_xfer: InstanceSignal | None = None
        p2_branch_notes: list[str] = []
        p2_gain_rejected: float | None = None
        if rollout_full and rm_full:
            rollout_down_xfer = self._pick_scale_down_candidate_for_bottleneck_transfer(
                rollout_signals, instance_mu
            )
            rm_down_xfer = self._pick_scale_down_candidate_for_bottleneck_transfer(rm_signals, instance_mu)

        rm_up_xfer = (
            self._pick_scale_up_candidate_for_victim(rm_signals, rollout_down_xfer, instance_mu)
            if rollout_down_xfer is not None
            else None
        ) or rm_up
        if rollout_down_xfer is not None and rm_up_xfer is not None:
            rollout_mu = self._estimate_role_total_mu_with_rebalance(
                rollout_signals,
                scale_down_signal=rollout_down_xfer,
            )
            rm_mu = self._estimate_role_total_mu_with_rebalance(
                rm_signals,
                scale_up_signal=rm_up_xfer,
            )
            if rollout_mu >= 0 and rm_mu >= 0:
                bottleneck_after = min(rollout_mu, rm_mu)
                action = self._build_scale_up_action_with_elastic_placement(
                    wake=rm_up_xfer,
                    target_role_signals=rm_signals,
                    other_role_signals=rollout_signals,
                    instance_mu=instance_mu,
                    reason="optimize_bottleneck_rollout_to_rm",
                )
                transfer_candidates.append(
                    (
                        bottleneck_after - bottleneck_before,
                        action,
                        "optimize_rollout_to_rm",
                    )
                )
            else:
                p2_branch_notes.append(
                    f"p2_opt_rollout_to_rm_sim_invalid rollout_mu={rollout_mu} rm_mu={rm_mu}"
                )
        elif rollout_full and rm_full:
            if rollout_down_xfer is None:
                p2_branch_notes.append("p2_opt_rollout_to_rm_skip_no_rollout_down_xfer")
            if rm_up is None:
                p2_branch_notes.append("p2_opt_rollout_to_rm_skip_no_rm_up")

        rollout_up_xfer = (
            self._pick_scale_up_candidate_for_victim(rollout_signals, rm_down_xfer, instance_mu)
            if rm_down_xfer is not None
            else None
        ) or rollout_up
        if rm_down_xfer is not None and rollout_up_xfer is not None:
            rollout_mu = self._estimate_role_total_mu_with_rebalance(
                rollout_signals,
                scale_up_signal=rollout_up_xfer,
            )
            rm_mu = self._estimate_role_total_mu_with_rebalance(
                rm_signals,
                scale_down_signal=rm_down_xfer,
            )
            if rollout_mu >= 0 and rm_mu >= 0:
                bottleneck_after = min(rollout_mu, rm_mu)
                action = self._build_scale_up_action_with_elastic_placement(
                    wake=rollout_up_xfer,
                    target_role_signals=rollout_signals,
                    other_role_signals=rm_signals,
                    instance_mu=instance_mu,
                    reason="optimize_bottleneck_rm_to_rollout",
                )
                transfer_candidates.append(
                    (
                        bottleneck_after - bottleneck_before,
                        action,
                        "optimize_rm_to_rollout",
                    )
                )
            else:
                p2_branch_notes.append(
                    f"p2_opt_rm_to_rollout_sim_invalid rollout_mu={rollout_mu} rm_mu={rm_mu}"
                )
        elif rollout_full and rm_full:
            if rm_down_xfer is None:
                p2_branch_notes.append("p2_opt_rm_to_rollout_skip_no_rm_down_xfer")
            if rollout_up is None:
                p2_branch_notes.append("p2_opt_rm_to_rollout_skip_no_rollout_up")

        if transfer_candidates:
            transfer_candidates.sort(key=lambda x: x[0], reverse=True)
            best_gain, best_action, reason = transfer_candidates[0]
            if best_gain > self.hysteresis:
                actions.append(best_action)
                self._policy_log(
                    "decision",
                    action=best_action.action_type,
                    bottleneck_before=bottleneck_before,
                    gain=best_gain,
                    instance_id=(best_action.preferred_instance_ids or [None])[0],
                    model_name=best_action.model_name,
                    outcome="action",
                    policy_branch="p2_bottleneck_transfer",
                    reason=f"{reason}_gain_{best_gain:.6f}",
                )
                return actions, f"{reason}_gain_{best_gain:.6f}"
            p2_gain_rejected = best_gain

        # Priority 4: purely underloaded side can scale down itself.
        if (
            rollout_low
            and rollout_down is not None
            and not rm_full
            and rollout_down.waiting_queue_num <= self.max_waiting_queue_for_scale_down
        ):
            actions.append(
                ScalingAction(
                    action_type="scale_down",
                    role_name=rollout_down.role_name,
                    model_name=rollout_down.model_name,
                    preferred_instance_ids=[rollout_down.instance_id],
                    reason="rollout_low_scale_down",
                )
            )
            self._policy_log(
                "decision",
                action="scale_down",
                instance_id=rollout_down.instance_id,
                model_name=rollout_down.model_name,
                outcome="action",
                policy_branch="p3_rollout_self_down",
                reason="rollout_low_scale_down",
            )
            return actions, "rollout_low_scale_down"
        if (
            rm_low
            and rm_down is not None
            and not rollout_full
            and rm_down.waiting_queue_num <= self.max_waiting_queue_for_scale_down
        ):
            actions.append(
                ScalingAction(
                    action_type="scale_down",
                    role_name=rm_down.role_name,
                    model_name=rm_down.model_name,
                    preferred_instance_ids=[rm_down.instance_id],
                    reason="rm_low_scale_down",
                )
            )
            self._policy_log(
                "decision",
                action="scale_down",
                instance_id=rm_down.instance_id,
                model_name=rm_down.model_name,
                outcome="action",
                policy_branch="p3_rm_self_down",
                reason="rm_low_scale_down",
            )
            return actions, "rm_low_scale_down"

        detail = self._no_action_detail_strings(
            trainer_busy=trainer_busy,
            pending_total=pending_total,
            waiting_on=waiting_on,
            rollout_full=rollout_full,
            rm_full=rm_full,
            rollout_low=rollout_low,
            rm_low=rm_low,
            rollout_down_waiting=rollout_down_waiting,
            rm_down_waiting=rm_down_waiting,
            rollout_up=rollout_up,
            rm_up=rm_up,
            rollout_down=rollout_down,
            rm_down=rm_down,
            rollout_free_up=rollout_free_up,
            rm_free_up=rm_free_up,
            rollout_down_xfer=rollout_down_xfer,
            rm_down_xfer=rm_down_xfer,
            p2_gain_rejected=p2_gain_rejected,
            p2_branch_notes=p2_branch_notes,
        )
        if diagnostics:
            diagnostics.sort()
            merged = "|".join(diagnostics + detail)
            final_reason = f"no_action_{merged}"[:4096]
            self._policy_log_no_action(final_reason, diagnostics, detail)
            return [], final_reason
        merged_detail = " ;; ".join(detail)
        final_reason = f"no_action_{merged_detail}"[:4096]
        self._policy_log_no_action(final_reason, [], detail)
        return [], final_reason

    def decide(
        self,
        signals: list[InstanceSignal],
        execution_in_progress: bool = False,
        router_backlog_by_role: dict[PSRL_Role, int] | None = None,
        trainer_waiting_hint: dict[str, Any] | None = None,
        pending_scale_up_by_role: dict[PSRL_Role, int] | None = None,
    ) -> ScalingDecision:
        """
        Decide the scaling actions for the given signals.

        Args:
            signals: Per-instance snapshots (queues, utilization, awake/asleep, model
                metadata, etc.), usually assembled by ElasticMonitor from coordinators.
            execution_in_progress: True while ElasticExecutor is still executing a prior
                decision (SLEEP/WAKE_UP/ABORT).
            router_backlog_by_role: Pending request counts at the router, keyed by
                ``PSRL_Role``. Drives force-wake when a role has router backlog but no
                awaken instance (and no in-flight wake). Treated as empty when None.
            trainer_waiting_hint: Trainer-side view for Priority -1. Expected keys include
                ``trainer_busy`` (bool), ``waiting_on`` (e.g. ``"rollout"``,
                ``"reward"``, ``"none"``), and ``breakdown`` with ``pending_total``
                to bias scale-up toward the bottleneck when the trainer is idle but work
                remains. Treated as empty when None.
            pending_scale_up_by_role: In-flight scale-up counts per role from ElasticExecutor
                (queued handlers not yet finished). Suppresses duplicate force-wake while a
                wakeup RPC is already pending.

        Returns:
            ScalingDecision with actions (if any), reason string, ``estimated_lambda``,
            and ``role_to_total_mu`` for logging and telemetry.
        """
        if not self.enable:
            self._policy_log("decision", outcome="skipped", reason="policy_disabled")
            return ScalingDecision(actions=[], reason="policy_disabled", estimated_lambda=0.0, role_to_total_mu={})
        if not signals:
            self._policy_log("decision", outcome="skipped", reason="empty_signals")
            return ScalingDecision(actions=[], reason="empty_signals", estimated_lambda=0.0, role_to_total_mu={})
        
        # HIGHEST RULE: Must guarantee that any decision's execution is an ATOMIC operation.
        if execution_in_progress:
            self._policy_log("decision", outcome="skipped", reason="decision_execution_in_progress")
            return ScalingDecision(
                actions=[],
                reason="decision_execution_in_progress",
                estimated_lambda=0.0,
                role_to_total_mu={},
            )

        grouped = self._group_by_role(signals)
        instance_mu, role_total_mu, instance_mu_source = self._build_mu_maps(signals)
        estimated_lambda = self._estimate_lambda(signals, role_total_mu)
        source_counts: dict[str, int] = {}
        source_details: list[str] = []
        for signal in signals:
            key = (signal.role_name, signal.model_name, signal.instance_id)
            source = instance_mu_source.get(key, "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            source_details.append(
                f"{signal.role_name.name}/{signal.model_name}/{signal.instance_id}:{source}"
            )
        self._policy_log(
            "mu_source",
            details="|".join(source_details),
            formula_count=source_counts.get("formula", 0),
            fallback_runtime_count=source_counts.get("fallback_runtime", 0),
        )

        # Hard guarantee: if one role has zero awaken instances but router backlog exists,
        # force wake one instance for that role
        now_ms = time.time() * 1000
        backlog_map = router_backlog_by_role or {}
        pending_up = pending_scale_up_by_role or {}
        for role_name, role_signals in grouped.items():
            awaken_cnt = sum(1 for s in role_signals if s.is_awaken)
            backlog_cnt = int(backlog_map.get(role_name, 0))
            if awaken_cnt > 0 or backlog_cnt <= 0:
                continue
            if int(pending_up.get(role_name, 0)) > 0:
                continue
            other_role = PSRL_Role.RewardModel if role_name == PSRL_Role.Rollout else PSRL_Role.Rollout
            other_signals = grouped.get(other_role, [])
            force_pair = self._pick_scale_up_candidate_by_force(
                role_signals, other_signals, instance_mu
            )
            role_tag = role_name.name
            if force_pair is None:
                r = f"force_wake_needed_but_no_candidate_{role_tag}_backlog_{backlog_cnt}"
                self._policy_log(
                    "decision",
                    backlog_cnt=backlog_cnt,
                    outcome="no_action",
                    reason=r,
                    role_name=role_name,
                )
                return ScalingDecision(
                    actions=[],
                    reason=r,
                    estimated_lambda=estimated_lambda,
                    role_to_total_mu=role_total_mu,
                )
            force_up, sleep_victim = force_pair
            action = self._build_scale_up_action_with_elastic_placement(
                wake=force_up,
                target_role_signals=role_signals,
                other_role_signals=other_signals,
                instance_mu=instance_mu,
                reason=f"force_wake_from_router_backlog_{role_tag}_{backlog_cnt}",
            )
            if sleep_victim is not None and not action.pre_sleep_other_preferred:
                action.pre_sleep_other_preferred = [self._signal_entry(sleep_victim)]
            self.last_action_time_ms = now_ms
            fw_reason = f"force_wake_{role_tag}_backlog_{backlog_cnt}"
            self._policy_log(
                "decision",
                action="scale_up",
                backlog_cnt=backlog_cnt,
                instance_id=force_up.instance_id,
                model_name=force_up.model_name,
                outcome="action",
                policy_branch="force_wake_backlog",
                reason=fw_reason,
                role_name=role_name,
                pre_sleep_other=action.pre_sleep_other_preferred,
            )
            return ScalingDecision(
                actions=[action],
                reason=fw_reason,
                estimated_lambda=estimated_lambda,
                role_to_total_mu=role_total_mu,
            )

        if now_ms - self.last_action_time_ms < self.cooldown_ms:
            remain = self.cooldown_ms - (now_ms - self.last_action_time_ms)
            self._policy_log(
                "decision",
                cooldown_remaining_ms=remain,
                outcome="skipped",
                reason="cooldown",
            )
            return ScalingDecision(actions=[], reason="cooldown", estimated_lambda=0.0, role_to_total_mu={})

        # Safety guard: skip aggressive scale decisions when snapshots are stale.
        # all signals are unreliable, so we skip the decision.
        stale_count = 0
        for signal in signals:
            snapshot = {"timestamp": signal.snapshot_timestamp}
            if self._is_snapshot_staled(snapshot):
                stale_count += 1
        if stale_count == len(signals):
            self._policy_log(
                "decision",
                outcome="skipped",
                reason="all_signals_stale",
                stale_count=stale_count,
            )
            return ScalingDecision(actions=[], reason="all_signals_stale", estimated_lambda=0.0, role_to_total_mu={})
        
        actions, reason = self._make_stepwise_decision(
            grouped,
            instance_mu,
            role_total_mu,
            router_backlog_by_role=router_backlog_by_role,
            trainer_waiting_hint=trainer_waiting_hint,
        )

        if actions:
            self.last_action_time_ms = now_ms
        self._policy_log(
            "decision_summary",
            actions_count=len(actions),
            estimated_lambda=estimated_lambda,
            reason=reason,
            role_to_total_mu=dict(role_total_mu),
        )
        return ScalingDecision(
            actions=actions,
            reason=reason,
            estimated_lambda=estimated_lambda,
            role_to_total_mu=role_total_mu,
        )
