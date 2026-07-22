"""ITL/Throughput/L based elastic scaling policy for Rollout and RewardModel pools.

This module implements ``ITLScalingPolicy``, which chooses scale-up / scale-down
actions by maximizing a windowed objective built from a fitted latency model.

Latency model (per instance)::

    ITL = A * total_token_num + max(B, C * running_queue_num) + D

Symbols used in decisions:

- **Throughput**: Per-role effective throughput. The balanced path estimates
  ``requests / ITL`` at average queue depth and token count. The summed path
  estimates each active instance separately and sums its throughput. When
  ``role_throughput_weight_enable`` is on, system throughput becomes
  ``min(rollout_tp / w_rollout, rm_tp / w_rm)`` where the divisor is derived
  from per-role pressure.
- **Phi**: Estimated routing penalty from sleeping instances, proportional to
  ``migrate_time_s / ITL * delta_requests`` for each affected request.
- **L**: Windowed throughput gain minus ``Phi``.

Config lives under ``policy_config["itl_policy"]`` (see ``psrl.yaml``).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any

from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.cost_model_path import load_cost_model_json
from psrl.utils.deployment_mode import expand_ngpus_per_node
from psrl.utils.elastic_rm.scaling_policy import (
    InstanceSignal,
    ScalingAction,
    ScalingDecision,
    ScalingPolicy,
)


@dataclass(frozen=True)
class ITLModelParams:
    """Coefficients for the per-instance ITL latency model."""

    A: float = 0.0
    B: float = 1.0
    C: float = 1.0
    D: float = 0.0


@dataclass
class _ITLCandidate:
    """One evaluated scale-up or scale-down option for objective comparison."""

    action: ScalingAction
    rollout_n: int
    rm_n: int
    current_throughput: float
    next_throughput: float
    delta_throughput: float
    phi: float
    gain_l: float


@dataclass(frozen=True)
class _RouterWaitingLoad:
    """Router waiting queue prefix load used by the ITL objective."""

    request_count: float = 0.0
    token_count: float = 0.0


@dataclass(frozen=True)
class _RoleLoadSnapshot:
    """Per-role load snapshot used by current-state throughput calculations."""

    request_count: float
    token_count: float
    instance_rows: tuple[tuple[InstanceSignal, float, float], ...]


@dataclass(frozen=True)
class _WakePrefixPlan:
    """Deterministic wake prefix and the opposite-role instances it must evict."""

    wakes: tuple[InstanceSignal, ...]
    pre_sleep: tuple[InstanceSignal, ...]


def compute_itl(params: ITLModelParams, total_token_num: float, running_queue_num: float) -> float:
    """Compute per-instance ITL latency proxy in seconds."""
    x = max(0.0, float(running_queue_num))
    total_tokens = max(0.0, float(total_token_num))
    return max(float(params.A) * total_tokens + max(float(params.B), float(params.C) * x) + float(params.D), 1e-9)


def compute_itl_delta(params: ITLModelParams, running_queue_num: float, total_token_num: float = 0.0) -> float:
    """Compute the ITL increment from routing one more request to an instance."""
    x = max(0.0, float(running_queue_num))
    total_tokens = max(0.0, float(total_token_num))
    current = float(params.A) * total_tokens + max(float(params.B), float(params.C) * x) + float(params.D)
    next_value = float(params.A) * total_tokens + max(float(params.B), float(params.C) * (x + 1.0)) + float(params.D)
    return next_value - current


_TP_PP_KEY_RE = re.compile(r"^TP\d+_PP\d+$", re.IGNORECASE)


def format_tp_pp_key(tensor_parallel: int = 1, pipeline_parallel: int = 1) -> str:
    """Format a cost-model JSON bucket key such as ``TP1_PP1``."""
    return f"TP{int(tensor_parallel)}_PP{int(pipeline_parallel)}"


def _read_nested_config_value(config: Any, *keys: str, default: Any = None) -> Any:
    current = config
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue
        if not hasattr(current, key):
            return default
        current = getattr(current, key)
    return current


def _read_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def resolve_tp_pp_for_role_model(config: Any, role_name: PSRL_Role | str, model_name: str) -> str | None:
    """Derive ``TP{tp}_PP{pp}`` from rollout parallel settings for a role/model pair."""
    if config is None:
        return None
    role_key = getattr(role_name, "name", str(role_name)).lower()
    if role_key == PSRL_Role.Rollout.name.lower():
        rollout_cfg = _read_nested_config_value(config, "gen_actor_rollout_ref", "rollout")
        if rollout_cfg is None:
            return None
        tp = int(_read_config_value(rollout_cfg, "tensor_model_parallel_size", 1) or 1)
        pp = int(_read_config_value(rollout_cfg, "pipeline_model_parallel_size", 1) or 1)
        return format_tp_pp_key(tp, pp)
    if role_key == PSRL_Role.RewardModel.name.lower():
        reward_models = _read_nested_config_value(config, "reward_models_config", "reward_models")
        for reward_cfg in reward_models or []:
            reward_model_name = _read_config_value(reward_cfg, "reward_model_name")
            model_path = _read_nested_config_value(reward_cfg, "model", "path")
            stem = str(model_path).rstrip("/").split("/")[-1] if model_path else None
            if reward_model_name not in (None, model_name) and stem != model_name:
                continue
            rollout_cfg = _read_config_value(reward_cfg, "rollout", reward_cfg)
            tp = int(_read_config_value(rollout_cfg, "tensor_model_parallel_size", 1) or 1)
            pp = int(_read_config_value(rollout_cfg, "pipeline_model_parallel_size", 1) or 1)
            return format_tp_pp_key(tp, pp)
    return None


def _lookup_tp_pp_params_entry(payload: dict[str, Any], tp_pp: str | None) -> dict[str, Any] | None:
    if tp_pp and isinstance(payload.get(tp_pp), dict):
        return payload[tp_pp]
    for key in sorted(payload):
        if _TP_PP_KEY_RE.match(str(key)) and isinstance(payload[key], dict):
            return payload[key]
    return None


def _lookup_params_entry(
    payload: dict[str, Any],
    candidates: tuple[str, ...],
    role_key: str,
    model_name: str,
    *,
    tp_pp: str | None = None,
) -> dict[str, Any] | None:
    """Return the first matching parameter dict inside a cost-model payload."""
    tp_pp_entry = _lookup_tp_pp_params_entry(payload, tp_pp)
    if tp_pp_entry is not None:
        return tp_pp_entry
    for key in candidates:
        entry = payload.get(key)
        if isinstance(entry, dict):
            nested = _lookup_tp_pp_params_entry(entry, tp_pp)
            return nested or entry
    roles = payload.get("roles")
    if isinstance(roles, dict):
        for role_candidate in (role_key, role_key.lower()):
            role_entry = roles.get(role_candidate)
            if not isinstance(role_entry, dict):
                continue
            model_entry = role_entry.get(model_name)
            if isinstance(model_entry, dict):
                nested = _lookup_tp_pp_params_entry(model_entry, tp_pp)
                return nested or model_entry
            if all(k in role_entry for k in ("A", "B", "C", "D")):
                return role_entry
    models = payload.get("models")
    if isinstance(models, dict):
        model_entry = models.get(model_name)
        if isinstance(model_entry, dict):
            nested = _lookup_tp_pp_params_entry(model_entry, tp_pp)
            return nested or model_entry
    if all(k in payload for k in ("A", "B", "C", "D")):
        return payload
    return None


def _coerce_params(entry: dict[str, Any]) -> ITLModelParams | None:
    """Parse A/B/C/D or legacy latency field names from a config dict."""
    def _get(primary: str, legacy: str | None = None, default: float = 0.0) -> float:
        if primary in entry:
            return float(entry[primary])
        if legacy is not None and legacy in entry:
            return float(entry[legacy])
        return float(default)

    try:
        return ITLModelParams(
            A=_get("A", "attn_latency_k", 0.0),
            B=_get("B", "other_threshold", 1.0),
            C=_get("C", "other_latency_k", 1.0),
            D=_get("D", "attn_latency_b", 0.0),
        )
    except (TypeError, ValueError):
        return None


def resolve_itl_model_params(
    role_name: PSRL_Role | str,
    model_name: str,
    itl_config: dict[str, Any] | None = None,
    fallback_config: dict[str, Any] | None = None,
    tp_pp: str | None = None,
    config: Any = None,
) -> ITLModelParams:
    """Resolve fitted ITL coefficients for a role/model pair."""
    itl_cfg = itl_config or {}
    fallback_cfg = fallback_config or {}
    resolved_tp_pp = tp_pp or resolve_tp_pp_for_role_model(config, role_name, model_name)
    payloads: list[dict[str, Any]] = []
    inline = itl_cfg.get("model_params", {})
    if isinstance(inline, dict):
        payloads.append(inline)
    cost_model_path = itl_cfg.get("cost_model_path")
    if cost_model_path:
        loaded = load_cost_model_json(str(cost_model_path), model_name)
        if loaded is not None:
            payloads.append(loaded)
    for key in ("cost_model", "profile_cost_model"):
        legacy = fallback_cfg.get(key) if isinstance(fallback_cfg, dict) else None
        if isinstance(legacy, dict):
            payloads.append(legacy)

    role_key = getattr(role_name, "name", str(role_name))
    role_key_lower = role_key.lower()
    candidates = (
        f"{role_key}:{model_name}",
        f"{role_key_lower}:{model_name}",
        model_name,
        role_key,
        role_key_lower,
        "default",
    )
    for payload in payloads:
        entry = _lookup_params_entry(payload, candidates, role_key, model_name, tp_pp=resolved_tp_pp)
        if entry is None:
            continue
        params = _coerce_params(entry)
        if params is not None:
            return params
    return ITLModelParams()


class ITLScalingPolicy(ScalingPolicy):
    """Elastic policy that maximizes the ITL/throughput/L objective."""

    def __init__(self, config: dict, policy_config: dict | None = None):
        """Initialize ITL policy knobs from ``policy_config["itl_policy"]``."""
        super().__init__(config=config, policy_config=policy_config)
        self.policy_config = policy_config or {}
        itl_cfg = self.policy_config.get("itl_policy", {})
        if not isinstance(itl_cfg, dict):
            itl_cfg = {}
        self.itl_config = itl_cfg
        self.decision_window_s = float(itl_cfg.get("decision_window_s", 30.0))
        self.migrate_time_s = float(itl_cfg.get("migrate_time_s_initial", 30.0))
        self.migrate_time_ewma_alpha = float(itl_cfg.get("migrate_time_ewma_alpha", 0.2))
        self.min_gain = float(itl_cfg.get("min_gain", self.hysteresis))
        self.router_waiting_top_t = int(itl_cfg.get("router_waiting_top_t", 0))
        self.max_scale_instances_per_action = max(1, int(itl_cfg.get("max_scale_instances_per_action", 1)))
        self.throughput_objective = self._normalize_throughput_objective(
            itl_cfg.get("throughput_objective", itl_cfg.get("objective", "balanced_min"))
        )
        self.vllm_current_queue_scope = self._normalize_vllm_current_queue_scope(
            itl_cfg.get("vllm_current_queue_scope", "running_waiting")
        )
        self.current_state_include_router_waiting = self._normalize_bool(
            itl_cfg.get("current_state_include_router_waiting", True)
        )
        self.role_throughput_weight_enable = self._normalize_bool(
            itl_cfg.get("role_throughput_weight_enable", False)
        )
        self.role_throughput_weight_basis = self._normalize_role_throughput_weight_basis(
            itl_cfg.get("role_throughput_weight_basis", "request_count")
        )
        self.role_throughput_weight_mode = self._normalize_role_throughput_weight_mode(
            itl_cfg.get("role_throughput_weight_mode", "share")
        )
        # Starvation fallback: when one role has awake instances but zero demand
        # (no in-flight requests and no router backlog) the harmonic objective
        # collapses to 0 and can no longer distinguish candidates. The fallback
        # degrades the objective to the loaded side's bottleneck throughput and
        # re-flags the loaded role as the capacity bottleneck so the policy can
        # still reallocate capacity toward the side that actually has work.
        self.starvation_fallback = self._normalize_bool(
            itl_cfg.get("starvation_fallback", True)
        )

    @staticmethod
    def _normalize_bool(raw: Any) -> bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _normalize_throughput_objective(raw: Any) -> str:
        """Normalize the configured throughput objective mode."""
        mode = str(raw or "balanced_min").strip().lower()
        if mode in {"sum", "instance_sum", "sum_throughput", "throughput_sum"}:
            return "sum"
        return "balanced_min"

    @staticmethod
    def _normalize_vllm_current_queue_scope(raw: Any) -> str:
        """Normalize which vLLM queues contribute to current throughput estimates."""
        mode = str(raw or "running_waiting").strip().lower()
        if mode in {"running", "running_only", "vllm_running"}:
            return "running"
        return "running_waiting"

    @staticmethod
    def _normalize_role_throughput_weight_basis(raw: Any) -> str:
        """Normalize the per-role throughput weighting pressure basis."""
        mode = str(raw or "request_count").strip().lower()
        if mode in {"token_count", "tokens", "token", "total_tokens"}:
            return "token_count"
        if mode in {"none", "off", "disabled", "uniform"}:
            return "none"
        return "request_count"

    @staticmethod
    def _normalize_role_throughput_weight_mode(raw: Any) -> str:
        """Normalize how pressure becomes a per-role throughput divisor."""
        mode = str(raw or "share").strip().lower()
        if mode in {"raw", "absolute", "count"}:
            return "raw"
        return "share"

    def _vllm_current_request_load(self, signal: InstanceSignal) -> float:
        load = float(signal.running_queue_num)
        if self.vllm_current_queue_scope == "running_waiting":
            load += float(signal.waiting_queue_num)
        return load

    def _params_for_signal(self, signal: InstanceSignal) -> ITLModelParams:
        """Resolve ITL coefficients for the signal's role and model."""
        return resolve_itl_model_params(
            role_name=signal.role_name,
            model_name=signal.model_name,
            itl_config=self.itl_config,
            fallback_config=self.policy_config,
            config=self.config,
        )

    @staticmethod
    def _normalize_router_waiting_load(raw: Any) -> _RouterWaitingLoad:
        """Parse router waiting summary into count/token load."""
        if raw is None:
            return _RouterWaitingLoad()
        if isinstance(raw, dict):
            requests = raw.get("count", raw.get("request_count", raw.get("pending", 0)))
            tokens = raw.get("total_tokens", raw.get("token_count", 0))
        else:
            requests = raw
            tokens = 0
        try:
            request_count = max(0.0, float(requests))
        except (TypeError, ValueError):
            request_count = 0.0
        try:
            token_count = max(0.0, float(tokens))
        except (TypeError, ValueError):
            token_count = 0.0
        return _RouterWaitingLoad(request_count=request_count, token_count=token_count)

    def _current_router_waiting_load(self, raw: Any) -> _RouterWaitingLoad:
        """Router load used by current-state throughput estimates."""
        if not self.current_state_include_router_waiting:
            return _RouterWaitingLoad()
        return self._normalize_router_waiting_load(raw)

    def _role_total_load(
        self,
        role_signals: list[InstanceSignal],
        router_waiting_load: _RouterWaitingLoad | None = None,
    ) -> tuple[float, float]:
        """Return total request and token pressure for a role."""
        waiting = router_waiting_load or _RouterWaitingLoad()
        return (
            sum(self._vllm_current_request_load(signal) for signal in role_signals) + waiting.request_count,
            sum(float(signal.total_token_num) for signal in role_signals) + waiting.token_count,
        )

    def _role_pressure(
        self,
        role_signals: list[InstanceSignal],
        router_waiting_load: _RouterWaitingLoad | None,
    ) -> float:
        """Per-role pressure value used for throughput weighting."""
        request_count, token_count = self._role_total_load(role_signals, router_waiting_load)
        if self.role_throughput_weight_basis == "token_count":
            return max(0.0, token_count)
        if self.role_throughput_weight_basis == "none":
            return 0.0
        return max(0.0, request_count)

    def _role_throughput_weights(
        self,
        rollout_signals: list[InstanceSignal],
        rm_signals: list[InstanceSignal],
        rollout_waiting: _RouterWaitingLoad | None,
        rm_waiting: _RouterWaitingLoad | None,
    ) -> tuple[float, float]:
        """Return per-role throughput divisors ``(w_rollout, w_rm)``."""
        if not self.role_throughput_weight_enable or self.role_throughput_weight_basis == "none":
            return (0.5, 0.5)
        rollout_pressure = self._role_pressure(rollout_signals, rollout_waiting)
        rm_pressure = self._role_pressure(rm_signals, rm_waiting)
        if self.role_throughput_weight_mode == "raw":
            return (rollout_pressure if rollout_pressure > 0.0 else 1.0, rm_pressure if rm_pressure > 0.0 else 1.0)
        total = rollout_pressure + rm_pressure
        if total <= 0.0:
            return (0.5, 0.5)
        return (rollout_pressure / total, rm_pressure / total)

    @staticmethod
    def _weighted_system_throughput(rollout_tp: float, rm_tp: float, w_rollout: float, w_rm: float) -> float:
        """Apply per-role throughput divisors and return the bottleneck."""
        wr = w_rollout if w_rollout > 0.0 else 1.0
        wm = w_rm if w_rm > 0.0 else 1.0
        return min(rollout_tp / wr, rm_tp / wm)

    @staticmethod
    def _idle_instance_throughput() -> float:
        """Throughput assigned to an instance/role with no outstanding requests.

        The bottleneck (``min``) objective treats idle as ``inf`` so an idle
        side is never flagged as the bottleneck. The harmonic objective
        overrides this to ``0.0``: an idle instance produces nothing, so it
        must not inflate the harmonic mean (which would otherwise treat the
        side as having no resistance and push system throughput toward inf).
        """
        return math.inf

    def _role_throughput(
        self,
        role_signals: list[InstanceSignal],
        n_instances: int,
        router_waiting_load: _RouterWaitingLoad | None = None,
    ) -> float:
        """Per-role throughput under balanced load across ``n_instances``."""
        reference = next((s for s in role_signals if s.is_awaken), role_signals[0] if role_signals else None)
        total_requests, total_tokens = self._role_total_load(role_signals, router_waiting_load)
        if total_requests <= 0.0:
            return self._idle_instance_throughput()
        if n_instances <= 0:
            return 0.0
        if reference is None:
            return 0.0
        avg_requests = total_requests / float(n_instances)
        avg_tokens = total_tokens / float(n_instances)
        itl = compute_itl(self._params_for_signal(reference), avg_tokens, avg_requests)
        return avg_requests / itl

    @staticmethod
    def _active_role_signals(
        role_signals: list[InstanceSignal],
        n_instances: int,
        wake_instance_ids: set[int] | None = None,
        sleep_instance_ids: set[int] | None = None,
    ) -> list[InstanceSignal]:
        """Resolve the active signal set for a hypothetical candidate state."""
        wake_ids = wake_instance_ids or set()
        sleep_ids = sleep_instance_ids or set()
        active = [
            signal
            for signal in role_signals
            if (signal.is_awaken or int(signal.instance_id) in wake_ids) and int(signal.instance_id) not in sleep_ids
        ]
        if len(active) < n_instances:
            active_ids = {int(signal.instance_id) for signal in active}
            active.extend(
                signal
                for signal in role_signals
                if int(signal.instance_id) not in active_ids
                and int(signal.instance_id) not in sleep_ids
                and not signal.is_awaken
            )
        return active[: max(0, int(n_instances))]

    def _role_throughput_sum(
        self,
        role_signals: list[InstanceSignal],
        n_instances: int,
        router_waiting_load: _RouterWaitingLoad | None = None,
        wake_instance_ids: set[int] | None = None,
        sleep_instance_ids: set[int] | None = None,
    ) -> float:
        """Per-role throughput as a sum of per-instance throughput estimates."""
        active = self._active_role_signals(role_signals, n_instances, wake_instance_ids, sleep_instance_ids)
        total_requests, _ = self._role_total_load(role_signals, router_waiting_load)
        if total_requests <= 0.0:
            return self._idle_instance_throughput()
        if n_instances <= 0 or not active:
            return 0.0
        waiting_load = router_waiting_load or _RouterWaitingLoad()
        waiting_requests_per_instance = waiting_load.request_count / float(len(active))
        waiting_tokens_per_instance = waiting_load.token_count / float(len(active))
        throughput_sum = 0.0
        for signal in active:
            request_load = self._vllm_current_request_load(signal) + waiting_requests_per_instance
            token_load = float(signal.total_token_num) + waiting_tokens_per_instance
            if request_load <= 0.0:
                throughput_sum += self._idle_instance_throughput()
                continue
            itl = compute_itl(self._params_for_signal(signal), token_load, request_load)
            throughput_sum += request_load / itl
        return throughput_sum

    def _current_role_load_snapshot(
        self,
        role_signals: list[InstanceSignal],
        n_instances: int,
        router_waiting_load: _RouterWaitingLoad | None = None,
    ) -> _RoleLoadSnapshot:
        """Build current per-instance load rows, distributing router waiting load."""
        active = self._active_role_signals(role_signals, n_instances)
        waiting = router_waiting_load or _RouterWaitingLoad()
        if not active:
            return _RoleLoadSnapshot(waiting.request_count, waiting.token_count, tuple())
        waiting_requests_per_instance = waiting.request_count / float(len(active))
        waiting_tokens_per_instance = waiting.token_count / float(len(active))
        rows = tuple(
            (
                signal,
                self._vllm_current_request_load(signal) + waiting_requests_per_instance,
                float(signal.total_token_num) + waiting_tokens_per_instance,
            )
            for signal in active
        )
        return _RoleLoadSnapshot(
            request_count=sum(row[1] for row in rows),
            token_count=sum(row[2] for row in rows),
            instance_rows=rows,
        )

    @staticmethod
    def _throughput_delta(next_throughput: float, current_throughput: float) -> float:
        """Compute throughput delta while keeping all-idle ``inf -> inf`` transitions stable."""
        if math.isinf(next_throughput) and math.isinf(current_throughput) and next_throughput == current_throughput:
            return 0.0
        return next_throughput - current_throughput

    def _windowed_throughput_gain(self, delta_throughput: float, rollout_n: int, rm_n: int) -> float:
        """Compute the throughput part of L for the configured objective."""
        if self.throughput_objective == "sum":
            return self.decision_window_s * delta_throughput
        return self.decision_window_s * delta_throughput * (rollout_n + rm_n)

    def record_migration_observation(self, sleep_time_s: float, wake_time_s: float) -> None:
        """Update migration-time estimate from observed SLEEP and WAKE_UP durations."""
        observed = max(0.0, float(sleep_time_s)) + max(0.0, float(wake_time_s))
        if observed <= 0.0:
            return
        alpha = min(1.0, max(0.0, float(self.migrate_time_ewma_alpha)))
        self.migrate_time_s = alpha * observed + (1.0 - alpha) * self.migrate_time_s
        self._policy_log(
            "itl_migration_observation",
            observed_s=f"{observed:.3f}",
            sleep_s=f"{float(sleep_time_s):.3f}",
            wake_s=f"{float(wake_time_s):.3f}",
            ewma_s=f"{self.migrate_time_s:.3f}",
        )

    def _system_throughput(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        rollout_n: int,
        rm_n: int,
        router_backlog_by_role: dict[PSRL_Role, Any] | None = None,
        wake_ids_by_role: dict[PSRL_Role, set[int]] | None = None,
        sleep_ids_by_role: dict[PSRL_Role, set[int]] | None = None,
    ) -> float:
        """System throughput: bottleneck across Rollout and RewardModel."""
        backlog_map = router_backlog_by_role or {}
        wake_map = wake_ids_by_role or {}
        sleep_map = sleep_ids_by_role or {}
        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        rollout_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.Rollout))
        rm_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel))
        if self.throughput_objective == "sum":
            rollout_tp = self._role_throughput_sum(
                rollout_signals,
                rollout_n,
                rollout_waiting,
                wake_map.get(PSRL_Role.Rollout),
                sleep_map.get(PSRL_Role.Rollout),
            )
            rm_tp = self._role_throughput_sum(
                rm_signals,
                rm_n,
                rm_waiting,
                wake_map.get(PSRL_Role.RewardModel),
                sleep_map.get(PSRL_Role.RewardModel),
            )
        else:
            rollout_tp = self._role_throughput(rollout_signals, rollout_n, rollout_waiting)
            rm_tp = self._role_throughput(rm_signals, rm_n, rm_waiting)
        if not self.role_throughput_weight_enable:
            return min(rollout_tp, rm_tp)
        w_rollout, w_rm = self._role_throughput_weights(rollout_signals, rm_signals, rollout_waiting, rm_waiting)
        return self._weighted_system_throughput(rollout_tp, rm_tp, w_rollout, w_rm)

    def _current_role_throughputs(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        rollout_n: int,
        rm_n: int,
        router_backlog_by_role: dict[PSRL_Role, Any] | None = None,
    ) -> dict[PSRL_Role, float]:
        """Compute per-role throughput from the current per-instance load."""
        backlog_map = router_backlog_by_role or {}
        return {
            PSRL_Role.Rollout: self._role_throughput_sum(
                grouped.get(PSRL_Role.Rollout, []),
                rollout_n,
                self._current_router_waiting_load(backlog_map.get(PSRL_Role.Rollout)),
            ),
            PSRL_Role.RewardModel: self._role_throughput_sum(
                grouped.get(PSRL_Role.RewardModel, []),
                rm_n,
                self._current_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel)),
            ),
        }

    def _instance_phi(self, signal: InstanceSignal, delta_requests: float) -> float:
        """Routing penalty for moving ``delta_requests`` off an instance."""
        if delta_requests <= 0.0:
            return 0.0
        itl = compute_itl(self._params_for_signal(signal), signal.total_token_num, signal.running_queue_num)
        return self.migrate_time_s / itl * delta_requests

    def _sleep_phi(self, victims: list[InstanceSignal]) -> float:
        """Phi for putting instances to sleep: re-route all their running requests."""
        return sum(self._instance_phi(victim, float(victim.running_queue_num)) for victim in victims)

    @staticmethod
    def _bundle_conflict_free(awake_after: list[InstanceSignal]) -> bool:
        """Return False if two awake instances would share the same bundle key."""
        occupied: set[tuple[str, int]] = set()
        for signal in awake_after:
            if not signal.bundle_keys:
                continue
            if occupied.intersection(signal.bundle_keys):
                return False
            occupied.update(signal.bundle_keys)
        return True

    @staticmethod
    def _wake_conflicts(wakes: list[InstanceSignal], other_signals: list[InstanceSignal]) -> list[InstanceSignal]:
        """Return awake instances from the other role whose bundles overlap ``wakes``."""
        other_awake = [s for s in other_signals if s.is_awaken]
        return [
            s
            for s in other_awake
            if any(wake.bundle_keys and s.bundle_keys and wake.bundle_keys.intersection(s.bundle_keys) for wake in wakes)
        ]

    @staticmethod
    def _trainer_pool_is_available(trainer_waiting_hint: dict[str, Any] | None) -> bool:
        """Return True when elastic replicas may use trainer-pool devices."""
        hint = trainer_waiting_hint or {}
        return not bool(hint.get("trainer_busy", True))

    def _signal_pool_available(
        self,
        signal: InstanceSignal,
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> bool:
        """Return whether a signal's pool can be used for elastic candidates."""
        if signal.pool_id != "train_pool":
            return True
        return self._trainer_pool_is_available(trainer_waiting_hint)

    @staticmethod
    def _pool_wake_priority(pool_id: str | None) -> int:
        """Wake-selection priority for a pool: share pool (0) before train pool (1).

        Scale-up candidates must prefer free devices in the share pool and only
        fall back to the train pool when no share-pool device is available.
        """
        return 0 if pool_id != "train_pool" else 1

    @staticmethod
    def _bundle_conflicts_with_occupied(signal: InstanceSignal, occupied: set[tuple[str, int]]) -> bool:
        if not signal.bundle_keys:
            return False
        return bool(occupied.intersection(signal.bundle_keys))

    def _pick_first_conflict_free_batch(
        self,
        ordered: list[InstanceSignal],
        num_instances: int,
        awake_after_sleep: list[InstanceSignal],
    ) -> list[InstanceSignal] | None:
        """Pick the first priority-ordered wake batch without bundle conflicts.

        This intentionally avoids enumerating all ``combinations(n, k)``. Large
        elastic pools may request k=16..32 wakes from 32 candidates; exhaustive
        combination search is intractable and can stall the monitor loop.
        """
        if num_instances <= 0:
            return []
        occupied: set[tuple[str, int]] = set()
        for signal in awake_after_sleep:
            if signal.bundle_keys:
                occupied.update(signal.bundle_keys)
        batch: list[InstanceSignal] = []
        for signal in ordered:
            if self._bundle_conflicts_with_occupied(signal, occupied):
                continue
            batch.append(signal)
            if signal.bundle_keys:
                occupied.update(signal.bundle_keys)
            if len(batch) >= num_instances:
                return batch
        return None

    def _pick_first_batch_with_eviction(
        self,
        ordered: list[InstanceSignal],
        num_instances: int,
        other_awake_after_sleep: list[InstanceSignal],
        other_signals: list[InstanceSignal],
    ) -> tuple[list[InstanceSignal], list[InstanceSignal]] | None:
        """Pick a priority-ordered wake batch and its cross-role eviction set.

        The ordered list already encodes shared-pool-first and instance-id
        tie-breaks. We keep the batch itself bundle-conflict-free, then evict
        the awake instances from the other role that overlap selected wakes.
        """
        if num_instances <= 0:
            return [], []
        batch: list[InstanceSignal] = []
        occupied: set[tuple[str, int]] = set()
        for signal in ordered:
            if self._bundle_conflicts_with_occupied(signal, occupied):
                continue
            batch.append(signal)
            if signal.bundle_keys:
                occupied.update(signal.bundle_keys)
            if len(batch) >= num_instances:
                conflicts = self._wake_conflicts(batch, other_signals)
                remaining_other = len(other_awake_after_sleep) - len(conflicts)
                if remaining_other < self.min_awake_per_role:
                    return None
                surviving = [s for s in other_awake_after_sleep if s not in conflicts]
                if not self._bundle_conflict_free(surviving + batch):
                    return None
                return batch, conflicts
        return None

    def _elastic_device_capacity(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> int:
        """Estimate device capacity available to Rollout and RewardModel replicas."""
        bundle_keys = {
            key
            for role_signals in grouped.values()
            for signal in role_signals
            if self._signal_pool_available(signal, trainer_waiting_hint)
            for key in (signal.bundle_keys or frozenset())
        }
        if bundle_keys:
            return len(bundle_keys)
        elastic_cfg = _read_nested_config_value(self.config, "psrl", "deployment", "elastic_rm", default={})
        shared_nodes = int(_read_config_value(elastic_cfg, "shared_nnodes", 1) or 1)
        shared_gpu_spec = expand_ngpus_per_node(
            _read_config_value(elastic_cfg, "shared_ngpus_per_node", 1),
            shared_nodes,
            "psrl.deployment.elastic_rm.shared_ngpus_per_node",
        )
        capacity = max(0, sum(shared_gpu_spec))
        if bool(_read_config_value(elastic_cfg, "enable_trainer_pool", False)) and self._trainer_pool_is_available(
            trainer_waiting_hint
        ):
            trainer_nodes = int(_read_nested_config_value(self.config, "trainer", "nnodes", default=0) or 0)
            trainer_gpus = int(_read_nested_config_value(self.config, "trainer", "n_gpus_per_node", default=0) or 0)
            capacity += max(0, trainer_nodes * trainer_gpus)
        return capacity

    @staticmethod
    def _role_device_size(role_signals: list[InstanceSignal]) -> int:
        """Return a conservative per-instance device size for a role."""
        sizes = [len(signal.bundle_keys) for signal in role_signals if signal.bundle_keys]
        return max(sizes) if sizes else 1

    def _candidate_device_usage(
        self,
        rollout_signals: list[InstanceSignal],
        rm_signals: list[InstanceSignal],
        rollout_n: int,
        rm_n: int,
    ) -> int:
        """Estimate devices occupied by a candidate role-count pair."""
        return (
            max(0, int(rollout_n)) * self._role_device_size(rollout_signals)
            + max(0, int(rm_n)) * self._role_device_size(rm_signals)
        )

    def _estimated_instance_throughput(self, signal: InstanceSignal) -> float:
        """Estimate current per-instance throughput for shrink victim ordering."""
        if signal.generation_throughput > 0.0:
            return float(signal.generation_throughput)
        request_load = self._vllm_current_request_load(signal)
        if request_load <= 0.0:
            return 0.0
        itl = compute_itl(self._params_for_signal(signal), float(signal.total_token_num), request_load)
        return request_load / itl

    def _pick_low_throughput_victims(
        self,
        role_signals: list[InstanceSignal],
        num_instances: int,
        ordered_victims_by_model: dict[str, list[InstanceSignal]] | None = None,
    ) -> list[InstanceSignal] | None:
        """Pick same-model awake victims with the lowest current throughput."""
        if num_instances <= 0:
            return []
        by_model = ordered_victims_by_model
        if by_model is None:
            by_model = {}
            for signal in role_signals:
                if signal.is_awaken and not signal.is_training:
                    by_model.setdefault(signal.model_name, []).append(signal)
        candidate_groups: list[list[InstanceSignal]] = []
        for signals in by_model.values():
            if len(signals) < num_instances:
                continue
            if ordered_victims_by_model is None:
                signals = sorted(
                    signals,
                    key=lambda item: (
                        self._estimated_instance_throughput(item),
                        self._instance_phi(item, float(item.running_queue_num)),
                        int(item.instance_id),
                    ),
                )
            candidate_groups.append(signals[:num_instances])
        if not candidate_groups:
            return None
        return min(
            candidate_groups,
            key=lambda items: (
                sum(self._estimated_instance_throughput(item) for item in items),
                self._sleep_phi(items),
                [int(item.instance_id) for item in items],
            ),
        )

    def _pick_wake_batch(
        self,
        *,
        role_signals: list[InstanceSignal],
        num_instances: int,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        sleep_victims: list[InstanceSignal],
        trainer_waiting_hint: dict[str, Any] | None = None,
        ordered_wakeable_by_model: dict[str, list[InstanceSignal]] | None = None,
    ) -> list[InstanceSignal] | None:
        """Pick same-model wake targets that fit after planned sleeps."""
        if num_instances <= 0:
            return []
        excluded = {
            (victim.role_name, victim.model_name, int(victim.instance_id))
            for victim in sleep_victims
        }
        awake_after_sleep = [
            signal
            for signals in grouped.values()
            for signal in signals
            if signal.is_awaken
            and (signal.role_name, signal.model_name, int(signal.instance_id)) not in excluded
        ]
        by_model = ordered_wakeable_by_model
        if by_model is None:
            wakeable = [
                signal
                for signal in role_signals
                if self._is_scale_up_available(signal) and self._signal_pool_available(signal, trainer_waiting_hint)
            ]
            by_model = {}
            for signal in wakeable:
                by_model.setdefault(signal.model_name, []).append(signal)
        batches: list[list[InstanceSignal]] = []
        for signals in by_model.values():
            if len(signals) < num_instances:
                continue
            if ordered_wakeable_by_model is None:
                # Share-pool devices first; train-pool devices only fill the rest.
                signals = sorted(
                    signals,
                    key=lambda item: (self._pool_wake_priority(item.pool_id), int(item.instance_id)),
                )
            batch = self._pick_first_conflict_free_batch(signals, num_instances, awake_after_sleep)
            if batch is not None:
                batches.append(batch)
        if not batches:
            return None
        # Prefer batches with the fewest train-pool instances (share pool first),
        # then by pool priority and instance id for a stable tie-break.
        def _batch_key(items: list[InstanceSignal]) -> tuple[Any, ...]:
            train_count = sum(1 for it in items if self._pool_wake_priority(it.pool_id))
            pool_id_keys = [
                (self._pool_wake_priority(it.pool_id), int(it.instance_id)) for it in items
            ]
            return (train_count, *pool_id_keys)

        return min(batches, key=_batch_key)

    def _pick_wake_batch_with_eviction(
        self,
        *,
        target_signals: list[InstanceSignal],
        other_signals: list[InstanceSignal],
        num_instances: int,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        sleep_victims: list[InstanceSignal],
        trainer_waiting_hint: dict[str, Any] | None = None,
        ordered_wakeable_by_model: dict[str, list[InstanceSignal]] | None = None,
    ) -> tuple[list[InstanceSignal], list[InstanceSignal]] | None:
        """Pick a wake batch, evicting cross-role bundle conflicts if needed.

        Returns ``(wakes, evicted_other)`` where ``evicted_other`` are awake
        instances on the other role whose bundles overlap the chosen wakes and
        must be slept first. Tries a conflict-free batch first; only when none
        exists does it fall back to evicting the other role. Eviction is rejected
        if it would drop the other role below ``min_awake_per_role``.
        """
        wakes = self._pick_wake_batch(
            role_signals=target_signals,
            num_instances=num_instances,
            grouped=grouped,
            sleep_victims=sleep_victims,
            trainer_waiting_hint=trainer_waiting_hint,
            ordered_wakeable_by_model=ordered_wakeable_by_model,
        )
        if wakes:
            return wakes, []

        if num_instances <= 0:
            return [], []
        excluded = {
            (victim.role_name, victim.model_name, int(victim.instance_id))
            for victim in sleep_victims
        }
        other_awake_after_sleep = [
            s for s in other_signals
            if s.is_awaken and (s.role_name, s.model_name, int(s.instance_id)) not in excluded
        ]
        by_model = ordered_wakeable_by_model
        if by_model is None:
            wakeable = [
                s for s in target_signals
                if self._is_scale_up_available(s) and self._signal_pool_available(s, trainer_waiting_hint)
            ]
            by_model = {}
            for signal in wakeable:
                by_model.setdefault(signal.model_name, []).append(signal)
        best: tuple[list[InstanceSignal], list[InstanceSignal]] | None = None
        best_key: tuple[Any, ...] | None = None
        for signals in by_model.values():
            if len(signals) < num_instances:
                continue
            if ordered_wakeable_by_model is None:
                # Share-pool devices first; train-pool devices only fill the rest.
                signals = sorted(
                    signals,
                    key=lambda item: (self._pool_wake_priority(item.pool_id), int(item.instance_id)),
                )
            pick = self._pick_first_batch_with_eviction(
                signals,
                num_instances,
                other_awake_after_sleep,
                other_signals,
            )
            if pick is None:
                continue
            batch, conflicts = pick
            train_count = sum(1 for it in batch if self._pool_wake_priority(it.pool_id))
            pool_id_keys = [
                (self._pool_wake_priority(it.pool_id), int(it.instance_id)) for it in batch
            ]
            key = (len(conflicts), train_count, pool_id_keys)
            if best_key is None or key < best_key:
                best = (batch, conflicts)
                best_key = key
        return best

    def _role_count_bounds(
        self,
        role_signals: list[InstanceSignal],
        current_n: int,
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> tuple[int, int]:
        """Return feasible min/max awake counts for one role."""
        wakeable_n = sum(
            1
            for signal in role_signals
            if self._is_scale_up_available(signal) and self._signal_pool_available(signal, trainer_waiting_hint)
        )
        return self.min_awake_per_role, current_n + wakeable_n

    def _selection_orders_for_role(
        self,
        role_signals: list[InstanceSignal],
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> tuple[dict[str, list[InstanceSignal]], dict[str, list[InstanceSignal]]]:
        """Precompute deterministic wake/sleep priority orders for one decide cycle."""
        wakeable_by_model: dict[str, list[InstanceSignal]] = {}
        victims_by_model: dict[str, list[InstanceSignal]] = {}
        for signal in role_signals:
            if self._is_scale_up_available(signal) and self._signal_pool_available(signal, trainer_waiting_hint):
                wakeable_by_model.setdefault(signal.model_name, []).append(signal)
            if signal.is_awaken and not signal.is_training:
                victims_by_model.setdefault(signal.model_name, []).append(signal)

        for model_name, signals in list(wakeable_by_model.items()):
            wakeable_by_model[model_name] = sorted(
                signals,
                key=lambda item: (self._pool_wake_priority(item.pool_id), int(item.instance_id)),
            )
        for model_name, signals in list(victims_by_model.items()):
            victims_by_model[model_name] = sorted(
                signals,
                key=lambda item: (
                    self._estimated_instance_throughput(item),
                    self._instance_phi(item, float(item.running_queue_num)),
                    int(item.instance_id),
                ),
            )
        return wakeable_by_model, victims_by_model

    @staticmethod
    def _signal_key(signal: InstanceSignal) -> tuple[str, str, int]:
        return (str(signal.role_name), signal.model_name, int(signal.instance_id))

    def _wake_prefix_plans_for_role(
        self,
        *,
        target_signals: list[InstanceSignal],
        other_signals: list[InstanceSignal],
        current_other_n: int,
        max_batch: int,
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> dict[int, _WakePrefixPlan]:
        """Build deterministic wake prefixes for one scale-up side.

        A count delta maps to exactly one concrete wake/presleep choice: sort all
        wakeable target instances once, then take the first feasible prefix. Free
        devices come before occupied devices; among free devices, shared-pool
        instances come before train-pool instances; occupied devices prefer the
        ones whose opposite-role eviction has lower migration cost.
        """
        if max_batch <= 0:
            return {}

        other_awake = [signal for signal in other_signals if signal.is_awaken and not signal.is_training]
        other_by_bundle: dict[tuple[str, int], list[InstanceSignal]] = {}
        for signal in other_awake:
            for bundle_key in signal.bundle_keys or ():
                other_by_bundle.setdefault(bundle_key, []).append(signal)

        def conflicts_for(signal: InstanceSignal) -> tuple[InstanceSignal, ...]:
            conflicts: dict[tuple[str, str, int], InstanceSignal] = {}
            for bundle_key in signal.bundle_keys or ():
                for other in other_by_bundle.get(bundle_key, ()):
                    conflicts[self._signal_key(other)] = other
            return tuple(sorted(conflicts.values(), key=lambda item: int(item.instance_id)))

        def wake_key(signal: InstanceSignal) -> tuple[Any, ...]:
            conflicts = conflicts_for(signal)
            free_device = 0 if not conflicts else 1
            conflict_phi = sum(self._instance_phi(item, float(item.running_queue_num)) for item in conflicts)
            conflict_requests = sum(float(item.running_queue_num) for item in conflicts)
            return (
                free_device,
                self._pool_wake_priority(signal.pool_id),
                conflict_phi,
                conflict_requests,
                len(conflicts),
                int(signal.instance_id),
            )

        wakeable_by_model: dict[str, list[InstanceSignal]] = {}
        for signal in target_signals:
            if self._is_scale_up_available(signal) and self._signal_pool_available(signal, trainer_waiting_hint):
                wakeable_by_model.setdefault(signal.model_name, []).append(signal)
        if not wakeable_by_model:
            return {}

        def plan_key(plan: _WakePrefixPlan) -> tuple[Any, ...]:
            train_count = sum(1 for signal in plan.wakes if self._pool_wake_priority(signal.pool_id))
            pre_sleep_phi = sum(self._instance_phi(item, float(item.running_queue_num)) for item in plan.pre_sleep)
            pre_sleep_requests = sum(float(item.running_queue_num) for item in plan.pre_sleep)
            return (
                pre_sleep_phi,
                pre_sleep_requests,
                len(plan.pre_sleep),
                train_count,
                [int(signal.instance_id) for signal in plan.wakes],
            )

        best_plans: dict[int, _WakePrefixPlan] = {}
        best_keys: dict[int, tuple[Any, ...]] = {}
        for model_signals in wakeable_by_model.values():
            selected: list[InstanceSignal] = []
            selected_target_bundles: set[tuple[str, int]] = set()
            pre_sleep_by_key: dict[tuple[str, str, int], InstanceSignal] = {}

            for signal in sorted(model_signals, key=wake_key):
                if signal.bundle_keys and selected_target_bundles.intersection(signal.bundle_keys):
                    continue
                signal_conflicts = conflicts_for(signal)
                next_pre_sleep = dict(pre_sleep_by_key)
                for victim in signal_conflicts:
                    next_pre_sleep[self._signal_key(victim)] = victim
                if current_other_n - len(next_pre_sleep) < self.min_awake_per_role:
                    continue

                selected.append(signal)
                if signal.bundle_keys:
                    selected_target_bundles.update(signal.bundle_keys)
                pre_sleep_by_key = next_pre_sleep
                batch_size = len(selected)
                if batch_size <= max_batch:
                    plan = _WakePrefixPlan(
                        wakes=tuple(selected),
                        pre_sleep=tuple(pre_sleep_by_key.values()),
                    )
                    key = plan_key(plan)
                    if batch_size not in best_keys or key < best_keys[batch_size]:
                        best_plans[batch_size] = plan
                        best_keys[batch_size] = key
                if batch_size >= max_batch:
                    break

        return best_plans

    def _build_rebalanced_count_candidate(
        self,
        *,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        rollout_n: int,
        rm_n: int,
        current_rollout_n: int,
        current_rm_n: int,
        current_throughput: float,
        router_backlog_by_role: dict[PSRL_Role, Any] | None = None,
        trainer_waiting_hint: dict[str, Any] | None = None,
        victim_orders_by_role: dict[PSRL_Role, dict[str, list[InstanceSignal]]] | None = None,
        wake_prefix_plans_by_role: dict[PSRL_Role, dict[int, _WakePrefixPlan]] | None = None,
    ) -> _ITLCandidate | None:
        """Build one candidate from target role counts after planned rebalancing."""
        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        rollout_delta = int(rollout_n) - int(current_rollout_n)
        rm_delta = int(rm_n) - int(current_rm_n)
        if rollout_delta == 0 and rm_delta == 0:
            return None
        sleep_victims: list[InstanceSignal] = []
        wakes: list[InstanceSignal] = []
        if rollout_delta < 0:
            victims = self._pick_low_throughput_victims(
                rollout_signals,
                -rollout_delta,
                (victim_orders_by_role or {}).get(PSRL_Role.Rollout),
            )
            if victims is None:
                return None
            sleep_victims.extend(victims)
        if rm_delta < 0:
            victims = self._pick_low_throughput_victims(
                rm_signals,
                -rm_delta,
                (victim_orders_by_role or {}).get(PSRL_Role.RewardModel),
            )
            if victims is None:
                return None
            sleep_victims.extend(victims)
        if rollout_delta > 0 and rm_delta > 0:
            return None
        # Evicted cross-role instances (awake on the other role whose bundles
        # overlap the chosen wakes). Filled in by the scale-up branches below and
        # folded into sleep_victims / the other role's count afterwards.
        evicted_other: list[InstanceSignal] = []
        if rollout_delta > 0:
            plan = (wake_prefix_plans_by_role or {}).get(PSRL_Role.Rollout, {}).get(rollout_delta)
            if plan is None:
                return None
            wakes = list(plan.wakes)
            evicted_other = list(plan.pre_sleep)
            action = ScalingAction(
                action_type="scale_up",
                role_name=PSRL_Role.Rollout,
                model_name=wakes[0].model_name,
                num_instances=len(wakes),
                preferred_instance_ids=[int(wake.instance_id) for wake in wakes],
                reason="itl_objective_scale_up",
                pre_sleep_other_preferred=[self._signal_entry(victim) for victim in sleep_victims] or None,
            )
        elif rm_delta > 0:
            plan = (wake_prefix_plans_by_role or {}).get(PSRL_Role.RewardModel, {}).get(rm_delta)
            if plan is None:
                return None
            wakes = list(plan.wakes)
            evicted_other = list(plan.pre_sleep)
            action = ScalingAction(
                action_type="scale_up",
                role_name=PSRL_Role.RewardModel,
                model_name=wakes[0].model_name,
                num_instances=len(wakes),
                preferred_instance_ids=[int(wake.instance_id) for wake in wakes],
                reason="itl_objective_scale_up",
                pre_sleep_other_preferred=[self._signal_entry(victim) for victim in sleep_victims] or None,
            )
        else:
            if not sleep_victims:
                return None
            target_role = sleep_victims[0].role_name
            target_model = sleep_victims[0].model_name
            if any(victim.role_name != target_role or victim.model_name != target_model for victim in sleep_victims):
                return None
            action = ScalingAction(
                action_type="scale_down",
                role_name=target_role,
                model_name=target_model,
                num_instances=len(sleep_victims),
                preferred_instance_ids=[int(victim.instance_id) for victim in sleep_victims],
                reason="itl_objective_scale_down",
            )
        # Fold cross-role evictions into the sleep set and adjust the other role's
        # awake count. ``evicted_other`` is non-empty only for scale-up branches
        # where the chosen wakes overlap awake instances on the other role.
        if evicted_other:
            sleep_victims.extend(evicted_other)
            if evicted_other[0].role_name == PSRL_Role.Rollout:
                rollout_n -= len(evicted_other)
            else:
                rm_n -= len(evicted_other)
            if action.pre_sleep_other_preferred is None:
                action.pre_sleep_other_preferred = []
            action.pre_sleep_other_preferred.extend(self._signal_entry(v) for v in evicted_other)
            if rollout_n < self.min_awake_per_role or rm_n < self.min_awake_per_role:
                return None
        # Build wake/sleep id maps so the sum objective path (``_role_throughput_sum``)
        # resolves the exact hypothetical active set, matching the balanced path's
        # use of ``n_instances``. This keeps ``next_throughput`` on the same
        # objective-aware口径 as ``current_throughput`` (both via ``_system_throughput``).
        wake_ids_by_role: dict[PSRL_Role, set[int]] = {}
        sleep_ids_by_role: dict[PSRL_Role, set[int]] = {}
        if wakes:
            wake_ids_by_role[wakes[0].role_name] = {int(wake.instance_id) for wake in wakes}
        for victim in sleep_victims:
            sleep_ids_by_role.setdefault(victim.role_name, set()).add(int(victim.instance_id))
        next_throughput = self._system_throughput(
            grouped,
            rollout_n,
            rm_n,
            router_backlog_by_role,
            wake_ids_by_role=wake_ids_by_role,
            sleep_ids_by_role=sleep_ids_by_role,
        )
        delta_throughput = self._throughput_delta(next_throughput, current_throughput)
        phi = self._sleep_phi(sleep_victims)
        gain_l = self._windowed_throughput_gain(delta_throughput, rollout_n, rm_n) - phi
        return _ITLCandidate(action, rollout_n, rm_n, current_throughput, next_throughput, delta_throughput, phi, gain_l)

    def _enumerate_bottleneck_roles(
        self,
        *,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        role_throughputs: dict[PSRL_Role, float],
        rollout_signals: list[InstanceSignal],
        rm_signals: list[InstanceSignal],
        rollout_waiting: _RouterWaitingLoad,
        rm_waiting: _RouterWaitingLoad,
        w_rollout: float,
        w_rm: float,
        rollout_bottleneck: float,
        rm_bottleneck: float,
        router_backlog_by_role: dict[PSRL_Role, Any] | None,
    ) -> list[PSRL_Role]:
        """Pick which role(s) to vary when enumerating count candidates.

        Default behaviour is the pressure-weighted bottleneck comparison: the
        role with the smaller ``throughput / weight`` is the capacity bottleneck
        and is the side whose awake count gets scaled up/down. Subclasses (e.g.
        the harmonic variant under demand starvation) can override this to flip
        the bottleneck to the loaded role when the smaller-throughput side is
        idle because of missing demand rather than missing capacity.
        """
        _ = grouped, rollout_signals, rm_signals, rollout_waiting, rm_waiting, router_backlog_by_role
        if rm_bottleneck < rollout_bottleneck:
            return [PSRL_Role.RewardModel]
        if rm_bottleneck == rollout_bottleneck:
            return [PSRL_Role.Rollout, PSRL_Role.RewardModel]
        return [PSRL_Role.Rollout]

    def _objective_log_extras(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        router_backlog_by_role: dict[PSRL_Role, Any] | None,
    ) -> dict[str, Any]:
        """Extra key/value pairs to merge into the ``itl_objective`` log section.

        Overridden by subclasses to surface degraded-mode state (e.g. which role
        is demand-starved) without rewriting the surrounding ``decide`` logging.
        """
        _ = grouped, router_backlog_by_role
        return {}

    def _enumerate_candidates(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        router_backlog_by_role: dict[PSRL_Role, Any] | None = None,
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> list[_ITLCandidate]:
        """Build count-based candidates scored after post-action rebalancing."""
        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        current_rollout_n = sum(1 for s in rollout_signals if s.is_awaken)
        current_rm_n = sum(1 for s in rm_signals if s.is_awaken)
        _, rollout_victim_order = self._selection_orders_for_role(
            rollout_signals,
            trainer_waiting_hint,
        )
        _, rm_victim_order = self._selection_orders_for_role(
            rm_signals,
            trainer_waiting_hint,
        )
        victim_orders_by_role = {
            PSRL_Role.Rollout: rollout_victim_order,
            PSRL_Role.RewardModel: rm_victim_order,
        }
        role_throughputs = self._current_role_throughputs(
            grouped,
            current_rollout_n,
            current_rm_n,
            router_backlog_by_role,
        )
        # Use the same objective-aware path (``_system_throughput`` with no
        # wake/sleep) for the current baseline so that ``delta_throughput =
        # next_throughput - current_throughput`` is computed under one
        # consistent throughput口径 (sum vs balanced) and one consistent router
        # backlog口径, instead of mixing sum-current with balanced-next.
        backlog_map = router_backlog_by_role or {}
        current_throughput = self._system_throughput(
            grouped,
            current_rollout_n,
            current_rm_n,
            router_backlog_by_role,
        )
        candidates: list[_ITLCandidate] = []
        max_batch = max(1, int(self.max_scale_instances_per_action))
        # Pressure-weighted per-role bottleneck metric: tp / w. The side carrying
        # more router/instance pressure gets a larger divisor, shrinking its
        # throughput so it is more likely to be flagged as the bottleneck. When
        # weighting is disabled the divisors are uniform 0.5/0.5, which preserves
        # the legacy ``min(tp)``-based comparison up to a constant factor.
        # Bottleneck weights always reflect the full per-role pressure
        # (in-instance load + router backlog), independent of
        # ``current_state_include_router_waiting``. That keeps the divisor
        #口径 consistent with ``_system_throughput``'s weight, so the side
        # carrying more router backlog is still flagged as the bottleneck even
        # when current-state throughput estimates exclude router waiting.
        rollout_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.Rollout))
        rm_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel))
        w_rollout, w_rm = self._role_throughput_weights(
            rollout_signals, rm_signals, rollout_waiting, rm_waiting
        )
        rollout_bottleneck = (
            role_throughputs[PSRL_Role.Rollout] / w_rollout if w_rollout > 0.0 else role_throughputs[PSRL_Role.Rollout]
        )
        rm_bottleneck = (
            role_throughputs[PSRL_Role.RewardModel] / w_rm if w_rm > 0.0 else role_throughputs[PSRL_Role.RewardModel]
        )
        bottleneck_roles = self._enumerate_bottleneck_roles(
            grouped=grouped,
            role_throughputs=role_throughputs,
            rollout_signals=rollout_signals,
            rm_signals=rm_signals,
            rollout_waiting=rollout_waiting,
            rm_waiting=rm_waiting,
            w_rollout=w_rollout,
            w_rm=w_rm,
            rollout_bottleneck=rollout_bottleneck,
            rm_bottleneck=rm_bottleneck,
            router_backlog_by_role=router_backlog_by_role,
        )
        capacity = self._elastic_device_capacity(grouped, trainer_waiting_hint)
        rollout_min_n, rollout_max_n = self._role_count_bounds(
            rollout_signals,
            current_rollout_n,
            trainer_waiting_hint,
        )
        rm_min_n, rm_max_n = self._role_count_bounds(
            rm_signals,
            current_rm_n,
            trainer_waiting_hint,
        )
        wake_prefix_plans_by_role = {
            PSRL_Role.Rollout: self._wake_prefix_plans_for_role(
                target_signals=rollout_signals,
                other_signals=rm_signals,
                current_other_n=current_rm_n,
                max_batch=max_batch,
                trainer_waiting_hint=trainer_waiting_hint,
            ),
            PSRL_Role.RewardModel: self._wake_prefix_plans_for_role(
                target_signals=rm_signals,
                other_signals=rollout_signals,
                current_other_n=current_rollout_n,
                max_batch=max_batch,
                trainer_waiting_hint=trainer_waiting_hint,
            ),
        }
        seen_counts: set[tuple[int, int]] = set()

        for bottleneck_role in bottleneck_roles:
            if bottleneck_role == PSRL_Role.Rollout:
                min_target = max(rollout_min_n, current_rollout_n - max_batch)
                max_target = min(rollout_max_n, current_rollout_n + max_batch)
                target_counts = [n for n in range(min_target, max_target + 1) if n != current_rollout_n]
                for rollout_n in target_counts:
                    rm_n = current_rm_n
                    if rollout_n < current_rollout_n:
                        if rm_n < rm_min_n or rm_n > rm_max_n:
                            continue
                        if (
                            capacity > 0
                            and self._candidate_device_usage(rollout_signals, rm_signals, rollout_n, rm_n) > capacity
                        ):
                            continue
                    count_key = (rollout_n, rm_n)
                    if count_key in seen_counts:
                        continue
                    seen_counts.add(count_key)
                    candidate = self._build_rebalanced_count_candidate(
                        grouped=grouped,
                        rollout_n=rollout_n,
                        rm_n=rm_n,
                        current_rollout_n=current_rollout_n,
                        current_rm_n=current_rm_n,
                        current_throughput=current_throughput,
                        router_backlog_by_role=router_backlog_by_role,
                        trainer_waiting_hint=trainer_waiting_hint,
                        victim_orders_by_role=victim_orders_by_role,
                        wake_prefix_plans_by_role=wake_prefix_plans_by_role,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
            else:
                min_target = max(rm_min_n, current_rm_n - max_batch)
                max_target = min(rm_max_n, current_rm_n + max_batch)
                target_counts = [n for n in range(min_target, max_target + 1) if n != current_rm_n]
                for rm_n in target_counts:
                    rollout_n = current_rollout_n
                    if rm_n < current_rm_n:
                        if rollout_n < rollout_min_n or rollout_n > rollout_max_n:
                            continue
                        if (
                            capacity > 0
                            and self._candidate_device_usage(rollout_signals, rm_signals, rollout_n, rm_n) > capacity
                        ):
                            continue
                    count_key = (rollout_n, rm_n)
                    if count_key in seen_counts:
                        continue
                    seen_counts.add(count_key)
                    candidate = self._build_rebalanced_count_candidate(
                        grouped=grouped,
                        rollout_n=rollout_n,
                        rm_n=rm_n,
                        current_rollout_n=current_rollout_n,
                        current_rm_n=current_rm_n,
                        current_throughput=current_throughput,
                        router_backlog_by_role=router_backlog_by_role,
                        trainer_waiting_hint=trainer_waiting_hint,
                        victim_orders_by_role=victim_orders_by_role,
                        wake_prefix_plans_by_role=wake_prefix_plans_by_role,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
        return candidates

    @staticmethod
    def _format_throughput_value(value: float) -> str:
        """Format a throughput for logs, preserving the +inf (idle) bottleneck case."""
        if math.isinf(value):
            return "inf"
        return f"{value:.6f}"

    def _instance_throughput_from_load(
        self, signal: InstanceSignal, request_load: float, token_load: float
    ) -> float:
        """Estimate one instance's throughput from a hypothetical request/token load."""
        if request_load <= 0.0:
            return self._idle_instance_throughput()
        itl = compute_itl(self._params_for_signal(signal), token_load, request_load)
        return request_load / itl

    def _role_throughput_for_objective(
        self,
        role_signals: list[InstanceSignal],
        n_instances: int,
        router_waiting_load: _RouterWaitingLoad,
        wake_instance_ids: set[int] | None = None,
        sleep_instance_ids: set[int] | None = None,
    ) -> float:
        """Per-role throughput under the active objective (matches ``_system_throughput``)."""
        if self.throughput_objective == "sum":
            return self._role_throughput_sum(
                role_signals, n_instances, router_waiting_load, wake_instance_ids, sleep_instance_ids
            )
        return self._role_throughput(role_signals, n_instances, router_waiting_load)

    def _role_instance_load_rows(
        self,
        role_signals: list[InstanceSignal],
        n_instances: int,
        router_waiting_load: _RouterWaitingLoad,
        wake_instance_ids: set[int] | None = None,
        sleep_instance_ids: set[int] | None = None,
    ) -> list[tuple[int, float, float, float]]:
        """Per-instance ``(id, requests, tokens, throughput)`` rows under the active objective.

        The ``sum`` objective reports each active instance's own running load plus an
        even share of router backlog; the balanced objective reports the averaged
        per-instance load (identical across instances), matching how the role/system
        throughput is actually scored.
        """
        active = sorted(
            self._active_role_signals(role_signals, n_instances, wake_instance_ids, sleep_instance_ids),
            key=lambda item: int(item.instance_id),
        )
        if not active:
            return []
        rows: list[tuple[int, float, float, float]] = []
        if self.throughput_objective == "sum":
            waiting_requests_per_instance = router_waiting_load.request_count / float(len(active))
            waiting_tokens_per_instance = router_waiting_load.token_count / float(len(active))
            for signal in active:
                request_load = self._vllm_current_request_load(signal) + waiting_requests_per_instance
                token_load = float(signal.total_token_num) + waiting_tokens_per_instance
                rows.append(
                    (
                        int(signal.instance_id),
                        request_load,
                        token_load,
                        self._instance_throughput_from_load(signal, request_load, token_load),
                    )
                )
            return rows
        total_requests, total_tokens = self._role_total_load(role_signals, router_waiting_load)
        avg_requests = total_requests / float(n_instances) if n_instances > 0 else 0.0
        avg_tokens = total_tokens / float(n_instances) if n_instances > 0 else 0.0
        avg_throughput = (
            self._instance_throughput_from_load(active[0], avg_requests, avg_tokens)
            if avg_requests > 0.0
            else self._idle_instance_throughput()
        )
        for signal in active:
            rows.append((int(signal.instance_id), avg_requests, avg_tokens, avg_throughput))
        return rows

    @staticmethod
    def _format_instance_load_rows(rows: list[tuple[int, float, float, float]]) -> str:
        """Render per-instance load rows for the decision cycle log."""
        if not rows:
            return "[]"
        parts = [
            f"id{instance_id}:req={request_load:.1f},tok={token_load:.0f},"
            f"tp={ITLScalingPolicy._format_throughput_value(throughput)}"
            for instance_id, request_load, token_load, throughput in rows
        ]
        return "[" + "; ".join(parts) + "]"

    def _log_role_state(
        self,
        cycle_log: Any,
        label: str,
        role_signals: list[InstanceSignal],
        n_instances: int,
        load_waiting: _RouterWaitingLoad,
        router_backlog: _RouterWaitingLoad,
        wake_instance_ids: set[int] | None = None,
        sleep_instance_ids: set[int] | None = None,
    ) -> None:
        """Log one role's instance count, per-instance load, router backlog and role throughput.

        ``load_waiting`` drives the per-instance/role throughput estimates (subject to
        ``current_state_include_router_waiting`` for the current state), while
        ``router_backlog`` is always the real backlog reported for visibility.
        """
        rows = self._role_instance_load_rows(
            role_signals, n_instances, load_waiting, wake_instance_ids, sleep_instance_ids
        )
        total_requests, total_tokens = self._role_total_load(role_signals, load_waiting)
        role_throughput = self._role_throughput_for_objective(
            role_signals, n_instances, load_waiting, wake_instance_ids, sleep_instance_ids
        )
        cycle_log.kv(
            **{
                f"{label}_n": n_instances,
                f"{label}_total_req": f"{total_requests:.1f}",
                f"{label}_total_tok": f"{total_tokens:.0f}",
                f"{label}_role_tp": self._format_throughput_value(role_throughput),
                f"{label}_router_req": f"{router_backlog.request_count:.0f}",
                f"{label}_router_tok": f"{router_backlog.token_count:.0f}",
            }
        )
        cycle_log.note(f"{label}_instances={self._format_instance_load_rows(rows)}")

    @staticmethod
    def _candidate_wake_sleep_by_role(
        candidate: _ITLCandidate,
    ) -> tuple[dict[PSRL_Role, set[int]], dict[PSRL_Role, set[int]]]:
        """Derive per-role wake/sleep instance id sets from a candidate's action."""
        wake_by_role: dict[PSRL_Role, set[int]] = {}
        sleep_by_role: dict[PSRL_Role, set[int]] = {}
        action = candidate.action
        if action.action_type == "scale_up":
            if action.preferred_instance_ids:
                wake_by_role[action.role_name] = {int(i) for i in action.preferred_instance_ids}
            for entry in action.pre_sleep_other_preferred or []:
                sleep_by_role.setdefault(entry["role_name"], set()).add(int(entry["instance_id"]))
        elif action.preferred_instance_ids:
            sleep_by_role[action.role_name] = {int(i) for i in action.preferred_instance_ids}
        return wake_by_role, sleep_by_role

    def _log_candidate_state(
        self,
        cycle_log: Any,
        index: int,
        candidate: _ITLCandidate,
        rollout_signals: list[InstanceSignal],
        rm_signals: list[InstanceSignal],
        rollout_waiting: _RouterWaitingLoad,
        rm_waiting: _RouterWaitingLoad,
    ) -> None:
        """Log one candidate's action, estimated per-side load/throughput, Phi and L."""
        action = candidate.action
        wake_by_role, sleep_by_role = self._candidate_wake_sleep_by_role(candidate)
        cycle_log.note(
            f"candidate[{index}] {action.action_type} {action.role_name.name}/{action.model_name} "
            f"num_instances={action.num_instances} preferred={action.preferred_instance_ids} "
            f"pre_sleep={action.pre_sleep_other_preferred}"
        )
        cycle_log.kv(
            rollout_n=candidate.rollout_n,
            rm_n=candidate.rm_n,
            next_throughput=self._format_throughput_value(candidate.next_throughput),
            delta_throughput=f"{candidate.delta_throughput:.6f}",
            Phi=f"{candidate.phi:.6f}",
            L=f"{candidate.gain_l:.6f}",
        )
        self._log_role_state(
            cycle_log,
            "rollout_est",
            rollout_signals,
            candidate.rollout_n,
            rollout_waiting,
            rollout_waiting,
            wake_by_role.get(PSRL_Role.Rollout),
            sleep_by_role.get(PSRL_Role.Rollout),
        )
        self._log_role_state(
            cycle_log,
            "rm_est",
            rm_signals,
            candidate.rm_n,
            rm_waiting,
            rm_waiting,
            wake_by_role.get(PSRL_Role.RewardModel),
            sleep_by_role.get(PSRL_Role.RewardModel),
        )

    def decide(
        self,
        signals: list[InstanceSignal],
        execution_in_progress: bool = False,
        router_backlog_by_role: dict[PSRL_Role, Any] | None = None,
        trainer_waiting_hint: dict[str, Any] | None = None,
        pending_scale_up_by_role: dict[PSRL_Role, int] | None = None,
    ) -> ScalingDecision:
        """Pick the scale action with highest ``gain_l`` if it exceeds ``min_gain``."""
        _ = pending_scale_up_by_role
        cy = self._cycle_log
        cy.start()
        cy.section("inputs")
        cy.kv(n_signals=len(signals), execution_in_progress=execution_in_progress)
        if not self.enable:
            self._finish_cycle("skipped", "policy_disabled")
            return ScalingDecision(actions=[], reason="policy_disabled", estimated_lambda=0.0, role_to_total_mu={})
        if not signals:
            self._finish_cycle("skipped", "empty_signals")
            return ScalingDecision(actions=[], reason="empty_signals", estimated_lambda=0.0, role_to_total_mu={})
        if execution_in_progress:
            self._finish_cycle("skipped", "decision_execution_in_progress")
            return ScalingDecision(actions=[], reason="decision_execution_in_progress", estimated_lambda=0.0, role_to_total_mu={})
        grouped = self._group_by_role(signals)
        _, role_total_mu, _ = self._build_mu_maps(signals)
        estimated_lambda = self._estimate_lambda(signals, role_total_mu)
        now_ms = time.time() * 1000
        if now_ms - self.last_action_time_ms < self.cooldown_ms:
            self._finish_cycle("skipped", "cooldown")
            return ScalingDecision(actions=[], reason="cooldown", estimated_lambda=0.0, role_to_total_mu=role_total_mu)
        stale_count = sum(1 for signal in signals if self._is_signal_staled(signal))
        all_stale = stale_count == len(signals)
        backlog_positive = self._router_backlog_positive(router_backlog_by_role)
        if all_stale and not backlog_positive:
            self._finish_cycle("skipped", "all_signals_stale", stale_count=stale_count)
            return ScalingDecision(actions=[], reason="all_signals_stale", estimated_lambda=estimated_lambda, role_to_total_mu=role_total_mu)
        if all_stale and backlog_positive:
            cy.note("all instance snapshots are stale, but router backlog is positive; continuing for backlog-driven wake")

        candidates = self._enumerate_candidates(grouped, router_backlog_by_role, trainer_waiting_hint)
        candidates.sort(key=lambda item: item.gain_l, reverse=True)
        best = candidates[0] if candidates else None
        current_rollout_n = sum(1 for s in grouped.get(PSRL_Role.Rollout, []) if s.is_awaken)
        current_rm_n = sum(1 for s in grouped.get(PSRL_Role.RewardModel, []) if s.is_awaken)
        # Same objective-aware path used inside ``_enumerate_candidates`` so the
        # logged baseline matches the baseline used in candidate ``delta_throughput``.
        current_throughput = self._system_throughput(
            grouped,
            current_rollout_n,
            current_rm_n,
            router_backlog_by_role,
        )
        cy.section("itl_objective")
        cy.kv(
            current_throughput=f"{current_throughput:.8f}",
            candidates=len(candidates),
            decision_window_s=f"{self.decision_window_s:.3f}",
            migrate_time_s=f"{self.migrate_time_s:.3f}",
            min_gain=f"{self.min_gain:.8f}",
            throughput_objective=self.throughput_objective,
            vllm_current_queue_scope=self.vllm_current_queue_scope,
            max_scale_instances_per_action=self.max_scale_instances_per_action,
            router_waiting_top_t=self.router_waiting_top_t,
            current_state_include_router_waiting=self.current_state_include_router_waiting,
            role_throughput_weight_enable=self.role_throughput_weight_enable,
            role_throughput_weight_basis=self.role_throughput_weight_basis,
            role_throughput_weight_mode=self.role_throughput_weight_mode,
            **self._objective_log_extras(grouped, router_backlog_by_role),
        )

        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        backlog_map = router_backlog_by_role or {}
        # Real router backlog (always shown) vs the current-state load fed into
        # throughput (zeroed when current_state_include_router_waiting is off).
        rollout_backlog = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.Rollout))
        rm_backlog = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel))
        rollout_current_waiting = self._current_router_waiting_load(backlog_map.get(PSRL_Role.Rollout))
        rm_current_waiting = self._current_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel))
        cy.section("current_state")
        self._log_role_state(
            cy, "rollout", rollout_signals, current_rollout_n, rollout_current_waiting, rollout_backlog
        )
        self._log_role_state(cy, "rm", rm_signals, current_rm_n, rm_current_waiting, rm_backlog)
        cy.kv(system_throughput=self._format_throughput_value(current_throughput))
        cy.section("candidates")
        for index, candidate in enumerate(candidates):
            self._log_candidate_state(
                cy, index, candidate, rollout_signals, rm_signals, rollout_backlog, rm_backlog
            )

        cy.section("itl_decision")
        if best is None:
            self._finish_cycle("no_action", "itl_no_feasible_candidate", current_throughput=f"{current_throughput:.8f}")
            return ScalingDecision(actions=[], reason="itl_no_feasible_candidate", estimated_lambda=estimated_lambda, role_to_total_mu=role_total_mu)
        cy.kv(
            best_action=best.action.action_type,
            best_role=best.action.role_name.name,
            best_model=best.action.model_name,
            best_num_instances=best.action.num_instances,
            preferred=best.action.preferred_instance_ids,
            pre_sleep=best.action.pre_sleep_other_preferred,
            next_rollout_n=best.rollout_n,
            next_rm_n=best.rm_n,
            next_throughput=f"{best.next_throughput:.8f}",
            delta_throughput=f"{best.delta_throughput:.8f}",
            Phi=f"{best.phi:.8f}",
            L=f"{best.gain_l:.8f}",
        )
        if best.gain_l <= self.min_gain:
            self._finish_cycle(
                "no_action",
                "itl_gain_below_min_gain",
                best_L=f"{best.gain_l:.8f}",
                min_gain=f"{self.min_gain:.8f}",
            )
            return ScalingDecision(actions=[], reason="itl_gain_below_min_gain", estimated_lambda=estimated_lambda, role_to_total_mu=role_total_mu)
        self.last_action_time_ms = now_ms
        reason = f"itl_best_{best.action.action_type}_{best.action.role_name.name}"
        best.action.reason = reason
        self._finish_cycle(
            "action",
            reason,
            current_throughput=f"{best.current_throughput:.8f}",
            next_throughput=f"{best.next_throughput:.8f}",
            delta_throughput=f"{best.delta_throughput:.8f}",
            Phi=f"{best.phi:.8f}",
            L=f"{best.gain_l:.8f}",
        )
        return ScalingDecision(
            actions=[best.action],
            reason=reason,
            estimated_lambda=estimated_lambda,
            role_to_total_mu=role_total_mu,
        )
