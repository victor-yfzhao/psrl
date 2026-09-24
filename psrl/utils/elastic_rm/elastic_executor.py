import asyncio
import enum
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

import ray

from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.elastic_rm.diagnostics import log_elastic_rm_backlog_diag
from psrl.utils.elastic_rm.dummy_scaling_policy import DummyScalingPolicy
from psrl.utils.elastic_rm.itl_harmonic_scaling_policy import ITLHarmonicScalingPolicy
from psrl.utils.elastic_rm.itl_scaling_policy import ITLScalingPolicy
from psrl.utils.elastic_rm.rule_based_scaling_policy import RuleBasedScalingPolicy
from psrl.utils.elastic_rm.scaling_policy import InstanceSignal, ScalingPolicy
from psrl.utils.logger import DualOutputHandler, FileOnlyHandler
from psrl.utils.server.command import Command, CommandType

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

monitor_logger = logging.getLogger("ElasticMonitor")
monitor_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Sentinel: use config default for coordinator Ray RPC timeout (see _await_elastic_coordinator_command).
_ELASTIC_COORD_CMD_TIMEOUT_UNSET = object()
_PLANNER_PHASE_KEYS = (
    "state_analysis_s",
    "candidate_ordering_s",
    "candidate_set_construction_s",
    "simulation_input_preparation_s",
    "candidate_evaluation_wall_s",
    "candidate_scoring_s",
    "best_candidate_selection_s",
)
_PLANNER_DIAGNOSTIC_KEYS = (
    "rebalance_simulation_s",
    "router_simulation_s",
    "simulation_wall_s",
    "rebalance_router_overlap_s",
)
_STEP_SCALING_COUNT_KEYS = (
    "actual_actions",
    "scale_up_actions",
    "scale_down_actions",
    "sleep_instances",
    "wakeup_instances",
    "instance_transitions",
)


def _normalized_planner_breakdown(policy, planner_s: float) -> dict[str, float]:
    """Make policy phase attribution add up to the outer planner wall time."""
    total_s = max(0.0, float(planner_s))
    raw = getattr(policy, "last_planner_breakdown", {})
    if not isinstance(raw, dict):
        raw = {}
    breakdown = {key: max(0.0, float(raw.get(key, 0.0))) for key in _PLANNER_PHASE_KEYS}
    attributed_s = sum(breakdown.values())
    if attributed_s > total_s and attributed_s > 0.0:
        scale = total_s / attributed_s
        breakdown = {key: value * scale for key, value in breakdown.items()}
        attributed_s = total_s
    breakdown["other_s"] = max(0.0, total_s - attributed_s)
    breakdown.update({key: max(0.0, float(raw.get(key, 0.0))) for key in _PLANNER_DIAGNOSTIC_KEYS})
    return breakdown


def _rebalance_after_scale_up_enabled(policy) -> bool:
    """Keep established post-wake behavior unless a policy explicitly opts out."""
    return bool(getattr(policy, "rebalance_after_scale_up", True))


def _preemptive_scale_up_enabled(policy) -> bool:
    """Keep established pre-sleep behavior unless a policy explicitly opts out."""
    return bool(getattr(policy, "allow_preemptive_scale_up", True))


class InstanceStatus(Enum):
    ASLEEP = enum.auto()
    AWAKEN = enum.auto()
    TRAINING = enum.auto()
    RECOVERING = enum.auto()


@dataclass(frozen=True)
class _PostScaleUpRequestCandidate:
    """A request candidate considered during post-scale-up rebalancing."""

    instance_id: int
    request_id: str
    seq_len: int
    is_waiting: bool


@dataclass(frozen=True)
class _PostScaleUpDonorPlan:
    """A pre-sorted donor instance plan for post-scale-up rebalancing."""

    instance_id: int
    request_count: float
    token_sum: float
    score: float
    over_request_ratio: float
    over_length_ratio: float
    candidates: list[_PostScaleUpRequestCandidate]


@ray.remote
class ElasticExecutor:
    def __init__(
        self,
        config,
        roles: list[tuple[PSRL_Role, str]],
        coordinators: dict[PSRL_Role, dict[str, ray.actor.ActorHandle]],
        agent_loop_manager: ray.actor.ActorHandle | None = None,
        elastic_rm_config: dict | None = None,
        train_pool_available: bool = False,
    ):
        self.coordinators = coordinators
        self.roles = roles
        self.elastic_rm_config = elastic_rm_config or {}
        self.config = config
        self.agent_loop_manager = agent_loop_manager
        # Whether train_pool GPUs are available for elastic rollout/RM replicas.
        # True when the trainer actor is NIXL-slept (train_pool lent out); False
        # while a training step owns train_pool. Flipped by enter/leave_training_pool
        # and seeded at construction with the trainer's initial sleep state. Drives
        # the ``trainer_busy`` flag reported to the scaling policy.
        self._train_pool_available = bool(train_pool_available)

        self.instances_status_flags: dict[PSRL_Role, dict[str, dict[int, InstanceStatus]]] = {}
        self.instances_engine_stats: dict[PSRL_Role, dict[str, dict[int, dict | None]]] = {}
        # Per-instance pool-local bundle_range (start, end); used for cross-role conflict checks.
        self.instance_bundle_mappings: dict[PSRL_Role, dict[str, dict[int, dict[str, object]]]] = {}
        self.bundle_to_instances: dict[tuple[str, int], set[tuple[PSRL_Role, str, int]]] = {}

        for role_name, model_name in self.roles:
            self.instances_status_flags.setdefault(role_name, {}).setdefault(model_name, {})
            self.instances_engine_stats.setdefault(role_name, {}).setdefault(model_name, {})
            self.instance_bundle_mappings.setdefault(role_name, {}).setdefault(model_name, {})

        self.scale_up_task_queue: asyncio.Queue = asyncio.Queue()
        self.scale_down_task_queue: asyncio.Queue = asyncio.Queue()

        self.running_loop = None
        self.monitor_task = None
        self.scale_up_task = None
        self.scale_down_task = None
        self.interrupt_vllm_waiting_task = None

        self.stop_monitor = False
        self.stop_scale_up = False
        self.stop_scale_down = False
        self.stop_interrupt_vllm_waiting = False
        policy_variant = str(self.elastic_rm_config.get("scaling_policy_variant", "normal")).lower()
        policy_cls_by_variant = {
            "normal": ScalingPolicy,
            "dummy": DummyScalingPolicy,
            "itl": ITLScalingPolicy,
            "itl_harmonic": ITLHarmonicScalingPolicy,
            "rule_based": RuleBasedScalingPolicy,
        }
        policy_cls = policy_cls_by_variant.get(policy_variant, ScalingPolicy)
        self.scaling_policy = policy_cls(config=self.config, policy_config=self.elastic_rm_config)

        # When set, elastic scaling policy may accept new decisions. Cleared while a decision
        # is executing (see _decision_pending_action_counts) or during perform_sync_swap;
        # await .wait() before swap to serialize with scaling handlers. Stall/abandon only
        # applies when idle is cleared *and* there are pending decision actions (swap uses
        # the same event but leaves pending empty).
        self._policy_scaling_idle = asyncio.Event()
        self._policy_scaling_idle.set()
        self._policy_scaling_holder: str | None = None
        self._policy_scaling_holder_since_s: float | None = None
        self._policy_scaling_owner_token: int | None = None
        self._next_policy_scaling_owner_token = 1
        self._policy_scaling_waiters: list[asyncio.Future] = []
        self._next_decision_id = 1
        self._next_migration_sequence = 1
        self._decision_pending_action_counts: dict[int, int] = {}
        self._decision_gate_tokens: dict[int, int] = {}
        self._current_training_step = -1
        self._step_scaling_action_counts: dict[int, dict[str, int]] = {}
        # Consecutive monitor ticks where policy is blocked by an unfinished scale decision.
        self._execution_in_progress_stall_ticks: int = 0
        # When stall ticks reach this threshold, clear in-flight decision state so policy can proceed.
        # 0 = disabled (only warnings, no abandon).
        self._decision_abandon_stall_ticks: int = int(
            self.elastic_rm_config.get("decision_execution_abandon_stall_ticks", 0)
        )

        self._last_monitor_instance_log_ms: float = 0.0
        self._monitor_instance_log_interval_ms: int = int(
            self.elastic_rm_config.get("monitor_instance_log_interval_ms", 5000)
        )
        self._enable_monitor_instance_log: bool = bool(self.elastic_rm_config.get("enable_monitor_instance_log", True))
        self.router_backlog_by_role: dict[PSRL_Role, int] = {}
        self.router_backlog_summary_by_role: dict[PSRL_Role, dict[str, int]] = {}
        self.request_level_snapshots_by_role: dict[PSRL_Role, object] = {}
        self.trainer_waiting_hint: dict[str, object] = {
            "trainer_busy": not self._train_pool_available,
            "waiting_buffer_id": None,
            "waiting_on": "none",
            "breakdown": {},
        }

        self._post_scale_up_rebalance_request_weight = max(
            0.0,
            float(self.elastic_rm_config.get("post_scale_up_rebalance_request_weight", 1.0)),
        )
        self._post_scale_up_rebalance_length_weight = max(
            0.0,
            float(self.elastic_rm_config.get("post_scale_up_rebalance_length_weight", 1.0)),
        )
        if self._post_scale_up_rebalance_request_weight <= 0.0 and self._post_scale_up_rebalance_length_weight <= 0.0:
            self._post_scale_up_rebalance_request_weight = 1.0
        self._interrupt_vllm_waiting_enabled = bool(
            self.elastic_rm_config.get("interrupt_vllm_waiting_when_running_only", False)
        )
        self._interrupt_vllm_waiting_interval_s = max(
            0.01,
            float(self.elastic_rm_config.get("interrupt_vllm_waiting_interval_s", 2.0)),
        )
        _rt_waiting_ratio = float(self.elastic_rm_config.get("interrupt_vllm_waiting_ratio", 1.0))
        self._interrupt_vllm_waiting_ratio = max(0.0, min(1.0, _rt_waiting_ratio))

        # Wake-up immunity: after an instance wakes up (or completes an in-place sync), it is
        # protected from being scaled down for this many milliseconds.  Set to 0 to disable.
        self._wakeup_immunity_ms: int = int(self.elastic_rm_config.get("wakeup_immunity_ms", 30000))
        # Maps (role_name, model_name, instance_id) → monotonic timestamp (ms) until which the
        # instance is immune from scale-down.
        self._instance_immunity_until_ms: dict[tuple, float] = {}
        # Per-tick timeout for coordinator Ray RPCs; avoids monitor loop hanging forever when coordinators stall.
        self._coordinator_sync_timeout_s = float(self.elastic_rm_config.get("coordinator_sync_timeout_s", 60.0))
        # Optional timeout for SLEEP/WAKE_UP/ABORT issued during elastic scale (None = wait forever).
        _cmd_to = self.elastic_rm_config.get("coordinator_command_timeout_s", None)
        if _cmd_to is None or float(_cmd_to) <= 0:
            self._coordinator_command_timeout_s: float | None = None
        else:
            self._coordinator_command_timeout_s = float(_cmd_to)

        # Logger
        self.log_prefix = "ElasticExecutor"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        monitor_logger.propagate = False
        monitor_logger.addHandler(FileOnlyHandler(self.config.psrl.logging_path, "ElasticMonitor"))

    def register_instances(self, role_name: PSRL_Role, model_name: str, num_instances: int):
        self.instances_status_flags.setdefault(role_name, {}).setdefault(model_name, {})
        self.instances_engine_stats.setdefault(role_name, {}).setdefault(model_name, {})
        self.instance_bundle_mappings.setdefault(role_name, {}).setdefault(model_name, {})
        for instance_id in range(num_instances):
            self.instances_status_flags[role_name][model_name][instance_id] = InstanceStatus.ASLEEP
            self.instances_engine_stats[role_name][model_name][instance_id] = None
            self.instance_bundle_mappings[role_name][model_name].setdefault(
                instance_id,
                {"bundle_range": None, "pool_id": "shared_rollout_pool"},
            )

    def register_instance_bundle_mapping(
        self,
        role_name: PSRL_Role,
        model_name: str,
        instance_id: int,
        bundle_range: tuple[int, int] | None,
        pool_id: str | None = None,
    ):
        self.instance_bundle_mappings.setdefault(role_name, {}).setdefault(model_name, {})
        self._remove_instance_from_bundle_reverse_index(role_name, model_name, instance_id)
        self.instance_bundle_mappings[role_name][model_name][instance_id] = {
            "bundle_range": (int(bundle_range[0]), int(bundle_range[1])) if bundle_range is not None else None,
            "pool_id": str(pool_id or "shared_rollout_pool"),
        }
        self._add_instance_to_bundle_reverse_index(role_name, model_name, instance_id)

    def initialize_instance_states(
        self,
        awaken_instances: list[dict] | None = None,
    ):
        awaken_set = set()
        for item in awaken_instances or []:
            awaken_set.add((item["role_name"], item["model_name"], int(item["instance_id"])))

        for role_name, role_data in self.instances_status_flags.items():
            for model_name, instance_status in role_data.items():
                for instance_id in list(instance_status.keys()):
                    key = (role_name, model_name, int(instance_id))
                    if key in awaken_set:
                        self.instances_status_flags[role_name][model_name][instance_id] = InstanceStatus.AWAKEN
                    else:
                        self.instances_status_flags[role_name][model_name][instance_id] = InstanceStatus.ASLEEP

    def initialize_runtime(
        self,
        registrations: list[dict] | None = None,
        bundle_mappings: list[dict] | None = None,
        awaken_instances: list[dict] | None = None,
    ):
        for item in registrations or []:
            self.register_instances(
                role_name=item["role_name"],
                model_name=item["model_name"],
                num_instances=int(item["num_instances"]),
            )
        for item in bundle_mappings or []:
            br = item.get("bundle_range")
            if br is not None and len(br) == 2:
                bundle_range = (int(br[0]), int(br[1]))
            else:
                bundle_range = None
            self.register_instance_bundle_mapping(
                role_name=item["role_name"],
                model_name=item["model_name"],
                instance_id=int(item["instance_id"]),
                bundle_range=bundle_range,
                pool_id=item.get("pool_id"),
            )
        self.initialize_instance_states(awaken_instances=awaken_instances)

    # ------------------------------------------------------------------
    # Wake-up immunity helpers
    # ------------------------------------------------------------------

    def _set_instance_immunity(self, role_name: PSRL_Role, model_name: str, instance_id: int) -> None:
        """Grant the instance an immunity window starting now."""
        if self._wakeup_immunity_ms <= 0:
            return
        key = self._instance_key(role_name, model_name, instance_id)
        now_ms = time.monotonic() * 1000
        self._instance_immunity_until_ms[key] = now_ms + self._wakeup_immunity_ms
        psrl_logger.info(
            "elastic_rm: set wake-up immunity for role=%s model=%s instance=%d, expires in %.0f ms.",
            getattr(role_name, "name", role_name),
            model_name,
            instance_id,
            self._wakeup_immunity_ms,
        )

    def _is_instance_in_immunity(self, role_name: PSRL_Role, model_name: str, instance_id: int) -> bool:
        """Return True if the instance is currently within its wake-up immunity window."""
        if self._wakeup_immunity_ms <= 0:
            return False
        key = self._instance_key(role_name, model_name, instance_id)
        until_ms = self._instance_immunity_until_ms.get(key, 0.0)
        return time.monotonic() * 1000 < until_ms

    def set_instance_immunity(self, role_name: PSRL_Role, model_name: str, instance_id: int) -> None:
        """Public entry point so external actors (e.g. RolloutCoordinator) can grant immunity
        after an in-place model sync completes."""
        self._set_instance_immunity(role_name, model_name, instance_id)

    def get_free_instances(
        self,
        role_name: PSRL_Role,
        model_name: str,
        exclude_instance_ids: list[int] | None = None,
    ) -> list[int]:
        """Return asleep instance IDs that have no bundle overlap conflict with other awake roles
        and are not currently in an immunity window.  Used by coordinator to find a
        candidate for swap-based sync."""
        exclude_ids = {int(i) for i in (exclude_instance_ids or [])}
        role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        free_ids: list[int] = []
        for instance_id, status in role_status.items():
            if status != InstanceStatus.ASLEEP:
                continue
            iid = int(instance_id)
            if iid in exclude_ids:
                continue
            if self._has_other_role_awaken_on_shared_bundle(role_name, model_name, iid):
                continue
            free_ids.append(iid)
        return free_ids

    def _policy_scaling_holder_elapsed_s(self) -> float | None:
        if self._policy_scaling_holder_since_s is None:
            return None
        return time.monotonic() - self._policy_scaling_holder_since_s

    def _policy_scaling_wait_context(self) -> dict:
        elapsed_s = self._policy_scaling_holder_elapsed_s()
        return {
            "holder": self._policy_scaling_holder,
            "holder_elapsed_s": None if elapsed_s is None else round(elapsed_s, 3),
            "pending_decision_action_counts": dict(self._decision_pending_action_counts),
        }

    async def _wait_for_policy_scaling_idle(self, waiter: str) -> None:
        while self._policy_scaling_owner_token is not None:
            psrl_logger.info(
                "elastic_rm policy_scaling_idle wait begin: waiter=%s waiting_for=%s",
                waiter,
                self._policy_scaling_wait_context(),
            )
            t0 = time.monotonic()
            future = asyncio.get_running_loop().create_future()
            self._policy_scaling_waiters.append(future)
            await future
            psrl_logger.info(
                "elastic_rm policy_scaling_idle wait end: waiter=%s waited_s=%.3f",
                waiter,
                time.monotonic() - t0,
            )
        return

    def _acquire_policy_scaling_idle(self, holder: str) -> int:
        if self._policy_scaling_owner_token is not None:
            raise RuntimeError(
                "policy scaling gate is already owned by "
                f"{self._policy_scaling_holder} token={self._policy_scaling_owner_token}"
            )
        token = self._next_policy_scaling_owner_token
        self._next_policy_scaling_owner_token += 1
        self._policy_scaling_owner_token = token
        self._policy_scaling_idle.clear()
        self._policy_scaling_holder = holder
        self._policy_scaling_holder_since_s = time.monotonic()
        return token

    async def _wait_and_acquire_policy_scaling_idle(self, holder: str) -> int:
        await self._wait_for_policy_scaling_idle(holder)
        return self._acquire_policy_scaling_idle(holder)

    def _release_policy_scaling_idle_if_clear(self, owner_token: int) -> None:
        """Set idle event when no in-flight decision actions remain."""
        if self._decision_pending_action_counts:
            return
        if owner_token != self._policy_scaling_owner_token:
            psrl_logger.error(
                "Refuse to release policy scaling gate with stale token=%s; current_token=%s holder=%s",
                owner_token,
                self._policy_scaling_owner_token,
                self._policy_scaling_holder,
            )
            return
        holder = self._policy_scaling_holder
        elapsed_s = self._policy_scaling_holder_elapsed_s()
        if holder is not None:
            psrl_logger.info(
                "elastic_rm policy_scaling_idle release: holder=%s held_s=%.3f",
                holder,
                0.0 if elapsed_s is None else elapsed_s,
            )
        self._policy_scaling_holder = None
        self._policy_scaling_holder_since_s = None
        self._policy_scaling_owner_token = None
        self._policy_scaling_idle.set()
        while self._policy_scaling_waiters:
            waiter = self._policy_scaling_waiters.pop(0)
            if not waiter.done():
                waiter.set_result(None)
                break

    def _check_sync_swap_legality(
        self,
        role_name: PSRL_Role,
        model_name: str,
        wake_instance_id: int,
        sleep_instance_id: int,
    ) -> tuple[bool, str]:
        """Fast legality check for swap intent.

        Returns:
            (is_legal, reason). reason is set when illegal.
        """
        role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        wake_status = role_status.get(int(wake_instance_id))
        sleep_status = role_status.get(int(sleep_instance_id))

        # Illegal wake: target must be ASLEEP.
        if wake_status != InstanceStatus.ASLEEP:
            return (
                False,
                (
                    f"illegal_wake_status wake={int(wake_instance_id)} "
                    f"status={getattr(wake_status, 'name', wake_status)} expected=ASLEEP"
                ),
            )
        # Illegal wake: target bundles must not overlap other awake role.
        if self._has_other_role_awaken_on_shared_bundle(role_name, model_name, int(wake_instance_id)):
            return (
                False,
                f"illegal_wake_bundle_conflict wake={int(wake_instance_id)} other_role_awaken_on_shared_bundle",
            )

        # Illegal sleep: target must be AWAKEN.
        if sleep_status != InstanceStatus.AWAKEN:
            return (
                False,
                (
                    f"illegal_sleep_status sleep={int(sleep_instance_id)} "
                    f"status={getattr(sleep_status, 'name', sleep_status)} expected=AWAKEN"
                ),
            )

        return True, ""

    async def enter_inplace_sync_policy_gate(
        self,
        role_name: PSRL_Role,
        model_name: str,
        instance_ids: list[int],
    ) -> int:
        """Serialize policy decisions while coordinator executes in-place sync."""
        owner_token = await self._wait_and_acquire_policy_scaling_idle(
            "enter_inplace_sync_policy_gate "
            f"role={getattr(role_name, 'name', role_name)} model={model_name} "
            f"instances={[int(i) for i in instance_ids]}"
        )
        psrl_logger.info(
            "elastic_rm enter_inplace_sync_policy_gate: role=%s model=%s instances=%s",
            getattr(role_name, "name", role_name),
            model_name,
            [int(i) for i in instance_ids],
        )
        return owner_token

    def leave_inplace_sync_policy_gate(
        self,
        role_name: PSRL_Role,
        model_name: str,
        instance_ids: list[int],
        owner_token: int,
    ) -> None:
        """Release policy gate after coordinator finishes in-place sync."""
        self._release_policy_scaling_idle_if_clear(int(owner_token))
        psrl_logger.info(
            "elastic_rm leave_inplace_sync_policy_gate: role=%s model=%s instances=%s",
            getattr(role_name, "name", role_name),
            model_name,
            [int(i) for i in instance_ids],
        )

    async def perform_sync_swap(
        self,
        role_name: PSRL_Role,
        model_name: str,
        wake_instance_id: int,
        sleep_instance_id: int,
        sync_operation_id: int,
        policy_gate_token: int | None = None,
    ) -> bool:
        return await self.perform_sync_swaps(
            role_name,
            model_name,
            [(int(wake_instance_id), int(sleep_instance_id))],
            sync_operation_id,
            policy_gate_token,
        )

    async def perform_sync_swaps(
        self,
        role_name: PSRL_Role,
        model_name: str,
        swap_pairs: list[tuple[int, int]],
        sync_operation_id: int,
        policy_gate_token: int | None = None,
    ) -> bool:
        """Execute a whole sync-swap batch under one policy transition lease."""
        owns_policy_gate = policy_gate_token is None
        if owns_policy_gate:
            owner_token = await self._wait_and_acquire_policy_scaling_idle(
                "perform_sync_swaps "
                f"role={getattr(role_name, 'name', role_name)} model={model_name} "
                f"pairs={[(int(wake), int(sleep)) for wake, sleep in swap_pairs]}"
            )
        else:
            owner_token = int(policy_gate_token)
            if owner_token != self._policy_scaling_owner_token:
                psrl_logger.error(
                    "Refuse sync swap with stale policy gate token=%s current_token=%s",
                    owner_token,
                    self._policy_scaling_owner_token,
                )
                return False
        try:
            for wake_instance_id, sleep_instance_id in swap_pairs:
                legal, reason = self._check_sync_swap_legality(
                    role_name=role_name,
                    model_name=model_name,
                    wake_instance_id=wake_instance_id,
                    sleep_instance_id=sleep_instance_id,
                )
                if not legal:
                    psrl_logger.warning(
                        "elastic_rm sync swap rejected: role=%s model=%s wake=%d sleep=%d reason=%s",
                        getattr(role_name, "name", role_name),
                        model_name,
                        int(wake_instance_id),
                        int(sleep_instance_id),
                        reason,
                    )
                    return False
                wake_ok = await self._scale_up_instance(
                    {
                        "role_name": role_name,
                        "model_name": model_name,
                        "instance_id": wake_instance_id,
                        "sync_operation_id": sync_operation_id,
                    }
                )
                if not wake_ok:
                    return False
                sleep_ok = await self._scale_down_instance(
                    {
                        "role_name": role_name,
                        "model_name": model_name,
                        "instance_id": sleep_instance_id,
                        "sync_operation_id": sync_operation_id,
                    }
                )
                if not sleep_ok:
                    psrl_logger.warning(
                        "sync swap kept both instances available because sleeping old instance %d failed",
                        sleep_instance_id,
                    )
                    return False
            return True
        finally:
            if owns_policy_gate:
                self._release_policy_scaling_idle_if_clear(owner_token)

    def snapshot(self) -> dict:
        return {
            "status": self.instances_status_flags,
            "bundle_mappings": self.instance_bundle_mappings,
            "bundle_to_instances": self.bundle_to_instances,
        }

    def get_awake_instance_counts(self) -> dict[str, int]:
        """Return the live count of AWAKEN instances per role.

        Keyed by role name string (e.g. ``Rollout`` / ``RewardModel``) so the result
        is plain-JSON for cross-process transport to the trainer driver, which pushes
        it to wandb. TRAINING and ASLEEP instances are not counted as awake, matching
        the awake summary in ``_maybe_log_instance_signals``.
        """
        counts: dict[str, int] = {}
        for role_name, role_data in self.instances_status_flags.items():
            role_key = getattr(role_name, "name", str(role_name))
            awake = sum(
                1
                for instance_status in role_data.values()
                for status in instance_status.values()
                if status == InstanceStatus.AWAKEN
            )
            counts[role_key] = counts.get(role_key, 0) + awake
        return counts

    def set_current_training_step(self, step: int) -> None:
        """Tag subsequently accepted policy actions with the trainer step."""
        self._current_training_step = int(step)

    def get_step_scaling_action_counts(self, step: int) -> dict[str, int]:
        """Return completed policy actions and instance transitions for one step."""
        step_id = int(step)
        counts = self._step_scaling_action_counts.get(step_id, {})
        return {
            "step": step_id,
            **{key: int(counts.get(key, 0)) for key in _STEP_SCALING_COUNT_KEYS},
        }

    def _snapshot_instance_statuses(
        self,
        instances: list[dict],
    ) -> dict[tuple[PSRL_Role, str, int], InstanceStatus | None]:
        snapshot: dict[tuple[PSRL_Role, str, int], InstanceStatus | None] = {}
        for instance in instances:
            role_name = instance["role_name"]
            model_name = instance["model_name"]
            instance_id = int(instance["instance_id"])
            key = self._instance_key(role_name, model_name, instance_id)
            snapshot[key] = self.instances_status_flags.get(role_name, {}).get(model_name, {}).get(instance_id)
        return snapshot

    def _count_instance_status_transitions(
        self,
        before: dict[tuple[PSRL_Role, str, int], InstanceStatus | None],
        *,
        from_status: InstanceStatus,
        to_status: InstanceStatus,
        instances: list[dict] | None = None,
    ) -> int:
        selected_keys = None
        if instances is not None:
            selected_keys = {
                self._instance_key(
                    instance["role_name"],
                    instance["model_name"],
                    int(instance["instance_id"]),
                )
                for instance in instances
            }
        transitions = 0
        for key, previous_status in before.items():
            if selected_keys is not None and key not in selected_keys:
                continue
            role_name, model_name, instance_id = key
            current_status = self.instances_status_flags.get(role_name, {}).get(model_name, {}).get(instance_id)
            if previous_status == from_status and current_status == to_status:
                transitions += 1
        return transitions

    def _record_step_scaling_action(
        self,
        *,
        step: int,
        decision_id: int | None,
        action_type: str,
        succeeded: bool,
        sleep_instances: int,
        wakeup_instances: int,
    ) -> None:
        step_id = int(step)
        counts_by_step = getattr(self, "_step_scaling_action_counts", None)
        if counts_by_step is None:
            counts_by_step = {}
            self._step_scaling_action_counts = counts_by_step
        counts = counts_by_step.setdefault(
            step_id,
            {key: 0 for key in _STEP_SCALING_COUNT_KEYS},
        )
        actual_actions = int(bool(succeeded))
        scale_up_actions = actual_actions if action_type == "scale_up" else 0
        scale_down_actions = actual_actions if action_type == "scale_down" else 0
        sleep_count = max(0, int(sleep_instances))
        wakeup_count = max(0, int(wakeup_instances))
        transition_count = sleep_count + wakeup_count
        increments = {
            "actual_actions": actual_actions,
            "scale_up_actions": scale_up_actions,
            "scale_down_actions": scale_down_actions,
            "sleep_instances": sleep_count,
            "wakeup_instances": wakeup_count,
            "instance_transitions": transition_count,
        }
        for key, value in increments.items():
            counts[key] += value
        psrl_logger.info(
            "[ELASTIC_OVERHEAD] operation=scaling_action step=%d decision_id=%s "
            "action_type=%s success=%s actual_actions=%d scale_up_actions=%d "
            "scale_down_actions=%d sleep_instances=%d wakeup_instances=%d "
            "instance_transitions=%d step_actual_actions_total=%d "
            "step_scale_up_actions=%d step_scale_down_actions=%d "
            "step_sleep_instances=%d step_wakeup_instances=%d "
            "step_instance_transitions=%d",
            step_id,
            decision_id,
            action_type,
            bool(succeeded),
            actual_actions,
            scale_up_actions,
            scale_down_actions,
            sleep_count,
            wakeup_count,
            transition_count,
            counts["actual_actions"],
            counts["scale_up_actions"],
            counts["scale_down_actions"],
            counts["sleep_instances"],
            counts["wakeup_instances"],
            counts["instance_transitions"],
        )

    @staticmethod
    def _instance_key(role_name: PSRL_Role, model_name: str, instance_id: int) -> tuple[PSRL_Role, str, int]:
        return (role_name, model_name, int(instance_id))

    def _normalize_instance_entries(self, instances: list[dict] | None) -> list[dict]:
        normalized: list[dict] = []
        seen: set[tuple[PSRL_Role, str, int]] = set()
        for item in instances or []:
            if not isinstance(item, dict):
                continue
            role_name = item.get("role_name")
            model_name = item.get("model_name")
            if role_name is None or model_name is None:
                continue
            try:
                instance_id = int(item["instance_id"])
            except (KeyError, TypeError, ValueError):
                continue
            key = self._instance_key(role_name, model_name, instance_id)
            if key in seen:
                continue
            if instance_id not in self.instances_status_flags.get(role_name, {}).get(model_name, {}):
                psrl_logger.warning(
                    "Skip unknown elastic instance entry: role=%s model=%s instance=%s.",
                    role_name,
                    model_name,
                    instance_id,
                )
                continue
            normalized.append({"role_name": role_name, "model_name": model_name, "instance_id": instance_id})
            seen.add(key)
        return normalized

    def _get_instance_pool_id(self, role_name: PSRL_Role, model_name: str, instance_id: int) -> str:
        mapping = self.instance_bundle_mappings.get(role_name, {}).get(model_name, {}).get(instance_id, {})
        return str(mapping.get("pool_id") or "shared_rollout_pool")

    def _get_instance_bundle_indices(self, role_name: PSRL_Role, model_name: str, instance_id: int) -> list[int]:
        mapping = self.instance_bundle_mappings.get(role_name, {}).get(model_name, {}).get(instance_id, {})
        br = mapping.get("bundle_range")
        if br is None:
            return []
        start, end = int(br[0]), int(br[1])
        if start >= end:
            return []
        return list(range(start, end))

    def _remove_instance_from_bundle_reverse_index(self, role_name: PSRL_Role, model_name: str, instance_id: int):
        target_key = self._instance_key(role_name, model_name, instance_id)
        pool_id = self._get_instance_pool_id(role_name, model_name, instance_id)
        for bundle_idx in self._get_instance_bundle_indices(role_name, model_name, instance_id):
            instance_set = self.bundle_to_instances.get((pool_id, bundle_idx))
            if not instance_set:
                continue
            instance_set.discard(target_key)
            if not instance_set:
                self.bundle_to_instances.pop((pool_id, bundle_idx), None)

    def _add_instance_to_bundle_reverse_index(self, role_name: PSRL_Role, model_name: str, instance_id: int):
        target_key = self._instance_key(role_name, model_name, instance_id)
        pool_id = self._get_instance_pool_id(role_name, model_name, instance_id)
        for bundle_idx in self._get_instance_bundle_indices(role_name, model_name, instance_id):
            self.bundle_to_instances.setdefault((pool_id, bundle_idx), set()).add(target_key)

    def _has_other_role_awaken_on_shared_bundle(self, role_name: PSRL_Role, model_name: str, instance_id: int) -> bool:
        pool_id = self._get_instance_pool_id(role_name, model_name, instance_id)
        for bundle_idx in self._get_instance_bundle_indices(role_name, model_name, instance_id):
            for other_role, other_model, other_instance_id in self.bundle_to_instances.get(
                (pool_id, bundle_idx), set()
            ):
                if other_role == role_name:
                    continue
                other_status = (
                    self.instances_status_flags.get(other_role, {}).get(other_model, {}).get(other_instance_id)
                )
                if other_status in (InstanceStatus.AWAKEN, InstanceStatus.TRAINING):
                    return True
        return False

    async def start_busy_loop(self):
        if self.monitor_task is not None and not self.monitor_task.done():
            return

        def _log_background_task_done(task_name: str):
            def _callback(task: asyncio.Task) -> None:
                try:
                    task.result()
                except asyncio.CancelledError:
                    psrl_logger.warning("elastic_rm background task cancelled: %s", task_name)
                except Exception:
                    psrl_logger.exception("elastic_rm background task failed: %s", task_name)
                else:
                    psrl_logger.warning("elastic_rm background task exited unexpectedly: %s", task_name)

            return _callback

        self.running_loop = asyncio.get_running_loop()
        self.stop_monitor = False
        self.stop_scale_up = False
        self.stop_scale_down = False
        self.stop_interrupt_vllm_waiting = False

        self.monitor_task = self.running_loop.create_task(self._monitor_loop())
        self.monitor_task.add_done_callback(_log_background_task_done("monitor_loop"))

        self.scale_up_task = self.running_loop.create_task(self._scale_up_handler_loop())
        self.scale_up_task.add_done_callback(_log_background_task_done("scale_up_handler_loop"))

        self.scale_down_task = self.running_loop.create_task(self._scale_down_handler_loop())
        self.scale_down_task.add_done_callback(_log_background_task_done("scale_down_handler_loop"))

        if self._interrupt_vllm_waiting_enabled:
            self.interrupt_vllm_waiting_task = self.running_loop.create_task(self._interrupt_vllm_waiting_loop())
            self.interrupt_vllm_waiting_task.add_done_callback(
                _log_background_task_done("interrupt_vllm_waiting_loop")
            )

    async def stop(self):
        close_policy = getattr(self.scaling_policy, "close", None)
        if (
            self.monitor_task is None
            and self.scale_up_task is None
            and self.scale_down_task is None
            and self.interrupt_vllm_waiting_task is None
        ):
            if callable(close_policy):
                close_policy()
            return

        self.stop_monitor = True
        self.stop_scale_up = True
        self.stop_scale_down = True
        self.stop_interrupt_vllm_waiting = True

        tasks_to_wait = []
        if self.monitor_task is not None:
            tasks_to_wait.append(self.monitor_task)
        if self.scale_up_task is not None:
            tasks_to_wait.append(self.scale_up_task)
        if self.scale_down_task is not None:
            tasks_to_wait.append(self.scale_down_task)
        if self.interrupt_vllm_waiting_task is not None:
            tasks_to_wait.append(self.interrupt_vllm_waiting_task)
        try:
            if tasks_to_wait:
                await asyncio.gather(*tasks_to_wait, return_exceptions=True)
        finally:
            if callable(close_policy):
                close_policy()

    async def _monitor_loop(self):
        while not self.stop_monitor:
            try:
                # Pull fresh engine status from coordinators first, then decide scaling.
                policy_input_started_s = time.monotonic()
                stage_started_s = time.monotonic()
                await self._sync_engine_status_from_coordinators()
                engine_status_s = time.monotonic() - stage_started_s
                stage_started_s = time.monotonic()
                await self._sync_router_backlog_from_coordinators()
                router_backlog_s = time.monotonic() - stage_started_s
                request_snapshot_s = 0.0
                if getattr(
                    self.scaling_policy,
                    "enable_request_level_candidate_evaluation",
                    False,
                ):
                    stage_started_s = time.monotonic()
                    await self._sync_request_level_snapshots_from_coordinators()
                    request_snapshot_s = time.monotonic() - stage_started_s
                stage_started_s = time.monotonic()
                await self._sync_trainer_waiting_hint()
                trainer_hint_s = time.monotonic() - stage_started_s
                stage_started_s = time.monotonic()
                signals = self._build_instance_signals()
                signal_build_s = time.monotonic() - stage_started_s
                stage_started_s = time.monotonic()
                self._maybe_log_instance_signals(signals)
                signal_logging_s = time.monotonic() - stage_started_s
                policy_input_total_s = time.monotonic() - policy_input_started_s
                policy_input_other_s = max(
                    0.0,
                    policy_input_total_s
                    - engine_status_s
                    - router_backlog_s
                    - request_snapshot_s
                    - trainer_hint_s
                    - signal_build_s
                    - signal_logging_s,
                )
                psrl_logger.info(
                    "[ELASTIC_OVERHEAD] operation=policy_input "
                    "engine_status_s=%.6f router_backlog_s=%.6f "
                    "request_snapshot_s=%.6f trainer_hint_s=%.6f "
                    "signal_build_s=%.6f signal_logging_s=%.6f "
                    "input_other_s=%.6f input_total_s=%.6f",
                    engine_status_s,
                    router_backlog_s,
                    request_snapshot_s,
                    trainer_hint_s,
                    signal_build_s,
                    signal_logging_s,
                    policy_input_other_s,
                    policy_input_total_s,
                )
                router_backlog_for_policy = (
                    self.router_backlog_summary_by_role
                    if isinstance(self.scaling_policy, ITLScalingPolicy)
                    else self.router_backlog_by_role
                )
                planner_started_s = time.monotonic()
                decision_kwargs = {
                    "execution_in_progress": not self._policy_scaling_idle.is_set(),
                    "router_backlog_by_role": router_backlog_for_policy,
                    "trainer_waiting_hint": self.trainer_waiting_hint,
                }
                if isinstance(self.scaling_policy, ITLScalingPolicy):
                    decision_kwargs["request_level_snapshots_by_role"] = self.request_level_snapshots_by_role
                decision = self.scaling_policy.decide(signals, **decision_kwargs)
                planner_elapsed_s = time.monotonic() - planner_started_s
                planner_breakdown = _normalized_planner_breakdown(
                    self.scaling_policy,
                    planner_elapsed_s,
                )
                psrl_logger.info(
                    "[ELASTIC_OVERHEAD] operation=policy_planner policy=%s planner_s=%.6f "
                    "state_analysis_s=%.6f candidate_ordering_s=%.6f "
                    "candidate_set_construction_s=%.6f "
                    "simulation_input_preparation_s=%.6f candidate_evaluation_wall_s=%.6f "
                    "rebalance_simulation_s=%.6f router_simulation_s=%.6f "
                    "simulation_wall_s=%.6f rebalance_router_overlap_s=%.6f "
                    "candidate_scoring_s=%.6f "
                    "best_candidate_selection_s=%.6f "
                    "other_s=%.6f timing_semantics=direct_wall_stage_unions_overlap "
                    "actions=%d reason=%s",
                    type(self.scaling_policy).__name__,
                    planner_elapsed_s,
                    planner_breakdown["state_analysis_s"],
                    planner_breakdown["candidate_ordering_s"],
                    planner_breakdown["candidate_set_construction_s"],
                    planner_breakdown["simulation_input_preparation_s"],
                    planner_breakdown["candidate_evaluation_wall_s"],
                    planner_breakdown["rebalance_simulation_s"],
                    planner_breakdown["router_simulation_s"],
                    planner_breakdown["simulation_wall_s"],
                    planner_breakdown["rebalance_router_overlap_s"],
                    planner_breakdown["candidate_scoring_s"],
                    planner_breakdown["best_candidate_selection_s"],
                    planner_breakdown["other_s"],
                    len(decision.actions),
                    decision.reason,
                )
                if not decision.actions and decision.reason == "decision_execution_in_progress":
                    # perform_sync_swap clears the same idle event but does not use
                    # _decision_pending_action_counts — only count stalls for real policy work.
                    if self._decision_pending_action_counts:
                        self._execution_in_progress_stall_ticks += 1
                    else:
                        self._execution_in_progress_stall_ticks = 0
                    if self._decision_abandon_stall_ticks > 0 and (
                        self._execution_in_progress_stall_ticks == self._decision_abandon_stall_ticks
                    ):
                        self._abandon_in_flight_decision(
                            reason="decision_execution_stall_ticks_exceeded",
                            stall_ticks=self._execution_in_progress_stall_ticks,
                        )
                    elif self._execution_in_progress_stall_ticks in (1, 30, 60) or (
                        self._execution_in_progress_stall_ticks > 60
                        and self._execution_in_progress_stall_ticks % 120 == 0
                    ):
                        psrl_logger.warning(
                            "elastic_rm: policy blocked by in-flight scaling for %d monitor ticks; "
                            "context=%s. Likely causes: (1) coordinator.exec_command stuck inside "
                            "SLEEP/WAKE_UP/ABORT (gen/RM workers or router not returning); "
                            "(2) another client's command ahead in the same coordinator queue never finishes "
                            "(head-of-line blocking). Check ElasticExecutor.log for the last "
                            "coordinator_cmd START line without matching END.",
                            self._execution_in_progress_stall_ticks,
                            self._policy_scaling_wait_context(),
                        )
                else:
                    self._execution_in_progress_stall_ticks = 0

                if decision.actions:
                    decision_id = self._next_decision_id
                    self._next_decision_id += 1
                    decision_gate_token = self._acquire_policy_scaling_idle(
                        f"policy_decision decision_id={decision_id} reason={decision.reason}"
                    )
                    self._decision_gate_tokens[decision_id] = decision_gate_token
                    self._decision_pending_action_counts[decision_id] = len(decision.actions)
                    psrl_logger.info(
                        "Policy decision accepted: decision_id=%d, reason=%s, lambda=%.6f, role_to_total_mu=%s, actions=%s",
                        decision_id,
                        decision.reason,
                        decision.estimated_lambda,
                        decision.role_to_total_mu,
                        [
                            {
                                "action_type": action.action_type,
                                "role_name": action.role_name.name,
                                "model_name": action.model_name,
                                "num_instances": action.num_instances,
                                "preferred_instance_ids": action.preferred_instance_ids,
                                "reason": action.reason,
                                "pre_sleep_other_preferred": action.pre_sleep_other_preferred,
                                "pre_wake_other_preferred": action.pre_wake_other_preferred,
                                "planned_request_migrations": action.planned_request_migrations,
                            }
                            for action in decision.actions
                        ],
                    )
                    psrl_logger.info("elastic_rm trainer_waiting_hint=%s", self.trainer_waiting_hint)
                    for action in decision.actions:
                        task = {
                            "role_name": action.role_name,
                            "model_name": action.model_name,
                            "num_instances": action.num_instances,
                            "preferred_instance_ids": action.preferred_instance_ids or [],
                            "reason": action.reason,
                            "decision_id": decision_id,
                            "training_step": self._current_training_step,
                            "planner_elapsed_s": planner_elapsed_s,
                            "planner_breakdown": dict(planner_breakdown),
                            "pre_sleep_other_preferred": action.pre_sleep_other_preferred or [],
                            "pre_wake_other_preferred": action.pre_wake_other_preferred or [],
                            "planned_request_migrations": action.planned_request_migrations or [],
                        }
                        if action.action_type == "scale_up":
                            self.scale_up_task_queue.put_nowait(task)
                        elif action.action_type == "scale_down":
                            self.scale_down_task_queue.put_nowait(task)
                        else:
                            psrl_logger.warning(
                                "Unknown action_type=%s for decision_id=%d, mark as finished immediately.",
                                action.action_type,
                                decision_id,
                            )
                            self._mark_decision_action_finished(decision_id)
            except Exception:
                # Without this, a single Ray/sync/decide exception kills the monitor task forever
                # (see start_busy_loop done_callback calling task.result()).
                psrl_logger.exception("elastic_rm _monitor_loop iteration failed; will retry after sleep.")
            await asyncio.sleep(max(self.scaling_policy.monitor_interval_ms / 1000, 0.01))

    async def _scale_up_handler_loop(self):
        while not self.stop_scale_up:
            if self.scale_up_task_queue.empty():
                await asyncio.sleep(0)
                continue

            role_need_to_scale_up = self.scale_up_task_queue.get_nowait()
            decision_id = role_need_to_scale_up.get("decision_id")
            training_step = int(
                role_need_to_scale_up.get(
                    "training_step",
                    getattr(self, "_current_training_step", -1),
                )
            )
            action_started_s = time.monotonic()
            sleep_elapsed_s = 0.0
            wake_elapsed_s = 0.0
            migration_elapsed_s = 0.0
            actual_sleep_instances = 0
            actual_wakeup_instances = 0
            primary_wakeup_instances = 0
            try:
                psrl_logger.info(
                    "elastic_rm scale_up_handler decision_id=%s begin task=%s",
                    decision_id,
                    role_need_to_scale_up,
                )
                instances_to_pre_wake = (
                    self._resolve_preferred_instances_to_scaled_up(
                        role_need_to_scale_up.get("pre_wake_other_preferred") or []
                    )
                    or []
                )
                allow_preemptive_scale_up = _preemptive_scale_up_enabled(getattr(self, "scaling_policy", None))
                instances_to_scaled_down = (
                    self._find_instances_to_scaled_down_for_other_roles(role_need_to_scale_up)
                    if allow_preemptive_scale_up
                    else []
                )
                if instances_to_scaled_down:
                    psrl_logger.info(
                        "elastic_rm scale_up_handler decision_id=%s pre_sleep_other count=%s detail=%s",
                        decision_id,
                        len(instances_to_scaled_down),
                        instances_to_scaled_down,
                    )
                    sleep_status_before = self._snapshot_instance_statuses(instances_to_scaled_down)
                    t0 = time.monotonic()
                    await self._scale_down_instances(instances_to_scaled_down)
                    sleep_elapsed_s = time.monotonic() - t0
                    actual_sleep_instances += self._count_instance_status_transitions(
                        sleep_status_before,
                        from_status=InstanceStatus.AWAKEN,
                        to_status=InstanceStatus.ASLEEP,
                    )
                    psrl_logger.info(
                        "elastic_rm scale_up_handler decision_id=%s pre_sleep_other done",
                        decision_id,
                    )

                instances_to_scaled_up = self._find_instances_to_scaled_up(role_need_to_scale_up)
                if not instances_to_scaled_up:
                    psrl_logger.warning("No instances can be scaled up for role %s", role_need_to_scale_up)
                    continue
                instances_to_wake = instances_to_pre_wake + instances_to_scaled_up
                psrl_logger.info(
                    "elastic_rm scale_up_handler decision_id=%s combined_wake pre_wake_targets=%s wake_targets=%s",
                    decision_id,
                    instances_to_pre_wake,
                    instances_to_scaled_up,
                )
                wake_status_before = self._snapshot_instance_statuses(instances_to_wake)
                t0 = time.monotonic()
                await self._scale_up_instances(instances_to_wake)
                wake_elapsed_s = time.monotonic() - t0
                actual_wakeup_instances += self._count_instance_status_transitions(
                    wake_status_before,
                    from_status=InstanceStatus.ASLEEP,
                    to_status=InstanceStatus.AWAKEN,
                )
                successful_wake_instance_ids = {
                    int(item["instance_id"])
                    for item in instances_to_scaled_up
                    if wake_status_before.get(
                        self._instance_key(
                            item["role_name"],
                            item["model_name"],
                            int(item["instance_id"]),
                        )
                    )
                    == InstanceStatus.ASLEEP
                    and self.instances_status_flags.get(item["role_name"], {})
                    .get(item["model_name"], {})
                    .get(int(item["instance_id"]))
                    == InstanceStatus.AWAKEN
                }
                primary_wakeup_instances = len(successful_wake_instance_ids)
                self._record_policy_migration_observation(sleep_elapsed_s, wake_elapsed_s)
                rebalance_after_scale_up = _rebalance_after_scale_up_enabled(getattr(self, "scaling_policy", None))
                psrl_logger.info(
                    "elastic_rm scale_up_handler decision_id=%s wake_targets done; rebalance_after_scale_up=%s",
                    decision_id,
                    rebalance_after_scale_up,
                )
                migration_started_s = time.monotonic()
                request_level_enabled = bool(
                    getattr(
                        getattr(self, "scaling_policy", None),
                        "enable_request_level_candidate_evaluation",
                        False,
                    )
                )
                if not rebalance_after_scale_up:
                    psrl_logger.info(
                        "Skip post-scale-up request rebalance for decision_id=%s policy=%s",
                        decision_id,
                        type(getattr(self, "scaling_policy", None)).__name__,
                    )
                elif request_level_enabled:
                    planned_request_migrations = role_need_to_scale_up.get("planned_request_migrations") or []
                    executable_request_migrations = []
                    dropped_request_migrations = []
                    for migration in planned_request_migrations:
                        try:
                            destination_instance_id = int(migration["destination_instance_id"])
                        except (KeyError, TypeError, ValueError):
                            dropped_request_migrations.append(migration)
                            continue
                        if destination_instance_id in successful_wake_instance_ids:
                            executable_request_migrations.append(migration)
                        else:
                            dropped_request_migrations.append(migration)
                    if dropped_request_migrations:
                        psrl_logger.warning(
                            "Filter planned request migrations to realized awake targets: "
                            "decision_id=%s role=%s model=%s planned=%d executable=%d dropped=%d "
                            "successful_wake_instances=%s dropped_samples=%s",
                            decision_id,
                            role_need_to_scale_up["role_name"],
                            role_need_to_scale_up["model_name"],
                            len(planned_request_migrations),
                            len(executable_request_migrations),
                            len(dropped_request_migrations),
                            sorted(successful_wake_instance_ids),
                            dropped_request_migrations[:3],
                        )
                    if executable_request_migrations or not planned_request_migrations:
                        await self._execute_planned_request_migrations(
                            role_name=role_need_to_scale_up["role_name"],
                            model_name=role_need_to_scale_up["model_name"],
                            request_migrations=executable_request_migrations,
                            decision_id=decision_id,
                        )
                    else:
                        psrl_logger.warning(
                            "Skip planned request migrations because none of their destinations "
                            "were actually awakened: decision_id=%s wake_targets=%s",
                            decision_id,
                            instances_to_scaled_up,
                        )
                else:
                    await self._interrupt_waiting_after_scale_up(
                        role_name=role_need_to_scale_up["role_name"],
                        model_name=role_need_to_scale_up["model_name"],
                        wake_instance_ids=successful_wake_instance_ids,
                        decision_id=decision_id,
                    )
                migration_elapsed_s = time.monotonic() - migration_started_s
                psrl_logger.info("elastic_rm scale_up_handler decision_id=%s post_scale_up_abort done", decision_id)
            except Exception:
                psrl_logger.exception(
                    "elastic_rm scale_up_handler decision_id=%s failed; continue processing queued actions.",
                    decision_id,
                )
            finally:
                self._record_step_scaling_action(
                    step=training_step,
                    decision_id=decision_id,
                    action_type="scale_up",
                    succeeded=primary_wakeup_instances > 0,
                    sleep_instances=actual_sleep_instances,
                    wakeup_instances=actual_wakeup_instances,
                )
                execution_elapsed_s = time.monotonic() - action_started_s
                policy_planner_s = float(role_need_to_scale_up.get("planner_elapsed_s", 0.0))
                planner_breakdown = role_need_to_scale_up.get("planner_breakdown") or {}
                handler_other_s = max(
                    0.0,
                    execution_elapsed_s - sleep_elapsed_s - wake_elapsed_s - migration_elapsed_s,
                )
                psrl_logger.info(
                    "[ELASTIC_OVERHEAD] operation=scale_up decision_id=%s planner_s=%.6f "
                    "planner_scope=decision_batch state_analysis_s=%.6f "
                    "candidate_ordering_s=%.6f "
                    "candidate_set_construction_s=%.6f simulation_input_preparation_s=%.6f "
                    "candidate_evaluation_wall_s=%.6f rebalance_simulation_s=%.6f "
                    "router_simulation_s=%.6f simulation_wall_s=%.6f "
                    "rebalance_router_overlap_s=%.6f "
                    "candidate_scoring_s=%.6f best_candidate_selection_s=%.6f "
                    "planner_other_s=%.6f "
                    "sleep_s=%.6f wakeup_s=%.6f "
                    "post_scale_up_rebalance_trigger_s=%.6f handler_other_s=%.6f "
                    "execution_s=%.6f total_s=%.6f",
                    decision_id,
                    policy_planner_s,
                    float(planner_breakdown.get("state_analysis_s", 0.0)),
                    float(planner_breakdown.get("candidate_ordering_s", 0.0)),
                    float(planner_breakdown.get("candidate_set_construction_s", 0.0)),
                    float(planner_breakdown.get("simulation_input_preparation_s", 0.0)),
                    float(planner_breakdown.get("candidate_evaluation_wall_s", 0.0)),
                    float(planner_breakdown.get("rebalance_simulation_s", 0.0)),
                    float(planner_breakdown.get("router_simulation_s", 0.0)),
                    float(planner_breakdown.get("simulation_wall_s", 0.0)),
                    float(planner_breakdown.get("rebalance_router_overlap_s", 0.0)),
                    float(planner_breakdown.get("candidate_scoring_s", 0.0)),
                    float(planner_breakdown.get("best_candidate_selection_s", 0.0)),
                    float(planner_breakdown.get("other_s", 0.0)),
                    sleep_elapsed_s,
                    wake_elapsed_s,
                    migration_elapsed_s,
                    handler_other_s,
                    execution_elapsed_s,
                    policy_planner_s + execution_elapsed_s,
                )
                self._mark_decision_action_finished(decision_id)

    async def _scale_down_handler_loop(self):
        while not self.stop_scale_down:
            if self.scale_down_task_queue.empty():
                await asyncio.sleep(0)
                continue

            role_need_to_scale_down = self.scale_down_task_queue.get_nowait()
            decision_id = role_need_to_scale_down.get("decision_id")
            training_step = int(
                role_need_to_scale_down.get(
                    "training_step",
                    getattr(self, "_current_training_step", -1),
                )
            )
            action_started_s = time.monotonic()
            sleep_elapsed_s = 0.0
            actual_sleep_instances = 0
            try:
                psrl_logger.info(
                    "elastic_rm scale_down_handler decision_id=%s begin task=%s",
                    decision_id,
                    role_need_to_scale_down,
                )
                instances_to_scaled_down = self._find_instances_to_scaled_down_in_role(role_need_to_scale_down)
                if not instances_to_scaled_down:
                    psrl_logger.warning("No instances can be scaled down for role %s", role_need_to_scale_down)
                    continue
                psrl_logger.info(
                    "elastic_rm scale_down_handler decision_id=%s sleep_targets=%s",
                    decision_id,
                    instances_to_scaled_down,
                )
                sleep_status_before = self._snapshot_instance_statuses(instances_to_scaled_down)
                sleep_started_s = time.monotonic()
                await self._scale_down_instances(instances_to_scaled_down)
                sleep_elapsed_s = time.monotonic() - sleep_started_s
                actual_sleep_instances = self._count_instance_status_transitions(
                    sleep_status_before,
                    from_status=InstanceStatus.AWAKEN,
                    to_status=InstanceStatus.ASLEEP,
                )
                psrl_logger.info("elastic_rm scale_down_handler decision_id=%s sleep_targets done", decision_id)
            except Exception:
                psrl_logger.exception(
                    "elastic_rm scale_down_handler decision_id=%s failed; continue processing queued actions.",
                    decision_id,
                )
            finally:
                self._record_step_scaling_action(
                    step=training_step,
                    decision_id=decision_id,
                    action_type="scale_down",
                    succeeded=actual_sleep_instances > 0,
                    sleep_instances=actual_sleep_instances,
                    wakeup_instances=0,
                )
                execution_elapsed_s = time.monotonic() - action_started_s
                policy_planner_s = float(role_need_to_scale_down.get("planner_elapsed_s", 0.0))
                planner_breakdown = role_need_to_scale_down.get("planner_breakdown") or {}
                handler_other_s = max(0.0, execution_elapsed_s - sleep_elapsed_s)
                psrl_logger.info(
                    "[ELASTIC_OVERHEAD] operation=scale_down decision_id=%s planner_s=%.6f "
                    "planner_scope=decision_batch state_analysis_s=%.6f "
                    "candidate_ordering_s=%.6f "
                    "candidate_set_construction_s=%.6f simulation_input_preparation_s=%.6f "
                    "candidate_evaluation_wall_s=%.6f rebalance_simulation_s=%.6f "
                    "router_simulation_s=%.6f simulation_wall_s=%.6f "
                    "rebalance_router_overlap_s=%.6f "
                    "candidate_scoring_s=%.6f best_candidate_selection_s=%.6f "
                    "planner_other_s=%.6f "
                    "sleep_s=%.6f wakeup_s=0.000000 "
                    "post_scale_up_rebalance_trigger_s=0.000000 handler_other_s=%.6f "
                    "execution_s=%.6f total_s=%.6f",
                    decision_id,
                    policy_planner_s,
                    float(planner_breakdown.get("state_analysis_s", 0.0)),
                    float(planner_breakdown.get("candidate_ordering_s", 0.0)),
                    float(planner_breakdown.get("candidate_set_construction_s", 0.0)),
                    float(planner_breakdown.get("simulation_input_preparation_s", 0.0)),
                    float(planner_breakdown.get("candidate_evaluation_wall_s", 0.0)),
                    float(planner_breakdown.get("rebalance_simulation_s", 0.0)),
                    float(planner_breakdown.get("router_simulation_s", 0.0)),
                    float(planner_breakdown.get("simulation_wall_s", 0.0)),
                    float(planner_breakdown.get("rebalance_router_overlap_s", 0.0)),
                    float(planner_breakdown.get("candidate_scoring_s", 0.0)),
                    float(planner_breakdown.get("best_candidate_selection_s", 0.0)),
                    float(planner_breakdown.get("other_s", 0.0)),
                    sleep_elapsed_s,
                    handler_other_s,
                    execution_elapsed_s,
                    policy_planner_s + execution_elapsed_s,
                )
                self._mark_decision_action_finished(decision_id)

    def _abandon_in_flight_decision(self, *, reason: str, stall_ticks: int) -> None:
        """Report a stalled decision without releasing its live transition lease."""
        if self._policy_scaling_idle.is_set() and not self._decision_pending_action_counts:
            return
        pending = dict(self._decision_pending_action_counts)
        psrl_logger.error(
            "elastic_rm: in-flight scaling decision exceeded stall threshold (%s): "
            "stall_ticks=%d threshold=%d pending_action_counts=%s. Keeping the transition lease "
            "until the original handlers finish to avoid overlapping state changes.",
            reason,
            stall_ticks,
            self._decision_abandon_stall_ticks,
            pending,
        )

    def _mark_decision_action_finished(self, decision_id: int | None):
        if decision_id is None:
            return
        if decision_id not in self._decision_pending_action_counts:
            return
        remaining = self._decision_pending_action_counts[decision_id] - 1
        if remaining > 0:
            self._decision_pending_action_counts[decision_id] = remaining
            return
        self._decision_pending_action_counts.pop(decision_id, None)
        if not self._decision_pending_action_counts:
            psrl_logger.info("elastic_rm decision_id=%d execution completed.", decision_id)
            owner_token = self._decision_gate_tokens.pop(decision_id)
            self._release_policy_scaling_idle_if_clear(owner_token)

    def _record_policy_migration_observation(self, sleep_elapsed_s: float, wake_elapsed_s: float) -> None:
        recorder = getattr(self.scaling_policy, "record_migration_observation", None)
        if recorder is None:
            return
        try:
            recorder(sleep_elapsed_s, wake_elapsed_s)
        except Exception:
            psrl_logger.exception("elastic_rm failed to record policy migration observation.")

    def _maybe_log_instance_signals(self, signals: list[InstanceSignal]):
        if not self._enable_monitor_instance_log:
            return
        if not signals:
            return
        now_ms = asyncio.get_running_loop().time() * 1000
        if now_ms - self._last_monitor_instance_log_ms < max(self._monitor_instance_log_interval_ms, 0):
            return
        self._last_monitor_instance_log_ms = now_ms

        rows: list[dict] = []
        for s in signals:
            role_name = getattr(s.role_name, "name", str(s.role_name))
            rows.append(
                {
                    "role": role_name,
                    "model": s.model_name,
                    "instance": int(s.instance_id),
                    "status": "TRAINING" if bool(s.is_training) else ("AWAKEN" if bool(s.is_awaken) else "ASLEEP"),
                    "pool_id": s.pool_id,
                    "running": int(s.running_queue_num),
                    "waiting": int(s.waiting_queue_num),
                    "kv_cache": float(s.kv_cache_utilization),
                    "throughput": float(s.generation_throughput),
                    "ts": s.snapshot_timestamp,
                }
            )
        rows.sort(key=lambda r: (r["status"] != "AWAKEN", r["role"], r["model"], r["instance"]))
        awake_by_role: dict[str, int] = {}
        awake_by_role_model: dict[str, int] = {}
        for row in rows:
            if row["status"] != "AWAKEN":
                continue
            role_key = str(row["role"])
            role_model_key = f"{role_key}/{row['model']}"
            awake_by_role[role_key] = awake_by_role.get(role_key, 0) + 1
            awake_by_role_model[role_model_key] = awake_by_role_model.get(role_model_key, 0) + 1
        monitor_logger.info("------------------------------------------------------------")
        for row in rows:
            monitor_logger.info("Instance current Status: %s", row)
        monitor_logger.info("------------------------------------------------------------")
        psrl_logger.info(
            "Awake instances summary: by_role=%s, by_role_model=%s",
            dict(sorted(awake_by_role.items())),
            dict(sorted(awake_by_role_model.items())),
        )
        monitor_logger.info(
            "Awake instances summary: by_role=%s, by_role_model=%s",
            dict(sorted(awake_by_role.items())),
            dict(sorted(awake_by_role_model.items())),
        )
        router_backlog_log: dict[str, int] = {}
        for role_key, backlog in self.router_backlog_by_role.items():
            role_name = getattr(role_key, "name", str(role_key))
            router_backlog_log[role_name] = int(backlog)
        router_backlog_log = dict(sorted(router_backlog_log.items()))
        monitor_logger.info("------------------------------------------------------------")
        psrl_logger.info("Router backlog summary: by_role=%s", router_backlog_log)
        monitor_logger.info("Router backlog summary: by_role=%s", router_backlog_log)
        monitor_logger.info("------------------------------------------------------------")

    async def _await_elastic_coordinator_command(
        self,
        coordinator: ray.actor.ActorHandle,
        command: Command,
        *,
        stage: str,
        timeout_s: object = _ELASTIC_COORD_CMD_TIMEOUT_UNSET,
    ) -> object | None:
        """Run coordinator.exec_command with optional timeout and structured tracing for elastic_rm debugging."""
        if timeout_s is _ELASTIC_COORD_CMD_TIMEOUT_UNSET:
            eff_timeout = self._coordinator_command_timeout_s
        else:
            eff_timeout = timeout_s  # float | None: None = no Ray timeout (wait indefinitely)
        t0 = time.monotonic()
        psrl_logger.info(
            "elastic_rm coordinator_cmd START stage=%s type=%s args=%s timeout_s=%s",
            stage,
            command.type.name,
            command.get_args(),
            eff_timeout,
        )
        kwargs: dict = {"blocking": True}
        if eff_timeout is not None:
            kwargs["timeout"] = eff_timeout
        try:
            result = await coordinator.exec_command.remote(command, **kwargs)
        except Exception:
            psrl_logger.exception(
                "elastic_rm coordinator_cmd EXCEPTION stage=%s type=%s elapsed_s=%.3f",
                stage,
                command.type.name,
                time.monotonic() - t0,
            )
            raise
        if isinstance(result, dict) and result.get("command_id") is not None:
            command_id = int(result["command_id"])
            psrl_logger.warning(
                "elastic_rm coordinator_cmd timed out but remains active: stage=%s command_id=%d status=%s; "
                "waiting for the original command instead of issuing a duplicate",
                stage,
                command_id,
                result.get("status"),
            )
            result = await coordinator.synchronize_command.remote(command_id, timeout=None)
        elapsed = time.monotonic() - t0
        psrl_logger.info(
            "elastic_rm coordinator_cmd END stage=%s type=%s elapsed_s=%.3f result=%r",
            stage,
            command.type.name,
            elapsed,
            result,
        )
        if result is None and eff_timeout is not None:
            psrl_logger.error(
                "elastic_rm coordinator_cmd got None (timeout or failure) stage=%s — "
                "ElasticExecutor local flags may diverge from cluster; consider coordinator_command_timeout_s "
                "and check coordinator logs for stuck SLEEP/WAKE_UP/ABORT.",
                stage,
            )
        return result

    async def _scale_up_instance(self, instance_to_scaled_up: dict) -> bool:
        instance_role = instance_to_scaled_up["role_name"]
        instance_model_name = instance_to_scaled_up["model_name"]
        instance_id = int(instance_to_scaled_up["instance_id"])
        status = self.instances_status_flags.get(instance_role, {}).get(instance_model_name, {}).get(instance_id)
        if status == InstanceStatus.TRAINING:
            psrl_logger.info(
                "Skip scale up for role=%s model=%s instance=%d: instance is reserved by trainer pool.",
                instance_role,
                instance_model_name,
                instance_id,
            )
            return False

        coordinator = self.coordinators[instance_role][instance_model_name]
        primary_stage = (
            f"WAKE_UP role={getattr(instance_role, 'name', instance_role)} "
            f"model={instance_model_name} instance={instance_id}"
        )
        try:
            result = await self._await_elastic_coordinator_command(
                coordinator,
                Command(
                    type=CommandType.WAKE_UP,
                    instance_ids=[instance_id],
                    sync_operation_id=instance_to_scaled_up.get("sync_operation_id"),
                ),
                stage=primary_stage,
            )
        except Exception:
            psrl_logger.exception(
                "elastic_rm WAKE_UP RPC raised role=%s model=%s instance=%s; marking RECOVERING",
                getattr(instance_role, "name", instance_role),
                instance_model_name,
                instance_id,
            )
            self.instances_status_flags[instance_role][instance_model_name][instance_id] = InstanceStatus.RECOVERING
            return False
        if result is True:
            self.instances_status_flags[instance_role][instance_model_name][instance_id] = InstanceStatus.AWAKEN
            self._set_instance_immunity(instance_role, instance_model_name, instance_id)
            return True
        if result is False:
            psrl_logger.info(
                "elastic_rm WAKE_UP rejected by coordinator role=%s model=%s instance=%s (e.g. sync-lock); "
                "leaving the local state unchanged.",
                getattr(instance_role, "name", instance_role),
                instance_model_name,
                instance_id,
            )
            return False
        self.instances_status_flags[instance_role][instance_model_name][instance_id] = InstanceStatus.RECOVERING
        return False

    async def _scale_up_instances(self, instances_to_scaled_up: list[dict]) -> None:
        """Batch WAKE_UP for same role/model instances without duplicate retries."""
        if not instances_to_scaled_up:
            return
        if len(instances_to_scaled_up) == 1:
            await self._scale_up_instance(instances_to_scaled_up[0])
            return

        role_set = {item["role_name"] for item in instances_to_scaled_up}
        model_set = {item["model_name"] for item in instances_to_scaled_up}
        if len(role_set) != 1 or len(model_set) != 1:
            await asyncio.gather(*[self._scale_up_instance(item) for item in instances_to_scaled_up])
            return

        instance_role = next(iter(role_set))
        instance_model_name = next(iter(model_set))
        instance_ids = [int(item["instance_id"]) for item in instances_to_scaled_up]
        coordinator = self.coordinators[instance_role][instance_model_name]
        primary_stage = (
            f"WAKE_UP role={getattr(instance_role, 'name', instance_role)} "
            f"model={instance_model_name} instances={instance_ids}"
        )

        result = None
        try:
            result = await self._await_elastic_coordinator_command(
                coordinator,
                Command(type=CommandType.WAKE_UP, instance_ids=instance_ids),
                stage=primary_stage,
            )
        except Exception:
            psrl_logger.exception(
                "elastic_rm batch WAKE_UP RPC raised role=%s model=%s instance_ids=%s; not retrying blindly.",
                getattr(instance_role, "name", instance_role),
                instance_model_name,
                instance_ids,
            )

        if result is True:
            for instance_id in instance_ids:
                self.instances_status_flags[instance_role][instance_model_name][instance_id] = InstanceStatus.AWAKEN
                self._set_instance_immunity(instance_role, instance_model_name, instance_id)
            return

        psrl_logger.warning(
            "elastic_rm batch WAKE_UP did not fully succeed (result=%r) role=%s model=%s instance_ids=%s; "
            "not issuing per-instance duplicates.",
            result,
            getattr(instance_role, "name", instance_role),
            instance_model_name,
            instance_ids,
        )
        return

    def _waiting_uids_for_abort_by_ratio(
        self,
        normalized_waiting_uids: list[str],
        ratio: float,
    ) -> list[str]:
        """Take the first k waiting uids (FIFO vs queue order); k = floor(n * ratio)."""
        if not normalized_waiting_uids:
            return []
        r = max(0.0, min(1.0, float(ratio)))
        if r <= 0.0:
            return []
        if r >= 1.0:
            return list(normalized_waiting_uids)
        k = min(
            len(normalized_waiting_uids),
            int(math.floor(float(len(normalized_waiting_uids)) * r + 1e-12)),
        )
        return normalized_waiting_uids[:k]

    async def _interrupt_vllm_waiting_loop(self):
        while not self.stop_interrupt_vllm_waiting:
            try:
                if self._interrupt_vllm_waiting_ratio > 0.0:
                    await self._sync_engine_status_from_coordinators()
                    for role_name, model_name in self.roles:
                        await self._interrupt_vllm_waiting_for_role(role_name, model_name)
            except Exception:
                psrl_logger.exception("elastic_rm vLLM waiting interrupt loop failed; will retry after sleep.")
            await asyncio.sleep(self._interrupt_vllm_waiting_interval_s)

    async def _interrupt_vllm_waiting_for_role(self, role_name: PSRL_Role, model_name: str):
        role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        awaken_instance_ids = [
            int(instance_id) for instance_id, status in role_status.items() if status == InstanceStatus.AWAKEN
        ]
        if not awaken_instance_ids:
            return

        instance_to_uids: dict[int, list[str]] = {}
        role_engine_stats = self.instances_engine_stats.get(role_name, {}).get(model_name, {})
        for instance_id in awaken_instance_ids:
            snapshot = role_engine_stats.get(instance_id, {})
            if not isinstance(snapshot, dict):
                continue
            scheduler_stats = snapshot.get("scheduler_stats", {})
            if not isinstance(scheduler_stats, dict):
                continue
            waiting_uids = scheduler_stats.get("req_id_in_waiting", [])
            normalized_waiting_uids = [str(uid) for uid in waiting_uids if uid is not None]
            selected = self._waiting_uids_for_abort_by_ratio(
                normalized_waiting_uids,
                self._interrupt_vllm_waiting_ratio,
            )
            if selected:
                instance_to_uids[int(instance_id)] = selected

        if not instance_to_uids:
            return

        coordinator = self.coordinators.get(role_name, {}).get(model_name)
        if coordinator is None:
            psrl_logger.warning(
                "Skip vLLM waiting ABORT: coordinator missing for role=%s model=%s.",
                role_name,
                model_name,
            )
            return

        interrupted_request_num = await self._await_elastic_coordinator_command(
            coordinator,
            Command(
                type=CommandType.ABORT,
                instance_to_uids=instance_to_uids,
            ),
            stage=(
                f"ABORT_vllm_waiting_requeue role={getattr(role_name, 'name', role_name)} "
                f"model={model_name} instances={sorted(instance_to_uids.keys())}"
            ),
        )
        psrl_logger.info(
            (
                "vLLM waiting requeue abort executed: role=%s model=%s "
                "uid_sources=%s interrupted=%s abort_waiting_ratio=%.4f"
            ),
            role_name,
            model_name,
            sorted(instance_to_uids.keys()),
            interrupted_request_num,
            self._interrupt_vllm_waiting_ratio,
        )

    @staticmethod
    def _normalize_request_token_map(raw: dict | None) -> dict[str, int]:
        if not isinstance(raw, dict):
            return {}
        normalized: dict[str, int] = {}
        for key, value in raw.items():
            try:
                normalized[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return normalized

    def _build_post_scale_up_request_candidates(
        self,
        *,
        role_name: PSRL_Role,
        model_name: str,
        wake_instance_ids: set[int],
    ) -> tuple[dict[int, list[_PostScaleUpRequestCandidate]], float, float]:
        """Collect donor-side requests visible in scheduler stats for post-scale-up rebalancing.

        We only look at currently awake donor instances, excluding the new wake targets.
        The goal is to approximate the post-scale-up load distribution using request
        counts and sequence lengths already exposed in scheduler stats.
        """
        # Only scheduler-visible requests are considered here. Waiting requests are
        # cheaper to move, but running requests are still eligible if they are the
        # only way to reduce imbalance on the donor side.
        role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        role_engine_stats = self.instances_engine_stats.get(role_name, {}).get(model_name, {})
        candidates_by_instance: dict[int, list[_PostScaleUpRequestCandidate]] = {}
        total_requests = 0.0
        total_tokens = 0.0

        for instance_id, status in role_status.items():
            iid = int(instance_id)
            if status != InstanceStatus.AWAKEN or iid in wake_instance_ids:
                continue
            snapshot = role_engine_stats.get(iid, {})
            if not isinstance(snapshot, dict):
                continue
            scheduler_stats = snapshot.get("scheduler_stats", {})
            if not isinstance(scheduler_stats, dict):
                continue

            prompt_map = self._normalize_request_token_map(scheduler_stats.get("req_id_to_prompt_token_num"))
            response_map = self._normalize_request_token_map(scheduler_stats.get("req_id_to_response_token_num"))
            waiting_uids = {str(uid) for uid in scheduler_stats.get("req_id_in_waiting", []) if uid is not None}
            request_ids = set(prompt_map) | set(response_map) | waiting_uids
            if not request_ids:
                continue

            instance_candidates: list[_PostScaleUpRequestCandidate] = []
            instance_token_sum = 0.0
            for request_id in request_ids:
                seq_len = max(0, int(prompt_map.get(request_id, 0)) + int(response_map.get(request_id, 0)))
                instance_candidates.append(
                    _PostScaleUpRequestCandidate(
                        instance_id=iid,
                        request_id=request_id,
                        seq_len=seq_len,
                        is_waiting=request_id in waiting_uids,
                    )
                )
                instance_token_sum += float(seq_len)

            if instance_candidates:
                candidates_by_instance[iid] = instance_candidates
                total_requests += float(len(instance_candidates))
                total_tokens += float(instance_token_sum)

        return candidates_by_instance, total_requests, total_tokens

    async def _interrupt_waiting_after_scale_up_balanced_candidates(
        self,
        role_name: PSRL_Role,
        model_name: str,
        wake_instance_ids: set[int] | None = None,
        decision_id: int | None = None,
    ):
        # Rebalance only after the wake step is done. We approximate the new
        # post-scale-up steady state from scheduler stats, then choose the
        # smallest interrupt set that moves the donor side toward even
        # request-count and sequence-length balance. New wake targets are
        # excluded from donors.
        planner_started_s = time.monotonic()
        wake_ids = {int(i) for i in (wake_instance_ids or set())}
        candidates_by_instance, total_requests, total_tokens = self._build_post_scale_up_request_candidates(
            role_name=role_name,
            model_name=model_name,
            wake_instance_ids=wake_ids,
        )
        if not candidates_by_instance or total_requests <= 0.0:
            psrl_logger.info(
                "Skip post-scale-up rebalancing: no donor candidates role=%s model=%s wake_ids=%s.",
                role_name,
                model_name,
                sorted(wake_ids),
            )
            return

        source_instance_count = len(candidates_by_instance)
        target_instance_count = source_instance_count + len(wake_ids)
        if target_instance_count <= 0:
            return

        moved_request_budget = total_requests * float(len(wake_ids)) / float(target_instance_count)
        moved_request_count = int(math.floor(moved_request_budget + 0.5))
        if moved_request_count <= 0:
            psrl_logger.info(
                "Skip post-scale-up rebalancing: moved_request_budget=%.4f < 1 role=%s model=%s.",
                moved_request_budget,
                role_name,
                model_name,
            )
            return

        target_requests_per_instance = total_requests / float(target_instance_count)
        target_tokens_per_instance = total_tokens / float(target_instance_count)
        selected_by_instance: dict[int, list[str]] = {}
        selected_total = 0
        selected_tokens = 0.0

        request_weight = self._post_scale_up_rebalance_request_weight
        length_weight = self._post_scale_up_rebalance_length_weight
        request_denominator = max(target_requests_per_instance, 1e-12)
        length_denominator = max(target_tokens_per_instance, 1e-12)

        def donor_score(request_count: float, token_sum: float) -> tuple[float, float, float]:
            over_request_ratio = (request_count - target_requests_per_instance) / request_denominator
            over_length_ratio = (token_sum - target_tokens_per_instance) / length_denominator
            score = request_weight * over_request_ratio + length_weight * over_length_ratio
            return score, over_request_ratio, over_length_ratio

        donor_plans: list[_PostScaleUpDonorPlan] = []
        for instance_id, cands in candidates_by_instance.items():
            request_count = float(len(cands))
            token_sum = float(sum(candidate.seq_len for candidate in cands))
            score, over_request_ratio, over_length_ratio = donor_score(request_count, token_sum)
            if score <= 0.0:
                continue

            def candidate_sort_key(candidate: _PostScaleUpRequestCandidate) -> tuple[float, int, int, str]:
                marginal_score_drop = (
                    request_weight / request_denominator
                    + length_weight * float(candidate.seq_len) / length_denominator
                )
                return (
                    -marginal_score_drop,
                    0 if candidate.is_waiting else 1,
                    -int(candidate.seq_len),
                    candidate.request_id,
                )

            donor_plans.append(
                _PostScaleUpDonorPlan(
                    instance_id=int(instance_id),
                    request_count=request_count,
                    token_sum=token_sum,
                    score=score,
                    over_request_ratio=over_request_ratio,
                    over_length_ratio=over_length_ratio,
                    candidates=sorted(cands, key=candidate_sort_key),
                )
            )

        donor_plans.sort(
            key=lambda item: (
                -item.score,
                -item.over_request_ratio,
                -item.over_length_ratio,
                item.instance_id,
            )
        )

        for donor_plan in donor_plans:
            if selected_total >= moved_request_count:
                break
            instance_id = donor_plan.instance_id
            remaining_count = donor_plan.request_count
            remaining_tokens = donor_plan.token_sum
            selected_for_instance: list[str] = []

            for candidate in donor_plan.candidates:
                if selected_total >= moved_request_count:
                    break
                selected_for_instance.append(candidate.request_id)
                selected_total += 1
                selected_tokens += float(candidate.seq_len)
                remaining_count -= 1.0
                remaining_tokens = max(0.0, remaining_tokens - float(candidate.seq_len))
                score, _, _ = donor_score(remaining_count, remaining_tokens)
                if score < 0.0:
                    break

            if selected_for_instance:
                selected_by_instance[instance_id] = selected_for_instance

        if not selected_by_instance:
            psrl_logger.info(
                "Skip post-scale-up rebalancing: selection produced no aborts role=%s model=%s wake_ids=%s.",
                role_name,
                model_name,
                sorted(wake_ids),
            )
            return

        coordinator = self.coordinators.get(role_name, {}).get(model_name)
        if coordinator is None:
            psrl_logger.warning(
                "Skip post-scale-up rebalancing: coordinator missing for role=%s model=%s.",
                role_name,
                model_name,
            )
            return

        instance_to_uids = {instance_id: uids for instance_id, uids in selected_by_instance.items() if uids}
        if not instance_to_uids:
            return

        planner_elapsed_s = time.monotonic() - planner_started_s
        migration_sequence = int(getattr(self, "_next_migration_sequence", 1))
        self._next_migration_sequence = migration_sequence + 1
        migration_id = f"{decision_id}:{migration_sequence}:{getattr(role_name, 'name', role_name)}:{model_name}"
        migration_context = {
            "migration_id": migration_id,
            "decision_id": decision_id,
            "role_name": getattr(role_name, "name", str(role_name)),
            "model_name": model_name,
            "planner_s": planner_elapsed_s,
            "selected_count": selected_total,
            "selected_tokens": selected_tokens,
        }

        try:
            network_trigger_started_s = time.monotonic()
            interrupted_request_num = await self._await_elastic_coordinator_command(
                coordinator,
                Command(
                    type=CommandType.ABORT,
                    instance_to_uids=instance_to_uids,
                    migration_context=migration_context,
                ),
                stage=(
                    f"ABORT_post_scale_up_rebalance role={getattr(role_name, 'name', role_name)} model={model_name} "
                    f"wake_instances={sorted(wake_ids)} donor_instances={sorted(instance_to_uids.keys())}"
                ),
            )
            network_trigger_elapsed_s = time.monotonic() - network_trigger_started_s
            if interrupted_request_num is None:
                psrl_logger.warning(
                    "elastic_rm post-scale-up rebalance ABORT returned None (timeout?); role=%s model=%s",
                    role_name,
                    model_name,
                )
                return
            psrl_logger.info(
                "[ELASTIC_OVERHEAD] operation=post_scale_up_rebalance_trigger "
                "scope=post_scale_up_rebalance migration_id=%s decision_id=%s "
                "planner_s=%.6f network_trigger_s=%.6f reprefill_s=pending selected=%d "
                "interrupted=%s selected_tokens=%.0f",
                migration_id,
                decision_id,
                planner_elapsed_s,
                network_trigger_elapsed_s,
                selected_total,
                interrupted_request_num,
                selected_tokens,
            )
            psrl_logger.info(
                (
                    "Post-scale-up rebalancing executed: role=%s model=%s wake_instances=%s donor_instances=%s "
                    "selected=%s interrupted=%s selected_tokens=%.0f "
                    "target_requests_per_instance=%.2f target_tokens_per_instance=%.2f "
                    "request_weight=%.4f length_weight=%.4f"
                ),
                role_name,
                model_name,
                sorted(wake_ids),
                sorted(instance_to_uids.keys()),
                selected_total,
                interrupted_request_num,
                selected_tokens,
                target_requests_per_instance,
                target_tokens_per_instance,
                request_weight,
                length_weight,
            )
        except Exception as exc:
            psrl_logger.warning(
                "elastic_rm post-scale-up rebalancing failed for role=%s model=%s wake_instances=%s: %s",
                role_name,
                model_name,
                sorted(wake_ids),
                exc,
            )

    async def _interrupt_waiting_after_scale_up(
        self,
        role_name: PSRL_Role,
        model_name: str,
        wake_instance_ids: set[int] | None = None,
        decision_id: int | None = None,
    ):
        wake_ids = {int(i) for i in (wake_instance_ids or set())}
        await self._interrupt_waiting_after_scale_up_balanced_candidates(
            role_name=role_name,
            model_name=model_name,
            wake_instance_ids=wake_ids,
            decision_id=decision_id,
        )

    async def _execute_planned_request_migrations(
        self,
        *,
        role_name: PSRL_Role,
        model_name: str,
        request_migrations: list[dict],
        decision_id: int | None,
    ) -> int:
        """Trigger only the request migrations selected by candidate evaluation."""
        planned = len(request_migrations)
        if planned == 0:
            psrl_logger.info(
                "Request-level rebalance selected no migrations: decision_id=%s role=%s model=%s; "
                "legacy rebalance remains disabled for this action.",
                decision_id,
                role_name,
                model_name,
            )
            return 0
        coordinator = self.coordinators.get(role_name, {}).get(model_name)
        if coordinator is None:
            psrl_logger.warning(
                "Skip planned request migrations: coordinator missing role=%s model=%s",
                role_name,
                model_name,
            )
            return 0
        migration_sequence = int(getattr(self, "_next_migration_sequence", 1))
        self._next_migration_sequence = migration_sequence + 1
        migration_id = (
            f"{decision_id}:{migration_sequence}:{getattr(role_name, 'name', role_name)}:{model_name}:planned"
        )
        migration_context = {
            "migration_id": migration_id,
            "decision_id": decision_id,
            "role_name": getattr(role_name, "name", str(role_name)),
            "model_name": model_name,
            "selected_count": planned,
        }
        interrupted = await self._await_elastic_coordinator_command(
            coordinator,
            Command(
                type=CommandType.ABORT,
                request_migrations=request_migrations,
                migration_context=migration_context,
            ),
            stage=(
                f"ABORT_planned_request_migrations role="
                f"{getattr(role_name, 'name', role_name)} model={model_name} planned={planned}"
            ),
        )
        interrupted_count = 0 if interrupted is None else int(interrupted)
        psrl_logger.info(
            "[ELASTIC_OVERHEAD] operation=planned_request_rebalance migration_id=%s "
            "decision_id=%s role=%s model=%s planned=%d interrupted=%d",
            migration_id,
            decision_id,
            getattr(role_name, "name", role_name),
            model_name,
            planned,
            interrupted_count,
        )
        return interrupted_count

    async def _scale_down_instance(self, instance_to_scaled_down: dict) -> bool:
        return await self._scale_down_instances([instance_to_scaled_down])

    async def _scale_down_instances(self, instances_to_scaled_down: list[dict]) -> bool:
        grouped: dict[tuple[PSRL_Role, str, int | None], list[int]] = defaultdict(list)
        min_awake_per_role = max(0, int(getattr(self.scaling_policy, "min_awake_per_role", 0)))

        for instance_to_scaled_down in instances_to_scaled_down:
            instance_role = instance_to_scaled_down["role_name"]
            instance_model_name = instance_to_scaled_down["model_name"]
            instance_id = int(instance_to_scaled_down["instance_id"])
            sync_operation_id = instance_to_scaled_down.get("sync_operation_id")

            status = self.instances_status_flags.get(instance_role, {}).get(instance_model_name, {}).get(instance_id)
            if status == InstanceStatus.TRAINING:
                psrl_logger.info(
                    "Skip scale down for role=%s model=%s instance=%d: instance is reserved by trainer pool.",
                    instance_role,
                    instance_model_name,
                    instance_id,
                )
                continue
            if status != InstanceStatus.AWAKEN:
                psrl_logger.info(
                    "Skip scale down for role=%s model=%s instance=%d: status=%s.",
                    instance_role,
                    instance_model_name,
                    instance_id,
                    getattr(status, "name", status),
                )
                continue

            if self._is_instance_in_immunity(instance_role, instance_model_name, instance_id):
                remaining_ms = (
                    self._instance_immunity_until_ms.get(
                        self._instance_key(instance_role, instance_model_name, instance_id), 0.0
                    )
                    - time.monotonic() * 1000
                )
                psrl_logger.info(
                    "Skip scale down for role=%s model=%s instance=%d: within wake-up immunity (%.0f ms remaining).",
                    instance_role,
                    instance_model_name,
                    instance_id,
                    max(0.0, remaining_ms),
                )
                continue

            grouped[(instance_role, instance_model_name, sync_operation_id)].append(instance_id)

        async def _sleep_group(
            instance_role: PSRL_Role,
            instance_model_name: str,
            sync_operation_id: int | None,
            candidate_ids: list[int],
        ) -> bool:
            role_status = self.instances_status_flags.get(instance_role, {}).get(instance_model_name, {})
            awaken_count = sum(1 for status in role_status.values() if status == InstanceStatus.AWAKEN)
            allowed_to_sleep = max(0, awaken_count - min_awake_per_role)
            if allowed_to_sleep <= 0:
                psrl_logger.info(
                    "Skip scale down for role=%s model=%s instances=%s: keep at least %d awaken instances.",
                    instance_role,
                    instance_model_name,
                    candidate_ids,
                    min_awake_per_role,
                )
                return False

            instance_ids = candidate_ids[:allowed_to_sleep]
            if len(instance_ids) < len(candidate_ids):
                psrl_logger.info(
                    "Truncate scale down for role=%s model=%s instances=%s -> %s to keep min_awake_per_role=%d.",
                    instance_role,
                    instance_model_name,
                    candidate_ids,
                    instance_ids,
                    min_awake_per_role,
                )

            if min_awake_per_role == 0 and awaken_count == 1 and instance_ids:
                instance_id = int(instance_ids[0])
                running_n, waiting_n = self._get_instance_running_waiting(
                    instance_role, instance_model_name, instance_id
                )
                if running_n > 0 or waiting_n > 0:
                    psrl_logger.info(
                        "Skip scale down for role=%s model=%s instance=%d: sole awake instance still has "
                        "running=%d waiting=%d (min_awake_per_role=0).",
                        instance_role,
                        instance_model_name,
                        instance_id,
                        running_n,
                        waiting_n,
                    )
                    return False

            coordinator = self.coordinators[instance_role][instance_model_name]
            sleep_stage = (
                f"SLEEP role={getattr(instance_role, 'name', instance_role)} "
                f"model={instance_model_name} instances={instance_ids}"
            )
            try:
                result = await self._await_elastic_coordinator_command(
                    coordinator,
                    Command(
                        type=CommandType.SLEEP,
                        instance_ids=instance_ids,
                        sync_operation_id=sync_operation_id,
                    ),
                    stage=sleep_stage,
                )
            except Exception:
                psrl_logger.exception(
                    "elastic_rm SLEEP RPC raised role=%s model=%s instances=%s; marking RECOVERING",
                    getattr(instance_role, "name", instance_role),
                    instance_model_name,
                    instance_ids,
                )
                for instance_id in instance_ids:
                    self.instances_status_flags[instance_role][instance_model_name][instance_id] = (
                        InstanceStatus.RECOVERING
                    )
                return False
            if result is None:
                for instance_id in instance_ids:
                    self.instances_status_flags[instance_role][instance_model_name][instance_id] = (
                        InstanceStatus.RECOVERING
                    )
                return False
            if result is False:
                psrl_logger.info(
                    "elastic_rm SLEEP rejected by coordinator for role=%s model=%s instances=%s "
                    "(an instance may be in sync-lock); skipping status update.",
                    instance_role,
                    instance_model_name,
                    instance_ids,
                )
                return False
            for instance_id in instance_ids:
                self.instances_status_flags[instance_role][instance_model_name][instance_id] = InstanceStatus.ASLEEP
                self.instances_engine_stats[instance_role][instance_model_name][instance_id] = None
            return True

        results = await asyncio.gather(
            *[
                _sleep_group(instance_role, instance_model_name, sync_operation_id, instance_ids)
                for (instance_role, instance_model_name, sync_operation_id), instance_ids in grouped.items()
            ]
        )
        return all(results) if results else False

    async def _sleep_instances_for_training_pool(self, instances: list[dict]) -> None:
        grouped: dict[tuple[PSRL_Role, str], list[int]] = defaultdict(list)
        for instance in instances:
            instance_role = instance["role_name"]
            instance_model_name = instance["model_name"]
            instance_id = int(instance["instance_id"])
            status = self.instances_status_flags.get(instance_role, {}).get(instance_model_name, {}).get(instance_id)
            if status in (InstanceStatus.ASLEEP, InstanceStatus.TRAINING):
                continue
            if status != InstanceStatus.AWAKEN:
                psrl_logger.warning(
                    "Cannot reserve trainer-pool instance with unexpected status: role=%s model=%s instance=%d status=%s.",
                    instance_role,
                    instance_model_name,
                    instance_id,
                    getattr(status, "name", status),
                )
                raise RuntimeError(f"Failed to reserve trainer-pool instance for training: {instance!r}")
            grouped[(instance_role, instance_model_name)].append(instance_id)

        async def _sleep_group(instance_role: PSRL_Role, instance_model_name: str, instance_ids: list[int]) -> None:
            coordinator = self.coordinators[instance_role][instance_model_name]
            result = await self._await_elastic_coordinator_command(
                coordinator,
                Command(type=CommandType.SLEEP, instance_ids=instance_ids),
                stage=(
                    f"SLEEP_for_training_pool role={getattr(instance_role, 'name', instance_role)} "
                    f"model={instance_model_name} instances={instance_ids}"
                ),
                timeout_s=None,
            )
            if result is not True:
                psrl_logger.warning(
                    "Failed to sleep trainer-pool instances before training: role=%s model=%s instance_ids=%s result=%r.",
                    instance_role,
                    instance_model_name,
                    instance_ids,
                    result,
                )
                raise RuntimeError(
                    "Failed to reserve trainer-pool instances for training: "
                    f"role={instance_role!r} model={instance_model_name!r} instance_ids={instance_ids!r}"
                )
            for instance_id in instance_ids:
                self.instances_status_flags[instance_role][instance_model_name][instance_id] = InstanceStatus.ASLEEP
                self.instances_engine_stats[instance_role][instance_model_name][instance_id] = None

        await asyncio.gather(
            *[
                _sleep_group(instance_role, instance_model_name, instance_ids)
                for (instance_role, instance_model_name), instance_ids in grouped.items()
            ]
        )

    def _release_instances_from_training_pool(self, instances: list[dict]) -> None:
        for instance in instances:
            instance_role = instance["role_name"]
            instance_model_name = instance["model_name"]
            instance_id = int(instance["instance_id"])
            status = self.instances_status_flags.get(instance_role, {}).get(instance_model_name, {}).get(instance_id)
            if status != InstanceStatus.TRAINING:
                psrl_logger.warning(
                    "Cannot release trainer-pool instance with unexpected status: role=%s model=%s instance=%d status=%s.",
                    instance_role,
                    instance_model_name,
                    instance_id,
                    getattr(status, "name", status),
                )
                continue
            self.instances_status_flags[instance_role][instance_model_name][instance_id] = InstanceStatus.ASLEEP
            self.instances_engine_stats[instance_role][instance_model_name][instance_id] = None

    async def enter_training_pool(self, instances: list[dict]) -> None:
        normalized = self._normalize_instance_entries(instances)
        owner_token = await self._wait_and_acquire_policy_scaling_idle(f"enter_training_pool instances={normalized}")
        entered = False
        try:
            await self._sleep_instances_for_training_pool(normalized)
            for instance in normalized:
                role_name = instance["role_name"]
                model_name = instance["model_name"]
                instance_id = int(instance["instance_id"])
                self.instances_status_flags[role_name][model_name][instance_id] = InstanceStatus.TRAINING
                self.instances_engine_stats[role_name][model_name][instance_id] = None
            psrl_logger.info("Reserved trainer-pool elastic instances for training: %s.", normalized)
            entered = True
        finally:
            # A training step now owns train_pool GPUs; they are no longer
            # available for elastic replicas. Set before releasing the idle
            # gate so the next policy tick observes the updated trainer_busy.
            if entered:
                self._train_pool_available = False
            self._release_policy_scaling_idle_if_clear(owner_token)

    async def leave_training_pool(self, instances: list[dict]) -> None:
        normalized = self._normalize_instance_entries(instances)
        self._release_instances_from_training_pool(normalized)
        # Training step done; trainer actor is NIXL-slept again and train_pool
        # GPUs are lent back to elastic rollout/RM replicas.
        self._train_pool_available = True
        psrl_logger.info("Released trainer-pool elastic instances from training: %s.", normalized)

    def _find_instances_to_scaled_up(self, role_need_to_scale_up: dict):
        role_name = role_need_to_scale_up["role_name"]
        model_name = role_need_to_scale_up["model_name"]
        num_instances = int(role_need_to_scale_up.get("num_instances", 1))
        preferred_instance_ids = [int(i) for i in role_need_to_scale_up.get("preferred_instance_ids", [])]
        status_dict = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        all_asleep_ids = [
            instance_id for instance_id, status in status_dict.items() if status == InstanceStatus.ASLEEP
        ]
        if not all_asleep_ids:
            return None
        # Prefer the suggested instances; if none are available (e.g. already awake due to state race),
        # fall back to any asleep instance rather than failing the entire scale-up.
        if preferred_instance_ids:
            preferred_available = [
                instance_id for instance_id in all_asleep_ids if instance_id in preferred_instance_ids
            ]
            if preferred_available:
                candidate_ids = preferred_available
            else:
                psrl_logger.info(
                    "Preferred instances %s not asleep for role=%s model=%s, falling back to all asleep instances.",
                    preferred_instance_ids,
                    role_name,
                    model_name,
                )
                candidate_ids = all_asleep_ids
        else:
            candidate_ids = all_asleep_ids

        # First pass: respect preferred ordering and cross-role GPU conflict guard.
        filtered_ids = [
            instance_id
            for instance_id in candidate_ids
            if not self._has_other_role_awaken_on_shared_bundle(role_name, model_name, int(instance_id))
        ]

        # Fallback: when preferred candidates are all filtered out by conflict guard,
        # try all asleep instances. This avoids force-wake starvation where policy
        # keeps requesting a fixed preferred instance id that is temporarily conflicted.
        if not filtered_ids and preferred_instance_ids:
            fallback_filtered_ids = [
                instance_id
                for instance_id in all_asleep_ids
                if not self._has_other_role_awaken_on_shared_bundle(role_name, model_name, int(instance_id))
            ]
            if fallback_filtered_ids:
                psrl_logger.info(
                    (
                        "Preferred instances %s are conflict-filtered for role=%s model=%s; "
                        "fallback to non-conflicting asleep instances %s."
                    ),
                    preferred_instance_ids,
                    role_name,
                    model_name,
                    fallback_filtered_ids,
                )
                filtered_ids = fallback_filtered_ids
        if not filtered_ids:
            return None
        return [
            {"role_name": role_name, "model_name": model_name, "instance_id": instance_id}
            for instance_id in filtered_ids[:num_instances]
        ]

    def _resolve_preferred_instances_to_scaled_up(self, preferred_entries: list[dict]) -> list[dict]:
        if not preferred_entries:
            return []
        resolved: list[dict] = []
        seen: set[tuple[PSRL_Role, str, int]] = set()
        for entry in preferred_entries:
            if not isinstance(entry, dict):
                continue
            role_name = entry.get("role_name")
            model_name = entry.get("model_name")
            if role_name is None or model_name is None:
                continue
            try:
                instance_id = int(entry["instance_id"])
            except (KeyError, TypeError, ValueError):
                continue
            key = self._instance_key(role_name, model_name, instance_id)
            if key in seen:
                continue
            role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
            if role_status.get(instance_id) != InstanceStatus.ASLEEP:
                continue
            if self._has_other_role_awaken_on_shared_bundle(role_name, model_name, instance_id):
                continue
            resolved.append({"role_name": role_name, "model_name": model_name, "instance_id": instance_id})
            seen.add(key)
        return resolved

    def _find_instances_to_scaled_down_for_other_roles(self, role_need_to_scale_up: dict):
        target_role = role_need_to_scale_up["role_name"]
        num_instances = int(role_need_to_scale_up.get("num_instances", 1))
        preferred_entries = role_need_to_scale_up.get("pre_sleep_other_preferred") or []
        removable_budget: dict[tuple[PSRL_Role, str], int] = {}
        min_awake_per_role = max(0, int(getattr(self.scaling_policy, "min_awake_per_role", 0)))

        for role_name, model_name in self.roles:
            if role_name == target_role:
                continue
            role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
            awaken_ids = [
                instance_id for instance_id, status in role_status.items() if status == InstanceStatus.AWAKEN
            ]
            # Do not cede more than the removable budget of this role/model.
            max_removable = max(0, len(awaken_ids) - min_awake_per_role)
            if max_removable <= 0:
                continue
            removable_budget[(role_name, model_name)] = max_removable

        candidates: list[dict] = []
        for role_name, model_name in self.roles:
            if role_name == target_role:
                continue
            role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
            for instance_id, status in role_status.items():
                if status == InstanceStatus.AWAKEN:
                    if self._is_instance_in_immunity(role_name, model_name, int(instance_id)):
                        continue
                    candidates.append(
                        {"role_name": role_name, "model_name": model_name, "instance_id": int(instance_id)}
                    )

        if not candidates and not preferred_entries:
            return None

        picked: list[dict] = []
        seen: set[tuple[PSRL_Role, str, int]] = set()

        def _try_pick_preferred(entry: dict) -> None:
            r = entry.get("role_name")
            m = entry.get("model_name")
            if r is None or m is None:
                return
            try:
                iid = int(entry["instance_id"])
            except (KeyError, TypeError, ValueError):
                return
            if r == target_role:
                return
            sig = (r, m, iid)
            if sig in seen:
                return
            key = (r, m)
            if removable_budget.get(key, 0) <= 0:
                return
            role_status = self.instances_status_flags.get(r, {}).get(m, {})
            if role_status.get(iid) != InstanceStatus.AWAKEN:
                return
            if self._is_instance_in_immunity(r, m, iid):
                return
            cand = {"role_name": r, "model_name": m, "instance_id": iid}
            picked.append(cand)
            seen.add(sig)
            removable_budget[key] -= 1

        for entry in preferred_entries:
            if isinstance(entry, dict):
                _try_pick_preferred(entry)

        if preferred_entries:
            return picked

        rest = [c for c in candidates if (c["role_name"], c["model_name"], c["instance_id"]) not in seen]
        if not rest and not picked:
            return None
        rest.sort(
            key=lambda item: self._get_instance_kv_cache_usage(
                item["role_name"], item["model_name"], item["instance_id"]
            )
        )
        for candidate in rest:
            key = (candidate["role_name"], candidate["model_name"])
            if removable_budget.get(key, 0) <= 0:
                continue
            picked.append(candidate)
            seen.add((candidate["role_name"], candidate["model_name"], candidate["instance_id"]))
            removable_budget[key] -= 1
            if len(picked) >= num_instances:
                break
        return picked if picked else None

    def _find_instances_to_scaled_down_in_role(self, role_need_to_scale_down: dict):
        role_name = role_need_to_scale_down["role_name"]
        model_name = role_need_to_scale_down["model_name"]
        num_instances = int(role_need_to_scale_down.get("num_instances", 1))
        min_awake_per_role = max(0, int(getattr(self.scaling_policy, "min_awake_per_role", 0)))
        preferred_instance_ids = [int(i) for i in role_need_to_scale_down.get("preferred_instance_ids", [])]
        role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        # Compute max_scalable_down from the TOTAL awake count, not the preferred-filtered subset.
        # Previously this was computed after preferred filtering, which caused max_scalable_down=0
        # whenever only 1 preferred instance was awake (e.g. 1 preferred out of 8 awake total).
        all_awake_ids = [instance_id for instance_id, status in role_status.items() if status == InstanceStatus.AWAKEN]
        if not all_awake_ids:
            return None
        max_scalable_down = len(all_awake_ids) - min_awake_per_role
        if max_scalable_down <= 0:
            psrl_logger.info(
                "Skip scale down for role=%s model=%s: keep at least %d awaken instances (total awake=%d).",
                role_name,
                model_name,
                min_awake_per_role,
                len(all_awake_ids),
            )
            return None
        # Prefer the suggested instances; if none are awake (e.g. already asleep due to state race),
        # fall back to all awake instances rather than failing the entire scale-down.
        if preferred_instance_ids:
            preferred_available = [
                instance_id for instance_id in all_awake_ids if instance_id in preferred_instance_ids
            ]
            if preferred_available:
                candidate_ids = preferred_available
            else:
                psrl_logger.info(
                    "Preferred instances %s not awake for role=%s model=%s, falling back to all awake instances.",
                    preferred_instance_ids,
                    role_name,
                    model_name,
                )
                candidate_ids = list(all_awake_ids)
        else:
            candidate_ids = list(all_awake_ids)
        # Filter out instances that are within their wake-up immunity window.
        candidate_ids = [iid for iid in candidate_ids if not self._is_instance_in_immunity(role_name, model_name, iid)]
        if not candidate_ids:
            return None
        candidate_ids = sorted(
            candidate_ids,
            key=lambda instance_id: self._get_instance_kv_cache_usage(role_name, model_name, instance_id),
        )
        effective_num_instances = min(num_instances, max_scalable_down)
        return [
            {"role_name": role_name, "model_name": model_name, "instance_id": int(instance_id)}
            for instance_id in candidate_ids[:effective_num_instances]
        ]

    async def _await_coordinator_refs_with_per_ref_timeout(
        self,
        refs: list,
        task_keys: list,
        op_label: str,
        *,
        trace_router_backlog: bool = False,
    ) -> list:
        """Await each coordinator Ray ObjectRef with its own timeout (parallel).

        Wrapping a single ``asyncio.gather`` in one ``wait_for`` lets the slowest
        RPC drop every other result for that tick; elastic_rm then mis-reads load.
        """
        timeout_s = self._coordinator_sync_timeout_s

        async def _one(ref, key):
            t0 = time.monotonic()
            if trace_router_backlog:
                log_elastic_rm_backlog_diag(
                    psrl_logger,
                    "stage=ElasticExecutor_wait_ref_begin op=%s key=%s timeout_s=%s",
                    op_label,
                    key,
                    timeout_s if timeout_s > 0 else "none",
                )
            try:
                if timeout_s > 0:
                    out = await asyncio.wait_for(ref, timeout=timeout_s)
                else:
                    out = await ref
                if trace_router_backlog:
                    log_elastic_rm_backlog_diag(
                        psrl_logger,
                        "stage=ElasticExecutor_wait_ref_done op=%s key=%s total_s=%.3f",
                        op_label,
                        key,
                        time.monotonic() - t0,
                    )
                return out
            except asyncio.TimeoutError:
                psrl_logger.warning(
                    "elastic_rm: %s RPC timed out after %.1fs for key=%s; skipped for this tick. "
                    "(If PSRL_ELASTIC_RM_BACKLOG_DIAG=1, compare coordinator/router stages above.)",
                    op_label,
                    timeout_s,
                    key,
                )
                return TimeoutError(f"{op_label} key={key}")

        return await asyncio.gather(*[_one(r, k) for r, k in zip(refs, task_keys)], return_exceptions=True)

    async def _sync_engine_status_from_coordinators(self):
        refs = []
        task_keys: list[tuple[PSRL_Role, str]] = []
        for role_name, model_name in self.roles:
            coordinator = self.coordinators.get(role_name, {}).get(model_name)
            if coordinator is None:
                continue
            refs.append(coordinator.get_instance_engine_status_snapshot.remote())
            task_keys.append((role_name, model_name))

        if not refs:
            return
        results = await self._await_coordinator_refs_with_per_ref_timeout(
            refs, task_keys, op_label="get_instance_engine_status_snapshot"
        )
        for (role_name, model_name), result in zip(task_keys, results):
            if isinstance(result, Exception):
                psrl_logger.warning(
                    "Failed to fetch engine status from coordinator role=%s model=%s: %s",
                    role_name,
                    model_name,
                    result,
                )
                continue
            role_stats = self.instances_engine_stats.setdefault(role_name, {}).setdefault(model_name, {})
            for instance_id_str, snapshot in result.items():
                try:
                    instance_id = int(instance_id_str)
                except (TypeError, ValueError):
                    instance_id = int(snapshot.get("instance_id", -1))
                if instance_id < 0:
                    continue
                role_stats[instance_id] = snapshot

    async def _sync_router_backlog_from_coordinators(self):
        refs = []
        task_keys: list[tuple[PSRL_Role, str]] = []
        router_waiting_top_t = int(getattr(self.scaling_policy, "router_waiting_top_t", 0))
        for role_name, model_name in self.roles:
            coordinator = self.coordinators.get(role_name, {}).get(model_name)
            if coordinator is None:
                continue
            if router_waiting_top_t != 0:
                summary_top_t = None if router_waiting_top_t < 0 else router_waiting_top_t
                refs.append(coordinator.get_router_backlog_summary.remote(summary_top_t))
            else:
                refs.append(coordinator.get_router_backlog_size.remote())
            task_keys.append((role_name, model_name))
        if not refs:
            self.router_backlog_by_role = {}
            self.router_backlog_summary_by_role = {}
            return
        results = await self._await_coordinator_refs_with_per_ref_timeout(
            refs,
            task_keys,
            op_label="get_router_backlog_size",
            trace_router_backlog=True,
        )
        prev_by_role = dict(self.router_backlog_by_role)
        prev_summary_by_role = dict(self.router_backlog_summary_by_role)
        grouped: dict[PSRL_Role, list] = defaultdict(list)
        for (role_name, model_name), result in zip(task_keys, results):
            grouped[role_name].append(result)

        role_backlog: dict[PSRL_Role, int] = {}
        role_summary: dict[PSRL_Role, dict[str, int]] = {}
        for role_key, result_list in grouped.items():
            total = 0
            selected_count = 0
            selected_tokens = 0
            for result in result_list:
                if isinstance(result, Exception):
                    psrl_logger.warning("Failed to fetch router backlog for role=%s: %s", role_key, result)
                    role_backlog[role_key] = prev_by_role.get(role_key, 0)
                    role_summary[role_key] = prev_summary_by_role.get(
                        role_key,
                        {"pending": prev_by_role.get(role_key, 0), "count": 0, "total_tokens": 0},
                    )
                    break
                if isinstance(result, dict):
                    total += max(0, int(result.get("pending", result.get("count", 0))))
                    selected_count += max(0, int(result.get("count", 0)))
                    selected_tokens += max(0, int(result.get("total_tokens", 0)))
                else:
                    total += max(0, int(result))
            else:
                role_backlog[role_key] = total
                role_summary[role_key] = {
                    "pending": total,
                    "count": selected_count,
                    "total_tokens": selected_tokens,
                }
        self.router_backlog_by_role = role_backlog
        self.router_backlog_summary_by_role = role_summary

    async def _sync_request_level_snapshots_from_coordinators(self) -> None:
        """Fetch rollout/RM compact snapshots concurrently for one policy tick."""
        refs = []
        task_keys: list[tuple[PSRL_Role, str]] = []
        router_waiting_top_t = int(getattr(self.scaling_policy, "router_waiting_top_t", 0))
        top_t = None if router_waiting_top_t < 0 else router_waiting_top_t
        for role_name, model_name in self.roles:
            coordinator = self.coordinators.get(role_name, {}).get(model_name)
            if coordinator is None:
                continue
            refs.append(coordinator.get_candidate_evaluation_snapshot.remote(top_t))
            task_keys.append((role_name, model_name))
        if not refs:
            self.request_level_snapshots_by_role = {}
            return
        results = await self._await_coordinator_refs_with_per_ref_timeout(
            refs,
            task_keys,
            op_label="get_candidate_evaluation_snapshot",
        )
        grouped_results: dict[PSRL_Role, list[tuple[str, object]]] = defaultdict(list)
        for (role_name, model_name), result in zip(task_keys, results, strict=True):
            grouped_results[role_name].append((model_name, result))

        snapshots: dict[PSRL_Role, object] = {}
        for role_name in (PSRL_Role.Rollout, PSRL_Role.RewardModel):
            role_results = grouped_results.get(role_name, [])
            if len(role_results) != 1:
                snapshots[role_name] = RuntimeError(
                    f"request-level evaluation requires one router snapshot for "
                    f"{role_name.name}, got {len(role_results)}"
                )
                continue
            model_name, result = role_results[0]
            if isinstance(result, Exception):
                psrl_logger.warning(
                    "Failed to fetch request-level snapshot role=%s model=%s: %s",
                    role_name,
                    model_name,
                    result,
                )
            snapshots[role_name] = result
        self.request_level_snapshots_by_role = snapshots

    async def _sync_trainer_waiting_hint(self):
        # trainer_busy is owned by the executor via _train_pool_available, which
        # is flipped by enter/leave_training_pool (called every training step)
        # and seeded at construction with the trainer's initial sleep state. This
        # avoids depending on the AgentLoopManager, whose handle is None here
        # (the executor is created before the manager) and which does not
        # implement get_trainer_waiting_hint.
        self.trainer_waiting_hint = {
            "trainer_busy": not self._train_pool_available,
            "waiting_buffer_id": None,
            "waiting_on": "none",
            "breakdown": {},
        }

    def _get_instance_kv_cache_usage(self, role_name: PSRL_Role, model_name: str, instance_id: int) -> float:
        snapshot = self.instances_engine_stats.get(role_name, {}).get(model_name, {}).get(instance_id, {})
        if not isinstance(snapshot, dict):
            return 1.0
        scheduler_stats = snapshot.get("scheduler_stats", {})
        if not isinstance(scheduler_stats, dict):
            return 1.0
        return float(scheduler_stats.get("kv_cache_usage", 1.0))

    def _get_instance_running_waiting(
        self, role_name: PSRL_Role, model_name: str, instance_id: int
    ) -> tuple[int, int]:
        """Engine scheduler queue depth for elastic sleep gating (same keys as InstanceSignal)."""
        snapshot = self.instances_engine_stats.get(role_name, {}).get(model_name, {}).get(instance_id, {})
        if not isinstance(snapshot, dict):
            return (0, 0)
        scheduler_stats = snapshot.get("scheduler_stats", {})
        if not isinstance(scheduler_stats, dict):
            return (0, 0)
        return (
            int(scheduler_stats.get("num_running_reqs", 0)),
            int(scheduler_stats.get("num_waiting_reqs", 0)),
        )

    def _build_instance_signals(self) -> list[InstanceSignal]:
        signals: list[InstanceSignal] = []
        for role_name, role_data in self.instances_status_flags.items():
            for model_name, instance_status in role_data.items():
                for instance_id, status in instance_status.items():
                    snapshot = self.instances_engine_stats.get(role_name, {}).get(model_name, {}).get(instance_id, {})
                    if not isinstance(snapshot, dict):
                        snapshot = {}
                    scheduler_stats = snapshot.get("scheduler_stats", {})
                    if not isinstance(scheduler_stats, dict):
                        scheduler_stats = {}
                    if status != InstanceStatus.AWAKEN:
                        scheduler_stats = {}
                        snapshot = {}
                    bundle_mapping = (
                        self.instance_bundle_mappings.get(role_name, {}).get(model_name, {}).get(instance_id, {})
                    )
                    br = bundle_mapping.get("bundle_range")
                    pool_id = str(bundle_mapping.get("pool_id") or "shared_rollout_pool")
                    if br is not None and len(br) == 2:
                        b0, b1 = int(br[0]), int(br[1])
                        bundle_keys = frozenset((pool_id, idx) for idx in range(b0, b1)) if b0 < b1 else None
                    else:
                        bundle_keys = None
                    signal = InstanceSignal(
                        role_name=role_name,
                        model_name=model_name,
                        instance_id=int(instance_id),
                        is_awaken=status == InstanceStatus.AWAKEN,
                        is_training=status == InstanceStatus.TRAINING,
                        pool_id=pool_id,
                        kv_cache_utilization=float(scheduler_stats.get("kv_cache_usage", 0.0)),
                        running_queue_num=int(scheduler_stats.get("num_running_reqs", 0)),
                        waiting_queue_num=int(scheduler_stats.get("num_waiting_reqs", 0)),
                        generation_throughput=float(snapshot.get("generation_throughput", 0.0)),
                        total_token_num=int(
                            sum((scheduler_stats.get("req_id_to_prompt_token_num") or {}).values())
                            + sum((scheduler_stats.get("req_id_to_response_token_num") or {}).values())
                        ),
                        snapshot_timestamp=snapshot.get("timestamp"),
                        bundle_keys=bundle_keys,
                    )
                    signals.append(signal)
        return signals
