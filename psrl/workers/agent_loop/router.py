import asyncio
import concurrent.futures
import logging
import os
import time
from collections import deque
from typing import Any

import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from verl import DataProto
from vllm.sampling_params import RequestOutputKind

from psrl.utils.elastic_rm.candidate_routing import resolve_candidate_model_versions
from psrl.utils.elastic_rm.diagnostics import log_elastic_rm_backlog_diag
from psrl.utils.elastic_rm.overhead import RequestMigrationOverheadTracker
from psrl.utils.logger import DualOutputHandler, EventType, deprecated, log_dual_events
from psrl.utils.ray import AsyncBusyPollingRayLock
from psrl.utils.rollout.reprefill import filter_rollout_request_ids
from psrl.utils.rollout.request_id import canonical_psrl_request_id
from psrl.utils.rollout.rollout_trace import rollout_trace_op
from psrl.workers.agent_loop.request_queue import (
    MultiPriorityRequestQueue,
    PriorityRequestQueue,
    RequestSortIndicator,
    get_priority_by_version,
    get_priority_by_version_and_id,
    get_priority_by_version_and_token_num,
)
from psrl.workers.agent_loop.route_strategy import (
    RouteStrategyBase,
    get_route_strategy_class,
)
from psrl.workers.agent_loop.transition_tracker import InstanceTransitionTracker
from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ray.remote(concurrency_groups={"control": 1, "transition_wait": 32, "monitor": 1})
class RolloutRouter:
    def _partial_rollout_trace_enabled(self) -> bool:
        return bool(OmegaConf.select(self.config, "psrl.partial_rollout.trace.enable", default=False))

    def _log_partial_rollout_route(
        self,
        *,
        request_id: int,
        is_partial: bool,
        previous_instance: int | None,
        requested_version: int,
        chosen_instance: int | None,
        route_reason: str,
        candidates: list[int],
    ) -> None:
        """Emit the routing side of a partial-rollout continuation audit."""
        if not self._partial_rollout_trace_enabled() or not is_partial:
            return

        selected_version = (
            self.instance_to_version_after_sync.get(chosen_instance) if chosen_instance is not None else None
        )
        candidate_versions = ",".join(
            f"{instance_id}:{self.instance_to_version_after_sync[instance_id]}"
            for instance_id in sorted(candidates)
        )
        psrl_logger.warning(
            "[PARTIAL_ROLLOUT_TRACE] stage=route uid=%s partial=1 previous_instance=%s "
            "requested_version=%s selected_instance=%s selected_version=%s route_reason=%s candidates=%s",
            request_id,
            previous_instance if previous_instance is not None else "none",
            requested_version,
            chosen_instance if chosen_instance is not None else "none",
            selected_version if selected_version is not None else "none",
            route_reason,
            candidate_versions or "none",
        )

    def __init__(
        self,
        config: DictConfig,
        ps_manager_handle,
        tokenizer,
        rollout_wg_list,
    ):
        """Initialize the rollout router.
        Managing rollout requests across multiple worker groups.
        Handles request routing, load balancing, and consolidation of generation results.

        Args:
            config (DictConfig): Configuration containing rollout settings.
            ps_manager_handle: Handle to the parameter server manager.
            rollout_wg_list: List of rollout worker groups.
        """
        self.config = config
        self.staleness = self.config.psrl.staleness
        self.n_rollout_instances = self.config.psrl.deployment.n_rollout_instances
        self.n_validate_instances = (
            self.config.psrl.deployment.n_validate_instances if self.config.psrl.colocate_validate_and_train else 0
        )

        # TODO(linsh): currently we only support balance strategy on rollout instances
        # we may extend it to validate instances in the future with dynamic routing strategy
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
            self.balanced_concurrent_seqs_per_instance = (
                self.config.psrl.redundant_rollout.redundant_global_batch_size
                * self.rollout_n
                // self.n_rollout_instances
            )
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
            self.balanced_concurrent_seqs_per_instance = (
                self.config.psrl.staleness_buffer_entries * self.rollout_n // self.n_rollout_instances
            )

        self.val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n
        self.ps_manager_handle = ps_manager_handle
        self.tokenizer = tokenizer
        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_size = len(rollout_wg_list)
        assert self.rollout_wg_size == self.n_rollout_instances + self.n_validate_instances, (
            "Rollout worker group size must match the number of deployment instances"
        )

        # Build logger
        self.log_prefix = "RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

        # Routing related attributes
        if self.config.psrl.routing_strategy.enable_multi_priority_queue:
            self.requests_to_route = MultiPriorityRequestQueue(
                self.staleness,
                request_sort_indicator=RequestSortIndicator(self.config.psrl.routing_strategy.request_sort_indicator),
            )
        else:
            self.requests_to_route = PriorityRequestQueue(
                self.staleness,
                request_sort_indicator=RequestSortIndicator(self.config.psrl.routing_strategy.request_sort_indicator),
            )
        self._is_routing = False
        self._pause_routing = False
        self.scheduler_task = None  # Will be created in async context
        # Track the inflight request ids for each instance (i.e., request that is being generated
        # and is not yet completed or queued in the priority queue): {instance_id: [request_id, ...]}
        self.instance_to_inflight_request_ids = {i: [] for i in range(self.rollout_wg_size)}
        # Track the instance id for each incomplete request (i.e., request that is not completed yet):
        # {request_id: instance_id}
        self.incomplete_request_to_instance = {}
        self.request_futures = {}  # Track request futures: {request_id: Future}
        # Keep validation state separate so experimental rollout-only
        # interruption can never include validation requests.
        self._request_is_validate: dict[int, bool] = {}
        self._force_reroute_request_ids: set[int] = set()
        # Track the version after synchronization for each instance: {instance_id: ps_model_version}
        self.instance_to_version_after_sync = {i: 0 for i in range(self.rollout_wg_size)}
        # Track the instance ids that are currently paused (not available for routing)
        self.currently_paused_instance_ids = set()
        # A transition tracks only requests that were in flight when its instances
        # were paused (plus dispatches that crossed the pause boundary). This keeps
        # model-sync waits independent from unrelated requests routed afterwards.
        self._instance_transitions = InstanceTransitionTracker()
        # Track requests in sticky session: {request_id: bool}
        self.sticky_session_requests = {}
        self._migration_overhead = RequestMigrationOverheadTracker()
        # Compact metadata retained while a request is dispatched. It is used
        # only to build read-only candidate-evaluation snapshots.
        self._candidate_evaluation_inflight_requests: dict[str, DataProto] = {}
        # request id -> planned destination. Counters are changed only when the
        # interrupted request actually returns to the normal dispatch path.
        self._planned_migration_destinations: dict[str, int] = {}
        self._planned_migration_counters = {
            "planned": 0,
            "accepted": 0,
            "skipped": 0,
            "forced": 0,
            "fallback": 0,
        }
        self._exclusive_rebalance_migration_queue_enabled = bool(
            OmegaConf.select(
                self.config,
                "psrl.deployment.elastic_rm.itl_policy.exclusive_rebalance_migration_queue",
                default=False,
            )
        )
        self._rebalance_requests_to_route: deque[DataProto] = deque()
        self._rebalance_pending_request_ids: set[str] = set()
        self._rebalance_queued_request_ids: set[str] = set()
        self._rebalance_dispatching_request_ids: set[str] = set()
        self._routing_wakeup_event: asyncio.Event | None = None
        self._routing_event_loop: asyncio.AbstractEventLoop | None = None
        self._rebalance_completion_waiters: set[concurrent.futures.Future[None]] = set()

        # Build logger
        self.log_prefix = "RolloutRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info("Initialized RolloutRouter")

    def init_route_strategy(self, **kwargs):
        """Initialize the route strategy for the router.

        Args:
            **kwargs: Keyword arguments for the route strategy.
        """
        if (
            self.config.psrl.routing_strategy.method == "request_num_balance"
            or self.config.psrl.routing_strategy.method == "throughput_optimal"
        ):
            status_required = (
                "Status collection must be enabled when using request num "
                "balance or throughput optimal routing strategy"
            )
            assert self.config.psrl.status_collection.enable, status_required
        n_instances = self.rollout_wg_size
        if self.config.psrl.deployment.heterogeneous_rollout.enable:
            het_config = self.config.psrl.deployment.heterogeneous_rollout
            tp_sizes = het_config.tensor_model_parallel_size_per_instance
            pp_sizes = het_config.pipeline_model_parallel_size_per_instance
            instance_to_tp_pp = {i: f"TP{tp_sizes[i]}_PP{pp_sizes[i]}" for i in range(n_instances)}
        else:
            rollout_config = self.config.gen_actor_rollout_ref.rollout
            instance_to_tp_pp = {
                i: f"TP{rollout_config.tensor_model_parallel_size}_PP{rollout_config.pipeline_model_parallel_size}"
                for i in range(n_instances)
            }
        rollout_model_name = self.config.gen_actor_rollout_ref.model.path.split("/")[-1]
        strategy_kwargs = {
            "logging_interval_in_ms": self.config.psrl.routing_strategy.logging_interval_in_ms,
            "cost_model_path": self.config.psrl.routing_strategy.cost_model_path,
            "model_name": rollout_model_name,
            "instance_to_tp_pp": instance_to_tp_pp,
            "max_num_waiting_reqs_after_preemption": (
                self.config.psrl.routing_strategy.max_num_waiting_reqs_after_preemption
            ),
            "balanced_concurrent_seqs_per_instance": self.balanced_concurrent_seqs_per_instance,
            "max_concurrent_seqs_per_instance": (self.config.psrl.routing_strategy.max_concurrent_seqs_per_instance),
            "delta_throughput_threshold": self.config.psrl.routing_strategy.delta_throughput_threshold,
            "max_prompt_length": self.config.data.max_prompt_length,
            "request_budget": self.config.psrl.routing_strategy.request_budget,
            "snapshot_staleness_threshold_in_ms": self.config.psrl.routing_strategy.snapshot_staleness_threshold_in_ms,
            "logger": psrl_logger,
            **kwargs,
        }
        try:
            route_strategy_class = get_route_strategy_class(self.config.psrl.routing_strategy.method)
            self.route_strategy: RouteStrategyBase = route_strategy_class(n_instances, strategy_kwargs)
            psrl_logger.info(f"Initialized route strategy: {self.config.psrl.routing_strategy.method}")
        except Exception as e:
            psrl_logger.warning(f"Route strategy error: {e}")
            psrl_logger.warning("Falling back to 'round_robin' strategy")
            from psrl.workers.agent_loop.route_strategy import RoundRobinRouteStrategy

            self.route_strategy: RouteStrategyBase = RoundRobinRouteStrategy(n_instances, strategy_kwargs)

    @ray.method(concurrency_group="control")
    async def update_instance_status(self, instance_to_engine_status: dict[int, EngineStats], **kwargs):
        """Update the instance status with latest information from coordinator.

        Args:
            instance_to_engine_status (dict[int, EngineStats]): Latest engine status information.
            **kwargs: Keyword arguments for the update.
        """
        # NOTE(lhy): This method is called by RolloutCoordinator
        # Each agent loop worker contains a RolloutRouter, which shares the same engine status
        # Note that the instance_to_engine_status may be stale and some instances may be absent at beginning

        # Filter out the stale engine status (has a large bias from current engine status)
        filtered_instance_ids = []
        for instance_id, engine_status in instance_to_engine_status.items():
            if self.route_strategy.is_staled(instance_id, engine_status):
                # psrl_logger.warning(f"Instance {instance_id} collected engine status is stale, skipping")
                continue
            filtered_instance_ids.append(instance_id)
        self.route_strategy.update_instance_to_engine_status(
            {instance_id: instance_to_engine_status[instance_id] for instance_id in filtered_instance_ids}
        )

    @ray.method(concurrency_group="control")
    async def update_currently_syncing_instances(self, instance_ids: list[int], ps_model_version: int):
        """Update the currently syncing instances.

        Args:
            instance_ids (List[int]): The instance IDs to update.
            ps_model_version (int): The version of the PS model to update.
        """
        for instance_id in instance_ids:
            self.instance_to_version_after_sync[instance_id] = ps_model_version
        psrl_logger.info(f"Updated currently syncing instances: {instance_ids} to version {ps_model_version}")

    @ray.method(concurrency_group="control")
    async def enter_sticky_session(self, request_id: int):
        """Mark a request as entering sticky session.

        Args:
            request_id (int): The request ID to mark.
        """
        self.sticky_session_requests[request_id] = True

    @ray.method(concurrency_group="control")
    async def exit_sticky_session(self, request_id: int):
        """Mark a request as exiting sticky session.

        Args:
            request_id (int): The request ID to unmark.
        """
        self.sticky_session_requests.pop(request_id, None)

    @ray.method(concurrency_group="control")
    async def pause_instances(self, instance_ids: list[int]):
        """Notify the router about paused instances.

        Args:
            instance_ids (List[int]): List of instance IDs that are paused.
        """
        for instance_id in instance_ids:
            self.currently_paused_instance_ids.add(instance_id)

    @ray.method(concurrency_group="control")
    def begin_instance_transition(self, instance_ids: list[int]) -> int:
        """Pause instances and snapshot requests belonging to this transition."""
        normalized_ids = {int(instance_id) for instance_id in instance_ids}
        self.currently_paused_instance_ids.update(normalized_ids)
        transition_id = self._instance_transitions.begin(
            normalized_ids,
            self.instance_to_inflight_request_ids,
        )
        pending_ids = self._instance_transitions.pending_request_ids(transition_id) or set()
        psrl_logger.info(
            "Started instance transition %d: instances=%s pending_request_ids=%s",
            transition_id,
            sorted(normalized_ids),
            sorted(pending_ids),
        )
        return transition_id

    def _track_transition_request_started(self, instance_id: int, request_id: int) -> None:
        """Capture a dispatch that crossed an already-established pause boundary."""
        self._instance_transitions.request_started(instance_id, request_id)

    def _track_transition_request_resolved(self, request_id: int) -> None:
        self._instance_transitions.request_resolved(request_id)

    @ray.method(concurrency_group="transition_wait")
    async def wait_instance_transition(self, transition_id: int) -> None:
        """Wait until requests captured by one transition have returned from workers."""
        transition_id = int(transition_id)
        psrl_logger.info("Waiting for instance transition %d requests to resolve", transition_id)
        while self._instance_transitions.pending_request_ids(transition_id):
            await asyncio.sleep(0.01)
        psrl_logger.info("Instance transition %d requests resolved", transition_id)

    @ray.method(concurrency_group="control")
    def finish_instance_transition(
        self,
        transition_id: int,
        resume: bool = True,
        resume_instance_ids: list[int] | None = None,
    ) -> None:
        """Forget transition bookkeeping and optionally make its instances routable."""
        transition_id = int(transition_id)
        target_ids = self._instance_transitions.finish(transition_id)
        resumed_ids = target_ids if resume_instance_ids is None else {int(i) for i in resume_instance_ids}
        if resume:
            self.currently_paused_instance_ids.difference_update(resumed_ids)
        psrl_logger.info(
            "Finished instance transition %d: instances=%s resumed_instances=%s",
            transition_id,
            sorted(target_ids),
            sorted(resumed_ids) if resume else [],
        )

    @ray.method(concurrency_group="control")
    async def resume_instances(self, instance_ids: list[int]):
        """Notify the router about resumed instances.

        Args:
            instance_ids (List[int]): List of instance IDs that are resumed.
        """
        for instance_id in instance_ids:
            self.currently_paused_instance_ids.discard(instance_id)

    @staticmethod
    def _log_migration_interrupt(request_id: Any, overhead: dict[str, Any]) -> None:
        psrl_logger.info(
            "[ELASTIC_OVERHEAD] operation=request_migration_interrupt "
            "scope=post_scale_up_rebalance role=Rollout migration_id=%s decision_id=%s "
            "request_id=%s source_instance=%s selected_count=%s interrupt_s=%.6f "
            "interrupt_scope=abort_to_requeue",
            overhead.get("migration_id"),
            overhead.get("decision_id"),
            request_id,
            overhead.get("source_instance_id"),
            overhead.get("selected_count"),
            overhead["interrupt_s"],
        )

    def _mark_migration_requeued(self, request_id: Any, *, defer_log: bool = False) -> None:
        overhead = self._migration_overhead.mark_requeued(request_id)
        if overhead is None or defer_log:
            return
        self._log_migration_interrupt(request_id, overhead)

    def _exclusive_rebalance_active(self) -> bool:
        return bool(
            getattr(self, "_exclusive_rebalance_migration_queue_enabled", False)
            and getattr(self, "_rebalance_pending_request_ids", set())
        )

    def _exclusive_rebalance_ready_to_dispatch(self) -> bool:
        return bool(
            self._exclusive_rebalance_active()
            and getattr(self, "_rebalance_requests_to_route", ())
            and not getattr(self, "_rebalance_dispatching_request_ids", set())
            and not getattr(self, "_pause_routing", False)
        )

    def _wake_routing_loop(self) -> None:
        """Wake the routing loop without waiting for its periodic poll."""
        event = getattr(self, "_routing_wakeup_event", None)
        loop = getattr(self, "_routing_event_loop", None)
        if event is None or loop is None or loop.is_closed():
            return
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            event.set()
        else:
            loop.call_soon_threadsafe(event.set)

    def _notify_exclusive_rebalance_completed(self) -> None:
        """Release waiters that may run on a different Ray concurrency-group loop."""
        waiters = getattr(self, "_rebalance_completion_waiters", set())
        for waiter in tuple(waiters):
            try:
                waiter.set_result(None)
            except concurrent.futures.InvalidStateError:
                pass
        waiters.clear()

    async def _wait_for_routing_iteration(self, timeout_s: float) -> None:
        """Keep normal polling while allowing rebalance dispatch to wake immediately."""
        event = getattr(self, "_routing_wakeup_event", None)
        if event is None:
            await asyncio.sleep(timeout_s)
            return
        event.clear()
        # Closing the clear/check race is important: settlement may have
        # happened just before this iteration reached the wait boundary.
        if self._exclusive_rebalance_ready_to_dispatch():
            return
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
        except TimeoutError:
            pass

    def _start_exclusive_rebalance(self, instance_to_uids: dict, migration_context: dict) -> None:
        if not getattr(self, "_exclusive_rebalance_migration_queue_enabled", False):
            return
        request_ids = {
            canonical_psrl_request_id(uid)
            for uids in (instance_to_uids or {}).values()
            for uid in (uids if isinstance(uids, (list, tuple, set)) else [uids])
            if uid is not None
        }
        # Legacy ratio-based rebalance has no simulated destination. Keep its
        # existing routing behavior even when the exclusive queue is enabled.
        request_ids.intersection_update(self._planned_migration_destinations)
        if not request_ids:
            return
        self._rebalance_pending_request_ids.update(request_ids)
        psrl_logger.info(
            "Exclusive rollout rebalance started: migration_id=%s pending=%d normal_pending=%d",
            migration_context.get("migration_id"),
            len(self._rebalance_pending_request_ids),
            self.requests_to_route.size(),
        )

    def _finish_exclusive_rebalance_request(self, request_id: Any, *, reason: str) -> None:
        request_key = canonical_psrl_request_id(request_id)
        if request_key not in getattr(self, "_rebalance_pending_request_ids", set()):
            return
        self._rebalance_pending_request_ids.discard(request_key)
        self._rebalance_queued_request_ids.discard(request_key)
        self._rebalance_dispatching_request_ids.discard(request_key)
        self._planned_migration_destinations.pop(request_key, None)
        psrl_logger.info(
            "Exclusive rollout rebalance request settled: request=%s reason=%s remaining=%d",
            request_key,
            reason,
            len(self._rebalance_pending_request_ids),
        )
        if not self._rebalance_pending_request_ids:
            psrl_logger.info("Exclusive rollout rebalance completed; resuming normal routing.")
            self._notify_exclusive_rebalance_completed()
            self._wake_routing_loop()
        elif self._exclusive_rebalance_ready_to_dispatch():
            self._wake_routing_loop()

    def _enqueue_exclusive_rebalance_request(self, request: DataProto) -> bool:
        if not self._exclusive_rebalance_active():
            return False
        request_id = request.non_tensor_batch["uid"][0]
        request_key = canonical_psrl_request_id(request_id)
        if request_key not in self._rebalance_pending_request_ids:
            return False
        if request_key not in self._planned_migration_destinations:
            psrl_logger.error(
                "Exclusive rollout rebalance lost planned destination for request %s; "
                "returning it to normal routing.",
                request_key,
            )
            self._finish_exclusive_rebalance_request(request_key, reason="missing_destination")
            return False
        if request_key in self._rebalance_queued_request_ids:
            return True
        self._rebalance_requests_to_route.append(request)
        self._rebalance_queued_request_ids.add(request_key)
        psrl_logger.info(
            "Queued interrupted rollout request for exclusive rebalance: request=%s destination=%s fifo_depth=%d",
            request_key,
            self._planned_migration_destinations[request_key],
            len(self._rebalance_requests_to_route),
        )
        self._wake_routing_loop()
        return True

    def _exclusive_rebalance_destination(self, request: DataProto) -> int | None:
        request_key = canonical_psrl_request_id(request.non_tensor_batch["uid"][0])
        destination = self._planned_migration_destinations.get(request_key)
        if destination is None:
            return None
        destination = int(destination)
        if not 0 <= destination < self.rollout_wg_size - self.n_validate_instances:
            return None
        if destination in self.currently_paused_instance_ids:
            return None
        return destination

    async def _dispatch_exclusive_rebalance_requests(self) -> None:
        if self._rebalance_dispatching_request_ids:
            return
        while self._rebalance_requests_to_route and not self._pause_routing:
            request = self._rebalance_requests_to_route[0]
            request_id = request.non_tensor_batch["uid"][0]
            request_key = canonical_psrl_request_id(request_id)
            if await self.ps_manager_handle.check_aborted_requests.remote(request_id, remove=True):
                self._rebalance_requests_to_route.popleft()
                self._rebalance_queued_request_ids.discard(request_key)
                self._migration_overhead.discard(request_key)
                self._finish_exclusive_rebalance_request(request_key, reason="request_aborted")
                self._set_result(request_id, None)
                continue
            destination = self._exclusive_rebalance_destination(request)
            if destination is None:
                planned_destination = self._planned_migration_destinations.get(request_key)
                self._rebalance_requests_to_route.popleft()
                self._rebalance_queued_request_ids.discard(request_key)
                self._migration_overhead.discard(request_key)
                self._planned_migration_counters["fallback"] += 1
                self._finish_exclusive_rebalance_request(request_key, reason="fallback_normal_routing")

                try:
                    fallback_destination = await self._choose_new_rollout_instance(request)
                except Exception:
                    self.requests_to_route.put(request)
                    psrl_logger.exception(
                        "Exclusive rollout rebalance normal fallback failed; request=%s destination=%s "
                        "requeued to normal routing; migration timing discarded.",
                        request_key,
                        planned_destination,
                    )
                    continue
                if fallback_destination is None:
                    self.requests_to_route.put(request)
                    psrl_logger.warning(
                        "Exclusive rollout rebalance destination unreachable; request=%s destination=%s "
                        "normal strategy found no route, requeued to normal routing.",
                        request_key,
                        planned_destination,
                    )
                    continue

                self.incomplete_request_to_instance[request_id] = fallback_destination
                task = asyncio.create_task(self._route_single_request(request, fallback_destination))
                task.add_done_callback(lambda future: future.result())
                self._is_routing = True
                psrl_logger.warning(
                    "Exclusive rollout rebalance destination unreachable; request=%s destination=%s "
                    "fell back to normal destination=%s; migration timing discarded.",
                    request_key,
                    planned_destination,
                    fallback_destination,
                )
                continue
            self._rebalance_requests_to_route.popleft()
            self._rebalance_queued_request_ids.discard(request_key)
            self._rebalance_dispatching_request_ids.add(request_key)
            self._planned_migration_counters["forced"] += 1
            force_route = getattr(self.route_strategy, "force_route_unchecked", None)
            if force_route is not None:
                destination = int(force_route(request, destination))
            self.incomplete_request_to_instance[request_id] = destination
            task = asyncio.create_task(self._route_single_request(request, destination))
            task.add_done_callback(lambda future: future.result())
            self._is_routing = True
            # Preserve FIFO dispatch order, including the asynchronous status
            # update that marks the request as redispatched.
            return

    @ray.method(concurrency_group="control")
    def mark_migration_requests(self, instance_to_uids: dict, migration_context: dict) -> None:
        """Start distributed overhead tracking before selected requests are aborted."""
        self._migration_overhead.mark_batch(instance_to_uids, migration_context)
        self._start_exclusive_rebalance(instance_to_uids, migration_context)

    @ray.method(concurrency_group="transition_wait")
    async def wait_for_exclusive_rebalance(self) -> None:
        """Wait until all accepted exclusive-rebalance requests are redispatched."""
        if not self._exclusive_rebalance_active():
            return
        waiter: concurrent.futures.Future[None] = concurrent.futures.Future()
        waiters = getattr(self, "_rebalance_completion_waiters", None)
        if waiters is None:
            waiters = set()
            self._rebalance_completion_waiters = waiters
        waiters.add(waiter)
        # Close the registration/completion race without binding an
        # asyncio.Event to either Ray concurrency-group event loop.
        if not self._exclusive_rebalance_active():
            waiters.discard(waiter)
            return
        try:
            await asyncio.wrap_future(waiter)
        finally:
            waiters.discard(waiter)

    @ray.method(concurrency_group="control")
    def get_inflight_rollout_request_ids(self) -> dict[int, list[int]]:
        """Return in-flight training-rollout requests grouped by instance."""
        return filter_rollout_request_ids(self.instance_to_inflight_request_ids, self._request_is_validate)

    @ray.method(concurrency_group="control")
    def mark_periodic_interrupt_requests(self, request_ids: list[int]) -> int:
        """Mark requests interrupted by the disaggregated experiment for rerouting."""
        marked = 0
        for request_id in request_ids:
            request_id = int(request_id)
            if not self._request_is_validate.get(request_id, False) and request_id in self.request_futures:
                self._force_reroute_request_ids.add(request_id)
                marked += 1
        return marked

    @ray.method(concurrency_group="control")
    def prepare_request_migrations(
        self,
        request_migrations: list[dict],
        migration_context: dict | None = None,
    ) -> dict:
        """Validate a plan and atomically arm migration tracking before abort."""
        instance_to_uids: dict[int, list[str]] = {}
        accepted = 0
        skipped = 0
        skip_reasons: dict[str, int] = {}
        skip_samples: dict[str, list[str]] = {}
        source_by_uid = {
            canonical_psrl_request_id(request_id): int(instance_id)
            for instance_id, request_ids in self.instance_to_inflight_request_ids.items()
            for request_id in request_ids
        }
        future_by_uid = {
            str(request_id): request_future
            for request_id, request_future in self.request_futures.items()
        }

        def record_skip(reason: str, request_id: str) -> None:
            nonlocal skipped
            skipped += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            samples = skip_samples.setdefault(reason, [])
            if len(samples) < 3:
                samples.append(request_id)

        for migration in request_migrations or []:
            engine_request_id = str(migration.get("request_id", ""))
            logical_request_id = canonical_psrl_request_id(engine_request_id)
            try:
                source = int(migration["source_instance_id"])
                destination = int(migration["destination_instance_id"])
            except (KeyError, TypeError, ValueError):
                record_skip("invalid_migration", engine_request_id)
                continue
            if not engine_request_id:
                record_skip("invalid_request_id", engine_request_id)
                continue
            current_source = source_by_uid.get(logical_request_id)
            future = future_by_uid.get(logical_request_id)
            if future is None:
                record_skip("request_missing", engine_request_id)
                continue
            if future.done():
                record_skip("request_completed", engine_request_id)
                continue
            if current_source is None:
                record_skip("request_not_inflight", engine_request_id)
                continue
            if current_source != source:
                record_skip("source_changed", engine_request_id)
                continue
            if not 0 <= destination < self.rollout_wg_size - self.n_validate_instances:
                record_skip("invalid_destination", engine_request_id)
                continue
            self._planned_migration_destinations[logical_request_id] = destination
            # Workers abort by scheduler/vLLM ID, while the router consumes the
            # destination intent later by logical UID after the request returns.
            instance_to_uids.setdefault(source, []).append(engine_request_id)
            accepted += 1
        self._planned_migration_counters["accepted"] += accepted
        self._planned_migration_counters["skipped"] += skipped
        self._planned_migration_counters["planned"] += len(request_migrations or [])
        if migration_context is not None:
            self._migration_overhead.mark_batch(instance_to_uids, migration_context)
            self._start_exclusive_rebalance(instance_to_uids, migration_context)
        psrl_logger.info(
            "Prepared rollout request migrations: planned=%d accepted=%d skipped=%d "
            "skip_reasons=%s skip_samples=%s sources=%s",
            len(request_migrations or []),
            accepted,
            skipped,
            skip_reasons,
            skip_samples,
            sorted(instance_to_uids),
        )
        return {
            "instance_to_uids": instance_to_uids,
            "planned": len(request_migrations or []),
            "accepted": accepted,
            "skipped": skipped,
            "skip_reasons": skip_reasons,
        }

    async def _choose_new_rollout_instance(self, request: DataProto) -> int:
        """Select the best rollout instance for handling the generation request.

        Args:
            request (DataProto): The request to be routed.

        Returns:
            int: Index of the selected rollout instance.
        """
        # Ensure the whole routing process is atomic from the PS manager side
        # psrl_logger.info(f"Choosing new rollout instance for request {request.non_tensor_batch['uid'][0]}")
        request_id = request.non_tensor_batch["uid"][0]
        is_validate = request.meta_info.get("validate", False)
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        assert "version_tag" in request.non_tensor_batch and request.non_tensor_batch["version_tag"][0] is not None, (
            "Request must have 'version_tag' for routing and it must not be None"
        )
        needed_model_version = request.non_tensor_batch["version_tag"][0]
        requested_version = int(needed_model_version)
        is_partial = "raw_response_ids" in request.non_tensor_batch
        previous_instance = None
        if "rollout_instance_id" in request.non_tensor_batch:
            previous_instance = int(request.non_tensor_batch["rollout_instance_id"][0])

        # 1. Filter the rollout instances that are not paused and can tolerate the needed staleness of the request
        # This guarantees that the gen worker will have no ahead-of-time version tag when generating
        if self.config.psrl.fuse_rollout_with_validate:
            available_instance_ids = set(range(self.rollout_wg_size))
        else:
            # If not fusing rollout with validate, separate the instance IDs for rollout and validate
            available_instance_ids = set(
                range(self.rollout_wg_size - self.n_validate_instances)
                if not is_validate
                else range(self.rollout_wg_size - self.n_validate_instances, self.rollout_wg_size)
            )
        available_instance_ids = available_instance_ids - self.currently_paused_instance_ids
        candidates = [
            i
            for i, version in self.instance_to_version_after_sync.items()
            if i in available_instance_ids and version >= needed_model_version
        ]
        fallback_candidates = list(candidates)
        binding_reasons: list[str] = []
        psrl_logger.debug(
            f"Routing candidates of request {request_id} is {candidates}, where "
            f"available instance: {available_instance_ids}, "
            f"instance_to_version: {self.instance_to_version_after_sync}"
        )

        # 2. If forbidden global migration and the request is a partial rollout request,
        # only consider the specific instance for routing
        force_reroute = bool(request.non_tensor_batch.get("_psrl_force_reroute", False))
        if (
            "rollout_instance_id" in request.non_tensor_batch
            and not self.config.psrl.sync_and_mig_strategy.mig.enable
            and not force_reroute
        ):
            old_instance_id = request.non_tensor_batch["rollout_instance_id"][0]
            if old_instance_id in candidates:
                candidates = [old_instance_id]
                binding_reasons.append("rollout_instance_id")
            else:
                # Elastic scale/sleep may invalidate historical rollout_instance_id.
                # Degrade gracefully to current candidate set instead of crashing router loop.
                psrl_logger.warning(
                    (
                        "Old rollout instance %s is not in candidates for request %s; "
                        "fallback to normal routing. candidates=%s"
                    ),
                    old_instance_id,
                    request_id,
                    candidates,
                )

        # 2.5. If request is in sticky session, keep the existing instance
        if (
            self.sticky_session_requests.get(request_id, False)
            and "rollout_instance_id" in request.non_tensor_batch
            and not force_reroute
        ):
            old_instance_id = request.non_tensor_batch["rollout_instance_id"][0]
            if old_instance_id in candidates:
                candidates = [old_instance_id]
                binding_reasons.append("sticky_session")
            else:
                # Sticky binding can also become stale after elastic scaling.
                psrl_logger.warning(
                    (
                        "Sticky session instance %s is not in candidates for request %s; "
                        "fallback to normal routing. candidates=%s"
                    ),
                    old_instance_id,
                    request_id,
                    candidates,
                )

        # 3. If forbidden group sampling on multiple instances, only consider the
        # instance that other requests in the same group are already routed to
        enable_multi_instance_group = self.config.psrl.routing_strategy.enable_group_sampling_on_multi_instances
        if not enable_multi_instance_group:
            group_request_instance_ids = [
                instance_id
                for incomplete_request_id, instance_id in self.incomplete_request_to_instance.items()
                if incomplete_request_id // rollout_n == request_id // rollout_n
            ]
            if len(group_request_instance_ids) > 0:
                first_instance = group_request_instance_ids[0]
                assert all(instance_id == first_instance for instance_id in group_request_instance_ids), (
                    f"All requests in the same group must be routed to "
                    f"the same instance, but found different instances: "
                    f"{group_request_instance_ids}"
                )
                group_instance = group_request_instance_ids[0]
                assert group_instance in candidates, (
                    f"Group request instance {group_instance} of request {request_id} is not in the candidates "
                    f"{candidates}, instance versions: {self.instance_to_version_after_sync}, "
                    f"needed model version: {needed_model_version}"
                )
                candidates = [group_instance]

        async def _route_with_candidates(
            route_candidates: list[int],
            route_reason: str,
            *,
            force_destination: bool = False,
        ) -> int | None:
            # Filter the rollout instances that can reserve the request for the
            # current instance model version. This is only used when the needed
            # model version is -1 (i.e. new request).
            route_candidates = list(route_candidates)
            if needed_model_version == -1:
                all_candidate_model_versions = list(
                    set([self.instance_to_version_after_sync[candidate] for candidate in route_candidates])
                )
                if len(all_candidate_model_versions) == 0:
                    return None
                can_reserve_results = await self.ps_manager_handle.can_reserve_request.remote(
                    request_id, all_candidate_model_versions, is_validate=is_validate
                )
                route_candidates = [
                    candidate
                    for candidate in route_candidates
                    if can_reserve_results[
                        all_candidate_model_versions.index(self.instance_to_version_after_sync[candidate])
                    ]
                ]

            # Provide the indicator list to sort candidates for the route strategy.
            candidate_indicator_list = []
            if self.config.psrl.routing_strategy.candidate_sort_indicator == "version":
                for candidate in route_candidates:
                    version = self.instance_to_version_after_sync[candidate]
                    # New request: sort by version in descending order.
                    # Existing request: sort by version in ascending order.
                    if needed_model_version == -1:
                        version_indicator = -version
                    else:
                        version_indicator = version
                    candidate_indicator_list.append(version_indicator)
            elif self.config.psrl.routing_strategy.candidate_sort_indicator == "reserve_capability":
                # Use the (reserve_indicator, version) pair as the final indicator.
                all_candidate_model_versions = list(
                    set([self.instance_to_version_after_sync[candidate] for candidate in route_candidates])
                )
                if len(all_candidate_model_versions) == 0:
                    return None
                indicator_results = await self.ps_manager_handle.get_reserve_indicator.remote(
                    request_id, all_candidate_model_versions, is_validate=is_validate
                )
                for candidate in route_candidates:
                    version = self.instance_to_version_after_sync[candidate]
                    if needed_model_version == -1:
                        version_indicator = -version
                    else:
                        version_indicator = version
                    reserve_indicator = indicator_results[all_candidate_model_versions.index(version)]
                    candidate_indicator_list.append((reserve_indicator, version_indicator))
            else:
                raise ValueError(
                    f"Invalid candidate sort indicator: {self.config.psrl.routing_strategy.candidate_sort_indicator}"
                )

            route_kwargs = {"candidate_indicator_list": candidate_indicator_list}
            if force_destination and len(route_candidates) == 1 and hasattr(
                self.route_strategy,
                "force_route",
            ):
                chosen = self.route_strategy.force_route(request, route_candidates[0])
            else:
                chosen = self.route_strategy.route(
                    request,
                    candidates=route_candidates,
                    route_kwargs=route_kwargs,
                )
            if chosen is None:
                psrl_logger.debug(
                    "No rollout instance selected for request %s with %s candidates=%s.",
                    request_id,
                    route_reason,
                    route_candidates,
                )
            return chosen

        # 4-6. A prepared migration may force one destination, but only while it
        # remains version/reserve/capacity compatible. Invalid intents are
        # cleared immediately and normal strategy selection proceeds.
        migration_key = str(request_id)
        planned_destination = self._planned_migration_destinations.get(migration_key)
        chosen_rollout_instance = None
        route_reason = "none"
        if planned_destination is not None:
            if int(planned_destination) in fallback_candidates:
                chosen_rollout_instance = await _route_with_candidates(
                    [int(planned_destination)],
                    "planned_migration",
                    force_destination=True,
                )
                if chosen_rollout_instance is not None:
                    route_reason = "planned_migration"
            self._planned_migration_destinations.pop(migration_key, None)
            if chosen_rollout_instance is None:
                self._planned_migration_counters["fallback"] += 1
                psrl_logger.info(
                    "Planned rollout migration target invalid; request=%s destination=%s "
                    "fallback_candidates=%s. Falling back immediately.",
                    request_id,
                    planned_destination,
                    fallback_candidates,
                )
            else:
                self._planned_migration_counters["forced"] += 1
                psrl_logger.info(
                    "Forced planned rollout migration: request=%s destination=%s",
                    request_id,
                    chosen_rollout_instance,
                )

        if chosen_rollout_instance is None:
            chosen_rollout_instance = await _route_with_candidates(candidates, "primary")
            if chosen_rollout_instance is not None:
                route_reason = "primary"
        if (
            chosen_rollout_instance is None
            and binding_reasons
            and set(fallback_candidates) != set(candidates)
        ):
            psrl_logger.warning(
                (
                    "Request %s could not route with bound candidates=%s "
                    "(reasons=%s); retrying with version-compatible fallback candidates=%s."
                ),
                request_id,
                candidates,
                binding_reasons,
                fallback_candidates,
            )
            chosen_rollout_instance = await _route_with_candidates(fallback_candidates, "fallback")
            if chosen_rollout_instance is not None:
                route_reason = "fallback"

        # 7. If not None, the request is routed to the chosen rollout instance
        if chosen_rollout_instance is not None:
            request.non_tensor_batch.pop("_psrl_force_reroute", None)
            # Allocate the version tag and reserve the request for the chosen
            # rollout instance if the request is not routed before
            not_routed_before = "rollout_instance_id" not in request.non_tensor_batch
            if not_routed_before:
                needed_model_version = self.instance_to_version_after_sync[chosen_rollout_instance]
                request.non_tensor_batch["version_tag"] = np.array([needed_model_version], dtype=int)
                await self.ps_manager_handle.reserve_rollout_instance_requests.remote(
                    rollout_instance_ids=chosen_rollout_instance,
                    request_ids=request_id,
                    model_versions=needed_model_version,
                    is_validate=is_validate,
                )
            # Otherwise, the request is already reserved
            # Only need to update the request instance id
            else:
                await self.ps_manager_handle.update_request_instance_id.remote(
                    request_id=request_id,
                    new_instance_id=chosen_rollout_instance,
                    is_validate=is_validate,
                )
        else:
            pass

        self._log_partial_rollout_route(
            request_id=int(request_id),
            is_partial=is_partial,
            previous_instance=previous_instance,
            requested_version=requested_version,
            chosen_instance=chosen_rollout_instance,
            route_reason=route_reason,
            candidates=fallback_candidates,
        )

        return chosen_rollout_instance

    # TODO(lhy): move this back to router again
    # since log_prob no need to transfered to vllm rollout engine many times (partial rollout)
    @deprecated("It is moved to the `post_process_outputs` inside vllm rollout now")
    def _consolidate_responses(
        self,
        prompts: DataProto,
        vllm_outputs,
    ) -> DataProto:
        """Consolidate VLLM generation outputs with input prompts.

        Args:
            prompts (DataProto): Original input prompts.
            vllm_outputs: Generation outputs from VLLM engine.

        Returns:
            DataProto: Consolidated response data.
        """
        if not isinstance(vllm_outputs, list):
            vllm_outputs = [vllm_outputs]
        assert len(vllm_outputs) == len(prompts), "Mismatched batch size between prompts and VLLM outputs."

        batch_size = len(prompts)
        non_tensor_batch = prompts.non_tensor_batch

        response_ids_list = []
        response_len_list = []
        interrupted_list = []
        all_log_prob_list = []

        for i in range(batch_size):
            vllm_output = vllm_outputs[i]
            assert len(vllm_output.outputs) == 1, "RolloutRouter only supports single request generation."

            response_ids = vllm_output.outputs[0].token_ids
            response_len = len(response_ids)
            interrupted = vllm_output.outputs[0].finish_reason == "abort"

            response_ids_list.append(response_ids)
            response_len_list.append(response_len)
            interrupted_list.append(interrupted)

            log_prob_list = []
            # if inference logprobs is required, we need to collect the log probabilities
            if (
                self.config.psrl.log_prob.enable_rollout_engine_log_prob
                and hasattr(vllm_output.outputs[0], "logprobs")
                and vllm_output.outputs[0].logprobs is not None
            ):
                if self.config.psrl.partial_rollout.interrupt_as_prompt:
                    curr_response_len = non_tensor_batch.get("response_unpadded_len", 0)
                    # Collect log probs only when the request finished normally
                    # The response log probs are collected in two parts:
                    # 1. The log probs of the accumulated response tokens (in current prompt tokens)
                    # 2. The log probs of the current response tokens
                    if not interrupted and curr_response_len > 0:
                        # partial response log probs from prompt log probs
                        prompt_token_ids = vllm_output.prompt_token_ids
                        for i, logprob in enumerate(vllm_output.prompt_logprobs[-curr_response_len:]):
                            log_prob_list.append(logprob[prompt_token_ids[i - curr_response_len]].logprob)
                        # new response log probs from decode log probs
                        for i, logprob in enumerate(vllm_output.outputs[0].logprobs):
                            log_prob_list.append(logprob[response_ids[i]].logprob)
                else:
                    # Response log probs from decode log probs
                    for i, logprob in enumerate(vllm_output.outputs[0].logprobs):
                        log_prob_list.append(logprob[response_ids[i]].logprob)
            all_log_prob_list.append(log_prob_list)

        # Consolidate batch results
        if "raw_response_ids" in non_tensor_batch:
            raw_response_ids = non_tensor_batch.pop("raw_response_ids")
            raw_response_ids = np.fromiter(raw_response_ids.tolist(), dtype=object)
        else:
            raw_response_ids = np.fromiter(([] for _ in range(batch_size)), dtype=object)

        raw_response_ids = raw_response_ids + np.fromiter(response_ids_list, dtype=object)
        non_tensor_batch["raw_response_ids"] = raw_response_ids

        if "response_unpadded_len" in non_tensor_batch:
            curr_response_unpadded_len = non_tensor_batch["response_unpadded_len"]
        else:
            curr_response_unpadded_len = [0] * batch_size
        response_unpadded_len = [curr_response_unpadded_len[i] + response_len_list[i] for i in range(batch_size)]
        non_tensor_batch["response_unpadded_len"] = np.array(response_unpadded_len, dtype=int)
        non_tensor_batch["interrupted"] = np.array(interrupted_list, dtype=bool)

        # Update rollout_log_probs
        if self.config.psrl.log_prob.enable_rollout_engine_log_prob:
            if "rollout_log_probs" in non_tensor_batch:
                curr_rollout_log_probs = non_tensor_batch.pop("rollout_log_probs")
                curr_rollout_log_probs = np.fromiter(curr_rollout_log_probs.tolist(), dtype=object)
            else:
                curr_rollout_log_probs = np.fromiter(([] for _ in range(batch_size)), dtype=object)
            curr_rollout_log_probs = curr_rollout_log_probs + np.fromiter(all_log_prob_list, dtype=object)
            non_tensor_batch["rollout_log_probs"] = curr_rollout_log_probs

        batch = TensorDict(
            {
                "input_ids": prompts.batch["input_ids"],
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    @rollout_trace_op
    async def generate_async(
        self,
        request: DataProto,
    ) -> DataProto:
        """Asynchronously generate response for a single request.

        Args:
            request (DataProto): Single generation request.

        Returns:
            DataProto or None: Generated result or None if request is invalid.
        """
        assert len(request) == 1, "RolloutRouter only supports single request generation."
        if self.scheduler_task is None:
            self._routing_event_loop = asyncio.get_running_loop()
            self._routing_wakeup_event = asyncio.Event()
            if self.config.psrl.routing_strategy.enable_multi_priority_queue:
                task_coro = self._multi_priority_queue_routing_loop()
                self.scheduler_task = asyncio.create_task(task_coro)
            else:
                task_coro = self._single_priority_queue_routing_loop()
                self.scheduler_task = asyncio.create_task(task_coro)
            # To avoid silent error in async tasks
            self.scheduler_task.add_done_callback(lambda f: f.result())
            psrl_logger.info("Started routing loop")

        request_id = request.non_tensor_batch["uid"][0]
        is_validate = request.meta_info.get("validate", False)
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            [request_id],
            PSRL_RequestStatus.ROLLOUT_ROUTING,
            is_validate=is_validate,
        )
        if not update_status_success[0]:
            # Means the request is aborted
            return None

        # Create a future to track this request's completion
        result_future = asyncio.Future()
        # Store the future in a way that the scheduler can access it
        self.request_futures[request_id] = result_future
        self._request_is_validate[int(request_id)] = bool(is_validate)
        # Add request to priority queue
        self.requests_to_route.put(request)
        # psrl_logger.info(f"Adding request {request_id} to priority queue")
        # Wait for the request to be processed
        with log_dual_events(
            f"Routing request {request_id} and waiting for it to be processed",
            psrl_logger,
            level=logging.DEBUG,
            event_type=EventType.GEN,
        ):
            result = await result_future
        # Clean up the future
        self.request_futures.pop(request_id)
        return result

    def _set_result(self, request_id: int, result: DataProto | None):
        """Set the result for a request.

        Args:
            request_id: The id of the request.
            result: The result to set.
        """
        assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
        assert not self.request_futures[request_id].done(), f"Request {request_id} should not be done"
        self._finish_exclusive_rebalance_request(request_id, reason="terminal")
        self.request_futures[request_id].set_result(result)
        self.incomplete_request_to_instance.pop(request_id, None)
        self.sticky_session_requests.pop(request_id, None)
        self._request_is_validate.pop(int(request_id), None)
        self._force_reroute_request_ids.discard(int(request_id))

    def is_routing(self) -> bool:
        """Check if the router is currently routing requests."""
        return self._is_routing

    @ray.method(concurrency_group="monitor")
    def get_pending_request_count(self) -> int:
        """Return current number of requests waiting in router queue."""
        t0 = time.monotonic()
        log_elastic_rm_backlog_diag(psrl_logger, "stage=RolloutRouter_enter")
        n = int(self.requests_to_route.size()) + len(self._rebalance_requests_to_route)
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutRouter_exit pending=%d body_s=%.6f",
            n,
            time.monotonic() - t0,
        )
        return n

    @ray.method(concurrency_group="monitor")
    def get_pending_request_summary(self, top_t: int | None = None) -> dict[str, int]:
        """Return count and token load for the leading waiting requests."""
        t0 = time.monotonic()
        log_elastic_rm_backlog_diag(psrl_logger, "stage=RolloutRouter_summary_enter")
        total_pending = int(self.requests_to_route.size()) + len(self._rebalance_requests_to_route)
        limit = total_pending if top_t is None else max(0, min(total_pending, int(top_t)))
        requests = list(self._iter_pending_requests_in_route_order(limit))
        summary = {
            "pending": total_pending,
            "count": len(requests),
            "total_tokens": sum(self._request_token_num(request) for request in requests),
        }
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutRouter_summary_exit pending=%d count=%d total_tokens=%d body_s=%.6f",
            summary["pending"],
            summary["count"],
            summary["total_tokens"],
            time.monotonic() - t0,
        )
        return summary

    @staticmethod
    def _candidate_evaluation_scalar(value):
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    def _candidate_evaluation_queue_priority(self, request: DataProto):
        indicator = RequestSortIndicator(
            self.config.psrl.routing_strategy.request_sort_indicator
        )
        if indicator == RequestSortIndicator.SHORT_LENGTH:
            request_priority = get_priority_by_version_and_token_num(
                request,
                self.staleness,
                True,
            )
        elif indicator == RequestSortIndicator.LONG_LENGTH:
            request_priority = get_priority_by_version_and_token_num(
                request,
                self.staleness,
                False,
            )
        else:
            request_priority = get_priority_by_version_and_id(
                request,
                self.staleness,
            )
        if isinstance(self.requests_to_route, MultiPriorityRequestQueue):
            return (
                get_priority_by_version(request, self.staleness),
                request_priority,
            )
        return (request_priority,)

    async def _candidate_evaluation_request_row(
        self,
        request: DataProto,
        *,
        route_order: int,
        source_instance_id: int | None,
        is_waiting: bool,
        candidate_instance_versions: dict[int, int],
        evaluation_request_id: str | None = None,
    ) -> dict:
        non_tensor_batch = getattr(request, "non_tensor_batch", {}) or {}
        request_id = self._candidate_evaluation_scalar(non_tensor_batch.get("uid", ""))
        needed_version = int(
            self._candidate_evaluation_scalar(non_tensor_batch.get("version_tag", -1))
        )
        is_validate = bool(getattr(request, "meta_info", {}).get("validate", False))
        if self.config.psrl.fuse_rollout_with_validate:
            available_ids = list(range(self.rollout_wg_size))
        elif is_validate:
            available_ids = list(
                range(
                    self.rollout_wg_size - self.n_validate_instances,
                    self.rollout_wg_size,
                )
            )
        else:
            available_ids = list(range(self.rollout_wg_size - self.n_validate_instances))
        fallback_ids = [
            instance_id
            for instance_id in available_ids
            if candidate_instance_versions.get(instance_id, -1) >= needed_version
        ]
        if needed_version == -1 and fallback_ids:
            versions = sorted(
                {candidate_instance_versions[instance_id] for instance_id in fallback_ids}
            )
            can_reserve = await self.ps_manager_handle.can_reserve_request.remote(
                request_id,
                versions,
                is_validate=is_validate,
            )
            allowed_versions = {
                version for version, allowed in zip(versions, can_reserve, strict=True) if allowed
            }
            fallback_ids = [
                instance_id
                for instance_id in fallback_ids
                if candidate_instance_versions[instance_id] in allowed_versions
            ]

        eligible_ids = list(fallback_ids)
        old_instance = self._candidate_evaluation_scalar(
            non_tensor_batch.get("rollout_instance_id")
        )
        if old_instance is not None:
            old_instance = int(old_instance)
            if (
                not self.config.psrl.sync_and_mig_strategy.mig.enable
                or self.sticky_session_requests.get(request_id, False)
            ) and old_instance in eligible_ids:
                eligible_ids = [old_instance]

        priority_by_id: dict[int, object] = {}
        priority_ids = list(fallback_ids)
        if priority_ids:
            if self.config.psrl.routing_strategy.candidate_sort_indicator == "version":
                for instance_id in priority_ids:
                    version = candidate_instance_versions[instance_id]
                    priority_by_id[instance_id] = -version if needed_version == -1 else version
            else:
                versions = sorted(
                    {candidate_instance_versions[instance_id] for instance_id in priority_ids}
                )
                reserve_indicators = await self.ps_manager_handle.get_reserve_indicator.remote(
                    request_id,
                    versions,
                    is_validate=is_validate,
                )
                indicator_by_version = dict(zip(versions, reserve_indicators, strict=True))
                for instance_id in priority_ids:
                    version = candidate_instance_versions[instance_id]
                    version_indicator = -version if needed_version == -1 else version
                    reserve_indicator = self._candidate_evaluation_scalar(
                        indicator_by_version[version]
                    )
                    priority_by_id[instance_id] = (reserve_indicator, version_indicator)

        return {
            "request_id": str(request_id if evaluation_request_id is None else evaluation_request_id),
            "seq_len": self._request_token_num(request),
            "source_instance_id": source_instance_id,
            "is_waiting": bool(is_waiting),
            "route_order": int(route_order),
            "routing_priority": self._candidate_evaluation_queue_priority(request),
            "eligible_instance_ids": eligible_ids,
            "fallback_instance_ids": fallback_ids,
            "candidate_priorities": [
                [instance_id, priority_by_id.get(instance_id, 0)]
                for instance_id in priority_ids
            ],
        }

    @ray.method(concurrency_group="monitor")
    async def get_candidate_evaluation_snapshot(self, top_t: int | None = None) -> dict:
        """Return a compact, read-only snapshot for request-level simulation."""
        if not hasattr(self, "route_strategy"):
            raise RuntimeError("rollout route strategy is not initialized")
        current_ps_model_version = int(
            await self.ps_manager_handle.get_ps_model_version.remote(
                debug_info="candidate_evaluation_snapshot"
            )
        )
        paused_instance_ids = frozenset(
            int(instance_id) for instance_id in self.currently_paused_instance_ids
        )
        candidate_instance_versions = resolve_candidate_model_versions(
            self.instance_to_version_after_sync,
            paused_instance_ids,
            current_ps_model_version,
        )
        total_queued = int(self.requests_to_route.size()) + len(self._rebalance_requests_to_route)
        pending_requests = [
            request
            for request in self._iter_pending_requests_in_route_order(total_queued)
            if not bool((getattr(request, "meta_info", {}) or {}).get("validate", False))
        ]
        total_pending = len(pending_requests)
        limit = total_pending if top_t is None else max(0, min(total_pending, int(top_t)))
        pending_requests = pending_requests[:limit]
        pending_rows = await asyncio.gather(
            *(
                self._candidate_evaluation_request_row(
                    request,
                    route_order=index,
                    source_instance_id=None,
                    is_waiting=True,
                    candidate_instance_versions=candidate_instance_versions,
                )
                for index, request in enumerate(pending_requests)
            )
        )

        instances: list[dict] = []
        inflight_route_order = total_pending
        # Dedicated validation instances share this router but are not registered
        # with ElasticExecutor and must not appear in elastic policy snapshots.
        for instance_id in range(self.n_rollout_instances):
            engine_status = self.route_strategy.instance_to_engine_status.get(instance_id)
            scheduler_stats = (
                engine_status.snapshot.get("scheduler_stats", {})
                if engine_status is not None
                else {}
            )
            prompt_map = {
                str(request_id): int(value)
                for request_id, value in (
                    scheduler_stats.get("req_id_to_prompt_token_num", {}) or {}
                ).items()
            }
            response_map = {
                str(request_id): int(value)
                for request_id, value in (
                    scheduler_stats.get("req_id_to_response_token_num", {}) or {}
                ).items()
            }
            waiting_ids = {
                str(request_id)
                for request_id in scheduler_stats.get("req_id_in_waiting", []) or []
            }
            request_ids = sorted(
                {str(request_id) for request_id in prompt_map}
                | {str(request_id) for request_id in response_map}
            )
            request_rows: list[dict] = []
            for request_key in request_ids:
                logical_request_id = canonical_psrl_request_id(request_key)
                inflight_request = self._candidate_evaluation_inflight_requests.get(logical_request_id)
                if inflight_request is None:
                    request_rows.append(
                        {
                            "request_id": request_key,
                            "seq_len": int(prompt_map.get(request_key, 0))
                            + int(response_map.get(request_key, 0)),
                            "source_instance_id": instance_id,
                            "is_waiting": request_key in waiting_ids,
                            "route_order": inflight_route_order,
                        }
                    )
                else:
                    request_rows.append(
                        await self._candidate_evaluation_request_row(
                            inflight_request,
                            route_order=inflight_route_order,
                            source_instance_id=instance_id,
                            is_waiting=request_key in waiting_ids,
                            candidate_instance_versions=candidate_instance_versions,
                            evaluation_request_id=request_key,
                        )
                    )
                inflight_route_order += 1
            tp_pp = getattr(self.route_strategy, "instance_to_tp_pp", {}).get(instance_id)
            route_model = getattr(self.route_strategy, "cost_model", {}).get(tp_pp, {})
            instances.append(
                {
                    "instance_id": instance_id,
                    "is_awake": instance_id not in paused_instance_ids,
                    "model_version": self.instance_to_version_after_sync.get(instance_id, 0),
                    "candidate_model_version": candidate_instance_versions.get(instance_id, 0),
                    "requests": request_rows,
                    "route_request_count": getattr(
                        self.route_strategy,
                        "instance_to_request_num",
                        {},
                    ).get(instance_id, len(request_rows)),
                    "running_count": getattr(
                        self.route_strategy,
                        "instance_to_running_request_num",
                        {},
                    ).get(instance_id, scheduler_stats.get("num_running_reqs", 0)),
                    "waiting_count": getattr(
                        self.route_strategy,
                        "instance_to_waiting_request_num",
                        {},
                    ).get(instance_id, scheduler_stats.get("num_waiting_reqs", 0)),
                    "token_count": getattr(
                        self.route_strategy,
                        "instance_to_token_num",
                        {},
                    ).get(
                        instance_id,
                        sum(prompt_map.values()) + sum(response_map.values()),
                    ),
                    "max_model_len": getattr(
                        self.route_strategy,
                        "instance_to_max_model_len",
                        {},
                    ).get(instance_id, 2**63 - 1),
                    "route_cost_params": [
                        route_model.get("other_threshold", 0.0),
                        route_model.get("other_latency_b", 0.0),
                        route_model.get("other_latency_k", 1.0),
                        route_model.get("attn_latency_b", 0.0),
                        route_model.get("attn_latency_k", 0.0),
                    ],
                }
            )
        return {
            "role": "Rollout",
            "strategy": (
                "throughput_optimal"
                if self.route_strategy.__class__.__name__
                == "ThroughputOptimalRouteStrategy"
                else self.route_strategy.__class__.__name__.lower()
            ),
            "instances": instances,
            "pending_requests": pending_rows,
            "pending_total": total_pending,
            "current_ps_model_version": current_ps_model_version,
            "max_concurrent_requests": getattr(
                self.route_strategy,
                "max_concurrent_seqs_per_instance",
                None,
            ),
            "delta_throughput_threshold": getattr(
                self.route_strategy,
                "delta_throughput_threshold",
                0.0,
            ),
            "migration_counters": dict(self._planned_migration_counters),
        }

    def _iter_pending_requests_in_route_order(self, limit: int):
        """Yield pending requests in the same priority order used by routing."""
        if limit <= 0:
            return
        yielded = 0
        for request in self._rebalance_requests_to_route:
            yield request
            yielded += 1
            if yielded >= limit:
                return
        if isinstance(self.requests_to_route, MultiPriorityRequestQueue):
            iterator = (request for _, request in self.requests_to_route.iter_all_requests())
        else:
            iterator = self.requests_to_route.iter_priority()
        for request in iterator:
            yield request
            yielded += 1
            if yielded >= limit:
                return

    @staticmethod
    def _request_token_num(request: DataProto) -> int:
        """Best-effort token count for a queued single-sequence request."""
        non_tensor_batch = getattr(request, "non_tensor_batch", {}) or {}
        raw_prompt_ids = non_tensor_batch.get("raw_prompt_ids")
        if raw_prompt_ids is not None:
            prompt_ids = raw_prompt_ids[0] if len(raw_prompt_ids) > 0 else []
            response_len = non_tensor_batch.get("response_unpadded_len", [0])
            if hasattr(response_len, "tolist"):
                response_len = response_len.tolist()
            if isinstance(response_len, (list, tuple, np.ndarray)):
                response_len = response_len[0] if len(response_len) > 0 else 0
            return max(0, len(prompt_ids) + int(response_len))

        batch = getattr(request, "batch", None)
        if batch is None:
            batch = {}
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            if hasattr(attention_mask, "sum"):
                return max(0, int(attention_mask.sum().item()))
            return max(0, int(np.asarray(attention_mask).sum()))
        input_ids = batch.get("input_ids", None)
        if input_ids is None:
            return 0
        shape = getattr(input_ids, "shape", None)
        if shape is not None and len(shape) > 0:
            return max(0, int(shape[-1]))
        return max(0, len(input_ids))

    @ray.method(concurrency_group="control")
    async def pause_routing(self):
        """Interrupt the routing."""
        self._pause_routing = True
        # NOTE(lhy): asyncio lock cannot be shared across concurrency groups (each group has its
        # own event loop). Poll _is_routing instead because a plain bool read/write is safe across threads
        # under CPython's GIL.
        while self._is_routing:
            await asyncio.sleep(self.config.psrl.routing_strategy.check_interval_in_ms / 1000)
        psrl_logger.info("Pausing routing")

    @ray.method(concurrency_group="control")
    async def resume_routing(self):
        """Resume the routing."""
        self._pause_routing = False
        self._wake_routing_loop()
        psrl_logger.info("Resuming routing")

    async def _single_priority_queue_routing_loop(self):
        """Continuous routing loop for a single priority queue.

        This loop processes requests from the single priority queue.
        """
        psrl_logger.info("Started single priority queue routing loop")
        while True:
            self._is_routing = False
            async with (
                AsyncBusyPollingRayLock(self.ps_manager_handle),
            ):
                if self._exclusive_rebalance_active():
                    await self._dispatch_exclusive_rebalance_requests()
                while (
                    not self._exclusive_rebalance_active()
                    and not self.requests_to_route.empty()
                    and not self._pause_routing
                ):
                    self._is_routing = True
                    request = self.requests_to_route.pop()
                    assert request is not None, "Request should not be None in priority queue"
                    request_id = request.non_tensor_batch["uid"][0]
                    assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
                    if await self.ps_manager_handle.check_aborted_requests.remote(request_id, remove=True):
                        self._set_result(request_id, None)
                        continue
                    # time_begin = time.time()
                    new_instance_id = await self._choose_new_rollout_instance(request)
                    # time_end = time.time()
                    # psrl_logger.info(f"Choosing rollout instance for request {request_id} to {new_instance_id} in {time_end - time_begin} seconds") # noqa: E501
                    if new_instance_id is None:
                        # new_instance_id is None indicates that we cannot find a suitable rollout instance
                        # for the request due to the current engine status (e.g.,
                        # version staleness, instance overload).
                        # Need to wait for engine status update to try again:
                        # 1. The overall engine status could be updated by the
                        #    coordinator periodically.
                        # 2. The engine status of the specific instance could be
                        #    updated after one request is added/completed.
                        self.requests_to_route.put(request)
                        break
                    self.incomplete_request_to_instance[request_id] = new_instance_id
                    task_coro = self._route_single_request(request, new_instance_id)
                    task = asyncio.create_task(task_coro)
                    # To avoid silent error in async tasks
                    task.add_done_callback(lambda f: f.result())
            self._is_routing = False
            sleep_time = self.config.psrl.routing_strategy.check_interval_in_ms / 1000
            await self._wait_for_routing_iteration(sleep_time)

    async def _multi_priority_queue_routing_loop(self):
        """Continuous routing loop for multiple priority queues.

        This loop processes requests from the multiple priority queues.
        """
        psrl_logger.info("Started multi priority queue routing loop")
        while True:
            # Process all requests in the multiple priority queues
            self._is_routing = False
            route_num = 0
            begin_time = time.time()
            async with (
                AsyncBusyPollingRayLock(self.ps_manager_handle),
            ):
                if self._exclusive_rebalance_active():
                    await self._dispatch_exclusive_rebalance_requests()
                self.requests_to_route.remove_empty_queues()
                remain_requests = []
                for queue_id, request_queue in self.requests_to_route.iter_queues():
                    if self._exclusive_rebalance_active():
                        break
                    if len(remain_requests) != 0:
                        # Method 1: If the last queue still has requests, we will not process the other queues
                        break
                        # Method 2: Try to process the other queues
                        # remain_requests.clear()
                    while not request_queue.empty() and not self._pause_routing:
                        request = request_queue.pop()
                        assert request is not None, "Request should not be None in priority queue"
                        request_id = request.non_tensor_batch["uid"][0]
                        assert request_id in self.request_futures, f"Request {request_id} should be in request futures"
                        if await self.ps_manager_handle.check_aborted_requests.remote(request_id, remove=True):
                            self._set_result(request_id, None)
                            continue
                        new_instance_id = await self._choose_new_rollout_instance(request)
                        if new_instance_id is None:
                            # new_instance_id is None indicates that we cannot find a suitable rollout instance
                            # for the request due to the current engine status (e.g.,
                            # version staleness, instance overload).
                            # Need to wait for engine status update to try again:
                            # 1. The overall engine status could be updated by the
                            #    coordinator periodically.
                            # 2. The engine status of the specific instance could be
                            #    updated after one request is added/completed.
                            remain_requests.append(request)
                            break
                        self.incomplete_request_to_instance[request_id] = new_instance_id
                        # Create a task to process this request
                        task_coro = self._route_single_request(request, new_instance_id)
                        task = asyncio.create_task(task_coro)
                        # To avoid silent error in async tasks
                        task.add_done_callback(lambda f: f.result())
                        route_num += 1
                    for request in remain_requests:
                        request_queue.put(request)
            self._is_routing = False
            sleep_time = self.config.psrl.routing_strategy.check_interval_in_ms / 1000
            if route_num > 0:
                psrl_logger.debug(
                    f"Routing {route_num} requests in multi priority queue "
                    f"routing loop, time cost: {time.time() - begin_time} seconds"
                )
            await self._wait_for_routing_iteration(sleep_time)

    async def _route_single_request(self, request: DataProto, new_instance_id: int):
        """Route a single request to a rollout instance.

        Args:
            request (DataProto): The request to process.
            new_instance_id (int): The new rollout instance id that the request
                will be routed to.
        """
        # Update request non-tensor batch
        # psrl_logger.info(
        #     f"Routing single request {request.non_tensor_batch['uid'][0]} "
        #     f"to rollout instance {new_instance_id}"
        # )
        request_id = request.non_tensor_batch["uid"][0]
        is_validate = request.meta_info.get("validate", False)
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        request.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
        assert "version_tag" in request.non_tensor_batch, "Request must have 'version_tag' for routing"
        needed_model_version = request.non_tensor_batch["version_tag"][0]
        version_tag_error = (
            "The version tag should not be -1 (new request that is not "
            "allocated a version tag yet when enabled dynamic version tag) "
            "after routing"
        )
        assert needed_model_version != -1, version_tag_error
        # Update request status
        # psrl_logger.info(
        #     f"Updating request {request_id} status to "
        #     f"ROLLOUT_DISPATCHED with version tag {needed_model_version}"
        # )
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            [request_id],
            PSRL_RequestStatus.ROLLOUT_DISPATCHED,
            model_version=request.non_tensor_batch["version_tag"].tolist(),
            is_validate=request.meta_info.get("validate", False),
        )
        # psrl_logger.info(
        #     f"Update request {request_id} status to "
        #     f"ROLLOUT_DISPATCHED success: {update_status_success[0]}"
        # )

        if update_status_success[0]:
            request_key = canonical_psrl_request_id(request_id)
            is_exclusive_redispatch = request_key in getattr(
                self,
                "_rebalance_dispatching_request_ids",
                set(),
            )
            dispatch_overhead = self._migration_overhead.mark_dispatched(
                request_id,
                new_instance_id,
            )
            if dispatch_overhead is not None:
                if is_exclusive_redispatch:
                    self._log_migration_interrupt(request_id, dispatch_overhead)
                psrl_logger.info(
                    "[ELASTIC_OVERHEAD] operation=request_migration_dispatch "
                    "scope=post_scale_up_rebalance role=Rollout migration_id=%s decision_id=%s "
                    "request_id=%s source_instance=%s destination_instance=%s selected_count=%s "
                    "interrupt_s=%.6f network_s=%.6f abort_to_redispatch_s=%.6f "
                    "interrupt_scope=abort_to_requeue network_scope=requeue_to_redispatch",
                    dispatch_overhead.get("migration_id"),
                    dispatch_overhead.get("decision_id"),
                    request_id,
                    dispatch_overhead.get("source_instance_id"),
                    dispatch_overhead.get("destination_instance_id"),
                    dispatch_overhead.get("selected_count"),
                    dispatch_overhead["interrupt_s"],
                    dispatch_overhead["network_s"],
                    dispatch_overhead["abort_to_redispatch_s"],
                )
            self._candidate_evaluation_inflight_requests[str(request_id)] = request
            # Change engine status
            self.route_strategy.push_request(request, new_instance_id)
            # Add request to inflight request ids for the instance
            self.instance_to_inflight_request_ids[new_instance_id].append(request_id)
            self._track_transition_request_started(new_instance_id, request_id)
            if request_key in self._rebalance_dispatching_request_ids:
                self._finish_exclusive_rebalance_request(request_key, reason="redispatched")

            # Set sampling params
            rollout_config = self.config.gen_actor_rollout_ref.rollout
            sampling_params = dict(
                n=1,
                logprobs=0,  # can be set to 0 and let actor to recompute
                temperature=rollout_config.temperature,
                top_p=rollout_config.top_p,
                repetition_penalty=rollout_config.get("repetition_penalty", 1.0),
                output_kind=RequestOutputKind.CUMULATIVE,
                detokenize=False,
            )
            if (
                self.config.psrl.log_prob.enable_rollout_engine_log_prob
                and (
                    self.config.psrl.log_prob.get("update_reprefill_log_probs", False)
                    or self.config.psrl.partial_rollout.interrupt_as_prompt
                )
                and "raw_response_ids" in request.non_tensor_batch
            ):
                # Prompt logprobs are needed to refresh the response prefix on
                # a continuation; logprobs=0 alone only covers decoded tokens.
                sampling_params["prompt_logprobs"] = 0

            # override sampling params for validation
            if request.meta_info.get("validate", False):
                val_config = self.config.train_actor_rollout_ref.rollout.val_kwargs
                sampling_params["top_k"] = val_config.top_k
                sampling_params["top_p"] = val_config.top_p
                sampling_params["temperature"] = val_config.temperature

            # Generate response
            # psrl_logger.info(f"Generating response for request {request_id} on instance {new_instance_id}")
            consolidated_output, update_status = await self.rollout_wg_list[new_instance_id].execute_rank_zero_async(
                "generate_async", request, sampling_params
            )
            migration_overhead = self._migration_overhead.complete(request_id, consolidated_output)
            if migration_overhead is not None:
                psrl_logger.info(
                    "[ELASTIC_OVERHEAD] operation=post_scale_up_rebalance "
                    "scope=post_scale_up_rebalance migration_id=%s decision_id=%s "
                    "request_id=%s source_instance=%s destination_instance=%s selected_count=%s "
                    "interrupt_s=%.6f network_s=%.6f abort_to_redispatch_s=%.6f "
                    "reprefill_s=%.6f migration_s=%.6f "
                    "interrupt_scope=abort_to_requeue network_scope=requeue_to_redispatch "
                    "reprefill_scope=vllm_scheduled_to_first_token",
                    migration_overhead.get("migration_id"),
                    migration_overhead.get("decision_id"),
                    request_id,
                    migration_overhead.get("source_instance_id"),
                    migration_overhead.get("destination_instance_id"),
                    migration_overhead.get("selected_count"),
                    migration_overhead["interrupt_s"],
                    migration_overhead["network_s"],
                    migration_overhead["abort_to_redispatch_s"],
                    migration_overhead["reprefill_s"],
                    migration_overhead["migration_s"],
                )

            # Change engine status
            self.route_strategy.pop_request(request, new_instance_id)
            self._candidate_evaluation_inflight_requests.pop(str(request_id), None)
            # Remove request from inflight request ids for the instance
            self.instance_to_inflight_request_ids[new_instance_id].remove(request_id)

            # Check if request was interrupted and needs to be requeued
            if update_status == PSRL_RequestStatus.ROLLOUT_INTERRUPTED_BY_SCHEDULER:
                psrl_logger.debug(
                    f"Request {request_id} on instance {new_instance_id} was interrupted "
                    "by scheduler (most likely due to kv cache full and preemption), requeueing"
                )
                # Put back in priority queue for partial rollout
                # Ensure that the consolidated output has the rollout instance id recorded
                consolidated_output.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
                if int(request_id) in self._force_reroute_request_ids:
                    consolidated_output.non_tensor_batch["_psrl_force_reroute"] = np.array([True], dtype=bool)
                    self._force_reroute_request_ids.discard(int(request_id))
                request_key = canonical_psrl_request_id(request_id)
                was_exclusive_pending = request_key in getattr(self, "_rebalance_pending_request_ids", set())
                exclusive_enqueued = self._enqueue_exclusive_rebalance_request(consolidated_output)
                if exclusive_enqueued:
                    self._mark_migration_requeued(request_id, defer_log=True)
                elif was_exclusive_pending:
                    self._migration_overhead.discard(request_id)
                else:
                    self._mark_migration_requeued(request_id)
                if not exclusive_enqueued:
                    self.requests_to_route.put(consolidated_output)
                self._track_transition_request_resolved(request_id)
                # No result to set since the request is not completed
                return
            elif update_status == PSRL_RequestStatus.ROLLOUT_INTERRUPTED:
                psrl_logger.debug(
                    f"Request {request_id} on instance {new_instance_id} was interrupted "
                    "(due to model synchronization when enabled partial rollout), requeueing"
                )
                # Put back in priority queue for partial rollout
                # Ensure that the consolidated output has the rollout instance id recorded
                consolidated_output.non_tensor_batch["rollout_instance_id"] = np.array([new_instance_id], dtype=int)
                if int(request_id) in self._force_reroute_request_ids:
                    consolidated_output.non_tensor_batch["_psrl_force_reroute"] = np.array([True], dtype=bool)
                    self._force_reroute_request_ids.discard(int(request_id))
                request_key = canonical_psrl_request_id(request_id)
                was_exclusive_pending = request_key in getattr(self, "_rebalance_pending_request_ids", set())
                exclusive_enqueued = self._enqueue_exclusive_rebalance_request(consolidated_output)
                if exclusive_enqueued:
                    self._mark_migration_requeued(request_id, defer_log=True)
                elif was_exclusive_pending:
                    self._migration_overhead.discard(request_id)
                else:
                    self._mark_migration_requeued(request_id)
                if not exclusive_enqueued:
                    self.requests_to_route.put(consolidated_output)
                self._track_transition_request_resolved(request_id)
                # No result to set since the request is not completed
                return
            elif update_status == PSRL_RequestStatus.ROLLOUT_COMPLETED:
                self._planned_migration_destinations.pop(str(request_id), None)
                response_len = consolidated_output.non_tensor_batch["response_unpadded_len"][0]
                parent_prompt_id = request_id // rollout_n
                psrl_logger.debug(
                    f"Request {request_id} on instance {new_instance_id} of "
                    f"parent prompt {parent_prompt_id} completed successfully, "
                    f"length is {response_len}"
                )
                result = consolidated_output
                self._track_transition_request_resolved(request_id)
            else:
                # Means the request is aborted
                assert update_status is None, "The update status should be None if the request is aborted"
                self._planned_migration_destinations.pop(str(request_id), None)
                # psrl_logger.info(f"Request {request_id} on instance {new_instance_id} is aborted")
                result = None
                self._track_transition_request_resolved(request_id)
        else:
            # Means the request is aborted
            # psrl_logger.info(f"Request {request_id} is aborted")
            self._planned_migration_destinations.pop(str(request_id), None)
            result = None

        # Set the result for the request
        self._set_result(request_id, result)

    @ray.method(concurrency_group="control")
    async def check_should_migrate(self) -> list[int]:
        """Check which instances should be interrupted to migrate to others due to starvation.

        Returns:
            List[int]: The instance IDs that should be interrupted to migrate.
        """
        # psrl_logger.info("Checking which instances should be interrupted to migrate to others due to starvation")
        instance_to_status = self.route_strategy.instance_to_engine_status
        filtered_instance_ids = []
        # Filter instances that can be routed to (version is not aborted)
        # but no requests in the priority queue can be routed to it.
        # In this case, the instance is starving
        # and we may migrate requests from other instances to it.
        async with AsyncBusyPollingRayLock(self.ps_manager_handle):
            for instance_id in range(self.rollout_wg_size):
                if instance_to_status[instance_id].get_waiting_queue_size() != 0:
                    continue
                instance_version = self.instance_to_version_after_sync[instance_id]
                if await self.ps_manager_handle.check_aborted_model_versions.remote(instance_version):
                    continue

                def version_filter(request, instance_version=instance_version):
                    assert "version_tag" in request.non_tensor_batch, (
                        "Request must have 'version_tag' for checking version"
                    )
                    request_version = request.non_tensor_batch["version_tag"][0]
                    return request_version <= instance_version

                filtered_requests = self.requests_to_route.filter_by_condition(version_filter)
                filtered_request_ids = [request.non_tensor_batch["uid"][0] for request in filtered_requests]

                if len(filtered_request_ids) > 0:
                    is_aborted = await self.ps_manager_handle.check_aborted_requests.remote(
                        filtered_request_ids, remove=False
                    )
                    filtered_request_ids = [
                        request_id for i, request_id in enumerate(filtered_request_ids) if not is_aborted[i]
                    ]
                    is_validate_list = [request.meta_info.get("validate", False) for request in filtered_requests]
                    can_reserve = await self.ps_manager_handle.can_reserve_request.remote(
                        filtered_request_ids,
                        [instance_version],
                        without_new_reserve_entry=False,
                        is_validate=is_validate_list,
                    )
                    filtered_request_ids = [
                        request_id for i, request_id in enumerate(filtered_request_ids) if can_reserve[i] == [True]
                    ]
                if len(filtered_request_ids) == 0:
                    filtered_instance_ids.append(instance_id)

        candidate_migrate_instance_ids = []  # (instance_id, ratio)
        for starved_instance_id in filtered_instance_ids:
            for instance_id in range(self.rollout_wg_size):
                if instance_id == starved_instance_id:
                    continue
                if (
                    self.instance_to_version_after_sync[instance_id]
                    > self.instance_to_version_after_sync[starved_instance_id]
                ):
                    continue

                if self.config.psrl.sync_and_mig_strategy.mig.indicator == "request_num":
                    request_num = instance_to_status[instance_id].get_waiting_and_running_queue_size()
                    starved_request_num = instance_to_status[starved_instance_id].get_waiting_and_running_queue_size()
                    if starved_request_num == 0:
                        ratio = float("inf") if request_num > 0 else 1
                    else:
                        ratio = request_num / starved_request_num
                elif self.config.psrl.sync_and_mig_strategy.mig.indicator == "throughput":
                    throughput = instance_to_status[instance_id].get_generation_throughput()
                    starved_throughput = instance_to_status[starved_instance_id].get_generation_throughput()
                    if starved_throughput == 0:
                        ratio = float("inf") if throughput > 0 else 1
                    else:
                        ratio = throughput / starved_throughput
                elif self.config.psrl.sync_and_mig_strategy.mig.indicator == "kv_cache":
                    kv_cache_utilization = instance_to_status[instance_id].get_kv_cache_utilization()
                    starved_kv_cache_utilization = instance_to_status[starved_instance_id].get_kv_cache_utilization()
                    if starved_kv_cache_utilization == 0:
                        ratio = float("inf") if kv_cache_utilization > 0 else 1
                    else:
                        ratio = kv_cache_utilization / starved_kv_cache_utilization
                else:
                    raise ValueError(
                        f"Unknown migrate indicator: {self.config.psrl.sync_and_mig_strategy.mig.indicator}"
                    )

                if ratio > self.config.psrl.sync_and_mig_strategy.mig.threshold:
                    # psrl_logger.info(
                    #     f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                    #     f"has a ratio of {ratio} for migrating to instance {starved_instance_id} "
                    #     f"(version {self.instance_to_version_after_sync[starved_instance_id]})"
                    # )
                    candidate_migrate_instance_ids.append((instance_id, ratio))

        # We choose the instance with the highest ratio to migrate
        # TODO(lhy): support multiple instances to migrate and finer-grained migration strategy
        # Currently, we only support one instance to migrate,
        # and all the requests on the instance will be interrupted and looped back to the router.
        if len(candidate_migrate_instance_ids) > 0:
            candidate_migrate_instance_ids.sort(key=lambda x: x[1], reverse=True)
            migrate_instance_id = candidate_migrate_instance_ids[0][0]
            if self.config.psrl.sync_and_mig_strategy.mig.stop_indicator == "request_num":
                request_num = instance_to_status[migrate_instance_id].get_waiting_and_running_queue_size()
                if request_num < self.config.psrl.sync_and_mig_strategy.mig.stop_threshold:
                    return []
            elif self.config.psrl.sync_and_mig_strategy.mig.stop_indicator == "throughput":
                throughput = instance_to_status[migrate_instance_id].get_generation_throughput()
                if throughput < self.config.psrl.sync_and_mig_strategy.mig.stop_threshold:
                    return []
            elif self.config.psrl.sync_and_mig_strategy.mig.stop_indicator == "kv_cache":
                kv_cache_utilization = instance_to_status[migrate_instance_id].get_kv_cache_utilization()
                if kv_cache_utilization < self.config.psrl.sync_and_mig_strategy.mig.stop_threshold:
                    return []
            else:
                raise ValueError(
                    f"Unknown stop indicator: {self.config.psrl.sync_and_mig_strategy.mig.stop_indicator}"
                )
            return [migrate_instance_id]
        return []

    @ray.method(concurrency_group="control")
    async def check_should_sync(self, instance_id: int, ps_model_version: int) -> bool:
        """Check if the instance should synchronize with PS.

        Args:
            instance_id (int): The instance ID to synchronize with.
            ps_model_version (int): The version of the PS model to synchronize with.

        Returns:
            bool: True if the instance should synchronize with PS, False otherwise.
        """
        # psrl_logger.info(f"Checking if instance {instance_id} should synchronize with PS")
        # If there are requests in the waiting queue
        # we will not attempt to synchronize with PS since the instance is still busy.
        instance_status = self.route_strategy.instance_to_engine_status[instance_id]
        if instance_status.get_waiting_queue_size() > 0:
            return False

        # Check if there are any requests that still can be routed to the instance
        # In this case, we will not attempt to synchronize with PS
        async with AsyncBusyPollingRayLock(self.ps_manager_handle):
            # 1. Check if there are any requests version satisfies the condition before synchronization
            get_version_remote = self.ps_manager_handle.get_rollout_instance_model_version.remote
            current_instance_version = await get_version_remote(instance_id)
            if await self.ps_manager_handle.check_aborted_model_versions.remote(current_instance_version):
                filtered_request_ids = []
            else:

                def version_filter(request):
                    assert "version_tag" in request.non_tensor_batch, (
                        "Request must have 'version_tag' for checking version"
                    )
                    request_version = request.non_tensor_batch["version_tag"][0]
                    return request_version <= current_instance_version

                filtered_requests = self.requests_to_route.filter_by_condition(version_filter)
                filtered_request_ids = [request.non_tensor_batch["uid"][0] for request in filtered_requests]

            # 2. Check if there are any requests
            # that can be RESERVED for the instance but no need to reserve new entry
            # before synchronization
            if len(filtered_request_ids) > 0:
                is_aborted = await self.ps_manager_handle.check_aborted_requests.remote(
                    filtered_request_ids, remove=False
                )
                filtered_request_ids = [
                    request_id for i, request_id in enumerate(filtered_request_ids) if not is_aborted[i]
                ]
                can_reserve_without_new_reserve_entry = await self.ps_manager_handle.can_reserve_request.remote(
                    filtered_request_ids,
                    [current_instance_version],
                    without_new_reserve_entry=True,
                )
                filtered_request_ids = [
                    request_id
                    for i, request_id in enumerate(filtered_request_ids)
                    if can_reserve_without_new_reserve_entry[i] == [True]
                ]

        # If there are requests that can still be routed to
        # the instance before synchronization without new reserve entry
        # we will not attempt to synchronize with PS
        if len(filtered_request_ids) > 0 and self.config.psrl.sync_and_mig_strategy.sync.check_req_before_sync:
            return False

        # 3. Check indicator to determine whether to synchronize with PS
        if self.config.psrl.sync_and_mig_strategy.sync.indicator == "request_num":
            # Check whether request num is above threshold
            request_num = instance_status.get_waiting_and_running_queue_size()
            psrl_logger.debug(
                f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                f"request_num: {request_num}, "
                f"threshold: {self.config.psrl.sync_and_mig_strategy.sync.threshold}"
            )
            if request_num > self.config.psrl.sync_and_mig_strategy.sync.threshold:
                return False
        elif self.config.psrl.sync_and_mig_strategy.sync.indicator == "throughput":
            # Check whether throughput is above threshold
            throughput = self.route_strategy.instance_to_engine_status[instance_id].get_generation_throughput()
            psrl_logger.debug(
                f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                f"throughput: {throughput}, "
                f"threshold: {self.config.psrl.sync_and_mig_strategy.sync.threshold}"
            )
            if throughput > self.config.psrl.sync_and_mig_strategy.sync.threshold:
                return False
        elif self.config.psrl.sync_and_mig_strategy.sync.indicator == "kv_cache":
            # Check whether KV Cache is above threshold
            kv_cache_utilization = instance_status.get_kv_cache_utilization()
            psrl_logger.debug(
                f"Instance {instance_id} (version {self.instance_to_version_after_sync[instance_id]}) "
                f"kv_cache_utilization: {kv_cache_utilization}, "
                f"threshold: {self.config.psrl.sync_and_mig_strategy.sync.threshold}"
            )
            if kv_cache_utilization > self.config.psrl.sync_and_mig_strategy.sync.threshold:
                return False
        elif self.config.psrl.sync_and_mig_strategy.sync.indicator == "hypothesis_test":
            # TODO(lhy): Implement hypothesis test after refactor
            # We attempt to synchronize with PS and check if there is any
            # benefit from synchronization
            raise NotImplementedError("Hypothesis test is not implemented")
            """
            def filter_func(request):
                version = request.non_tensor_batch.get(
                    "version_tag", [ps_model_version + 1]
                )[0]
                min_version_limit = request.non_tensor_batch.get(
                    "min_version_limit",
                    [ps_model_version + 1 + self.staleness]
                )[0]
                return (
                    version <= ps_model_version
                    or min_version_limit <= ps_model_version + self.staleness
                )
            
            new_filtered_requests = (
                self.requests_to_route.filter_by_condition(filter_func)
            )
            # Requests may be able to be routed to the instance {instance_id}
            # after synchronization, checking routing benefit...
            for request in new_filtered_requests:
                routing_benefit = (
                    self.route_strategy.calculate_routing_benefit(
                        request, instance_id
                    )
                )
                if routing_benefit > 0:
                    return True
            # No requests will benefit from routing to the instance
            # {instance_id} after synchronization
            """
        else:
            raise ValueError(f"Unknown sync indicator: {self.config.psrl.sync_and_mig_strategy.sync.indicator}")

        return True
