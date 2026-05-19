# Modified from verl/experimental/reward/router/naive_router.py
import asyncio
import heapq
import logging
import os
import time
from typing import Any

import ray
from omegaconf import DictConfig
from verl import DataProto

from psrl.utils.elastic_rm.diagnostics import log_elastic_rm_backlog_diag
from psrl.utils.logger import DualOutputHandler

psrl_logger = logging.getLogger("reward_model_router")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

@ray.remote(concurrency_groups={"control": 5})
class PSRL_RewardModelRouter:
    """
    Simple round-robin router for reward model replicas.
    
    This router accepts Ray ActorHandles and forwards generation requests
    to the least-loaded replica.
    
    NOTE(zyf): will enable to use http-server instead of ray actor in the future.
    """

    def __init__(
        self,
        worker_handles: list[ray.actor.ActorHandle],
        config: DictConfig,
        reward_model_config: DictConfig | None = None,
        retry_delay: float = 2.0,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the router with reward model worker handles.

        Args:
            worker_handles: List of Ray ActorHandles for RewardModelWorker instances.
            config: Full PSRL Hydra config (needs psrl.logging_path for file logging).
            retry_delay: Delay between retries (in seconds).
            verbose: Enable verbose logging.
        """
        self.config = config
        self.verbose = verbose
        self.worker_handles = worker_handles
        self.reward_model_config = reward_model_config
        self.request_counts = {i: 0 for i in range(len(worker_handles))}
        self.paused_worker_indices: set[int] = set()
        self.retry_delay = retry_delay
        self.request_futures: dict[str, asyncio.Future] = {}
        # Min-heap items (see _enqueue_request): (missing_flag, buffer_id, fifo_seq, request_key, request).
        # missing_flag: 0 if buffer_id present (routed first), 1 if absent (demoted).
        # buffer_id: normalized id for ordering (smaller first); 0 when missing_flag==1.
        # fifo_seq: monotonic tie-break for FIFO among equal priority.
        self.requests_to_route: list[tuple[int, int, int, str, DataProto]] = []
        self.routing_lock = asyncio.Lock()
        self.routing_status_update_event = asyncio.Event()
        self._is_routing = False
        self._interrupt_routing = False
        self.scheduler_task: asyncio.Task | None = None
        self._request_key_counter = 0
        self._waiting_seq_counter = 0

        raw_cap = None
        if self.reward_model_config is not None:
            raw_cap = self.reward_model_config.get("max_concurrent_requests_per_instance", None)
        self.max_concurrent_requests_per_instance: int | None = None
        if raw_cap is not None:
            try:
                cap = int(raw_cap)
                if cap > 0:
                    self.max_concurrent_requests_per_instance = cap
            except (TypeError, ValueError):
                self.max_concurrent_requests_per_instance = None

        # Build logger
        self.log_prefix = "RewardModelRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info(
            "RewardModelRouter initialized with %d workers (max_concurrent_requests_per_instance=%s)",
            len(worker_handles),
            self.max_concurrent_requests_per_instance,
        )

    async def _get_worker_active_task_num(self, worker_idx: int) -> int | None:
        """
        Best-effort fetch of reward worker's active task number.
        Returns None on any failure (router should fall back to local counters).
        """
        try:
            worker_handle = self.worker_handles[int(worker_idx)]
            # `get_active_task_num` is a lightweight method on RewardModelWorker.
            # It may not exist for older worker implementations.
            return int(await worker_handle.get_active_task_num.remote())
        except Exception:
            return None

    async def generate(self, request: DataProto) -> DataProto | None:
        """
        Route one request through queue-based scheduler.

        Args:
            request: DataProto containing the input prompt.

        Returns:
            DataProto from the worker, or None if interrupted.
        """
        return await self.generate_async(request)

    async def generate_async(self, request: DataProto) -> DataProto | None:
        """
        Queue a request and wait for routing result.
        """
        if self.scheduler_task is None:
            self.scheduler_task = asyncio.create_task(self._routing_loop())
            # Fail fast if the background loop crashes.
            self.scheduler_task.add_done_callback(lambda task: task.result())
            psrl_logger.info("[router] Started reward routing loop")

        request_key = self._build_request_key(request)
        result_future = asyncio.get_running_loop().create_future()
        self.request_futures[request_key] = result_future
        self._enqueue_request(request_key, request)
        try:
            return await result_future
        finally:
            self.request_futures.pop(request_key, None)

    async def _routing_loop(self):
        """Continuously pop queued requests and route them."""
        while True:
            if self._interrupt_routing:
                self._is_routing = False
                self.routing_status_update_event.clear()
                await self.routing_status_update_event.wait()
                continue

            worker_idx = self._select_worker()
            if worker_idx is None:
                self._is_routing = False
                self.routing_status_update_event.clear()
                await self.routing_status_update_event.wait()
                continue

            next_request = self._dequeue_request()
            if next_request is None:
                self._release_worker(worker_idx)
                self._is_routing = False
                self.routing_status_update_event.clear()
                await self.routing_status_update_event.wait()
                continue

            request_key, request = next_request
            self._is_routing = True
            task = asyncio.create_task(self._route_single_request(request_key, request, worker_idx))
            task.add_done_callback(lambda t: t.result())
            await asyncio.sleep(0)

    async def _route_single_request(self, request_key: str, request: DataProto, worker_idx: int):
        """Route one request to a selected worker; requeue when needed."""
        # The caller might have already cancelled/aborted this request.
        if request_key not in self.request_futures:
            self._release_worker(worker_idx)
            return

        request_uids = self._format_request_uids(request)
        if self._interrupt_routing:
            self._release_worker(worker_idx)
            self._enqueue_request(request_key, request)
            psrl_logger.info(
                "[router] Routing is interrupted, Request %s, requeueing original request.",
                request_uids,
            )
            return

        worker_handle = self.worker_handles[worker_idx]
        inflight = await self._get_worker_active_task_num(worker_idx)
        if inflight is None:
            inflight = self.request_counts[worker_idx]
        psrl_logger.info(
            "[router] Routing reward request %s to worker %d (inflight=%d)",
            request_uids,
            worker_idx,
            inflight,
        )
        try:
            result = await worker_handle.generate_async.remote(request)
        finally:
            self._release_worker(worker_idx)

        if result is None:
            psrl_logger.info("Request %s interrupted or unavailable, requeueing original request.", request_uids)
            self._enqueue_request(request_key, request)
            await asyncio.sleep(self.retry_delay)
            return

        interrupted = False
        try:
            interrupted = bool(result.non_tensor_batch.get("interrupted", [False])[0])
        except Exception:
            interrupted = False
        if interrupted:
            # Requeue the partial output to continue generation from existing tokens.
            psrl_logger.info("Request %s interrupted, requeueing partial output for continuation.", request_uids)
            self._enqueue_request(request_key, result)
            await asyncio.sleep(self.retry_delay)
            return

        psrl_logger.info("[router] Reward request %s finished on worker %d", request_uids, worker_idx)
        self._set_result(request_key, result)
        return

    def _select_worker(self) -> int | None:
        """
        Select the least-loaded worker (simple by request count).
        """
        available_indices = [idx for idx in self.request_counts if idx not in self.paused_worker_indices]
        if self.max_concurrent_requests_per_instance is not None:
            available_indices = [
                idx
                for idx in available_indices
                if self.request_counts[idx] < self.max_concurrent_requests_per_instance
            ]
        if not available_indices:
            return None
        worker_idx = min(available_indices, key=lambda idx: self.request_counts[idx])
        self.request_counts[worker_idx] += 1
        return worker_idx

    def _release_worker(self, worker_idx: int) -> None:
        """
        Mark worker as free after request completes.
        """
        self.request_counts[worker_idx] = max(0, self.request_counts[worker_idx] - 1)
        self.routing_status_update_event.set()

    def _set_result(self, request_key: str, result: DataProto | None):
        """Resolve the waiting future of a routed request."""
        request_future = self.request_futures.get(request_key, None)
        if request_future is None:
            return
        if request_future.done():
            return
        request_future.set_result(result)

    def _build_request_key(self, request: DataProto) -> str:
        """
        Build a unique key for in-flight request tracking.

        NOTE: uid may be absent or duplicated in edge cases, so we add a local
        monotonic suffix to avoid key collision inside the router.
        """
        uid_repr = self._format_request_uids(request)
        request_key = f"{uid_repr}#{self._request_key_counter}"
        self._request_key_counter += 1
        return request_key

    def _enqueue_request(self, request_key: str, request: DataProto) -> None:
        buffer_id = self._extract_buffer_id(request)
        # Smaller buffer_id should be routed first. Requests without buffer_id are
        # demoted behind known buffer_id requests while preserving FIFO order.
        missing_flag = 1 if buffer_id is None else 0
        normalized_buffer_id = buffer_id if buffer_id is not None else 0
        item = (
            missing_flag,
            normalized_buffer_id,
            self._waiting_seq_counter,
            request_key,
            request,
        )
        self._waiting_seq_counter += 1
        heapq.heappush(self.requests_to_route, item)
        self.routing_status_update_event.set()

    def _dequeue_request(self) -> tuple[str, DataProto] | None:
        if not self.requests_to_route:
            return None
        _, _, _, request_key, request = heapq.heappop(self.requests_to_route)
        return request_key, request

    @ray.method(concurrency_group="control")
    def pause_instances(self, instance_ids: list[int]):
        for instance_id in instance_ids:
            self.paused_worker_indices.add(int(instance_id))

    @ray.method(concurrency_group="control")
    def resume_instances(self, instance_ids: list[int]):
        for instance_id in instance_ids:
            self.paused_worker_indices.discard(int(instance_id))
        self.routing_status_update_event.set()

    @ray.method(concurrency_group="control")
    def is_routing(self) -> bool:
        """Check whether router is actively dispatching requests."""
        return self._is_routing

    @ray.method(concurrency_group="control")
    def get_pending_request_count(self) -> int:
        """Return waiting-queue depth for reward routing (elastic_rm backlog signal).

        Uses ``requests_to_route`` size: requests not yet dequeued by ``_routing_loop``.
        Excludes requests already in ``_route_single_request`` or running on a worker.
        """
        t0 = time.monotonic()
        log_elastic_rm_backlog_diag(psrl_logger, "stage=RewardModelRouter_enter")
        n = int(len(self.requests_to_route))
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RewardModelRouter_exit pending=%d body_s=%.6f",
            n,
            time.monotonic() - t0,
        )
        return n

    @ray.method(concurrency_group="control")
    async def interrupt_routing(self):
        """Pause routing (used during coordinated transitions)."""
        async with self.routing_lock:
            self._interrupt_routing = True

    @ray.method(concurrency_group="control")
    async def resume_routing(self):
        """Resume routing and wake waiting routing loop."""
        async with self.routing_lock:
            self._interrupt_routing = False
        self.routing_status_update_event.set()

    @staticmethod
    def _format_request_uids(request: DataProto) -> str:
        uid_value = request.non_tensor_batch.get("uid")
        if uid_value is None:
            return "unknown"
        if hasattr(uid_value, "tolist"):
            uid_value = uid_value.tolist()
        if isinstance(uid_value, (list, tuple)):
            if len(uid_value) == 1:
                return str(uid_value[0])
            return ",".join(str(u) for u in uid_value)
        return str(uid_value)

    @staticmethod
    def _extract_buffer_id(request: DataProto) -> int | None:
        non_tensor_batch = getattr(request, "non_tensor_batch", {}) or {}
        candidate_keys = ("buffer_id", "waiting_buffer_id", "train_buffer_id")

        def _normalize_int(v: Any) -> int | None:
            if hasattr(v, "tolist"):
                v = v.tolist()
            if isinstance(v, (list, tuple)):
                if not v:
                    return None
                v = v[0]
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        for key in candidate_keys:
            value = non_tensor_batch.get(key, None)
            normalized = _normalize_int(value)
            if normalized is not None:
                return normalized

        extra_info = non_tensor_batch.get("extra_info", None)
        if isinstance(extra_info, dict):
            normalized = _normalize_int(extra_info.get("buffer_id", None))
            if normalized is not None:
                return normalized
        return None


def launch_router_process(
    worker_handles: list[ray.actor.ActorHandle],
    config: DictConfig,
    reward_model_config: DictConfig | None = None,
    max_attempts: int = 3,
    retry_delay: float = 2.0,
    verbose: bool = False,
    max_concurrency: int = 1,
) -> ray.actor.ActorHandle:
    """
    Launch a router process (or return a router handle for direct access).
    
    For PSRL training, we return a Ray actor handle instead of starting
    a separate HTTP server process.

    Args:
        worker_handles: List of RewardModelWorker Ray handles.
        config: Full PSRL Hydra config for the router actor.
        max_attempts: Maximum retry attempts for failed requests.
        retry_delay: Delay between retries (in seconds).
        verbose: Enable verbose logging.
        
    Returns:
        Tuple of (router_address, router_handle).
        - router_address: A placeholder string (since we're not using HTTP).
        - router_handle: Ray ActorHandle for the router.
    """
    # Create a Ray actor for the router
    reward_model_router_cls = PSRL_RewardModelRouter
    router_handle = reward_model_router_cls.options(max_concurrency=max_concurrency).remote(
        worker_handles=worker_handles,
        config=config,
        reward_model_config=reward_model_config,
        retry_delay=retry_delay,
        verbose=verbose,
    )
    psrl_logger.info(f"RewardModelRouter launched as Ray actor")
    return router_handle
