# Modified from verl/experimental/reward/router/naive_router.py
import asyncio
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
        self.request_counts = {i: 0 for i in range(len(worker_handles))}
        self.paused_worker_indices: set[int] = set()
        self.retry_delay = retry_delay
        self.request_futures: dict[str, asyncio.Future] = {}
        self.requests_to_route: asyncio.Queue[tuple[str, DataProto]] = asyncio.Queue()
        self.routing_lock = asyncio.Lock()
        self.routing_status_update_event = asyncio.Event()
        self._is_routing = False
        self._interrupt_routing = False
        self.scheduler_task: asyncio.Task | None = None
        self._request_key_counter = 0

        # Build logger
        self.log_prefix = "RewardModelRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info(f"RewardModelRouter initialized with {len(worker_handles)} workers")

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
        self.requests_to_route.put_nowait((request_key, request))
        self.routing_status_update_event.set()
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

            try:
                request_key, request = self.requests_to_route.get_nowait()
            except asyncio.QueueEmpty:
                self._is_routing = False
                self.routing_status_update_event.clear()
                await self.routing_status_update_event.wait()
                continue

            self._is_routing = True
            task = asyncio.create_task(self._route_single_request(request_key, request))
            task.add_done_callback(lambda t: t.result())
            await asyncio.sleep(0)

    async def _route_single_request(self, request_key: str, request: DataProto):
        """Route one request to an available worker; requeue when needed."""
        # The caller might have already cancelled/aborted this request.
        if request_key not in self.request_futures:
            return

        request_uids = self._format_request_uids(request)
        while True:
            if self._interrupt_routing:
                self.requests_to_route.put_nowait((request_key, request))
                self.routing_status_update_event.set()
                psrl_logger.info(f"[router] Routing is interrupted, Request {request_uids}, requeueing original request.")
                return

            worker_idx = self._select_worker()
            if worker_idx is None:
                psrl_logger.warning("[router] No available reward worker, retrying in %.2fs", self.retry_delay)
                await asyncio.sleep(self.retry_delay)
                continue

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
                self.requests_to_route.put_nowait((request_key, request))
                self.routing_status_update_event.set()
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
                self.requests_to_route.put_nowait((request_key, result))
                self.routing_status_update_event.set()
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
        """Return outstanding reward requests not yet completed (elastic_rm backlog signal).

        `requests_to_route` only counts items not yet dequeued by `_routing_loop`. When all
        workers are paused (e.g. elastic sleep), `_route_single_request` dequeue then spins
        in the retry loop — the queue can be empty while many callers still wait on
        `request_futures`. Count futures so coordinators see real pressure and can force WAKE_UP.
        """
        t0 = time.monotonic()
        log_elastic_rm_backlog_diag(psrl_logger, "stage=RewardModelRouter_enter")
        n = int(len(self.request_futures))
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


def launch_router_process(
    worker_handles: list[ray.actor.ActorHandle],
    config: DictConfig,
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
        retry_delay=retry_delay,
        verbose=verbose,
    )
    psrl_logger.info(f"RewardModelRouter launched as Ray actor")
    return router_handle
