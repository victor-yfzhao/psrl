# Modified from verl/experimental/reward/router/naive_router.py
import asyncio
import logging
import os
from typing import Any

import ray
from verl import DataProto

psrl_logger = logging.getLogger("reward_model_router")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


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
        retry_delay: float = 2.0,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the router with reward model worker handles.

        Args:
            worker_handles: List of Ray ActorHandles for RewardModelWorker instances.
            retry_delay: Delay between retries (in seconds).
            verbose: Enable verbose logging.
        """
        self.verbose = verbose
        self.worker_handles = worker_handles
        self.request_counts = {i: 0 for i in range(len(worker_handles))}

        self.retry_delay = retry_delay

        psrl_logger.info(f"RewardModelRouter initialized with {len(worker_handles)} workers")

    async def generate(self, request: DataProto) -> DataProto | None:
        """
        Route a generation request to an available worker.

        Args:
            request: DataProto containing the input prompt.

        Returns:
            DataProto from the worker.
        """
        worker_idx = self._select_worker()
        worker_handle = self.worker_handles[worker_idx]
        request_uids = self._format_request_uids(request)

        psrl_logger.info(
            "[router] Routing reward request %s to worker %d (inflight=%d)",
            request_uids,
            worker_idx,
            self.request_counts[worker_idx],
        )

        result = await worker_handle.generate_async.remote(request)
        if result is None:
            psrl_logger.info(
                f"Request {request.non_tensor_batch['uid'][0]} is interrupted (instance sleep), "
                "need to requeue."
                )
            result = await self.generate(request)
        else:
            psrl_logger.info(f"Request {request.non_tensor_batch['uid'][0]} is completed (finished generation)")
        self._release_worker(worker_idx)
        psrl_logger.info(
            "[router] Reward request %s finished on worker %d", request_uids, worker_idx
        )
        return result

    async def generate_async(self, request: DataProto) -> DataProto | None:
        """
        Async alias for generate().
        """
        return await self.generate(request)

    def _select_worker(self) -> int:
        """
        Select the least-loaded worker (simple round-robin by request count).
        """
        worker_idx = min(self.request_counts, key=self.request_counts.get)
        self.request_counts[worker_idx] += 1
        return worker_idx

    def _release_worker(self, worker_idx: int) -> None:
        """
        Mark worker as free after request completes.
        """
        self.request_counts[worker_idx] = max(0, self.request_counts[worker_idx] - 1)

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
    max_attempts: int = 3,
    retry_delay: float = 2.0,
    verbose: bool = False,
) -> ray.actor.ActorHandle:
    """
    Launch a router process (or return a router handle for direct access).
    
    For PSRL training, we return a Ray actor handle instead of starting
    a separate HTTP server process.

    Args:
        worker_handles: List of RewardModelWorker Ray handles.
        max_attempts: Maximum retry attempts for failed requests.
        retry_delay: Delay between retries (in seconds).
        verbose: Enable verbose logging.
        
    Returns:
        Tuple of (router_address, router_handle).
        - router_address: A placeholder string (since we're not using HTTP).
        - router_handle: Ray ActorHandle for the router.
    """
    # Create a Ray actor for the router
    reward_model_roter_cls = ray.remote(PSRL_RewardModelRouter)
    router_handle = reward_model_roter_cls.remote(
        worker_handles=worker_handles,
        retry_delay=retry_delay,
        verbose=verbose,
    )
    psrl_logger.info(f"RewardModelRouter launched as Ray actor")
    return router_handle
