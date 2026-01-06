# Modified from verl/experimental/reward/reward_model.py
import asyncio
import logging
import os

from omegaconf import DictConfig
import ray
from verl.single_controller.ray.base import RayResourcePool

from psrl.workers.config import HFModelConfig
from psrl.workers.reward.reward_model.replica import PSRL_RewardModelReplica

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_RewardModelManager:
    """
    High-level manager for reward model replicas and optional router.
    
    This manager:
    - Creates multiple RewardModelReplica instances (one per TP group).
    - Optionally launches a router for load balancing across replicas.
    - Provides a unified interface for reward loop managers to query.
    """

    def __init__(
        self,
        reward_model_name: str,
        config: DictConfig,
        reward_model_config: DictConfig,
        resource_pools: RayResourcePool | list[RayResourcePool],
    ):
        """
        Initialize the reward model manager.

        """
        self.config = config
        self.reward_model_name = reward_model_name
        self.reward_model_config = reward_model_config
        if isinstance(resource_pools, RayResourcePool):
            resource_pools = [resource_pools]
        if not resource_pools:
            raise ValueError("At least one RayResourcePool is required to launch reward model replicas")
        self.resource_pools: list[RayResourcePool] = list(resource_pools)
        self.replicas: list[PSRL_RewardModelReplica] = []
        self.router_process: ray.actor.ActorHandle | None = None
        self.router_address: str | None = None

        # Initialize replicas and router
        self._initialize_replicas()
        self._initialize_router()

    def _initialize_replicas(self):
        """
        Create and initialize reward model replicas across available GPUs.
        """
        rollout_config = self.reward_model_config.rollout
        model_config = HFModelConfig(
            path=self.reward_model_config.model.path,
            external_lib=self.reward_model_config.model.get("external_lib"),
            trust_remote_code=self.reward_model_config.model.get("trust_remote_code", False),
        )
        self.reward_model_tokenizer = model_config.tokenizer
        requested_replicas = self.reward_model_config.get("num_replicas", len(self.resource_pools))
        if requested_replicas > len(self.resource_pools):
            raise ValueError(
                f"Reward model requires {requested_replicas} replicas but only {len(self.resource_pools)} "
                "resource pools are available."
            )

        self.replicas = []
        for i in range(requested_replicas):
            resource_pool = self.resource_pools[i]
            self.replicas.append(
                PSRL_RewardModelReplica(
                    replica_rank=i,
                    reward_model_config=self.reward_model_config,
                    rollout_config=rollout_config,
                    model_config=model_config,
                    psrl_config=self.config.psrl,
                    resource_pool=resource_pool,
                    reward_model_name=self.reward_model_name,
                )
            )

        self._run_coroutines_blocking([self._init_all_replicas()])

    async def _init_all_replicas(self):
        """
        Initialize all replicas and their models in parallel.
        """
        await asyncio.gather(*[replica.init_replica() for replica in self.replicas])
        await asyncio.gather(*[replica.init_model() for replica in self.replicas])
        psrl_logger.info(f"All {len(self.replicas)} reward model replicas initialized.")

    def _run_coroutines_blocking(self, coroutines: list[asyncio.Future]):
        if not coroutines:
            return
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(asyncio.gather(*coroutines))
        finally:
            loop.close()
            if previous_loop is not None:
                asyncio.set_event_loop(previous_loop)
            else:
                asyncio.set_event_loop(None)

    def _initialize_router(self):
        """
        Launch a router process for load balancing across replicas.
        
        The router exposes a single HTTP endpoint and distributes requests
        to the least-loaded replica worker.
        """
        from psrl.workers.reward.reward_model.router import launch_router_process

        # Collect worker handles from all replicas
        worker_handles = [replica.worker_handle for replica in self.replicas if replica.worker_handle is not None]
        if len(worker_handles) <= 1:
            psrl_logger.warning("No worker handles available; skipping router initialization.")
            return

        # Launch the router as a separate Ray actor or process
        self.router_process = launch_router_process(worker_handles=worker_handles)
        psrl_logger.info(f"RewardModelRouter launched at {self.router_address}")

    def get_router_address(self) -> str | None:
        pass

    def get_replica_handles(self) -> list:
        """
        Return all replica worker handles for direct access (no router).
        """
        return [replica.worker_handle for replica in self.replicas if replica.worker_handle is not None]
    
    def get_router_process(self) -> ray.actor.ActorHandle | None:
        """
        Return the router process.
        """
        return self.router_process
    
    def get_reward_model_tokenizer(self):
        """
        Return the reward model tokenizer.
        """
        return self.reward_model_tokenizer