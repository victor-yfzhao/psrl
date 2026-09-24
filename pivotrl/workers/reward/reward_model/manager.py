# Modified from verl/experimental/reward/reward_model.py
import asyncio
import logging
import os

from omegaconf import DictConfig
import ray
from ray.util.queue import Queue as RayQueue
from verl.single_controller.ray.base import RayResourcePool

from pivotrl.workers.config import HFModelConfig
from pivotrl.workers.reward.reward_model.coordinator import RewardModelCoordinator
from pivotrl.workers.reward.reward_model.replica import PivotRL_RewardModelReplica

pivotrl_logger = logging.getLogger(__file__)
pivotrl_logger.setLevel(os.getenv("PIVOTRL_LOGGING_LEVEL", "WARN"))


class PivotRL_RewardModelManager:
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
        reward_model_wg_list: list[ray.actor.ActorHandle],
        status_queues: list[RayQueue],
        max_concurrency: int = 1,
    ):
        """
        Initialize the reward model manager.

        """
        self.config = config
        self.reward_model_name = reward_model_name
        self.reward_model_config = reward_model_config
        self.reward_model_wg_list = reward_model_wg_list
        self.replicas: list[PivotRL_RewardModelReplica] = []
        self.router_process: ray.actor.ActorHandle | None = None
        self.status_queues: list[RayQueue] = status_queues

        self.max_concurrency = max_concurrency

        # self.router_address: str | None = None

        self.reward_model_coordinator = RewardModelCoordinator.remote(
            config=self.config,
            rm_config=self.reward_model_config,
            status_queues=self.status_queues,
        )

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

        self.replicas = []
        for i in range(len(self.reward_model_wg_list)):
            reward_model_wg = self.reward_model_wg_list[i]
            status_queue = self.status_queues[i]
            self.replicas.append(
                PivotRL_RewardModelReplica(
                    replica_rank=i,
                    reward_model_config=self.reward_model_config,
                    rollout_config=rollout_config,
                    model_config=model_config,
                    pivotrl_config=self.config.pivotrl,
                    reward_model_wg=reward_model_wg,
                    reward_model_name=self.reward_model_name,
                    status_queue=status_queue,
                )
            )

        self._init_all_replicas()

    def _init_all_replicas(self):
        """
        Initialize all replicas and their models in parallel.
        """
        ray.get(self.reward_model_coordinator.set_reward_model_wg_list.remote(self.reward_model_wg_list))
        ray.get(self.reward_model_coordinator.init_model.remote())
        ray.get(self.reward_model_coordinator.start_busy_loop.remote())
        pivotrl_logger.info(f"Reward model coordinator started!")

        pivotrl_logger.info(f"All {len(self.replicas)} reward model replicas initialized.")

    def _initialize_router(self):
        """
        Launch a router process for load balancing across replicas.
        
        The router exposes a single HTTP endpoint and distributes requests
        to the least-loaded replica worker.
        """
        from pivotrl.workers.reward.reward_model.router import launch_router_process

        # Collect worker handles from all replicas
        worker_handles = [replica.worker_handle for replica in self.replicas if replica.worker_handle is not None]
        if len(worker_handles) == 0:
            pivotrl_logger.warning("No worker handles available; skipping router initialization.")
            return

        # Launch the router as a separate Ray actor or process
        self.router_process = launch_router_process(
            worker_handles=worker_handles,
            worker_groups=self.reward_model_wg_list,
            config=self.config,
            reward_model_config=self.reward_model_config,
            max_concurrency=self.max_concurrency,
        )
        ray.get(self.reward_model_coordinator.set_reward_model_router.remote(self.router_process))
        pivotrl_logger.info(f"RewardModelRouter launched!")

    # def get_router_address(self) -> str | None:
    #     pass

    def get_replica_handles(self) -> list:
        """
        Return all replicas for direct access (no router).
        """
        return self.replicas
    
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

    def shutdown(self) -> None:
        """Gracefully close reward engines and terminate coordinator/router actors."""
        shutdown_error = None
        if self.router_process is not None:
            ray.kill(self.router_process, no_restart=True)
            self.router_process = None
        try:
            ray.get(self.reward_model_coordinator.shutdown_reward_engines.remote())
        except Exception as exc:
            shutdown_error = exc
        finally:
            ray.kill(self.reward_model_coordinator, no_restart=True)

        if shutdown_error is not None:
            raise RuntimeError(
                f"Failed to shut down reward model manager {self.reward_model_name!r}."
            ) from shutdown_error