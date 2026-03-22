import logging
import os

import ray
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ray.util.queue import Queue as RayQueue
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import RayResourcePool

from psrl.workers.config import HFModelConfig
from psrl.workers.reward.reward_model.worker import PSRL_RewardModelWorker

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_RewardModelReplicaBase(ABC):
    """
    Manages a single replica of the reward model.
    
    NOTE(zyf): will enable to use http-server instead of ray actor in the future.
    """
    def __init__(
        self,
        replica_rank: int,
        reward_model_config: DictConfig,
        rollout_config: DictConfig,
        model_config: HFModelConfig,
        psrl_config: DictConfig,
        reward_model_wg: RayWorkerGroup,
    ):
        """
        Initialize a reward model replica.

        Args:
            replica_rank: Rank of this replica (for uniqueness).
            rollout_config: vLLM rollout configuration.
            model_config: HuggingFace model configuration.
            psrl_config: PSRL-specific configuration.
            reward_model_wg: Worker group for this replica.
        """
        self.replica_rank = replica_rank
        self.reward_model_config = reward_model_config
        self.rollout_config = rollout_config
        self.model_config = model_config
        self.psrl_config = psrl_config
        self.worker_group = reward_model_wg

    @abstractmethod
    async def generate(self, request: DataProto) -> DataProto | None:
        """
        Generate a response for the given request.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_async(self, request: DataProto) -> DataProto | None:
        """
        Generate a response for the given request asynchronously.
        """
        raise NotImplementedError


class PSRL_RewardModelReplica(PSRL_RewardModelReplicaBase):
    def __init__(
        self,
        replica_rank: int,
        reward_model_config: DictConfig,
        rollout_config: DictConfig,
        model_config: HFModelConfig,
        psrl_config: DictConfig,
        reward_model_wg: RayWorkerGroup,
        reward_model_name: str,
        status_queue: RayQueue,
    ):
        super().__init__(replica_rank, reward_model_config, rollout_config, model_config, psrl_config, reward_model_wg)
        self.reward_model_name = reward_model_name
        self.status_queue = status_queue
        self._worker_handle = self.worker_group.workers[0] if self.worker_group.workers else None

    @property
    def worker_handle(self):
        return self._worker_handle

    async def generate(self, request):
        if self._worker_handle is None:
            raise RuntimeError("Replica not initialized.")
        return await self._worker_handle.generate.remote(request)
    
    async def generate_async(self, request, consolidate: bool = True):
        if self._worker_handle is None:
            raise RuntimeError("Replica not initialized.")
        return await self._worker_handle.generate_async.remote(request, consolidate=consolidate)


class PSRL_RewardModelHttpServerReplica(PSRL_RewardModelReplicaBase):
    def __init__(
        self,
        replica_rank: int,
        reward_model_config: DictConfig,
        rollout_config: DictConfig,
        model_config: HFModelConfig,
        psrl_config: DictConfig,
        resource_pool: RayResourcePool,
    ):
        super().__init__(replica_rank, reward_model_config, rollout_config, model_config, psrl_config, resource_pool)

    async def init_replica(self):
        pass
    
    async def generate(self, request):
        pass

    async def generate_async(self, request):
        pass
