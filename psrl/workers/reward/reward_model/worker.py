import asyncio
import logging
import os
import queue
import time
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer, omega_conf_to_dataclass
from verl.utils.device import get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.model import get_generation_config, update_model_config

from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    get_worker_info,
    log_begin_event,
    log_dual_events,
    log_end_event,
    log_single_event,
)
from psrl.workers.config import HFModelConfig, RolloutConfig
from psrl.workers.gen import PSRL_vLLMRollout

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_RewardModelWorker(Worker):
    """
    A lightweight rollout worker dedicated to reward-model inference.

    The worker receives preprocessed prompts, forwards them to vLLM, and returns
    generated completions. No parameter-server specific logic is included.
    """

    @staticmethod
    def configure_worker(
        config,
        num_gpus: int | float,
        dp_idx: int,
        bundle_indices: list[int] | None,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        """
        Provides complete worker configuration (resource assignment, init args and environment variables)
        for vLLM tensor and pipeline parallelism.

        Args:
            config (DictConfig): The configuration for the worker.
            num_gpus (int | float): The number of GPUs available for the worker.
            dp_idx (int): The data parallel index of the worker.
            bundle_indices (list[int]): Indices of the bundles to which this worker belongs.

        Returns:
            tuple: A tuple containing:
                - resources (dict[str, Any]): Resources assigned to the worker.
                - env_vars (dict[str, str]): Environment variables for the worker.
                - init_kwargs (dict[str, Any]): Initialization arguments for the worker.
        """
        resources: dict[str, Any] = {}
        init_kwargs: dict[str, Any] = {}
        env_vars: dict[str, str] = {}
        
        # Nothing to do for SPMD-style synchronous rollout engines
        if config is not None and hasattr(config, "rollout") and config.rollout.mode == "sync":
            return resources, env_vars, init_kwargs
        
        resources["num_gpus"] = num_gpus
        psrl_logger.info("Configuring PSRL RewardModelWorker...")

        # Initialize configuration
        if bundle_indices is not None:
            bundle_id = bundle_indices[0] // len(bundle_indices)
            # NOTE: bundle_id is 0 if we prepare pg for each dp manually
            seed = dp_idx + 1000 + bundle_id

            init_kwargs["seed"] = seed
            # Need to give each DP group its own vllm cache to address:
            # https://github.com/vllm-project/vllm/issues/18851
            env_vars["VLLM_CACHE_ROOT"] = os.path.expanduser(f"~/.cache/vllm/vllm_{seed}")

        # Check if this worker is part of a parallel group (TP or TP+PP).
        is_part_of_parallel_workers = (
            bundle_indices is not None and len(bundle_indices) > 1
        ) or bundle_indices is None

        # Leave the GPU assignment management of inner parallel workers to vLLM + Ray
        if is_part_of_parallel_workers:
            resources["num_gpus"] = 0
            resources["num_cpus"] = 0
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env_vars["VLLM_SKIP_P2P_CHECK"] = "1"
        if config.rollout.disable_attn:
            warnings.warn(
                "CAUTION: you are disabling the attention, "
                "this should only be used for analysis purposes, not for training!",
                stacklevel=2,
            )
            env_vars["VLLM_DISABLE_ATTN"] = "1"
        return resources, env_vars, init_kwargs

    def __init__(
        self,
        config: DictConfig,
        role: str,
        psrl_config: DictConfig,
        **kwargs,
    ) -> None:
        """
        Initialize the PSRL RewardModelWorker.

        Args:
            config (DictConfig): The configuration for the worker.
            role (str): The role of the worker (e.g., "gen").
            psrl_config (DictConfig): The PSRL configuration.
            **kwargs: Additional keyword arguments, including 'seed'.
        """
        super().__init__()
        self.config = config
        self.psrl_config = psrl_config
        self.seed = kwargs.get("seed", 0)
        self.instance_id = kwargs.get("instance_id", 0)
        self.rollout: PSRL_vLLMRollout | None = None
        self.rollout_config: RolloutConfig | None = None
        self.model_config: HFModelConfig | None = None
        
        # Rollout loop management
        self._generate_loop = asyncio.get_running_loop()  # Background async loop for generation
        self.gen_task = None  # Generation task in stream mode
        self._generate_thread = None  # Generation thread in batch mode
        self._rollout_running = False
        self.active_tasks = set()  # Active tasks for the current generation loop

        # Async event management
        self._is_init_model = asyncio.Event()
        self._async_interrupt_event = asyncio.Event()
        
        self.request_num_queue = queue.Queue()
        self.request_id_to_active_tasks: dict[int, set[asyncio.Task]] = defaultdict(lambda: set())
        
        # Track start time for active tasks logging
        self.active_tasks_start_time = None

        if self.is_instance_representative_rank:
            logging_path = getattr(self.psrl_config, "logging_path", None)
            if logging_path:
                self.log_prefix = f"RewardModelWorker_I{self.instance_id}_R{getattr(self, 'rank', 0)}"
                psrl_logger.addHandler(DualOutputHandler(logging_path, self.log_prefix))
        psrl_logger.info(
            "Initialized RewardModelWorker(instance=%s, rank=%s, seed=%s) on %s",
            self.instance_id,
            getattr(self, "rank", 0),
            self.seed,
            get_worker_info(),
        )

    def _build_distributed(self):
        """Build the distributed process group for the rollout instance."""
        # Initialize the distributed process group
        if not dist.is_initialized():
            is_cuda_available = torch.cuda.is_available()
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            dist.init_process_group(
                backend="cpu:gloo,cuda:nccl" if is_cuda_available else "cpu:gloo,npu:hccl",
                rank=rank,
                world_size=world_size,
            )

    @property
    def is_instance_representative_rank(self) -> bool:
        return getattr(self, "rank", 0) == 0
    
    def get_instance_id(self) -> int:
        return self.instance_id

    def _build_rollout(self, trust_remote_code: bool = False) -> PSRL_vLLMRollout:
        """
        Build the rollout engine and sharding manager for the PSRL RewardModelWorker.
        NOTE: This method only supports building for one rollout instance at a time.
        """
        rollout_name = self.config.rollout.name
        assert rollout_name == "vllm", "Only support vLLM rollout for now"
        try:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
            model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        except Exception as e:
            psrl_logger.error(f"Failed to parse rollout config or model config: {e}")
            raise
        self.model_config = model_config
        tp = self.config.rollout.get("tensor_model_parallel_size", 1)
        pp = self.config.rollout.get("pipeline_model_parallel_size", 1)
        assert self.world_size == tp * pp, "Only support dp=1 for now"

        if self.config.rollout.mode == "sync":
            self._build_distributed()
        elif self.config.rollout.mode == "psrl_async":
            # NOTE(lhy): No need to build distributed for psrl_async mode
            # vllm will handle the distributed communication internally
            pass
        else:
            raise ValueError(f"Invalid rollout mode: {self.config.rollout.mode}")

        # Build the rollout engine
        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
        # Get the tokenizer
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        # Get the model config
        self.model_hf_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
        )
        # patch for kimi-vl
        if getattr(self.model_hf_config, "model_type", None) == "kimi_vl":
            self.model_hf_config.text_config.topk_method = "greedy"
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        if isinstance(override_model_config, dict) and "model_config" in override_model_config:
            override_config_kwargs.update(override_model_config["model_config"])  # Megatron model style
        else:
            override_config_kwargs.update(override_model_config)  # FSDP model style
        update_model_config(self.model_hf_config, override_config_kwargs=override_config_kwargs)
        if getattr(self, "rank", 0) == 0:
            psrl_logger.info(f"Model config after override: {self.model_hf_config}")

        get_torch_device().manual_seed(self.seed)

        psrl_logger.info(f"Building {rollout_name} rollout with seed {self.seed}.")
        rollout = PSRL_vLLMRollout(
            psrl_config=self.psrl_config,
            config=rollout_config,
            model_config=model_config,
            seed=self.seed,
            instance_id=self.get_instance_id(),
        )
        return rollout

    def _ensure_model_ready(self) -> None:
        if not self._is_init_model.is_set() or self.rollout is None:
            raise RuntimeError("RewardModelWorker is not initialized. Call init_model() first.")
    
    def get_active_task_num(self) -> int:
        """
        Get the number of active tasks.
        """
        return len(self.active_tasks)
    
    def log_active_tasks(self, task_added: bool = False, task_done: bool = False):
        """
        Log the active tasks.
        """
        assert task_added ^ task_done, "Exactly one of task_added or task_done must be True"
        active_task_num = self.get_active_task_num()
        psrl_logger.debug(f"Active tasks: {active_task_num}")
        if task_added and active_task_num == 1:
            self.active_tasks_start_time = time.time()
            log_begin_event(
                f"Streaming generate",
                psrl_logger,
                event_type=EventType.GEN,
            )
        if task_done and active_task_num == 0:
            if self.active_tasks_start_time is not None:
                duration = time.time() - self.active_tasks_start_time
                log_end_event(
                    f"Streaming generate",
                    psrl_logger,
                    event_type=EventType.GEN,
                    duration=duration,
                )
                self.active_tasks_start_time = None
        # if active_task_num % self.log_active_tasks_interval == 0:
        #     log_single_event(
        #         f"Active tasks: {active_task_num} ({active_task_num / self.avg_max_active_tasks_len * 100:.2f}%)",
        #         psrl_logger,
        #         event_type=EventType.OTHER,
        #     )

    async def _generate_async_task(self, request: DataProto):
        """
        An async task to generate sequences for a single request.

        Args:
            request (DataProto): The generation request.

        Returns:
            The processed generation result.
        """
        assert len(request) == 1, f"Expected request length to be 1, got {len(request)}"

        rollout_instance_id = self.get_instance_id()

        # Prepare the request for generation
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        request.meta_info.update(meta_info)
        request.non_tensor_batch["rollout_instance_id"] = np.array([rollout_instance_id] * len(request.batch))

        # Start the generation
        with log_dual_events("Reward model generate", psrl_logger, event_type=EventType.GEN):
            result = await self.rollout.generate_sequences_async(request)

            assert len(result) == 1, (
                f"Expected 1 output for single request, got {len(result)} outputs."
            )
            
        return result
    
    def _create_task_done_callback(self, request_id: int):
        # Remove from the active tasks tracker when the task is done
        def task_done_callback(task):
            self.request_id_to_active_tasks[request_id].discard(task)
            self.active_tasks.discard(task)
            self.log_active_tasks(task_done=True)

        return task_done_callback
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_model(self):
        with log_dual_events("Initialize reward model", psrl_logger, event_type=EventType.INIT):
            self.rollout = self._build_rollout()
        self._is_init_model.set()
    
    def generate(
        self,
        requests: DataProto,
        consolidate: bool = True,
        return_only_on_representative_rank: bool = True,
    ):
        """
        Generate completions for a batch of prompts.

        Args:
            requests: Batched prompts in DataProto format.
            consolidate: Whether to run rollout.post_process_outputs before returning.
            return_only_on_representative_rank: If True, non-zero ranks return (None, None).
        """
        self._ensure_model_ready()
        rollout_instance_id = self.get_instance_id()

        request_ids = requests.non_tensor_batch["uid"]
        psrl_logger.debug(
            f"Reward Model Rollout instance {rollout_instance_id} is generating requests with request ids: {request_ids}"
        )

        # Prepare the request for generation
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        requests.meta_info.update(meta_info)
        requests.non_tensor_batch["rollout_instance_id"] = np.array(
            [rollout_instance_id] * len(requests.batch)
        )

        with log_dual_events("Reward model generate", psrl_logger, event_type=EventType.GEN):
            outputs = self.rollout.raw_generate_sequences(requests)

            if return_only_on_representative_rank and not self.is_instance_representative_rank:
                return None

        final_outputs = (
            self.rollout.post_process_outputs(requests, outputs) if consolidate else outputs
        )
        return final_outputs

    async def generate_async(self, request: DataProto, consolidate: bool = True):
        """
        Async generation entry point. This covers the streaming RM inference path.
        """
        self._ensure_model_ready()
        assert len(request) == 1, f"Expected request length to be 1, got {len(request)}"
        
        if self._async_interrupt_event and self._async_interrupt_event.is_set():
            psrl_logger.debug("Once generation is interrupted, we will not generate again")
            return None
        
        request_id = int(request.non_tensor_batch["uid"][0])
        
        task = self._generate_loop.create_task(self._generate_async_task(request))
        task.add_done_callback(
            self._create_task_done_callback(request_id)
        )
        self.request_id_to_active_tasks[request_id].add(task)
        self.active_tasks.add(task)
        self.log_active_tasks(task_added=True)
        # Wait for the task to finish
        result = await task
        return result
