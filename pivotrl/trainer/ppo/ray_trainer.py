import json
import logging
import math
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from ray.exceptions import RayTaskError
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tqdm import tqdm
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import RayResourcePool, SubRayResourcePool, create_colocated_worker_cls_fused
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.utils import WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.tracking import ValidationGenerationsLogger

from pivotrl.trainer.ppo.utils import (
    PivotRL_compute_advantage,
    PivotRL_Role,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_response_mask,
    need_critic,
    need_reference_policy,
    need_reward_model,
    record_rollout_rm_metrics,
)
from pivotrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from pivotrl.utils.common.tms_env import build_ray_train_worker_tms_env
from pivotrl.utils.common.worker_naming import WorkerKey, ps_agent_name, train_client_name
from pivotrl.utils.dataset import DataProcessor, DatasetType
from pivotrl.utils.elastic_rm.elastic_executor import ElasticExecutor
from pivotrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_data_protocol,
    log_dual_events,
)
from pivotrl.utils.nixl import (
    GLOBAL_PORT_SCANNER,
    NIXLInterface,
    compare_weight_fingerprints,
    resolve_weight_fingerprint_options,
    weight_fingerprint_flow_enabled,
)
from pivotrl.utils.reward_token_metrics import extract_reward_model_token_counts
from pivotrl.utils.server.command import Command, CommandType
from pivotrl.workers.agent_loop import PivotRL_AgentLoopManager, PivotRL_AgentLoopWorker
from pivotrl.workers.agent_loop.router import RolloutRouter
from pivotrl.workers.gen import GenInterface, RolloutCoordinator
from pivotrl.workers.gen.rollout_gateway import RolloutGateway
from pivotrl.workers.ps import (
    PSClassWithInitArgs,
    PSManager,
    PSResourcePool,
    PSResourceSpec,
    PSStoragePlan,
    PSStorageWorker,
    PSWorkerGroup,
)
from pivotrl.workers.reward import RewardManager
from pivotrl.workers.reward.reward_model import PivotRL_RewardModelManager
from pivotrl.workers.train import TrainInterface

pivotrl_logger = logging.getLogger(__file__)
pivotrl_logger.setLevel(os.getenv("PIVOTRL_LOGGING_LEVEL", "WARN"))


class PivotRL_RayPPOTrainer:
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[PivotRL_Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        collate_fn=None,
        group_post_process_fn=None,
        buffer_post_process_fn=None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process and is responsible for managing the training process.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[PivotRL_Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resources.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data.
            reward_fn: Function to compute rewards for the training data.
            collate_fn: Optional function to collate data into batches.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.collate_fn = collate_fn
        self.group_post_process_fn = group_post_process_fn
        self.buffer_post_process_fn = buffer_post_process_fn

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls  # NOTE: ray_worker_group_cls is used only in train side
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.train_actor_rollout_ref.model.get("lora_rank", 0) > 0

        # CPU workers for Streaming Rollout
        self.data_processor = None
        self.agent_loop_manager = None
        self.rollout_coordinator = None
        self.reward_manager = None

        self.reward_model_status_queues_mapping = {}
        self.reward_model_manager_mapping = {}

        # Elastic rm
        # Resolve unified deployment.mode into concrete flags before reading them.
        # Mutates config in place so that downstream code (main_ppo pool spec, init
        # paths) observes a consistent set of flags regardless of how the user
        # configured the job.
        self.deployment_mode = self._resolve_deployment_mode(self.config)
        self.colocated_mode = self.deployment_mode == "colocated"
        self.rollout_rm_colocated_mode = self.deployment_mode == "rollout_rm_colocated"
        self.trainer_pool_only_mode = self.deployment_mode == "trainer_pool_only"
        # SleepWakeOrchestrator drives phased sleep/wake for colocated modes.
        self.sleep_wake_orchestrator = None

        self.elastic_rm_mode = config.pivotrl.deployment.elastic_rm.enable
        self.elastic_trainer_pool_mode = bool(config.pivotrl.deployment.elastic_rm.get("enable_trainer_pool", False))
        self.elastic_executor = None
        # Filled in init_workers when elastic_rm_mode: SubRayResourcePool bundle ranges [start, end) per instance.
        self._elastic_bundle_range_by_rollout_instance: list[tuple[int, int]] | None = None
        self._elastic_bundle_range_by_reward_model: dict[str, list[tuple[int, int]]] | None = None
        self._elastic_pool_id_by_rollout_instance: list[str] | None = None
        self._elastic_pool_id_by_reward_model: dict[str, list[str]] | None = None
        self._elastic_trainer_pool_training_active = False
        self._elastic_trainer_pool_trainer_sleeping = False
        self._elastic_trainer_pool_entries: list[dict] | None = None
        self._trainer_before_sleep_weight_fingerprints: dict | None = None

        # Rollout gateway handle
        self.rollout_gateway = None
        self.gateway_base_url = None

        # Parameter server handle for other workers to access
        self.ps_manager_handle = None

        # Async rollout mode for training worker
        self.async_rollout_mode = False

        # Indicate whether current mode is rollout mode in actor
        self.is_rollout_mode_in_actor = (
            self.config.pivotrl.colocate_validate_and_train and self.config.trainer.val_before_train
        )
        pivotrl_logger.info(
            f"Initializing PivotRL_RayPPOTrainer with is_rollout_mode_in_actor: {self.is_rollout_mode_in_actor}"
        )

        # Mappings from WorkerKey to Ray node id and PS instance index for NIXL.
        self.worker_to_node_id: dict[WorkerKey, str] = {}
        self.worker_to_ps_idx: dict[WorkerKey, int] = {}

        self.n_rollout_instances = self.config.pivotrl.deployment.n_rollout_instances
        self.n_validate_instances = (
            self.config.pivotrl.deployment.n_validate_instances if self.config.pivotrl.colocate_validate_and_train else 0
        )

        if self.config.pivotrl.redundant_rollout.enable:
            self.max_concurrency = (
                self.config.pivotrl.redundant_rollout.redundant_rollout_n
                * self.config.pivotrl.redundant_rollout.redundant_global_batch_size
                * (self.config.pivotrl.staleness + 1)
            )
        else:
            self.max_concurrency = (
                self.config.pivotrl.rollout_n
                * self.config.pivotrl.staleness_buffer_entries
                * (self.config.pivotrl.staleness + 1)
            )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        # Build logger
        self.log_prefix = "MainRayTrainer"
        pivotrl_logger.addHandler(DualOutputHandler(self.config.pivotrl.logging_path, self.log_prefix))
        pivotrl_logger.info("Initialized major ray trainer (single controller).")

        self._initialize_queue_buffers()

        self._validate_config()

        self._init_ps_manager()

        # initialize data processor
        # NOTE: data processor must be initialized before initializing other workers
        # so that the total_training_steps can be obtained and the optimizer config
        # (related to weight decay, lr schedule, etc.) can be set
        # otherwise, it will cause error when running Megatron backend
        self._init_data_processor()

    @staticmethod
    def _select_opd_teacher_value(reward_result: dict, field: str):
        teacher_values = reward_result.get(field, {})
        if not isinstance(teacher_values, dict) or not teacher_values:
            return None
        for key, value in teacher_values.items():
            if str(key).startswith("opd/"):
                return value
        return next(iter(teacher_values.values()))

    def _normalize_sync_reward_tensor(self, batch: DataProto, reward_tensor: torch.Tensor | None) -> torch.Tensor | None:
        """Apply reward normalization after sync rewards have been gathered into a trainer batch."""
        reward_normalization = self.config.reward_models_config.reward_normalization
        if reward_tensor is None or reward_normalization not in ("batch", "group"):
            return reward_tensor

        if reward_normalization == "batch":
            if "data_source" not in batch.non_tensor_batch:
                pivotrl_logger.warning("Skip sync batch reward normalization because data_source is missing.")
                return reward_tensor
            group_ids = batch.non_tensor_batch["data_source"]
        else:
            if "parent_id" in batch.non_tensor_batch:
                group_ids = batch.non_tensor_batch["parent_id"]
            elif "uid" in batch.non_tensor_batch:
                group_ids = batch.non_tensor_batch["uid"]
            else:
                pivotrl_logger.warning("Skip sync group reward normalization because neither parent_id nor uid exists.")
                return reward_tensor

        if hasattr(group_ids, "tolist"):
            group_ids = group_ids.tolist()

        reward_scores = reward_tensor.sum(dim=-1).to(torch.float32)
        normalized_scores = reward_scores.clone()
        group_to_indices = defaultdict(list)
        for idx, group_id in enumerate(group_ids):
            group_to_indices[group_id].append(idx)

        for indices in group_to_indices.values():
            group_scores = reward_scores[indices]
            normalized_scores[indices] = (group_scores - group_scores.mean()) / (
                group_scores.std(unbiased=False) + 1e-8
            )

        normalized_reward_tensor = torch.zeros_like(reward_tensor, dtype=torch.float32)
        attention_mask = batch.batch.get("attention_mask", None)
        prompt_len = batch.batch["prompts"].size(1)
        for idx, score in enumerate(normalized_scores):
            if attention_mask is not None:
                valid_len = int(attention_mask[idx, prompt_len:].sum().item()) - 1
            else:
                valid_len = int(batch.batch["response_mask"][idx].sum().item()) - 1
            if valid_len < 0:
                continue
            normalized_reward_tensor[idx, valid_len] = score
        return normalized_reward_tensor

    @classmethod
    def _merge_teacher_tensor_from_rewards(
        cls,
        request_id_to_reward: dict,
        request_ids: list[int],
        response_mask: torch.Tensor,
        field: str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        teacher_values = []
        missing_count = 0
        mismatch_count = 0
        topk_width = 0
        for request_id in request_ids:
            value = cls._select_opd_teacher_value(request_id_to_reward[request_id], field)
            if value is None:
                teacher_values.append(None)
                missing_count += 1
                continue
            tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.tensor(value)
            if tensor.ndim == 2:
                topk_width = max(topk_width, int(tensor.shape[-1]))
            teacher_values.append(tensor)

        if missing_count == len(request_ids):
            return None, {}

        batch_size, response_width = response_mask.shape
        output_shape = (batch_size, response_width, topk_width) if topk_width > 0 else (batch_size, response_width)
        merged = torch.zeros(output_shape, dtype=dtype)
        for idx, tensor in enumerate(teacher_values):
            if tensor is None:
                continue
            valid_len = int(response_mask[idx].sum().item())
            if tensor.shape[0] != valid_len:
                mismatch_count += 1
            copy_len = min(int(tensor.shape[0]), valid_len, response_width)
            if copy_len <= 0:
                continue
            if topk_width > 0:
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(-1)
                copy_width = min(int(tensor.shape[-1]), topk_width)
                merged[idx, :copy_len, :copy_width] = tensor[:copy_len, :copy_width].to(dtype)
            else:
                merged[idx, :copy_len] = tensor[:copy_len].to(dtype)

        metrics = {
            f"distillation/{field}_missing": float(missing_count),
            f"distillation/{field}_mismatch": float(mismatch_count),
        }
        if topk_width > 0:
            metrics[f"distillation/{field}_topk"] = float(topk_width)
        return merged, metrics

    def _initialize_queue_buffers(self):
        if self.config.pivotrl.redundant_rollout.enable:
            self.rollout_n = self.config.pivotrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.pivotrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, (
            f"Rollout n {self.rollout_n} must be greater than or equal to alg_rollout_n {self.alg_rollout_n}."
        )

        # Data queue is the communication handle between the data processor and the rollout server.
        # The size of the queue is determined by the batch size and the rollout n.
        self.data_queue_size = (
            self.config.data.get("gen_batch_size", self.config.data.train_batch_size) * self.rollout_n
        )

        # Status queues are used to store the status of the rollout instances.
        # The status is collected by the rollout coordinator and sent to the agent loop workers.
        # The number of status queues is the same as the number of rollout instances.
        self.status_queues = [RayQueue() for _ in range(self.n_rollout_instances + self.n_validate_instances)]

        for reward_model in self.config.reward_models_config.reward_models:
            if reward_model.reward_loop_type not in ("gen", "opd"):
                continue
            reward_model_name = reward_model.get("reward_model_name", reward_model.model.path.split("/")[-1])
            reward_model_replica_num = reward_model.get("num_replicas", 1)
            self.reward_model_status_queues_mapping[reward_model_name] = [
                RayQueue() for _ in range(reward_model_replica_num)
            ]

        pivotrl_logger.debug(
            "Initialized data_queue, and status_queue with sizes: %d, and unlimited respectively.",
            self.data_queue_size,
        )

    _VALID_DEPLOYMENT_MODES = (
        "disaggregated",
        "colocated",
        "rollout_rm_colocated",
        "trainer_pool_only",
        "elastic_rl",
    )

    @staticmethod
    def _resolve_deployment_mode(config) -> str:
        """Resolve `pivotrl.deployment.mode` into concrete elastic_rm flags.

        Thin wrapper around `pivotrl.utils.deployment_mode.resolve_deployment_mode`
        so the trainer derives its flags from the same single source of truth
        used by `main_ppo.TaskRunner` (which resolves the mode before laying out
        resource pools). Mutates `config` in place; idempotent.
        """
        from pivotrl.utils.deployment_mode import resolve_deployment_mode

        return resolve_deployment_mode(config)

    def _validate_config(self):
        config = self.config
        from pivotrl.utils.deployment_mode import validate_trainer_sleep_optimizer_offload

        validate_trainer_sleep_optimizer_offload(config, self.deployment_mode)
        # number of GPUs used in training
        train_n_gpus = config.pivotrl.deployment.train_ngpus_per_node * config.pivotrl.deployment.train_nnodes
        if config.train_actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            context_parallel_size = config.train_actor_rollout_ref.actor.megatron.context_parallel_size
            assert train_n_gpus % (model_parallel_size * context_parallel_size) == 0, (
                f"train_n_gpus ({train_n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times"
                f" context_parallel_size ({context_parallel_size})"
            )
            megatron_dp = train_n_gpus // (model_parallel_size * context_parallel_size)
            minimal_bsz = megatron_dp * config.train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = train_n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.train_actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "train_actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "train_actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "train_actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.train_actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.train_actor_rollout_ref.actor.ppo_micro_batch_size,
                config.train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "train_actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.train_actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.train_actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "train_actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.train_actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "train_actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size,
                config.critic.ppo_micro_batch_size_per_gpu,
                "critic",
            )

        # Check for reward model micro-batch size conflicts
        # if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
        #     check_mutually_exclusive(
        #         config.reward_model.micro_batch_size,
        #         config.reward_model.micro_batch_size_per_gpu,
        #         "reward_model",
        #     )

        # Actor training
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.train_actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.train_actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.train_actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.train_actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.train_actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.train_actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.train_actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= train_n_gpus

        assert config.train_actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.train_actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.train_actor_rollout_ref.actor.use_kl_loss:
            pivotrl_logger.info("NOTICE: You have both enabled in-reward kl and kl loss.")

        # Critic training
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= train_n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.train_actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"} and (
            config.train_actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.train_actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.train_actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy in {"fsdp", "fsdp2"}:
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # Check eval config
        if config.train_actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.train_actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # check multi_turn with tool config
        if config.train_actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.train_actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # Check NIXL compatibility
        if self.config.pivotrl.ps_mode == "nixl_cpu" or self.config.pivotrl.ps_mode == "nixl_gpu":
            assert self.config.pivotrl.nixl.server_ip == self.config.pivotrl.ps_manager_ip, (
                "PSManager IP and NIXL server IP must be the same"
            )
            assert self.config.train_actor_rollout_ref.actor.strategy != "fsdp", (
                "FSDP1 is not supported for NIXL because it uses flat_param"
            )
            pivotrl_logger.info(
                f"NOTICE: NIXL is enabled. Actor strategy used is {self.config.train_actor_rollout_ref.actor.strategy}"
            )

        # Check colocate mode
        if self.config.pivotrl.colocate:
            assert self.config.pivotrl.staleness == 0, "staleness must be 0 when using colocate mode"

        # Check unified deployment mode constraints.
        if self.colocated_mode or self.rollout_rm_colocated_mode or self.trainer_pool_only_mode:
            assert self.config.pivotrl.ps_mode in ("nixl_cpu", "nixl_gpu"), (
                f"deployment.mode={self.deployment_mode} requires pivotrl.ps_mode to be 'nixl_cpu' or 'nixl_gpu' "
                f"(got {self.config.pivotrl.ps_mode!r})."
            )
            assert not self.config.pivotrl.colocate, (
                "pivotrl.colocate (legacy sync colocate) is incompatible with deployment.mode colocated/"
                "rollout_rm_colocated/trainer_pool_only."
            )
        if self.colocated_mode:
            assert not self.config.pivotrl.colocate_validate_and_train, (
                "colocate_validate_and_train is incompatible with deployment.mode=colocated "
                "(trainer participates in the shared-pool sleep/wake cycle)."
            )
            # Sequential per-buffer phasing needs a single in-flight buffer.
            assert self.config.pivotrl.staleness == 0, (
                f"deployment.mode={self.deployment_mode} requires pivotrl.staleness=0 "
                "(per-buffer sequential rollout->reward phasing)."
            )
        if self.colocated_mode or self.rollout_rm_colocated_mode:
            # Phased sleep/wake relies on async reward so that a buffer becomes
            # "ready" at rollout-complete (rewards still queued in the rm router,
            # processed during the REWARD phase). With sync reward the rm would
            # block the rollout worker while asleep -> deadlock.
            assert self.config.reward_models_config.launch_reward_fn_async, (
                f"deployment.mode={self.deployment_mode} requires "
                "reward_models_config.launch_reward_fn_async=True."
            )
        if self.trainer_pool_only_mode:
            assert self.config.pivotrl.deployment.trainer_pool_idle_rollout_instances >= 0, (
                "trainer_pool_idle_rollout_instances must be >= 0."
            )
            assert self.config.pivotrl.deployment.trainer_pool_idle_rm_instances >= 0, (
                "trainer_pool_idle_rm_instances must be >= 0."
            )
            idle_rollout = int(self.config.pivotrl.deployment.trainer_pool_idle_rollout_instances)
            idle_rm = int(self.config.pivotrl.deployment.trainer_pool_idle_rm_instances)
            train_pool_gpus = (
                int(self.config.pivotrl.deployment.train_nnodes)
                * int(self.config.pivotrl.deployment.train_ngpus_per_node)
            )
            if idle_rollout + idle_rm > train_pool_gpus:
                raise ValueError(
                    f"deployment.mode=trainer_pool_only requires "
                    f"trainer_pool_idle_rollout_instances + trainer_pool_idle_rm_instances "
                    f"<= train_nnodes * train_ngpus_per_node "
                    f"({idle_rollout} + {idle_rm} = {idle_rollout + idle_rm} > {train_pool_gpus}). "
                    "Idle rollout and rm replicas on train_pool must use disjoint GPU bundles."
                )

        # Check validate mode
        if self.config.pivotrl.colocate_validate_and_train:
            assert self.config.pivotrl.tms.range == "train" or self.config.pivotrl.tms.range == "all", (
                "TMS range must be 'train' or 'all' when using colocate_validate_and_train"
            )
            if self.elastic_trainer_pool_mode:
                assert self.config.pivotrl.ps_mode in ("nixl_cpu", "nixl_gpu"), (
                    "Elastic trainer-pool validation colocation requires pivotrl.ps_mode "
                    f"to be 'nixl_cpu' or 'nixl_gpu', got: {self.config.pivotrl.ps_mode!r}."
                )
                assert not self.config.pivotrl.fuse_rollout_with_validate, (
                    "Elastic trainer-pool validation colocation requires "
                    "pivotrl.fuse_rollout_with_validate=False."
                )
        else:
            assert self.config.pivotrl.fuse_rollout_with_validate, (
                "fuse_rollout_with_validate must be enabled when not colocate_validate_and_train"
            )

        # Check routing strategy
        if (
            self.config.pivotrl.routing_strategy.method == "request_num_balance"
            or self.config.pivotrl.routing_strategy.method == "throughput_balance"
        ):
            assert self.config.pivotrl.status_collection.enable, (
                "status collection must be enabled when using "
                "request num balance or throughput balance routing strategy"
            )

        # Check TMS configuration
        if self.config.pivotrl.tms.enable_cuda_graph:
            assert self.config.pivotrl.tms.range == "all", "TMS CUDA graph can only be enabled when TMS range is 'all'"
        if self.config.pivotrl.tms.range not in ["train", "all"]:
            assert (
                self.config.train_actor_rollout_ref.actor.strategy == "megatron"
                and self.config.train_actor_rollout_ref.actor.megatron.optimizer_offload
                or self.config.train_actor_rollout_ref.actor.strategy == "fsdp2"
                and self.config.train_actor_rollout_ref.actor.fsdp_config.optimizer_offload
            ), "Optimizer offload must be enabled when TMS is not enabled for training workers"

        pivotrl_logger.info("[validate_config] All configuration checks passed successfully!")

    def _init_ps_manager(self):
        """Initialize the PS manager for handling model version, requests condition and staleness."""
        # Set the validation rollout number in the config
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "pivotrl"):
                    self.config.pivotrl.val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n
        except Exception as e:
            pivotrl_logger.warning(f"Could not set val_rollout_n in config. Structure missing? Error: {e}")

        ip_to_node_id = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes()}
        assert self.config.pivotrl.ps_manager_ip in ip_to_node_id, (
            f"PSManager IP {self.config.pivotrl.ps_manager_ip} not found in ray nodes"
        )
        pivotrl_logger.info("Getting the handle of the PSManager")
        self.ps_manager_handle = (
            ray.remote(PSManager)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ip_to_node_id[self.config.pivotrl.ps_manager_ip], soft=False
                )
            )
            .remote(self.config.pivotrl)
        )

    def _init_data_processor(self):
        """Initialize the data processor for handling data preprocessing and batching."""
        if self.data_processor is not None:
            return

        # Initialize the data processor
        self.data_processor = DataProcessor.remote(
            self.config, self.tokenizer, self.processor, self.ps_manager_handle, collate_fn=self.collate_fn
        )

        # Get total training steps from the data processor where dataloaders are built
        self.total_training_steps = ray.get(self.data_processor.get_total_training_steps.remote())

        pivotrl_logger.info(f"Total training steps: {self.total_training_steps}")

        # Set the total training steps in the config
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "train_actor_rollout_ref.actor.optim"):
                    self.config.train_actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = self.total_training_steps
        except Exception as e:
            pivotrl_logger.warning(f"Could not set total_training_steps in config. Structure missing? Error: {e}")

    def start_data_processor(self):
        """Launch the data processor for processing data in the background."""
        assert self.data_processor is not None, "Data processor must be initialized before starting it."

        ray.get(self.data_processor.start_busy_loop.remote())

    def stop_data_processor(self):
        """Stop the data processor."""
        if self.data_processor is not None:
            pivotrl_logger.debug("Stopping data processor...")
            ray.get(self.data_processor.stop_busy_loop.remote())
            self.data_processor = None
            pivotrl_logger.debug("Data processor stopped successfully.")
        else:
            pivotrl_logger.warning("Data processor is not initialized, skipping stop operation.")

    def init_agent_loop_manager(self):
        if self.agent_loop_manager is not None:
            return

        # Initialize the agent loop manager
        self.agent_loop_manager = (
            ray.remote(PivotRL_AgentLoopManager)
            .options(max_concurrency=self.max_concurrency)
            .remote(
                self.config,
                self.data_queue_size,
                self.agent_loop_workers,
                self.ps_manager_handle,
                self.gateway_base_url,
                group_post_process_fn=self.group_post_process_fn,
                buffer_post_process_fn=self.buffer_post_process_fn,
            )
        )

    def start_agent_loop_manager(self):
        """Start the agent loop manager to handle agent loops in the background."""
        assert self.agent_loop_manager is not None, "Agent loop manager must be initialized before starting it."

        ray.get(self.agent_loop_manager.start_busy_loop.remote())

    def stop_agent_loop_manager(self):
        """Stop the agent loop manager."""
        if self.agent_loop_manager is not None:
            pivotrl_logger.debug("Stopping agent loop manager...")
            ray.get(self.agent_loop_manager.stop_busy_loop.remote())
            self.agent_loop_manager = None
            pivotrl_logger.debug("Agent loop manager stopped successfully.")
        else:
            pivotrl_logger.warning("Agent loop manager is not initialized, skipping stop operation.")

    def init_rollout_router(self):
        """Initialize the rollout router for routing requests to rollout instances."""
        self.rollout_router = RolloutRouter.options(max_concurrency=self.max_concurrency).remote(
            self.config,
            self.ps_manager_handle,
            self.tokenizer,
            self.rollout_wg_list + self.validate_wg_list,
        )

    def start_rollout_gateway(self):
        """Start Rollout Gateway as a Ray actor (no Ray Serve)."""
        if not self.config.pivotrl.server_rollout.enable or self.rollout_gateway is not None:
            return

        ip_to_node_id = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes()}
        assert self.rollout_router is not None, "Rollout router must be initialized before starting gateway."

        self.rollout_gateway = (
            ray.remote(RolloutGateway)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ip_to_node_id[self.config.pivotrl.server_rollout.gateway.router_ip],
                    soft=False,
                )
            )
            .remote(
                host=self.config.pivotrl.server_rollout.gateway.get("router_ip", "127.0.0.1"),
                port=int(self.config.pivotrl.server_rollout.gateway.get("router_port", 8000)),
                concurrency=int(self.config.pivotrl.server_rollout.get("server_concurrency", 64)),
                n_rollout_instances=int(self.config.pivotrl.deployment.get("n_rollout_instances", 1)),
                rollout_router=self.rollout_router,
            )
        )

        ray.get(self.rollout_gateway.start.remote())

        bind = ray.get(self.rollout_gateway.get_bind.remote())
        self.gateway_base_url = f"http://{bind['host']}:{bind['port']}"
        assert bind["host"] == self.config.pivotrl.server_rollout.gateway.get("router_ip", "127.0.0.1"), (
            "Rollout Gateway host must be the same as router_ip"
        )
        assert bind["port"] == int(self.config.pivotrl.server_rollout.gateway.get("router_port", 8000)), (
            "Rollout Gateway port must be the same as router_port"
        )
        pivotrl_logger.info(f"Rollout Gateway started at {self.gateway_base_url}")

        futures = []
        for i in range(self.n_rollout_instances):
            # Configure the rollout instance's representative rank GenWorker to
            # self-start an in-process HTTP server and register to gateway.
            futures.append(
                self.rollout_wg_list[i].execute_rank_zero_async("set_rollout_gateway_base_url", self.gateway_base_url)
            )
        ray.get(futures)

    def stop_rollout_gateway(self):
        """Stop Rollout Gateway actor if it's running."""
        if self.rollout_gateway is None:
            return
        ray.get(self.rollout_gateway.stop.remote())

    def init_rollout_coordinator(self):
        assert self.rollout_router is not None, (
            "Rollout router must be initialized before initializing rollout coordinator."
        )
        self.rollout_coordinator = RolloutCoordinator.remote(
            self.config,
            self.rollout_router,
            self.rollout_wg_list,
            self.validate_wg_list,
            self.agent_loop_workers,
            self.status_queues,
        )

    def start_rollout_coordinator(self):
        assert self.rollout_coordinator is not None, "Rollout coordinator must be initialized before starting it."

        ray.get(self.rollout_coordinator.start_busy_loop.remote())

    def stop_rollout_coordinator(self):
        """Stop the rollout coordinator."""
        if self.rollout_coordinator is not None:
            pivotrl_logger.debug("Stopping rollout coordinator...")
            ray.get(self.rollout_coordinator.stop_busy_loop.remote())
            self.rollout_coordinator = None
            pivotrl_logger.debug("Rollout coordinator stopped successfully.")
        else:
            pivotrl_logger.warning("Rollout coordinator is not initialized, skipping stop operation.")

    @staticmethod
    def _select_non_conflicting_awake_ids(
        instance_to_bundle_indices: dict[int, set[tuple[str, int]]],
        target_awake_num: int,
        occupied_bundle_indices: set[tuple[str, int]],
        min_awake_num: int = 1,
    ) -> tuple[list[int], set[tuple[str, int]]]:
        """Pick instance ids whose placement bundles do not overlap ``occupied_bundle_indices`` (elastic RM PG)."""
        n_target = int(target_awake_num)
        if n_target <= 0:
            if min_awake_num > 0:
                raise RuntimeError(
                    "Cannot select non-conflicting awake instances: target_awake_num<=0 but "
                    f"min_awake_num={min_awake_num} is positive."
                )
            return [], occupied_bundle_indices
        selected_ids: list[int] = []
        for instance_id in sorted(instance_to_bundle_indices.keys()):
            bundle_indices = instance_to_bundle_indices[instance_id]
            if not bundle_indices.isdisjoint(occupied_bundle_indices):
                continue
            selected_ids.append(instance_id)
            occupied_bundle_indices.update(bundle_indices)
            if len(selected_ids) >= n_target:
                break
        if len(selected_ids) < min_awake_num:
            raise RuntimeError(
                "Cannot select non-conflicting awake instances "
                f"(target={n_target}, selected={len(selected_ids)}, min_required={min_awake_num})."
            )
        return selected_ids, occupied_bundle_indices

    def _init_elastic_rm_runtime(self):
        if not self.elastic_rm_mode:
            return
        if self.rollout_coordinator is None:
            raise RuntimeError("Rollout coordinator must be initialized before elastic_rm init.")

        pivotrl_logger.info("Initializing elastic_executor runtime (coordinator sleep/wake, ElasticExecutor).")
        rollout_model_name = self.config.gen_actor_rollout_ref.model.path.split("/")[-1]
        rollout_instance_num = len(self.rollout_wg_list)
        pivotrl_logger.info(
            "Elastic_RM: rollout model_name=%s, n_instances=%d",
            rollout_model_name,
            rollout_instance_num,
        )
        rollout_all_ids = list(range(rollout_instance_num))

        # Enable coordinator command handling before elastic sleep/wake orchestration.
        self.start_rollout_coordinator()
        pivotrl_logger.info("Rollout coordinator busy loop started for elastic_rm command handling.")

        if rollout_all_ids:
            pivotrl_logger.info(
                "Elastic_RM: putting all rollout instances to sleep (instance_ids=%s).",
                rollout_all_ids,
            )
            sleep_result = ray.get(
                self.rollout_coordinator.exec_command.remote(
                    Command(type=CommandType.SLEEP, instance_ids=rollout_all_ids),
                    blocking=True,
                )
            )
            if sleep_result is not True:
                raise RuntimeError(
                    f"Failed to put rollout instances {rollout_all_ids} to sleep: {sleep_result!r}"
                )
            pivotrl_logger.info("Elastic_RM: all rollout instances slept.")

        registrations = [
            {
                "role_name": PivotRL_Role.Rollout,
                "model_name": rollout_model_name,
                "num_instances": rollout_instance_num,
            }
        ]
        if self._elastic_bundle_range_by_rollout_instance is None or len(self._elastic_bundle_range_by_rollout_instance) != rollout_instance_num:
            raise RuntimeError(
                "Elastic_RM: bundle ranges for rollout instances are missing or mismatched; "
                "expected init_workers to populate _elastic_bundle_range_by_rollout_instance."
            )
        if self._elastic_pool_id_by_rollout_instance is None or (
            len(self._elastic_pool_id_by_rollout_instance) != rollout_instance_num
        ):
            raise RuntimeError(
                "Elastic_RM: pool ids for rollout instances are missing or mismatched; "
                "expected init_workers to populate _elastic_pool_id_by_rollout_instance."
            )
        bundle_mappings = []
        rollout_instance_to_bundle_indices: dict[int, set[tuple[str, int]]] = {}
        for instance_id in range(rollout_instance_num):
            br = self._elastic_bundle_range_by_rollout_instance[instance_id]
            pool_id = self._elastic_pool_id_by_rollout_instance[instance_id]
            rollout_instance_to_bundle_indices[instance_id] = {
                (pool_id, bundle_idx) for bundle_idx in range(br[0], br[1])
            }
            bundle_mappings.append(
                {
                    "role_name": PivotRL_Role.Rollout,
                    "model_name": rollout_model_name,
                    "instance_id": instance_id,
                    "bundle_range": (br[0], br[1]),
                    "pool_id": pool_id,
                }
            )

        pivotrl_logger.info(
            "Elastic_RM: collected rollout bundle mappings (%d entries).",
            len(bundle_mappings),
        )

        reward_coordinators: dict[str, ray.actor.ActorHandle] = {}
        reward_model_to_instance_bundle_indices: dict[str, dict[int, set[tuple[str, int]]]] = {}
        for reward_model_name, manager in self.reward_model_manager_mapping.items():
            rm_instance_num = len(manager.reward_model_wg_list)
            rm_all_ids = list(range(rm_instance_num))

            pivotrl_logger.info(
                "Elastic_RM: putting reward model replicas to sleep (name=%s, instance_ids=%s).",
                reward_model_name,
                rm_all_ids,
            )
            sleep_result = ray.get(
                manager.reward_model_coordinator.exec_command.remote(
                    Command(type=CommandType.SLEEP, instance_ids=rm_all_ids),
                    blocking=True,
                )
            )
            if sleep_result is not True:
                raise RuntimeError(
                    f"Failed to put reward model {reward_model_name} instances {rm_all_ids} to sleep: "
                    f"{sleep_result!r}"
                )
            pivotrl_logger.info("Elastic_RM: reward model %s replicas slept.", reward_model_name)

            reward_coordinators[reward_model_name] = manager.reward_model_coordinator
            reward_model_to_instance_bundle_indices[reward_model_name] = {}
            registrations.append(
                {
                    "role_name": PivotRL_Role.RewardModel,
                    "model_name": reward_model_name,
                    "num_instances": rm_instance_num,
                }
            )
            rm_ranges = self._elastic_bundle_range_by_reward_model.get(reward_model_name)
            if rm_ranges is None or len(rm_ranges) != rm_instance_num:
                raise RuntimeError(
                    f"Elastic_RM: bundle ranges for reward model {reward_model_name} are missing or "
                    f"mismatched (expected {rm_instance_num} entries)."
                )
            rm_pool_ids = (self._elastic_pool_id_by_reward_model or {}).get(reward_model_name)
            if rm_pool_ids is None or len(rm_pool_ids) != rm_instance_num:
                raise RuntimeError(
                    f"Elastic_RM: pool ids for reward model {reward_model_name} are missing or "
                    f"mismatched (expected {rm_instance_num} entries)."
                )
            for instance_id in range(rm_instance_num):
                br = rm_ranges[instance_id]
                pool_id = rm_pool_ids[instance_id]
                reward_model_to_instance_bundle_indices[reward_model_name][instance_id] = {
                    (pool_id, bundle_idx) for bundle_idx in range(br[0], br[1])
                }
                bundle_mappings.append(
                    {
                        "role_name": PivotRL_Role.RewardModel,
                        "model_name": reward_model_name,
                        "instance_id": instance_id,
                        "bundle_range": (br[0], br[1]),
                        "pool_id": pool_id,
                    }
                )

        # Modes 2 (colocated) and 3 (rollout_rm_colocated): replace ElasticExecutor
        # with SleepWakeOrchestrator for phased time-multiplexed sleep/wake. All
        # rollout/rm instances were just SLEPT above; the orchestrator wakes the
        # ROLLOUT role on start(). No bundle mapping / ElasticExecutor needed.
        if self.colocated_mode or self.rollout_rm_colocated_mode:
            self._init_sleep_wake_orchestrator(rollout_model_name, reward_coordinators)
            return

        pivotrl_logger.info(
            "Elastic_RM: total instance bundle mapping entries (rollout + reward)=%d, registration_roles=%d.",
            len(bundle_mappings),
            len(registrations),
        )
        if self._elastic_bundle_range_by_rollout_instance is not None:
            pivotrl_logger.info(
                "Elastic_RM: per-instance placement bundles (pool-local linear bundle indices, range [start, end)):"
            )
            for i, (b_start, b_end) in enumerate(self._elastic_bundle_range_by_rollout_instance):
                pivotrl_logger.info(
                    "Elastic_RM:   [Rollout] instance_id=%d pool_id=%s bundle_range=[%d, %d)",
                    i,
                    self._elastic_pool_id_by_rollout_instance[i],
                    b_start,
                    b_end,
                )
            if self._elastic_bundle_range_by_reward_model:
                for rm_name, ranges in self._elastic_bundle_range_by_reward_model.items():
                    for i, (b_start, b_end) in enumerate(ranges):
                        pivotrl_logger.info(
                            "Elastic_RM:   [RewardModel] model=%s instance_id=%d pool_id=%s bundle_range=[%d, %d)",
                            rm_name,
                            i,
                            self._elastic_pool_id_by_reward_model[rm_name][i],
                            b_start,
                            b_end,
                        )

        occupied_bundle_indices: set[tuple[str, int]] = set()
        awaken_instances: list[dict] = []
        min_awake_per_role = max(0, int(self.config.pivotrl.deployment.elastic_rm.min_awake_per_role))

        # Wake reward-model replicas first (up to min_awake_per_role per RM model; may be 0).
        for reward_model_name, manager in self.reward_model_manager_mapping.items():
            rm_instance_num = len(manager.reward_model_wg_list)
            rm_awake_num = max(min_awake_per_role, 0)
            rm_awake_ids, occupied_bundle_indices = self._select_non_conflicting_awake_ids(
                instance_to_bundle_indices=reward_model_to_instance_bundle_indices[reward_model_name],
                target_awake_num=rm_awake_num,
                occupied_bundle_indices=occupied_bundle_indices,
                min_awake_num=min_awake_per_role if rm_instance_num > 0 else 0,
            )
            pivotrl_logger.info(
                "Elastic_RM: waking up reward model replicas (name=%s, count=%d, instance_ids=%s).",
                reward_model_name,
                len(rm_awake_ids),
                rm_awake_ids,
            )
            wake_result = ray.get(
                manager.reward_model_coordinator.exec_command.remote(
                    Command(type=CommandType.WAKE_UP, instance_ids=rm_awake_ids),
                    blocking=True,
                )
            )
            if wake_result is not True:
                raise RuntimeError(
                    f"Failed to wake reward model {reward_model_name} instances {rm_awake_ids}: "
                    f"{wake_result!r}"
                )
            pivotrl_logger.info("Elastic_RM: reward model %s wake_up completed.", reward_model_name)
            awaken_instances.extend(
                [
                    {
                        "role_name": PivotRL_Role.RewardModel,
                        "model_name": reward_model_name,
                        "instance_id": instance_id,
                    }
                    for instance_id in rm_awake_ids
                ]
            )

        rollout_awake_num = max(min_awake_per_role, rollout_instance_num)
        awake_ids, occupied_bundle_indices = self._select_non_conflicting_awake_ids(
            instance_to_bundle_indices=rollout_instance_to_bundle_indices,
            target_awake_num=rollout_awake_num,
            occupied_bundle_indices=occupied_bundle_indices,
            min_awake_num=min_awake_per_role if rollout_instance_num > 0 else 0,
        )
        pivotrl_logger.info(
            "Elastic_RM: waking up rollout instances (count=%d, instance_ids=%s).",
            len(awake_ids),
            awake_ids,
        )
        wake_result = ray.get(
            self.rollout_coordinator.exec_command.remote(
                Command(type=CommandType.WAKE_UP, instance_ids=awake_ids),
                blocking=True,
            )
        )
        if wake_result is not True:
            raise RuntimeError(f"Failed to initialize rollout instances {awake_ids}: {wake_result!r}")
        pivotrl_logger.info("Elastic_RM: rollout wake_up completed.")
        awaken_instances.extend(
            [
                {
                    "role_name": PivotRL_Role.Rollout,
                    "model_name": rollout_model_name,
                    "instance_id": instance_id,
                }
                for instance_id in awake_ids
            ]
        )

        roles = [(PivotRL_Role.Rollout, rollout_model_name)]
        roles.extend([(PivotRL_Role.RewardModel, reward_model_name) for reward_model_name in reward_coordinators.keys()])
        coordinators = {
            PivotRL_Role.Rollout: {rollout_model_name: self.rollout_coordinator},
            PivotRL_Role.RewardModel: reward_coordinators,
        }
        elastic_rm_cfg = OmegaConf.to_container(self.config.pivotrl.deployment.elastic_rm, resolve=True)
        assert isinstance(elastic_rm_cfg, dict), "elastic_rm config should be resolved as dict"
        pivotrl_logger.info(
            "Elastic_RM: creating ElasticExecutor (roles=%s, awaken_instances=%d).",
            [(r.name, m) for r, m in roles],
            len(awaken_instances),
        )
        self.elastic_executor = ElasticExecutor.remote(
            config=self.config,
            roles=roles,
            coordinators=coordinators,
            agent_loop_manager=self.agent_loop_manager,
            elastic_rm_config=elastic_rm_cfg,
            # The trainer is NIXL-slept at startup (init_workers) when the
            # elastic trainer pool is active, lending train_pool GPUs to elastic
            # replicas. Seed trainer_busy=False in that case so the policy counts
            # the full shared+train capacity from the first tick.
            train_pool_available=bool(self._elastic_trainer_pool_trainer_sleeping),
        )
        ray.get(
            self.elastic_executor.initialize_runtime.remote(
                registrations=registrations,
                bundle_mappings=bundle_mappings,
                awaken_instances=awaken_instances,
            )
        )
        pivotrl_logger.info("Elastic_RM: ElasticExecutor.initialize_runtime done.")
        ray.get(self.elastic_executor.start_busy_loop.remote())
        pivotrl_logger.info("Elastic_RM: ElasticExecutor busy loop started; Elastic_RM runtime ready.")

        # Wire ElasticExecutor into RolloutCoordinator so sync_with_ps can use swap-based sync.
        ray.get(
            self.rollout_coordinator.set_elastic_executor.remote(
                self.elastic_executor,
                PivotRL_Role.Rollout,
                rollout_model_name,
            )
        )
        pivotrl_logger.info("Elastic_RM: ElasticExecutor wired into RolloutCoordinator for swap-based sync.")

    def _init_sleep_wake_orchestrator(self, rollout_model_name, reward_coordinators):
        """Initialize SleepWakeOrchestrator for colocated modes (2 and 3).

        Called from ``_init_elastic_rm_runtime`` after all rollout/rm instances have
        been SLEPT. Creates the orchestrator, injects it into the async pipeline
        components for phase gating, and starts it at the ROLLOUT phase.

        For mode 2 (colocated), the actor is already NIXL-slept at the end of
        ``init_workers`` (enable_trainer_pool=True), so the ROLLOUT phase correctly
        owns the shared pool. For mode 3, the actor stays awake on its separate
        train_pool and runs concurrently with the shared-pool phase cycle.
        """
        from pivotrl.utils.elastic_rm.sleep_wake_orchestrator import SleepWakeOrchestrator

        mode = self.deployment_mode
        pivotrl_logger.info(
            "Initializing SleepWakeOrchestrator (mode=%s, rollout_model=%s, rm_models=%s).",
            mode,
            rollout_model_name,
            list(reward_coordinators.keys()),
        )
        self.sleep_wake_orchestrator = SleepWakeOrchestrator.remote(
            mode=mode,
            rollout_coordinator=self.rollout_coordinator,
            reward_model_coordinators=reward_coordinators,
            rollout_model_name=rollout_model_name,
        )
        # Inject into async pipeline components for phase gating.
        injection_futures = []
        if self.agent_loop_manager is not None:
            injection_futures.append(
                self.agent_loop_manager.set_sleep_wake_orchestrator.remote(self.sleep_wake_orchestrator)
            )
        if self.reward_manager is not None:
            injection_futures.append(
                self.reward_manager.set_sleep_wake_orchestrator.remote(self.sleep_wake_orchestrator)
            )
        if self.rollout_coordinator is not None and hasattr(self.rollout_coordinator, "set_sleep_wake_orchestrator"):
            injection_futures.append(
                self.rollout_coordinator.set_sleep_wake_orchestrator.remote(self.sleep_wake_orchestrator)
            )
        if injection_futures:
            ray.get(injection_futures)
        # Start at ROLLOUT: wake rollout (rm already asleep; actor already asleep for mode 2).
        ray.get(self.sleep_wake_orchestrator.start.remote())
        pivotrl_logger.info("SleepWakeOrchestrator started (mode=%s, phase=rollout).", mode)

    def _elastic_trainer_pool_instance_entries(self) -> list[dict]:
        """Return rollout/RM instances mapped to ``train_pool`` for elastic training-window control.

        The result depends only on ``_elastic_pool_id_by_*`` (filled in ``init_workers``)
        and the static model path, so it is immutable after ``init_workers``. It is
        computed once on the first call (the call at the end of ``init_workers``) and
        cached in ``self._elastic_trainer_pool_entries`` for reuse by every
        enter/leave training-window call afterwards.
        """
        if self._elastic_trainer_pool_entries is not None:
            return self._elastic_trainer_pool_entries

        entries: list[dict] = []
        if (self.elastic_trainer_pool_mode or self.trainer_pool_only_mode) and self._elastic_pool_id_by_rollout_instance is not None:
            rollout_model_name = self.config.gen_actor_rollout_ref.model.path.split("/")[-1]
            for instance_id, pool_id in enumerate(self._elastic_pool_id_by_rollout_instance):
                if pool_id != "train_pool":
                    continue
                entries.append(
                    {
                        "role_name": PivotRL_Role.Rollout,
                        "model_name": rollout_model_name,
                        "instance_id": int(instance_id),
                    }
                )

            for reward_model_name, pool_ids in (self._elastic_pool_id_by_reward_model or {}).items():
                for instance_id, pool_id in enumerate(pool_ids):
                    if pool_id != "train_pool":
                        continue
                    entries.append(
                        {
                            "role_name": PivotRL_Role.RewardModel,
                            "model_name": reward_model_name,
                            "instance_id": int(instance_id),
                        }
                    )

        self._elastic_trainer_pool_entries = entries
        return entries

    def _sleep_trainer_for_elastic_trainer_pool(self):
        """NIXL-sleep the actor on ``train_pool`` so elastic rollout/RM replicas can use its GPUs."""
        if not (self.elastic_trainer_pool_mode or self.trainer_pool_only_mode) or self._elastic_trainer_pool_trainer_sleeping:
            return
        if self.config.pivotrl.ps_mode not in ("nixl_cpu", "nixl_gpu"):
            raise RuntimeError("Elastic trainer pool requires NIXL PS mode for trainer sleep.")
        if getattr(self, "actor_wg", None) is None:
            raise RuntimeError("Actor worker group must be initialized before sleeping trainer.")
        pivotrl_logger.info("Elastic trainer pool: sleeping trainer actor.")
        self._trainer_before_sleep_weight_fingerprints = None
        if weight_fingerprint_flow_enabled(self.config.pivotrl, flow="trainer_sleep_wake"):
            model_version = int(ray.get(self.ps_manager_handle.get_ps_model_version.remote("trainer_before_sleep")))
            fingerprint_options = resolve_weight_fingerprint_options(
                self.config.pivotrl,
                flow="trainer_sleep_wake",
                model_version=model_version,
            )
        else:
            model_version = -1
            fingerprint_options = None
        if fingerprint_options is not None:
            before_sleep = ray.get(
                self.actor_wg.execute_all_async(
                    "capture_weight_fingerprint",
                    "trainer_before_sleep",
                    model_version,
                    "trainer_sleep_wake",
                )
            )
            if any(record is not None for record in before_sleep):
                self._trainer_before_sleep_weight_fingerprints = {
                    "model_version": model_version,
                    "records": before_sleep,
                }
            else:
                self._trainer_before_sleep_weight_fingerprints = None
        sleep_started_s = time.perf_counter()
        ray.get(self.actor_wg.execute_all_async("nixl_sleep", "full"))
        sleep_elapsed_s = time.perf_counter() - sleep_started_s
        pivotrl_logger.warning(
            "[ELASTIC_OVERHEAD] operation=sleep role=Trainer world_size=%d "
            "engine_sleep_s=%.6f total_s=%.6f",
            self.actor_wg.world_size,
            sleep_elapsed_s,
            sleep_elapsed_s,
        )
        self._elastic_trainer_pool_trainer_sleeping = True

    def _wake_trainer_for_elastic_trainer_pool(self):
        """NIXL-wake the actor and pull weights so training can run on ``train_pool`` GPUs."""
        if not (self.elastic_trainer_pool_mode or self.trainer_pool_only_mode) or not self._elastic_trainer_pool_trainer_sleeping:
            return
        if self.config.pivotrl.ps_mode not in ("nixl_cpu", "nixl_gpu"):
            raise RuntimeError("Elastic trainer pool requires NIXL PS mode for trainer wake-up.")
        with log_dual_events(
            "Wake trainer actor for elastic trainer pool",
            pivotrl_logger,
            event_type=EventType.SWITCH,
        ):
            total_start = time.perf_counter()
            pivotrl_logger.info("Elastic trainer pool: waking trainer actor.")
            stage_start = time.perf_counter()
            ray.get(self.actor_wg.execute_all_async("nixl_wake_up"))
            wake_rpc_s = time.perf_counter() - stage_start
            baseline = self._trainer_before_sleep_weight_fingerprints
            mapping_records = None
            if baseline is not None:
                mapping_records = ray.get(
                    self.actor_wg.execute_all_async(
                        "capture_weight_mapping_signature",
                        "trainer_after_wake_register",
                        baseline["model_version"],
                        "trainer_sleep_wake",
                    )
                )
            updated_client_names = [train_client_name(i) for i in range(self.actor_wg.world_size)]
            futures = []
            stage_start = time.perf_counter()
            futures.extend(self.actor_wg.execute_all_async("nixl_send_local_info_to", NIXL_META_SERVER_NAME))
            futures.append(self.ps_manager_handle.nixl_wait_for_update_infos.remote(self.actor_wg.world_size))
            ray.get(futures)
            gather_infos_s = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            self._broadcast_updated_client_infos_from_ps_manager(updated_client_names)
            broadcast_infos_s = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
            pull_results = ray.get(self.actor_wg.execute_all_async("pull_model"))
            pull_model_s = time.perf_counter() - stage_start
            self._clear_fsdp2_grads_after_trainer_wake()
            self._verify_trainer_sleep_wake_weights(mapping_records, pull_results)
            total_s = time.perf_counter() - total_start
            pivotrl_logger.warning(
                "[TRAINER_WAKE_TIMING] scope=trainer_controller world_size=%d "
                "wake_rpc_s=%.6f gather_infos_s=%.6f broadcast_infos_s=%.6f "
                "pull_model_s=%.6f total_s=%.6f",
                self.actor_wg.world_size,
                wake_rpc_s,
                gather_infos_s,
                broadcast_infos_s,
                pull_model_s,
                total_s,
            )
            pivotrl_logger.warning(
                "[ELASTIC_OVERHEAD] operation=wakeup role=Trainer world_size=%d "
                "engine_wakeup_s=%.6f gather_infos_s=%.6f broadcast_infos_s=%.6f "
                "pull_model_s=%.6f total_s=%.6f",
                self.actor_wg.world_size,
                wake_rpc_s,
                gather_infos_s,
                broadcast_infos_s,
                pull_model_s,
                total_s,
            )
        self._elastic_trainer_pool_trainer_sleeping = False

    def _verify_trainer_sleep_wake_weights(self, mapping_records: list | None, pull_results: list) -> None:
        baseline = self._trainer_before_sleep_weight_fingerprints
        self._trainer_before_sleep_weight_fingerprints = None
        if baseline is None:
            return

        expected_records = {
            int(record["rank"]): record for record in baseline["records"] if record is not None
        }
        mapping_by_rank = {
            int(record["rank"]): record for record in (mapping_records or []) if record is not None
        }
        actual_records = {
            int(result["final_fingerprint"]["rank"]): result["final_fingerprint"]
            for result in pull_results
            if result is not None and result.get("final_fingerprint") is not None
        }
        actual_versions = sorted(
            {
                int(result["model_version"])
                for result in pull_results
                if result is not None and result.get("model_version") is not None
            }
        )
        expected_version = int(baseline["model_version"])
        rank_results = {}
        mismatch = False
        for rank in sorted(expected_records.keys() | actual_records.keys() | mapping_by_rank.keys()):
            expected = expected_records.get(rank)
            actual = actual_records.get(rank)
            mapping = mapping_by_rank.get(rank)
            if expected is None or actual is None:
                mismatch = True
                rank_results[rank] = {"match": False, "reason": "missing_fingerprint"}
                continue
            comparison = compare_weight_fingerprints(expected, actual)
            mapping_match = mapping is not None and expected.get("mapping_digest") == mapping.get("mapping_digest")
            address_match = mapping is not None and expected.get("address_digest") == mapping.get("address_digest")
            content_match = bool(comparison["match"] and comparison["mapping_match"])
            mismatch = mismatch or not content_match or not mapping_match
            rank_results[rank] = {
                "match": content_match and mapping_match,
                "content_match": content_match,
                "mapping_match_after_register": mapping_match,
                "address_match_after_register": address_match,
                "differing_tensor_count": comparison["differing_tensor_count"],
                "differing_tensors": comparison["differing_tensors"][:20],
            }

        version_match = actual_versions == [expected_version]
        mismatch = mismatch or not version_match
        record = {
            "expected_model_version": expected_version,
            "actual_model_versions": actual_versions,
            "version_match": version_match,
            "status": "mismatch" if mismatch else "match",
            "rank_results": rank_results,
        }
        log_method = pivotrl_logger.error if mismatch else pivotrl_logger.warning
        log_method("[TRAINER_SLEEP_WAKE_WEIGHT_CHECK] %s", json.dumps(record, sort_keys=True))

        options = resolve_weight_fingerprint_options(
            self.config.pivotrl,
            flow="trainer_sleep_wake",
            model_version=expected_version,
        )
        if mismatch and options is not None and options["fail_on_mismatch"]:
            raise RuntimeError(f"Trainer sleep/wake weight verification failed: {record}")

    def _clear_fsdp2_grads_after_trainer_wake(self) -> list[int]:
        """Reset every FSDP2 gradient representation before the next update."""
        if self.config.train_actor_rollout_ref.actor.strategy != "fsdp2":
            return []
        dirty_counts = ray.get(self.actor_wg.execute_all_async("clear_fsdp2_grads")) or []
        normalized_counts = [int(count or 0) for count in dirty_counts]
        pivotrl_logger.info(
            "[TRAINER_WAKE_GRAD_CHECK] world_size=%d dirty_counts=%s total_cleared=%d",
            self.actor_wg.world_size,
            normalized_counts,
            sum(normalized_counts),
        )
        return normalized_counts

    def _trainer_pool_only_replica_entries(self) -> list[dict]:
        """Train_pool rollout/rm replica entries for mode 4 (subset of instance entries)."""
        return self._elastic_trainer_pool_instance_entries()

    def _sleep_trainer_pool_only_replicas(self):
        """SLEEP the fixed extra rollout/rm replicas on train_pool (mode 4)."""
        self._exec_trainer_pool_only_replicas_command(CommandType.SLEEP, "slept")

    def _wake_trainer_pool_only_replicas(self):
        """WAKE_UP the fixed extra rollout/rm replicas on train_pool (mode 4)."""
        self._exec_trainer_pool_only_replicas_command(CommandType.WAKE_UP, "woke")

    def _exec_trainer_pool_only_replicas_command(self, command_type: CommandType, verb: str):
        """Issue a SLEEP/WAKE_UP to all train_pool rollout/rm replicas concurrently.

        All coordinator ``exec_command`` calls are dispatched as Ray futures first
        and awaited together, so different roles / reward models transition in
        parallel rather than serially. Within a single coordinator the command
        loop already parallelizes across instances (asyncio.gather of per-instance
        abort/sleep/wake), so per-coordinator is the right concurrency granularity.
        """
        if not self.trainer_pool_only_mode:
            return
        entries = self._trainer_pool_only_replica_entries()
        if not entries:
            return
        rollout_ids: list[int] = []
        rm_by_model: dict[str, list[int]] = {}
        for entry in entries:
            if entry["role_name"] == PivotRL_Role.Rollout:
                rollout_ids.append(int(entry["instance_id"]))
            elif entry["role_name"] == PivotRL_Role.RewardModel:
                rm_by_model.setdefault(entry["model_name"], []).append(int(entry["instance_id"]))
        futures = []
        if rollout_ids and self.rollout_coordinator is not None:
            futures.append(
                self.rollout_coordinator.exec_command.remote(
                    Command(type=command_type, instance_ids=rollout_ids),
                    blocking=True,
                )
            )
        for rm_name, ids in rm_by_model.items():
            manager = self.reward_model_manager_mapping.get(rm_name)
            if manager is not None and ids:
                futures.append(
                    manager.reward_model_coordinator.exec_command.remote(
                        Command(type=command_type, instance_ids=ids),
                        blocking=True,
                    )
                )
        if futures:
            results = ray.get(futures)
            if any(result is not True for result in results):
                raise RuntimeError(
                    f"Trainer-pool-only {command_type.name} failed: results={results!r}"
                )
        pivotrl_logger.info(
            "Trainer-pool-only: %s train_pool replicas (rollout=%s, rm=%s).",
            verb,
            rollout_ids,
            rm_by_model,
        )

    def _enter_elastic_trainer_pool_training_window(self):
        """Reserve train-pool elastic instances (TRAINING) and wake the actor before a training step."""
        if not (self.elastic_trainer_pool_mode or self.trainer_pool_only_mode) or self._elastic_trainer_pool_training_active:
            return
        entries = self._elastic_trainer_pool_instance_entries()
        if self.trainer_pool_only_mode:
            # No ElasticExecutor in mode 4: directly SLEEP the train_pool replicas
            # so the actor can own train_pool GPUs for the training step.
            self._sleep_trainer_pool_only_replicas()
        elif self.elastic_executor is not None and entries:
            ray.get(self.elastic_executor.enter_training_pool.remote(entries))
        self._wake_trainer_for_elastic_trainer_pool()
        self._elastic_trainer_pool_training_active = True
        pivotrl_logger.info("Elastic trainer pool: entered training window with entries=%s.", entries)

    def _enter_elastic_trainer_pool_validation_window(self):
        """
        Reserve train-pool elastic instances without waking the trainer.

        Dedicated validation workers time-share the trainer GPUs. Reserving the
        elastic entries keeps policy-driven rollout and reward-model wakeups off
        those GPUs until validation finishes.
        """
        if not self.elastic_trainer_pool_mode or self._elastic_trainer_pool_training_active:
            return
        entries = self._elastic_trainer_pool_instance_entries()
        if self.elastic_executor is not None and entries:
            ray.get(self.elastic_executor.enter_training_pool.remote(entries))
        self._elastic_trainer_pool_training_active = True
        pivotrl_logger.info("Elastic trainer pool: entered validation window with entries=%s.", entries)

    def _sleep_validation_workers_after_initialization(self) -> None:
        """Sleep dedicated validation workers before initializing train-pool rollout replicas."""
        validation_instance_ids = list(
            range(
                self.n_rollout_instances,
                self.n_rollout_instances + self.n_validate_instances,
            )
        )
        ray.get(
            [
                self.rollout_coordinator.sleep.remote("validate"),
                self.rollout_router.pause_instances.remote(validation_instance_ids),
            ]
        )
        self.is_rollout_mode_in_actor = False

    def _leave_elastic_trainer_pool_training_window(self):
        """Sleep the actor and release train-pool elastic instances after a training step."""
        if not (self.elastic_trainer_pool_mode or self.trainer_pool_only_mode) or not self._elastic_trainer_pool_training_active:
            return
        entries = self._elastic_trainer_pool_instance_entries()
        self._sleep_trainer_for_elastic_trainer_pool()
        if self.trainer_pool_only_mode:
            # Trainer is now idle: wake the fixed extra replicas on train_pool.
            self._wake_trainer_pool_only_replicas()
        elif self.elastic_executor is not None and entries:
            ray.get(self.elastic_executor.leave_training_pool.remote(entries))
        self._elastic_trainer_pool_training_active = False
        pivotrl_logger.info("Elastic trainer pool: left training window with entries=%s.", entries)

    def _set_elastic_training_step(self) -> None:
        """Tell ElasticExecutor which trainer step owns subsequently accepted actions."""
        if self.elastic_executor is None:
            return
        try:
            ray.get(
                self.elastic_executor.set_current_training_step.remote(
                    self.global_steps
                )
            )
        except Exception:
            pivotrl_logger.exception(
                "Failed to set elastic_rm training step=%s; actions may be tagged step=-1.",
                self.global_steps,
            )

    def _collect_elastic_awake_metrics(self) -> dict[str, int]:
        """Pull live awake counts and completed scaling actions for this step.

        Returns an empty dict when elastic_rm is disabled or the snapshot cannot be
        fetched, so the failure never blocks the per-step metric logging.
        """
        if self.elastic_executor is None:
            return {}
        try:
            awake_counts, scaling_counts = ray.get(
                [
                    self.elastic_executor.get_awake_instance_counts.remote(),
                    self.elastic_executor.get_step_scaling_action_counts.remote(
                        self.global_steps
                    ),
                ]
            )
        except Exception:
            pivotrl_logger.exception(
                "Failed to fetch elastic_rm step metrics; skipping this step."
            )
            return {}
        metrics = {
            f"elastic_rm/awake_instances/{role_key}": int(count)
            for role_key, count in awake_counts.items()
        }
        metrics.update(
            {
                "elastic_rm/scaling_actions/total": int(
                    scaling_counts["actual_actions"]
                ),
                "elastic_rm/scaling_actions/scale_up": int(
                    scaling_counts["scale_up_actions"]
                ),
                "elastic_rm/scaling_actions/scale_down": int(
                    scaling_counts["scale_down_actions"]
                ),
                "elastic_rm/scaling_instance_transitions/total": int(
                    scaling_counts["instance_transitions"]
                ),
                "elastic_rm/scaling_instance_transitions/sleep": int(
                    scaling_counts["sleep_instances"]
                ),
                "elastic_rm/scaling_instance_transitions/wakeup": int(
                    scaling_counts["wakeup_instances"]
                ),
            }
        )
        return metrics

    def init_reward_manager(self, validation: bool = False):
        """Initialize the reward manager for computing rewards during training."""
        if validation:
            ip_to_node_id = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes()}
            self.val_reward_manager = (
                ray.remote(RewardManager)
                .options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=ip_to_node_id[self.config.pivotrl.reward_service_ip],
                        soft=False,
                    )
                )
                .remote(
                    config=self.config,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    reward_model_configs=self.config.reward_models_config.reward_models,
                    reward_model_manager_mapping=self.reward_model_manager_mapping,
                    ps_manager_handle=self.ps_manager_handle,
                    validation=validation,
                )
            )
            return
        assert self.data_processor is not None, (
            "Data processor must be initialized before starting reward computation."
        )
        assert self.rollout_coordinator is not None, (
            "Rollout server must be initialized before starting reward computation."
        )

        ip_to_node_id = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes()}
        self.reward_manager = (
            ray.remote(RewardManager)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ip_to_node_id[self.config.pivotrl.reward_service_ip],
                    soft=False,
                )
            )
            .remote(
                # self.config,
                # self.tokenizer,
                # self.processor,
                # self.ps_manager_handle,
                config=self.config,
                tokenizer=self.tokenizer,
                processor=self.processor,
                ps_manager_handle=self.ps_manager_handle,
                reward_model_configs=self.config.reward_models_config.reward_models,
                reward_model_manager_mapping=self.reward_model_manager_mapping,
                validation=False,
            )
        )

    def start_reward_manager(self):
        """Start the reward manager to handle reward computation requests in the background."""
        assert self.reward_manager is not None, "Reward manager must be initialized before starting it."

        ray.get(self.reward_manager.start_busy_loop.remote())

    def stop_reward_manager(self):
        """Stop the reward manager."""
        if self.reward_manager is not None:
            pivotrl_logger.debug("Stopping reward manager...")
            ray.get(self.reward_manager.stop_busy_loop.remote())
            self.reward_manager = None
            pivotrl_logger.debug("Reward manager stopped successfully.")
        else:
            pivotrl_logger.warning("Reward manager is not initialized, skipping stop operation.")

    def shutdown_reward_model_managers(self) -> None:
        """Gracefully close all GenRM engines and release node-shared caches."""
        failures = []
        for reward_model_name, manager in self.reward_model_manager_mapping.items():
            try:
                manager.shutdown()
            except Exception as exc:
                failures.append((reward_model_name, exc))
                pivotrl_logger.exception(
                    "Failed to shut down reward model manager %s.",
                    reward_model_name,
                )
        self.reward_model_manager_mapping.clear()
        if failures:
            pivotrl_logger.warning(
                "Reward model manager shutdown completed with %d failures.",
                len(failures),
            )

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        pivotrl_logger.info(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self,
        batch: DataProto,
        reward_extra_infos_dict: dict,
        timing_raw: dict,
        rollout_data_dir: str,
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        """Validate the model using the validation dataset.

        Note that we use the training side to do val for overlapping with generation.
        """
        elastic_validation_colocation = (
            self.elastic_trainer_pool_mode and self.config.pivotrl.colocate_validate_and_train
        )
        if elastic_validation_colocation:
            self._enter_elastic_trainer_pool_validation_window()

        try:
            with log_dual_events("Switch to rollout mode", pivotrl_logger, event_type=EventType.SWITCH):
                self.switch_to_rollout_mode()
            return self._run_validation()
        finally:
            if self.is_rollout_mode_in_actor:
                with log_dual_events("Switch validation workers out", pivotrl_logger, event_type=EventType.SWITCH):
                    self.switch_to_trainer_mode(restore_trainer=not elastic_validation_colocation)

    def _run_validation(self):
        """Run validation after the required worker resources are active."""
        pivotrl_logger.debug("Starting validation process")
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_parent_ids = []
        sample_response_lengths = []
        request_ids = []

        test_batch_list = []
        batch_count = 0
        while True:
            try:
                test_data = ray.get(self.data_processor.get_single_controller_batch.remote(DatasetType.val))
                batch_count += 1
            except RayTaskError as e:
                if isinstance(e.cause, StopIteration):
                    pivotrl_logger.debug(
                        "Reached end of validation dataset after %d batches",
                        batch_count,
                    )
                    break
                else:
                    pivotrl_logger.error(f"Unknown exception happened during obtaining validation data: {type(e.cause)}")
                    raise
            test_batch = DataProto.from_single_dict(test_data)

            if "parent_id" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["parent_id"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                    dtype=object,
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.train_actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # Ensure each sample has a stable id for reward mapping.
            # Some validation datasets/controllers do not provide `uid`.
            test_batch.non_tensor_batch["uid"] = np.arange(
                len(request_ids), len(request_ids) + len(test_batch), 
                dtype=np.int64
            )

            # we only do validation on rule-based rm
            if test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            test_batch_list.append(test_batch)

        val_data_size = sum(len(batch.batch) for batch in test_batch_list)
        futures = []
        futures.append(self.ps_manager_handle.set_val_staleness_inventory_capacity.remote(val_data_size))
        futures.append(self.agent_loop_manager.set_val_buffer_size.remote(val_data_size))
        ray.get(futures)

        val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n
        for test_batch in test_batch_list:
            batch_size = len(test_batch.batch)

            sample_ids = ray.get(self.data_processor.get_val_sample_ids.remote(batch_size))
            test_batch.non_tensor_batch["parent_id" if val_rollout_n > 1 else "uid"] = np.array(sample_ids)
            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=val_rollout_n, interleave=True)

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO(verl): Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_parent_ids.extend(test_batch.non_tensor_batch["parent_id" if val_rollout_n > 1 else "uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            non_tensor_batch_keys_to_pop.append("parent_id" if val_rollout_n > 1 else "uid")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            if val_rollout_n > 1:
                uid_list = []
                for i in range(batch_size):
                    for j in range(val_rollout_n):
                        child_id = sample_ids[i] * val_rollout_n + j
                        uid_list.append(child_id)
                test_gen_batch.non_tensor_batch["uid"] = np.array(uid_list)

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.train_actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            pivotrl_logger.debug(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            val_buffer_id = ray.get(self.agent_loop_manager.generate_validate_sequences.remote(test_gen_batch))
            with log_dual_events(f"Wait for validation batch {val_buffer_id}", pivotrl_logger, event_type=EventType.WAIT):
                test_output_gen_batch = ray.get(
                    self.agent_loop_manager.wait_for_validation_batch.remote(val_buffer_id)
                )

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_manager is None:
                raise ValueError("Reward manager must be provided for validation.")

            request_id_to_reward = ray.get(self.val_reward_manager.compute_score_for_validation.remote(test_batch))
            _request_ids = test_batch.non_tensor_batch["uid"].tolist()
            scores = []
            reward_extra_infos_dict_list = []
            acc_list = []
            output_non_tensor_batch = test_output_gen_batch.non_tensor_batch
            output_uids = output_non_tensor_batch.get("uid", None)
            output_response_lengths = output_non_tensor_batch.get("response_unpadded_len", None)
            if output_response_lengths is None:
                output_response_lengths = np.full(
                    len(test_output_gen_batch),
                    test_output_gen_batch.batch["responses"].shape[-1],
                    dtype=np.int64,
                )
            else:
                output_response_lengths = np.asarray(output_response_lengths, dtype=np.int64)
            if output_uids is not None:
                response_length_by_uid = dict(
                    zip(np.asarray(output_uids).tolist(), output_response_lengths.tolist(), strict=True)
                )
                response_lengths = np.asarray(
                    [response_length_by_uid[request_id] for request_id in _request_ids],
                    dtype=np.int64,
                )
            else:
                response_lengths = output_response_lengths
            if len(response_lengths) != len(_request_ids):
                raise ValueError(
                    "Validation response length count does not match request count: "
                    f"{len(response_lengths)} != {len(_request_ids)}."
                )
            for request_id in _request_ids:
                reward_score = request_id_to_reward[request_id]["reward_score"]
                extra_info = request_id_to_reward[request_id].get("reward_extra_info", {})
                acc = extra_info.get("acc", 0.0)
                scores.append(reward_score)
                reward_extra_infos_dict_list.append(extra_info)
                acc_list.append(acc)
            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_infos_dict["reward_extra_info"].extend(reward_extra_infos_dict_list)
            reward_extra_infos_dict["acc"].extend(acc_list)
            reward_extra_infos_dict["response_length"].extend(response_lengths.tolist())
            sample_response_lengths.extend(response_lengths.tolist())

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(_request_ids))
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_parent_ids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        if sample_response_lengths:
            response_lengths = np.asarray(sample_response_lengths, dtype=np.int64)
            response_length_limit = int(self.config.train_actor_rollout_ref.rollout.response_length)
            metric_dict["val-aux/response_length/mean"] = float(response_lengths.mean())
            metric_dict["val-aux/response_length/max"] = int(response_lengths.max())
            metric_dict["val-aux/response_length/min"] = int(response_lengths.min())
            metric_dict["val-aux/response_length/at_limit_ratio"] = float(
                np.mean(response_lengths >= response_length_limit)
            )

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)

        Note that we use multi-threading to speed up the initialization of worker groups.
        For rollout instances, we create multiple worker groups based on
        the number of instances specified in the configuration,
        instead of creating a unified worker group for all instances.
        """

        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        elastic_base_pools: dict[str, RayResourcePool] = {}
        elastic_subpool_group_idx_by_group: dict[tuple[str, str], int] = {}
        trainer_pool_next_bundle_index: dict[str, int] = {}

        if self.elastic_rm_mode:
            elastic_shared_pool = self.resource_pool_manager.get_resource_pool(PivotRL_Role.Rollout, 0)
            # Materialize placement groups once and reuse them across all elastic sub resource pools.
            elastic_shared_pool.get_placement_groups(strategy="STRICT_PACK", device_name=self.device_name)
            elastic_base_pools["shared_rollout_pool"] = elastic_shared_pool
            pivotrl_logger.info(
                "Elastic RM shared RayResourcePool: name_prefix=%s world_size=%d store=%s max_colocate_count=%s",
                elastic_shared_pool.name_prefix,
                elastic_shared_pool.world_size,
                list(elastic_shared_pool.store),
                elastic_shared_pool.max_colocate_count,
            )
            if self.elastic_trainer_pool_mode:
                elastic_train_pool = self.resource_pool_manager.get_resource_pool(PivotRL_Role.Actor)
                elastic_train_pool.get_placement_groups(strategy="STRICT_PACK", device_name=self.device_name)
                elastic_base_pools["train_pool"] = elastic_train_pool
                pivotrl_logger.info(
                    "Elastic RM trainer RayResourcePool: name_prefix=%s world_size=%d store=%s max_colocate_count=%s",
                    elastic_train_pool.name_prefix,
                    elastic_train_pool.world_size,
                    list(elastic_train_pool.store),
                    elastic_train_pool.max_colocate_count,
                )
            self._elastic_bundle_range_by_rollout_instance = []
            self._elastic_bundle_range_by_reward_model = {}
            self._elastic_pool_id_by_rollout_instance = []
            self._elastic_pool_id_by_reward_model = {}
        elif self.trainer_pool_only_mode:
            # Mode 4: disaggregated main rollout/rm on independent pools + a fixed
            # number of extra rollout/rm replicas on train_pool (time-multiplexed
            # with the actor via NIXL sleep/wake). Only train_pool needs the elastic
            # SubRayResourcePool slicing; main instances use the non-elastic path.
            elastic_train_pool = self.resource_pool_manager.get_resource_pool(PivotRL_Role.Actor)
            elastic_train_pool.get_placement_groups(strategy="STRICT_PACK", device_name=self.device_name)
            elastic_base_pools["train_pool"] = elastic_train_pool
            pivotrl_logger.info(
                "Trainer-pool-only: train RayResourcePool ready for idle replicas: "
                "name_prefix=%s world_size=%d store=%s max_colocate_count=%s",
                elastic_train_pool.name_prefix,
                elastic_train_pool.world_size,
                list(elastic_train_pool.store),
                elastic_train_pool.max_colocate_count,
            )
            self._elastic_bundle_range_by_rollout_instance = []
            self._elastic_bundle_range_by_reward_model = {}
            self._elastic_pool_id_by_rollout_instance = []
            self._elastic_pool_id_by_reward_model = {}
            # Mark non-train_pool (main independent) instances with None bundle range
            # so _elastic_trainer_pool_instance_entries can filter them out.

        def _register_resource_pool(resource_pool):
            self.resource_pool_to_cls.setdefault(resource_pool, {})

        def _build_elastic_sub_resource_pool(
            subgroup_world_size: int,
            tag: str,
            group_key: str,
            pool_id: str,
        ) -> SubRayResourcePool:
            elastic_pool = elastic_base_pools.get(pool_id)
            assert elastic_pool is not None, f"elastic pool {pool_id!r} must be initialized in elastic_rm_mode."
            if subgroup_world_size <= 0:
                raise ValueError(f"subgroup_world_size must be > 0, but got {subgroup_world_size} ({tag})")
            if subgroup_world_size > elastic_pool.world_size:
                raise ValueError(
                    f"subgroup_world_size={subgroup_world_size} exceeds shared pool world_size="
                    f"{elastic_pool.world_size} ({tag}, pool_id={pool_id})"
                )

            # SubRayResourcePool requires a contiguous bundle range [start, start + subgroup_world_size).
            # Align starts by subgroup size so a larger-parallelism instance maps to a group of
            # smaller power-of-two instances, e.g. [0, 4) corresponds to [0, 2) and [2, 4).
            #
            # Bundle-index allocation:
            # - colocated (mode 2): rollout and rm time-multiplex the SAME bundles on train_pool,
            #   so keep separate group_key counters that both cycle from bundle 0.
            # - trainer-pool-only (mode 4): all idle replicas may be awake concurrently, so allocate
            #   disjoint ranges using a bundle cursor. An instance counter is incorrect when TP sizes
            #   differ because changing the TP size rescales the counter and can wrap onto live ranges.
            # - other non-colocated train-pool modes share an instance counter across roles.
            if pool_id == "train_pool" and self.trainer_pool_only_mode:
                next_bundle_index = trainer_pool_next_bundle_index.get(pool_id, 0)
                start_bundle_index = (
                    (next_bundle_index + subgroup_world_size - 1) // subgroup_world_size
                ) * subgroup_world_size
                end_bundle = start_bundle_index + subgroup_world_size
                if end_bundle > elastic_pool.world_size:
                    raise ValueError(
                        f"Trainer-pool-only idle replicas exceed train_pool capacity while placing {tag}: "
                        f"aligned bundle_range=[{start_bundle_index}, {end_bundle}), "
                        f"train_pool world_size={elastic_pool.world_size}. "
                        "Idle rollout and reward-model replicas must use disjoint GPU bundles."
                    )
                trainer_pool_next_bundle_index[pool_id] = end_bundle
                group_idx = start_bundle_index // subgroup_world_size
            elif pool_id == "train_pool" and not self.colocated_mode:
                group_idx_key = (pool_id, "__sequential__")
                group_idx = elastic_subpool_group_idx_by_group.get(group_idx_key, 0)
                slots_per_cycle = elastic_pool.world_size // subgroup_world_size
                start_bundle_index = (group_idx % slots_per_cycle) * subgroup_world_size
                elastic_subpool_group_idx_by_group[group_idx_key] = group_idx + 1
            else:
                group_idx_key = (pool_id, group_key)
                group_idx = elastic_subpool_group_idx_by_group.get(group_idx_key, 0)
                slots_per_cycle = elastic_pool.world_size // subgroup_world_size
                start_bundle_index = (group_idx % slots_per_cycle) * subgroup_world_size
                elastic_subpool_group_idx_by_group[group_idx_key] = group_idx + 1
            if elastic_pool.world_size % subgroup_world_size != 0:
                raise ValueError(
                    f"subgroup_world_size={subgroup_world_size} must divide shared pool world_size="
                    f"{elastic_pool.world_size} for elastic_rm power-of-two placement ({tag}, pool_id={pool_id})."
                )
            if subgroup_world_size & (subgroup_world_size - 1) != 0:
                raise ValueError(
                    f"subgroup_world_size={subgroup_world_size} must be a power of two for elastic_rm "
                    f"different-parallelism placement ({tag}, pool_id={pool_id})."
                )

            sub_rp = SubRayResourcePool(
                process_on_nodes=elastic_pool.store,
                use_gpu=elastic_pool.use_gpu,
                name_prefix=f"{elastic_pool.name_prefix}_{tag}",
                max_colocate_count=elastic_pool.max_colocate_count,
                detached=elastic_pool.detached,
                accelerator_type=elastic_pool.accelerator_type,
                resource_num_per_bundle=elastic_pool.resource_num_per_bundle,
                placement_groups=elastic_pool.pgs,
                start_bundle_index=start_bundle_index,
                subgroup_world_size=subgroup_world_size,
            )
            end_bundle = start_bundle_index + subgroup_world_size
            pg_ids = [getattr(pg, "id", None) for pg in (elastic_pool.pgs or [])]
            pivotrl_logger.info(
                "Elastic SubRayResourcePool[%s]: pool_id=%s group_key=%s group_idx=%d type=%s name_prefix=%s subgroup_world_size=%d "
                "start_bundle_index=%d bundle_range=[%d, %d) shared_world_size=%d shared_store=%s pg_count=%s pg_ids=%s",
                tag,
                pool_id,
                group_key,
                group_idx,
                type(sub_rp).__name__,
                sub_rp.name_prefix,
                subgroup_world_size,
                start_bundle_index,
                start_bundle_index,
                end_bundle,
                elastic_pool.world_size,
                list(elastic_pool.store),
                len(elastic_pool.pgs) if elastic_pool.pgs is not None else 0,
                pg_ids,
            )
            return sub_rp

        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                )
        wg_kwargs["device_name"] = self.device_name

        # create rollout, actor and ps
        # PS need to be created before rollout and actor to pass the ps_manager_handle
        assert PivotRL_Role.Rollout in self.role_worker_mapping and PivotRL_Role.Actor in self.role_worker_mapping, (
            "Rollout and Actor must be in role_worker_mapping."
        )

        # create nixl interface
        nixl_interface = NIXLInterface(port_scanner=GLOBAL_PORT_SCANNER)

        # create rollout instances
        for i in range(self.n_rollout_instances):
            gen_interface = GenInterface(
                rollout_instance_id=i,
                status_queue=self.status_queues[i],
                ps_manager_handle=self.ps_manager_handle,
            )
            rollout_config = self.config.gen_actor_rollout_ref
            if self.config.pivotrl.deployment.heterogeneous_rollout.enable:
                rollout_config.rollout.tensor_model_parallel_size = (
                    self.config.pivotrl.deployment.heterogeneous_rollout.tensor_model_parallel_size_per_instance[i]
                )
                rollout_config.rollout.pipeline_model_parallel_size = (
                    self.config.pivotrl.deployment.heterogeneous_rollout.pipeline_model_parallel_size_per_instance[i]
                )

            rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PivotRL_Role.Rollout],
                config=rollout_config,
                role="rollout",
                pivotrl_config=self.config.pivotrl,
                gen_interface=gen_interface,
                nixl_interface=nixl_interface,
            )
            if self.elastic_rm_mode or self.trainer_pool_only_mode:
                rollout_pool_id = self.resource_pool_manager.mapping[PivotRL_Role.Rollout][i]
                rollout_world_size = (
                    rollout_config.rollout.tensor_model_parallel_size
                    * rollout_config.rollout.pipeline_model_parallel_size
                    * rollout_config.rollout.get("data_parallel_size", 1)
                )
                if self.trainer_pool_only_mode and rollout_pool_id != "train_pool":
                    # Main disaggregated instance on an independent pool: non-elastic.
                    rollout_resource_pool = self.resource_pool_manager.get_resource_pool(PivotRL_Role.Rollout, i)
                    self._elastic_bundle_range_by_rollout_instance.append(None)
                    self._elastic_pool_id_by_rollout_instance.append(rollout_pool_id)
                else:
                    rollout_resource_pool = _build_elastic_sub_resource_pool(
                        subgroup_world_size=rollout_world_size,
                        tag=f"rollout_{i}",
                        group_key="rollout",
                        pool_id=rollout_pool_id,
                    )
                    sb = rollout_resource_pool.start_bundle_index
                    sw = rollout_resource_pool.subgroup_world_size
                    self._elastic_bundle_range_by_rollout_instance.append((sb, sb + sw))
                    self._elastic_pool_id_by_rollout_instance.append(rollout_pool_id)
            else:
                rollout_resource_pool = self.resource_pool_manager.get_resource_pool(PivotRL_Role.Rollout, i)
            _register_resource_pool(rollout_resource_pool)
            self.resource_pool_to_cls[rollout_resource_pool][f"rollout_{i}"] = rollout_cls

        # create validation rollout instance
        for i in range(self.n_validate_instances):
            gen_interface = GenInterface(
                rollout_instance_id=self.n_rollout_instances + i,
                ps_manager_handle=self.ps_manager_handle,
                status_queue=self.status_queues[self.n_rollout_instances + i],
            )
            val_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PivotRL_Role.Validate],
                config=self.config.train_actor_rollout_ref,
                role="validate",
                pivotrl_config=self.config.pivotrl,
                gen_interface=gen_interface,
                nixl_interface=nixl_interface,
            )
            val_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(PivotRL_Role.Validate, i)
            self.resource_pool_to_cls[val_rollout_resource_pool][f"validate_{i}"] = val_rollout_cls

        # create actor (train only)
        train_interface = TrainInterface(ps_manager_handle=self.ps_manager_handle)
        actor_resource_pool = self.resource_pool_manager.get_resource_pool(PivotRL_Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[PivotRL_Role.Actor],
            config=self.config.train_actor_rollout_ref,
            role="actor",
            pivotrl_config=self.config.pivotrl,
            train_interface=train_interface,
            nixl_interface=nixl_interface,
            distillation_config=self.config.get("distillation", None),
        )
        self.resource_pool_to_cls[actor_resource_pool]["actor"] = actor_cls

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(PivotRL_Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PivotRL_Role.Critic],
                config=self.config.critic,
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(PivotRL_Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[PivotRL_Role.RefPolicy],
                config=self.config.train_actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create reward model instances
        reward_model_pool_offset = 0
        for reward_model in self.config.reward_models_config.reward_models:
            if reward_model.reward_loop_type not in ("gen", "opd"):
                continue
            reward_model_name = reward_model.get("reward_model_name", reward_model.model.path.split("/")[-1])
            reward_model_cfg = reward_model
            for i in range(reward_model.num_replicas):
                if self.elastic_rm_mode or self.trainer_pool_only_mode:
                    reward_model_pool_id = self.resource_pool_manager.mapping[PivotRL_Role.RewardModel][
                        reward_model_pool_offset
                    ]
                    reward_model_world_size = (
                        reward_model_cfg.rollout.tensor_model_parallel_size
                        * reward_model_cfg.rollout.pipeline_model_parallel_size
                        * reward_model_cfg.rollout.get("data_parallel_size", 1)
                    )
                    if self.trainer_pool_only_mode and reward_model_pool_id != "train_pool":
                        reward_model_resource_pool = self.resource_pool_manager.resource_pool_dict[
                            f"reward_pool_{reward_model_name}_{i}"
                        ]
                        self._elastic_bundle_range_by_reward_model.setdefault(reward_model_name, []).append(None)
                        self._elastic_pool_id_by_reward_model.setdefault(reward_model_name, []).append(reward_model_pool_id)
                        reward_model_pool_offset += 1
                    else:
                        reward_model_resource_pool = _build_elastic_sub_resource_pool(
                            subgroup_world_size=reward_model_world_size,
                            tag=f"reward_model_{reward_model_name}_{i}",
                            group_key="reward_model",
                            pool_id=reward_model_pool_id,
                        )
                        sb = reward_model_resource_pool.start_bundle_index
                        sw = reward_model_resource_pool.subgroup_world_size
                        self._elastic_bundle_range_by_reward_model.setdefault(reward_model_name, []).append((sb, sb + sw))
                        self._elastic_pool_id_by_reward_model.setdefault(reward_model_name, []).append(reward_model_pool_id)
                        reward_model_pool_offset += 1
                else:
                    reward_model_resource_pool = self.resource_pool_manager.resource_pool_dict[f"reward_pool_{reward_model_name}_{i}"]
                _register_resource_pool(reward_model_resource_pool)

                reward_model_gen_if = GenInterface(
                    rollout_instance_id=i,
                    status_queue=self.reward_model_status_queues_mapping[reward_model_name][i],
                )
                reward_model_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[PivotRL_Role.RewardModel],
                    config=reward_model_cfg,
                    role="reward",
                    pivotrl_config=self.config.pivotrl,
                    instance_id=i,
                    gen_interface=reward_model_gen_if,
                    reward_model_name=reward_model_name,
                    rm_config=reward_model,
                    is_teacher_model=reward_model.reward_loop_type == "opd",
                )
                # max_concurrency only: Ray disallows concurrency_groups in .options() for this version;
                # concurrency_groups is set on ray.remote(PivotRL_RewardModelWorker, ...) in main_ppo.py.
                # reward_model_cls.update_options({"max_concurrency": self.max_concurrency})

                self.resource_pool_to_cls[reward_model_resource_pool][f"reward_model_{reward_model_name}_{i}"] = reward_model_cls
        
        if not self.use_critic and not self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(PivotRL_Role.DummyPolicy)
            dummy_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[PivotRL_Role.DummyPolicy],
                config=self.config.train_actor_rollout_ref,
                role="dummy",
            )
            self.resource_pool_to_cls[resource_pool]["dummy"] = dummy_policy_cls

        # initialize WorkerGroup
        pivotrl_logger.info("Initializing WorkerGroup for other roles")

        # NOTE(verl): if you want to use a different resource pool for each role,
        # which can support different parallel size,
        # you should not use `create_colocated_worker_cls_fused`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        def create_worker_group(resource_pool, class_dict, wg_kwargs=wg_kwargs):
            # if there is only one worker class in the resource pool, we can directly create a worker group
            # so that we can use 'execute_all_async' and other low-level APIs
            # NOTE: in newest verl, we can use `create_colocated_worker_cls_fused`
            # to create a fused worker group and low-level APIs can also be used
            if len(class_dict) == 1:
                role = next(iter(class_dict.keys()))
                ray_worker_group_cls = (
                    RayWorkerGroup if "rollout" in role or "validate" in role else self.ray_worker_group_cls
                )
                return {
                    role: ray_worker_group_cls(
                        resource_pool=resource_pool,
                        ray_cls_with_init=class_dict[role],
                        **wg_kwargs,
                    )
                }
            # colocate
            else:
                worker_dict_cls = create_colocated_worker_cls_fused(class_dict=class_dict)
                wg_dict = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=worker_dict_cls,
                    **wg_kwargs,
                )
                return wg_dict.spawn(prefix_set=class_dict.keys())

        def _run_worker_group_tasks(tasks, label: str):
            """Create worker groups with a thread pool; safely handle empty task lists."""

            if not tasks:
                pivotrl_logger.info(f"No {label} worker group to create; skipping.")
                return

            # We create one thread per task; ThreadPoolExecutor requires max_workers > 0
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                futures = {}
                for resource_pool, class_dict, task_wg_kwargs in tasks:
                    future = executor.submit(create_worker_group, resource_pool, class_dict, task_wg_kwargs)
                    futures[future] = (resource_pool, class_dict)
                for future in futures:
                    result = future.result()
                    all_wg.update(result)

        # coroutine version
        """
        async def async_create_worker_groups():
            tasks = []
            for resource_pool, class_dict in self.resource_pool_to_cls.items():
                pivotrl_logger.info(f"Creating worker group for resource pool: {resource_pool}, classes: {class_dict}")
                if "ps" in class_dict:
                    assert class_dict.keys() == {"ps"}, "PS resource pool should only have PS role."
                    continue
                tasks.append((resource_pool, class_dict))
            
            async def create_single_worker_group(resource_pool, class_dict):
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = loop.run_in_executor(executor, create_worker_group, resource_pool, class_dict)
                    return await future
            
            # Concurrency control within max_concurrent tasks at a time
            # NOTE: currently set to 1 (the default sync version) to
            # avoid the stuck issue when multiple bundles are trying to be placed
            # at the same time (verl is using STRICT_PACK mode)
            # To reproduce the issue, you can set it to 16 and
            # run `pivotrl/examples/precision_test/dapo/megatron_qwen_7b_aime.sh`
            # where GEN_NNODES=$(( ${NNODES} / 2 )) and TRAIN_NNODES=$(( ${NNODES} / 2 )) 
            max_concurrent = min(len(tasks), 1)
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def controlled_create(resource_pool, class_dict):
                async with semaphore:
                    return await create_single_worker_group(resource_pool, class_dict)
            
            coroutines = [controlled_create(rp, cd) for rp, cd in tasks]
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            all_wg_async = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    resource_pool, class_dict = tasks[i]
                    pivotrl_logger.error(
                        f"Error creating worker group for {resource_pool}, "
                        f"class {class_dict}: {str(result)}"
                    )
                    raise result
                all_wg_async.update(result)
            
            return all_wg_async

        async_results = asyncio.run(async_create_worker_groups())
        all_wg.update(async_results)
        """

        # multi-thread version
        train_tasks = []
        gen_tasks = []
        val_tasks = []
        reward_model_tasks = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            pivotrl_logger.info(f"Creating worker group for resource pool: {resource_pool}, classes: {class_dict}")
            if "ps" in class_dict:
                assert class_dict.keys() == {"ps"}, "PS resource pool should only have PS role."
                continue
            if any("rollout" in key for key in class_dict.keys()):
                assert len(class_dict) == 1, "Rollout resource pool should only have one worker class."
                gen_tasks.append((resource_pool, class_dict, wg_kwargs))
            elif any("validate" in key for key in class_dict.keys()):
                assert len(class_dict) == 1, "Validate resource pool should only have one worker class."
                val_tasks.append((resource_pool, class_dict, wg_kwargs))
            elif any("reward_model" in key for key in class_dict.keys()):
                assert len(class_dict) == 1, "Reward model resource pool should only have one worker class."
                reward_model_tasks.append((resource_pool, class_dict, wg_kwargs))
            else:
                # NOTE: adapt wg_kwargs for fused train worker
                # if want to add specific env args.
                if self.config.pivotrl.tms.range in ["train", "all"] or self.config.pivotrl.tms.enable_nixl:
                    # add tms config to train workers
                    import torch_memory_saver

                    dynlib_path = os.path.join(
                        os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                        "torch_memory_saver_hook_mode_preload.abi3.so",
                    )
                    assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

                    train_wg_kwargs = wg_kwargs.copy()
                    train_wg_kwargs["worker_env"] = {
                        "LD_PRELOAD": dynlib_path,
                        "TMS_INIT_ENABLE": "1",
                        "TMS_INIT_ENABLE_CPU_BACKUP": "0",
                        # NOTE: torch_memory_saver is not compatible with expandable segments
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                        "PIVOTRL_TMS_ENABLE": "1" if self.config.pivotrl.tms.range in ["train", "all"] else "",
                    }
                else:
                    train_wg_kwargs = wg_kwargs
                    # NOTE: Still cannot use expandable segments, will cause NIXL error
                    # train_wg_kwargs["worker_env"] = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                pivotrl_logger.info(f"train_wg_kwargs: {train_wg_kwargs}")
                train_tasks.append((resource_pool, class_dict, train_wg_kwargs))
        # We must execute train tasks first because rollout instances may occupy
        # the resources randomly and no structured resources are available for training
        _run_worker_group_tasks(train_tasks, label="train")
        _run_worker_group_tasks(reward_model_tasks, label="reward_model")
        _run_worker_group_tasks(gen_tasks, label="gen")
        _run_worker_group_tasks(val_tasks, label="validate")
        """
        # sync version
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if "ps" in class_dict:
                assert class_dict.keys() == {"ps"}, "PS resource pool should only have one worker class."
                continue # PS is created first, so we skip it here
            all_wg.update(create_worker_group(resource_pool, class_dict))
        """

        # create reward model managers
        pivotrl_logger.info("Creating reward model managers")
        reward_models_config = self.config.reward_models_config
        for reward_model in reward_models_config.reward_models:
            if reward_model.reward_loop_type not in ("gen", "opd"):
                continue
            reward_model_name = reward_model.get("reward_model_name", reward_model.model.path.split("/")[-1])
            reward_model_wg_list = [all_wg[f"reward_model_{reward_model_name}_{i}"] for i in range(reward_model.num_replicas)]
            reward_model_status_queues = [self.reward_model_status_queues_mapping[reward_model_name][i] for i in range(reward_model.num_replicas)]
            self.reward_model_manager_mapping[reward_model_name] = PivotRL_RewardModelManager(
                reward_model_name=reward_model_name,
                config=self.config,
                reward_model_config=reward_model,
                reward_model_wg_list=reward_model_wg_list,
                status_queues=reward_model_status_queues,
                max_concurrency=self.max_concurrency,
            )
        pivotrl_logger.info(f"reward_model_manager_mapping: {self.reward_model_manager_mapping}")
        
        # create agent loop workers
        self.agent_loop_workers = []
        self.rollout_wg_list = [all_wg[f"rollout_{i}"] for i in range(self.n_rollout_instances)]
        self.validate_wg_list = [all_wg[f"validate_{i}"] for i in range(self.n_validate_instances)]
        self.init_rollout_router()
        max_concurrency_per_worker = (
            self.max_concurrency // self.config.gen_actor_rollout_ref.rollout.agent.num_workers
        )
        for i in range(self.config.gen_actor_rollout_ref.rollout.agent.num_workers):
            self.agent_loop_workers.append(
                PivotRL_AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}", max_concurrency=max_concurrency_per_worker
                ).remote(
                    self.config,
                    self.ps_manager_handle,
                    self.rollout_router,
                    self.rollout_wg_list + self.validate_wg_list,
                )
            )

        pivotrl_logger.info("Initializing models and NIXL clients")
        nixl_client_futures = []
        model_init_futures = []

        # create PS WorkerGroup
        pivotrl_logger.info("Create PS WorkerGroup")
        train_model_dtype = (
            torch.bfloat16 if self.config.train_actor_rollout_ref.actor.strategy == "megatron" else torch.float32
        )
        storage_plan = PSStoragePlan(
            train_model_dtype=train_model_dtype,
            gen_model_dtype=PrecisionType.to_dtype(self.config.gen_actor_rollout_ref.rollout.dtype),
        )
        if self.config.pivotrl.ps_mode == "cpu" or self.config.pivotrl.ps_mode == "cpu_ref":
            # PSManager is used to store the model state dict
            # No need to create PS WorkerGroup
            pass
        elif self.config.pivotrl.ps_mode == "nixl_cpu" or self.config.pivotrl.ps_mode == "nixl_gpu":
            # PSManager is only used to build the nixl meta server
            # The PS WorkerGroup is used to store the model state dict
            # It is colocate with the rollout instances
            assert self.config.pivotrl.nixl.server_ip == self.config.pivotrl.ps_manager_ip, (
                "PSManager IP and NIXL server IP must be the same"
            )
            if self.config.pivotrl.ps_mode == "nixl_cpu":
                # ps is deployed on both generation and (maybe) actor nodes
                ps_node_ids = set()

                # Get all rollout instances' distinct node ids
                for i in range(self.n_rollout_instances):
                    rollout_instance_node_ids = all_wg[f"rollout_{i}"].execute_all_sync("get_node_id")
                    for node_id in rollout_instance_node_ids:
                        ps_node_ids.add(node_id)
                    self.worker_to_node_id.update(
                        {
                            WorkerKey("rollout", i, idx): node_id
                            for idx, node_id in enumerate(rollout_instance_node_ids)
                        }
                    )

                # Get all actor instances' distinct node ids
                actor_instance_node_ids = all_wg["actor"].execute_all_sync("get_node_id")
                for node_id in actor_instance_node_ids:
                    ps_node_ids.add(node_id)
                self.worker_to_node_id.update(
                    {WorkerKey("actor", 0, idx): node_id for idx, node_id in enumerate(actor_instance_node_ids)}
                )

                # Get all validate instances' distinct node ids
                for i in range(self.n_validate_instances):
                    validate_instance_node_ids = all_wg[f"validate_{i}"].execute_all_sync("get_node_id")
                    for node_id in validate_instance_node_ids:
                        ps_node_ids.add(node_id)
                    self.worker_to_node_id.update(
                        {
                            WorkerKey("validate", i, idx): node_id
                            for idx, node_id in enumerate(validate_instance_node_ids)
                        }
                    )

                ps_spec_list = []
                ps_node_ids = list(ps_node_ids)
                # Map each worker to a PS index
                for i, node_id in enumerate(ps_node_ids):
                    self.worker_to_ps_idx.update(
                        {worker: i for worker, nid in self.worker_to_node_id.items() if nid == node_id}
                    )
                pivotrl_logger.info(f"Worker to node id: {self.worker_to_node_id}")
                pivotrl_logger.info(f"Worker to PS id: {self.worker_to_ps_idx}")

                for node_id in ps_node_ids:
                    ps_spec_list.append(PSResourceSpec(node_id=node_id, attached_gpu_id=None))
                ps_resource_pool = PSResourcePool(ps_spec_list=ps_spec_list)
                pivotrl_logger.info(f"PS resource pool: {ps_resource_pool}")
                self.ps_wg = PSWorkerGroup(
                    resource_pool=ps_resource_pool,
                    ps_cls_with_init=PSClassWithInitArgs(
                        cls=ray.remote(PSStorageWorker),
                        storage_plan=storage_plan,
                        model_config=self.config.train_actor_rollout_ref.model,
                        pivotrl_config=self.config.pivotrl,
                        nixl_interface=nixl_interface,
                    ),
                )
                if self.config.pivotrl.ps_mode == "nixl_cpu" or self.config.pivotrl.ps_mode == "nixl_gpu":
                    nixl_client_futures.extend(self.ps_wg.execute_all_async("init_nixl_client"))
                # Init model skeleton on meta device; weights are loaded after NIXL protocol completes.
                model_init_futures.extend(self.ps_wg.execute_all_async("init_model"))
                # NOTE: dispatch preload immediately into each PS actor's serial queue; it will
                # start as soon as that actor's init_model completes, overlapping with gen/val/train
                # model initialization and NIXL protocol to hide disk I/O latency.
                preload_futures = self.ps_wg.execute_all_async("preload_checkpoint_to_cpu")
                pivotrl_logger.info("PS model initialized successfully!")
            elif self.config.pivotrl.ps_mode == "nixl_gpu":
                raise NotImplementedError("PS mode 'nixl_gpu' is not implemented yet")
        else:
            raise ValueError(f"Invalid PS mode: {self.config.pivotrl.ps_mode}")

        # Start rollout gateway to build rollout service
        self.start_rollout_gateway()

        pivotrl_logger.info("Initializing models in all rollout instances")
        # start rollout coordinator
        self.init_rollout_coordinator()
        isolated_elastic_validation_init = (
            self.elastic_trainer_pool_mode
            and self.config.pivotrl.colocate_validate_and_train
            and self.n_validate_instances > 0
        )
        # NOTE: must use rollout coordinator to init model so it sets _is_init_model_events
        # for rollout indices, which init_nixl_client waits on.
        rollout_init_mode = "full" if self.config.pivotrl.ps_mode == "cpu_ref" else "empty"
        if not isolated_elastic_validation_init:
            model_init_futures.append(
                self.rollout_coordinator.init_model.remote("rollout", rollout_init_mode)
            )

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        # self.rm_wg = None
        # if self.use_rm:
        #     self.rm_wg = all_wg["rm"]
        #     self.rm_wg.init_model()

        if not (self.use_critic or (self.use_reference_policy and not self.ref_in_actor)):
            # NOTE: when not using critic or reference policy,
            # if we directly call `init_model` of actor_wg, Ray will view the fused worker
            # as an async actor and run `run_async_func_or_coro_in_event_loop`, which will
            # make it invalid to call async function such as `trainer_mode` in `init_model`.
            # So here we create a dummy worker group and call its dummy method to avoid this issue.
            self.dummy_wg = all_wg["dummy"]
            self.dummy_wg.init_model()

        if self.is_rollout_mode_in_actor or isolated_elastic_validation_init:
            assert self.config.pivotrl.ps_mode == "nixl_cpu" or self.config.pivotrl.ps_mode == "nixl_gpu", (
                "Fused trainer and validator only support NIXL PS mode."
            )
            # init actor wg -> offload -> init validate wg
            pivotrl_logger.info("Initializing actor model")
            self.actor_wg = all_wg["actor"]
            nixl_client_futures.extend(self.actor_wg.execute_all_async("init_nixl_client"))
            ray.get(nixl_client_futures)
            pivotrl_logger.info("Initialized NIXL client in actor worker group")
            self.actor_wg.init_model("empty")
            ray.get(self.actor_wg.execute_all_async("nixl_convert_params"))
            if isolated_elastic_validation_init:
                self._sleep_trainer_for_elastic_trainer_pool()
            else:
                ray.get(self.actor_wg.execute_all_async("nixl_sleep", "meta"))

            pivotrl_logger.info("Initializing validation model")
            # NOTE: here we must use rollout coordinator to init model
            # for setting init event inside it.
            model_init_futures.append(self.rollout_coordinator.init_model.remote("validate", "empty"))
            ray.get(model_init_futures)
            if isolated_elastic_validation_init:
                self._sleep_validation_workers_after_initialization()
                pivotrl_logger.info(
                    "Elastic trainer pool: validation workers initialized and slept; "
                    "initializing rollout workers with train_pool released."
                )
                ray.get(
                    self.rollout_coordinator.init_model.remote(
                        "rollout",
                        rollout_init_mode,
                    )
                )
            ray.get(self.rollout_coordinator.init_nixl_client.remote())
            ray.get(self.rollout_coordinator.nixl_convert_params.remote())
            ray.get(self.rollout_coordinator.init_route_strategy.remote("all"))

        else:
            # init validate wg -> offload -> init actor wg
            # ray.get(model_init_futures)
            pivotrl_logger.info("Initializing validation model")
            # NOTE: here we must use rollout coordinator to init model
            # for setting init event inside it.
            model_init_futures.append(self.rollout_coordinator.init_model.remote("validate", "empty"))
            ray.get(model_init_futures)
            ray.get(self.rollout_coordinator.init_nixl_client.remote())
            ray.get(self.rollout_coordinator.nixl_convert_params.remote())
            ray.get(self.rollout_coordinator.init_route_strategy.remote("all"))
            # Pause validate instances in the router before sleeping them, so that the router
            # does not route rollout requests to validate instances that are in sleep state.
            # (fuse_rollout_with_validate=True makes all instances available by default)
            init_paused_instance_ids = list(
                range(self.n_rollout_instances, self.n_rollout_instances + self.n_validate_instances)
            )
            ray.get(
                [
                    self.rollout_coordinator.sleep.remote("validate"),
                    self.rollout_router.pause_instances.remote(init_paused_instance_ids),
                ]
            )

            pivotrl_logger.info("Initializing actor model")
            self.actor_wg = all_wg["actor"]
            self.actor_wg.init_model("empty")
            nixl_client_futures.extend(self.actor_wg.execute_all_async("init_nixl_client"))
            ray.get(nixl_client_futures)
            ray.get(self.actor_wg.execute_all_async("nixl_convert_params"))

        pivotrl_logger.info("All workers' models initialized successfully!")

        # initialize NIXL
        if self.config.pivotrl.ps_mode == "nixl_cpu" or self.config.pivotrl.ps_mode == "nixl_gpu":
            rollout_world_size = ray.get(self.rollout_coordinator.world_size.remote())
            pivotrl_logger.info(
                f"Initializing NIXL server with {self.ps_wg.world_size} PS workers, "
                f"{self.actor_wg.world_size} actor workers, {rollout_world_size} rollout workers"
            )
            expected_nixl_client_agents = self.ps_wg.world_size + self.actor_wg.world_size + rollout_world_size
            ray.get(self.ps_manager_handle.init_nixl_server.remote(expected_nixl_client_agents))
            actor_protocol_mode = "meta" if self._elastic_trainer_pool_trainer_sleeping else "full"
            rollout_full_tag = "all" if self.is_rollout_mode_in_actor else "rollout"

            with log_dual_events("Executing NIXL protocol", pivotrl_logger, event_type=EventType.INIT):
                futures = []
                futures.append(self.ps_manager_handle.nixl_protocol.remote())
                futures.extend(self.ps_wg.execute_all_async("nixl_protocol"))
                futures.extend(self.actor_wg.execute_all_async("nixl_protocol", actor_protocol_mode))
                futures.append(self.rollout_coordinator.nixl_protocol.remote(rollout_full_tag))
                pending_futures = list(futures)
                while pending_futures:
                    ready_futures, pending_futures = ray.wait(pending_futures, num_returns=1)
                    # Consume each completed future immediately so a failed server/client
                    # task is surfaced instead of being hidden behind other 600s waits.
                    ray.get(ready_futures)

            # Now that all NIXL buffers are allocated (meta tensors replaced),
            # write the preloaded checkpoint tensors into the PS registered buffers.
            with log_dual_events("Loading PS checkpoint weights", pivotrl_logger, event_type=EventType.INIT):
                # Ensure prefetch finished (likely already done; blocks only if preload
                # outlasted NIXL protocol, which would be unusual).
                ray.get(preload_futures)
                ray.get(self.ps_wg.execute_all_async("write_checkpoint_to_registered_tensors"))

            # Bind the PS worker group to PSManager before initial pull, so that PSManager's
            # ps_nixl_agent_names are populated and gen workers can call get_ps_nixl_agent_names.
            pivotrl_logger.info("Binding PS worker group")
            ray.get(self.ps_manager_handle.bind_ps_worker_group.remote(self.ps_wg))
            pivotrl_logger.info("PS worker group bound successfully!")
            if (
                resolve_weight_fingerprint_options(
                    self.config.pivotrl,
                    flow="transfer_chain",
                    model_version=0,
                )
                is not None
            ):
                ray.get(self.ps_wg.execute_all_async("log_weight_fingerprints", 0))

            # Pull checkpoint weights from PS into active gen and actor workers via NIXL.
            # Sleeping actor/validation workers pull after their corresponding wake transition.
            initial_pull_tag = "all" if self.is_rollout_mode_in_actor else "rollout"
            with log_dual_events("Initial pull: PS → gen/actor workers", pivotrl_logger, event_type=EventType.INIT):
                initial_pull_futures = []
                initial_pull_futures.append(self.rollout_coordinator.initial_pull_from_ps.remote(initial_pull_tag))
                if not self._elastic_trainer_pool_trainer_sleeping:
                    initial_pull_futures.extend(self.actor_wg.execute_all_async("pull_model"))
                ray.get(initial_pull_futures)

        if self._elastic_trainer_pool_instance_entries():
            self._sleep_trainer_for_elastic_trainer_pool()
        self._init_elastic_rm_runtime()

    def switch_to_rollout_mode(self):
        """Switch the PivotRL colocate part to rollout mode for validation.

        This involves several steps to ensure that the system transitions smoothly
        from training to rollout mode, particularly when validation and training are
        colocated.
        1. Deregister actor clients from NIXL to free up resources.
        2. Wake up validation instances and re-register their local tensors.
        3. Sync with the PS manager to update client information.
        4. Broadcast updated client information to all relevant clients.
        5. Pull current weights into validation instances.
        6. Resume validation instances in the router and coordinator.
        """
        if not self.config.pivotrl.colocate_validate_and_train or self.is_rollout_mode_in_actor:
            return
        if self.n_validate_instances == 0:
            pivotrl_logger.info("Skip switching to rollout mode because no validation instances are configured.")
            return

        _switch_start = time.time()
        pivotrl_logger.info("Switching to rollout mode...")

        _t = time.time()
        pivotrl_logger.info("Step 1 - Deregistering actor clients from NIXL...")
        if self.elastic_trainer_pool_mode:
            self._sleep_trainer_for_elastic_trainer_pool()
        else:
            release_futures = self.actor_wg.execute_all_async("nixl_sleep", "full")
            ray.get(release_futures)
        # Mark the transition before waking validation workers so the finally
        # path can reclaim partially-woken instances if any later stage fails.
        self.is_rollout_mode_in_actor = True
        pivotrl_logger.info(f"Step 1 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        pivotrl_logger.info("Step 2 - Waking up validation instances...")
        resumed_instance_ids = list(
            range(
                self.n_rollout_instances,
                self.n_rollout_instances + self.n_validate_instances,
            )
        )
        wake_futures = [
            validate_wg.execute_rank_zero_async("nixl_wake_up")
            for validate_wg in self.validate_wg_list
        ]
        ray.get(wake_futures)
        pivotrl_logger.info(f"Step 2 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        pivotrl_logger.info("Step 3 - Syncing with ps manager...")
        # sync with server
        updated_client_names = []  # to collect all updated client names for broadcasting
        futures = []
        expected_update_infos = 0
        for i in range(self.n_validate_instances):
            tp_size = sum(1 for k in self.worker_to_ps_idx if k.role == "validate" and k.instance_id == i)
            expected_update_infos += tp_size
            for rank in range(tp_size):
                updated_client_names.append(
                    WorkerKey("validate", i, rank).to_nixl_client_name(self.n_rollout_instances)
                )
            futures.append(
                self.validate_wg_list[i].execute_rank_zero_async("nixl_send_local_info_to", NIXL_META_SERVER_NAME)
            )
        # wait for ps manager to collect all infos
        futures.append(self.ps_manager_handle.nixl_wait_for_update_infos.remote(expected_update_infos))
        ray.get(futures)
        pivotrl_logger.info(f"Step 3 done in {time.time() - _t:.2f}s.")

        # broadcast to other clients
        _t = time.time()
        pivotrl_logger.info("Step 4 - PS manager broadcasting updated client infos...")
        self._broadcast_updated_client_infos_from_ps_manager(updated_client_names)
        pivotrl_logger.info(f"Step 4 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        pivotrl_logger.info("Step 5 - Pulling validation weights from PS...")
        ray.get(self.rollout_coordinator.sync_with_ps.remote(resumed_instance_ids))
        pivotrl_logger.info(f"Step 5 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        pivotrl_logger.info("Step 6 - Resuming validation instances...")
        # resume validation instances in router and coordinator
        futures = []
        futures.append(self.rollout_router.resume_instances.remote(resumed_instance_ids))
        futures.append(self.rollout_coordinator.resume_instances.remote(resumed_instance_ids))
        ray.get(futures)
        pivotrl_logger.info(f"Step 6 done in {time.time() - _t:.2f}s.")

        pivotrl_logger.info(f"Switched to rollout mode in {time.time() - _switch_start:.2f}s.")

    def switch_to_trainer_mode(self, *, restore_trainer: bool = True):
        """
        Switch colocated validation workers out of rollout mode.

        Args:
            restore_trainer (bool): Whether to wake and synchronize the trainer.
                Elastic validation sets this to False and returns train_pool
                directly to elastic rollout and reward-model replicas.

        This involves several steps to ensure that the system transitions smoothly.
        1. Pause and sleep validation instances through the coordinator.
        2. Wake up the training actor and allocate necessary resources.
        3. Sync with the PS manager and broadcast updated client information.
        4. Pull the latest model weights from the PS to the actor.
        """
        # notify coordinator + interrupt + sleep + upload actor
        if not self.config.pivotrl.colocate_validate_and_train or not self.is_rollout_mode_in_actor:
            return

        _switch_start = time.time()
        pivotrl_logger.info("Switching to trainer mode...")

        _t = time.time()
        pivotrl_logger.info("Step 1 - Pausing and sleeping validation instances...")
        paused_instance_ids = list(range(
            self.n_rollout_instances,
            self.n_rollout_instances + self.n_validate_instances,
        ))
        ray.get(self.rollout_coordinator.pause_instances.remote(paused_instance_ids))
        sleep_result = ray.get(
            self.rollout_coordinator.exec_command.remote(
                Command(type=CommandType.SLEEP, instance_ids=paused_instance_ids),
                blocking=True,
            )
        )
        if sleep_result is not True:
            raise RuntimeError(
                f"Failed to sleep validation instances {paused_instance_ids}: {sleep_result!r}"
            )
        pivotrl_logger.info(f"Step 1 done in {time.time() - _t:.2f}s.")

        if not restore_trainer:
            self.is_rollout_mode_in_actor = False
            if self.elastic_trainer_pool_mode:
                self._leave_elastic_trainer_pool_training_window()
            pivotrl_logger.info(
                f"Switched validation workers to sleep and returned train_pool "
                f"to elastic replicas in {time.time() - _switch_start:.2f}s."
            )
            return

        _t = time.time()
        pivotrl_logger.info("Step 2 - Waking up training actor...")
        # Allocate trainer space and register
        ray.get(self.actor_wg.execute_all_async("nixl_wake_up"))
        pivotrl_logger.info(f"Step 2 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        pivotrl_logger.info("Step 3 - Syncing with ps manager...")
        # sync with server
        update_client_names = []  # to collect all updated client names for broadcasting
        futures = []
        for i in range(self.actor_wg.world_size):
            update_client_names.append(train_client_name(i))
        # sender side: actor workers
        futures.extend(self.actor_wg.execute_all_async("nixl_send_local_info_to", NIXL_META_SERVER_NAME))
        # receiver side: ps manager
        futures.append(self.ps_manager_handle.nixl_wait_for_update_infos.remote(self.actor_wg.world_size))
        ray.get(futures)
        pivotrl_logger.info(f"Step 3 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        pivotrl_logger.info("Step 4 - PS manager broadcasting updated client infos...")
        self._broadcast_updated_client_infos_from_ps_manager(update_client_names)
        pivotrl_logger.info(f"Step 4 done in {time.time() - _t:.2f}s.")

        _t = time.time()
        pivotrl_logger.info("Step 5 - Pulling actor model from PS...")
        # pull actor model
        ray.get(self.actor_wg.execute_all_async("pull_model"))
        self._clear_fsdp2_grads_after_trainer_wake()
        pivotrl_logger.info(f"Step 5 done in {time.time() - _t:.2f}s.")

        pivotrl_logger.info(f"Switched to trainer mode in {time.time() - _switch_start:.2f}s.")
        self.is_rollout_mode_in_actor = False

    def _make_broadcast_plan(self, src_agent_names, dst_agent_names) -> dict:
        """Create a broadcast plan mapping source agents to destination agents.

        Args:
            src_agent_names (list): List of source agent names.
            dst_agent_names (list): List of destination agent names.
        Returns:
            dict: A dictionary mapping each source agent to a list of destination agents.
        """
        # simple round-robin broadcast plan
        # NOTE: currently only PS manager broadcasting is implemented.
        # This method can be extended for more complex plans if needed.
        broadcast_plan = {src_agent: [] for src_agent in src_agent_names}
        for i, dst_agent in enumerate(dst_agent_names):
            src_agent = src_agent_names[i % len(src_agent_names)]
            broadcast_plan[src_agent].append(dst_agent)
        return broadcast_plan

    def _broadcast_updated_client_infos_from_ps_manager(self, updated_client_names: list):
        """Broadcast updated client infos from PS manager to all nixl clients.

        Args:
            updated_client_names (list): List of updated client names to broadcast.
        """
        src_agent_names = [NIXL_META_SERVER_NAME]
        dst_agent_names = []
        # 1. PS storage workers
        for i in range(self.ps_wg.world_size):
            dst_agent_names.append(ps_agent_name(i))
        # 2. rollout workers
        for i in range(self.n_rollout_instances):
            tp_size = sum(1 for k in self.worker_to_ps_idx if k.role == "rollout" and k.instance_id == i)
            for rank in range(tp_size):
                dst_agent_names.append(WorkerKey("rollout", i, rank).to_nixl_client_name())
        # 3. validate workers
        for i in range(self.n_validate_instances):
            tp_size = sum(1 for k in self.worker_to_ps_idx if k.role == "validate" and k.instance_id == i)
            for rank in range(tp_size):
                dst_agent_names.append(WorkerKey("validate", i, rank).to_nixl_client_name(self.n_rollout_instances))
        # 4. actor workers
        for i in range(self.actor_wg.world_size):
            dst_agent_names.append(train_client_name(i))
        pivotrl_logger.debug(f"Destination agent names for broadcasting: {dst_agent_names}")

        broadcast_plan = self._make_broadcast_plan(src_agent_names, dst_agent_names)
        futures = []
        for src_agent, dst_agents in broadcast_plan.items():
            if src_agent == NIXL_META_SERVER_NAME:
                futures.append(
                    self.ps_manager_handle.nixl_broadcast_update_client_infos.remote(dst_agents, updated_client_names)
                )
            else:
                raise NotImplementedError(
                    "Only meta server broadcasting is implemented in _broadcast_updated_client_infos_from_ps_manager"
                )

        # recv broadcast results by all clients
        # 1. ps storage workers
        futures.extend(self.ps_wg.execute_all_async("nixl_wait_for_update_infos", 1))
        # 2. rollout workers
        for i in range(self.n_rollout_instances):
            futures.append(self.rollout_wg_list[i].execute_rank_zero_async("nixl_wait_for_update_infos", 1))
        for i in range(self.n_validate_instances):
            futures.append(self.validate_wg_list[i].execute_rank_zero_async("nixl_wait_for_update_infos", 1))
        # 3. actor workers
        futures.extend(self.actor_wg.execute_all_async("nixl_wait_for_update_infos", 1))
        ray.get(futures)

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        pivotrl_logger.info(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            pivotrl_logger.warning(
                "remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "critic",
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        ray.get(self.data_processor.save_train_dataloader.remote(dataloader_local_path))

        # latest checkpointed iteration tracker (for atomic usage)
        local_mkdir_safe(self.config.trainer.default_local_dir)
        pivotrl_logger.info(f"Saving latest checkpointed iteration to {self.config.trainer.default_local_dir}")
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                pivotrl_logger.info("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        pivotrl_logger.info(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        pivotrl_logger.info(f"Setting global step to {self.global_steps}")
        pivotrl_logger.info(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor (train only)
        self.actor_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # TODO: push the actor model state dict to the PS worker (though it is not necessary to do so)

        # load dataloader
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            ray.get(self.data_processor.load_train_dataloader.remote(dataloader_local_path))
        else:
            pivotrl_logger.info(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            # if self.use_rm:
            #     self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            # if self.use_rm:
            #     self.rm_wg.stop_profile()

    def _get_dp_size(self):
        # query dispatch info of the actor worker group
        if "actor" not in self.actor_wg._dispatch_info:
            self.actor_wg._dispatch_info["actor"] = self.actor_wg._query_dispatch_info("actor")
            assert len(self.actor_wg._dispatch_info["actor"]) == self.actor_wg.world_size
        dp_rank_mapping = self.actor_wg._dispatch_info["actor"]
        dp_size = max(dp_rank_mapping) + 1
        return dp_size

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        dp_size = self._get_dp_size()
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=dp_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        pivotrl_logger.info(
            f"Initialized tracking logger with project: {self.config.trainer.project_name}, "
            f"experiment: {self.config.trainer.experiment_name}"
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if self.global_steps >= self.total_training_steps:
            pivotrl_logger.warning(
                f"Global steps {self.global_steps} >= total training steps {self.total_training_steps}, "
                "skipping training."
            )
            return

        self.init_reward_manager(validation=True)
        # perform validation before training
        if self.val_reward_manager is not None and self.config.trainer.get("val_before_train", True):
            self.init_agent_loop_manager()
            futures = []
            futures.append(self.data_processor.set_agent_loop_manager.remote(self.agent_loop_manager))
            futures.append(self.ps_manager_handle.set_rollout_coordinator.remote(self.rollout_coordinator))
            for agent_loop_worker in self.agent_loop_workers:
                futures.append(agent_loop_worker.set_agent_loop_manager.remote(self.agent_loop_manager))
                # Validation generation may still go through generation agent loops that
                # call `reward_manager.compute_score(...)` (internally they can no-op on validate=True).
                futures.append(agent_loop_worker.set_reward_manager.remote(self.val_reward_manager))
            ray.get(futures)

            self.start_rollout_coordinator()
            self.start_agent_loop_manager()

            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pivotrl_logger.info(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        self.init_agent_loop_manager()

        futures = []
        futures.append(self.data_processor.set_agent_loop_manager.remote(self.agent_loop_manager))
        futures.append(self.ps_manager_handle.set_rollout_coordinator.remote(self.rollout_coordinator))
        for agent_loop_worker in self.agent_loop_workers:
            futures.append(agent_loop_worker.set_agent_loop_manager.remote(self.agent_loop_manager))
        ray.get(futures)

        self.init_reward_manager()
        futures = []
        futures.append(self.data_processor.set_reward_manager.remote(self.reward_manager))
        futures.append(self.ps_manager_handle.set_reward_manager.remote(self.reward_manager))
        for agent_loop_worker in self.agent_loop_workers:
            futures.append(agent_loop_worker.set_reward_manager.remote(self.reward_manager))
        ray.get(futures)

        # Start data pipeline
        if not self.config.pivotrl.colocate:
            # Start rollout coordinator to handle rollouts and data generation
            pivotrl_logger.info("Starting rollout coordinator...")
            self.start_rollout_coordinator()
            pivotrl_logger.info("Rollout coordinator started successfully.")

            # Start agent loop manager to handle agent-environment interactions
            pivotrl_logger.info("Starting agent loop manager...")
            self.start_agent_loop_manager()
            pivotrl_logger.info("Agent loop manager started successfully.")

            # Start reward manager to handle reward computation requests
            pivotrl_logger.info("Starting reward manager...")
            self.start_reward_manager()
            pivotrl_logger.info("Reward manager started successfully.")

        pivotrl_logger.info("All data pipeline components started successfully.")

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_manager is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pivotrl_logger.info(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # Start data processor to handle data preprocessing and batching
        pivotrl_logger.info("Starting data processor...")
        self.start_data_processor()
        pivotrl_logger.info("Data processor started successfully.")

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # busy loop for training
        while True:
            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps == self.total_training_steps
            self._set_elastic_training_step()

            with marked_timer("step", timing_raw):
                # Wait for the training batch to be ready
                with marked_timer("wait_for_gen", timing_raw, color="gray"):
                    if not self.config.pivotrl.colocate:
                        buffer_id = self.global_steps - 1
                        # Apply the same proactive buffer preparation used by the
                        # blocking wait before deciding whether to lend train_pool
                        # GPUs. This keeps the actor awake when a near-ready buffer
                        # can be completed immediately.
                        batch_ready_without_switch = False
                        if (
                            (self.elastic_trainer_pool_mode or self.trainer_pool_only_mode)
                            and not self.colocated_mode
                            and self._elastic_trainer_pool_training_active
                        ):
                            batch_ready_without_switch = ray.get(
                                self.agent_loop_manager.prepare_training_batch_if_ready.remote(buffer_id)
                            )
                            grace_s = float(
                                self.config.pivotrl.deployment.elastic_rm.get("trainer_ready_grace_s", 0.0)
                            )
                            if not batch_ready_without_switch and grace_s > 0:
                                pivotrl_logger.info(
                                    "Keeping trainer awake for up to %.2fs while buffer %d becomes ready.",
                                    grace_s,
                                    buffer_id,
                                )
                                batch_ready_without_switch = ray.get(
                                    self.agent_loop_manager.wait_for_training_batch_ready.remote(
                                        buffer_id,
                                        grace_s,
                                    )
                                )
                        if (
                            (self.elastic_trainer_pool_mode or self.trainer_pool_only_mode)
                            and not self.colocated_mode
                            and self._elastic_trainer_pool_training_active
                            and not batch_ready_without_switch
                        ):
                            with log_dual_events(
                                "Leave elastic trainer-pool training window",
                                pivotrl_logger,
                                event_type=EventType.SWITCH,
                            ):
                                self._leave_elastic_trainer_pool_training_window()
                        # will block until the training batch is ready
                        pivotrl_logger.debug("Waiting for training batch with buffer_id %d", buffer_id)
                        with log_dual_events(
                            f"Wait for training batch {buffer_id}",
                            pivotrl_logger,
                            event_type=EventType.WAIT,
                        ):
                            batch = ray.get(self.agent_loop_manager.wait_for_training_batch.remote(buffer_id))
                        pivotrl_logger.debug(
                            "Received training batch for step %d, batch size: %d",
                            self.global_steps,
                            len(batch) if batch is not None else 0,
                        )
                        # Symmetric guard: only (re-)enter the training window when we
                        # are not already in it (e.g. we skipped leave because the batch
                        # was ready). Keeps the SWITCH log accurate and avoids a no-op.
                        if (
                            (self.elastic_trainer_pool_mode or self.trainer_pool_only_mode)
                            and not self.colocated_mode
                            and not self._elastic_trainer_pool_training_active
                        ):
                            with log_dual_events(
                                "Enter elastic trainer-pool training window",
                                pivotrl_logger,
                                event_type=EventType.SWITCH,
                            ):
                                self._enter_elastic_trainer_pool_training_window()
                        # Modes 2 (colocated) & 3 (rollout_rm_colocated): drive phased
                        # sleep/wake. The batch is now ready == rollout-done (requires
                        # launch_reward_fn_async). Switch to REWARD so rm processes the
                        # queued rewards. For mode 2, also wait for rewards to finish
                        # (so we can sleep rm) and wake the actor before training ops.
                        if self.colocated_mode or self.rollout_rm_colocated_mode:
                            with log_dual_events(
                                "Phase -> REWARD (colocated)", pivotrl_logger, event_type=EventType.SWITCH
                            ):
                                ray.get(self.sleep_wake_orchestrator.set_phase.remote("reward"))
                            if self.colocated_mode:
                                _req_ids = batch.non_tensor_batch["uid"].tolist()
                                with log_dual_events(
                                    "Wait for reward (colocated)", pivotrl_logger, event_type=EventType.WAIT
                                ):
                                    # Non-destructive wait: ensure the rm finishes scoring before
                                    # we sleep it (TRAIN phase). The actual pop + post-processing
                                    # of rm_scores happens later in the step via the existing
                                    # wait_for_reward_of_requests call.
                                    ray.get(
                                        self.reward_manager.wait_for_reward_ready.remote(_req_ids)
                                    )
                                with log_dual_events(
                                    "Phase -> TRAIN (colocated)", pivotrl_logger, event_type=EventType.SWITCH
                                ):
                                    ray.get(self.sleep_wake_orchestrator.set_phase.remote("train"))
                                self._wake_trainer_for_elastic_trainer_pool()
                        with log_dual_events("Switch to trainer mode", pivotrl_logger, event_type=EventType.SWITCH):
                            self.switch_to_trainer_mode()
                    else:
                        from verl.trainer.ppo.reward import compute_reward

                        batch = ray.get(self.agent_loop_manager.get_data.remote())
                        if batch is None:
                            pivotrl_logger.info(
                                "No more data from agent loop manager, ending training at step %d",
                                self.global_steps,
                            )
                            break
                        batch_keys_to_pop = [
                            "input_ids",
                            "attention_mask",
                            "position_ids",
                        ]
                        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                        if "multi_modal_data" in batch.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append("multi_modal_data")
                        if "raw_prompt" in batch.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append("raw_prompt")
                        if "tools_kwargs" in batch.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append("tools_kwargs")
                        if "interaction_kwargs" in batch.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                        if "index" in batch.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append("index")
                        if "agent_name" in batch.non_tensor_batch:
                            non_tensor_batch_keys_to_pop.append("agent_name")
                        gen_batch = batch.pop(
                            batch_keys=batch_keys_to_pop,
                            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                        )
                        gen_batch.meta_info["need_pull_model"] = self.global_steps != 1
                        # Verl original colocate method
                        output_batch = self.actor_wg.generate_sequences(gen_batch)
                        batch = batch.union(output_batch)
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        batch.batch["reward"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # Log multi-turn or other metrics
                if "metrics" in batch.meta_info:
                    gen_metrics = batch.meta_info["metrics"]
                    metrics.update(reduce_metrics(gen_metrics))
                    batch.meta_info["metrics"].clear()

                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                # Please take care when you implement group based adv computation such as GRPO and rloo
                # TODO(verl): Decouple the DP balancing and mini-batching.
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                batch.meta_info["temperature"] = self.config.gen_actor_rollout_ref.rollout.temperature

                # Operating Mode Selection:
                # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)

                if bypass_recomputing_logprobs:
                    from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                    apply_rollout_correction(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.train_actor_rollout_ref.actor.policy_loss,
                    )

                    batch.batch.pop("rollout_log_probs")
                else:
                    # recompute log_probs in the training side
                    with marked_timer("recompute_log_prob", timing_raw, color="orange"):
                        with log_dual_events(
                            "Recompute log_prob on training side",
                            pivotrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            recomputed_log_prob = self.actor_wg.compute_log_prob(batch)
                            entropys = recomputed_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.train_actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=loss_agg_mode,
                            )
                            metrics.update({"actor/entropy": entropy_agg.detach().item()})
                            recomputed_log_prob.batch.pop("entropys")
                            batch = batch.union(recomputed_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                recomputed_log_probs = batch.batch["recomputed_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                recomputed_probs = torch.exp(recomputed_log_probs)
                                probs_diff = torch.abs(rollout_probs - recomputed_probs)
                                probs_diff = torch.masked_select(probs_diff, response_mask.bool())
                                probs_diff_max = torch.max(probs_diff)
                                probs_diff_mean = torch.mean(probs_diff)
                                probs_diff_std = torch.std(probs_diff)
                                metrics.update(
                                    {
                                        "training/probs_diff_max": probs_diff_max.detach().item(),
                                        "training/probs_diff_mean": probs_diff_mean.detach().item(),
                                        "training/probs_diff_std": probs_diff_std.detach().item(),
                                    }
                                )
                    batch.batch["old_log_probs"] = batch.batch["recomputed_log_probs"]
                    batch.batch.pop("recomputed_log_probs")

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer("ref", timing_raw, color="olive"):
                        with log_dual_events(
                            "Compute reference log_prob",
                            pivotrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        with log_dual_events(
                            "Compute critic values",
                            pivotrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                # compute reward model score
                # if self.use_rm and "rm_scores" not in batch.batch.keys():
                #     with marked_timer("reward", timing_raw, color="yellow"):
                #         with log_dual_events(
                #             "Compute reward model score",
                #             pivotrl_logger,
                #             event_type=EventType.OTHER,
                #         ):
                #             # compute reward model score
                #             reward_tensor = self.rm_wg.compute_rm_score(batch)
                #             batch = batch.union(reward_tensor)
                # elif self.config.reward_model.launch_reward_fn_async:
                if self.config.reward_models_config.launch_reward_fn_async:
                    # Overlap reward computation with log_prob computation in trainer
                    with marked_timer("async_reward_get", timing_raw, color="yellow"):
                        with log_dual_events(
                            "Wait for async reward model score",
                            pivotrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            request_ids = batch.non_tensor_batch["uid"].tolist()
                            print(f"Waiting for reward of request_ids: {request_ids}")
                            assert self.reward_manager is not None, "Reward manager is not initialized"
                            request_id_to_reward = ray.get(
                                self.reward_manager.wait_for_reward_of_requests.remote(request_ids)
                            )

                        with log_dual_events(
                            "Post process async reward model score",
                            pivotrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            # result structure:
                            # {
                            #     "reward_score": float,
                            #     "reward_extra_info": {
                            #         "original_reward_score": float,
                            #         "data_source": string
                            #         $reward_model_key_0: {
                            #             score: float,
                            #             acc: float,
                            #             ...
                            #         },
                            #         $reward_model_key_1: {...},
                            #         ...
                            #     },
                            # }
                            scores = []
                            reward_extra_infos_dict_list = []
                            reward_metrics_dict_list = []
                            rm_input_token_nums = []
                            rm_generated_token_nums = []
                            for request_id in request_ids:
                                reward_score = request_id_to_reward[request_id]["reward_score"]
                                extra_info = request_id_to_reward[request_id].get("reward_extra_info", {})
                                reward_metrics = request_id_to_reward[request_id].get("reward_metrics", {})
                                scores.append(reward_score)
                                reward_extra_infos_dict_list.append(extra_info)
                                reward_metrics_dict_list.append(reward_metrics)
                                rm_input_token_num, rm_generated_token_num = extract_reward_model_token_counts(
                                    extra_info
                                )
                                rm_input_token_nums.append(rm_input_token_num)
                                rm_generated_token_nums.append(rm_generated_token_num)
                            prompt_length = batch.batch["prompts"].size(1)
                            response_length = batch.batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1
                            rm_scores = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                            rm_scores[
                                torch.arange(batch.batch["response_mask"].size(0)),
                                response_length,
                            ] = torch.tensor(scores, dtype=torch.float32)
                            reward_tensor = rm_scores  # [bsz, response_length]

                            # add reward_extra_info to non_tensor_batch
                            reward_extra_infos_dict = defaultdict(list)
                            for reward_extra_infos in reward_extra_infos_dict_list:
                                for key, value in reward_extra_infos.items():
                                    # if not isinstance(value, list):
                                    #     value = [value]
                                    # reward_extra_infos_dict[key].extend(value)
                                    # if key == "score" or key == "acc" or key == "data_source":
                                    #     if not isinstance(value, list):
                                    #         value = [value]
                                    #         reward_extra_infos_dict[key].extend(value)
                                    if key in ("data_source", "original_reward_score"):
                                        if not isinstance(value, list):
                                            value = [value]
                                        reward_extra_infos_dict[key].extend(value)
                                reward_extra_infos_dict["reward_extra_info"].append(reward_extra_infos)
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            batch.non_tensor_batch["rm_input_token_num"] = np.array(
                                rm_input_token_nums, dtype=np.int64
                            )
                            batch.non_tensor_batch["rm_generated_token_num"] = np.array(
                                rm_generated_token_nums, dtype=np.int64
                            )
                            batch.meta_info["reward_metrics"] = np.array(reward_metrics_dict_list, dtype=object)
                            teacher_logprobs, teacher_logprob_metrics = self._merge_teacher_tensor_from_rewards(
                                request_id_to_reward=request_id_to_reward,
                                request_ids=request_ids,
                                response_mask=batch.batch["response_mask"],
                                field="teacher_logprobs",
                                dtype=torch.float32,
                            )
                            if teacher_logprobs is not None:
                                batch.batch["teacher_logprobs"] = teacher_logprobs
                                valid_teacher_logprobs = teacher_logprobs[
                                    batch.batch["response_mask"].bool()
                                ]
                                if valid_teacher_logprobs.numel() > 0:
                                    teacher_logprob_metrics["distillation/teacher_logprobs_mean"] = (
                                        valid_teacher_logprobs.float().mean().item()
                                    )
                            metrics.update(teacher_logprob_metrics)
                            teacher_ids, teacher_id_metrics = self._merge_teacher_tensor_from_rewards(
                                request_id_to_reward=request_id_to_reward,
                                request_ids=request_ids,
                                response_mask=batch.batch["response_mask"],
                                field="teacher_ids",
                                dtype=torch.long,
                            )
                            if teacher_ids is not None:
                                batch.batch["teacher_ids"] = teacher_ids
                            metrics.update(teacher_id_metrics)
                else:
                    reward_tensor = batch.batch.pop("rm_scores", None)
                    reward_tensor = self._normalize_sync_reward_tensor(batch, reward_tensor)

                batch.batch["token_level_scores"] = reward_tensor

                # record_rollout_rm_metrics(batch, output_path="logs/test_metrics.jsonl")
                with marked_timer("adv", timing_raw, color="brown"):
                    with log_dual_events("Compute advantage", pivotrl_logger, event_type=EventType.OTHER):
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import (
                                compute_rollout_correction_and_add_to_batch,
                            )

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        log_data_protocol(
                            batch,
                            pivotrl_logger,
                            self.log_prefix + " before compute advantage",
                            level=logging.DEBUG,
                        )
                        batch = PivotRL_compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.gen_actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        with log_dual_events("Update critic", pivotrl_logger, event_type=EventType.TRAIN):
                            critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        with log_dual_events("Update actor", pivotrl_logger, event_type=EventType.TRAIN):
                            batch.meta_info["multi_turn"] = self.config.gen_actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_wg.update_actor(batch)
                    # pivotrl_logger.info(
                    #     f"Update actor ppo_kl: {actor_output.meta_info['metrics']['actor/ppo_kl']}, "
                    #     f"len: {len(actor_output.meta_info['metrics']['actor/ppo_kl'])}"
                    # )
                    # pivotrl_logger.info(
                    #     f"Update actor pg_loss: {actor_output.meta_info['metrics']['actor/pg_loss']}, "
                    #     f"len: {len(actor_output.meta_info['metrics']['actor/pg_loss'])}"
                    # )
                    # pivotrl_logger.info(
                    #     f"Update actor grad_norm: {actor_output.meta_info['metrics']['actor/grad_norm']}, "
                    #     f"len: {len(actor_output.meta_info['metrics']['actor/grad_norm'])}"
                    # )
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Modes 2 (colocated) & 3 (rollout_rm_colocated): end-of-step phase
                # transition back to ROLLOUT for the next buffer. For mode 2 the actor
                # is NIXL-slept so its GPUs lend to rollout; rollout wakes and pulls
                # the freshly updated weights via sync_with_ps. Mode 3 actor stays
                # awake on its separate train_pool.
                if self.colocated_mode:
                    self._sleep_trainer_for_elastic_trainer_pool()
                    with log_dual_events(
                        "Phase -> ROLLOUT (colocated)", pivotrl_logger, event_type=EventType.SWITCH
                    ):
                        ray.get(self.sleep_wake_orchestrator.set_phase.remote("rollout"))
                elif self.rollout_rm_colocated_mode:
                    with log_dual_events(
                        "Phase -> ROLLOUT (rollout_rm_colocated)", pivotrl_logger, event_type=EventType.SWITCH
                    ):
                        ray.get(self.sleep_wake_orchestrator.set_phase.remote("rollout"))

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                        with log_dual_events(
                            "Dump rollout generations",
                            pivotrl_logger,
                            event_type=EventType.OTHER,
                        ):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                should_validate = (
                    self.val_reward_manager is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                )
                should_save = self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                )
                save_before_validation = (
                    should_save
                    and should_validate
                    and self.elastic_trainer_pool_mode
                    and self.config.pivotrl.colocate_validate_and_train
                )

                # Save while the actor is still awake; elastic validation returns
                # train_pool directly to rollout and reward-model replicas.
                if save_before_validation:
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        with log_dual_events("Save checkpoint", pivotrl_logger, event_type=EventType.OTHER):
                            self._save_checkpoint()

                # validate
                if should_validate:
                    with marked_timer("testing", timing_raw, color="green"):
                        with log_dual_events("Validate", pivotrl_logger, event_type=EventType.VAL):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if should_save and not save_before_validation:
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        with log_dual_events("Save checkpoint", pivotrl_logger, event_type=EventType.OTHER):
                            self._save_checkpoint()

            with marked_timer("stop_profile", timing_raw):
                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                self._stop_profiling(
                    curr_step_profile and not next_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            # training metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                }
            )
            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO(verl): implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            # Live elastic_rm awake-instance counts per role (Rollout / RewardModel).
            metrics.update(self._collect_elastic_awake_metrics())

            # TODO(verl): make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if (
                hasattr(self.config.train_actor_rollout_ref.actor, "profiler")
                and self.config.train_actor_rollout_ref.actor.profiler.tool == "torch_memory"
            ):
                self.actor_wg.dump_memory_snapshot(
                    tag=f"post_update_step{self.global_steps}",
                    sub_dir=f"step{self.global_steps}",
                )

            if is_last_step:
                pivotrl_logger.info(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                break

        # Stop all components
        pivotrl_logger.info("Stopping all data pipeline components...")
        self.stop_reward_manager()
        self.stop_agent_loop_manager()
        if self.elastic_executor is not None:
            ray.get(self.elastic_executor.stop.remote())
            self.elastic_executor = None
        self.shutdown_reward_model_managers()
        self.stop_rollout_coordinator()
        self.stop_data_processor()
        self.stop_rollout_gateway()

        pivotrl_logger.info("Training completed successfully!")
