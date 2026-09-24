import logging
import os
import random
import socket

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from verl.trainer.ppo.reward import load_reward_manager

from pivotrl.trainer.constants_ppo import get_ppo_ray_runtime_env
from pivotrl.trainer.ppo.utils import PivotRL_Role
from pivotrl.utils.deployment_mode import (
    expand_ngpus_per_node,
    resolve_deployment_mode,
    validate_trainer_sleep_optimizer_offload,
)
from pivotrl.utils.post_processor import (
    load_buffer_post_processor,
    load_group_post_processor,
)
from pivotrl.utils.ray_storage import inspect_local_plasma_backing, plasma_backing_error

pivotrl_logger = logging.getLogger(__file__)
pivotrl_logger.setLevel(os.getenv("PIVOTRL_LOGGING_LEVEL", "WARN"))


def seed_everything(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


@ray.remote(num_gpus=1, num_cpus=0)
class _GpuSlotReserver:
    """Lightweight Ray actor that holds a GPU slot without touching CUDA.

    Used to prevent Ray from scheduling job workers onto nodes that the job
    does not intend to use, which would otherwise cause multiple validate
    instances to land on the same GPU when the cluster has excess capacity.
    """

    def ping(self) -> str:
        return "ok"


@ray.remote(num_cpus=0)
def _inspect_plasma_backing_on_node() -> dict:
    node_id = ray.get_runtime_context().get_node_id()
    return inspect_local_plasma_backing(node_id)


def _validate_cluster_plasma_backing(config) -> None:
    """Fail before model startup when Plasma's mmap filesystem is overcommitted."""
    alive_gpu_nodes = sorted(
        [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0],
        key=lambda node: node["NodeID"],
    )
    total_nnodes = config.pivotrl.deployment.get("total_nnodes", None)
    if total_nnodes is not None:
        alive_gpu_nodes = alive_gpu_nodes[: int(total_nnodes)]
    refs = [
        _inspect_plasma_backing_on_node.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        ).remote()
        for node in alive_gpu_nodes
    ]
    snapshots = ray.get(refs, timeout=60)
    reserve_bytes = int(os.environ.get("PIVOTRL_PLASMA_BACKING_RESERVE_BYTES", 16 * 1024**3))
    if reserve_bytes < 0:
        raise ValueError("PIVOTRL_PLASMA_BACKING_RESERVE_BYTES must be non-negative.")
    errors = [
        error
        for snapshot in snapshots
        if (error := plasma_backing_error(snapshot, reserve_bytes)) is not None
    ]
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeError(
            "Unsafe Ray Plasma backing configuration detected before model startup:\n"
            f"{details}\n"
            "Reduce Ray object-store size or clear stale files from its backing filesystem "
            "until the reported reserve is available."
        )


def _reserve_excess_nodes(config) -> list:
    """Block GPU slots on nodes that are not needed by this job.

    Reads pivotrl.deployment.total_nnodes from config. If null/None, does
    nothing. Otherwise, compares against the live cluster node count and
    creates one _GpuSlotReserver actor per GPU on every excess node to
    prevent Ray from scheduling job workers there.

    Args:
        config: Hydra config with pivotrl.deployment fields.

    Returns:
        list: The list of _GpuSlotReserver actor handles (kept alive by caller).
    """
    total_nnodes = config.pivotrl.deployment.get("total_nnodes", None)
    if total_nnodes is None:
        return []
    total_nnodes = int(total_nnodes)

    alive_gpu_nodes = sorted(
        [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0],
        key=lambda n: n["NodeID"],
    )
    cluster_nnodes = len(alive_gpu_nodes)

    if cluster_nnodes <= total_nnodes:
        pivotrl_logger.info(f"Cluster has {cluster_nnodes} GPU nodes, job needs {total_nnodes}; no reservation needed.")
        return []

    excess_nodes = alive_gpu_nodes[total_nnodes:]
    reservers = []
    for node in excess_nodes:
        node_id = node["NodeID"]
        n_gpus = int(node["Resources"]["GPU"])
        pivotrl_logger.info(
            f"Reserving {n_gpus} GPU slot(s) on excess node {node['NodeManagerAddress']} "
            f"(node_id={node_id}) to prevent stray worker placement."
        )
        for _ in range(n_gpus):
            actor = _GpuSlotReserver.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote()
            reservers.append(actor)

    # Verify actors are ready before proceeding so placement is confirmed.
    ray.get([r.ping.remote() for r in reservers])
    print(
        f"[PivotRL] Reserved {len(reservers)} GPU slot(s) across "
        f"{len(excess_nodes)} excess node(s) "
        f"(cluster={cluster_nnodes}, job needs={total_nnodes})."
    )
    return reservers


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        default_runtime_env["env_vars"]["PIVOTRL_LOGGING_PATH"] = config.pivotrl.logging_path
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    _validate_cluster_plasma_backing(config)

    # NOTE(claude): keep the handle list alive for the entire job lifetime so Ray
    # does not garbage-collect the reservation actors before the job finishes.
    _slot_reservers = _reserve_excess_nodes(config)

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        from pivotrl.workers.gen.gen_worker import PivotRL_GenWorker

        if config.train_actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in [
                "fsdp",
                "fsdp2",
            ], "Critic strategy must be the same as actor strategy: 'fsdp' or 'fsdp2'."
            from verl.workers.fsdp_workers import ActorRolloutRefWorker

            from pivotrl.workers.train.fsdp_train_worker import (
                PivotRL_FSDPTrainWorker as PivotRL_TrainWorker,
            )

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.train_actor_rollout_ref.actor.strategy == "megatron":
            assert config.train_actor_rollout_ref.actor.strategy == config.critic.strategy, (
                "Critic strategy must be the same as actor strategy: 'megatron'."
            )
            from verl.workers.megatron_workers import ActorRolloutRefWorker

            from pivotrl.workers.train.megatron_train_worker import (
                PivotRL_MegatronTrainWorker as PivotRL_TrainWorker,
            )

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        self.role_worker_mapping[PivotRL_Role.Rollout] = ray.remote(PivotRL_GenWorker)
        self.role_worker_mapping[PivotRL_Role.Actor] = ray.remote(PivotRL_TrainWorker)
        if config.pivotrl.colocate_validate_and_train:
            self.role_worker_mapping[PivotRL_Role.Validate] = ray.remote(PivotRL_GenWorker)

        return actor_rollout_cls, ray_worker_group_cls

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        deployment_config = config.pivotrl.deployment
        train_pool_id = "train_pool"
        train_bundle_resource_num = 0.9 if config.pivotrl.colocate_validate_and_train else 1.0
        resource_pool_spec = {
            train_pool_id: [deployment_config.train_ngpus_per_node] * deployment_config.train_nnodes,
        }
        # Validation resource pool share with training pool by default.
        # But the granularity of validation is per DP worker, while training is the whole training job.
        # Thus we set different resource fraction to enable resource sharing between training and validation.
        # The training pool gets higher fraction due to initialization order.
        # Note that 50% is not safe because two bundle in one pool may share one GPU, which causes error.
        resource_num_per_bundle = {
            train_pool_id: train_bundle_resource_num,
        }

        self.mapping[PivotRL_Role.Actor] = [train_pool_id]
        self.mapping[PivotRL_Role.Critic] = [train_pool_id]

        # Set the resource pool spec for each rollout instance.
        total_rollout_gpus = 0
        if deployment_config.elastic_rm.enable:
            rollout_pool_id_list = ["shared_rollout_pool"]
        else:
            rollout_pool_id_list = [f"rollout_pool_{i}" for i in range(deployment_config.n_rollout_instances)]

        # If heterogeneous rollout is enabled, we will use the heterogeneous rollout configuration.
        if deployment_config.heterogeneous_rollout.enable:
            if deployment_config.elastic_rm.enable:
                raise ValueError("heterogeneous_rollout is not supported when elastic_rm.enable is True.")
            heterogeneous_deployment_config = deployment_config.heterogeneous_rollout
            assert (
                len(heterogeneous_deployment_config.rollout_nnodes_per_instance)
                == heterogeneous_deployment_config.n_rollout_instances
            ), "The number of rollout nnodes per instance must match the number of rollout instances."
            assert (
                len(heterogeneous_deployment_config.rollout_ngpus_per_node_per_instance)
                == heterogeneous_deployment_config.n_rollout_instances
            ), "The number of rollout ngpus per node per instance must match the number of rollout instances."
            assert (
                len(heterogeneous_deployment_config.tensor_model_parallel_size_per_instance)
                == heterogeneous_deployment_config.n_rollout_instances
            ), "The number of tensor model parallel size per instance must match the number of rollout instances."
            assert (
                len(heterogeneous_deployment_config.pipeline_model_parallel_size_per_instance)
                == heterogeneous_deployment_config.n_rollout_instances
            ), "The number of pipeline model parallel size per instance must match the number of rollout instances."

            for i in range(deployment_config.n_rollout_instances):
                rollout_pool_id = rollout_pool_id_list[i]
                resource_pool_spec[rollout_pool_id] = [
                    heterogeneous_deployment_config.rollout_ngpus_per_node_per_instance[i]
                ] * heterogeneous_deployment_config.rollout_nnodes_per_instance[i]
        elif deployment_config.elastic_rm.enable:
            enable_trainer_pool = bool(deployment_config.elastic_rm.get("enable_trainer_pool", False))
            if enable_trainer_pool and config.pivotrl.ps_mode not in ("nixl_cpu", "nixl_gpu"):
                raise ValueError(
                    "elastic_rm.enable_trainer_pool requires pivotrl.ps_mode to be 'nixl_cpu' or 'nixl_gpu'."
                )
            deployment_mode = deployment_config.get("mode", None)
            colocated_mode = deployment_mode == "colocated"
            if colocated_mode:
                # Mode 2: all three roles share train_pool (colocated), time-multiplexed.
                # No separate shared_rollout_pool; all rollout/rm replicas live on
                # train_pool alongside the actor.
                shared_rollout_gpus = 0
                shared_rollout_gpu_spec = []
                trainer_elastic_gpus = (
                    deployment_config.train_ngpus_per_node * deployment_config.train_nnodes
                )
            else:
                shared_rollout_gpu_spec = expand_ngpus_per_node(
                    deployment_config.elastic_rm.shared_ngpus_per_node,
                    deployment_config.elastic_rm.shared_nnodes,
                    "pivotrl.deployment.elastic_rm.shared_ngpus_per_node",
                )
                shared_rollout_gpus = sum(shared_rollout_gpu_spec)
                trainer_elastic_gpus = (
                    deployment_config.train_ngpus_per_node * deployment_config.train_nnodes
                    if enable_trainer_pool
                    else 0
                )
            rollout_instance_world_size = (
                config.gen_actor_rollout_ref.rollout.tensor_model_parallel_size
                * config.gen_actor_rollout_ref.rollout.pipeline_model_parallel_size
                * config.gen_actor_rollout_ref.rollout.data_parallel_size
            )
            shared_rollout_instances = shared_rollout_gpus // rollout_instance_world_size
            trainer_rollout_instances = trainer_elastic_gpus // rollout_instance_world_size
            total_rollout_gpus = shared_rollout_gpus + trainer_elastic_gpus
            deployment_config.n_rollout_instances = shared_rollout_instances + trainer_rollout_instances
            print(
                "[Elastic RM] Maximum number of rollout instances = "
                f"{deployment_config.n_rollout_instances} "
                f"(shared={shared_rollout_instances}, trainer_pool={trainer_rollout_instances}, "
                f"mode={deployment_mode})"
            )
            if colocated_mode:
                # No shared_rollout_pool; rollout/rm replicas are all on train_pool.
                rollout_pool_id_list = [train_pool_id] * trainer_rollout_instances
            else:
                resource_pool_spec["shared_rollout_pool"] = shared_rollout_gpu_spec
                rollout_pool_id_list = (
                    ["shared_rollout_pool"] * shared_rollout_instances
                    + [train_pool_id] * trainer_rollout_instances
                )
        else:
            for i in range(deployment_config.n_rollout_instances):
                rollout_pool_id = rollout_pool_id_list[i]
                resource_pool_spec[rollout_pool_id] = [
                    deployment_config.rollout_ngpus_per_node_per_instance
                ] * deployment_config.rollout_nnodes_per_instance

            # Mode 4 (trainer_pool_only): append a fixed number of extra rollout
            # replicas on train_pool. train_pool already exists in resource_pool_spec
            # (created above for the actor); do NOT overwrite its spec. These replicas
            # are time-multiplexed with the actor via NIXL sleep/wake.
            if deployment_config.get("mode", None) == "trainer_pool_only":
                idle_rollout = int(deployment_config.get("trainer_pool_idle_rollout_instances", 0))
                if idle_rollout < 0:
                    raise ValueError("trainer_pool_idle_rollout_instances must be >= 0.")
                rollout_pool_id_list = rollout_pool_id_list + [train_pool_id] * idle_rollout
                deployment_config.n_rollout_instances = len(rollout_pool_id_list)
                print(
                    f"[Trainer-pool-only] rollout instances = {deployment_config.n_rollout_instances} "
                    f"(main={deployment_config.n_rollout_instances - idle_rollout}, "
                    f"train_pool_idle={idle_rollout})"
                )

        # Set the resource pool spec for each validation instance.
        if config.pivotrl.colocate_validate_and_train:
            val_pool_id_list = [f"validate_pool_{i}" for i in range(deployment_config.n_validate_instances)]
            for i in range(deployment_config.n_validate_instances):
                validate_pool_id = val_pool_id_list[i]
                resource_pool_spec[validate_pool_id] = [
                    deployment_config.validate_ngpus_per_node_per_instance
                ] * deployment_config.validate_nnodes_per_instance
                resource_num_per_bundle[validate_pool_id] = 1.0 - train_bundle_resource_num
            self.mapping[PivotRL_Role.Validate] = val_pool_id_list

        self.mapping[PivotRL_Role.Rollout] = rollout_pool_id_list

        # Reward model resource pool
        total_reward_pool_id_list = []
        reward_models_config = config.reward_models_config
        for reward_model in reward_models_config.reward_models:
            if reward_model.reward_loop_type not in ("gen", "opd"):
                continue
            if deployment_config.elastic_rm.enable:
                reward_model_world_size = (
                    reward_model.rollout.tensor_model_parallel_size
                    * reward_model.rollout.pipeline_model_parallel_size
                    * reward_model.rollout.data_parallel_size
                )
                shared_reward_instances = shared_rollout_gpus // reward_model_world_size
                trainer_reward_instances = trainer_elastic_gpus // reward_model_world_size
                reward_model.num_replicas = shared_reward_instances + trainer_reward_instances
                total_reward_pool_id_list.extend(
                    ["shared_rollout_pool"] * shared_reward_instances
                    + [train_pool_id] * trainer_reward_instances
                )
                print(
                    f"[Elastic RM] Maximum number of reward model({reward_model.reward_model_name}) "
                    f"instances = {reward_model.num_replicas} "
                    f"(shared={shared_reward_instances}, trainer_pool={trainer_reward_instances})"
                )
            elif reward_model.enable_resource_pool:
                if reward_model.n_gpus_per_node <= 0:
                    raise ValueError("reward_model.n_gpus_per_node must be greater than 0")
                if reward_model.nnodes <= 0:
                    raise ValueError("reward_model.nnodes must be greater than 0")

                reward_model_instances = reward_model.get("num_replicas", 1)
                reward_model_name = reward_model.get("reward_model_name", reward_model.model.path.split("/")[-1])
                reward_pool_id_list = [
                    f"reward_pool_{reward_model_name}_{i}" for i in range(reward_model_instances)
                ]
                for i in range(reward_model_instances):
                    resource_pool_spec[reward_pool_id_list[i]] = [
                        reward_model.rollout_ngpus_per_instance_per_node
                    ] * reward_model.rollout_nnodes_per_instance
                total_reward_pool_id_list.extend(reward_pool_id_list)
                # Mode 4: append fixed extra RM replicas on train_pool (time-multiplexed
                # with the actor). train_pool spec already exists; do not overwrite.
                if deployment_config.get("mode", None) == "trainer_pool_only":
                    idle_rm = int(deployment_config.get("trainer_pool_idle_rm_instances", 0))
                    if idle_rm < 0:
                        raise ValueError("trainer_pool_idle_rm_instances must be >= 0.")
                    reward_model.num_replicas = reward_model_instances + idle_rm
                    total_reward_pool_id_list.extend([train_pool_id] * idle_rm)
                    print(
                        f"[Trainer-pool-only] reward model({reward_model_name}) replicas = "
                        f"{reward_model.num_replicas} (main={reward_model_instances}, "
                        f"train_pool_idle={idle_rm})"
                    )
            else:
                raise ValueError("reward_model.enable_resource_pool must be True when elastic_rm.enable is False")

        self.mapping[PivotRL_Role.RewardModel] = total_reward_pool_id_list

        from pivotrl.trainer.ppo.utils import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
            resource_num_per_bundle=resource_num_per_bundle,
        )

        print(f"resource_pool_spec = {resource_pool_spec}, mapping = {self.mapping}")

        return resource_pool_manager

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import CriticWorker
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError

        self.role_worker_mapping[PivotRL_Role.Critic] = ray.remote(CriticWorker)

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from pivotrl.workers.gen.gen_worker import PivotRL_GenWorker
        # self.role_worker_mapping[PivotRL_Role.RewardModel] = ray.remote(PivotRL_RewardModelWorker)
        # # concurrency_groups must be set on the actor class (not .options() in verl); use
        # # ray.remote(**opts)(Cls) — cannot pass Cls and kwargs in one ray.remote call.
        self.role_worker_mapping[PivotRL_Role.RewardModel] = ray.remote(
            max_concurrency=10000,
        )(PivotRL_GenWorker)

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        if config.algorithm.use_kl_in_reward or config.train_actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[PivotRL_Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[PivotRL_Role.RefPolicy] = ["train_pool"]

    def add_dummy_worker(self, config):
        from pivotrl.trainer.ppo.utils import PivotRL_DummyWorker

        self.role_worker_mapping[PivotRL_Role.DummyPolicy] = ray.remote(PivotRL_DummyWorker)
        self.mapping[PivotRL_Role.DummyPolicy] = ["train_pool"]

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # Resolve the unified deployment.mode into concrete elastic_rm flags BEFORE
        # laying out resource pools / adding workers, so init_resource_pool_mgr and
        # the trainer both observe a consistent flag set (modes 2/3 force
        # elastic_rm.enable=True even though the user only set deployment.mode).
        resolved_mode = resolve_deployment_mode(config)
        validate_trainer_sleep_optimizer_offload(config, resolved_mode)
        print(f"[Deployment] resolved mode = {resolved_mode}")

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # NOTE(linsh): add a dummy worker to actor/critic/ref actors to avoid detected as async actor in Ray
        self.add_dummy_worker(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.train_actor_rollout_ref.model.path,
            use_shm=config.train_actor_rollout_ref.model.get("use_shm", False),
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        # NOTE(linsh): lazily import `PivotRL_RayPPOTrainer` here to avoid implicit ray.init()
        # during the initialization of `GLOBAL_PORT_SCANNER` in nixl.`
        from verl.utils.dataset.rl_dataset import collate_fn

        from pivotrl.trainer.ppo.ray_trainer import PivotRL_RayPPOTrainer

        # Load post-processor from configuration
        group_post_process_fn = load_group_post_processor(config)
        buffer_post_process_fn = load_buffer_post_processor(config)

        # Initialize the PPO trainer.
        trainer = PivotRL_RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            collate_fn=collate_fn,
            group_post_process_fn=group_post_process_fn,
            buffer_post_process_fn=buffer_post_process_fn,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


if __name__ == "__main__":
    seed_everything(0)
    main()
