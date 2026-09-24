import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf
from psrl.workers.gen.gen_worker import GenInterface, PSRL_GenWorker
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)
from tensordict import TensorDict
from transformers import AutoTokenizer
from verl import DataProto

# Production elastic RM defaults (rollout_qwen_7b_rm_qwen_8b_fsdp.sh).
_PROD_PROMPT_LENGTH = 1024 * 11
_PROD_RESPONSE_LENGTH = 1024 * 10
_PROD_MAX_MODEL_LEN = _PROD_PROMPT_LENGTH + _PROD_RESPONSE_LENGTH
# Rollout-aligned context (rollout_qwen_7b script: 1024 prompt + 10240 response).
_ROLLOUT_ALIGNED_MAX_MODEL_LEN = 1024 + 10240


@ray.remote(num_cpus=0)
def _gpu_memory_snapshot(stage: str) -> dict[str, str]:
    def run_nvidia_smi(args: list[str]) -> str:
        try:
            result = subprocess.run(
                args,
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception as exc:
            return f"<failed to run {' '.join(args)}: {exc}>"
        output = result.stdout.strip()
        if result.stderr.strip():
            output = f"{output}\nstderr: {result.stderr.strip()}".strip()
        if result.returncode != 0:
            output = f"{output}\nreturncode: {result.returncode}".strip()
        return output or "<empty>"

    return {
        "stage": stage,
        "node_id": ray.get_runtime_context().get_node_id(),
        "node_ip": ray.util.get_node_ip_address(),
        "hostname": socket.gethostname(),
        "gpu_memory": run_nvidia_smi(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "compute_apps": run_nvidia_smi(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


@ray.remote(num_cpus=0)
def _probe_node_identity() -> dict[str, str]:
    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "node_ip": ray.util.get_node_ip_address(),
    }


@ray.remote(num_cpus=0, num_gpus=1)
def _probe_bundle_gpu_identity() -> dict[str, object]:
    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "node_ip": ray.util.get_node_ip_address(),
        "gpu_ids": [str(gpu_id) for gpu_id in ray.get_gpu_ids()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


@ray.remote(num_cpus=0)
class _NIXLProbeMetaServer:
    def __init__(self, nixl_config, expected_agents: int):
        from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
        from psrl.utils.nixl import NIXLMetaServer

        self.server = NIXLMetaServer(NIXL_META_SERVER_NAME, nixl_config)
        self.expected_agents = expected_agents

    def ready(self) -> bool:
        return True

    def protocol(self, timeout_s: int) -> None:
        self.server.wait_for_client_shardings(self.expected_agents, timeout=timeout_s)
        self.server.make_unified_sharding()
        self.server.notify_all_client_shardings()
        self.server.wait_for_client_infos(self.expected_agents, timeout=timeout_s)
        self.server.make_comm_plan()
        self.server.notify_all_client_infos_and_comm_plan()
        self.server.wait_for_client_temp_mappings(self.expected_agents, timeout=timeout_s)
        self.server.notify_all_client_temp_mappings()

    def shutdown(self) -> None:
        self.server.shutdown()


@ray.remote(num_cpus=0)
class _NIXLProbePSControl:
    def __init__(self):
        self.agent_names = None
        self.gen_storage_client_names = None
        self.version = 0
        self.instance_versions = {}
        self.shared_pull_count = 0

    def set_endpoints(self, agent_names: list[str], gen_storage_client_names: list[str]) -> None:
        self.agent_names = agent_names
        self.gen_storage_client_names = gen_storage_client_names

    def get_ps_nixl_agent_names(self) -> list[str]:
        assert self.agent_names is not None
        return self.agent_names

    def get_ps_nixl_gen_storage_client_names(self) -> list[str]:
        assert self.gen_storage_client_names is not None
        return self.gen_storage_client_names

    def pull_model_state_dict_nixl(self, rollout_instance_id: int) -> int:
        self.instance_versions[rollout_instance_id] = self.version
        return self.version

    def get_rollout_instance_model_version(self, rollout_instance_id: int) -> int:
        return self.instance_versions.get(rollout_instance_id, self.version)

    def update_request_status(self, request_ids: list[int] | int, *_args, **_kwargs) -> list[bool] | bool:
        if isinstance(request_ids, list):
            return [True] * len(request_ids)
        return True

    def update_request_version_tag(self, *_args, **_kwargs) -> None:
        return None

    def _try_acquire_shared_pull_lock(self) -> bool:
        self.shared_pull_count += 1
        return True

    def _release_shared_pull_lock(self) -> None:
        assert self.shared_pull_count > 0
        self.shared_pull_count -= 1


def _print_cluster_gpu_memory(stage: str) -> list[dict[str, str]]:
    snapshots = _collect_gpu_snapshots(stage)
    print(f"\n========== GPU memory snapshot: {stage} ==========", flush=True)
    for snapshot in sorted(snapshots, key=lambda item: item["node_ip"]):
        print(
            f"[node {snapshot['node_ip']} host={snapshot['hostname']} id={snapshot['node_id']}]",
            flush=True,
        )
        print("GPU memory (index, name, used MiB, free MiB, total MiB):", flush=True)
        print(snapshot["gpu_memory"], flush=True)
        print("Compute apps (gpu_uuid, pid, process_name, used MiB):", flush=True)
        print(snapshot["compute_apps"], flush=True)
    print("====================================================\n", flush=True)
    return snapshots


def _free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return str(sock.getsockname()[1])


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_tms_weight_arena_config(probe_role: str):
    reward_materialization = os.environ.get(
        "PSRL_TMS_VLLM_REWARD_MATERIALIZATION", "direct"
    )
    if reward_materialization not in {"direct", "repack"}:
        raise ValueError(
            "PSRL_TMS_VLLM_REWARD_MATERIALIZATION must be direct or repack, "
            f"got {reward_materialization!r}"
        )
    return OmegaConf.create(
        {
            "actor_enabled": False,
            "rollout_enabled": probe_role == "rollout",
            "reward_enabled": probe_role == "reward",
            "rollout_materialization": "direct",
            "reward_materialization": reward_materialization,
            "reward_cpu_cache_mode": os.environ.get(
                "PSRL_TMS_VLLM_REWARD_CPU_CACHE_MODE", "node_shared"
            ),
            "reward_node_cache_dir": os.environ.get(
                "PSRL_TMS_VLLM_REWARD_NODE_CACHE_DIR",
                "/dev/shm/psrl-rm-weight-cache",
            ),
            "reward_node_cache_wait_timeout_s": float(
                os.environ.get(
                    "PSRL_TMS_VLLM_REWARD_NODE_CACHE_WAIT_TIMEOUT_S", "1800"
                )
            ),
            "reward_cpu_cache_pin_memory": _env_bool(
                "PSRL_TMS_VLLM_REWARD_CPU_CACHE_PIN_MEMORY", False
            ),
            "max_chunk_gb": float(
                os.environ.get("PSRL_TMS_VLLM_WEIGHT_ARENA_MAX_CHUNK_GB", "4")
            ),
            "alignment_bytes": int(
                os.environ.get(
                    "PSRL_TMS_VLLM_WEIGHT_ARENA_ALIGNMENT_BYTES", "256"
                )
            ),
        }
    )


def _parse_gpu0_used_mib(snapshots: list[dict[str, str]]) -> int | None:
    """Parse GPU0 used MiB from nvidia-smi snapshot on the busiest node."""
    best: int | None = None
    for snapshot in snapshots:
        for line in snapshot.get("gpu_memory", "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or parts[0] != "0":
                continue
            try:
                used = int(parts[2])
            except ValueError:
                continue
            if best is None or used > best:
                best = used
    return best


def _collect_gpu_snapshots(stage: str) -> list[dict[str, str]]:
    refs = []
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        refs.append(
            _gpu_memory_snapshot.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,
                )
            ).remote(stage)
        )
    return ray.get(refs, timeout=60)


def _default_model_path() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    local_model = repo_root / "models" / "Qwen" / "Qwen2.5-0.5B"
    return os.environ.get("PSRL_RM_SMOKE_MODEL", str(local_model))


def _build_psrl_config(tmp_path: Path, *, load_real_weights_at_init: bool):
    # Without an arena, elastic_rm.enable gates whether init_model loads GPU weights
    # after CPU preload. Arena-backed reward workers always construct the final CPU
    # mirror during init so the first wake is not a checkpoint-layout reload.
    elastic_rm_enable = _env_bool("PSRL_RM_SMOKE_ELASTIC_RM", True)
    if load_real_weights_at_init:
        elastic_rm_enable = False
    return OmegaConf.create(
        {
            "logging_path": str(tmp_path),
            "ps_mode": "cpu",
            "rollout_n": 1,
            "staleness_buffer_entries": 8,
            "redundant_rollout": {
                "enable": False,
                "redundant_global_batch_size": 1,
                "redundant_rollout_n": 1,
            },
            "deployment": {
                "n_rollout_instances": 1,
                "elastic_rm": {"enable": elastic_rm_enable},
            },
            "tms": {
                "range": "all",
                "enable_nixl": False,
                "enable_cuda_graph": _env_bool("PSRL_RM_SMOKE_ENABLE_CUDA_GRAPH", False),
            },
            "server_rollout": {"enable": False},
            "status_collection": {"enable": False},
            "routing_strategy": {
                "max_num_waiting_reqs_after_preemption": 0,
                "max_estimated_concurrent_seqs_per_instance": 1,
            },
            "log_prob": {"enable_rollout_engine_log_prob": False},
            "partial_rollout": {"interrupt_as_prompt": False},
            "profile": {"fix_weight": False},
            "nixl": {},
        }
    )


def _build_reward_config(model_path: str):
    # Fast profile for small models / local iteration (PSRL_RM_SMOKE_FAST=1).
    if _env_bool("PSRL_RM_SMOKE_FAST", False):
        prompt_length = 32
        response_length = int(os.environ.get("PSRL_RM_SMOKE_MAX_TOKENS", "4"))
        max_model_len = int(os.environ.get("PSRL_RM_SMOKE_MAX_MODEL_LEN", "64"))
        max_num_batched_tokens = max_model_len
        max_num_seqs = 1
        gpu_memory_utilization = float(os.environ.get("PSRL_RM_SMOKE_GPU_MEMORY_UTILIZATION", "0.4"))
        disable_kv_cache = True
        enforce_eager = _env_bool("PSRL_RM_SMOKE_ENFORCE_EAGER", True)
    else:
        prompt_length = int(os.environ.get("PSRL_RM_SMOKE_PROMPT_LENGTH", str(_PROD_PROMPT_LENGTH)))
        response_length = int(os.environ.get("PSRL_RM_SMOKE_RESPONSE_LENGTH", str(_PROD_RESPONSE_LENGTH)))
        max_model_len = int(os.environ.get("PSRL_RM_SMOKE_MAX_MODEL_LEN", str(_PROD_MAX_MODEL_LEN)))
        max_num_batched_tokens = int(
            os.environ.get("PSRL_RM_SMOKE_MAX_NUM_BATCHED_TOKENS", str(max_model_len))
        )
        max_num_seqs = int(os.environ.get("PSRL_RM_SMOKE_MAX_NUM_SEQS", "10240"))
        gpu_memory_utilization = float(os.environ.get("PSRL_RM_SMOKE_GPU_MEMORY_UTILIZATION", "0.8"))
        disable_kv_cache = _env_bool("PSRL_RM_SMOKE_DISABLE_KV_CACHE", False)
        if "PSRL_RM_SMOKE_ENFORCE_EAGER" in os.environ:
            enforce_eager = _env_bool("PSRL_RM_SMOKE_ENFORCE_EAGER", True)
        else:
            enforce_eager = True

    return OmegaConf.create(
        {
            "reward_model_name": "rm_tms_smoke",
            "num_replicas": 1,
            "model": {
                "path": model_path,
                "external_lib": None,
                "trust_remote_code": False,
                "use_shm": False,
                "override_config": {},
            },
            "rollout": {
                "_target_": "psrl.workers.config.RolloutConfig",
                "name": "vllm",
                "mode": "psrl_async",
                "disable_attn": False,
                "dtype": "bfloat16",
                "gpu_memory_utilization": gpu_memory_utilization,
                "enforce_eager": enforce_eager,
                "free_cache_engine": True,
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "data_parallel_size": 1,
                "expert_parallel_size": 1,
                "max_model_len": max_model_len,
                "prompt_length": prompt_length,
                "response_length": response_length,
                "max_num_batched_tokens": max_num_batched_tokens,
                "max_num_seqs": max_num_seqs,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "disable_kv_cache": disable_kv_cache,
                "disable_log_stats": True,
                "skip_tokenizer_init": False,
                "load_format": "dummy",
                "logprobs_mode": "raw_logprobs",
                "engine_kwargs": {},
                "limit_images": None,
                "runner": "generate",
                "task": "generate",
            },
        }
    )


def _build_prompt(model_path: str, uid: int) -> DataProto:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    encoded = tokenizer("Hello, please answer with one short sentence.", return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))

    raw_prompt_ids = np.empty(1, dtype=object)
    raw_prompt_ids[0] = input_ids[0].tolist()
    raw_response_ids = np.empty(1, dtype=object)
    raw_response_ids[0] = []

    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        batch_size=1,
    )
    return DataProto(
        batch=batch,
        non_tensor_batch={
            "uid": np.array([uid], dtype=object),
            "version_tag": np.array([0], dtype=int),
            "raw_prompt_ids": raw_prompt_ids,
            "raw_response_ids": raw_response_ids,
        },
    )


def _assert_generated(output: DataProto | tuple[DataProto, object], stage: str) -> DataProto:
    if isinstance(output, tuple):
        output = output[0]
    assert output is not None, f"{stage} returned None"
    assert "raw_response_ids" in output.non_tensor_batch, f"{stage} did not return raw_response_ids"
    assert len(output.non_tensor_batch["raw_response_ids"]) == 1
    return output


def _init_timeout_s() -> int:
    return int(os.environ.get("PSRL_RM_SMOKE_INIT_TIMEOUT_S", "900"))


def _wake_timeout_s() -> int:
    return int(os.environ.get("PSRL_RM_SMOKE_WAKE_TIMEOUT_S", "900"))


@pytest.mark.integration
@pytest.mark.skipduringci
def test_reward_model_tms_generate_sleep_wakeup_generate(tmp_path: Path):
    """Exercise the RM TMS path end-to-end.

    Default (PSRL_RM_SMOKE_LOAD_REAL_WEIGHTS_AT_INIT=1): load real GPU weights
    during init_model after CPU preload, then sleep — tests whether sleep footprint
    differs from the dummy-only elastic init path.

    Flow:
      1. init_model("empty")  -> dummy vLLM init + CPU preload + GPU load (if enabled)
      2. sleep(level=2)
      3. wake_up -> generate -> sleep -> wake_up -> generate

    Set PSRL_RM_SMOKE_LOAD_REAL_WEIGHTS_AT_INIT=0 to reproduce production elastic
    init (skip GPU load before first sleep).
    Set PSRL_RM_SMOKE_FAST=1 for the small-config smoke profile.
    """
    if os.environ.get("PSRL_RUN_RM_TMS_SMOKE") != "1":
        pytest.skip("Set PSRL_RUN_RM_TMS_SMOKE=1 to run the RM TMS smoke test.")
    if not torch.cuda.is_available():
        pytest.skip("RM TMS smoke test requires CUDA.")

    model_path = _default_model_path()
    if not (Path(model_path) / "config.json").exists():
        pytest.skip(f"RM smoke model is unavailable: {model_path}")

    load_real_weights_at_init = _env_bool("PSRL_RM_SMOKE_LOAD_REAL_WEIGHTS_AT_INIT", True)
    psrl_config = _build_psrl_config(tmp_path, load_real_weights_at_init=load_real_weights_at_init)
    reward_config = _build_reward_config(model_path)
    init_timeout_s = _init_timeout_s()
    wake_timeout_s = _wake_timeout_s()

    resources, env_vars, init_kwargs = PSRL_GenWorker.configure_worker(
        config=reward_config,
        psrl_config=psrl_config,
        num_gpus=1,
        dp_idx=0,
        bundle_indices=[0],
        role="reward",
    )
    actor_env = {
        **env_vars,
        "WG_BACKEND": "ray",
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": _free_port(),
        "TOKENIZERS_PARALLELISM": "false",
    }

    started_ray = False
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        started_ray = True

    actor = None
    try:
        print(
            f"RM smoke profile: load_real_weights_at_init={load_real_weights_at_init}, "
            f"gen_worker_elastic_rm_enable={bool(psrl_config.deployment.elastic_rm.enable)}, "
            f"max_model_len={reward_config.rollout.max_model_len}, "
            f"max_num_seqs={reward_config.rollout.max_num_seqs}, "
            f"gpu_memory_utilization={reward_config.rollout.gpu_memory_utilization}",
            flush=True,
        )
        _print_cluster_gpu_memory("before_actor_start")

        status_queue = ray.util.queue.Queue()
        gen_interface = GenInterface(rollout_instance_id=0, status_queue=status_queue)
        RewardWorker = ray.remote(
            max_concurrency=10000,
            num_gpus=resources.get("num_gpus", 1),
            num_cpus=resources.get("num_cpus", 1),
            runtime_env={"env_vars": actor_env},
        )(PSRL_GenWorker)
        actor = RewardWorker.remote(
            config=reward_config,
            role="reward",
            psrl_config=psrl_config,
            gen_interface=gen_interface,
            instance_id=0,
            reward_model_name="rm_tms_smoke",
            **init_kwargs,
        )

        ray.get(actor.init_model.remote("empty"), timeout=init_timeout_s)
        if load_real_weights_at_init:
            _print_cluster_gpu_memory("after_init_model_with_real_weights")
        else:
            _print_cluster_gpu_memory("after_init_model_empty")

        # Coordinator sleeps immediately after init in production elastic RM.
        ray.get(actor.sleep.remote(), timeout=300)
        _print_cluster_gpu_memory("after_init_sleep")

        ray.get(actor.wake_up.remote(), timeout=wake_timeout_s)
        _print_cluster_gpu_memory("after_wake_up")

        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": reward_config.rollout.response_length,
        }
        first = ray.get(
            actor.generate_async.remote(_build_prompt(model_path, uid=1), sampling_params),
            timeout=300,
        )
        _assert_generated(first, "first generate")
        _print_cluster_gpu_memory("after_first_generate")

        ray.get(actor.sleep.remote(), timeout=300)
        _print_cluster_gpu_memory("after_second_sleep")

        ray.get(actor.wake_up.remote(), timeout=wake_timeout_s)
        _print_cluster_gpu_memory("after_second_wake_up")

        second = ray.get(
            actor.generate_async.remote(_build_prompt(model_path, uid=2), sampling_params),
            timeout=300,
        )
        _assert_generated(second, "second generate")
        _print_cluster_gpu_memory("after_second_generate")
    finally:
        if actor is not None:
            ray.kill(actor, no_restart=True)
            time.sleep(5)
            _print_cluster_gpu_memory("after_actor_kill")
        if started_ray:
            ray.shutdown()


def _run_bisect_case(
    tmp_path: Path,
    *,
    experiment: str,
    description: str,
    init_timeout_s: int,
) -> int:
    """Init RM (elastic dummy-only) -> sleep -> return GPU0 used MiB after sleep."""
    model_path = _default_model_path()
    load_real_weights_at_init = _env_bool("PSRL_RM_SMOKE_LOAD_REAL_WEIGHTS_AT_INIT", False)
    psrl_config = _build_psrl_config(tmp_path, load_real_weights_at_init=load_real_weights_at_init)
    reward_config = _build_reward_config(model_path)

    resources, env_vars, init_kwargs = PSRL_GenWorker.configure_worker(
        config=reward_config,
        psrl_config=psrl_config,
        num_gpus=1,
        dp_idx=0,
        bundle_indices=[0],
        role="reward",
    )
    vllm_patches_override = os.environ.get("PSRL_RM_SMOKE_VLLM_PATCHES", "").strip()
    actor_env = {
        **env_vars,
        "WG_BACKEND": "ray",
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": _free_port(),
        "TOKENIZERS_PARALLELISM": "false",
    }
    if vllm_patches_override:
        actor_env["PSRL_VLLM_PATCHES"] = vllm_patches_override

    print(
        f"\n{'=' * 72}\n"
        f"BISect {experiment}: {description}\n"
        f"  load_real_weights_at_init={load_real_weights_at_init}\n"
        f"  elastic_rm.enable={bool(psrl_config.deployment.elastic_rm.enable)}\n"
        f"  max_model_len={reward_config.rollout.max_model_len}\n"
        f"  max_num_seqs={reward_config.rollout.max_num_seqs}\n"
        f"  enforce_eager={reward_config.rollout.enforce_eager}\n"
        f"  PSRL_VLLM_PATCHES={actor_env.get('PSRL_VLLM_PATCHES', '<unset>')}\n"
        f"  tms.enable_cuda_graph={bool(psrl_config.tms.enable_cuda_graph)}\n"
        f"{'=' * 72}",
        flush=True,
    )

    status_queue = ray.util.queue.Queue()
    gen_interface = GenInterface(rollout_instance_id=0, status_queue=status_queue)
    RewardWorker = ray.remote(
        max_concurrency=10000,
        num_gpus=resources.get("num_gpus", 1),
        num_cpus=resources.get("num_cpus", 1),
        runtime_env={"env_vars": actor_env},
    )(PSRL_GenWorker)
    actor = RewardWorker.remote(
        config=reward_config,
        role="reward",
        psrl_config=psrl_config,
        gen_interface=gen_interface,
        instance_id=0,
        reward_model_name="rm_tms_smoke",
        **init_kwargs,
    )
    try:
        ray.get(actor.init_model.remote("empty"), timeout=init_timeout_s)
        ray.get(actor.sleep.remote(), timeout=300)
        snapshots = _collect_gpu_snapshots(f"bisect_{experiment}_after_init_sleep")
        _print_cluster_gpu_memory(f"bisect_{experiment}_after_init_sleep")
        gpu0_mib = _parse_gpu0_used_mib(snapshots)
        if gpu0_mib is None:
            raise RuntimeError(f"BISect {experiment}: failed to parse GPU0 memory from snapshot")
        print(f"BISect {experiment} result: GPU0 after_init_sleep = {gpu0_mib} MiB", flush=True)
        return gpu0_mib
    finally:
        ray.kill(actor, no_restart=True)
        time.sleep(5)


@pytest.mark.integration
@pytest.mark.skipduringci
def test_rm_tms_sleep_bisect_experiments(tmp_path: Path):
    """Bisect why RM sleep retains ~24GB while rollout sleep is ~2GB.

    Runs baseline (prod RM config) then A–D one at a time:
      A: max_num_seqs 10240 -> 1024 (rollout default)
      B: enforce_eager True -> False (rollout default)
      C: PSRL_VLLM_PATCHES TMS -> TMS:GRAPH (rollout patch stack)
      D: max_model_len 21504 -> 11264 (rollout-aligned)

    Set PSRL_RUN_RM_TMS_BISECT=1 to run. Uses elastic init (no GPU load before sleep).
    """
    if os.environ.get("PSRL_RUN_RM_TMS_BISECT") != "1":
        pytest.skip("Set PSRL_RUN_RM_TMS_BISECT=1 to run RM sleep bisect experiments.")
    if not torch.cuda.is_available():
        pytest.skip("RM sleep bisect requires CUDA.")

    model_path = _default_model_path()
    if not (Path(model_path) / "config.json").exists():
        pytest.skip(f"RM sleep bisect model is unavailable: {model_path}")

    init_timeout_s = _init_timeout_s()
    started_ray = False
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        started_ray = True

    # Each case uses a dedicated tmp subdir and fresh env (set via os.environ).
    cases: list[tuple[str, str, dict[str, str]]] = [
        (
            "baseline",
            "production RM config (reference)",
            {},
        ),
        (
            "A",
            "max_num_seqs=1024 (rollout default)",
            {"PSRL_RM_SMOKE_MAX_NUM_SEQS": "1024"},
        ),
        (
            "B",
            "enforce_eager=False (rollout default)",
            {"PSRL_RM_SMOKE_ENFORCE_EAGER": "0"},
        ),
        (
            "C",
            "PSRL_VLLM_PATCHES=TMS:GRAPH (rollout patch stack)",
            {
                "PSRL_RM_SMOKE_VLLM_PATCHES": "TMS:GRAPH",
                "PSRL_RM_SMOKE_ENABLE_CUDA_GRAPH": "1",
            },
        ),
        (
            "D",
            "max_model_len=11264 (rollout-aligned)",
            {
                "PSRL_RM_SMOKE_MAX_MODEL_LEN": str(_ROLLOUT_ALIGNED_MAX_MODEL_LEN),
                "PSRL_RM_SMOKE_MAX_NUM_BATCHED_TOKENS": str(_ROLLOUT_ALIGNED_MAX_MODEL_LEN),
            },
        ),
    ]

    # Keys we own for bisect; restore after all cases.
    bisect_keys = {
        "PSRL_RM_SMOKE_MAX_NUM_SEQS",
        "PSRL_RM_SMOKE_ENFORCE_EAGER",
        "PSRL_RM_SMOKE_VLLM_PATCHES",
        "PSRL_RM_SMOKE_ENABLE_CUDA_GRAPH",
        "PSRL_RM_SMOKE_MAX_MODEL_LEN",
        "PSRL_RM_SMOKE_MAX_NUM_BATCHED_TOKENS",
        "PSRL_RM_SMOKE_LOAD_REAL_WEIGHTS_AT_INIT",
    }
    saved_env = {k: os.environ.get(k) for k in bisect_keys}

    results: list[tuple[str, str, int]] = []
    try:
        for experiment, description, overrides in cases:
            for key in bisect_keys:
                if saved_env.get(key) is not None:
                    os.environ[key] = saved_env[key]
                else:
                    os.environ.pop(key, None)
            os.environ["PSRL_RM_SMOKE_LOAD_REAL_WEIGHTS_AT_INIT"] = "0"
            for key, value in overrides.items():
                os.environ[key] = value
            case_tmp = tmp_path / f"bisect_{experiment}"
            case_tmp.mkdir(parents=True, exist_ok=True)
            gpu0_mib = _run_bisect_case(
                case_tmp,
                experiment=experiment,
                description=description,
                init_timeout_s=init_timeout_s,
            )
            results.append((experiment, description, gpu0_mib))

        print("\n" + "=" * 72, flush=True)
        print("RM sleep bisect summary (GPU0 MiB after init_sleep):", flush=True)
        print("=" * 72, flush=True)
        baseline_mib = results[0][2]
        for experiment, description, gpu0_mib in results:
            delta = gpu0_mib - baseline_mib
            flag = " <-- improved" if experiment != "baseline" and gpu0_mib < 8000 else ""
            print(
                f"  {experiment:8s}  {gpu0_mib:6d} MiB  (delta {delta:+6d})  {description}{flag}",
                flush=True,
            )
        print("=" * 72 + "\n", flush=True)
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if started_ray:
            ray.shutdown()


def _tms_vllm_model_path(env_name: str, relative_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return os.environ.get(env_name, str(repo_root / relative_path))


def _build_tms_comparison_reward_config(model_path: str, tp_size: int, ep_size: int = 1):
    reward_config = _build_reward_config(model_path)
    max_model_len = int(os.environ.get("PSRL_TMS_VLLM_MAX_MODEL_LEN", "16384"))
    reward_config.rollout.tensor_model_parallel_size = tp_size
    reward_config.rollout.expert_parallel_size = ep_size
    reward_config.rollout.prompt_length = int(
        os.environ.get("PSRL_TMS_VLLM_PROMPT_LENGTH", "1024")
    )
    reward_config.rollout.response_length = int(
        os.environ.get("PSRL_TMS_VLLM_MAX_TOKENS", "32")
    )
    reward_config.rollout.max_model_len = max_model_len
    reward_config.rollout.max_num_batched_tokens = int(
        os.environ.get("PSRL_TMS_VLLM_MAX_NUM_BATCHED_TOKENS", str(max_model_len))
    )
    reward_config.rollout.max_num_seqs = int(
        os.environ.get("PSRL_TMS_VLLM_MAX_NUM_SEQS", "64")
    )
    reward_config.rollout.gpu_memory_utilization = float(
        os.environ.get("PSRL_TMS_VLLM_GPU_MEMORY_UTILIZATION", "0.7")
    )
    reward_config.rollout.enforce_eager = _env_bool(
        "PSRL_TMS_VLLM_ENFORCE_EAGER", False
    )
    reward_config.rollout.enable_chunked_prefill = True
    reward_config.rollout.enable_prefix_caching = False
    reward_config.rollout.disable_kv_cache = False
    reward_config.rollout.use_psrl_scheduler = False
    return reward_config


def _record_tms_vllm_memory(
    report: dict[str, object], case_name: str, stage: str
) -> list[dict[str, str]]:
    snapshots = _print_cluster_gpu_memory(f"tms_{case_name}_{stage}")
    report.setdefault("memory_snapshots", {})[stage] = snapshots
    return snapshots


def _wait_concurrent_refs(refs_by_instance: dict[int, ray.ObjectRef]) -> dict[str, object]:
    """Wait for a concurrent wave and retain each instance's completion offset."""
    started_at = time.perf_counter()
    pending = {ref: instance_id for instance_id, ref in refs_by_instance.items()}
    completion_s = {}
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        for ref in ready:
            instance_id = pending.pop(ref)
            ray.get(ref)
            completion_s[str(instance_id)] = time.perf_counter() - started_at
    return {
        "wall_s": time.perf_counter() - started_at,
        "instance_completion_s": completion_s,
    }


def _tms_generation_summary(output: DataProto) -> dict[str, object]:
    response_ids = [int(token_id) for token_id in output.non_tensor_batch["raw_response_ids"][0]]
    return {"num_output_tokens": len(response_ids), "response_ids": response_ids}


def _tms_stress_generation_config(reward_config) -> dict[str, int]:
    batch_size = int(os.environ.get("PSRL_TMS_VLLM_STRESS_BATCH_SIZE", "0"))
    cycles = int(os.environ.get("PSRL_TMS_VLLM_STRESS_CYCLES", "2"))
    max_tokens = int(
        os.environ.get(
            "PSRL_TMS_VLLM_STRESS_MAX_TOKENS",
            str(reward_config.rollout.response_length),
        )
    )
    min_tokens = int(os.environ.get("PSRL_TMS_VLLM_STRESS_MIN_TOKENS", str(max_tokens)))
    sleep_growth_limit_mib = int(
        os.environ.get("PSRL_TMS_VLLM_SLEEP_GROWTH_LIMIT_MIB", "1024")
    )
    if batch_size < 0:
        raise ValueError(f"PSRL_TMS_VLLM_STRESS_BATCH_SIZE must be non-negative, got {batch_size}")
    if batch_size > int(reward_config.rollout.max_num_seqs):
        raise ValueError(
            "PSRL_TMS_VLLM_STRESS_BATCH_SIZE exceeds rollout.max_num_seqs: "
            f"batch_size={batch_size} max_num_seqs={reward_config.rollout.max_num_seqs}"
        )
    if cycles <= 0:
        raise ValueError(f"PSRL_TMS_VLLM_STRESS_CYCLES must be positive, got {cycles}")
    if max_tokens > int(reward_config.rollout.response_length):
        raise ValueError(
            "PSRL_TMS_VLLM_STRESS_MAX_TOKENS exceeds rollout.response_length; "
            "set PSRL_TMS_VLLM_MAX_TOKENS to the same or a larger value: "
            f"stress_max_tokens={max_tokens} response_length={reward_config.rollout.response_length}"
        )
    if not 0 <= min_tokens <= max_tokens:
        raise ValueError(
            f"Stress min_tokens must be in [0, max_tokens], got min={min_tokens} max={max_tokens}"
        )
    if sleep_growth_limit_mib < 0:
        raise ValueError(
            "PSRL_TMS_VLLM_SLEEP_GROWTH_LIMIT_MIB must be non-negative, "
            f"got {sleep_growth_limit_mib}"
        )
    return {
        "batch_size": batch_size,
        "cycles": cycles,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "sleep_growth_limit_mib": sleep_growth_limit_mib,
    }


def _run_tms_stress_generation_batch(
    actor,
    *,
    model_path: str,
    uid_start: int,
    batch_size: int,
    sampling_params: dict[str, object],
    min_tokens: int,
    timeout_s: int,
) -> list[dict[str, object]]:
    refs = [
        actor.generate_async.remote(
            _build_prompt(model_path, uid=uid_start + index),
            sampling_params,
        )
        for index in range(batch_size)
    ]
    outputs = ray.get(refs, timeout=timeout_s)
    summaries = []
    for index, output in enumerate(outputs):
        generated = _assert_generated(output, f"stress generate request {index}")
        summary = _tms_generation_summary(generated)
        if int(summary["num_output_tokens"]) < min_tokens:
            raise AssertionError(
                "Stress generation ended before min_tokens: "
                f"request={index} generated={summary['num_output_tokens']} min_tokens={min_tokens}"
            )
        summaries.append(summary)
    return summaries


def _validate_weight_arena_infos(
    infos: list[dict[str, object]],
    *,
    tp_size: int,
    expected_node_id: str,
    expected_base_addresses: dict[int, list[int]] | None = None,
) -> dict[int, list[int]]:
    """Validate arena activation and address stability across all TP ranks."""
    if len(infos) != tp_size:
        raise AssertionError(f"Expected {tp_size} arena worker infos, got {len(infos)}.")
    by_rank: dict[int, list[int]] = {}
    for info in infos:
        if not info.get("enabled"):
            raise AssertionError(f"Weight arena was not enabled in worker info: {info}")
        if info.get("node_id") != expected_node_id:
            raise AssertionError(f"Arena worker escaped the local node: {info}")
        rank = int(info["tp_rank"])
        addresses = [int(address) for address in info["base_addresses"]]
        stats = info.get("stats") or {}
        if int(stats.get("arena_count", 0)) <= 0:
            raise AssertionError(f"Arena worker reported no allocations: {info}")
        by_rank[rank] = addresses
        if expected_base_addresses is not None and addresses != expected_base_addresses.get(rank):
            raise AssertionError(
                f"Arena virtual addresses changed for TP rank {rank}: "
                f"expected {expected_base_addresses.get(rank)}, got {addresses}."
            )
    if set(by_rank) != set(range(tp_size)):
        raise AssertionError(f"Missing TP ranks in arena info: {sorted(by_rank)}")
    return by_rank


def _validate_reward_arena_cache_infos(
    infos: list[dict[str, object]],
    *,
    expected_inodes: dict[int, int] | None = None,
) -> dict[int, int]:
    """Validate that every reward TP rank owns a complete arena-layout CPU mirror."""
    inodes: dict[int, int] = {}
    for info in infos:
        stats = info.get("stats") or {}
        cache = info.get("reward_cpu_cache") or {}
        state = info.get("reward_cpu_cache_state") or {}
        if state.get("source") not in ("arena", "node_shared_arena") or not state.get("ready"):
            raise AssertionError(f"Reward arena CPU cache is not ready: {info}")
        if int(cache.get("arena_count", 0)) != int(stats.get("arena_count", -1)):
            raise AssertionError(f"Reward CPU cache count does not match the GPU arena: {info}")
        if int(cache.get("total_bytes", 0)) != int(stats.get("total_arena_bytes", -1)):
            raise AssertionError(f"Reward CPU cache bytes do not match the GPU arena: {info}")
        if state.get("source") == "node_shared_arena":
            rank = int(info["tp_rank"])
            inode = int(cache.get("inode", 0))
            if inode <= 0:
                raise AssertionError(f"Reward node-shared cache is missing its inode: {info}")
            inodes[rank] = inode
            if expected_inodes is not None and inode != expected_inodes.get(rank):
                raise AssertionError(
                    f"Reward node-shared cache inode changed for TP rank {rank}: "
                    f"expected {expected_inodes.get(rank)}, got {inode}."
                )
    return inodes


@pytest.mark.integration
@pytest.mark.skipduringci
@pytest.mark.parametrize(
    ("case_name", "model_env", "relative_model_path", "tp_size", "ep_size"),
    [
        pytest.param(
            "qwen2_5_1p5b_tp1",
            "PSRL_TMS_VLLM_1P5B_MODEL",
            "models/Qwen2.5-1.5B",
            1,
            1,
            id="qwen2.5-1.5b-tp1",
        ),
        pytest.param(
            "qwen2_5_7b_tp1",
            "PSRL_TMS_VLLM_7B_MODEL",
            "models/Qwen2.5-7B",
            1,
            1,
            id="qwen2.5-7b-tp1",
        ),
        pytest.param(
            "qwen2_5_32b_tp4",
            "PSRL_TMS_VLLM_32B_MODEL",
            "models/Qwen2.5-32B",
            4,
            1,
            id="qwen2.5-32b-tp4",
        ),
        pytest.param(
            "qwen2_5_72b_tp8",
            "PSRL_TMS_VLLM_72B_MODEL",
            "models/Qwen2.5-72B",
            8,
            1,
            id="qwen2.5-72b-tp8",
        ),
        pytest.param(
            "glm_z1_9b_tp1",
            "PSRL_TMS_VLLM_GLM_9B_MODEL",
            "models/GLM-Z1-9B-0414",
            1,
            1,
            id="glm-z1-9b-tp1",
        ),
        pytest.param(
            "qwen3_30b_a3b_tp4",
            "PSRL_TMS_VLLM_MOE_MODEL",
            "models/Qwen3-30B-A3B-Thinking-2507",
            4,
            1,
            id="qwen3-30b-a3b-tp4",
        ),
        pytest.param(
            "qwen3_30b_a3b_tp4_ep4",
            "PSRL_TMS_VLLM_MOE_MODEL",
            "models/Qwen3-30B-A3B-Thinking-2507",
            4,
            4,
            id="qwen3-30b-a3b-tp4-ep4",
        ),
        pytest.param(
            "qwen3_5_122b_a10b_tp8_ep8",
            "PSRL_TMS_VLLM_QWEN3_5_122B_MODEL",
            "models/Qwen3.5-122B-A10B",
            8,
            8,
            id="qwen3.5-122b-a10b-tp8-ep8",
        ),
        pytest.param(
            "qwen3_next_80b_a3b_tp8_ep8",
            "PSRL_TMS_VLLM_QWEN3_NEXT_80B_MODEL",
            "models/Qwen3-Next-80B-A3B-Thinking",
            8,
            8,
            id="qwen3-next-80b-a3b-tp8-ep8",
        ),
    ],
)
def test_tms_vllm_level2_sleep_memory(
    tmp_path: Path,
    case_name: str,
    model_env: str,
    relative_model_path: str,
    tp_size: int,
    ep_size: int,
):
    """Measure TMS-managed vLLM sleep over two wake cycles."""
    if os.environ.get("PSRL_RUN_TMS_VLLM_SLEEP_SMOKE") != "1":
        pytest.skip(
            "Set PSRL_RUN_TMS_VLLM_SLEEP_SMOKE=1 to run TMS vLLM sleep probes."
        )

    model_path = _tms_vllm_model_path(model_env, relative_model_path)
    if not (Path(model_path) / "config.json").exists():
        pytest.skip(f"TMS vLLM sleep model is unavailable: {model_path}")
    tp_size = int(os.environ.get("PSRL_TMS_VLLM_TP_SIZE", str(tp_size)))

    started_ray = False
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        started_ray = True
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus < tp_size:
        if started_ray:
            ray.shutdown()
        pytest.skip(f"{case_name} needs {tp_size} GPUs, but Ray sees {available_gpus}")

    probe_role = os.environ.get("PSRL_TMS_VLLM_PROBE_ROLE", "reward").strip().lower()
    if probe_role not in {"reward", "rollout"}:
        raise ValueError(f"PSRL_TMS_VLLM_PROBE_ROLE must be reward or rollout, got {probe_role!r}")

    load_real_weights_at_init = _env_bool("PSRL_RM_SMOKE_LOAD_REAL_WEIGHTS_AT_INIT", False)
    psrl_config = _build_psrl_config(
        tmp_path,
        load_real_weights_at_init=load_real_weights_at_init,
    )
    psrl_config.tms.enable_cuda_graph = probe_role == "rollout"
    if probe_role == "rollout":
        psrl_config.ps_mode = "nixl_cpu"
        psrl_config.tms.enable_nixl = True
    weight_arena_enabled = _env_bool("PSRL_TMS_VLLM_WEIGHT_ARENA", False)
    if weight_arena_enabled:
        psrl_config.nixl.weight_arena = _build_tms_weight_arena_config(probe_role)
    local_node_only = _env_bool("PSRL_TMS_VLLM_LOCAL_NODE_ONLY", False)
    verify_generation = probe_role == "rollout" and _env_bool(
        "PSRL_TMS_VLLM_VERIFY_GENERATION", False
    )
    reward_config = _build_tms_comparison_reward_config(model_path, tp_size, ep_size)
    stress_config = _tms_stress_generation_config(reward_config)
    init_timeout_s = int(os.environ.get("PSRL_TMS_VLLM_INIT_TIMEOUT_S", "1800"))
    operation_timeout_s = int(
        os.environ.get("PSRL_TMS_VLLM_OPERATION_TIMEOUT_S", "900")
    )
    timings: dict[str, float] = {}
    generations: dict[str, object] = {}
    worker_stage_timings: dict[str, list[dict[str, object]]] = {}
    arena_base_addresses: dict[int, list[int]] | None = None
    registration_samples: list[float] = []
    reward_restore_samples: list[float] = []
    report: dict[str, object] = {
        "case": case_name,
        "probe_role": probe_role,
        "model_path": model_path,
        "tp_size": tp_size,
        "ep_size": ep_size,
        "weight_arena_enabled": weight_arena_enabled,
        "elastic_rm_enabled": bool(psrl_config.deployment.elastic_rm.enable),
        "local_node_only": local_node_only,
        "verify_generation": verify_generation,
        "stress_generation": stress_config,
        "reward_config": OmegaConf.to_container(reward_config, resolve=True),
        "timings_s": timings,
        "generations": generations,
    }

    output_dir_raw = os.environ.get("PSRL_TMS_VLLM_SLEEP_OUTPUT_DIR")
    output_dir = Path(output_dir_raw).expanduser() if output_dir_raw else tmp_path
    output_dir.mkdir(parents=True, exist_ok=True)
    report_name = case_name if probe_role == "reward" else f"{probe_role}_{case_name}"
    report_path = output_dir / f"tms_vllm_sleep_{report_name}.json"

    resources, env_vars, init_kwargs = PSRL_GenWorker.configure_worker(
        config=reward_config,
        psrl_config=psrl_config,
        num_gpus=1,
        dp_idx=0,
        bundle_indices=list(range(tp_size)),
        role=probe_role,
    )
    actor_env = {
        **env_vars,
        "WG_BACKEND": "ray",
        "WORLD_SIZE": str(tp_size),
        "RANK": "0",
        "LOCAL_WORLD_SIZE": str(tp_size),
        "LOCAL_RANK": "0",
        "RAY_LOCAL_WORLD_SIZE": str(tp_size),
        "RAY_LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": _free_port(),
        "TOKENIZERS_PARALLELISM": "false",
    }
    tms_info = {
        "psrl_vllm_patches": actor_env.get("PSRL_VLLM_PATCHES", ""),
        "psrl_vllm_weight_arena": actor_env.get("PSRL_VLLM_WEIGHT_ARENA", ""),
        "vllm_psrl_weight_arena": actor_env.get("VLLM_PSRL_WEIGHT_ARENA", ""),
        "ld_preload": actor_env.get("LD_PRELOAD", ""),
        "tms_init_enable": actor_env.get("TMS_INIT_ENABLE", ""),
        "tms_init_enable_cpu_backup": actor_env.get(
            "TMS_INIT_ENABLE_CPU_BACKUP", ""
        ),
    }
    expected_patch = "TMS:GRAPH" if probe_role == "rollout" else "TMS"
    if tms_info["psrl_vllm_patches"] != expected_patch:
        raise RuntimeError(f"TMS patch is not configured: {tms_info}")
    if "torch_memory_saver_hook_mode_preload" not in tms_info["ld_preload"]:
        raise RuntimeError(f"TMS LD_PRELOAD hook is not configured: {tms_info}")
    report["tms"] = tms_info

    actor = None
    probe_pg = None
    meta_server = None
    ps_control = None
    ps_actor = None
    nixl_interface = None
    ps_preload_ref = None
    try:
        _record_tms_vllm_memory(report, case_name, "before_actor_start")
        bundle = {"CPU": 1, "GPU": 1}
        local_node = None
        if local_node_only:
            local_node_ip = ray.util.get_node_ip_address()
            local_node = next(
                (
                    node
                    for node in ray.nodes()
                    if node.get("Alive", False) and node.get("NodeManagerAddress") == local_node_ip
                ),
                None,
            )
            if local_node is None:
                raise RuntimeError(f"Cannot resolve local Ray node for IP {local_node_ip}.")
            local_node_resource = f"node:{local_node_ip}"
            if local_node_resource not in local_node["Resources"]:
                raise RuntimeError(f"Local Ray node does not advertise {local_node_resource}.")
            bundle[local_node_resource] = 0.001
        probe_pg = placement_group([dict(bundle) for _ in range(tp_size)], strategy="STRICT_PACK")
        ray.get(probe_pg.ready(), timeout=init_timeout_s)

        identity = ray.get(
            _probe_node_identity.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    probe_pg,
                    placement_group_bundle_index=0,
                )
            ).remote(),
            timeout=60,
        )
        if local_node is not None and identity["node_id"] != local_node["NodeID"]:
            raise RuntimeError(
                f"Probe placement escaped the local node: expected {local_node['NodeID']}, got {identity['node_id']}."
            )
        if probe_role == "rollout":
            from psrl.utils.nixl import GLOBAL_PORT_SCANNER, NIXLInterface
            from psrl.workers.ps import PSStoragePlan, PSStorageWorker

            server_port = ray.get(
                GLOBAL_PORT_SCANNER.find_free_port.remote(host=identity["node_ip"]),
                timeout=60,
            )
            nixl_runtime_config = {
                "server_ip": identity["node_ip"],
                "server_port": server_port,
                "max_pinned_temp_memory_slots": 64,
                "enable_tms_for_temp_buffers": True,
            }
            if weight_arena_enabled:
                nixl_runtime_config["weight_arena"] = OmegaConf.to_container(
                    psrl_config.nixl.weight_arena,
                    resolve=True,
                )
            psrl_config.nixl = OmegaConf.create(nixl_runtime_config)
            nixl_interface = NIXLInterface(port_scanner=GLOBAL_PORT_SCANNER)
            node_affinity = NodeAffinitySchedulingStrategy(
                node_id=identity["node_id"],
                soft=False,
            )
            meta_server = _NIXLProbeMetaServer.options(
                scheduling_strategy=node_affinity,
            ).remote(psrl_config.nixl, tp_size + 1)
            ray.get(meta_server.ready.remote(), timeout=60)
            ps_control = _NIXLProbePSControl.options(
                scheduling_strategy=node_affinity,
            ).remote()

            PSActor = ray.remote(num_cpus=0)(PSStorageWorker)
            ps_actor = PSActor.options(
                scheduling_strategy=node_affinity,
                runtime_env={
                    "env_vars": {
                        "WORLD_SIZE": "1",
                        "RANK": "0",
                        "PS_NODE_IP": identity["node_ip"],
                    }
                },
            ).remote(
                PSStoragePlan(
                    train_model_dtype=torch.bfloat16,
                    gen_model_dtype=torch.bfloat16,
                ),
                reward_config.model,
                psrl_config,
                nixl_interface,
            )
            ps_init_start = time.perf_counter()
            ps_actor.init_nixl_client.remote()
            ps_actor.init_model.remote()
            ps_preload_ref = ps_actor.preload_checkpoint_to_cpu.remote()
            report["ps_probe"] = {
                "node_id": identity["node_id"],
                "node_ip": identity["node_ip"],
                "server_port": server_port,
                "expected_nixl_agents": tp_size + 1,
            }

        RewardWorker = ray.remote(
            max_concurrency=10000,
            num_gpus=resources.get("num_gpus", 1),
            num_cpus=resources.get("num_cpus", 1),
            runtime_env={"env_vars": actor_env},
        )(PSRL_GenWorker)
        actor = RewardWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                probe_pg,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )
        ).remote(
            config=reward_config,
            role=probe_role,
            psrl_config=psrl_config,
            gen_interface=GenInterface(
                rollout_instance_id=0,
                status_queue=ray.util.queue.Queue(),
                ps_manager_handle=ps_control,
            ),
            nixl_interface=nixl_interface,
            instance_id=0,
            reward_model_name="rm_tms_sleep_probe",
            **init_kwargs,
        )

        start = time.perf_counter()
        ray.get(actor.init_model.remote("empty"), timeout=init_timeout_s)
        timings["engine_init"] = time.perf_counter() - start
        print(f"TMS vLLM probe environment: {tms_info}", flush=True)
        if weight_arena_enabled:
            arena_infos = ray.get(actor.get_weight_arena_info.remote(), timeout=operation_timeout_s)
            arena_base_addresses = _validate_weight_arena_infos(
                arena_infos,
                tp_size=tp_size,
                expected_node_id=identity["node_id"],
            )
            if probe_role == "reward":
                reward_cache_inodes = _validate_reward_arena_cache_infos(arena_infos)
                worker_stage_timings["initial_weight_cache_transition"] = ray.get(
                    actor.get_worker_stage_timing.remote("load_weights_from_cpu_cache"),
                    timeout=operation_timeout_s,
                )
                for worker_timing in worker_stage_timings["initial_weight_cache_transition"]:
                    timing = worker_timing.get("timing") or {}
                    if timing.get("cache_source") not in (
                        "checkpoint",
                        "node_shared_checkpoint_builder",
                        "node_shared_arena_consumer",
                    ) or not timing.get("cache_transitioned"):
                        raise AssertionError(
                            f"Reward init did not transition checkpoint cache to arena cache: {worker_timing}"
                        )
            report["weight_arena"] = {"after_engine_init": arena_infos}
        _record_tms_vllm_memory(report, case_name, "after_engine_init")

        if probe_role == "rollout":
            assert ps_actor is not None
            assert ps_control is not None
            assert meta_server is not None
            assert ps_preload_ref is not None

            ray.get(ps_preload_ref, timeout=init_timeout_s)
            timings["ps_init_and_checkpoint_preload"] = time.perf_counter() - ps_init_start

            start = time.perf_counter()
            ray.get(actor.init_nixl_client.remote(), timeout=operation_timeout_s)
            ray.get(actor.nixl_convert_params.remote(), timeout=operation_timeout_s)
            timings["rollout_nixl_client_init_and_convert"] = time.perf_counter() - start

            start = time.perf_counter()
            protocol_refs = [
                meta_server.protocol.remote(operation_timeout_s),
                ps_actor.nixl_protocol.remote(),
                actor.nixl_protocol.remote(),
            ]
            ray.get(protocol_refs, timeout=operation_timeout_s)
            timings["nixl_protocol"] = time.perf_counter() - start

            start = time.perf_counter()
            ray.get(
                ps_actor.write_checkpoint_to_registered_tensors.remote(),
                timeout=init_timeout_s,
            )
            timings["ps_write_checkpoint"] = time.perf_counter() - start

            ps_agent_name = ray.get(ps_actor.get_nixl_agent_name.remote(), timeout=60)
            ps_gen_client_name = ray.get(
                ps_actor.get_nixl_gen_storage_client_name.remote(),
                timeout=60,
            )
            ray.get(
                ps_control.set_endpoints.remote([ps_agent_name], [ps_gen_client_name]),
                timeout=60,
            )
            start = time.perf_counter()
            ray.get(actor.pull_model_async.remote(), timeout=operation_timeout_s)
            timings["initial_pull_from_ps"] = time.perf_counter() - start
            _record_tms_vllm_memory(report, case_name, "after_initial_pull_from_ps")

        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": reward_config.rollout.response_length,
        }
        if probe_role == "reward":
            ray.get(actor.resume_generation.remote(), timeout=operation_timeout_s)
            start = time.perf_counter()
            baseline = ray.get(
                actor.generate_async.remote(_build_prompt(model_path, uid=0), sampling_params),
                timeout=operation_timeout_s,
            )
            timings["baseline_generate"] = time.perf_counter() - start
            baseline = _assert_generated(baseline, "baseline generate")
            generations["baseline"] = _tms_generation_summary(baseline)
            _record_tms_vllm_memory(report, case_name, "after_baseline_generate")

        start = time.perf_counter()
        sleep_ref = actor.nixl_sleep.remote() if probe_role == "rollout" else actor.sleep.remote()
        ray.get(sleep_ref, timeout=operation_timeout_s)
        timings["first_sleep_level2"] = time.perf_counter() - start
        first_sleep_snapshots = _record_tms_vllm_memory(
            report,
            case_name,
            "after_first_sleep_level2",
        )
        first_sleep_gpu0_mib = _parse_gpu0_used_mib(first_sleep_snapshots)
        report["first_sleep_gpu0_mib"] = first_sleep_gpu0_mib
        if stress_config["batch_size"] > 0 and first_sleep_gpu0_mib is None:
            raise RuntimeError("Failed to parse the first-sleep GPU0 memory baseline")

        start = time.perf_counter()
        wake_ref = actor.nixl_wake_up.remote() if probe_role == "rollout" else actor.wake_up.remote()
        ray.get(wake_ref, timeout=operation_timeout_s)
        timings["first_wake"] = time.perf_counter() - start
        if probe_role == "rollout":
            pull_start = time.perf_counter()
            ray.get(actor.pull_model_async.remote(), timeout=operation_timeout_s)
            timings["first_pull_from_ps"] = time.perf_counter() - pull_start
            timings["first_wake_and_pull"] = time.perf_counter() - start
            registration_infos = ray.get(actor.get_weight_arena_info.remote(), timeout=operation_timeout_s)
            report.setdefault("nixl_registration", {})["after_first_wake"] = registration_infos
            for info in registration_infos:
                registration = info.get("nixl_registration")
                if registration is None:
                    raise AssertionError(f"Missing NIXL registration timing: {info}")
                registration_samples.append(float(registration["register_memory"]))
            worker_stage_timings["first_wake"] = ray.get(
                actor.get_worker_stage_timing.remote("nixl_register_after_wake_up"),
                timeout=operation_timeout_s,
            )
            if weight_arena_enabled:
                _validate_weight_arena_infos(
                    registration_infos,
                    tp_size=tp_size,
                    expected_node_id=report["ps_probe"]["node_id"],
                    expected_base_addresses=arena_base_addresses,
                )
                report["weight_arena"]["after_first_wake"] = registration_infos
                for info in registration_infos:
                    registration = info["nixl_registration"]
                    arena_count = int(info["stats"]["arena_count"])
                    if int(registration["registered_regions"]) != arena_count:
                        raise AssertionError(f"NIXL regions do not match arenas: {info}")
        elif weight_arena_enabled:
            arena_infos = ray.get(actor.get_weight_arena_info.remote(), timeout=operation_timeout_s)
            _validate_weight_arena_infos(
                arena_infos,
                tp_size=tp_size,
                expected_node_id=identity["node_id"],
                expected_base_addresses=arena_base_addresses,
            )
            _validate_reward_arena_cache_infos(
                arena_infos,
                expected_inodes=reward_cache_inodes,
            )
            report["weight_arena"]["after_first_wake"] = arena_infos
            worker_stage_timings["first_wake"] = ray.get(
                actor.get_worker_stage_timing.remote("load_weights_from_cpu_cache"),
                timeout=operation_timeout_s,
            )
            for worker_timing in worker_stage_timings["first_wake"]:
                timing = worker_timing.get("timing") or {}
                if timing.get("cache_source") not in ("arena", "node_shared_arena"):
                    raise AssertionError(f"Reward wake did not use the arena CPU cache: {worker_timing}")
                if int(timing.get("arena_restore_bytes", 0)) != int(timing.get("arena_bytes", -1)):
                    raise AssertionError(f"Reward wake restored incomplete arena bytes: {worker_timing}")
                reward_restore_samples.append(float(timing["gpu_copy_elapsed_s"]))
        _record_tms_vllm_memory(report, case_name, "after_first_wake")

        start = time.perf_counter()
        if probe_role == "reward":
            first = ray.get(
                actor.generate_async.remote(
                    _build_prompt(model_path, uid=1), sampling_params
                ),
                timeout=operation_timeout_s,
            )
            timings["first_generate"] = time.perf_counter() - start
            first = _assert_generated(first, "first generate")
            generations["first"] = _tms_generation_summary(first)
            assert generations["first"]["response_ids"] == generations["baseline"]["response_ids"], (
                "Reward output changed after the first TMS wake."
            )
            _record_tms_vllm_memory(report, case_name, "after_first_generate")
        else:
            ray.get(actor.resume_generation.remote(), timeout=operation_timeout_s)
            if verify_generation:
                first = ray.get(
                    actor.generate_async.remote(_build_prompt(model_path, uid=1), sampling_params),
                    timeout=operation_timeout_s,
                )
                timings["first_generate"] = time.perf_counter() - start
                first = _assert_generated(first, "first rollout generate")
                generations["first"] = _tms_generation_summary(first)
            else:
                generations["first"] = {"skipped": "rollout timing probe pulled weights from PS"}

        start = time.perf_counter()
        sleep_ref = actor.nixl_sleep.remote() if probe_role == "rollout" else actor.sleep.remote()
        ray.get(sleep_ref, timeout=operation_timeout_s)
        timings["second_sleep_level2"] = time.perf_counter() - start
        _record_tms_vllm_memory(report, case_name, "after_second_sleep_level2")

        start = time.perf_counter()
        wake_ref = actor.nixl_wake_up.remote() if probe_role == "rollout" else actor.wake_up.remote()
        ray.get(wake_ref, timeout=operation_timeout_s)
        timings["second_wake"] = time.perf_counter() - start
        if probe_role == "rollout":
            pull_start = time.perf_counter()
            ray.get(actor.pull_model_async.remote(), timeout=operation_timeout_s)
            timings["second_pull_from_ps"] = time.perf_counter() - pull_start
            timings["second_wake_and_pull"] = time.perf_counter() - start
            registration_infos = ray.get(actor.get_weight_arena_info.remote(), timeout=operation_timeout_s)
            report.setdefault("nixl_registration", {})["after_second_wake"] = registration_infos
            for info in registration_infos:
                registration = info.get("nixl_registration")
                if registration is None:
                    raise AssertionError(f"Missing NIXL registration timing: {info}")
                registration_samples.append(float(registration["register_memory"]))
            worker_stage_timings["second_wake"] = ray.get(
                actor.get_worker_stage_timing.remote("nixl_register_after_wake_up"),
                timeout=operation_timeout_s,
            )
            if weight_arena_enabled:
                _validate_weight_arena_infos(
                    registration_infos,
                    tp_size=tp_size,
                    expected_node_id=report["ps_probe"]["node_id"],
                    expected_base_addresses=arena_base_addresses,
                )
                report["weight_arena"]["after_second_wake"] = registration_infos
                for info in registration_infos:
                    registration = info["nixl_registration"]
                    arena_count = int(info["stats"]["arena_count"])
                    if int(registration["registered_regions"]) != arena_count:
                        raise AssertionError(f"NIXL regions do not match arenas: {info}")
                if float(np.percentile(registration_samples, 95)) > 2.0:
                    raise AssertionError(
                        "Rollout arena post-wake NIXL register p95 exceeded 2 seconds: "
                        f"{np.percentile(registration_samples, 95):.6f}s"
                    )
        elif weight_arena_enabled:
            arena_infos = ray.get(actor.get_weight_arena_info.remote(), timeout=operation_timeout_s)
            _validate_weight_arena_infos(
                arena_infos,
                tp_size=tp_size,
                expected_node_id=identity["node_id"],
                expected_base_addresses=arena_base_addresses,
            )
            _validate_reward_arena_cache_infos(
                arena_infos,
                expected_inodes=reward_cache_inodes,
            )
            report["weight_arena"]["after_second_wake"] = arena_infos
            worker_stage_timings["second_wake"] = ray.get(
                actor.get_worker_stage_timing.remote("load_weights_from_cpu_cache"),
                timeout=operation_timeout_s,
            )
            for worker_timing in worker_stage_timings["second_wake"]:
                timing = worker_timing.get("timing") or {}
                if timing.get("cache_source") not in ("arena", "node_shared_arena"):
                    raise AssertionError(f"Reward wake did not use the arena CPU cache: {worker_timing}")
                if int(timing.get("arena_restore_bytes", 0)) != int(timing.get("arena_bytes", -1)):
                    raise AssertionError(f"Reward wake restored incomplete arena bytes: {worker_timing}")
                reward_restore_samples.append(float(timing["gpu_copy_elapsed_s"]))
            restore_p95 = float(np.percentile(reward_restore_samples, 95))
            restore_limit_s = float(os.environ.get("PSRL_TMS_VLLM_REWARD_RESTORE_P95_LIMIT_S", "2.0"))
            if restore_p95 > restore_limit_s:
                raise AssertionError(
                    "Reward arena H2D restore p95 exceeded the configured limit: "
                    f"p95={restore_p95:.6f}s limit={restore_limit_s:.6f}s"
                )
        _record_tms_vllm_memory(report, case_name, "after_second_wake")

        start = time.perf_counter()
        if probe_role == "reward":
            second = ray.get(
                actor.generate_async.remote(
                    _build_prompt(model_path, uid=2), sampling_params
                ),
                timeout=operation_timeout_s,
            )
            timings["second_generate"] = time.perf_counter() - start
            second = _assert_generated(second, "second generate")
            generations["second"] = _tms_generation_summary(second)
            assert generations["second"]["response_ids"] == generations["baseline"]["response_ids"], (
                "Reward output changed after the second TMS wake."
            )
            generations["exact_match_across_wakes"] = True
            _record_tms_vllm_memory(report, case_name, "after_second_generate")
        else:
            ray.get(actor.resume_generation.remote(), timeout=operation_timeout_s)
            if verify_generation:
                second = ray.get(
                    actor.generate_async.remote(_build_prompt(model_path, uid=2), sampling_params),
                    timeout=operation_timeout_s,
                )
                timings["second_generate"] = time.perf_counter() - start
                second = _assert_generated(second, "second rollout generate")
                generations["second"] = _tms_generation_summary(second)
                assert generations["second"]["response_ids"] == generations["first"]["response_ids"], (
                    "Deterministic rollout output changed across TMS/NIXL wake cycles."
                )
                generations["exact_match_across_wakes"] = True
            else:
                generations["second"] = {"skipped": "rollout timing probe pulled weights from PS"}

        if probe_role == "reward" and weight_arena_enabled:
            start = time.perf_counter()
            ray.get(actor.sleep.remote(), timeout=operation_timeout_s)
            timings["third_sleep_level2"] = time.perf_counter() - start
            _record_tms_vllm_memory(report, case_name, "after_third_sleep_level2")

            start = time.perf_counter()
            ray.get(actor.wake_up.remote(), timeout=operation_timeout_s)
            timings["third_wake"] = time.perf_counter() - start
            arena_infos = ray.get(actor.get_weight_arena_info.remote(), timeout=operation_timeout_s)
            _validate_weight_arena_infos(
                arena_infos,
                tp_size=tp_size,
                expected_node_id=identity["node_id"],
                expected_base_addresses=arena_base_addresses,
            )
            _validate_reward_arena_cache_infos(
                arena_infos,
                expected_inodes=reward_cache_inodes,
            )
            report["weight_arena"]["after_third_wake"] = arena_infos

            third = ray.get(
                actor.generate_async.remote(_build_prompt(model_path, uid=3), sampling_params),
                timeout=operation_timeout_s,
            )
            third = _assert_generated(third, "third generate")
            generations["third"] = _tms_generation_summary(third)
            assert generations["third"]["response_ids"] == generations["baseline"]["response_ids"], (
                "Reward output changed after the third TMS wake."
            )
            generations["exact_match_across_wakes"] = True
            _record_tms_vllm_memory(report, case_name, "after_third_generate")

        if stress_config["batch_size"] > 0:
            stress_report = {
                "config": stress_config,
                "first_sleep_gpu0_mib": first_sleep_gpu0_mib,
                "cycles": [],
            }
            report["stress_generation"] = stress_report
            stress_sampling_params = {
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "min_tokens": stress_config["min_tokens"],
                "max_tokens": stress_config["max_tokens"],
            }
            for cycle in range(stress_config["cycles"]):
                start = time.perf_counter()
                summaries = _run_tms_stress_generation_batch(
                    actor,
                    model_path=model_path,
                    uid_start=10_000 + cycle * stress_config["batch_size"],
                    batch_size=stress_config["batch_size"],
                    sampling_params=stress_sampling_params,
                    min_tokens=stress_config["min_tokens"],
                    timeout_s=operation_timeout_s,
                )
                timings[f"stress_generate_{cycle}"] = time.perf_counter() - start
                _record_tms_vllm_memory(report, case_name, f"after_stress_generate_{cycle}")

                start = time.perf_counter()
                sleep_ref = actor.nixl_sleep.remote() if probe_role == "rollout" else actor.sleep.remote()
                ray.get(sleep_ref, timeout=operation_timeout_s)
                timings[f"stress_sleep_{cycle}"] = time.perf_counter() - start
                sleep_snapshots = _record_tms_vllm_memory(
                    report,
                    case_name,
                    f"after_stress_sleep_{cycle}",
                )
                sleep_gpu0_mib = _parse_gpu0_used_mib(sleep_snapshots)
                if sleep_gpu0_mib is None:
                    raise RuntimeError(f"Failed to parse GPU0 memory after stress sleep {cycle}")
                sleep_growth_mib = sleep_gpu0_mib - int(first_sleep_gpu0_mib)
                cycle_report = {
                    "cycle": cycle,
                    "num_requests": len(summaries),
                    "output_tokens": [summary["num_output_tokens"] for summary in summaries],
                    "sleep_gpu0_mib": sleep_gpu0_mib,
                    "sleep_growth_mib": sleep_growth_mib,
                }
                stress_report["cycles"].append(cycle_report)
                if sleep_growth_mib > stress_config["sleep_growth_limit_mib"]:
                    raise AssertionError(
                        "TMS sleep memory did not return to its initial baseline after long generation: "
                        f"cycle={cycle} first_sleep={first_sleep_gpu0_mib} MiB "
                        f"current_sleep={sleep_gpu0_mib} MiB growth={sleep_growth_mib} MiB "
                        f"limit={stress_config['sleep_growth_limit_mib']} MiB"
                    )

                if cycle + 1 < stress_config["cycles"]:
                    start = time.perf_counter()
                    wake_ref = (
                        actor.nixl_wake_up.remote()
                        if probe_role == "rollout"
                        else actor.wake_up.remote()
                    )
                    ray.get(wake_ref, timeout=operation_timeout_s)
                    if probe_role == "rollout":
                        ray.get(actor.pull_model_async.remote(), timeout=operation_timeout_s)
                        ray.get(actor.resume_generation.remote(), timeout=operation_timeout_s)
                    timings[f"stress_wake_{cycle + 1}"] = time.perf_counter() - start
                    _record_tms_vllm_memory(
                        report,
                        case_name,
                        f"after_stress_wake_{cycle + 1}",
                    )
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        cleanup_errors = []
        if actor is not None:
            try:
                ray.get(actor.shutdown_rollout_engine.remote(), timeout=180)
            except Exception as exc:
                cleanup_errors.append(f"rollout: {exc!r}")
            ray.kill(actor, no_restart=True)
        if ps_actor is not None:
            try:
                ray.get(ps_actor.shutdown.remote(), timeout=60)
            except Exception as exc:
                cleanup_errors.append(f"ps: {exc!r}")
            ray.kill(ps_actor, no_restart=True)
        if meta_server is not None:
            try:
                ray.get(meta_server.shutdown.remote(), timeout=60)
            except Exception as exc:
                cleanup_errors.append(f"meta_server: {exc!r}")
            ray.kill(meta_server, no_restart=True)
        if ps_control is not None:
            ray.kill(ps_control, no_restart=True)
        if cleanup_errors:
            report["cleanup_errors"] = cleanup_errors
        if worker_stage_timings:
            report["worker_stage_timings"] = worker_stage_timings
        if registration_samples:
            registration_summary = {
                "samples": registration_samples,
                "mean": float(np.mean(registration_samples)),
                "p95": float(np.percentile(registration_samples, 95)),
                "max": float(np.max(registration_samples)),
            }
            report.setdefault("nixl_registration", {})["register_memory_s_summary"] = registration_summary
            if weight_arena_enabled:
                report.setdefault("weight_arena", {})["register_memory_s_summary"] = registration_summary
        if reward_restore_samples:
            report.setdefault("weight_arena", {})["reward_restore_s_summary"] = {
                "samples": reward_restore_samples,
                "mean": float(np.mean(reward_restore_samples)),
                "p95": float(np.percentile(reward_restore_samples, 95)),
                "max": float(np.max(reward_restore_samples)),
            }
        if probe_pg is not None:
            remove_placement_group(probe_pg)
        if actor is not None or probe_pg is not None:
            time.sleep(10)
            _record_tms_vllm_memory(report, case_name, "after_actor_kill")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"TMS vLLM sleep memory report: {report_path}", flush=True)
        if started_ray:
            ray.shutdown()


@pytest.mark.integration
@pytest.mark.skipduringci
@pytest.mark.parametrize(
    ("case_name", "model_env", "relative_model_path", "tp_size"),
    [
        pytest.param(
            "qwen2_5_7b_tp1",
            "PSRL_TMS_VLLM_7B_MODEL",
            "models/Qwen2.5-7B",
            1,
            id="qwen2.5-7b-tp1",
        ),
        pytest.param(
            "qwen2_5_32b_tp4",
            "PSRL_TMS_VLLM_32B_MODEL",
            "models/Qwen2.5-32B",
            4,
            id="qwen2.5-32b-tp4",
        ),
    ],
)
def test_tms_rollout_concurrent_sleep_wake(
    tmp_path: Path,
    case_name: str,
    model_env: str,
    relative_model_path: str,
    tp_size: int,
) -> None:
    """Run coordinator-shaped concurrent sleep/wake/sync waves."""
    if os.environ.get("PSRL_RUN_TMS_ROLLOUT_CONCURRENT_PROBE") != "1":
        pytest.skip("Set PSRL_RUN_TMS_ROLLOUT_CONCURRENT_PROBE=1 to run this probe.")

    model_path = _tms_vllm_model_path(model_env, relative_model_path)
    if not (Path(model_path) / "config.json").exists():
        pytest.skip(f"TMS vLLM sleep model is unavailable: {model_path}")

    num_instances = int(os.environ.get("PSRL_TMS_VLLM_NUM_INSTANCES", "4"))
    if num_instances < 2:
        raise ValueError("Concurrent rollout probe needs at least two instances.")
    cycles = int(os.environ.get("PSRL_TMS_VLLM_CYCLES", "2"))
    if cycles < 1:
        raise ValueError("PSRL_TMS_VLLM_CYCLES must be a positive integer.")
    wake_schedule = os.environ.get(
        "PSRL_TMS_VLLM_WAKE_SCHEDULE", "concurrent"
    )
    if wake_schedule not in {"concurrent", "node_serial"}:
        raise ValueError(
            "PSRL_TMS_VLLM_WAKE_SCHEDULE must be concurrent or node_serial."
        )
    ucx_device_groups_raw = os.environ.get(
        "PSRL_TMS_VLLM_UCX_DEVICE_GROUPS", ""
    ).strip()
    ucx_device_groups = [
        group.strip()
        for group in ucx_device_groups_raw.split("|")
        if group.strip()
    ]
    probe_variant = os.environ.get(
        "PSRL_TMS_VLLM_PROBE_VARIANT",
        f"{wake_schedule}_hca_split" if ucx_device_groups else wake_schedule,
    ).strip()
    if not probe_variant:
        raise ValueError("PSRL_TMS_VLLM_PROBE_VARIANT must not be empty.")

    started_ray = False
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        started_ray = True
    required_gpus = num_instances * tp_size
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus < required_gpus:
        if started_ray:
            ray.shutdown()
        pytest.skip(
            f"{case_name} x{num_instances} needs {required_gpus} GPUs, "
            f"but Ray sees {available_gpus}"
        )

    psrl_config = _build_psrl_config(tmp_path, load_real_weights_at_init=True)
    psrl_config.ps_mode = "nixl_cpu"
    psrl_config.tms.enable_nixl = True
    psrl_config.tms.enable_cuda_graph = True
    weight_arena_enabled = _env_bool("PSRL_TMS_VLLM_WEIGHT_ARENA", False)
    if weight_arena_enabled:
        psrl_config.nixl.weight_arena = _build_tms_weight_arena_config("rollout")
    rollout_config = _build_tms_comparison_reward_config(model_path, tp_size)
    init_timeout_s = int(os.environ.get("PSRL_TMS_VLLM_INIT_TIMEOUT_S", "1800"))
    operation_timeout_s = int(
        os.environ.get("PSRL_TMS_VLLM_OPERATION_TIMEOUT_S", "900")
    )
    report: dict[str, object] = {
        "case": case_name,
        "probe_role": "rollout",
        "model_path": model_path,
        "tp_size": tp_size,
        "num_instances": num_instances,
        "cycles": cycles,
        "required_gpus": required_gpus,
        "weight_arena_enabled": weight_arena_enabled,
        "wake_schedule": wake_schedule,
        "probe_variant": probe_variant,
        "ucx_device_groups": ucx_device_groups,
        "rollout_config": OmegaConf.to_container(rollout_config, resolve=True),
        "setup_timings_s": {},
        "waves": [],
    }
    output_dir_raw = os.environ.get("PSRL_TMS_VLLM_SLEEP_OUTPUT_DIR")
    output_dir = Path(output_dir_raw).expanduser() if output_dir_raw else tmp_path
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / (
        f"tms_rollout_concurrent_{case_name}_x{num_instances}_{probe_variant}.json"
    )

    actors = []
    probe_pgs = []
    meta_server = None
    ps_control = None
    ps_actor = None
    arena_base_addresses: dict[int, dict[int, list[int]]] = {}
    try:
        _record_tms_vllm_memory(report, f"{case_name}_x{num_instances}", "before_actor_start")
        probe_pgs = [
            placement_group(
                [{"CPU": 1, "GPU": 1} for _ in range(tp_size)],
                strategy="STRICT_PACK",
            )
            for _ in range(num_instances)
        ]
        ray.get([pg.ready() for pg in probe_pgs], timeout=init_timeout_s)
        identities = ray.get(
            [
                _probe_node_identity.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        pg,
                        placement_group_bundle_index=0,
                    )
                ).remote()
                for pg in probe_pgs
            ],
            timeout=60,
        )
        report["instance_placement"] = {
            str(instance_id): identity
            for instance_id, identity in enumerate(identities)
        }
        gpu_identity_refs = {
            (instance_id, bundle_idx): _probe_bundle_gpu_identity.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    pg,
                    placement_group_bundle_index=bundle_idx,
                )
            ).remote()
            for instance_id, pg in enumerate(probe_pgs)
            for bundle_idx in range(tp_size)
        }
        gpu_identities = dict(
            zip(
                gpu_identity_refs,
                ray.get(list(gpu_identity_refs.values()), timeout=60),
            )
        )
        instance_gpu_ids = {
            instance_id: sorted(
                gpu_identities[(instance_id, bundle_idx)]["gpu_ids"][0]
                for bundle_idx in range(tp_size)
            )
            for instance_id in range(num_instances)
        }
        report["instance_gpu_ids"] = {
            str(instance_id): gpu_ids
            for instance_id, gpu_ids in instance_gpu_ids.items()
        }
        node_instance_ids: dict[str, list[int]] = {}
        for instance_id, identity in enumerate(identities):
            node_instance_ids.setdefault(identity["node_id"], []).append(instance_id)
        for instance_ids in node_instance_ids.values():
            instance_ids.sort(
                key=lambda instance_id: min(
                    int(float(gpu_id))
                    for gpu_id in instance_gpu_ids[instance_id]
                )
            )
        instance_ucx_devices: dict[int, str] = {}
        if ucx_device_groups:
            max_instances_per_node = max(map(len, node_instance_ids.values()))
            if len(ucx_device_groups) < max_instances_per_node:
                raise ValueError(
                    "PSRL_TMS_VLLM_UCX_DEVICE_GROUPS must provide at least "
                    f"{max_instances_per_node} groups separated by '|', got "
                    f"{len(ucx_device_groups)}."
                )
            for instance_ids in node_instance_ids.values():
                for local_instance_idx, instance_id in enumerate(instance_ids):
                    instance_ucx_devices[instance_id] = ucx_device_groups[
                        local_instance_idx
                    ]
        report["instance_ucx_devices"] = {
            str(instance_id): devices
            for instance_id, devices in instance_ucx_devices.items()
        }
        node_serial_rounds = [
            [instance_ids[round_idx] for instance_ids in node_instance_ids.values() if round_idx < len(instance_ids)]
            for round_idx in range(max(map(len, node_instance_ids.values())))
        ]
        report["node_instance_ids"] = node_instance_ids
        report["node_serial_rounds"] = node_serial_rounds

        from psrl.utils.nixl import GLOBAL_PORT_SCANNER, NIXLInterface
        from psrl.workers.ps import PSStoragePlan, PSStorageWorker

        server_port = ray.get(
            GLOBAL_PORT_SCANNER.find_free_port.remote(host=identities[0]["node_ip"]),
            timeout=60,
        )
        nixl_runtime_config = {
            "server_ip": identities[0]["node_ip"],
            "server_port": server_port,
            "max_pinned_temp_memory_slots": 64,
            "enable_tms_for_temp_buffers": True,
        }
        if weight_arena_enabled:
            nixl_runtime_config["weight_arena"] = OmegaConf.to_container(
                psrl_config.nixl.weight_arena,
                resolve=True,
            )
        psrl_config.nixl = OmegaConf.create(nixl_runtime_config)
        nixl_interface = NIXLInterface(port_scanner=GLOBAL_PORT_SCANNER)
        ps_node_affinity = NodeAffinitySchedulingStrategy(
            node_id=identities[0]["node_id"],
            soft=False,
        )
        meta_server = _NIXLProbeMetaServer.options(
            scheduling_strategy=ps_node_affinity,
        ).remote(psrl_config.nixl, required_gpus + 1)
        ray.get(meta_server.ready.remote(), timeout=60)
        ps_control = _NIXLProbePSControl.options(
            scheduling_strategy=ps_node_affinity,
        ).remote()
        PSActor = ray.remote(num_cpus=0)(PSStorageWorker)
        ps_actor = PSActor.options(
            scheduling_strategy=ps_node_affinity,
            runtime_env={
                "env_vars": {
                    "WORLD_SIZE": "1",
                    "RANK": "0",
                    "PS_NODE_IP": identities[0]["node_ip"],
                }
            },
        ).remote(
            PSStoragePlan(
                train_model_dtype=torch.bfloat16,
                gen_model_dtype=torch.bfloat16,
            ),
            rollout_config.model,
            psrl_config,
            nixl_interface,
        )
        setup_timings = report["setup_timings_s"]
        ps_init_start = time.perf_counter()
        ps_actor.init_nixl_client.remote()
        ps_actor.init_model.remote()
        ps_preload_ref = ps_actor.preload_checkpoint_to_cpu.remote()

        GenActor = ray.remote(max_concurrency=10000)(PSRL_GenWorker)
        for instance_id, (pg, identity) in enumerate(zip(probe_pgs, identities)):
            resources, env_vars, init_kwargs = PSRL_GenWorker.configure_worker(
                config=rollout_config,
                psrl_config=psrl_config,
                num_gpus=1,
                dp_idx=instance_id,
                bundle_indices=list(range(tp_size)),
                role="rollout",
            )
            master_port = ray.get(
                GLOBAL_PORT_SCANNER.find_free_port.remote(host=identity["node_ip"]),
                timeout=60,
            )
            actor_env = {
                **env_vars,
                "WG_BACKEND": "ray",
                "WORLD_SIZE": str(tp_size),
                "RANK": "0",
                "LOCAL_WORLD_SIZE": str(tp_size),
                "LOCAL_RANK": "0",
                "RAY_LOCAL_WORLD_SIZE": str(tp_size),
                "RAY_LOCAL_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(master_port),
                "TOKENIZERS_PARALLELISM": "false",
            }
            if instance_id in instance_ucx_devices:
                actor_env["UCX_NET_DEVICES"] = instance_ucx_devices[instance_id]
                print(
                    "HCA isolation: "
                    f"instance={instance_id} node={identity['node_ip']} "
                    f"gpu_ids={instance_gpu_ids[instance_id]} "
                    f"UCX_NET_DEVICES={instance_ucx_devices[instance_id]}",
                    flush=True,
                )
            actors.append(
                GenActor.options(
                    num_gpus=resources.get("num_gpus", 1),
                    num_cpus=resources.get("num_cpus", 1),
                    runtime_env={"env_vars": actor_env},
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        pg,
                        placement_group_bundle_index=0,
                        placement_group_capture_child_tasks=True,
                    ),
                ).remote(
                    config=rollout_config,
                    role="rollout",
                    psrl_config=psrl_config,
                    gen_interface=GenInterface(
                        rollout_instance_id=instance_id,
                        status_queue=ray.util.queue.Queue(),
                        ps_manager_handle=ps_control,
                    ),
                    nixl_interface=nixl_interface,
                    instance_id=instance_id,
                    reward_model_name="rollout_tms_concurrent_probe",
                    **init_kwargs,
                )
            )

        setup_timings["engine_init_wave"] = _wait_concurrent_refs(
            {
                instance_id: actor.init_model.remote("empty")
                for instance_id, actor in enumerate(actors)
            }
        )
        if weight_arena_enabled:
            arena_infos_by_instance = ray.get(
                [actor.get_weight_arena_info.remote() for actor in actors],
                timeout=operation_timeout_s,
            )
            for instance_id, arena_infos in enumerate(arena_infos_by_instance):
                arena_base_addresses[instance_id] = _validate_weight_arena_infos(
                    arena_infos,
                    tp_size=tp_size,
                    expected_node_id=identities[instance_id]["node_id"],
                )
            report["weight_arena"] = {
                "after_engine_init": {
                    str(instance_id): arena_infos
                    for instance_id, arena_infos in enumerate(arena_infos_by_instance)
                }
            }
        ray.get(ps_preload_ref, timeout=init_timeout_s)
        setup_timings["ps_init_and_checkpoint_preload_wall_s"] = (
            time.perf_counter() - ps_init_start
        )
        _record_tms_vllm_memory(report, f"{case_name}_x{num_instances}", "after_engine_init")

        setup_timings["rollout_nixl_client_init"] = _wait_concurrent_refs(
            {
                instance_id: actor.init_nixl_client.remote()
                for instance_id, actor in enumerate(actors)
            }
        )
        setup_timings["rollout_nixl_convert_params"] = _wait_concurrent_refs(
            {
                instance_id: actor.nixl_convert_params.remote()
                for instance_id, actor in enumerate(actors)
            }
        )
        protocol_start = time.perf_counter()
        ray.get(
            [
                meta_server.protocol.remote(operation_timeout_s),
                ps_actor.nixl_protocol.remote(),
                *[actor.nixl_protocol.remote() for actor in actors],
            ],
            timeout=operation_timeout_s,
        )
        setup_timings["nixl_protocol_wall_s"] = time.perf_counter() - protocol_start
        ps_write_start = time.perf_counter()
        ray.get(
            ps_actor.write_checkpoint_to_registered_tensors.remote(),
            timeout=init_timeout_s,
        )
        setup_timings["ps_write_checkpoint_s"] = time.perf_counter() - ps_write_start
        ps_agent_name = ray.get(ps_actor.get_nixl_agent_name.remote(), timeout=60)
        ps_gen_client_name = ray.get(
            ps_actor.get_nixl_gen_storage_client_name.remote(),
            timeout=60,
        )
        ray.get(
            ps_control.set_endpoints.remote([ps_agent_name], [ps_gen_client_name]),
            timeout=60,
        )
        setup_timings["initial_pull_wave"] = _wait_concurrent_refs(
            {
                instance_id: actor.pull_model_async.remote()
                for instance_id, actor in enumerate(actors)
            }
        )
        _record_tms_vllm_memory(report, f"{case_name}_x{num_instances}", "after_initial_pull")

        for cycle in range(1, cycles + 1):
            wave = {"cycle": cycle}
            wave["sleep"] = _wait_concurrent_refs(
                {
                    instance_id: actor.nixl_sleep.remote()
                    for instance_id, actor in enumerate(actors)
                }
            )
            _record_tms_vllm_memory(
                report,
                f"{case_name}_x{num_instances}",
                f"after_sleep_{cycle}",
            )
            if wake_schedule == "concurrent":
                wave["wake"] = _wait_concurrent_refs(
                    {
                        instance_id: actor.nixl_wake_up.remote()
                        for instance_id, actor in enumerate(actors)
                    }
                )
            else:
                wake_started_at = time.perf_counter()
                round_reports = []
                for round_idx, instance_ids in enumerate(node_serial_rounds):
                    round_report = _wait_concurrent_refs(
                        {
                            instance_id: actors[instance_id].nixl_wake_up.remote()
                            for instance_id in instance_ids
                        }
                    )
                    round_report["round"] = round_idx
                    round_report["instance_ids"] = instance_ids
                    round_reports.append(round_report)
                wave["wake"] = {
                    "wall_s": time.perf_counter() - wake_started_at,
                    "rounds": round_reports,
                }
            wave["sync_with_ps"] = _wait_concurrent_refs(
                {
                    instance_id: actor.sync_with_ps.remote(
                        ps_version=0,
                        interrupt_generation=False,
                        sync_after_wake_up=True,
                    )
                    for instance_id, actor in enumerate(actors)
                }
            )
            wave["wake_and_sync_wall_s"] = (
                wave["wake"]["wall_s"] + wave["sync_with_ps"]["wall_s"]
            )
            if weight_arena_enabled:
                arena_infos_by_instance = ray.get(
                    [actor.get_weight_arena_info.remote() for actor in actors],
                    timeout=operation_timeout_s,
                )
                for instance_id, arena_infos in enumerate(arena_infos_by_instance):
                    _validate_weight_arena_infos(
                        arena_infos,
                        tp_size=tp_size,
                        expected_node_id=identities[instance_id]["node_id"],
                        expected_base_addresses=arena_base_addresses[instance_id],
                    )
                wave["weight_arena_after_wake_and_sync"] = {
                    str(instance_id): arena_infos
                    for instance_id, arena_infos in enumerate(arena_infos_by_instance)
                }
            report["waves"].append(wave)
            _record_tms_vllm_memory(
                report,
                f"{case_name}_x{num_instances}",
                f"after_wake_and_sync_{cycle}",
            )
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        cleanup_errors = []
        if actors:
            try:
                ray.get(
                    [actor.shutdown_rollout_engine.remote() for actor in actors],
                    timeout=180,
                )
            except Exception as exc:
                cleanup_errors.append(f"rollouts: {exc!r}")
            for actor in actors:
                ray.kill(actor, no_restart=True)
        if ps_actor is not None:
            try:
                ray.get(ps_actor.shutdown.remote(), timeout=60)
            except Exception as exc:
                cleanup_errors.append(f"ps: {exc!r}")
            ray.kill(ps_actor, no_restart=True)
        if meta_server is not None:
            try:
                ray.get(meta_server.shutdown.remote(), timeout=60)
            except Exception as exc:
                cleanup_errors.append(f"meta_server: {exc!r}")
            ray.kill(meta_server, no_restart=True)
        if ps_control is not None:
            ray.kill(ps_control, no_restart=True)
        for pg in probe_pgs:
            remove_placement_group(pg)
        if cleanup_errors:
            report["cleanup_errors"] = cleanup_errors
        if actors or probe_pgs:
            time.sleep(10)
            _record_tms_vllm_memory(
                report,
                f"{case_name}_x{num_instances}",
                "after_actor_kill",
            )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Concurrent rollout timing report: {report_path}", flush=True)
        if started_ray:
            ray.shutdown()


_NATIVE_VLLM_ACTOR_ENV = {
    # The plugin is installed in this checkout, so explicitly keep the native
    # vLLM CuMemAllocator implementation active for this probe.
    "PSRL_VLLM_PATCHES": "",
    "TMS_INIT_ENABLE": "0",
    "TMS_INIT_ENABLE_CPU_BACKUP": "0",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_RAY_PER_WORKER_GPUS": "1",
    "VLLM_RAY_BUNDLE_INDICES": "",
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
    "TOKENIZERS_PARALLELISM": "false",
}


@ray.remote(num_cpus=1, runtime_env={"env_vars": _NATIVE_VLLM_ACTOR_ENV})
class _NativeVLLMSleepProbe:
    def __init__(self):
        self.llm = None

    def initialize(self, engine_config: dict) -> dict[str, str]:
        if os.environ.get("PSRL_VLLM_PATCHES"):
            raise RuntimeError(
                "Native vLLM sleep probe must run without PSRL_VLLM_PATCHES; "
                f"got {os.environ['PSRL_VLLM_PATCHES']!r}."
            )

        import vllm
        from vllm import LLM
        from vllm.v1.worker.gpu_worker import Worker

        self.llm = LLM(**engine_config)
        sleep_impl = f"{Worker.sleep.__module__}.{Worker.sleep.__qualname__}"
        if Worker.sleep.__module__ != "vllm.v1.worker.gpu_worker":
            raise RuntimeError(f"Worker.sleep is patched, not native vLLM: {sleep_impl}")
        return {"vllm_version": vllm.__version__, "sleep_impl": sleep_impl}

    def sleep_level2(self) -> float:
        assert self.llm is not None
        start = time.perf_counter()
        self.llm.sleep(level=2)
        return time.perf_counter() - start

    def wake_weights(self) -> float:
        assert self.llm is not None
        start = time.perf_counter()
        self.llm.wake_up(tags=["weights"])
        return time.perf_counter() - start

    def reload_weights(self) -> float:
        assert self.llm is not None
        start = time.perf_counter()
        self.llm.collective_rpc("reload_weights")
        return time.perf_counter() - start

    def wake_kv_cache(self) -> float:
        assert self.llm is not None
        start = time.perf_counter()
        self.llm.wake_up(tags=["kv_cache"])
        return time.perf_counter() - start

    def generate(self, max_tokens: int) -> dict[str, object]:
        assert self.llm is not None
        from vllm import SamplingParams

        start = time.perf_counter()
        outputs = self.llm.generate(
            ["Hello, please answer with one short sentence."],
            SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens),
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - start
        output = outputs[0].outputs[0]
        return {
            "elapsed_s": elapsed,
            "text": output.text,
            "num_output_tokens": len(output.token_ids),
        }


def _native_vllm_model_path(env_name: str, relative_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return os.environ.get(env_name, str(repo_root / relative_path))


def _native_vllm_engine_config(model_path: str, tp_size: int) -> dict[str, object]:
    max_model_len = int(os.environ.get("PSRL_NATIVE_VLLM_MAX_MODEL_LEN", "16384"))
    max_num_batched_tokens = int(
        os.environ.get("PSRL_NATIVE_VLLM_MAX_NUM_BATCHED_TOKENS", str(max_model_len))
    )
    return {
        "model": model_path,
        "enable_sleep_mode": True,
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": 1,
        "distributed_executor_backend": "ray",
        "dtype": "bfloat16",
        "gpu_memory_utilization": float(os.environ.get("PSRL_NATIVE_VLLM_GPU_MEMORY_UTILIZATION", "0.7")),
        "enforce_eager": _env_bool("PSRL_NATIVE_VLLM_ENFORCE_EAGER", False),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": int(os.environ.get("PSRL_NATIVE_VLLM_MAX_NUM_SEQS", "64")),
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "disable_custom_all_reduce": True,
        "trust_remote_code": False,
        "load_format": "auto",
        "disable_log_stats": True,
    }


def _record_native_vllm_memory(
    report: dict[str, object], case_name: str, stage: str
) -> None:
    snapshots = _print_cluster_gpu_memory(f"native_{case_name}_{stage}")
    report.setdefault("memory_snapshots", {})[stage] = snapshots


@pytest.mark.integration
@pytest.mark.skipduringci
@pytest.mark.parametrize(
    ("case_name", "model_env", "relative_model_path", "tp_size"),
    [
        pytest.param(
            "qwen2_5_7b_tp1",
            "PSRL_NATIVE_VLLM_7B_MODEL",
            "models/Qwen2.5-7B",
            1,
            id="qwen2.5-7b-tp1",
        ),
        pytest.param(
            "qwen2_5_32b_tp4",
            "PSRL_NATIVE_VLLM_32B_MODEL",
            "models/Qwen2.5-32B",
            4,
            id="qwen2.5-32b-tp4",
        ),
    ],
)
def test_native_vllm_level2_sleep_memory(
    tmp_path: Path,
    case_name: str,
    model_env: str,
    relative_model_path: str,
    tp_size: int,
):
    """Measure native vLLM level-2 sleep over two wake/generate cycles.

    The 7B case uses TP1 and the 32B case uses TP4. Each wake follows the
    native level-2 contract: allocate weights, reload them from disk, then
    allocate KV cache before inference.
    """
    if os.environ.get("PSRL_RUN_NATIVE_VLLM_SLEEP_SMOKE") != "1":
        pytest.skip(
            "Set PSRL_RUN_NATIVE_VLLM_SLEEP_SMOKE=1 to run native vLLM sleep probes."
        )

    model_path = _native_vllm_model_path(model_env, relative_model_path)
    if not (Path(model_path) / "config.json").exists():
        pytest.skip(f"Native vLLM sleep model is unavailable: {model_path}")

    started_ray = False
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        started_ray = True
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus < tp_size:
        if started_ray:
            ray.shutdown()
        pytest.skip(f"{case_name} needs {tp_size} GPUs, but Ray sees {available_gpus}")

    engine_config = _native_vllm_engine_config(model_path, tp_size)
    max_tokens = int(os.environ.get("PSRL_NATIVE_VLLM_MAX_TOKENS", "32"))
    init_timeout_s = int(os.environ.get("PSRL_NATIVE_VLLM_INIT_TIMEOUT_S", "1800"))
    operation_timeout_s = int(os.environ.get("PSRL_NATIVE_VLLM_OPERATION_TIMEOUT_S", "900"))
    report: dict[str, object] = {
        "case": case_name,
        "model_path": model_path,
        "tp_size": tp_size,
        "engine_config": engine_config,
        "timings_s": {},
        "generations": {},
    }

    output_dir_raw = os.environ.get("PSRL_NATIVE_VLLM_SLEEP_OUTPUT_DIR")
    output_dir = Path(output_dir_raw).expanduser() if output_dir_raw else tmp_path
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"native_vllm_sleep_{case_name}.json"

    actor = None
    probe_pg = None
    try:
        _record_native_vllm_memory(report, case_name, "before_actor_start")
        probe_pg = placement_group(
            [{"CPU": 1}] + [{"GPU": 1} for _ in range(tp_size)],
            strategy="STRICT_PACK",
        )
        ray.get(probe_pg.ready(), timeout=init_timeout_s)
        actor = _NativeVLLMSleepProbe.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                probe_pg,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=True,
            )
        ).remote()
        native_info = ray.get(actor.initialize.remote(engine_config), timeout=init_timeout_s)
        report["native_vllm"] = native_info
        print(f"Native vLLM probe implementation: {native_info}", flush=True)
        _record_native_vllm_memory(report, case_name, "after_engine_init")

        timings = report["timings_s"]
        generations = report["generations"]

        timings["first_sleep_level2"] = ray.get(
            actor.sleep_level2.remote(), timeout=operation_timeout_s
        )
        _record_native_vllm_memory(report, case_name, "after_first_sleep_level2")

        timings["first_wake_weights"] = ray.get(
            actor.wake_weights.remote(), timeout=operation_timeout_s
        )
        _record_native_vllm_memory(report, case_name, "after_first_weights_reallocated")
        timings["first_reload_weights"] = ray.get(
            actor.reload_weights.remote(), timeout=operation_timeout_s
        )
        _record_native_vllm_memory(report, case_name, "after_first_weights_reloaded")
        timings["first_wake_kv_cache"] = ray.get(
            actor.wake_kv_cache.remote(), timeout=operation_timeout_s
        )
        _record_native_vllm_memory(report, case_name, "after_first_kv_cache_reallocated")

        first_generation = ray.get(
            actor.generate.remote(max_tokens), timeout=operation_timeout_s
        )
        assert first_generation["num_output_tokens"] > 0
        generations["first"] = first_generation
        _record_native_vllm_memory(report, case_name, "after_first_generate")

        timings["second_sleep_level2"] = ray.get(
            actor.sleep_level2.remote(), timeout=operation_timeout_s
        )
        _record_native_vllm_memory(report, case_name, "after_second_sleep_level2")

        timings["second_wake_weights"] = ray.get(
            actor.wake_weights.remote(), timeout=operation_timeout_s
        )
        _record_native_vllm_memory(report, case_name, "after_second_weights_reallocated")
        timings["second_reload_weights"] = ray.get(
            actor.reload_weights.remote(), timeout=operation_timeout_s
        )
        _record_native_vllm_memory(report, case_name, "after_second_weights_reloaded")
        timings["second_wake_kv_cache"] = ray.get(
            actor.wake_kv_cache.remote(), timeout=operation_timeout_s
        )
        _record_native_vllm_memory(report, case_name, "after_second_kv_cache_reallocated")

        second_generation = ray.get(
            actor.generate.remote(max_tokens), timeout=operation_timeout_s
        )
        assert second_generation["num_output_tokens"] > 0
        generations["second"] = second_generation
        _record_native_vllm_memory(report, case_name, "after_second_generate")
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        if actor is not None:
            ray.kill(actor, no_restart=True)
        if probe_pg is not None:
            remove_placement_group(probe_pg)
        if actor is not None or probe_pg is not None:
            time.sleep(10)
            _record_native_vllm_memory(report, case_name, "after_actor_kill")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Native vLLM sleep memory report: {report_path}", flush=True)
        if started_ray:
            ray.shutdown()
