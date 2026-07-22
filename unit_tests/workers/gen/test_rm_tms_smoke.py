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
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tensordict import TensorDict
from transformers import AutoTokenizer
from verl import DataProto

from psrl.workers.gen.gen_worker import GenInterface, PSRL_GenWorker

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


def _print_cluster_gpu_memory(stage: str) -> None:
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


def _free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return str(sock.getsockname()[1])


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
    # elastic_rm.enable in gen_worker only gates whether init_model loads GPU weights
    # after CPU preload. When load_real_weights_at_init=True we set it False so init
    # loads real weights, while all other rollout/vLLM settings stay on the prod profile.
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
            "raw_prompt_ids": raw_prompt_ids,
            "raw_response_ids": raw_response_ids,
        },
    )


def _assert_generated(output: DataProto, stage: str) -> None:
    assert output is not None, f"{stage} returned None"
    assert "raw_response_ids" in output.non_tensor_batch, f"{stage} did not return raw_response_ids"
    assert len(output.non_tensor_batch["raw_response_ids"]) == 1


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
