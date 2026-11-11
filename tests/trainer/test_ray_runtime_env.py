import pytest
from omegaconf import OmegaConf
from psrl.trainer.constants_ppo import get_ppo_ray_init_kwargs, get_ppo_ray_runtime_env

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("value", [None, "1", "2"])
def test_cuda_connection_limit_reaches_ray_workers(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    else:
        monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", value)

    env_vars = get_ppo_ray_runtime_env()["env_vars"]

    assert env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == (value or "1"), (
        "CUDA connection settings must be explicit because Ray workers run outside the driver environment."
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("UCX_TLS", "tcp,cuda_copy,cuda_ipc"),
        ("UCX_NET_DEVICES", "eth0"),
        ("UCX_MAX_RMA_RAILS", "1"),
        ("UCX_LOG_LEVEL", "debug"),
        ("NIXL_PLUGIN_DIR", "/test/nixl/plugins"),
        ("LD_LIBRARY_PATH", "/test/nixl/lib:/test/ucx/lib"),
        ("PYTHONPATH", "/test/nixl/python"),
    ],
)
def test_driver_communication_environment_is_explicit_in_ray_runtime_env(monkeypatch, key, value):
    monkeypatch.setenv(key, value)

    env_vars = get_ppo_ray_runtime_env()["env_vars"]

    assert env_vars[key] == value, f"Driver communication setting {key!r} must reach Ray workers."


def test_unset_ucx_transports_are_not_overridden(monkeypatch):
    monkeypatch.delenv("UCX_TLS", raising=False)

    assert "UCX_TLS" not in get_ppo_ray_runtime_env()["env_vars"], "Ray must preserve UCX transport auto-selection."


def test_ucx_overrides_do_not_change_defaults_for_subsequent_jobs(monkeypatch):
    monkeypatch.setenv("UCX_TLS", "tcp")
    get_ppo_ray_runtime_env()
    monkeypatch.delenv("UCX_TLS")

    assert "UCX_TLS" not in get_ppo_ray_runtime_env()["env_vars"], (
        "A job transport override must not alter subsequent job defaults."
    )


def test_unrelated_driver_environment_is_not_copied(monkeypatch):
    monkeypatch.setenv("PSRL_UNRELATED_TEST_SETTING", "private")

    assert "PSRL_UNRELATED_TEST_SETTING" not in get_ppo_ray_runtime_env()["env_vars"], (
        "Runtime environment propagation must remain limited to explicit worker settings."
    )


@pytest.mark.parametrize("transfer_queue_enabled", [False, True])
def test_ray_init_options_accept_immutable_hydra_runtime_overrides(monkeypatch, transfer_queue_enabled):
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "2")
    config = OmegaConf.create(
        {
            "num_cpus": None,
            "runtime_env": {
                "env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "per_process_memory_fraction:0.85", "NCCL_DEBUG": "WARN"},
                "excludes": ["outputs"],
            },
        }
    )
    OmegaConf.set_struct(config, True)
    OmegaConf.set_readonly(config, True)
    original = OmegaConf.to_container(config, resolve=True)

    result = get_ppo_ray_init_kwargs(config, transfer_queue_enabled=transfer_queue_enabled)

    env_vars = result["runtime_env"]["env_vars"]
    assert env_vars["PYTORCH_CUDA_ALLOC_CONF"] == "per_process_memory_fraction:0.85", (
        "The allocator budget must survive runtime environment assembly."
    )
    assert env_vars["NCCL_DEBUG"] == "WARN", "Explicit runtime settings must override driver defaults."
    assert env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == "2", "Runtime overrides must preserve driver worker settings."
    assert (env_vars.get("TRANSFER_QUEUE_ENABLE") == "1") == transfer_queue_enabled, (
        "Transfer queue configuration must control its worker environment flag."
    )
    assert result["runtime_env"]["excludes"] == ["outputs"], "Ray runtime options must survive environment assembly."
    assert result["num_cpus"] is None, "Ray initialization options must survive environment assembly."
    assert OmegaConf.to_container(config, resolve=True) == original, "Ray initialization must not mutate Hydra config."


def test_ray_init_options_resolve_references_to_parent_configuration():
    config = OmegaConf.create({"ray_cpus": 2, "ray_kwargs": {"ray_init": {"num_cpus": "${ray_cpus}"}}})
    OmegaConf.set_struct(config, True)
    OmegaConf.set_readonly(config, True)

    result = get_ppo_ray_init_kwargs(config.ray_kwargs.ray_init, transfer_queue_enabled=True)

    assert result["num_cpus"] == 2, "Ray initialization must resolve interpolations before detaching config nodes."


@pytest.mark.parametrize(
    "key,value",
    [
        ("WANDB_MODE", "offline"),
        ("WANDB_RUN_ID", "resume-run"),
        ("WANDB_RESUME", "must"),
        ("WANDB_ENTITY", "training-team"),
    ],
)
def test_tracking_identity_reaches_the_task_runner_runtime(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    result = get_ppo_ray_init_kwargs(OmegaConf.create({}), transfer_queue_enabled=True)
    assert result["runtime_env"]["env_vars"][key] == value, (
        "The TaskRunner must receive the driver's tracking identity before initializing its logger."
    )


@pytest.mark.parametrize("key", ["WANDB_MODE", "WANDB_RUN_ID", "WANDB_RESUME", "WANDB_ENTITY"])
def test_tracking_identity_is_not_retained_for_another_driver(monkeypatch, key):
    monkeypatch.setenv(key, "first-run")
    get_ppo_ray_runtime_env()
    monkeypatch.delenv(key)
    assert key not in get_ppo_ray_runtime_env()["env_vars"], "A later job must not inherit an earlier run identity."


def test_explicit_runtime_tracking_settings_override_driver_identity(monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "driver-run")
    config = OmegaConf.create({"runtime_env": {"env_vars": {"WANDB_RUN_ID": "configured-run"}}})
    result = get_ppo_ray_init_kwargs(config, transfer_queue_enabled=True)
    assert result["runtime_env"]["env_vars"]["WANDB_RUN_ID"] == "configured-run", (
        "An explicit runtime configuration remains authoritative."
    )
