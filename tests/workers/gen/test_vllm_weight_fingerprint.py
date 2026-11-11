"""
Verify rollout pulls forward and persist the configured native weight fingerprints.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf
from psrl.workers.gen import vllm_async_server as server_module

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def rollout_pull(monkeypatch):
    server = object.__new__(server_module.PSRL_vLLMHttpServer)
    server.psrl_config = OmegaConf.create(
        {
            "ps_mode": "nixl_gpu",
            "profile": {"fix_weight": False},
            "nixl": {
                "weight_verification": {
                    "enable": True,
                    "transfer_chain": True,
                    "sample_every_n_versions": 2,
                    "full_every_n_versions": 4,
                    "sample_count_per_tensor": 11,
                    "full_hash_chunk_bytes": 1024,
                    "include_tensor_digests": True,
                    "fail_on_mismatch": False,
                }
            },
        }
    )
    ps_manager = SimpleNamespace(
        get_ps_model_version=SimpleNamespace(remote=AsyncMock(return_value=2)),
        get_ps_nixl_agent_names=SimpleNamespace(remote=AsyncMock(return_value=["ps_agent"])),
        get_ps_nixl_gen_storage_client_names=SimpleNamespace(remote=AsyncMock(return_value=["ps_client"])),
        pull_model_state_dict_nixl=SimpleNamespace(remote=AsyncMock()),
    )
    server.gen_interface = SimpleNamespace(ps_manager_handle=ps_manager)
    server._cached_ps_nixl_agent_names = None
    server._cached_ps_nixl_gen_storage_client_names = None
    records = [
        {
            "model_version": 2,
            "raw_fingerprint": {"stage": "rollout_after_raw_pull", "model_version": 2, "rank": rank},
            "final_fingerprint": {"stage": "rollout_after_param_sync", "model_version": 2, "rank": rank},
        }
        for rank in range(2)
    ]
    server.engine = SimpleNamespace(collective_rpc=AsyncMock(return_value=records))
    server.collective_rpc = AsyncMock()
    server.clear_kv_cache = AsyncMock()
    server.kv_cache_manager = None
    server.base_worker_id = 3
    server.get_instance_num = lambda: 2
    logger = Mock()
    monkeypatch.setattr(server_module, "psrl_logger", logger)
    return SimpleNamespace(server=server, ps_manager=ps_manager, records=records, logger=logger)


@pytest.mark.parametrize("model_version,mode", [(2, "sampled"), (4, "full")])
async def test_pull_forwards_native_fingerprint_options_and_logs_each_rank(rollout_pull, model_version, mode):
    pull = rollout_pull
    pull.ps_manager.get_ps_model_version.remote.return_value = model_version
    for result in pull.records:
        result["model_version"] = model_version
        result["raw_fingerprint"]["model_version"] = model_version
        result["final_fingerprint"]["model_version"] = model_version

    await pull.server.nixl_pull_model()

    pull.ps_manager.get_ps_model_version.remote.assert_awaited_once_with(debug_info="rollout_weight_fingerprint")
    pull.server.engine.collective_rpc.assert_awaited_once_with(
        "nixl_pull_model_core",
        args=(
            ["ps_agent"],
            ["ps_client"],
            model_version,
            {
                "mode": mode,
                "sample_count": 11,
                "chunk_bytes": 1024,
                "include_tensor_digests": True,
                "fail_on_mismatch": False,
            },
        ),
    )
    logged = [
        json.loads(call.args[0].removeprefix("[WEIGHT_FINGERPRINT] ")) for call in pull.logger.warning.call_args_list
    ]
    expected = [result[key] for result in pull.records for key in ("raw_fingerprint", "final_fingerprint")]
    assert logged == expected, "Every rank must persist its raw and transformed fingerprints unchanged."


@pytest.mark.parametrize("disabled", ["missing", "enable", "transfer_chain"])
async def test_disabled_fingerprints_preserve_the_pull_without_extra_rpc(rollout_pull, monkeypatch, disabled):
    pull = rollout_pull
    if disabled == "missing":
        del pull.server.psrl_config.nixl.weight_verification
    else:
        pull.server.psrl_config.nixl.weight_verification[disabled] = False
    resolver = Mock(side_effect=AssertionError("Disabled fingerprinting must not resolve hash options."))
    monkeypatch.setattr(server_module, "resolve_weight_fingerprint_options", resolver)
    pull.server.engine.collective_rpc.return_value = None

    await pull.server.nixl_pull_model()

    pull.server.engine.collective_rpc.assert_awaited_once_with(
        "nixl_pull_model_core", args=(["ps_agent"], ["ps_client"])
    )
    pull.ps_manager.get_ps_model_version.remote.assert_not_awaited()
    resolver.assert_not_called()
    pull.logger.warning.assert_not_called()


async def test_unsampled_version_pulls_once_without_fingerprint_logs(rollout_pull):
    pull = rollout_pull
    pull.ps_manager.get_ps_model_version.remote.return_value = 3
    pull.server.engine.collective_rpc.return_value = None

    await pull.server.nixl_pull_model()

    pull.server.engine.collective_rpc.assert_awaited_once_with(
        "nixl_pull_model_core", args=(["ps_agent"], ["ps_client"], 3, None)
    )
    pull.logger.warning.assert_not_called()


async def test_fingerprints_do_not_repeat_weight_pull_or_finalize(rollout_pull):
    pull = rollout_pull

    await pull.server.pull_model()

    pull.server.engine.collective_rpc.assert_awaited_once()
    pull.server.collective_rpc.assert_awaited_once_with(method="finalize_weight_loading")
    pull.ps_manager.pull_model_state_dict_nixl.remote.assert_awaited_once_with([(3, 0), (3, 1)])
    pull.server.clear_kv_cache.assert_awaited_once_with()


async def test_fixed_weights_skip_the_pull_and_fingerprint_query(rollout_pull):
    pull = rollout_pull
    pull.server.psrl_config.profile.fix_weight = True

    await pull.server.pull_model()

    pull.server.engine.collective_rpc.assert_not_awaited()
    pull.server.collective_rpc.assert_not_awaited()
    pull.ps_manager.get_ps_model_version.remote.assert_not_awaited()
    pull.logger.warning.assert_not_called()
