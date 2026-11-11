"""
Verify model overrides reach vLLM without constructing a GPU engine.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf
from psrl.workers.config import RolloutConfig
from psrl.workers.gen import vllm_async_server as server_module
from transformers import Glm4Config, Qwen3Config
from verl.workers.config.model import HFModelConfig
from vllm.config import ModelConfig

pytestmark = pytest.mark.cpu_test


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_type", "architecture", "factor", "theta"),
    [
        (Qwen3Config, "Qwen3ForCausalLM", 4.0, 1000000.0),
        (Glm4Config, "Glm4ForCausalLM", 1.25, 10000.0),
    ],
)
async def test_model_overrides_reach_vllm(tmp_path, monkeypatch, config_type, architecture, factor, theta):
    config_type(
        architectures=[architecture],
        max_position_embeddings=32768,
        rope_theta=theta,
    ).save_pretrained(tmp_path)
    max_model_len = int(32768 * factor)
    model = HFModelConfig(
        path=str(tmp_path),
        load_tokenizer=False,
        override_config={
            "max_position_embeddings": max_model_len,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": 32768,
            },
        },
    )
    server = object.__new__(server_module.PSRL_vLLMHttpServer)
    server.model_config = model
    server.config = RolloutConfig(
        max_model_len=max_model_len,
        enable_sleep_mode=True,
        engine_kwargs={"vllm": {"hf_overrides": {"eos_token_id": 42}}},
    )
    server.node_rank = 0
    server.replica_rank = 1
    server.nnodes = 1
    server.gen_interface = SimpleNamespace(rollout_replica_idx=0)
    server.psrl_config = OmegaConf.create({})
    server.profiler_controller = SimpleNamespace(config=None, tool_config=None)
    server._build_kv_cache_manager = Mock(return_value=SimpleNamespace(get_engine_kwargs=lambda: {}))
    server._build_kv_events_args = Mock(return_value={})
    server._tms_weights_cpu_backup_enabled = Mock(return_value=False)
    server._is_reward_model = Mock(return_value=False)
    server._apply_quantization = Mock(return_value=(None, {"quantization_config": {"quant_method": "test"}}))
    monkeypatch.setattr(server_module, "build_vllm_profiler_args", lambda *args: {})
    captured = {}

    class ArgumentsCaptured(Exception):
        pass

    def capture_arguments(args):
        captured.update(args)
        raise ArgumentsCaptured

    monkeypatch.setattr(server_module, "build_cli_args_from_config", capture_arguments)

    server._validate_configs()
    with pytest.raises(ArgumentsCaptured):
        await server.launch_server()

    overrides = captured["hf_overrides"]
    assert overrides["max_position_embeddings"] == max_model_len, "vLLM must receive the model context override."
    assert overrides["rope_parameters"]["rope_theta"] == theta, "Checkpoint RoPE fields must be preserved."
    assert overrides["eos_token_id"] == 42, "Explicit engine overrides must reach vLLM."
    assert overrides.pop("quantization_config") == {"quant_method": "test"}, (
        "Quantization overrides must coexist with model and engine overrides."
    )
    vllm_model = ModelConfig(model=str(tmp_path), hf_overrides=overrides, max_model_len=max_model_len)
    assert vllm_model.max_model_len == max_model_len, "vLLM must accept the extended context limit."
    assert vllm_model.hf_text_config.rope_parameters["rope_type"] == "yarn", "vLLM must use YaRN."
    assert vllm_model.hf_text_config.rope_parameters["factor"] == factor, "vLLM must use the configured factor."
