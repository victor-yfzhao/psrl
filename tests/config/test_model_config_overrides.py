from types import SimpleNamespace

import pytest
from psrl.workers.config.model import HFModelConfig
from transformers import Qwen3Config
from verl.utils.model import update_model_config
from verl.workers.config.model import HFModelConfig as VerlHFModelConfig

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("config_type", [dict, SimpleNamespace])
def test_overrides_merge_nested_objects_and_rope_dictionaries(config_type):
    """
    Preserve unspecified model fields when overriding dictionary RoPE parameters.
    """
    rope = {"rope_type": "default", "rope_theta": 1000000.0}
    nested = SimpleNamespace(hidden_size=64, rope_scaling=rope)
    config = config_type(text_config=nested, max_position_embeddings=32768)

    update_model_config(
        config,
        {
            "text_config": {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
            "max_position_embeddings": 131072,
        },
    )

    assert nested.hidden_size == 64, "Unspecified nested model fields must be preserved."
    assert nested.rope_scaling == {"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 4.0}, (
        "RoPE overrides must preserve unspecified dictionary fields."
    )
    position_limit = config["max_position_embeddings"] if config_type is dict else config.max_position_embeddings
    assert position_limit == 131072, "The configured context limit must be updated."


@pytest.mark.parametrize("config_type", [dict, SimpleNamespace])
@pytest.mark.parametrize("initial", [{}, {"rope_scaling": None}])
def test_overrides_initialize_missing_or_null_dictionary_fields(config_type, initial):
    """
    Allow dictionary overrides when the checkpoint has no configured RoPE scaling.
    """
    config = config_type(**initial)

    update_model_config(config, {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}})

    rope = config["rope_scaling"] if config_type is dict else config.rope_scaling
    assert rope == {"rope_type": "yarn", "factor": 4.0}, "RoPE scaling must be initialized from the override."


@pytest.mark.parametrize("config_type", [VerlHFModelConfig, HFModelConfig])
def test_hf_model_config_applies_qwen3_yarn_overrides(tmp_path, config_type):
    """
    Construct the effective model configuration without loading model weights.
    """
    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        max_position_embeddings=32768,
        rope_theta=1000000.0,
    ).save_pretrained(tmp_path)
    rope = {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
        "rope_theta": 1000000.0,
    }

    model = config_type(
        path=str(tmp_path),
        load_tokenizer=False,
        override_config={"max_position_embeddings": 131072, "rope_scaling": rope},
    )

    assert model.hf_config.max_position_embeddings == 131072, "The model context limit must match the override."
    assert all(model.hf_config.rope_scaling[key] == value for key, value in rope.items()), (
        "The HF model configuration must retain every YaRN parameter."
    )
