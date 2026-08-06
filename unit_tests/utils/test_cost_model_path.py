from pathlib import Path

from psrl.utils.cost_model_path import model_name_to_cost_model_stems, resolve_cost_model_json_path


def test_qwen2_5_32b_uses_qwen_32b_cost_model():
    assert model_name_to_cost_model_stems("Qwen2.5-32B")[0] == "qwen_32b"
    assert model_name_to_cost_model_stems("Qwen/Qwen2.5-32B")[0] == "qwen_32b"

    cost_model_dir = Path("psrl/trainer/config/cost_model")
    resolved = resolve_cost_model_json_path(str(cost_model_dir), "Qwen2.5-32B")

    assert resolved == str(cost_model_dir / "qwen_32b.json")


def test_other_qwen2_5_sizes_keep_their_family():
    assert model_name_to_cost_model_stems("Qwen2.5-7B")[0] == "qwen2.5_7b"


def test_qwen3_30b_thinking_2507_uses_its_model_specific_cost_model():
    cost_model_dir = Path("psrl/trainer/config/cost_model")
    resolved = resolve_cost_model_json_path(
        str(cost_model_dir), "Qwen3-30B-A3B-Thinking-2507"
    )

    assert resolved == str(cost_model_dir / "qwen3_30b_a3b_thinking_2507.json")
