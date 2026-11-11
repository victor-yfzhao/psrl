"""
Exercise text and visual position IDs with the installed Qwen RoPE implementation.
"""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import psrl.workers.agent_loop.worker as worker_module
import pytest
import torch
from psrl.workers.gen.utils import TokenOutput
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from verl.utils import tensordict_utils as tu

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def rope_worker():
    cls = worker_module.PSRL_AgentLoopWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    worker.processor = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)),
        image_token_id=650,
        video_token_id=651,
    )
    worker.processor.get_rope_index = MethodType(Qwen3VLModel.get_rope_index, worker.processor)
    worker.processor.get_vision_position_ids = MethodType(Qwen3VLModel.get_vision_position_ids, worker.processor)
    worker.overlong_filtering = False
    return worker


@pytest.mark.parametrize("input_ids", [[[10, 650, 11]], [[10, 651, 11]], [[650, 651, 650]]])
def test_visual_token_ids_without_media_have_text_positions(rope_worker, input_ids):
    ids = torch.tensor(input_ids)
    positions = rope_worker._compute_position_ids(
        ids,
        torch.ones_like(ids),
        {"mm_token_type_ids": torch.ones_like(ids)},
        torch.tensor([[False, True, True]]),
    )

    assert positions.shape == (1, 4, 3), "Text trajectories must retain all four Qwen position planes."
    assert positions.tolist() == [[[0, 1, 2]] * 4], "Visual token IDs without media must advance as text."


def test_text_positions_respect_padding(rope_worker):
    ids = torch.tensor([[0, 10, 650, 651, 0]])
    attention = torch.tensor([[0, 1, 1, 1, 0]])
    positions = rope_worker._compute_position_ids(ids, attention, {}, torch.zeros_like(ids, dtype=torch.bool))

    assert positions.tolist() == [[[0, 0, 1, 2, 2]] * 4], "Padding must not advance any position plane."


@pytest.mark.parametrize("modality,token_id", [("image", 650), ("video", 651)])
def test_real_media_preserves_visual_positions_and_generated_tokens_remain_text(rope_worker, modality, token_id):
    ids = torch.tensor([[10, token_id, token_id, token_id, token_id, 11, 650, 651]])
    generation_mask = torch.tensor([[False, False, False, False, False, False, True, True]])
    media = {f"{modality}_grid_thw": torch.tensor([[1, 4, 4]]), "mm_token_type_ids": torch.zeros_like(ids)}
    positions = rope_worker._compute_position_ids(ids, torch.ones_like(ids), media, generation_mask)

    assert positions.shape == (1, 4, 8), "Visual positions must stay aligned to the original input tokens."
    assert positions[0, 0].tolist() == list(range(8)), "The text plane must retain its original token positions."
    assert positions[0, 2, 1:5].tolist() == [1, 1, 2, 2], "Image and video rows must use their spatial grid."
    assert positions[0, 3, 1:5].tolist() == [1, 2, 1, 2], "Image and video columns must use their spatial grid."
    assert positions[0, 1:, -2:].tolist() == [[4, 5]] * 3, "Generated visual IDs must advance as text."


def test_invalid_visual_grid_does_not_silently_use_text_positions(rope_worker):
    ids = torch.tensor([[10, 650, 650, 11]])
    media = {"image_grid_thw": torch.tensor([[1, 4, 4]]), "mm_token_type_ids": torch.zeros_like(ids)}

    with pytest.raises(RuntimeError):
        rope_worker._compute_position_ids(ids, torch.ones_like(ids), media, torch.zeros_like(ids, dtype=torch.bool))


@pytest.mark.parametrize("modality", ["image", "video"])
def test_supplied_media_requires_visual_grid_metadata(rope_worker, monkeypatch, modality):
    rope_worker.tokenizer = SimpleNamespace(decode=Mock(return_value="prompt"))
    monkeypatch.setattr(worker_module, "build_multimodal_processor_inputs", Mock(return_value=BatchFeature({})))
    output = TokenOutput(
        prompt_ids=[10], response_ids=[11], response_mask=[1], multi_modal_data={f"{modality}s": [object()]}
    )

    with pytest.raises(ValueError, match=f"Missing {modality}_grid_thw"):
        rope_worker._compute_multi_modal_inputs(output, torch.tensor([10, 11]))


def test_text_output_keeps_tokens_masks_and_logprobs_without_calling_the_processor(rope_worker):
    rope_worker.tokenizer = SimpleNamespace(decode=Mock(side_effect=AssertionError("Text must retain its IDs.")))
    output = TokenOutput(
        prompt_ids=[10],
        response_ids=[650, 11, 651],
        response_mask=[1, 0, 1],
        response_log_probs=[-0.1, 0.0, -0.2],
    )
    batch = tu.get_tensordict({"uid": [7], "validate": [False], "version_tag": [0]})

    keys, fields = rope_worker._build_output_fields([output], batch, 7, 7, 0)

    assert keys == ["7"], "Output keys must retain their request identity."
    assert fields[0]["input_ids"].tolist() == [10, 650, 11, 651], "Every original token must survive packing."
    assert fields[0]["response_mask"].tolist() == [1, 0, 1], "Environment and assistant masks must remain aligned."
    assert fields[0]["rollout_log_probs"].tolist() == pytest.approx([-0.1, 0.0, -0.2]), (
        "Position construction must preserve every token log probability."
    )
    assert fields[0]["position_ids"].tolist() == [[0, 1, 2, 3]] * 4, "Visual IDs must receive text positions."
    rope_worker.tokenizer.decode.assert_not_called()
