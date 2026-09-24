import torch
from verl.utils.dataset.rl_dataset import _prepare_multi_modal_inputs_for_training
from verl.utils.model import extract_multi_modal_inputs


def test_qwen35_sequence_aligned_processor_fields_are_not_forwarded_to_training():
    mm_token_type_ids = torch.zeros((1, 17), dtype=torch.long)
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    processor_inputs = {
        "mm_token_type_ids": mm_token_type_ids,
        "image_grid_thw": image_grid_thw,
    }

    training_inputs = _prepare_multi_modal_inputs_for_training(processor_inputs)

    assert "mm_token_type_ids" not in training_inputs
    assert training_inputs["image_grid_thw"] is image_grid_thw
    assert processor_inputs["mm_token_type_ids"] is mm_token_type_ids


def test_qwen35_training_inputs_with_different_prompt_lengths_can_be_batched():
    samples = [
        _prepare_multi_modal_inputs_for_training(
            {
                "mm_token_type_ids": torch.zeros((1, prompt_length), dtype=torch.long),
                "image_grid_thw": torch.tensor([[1, side, side]], dtype=torch.long),
            }
        )
        for prompt_length, side in ((46, 4), (89, 6))
    ]

    batched = extract_multi_modal_inputs(samples)

    assert "mm_token_type_ids" not in batched
    assert torch.equal(batched["image_grid_thw"], torch.tensor([[1, 4, 4], [1, 6, 6]]))
