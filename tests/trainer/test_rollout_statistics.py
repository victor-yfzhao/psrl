"""
Check rollout statistics against configured limits and raw importance ratios.
"""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights

pytestmark = pytest.mark.cpu_test


def test_rollout_logprobs_share_verl_processed_probability_definition():
    config_dir = Path(__file__).resolve().parents[2] / "psrl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=["model_engine=megatron"])

    for name in ("train_actor_rollout_ref", "gen_actor_rollout_ref"):
        assert config[name].rollout.logprobs_mode == "processed_logprobs", (
            "Rollout logprobs must include the configured temperature and sampling filters."
        )


def test_sequence_outlier_fractions_use_raw_weights_and_exclude_masked_sequences():
    raw_ratios = torch.tensor([[4.0, 4.0], [0.25, 0.25], [100.0, 100.0]])
    mask = torch.tensor([[1, 1], [1, 1], [0, 0]])
    weights, metrics = compute_rollout_correction_weights(raw_ratios.log(), mask, rollout_is_threshold=2.0)

    assert weights.max().item() == 2.0, "Applied token weights must remain clipped."
    assert metrics["rollout_is_seq_fraction_high"] == 0.5, "Raw upper outliers must remain observable after clipping."
    assert metrics["rollout_is_seq_fraction_low"] == 0.5, (
        "Only sequences with training tokens belong in the denominator."
    )
    assert metrics["rollout_is_seq_mean"] == 1.125, "Sequence means must describe the applied weights."


def test_fully_masked_sequence_cannot_create_a_deviation_from_unit_weights():
    ratios = torch.tensor([[1.0, 1.0], [100.0, 100.0]])
    _, metrics = compute_rollout_correction_weights(ratios.log(), torch.tensor([[1, 1], [0, 0]]))

    assert metrics["rollout_is_seq_max_deviation"] == 0, (
        "Discarded sequences must not contribute artificial deviations."
    )


def test_length_clip_ratios_use_configured_limits_and_count_environment_tokens():
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.zeros(2, 4, dtype=torch.long),
            "responses": torch.zeros(2, 6, dtype=torch.long),
            "attention_mask": torch.tensor([[1] * 10, [0, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
            "response_mask": torch.tensor([[1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 0]]),
            "token_level_scores": torch.zeros(2, 6),
            "token_level_rewards": torch.zeros(2, 6),
            "advantages": torch.ones(2, 6),
            "returns": torch.ones(2, 6),
        },
        meta_info={"max_prompt_length": 10, "max_response_length": 20},
    )
    metrics = compute_data_metrics(batch, use_critic=False)

    assert metrics["prompt_length/clip_ratio"] == 0, "The largest batch prompt need not reach the configured cap."
    assert metrics["response_length/clip_ratio"] == 0, "The largest batch response need not reach the configured cap."
    assert metrics["response_length/mean"] == 5, "Response length must include environment context."
    batch.meta_info.update(max_prompt_length=4, max_response_length=6)
    metrics = compute_data_metrics(batch, use_critic=False)
    assert metrics["prompt_length/clip_ratio"] == 0.5, "Only prompts at the configured cap count as clipped."
    assert metrics["response_length/clip_ratio"] == 0.5, "Only responses at the configured cap count as clipped."
