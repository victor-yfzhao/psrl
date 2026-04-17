import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto

from psrl.utils.post_processor.buffer_post_process.gigpo import GigpoStepMetadataProcessor


def _make_config():
    return OmegaConf.create(
        {
            "psrl": {"redundant_rollout": {"enable": False}},
            "gen_actor_rollout_ref": {"rollout": {"n": 1}},
        }
    )


def _make_batch():
    non_tensor_batch = {
        "uid": np.array([11], dtype=object),
        "step_count": np.array([2], dtype=np.int32),
        "step_idx": np.array([[0, 1]], dtype=object),
        "anchor_obs": np.array([["a0", "a1"]], dtype=object),
        "step_reward": np.array([[0.5, 1.25]], dtype=object),
        "step_done": np.array([[False, True]], dtype=object),
        "response_start": np.array([[3, 5]], dtype=object),
        "response_end": np.array([[5, 5]], dtype=object),
    }

    return DataProto(
        batch={
            "prompts": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[4, 5, 0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long),
        },
        non_tensor_batch=non_tensor_batch,
        meta_info={},
    )


class GigpoStepMetadataProcessorTest(unittest.TestCase):
    def test_builds_canonical_step_metadata(self):
        processor = GigpoStepMetadataProcessor(_make_config())
        data = _make_batch()

        processed = processor(data)

        self.assertEqual(processed.non_tensor_batch["gigpo_step_count"].tolist(), [2])
        steps = processed.non_tensor_batch["gigpo_steps"][0]
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["response_start_abs"], 3)
        self.assertEqual(steps[0]["response_end_abs"], 5)
        self.assertEqual(steps[0]["response_start_rel"], 0)
        self.assertEqual(steps[0]["response_end_rel"], 2)
        self.assertTrue(steps[0]["has_response"])
        self.assertEqual(steps[1]["response_start_rel"], 2)
        self.assertEqual(steps[1]["response_end_rel"], 2)
        self.assertFalse(steps[1]["has_response"])
        self.assertEqual(steps[1]["step_reward"], 1.25)

    def test_raises_on_invalid_relative_span(self):
        processor = GigpoStepMetadataProcessor(_make_config())
        data = _make_batch()
        data.non_tensor_batch["response_start"] = np.array([[2, 5]], dtype=object)

        with self.assertRaises(ValueError):
            processor(data)

    def test_raises_on_non_contiguous_step_idx(self):
        processor = GigpoStepMetadataProcessor(_make_config())
        data = _make_batch()
        data.non_tensor_batch["step_idx"] = np.array([[0, 2]], dtype=object)

        with self.assertRaises(ValueError):
            processor(data)


if __name__ == "__main__":
    unittest.main()
