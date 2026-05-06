import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto

from psrl.utils.post_processor.buffer_post_process.gigpo import GigpoStepRowProcessor


def _make_config():
    return OmegaConf.create(
        {
            "algorithm": {"gamma": 1.0},
            "psrl": {"redundant_rollout": {"enable": False}},
            "gen_actor_rollout_ref": {"rollout": {"n": 1}},
        }
    )


def _make_batch():
    non_tensor_batch = {
        "uid": np.array([11], dtype=object),
        "parent_id": np.array(["ep0"], dtype=object),
        "step_count": np.array([2], dtype=np.int32),
        "step_idx": np.array([[0, 1]], dtype=object),
        "anchor_obs": np.array([["a0", "a1"]], dtype=object),
        "step_reward": np.array([[0.5, 1.25]], dtype=object),
        "step_done": np.array([[False, True]], dtype=object),
        "response_start": np.array([[3, 5]], dtype=object),
        "response_end": np.array([[5, 5]], dtype=object),
        "rollout_instance_id": np.array([0], dtype=object),
        "version_tag": np.array([1], dtype=object),
    }

    return DataProto(
        batch={
            "prompts": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[4, 5, 0, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
            "input_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long),
            "rollout_log_probs": torch.tensor([[-0.1, -0.2, -1.0, -1.0]], dtype=torch.float32),
        },
        non_tensor_batch=non_tensor_batch,
        meta_info={},
    )


def _make_multi_response_batch():
    data = _make_batch()
    data.non_tensor_batch["step_reward"] = np.array([[0.5, 1.25]], dtype=object)
    data.non_tensor_batch["response_start"] = np.array([[3, 6]], dtype=object)
    data.non_tensor_batch["response_end"] = np.array([[5, 7]], dtype=object)
    data.batch["responses"] = torch.tensor([[4, 5, 6, 7, 0]], dtype=torch.long)
    data.batch["response_mask"] = torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.long)
    data.batch["input_ids"] = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
    data.batch["attention_mask"] = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long)
    data.batch["rollout_log_probs"] = torch.tensor([[-0.1, -0.2, 0.0, -0.4, -1.0]], dtype=torch.float32)
    return data


class GigpoStepRowProcessorTest(unittest.TestCase):
    def test_expands_response_steps_to_step_rows(self):
        processor = GigpoStepRowProcessor(_make_config())
        data = _make_batch()

        processed = processor(data)

        self.assertEqual(len(processed), 1)
        self.assertEqual(processed.non_tensor_batch["uid"].tolist(), ["11:step:0"])
        self.assertEqual(processed.non_tensor_batch["traj_uid"].tolist(), [11])
        self.assertEqual(processed.non_tensor_batch["parent_id"].tolist(), ["ep0"])
        self.assertEqual(processed.non_tensor_batch["gigpo_anchor_obs"].tolist(), ["a0"])
        self.assertEqual(processed.non_tensor_batch["gigpo_step_idx"].tolist(), [0])
        self.assertEqual(processed.non_tensor_batch["gigpo_step_return"].tolist(), [1.75])
        self.assertEqual(processed.non_tensor_batch["gigpo_episode_return"].tolist(), [1.75])
        self.assertTrue(torch.equal(processed.batch["prompts"], torch.tensor([[0, 1, 2, 3]], dtype=torch.long)))
        self.assertTrue(torch.equal(processed.batch["responses"], torch.tensor([[4, 5]], dtype=torch.long)))
        self.assertTrue(torch.equal(processed.batch["response_mask"], torch.tensor([[1, 1]], dtype=torch.long)))
        self.assertTrue(torch.equal(processed.batch["rm_scores"], torch.zeros((1, 2), dtype=torch.float32)))
        self.assertTrue(
            torch.allclose(processed.batch["rollout_log_probs"], torch.tensor([[-0.1, -0.2]], dtype=torch.float32))
        )

    def test_expands_multiple_response_steps(self):
        processor = GigpoStepRowProcessor(_make_config())
        data = _make_multi_response_batch()

        processed = processor(data)

        self.assertEqual(len(processed), 2)
        self.assertEqual(processed.non_tensor_batch["uid"].tolist(), ["11:step:0", "11:step:1"])
        self.assertEqual(processed.non_tensor_batch["gigpo_anchor_obs"].tolist(), ["a0", "a1"])
        self.assertEqual(processed.non_tensor_batch["gigpo_step_return"].tolist(), [1.75, 1.25])
        self.assertTrue(
            torch.equal(
                processed.batch["prompts"],
                torch.tensor(
                    [
                        [0, 0, 0, 1, 2, 3],
                        [1, 2, 3, 4, 5, 6],
                    ],
                    dtype=torch.long,
                ),
            )
        )
        self.assertTrue(torch.equal(processed.batch["responses"], torch.tensor([[4, 5], [7, 0]], dtype=torch.long)))
        self.assertTrue(torch.equal(processed.batch["response_mask"], torch.tensor([[1, 1], [1, 0]], dtype=torch.long)))
        self.assertTrue(
            torch.allclose(
                processed.batch["rollout_log_probs"],
                torch.tensor([[-0.1, -0.2], [-0.4, -1.0]], dtype=torch.float32),
            )
        )

    def test_raises_on_invalid_relative_span(self):
        processor = GigpoStepRowProcessor(_make_config())
        data = _make_batch()
        data.non_tensor_batch["response_start"] = np.array([[2, 5]], dtype=object)

        with self.assertRaises(ValueError):
            processor(data)

    def test_raises_on_non_contiguous_step_idx(self):
        processor = GigpoStepRowProcessor(_make_config())
        data = _make_batch()
        data.non_tensor_batch["step_idx"] = np.array([[0, 2]], dtype=object)

        with self.assertRaises(ValueError):
            processor(data)


if __name__ == "__main__":
    unittest.main()
