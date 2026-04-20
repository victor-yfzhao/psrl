import unittest

import numpy as np
import torch
from verl import DataProto

from psrl.utils.post_processor.buffer_post_process.gigpo_aux import (
    build_gigpo_step_auxiliary,
    build_step_group,
    compute_step_discounted_returns,
    step_norm_reward,
)


class GigpoAuxTest(unittest.TestCase):
    def test_compute_step_discounted_returns(self):
        step_rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        traj_index = np.array(["traj0", "traj0", "traj1"], dtype=object)

        returns = compute_step_discounted_returns(step_rewards, traj_index=traj_index, gamma=0.5)

        self.assertTrue(torch.allclose(returns, torch.tensor([2.0, 2.0, 3.0], dtype=torch.float32)))

    def test_build_step_group_scopes_to_episode_group(self):
        anchor_obs = np.array(["same", "same", "same"], dtype=object)
        episode_index = np.array(["ep0", "ep0", "ep1"], dtype=object)

        group_ids = build_step_group(anchor_obs=anchor_obs, index=episode_index)

        self.assertEqual(group_ids[0], group_ids[1])
        self.assertNotEqual(group_ids[0], group_ids[2])

    def test_step_norm_reward_supports_both_modes(self):
        step_returns = torch.tensor([1.0, 3.0], dtype=torch.float32)
        group_ids = np.array(["g0", "g0"], dtype=object)

        mean_norm = step_norm_reward(step_returns, index=group_ids, mode="mean_norm")
        mean_std_norm = step_norm_reward(step_returns, index=group_ids, mode="mean_std_norm")

        self.assertTrue(torch.allclose(mean_norm, torch.tensor([-1.0, 1.0], dtype=torch.float32)))
        expected = torch.tensor([-0.7071064, 0.7071064], dtype=torch.float32)
        self.assertTrue(torch.allclose(mean_std_norm, expected, atol=1e-5))

    def test_build_gigpo_step_auxiliary_scatter_and_returns(self):
        data = DataProto(
            batch={
                "response_mask": torch.tensor(
                    [
                        [1, 1, 1, 0],
                        [1, 1, 0, 0],
                    ],
                    dtype=torch.long,
                )
            },
            non_tensor_batch={
                "uid": np.array(["traj0", "traj1"], dtype=object),
                "parent_id": np.array(["ep0", "ep0"], dtype=object),
                "gigpo_steps": np.array(
                    [
                        [
                            {
                                "step_idx": 0,
                                "anchor_obs": "shared",
                                "step_reward": 1.0,
                                "step_done": False,
                                "response_start_rel": 0,
                                "response_end_rel": 2,
                                "has_response": True,
                            },
                            {
                                "step_idx": 1,
                                "anchor_obs": "empty",
                                "step_reward": 0.0,
                                "step_done": True,
                                "response_start_rel": 2,
                                "response_end_rel": 2,
                                "has_response": False,
                            },
                        ],
                        [
                            {
                                "step_idx": 0,
                                "anchor_obs": "shared",
                                "step_reward": 3.0,
                                "step_done": True,
                                "response_start_rel": 0,
                                "response_end_rel": 2,
                                "has_response": True,
                            }
                        ],
                    ],
                    dtype=object,
                ),
            },
            meta_info={},
        )

        processed = build_gigpo_step_auxiliary(data, gamma=1.0, mode="mean_norm")

        self.assertEqual(processed.non_tensor_batch["gigpo_step_returns"].tolist(), [[1.0, 0.0], [3.0]])
        expected_advantages = torch.tensor(
            [
                [-1.0, -1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(processed.batch["gigpo_step_advantages"], expected_advantages))
        expected_mask = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(processed.batch["gigpo_step_token_mask"], expected_mask))

    def test_empty_response_step_participates_without_scatter(self):
        data = DataProto(
            batch={"response_mask": torch.tensor([[1, 1, 0]], dtype=torch.long)},
            non_tensor_batch={
                "uid": np.array(["traj0"], dtype=object),
                "parent_id": np.array(["ep0"], dtype=object),
                "gigpo_steps": np.array(
                    [
                        [
                            {
                                "step_idx": 0,
                                "anchor_obs": "a0",
                                "step_reward": 1.0,
                                "step_done": False,
                                "response_start_rel": 0,
                                "response_end_rel": 2,
                                "has_response": True,
                            },
                            {
                                "step_idx": 1,
                                "anchor_obs": "a1",
                                "step_reward": 0.5,
                                "step_done": True,
                                "response_start_rel": 2,
                                "response_end_rel": 2,
                                "has_response": False,
                            },
                        ]
                    ],
                    dtype=object,
                ),
            },
            meta_info={},
        )

        processed = build_gigpo_step_auxiliary(data, gamma=1.0, mode="mean_norm")

        self.assertEqual(processed.non_tensor_batch["gigpo_step_returns"].tolist(), [[1.5, 0.5]])
        self.assertTrue(torch.equal(processed.batch["gigpo_step_token_mask"], torch.tensor([[1, 1, 0]], dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
