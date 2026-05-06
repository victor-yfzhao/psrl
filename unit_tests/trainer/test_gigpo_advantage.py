import unittest

import numpy as np
import torch
from verl import DataProto

from psrl.trainer.ppo.adv_estimators.gigpo import (
    build_step_group,
    compute_gigpo_advantage,
    compute_step_discounted_returns,
    step_norm_reward,
)
from psrl.trainer.ppo.utils import PSRL_compute_advantage


class GigpoAdvantageTest(unittest.TestCase):
    def _make_step_row_data(self):
        return DataProto(
            batch={
                "response_mask": torch.tensor(
                    [
                        [1, 1, 0],
                        [1, 1, 0],
                    ],
                    dtype=torch.long,
                ),
                "token_level_rewards": torch.zeros((2, 3), dtype=torch.float32),
            },
            non_tensor_batch={
                "uid": np.array(["traj0:step:0", "traj1:step:0"], dtype=object),
                "traj_uid": np.array(["traj0", "traj1"], dtype=object),
                "parent_id": np.array(["ep0", "ep0"], dtype=object),
                "gigpo_anchor_obs": np.array(["shared", "shared"], dtype=object),
                "gigpo_step_idx": np.array([0, 0], dtype=object),
                "gigpo_step_return": np.array([1.0, 3.0], dtype=object),
                "gigpo_episode_return": np.array([1.0, 3.0], dtype=object),
            },
            meta_info={},
        )

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

    def test_build_step_group_supports_none_anchor_obs(self):
        anchor_obs = np.array([None, None, "other"], dtype=object)
        episode_index = np.array(["ep0", "ep0", "ep0"], dtype=object)

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

    def test_compute_gigpo_advantage_combines_episode_and_step_advantages(self):
        data = self._make_step_row_data()

        advantages, returns = compute_gigpo_advantage(
            data,
            gamma=1.0,
            norm_adv_by_std_in_grpo=False,
            step_advantage_weight=1.0,
        )

        expected_advantages = torch.tensor(
            [
                [-2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(data.batch["gigpo_total_advantages"], expected_advantages))
        self.assertTrue(torch.allclose(advantages, expected_advantages))
        self.assertTrue(torch.allclose(returns, expected_advantages))

    def test_compute_gigpo_advantage_supports_zero_step_weight(self):
        data = self._make_step_row_data()

        advantages, _ = compute_gigpo_advantage(
            data,
            gamma=1.0,
            norm_adv_by_std_in_grpo=False,
            step_advantage_weight=0.0,
        )

        expected_advantages = torch.tensor(
            [
                [-1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(advantages, expected_advantages))

    def test_episode_advantage_counts_each_trajectory_once(self):
        data = DataProto(
            batch={
                "response_mask": torch.tensor([[1, 0], [1, 0], [1, 0]], dtype=torch.long),
                "token_level_rewards": torch.zeros((3, 2), dtype=torch.float32),
            },
            non_tensor_batch={
                "uid": np.array(["traj0:step:0", "traj0:step:1", "traj1:step:0"], dtype=object),
                "traj_uid": np.array(["traj0", "traj0", "traj1"], dtype=object),
                "parent_id": np.array(["ep0", "ep0", "ep0"], dtype=object),
                "gigpo_anchor_obs": np.array(["a0", "a1", "a2"], dtype=object),
                "gigpo_step_idx": np.array([0, 1, 0], dtype=object),
                "gigpo_step_return": np.array([0.0, 0.0, 0.0], dtype=object),
                "gigpo_episode_return": np.array([1.0, 1.0, 3.0], dtype=object),
            },
            meta_info={},
        )

        advantages, _ = compute_gigpo_advantage(
            data,
            gamma=1.0,
            norm_adv_by_std_in_grpo=False,
            step_advantage_weight=0.0,
        )

        expected_advantages = torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(advantages, expected_advantages))

    def test_compute_gigpo_advantage_inherits_grpo_std_normalization_flag(self):
        data = self._make_step_row_data()

        advantages, _ = compute_gigpo_advantage(
            data,
            gamma=1.0,
            norm_adv_by_std_in_grpo=True,
            step_advantage_weight=1.0,
        )

        expected_advantages = torch.tensor(
            [
                [-1.4142128, -1.4142128, 0.0],
                [1.4142128, 1.4142128, 0.0],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(advantages, expected_advantages, atol=1e-5))

    def test_parent_id_is_required(self):
        data = self._make_step_row_data()
        del data.non_tensor_batch["parent_id"]

        with self.assertRaisesRegex(ValueError, "parent_id"):
            compute_gigpo_advantage(
                data,
                gamma=1.0,
                norm_adv_by_std_in_grpo=False,
            )

    def test_psrl_compute_advantage_dispatches_gigpo_estimator(self):
        expected_advantages = torch.tensor(
            [
                [-1.5, -1.5, 0.0],
                [1.5, 1.5, 0.0],
            ],
            dtype=torch.float32,
        )
        data = self._make_step_row_data()
        processed = PSRL_compute_advantage(
            data,
            adv_estimator="gigpo",
            gamma=1.0,
            norm_adv_by_std_in_grpo=False,
            config={"gigpo_step_advantage_weight": 0.5},
        )

        self.assertTrue(torch.allclose(processed.batch["advantages"], expected_advantages))
        self.assertTrue(torch.allclose(processed.batch["returns"], expected_advantages))


if __name__ == "__main__":
    unittest.main()
