"""Trainer-side GiGPO advantage utilities."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import torch
from verl import DataProto


def to_hashable(value: Any) -> Any:
    """Convert nested ``anchor_obs`` payloads into a stable exact-match key."""
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(value.flatten().tolist())
    if isinstance(value, (list, tuple)):
        return tuple(to_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((repr(key), to_hashable(item)) for key, item in value.items()))
    raise TypeError(f"Unsupported anchor_obs type for GiGPO grouping: {type(value)}")


def are_similar(left: str, right: str, threshold: float = 0.95) -> bool:
    """Match verl-agent's optional text-similarity grouping mode."""
    if not isinstance(left, str) or not isinstance(right, str):
        raise ValueError("Similarity-based GiGPO grouping only supports string anchor_obs values.")
    return SequenceMatcher(None, left, right).ratio() >= threshold


def compute_step_discounted_returns(
    step_rewards: torch.Tensor,
    traj_index: np.ndarray,
    gamma: float,
) -> torch.Tensor:
    """Compute discounted returns over steps within each trajectory."""
    if step_rewards.ndim != 1:
        raise ValueError(f"step_rewards must be 1D, got shape {tuple(step_rewards.shape)}")
    if len(traj_index) != step_rewards.shape[0]:
        raise ValueError("traj_index length must match step_rewards length")

    returns = torch.zeros_like(step_rewards)
    traj_to_positions: dict[Any, list[int]] = defaultdict(list)
    for pos, traj_id in enumerate(traj_index):
        traj_to_positions[traj_id].append(pos)

    for positions in traj_to_positions.values():
        running_return = step_rewards.new_tensor(0.0)
        for pos in reversed(positions):
            running_return = step_rewards[pos] + gamma * running_return
            returns[pos] = running_return
    return returns


def build_step_group(
    anchor_obs: np.ndarray,
    index: np.ndarray,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
) -> np.ndarray:
    """Build step-level groups inside each episode group."""
    if anchor_obs.shape[0] != index.shape[0]:
        raise ValueError("anchor_obs and index must have the same length")
    if enable_similarity and not (0.0 < similarity_thresh < 1.0):
        raise ValueError("similarity_thresh must be in (0, 1) when similarity grouping is enabled")

    step_group_ids = np.empty(anchor_obs.shape[0], dtype=object)
    next_group_id = 0

    for episode_group in np.unique(index):
        positions = np.where(index == episode_group)[0]
        obs_group = anchor_obs[positions]

        if not enable_similarity:
            clusters: dict[Any, list[int]] = defaultdict(list)
            for local_idx, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(positions[local_idx])
            for clustered_positions in clusters.values():
                group_id = f"step_group_{next_group_id}"
                next_group_id += 1
                for pos in clustered_positions:
                    step_group_ids[pos] = group_id
            continue

        clusters_sim: list[dict[str, Any]] = []
        for obs, pos in zip(obs_group, positions):
            matched = False
            for cluster in clusters_sim:
                if are_similar(obs, cluster["rep"], similarity_thresh):
                    cluster["positions"].append(pos)
                    matched = True
                    break
            if not matched:
                clusters_sim.append({"rep": obs, "positions": [pos]})
        for cluster in clusters_sim:
            group_id = f"step_group_{next_group_id}"
            next_group_id += 1
            for pos in cluster["positions"]:
                step_group_ids[pos] = group_id

    if any(group_id is None for group_id in step_group_ids.tolist()):
        raise ValueError("Failed to assign GiGPO step groups to every step.")
    return step_group_ids


def step_norm_reward(
    step_rewards: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    mode: str = "mean_std_norm",
) -> torch.Tensor:
    """Normalize scalar returns inside each group."""
    if step_rewards.ndim != 1:
        raise ValueError(f"step_rewards must be 1D, got shape {tuple(step_rewards.shape)}")
    if len(index) != step_rewards.shape[0]:
        raise ValueError("index length must match step_rewards length")
    if mode not in {"mean_norm", "mean_std_norm"}:
        raise ValueError(f"Unsupported GiGPO mode: {mode}")

    normalized = step_rewards.clone()
    id_to_positions: dict[Any, list[int]] = defaultdict(list)
    for pos, group_id in enumerate(index):
        id_to_positions[group_id].append(pos)

    for positions in id_to_positions.values():
        group_values = step_rewards[positions]
        group_mean = group_values.mean()
        if mode == "mean_norm" or len(positions) == 1:
            group_std = step_rewards.new_tensor(1.0)
        else:
            group_std = group_values.std(unbiased=True)
        if mode == "mean_norm":
            normalized[positions] = group_values - group_mean
        else:
            normalized[positions] = (group_values - group_mean) / (group_std + epsilon)
    return normalized


def compute_gigpo_advantage(
    data: DataProto,
    gamma: float,
    norm_adv_by_std_in_grpo: bool,
    step_advantage_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GiGPO total advantages for step-row PPO optimization."""
    required_non_tensor_keys = {
        "parent_id",
        "traj_uid",
        "gigpo_anchor_obs",
        "gigpo_step_return",
        "gigpo_episode_return",
    }
    missing_keys = required_non_tensor_keys.difference(data.non_tensor_batch.keys())
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"GiGPO advantage expects step-row non_tensor fields: {missing}.")
    if "response_mask" not in data.batch:
        raise ValueError("GiGPO advantage expects data.batch['response_mask'].")

    device = data.batch["response_mask"].device
    mode = "mean_std_norm" if norm_adv_by_std_in_grpo else "mean_norm"
    parent_index = np.array(data.non_tensor_batch["parent_id"], dtype=object)
    traj_index = np.array(data.non_tensor_batch["traj_uid"], dtype=object)
    anchor_obs = np.array(data.non_tensor_batch["gigpo_anchor_obs"], dtype=object)
    step_returns = torch.tensor(
        data.non_tensor_batch["gigpo_step_return"].astype(float),
        dtype=torch.float32,
        device=device,
    )

    traj_keys: list[tuple[Any, Any]] = []
    traj_returns: list[float] = []
    traj_parent_index: list[Any] = []
    traj_key_to_pos: dict[tuple[Any, Any], int] = {}
    for row_idx, (parent_id, traj_uid, episode_return) in enumerate(
        zip(parent_index, traj_index, data.non_tensor_batch["gigpo_episode_return"])
    ):
        traj_key = (parent_id, traj_uid)
        if traj_key in traj_key_to_pos:
            prev_return = traj_returns[traj_key_to_pos[traj_key]]
            if float(episode_return) != prev_return:
                raise ValueError(
                    f"Inconsistent GiGPO episode returns for trajectory {traj_uid}: "
                    f"{prev_return} vs {float(episode_return)} at row {row_idx}."
                )
            continue
        traj_key_to_pos[traj_key] = len(traj_keys)
        traj_keys.append(traj_key)
        traj_parent_index.append(parent_id)
        traj_returns.append(float(episode_return))

    unique_episode_returns = torch.tensor(traj_returns, dtype=torch.float32, device=device)
    unique_episode_advantages = step_norm_reward(
        unique_episode_returns,
        index=np.array(traj_parent_index, dtype=object),
        mode=mode,
    )
    episode_advantages = torch.empty(len(parent_index), dtype=torch.float32, device=device)
    for row_idx, (parent_id, traj_uid) in enumerate(zip(parent_index, traj_index)):
        episode_advantages[row_idx] = unique_episode_advantages[traj_key_to_pos[(parent_id, traj_uid)]]

    step_group_ids = build_step_group(anchor_obs=anchor_obs, index=parent_index)
    step_advantages = step_norm_reward(step_returns, index=step_group_ids, mode=mode)
    total_advantages = episode_advantages + float(step_advantage_weight) * step_advantages

    response_mask = data.batch["response_mask"]
    token_advantages = total_advantages.unsqueeze(-1) * response_mask
    data.batch["gigpo_episode_advantages"] = episode_advantages.unsqueeze(-1) * response_mask
    data.batch["gigpo_step_advantages"] = step_advantages.unsqueeze(-1) * response_mask
    data.batch["gigpo_total_advantages"] = token_advantages
    return token_advantages, token_advantages
