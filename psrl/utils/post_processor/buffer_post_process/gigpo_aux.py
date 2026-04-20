from __future__ import annotations

"""GiGPO step auxiliary builder.

This module lives under ``buffer_post_process`` because it operates on the same
training-ready ``DataProto`` shape as other buffer-level processors:

1. rollout sidecar has already been rebuilt into canonical ``gigpo_steps``
2. tensors such as ``response_mask`` already exist
3. the output is still the same trajectory-row batch, only with extra GiGPO
   auxiliary fields attached for trainer-side opt-in consumption

The implementation here follows the step grouping / discounted return /
normalization semantics from ``verl-agent``, but deliberately stops short of
changing the main PPO/GRPO ``advantages`` and ``returns`` fields.
"""

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import torch
from verl import DataProto


def to_hashable(value: Any) -> Any:
    """Convert nested ``anchor_obs`` payloads into a stable exact-match key."""
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(value.flatten().tolist())
    if isinstance(value, (list, tuple)):
        return tuple(to_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, to_hashable(item)) for key, item in value.items()))
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
    """Compute discounted returns over steps within each trajectory.

    Args:
        step_rewards: Flat per-step immediate rewards in batch order.
        traj_index: Trajectory id for each flattened step.
        gamma: Discount factor used inside each trajectory.

    Returns:
        A 1D tensor aligned with ``step_rewards`` where each value is the
        discounted return from the current step to the end of its trajectory.
    """
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
    """Build step-level groups inside each episode group.

    Two steps can only share a GiGPO step group if they belong to the same
    episode group. Within one episode group, grouping is either:

    - exact match on ``anchor_obs`` after canonicalization
    - approximate text match when similarity mode is enabled
    """
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
    """Normalize scalar step returns inside each step group.

    ``mean_norm`` matches the "subtract group mean only" variant.
    ``mean_std_norm`` matches the standard "subtract mean and divide by std"
    variant used in GiGPO.
    """
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


def build_gigpo_step_auxiliary(
    data: DataProto,
    gamma: float = 1.0,
    mode: str = "mean_std_norm",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    epsilon: float = 1e-6,
) -> DataProto:
    """Attach GiGPO step-level auxiliary fields to a trajectory-row batch.

    Expected inputs:
        - ``batch["response_mask"]``: standard response-token validity mask
        - ``non_tensor_batch["gigpo_steps"]``: canonical step metadata rebuilt
          from rollout sidecar
        - ``non_tensor_batch["uid"]``: trajectory id
        - optional ``non_tensor_batch["parent_id"]``: episode group id

    Attached outputs:
        - ``batch["gigpo_step_advantages"]``: token-shaped step advantages
        - ``batch["gigpo_step_token_mask"]``: where step advantages were written
        - ``non_tensor_batch["gigpo_step_returns"]``: per-row scalar step returns

    This function does not modify the main training fields such as
    ``advantages`` or ``returns``.
    """
    if "response_mask" not in data.batch:
        raise ValueError("GiGPO step auxiliary expects data.batch['response_mask'].")
    if "gigpo_steps" not in data.non_tensor_batch:
        raise ValueError("GiGPO step auxiliary expects non_tensor_batch['gigpo_steps'].")
    if "uid" not in data.non_tensor_batch:
        raise ValueError("GiGPO step auxiliary expects non_tensor_batch['uid'].")

    batch_size, _ = data.batch["response_mask"].shape
    response_mask = data.batch["response_mask"]

    # Flatten per-row step metadata so the GiGPO math can run on a simple
    # 1D "all steps in batch order" view, then scatter results back later.
    flat_step_rewards: list[float] = []
    flat_anchor_obs: list[Any] = []
    flat_episode_index: list[Any] = []
    flat_traj_index: list[Any] = []
    flat_row_index: list[int] = []
    flat_response_start: list[int] = []
    flat_response_end: list[int] = []
    flat_has_response: list[bool] = []

    for row_idx in range(batch_size):
        row_uid = data.non_tensor_batch["uid"][row_idx]
        row_parent_id = data.non_tensor_batch["parent_id"][row_idx] if "parent_id" in data.non_tensor_batch else row_uid
        row_steps = data.non_tensor_batch["gigpo_steps"][row_idx]
        # ``response_start_rel`` / ``response_end_rel`` are coordinates in the
        # local response view, so validate them against the actual valid
        # response length for this row instead of the padded width.
        row_response_length = int(response_mask[row_idx].sum().item())
        for step in row_steps:
            required_keys = {
                "anchor_obs",
                "step_reward",
                "response_start_rel",
                "response_end_rel",
                "has_response",
            }
            missing_keys = required_keys.difference(step.keys())
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(f"GiGPO canonical step metadata is missing keys: {missing}")

            response_start_rel = int(step["response_start_rel"])
            response_end_rel = int(step["response_end_rel"])
            if not (0 <= response_start_rel <= response_end_rel <= row_response_length):
                raise ValueError(
                    f"Invalid GiGPO relative span for row {row_idx}: "
                    f"({response_start_rel}, {response_end_rel}) with response_length={row_response_length}."
                )
            flat_step_rewards.append(float(step["step_reward"]))
            flat_anchor_obs.append(step["anchor_obs"])
            flat_episode_index.append(row_parent_id)
            flat_traj_index.append(row_uid)
            flat_row_index.append(row_idx)
            flat_response_start.append(response_start_rel)
            flat_response_end.append(response_end_rel)
            flat_has_response.append(bool(step["has_response"]))

    if not flat_step_rewards:
        data.batch["gigpo_step_advantages"] = torch.zeros_like(response_mask, dtype=torch.float32)
        data.batch["gigpo_step_token_mask"] = torch.zeros_like(response_mask)
        data.non_tensor_batch["gigpo_step_returns"] = np.array([[] for _ in range(batch_size)], dtype=object)
        return data

    step_rewards_tensor = torch.tensor(flat_step_rewards, dtype=torch.float32, device=response_mask.device)
    traj_index = np.array(flat_traj_index, dtype=object)
    episode_index = np.array(flat_episode_index, dtype=object)
    anchor_obs = np.array(flat_anchor_obs, dtype=object)

    step_returns = compute_step_discounted_returns(step_rewards_tensor, traj_index=traj_index, gamma=gamma)
    step_group_ids = build_step_group(
        anchor_obs=anchor_obs,
        index=episode_index,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    step_advantages = step_norm_reward(step_returns, index=step_group_ids, epsilon=epsilon, mode=mode)

    gigpo_step_advantages = torch.zeros(
        response_mask.shape,
        dtype=step_advantages.dtype,
        device=response_mask.device,
    )
    gigpo_step_token_mask = torch.zeros_like(response_mask)

    # GiGPO computes a scalar advantage per step. Trainer-side consumers still
    # expect token-shaped tensors, so we only write each step score into that
    # step's response span.
    per_row_returns: list[list[float]] = [[] for _ in range(batch_size)]
    for flat_idx, row_idx in enumerate(flat_row_index):
        per_row_returns[row_idx].append(float(step_returns[flat_idx].item()))
        if not flat_has_response[flat_idx]:
            # Empty-response steps still contribute to return computation and
            # group normalization, but they have no tokens to supervise.
            continue
        start = flat_response_start[flat_idx]
        end = flat_response_end[flat_idx]
        if start == end:
            continue
        gigpo_step_advantages[row_idx, start:end] = step_advantages[flat_idx]
        gigpo_step_token_mask[row_idx, start:end] = 1

    data.batch["gigpo_step_advantages"] = gigpo_step_advantages
    data.batch["gigpo_step_token_mask"] = gigpo_step_token_mask
    data.non_tensor_batch["gigpo_step_returns"] = np.array(per_row_returns, dtype=object)
    return data
