from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

from ..base import BaseBufferPostProcessor, BufferPostProcessorRegistry


def _to_python_list(value: Any) -> list[Any]:
    """Normalize per-row sidecar fields into plain Python lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running_return = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running_return = rewards[idx] + gamma * running_return
        returns[idx] = running_return
    return returns


def _infer_pad_token_id(data: DataProto) -> int:
    attention_mask = data.batch["attention_mask"]
    if "input_ids" in data.batch:
        pad_tokens = data.batch["input_ids"][attention_mask == 0]
        if pad_tokens.numel() > 0:
            return int(pad_tokens[0].item())

    prompt_width = data.batch["prompts"].shape[1]
    prompt_pad_tokens = data.batch["prompts"][data.batch["attention_mask"][:, :prompt_width] == 0]
    if prompt_pad_tokens.numel() > 0:
        return int(prompt_pad_tokens[0].item())

    response_pad_tokens = data.batch["responses"][data.batch["attention_mask"][:, prompt_width:] == 0]
    if response_pad_tokens.numel() > 0:
        return int(response_pad_tokens[0].item())

    return 0


def _left_pad(sequences: list[list[int]], width: int, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    masks = torch.zeros((len(sequences), width), dtype=torch.long)
    for row_idx, sequence in enumerate(sequences):
        if len(sequence) > width:
            raise ValueError(f"GiGPO step-row prompt length {len(sequence)} exceeds padded width {width}.")
        if sequence:
            values[row_idx, -len(sequence) :] = torch.tensor(sequence, dtype=torch.long)
            masks[row_idx, -len(sequence) :] = 1
    return values, masks


def _right_pad(sequences: list[list[int]], width: int, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    masks = torch.zeros((len(sequences), width), dtype=torch.long)
    for row_idx, sequence in enumerate(sequences):
        if len(sequence) > width:
            raise ValueError(f"GiGPO step-row response length {len(sequence)} exceeds padded width {width}.")
        if sequence:
            values[row_idx, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            masks[row_idx, : len(sequence)] = 1
    return values, masks


def _right_pad_float(sequences: list[list[float]], width: int, pad_value: float) -> torch.Tensor:
    values = torch.full((len(sequences), width), pad_value, dtype=torch.float32)
    for row_idx, sequence in enumerate(sequences):
        if len(sequence) > width:
            raise ValueError(f"GiGPO step-row float sequence length {len(sequence)} exceeds padded width {width}.")
        if sequence:
            values[row_idx, : len(sequence)] = torch.tensor(sequence, dtype=torch.float32)
    return values


def _canonicalize_steps(data: DataProto) -> list[list[dict[str, Any]]]:
    prompt_width = data.batch["prompts"].shape[1]
    gigpo_steps: list[list[dict[str, Any]]] = []

    for row_idx in range(len(data)):
        step_count = int(data.non_tensor_batch["step_count"][row_idx])
        step_idx_list = _to_python_list(data.non_tensor_batch["step_idx"][row_idx])
        anchor_obs_list = _to_python_list(data.non_tensor_batch["anchor_obs"][row_idx])
        step_reward_list = _to_python_list(data.non_tensor_batch["step_reward"][row_idx])
        step_done_list = _to_python_list(data.non_tensor_batch["step_done"][row_idx])
        response_start_list = _to_python_list(data.non_tensor_batch["response_start"][row_idx])
        response_end_list = _to_python_list(data.non_tensor_batch["response_end"][row_idx])

        sidecar_fields = {
            "step_idx": step_idx_list,
            "anchor_obs": anchor_obs_list,
            "step_reward": step_reward_list,
            "step_done": step_done_list,
            "response_start": response_start_list,
            "response_end": response_end_list,
        }
        for key, values in sidecar_fields.items():
            if len(values) != step_count:
                raise ValueError(
                    f"GiGPO sidecar field '{key}' length {len(values)} does not match step_count {step_count} "
                    f"for row {row_idx}."
                )

        expected_step_indices = list(range(step_count))
        if [int(step_idx) for step_idx in step_idx_list] != expected_step_indices:
            raise ValueError(
                f"GiGPO sidecar step_idx must be contiguous from 0 for row {row_idx}. "
                f"Got {step_idx_list}, expected {expected_step_indices}."
            )

        prompt_valid_length = int(data.batch["attention_mask"][row_idx, :prompt_width].sum().item())
        response_valid_length = int(data.batch["attention_mask"][row_idx, prompt_width:].sum().item())

        row_steps: list[dict[str, Any]] = []
        prev_response_end_rel = 0
        for step_pos in range(step_count):
            response_start_abs = int(response_start_list[step_pos])
            response_end_abs = int(response_end_list[step_pos])
            response_start_rel = response_start_abs - prompt_valid_length
            response_end_rel = response_end_abs - prompt_valid_length
            if not (0 <= response_start_rel <= response_end_rel <= response_valid_length):
                raise ValueError(
                    f"Invalid GiGPO response span for row {row_idx}, step {step_pos}: "
                    f"abs=({response_start_abs}, {response_end_abs}), "
                    f"rel=({response_start_rel}, {response_end_rel}), "
                    f"prompt_valid_length={prompt_valid_length}, response_valid_length={response_valid_length}."
                )
            if response_start_rel < prev_response_end_rel:
                raise ValueError(
                    f"GiGPO response spans must be monotonic for row {row_idx}: "
                    f"{response_start_rel} < previous end {prev_response_end_rel}."
                )
            prev_response_end_rel = response_end_rel

            row_steps.append(
                {
                    "step_idx": int(step_idx_list[step_pos]),
                    "anchor_obs": anchor_obs_list[step_pos],
                    "step_reward": float(step_reward_list[step_pos]),
                    "step_done": bool(step_done_list[step_pos]),
                    "response_start_abs": response_start_abs,
                    "response_end_abs": response_end_abs,
                    "response_start_rel": response_start_rel,
                    "response_end_rel": response_end_rel,
                    "has_response": response_end_rel > response_start_rel,
                }
            )
        gigpo_steps.append(row_steps)

    return gigpo_steps


@BufferPostProcessorRegistry.register("gigpo_step_row")
class GigpoStepRowProcessor(BaseBufferPostProcessor):
    """Rebuild GiGPO step metadata and expand trajectory rows into step rows."""

    REQUIRED_SIDECAR_KEYS = (
        "step_count",
        "step_idx",
        "anchor_obs",
        "step_reward",
        "step_done",
        "response_start",
        "response_end",
        "parent_id",
        "uid",
    )

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def __call__(self, data: DataProto) -> DataProto | None:
        assert not data.meta_info.get("validate", False), "GigpoStepRowProcessor should not run during validation."
        for key in self.REQUIRED_SIDECAR_KEYS:
            if key not in data.non_tensor_batch:
                raise ValueError(f"Missing required GiGPO rollout sidecar field: {key}")
        required_batch_keys = {"prompts", "responses", "attention_mask", "response_mask"}
        missing_batch_keys = required_batch_keys.difference(data.batch.keys())
        if missing_batch_keys:
            missing = ", ".join(sorted(missing_batch_keys))
            raise ValueError(f"GigpoStepRowProcessor expects tensorized trajectory rows with: {missing}.")

        gamma = float(self.config.algorithm.get("gamma", 1.0))
        prompt_width = data.batch["prompts"].shape[1]
        pad_token_id = _infer_pad_token_id(data)
        canonical_steps = _canonicalize_steps(data)
        has_rollout_log_probs = "rollout_log_probs" in data.batch

        step_prompts: list[list[int]] = []
        step_responses: list[list[int]] = []
        step_response_masks: list[list[int]] = []
        step_rollout_log_probs: list[list[float]] = []
        non_tensor_rows: dict[str, list[Any]] = {
            "uid": [],
            "traj_uid": [],
            "parent_id": [],
            "gigpo_anchor_obs": [],
            "gigpo_step_idx": [],
            "gigpo_step_reward": [],
            "gigpo_step_return": [],
            "gigpo_episode_return": [],
        }

        passthrough_keys = [
            key
            for key in data.non_tensor_batch.keys()
            if key
            not in {
                "uid",
                "parent_id",
                "step_count",
                "step_idx",
                "anchor_obs",
                "step_reward",
                "step_done",
                "step_start",
                "step_end",
                "response_start",
                "response_end",
                "gigpo_steps",
                "gigpo_step_count",
            }
        ]
        for key in passthrough_keys:
            if key not in non_tensor_rows:
                non_tensor_rows[key] = []

        for row_idx, row_steps in enumerate(canonical_steps):
            row_step_count_before = len(step_prompts)
            step_rewards = [step["step_reward"] for step in row_steps]
            step_returns = _discounted_returns(step_rewards, gamma=gamma)
            episode_return = float(sum(step_rewards))

            prompt_mask = data.batch["attention_mask"][row_idx, :prompt_width].bool()
            response_valid_length = int(data.batch["attention_mask"][row_idx, prompt_width:].sum().item())
            prompt_ids = data.batch["prompts"][row_idx][prompt_mask].tolist()
            response_ids = data.batch["responses"][row_idx, :response_valid_length].tolist()
            response_mask = data.batch["response_mask"][row_idx, :response_valid_length].tolist()
            rollout_log_probs = (
                data.batch["rollout_log_probs"][row_idx, :response_valid_length].tolist()
                if has_rollout_log_probs
                else None
            )

            traj_uid = data.non_tensor_batch["uid"][row_idx]
            parent_id = data.non_tensor_batch["parent_id"][row_idx]

            for step, step_return in zip(row_steps, step_returns):
                if not step["has_response"]:
                    continue
                start = step["response_start_rel"]
                end = step["response_end_rel"]
                step_response = response_ids[start:end]
                step_mask = response_mask[start:end]
                if not step_response or not any(step_mask):
                    continue

                step_prompts.append(prompt_ids + response_ids[:start])
                step_responses.append(step_response)
                step_response_masks.append([int(mask) for mask in step_mask])
                if rollout_log_probs is not None:
                    step_rollout_log_probs.append([float(log_prob) for log_prob in rollout_log_probs[start:end]])

                step_idx = int(step["step_idx"])
                non_tensor_rows["uid"].append(f"{traj_uid}:step:{step_idx}")
                non_tensor_rows["traj_uid"].append(traj_uid)
                non_tensor_rows["parent_id"].append(parent_id)
                non_tensor_rows["gigpo_anchor_obs"].append(step["anchor_obs"])
                non_tensor_rows["gigpo_step_idx"].append(step_idx)
                non_tensor_rows["gigpo_step_reward"].append(float(step["step_reward"]))
                non_tensor_rows["gigpo_step_return"].append(float(step_return))
                non_tensor_rows["gigpo_episode_return"].append(episode_return)
                for key in passthrough_keys:
                    if key in non_tensor_rows:
                        non_tensor_rows[key].append(data.non_tensor_batch[key][row_idx])

            if len(step_prompts) == row_step_count_before:
                raise ValueError(f"GiGPO trajectory row {row_idx} did not produce any trainable step rows.")

        if not step_prompts:
            return None

        step_prompt_width = max(prompt_width, max(len(prompt) for prompt in step_prompts))
        step_response_width = max(1, max(len(response) for response in step_responses))
        prompts, prompt_attention_mask = _left_pad(step_prompts, step_prompt_width, pad_token_id)
        responses, response_attention_mask = _right_pad(step_responses, step_response_width, pad_token_id)
        response_mask_values, _ = _right_pad(step_response_masks, step_response_width, 0)
        response_mask_values = response_mask_values * response_attention_mask

        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        input_ids = torch.cat([prompts, responses], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_tensors = {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "rm_scores": torch.zeros_like(response_mask_values, dtype=torch.float32),
        }
        if has_rollout_log_probs:
            batch_tensors["rollout_log_probs"] = _right_pad_float(step_rollout_log_probs, step_response_width, -1.0)

        batch = TensorDict(batch_tensors, batch_size=len(step_prompts))
        non_tensor_batch = {key: np.array(values, dtype=object) for key, values in non_tensor_rows.items()}
        meta_info = dict(data.meta_info)
        meta_info["gigpo_step_row"] = True
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
