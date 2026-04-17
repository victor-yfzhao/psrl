from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig
from verl import DataProto

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


@BufferPostProcessorRegistry.register("gigpo_step_metadata")
class GigpoStepMetadataProcessor(BaseBufferPostProcessor):
    """Rebuild canonical GiGPO step metadata from trajectory-row rollout sidecar."""

    REQUIRED_SIDECAR_KEYS = (
        "step_count",
        "step_idx",
        "anchor_obs",
        "step_reward",
        "step_done",
        "response_start",
        "response_end",
    )

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def __call__(self, data: DataProto) -> DataProto | None:
        assert not data.meta_info.get("validate", False), (
            "GiGPOStepMetadataProcessor should not run during validation."
        )
        for key in self.REQUIRED_SIDECAR_KEYS:
            if key not in data.non_tensor_batch:
                raise ValueError(f"Missing required GiGPO rollout sidecar field: {key}")
        if "prompts" not in data.batch or "responses" not in data.batch or "attention_mask" not in data.batch:
            raise ValueError(
                "GiGPOStepMetadataProcessor expects tensorized trajectory rows with prompts/responses/attention_mask."
            )

        prompt_width = data.batch["prompts"].shape[1]
        gigpo_steps: list[list[dict[str, Any]]] = []
        gigpo_step_counts: list[int] = []

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
            gigpo_step_counts.append(step_count)

        data.non_tensor_batch["gigpo_steps"] = np.array(gigpo_steps, dtype=object)
        data.non_tensor_batch["gigpo_step_count"] = np.array(gigpo_step_counts, dtype=np.int32)
        return data
