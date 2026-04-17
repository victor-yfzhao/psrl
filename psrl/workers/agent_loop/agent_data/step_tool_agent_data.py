"""Step-aware tool agent data for rollout-side step metadata collection.

This module defines a dedicated ToolAgentData variant used when rollout needs to
preserve per-step metadata for later step-level training reconstruction.

Compared with `tool_agent_data.py`, the key difference is that this module:
- extends each step with step-level sidecar fields such as anchor state and spans
- records those fields during multi-turn rollout
- exports them through `DataProto.non_tensor_batch` at finalize time

The default `ToolAgentData` remains the plain trajectory-oriented implementation.
This module exists specifically to avoid pushing GiGPO / step-aware rollout
requirements back into the default path.
"""

from dataclasses import dataclass

import numpy as np
from verl import DataProto

from psrl.environments.base import ConversationType
from psrl.environments.tool_env import ToolAction
from psrl.workers.agent_loop.agent_data.base import Step
from psrl.workers.agent_loop.agent_data.tool_agent_data import ToolAgentData


@dataclass
class StepToolStep(Step):
    """Tool-agent step metadata for step-aware rollout export.

    This is a narrow extension of the base `Step` dataclass. The added fields are
    not part of generic PSRL rollout state; they are only needed when later
    trainer-side logic wants to reconstruct step samples from a trajectory-row
    rollout output.
    """

    # Env-provided anchor state used for step grouping / hashing.
    anchor_obs: object = None
    # Monotonic step index within the current trajectory, starting from 0.
    step_idx: int | None = None
    # Start offset of the full step span in the unified trajectory raw-token space.
    step_start: int | None = None
    # End offset of the full step span in the same raw-token space.
    step_end: int | None = None
    # Start offset of the assistant-generated response for this step in the same trajectory raw-token space.
    response_start: int | None = None
    # End offset of the assistant-generated response for this step.
    response_end: int | None = None

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(
            {
                "anchor_obs": self.anchor_obs,
                "step_idx": self.step_idx,
                "step_start": self.step_start,
                "step_end": self.step_end,
                "response_start": self.response_start,
                "response_end": self.response_end,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "StepToolStep":
        return cls(
            chat_completions=data.get("chat_completions", []),
            observation=data.get("observation"),
            thought=data.get("thought", ""),
            action=data.get("action"),
            model_response=data.get("model_response", ""),
            info=data.get("info", {}),
            tool_reward=data.get("tool_reward", 0.0),
            model_reward=data.get("model_reward", 0.0),
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            anchor_obs=data.get("anchor_obs"),
            step_idx=data.get("step_idx"),
            step_start=data.get("step_start"),
            step_end=data.get("step_end"),
            response_start=data.get("response_start"),
            response_end=data.get("response_end"),
        )


@ToolAgentData.register("step_tool_agent_data")
class StepToolAgentData(ToolAgentData):
    """Tool agent data variant that records and exports step-aware rollout metadata.

    This subclass keeps the regular tool-agent behavior from `ToolAgentData`, but
    additionally tracks step boundaries and env-provided anchor states so the
    finalized trajectory can carry a rollout-side step sidecar.
    """

    def start_step(
        self, observation: ConversationType, reward: float | None, done: bool, info: dict | None
    ) -> StepToolStep:
        """Create a step-aware tool step.

        The base class only creates a generic `Step`. Here we additionally stamp:
        - `step_idx`: step order within the trajectory
        - `step_start`: start offset in the full raw token trajectory space

        Later, `update_from_model_token_ids()` fills the response span and step end.
        """
        step = StepToolStep(
            chat_completions=[],
            observation=observation,
            # Step numbering is trajectory-local and preserves rollout order.
            step_idx=len(self.trajectory.steps),
            # All offsets are measured in one coordinate space:
            # len(prompt_ids) + len(response_ids) accumulated so far.
            step_start=len(self.trajectory.prompt_ids) + len(self.trajectory.response_ids),
        )
        step.tool_reward = reward
        step.done = done
        step.info = info or {}
        self.trajectory.steps.append(step)
        return step

    def _is_exportable_step(self, step: Step, max_raw_token_length: int) -> bool:
        """Return whether a step has enough metadata to be exported as sidecar.

        Only complete steps are exported. If rollout stops early or span bookkeeping 
        is incomplete, that step is kept out of the sidecar.
        """
        if not isinstance(step, StepToolStep):
            return False
        required = (
            step.step_idx,
            step.step_start,
            step.step_end,
            step.response_start,
            step.response_end,
        )
        if any(v is None for v in required):
            return False
        if not (step.step_start <= step.response_start <= step.response_end <= step.step_end):
            return False
        if step.step_end > max_raw_token_length:
            return False
        return True

    def _build_step_sidecar(self) -> dict[str, np.ndarray]:
        """Assemble trajectory-level step sidecar arrays from collected steps.

        Output format intentionally stays trajectory-row oriented: each field is
        stored as an object array containing one per-trajectory list. Trainer-side
        code can later decide how to flatten or reinterpret these step records.
        """
        max_raw_token_length = len(self.trajectory.prompt_ids) + len(self.trajectory.response_ids)
        # Only fully-formed steps are exported. Partial steps stay in trajectory
        # state but are not serialized into the sidecar.
        exportable_steps = [step for step in self.trajectory.steps if self._is_exportable_step(step, max_raw_token_length)]

        step_count = len(exportable_steps)
        if step_count > 0:
            step_indices = [step.step_idx for step in exportable_steps]
            assert step_indices == list(range(step_count)), (
                f"Step indices must be contiguous from 0. Got {step_indices} for request {self.trajectory.request_id}."
            )
            prev_step_end = -1
            prev_response_end = -1
            for step in exportable_steps:
                # All span metadata lives in the same raw-token coordinate space.
                assert step.step_start <= step.response_start <= step.response_end <= step.step_end, (
                    f"Invalid step span ordering for request {self.trajectory.request_id}, step {step.step_idx}: "
                    f"step=({step.step_start}, {step.step_end}), response=({step.response_start}, {step.response_end})."
                )
                # Steps must appear in rollout order without going backwards.
                assert step.step_start >= prev_step_end, (
                    f"Step spans must be monotonic for request {self.trajectory.request_id}: "
                    f"{step.step_start} < previous end {prev_step_end}."
                )
                # Response spans should also be monotonic across steps.
                assert step.response_start >= prev_response_end, (
                    f"Response spans must be monotonic for request {self.trajectory.request_id}: "
                    f"{step.response_start} < previous end {prev_response_end}."
                )
                prev_step_end = step.step_end
                prev_response_end = step.response_end

        return {
            # Keep the rollout output trajectory-row oriented. Each field stores one
            # per-trajectory list instead of flattening into step rows during rollout.
            "step_count": np.array([step_count], dtype=np.int32),
            "step_idx": np.array([[step.step_idx for step in exportable_steps]], dtype=object),
            "anchor_obs": np.array([[step.anchor_obs for step in exportable_steps]], dtype=object),
            "step_start": np.array([[step.step_start for step in exportable_steps]], dtype=object),
            "step_end": np.array([[step.step_end for step in exportable_steps]], dtype=object),
            "response_start": np.array([[step.response_start for step in exportable_steps]], dtype=object),
            "response_end": np.array([[step.response_end for step in exportable_steps]], dtype=object),
            "step_reward": np.array(
                # Export raw env immediate reward for GiGPO-side reconstruction.
                # This intentionally differs from `Step.reward` in base.py, which
                # may include model reward, reward shaping, or discounted returns.
                [[step.tool_reward for step in exportable_steps]],
                dtype=object,
            ),
            "step_done": np.array([[step.done for step in exportable_steps]], dtype=object),
        }

    async def update_from_env(
        self,
        observation: ConversationType,
        reward: float | None,
        done: bool,
        info: dict,
        **kwargs,
    ) -> bool:
        """Update rollout state from env and attach step-level anchor metadata.

        The parent class handles the normal tool-agent observation flow. After that
        we capture `anchor_obs` from env `info` onto the current step when the step
        is one of our step-aware `StepToolStep` instances.
        """
        reached_limit = await super().update_from_env(observation, reward, done, info, **kwargs)
        if len(self.trajectory.steps) > 0 and isinstance(self.get_current_step(), StepToolStep):
            # Prefer env-provided anchor state when available; otherwise fall back
            # to the raw observation so GiGPO-specific rollout stays self-contained
            # without requiring ToolEnvironment changes.
            if info is not None and "anchor_obs" in info:
                anchor_obs = info["anchor_obs"]
            else:
                anchor_obs = observation
            self.get_current_step().anchor_obs = anchor_obs
        return reached_limit

    async def update_from_model_token_ids(self, output: DataProto, **kwargs) -> tuple[ToolAction, bool]:
        """Record assistant response span in full trajectory raw-token space.

        We snapshot the trajectory length before and after delegating to the parent
        implementation. The difference gives the assistant response span for the
        current step. In the current append-only rollout assumption, `step_end`
        coincides with `response_end`.
        """
        # Snapshot the start before the parent update mutates response_ids.
        response_start = len(self.trajectory.prompt_ids) + len(self.trajectory.response_ids)
        action, done = await super().update_from_model_token_ids(output, **kwargs)
        step = self.get_current_step()
        if isinstance(step, StepToolStep):
            # Snapshot the end after assistant tokens have been appended.
            response_end = len(self.trajectory.prompt_ids) + len(self.trajectory.response_ids)
            step.response_start = response_start
            step.response_end = response_end
            # Under the current append-only rollout assumption the step ends at the
            # end of this assistant response.
            step.step_end = response_end
        return action, done

    async def finalize_output(self, request: DataProto) -> DataProto:
        """Finalize rollout output and append step sidecar metadata.

        Core trajectory export still comes from the parent class. This override only
        adds the rollout-side step sidecar for later trainer-side consumption.
        """
        output = await super().finalize_output(request)
        output.non_tensor_batch.update(self._build_step_sidecar())
        return output
