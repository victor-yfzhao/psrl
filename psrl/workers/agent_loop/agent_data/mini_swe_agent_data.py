"""Agent data adapter for mini-SWE-agent trajectories."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict

from omegaconf import DictConfig

from psrl.environments.base import Environment
from psrl.workers.agent_loop.agent_data.base import AgentData
from psrl.workers.agent_loop.agent_data.conversation_agent_data import (
    ConversationAgentData,
    normalize_openai_messages,
)
from psrl.workers.reward.client import RewardSubmissionClient

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@AgentData.register("mini_swe_agent_data")
class MiniSWEAgentData(ConversationAgentData):
    """
    Attach SWE patch and grading inputs to a conversation trajectory.
    """

    def __init__(
        self,
        config: DictConfig,
        reward_manager: RewardSubmissionClient,
        env: Environment,
        **kwargs,
    ):
        super().__init__(config, reward_manager, env, **kwargs)
        self.patch: str | None = None
        self.problem_statement = ""
        self.grading_context: dict = {}
        self.timing: dict = {}

    def reset(self) -> None:
        """Reset trajectory and SWE-specific state."""
        super().reset()
        self.patch = None
        self.problem_statement = ""
        self.grading_context = {}
        self.timing = {}

    def init_trajectory(self, request: dict) -> None:
        """Initialize trajectory metadata and capture the problem statement."""
        super().init_trajectory(request)
        self.problem_statement = self.session_data.extra_info.get("problem_statement", "")

    def set_patch(self, patch: str | None) -> None:
        """Store the patch submitted by mini-SWE-agent."""
        self.patch = patch

    def set_grading_context(
        self,
        observation: dict,
    ) -> None:
        """
        Capture resolved task inputs and resource limits for the reward service.
        """
        if observation.get("swe_grader") != "swebench_fresh_container":
            return
        self.grading_context = {
            key: observation[key] for key in ("swe_problem", "swe_problem_image", "swe_grader_kind", "swe_task_id")
        }
        environment = asdict(observation["runtime_config"].sandbox_config.environment)
        self.grading_context["environment"] = {
            key: environment[key]
            for key in (
                "grader_timeout",
                "grader_command_timeout",
                "grader_max_attempts",
                "grader_memory",
                "grader_shm_size",
                "run_args",
                "env",
            )
        }

    def set_timing(self, timing: dict) -> None:
        """Store per-trajectory wall-clock timing reported by the runner."""
        self.timing = timing or {}

    async def encode_messages(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """Tokenize normalized OpenAI-format messages."""
        return await self._apply_chat_template_ids(
            normalize_openai_messages(messages),
            add_generation_prompt=add_generation_prompt,
        )

    async def finalize_output(self):
        """Finalize output after attaching SWE metadata to the agent namespace."""
        num_turns = self.session_data.trajectories[-1].assistant_turns
        self.session_data.agent_reward_info.update(
            {
                "patch": self.patch,
                "num_turns": num_turns,
                "actual_num_turns": num_turns,
                "alignment_failed": False,
                "alignment_failure_reason": "",
                "swe_grading": self.grading_context,
            }
        )
        output = await super().finalize_output()
        if output is not None:
            for item in output if isinstance(output, list) else [output]:
                item.extra_fields["loop_timing"] = self.timing
        return output
