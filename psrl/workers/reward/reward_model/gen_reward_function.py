from abc import ABC, abstractmethod
import logging
import os
from typing import Optional

import torch
from verl import DataProto
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class GenRewardFunctionBase(ABC):
    @abstractmethod
    def compute_score(
        self,
        data_source: str,
        solution_str: str,
        rm_output: str,
        ground_truth: str = "",
        extra_info: Optional[dict] = None,
        **kwargs,
    ) -> float:
        """
        Scoring function for reward function.
        
        Args:
            data_source: Dataset source identifier.
            solution_str: Agent's solution string.
            rm_output: Reward model's generated output.
            ground_truth: Ground truth answer.
            extra_info: Additional information.
            **kwargs: Additional keyword arguments.
        
        Returns:
            float: Reward score.
        """
        raise NotImplementedError
    
    @abstractmethod
    def prompt_constructor(self, **kwargs) -> str:
        """
        Prompt constructor for reward function.

        Args:
            **kwargs: Keyword arguments. Self-defined.

        Returns:
            str: Prompt.
        """
        raise NotImplementedError
    
    
DEFAULT_GENRM_PROMPT_TEMPLATE = """
The following is a math problem and an AI solution:

[Math Problem]

{problem}

[AI Solution]

{solution}

Your task is to review and critique the solution step by step, and output whether the AI solution is correct.

Please put your final answer (i.e., 'True' or 'False') in \\boxed{{}}.
""".strip()


class DefaultGenRewardFunction(GenRewardFunctionBase):
    def __init__(self):
        self.prompt_template = DEFAULT_GENRM_PROMPT_TEMPLATE
    
    def prompt_constructor(self, prompt_str: str, response_str: str) -> str:      
        # Use template to construct prompt (default: GENRM_PROMPT_TEMPLATE format)
        rm_prompt = DEFAULT_GENRM_PROMPT_TEMPLATE.format(problem=prompt_str, solution=response_str)
        return rm_prompt

    def compute_score(
        self,
        data_source: str,
        solution_str: str,
        rm_output: str,
        ground_truth: str = "",
        extra_info: Optional[dict] = None,
        **kwargs,
    ) -> float:
        reward_score = 0.0
        try:
            boxed_result = last_boxed_only_string(rm_output)
            if boxed_result is not None:
                result = remove_boxed(boxed_result)
                reward_score = float(result == "True")
        except Exception as e:
            psrl_logger.warning(f"Error computing reward score from RM output: {e}")
        return reward_score
    