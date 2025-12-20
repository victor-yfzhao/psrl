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
    def prompt_constructor(self, data_item: DataProto, response_str: str, tokenizer) -> str:
        """
        Prompt constructor for reward function.

        Args:
            data_item: Data item.
            response_str: Agent's solution string.
            tokenizer: Tokenizer (Rollout's tokenizer, ** NOT Reward Model's tokenizer **).

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
    
    def prompt_constructor(self, data_item: DataProto, response_str: str, tokenizer) -> str:
        # Extract problem/question
        problem = data_item.non_tensor_batch["reward_model"].get("question", "")
        if not problem:
            # Fallback: try to get from extra_info
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            problem = extra_info.get("question", "")
        
        if not problem:
            # Final fallback: decode input_ids as question
            input_ids = data_item.batch.get("input_ids", torch.tensor([]))
            if len(input_ids) > 0:
                problem = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Use template to construct prompt (default: GENRM_PROMPT_TEMPLATE format)
        rm_prompt = DEFAULT_GENRM_PROMPT_TEMPLATE.format(problem=problem, solution=response_str)
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
    