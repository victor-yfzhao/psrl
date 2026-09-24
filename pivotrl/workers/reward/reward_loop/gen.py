import inspect
import logging
import os
import threading
from typing import Callable, TypeVar

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from pivotrl.utils.dataset.utils import _pre_process_inputs
from pivotrl.utils.logger import DualOutputHandler
from pivotrl.workers.reward.reward_model.manager import PivotRL_RewardModelManager
from pivotrl.workers.reward.reward_loop import register
from pivotrl.workers.reward.reward_loop.base import RewardLoopManagerBase
from pivotrl.workers.reward.gen_reward_function import DefaultGenRewardFunction, GenRewardFunctionBase

pivotrl_logger = logging.getLogger(__file__)
pivotrl_logger.setLevel(os.getenv("PIVOTRL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


def _extract_problem(extra_info: dict, raw_prompt, fallback: str) -> str:
    if isinstance(extra_info, dict):
        for key in ("question", "prompt", "problem"):
            value = extra_info.get(key)
            if isinstance(value, str) and value:
                return value
    if hasattr(raw_prompt, "item"):
        raw_prompt = raw_prompt.item()
    if isinstance(raw_prompt, list) and raw_prompt:
        first_message = raw_prompt[0]
        if isinstance(first_message, dict):
            content = first_message.get("content")
            if isinstance(content, str) and content:
                return content
    return fallback


def tokenize_rm_chat_prompt(
    tokenizer,
    messages: list[dict],
    *,
    add_generation_prompt: bool,
    max_length: int,
):
    """
    Tokenize a gen-RM chat prompt and left-truncate to ``max_length``.

    HuggingFace ``truncation=True`` without ``max_length`` uses
    ``tokenizer.model_max_length`` (128000 for GLM-Z1), so it does not enforce
    ``reward_models_config.*.rollout.prompt_length``. Default truncation is
    right-sided and would drop the trailing ``\\boxed{True/False}`` instruction.

    Args:
        tokenizer: Reward-model tokenizer.
        messages (list[dict]): Chat messages from ``prompt_constructor``.
        add_generation_prompt (bool): Whether to append the assistant prefix.
        max_length (int): Maximum RM prompt tokens (``rollout.prompt_length``).

    Returns:
        Tokenized RM prompt in the tokenizer's usual return type (tensor, list,
        or BatchEncoding).
    """
    assert max_length > 0, f"RM prompt max_length must be positive, got: {max_length}."
    original_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        # NOTE: padding=False so short prompts are not padded to prompt_length
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    finally:
        tokenizer.truncation_side = original_side


def _count_rm_input_tokens(rm_inputs) -> int:
    if isinstance(rm_inputs, torch.Tensor):
        input_ids_tensor = rm_inputs
        attention_mask_tensor = None
    else:
        input_ids_tensor = rm_inputs.get("input_ids")
        attention_mask_tensor = rm_inputs.get("attention_mask", None)
    if isinstance(attention_mask_tensor, torch.Tensor):
        if attention_mask_tensor.dim() == 2:
            return int(attention_mask_tensor[0].sum().item())
        return int(attention_mask_tensor.sum().item())
    if isinstance(input_ids_tensor, torch.Tensor):
        if input_ids_tensor.dim() == 2:
            return int(input_ids_tensor[0].numel())
        return int(input_ids_tensor.numel())
    if isinstance(input_ids_tensor, list):
        if input_ids_tensor and isinstance(input_ids_tensor[0], list):
            return len(input_ids_tensor[0])
        return len(input_ids_tensor)
    raise TypeError(f"Unsupported RM tokenizer output type: {type(rm_inputs)!r}.")


@register("gen")
class GenRewardLoopManager(RewardLoopManagerBase):
    """
    Reward loop for generative reward model.
    
    This manager handles:
    1. Constructing RM prompts from agent responses
    2. Sending prompts to RewardModelManager (via router or direct replica access)
    3. Parsing RM outputs and computing reward scores
    """
    _tokenizer_locks: dict[int, threading.RLock] = {}
    _tokenizer_locks_guard = threading.Lock()

    def __init__(
        self,
        config,
        tokenizer,
        reward_model_manager: PivotRL_RewardModelManager = None,
        reward_function: GenRewardFunctionBase = DefaultGenRewardFunction(),
        **reward_kwargs,
    ):
        """
        Initialize GenRewardLoopManager.

        Args:
            config: PivotRL config.
            tokenizer: Policy tokenizer (for decoding agent responses).
            compute_score: Custom scoring function to compute reward from RM output.
            reward_model_manager: RewardModelManager instance (provides router or replica handles).
            prompt_constructor: Optional custom function to construct RM prompts.
                Should accept (data_item, response_str: str) and return str.
                If provided, this will override the default prompt construction.
            reward_kwargs: Additional keyword arguments for compute_score.
        """
        super().__init__(config, tokenizer)

        self.reward_function = reward_function
        self.is_async_reward_score = inspect.iscoroutinefunction(self.reward_function.compute_score)

        self.reward_model_manager = reward_model_manager
        self.reward_model_tokenizer = self.reward_model_manager.get_reward_model_tokenizer()
        self.router_process = self.reward_model_manager.get_router_process()
        self.replica_handles = self.reward_model_manager.get_replica_handles()
        self.rm_prompt_length = int(self.reward_model_manager.reward_model_config.rollout.prompt_length)
        self._replica_rr_index = 0
        self._reward_model_tokenizer_lock = self._get_tokenizer_lock(self.reward_model_tokenizer)
        self.reward_kwargs = reward_kwargs
        pivotrl_logger.addHandler(DualOutputHandler(self.config.pivotrl.logging_path, "gen_reward_loop"))
        pivotrl_logger.info(
            f"[gen_reward_loop] Initialized GenRewardLoopManager with "
            f"rm_prompt_length={self.rm_prompt_length}."
        )

    @classmethod
    def _get_tokenizer_lock(cls, tokenizer) -> threading.RLock:
        tokenizer_id = id(tokenizer)
        with cls._tokenizer_locks_guard:
            lock = cls._tokenizer_locks.get(tokenizer_id)
            if lock is None:
                lock = threading.RLock()
                cls._tokenizer_locks[tokenizer_id] = lock
            return lock

    @staticmethod
    def _call_with_lock(lock: threading.RLock, fn: Callable[[], T]) -> T:
        with lock:
            return fn()

    async def _run_reward_model_tokenizer_call(self, fn: Callable[[], T]) -> T:
        return await self.loop.run_in_executor(
            None,
            lambda: self._call_with_lock(self._reward_model_tokenizer_lock, fn),
        )

    async def run_single(self, data: DataProto) -> dict:
        """
        Process a single data item through the generative reward model.

        Args:
            data: DataProto containing a single agent response.

        Returns:
            dict with keys:
                - reward_score: Final scalar reward
                - reward_extra_info: Additional metrics (e.g., accuracy, RM raw output)
        """
        assert len(data) == 1, "Only support single data item in run_single"
        data_item = data[0]
        request_uid = self._format_request_uid(data_item.non_tensor_batch.get("uid"))
        
        # Extract prompt
        prompt_ids = data_item.batch["prompts"]
        prompt_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(prompt_ids, skip_special_tokens=True),
        )
        # Extract agent response
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # Decode agent response
        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )

        # Extract metadata for reward computation
        data_source = data_item.non_tensor_batch.get("data_source", "unknown")
        reward_model_info = data_item.non_tensor_batch.get("reward_model")
        if isinstance(reward_model_info, dict):
            ground_truth = reward_model_info.get("ground_truth", "")
        else:
            ground_truth = ""
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        pivotrl_logger.debug("Reward loop received uid=%s source=%s", request_uid, data_source)

        # Construct RM prompt (e.g., "Question: ... Answer: ..." or custom template)
        problem_str = _extract_problem(
            extra_info,
            raw_prompt=data_item.non_tensor_batch.get("raw_prompt"),
            fallback=prompt_str,
        )
        rm_prompt = self.reward_function.prompt_constructor(prompt_str=problem_str, response_str=response_str)
        using_sys_prompt = self.reward_function.using_sys_prompt

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            policy_prompt_len = int(prompt_ids.numel())
        else:
            policy_prompt_len = int((prompt_ids != pad_id).sum().item())
        policy_response_len = (
            int(valid_response_length.item())
            if torch.is_tensor(valid_response_length)
            else int(valid_response_length)
        )

        # Tokenize RM prompt, left-truncated to rollout.prompt_length.
        rm_inputs = await self._run_reward_model_tokenizer_call(
            lambda: tokenize_rm_chat_prompt(
                self.reward_model_tokenizer,
                rm_prompt,
                add_generation_prompt=using_sys_prompt,
                max_length=self.rm_prompt_length,
            ),
        )

        # Handle case where apply_chat_template returns a tensor instead of a dict
        # Some tokenizers return the input_ids tensor directly when return_tensors="pt"
        if isinstance(rm_inputs, torch.Tensor):
            rm_inputs = {"input_ids": rm_inputs}

        rm_input_len = _count_rm_input_tokens(rm_inputs)
        if rm_input_len >= self.rm_prompt_length:
            pivotrl_logger.warning(
                f"[gen_reward_loop] RM prompt hit max_length={self.rm_prompt_length} for uid={request_uid}: "
                f"policy_prompt_tokens={policy_prompt_len} policy_response_tokens={policy_response_len} "
                f"problem_chars={len(problem_str)} solution_chars={len(response_str)} "
                f"rm_input_tokens={rm_input_len}."
            )

        # Build DataProto for RM inference
        rm_data_proto = self._build_rm_data_proto(rm_inputs, request_uid)

        # Send to reward model and get generated output (both string and logits)
        rm_output_dict = await self._query_reward_model(rm_data_proto, request_uid)
        rm_output_str = rm_output_dict.get("rm_output_str", "")
        rm_output_value = rm_output_dict.get("rm_output_value")

        reward_metrics = rm_output_dict.get("reward_metrics", {})
        if not isinstance(reward_metrics, dict):
            reward_metrics = {}

        # Compute final reward score using custom scoring function
        # Pass both rm_output_str and rm_output_logits, let compute_score choose which one to use
        if self.is_async_reward_score:
            result = await self.reward_function.compute_score(
                data_source=data_source,
                solution_str=response_str,
                rm_output=rm_output_str,
                rm_output_value=rm_output_value,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **self.reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.reward_function.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    rm_output=rm_output_str,
                    rm_output_value=rm_output_value,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **self.reward_kwargs,
                ),
            )

        # Parse result
        reward_extra_info = {}
        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["score"] = score
            reward_extra_info["acc"] = score
        reward_extra_info["rm_output"] = rm_output_str
        if rm_output_value is not None:
            reward_extra_info["rm_output_value"] = rm_output_value
        reward_extra_info["agent_response"] = response_str

        # Attach RM input/output token lengths to reward_extra_info
        rm_output_len = rm_output_dict.get("rm_output_len", None)
        if rm_input_len is not None:
            reward_extra_info["rm_input_len"] = rm_input_len
        if rm_output_len is not None:
            reward_extra_info["rm_output_len"] = rm_output_len

        pivotrl_logger.debug(
            "Reward computed uid=%s score=%.4f extra=%s",
            request_uid,
            score,
            {k: v for k, v in reward_extra_info.items() if k not in {"rm_output", "agent_response"}},
        )

        return {"reward_score": score, "reward_extra_info": reward_extra_info, "reward_metrics": reward_metrics}

    def _build_rm_data_proto(self, rm_inputs: dict, request_uid: str | None) -> DataProto:
        """
        Build a DataProto object for RM inference from tokenized inputs.
        """
        input_ids = rm_inputs["input_ids"]
        
        # Ensure input_ids is a 2D tensor [batch_size, seq_len]
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 1:
                # If 1D, add batch dimension: [seq_len] -> [1, seq_len]
                input_ids = input_ids.unsqueeze(0)
            elif input_ids.dim() > 2:
                raise ValueError(f"input_ids has unexpected dimension {input_ids.dim()}, expected 1 or 2")
        else:
            # Convert to tensor if not already
            input_ids = torch.tensor(input_ids)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        
        attention_mask = rm_inputs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif isinstance(attention_mask, torch.Tensor):
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
        else:
            attention_mask = torch.tensor(attention_mask)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

        # Create TensorDict for batch data
        batch_size = input_ids.shape[0]
        batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            batch_size=batch_size,
        )
        uid_value = request_uid if request_uid is not None else "unknown"
        
        # Extract raw_prompt_ids (remove padding) for vllm pre-processing
        pad_token_id = self.reward_model_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.reward_model_tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        raw_prompt_ids = [
            _pre_process_inputs(pad_token_id, input_ids[i]) for i in range(batch_size)
        ]
        
        # Add raw_prompt_ids and raw_response_ids (empty) for vllm pre-processing
        raw_response_ids = [[] for _ in range(batch_size)]

        raw_prompt_ids_array = np.empty(len(raw_prompt_ids), dtype=object)
        for i, prompt_list in enumerate(raw_prompt_ids):
            if isinstance(prompt_list, np.ndarray):
                raw_prompt_ids_array[i] = prompt_list.tolist()
            else:
                raw_prompt_ids_array[i] = list(prompt_list)
        
        raw_response_ids_array = np.empty(len(raw_response_ids), dtype=object)
        for i, response_list in enumerate(raw_response_ids):
            if isinstance(response_list, np.ndarray):
                raw_response_ids_array[i] = response_list.tolist()
            else:
                raw_response_ids_array[i] = list(response_list)
        
        non_tensor_batch = {
            "uid": np.array([uid_value], dtype=object),
            "raw_prompt_ids": raw_prompt_ids_array,
            "raw_response_ids": raw_response_ids_array,
        }
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def _query_reward_model(self, rm_data_proto: DataProto, request_uid: str | None) -> dict:
        """
        Send a DataProto to the reward model and return both the generated text and logits.
        
        Returns a dictionary with:
            - rm_output_str: Decoded string output (for normal gen_rm)
            - rm_output_logits: Last token logits or pooling output (for Skywork-like models)
        
        Uses either the router (if available) or directly calls a replica handle.
        """
        rm_outputs = None
        if self.router_process is not None:
            rm_outputs = await self.router_process.generate_async.remote(rm_data_proto)
        else:
            replica_idx, replica_handle = self._get_next_replica_handle()
            rm_outputs = await replica_handle.generate_async.remote(rm_data_proto)

        result = {"rm_output_str": "", "rm_output_value": None, "reward_metrics": {}}

        if rm_outputs is None or len(rm_outputs) == 0:
            pivotrl_logger.warning("Reward model returned empty output")
            return result

        reward_metrics = rm_outputs.meta_info.pop("vllm_metrics", None)

        # print(f"dump reward_metrics: {reward_metrics=}")
        if reward_metrics is not None:
            result["reward_metrics"] = reward_metrics[0]
        else:
            result["reward_metrics"] = {}

        # Try to extract pooling output (for Skywork-like reward models)
        if "pooling_output" in rm_outputs.non_tensor_batch:
            pooling_output = rm_outputs.non_tensor_batch["pooling_output"]
            if len(pooling_output) > 0:
                # Extract the scalar reward score from pooling output
                # pooling_output is typically a numpy array or tensor
                rm_output_value = pooling_output[0]
                if hasattr(rm_output_value, "item"):
                    result["rm_output_value"] = float(rm_output_value.item())
                elif isinstance(rm_output_value, (int, float, np.number)):
                    result["rm_output_value"] = float(rm_output_value)
                else:
                    # If it's an array, take the first element or mean
                    rm_output_array = np.array(rm_output_value)
                    if rm_output_array.size > 0:
                        result["rm_output_value"] = float(rm_output_array.flat[0])

        # Extract and decode string output (for normal gen_rm)
        if "raw_response_ids" in rm_outputs.non_tensor_batch:
            raw_response_ids = rm_outputs.non_tensor_batch["raw_response_ids"]
            if len(raw_response_ids) > 0 and len(raw_response_ids[0]) > 0:
                generated_ids = raw_response_ids[0]
                generated_str = await self._run_reward_model_tokenizer_call(
                    lambda: self.reward_model_tokenizer.decode(generated_ids, skip_special_tokens=True),
                )
                result["rm_output_str"] = generated_str
                result["rm_output_len"] = len(generated_ids)
        else:
            pivotrl_logger.warning("No raw_response_ids in RM output non_tensor_batch")

        return result

    def _get_next_replica_handle(self):
        if not self.replica_handles:
            raise RuntimeError("Replica handles are not available for reward model inference")
        handle = self.replica_handles[self._replica_rr_index]
        replica_idx = self._replica_rr_index
        self._replica_rr_index = (self._replica_rr_index + 1) % len(self.replica_handles)
        return replica_idx, handle

    @staticmethod
    def _get_uid_list(uid_value) -> list:
        if uid_value is None:
            return []
        if hasattr(uid_value, "tolist"):
            uid_value = uid_value.tolist()
        if isinstance(uid_value, (list, tuple)):
            return list(uid_value)
        return [uid_value]

    @classmethod
    def _format_request_uid(cls, uid_value) -> str:
        uid_list = cls._get_uid_list(uid_value)
        if not uid_list:
            return "unknown"
        if len(uid_list) == 1:
            return str(uid_list[0])
        return ",".join(str(uid) for uid in uid_list)
