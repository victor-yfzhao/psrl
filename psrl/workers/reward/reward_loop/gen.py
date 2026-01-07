import inspect
import logging
import os

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.utils.logger import DualOutputHandler
from psrl.workers.reward.reward_model.manager import PSRL_RewardModelManager
from psrl.workers.reward.reward_loop import register
from psrl.workers.reward.reward_loop.base import RewardLoopManagerBase
from psrl.workers.reward.gen_reward_function import DefaultGenRewardFunction, GenRewardFunctionBase

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


@register("gen")
class GenRewardLoopManager(RewardLoopManagerBase):
    """
    Reward loop for generative reward model.
    
    This manager handles:
    1. Constructing RM prompts from agent responses
    2. Sending prompts to RewardModelManager (via router or direct replica access)
    3. Parsing RM outputs and computing reward scores
    """

    def __init__(
        self,
        config,
        tokenizer,
        reward_model_manager: PSRL_RewardModelManager = None,
        reward_function: GenRewardFunctionBase = DefaultGenRewardFunction(),
        **reward_kwargs,
    ):
        """
        Initialize GenRewardLoopManager.

        Args:
            config: PSRL config.
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
        self._replica_rr_index = 0
        self.reward_kwargs = reward_kwargs
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, "gen_reward_loop"))
        psrl_logger.info("Initialized GenRewardLoopManager.")

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
        ground_truth = data_item.non_tensor_batch["reward_model"].get("ground_truth", "")
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        psrl_logger.info("Reward loop received uid=%s source=%s", request_uid, data_source)

        # Construct RM prompt (e.g., "Question: ... Answer: ..." or custom template)
        rm_prompt = self.reward_function.prompt_constructor(prompt_str=prompt_str, response_str=response_str)
        using_sys_prompt = self.reward_function.using_sys_prompt

        # Tokenize RM prompt
        rm_inputs = await self.loop.run_in_executor(
            None,
            lambda: self.reward_model_tokenizer.apply_chat_template(
                rm_prompt,
                tokenize=True,
                add_generation_prompt=using_sys_prompt,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ),
        )

        # Handle case where apply_chat_template returns a tensor instead of a dict
        # Some tokenizers return the input_ids tensor directly when return_tensors="pt"
        if isinstance(rm_inputs, torch.Tensor):
            rm_inputs = {"input_ids": rm_inputs}

        # Build DataProto for RM inference
        rm_data_proto = self._build_rm_data_proto(rm_inputs, request_uid)

        # Send to reward model and get generated output (both string and logits)
        rm_output_dict = await self._query_reward_model(rm_data_proto, request_uid)
        rm_output_str = rm_output_dict.get("rm_output_str", "")
        rm_output_value = rm_output_dict.get("rm_output_value")

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

        psrl_logger.info(
            "Reward computed uid=%s score=%.4f extra=%s",
            request_uid,
            score,
            {k: v for k, v in reward_extra_info.items() if k not in {"rm_output", "agent_response"}},
        )

        return {"reward_score": score, "reward_extra_info": reward_extra_info}

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
            psrl_logger.info("Routing reward uid=%s through router", request_uid)
            rm_outputs = await self.router_process.generate_async.remote(rm_data_proto)
        else:
            replica_idx, replica_handle = self._get_next_replica_handle()
            psrl_logger.info("Sending reward uid=%s to replica_%d", request_uid, replica_idx)
            rm_outputs = await replica_handle.generate_async.remote(rm_data_proto)

        result = {"rm_output_str": "", "rm_output_value": None}

        if rm_outputs is None or len(rm_outputs) == 0:
            psrl_logger.warning("Reward model returned empty output")
            return result

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
                generated_str = await self.loop.run_in_executor(
                    None,
                    lambda: self.reward_model_tokenizer.decode(generated_ids, skip_special_tokens=True),
                )
                result["rm_output_str"] = generated_str
                psrl_logger.info(
                    "Reward model response ready uid=%s tokens=%d", request_uid, len(generated_ids)
                )
        else:
            psrl_logger.warning("No raw_response_ids in RM output non_tensor_batch")

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
