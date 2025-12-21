import inspect
import logging
import os

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.workers.reward.reward_model.manager import PSRL_RewardModelManager
from psrl.workers.reward.reward_loop import register
from psrl.workers.reward.reward_loop.base import RewardLoopManagerBase
from psrl.workers.reward.reward_model import DefaultGenRewardFunction, GenRewardFunctionBase

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


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
        reward_model_manager: PSRL_RewardModelManager | None = None,
        reward_function: GenRewardFunctionBase = DefaultGenRewardFunction(),
        router_process=None,
        replica_handles: list | None = None,
        reward_model_tokenizer=None,
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
        self.reward_model_tokenizer = reward_model_tokenizer
        if self.reward_model_manager is not None:
            if self.reward_model_tokenizer is None:
                self.reward_model_tokenizer = self.reward_model_manager.get_reward_model_tokenizer()
            router_process = router_process or self.reward_model_manager.get_router_process()
            replica_handles = replica_handles or self.reward_model_manager.get_replica_handles()
        self.router_process = router_process
        self.replica_handles = [handle for handle in (replica_handles or []) if handle is not None]
        if self.reward_model_tokenizer is None:
            raise ValueError("Reward model tokenizer must be provided for GenRewardLoopManager.")
        if self.router_process is None and not self.replica_handles:
            raise ValueError("No router process or replica handles provided for reward model inference.")
        self._replica_rr_index = 0
        self.reward_kwargs = reward_kwargs
        psrl_logger.info(f"RewardModelManager's Initialization Complete")

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

        # Tokenize RM prompt
        rm_inputs = await self.loop.run_in_executor(
            None,
            lambda: self.reward_model_tokenizer(
                rm_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ),
        )

        # Build DataProto for RM inference
        rm_data_proto = self._build_rm_data_proto(rm_inputs, request_uid)

        # Send to reward model and get generated output
        rm_output_str = await self._query_reward_model(rm_data_proto, request_uid)

        # Compute final reward score using custom scoring function
        if self.is_async_reward_score:
            result = await self.reward_function.compute_score(
                data_source=data_source,
                solution_str=response_str,
                rm_output=rm_output_str,
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
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **self.reward_kwargs,
                ),
            )

        # Parse result
        reward_extra_info = {}
        if isinstance(result, dict):
            score = result["score"]
            reward_extra_info.update(result)
        else:
            score = float(result)
            reward_extra_info["acc"] = score

        reward_extra_info["rm_output"] = rm_output_str
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
        attention_mask = rm_inputs.get("attention_mask", torch.ones_like(input_ids))

        # Create TensorDict for batch data
        batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            batch_size=len(input_ids),
        )
        uid_value = request_uid if request_uid is not None else "unknown"
        batch_size = len(input_ids)
        
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

    async def _query_reward_model(self, rm_data_proto: DataProto, request_uid: str | None) -> str:
        """
        Send a DataProto to the reward model and return the generated text.
        
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

        if rm_outputs is None or len(rm_outputs) == 0:
            psrl_logger.warning("Reward model returned empty output")
            return ""

        if "raw_response_ids" not in rm_outputs.non_tensor_batch:
            psrl_logger.error("No raw_response_ids in RM output non_tensor_batch")
            return ""
        
        raw_response_ids = rm_outputs.non_tensor_batch["raw_response_ids"]
        if len(raw_response_ids) == 0:
            psrl_logger.warning("raw_response_ids is empty")
            return ""

        generated_ids = raw_response_ids[0]

        generated_str = await self.loop.run_in_executor(
            None,
            lambda: self.reward_model_tokenizer.decode(generated_ids, skip_special_tokens=True),
        )
        psrl_logger.info(
            "Reward model response ready uid=%s tokens=%d", request_uid, len(generated_ids)
        )
        return generated_str

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
