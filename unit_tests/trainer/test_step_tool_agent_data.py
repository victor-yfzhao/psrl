import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto

from psrl.workers.agent_loop.agent_data.step_tool_agent_data import StepToolAgentData, StepToolStep
from psrl.workers.agent_loop.manager import PSRL_AgentLoopManager


class _FakeParser:
    def extract_tool_calls(self, token_ids):
        return None, []


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.padding_side = "right"

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, **kwargs):
        content = "".join(msg.get("content", "") for msg in messages if isinstance(msg, dict))
        token_ids = [100]
        token_ids.extend((ord(ch) % 50) + 1 for ch in content)
        if add_generation_prompt:
            token_ids.append(101)
        if tokenize:
            return token_ids
        return content

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(tok) for tok in token_ids)

    def pad(
        self,
        encoded_inputs,
        padding="max_length",
        max_length=None,
        return_tensors="pt",
        return_attention_mask=True,
    ):
        padded = []
        masks = []
        for item in encoded_inputs:
            input_ids = list(item["input_ids"])
            pad_len = max_length - len(input_ids)
            if self.padding_side == "left":
                padded_ids = [self.pad_token_id] * pad_len + input_ids
                attention_mask = [0] * pad_len + [1] * len(input_ids)
            else:
                padded_ids = input_ids + [self.pad_token_id] * pad_len
                attention_mask = [1] * len(input_ids) + [0] * pad_len
            padded.append(padded_ids)
            masks.append(attention_mask)
        output = {"input_ids": torch.tensor(padded, dtype=torch.long)}
        if return_attention_mask:
            output["attention_mask"] = torch.tensor(masks, dtype=torch.long)
        return output


class _FakeEnv:
    def get_tool_schemas(self):
        return []


class _RemoteFn:
    def __init__(self, fn):
        self._fn = fn

    async def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeRewardManager:
    def __init__(self):
        self.compute_score = _RemoteFn(self._compute_score)

    def _compute_score(self, data):
        uid = int(data.non_tensor_batch["uid"][0])
        return {uid: {"reward_score": 1.5, "reward_extra_info": {}}}


def _make_config():
    return OmegaConf.create(
        {
            "data": {"apply_chat_template_kwargs": {}},
            "reward_model": {"launch_reward_fn_async": False},
            "gen_actor_rollout_ref": {
                "rollout": {
                    "response_length": 32,
                    "prompt_length": 32,
                    "multi_turn": {"max_turns": 4, "format": "hermes"},
                    "agent": {"gamma": 0.0, "reward_bonus_coeff": 0.0, "traj_reward_mode": "traj"},
                }
            },
            "psrl": {"log_prob": {"enable_rollout_engine_log_prob": False}},
            "log_prob": {"enable_rollout_engine_log_prob": False},
        }
    )


def _make_request():
    return DataProto(
        batch={"input_ids": torch.tensor([[1]], dtype=torch.long)},
        non_tensor_batch={"uid": np.array([11]), "parent_id": np.array([7])},
        meta_info={},
    )


def _make_model_output(response_ids, logprobs):
    return DataProto(
        non_tensor_batch={
            "raw_response_ids": np.array([response_ids], dtype=object),
            "rollout_log_probs": np.array([logprobs], dtype=object),
            "rollout_instance_id": np.array([3]),
            "version_tag": np.array([5]),
        }
    )


class StepToolAgentDataTest(unittest.TestCase):
    def setUp(self):
        StepToolAgentData._class_initialized = False
        patcher = mock.patch(
            "psrl.workers.agent_loop.agent_data.tool_agent_data.ToolParser.get_tool_parser",
            return_value=_FakeParser(),
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.agent_data = StepToolAgentData(
            config=_make_config(),
            reward_manager=_FakeRewardManager(),
            tokenizer=_FakeTokenizer(),
            env=_FakeEnv(),
        )

    def test_step_tool_step_round_trip(self):
        step = StepToolStep(
            chat_completions=[{"role": "user", "content": "obs"}],
            observation="obs",
            thought="t",
            action=[{"type": "function"}],
            model_response="resp",
            info={"k": "v"},
            tool_reward=0.5,
            model_reward=0.25,
            reward=0.75,
            done=True,
            anchor_obs={"state": 1},
            step_idx=2,
            step_start=10,
            step_end=15,
            response_start=12,
            response_end=15,
        )

        restored = StepToolStep.from_dict(step.to_dict())

        self.assertEqual(restored.chat_completions, step.chat_completions)
        self.assertEqual(restored.observation, step.observation)
        self.assertEqual(restored.action, step.action)
        self.assertEqual(restored.model_response, step.model_response)
        self.assertEqual(restored.info, step.info)
        self.assertEqual(restored.tool_reward, step.tool_reward)
        self.assertEqual(restored.model_reward, step.model_reward)
        self.assertEqual(restored.reward, step.reward)
        self.assertEqual(restored.done, step.done)
        self.assertEqual(restored.anchor_obs, step.anchor_obs)
        self.assertEqual(restored.step_idx, step.step_idx)
        self.assertEqual(restored.step_start, step.step_start)
        self.assertEqual(restored.step_end, step.step_end)
        self.assertEqual(restored.response_start, step.response_start)
        self.assertEqual(restored.response_end, step.response_end)

    def test_exports_step_sidecar(self):
        request = _make_request()
        self.agent_data.init_trajectory(request)

        asyncio.run(self.agent_data.update_from_env([{"role": "user", "content": "obs0"}], 0.0, False, {"anchor_obs": "a0"}))
        asyncio.run(self.agent_data.update_from_model_token_ids(_make_model_output([11, 12], [-0.1, -0.2])))
        asyncio.run(self.agent_data.update_from_env([{"role": "tool", "content": "obs1"}], 0.5, False, {"anchor_obs": "a1"}))
        asyncio.run(self.agent_data.update_from_model_token_ids(_make_model_output([13], [-0.3])))

        output = asyncio.run(self.agent_data.finalize_output(request))
        non_tensor = output.non_tensor_batch

        self.assertEqual(non_tensor["step_count"][0], 2)
        self.assertEqual(non_tensor["step_idx"][0], [0, 1])
        self.assertEqual(non_tensor["anchor_obs"][0], ["a0", "a1"])
        self.assertEqual(non_tensor["step_reward"][0], [0.0, 0.5])
        self.assertEqual(non_tensor["step_done"][0], [False, False])

        step_starts = non_tensor["step_start"][0]
        step_ends = non_tensor["step_end"][0]
        response_starts = non_tensor["response_start"][0]
        response_ends = non_tensor["response_end"][0]

        self.assertEqual(step_starts[0], 0)
        self.assertLessEqual(step_starts[0], response_starts[0])
        self.assertLessEqual(response_starts[0], response_ends[0])
        self.assertLessEqual(response_ends[0], step_ends[0])
        self.assertLessEqual(step_starts[1], response_starts[1])
        self.assertLessEqual(response_starts[1], response_ends[1])
        self.assertLessEqual(response_ends[1], step_ends[1])
        self.assertGreaterEqual(step_starts[1], step_ends[0])
        self.assertGreaterEqual(response_starts[1], response_ends[0])

    def test_response_logprobs_align_with_full_response_ids(self):
        request = _make_request()
        self.agent_data.init_trajectory(request)

        asyncio.run(self.agent_data.update_from_env([{"role": "user", "content": "obs0"}], 0.0, False, {"anchor_obs": "a0"}))
        asyncio.run(self.agent_data.update_from_model_token_ids(_make_model_output([11], [-0.1])))
        asyncio.run(self.agent_data.update_from_env([{"role": "tool", "content": "obs1"}], 0.5, False, {"anchor_obs": "a1"}))
        asyncio.run(self.agent_data.update_from_model_token_ids(_make_model_output([12], [-0.2])))

        output = asyncio.run(self.agent_data.finalize_output(request))
        response_ids = output.non_tensor_batch["raw_response_ids"][0]
        rollout_log_probs = output.non_tensor_batch["rollout_log_probs"][0]
        prompt_length = len(self.agent_data.trajectory.prompt_ids)
        first_response_start = self.agent_data.trajectory.steps[0].response_start - prompt_length
        first_response_end = self.agent_data.trajectory.steps[0].response_end - prompt_length
        second_response_start = self.agent_data.trajectory.steps[1].response_start - prompt_length

        self.assertEqual(len(rollout_log_probs), len(response_ids))
        self.assertEqual(rollout_log_probs[0], 0.0)
        self.assertEqual(rollout_log_probs[first_response_start], -0.1)
        env_token_start = first_response_end
        self.assertEqual(rollout_log_probs[env_token_start], 0.0)
        self.assertEqual(rollout_log_probs[second_response_start], -0.2)

    def test_allows_empty_response_span(self):
        request = _make_request()
        self.agent_data.init_trajectory(request)

        asyncio.run(self.agent_data.update_from_env([{"role": "user", "content": "obs0"}], 0.0, False, {"anchor_obs": None}))
        asyncio.run(self.agent_data.update_from_model_token_ids(_make_model_output([], [])))

        output = asyncio.run(self.agent_data.finalize_output(request))
        non_tensor = output.non_tensor_batch

        self.assertEqual(non_tensor["step_count"][0], 1)
        self.assertEqual(non_tensor["response_start"][0][0], non_tensor["response_end"][0][0])

    def test_falls_back_to_observation_when_anchor_obs_missing(self):
        request = _make_request()
        self.agent_data.init_trajectory(request)

        observation = [{"role": "tool", "content": "obs0"}]
        asyncio.run(self.agent_data.update_from_env(observation, 0.0, False, {}))
        asyncio.run(self.agent_data.update_from_model_token_ids(_make_model_output([11], [-0.1])))

        output = asyncio.run(self.agent_data.finalize_output(request))

        self.assertEqual(output.non_tensor_batch["anchor_obs"][0], [observation])

    def test_preserves_explicit_anchor_obs_override(self):
        request = _make_request()
        self.agent_data.init_trajectory(request)

        observation = [{"role": "tool", "content": "obs0"}]
        asyncio.run(self.agent_data.update_from_env(observation, 0.0, False, {"anchor_obs": "anchor-0"}))
        asyncio.run(self.agent_data.update_from_model_token_ids(_make_model_output([11], [-0.1])))

        output = asyncio.run(self.agent_data.finalize_output(request))

        self.assertEqual(output.non_tensor_batch["anchor_obs"][0], ["anchor-0"])

    def test_step_reward_exports_raw_env_reward_not_adjusted_reward(self):
        request = _make_request()
        self.agent_data.init_trajectory(request)

        asyncio.run(self.agent_data.update_from_env([{"role": "tool", "content": "obs0"}], 0.5, False, {"anchor_obs": "a0"}))
        current_step = self.agent_data.get_current_step()
        current_step.reward = 9.5
        asyncio.run(self.agent_data.update_from_model_token_ids(_make_model_output([11], [-0.1])))

        output = asyncio.run(self.agent_data.finalize_output(request))

        self.assertEqual(output.non_tensor_batch["step_reward"][0], [0.5])

    def test_manager_post_process_preserves_step_sidecar(self):
        request = _make_request()
        self.agent_data.init_trajectory(request)

        asyncio.run(self.agent_data.update_from_env([{"role": "user", "content": "obs0"}], 0.0, False, {"anchor_obs": "a0"}))
        asyncio.run(self.agent_data.update_from_model_token_ids(_make_model_output([11, 12], [-0.1, -0.2])))
        output = asyncio.run(self.agent_data.finalize_output(request))

        dummy_manager = SimpleNamespace(
            tokenizer=_FakeTokenizer(),
            processor=None,
            config=_make_config(),
            log_prefix="test",
        )
        processed = PSRL_AgentLoopManager._post_process(dummy_manager, output)

        self.assertIn("step_count", processed.non_tensor_batch)
        self.assertIn("step_idx", processed.non_tensor_batch)
        self.assertEqual(processed.non_tensor_batch["step_count"][0], 1)
        self.assertEqual(processed.non_tensor_batch["anchor_obs"][0], ["a0"])


if __name__ == "__main__":
    unittest.main()
