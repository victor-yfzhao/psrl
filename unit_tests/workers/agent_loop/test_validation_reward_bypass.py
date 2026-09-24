import asyncio
import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


class _RemoteMethod:
    def __init__(self, return_value):
        self.return_value = return_value
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def result():
            return self.return_value

        return result()


class _Output:
    def __init__(self, *, validate: bool, batch_size: int = 1):
        self.meta_info = {"validate": validate, "eos_token_id": 2}
        self.non_tensor_batch = {
            "uid": np.arange(batch_size),
            "raw_response_ids": np.array([[10, 11, 2]] * batch_size, dtype=object),
        }

    def __len__(self):
        return len(self.non_tensor_batch["uid"])


def _restore_modules(previous_modules):
    for name, previous_module in previous_modules.items():
        if previous_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module


def _load_generation_loop_classes():
    class AgentLoopBase:
        def _post_process_and_merge_reward(self, reward_result, outputs):
            outputs.merged_reward_result = reward_result
            return outputs

    class TerminateReason(Enum):
        FINISHED = "finished"
        UNKNOWN = "unknown"

    class RolloutGatewayClient:
        @classmethod
        def from_config(cls, config):
            raise AssertionError("server rollout is disabled in this test")

    agent_loop_package = types.ModuleType("pivotrl.workers.agent_loop")
    agent_loop_package.__path__ = []
    loops_package = types.ModuleType("pivotrl.workers.agent_loop.loops")
    loops_package.__path__ = []
    base_module = types.ModuleType("pivotrl.workers.agent_loop.loops.base_agent_loop")
    base_module.AgentLoopBase = AgentLoopBase
    gateway_module = types.ModuleType("pivotrl.workers.agent_loop.gateway_client")
    gateway_module.RolloutGatewayClient = RolloutGatewayClient
    utils_module = types.ModuleType("pivotrl.workers.agent_loop.loops.utils")
    utils_module.TerminateReason = TerminateReason
    utils_module.register = lambda name: lambda cls: cls
    verl_module = types.ModuleType("verl")
    verl_module.DataProto = _Output

    stub_modules = {
        "pivotrl.workers.agent_loop": agent_loop_package,
        "pivotrl.workers.agent_loop.loops": loops_package,
        "pivotrl.workers.agent_loop.loops.base_agent_loop": base_module,
        "pivotrl.workers.agent_loop.gateway_client": gateway_module,
        "pivotrl.workers.agent_loop.loops.utils": utils_module,
        "verl": verl_module,
    }
    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)
    try:
        classes = []
        for module_name, file_name, class_name in (
            ("generate_agent_loop_for_reward_bypass_test", "generate_agent_loop.py", "GenerateAgentLoop"),
            (
                "batch_generate_agent_loop_for_reward_bypass_test",
                "batch_generate_agent_loop.py",
                "BatchGenerateAgentLoop",
            ),
        ):
            module_path = REPO_ROOT / "pivotrl" / "workers" / "agent_loop" / "loops" / file_name
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            classes.append((getattr(module, class_name), TerminateReason))
        return classes
    finally:
        _restore_modules(previous_modules)


GENERATION_LOOP_CLASSES = _load_generation_loop_classes()


def _make_generation_loop(loop_class, *, output: _Output, reward_method: _RemoteMethod):
    loop = loop_class.__new__(loop_class)
    loop.config = SimpleNamespace(
        pivotrl=SimpleNamespace(server_rollout=SimpleNamespace(enable=False)),
        reward_models_config=SimpleNamespace(launch_reward_fn_async=False),
    )
    loop.response_length = 16
    loop.rollout_router = SimpleNamespace(
        generate_async=_RemoteMethod(output),
        generate=_RemoteMethod(output),
    )
    loop.reward_manager = SimpleNamespace(compute_score=reward_method)
    return loop


@pytest.mark.parametrize("loop_class,terminate_reason", GENERATION_LOOP_CLASSES)
def test_generation_loops_skip_reward_rpc_for_validation(loop_class, terminate_reason):
    async def scenario():
        output = _Output(validate=True, batch_size=2)
        reward_method = _RemoteMethod({})
        loop = _make_generation_loop(loop_class, output=output, reward_method=reward_method)

        result, reason = await loop.run(SimpleNamespace())

        assert result is output
        assert reason is terminate_reason.FINISHED
        assert reward_method.calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("loop_class,terminate_reason", GENERATION_LOOP_CLASSES)
def test_generation_loops_keep_reward_rpc_for_training(loop_class, terminate_reason):
    async def scenario():
        output = _Output(validate=False, batch_size=2)
        reward_result = {0: {"reward_score": 1.0}, 1: {"reward_score": 0.0}}
        reward_method = _RemoteMethod(reward_result)
        loop = _make_generation_loop(loop_class, output=output, reward_method=reward_method)

        result, reason = await loop.run(SimpleNamespace())

        assert reason is terminate_reason.FINISHED
        assert reward_method.calls == [((output,), {})]
        assert result.merged_reward_result == reward_result

    asyncio.run(scenario())


def _load_agent_data_types():
    class DataProto:
        def __init__(self, *, non_tensor_batch=None, meta_info=None, batch=None):
            self.non_tensor_batch = non_tensor_batch or {}
            self.meta_info = meta_info or {}
            self.batch = batch

    environments_module = types.ModuleType("pivotrl.environments.base")
    environments_module.ConversationType = list
    environments_module.Environment = object
    metrics_module = types.ModuleType("pivotrl.utils.reward_token_metrics")
    metrics_module.extract_reward_model_token_counts = lambda extra_info: (0, 0)
    omegaconf_module = types.ModuleType("omegaconf")
    omegaconf_module.DictConfig = dict
    ray_module = types.ModuleType("ray")
    ray_module.actor = SimpleNamespace(ActorHandle=object)
    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = object
    verl_module = types.ModuleType("verl")
    verl_module.DataProto = DataProto

    stub_modules = {
        "pivotrl.environments.base": environments_module,
        "pivotrl.utils.reward_token_metrics": metrics_module,
        "omegaconf": omegaconf_module,
        "ray": ray_module,
        "transformers": transformers_module,
        "verl": verl_module,
    }
    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)
    try:
        module_path = REPO_ROOT / "pivotrl" / "workers" / "agent_loop" / "agent_data" / "base.py"
        spec = importlib.util.spec_from_file_location("agent_data_base_for_reward_bypass_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.AgentData, module.Trajectory, DataProto
    finally:
        sys.modules.pop("agent_data_base_for_reward_bypass_test", None)
        _restore_modules(previous_modules)


AgentData, Trajectory, DataProto = _load_agent_data_types()


class _ConcreteAgentData(AgentData):
    def format_chat_completions(self, observation, *, is_init):
        return []

    def reset(self):
        return None

    def encode_observation(self, observation, *, is_init):
        return [], is_init

    def decode_action_from_token_ids(self, token_ids):
        return token_ids


def _make_agent_data(*, reward_method: _RemoteMethod):
    agent_data = _ConcreteAgentData.__new__(_ConcreteAgentData)
    agent_data.config = SimpleNamespace(
        gen_actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(
                response_length=16,
                agent=SimpleNamespace(traj_reward_mode="traj"),
            )
        ),
        reward_models_config=SimpleNamespace(launch_reward_fn_async=False),
    )
    agent_data.trajectory = Trajectory(prompt_ids=[1], response_ids=[2], response_mask=[1])
    agent_data.reward_manager = SimpleNamespace(compute_score=reward_method)
    agent_data._post_process_and_merge_reward = MethodType(
        lambda self, reward_result, output: setattr(output, "merged_reward_result", reward_result) or output,
        agent_data,
    )
    return agent_data


def test_trajectory_reward_skips_rpc_for_validation_and_keeps_training_behavior():
    async def scenario():
        validation_reward = _RemoteMethod({})
        validation_agent_data = _make_agent_data(reward_method=validation_reward)
        validation_request = DataProto(
            non_tensor_batch={"uid": np.array([7])},
            meta_info={"validate": True},
        )

        validation_output = await validation_agent_data.finalize_output(validation_request)

        assert validation_output.meta_info["validate"] is True
        assert validation_reward.calls == []

        training_result = {7: {"reward_score": 1.0}}
        training_reward = _RemoteMethod(training_result)
        training_agent_data = _make_agent_data(reward_method=training_reward)
        training_request = DataProto(
            non_tensor_batch={"uid": np.array([7])},
            meta_info={"validate": False},
        )

        training_output = await training_agent_data.finalize_output(training_request)

        assert training_reward.calls == [((training_output,), {})]
        assert training_output.merged_reward_result == training_result

    asyncio.run(scenario())
