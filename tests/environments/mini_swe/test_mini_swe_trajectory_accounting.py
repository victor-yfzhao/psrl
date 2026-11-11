"""
Verify Mini-SWE format error recovery and termination statistics.
"""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from examples.mini_swe.config import MiniSWEAgentRuntimeConfig
from examples.mini_swe.exceptions import MiniSweExceptionClassifier
from omegaconf import OmegaConf
from psrl.utils.concurrency import SlotManager
from psrl.workers.agent_loop.loops import mini_swe_agent_loop_v1 as loop_module
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.gen.utils import TokenOutput

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from minisweagent.models import get_model

    config = asdict(MiniSWEAgentRuntimeConfig().model)
    config.update(
        model_name="openai/test-model", cost_tracking="ignore_errors", action_regex=r"```bash\s*\n(.*?)\n```"
    )
    model = get_model(config=config)
    monkeypatch.setattr(model, "_calculate_cost", lambda response: {"cost": 0.125})
    return model


@pytest.mark.parametrize("repeated_errors", [False, True])
def test_default_agent_retries_format_errors_and_charges_every_call(
    model,
    monkeypatch,
    repeated_errors,
):
    from litellm import ModelResponse
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.exceptions import Submitted

    failed = "Analyze the task. No command."
    contents = (
        [failed] * 3 if repeated_errors else [failed, "```bash\necho inspect\n```", failed, "```bash\nsubmit\n```"]
    )
    responses = iter(ModelResponse(choices=[{"message": {"role": "assistant", "content": text}}]) for text in contents)
    requests = []
    executed = []

    def query(messages, **kwargs):
        requests.append(deepcopy(messages))
        return next(responses)

    def execute(action):
        executed.append(action["command"])
        if action["command"] == "submit":
            raise Submitted(
                {"role": "exit", "content": "patch", "extra": {"exit_status": "Submitted", "submission": "patch"}}
            )
        return {"returncode": 0, "output": "inspected", "exception_info": None}

    monkeypatch.setattr(model, "_query", query)
    environment = SimpleNamespace(get_template_vars=lambda: {}, serialize=lambda: {}, execute=execute)
    agent = DefaultAgent(
        model, environment, system_template="Use one shell action.", instance_template="{{task}}", cost_limit=0
    )
    result = agent.run("Fix the issue.")

    assistant_messages = [message for message in agent.messages if message["role"] == "assistant"]
    expected = [content for content in contents if content != failed]
    assert [message["content"] for message in assistant_messages] == expected, (
        "Only replies with a valid action must enter the assistant history."
    )
    assert agent.cost == 0.125 * len(contents), "Failed calls must be billed exactly once."
    assert executed == ([] if repeated_errors else ["echo inspect", "submit"]), (
        "Only valid actions must reach the environment."
    )
    assert requests[1][-1]["role"] == "user", "A format error must append user feedback before retrying."
    assert all(message.get("content") != failed for message in requests[1]), (
        "The retry prompt must omit the rejected assistant reply."
    )
    feedback = [
        message for message in agent.messages if message.get("extra", {}).get("interrupt_type") == "FormatError"
    ]
    assert all(message["extra"]["model_response"] == failed for message in feedback), (
        "Format feedback must retain the rejected text as diagnostic metadata."
    )
    assert all("response" in message["extra"] for message in feedback), (
        "The provider response must remain inspectable."
    )
    assert result["exit_status"] == ("RepeatedFormatError" if repeated_errors else "Submitted"), (
        "Format recovery must preserve the agent's exit condition."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,turns,expected",
    [
        ("Submitted", 50, TerminateReason.FINISHED),
        ("RepeatedFormatError", 3, TerminateReason.FORMAT_ERROR),
        ("LimitsExceeded", 50, TerminateReason.MAX_TURNS_EXCEEDED),
        ("TimeExceeded", 4, TerminateReason.AGENT_TIMEOUT),
        ("context_exceeded", 4, TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED),
    ],
)
async def test_loop_preserves_original_exit_status_without_discarding_completed_turns(
    monkeypatch,
    status,
    turns,
    expected,
):
    loop = loop_module.MiniSWEAgentLoopV1.__new__(loop_module.MiniSWEAgentLoopV1)
    loop.inference_orchestrator = None
    loop.reward_manager = loop.tokenizer = loop.processor = None
    loop.dataset_cls = object
    loop.config = OmegaConf.create(
        {"gen_actor_rollout_ref": {"rollout": {"agent": {"env": {"name": "test"}, "data": {"name": "test"}}}}}
    )
    loop.runtime_config = runtime = MiniSWEAgentRuntimeConfig()
    runtime.sandbox_config.max_parallel_tasks_per_worker = 2
    loop.rollout_config = OmegaConf.create({"response_length": 1000})
    loop.max_turns = 50
    environment = SimpleNamespace(reset=AsyncMock(return_value=({"runtime_config": runtime}, None)), close=AsyncMock())
    monkeypatch.setattr(loop_module.Environment, "get_environment", Mock(return_value=environment))
    data = loop_module.MiniSWEAgentData.__new__(loop_module.MiniSWEAgentData)
    data.reset = data.init_trajectory = data.set_patch = Mock()
    data.session_data = SimpleNamespace(trajectories=[SimpleNamespace()])
    output = TokenOutput(prompt_ids=[1], response_ids=[2, 3], response_mask=[1, 1])

    async def finalize_output():
        environment.close.assert_awaited_once()
        loop.delete_session.assert_awaited_once_with("session")
        assert loop.run_slots._semaphore._value == 2, "Reward submission must not retain a rollout permit."
        return output

    data.finalize_output = AsyncMock(side_effect=finalize_output)
    monkeypatch.setattr(loop_module.AgentData, "get_agent_data", Mock(return_value=data))
    loop.run_slots = SlotManager()
    loop.create_session = AsyncMock(return_value="session")
    loop.delete_session = AsyncMock()
    loop._run_agent = AsyncMock(return_value={"exit_status": status})
    loop.get_primary_training_data = AsyncMock(
        return_value={
            "prompt_ids": [1],
            "response_ids": [2, 3],
            "response_mask": [1, 1],
            "logprobs": [-0.1, -0.2],
            "num_turns": turns,
            "routed_experts": None,
        }
    )

    actual, reason = await loop.run({"uid": 1})

    assert actual is output and reason is expected, "Termination must reflect the agent's actual exit condition."
    assert reason.is_successful and not reason.needs_worker_retry(), "Completed generations must remain trainable."
    assert actual.extra_fields["agent_exit_status"] == status, "Training data must retain the original agent status."
    assert actual.extra_fields["assistant_turns"] == turns, "Assistant calls must include rejected actions."
    text = AgentLoopBase._build_summary_text(SimpleNamespace(), actual, reason)
    assert f"agent_exit_status: {status}" in text, "Trajectory dumps must expose the original agent status."
    assert f"stop: {expected.value}" in text, "Trajectory dumps must expose the classified termination."
    assert MiniSweExceptionClassifier().classify_exit_status(status) is expected, "Status consumers must agree."
    loop.delete_session.assert_awaited_once_with("session")
