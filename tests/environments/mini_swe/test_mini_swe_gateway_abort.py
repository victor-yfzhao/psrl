"""
Stop deliberate gateway aborts before model retries, grading, or trajectory publication.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from examples.mini_swe import runner
from examples.mini_swe.config import MiniSWEAgentRuntimeConfig
from examples.mini_swe.exceptions import MiniSweExceptionClassifier
from omegaconf import OmegaConf
from psrl.utils.agent.gateway_abort import ensure_gateway_abort_handling
from psrl.utils.common.http_utils import RequestAbortedByGatewayError
from psrl.utils.concurrency import SlotManager
from psrl.workers.agent_loop.loops import mini_swe_agent_loop_v1 as loop_module
from psrl.workers.agent_loop.loops.utils import TerminateReason
from tenacity import wait_none

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "3")
    retry_module = importlib.import_module("minisweagent.models.utils.retry")
    monkeypatch.setattr(retry_module, "wait_exponential", lambda **kwargs: wait_none())
    from minisweagent.models.litellm_model import LitellmModel

    return LitellmModel(model_name="openai/test-model", cost_tracking="ignore_errors")


def bad_request(headers=None, body=None, response_body=None):
    from litellm import BadRequestError

    response = httpx.Response(
        400,
        request=httpx.Request("POST", "http://gateway/v1/chat/completions"),
        headers=headers,
        json=response_body,
    )
    return BadRequestError(
        "Request aborted by PS Manager", model="test-model", llm_provider="openai", response=response, body=body
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_real_model_query_retries_a_bare_abort_but_stops_the_typed_sentinel(model, wrapped):
    error = bad_request(headers={"x-smg-error-code": "request_aborted"})
    query = Mock(side_effect=error)
    model._query = query
    if wrapped:
        ensure_gateway_abort_handling(model)
    expected = RequestAbortedByGatewayError if wrapped else type(error)
    with pytest.raises(expected) as raised:
        model.query([{"role": "user", "content": "Continue."}])

    assert query.call_count == (1 if wrapped else 3), "Only a deliberate gateway abort may bypass model retries."
    if wrapped:
        assert raised.value.__cause__ is error, "Preserve the original structured HTTP error for diagnosis."


@pytest.mark.parametrize(
    "headers,body,response_body",
    [
        ({"X-SMG-Error-Code": "request_aborted"}, None, None),
        ({}, {"code": "request_aborted"}, None),
        ({}, {"error": {"code": "request_aborted"}}, None),
        ({}, None, {"error": {"code": "request_aborted"}}),
    ],
)
def test_structured_abort_codes_survive_the_real_model_retry_loop(model, headers, body, response_body):
    model._query = query = Mock(side_effect=bad_request(headers, body, response_body))
    ensure_gateway_abort_handling(model)
    with pytest.raises(RequestAbortedByGatewayError):
        model.query([{"role": "user", "content": "Continue."}])

    assert query.call_count == 1, "Header and structured body sentinels must terminate the first attempt."


@pytest.mark.parametrize(
    "headers,body",
    [
        ({}, {"message": "Request aborted by PS Manager"}),
        ({}, {"error": {"code": "other", "message": "request_aborted"}}),
        ({}, "request_aborted"),
        ({"x-smg-error-code": "request_aborted_extra"}, None),
        ({"x-smg-error-code": "invalid_request"}, {"code": "request_aborted"}),
    ],
)
def test_other_bad_requests_keep_their_original_errors_and_retries(model, headers, body):
    error = bad_request(headers, body)
    model._query = query = Mock(side_effect=error)
    ensure_gateway_abort_handling(model)
    with pytest.raises(type(error)) as raised:
        model.query([{"role": "user", "content": "Continue."}])

    assert raised.value is error, "Unproven abort messages must retain their original error type and identity."
    assert query.call_count == 3, "Other HTTP 400 failures must retain the client's retry policy."


def test_transient_http_failures_still_retry_even_with_an_abort_code(model):
    from openai import APIStatusError

    response = httpx.Response(
        503,
        request=httpx.Request("POST", "http://gateway/v1/chat/completions"),
        headers={"x-smg-error-code": "request_aborted"},
    )
    error = APIStatusError("Transient failure.", response=response, body={"code": "request_aborted"})
    model._query = query = Mock(side_effect=error)
    ensure_gateway_abort_handling(model)
    with pytest.raises(APIStatusError) as raised:
        model.query([{"role": "user", "content": "Continue."}])

    assert raised.value is error, "Only HTTP 400 may carry the terminal abort sentinel."
    assert query.call_count == 3, "Transient failures must retain the client's retry policy."


def test_abort_wrapper_is_idempotent_and_does_not_mutate_class_retry_configuration(model):
    original_types = list(type(model).abort_exceptions)
    model._query = query = Mock(side_effect=bad_request(headers={"x-smg-error-code": "request_aborted"}))
    ensure_gateway_abort_handling(model)
    first_wrapper = model._query
    ensure_gateway_abort_handling(model)

    assert model._query is first_wrapper, "Repeated setup must not add another query wrapper."
    assert model.abort_exceptions.count(RequestAbortedByGatewayError) == 1, "Register the sentinel once per model."
    assert type(model).abort_exceptions == original_types, "A model must not change other model instances."
    with pytest.raises(RequestAbortedByGatewayError):
        model.query([{"role": "user", "content": "Continue."}])
    assert query.call_count == 1, "Idempotent wrapping must preserve terminal behavior."


def test_runner_aborts_without_grading_and_cleans_its_environment(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from minisweagent import environments, models
    from minisweagent.agents import default

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, task):
            raise RequestAbortedByGatewayError("uid", "Deliberate gateway abort.")

    environment = SimpleNamespace(cleanup=Mock())
    model = SimpleNamespace(abort_exceptions=[], _query=Mock())
    monkeypatch.setattr(default, "DefaultAgent", Agent)
    monkeypatch.setattr(environments, "get_environment", Mock(return_value=environment))
    monkeypatch.setattr(models, "get_model", Mock(return_value=model))
    monkeypatch.setattr(runner, "_silence_litellm", lambda: None)
    grader = Mock(side_effect=AssertionError("An aborted task must not reach grading."))
    from examples.mini_swe import swebench_grader

    monkeypatch.setattr(swebench_grader, "grade_fresh_container", grader)
    result = runner.run_agent(
        {
            "runtime_config": {
                "sandbox_config": {"environment": {}, "rollout_turn_timeout": 60},
                "model": {},
                "agent": {"problem_template": "task"},
            },
            "observation": {},
            "sampling_params": {},
            "base_url": "http://gateway",
            "model": "test-model",
            "max_turns": 2,
            "task": "Fix the issue.",
        }
    )

    assert result["exit_status"] == "aborted", "A terminal abort must not become a rollout error."
    assert result["submission"] == "", "An aborted episode has no submission."
    environment.cleanup.assert_called_once()
    grader.assert_not_called()


@pytest.mark.asyncio
async def test_loop_releases_aborted_episode_without_fetching_training_data(monkeypatch):
    loop = loop_module.MiniSWEAgentLoopV1.__new__(loop_module.MiniSWEAgentLoopV1)
    loop.inference_orchestrator = None
    loop.reward_manager = None
    loop.tokenizer = None
    loop.processor = None
    loop.dataset_cls = object
    loop.config = OmegaConf.create(
        {
            "gen_actor_rollout_ref": {"rollout": {"agent": {"env": {"name": "test"}, "data": {"name": "test"}}}},
            "trainer": {"project_name": "project", "experiment_name": "experiment"},
        }
    )
    loop.runtime_config = runtime = MiniSWEAgentRuntimeConfig()
    runtime.sandbox_config.max_parallel_tasks_per_worker = 2
    environment = SimpleNamespace(reset=AsyncMock(return_value=({"runtime_config": runtime}, None)), close=AsyncMock())
    monkeypatch.setattr(loop_module.Environment, "get_environment", Mock(return_value=environment))
    data = loop_module.MiniSWEAgentData.__new__(loop_module.MiniSWEAgentData)
    data.reset = Mock()
    data.init_trajectory = Mock()
    monkeypatch.setattr(loop_module.AgentData, "get_agent_data", Mock(return_value=data))
    loop.run_slots = SlotManager()
    acquire = AsyncMock(return_value=(7, 9))
    release = Mock()
    monkeypatch.setattr(loop.run_slots, "acquire", acquire)
    monkeypatch.setattr(loop.run_slots, "release", release)
    loop.create_session = AsyncMock(return_value="sid")
    loop.delete_session = AsyncMock()
    loop._run_agent = AsyncMock(return_value={"exit_status": "aborted"})
    loop.get_primary_training_data = AsyncMock(side_effect=AssertionError("Aborted data must not be fetched."))

    output, reason = await loop.run({"uid": 1})

    assert output is None and reason is TerminateReason.ABORTED, "A deliberate abort must not publish a trajectory."
    assert not reason.needs_manager_retry() and not reason.needs_worker_retry(), (
        "Aborted groups must not refill twice."
    )
    loop.get_primary_training_data.assert_not_awaited()
    environment.close.assert_awaited_once()
    release.assert_called_once_with((7, 9))
    loop.delete_session.assert_awaited_once_with("sid")


def test_classifier_uses_the_same_nonretryable_aborted_lifecycle():
    reason = MiniSweExceptionClassifier().classify_exit_status("aborted")

    assert reason is TerminateReason.ABORTED, "Runner statuses and loop termination must use the same lifecycle."
    assert not reason.needs_manager_retry() and not reason.needs_worker_retry(), "Deliberate aborts must never refill."
