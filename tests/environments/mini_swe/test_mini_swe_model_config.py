"""
Verify Mini-SWE model configuration and native transport handling.
"""

import json
from concurrent.futures import Future
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from examples.mini_swe import runner
from examples.mini_swe.config import MiniSWEAgentRuntimeConfig
from omegaconf import OmegaConf
from psrl.workers.agent_loop.loops import mini_swe_agent_loop_v1 as loop_module

pytestmark = pytest.mark.cpu_test


def _payload():
    return {
        "runtime_config": {
            "model": {
                "model_class": "litellm_textbased",
                "action_regex": r"```bash\s*\n(.*?)\n```",
            },
            "sandbox_config": {"rollout_turn_timeout": 30},
        },
        "sampling_params": {"top_k": 20, "repetition_penalty": 1.1, "ignore_eos": False},
        "chat_template_kwargs": {"enable_thinking": False},
        "base_url": "http://session-router/sessions/sid/v1",
        "model": "openai/model",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("enable_thinking", [False, True])
async def test_loop_forwards_resolved_data_template_options(monkeypatch, validate, enable_thinking):
    loop = loop_module.MiniSWEAgentLoopV1.__new__(loop_module.MiniSWEAgentLoopV1)
    data_config = OmegaConf.create(
        {
            "thinking": enable_thinking,
            "apply_chat_template_kwargs": {
                "enable_thinking": "${thinking}",
                "custom": {"modes": ["example"]},
            },
        }
    )
    loop.apply_chat_template_kwargs = data_config.apply_chat_template_kwargs
    loop.model_config = SimpleNamespace(path="model")
    loop.rollout_config = OmegaConf.create({"temperature": 1.0, "top_p": 0.9, "top_k": 20, "response_length": 100})
    loop.config = OmegaConf.create(
        {"train_actor_rollout_ref": {"rollout": {"val_kwargs": {"temperature": 0.0, "top_p": 1.0, "top_k": -1}}}}
    )
    loop.max_turns = 8
    loop.session_router_url = "http://session-router"
    loop.trajectory_id_strategy = "manual"
    monkeypatch.setattr(loop_module, "run_agent", lambda payload: payload)

    def submit(function, *args):
        future = Future()
        future.set_result(function(*args))
        return future

    monkeypatch.setattr(loop_module, "_RUNNER_THREAD_POOL", SimpleNamespace(submit=submit))
    payload = await loop._run_agent(
        {"validate": validate}, {"problem_statement": "Fix the bug."}, MiniSWEAgentRuntimeConfig(), "sid"
    )
    expected = {"enable_thinking": enable_thinking, "custom": {"modes": ["example"]}}
    assert payload["chat_template_kwargs"] == expected, "Training and validation must use the configured template."
    json.dumps(payload)
    model_kwargs = runner._build_model_config(payload)["model_kwargs"]
    assert model_kwargs["extra_body"]["chat_template_kwargs"] == expected, (
        "The runner must preserve resolved booleans and nested template options."
    )
    assert model_kwargs["temperature"] == (0.0 if validate else 1.0), (
        "Template forwarding must preserve the request's sampling mode."
    )


@pytest.mark.parametrize("top_k", [-1, 0, 20])
def test_model_config_preserves_extensions(top_k):
    payload = _payload()
    payload["sampling_params"] = {
        "top_k": top_k,
        "extra_body": {
            "separate_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
            "min_p": 0.1,
        },
    }
    original = deepcopy(payload)
    kwargs = runner._build_model_config(payload)["model_kwargs"]
    expected = {"separate_reasoning": True, "chat_template_kwargs": {"enable_thinking": False}, "min_p": 0.1}
    if top_k >= 0:
        expected["top_k"] = top_k
    assert kwargs["extra_body"] == expected, "Data template options must reach the request alongside sampling extras."
    assert "top_k" not in kwargs, "The OpenAI client requires top_k inside extra_body."
    assert payload == original, "Building a model config must not mutate the request payload."


def test_unspecified_template_options_preserve_the_model_default():
    payload = _payload()
    payload["chat_template_kwargs"] = {}
    payload["sampling_params"] = {"top_k": -1}
    kwargs = runner._build_model_config(payload)["model_kwargs"]
    assert kwargs["extra_body"] == {}, "Unspecified template and reasoning options must retain their defaults."


@pytest.mark.parametrize(
    "status,headers,message,expected,attempts",
    [
        (400, {"x-smg-error-code": "request_aborted"}, "Request aborted.", "abort", 1),
        (400, {}, "The decoder prompt exceeds the maximum model length.", "overflow", 1),
        (401, {}, "Authentication failed.", "auth", 1),
        (400, {}, "Other invalid request.", "bad_request", 3),
        (503, {}, "Service unavailable.", "transient", 3),
    ],
)
def test_native_model_transport_preserves_abort_overflow_and_retry_handling(
    monkeypatch,
    status,
    headers,
    message,
    expected,
    attempts,
):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "3")

    import importlib

    from litellm import AuthenticationError, BadRequestError, ServiceUnavailableError
    from minisweagent.models import get_model
    from openai import OpenAI
    from psrl.utils.agent.gateway_abort import ensure_gateway_abort_handling
    from psrl.utils.agent.overflow import PromptOverflowError, ensure_overflow_handling
    from psrl.utils.common.http_utils import RequestAbortedByGatewayError
    from tenacity import wait_none

    retry_module = importlib.import_module("minisweagent.models.utils.retry")
    monkeypatch.setattr(retry_module, "wait_exponential", lambda **kwargs: wait_none())
    captured = []

    def respond(request):
        captured.append(request)
        error = {"message": message}
        if "x-smg-error-code" in headers:
            error["code"] = headers["x-smg-error-code"]
        return httpx.Response(status, headers=headers, json={"error": error})

    model = get_model(config=runner._build_model_config(_payload()))
    model.config.model_kwargs["max_retries"] = 0
    model.config.model_kwargs["client"] = OpenAI(
        base_url="http://session-router/sessions/sid/v1",
        api_key="EMPTY",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    ensure_overflow_handling(model)
    ensure_gateway_abort_handling(model)
    error_types = {
        "abort": RequestAbortedByGatewayError,
        "overflow": PromptOverflowError,
        "auth": AuthenticationError,
        "bad_request": BadRequestError,
        "transient": ServiceUnavailableError,
    }
    with pytest.raises(error_types[expected]):
        model.query([{"role": "user", "content": "Continue."}])

    assert len(captured) == attempts, "Only recoverable model requests may retry."
