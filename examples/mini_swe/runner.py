"""Run one mini-SWE-agent task through its standard Python bindings."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Any

from psrl.utils.agent.gateway_abort import ensure_gateway_abort_handling
from psrl.utils.common.http_utils import RequestAbortedByGatewayError
from psrl.utils.rollout.loop_timer import LoopTimer

from examples.mini_swe.task_setup import initialize_benchmark_task, initialize_smith_task

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

psrl_logger = logging.getLogger(__name__)
logging.getLogger("minisweagent.environment").setLevel(logging.WARNING)


def _silence_litellm() -> None:
    """Suppress LiteLLM's BadRequestError banner (print, not logging).

    LiteLLM's exception mapper always ``print``s
    ``Give Feedback / Get Help`` + ``LiteLLM.Info: ...`` unless
    ``litellm.suppress_debug_info`` is True. Logger level tweaks do nothing.
    """
    try:
        import litellm

        litellm.suppress_debug_info = True
    except Exception:
        pass
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("litellm").setLevel(logging.ERROR)


_PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
]


def _build_environment_config(payload: dict[str, Any]) -> dict[str, Any]:
    environment = dict(payload["runtime_config"]["sandbox_config"]["environment"])
    observation = payload["observation"]
    use_preexisting_repo = observation.get("use_preexisting_repo", True)
    preexisting_repo_name = observation.get("preexisting_repo_name", "")
    if use_preexisting_repo and preexisting_repo_name:
        environment["cwd"] = f"/{preexisting_repo_name}"

    run_args = list(environment.get("run_args", []))
    swe_task_id = observation.get("swe_task_id", "")
    if swe_task_id:
        run_args.extend(["--label", f"psrl.swe_task_id={swe_task_id}"])
    actor_id = payload.get("actor_id", "")
    if actor_id:
        run_args.extend(["--label", f"psrl.actor_id={actor_id}"])
    if not use_preexisting_repo and observation.get("repo_path"):
        run_args.extend(["--volume", f"{observation['repo_path']}:/testbed"])

    environment.update(
        {
            "environment_class": environment.get("environment_class", "docker"),
            "forward_env": _PROXY_ENV_KEYS,
            "run_args": run_args,
        }
    )
    environment.pop("grader_memory", None)
    environment.pop("grader_shm_size", None)
    environment.pop("grader_timeout", None)
    environment.pop("grader_command_timeout", None)
    environment.pop("grader_max_attempts", None)
    return environment


def _build_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(payload["runtime_config"]["model"])
    model_kwargs = dict(payload["sampling_params"])
    extra_body = dict(model_kwargs.pop("extra_body", {}) or {})
    top_k = model_kwargs.pop("top_k", None)
    if top_k is not None and int(top_k) >= 0:
        extra_body["top_k"] = int(top_k)
    for key in ("repetition_penalty", "ignore_eos"):
        if key in model_kwargs:
            extra_body[key] = model_kwargs.pop(key)
    if payload.get("chat_template_kwargs"):
        extra_body["chat_template_kwargs"] = dict(payload["chat_template_kwargs"])
    model_kwargs["extra_body"] = extra_body
    extra_headers = dict(model_kwargs.get("extra_headers") or {})
    if payload.get("trajectory_id_strategy", "manual") == "manual":
        extra_headers["x-smg-tito-trajectory-id"] = "0"
    else:
        extra_headers.pop("x-smg-tito-trajectory-id", None)
    if extra_headers:
        model_kwargs["extra_headers"] = extra_headers
    else:
        model_kwargs.pop("extra_headers", None)
    model_kwargs.update(
        {
            "api_base": payload["base_url"],
            "api_key": "EMPTY",
            "timeout": payload["runtime_config"]["sandbox_config"]["rollout_turn_timeout"],
        }
    )
    model_config.update(
        {
            "model_name": payload["model"],
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
        }
    )
    return model_config


def run_agent(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one task through mini-SWE-agent's standard Python bindings."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model
    from psrl.utils.agent.overflow import PromptOverflowError, ensure_overflow_handling

    _silence_litellm()

    class _TimedAgent(DefaultAgent):
        """DefaultAgent that accumulates wall-clock model vs environment time.

        ``step()`` decomposes into ``query()`` (model/assistant turn) and
        ``execute_actions()`` (environment/tool execution), so timing each gives a
        clean assistant-vs-env split for the trajectory summary.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.timer = LoopTimer()

        def query(self, *args: Any, **kwargs: Any) -> Any:
            with self.timer.generation():
                return super().query(*args, **kwargs)

        def execute_actions(self, *args: Any, **kwargs: Any) -> Any:
            with self.timer.env():
                return super().execute_actions(*args, **kwargs)

    environment = None
    agent = None
    run_start = time.perf_counter()
    # Return dictionaries share `timing`, so the finalizer records elapsed values
    # for success and exception paths.
    timing: dict[str, float | list[float]] = {
        "prep_s": 0.0,
        "generation_s": 0.0,
        "env_s": 0.0,
        "elapsed_s": 0.0,
        "generation_turns_s": [],
        "env_turns_s": [],
    }
    try:
        environment = get_environment(_build_environment_config(payload))
        observation = payload["observation"]
        if observation.get("swe_restore_tests", False):
            initialize_smith_task(environment, observation.get("swe_problem", {}))
        elif observation.get("swe_grader") == "swebench_fresh_container":
            initialize_benchmark_task(environment, observation["swe_problem"])
        model = get_model(config=_build_model_config(payload))
        ensure_overflow_handling(model)
        ensure_gateway_abort_handling(model)
        agent_config = dict(payload["runtime_config"]["agent"])
        agent_config["instance_template"] = agent_config.pop("problem_template")
        agent_config["step_limit"] = payload["max_turns"]
        agent_config["output_path"] = None
        timing["prep_s"] = time.perf_counter() - run_start
        agent = _TimedAgent(model, environment, **agent_config)
        result = agent.run(payload["task"])
        patch = result.get("submission", "") or ""
        return {
            "exit_status": result.get("exit_status", ""),
            "submission": patch,
            "timing": timing,
        }
    except RequestAbortedByGatewayError:
        return {
            "exit_status": "aborted",
            "submission": "",
            "timing": timing,
        }
    except PromptOverflowError as exc:
        # Preserve turns generated before context overflow and terminate normally.
        psrl_logger.warning("mini-SWE-agent stopped on context overflow: %s.", exc)
        return {
            "exit_status": "context_exceeded",
            "submission": "",
            "timing": timing,
        }
    except Exception as exc:
        error = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            error = f"{error} stderr: {(exc.stderr or '').strip()} stdout: {(exc.stdout or '').strip()}"
        psrl_logger.warning(f"mini-SWE-agent task failed: {error}.", exc_info=True)
        return {
            "exit_status": "error",
            "submission": "",
            "error": error,
            "timing": timing,
        }
    finally:
        # Shared return dictionaries receive final timing before control leaves.
        if agent is not None:
            timing.update(agent.timer.as_dict())
        timing["elapsed_s"] = time.perf_counter() - run_start
        if environment is not None:
            try:
                environment.cleanup()
            except Exception:
                psrl_logger.warning("mini-SWE-agent environment cleanup failed.", exc_info=True)
