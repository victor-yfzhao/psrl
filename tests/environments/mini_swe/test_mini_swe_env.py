"""
Verify simple and benchmark datasets through mini-SWE environment setup and reward routing.
"""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
from examples.mini_swe.config import build_runtime_config
from examples.mini_swe.prepare.prepare_simple_data import generate_simple_test_data
from examples.mini_swe.reward import compute_score
from examples.mini_swe.runner import _build_environment_config
from omegaconf import OmegaConf
from psrl.environments.mini_swe_env import MiniSWEEnvironment
from psrl.workers.agent_loop.agent_data.mini_swe_agent_data import MiniSWEAgentData

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def environment():
    path = Path(__file__).resolve().parents[3] / "examples/mini_swe/config/simple_agent_config.yaml"
    runtime_config = build_runtime_config(OmegaConf.load(path)[0])
    return MiniSWEEnvironment(OmegaConf.create({}), None, None, runtime_config=runtime_config)


@pytest.mark.asyncio
@pytest.mark.parametrize("split,repo_name", [("train", "train_0"), ("test", "val_0")])
@pytest.mark.parametrize("serialize_extra_info", [False, True])
async def test_simple_parquet_reaches_patch_reward_without_benchmark_grader(
    environment, tmp_path, split, repo_name, serialize_extra_info
):
    path = tmp_path / f"{split}.parquet"
    generate_simple_test_data(1, split).to_parquet(path)
    task = pd.read_parquet(path).iloc[0].to_dict()
    expected_patch = task["reward_model"]["ground_truth"]
    if serialize_extra_info:
        task["extra_info"] = json.dumps(task["extra_info"])

    observation, _ = await environment.reset(task)
    payload = {"runtime_config": asdict(observation["runtime_config"]), "observation": observation}
    sandbox = _build_environment_config(payload)

    assert observation["problem_statement"] == task["prompt"][0]["content"], "The task text must reach the runner."
    assert sandbox["cwd"] == f"/{repo_name}", "Simple episodes must use the prepared repository."
    assert sandbox["image"] == "python:3.11-slim", "Simple episodes must retain the configured image."
    assert "swe_grader" not in observation, "Simple patches must be scored without a benchmark grader."
    reward = compute_score(
        task["data_source"],
        solution_str="",
        ground_truth=expected_patch,
        extra_info={"patch": expected_patch, "num_turns": 1},
    )
    assert reward == 1.0, "The expected simple patch must receive full reward."


def _benchmark_task(data_source):
    return {
        "data_source": data_source,
        "extra_info": {
            "problem_statement": "Fix the failing test.",
            "swe_grader": "swebench_fresh_container",
            "swe_problem": {"instance_id": "example__repo-1", "FAIL_TO_PASS": ["test_fix"]},
            "swe_problem_image": "example/swe:latest",
            "swe_restore_tests": data_source == "swe_smith_py",
            "sandbox_overrides": {"environment": {"image": "example/swe:latest", "cwd": "/testbed"}},
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("data_source", ["swebench_verified", "swe_smith_py", "swe_gym"])
@pytest.mark.parametrize("grader_timeout", [1800, 2400])
async def test_benchmark_exports_resolved_grading_inputs(environment, data_source, grader_timeout):
    task = _benchmark_task(data_source)
    grading_overrides = {
        "grader_timeout": grader_timeout,
        "grader_command_timeout": 75,
        "grader_max_attempts": 2,
    }
    task["extra_info"]["sandbox_overrides"]["environment"].update(grading_overrides)
    if data_source in ("swe_gym", "swebench_verified"):
        task["extra_info"]["swe_problem"]["eval_script"] = "pytest"

    observation, _ = await environment.reset(task)
    payload = {"runtime_config": asdict(observation["runtime_config"]), "observation": observation}
    sandbox = _build_environment_config(payload)

    assert sandbox["image"] == task["extra_info"]["swe_problem_image"], "Benchmark image overrides must survive."
    assert sandbox["cwd"] == "/testbed", "Benchmark episodes must use the image's testbed."
    for key in grading_overrides:
        assert key not in sandbox, "Grading settings must not leak into the rollout Docker constructor."
    data = MiniSWEAgentData.__new__(MiniSWEAgentData)
    data.set_grading_context(observation)
    context = data.grading_context
    expected_kind = {"swebench_verified": "verified", "swe_smith_py": "smith", "swe_gym": "gym"}[data_source]
    assert context["swe_problem"] == task["extra_info"]["swe_problem"], "The service must receive the complete task."
    assert context["swe_grader_kind"] == expected_kind, "Each benchmark must select its own harness."
    assert context["swe_problem_image"] == task["extra_info"]["swe_problem_image"], "The grading image must survive."
    assert context["swe_task_id"] == observation["swe_task_id"], (
        "Sibling episodes must retain distinct cleanup labels."
    )
    environment_config = payload["runtime_config"]["sandbox_config"]["environment"]
    for key, value in grading_overrides.items():
        assert context["environment"][key] == value, "Task grading overrides must reach the reward service."
    for key in ("run_args", "env", "grader_memory", "grader_shm_size"):
        assert context["environment"][key] == environment_config[key], (
            "Effective resource limits must reach the service."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["swe_grader", "swe_problem", "swe_problem_image"])
async def test_incomplete_benchmark_is_rejected(environment, field):
    task = _benchmark_task("swebench_verified")
    del task["extra_info"][field]

    with pytest.raises(ValueError, match=field):
        await environment.reset(task)


@pytest.mark.asyncio
async def test_unknown_source_cannot_silently_skip_benchmark_grading(environment):
    with pytest.raises(ValueError, match="Unsupported mini-SWE data source"):
        await environment.reset({"data_source": "unknown", "extra_info": {}})
