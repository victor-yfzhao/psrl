"""
Validate task baseline isolation, concurrent deduplication, and fail closed readiness caching.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from examples.mini_swe import swebench_grader
from examples.mini_swe.benchmark_grading import syntax_probe_manifest

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def control(tmp_path, monkeypatch):
    monkeypatch.setattr(swebench_grader, "_BASELINE_CACHE_DIRECTORY", tmp_path / "baseline")
    report = {"resolved_by": "harness", "resolved": False, "test_results_observed": 2}
    evaluator = Mock(return_value=report)
    monkeypatch.setattr(swebench_grader, "grade_fresh_container", evaluator)
    task = {
        "instance_id": "example__repo-1",
        "base_commit": "a" * 40,
        "eval_script": "pytest -rA tests",
        "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_keep.py::test_keep"],
    }

    def evaluate(**overrides):
        options = {
            "swe_problem": task,
            "grader_kind": "gym",
            "image_id": "sha256:" + "b" * 64,
            "timeout": 900,
            "swe_task_id": "episode-1",
            "memory": "30g",
            "run_args": ["--cpus=8", "--network", "host"],
            "env": {"OMP_NUM_THREADS": "1"},
            "shm_size": "16g",
        }
        options.update(overrides)
        return swebench_grader._get_baseline_report(**options)

    return evaluate, evaluator, task


def test_baseline_uses_the_exact_task_image_and_environment_without_model_source_changes(control):
    evaluate, evaluator, task = control
    run_args = ["--cpus=8", "--network", "host", "--label", "psrl.actor_id=reward-loop"]
    result = evaluate(run_args=run_args)
    args, kwargs = evaluator.call_args

    assert args[0] == task, "Use the original task revision, test patch, and setup script."
    assert args[3] == "sha256:" + "b" * 64, "Pin controls to the model container's immutable image."
    assert not syntax_probe_manifest(args[1])["files"], "A control must not change any Python model or test source."
    assert "psrl_baseline_" in args[1] and "new file mode" in args[1], "Add only an inert control artifact."
    assert args[5] == "episode-1__baseline", "Control containers need cleanup labels separate from model evaluation."
    assert kwargs["env"] == {"OMP_NUM_THREADS": "1"}, "Keep evaluation environment variables identical."
    assert kwargs["run_args"] == run_args, "Baseline containers must retain resource limits and loop ownership."
    assert result["baseline_image_id"] == args[3], "Retain the verified image identity with the baseline report."


def test_concurrent_episodes_share_one_baseline_control(control):
    evaluate, evaluator, _ = control
    report = evaluator.return_value

    def run(*args, **kwargs):
        time.sleep(0.03)
        return report

    evaluator.side_effect = run
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda episode: evaluate(swe_task_id=f"episode-{episode}"), range(16)))

    assert evaluator.call_count == 1, "Concurrent rollouts must not multiply expensive baseline evaluations."
    assert all(row["test_results_observed"] == 2 for row in results), "All waiters must receive the verified report."


@pytest.mark.parametrize(
    "change",
    [
        {"image_id": "sha256:" + "c" * 64},
        {"env": {"OMP_NUM_THREADS": "2"}},
        {"timeout": 60},
        {"run_args": ["--cpus=4", "--network", "host"]},
    ],
)
def test_baseline_cache_never_crosses_evaluation_configuration_boundaries(control, change):
    evaluate, evaluator, _ = control
    evaluate()
    evaluate(**change)

    assert evaluator.call_count == 2, "A different image or evaluation configuration needs an independent control."


def test_baseline_cache_never_crosses_task_revision_or_test_setup_boundaries(control):
    evaluate, evaluator, task = control
    evaluate()
    evaluate(swe_problem={**task, "base_commit": "d" * 40})
    evaluate(swe_problem={**task, "eval_script": "pytest -rA different_tests"})

    assert evaluator.call_count == 3, "Task revision and setup changes must invalidate baseline evidence."


@pytest.mark.parametrize("failure", ["empty", "partial", "attributed", "execution"])
def test_failed_baselines_remain_execution_errors_for_the_actor_lifetime(control, failure):
    evaluate, evaluator, _ = control
    if failure == "empty":
        evaluator.return_value = {"resolved_by": "harness", "test_results_observed": 0}
    elif failure == "partial":
        evaluator.return_value = {"resolved_by": "harness", "test_results_observed": 1}
    elif failure == "attributed":
        evaluator.return_value = {"test_results_observed": 2, "model_failure": {"error_type": "NameError"}}
    else:
        evaluator.side_effect = swebench_grader.SWEGradingError("Baseline transport timed out.")
    for episode in ["first", "second"]:
        with pytest.raises(swebench_grader.SWEGradingError, match="Baseline control failed"):
            evaluate(swe_task_id=episode)

    assert evaluator.call_count == 1, "A known unready task must fail closed without a control stampede."
