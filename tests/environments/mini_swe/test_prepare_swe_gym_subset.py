"""
Verify subset adaptation preserves task identity and strict grading failure boundaries.
"""

import subprocess

import pandas as pd
import pytest
from examples.mini_swe.benchmark_grading import TEST_OUTPUT_END, frame_eval_script, parse_benchmark_report
from examples.mini_swe.prepare.prepare_swe_gym_subset import prepare_eval_script, prepare_subset

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("exit_code", [0, 1, 2])
def test_pytest_failures_preserve_reports_and_postlude_environment(exit_code):
    script = (
        "set -e\n"
        "export TEST_ACTIVATED_ENV=task_environment\n"
        f"pytest() {{ printf 'FAILED tests/test_fix.py::test_one\\n'; return {exit_code}; }}\n"
        "pytest -rA tests/test_fix.py\n"
    )
    result = subprocess.run(
        ["bash"],
        input=frame_eval_script(prepare_eval_script(script), postlude='printf "%s\\n" "$TEST_ACTIVATED_ENV"'),
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, "Test failures must reach report parsing."
    assert TEST_OUTPUT_END in result.stdout, "A completed test command must emit the report boundary."
    assert "task_environment" in result.stdout, "The postlude must use the task's activated environment."
    report = parse_benchmark_report(
        {"instance_id": "example__repo-1", "FAIL_TO_PASS": ["tests/test_fix.py::test_one"], "PASS_TO_PASS": []},
        result.stdout,
        "gym",
    )
    assert not report["resolved"], "A failed official test must remain unresolved."


def test_setup_failures_do_not_become_completed_reports():
    script = prepare_eval_script("set -e\nfalse\npytest -rA tests/test_fix.py\n")
    result = subprocess.run(["bash"], input=frame_eval_script(script), text=True, capture_output=True, timeout=10)
    assert result.returncode != 0, "Setup failures must retain their unsuccessful exit status."
    assert TEST_OUTPUT_END not in result.stdout, "Setup failure must not manufacture a completed report."


def test_missing_pytest_cannot_become_a_model_outcome():
    script = prepare_eval_script("set -e\npytest() { return 127; }\npytest -rA tests/test_fix.py\n")
    result = subprocess.run(["bash"], input=frame_eval_script(script), text=True, capture_output=True, timeout=10)
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(
            {"instance_id": "example__repo-1", "FAIL_TO_PASS": ["tests/test_fix.py::test_one"], "PASS_TO_PASS": []},
            result.stdout,
            "gym",
        )


@pytest.mark.parametrize("command", ["true", "pytest tests | tee log", "pytest tests && echo done", "pytest tests \\"])
def test_unsupported_test_commands_fail_closed(command):
    with pytest.raises(ValueError, match="pytest"):
        prepare_eval_script(command)


def test_preparation_preserves_the_complete_source_and_original_script(tmp_path):
    tasks = [
        {
            "instance_id": f"example__repo-{index}",
            "repo": "example/repo",
            "base_commit": "abc1234",
            "problem_statement": "Repair the bug.",
            "FAIL_TO_PASS": ["tests/test_fix.py::test_one"],
            "PASS_TO_PASS": [],
            "eval_script": "set -e\npytest -rA tests/test_fix.py\n",
            "patch": "upstream repair",
            "test_patch": "upstream test patch",
        }
        for index in range(100)
    ]
    manifest = prepare_subset(tmp_path, tasks, {"revision": "test"})
    rows = pd.read_parquet(tmp_path / "train.parquet").to_dict("records")
    assert len(rows) == manifest["rows"] == 100, "Retain every official training task."
    for row, task in zip(rows, tasks, strict=True):
        prepared = row["extra_info"]["swe_problem"]
        assert prepared["instance_id"] == task["instance_id"], "Preserve source task ordering and IDs."
        assert prepared["upstream_eval_script"] == task["eval_script"], "Retain the unmodified upstream script."
        assert prepared["patch"] == task["patch"], "Preserve the task's gold repair."
        assert "pytest -rA tests/test_fix.py || true" in prepared["eval_script"], (
            "Only the test command may tolerate a nonzero exit."
        )
