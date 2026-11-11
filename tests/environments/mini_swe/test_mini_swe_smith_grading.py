"""
Verify SWE-smith output framing and official report based rewards.
"""

import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from examples.mini_swe import runner, swebench_grader
from examples.mini_swe.reward import compute_score

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def harness():
    return pytest.importorskip("swesmith.harness.grading")


@pytest.fixture
def problem():
    return {
        "instance_id": "oauthlib__oauthlib.1fd52536.test_task",
        "repo": "oauthlib__oauthlib.1fd52536",
        "FAIL_TO_PASS": ["tests/test_fix.py::test_first", "tests/test_fix.py::test_second"],
        "PASS_TO_PASS": ["tests/test_existing.py::test_first", "tests/test_existing.py::test_second"],
    }


@pytest.mark.parametrize(
    "outcome,expected_score,expected_f2p,expected_p2p",
    [("resolved", 1.0, 20, 653), ("partial", 0.4, 1, 2), ("regression", -0.3, 2, 1)],
)
def test_generated_script_reaches_official_harness_and_reward(
    harness, problem, monkeypatch, tmp_path, outcome, expected_score, expected_f2p, expected_p2p
):
    if outcome == "resolved":
        problem["FAIL_TO_PASS"] = [f"tests/test_fix.py::test_{index}" for index in range(20)]
        problem["PASS_TO_PASS"] = [f"tests/test_existing.py::test_{index}" for index in range(653)]
    lines = []
    for key, passed in [("FAIL_TO_PASS", expected_f2p), ("PASS_TO_PASS", expected_p2p)]:
        lines.extend(
            f"{test_id} {'PASSED' if index < passed else 'FAILED'}" for index, test_id in enumerate(problem[key])
        )
    command = "printf '%s\\n' " + " ".join(map(shlex.quote, lines))
    if outcome != "resolved":
        command += "\n(exit 1)"
    profile = harness.registry.get_from_inst(problem)
    get_test_cmd = Mock(return_value=(command, []))
    monkeypatch.setattr(profile, "get_test_cmd", get_test_cmd)
    monkeypatch.setattr(harness.registry, "get_from_inst", lambda problem: profile)
    script = swebench_grader._get_smith_eval_script(problem)
    get_test_cmd.assert_called_once_with(problem, f2p_only=False)
    script = script.replace("cd /testbed", f"cd {shlex.quote(str(tmp_path))}")
    executed = subprocess.run(
        ["bash"], input=script, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    log = tmp_path / "output.log"
    log.write_text(executed.stdout)
    output, found = harness.read_test_output(str(log))
    assert found and lines[-1] in output, "The harness must find the complete test output, including failing tests."

    grade = swebench_grader._grade_smith(problem, "patch", executed.stdout, True, executed.returncode)
    assert grade["resolved_by"] == "harness", "Only the official report may determine the result."
    assert grade["f2p_pass"] == expected_f2p, "The official F2P count must reach the reward function."
    assert grade["p2p_pass"] == expected_p2p, "The official P2P count must reach the reward function."
    reward = compute_score(
        "swe_smith_py",
        "",
        {},
        extra_info={"patch": "patch", "num_turns": 2, "grader_result": {"apply_ok": True, **grade}},
        reward_mode="partial_credit",
    )
    assert reward["score"] == pytest.approx(expected_score), (
        "Rewards must distinguish repair, progress, and regression."
    )


@pytest.mark.parametrize("returncode", [0, 1])
@pytest.mark.parametrize("markers", ["none", "start", "end", "reversed"])
def test_incomplete_output_cannot_become_a_test_failure_or_exit_code_reward(harness, problem, markers, returncode):
    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START

    output = "tests/test_fix.py::test_first PASSED\n"
    if markers in {"start", "reversed"}:
        output = f"+ : '{TEST_OUTPUT_START}'\n" + output
    if markers == "end":
        output += f"+ : '{TEST_OUTPUT_END}'\n"
    if markers == "reversed":
        output = f"+ : '{TEST_OUTPUT_END}'\n" + output
    with pytest.raises(swebench_grader.SWEGradingError, match="SWE-smith harness grading failed"):
        swebench_grader._grade_smith(problem, "patch", output, True, returncode)


@pytest.mark.parametrize("report", [{"resolved": False}, {"resolved": True, "tests_status": {}}, None])
def test_invalid_harness_report_is_rejected_and_temporary_log_is_removed(harness, problem, monkeypatch, report):
    paths = []

    def get_report(prediction, instance, path):
        paths.append(Path(path))
        assert paths[-1].read_text() == "output", "The harness must read the flushed output log."
        return report

    monkeypatch.setattr(harness, "get_eval_report", get_report)
    with pytest.raises(swebench_grader.SWEGradingError):
        swebench_grader._grade_smith(problem, "patch", "output", True, 0)
    assert paths and not paths[0].exists(), "Failed grading must remove its temporary log."


def test_parser_exception_cannot_fall_back_to_success(harness, problem, monkeypatch):
    monkeypatch.setattr(harness, "get_eval_report", Mock(side_effect=RuntimeError("Parser failed.")))
    with pytest.raises(swebench_grader.SWEGradingError, match="Parser failed"):
        swebench_grader._grade_smith(problem, "patch", "output", True, 0)


def test_unapplied_patch_retains_apply_failure_result(harness, problem):
    grade = swebench_grader._grade_smith(problem, "patch", "", False, 0)
    assert not grade["resolved"], "An unapplied patch cannot resolve a task."
    assert grade["resolved_by"] == "apply_failed", (
        "Patch application failure must remain distinct from harness failure."
    )


def test_runner_returns_submission_without_grading(problem, monkeypatch):
    from minisweagent import environments, models
    from minisweagent.agents import default
    from psrl.utils.agent import overflow

    monkeypatch.setattr(
        swebench_grader, "grade_fresh_container", Mock(side_effect=swebench_grader.SWEGradingError("Missing report."))
    )
    payload = {
        "observation": {
            "swe_problem": problem,
            "swe_problem_image": "example/swesmith:latest",
            "swe_grader": "swebench_fresh_container",
            "swe_grader_kind": "smith",
            "swe_restore_tests": True,
        },
        "runtime_config": {
            "sandbox_config": {"environment": {"grader_timeout": 1800}, "rollout_turn_timeout": 60},
            "model": {},
            "agent": {"problem_template": "task"},
        },
        "sampling_params": {},
        "base_url": "http://gateway",
        "model": "test-model",
        "max_turns": 2,
        "task": "Fix the issue.",
    }
    monkeypatch.setattr(runner, "_silence_litellm", lambda: None)
    monkeypatch.setattr(runner, "initialize_smith_task", Mock())
    monkeypatch.setattr(overflow, "ensure_overflow_handling", Mock())
    monkeypatch.setattr(environments, "get_environment", Mock(return_value=Mock()))
    model = SimpleNamespace(abort_exceptions=[], _query=Mock())
    monkeypatch.setattr(models, "get_model", Mock(return_value=model))

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, task):
            return {"exit_status": "submitted", "submission": "patch"}

    monkeypatch.setattr(default, "DefaultAgent", Agent)
    result = runner.run_agent(payload)
    assert result["exit_status"] == "submitted", "The rollout must finish without executing the grader."
    assert result["submission"] == "patch", "The reward service must receive the submitted patch."
    assert "grader_result" not in result, "Runner outputs must contain no grading result."
    swebench_grader.grade_fresh_container.assert_not_called()
