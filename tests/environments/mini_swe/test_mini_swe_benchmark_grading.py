"""
Verify SWE-Gym reports, patch failures, and environment failures remain distinct.
"""

import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from examples.mini_swe import swebench_grader
from examples.mini_swe.benchmark_grading import (
    TEST_OUTPUT_END,
    TEST_OUTPUT_START,
    configure_gym_test_runtime,
    frame_eval_script,
    parse_benchmark_report,
)
from examples.mini_swe.task_setup import SWETaskSetupError

pytestmark = pytest.mark.cpu_test

PATCH = "diff --git a/value.py b/value.py\n--- a/value.py\n+++ b/value.py\n@@ -1 +1 @@\n-old\n+new\n"


@pytest.mark.parametrize(
    "repo,test_path",
    [
        ("iterative/dvc", "tests/func/test_data_cloud.py"),
        ("pandas-dev/pandas", "pandas/tests/io/xml/test_xml.py"),
    ],
)
def test_gym_fixture_runtime_preserves_pytest_arguments(tmp_path, repo, test_path):
    (tmp_path / "pytest.py").write_text(
        "import json, socketserver, urllib.request\n"
        "def main(args):\n"
        "    print(json.dumps({'args': args, 'backlog': socketserver.TCPServer.request_queue_size,\n"
        "                      'headers': urllib.request._opener.addheaders if urllib.request._opener else []}))\n"
        "    return 0\n"
    )
    script = configure_gym_test_runtime(f"pytest -rA {test_path} -k 'a space'\n", repo)
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        timeout=10,
    )
    assert result.returncode == 0, f"Fixture setup must start pytest successfully: {result.stderr!r}."
    observed = json.loads(result.stdout)
    assert observed["args"] == ["-rA", test_path, "-k", "a space"], (
        "Fixture setup must preserve official pytest arguments."
    )
    if repo == "iterative/dvc":
        assert observed["backlog"] >= 1024, "The WebDAV fixture must admit the host-sized DVC client pool."
    else:
        assert ["User-Agent", "Mozilla/5.0"] in observed["headers"], "The XML fixture needs an accepted user agent."


def test_gym_fixture_runtime_leaves_unrelated_tests_unchanged():
    script = "pytest -rA tests/unit/test_regular.py\n"
    assert configure_gym_test_runtime(script, "iterative/dvc") == script, (
        "Unrelated tests must retain their interpreter."
    )


def test_eval_framing_preserves_activation_shell_options(
    tmp_path,
    monkeypatch,
):
    """
    Allow Conda activation hooks to read optional unset environment variables.
    """
    monkeypatch.delenv("PSRL_TEST_UNSET_CONDA_FLAG", raising=False)
    activation = tmp_path / "activate.sh"
    activation.write_text('previous_value="$PSRL_TEST_UNSET_CONDA_FLAG"\n')
    script = f"set -xo pipefail\nsource {shlex.quote(str(activation))}\nprintf 'environment_ready\\n'\n"

    result = subprocess.run(
        ["bash"],
        input=frame_eval_script(script),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )

    assert result.returncode == 0, "Output framing must preserve the activation script's shell behavior."
    assert "environment_ready" in result.stdout, "Environment activation must complete."
    assert TEST_OUTPUT_START in result.stdout and TEST_OUTPUT_END in result.stdout, (
        "Completed scripts must emit both output markers."
    )


def test_eval_framing_does_not_require_shell_tracing():
    """
    Emit completion markers even when the evaluation script disables command tracing.
    """
    result = subprocess.run(
        ["bash"],
        input=frame_eval_script("set +x\nprintf 'test_output\\n'\n"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )

    assert result.returncode == 0, "A completed script must exit successfully."
    assert TEST_OUTPUT_START in result.stdout and TEST_OUTPUT_END in result.stdout, (
        "Output markers must be independent of shell tracing."
    )


@pytest.mark.parametrize("script", ['set -u\nprintf "%s" "$PSRL_TEST_UNSET_CONDA_FLAG"', "exit 1"])
def test_eval_framing_does_not_mark_interrupted_scripts_complete(
    script,
    monkeypatch,
):
    """
    Preserve explicit script failures without manufacturing a complete report.
    """
    monkeypatch.delenv("PSRL_TEST_UNSET_CONDA_FLAG", raising=False)
    result = subprocess.run(
        ["bash"],
        input=frame_eval_script(script),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )

    assert result.returncode != 0, "Evaluation failures must retain their nonzero exit status."
    assert TEST_OUTPUT_END not in result.stdout, "An interrupted script must not produce a completion marker."


@pytest.fixture
def task():
    return {
        "instance_id": "example__repo-1",
        "repo": "example/repo",
        "base_commit": "abc1234",
        "image_name": "example/image",
        "problem_statement": "Repair the issue.",
        "eval_script": "pytest -rA tests",
        "FAIL_TO_PASS": ["tests/test_fix.py::test_one", "tests/test_fix.py::test_two"],
        "PASS_TO_PASS": ["tests/test_keep.py::test_keep"],
    }


def report(second="PASSED", keep="PASSED"):
    return (
        f"{TEST_OUTPUT_START}\n"
        "PASSED tests/test_fix.py::test_one\n"
        f"{second} tests/test_fix.py::test_two\n"
        f"{keep} tests/test_keep.py::test_keep\n"
        f"{TEST_OUTPUT_END}\n"
    )


@pytest.mark.parametrize(
    "second,keep,resolved,f2p,p2p",
    [("PASSED", "PASSED", True, 2, 1), ("FAILED", "PASSED", False, 1, 1), ("PASSED", "FAILED", False, 2, 0)],
)
def test_gym_uses_official_resolution_rules(task, second, keep, resolved, f2p, p2p):
    result = parse_benchmark_report(task, report(second, keep), "gym")
    assert result["resolved"] is resolved, "Resolution requires all F2P and P2P tests to pass."
    assert result["f2p_pass"] == f2p, "Preserve official F2P counts."
    assert result["p2p_pass"] == p2p, "Preserve official P2P counts."
    assert result["test_results_observed"] == 3, "Count observed task results with the selected official parser."


@pytest.mark.parametrize("ansi", [False, True])
@pytest.mark.parametrize("failure_status,resolved", [("FAILED", False), ("PASSED", True)])
def test_gym_parses_pytest_sugar_suffix_results_and_counts_task_observations(ansi, failure_status, resolved):
    """
    Parse pytest sugar status suffixes independently of colored output and aggregate summaries.
    """
    task = {
        "instance_id": "pydantic__pydantic-8072",
        "FAIL_TO_PASS": ["tests/test_edge_cases.py::test_no_name_conflict_in_constructor"],
        "PASS_TO_PASS": [
            "tests/test_edge_cases.py::test_hashable_required",
            "tests/test_edge_cases.py::test_custom_exception_handler",
        ],
    }
    statuses = [
        f"tests/test_edge_cases.py::test_no_name_conflict_in_constructor {failure_status}",
        "tests/test_edge_cases.py::test_hashable_required PASSED",
        "tests/test_edge_cases.py::test_custom_exception_handler PASSED",
        "tests/test_edge_cases.py::test_unrelated PASSED",
    ]
    if ansi:
        statuses = [f"\x1b[32m{line}\x1b[0m" for line in statuses]
    text = "\n".join(
        [
            TEST_OUTPUT_START,
            *statuses,
            "                              Summary of Failures",
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓",
            "┃ File                      ┃ Function                           ┃",
            "│ tests/test_edge_cases.py  │ test_no_name_conflict_in_constructor│",
            "└────────────────────────────────────────────────────────────────┘",
            "Results (1.01s):",
            "         1 failed",
            "       160 passed",
            "        10 skipped",
            TEST_OUTPUT_END,
        ]
    )

    result = parse_benchmark_report(task, text, "gym")

    assert result["resolved"] is resolved, "Official task test outcomes must determine resolution."
    assert result["f2p_pass"] == int(resolved), "Preserve the observed suffix status for F2P."
    assert result["p2p_pass"] == 2, "Preserve passing P2P suffix results."
    assert result["test_results_observed"] == 3, "Ignore unrelated tests and aggregate summary counts."
    assert "model_failure" not in result, "A parsed test report must not require startup failure attribution."


def test_gym_counts_mixed_prefix_and_suffix_results_with_one_official_parser(task):
    """
    Preserve mixed pytest formats when calculating baseline readiness evidence.
    """
    text = "\n".join(
        [
            TEST_OUTPUT_START,
            "PASSED tests/test_fix.py::test_one",
            "tests/test_fix.py::test_two FAILED",
            "\x1b[32mtests/test_keep.py::test_keep PASSED\x1b[0m",
            "tests/test_other.py::test_unrelated PASSED",
            TEST_OUTPUT_END,
        ]
    )

    result = parse_benchmark_report(task, text, "gym")

    assert not result["resolved"], "A failed F2P test remains unresolved."
    assert result["f2p_pass"] == result["p2p_pass"] == 1, "Parse both status positions consistently."
    assert result["test_results_observed"] == 3, "Baseline observations must use the same parsed task results."


@pytest.mark.parametrize(
    "output",
    ["all tests passed", "", f"{TEST_OUTPUT_START}\nno report\n{TEST_OUTPUT_END}", f"{TEST_OUTPUT_START}\nincomplete"],
)
def test_gym_rejects_missing_or_unparseable_reports(task, output):
    with pytest.raises(ValueError, match="markers|results|Incomplete"):
        parse_benchmark_report(task, output, "gym")


def test_complete_collection_failure_remains_a_model_outcome(task):
    output = f"{TEST_OUTPUT_START}\nERROR collecting tests/test_fix.py\n{TEST_OUTPUT_END}"
    result = parse_benchmark_report(task, output, "gym")
    assert not result["resolved"], "A complete collection failure is unresolved."
    assert result["f2p_pass"] == result["p2p_pass"] == 0, "Missing tests remain failed tests."


def test_verified_uses_its_repository_parser(monkeypatch, task):
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    from swebench.harness.test_spec import test_spec

    spec = object()
    parser = Mock(return_value={test: "PASSED" for test in task["FAIL_TO_PASS"] + task["PASS_TO_PASS"]})
    monkeypatch.setattr(test_spec, "make_test_spec", Mock(return_value=spec))
    monkeypatch.setitem(MAP_REPO_TO_PARSER, task["repo"], parser)
    result = parse_benchmark_report(task, report(), "verified")
    assert result["resolved"], "The repository parser must supply the verification results."
    assert parser.call_args.args[1] is spec, "Pass the official test specification to the repository parser."


@pytest.fixture
def backend(monkeypatch):
    from minisweagent.environments import docker

    state = {"phase": None, "output": report(), "commands": [], "staged": {}}
    environment = Mock(container_id="gym-grader")

    def execute(action, **kwargs):
        command = action["command"]
        state["commands"].append(command)
        phase = "test" if command.startswith("bash ") else "apply" if command.startswith("git apply") else "setup"
        if state["phase"] == phase:
            return {"returncode": 0, "output": "", "exception_info": f"{phase} transport failed"}
        if phase == "apply" and state["phase"] == "unappliable":
            return {"returncode": 1, "output": "rejected patch", "exception_info": ""}
        return {"returncode": 0, "output": state["output"] if phase == "test" else "abc1234", "exception_info": ""}

    environment.execute.side_effect = execute
    monkeypatch.setattr(
        swebench_grader,
        "_stage_grading_inputs",
        lambda env, patch, script, *, timeout: state["staged"].update(model_patch=patch, eval_script=script),
    )
    monkeypatch.setattr(docker, "DockerEnvironment", Mock(return_value=environment))
    monkeypatch.setattr(swebench_grader, "force_remove_containers_by_label", Mock())
    return state, environment


@pytest.mark.parametrize("phase", ["setup", "apply", "test"])
def test_gym_execution_errors_propagate(task, backend, phase):
    state, _ = backend
    state["phase"] = phase
    with pytest.raises((SWETaskSetupError, swebench_grader.SWEGradingError)):
        swebench_grader.grade_fresh_container(task, PATCH, "gym", task["image_name"])


def test_gym_invalid_report_propagates(task, backend):
    state, _ = backend
    state["output"] = "all passed"
    with pytest.raises(swebench_grader.SWEGradingError, match="report") as error:
        swebench_grader.grade_fresh_container(task, PATCH, "gym", task["image_name"])
    assert "all passed" in str(error.value), "Invalid reports must preserve output diagnostics."


def test_gym_grader_attributes_model_syntax_failure_to_the_applied_patch(task, backend, monkeypatch):
    """
    Require task baseline evidence before attributing a startup failure without compile proof.
    """
    state, _ = backend
    monkeypatch.setattr(swebench_grader, "_container_image_id", Mock(return_value="sha256:" + "a" * 64))
    monkeypatch.setattr(
        swebench_grader,
        "_get_baseline_report",
        Mock(
            return_value={
                "resolved_by": "harness",
                "test_results_observed": 3,
                "baseline_image_id": "sha256:" + "a" * 64,
            }
        ),
    )
    state["output"] = (
        f"{TEST_OUTPUT_START}\n"
        "ImportError while loading conftest '/testbed/tests/conftest.py'.\n"
        '  File "/testbed/value.py", line 1\n'
        "    )\n"
        "SyntaxError: unmatched ')'\n"
        f"{TEST_OUTPUT_END}\n"
    )

    result = swebench_grader.grade_fresh_container(task, PATCH.replace("+new", "+)"), "gym", task["image_name"])

    assert not result["resolved"], "A patch that breaks test imports cannot resolve the task."
    assert result["f2p_pass"] == result["p2p_pass"] == 0, "No tests passed before collection failed."
    assert result["model_failure"]["path"] == "value.py", "Retain the responsible source file."


def test_gym_unappliable_patch_skips_tests(task, backend):
    state, environment = backend
    state["phase"] = "unappliable"
    result = swebench_grader.grade_fresh_container(task, PATCH, "gym", task["image_name"])
    assert not result["apply_ok"], "An invalid repair remains a model failure."
    assert not any(command.startswith("bash ") for command in state["commands"]), "Rejected patches must skip tests."
    environment.cleanup.assert_called_once()


def test_gym_selects_base_commit_and_frames_report(task, backend):
    state, _ = backend
    result = swebench_grader.grade_fresh_container(task, PATCH, "gym", task["image_name"])
    assert result["resolved"], "Valid passing reports must resolve the task."
    assert "git checkout --detach abc1234" in state["commands"], "Use the task's exact base revision."
    assert not any("HEAD~1" in command for command in state["commands"]), "Use the benchmark revision directly."
    script = state["staged"]["eval_script"]
    assert TEST_OUTPUT_START in script and TEST_OUTPUT_END in script, "Evaluation must produce a complete report."


def test_large_grading_inputs_are_copied_without_expanding_process_arguments(monkeypatch):
    """
    Transfer large Unicode patches and scripts beyond the Linux per argument limit.
    """
    patch = "# patch data \u03bb\n" * 30000
    script = "# script data \u03bb\n" * 30000
    environment = SimpleNamespace(container_id="grader", config=SimpleNamespace(executable="docker"))
    copied = []

    def copy(command, **kwargs):
        assert command[:2] == ["docker", "cp"], "Use Docker's file transfer operation."
        assert sum(len(arg.encode()) for arg in command) < 4096, "Process arguments must not contain file contents."
        root = Path(command[2])
        assert (root / "model.patch").read_text() == patch + "\n", "Preserve every patch byte."
        assert (root / "eval.sh").read_text() == script + "\n", "Preserve every evaluation script byte."
        copied.append(root)
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(swebench_grader.subprocess, "run", copy)
    swebench_grader._stage_grading_inputs(environment, patch, script)

    assert copied and not copied[0].exists(), "Temporary host artifacts must be removed after transfer."


def test_failed_grading_input_transfer_propagates(monkeypatch):
    """
    Keep file transport errors outside the model reward path.
    """
    environment = SimpleNamespace(container_id="grader", config=SimpleNamespace(executable="docker"))
    monkeypatch.setattr(
        swebench_grader.subprocess,
        "run",
        Mock(side_effect=subprocess.CalledProcessError(1, ["docker", "cp"], output="copy failed")),
    )
    with pytest.raises(swebench_grader.SWEGradingError, match="input transfer failed"):
        swebench_grader._stage_grading_inputs(environment, PATCH, "pytest")
