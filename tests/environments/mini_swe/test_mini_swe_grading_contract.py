"""
Verify SWE-smith grading separates execution errors from model outcomes.
"""

import json
import shlex
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from examples.mini_swe import swebench_grader
from examples.mini_swe.reward import compute_score
from examples.mini_swe.task_setup import SWETaskSetupError

pytestmark = pytest.mark.cpu_test

PATCH = "diff --git a/value.txt b/value.txt\n--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-1\n+0\n"


@pytest.fixture
def task():
    return {
        "instance_id": "oauthlib__oauthlib.1fd52536.contract",
        "repo": "swesmith/oauthlib__oauthlib.1fd52536",
        "image_name": "example/image",
        "problem_statement": "Fix the value.",
        "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_keep.py::test_keep"],
    }


def _output():
    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START

    return (
        f"+ : '{TEST_OUTPUT_START}'\n"
        "tests/test_fix.py::test_fix PASSED\n"
        "tests/test_keep.py::test_keep PASSED\n"
        f"+ : '{TEST_OUTPUT_END}'\n"
    )


@pytest.fixture
def backend(monkeypatch, task):
    from minisweagent.environments import docker

    environment = Mock(container_id="test-grader")
    state = SimpleNamespace(failure=None, commands=[], returncode=0, output=_output())

    def execute(action, **kwargs):
        command = action["command"]
        state.commands.append(command)
        phase = (
            "apply"
            if command.startswith("git apply")
            else "restore"
            if command.startswith("git checkout --")
            else "test"
        )
        if state.failure == f"{phase}_raise":
            raise RuntimeError(f"{phase} transport failed.")
        if state.failure == f"{phase}_reported":
            return {"returncode": 0, "output": state.output, "exception_info": f"{phase} transport failed."}
        if phase == "apply" and state.failure == "unappliable":
            return {"returncode": 1, "output": "Patch does not apply.", "exception_info": ""}
        return {"returncode": state.returncode if phase == "test" else 0, "output": state.output, "exception_info": ""}

    environment.execute.side_effect = execute
    constructor = Mock(return_value=environment)
    monkeypatch.setattr(docker, "DockerEnvironment", constructor)
    monkeypatch.setattr(swebench_grader, "initialize_smith_task", Mock())
    monkeypatch.setattr(swebench_grader, "initialize_benchmark_task", Mock())
    monkeypatch.setattr(swebench_grader, "_stage_grading_inputs", Mock())
    monkeypatch.setattr(swebench_grader, "smith_evaluation_test_files", Mock(return_value=["tests/test_fix.py"]))
    builder = Mock(return_value="pytest")
    monkeypatch.setattr(swebench_grader, "_get_smith_eval_script", builder)
    monkeypatch.setattr(swebench_grader, "_get_gym_eval_script", builder)
    sweep = Mock()
    monkeypatch.setattr(swebench_grader, "force_remove_containers_by_label", sweep)

    def grade(patch=PATCH, grader_kind="smith"):
        return swebench_grader.grade_fresh_container(task, patch, grader_kind, task["image_name"])

    return SimpleNamespace(
        state=state,
        environment=environment,
        constructor=constructor,
        builder=builder,
        sweep=sweep,
        grade=grade,
        errors=(swebench_grader.SWEGradingError, SWETaskSetupError),
    )


@pytest.mark.parametrize(
    "failure",
    [
        "build",
        "start",
        "apply_raise",
        "apply_reported",
        "restore_raise",
        "restore_reported",
        "test_raise",
        "test_reported",
        "cleanup",
        "sweep",
        "report",
    ],
)
def test_execution_errors_never_produce_a_task_result(backend, failure):
    backend.state.failure = failure
    if failure == "build":
        backend.builder.side_effect = RuntimeError("Profile failed.")
    elif failure == "start":
        backend.constructor.side_effect = RuntimeError("Docker unavailable.")
    elif failure == "cleanup":
        backend.environment.cleanup.side_effect = RuntimeError("Cleanup failed.")
    elif failure == "sweep":
        backend.sweep.side_effect = RuntimeError("Sweep failed.")
    elif failure == "report":
        backend.state.output = "No complete report."
    with pytest.raises(backend.errors):
        backend.grade()


@pytest.mark.parametrize("failure", ["report", "test_reported"])
def test_grading_errors_preserve_exact_inputs_and_partial_output(backend, task, monkeypatch, tmp_path, failure):
    monkeypatch.setattr(swebench_grader.tempfile, "tempdir", str(tmp_path))
    backend.state.failure = failure
    backend.state.output = "Partial evaluation output.\n"
    with pytest.raises(swebench_grader.SWEGradingError) as caught:
        backend.grade()
    directories = list(tmp_path.glob("psrl-swe-failure-*"))
    assert len(directories) == 1, "One failed evaluation must produce one diagnostic bundle."
    directory = directories[0]
    assert json.loads((directory / "task.json").read_text()) == task, "The exact task must survive cleanup."
    assert (directory / "model.patch").read_text() == PATCH, "The exact submitted patch must survive cleanup."
    assert (directory / "eval.sh").read_text() == "pytest", "The executed script must survive cleanup."
    assert (directory / "eval.log").read_text() == backend.state.output, "Partial failure output must be retained."
    assert str(directory) in caught.value.__notes__[0], "The raised error must identify its diagnostic bundle."
    backend.environment.cleanup.assert_called_once()
    backend.sweep.assert_called_once()


def test_diagnostic_write_failure_preserves_the_grading_error(backend, monkeypatch):
    backend.state.failure = "test_reported"
    monkeypatch.setattr(swebench_grader.tempfile, "mkdtemp", Mock(side_effect=OSError("Disk full.")))
    with pytest.raises(swebench_grader.SWEGradingError, match="test transport failed"):
        backend.grade()
    backend.environment.cleanup.assert_called_once()


def test_unappliable_patch_is_a_model_failure_and_skips_tests(backend):
    backend.state.failure = "unappliable"
    result = backend.grade()
    assert not result["apply_ok"], "A rejected diff must remain a model application failure."
    assert not any(command.startswith("bash ") for command in backend.state.commands), (
        "An unappliable patch must not start tests."
    )
    backend.environment.cleanup.assert_called_once()
    score = compute_score(
        "swe_smith_py",
        "",
        {},
        extra_info={"patch": PATCH, "num_turns": 3, "grader_result": result},
        reward_mode="partial_credit",
    )
    assert score["score"] == -0.2, "Only a normal patch application failure may receive the application penalty."


def test_grading_container_outlives_the_official_test_budget(backend):
    backend.grade()
    assert backend.constructor.call_args.kwargs["container_timeout"] == "2h", (
        "The grader must retain setup and cleanup time beyond the 900 second test budget."
    )
    calls = [
        call for call in backend.environment.execute.call_args_list if call.args[0]["command"].startswith("bash ")
    ]
    assert len(calls) == 1 and calls[0].kwargs["timeout"] == 900, (
        "The actual harness command must receive the default test execution budget."
    )
    assert backend.constructor.call_args.kwargs["timeout"] == 120, (
        "Preparation commands need an independent deadline from test execution."
    )


@pytest.mark.parametrize("stage", ["start", "staging", "apply"])
def test_preparation_timeouts_are_retryable_only_after_cleanup(
    backend,
    monkeypatch,
    stage,
):
    timeout = subprocess.TimeoutExpired("grading preparation", 120)
    if stage == "start":
        backend.constructor.side_effect = timeout
    elif stage == "staging":
        monkeypatch.setattr(swebench_grader, "_stage_grading_inputs", Mock(side_effect=timeout))
    else:
        backend.environment.execute.return_value = {}
        backend.environment.execute.side_effect = lambda *args, **kwargs: {
            "returncode": -1,
            "exception_info": "Docker execution timed out.",
            "extra": {"exception_type": "TimeoutExpired"},
        }
    with pytest.raises(swebench_grader.SWEGradingPreparationTimeout):
        backend.grade()
    backend.sweep.assert_called_once()
    if stage != "start":
        backend.environment.cleanup.assert_called_once()


@pytest.mark.parametrize("source", ["reported_subprocess", "reported_builtin", "raised_subprocess", "raised_builtin"])
@pytest.mark.parametrize("grader_kind", ["smith", "gym"])
def test_test_execution_timeouts_raise_without_retrying(
    backend,
    monkeypatch,
    source,
    grader_kind,
):
    execute = backend.environment.execute.side_effect

    def timeout_test(action, **kwargs):
        if action["command"].startswith("bash "):
            if source == "raised_subprocess":
                raise subprocess.TimeoutExpired(
                    "test execution", 900, output=b"Partial output.\n", stderr=b"Timeout.\n"
                )
            if source == "raised_builtin":
                raise TimeoutError("Tests exceeded the execution budget.")
            return {
                "returncode": -1,
                "output": "Partial output.\n",
                "exception_info": "Tests timed out.",
                "extra": {"exception_type": "TimeoutExpired" if source == "reported_subprocess" else "TimeoutError"},
            }
        return execute(action, **kwargs)

    backend.environment.execute.side_effect = timeout_test
    save = Mock()
    monkeypatch.setattr(swebench_grader, "_save_failure_artifacts", save)
    with pytest.raises(swebench_grader.SWEGradingError, match="SWE test execution"):
        backend.grade(grader_kind=grader_kind)
    test_calls = [
        call for call in backend.environment.execute.call_args_list if call.args[0]["command"].startswith("bash ")
    ]
    assert len(test_calls) == 1, "A test execution timeout must not extend its budget through retries."
    backend.environment.cleanup.assert_called_once()
    backend.sweep.assert_called_once()
    if source == "raised_subprocess":
        assert save.call_args.args[4] == "Partial output.\nTimeout.\n", (
            "Timeouts must retain partial stdout and stderr."
        )


def test_timeout_does_not_hide_cleanup_failure(backend):
    execute = backend.environment.execute.side_effect

    def timeout_test(action, **kwargs):
        if action["command"].startswith("bash "):
            raise subprocess.TimeoutExpired("test execution", 900)
        return execute(action, **kwargs)

    backend.environment.execute.side_effect = timeout_test
    backend.environment.cleanup.side_effect = RuntimeError("Cleanup failed.")
    with pytest.raises(swebench_grader.SWEGradingError, match="cleanup"):
        backend.grade()
    backend.sweep.assert_called_once()


@pytest.mark.parametrize("path", ["config.yaml", "pyproject.toml", "pytest.ini", "requirements.txt", "Dockerfile"])
def test_configuration_changes_are_evaluated(backend, path):
    result = backend.grade(PATCH.replace("value.txt", path))
    assert result["resolved"], "Configuration edits must be judged by the official test report."
    assert not result["policy_violated"], "SWE-smith must not reject configuration filenames."


def test_collection_failure_is_a_model_outcome_with_a_complete_report(backend):
    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START

    backend.state.output = f"+ : '{TEST_OUTPUT_START}'\nERROR collecting tests\n+ : '{TEST_OUTPUT_END}'\n"
    backend.state.returncode = 1
    result = backend.grade()
    assert result["apply_ok"] and not result["resolved"], "A source patch can make the original tests fail collection."
    assert result["f2p_pass"] == result["p2p_pass"] == 0, "The official harness must account for missing tests."
    score = compute_score(
        "swe_smith_py",
        "",
        {},
        extra_info={"patch": PATCH, "num_turns": 3, "grader_result": result},
        reward_mode="partial_credit",
    )
    assert score["score"] == -0.3, "A complete failing test report must retain the task's regression reward."


@pytest.mark.parametrize("has_source_fix", [False, True])
def test_real_git_restores_tests_and_preserves_configuration_edits(monkeypatch, tmp_path, task, has_source_fix):
    from minisweagent.environments import docker
    from swesmith.profiles import registry

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout

    git("init", "--initial-branch=main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (tmp_path / "tests").mkdir()
    (tmp_path / "value.txt").write_text("1\n")
    (tmp_path / "config.yaml").write_text("value: original\n")
    original_fix = "from pathlib import Path\ndef test_fix():\n    assert Path('value.txt').read_text() == '0\\n'\n"
    original_keep = "def test_keep():\n    assert True\n"
    (tmp_path / "tests/test_fix.py").write_text(original_fix)
    (tmp_path / "tests/test_keep.py").write_text(original_keep)
    git("add", ".")
    git("commit", "--quiet", "-m", "Bug revision with original tests")
    bug_commit = git("rev-parse", "HEAD").strip()
    git("checkout", "-b", task["instance_id"])
    git("rm", "tests/test_fix.py")
    git("commit", "--quiet", "-m", "Remove hidden test")
    git("checkout", "--detach", bug_commit)
    (tmp_path / "config.yaml").write_text("value: changed\n")
    (tmp_path / "tests/test_fix.py").write_text("def test_fix():\n    assert True\n")
    (tmp_path / "tests/test_keep.py").write_text("def test_keep():\n    assert False\n")
    if has_source_fix:
        (tmp_path / "value.txt").write_text("0\n")
    patch = git("diff")
    git("reset", "--hard", "HEAD")

    class Environment:
        container_id = "local-grader"

        def execute(self, action, *, cwd="/testbed", timeout=60):
            command = (
                action["command"]
                .replace("cd /testbed", f"cd {shlex.quote(str(tmp_path))}")
                .replace("/tmp/psrl-grading", shlex.quote(str(tmp_path / "grading-inputs")))
            )
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=tmp_path,
                timeout=timeout,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return {"returncode": result.returncode, "output": result.stdout, "exception_info": ""}

        def cleanup(self):
            pass

    profile = registry.get_from_inst(task)
    monkeypatch.setattr(
        profile, "get_test_cmd", lambda *args, **kwargs: ("python -m pytest -v -c /dev/null --rootdir=. tests", [])
    )
    monkeypatch.setattr(registry, "get_from_inst", lambda task: profile)
    monkeypatch.setattr(docker, "DockerEnvironment", lambda **kwargs: Environment())
    monkeypatch.setattr(swebench_grader, "force_remove_containers_by_label", Mock())

    def stage_inputs(
        environment,
        model_patch,
        eval_script,
        *,
        timeout,
    ):
        assert timeout == 120, "Real Git grading must retain the preparation timeout."
        directory = tmp_path / "grading-inputs"
        directory.mkdir(exist_ok=True)
        (directory / "model.patch").write_text(model_patch + "\n")
        script = eval_script.replace("cd /testbed", f"cd {shlex.quote(str(tmp_path))}")
        (directory / "eval.sh").write_text(script + "\n")

    monkeypatch.setattr(swebench_grader, "_stage_grading_inputs", stage_inputs)
    result = swebench_grader.grade_fresh_container(task, patch, "smith", task["image_name"])
    assert result["resolved"] is has_source_fix, f"Test edits must not resolve an unrepaired source: {result!r}."
    assert (tmp_path / "config.yaml").read_text() == "value: changed\n", "Configuration edits must reach evaluation."
    assert (tmp_path / "tests/test_fix.py").read_text() == original_fix, "Hidden evaluation tests must be restored."
    assert (tmp_path / "tests/test_keep.py").read_text() == original_keep, "Existing P2P tests must be restored."
