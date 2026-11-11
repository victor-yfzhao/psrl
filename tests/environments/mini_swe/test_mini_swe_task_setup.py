"""
Verify SWE-smith rollout and grading select the task's distinct Git revisions.
"""

import shlex
from dataclasses import dataclass
from pathlib import Path
from subprocess import run
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from examples.mini_swe import runner, swebench_grader
from examples.mini_swe.task_setup import (
    SWETaskSetupError,
    execute_setup_command,
    initialize_smith_task,
    smith_evaluation_test_files,
    smith_hidden_test_files,
)

pytestmark = pytest.mark.cpu_test

TASK_IDS = ("owner__repo.abc.func_basic__task_a", "owner__repo.abc.func_basic__task_b")
PATCH = "diff --git a/value.txt b/value.txt\n--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-1\n+0\n"


def _git(path, *args):
    return run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()


class GitEnvironment:
    """
    Run real workspace commands locally through mini-SWE-agent's environment API.
    """

    def __init__(self, path):
        self.path = path
        self.container_id = "local-testbed"
        self.commands = []
        self.failure = None
        self.cleanup = Mock()

    def execute(self, action, *, cwd="/testbed", timeout=60):
        assert cwd == "/testbed", "Task commands must run in the task repository."
        command = action["command"]
        self.commands.append(command)
        if self.failure and command.startswith(self.failure):
            return {"returncode": 0, "output": "", "exception_info": "Command execution timed out."}
        command = command.replace("/tmp/psrl-grading", shlex.quote(str(self.path / "grading-inputs")))
        result = run(["bash", "-c", command], cwd=self.path, capture_output=True, text=True, timeout=timeout)
        return {"returncode": result.returncode, "output": result.stdout + result.stderr}


@dataclass
class TaskImage:
    path: Path
    commits: dict

    def workspace(self, path):
        _git(self.path, "clone", "--quiet", str(self.path), str(path))
        return GitEnvironment(path)


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "image"
    path.mkdir()
    _git(path, "init", "--initial-branch=main")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "user.email", "test@example.invalid")
    (path / "value.txt").write_text("0\n")
    (path / "tests").mkdir()
    (path / "tests/test_value.py").write_text(
        "from pathlib import Path\n"
        "assert (Path(__file__).parents[1] / 'value.txt').read_text() == '0\\n'\n"
        "print('PASS')\n"
    )
    _git(path, "add", ".")
    _git(path, "commit", "--quiet", "-m", "Base")
    commits = {}
    for index, task_id in enumerate(TASK_IDS, start=1):
        _git(path, "checkout", "-b", task_id, "main")
        (path / "value.txt").write_text(f"{index}\n")
        _git(path, "commit", "--quiet", "-am", "Introduce bug")
        bug = _git(path, "rev-parse", "HEAD")
        _git(path, "rm", "--quiet", "tests/test_value.py")
        _git(path, "commit", "--quiet", "-m", "Remove hidden test")
        commits[task_id] = (bug, _git(path, "rev-parse", "HEAD"))
    _git(path, "checkout", "main")
    return TaskImage(path, commits)


@pytest.mark.parametrize("task_index", [0, 1])
@pytest.mark.parametrize("restore_tests", [False, True])
def test_task_revision_and_hidden_tests_match_instance_id(image, tmp_path, task_index, restore_tests):
    environment = image.workspace(tmp_path / "workspace")
    (environment.path / "value.txt").write_text("Unrelated work.\n")
    (environment.path / "untracked.txt").write_text("Unrelated file.\n")
    task_id = TASK_IDS[task_index]
    commit = initialize_smith_task(environment, {"instance_id": task_id}, restore_tests=restore_tests)

    expected_commit = image.commits[task_id][0 if restore_tests else 1]
    assert commit == expected_commit, "The selected revision must belong to the requested task."
    assert (environment.path / "value.txt").read_text() == f"{task_index + 1}\n", (
        "Both task revisions must contain that task's bug."
    )
    assert (environment.path / "tests/test_value.py").exists() == restore_tests, (
        "Hidden tests must be available only to the grader."
    )
    assert not _git(environment.path, "status", "--porcelain"), "Task initialization must leave a clean tree."


@pytest.mark.parametrize("restore_tests", [False, True])
def test_base_only_image_fetches_exact_task_revision(image, tmp_path, restore_tests):
    path = tmp_path / "base-only-image"
    _git(image.path, "clone", "--quiet", "--single-branch", "--branch", "main", str(image.path), str(path))
    environment = GitEnvironment(path)
    task_id = TASK_IDS[0]

    commit = initialize_smith_task(environment, {"instance_id": task_id}, restore_tests=restore_tests)

    assert commit == image.commits[task_id][0 if restore_tests else 1], (
        "A base snapshot must fetch and select the requested task rather than use its default revision."
    )
    assert (path / "tests/test_value.py").exists() == restore_tests, (
        "Fetching the task must preserve the distinction between rollout and grading revisions."
    )


def test_partial_test_removal_commit_protects_modified_test_file(image, tmp_path, monkeypatch):
    task_id = "owner__repo.abc.func_basic__partial"
    _git(image.path, "checkout", "-b", task_id, "main")
    test_path = image.path / "tests/test_value.py"
    test_path.write_text("def test_hidden():\n    assert False\n\ndef test_public():\n    assert True\n")
    _git(image.path, "commit", "--quiet", "-am", "Introduce bug")
    test_path.write_text("def test_public():\n    assert True\n")
    _git(image.path, "commit", "--quiet", "-am", "Remove hidden test function")
    environment = image.workspace(tmp_path / "grader")
    problem = {"instance_id": task_id, "FAIL_TO_PASS": ["test_hidden (__main__.Tests)"]}
    initialize_smith_task(environment, problem, restore_tests=True)

    hidden_files = smith_hidden_test_files(environment, problem)

    assert hidden_files == ["tests/test_value.py"], "Partially removed test files must retain their protection."
    assert "def test_hidden" in (environment.path / hidden_files[0]).read_text(), (
        "The grading revision must restore hidden functions inside retained files."
    )
    from swesmith.profiles import registry

    monkeypatch.setattr(registry, "get_from_inst", lambda task: SimpleNamespace(get_test_files=lambda task: ([], [])))
    files = smith_evaluation_test_files(environment, problem)
    original = (environment.path / hidden_files[0]).read_text()
    (environment.path / hidden_files[0]).write_text("print('MODIFIED')\n")
    execute_setup_command(environment, "git checkout -- " + " ".join(files))
    assert (environment.path / hidden_files[0]).read_text() == original, (
        "Partially removed hidden tests must be restored despite unittest display identifiers."
    )


@pytest.mark.parametrize("instance_id", [None, "", "-b", "task; touch marker", "task\nother"])
def test_invalid_instance_id_cannot_execute_workspace_commands(instance_id):
    environment = Mock()
    with pytest.raises(SWETaskSetupError, match="Invalid SWE-smith instance_id"):
        initialize_smith_task(environment, {"instance_id": instance_id})
    environment.execute.assert_not_called()


@pytest.mark.parametrize("missing_parent", [False, True])
def test_missing_task_revision_is_an_initialization_error(image, tmp_path, missing_parent):
    environment = image.workspace(tmp_path / "workspace")
    task_id = "owner__repo.abc.missing"
    if missing_parent:
        _git(environment.path, "branch", task_id, "main")
    with pytest.raises(SWETaskSetupError, match="Task setup command"):
        initialize_smith_task(environment, {"instance_id": task_id}, restore_tests=True)


@pytest.mark.parametrize("raises", [False, True])
def test_environment_exceptions_cannot_report_successful_setup(raises):
    environment = Mock()
    if raises:
        environment.execute.side_effect = TimeoutError("Command timed out.")
    else:
        environment.execute.return_value = {"returncode": 0, "exception_info": "Command timed out."}
    with pytest.raises(SWETaskSetupError, match="Command timed out"):
        execute_setup_command(environment, "git rev-parse HEAD")


def _problem(task_id=TASK_IDS[0]):
    return {"instance_id": task_id, "FAIL_TO_PASS": ["tests/test_value.py::test_value"], "PASS_TO_PASS": []}


def _configure_grader(monkeypatch, environment):
    from minisweagent.environments import docker
    from swesmith.profiles import registry

    monkeypatch.setattr(registry, "get_from_inst", lambda task: SimpleNamespace(get_test_files=lambda task: ([], [])))
    monkeypatch.setattr(docker, "DockerEnvironment", Mock(return_value=environment))
    monkeypatch.setattr(swebench_grader, "force_remove_containers_by_label", Mock())
    monkeypatch.setattr(swebench_grader, "_get_smith_eval_script", lambda problem: "python tests/test_value.py")

    def stage_inputs(
        environment,
        model_patch,
        eval_script,
        *,
        timeout,
    ):
        assert timeout == 120, "Task setup must retain the grading preparation timeout."
        directory = environment.path / "grading-inputs"
        directory.mkdir(exist_ok=True)
        (directory / "model.patch").write_text(model_patch + "\n")
        (directory / "eval.sh").write_text(eval_script + "\n")

    monkeypatch.setattr(swebench_grader, "_stage_grading_inputs", stage_inputs)

    def parse_grade(problem, patch, output, apply_ok, returncode):
        assert apply_ok and returncode == 0, "The repair must apply to the selected bug and pass its restored test."
        assert "PASS" in output, "The restored test must execute successfully."
        return {
            "resolved": True,
            "f2p_pass": 1,
            "f2p_total": 1,
            "p2p_pass": 0,
            "p2p_total": 0,
            "resolved_by": "test_harness",
        }

    parse = Mock(side_effect=parse_grade)
    monkeypatch.setattr(swebench_grader, "_grade_smith", parse)
    return parse


@pytest.mark.parametrize("failure", [None, "git checkout --force", "git checkout --detach", "git checkout -- tests/"])
def test_grader_uses_task_parent_and_stops_on_setup_failure(image, tmp_path, monkeypatch, failure):
    environment = image.workspace(tmp_path / "grader")
    environment.failure = failure
    parse = _configure_grader(monkeypatch, environment)
    if failure:
        with pytest.raises(SWETaskSetupError):
            swebench_grader.grade_fresh_container(_problem(), PATCH, "smith", "example/swesmith:latest")
        parse.assert_not_called()
        assert "bash /tmp/psrl-grading/eval.sh" not in environment.commands, (
            "Initialization failure must prevent test execution and scoring."
        )
    else:
        result = swebench_grader.grade_fresh_container(_problem(), PATCH, "smith", "example/swesmith:latest")
        assert result["resolved"], "The patch must be graded against the requested task."
        assert _git(environment.path, "rev-parse", "HEAD") == image.commits[TASK_IDS[0]][0], (
            "The grader must restore tests from the requested task's parent."
        )
        assert (environment.path / "tests/test_value.py").exists(), "Grading must retain the restored hidden test."
    environment.cleanup.assert_called_once()


@pytest.mark.parametrize("custom_resources", [False, True])
def test_grader_applies_shared_container_resources(image, tmp_path, monkeypatch, custom_resources):
    from minisweagent.environments import docker

    environment = image.workspace(tmp_path / "grader")
    _configure_grader(monkeypatch, environment)
    monkeypatch.setenv("PSRL_ACTOR_ID", "unrelated-owner")
    resources = (
        {
            "run_args": ["--rm", "--cpus=2", "--memory=4g", "--shm-size=1g", "--label", "psrl.actor_id=loop-owner"],
            "env": {"MODIN_CPUS": "2"},
            "memory": "12g",
            "shm_size": "6g",
        }
        if custom_resources
        else {}
    )

    result = swebench_grader.grade_fresh_container(_problem(), PATCH, "smith", "example/swesmith:latest", **resources)

    assert result["resolved"], "Resource configuration must retain a working fresh-container grading path."
    options = docker.DockerEnvironment.call_args.kwargs
    assert "psrl.actor_id=unrelated-owner" not in options["run_args"], "Grading must not inherit another loop's owner."
    assert ("psrl.actor_id=loop-owner" in options["run_args"]) is custom_resources, (
        "Explicit container ownership must survive harness setup."
    )
    assert f"--cpus={2 if custom_resources else 8}" in options["run_args"], (
        "The grader must enforce the selected CPU quota."
    )
    assert [arg for arg in options["run_args"] if arg.startswith("--memory=")] == [
        "--memory=12g" if custom_resources else "--memory=30g"
    ], "The grader must replace rollout memory with one grading memory limit."
    assert [arg for arg in options["run_args"] if arg.startswith("--shm-size=")] == [
        "--shm-size=6g" if custom_resources else "--shm-size=16g"
    ], "The grader must replace rollout shared memory with its grading limit."
    assert options["env"]["MODIN_CPUS"] == ("2" if custom_resources else "8"), (
        "The grader must bound library worker discovery alongside the Docker quota."
    )


@pytest.mark.parametrize("test_id", ["../dev/tests/test_value.py::test_value", "test_value (__main__.TestValue)"])
def test_grader_reads_hidden_test_paths_from_task_commit(image, tmp_path, monkeypatch, test_id):
    environment = image.workspace(tmp_path / "grader")
    _configure_grader(monkeypatch, environment)
    problem = _problem()
    problem["FAIL_TO_PASS"] = [test_id]

    result = swebench_grader.grade_fresh_container(problem, PATCH, "smith", "example/swesmith:latest")

    assert result["resolved"], "Report identifiers must not prevent grading the requested task's restored tests."
    assert "git checkout -- tests/test_value.py" in environment.commands, (
        "The grader must restore the tracked path from the hidden test removal commit."
    )


@pytest.mark.parametrize("test_id", ["../dev/tests/test_value.py::test_value", "test_value (__main__.TestValue)"])
def test_report_identifiers_cannot_bypass_hidden_test_restoration(image, tmp_path, monkeypatch, test_id):
    environment = image.workspace(tmp_path / "grader")
    parse = _configure_grader(monkeypatch, environment)
    problem = _problem()
    problem["FAIL_TO_PASS"] = [test_id]
    patch = PATCH + (
        "diff --git a/tests/test_value.py b/tests/test_value.py\n"
        "--- a/tests/test_value.py\n"
        "+++ b/tests/test_value.py\n"
        "@@ -3 +3 @@\n"
        "-print('PASS')\n"
        "+print('MODIFIED')\n"
    )

    result = swebench_grader.grade_fresh_container(problem, patch, "smith", "example/swesmith:latest")

    assert result["resolved"], "Source repairs must be scored after restoring the original hidden tests."
    assert not result["policy_violated"], "SWE-smith submissions must follow the official test restoration policy."
    assert "print('PASS')" in (environment.path / "tests/test_value.py").read_text(), (
        "Test restoration must use the task's authoritative hidden test paths."
    )
    parse.assert_called_once()
    environment.cleanup.assert_called_once()


@pytest.mark.parametrize("failure_stage", [None, "rollout", "grader"])
def test_rollout_and_reward_service_initialize_separate_task_workspaces(image, tmp_path, monkeypatch, failure_stage):
    from minisweagent import environments, models
    from minisweagent.agents import default
    from psrl.utils.agent import overflow

    rollout_environment = image.workspace(tmp_path / "rollout")
    grader_environment = image.workspace(tmp_path / "grader")
    if failure_stage == "rollout":
        rollout_environment.failure = "git checkout --force"
    elif failure_stage == "grader":
        grader_environment.failure = "git checkout --detach"
    parse = _configure_grader(monkeypatch, grader_environment)
    agent_runs = Mock()

    class Agent:
        def __init__(self, model, environment, **kwargs):
            self.environment = environment

        def run(self, task):
            agent_runs()
            assert (self.environment.path / "value.txt").read_text() == "1\n", (
                "The agent must see its task's bug before generating a repair."
            )
            assert not (self.environment.path / "tests/test_value.py").exists(), (
                "The rollout must not expose the grader's hidden test."
            )
            (self.environment.path / "value.txt").write_text("0\n")
            return {"submission": _git(self.environment.path, "diff") + "\n", "exit_status": "submitted"}

    monkeypatch.setattr(default, "DefaultAgent", Agent)
    monkeypatch.setattr(environments, "get_environment", Mock(return_value=rollout_environment))
    get_model = Mock(return_value=SimpleNamespace(abort_exceptions=[], _query=Mock()))
    monkeypatch.setattr(models, "get_model", get_model)
    monkeypatch.setattr(overflow, "ensure_overflow_handling", lambda model: None)
    monkeypatch.setattr(runner, "_silence_litellm", lambda: None)
    payload = {
        "runtime_config": {
            "sandbox_config": {
                "environment": {"image": "example/swesmith:latest", "grader_timeout": 1800},
                "rollout_turn_timeout": 60,
            },
            "model": {},
            "agent": {"problem_template": "task"},
        },
        "observation": {
            "swe_restore_tests": True,
            "swe_problem": _problem(),
            "swe_grader": "swebench_fresh_container",
            "swe_grader_kind": "smith",
            "swe_problem_image": "example/swesmith:latest",
        },
        "sampling_params": {},
        "base_url": "http://gateway",
        "model": "test-model",
        "max_turns": 2,
        "task": "Fix the value.",
    }

    result = runner.run_agent(payload)
    rollout_environment.cleanup.assert_called_once()
    if failure_stage == "rollout":
        assert result["exit_status"] == "error", "Rollout setup failures must enter the rollout retry path."
        parse.assert_not_called()
        agent_runs.assert_not_called()
        get_model.assert_not_called()
        grader_environment.cleanup.assert_not_called()
        return

    from dataclasses import asdict

    from examples.mini_swe.config import MiniEnvironmentConfig
    from examples.mini_swe.reward import MiniSWERewardLoop

    assert result["exit_status"] == "submitted", "Rollout success must not depend on grading completion."
    assert "grader_result" not in result, "Grading results must originate in the reward service."
    grader_environment.cleanup.assert_not_called()
    agent_runs.assert_called_once()
    context = {**payload["observation"], "swe_task_id": "episode", "environment": asdict(MiniEnvironmentConfig())}
    reward_loop = MiniSWERewardLoop.__new__(MiniSWERewardLoop)
    reward_loop._container_owner = "test-grader"
    if failure_stage == "grader":
        with pytest.raises(SWETaskSetupError):
            reward_loop._run_grader(context, result["submission"])
        parse.assert_not_called()
    else:
        grade = reward_loop._run_grader(context, result["submission"])
        assert grade["resolved"], "The agent's repair must pass in the separately initialized grader."
    grader_environment.cleanup.assert_called_once()


@pytest.mark.parametrize("failure", [False, True])
def test_standalone_evaluation_initializes_task_before_agent(image, tmp_path, monkeypatch, failure):
    from examples.mini_swe.eval.eval_swebench import _run_agent_on_swe_problem
    from minisweagent import environments, models
    from minisweagent.agents import default
    from psrl.utils.agent import overflow

    environment = image.workspace(tmp_path / "evaluation")
    if failure:
        environment.failure = "git checkout --force"
    agent_run = Mock()

    class Agent:
        def __init__(self, model, env, **kwargs):
            pass

        def run(self, task):
            agent_run()
            assert (environment.path / "value.txt").read_text() == "1\n", (
                "Standalone evaluation must select the same bug revision as training."
            )
            assert not (environment.path / "tests/test_value.py").exists(), (
                "Standalone rollout must keep hidden tests unavailable."
            )
            return {"submission": PATCH, "exit_status": "submitted"}

    monkeypatch.setattr(default, "DefaultAgent", Agent)
    monkeypatch.setattr(environments, "get_environment", Mock(return_value=environment))
    monkeypatch.setattr(models, "get_model", Mock(return_value=SimpleNamespace(abort_exceptions=[], _query=Mock())))
    monkeypatch.setattr(overflow, "ensure_overflow_handling", lambda model: None)
    config_path = Path(__file__).resolve().parents[3] / "examples/mini_swe/config/swe_smith_agent_config.yaml"
    problem = {**_problem(), "image_name": "example/swesmith:latest", "problem_statement": "Fix the value."}
    result = _run_agent_on_swe_problem(problem, "test-model", str(config_path), 2, 1.0)
    assert result["exit_status"] == ("error" if failure else "submitted"), (
        "Standalone task initialization errors must be reported as episode failures."
    )
    if failure:
        agent_run.assert_not_called()
    else:
        agent_run.assert_called_once()
    environment.cleanup.assert_called_once()
