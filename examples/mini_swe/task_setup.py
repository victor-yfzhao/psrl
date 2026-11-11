"""
Select SWE-smith task revisions consistently for rollout and grading.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from typing import Any

psrl_logger = logging.getLogger(__name__)


class SWETaskSetupError(RuntimeError):
    """
    A task workspace could not be initialized and must not receive a reward.
    """


class SWETaskSetupTimeout(SWETaskSetupError):
    """
    A preparation command exceeded its execution deadline.
    """


def execute_setup_command(
    environment: Any,
    command: str,
) -> str:
    """
    Reject failed setup commands before model execution or test scoring.

    Args:
        environment: mini-SWE-agent environment containing the task repository.
        command: Setup command to execute in the task repository.

    Returns:
        str: Command output.

    Raises:
        SWETaskSetupError: Execution failed or reported an environment exception.
    """
    try:
        result = environment.execute({"command": command}, cwd="/testbed", timeout=60)
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        raise SWETaskSetupTimeout(f"Task setup command {command!r} timed out: {exc}.") from exc
    except Exception as exc:
        raise SWETaskSetupError(f"Task setup command {command!r} failed: {exc}.") from exc
    if result.get("returncode") != 0 or result.get("exception_info"):
        if result.get("extra", {}).get("exception_type") in ("TimeoutExpired", "TimeoutError"):
            raise SWETaskSetupTimeout(f"Task setup command {command!r} timed out: {result!r}.")
        raise SWETaskSetupError(f"Task setup command {command!r} failed: {result!r}.")
    return result.get("output", "")


def initialize_smith_task(
    environment: Any,
    swe_problem: dict[str, Any],
    *,
    restore_tests: bool = False,
) -> str:
    """
    Check out the task branch and restore its hidden tests only for grading.

    Args:
        environment: mini-SWE-agent environment containing the task repository.
        swe_problem: SWE-smith dataset row with its task branch in `instance_id`.
        restore_tests: Whether to select the parent containing the removed tests.

    Returns:
        str: Selected commit hash.

    Raises:
        SWETaskSetupError: The task ID is invalid or workspace setup failed.
    """
    instance_id = swe_problem.get("instance_id")
    if not isinstance(instance_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", instance_id):
        raise SWETaskSetupError(f"Invalid SWE-smith instance_id {instance_id!r}.")
    execute_setup_command(environment, "git reset --hard HEAD && git clean -fd")
    local_ref = shlex.quote(f"refs/heads/{instance_id}")
    remote_ref = shlex.quote(f"refs/remotes/origin/{instance_id}")
    refspec = shlex.quote(f"refs/heads/{instance_id}:refs/heads/{instance_id}")
    # Some images contain only the base snapshot, so materialize the exact task ref from its origin.
    execute_setup_command(
        environment,
        f"git show-ref --verify --quiet {local_ref} || "
        f"git show-ref --verify --quiet {remote_ref} || "
        f"git fetch --no-tags origin {refspec}",
    )
    execute_setup_command(environment, f"git checkout --force {shlex.quote(instance_id)}")
    if restore_tests:
        # Each task branch removes hidden tests in the commit immediately after the bug.
        execute_setup_command(environment, "git checkout --detach HEAD~1")
    execute_setup_command(environment, "git reset --hard HEAD && git clean -fd")
    commit = execute_setup_command(environment, "git rev-parse HEAD").strip()
    psrl_logger.info(f"Initialized SWE-smith task {instance_id!r} at {commit!r}, restore_tests={restore_tests!r}.")
    return commit


def smith_hidden_test_files(
    environment: Any,
    swe_problem: dict[str, Any],
) -> list[str]:
    """
    Read hidden test paths from the task's test removal commit after restoring its parent.

    Args:
        environment: Initialized grading workspace at the task branch's parent.
        swe_problem: SWE-smith dataset row identifying the task branch.

    Returns:
        list[str]: Repository relative paths changed by the test removal commit.
    """
    task_ref = shlex.quote(swe_problem["instance_id"])
    output = execute_setup_command(environment, f"git diff --name-only -z HEAD {task_ref} --")
    return sorted(set(filter(None, output.split("\0"))))


def smith_evaluation_test_files(
    environment: Any,
    swe_problem: dict[str, Any],
) -> list[str]:
    """
    Resolve tracked profile tests and hidden tests in the restored grading revision.
    """
    from swesmith.profiles import registry

    profile = registry.get_from_inst(swe_problem)
    f2p_files, p2p_files = profile.get_test_files(swe_problem)
    candidates = sorted({str(path) for path in f2p_files + p2p_files})
    files = set(smith_hidden_test_files(environment, swe_problem))
    if candidates:
        paths = " ".join(shlex.quote(f":(literal){path}") for path in candidates)
        output = execute_setup_command(environment, f"git ls-files -z -- {paths}")
        files.update(filter(None, output.split("\0")))
    return sorted(files)


def initialize_benchmark_task(
    environment: Any,
    swe_problem: dict[str, Any],
) -> str:
    """
    Select the exact SWE-Gym or SWE-bench base commit for rollout and grading.
    """
    commit = swe_problem.get("base_commit", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        raise SWETaskSetupError(f"Invalid benchmark base_commit {commit!r}.")
    execute_setup_command(environment, "git reset --hard HEAD && git clean -fd")
    execute_setup_command(environment, f"git checkout --detach {shlex.quote(commit)}")
    execute_setup_command(environment, "git reset --hard HEAD && git clean -fd")
    return execute_setup_command(environment, "git rev-parse HEAD").strip()
