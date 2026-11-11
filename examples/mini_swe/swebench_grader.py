"""
Grade SWE-bench patches in fresh Docker containers.

The grader enforces patch policy, runs tests, and parses official harness results.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from filelock import FileLock
from psrl.utils.common.docker_utils import force_remove_containers_by_label

from examples.mini_swe.benchmark_grading import (
    analyze_patch_policy,
    baseline_report_is_complete,
    configure_gym_test_runtime,
    frame_eval_script,
    model_collection_failure_candidate,
    parse_benchmark_report,
    syntax_probe_manifest,
)
from examples.mini_swe.config import MiniEnvironmentConfig
from examples.mini_swe.task_setup import (
    SWETaskSetupError,
    SWETaskSetupTimeout,
    execute_setup_command,
    initialize_benchmark_task,
    initialize_smith_task,
    smith_evaluation_test_files,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# --- Constants ---

_DEFAULT_EVAL_TIMEOUT = MiniEnvironmentConfig.grader_timeout
_OUTPUT_TAIL_BYTES = 4096
_BASELINE_CACHE_DIRECTORY = Path(tempfile.gettempdir()) / f"psrl-swe-baseline-{os.getpid()}-{uuid4().hex}"
_FORWARDED_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
]


class SWEGradingError(RuntimeError):
    """
    Indicate that a harness failure prevents a reliable training reward.
    """


class SWEGradingPreparationTimeout(SWEGradingError):
    """
    Preparation timed out before the test execution budget started.
    """


def _save_failure_artifacts(
    error: Exception,
    task: dict[str, Any],
    model_patch: str,
    eval_script: str,
    eval_output: str,
    image_name: str,
    task_id: str | None,
) -> None:
    """
    Preserve exact grading inputs and output after disposable containers are removed.
    """
    try:
        directory = Path(tempfile.mkdtemp(prefix="psrl-swe-failure-"))
        (directory / "task.json").write_text(json.dumps(task, default=str, indent=2), encoding="utf-8")
        (directory / "model.patch").write_text(model_patch, encoding="utf-8")
        (directory / "eval.sh").write_text(eval_script, encoding="utf-8")
        (directory / "eval.log").write_text(eval_output, encoding="utf-8")
        (directory / "failure.json").write_text(
            json.dumps({"error": str(error), "image_name": image_name, "task_id": task_id}, indent=2),
            encoding="utf-8",
        )
        error.add_note(f"Grading failure artifacts: {directory}.")
        psrl_logger.warning("Saved grading failure artifacts for task %r to %s.", task_id, directory)
    except Exception:
        psrl_logger.warning("Could not preserve grading failure artifacts.", exc_info=True)


# --- Patch policy ---


def _policy_blocked_result(
    policy: dict[str, Any],
    swe_problem: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    """
    Report a rejected patch without running the test harness.
    """
    return {
        "policy_violated": True,
        "policy_reasons": policy["reasons"],
        "resolved": False,
        "apply_ok": False,
        "f2p_pass": 0,
        "f2p_total": len(swe_problem.get("FAIL_TO_PASS", [])),
        "p2p_pass": 0,
        "p2p_total": len(swe_problem.get("PASS_TO_PASS", [])),
        "timeout": False,
        "error": None,
        "elapsed_s": elapsed_s,
        "output_tail": "",
        "resolved_by": "policy_blocked",
    }


# --- Evaluation script resolution ---


@lru_cache(maxsize=2048)
def _get_verified_eval_script(swe_problem_id: str, swe_problem_json: str) -> str:
    """
    Build and cache the eval script for a SWE-bench Verified SWE problem.

    Args:
        swe_problem_id (str): SWE problem ID (the HF ``instance_id`` field),
            used as the cache key.
        swe_problem_json (str): JSON-serialised SWE problem dict (full row).

    Returns:
        str: Bash eval script from ``make_test_spec``.
    """
    import json

    from swebench.harness.test_spec.test_spec import make_test_spec

    swe_problem = json.loads(swe_problem_json)
    ts = make_test_spec(swe_problem)
    return frame_eval_script(ts.eval_script, postlude="python /tmp/psrl-grading/syntax_probe.py check")


def _get_smith_eval_script(swe_problem: dict[str, Any]) -> str:
    """
    Build the eval script for a SWE-smith SWE problem using the profiles registry.

    Args:
        swe_problem (dict[str, Any]): Full SWE-smith dataset row.

    Returns:
        str: Bash eval script string.
    """
    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START
    from swesmith.profiles import registry

    rp = registry.get_from_inst(swe_problem)
    cmd, _ = rp.get_test_cmd(swe_problem, f2p_only=False)
    return "\n".join(
        [
            "#!/bin/bash",
            "set -uxo pipefail",
            "cd /testbed",
            f": '{TEST_OUTPUT_START}'",
            cmd,
            f": '{TEST_OUTPUT_END}'",
            "",
        ]
    )


# --- Grading helpers ---


def _stage_grading_inputs(
    docker_env: Any,
    model_patch: str,
    eval_script: str,
    *,
    timeout: int = MiniEnvironmentConfig.grader_command_timeout,
) -> None:
    """
    Copy grading artifacts as files without placing their contents in process arguments.
    """
    with tempfile.TemporaryDirectory(prefix="psrl-swe-grading-") as directory:
        root = Path(directory)
        (root / "model.patch").write_text(model_patch + "\n", encoding="utf-8")
        (root / "eval.sh").write_text(eval_script + "\n", encoding="utf-8")
        (root / "syntax_files.json").write_text(
            json.dumps(syntax_probe_manifest(model_patch, timeout=timeout)), encoding="utf-8"
        )
        (root / "syntax_probe.py").write_bytes(Path(__file__).with_name("syntax_probe.py").read_bytes())
        try:
            subprocess.run(
                [
                    docker_env.config.executable,
                    "cp",
                    f"{directory}/.",
                    f"{docker_env.container_id}:/tmp/psrl-grading",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            raise SWEGradingError(f"SWE grading input transfer failed: {exc.stdout}.") from exc


def _grade_benchmark(
    swe_problem: dict[str, Any],
    eval_output: str,
    apply_ok: bool,
    grader_kind: str,
    model_patch: str,
    *,
    baseline_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Separate patch application failures from invalid benchmark reports.
    """
    if not apply_ok:
        return {
            "resolved": False,
            "f2p_pass": 0,
            "f2p_total": len(swe_problem["FAIL_TO_PASS"]),
            "p2p_pass": 0,
            "p2p_total": len(swe_problem["PASS_TO_PASS"]),
            "resolved_by": "apply_failed",
        }
    try:
        return parse_benchmark_report(
            swe_problem, eval_output, grader_kind, model_patch=model_patch, baseline_report=baseline_report
        )
    except Exception as exc:
        raise SWEGradingError(
            f"Invalid benchmark report for {swe_problem['instance_id']!r}: {exc}. "
            f"Evaluation output tail: {eval_output[-_OUTPUT_TAIL_BYTES:]!r}."
        ) from exc


def _container_image_id(docker_env: Any) -> str:
    """
    Pin the baseline control to the evaluated container's immutable image.
    """
    result = subprocess.run(
        [docker_env.config.executable, "inspect", "--format={{.Image}}", docker_env.container_id],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    image_id = result.stdout.strip()
    if not image_id.startswith("sha256:"):
        raise SWEGradingError(f"Invalid evaluation container image ID {image_id!r}.")
    return image_id


def _get_baseline_report(
    swe_problem: dict[str, Any],
    grader_kind: str,
    image_id: str,
    timeout: int,
    swe_task_id: str,
    memory: str,
    run_args: list[str],
    env: dict[str, str],
    shm_size: str,
) -> dict[str, Any]:
    """
    Cache baseline readiness or its execution error for one actor lifetime and deduplicate concurrent controls.
    """
    import hashlib

    context = {
        "task": swe_problem,
        "grader_kind": grader_kind,
        "image_id": image_id,
        "timeout": timeout,
        "memory": memory,
        "run_args": run_args,
        "env": env,
        "shm_size": shm_size,
        "forwarded_env": {key: os.environ.get(key) for key in _FORWARDED_ENV_KEYS},
    }
    key = hashlib.sha256(json.dumps(context, sort_keys=True).encode("utf-8")).hexdigest()
    _BASELINE_CACHE_DIRECTORY.mkdir(exist_ok=True)
    path = _BASELINE_CACHE_DIRECTORY / f"{key}.json"
    # Wait for the bounded control to finish before reusing its readiness result.
    with FileLock(str(path.with_suffix(".lock"))):
        if path.exists():
            cached = json.loads(path.read_text())
        else:
            artifact = f"psrl_baseline_{key}.txt"
            baseline_patch = (
                f"diff --git a/{artifact} b/{artifact}\nnew file mode 100644\n"
                f"--- /dev/null\n+++ b/{artifact}\n@@ -0,0 +1 @@\n+Baseline readiness control.\n"
            )
            psrl_logger.info(
                f"Verifying task baseline: instance_id={swe_problem['instance_id']!r}, image_id={image_id!r}."
            )
            try:
                report = grade_fresh_container(
                    swe_problem,
                    baseline_patch,
                    grader_kind,
                    image_id,
                    timeout,
                    f"{swe_task_id or swe_problem['instance_id']}__baseline",
                    memory,
                    run_args=run_args,
                    env=env,
                    shm_size=shm_size,
                )
                if not baseline_report_is_complete(swe_problem, report):
                    raise SWEGradingError("The baseline did not observe all official task test results.")
                cached = {"status": "ready", "report": {**report, "baseline_image_id": image_id}}
            except Exception as exc:
                cached = {"status": "error", "error": str(exc)}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(cached))
            temporary.replace(path)
        if cached["status"] != "ready":
            raise SWEGradingError(f"Baseline control failed for {swe_problem['instance_id']!r}: {cached['error']}.")
        return cached["report"]


def _grade_smith(
    swe_problem: dict[str, Any],
    model_patch: str,
    eval_output: str,
    apply_ok: bool,
    returncode: int,
) -> dict[str, Any]:
    """
    Grade a SWE-smith-py SWE problem rollout with the swesmith harness.

    Args:
        swe_problem (dict[str, Any]): Full SWE-smith-py row.
        model_patch (str): Patch submitted by the model.
        eval_output (str): Raw stdout/stderr from the eval script.
        apply_ok (bool): Whether git apply succeeded.
        returncode (int): Exit code of the eval script.

    Returns:
        dict[str, Any]: Grading result.

    Raises:
        SWEGradingError: If the harness cannot produce a complete test report.
    """

    from swesmith.harness.grading import get_eval_report as smith_get_eval_report

    f2p = swe_problem.get("FAIL_TO_PASS", [])
    p2p = swe_problem.get("PASS_TO_PASS", [])

    if not apply_ok:
        return {
            "resolved": False,
            "f2p_pass": 0,
            "f2p_total": len(f2p),
            "p2p_pass": 0,
            "p2p_total": len(p2p),
            "resolved_by": "apply_failed",
        }

    try:
        prediction = {
            "instance_id": swe_problem["instance_id"],
            "model_name_or_path": "psrl/rollout",
            "model_patch": model_patch,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log") as tf:
            tf.write(eval_output)
            tf.flush()
            report = smith_get_eval_report(prediction, swe_problem, tf.name)
        if "tests_status" not in report:
            raise ValueError("SWE-smith harness report is missing tests_status.")
        tests_status = report["tests_status"]
        f2p_status = tests_status["FAIL_TO_PASS"]
        p2p_status = tests_status["PASS_TO_PASS"]
        return {
            "resolved": bool(report["resolved"]),
            "f2p_pass": len(f2p_status["success"]),
            "f2p_total": len(f2p),
            "p2p_pass": len(p2p_status["success"]),
            "p2p_total": len(p2p),
            "resolved_by": "harness",
        }
    except Exception as exc:
        raise SWEGradingError(
            f"SWE-smith harness grading failed for {swe_problem['instance_id']!r} "
            f"with eval returncode {returncode}: {exc}."
        ) from exc


# --- SWE-Gym grading ---


def _get_gym_eval_script(swe_problem: dict[str, Any]) -> str:
    """Retrieve the pre-computed eval_script from the swe_problem dict.

    SWE-Gym instances store the eval_script directly in the parquet
    (generated by prepare_swe_gym.py) because the swegym fork of swebench
    cannot coexist with swebench 4.x at runtime.

    Args:
        swe_problem (dict[str, Any]): Full SWE-Gym dataset row.

    Returns:
        str: Bash eval script string.

    Raises:
        ValueError: If eval_script is missing from swe_problem.
    """
    eval_script = swe_problem.get("eval_script", "")
    if not eval_script:
        raise ValueError(
            f"SWE-Gym instance {swe_problem.get('instance_id', '?')} "
            f"missing eval_script in swe_problem. "
            f"Re-run prepare_swe_gym.py to regenerate the parquet."
        )
    eval_script = configure_gym_test_runtime(eval_script, swe_problem["repo"])
    return frame_eval_script(eval_script, postlude="python /tmp/psrl-grading/syntax_probe.py check")


# --- Main grader entry point ---


def grade_fresh_container(
    swe_problem: dict[str, Any],
    model_patch: str,
    grader_kind: str,
    image_name: str,
    timeout: int = _DEFAULT_EVAL_TIMEOUT,
    swe_task_id: str = "",
    memory: str = "",
    *,
    run_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    shm_size: str = "",
    command_timeout: int = MiniEnvironmentConfig.grader_command_timeout,
) -> dict[str, Any]:
    """
    Grade a model patch in a fresh Docker container.

    Patch policy is checked before the isolated test harness runs.

    Args:
        swe_problem (dict[str, Any]): Full HF dataset row for one SWE problem.
        model_patch (str): Patch submitted by the agent (unified diff).
        grader_kind (str): ``"verified"``, ``"gym"``, or ``"smith"``.
        image_name (str): Docker image used for the rollout.
        timeout (int): Eval script execution timeout in seconds.
        swe_task_id (str): PSRL rollout episode ID for container labelling.
        memory (str): ``--memory`` limit for the grading container.
        run_args (list[str] | None): Docker options shared with the rollout container.
        env (dict[str, str] | None): Environment shared with the rollout container.
        shm_size (str): Shared memory limit for the grading container.
        command_timeout (int): Preparation command timeout in seconds.

    Returns:
        dict[str, Any]: Test counts, policy state, and execution diagnostics.
    """
    from minisweagent.environments.docker import DockerEnvironment

    swe_problem_id: str = swe_problem.get("instance_id", "unknown")
    log_prefix = f"[swebench_grader, task_id={swe_task_id or swe_problem_id}]"
    t0 = time.monotonic()

    f2p = swe_problem.get("FAIL_TO_PASS", [])
    p2p = swe_problem.get("PASS_TO_PASS", [])

    # --- 0. Patch policy guard ---
    if not model_patch:
        psrl_logger.info(f"{log_prefix} No patch submitted, skipping grading.")
        return {
            "policy_violated": False,
            "policy_reasons": [],
            "resolved": False,
            "apply_ok": False,
            "f2p_pass": 0,
            "f2p_total": len(f2p),
            "p2p_pass": 0,
            "p2p_total": len(p2p),
            "timeout": False,
            "error": "no_patch",
            "elapsed_s": 0.0,
            "output_tail": "",
            "resolved_by": "no_patch",
        }

    if grader_kind != "smith":
        policy = analyze_patch_policy(model_patch, swe_problem)
        if policy["violated"]:
            psrl_logger.info(f"{log_prefix} Patch policy violated: {policy['reasons']!r}, skipping eval container.")
            return _policy_blocked_result(policy, swe_problem, time.monotonic() - t0)

    # --- 1. Build eval script before spawning container (may raise) ---
    import json

    eval_script: str = ""
    try:
        if grader_kind == "gym":
            eval_script = _get_gym_eval_script(swe_problem)
        elif grader_kind == "smith":
            eval_script = _get_smith_eval_script(swe_problem)
        else:
            swe_problem_json = json.dumps(swe_problem, default=str)
            eval_script = _get_verified_eval_script(swe_problem_id, swe_problem_json)
    except Exception as exc:
        raise SWEGradingError(f"Failed to build eval script for {swe_problem_id!r}: {exc}.") from exc

    # --- 2. Spawn fresh eval container ---
    grader_label = (
        f"psrl.grader_task_id={swe_task_id}__eval" if swe_task_id else f"psrl.grader_task_id={swe_problem_id}__eval"
    )
    defaults = MiniEnvironmentConfig()
    baseline_run_args = list(defaults.run_args if run_args is None else run_args)
    run_args = list(baseline_run_args)
    env = defaults.env if env is None else env
    run_args = [a for a in run_args if not a.startswith(("--memory=", "--shm-size="))]
    run_args.append(f"--memory={memory or defaults.grader_memory}")
    run_args.append(f"--shm-size={shm_size or defaults.grader_shm_size}")
    run_args += ["--label", grader_label]
    docker_env: DockerEnvironment | None = None
    apply_ok = False
    eval_output = ""
    eval_returncode = -1
    error_msg: str | None = None
    grade: dict[str, Any] | None = None
    baseline_image_id: str | None = None
    evaluation_started = False

    try:
        psrl_logger.info(f"{log_prefix} Spawning eval container: image={image_name!r}, grader_kind={grader_kind!r}.")
        docker_env = DockerEnvironment(
            image=image_name,
            cwd="/testbed",
            run_args=run_args,
            forward_env=_FORWARDED_ENV_KEYS,
            env=defaults.env if env is None else env,
            container_timeout=defaults.container_timeout,
            timeout=command_timeout,
        )
        psrl_logger.info(f"{log_prefix} Eval container started: id={docker_env.container_id!r}.")

        # --- 3. Initialize the grading revision ---
        if grader_kind == "smith":
            initialize_smith_task(docker_env, swe_problem, restore_tests=True)
            eval_test_files = smith_evaluation_test_files(docker_env, swe_problem)
        else:
            initialize_benchmark_task(docker_env, swe_problem)

        # --- 4. Stage grading artifacts and apply the model patch ---
        _stage_grading_inputs(docker_env, model_patch, eval_script, timeout=command_timeout)
        out2 = docker_env.execute({"command": "git apply /tmp/psrl-grading/model.patch"}, cwd="/testbed")
        if out2.get("exception_info"):
            if out2.get("extra", {}).get("exception_type") in ("TimeoutExpired", "TimeoutError"):
                raise SWEGradingPreparationTimeout(f"SWE patch application timed out: {out2['exception_info']}.")
            raise SWEGradingError(f"SWE patch application failed to execute: {out2['exception_info']}.")
        apply_ok = out2["returncode"] == 0
        if not apply_ok:
            psrl_logger.info(f"{log_prefix} git apply failed (rc={out2['returncode']}): {out2['output'][:300]}.")

        # --- 5. SWE-smith: revert test-file changes from the patch ---
        if grader_kind == "smith" and apply_ok:
            if eval_test_files:
                files_str = " ".join(map(shlex.quote, eval_test_files))
                execute_setup_command(docker_env, f"git checkout -- {files_str}")

        # --- 6. Run eval script ---
        if apply_ok:
            if grader_kind != "smith":
                execute_setup_command(docker_env, "python /tmp/psrl-grading/syntax_probe.py capture")
            eval_cmd = "bash /tmp/psrl-grading/eval.sh"
            psrl_logger.info(f"{log_prefix} Running eval script (timeout={timeout}s)...")

            # Pass a per-command override because the environment timeout covers the container.
            evaluation_started = True
            try:
                out_eval = docker_env.execute(
                    {"command": eval_cmd},
                    cwd="/testbed",
                    timeout=timeout,
                )
                eval_output = out_eval.get("output", "")
                eval_returncode = out_eval.get("returncode", -1)
                if out_eval.get("exception_info"):
                    raise SWEGradingError(f"SWE test execution failed: {out_eval['exception_info']}.")
            except (TimeoutError, subprocess.TimeoutExpired) as exc:
                output = getattr(exc, "output", "") or ""
                stderr = getattr(exc, "stderr", "") or ""
                eval_output = (output.decode(errors="replace") if isinstance(output, bytes) else output) + (
                    stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
                )
                raise SWEGradingError(f"SWE test execution timed out: {exc}.") from exc
            except SWEGradingError:
                raise
            except Exception as exc:
                raise SWEGradingError(f"SWE test execution failed: {exc}.") from exc

        if grader_kind == "smith":
            grade = _grade_smith(swe_problem, model_patch, eval_output, apply_ok, eval_returncode)
        else:
            try:
                grade = _grade_benchmark(swe_problem, eval_output, apply_ok, grader_kind, model_patch)
            except SWEGradingError:
                if model_collection_failure_candidate(eval_output, model_patch) is None:
                    raise
                baseline_image_id = _container_image_id(docker_env)

    except SWETaskSetupTimeout as exc:
        _save_failure_artifacts(exc, swe_problem, model_patch, eval_script, eval_output, image_name, swe_task_id)
        raise SWEGradingPreparationTimeout(f"SWE task preparation timed out: {exc}.") from exc
    except (SWETaskSetupError, SWEGradingError) as exc:
        _save_failure_artifacts(exc, swe_problem, model_patch, eval_script, eval_output, image_name, swe_task_id)
        raise
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        _save_failure_artifacts(exc, swe_problem, model_patch, eval_script, eval_output, image_name, swe_task_id)
        if not evaluation_started:
            raise SWEGradingPreparationTimeout(f"SWE grading preparation timed out: {exc}.") from exc
        raise SWEGradingError(f"SWE test execution timed out: {exc}.") from exc
    except Exception as exc:
        _save_failure_artifacts(exc, swe_problem, model_patch, eval_script, eval_output, image_name, swe_task_id)
        raise SWEGradingError(f"SWE grading environment failed for {swe_problem_id!r}: {exc}.") from exc
    finally:
        cleanup_errors = []
        if docker_env is not None:
            try:
                docker_env.cleanup()
            except Exception as cleanup_exc:
                cleanup_errors.append(str(cleanup_exc))
                psrl_logger.warning(f"{log_prefix} Container cleanup failed: {cleanup_exc}.")
        # Sweep by label because asynchronous cleanup may return before removal.
        try:
            force_remove_containers_by_label("psrl.grader_task_id", grader_label.split("=", 1)[1])
        except Exception as sweep_exc:
            cleanup_errors.append(str(sweep_exc))
            psrl_logger.warning(f"{log_prefix} Label sweep failed: {sweep_exc}.")
        if cleanup_errors:
            raise SWEGradingError(f"SWE grading cleanup failed: {cleanup_errors!r}.")

    if grade is None:
        assert baseline_image_id is not None, "Unproven collection failures require the evaluated image identity."
        try:
            baseline_report = _get_baseline_report(
                swe_problem,
                grader_kind,
                baseline_image_id,
                timeout,
                swe_task_id,
                memory,
                baseline_run_args,
                env,
                shm_size,
            )
        except SWEGradingError as exc:
            _save_failure_artifacts(exc, swe_problem, model_patch, eval_script, eval_output, image_name, swe_task_id)
            raise
        except Exception as exc:
            _save_failure_artifacts(exc, swe_problem, model_patch, eval_script, eval_output, image_name, swe_task_id)
            raise SWEGradingError(f"Baseline verification failed for {swe_problem_id!r}: {exc}.") from exc
        grade = _grade_benchmark(
            swe_problem, eval_output, apply_ok, grader_kind, model_patch, baseline_report=baseline_report
        )
    elapsed = time.monotonic() - t0
    output_tail = eval_output[-_OUTPUT_TAIL_BYTES:] if eval_output else ""

    # Merge grading result with meta fields.
    result: dict[str, Any] = {
        "policy_violated": False,
        "policy_reasons": [],
        "apply_ok": apply_ok,
        "timeout": False,
        "error": error_msg,
        "elapsed_s": round(elapsed, 2),
        "output_tail": output_tail,
        **grade,
    }

    if failure := grade.get("model_failure"):
        psrl_logger.info(
            f"{log_prefix} Attributed model collection failure: error_type={failure['error_type']!r}, "
            f"path={failure['path']!r}, line={failure['line']}, evidence={failure['attribution']!r}."
        )

    psrl_logger.info(
        f"{log_prefix} Grading complete: resolved={result['resolved']}, "
        f"apply_ok={apply_ok}, f2p={result['f2p_pass']}/{result['f2p_total']}, "
        f"p2p={result['p2p_pass']}/{result['p2p_total']}, "
        f"elapsed={elapsed:.1f}s, resolved_by={result['resolved_by']!r}."
    )
    return result
