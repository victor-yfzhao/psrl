"""
Apply the same patch policy and SWE test report protocol in both training integrations.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from examples.mini_swe.syntax_probe import SYNTAX_REGRESSION_MARKER

TEST_OUTPUT_START = ">>>>> Start Test Output"
TEST_OUTPUT_END = ">>>>> End Test Output"
_EXECUTION_FAILURE_PATTERN = re.compile(r">>>>> (?:Tests Errored|Tests Timed Out|Patch Apply Failed|Reset Failed)")
_INFRASTRUCTURE_FAILURE_PATTERN = re.compile(
    r"No matching distribution found|Could not find a version that satisfies|Could not build wheels|"
    r"Failed building wheel|subprocess-exited-with-error|(?:Transport|Connection|Network|Timeout)Error|"
    r"Connection refused|Connection reset|Could not resolve host|Network is unreachable|TimeoutExpired|timed out|"
    r">>>>> (?:Tests Errored|Tests Timed Out|Patch Apply Failed|Reset Failed)|Could not install packages due to|"
    r"Failed to install (?:packages|requirements|dependencies)|ResolutionImpossible",
    re.IGNORECASE | re.MULTILINE,
)


def frame_eval_script(script: str, *, postlude: str = "") -> str:
    """
    Mark completed evaluation output while preserving failing test reports.
    """
    if not isinstance(script, str) or not script.strip():
        raise ValueError("An evaluation script is required.")
    return f"printf '%s\\n' '{TEST_OUTPUT_START}'\n{script}\n{postlude}\nprintf '%s\\n' '{TEST_OUTPUT_END}'\n"


def configure_gym_test_runtime(script: str, repo: str) -> str:
    """
    Preserve prepared dependencies and configure the benchmark's external test fixtures.
    """
    script = script.replace("pdm add pre-commit", "pdm add --update-reuse-installed pre-commit", 1)
    setup = []
    if repo == "iterative/dvc" and "tests/func/test_data_cloud.py" in script:
        # DVC's host-sized client pool must fit the local WebDAV server's connection backlog.
        setup.extend(
            [
                "import socketserver",
                "socketserver.TCPServer.request_queue_size = max(1024, socketserver.TCPServer.request_queue_size)",
            ]
        )
    if repo == "pandas-dev/pandas" and "pandas/tests/io/xml/test_xml.py" in script:
        # The benchmark's W3Schools fixtures reject urllib's default user agent.
        setup.extend(
            [
                "import urllib.request",
                "opener = urllib.request.build_opener()",
                "opener.addheaders = [('User-Agent', 'Mozilla/5.0')]",
                "urllib.request.install_opener(opener)",
            ]
        )
    if setup:
        runner = "; ".join(["import sys", *setup, "import pytest", "sys.exit(pytest.main(sys.argv[1:]))"])
        script = re.sub(r"(?m)^pytest(?=\s)", lambda _: f"python -c {shlex.quote(runner)}", script)
    return script


def _repository_source_path(path: str) -> str | None:
    """
    Resolve traceback paths within the task checkout.
    """
    if path.startswith("/testbed/"):
        path = path.removeprefix("/testbed/")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or "site-packages" in parsed.parts:
        return None
    return str(parsed)


def _repository_python_path(path: str) -> str | None:
    """
    Resolve Python traceback paths within the task checkout.
    """
    resolved = _repository_source_path(path)
    return resolved if resolved is not None and PurePosixPath(resolved).suffix == ".py" else None


def _git_patch_paths(
    model_patch: str,
    *,
    reverse: bool = False,
    timeout: float = 120,
) -> list[str]:
    """
    Resolve patch filenames with the same prefix and quoting rules as `git apply`.
    """
    output = subprocess.check_output(
        ["git", "apply", "--numstat", "-z", *(["--reverse"] if reverse else []), "-"],
        input=model_patch.encode("utf-8"),
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return [os.fsdecode(record.split(b"\t", 2)[2]) for record in output.split(b"\0") if record]


def _patch_source_changes(
    model_patch: str,
    *,
    timeout: float = 120,
) -> dict[str, dict[str, Any]]:
    """
    Identify modified files and their original task source paths.
    """
    from unidiff import PatchSet, UnidiffParseError

    try:
        patch = PatchSet(model_patch)
        if not patch:
            return {}
        paths = []
        for entry in patch:
            file_patch = str(entry)
            [target] = _git_patch_paths(file_patch, timeout=timeout)
            has_source_metadata = any(
                line.startswith(("rename from ", "copy from ")) for line in entry.patch_info or []
            )
            [source] = _git_patch_paths(file_patch, reverse=True, timeout=timeout) if has_source_metadata else [target]
            paths.append((target, source, has_source_metadata))
    except (UnidiffParseError, ValueError, subprocess.CalledProcessError):
        return {}
    changes = {}
    patch_lines = model_patch.splitlines()
    for patched_file, (target, source, has_source_metadata) in zip(patch, paths, strict=True):
        metadata = patched_file.patch_info or []
        source_is_null = patched_file.source_file == "/dev/null"
        target_is_null = patched_file.target_file == "/dev/null"
        if patched_file.patch_info:
            # `unidiff` can replace an explicit null header with its earlier Git filename.
            for offset, line in enumerate(patch_lines[patched_file.diff_line_no - 1 :]):
                if line.startswith("@@") or (offset and line.startswith("diff --git ")):
                    break
                source_is_null |= re.fullmatch(r"--- /dev/null(?:\t.*)?", line) is not None
                target_is_null |= re.fullmatch(r"\+\+\+ /dev/null(?:\t.*)?", line) is not None
        deleted = target_is_null or any(line.startswith("deleted file mode ") for line in metadata)
        path = _repository_source_path(target)
        if deleted or path is None:
            continue
        created = source_is_null or any(line.startswith("new file mode ") for line in metadata)
        if not (patched_file.added or patched_file.removed or created or has_source_metadata):
            continue
        changes[path] = {
            "source_path": None if created else _repository_source_path(source if has_source_metadata else target)
        }
    return changes


def _patch_python_changes(
    model_patch: str,
    *,
    timeout: float = 120,
) -> dict[str, dict[str, Any]]:
    """
    Select modified Python sources for interpreter based syntax checks.
    """
    return {
        path: change
        for path, change in _patch_source_changes(model_patch, timeout=timeout).items()
        if path.endswith(".py")
    }


def syntax_probe_manifest(
    model_patch: str,
    *,
    timeout: float = 120,
) -> dict[str, Any]:
    """
    Identify source blobs needed to prove model induced syntax regressions.
    """
    return {
        "model_patch_sha256": hashlib.sha256(model_patch.encode("utf-8")).hexdigest(),
        "files": [
            {"path": path, "source_path": change["source_path"]}
            for path, change in _patch_python_changes(model_patch, timeout=timeout).items()
        ],
    }


def _framed_test_output(output: str) -> str | None:
    """
    Extract only a completed evaluation report.
    """
    if TEST_OUTPUT_START not in output:
        return None
    test_output = output.split(TEST_OUTPUT_START, 1)[1]
    if TEST_OUTPUT_END not in test_output:
        return None
    return test_output.rsplit(TEST_OUTPUT_END, 1)[0]


def _pytest_startup_offset(test_output: str) -> int | None:
    """
    Identify pytest configuration startup from its short diagnostic or bootstrap traceback.
    """
    short_header = test_output.find("ImportError while loading conftest")
    if short_header >= 0:
        return short_header
    tracebacks = list(re.finditer(r"(?m)^[ \t]*(?:E[ \t]+)?Traceback \(most recent call last\):", test_output))
    loader_frame = re.compile(
        r'^[ \t]*(?:E[ \t]+)?File "[^"\n]*/_pytest/config(?:/[^"\n]*)?\.py", '
        r"line \d+, in (?:_prepareconfig|_importconftest|_loadconftestmodules)\b",
        re.MULTILINE,
    )
    conftest_failure = re.compile(r"^[ \t]*(?:E[ \t]+)?(?:_pytest\.config\.)?ConftestImportFailure:", re.MULTILINE)
    for index, traceback in enumerate(tracebacks):
        end = tracebacks[index + 1].start() if index + 1 < len(tracebacks) else len(test_output)
        segment = test_output[traceback.start() : end]
        if loader_frame.search(segment) or conftest_failure.search(segment):
            return traceback.start()
    return None


def _cython_startup_failure_candidate(
    test_output: str,
    startup: int,
    error_type: str,
    message: str,
    changes: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Connect an explicit changed Cython source diagnostic to a failed pytest import build.
    """
    command_error = re.fullmatch(r"Command '(\[.*\])' returned non-zero exit status [1-9]\d*\.", message)
    if error_type != "CalledProcessError" or command_error is None:
        return None
    try:
        command = ast.literal_eval(command_error.group(1))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(command, list) or not command or not isinstance(command[0], str):
        return None
    if PurePosixPath(command[0]).name != "ninja":
        return None
    build_output = test_output[:startup]
    failed_builds = list(re.finditer(r"(?m)^ninja: build stopped: subcommand failed\.\s*$", build_output))
    if not failed_builds:
        return None
    compilation_output = build_output[: failed_builds[-1].start()]
    compiler_headers = list(re.finditer(r"(?m)^Error compiling Cython file:[ \t]*$", compilation_output))
    diagnostic = re.compile(r"(?m)^([^\n:]+\.pyx):([1-9]\d*):(\d+):[ \t]*(.+)$")
    for index, header in enumerate(compiler_headers):
        end = compiler_headers[index + 1].start() if index + 1 < len(compiler_headers) else len(compilation_output)
        if _INFRASTRUCTURE_FAILURE_PATTERN.search(build_output[header.start() :]):
            continue
        for match in diagnostic.finditer(compilation_output[header.end() : end]):
            path = _repository_source_path(match.group(1))
            if path not in changes:
                continue
            return {
                "error_type": "CythonCompileError",
                "message": match.group(4),
                "path": path,
                "line": int(match.group(2)),
                "column": int(match.group(3)),
            }
    return None


def _pytest_empty_collection_offset(test_output: str) -> int | None:
    """
    Require a completed pytest session that explicitly collected no tests.
    """
    sessions = list(re.finditer(r"(?m)^=+ test session starts =+[ \t\r]*$", test_output))
    if not sessions:
        return None
    start = sessions[-1].start()
    session_output = test_output[start:]
    collected = re.search(r"(?m)^collected 0 items[ \t\r]*$", session_output)
    summary = re.search(r"(?m)^=+ no tests ran in \d+(?:\.\d+)?s(?: \([^\n]+\))? =+[ \t\r]*$", session_output)
    if collected is None or summary is None or collected.end() > summary.start():
        return None
    return start


def model_collection_failure_candidate(output: str, model_patch: str) -> dict[str, Any] | None:
    """
    Identify an unproven pytest collection failure after a model source change.
    """
    test_output = _framed_test_output(output)
    if test_output is None or _EXECUTION_FAILURE_PATTERN.search(output):
        return None
    startup = _pytest_startup_offset(test_output)
    if startup is None:
        empty_collection = _pytest_empty_collection_offset(test_output)
        if empty_collection is None or _INFRASTRUCTURE_FAILURE_PATTERN.search(test_output[empty_collection:]):
            return None
        modified_sources = sorted(_patch_python_changes(model_patch))
        if not modified_sources:
            return None
        return {
            "error_type": "NoTestsCollected",
            "message": "Pytest collected no tests.",
            "path": None,
            "line": None,
            "model_patch_sha256": hashlib.sha256(model_patch.encode("utf-8")).hexdigest(),
            "modified_sources": modified_sources,
        }
    startup_output = test_output[startup:]
    if _INFRASTRUCTURE_FAILURE_PATTERN.search(startup_output):
        return None
    exceptions = [
        match
        for match in re.finditer(
            r"^\s*(?:E\s+)?(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*):\s*(.+)$",
            startup_output,
            re.MULTILINE,
        )
        if match.group(1) != "ConftestImportFailure"
    ]
    if not exceptions:
        return None
    exception = exceptions[-1]
    error_type, message = exception.groups()
    changes = _patch_source_changes(model_patch)
    if candidate := _cython_startup_failure_candidate(test_output, startup, error_type, message, changes):
        return {
            **candidate,
            "model_patch_sha256": hashlib.sha256(model_patch.encode("utf-8")).hexdigest(),
            "modified_sources": sorted(changes),
        }
    frame_pattern = re.compile(
        r'^\s*(?:E\s+)?(?:File "(?P<quoted>[^"\n]+)", line (?P<number>\d+)'
        r"|(?P<plain>[^\s:]+\.py):(?P<plain_number>\d+): in [^\n]+)",
        re.MULTILINE,
    )
    frames = list(frame_pattern.finditer(startup_output[: exception.start()]))
    if frames:
        terminal = frames[-1]
        path = _repository_python_path(terminal.group("quoted") or terminal.group("plain"))
        line = int(terminal.group("number") or terminal.group("plain_number"))
    else:
        conftest = re.match(
            r"ImportError while loading conftest ['\"]([^'\"]+)['\"]\.[ \t]*(?:\r?\n|$)", startup_output
        )
        path = _repository_python_path(conftest.group(1)) if conftest is not None else None
        line = None
    modified_sources = sorted(source for source in changes if source.endswith(".py"))
    if path is None or not modified_sources:
        return None
    return {
        "error_type": error_type,
        "message": message,
        "path": path,
        "line": line,
        "model_patch_sha256": hashlib.sha256(model_patch.encode("utf-8")).hexdigest(),
        "modified_sources": modified_sources,
    }


def baseline_report_is_complete(task: dict[str, Any], report: dict[str, Any]) -> bool:
    """
    Require observed official results for every unique task test before proving readiness.
    """
    expected = set([*task["FAIL_TO_PASS"], *task["PASS_TO_PASS"]])
    observed = report.get("test_results_observed")
    return (
        bool(expected)
        and isinstance(observed, int)
        and not isinstance(observed, bool)
        and observed == len(expected)
        and report.get("resolved_by") == "harness"
        and not report.get("model_failure")
    )


def _model_collection_failure(
    output: str,
    model_patch: str,
    baseline_report: dict[str, Any] | None,
    task: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Require compile evidence or a complete task baseline before assigning a model collection failure.
    """
    candidate = model_collection_failure_candidate(output, model_patch)
    if candidate is None:
        return None
    test_output = _framed_test_output(output)
    if candidate["error_type"] in {"SyntaxError", "IndentationError", "TabError"}:
        for line in test_output.splitlines():
            if not line.startswith(SYNTAX_REGRESSION_MARKER):
                continue
            try:
                evidence = json.loads(line.removeprefix(SYNTAX_REGRESSION_MARKER))
            except (ValueError, TypeError):
                continue
            if (
                isinstance(evidence, dict)
                and candidate["path"] in candidate["modified_sources"]
                and all(
                    isinstance(evidence.get(key), str) and evidence[key]
                    for key in ["revision", "before_sha256", "after_sha256", "python_executable", "python_version"]
                )
                and all(
                    evidence.get(key) == candidate[key]
                    for key in ["path", "error_type", "line", "message", "model_patch_sha256"]
                )
                and evidence["before_sha256"] != evidence["after_sha256"]
            ):
                return {
                    **evidence,
                    "modified_sources": candidate["modified_sources"],
                    "attribution": "syntax_regression",
                }
    if (
        baseline_report is not None
        and baseline_report_is_complete(task, baseline_report)
        and baseline_report.get("baseline_image_id")
    ):
        return {
            **candidate,
            "attribution": "baseline_report",
            "baseline_image_id": baseline_report["baseline_image_id"],
            "baseline_test_results_observed": baseline_report["test_results_observed"],
        }
    return None


def parse_benchmark_report(
    task: dict[str, Any],
    output: str,
    kind: str,
    *,
    model_patch: str = "",
    baseline_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Parse SWE-Gym or SWE-bench output with official parsers and resolution rules.
    """
    from swebench.harness.constants import KEY_INSTANCE_ID
    from swebench.harness.grading import EvalType, ResolvedStatus, get_eval_tests_report, get_resolution_status
    from swebench.harness.log_parsers.python import parse_log_pytest_v2

    if _EXECUTION_FAILURE_PATTERN.search(output):
        raise ValueError(f"No parseable task test results for {task['instance_id']!r}: Evaluation execution failed.")
    test_output = _framed_test_output(output)
    if test_output is None:
        raise ValueError(f"Missing or incomplete test output markers for {task['instance_id']!r}.")
    if kind == "gym":
        statuses = parse_log_pytest_v2(test_output, None)
    elif kind == "verified":
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
        from swebench.harness.test_spec.test_spec import make_test_spec

        spec = make_test_spec(task)
        statuses = MAP_REPO_TO_PARSER[task["repo"]](test_output, spec)
    else:
        raise ValueError(f"Unsupported benchmark grader {kind!r}.")
    f2p = task["FAIL_TO_PASS"]
    p2p = task["PASS_TO_PASS"]
    if not f2p:
        raise ValueError(f"Task {task['instance_id']!r} has no FAIL_TO_PASS tests.")
    collection_failed = "ERROR collecting" in test_output or ("Ran " in test_output and "FAILED (" in test_output)
    model_failure = None
    observed = len(set(statuses).intersection([*f2p, *p2p]))
    if not observed and not collection_failed:
        model_failure = _model_collection_failure(output, model_patch, baseline_report, task)
        if model_failure is None:
            raise ValueError(f"No parseable task test results for {task['instance_id']!r}.")
    report = get_eval_tests_report(
        statuses,
        {KEY_INSTANCE_ID: task["instance_id"], "FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p},
        eval_type=EvalType.PASS_AND_FAIL,
    )
    result = {
        "resolved": get_resolution_status(report) == ResolvedStatus.FULL.value,
        "f2p_pass": len(report["FAIL_TO_PASS"]["success"]),
        "f2p_total": len(f2p),
        "p2p_pass": len(report["PASS_TO_PASS"]["success"]),
        "p2p_total": len(p2p),
        "resolved_by": "harness",
        "test_results_observed": observed,
    }
    if model_failure is not None:
        result["model_failure"] = model_failure
    return result


def _changed_files_from_patch(patch_text: str) -> list[str]:
    """
    Extract the list of changed file paths from a unified git diff.

    Args:
        patch_text (str): Unified diff produced by ``git diff``.

    Returns:
        list[str]: Sorted list of changed file paths.
    """
    files: set[str] = set()
    for line in patch_text.splitlines():
        m = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if m:
            files.add(m.group(2))
    return sorted(files)


def _is_test_like_path(path: str) -> bool:
    """
    Return True if a path looks like a test file or test directory.

    Args:
        path (str): File path relative to the repo root.

    Returns:
        bool: True when the path appears to be a test artifact.
    """
    lower = path.lower()
    parts = lower.split("/")
    return (
        any(part in {"tests", "test", "specs", "spec"} for part in parts)
        or Path(lower).stem.startswith("test_")
        or Path(lower).stem.endswith("_test")
        or Path(lower).stem.startswith("spec_")
        or Path(lower).stem.endswith("_spec")
    )


def _is_config_like_path(path: str) -> bool:
    """
    Return True if a path looks like a project configuration file.

    Args:
        path (str): File path relative to the repo root.

    Returns:
        bool: True when the path appears to be a configuration artifact.
    """
    lower = path.lower()
    name = Path(lower).name
    return name in {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "noxfile.py",
        "makefile",
        "dockerfile",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        ".pre-commit-config.yaml",
        "conftest.py",
    }


def _extract_eval_test_files(swe_problem: dict[str, Any]) -> list[str]:
    """
    Extract F2P + P2P test file paths from a SWE problem dict.

    Args:
        swe_problem (dict[str, Any]): Dataset row with FAIL_TO_PASS / PASS_TO_PASS.

    Returns:
        list[str]: Sorted list of unique test-file paths for the eval tests.
    """
    files: set[str] = set()
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        for test_id in swe_problem.get(key, []):
            # test_id format: "path/to/test_file.py::ClassName::method"
            file_part = test_id.split("::")[0]
            if file_part:
                files.add(file_part)
    return sorted(files)


def analyze_patch_policy(
    patch_text: str,
    swe_problem: dict[str, Any],
    *,
    eval_test_files: list[str] | None = None,
) -> dict[str, Any]:
    """
    Check whether a model patch violates submission policy.

    Policy rules (configurable via env vars, all enabled by default):

    - ``SWE_STRICT_NO_TEST_PATCH=1``: disallow changes to test files.
    - ``SWE_STRICT_NO_CONFIG_PATCH=1``: disallow changes to config files.
    - ``SWE_TEST_PATCH_POLICY_SCOPE=eval_tests_only``: when enforcing the
      test-file rule, only flag files that appear in the FAIL_TO_PASS /
      PASS_TO_PASS lists. Set to ``all_tests`` to flag any test-like path.

    Args:
        patch_text (str): Unified diff produced by the agent.
        swe_problem (dict[str, Any]): Dataset row for the SWE problem
            (needs FAIL_TO_PASS, PASS_TO_PASS).
        eval_test_files: Authoritative repository paths when supplied by the task revision.

    Returns:
        dict[str, Any]: Policy analysis result containing at least
            ``violated`` (bool) and ``reasons`` (list[str]).
    """
    strict_no_test = os.getenv("SWE_STRICT_NO_TEST_PATCH", "1").strip() != "0"
    strict_no_config = os.getenv("SWE_STRICT_NO_CONFIG_PATCH", "1").strip() != "0"
    scope = os.getenv("SWE_TEST_PATCH_POLICY_SCOPE", "eval_tests_only").strip().lower()
    if scope not in {"all_tests", "eval_tests_only"}:
        scope = "eval_tests_only"

    changed_files = _changed_files_from_patch(patch_text)
    test_files = [f for f in changed_files if _is_test_like_path(f)]
    config_files = [f for f in changed_files if _is_config_like_path(f)]
    if eval_test_files is None:
        eval_test_files = _extract_eval_test_files(swe_problem)
    eval_test_file_set = set(eval_test_files)
    matched_eval_test_files = [f for f in changed_files if f in eval_test_file_set]
    test_files = sorted(set(test_files) | set(matched_eval_test_files))

    reasons: list[str] = []
    if strict_no_test:
        if scope == "all_tests" and test_files:
            reasons.append("test_file_modified")
        elif scope == "eval_tests_only" and matched_eval_test_files:
            reasons.append("eval_test_file_modified")
    if strict_no_config and config_files:
        reasons.append("config_file_modified")

    return {
        "violated": len(reasons) > 0,
        "reasons": reasons,
        "changed_files": changed_files,
        "test_files": test_files,
        "config_files": config_files,
        "eval_test_files": eval_test_files,
        "matched_eval_test_files": matched_eval_test_files,
        "test_policy_scope": scope,
    }
