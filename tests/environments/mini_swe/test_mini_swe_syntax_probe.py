"""
Prove syntax regressions without importing model code or changing the task checkout.
"""

import base64
import json
import os
import subprocess
import sys
from difflib import unified_diff
from pathlib import Path

import pytest
from examples.mini_swe import syntax_probe
from examples.mini_swe.benchmark_grading import (
    TEST_OUTPUT_END,
    TEST_OUTPUT_START,
    parse_benchmark_report,
    syntax_probe_manifest,
)
from unidiff import PatchSet

pytestmark = pytest.mark.cpu_test

BEFORE = "def repair(\n    first=1,\n    second=2,\n    third=3,\n):\n    return first\n"
AFTER = BEFORE.replace("def repair(\n", "values = [\n")


def patch(before: str, after: str, path: str = "package/model.py") -> str:
    """
    Build a patch whose source and target line numbers match the fixture.
    """
    return f"diff --git a/{path} b/{path}\n" + "".join(
        unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{path}", f"b/{path}")
    )


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    path = tmp_path / "package/model.py"
    path.parent.mkdir()
    path.write_text(BEFORE)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "Fixture."],
        check=True,
    )
    path.write_text(AFTER)
    directory = tmp_path / "probe"
    directory.mkdir()
    model_patch = patch(BEFORE, AFTER)
    (directory / "syntax_files.json").write_text(json.dumps(syntax_probe_manifest(model_patch)))
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "pytest").write_text(f"#!{sys.executable}\n")
    (binary / "pytest").chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary}{os.pathsep}{os.environ['PATH']}")
    return directory, path, model_patch


@pytest.mark.parametrize("setup_output", ["", "ERROR: No matching distribution found for an unused extra\n"])
def test_syntax_error_on_unchanged_line_is_attributed_with_compile_proof(checkout, capsys, setup_output):
    directory, path, model_patch = checkout
    syntax_probe.capture_sources(directory)
    syntax_probe.check_sources(directory)
    proof = capsys.readouterr().out
    evidence = json.loads(proof.removeprefix(syntax_probe.SYNTAX_REGRESSION_MARKER))

    assert evidence["line"] == 5, "The syntax error occurs after the changed opening line."
    assert evidence["python_executable"] == sys.executable, "Compile using the active pytest interpreter."
    assert evidence["before_sha256"] != evidence["after_sha256"], "Proof must identify distinct source bytes."
    assert path.read_text() == AFTER, "Syntax validation must not alter model source."
    text = (
        f"{TEST_OUTPUT_START}\n{setup_output}ImportError while loading conftest '/testbed/tests/conftest.py'.\n"
        f'E     File "/testbed/package/model.py", line 5\nE   SyntaxError: {evidence["message"]}\n'
        f"{proof}{TEST_OUTPUT_END}\n"
    )
    task = {"instance_id": "example__repo-1", "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"], "PASS_TO_PASS": []}
    result = parse_benchmark_report(task, text, "gym", model_patch=model_patch)

    assert not result["resolved"], "A proven model syntax regression must remain unresolved."
    assert result["model_failure"]["attribution"] == "syntax_regression", "Retain compile based causal evidence."


@pytest.mark.parametrize("failure", ["baseline", "rewritten", "interpreter"])
def test_unproven_syntax_failure_emits_no_evidence(checkout, failure, capsys):
    directory, path, _ = checkout
    if failure == "baseline":
        path.write_text("baseline = [\n")
        subprocess.run(["git", "add", "package/model.py"], check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "Fixture."],
            check=True,
        )
        path.write_text(AFTER)
    syntax_probe.capture_sources(directory)
    if failure == "rewritten":
        path.write_text("other_failure = (\n")
    elif failure == "interpreter":
        (directory.parent / "bin/pytest").write_text("#!/missing/python\n")
    syntax_probe.check_sources(directory)

    assert not capsys.readouterr().out, "Unproven failures must not produce model attribution evidence."


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", "package/other.py"),
        ("line", 99),
        ("error_type", "IndentationError"),
        ("message", "A different syntax error."),
        ("model_patch_sha256", "0" * 64),
        ("python_executable", ""),
    ],
)
def test_compile_evidence_must_match_the_evaluated_patch_and_terminal_error(checkout, capsys, field, value):
    directory, _, model_patch = checkout
    syntax_probe.capture_sources(directory)
    syntax_probe.check_sources(directory)
    evidence = json.loads(capsys.readouterr().out.removeprefix(syntax_probe.SYNTAX_REGRESSION_MARKER))
    message = evidence["message"]
    evidence[field] = value
    proof = syntax_probe.SYNTAX_REGRESSION_MARKER + json.dumps(evidence)
    text = (
        f"{TEST_OUTPUT_START}\nImportError while loading conftest '/testbed/tests/conftest.py'.\n"
        f'E     File "/testbed/package/model.py", line 5\nE   SyntaxError: {message}\n'
        f"{proof}\n{TEST_OUTPUT_END}\n"
    )
    task = {"instance_id": "example__repo-1", "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"], "PASS_TO_PASS": []}
    with pytest.raises(ValueError, match="No parseable"):
        parse_benchmark_report(task, text, "gym", model_patch=model_patch)


def test_probe_handles_new_python_file_without_a_baseline_blob(checkout, capsys):
    directory, path, _ = checkout
    new_path = path.with_name("new.py")
    new_path.write_text("value = (\n")
    model_patch = patch("", new_path.read_text(), "package/new.py").replace("--- a/package/new.py", "--- /dev/null")
    (directory / "syntax_files.json").write_text(json.dumps(syntax_probe_manifest(model_patch)))
    syntax_probe.capture_sources(directory)
    syntax_probe.check_sources(directory)
    evidence = json.loads(capsys.readouterr().out.removeprefix(syntax_probe.SYNTAX_REGRESSION_MARKER))

    assert evidence["path"] == "package/new.py", "New files have an empty baseline source."


def test_syntax_probe_preserves_valid_source_and_never_executes_it(checkout, capsys):
    directory, path, _ = checkout
    path.write_text("raise RuntimeError('Never execute model code.')\n")
    syntax_probe.capture_sources(directory)
    syntax_probe.check_sources(directory)

    assert not capsys.readouterr().out, "Valid Python source must not produce syntax regression evidence."
    assert not list(Path.cwd().rglob("*.pyc")), "Compiling source must not create import cache files."


@pytest.mark.parametrize(
    "source,target,applied_path",
    [
        ("a/package/model.py", "b/package/model.py", "package/model.py"),
        ("a/orig_array.py", "b/package/model.py", "package/model.py"),
        ("orig_array.py", "package/model.py", "model.py"),
    ],
)
def test_probe_uses_the_path_git_apply_modified(checkout, capsys, source, target, applied_path):
    directory, original_path, _ = checkout
    original_path.write_text(BEFORE)
    path = directory.parent / applied_path
    if path != original_path:
        path.write_text(BEFORE)
        subprocess.run(["git", "add", applied_path], check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "Fixture."],
            check=True,
        )
    model_patch = "".join(unified_diff(BEFORE.splitlines(True), AFTER.splitlines(True), source, target))
    subprocess.run(["git", "apply", "-"], input=model_patch, text=True, check=True)
    manifest = syntax_probe_manifest(model_patch)

    assert manifest["files"] == [{"path": applied_path, "source_path": applied_path}], (
        "Capture the actual Git target and its task revision, regardless of source header aliases."
    )
    (directory / "syntax_files.json").write_text(json.dumps(manifest))
    syntax_probe.capture_sources(directory)
    syntax_probe.check_sources(directory)
    evidence = json.loads(capsys.readouterr().out.removeprefix(syntax_probe.SYNTAX_REGRESSION_MARKER))

    assert evidence["path"] == applied_path, "Syntax evidence must identify the path changed by git apply."
    assert path.read_text() == AFTER, "Capture must preserve the applied model source."


@pytest.mark.parametrize("operation", ["rename", "copy", "add", "delete"])
@pytest.mark.parametrize("target", ["package/new.py", "package/new café.py"])
def test_probe_tracks_git_rename_add_and_delete_targets(checkout, capsys, operation, target):
    directory, original_path, _ = checkout
    original_path.write_text(BEFORE)
    if operation in {"rename", "copy"}:
        if operation == "rename":
            subprocess.run(["git", "mv", "package/model.py", target], check=True)
        (directory.parent / target).write_text(AFTER)
        subprocess.run(["git", "add", target], check=True)
        model_patch = subprocess.check_output(
            ["git", "diff", "--find-copies", "--find-copies-harder", "HEAD"], text=True
        )
        if operation == "copy":
            assert any(
                line.startswith("copy from ") for entry in PatchSet(model_patch) for line in entry.patch_info or []
            ), "The Git fixture must declare a copy operation."
        subprocess.run(["git", "reset", "--hard", "-q", "HEAD"], check=True)
        expected = [{"path": target, "source_path": "package/model.py"}]
    elif operation == "add":
        model_patch = "".join(unified_diff([], AFTER.splitlines(True), "/dev/null", f"b/{target}"))
        expected = [{"path": target, "source_path": None}]
    else:
        model_patch = "".join(unified_diff(BEFORE.splitlines(True), [], "a/package/model.py", "/dev/null"))
        expected = []
    subprocess.run(["git", "apply", "-"], input=model_patch, text=True, check=True)
    manifest = syntax_probe_manifest(model_patch)

    assert manifest["files"] == expected, "Only surviving Python targets may be captured."
    (directory / "syntax_files.json").write_text(json.dumps(manifest))
    syntax_probe.capture_sources(directory)
    syntax_probe.check_sources(directory)
    output = capsys.readouterr().out
    if operation == "delete":
        assert not output, "Removing a file cannot supply unchanged source bytes for a syntax proof."
        assert not original_path.exists(), "The model deletion must remain applied."
    else:
        evidence = json.loads(output.removeprefix(syntax_probe.SYNTAX_REGRESSION_MARKER))
        assert evidence["path"] == target, "Renamed and added files must use their destination paths."


def test_probe_still_rejects_an_unexpected_missing_target(checkout):
    directory, path, _ = checkout
    path.unlink()

    with pytest.raises(FileNotFoundError):
        syntax_probe.capture_sources(directory)


@pytest.mark.parametrize("operation", ["new_rename", "new_copy", "delete_rename"])
@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("target", ["package/moved.py", "package/moved café.py"])
def test_probe_preserves_source_mapping_across_combined_git_operations(checkout, operation, reverse_order, target):
    directory, original_path, _ = checkout
    original_path.write_text(BEFORE)
    extra_path = directory.parent / "package/other.py"
    extra_before = "other = 1\n" if operation == "delete_rename" else ""
    extra_after = "new_source = [\n"
    if extra_before:
        extra_path.write_text(extra_before)
        subprocess.run(["git", "add", "package/other.py"], check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "Fixture."],
            check=True,
        )
        subprocess.run(["git", "rm", "-q", "package/other.py"], check=True)
    else:
        extra_path.write_text(extra_after)
        subprocess.run(["git", "add", "package/other.py"], check=True)
    if operation == "new_copy":
        (directory.parent / target).write_text(BEFORE)
        subprocess.run(["git", "add", target], check=True)
    else:
        subprocess.run(["git", "mv", "package/model.py", target], check=True)
    git_patch = subprocess.check_output(
        ["git", "diff", "--cached", "--binary", "--find-copies", "--find-copies-harder", "HEAD"], text=True
    )
    entries = list(PatchSet(git_patch))
    if operation == "new_copy":
        assert any(line.startswith("copy from ") for entry in entries for line in entry.patch_info or []), (
            "The combined Git fixture must declare a copy operation."
        )
    entries.sort(
        key=lambda entry: not any(line.startswith(("rename from ", "copy from ")) for line in entry.patch_info or []),
        reverse=reverse_order,
    )
    model_patch = "".join(map(str, entries))
    subprocess.run(["git", "reset", "--hard", "-q", "HEAD"], check=True)
    subprocess.run(["git", "apply", "-"], input=model_patch, text=True, check=True)
    manifest = syntax_probe_manifest(model_patch)
    expected = {target: "package/model.py"}
    if not extra_before:
        expected["package/other.py"] = None

    assert {entry["path"]: entry["source_path"] for entry in manifest["files"]} == expected, (
        "Each destination must retain its own Git source regardless of patch record order."
    )
    (directory / "syntax_files.json").write_text(json.dumps(manifest))
    syntax_probe.capture_sources(directory)
    sources = json.loads((directory / "syntax_sources.json").read_text())["sources"]
    for entry in sources:
        expected_before = BEFORE if entry["path"] == target else ""
        expected_after = BEFORE if entry["path"] == target else extra_after
        assert base64.b64decode(entry["before"]) == expected_before.encode(), (
            "Capture the baseline bytes from the source paired with this destination."
        )
        assert base64.b64decode(entry["after"]) == expected_after.encode(), (
            "Capture the applied bytes from this destination."
        )


@pytest.mark.parametrize(
    "operation", ["truncate", "fill_empty", "create_empty", "delete_empty", "rename_empty", "copy_empty"]
)
def test_probe_distinguishes_empty_files_from_declared_creation_and_deletion(checkout, capsys, operation):
    import base64

    directory, original_path, _ = checkout
    empty_baseline = operation in {"fill_empty", "delete_empty", "rename_empty", "copy_empty"}
    before = "" if empty_baseline else BEFORE
    original_path.write_text(before)
    if empty_baseline:
        subprocess.run(["git", "add", "package/model.py"], check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "Empty fixture."],
            check=True,
        )
    path = "package/model.py"
    source_path = path
    after = ""
    if operation == "truncate":
        original_path.write_text("")
    elif operation == "fill_empty":
        after = AFTER
        original_path.write_text(after)
    elif operation == "create_empty":
        path = "package/empty.py"
        source_path = None
        (directory.parent / path).touch()
        subprocess.run(["git", "add", path], check=True)
    elif operation == "delete_empty":
        subprocess.run(["git", "rm", "-q", path], check=True)
    else:
        path = "package/relocated.py"
        verb = "rename" if operation == "rename_empty" else "copy"
        model_patch = (
            f"diff --git a/package/model.py b/{path}\nsimilarity index 100%\n"
            f"{verb} from package/model.py\n{verb} to {path}\n"
        )
    if operation not in {"rename_empty", "copy_empty"}:
        model_patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], text=True)
        subprocess.run(["git", "reset", "--hard", "-q", "HEAD"], check=True)
    subprocess.run(["git", "apply", "-"], input=model_patch, text=True, check=True)
    manifest = syntax_probe_manifest(model_patch)
    expected = [] if operation == "delete_empty" else [{"path": path, "source_path": source_path}]

    assert manifest["files"] == expected, "Git file lifecycle metadata must distinguish an empty file from deletion."
    (directory / "syntax_files.json").write_text(json.dumps(manifest))
    syntax_probe.capture_sources(directory)
    snapshot = json.loads((directory / "syntax_sources.json").read_text())
    syntax_probe.check_sources(directory)
    proof = capsys.readouterr().out
    if operation == "delete_empty":
        assert not snapshot["sources"] and not original_path.exists(), "A declared deletion must stay deleted."
    else:
        assert (directory.parent / path).read_text() == after, "Capture must preserve the actual applied target bytes."
        source = snapshot["sources"][0]
        assert base64.b64decode(source["before"]) == (b"" if source_path is None else before.encode()), (
            "Only a declared creation may omit the task revision source path."
        )
        assert base64.b64decode(source["after"]) == after.encode(), "An empty target is still a captured file."
    assert bool(proof) is (operation == "fill_empty"), "Only the invalid newly filled source has a syntax regression."


def test_mode_only_patch_does_not_identify_unchanged_source_as_a_candidate(checkout, capsys):
    directory, path, _ = checkout
    path.write_text(BEFORE)
    subprocess.run(["git", "update-index", "--chmod=+x", "package/model.py"], check=True)
    model_patch = subprocess.check_output(["git", "diff", "--cached", "--binary"], text=True)
    subprocess.run(["git", "reset", "--hard", "-q", "HEAD"], check=True)
    subprocess.run(["git", "apply", "-"], input=model_patch, text=True, check=True)
    manifest = syntax_probe_manifest(model_patch)

    assert path.read_text() == BEFORE, "An executable mode change must not alter Python source bytes."
    assert path.stat().st_mode & 0o111, "The real Git patch must still apply its executable mode change."
    assert not manifest["files"], "A mode change alone cannot explain a Python source regression."
    (directory / "syntax_files.json").write_text(json.dumps(manifest))
    syntax_probe.capture_sources(directory)
    syntax_probe.check_sources(directory)

    assert not capsys.readouterr().out, "Unchanged source must not receive syntax regression evidence."
