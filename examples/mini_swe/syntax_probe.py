"""
Verify Python syntax regressions with the evaluation environment's interpreter.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SYNTAX_REGRESSION_MARKER = "PSRL_SYNTAX_REGRESSION "


def _digest(source: bytes) -> str:
    """
    Identify the exact source bytes used by the syntax check.
    """
    return hashlib.sha256(source).hexdigest()


def capture_sources(directory: Path) -> None:
    """
    Save task revision and applied model sources before the evaluation script runs.
    """
    manifest = json.loads((directory / "syntax_files.json").read_text())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], timeout=30, text=True).strip()
    sources = []
    for entry in manifest["files"]:
        before = (
            subprocess.check_output(["git", "show", f"{revision}:{entry['source_path']}"], timeout=30)
            if entry["source_path"] is not None
            else b""
        )
        after = Path(entry["path"]).read_bytes()
        sources.append(
            {
                "path": entry["path"],
                "before": base64.b64encode(before).decode("ascii"),
                "after": base64.b64encode(after).decode("ascii"),
            }
        )
    (directory / "syntax_sources.json").write_text(
        json.dumps({"revision": revision, "model_patch_sha256": manifest["model_patch_sha256"], "sources": sources})
    )


def _uses_pytest_interpreter() -> bool:
    """
    Require the active Python to match the pytest entry point's interpreter.
    """
    executable = shutil.which("pytest")
    if executable is None:
        return False
    with Path(executable).open("rb") as stream:
        first_line = stream.readline().decode("utf-8", errors="replace").strip()
    if not first_line.startswith("#!"):
        return False
    command = shlex.split(first_line[2:])
    if not command:
        return False
    interpreter = command[0]
    if Path(interpreter).name == "env":
        if len(command) != 2:
            return False
        interpreter = shutil.which(command[1])
    return interpreter is not None and Path(interpreter).resolve() == Path(sys.executable).resolve()


def check_sources(directory: Path) -> None:
    """
    Emit proof only when unchanged evaluation sources newly fail to compile.
    """
    if not _uses_pytest_interpreter():
        return
    snapshot = json.loads((directory / "syntax_sources.json").read_text())
    for entry in snapshot["sources"]:
        before = base64.b64decode(entry["before"])
        after = base64.b64decode(entry["after"])
        if before == after or Path(entry["path"]).read_bytes() != after:
            continue
        try:
            compile(before, entry["path"], "exec", dont_inherit=True)
        except SyntaxError:
            continue
        try:
            compile(after, entry["path"], "exec", dont_inherit=True)
        except SyntaxError as exc:
            evidence = {
                "path": entry["path"],
                "error_type": type(exc).__name__,
                "line": exc.lineno,
                "message": exc.msg,
                "offset": exc.offset,
                "revision": snapshot["revision"],
                "model_patch_sha256": snapshot["model_patch_sha256"],
                "before_sha256": _digest(before),
                "after_sha256": _digest(after),
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
            }
            print(SYNTAX_REGRESSION_MARKER + json.dumps(evidence), flush=True)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    if sys.argv[1] == "capture":
        capture_sources(directory)
    elif sys.argv[1] == "check":
        check_sources(directory)
    else:
        raise ValueError(f"Unknown syntax probe operation {sys.argv[1]!r}.")
