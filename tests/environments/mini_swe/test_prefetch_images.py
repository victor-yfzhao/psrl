"""Verify that mirror failures reach the configured registry destinations."""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("direct_fallback", [True, False])
def test_mirror_failures_reach_docker_hub(tmp_path: Path, direct_fallback: bool):
    """Run the complete shell dispatcher without network or Docker access."""
    docker = tmp_path / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
case "$1 $2" in
    'image ls') exit 0 ;;
    'image inspect') exit 1 ;;
esac
if [[ "$1" == pull ]]; then
    printf '%s\\n' "$2" >> "$PSRL_TEST_PULLS"
    [[ "$2" != *.invalid/* ]]
else
    exit 0
fi
"""
    )
    docker.chmod(0o755)
    pulls = tmp_path / "pulls.txt"
    image = "example/swe:latest"
    script = Path(__file__).resolve().parents[3] / "examples/mini_swe/prepare/docker_scripts/prefetch_images.sh"
    command = [
        "bash",
        str(script),
        "--only",
        image,
        "--method",
        "docker",
        "--workers",
        "1",
        "--retries",
        "1",
        "--mirrors",
        "first.invalid,second.invalid",
        "--log-dir",
        str(tmp_path / "logs"),
    ]
    if not direct_fallback:
        command.append("--no-direct-fallback")
    result = subprocess.run(
        command,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "PSRL_WORKSPACE": str(tmp_path),
            "PSRL_TEST_PULLS": str(pulls),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    expected = [f"first.invalid/{image}", f"second.invalid/{image}"]
    if direct_fallback:
        expected.append(image)
    assert pulls.read_text().splitlines() == expected, "Workers must preserve every configured destination."
    assert (result.returncode == 0) is direct_fallback, result.stdout + result.stderr
