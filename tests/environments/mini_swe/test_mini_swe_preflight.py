"""
Verify complete image checks and Docker failures before mini-SWE model startup.
"""

import json
import subprocess
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from examples.mini_swe import preflight, runner

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("serialize_extra_info", [False, True])
def test_required_images_include_every_split_and_grader(tmp_path, serialize_extra_info):
    files = []
    for split, image, grader in (
        ("train", "docker.io/example/train:latest", "example/grader:latest"),
        ("validation", "example/validation:latest", "example/grader:latest"),
    ):
        extra = {"sandbox_overrides": {"environment": {"image": image}}, "swe_problem_image": grader}
        path = tmp_path / f"{split}.parquet"
        pq.write_table(
            pa.Table.from_pylist([{"extra_info": json.dumps(extra) if serialize_extra_info else extra}]), path
        )
        files.append(path)
    assert preflight.required_images(files) == [
        "example/grader:latest",
        "example/train:latest",
        "example/validation:latest",
    ], "Training, validation and grading images must all be checked."


def test_required_images_reject_missing_environment_image(tmp_path):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist([{"extra_info": {"swe_problem_image": "example/grader:latest"}}]), path)
    with pytest.raises(ValueError, match="row 0 has no environment image"):
        preflight.required_images([path])


def test_docker_inventory_detects_missing_validation_images(monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    execute = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout=preflight.DOCKER_SOCKET),
            subprocess.CompletedProcess([], 0, stdout="26.1.3"),
            subprocess.CompletedProcess([], 0, stdout="example/train:latest\n"),
        ]
    )
    monkeypatch.setattr(preflight.subprocess, "run", execute)
    report = preflight.check_docker(["example/train:latest", "example/validation:latest"])
    assert report == {"error": None, "missing_images": ["example/validation:latest"]}, (
        "The entire required inventory must be checked instead of sampling training images."
    )
    assert execute.call_count == 3, "One inventory request must cover every required image."


def test_docker_daemon_failure_preserves_stderr(monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    execute = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0, stdout=preflight.DOCKER_SOCKET),
            subprocess.CalledProcessError(1, ["docker", "info"], stderr="Cannot connect to the Docker daemon"),
        ]
    )
    monkeypatch.setattr(preflight.subprocess, "run", execute)
    report = preflight.check_docker(["example/train:latest"])
    assert "Cannot connect to the Docker daemon" in report["error"], "Daemon failures must expose their stderr."
    assert execute.call_count == 2, "Image checks must stop when the daemon is unreachable."


def test_docker_check_rejects_another_endpoint(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/another-docker.sock")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 0, stdout=preflight.DOCKER_SOCKET)),
    )
    report = preflight.check_docker([])
    assert "Docker endpoint must be" in report["error"], "Preparation and training must use the same daemon."


def test_cluster_check_is_pinned_to_every_eligible_node(monkeypatch):
    import ray

    selected = []

    class Check:
        def options(self, *, scheduling_strategy):
            selected.append((scheduling_strategy.node_id, scheduling_strategy.soft))
            return self

        def remote(self, images):
            return {"error": None, "missing_images": []}

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        ray,
        "nodes",
        lambda: [
            {"NodeID": "first", "NodeManagerAddress": "node-a", "Alive": True},
            {"NodeID": "second", "NodeManagerAddress": "node-b", "Alive": True},
            {"NodeID": "dead", "NodeManagerAddress": "node-c", "Alive": False},
        ],
    )
    monkeypatch.setattr(ray, "remote", lambda **kwargs: lambda fn: Check())
    monkeypatch.setattr(ray, "get", lambda refs, **kwargs: refs)
    reports = preflight.check_cluster(["example/train:latest"])
    assert selected == [("first", False), ("second", False)], (
        "Checks must execute on each live node without relocation."
    )
    assert [report["node_ip"] for report in reports] == ["node-a", "node-b"], (
        "Reports must identify each checked node."
    )
    selected.clear()
    preflight.check_cluster([], node_ips=["node-b"])
    assert selected == [("second", False)], "Restricted Agent placement must restrict the checked nodes."
    with pytest.raises(ValueError, match="absent from Ray"):
        preflight.check_cluster([], node_ips=["node-c"])


@pytest.mark.parametrize("error,missing", [("Daemon unavailable.", []), (None, ["example/validation:latest"])])
def test_preflight_exits_unsuccessfully_for_any_unprepared_node(monkeypatch, tmp_path, error, missing):
    report = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["preflight", "--parquet", "data.parquet", "--report", str(report)])
    monkeypatch.setattr(preflight, "required_images", lambda files: ["example/validation:latest"])
    reports = [{"node_ip": "node-a", "error": error, "missing_images": missing}]
    monkeypatch.setattr(preflight, "check_cluster", lambda *args: reports)
    with pytest.raises(SystemExit) as exc:
        preflight.main()
    assert exc.value.code == 1, "An unprepared node must prevent training startup."
    assert json.loads(report.read_text()) == reports, "The report must retain every missing image and daemon failure."


def test_runner_returns_and_logs_docker_stderr(monkeypatch, caplog):
    from minisweagent import environments

    monkeypatch.setattr(runner, "_silence_litellm", lambda: None)
    monkeypatch.setattr(
        environments,
        "get_environment",
        Mock(
            side_effect=subprocess.CalledProcessError(
                125,
                ["docker", "run", "example/swe:latest"],
                output="container-id",
                stderr="Cannot connect to Docker daemon",
            )
        ),
    )
    payload = {
        "runtime_config": {"sandbox_config": {"environment": {"image": "example/swe:latest"}}},
        "observation": {},
    }
    result = runner.run_agent(payload)
    assert result["exit_status"] == "error", "A failed Docker command must fail the episode."
    assert "125" in result["error"] and "docker" in result["error"], "The failed command and exit code must survive."
    assert "Cannot connect to Docker daemon" in result["error"], "The returned runner error must preserve stderr."
    assert "container-id" in result["error"], "The returned runner error must preserve stdout."
    assert "Cannot connect to Docker daemon" in caplog.text, "The task log must expose Docker stderr."
