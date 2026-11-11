"""
Check prepared mini-SWE images on every eligible Ray node before model initialization.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

psrl_logger = logging.getLogger(__name__)

DOCKER_SOCKET = "unix:///var/run/docker.sock"


def normalize_image(
    image: str,
) -> str:
    """
    Match Docker Hub references to the spelling used by Docker image listings.
    """
    image = image.removeprefix("docker.io/").removeprefix("index.docker.io/").removeprefix("library/")
    if "@" not in image and ":" not in image.rsplit("/", 1)[-1]:
        image += ":latest"
    return image


def required_images(
    parquets: list[Path],
) -> list[str]:
    """
    Collect every environment and grader image from the supplied dataset splits.
    """
    import pyarrow.parquet as pq

    images = set()
    for path in parquets:
        rows = pq.read_table(path, columns=["extra_info"]).column(0).to_pylist()
        if not rows:
            raise ValueError(f"Dataset {str(path)!r} is empty.")
        for index, extra in enumerate(rows):
            if isinstance(extra, str):
                extra = json.loads(extra)
            if not isinstance(extra, dict):
                raise ValueError(f"Dataset {str(path)!r}, row {index} has invalid extra_info.")
            image = extra.get("sandbox_overrides", {}).get("environment", {}).get("image")
            if not isinstance(image, str) or not image.strip():
                raise ValueError(f"Dataset {str(path)!r}, row {index} has no environment image.")
            images.add(normalize_image(image.strip()))
            grader_image = extra.get("swe_problem_image")
            if grader_image:
                images.add(normalize_image(grader_image.strip()))
    return sorted(images)


def check_docker(
    images: list[str],
    timeout: float = 30,
) -> dict[str, Any]:
    """
    Check the default daemon and its complete image inventory without modifying it.
    """
    report: dict[str, Any] = {"error": None, "missing_images": []}
    try:
        context = subprocess.run(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        endpoint = context.stdout.strip()
        if not os.environ.get("DOCKER_CONTEXT"):
            endpoint = os.environ.get("DOCKER_HOST") or endpoint
        if endpoint != DOCKER_SOCKET:
            raise ValueError(f"Docker endpoint must be {DOCKER_SOCKET!r}, got {endpoint!r}.")
        subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        inventory = subprocess.run(
            ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        loaded = {normalize_image(image.strip()) for image in inventory.stdout.splitlines() if image.strip()}
        report["missing_images"] = sorted(set(images) - loaded)
    except subprocess.CalledProcessError as exc:
        report["error"] = f"{exc}. stderr: {(exc.stderr or '').strip()}"
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        report["error"] = str(exc)
    return report


def check_cluster(
    images: list[str],
    node_ips: list[str] | None = None,
    timeout: float = 180,
) -> list[dict[str, Any]]:
    """
    Pin one read-only Docker check to each eligible node in the active Ray cluster.
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    initialized = ray.is_initialized()
    if not initialized:
        ray.init(address="auto", log_to_driver=False)
    try:
        nodes = [node for node in ray.nodes() if node["Alive"]]
        if node_ips:
            unavailable = set(node_ips) - {node["NodeManagerAddress"] for node in nodes}
            if unavailable:
                raise ValueError(f"Agent nodes are absent from Ray: {sorted(unavailable)!r}.")
            nodes = [node for node in nodes if node["NodeManagerAddress"] in node_ips]
        if not nodes:
            raise ValueError("No eligible Agent nodes are alive in Ray.")
        check = ray.remote(num_cpus=0)(check_docker)
        futures = [
            check.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            ).remote(images)
            for node in nodes
        ]
        results = ray.get(futures, timeout=timeout)
        return [
            {"node_id": node["NodeID"], "node_ip": node["NodeManagerAddress"], **result}
            for node, result in zip(nodes, results, strict=True)
        ]
    finally:
        if not initialized:
            ray.shutdown()


def main() -> None:
    """
    List required images or fail startup when any eligible node is unprepared.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", action="append", type=Path, required=True, help="Training or validation parquet.")
    parser.add_argument(
        "--node-ips", nargs="+", help="Match rollout.agent.node_ips when Agent placement is restricted."
    )
    parser.add_argument(
        "--list-images", action="store_true", help="Print the complete image list without contacting Ray."
    )
    parser.add_argument("--report", type=Path, help="Write all node errors and missing images as JSON.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    images = required_images(args.parquet)
    if args.list_images:
        for image in images:
            print(image)
        return
    reports = check_cluster(images, args.node_ips)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2) + "\n")
    failed = False
    for report in reports:
        node = report["node_ip"]
        if report["error"]:
            psrl_logger.error(f"Node {node}: Docker check failed: {report['error']}.")
            failed = True
        elif report["missing_images"]:
            psrl_logger.error(f"Node {node}: Missing {len(report['missing_images'])} required images.")
            for image in report["missing_images"]:
                psrl_logger.error(f"Missing image: {image}.")
            failed = True
        else:
            psrl_logger.info(f"Node {node}: Docker is ready with all {len(images)} required images.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
