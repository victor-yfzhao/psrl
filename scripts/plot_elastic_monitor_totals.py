#!/usr/bin/env python3
"""
Parse ElasticMonitor logs and plot RM/Rollout aggregate metrics over time.

Metrics:
- total throughput
- total running queue
- total waiting queue
"""

from __future__ import annotations

import argparse
import ast
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = (
    REPO_ROOT
    / "logs/psrl_elastic_rm/sync_elastic_rm_4_qwen_30b_a3b_rollout_4_ds_distill_staleness_2/ElasticMonitor.log"
)
RE_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r".*Instance current Status:\s*(?P<status>\{.*\})\s*$"
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot RM/Rollout total throughput/running/waiting from ElasticMonitor.log."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Path to ElasticMonitor.log (default: {DEFAULT_LOG_PATH}).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Output png path (default: <log_dir>/<log_stem>_rm_rollout_totals.png).",
    )
    return parser.parse_args()


def parse_log(
    log_path: Path,
) -> tuple[list[datetime], dict[str, dict[str, list[float]]]]:
    """
    Parse log and aggregate metrics by monitor timestamp and role.

    Args:
        log_path (Path): ElasticMonitor log path.

    Returns:
        tuple[list[datetime], dict[str, dict[str, list[float]]]]: Time axis and
            nested metric series, formatted as:
            data[role]["throughput"|"running"|"waiting"] -> list aligned to times.
    """
    grouped: dict[datetime, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"throughput": 0.0, "running": 0.0, "waiting": 0.0})
    )

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_LINE.match(line)
            if not m:
                continue

            dt = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
            try:
                status_dict = ast.literal_eval(m.group("status"))
            except (SyntaxError, ValueError):
                continue

            role = status_dict.get("role")
            if role not in ("RewardModel", "Rollout"):
                continue

            try:
                grouped[dt][role]["throughput"] += float(status_dict.get("throughput", 0.0))
                grouped[dt][role]["running"] += float(status_dict.get("running", 0.0))
                grouped[dt][role]["waiting"] += float(status_dict.get("waiting", 0.0))
            except (TypeError, ValueError):
                continue

    if not grouped:
        return [], {}

    times = sorted(grouped.keys())
    roles = ("RewardModel", "Rollout")
    metrics = ("throughput", "running", "waiting")
    data: dict[str, dict[str, list[float]]] = {
        role: {metric: [] for metric in metrics} for role in roles
    }
    for dt in times:
        for role in roles:
            for metric in metrics:
                data[role][metric].append(grouped[dt][role][metric])

    return times, data


def trim_trailing_all_zero_points(
    times: list[datetime],
    data: dict[str, dict[str, list[float]]],
) -> tuple[list[datetime], dict[str, dict[str, list[float]]]]:
    """
    Trim trailing points where all plotted series are zero.

    Args:
        times (list[datetime]): Time axis.
        data (dict[str, dict[str, list[float]]]): Aggregated metric series.

    Returns:
        tuple[list[datetime], dict[str, dict[str, list[float]]]]: Trimmed time axis
            and metric series.
    """
    if not times:
        return times, data

    roles = ("RewardModel", "Rollout")
    metrics = ("throughput", "running", "waiting")

    last_non_zero_idx = -1
    for idx in range(len(times)):
        point_sum = 0.0
        for role in roles:
            for metric in metrics:
                point_sum += data[role][metric][idx]
        if point_sum > 0.0:
            last_non_zero_idx = idx

    if last_non_zero_idx < 0:
        return [], {role: {metric: [] for metric in metrics} for role in roles}

    end = last_non_zero_idx + 1
    trimmed_data: dict[str, dict[str, list[float]]] = {
        role: {metric: data[role][metric][:end] for metric in metrics} for role in roles
    }
    return times[:end], trimmed_data


def plot_totals(
    times: list[datetime],
    data: dict[str, dict[str, list[float]]],
    out_path: Path,
) -> None:
    """
    Plot aggregate RM and Rollout metrics over time.

    Args:
        times (list[datetime]): Time axis from monitor timestamp.
        data (dict[str, dict[str, list[float]]]): Aggregated metric series.
        out_path (Path): Output figure path.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")

    metric_specs = [
        ("throughput", "Total Throughput"),
        ("running", "Total Running Queue"),
        ("waiting", "Total Waiting Queue"),
    ]
    role_styles = {
        "RewardModel": {"label": "RewardModel", "color": "#c0392b"},
        "Rollout": {"label": "Rollout", "color": "#2980b9"},
    }

    for ax, (metric, ylabel) in zip(axes, metric_specs):
        for role, style in role_styles.items():
            ax.plot(
                times,
                data[role][metric],
                label=style["label"],
                color=style["color"],
                linewidth=0.8,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.suptitle("ElasticMonitor Aggregate Metrics: RewardModel vs Rollout")
    fig.savefig(out_path, dpi=160)


def main() -> None:
    """
    Parse log, aggregate metrics, and save the figure.
    """
    args = parse_args()
    log_path = args.log_path.resolve()
    if not log_path.is_file():
        raise SystemExit(f"Log file not found: {log_path}")

    out_path = (
        args.out_path.resolve()
        if args.out_path is not None
        else (log_path.parent / f"{log_path.stem}_rm_rollout_totals.png").resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    times, data = parse_log(log_path)
    if not times:
        raise SystemExit("No valid 'Instance current Status' records were parsed.")
    times, data = trim_trailing_all_zero_points(times, data)
    if not times:
        raise SystemExit("All parsed points are zero after trailing-zero trimming.")

    plot_totals(times, data, out_path)
    print(f"Wrote {out_path} ({len(times)} points)")


if __name__ == "__main__":
    main()
