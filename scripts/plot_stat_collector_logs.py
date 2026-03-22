#!/usr/bin/env python3
"""
Parse StatCollector logs (rollout and reward model) and plot num_running_reqs,
num_waiting_reqs, kv_cache_usage vs time for each instance.
Outputs three figures under OUT_DIR.
"""

import re
from datetime import datetime
from pathlib import Path

# Constants: customize log dir and output dir here
LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "psrl_elastic_rm" / "dis_rm_rollout_4_rm_4_stat_collects"
OUT_DIR = LOG_DIR / "plots"

# Time range (relative seconds from each instance's first snapshot). None = no bound.
TIME_RANGE_START_SEC: float | None = 2000.0  # e.g. 0.0
TIME_RANGE_END_SEC: float | None = 3000.0   # e.g. 3600.0

# Regex to extract fields from a Snapshot log line (scheduler_stats subset)
RE_TIMESTAMP = re.compile(r"'timestamp':\s*'([^']+)'")
RE_NUM_RUNNING = re.compile(r"'num_running_reqs':\s*(\d+)")
RE_NUM_WAITING = re.compile(r"'num_waiting_reqs':\s*(\d+)")
RE_KV_CACHE = re.compile(r"'kv_cache_usage':\s*([\d.]+)")


def parse_log_file(path: Path) -> list[tuple[datetime, int, int, float]] | None:
    """Parse a single log file line-by-line. Returns list of (timestamp, num_running_reqs, num_waiting_reqs, kv_cache_usage)."""
    records = []
    for line in path.open("r", encoding="utf-8", errors="replace"):
        if "Snapshot (model version" not in line:
            continue
        ts = RE_TIMESTAMP.search(line)
        nr = RE_NUM_RUNNING.search(line)
        nw = RE_NUM_WAITING.search(line)
        kv = RE_KV_CACHE.search(line)
        if not (ts and nr and nw and kv):
            continue
        try:
            dt = datetime.fromisoformat(ts.group(1))
            records.append((dt, int(nr.group(1)), int(nw.group(1)), float(kv.group(1))))
        except (ValueError, TypeError):
            continue
    return records if records else None


def classify_and_load(
    log_dir: Path,
    time_start_sec: float | None = None,
    time_end_sec: float | None = None,
) -> dict[str, list[tuple[float, int, int, float]]]:
    """
    Load all StatCollector log files under log_dir. Returns a dict:
    label -> list of (relative_seconds, num_running_reqs, num_waiting_reqs, kv_cache_usage).
    Labels are e.g. "Rollout I0", "RM I1".
    If time_start_sec/time_end_sec are set, only points within [start, end] are kept.
    """
    rollout_pattern = re.compile(r"StatCollector_I(\d+)\.log$")
    rm_pattern = re.compile(r"StatCollector_RM_Qwen3-8B_I(\d+)\.log$")

    out: dict[str, list[tuple[float, int, int, float]]] = {}

    for path in sorted(log_dir.iterdir()):
        if not path.is_file() or not path.suffix == ".log":
            continue
        name = path.name
        if m := rollout_pattern.match(name):
            label = f"Rollout I{m.group(1)}"
        elif m := rm_pattern.match(name):
            label = f"RM I{m.group(1)}"
        else:
            continue

        raw = parse_log_file(path)
        if not raw:
            print(f"Skip (no valid snapshots): {path.name}")
            continue

        raw.sort(key=lambda r: r[0])
        t0 = raw[0][0]
        series = [((r[0] - t0).total_seconds(), r[1], r[2], r[3]) for r in raw]

        if time_start_sec is not None or time_end_sec is not None:
            series = [
                s for s in series
                if (time_start_sec is None or s[0] >= time_start_sec)
                and (time_end_sec is None or s[0] <= time_end_sec)
            ]
        out[label] = series

    return out


def plot_metric(
    data: dict[str, list[tuple[float, int, int, float]]],
    metric_index: int,
    ylabel: str,
    out_path: Path,
    linewidth: float = 0.8,
) -> None:
    """Plot one metric (by metric_index: 1=num_running, 2=num_waiting, 3=kv_cache) for all series."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for label, series in data.items():
        if not series:
            continue
        x = [s[0] for s in series]
        y = [s[metric_index] for s in series]
        linestyle = "solid" if label.startswith("Rollout") else "dashed"
        ax.plot(x, y, label=label, linestyle=linestyle, linewidth=linewidth)

    ax.set_xlabel("Time (s, relative to first snapshot)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not LOG_DIR.is_dir():
        print(f"Log directory not found: {LOG_DIR}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = classify_and_load(
        LOG_DIR,
        time_start_sec=TIME_RANGE_START_SEC,
        time_end_sec=TIME_RANGE_END_SEC,
    )
    if not data:
        print("No valid log data found.")
        return

    # 1: num_running_reqs (index 1 in each record)
    plot_metric(
        data,
        metric_index=1,
        ylabel="num_running_reqs",
        out_path=OUT_DIR / "num_running_reqs.png",
    )
    # 2: num_waiting_reqs (index 2)
    plot_metric(
        data,
        metric_index=2,
        ylabel="num_waiting_reqs",
        out_path=OUT_DIR / "num_waiting_reqs.png",
    )
    # 3: kv_cache_usage (index 3)
    plot_metric(
        data,
        metric_index=3,
        ylabel="kv_cache_usage",
        out_path=OUT_DIR / "kv_cache_usage.png",
    )
    print(f"Plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
