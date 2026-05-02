#!/usr/bin/env python3
"""
Parse ElasticExecutor.log lines of the form:

  YYYY-mm-dd HH:MM:SS,mmm - ... - Awake instances summary: by_role={'RewardModel': N, 'Rollout': M}, ...

Plot RewardModel and Rollout awake instance counts vs time on one figure.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

# --- paths: edit here ---
REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = (
    REPO_ROOT
    / "logs/psrl_elastic_rm/sync_elastic_rm_4_qwen_30b_a3b_rollout_4_ds_distill_staleness_2/ElasticExecutor.log"
)
OUT_PATH = LOG_PATH.parent / f"{LOG_PATH.stem}_awake_instances.png"

RE_AWAKE_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r".*Awake instances summary: by_role=\{(?P<br>[^}]+)\}"
)
RE_REWARD_MODEL = re.compile(r"'RewardModel':\s*(\d+)")
RE_ROLLOUT = re.compile(r"'Rollout':\s*(\d+)")


def parse_elastic_executor_log(path: Path) -> tuple[list[datetime], list[int], list[int]]:
    times: list[datetime] = []
    reward_model: list[int] = []
    rollout: list[int] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_AWAKE_LINE.match(line)
            if not m:
                continue
            br = m.group("br")
            m_rm = RE_REWARD_MODEL.search(br)
            m_ro = RE_ROLLOUT.search(br)
            if not (m_rm and m_ro):
                continue
            dt = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
            times.append(dt)
            reward_model.append(int(m_rm.group(1)))
            rollout.append(int(m_ro.group(1)))
    return times, reward_model, rollout


def trim_trailing_constant_tail(
    times: list[datetime],
    reward_model: list[int],
    rollout: list[int],
) -> tuple[list[datetime], list[int], list[int]]:
    """
    Trim the trailing constant tail for (RewardModel, Rollout).

    Args:
        times (list[datetime]): Time axis.
        reward_model (list[int]): RewardModel awake counts.
        rollout (list[int]): Rollout awake counts.

    Returns:
        tuple[list[datetime], list[int], list[int]]: Trimmed series.
    """
    if not times:
        return times, reward_model, rollout

    final_pair = (reward_model[-1], rollout[-1])
    last_change_idx = -1
    for idx in range(len(times) - 1):
        current_pair = (reward_model[idx], rollout[idx])
        if current_pair != final_pair:
            last_change_idx = idx

    if last_change_idx < 0:
        return times, reward_model, rollout

    end = last_change_idx + 1
    return times[:end], reward_model[:end], rollout[:end]


def main() -> None:
    log_path = LOG_PATH.resolve()
    if not log_path.is_file():
        raise SystemExit(f"File not found: {log_path}")

    times, rm, ro = parse_elastic_executor_log(log_path)
    if not times:
        raise SystemExit("No 'Awake instances summary' lines with by_role parsed.")
    times, rm, ro = trim_trailing_constant_tail(times, rm, ro)
    if not times:
        raise SystemExit("No points left after trailing-constant-tail trimming.")

    out = OUT_PATH.resolve()

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.plot(times, rm, label="RewardModel (awake)", color="#c0392b", linewidth=1.2)
    ax.plot(times, ro, label="Rollout (awake)", color="#2980b9", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Awake instances")
    ax.set_title(f"Awake instances over time — {log_path.name}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.35)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=30, ha="right")

    fig.savefig(out, dpi=150)
    print(f"Wrote {out} ({len(times)} points)")


if __name__ == "__main__":
    main()
