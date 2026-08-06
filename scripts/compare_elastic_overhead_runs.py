#!/usr/bin/env python3
"""Extract and compare elastic scaling overheads from two PSRL runs.

The parser reads ElasticExecutor.log, RolloutRouter.log, and
RewardModelRouter.log from each run directory. It prints a compact comparison
and can write four CSV files:

* overhead_events.csv: one row per extracted numeric metric.
* overhead_summary.csv: descriptive statistics for each metric.
* rebalance_effectiveness.csv: planned/accepted/interrupted/routed counts.
* run_windows.csv: source-log timestamps and the selected relative-time window.

Percentiles use linear interpolation over sorted samples. ``network_s`` keeps
the runtime log's scope: ABORT to redispatch, not pure network transfer.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
TIMESTAMP_RE = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
OVERHEAD_RE = re.compile(r"\[ELASTIC_OVERHEAD\]\s+(?P<body>.*)$")
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s]+)")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
PREPARED_RE = re.compile(
    r"Prepared (?P<role>RM|rollout) request migrations: "
    r"planned=(?P<planned>\d+) accepted=(?P<accepted>\d+) skipped=(?P<skipped>\d+)"
)

FAST_POLICY_REASONS = {"cooldown", "decision_execution_in_progress"}
LOG_FILES = ("ElasticExecutor.log", "RolloutRouter.log", "RewardModelRouter.log")
TIME_METRICS = {
    "planner_s",
    "sleep_s",
    "wakeup_s",
    "post_scale_up_rebalance_trigger_s",
    "handler_other_s",
    "execution_s",
    "total_s",
    "network_trigger_s",
    "planner_batch_s",
    "planner_share_s",
    "network_s",
    "reprefill_s",
    "migration_s",
}


@dataclass(frozen=True)
class MetricEvent:
    run: str
    run_path: str
    timestamp: str
    elapsed_min: float
    source: str
    operation: str
    cohort: str
    role: str
    method: str
    decision_id: str
    migration_id: str
    reason: str
    metric: str
    value: float


@dataclass(frozen=True)
class MetricSummary:
    run: str
    operation: str
    cohort: str
    role: str
    metric: str
    count: int
    mean: float
    stddev: float
    minimum: float
    p50: float
    p90: float
    p95: float
    p99: float
    maximum: float
    total: float


@dataclass
class RebalanceCounts:
    run: str
    role: str
    method: str = "unknown"
    trigger_events: int = 0
    planned: int = 0
    accepted: int = 0
    interrupted: int = 0
    skipped: int = 0
    forced: int = 0
    fallback: int = 0
    redispatched_overhead: int = 0
    completed_overhead: int = 0

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        request_level = self.method == "request_level"
        row["accepted_available"] = request_level
        row["accepted_ratio"] = _safe_ratio(self.accepted, self.planned) if request_level else float("nan")
        row["interrupted_ratio"] = _safe_ratio(self.interrupted, self.planned)
        row["forced_ratio"] = _safe_ratio(self.forced, self.forced + self.fallback)
        return row


@dataclass(frozen=True)
class ParsedOverhead:
    timestamp: datetime
    source: str
    fields: dict[str, str]


@dataclass(frozen=True)
class RunWindow:
    run: str
    run_path: str
    first_timestamp: str
    last_timestamp: str
    observed_minutes: float
    selected_start_min: float
    selected_end_min: float


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def _parse_number(raw: str | None) -> float | None:
    if raw is None:
        return None
    normalized = raw.rstrip(",")
    if not NUMBER_RE.fullmatch(normalized):
        return None
    value = float(normalized)
    return value if math.isfinite(value) else None


def _parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if match is None:
        return None
    return datetime.strptime(match.group("timestamp"), TIMESTAMP_FORMAT)


def _role_for_source(source: str) -> str:
    if source == "RolloutRouter.log":
        return "Rollout"
    if source == "RewardModelRouter.log":
        return "RewardModel"
    return "all"


def _role_for_overhead(parsed: ParsedOverhead) -> str:
    explicit_role = parsed.fields.get("role")
    if explicit_role:
        return explicit_role
    migration_parts = parsed.fields.get("migration_id", "").split(":")
    if len(migration_parts) >= 3 and migration_parts[2] in {"RewardModel", "Rollout"}:
        return migration_parts[2]
    return _role_for_source(parsed.source)


def _normalize_overhead(parsed: ParsedOverhead) -> tuple[str, str, str, str, dict[str, float]] | None:
    fields = parsed.fields
    raw_operation = fields.get("operation", "")
    role = _role_for_overhead(parsed)
    method = "unknown"
    cohort = "all"
    metrics: dict[str, float] = {}

    if raw_operation == "policy_planner":
        operation = "policy_planner"
        reason = fields.get("reason", "")
        actions = int(_parse_number(fields.get("actions")) or 0)
        if actions > 0:
            cohort = "action"
            if reason.endswith("RewardModel"):
                role = "RewardModel"
            elif reason.endswith("Rollout"):
                role = "Rollout"
        elif reason in FAST_POLICY_REASONS:
            cohort = "fast_gate"
        else:
            cohort = "no_action_evaluation"
        planner_s = _parse_number(fields.get("planner_s"))
        if planner_s is not None:
            metrics["planner_s"] = planner_s
    elif raw_operation == "scale_up":
        operation = "scale_up"
        for metric in TIME_METRICS:
            value = _parse_number(fields.get(metric))
            if value is not None:
                metrics[metric] = value
    elif raw_operation == "post_scale_up_rebalance_trigger":
        operation = "rebalance_trigger"
        method = "legacy"
        for metric in ("planner_s", "network_trigger_s"):
            value = _parse_number(fields.get(metric))
            if value is not None:
                metrics[metric] = value
        planned = _parse_number(fields.get("selected"))
        interrupted = _parse_number(fields.get("interrupted"))
        if planned is not None:
            metrics["planned_requests"] = planned
        if interrupted is not None:
            metrics["interrupted_requests"] = interrupted
        if planned and interrupted is not None:
            metrics["interrupt_ratio"] = interrupted / planned
    elif raw_operation == "planned_request_rebalance":
        operation = "rebalance_trigger"
        method = "request_level"
        planned = _parse_number(fields.get("planned"))
        interrupted = _parse_number(fields.get("interrupted"))
        if planned is not None:
            metrics["planned_requests"] = planned
        if interrupted is not None:
            metrics["interrupted_requests"] = interrupted
        if planned and interrupted is not None:
            metrics["interrupt_ratio"] = interrupted / planned
    elif raw_operation == "post_scale_up_rebalance":
        operation = "request_migration"
        migration_id = fields.get("migration_id", "")
        method = "request_level" if migration_id.endswith(":planned") else "legacy"
        for metric in (
            "planner_batch_s",
            "planner_share_s",
            "network_s",
            "reprefill_s",
            "migration_s",
        ):
            value = _parse_number(fields.get(metric))
            if value is not None:
                metrics[metric] = value
    elif raw_operation == "request_migration_dispatch":
        operation = "request_migration_dispatch"
        migration_id = fields.get("migration_id", "")
        method = "request_level" if migration_id.endswith(":planned") else "legacy"
        for metric in ("planner_batch_s", "planner_share_s", "network_s"):
            value = _parse_number(fields.get(metric))
            if value is not None:
                metrics[metric] = value
    else:
        return None

    if not metrics:
        return None
    return operation, cohort, role, method, metrics


def parse_overhead_lines(run_dir: Path) -> list[ParsedOverhead]:
    parsed: list[ParsedOverhead] = []
    for filename in LOG_FILES:
        log_path = run_dir / filename
        if not log_path.is_file():
            continue
        with log_path.open(encoding="utf-8", errors="replace") as log_file:
            for line in log_file:
                overhead_match = OVERHEAD_RE.search(line)
                if overhead_match is None:
                    continue
                timestamp = _parse_timestamp(line)
                if timestamp is None:
                    continue
                fields = {
                    match.group("key"): match.group("value")
                    for match in KEY_VALUE_RE.finditer(overhead_match.group("body"))
                }
                parsed.append(ParsedOverhead(timestamp, filename, fields))
    return parsed


def collect_metric_events(
    label: str,
    run_dir: Path,
    *,
    start_min: float,
    end_min: float | None,
) -> list[MetricEvent]:
    parsed = parse_overhead_lines(run_dir)
    if not parsed:
        return []
    origin = min(item.timestamp for item in parsed)
    events: list[MetricEvent] = []
    for item in parsed:
        elapsed_min = (item.timestamp - origin).total_seconds() / 60.0
        if elapsed_min < start_min or (end_min is not None and elapsed_min > end_min):
            continue
        normalized = _normalize_overhead(item)
        if normalized is None:
            continue
        operation, cohort, role, method, metrics = normalized
        for metric, value in metrics.items():
            events.append(
                MetricEvent(
                    run=label,
                    run_path=str(run_dir),
                    timestamp=item.timestamp.isoformat(sep=" ", timespec="milliseconds"),
                    elapsed_min=elapsed_min,
                    source=item.source,
                    operation=operation,
                    cohort=cohort,
                    role=role,
                    method=method,
                    decision_id=item.fields.get("decision_id", ""),
                    migration_id=item.fields.get("migration_id", ""),
                    reason=item.fields.get("reason", ""),
                    metric=metric,
                    value=value,
                )
            )
    return events


def describe_run_window(
    label: str,
    run_dir: Path,
    parsed: Sequence[ParsedOverhead],
    *,
    start_min: float,
    end_min: float | None,
) -> RunWindow:
    first = min(item.timestamp for item in parsed)
    last = max(item.timestamp for item in parsed)
    observed_minutes = (last - first).total_seconds() / 60.0
    return RunWindow(
        run=label,
        run_path=str(run_dir),
        first_timestamp=first.isoformat(sep=" ", timespec="milliseconds"),
        last_timestamp=last.isoformat(sep=" ", timespec="milliseconds"),
        observed_minutes=observed_minutes,
        selected_start_min=start_min,
        selected_end_min=min(end_min, observed_minutes) if end_min is not None else observed_minutes,
    )


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_metric_events(events: Iterable[MetricEvent]) -> list[MetricSummary]:
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for event in events:
        grouped[(event.run, event.operation, event.cohort, event.role, event.metric)].append(event.value)

    summaries: list[MetricSummary] = []
    for (run, operation, cohort, role, metric), raw_values in sorted(grouped.items()):
        values = sorted(raw_values)
        summaries.append(
            MetricSummary(
                run=run,
                operation=operation,
                cohort=cohort,
                role=role,
                metric=metric,
                count=len(values),
                mean=statistics.fmean(values),
                stddev=statistics.pstdev(values),
                minimum=values[0],
                p50=_percentile(values, 0.50),
                p90=_percentile(values, 0.90),
                p95=_percentile(values, 0.95),
                p99=_percentile(values, 0.99),
                maximum=values[-1],
                total=sum(values),
            )
        )
    return summaries


def _in_window(timestamp: datetime | None, origin: datetime | None, start_min: float, end_min: float | None) -> bool:
    if timestamp is None or origin is None:
        return False
    elapsed_min = (timestamp - origin).total_seconds() / 60.0
    return elapsed_min >= start_min and (end_min is None or elapsed_min <= end_min)


def collect_rebalance_counts(
    label: str,
    run_dir: Path,
    parsed_overheads: Sequence[ParsedOverhead],
    *,
    start_min: float,
    end_min: float | None,
) -> list[RebalanceCounts]:
    origin = min((item.timestamp for item in parsed_overheads), default=None)
    counts = {role: RebalanceCounts(run=label, role=role) for role in ("RewardModel", "Rollout")}

    for item in parsed_overheads:
        if not _in_window(item.timestamp, origin, start_min, end_min):
            continue
        operation = item.fields.get("operation")
        role = _role_for_overhead(item)
        if role not in counts:
            continue
        row = counts[role]
        if operation == "post_scale_up_rebalance_trigger":
            row.method = "legacy"
            row.trigger_events += 1
            row.planned += int(_parse_number(item.fields.get("selected")) or 0)
            row.interrupted += int(_parse_number(item.fields.get("interrupted")) or 0)
        elif operation == "planned_request_rebalance":
            row.method = "request_level"
            row.trigger_events += 1
            row.planned += int(_parse_number(item.fields.get("planned")) or 0)
            row.interrupted += int(_parse_number(item.fields.get("interrupted")) or 0)
        elif operation == "post_scale_up_rebalance":
            row.completed_overhead += 1
        elif operation == "request_migration_dispatch":
            row.redispatched_overhead += 1

    for filename in ("RolloutRouter.log", "RewardModelRouter.log"):
        log_path = run_dir / filename
        if not log_path.is_file():
            continue
        role = _role_for_source(filename)
        row = counts[role]
        with log_path.open(encoding="utf-8", errors="replace") as log_file:
            for line in log_file:
                timestamp = _parse_timestamp(line)
                if not _in_window(timestamp, origin, start_min, end_min):
                    continue
                prepared = PREPARED_RE.search(line)
                if prepared is not None:
                    row.accepted += int(prepared.group("accepted"))
                    row.skipped += int(prepared.group("skipped"))
                if "Forced planned " in line and " migration:" in line:
                    row.forced += 1
                elif "migration target invalid" in line:
                    row.fallback += 1
    return [counts["RewardModel"], counts["Rollout"]]


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use LABEL=RUN_DIR")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise argparse.ArgumentTypeError("run label must not be empty")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"run directory does not exist: {path}")
    return label, path


def _format_stat(summary: MetricSummary | None) -> str:
    if summary is None:
        return "-"
    return (
        f"mean={summary.mean:.4f} p50={summary.p50:.4f} "
        f"p95={summary.p95:.4f} max={summary.maximum:.4f} n={summary.count}"
    )


def print_comparison(summaries: Sequence[MetricSummary], labels: Sequence[str]) -> None:
    by_key = {(item.run, item.operation, item.cohort, item.role, item.metric): item for item in summaries}
    metric_keys = sorted({(item.operation, item.cohort, item.role, item.metric) for item in summaries})
    print("\nOverhead statistics (seconds for *_s metrics):")
    print(f"| operation/cohort/role/metric | {labels[0]} | {labels[1]} |")
    print("|---|---|---|")
    for operation, cohort, role, metric in metric_keys:
        left = by_key.get((labels[0], operation, cohort, role, metric))
        right = by_key.get((labels[1], operation, cohort, role, metric))
        key = f"{operation}/{cohort}/{role}/{metric}"
        print(f"| {key} | {_format_stat(left)} | {_format_stat(right)} |")


def print_rebalance_effectiveness(rows: Sequence[RebalanceCounts]) -> None:
    print("\nRebalance effectiveness:")
    print(
        "| run | role | events | planned | accepted | interrupted | skipped | "
        "forced | fallback | redispatched | completed |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        accepted = str(row.accepted) if row.method == "request_level" else "-"
        skipped = str(row.skipped) if row.method == "request_level" else "-"
        forced = str(row.forced) if row.method == "request_level" else "-"
        fallback = str(row.fallback) if row.method == "request_level" else "-"
        print(
            f"| {row.run} | {row.role} | {row.trigger_events} | {row.planned} | "
            f"{accepted} | {row.interrupted} | {skipped} | {forced} | "
            f"{fallback} | {row.redispatched_overhead} | {row.completed_overhead} |"
        )


def print_run_windows(windows: Sequence[RunWindow]) -> None:
    print("Input log windows:")
    print("| run | first overhead | last overhead | observed min | selected min |")
    print("|---|---|---|---:|---:|")
    for window in windows:
        print(
            f"| {window.run} | {window.first_timestamp} | {window.last_timestamp} | "
            f"{window.observed_minutes:.2f} | "
            f"{window.selected_start_min:.2f}-{window.selected_end_min:.2f} |"
        )


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_parse_run,
        metavar="LABEL=RUN_DIR",
        help="Run label and directory; pass exactly twice.",
    )
    parser.add_argument(
        "--start-min",
        type=float,
        default=0.0,
        help="Start of each run's relative-time window (default: 0).",
    )
    parser.add_argument(
        "--end-min",
        type=float,
        default=None,
        help="End of each run's relative-time window (default: full run).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for extracted events and summary CSV files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.run) != 2:
        raise SystemExit("--run must be passed exactly twice")
    labels = [label for label, _ in args.run]
    if labels[0] == labels[1]:
        raise SystemExit("run labels must be unique")
    if args.start_min < 0:
        raise SystemExit("--start-min must be non-negative")
    if args.end_min is not None and args.end_min < args.start_min:
        raise SystemExit("--end-min must be >= --start-min")

    all_events: list[MetricEvent] = []
    effectiveness: list[RebalanceCounts] = []
    windows: list[RunWindow] = []
    for label, run_dir in args.run:
        parsed = parse_overhead_lines(run_dir)
        events = collect_metric_events(
            label,
            run_dir,
            start_min=args.start_min,
            end_min=args.end_min,
        )
        if not events:
            raise SystemExit(f"no supported overhead events found in {run_dir}")
        windows.append(
            describe_run_window(
                label,
                run_dir,
                parsed,
                start_min=args.start_min,
                end_min=args.end_min,
            )
        )
        all_events.extend(events)
        effectiveness.extend(
            collect_rebalance_counts(
                label,
                run_dir,
                parsed,
                start_min=args.start_min,
                end_min=args.end_min,
            )
        )

    summaries = summarize_metric_events(all_events)
    print_run_windows(windows)
    print_comparison(summaries, labels)
    print_rebalance_effectiveness(effectiveness)

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        _write_csv(output_dir / "overhead_events.csv", [asdict(item) for item in all_events])
        _write_csv(output_dir / "overhead_summary.csv", [asdict(item) for item in summaries])
        _write_csv(output_dir / "run_windows.csv", [asdict(item) for item in windows])
        _write_csv(
            output_dir / "rebalance_effectiveness.csv",
            [item.as_row() for item in effectiveness],
        )
        print(f"\nWrote CSV files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
