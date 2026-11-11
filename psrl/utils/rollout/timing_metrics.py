"""
Aggregate request and turn durations for the requests consumed by a training step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean


def compute_rollout_timing_metrics(
    request_ids: Sequence,
    timings: Sequence[dict],
) -> dict[str, float]:
    """
    Summarize cumulative request durations and individual call durations in seconds.

    Multiple training samples can reference the same request. Its complete timing
    record must be identical on every sample and is counted once per step.
    """
    seen: dict[str, dict] = {}
    values = {level: {phase: [] for phase in ("rollout", "env")} for level in ("request", "turn")}
    for request_id, timing in zip(request_ids, timings, strict=True):
        uid = str(request_id)
        if uid in seen:
            if timing != seen[uid]:
                raise ValueError(f"Conflicting timing records for request {uid!r}.")
            continue
        seen[uid] = timing
        for phase, field in (("rollout", "generation_turns_s"), ("env", "env_turns_s")):
            turns = [float(duration) for duration in timing[field]]
            if any(not math.isfinite(duration) or duration < 0 for duration in turns):
                raise ValueError(f"Invalid {phase} durations for request {uid!r}: {turns!r}.")
            values["request"][phase].append(sum(turns))
            values["turn"][phase].extend(turns)

    metrics = {}
    for level, phases in values.items():
        for phase, durations in phases.items():
            prefix = f"rollout_timing/{level}/{phase}_s"
            metrics[f"{prefix}/count"] = len(durations)
            if durations:
                metrics.update(
                    {
                        f"{prefix}/mean": fmean(durations),
                        f"{prefix}/min": min(durations),
                        f"{prefix}/max": max(durations),
                    }
                )
    return metrics
