"""Helpers for elastic scaling and request-migration overhead logging."""

from __future__ import annotations

import time
from typing import Any


def extract_vllm_reprefill_s(result: Any) -> float | None:
    """Return vLLM scheduled-to-first-token time from a generated DataProto."""
    if result is None:
        return None
    meta_info = getattr(result, "meta_info", {}) or {}
    metrics = meta_info.get("vllm_metrics")
    if metrics is None:
        return None
    if hasattr(metrics, "tolist"):
        metrics = metrics.tolist()
    if isinstance(metrics, (list, tuple)):
        metrics = metrics[0] if metrics else None
    if metrics is None:
        return None

    if isinstance(metrics, dict):
        if metrics.get("prefill_time") is not None:
            try:
                return max(0.0, float(metrics["prefill_time"]))
            except (TypeError, ValueError):
                return None
        scheduled_ts = metrics.get("scheduled_ts", 0.0)
        first_token_ts = metrics.get("first_token_ts", 0.0)
    else:
        if getattr(metrics, "prefill_time", None) is not None:
            try:
                return max(0.0, float(metrics.prefill_time))
            except (TypeError, ValueError):
                return None
        scheduled_ts = getattr(metrics, "scheduled_ts", 0.0)
        first_token_ts = getattr(metrics, "first_token_ts", 0.0)
    try:
        scheduled_ts = float(scheduled_ts)
        first_token_ts = float(first_token_ts)
    except (TypeError, ValueError):
        return None
    if scheduled_ts <= 0.0 or first_token_ts < scheduled_ts:
        return None
    return first_token_ts - scheduled_ts


class RequestMigrationOverheadTracker:
    """Track abort-to-redispatch and re-prefill overhead for migrated requests."""

    def __init__(self) -> None:
        self._requests: dict[str, dict[str, Any]] = {}

    def mark_batch(self, instance_to_uids: dict, context: dict | None) -> None:
        if not context:
            return
        selected_count = max(1, int(context.get("selected_count", 1)))
        planner_s = max(0.0, float(context.get("planner_s", 0.0)))
        started_s = time.monotonic()
        for source_instance_id, uids in instance_to_uids.items():
            normalized_uids = uids if isinstance(uids, (list, tuple, set)) else [uids]
            for uid in normalized_uids:
                if uid is None:
                    continue
                self._requests[str(uid)] = {
                    **context,
                    "source_instance_id": int(source_instance_id),
                    "selected_count": selected_count,
                    "planner_s": planner_s,
                    "planner_share_s": planner_s / float(selected_count),
                    "network_started_s": started_s,
                    "requeued": False,
                    "network_s": None,
                }

    def _lookup_key(self, request_id: Any) -> str | None:
        """Resolve a router logical ID to its unique scheduler request ID.

        vLLM scheduler stats expose IDs such as ``<logical-id>-<uuid>``, while
        router callbacks retain only ``<logical-id>``. Prefer exact matching;
        use the prefix form only when it identifies one tracked request.
        """
        request_key = str(request_id)
        if request_key in self._requests:
            return request_key
        matches = [key for key in self._requests if key.startswith(f"{request_key}-")]
        if not matches:
            return None
        # A request can be selected again after vLLM assigns a new internal
        # suffix. Prefer its newest migration context instead of dropping the
        # sample when more than one historical suffix is present.
        return max(
            matches,
            key=lambda key: float(self._requests[key].get("network_started_s", 0.0)),
        )

    def mark_requeued(self, request_id: Any) -> None:
        key = self._lookup_key(request_id)
        context = self._requests.get(key) if key is not None else None
        if context is not None:
            context["requeued"] = True

    def mark_dispatched(
        self,
        request_id: Any,
        destination_instance_id: int,
    ) -> dict[str, Any] | None:
        key = self._lookup_key(request_id)
        context = self._requests.get(key) if key is not None else None
        if context is None or not context["requeued"] or context["network_s"] is not None:
            return None
        context["destination_instance_id"] = int(destination_instance_id)
        context["network_s"] = max(0.0, time.monotonic() - context["network_started_s"])
        return dict(context)

    def complete_with_status(
        self,
        request_id: Any,
        result: Any,
    ) -> tuple[dict[str, Any] | None, str]:
        """Complete a tracked migration and explain why no sample was emitted."""
        key = self._lookup_key(request_id)
        context = self._requests.get(key) if key is not None else None
        if context is None:
            return None, "not_tracked"
        if context["network_s"] is None:
            return dict(context), "not_redispatched"
        reprefill_s = extract_vllm_reprefill_s(result)
        if reprefill_s is None:
            return dict(context), "missing_vllm_prefill"
        self.discard(request_id)
        context["reprefill_s"] = reprefill_s
        context["migration_s"] = context["planner_share_s"] + context["network_s"] + reprefill_s
        return context, "completed"

    def complete(self, request_id: Any, result: Any) -> dict[str, Any] | None:
        context, status = self.complete_with_status(request_id, result)
        return context if status == "completed" else None

    def discard(self, request_id: Any) -> None:
        """Drop terminal tracking state that cannot produce a complete sample."""
        request_key = str(request_id)
        matching_keys = [key for key in self._requests if key == request_key or key.startswith(f"{request_key}-")]
        for key in matching_keys:
            self._requests.pop(key, None)
