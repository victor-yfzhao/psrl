# Copyright (c) 2026, PSRL Authors.

"""Persist complete dynamic CP schedules through the training diagnostics logger."""

import json
import logging
from pathlib import Path

_LOGGER_NAME = "verl.utils.training_diagnostics"
_HANDLER_NAME = "psrl_dynamic_cp_schedule"


class _ScheduleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "training_diagnostics", {}).get("event") == "dynamic_cp_schedule"


class _ScheduleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"timestamp": record.created, **record.training_diagnostics}, sort_keys=True)


def setup_dynamic_cp_logger(logging_path: str) -> Path:
    """
    Append schedule events to one JSONL file owned by the representative worker.
    """
    path = (Path(logging_path).expanduser() / "dynamic_cp.jsonl").resolve()
    psrl_logger = logging.getLogger(_LOGGER_NAME)
    psrl_logger.setLevel(logging.INFO)
    psrl_logger.disabled = False
    for handler in list(psrl_logger.handlers):
        if handler.get_name() != _HANDLER_NAME:
            continue
        if handler.baseFilename == str(path):
            return path
        psrl_logger.removeHandler(handler)
        handler.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8", delay=True)
    handler.set_name(_HANDLER_NAME)
    handler.addFilter(_ScheduleFilter())
    handler.setFormatter(_ScheduleFormatter())
    psrl_logger.addHandler(handler)
    return path
