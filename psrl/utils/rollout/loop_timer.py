"""Lightweight per-trajectory wall-clock timing accumulator for agent loops."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class LoopTimer:
    """
    Record model call and environment durations for one request.
    """

    def __init__(self) -> None:
        self.generation_s: float = 0.0
        self.env_s: float = 0.0
        self.generation_turns_s: list[float] = []
        self.env_turns_s: list[float] = []
        self._t0: float = time.perf_counter()

    @contextmanager
    def generation(self) -> Iterator[None]:
        """
        Measure one model call, including request waiting and failed calls.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.generation_s += duration
            self.generation_turns_s.append(duration)

    @contextmanager
    def env(self) -> Iterator[None]:
        """
        Measure one environment interaction, including failed interactions.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.env_s += duration
            self.env_turns_s.append(duration)

    def elapsed(self) -> float:
        """
        Return elapsed time since this timer was created.
        """
        return time.perf_counter() - self._t0

    def as_dict(self) -> dict[str, float | list[float]]:
        """
        Return a snapshot of request totals and individual turn durations.
        """
        return {
            "generation_s": self.generation_s,
            "env_s": self.env_s,
            "elapsed_s": self.elapsed(),
            "generation_turns_s": list(self.generation_turns_s),
            "env_turns_s": list(self.env_turns_s),
        }
