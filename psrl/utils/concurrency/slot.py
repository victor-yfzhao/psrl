"""
Limit episode concurrency with a fair worker owned permit queue.
"""

from __future__ import annotations

import asyncio


class SlotManager:
    """
    Own one episode capacity and its waiting queue for an agent worker.
    """

    def __init__(self) -> None:
        self._capacity: int | None = None
        self._semaphore: asyncio.BoundedSemaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def acquire(
        self,
        max_slots: int,
    ) -> asyncio.BoundedSemaphore | None:
        """
        Wait behind existing episodes before taking an execution permit.

        Args:
            max_slots: Worker capacity. Nonpositive values disable the limit.
                Every episode on one worker must use the same capacity.

        Returns:
            The acquired permit pool, or `None` when limiting is disabled.
        """
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("Run slots belong to one worker event loop.")
        capacity = max(0, max_slots)
        if self._capacity is None:
            self._capacity = capacity
            self._semaphore = asyncio.BoundedSemaphore(capacity) if capacity else None
        elif capacity != self._capacity:
            raise ValueError(f"Worker slot capacity cannot change from {self._capacity!r} to {capacity!r}.")
        if self._semaphore is None:
            return None
        await self._semaphore.acquire()
        return self._semaphore

    def release(
        self,
        run_slot: asyncio.BoundedSemaphore | None,
    ) -> None:
        """
        Return one occupied permit to the next waiting episode.
        """
        if run_slot is None:
            return
        if run_slot is not self._semaphore:
            raise ValueError("Run slot belongs to a different worker.")
        run_slot.release()
