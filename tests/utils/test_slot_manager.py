"""
Preserve episode admission order and capacity through cancellation.
"""

import asyncio

import pytest
from psrl.utils.concurrency import SlotManager

pytestmark = pytest.mark.cpu_test


@pytest.mark.asyncio
async def test_later_episodes_cannot_overtake_queued_episodes():
    slots = SlotManager()
    holder = await slots.acquire(1)
    order = []

    async def episode(index):
        token = await slots.acquire(1)
        try:
            order.append(index)
            await asyncio.sleep(0)
        finally:
            slots.release(token)

    older = [asyncio.create_task(episode(index)) for index in range(3)]
    await asyncio.sleep(0)
    slots.release(holder)
    later = asyncio.create_task(episode(3))
    await asyncio.wait_for(asyncio.gather(*older, later), timeout=1)

    assert order == [0, 1, 2, 3], "Later arrivals must not steal a permit reserved for an older waiter."


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity", [0, 1, 4])
async def test_worker_capacity_is_preserved_under_contention(capacity):
    slots = SlotManager()
    gate = asyncio.Event()
    active = 0
    peak = 0

    async def episode():
        nonlocal active, peak
        token = await slots.acquire(capacity)
        try:
            active += 1
            peak = max(peak, active)
            await gate.wait()
        finally:
            active -= 1
            slots.release(token)

    tasks = [asyncio.create_task(episode()) for _ in range(40)]
    await asyncio.sleep(0)
    assert active == (capacity or 40), "Only the configured number of episodes may enter the runner."
    gate.set()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
    assert peak == (capacity or 40) and active == 0, "Every occupied permit must return after completion."


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_after_release", [False, True])
async def test_cancelled_waiters_do_not_leak_or_steal_capacity(cancel_after_release):
    slots = SlotManager()
    holder = await slots.acquire(1)
    waiting = asyncio.create_task(slots.acquire(1))
    await asyncio.sleep(0)
    if cancel_after_release:
        slots.release(holder)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    if not cancel_after_release:
        slots.release(holder)

    token = await asyncio.wait_for(slots.acquire(1), timeout=1)
    slots.release(token)


@pytest.mark.asyncio
async def test_slot_ownership_is_local_to_one_worker():
    first, second = SlotManager(), SlotManager()
    first_token = await first.acquire(1)
    second_token = await asyncio.wait_for(second.acquire(1), timeout=1)
    with pytest.raises(ValueError, match="different worker"):
        first.release(second_token)
    first.release(first_token)
    second.release(second_token)


@pytest.mark.asyncio
async def test_double_release_is_rejected():
    slots = SlotManager()
    token = await slots.acquire(1)
    slots.release(token)
    with pytest.raises(ValueError):
        slots.release(token)


@pytest.mark.asyncio
async def test_one_worker_cannot_silently_create_pools_with_different_limits():
    slots = SlotManager()
    token = await slots.acquire(1)
    slots.release(token)
    with pytest.raises(ValueError, match="capacity"):
        await slots.acquire(2)


def test_slot_pool_cannot_be_reused_by_another_event_loop():
    slots = SlotManager()

    async def episode():
        token = await slots.acquire(1)
        slots.release(token)

    asyncio.run(episode())
    with pytest.raises(RuntimeError, match="event loop"):
        asyncio.run(episode())
