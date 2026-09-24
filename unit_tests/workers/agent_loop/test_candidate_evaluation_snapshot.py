import asyncio
import importlib.util
import sys
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock


def _load_rollout_router():
    repo_root = Path(__file__).resolve().parents[3]
    package_paths = {
        "psrl.workers.agent_loop": repo_root / "psrl" / "workers" / "agent_loop",
        "psrl.workers.ps": repo_root / "psrl" / "workers" / "ps",
        "psrl.workers.gen": repo_root / "psrl" / "workers" / "gen",
    }
    stub_packages = {}
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        stub_packages[name] = package
    previous_modules = {name: sys.modules.get(name) for name in stub_packages}
    sys.modules.update(stub_packages)
    try:
        module_path = package_paths["psrl.workers.agent_loop"] / "router.py"
        spec = importlib.util.spec_from_file_location("psrl.workers.agent_loop.router", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.RolloutRouter
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


RolloutRouter = _load_rollout_router()


class ThroughputOptimalRouteStrategy:
    pass


def _make_router(requests):
    router_cls = RolloutRouter.__ray_metadata__.modified_class
    router = router_cls.__new__(router_cls)
    router.n_rollout_instances = 2
    router.n_validate_instances = 1
    router.rollout_wg_size = 3
    router.currently_paused_instance_ids = {2}
    router.instance_to_version_after_sync = {0: 0, 1: 0, 2: 0}
    router.requests_to_route = SimpleNamespace(size=lambda: len(requests))
    router._rebalance_requests_to_route = deque()
    router._iter_pending_requests_in_route_order = lambda limit: iter(requests[:limit])
    router.ps_manager_handle = SimpleNamespace(
        get_ps_model_version=SimpleNamespace(remote=AsyncMock(return_value=0)),
    )
    router.route_strategy = ThroughputOptimalRouteStrategy()
    router.route_strategy.instance_to_engine_status = {}
    router._candidate_evaluation_inflight_requests = {}
    router._planned_migration_counters = {}
    return router


def test_candidate_snapshot_excludes_dedicated_validation_instances():
    router = _make_router([])

    snapshot = asyncio.run(router.get_candidate_evaluation_snapshot())

    assert [instance["instance_id"] for instance in snapshot["instances"]] == [0, 1]


def test_candidate_snapshot_excludes_validation_requests():
    train_request_0 = SimpleNamespace(meta_info={"validate": False})
    validation_request = SimpleNamespace(meta_info={"validate": True})
    train_request_1 = SimpleNamespace(meta_info={})
    router = _make_router([train_request_0, validation_request, train_request_1])
    router._candidate_evaluation_request_row = AsyncMock(
        side_effect=[
            {"request_id": "train-0"},
            {"request_id": "train-1"},
        ]
    )

    snapshot = asyncio.run(router.get_candidate_evaluation_snapshot())

    assert snapshot["pending_total"] == 2
    assert snapshot["pending_requests"] == [
        {"request_id": "train-0"},
        {"request_id": "train-1"},
    ]
    routed_requests = [call.args[0] for call in router._candidate_evaluation_request_row.await_args_list]
    assert routed_requests == [train_request_0, train_request_1]


def test_rollout_prepare_request_migrations_rejects_validation_destination():
    router = _make_router([])
    router.incomplete_request_to_instance = {1: 0}
    router.instance_to_inflight_request_ids = {0: [1], 1: [], 2: []}
    router.request_futures = {1: SimpleNamespace(done=lambda: False)}
    router._planned_migration_destinations = {}
    router._planned_migration_counters = {
        "planned": 0,
        "accepted": 0,
        "skipped": 0,
        "forced": 0,
        "fallback": 0,
    }

    result = router.prepare_request_migrations(
        [
            {
                "request_id": "1-deadbeef",
                "source_instance_id": 0,
                "destination_instance_id": 2,
            }
        ]
    )

    assert result == {
        "instance_to_uids": {},
        "planned": 1,
        "accepted": 0,
        "skipped": 1,
        "skip_reasons": {"invalid_destination": 1},
    }


def test_rollout_prepare_atomically_starts_exclusive_rebalance_tracking():
    router = _make_router([])
    router._exclusive_rebalance_migration_queue_enabled = True
    router._rebalance_pending_request_ids = set()
    router._rebalance_queued_request_ids = set()
    router._rebalance_dispatching_request_ids = set()
    router.instance_to_inflight_request_ids = {0: [1], 1: [], 2: []}
    router.request_futures = {1: SimpleNamespace(done=lambda: False)}
    router._planned_migration_destinations = {}
    router._planned_migration_counters = {
        "planned": 0,
        "accepted": 0,
        "skipped": 0,
        "forced": 0,
        "fallback": 0,
    }
    tracked = []
    router._migration_overhead = SimpleNamespace(
        mark_batch=lambda instance_to_uids, context: tracked.append((instance_to_uids, context))
    )
    context = {"migration_id": "7:1:Rollout:model:planned"}

    result = router.prepare_request_migrations(
        [
            {
                "request_id": "1-deadbeef",
                "source_instance_id": 0,
                "destination_instance_id": 1,
            }
        ],
        context,
    )

    assert result["instance_to_uids"] == {0: ["1-deadbeef"]}
    assert tracked == [({0: ["1-deadbeef"]}, context)]
    assert router._rebalance_pending_request_ids == {"1"}
    assert router._exclusive_rebalance_active()


def test_rollout_exclusive_rebalance_queue_preserves_interrupt_fifo_order():
    class _Strategy:
        def __init__(self):
            self.forced = []

        def force_route_unchecked(self, request, destination):
            self.forced.append((request.non_tensor_batch["uid"][0], destination))
            return destination

    async def scenario():
        router = _make_router([])
        router._exclusive_rebalance_migration_queue_enabled = True
        router._rebalance_requests_to_route = deque()
        router._rebalance_pending_request_ids = set()
        router._rebalance_queued_request_ids = set()
        router._rebalance_dispatching_request_ids = set()
        router._planned_migration_destinations = {"2": 1, "1": 0}
        router._planned_migration_counters = {"forced": 0}
        router.requests_to_route = SimpleNamespace(size=lambda: 3)
        router.route_strategy = _Strategy()
        router.incomplete_request_to_instance = {}
        router._pause_routing = False
        router._is_routing = False
        router.ps_manager_handle = SimpleNamespace(
            check_aborted_requests=SimpleNamespace(remote=AsyncMock(return_value=False)),
        )

        dispatched = []

        async def route_single(request, destination):
            request_id = request.non_tensor_batch["uid"][0]
            dispatched.append((request_id, destination))
            router._finish_exclusive_rebalance_request(request_id, reason="redispatched")

        router._route_single_request = route_single
        request_2 = SimpleNamespace(non_tensor_batch={"uid": [2]})
        request_1 = SimpleNamespace(non_tensor_batch={"uid": [1]})
        router._start_exclusive_rebalance(
            {0: ["2-deadbeef", "1-deadbeef"]},
            {"migration_id": "7:1:Rollout:model:planned"},
        )

        assert router._enqueue_exclusive_rebalance_request(request_2)
        assert router._enqueue_exclusive_rebalance_request(request_1)
        assert router.get_pending_request_count() == 5
        assert [request.non_tensor_batch["uid"][0] for request in router._rebalance_requests_to_route] == [2, 1]

        await router._dispatch_exclusive_rebalance_requests()
        await asyncio.sleep(0)
        assert dispatched == [(2, 1)]

        await router._dispatch_exclusive_rebalance_requests()
        await asyncio.sleep(0)
        assert dispatched == [(2, 1), (1, 0)]
        assert router.route_strategy.forced == [(2, 1), (1, 0)]
        assert not router._exclusive_rebalance_active()

    asyncio.run(scenario())


def test_rollout_rebalance_settlement_wakes_next_dispatch_without_poll_delay():
    class _Strategy:
        def force_route_unchecked(self, request, destination):
            return destination

    async def scenario():
        router = _make_router([])
        router._exclusive_rebalance_migration_queue_enabled = True
        router._rebalance_requests_to_route = deque()
        router._rebalance_pending_request_ids = set()
        router._rebalance_queued_request_ids = set()
        router._rebalance_dispatching_request_ids = set()
        router._planned_migration_destinations = {"2": 1, "1": 0}
        router._planned_migration_counters = {"forced": 0}
        router.requests_to_route = SimpleNamespace(size=lambda: 0)
        router.route_strategy = _Strategy()
        router.incomplete_request_to_instance = {}
        router._pause_routing = False
        router._is_routing = False
        router.ps_manager_handle = SimpleNamespace(
            check_aborted_requests=SimpleNamespace(remote=AsyncMock(return_value=False)),
        )
        router._routing_event_loop = asyncio.get_running_loop()
        router._routing_wakeup_event = asyncio.Event()

        settle_first = asyncio.Event()
        dispatched = []

        async def route_single(request, destination):
            request_id = request.non_tensor_batch["uid"][0]
            dispatched.append((request_id, destination))
            if request_id == 2:
                await settle_first.wait()
            router._finish_exclusive_rebalance_request(request_id, reason="redispatched")

        router._route_single_request = route_single
        router._start_exclusive_rebalance(
            {0: ["2-deadbeef", "1-deadbeef"]},
            {"migration_id": "7:1:Rollout:model:planned"},
        )
        assert router._enqueue_exclusive_rebalance_request(
            SimpleNamespace(non_tensor_batch={"uid": [2]})
        )
        assert router._enqueue_exclusive_rebalance_request(
            SimpleNamespace(non_tensor_batch={"uid": [1]})
        )

        await router._dispatch_exclusive_rebalance_requests()
        await asyncio.sleep(0)
        assert dispatched == [(2, 1)]

        router._routing_wakeup_event.clear()
        waiter = asyncio.create_task(router._wait_for_routing_iteration(10.0))
        await asyncio.sleep(0)
        assert not waiter.done()
        settle_first.set()
        await asyncio.wait_for(waiter, timeout=0.1)

        await router._dispatch_exclusive_rebalance_requests()
        await asyncio.sleep(0)
        assert dispatched == [(2, 1), (1, 0)]
        assert not router._exclusive_rebalance_active()

    asyncio.run(scenario())


def test_rollout_rebalance_completion_wakes_waiter_without_poll_delay():
    async def scenario():
        router = _make_router([])
        router._exclusive_rebalance_migration_queue_enabled = True
        router._rebalance_pending_request_ids = {"2"}
        router._rebalance_queued_request_ids = {"2"}
        router._rebalance_dispatching_request_ids = {"2"}
        router._planned_migration_destinations = {"2": 1}
        router._rebalance_completion_waiters = set()
        router._routing_event_loop = asyncio.get_running_loop()
        router._routing_wakeup_event = asyncio.Event()

        waiter = asyncio.create_task(router.wait_for_exclusive_rebalance())
        await asyncio.sleep(0)
        assert not waiter.done()

        router._finish_exclusive_rebalance_request(2, reason="redispatched")
        await asyncio.wait_for(waiter, timeout=0.1)
        assert not router._exclusive_rebalance_active()
        assert router._routing_wakeup_event.is_set()

    asyncio.run(scenario())


def test_rollout_unreachable_rebalance_target_requeues_without_migration_timing():
    class _NormalQueue:
        def __init__(self):
            self.requests = []

        def put(self, request):
            self.requests.append(request)

        def size(self):
            return len(self.requests)

    async def scenario():
        router = _make_router([])
        router._exclusive_rebalance_migration_queue_enabled = True
        router._rebalance_requests_to_route = deque()
        router._rebalance_pending_request_ids = set()
        router._rebalance_queued_request_ids = set()
        router._rebalance_dispatching_request_ids = set()
        router._planned_migration_destinations = {"2": 1}
        router._planned_migration_counters = {"forced": 0, "fallback": 0}
        router.currently_paused_instance_ids = {1, 2}
        router.requests_to_route = _NormalQueue()
        discarded = []
        router._migration_overhead = SimpleNamespace(
            discard=lambda request_id: discarded.append(request_id)
        )
        router._pause_routing = False
        router._is_routing = False
        router.ps_manager_handle = SimpleNamespace(
            check_aborted_requests=SimpleNamespace(remote=AsyncMock(return_value=False)),
        )
        router._choose_new_rollout_instance = AsyncMock(return_value=None)

        request = SimpleNamespace(non_tensor_batch={"uid": [2]})
        router._start_exclusive_rebalance(
            {0: ["2-deadbeef"]},
            {"migration_id": "7:1:Rollout:model:planned"},
        )
        assert router._enqueue_exclusive_rebalance_request(request)

        await router._dispatch_exclusive_rebalance_requests()

        assert router.requests_to_route.requests == [request]
        assert discarded == ["2"]
        assert router._planned_migration_counters["fallback"] == 1
        assert not router._exclusive_rebalance_active()

    asyncio.run(scenario())


def test_rollout_unreachable_rebalance_target_uses_normal_route_when_available():
    async def scenario():
        normal_queue = []
        router = _make_router([])
        router._exclusive_rebalance_migration_queue_enabled = True
        router._rebalance_requests_to_route = deque()
        router._rebalance_pending_request_ids = set()
        router._rebalance_queued_request_ids = set()
        router._rebalance_dispatching_request_ids = set()
        router._planned_migration_destinations = {"2": 1}
        router._planned_migration_counters = {"forced": 0, "fallback": 0}
        router.currently_paused_instance_ids = {1, 2}
        router.requests_to_route = SimpleNamespace(
            put=normal_queue.append,
            size=lambda: len(normal_queue),
        )
        discarded = []
        router._migration_overhead = SimpleNamespace(discard=discarded.append)
        router._pause_routing = False
        router._is_routing = False
        router.incomplete_request_to_instance = {}
        router.ps_manager_handle = SimpleNamespace(
            check_aborted_requests=SimpleNamespace(remote=AsyncMock(return_value=False)),
        )
        router._choose_new_rollout_instance = AsyncMock(return_value=0)
        dispatched = []

        async def route_single(request, destination):
            dispatched.append((request, destination))

        router._route_single_request = route_single
        request = SimpleNamespace(non_tensor_batch={"uid": [2]})
        router._start_exclusive_rebalance(
            {0: ["2-deadbeef"]},
            {"migration_id": "7:1:Rollout:model:planned"},
        )
        assert router._enqueue_exclusive_rebalance_request(request)

        await router._dispatch_exclusive_rebalance_requests()
        await asyncio.sleep(0)

        assert dispatched == [(request, 0)]
        assert normal_queue == []
        assert discarded == ["2"]
        assert router.incomplete_request_to_instance[2] == 0
        assert not router._exclusive_rebalance_active()

    asyncio.run(scenario())


def test_rollout_normal_fallback_exception_requeues_request():
    async def scenario():
        normal_queue = []
        router = _make_router([])
        router._exclusive_rebalance_migration_queue_enabled = True
        router._rebalance_requests_to_route = deque()
        router._rebalance_pending_request_ids = set()
        router._rebalance_queued_request_ids = set()
        router._rebalance_dispatching_request_ids = set()
        router._planned_migration_destinations = {"2": 1}
        router._planned_migration_counters = {"forced": 0, "fallback": 0}
        router.currently_paused_instance_ids = {1, 2}
        router.requests_to_route = SimpleNamespace(
            put=normal_queue.append,
            size=lambda: len(normal_queue),
        )
        router._migration_overhead = SimpleNamespace(discard=lambda request_id: None)
        router._pause_routing = False
        router._is_routing = False
        router.ps_manager_handle = SimpleNamespace(
            check_aborted_requests=SimpleNamespace(remote=AsyncMock(return_value=False)),
        )
        router._choose_new_rollout_instance = AsyncMock(side_effect=RuntimeError("route failed"))

        request = SimpleNamespace(non_tensor_batch={"uid": [2]})
        router._start_exclusive_rebalance(
            {0: ["2-deadbeef"]},
            {"migration_id": "7:1:Rollout:model:planned"},
        )
        assert router._enqueue_exclusive_rebalance_request(request)

        await router._dispatch_exclusive_rebalance_requests()

        assert normal_queue == [request]
        assert not router._exclusive_rebalance_active()

    asyncio.run(scenario())


def test_rollout_deferred_interrupt_timing_is_not_logged_at_requeue():
    router = _make_router([])
    overhead = {"interrupt_s": 0.1}
    router._migration_overhead = SimpleNamespace(mark_requeued=lambda request_id: overhead)
    logged = []
    router._log_migration_interrupt = lambda request_id, sample: logged.append((request_id, sample))

    router._mark_migration_requeued("2", defer_log=True)

    assert logged == []


def test_rollout_exclusive_rebalance_queue_disabled_preserves_normal_requeue():
    router = _make_router([])
    router._exclusive_rebalance_migration_queue_enabled = False
    router._rebalance_pending_request_ids = set()
    router._rebalance_requests_to_route = deque()
    request = SimpleNamespace(non_tensor_batch={"uid": [2]})

    router._start_exclusive_rebalance(
        {0: ["2-deadbeef"]},
        {"migration_id": "7:1:Rollout:model:planned"},
    )

    assert not router._exclusive_rebalance_active()
    assert not router._enqueue_exclusive_rebalance_request(request)
