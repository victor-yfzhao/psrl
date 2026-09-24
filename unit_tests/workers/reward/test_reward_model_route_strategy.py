import asyncio
import importlib.util
import logging
import sys
import types
from collections import Counter, deque
from pathlib import Path

from omegaconf import OmegaConf


def _load_round_robin_strategy():
    stub_modules = {}

    ray_module = types.ModuleType("ray")

    def _remote(*args, **kwargs):
        if args and isinstance(args[0], type):
            return args[0]
        return lambda cls: cls

    ray_module.remote = _remote
    ray_module.method = lambda *args, **kwargs: lambda func: func
    ray_module.actor = types.SimpleNamespace(ActorHandle=object)
    stub_modules["ray"] = ray_module

    omegaconf_module = types.ModuleType("omegaconf")
    omegaconf_module.DictConfig = dict
    stub_modules["omegaconf"] = omegaconf_module

    verl_module = types.ModuleType("verl")
    verl_module.DataProto = object
    stub_modules["verl"] = verl_module

    cost_model_module = types.ModuleType("pivotrl.utils.cost_model_path")
    cost_model_module.resolve_cost_model_json_path = lambda *args, **kwargs: None
    stub_modules["pivotrl.utils.cost_model_path"] = cost_model_module

    diagnostics_module = types.ModuleType("pivotrl.utils.elastic_rm.diagnostics")
    diagnostics_module.log_elastic_rm_backlog_diag = lambda *args, **kwargs: None
    stub_modules["pivotrl.utils.elastic_rm.diagnostics"] = diagnostics_module

    itl_module = types.ModuleType("pivotrl.utils.elastic_rm.itl_scaling_policy")
    itl_module.ITLModelParams = object
    itl_module.compute_itl = lambda *args, **kwargs: 0.0
    itl_module.resolve_itl_model_params = lambda *args, **kwargs: None
    stub_modules["pivotrl.utils.elastic_rm.itl_scaling_policy"] = itl_module

    logger_module = types.ModuleType("pivotrl.utils.logger")
    logger_module.DualOutputHandler = object
    stub_modules["pivotrl.utils.logger"] = logger_module

    gen_package = types.ModuleType("pivotrl.workers.gen")
    gen_package.__path__ = []
    stub_modules["pivotrl.workers.gen"] = gen_package

    stats_module = types.ModuleType("pivotrl.workers.gen.stats_collector")

    class _EngineStats:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @staticmethod
        def get_default_snapshot():
            return {}

    stats_module.EngineStats = _EngineStats
    stub_modules["pivotrl.workers.gen.stats_collector"] = stats_module

    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)
    try:
        module_path = (
            Path(__file__).resolve().parents[3] / "pivotrl" / "workers" / "reward" / "reward_model" / "router.py"
        )
        spec = importlib.util.spec_from_file_location("reward_model_router_for_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


ROUTER_MODULE = _load_round_robin_strategy()
RoundRobinRewardModelRouteStrategy = ROUTER_MODULE.RoundRobinRewardModelRouteStrategy


def _route_counts(strategy, candidates: list[int], request_count: int) -> Counter:
    return Counter(strategy.route(object(), candidates=candidates) for _ in range(request_count))


def test_round_robin_balances_across_dynamic_candidate_sets():
    strategy = RoundRobinRewardModelRouteStrategy(n_instances=22)
    main_candidates = list(range(18))
    all_candidates = list(range(22))

    assert _route_counts(strategy, main_candidates, 36) == Counter({idx: 2 for idx in main_candidates})
    assert _route_counts(strategy, all_candidates, 44) == Counter({idx: 2 for idx in all_candidates})
    assert _route_counts(strategy, main_candidates, 36) == Counter({idx: 2 for idx in main_candidates})


def test_round_robin_returns_none_without_candidates():
    strategy = RoundRobinRewardModelRouteStrategy(n_instances=22)

    assert strategy.route(object(), candidates=[]) is None


def test_rm_router_reads_exclusive_rebalance_switch_from_nested_dictconfig(monkeypatch):
    monkeypatch.setattr(
        ROUTER_MODULE,
        "DualOutputHandler",
        lambda *args, **kwargs: logging.NullHandler(),
    )
    config = OmegaConf.create(
        {
            "pivotrl": {
                "deployment": {
                    "elastic_rm": {
                        "itl_policy": {
                            "exclusive_rebalance_migration_queue": True,
                        }
                    }
                },
                "logging_path": "/tmp",
            },
            "data": {"max_prompt_length": 128},
        }
    )

    router = ROUTER_MODULE.PivotRL_RewardModelRouter(
        worker_handles=[],
        worker_groups=None,
        config=config,
    )

    assert router._exclusive_rebalance_migration_queue_enabled is True


def _request(uid):
    return types.SimpleNamespace(non_tensor_batch={"uid": [uid]})


def test_prepare_request_migrations_accepts_only_matching_inflight_sources():
    router = ROUTER_MODULE.PivotRL_RewardModelRouter.__new__(ROUTER_MODULE.PivotRL_RewardModelRouter)
    router._uid_to_inflight_instance = {"active": 0, "moved": 1, "389": 6, "invalid-target": 0}
    router.request_counts = {instance_id: 0 for instance_id in range(7)}
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
                "request_id": "active",
                "source_instance_id": 0,
                "destination_instance_id": 2,
            },
            {
                "request_id": "completed",
                "source_instance_id": 0,
                "destination_instance_id": 2,
            },
            {
                "request_id": "moved",
                "source_instance_id": 0,
                "destination_instance_id": 2,
            },
            {
                "request_id": "389-b51b174a",
                "source_instance_id": 6,
                "destination_instance_id": 3,
            },
            {
                "request_id": "invalid-target",
                "source_instance_id": 0,
                "destination_instance_id": 7,
            },
        ]
    )

    assert result == {
        "instance_to_uids": {0: ["active"], 6: ["389-b51b174a"]},
        "planned": 5,
        "accepted": 2,
        "skipped": 3,
        "skip_reasons": {
            "request_not_inflight": 1,
            "source_changed": 1,
            "invalid_destination": 1,
        },
    }
    assert router._planned_migration_destinations == {"active": 2, "389": 3}


def test_rm_planned_destination_forces_valid_target_and_invalid_target_falls_back():
    class _Strategy:
        def __init__(self):
            self.calls = []

        def route(self, request, candidates, route_kwargs):
            self.calls.append((list(candidates), dict(route_kwargs["active_loads"])))
            return min(candidates) if candidates else None

    router = ROUTER_MODULE.PivotRL_RewardModelRouter.__new__(ROUTER_MODULE.PivotRL_RewardModelRouter)
    router.paused_worker_indices = set()
    router.worker_probe_backoff_until = {}
    router.waiting_admission_cap = None
    router.max_concurrent_requests_per_instance = 4
    router.request_counts = {0: 0, 1: 0}
    router.instance_to_engine_status = {}
    router.route_strategy = _Strategy()
    router._planned_migration_destinations = {"forced": 1}
    router._planned_migration_counters = {
        "planned": 0,
        "accepted": 0,
        "skipped": 0,
        "forced": 0,
        "fallback": 0,
    }

    assert router._select_worker_for_request(_request("forced"), {0: 0, 1: 0}) == 1
    assert router.request_counts == {0: 0, 1: 1}
    assert router.route_strategy.calls == []
    assert router._planned_migration_destinations == {}

    router._planned_migration_destinations["fallback"] = 9
    assert router._select_worker_for_request(_request("fallback"), {0: 0, 1: 1}) == 0
    assert router.route_strategy.calls == [([0, 1], {0: 0, 1: 1})]
    assert router._planned_migration_destinations == {}
    assert router._planned_migration_counters["fallback"] == 1


def test_rm_interrupted_result_requeues_before_migration_completion():
    class _Tracker:
        def __init__(self):
            self.requeued = []
            self.completed = []

        def mark_dispatched(self, request_uid, worker_idx):
            return None

        def mark_requeued(self, request_uid):
            self.requeued.append(request_uid)

        def complete_with_status(self, request_uid, result):
            self.completed.append(request_uid)
            return None, "not_tracked"

    router = ROUTER_MODULE.PivotRL_RewardModelRouter.__new__(ROUTER_MODULE.PivotRL_RewardModelRouter)
    tracker = _Tracker()
    request = types.SimpleNamespace(non_tensor_batch={"uid": ["42"]})
    result = types.SimpleNamespace(
        non_tensor_batch={"uid": ["42"], "interrupted": [True]},
        meta_info={},
    )
    requeued = []
    router.request_futures = {"request-key": object()}
    router._interrupt_routing = False
    router.worker_handles = [object()]
    router._migration_overhead = tracker
    router.retry_delay = 0.0
    router._release_strategy_worker = lambda *args: None
    router._enqueue_request = lambda request_key, queued: requeued.append((request_key, queued))

    async def _generate_with_worker(*args):
        return result

    router._generate_with_worker = _generate_with_worker

    asyncio.run(router._route_single_request("request-key", request, 0, 0))

    assert tracker.requeued == ["42"]
    assert tracker.completed == []
    assert requeued == [("request-key", result)]


def test_rm_exclusive_rebalance_queue_is_fifo_and_forces_simulated_destinations():
    class _Strategy:
        def __init__(self):
            self.forced = []

        def force_route_unchecked(self, request, destination):
            self.forced.append((request.non_tensor_batch["uid"][0], destination))
            return destination

    async def scenario():
        router = ROUTER_MODULE.PivotRL_RewardModelRouter.__new__(ROUTER_MODULE.PivotRL_RewardModelRouter)
        router._exclusive_rebalance_migration_queue_enabled = True
        router._rebalance_requests_to_route = deque()
        router._rebalance_pending_request_ids = set()
        router._rebalance_queued_request_keys = set()
        router._rebalance_dispatching_request_keys = set()
        router._planned_migration_destinations = {"2": 1, "1": 0}
        router._planned_migration_counters = {
            "planned": 2,
            "accepted": 2,
            "skipped": 0,
            "forced": 0,
            "fallback": 0,
        }
        router._pending_count = 0
        router._interrupt_routing = False
        router.request_counts = {0: 0, 1: 0}
        router.paused_worker_indices = set()
        router.worker_probe_backoff_until = {}
        router.route_strategy = _Strategy()
        router.request_futures = {"request-2": object(), "request-1": object()}
        router._request_to_strategy_instance = {}
        router._candidate_evaluation_inflight_requests = {}
        router._uid_to_inflight_instance = {}
        router._signal_routing_update = lambda: None

        dispatched = []

        async def route_single(request_key, request, destination, inflight_at_dispatch):
            dispatched.append((request_key, request.non_tensor_batch["uid"][0], destination))
            for request_uid in router._request_uid_values(request):
                router._finish_exclusive_rebalance_request(request_uid, reason="redispatched")
            router._rebalance_dispatching_request_keys.discard(request_key)

        router._route_single_request = route_single
        router._start_exclusive_rebalance(
            {0: ["2-deadbeef", "1-deadbeef"]},
            {"migration_id": "7:1:RewardModel:model:planned"},
        )
        assert router._enqueue_exclusive_rebalance_request("request-2", _request("2"))
        assert router._enqueue_exclusive_rebalance_request("request-1", _request("1"))
        assert router.get_pending_request_count() == 2

        assert await router._dispatch_exclusive_rebalance_requests()
        await asyncio.sleep(0)
        assert dispatched == [("request-2", "2", 1)]

        assert await router._dispatch_exclusive_rebalance_requests()
        await asyncio.sleep(0)

        assert dispatched == [("request-2", "2", 1), ("request-1", "1", 0)]
        assert router.route_strategy.forced == [("2", 1), ("1", 0)]
        assert router.request_counts == {0: 1, 1: 1}
        assert not router._exclusive_rebalance_active()

    asyncio.run(scenario())


def test_rm_unreachable_rebalance_target_requeues_without_migration_timing():
    async def scenario():
        router = ROUTER_MODULE.PivotRL_RewardModelRouter.__new__(ROUTER_MODULE.PivotRL_RewardModelRouter)
        router._exclusive_rebalance_migration_queue_enabled = True
        router._rebalance_requests_to_route = deque()
        router._rebalance_pending_request_ids = set()
        router._rebalance_queued_request_keys = set()
        router._rebalance_dispatching_request_keys = set()
        router._planned_migration_destinations = {"2": 1}
        router._planned_migration_counters = {
            "planned": 1,
            "accepted": 1,
            "skipped": 0,
            "forced": 0,
            "fallback": 0,
        }
        router._pending_count = 0
        router._interrupt_routing = False
        router.request_counts = {0: 0, 1: 0}
        router.paused_worker_indices = {1}
        router.request_futures = {"request-2": object()}
        router._signal_routing_update = lambda: None
        discarded = []
        router._migration_overhead = types.SimpleNamespace(
            discard=lambda request_uid: discarded.append(request_uid)
        )
        router._get_cached_active_loads = lambda: asyncio.sleep(0, result={0: 0})
        router._select_worker_for_request = lambda request, active_loads: None
        normally_queued = []
        router._enqueue_request = lambda request_key, request: normally_queued.append(
            (request_key, request)
        )

        request = _request("2")
        router._start_exclusive_rebalance(
            {0: ["2-deadbeef"]},
            {"migration_id": "7:1:RewardModel:model:planned"},
        )
        assert router._enqueue_exclusive_rebalance_request("request-2", request)

        assert await router._dispatch_exclusive_rebalance_requests()

        assert normally_queued == [("request-2", request)]
        assert discarded == ["2"]
        assert router._planned_migration_counters["fallback"] == 1
        assert not router._exclusive_rebalance_active()

    asyncio.run(scenario())


def test_rm_unreachable_rebalance_target_uses_normal_route_when_available():
    async def scenario():
        router = ROUTER_MODULE.PivotRL_RewardModelRouter.__new__(ROUTER_MODULE.PivotRL_RewardModelRouter)
        router._exclusive_rebalance_migration_queue_enabled = True
        router._rebalance_requests_to_route = deque()
        router._rebalance_pending_request_ids = set()
        router._rebalance_queued_request_keys = set()
        router._rebalance_dispatching_request_keys = set()
        router._planned_migration_destinations = {"2": 1}
        router._planned_migration_counters = {
            "planned": 1,
            "accepted": 1,
            "skipped": 0,
            "forced": 0,
            "fallback": 0,
        }
        router._pending_count = 0
        router._interrupt_routing = False
        router.request_counts = {0: 0, 1: 0}
        router.paused_worker_indices = {1}
        router.request_futures = {"request-2": object()}
        router._signal_routing_update = lambda: None
        router._request_to_strategy_instance = {}
        router._candidate_evaluation_inflight_requests = {}
        router._uid_to_inflight_instance = {}
        discarded = []
        router._migration_overhead = types.SimpleNamespace(discard=discarded.append)
        router._get_cached_active_loads = lambda: asyncio.sleep(0, result={0: 0})
        router._select_worker_for_request = lambda request, active_loads: 0
        normally_queued = []
        router._enqueue_request = lambda request_key, request: normally_queued.append((request_key, request))
        dispatched = []

        async def route_single(request_key, request, destination, inflight_at_dispatch):
            dispatched.append((request_key, request, destination, inflight_at_dispatch))

        router._route_single_request = route_single
        request = _request("2")
        router._start_exclusive_rebalance(
            {0: ["2-deadbeef"]},
            {"migration_id": "7:1:RewardModel:model:planned"},
        )
        assert router._enqueue_exclusive_rebalance_request("request-2", request)

        assert await router._dispatch_exclusive_rebalance_requests()
        await asyncio.sleep(0)

        assert dispatched == [("request-2", request, 0, 0)]
        assert normally_queued == []
        assert discarded == ["2"]
        assert not router._exclusive_rebalance_active()

    asyncio.run(scenario())


def test_rm_normal_fallback_exception_requeues_request():
    async def scenario():
        router = ROUTER_MODULE.PivotRL_RewardModelRouter.__new__(ROUTER_MODULE.PivotRL_RewardModelRouter)
        router._exclusive_rebalance_migration_queue_enabled = True
        router._rebalance_requests_to_route = deque()
        router._rebalance_pending_request_ids = set()
        router._rebalance_queued_request_keys = set()
        router._rebalance_dispatching_request_keys = set()
        router._planned_migration_destinations = {"2": 1}
        router._planned_migration_counters = {
            "planned": 1,
            "accepted": 1,
            "skipped": 0,
            "forced": 0,
            "fallback": 0,
        }
        router._pending_count = 0
        router._interrupt_routing = False
        router.request_counts = {0: 0, 1: 0}
        router.paused_worker_indices = {1}
        router.request_futures = {"request-2": object()}
        router._signal_routing_update = lambda: None
        router._migration_overhead = types.SimpleNamespace(discard=lambda request_uid: None)
        router._get_cached_active_loads = lambda: asyncio.sleep(0, result={0: 0})

        def raise_route_error(request, active_loads):
            raise RuntimeError("route failed")

        router._select_worker_for_request = raise_route_error
        normally_queued = []
        router._enqueue_request = lambda request_key, request: normally_queued.append((request_key, request))
        request = _request("2")
        router._start_exclusive_rebalance(
            {0: ["2-deadbeef"]},
            {"migration_id": "7:1:RewardModel:model:planned"},
        )
        assert router._enqueue_exclusive_rebalance_request("request-2", request)

        assert await router._dispatch_exclusive_rebalance_requests()

        assert normally_queued == [("request-2", request)]
        assert not router._exclusive_rebalance_active()

    asyncio.run(scenario())


def test_rm_deferred_interrupt_timing_is_not_logged_at_requeue():
    router = ROUTER_MODULE.PivotRL_RewardModelRouter.__new__(ROUTER_MODULE.PivotRL_RewardModelRouter)
    overhead = {"interrupt_s": 0.1}
    router._migration_overhead = types.SimpleNamespace(mark_requeued=lambda request_uid: overhead)
    logged = []
    router._log_migration_interrupt = lambda request_uid, sample: logged.append((request_uid, sample))

    router._mark_migration_requeued("2", defer_log=True)

    assert logged == []


def test_rm_exclusive_rebalance_queue_disabled_preserves_normal_requeue():
    router = ROUTER_MODULE.PivotRL_RewardModelRouter.__new__(ROUTER_MODULE.PivotRL_RewardModelRouter)
    router._exclusive_rebalance_migration_queue_enabled = False
    router._rebalance_pending_request_ids = set()
    router._rebalance_requests_to_route = deque()

    router._start_exclusive_rebalance(
        {0: ["2-deadbeef"]},
        {"migration_id": "7:1:RewardModel:model:planned"},
    )

    assert not router._exclusive_rebalance_active()
    assert not router._enqueue_exclusive_rebalance_request("request-2", _request("2"))
