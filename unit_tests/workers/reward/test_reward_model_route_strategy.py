import asyncio
import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path


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

    cost_model_module = types.ModuleType("psrl.utils.cost_model_path")
    cost_model_module.resolve_cost_model_json_path = lambda *args, **kwargs: None
    stub_modules["psrl.utils.cost_model_path"] = cost_model_module

    diagnostics_module = types.ModuleType("psrl.utils.elastic_rm.diagnostics")
    diagnostics_module.log_elastic_rm_backlog_diag = lambda *args, **kwargs: None
    stub_modules["psrl.utils.elastic_rm.diagnostics"] = diagnostics_module

    itl_module = types.ModuleType("psrl.utils.elastic_rm.itl_scaling_policy")
    itl_module.ITLModelParams = object
    itl_module.compute_itl = lambda *args, **kwargs: 0.0
    itl_module.resolve_itl_model_params = lambda *args, **kwargs: None
    stub_modules["psrl.utils.elastic_rm.itl_scaling_policy"] = itl_module

    logger_module = types.ModuleType("psrl.utils.logger")
    logger_module.DualOutputHandler = object
    stub_modules["psrl.utils.logger"] = logger_module

    gen_package = types.ModuleType("psrl.workers.gen")
    gen_package.__path__ = []
    stub_modules["psrl.workers.gen"] = gen_package

    stats_module = types.ModuleType("psrl.workers.gen.stats_collector")

    class _EngineStats:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        @staticmethod
        def get_default_snapshot():
            return {}

    stats_module.EngineStats = _EngineStats
    stub_modules["psrl.workers.gen.stats_collector"] = stats_module

    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)
    try:
        module_path = (
            Path(__file__).resolve().parents[3] / "psrl" / "workers" / "reward" / "reward_model" / "router.py"
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


def _request(uid):
    return types.SimpleNamespace(non_tensor_batch={"uid": [uid]})


def test_prepare_request_migrations_accepts_only_matching_inflight_sources():
    router = ROUTER_MODULE.PSRL_RewardModelRouter.__new__(ROUTER_MODULE.PSRL_RewardModelRouter)
    router._uid_to_inflight_instance = {"active": 0, "moved": 1, "389": 6}
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
        ]
    )

    assert result == {
        "instance_to_uids": {0: ["active"], 6: ["389-b51b174a"]},
        "planned": 4,
        "accepted": 2,
        "skipped": 2,
        "skip_reasons": {"request_not_inflight": 1, "source_changed": 1},
    }
    assert router._planned_migration_destinations == {"active": 2, "389": 3}


def test_rm_planned_destination_forces_valid_target_and_invalid_target_falls_back():
    class _Strategy:
        def __init__(self):
            self.calls = []

        def route(self, request, candidates, route_kwargs):
            self.calls.append((list(candidates), dict(route_kwargs["active_loads"])))
            return min(candidates) if candidates else None

    router = ROUTER_MODULE.PSRL_RewardModelRouter.__new__(ROUTER_MODULE.PSRL_RewardModelRouter)
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

    router = ROUTER_MODULE.PSRL_RewardModelRouter.__new__(ROUTER_MODULE.PSRL_RewardModelRouter)
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
