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
            Path(__file__).resolve().parents[3]
            / "psrl"
            / "workers"
            / "reward"
            / "reward_model"
            / "router.py"
        )
        spec = importlib.util.spec_from_file_location("reward_model_router_for_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.RoundRobinRewardModelRouteStrategy
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


RoundRobinRewardModelRouteStrategy = _load_round_robin_strategy()


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
