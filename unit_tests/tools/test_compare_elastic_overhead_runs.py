import importlib.util
import sys
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "compare_elastic_overhead_runs.py"
    spec = importlib.util.spec_from_file_location("compare_elastic_overhead_runs", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_extracts_and_normalizes_legacy_and_request_level_overheads(tmp_path):
    legacy = tmp_path / "legacy"
    request_level = tmp_path / "request_level"
    legacy.mkdir()
    request_level.mkdir()
    (legacy / "ElasticExecutor.log").write_text(
        "2026-08-04 10:00:00,000 - x - [ELASTIC_OVERHEAD] "
        "operation=policy_planner planner_s=0.100 actions=1 reason=itl_best_scale_up_Rollout\n"
        "2026-08-04 10:00:01,000 - x - [ELASTIC_OVERHEAD] operation=scale_up "
        "planner_s=0.100 sleep_s=1.0 wakeup_s=2.0 "
        "post_scale_up_rebalance_trigger_s=0.01 execution_s=3.01 total_s=3.11\n"
        "2026-08-04 10:00:02,000 - x - [ELASTIC_OVERHEAD] "
        "operation=post_scale_up_rebalance_trigger "
        "migration_id=1:1:Rollout:model planner_s=0.002 "
        "network_trigger_s=0.01 selected=10 interrupted=8\n",
        encoding="utf-8",
    )
    (legacy / "RolloutRouter.log").write_text(
        "2026-08-04 10:00:03,000 - x - [ELASTIC_OVERHEAD] "
        "operation=post_scale_up_rebalance migration_id=1:1:Rollout:model "
        "network_s=4.0 reprefill_s=0.5 migration_s=4.502\n",
        encoding="utf-8",
    )
    (request_level / "ElasticExecutor.log").write_text(
        "2026-08-04 11:00:00,000 - x - [ELASTIC_OVERHEAD] "
        "operation=planned_request_rebalance role=Rollout planned=12 interrupted=9\n",
        encoding="utf-8",
    )
    (request_level / "RolloutRouter.log").write_text(
        "2026-08-04 11:00:01,000 - x - Prepared rollout request migrations: "
        "planned=12 accepted=9 skipped=3 skip_reasons={} sources=[0]\n"
        "2026-08-04 11:00:02,000 - x - Forced planned rollout migration: request=1 destination=2\n"
        "2026-08-04 11:00:03,000 - x - [ELASTIC_OVERHEAD] "
        "operation=post_scale_up_rebalance migration_id=2:1:Rollout:model:planned "
        "network_s=3.0 reprefill_s=0.25 migration_s=3.25\n",
        encoding="utf-8",
    )
    (request_level / "RewardModelRouter.log").write_text(
        "2026-08-04 11:00:02,500 - x - [ELASTIC_OVERHEAD] "
        "operation=request_migration_dispatch role=RewardModel "
        "migration_id=3:2:RewardModel:model:planned request_id=rm-1 "
        "planner_batch_s=0.2 planner_share_s=0.01 network_s=1.5\n",
        encoding="utf-8",
    )

    legacy_events = MODULE.collect_metric_events("legacy", legacy, start_min=0.0, end_min=None)
    request_events = MODULE.collect_metric_events("request", request_level, start_min=0.0, end_min=None)
    summaries = MODULE.summarize_metric_events(legacy_events + request_events)

    legacy_network = next(
        item
        for item in summaries
        if item.run == "legacy" and item.operation == "request_migration" and item.metric == "network_s"
    )
    request_network = next(
        item
        for item in summaries
        if item.run == "request" and item.operation == "request_migration" and item.metric == "network_s"
    )
    assert legacy_network.mean == 4.0
    assert request_network.mean == 3.0
    rm_dispatch_network = next(
        item
        for item in summaries
        if item.run == "request"
        and item.operation == "request_migration_dispatch"
        and item.role == "RewardModel"
        and item.metric == "network_s"
    )
    assert rm_dispatch_network.mean == 1.5

    legacy_parsed = MODULE.parse_overhead_lines(legacy)
    legacy_counts = MODULE.collect_rebalance_counts(
        "legacy",
        legacy,
        legacy_parsed,
        start_min=0.0,
        end_min=None,
    )
    legacy_rollout = next(item for item in legacy_counts if item.role == "Rollout")
    assert legacy_rollout.method == "legacy"
    assert legacy_rollout.planned == 10
    assert legacy_rollout.interrupted == 8

    parsed = MODULE.parse_overhead_lines(request_level)
    counts = MODULE.collect_rebalance_counts(
        "request",
        request_level,
        parsed,
        start_min=0.0,
        end_min=None,
    )
    rollout = next(item for item in counts if item.role == "Rollout")
    assert rollout.planned == 12
    assert rollout.accepted == 9
    assert rollout.interrupted == 9
    assert rollout.skipped == 3
    assert rollout.forced == 1
    assert rollout.completed_overhead == 1


def test_percentile_uses_linear_interpolation():
    assert MODULE._percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.5
