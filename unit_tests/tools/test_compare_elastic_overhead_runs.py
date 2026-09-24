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
        "operation=policy_input engine_status_s=0.010 router_backlog_s=0.020 "
        "request_snapshot_s=0.030 trainer_hint_s=0.010 signal_build_s=0.005 "
        "signal_logging_s=0.002 input_other_s=0.003 input_total_s=0.080\n"
        "2026-08-04 10:00:00,100 - x - [ELASTIC_OVERHEAD] "
        "operation=policy_planner planner_s=0.100 state_analysis_s=0.005 "
        "candidate_ordering_s=0.005 candidate_set_construction_s=0.010 "
        "simulation_input_preparation_s=0.010 rebalance_simulation_s=0.025 "
        "router_simulation_s=0.030 candidate_scoring_s=0.010 "
        "best_candidate_selection_s=0.002 other_s=0.003 "
        "actions=1 reason=itl_best_scale_up_Rollout\n"
        "2026-08-04 10:00:01,000 - x - [ELASTIC_OVERHEAD] operation=scale_up "
        "planner_s=0.100 sleep_s=1.0 wakeup_s=2.0 "
        "post_scale_up_rebalance_trigger_s=0.01 execution_s=3.01 total_s=3.11\n"
        "2026-08-04 10:00:02,000 - x - [ELASTIC_OVERHEAD] "
        "operation=post_scale_up_rebalance_trigger "
        "migration_id=1:1:Rollout:model planner_s=0.002 "
        "network_trigger_s=0.01 selected=10 interrupted=8\n"
        "2026-08-04 10:00:02,100 - x - [ELASTIC_OVERHEAD] "
        "operation=scaling_action step=3 decision_id=1 action_type=scale_up success=True "
        "actual_actions=1 scale_up_actions=1 scale_down_actions=0 "
        "sleep_instances=1 wakeup_instances=2 instance_transitions=3\n"
        "2026-08-04 10:00:02,200 - x - [ELASTIC_OVERHEAD] "
        "operation=scaling_action step=3 decision_id=2 action_type=scale_down success=True "
        "actual_actions=1 scale_up_actions=0 scale_down_actions=1 "
        "sleep_instances=1 wakeup_instances=0 instance_transitions=1\n",
        encoding="utf-8",
    )
    (legacy / "RolloutCoordinator.log").write_text(
        "2026-08-04 10:00:04,000 - x - [ELASTIC_OVERHEAD] operation=sleep "
        "role=Rollout router_pause_s=0.1 state_probe_s=0.2 interrupt_s=0.3 "
        "engine_sleep_s=0.4 total_s=1.0\n"
        "2026-08-04 10:00:05,000 - x - [ELASTIC_OVERHEAD] operation=wakeup "
        "role=Rollout state_probe_s=0.1 engine_wakeup_s=1.2 router_update_s=0.2 "
        "model_sync_s=0.3 router_resume_s=0.1 total_s=1.9\n",
        encoding="utf-8",
    )
    (legacy / "RolloutRouter.log").write_text(
        "2026-08-04 10:00:03,000 - x - [ELASTIC_OVERHEAD] "
        "operation=post_scale_up_rebalance migration_id=1:1:Rollout:model "
        "interrupt_s=1.0 network_s=3.0 abort_to_redispatch_s=4.0 "
        "reprefill_s=0.5 migration_s=4.5\n",
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
        "interrupt_s=0.5 network_s=2.5 abort_to_redispatch_s=3.0 "
        "reprefill_s=0.25 migration_s=3.25\n",
        encoding="utf-8",
    )
    (request_level / "RewardModelRouter.log").write_text(
        "2026-08-04 11:00:02,250 - x - [ELASTIC_OVERHEAD] "
        "operation=request_migration_interrupt role=RewardModel "
        "migration_id=3:2:RewardModel:model:planned request_id=rm-1 interrupt_s=0.4\n"
        "2026-08-04 11:00:02,500 - x - [ELASTIC_OVERHEAD] "
        "operation=request_migration_dispatch role=RewardModel "
        "migration_id=3:2:RewardModel:model:planned request_id=rm-1 "
        "interrupt_s=0.4 network_s=1.1 abort_to_redispatch_s=1.5\n",
        encoding="utf-8",
    )

    legacy_events = MODULE.collect_metric_events("legacy", legacy, start_min=0.0, end_min=None)
    request_events = MODULE.collect_metric_events("request", request_level, start_min=0.0, end_min=None)
    summaries = MODULE.summarize_metric_events(legacy_events + request_events)
    scaling_steps = MODULE.summarize_scaling_actions_per_step(
        legacy_events + request_events
    )

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
    assert legacy_network.mean == 3.0
    assert request_network.mean == 2.5
    rm_dispatch_network = next(
        item
        for item in summaries
        if item.run == "request"
        and item.operation == "request_migration_dispatch"
        and item.role == "RewardModel"
        and item.metric == "network_s"
    )
    assert rm_dispatch_network.mean == 1.1
    rm_interrupt = next(
        item
        for item in summaries
        if item.run == "request"
        and item.operation == "request_migration_interrupt"
        and item.role == "RewardModel"
        and item.metric == "interrupt_s"
    )
    assert rm_interrupt.mean == 0.4
    planner_router = next(
        item
        for item in summaries
        if item.run == "legacy"
        and item.operation == "policy_planner"
        and item.metric == "router_simulation_s"
    )
    assert planner_router.mean == 0.03
    policy_input_engine = next(
        item
        for item in summaries
        if item.run == "legacy"
        and item.operation == "policy_input"
        and item.metric == "engine_status_s"
    )
    assert policy_input_engine.mean == 0.01
    sleep_engine = next(
        item
        for item in summaries
        if item.run == "legacy"
        and item.operation == "sleep"
        and item.role == "Rollout"
        and item.metric == "engine_sleep_s"
    )
    assert sleep_engine.mean == 0.4
    legacy_step = next(
        item for item in scaling_steps if item.run == "legacy" and item.step == 3
    )
    assert legacy_step.actual_actions == 2
    assert legacy_step.scale_up_actions == 1
    assert legacy_step.scale_down_actions == 1
    assert legacy_step.sleep_instances == 2
    assert legacy_step.wakeup_instances == 2
    assert legacy_step.instance_transitions == 4

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


def test_filter_usable_migration_events_requires_terminal_completion():
    complete = MODULE.MetricEvent(
        run="run",
        run_path="/tmp/run",
        timestamp="2026-08-04 10:00:00.000",
        elapsed_min=0.0,
        source="RolloutRouter.log",
        operation="request_migration",
        cohort="all",
        role="Rollout",
        method="request_level",
        step="",
        decision_id="1",
        migration_id="1:0:Rollout:model:planned",
        request_id="req-1",
        reason="",
        metric="migration_s",
        value=1.0,
    )
    dispatch = MODULE.MetricEvent(
        **{**complete.__dict__, "operation": "request_migration_dispatch", "metric": "network_s", "value": 0.5}
    )
    reprefill = MODULE.MetricEvent(
        **{**complete.__dict__, "metric": "reprefill_s", "value": 0.2}
    )
    incomplete = MODULE.MetricEvent(
        **{**complete.__dict__, "request_id": "req-2", "metric": "interrupt_s", "value": 9.0}
    )

    filtered = MODULE.filter_usable_migration_events([complete, reprefill, dispatch, incomplete])

    assert filtered == [complete, reprefill, dispatch]


def test_filter_usable_migration_events_requires_reprefill_metric():
    terminal_without_reprefill = MODULE.MetricEvent(
        run="run",
        run_path="/tmp/run",
        timestamp="2026-08-04 10:00:00.000",
        elapsed_min=0.0,
        source="RolloutRouter.log",
        operation="request_migration",
        cohort="all",
        role="Rollout",
        method="request_level",
        step="",
        decision_id="1",
        migration_id="1:0:Rollout:model:planned",
        request_id="req-1",
        reason="",
        metric="migration_s",
        value=1.0,
    )

    assert MODULE.filter_usable_migration_events([terminal_without_reprefill]) == []


def test_filter_usable_migration_events_matches_terminal_metrics_per_request():
    def event(request_id, metric):
        return MODULE.MetricEvent(
            run="run",
            run_path="/tmp/run",
            timestamp="2026-08-04 10:00:00.000",
            elapsed_min=0.0,
            source="RolloutRouter.log",
            operation="request_migration",
            cohort="all",
            role="Rollout",
            method="request_level",
            step="",
            decision_id="1",
            migration_id="1:0:Rollout:model:planned",
            request_id=request_id,
            reason="",
            metric=metric,
            value=1.0,
        )

    migration_s = event("req-1", "migration_s")
    reprefill_s = event("req-2", "reprefill_s")

    assert MODULE.filter_usable_migration_events([migration_s, reprefill_s]) == []
