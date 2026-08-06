from datetime import datetime
from types import SimpleNamespace

import psrl.utils.elastic_rm.itl_scaling_policy as itl_policy_module
import pytest
from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.elastic_rm.itl_scaling_policy import ITLScalingPolicy, _ITLCandidate
from psrl.utils.elastic_rm.request_level_candidate_evaluator import RoleCandidatePlan
from psrl.utils.elastic_rm.scaling_policy import InstanceSignal, ScalingAction


def _config(rollout_method="throughput_optimal", rm_method="itl"):
    return SimpleNamespace(
        psrl=SimpleNamespace(
            logging_path="/tmp",
            routing_strategy=SimpleNamespace(method=rollout_method),
        ),
        reward_models_config=SimpleNamespace(
            reward_models=[
                SimpleNamespace(
                    routing_strategy=SimpleNamespace(method=rm_method),
                )
            ]
        ),
    )


def _policy(
    *,
    max_workers=8,
    router_waiting_top_t=-1,
    min_awake_per_role=1,
    config=None,
):
    return ITLScalingPolicy(
        config=config or _config(),
        policy_config={
            "enable_policy": True,
            "cooldown_ms": 0,
            "min_awake_per_role": min_awake_per_role,
            "itl_policy": {
                "enable_request_level_candidate_evaluation": True,
                "candidate_evaluation_max_workers": max_workers,
                "router_waiting_top_t": router_waiting_top_t,
                "rm_router_enable": True,
                "throughput_objective": "sum",
                "vllm_current_queue_scope": "running",
                "decision_window_s": 30.0,
                "min_gain": 0.0,
                "model_params": {
                    "default": {"A": 0.1, "B": 1.0, "C": 0.0, "D": 0.0}
                },
            },
        },
    )


def _legacy_policy():
    return ITLScalingPolicy(
        config=_config(),
        policy_config={
            "enable_policy": True,
            "cooldown_ms": 0,
            "min_awake_per_role": 1,
            "itl_policy": {
                "enable_request_level_candidate_evaluation": False,
                "router_waiting_top_t": -1,
                "rm_router_enable": True,
                "throughput_objective": "sum",
                "vllm_current_queue_scope": "running",
                "decision_window_s": 30.0,
                "min_gain": 0.0,
                "model_params": {
                    "default": {"A": 0.1, "B": 1.0, "C": 0.0, "D": 0.0}
                },
            },
        },
    )


def _signal(role, instance_id, *, awake, running=0, tokens=0):
    return InstanceSignal(
        role_name=role,
        model_name="model",
        instance_id=instance_id,
        is_awaken=awake,
        kv_cache_utilization=0.0,
        running_queue_num=running,
        waiting_queue_num=0,
        generation_throughput=0.0,
        total_token_num=tokens,
        snapshot_timestamp=datetime.now().isoformat(),
        bundle_keys=frozenset({("pool", instance_id)}),
        pool_id="pool",
    )


def _instance(instance_id, *, awake, requests):
    return {
        "instance_id": instance_id,
        "is_awake": awake,
        "requests": [
            {
                "request_id": request_id,
                "seq_len": seq_len,
                "source_instance_id": instance_id,
                "is_waiting": False,
            }
            for request_id, seq_len in requests
        ],
        "route_request_count": len(requests),
        "running_count": len(requests),
        "waiting_count": 0,
        "token_count": sum(seq_len for _, seq_len in requests),
        "max_model_len": 10000,
        "route_cost_params": [0.0, 0.0, 0.0, 1.0, 0.01],
    }


def _fixture():
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=3, tokens=13),
        _signal(PSRL_Role.Rollout, 1, awake=False),
        _signal(PSRL_Role.RewardModel, 0, awake=True, running=1, tokens=1),
        _signal(PSRL_Role.RewardModel, 1, awake=False),
    ]
    grouped = {
        PSRL_Role.Rollout: signals[:2],
        PSRL_Role.RewardModel: signals[2:],
    }
    snapshots = {
        PSRL_Role.Rollout: {
            "role": "Rollout",
            "strategy": "throughput_optimal",
            "instances": [
                _instance(0, awake=True, requests=[("10", 10), ("2", 2), ("1", 1)]),
                _instance(1, awake=False, requests=[]),
            ],
            "pending_requests": [],
            "max_concurrent_requests": 32,
        },
        PSRL_Role.RewardModel: {
            "role": "RewardModel",
            "strategy": "itl",
            "instances": [
                _instance(0, awake=True, requests=[("rm", 1)]),
                _instance(1, awake=False, requests=[]),
            ],
            "pending_requests": [],
            "max_concurrent_requests": 32,
        },
    }
    candidate = _ITLCandidate(
        action=ScalingAction(
            action_type="scale_up",
            role_name=PSRL_Role.Rollout,
            model_name="model",
            num_instances=1,
            preferred_instance_ids=[1],
        ),
        rollout_n=2,
        rm_n=1,
        current_throughput=0.0,
        next_throughput=0.0,
        delta_throughput=0.0,
        phi=0.0,
        gain_l=0.0,
    )
    return grouped, snapshots, candidate


def test_request_level_policy_scores_candidate_and_attaches_rebalance_plan():
    policy = _policy(max_workers=2)
    grouped, snapshots, candidate = _fixture()
    try:
        baseline = policy._evaluate_request_level_candidates(
            candidates=[candidate],
            grouped=grouped,
            router_backlog_by_role={},
            request_level_snapshots_by_role=snapshots,
        )
    finally:
        policy._candidate_evaluation_executor.shutdown(wait=True)

    assert baseline == candidate.current_throughput
    assert candidate.request_level_results_by_role is not None
    assert [
        migration["request_id"]
        for migration in candidate.action.planned_request_migrations
    ] == ["1", "2"]


def test_request_level_policy_selects_scale_up_from_zero_awake_rm(monkeypatch):
    policy = _policy(max_workers=2, min_awake_per_role=0)
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=1, tokens=1),
        _signal(PSRL_Role.RewardModel, 0, awake=False),
    ]
    snapshots = {
        PSRL_Role.Rollout: {
            "role": "Rollout",
            "strategy": "throughput_optimal",
            "instances": [
                _instance(0, awake=True, requests=[("rollout", 1)]),
            ],
            "pending_requests": [],
            "max_concurrent_requests": 32,
        },
        PSRL_Role.RewardModel: {
            "role": "RewardModel",
            "strategy": "itl",
            "instances": [
                _instance(0, awake=False, requests=[]),
            ],
            "pending_requests": [
                {"request_id": "rm-pending", "seq_len": 1, "route_order": 0},
            ],
            "max_concurrent_requests": 32,
        },
    }
    candidate = _ITLCandidate(
        action=ScalingAction(
            action_type="scale_up",
            role_name=PSRL_Role.RewardModel,
            model_name="model",
            num_instances=1,
            preferred_instance_ids=[0],
        ),
        rollout_n=1,
        rm_n=1,
        current_throughput=0.0,
        next_throughput=0.0,
        delta_throughput=0.0,
        phi=0.0,
        gain_l=0.0,
    )
    monkeypatch.setattr(policy, "_enumerate_candidates", lambda *_args, **_kwargs: [candidate])

    try:
        decision = policy.decide(
            signals,
            router_backlog_by_role={
                PSRL_Role.RewardModel: {"request_count": 1, "token_count": 1}
            },
            request_level_snapshots_by_role=snapshots,
        )
    finally:
        policy._candidate_evaluation_executor.shutdown(wait=True)

    assert decision.reason == "itl_best_scale_up_RewardModel"
    assert decision.actions == [candidate.action]
    assert candidate.current_throughput == 0.0
    assert candidate.next_throughput > 0.0
    assert candidate.request_level_results_by_role is not None
    assert candidate.request_level_results_by_role[PSRL_Role.RewardModel].routed_count == 1


def test_request_level_switch_preserves_candidate_construction_and_phi():
    enabled = _policy()
    disabled = _legacy_policy()
    grouped, _, _ = _fixture()
    try:
        enabled_candidates = enabled._enumerate_candidates(grouped, {})
        disabled_candidates = disabled._enumerate_candidates(grouped, {})
    finally:
        enabled._candidate_evaluation_executor.shutdown(wait=True)

    def planning_fields(candidate):
        action = candidate.action
        return (
            action.action_type,
            action.role_name,
            action.model_name,
            action.num_instances,
            action.preferred_instance_ids,
            action.pre_sleep_other_preferred,
            action.pre_wake_other_preferred,
            candidate.rollout_n,
            candidate.rm_n,
            candidate.phi,
        )

    assert [planning_fields(item) for item in enabled_candidates] == [
        planning_fields(item) for item in disabled_candidates
    ]
    assert all(item.action.planned_request_migrations is None for item in disabled_candidates)


def test_request_level_candidate_results_are_independent_of_worker_count():
    outputs = []
    for max_workers in (1, 8):
        policy = _policy(max_workers=max_workers)
        grouped, snapshots, candidate = _fixture()
        try:
            baseline = policy._evaluate_request_level_candidates(
                candidates=[candidate],
                grouped=grouped,
                router_backlog_by_role={},
                request_level_snapshots_by_role=snapshots,
            )
        finally:
            policy._candidate_evaluation_executor.shutdown(wait=True)
        outputs.append(
            (
                baseline,
                candidate.next_throughput,
                candidate.gain_l,
                candidate.action.planned_request_migrations,
            )
        )

    assert outputs[0] == outputs[1]


def test_request_level_policy_rejects_missing_snapshot_for_whole_cycle():
    policy = _policy()
    grouped, _, _ = _fixture()
    signals = grouped[PSRL_Role.Rollout] + grouped[PSRL_Role.RewardModel]
    try:
        decision = policy.decide(
            signals,
            router_backlog_by_role={},
            request_level_snapshots_by_role={},
        )
    finally:
        policy._candidate_evaluation_executor.shutdown(wait=True)

    assert decision.actions == []
    assert decision.reason == "request_level_candidate_evaluation_failed"


def test_request_level_policy_validates_top_t_and_exact_strategies():
    with pytest.raises(ValueError, match="router_waiting_top_t"):
        _policy(router_waiting_top_t=0)
    with pytest.raises(ValueError, match="throughput_optimal"):
        _policy(config=_config(rollout_method="round_robin"))
    with pytest.raises(ValueError, match="reward-model routing strategy 'itl'"):
        _policy(config=_config(rm_method="round_robin"))


def test_request_level_policy_prepares_once_and_deduplicates_unchanged_role(
    monkeypatch,
):
    prepared_roles = []
    evaluated_plans = []
    original_prepare = itl_policy_module.prepare_role_evaluation_context
    original_evaluate = itl_policy_module.evaluate_role_candidate

    def recording_prepare(snapshot):
        context = original_prepare(snapshot)
        prepared_roles.append(context.snapshot.role)
        return context

    def recording_evaluate(context, plan):
        evaluated_plans.append((context.snapshot.role, plan))
        return original_evaluate(context, plan)

    monkeypatch.setattr(
        itl_policy_module,
        "prepare_role_evaluation_context",
        recording_prepare,
    )
    monkeypatch.setattr(
        itl_policy_module,
        "evaluate_role_candidate",
        recording_evaluate,
    )
    policy = _policy(max_workers=2)
    grouped, snapshots, candidate = _fixture()
    try:
        policy._evaluate_request_level_candidates(
            candidates=[candidate],
            grouped=grouped,
            router_backlog_by_role={},
            request_level_snapshots_by_role=snapshots,
        )
    finally:
        policy._candidate_evaluation_executor.shutdown(wait=True)

    assert sorted(prepared_roles) == ["RewardModel", "Rollout"]
    assert len(evaluated_plans) == 3
    assert sum(
        role == "RewardModel" and plan == RoleCandidatePlan()
        for role, plan in evaluated_plans
    ) == 1
