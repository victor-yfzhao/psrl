import asyncio
import math
import os
from datetime import datetime
from types import SimpleNamespace

import pytest

from pivotrl.trainer.ppo.utils import PivotRL_Role
from pivotrl.utils.elastic_rm.itl_harmonic_scaling_policy import ITLHarmonicScalingPolicy
from pivotrl.utils.elastic_rm.itl_scaling_policy import (
    ITLModelParams,
    ITLScalingPolicy,
    compute_itl,
    resolve_itl_model_params,
)
from pivotrl.utils.elastic_rm.scaling_policy import InstanceSignal
from pivotrl.workers.reward.reward_model.router import (
    ITLBalanceRewardModelRouteStrategy,
    PivotRL_RewardModelRouter,
    ThroughputOptimalRewardModelRouteStrategy,
)


class _Event:
    def set(self):
        pass


class _RemoteMethod:
    def __init__(self, value: int):
        self.value = value

    async def remote(self):
        return self.value


class _Worker:
    def __init__(self, active_task_num: int):
        self.get_active_task_num = _RemoteMethod(active_task_num)


def _config():
    return SimpleNamespace(pivotrl=SimpleNamespace(logging_path="/tmp"))


def _policy(
    model_params: dict | None = None,
    *,
    max_scale_instances_per_action: int = 1,
    throughput_objective: str = "balanced_min",
    vllm_current_queue_scope: str = "running_waiting",
    current_state_include_router_waiting: bool = True,
    role_throughput_weight_enable: bool = False,
    role_throughput_weight_basis: str = "request_count",
    role_throughput_weight_mode: str = "share",
    harmonic_denominator_use_max: bool | str = False,
    enable_heterogeneous_parallelism_candidates: bool = False,
    policy_cls: type[ITLScalingPolicy] = ITLScalingPolicy,
) -> ITLScalingPolicy:
    return policy_cls(
        config=_config(),
        policy_config={
            "enable_policy": True,
            "cooldown_ms": 0,
            "min_awake_per_role": 1,
            "itl_policy": {
                "decision_window_s": 30.0,
                "migrate_time_s_initial": 30.0,
                "min_gain": 0.0,
                "max_scale_instances_per_action": max_scale_instances_per_action,
                "throughput_objective": throughput_objective,
                "vllm_current_queue_scope": vllm_current_queue_scope,
                "current_state_include_router_waiting": current_state_include_router_waiting,
                "role_throughput_weight_enable": role_throughput_weight_enable,
                "role_throughput_weight_basis": role_throughput_weight_basis,
                "role_throughput_weight_mode": role_throughput_weight_mode,
                "harmonic_denominator_use_max": harmonic_denominator_use_max,
                "enable_heterogeneous_parallelism_candidates": enable_heterogeneous_parallelism_candidates,
                "model_params": model_params or {"default": {"A": 0.0, "B": 10.0, "C": 1.0, "D": 0.0}},
            },
        },
    )


def _signal(
    role: PivotRL_Role,
    instance_id: int,
    *,
    awake: bool,
    running: int,
    tokens: int,
    bundle: int | tuple[int, ...],
    waiting: int = 0,
    pool_id: str | None = "shared_rollout_pool",
) -> InstanceSignal:
    return InstanceSignal(
        role_name=role,
        model_name="model",
        instance_id=instance_id,
        is_awaken=awake,
        kv_cache_utilization=0.5,
        running_queue_num=running,
        waiting_queue_num=waiting,
        generation_throughput=0.0,
        total_token_num=tokens,
        snapshot_timestamp=datetime.now().isoformat(),
        bundle_keys=frozenset(
            (pool_id, bundle_index) for bundle_index in ((bundle,) if isinstance(bundle, int) else bundle)
        ),
        pool_id=pool_id,
    )


def test_resolve_itl_model_params_reads_tp_pp_cost_model_json():
    cost_path = "pivotrl/trainer/config/cost_model"
    rollout_params = resolve_itl_model_params(
        role_name=PivotRL_Role.Rollout,
        model_name="Qwen2.5-7B",
        itl_config={"cost_model_path": cost_path},
        tp_pp="TP1_PP1",
    )
    assert rollout_params.A == pytest.approx(3.71e-08)
    assert rollout_params.B == pytest.approx(0.0051426)
    assert rollout_params.C == pytest.approx(0.00008)
    assert rollout_params.D == pytest.approx(0.0005060241)

    rm_params = resolve_itl_model_params(
        role_name=PivotRL_Role.RewardModel,
        model_name="Qwen3-8B",
        itl_config={"cost_model_path": cost_path},
        tp_pp="TP1_PP1",
    )
    assert rm_params.A == pytest.approx(8.57e-08)
    assert rm_params.B == pytest.approx(0.0057469)
    assert rm_params.C == pytest.approx(7.59e-05)
    assert rm_params.D == pytest.approx(0.0010808344)


def test_resolve_itl_model_params_auto_selects_single_tp_pp_bucket():
    cost_path = "pivotrl/trainer/config/cost_model"
    params = resolve_itl_model_params(
        role_name=PivotRL_Role.Rollout,
        model_name="Qwen2.5-7B",
        itl_config={"cost_model_path": cost_path},
    )
    assert params.B == pytest.approx(0.0051426)


def test_resolve_itl_model_params_uses_config_parallelism_for_tp_pp():
    cost_path = "pivotrl/trainer/config/cost_model"
    config = SimpleNamespace(
        gen_actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        ),
        reward_models_config=SimpleNamespace(
            reward_models=[
                SimpleNamespace(
                    reward_model_name="Qwen3-8B",
                    rollout=SimpleNamespace(tensor_model_parallel_size=1, pipeline_model_parallel_size=1),
                )
            ]
        ),
    )
    rollout_params = resolve_itl_model_params(
        role_name=PivotRL_Role.Rollout,
        model_name="Qwen2.5-7B",
        itl_config={"cost_model_path": cost_path},
        config=config,
    )
    rm_params = resolve_itl_model_params(
        role_name=PivotRL_Role.RewardModel,
        model_name="Qwen3-8B",
        itl_config={"cost_model_path": cost_path},
        config=config,
    )
    assert rollout_params.B == pytest.approx(0.0051426)
    assert rm_params.B == pytest.approx(0.0057469)


def test_current_state_router_waiting_switch_controls_current_throughput():
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=3, tokens=30, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=True, running=3, tokens=30, bundle=1),
    ]
    grouped = {PivotRL_Role.Rollout: rollout}
    router_backlog = {PivotRL_Role.Rollout: {"count": 14, "total_tokens": 0}}

    enabled = _policy(throughput_objective="sum", current_state_include_router_waiting=True)
    enabled_tps = enabled._current_role_throughputs(
        grouped,
        rollout_n=2,
        rm_n=0,
        router_backlog_by_role=router_backlog,
    )
    enabled_waiting = enabled._current_router_waiting_load(router_backlog[PivotRL_Role.Rollout])
    enabled_snapshot = enabled._current_role_load_snapshot(rollout, 2, enabled_waiting)

    disabled = _policy(throughput_objective="sum", current_state_include_router_waiting=False)
    disabled_tps = disabled._current_role_throughputs(
        grouped,
        rollout_n=2,
        rm_n=0,
        router_backlog_by_role=router_backlog,
    )
    disabled_waiting = disabled._current_router_waiting_load(router_backlog[PivotRL_Role.Rollout])
    disabled_snapshot = disabled._current_role_load_snapshot(rollout, 2, disabled_waiting)

    assert enabled_tps[PivotRL_Role.Rollout] == pytest.approx(2.0)
    assert [row[1] for row in enabled_snapshot.instance_rows] == pytest.approx([10.0, 10.0])
    assert disabled_tps[PivotRL_Role.Rollout] == pytest.approx(0.6)
    assert [row[1] for row in disabled_snapshot.instance_rows] == pytest.approx([3.0, 3.0])


def test_itl_role_throughput_uses_balanced_average_state():
    policy = _policy()
    signals = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=1),
    ]

    assert policy._role_throughput(signals, 2) == 0.6


def test_itl_role_throughput_can_ignore_vllm_waiting_load():
    signals = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=4, waiting=8, tokens=0, bundle=0),
    ]

    legacy_policy = _policy()
    running_only_policy = _policy(vllm_current_queue_scope="running")

    assert legacy_policy._role_throughput(signals, 1) == 1.0
    assert running_only_policy._role_throughput(signals, 1) == 0.4


def test_itl_role_throughput_includes_router_waiting_prefix_load():
    policy = _policy({"default": {"A": 0.1, "B": 10.0, "C": 1.0, "D": 0.0}})
    signals = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=1),
    ]
    waiting = policy._normalize_router_waiting_load({"count": 4, "total_tokens": 40})

    assert policy._role_throughput(signals, 2, waiting) == 8.0 / 18.0


def test_itl_role_throughput_idle_role_is_not_bottleneck_even_with_zero_instances():
    policy = _policy()
    signals = [
        _signal(PivotRL_Role.Rollout, 0, awake=False, running=0, tokens=0, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=1),
    ]

    assert math.isinf(policy._role_throughput(signals, 0))


def test_itl_role_throughput_waiting_role_without_instances_is_bottleneck():
    policy = _policy()
    signals = [
        _signal(PivotRL_Role.Rollout, 0, awake=False, running=0, tokens=0, bundle=0),
    ]
    waiting = policy._normalize_router_waiting_load({"count": 1, "total_tokens": 10})

    assert policy._role_throughput(signals, 0, waiting) == 0.0


def test_itl_sum_objective_adds_instance_throughputs_with_idle_as_inf():
    policy = _policy(throughput_objective="sum")
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=True, running=0, tokens=0, bundle=1),
    ]
    rm = [
        _signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=2),
    ]

    assert math.isinf(policy._role_throughput_sum(rollout, 2))
    assert policy._system_throughput({PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}, 2, 1) == 0.5


def test_role_throughput_weights_share_pressure_across_both_roles():
    policy = _policy(role_throughput_weight_enable=True, role_throughput_weight_basis="request_count")
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=3, tokens=30, bundle=2)]
    rollout_waiting = policy._normalize_router_waiting_load({"count": 8, "total_tokens": 80})
    rm_waiting = policy._normalize_router_waiting_load({"count": 2, "total_tokens": 20})

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, rollout_waiting, rm_waiting)

    assert w_rollout == pytest.approx((12 + 8) / ((12 + 8) + (3 + 2)))
    assert w_rm == pytest.approx((3 + 2) / ((12 + 8) + (3 + 2)))
    assert w_rollout + w_rm == pytest.approx(1.0)


def test_role_throughput_weights_use_token_basis_when_configured():
    policy = _policy(role_throughput_weight_enable=True, role_throughput_weight_basis="token_count")
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=3, tokens=30, bundle=2)]
    rollout_waiting = policy._normalize_router_waiting_load({"count": 8, "total_tokens": 80})
    rm_waiting = policy._normalize_router_waiting_load({"count": 2, "total_tokens": 20})

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, rollout_waiting, rm_waiting)

    assert w_rollout == pytest.approx((120 + 80) / ((120 + 80) + (30 + 20)))
    assert w_rm == pytest.approx((30 + 20) / ((120 + 80) + (30 + 20)))


def test_role_throughput_weights_uniform_when_disabled():
    policy = _policy(role_throughput_weight_enable=False)
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=3, tokens=30, bundle=2)]
    rollout_waiting = policy._normalize_router_waiting_load({"count": 8, "total_tokens": 80})
    rm_waiting = policy._normalize_router_waiting_load({"count": 2, "total_tokens": 20})

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, rollout_waiting, rm_waiting)

    assert w_rollout == 0.5
    assert w_rm == 0.5


def test_role_throughput_weights_uniform_when_both_sides_idle():
    policy = _policy(role_throughput_weight_enable=True, role_throughput_weight_basis="request_count")
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=0, tokens=0, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=0, tokens=0, bundle=2)]

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, None, None)

    assert w_rollout == 0.5
    assert w_rm == 0.5


def test_system_throughput_weighted_divides_by_role_pressure_share():
    # ITL = max(10, running). rollout_tp = 10/10 = 1.0, rm_tp = 5/10 = 0.5.
    # Pressure: rollout=10, rm=5 -> w_rollout=10/15, w_rm=5/15.
    # Weighted bottleneck = min(1.0 / (10/15), 0.5 / (5/15)) = min(1.5, 1.5) = 1.5.
    policy = _policy(
        throughput_objective="sum",
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="request_count",
    )
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=10, tokens=100, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=2)]

    tp = policy._system_throughput({PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}, 1, 1)

    assert tp == pytest.approx(1.5)


def test_system_throughput_high_pressure_side_becomes_bottleneck():
    # rollout running=12 -> ITL=12, tp=1.0. rm running=3 -> ITL=10, tp=0.3.
    # Pressure: rollout=12, rm=3 -> w_rollout=0.8, w_rm=0.2.
    # Weighted bottleneck = min(1.0/0.8, 0.3/0.2) = min(1.25, 1.5) = 1.25 -> rollout bottleneck.
    policy = _policy(
        throughput_objective="sum",
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="request_count",
    )
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=3, tokens=30, bundle=2)]

    tp = policy._system_throughput({PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}, 1, 1)

    assert tp == pytest.approx(1.25)


def test_system_throughput_weight_uses_router_backlog_pressure():
    # Both roles have the same in-instance load (running=5), but the router
    # backlog piles up on Rollout only.
    #   rollout: request_load = 5 + 10 = 15 -> ITL=15 -> tp = 1.0
    #   rm:      request_load = 5 + 0  = 5  -> ITL=10 -> tp = 0.5
    # Pressure shares: w_rollout = 15/20 = 0.75, w_rm = 0.25.
    # Weighted bottleneck = min(1.0/0.75, 0.5/0.25) = min(1.333, 2.0) = 1.333.
    # Without weighting the bottleneck would be rm (0.5); weighting shifts the
    # bottleneck to the high-pressure Rollout side (1.333), which is the intended
    # effect of pressure-weighted throughput.
    weighted_policy = _policy(
        throughput_objective="sum",
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="request_count",
    )
    unweighted_policy = _policy(
        throughput_objective="sum",
        role_throughput_weight_enable=False,
    )
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=5, tokens=50, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=2)]
    backlog = {
        PivotRL_Role.Rollout: {"count": 10, "total_tokens": 100},
        PivotRL_Role.RewardModel: {"count": 0, "total_tokens": 0},
    }
    grouped = {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}

    tp = weighted_policy._system_throughput(grouped, 1, 1, router_backlog_by_role=backlog)
    unweighted = unweighted_policy._system_throughput(grouped, 1, 1, router_backlog_by_role=backlog)

    assert unweighted == 0.5
    assert tp == pytest.approx(1.0 / 0.75)


def test_system_throughput_unweighted_when_disabled_matches_legacy_min():
    policy = _policy(throughput_objective="sum", role_throughput_weight_enable=False)
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=2)]

    tp = policy._system_throughput({PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}, 1, 1)

    assert tp == 0.5


def test_role_throughput_weights_raw_mode_returns_raw_pressure_values():
    # raw mode: divisors are the raw per-role pressure magnitudes, not shares.
    policy = _policy(
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="request_count",
        role_throughput_weight_mode="raw",
    )
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=3, tokens=30, bundle=2)]
    rollout_waiting = policy._normalize_router_waiting_load({"count": 8, "total_tokens": 80})
    rm_waiting = policy._normalize_router_waiting_load({"count": 2, "total_tokens": 20})

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, rollout_waiting, rm_waiting)

    assert w_rollout == pytest.approx(20.0)  # 12 + 8
    assert w_rm == pytest.approx(5.0)  # 3 + 2


def test_role_throughput_weights_raw_mode_zero_pressure_falls_back_to_one():
    # A side with zero pressure uses divisor 1.0 so it is not bottlenecked by /0.
    policy = _policy(
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="request_count",
        role_throughput_weight_mode="raw",
    )
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=0, tokens=0, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=7, tokens=70, bundle=2)]

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, None, None)

    assert w_rollout == 1.0
    assert w_rm == pytest.approx(7.0)


def test_role_throughput_weights_raw_mode_token_basis_uses_token_totals():
    policy = _policy(
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="token_count",
        role_throughput_weight_mode="raw",
    )
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=3, tokens=30, bundle=2)]
    rollout_waiting = policy._normalize_router_waiting_load({"count": 8, "total_tokens": 80})
    rm_waiting = policy._normalize_router_waiting_load({"count": 2, "total_tokens": 20})

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, rollout_waiting, rm_waiting)

    assert w_rollout == pytest.approx(200.0)  # 120 + 80
    assert w_rm == pytest.approx(50.0)  # 30 + 20


def test_system_throughput_raw_mode_divides_by_raw_pressure():
    # ITL = max(10, running). rollout running=10 -> tp=1.0, pressure=10.
    # rm running=5 -> tp=0.5, pressure=5.
    # raw mode: bottleneck = min(1.0/10, 0.5/5) = min(0.1, 0.1) = 0.1.
    policy = _policy(
        throughput_objective="sum",
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="request_count",
        role_throughput_weight_mode="raw",
    )
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=10, tokens=100, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=2)]

    tp = policy._system_throughput({PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}, 1, 1)

    assert tp == pytest.approx(0.1)


def test_system_throughput_raw_mode_high_pressure_side_becomes_bottleneck():
    # rollout running=12 -> tp=1.0, pressure=12. rm running=3 -> tp=0.3, pressure=3.
    # raw mode: min(1.0/12, 0.3/3) = min(0.0833, 0.1) = 0.0833 -> rollout bottleneck
    # (the heavier-pressure side, even though its raw tp is higher).
    policy = _policy(
        throughput_objective="sum",
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="request_count",
        role_throughput_weight_mode="raw",
    )
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=3, tokens=30, bundle=2)]

    tp = policy._system_throughput({PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}, 1, 1)

    assert tp == pytest.approx(1.0 / 12.0)


def test_itl_sum_objective_windowed_gain_does_not_multiply_instance_count():
    sum_policy = _policy(throughput_objective="sum")
    balanced_policy = _policy()

    assert sum_policy._windowed_throughput_gain(2.0, 2, 3) == 60.0
    assert balanced_policy._windowed_throughput_gain(2.0, 2, 3) == 300.0


def test_harmonic_system_throughput_unweighted_is_standard_harmonic_mean():
    # ITL = max(10, running). rollout running=12 -> tp=1.0, rm running=5 -> tp=0.5.
    # Unweighted harmonic mean = 2*1.0*0.5 / (1.0 + 0.5) = 0.6667.
    policy = _policy(throughput_objective="sum", policy_cls=ITLHarmonicScalingPolicy)
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=2)]

    tp = policy._system_throughput({PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}, 1, 1)

    assert tp == pytest.approx(2.0 * 1.0 * 0.5 / (1.0 + 0.5))


def test_harmonic_max_denominator_switch_preserves_default_and_uses_larger_term():
    # rollout tp=1.0, rm tp=0.5. Uniform weights produce reciprocal terms
    # 0.5 and 1.0: the default sum denominator is 1.5, while the enabled
    # max denominator is 1.0.
    default_policy = _policy(policy_cls=ITLHarmonicScalingPolicy)
    max_policy = _policy(
        policy_cls=ITLHarmonicScalingPolicy,
        harmonic_denominator_use_max="true",
    )

    assert default_policy._weighted_system_throughput(1.0, 0.5, 0.5, 0.5) == pytest.approx(2.0 / 3.0)
    assert max_policy._weighted_system_throughput(1.0, 0.5, 0.5, 0.5) == pytest.approx(1.0)
    assert max_policy.harmonic_denominator_use_max is True


def test_harmonic_idle_instance_contributes_zero_not_infinite():
    # Under the harmonic policy an instance with no requests contributes 0
    # throughput, not inf. Adding an idle instance to a role that already has
    # busy instances must keep the role throughput finite (sum of busy tps) and
    # must not inflate the system throughput toward inf. The bottleneck policy
    # still reports inf for the idle instance, so its role throughput is inf.
    # ITL = max(10, running); busy instance running=12 -> tp=1.0.
    harmonic = _policy(throughput_objective="sum", policy_cls=ITLHarmonicScalingPolicy)
    bottleneck = _policy(throughput_objective="sum", policy_cls=ITLScalingPolicy)
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0)]
    rm_busy = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=2)]
    rm_idle = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=2),
               _signal(PivotRL_Role.RewardModel, 1, awake=True, running=0, tokens=0, bundle=3)]

    # Role throughput: harmonic keeps the busy instance's 0.5 (idle adds 0);
    # bottleneck sums inf from the idle instance -> inf.
    assert harmonic._role_throughput_sum(rm_idle, 2) == pytest.approx(0.5)
    assert math.isinf(bottleneck._role_throughput_sum(rm_idle, 2))

    # System throughput stays finite for harmonic; it does not blow up to inf.
    tp_one = harmonic._system_throughput(
        {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm_busy}, 1, 1
    )
    tp_two = harmonic._system_throughput(
        {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm_idle}, 1, 2
    )
    assert not math.isinf(tp_two)
    # Adding an idle RM instance (no backlog to redistribute) does not raise the
    # objective above the single-busy-instance baseline.
    assert tp_two == pytest.approx(tp_one)


def test_harmonic_system_throughput_weighted_request_basis():
    # rollout tp=1.0 (pressure 12), rm tp=0.5 (pressure 5).
    # w_rollout=12/17, w_rm=5/17. H = (w_r+w_m) / (w_r/1.0 + w_m/0.5).
    policy = _policy(
        throughput_objective="sum",
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="request_count",
        policy_cls=ITLHarmonicScalingPolicy,
    )
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=2)]

    tp = policy._system_throughput({PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}, 1, 1)

    w_r, w_m = 12.0 / 17.0, 5.0 / 17.0
    assert tp == pytest.approx((w_r + w_m) / (w_r / 1.0 + w_m / 0.5))


def test_harmonic_system_throughput_weighted_token_basis():
    # rollout tp=1.0 (tokens 120), rm tp=0.5 (tokens 50).
    # w_rollout=120/170, w_rm=50/170.
    policy = _policy(
        throughput_objective="sum",
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="token_count",
        policy_cls=ITLHarmonicScalingPolicy,
    )
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=12, tokens=120, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=2)]

    tp = policy._system_throughput({PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}, 1, 1)

    w_r, w_m = 120.0 / 170.0, 50.0 / 170.0
    assert tp == pytest.approx((w_r + w_m) / (w_r / 1.0 + w_m / 0.5))


def test_harmonic_system_throughput_zero_side_collapses_to_zero():
    # rm has running queue but its role throughput is computed as 0 only when it
    # has waiting load and zero instances; here we force the collapse directly via
    # the weighted combine: a side with throughput 0 must drive H to 0.
    policy = _policy(policy_cls=ITLHarmonicScalingPolicy)

    assert policy._weighted_system_throughput(1.0, 0.0, 0.5, 0.5) == 0.0
    assert policy._weighted_system_throughput(0.0, 1.0, 0.5, 0.5) == 0.0


def test_harmonic_system_throughput_both_idle_collapses_to_zero():
    # Both roles idle (no outstanding requests). Under the harmonic policy an
    # idle instance contributes 0 (not inf), so each role throughput is 0 and
    # the harmonic mean collapses to 0 instead of being inflated to inf.
    policy = _policy(policy_cls=ITLHarmonicScalingPolicy)
    rollout = [_signal(PivotRL_Role.Rollout, 0, awake=True, running=0, tokens=0, bundle=0)]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=0, tokens=0, bundle=2)]

    tp = policy._system_throughput({PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}, 1, 1)

    assert tp == 0.0


def test_harmonic_policy_picks_same_scale_up_side_as_bottleneck_policy():
    # Rollout is the bottleneck (low load -> low tp). Both policies must select
    # Rollout as the side to scale up; only the objective value differs.
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=3, tokens=30, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=1),
        _signal(PivotRL_Role.Rollout, 2, awake=False, running=0, tokens=0, bundle=2),
    ]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=3)]
    grouped = {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}

    harmonic = _policy(
        max_scale_instances_per_action=2,
        throughput_objective="sum",
        policy_cls=ITLHarmonicScalingPolicy,
    )
    bottleneck = _policy(
        max_scale_instances_per_action=2,
        throughput_objective="sum",
        policy_cls=ITLScalingPolicy,
    )

    h_candidates = harmonic._enumerate_candidates(grouped, router_backlog_by_role={})
    b_candidates = bottleneck._enumerate_candidates(grouped, router_backlog_by_role={})

    def summarize(cands):
        return sorted(
            (c.action.action_type, c.action.role_name.name, tuple(c.action.preferred_instance_ids or ()))
            for c in cands
        )

    assert summarize(h_candidates) == summarize(b_candidates)
    h_best = max(h_candidates, key=lambda c: c.gain_l)
    b_best = max(b_candidates, key=lambda c: c.gain_l)
    assert h_best.action.action_type == b_best.action.action_type
    assert h_best.action.role_name == b_best.action.role_name


def test_itl_transfer_candidate_sets_pre_sleep_before_wake():
    # Rollout is the bottleneck (tp 0.5 < rm tp 1.0). Waking rollout[1] needs
    # bundle 1, which is held by the awake rm[0]. Capacity is large enough that
    # rm_n is not auto-reduced, so the cross-role eviction path must sleep rm[0]
    # first and record it in pre_sleep_other_preferred.
    policy = _policy(throughput_objective="sum")
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=5, tokens=50, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=1),
    ]
    rm = [
        _signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=1),
        _signal(PivotRL_Role.RewardModel, 1, awake=True, running=5, tokens=50, bundle=2),
    ]
    grouped = {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(grouped, router_backlog_by_role={})
    rollout_scale_ups = [
        c for c in candidates
        if c.action.action_type == "scale_up" and c.action.role_name == PivotRL_Role.Rollout
    ]

    assert rollout_scale_ups, "expected at least one rollout scale-up candidate"
    candidate = rollout_scale_ups[0]
    assert candidate.action.preferred_instance_ids == [1]
    assert candidate.action.pre_sleep_other_preferred == [
        {"role_name": PivotRL_Role.RewardModel, "model_name": "model", "instance_id": 0}
    ]
    assert candidate.rollout_n == 2
    assert candidate.rm_n == 1


def test_itl_sum_objective_prunes_conflict_scale_up_when_free_bundle_exists():
    # Rollout is the bottleneck (low load -> low tp). A free bundle exists on
    # rollout[2] (bundle 2, not held by any awake rm), while rollout[1] (bundle 1)
    # collides with the awake rm[0]. The sum objective must prefer the conflict-free
    # wake (rollout[2]) and skip the conflicting one, so no pre_sleep is needed.
    policy = _policy(throughput_objective="sum")
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=5, tokens=50, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=1),
        _signal(PivotRL_Role.Rollout, 2, awake=False, running=0, tokens=0, bundle=2),
    ]
    rm = [
        _signal(PivotRL_Role.RewardModel, 0, awake=True, running=12, tokens=120, bundle=1),
        _signal(PivotRL_Role.RewardModel, 1, awake=True, running=12, tokens=120, bundle=3),
    ]
    grouped = {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}

    rollout_scale_ups = [
        candidate
        for candidate in policy._enumerate_candidates(grouped, router_backlog_by_role={})
        if candidate.action.action_type == "scale_up" and candidate.action.role_name == PivotRL_Role.Rollout
    ]

    assert len(rollout_scale_ups) == 1
    assert rollout_scale_ups[0].action.preferred_instance_ids == [2]
    assert rollout_scale_ups[0].action.pre_sleep_other_preferred is None


def test_itl_balanced_objective_rejects_eviction_below_min_awake():
    # Rollout is the strict bottleneck. The only wakeable rollout instance
    # (rollout[1], bundle 1) collides with the sole awake rm[0] (bundle 1).
    # Evicting rm[0] would drop the rm role below min_awake_per_role (1), so the
    # cross-role eviction must be rejected and no rollout scale-up is produced.
    policy = _policy(throughput_objective="balanced_min")
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=3, tokens=30, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=1),
    ]
    rm = [
        _signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=1),
        _signal(PivotRL_Role.RewardModel, 1, awake=False, running=0, tokens=0, bundle=2),
    ]
    grouped = {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}

    rollout_scale_ups = [
        candidate
        for candidate in policy._enumerate_candidates(grouped, router_backlog_by_role={})
        if candidate.action.action_type == "scale_up" and candidate.action.role_name == PivotRL_Role.Rollout
    ]

    assert rollout_scale_ups == []


def test_itl_enumerates_batch_scale_up_candidates_when_enabled():
    # Rollout is the strict bottleneck. With max_scale_instances_per_action=2 the
    # count enumeration must reach rollout_n=3, producing a single batch wake of
    # rollout[1] and rollout[2] (both conflict-free against the rm on bundle 3).
    policy = _policy(max_scale_instances_per_action=2)
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=3, tokens=30, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=1),
        _signal(PivotRL_Role.Rollout, 2, awake=False, running=0, tokens=0, bundle=2),
    ]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=3)]
    grouped = {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(grouped, router_backlog_by_role={})
    batch = [
        c
        for c in candidates
        if c.action.action_type == "scale_up"
        and c.action.role_name == PivotRL_Role.Rollout
        and c.action.num_instances == 2
    ]

    assert len(batch) == 1
    assert batch[0].action.preferred_instance_ids == [1, 2]
    assert batch[0].rollout_n == 3


def test_itl_unlimited_max_scale_instances_enumerates_all_wakeable():
    # Rollout is the strict bottleneck with three wakeable instances. With
    # max_scale_instances_per_action=-1 the count enumeration must reach
    # rollout_n=4, producing a batch wake of all three sleepers.
    policy = _policy(max_scale_instances_per_action=-1)
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=3, tokens=30, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=1),
        _signal(PivotRL_Role.Rollout, 2, awake=False, running=0, tokens=0, bundle=2),
        _signal(PivotRL_Role.Rollout, 3, awake=False, running=0, tokens=0, bundle=3),
    ]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=4)]
    grouped = {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}

    assert policy.max_scale_instances_per_action == -1
    candidates = policy._enumerate_candidates(grouped, router_backlog_by_role={})
    batch = [
        c
        for c in candidates
        if c.action.action_type == "scale_up"
        and c.action.role_name == PivotRL_Role.Rollout
        and c.action.num_instances == 3
    ]

    assert len(batch) == 1
    assert batch[0].action.preferred_instance_ids == [1, 2, 3]
    assert batch[0].rollout_n == 4
    assert {c.action.num_instances for c in candidates if c.action.role_name == PivotRL_Role.Rollout} == {1, 2, 3}


def test_itl_keeps_single_step_candidates_by_default():
    # Rollout is the strict bottleneck with two wakeable instances. With the
    # default max_scale_instances_per_action=1 the count enumeration must only
    # reach rollout_n=2, so every candidate is a single-step action.
    policy = _policy()
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=3, tokens=30, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=1),
        _signal(PivotRL_Role.Rollout, 2, awake=False, running=0, tokens=0, bundle=2),
    ]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, bundle=3)]
    grouped = {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(grouped, router_backlog_by_role={})

    assert candidates, "expected at least one candidate"
    assert all(c.action.num_instances == 1 for c in candidates)


def test_heterogeneous_switch_preserves_equal_parallel_log_fixture():
    # Fixture shape and preferred ids come from ScalingPolicy/ElasticExecutor
    # cycle 7816 (decision 354): eight TP1 rollout wakes competing with TP1 RM.
    wake_ids = (6, 12, 20, 22, 16, 1, 2, 11)
    running_loads = (39, 38, 39, 38, 39, 38, 39, 38)
    token_loads = (167963, 163713, 158012, 159014, 145643, 166654, 159014, 163713)
    rollout = [
        _signal(PivotRL_Role.Rollout, instance_id, awake=False, running=0, tokens=0, bundle=instance_id)
        for instance_id in wake_ids
    ]
    rm = [
        _signal(
            PivotRL_Role.RewardModel,
            instance_id,
            awake=True,
            running=running,
            tokens=tokens,
            bundle=instance_id,
        )
        for instance_id, running, tokens in zip(wake_ids, running_loads, token_loads)
    ]
    rm.append(_signal(PivotRL_Role.RewardModel, 99, awake=True, running=38, tokens=159014, bundle=99))

    legacy = _policy(max_scale_instances_per_action=8, throughput_objective="sum")
    enabled = _policy(
        max_scale_instances_per_action=8,
        throughput_objective="sum",
        enable_heterogeneous_parallelism_candidates=True,
    )
    legacy_plans = legacy._wake_prefix_plans_for_role(
        target_signals=rollout,
        other_signals=rm,
        current_other_n=9,
        max_batch=8,
    )
    enabled_plans = enabled._wake_prefix_plans_for_role(
        target_signals=rollout,
        other_signals=rm,
        current_other_n=9,
        max_batch=8,
    )

    def summarize(plans):
        return {
            count: (
                tuple(signal.instance_id for signal in plan.wakes),
                tuple(signal.instance_id for signal in plan.pre_sleep),
                tuple(signal.instance_id for signal in plan.pre_wake),
            )
            for count, plan in plans.items()
        }

    assert summarize(enabled_plans) == summarize(legacy_plans)


@pytest.mark.parametrize("policy_cls", [ITLScalingPolicy, ITLHarmonicScalingPolicy])
def test_heterogeneous_large_wake_migrates_small_conflicts_and_preserves_load(policy_cls):
    # RM loads are sampled from ScalingPolicy cycle 7815. A TP4 rollout wake
    # displaces two TP1 RM instances, which can move to free TP1 placements.
    policy = _policy(
        model_params={"default": {"A": 1e-6, "B": 10.0, "C": 1.0, "D": 0.0}},
        throughput_objective="sum",
        enable_heterogeneous_parallelism_candidates=True,
        policy_cls=policy_cls,
    )
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=3, tokens=30, bundle=(12, 13, 14, 15)),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=(0, 1, 2, 3)),
    ]
    rm = [
        _signal(PivotRL_Role.RewardModel, 0, awake=True, running=39, tokens=167963, bundle=0),
        _signal(PivotRL_Role.RewardModel, 1, awake=True, running=38, tokens=163713, bundle=1),
        _signal(PivotRL_Role.RewardModel, 2, awake=False, running=0, tokens=0, bundle=8),
        _signal(PivotRL_Role.RewardModel, 3, awake=False, running=0, tokens=0, bundle=9),
    ]
    grouped = {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(grouped, router_backlog_by_role={})
    candidate = next(
        item
        for item in candidates
        if item.action.action_type == "scale_up" and item.action.role_name == PivotRL_Role.Rollout
    )

    assert candidate.action.preferred_instance_ids == [1]
    assert candidate.action.pre_sleep_other_preferred == [
        {"role_name": PivotRL_Role.RewardModel, "model_name": "model", "instance_id": 0},
        {"role_name": PivotRL_Role.RewardModel, "model_name": "model", "instance_id": 1},
    ]
    assert candidate.action.pre_wake_other_preferred == [
        {"role_name": PivotRL_Role.RewardModel, "model_name": "model", "instance_id": 2},
        {"role_name": PivotRL_Role.RewardModel, "model_name": "model", "instance_id": 3},
    ]
    assert candidate.rollout_n == 2
    assert candidate.rm_n == 2
    assert candidate.load_overrides_by_role == {PivotRL_Role.RewardModel: {2: (39.0, 167963.0), 3: (38.0, 163713.0)}}
    assert candidate.next_throughput > 0.0

    disabled = _policy(throughput_objective="sum")
    disabled_plans = disabled._wake_prefix_plans_for_role(
        target_signals=rollout,
        other_signals=rm,
        current_other_n=2,
        max_batch=1,
    )
    assert disabled_plans == {}


def test_heterogeneous_wake_cost_treats_conflict_free_as_zero_phi():
    policy = _policy(enable_heterogeneous_parallelism_candidates=True)
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=False, running=0, tokens=0, bundle=(0, 1, 2, 3)),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=(4, 5, 6, 7)),
    ]
    rm = [
        _signal(PivotRL_Role.RewardModel, 0, awake=True, running=39, tokens=167963, bundle=0),
        _signal(PivotRL_Role.RewardModel, 1, awake=True, running=38, tokens=163713, bundle=8),
    ]

    plans = policy._wake_prefix_plans_for_role(
        target_signals=rollout,
        other_signals=rm,
        current_other_n=2,
        max_batch=1,
    )

    assert [signal.instance_id for signal in plans[1].wakes] == [1]
    assert plans[1].pre_sleep == ()
    assert plans[1].pre_wake == ()


def test_heterogeneous_small_wakes_charge_large_victim_phi_once():
    policy = _policy(
        model_params={"default": {"A": 0.001, "B": 10.0, "C": 1.0, "D": 0.0}},
        max_scale_instances_per_action=4,
        throughput_objective="sum",
        enable_heterogeneous_parallelism_candidates=True,
    )
    rollout = [
        _signal(PivotRL_Role.Rollout, 8, awake=True, running=3, tokens=30, bundle=8),
        *[
            _signal(PivotRL_Role.Rollout, instance_id, awake=False, running=0, tokens=0, bundle=instance_id)
            for instance_id in range(8)
        ],
    ]
    rm = [
        _signal(PivotRL_Role.RewardModel, 0, awake=True, running=39, tokens=167963, bundle=(0, 1, 2, 3)),
        _signal(PivotRL_Role.RewardModel, 1, awake=True, running=38, tokens=159014, bundle=(4, 5, 6, 7)),
    ]
    grouped = {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(grouped, router_backlog_by_role={})
    candidate = next(
        item
        for item in candidates
        if item.action.action_type == "scale_up"
        and item.action.role_name == PivotRL_Role.Rollout
        and item.action.num_instances == 4
    )

    assert candidate.action.preferred_instance_ids == [0, 1, 2, 3]
    assert candidate.action.pre_sleep_other_preferred == [
        {"role_name": PivotRL_Role.RewardModel, "model_name": "model", "instance_id": 0}
    ]
    assert candidate.action.pre_wake_other_preferred is None
    assert candidate.rm_n == 1
    assert candidate.phi == pytest.approx(policy._sleep_phi([rm[0]]))
    assert candidate.phi != pytest.approx(4.0 * policy._sleep_phi([rm[0]]))


def test_itl_scale_up_prefers_share_pool_over_train_pool():
    # Rollout is the bottleneck. Two wakeable instances with free bundles:
    # rollout[1] in the share pool (bundle 1) and rollout[2] in the train pool
    # (bundle 2). The trainer is idle so train_pool is available, but the share
    # pool has a free device, so the wake must pick the share-pool instance.
    policy = _policy(throughput_objective="sum")
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=5, tokens=50, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=1, pool_id="shared_rollout_pool"),
        _signal(PivotRL_Role.Rollout, 2, awake=False, running=0, tokens=0, bundle=2, pool_id="train_pool"),
    ]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=12, tokens=120, bundle=3)]
    grouped = {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(
        grouped, router_backlog_by_role={}, trainer_waiting_hint={"trainer_busy": False}
    )
    rollout_scale_ups = [
        c for c in candidates
        if c.action.action_type == "scale_up" and c.action.role_name == PivotRL_Role.Rollout
    ]

    assert rollout_scale_ups
    assert rollout_scale_ups[0].action.preferred_instance_ids == [1]


def test_itl_scale_up_falls_back_to_train_pool_when_share_pool_full():
    # Rollout is the bottleneck. The only share-pool wakeable (rollout[1],
    # bundle 1) collides with the awake rm[0] (bundle 1) and cannot be used.
    # The train-pool instance rollout[2] (bundle 2) is free, so the wake must
    # fall back to it.
    policy = _policy(throughput_objective="sum")
    rollout = [
        _signal(PivotRL_Role.Rollout, 0, awake=True, running=5, tokens=50, bundle=0),
        _signal(PivotRL_Role.Rollout, 1, awake=False, running=0, tokens=0, bundle=1, pool_id="shared_rollout_pool"),
        _signal(PivotRL_Role.Rollout, 2, awake=False, running=0, tokens=0, bundle=2, pool_id="train_pool"),
    ]
    rm = [_signal(PivotRL_Role.RewardModel, 0, awake=True, running=12, tokens=120, bundle=1, pool_id="shared_rollout_pool")]
    grouped = {PivotRL_Role.Rollout: rollout, PivotRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(
        grouped, router_backlog_by_role={}, trainer_waiting_hint={"trainer_busy": False}
    )
    rollout_scale_ups = [
        c for c in candidates
        if c.action.action_type == "scale_up" and c.action.role_name == PivotRL_Role.Rollout
    ]

    assert rollout_scale_ups
    assert rollout_scale_ups[0].action.preferred_instance_ids == [2]


def _router():
    router_cls = PivotRL_RewardModelRouter.__ray_metadata__.modified_class
    router = router_cls.__new__(router_cls)
    router.request_counts = {0: 0, 1: 2, 2: 4}
    router.worker_handles = [_Worker(0), _Worker(2), _Worker(4)]
    router.paused_worker_indices = set()
    router.worker_probe_backoff_until = {}
    router.worker_active_task_timeout_s = 1.0
    router.worker_probe_backoff_s = 1.0
    router.max_concurrent_requests_per_instance = None
    router.waiting_admission_cap = 3
    router.instance_to_engine_status = {}
    router._waiting_seq_counter = 0
    router._pending_count = 0
    router._load_cache = {}
    router._load_cache_ts = -1.0
    router.load_cache_ttl_s = 0.0
    router._itl_router_enable = True
    router._itl_router_max_itl = None
    router._itl_router_params = policy_params = _policy()._params_for_signal(
        _signal(PivotRL_Role.RewardModel, 0, awake=True, running=0, tokens=0, bundle=0)
    )
    assert policy_params.B == 10.0
    router.route_strategy = ITLBalanceRewardModelRouteStrategy(
        n_instances=len(router.worker_handles),
        strategy_kwargs={
            "itl_model_params": router._itl_router_params,
            "itl_max_itl": router._itl_router_max_itl,
        },
    )
    return router


def _active_loads(router) -> dict[int, int]:
    return {
        idx: max(0, router.worker_handles[idx].get_active_task_num.value, router.request_counts.get(idx, 0))
        for idx in router.request_counts
    }


def _select(router) -> int | None:
    return router._select_worker_for_request(SimpleNamespace(), _active_loads(router))


def _engine_status(instance_id: int, *, running: int, waiting: int) -> SimpleNamespace:
    return SimpleNamespace(
        instance_id=instance_id,
        model_version=0,
        snapshot={
            "scheduler_stats": {
                "num_running_reqs": running,
                "num_waiting_reqs": waiting,
            }
        },
    )


def test_rm_router_selects_smallest_post_route_itl():
    router = _router()
    router.route_strategy._itl_params = ITLModelParams(A=0.0, B=10.0, C=3.0, D=0.0)

    assert _select(router) == 0


def test_rm_router_task_num_cap_stops_full_workers():
    router = _router()
    router.max_concurrent_requests_per_instance = 2

    assert _select(router) == 0


def test_rm_router_waiting_admission_cap_filters_full_workers():
    router = _router()
    router.instance_to_engine_status = {
        0: _engine_status(0, running=5, waiting=3),
        1: _engine_status(1, running=5, waiting=2),
        2: _engine_status(2, running=6, waiting=3),
    }

    assert router._select_worker_for_request(SimpleNamespace(), {0: 8, 1: 7, 2: 9}) == 1


def test_rm_router_waiting_admission_cap_stops_at_running_plus_cap():
    router = _router()
    router.instance_to_engine_status = {
        0: _engine_status(0, running=5, waiting=2),
        1: _engine_status(1, running=4, waiting=3),
        2: _engine_status(2, running=6, waiting=3),
    }

    assert router._select_worker_for_request(SimpleNamespace(), {0: 8, 1: 7, 2: 9}) is None


def test_rm_router_waiting_admission_cap_bootstraps_without_status():
    router = _router()

    assert (
        router._select_worker_for_request(SimpleNamespace(), {0: 3, 1: 3, 2: 3})
        is None
    )
    assert router._select_worker_for_request(SimpleNamespace(), {0: 2, 1: 3, 2: 3}) == 0


def test_rm_router_itl_threshold_stops_routing():
    router = _router()
    router.route_strategy._itl_params = ITLModelParams(A=0.0, B=10.0, C=3.0, D=0.0)
    router.route_strategy._itl_max_itl = 9.0

    assert _select(router) is None


def test_rm_router_keeps_buffer_id_heap_order():
    router = _router()
    router.requests_to_route = []
    router._pending_count = 0
    router._waiting_seq_counter = 0
    req_late = SimpleNamespace(non_tensor_batch={"buffer_id": 5})
    req_early = SimpleNamespace(non_tensor_batch={"buffer_id": 1})
    req_missing = SimpleNamespace(non_tensor_batch={})

    router._enqueue_request("late", req_late)
    router._enqueue_request("missing", req_missing)
    router._enqueue_request("early", req_early)

    assert router._dequeue_request()[0] == "early"
    assert router._dequeue_request()[0] == "late"
    assert router._dequeue_request()[0] == "missing"


def test_rm_router_pending_summary_uses_front_t_requests():
    router = _router()
    router.requests_to_route = []
    router._pending_count = 0
    req_late = SimpleNamespace(non_tensor_batch={"buffer_id": 5}, batch={"attention_mask": [1, 1, 1]})
    req_early = SimpleNamespace(non_tensor_batch={"buffer_id": 1}, batch={"attention_mask": [1, 1]})
    req_missing = SimpleNamespace(non_tensor_batch={}, batch={"attention_mask": [1, 1, 1, 1]})

    router._enqueue_request("late", req_late)
    router._enqueue_request("missing", req_missing)
    router._enqueue_request("early", req_early)

    assert router.get_pending_request_summary(2) == {"pending": 3, "count": 2, "total_tokens": 5}


def test_rm_router_routing_update_signal_is_sticky():
    router = _router()
    router.routing_status_update_queue = asyncio.Queue(maxsize=1)

    router._signal_routing_update()

    asyncio.run(asyncio.wait_for(router._wait_for_routing_update(), timeout=0.5))


_QWEN3_8B_COST_MODEL = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "pivotrl",
        "trainer",
        "config",
        "cost_model",
        "qwen3_8b.json",
    )
)


def _throughput_optimal_strategy(max_concurrent_seqs_per_instance: int = 4):
    return ThroughputOptimalRewardModelRouteStrategy(
        n_instances=3,
        strategy_kwargs={
            "cost_model_path": _QWEN3_8B_COST_MODEL,
            "model_name": "qwen3_8b",
            "instance_to_tp_pp": {0: "TP1_PP1", 1: "TP1_PP1", 2: "TP1_PP1"},
            "max_num_waiting_reqs_after_preemption": 3,
            "max_concurrent_seqs_per_instance": max_concurrent_seqs_per_instance,
            "delta_throughput_threshold": 0.2,
            "max_prompt_length": 256,
            "request_budget": 1024,
            "instance_to_max_model_len": {0: 769, 1: 769, 2: 769},
        },
    )


def test_rm_router_throughput_optimal_blind_degrade_dispatches_eagerly():
    if not os.path.exists(_QWEN3_8B_COST_MODEL):
        pytest.skip("qwen3_8b cost model not available in this environment")
    strategy = _throughput_optimal_strategy(max_concurrent_seqs_per_instance=4)
    request = SimpleNamespace(non_tensor_batch={}, batch={"attention_mask": [1, 1]})

    # No engine status pushed yet -> blind degrade path picks least-loaded.
    assert strategy.route(request, candidates=[0, 1, 2], route_kwargs={"active_loads": {0: 0, 1: 2, 2: 4}}) == 0

    # Cap respected: every candidate at cap -> refuse.
    strategy.instance_to_request_num = {0: 4, 1: 4, 2: 4}
    assert strategy.route(request, candidates=[0, 1, 2], route_kwargs={"active_loads": {0: 0, 1: 0, 2: 0}}) is None
