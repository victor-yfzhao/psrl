from __future__ import annotations

import importlib.util
import json
import sys
from datetime import timedelta
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "extract_elastic_simulation_inputs.py"
_SPEC = importlib.util.spec_from_file_location("extract_elastic_simulation_inputs", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

parse_candidate = _MODULE.parse_candidate
parse_cycle_ids = _MODULE.parse_cycle_ids
parse_role_instance_ids = _MODULE.parse_role_instance_ids
parse_role_instance_loads = _MODULE.parse_role_instance_loads
parse_stat_snapshot_line = _MODULE.parse_stat_snapshot_line
iter_policy_cycles = _MODULE.iter_policy_cycles


def test_parse_stat_snapshot_line_builds_request_lengths() -> None:
    line = (
        "2026-09-04 01:54:48,679 - stats_collector.py - 268 - "
        "Snapshot (model version 3): {'timestamp': '2026-09-04T01:54:48.679064', "
        "'scheduler_stats': {'req_id_to_prompt_token_num': {'a-1': 100, 'b-2': 20}, "
        "'req_id_to_response_token_num': {'a-1': 30, 'b-2': 4}, "
        "'req_id_in_waiting': ['b-2'], 'num_running_reqs': 1, "
        "'num_waiting_reqs': 1, 'kv_cache_usage': 0.25}, "
        "'generation_throughput': 12.5}"
    )
    point = parse_stat_snapshot_line(line)
    assert point is not None
    assert point.model_version == 3
    assert point.request_ids == ("a-1", "b-2")
    assert point.request_length("a-1") == 130
    assert point.request_length("b-2") == 24
    assert point.waiting_ids == {"b-2"}
    assert point.running_count == 1
    assert point.waiting_count == 1


def test_parse_candidate_normalizes_python_repr() -> None:
    body = (
        "* candidate[2] scale_up RewardModel/Qwen3-30B-A3B-Thinking-2507 "
        "num_instances=1 preferred=[4] pre_wake=None "
        "pre_sleep=[{'role_name': <PSRL_Role.Rollout: 2>, "
        "'model_name': 'Qwen2.5-7B', 'instance_id': 9}]"
    )
    candidate = parse_candidate(body)
    assert candidate is not None
    assert candidate.index == 2
    assert candidate.pre_sleep == ({"role_name": "Rollout", "instance_id": 9},)
    assert candidate.role_plans == {
        "Rollout": {
            "wake_instance_ids": [],
            "sleep_instance_ids": [9],
            "primary_scale_up": False,
        },
        "RewardModel": {
            "wake_instance_ids": [4],
            "sleep_instance_ids": [],
            "primary_scale_up": True,
        },
    }


def test_parse_role_instance_ids_uses_current_state_rows() -> None:
    assert parse_role_instance_ids(
        "  * rollout_instances=[id0:req=4.0,tok=1338,tp=701.9; "
        "id56:req=4.0,tok=1260,tp=702.3]"
    ) == ("Rollout", (0, 56))
    assert parse_role_instance_ids("  * rm_instances=[]") == ("RewardModel", ())
    parsed = parse_role_instance_loads(
        "  * rollout_instances=[id0:req=4.0,tok=1338,tp=701.9]"
    )
    assert parsed is not None
    assert parsed[1][0].request_count == 4.0
    assert parsed[1][0].token_count == 1338


def test_iter_policy_cycles_attaches_timing_and_summaries(tmp_path: Path) -> None:
    path = tmp_path / "ScalingPolicy.log"
    path.write_text(
        "\n".join(
            [
                "2026-09-04 01:54:01,139 - scaling_policy.py - 352 - "
                "elastic_rm_policy request_level_candidate_evaluation | "
                "candidate_scoring_s='0.000328' candidates=1 "
                "rebalance_simulation_s='0.000217' router_simulation_s='0.000791' "
                "elapsed_s='0.100000'",
                "2026-09-04 01:54:01,141 - scaling_policy.py - 86 - "
                "elastic_rm_policy ========== cycle 7 BEGIN ==========",
                "2026-09-04 01:54:01,141 - scaling_policy.py - 88 - "
                "elastic_rm_policy cycle=7 |   vllm_current_queue_scope=running "
                "router_waiting_top_t=-1",
                "2026-09-04 01:54:01,141 - scaling_policy.py - 88 - "
                "elastic_rm_policy cycle=7 |   rollout_n=2 rollout_total_req=3.0 "
                "rollout_total_tok=300 rollout_role_tp=1.0 rollout_router_req=1 "
                "rollout_router_tok=100",
                "2026-09-04 01:54:01,141 - scaling_policy.py - 88 - "
                "elastic_rm_policy cycle=7 |   * rollout_instances=["
                "id0:req=2.0,tok=200,tp=1.0; id2:req=1.0,tok=100,tp=1.0]",
                "2026-09-04 01:54:01,142 - scaling_policy.py - 88 - "
                "elastic_rm_policy cycle=7 |   * candidate[0] scale_down "
                "Rollout/Qwen2.5-7B num_instances=1 preferred=[1] pre_wake=None "
                "pre_sleep=None",
                "2026-09-04 01:54:01,143 - scaling_policy.py - 88 - "
                "elastic_rm_policy cycle=7 | >>> OUTCOME: action | "
                "reason=itl_best_scale_down_Rollout",
                "2026-09-04 01:55:01,141 - scaling_policy.py - 86 - "
                "elastic_rm_policy ========== cycle 8 BEGIN ==========",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cycles = list(iter_policy_cycles(path))
    assert len(cycles) == 2
    cycle = cycles[0]
    assert cycle.cycle_id == 7
    assert cycle.policy_fields["vllm_current_queue_scope"] == "running"
    assert cycle.role_summary["Rollout"]["router_request_count"] == 1.0
    assert cycle.role_instance_ids["Rollout"] == (0, 2)
    assert sum(load.request_count for load in cycle.role_instance_loads["Rollout"]) == 3.0
    assert cycle.evaluation_timing is not None
    assert cycle.evaluation_timing["router_simulation_s"] == 0.000791
    assert cycle.simulation_timestamp == cycle.evaluation_timing_timestamp - timedelta(seconds=0.1)
    assert cycle.accepted_action_role == "Rollout"
    assert cycle.candidates[0].role_plans["Rollout"]["sleep_instance_ids"] == [1]
    assert cycles[1].evaluation_timing is None
    assert cycles[1].simulation_timestamp == cycles[1].timestamp

    # The candidate and timing structures contain only JSON-compatible values.
    json.dumps(
        {
            "cycle": cycle.cycle_id,
            "candidates": [candidate.role_plans for candidate in cycle.candidates],
            "timing": cycle.evaluation_timing,
        }
    )


def test_parse_cycle_ids_deduplicates_values() -> None:
    assert parse_cycle_ids("7, 11,7") == {7, 11}
