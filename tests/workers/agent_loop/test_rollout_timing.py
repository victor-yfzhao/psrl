"""
Verify turn timing and aggregation across the rollout metadata boundary.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.utils.rollout.loop_timer import LoopTimer
from psrl.utils.rollout.timing_metrics import compute_rollout_timing_metrics
from psrl.workers.agent_loop.agent_data.conversation_agent_data import ConversationAgentData
from psrl.workers.agent_loop.agent_data.mini_swe_agent_data import MiniSWEAgentData
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.gen.utils import TokenOutput
from verl.utils import tensordict_utils as tu

pytestmark = pytest.mark.cpu_test


def test_timer_records_each_call_and_preserves_failed_environment_time(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr("psrl.utils.rollout.loop_timer.time.perf_counter", lambda: clock.now)
    timer = LoopTimer()
    with timer.generation():
        clock.now += 2
    with timer.env():
        clock.now += 3
    snapshot = timer.as_dict()
    clock.now += 100
    with timer.generation():
        clock.now += 4
    with pytest.raises(TimeoutError), timer.env():
        clock.now += 7
        raise TimeoutError("Environment command timed out.")

    timing = timer.as_dict()
    assert timing["generation_turns_s"] == [2, 4]
    assert timing["env_turns_s"] == [3, 7]
    assert timing["generation_s"] == 6
    assert timing["env_s"] == 10
    assert timing["elapsed_s"] == 116
    assert snapshot["generation_turns_s"] == [2]
    assert snapshot["env_turns_s"] == [3]


def test_step_metrics_deduplicate_requests_and_weight_each_turn_equally():
    first = {"generation_turns_s": [1, 9], "env_turns_s": [3, 5], "grading_s": 900}
    second = {"generation_turns_s": [2], "env_turns_s": [7], "grading_s": 100}
    metrics = compute_rollout_timing_metrics([11, 11, 22], [first, first.copy(), second])

    for level, phase, count, mean, minimum, maximum in (
        ("request", "rollout", 2, 6, 2, 10),
        ("request", "env", 2, 7.5, 7, 8),
        ("turn", "rollout", 3, 4, 1, 9),
        ("turn", "env", 3, 5, 3, 7),
    ):
        prefix = f"rollout_timing/{level}/{phase}_s"
        assert metrics[f"{prefix}/count"] == count
        assert metrics[f"{prefix}/mean"] == mean
        assert metrics[f"{prefix}/min"] == minimum
        assert metrics[f"{prefix}/max"] == maximum


def test_no_environment_calls_do_not_create_artificial_zero_duration_turns():
    metrics = compute_rollout_timing_metrics([1], [{"generation_turns_s": [2], "env_turns_s": []}])
    assert metrics["rollout_timing/request/env_s/mean"] == 0
    assert metrics["rollout_timing/turn/env_s/count"] == 0
    assert "rollout_timing/turn/env_s/min" not in metrics


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_invalid_durations_cannot_silently_corrupt_step_metrics(duration):
    with pytest.raises(ValueError, match="Invalid rollout durations"):
        compute_rollout_timing_metrics([1], [{"generation_turns_s": [duration], "env_turns_s": []}])


def test_timing_records_must_match_their_requests():
    first = {"generation_turns_s": [1], "env_turns_s": []}
    second = {"generation_turns_s": [2], "env_turns_s": []}
    with pytest.raises(ValueError, match="Conflicting timing records"):
        compute_rollout_timing_metrics([1, 1], [first, second])
    with pytest.raises(ValueError, match="zip"):
        compute_rollout_timing_metrics([1, 2], [first])


@pytest.mark.asyncio
async def test_mini_swe_timing_survives_output_serialization_without_outer_timer_overwrite(monkeypatch):
    timing = {
        "generation_s": 10,
        "env_s": 8,
        "elapsed_s": 18,
        "generation_turns_s": [1, 9],
        "env_turns_s": [3, 5],
    }
    data = MiniSWEAgentData.__new__(MiniSWEAgentData)
    data.session_data = SimpleNamespace(
        trajectories=[SimpleNamespace(assistant_turns=2)],
        agent_reward_info={},
    )
    data.patch = "patch"
    data.grading_context = {}
    data.set_timing(timing)
    outputs = [TokenOutput(prompt_ids=[1], response_ids=[2], response_mask=[1]) for _ in range(2)]
    monkeypatch.setattr(ConversationAgentData, "finalize_output", AsyncMock(return_value=outputs))

    assert await data.finalize_output() is outputs
    AgentLoopBase._attach_loop_timing(SimpleNamespace(timer=LoopTimer()), outputs)
    fields = []
    for output in outputs:
        field = output.as_dict()
        field["uid"] = 11
        fields.append(field)
    batch = tu.list_of_dict_to_tensordict(fields)
    metrics = compute_rollout_timing_metrics(tu.get(batch, "uid"), tu.get(batch, "loop_timing"))

    assert "timing" not in data.session_data.agent_reward_info
    assert metrics["rollout_timing/request/rollout_s/mean"] == 10
    assert metrics["rollout_timing/request/env_s/mean"] == 8
    assert metrics["rollout_timing/turn/rollout_s/mean"] == 5
    assert metrics["rollout_timing/turn/env_s/count"] == 2
    summary = AgentLoopBase._build_summary_text(SimpleNamespace(), outputs[-1], SimpleNamespace(value="finished"))
    assert "generation: 10.0s | env: 8.0s" in summary
    assert "grading:" not in summary, "Rollout summaries must not report reward service grading as rollout time."
