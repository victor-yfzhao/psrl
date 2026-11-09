"""Exercise complete native CP plans through the persistent logging sink."""

import json
import logging

import pytest
import torch
from psrl.utils.logger.dynamic_cp_logger import setup_dynamic_cp_logger
from tensordict import TensorDict
from verl.utils.training_diagnostics import log_training_event, training_phase
from verl.workers.engine.megatron import dynamic_cp

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def schedule_logger(monkeypatch):
    psrl_logger = logging.getLogger("verl.utils.training_diagnostics")
    monkeypatch.setattr(psrl_logger, "handlers", [])
    monkeypatch.setattr(psrl_logger, "disabled", False)
    try:
        yield psrl_logger
    finally:
        for handler in psrl_logger.handlers:
            handler.close()


@pytest.mark.parametrize("forward_only", [False, True])
def test_native_plan_is_persisted_once_across_tensor_and_pool_ranks(
    tmp_path, monkeypatch, schedule_logger, forward_only
):
    path = setup_dynamic_cp_logger(str(tmp_path))
    lengths = [128, 64, 32, 32]
    data = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor(
                [torch.zeros(length, dtype=torch.long) for length in lengths], layout=torch.jagged
            ),
        },
        batch_size=[len(lengths)],
    )
    global_rank = [0]
    monkeypatch.setattr(dynamic_cp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dynamic_cp.dist, "get_rank", lambda *args, **kwargs: global_rank[0])
    for rank in range(16):
        global_rank[0] = rank
        with training_phase(
            "packing",
            global_steps=7,
            forward_only=forward_only,
            tp_rank=rank % 2,
            pp_rank=0,
            pool_global_ranks=list(range(0, 16, 2)),
        ):
            dynamic_cp.prepare_dynamic_cp_micro_batches(data, 32, 2, 8, 1, rank // 2)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1, "Only global rank zero may persist the complete schedule."
    record = records[0]
    assert record["global_steps"] == 7 and record["forward_only"] == forward_only, (
        "Schedules must retain the enclosing training step and phase."
    )
    assert record["rank"] == 0 and record["tp_rank"] == 0 and record["pp_rank"] == 0, (
        "The global rank map must identify its tensor and pipeline slice."
    )
    assert record["input_lengths"] == lengths and record["padded_lengths"] == lengths, (
        "Schedules must preserve exact input lengths and scheduler padding."
    )
    assert record["num_rounds"] == 1 and record["token_budget_fill_fraction"] == 1.0, (
        "This mixed CP round must fill the complete token budget."
    )
    assignments = record["rounds"][0]["assignments"]
    assert {assignment["cp_size"] for assignment in assignments} == {1, 2, 4}, (
        "The persisted schedule must include every distinct CP group."
    )
    assert sorted(row for assignment in assignments for row in assignment["row_indices"]) == list(range(4)), (
        "Every input sample must appear once without CP replication."
    )
    assert sorted(rank for assignment in assignments for rank in assignment["pool_ranks"]) == list(range(8)), (
        "The complete schedule must cover every DP and CP pool rank."
    )
    assert all(assignment["tokens_per_rank"] == 32 for assignment in assignments), (
        "Per rank token counts must account for CP splitting."
    )
    assert record["row_index_scope"] == "input_minibatch", "Sample indices must declare their scope."


def test_schedule_sink_appends_without_duplicates_after_handler_close(tmp_path, schedule_logger):
    path = tmp_path / "dynamic_cp.jsonl"
    path.write_text('{"event": "existing"}\n')
    assert setup_dynamic_cp_logger(str(tmp_path)) == path, "The sink must use the configured run directory."
    setup_dynamic_cp_logger(str(tmp_path))
    assert len(schedule_logger.handlers) == 1, "Repeated setup must not add duplicate handlers."
    log_training_event("phase_start")
    log_training_event("dynamic_cp_schedule", sequence=1)
    schedule_logger.handlers[0].close()
    log_training_event("dynamic_cp_schedule", sequence=2)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["existing", "dynamic_cp_schedule", "dynamic_cp_schedule"], (
        "Only schedule events must be appended without truncating earlier records."
    )
    assert [record["sequence"] for record in records[1:]] == [1, 2], (
        "The append sink must keep recording after logging handlers are closed."
    )
