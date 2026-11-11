# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Verify training diagnostics preserve computation and expose compiler boundaries.
"""

import json
import logging

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils import training_diagnostics as diagnostics
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.engine.base import BaseEngine, BaseEngineCtx

pytestmark = pytest.mark.cpu_test


def events(caplog):
    return [
        json.loads(record.getMessage().removeprefix("[TrainingDiagnostics] ").removesuffix("."))
        for record in caplog.records
        if record.name == diagnostics.psrl_logger.name
    ]


def test_phase_error_keeps_exception_and_restores_context(caplog, monkeypatch):
    caplog.set_level(logging.INFO)

    def forbidden_sync(*args, **kwargs):
        raise AssertionError("Diagnostic logging must not synchronize devices or process groups.")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden_sync)
    monkeypatch.setattr(torch.distributed, "barrier", forbidden_sync)
    timings = {}
    with pytest.raises(ValueError, match="probe failure"):
        with diagnostics.training_phase("outer", global_steps=7):
            with diagnostics.training_phase("inner", timings=timings):
                raise ValueError("probe failure")
    with diagnostics.training_phase("next"):
        pass
    records = events(caplog)
    ends = [record for record in records if record["event"] == "phase_end"]
    assert [record["outcome"] for record in ends] == ["error", "error", "ok"], "Errors must retain their scope."
    assert ends[0]["parent_phase_id"] == ends[1]["phase_id"], "Child phases must reference their parent."
    assert ends[0]["error_type"] == "ValueError", "The original error type must be observable."
    assert ends[2]["parent_phase_id"] is None, "An exception must not leak context into another call."
    assert "global_steps" not in ends[2], "Step metadata must not leak into another call."
    assert timings["inner"] >= 0, "Timings must use a nonnegative host wall clock."


def test_engine_logs_forward_and_optimizer_without_changing_calls(caplog):
    calls = []

    class ProbeEngine(BaseEngine):
        def optimizer_zero_grad(self):
            calls.append("zero")

        def forward_backward_batch(self, data, loss_function, forward_only=False):
            calls.append(("forward", forward_only))
            return {"metrics": {}}

        def optimizer_step(self):
            calls.append("step")
            return 1.5

        def is_mp_src_rank_with_outputs(self):
            return True

    data = TensorDict({"input_ids": torch.ones(2, 3)}, batch_size=[2])
    tu.assign_non_tensor(data, global_steps=9, training_diagnostics={"parameter_version": 4})
    result = ProbeEngine().train_batch(data, lambda: None)
    assert result["metrics"]["grad_norm"] == 1.5, "The engine result must remain unchanged."
    assert calls == ["zero", ("forward", False), "step"], "Logging must preserve engine execution order."
    starts = [record for record in events(caplog) if record["event"] == "phase_start"]
    assert [record["phase"] for record in starts] == ["engine_train_batch", "forward_backward", "optimizer_step"], (
        "Compute and optimizer phases must be distinguishable."
    )
    assert all(record["global_steps"] == 9 for record in starts), "Every nested phase must retain the step."
    assert all(record["parameter_version"] == 4 for record in starts), "Every nested phase must retain the version."


def test_context_logs_actual_load_and_offload_flags(caplog, monkeypatch):
    from types import SimpleNamespace

    from verl.workers.engine import base

    calls = []
    engine = SimpleNamespace(
        is_param_offload_enabled=False,
        is_optimizer_offload_enabled=True,
        to=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(base, "get_device_name", lambda: "cuda")
    with diagnostics.training_phase("update", global_steps=2), BaseEngineCtx(engine, "train"):
        pass
    assert [call["device"] for call in calls] == ["cuda", "cpu"], "The original movement order must be retained."
    assert all(call["optimizer"] and not call["model"] for call in calls), "Movement flags must remain unchanged."
    ends = [record for record in events(caplog) if record["event"] == "phase_end"]
    assert [record["phase"] for record in ends] == ["engine_load", "engine_offload", "update"], (
        "Automatic movement must have separate boundaries."
    )
    assert all(record["clock"] == "host" for record in ends), "Host time must not be labeled as GPU kernel time."


def test_micro_batch_observer_consumes_one_input(caplog):
    first = TensorDict(
        {"input_ids": torch.nested.nested_tensor([torch.arange(2), torch.arange(5)], layout=torch.jagged)},
        batch_size=[2],
    )
    tu.assign_non_tensor(first, training_micro_batch_index=3, global_steps=8)
    sentinel = object()
    iterator = iter((first, sentinel))

    class Worker:
        @diagnostics.trace_micro_batch_forward
        def forward(self, batches):
            return next(batches)

    assert Worker().forward(iterator) is first, "The model must receive the original TensorDict."
    assert next(iterator) is sentinel, "Observation must not consume another micro batch."
    start = next(record for record in events(caplog) if record["event"] == "phase_start")
    assert start["micro_batch_index"] == 3, "Compiler events need the micro batch identity."
    assert start["input_storage_tokens"] == 7, "Jagged tensor storage must exclude padding."
    assert start["rows"] == 2, "The row count must describe the local micro batch."


@pytest.mark.parametrize("group_size", [1, 2, 3])
def test_packing_plan_matches_real_sample_assignments(group_size, caplog):
    lengths = [3, 4, 5, 7, 9, 11]
    data = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor([torch.arange(n) for n in lengths], layout=torch.jagged),
            "sample_id": torch.arange(len(lengths)),
        },
        batch_size=[len(lengths)],
    )
    with diagnostics.training_phase("packing", global_steps=1):
        batches, partitions = rearrange_micro_batches(data, max_token_len=32, force_group_size=group_size)
    assert sorted(row for partition in partitions for row in partition) == list(range(len(lengths))), (
        "Diagnostics must preserve complete sample coverage."
    )
    plan = next(record for record in events(caplog) if record["event"] == "micro_batch_plan")
    assert plan["num_micro_batches"] == len(batches), "The plan must describe the executed partition count."
    for batch, rows, shape in zip(batches, partitions, plan["micro_batches"], strict=True):
        assert batch["sample_id"].tolist() == rows, "Rows and their tensor fields must stay aligned."
        assert shape["input_lengths"] == [lengths[row] for row in rows], "Logged lengths must match actual rows."
        assert shape["input_tokens"] == sum(lengths[row] for row in rows), "Token sums must match the local work."
        assert shape["sum_input_length_squared"] == sum(lengths[row] ** 2 for row in rows), (
            "The attention workload statistic must retain the length distribution."
        )
        assert tu.get(batch, "training_micro_batch_shape")["input_tokens"] == shape["input_tokens"], (
            "The forward observer must receive the executed packing shape."
        )


def test_snapshot_byte_counts_do_not_copy_values():
    state = {"weights": torch.arange(7), "nested": ([torch.zeros(3)], "metadata")}
    assert diagnostics.snapshot_tensor_stats(state) == {"tensor_bytes": 68, "pinned_bytes": 0, "tensor_count": 2}, (
        "Snapshot diagnostics must count logical tensor bytes without changing tensors."
    )


def test_real_compilations_retain_step_and_micro_batch_context(caplog, monkeypatch):
    from torch._dynamo.callback import callback_handler

    monkeypatch.setattr(callback_handler, "start_callbacks", list(callback_handler.start_callbacks))
    monkeypatch.setattr(callback_handler, "end_callbacks", list(callback_handler.end_callbacks))
    monkeypatch.setattr(diagnostics, "_compile_observer_installed", False)
    starts_before = len(callback_handler.start_callbacks)
    diagnostics.install_compile_observer()
    diagnostics.install_compile_observer()
    assert len(callback_handler.start_callbacks) == starts_before + 1, "Observers must be installed once per process."

    def transform(value):
        return value.sin() * 2

    compiled = torch.compile(transform, backend="eager", dynamic=False)
    for index, size in enumerate((2, 3)):
        with diagnostics.training_phase("micro_batch_forward", global_steps=5, micro_batch_index=index):
            assert torch.equal(compiled(torch.ones(size)), transform(torch.ones(size))), (
                "Compiler observation must preserve the numerical result."
            )
    records = [record for record in events(caplog) if record["event"] == "compile_end"]
    assert {record["micro_batch_index"] for record in records} == {0, 1}, "Both input shapes must be observable."
    assert all(record["global_steps"] == 5 for record in records), "Compilation must retain its training step."
    assert all(record["wall_s"] >= 0 for record in records), "Compilation must have a host duration."
    assert all(record["compile_id"] for record in records), "Compilation IDs must be logged for later joins."
