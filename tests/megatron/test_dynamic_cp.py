"""Verify native dynamic CP scheduling preserves RL samples and normalization."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.engine.megatron.dynamic_cp import prepare_dynamic_cp_micro_batches, restore_dynamic_cp_results

pytestmark = pytest.mark.cpu_test


def make_batch(lengths):
    return TensorDict(
        {
            "input_ids": torch.nested.nested_tensor(
                [torch.arange(length) + row * 1000 for row, length in enumerate(lengths)], layout=torch.jagged
            ),
            "loss_mask": torch.nested.nested_tensor([torch.ones(length) for length in lengths], layout=torch.jagged),
            "sample_id": torch.arange(len(lengths)),
            "advantages": torch.arange(len(lengths)).float(),
            "multi_modal_inputs": TensorDict.lazy_stack(
                [TensorDict({"feature": torch.tensor(row)}) for row in range(len(lengths))]
            ),
        },
        batch_size=[len(lengths)],
    )


@pytest.mark.parametrize("lengths", [[1], [17, 33, 49], [8] * 17, [8, 16, 24, 40, 72, 96, 104, 128]])
@pytest.mark.parametrize("multiple", [None, 2])
def test_native_schedule_preserves_samples_and_fields(lengths, multiple):
    from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler

    batch = make_batch(lengths)
    if multiple is not None and len(lengths) < 2:
        with pytest.raises(AssertionError, match="enough ids"):
            prepare_dynamic_cp_micro_batches(batch, 32, 1, 8, 1, 0, multiple)
        return
    padded = [((length + 15) // 16) * 16 for length in lengths]
    expected = DefaultDynamicCPScheduler(32, 1, 8, multiple).get_groups_and_subsamples(list(enumerate(padded)))
    collected = []
    for rank in range(8):
        micro_batches, plan = prepare_dynamic_cp_micro_batches(batch, 32, 1, 8, 1, rank, multiple)
        assert plan.row_ids == [group[rank] for group in expected], "The native scheduler must own sample grouping."
        if multiple:
            assert len(micro_batches) % multiple == 0, "VPP schedules must have aligned round counts."
        for micro_batch, rows, leader in zip(micro_batches, plan.row_ids, plan.leaders, strict=True):
            cp_size = tu.get_non_tensor_data(micro_batch, "local_cp_size", default=None)
            assert sum(padded[row] for row in rows) <= 32 * cp_size, "Padding must fit the per rank budget."
            assert micro_batch["sample_id"].tolist() == rows, "Sample IDs must follow the schedule."
            assert micro_batch["advantages"].tolist() == rows, "PPO metadata must follow the selected rows."
            assert micro_batch["multi_modal_inputs"]["feature"].tolist() == rows, (
                "Multimodal fields must stay aligned."
            )
            assert [tokens[0].item() // 1000 for tokens in micro_batch["input_ids"].unbind()] == rows, (
                "Token tensors must follow the selected rows."
            )
            if leader:
                collected.extend(rows)
    assert sorted(collected) == list(range(len(lengths))), "Every sample must have exactly one output owner."


def test_native_schedule_uses_mixed_cp_sizes_in_one_round():
    batches = [
        prepare_dynamic_cp_micro_batches(make_batch([128, 64, 32, 32]), 32, 1, 8, 1, rank)[0] for rank in range(8)
    ]
    sizes = {tu.get_non_tensor_data(batches[0], "local_cp_size", default=None) for batches in batches}
    assert sizes == {1, 2, 4}, "One round must support different CP group sizes."


@pytest.mark.parametrize("lengths,budget,pool", [([257], 32, 8), ([0], 32, 8), ([1], 0, 8), ([1], 32, 3)])
def test_invalid_capacity_fails_before_scheduling(lengths, budget, pool):
    with pytest.raises(ValueError):
        prepare_dynamic_cp_micro_batches(make_batch(lengths), budget, 1, pool, 1, 0)


def test_result_restore_removes_cp_duplicates_and_normalizes_metrics():
    batch = make_batch([128, 64, 32, 32])
    gathered = []
    for rank in range(8):
        micro_batches, plan = prepare_dynamic_cp_micro_batches(batch, 32, 1, 8, 1, rank)
        records = []
        for micro_batch, rows, leader in zip(micro_batches, plan.row_ids, plan.leaders, strict=True):
            if leader:
                weight = sum(batch["input_ids"].offsets().diff()[row].item() for row in rows)
                contribution = sum(rows) * 8
                metrics = {
                    "objective": Metric(AggregationType.SUM, contribution),
                    "mean": Metric(AggregationType.MEAN, sum(rows) / len(rows)),
                }
                records.append(
                    (rows, {"log_probs": list(micro_batch["input_ids"].unbind())}, contribution, metrics, weight)
                )
        gathered.append(records)
    result, stats = restore_dynamic_cp_results(gathered, 4, 8, torch.device("cpu"))
    assert [row[0].item() // 1000 for row in result["log_probs"].unbind()] == list(range(4)), (
        "Collected tensors must return to the original input order."
    )
    assert stats["loss"] == [6], "Global loss must undo the physical DP normalization."
    assert stats["metrics"]["objective"].aggregate() == 6, "SUM metrics must count each sample once."
    assert stats["metrics"]["mean"].aggregate() == pytest.approx((64 + 64 + 96) / 256), (
        "MEAN metrics must weight unique valid tokens."
    )
    with pytest.raises(ValueError, match="duplicate"):
        restore_dynamic_cp_results(gathered + gathered, 4, 8, torch.device("cpu"))
    with pytest.raises(ValueError, match="do not match"):
        restore_dynamic_cp_results(gathered, 5, 8, torch.device("cpu"))


@pytest.mark.parametrize("dynamic_cp", [False, True])
def test_result_collection_preserves_rows_and_metric_participants(monkeypatch, dynamic_cp):
    from itertools import product

    from verl.workers.engine.megatron import transformer_impl

    engine = object.__new__(transformer_impl.MegatronEngine)
    engine.engine_config = SimpleNamespace(dynamic_context_parallel=dynamic_cp)
    ranks = {}
    for axis in ("tensor_model", "pipeline_model", "context", "data"):
        monkeypatch.setattr(transformer_impl.mpu, f"get_{axis}_parallel_rank", lambda axis=axis: ranks[axis])
    monkeypatch.setattr(transformer_impl.mpu, "get_pipeline_model_parallel_world_size", lambda: 2)
    collected_rows = []
    metric_participants = []
    sample_count = 32
    for dp, pp, tp, cp in product(range(4), range(2), range(2), range(2)):
        ranks.update(data=dp, pipeline_model=pp, tensor_model=tp, context=cp)
        if engine.is_mp_src_rank_with_outputs():
            metric_participants.append(dp)
        if engine.is_dispatch_collect_rank():
            rows = range(sample_count) if dynamic_cp else range(dp * 8, (dp + 1) * 8)
            collected_rows.extend(rows)

    assert collected_rows == list(range(sample_count)), "Each sample must be collected exactly once in batch order."
    assert metric_participants == list(range(4)), "Every physical DP source must participate in metric collectives."


@pytest.mark.parametrize("dynamic_cp", [False, True])
@pytest.mark.parametrize("global_rank", [0, 1])
@pytest.mark.parametrize("per_token_loss", [False, True])
def test_engine_selects_one_microbatch_organizer(monkeypatch, caplog, dynamic_cp, global_rank, per_token_loss):
    from verl.workers.engine.megatron import transformer_impl as impl
    from verl.workers.engine.megatron.dynamic_cp import DynamicCPPlan

    engine = object.__new__(impl.MegatronEngineWithLMHead)
    engine.engine_config = SimpleNamespace(
        dynamic_context_parallel=dynamic_cp, context_parallel_size=1, max_seqlen_per_dp_cp_rank=32
    )
    engine.tf_config = SimpleNamespace(fp8=None, calculate_per_token_loss=per_token_loss)
    engine.model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.module = [object()]
    data = make_batch([17, 33])
    calls = []
    monkeypatch.setattr(impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(impl.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None)
    monkeypatch.setattr(impl.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(impl.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(impl.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(impl.mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(impl.mpu, "get_data_parallel_world_size", lambda: 4)
    monkeypatch.setattr(impl.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(impl.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    for axis in ("tensor_model", "pipeline_model", "context", "data"):
        monkeypatch.setattr(impl.mpu, f"get_{axis}_parallel_rank", lambda: 0)
    monkeypatch.setattr(impl.mpu, "get_data_parallel_group", lambda **kwargs: None)
    monkeypatch.setattr(impl.mpu, "is_pipeline_last_stage", lambda **kwargs: True)
    monkeypatch.setattr(impl.torch.distributed, "get_rank", lambda *args: global_rank)
    monkeypatch.setattr(impl.torch.distributed, "get_process_group_ranks", lambda group: list(range(4)))
    monkeypatch.setattr(impl.torch.distributed, "all_reduce", lambda *args, **kwargs: calls.append("normalizer"))
    monkeypatch.setattr(impl.RouterReplayHelper, "is_r2_record_action", lambda *args: False)
    monkeypatch.setattr(engine, "get_data_parallel_size", lambda: 4)

    def organize(**kwargs):
        calls.append("native" if dynamic_cp else "verl")
        plan = DynamicCPPlan([[0, 1]], [True], 2, 4) if dynamic_cp else None
        return [kwargs["data"]], plan

    def forbidden(**kwargs):
        pytest.fail("The other microbatch organizer must not run.")

    monkeypatch.setattr(impl, "prepare_dynamic_cp_micro_batches", organize if dynamic_cp else forbidden)
    monkeypatch.setattr(impl, "prepare_micro_batches", forbidden if dynamic_cp else organize)
    monkeypatch.setattr(impl, "get_forward_backward_func", lambda: lambda **kwargs: [])
    monkeypatch.setattr(impl, "collect_dynamic_cp_results", lambda *args: {"native": True})
    monkeypatch.setattr(impl, "postprocess_batch_func", lambda **kwargs: {"verl": True})
    loss_function = (lambda **kwargs: None) if per_token_loss else None
    result = engine.forward_backward_batch(data, loss_function, forward_only=True)
    expected_calls = ["native"] if dynamic_cp else ["normalizer"] * (2 if per_token_loss else 1) + ["verl"]
    assert calls == expected_calls, (
        "Only the selected organizer may run, and replicated normalizers must not be reduced."
    )
    if per_token_loss:
        assert data["routed_num_tokens"] == 50, "The full input count must reach every microbatch."
    assert result == {"native" if dynamic_cp else "verl": True}, "The matching output collector must run."
    if dynamic_cp and global_rank == 0:
        packing_events = [
            record.training_diagnostics
            for record in caplog.records
            if getattr(record, "training_diagnostics", {}).get("event") == "phase_end"
            and record.training_diagnostics.get("phase") == "packing"
        ]
        assert packing_events[0]["pool_global_ranks"] == list(range(4)), (
            "The scheduler context must include its actual process group rank map."
        )


def test_collection_broadcasts_restore_errors_before_raising(monkeypatch):
    from verl.workers.engine.megatron import dynamic_cp

    data = make_batch([17])
    plan = dynamic_cp.DynamicCPPlan([[0]], [True], 1, 1)
    outputs = [{"model_output": {"log_probs": data["input_ids"]}, "loss": 1.0, "metrics": {}}]
    broadcasts = []
    monkeypatch.setattr(dynamic_cp.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dynamic_cp.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(dynamic_cp.dist, "get_process_group_ranks", lambda group: [0])

    def duplicate_records(records, gathered, **kwargs):
        gathered[0] = records * 2

    monkeypatch.setattr(dynamic_cp.dist, "gather_object", duplicate_records)
    monkeypatch.setattr(dynamic_cp.dist, "broadcast_object_list", lambda stats, **kwargs: broadcasts.append(stats[0]))
    with pytest.raises(RuntimeError, match="duplicate"):
        dynamic_cp.collect_dynamic_cp_results(outputs, plan, data, None)
    assert len(broadcasts) == 1, "Restore failures must reach every rank before any participant raises."


def test_native_metrics_cross_worker_and_trainer_boundaries(monkeypatch):
    import copy
    from contextlib import nullcontext

    from psrl.trainer.ppo.ray_trainer import PSRL_RayPPOTrainer
    from verl.workers import engine_workers

    worker = object.__new__(engine_workers.TrainingWorker)
    worker.device_name = "cpu"
    worker.engine = SimpleNamespace(
        get_data_parallel_size=lambda: 1,
        get_data_parallel_rank=lambda: 0,
        get_data_parallel_group=lambda: "physical_dp",
        train_mode=lambda **kwargs: nullcontext(),
        is_mp_src_rank_with_outputs=lambda: True,
    )
    worker.flops_counter = SimpleNamespace(estimate_flops=lambda *args, **kwargs: (4.0, 8.0))
    monkeypatch.setattr(engine_workers.torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(engine_workers.torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    def gather_objects(object_list, obj, group=None):
        object_list[:] = [copy.deepcopy(obj), copy.deepcopy(obj)]

    monkeypatch.setattr(engine_workers.torch.distributed, "all_gather_object", gather_objects)
    monkeypatch.setattr(
        engine_workers,
        "get_torch_device",
        lambda: SimpleNamespace(max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0),
    )

    def train_batch(data):
        value = float((data["sample_id"] + 1).sum())
        rows = list(range(len(data)))
        records = [
            (
                rows,
                {},
                value * 2,
                {
                    "objective": Metric(AggregationType.SUM, value * 2),
                    "mean": Metric(AggregationType.MEAN, value / len(data)),
                    "coefficient": 0.1,
                },
                len(data),
            )
        ]
        _, stats = restore_dynamic_cp_results([records], len(data), 2, torch.device("cpu"))
        return worker._postprocess_output(
            {"model_output": {}, **stats},
            global_token_num=[1, 1],
            delta_time=1,
            forward_only=False,
            images_seqlens=None,
        )

    worker.train_batch = train_batch
    data = make_batch([17, 33, 49, 65])
    tu.assign_non_tensor(data, mini_batch_size=2, dataloader_kwargs={"shuffle": False})
    output = worker.train_mini_batch(data)
    reduced = PSRL_RayPPOTrainer._reduce_scheduled_metrics([output], "actor/", "perf/mfu")
    assert reduced["actor/objective"] == 5.0, "Worker and trainer reductions must preserve normalized SUM metrics."
    assert reduced["actor/mean"] == 2.5, "Worker and trainer reductions must preserve MEAN metrics."
    assert reduced["actor/coefficient"] == 0.1, "Scalar metrics must remain numeric across worker boundaries."
    assert reduced["actor/loss"] == 5.0, "Loss must not be counted again across physical DP replicas."
