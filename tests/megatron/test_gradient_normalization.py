# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""
Verify Megatron's loss callback preserves a fixed global objective across parallel layouts.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from megatron.core.pipeline_parallel.schedules import forward_step_calc_loss
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from tensordict import TensorDict
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils import tensordict_utils as tu
from verl.workers.engine.megatron import transformer_impl
from verl.workers.engine.megatron.dynamic_cp import prepare_dynamic_cp_micro_batches

pytestmark = pytest.mark.cpu_test

LAYOUTS = [(1, 1), (2, 1), (1, 2), (3, 2), (2, 4)]
AGGREGATIONS = ["token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean"]


def make_engine(per_token=True, dynamic_cp=False, auxiliary=False):
    engine = object.__new__(transformer_impl.MegatronEngineWithLMHead)
    engine.tf_config = SimpleNamespace(
        calculate_per_token_loss=per_token,
        timers=None,
        num_moe_experts=2 if auxiliary else None,
        grad_scale_func=None,
    )
    engine.engine_config = SimpleNamespace(dynamic_context_parallel=dynamic_cp)
    engine.prepare_model_outputs = lambda output, data: output
    engine.get_data_parallel_group = lambda: None
    return engine


def objective(model_output, data, dp_group, aggregation):
    return agg_loss(
        model_output["loss_mat"],
        data["loss_mask"],
        aggregation,
        dp_size=data["dp_size"],
        batch_num_tokens=data["batch_num_tokens"],
        global_batch_size=12,
        loss_scale_factor=8,
    ), {"sentinel": 7}


def check_fixed_batch(monkeypatch, dp, cp, micro_batches, per_token, aggregation, auxiliary=False):
    features = torch.linspace(-1.2, 2.1, 96, dtype=torch.float64).reshape(12, 8)
    targets = torch.linspace(0.4, -0.3, 96, dtype=torch.float64).reshape(12, 8)
    mask = (torch.arange(96).reshape(12, 8) % 7 != 0).double()
    reference = torch.tensor([1.3, -0.2, 0.7], dtype=torch.float64, requires_grad=True)
    expected = agg_loss(
        (reference[0] * features + reference[1] - targets).square(),
        mask,
        aggregation,
        global_batch_size=12,
        loss_scale_factor=8,
    )
    if auxiliary:
        expected = expected + (reference[2] * features - targets).square().mean()
    expected.backward()

    weights = reference.detach().clone().requires_grad_()
    engine = make_engine(per_token, auxiliary=auxiliary)
    rank = [0]
    monkeypatch.setattr(
        transformer_impl.mpu, "get_context_parallel_group", lambda: SimpleNamespace(rank=lambda: rank[0])
    )
    losses, counts = [], []
    rows_per_micro = 12 // (dp * micro_batches)
    for dp_rank in range(dp):
        for micro in range(micro_batches):
            start = (dp_rank * micro_batches + micro) * rows_per_micro
            stop = start + rows_per_micro
            data = TensorDict(
                {
                    "input_ids": torch.ones(rows_per_micro, 8, dtype=torch.long),
                    "loss_mask": mask[start:stop],
                },
                batch_size=[rows_per_micro],
            )
            tu.assign_non_tensor(
                data,
                dp_size=dp,
                num_micro_batch=micro_batches,
                batch_num_tokens=mask.sum().item(),
                routed_num_tokens=96,
            )
            for cp_rank in range(cp):
                rank[0] = cp_rank
                values = (weights[0] * features[start:stop] + weights[1] - targets[start:stop]).square()
                local = torch.arange(values.numel()).reshape_as(values) % cp == cp_rank
                # CP gathering retains autograd only for the current rank's token shard.
                values = torch.where(local, values, values.detach())
                if auxiliary:
                    aux_sum = (weights[2] * features[start:stop][local] - targets[start:stop][local]).square().sum()
                    values = MoEAuxLossAutoScaler.apply(values, aux_sum)
                callback = partial(
                    engine.postprocess_micro_batch_func,
                    data=data,
                    forward_only=False,
                    loss_function=partial(objective, aggregation=aggregation),
                )
                stored = []
                loss, count = forward_step_calc_loss(
                    model=SimpleNamespace(),
                    output_tensor={"loss_mat": values},
                    loss_func=callback,
                    config=engine.tf_config,
                    vp_stage=None,
                    collect_non_loss_data=False,
                    num_microbatches=micro_batches,
                    forward_data_store=stored,
                    cp_group_size=cp,
                    is_last_stage=True,
                )
                assert stored[0]["metrics"] == {"sentinel": 7}, "The schedule must preserve loss metrics."
                losses.append(loss)
                counts.append(count)
    if per_token:
        total_tokens = torch.stack(counts).sum()
        assert total_tokens.item() == 96, "The global count must include each input token exactly once."
        scaling = 1.0 / total_tokens.clamp_min(1)
    else:
        scaling = 1.0 / (dp * cp)
    torch.stack(losses).sum().backward()
    # Megatron scales gradient buffers after autograd, including auxiliary gradients.
    weights.grad.mul_(scaling)
    torch.testing.assert_close(weights.grad, reference.grad)
    torch.testing.assert_close(weights.grad.norm(), reference.grad.norm())


@pytest.mark.parametrize("dp,cp", LAYOUTS)
@pytest.mark.parametrize("micro_batches", [1, 2])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_fixed_global_batch_gradient(monkeypatch, dp, cp, micro_batches, per_token, aggregation):
    check_fixed_batch(monkeypatch, dp, cp, micro_batches, per_token, aggregation)


@pytest.mark.parametrize("dp,cp", LAYOUTS)
@pytest.mark.parametrize("micro_batches", [1, 2])
def test_auxiliary_gradients_share_token_normalization(monkeypatch, dp, cp, micro_batches):
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)
    check_fixed_batch(monkeypatch, dp, cp, micro_batches, True, "token-mean", auxiliary=True)


def test_mixed_dynamic_cp_preserves_global_gradient(monkeypatch):
    lengths = [37, 13, 29, 9, 17, 7, 23]
    features = [torch.linspace(-1.0, 2.0, length, dtype=torch.float64) for length in lengths]
    masks = [torch.arange(length).ge(2).double() for length in lengths]
    total_tokens = sum(lengths)
    valid_tokens = sum(mask.sum().item() for mask in masks)
    data = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor([torch.arange(length) for length in lengths], layout=torch.jagged),
            "loss_mask": torch.nested.nested_tensor(masks, layout=torch.jagged),
            "features": torch.nested.nested_tensor(features, layout=torch.jagged),
        },
        batch_size=[len(lengths)],
    )
    tu.assign_non_tensor(data, dp_size=8, routed_num_tokens=total_tokens)
    reference = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    expected = sum(((reference * values).square() * mask).sum() for values, mask in zip(features, masks, strict=True))
    (expected / valid_tokens).backward()
    weight = reference.detach().clone().requires_grad_()
    engine = make_engine(dynamic_cp=True)
    rank = [0]
    monkeypatch.setattr(
        transformer_impl.mpu,
        "get_dynamic_data_context_parallel_groups",
        lambda group_size: SimpleNamespace(rank=lambda: rank[0] % group_size),
    )
    losses, counts, sizes = [], [], set()
    for pool_rank in range(8):
        rank[0] = pool_rank
        batches, _ = prepare_dynamic_cp_micro_batches(data, 16, 1, 8, 1, pool_rank)
        for batch in batches:
            cp_size = batch["local_cp_size"]
            sizes.add(cp_size)
            values = (weight * batch["features"].values()).square() * batch["loss_mask"].values()
            local = torch.arange(values.numel()) % cp_size == pool_rank % cp_size
            output = {"loss_sum": torch.where(local, values, values.detach()).sum()}
            tu.assign_non_tensor(batch, num_micro_batch=len(batches))
            loss, count, _ = engine.postprocess_micro_batch_func(
                output,
                batch,
                False,
                lambda model_output, data, dp_group: (model_output["loss_sum"] * data["dp_size"] / valid_tokens, {}),
            )
            losses.append(loss)
            counts.append(count)
    assert len(sizes) > 1, "The normalization check must exercise different CP group sizes."
    assert torch.stack(counts).sum().item() == total_tokens, "Dynamic groups must count every input token once."
    torch.stack(losses).sum().backward()
    weight.grad.div_(total_tokens)
    torch.testing.assert_close(weight.grad, reference.grad)


@pytest.mark.parametrize("layout", ["padded", "packed", "dense"])
def test_routed_count_includes_prompts_and_excludes_padding(layout):
    engine = make_engine()
    if layout == "padded":
        data = {"input_ids": torch.ones(2, 5), "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])}
    elif layout == "packed":
        data = {"input_ids": torch.nested.nested_tensor([torch.arange(3), torch.arange(4)], layout=torch.jagged)}
    else:
        data = {"input_ids": torch.arange(7)}
    data["loss_mask"] = torch.tensor([0, 0, 1, 0, 0, 1, 1])
    assert engine._routed_num_tokens(data).item() == 7, "Input normalization must include prompt tokens."


@pytest.mark.parametrize("dynamic_cp", [False, True])
def test_cp_replica_counts_preserve_an_odd_token_total(monkeypatch, dynamic_cp):
    engine = make_engine(dynamic_cp=dynamic_cp)
    data = TensorDict({"input_ids": torch.arange(7)}, batch_size=[])
    tu.assign_non_tensor(data, num_micro_batch=2, dp_size=3, routed_num_tokens=19, local_cp_size=4)
    rank = [0]
    group = SimpleNamespace(rank=lambda: rank[0])
    monkeypatch.setattr(transformer_impl.mpu, "get_context_parallel_group", lambda: group)

    def dynamic_group(group_size):
        assert group_size == 4, "Dynamic normalization must use the microbatch's CP group."
        return group

    monkeypatch.setattr(transformer_impl.mpu, "get_dynamic_data_context_parallel_groups", dynamic_group)
    counts = []
    for cp_rank in range(4):
        rank[0] = cp_rank
        loss, count, output = engine.postprocess_micro_batch_func(
            {}, data, False, lambda **kwargs: (torch.tensor(2.0), {})
        )
        torch.testing.assert_close(loss, torch.tensor(2.0 * 19 / 3))
        assert output["loss"] == 2.0, "Reported loss must retain its original normalization."
        counts.append(count)
    assert torch.stack(counts).sum().item() == 7, "CP replication must preserve indivisible token counts."


def test_forward_only_without_loss_preserves_output_contract():
    engine = make_engine()
    data = TensorDict({"input_ids": torch.arange(7)}, batch_size=[])
    output = {"log_probs": torch.ones(7)}
    loss, stored = engine.postprocess_micro_batch_func(output, data, True, None)
    assert loss.item() == 1.0, "Inference must retain its placeholder loss."
    assert stored["model_output"] is output, "Inference outputs must remain available to the collector."
