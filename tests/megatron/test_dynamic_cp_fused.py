"""Compare fused and unfused Qwen3.5 forwards across dynamic CP groups."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _run_fused_check(model_path, run_baseline=True, tp_size=1):
    import torch
    import torch.distributed as distributed
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from transformers import AutoConfig
    from verl.models.mcore.model_forward import gptmodel_forward_model_engine
    from verl.models.mcore.model_forward_fused import (
        fused_forward_model_engine,
        patch_fused_forward,
        unpatch_fused_forward,
    )
    from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init_process_group(backend="nccl")
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size, dynamic_context_parallel=True)
        model_parallel_cuda_manual_seed(42)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        text = config.text_config
        for key, value in {
            "num_hidden_layers": 2,
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "linear_num_key_heads": 8,
            "linear_num_value_heads": 16,
            "vocab_size": 256,
            "max_position_embeddings": 256,
            "full_attention_interval": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "mtp_num_hidden_layers": 0,
        }.items():
            setattr(text, key, value)
        text.rope_parameters["mrope_section"] = [3, 3, 2]
        vision = config.vision_config
        vision.depth = 1
        vision.hidden_size = 128
        vision.intermediate_size = 256
        vision.num_heads = 4
        vision.out_hidden_size = 512
        bridge = AutoBridge.from_hf_config(config)
        provider = bridge.to_megatron_provider(load_weights=False)
        provider.apply_overrides_and_finalize(
            dtype=torch.bfloat16,
            overrides={
                "tensor_model_parallel_size": tp_size,
                "context_parallel_size": 1,
                "dynamic_context_parallel": True,
                "max_seqlen_per_dp_cp_rank": 64,
                "sequence_parallel": tp_size > 1,
                "variable_seq_lengths": True,
                "attention_backend": AttnBackend.flash,
                "hidden_dropout": 0.0,
                "attention_dropout": 0.0,
                "gradient_accumulation_fusion": False,
            },
        )
        model = provider.provide_distributed_model(wrap_with_ddp=False)[0]
        model.train()
        tokens = torch.nested.nested_tensor(
            [torch.arange(37, device="cuda") % 128 + 10, torch.arange(61, device="cuda") % 128 + 10],
            layout=torch.jagged,
        )

        def process_logits(logits, label):
            logits = logits.float() / 1.4
            return {
                "log_probs": vocab_parallel_log_probs_from_logits(logits.clone(), label),
                "entropy": vocab_parallel_entropy(logits.clone()),
            }

        def collect_gradients() -> dict[str, torch.Tensor]:
            _finalize_tp_gradients(model)
            gradients = {}
            group = parallel_state.get_data_parallel_group()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    gradient = parameter.grad.float().clone()
                    distributed.all_reduce(gradient, group=group)
                    gradients[name] = gradient / distributed.get_world_size(group)
            return gradients

        gradient_mismatches = []
        for cp_size in (1, 2, 4) if run_baseline else ():
            if distributed.get_rank() == 0:
                print(f"Checking dynamic CP {cp_size}.", flush=True)
            model.zero_grad(set_to_none=True)
            reference = gptmodel_forward_model_engine(
                model,
                tokens,
                {},
                logits_processor=process_logits,
                logits_processor_args={"label": tokens},
                vision_model=True,
                pad_token_id=0,
                local_cp_size=cp_size,
            )
            loss = -reference["log_probs"].values().sum() - 0.01 * reference["entropy"].values().sum()
            loss.backward()
            gradients = collect_gradients()
            expected = {key: value.values().detach().clone() for key, value in reference.items()}
            model.zero_grad(set_to_none=True)
            patch_fused_forward(model)
            actual = fused_forward_model_engine(vision_model=True)(
                model, tokens, tokens, {}, 1.4, True, 0, local_cp_size=cp_size
            )
            for key in expected:
                torch.testing.assert_close(actual[key].values(), expected[key], atol=0.03, rtol=0.01)
            loss = -actual["log_probs"].values().sum() - 0.01 * actual["entropy"].values().sum()
            loss.backward()
            actual_gradients = collect_gradients()
            gradient_errors = []
            diagnostics = []
            assert actual_gradients.keys() == gradients.keys(), "Fused forward must preserve all parameter gradients."
            for name, reference_grad in gradients.items():
                difference = (actual_gradients[name] - reference_grad).norm()
                reference_norm = reference_grad.norm()
                error = difference / reference_norm.clamp_min(1e-12)
                gradient_errors.append(error)
                diagnostics.append((error.item(), name, reference_norm.item(), difference.item()))
            if parallel_state.get_data_parallel_rank() == 0:
                print(
                    f"CP {cp_size} largest gradient differences: {sorted(diagnostics, reverse=True)[:3]}", flush=True
                )
            max_gradient_error = torch.stack(gradient_errors).max()
            distributed.all_reduce(max_gradient_error, op=distributed.ReduceOp.MAX)
            tolerance = 4 * torch.finfo(torch.bfloat16).eps
            if distributed.get_rank() == 0:
                print(
                    f"Dynamic CP {cp_size}: maximum relative gradient error={max_gradient_error.item():.4%}.",
                    flush=True,
                )
            if max_gradient_error.item() > tolerance:
                gradient_mismatches.append((cp_size, max_gradient_error.item()))
            unpatch_fused_forward(model)
            distributed.barrier()
            if distributed.get_rank() == 0:
                print(f"Dynamic CP {cp_size}: fused comparison complete.", flush=True)
        _run_native_schedule_check(model, config)
        assert not gradient_mismatches, f"Fused gradients must match within bf16 precision. {gradient_mismatches!r}."
    finally:
        parallel_state.destroy_model_parallel()
        distributed.destroy_process_group()


def _finalize_tp_gradients(model):
    from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads

    model.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    _allreduce_non_tensor_model_parallel_grads([model], model.config)


def _run_native_schedule_check(model, config):
    from functools import partial

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu
    from tensordict import TensorDict
    from verl.models.mcore.model_forward import gptmodel_forward_model_engine
    from verl.models.mcore.model_forward_fused import unpatch_fused_forward
    from verl.utils import tensordict_utils as tu
    from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
    from verl.workers.config import ActorConfig
    from verl.workers.engine.megatron.dynamic_cp import prepare_dynamic_cp_micro_batches
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
    from verl.workers.utils.losses import ppo_loss

    lengths = [256, 128, 64, 64, 17, 33, 49]
    tokens = torch.nested.nested_tensor(
        [(torch.arange(length, device="cuda") + row * 7) % 128 + 10 for row, length in enumerate(lengths)],
        layout=torch.jagged,
    )
    prompts = torch.nested.nested_tensor([row[:3] for row in tokens.unbind()], layout=torch.jagged)
    responses = torch.nested.nested_tensor([row[3:] for row in tokens.unbind()], layout=torch.jagged)
    mask = torch.nested.nested_tensor([torch.ones_like(row) for row in responses.unbind()], layout=torch.jagged)
    advantages = torch.nested.nested_tensor(
        [
            torch.full_like(row, (-1.0) ** index * (index + 1) / 7, dtype=torch.float32)
            for index, row in enumerate(responses.unbind())
        ],
        layout=torch.jagged,
    )
    data = TensorDict(
        {
            "input_ids": tokens,
            "prompts": prompts,
            "responses": responses,
            "response_mask": mask,
            "loss_mask": mask,
            "advantages": advantages,
        },
        batch_size=[len(lengths)],
    )
    tu.assign_non_tensor(
        data,
        temperature=1.4,
        calculate_entropy=True,
        global_batch_size=len(lengths),
        dp_size=1,
        batch_num_tokens=mask.sum().item(),
        use_dynamic_bsz=True,
    )
    engine = object.__new__(MegatronEngineWithLMHead)
    engine.engine_config = SimpleNamespace(
        dynamic_context_parallel=True, context_parallel_size=1, max_seqlen_per_dp_cp_rank=64, use_remove_padding=True
    )
    engine.tf_config = model.config
    engine.module = [model]
    engine.model_config = SimpleNamespace(
        hf_config=config, tokenizer=SimpleNamespace(pad_token_id=0), mtp=SimpleNamespace(enable=False)
    )
    pool_size = mpu.get_data_parallel_world_size()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    rounds = []
    for rank in range(pool_size):
        batches, _ = prepare_dynamic_cp_micro_batches(data, 64, tp_size, pool_size, 1, rank)
        rounds.append([tu.get_non_tensor_data(batch, "local_cp_size", default=None) for batch in batches])
    sizes_per_round = [set(sizes) for sizes in zip(*rounds, strict=True)]
    assert any(len(sizes) > 1 for sizes in sizes_per_round), "The model test must mix CP sizes within a round."
    if pool_size == 8:
        assert sizes_per_round[0] == {1, 2, 4}, "The eight rank pool must mix CP=1/2/4 in one round."

    def logits_processor(logits, label):
        logits.div_(torch.tensor(1.4, dtype=logits.dtype, device=logits.device))
        saved = logits.clone()
        entropy = vocab_parallel_entropy(logits)
        return {"log_probs": vocab_parallel_log_probs_from_logits(saved, label), "entropy": entropy}

    def gradients():
        _finalize_tp_gradients(model)
        result = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                value = parameter.grad.float().clone()
                dist.all_reduce(value, group=mpu.get_data_parallel_group())
                value /= mpu.get_data_parallel_world_size()
                result[name] = value
        return result

    for aggregation in ("token-mean", "seq-mean-token-mean"):
        actor = ActorConfig(
            strategy="megatron", rollout_n=1, use_dynamic_bsz=True, loss_agg_mode=aggregation, entropy_coeff=0.01
        )
        loss_fn = partial(ppo_loss, actor)
        model.zero_grad(set_to_none=True)
        reference = gptmodel_forward_model_engine(
            model,
            tokens,
            {},
            logits_processor=logits_processor,
            logits_processor_args={"label": tokens},
            vision_model=True,
            pad_token_id=0,
            local_cp_size=1,
        )
        data["old_log_probs"] = torch.nested.nested_tensor(
            [row[2:-1].detach() - 0.02 for row in reference["log_probs"].unbind()], layout=torch.jagged
        )
        expected_loss, _ = loss_fn(model_output=reference, data=data)
        expected_loss.backward()
        cp_one_gradients = gradients()
        expected = {key: value.values().detach().clone() for key, value in reference.items()}
        valid = torch.cat(
            [
                torch.arange(length, device="cuda").ge(2) & torch.arange(length, device="cuda").lt(length - 1)
                for length in lengths
            ]
        )
        initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        for fused in (False, True):
            model.zero_grad(set_to_none=True)
            tu.assign_non_tensor(data, use_fused_kernels=fused)
            if fused:
                from verl.models.mcore.model_forward_fused import patch_fused_forward

                patch_fused_forward(model)
            batches, plan = prepare_dynamic_cp_micro_batches(
                data, 64, tp_size, pool_size, 1, mpu.get_data_parallel_rank()
            )
            matched_loss = torch.zeros((), device="cuda")
            for batch, leader in zip(batches, plan.leaders, strict=True):
                local_cp = tu.get_non_tensor_data(batch, "local_cp_size", default=None)
                if fused:
                    from verl.models.mcore.model_forward_fused import fused_forward_model_engine

                    output = fused_forward_model_engine(vision_model=True)(
                        model, batch["input_ids"], batch["input_ids"], {}, 1.4, True, 0, local_cp_size=local_cp
                    )
                else:
                    output = gptmodel_forward_model_engine(
                        model,
                        batch["input_ids"],
                        {},
                        logits_processor=logits_processor,
                        logits_processor_args={"label": batch["input_ids"]},
                        vision_model=True,
                        pad_token_id=0,
                        local_cp_size=local_cp,
                    )
                tu.assign_non_tensor(batch, dp_size=pool_size)
                local_loss, _ = loss_fn(model_output=output, data=batch)
                local_loss.backward()
                if leader:
                    matched_loss += local_loss.detach() / pool_size
            dist.all_reduce(matched_loss, group=mpu.get_data_parallel_group())
            expected_gradients = gradients()
            model.zero_grad(set_to_none=True)
            actual = engine.forward_backward_batch(data.clone(), loss_fn)
            actual_gradients = gradients()
            if dist.get_rank() == 0:
                for key, value in expected.items():
                    torch.testing.assert_close(
                        actual["model_output"][key].values()[valid], value[valid], atol=0.03, rtol=0.01
                    )
            torch.testing.assert_close(
                torch.tensor(actual["loss"][0]), expected_loss.detach().cpu(), atol=0.002, rtol=0.01
            )
            torch.testing.assert_close(torch.tensor(actual["loss"][0]), matched_loss.cpu(), atol=1e-5, rtol=1e-4)
            errors = []
            for name, value in expected_gradients.items():
                error = (actual_gradients[name] - value).norm() / value.norm().clamp_min(1e-8)
                errors.append((error.item(), name))
            assert max(errors)[0] < 4 * torch.finfo(torch.bfloat16).eps, (
                f"Native scheduling must preserve PPO parameter gradients. {sorted(errors, reverse=True)[:3]!r}."
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            for name, parameter in model.named_parameters():
                if name in actual_gradients:
                    parameter.grad.copy_(actual_gradients[name])
            optimizer.step()
            for name, parameter in model.named_parameters():
                if name in expected_gradients:
                    target = initial[name] - 0.1 * expected_gradients[name].to(parameter.dtype)
                    torch.testing.assert_close(parameter, target, atol=0.001, rtol=0.01)
                parameter.data.copy_(initial[name])
            if fused:
                unpatch_fused_forward(model)
            if dist.get_rank() == 0:
                diagnostics = [
                    ((expected_gradients[name] - value).norm().item(), value.norm().item(), name)
                    for name, value in cp_one_gradients.items()
                ]
                drift = sum(error**2 for error, _, _ in diagnostics) ** 0.5
                norm = sum(value**2 for _, value, _ in diagnostics) ** 0.5
                print(
                    f"CP=1 versus grouped bf16 full gradient relative L2: {drift / max(norm, 1e-8):.3%}.", flush=True
                )
                print(
                    f"Native CP PPO {aggregation}, fused={fused}: loss, gradients, update and row order passed.",
                    flush=True,
                )
    tu.assign_non_tensor(data, use_fused_kernels=False)
    with torch.no_grad():
        forward = engine.forward_backward_batch(data.clone(), None, forward_only=True)
    if dist.get_rank() == 0:
        assert len(forward["model_output"]["log_probs"]) == len(lengths), "Inference must return every sample once."
        print("Native CP forward only collection passed.", flush=True)


@pytest.mark.parametrize("tp_size", [1, 2])
def test_dynamic_cp_fused_qwen35(gpu_model_path, tp_size):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=8",
        str(Path(__file__).resolve()),
        "--model",
        gpu_model_path,
        "--tp",
        str(tp_size),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1200)
    assert result.returncode == 0, f"Dynamic CP fused comparison failed. {result.stdout} {result.stderr}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()
    _run_fused_check(args.model, run_baseline=not args.native_only, tp_size=args.tp)
