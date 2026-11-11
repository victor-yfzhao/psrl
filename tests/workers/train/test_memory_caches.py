"""Verify derived Megatron tensors cannot survive a storage handoff."""

import sys
from functools import lru_cache
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from psrl.workers.train.engine_train_worker import clear_megatron_memory_caches

pytestmark = pytest.mark.cpu_test


def test_sleep_releases_rotary_workspaces_dummy_gradients_and_global_buffers(monkeypatch):
    @lru_cache
    def forward(length):
        return object()

    @lru_cache
    def workspace(device):
        return object()

    forward(64)
    workspace(0)
    global_buffer = SimpleNamespace(buffer={"attention": object()})
    dummy_gradients = {"wgrad": object()}
    modules = {
        "megatron.core": {"parallel_state": SimpleNamespace(get_global_memory_buffer=lambda: global_buffer)},
        "transformer_engine.pytorch.cpp_extensions.gemm": {"get_cublas_workspace": workspace},
        "transformer_engine.pytorch.module": {"base": SimpleNamespace(_dummy_wgrads=dummy_gradients)},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    cublas_clear = Mock()
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", cublas_clear)
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(forward=forward)])

    clear_megatron_memory_caches([model])

    assert forward.cache_info().currsize == 0, "RoPE derived tensors must not retain GPU storage."
    assert workspace.cache_info().currsize == 0, "TE workspaces must not retain GPU storage."
    assert not dummy_gradients and not global_buffer.buffer, "Reusable gradient and attention buffers must release."
    cublas_clear.assert_called_once_with()


def _rotary_module(size):
    rotary = torch.nn.Module()
    rotary.inv_freq = torch.zeros(size)

    @lru_cache
    def forward(length):
        return rotary.inv_freq * length

    rotary.forward = forward
    return rotary


def _restore_worker_buffers(monkeypatch, model, buffers):
    from psrl.workers.train import engine_train_worker as module

    worker = SimpleNamespace(
        config=SimpleNamespace(actor=SimpleNamespace(strategy="megatron")),
        actor=SimpleNamespace(engine=SimpleNamespace(module=[model])),
        _get_non_persistent_buffers_from_ps=lambda: buffers,
    )
    clear_caches = Mock()
    monkeypatch.setattr(module, "clear_megatron_memory_caches", clear_caches)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    module.PSRL_EngineTrainWorker._restore_non_persistent_buffers_from_ps(worker)
    clear_caches.assert_called_once_with([model])


@pytest.mark.parametrize("vision_size", [4, 8])
@pytest.mark.parametrize("vision_first", [True, False])
@pytest.mark.parametrize("wrapped", [True, False])
def test_weight_pull_restores_multimodal_rotary_frequencies(monkeypatch, vision_size, vision_first, wrapped):
    from megatron.core.transformer.module import Float16Module

    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.rotary_pos_emb = language = _rotary_module(8)
    model.vision_model = torch.nn.Module()
    model.vision_model.rotary_pos_emb = vision = _rotary_module(vision_size)
    language.forward(2)
    vision.forward(2)
    buffers = {
        "model.visual.rotary_pos_emb.inv_freq": torch.full((vision_size,), 7.0),
        "model.language_model.rotary_emb.inv_freq": torch.arange(8, dtype=torch.float64),
        "model.language_model.rotary_emb.original_inv_freq": torch.full((8,), -5.0),
    }
    if not vision_first:
        buffers = dict(reversed(list(buffers.items())))
    if wrapped:
        wrapper = Float16Module.__new__(Float16Module)
        torch.nn.Module.__init__(wrapper)
        wrapper.module = model
        model = wrapper

    _restore_worker_buffers(monkeypatch, model, buffers)

    assert torch.equal(language.inv_freq, torch.arange(8, dtype=torch.float32)), (
        "Language RoPE must restore the language buffer independently of its order and shape."
    )
    assert torch.equal(vision.inv_freq, torch.full((vision_size,), 7.0)), (
        "Vision RoPE must restore the vision buffer even when its shape matches language RoPE."
    )
    assert language.inv_freq.dtype == torch.float32, "Restoring buffers must preserve the native module dtype."
    assert language.forward.cache_info().currsize == vision.forward.cache_info().currsize == 0, (
        "Restoring frequencies must invalidate cached tensors."
    )


@pytest.mark.parametrize(
    "source_name",
    ["rotary_emb.inv_freq", "model.rotary_emb.inv_freq", "model.language_model.rotary_emb.inv_freq"],
)
def test_weight_pull_restores_text_only_rotary_frequencies(monkeypatch, source_name):
    model = torch.nn.Module()
    model.rotary_pos_emb = rotary = _rotary_module(4)

    _restore_worker_buffers(monkeypatch, model, {source_name: torch.arange(4)})

    assert torch.equal(rotary.inv_freq, torch.arange(4)), "Text RoPE must match its HF buffer owner."


@pytest.mark.parametrize("vision_initialized", [True, False])
def test_weight_pull_restores_native_qwen_multimodal_rotary_modules(monkeypatch, vision_initialized):
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import Qwen3VLMultimodalRotaryEmbedding
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import Qwen3VLVisionRotaryEmbedding

    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.rotary_pos_emb = language = Qwen3VLMultimodalRotaryEmbedding(
        kv_channels=16, cp_group=object()
    )
    model.vision_model = torch.nn.Module()
    model.vision_model.rotary_pos_emb = vision = Qwen3VLVisionRotaryEmbedding(8)
    model.rotary_pos_emb = language
    if vision_initialized:
        vision(1)
    buffers = {
        "model.visual.rotary_pos_emb.inv_freq": torch.full((4,), 2.0),
        "model.language_model.rotary_emb.inv_freq": torch.full_like(language.inv_freq, 3.0),
    }
    _restore_worker_buffers(monkeypatch, model, buffers)

    if vision_initialized:
        assert torch.equal(vision.inv_freq, torch.full((4,), 2.0)), "Native vision RoPE must restore its frequencies."
    else:
        assert not hasattr(vision, "inv_freq"), "Restoring buffers must preserve lazy vision initialization."
    assert torch.equal(language.inv_freq, torch.full((8,), 3.0)), "Native language RoPE must restore its frequencies."


def test_weight_pull_rejects_missing_vision_buffer(monkeypatch):
    model = torch.nn.Module()
    model.vision_model = torch.nn.Module()
    model.vision_model.rotary_pos_emb = _rotary_module(4)

    with pytest.raises(RuntimeError, match="No parameter server rotary buffer.*vision_model.rotary_pos_emb.inv_freq"):
        _restore_worker_buffers(monkeypatch, model, {"model.language_model.rotary_emb.inv_freq": torch.ones(4)})


def test_weight_pull_preserves_rotary_shape_validation(monkeypatch):
    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.rotary_pos_emb = _rotary_module(4)

    with pytest.raises(RuntimeError, match="Rotary frequency shape.*language_model.rotary_pos_emb.inv_freq.*4.*8"):
        _restore_worker_buffers(monkeypatch, model, {"model.language_model.rotary_emb.inv_freq": torch.ones(8)})


@pytest.mark.parametrize("equal", [True, False])
def test_weight_pull_requires_aliased_buffer_values_to_agree(monkeypatch, equal):
    model = torch.nn.Module()
    model.rotary_pos_emb = _rotary_module(4)
    buffers = {
        "model.rotary_emb.inv_freq": torch.ones(4),
        "model.language_model.rotary_emb.inv_freq": torch.full((4,), 1.0 if equal else 2.0),
    }
    if equal:
        _restore_worker_buffers(monkeypatch, model, buffers)
        assert torch.equal(model.rotary_pos_emb.inv_freq, torch.ones(4)), (
            "Equal aliases must restore exact frequencies."
        )
    else:
        with pytest.raises(RuntimeError, match="Conflicting parameter server rotary buffers"):
            _restore_worker_buffers(monkeypatch, model, buffers)


def test_weight_pull_does_not_treat_unknown_tower_as_language(monkeypatch):
    model = torch.nn.Module()
    model.audio_model = torch.nn.Module()
    model.audio_model.rotary_pos_emb = _rotary_module(4)

    with pytest.raises(RuntimeError, match="No parameter server rotary buffer.*audio_model"):
        _restore_worker_buffers(monkeypatch, model, {"model.rotary_emb.inv_freq": torch.ones(4)})
