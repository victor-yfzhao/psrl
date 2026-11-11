"""Verify derived tensors release GPU segments before the training actor sleeps."""

import gc
import weakref
from functools import cache
from types import SimpleNamespace

import pytest
import torch
from psrl.workers.train.engine_train_worker import clear_megatron_memory_caches


def _models_and_rotary():
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    rotary = RotaryEmbedding(kv_channels=128, rotary_percent=1.0, use_cpu_initialization=True)
    model = torch.nn.Module()
    model.rotary_pos_emb = rotary
    model.rotary_pos_emb_cache = {}
    return [model], rotary, model


@pytest.mark.cpu_test
def test_rotary_cache_release_invalidates_every_derived_cache(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_global_memory_buffer", lambda: SimpleNamespace(buffer={}))
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: None)
    models, rotary, model = _models_and_rotary()
    monkeypatch.setattr(rotary, "get_emb", lambda length, offset=0: torch.ones(length, 1, 1, 128))
    embedding = rotary.forward(8)
    embedding_ref = weakref.ref(embedding)
    cos, sin = rotary.get_cached_cos_sin(8)
    cos_ref, sin_ref = weakref.ref(cos), weakref.ref(sin)
    model.rotary_pos_emb_cache[8] = (cos, sin)
    frequencies = rotary.inv_freq.clone()
    del embedding, cos, sin

    clear_megatron_memory_caches(models)
    gc.collect()

    assert rotary.forward.cache_info().currsize == 0, "The forward cache must release its tensors."
    assert embedding_ref() is None, "Cached rotary embeddings must not survive the memory handoff."
    assert cos_ref() is None and sin_ref() is None, "Cached trigonometric tensors must be released."
    assert not model.rotary_pos_emb_cache, "The model's inference cache must release its tensors."
    assert rotary.cos_cached is None and rotary.sin_cached is None, "Derived buffers must release their storage."
    assert torch.equal(rotary.inv_freq, frequencies), "Releasing derived caches must preserve rotary frequencies."
    rebuilt_cos, rebuilt_sin = rotary.get_cached_cos_sin(8)
    assert rebuilt_cos.numel() == rebuilt_sin.numel() == 8 * 128, "Rotary caches must rebuild after release."
    clear_megatron_memory_caches(models)


@pytest.mark.gpu
def test_rotary_cache_release_unpins_a_large_cuda_segment(monkeypatch):
    from megatron.core import parallel_state
    from megatron.core.utils import GlobalMemoryBuffer

    global_buffer = GlobalMemoryBuffer()
    monkeypatch.setattr(parallel_state, "get_global_memory_buffer", lambda: global_buffer)
    models, rotary, _ = _models_and_rotary()
    rotary.inv_freq = rotary.inv_freq.cuda()
    torch.cuda.empty_cache()
    large = torch.empty(192 * 1024**2, dtype=torch.uint8, device="cuda")
    del large
    embedding = rotary.forward(8192)
    embedding_ref = weakref.ref(embedding)
    del embedding
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    reserved_before = torch.cuda.memory_reserved()

    clear_megatron_memory_caches(models)
    gc.collect()
    torch.cuda.empty_cache()
    reserved_after = torch.cuda.memory_reserved()

    assert embedding_ref() is None, "The rotary cache must release its CUDA tensor."
    assert reserved_before - reserved_after >= 128 * 1024**2, (
        "Clearing rotary caches must release the large CUDA segment pinned by their small allocations."
    )


@pytest.mark.cpu_test
def test_workspace_release_clears_every_lazy_cache(monkeypatch):
    from megatron.core import parallel_state
    from transformer_engine.pytorch.cpp_extensions import gemm
    from transformer_engine.pytorch.module import base

    models, _, _ = _models_and_rotary()

    @cache
    def get_cublas_workspace(grouped_gemm):
        return torch.ones(8)

    tensors = [torch.ones(8), get_cublas_workspace(False), get_cublas_workspace(True), torch.ones(8)]
    references = [weakref.ref(tensor) for tensor in tensors]
    global_buffer = SimpleNamespace(buffer={"mpu": tensors[0]})
    dummy_wgrads = {"weight": tensors[3]}
    monkeypatch.setattr(parallel_state, "get_global_memory_buffer", lambda: global_buffer)
    monkeypatch.setattr(gemm, "get_cublas_workspace", get_cublas_workspace)
    monkeypatch.setattr(base, "_dummy_wgrads", dummy_wgrads)
    cleared_torch_workspaces = []
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: cleared_torch_workspaces.append(True))
    del tensors

    clear_megatron_memory_caches(models)

    assert all(reference() is None for reference in references), "Every workspace owner must release its tensors."
    assert get_cublas_workspace.cache_info().currsize == 0, "Every GEMM workspace must rebuild lazily."
    assert not dummy_wgrads and not global_buffer.buffer, "Collective and dummy gradient caches must empty."
    assert cleared_torch_workspaces == [True], "PyTorch cuBLAS workspaces must also release their allocations."


@pytest.mark.gpu
def test_workspace_release_unpins_a_large_cuda_segment(monkeypatch):
    from megatron.core import parallel_state
    from megatron.core.utils import GlobalMemoryBuffer
    from transformer_engine.pytorch.cpp_extensions.gemm import get_cublas_workspace

    models, _, _ = _models_and_rotary()
    global_buffer = GlobalMemoryBuffer()
    monkeypatch.setattr(parallel_state, "get_global_memory_buffer", lambda: global_buffer)
    clear_megatron_memory_caches(models)
    gc.collect()
    torch.cuda.empty_cache()
    large = torch.empty(384 * 1024**2, dtype=torch.uint8, device="cuda")
    del large
    global_buffer.get_tensor((32 * 1024**2,), torch.uint8, "mpu")
    device = torch.cuda.current_device()
    get_cublas_workspace(device, False, False)
    get_cublas_workspace(device, False, True)
    get_cublas_workspace(device, True, False)
    matrix = torch.ones((512, 512), device="cuda", dtype=torch.bfloat16)
    product = matrix @ matrix
    del matrix, product
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    reserved_before = torch.cuda.memory_reserved()

    clear_megatron_memory_caches(models)
    gc.collect()
    torch.cuda.empty_cache()
    reserved_after = torch.cuda.memory_reserved()

    assert reserved_before - reserved_after >= 256 * 1024**2, (
        "Clearing kernel workspaces must release large CUDA segments pinned by their small allocations."
    )
    rebuilt = get_cublas_workspace(device, False, False)
    assert rebuilt.numel() > 0 and rebuilt.is_cuda, "The GEMM workspace must rebuild after wake."
    del rebuilt
    clear_megatron_memory_caches(models)
