"""
Verify failed vLLM memory restoration releases mappings and preserves backups.
"""

import ast
import dataclasses
import gc
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

_VLLM = Path(__file__).resolve().parents[3] / "third_party/vllm/vllm"


def _load_classes(path, names, namespace):
    tree = ast.parse((_VLLM / path).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert {node.name for node in selected} == set(names), "Required allocator classes must exist."
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(_VLLM / path), "exec"), namespace)
    return namespace


@pytest.fixture
def memory_state(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    mapped = set()

    def map_memory(handle):
        assert handle[2] not in mapped, "A CUDA allocation must not be mapped twice."
        mapped.add(handle[2])

    def unmap_memory(handle):
        assert handle[2] in mapped, "A CUDA allocation must not be unmapped twice."
        mapped.remove(handle[2])

    scope = _load_classes(
        "device_allocator/cumem.py",
        ["AllocationData", "CuMemAllocator"],
        {
            "__name__": __name__,
            "dataclasses": dataclasses,
            "gc": gc,
            "os": os,
            "torch": torch,
            "Callable": Callable,
            "Iterator": Iterator,
            "contextmanager": contextmanager,
            "Any": Any,
            "HandleType": tuple,
            "logger": Mock(),
            "cumem_available": True,
            "is_pin_memory_available": lambda: False,
            "libcudart": SimpleNamespace(cudaMemcpy=Mock()),
            "create_and_map": Mock(side_effect=map_memory),
            "unmap_and_release": Mock(side_effect=unmap_memory),
        },
    )
    allocator = scope["CuMemAllocator"]()
    handles = [(0, 16, 100 + index, 200 + index) for index in range(3)]
    for index, handle in enumerate(handles):
        allocator.current_tag = "weights" if index < 2 else "kv_cache"
        allocator._python_malloc_callback(handle)
        mapped.add(handle[2])
    allocator.sleep(offload_tags=("weights",))
    return SimpleNamespace(allocator=allocator, scope=scope, mapped=mapped, handles=handles)


@pytest.mark.cpu_test
@pytest.mark.parametrize("fail_at", [2, 3])
def test_failed_wake_rolls_back_mappings_and_retains_cpu_backups(memory_state, fail_at):
    state = memory_state
    backups = {ptr: data.cpu_backup_tensor for ptr, data in state.allocator.pointer_to_data.items()}
    original_map = state.scope["create_and_map"]
    calls = 0

    def fail_mapping(handle):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise torch.OutOfMemoryError("Injected CUDA allocation failure.")
        original_map(handle)

    state.scope["create_and_map"] = fail_mapping
    with pytest.raises(torch.OutOfMemoryError, match="Injected"):
        state.allocator.wake_up()
    assert not state.mapped, "A failed wake must release all mappings restored by that call."
    assert all(not data.is_mapped for data in state.allocator.pointer_to_data.values()), (
        "Allocation residency must match the released CUDA mappings."
    )
    assert all(data.cpu_backup_tensor is backups[ptr] for ptr, data in state.allocator.pointer_to_data.items()), (
        "CPU backups must survive an incomplete restore."
    )

    state.allocator.sleep(offload_tags=())
    state.allocator.sleep(offload_tags=())
    state.scope["create_and_map"] = original_map
    state.allocator.wake_up()
    state.allocator.wake_up()
    assert state.mapped == {handle[2] for handle in state.handles}, "Every allocation must be recoverable."
    assert all(data.cpu_backup_tensor is None for data in state.allocator.pointer_to_data.values()), (
        "Successful restoration must release the CPU backups."
    )


@pytest.mark.cpu_test
def test_failed_partial_wake_preserves_previously_awake_weights(memory_state):
    state = memory_state
    state.allocator.wake_up(tags=["weights"])
    weights = {state.handles[0][2], state.handles[1][2]}
    state.scope["create_and_map"] = Mock(side_effect=torch.OutOfMemoryError("Injected KV allocation failure."))

    with pytest.raises(torch.OutOfMemoryError, match="KV"):
        state.allocator.wake_up(tags=["kv_cache"])
    assert state.mapped == weights, "A failed tag restore must preserve earlier successful tag restores."
    state.allocator.sleep(offload_tags=())
    state.allocator.sleep(offload_tags=())
    assert not state.mapped, "Cleanup must release previously restored tags without touching unmapped tags."


@pytest.mark.cpu_test
def test_failed_rollback_keeps_unreleased_mappings_visible(memory_state):
    state = memory_state
    original_map = state.scope["create_and_map"]
    original_unmap = state.scope["unmap_and_release"]

    def fail_after_first_mapping(handle):
        if handle == state.handles[0]:
            original_map(handle)
        else:
            raise RuntimeError("Injected map failure.")

    state.scope["create_and_map"] = fail_after_first_mapping
    state.scope["unmap_and_release"] = Mock(side_effect=RuntimeError("Injected release failure."))

    with pytest.raises(RuntimeError, match="mapping cleanup failed") as error:
        state.allocator.wake_up()
    assert "map failure" in str(error.value.__cause__), "Cleanup failures must retain the initial wake failure."
    assert state.allocator.pointer_to_data[state.handles[0][2]].is_mapped, (
        "An unconfirmed release must remain visible to the cleanup owner."
    )
    state.scope["create_and_map"] = original_map
    state.scope["unmap_and_release"] = original_unmap
    state.allocator.sleep(offload_tags=())
    assert not state.mapped, "Cleanup must remain retryable after a release failure."


@pytest.mark.cpu_test
@pytest.mark.parametrize("weights_suspended", [False, True])
def test_worker_sleep_preserves_buffers_when_weights_are_unmapped(monkeypatch, weights_suspended):
    tree = ast.parse((_VLLM / "v1/worker/gpu_worker.py").read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Worker")
    sleep = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "sleep")
    allocator = SimpleNamespace(sleep=Mock())
    cumem = ModuleType("vllm.device_allocator.cumem")
    cumem.CuMemAllocator = SimpleNamespace(get_instance=lambda: allocator)
    communicators = ModuleType("vllm.distributed.device_communicators")
    communicators.nccl_suspend = SimpleNamespace(suspend_nccl_comms=Mock())
    monkeypatch.setitem(sys.modules, cumem.__name__, cumem)
    monkeypatch.setitem(sys.modules, communicators.__name__, communicators)
    scope = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (10, 20))),
        "logger": Mock(),
        "format_gib": str,
    }
    exec(compile(ast.Module(body=[sleep], type_ignores=[]), str(_VLLM / "v1/worker/gpu_worker.py"), "exec"), scope)
    backup = torch.ones(4)
    buffer = Mock()
    buffer.cpu.return_value.clone.return_value = backup
    model = SimpleNamespace(named_buffers=Mock(return_value=[("buffer", buffer)]))
    previous = {"buffer": backup}
    instance = SimpleNamespace(
        _weights_suspended=weights_suspended,
        _cudagraphs_released=weights_suspended,
        _sleep_saved_buffers=previous,
        model_runner=SimpleNamespace(model=model, release_cudagraphs=Mock(return_value=False)),
    )

    scope["sleep"](instance, level=2)
    assert model.named_buffers.call_count == (0 if weights_suspended else 1), (
        "Sleep must not read model buffers after an incomplete wake."
    )
    assert instance._sleep_saved_buffers["buffer"] is backup, "The valid CPU buffer backup must remain available."
    assert instance._cudagraphs_released == weights_suspended, "Cleanup must retain a pending graph recapture."
    allocator.sleep.assert_called_once_with(offload_tags=())


@pytest.mark.cpu_test
@pytest.mark.parametrize("tags", [["weights"], ["kv_cache"], None])
def test_worker_partial_wake_restores_buffers_only_after_weight_mapping(monkeypatch, tags):
    tree = ast.parse((_VLLM / "v1/worker/gpu_worker.py").read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Worker")
    wake = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "wake_up")
    allocator = SimpleNamespace(wake_up=Mock())
    cumem = ModuleType("vllm.device_allocator.cumem")
    cumem.CuMemAllocator = SimpleNamespace(get_instance=lambda: allocator)
    communicators = ModuleType("vllm.distributed.device_communicators")
    communicators.nccl_suspend = SimpleNamespace(resume_nccl_comms=Mock())
    monkeypatch.setitem(sys.modules, cumem.__name__, cumem)
    monkeypatch.setitem(sys.modules, communicators.__name__, communicators)
    scope = {}
    exec(compile(ast.Module(body=[wake], type_ignores=[]), str(_VLLM / "v1/worker/gpu_worker.py"), "exec"), scope)
    buffer = Mock()
    model = SimpleNamespace(named_buffers=Mock(return_value=[("buffer", buffer)]))
    instance = SimpleNamespace(
        _weights_suspended=True,
        _kv_cache_suspended=True,
        _cudagraphs_released=True,
        _sleep_saved_buffers={"buffer": torch.ones(4)},
        model_runner=SimpleNamespace(model=model, post_kv_cache_wake_up=Mock(), recapture_cudagraphs=Mock()),
    )

    scope["wake_up"](instance, tags=tags)
    weights_mapped = tags is None or "weights" in tags
    assert buffer.data.copy_.call_count == int(weights_mapped), "Unmapped model buffers must not be accessed."
    if tags is not None:
        scope["wake_up"](instance, tags=["kv_cache"] if weights_mapped else ["weights"])
    instance.model_runner.recapture_cudagraphs.assert_called_once()
    assert not instance._cudagraphs_released, "A complete wake must acknowledge graph recapture."
    assert not instance._sleep_saved_buffers, "A complete wake must consume its CPU backups."


@pytest.mark.gpu
def test_cuda_failed_wake_releases_memory_and_restores_tensor_values(monkeypatch):
    from vllm.device_allocator import cumem

    allocator = cumem.CuMemAllocator.get_instance()
    tag = "psrl_wake_recovery_test"
    with allocator.use_memory_pool(tag):
        tensors = [torch.full((4 * 1024**2,), value, device="cuda") for value in (1.0, 2.0)]
    allocations = [data for data in allocator.pointer_to_data.values() if data.tag == tag]
    assert len(allocations) >= 2, "The CUDA regression must exercise more than one physical mapping."
    allocator.sleep(offload_tags=(tag,))
    original_map = cumem.create_and_map
    calls = 0

    def fail_second_mapping(handle):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise torch.OutOfMemoryError("Injected CUDA mapping failure.")
        original_map(handle)

    try:
        monkeypatch.setattr(cumem, "create_and_map", fail_second_mapping)
        with pytest.raises(torch.OutOfMemoryError, match="Injected"):
            allocator.wake_up(tags=[tag])
        assert all(not data.is_mapped for data in allocations), "Failed CUDA restore must release partial mappings."
        allocator.sleep(offload_tags=())
        monkeypatch.setattr(cumem, "create_and_map", original_map)
        allocator.wake_up(tags=[tag])
        for index, expected in enumerate((1.0, 2.0)):
            assert torch.all(tensors[index] == expected).item(), "Retry must restore the original tensor contents."
    finally:
        monkeypatch.setattr(cumem, "create_and_map", original_map)
        allocator.wake_up(tags=[tag])
        tensors.clear()
        gc.collect()
        allocator.allocator_and_pools.pop(tag)
        torch.cuda.empty_cache()
