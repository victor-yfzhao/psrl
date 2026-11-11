"""
Verify graph release renews shared pools and preserves dedicated draft pools.
"""

import ast
import gc
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3] / "third_party/vllm/vllm"


def _load_class(path, name, methods, namespace, parent=None):
    tree = ast.parse((_ROOT / path).read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    selected = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in selected} == set(methods), "Every selected production method must exist."
    cls = ast.ClassDef(
        name=name,
        bases=[ast.Name(id="Parent", ctx=ast.Load())] if parent is not None else [],
        keywords=[],
        body=selected,
        decorator_list=[],
    )
    scope = dict(namespace, Parent=parent)
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(_ROOT / path), "exec"), scope)
    return scope[name]


def _lifecycle(torch_module, pool_factory, legacy=False):
    scope = {"Any": Any, "gc": gc, "torch": torch_module, "logger": Mock()}
    platform_cls = _load_class(
        "platforms/interface.py", "Platform", ["get_global_graph_pool", "reset_global_graph_pool"], scope
    )
    platform_cls._global_graph_pool = None
    platform_cls.graph_pool_handle = staticmethod(pool_factory)
    platform = platform_cls()
    scope["current_platform"] = platform
    scope["CUDAGraphMode"] = SimpleNamespace(NONE="none")
    wrappers = []
    for path, name, entries in (
        ("compilation/cuda_graph.py", "CUDAGraphWrapper", "concrete_cudagraph_entries"),
        ("compilation/breakable_cudagraph.py", "BreakableCUDAGraphWrapper", "entries"),
    ):
        cls = _load_class(path, name, ["clear_graphs", "clear_all_graphs"], scope)
        wrapper = cls()
        setattr(wrapper, entries, {"batch": object()})
        wrapper.graph_pool = platform.get_global_graph_pool()
        cls._all_instances = [wrapper]
        scope[name] = cls
        wrappers.append(wrapper)
    manager_cls = _load_class(
        "v1/worker/gpu/cudagraph_utils.py",
        "CudaGraphManager",
        ["_get_graph_pool", "reset_graph_pool", "needs_capture", "release"],
        scope,
    )
    draft_cls = _load_class(
        "v1/worker/gpu/spec_decode/eagle/cudagraph.py",
        "EagleCudaGraphManagerBase",
        ["_get_graph_pool"],
        scope,
        parent=manager_cls,
    )

    def make_manager(cls=manager_cls):
        manager = cls()
        manager.cudagraph_mode = "piecewise"
        manager._capture_descs = {"piecewise": []}
        manager._graphs_captured = True
        manager.graphs = {"batch": object()}
        manager.reset_graph_pool()
        return manager

    path = "v1/worker/gpu_model_runner.py" if legacy else "v1/worker/gpu/model_runner.py"
    runner_cls = _load_class(path, "GPUModelRunner", ["release_cudagraphs"], scope)
    runner = runner_cls()
    runner.cudagraph_manager = make_manager()
    runner.speculator = None
    runner.compilation_config = SimpleNamespace(cudagraph_mode="piecewise")
    runner.encoder_cudagraph_manager = None
    return SimpleNamespace(
        runner=runner, platform=platform, wrappers=wrappers, make_manager=make_manager, draft_cls=draft_cls
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("legacy", [False, True], ids=["runner-v2", "runner-v1"])
def test_release_renews_every_shared_pool_reference(legacy):
    pool_factory = count().__next__
    fake_torch = SimpleNamespace(
        accelerator=SimpleNamespace(synchronize=Mock(), empty_cache=Mock()),
        cuda=SimpleNamespace(graph_pool_handle=pool_factory),
    )
    state = _lifecycle(fake_torch, pool_factory, legacy)
    old_pool = state.platform.get_global_graph_pool()
    assert state.runner.release_cudagraphs()
    new_pool = state.platform.get_global_graph_pool()
    assert new_pool != old_pool
    assert all(wrapper.graph_pool == new_pool for wrapper in state.wrappers)
    if not legacy:
        assert state.runner.cudagraph_manager.pool == new_pool
        assert not state.runner.cudagraph_manager.graphs


@pytest.mark.cpu_test
def test_release_preserves_distinct_draft_pools():
    pool_factory = count().__next__
    fake_torch = SimpleNamespace(
        accelerator=SimpleNamespace(synchronize=Mock(), empty_cache=Mock()),
        cuda=SimpleNamespace(graph_pool_handle=pool_factory),
    )
    state = _lifecycle(fake_torch, pool_factory)
    draft_managers = [state.make_manager(state.draft_cls) for _ in range(2)]
    old_pools = {manager.pool for manager in draft_managers}
    state.runner.speculator = SimpleNamespace(
        prefill_cudagraph_manager=draft_managers[0], decode_cudagraph_manager=draft_managers[1]
    )
    assert state.runner.release_cudagraphs()
    new_pools = {manager.pool for manager in draft_managers}
    assert len(new_pools) == 2
    assert not new_pools & old_pools
    assert state.platform.get_global_graph_pool() not in new_pools


@pytest.mark.cpu_test
def test_release_without_captured_graphs_keeps_pool_identity():
    pool_factory = count().__next__
    fake_torch = SimpleNamespace(
        accelerator=SimpleNamespace(synchronize=Mock(), empty_cache=Mock()),
        cuda=SimpleNamespace(graph_pool_handle=pool_factory),
    )
    state = _lifecycle(fake_torch, pool_factory)
    state.runner.cudagraph_manager.graphs.clear()
    state.runner.cudagraph_manager._graphs_captured = False
    old_pool = state.platform.get_global_graph_pool()
    assert not state.runner.release_cudagraphs()
    assert state.platform.get_global_graph_pool() == old_pool


@pytest.mark.gpu
@pytest.mark.parametrize("legacy", [False, True], ids=["runner-v2", "runner-v1"])
def test_recapture_after_release_with_retained_pool_tensors(legacy):
    state = _lifecycle(torch, torch.cuda.graph_pool_handle, legacy)
    source = torch.ones(1024, device="cuda")
    retained_outputs = []
    for _ in range(3):
        pool = state.wrappers[0].graph_pool
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            warmup = source + 1
        stream.synchronize()
        del warmup
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            output = source + 1
        graph.replay()
        torch.cuda.synchronize()
        retained_outputs.append(output)
        state.wrappers[0].concrete_cudagraph_entries["batch"] = SimpleNamespace(cudagraph=graph)
        if not legacy:
            state.runner.cudagraph_manager.graphs["batch"] = graph
            state.runner.cudagraph_manager._graphs_captured = True
        del graph
        assert state.runner.release_cudagraphs()
        assert state.wrappers[0].graph_pool != pool
    assert all(torch.equal(output, source + 1) for output in retained_outputs)
