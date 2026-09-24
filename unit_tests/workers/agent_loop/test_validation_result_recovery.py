import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import MethodType, SimpleNamespace


def _load_manager_class():
    agent_loop_package = types.ModuleType("psrl.workers.agent_loop")
    agent_loop_package.__path__ = []
    ps_package = types.ModuleType("psrl.workers.ps")
    ps_package.__path__ = []

    prometheus_module = types.ModuleType("psrl.workers.agent_loop.prometheus_utils")
    prometheus_module.update_prometheus_config = lambda *args, **kwargs: None
    request_status_module = types.ModuleType("psrl.workers.ps.request_status_tracker")
    request_status_module.PSRL_RequestStatus = SimpleNamespace(RUNNING="RUNNING")
    staleness_module = types.ModuleType("psrl.workers.ps.staleness_controller")
    staleness_module.EntryInfo = type("EntryInfo", (), {})

    stub_modules = {
        "psrl.workers.agent_loop": agent_loop_package,
        "psrl.workers.agent_loop.prometheus_utils": prometheus_module,
        "psrl.workers.ps": ps_package,
        "psrl.workers.ps.request_status_tracker": request_status_module,
        "psrl.workers.ps.staleness_controller": staleness_module,
    }
    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)
    try:
        module_path = Path(__file__).resolve().parents[3] / "psrl" / "workers" / "agent_loop" / "manager.py"
        spec = importlib.util.spec_from_file_location("agent_loop_manager_for_recovery_test", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.PSRL_AgentLoopManager
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


PSRL_AgentLoopManager = _load_manager_class()


class _RemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def result():
            return None

        return result()


def _make_manager():
    manager = PSRL_AgentLoopManager.__new__(PSRL_AgentLoopManager)
    manager.train_result_queue = asyncio.Queue()
    manager.val_result_queue = asyncio.Queue()
    manager._validation_recovery_done = asyncio.Event()
    manager._validation_recovery_done.set()
    manager._active_val_buffer_ids = set()
    manager.stop_collect_task = False
    manager._post_process = MethodType(lambda self, result: result, manager)
    return manager


def test_put_result_routes_validation_to_a_dedicated_queue():
    async def scenario():
        manager = _make_manager()
        training_result = SimpleNamespace(name="training", meta_info={"validate": False})
        validation_result = SimpleNamespace(name="validation", meta_info={"validate": True})

        await manager.put_result(training_result)
        await manager.put_result(validation_result)

        assert manager.train_result_queue.get_nowait() is training_result
        assert manager.val_result_queue.get_nowait() is validation_result
        assert not manager._validation_recovery_done.is_set()

    asyncio.run(scenario())


def test_ready_validation_buffer_releases_training_recovery():
    async def scenario():
        manager = _make_manager()
        manager._active_val_buffer_ids.add(5)
        manager._validation_recovery_done.clear()
        manager.val_data_buffers = {5: ["ready"]}
        manager._val_buffer_waiters = {}
        manager.logged_ready_val_buffer_ids = set()
        manager.log_ready_buffer = MethodType(lambda self, buffer_id, is_validate=False: None, manager)
        maybe_delete_buffer = _RemoteMethod()
        manager.ps_manager_handle = SimpleNamespace(maybe_delete_buffer=maybe_delete_buffer)

        await manager.handle_ready_buffer(5, is_validate=True)

        assert manager._validation_recovery_done.is_set()
        assert manager._active_val_buffer_ids == set()
        assert maybe_delete_buffer.calls == [((5, True), {})]

    asyncio.run(scenario())


def test_validation_recovery_bypasses_training_backlog_and_then_releases_it():
    async def scenario():
        manager = _make_manager()
        manager._active_val_buffer_ids.add(5)
        processed = []
        validation_processed = asyncio.Event()
        training_processed = asyncio.Event()

        async def occupy_requests(self, result):
            processed.append(result.name)
            if result.meta_info["validate"]:
                validation_processed.set()
            else:
                training_processed.set()

        manager.occupy_requests = MethodType(occupy_requests, manager)

        training_result = SimpleNamespace(name="training", meta_info={"validate": False})
        validation_result = SimpleNamespace(name="validation", meta_info={"validate": True})
        await manager.put_result(training_result)
        await manager.put_result(validation_result)

        train_collector = asyncio.create_task(manager._collect_results(is_validate=False))
        val_collector = asyncio.create_task(manager._collect_results(is_validate=True))
        try:
            await asyncio.wait_for(validation_processed.wait(), timeout=1)
            await asyncio.sleep(0)
            assert processed == ["validation"]
            assert not training_processed.is_set()

            manager._finish_validation_recovery(5)
            await asyncio.wait_for(training_processed.wait(), timeout=1)
            assert processed == ["validation", "training"]
        finally:
            manager.stop_collect_task = True
            manager._validation_recovery_done.set()
            await asyncio.gather(train_collector, val_collector)

    asyncio.run(scenario())
