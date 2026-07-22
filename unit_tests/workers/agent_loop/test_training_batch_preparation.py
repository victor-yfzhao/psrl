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
        spec = importlib.util.spec_from_file_location("agent_loop_manager_for_test", module_path)
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
    def __init__(self, result):
        self.result = result

    def remote(self, *args, **kwargs):
        async def _result():
            return self.result

        return _result()


class _FakePSManager:
    def __init__(self):
        self.ensure_train_buffer_exists = _RemoteMethod(None)
        self.acquire = _RemoteMethod(True)
        self.release = _RemoteMethod(None)
        self.abort_reserved_requests = _RemoteMethod((0, []))
        self.move_occupied_entries = _RemoteMethod(None)


def _make_manager(current_entries, later_entries):
    manager = PSRL_AgentLoopManager.__new__(PSRL_AgentLoopManager)
    manager.config = SimpleNamespace(
        psrl=SimpleNamespace(
            proactive_filter_strategy=SimpleNamespace(method="retry", threshold=4),
        )
    )
    manager.ps_manager_handle = _FakePSManager()
    manager.ready_entries_per_buffer = 2
    manager.train_data_buffers = {}
    manager.train_accumulated_buffers = {7: {0: list(current_entries)}}
    manager.train_accumulated_buffer_size = {7: len(current_entries)}
    if later_entries:
        manager.train_accumulated_buffers[8] = {0: list(later_entries)}
        manager.train_accumulated_buffer_size[8] = len(later_entries)

    manager.get_buffer_from_data_pool = MethodType(lambda self, entries: list(entries), manager)

    def _add_buffer(self, buffer_id, data, is_validate=False):
        self.train_data_buffers[buffer_id] = data
        return True

    manager.maybe_add_buffer = MethodType(_add_buffer, manager)

    async def _handle_ready_buffer(self, buffer_id, is_validate=False):
        return None

    manager.handle_ready_buffer = MethodType(_handle_ready_buffer, manager)
    manager.remove_buffer_from_data_pool = MethodType(lambda self, entries, is_validate=False: None, manager)
    manager.log_buffer = MethodType(lambda self, buffer_id, is_validate=False: None, manager)
    return manager


def test_prepare_training_batch_promotes_proactively_fillable_buffer_without_consuming():
    current_entry = SimpleNamespace(prompt_id=1)
    later_entry = SimpleNamespace(prompt_id=2)
    manager = _make_manager([current_entry], [later_entry])

    ready = asyncio.run(manager.prepare_training_batch_if_ready(7))

    assert ready is True
    assert manager.train_data_buffers[7] == [current_entry, later_entry]
    assert 7 not in manager.train_accumulated_buffers


def test_prepare_training_batch_keeps_incomplete_buffer_when_no_entries_can_be_moved():
    current_entry = SimpleNamespace(prompt_id=1)
    manager = _make_manager([current_entry], [])

    ready = asyncio.run(manager.prepare_training_batch_if_ready(7))

    assert ready is False
    assert manager.train_accumulated_buffer_size[7] == 1
    assert 7 not in manager.train_data_buffers


def test_wait_for_training_batch_consumes_proactively_prepared_buffer_once():
    current_entry = SimpleNamespace(prompt_id=1)
    later_entry = SimpleNamespace(prompt_id=2)
    manager = _make_manager([current_entry], [later_entry])

    batch = asyncio.run(manager.wait_for_training_batch(7))

    assert batch == [current_entry, later_entry]
    assert 7 not in manager.train_data_buffers
    assert 7 not in manager.train_accumulated_buffers


def test_wait_for_training_batch_ready_observes_buffer_without_consuming():
    async def scenario():
        manager = _make_manager([], [])
        manager.train_accumulated_buffers = {}
        manager.train_accumulated_buffer_size = {}

        async def mark_ready():
            await asyncio.sleep(0.01)
            manager.train_data_buffers[7] = ["ready"]

        marker = asyncio.create_task(mark_ready())
        ready = await manager.wait_for_training_batch_ready(7, timeout_s=0.2)
        await marker
        return manager, ready

    manager, ready = asyncio.run(scenario())
    assert ready is True
    assert manager.train_data_buffers[7] == ["ready"]


def test_wait_for_training_batch_ready_times_out_without_consuming():
    manager = _make_manager([], [])
    manager.train_accumulated_buffers = {}
    manager.train_accumulated_buffer_size = {}

    ready = asyncio.run(manager.wait_for_training_batch_ready(7, timeout_s=0.01))

    assert ready is False
    assert 7 not in manager.train_data_buffers
