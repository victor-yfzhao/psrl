import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
import torch
from pivotrl.utils.server.command import Command, CommandExtension, CommandType
from pivotrl.workers.gen import gen_worker
from pivotrl.workers.gen.gen_worker import PivotRL_GenWorker
from pivotrl.workers.gen.rollout_coordinator import RolloutCoordinator


def _make_coordinator():
    coordinator_cls = RolloutCoordinator.__ray_metadata__.modified_class
    coordinator = coordinator_cls.__new__(coordinator_cls)
    coordinator.rollout_wg_size = 2
    coordinator.gen_wg_size = 3
    coordinator.enable_elastic_rm = True
    coordinator._sync_instances_directly = AsyncMock()
    return coordinator


def test_elastic_validation_sync_uses_direct_path():
    coordinator = _make_coordinator()

    asyncio.run(coordinator.sync_with_ps([2]))

    coordinator._sync_instances_directly.assert_awaited_once_with(
        [2],
        wait_interrupted_partial_requests_loop_back=True,
    )


def test_sync_rejects_out_of_range_generation_instance():
    coordinator = _make_coordinator()

    with pytest.raises(ValueError, match="must be in"):
        asyncio.run(coordinator.sync_with_ps([3]))

    coordinator._sync_instances_directly.assert_not_awaited()


def test_wake_validation_instance_targets_validation_worker_with_latest_coordinator_version():
    coordinator_cls = RolloutCoordinator.__ray_metadata__.modified_class
    coordinator = coordinator_cls.__new__(coordinator_cls)
    CommandExtension.__init__(coordinator)

    rollout_execute = AsyncMock(side_effect=AssertionError("rollout worker must not handle validation wake"))

    async def execute_validation(command, *args, **kwargs):
        if command == "is_rollout_engine_sleeping":
            return True
        if command in {"nixl_wake_up", "sync_with_ps"}:
            return 0
        if command == "get_active_task_num":
            return 0
        raise AssertionError(f"Unexpected validation command: {command}")

    validation_execute = AsyncMock(side_effect=execute_validation)
    rollout_wg = SimpleNamespace(execute_rank_zero_async=rollout_execute)
    validation_wg = SimpleNamespace(execute_rank_zero_async=validation_execute)
    coordinator.rollout_wg_list = [rollout_wg]
    coordinator.gen_wg_list = [rollout_wg, validation_wg]
    coordinator.rollout_wg_size = 1
    coordinator.gen_wg_size = 2
    coordinator.ps_model_version = 7
    coordinator.rollout_router = SimpleNamespace(
        update_currently_syncing_instances=SimpleNamespace(remote=AsyncMock()),
        resume_instances=SimpleNamespace(remote=AsyncMock()),
    )
    coordinator._syncing_locked_instance_ids = set()
    coordinator._syncing_lock_owner_by_instance = {}
    coordinator.stop_command_handler = False
    coordinator._active_command_id = None

    async def wake_validation():
        handler = asyncio.create_task(coordinator._command_handler_loop())
        try:
            return await asyncio.wait_for(
                coordinator.exec_command(
                    Command(CommandType.WAKE_UP, instance_ids=[1]),
                    blocking=True,
                ),
                timeout=1,
            )
        finally:
            coordinator.stop_command_handler = True
            await asyncio.wait_for(handler, timeout=1)

    assert asyncio.run(wake_validation()) is True
    rollout_execute.assert_not_awaited()
    assert validation_execute.await_args_list == [
        call("is_rollout_engine_sleeping"),
        call("nixl_wake_up"),
        call("get_active_task_num"),
        call(
            "sync_with_ps",
            ps_version=7,
            interrupt_generation=False,
            sync_after_wake_up=True,
        ),
    ]
    coordinator.rollout_router.update_currently_syncing_instances.remote.assert_awaited_once_with([1], 7)
    coordinator.rollout_router.resume_instances.remote.assert_awaited_once_with([1])


def test_first_validation_wake_replaces_dummy_weights_at_version_zero(monkeypatch):
    dummy_weights = torch.full((4,), -1.0)
    ps_weights = torch.arange(4, dtype=torch.float32)

    async def pull_model():
        dummy_weights.copy_(ps_weights)

    @asynccontextmanager
    async def unlocked_pull(_ps_manager):
        yield

    get_model_version = AsyncMock(return_value=0)
    worker = SimpleNamespace(
        curr_rollout_instance_model_version=0,
        active_tasks=set(),
        pull_model_async=AsyncMock(side_effect=pull_model),
        get_instance_id=MagicMock(return_value=1),
        gen_interface=SimpleNamespace(
            ps_manager_handle=SimpleNamespace(
                get_rollout_instance_model_version=SimpleNamespace(remote=get_model_version),
            )
        ),
        rollout=SimpleNamespace(stat_collector=None),
        resume_generation=MagicMock(),
    )
    monkeypatch.setattr(gen_worker, "shared_pull_model_context_async", unlocked_pull)

    skipped = asyncio.run(
        PivotRL_GenWorker.sync_with_ps(
            worker,
            ps_version=0,
            sync_after_wake_up=False,
        )
    )
    assert skipped == 0
    torch.testing.assert_close(dummy_weights, torch.full((4,), -1.0))
    worker.pull_model_async.assert_not_awaited()

    asyncio.run(
        PivotRL_GenWorker.sync_with_ps(
            worker,
            ps_version=0,
            sync_after_wake_up=True,
        )
    )
    torch.testing.assert_close(dummy_weights, ps_weights)
    worker.pull_model_async.assert_awaited_once_with()
    get_model_version.assert_awaited_once_with(worker.get_instance_id())
    worker.resume_generation.assert_called_once_with()


def test_wake_pull_tracks_actual_ps_version_newer_than_target(monkeypatch):
    @asynccontextmanager
    async def unlocked_pull(_ps_manager):
        yield

    get_model_version = AsyncMock(return_value=5)
    stat_collector = SimpleNamespace(record_model_version_update=MagicMock())
    worker = SimpleNamespace(
        curr_rollout_instance_model_version=3,
        active_tasks=set(),
        pull_model_async=AsyncMock(),
        get_instance_id=MagicMock(return_value=2),
        gen_interface=SimpleNamespace(
            ps_manager_handle=SimpleNamespace(
                get_rollout_instance_model_version=SimpleNamespace(remote=get_model_version),
            )
        ),
        rollout=SimpleNamespace(stat_collector=stat_collector),
        resume_generation=MagicMock(),
    )
    monkeypatch.setattr(gen_worker, "shared_pull_model_context_async", unlocked_pull)

    asyncio.run(
        PivotRL_GenWorker.sync_with_ps(
            worker,
            ps_version=4,
            sync_after_wake_up=True,
        )
    )

    worker.pull_model_async.assert_awaited_once_with()
    assert worker.curr_rollout_instance_model_version == 5
    stat_collector.record_model_version_update.assert_called_once_with(5)
    worker.resume_generation.assert_called_once_with()


def test_rollout_fingerprints_are_persisted_by_gen_worker(monkeypatch):
    records = [
        {
            "model_version": 3,
            "raw_fingerprint": {
                "stage": "rollout_after_raw_pull",
                "model_version": 3,
                "digest": "raw",
            },
            "final_fingerprint": {
                "stage": "rollout_after_param_sync",
                "model_version": 3,
                "digest": "final",
            },
        }
    ]
    warning = MagicMock()
    monkeypatch.setattr(gen_worker.pivotrl_logger, "warning", warning)

    PivotRL_GenWorker._log_rollout_weight_fingerprints(object(), records, model_version=3)

    assert warning.call_count == 2
    logged_records = [json.loads(log_call.args[1]) for log_call in warning.call_args_list]
    assert [record["stage"] for record in logged_records] == [
        "rollout_after_raw_pull",
        "rollout_after_param_sync",
    ]


def test_rollout_fingerprint_logging_rejects_missing_engine_core_record():
    records = [
        {
            "model_version": 3,
            "raw_fingerprint": {"stage": "rollout_after_raw_pull", "model_version": 3},
            "final_fingerprint": None,
        }
    ]

    with pytest.raises(RuntimeError, match="did not return a fingerprint record"):
        PivotRL_GenWorker._log_rollout_weight_fingerprints(object(), records, model_version=3)
