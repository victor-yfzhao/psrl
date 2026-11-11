"""
Verify output ownership, failed group recovery, and publication acknowledgements.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import psrl.workers.agent_loop.worker as worker_module
import pytest
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.gen.utils import TokenOutput
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus
from verl.utils import tensordict_utils as tu

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def commit_worker(monkeypatch):
    cls = worker_module.PSRL_AgentLoopWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    worker.processor = None
    worker.overlong_filtering = False
    worker.storage = {"train": {}, "val": {}}
    worker.events = []

    async def write(*, keys, partition_id, fields, tags):
        worker.events.append("write")
        worker.storage[partition_id].update(zip(keys, tags))

    async def list_keys(*, partition_id):
        return {partition_id: dict(worker.storage[partition_id])}

    async def clear(*, keys, partition_id):
        worker.events.append("clear")
        for key in keys:
            del worker.storage[partition_id][key]

    async def complete(*args, **kwargs):
        worker.events.append("complete")
        return True

    async def publish(result):
        worker.events.append("publish")

    monkeypatch.setattr(worker_module.tq, "async_kv_batch_put", AsyncMock(side_effect=write))
    monkeypatch.setattr(worker_module.tq, "async_kv_list", AsyncMock(side_effect=list_keys))
    monkeypatch.setattr(worker_module.tq, "async_kv_clear", AsyncMock(side_effect=clear))
    worker.agent_loop_manager = SimpleNamespace(
        put_result=SimpleNamespace(remote=AsyncMock(side_effect=publish)),
        notify_group_failed=SimpleNamespace(remote=AsyncMock()),
        fail_pipeline=SimpleNamespace(remote=AsyncMock()),
    )
    worker.ps_manager_handle = SimpleNamespace(
        update_request_status=SimpleNamespace(remote=AsyncMock(side_effect=complete)),
        abort_requests=SimpleNamespace(remote=AsyncMock()),
    )
    return worker


def _batch(validate=False):
    return tu.get_tensordict({"uid": [7], "parent_id": [3], "validate": [validate], "version_tag": [0]})


def _output():
    return TokenOutput(prompt_ids=[1], response_ids=[2, 3], response_mask=[1, 0])


@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [False, True])
async def test_completion_follows_payload_commit_and_precedes_publication(commit_worker, validate):
    worker = commit_worker
    await worker.postprocess_output(_output(), _batch(validate))

    assert worker.events == ["write", "complete", "publish"], "Completion must require a committed payload."
    assert "7" in worker.storage["val" if validate else "train"], "Published fields must remain manager owned."
    worker.ps_manager_handle.update_request_status.remote.assert_awaited_once_with(
        7, PSRL_RequestStatus.COMPLETED, is_validate=validate
    )
    worker.agent_loop_manager.notify_group_failed.remote.assert_not_awaited()
    worker.agent_loop_manager.fail_pipeline.remote.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("stage", ["build", "write", "complete"])
async def test_output_failures_reclaim_partial_fields_and_refill_the_group(
    commit_worker, monkeypatch, validate, stage
):
    worker = commit_worker
    error = RuntimeError(f"Failure during {stage}.")
    if stage == "build":
        monkeypatch.setattr(worker, "_build_output_fields", Mock(side_effect=error))
    elif stage == "write":

        async def partial_write(*, keys, partition_id, **kwargs):
            worker.storage[partition_id][keys[0]] = {"status": "success"}
            raise error

        monkeypatch.setattr(worker_module.tq, "async_kv_batch_put", AsyncMock(side_effect=partial_write))
    else:
        worker.ps_manager_handle.update_request_status.remote.side_effect = error

    with pytest.raises(RuntimeError, match=f"Failure during {stage}") as raised:
        await worker.postprocess_output([_output(), _output()], _batch(validate))

    assert raised.value is error, "Recovery must preserve the original failure."
    assert not worker.storage["val" if validate else "train"], "Failed outputs must leave no partial fields."
    worker.agent_loop_manager.notify_group_failed.remote.assert_awaited_once_with(
        parent_id=3, failed_uid=7, is_validate=validate, terminate_reason=TerminateReason.ROLLOUT_ERROR
    )
    worker.ps_manager_handle.abort_requests.remote.assert_awaited_once_with([7])
    worker.agent_loop_manager.put_result.remote.assert_not_awaited()
    if stage != "complete":
        worker.ps_manager_handle.update_request_status.remote.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_write_reclaims_fields_and_releases_the_group(commit_worker, monkeypatch):
    worker = commit_worker

    async def cancel_write(*, keys, partition_id, **kwargs):
        worker.storage[partition_id][keys[0]] = {"status": "success"}
        raise asyncio.CancelledError()

    monkeypatch.setattr(worker_module.tq, "async_kv_batch_put", AsyncMock(side_effect=cancel_write))
    with pytest.raises(asyncio.CancelledError):
        await worker.postprocess_output(_output(), _batch())

    assert not worker.storage["train"], "Cancelled commits must release their fields."
    worker.agent_loop_manager.notify_group_failed.remote.assert_awaited_once()
    worker.ps_manager_handle.abort_requests.remote.assert_awaited_once_with([7])


@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [False, True])
async def test_abort_during_commit_cleans_all_trajectory_keys_without_publication(commit_worker, validate):
    worker = commit_worker
    worker.ps_manager_handle.update_request_status.remote.side_effect = None
    worker.ps_manager_handle.update_request_status.remote.return_value = False

    await worker.postprocess_output([_output(), _output()], _batch(validate))

    assert not worker.storage["val" if validate else "train"], "Rejected outputs must release every trajectory."
    worker.agent_loop_manager.put_result.remote.assert_not_awaited()
    worker.agent_loop_manager.notify_group_failed.remote.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["clear", "notify", "abort"])
async def test_recovery_failure_releases_request_and_latches_a_fatal_diagnosis(commit_worker, monkeypatch, stage):
    worker = commit_worker
    monkeypatch.setattr(worker, "_build_output_fields", Mock(side_effect=ValueError("Invalid output shape.")))
    recovery_error = RuntimeError(f"Failed to {stage}.")
    if stage == "clear":
        monkeypatch.setattr(worker_module, "async_clear_payload", AsyncMock(side_effect=recovery_error))
    elif stage == "notify":
        worker.agent_loop_manager.notify_group_failed.remote.side_effect = recovery_error
    else:
        worker.ps_manager_handle.abort_requests.remote.side_effect = recovery_error

    with pytest.raises(ValueError, match="Invalid output shape"):
        await worker.postprocess_output(_output(), _batch())

    worker.ps_manager_handle.abort_requests.remote.assert_awaited_once_with([7])
    worker.agent_loop_manager.notify_group_failed.remote.assert_awaited_once()
    worker.agent_loop_manager.fail_pipeline.remote.assert_awaited_once()
    diagnosis = worker.agent_loop_manager.fail_pipeline.remote.call_args.args[0]
    assert "Invalid output shape" in diagnosis, "Fatal recovery must preserve the original cause."
    assert str(recovery_error) in diagnosis, "Fatal recovery must identify the failed cleanup operation."


@pytest.mark.asyncio
async def test_publication_acknowledgement_failure_stops_consumers_without_clearing_owned_fields(commit_worker):
    worker = commit_worker
    worker.agent_loop_manager.put_result.remote.side_effect = RuntimeError("Publication acknowledgement lost.")

    with pytest.raises(RuntimeError, match="acknowledgement lost"):
        await worker.postprocess_output(_output(), _batch())

    assert "7" in worker.storage["train"], "An uncertain acknowledgement must preserve potential manager ownership."
    worker.agent_loop_manager.fail_pipeline.remote.assert_awaited_once()
    worker.agent_loop_manager.notify_group_failed.remote.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_output_recovers_the_missing_group_slot(commit_worker):
    worker = commit_worker

    with pytest.raises(ValueError, match="n_trajectory must be positive"):
        await worker.postprocess_output([], _batch())

    worker.agent_loop_manager.notify_group_failed.remote.assert_awaited_once()
    worker.ps_manager_handle.abort_requests.remote.assert_awaited_once_with([7])
