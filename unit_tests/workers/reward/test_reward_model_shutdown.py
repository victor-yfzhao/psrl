import asyncio
from types import SimpleNamespace

import pytest
from psrl.workers.reward.reward_model.coordinator import RewardModelCoordinator


class _FakeRewardWorkerGroup:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []

    def execute_rank_zero_async(self, method_name: str):
        async def execute():
            self.calls.append(method_name)
            if self.fail:
                raise RuntimeError("engine shutdown failed")

        return execute()


def test_shutdown_reward_engines_cancels_tasks_and_closes_all_replicas() -> None:
    async def scenario() -> None:
        coordinator_cls = RewardModelCoordinator.__ray_metadata__.modified_class
        coordinator = object.__new__(coordinator_cls)
        coordinator.rm_config = SimpleNamespace(num_replicas=2)
        coordinator.stop_command_handler = False
        coordinator.stop_process_status_queue = [False, False]
        coordinator.stop_broadcast_status_to_router = False
        coordinator.command_handler_task = asyncio.create_task(asyncio.Event().wait())
        coordinator.process_status_queue_tasks = [
            asyncio.create_task(asyncio.Event().wait()),
            asyncio.create_task(asyncio.Event().wait()),
        ]
        coordinator.broadcast_status_to_router_task = asyncio.create_task(asyncio.Event().wait())
        worker_groups = [_FakeRewardWorkerGroup(), _FakeRewardWorkerGroup()]
        coordinator.reward_model_wg_list = worker_groups

        await coordinator.shutdown_reward_engines()

        assert coordinator.stop_command_handler
        assert coordinator.stop_process_status_queue == [True, True]
        assert coordinator.stop_broadcast_status_to_router
        assert coordinator.command_handler_task.cancelled()
        assert all(task.cancelled() for task in coordinator.process_status_queue_tasks)
        assert coordinator.broadcast_status_to_router_task.cancelled()
        assert [group.calls for group in worker_groups] == [
            ["shutdown_rollout_engine"],
            ["shutdown_rollout_engine"],
        ]

    asyncio.run(scenario())


def test_shutdown_reward_engines_reports_failures_after_closing_all_replicas() -> None:
    async def scenario() -> None:
        coordinator_cls = RewardModelCoordinator.__ray_metadata__.modified_class
        coordinator = object.__new__(coordinator_cls)
        coordinator.rm_config = SimpleNamespace(num_replicas=2)
        coordinator.stop_command_handler = True
        coordinator.stop_process_status_queue = [True, True]
        coordinator.stop_broadcast_status_to_router = True
        coordinator.command_handler_task = None
        coordinator.process_status_queue_tasks = []
        coordinator.broadcast_status_to_router_task = None
        worker_groups = [_FakeRewardWorkerGroup(fail=True), _FakeRewardWorkerGroup()]
        coordinator.reward_model_wg_list = worker_groups

        with pytest.raises(RuntimeError, match="Failed to shut down 1 reward model engines"):
            await coordinator.shutdown_reward_engines()

        assert [group.calls for group in worker_groups] == [
            ["shutdown_rollout_engine"],
            ["shutdown_rollout_engine"],
        ]

    asyncio.run(scenario())
