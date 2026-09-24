from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from pivotrl.trainer.ppo import ray_trainer


def _remote_handle():
    handle = MagicMock()
    handle.remote.return_value = object()
    return handle


def _make_validation_switch_trainer(*, elastic: bool):
    trainer = ray_trainer.PivotRL_RayPPOTrainer.__new__(ray_trainer.PivotRL_RayPPOTrainer)
    trainer.config = SimpleNamespace(
        pivotrl=SimpleNamespace(colocate_validate_and_train=True),
        train_actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(strategy="fsdp2"),
        ),
    )
    trainer.is_rollout_mode_in_actor = True
    trainer.elastic_trainer_pool_mode = elastic
    trainer.n_rollout_instances = 2
    trainer.n_validate_instances = 1
    trainer.rollout_router = SimpleNamespace(
        pause_instances=_remote_handle(),
        resume_instances=_remote_handle(),
    )
    trainer.rollout_coordinator = SimpleNamespace(
        pause_instances=_remote_handle(),
        resume_instances=_remote_handle(),
        sync_with_ps=_remote_handle(),
        exec_command=_remote_handle(),
    )
    validate_wg = MagicMock()
    validate_wg.execute_rank_zero_async.return_value = object()
    trainer.validate_wg_list = [validate_wg]
    trainer.actor_wg = MagicMock()
    trainer.actor_wg.world_size = 2
    trainer.actor_wg.execute_all_async.return_value = [object()]
    trainer.ps_manager_handle = SimpleNamespace(
        nixl_wait_for_update_infos=_remote_handle(),
    )
    trainer._broadcast_updated_client_infos_from_ps_manager = MagicMock()
    trainer._leave_elastic_trainer_pool_training_window = MagicMock()
    return trainer


def test_enter_elastic_validation_window_reserves_without_waking_trainer():
    trainer = ray_trainer.PivotRL_RayPPOTrainer.__new__(ray_trainer.PivotRL_RayPPOTrainer)
    trainer.elastic_trainer_pool_mode = True
    trainer._elastic_trainer_pool_training_active = False
    entries = [{"instance_id": 0}]
    trainer._elastic_trainer_pool_instance_entries = MagicMock(return_value=entries)
    trainer.elastic_executor = SimpleNamespace(enter_training_pool=_remote_handle())
    trainer._wake_trainer_for_elastic_trainer_pool = MagicMock()

    with patch.object(ray_trainer.ray, "get", return_value=None) as ray_get:
        trainer._enter_elastic_trainer_pool_validation_window()

    trainer.elastic_executor.enter_training_pool.remote.assert_called_once_with(entries)
    ray_get.assert_called_once()
    trainer._wake_trainer_for_elastic_trainer_pool.assert_not_called()
    assert trainer._elastic_trainer_pool_training_active is True


def test_validation_workers_sleep_before_train_pool_rollout_initialization():
    trainer = ray_trainer.PivotRL_RayPPOTrainer.__new__(ray_trainer.PivotRL_RayPPOTrainer)
    trainer.n_rollout_instances = 2
    trainer.n_validate_instances = 2
    trainer.is_rollout_mode_in_actor = True
    trainer.rollout_coordinator = SimpleNamespace(sleep=_remote_handle())
    trainer.rollout_router = SimpleNamespace(pause_instances=_remote_handle())

    with patch.object(ray_trainer.ray, "get", return_value=None) as ray_get:
        trainer._sleep_validation_workers_after_initialization()

    trainer.rollout_coordinator.sleep.remote.assert_called_once_with("validate")
    trainer.rollout_router.pause_instances.remote.assert_called_once_with([2, 3])
    ray_get.assert_called_once()
    assert trainer.is_rollout_mode_in_actor is False


def test_switch_to_validation_uses_elastic_trainer_sleep_helper():
    trainer = _make_validation_switch_trainer(elastic=True)
    trainer.is_rollout_mode_in_actor = False
    validate_key = MagicMock()
    validate_key.role = "validate"
    validate_key.instance_id = 0
    trainer.worker_to_ps_idx = {validate_key: "node-0"}
    trainer._sleep_trainer_for_elastic_trainer_pool = MagicMock()
    events = []
    trainer.validate_wg_list[0].execute_rank_zero_async.side_effect = (
        lambda method, *args: events.append(method) or object()
    )
    trainer._broadcast_updated_client_infos_from_ps_manager.side_effect = (
        lambda *args: events.append("broadcast_metadata")
    )
    trainer.rollout_coordinator.sync_with_ps.remote.side_effect = (
        lambda *args: events.append("pull_weights") or object()
    )
    trainer.rollout_router.resume_instances.remote.side_effect = (
        lambda *args: events.append("resume_router") or object()
    )
    trainer.rollout_coordinator.resume_instances.remote.side_effect = (
        lambda *args: events.append("resume_coordinator") or object()
    )

    with patch.object(ray_trainer.ray, "get", return_value=True):
        trainer.switch_to_rollout_mode()

    trainer._sleep_trainer_for_elastic_trainer_pool.assert_called_once_with()
    assert trainer.validate_wg_list[0].execute_rank_zero_async.call_args_list == [
        call("nixl_wake_up"),
        call("nixl_send_local_info_to", ray_trainer.NIXL_META_SERVER_NAME),
    ]
    trainer.rollout_coordinator.exec_command.remote.assert_not_called()
    trainer.rollout_coordinator.sync_with_ps.remote.assert_called_once_with([2])
    trainer.rollout_router.resume_instances.remote.assert_called_once_with([2])
    trainer.rollout_coordinator.resume_instances.remote.assert_called_once_with([2])
    assert events == [
        "nixl_wake_up",
        "nixl_send_local_info_to",
        "broadcast_metadata",
        "pull_weights",
        "resume_router",
        "resume_coordinator",
    ]
    assert trainer.is_rollout_mode_in_actor is True


def test_elastic_validation_finishes_by_returning_pool_without_waking_trainer():
    trainer = _make_validation_switch_trainer(elastic=True)

    with patch.object(ray_trainer.ray, "get", return_value=True):
        trainer.switch_to_trainer_mode(restore_trainer=False)

    trainer.rollout_coordinator.pause_instances.remote.assert_called_once_with([2])
    sleep_command = trainer.rollout_coordinator.exec_command.remote.call_args.args[0]
    assert sleep_command.type is ray_trainer.CommandType.SLEEP
    assert sleep_command.instance_ids == [2]
    trainer.validate_wg_list[0].execute_rank_zero_async.assert_not_called()
    trainer.actor_wg.execute_all_async.assert_not_called()
    trainer._leave_elastic_trainer_pool_training_window.assert_called_once_with()
    assert trainer.is_rollout_mode_in_actor is False


def test_non_elastic_validation_still_restores_trainer():
    trainer = _make_validation_switch_trainer(elastic=False)

    def fake_ray_get(value):
        return [0] if isinstance(value, list) else True

    with patch.object(ray_trainer.ray, "get", side_effect=fake_ray_get):
        trainer.switch_to_trainer_mode()

    sleep_command = trainer.rollout_coordinator.exec_command.remote.call_args.args[0]
    assert sleep_command.type is ray_trainer.CommandType.SLEEP
    assert sleep_command.instance_ids == [2]
    trainer.actor_wg.execute_all_async.assert_has_calls(
        [
            call("nixl_wake_up"),
            call("nixl_send_local_info_to", ray_trainer.NIXL_META_SERVER_NAME),
            call("pull_model"),
            call("clear_fsdp2_grads"),
        ]
    )
    trainer._leave_elastic_trainer_pool_training_window.assert_not_called()
    assert trainer.is_rollout_mode_in_actor is False


def test_validation_failure_still_returns_train_pool_to_elastic():
    trainer = ray_trainer.PivotRL_RayPPOTrainer.__new__(ray_trainer.PivotRL_RayPPOTrainer)
    trainer.elastic_trainer_pool_mode = True
    trainer.config = SimpleNamespace(
        pivotrl=SimpleNamespace(colocate_validate_and_train=True),
    )
    trainer.is_rollout_mode_in_actor = True
    trainer._enter_elastic_trainer_pool_validation_window = MagicMock()
    trainer.switch_to_rollout_mode = MagicMock()
    trainer._run_validation = MagicMock(side_effect=RuntimeError("validation failed"))
    trainer.switch_to_trainer_mode = MagicMock()

    with pytest.raises(RuntimeError, match="validation failed"):
        trainer._validate()

    trainer._enter_elastic_trainer_pool_validation_window.assert_called_once_with()
    trainer.switch_to_rollout_mode.assert_called_once_with()
    trainer.switch_to_trainer_mode.assert_called_once_with(restore_trainer=False)


def test_validation_switch_failure_after_trainer_sleep_still_runs_cleanup():
    trainer = ray_trainer.PivotRL_RayPPOTrainer.__new__(ray_trainer.PivotRL_RayPPOTrainer)
    trainer.elastic_trainer_pool_mode = True
    trainer.is_rollout_mode_in_actor = False
    trainer.config = SimpleNamespace(
        pivotrl=SimpleNamespace(colocate_validate_and_train=True),
    )
    trainer._enter_elastic_trainer_pool_validation_window = MagicMock()

    def fail_after_trainer_sleep():
        trainer.is_rollout_mode_in_actor = True
        raise RuntimeError("validation wake failed")

    trainer.switch_to_rollout_mode = MagicMock(side_effect=fail_after_trainer_sleep)
    trainer._run_validation = MagicMock()
    trainer.switch_to_trainer_mode = MagicMock()

    with pytest.raises(RuntimeError, match="validation wake failed"):
        trainer._validate()

    trainer._run_validation.assert_not_called()
    trainer.switch_to_trainer_mode.assert_called_once_with(restore_trainer=False)


def test_elastic_trainer_wake_clears_fsdp2_grads_after_model_pull():
    trainer = ray_trainer.PivotRL_RayPPOTrainer.__new__(ray_trainer.PivotRL_RayPPOTrainer)
    trainer.elastic_trainer_pool_mode = True
    trainer.trainer_pool_only_mode = False
    trainer._elastic_trainer_pool_trainer_sleeping = True
    trainer._trainer_before_sleep_weight_fingerprints = None
    trainer.config = SimpleNamespace(
        pivotrl=SimpleNamespace(ps_mode="nixl_cpu"),
        train_actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(strategy="fsdp2"),
        ),
    )
    trainer.actor_wg = MagicMock()
    trainer.actor_wg.world_size = 2
    trainer.actor_wg.execute_all_async.return_value = []
    trainer.ps_manager_handle = SimpleNamespace(
        nixl_wait_for_update_infos=_remote_handle(),
    )
    trainer._broadcast_updated_client_infos_from_ps_manager = MagicMock()
    trainer._verify_trainer_sleep_wake_weights = MagicMock()
    trainer._clear_fsdp2_grads_after_trainer_wake = MagicMock()

    with (
        patch.object(ray_trainer, "weight_fingerprint_flow_enabled", return_value=False),
        patch.object(ray_trainer.ray, "get", return_value=[]),
    ):
        trainer._wake_trainer_for_elastic_trainer_pool()

    trainer._clear_fsdp2_grads_after_trainer_wake.assert_called_once_with()
    trainer._verify_trainer_sleep_wake_weights.assert_called_once_with(None, [])
    assert trainer._elastic_trainer_pool_trainer_sleeping is False
