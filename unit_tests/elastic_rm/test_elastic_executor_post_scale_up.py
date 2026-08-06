from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path


def _install_stubs():
    stub_names = (
        "ray",
        "psrl.trainer.ppo.utils",
        "psrl.utils.elastic_rm.diagnostics",
        "psrl.utils.elastic_rm.dummy_scaling_policy",
        "psrl.utils.elastic_rm.itl_scaling_policy",
        "psrl.utils.elastic_rm.itl_harmonic_scaling_policy",
        "psrl.utils.elastic_rm.scaling_policy",
        "psrl.utils.logger",
        "psrl.utils.server.command",
    )
    previous_modules = {name: sys.modules.get(name) for name in stub_names}
    ray_stub = types.ModuleType("ray")

    def _remote(*args, **kwargs):
        if args and isinstance(args[0], type):
            return args[0]
        return lambda cls: cls

    ray_stub.remote = _remote
    ray_stub.actor = types.SimpleNamespace(ActorHandle=object)
    sys.modules["ray"] = ray_stub

    utils_mod = types.ModuleType("psrl.trainer.ppo.utils")
    utils_mod.PSRL_Role = types.SimpleNamespace(Rollout="Rollout", RewardModel="RewardModel")
    sys.modules["psrl.trainer.ppo.utils"] = utils_mod

    diag_mod = types.ModuleType("psrl.utils.elastic_rm.diagnostics")
    diag_mod.log_elastic_rm_backlog_diag = lambda *args, **kwargs: None
    sys.modules["psrl.utils.elastic_rm.diagnostics"] = diag_mod

    dummy_mod = types.ModuleType("psrl.utils.elastic_rm.dummy_scaling_policy")
    dummy_mod.DummyScalingPolicy = object
    sys.modules["psrl.utils.elastic_rm.dummy_scaling_policy"] = dummy_mod

    itl_mod = types.ModuleType("psrl.utils.elastic_rm.itl_scaling_policy")
    itl_mod.ITLScalingPolicy = object
    sys.modules["psrl.utils.elastic_rm.itl_scaling_policy"] = itl_mod

    harmonic_mod = types.ModuleType("psrl.utils.elastic_rm.itl_harmonic_scaling_policy")
    harmonic_mod.ITLHarmonicScalingPolicy = object
    sys.modules["psrl.utils.elastic_rm.itl_harmonic_scaling_policy"] = harmonic_mod

    scaling_mod = types.ModuleType("psrl.utils.elastic_rm.scaling_policy")
    scaling_mod.InstanceSignal = type("InstanceSignal", (), {})
    scaling_mod.ScalingPolicy = type("ScalingPolicy", (), {})
    sys.modules["psrl.utils.elastic_rm.scaling_policy"] = scaling_mod

    logger_mod = types.ModuleType("psrl.utils.logger")
    logger_mod.DualOutputHandler = type("H", (), {"__init__": lambda self, *a, **k: None})
    logger_mod.FileOnlyHandler = logger_mod.DualOutputHandler
    sys.modules["psrl.utils.logger"] = logger_mod

    cmd_mod = types.ModuleType("psrl.utils.server.command")

    class _CommandType:
        ABORT = "ABORT"
        SLEEP = "SLEEP"
        WAKE_UP = "WAKE_UP"

    cmd_mod.CommandType = _CommandType
    cmd_mod.Command = type("Command", (), {})
    sys.modules["psrl.utils.server.command"] = cmd_mod
    return previous_modules


def _load_module():
    previous_modules = _install_stubs()
    module_name = "psrl.utils.elastic_rm.elastic_executor_for_test"
    module_path = Path(__file__).resolve().parents[2] / "psrl" / "utils" / "elastic_rm" / "elastic_executor.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


def _make_executor(mod):
    executor = mod.ElasticExecutor.__new__(mod.ElasticExecutor)
    executor.instances_status_flags = {"Rollout": {"model": {}}}
    executor.instances_engine_stats = {"Rollout": {"model": {}}}
    return executor


def test_post_scale_up_candidate_builder_excludes_wake_targets_and_prefers_waiting():
    mod = _load_module()
    executor = _make_executor(mod)
    role = "Rollout"
    model = "model"
    executor.instances_status_flags[role][model] = {
        0: mod.InstanceStatus.AWAKEN,
        1: mod.InstanceStatus.AWAKEN,
        2: mod.InstanceStatus.AWAKEN,
    }
    executor.instances_engine_stats[role][model] = {
        0: {
            "scheduler_stats": {
                "req_id_in_waiting": ["0-wait"],
                "req_id_to_prompt_token_num": {"0-wait": 10, "0-run": 80},
                "req_id_to_response_token_num": {"0-wait": 10, "0-run": 40},
            }
        },
        1: {
            "scheduler_stats": {
                "req_id_in_waiting": ["1-wait-a", "1-wait-b"],
                "req_id_to_prompt_token_num": {"1-wait-a": 8, "1-wait-b": 12},
                "req_id_to_response_token_num": {"1-wait-a": 2, "1-wait-b": 3},
            }
        },
        2: {
            "scheduler_stats": {
                "req_id_in_waiting": ["2-wait"],
                "req_id_to_prompt_token_num": {"2-wait": 5},
                "req_id_to_response_token_num": {"2-wait": 5},
            }
        },
    }

    candidates, total_requests, total_tokens = executor._build_post_scale_up_request_candidates(
        role_name=role,
        model_name=model,
        wake_instance_ids={2},
    )

    assert sorted(candidates) == [0, 1]
    assert total_requests == 4.0
    assert total_tokens == 165.0
    assert {c.request_id for c in candidates[0]} == {"0-wait", "0-run"}
    assert {c.request_id for c in candidates[1]} == {"1-wait-a", "1-wait-b"}
    assert any(c.request_id == "0-wait" and c.is_waiting for c in candidates[0])
    assert any(c.request_id == "1-wait-a" and c.is_waiting for c in candidates[1])
    assert all(c.request_id != "2-wait" for c in candidates[0] + candidates[1])


def test_post_scale_up_candidate_builder_returns_empty_when_only_wake_targets_have_requests():
    mod = _load_module()
    executor = _make_executor(mod)
    role = "Rollout"
    model = "model"
    executor.instances_status_flags[role][model] = {0: mod.InstanceStatus.AWAKEN}
    executor.instances_engine_stats[role][model] = {
        0: {
            "scheduler_stats": {
                "req_id_in_waiting": ["0-wait"],
                "req_id_to_prompt_token_num": {"0-wait": 10},
                "req_id_to_response_token_num": {"0-wait": 10},
            }
        }
    }

    candidates, total_requests, total_tokens = executor._build_post_scale_up_request_candidates(
        role_name=role,
        model_name=model,
        wake_instance_ids={0},
    )

    assert candidates == {}
    assert total_requests == 0.0
    assert total_tokens == 0.0


def test_policy_scaling_gate_wakes_one_waiter_and_rejects_stale_release():
    mod = _load_module()
    executor = _make_executor(mod)
    executor._policy_scaling_idle = asyncio.Event()
    executor._policy_scaling_idle.set()
    executor._policy_scaling_holder = None
    executor._policy_scaling_holder_since_s = None
    executor._policy_scaling_owner_token = None
    executor._next_policy_scaling_owner_token = 1
    executor._policy_scaling_waiters = []
    executor._decision_pending_action_counts = {}

    async def scenario():
        first_token = await executor._wait_and_acquire_policy_scaling_idle("first")
        second_task = asyncio.create_task(executor._wait_and_acquire_policy_scaling_idle("second"))
        await asyncio.sleep(0)
        assert not second_task.done()

        executor._release_policy_scaling_idle_if_clear(first_token)
        second_token = await asyncio.wait_for(second_task, timeout=0.1)
        assert second_token != first_token
        assert executor._policy_scaling_holder == "second"

        executor._release_policy_scaling_idle_if_clear(first_token)
        assert executor._policy_scaling_owner_token == second_token
        executor._release_policy_scaling_idle_if_clear(second_token)
        assert executor._policy_scaling_idle.is_set()

    asyncio.run(scenario())


def test_sync_swap_batch_reuses_outer_gate_for_all_pairs():
    mod = _load_module()
    executor = _make_executor(mod)
    executor._policy_scaling_idle = asyncio.Event()
    executor._policy_scaling_idle.set()
    executor._policy_scaling_holder = None
    executor._policy_scaling_holder_since_s = None
    executor._policy_scaling_owner_token = None
    executor._next_policy_scaling_owner_token = 1
    executor._policy_scaling_waiters = []
    executor._decision_pending_action_counts = {}
    operations = []
    executor._check_sync_swap_legality = lambda **kwargs: (True, "")

    async def scale_up(item):
        operations.append(("wake", item["instance_id"]))
        return True

    async def scale_down(item):
        operations.append(("sleep", item["instance_id"]))
        return True

    executor._scale_up_instance = scale_up
    executor._scale_down_instance = scale_down

    async def scenario():
        token = await executor._wait_and_acquire_policy_scaling_idle("outer-sync")
        result = await executor.perform_sync_swaps(
            "Rollout",
            "model",
            [(8, 0), (9, 1)],
            sync_operation_id=7,
            policy_gate_token=token,
        )
        assert result is True
        assert executor._policy_scaling_owner_token == token
        assert not executor._policy_scaling_idle.is_set()
        executor._release_policy_scaling_idle_if_clear(token)

    asyncio.run(scenario())
    assert operations == [("wake", 8), ("sleep", 0), ("wake", 9), ("sleep", 1)]


def test_scale_up_handler_sleeps_then_combines_migration_and_primary_wakes():
    mod = _load_module()
    executor = _make_executor(mod)
    executor.stop_scale_up = False
    executor.scale_up_task_queue = asyncio.Queue()
    executor.scale_up_task_queue.put_nowait(
        {
            "decision_id": 17,
            "role_name": "Rollout",
            "model_name": "rollout-model",
            "pre_wake_other_preferred": [{"instance_id": 4}],
        }
    )
    operations = []
    pre_wake = {"role_name": "RewardModel", "model_name": "rm-model", "instance_id": 4}
    pre_sleep = {"role_name": "RewardModel", "model_name": "rm-model", "instance_id": 0}
    primary_wake = {"role_name": "Rollout", "model_name": "rollout-model", "instance_id": 2}

    def resolve_pre_wake(entries):
        operations.append(("resolve_pre_wake", entries))
        return [pre_wake]

    def find_pre_sleep(task):
        operations.append(("find_pre_sleep", task["decision_id"]))
        return [pre_sleep]

    async def sleep_instances(instances):
        operations.append(("sleep", instances))

    def find_primary_wake(task):
        operations.append(("find_primary_wake", task["decision_id"]))
        return [primary_wake]

    async def wake_instances(instances):
        operations.append(("wake", instances))
        executor.stop_scale_up = True

    async def interrupt_waiting(**kwargs):
        operations.append(("interrupt", kwargs))

    executor._resolve_preferred_instances_to_scaled_up = resolve_pre_wake
    executor._find_instances_to_scaled_down_for_other_roles = find_pre_sleep
    executor._scale_down_instances = sleep_instances
    executor._find_instances_to_scaled_up = find_primary_wake
    executor._scale_up_instances = wake_instances
    executor._record_policy_migration_observation = (
        lambda sleep_elapsed_s, wake_elapsed_s: operations.append(("record", sleep_elapsed_s, wake_elapsed_s))
    )
    executor._interrupt_waiting_after_scale_up = interrupt_waiting
    executor._mark_decision_action_finished = lambda decision_id: operations.append(("finished", decision_id))

    asyncio.run(executor._scale_up_handler_loop())

    assert [operation[0] for operation in operations] == [
        "resolve_pre_wake",
        "find_pre_sleep",
        "sleep",
        "find_primary_wake",
        "wake",
        "record",
        "interrupt",
        "finished",
    ]
    assert operations[4][1] == [pre_wake, primary_wake]
    assert operations[6][1]["wake_instance_ids"] == {2}


def test_resolve_preferred_instances_to_scaled_up_returns_empty_list():
    mod = _load_module()
    executor = _make_executor(mod)

    assert executor._resolve_preferred_instances_to_scaled_up([]) == []

    executor.instances_status_flags["RewardModel"] = {"rm-model": {4: mod.InstanceStatus.AWAKEN}}
    assert executor._resolve_preferred_instances_to_scaled_up(
        [{"role_name": "RewardModel", "model_name": "rm-model", "instance_id": 4}]
    ) == []


def test_scale_up_handler_treats_none_pre_wake_as_empty_and_wakes_primary():
    mod = _load_module()
    executor = _make_executor(mod)
    executor.stop_scale_up = False
    executor.scale_up_task_queue = asyncio.Queue()
    executor.scale_up_task_queue.put_nowait(
        {"decision_id": 18, "role_name": "RewardModel", "model_name": "rm-model"}
    )
    primary_wake = {"role_name": "RewardModel", "model_name": "rm-model", "instance_id": 7}
    operations = []

    executor._resolve_preferred_instances_to_scaled_up = lambda entries: None
    executor._find_instances_to_scaled_down_for_other_roles = lambda task: []
    executor._find_instances_to_scaled_up = lambda task: [primary_wake]

    async def wake_instances(instances):
        operations.append(("wake", instances))
        executor.stop_scale_up = True

    executor._scale_up_instances = wake_instances
    executor._record_policy_migration_observation = lambda *args: None
    executor._interrupt_waiting_after_scale_up = lambda **kwargs: asyncio.sleep(0)
    executor._mark_decision_action_finished = lambda decision_id: operations.append(("finished", decision_id))

    asyncio.run(executor._scale_up_handler_loop())

    assert operations == [("wake", [primary_wake]), ("finished", 18)]


def test_request_level_scale_up_uses_planned_migrations_and_skips_legacy_rebalance():
    mod = _load_module()
    executor = _make_executor(mod)
    executor.scaling_policy = types.SimpleNamespace(
        enable_request_level_candidate_evaluation=True
    )
    executor.stop_scale_up = False
    executor.scale_up_task_queue = asyncio.Queue()
    planned = [
        {
            "request_id": "r0",
            "source_instance_id": 0,
            "destination_instance_id": 2,
        }
    ]
    executor.scale_up_task_queue.put_nowait(
        {
            "decision_id": 19,
            "role_name": "Rollout",
            "model_name": "model",
            "planned_request_migrations": planned,
        }
    )
    primary_wake = {"role_name": "Rollout", "model_name": "model", "instance_id": 2}
    executor.instances_status_flags["Rollout"]["model"] = {
        2: mod.InstanceStatus.ASLEEP
    }
    operations = []
    executor._resolve_preferred_instances_to_scaled_up = lambda entries: []
    executor._find_instances_to_scaled_down_for_other_roles = lambda task: []
    executor._find_instances_to_scaled_up = lambda task: [primary_wake]

    async def wake_instances(instances):
        operations.append(("wake", instances))
        executor.instances_status_flags["Rollout"]["model"][2] = mod.InstanceStatus.AWAKEN
        executor.stop_scale_up = True

    async def execute_planned(**kwargs):
        operations.append(("planned", kwargs))
        return 1

    async def legacy(**kwargs):
        operations.append(("legacy", kwargs))

    executor._scale_up_instances = wake_instances
    executor._record_policy_migration_observation = lambda *args: None
    executor._execute_planned_request_migrations = execute_planned
    executor._interrupt_waiting_after_scale_up = legacy
    executor._mark_decision_action_finished = lambda decision_id: operations.append(
        ("finished", decision_id)
    )

    asyncio.run(executor._scale_up_handler_loop())

    assert [operation[0] for operation in operations] == ["wake", "planned", "finished"]
    assert operations[1][1]["request_migrations"] == planned


def test_request_level_scale_up_does_not_migrate_after_wake_failure():
    mod = _load_module()
    executor = _make_executor(mod)
    executor.scaling_policy = types.SimpleNamespace(
        enable_request_level_candidate_evaluation=True
    )
    executor.stop_scale_up = False
    executor.scale_up_task_queue = asyncio.Queue()
    executor.scale_up_task_queue.put_nowait(
        {
            "decision_id": 20,
            "role_name": "Rollout",
            "model_name": "model",
            "planned_request_migrations": [{"request_id": "r0"}],
        }
    )
    primary_wake = {"role_name": "Rollout", "model_name": "model", "instance_id": 2}
    executor.instances_status_flags["Rollout"]["model"] = {
        2: mod.InstanceStatus.RECOVERING
    }
    operations = []
    executor._resolve_preferred_instances_to_scaled_up = lambda entries: []
    executor._find_instances_to_scaled_down_for_other_roles = lambda task: []
    executor._find_instances_to_scaled_up = lambda task: [primary_wake]

    async def wake_instances(instances):
        operations.append(("wake", instances))
        executor.stop_scale_up = True

    executor._scale_up_instances = wake_instances
    executor._record_policy_migration_observation = lambda *args: None
    executor._execute_planned_request_migrations = lambda **kwargs: operations.append(
        ("planned", kwargs)
    )
    executor._interrupt_waiting_after_scale_up = lambda **kwargs: operations.append(
        ("legacy", kwargs)
    )
    executor._mark_decision_action_finished = lambda decision_id: operations.append(
        ("finished", decision_id)
    )

    asyncio.run(executor._scale_up_handler_loop())

    assert [operation[0] for operation in operations] == ["wake", "finished"]


def test_scale_up_handler_continues_after_action_failure():
    mod = _load_module()
    executor = _make_executor(mod)
    executor.stop_scale_up = False
    executor.scale_up_task_queue = asyncio.Queue()
    executor.scale_up_task_queue.put_nowait(
        {"decision_id": 21, "role_name": "RewardModel", "model_name": "rm-model"}
    )
    executor.scale_up_task_queue.put_nowait(
        {"decision_id": 22, "role_name": "RewardModel", "model_name": "rm-model"}
    )
    primary_wake = {"role_name": "RewardModel", "model_name": "rm-model", "instance_id": 7}
    operations = []

    executor._resolve_preferred_instances_to_scaled_up = lambda entries: []
    executor._find_instances_to_scaled_down_for_other_roles = lambda task: []

    def find_instances(task):
        if task["decision_id"] == 21:
            raise RuntimeError("injected scale-up failure")
        return [primary_wake]

    async def wake_instances(instances):
        operations.append(("wake", instances))
        executor.stop_scale_up = True

    executor._find_instances_to_scaled_up = find_instances
    executor._scale_up_instances = wake_instances
    executor._record_policy_migration_observation = lambda *args: None
    executor._interrupt_waiting_after_scale_up = lambda **kwargs: asyncio.sleep(0)
    executor._mark_decision_action_finished = lambda decision_id: operations.append(("finished", decision_id))

    asyncio.run(executor._scale_up_handler_loop())

    assert operations == [("finished", 21), ("wake", [primary_wake]), ("finished", 22)]


def test_failed_scale_up_action_releases_real_policy_gate():
    mod = _load_module()
    executor = _make_executor(mod)
    executor.stop_scale_up = False
    executor.scale_up_task_queue = asyncio.Queue()
    executor.scale_up_task_queue.put_nowait(
        {"decision_id": 23, "role_name": "RewardModel", "model_name": "rm-model"}
    )
    executor._policy_scaling_idle = asyncio.Event()
    executor._policy_scaling_holder = "policy_decision decision_id=23"
    executor._policy_scaling_holder_since_s = None
    executor._policy_scaling_owner_token = 7
    executor._next_policy_scaling_owner_token = 8
    executor._policy_scaling_waiters = []
    executor._decision_pending_action_counts = {23: 1}
    executor._decision_gate_tokens = {23: 7}
    executor._resolve_preferred_instances_to_scaled_up = lambda entries: []
    executor._find_instances_to_scaled_down_for_other_roles = lambda task: []

    def fail_action(task):
        executor.stop_scale_up = True
        raise RuntimeError("injected failure before wake")

    executor._find_instances_to_scaled_up = fail_action

    asyncio.run(executor._scale_up_handler_loop())

    assert executor._decision_pending_action_counts == {}
    assert executor._decision_gate_tokens == {}
    assert executor._policy_scaling_owner_token is None
    assert executor._policy_scaling_idle.is_set()


def test_scale_down_handler_continues_after_action_failure():
    mod = _load_module()
    executor = _make_executor(mod)
    executor.stop_scale_down = False
    executor.scale_down_task_queue = asyncio.Queue()
    executor.scale_down_task_queue.put_nowait(
        {"decision_id": 31, "role_name": "Rollout", "model_name": "model"}
    )
    executor.scale_down_task_queue.put_nowait(
        {"decision_id": 32, "role_name": "Rollout", "model_name": "model"}
    )
    sleep_target = {"role_name": "Rollout", "model_name": "model", "instance_id": 3}
    operations = []

    def find_instances(task):
        if task["decision_id"] == 31:
            raise RuntimeError("injected scale-down failure")
        return [sleep_target]

    async def sleep_instances(instances):
        operations.append(("sleep", instances))
        executor.stop_scale_down = True

    executor._find_instances_to_scaled_down_in_role = find_instances
    executor._scale_down_instances = sleep_instances
    executor._mark_decision_action_finished = lambda decision_id: operations.append(("finished", decision_id))

    asyncio.run(executor._scale_down_handler_loop())

    assert operations == [("finished", 31), ("sleep", [sleep_target]), ("finished", 32)]
