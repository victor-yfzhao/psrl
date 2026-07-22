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
