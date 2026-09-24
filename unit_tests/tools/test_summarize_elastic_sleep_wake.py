import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "summarize_elastic_sleep_wake.py"
    spec = importlib.util.spec_from_file_location("summarize_elastic_sleep_wake", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _line(second, text):
    return f"2026-08-01 00:00:{second:02d},000 - x - {text}\n"


def test_collect_run_excludes_initialization_and_incomplete_next_batch(tmp_path):
    spec = MODULE.RunSpec("1", "policy", 1, "rm", 1, None, "run")
    run = tmp_path / "run"
    run.mkdir()

    main_lines = [
        _line(0, "[ELASTIC_OVERHEAD] operation=sleep role=Trainer world_size=8 total_s=99.0"),
        _line(1, "[Begin Event] WAIT - Wait for training batch 0"),
        _line(2, "[ELASTIC_OVERHEAD] operation=wakeup role=Trainer world_size=8 total_s=4.0"),
        _line(3, "[ELASTIC_OVERHEAD] operation=sleep role=Trainer world_size=8 total_s=2.0"),
    ]
    for offset in range(21):
        main_lines.append(_line(4 + offset, "[End Event] TRAIN - Update actor - Time taken: 1.0 seconds"))
    main_lines.extend(
        [
            _line(25, "[ELASTIC_OVERHEAD] operation=wakeup role=Trainer world_size=8 total_s=6.0"),
            _line(26, "[ELASTIC_OVERHEAD] operation=sleep role=Trainer world_size=8 total_s=4.0"),
            _line(27, "[Begin Event] WAIT - Wait for training batch 21"),
            _line(28, "[ELASTIC_OVERHEAD] operation=wakeup role=Trainer world_size=8 total_s=88.0"),
        ]
    )
    (run / "MainRayTrainer.log").write_text("".join(main_lines), encoding="utf-8")

    executor_lines = [
        _line(
            2,
            "elastic_rm scale_up_handler decision_id=1 begin task={'role_name': "
            "<PSRL_Role.RewardModel: 6>, 'num_instances': 2, 'training_step': 1, "
            "'pre_wake_other_preferred': []}",
        ),
        _line(
            3,
            "[ELASTIC_OVERHEAD] operation=scaling_action decision_id=1 action_type=scale_up "
            "success=True sleep_instances=2 wakeup_instances=2",
        ),
        _line(
            3,
            "[ELASTIC_OVERHEAD] operation=scale_up decision_id=1 sleep_s=1.0 wakeup_s=3.0",
        ),
        _line(
            4,
            "elastic_rm scale_up_handler decision_id=2 begin task={'role_name': "
            "<PSRL_Role.Rollout: 2>, 'num_instances': 1, 'training_step': 1, "
            "'pre_wake_other_preferred': []}",
        ),
        _line(
            5,
            "[ELASTIC_OVERHEAD] operation=scale_up decision_id=2 sleep_s=2.0 wakeup_s=5.0",
        ),
        _line(4, "elastic_rm scale_up_handler decision_id=2 pre_sleep_other count=3 detail=[]"),
        _line(
            4,
            "elastic_rm scale_up_handler decision_id=2 combined_wake pre_wake_targets=[] "
            "wake_targets=[{'instance_id': 1}]",
        ),
        _line(
            6,
            "elastic_rm scale_down_handler decision_id=3 begin task={'role_name': "
            "<PSRL_Role.Rollout: 2>, 'num_instances': 1, 'training_step': 1, "
            "'pre_wake_other_preferred': []}",
        ),
        _line(
            7,
            "[ELASTIC_OVERHEAD] operation=scale_down decision_id=3 sleep_s=1.5 wakeup_s=0.0",
        ),
        _line(
            8,
            "elastic_rm scale_down_handler decision_id=4 begin task={'role_name': "
            "<PSRL_Role.RewardModel: 6>, 'num_instances': 1, 'training_step': 1, "
            "'pre_wake_other_preferred': []}",
        ),
        _line(
            9,
            "[ELASTIC_OVERHEAD] operation=scale_down decision_id=4 sleep_s=2.5 wakeup_s=0.0",
        ),
        _line(
            10,
            "elastic_rm scale_up_handler decision_id=6 begin task={'role_name': "
            "<PSRL_Role.RewardModel: 6>, 'num_instances': 1, 'training_step': 1, "
            "'pre_wake_other_preferred': []}",
        ),
        _line(
            11,
            "[ELASTIC_OVERHEAD] operation=scaling_action decision_id=6 action_type=scale_up "
            "success=False sleep_instances=1 wakeup_instances=1",
        ),
        _line(
            11,
            "[ELASTIC_OVERHEAD] operation=scale_up decision_id=6 sleep_s=9.0 wakeup_s=9.0",
        ),
        _line(
            28,
            "elastic_rm scale_up_handler decision_id=5 begin task={'role_name': "
            "<PSRL_Role.Rollout: 2>, 'num_instances': 1, 'training_step': 22, "
            "'pre_wake_other_preferred': []}",
        ),
        _line(
            29,
            "[ELASTIC_OVERHEAD] operation=scale_up decision_id=5 sleep_s=77.0 wakeup_s=77.0",
        ),
    ]
    (run / "ElasticExecutor.log").write_text("".join(executor_lines), encoding="utf-8")

    selected, events = MODULE.collect_run(spec, tmp_path, min_steps=20)

    assert selected.completed_steps == 21
    assert selected.terminal_batch == 21
    actor_wakes = [
        event.seconds
        for event in events
        if event.component == "policy_actor" and event.operation == "wakeup"
    ]
    assert actor_wakes == [4.0, 6.0]
    assert 99.0 not in [event.seconds for event in events]
    assert 77.0 not in [event.seconds for event in events]
    assert 9.0 not in [event.seconds for event in events]
    legacy_sleep = next(
        event
        for event in events
        if event.component == "genrm" and event.operation == "sleep" and event.seconds == 2.0
    )
    assert legacy_sleep.instance_count == 3
    inference_events = {
        (event.component, event.operation, event.seconds)
        for event in events
        if event.component != "policy_actor"
    }
    assert inference_events == {
        ("policy_rollout", "sleep", 1.0),
        ("genrm", "wakeup", 3.0),
        ("genrm", "sleep", 2.0),
        ("policy_rollout", "wakeup", 5.0),
        ("policy_rollout", "sleep", 1.5),
        ("genrm", "sleep", 2.5),
    }


def test_summary_uses_population_standard_deviation():
    spec = MODULE.RunSpec("1", "policy", 1, "rm", 1, None, "run")
    selected = MODULE.SelectedRun("1", "policy", 1, "rm", 1, None, 21, "a", "b", 0, 21, "/run")
    events = []
    for component in ("policy_rollout", "policy_actor", "genrm"):
        for operation in ("wakeup", "sleep"):
            for seconds in (1.0, 3.0):
                events.append(
                    MODULE.LatencyEvent(
                        "1", "policy", "rm", 21, "t", component, operation,
                        seconds, 1, "source", "", "",
                    )
                )

    summaries = MODULE.summarize_events([spec], [selected], events, ddof=0)

    assert all(item.mean_s == 2.0 for item in summaries)
    assert all(item.std_s == 1.0 for item in summaries)


def test_collect_run_requires_strictly_more_than_min_steps(tmp_path):
    spec = MODULE.RunSpec("1", "policy", 1, "rm", 1, None, "run")
    run = tmp_path / "run"
    run.mkdir()
    lines = [_line(0, "[Begin Event] WAIT - Wait for training batch 0")]
    lines.extend(_line(1, "[End Event] TRAIN - Update actor") for _ in range(20))
    (run / "MainRayTrainer.log").write_text("".join(lines), encoding="utf-8")
    (run / "ElasticExecutor.log").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="expected > 20"):
        MODULE.collect_run(spec, tmp_path, min_steps=20)
