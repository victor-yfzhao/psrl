"""
Verify mini-SWE model and environment timing excludes final grading and cleanup.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from examples.mini_swe import runner

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("overflow", [False, True])
def test_runner_preserves_turn_durations_on_success_and_context_overflow(monkeypatch, overflow):
    from minisweagent import environments, models
    from minisweagent.agents import default
    from psrl.utils.agent import overflow as overflow_module

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(runner, "_silence_litellm", lambda: None)
    monkeypatch.setattr(overflow_module, "ensure_overflow_handling", lambda model: None)

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        def query(self):
            clock.now += 2
            if overflow and clock.now > 5:
                raise overflow_module.PromptOverflowError("Context exceeded.")

        def execute_actions(self):
            clock.now += 3

        def run(self, task):
            self.query()
            self.execute_actions()
            self.query()
            self.execute_actions()
            return {"submission": "patch", "exit_status": "submitted"}

    def cleanup():
        clock.now += 50

    monkeypatch.setattr(default, "DefaultAgent", Agent)
    monkeypatch.setattr(environments, "get_environment", Mock(return_value=SimpleNamespace(cleanup=cleanup)))
    model = SimpleNamespace(abort_exceptions=[], _query=Mock())
    monkeypatch.setattr(models, "get_model", Mock(return_value=model))
    payload = {
        "runtime_config": {
            "sandbox_config": {"environment": {"image": "example/swe:latest"}, "rollout_turn_timeout": 60},
            "model": {},
            "agent": {"problem_template": "task"},
        },
        "observation": {},
        "sampling_params": {},
        "base_url": "http://gateway",
        "model": "test-model",
        "max_turns": 2,
        "task": "task",
    }

    result = runner.run_agent(payload)
    timing = result["timing"]
    assert result["exit_status"] == ("context_exceeded" if overflow else "submitted")
    assert timing["generation_turns_s"] == [2, 2]
    assert timing["generation_s"] == 4
    assert timing["env_turns_s"] == ([3] if overflow else [3, 3])
    assert timing["env_s"] == (3 if overflow else 6)
    assert "grading_s" not in timing, "Grading time belongs to reward service metrics."
