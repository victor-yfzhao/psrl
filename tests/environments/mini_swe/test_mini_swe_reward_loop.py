"""
Verify Mini-SWE grading through reward workers, including cancellation and failures.
"""

import asyncio
import subprocess
import threading
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch
from examples.mini_swe import reward as loop_module
from examples.mini_swe.config import MiniEnvironmentConfig
from examples.mini_swe.reward import compute_score
from examples.mini_swe.swebench_grader import SWEGradingError, SWEGradingPreparationTimeout
from examples.mini_swe.task_setup import SWETaskSetupError
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from psrl.utils.dataset.utils import as_object_sample, merge_dataset_config, resolve_split_specs
from psrl.workers.agent_loop.agent_data.mini_swe_agent_data import MiniSWEAgentData
from psrl.workers.reward.reward_loop import load_reward_loop
from psrl.workers.reward.reward_loop.dapo import DAPORewardLoop
from psrl.workers.reward.reward_worker import RewardLoopWorker, RewardSpec
from tenacity import wait_none
from verl.utils import tensordict_utils as tu

pytestmark = pytest.mark.cpu_test

PATCH = "diff --git a/value.py b/value.py\n--- a/value.py\n+++ b/value.py\n@@ -1 +1 @@\n-value = 0\n+value = 1\n"
SPEC_KEY = "mini_swe/compute_score/None"


@pytest.fixture(autouse=True)
def reaper(monkeypatch):
    spawn = Mock()
    monkeypatch.setattr(loop_module, "spawn_actor_reaper", spawn)
    return spawn


def make_request(data_source="swe_gym", *, validate=False, patch=PATCH, turns=4):
    agent_data = MiniSWEAgentData.__new__(MiniSWEAgentData)
    agent_data.set_grading_context(
        {
            "swe_grader": "swebench_fresh_container",
            "swe_problem": {"instance_id": "example__repo-1", "FAIL_TO_PASS": ["test_fix"]},
            "swe_problem_image": "example/swe:latest",
            "swe_grader_kind": {"swebench_verified": "verified", "swe_smith_py": "smith", "swe_gym": "gym"}.get(
                data_source
            ),
            "swe_task_id": "episode-1",
            "runtime_config": SimpleNamespace(
                sandbox_config=SimpleNamespace(
                    environment=MiniEnvironmentConfig(
                        grader_timeout=2400,
                        grader_command_timeout=75,
                        grader_memory="24g",
                        grader_shm_size="12g",
                    )
                )
            ),
        }
    )
    context = agent_data.grading_context
    return tu.get_tensordict(
        tensor_dict={
            "prompts": torch.tensor([[1, 2]]),
            "responses": torch.tensor([[3, 4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]),
            "uid": np.array(["episode-1"]),
            "num_turns": np.array([turns]),
            "data_source": np.array([data_source]),
            "reward_model": as_object_sample({"ground_truth": PATCH}),
            "extra_info": as_object_sample({"problem_statement": "Fix the value."}),
            "agent_reward_info": as_object_sample({"patch": patch, "actual_num_turns": turns, "swe_grading": context}),
            "reward_model_dicts": as_object_sample([{"reward_loop_type": "mini_swe", "reward_fn": "compute_score"}]),
            "request_keys": as_object_sample(["episode-1"]),
            "attempt_id": np.array([1]),
        },
        non_tensor_dict={"validate": validate},
    )


def make_loop(mode="binary", *, overlong=False):
    return loop_module.MiniSWERewardLoop(
        OmegaConf.create({"psrl": {"logging_path": None}}),
        SimpleNamespace(decode=Mock(return_value="submission")),
        partial(compute_score, reward_mode=mode),
        max_resp_len=4,
        overlong_buffer_cfg=OmegaConf.create({"enable": overlong, "len": 2, "penalty_factor": 1.0, "log": True}),
    )


def make_worker(loop, *, validate=False):
    cls = RewardLoopWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    worker.config = OmegaConf.create({})
    worker.worker_id = 0
    worker.is_validate = validate
    worker.inference_orchestrator = None
    worker.completed = worker.failed = 0
    worker.reward_spec_to_loop = {RewardSpec("mini_swe", "compute_score", None): loop}
    worker.reward_manager_client = SimpleNamespace(put_reward_result=AsyncMock())
    return worker


@pytest.mark.asyncio
@pytest.mark.parametrize("data_source", ["swe_gym", "swe_smith_py", "swebench_verified"])
@pytest.mark.parametrize("validate", [False, True])
async def test_reward_worker_grades_once_and_preserves_task_settings(monkeypatch, data_source, validate):
    grade = Mock(return_value={"resolved": True, "apply_ok": True, "f2p_pass": 1, "f2p_total": 1})
    monkeypatch.setattr(loop_module, "grade_fresh_container", grade)
    loop = make_loop(overlong=True)
    worker = make_worker(loop, validate=validate)
    request = make_request(data_source, validate=validate)

    await worker._run_reward_loop(request)

    sent = worker.reward_manager_client.put_reward_result.call_args.args[0]
    assert sent["error"] is None and sent["is_validate"] is validate, "The correct reward pool must receive success."
    assert sent["result"]["reward_score"] == 0.5, "The DAPO penalty must use the unpadded response length."
    info = sent["result"]["reward_extra_info"][SPEC_KEY]
    assert info["acc"] == 1.0 and info["score"] == 1.0, "Accuracy must remain the unpenalized harness result."
    assert info["grader_result"] == grade.return_value, "Grader diagnostics must survive the reward worker response."
    assert sent["result"]["reward_metrics"][SPEC_KEY]["grading_s"] >= 0, "Grading must have its own timing."
    assert sent["response_len"] == 4, "The manager must receive the original tensor width for reward placement."
    assert worker.completed == 1 and worker.failed == 0, "One episode must produce one completed reward."
    grade.assert_called_once()
    context = tu.get(request[0], "agent_reward_info")["swe_grading"]
    assert grade.call_args.args[:5] == (
        context["swe_problem"],
        PATCH,
        context["swe_grader_kind"],
        context["swe_problem_image"],
        2400,
    ), "Resolved task identity and grading limits must reach the harness."
    assert grade.call_args.args[5].startswith("episode-1__reward_"), "Each grading attempt must own its cleanup label."
    assert grade.call_args.args[6] == "24g", "The grader must retain its memory limit."
    assert grade.call_args.kwargs == {
        "run_args": [*context["environment"]["run_args"], "--label", f"psrl.actor_id={loop._container_owner}"],
        "env": context["environment"]["env"],
        "shm_size": "12g",
        "command_timeout": 75,
    }, "Container and library resource limits must survive dispatch."
    assert "grader_result" not in tu.get(request[0], "agent_reward_info"), "Scoring must not mutate shared inputs."
    assert loop.tokenizer.decode.call_args.args[0].tolist() == [3, 4, 5], "Padding must not enter reward text."


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["binary", "test_ratio", "partial_credit", "shaped"])
async def test_service_matches_existing_reward_and_length_penalty(monkeypatch, mode):
    grade = {"resolved": False, "apply_ok": True, "f2p_pass": 1, "f2p_total": 2, "p2p_pass": 3, "p2p_total": 3}
    monkeypatch.setattr(loop_module, "grade_fresh_container", Mock(return_value=grade))
    loop = make_loop(mode, overlong=True)
    request = make_request()
    expected_data = request.clone(recurse=False)
    tu.assign_non_tensor_stack(
        expected_data, "agent_reward_info", [{**tu.get(request[0], "agent_reward_info"), "grader_result": grade}]
    )
    expected = await DAPORewardLoop.run_single(loop, expected_data)
    actual = await loop.run_single(request)
    assert actual["reward_score"] == expected["reward_score"], "Moving grading must preserve the reward formula."
    for key, value in expected["reward_extra_info"].items():
        assert actual["reward_extra_info"][key] == value, "DAPO diagnostics must remain unchanged."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,turns,expected", [("binary", 4, 0.0), ("partial_credit", 1, -1.0), ("partial_credit", 4, -0.5)]
)
async def test_empty_submission_does_not_start_a_grader(monkeypatch, reaper, mode, turns, expected):
    grade = Mock(side_effect=AssertionError("An empty patch must not start Docker."))
    monkeypatch.setattr(loop_module, "grade_fresh_container", grade)
    result = await make_loop(mode).run_single(make_request(patch=None, turns=turns))
    assert result["reward_score"] == expected, "Missing submissions must retain their configured reward."
    grade.assert_not_called()
    reaper.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("data_source", ["mini_swe_agent_simple", "mini_swe_agent"])
async def test_simple_tasks_use_patch_scoring_without_grading_metadata(monkeypatch, reaper, data_source):
    grade = Mock(side_effect=AssertionError("Toy tasks must not start a benchmark grader."))
    monkeypatch.setattr(loop_module, "grade_fresh_container", grade)
    request = make_request(data_source)
    tu.assign_non_tensor_stack(request, "agent_reward_info", [{"patch": PATCH}])
    result = await make_loop().run_single(request)
    assert result["reward_score"] == 1.0, "Toy tasks must retain patch overlap rewards."
    grade.assert_not_called()
    reaper.assert_not_called()


@pytest.mark.asyncio
async def test_zero_turn_episodes_do_not_execute_grading(monkeypatch, reaper):
    grade = Mock()
    monkeypatch.setattr(loop_module, "grade_fresh_container", grade)
    result = await make_loop().run_single(make_request(turns=0))
    assert result["reward_score"] == 0.0, "Episodes without generated turns must have zero reward."
    grade.assert_not_called()
    reaper.assert_not_called()


@pytest.mark.parametrize("report", [None, {}])
@pytest.mark.parametrize("data_source", ["swe_gym", "swe_smith_py", "swebench_verified"])
def test_benchmark_scoring_rejects_a_missing_grading_report(data_source, report):
    extra_info = {"patch": PATCH, "num_turns": 4}
    if report is not None:
        extra_info["grader_result"] = report
    with pytest.raises(ValueError, match="mini_swe reward loop"):
        compute_score(data_source, "", PATCH, extra_info=extra_info)


@pytest.mark.asyncio
async def test_concurrent_grading_of_one_submission_uses_independent_cleanup_labels(monkeypatch):
    grade = Mock(return_value={"resolved": True})
    monkeypatch.setattr(loop_module, "grade_fresh_container", grade)
    request = make_request()
    loop = make_loop()
    results = await asyncio.gather(loop.run_single(request), loop.run_single(request))
    assert [result["reward_score"] for result in results] == [1.0, 1.0], "Both grading requests must complete."
    assert len({call.args[5] for call in grade.call_args_list}) == 2, "One grader must not remove another's container."
    assert tu.get(request[0], "agent_reward_info")["swe_grading"]["swe_task_id"] == "episode-1", (
        "Grading labels must not mutate the originating episode metadata."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field", ["swe_problem", "swe_problem_image", "swe_grader_kind", "swe_task_id", "environment"]
)
async def test_missing_grading_inputs_are_errors_even_without_a_patch(monkeypatch, field):
    grade = Mock()
    monkeypatch.setattr(loop_module, "grade_fresh_container", grade)
    request = make_request(patch=None)
    info = tu.get(request[0], "agent_reward_info")
    del info["swe_grading"][field]
    tu.assign_non_tensor_stack(request, "agent_reward_info", [info])
    with pytest.raises(SWEGradingError, match="grading inputs"):
        await make_loop().run_single(request)
    grade.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [SWEGradingError, SWETaskSetupError, RuntimeError])
async def test_infrastructure_failure_reaches_the_manager_without_a_zero_reward(monkeypatch, error_type):
    monkeypatch.setattr(loop_module, "grade_fresh_container", Mock(side_effect=error_type("Harness unavailable.")))
    worker = make_worker(make_loop())
    await worker._run_reward_loop(make_request())
    sent = worker.reward_manager_client.put_reward_result.call_args.args[0]
    assert sent["result"] is None, "Infrastructure failures must never become policy failures with zero reward."
    assert "Harness unavailable" in sent["error"], "The manager must retain the failure cause."
    assert worker.failed == 1 and worker.completed == 0, "Failed grading must be counted as a reward worker failure."


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize(
    "timeout",
    [
        SWEGradingPreparationTimeout("Docker preparation timed out."),
        subprocess.TimeoutExpired(["git", "apply", "--numstat"], 120),
    ],
)
async def test_preparation_timeouts_retry_with_isolated_labels_and_preserve_exhaustion(
    monkeypatch,
    recover,
    timeout,
):
    grade = Mock(side_effect=[timeout, timeout, {"resolved": True} if recover else timeout])
    monkeypatch.setattr(loop_module, "grade_fresh_container", grade)
    monkeypatch.setattr(loop_module, "wait_exponential", lambda **kwargs: wait_none())
    worker = make_worker(make_loop())

    await worker._run_reward_loop(make_request())

    assert grade.call_count == 3, "Preparation timeouts must respect the configured attempt limit."
    assert len({call.args[5] for call in grade.call_args_list}) == 3, "Retries must own independent cleanup labels."
    assert all(call.args[4] == 2400 for call in grade.call_args_list), (
        "Retries must preserve the test execution budget."
    )
    sent = worker.reward_manager_client.put_reward_result.call_args.args[0]
    if recover:
        assert sent["error"] is None and sent["result"]["reward_score"] == 1.0, (
            "A recovered grader must return its report."
        )
        assert worker.completed == 1 and worker.failed == 0, "Recovered preparation must complete one reward."
    else:
        assert sent["result"] is None and "timed out" in sent["error"], "Exhausted preparation must remain an error."
        assert worker.failed == 1 and worker.completed == 0, "Exhausted retries must count one failed reward."


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [None, SWEGradingError, SWEGradingPreparationTimeout])
async def test_cancelled_grading_retains_worker_capacity_until_cleanup(
    monkeypatch,
    failure_type,
):
    started = asyncio.Event()
    release = threading.Event()
    event_loop = asyncio.get_running_loop()
    attempts = []

    def grade(*args, **kwargs):
        attempts.append(args[5])
        event_loop.call_soon_threadsafe(started.set)
        if not release.wait(timeout=10):
            raise AssertionError("The test must release the grader thread.")
        if failure_type is not None:
            raise failure_type("Cancelled grading failed during cleanup.")
        return {"resolved": True}

    monkeypatch.setattr(loop_module, "grade_fresh_container", grade)
    worker = make_worker(make_loop())
    worker._semaphore = asyncio.Semaphore(0)
    task = asyncio.create_task(worker._run_reward_loop_with_release(make_request()))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done(), "Repeated cancellation must not detach a live Docker grader."
            assert worker._semaphore.locked(), "The worker permit must cover grader cleanup."
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    assert not worker._semaphore.locked(), "Cleanup completion must release worker capacity."
    assert len(attempts) == 1, "Cancellation must not start another grading attempt."
    worker.reward_manager_client.put_reward_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_recipe_resolves_service_loop_for_both_data_splits():
    config_dir = str(Path(__file__).resolve().parents[3] / "psrl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "reward=mini_swe",
                "data.reward_model_dicts.0.reward_loop_type=mini_swe",
                "data.reward_model_dicts.0.reward_fn=compute_score",
                "+data.train_files=[train.parquet]",
                "+data.val_files=[val.parquet]",
                "reward.reward_loop_worker.max_concurrency_per_worker=3",
            ],
        )
    preset = config.reward.reward_loops.mini_swe
    loop = load_reward_loop(
        config, SimpleNamespace(decode=Mock()), preset.reward_loop, preset.reward_fn[0], **preset.reward_kwargs
    )
    assert isinstance(loop, loop_module.MiniSWERewardLoop), "The selected preset must load the grading loop."
    assert preset.reward_loop.source == "importlib", "Recipe loops must load through the generic plugin interface."
    assert type(loop).__module__ == compute_score.__module__, "The loop and scoring function must share one module."
    assert list(config.reward.active_loops) == ["mini_swe"], "Only the service grading loop should be active."
    assert not config.reward.launch_reward_fn_async, "Grading must settle before a Mini-SWE group can be admitted."
    for split in ("train", "val"):
        data_config = merge_dataset_config(config.data, resolve_split_specs(config.data, split)[0])
        worker = make_worker(loop, validate=split == "val")
        keys, loops, coefs = worker._resolve_reward_loops(data_config.reward_model_dicts)
        assert keys == [SPEC_KEY] and loops == [loop] and coefs == [1.0], (
            "Both data splits must select service grading."
        )
    assert config.reward.reward_loop_worker.max_concurrency_per_worker == 3, (
        "Training concurrency must be configurable."
    )
    assert config.reward.validation_reward_loop_worker.max_concurrency_per_worker == 4, (
        "Validation needs its own limit."
    )


@pytest.mark.asyncio
async def test_grading_loop_owns_one_reaper_for_concurrent_requests(monkeypatch, reaper, tmp_path):
    monkeypatch.setattr(loop_module, "grade_fresh_container", Mock(return_value={"resolved": True}))
    loop = make_loop()
    loop.config.psrl.logging_path = str(tmp_path)
    reaper.assert_not_called()

    await asyncio.gather(loop.run_single(make_request()), loop.run_single(make_request()))

    reaper.assert_called_once_with(loop._container_owner, log_dir=str(tmp_path))
    assert loop._reaper_proc is reaper.return_value, "The loop must retain the sidecar that owns its containers."


@pytest.mark.asyncio
async def test_grading_loops_keep_container_ownership_independent(monkeypatch, reaper):
    monkeypatch.setenv("PSRL_ACTOR_ID", "rollout-owner")
    grade = Mock(return_value={"resolved": True})
    monkeypatch.setattr(loop_module, "grade_fresh_container", grade)
    loops = [make_loop(), make_loop()]

    await asyncio.gather(*(loop.run_single(make_request()) for loop in loops))

    owners = {loop._container_owner for loop in loops}
    assert len(owners) == 2, "Each loop must own its own grading containers."
    assert {call.args[0] for call in reaper.call_args_list} == owners, "Each reaper must watch its loop's containers."
    labels = {call.kwargs["run_args"][-1] for call in grade.call_args_list}
    assert labels == {f"psrl.actor_id={owner}" for owner in owners}, "The grader and its reaper must share a label."
    assert loop_module.os.environ["PSRL_ACTOR_ID"] == "rollout-owner", (
        "Loops must not replace process global ownership."
    )
