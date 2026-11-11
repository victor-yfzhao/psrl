"""
Verify asynchronous rollout and validation dumps serialize their submitted step.
"""

import json

import numpy as np
import pytest
from psrl.trainer.ppo.ray_trainer import PSRL_RayPPOTrainer

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("directory", ["rollouts", "validation"])
def test_async_generation_dump_preserves_step_and_numeric_metadata(tmp_path, directory):
    trainer = PSRL_RayPPOTrainer.__new__(PSRL_RayPPOTrainer)
    trainer.global_steps = 7
    trainer._init_dump_executor()
    dump_path = tmp_path / directory
    try:
        trainer._dump_generations(
            inputs=["Repair this repository."],
            outputs=["A repair patch."],
            gts=[{"instance_id": "example__repo-1"}],
            scores=[np.float32(0.5)],
            reward_extra_infos_dict={
                "uid": [np.int64(42)],
                "reward_extra_info": [{"resolved": np.bool_(True), "test_counts": np.array([1, 2])}],
            },
            dump_path=str(dump_path),
        )
        trainer.global_steps = 8
        trainer._shutdown_dump_executor()
    finally:
        trainer._dump_executor.shutdown(wait=True)

    entries = [json.loads(line) for line in (dump_path / "7.jsonl").read_text().splitlines()]
    assert entries == [
        {
            "input": "Repair this repository.",
            "output": "A repair patch.",
            "gts": {"instance_id": "example__repo-1"},
            "score": 0.5,
            "step": 7,
            "uid": 42,
            "reward_extra_info": {"resolved": True, "test_counts": [1, 2]},
        }
    ], "The background writer must preserve the submitted samples, numeric metadata, and step."
    assert not (dump_path / "8.jsonl").exists(), "Later trainer steps must not change the submitted output filename."


def test_async_generation_dump_surfaces_background_write_failure(tmp_path):
    trainer = PSRL_RayPPOTrainer.__new__(PSRL_RayPPOTrainer)
    trainer.global_steps = 1
    trainer._init_dump_executor()
    try:
        with pytest.raises(TypeError, match="Object of type object is not JSON serializable"):
            trainer._dump_generations(
                inputs=["Prompt."],
                outputs=["Output."],
                gts=[None],
                scores=[0.0],
                reward_extra_infos_dict={"invalid": [object()]},
                dump_path=str(tmp_path),
            )
            trainer._shutdown_dump_executor()
    finally:
        trainer._dump_executor.shutdown(wait=True)
