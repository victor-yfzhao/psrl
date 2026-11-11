"""
Verify SWE task metadata survives flat and nested dataset conversion.
"""

from unittest.mock import Mock

import pandas as pd
import pytest
from examples.mini_swe.prepare import prepare_swe_gym

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "dataset_key,hf_path,nested",
    [
        ("gym", "SWE-Gym/SWE-Gym", False),
        ("gym-subset", "SumanthRH/SWE-Gym-Subset", False),
        ("skyrl-293", "NovaSky-AI/SkyRL-v0-293-data", True),
    ],
)
def test_conversion_preserves_task_identity_and_grading_metadata(monkeypatch, tmp_path, dataset_key, hf_path, nested):
    problems = [
        {
            "instance_id": f"example__repo-{index}",
            "repo": "example/repo",
            "base_commit": "abc123",
            "version": "1.0",
            "problem_statement": f"Repair issue {index}.",
            "FAIL_TO_PASS": [f"tests/test_fix.py::test_{index}"],
            "PASS_TO_PASS": ["tests/test_existing.py::test_existing"],
            "patch": "gold patch",
            "test_patch": "hidden test patch",
            "eval_script": "#!/bin/bash\npytest -rA tests",
        }
        for index in range(3)
    ]
    source = [{"prompt": "outer prompt", "instance": problem} for problem in problems] if nested else problems
    load = Mock(return_value=source)
    monkeypatch.setattr("datasets.load_dataset", load)

    converted = prepare_swe_gym.convert_dataset(dataset_key)
    path = tmp_path / "train.parquet"
    converted.to_parquet(path, index=False)
    rows = pd.read_parquet(path).to_dict("records")

    if nested:
        load.assert_called_once_with(hf_path, split="train", data_files={"train": "train.parquet"})
    else:
        load.assert_called_once_with(hf_path, split="train")
    assert len(rows) == len(problems), "Conversion must retain every task without a sampling limit."
    for row, problem in zip(rows, problems, strict=True):
        extra = row["extra_info"]
        image = f"xingyaoww/sweb.eval.x86_64.{problem['instance_id'].replace('__', '_s_')}:latest"
        assert row["prompt"][0]["content"] == problem["problem_statement"], "Use the instance problem statement."
        assert extra["swe_problem_id"] == problem["instance_id"], "Preserve the official task ID."
        for key in ("base_commit", "patch", "test_patch", "eval_script"):
            assert extra["swe_problem"][key] == problem[key], f"Preserve grading field {key!r}."
        for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            assert list(extra["swe_problem"][key]) == problem[key], f"Preserve test references {key!r}."
        assert extra["swe_problem_image"] == image, "Select the image belonging to the task."
        assert extra["sandbox_overrides"]["environment"]["image"] == image, "Rollout and grading must share an image."
        assert row["data_source"] == "swe_gym", "SkyRL tasks must use SWE-Gym reward routing."
        assert extra["swe_grader"] == "swebench_fresh_container", "Preserve fresh container grading."
        assert extra["swe_restore_tests"] is False, "SWE-Gym tasks must use the SWE-Gym setup path."


def test_skyrl_train_does_not_load_incompatible_validation_schema(monkeypatch, tmp_path):
    from datasets import load_dataset

    problem = {
        "instance_id": "example__repo-1",
        "repo": "example/repo",
        "problem_statement": "Repair the issue.",
        "FAIL_TO_PASS": ["test_fix"],
        "PASS_TO_PASS": ["test_existing"],
        "eval_script": "pytest -rA",
    }
    pd.DataFrame([{"instance": problem}]).to_parquet(tmp_path / "train.parquet", index=False)
    validation = {**problem, "FAIL_TO_PASS": '["test_fix"]', "environment_setup_commit": "abc123"}
    pd.DataFrame([{"instance": validation}]).to_parquet(tmp_path / "validation.parquet", index=False)

    def load_local_dataset(hf_path, **kwargs):
        return load_dataset(str(tmp_path), cache_dir=str(tmp_path / "cache"), **kwargs)

    monkeypatch.setattr("datasets.load_dataset", load_local_dataset)
    converted = prepare_swe_gym.convert_dataset("skyrl-293")
    assert len(converted) == 1, "Only the requested split must reach conversion."
    assert converted.iloc[0]["extra_info"]["swe_problem"]["FAIL_TO_PASS"] == ["test_fix"], (
        "Validation schema inference must not change training test references."
    )
