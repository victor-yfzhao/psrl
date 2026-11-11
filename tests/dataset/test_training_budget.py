"""
Verify generation chunk sizes cannot shorten the trainer's requested update budget.
"""

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from psrl.utils.dataset import data_processor as data_processor_module

pytestmark = pytest.mark.cpu_test

_RAW_DATA_PROCESSOR = data_processor_module.DataProcessor.__ray_metadata__.modified_class


class EpochDataLoader:
    """
    Yield fixed prompt batches through repeated dataset epochs.
    """

    def __init__(self, dataset_size, batch_size):
        self.batch_size = batch_size
        self.batches_per_epoch = dataset_size // batch_size
        self.epochs_started = 0

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        self.epochs_started += 1
        for index in range(self.batches_per_epoch):
            yield {
                "input_ids": torch.full((self.batch_size, 1), index),
                "raw_prompt": np.array(["Task."] * self.batch_size, dtype=object),
            }


def _processor(generation_batch_size, *, dataset_size=640, total_epochs=2, total_steps=None, buffer_size=64):
    processor = object.__new__(_RAW_DATA_PROCESSOR)
    processor.config = OmegaConf.create(
        {
            "trainer": {"total_epochs": total_epochs, "total_training_steps": total_steps},
            "data": {"train_batch_size": 64, "gen_batch_size": generation_batch_size},
            "psrl": {
                "staleness_buffer_entries": buffer_size,
                "rollout_coordination": {"redundant_rollout": {"enable": False}},
            },
        }
    )
    processor.train_dataloaders = [EpochDataLoader(dataset_size, generation_batch_size)]
    processor.train_dataloader_sizes = [len(loader) for loader in processor.train_dataloaders]
    for method in (
        "build_train_and_val_dataset",
        "build_train_sampler",
        "build_train_dataloader",
        "build_val_dataloader",
    ):
        setattr(processor, method, MagicMock())
    processor.generation_batches = 0
    processor.rollout_n = 8
    processor._train_sample_idx = 0
    processor.MAX_TRAIN_ID = 10**9
    processor.dataloader_lock = threading.Lock()
    processor.train_dataloader_iters = None
    processor.stop_data_process = False
    processor.reward_manager_handle = object()
    processor.ps_manager_handle = MagicMock()
    processor.agent_loop_manager_handle = MagicMock()
    return processor


@pytest.mark.parametrize("generation_batch_size", [1, 64])
def test_epoch_budget_counts_training_buffers_instead_of_generation_batches(generation_batch_size):
    processor = _processor(generation_batch_size)
    processor._create_dataloader()
    assert processor.total_training_steps == 20, "Epoch-based trainer updates must be independent of chunk size."
    assert processor.total_generation_batches * generation_batch_size == 1280, (
        "The producer must supply every prompt required by the training buffers."
    )


def test_smoke_explicit_step_budget_cycles_small_dataset_until_all_buffers_can_fill():
    processor = _processor(1, dataset_size=32, total_epochs=100, total_steps=200)
    processor._create_dataloader()
    assert processor.total_training_steps == 200, "Explicit trainer steps must override the epoch-derived limit."
    assert processor.total_generation_batches == 12800, "Two hundred 64-prompt updates require 12800 source prompts."


def test_training_budget_uses_actual_buffer_capacity():
    processor = _processor(1, total_steps=3, buffer_size=128)
    processor._create_dataloader()
    assert processor.total_generation_batches == 384, "Each trainer iteration must be budgeted for its PS buffer."


def test_redundant_rollout_budget_preserves_oversampling_capacity():
    processor = _processor(128, total_steps=3)
    processor.config.psrl.rollout_coordination.redundant_rollout = {
        "enable": True,
        "alg_global_batch_size": 64,
        "redundant_global_batch_size": 128,
    }
    processor._create_dataloader()
    assert processor.total_generation_batches == 3, "Redundant rollout must keep one full oversampling batch per step."


def test_epoch_budget_rejects_an_incomplete_training_buffer():
    processor = _processor(1, dataset_size=32, total_epochs=1)
    with pytest.raises(ValueError, match="at least one complete training buffer"):
        processor._create_dataloader()


@pytest.mark.parametrize("generation_batch_size", [1, 64, 96])
def test_stream_produces_full_training_budget_across_epoch_boundaries(monkeypatch, generation_batch_size):
    processor = _processor(
        generation_batch_size,
        dataset_size=max(32, generation_batch_size * 2),
        total_epochs=1,
        total_steps=4,
    )
    processor._create_dataloader()
    monkeypatch.setattr(data_processor_module.ray, "get", lambda value: None)
    processor._process_data()

    calls = processor.agent_loop_manager_handle.put_data.remote.call_args_list
    assert calls[-1].args == (None,), "The stream must send END after dispatching its complete budget."
    generated = [call.args[0] for call in calls[:-1]]
    primary_prompts = sum(len(batch) // processor.rollout_n for batch in generated)
    assert 256 <= primary_prompts < 256 + generation_batch_size, (
        "Four training buffers must receive 256 prompts even when generation chunks are smaller or larger."
    )
    assert processor.generation_batches == processor.total_generation_batches, (
        "The producer must terminate against generation batches rather than trainer steps."
    )
    assert processor.train_dataloaders[0].epochs_started > 1, "Explicit training steps must cross source epoch limits."
    processor.agent_loop_manager_handle.fail_pipeline.remote.assert_not_called()


@pytest.mark.parametrize("empty", [False, True])
def test_producer_failure_releases_waiting_trainer_consumers(monkeypatch, empty):
    processor = _processor(1, total_steps=4)
    processor._create_dataloader()
    error = StopIteration if empty else RuntimeError("Source failed.")
    processor.get_single_controller_batch = MagicMock(side_effect=error)
    monkeypatch.setattr(data_processor_module.ray, "get", lambda value: None)
    processor._process_data()

    processor.agent_loop_manager_handle.fail_pipeline.remote.assert_called_once()
    processor.agent_loop_manager_handle.put_data.remote.assert_called_once_with(None)
    assert processor.generation_batches == 0, "A failed producer must not count undispatched batches."
    if empty:
        assert processor.get_single_controller_batch.call_count == 2, (
            "An empty source must stop after one epoch restart rather than spin forever."
        )
