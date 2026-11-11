"""
Verify a new data processor resumes the saved sampler position.
"""

import pytest
import torch
from psrl.utils.dataset.data_processor import DataProcessor
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

pytestmark = pytest.mark.cpu_test


def _loader():
    """
    Build a deterministic loader whose next task identifies its sampler position.
    """
    dataset = range(32)
    sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(42))
    return StatefulDataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)


def test_new_processor_restores_saved_sampler_position(tmp_path):
    source = _loader()
    iterator = iter(source)
    for _ in range(5):
        next(iterator)
    checkpoint = tmp_path / "data.pt"
    torch.save(source.state_dict(), checkpoint)
    expected = [int(next(iterator)[0]) for _ in range(10)]

    processor_class = DataProcessor.__ray_metadata__.modified_class
    processor = object.__new__(processor_class)
    processor.train_dataloaders = [_loader()]
    processor.train_dataloader_sizes = [32]
    processor.generation_batches = 0
    processor.load_train_dataloader(str(checkpoint))

    resumed = iter(processor.train_dataloaders[0])
    actual = [int(next(resumed)[0]) for _ in range(10)]
    assert actual == expected, "A newly started processor must continue from the saved task position."
