"""Verify complete training metric rows are committed to W&B."""

from unittest.mock import MagicMock

import pytest
from verl.utils.tracking import Tracking

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("backend", [None, ["wandb"], ["console"]])
def test_tracking_commits_wandb_rows_and_preserves_backend_selection(backend):
    tracking = object.__new__(Tracking)
    wandb = MagicMock()
    console = MagicMock()
    tracking.logger = {"wandb": wandb, "console": console}
    metrics = {"training/global_step": 3, "actor/loss": 0.01}

    tracking.log(metrics, step=3, backend=backend)

    if backend is None or "wandb" in backend:
        wandb.log.assert_called_once_with(data=metrics, step=3, commit=True)
    else:
        wandb.log.assert_not_called()
    if backend is None or "console" in backend:
        console.log.assert_called_once_with(data=metrics, step=3)
    else:
        console.log.assert_not_called()
