"""
TMS / LD_PRELOAD helpers shared by rollout (`configure_worker`) and trainer (`RayWorkerGroup.worker_env`).

Aligned with main_pivotrl agentic_rl: train workers use `TMS_INIT_ENABLE=1` and `PIVOTRL_TMS_ENABLE`
when `tms.range` is train/all; rollout gen_worker uses `TMS_INIT_ENABLE=0` (vLLM subprocess path).
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from omegaconf import DictConfig, OmegaConf


def build_ray_train_worker_tms_env(pivotrl_config: DictConfig | dict[str, Any] | None) -> dict[str, str] | None:
    """
    Environment variables for verl `RayWorkerGroup(worker_env=...)`, matching main_pivotrl ray_trainer.

    Returns None when TMS preload is not required for training workers.
    """
    if pivotrl_config is None:
        return None

    tms_range = OmegaConf.select(pivotrl_config, "tms.range", default=None)
    enable_nixl = bool(OmegaConf.select(pivotrl_config, "tms.enable_nixl", default=False))
    if tms_range not in ("train", "all") and not enable_nixl:
        return None

    torch_memory_saver = importlib.import_module("torch_memory_saver")
    dynlib_path = os.path.join(
        os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
        "torch_memory_saver_hook_mode_preload.abi3.so",
    )
    if not os.path.exists(dynlib_path):
        raise FileNotFoundError(f"LD_PRELOAD so file {dynlib_path} does not exist.")

    return {
        "LD_PRELOAD": dynlib_path,
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "PIVOTRL_TMS_ENABLE": "1" if tms_range in ("train", "all") else "",
    }
