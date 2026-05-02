"""
Reward Model module for PSRL.

This module provides:
- PSRL_RewardModelWorker: Minimal vLLM worker for RM inference
- RewardModelManager: Manages multiple RM replicas and optional router
- RewardModelRouter: Load balancer for RM replicas
"""

from .manager import PSRL_RewardModelManager
from .router import PSRL_RewardModelRouter, launch_router_process
# from .worker import PSRL_RewardModelWorker
from .replica import PSRL_RewardModelReplica

__all__ = [
    # "PSRL_RewardModelWorker",
    "PSRL_RewardModelReplica",
    "PSRL_RewardModelManager",
    "PSRL_RewardModelRouter",
    "launch_router_process",
]

