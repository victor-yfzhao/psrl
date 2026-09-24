"""
Reward Model module for PivotRL.

This module provides:
- PivotRL_RewardModelWorker: Minimal vLLM worker for RM inference
- RewardModelManager: Manages multiple RM replicas and optional router
- RewardModelRouter: Load balancer for RM replicas
"""

from .manager import PivotRL_RewardModelManager
from .router import PivotRL_RewardModelRouter, launch_router_process
# from .worker import PivotRL_RewardModelWorker
from .replica import PivotRL_RewardModelReplica

__all__ = [
    # "PivotRL_RewardModelWorker",
    "PivotRL_RewardModelReplica",
    "PivotRL_RewardModelManager",
    "PivotRL_RewardModelRouter",
    "launch_router_process",
]

