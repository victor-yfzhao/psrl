from dataclasses import dataclass, field
from typing import Protocol

import torch

_PRECISION_SENSITIVE_PATTERNS = ("linear_attn.norm.weight", "A_log")


def precision_sensitive_parameter_stats(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, dict[str, object]]:
    """Summarize small Qwen3.5 tensors that are sensitive to sync transforms."""
    stats: dict[str, dict[str, object]] = {}
    for pattern in _PRECISION_SENSITIVE_PATTERNS:
        matches = [(key, tensor) for key, tensor in state_dict.items() if pattern in key]
        if not matches:
            continue
        values = torch.cat([tensor.detach().float().reshape(-1) for _, tensor in matches])
        stats[pattern] = {
            "tensor_count": len(matches),
            "element_count": values.numel(),
            "dtypes": sorted({str(tensor.dtype) for _, tensor in matches}),
            "min": values.min().item(),
            "max": values.max().item(),
            "mean": values.mean().item(),
        }
    return stats


class ParamSyncAction(Protocol):
    """Lifecycle hooks for canonical tensors that are not direct model-parameter views."""

    def before_push(self, state_dict: dict[str, torch.Tensor]) -> None: ...

    def after_push(self, state_dict: dict[str, torch.Tensor]) -> None: ...

    def after_pull(self, state_dict: dict[str, torch.Tensor]) -> None: ...


@dataclass
class ParamSyncPlan:
    actions: list[ParamSyncAction] = field(default_factory=list)

    def add(self, action: ParamSyncAction) -> None:
        self.actions.append(action)

    def before_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for action in self.actions:
                action.before_push(state_dict)

    def after_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for action in reversed(self.actions):
                action.after_push(state_dict)

    def after_pull(self, state_dict: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for action in self.actions:
                action.after_pull(state_dict)


@dataclass
class ZeroCenteredGammaSync:
    """Expose Megatron zero-centered gamma as standard gamma during PS sync."""

    key: str

    def before_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key in state_dict:
            state_dict[self.key].add_(1)

    def after_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key in state_dict:
            state_dict[self.key].sub_(1)

    def after_pull(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key in state_dict:
            state_dict[self.key].sub_(1)


@dataclass
class DTypeCastSync:
    """Keep an externally exposed dtype copy synchronized with a train parameter."""

    key: str
    source_param: torch.Tensor

    def before_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key in state_dict:
            state_dict[self.key].copy_(self.source_param.to(dtype=state_dict[self.key].dtype))

    def after_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        pass

    def after_pull(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key in state_dict:
            self.source_param.copy_(state_dict[self.key].to(dtype=self.source_param.dtype))


def register_megatron_param_sync_actions(
    sync_plan: ParamSyncPlan,
    state_dict: dict[str, torch.Tensor],
    fp32_patterns: tuple[str, ...],
) -> None:
    """Register Megatron-only representation transforms at the PS boundary."""
    for key, tensor in state_dict.items():
        if key.endswith("linear_attn.norm.weight"):
            sync_plan.add(ZeroCenteredGammaSync(key=key))
        if tensor.dtype != torch.float32 and any(pattern in key for pattern in fp32_patterns):
            sync_plan.add(DTypeCastSync(key=key, source_param=tensor))
            state_dict[key] = tensor.float()
