from __future__ import annotations

import heapq
from typing import Literal

import torch

def place_experts_balanced_lpt(
    expert_load: torch.Tensor,
    ep_size: int,
    *,
    tie_break: Literal["rank_id", "fewest_experts"] = "fewest_experts",
    sorted_ids: list[int] | None = None,
) :
    """Compute a per-layer expert placement that balances total load across EP ranks.

    Uses a greedy Longest-Processing-Time (LPT) style assignment with a hard capacity
    constraint of exactly `num_experts // ep_size` experts per EP rank.

    Args:
        expert_load: Tensor[float|int] shape [num_experts]. Larger means heavier.
        ep_size: number of expert-parallel ranks.
        tie_break: how to break ties when multiple ranks have equal current load.

    Returns:
        logical_to_physical mapping.
    """
    num_experts = expert_load.numel()
    local_experts = num_experts // ep_size

    # Sort experts by descending load (stable for reproducibility).
    # Keep logical ids.
    if sorted_ids is None:
        sorted_ids = torch.argsort(expert_load, descending=True, stable=True).tolist()

    rank_loads = [0] * ep_size
    rank_experts: list[list[int]] = [[] for _ in range(ep_size)]

    # Fast path: use a min-heap to avoid scanning all EP ranks for each expert.
    # Key:
    # - fewest_experts: (load, num_assigned, rank_id)
    # - rank_id:        (load, rank_id)
    # Both retain deterministic behavior and hard capacity constraints.
    if tie_break == "fewest_experts":
        heap = [(0, 0, r) for r in range(ep_size)]
    else:
        heap = [(0, r) for r in range(ep_size)]
    heapq.heapify(heap)

    for logical_eid in sorted_ids:
        if tie_break == "fewest_experts":
            load, cnt, r = heapq.heappop(heap)
            rank_experts[r].append(logical_eid)
            new_load = load + int(expert_load[logical_eid].item())
            rank_loads[r] = new_load
            new_cnt = cnt + 1
            if new_cnt < local_experts:
                heapq.heappush(heap, (new_load, new_cnt, r))
        else:
            load, r = heapq.heappop(heap)
            rank_experts[r].append(logical_eid)
            new_load = load + int(expert_load[logical_eid].item())
            rank_loads[r] = new_load
            if len(rank_experts[r]) < local_experts:
                heapq.heappush(heap, (new_load, r))

    logical_to_physical = torch.empty((num_experts,), dtype=torch.int64)
    for r in range(ep_size):
        experts_r = rank_experts[r]
        for j, logical_eid in enumerate(experts_r):
            physical_eid = r * local_experts + j
            logical_to_physical[logical_eid] = physical_eid

    return logical_to_physical


def compute_layerwise_logical_to_physical_mapping(
    batch_expert_load: torch.Tensor,
    ep_size: int,
    *,
    tie_break: Literal["rank_id", "fewest_experts"] = "fewest_experts",
) -> torch.Tensor:
    """Compute per-layer logical->physical expert mapping.

    Args:
        batch_expert_load: Tensor shape [num_layers, num_experts].
        ep_size: expert-parallel size.

    Returns:
        Tensor[int64] shape [num_layers, num_experts].
    """
    num_layers, num_experts = batch_expert_load.shape
    # Batch argsort once across all layers to reduce repeated sorting overhead.
    sorted_ids_by_layer = torch.argsort(batch_expert_load, dim=1, descending=True, stable=True)

    mappings = torch.empty((num_layers, num_experts), dtype=torch.int64)
    for layer in range(num_layers):
        mappings[layer] = place_experts_balanced_lpt(
            batch_expert_load[layer],
            ep_size,
            tie_break=tie_break,
            sorted_ids=sorted_ids_by_layer[layer].tolist(),
        )

    return mappings


def _compute_layerwise_expert_load_from_routed_experts(
    attention_mask: torch.Tensor,
    routed_experts: torch.Tensor,
    *,
    num_experts: int,
) -> torch.Tensor:
    """Compute per-layer per-expert load counts from routed expert indices.

    Args:
        attention_mask: [B, S] bool/int (True for active tokens)
        routed_experts: [B, S, L, topk] integer expert ids (logical)
        num_experts: total number of experts (E)

    Returns:
        Tensor[int64] of shape [L, E]
    """
    mask = attention_mask.to(torch.bool)

    b, s, num_layers, topk = routed_experts.shape
    loads = torch.zeros((num_layers, num_experts), dtype=torch.int64, device=routed_experts.device)

    # Gather active tokens: [L, num_valid_tokens, topk]
    # 提前一次性将其转置为layer维度在前，避免每次循环都要分配新内存并拷贝
    active_ids = routed_experts[mask].transpose(0, 1).reshape(num_layers, -1)

    for layer in range(num_layers):
        ids = active_ids[layer, :]
        loads[layer] = torch.bincount(ids, minlength=num_experts)

    return loads


def compute_micro_batch_logical_to_physical_mapping_list(
    micro_batches: list,
    *,
    ep_size: int,
    num_experts: int,
    tie_break: Literal["rank_id", "fewest_experts"] = "fewest_experts",
) -> list[torch.Tensor]:
    """Compute best-effort balanced expert placement per micro-batch.

    For each micro-batch, we compute per-layer per-expert load from
    (`attention_mask`, `routed_experts`) and then run the per-layer
    placement algorithm to produce a logical->physical mapping.

    Args:
        micro_batches: list of `DataProto` or `TensorDict`.
        ep_size: expert-parallel size.
        num_experts: total number of experts.
        tie_break: forwarded to placement.

    Returns:
        logical_to_physical_expert_mapping_list: list length == len(micro_batches),
        each item is Tensor[int64] shape [num_layers, num_experts].
    """
    if not isinstance(micro_batches, list) or len(micro_batches) == 0:
        raise ValueError("micro_batches must be a non-empty list")

    mappings: list[torch.Tensor] = []
    for mb in micro_batches:
        td = mb.batch if hasattr(mb, "batch") else mb
        if "attention_mask" not in td or "routed_experts" not in td:
            raise KeyError("each micro-batch must contain 'attention_mask' and 'routed_experts'")
        batch_expert_load = _compute_layerwise_expert_load_from_routed_experts(
            attention_mask=td["attention_mask"],
            routed_experts=td["routed_experts"],
            num_experts=num_experts,
        )
        batch_expert_load = batch_expert_load.cpu()  # move to CPU for placement computation
        mapping = compute_layerwise_logical_to_physical_mapping(
            batch_expert_load,
            ep_size,
            tie_break=tie_break,
        )
        mappings.append(mapping)

    return mappings
