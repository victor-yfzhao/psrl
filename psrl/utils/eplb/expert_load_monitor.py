from __future__ import annotations

from typing import Any, Literal

import ray
import torch
from omegaconf import OmegaConf

from .expert_placement import compute_layerwise_logical_to_physical_mapping


@ray.remote
class ExpertLoadMonitor:
	def __init__(self, eplb_config):
		self.config = eplb_config

		eplb_threshold = OmegaConf.select(self.config, "batch_level_eplb_threshold")
		if eplb_threshold is None:
			eplb_threshold = 0.30
		eplb_tie_break = OmegaConf.select(self.config, "batch_level_eplb_tie_break")
		if eplb_tie_break is None:
			eplb_tie_break = "fewest_experts"
		eplb_min_tokens = OmegaConf.select(self.config, "batch_level_eplb_min_tokens_for_rebalance")
		if eplb_min_tokens is None:
			eplb_min_tokens = 1

		self._init(eplb_threshold, eplb_tie_break, eplb_min_tokens)

	def _init(
		self,
		imbalance_threshold: float = 0.40,
		tie_break: Literal["rank_id", "fewest_experts"] = "fewest_experts",
		min_tokens_for_rebalance: int = 1,
	) -> None:
		"""Ray actor for monitoring EP-rank expert load and triggering remapping.

		Args:
			imbalance_threshold: If imbalance score exceeds this threshold, rebalance is needed.
			tie_break: Tie-break policy for placement algorithm.
			min_tokens_for_rebalance: Rebalance is disabled when valid token count is below this value.
		"""
		if imbalance_threshold < 0:
			raise ValueError("imbalance_threshold must be non-negative")
		if min_tokens_for_rebalance < 0:
			raise ValueError("min_tokens_for_rebalance must be non-negative")

		self.imbalance_threshold = float(imbalance_threshold)
		self.tie_break = tie_break
		self.min_tokens_for_rebalance = int(min_tokens_for_rebalance)

		self._last_need_rebalance: bool = False
		self._last_new_mapping: torch.Tensor | None = None
		self._last_analysis: dict[str, Any] | None = None

	def analyze_expert_load(
		self,
		routed_experts: torch.Tensor,
		attention_mask: torch.Tensor,
		ep_size: int,
		logical_to_physical_mapping: torch.Tensor | None = None,
		num_experts: int | None = None,
		tie_break: Literal["rank_id", "fewest_experts"] | None = None,
	) -> dict[str, Any]:
		"""Analyze EP-rank load and optionally compute new expert mapping.

		Args:
			routed_experts: Tensor shape [B, S, L, topk], logical expert ids.
			attention_mask: Tensor shape [B, S], active-token mask.
			ep_size: EP world size.
			logical_to_physical_mapping: Current mapping Tensor[L, E]. If None, identity mapping is assumed.
			num_experts: Expert count E. If None, inferred from routed_experts / mapping.
			tie_break: Optional tie-break strategy overriding actor default.

		Returns:
			dict containing load stats and rebalance result.
		"""
		if routed_experts.ndim != 4:
			raise ValueError("routed_experts must have shape [B, S, L, topk]")
		if attention_mask.ndim != 2:
			raise ValueError("attention_mask must have shape [B, S]")
		if ep_size <= 0:
			raise ValueError("ep_size must be positive")
		if routed_experts.shape[:2] != attention_mask.shape:
			raise ValueError("routed_experts[:2] must match attention_mask shape")

		b, s, num_layers, _ = routed_experts.shape
		valid_mask = attention_mask.to(torch.bool)
		valid_token_count = int(valid_mask.sum().item())

		if num_experts is None:
			if logical_to_physical_mapping is not None:
				num_experts = int(logical_to_physical_mapping.shape[-1])
			else:
				num_experts = int(routed_experts.max().item()) + 1
		if num_experts <= 0:
			raise ValueError("num_experts must be positive")
		if num_experts % ep_size != 0:
			raise ValueError(f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})")

		if logical_to_physical_mapping is not None:
			if logical_to_physical_mapping.ndim != 2:
				raise ValueError("logical_to_physical_mapping must have shape [L, E]")
			if logical_to_physical_mapping.shape[0] != num_layers:
				raise ValueError(
					f"mapping num_layers mismatch: {logical_to_physical_mapping.shape[0]} vs {num_layers}"
				)
			if logical_to_physical_mapping.shape[1] != num_experts:
				raise ValueError(
					f"mapping num_experts mismatch: {logical_to_physical_mapping.shape[1]} vs {num_experts}"
				)

		local_experts = num_experts // ep_size
		rank_loads = torch.zeros((num_layers, ep_size), dtype=torch.int64)

		active_routed = routed_experts[valid_mask].reshape(-1, num_layers, routed_experts.shape[-1]).transpose(0, 1)

		for layer in range(num_layers):
			logical_ids = active_routed[layer].reshape(-1).to(torch.long)
			if logical_ids.numel() == 0:
				continue

			if logical_to_physical_mapping is not None:
				layer_mapping = logical_to_physical_mapping[layer].to(logical_ids.device)
				physical_ids = layer_mapping[logical_ids]
			else:
				physical_ids = logical_ids

			ep_rank_ids = torch.div(physical_ids, local_experts, rounding_mode="floor")
			layer_rank_load = torch.bincount(ep_rank_ids, minlength=ep_size)
			rank_loads[layer] = layer_rank_load.to(torch.int64)

		need_rebalance, imbalance_info = self._need_rebalance(rank_loads, valid_token_count=valid_token_count)

		new_mapping = None
		if need_rebalance:
			new_mapping = self._calculate_expert_mapping(
				routed_experts=routed_experts,
				attention_mask=attention_mask,
				ep_size=ep_size,
				num_experts=num_experts,
				tie_break=tie_break or self.tie_break,
			)

		self._last_need_rebalance = need_rebalance
		self._last_new_mapping = new_mapping
		self._last_analysis = {
			"need_rebalance": need_rebalance,
			"rank_loads": rank_loads,
			"imbalance_info": imbalance_info,
			"valid_token_count": valid_token_count,
			"batch_shape": (b, s, num_layers),
			"ep_size": ep_size,
			"num_experts": num_experts,
		}
		print(f"eplbdebug, {self._last_analysis=}")
		if new_mapping is not None:
			self._last_analysis["logical_to_physical_mapping"] = new_mapping

		return self._last_analysis

	def get_new_mapping(self) -> torch.Tensor | None:
		"""Return new mapping from the most recent analysis.

		Returns:
			None if no rebalance is required; otherwise Tensor[L, E].
		"""
		if not self._last_need_rebalance:
			return None
		return self._last_new_mapping

	def _need_rebalance(self, rank_loads: torch.Tensor, valid_token_count: int) -> tuple[bool, dict[str, Any]]:
		"""Decide whether expert rebalancing is needed.

		Imbalance score is defined per layer as
			(max_rank_load - mean_rank_load) / max(mean_rank_load, 1)
		and we use the max score across layers as the global criterion.
		"""
		if rank_loads.ndim != 2:
			raise ValueError("rank_loads must have shape [L, ep_size]")

		if valid_token_count < self.min_tokens_for_rebalance:
			return False, {
				"reason": "insufficient_tokens",
				"threshold": self.imbalance_threshold,
				"global_max_imbalance": 0.0,
				"layer_imbalance": torch.zeros(rank_loads.shape[0], dtype=torch.float32),
			}

		rank_loads_f = rank_loads.to(torch.float32)
		mean_per_layer = rank_loads_f.mean(dim=1)
		max_per_layer = rank_loads_f.max(dim=1).values
		layer_imbalance = (max_per_layer - mean_per_layer) / torch.clamp(mean_per_layer, min=1.0)
		global_max_imbalance = float(layer_imbalance.max().item()) if layer_imbalance.numel() > 0 else 0.0
		need = global_max_imbalance > self.imbalance_threshold

		return need, {
			"threshold": self.imbalance_threshold,
			"global_max_imbalance": global_max_imbalance,
			"layer_imbalance": layer_imbalance,
		}

	def _calculate_expert_mapping(
		self,
		routed_experts: torch.Tensor,
		attention_mask: torch.Tensor,
		ep_size: int,
		num_experts: int,
		tie_break: Literal["rank_id", "fewest_experts"] = "fewest_experts",
	) -> torch.Tensor:
		"""Compute new layer-wise logical->physical mapping.

		This method computes per-layer expert load from routed experts and calls
		`compute_layerwise_logical_to_physical_mapping`.
		"""
		if ep_size <= 0:
			raise ValueError("ep_size must be positive")
		if num_experts <= 0:
			raise ValueError("num_experts must be positive")

		mask = attention_mask.to(torch.bool)
		_, _, num_layers, _ = routed_experts.shape

		batch_expert_load = torch.zeros((num_layers, num_experts), dtype=torch.int64, device=routed_experts.device)
		active_ids = routed_experts[mask].transpose(0, 1).reshape(num_layers, -1)

		for layer in range(num_layers):
			layer_ids = active_ids[layer].to(torch.long)
			if layer_ids.numel() == 0:
				continue
			batch_expert_load[layer] = torch.bincount(layer_ids, minlength=num_experts)

		return compute_layerwise_logical_to_physical_mapping(
			batch_expert_load=batch_expert_load.cpu(),
			ep_size=ep_size,
			tie_break=tie_break,
		)

