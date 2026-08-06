"""ITL/Throughput/Harmonic-mean based elastic scaling policy.

This module implements ``ITLHarmonicScalingPolicy``, a variant of
``ITLScalingPolicy`` that reuses the same fitted ITL latency model, the same
candidate enumeration, the same wake/sleep selection, and the same bottleneck
role determination, but changes the system-level objective.

Objective difference vs ``ITLScalingPolicy``:

- ``ITLScalingPolicy`` maximizes the *bottleneck* throughput, i.e. the system
  throughput is ``min(rollout_tp, rm_tp)`` (or the pressure-weighted
  ``min(rollout_tp / w_rollout, rm_tp / w_rm)``).
- ``ITLHarmonicScalingPolicy`` maximizes the *weighted harmonic mean* of the two
  roles' throughputs::

      H = (w_rollout + w_rm) / (w_rollout / rollout_tp + w_rm / rm_tp)

  The harmonic mean penalizes imbalance: when one role starves (throughput 0)
  the whole objective collapses to 0, and it is dominated by the weaker side
  while still rewarding improvements on either side.

  Set ``itl_policy.harmonic_denominator_use_max`` to use the maximum of the
  two reciprocal-throughput terms as the denominator instead of their sum.
  This preserves the numerator and weight semantics while making the objective
  more bottleneck-like.

The per-role throughput estimates (``balanced_min`` vs ``sum``), the
``role_throughput_weight_basis`` (``request_count`` / ``token_count`` /
``none``), and the ``role_throughput_weight_mode`` (``share`` / ``raw``) all
keep the same meaning as in ``ITLScalingPolicy``. When weighting is disabled,
``_role_throughput_weights`` returns uniform ``(0.5, 0.5)`` divisors, which
makes ``H`` reduce to the standard (unweighted) harmonic mean
``2 * rollout_tp * rm_tp / (rollout_tp + rm_tp)``.

Idle-instance handling differs from ``ITLScalingPolicy``: an instance (or a
whole role) with no outstanding requests contributes ``0`` throughput, not
``inf``. The bottleneck variant uses ``inf`` so an idle side is never the
bottleneck; under the harmonic mean that ``inf`` would remove the side's term
from the denominator and inflate the objective, so it is recorded as ``0``
instead (collapsing ``H`` to 0 only when a side is genuinely idle/starved).

Starvation fallback (degraded mode): the ``idle -> 0`` convention makes the
harmonic objective *degenerate* when a whole role is demand-starved (awake
instances but zero in-flight requests and zero router backlog): ``H`` collapses
to 0 for every candidate, so ``delta_throughput`` is 0 everywhere and the policy
can never beat ``min_gain`` -- it is stuck in ``no_action`` even though the
loaded side is over-allocated and could absorb more capacity. When
``starvation_fallback`` is enabled (default) and exactly one role is
demand-starved, the policy degrades to the bottleneck-style objective for that
cycle: the starved side is treated as ``+inf`` (no resistance) and the system
throughput is scored as ``loaded_tp / w_loaded``, and the capacity bottleneck is
flipped to the *loaded* role so candidates scale the side that actually has
work. Scaling up the loaded role then evicts idle starved-side instances
(migration cost ``Phi`` ~= 0) for a real throughput gain, breaking the deadlock.

Which side to scale is *unchanged* from ``ITLScalingPolicy``: the bottleneck
role is still picked by comparing ``role_throughput / weight`` and the side
with the smaller value is the one eligible for expansion. Only the objective
used to score candidate gains (``gain_l``) switches from the bottleneck ``min``
to the harmonic mean.

Config lives under ``policy_config["itl_policy"]`` (see ``psrl.yaml``); select
this variant with ``scaling_policy_variant: itl_harmonic``.
"""

from __future__ import annotations

import math
from typing import Any

from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.elastic_rm.itl_scaling_policy import ITLScalingPolicy, _RouterWaitingLoad
from psrl.utils.elastic_rm.scaling_policy import InstanceSignal


def _inv_throughput(tp: float) -> float:
    """Return ``1 / tp`` with stable handling of 0 and infinite throughput.

    A role with zero throughput (starved, or idle under the harmonic policy where
    idle instances contribute 0) contributes ``inf``, collapsing the harmonic
    mean to 0. A genuinely idle role reporting ``inf`` (only possible under the
    bottleneck policy) contributes 0 so it does not drag the mean down.
    """
    if tp <= 0.0:
        return math.inf
    if math.isinf(tp):
        return 0.0
    return 1.0 / tp


class ITLHarmonicScalingPolicy(ITLScalingPolicy):
    """Elastic policy that maximizes the harmonic-mean throughput objective."""

    @staticmethod
    def _idle_instance_throughput() -> float:
        """Idle instances contribute 0 throughput so the harmonic mean is not inflated.

        With the bottleneck (``min``) objective an idle side returns ``inf`` so it
        is never the bottleneck; under the harmonic mean that ``inf`` would drop
        the side's term from the denominator and push system throughput toward
        ``inf``. Recording 0 instead makes an idle instance contribute nothing
        to the role's throughput and, when a whole role is idle, collapses the
        harmonic mean to 0 (the "starved side" behavior the objective intends).
        """
        return 0.0

    def _weighted_system_throughput(
        self,
        rollout_tp: float,
        rm_tp: float,
        w_rollout: float,
        w_rm: float,
    ) -> float:
        """Apply per-role throughput divisors and return the weighted harmonic mean.

        By default,
        ``H = (w_rollout + w_rm) / (w_rollout / rollout_tp + w_rm / rm_tp)``.
        When ``harmonic_denominator_use_max`` is enabled, the denominator is
        ``max(w_rollout / rollout_tp, w_rm / rm_tp)`` instead.
        Zero-weight sides fall back to divisor 1.0 (same convention as the
        bottleneck variant) so a disabled/uniform configuration reduces to the
        standard harmonic mean.
        """
        wr = w_rollout if w_rollout > 0.0 else 1.0
        wm = w_rm if w_rm > 0.0 else 1.0
        rollout_term = wr * _inv_throughput(rollout_tp)
        rm_term = wm * _inv_throughput(rm_tp)
        if self.harmonic_denominator_use_max:
            denom = max(rollout_term, rm_term)
        else:
            denom = rollout_term + rm_term
        if math.isinf(denom):
            # At least one role has zero throughput -> harmonic mean is 0.
            return 0.0
        if denom == 0.0:
            # Both roles idle (infinite throughput) -> objective is unbounded.
            return math.inf
        return (wr + wm) / denom

    def _system_throughput(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        rollout_n: int,
        rm_n: int,
        router_backlog_by_role: dict[PSRL_Role, Any] | None = None,
        wake_ids_by_role: dict[PSRL_Role, set[int]] | None = None,
        sleep_ids_by_role: dict[PSRL_Role, set[int]] | None = None,
        load_overrides_by_role: dict[PSRL_Role, dict[int, tuple[float, float]]] | None = None,
    ) -> float:
        """System throughput: weighted harmonic mean across Rollout and RewardModel.

        The per-role throughput calculation (``sum`` vs ``balanced_min``) and
        the per-role weight derivation are inherited unchanged from
        ``ITLScalingPolicy``; only the final combine switches from ``min`` to
        the harmonic mean. Weights are always computed so that a disabled
        ``role_throughput_weight_enable`` yields uniform ``(0.5, 0.5)``, i.e.
        the standard harmonic mean.
        """
        backlog_map = router_backlog_by_role or {}
        wake_map = wake_ids_by_role or {}
        sleep_map = sleep_ids_by_role or {}
        load_override_map = load_overrides_by_role or {}
        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        rollout_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.Rollout))
        rm_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel))
        if self.throughput_objective == "sum":
            rollout_tp = self._role_throughput_sum(
                rollout_signals,
                rollout_n,
                rollout_waiting,
                wake_map.get(PSRL_Role.Rollout),
                sleep_map.get(PSRL_Role.Rollout),
                load_override_map.get(PSRL_Role.Rollout),
            )
            rm_tp = self._role_throughput_sum(
                rm_signals,
                rm_n,
                rm_waiting,
                wake_map.get(PSRL_Role.RewardModel),
                sleep_map.get(PSRL_Role.RewardModel),
                load_override_map.get(PSRL_Role.RewardModel),
            )
        else:
            rollout_tp = self._role_throughput(rollout_signals, rollout_n, rollout_waiting)
            rm_tp = self._role_throughput(rm_signals, rm_n, rm_waiting)
        w_rollout, w_rm = self._role_throughput_weights(rollout_signals, rm_signals, rollout_waiting, rm_waiting)
        starved_role = self._detect_starved_role(
            rollout_signals=rollout_signals,
            rm_signals=rm_signals,
            rollout_waiting=rollout_waiting,
            rm_waiting=rm_waiting,
        )
        if starved_role is None:
            return self._weighted_system_throughput(rollout_tp, rm_tp, w_rollout, w_rm)
        # Degraded mode: exactly one role is demand-starved, so the harmonic mean
        # collapses to 0 and can no longer rank candidates. Fall back to the
        # bottleneck-style objective with the starved side treated as +inf (no
        # resistance): the system throughput is driven by the loaded side alone.
        #
        # The loaded side's throughput is estimated as a *balanced drain rate*
        # (``n * avg_requests / ITL(avg_tokens, avg_requests)``) rather than the
        # ``sum`` of per-instance current queues. Under starvation there is no
        # router backlog, so the ``sum`` objective would credit a newly-woken
        # instance with 0 load (its current running queue is 0) and report no
        # gain from added capacity. The balanced-drain model instead treats the
        # in-flight batch as redistributable over ``n`` instances, which is the
        # physically right model for a barrier-stalled batch and makes
        # ``delta_throughput`` reflect the real benefit of giving the loaded side
        # more capacity.
        if starved_role == PSRL_Role.Rollout:
            return self._degraded_loaded_throughput(rm_signals, rm_n, rm_waiting, w_rm)
        return self._degraded_loaded_throughput(rollout_signals, rollout_n, rollout_waiting, w_rollout)

    def _combine_request_level_role_throughputs(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        role_throughputs: dict[PSRL_Role, float],
        router_backlog_by_role: dict[PSRL_Role, Any] | None,
    ) -> float:
        """Apply harmonic/starvation math to request-level role simulations."""
        backlog_map = router_backlog_by_role or {}
        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        rollout_waiting = self._normalize_router_waiting_load(
            backlog_map.get(PSRL_Role.Rollout)
        )
        rm_waiting = self._normalize_router_waiting_load(
            backlog_map.get(PSRL_Role.RewardModel)
        )
        w_rollout, w_rm = self._role_throughput_weights(
            rollout_signals,
            rm_signals,
            rollout_waiting,
            rm_waiting,
        )
        starved_role = self._detect_starved_role(
            rollout_signals=rollout_signals,
            rm_signals=rm_signals,
            rollout_waiting=rollout_waiting,
            rm_waiting=rm_waiting,
        )
        if starved_role == PSRL_Role.Rollout:
            divisor = w_rm if w_rm > 0.0 else 1.0
            return role_throughputs[PSRL_Role.RewardModel] / divisor
        if starved_role == PSRL_Role.RewardModel:
            divisor = w_rollout if w_rollout > 0.0 else 1.0
            return role_throughputs[PSRL_Role.Rollout] / divisor
        return self._weighted_system_throughput(
            role_throughputs[PSRL_Role.Rollout],
            role_throughputs[PSRL_Role.RewardModel],
            w_rollout,
            w_rm,
        )

    def _degraded_loaded_throughput(
        self,
        loaded_signals: list[InstanceSignal],
        loaded_n: int,
        loaded_waiting: _RouterWaitingLoad,
        w_loaded: float,
    ) -> float:
        """Bottleneck-style throughput for the loaded side under starvation.

        Returns the loaded role's balanced batch-drain rate divided by its
        pressure weight: ``n * (avg_requests / ITL(avg_tokens, avg_requests)) / w``.
        Adding capacity (larger ``n``) lowers the per-instance queue depth and
        token load, shrinking ITL and raising the drain rate, so scale-up
        candidates on the loaded side earn a positive ``delta_throughput``.
        """
        if loaded_n <= 0:
            return 0.0
        per_instance_tp = self._role_throughput(loaded_signals, loaded_n, loaded_waiting)
        if math.isinf(per_instance_tp) or math.isnan(per_instance_tp):
            return per_instance_tp
        total_drain = per_instance_tp * float(loaded_n)
        divisor = w_loaded if w_loaded > 0.0 else 1.0
        return total_drain / divisor

    def _role_request_pressure(
        self,
        role_signals: list[InstanceSignal],
        waiting: _RouterWaitingLoad,
    ) -> float:
        """Total outstanding request pressure for a role (in-instance + router)."""
        in_instance = sum(self._vllm_current_request_load(signal) for signal in role_signals if signal.is_awaken)
        return in_instance + float(waiting.request_count)

    def _role_is_demand_starved(
        self,
        role_signals: list[InstanceSignal],
        waiting: _RouterWaitingLoad,
    ) -> bool:
        """True when a role has awake capacity but zero outstanding demand.

        Demand starvation is a *role-level* condition distinct from an idle
        instance within a loaded role: here the whole role has no in-flight
        requests and no router backlog, so its awake instances are producing
        nothing because there is nothing to do -- not because they are the
        slower side. The harmonic mean's ``idle -> 0`` convention is wrong for
        this case (it flags the starved role as the bottleneck), so the
        starvation fallback treats it as +inf instead.
        """
        awake_n = sum(1 for signal in role_signals if signal.is_awaken)
        if awake_n <= 0:
            return False
        return self._role_request_pressure(role_signals, waiting) <= 0.0

    def _detect_starved_role(
        self,
        *,
        rollout_signals: list[InstanceSignal],
        rm_signals: list[InstanceSignal],
        rollout_waiting: _RouterWaitingLoad,
        rm_waiting: _RouterWaitingLoad,
    ) -> PSRL_Role | None:
        """Return the single demand-starved role, or ``None`` if not degraded.

        Only triggers when exactly one role is starved. If both roles are idle
        the system has no work at all (leave the objective as-is); if neither is
        starved the normal harmonic objective applies.
        """
        if not self.starvation_fallback:
            return None
        rollout_starved = self._role_is_demand_starved(rollout_signals, rollout_waiting)
        rm_starved = self._role_is_demand_starved(rm_signals, rm_waiting)
        if rollout_starved and not rm_starved:
            return PSRL_Role.Rollout
        if rm_starved and not rollout_starved:
            return PSRL_Role.RewardModel
        return None

    def _enumerate_bottleneck_roles(
        self,
        *,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        role_throughputs: dict[PSRL_Role, float],
        rollout_signals: list[InstanceSignal],
        rm_signals: list[InstanceSignal],
        rollout_waiting: _RouterWaitingLoad,
        rm_waiting: _RouterWaitingLoad,
        w_rollout: float,
        w_rm: float,
        rollout_bottleneck: float,
        rm_bottleneck: float,
        router_backlog_by_role: dict[PSRL_Role, Any] | None,
    ) -> list[PSRL_Role]:
        """Flip the bottleneck to the loaded role under demand starvation.

        When a role is demand-starved its harmonic throughput is 0, which makes
        ``throughput / weight`` the smallest and flags it as the capacity
        bottleneck -- but expanding a role that has no demand is pointless (and
        steals capacity from the loaded side). The useful lever is the *loaded*
        role, so vary its awake count instead: scaling it up (waking sleeping
        instances, evicting idle starved-side instances that share bundles) adds
        real throughput at near-zero migration cost.
        """
        starved_role = self._detect_starved_role(
            rollout_signals=rollout_signals,
            rm_signals=rm_signals,
            rollout_waiting=rollout_waiting,
            rm_waiting=rm_waiting,
        )
        if starved_role == PSRL_Role.Rollout:
            return [PSRL_Role.RewardModel]
        if starved_role == PSRL_Role.RewardModel:
            return [PSRL_Role.Rollout]
        return super()._enumerate_bottleneck_roles(
            grouped=grouped,
            role_throughputs=role_throughputs,
            rollout_signals=rollout_signals,
            rm_signals=rm_signals,
            rollout_waiting=rollout_waiting,
            rm_waiting=rm_waiting,
            w_rollout=w_rollout,
            w_rm=w_rm,
            rollout_bottleneck=rollout_bottleneck,
            rm_bottleneck=rm_bottleneck,
            router_backlog_by_role=router_backlog_by_role,
        )

    def _objective_log_extras(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        router_backlog_by_role: dict[PSRL_Role, Any] | None,
    ) -> dict[str, Any]:
        """Surface starvation-fallback state in the ``itl_objective`` log section."""
        backlog_map = router_backlog_by_role or {}
        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        rollout_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.Rollout))
        rm_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel))
        starved_role = self._detect_starved_role(
            rollout_signals=rollout_signals,
            rm_signals=rm_signals,
            rollout_waiting=rollout_waiting,
            rm_waiting=rm_waiting,
        )
        if starved_role is None:
            return {"starvation_fallback": self.starvation_fallback, "degraded_mode": False}
        rollout_pressure = self._role_request_pressure(rollout_signals, rollout_waiting)
        rm_pressure = self._role_request_pressure(rm_signals, rm_waiting)
        return {
            "starvation_fallback": self.starvation_fallback,
            "degraded_mode": True,
            "starved_role": starved_role.name,
            "rollout_pressure": f"{rollout_pressure:.1f}",
            "rm_pressure": f"{rm_pressure:.1f}",
        }
