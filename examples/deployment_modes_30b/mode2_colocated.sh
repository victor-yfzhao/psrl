#!/usr/bin/env bash
# Mode 2 — fully colocated.
# All three roles share a single train_pool, time-multiplexed per training step:
#   rollout -> (rollout-done) -> rm -> (reward-done) -> actor trains -> sleep actor.
# One node x 8 GPUs; rollout/rm/actor all live on train_pool via NIXL sleep/wake.
#
# Requirements (enforced in PivotRL_RayPPOTrainer._validate_config):
#   pivotrl.ps_mode in {nixl_cpu, nixl_gpu}; launch_reward_fn_async=True; staleness=0;
#   colocate_validate_and_train=False.
#
# Usage: bash mode2_colocated.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PIVOTRL_DEPLOY_MODE=colocated
export PIVOTRL_DEPLOY_EXPERIMENT=mode2_bs_128_colocated_test_weight_arena
export PIVOTRL_DEPLOY_STALENESS=0
export PIVOTRL_DEPLOY_RM_ASYNC=True
export PIVOTRL_DEPLOY_NNODES=8
export PIVOTRL_DEPLOY_TRAIN_NNODES=8
export PIVOTRL_DEPLOY_TRAIN_NGPUS=8
# Select with TRAIN_BACKEND=megatron. The default remains fsdp2.
export PIVOTRL_DEPLOY_TRAIN_BACKEND=${PIVOTRL_DEPLOY_TRAIN_BACKEND:-${TRAIN_BACKEND:-fsdp2}}
export PIVOTRL_DEPLOY_MEGATRON_TP=${PIVOTRL_DEPLOY_MEGATRON_TP:-${MEGATRON_TP:-2}}
export PIVOTRL_DEPLOY_MEGATRON_PP=${PIVOTRL_DEPLOY_MEGATRON_PP:-${MEGATRON_PP:-2}}
export PIVOTRL_DEPLOY_MEGATRON_CP=${PIVOTRL_DEPLOY_MEGATRON_CP:-${MEGATRON_CP:-1}}
export PIVOTRL_DEPLOY_MEGATRON_EP=${PIVOTRL_DEPLOY_MEGATRON_EP:-${MEGATRON_EP:-8}}
export PIVOTRL_DEPLOY_MEGATRON_ETP=${PIVOTRL_DEPLOY_MEGATRON_ETP:-${MEGATRON_ETP:-1}}
export PIVOTRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU=${PIVOTRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU:-${MEGATRON_MICRO_BSZ_PER_GPU:-1}}
# No separate shared pool in mode 2; all replicas ride on train_pool. These are
# unused by main_ppo's colocated branch but kept for the common arg builder.
export PIVOTRL_DEPLOY_SHARED_NNODES=0
export PIVOTRL_DEPLOY_SHARED_NGPUS=0
export PIVOTRL_DEPLOY_RM_NUM_REPLICAS=0
export PIVOTRL_DEPLOY_EXTRA="${PIVOTRL_DEPLOY_EXTRA:-} ++pivotrl.nixl.weight_arena.reward_enabled=true ++pivotrl.nixl.weight_arena.reward_cpu_cache_pin_memory=true"
export PIVOTRL_DEPLOY_SMOKE=${1:-0}
shift || true

launch_deployment_mode "$@"
