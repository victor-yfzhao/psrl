#!/usr/bin/env bash
# Mode 3 — rollout / reward-model colocated.
# rollout + RM share shared_rollout_pool, time-multiplexed per buffer:
#   rollout -> (rollout-done) -> rm -> (reward-done) -> rollout (next buffer).
# Trainer lives on a separate train_pool and pipelines via staleness.
# Two nodes: shared_rollout_pool (8 GPUs, rollout+rm) + train_pool (8 GPUs, actor).
#
# Requirements: pivotrl.ps_mode in {nixl_cpu, nixl_gpu}; launch_reward_fn_async=True;
#   staleness=0.
#
# Usage: bash mode3_rollout_rm_colocated.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PIVOTRL_DEPLOY_MODE=rollout_rm_colocated
export PIVOTRL_DEPLOY_EXPERIMENT=mode3_bs_128_rollout_rm_colocated_rollout7b_rm8b
export PIVOTRL_DEPLOY_STALENESS=2
export PIVOTRL_DEPLOY_RM_ASYNC=True
export PIVOTRL_DEPLOY_NNODES=8
export PIVOTRL_DEPLOY_TRAIN_NNODES=4
export PIVOTRL_DEPLOY_TRAIN_NGPUS=8
# Select with TRAIN_BACKEND=megatron. The default remains fsdp2.
export PIVOTRL_DEPLOY_TRAIN_BACKEND=${PIVOTRL_DEPLOY_TRAIN_BACKEND:-${TRAIN_BACKEND:-fsdp2}}
export PIVOTRL_DEPLOY_MEGATRON_TP=${PIVOTRL_DEPLOY_MEGATRON_TP:-${MEGATRON_TP:-2}}
export PIVOTRL_DEPLOY_MEGATRON_PP=${PIVOTRL_DEPLOY_MEGATRON_PP:-${MEGATRON_PP:-2}}
export PIVOTRL_DEPLOY_MEGATRON_CP=${PIVOTRL_DEPLOY_MEGATRON_CP:-${MEGATRON_CP:-1}}
export PIVOTRL_DEPLOY_MEGATRON_EP=${PIVOTRL_DEPLOY_MEGATRON_EP:-${MEGATRON_EP:-8}}
export PIVOTRL_DEPLOY_MEGATRON_ETP=${PIVOTRL_DEPLOY_MEGATRON_ETP:-${MEGATRON_ETP:-1}}
export PIVOTRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU=${PIVOTRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU:-${MEGATRON_MICRO_BSZ_PER_GPU:-1}}
export PIVOTRL_DEPLOY_SHARED_NNODES=4
export PIVOTRL_DEPLOY_SHARED_NGPUS=8
export PIVOTRL_DEPLOY_RM_NUM_REPLICAS=0
export PIVOTRL_DEPLOY_SMOKE=${1:-0}
shift || true

launch_deployment_mode "$@"
