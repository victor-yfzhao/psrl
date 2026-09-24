#!/usr/bin/env bash
# Mode 1 — fully disaggregated.
# rollout / reward-model / trainer each in INDEPENDENT pools, concurrent async
# pipeline. Eight nodes: rollout pool (16 GPUs, 4 TP4 instances) + reward-model
# pool (16 GPUs, 4 TP4 RM replicas) + train pool (32 GPUs, actor).
#
# Usage: bash mode1_disaggregated.sh [smoke_test=0] [extra hydra overrides...]
#   smoke_test=1 -> 2 steps, small batch (quick launch check).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PIVOTRL_DEPLOY_MODE=disaggregated
# export PIVOTRL_DEPLOY_EXPERIMENT=pivotrl_roll_3_rm_9_mode1_disaggregated_rollout7b_rm8b
export PIVOTRL_DEPLOY_EXPERIMENT=mode1_roll_4_rm_4_disaggregated
export PIVOTRL_DEPLOY_STALENESS=${STALENESS:-2}
export PIVOTRL_DEPLOY_RM_ASYNC=True
export PIVOTRL_DEPLOY_NNODES=16
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
# Independent rollout pool (2 nodes x 8 GPUs -> 4 TP4 rollout instances).
export PIVOTRL_DEPLOY_SHARED_NNODES=4
export PIVOTRL_DEPLOY_SHARED_NGPUS=8
# Independent gen-RM pool (2 nodes x 8 GPUs -> 4 TP4 RM replicas).
export PIVOTRL_DEPLOY_RM_NUM_REPLICAS=4
export PIVOTRL_DEPLOY_SMOKE=${1:-0}
shift || true

launch_deployment_mode "$@"
