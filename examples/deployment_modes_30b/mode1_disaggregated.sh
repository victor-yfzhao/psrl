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

export PSRL_DEPLOY_MODE=disaggregated
# export PSRL_DEPLOY_EXPERIMENT=psrl_roll_3_rm_9_mode1_disaggregated_rollout7b_rm8b
export PSRL_DEPLOY_EXPERIMENT=mode1_bs_128_roll_4_rm_4_disaggregated
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
export PSRL_DEPLOY_RM_ASYNC=True
export PSRL_DEPLOY_NNODES=8
export PSRL_DEPLOY_TRAIN_NNODES=4
export PSRL_DEPLOY_TRAIN_NGPUS=8
# Independent rollout pool (2 nodes x 8 GPUs -> 4 TP4 rollout instances).
export PSRL_DEPLOY_SHARED_NNODES=2
export PSRL_DEPLOY_SHARED_NGPUS=8
# Independent gen-RM pool (2 nodes x 8 GPUs -> 4 TP4 RM replicas).
export PSRL_DEPLOY_RM_NUM_REPLICAS=4
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

launch_deployment_mode "$@"
