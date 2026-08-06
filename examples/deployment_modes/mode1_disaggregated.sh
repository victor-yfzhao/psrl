#!/usr/bin/env bash
# Mode 1 — fully disaggregated.
# rollout / reward-model / trainer each in INDEPENDENT pools, concurrent async
# pipeline. Three nodes: rollout pool (8 GPUs, 8 instances) + reward-model pool
# (8 GPUs, 8 RM replicas) + train pool (8 GPUs, actor).
#
# Usage: bash mode1_disaggregated.sh [smoke_test=0] [extra hydra overrides...]
#   smoke_test=1 -> 2 steps, small batch (quick launch check).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PSRL_DEPLOY_MODE=disaggregated
# export PSRL_DEPLOY_EXPERIMENT=psrl_roll_3_rm_9_mode1_disaggregated_rollout7b_rm8b
export PSRL_DEPLOY_EXPERIMENT=mode1_bs_128_roll_2_rm_6_disaggregated
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
export PSRL_DEPLOY_RM_ASYNC=True
export PSRL_DEPLOY_NNODES=2
export PSRL_DEPLOY_TRAIN_NNODES=1
export PSRL_DEPLOY_TRAIN_NGPUS=8
# Independent rollout pool (1 node x 4 GPUs -> 4 rollout instances).
export PSRL_DEPLOY_SHARED_NNODES=1
export PSRL_DEPLOY_SHARED_NGPUS=2
# Independent gen-RM pool (1 node x 4 GPUs -> 4 RM replicas, 1 GPU each).
export PSRL_DEPLOY_RM_NUM_REPLICAS=6
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

launch_deployment_mode "$@"
