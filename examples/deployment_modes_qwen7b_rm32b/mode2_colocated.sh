#!/usr/bin/env bash
# Mode 2: rollout, RM, and trainer time-multiplex one 32-GPU train pool.
# Usage: bash mode2_colocated.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PIVOTRL_DEPLOY_MODE=colocated
export PIVOTRL_DEPLOY_EXPERIMENT=mode2_bs_128_colocated
export PIVOTRL_DEPLOY_STALENESS=0
export PIVOTRL_DEPLOY_RM_ASYNC=True
export PIVOTRL_DEPLOY_NNODES=4
export PIVOTRL_DEPLOY_TRAIN_NNODES=4
export PIVOTRL_DEPLOY_TRAIN_NGPUS=8
export PIVOTRL_DEPLOY_SHARED_NNODES=0
export PIVOTRL_DEPLOY_SHARED_NGPUS=0
export PIVOTRL_DEPLOY_RM_NUM_REPLICAS=0
export PIVOTRL_DEPLOY_SMOKE=${1:-0}
shift || true

launch_qwen7b_rm32b_deployment_mode "$@"
