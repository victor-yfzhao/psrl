#!/usr/bin/env bash
# Mode 3 — rollout / reward-model colocated.
# rollout + RM share shared_rollout_pool, time-multiplexed per buffer:
#   rollout -> (rollout-done) -> rm -> (reward-done) -> rollout (next buffer).
# Trainer lives on a separate train_pool and pipelines via staleness.
# Two nodes: shared_rollout_pool (8 GPUs, rollout+rm) + train_pool (8 GPUs, actor).
#
# Requirements: psrl.ps_mode in {nixl_cpu, nixl_gpu}; launch_reward_fn_async=True;
#   staleness=0.
#
# Usage: bash mode3_rollout_rm_colocated.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PSRL_DEPLOY_MODE=rollout_rm_colocated
export PSRL_DEPLOY_EXPERIMENT=mode3_bs_128_rollout_rm_colocated_rollout7b_rm8b
export PSRL_DEPLOY_STALENESS=2
export PSRL_DEPLOY_RM_ASYNC=True
export PSRL_DEPLOY_NNODES=8
export PSRL_DEPLOY_TRAIN_NNODES=4
export PSRL_DEPLOY_TRAIN_NGPUS=8
export PSRL_DEPLOY_SHARED_NNODES=4
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_RM_NUM_REPLICAS=0
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

launch_deployment_mode "$@"
