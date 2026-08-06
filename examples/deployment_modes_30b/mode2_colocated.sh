#!/usr/bin/env bash
# Mode 2 — fully colocated.
# All three roles share a single train_pool, time-multiplexed per training step:
#   rollout -> (rollout-done) -> rm -> (reward-done) -> actor trains -> sleep actor.
# One node x 8 GPUs; rollout/rm/actor all live on train_pool via NIXL sleep/wake.
#
# Requirements (enforced in PSRL_RayPPOTrainer._validate_config):
#   psrl.ps_mode in {nixl_cpu, nixl_gpu}; launch_reward_fn_async=True; staleness=0;
#   colocate_validate_and_train=False.
#
# Usage: bash mode2_colocated.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PSRL_DEPLOY_MODE=colocated
export PSRL_DEPLOY_EXPERIMENT=mode2_bs_128_colocated_test_weight_arena
export PSRL_DEPLOY_STALENESS=0
export PSRL_DEPLOY_RM_ASYNC=True
export PSRL_DEPLOY_NNODES=8
export PSRL_DEPLOY_TRAIN_NNODES=8
export PSRL_DEPLOY_TRAIN_NGPUS=8
# No separate shared pool in mode 2; all replicas ride on train_pool. These are
# unused by main_ppo's colocated branch but kept for the common arg builder.
export PSRL_DEPLOY_SHARED_NNODES=0
export PSRL_DEPLOY_SHARED_NGPUS=0
export PSRL_DEPLOY_RM_NUM_REPLICAS=0
export PSRL_DEPLOY_EXTRA="${PSRL_DEPLOY_EXTRA:-} ++psrl.nixl.weight_arena.reward_enabled=true ++psrl.nixl.weight_arena.reward_cpu_cache_pin_memory=true"
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

launch_deployment_mode "$@"
