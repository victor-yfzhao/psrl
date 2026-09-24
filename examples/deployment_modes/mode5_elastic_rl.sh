#!/usr/bin/env bash
# Mode 5 — elastic RL (current elastic auto-scaling between rollout and rm).
# rollout + RM share shared_rollout_pool with a ScalingPolicy that auto-scales the
# awake instance counts; trainer on a separate train_pool. This is the existing
# elastic_rm path (elastic_rm.enable=True + enable_policy=True).
#
# Two nodes: shared_rollout_pool (8 GPUs, rollout+rm elastic) + train_pool (8 GPUs).
#
# Usage: bash mode5_elastic_rl.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"
COST_MODEL_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)/pivotrl/trainer/config/cost_model"

export PIVOTRL_DEPLOY_MODE=elastic_rl
export PIVOTRL_DEPLOY_EXPERIMENT=mode5_migration_bs_128_share_32_elastic_rl
export PIVOTRL_DEPLOY_STALENESS=${STALENESS:-2}
export PIVOTRL_DEPLOY_RM_ASYNC=False
export PIVOTRL_DEPLOY_NNODES=8
export PIVOTRL_DEPLOY_TRAIN_NNODES=4
export PIVOTRL_DEPLOY_TRAIN_NGPUS=8
export PIVOTRL_DEPLOY_SHARED_NNODES=4
export PIVOTRL_DEPLOY_SHARED_NGPUS=8
export PIVOTRL_DEPLOY_RM_NUM_REPLICAS=0
export PIVOTRL_DEPLOY_OPTIMIZER_OFFLOAD=${PIVOTRL_DEPLOY_OPTIMIZER_OFFLOAD:-True}
export PIVOTRL_DEPLOY_SMOKE=${1:-0}
export PIVOTRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION=${PIVOTRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION:-true}
export PIVOTRL_CANDIDATE_EVALUATION_BACKEND=${PIVOTRL_CANDIDATE_EVALUATION_BACKEND:-cpp}
export PIVOTRL_CANDIDATE_EVALUATION_MAX_WORKERS=${PIVOTRL_CANDIDATE_EVALUATION_MAX_WORKERS:-64}
export PIVOTRL_CANDIDATE_EVALUATION_CPP_BINARY=${PIVOTRL_CANDIDATE_EVALUATION_CPP_BINARY:-null}
export PIVOTRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S=${PIVOTRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S:-30.0}
shift || true

# ---- Elastic auto-scaling config ----
# enable_policy=True keeps the ElasticExecutor monitor loop running; the
# scaling_policy_variant selects the ScalingPolicy implementation. Below is a
# complete, tunable elastic_rm block (modeled on examples/elastic_rm/*_fsdp.sh):
# common load thresholds + cooldown + itl_harmonic policy tuning.
export PIVOTRL_DEPLOY_EXTRA="\
pivotrl.deployment.elastic_rm.enable_policy=True \
pivotrl.deployment.elastic_rm.scaling_policy_variant=itl_harmonic \
pivotrl.deployment.elastic_rm.enable_trainer_pool=True \
pivotrl.deployment.elastic_rm.min_awake_per_role=0 \
pivotrl.deployment.elastic_rm.cooldown_ms=10000 \
pivotrl.deployment.elastic_rm.hysteresis=0.05 \
pivotrl.deployment.elastic_rm.monitor_interval_ms=1000 \
pivotrl.deployment.elastic_rm.wakeup_immunity_ms=10000 \
pivotrl.deployment.elastic_rm.itl_policy.decision_window_s=60.0 \
pivotrl.deployment.elastic_rm.itl_policy.throughput_objective=sum \
pivotrl.deployment.elastic_rm.itl_policy.max_scale_instances_per_action=-1 \
pivotrl.deployment.elastic_rm.itl_policy.router_waiting_top_t=-1 \
pivotrl.deployment.elastic_rm.itl_policy.role_throughput_weight_enable=true \
pivotrl.deployment.elastic_rm.itl_policy.role_throughput_weight_basis=request_count \
pivotrl.deployment.elastic_rm.itl_policy.role_throughput_weight_mode=raw \
pivotrl.deployment.elastic_rm.itl_policy.enable_request_level_candidate_evaluation=${PIVOTRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION} \
pivotrl.deployment.elastic_rm.itl_policy.candidate_evaluation_backend=${PIVOTRL_CANDIDATE_EVALUATION_BACKEND} \
pivotrl.deployment.elastic_rm.itl_policy.candidate_evaluation_max_workers=${PIVOTRL_CANDIDATE_EVALUATION_MAX_WORKERS} \
pivotrl.deployment.elastic_rm.itl_policy.candidate_evaluation_cpp_binary=${PIVOTRL_CANDIDATE_EVALUATION_CPP_BINARY} \
pivotrl.deployment.elastic_rm.itl_policy.candidate_evaluation_cpp_timeout_s=${PIVOTRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S} \
pivotrl.deployment.elastic_rm.itl_policy.exclusive_rebalance_migration_queue=True \
+reward_models_config.reward_models.2.routing_strategy.method=itl \
+reward_models_config.reward_models.2.routing_strategy.cost_model_path=${COST_MODEL_DIR}/glm_z1_9b_0414.json \
+reward_models_config.reward_models.2.routing_strategy.delta_throughput_threshold=0.005 \
+reward_models_config.reward_models.2.routing_strategy.request_budget=1024 \
+reward_models_config.reward_models.2.routing_strategy.max_num_waiting_reqs_after_preemption=3 \
+reward_models_config.reward_models.2.routing_strategy.max_concurrent_seqs_per_instance=128 \
"


launch_deployment_mode "$@"
