#!/usr/bin/env bash
# Mode 5: ITL elastic scaling on a 24-GPU pool shared by rollout and RM.
# Usage: bash mode5_elastic_rl.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PIVOTRL_DEPLOY_MODE=elastic_rl
export PIVOTRL_DEPLOY_EXPERIMENT=mode5_bs_128_share_48_elastic_staleness_2_migration
export PIVOTRL_DEPLOY_STALENESS=${STALENESS:-2}
export PIVOTRL_DEPLOY_RM_ASYNC=False
export PIVOTRL_DEPLOY_NNODES=8
export PIVOTRL_DEPLOY_TRAIN_NNODES=2
export PIVOTRL_DEPLOY_TRAIN_NGPUS=8
export PIVOTRL_DEPLOY_SHARED_NNODES=6
export PIVOTRL_DEPLOY_SHARED_NGPUS=8
export PIVOTRL_DEPLOY_RM_NUM_REPLICAS=0
export PIVOTRL_DEPLOY_SMOKE=${1:-0}
export PIVOTRL_DEPLOY_DATASET=${PIVOTRL_DEPLOY_DATASET:-mixed}
export PIVOTRL_NEED_VALIDATION=${PIVOTRL_NEED_VALIDATION:-0}
export PIVOTRL_DEPLOY_OPTIMIZER_OFFLOAD=${PIVOTRL_DEPLOY_OPTIMIZER_OFFLOAD:-True}
export PIVOTRL_DEPLOY_ROLLOUT_RS=${PIVOTRL_DEPLOY_ROLLOUT_RS:-geometric}
# export PIVOTRL_DEPLOY_ROLLOUT_RS=${PIVOTRL_DEPLOY_ROLLOUT_RS:-null}
export PIVOTRL_DEPLOY_ROLLOUT_RS_THRESHOLD=${PIVOTRL_DEPLOY_ROLLOUT_RS_THRESHOLD:-1.005}
export PIVOTRL_WEIGHT_VERIFY=${PIVOTRL_WEIGHT_VERIFY:-False}
export PIVOTRL_WEIGHT_VERIFY_SAMPLE_EVERY=${PIVOTRL_WEIGHT_VERIFY_SAMPLE_EVERY:-1}
export PIVOTRL_WEIGHT_VERIFY_FULL_EVERY=${PIVOTRL_WEIGHT_VERIFY_FULL_EVERY:-0}
export PIVOTRL_WEIGHT_VERIFY_SAMPLE_COUNT=${PIVOTRL_WEIGHT_VERIFY_SAMPLE_COUNT:-16}
export PIVOTRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION=${PIVOTRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION:-true}
export PIVOTRL_CANDIDATE_EVALUATION_BACKEND=${PIVOTRL_CANDIDATE_EVALUATION_BACKEND:-cpp}
export PIVOTRL_CANDIDATE_EVALUATION_MAX_WORKERS=${PIVOTRL_CANDIDATE_EVALUATION_MAX_WORKERS:-64}
export PIVOTRL_CANDIDATE_EVALUATION_CPP_BINARY=${PIVOTRL_CANDIDATE_EVALUATION_CPP_BINARY:-null}
export PIVOTRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S=${PIVOTRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S:-30.0}
if [[ "${PIVOTRL_NEED_VALIDATION}" == "1" ]]; then
    export PIVOTRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN=True
    export PIVOTRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE=False
    export PIVOTRL_DEPLOY_VAL_GPU_MEMORY_UTILIZATION=${PIVOTRL_DEPLOY_VAL_GPU_MEMORY_UTILIZATION:-0.5}
    export PIVOTRL_DEPLOY_VAL_BEFORE_TRAIN=False
else
    export PIVOTRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN=False
    export PIVOTRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE=True
    export PIVOTRL_DEPLOY_VAL_BEFORE_TRAIN=False
fi
shift || true

export PIVOTRL_DEPLOY_EXTRA="\
pivotrl.nixl.weight_verification.enable=${PIVOTRL_WEIGHT_VERIFY} \
pivotrl.nixl.weight_verification.transfer_chain=True \
pivotrl.nixl.weight_verification.trainer_sleep_wake=True \
pivotrl.nixl.weight_verification.sample_every_n_versions=${PIVOTRL_WEIGHT_VERIFY_SAMPLE_EVERY} \
pivotrl.nixl.weight_verification.full_every_n_versions=${PIVOTRL_WEIGHT_VERIFY_FULL_EVERY} \
pivotrl.nixl.weight_verification.sample_count_per_tensor=${PIVOTRL_WEIGHT_VERIFY_SAMPLE_COUNT} \
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
pivotrl.deployment.elastic_rm.itl_policy.enable_heterogeneous_parallelism_candidates=true \
pivotrl.deployment.elastic_rm.itl_policy.enable_request_level_candidate_evaluation=${PIVOTRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION} \
pivotrl.deployment.elastic_rm.itl_policy.candidate_evaluation_backend=${PIVOTRL_CANDIDATE_EVALUATION_BACKEND} \
pivotrl.deployment.elastic_rm.itl_policy.candidate_evaluation_max_workers=${PIVOTRL_CANDIDATE_EVALUATION_MAX_WORKERS} \
pivotrl.deployment.elastic_rm.itl_policy.candidate_evaluation_cpp_binary=${PIVOTRL_CANDIDATE_EVALUATION_CPP_BINARY} \
pivotrl.deployment.elastic_rm.itl_policy.candidate_evaluation_cpp_timeout_s=${PIVOTRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S} \
pivotrl.deployment.elastic_rm.itl_policy.exclusive_rebalance_migration_queue=True \
algorithm.rollout_correction.rollout_rs=${PIVOTRL_DEPLOY_ROLLOUT_RS} \
algorithm.rollout_correction.rollout_rs_threshold=${PIVOTRL_DEPLOY_ROLLOUT_RS_THRESHOLD} \
+reward_models_config.reward_models.2.routing_strategy.method=itl \
+reward_models_config.reward_models.2.routing_strategy.cost_model_path=/apdcephfs_zwfy10/share_303541817/yfzhao/pivotrl/pivotrl/trainer/config/cost_model/qwen3_30b_a3b_thinking_2507.json \
+reward_models_config.reward_models.2.routing_strategy.delta_throughput_threshold=0.005 \
+reward_models_config.reward_models.2.routing_strategy.request_budget=1024 \
+reward_models_config.reward_models.2.routing_strategy.max_num_waiting_reqs_after_preemption=3 \
+reward_models_config.reward_models.2.routing_strategy.max_concurrent_seqs_per_instance=128 \
"

launch_qwen7b_rm32b_deployment_mode "$@"
