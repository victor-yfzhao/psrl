#!/usr/bin/env bash
# Mode 5: ITL elastic scaling on a 24-GPU pool shared by rollout and RM.
# Usage: bash mode5_elastic_rl.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PSRL_DEPLOY_MODE=elastic_rl
export PSRL_DEPLOY_EXPERIMENT=mode5_bs_128_share_48_elastic_staleness_2_migration
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
export PSRL_DEPLOY_RM_ASYNC=False
export PSRL_DEPLOY_NNODES=8
export PSRL_DEPLOY_TRAIN_NNODES=2
export PSRL_DEPLOY_TRAIN_NGPUS=8
export PSRL_DEPLOY_SHARED_NNODES=6
export PSRL_DEPLOY_SHARED_NGPUS=8
export PSRL_DEPLOY_RM_NUM_REPLICAS=0
export PSRL_DEPLOY_SMOKE=${1:-0}
export PSRL_DEPLOY_DATASET=${PSRL_DEPLOY_DATASET:-mixed}
export PSRL_NEED_VALIDATION=${PSRL_NEED_VALIDATION:-0}
export PSRL_DEPLOY_OPTIMIZER_OFFLOAD=${PSRL_DEPLOY_OPTIMIZER_OFFLOAD:-True}
export PSRL_DEPLOY_ROLLOUT_RS=${PSRL_DEPLOY_ROLLOUT_RS:-geometric}
# export PSRL_DEPLOY_ROLLOUT_RS=${PSRL_DEPLOY_ROLLOUT_RS:-null}
export PSRL_DEPLOY_ROLLOUT_RS_THRESHOLD=${PSRL_DEPLOY_ROLLOUT_RS_THRESHOLD:-1.005}
export PSRL_WEIGHT_VERIFY=${PSRL_WEIGHT_VERIFY:-False}
export PSRL_WEIGHT_VERIFY_SAMPLE_EVERY=${PSRL_WEIGHT_VERIFY_SAMPLE_EVERY:-1}
export PSRL_WEIGHT_VERIFY_FULL_EVERY=${PSRL_WEIGHT_VERIFY_FULL_EVERY:-0}
export PSRL_WEIGHT_VERIFY_SAMPLE_COUNT=${PSRL_WEIGHT_VERIFY_SAMPLE_COUNT:-16}
export PSRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION=${PSRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION:-true}
export PSRL_CANDIDATE_EVALUATION_BACKEND=${PSRL_CANDIDATE_EVALUATION_BACKEND:-cpp}
export PSRL_CANDIDATE_EVALUATION_MAX_WORKERS=${PSRL_CANDIDATE_EVALUATION_MAX_WORKERS:-64}
export PSRL_CANDIDATE_EVALUATION_CPP_BINARY=${PSRL_CANDIDATE_EVALUATION_CPP_BINARY:-null}
export PSRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S=${PSRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S:-30.0}
if [[ "${PSRL_NEED_VALIDATION}" == "1" ]]; then
    export PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN=True
    export PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE=False
    export PSRL_DEPLOY_VAL_GPU_MEMORY_UTILIZATION=${PSRL_DEPLOY_VAL_GPU_MEMORY_UTILIZATION:-0.5}
    export PSRL_DEPLOY_VAL_BEFORE_TRAIN=False
else
    export PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN=False
    export PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE=True
    export PSRL_DEPLOY_VAL_BEFORE_TRAIN=False
fi
shift || true

export PSRL_DEPLOY_EXTRA="\
psrl.nixl.weight_verification.enable=${PSRL_WEIGHT_VERIFY} \
psrl.nixl.weight_verification.transfer_chain=True \
psrl.nixl.weight_verification.trainer_sleep_wake=True \
psrl.nixl.weight_verification.sample_every_n_versions=${PSRL_WEIGHT_VERIFY_SAMPLE_EVERY} \
psrl.nixl.weight_verification.full_every_n_versions=${PSRL_WEIGHT_VERIFY_FULL_EVERY} \
psrl.nixl.weight_verification.sample_count_per_tensor=${PSRL_WEIGHT_VERIFY_SAMPLE_COUNT} \
psrl.deployment.elastic_rm.enable_policy=True \
psrl.deployment.elastic_rm.scaling_policy_variant=itl_harmonic \
psrl.deployment.elastic_rm.enable_trainer_pool=True \
psrl.deployment.elastic_rm.min_awake_per_role=0 \
psrl.deployment.elastic_rm.cooldown_ms=10000 \
psrl.deployment.elastic_rm.hysteresis=0.05 \
psrl.deployment.elastic_rm.monitor_interval_ms=1000 \
psrl.deployment.elastic_rm.wakeup_immunity_ms=10000 \
psrl.deployment.elastic_rm.itl_policy.decision_window_s=60.0 \
psrl.deployment.elastic_rm.itl_policy.throughput_objective=sum \
psrl.deployment.elastic_rm.itl_policy.max_scale_instances_per_action=-1 \
psrl.deployment.elastic_rm.itl_policy.router_waiting_top_t=-1 \
psrl.deployment.elastic_rm.itl_policy.role_throughput_weight_enable=true \
psrl.deployment.elastic_rm.itl_policy.role_throughput_weight_basis=request_count \
psrl.deployment.elastic_rm.itl_policy.role_throughput_weight_mode=raw \
psrl.deployment.elastic_rm.itl_policy.enable_heterogeneous_parallelism_candidates=true \
psrl.deployment.elastic_rm.itl_policy.enable_request_level_candidate_evaluation=${PSRL_ENABLE_REQUEST_LEVEL_CANDIDATE_EVALUATION} \
psrl.deployment.elastic_rm.itl_policy.candidate_evaluation_backend=${PSRL_CANDIDATE_EVALUATION_BACKEND} \
psrl.deployment.elastic_rm.itl_policy.candidate_evaluation_max_workers=${PSRL_CANDIDATE_EVALUATION_MAX_WORKERS} \
psrl.deployment.elastic_rm.itl_policy.candidate_evaluation_cpp_binary=${PSRL_CANDIDATE_EVALUATION_CPP_BINARY} \
psrl.deployment.elastic_rm.itl_policy.candidate_evaluation_cpp_timeout_s=${PSRL_CANDIDATE_EVALUATION_CPP_TIMEOUT_S} \
psrl.deployment.elastic_rm.itl_policy.exclusive_rebalance_migration_queue=True \
algorithm.rollout_correction.rollout_rs=${PSRL_DEPLOY_ROLLOUT_RS} \
algorithm.rollout_correction.rollout_rs_threshold=${PSRL_DEPLOY_ROLLOUT_RS_THRESHOLD} \
+reward_models_config.reward_models.2.routing_strategy.method=itl \
+reward_models_config.reward_models.2.routing_strategy.cost_model_path=/apdcephfs_zwfy10/share_303541817/yfzhao/psrl/psrl/trainer/config/cost_model/qwen3_30b_a3b_thinking_2507.json \
+reward_models_config.reward_models.2.routing_strategy.delta_throughput_threshold=0.005 \
+reward_models_config.reward_models.2.routing_strategy.request_budget=1024 \
+reward_models_config.reward_models.2.routing_strategy.max_num_waiting_reqs_after_preemption=3 \
+reward_models_config.reward_models.2.routing_strategy.max_concurrent_seqs_per_instance=128 \
"

launch_qwen7b_rm32b_deployment_mode "$@"
