#!/usr/bin/env bash
# Mode 5: ITL elastic scaling on a 24-GPU pool shared by rollout and RM.
# Usage: bash mode5_elastic_rl.sh [smoke_test=0] [extra hydra overrides...]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common_deployment.sh"

export PSRL_DEPLOY_MODE=elastic_rl
export PSRL_DEPLOY_EXPERIMENT=test_new_request_level_candidate_evaluation_mode5_bs_128_share_24_elastic_qwen7b_rm32b
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
export PSRL_DEPLOY_RM_ASYNC=False
export PSRL_DEPLOY_NNODES=4
export PSRL_DEPLOY_TRAIN_NNODES=1
export PSRL_DEPLOY_TRAIN_NGPUS=8
export PSRL_DEPLOY_SHARED_NNODES=3
export PSRL_DEPLOY_SHARED_NGPUS="[8, 8, 8]"
export PSRL_DEPLOY_RM_NUM_REPLICAS=0
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

export PSRL_DEPLOY_EXTRA="\
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
psrl.deployment.elastic_rm.itl_policy.max_scale_instances_per_action=32 \
psrl.deployment.elastic_rm.itl_policy.router_waiting_top_t=-1 \
psrl.deployment.elastic_rm.itl_policy.role_throughput_weight_enable=true \
psrl.deployment.elastic_rm.itl_policy.role_throughput_weight_basis=request_count \
psrl.deployment.elastic_rm.itl_policy.role_throughput_weight_mode=raw \
psrl.deployment.elastic_rm.itl_policy.enable_heterogeneous_parallelism_candidates=true \
+reward_models_config.reward_models.2.routing_strategy.method=itl \
+reward_models_config.reward_models.2.routing_strategy.cost_model_path=/apdcephfs_zwfy10/share_303541817/yfzhao/psrl/psrl/trainer/config/cost_model/deepseek_r1_distill_qwen_32b.json \
+reward_models_config.reward_models.2.routing_strategy.delta_throughput_threshold=0.005 \
+reward_models_config.reward_models.2.routing_strategy.request_budget=1024 \
+reward_models_config.reward_models.2.routing_strategy.max_num_waiting_reqs_after_preemption=3 \
+reward_models_config.reward_models.2.routing_strategy.max_concurrent_seqs_per_instance=128 \
"

launch_qwen7b_rm32b_deployment_mode "$@"
