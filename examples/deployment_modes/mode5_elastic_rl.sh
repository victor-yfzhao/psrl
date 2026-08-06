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

export PSRL_DEPLOY_MODE=elastic_rl
export PSRL_DEPLOY_EXPERIMENT=test_none_request_level_candidate_evaluation_mode5_bs_128_share_16_elastic_rl
export PSRL_DEPLOY_STALENESS=${STALENESS:-2}
export PSRL_DEPLOY_RM_ASYNC=False
export PSRL_DEPLOY_NNODES=4
export PSRL_DEPLOY_TRAIN_NNODES=2
export PSRL_DEPLOY_TRAIN_NGPUS=8
export PSRL_DEPLOY_SHARED_NNODES=2
export PSRL_DEPLOY_SHARED_NGPUS="[8,8]"
export PSRL_DEPLOY_RM_NUM_REPLICAS=0
export PSRL_DEPLOY_SMOKE=${1:-0}
shift || true

# ---- Elastic auto-scaling config ----
# enable_policy=True keeps the ElasticExecutor monitor loop running; the
# scaling_policy_variant selects the ScalingPolicy implementation. Below is a
# complete, tunable elastic_rm block (modeled on examples/elastic_rm/*_fsdp.sh):
# common load thresholds + cooldown + itl_harmonic policy tuning.
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
+reward_models_config.reward_models.2.routing_strategy.method=itl \
+reward_models_config.reward_models.2.routing_strategy.cost_model_path=/apdcephfs_zwfy10/share_303541817/yfzhao/psrl/psrl/trainer/config/cost_model/qwen3_8b.json \
+reward_models_config.reward_models.2.routing_strategy.delta_throughput_threshold=0.005 \
+reward_models_config.reward_models.2.routing_strategy.request_budget=1024 \
+reward_models_config.reward_models.2.routing_strategy.max_num_waiting_reqs_after_preemption=3 \
+reward_models_config.reward_models.2.routing_strategy.max_concurrent_seqs_per_instance=128 \
"


launch_deployment_mode "$@"
