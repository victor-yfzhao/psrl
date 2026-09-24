#!/usr/bin/env bash
# Shared launch body for the five deployment-mode example scripts.
#
# Each mode script (mode1_disaggregated.sh ... mode5_elastic_rl.sh) exports a
# small set of PIVOTRL_DEPLOY_* env vars describing its pool topology / mode flags,
# then sources this file and calls `launch_deployment_mode "$@"`. Extra hydra
# overrides can be passed as "$@" from the mode script.
#
# Required env vars (set by the mode script before calling):
#   PIVOTRL_DEPLOY_MODE          one of disaggregated|colocated|rollout_rm_colocated|
#                             trainer_pool_only|elastic_rl
#   PIVOTRL_DEPLOY_EXPERIMENT    experiment name string
#   PIVOTRL_DEPLOY_STALENESS     staleness (modes 2/3 require 0)
#   PIVOTRL_DEPLOY_RM_ASYNC      True|False (modes 2/3 require True)
#   PIVOTRL_DEPLOY_NNODES        total cluster nodes
#   PIVOTRL_DEPLOY_TRAIN_NNODES  train_pool nodes
#   PIVOTRL_DEPLOY_TRAIN_NGPUS   gpus per train node
#   PIVOTRL_DEPLOY_SHARED_NNODES   shared_rollout_pool nodes (0/unused for mode 2)
#   PIVOTRL_DEPLOY_SHARED_NGPUS    gpus per shared node, or a Hydra list like [4, 8]
#   PIVOTRL_DEPLOY_RM_NUM_REPLICAS  gen-RM num_replicas (non-elastic modes 1/4)
#                                set to 0 to leave default
#   PIVOTRL_DEPLOY_SMOKE         0|1 (smoke test: 2 steps, small bsz)
#   PIVOTRL_DEPLOY_DATASET       dapo|gsm8k|mixed (default: dapo; mixed is 1:1)
#   PIVOTRL_DEPLOY_TRAIN_BACKEND fsdp2|megatron (default: fsdp2)
#   PIVOTRL_DEPLOY_MEGATRON_TP   Megatron tensor parallel size
#   PIVOTRL_DEPLOY_MEGATRON_PP   Megatron pipeline parallel size
#   PIVOTRL_DEPLOY_MEGATRON_CP   Megatron context parallel size
#   PIVOTRL_DEPLOY_MEGATRON_EP   Megatron expert parallel size
#   PIVOTRL_DEPLOY_MEGATRON_ETP  Megatron expert tensor parallel size
#   PIVOTRL_DEPLOY_EXTRA         space-separated extra hydra overrides appended last
#
# Optional positional args to launch_deployment_mode are forwarded to main_ppo.

_pivotrl_total_gpus_from_node_spec() {
    local ngpus_spec=$1
    local nnodes=$2
    local total=0
    local value

    if [[ "${ngpus_spec}" =~ ^[0-9]+$ ]]; then
        echo $(( ngpus_spec * nnodes ))
        return
    fi

    if [[ "${ngpus_spec}" =~ ^\[(.*)\]$ ]]; then
        local values=${BASH_REMATCH[1]}
        values=${values// /}
        IFS=',' read -ra _pivotrl_gpu_values <<< "${values}"
        for value in "${_pivotrl_gpu_values[@]}"; do
            if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
                echo "Invalid GPU count '${value}' in PIVOTRL_DEPLOY_SHARED_NGPUS=${ngpus_spec}" >&2
                return 1
            fi
            total=$(( total + value ))
        done
        if [[ "${#_pivotrl_gpu_values[@]}" -ne "${nnodes}" ]]; then
            echo "PIVOTRL_DEPLOY_SHARED_NGPUS=${ngpus_spec} has ${#_pivotrl_gpu_values[@]} entries, expected PIVOTRL_DEPLOY_SHARED_NNODES=${nnodes}" >&2
            return 1
        fi
        echo "${total}"
        return
    fi

    echo "Invalid PIVOTRL_DEPLOY_SHARED_NGPUS=${ngpus_spec}; expected integer or Hydra list like [4, 8]" >&2
    return 1
}

_pivotrl_vllm_ep_size_for_model() {
    local model_path=$1
    local tp_size=$2

    python - "${model_path}" "${tp_size}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]) / "config.json"
tp_size = int(sys.argv[2])

with config_path.open(encoding="utf-8") as config_file:
    config = json.load(config_file)

expert_count_keys = {
    "moe_num_experts",
    "n_experts",
    "n_routed_experts",
    "num_experts",
    "num_local_experts",
    "num_routed_experts",
}


def is_moe(value):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = key.lower()
            if normalized_key in expert_count_keys and isinstance(child, (int, float)) and child > 1:
                return True
            if normalized_key == "moe_intermediate_size" and isinstance(child, (int, float)) and child > 0:
                return True
            if normalized_key in {"architectures", "model_type"} and "moe" in str(child).lower():
                return True
            if is_moe(child):
                return True
    elif isinstance(value, list):
        return any(is_moe(child) for child in value)
    return False


print(tp_size if is_moe(config) else 1)
PY
}

launch_deployment_mode() {
    set -xeuo pipefail

    : "${PIVOTRL_DEPLOY_MODE:?PIVOTRL_DEPLOY_MODE must be set by the mode script}"
    : "${PIVOTRL_DEPLOY_EXPERIMENT:?PIVOTRL_DEPLOY_EXPERIMENT must be set}"
    : "${PIVOTRL_DEPLOY_STALENESS:?PIVOTRL_DEPLOY_STALENESS must be set}"
    : "${PIVOTRL_DEPLOY_RM_ASYNC:?PIVOTRL_DEPLOY_RM_ASYNC must be set}"
    : "${PIVOTRL_DEPLOY_NNODES:?PIVOTRL_DEPLOY_NNODES must be set}"
    : "${PIVOTRL_DEPLOY_TRAIN_NNODES:?PIVOTRL_DEPLOY_TRAIN_NNODES must be set}"
    : "${PIVOTRL_DEPLOY_TRAIN_NGPUS:?PIVOTRL_DEPLOY_TRAIN_NGPUS must be set}"
    : "${PIVOTRL_DEPLOY_SHARED_NNODES:?PIVOTRL_DEPLOY_SHARED_NNODES must be set}"
    : "${PIVOTRL_DEPLOY_SHARED_NGPUS:?PIVOTRL_DEPLOY_SHARED_NGPUS must be set}"
    PIVOTRL_DEPLOY_SMOKE=${PIVOTRL_DEPLOY_SMOKE:-0}
    # PIVOTRL_DEPLOY_DATASET=${PIVOTRL_DEPLOY_DATASET:-gsm8k}
    # PIVOTRL_DEPLOY_DATASET=${PIVOTRL_DEPLOY_DATASET:-dapo}
    PIVOTRL_DEPLOY_DATASET=${PIVOTRL_DEPLOY_DATASET:-mixed}
    PIVOTRL_DEPLOY_RM_NUM_REPLICAS=${PIVOTRL_DEPLOY_RM_NUM_REPLICAS:-0}
    PIVOTRL_DEPLOY_TRAINER_READY_GRACE_S=${PIVOTRL_DEPLOY_TRAINER_READY_GRACE_S:-5}
    PIVOTRL_DEPLOY_EXTRA=${PIVOTRL_DEPLOY_EXTRA:-}
    PIVOTRL_DEPLOY_TRAIN_BACKEND=${PIVOTRL_DEPLOY_TRAIN_BACKEND:-fsdp2}
    PIVOTRL_DEPLOY_MEGATRON_TP=${PIVOTRL_DEPLOY_MEGATRON_TP:-2}
    PIVOTRL_DEPLOY_MEGATRON_PP=${PIVOTRL_DEPLOY_MEGATRON_PP:-2}
    PIVOTRL_DEPLOY_MEGATRON_CP=${PIVOTRL_DEPLOY_MEGATRON_CP:-1}
    PIVOTRL_DEPLOY_MEGATRON_EP=${PIVOTRL_DEPLOY_MEGATRON_EP:-1}
    PIVOTRL_DEPLOY_MEGATRON_ETP=${PIVOTRL_DEPLOY_MEGATRON_ETP:-1}
    PIVOTRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU=${PIVOTRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU:-1}
    PIVOTRL_DEPLOY_MEGATRON_USE_MBRIDGE=${PIVOTRL_DEPLOY_MEGATRON_USE_MBRIDGE:-True}
    PIVOTRL_DEPLOY_MEGATRON_PARAM_OFFLOAD=${PIVOTRL_DEPLOY_MEGATRON_PARAM_OFFLOAD:-False}
    PIVOTRL_DEPLOY_MEGATRON_GRAD_OFFLOAD=${PIVOTRL_DEPLOY_MEGATRON_GRAD_OFFLOAD:-True}
    PIVOTRL_DEPLOY_MEGATRON_OPTIMIZER_OFFLOAD=${PIVOTRL_DEPLOY_MEGATRON_OPTIMIZER_OFFLOAD:-True}
    PIVOTRL_DEPLOY_MEGATRON_RECOMPUTE_NUM_LAYERS=${PIVOTRL_DEPLOY_MEGATRON_RECOMPUTE_NUM_LAYERS:-1}
    PIVOTRL_DEPLOY_MEGATRON_MOE_AUX_LOSS_COEFF=${PIVOTRL_DEPLOY_MEGATRON_MOE_AUX_LOSS_COEFF:-0.01}
    PIVOTRL_DEPLOY_MEGATRON_MOE_Z_LOSS_COEFF=${PIVOTRL_DEPLOY_MEGATRON_MOE_Z_LOSS_COEFF:-0.001}
    if [[ -z "${PIVOTRL_DEPLOY_OPTIMIZER_OFFLOAD+x}" ]]; then
        case "${PIVOTRL_DEPLOY_MODE}" in
            colocated|trainer_pool_only)
                PIVOTRL_DEPLOY_OPTIMIZER_OFFLOAD=True
                ;;
            *)
                PIVOTRL_DEPLOY_OPTIMIZER_OFFLOAD=False
                ;;
        esac
    fi

    : "${PIVOTRL_WORKSPACE:?PIVOTRL_WORKSPACE must be set}"

    if [[ "${PIVOTRL_DEPLOY_SMOKE}" == "1" ]]; then
        experiment_suffix="_smoke"
        total_training_steps=50
        test_freq=-1
        save_freq=-1
        train_prompt_bsz=32
        train_prompt_mini_bsz=8
        n_resp_per_prompt=8
    else
        experiment_suffix=""
        total_training_steps=200
        test_freq=200
        save_freq=200
        train_prompt_bsz=128
        train_prompt_mini_bsz=32
        n_resp_per_prompt=16
    fi

    case "${PIVOTRL_DEPLOY_TRAIN_BACKEND}" in
        fsdp|fsdp2)
            TRAINER_CONFIG_NAME=ppo_trainer
            train_backend_suffix=""
            ;;
        megatron)
            TRAINER_CONFIG_NAME=ppo_megatron_trainer
            train_backend_suffix="_megatron"
            export CUDA_DEVICE_MAX_CONNECTIONS=1

            for value in \
                "${PIVOTRL_DEPLOY_MEGATRON_TP}" \
                "${PIVOTRL_DEPLOY_MEGATRON_PP}" \
                "${PIVOTRL_DEPLOY_MEGATRON_CP}" \
                "${PIVOTRL_DEPLOY_MEGATRON_EP}" \
                "${PIVOTRL_DEPLOY_MEGATRON_ETP}" \
                "${PIVOTRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU}"; do
                if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
                    echo "Megatron parallel and micro-batch sizes must be positive integers, got '${value}'" >&2
                    return 1
                fi
            done
            ;;
        *)
            echo "Invalid PIVOTRL_DEPLOY_TRAIN_BACKEND=${PIVOTRL_DEPLOY_TRAIN_BACKEND}; expected fsdp2 or megatron" >&2
            return 1
            ;;
    esac

    project_name='verl_deployment_modes'
    MODEL_NAME='Qwen2.5-72B'
    # RM_MODEL_NAME='DeepSeek-R1-Distill-Qwen-32B'
    # RM_MODEL_NAME='Qwen3-Next-80B-A3B-Thinking'
    RM_MODEL_NAME="Qwen3.5-122B-A10B"
    # RM_MODEL_NAME='Qwen3-8B'
    experiment_name="${PIVOTRL_DEPLOY_DATASET}_${PIVOTRL_DEPLOY_EXPERIMENT}_${MODEL_NAME}_${RM_MODEL_NAME}${train_backend_suffix}${experiment_suffix}"

    source ${PIVOTRL_WORKSPACE}/env/env_311.sh

    HOME=${PIVOTRL_WORKSPACE}
    PIVOTRL_PATH=$(python -c "import pivotrl; import os; print(os.path.dirname(os.path.dirname(pivotrl.__file__)))")
    
    HF_MODEL_PATH=${PIVOTRL_WORKSPACE}/models/${MODEL_NAME}
    RM_MODEL_PATH=${PIVOTRL_WORKSPACE}/models/${RM_MODEL_NAME}

    # Data config (Hydra / OmegaConf), aligned with pivotrl/trainer/config/data/multi_datasets.yaml
    GSM8K_TRAIN="${PIVOTRL_WORKSPACE}/data/gsm8k_verl/train.parquet"
    GSM8K_TEST="${PIVOTRL_WORKSPACE}/data/gsm8k_verl/test.parquet"
    DAPO_TRAIN="${PIVOTRL_WORKSPACE}/data/dapo/dapo-math-17k.parquet"
    DAPO_VAL="${PIVOTRL_WORKSPACE}/data/dapo/aime-2024.parquet"

    _DS_NAIVE="reward_fn_key:data_source,reward_loop_type:naive,reward_fn:default,reward_model_name:null,reward_coef:1.0"
    _DS_DAPO="reward_fn_key:data_source,reward_loop_type:dapo,reward_fn:default,reward_model_name:null,reward_coef:1.0"
    _DS_GEN="reward_fn_key:data_source,reward_loop_type:gen,reward_fn:default,reward_model_name:${RM_MODEL_NAME},reward_coef:0.001"

    _ROW_TRAIN_GSM8K="{file:${GSM8K_TRAIN},data_source_name:openai/gsm8k,prompt_key:prompt,reward_model_dicts:[{${_DS_NAIVE}},{${_DS_GEN}}]}"
    _ROW_TRAIN_DAPO="{file:${DAPO_TRAIN},data_source_name:dapo/dapo-math-17k,prompt_key:prompt,reward_model_dicts:[{${_DS_DAPO}},{${_DS_GEN}}]}"

    _ROW_VAL_GSM8K="{file:${GSM8K_TEST},prompt_key:prompt,reward_model_dicts:[{${_DS_NAIVE}}]}"
    _ROW_VAL_DAPO="{file:${DAPO_VAL},prompt_key:prompt,reward_model_dicts:[{${_DS_DAPO}}]}"

    case "${PIVOTRL_DEPLOY_DATASET}" in
        dapo)
            TRAIN_DATAS="[${_ROW_TRAIN_DAPO}]"
            VAL_DATAS="[${_ROW_VAL_DAPO}]"
            TRAIN_DATASETS_RATIOS='[1.0]'
            ;;
        gsm8k)
            TRAIN_DATAS="[${_ROW_TRAIN_GSM8K}]"
            VAL_DATAS="[${_ROW_VAL_GSM8K}]"
            TRAIN_DATASETS_RATIOS='[1.0]'
            ;;
        mixed)
            TRAIN_DATAS="[${_ROW_TRAIN_GSM8K},${_ROW_TRAIN_DAPO}]"
            VAL_DATAS="[${_ROW_VAL_GSM8K},${_ROW_VAL_DAPO}]"
            TRAIN_DATASETS_RATIOS='[0.5,0.5]'
            ;;
        *)
            echo "Invalid PIVOTRL_DEPLOY_DATASET=${PIVOTRL_DEPLOY_DATASET}; expected dapo, gsm8k, or mixed" >&2
            return 1
            ;;
    esac

    # rollout settings
    GEN_TP=8
    GEN_PP=1
    GEN_EP=$(_pivotrl_vllm_ep_size_for_model "${HF_MODEL_PATH}" "${GEN_TP}")
    GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} ))

    # reward-model rollout settings
    RM_TP=8
    RM_PP=1
    RM_EP=$(_pivotrl_vllm_ep_size_for_model "${RM_MODEL_PATH}" "${RM_TP}")
    RM_NGPUS_PER_NODE_PER_INSTANCE=$(( ${RM_TP} * ${RM_PP} ))

    # validation settings (on train_pool; does not use elastic path)
    VAL_TP=8
    VAL_PP=1
    VAL_EP=$(_pivotrl_vllm_ep_size_for_model "${HF_MODEL_PATH}" "${VAL_TP}")
    VAL_INSTANCES=$(( (${PIVOTRL_DEPLOY_TRAIN_NNODES} * ${PIVOTRL_DEPLOY_TRAIN_NGPUS}) / ( ${VAL_TP} * ${VAL_PP} ) ))
    VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( ${VAL_TP} * ${VAL_PP} ))

    sp_size=1
    fsdp_size=${PIVOTRL_DEPLOY_TRAIN_NGPUS}
    use_dynamic_bsz=True

    adv_estimator=grpo
    use_kl_in_reward=False
    kl_coef=0.0
    use_kl_loss=False
    kl_loss_coef=0.0
    clip_ratio_low=0.2
    clip_ratio_high=0.28
    max_prompt_length=$((1024 * 1))
    max_response_length=$((1024 * 15))
    max_num_batched_tokens=$((1024 * 16))
    packing_length=$(((max_prompt_length + max_response_length) * 2))
    enable_overlong_buffer=True
    overlong_buffer_len=$((1024 * 5))
    overlong_penalty_factor=1.0
    loss_agg_mode="token-mean"


    temperature=1.0
    top_p=1.0
    top_k=-1
    val_top_p=0.7
    filter_groups_metric=acc

    rollout_is="token"
    rollout_is_threshold=2.0

    offload=${PIVOTRL_DEPLOY_OPTIMIZER_OFFLOAD}

    case "${PIVOTRL_DEPLOY_TRAIN_BACKEND}" in
        fsdp|fsdp2)
            TRAIN_BACKEND_ARGS=(
                train_actor_rollout_ref.model.use_remove_padding=True
                train_actor_rollout_ref.model.enable_gradient_checkpointing=True
                train_actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
                train_actor_rollout_ref.actor.fsdp_config.param_offload=False
                train_actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}
                train_actor_rollout_ref.actor.grad_clip=1.0
                train_actor_rollout_ref.model.use_shm=False
            )
            ;;
        megatron)
            TRAIN_BACKEND_ARGS=(
                train_actor_rollout_ref.model.use_remove_padding=False
                train_actor_rollout_ref.model.use_shm=False
                train_actor_rollout_ref.actor.use_dynamic_bsz=False
                train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PIVOTRL_DEPLOY_MEGATRON_MICRO_BSZ_PER_GPU}
                train_actor_rollout_ref.actor.megatron.param_offload=${PIVOTRL_DEPLOY_MEGATRON_PARAM_OFFLOAD}
                train_actor_rollout_ref.actor.megatron.grad_offload=${PIVOTRL_DEPLOY_MEGATRON_GRAD_OFFLOAD}
                train_actor_rollout_ref.actor.megatron.optimizer_offload=${PIVOTRL_DEPLOY_MEGATRON_OPTIMIZER_OFFLOAD}
                train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${PIVOTRL_DEPLOY_MEGATRON_TP}
                train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PIVOTRL_DEPLOY_MEGATRON_PP}
                train_actor_rollout_ref.actor.megatron.context_parallel_size=${PIVOTRL_DEPLOY_MEGATRON_CP}
                train_actor_rollout_ref.actor.megatron.expert_model_parallel_size=${PIVOTRL_DEPLOY_MEGATRON_EP}
                train_actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${PIVOTRL_DEPLOY_MEGATRON_ETP}
                train_actor_rollout_ref.actor.megatron.use_mbridge=${PIVOTRL_DEPLOY_MEGATRON_USE_MBRIDGE}
                train_actor_rollout_ref.actor.megatron.use_remove_padding=False
                train_actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
                train_actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
                train_actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=${PIVOTRL_DEPLOY_MEGATRON_RECOMPUTE_NUM_LAYERS}
                +train_actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
                +train_actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
                +train_actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=${PIVOTRL_DEPLOY_MEGATRON_MOE_AUX_LOSS_COEFF}
                +train_actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=${PIVOTRL_DEPLOY_MEGATRON_MOE_Z_LOSS_COEFF}
                train_actor_rollout_ref.ref.megatron.param_offload=${PIVOTRL_DEPLOY_MEGATRON_PARAM_OFFLOAD}
                train_actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${PIVOTRL_DEPLOY_MEGATRON_TP}
                train_actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PIVOTRL_DEPLOY_MEGATRON_PP}
                train_actor_rollout_ref.ref.megatron.context_parallel_size=${PIVOTRL_DEPLOY_MEGATRON_CP}
                train_actor_rollout_ref.ref.megatron.expert_model_parallel_size=${PIVOTRL_DEPLOY_MEGATRON_EP}
                train_actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${PIVOTRL_DEPLOY_MEGATRON_ETP}
            )
            ;;
    esac

    REWARD_MODELS=(
        reward_models_config.reward_normalization=none
        reward_models_config.reward_models.0.reward_loop_type=naive
        reward_models_config.reward_models.0.reward_fn='["default"]'
        reward_models_config.reward_models.1.reward_loop_type=dapo
        reward_models_config.reward_models.1.reward_fn='["default"]'
        reward_models_config.reward_models.1.reward_loop_kwargs.max_resp_len=10240
        reward_models_config.reward_models.1.reward_loop_kwargs.overlong_buffer_cfg.enable=True
        reward_models_config.reward_models.1.reward_loop_kwargs.overlong_buffer_cfg.len=5120
        reward_models_config.reward_models.1.reward_loop_kwargs.overlong_buffer_cfg.penalty_factor=1.0
        reward_models_config.reward_models.1.reward_loop_kwargs.overlong_buffer_cfg.log=False
        reward_models_config.reward_models.2.reward_loop_type=gen
        reward_models_config.reward_models.2.reward_fn='["default"]'
        reward_models_config.reward_models.2.reward_model_name=${RM_MODEL_NAME}
        reward_models_config.reward_models.2.enable_resource_pool=True
        reward_models_config.reward_models.2.rollout_ngpus_per_instance_per_node=${RM_NGPUS_PER_NODE_PER_INSTANCE}
        reward_models_config.reward_models.2.rollout_nnodes_per_instance=1
        reward_models_config.reward_models.2.max_concurrent_requests_per_instance=128
        reward_models_config.reward_models.2.model.path=${RM_MODEL_PATH}
        reward_models_config.reward_models.2.model.use_shm=False
        reward_models_config.reward_models.2.model.trust_remote_code=False
        reward_models_config.reward_models.2.rollout._target_=pivotrl.workers.config.RolloutConfig
        reward_models_config.reward_models.2.rollout.name=vllm
        reward_models_config.reward_models.2.rollout.mode=pivotrl_async
        reward_models_config.reward_models.2.rollout.disable_attn=False
        reward_models_config.reward_models.2.rollout.dtype=bfloat16
        reward_models_config.reward_models.2.rollout.gpu_memory_utilization=0.6
        reward_models_config.reward_models.2.rollout.enforce_eager=true
        reward_models_config.reward_models.2.rollout.free_cache_engine=true
        reward_models_config.reward_models.2.rollout.data_parallel_size=1
        reward_models_config.reward_models.2.rollout.expert_parallel_size=${RM_EP}
        reward_models_config.reward_models.2.rollout.tensor_model_parallel_size=${RM_TP}
        reward_models_config.reward_models.2.rollout.pipeline_model_parallel_size=${RM_PP}
        reward_models_config.reward_models.2.rollout.max_num_batched_tokens=$((1024 * 16 + 1024 * 20))
        reward_models_config.reward_models.2.rollout.max_num_seqs=512
        reward_models_config.reward_models.2.rollout.enable_chunked_prefill=false
        reward_models_config.reward_models.2.rollout.enable_prefix_caching=false
        +reward_models_config.reward_models.2.rollout.engine_kwargs.vllm.async_scheduling=false
        reward_models_config.reward_models.2.rollout.disable_log_stats=false
        reward_models_config.reward_models.2.rollout.skip_tokenizer_init=false
        reward_models_config.reward_models.2.rollout.prompt_length=$((1024 * 16))
        reward_models_config.reward_models.2.rollout.response_length=$((1024 * 20))
        reward_models_config.reward_models.2.rollout.max_model_len=$((1024 * 16 + 1024 * 20))
        reward_models_config.reward_models.2.rollout.runner=generate
        reward_models_config.reward_models.2.rollout.task=generate
        reward_models_config.reward_models.2.sampling_config.temperature=1.0
        reward_models_config.reward_models.2.sampling_config.top_p=1.0
        reward_models_config.reward_models.2.sampling_config.top_k=-1
        ++reward_models_config.reward_models.2.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton
    )

    mkdir -p "${PIVOTRL_WORKSPACE}/logs/${project_name}"

    # ---- Mode-derived deployment args ----
    DEPLOY_ARGS=(
        pivotrl.deployment.mode=${PIVOTRL_DEPLOY_MODE}
        pivotrl.staleness=${PIVOTRL_DEPLOY_STALENESS}
        reward_models_config.launch_reward_fn_async=${PIVOTRL_DEPLOY_RM_ASYNC}
        pivotrl.deployment.elastic_rm.shared_nnodes=${PIVOTRL_DEPLOY_SHARED_NNODES}
        "pivotrl.deployment.elastic_rm.shared_ngpus_per_node=${PIVOTRL_DEPLOY_SHARED_NGPUS}"
        pivotrl.deployment.elastic_rm.trainer_ready_grace_s=${PIVOTRL_DEPLOY_TRAINER_READY_GRACE_S}
        pivotrl.deployment.train_nnodes=${PIVOTRL_DEPLOY_TRAIN_NNODES}
        pivotrl.deployment.train_ngpus_per_node=${PIVOTRL_DEPLOY_TRAIN_NGPUS}
        pivotrl.deployment.total_nnodes=${PIVOTRL_DEPLOY_NNODES}
        pivotrl.deployment.rollout_nnodes_per_instance=1
        pivotrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE}
        pivotrl.deployment.n_validate_instances=${VAL_INSTANCES}
        pivotrl.deployment.validate_nnodes_per_instance=1
        pivotrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE}
    )

    # Non-elastic modes (disaggregated, trainer_pool_only) use independent rollout
    # / reward pools; main_ppo needs an explicit n_rollout_instances and (for the
    # gen RM) num_replicas. Elastic modes auto-compute these from the pool sizes.
    case "${PIVOTRL_DEPLOY_MODE}" in
        disaggregated|trainer_pool_only)
            SHARED_TOTAL_NGPUS=$(_pivotrl_total_gpus_from_node_spec "${PIVOTRL_DEPLOY_SHARED_NGPUS}" "${PIVOTRL_DEPLOY_SHARED_NNODES}")
            N_ROLLOUT_INSTANCES=$(( SHARED_TOTAL_NGPUS / ( GEN_TP * GEN_PP ) ))
            DEPLOY_ARGS+=(
                pivotrl.deployment.n_rollout_instances=${N_ROLLOUT_INSTANCES}
                pivotrl.deployment.elastic_rm.enable=False
            )
            if [[ "${PIVOTRL_DEPLOY_RM_NUM_REPLICAS}" != "0" ]]; then
                DEPLOY_ARGS+=( reward_models_config.reward_models.2.num_replicas=${PIVOTRL_DEPLOY_RM_NUM_REPLICAS} )
            fi
            ;;
        colocated|rollout_rm_colocated|elastic_rl)
            DEPLOY_ARGS+=( pivotrl.deployment.elastic_rm.enable=True )
            ;;
        *)
            echo "Unknown PIVOTRL_DEPLOY_MODE=${PIVOTRL_DEPLOY_MODE}" >&2
            exit 1
            ;;
    esac

    # Modes 1-4 use simple routing with a practically unbounded preemption
    # waiting cap. Mode 5 keeps the stricter cap required by elastic scaling.
    case "${PIVOTRL_DEPLOY_MODE}" in
        elastic_rl)
            ROUTING_ARGS=(
                pivotrl.routing_strategy.method=throughput_optimal
                pivotrl.routing_strategy.candidate_sort_indicator=reserve_capability
                pivotrl.routing_strategy.enable_multi_priority_queue=True
                pivotrl.routing_strategy.enable_group_sampling_on_multi_instances=True
                pivotrl.routing_strategy.cost_model_path=${PIVOTRL_PATH}/pivotrl/trainer/config/cost_model/qwen2.5_72b.json
                pivotrl.routing_strategy.delta_throughput_threshold=0.2
                pivotrl.routing_strategy.request_budget=1024
                pivotrl.routing_strategy.max_num_waiting_reqs_after_preemption=3
                pivotrl.routing_strategy.max_concurrent_seqs_per_instance=512
            )
            ;;
        disaggregated|colocated|rollout_rm_colocated|trainer_pool_only)
            ROUTING_ARGS=(
                pivotrl.routing_strategy.method=round_robin
                pivotrl.routing_strategy.candidate_sort_indicator=version
                pivotrl.routing_strategy.enable_multi_priority_queue=False
                pivotrl.routing_strategy.max_num_waiting_reqs_after_preemption=10000
                ++gen_actor_rollout_ref.rollout.use_pivotrl_scheduler=True
                ++reward_models_config.reward_models.2.routing_strategy.method=round_robin
                ++reward_models_config.reward_models.2.rollout.use_pivotrl_scheduler=True
            )
            ;;
    esac

    # Extra mode-specific overrides (space-separated string) split into args.
    if [[ -n "${PIVOTRL_DEPLOY_EXTRA}" ]]; then
        # shellcheck disable=SC2206
        DEPLOY_ARGS+=( ${PIVOTRL_DEPLOY_EXTRA} )
    fi

    PYTHONUNBUFFERED=1 python -m pivotrl.trainer.main_ppo \
        --config-name="${TRAINER_CONFIG_NAME}" \
        pivotrl.ps_manager_ip=${LOCAL_IP} \
        pivotrl.reward_service_ip=${LOCAL_IP} \
        pivotrl.rollout_n=${n_resp_per_prompt} \
        pivotrl.staleness_buffer_entries=${train_prompt_bsz} \
        pivotrl.ps_mode=nixl_cpu \
        pivotrl.logging_path=${PIVOTRL_PATH}/logs/${project_name}/${experiment_name} \
        \
        "${DEPLOY_ARGS[@]}" \
        \
        pivotrl.log_prob.enable_rollout_engine_log_prob=True \
        pivotrl.colocate_validate_and_train=False \
        pivotrl.fuse_rollout_with_validate=True \
        \
        pivotrl.nixl.server_port=23456 \
        pivotrl.group_post_process.enable=False \
        pivotrl.group_post_process.name=dynamic_sampling_filter \
        pivotrl.redundant_rollout.enable=False \
        pivotrl.partial_rollout.enable=True \
        \
        "${ROUTING_ARGS[@]}" \
        \
        gen_actor_rollout_ref.model.path="$HF_MODEL_PATH" \
        gen_actor_rollout_ref.model.use_shm=False \
        gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        gen_actor_rollout_ref.rollout.expert_parallel_size=${GEN_EP} \
        gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
        gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
        gen_actor_rollout_ref.rollout.enable_chunked_prefill=True \
        gen_actor_rollout_ref.rollout.max_num_batched_tokens=${packing_length} \
        gen_actor_rollout_ref.rollout.temperature=${temperature} \
        gen_actor_rollout_ref.rollout.top_p=${top_p} \
        gen_actor_rollout_ref.rollout.top_k=${top_k} \
        gen_actor_rollout_ref.rollout.disable_log_stats=false \
        \
        train_actor_rollout_ref.model.path="$HF_MODEL_PATH" \
        "${TRAIN_BACKEND_ARGS[@]}" \
        train_actor_rollout_ref.rollout.max_num_batched_tokens=${packing_length} \
        train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_num_batched_tokens} \
        train_actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
        train_actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        train_actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
        train_actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
        train_actor_rollout_ref.rollout.val_kwargs.n=1 \
        train_actor_rollout_ref.rollout.expert_parallel_size=${VAL_EP} \
        train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
        train_actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        train_actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        train_actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        train_actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        train_actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        train_actor_rollout_ref.actor.clip_ratio_c=10.0 \
        train_actor_rollout_ref.actor.optim.lr=1e-6 \
        train_actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        train_actor_rollout_ref.actor.optim.weight_decay=0.1 \
        train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_num_batched_tokens} \
        train_actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        train_actor_rollout_ref.actor.entropy_coeff=0 \
        train_actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        \
        "${REWARD_MODELS[@]}" \
        \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.return_raw_chat=True \
        data.train_batch_size=${train_prompt_bsz} \
        data.train_datas="${TRAIN_DATAS}" \
        data.val_datas="${VAL_DATAS}" \
        data.train_datasets_ratios="${TRAIN_DATASETS_RATIOS}" \
        \
        algorithm.adv_estimator=${adv_estimator} \
        algorithm.use_kl_in_reward=${use_kl_in_reward} \
        algorithm.kl_ctrl.kl_coef=${kl_coef} \
        algorithm.rollout_correction.rollout_is=${rollout_is} \
        algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
        pivotrl.proactive_filter_strategy.method="retry" \
        pivotrl.proactive_filter_strategy.threshold=0 \
        \
        trainer.logger='["console", "wandb"]' \
        +trainer.wandb_proxy=http://star-proxy.oa.com:3128 \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${experiment_name}" \
        trainer.val_before_train=False \
        trainer.test_freq=${test_freq} \
        trainer.save_freq=${save_freq} \
        trainer.total_epochs=1 \
        trainer.total_training_steps=${total_training_steps} \
        "$@" \
        2>&1 | tee ${PIVOTRL_WORKSPACE}/logs/${experiment_name}_${LOCAL_IP}.log
}
