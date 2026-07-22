#!/usr/bin/env bash
set -xeuo pipefail

# Simplified rollout performance test script
# This script runs a simplified rollout performance test using vLLM AsyncLLM directly

# Set up environment
source ${PSRL_WORKSPACE}/env/env_311.sh

HOME=${PSRL_WORKSPACE}
PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")

# Model configuration
MODEL_NAME=${MODEL_NAME:-Qwen2.5-7B}
HF_MODEL_PATH=${HF_MODEL_PATH:-${PSRL_WORKSPACE}/models/${MODEL_NAME}}

# vLLM configuration (simplified - no complex deployment)
GEN_TP=${1:-1}  # Tensor parallel size for generation
GEN_PP=1  # Pipeline parallel size for generation

# Node configuration
NNODES=1  # Simplified to single node
NGPUS_PER_NODE=8

# Test parameters
max_prompt_length=${2:-128}
batch_size=${3:-1}
disable_attn=${4:-true}
max_response_length=8192

max_model_len=$((max_prompt_length + max_response_length))

num_iterations=1
warmup_iterations=1
test_mode="synthetic"  # "synthetic" or "real_data"

# Generation parameters
temperature=1.0
top_p=1.0
top_k=-1

enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-False}

profile_root=${PROFILE_ROOT:-${PSRL_WORKSPACE}/examples/bench/rollout/exp}
profile_logs_dir=${PROFILE_LOGS_DIR:-${profile_root}/details}
summary_dir=${PROFILE_SUMMARY_DIR:-${profile_root}/summary}
disable_tag=$(echo "${disable_attn}" | tr '[:upper:]' '[:lower:]')
profile_log_file=${PROFILE_LOG_FILE:-${MODEL_NAME}_Syn_TP${GEN_TP}_PP${GEN_PP}_B${batch_size}_P${max_prompt_length}_R${max_response_length}_disable_attn_${disable_tag}}
run_log=${RUN_LOG:-rollout_test_${MODEL_NAME}_tp${GEN_TP}_b${batch_size}_p${max_prompt_length}_r${max_response_length}_disable_attn_${disable_tag}.log}

mkdir -p "${profile_logs_dir}" "${summary_dir}"

# The bench rollout entrypoint does not wire rollout.disable_attn into the worker
# env like the training path does, so we set the vLLM env var explicitly here.
if [[ "${disable_tag}" == "true" ]]; then
    export VLLM_DISABLE_ATTN=1
else
    unset VLLM_DISABLE_ATTN || true
fi

PYTHONUNBUFFERED=1 python -m psrl.bench.rollout.main_rollout \
    psrl.logging_path=${summary_dir} \
    \
    model.path="${HF_MODEL_PATH}" \
    +model.override_config.max_position_embeddings=32768 \
    \
    rollout.gpu_memory_utilization=0.95 \
    rollout.tensor_parallel_size=${GEN_TP} \
    rollout.pipeline_parallel_size=${GEN_PP} \
    rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    rollout.max_num_seqs=${batch_size} \
    rollout.max_num_batched_tokens=$((max_prompt_length * batch_size)) \
    rollout.temperature=${temperature} \
    rollout.top_p=${top_p} \
    rollout.top_k=${top_k} \
    +rollout.disable_attn=${disable_tag} \
    rollout.disable_log_stats=false \
    rollout.ignore_eos=false \
    \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    \
    rollout_test.batch_size=${batch_size} \
    rollout_test.num_iterations=${num_iterations} \
    rollout_test.warmup_iterations=${warmup_iterations} \
    rollout_test.mode=${test_mode} \
    rollout_test.profile_logs_dir=${profile_logs_dir} \
    rollout_test.profile_log_file=${profile_log_file} \
    2>&1 | tee "${run_log}"
