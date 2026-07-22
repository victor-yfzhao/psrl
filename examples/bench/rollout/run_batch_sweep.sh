#!/usr/bin/env bash
set -euo pipefail

# Sweep rollout bench over models, disable_attn settings, and batch sizes.
#
# Usage:
#   export PSRL_WORKSPACE=/path/to/psrl
#   bash examples/bench/rollout/run_batch_sweep.sh [tp] [prompt_len]
#
# Example:
#   bash examples/bench/rollout/run_batch_sweep.sh 1 1024

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLLOUT_SCRIPT="${SCRIPT_DIR}/run_rollout_test.sh"

if [[ -z "${PSRL_WORKSPACE:-}" ]]; then
    echo "Error: PSRL_WORKSPACE is not set."
    exit 1
fi

if [[ ! -f "${ROLLOUT_SCRIPT}" ]]; then
    echo "Error: ${ROLLOUT_SCRIPT} not found."
    exit 1
fi
chmod +x "${ROLLOUT_SCRIPT}"

GEN_TP="${1:-2}"
MAX_PROMPT_LENGTH="${2:-1024}"

MODEL_NAMES=(DeepSeek-R1-Distill-Qwen-32B Qwen3-30B-A3B-Instruct)
DISABLE_ATTN_VALUES=(false true)
BATCH_SIZE_VALUES=(1 2 4 8 16 32 64 128)

total=$((${#MODEL_NAMES[@]} * ${#DISABLE_ATTN_VALUES[@]} * ${#BATCH_SIZE_VALUES[@]}))
current=0
failed=()

echo "=========================================="
echo "Rollout batch sweep"
echo "PSRL_WORKSPACE: ${PSRL_WORKSPACE}"
echo "TP: ${GEN_TP}, prompt_len: ${MAX_PROMPT_LENGTH}"
echo "Models: ${MODEL_NAMES[*]}"
echo "disable_attn: ${DISABLE_ATTN_VALUES[*]}"
echo "Batch sizes: ${BATCH_SIZE_VALUES[*]}"
echo "Total runs: ${total}"
echo "Start: $(date)"
echo "=========================================="

for model_name in "${MODEL_NAMES[@]}"; do
    for disable_attn in "${DISABLE_ATTN_VALUES[@]}"; do
        for batch_size in "${BATCH_SIZE_VALUES[@]}"; do
            current=$((current + 1))
            run_label="model=${model_name}, disable_attn=${disable_attn}, batch_size=${batch_size}"

            echo ""
            echo "=========================================="
            echo "Run ${current}/${total}: ${run_label}"
            echo "Start: $(date)"
            echo "=========================================="

            if MODEL_NAME="${model_name}" \
                "${ROLLOUT_SCRIPT}" "${GEN_TP}" "${MAX_PROMPT_LENGTH}" "${batch_size}" "${disable_attn}"; then
                echo "Run ${current}/${total} (${run_label}) succeeded."
            else
                echo "Run ${current}/${total} (${run_label}) failed (exit $?)."
                failed+=("${run_label}")
            fi

            echo "End: $(date)"
        done
    done
done

echo ""
echo "=========================================="
echo "Batch sweep finished: $(date)"
if ((${#failed[@]} == 0)); then
    echo "All ${total} runs succeeded."
else
    echo "Failed runs (${#failed[@]}/${total}):"
    for run in "${failed[@]}"; do
        echo "  - ${run}"
    done
    exit 1
fi
echo "=========================================="
