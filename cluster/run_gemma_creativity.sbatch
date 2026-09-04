#!/usr/bin/env bash
#SBATCH --account=dldevel
#SBATCH --partition=gpu2
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=5-00:00:00
#SBATCH --job-name=gemma-fitness

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR to the Image_generation checkout}"
: "${PYTHON_EXE:?Set PYTHON_EXE to the dedicated environment's Python}"
: "${BASE_PATH:?Set BASE_PATH to the PanFS output root}"
: "${HF_HOME:?Set HF_HOME to the persistent PanFS Hugging Face cache}"
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID to a unique run id}"
: "${GENERATION_PROMPT:?Set GENERATION_PROMPT to the Stable Diffusion prompt}"

cd "$PROJECT_DIR"
export GENERATION_CODE_COMMIT="${GENERATION_CODE_COMMIT:-$(git rev-parse HEAD)}"

gemma_revision_args=()
if [[ -n "${GEMMA_REVISION:-}" ]]; then
  gemma_revision_args=(--gemma_revision "$GEMMA_REVISION")
fi

"$PYTHON_EXE" run.py \
  --experiment_id "$EXPERIMENT_ID" \
  --id "${BASE_ID:-gemma-creativity}" \
  --directory "${RESULT_PATH:-results/simulations}" \
  --prompt "$GENERATION_PROMPT" \
  --seed "${RUN_SEED:-2025}" \
  --num_generations "${NUM_GENERATIONS:-30}" \
  --population_size "${POPULATION_SIZE:-100}" \
  --batch_size "${SDXL_BATCH_SIZE:-1}" \
  --evaluator gemma-creativity \
  --gemma_model "${GEMMA_MODEL:-google/gemma-4-26B-A4B-it}" \
  --gemma_max_new_tokens "${GEMMA_MAX_NEW_TOKENS:-64}" \
  "${gemma_revision_args[@]}"
