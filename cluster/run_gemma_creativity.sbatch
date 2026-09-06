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
: "${PYTHON_EXE:?Set PYTHON_EXE to the dedicated environment Python}"
: "${BASE_PATH:?Set BASE_PATH to the PanFS output root}"
: "${HF_HOME:?Set HF_HOME to the persistent PanFS Hugging Face cache}"
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID to a unique run id}"
: "${GENERATION_PROMPT:?Set GENERATION_PROMPT to the Stable Diffusion prompt}"

cd "$PROJECT_DIR"
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing to run from a dirty Image_generation checkout" >&2
  git status --short >&2
  exit 2
fi
source_commit="$(git rev-parse HEAD)"
if [[ -n "${GENERATION_CODE_COMMIT:-}" && "$GENERATION_CODE_COMMIT" != "$source_commit" ]]; then
  echo "GENERATION_CODE_COMMIT does not match the checked-out source: $source_commit" >&2
  exit 2
fi
export GENERATION_CODE_COMMIT="$source_commit"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HUB_CACHE"

GEMMA_REVISION="${GEMMA_REVISION:-4d7ae4984b7db7de8f8457170b3f1a419ee76d52}"

"$PYTHON_EXE" run.py \
  --experiment_id "$EXPERIMENT_ID" \
  --id "${BASE_ID:-gemma-creativity}" \
  --directory "${RESULT_PATH:-results/simulations}" \
  --prompt "$GENERATION_PROMPT" \
  --seed "${RUN_SEED:-2025}" \
  --num_generations "${NUM_GENERATIONS:-30}" \
  --population_size "${POPULATION_SIZE:-100}" \
  --batch_size "${SDXL_BATCH_SIZE:-1}" \
  --sdxl_revision "${SDXL_REVISION:-462165984030d82259a11f4367a4eed129e94a7b}" \
  --evaluator gemma-creativity \
  --gemma_model "${GEMMA_MODEL:-google/gemma-4-26B-A4B-it}" \
  --gemma_revision "$GEMMA_REVISION" \
  --gemma_max_new_tokens "${GEMMA_MAX_NEW_TOKENS:-64}" \
  --gemma_image_token_budget "${GEMMA_IMAGE_TOKEN_BUDGET:-280}"
