# Evolutionary image generation

This project evolves the Stable Diffusion XL input noise that produces a population of images.
It supports the original embedding-novelty objective and a Gemma 4 creativity objective.

## Gemma creativity fitness

With `--evaluator gemma-creativity`, every generated image is shown to
`google/gemma-4-26B-A4B-it` with the same English creativity question used in the VLM/human
comparison:

> How creative do you find the image? Use a continuous score from 1 to 5, where higher scores
> mean that you find the image more creative. Return only valid JSON in this exact shape:
> `{"score": <number>}`

The parsed 1-5 score for the current image is its fitness. Higher scores win tournament selection.
New or mutated candidates do not inherit a parent's old rating. Gemma is loaded once per process,
uses BF16 and greedy decoding, and evaluates the PIL images serially to keep GPU memory predictable.
Malformed, non-finite, or out-of-range responses stop the run instead of silently changing the
fitness objective. Before scoring, each generated image is encoded once as JPEG and decoded for
Gemma; those exact JPEG bytes are subsequently archived, so the saved phenotype is the one rated.

Gemma's visual input budget is explicit and auditable. `--gemma_image_token_budget` (Slurm
variable `GEMMA_IMAGE_TOKEN_BUDGET`) accepts Gemma 4's supported values 70, 140, 280, 560, or
1120 and defaults to 280 to preserve the validated evaluator. It configures the model's
`max_soft_tokens` image processor without resizing or replacing the archived JPEG and is recorded
as `evaluator_config.image_processing.max_soft_tokens` in the experiment JSON. Changing it creates
a different fitness condition; use a new experiment id and do not combine trajectories across
budgets without validating their agreement.

Gemma inference batching is separately controlled by `--gemma_batch_size` (Slurm variable
`GEMMA_BATCH_SIZE`) and defaults to 1, preserving the validated serial behavior. Values above 1
send ordered groups of image chats through one model call and can improve throughput, but consume
more activation memory and may introduce small numerical differences. Smoke-test batch size 2
with the intended token budget before a long run; do not assume that merely requesting more GPUs
accelerates this single-process pipeline.

The Gemma-only path does not load BLIP2 or calculate the legacy novelty, diversity, caption, and
prompt-fidelity diagnostics. This reduces the combined SDXL/Gemma footprint and is intended for
the project's one-A100-80-GB profile; verify it with the small smoke test before a full run.

## Cluster setup

Use a dedicated environment and persistent Hugging Face cache on PanFS, not the home directory.
The project requires Python 3.12, PyTorch 2.9.1/torchvision 0.24.1 with CUDA 12.8, Transformers 5.5,
and Diffusers 0.37. Install it into a dedicated environment rather than the completed VLM
evaluation environment.
PyTorch 2.8 cannot run Gemma 4's grouped-MoE kernel on an A100. Install the official matched
wheel pair before installing the project:

```bash
"$PYTHON_EXE" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

"$PYTHON_EXE" -m pip install -e '.[test]'
```

Gemma may require accepting the model terms on Hugging Face and setting `HF_TOKEN`. The default
`GEMMA_REVISION` is pinned to `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, the exact model
artifact used in the successful human-rating comparison. The requested and resolved revisions
are written to the experiment JSON. `HF_HOME` follows the standard Hub layout and model
repositories are reused from `$HF_HOME/hub`.

SDXL is also pinned: `--sdxl_revision` (Slurm variable `SDXL_REVISION`) defaults to
`462165984030d82259a11f4367a4eed129e94a7b`, the snapshot observed in the successful smoke
run. Only a full commit hash is accepted. Diffusers downloads/reuses the required pipeline
files; the program verifies the returned cache snapshot matches that hash and loads exclusively
from that directory. It records `requested_revision`, `resolved_revision`, `revision_source`,
and `snapshot_path` under `generative_model_config` in the experiment JSON. This works even
when Diffusers omits `_commit_hash` from its pipeline configuration. A mismatch stops loading.
The command-line run always pins SDXL; older library callers can still load an unpinned model
by leaving the optional class/loader revision argument unset.

Start with a small systems smoke test:

```bash
export PROJECT_DIR=/path/to/Image_generation
export PYTHON_EXE=/path/to/env/bin/python
export BASE_PATH=/panfs/path/to/output-root
export HF_HOME=/panfs/path/to/cache/huggingface
export EXPERIMENT_ID=smoke-001
export GENERATION_PROMPT=cat
export NUM_GENERATIONS=1
export POPULATION_SIZE=4
export SDXL_BATCH_SIZE=1
export GEMMA_REVISION=4d7ae4984b7db7de8f8457170b3f1a419ee76d52
export GEMMA_IMAGE_TOKEN_BUDGET=280
export GEMMA_BATCH_SIZE=1
export SDXL_REVISION=462165984030d82259a11f4367a4eed129e94a7b

mkdir -p "$BASE_PATH"
sbatch --export=ALL cluster/run_gemma_creativity.sbatch
```

The Slurm entry point refuses a dirty source checkout. After the smoke test, submit a new
experiment id with the intended population and generation counts. Experiment directories are
claimed atomically and an existing id is rejected because automatic resume is not implemented.

After pulling this update, run the unit tests and repeat the small smoke test with a fresh
experiment id. Preserve earlier outputs: the previously successful run's null SDXL revision
is part of its original audit trail. Confirm the new JSON has the same full SDXL commit in
both requested/resolved fields, with `revision_source` equal to `huggingface_cache_snapshot`,
and that the CSV/ZIP again contain eight observations. Record the working environment for the
team beside the run using `python -m pip freeze` from the dedicated environment; a complete
tested cluster environment lock has not yet been committed.

## Outputs and audit trail

Each run writes a same-stem ZIP, CSV, and JSON below
`$BASE_PATH/<result-path>/<base-id>_<experiment-id>/`.

- The ZIP contains every generated JPEG and its input noise tensor.
- The CSV records each candidate and generation. `fitness`, `score_value`, and the score embedded
  in `file_name` are the current image's Gemma creativity rating for Gemma runs.
- `fitness_raw_response` preserves Gemma's exact text. Successful state rows have an empty
  `fitness_parse_error`. If a response cannot be used, the run stops and writes its response,
  reason, candidate, generation, and image hash to `<experiment>.fitness_failures.jsonl`.
- The JSON records the exact prompt, requested/resolved Gemma revision, decoding settings, SDXL
  settings/requested and resolved revisions, software versions, hardware, seed, fitness aggregation, generator
  Git commit, and evaluator class.
- `<experiment>.generation_timings.jsonl` records SDXL generation, preparation, Gemma fitness
  evaluation, post-evaluation, and total seconds for every completed generation, together with
  the SDXL batch size, Gemma batch size, and image-token budget. Use this file to identify the
  actual bottleneck before requesting additional GPUs.

Run unit checks without downloading Gemma:

```bash
"$PYTHON_EXE" -m pytest -q
```

## Scientific interpretation

This procedure optimizes noise tensors indirectly: Gemma rates the images rendered from those
tensors, and its ratings determine reproductive selection. It does not rate the raw tensor.
Because Gemma is the optimization target, evaluate final trajectories with held-out human ratings
or a different VLM/prompt. A high Gemma fitness alone does not establish a human creativity gain
and may eventually reflect exploitation of model-specific preferences.
