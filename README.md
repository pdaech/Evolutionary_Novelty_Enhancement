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

mkdir -p "$BASE_PATH"
sbatch --export=ALL cluster/run_gemma_creativity.sbatch
```

The Slurm entry point refuses a dirty source checkout. After the smoke test, submit a new
experiment id with the intended population and generation counts. Experiment directories are
claimed atomically and an existing id is rejected because automatic resume is not implemented.

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
  settings/resolved revision, software versions, hardware, seed, fitness aggregation, generator
  Git commit, and evaluator class.

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
