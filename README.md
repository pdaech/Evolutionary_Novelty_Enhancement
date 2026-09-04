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
fitness objective.

The Gemma-only path does not load BLIP2 or calculate the legacy novelty, diversity, caption, and
prompt-fidelity diagnostics. This reduces the combined SDXL/Gemma footprint and is intended for
the project's one-A100-80-GB profile; verify it with the small smoke test before a full run.

## Cluster setup

Use a dedicated environment and persistent Hugging Face cache on PanFS, not the home directory.
The project requires Python 3.12, a CUDA-enabled PyTorch build, and Transformers 5.5 or newer.
For example, after installing the correct PyTorch wheel for the cluster:

```bash
"$PYTHON_EXE" -m pip install -e '.[test]'
```

Gemma may require accepting the model terms on Hugging Face and setting `HF_TOKEN`. Pin
`GEMMA_REVISION` to the full Hugging Face commit for a scientific run. The resolved revision is
also written to the experiment JSON.

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

mkdir -p "$BASE_PATH"
sbatch --export=ALL cluster/run_gemma_creativity.sbatch
```

After the smoke test, submit a new experiment id with the intended population and generation
counts. Never reuse an old id: the state CSV is append-only.

## Outputs and audit trail

Each run writes a same-stem ZIP, CSV, and JSON below
`$BASE_PATH/<result-path>/<base-id>_<experiment-id>/`.

- The ZIP contains every generated JPEG and its input noise tensor.
- The CSV records each candidate and generation. `fitness`, `score_value`, and the score embedded
  in `file_name` are the current image's Gemma creativity rating for Gemma runs.
- `fitness_raw_response` preserves Gemma's exact text and `fitness_parse_error` records parser
  status (successful rows are empty).
- The JSON records the exact prompt, requested/resolved model revision, decoding settings,
  software versions, seed, fitness aggregation, and evaluator class.

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
