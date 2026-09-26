# Six image-only Gemma fitness conditions

This workflow keeps the six generation prompts, SDXL settings, GA operators,
Gemma model revision, and image-only 1–5 scoring format from the creativity
campaign. It varies one adjective in Gemma's scoring question. Each condition
is an independent evolutionary campaign with seeds 2025, 2026 and 2027.

| Condition ID | Exact first question |
| --- | --- |
| `novelty` | How novel do you find the image? |
| `unusualness` | How unusual do you find the image? |
| `uncommonness` | How uncommon do you find the image? |
| `uniqueness` | How unique do you find the image? |
| `originality` | How original do you find the image? |
| `innovation` | How innovative do you find the image? |

Each question then continues with the same instruction: “Use a continuous
score from 1 to 5, where higher scores mean that you find the image more
`<adjective>`. Return only valid JSON in this exact shape:
`{"score": <number>}`”. No generation prompt or reference image collection is
shown to Gemma. These are subjective image judgments, not corpus uniqueness
measurements. `uncommonness` is the condition ID; the spelling `uncommoness`
is treated as a typo and is not an alternative ID.

One campaign has six prompts × three seeds × 31 generations × 100 images =
55,800 image observations. All six campaigns have 334,800 observations.
Generation 0 is saved, and every accepted run must have 100 images in every
generation 0–30. The 2025 seed is rerun from generation 0 independently of
the earlier creativity archive, which was incomplete for some prompts.

The launcher allocates three Slurm jobs of two GPUs each on `gpu2`, with two
exclusive one-GPU lanes per job. Each lane processes one image prompt for all
three seeds. The jobs share a completion barrier and retain every attempt.
The launcher audits each completed JSON/CSV/ZIP triplet and archives the
paired initial-noise tensors. A signal-9 interruption can create one fresh
attempt with the same seed; it cannot resume a generation because the GA's
random state is not checkpointed. Invalid scores fail the task for review.

Use one construct campaign at a time to stay within the requested six-GPU
budget. Each campaign requests a five-day allocation limit. The six campaigns
together require roughly nine times the image evaluations of the successful
two-seed creativity campaign, before reruns or audits.

## Keep the previous generator checkout available

The existing creativity campaign pins generator commit `661ced6`. Create a
separate Git worktree for these new conditions so the old checkout remains at
its producing revision. In this example, `REVISION` is the reviewed full commit
hash supplied with the release; do not use a changing branch tip for an
already submitted campaign.

```bash
ROOT=/panfs/vdura1/dldevel/public/Evolutionary_Novelty_Enhancement_Results/VLM_stuff
SOURCE="$ROOT/Image_generation"
CODE="$ROOT/Image_generation_fitness_constructs"
PYTHON_EXE="$ROOT/envs/gemma-ga-py312/bin/python"
REVISION=PUT_REVIEWED_FULL_COMMIT_HASH_HERE

umask 0002
git -C "$SOURCE" fetch origin codex/gemma-fitness-constructs
git -C "$SOURCE" worktree add --detach "$CODE" "$REVISION"
```

Review the exact plan and question for one condition before submission:

```bash
CONSTRUCT=novelty
CAMPAIGN_NAME="thesis6-fitness-${CONSTRUCT}-seeds2025-2027-v1"
"$PYTHON_EXE" "$CODE/cluster/thesis_replicates.py" preview-construct \
    --runtime-root "$ROOT" \
    --construct "$CONSTRUCT" \
    --campaign "$CAMPAIGN_NAME"
```

After the scientific wording and resource request are approved, the same
command with `submit-construct` creates the campaign and submits one
three-job array. Do not rerun an ambiguous submission; inspect its
`batches/batch-001/submission.json` and the queue first. Once a campaign has
finished, use `status` for a quick summary and `verify` for a full archive,
JPEG, score and hash audit:

```bash
CAMPAIGN="$ROOT/gemma_ga_outputs/submissions/$CAMPAIGN_NAME"
"$PYTHON_EXE" "$CODE/cluster/thesis_replicates.py" status \
    --campaign "$CAMPAIGN"
"$PYTHON_EXE" "$CODE/cluster/thesis_replicates.py" verify \
    --campaign "$CAMPAIGN"
```

If a prior submission has left the queue with incomplete receipts, inspect
its lane logs and attempt records. `recover --campaign "$CAMPAIGN"` submits
a new batch only when no previous batch remains active and unused attempts
remain. A normal program error or invalid score requires investigation;
recovery is not a way to accept an alternative rating.

Every completed attempt records the construct ID, exact scoring question,
prompt hash, model revision, software and hardware metadata, source commit,
seed and generation settings. The `completed/` receipt identifies the
accepted attempt. Do not count ZIPs from failed attempts as extra replicates.
