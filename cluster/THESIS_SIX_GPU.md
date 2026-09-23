# Two replicate seeds on six GPUs

`thesis_replicates.py` runs the six fixed thesis prompts with seeds 2026 and
2027. It uses three separate Slurm array allocations, each reserving two GPUs,
16 CPUs and 128 GiB host RAM on gpu30-022 in partition/QOS gpu2. Each allocation
requests five days. Six exclusive one-GPU workers each process one prompt with
seed 2026, then with seed 2027. Both seeds share one campaign and one array.

The scientific generator, Gemma prompt/model settings, GA operators, population
100, and generations 0-30 are unchanged. Expected output is 3,100 images per
prompt/seed and 37,200 images overall. Based on the earlier approximately 26-hour
prompt runs, budget roughly 52 hours plus queueing, startup and validation. Six
GPUs are the maximum simultaneous request; Slurm can start allocations at
different times. A finished lane holds its allocation until all workers finish.

## Completion protection and recovery

Earlier co-located allocations were killed with signal 9 when another allocation
ended. The exact site-level cause is unproven. The new shared exit barrier keeps
all three parent jobs alive until every worker has stopped and finished its
artifact audit, including when a worker fails. This applies across both seeds.
No parent exits simply because its two prompts finished first. The same approach
was introduced in the project's later multi-question evaluation campaigns.

Every completed trajectory must pass all of these checks before receiving a
`completed/<seed-prompt>.json` receipt:

- Exactly 100 unique candidates in every generation 0 through 30 (3,100 rows).
- Correct prompt, seed, generation settings, evaluator, fitness aggregation and
  generator commit; finite fitness in 1-5 and no recorded parse-invalid fitness.
- Exact correspondence between CSV filenames, ZIP JPEGs and archived noise.
- No duplicate ZIP members, good CRCs, and all 3,100 JPEGs fully decoded at 1024².
- SHA-256 hashes of the CSV, JSON and ZIP are stored in the receipt.

If the generator subprocess is killed with signal 9 while its worker survives,
the worker automatically makes one fresh attempt with the same seed. It retains
the interrupted attempt. It never retries a normal application error or invalid
Gemma rating to obtain a more convenient score. Maximum: two attempts per
prompt/seed, including any later recovery submission. A failed prompt does not
prevent the other seed in its lane from being attempted.

If an entire allocation or worker is killed, `recover` can be run after all jobs
from previous submissions have left the queue. It retains and revalidates
completed runs, detects complete artifacts whose final receipt was interrupted,
and restarts the incomplete prompt from generation zero with the same seed.
Recovery is not generation-level resume: the generator has no saved RNG state.
Every attempt has a unique directory; generations from different attempts are
never joined. A trajectory interrupted after generation 29 must be rerun in full.

Hardware failures, site cancellation and time limits can still interrupt jobs.
The launcher detects incomplete results rather than promising that a scheduler
can never stop them. Near the five-day limit, a waiting barrier fails explicitly
if another allocation never finished. Investigate repeated kills with the cluster
administrators. Keep the submitted code and environment unchanged until complete.

## Switching from the earlier two-GPU submissions

For the current submitted jobs, stop queued seed-2027 job **1780645 first**, wait
until it leaves the queue, then stop seed-2026 job **1780644** and wait again. The
order matters: cancelling 1780644 first could release 1780645's `afterany`
dependency. Preserve both old campaign directories. Their partial trajectories
cannot be resumed exactly; the new campaign starts both seeds from zero.

The recommended transition uses `switch_thesis_six_gpu.sh`. Fetch the reviewed
commit without changing the running checkout, then execute that committed script:

```bash
ROOT=/panfs/vdura1/dldevel/public/Evolutionary_Novelty_Enhancement_Results/VLM_stuff
REPO="$ROOT/Image_generation"
git -C "$REPO" fetch origin codex/gemma-creativity-fitness
REVISION=$(git -C "$REPO" rev-parse origin/codex/gemma-creativity-fitness)
git -C "$REPO" show "$REVISION:cluster/switch_thesis_six_gpu.sh" |
    bash -s -- "$ROOT" "$REVISION" 1780645 1780644
```

The script verifies the clean branch, fast-forward revision and both job
identities before cancellation. It stops jobs in the required order, waits for
them to leave the queue, updates to that exact revision, and submits the new
campaign. Its fixed name is `thesis6-sixgpu-2026-2027-r1780644-1780645`; running the
script again refuses to create a duplicate. Any existing partial output is kept.
If fetch fails, stop before executing the script. The explicitly reviewed commit
from the deployment instructions can be supplied as REVISION instead of resolving
the branch tip.

The longer equivalent manual procedure follows for reference.

Run this block in the cluster shell. It checks job owner and name before
cancelling, leaves all saved files in place, and updates the checkout only after
both old jobs have stopped. The parentheses keep `set -e` local to this block.

```bash
(
set -euo pipefail
ROOT=/panfs/vdura1/dldevel/public/Evolutionary_Novelty_Enhancement_Results/VLM_stuff
REPO="$ROOT/Image_generation"
PYTHON_EXE="$ROOT/envs/gemma-ga-py312/bin/python"

for JOB in 1780645 1780644; do
    ACTIVE=$(squeue -h -u "$USER" -o '%F|%u|%j' | awk -F'|' -v j="$JOB" '$1 == j')
    if [[ -n "$ACTIVE" ]]; then
        if printf '%s\n' "$ACTIVE" | grep -v -F "|$USER|gemma-thesis-full"; then
            echo "Unexpected job identity; stopping."
            exit 1
        fi
        scancel "$JOB"
    fi
    for CHECK in {1..60}; do
        ACTIVE=$(squeue -h -u "$USER" -o '%F' | awk -v j="$JOB" '$1 == j')
        [[ -z "$ACTIVE" ]] && break
        sleep 5
    done
    [[ -z "$ACTIVE" ]] || { echo "Job $JOB is still stopping; do not submit yet."; exit 1; }
done

cd "$REPO"
git pull --ff-only origin codex/gemma-creativity-fitness
CAMPAIGN_NAME="thesis6-sixgpu-2026-2027-$(date +%Y%m%d-%H%M%S)"
"$PYTHON_EXE" cluster/thesis_replicates.py preview --runtime-root "$ROOT" --campaign "$CAMPAIGN_NAME"
"$PYTHON_EXE" cluster/thesis_replicates.py submit --runtime-root "$ROOT" --campaign "$CAMPAIGN_NAME"
)
```

Do not rerun the entire block if the submission response is ambiguous. Inspect
`squeue` and the campaign's `batches/batch-001/submission.json` first. Campaign
names are unique and never overwrite previous output. No dependency on old job
1780644 or 1780645 is retained in the replacement campaign.

## Monitoring and final verification

Use the exact campaign directory printed by submission:

```bash
ROOT=/panfs/vdura1/dldevel/public/Evolutionary_Novelty_Enhancement_Results/VLM_stuff
REPO="$ROOT/Image_generation"
PYTHON_EXE="$ROOT/envs/gemma-ga-py312/bin/python"
CAMPAIGN="$ROOT/gemma_ga_outputs/submissions/PUT_PRINTED_CAMPAIGN_NAME_HERE"

squeue -u "$USER"
"$PYTHON_EXE" "$REPO/cluster/thesis_replicates.py" status --campaign "$CAMPAIGN"
```

`status` is a light read-only progress check. `batches/batch-001/lane-0.out/.err`
through `lane-5.out/.err` contain the worker logs; `job-*.out/.err` contain the
parent/barrier logs. Every attempt has its own receipt under `attempts/`.
`completed/` identifies the definitive complete attempt for each prompt and seed;
`completion.json` lists them together after a successful barrier. Do not treat
every ZIP found recursively as a separate scientific replicate: failed attempts
remain available for audit.

After all jobs end, `verify` rechecks every artifact and its stored hash:

```bash
"$PYTHON_EXE" "$REPO/cluster/thesis_replicates.py" verify --campaign "$CAMPAIGN"
```

It can take several minutes because it reads and decodes all images; each worker
already performed the same validation inside its Slurm allocation. A successful
final line is `VERIFY OK: 37,200 observations; all twelve runs have generations 0-30`.
Slurm accounting remains useful, but an exit status alone is not artifact proof.

For an interrupted allocation with attempts remaining, after checking the logs:

```bash
"$PYTHON_EXE" "$REPO/cluster/thesis_replicates.py" recover --campaign "$CAMPAIGN"
```

This submits another three-job array using the same immutable plan. It refuses
live previous jobs, unknown submission outcomes, or a campaign already marked
complete. Worker locks prevent duplicate trajectory writers. Application errors
and invalid fitness remain failures requiring investigation.
If a recovery submitter itself is killed, the `.recovery-submit-lock` directory
may remain; inspect Slurm and all submission receipts before clearing that empty
lock. It intentionally blocks another submission with an uncertain outcome.
