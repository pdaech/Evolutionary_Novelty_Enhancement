#!/usr/bin/env bash
# Run with git show REV:cluster/switch_thesis_six_gpu.sh |
# bash -s -- ROOT REV QUEUED_JOB RUNNING_JOB. Fetching leaves running source intact.
set -euo pipefail
umask 0002

ROOT="${1:?Supply shared VLM_stuff root}"
REVISION="${2:?Supply the full reviewed commit}"
QUEUED_JOB="${3:?Supply the old queued seed-2027 job ID}"
RUNNING_JOB="${4:?Supply the old running seed-2026 job ID}"
REPO="$ROOT/Image_generation"
PYTHON_EXE="$ROOT/envs/gemma-ga-py312/bin/python"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid revision'; exit 1; }
[[ "$QUEUED_JOB" =~ ^[1-9][0-9]*$ && "$RUNNING_JOB" =~ ^[1-9][0-9]*$ ]] || {
    echo 'Invalid job IDs'; exit 1;
}
[[ "$QUEUED_JOB" != "$RUNNING_JOB" ]] || { echo 'Job IDs must differ'; exit 1; }
CAMPAIGN_NAME="thesis6-sixgpu-2026-2027-r${RUNNING_JOB}-${QUEUED_JOB}"
CAMPAIGN="$ROOT/gemma_ga_outputs/submissions/$CAMPAIGN_NAME"
[[ ! -e "$CAMPAIGN" ]] || {
    echo "Replacement campaign already exists: $CAMPAIGN"
    echo 'Inspect its submission receipt and squeue; no duplicate job was submitted.'
    exit 1
}
[[ -x "$PYTHON_EXE" ]] || { echo "Missing project Python: $PYTHON_EXE"; exit 1; }
cd "$REPO"
[[ "$(git branch --show-current)" == 'codex/gemma-creativity-fitness' ]] || {
    echo 'Unexpected generator branch'; exit 1;
}
[[ -z "$(git status --porcelain)" ]] || { echo 'Generator checkout is not clean'; exit 1; }
git cat-file -e "$REVISION^{commit}"
git merge-base --is-ancestor HEAD "$REVISION"

# Check both identities before stopping either job. Only these explicit IDs are
# touched, and completed jobs simply fall out of the live queue query.
for JOB in "$QUEUED_JOB" "$RUNNING_JOB"; do
    LINES=$(squeue -h -u "$USER" -o '%F|%u|%j' | awk -F'|' -v j="$JOB" '$1 == j')
    if [[ -n "$LINES" ]] && ! printf '%s\n' "$LINES" |
        awk -F'|' -v u="$USER" '$2 != u || $3 != "gemma-thesis-full" {bad=1} END {exit bad}'; then
        echo "Unexpected identity for job $JOB; stopping before cancellation."
        exit 1
    fi
done

for JOB in "$QUEUED_JOB" "$RUNNING_JOB"; do
    ACTIVE=$(squeue -h -u "$USER" -o '%F' | awk -v j="$JOB" '$1 == j')
    if [[ -n "$ACTIVE" ]]; then
        printf 'Stopping old job %s; saved artifacts are retained.\n' "$JOB"
        scancel "$JOB"
    fi
    for CHECK in {1..60}; do
        ACTIVE=$(squeue -h -u "$USER" -o '%F' | awk -v j="$JOB" '$1 == j')
        [[ -z "$ACTIVE" ]] && break
        sleep 5
    done
    [[ -z "$ACTIVE" ]] || { echo "Job $JOB is still stopping; no new jobs submitted."; exit 1; }
done

git merge --ff-only "$REVISION"
[[ "$(git rev-parse HEAD)" == "$REVISION" ]] || { echo 'Revision mismatch'; exit 1; }
"$PYTHON_EXE" cluster/thesis_replicates.py preview --runtime-root "$ROOT" --campaign "$CAMPAIGN_NAME"
"$PYTHON_EXE" cluster/thesis_replicates.py submit --runtime-root "$ROOT" --campaign "$CAMPAIGN_NAME"
