#!/usr/bin/env bash
# Resources supplied by thesis_replicates.py; three or four two-GPU allocations.
set -euo pipefail
umask 0002
exec "$1" -u "$2" run-group \
    --campaign "$3" --sha256 "$4" --batch "$5" --index "$SLURM_ARRAY_TASK_ID"
