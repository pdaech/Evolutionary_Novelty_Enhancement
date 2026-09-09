#!/usr/bin/env bash
# Resources are supplied explicitly by thesis_full.py.
set -euo pipefail
exec "$1" -u "$2" run-group \
    --manifest "$3" --sha256 "$4" --index "$SLURM_ARRAY_TASK_ID"
