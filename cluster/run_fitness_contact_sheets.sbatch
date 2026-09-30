#!/usr/bin/env bash
# Resources and logs are supplied by the sbatch command on submission.
set -euo pipefail
umask 0002
exec "$1" -u "$2" \
    --novelty-campaign "$3" \
    --remaining-campaign "$4" \
    --output-zip "$5"
