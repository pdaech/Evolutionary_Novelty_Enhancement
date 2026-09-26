#!/usr/bin/env bash
# Resources are supplied by smoke_fitness_construct.py at submission.
set -euo pipefail
umask 0002
exec "$1" -u "$2" run-job --campaign "$3" --sha256 "$4"
