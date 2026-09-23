"""Exercise the real transition shell with fake Slurm/Git commands only."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "cluster/switch_thesis_six_gpu.sh"


def prepare(tmp_path, wrong_name=False):
    bash = os.environ.get("TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("Bash is unavailable")
    (tmp_path / "fakebin").mkdir()
    (tmp_path / "runtime/Image_generation").mkdir(parents=True)
    python_path = tmp_path / "runtime/envs/gemma-ga-py312/bin/python"
    python_path.parent.mkdir(parents=True)
    (tmp_path / "jobs").write_text("1780644\n1780645\n", encoding="utf-8")
    scripts = {
        "fakebin/git": """#!/usr/bin/env bash
case "$1" in
  branch) echo codex/gemma-creativity-fitness ;;
  status|cat-file|merge-base) exit 0 ;;
  merge) [[ ! -s "$FIXTURE/jobs" ]] || exit 99; echo merge >> "$FIXTURE/events" ;;
  rev-parse) printf '%s\\n' "$REVISION" ;;
  *) exit 98 ;;
esac
""",
        "fakebin/squeue": """#!/usr/bin/env bash
while read -r job; do
  if [[ "$*" == *'%F|%u|%j'* ]]; then
    printf '%s|%s|%s\\n' "$job" "$USER" "$JOB_NAME"
  else
    echo "$job"
  fi
done < "$FIXTURE/jobs"
""",
        "fakebin/scancel": """#!/usr/bin/env bash
echo "cancel-$1" >> "$FIXTURE/events"
awk -v j="$1" '$1 != j' "$FIXTURE/jobs" > "$FIXTURE/jobs.tmp"
mv "$FIXTURE/jobs.tmp" "$FIXTURE/jobs"
""",
        "runtime/envs/gemma-ga-py312/bin/python": """#!/usr/bin/env bash
echo "$2" >> "$FIXTURE/events"
if [[ "$2" == submit ]]; then
    campaign=thesis6-sixgpu-2026-2027-r1780644-1780645
    mkdir -p "$FIXTURE/runtime/gemma_ga_outputs/submissions/$campaign"
fi
""",
    }
    for name, text in scripts.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8", newline="\n")
        path.chmod(0o755)
    environment = dict(
        os.environ,
        USER="test-cluster-user",
        SWITCH=SCRIPT.as_posix(),
        REVISION="a" * 40,
        JOB_NAME="unrelated-job" if wrong_name else "gemma-thesis-full",
    )
    command = [
        bash,
        "-c",
        """export FIXTURE="$PWD"
export PATH="$PWD/fakebin:$PATH"
exec bash "$SWITCH" "$FIXTURE/runtime" "$REVISION" 1780645 1780644
""",
    ]
    return command, environment


def test_switch_stops_dependency_first_and_is_duplicate_safe(tmp_path):
    command, environment = prepare(tmp_path)
    result = subprocess.run(
        command, env=environment, cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = ["cancel-1780645", "cancel-1780644", "merge", "preview", "submit"]
    assert (tmp_path / "events").read_text().splitlines() == expected
    result = subprocess.run(
        command, env=environment, cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode != 0 and "already exists" in result.stdout
    assert (tmp_path / "events").read_text().splitlines() == expected


def test_switch_refuses_wrong_identity_before_any_cancellation(tmp_path):
    command, environment = prepare(tmp_path, wrong_name=True)
    result = subprocess.run(
        command, env=environment, cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Unexpected identity" in result.stdout
    assert not (tmp_path / "events").exists()
