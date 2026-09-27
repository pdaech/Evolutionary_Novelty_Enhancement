import itertools
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_thesis_replicates import launcher, make_artifact


@pytest.fixture
def pool(tmp_path):
    return launcher.make_remaining_plan(
        tmp_path / "code", tmp_path / "runtime", "remaining", "a" * 40
    )


def test_all_90_scientific_conditions_and_output_paths(pool, tmp_path):
    expected = set(
        itertools.product(
            launcher.REMAINING_CONSTRUCTS,
            [2025, 2026, 2027],
            [prompt for _, prompt in launcher.full.PROMPTS],
        )
    )
    actual = {
        (task["construct"], task["seed"], task["prompt"]) for task in pool["tasks"]
    }
    assert actual == expected and len(pool["tasks"]) == 90
    assert "novelty" not in pool["constructs"]
    assert len({task["key"] for task in pool["tasks"]}) == 90
    assert len({task["output_directory"] for task in pool["tasks"]}) == 90
    for task in pool["tasks"]:
        assert "lane" not in task
        scientific = launcher.task_plan(pool, task)
        construct = task["construct"]
        assert scientific["scoring_prompt"] == launcher.scoring_prompt(construct)
        assert scientific["options"] == dict(
            launcher.full.OPTIONS,
            evaluator="gemma-construct",
            gemma_fitness_construct=construct,
        )
        for attempt in (1, 2):
            current = launcher.attempt_task(task, attempt)
            command = launcher.full.inference_command(
                dict(scientific, seed=task["seed"]), current
            )
            options = dict(zip(command[3::2], command[4::2], strict=True))
            assert options["--gemma_fitness_construct"] == construct
            assert options["--seed"] == str(task["seed"])
            assert Path(current["output_directory"]) == (
                Path(pool["environment"]["BASE_PATH"])
                / options["--directory"]
                / f"{options['--id']}_{options['--experiment_id']}"
            )
    command = launcher.sbatch_command(pool, tmp_path, tmp_path / "batch-001", "hash")
    assert "--array=0-3%4" in command and "--gres=gpu:2" in command
    assert "--job-name=gemma-remaining" in command


def test_pool_plan_tampering_is_rejected(pool, tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    launcher.full.save_json(path, pool)
    monkeypatch.setattr(launcher.full, "clean_commit", lambda _: "a" * 40)
    assert launcher.verified_plan(tmp_path) == pool
    pool["tasks"][0]["construct"] = "novelty"
    path.write_text(json.dumps(pool), encoding="utf-8")
    with pytest.raises(ValueError, match="plan/settings changed"):
        launcher.verified_plan(tmp_path)


def test_eight_workers_claim_every_run_exactly_once_despite_failure(
    pool, tmp_path, monkeypatch
):
    launcher.atomic_json(
        tmp_path / "batches/batch-001/submission.json", {"job_id": "1234"}
    )
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1234")
    monkeypatch.setattr(launcher, "lane_lock", lambda *args: nullcontext())
    start = threading.Barrier(8)
    calls = []
    lock = threading.Lock()
    failure = pool["tasks"][7]["key"]

    def task(plan, campaign, item):
        with lock:
            calls.append(item["key"])
        time.sleep(0.002)
        return int(item["key"] == failure)

    def lane(index):
        start.wait(timeout=5)
        return launcher.run_lane(pool, tmp_path, index, "batch-001")

    monkeypatch.setattr(launcher, "run_task", task)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lane, range(8)))
    assert Counter(calls) == Counter(item["key"] for item in pool["tasks"])
    assert sum(results) == 1
    owners = [
        launcher.read_json(path)["lane"]
        for path in (tmp_path / "batches/batch-001/claims").glob("*/owner.json")
    ]
    assert set(owners) == set(range(8))


def test_new_batch_reclaims_interrupted_work_but_skips_completed(pool, tmp_path):
    first = next(launcher.claimed_tasks(pool, tmp_path, "batch-001", 0))
    launcher.atomic_json(
        launcher.completion_path(tmp_path, pool["tasks"][1]), {"status": "complete"}
    )
    remaining = list(launcher.claimed_tasks(pool, tmp_path, "batch-001", 1))
    assert len(remaining) == 88 and first not in remaining
    recovered = list(launcher.claimed_tasks(pool, tmp_path, "batch-002", 7))
    assert len(recovered) == 89 and first in recovered
    assert pool["tasks"][1] not in recovered


def test_pool_execution_uses_its_task_construct(pool, tmp_path, monkeypatch):
    task = pool["tasks"][-1]
    calls = []

    def execute(plan, current):
        calls.append(plan)
        assert plan["construct"] == "innovation"
        assert plan["seed"] == 2027
        assert current["construct"] == "innovation"
        return 0

    monkeypatch.setattr(launcher.full, "execute_task", execute)
    monkeypatch.setattr(
        launcher,
        "audit_artifacts",
        lambda plan, current: {"status": "complete", "task": current},
    )
    assert launcher.run_task(pool, tmp_path, task) == 0
    assert len(calls) == 1
    assert launcher.run_task(pool, tmp_path, task) == 0
    assert len(calls) == 1


def test_pool_artifact_audit_uses_correct_construct(pool):
    task = launcher.attempt_task(pool["tasks"][2], 1)
    scientific = launcher.task_plan(pool, task)
    make_artifact(scientific, task)
    assert launcher.audit_artifacts(pool, task)["rows"] == 3100
    task["construct"] = "originality"
    with pytest.raises(ValueError, match="metadata differs"):
        launcher.audit_artifacts(pool, task)


def test_pool_parent_passes_batch_and_writes_all_receipts_before_release(
    pool, tmp_path, monkeypatch
):
    batch = tmp_path / "batches/batch-001"
    launcher.atomic_json(batch / "submission.json", {"job_id": "1234"})
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1234")
    monkeypatch.setattr(launcher, "verified_plan", lambda *args: pool)
    events = []

    class Process:
        def __init__(self, command):
            assert command[command.index("--batch") + 1] == "batch-001"
            self.lane = int(command[command.index("--index") + 1])
            events.append(("launch", self.lane))

        def wait(self):
            events.append(("wait", self.lane))
            return 0

    def barrier(path, deadline, poll, groups):
        assert groups == 4
        assert events == [("launch", 0), ("launch", 1), ("wait", 0), ("wait", 1)]
        for task in pool["tasks"]:
            launcher.atomic_json(
                launcher.completion_path(tmp_path, task),
                {"status": "complete", "task": task},
            )
        return False

    original = launcher.atomic_json

    def save(path, value):
        if path.name == "release.json":
            report = launcher.read_json(tmp_path / "completion.json")
            assert report["total_rows"] == 279000 and len(report["tasks"]) == 90
        original(path, value)

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher, "wait_for_groups", barrier)
    monkeypatch.setattr(launcher, "atomic_json", save)
    assert launcher.run_group(tmp_path, "hash", "batch-001", 0) == 0


def test_pool_recovery_audits_retained_results_before_submit(
    pool, tmp_path, monkeypatch
):
    launcher.atomic_json(
        tmp_path / "batches/batch-001/submission.json", {"job_id": "1234"}
    )
    task = pool["tasks"][0]
    launcher.atomic_json(
        launcher.completion_path(tmp_path, task),
        {"task": task, "files": {"hash": "old"}},
    )
    monkeypatch.setattr(launcher, "verified_plan", lambda *args: pool)
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="")
    )
    monkeypatch.setattr(
        launcher, "audit_artifacts", lambda *args: {"files": {"hash": "changed"}}
    )
    monkeypatch.setattr(
        launcher,
        "submit_batch",
        lambda *args: pytest.fail("Must not submit changed artifacts"),
    )
    with pytest.raises(ValueError, match="hashes changed"):
        launcher.recover(tmp_path)
