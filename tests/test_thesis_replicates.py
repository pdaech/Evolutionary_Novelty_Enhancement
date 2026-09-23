import csv
import importlib.util
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

CLUSTER = Path(__file__).resolve().parents[1] / "cluster"
sys.path.insert(0, str(CLUSTER))
SPEC = importlib.util.spec_from_file_location(
    "thesis_replicates", CLUSTER / "thesis_replicates.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def plan(tmp_path):
    return launcher.make_plan(
        tmp_path / "code", tmp_path / "runtime", "replicates", [2026, 2027], "a" * 40
    )


def test_two_seeds_dispatched_exactly_once_on_six_gpu_lanes(plan, tmp_path):
    assert len(plan["tasks"]) == 12
    assert len({task["key"] for task in plan["tasks"]}) == 12
    for lane in range(6):
        tasks = [task for task in plan["tasks"] if task["lane"] == lane]
        assert [task["seed"] for task in tasks] == [2026, 2027]
        assert len({task["prompt"] for task in tasks}) == 1
        for task in tasks:
            current = launcher.attempt_task(task, 1)
            command = launcher.full.inference_command(
                dict(plan, seed=task["seed"]), current
            )
            options = dict(zip(command[3::2], command[4::2], strict=True))
            assert options["--seed"] == str(task["seed"])
            assert options["--num_generations"] == "30"
            assert options["--population_size"] == "100"
    command = launcher.sbatch_command(plan, tmp_path, tmp_path / "batch-001", "hash")
    for option in (
        "--array=0-2%3",
        "--gres=gpu:2",
        "--ntasks=2",
        "--mem=128G",
        "--partition=gpu2",
        "--time=5-00:00:00",
        "--no-requeue",
    ):
        assert option in command


def make_artifact(plan, task, *, last=30, invalid=False, missing_image=False):
    # Keep synthetic fixture paths below Windows MAX_PATH. Production Slurm paths
    # and experiment names are covered independently by the dispatch test.
    task["output_directory"] = str(Path(plan["project"]).parent / "artifact")
    task["run_name"] = "fixture"
    directory = Path(task["output_directory"])
    directory.mkdir(parents=True)
    stem = directory / task["run_name"]
    launcher.full.save_json(
        Path(f"{stem}.json"),
        {
            "prompt": task["prompt"],
            "random_seed": task["seed"],
            "population_size": 100,
            "num_generations": 30,
            "generation_code_commit": plan["generator_commit"],
            "fitness_aggregation": "latest",
            "evaluator": "GemmaCreativityEvaluator",
        },
    )
    buffer = io.BytesIO()
    Image.new("RGB", (1024, 1024), "white").save(buffer, format="JPEG")
    with (
        Path(f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle,
        zipfile.ZipFile(f"{stem}.zip", "w", zipfile.ZIP_DEFLATED) as archive,
    ):
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "generation",
                "candidate_id",
                "file_name",
                "fitness",
                "score_value",
                "fitness_parse_error",
            ],
        )
        writer.writeheader()
        for gen in range(last + 1):
            for candidate in range(100):
                name = f"g{gen}_idn_{candidate:04d}_f4.0"
                writer.writerow(
                    {
                        "generation": gen,
                        "candidate_id": f"n_{candidate:04d}",
                        "file_name": name,
                        "fitness": "nan" if invalid else "4.0",
                        "score_value": "4.0",
                        "fitness_parse_error": "",
                    }
                )
                if not (missing_image and gen == 30 and candidate == 99):
                    archive.writestr(f"/images/{name}.JPEG", buffer.getvalue())
                archive.writestr(f"/noise/{name}.pt", b"fixture-noise")
    return stem


def test_real_complete_archive_passes_and_retains_full_hashes(plan):
    task = launcher.attempt_task(plan["tasks"][0], 1)
    make_artifact(plan, task)
    report = launcher.audit_artifacts(plan, task)
    assert report["rows"] == 3100
    assert report["generations"] == list(range(31))
    assert all(len(value["sha256"]) == 64 for value in report["files"].values())


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"last": 29}, "3100 rows"),
        ({"invalid": True}, "Invalid creativity"),
        ({"missing_image": True}, "coverage differ"),
    ],
)
def test_partial_or_invalid_artifacts_never_mark_complete(plan, kwargs, message):
    task = launcher.attempt_task(plan["tasks"][0], 1)
    make_artifact(plan, task, **kwargs)
    with pytest.raises(ValueError, match=message):
        launcher.audit_artifacts(plan, task)


def fake_audit(plan, task):
    return {"status": "complete", "task": task, "rows": 3100, "files": {}}


def test_signal9_restarts_same_seed_in_new_directory(plan, tmp_path, monkeypatch):
    calls = []

    def execute(current_plan, task):
        calls.append((current_plan["seed"], task["output_directory"]))
        return -9 if len(calls) == 1 else 0

    monkeypatch.setattr(launcher.full, "execute_task", execute)
    monkeypatch.setattr(launcher, "audit_artifacts", fake_audit)
    task = plan["tasks"][0]
    assert launcher.run_task(plan, tmp_path, task) == 0
    assert calls[0][0] == calls[1][0] == 2026
    assert calls[0][1] != calls[1][1]
    assert launcher.completion_path(tmp_path, task).exists()
    assert (
        launcher.read_json(tmp_path / "attempts" / task["key"] / "attempt-01.json")[
            "returncode"
        ]
        == -9
    )


def test_application_failure_is_not_retried(plan, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        launcher.full, "execute_task", lambda *args: calls.append(args) or 1
    )
    assert launcher.run_task(plan, tmp_path, plan["tasks"][0]) == 1
    assert len(calls) == 1
    assert not launcher.completion_path(tmp_path, plan["tasks"][0]).exists()


def test_complete_runs_skipped_and_orphaned_success_adopted(
    plan, tmp_path, monkeypatch
):
    task = plan["tasks"][0]
    record = tmp_path / "attempts" / task["key"] / "attempt-01.json"
    launcher.atomic_json(
        record, {"task": launcher.attempt_task(task, 1), "status": "started"}
    )
    monkeypatch.setattr(launcher, "audit_artifacts", fake_audit)
    monkeypatch.setattr(
        launcher.full,
        "execute_task",
        lambda *args: pytest.fail("Must retain completed work"),
    )
    assert launcher.run_task(plan, tmp_path, task) == 0
    assert launcher.run_task(plan, tmp_path, task) == 0


def test_recovery_after_wrapper_kill_preserves_partial_attempt(
    plan, tmp_path, monkeypatch
):
    task = plan["tasks"][0]
    old = launcher.attempt_task(task, 1)
    record = tmp_path / "attempts" / task["key"] / "attempt-01.json"
    launcher.atomic_json(record, {"task": old, "status": "started"})

    def audit(configuration, current):
        if current["run_name"].endswith("attempt01"):
            raise ValueError("Only generation 29 saved")
        return fake_audit(configuration, current)

    calls = []
    monkeypatch.setattr(launcher, "audit_artifacts", audit)
    monkeypatch.setattr(
        launcher.full, "execute_task", lambda _, current: calls.append(current) or 0
    )
    assert launcher.run_task(plan, tmp_path, task) == 0
    assert calls[0]["run_name"].endswith("attempt02")
    assert launcher.read_json(record)["status"] == "started"


def test_lane_continues_other_seed_after_failure(plan, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, "lane_lock", lambda *args: nullcontext())
    monkeypatch.setattr(
        launcher, "run_task", lambda _, __, task: calls.append(task["seed"]) or 1
    )
    assert launcher.run_lane(plan, tmp_path, 0) == 1
    assert calls == [2026, 2027]


def test_barrier_does_not_release_fast_or_failed_allocation(tmp_path):
    launcher.atomic_json(tmp_path / "group-0.json", {"group": 0, "failed": False})
    launcher.atomic_json(tmp_path / "group-1.json", {"group": 1, "failed": True})
    with ThreadPoolExecutor() as executor:
        future = executor.submit(
            launcher.wait_for_groups, tmp_path, time.time() + 3, 0.01
        )
        # A missing third allocation must keep the parent alive, even on failure.
        with pytest.raises(TimeoutError):
            future.result(timeout=0.1)
        launcher.atomic_json(tmp_path / "group-2.json", {"group": 2, "failed": False})
        assert future.result(timeout=2) is True


def test_parent_waits_for_launched_worker_when_second_launch_fails(
    plan, tmp_path, monkeypatch
):
    batch = tmp_path / "batches/batch-001"
    launcher.atomic_json(batch / "submission.json", {"job_id": "1234"})
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1234")
    monkeypatch.setattr(launcher, "verified_plan", lambda *args: plan)
    events = []

    class Process:
        def __init__(self, command):
            lane = int(command[-1])
            if lane == 1:
                raise OSError("launch failed")
            events.append("start")

        def wait(self):
            events.append("wait")
            return 0

    def barrier(*args):
        assert events == ["start", "wait"]
        assert launcher.read_json(batch / "group-0.json")["failed"] is True
        events.append("barrier")
        return True

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher, "wait_for_groups", barrier)
    assert launcher.run_group(tmp_path, "hash", "batch-001", 0) == 1
    assert events[-1] == "barrier"


def test_recovery_refuses_live_or_ambiguous_jobs(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "verified_plan", lambda *args: plan)
    path = tmp_path / "batches/batch-001/submission.json"
    launcher.atomic_json(path, {"job_id": "1234"})
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="1234\n"),
    )
    monkeypatch.setattr(
        launcher,
        "submit_batch",
        lambda *args: pytest.fail("Unsafe duplicate submission"),
    )
    with pytest.raises(ValueError, match="still active"):
        launcher.recover(tmp_path)
    launcher.atomic_json(path, {"returncode": 1})
    with pytest.raises(ValueError, match="Ambiguous"):
        launcher.recover(tmp_path)


def test_submission_records_plan_and_sbatch_response(plan, tmp_path, monkeypatch):
    launcher.full.save_json(tmp_path / "plan.json", plan)
    monkeypatch.setenv("SBATCH_GRES", "gpu:8")

    def sbatch(command, **kwargs):
        assert (tmp_path / "batches/batch-001/request.json").exists()
        assert "SBATCH_GRES" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="1234;cluster\n", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", sbatch)
    launcher.submit_batch(plan, tmp_path)
    assert (
        launcher.read_json(tmp_path / "batches/batch-001/submission.json")["job_id"]
        == "1234"
    )


def test_verify_never_reports_success_for_missing_run(plan, tmp_path, capsys):
    launcher.full.save_json(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="not complete"):
        launcher.status(tmp_path, verify=True)
    assert "VERIFY OK" not in capsys.readouterr().out


def test_parent_saves_all_completion_reports_before_release(
    plan, tmp_path, monkeypatch
):
    batch = tmp_path / "batches/batch-001"
    launcher.atomic_json(batch / "submission.json", {"job_id": "1234"})
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1234")
    monkeypatch.setattr(launcher, "verified_plan", lambda *args: plan)
    events = []

    class Process:
        def __init__(self, command):
            self.lane = int(command[-1])
            events.append(("launch", self.lane))

        def wait(self):
            events.append(("wait", self.lane))
            return 0

    def barrier(*args):
        assert events == [("launch", 0), ("launch", 1), ("wait", 0), ("wait", 1)]
        assert not (batch / "release.json").exists()
        for task in plan["tasks"]:
            launcher.atomic_json(
                launcher.completion_path(tmp_path, task), fake_audit(plan, task)
            )
        return False

    original = launcher.atomic_json

    def save(path, value):
        if path.name == "release.json":
            report = launcher.read_json(tmp_path / "completion.json")
            assert report["total_rows"] == 37200 and len(report["tasks"]) == 12
        original(path, value)

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher, "wait_for_groups", barrier)
    monkeypatch.setattr(launcher, "atomic_json", save)
    assert launcher.run_group(tmp_path, "hash", "batch-001", 0) == 0


def test_recovery_submission_is_locked_through_queue_check(plan, tmp_path, monkeypatch):
    launcher.atomic_json(
        tmp_path / "batches/batch-001/submission.json", {"job_id": "1234"}
    )
    monkeypatch.setattr(launcher, "verified_plan", lambda *args: plan)

    def queue(command, **kwargs):
        assert "--jobs" not in command
        with pytest.raises(FileExistsError):
            launcher.recover(tmp_path)
        return SimpleNamespace(stdout="")

    def submit(*args):
        assert (tmp_path / ".recovery-submit-lock").is_dir()

    monkeypatch.setattr(launcher.subprocess, "run", queue)
    monkeypatch.setattr(launcher, "submit_batch", submit)
    launcher.recover(tmp_path)
    assert not (tmp_path / ".recovery-submit-lock").exists()
