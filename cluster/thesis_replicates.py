"""Six or eight GPU lanes, artifact audits, and a shared exit barrier.

Recovery restarts an interrupted trajectory from generation zero with its original
seed; it never splices generations. Construct campaigns use a distinct score prompt.
"""

import argparse
import csv
import getpass
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import zipfile
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import thesis_full as full

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gemma_constructs import ADJECTIVES, scoring_prompt  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(UTC).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_plan(project, runtime, campaign, seeds, commit):
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,59}", campaign),
        "Campaign must contain 1-60 letters, digits, - or _",
    )
    require(len(seeds) == 2 and len(set(seeds)) == 2, "Supply two distinct seeds")
    plans = [
        full.make_plan(
            project, runtime, f"{campaign}-s{seed}", seed, 2, "gpu30-022", commit
        )
        for seed in seeds
    ]
    plan = plans[0]
    plan.update(
        schema_version=2,
        kind="six-gpu-replicates",
        campaign=campaign,
        seeds=seeds,
        max_attempts=2,
        scheduling="three-allocations-shared-barrier",
    )
    tasks = []
    # One lane per prompt: first seed, then second seed. Every allocation stays
    # alive through both seeds, including the final artifact audits.
    for seed, seed_plan in zip(seeds, plans, strict=True):
        for lane, task in enumerate(seed_plan["tasks"]):
            tasks.append(
                dict(
                    task,
                    lane=lane,
                    seed=seed,
                    key=f"seed{seed}-{full.PROMPTS[lane][0]}",
                )
            )
    plan["tasks"] = tasks
    return plan


def make_construct_plan(project, runtime, campaign, seeds, commit, construct, gpus=6):
    require(gpus in (6, 8), "Choose six or eight GPUs")
    require(construct in ADJECTIVES, "Unknown fitness construct")
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,59}", campaign),
        "Campaign must contain 1-60 letters, digits, - or _",
    )
    require(
        list(seeds) == [2025, 2026, 2027],
        "Construct comparison uses matched seeds 2025, 2026 and 2027",
    )
    plans = [
        full.make_plan(
            project, runtime, f"{campaign}-s{seed}", seed, 2, "gpu30-022", commit
        )
        for seed in seeds
    ]
    plan = plans[0]
    plan.update(
        schema_version=3,
        kind="six-gpu-fitness-construct",
        campaign=campaign,
        construct=construct,
        scoring_prompt=scoring_prompt(construct),
        seeds=list(seeds),
        max_attempts=2,
        scheduling="three-allocations-shared-barrier",
        run_id=f"gemma-{construct}",
    )
    plan["options"] = dict(
        full.OPTIONS,
        evaluator="gemma-construct",
        gemma_fitness_construct=construct,
    )
    tasks = []
    for seed, seed_plan in zip(seeds, plans, strict=True):
        for lane, task in enumerate(seed_plan["tasks"]):
            slug = full.PROMPTS[lane][0]
            experiment = f"{campaign}-{slug}-p100-g30-seed{seed}"
            run_name = f"gemma-{construct}_{experiment}"
            tasks.append(
                dict(
                    task,
                    experiment_id=experiment,
                    run_name=run_name,
                    output_directory=str(
                        runtime / "gemma_ga_outputs/results/simulations" / run_name
                    ),
                    lane=lane,
                    seed=seed,
                    key=f"{construct}-seed{seed}-{slug}",
                )
            )
    plan["tasks"] = tasks
    if gpus == 8:
        plan.update(
            schema_version=4,
            gpu_count=8,
            scheduling="four-allocations-shared-barrier",
        )
        for index, task in enumerate(tasks):
            task["lane"] = index % 8
    return plan


def group_count(plan):
    return plan.get("gpu_count", 6) // 2


def verified_plan(campaign, digest=None):
    path = Path(campaign) / "plan.json"
    if digest is not None:
        require(sha256(path) == digest, "Campaign plan changed after submission")
    plan = read_json(path)
    if plan.get("kind") == "six-gpu-replicates" and plan.get("schema_version") == 2:
        require(plan["options"] == full.OPTIONS, "Scientific settings changed")
    else:
        require(
            plan.get("kind") == "six-gpu-fitness-construct"
            and plan.get("schema_version") in (3, 4),
            "Unexpected campaign schema",
        )
        construct = plan["construct"]
        require(construct in ADJECTIVES, "Unknown fitness construct")
        require(
            plan["options"]
            == dict(
                full.OPTIONS,
                evaluator="gemma-construct",
                gemma_fitness_construct=construct,
            ),
            "Construct scientific settings changed",
        )
        require(
            plan["scoring_prompt"] == scoring_prompt(construct),
            "Construct scoring prompt changed",
        )
        require(plan["seeds"] == [2025, 2026, 2027], "Construct seeds changed")
        require(plan["run_id"] == f"gemma-{construct}", "Construct run ID changed")
        require(len(plan["tasks"]) == 18, "Expected 18 construct runs")
        require(
            len({task["key"] for task in plan["tasks"]}) == 18,
            "Duplicate construct task key",
        )
        if plan["schema_version"] == 4:
            require(
                plan.get("gpu_count") == 8
                and plan.get("scheduling") == "four-allocations-shared-barrier"
                and [task["lane"] for task in plan["tasks"]]
                == [index % 8 for index in range(18)],
                "Eight-GPU scheduling changed",
            )
        else:
            require(plan.get("gpu_count", 6) == 6, "Legacy GPU count changed")
    require(
        full.clean_commit(plan["project"]) == plan["generator_commit"],
        "Generator checkout changed; restore its submitted commit",
    )
    return plan


def attempt_task(task, number):
    suffix = f"-attempt{number:02d}"
    result = dict(task)
    for field in ("experiment_id", "run_name"):
        result[field] += suffix
    result["output_directory"] = str(
        Path(task["output_directory"]).with_name(result["run_name"])
    )
    return result


def audit_artifacts(plan, task):
    """Require all populations, finite fitness, exact ZIP coverage, CRCs and JPEGs."""
    from PIL import Image

    stem = Path(task["output_directory"]) / task["run_name"]
    paths = {
        extension: Path(f"{stem}.{extension}") for extension in ("json", "csv", "zip")
    }
    config = read_json(paths["json"])
    for field, expected in {
        "prompt": task["prompt"],
        "random_seed": task["seed"],
        "population_size": 100,
        "num_generations": 30,
        "generation_code_commit": plan["generator_commit"],
        "fitness_aggregation": "latest",
        "evaluator": (
            "GemmaConstructEvaluator"
            if plan.get("kind") == "six-gpu-fitness-construct"
            else "GemmaCreativityEvaluator"
        ),
    }.items():
        require(config.get(field) == expected, f"{task['key']}: unexpected {field}")
    if plan.get("kind") == "six-gpu-fitness-construct":
        evaluator_config = config.get("evaluator_config", {})
        require(
            evaluator_config.get("construct") == plan["construct"]
            and evaluator_config.get("prompt") == plan["scoring_prompt"]
            and evaluator_config.get("prompt_sha256")
            == hashlib.sha256(plan["scoring_prompt"].encode("utf-8")).hexdigest()
            and evaluator_config.get("objective")
            == f"maximize_current_image_{plan['construct']}"
            and evaluator_config.get("model_id") == plan["options"]["gemma_model"]
            and evaluator_config.get("requested_revision")
            == plan["options"]["gemma_revision"]
            and evaluator_config.get("image_processing", {}).get("max_soft_tokens")
            == plan["options"]["gemma_image_token_budget"],
            f"{task['key']}: construct evaluator metadata differs",
        )
    failure = Path(f"{stem}.fitness_failures.jsonl")
    require(
        not failure.exists() or failure.stat().st_size == 0,
        "Fitness failures require investigation; never replace invalid scores",
    )
    with paths["csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 3100, f"Expected 3100 rows, found {len(rows)}")
    counts = Counter(int(row["generation"]) for row in rows)
    require(
        counts == {generation: 100 for generation in range(31)},
        f"Incomplete or unexpected generations: {dict(counts)}",
    )
    require(
        len({(int(row["generation"]), row["candidate_id"]) for row in rows}) == 3100,
        "Duplicate candidate within a generation",
    )
    names = [row["file_name"] for row in rows]
    require(len(set(names)) == 3100, "Duplicate image filename")
    for row in rows:
        score = float(row["fitness"])
        require(
            math.isfinite(score) and 1 <= score <= 5,
            "Invalid Gemma fitness"
            if plan.get("construct")
            else "Invalid creativity fitness",
        )
        require(float(row["score_value"]) == score, "Stored score and fitness differ")
        if plan.get("construct"):
            require(
                row["score_name"] == f"Gemma4{plan['construct'].title()}",
                "Stored score belongs to another construct",
            )
        require(not row["fitness_parse_error"].strip(), "Parse-invalid fitness")
        require(
            row["file_name"].startswith(f"g{int(row['generation'])}_id"),
            "Filename generation disagrees with CSV",
        )
    expected_members = {f"/images/{name}.JPEG" for name in names}
    expected_members.update(f"/noise/{name}.pt" for name in names)
    with zipfile.ZipFile(paths["zip"]) as archive:
        members = archive.namelist()
        require(len(members) == len(set(members)), "Duplicate ZIP member")
        require(
            set(members) == expected_members, "CSV, images and noise coverage differ"
        )
        require(archive.testzip() is None, "ZIP CRC integrity failure")
        for name in names:
            with Image.open(io.BytesIO(archive.read(f"/images/{name}.JPEG"))) as image:
                require(
                    image.format == "JPEG" and image.size == (1024, 1024),
                    "Unexpected image format or dimensions",
                )
                image.load()
    return {
        "status": "complete",
        "checked_at": now(),
        "key": task["key"],
        "seed": task["seed"],
        "prompt": task["prompt"],
        "task": task,
        "rows": 3100,
        "generations": list(range(31)),
        "files": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in paths.items()
        },
    }


@contextmanager
def lane_lock(campaign, lane):
    # PanFS flock is already used by the project's evaluation workers. The lock
    # is released by the OS even when the worker is killed.
    import fcntl

    path = campaign / "locks" / f"lane-{lane}.lock"
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def completion_path(campaign, task):
    return campaign / "completed" / f"{task['key']}.json"


def run_task(plan, campaign, task):
    complete = completion_path(campaign, task)
    if complete.exists():
        audit_artifacts(plan, read_json(complete)["task"])
        print(f"RETAIN COMPLETE: {task['key']}", flush=True)
        return 0
    attempts = campaign / "attempts" / task["key"]
    attempts.mkdir(parents=True, exist_ok=True)
    previous = sorted(attempts.glob("attempt-*.json"))
    if previous:
        record = read_json(previous[-1])
        # A killed wrapper may leave a complete, closed artifact without having
        # written its receipt. Audit it before allocating another attempt.
        try:
            audit = audit_artifacts(plan, record["task"])
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            stem = Path(record["task"]["output_directory"]) / record["task"]["run_name"]
            failure = Path(f"{stem}.fitness_failures.jsonl")
            require(
                not failure.exists() or failure.stat().st_size == 0,
                f"{task['key']}: invalid fitness; investigate instead of retrying",
            )
            require(
                record.get("returncode") in (None, -9, 137),
                f"{task['key']}: application/audit failure requires investigation",
            )
        else:
            atomic_json(complete, audit)
            print(f"RECOVER COMPLETE ARTIFACT: {task['key']}", flush=True)
            return 0
    for number in range(len(previous) + 1, plan["max_attempts"] + 1):
        current = attempt_task(task, number)
        receipt = attempts / f"attempt-{number:02d}.json"
        require(
            not Path(current["output_directory"]).exists(),
            "Attempt output already exists",
        )
        record = {
            "status": "started",
            "started_at": now(),
            "task": current,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        }
        full.save_json(receipt, record)
        current_plan = dict(plan, seed=task["seed"])
        print(f"START {task['key']} attempt {number}, generations 0-30", flush=True)
        try:
            code = full.execute_task(current_plan, current)
            record.update(returncode=code, ended_at=now(), status="failed")
            if code == 0:
                audit = audit_artifacts(plan, current)
                atomic_json(complete, audit)
                record["status"] = "complete"
                print(
                    f"TASK COMPLETE: {task['key']} 3100 rows, 31 generations",
                    flush=True,
                )
        except Exception as exc:
            record.update(status="failed", error=str(exc), ended_at=now(), returncode=1)
            code = 1
            traceback.print_exc()
        atomic_json(receipt, record)
        if code == 0:
            return 0
        if code not in (-9, 137):
            return 1
        stem = Path(current["output_directory"]) / current["run_name"]
        failure = Path(f"{stem}.fitness_failures.jsonl")
        if failure.exists() and failure.stat().st_size:
            print("Invalid fitness was recorded; automatic retry refused", flush=True)
            return 1
        print(
            f"Signal 9 interrupted {task['key']}; preserving this attempt. Restarting "
            "from the same seed if an attempt remains.",
            flush=True,
        )
    print(
        f"FAILED: {task['key']} exhausted {plan['max_attempts']} attempts", flush=True
    )
    return 1


def run_lane(plan, campaign, lane):
    require(lane in range(group_count(plan) * 2), "Invalid lane")
    failed = False
    with lane_lock(campaign, lane):
        for task in (item for item in plan["tasks"] if item["lane"] == lane):
            try:
                failed = bool(run_task(plan, campaign, task)) or failed
            except Exception:
                traceback.print_exc()
                failed = True
    return int(failed)


def wait_for_groups(batch, deadline, poll=15, groups=3):
    """Every parent, including a failed parent, stays alive until all workers stop."""
    announced = None
    while True:
        paths = [batch / f"group-{index}.json" for index in range(groups)]
        count = sum(path.exists() for path in paths)
        if count != announced:
            print(
                f"EXIT BARRIER: {count}/{groups} allocations finished their workers/audits",
                flush=True,
            )
            announced = count
        if count == groups:
            records = [read_json(path) for path in paths]
            require(
                [record["group"] for record in records] == list(range(groups)),
                "Unexpected barrier records",
            )
            return any(record["failed"] for record in records)
        if time.time() >= deadline:
            raise TimeoutError(
                "Slurm time limit approaching before every group finished; "
                "campaign incomplete, use recover after all jobs end"
            )
        time.sleep(poll)


def run_group(campaign, digest, batch_id, index):
    # Establish barrier size before launching workers or writing a group receipt.
    plan = verified_plan(campaign, digest)
    groups = group_count(plan)
    require(index in range(groups), "Invalid group")
    require(
        os.environ.get("SLURM_ARRAY_TASK_ID") == str(index), "Slurm array index differs"
    )
    batch = campaign / "batches" / batch_id
    processes = []
    failed = False
    try:
        plan = verified_plan(campaign, digest)
        # Slurm can start a job before sbatch returns its ID to the submitter.
        receipt_deadline = time.monotonic() + 120
        while not (batch / "submission.json").exists():
            require(
                time.monotonic() < receipt_deadline, "Submission receipt did not arrive"
            )
            time.sleep(1)
        receipt = read_json(batch / "submission.json")
        require(
            str(receipt["job_id"]) == os.environ.get("SLURM_ARRAY_JOB_ID"),
            "Submission does not match this Slurm array",
        )
        for lane in (index * 2, index * 2 + 1):
            command = [
                "srun",
                "--exclusive",
                "--exact",
                "--nodes=1",
                "--ntasks=1",
                "--gres=gpu:1",
                "--cpus-per-task=8",
                "--mem=64G",
                "--export=ALL",
                f"--output={batch}/lane-{lane}.out",
                f"--error={batch}/lane-{lane}.err",
                plan["python"],
                "-u",
                str(Path(plan["project"]) / "cluster/thesis_replicates.py"),
                "run-lane",
                "--campaign",
                str(campaign),
                "--sha256",
                digest,
                "--index",
                str(lane),
            ]
            processes.append(subprocess.Popen(command))
    except Exception:
        traceback.print_exc()
        failed = True
    finally:
        # Never let a launch failure release a parent while its first child runs.
        for process in processes:
            failed = bool(process.wait()) or failed
    atomic_json(
        batch / f"group-{index}.json",
        {"group": index, "failed": failed, "ended_at": now()},
    )
    deadline = int(os.environ.get("SLURM_JOB_END_TIME", time.time() + 5 * 86400)) - 120
    failed = wait_for_groups(batch, deadline, 15, groups) or failed
    if index == 0:
        try:
            if not failed:
                plan = verified_plan(campaign, digest)
                reports = [
                    read_json(completion_path(campaign, task)) for task in plan["tasks"]
                ]
                require(
                    all(report["status"] == "complete" for report in reports),
                    "Incomplete task",
                )
                atomic_json(
                    campaign / "completion.json",
                    {
                        "status": "complete",
                        "checked_at": now(),
                        "tasks": reports,
                        "total_rows": len(plan["tasks"]) * 3100,
                        "batch": batch_id,
                    },
                )
        except Exception:
            traceback.print_exc()
            failed = True
        atomic_json(batch / "release.json", {"failed": failed, "time": now()})
    # Save the campaign manifest before releasing any parent allocation.
    while not (batch / "release.json").exists():
        require(time.time() < deadline, "Coordinator did not release the barrier")
        time.sleep(5)
    failed = read_json(batch / "release.json")["failed"]
    if not failed:
        print(
            f"ALL {len(plan['tasks'])} PROMPT RUNS COMPLETE: generations 0-30 validated",
            flush=True,
        )
    else:
        print(
            "CAMPAIGN INCOMPLETE: inspect lane logs and status before recovery",
            flush=True,
        )
    return int(failed)


def sbatch_command(plan, campaign, batch, digest):
    groups = group_count(plan)
    return [
        "sbatch",
        "--parsable",
        "--account=dldevel",
        "--partition=gpu2",
        "--qos=gpu2",
        f"--array=0-{groups - 1}%{groups}",
        "--nodes=1",
        "--ntasks=2",
        "--gres=gpu:2",
        "--cpus-per-task=8",
        "--mem=128G",
        "--time=5-00:00:00",
        "--no-requeue",
        "--export=ALL",
        f"--job-name=gemma-{plan.get('construct', 'replicates')}",
        "--nodelist=gpu30-022",
        f"--chdir={plan['project']}",
        f"--output={batch}/job-%A_%a.out",
        f"--error={batch}/job-%A_%a.err",
        str(Path(plan["project"]) / "cluster/run_thesis_replicates.sbatch"),
        plan["python"],
        str(Path(plan["project"]) / "cluster/thesis_replicates.py"),
        str(campaign),
        digest,
        batch.name,
    ]


def submit_batch(plan, campaign):
    batches = campaign / "batches"
    batches.mkdir(exist_ok=True)
    batch = batches / f"batch-{len(list(batches.iterdir())) + 1:03d}"
    batch.mkdir()  # Atomic claim, including concurrent recovery attempts.
    command = sbatch_command(plan, campaign, batch, sha256(campaign / "plan.json"))
    full.save_json(batch / "request.json", {"command": command, "created_at": now()})
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")
    }
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    receipt = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    match = re.fullmatch(r"(\d+)(?:;[^\s;]+)?", result.stdout.strip())
    if result.returncode == 0 and match:
        receipt["job_id"] = match.group(1)
    atomic_json(batch / "submission.json", receipt)
    require(
        "job_id" in receipt,
        f"No confirmed job ID; inspect {batch} and squeue before retrying",
    )
    print(
        f"Submitted array {receipt['job_id']}: {group_count(plan)} jobs x 2 GPUs "
        f"= {group_count(plan) * 2} GPUs maximum\n"
        f"Campaign: {campaign}\nLogs: {batch}\n"
        f"Runs use seeds {plan['seeds']} across {group_count(plan) * 2} GPU lanes. "
        "All allocations wait at the exit barrier."
    )


def recover(campaign):
    # Lock queue inspection and sbatch together against concurrent recoveries.
    lock = campaign / ".recovery-submit-lock"
    lock.mkdir()
    try:
        plan = verified_plan(campaign)
        batches = sorted((campaign / "batches").glob("batch-*"))
        require(batches, "No previous submission")
        result = subprocess.run(
            ["squeue", "--noheader", "--user", getpass.getuser(), "--format=%F"],
            capture_output=True,
            text=True,
            check=True,
        )
        # A full user queue still works after individual old IDs are purged.
        active = set(result.stdout.split())
        for batch in batches:
            require(
                (batch / "submission.json").exists(),
                f"Missing submission receipt in {batch}; inspect before recovery",
            )
            receipt = read_json(batch / "submission.json")
            require(
                "job_id" in receipt,
                f"Ambiguous submission in {batch}; inspect before recovery",
            )
            require(
                str(receipt["job_id"]) not in active,
                f"Job {receipt['job_id']} is still active; recovery refused",
            )
        require(
            not all(completion_path(campaign, task).exists() for task in plan["tasks"]),
            "All runs already have completion audits; use verify instead",
        )
        submit_batch(plan, campaign)
    finally:
        lock.rmdir()


def status(campaign, verify=False):
    plan = read_json(campaign / "plan.json")
    complete = 0
    for task in plan["tasks"]:
        path = completion_path(campaign, task)
        if path.exists():
            report = read_json(path)
            if verify:
                checked = audit_artifacts(plan, report["task"])
                require(
                    checked["files"] == report["files"],
                    "Artifact hashes changed after audit",
                )
            complete += 1
            print(f"{task['key']:26} COMPLETE 3100/3100 rows; generations 0-30")
        else:
            attempts = sorted(
                (campaign / "attempts" / task["key"]).glob("attempt-*.json")
            )
            detail = "not started"
            if attempts:
                record = read_json(attempts[-1])
                current = record["task"]
                csv_path = (
                    Path(current["output_directory"]) / f"{current['run_name']}.csv"
                )
                if csv_path.exists():
                    with csv_path.open(encoding="utf-8", newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    counts = Counter(row.get("generation") for row in rows)
                    finished = sum(counts.get(str(gen)) == 100 for gen in range(31))
                    detail = f"{len(rows)}/3100 rows; {finished}/31 full generations"
                else:
                    detail = "no saved generations yet"
                detail += (
                    f"; last attempt {record['status']} (check Slurm for live state)"
                )
            print(f"{task['key']:26} INCOMPLETE {detail}")
    print(f"Verified completion receipts: {complete}/{len(plan['tasks'])} prompt runs")
    if verify:
        require(complete == len(plan["tasks"]), "Campaign is not complete")
        print(
            f"VERIFY OK: {len(plan['tasks']) * 3100:,} observations; "
            f"all {len(plan['tasks'])} runs have generations 0-30"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("preview", "submit"):
        sub = commands.add_parser(action)
        sub.add_argument("--runtime-root", type=Path, required=True)
        sub.add_argument("--campaign", required=True)
        sub.add_argument("--seeds", nargs=2, type=int, default=[2026, 2027])
    for action in ("preview-construct", "submit-construct"):
        sub = commands.add_parser(action)
        sub.add_argument("--runtime-root", type=Path, required=True)
        sub.add_argument("--campaign", required=True)
        sub.add_argument("--construct", choices=list(ADJECTIVES), required=True)
        sub.add_argument("--gpus", type=int, choices=(6, 8), default=6)
    for action in ("status", "verify", "recover", "run-group", "run-lane"):
        sub = commands.add_parser(action)
        sub.add_argument("--campaign", type=Path, required=True)
        if action in ("run-group", "run-lane"):
            sub.add_argument("--sha256", required=True)
            sub.add_argument("--index", type=int, required=True)
        if action == "run-group":
            sub.add_argument("--batch", required=True)
    args = parser.parse_args()
    if args.action in ("preview", "submit", "preview-construct", "submit-construct"):
        project = Path(__file__).resolve().parents[1]
        if args.action.endswith("-construct"):
            plan = make_construct_plan(
                project,
                args.runtime_root.resolve(),
                args.campaign,
                [2025, 2026, 2027],
                full.clean_commit(project),
                args.construct,
                args.gpus,
            )
        else:
            plan = make_plan(
                project,
                args.runtime_root.resolve(),
                args.campaign,
                args.seeds,
                full.clean_commit(project),
            )
        campaign = args.runtime_root / "gemma_ga_outputs/submissions" / args.campaign
        print(
            f"6 prompts x {len(plan['seeds'])} seeds x 100 images x 31 generations "
            f"= {len(plan['tasks']) * 3100:,} observations\n"
            f"Seeds: {plan['seeds']}; {group_count(plan)} separate jobs, "
            "2 GPUs each; five-day limit\n"
            f"Generator: {plan['generator_commit']}\nCampaign: {campaign}"
        )
        if plan.get("construct"):
            print(
                f"Fitness condition: {plan['construct']}\n"
                f"Exact Gemma question: {plan['scoring_prompt']}"
            )
        if args.action.startswith("submit"):
            campaign.mkdir(parents=True, exist_ok=False)
            full.save_json(campaign / "plan.json", plan)
            for key in (
                "HF_HOME",
                "HF_HUB_CACHE",
                "PIP_CACHE_DIR",
                "CONDA_PKGS_DIRS",
                "TMPDIR",
            ):
                Path(plan["environment"][key]).mkdir(parents=True, exist_ok=True)
            submit_batch(plan, campaign)
    elif args.action == "run-group":
        return run_group(args.campaign, args.sha256, args.batch, args.index)
    elif args.action == "run-lane":
        return run_lane(
            verified_plan(args.campaign, args.sha256), args.campaign, args.index
        )
    elif args.action == "recover":
        recover(args.campaign)
    else:
        status(args.campaign, verify=args.action == "verify")
    return 0


if __name__ == "__main__":
    os.umask(0o002)
    raise SystemExit(main())
