"""Submit and audit a small, complete Gemma-construct evolutionary run."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import thesis_full as full
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gemma_constructs import ADJECTIVES, scoring_prompt  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_plan(project, runtime, campaign_name, construct, commit):
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,59}", campaign_name),
        "Campaign name must contain 1-60 letters, digits, - or _",
    )
    require(construct in ADJECTIVES, "Unknown fitness construct")
    plan = full.make_plan(project, runtime, f"{campaign_name}-base", 2025, 1, "gpu30-022", commit)
    experiment = f"{campaign_name}-cat-p4-g1-seed2025"
    run_name = f"gemma-{construct}-smoke_{experiment}"
    task = dict(
        plan["tasks"][0],
        key=f"{construct}-cat-seed2025",
        experiment_id=experiment,
        run_name=run_name,
        output_directory=str(
            Path(plan["environment"]["BASE_PATH"]) / "results/simulations" / run_name
        ),
        prompt="a cat",
        seed=2025,
        expected_records=8,
    )
    plan.update(
        kind="gemma-construct-smoke",
        schema_version=1,
        campaign=campaign_name,
        construct=construct,
        scoring_prompt=scoring_prompt(construct),
        run_id=f"gemma-{construct}-smoke",
        options=dict(
            full.OPTIONS,
            population_size=4,
            num_generations=1,
            evaluator="gemma-construct",
            gemma_fitness_construct=construct,
        ),
        tasks=[task],
    )
    return plan


def verified_plan(campaign, digest=None):
    path = campaign / "plan.json"
    if digest is not None:
        require(sha256(path) == digest, "Smoke plan changed after submission")
    plan = read_json(path)
    require(
        plan.get("kind") == "gemma-construct-smoke" and plan.get("schema_version") == 1,
        "Unexpected smoke plan",
    )
    construct = plan["construct"]
    require(construct in ADJECTIVES, "Unknown fitness construct")
    require(
        plan["scoring_prompt"] == scoring_prompt(construct),
        "Smoke scoring question changed",
    )
    require(
        plan["options"]
        == dict(
            full.OPTIONS,
            population_size=4,
            num_generations=1,
            evaluator="gemma-construct",
            gemma_fitness_construct=construct,
        ),
        "Smoke scientific settings changed",
    )
    require(plan["run_id"] == f"gemma-{construct}-smoke", "Smoke run ID changed")
    require(len(plan["tasks"]) == 1, "Smoke must contain one task")
    require(
        full.clean_commit(plan["project"]) == plan["generator_commit"],
        "Generator checkout changed after smoke submission",
    )
    return plan


def artifact_paths(task):
    stem = Path(task["output_directory"]) / task["run_name"]
    return {extension: Path(f"{stem}.{extension}") for extension in ("json", "csv", "zip")}


def audit(plan):
    task = plan["tasks"][0]
    paths = artifact_paths(task)
    stem = Path(task["output_directory"]) / task["run_name"]
    failure = Path(f"{stem}.fitness_failures.jsonl")
    require(
        not failure.exists() or failure.stat().st_size == 0,
        "Smoke recorded invalid fitness responses",
    )
    config = read_json(paths["json"])
    for key, expected in {
        "prompt": "a cat",
        "random_seed": 2025,
        "population_size": 4,
        "num_generations": 1,
        "generation_code_commit": plan["generator_commit"],
        "fitness_aggregation": "latest",
        "evaluator": "GemmaConstructEvaluator",
    }.items():
        require(config.get(key) == expected, f"Unexpected smoke config: {key}")
    evaluator = config["evaluator_config"]
    require(evaluator["construct"] == plan["construct"], "Wrong fitness construct")
    require(evaluator["prompt"] == plan["scoring_prompt"], "Wrong scoring question")
    require(
        evaluator["prompt_sha256"]
        == hashlib.sha256(plan["scoring_prompt"].encode("utf-8")).hexdigest(),
        "Wrong scoring question hash",
    )
    require(
        evaluator["model_id"] == plan["options"]["gemma_model"]
        and evaluator["requested_revision"] == plan["options"]["gemma_revision"],
        "Wrong model revision",
    )
    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 8, f"Expected 8 smoke rows, found {len(rows)}")
    require(
        Counter(int(row["generation"]) for row in rows) == {0: 4, 1: 4},
        "Smoke generations are incomplete",
    )
    require(
        len({(row["generation"], row["candidate_id"]) for row in rows}) == 8,
        "Duplicate smoke candidate",
    )
    names = [row["file_name"] for row in rows]
    require(len(set(names)) == 8, "Duplicate smoke image name")
    for row in rows:
        score = float(row["fitness"])
        require(math.isfinite(score) and 1 <= score <= 5, "Invalid smoke fitness")
        require(float(row["score_value"]) == score, "Fitness and score differ")
        require(
            row["score_name"] == f"Gemma4{plan['construct'].title()}",
            "Wrong scoring condition",
        )
        require(not row["fitness_parse_error"].strip(), "Parse-invalid smoke score")
        require(
            row["file_name"].startswith(f"g{row['generation']}_id"),
            "Smoke filename generation mismatch",
        )
    expected = {f"/images/{name}.JPEG" for name in names}
    expected.update(f"/noise/{name}.pt" for name in names)
    with zipfile.ZipFile(paths["zip"]) as archive:
        members = archive.namelist()
        require(len(members) == len(set(members)), "Duplicate smoke ZIP member")
        require(set(members) == expected, "Smoke image/noise archive coverage differs")
        require(archive.testzip() is None, "Smoke ZIP CRC failure")
        for name in names:
            with Image.open(io.BytesIO(archive.read(f"/images/{name}.JPEG"))) as image:
                require(
                    image.format == "JPEG" and image.size == (1024, 1024),
                    "Unexpected smoke image format or size",
                )
                image.load()
    return {
        "status": "complete",
        "construct": plan["construct"],
        "rows": 8,
        "generations": [0, 1],
        "files": {
            extension: {"path": str(path), "sha256": sha256(path)}
            for extension, path in paths.items()
        },
    }


def sbatch_command(plan, campaign, digest):
    return [
        "sbatch",
        "--parsable",
        "--account=dldevel",
        "--partition=gpu2",
        "--qos=gpu2",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=02:00:00",
        "--no-requeue",
        "--export=ALL",
        "--job-name=gemma-construct-smoke",
        "--nodelist=gpu30-022",
        f"--chdir={plan['project']}",
        f"--output={campaign}/job-%j.out",
        f"--error={campaign}/job-%j.err",
        str(Path(plan["project"]) / "cluster/run_gemma_construct_smoke.sbatch"),
        plan["python"],
        str(Path(plan["project"]) / "cluster/smoke_fitness_construct.py"),
        str(campaign),
        digest,
    ]


def submit(plan, campaign):
    task = plan["tasks"][0]
    require(
        not Path(task["output_directory"]).exists(),
        "Smoke result directory already exists",
    )
    campaign.mkdir(parents=True, exist_ok=False)
    full.save_json(campaign / "plan.json", plan)
    for key in ("HF_HOME", "HF_HUB_CACHE", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS", "TMPDIR"):
        Path(plan["environment"][key]).mkdir(parents=True, exist_ok=True)
    command = sbatch_command(plan, campaign, sha256(campaign / "plan.json"))
    full.save_json(campaign / "submission_request.json", {"command": command})
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    receipt = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    match = re.fullmatch(r"(\d+)(?:;[^\s;]+)?", result.stdout.strip())
    if result.returncode == 0 and match:
        receipt["job_id"] = match.group(1)
    full.save_json(campaign / "submission.json", receipt)
    require("job_id" in receipt, f"Ambiguous smoke submission: inspect {campaign}")
    print(f"SMOKE JOB {receipt['job_id']}\nCampaign: {campaign}")


def run_job(campaign, digest):
    plan = verified_plan(campaign, digest)
    deadline = time.monotonic() + 120
    while not (campaign / "submission.json").exists():
        require(time.monotonic() < deadline, "Smoke submission receipt did not arrive")
        time.sleep(1)
    receipt = read_json(campaign / "submission.json")
    require(
        str(receipt["job_id"]) == os.environ.get("SLURM_JOB_ID"),
        "Slurm job does not match smoke submission",
    )
    task = plan["tasks"][0]
    require(not Path(task["output_directory"]).exists(), "Smoke output already exists")
    require(full.execute_task(plan, task) == 0, "Smoke inference failed")
    report = audit(plan)
    full.save_json(campaign / "completion.json", report)
    print("SMOKE OK: 8 valid scores, generations 0 and 1, images and noise verified")


def status(campaign, verify=False):
    plan = verified_plan(campaign)
    completion = campaign / "completion.json"
    if not completion.exists():
        print(f"INCOMPLETE: inspect Slurm and {campaign}/job-*.err")
        return
    report = read_json(completion)
    if verify:
        checked = audit(plan)
        require(checked["files"] == report["files"], "Smoke artifacts changed")
    print(f"SMOKE OK: {plan['construct']}, 8/8 valid scores, 4 images in each generation 0 and 1")


def recover_audit(campaign):
    """Audit c45162a's existing output without editing its plan or rerunning inference."""
    request = read_json(campaign / "submission_request.json")
    plan = verified_plan(campaign, request["command"][-1])
    require(
        plan["generator_commit"] == "c45162a93971be8ec3d565356915b29d067c8552",
        "Audit recovery only applies to the original smoke path bug",
    )
    task = plan["tasks"][0]
    require(
        Path(task["output_directory"]) == campaign / "results" / task["run_name"],
        "Not the known smoke output path mismatch",
    )
    command = full.inference_command(plan, task)
    options = dict(zip(command[3::2], command[4::2], strict=True))
    require(
        task["run_name"] == f"{options['--id']}_{options['--experiment_id']}",
        "Smoke output name differs from generator command",
    )
    actual = Path(plan["environment"]["BASE_PATH"]) / options["--directory"] / task["run_name"]
    corrected_task = dict(task, output_directory=str(actual))
    report = audit(dict(plan, tasks=[corrected_task]))
    report.update(
        recovery="smoke-output-path-c45162a",
        original_plan_sha256=sha256(campaign / "plan.json"),
        generator_commit=plan["generator_commit"],
        declared_output_directory=task["output_directory"],
        actual_output_directory=str(actual),
        audit_script_sha256=sha256(Path(__file__)),
    )
    destination = campaign / "audit-recovery.json"
    if destination.exists():
        require(read_json(destination) == report, "Recovered smoke audit or artifacts changed")
    else:
        full.save_json(destination, report)
    print(f"RECOVERED SMOKE OK: {plan['construct']}, 8/8 valid scores, generations 0 and 1")
    print(f"Existing images and noise verified; no inference performed.\nAudit: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("preview", "submit"):
        sub = commands.add_parser(action)
        sub.add_argument("--runtime-root", type=Path, required=True)
        sub.add_argument("--campaign", required=True)
        sub.add_argument("--construct", choices=list(ADJECTIVES), required=True)
    for action in ("run-job", "status", "verify", "recover-audit"):
        sub = commands.add_parser(action)
        sub.add_argument("--campaign", type=Path, required=True)
        if action == "run-job":
            sub.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.action in ("preview", "submit"):
        project = Path(__file__).resolve().parents[1]
        runtime = args.runtime_root.resolve()
        plan = make_plan(
            project, runtime, args.campaign, args.construct, full.clean_commit(project)
        )
        campaign = runtime / "gemma_ga_outputs/smoke" / args.campaign
        print(
            f"SMOKE: {args.construct}, 'a cat', seed 2025, population 4, "
            "generations 0-1, 8 ratings, one 80GB GPU, two-hour limit\n"
            f"Question: {plan['scoring_prompt']}\nCampaign: {campaign}"
        )
        if args.action == "submit":
            submit(plan, campaign)
    elif args.action == "run-job":
        run_job(args.campaign, args.sha256)
    elif args.action == "recover-audit":
        recover_audit(args.campaign)
    else:
        status(args.campaign, verify=args.action == "verify")


if __name__ == "__main__":
    os.umask(0o002)
    main()
