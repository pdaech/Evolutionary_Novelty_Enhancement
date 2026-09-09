"""Full six-prompt submission. Uses only the standard library on login nodes."""

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

PROMPTS = (
    ("cat", "a cat"),
    ("dog", "a dog"),
    ("chair", "a chair"),
    ("coffee-mug", "a coffee mug"),
    ("boredom", "boredom"),
    ("creativity", "creativity"),
)
OPTIONS = {
    "population_size": 100,
    "num_generations": 30,
    "batch_size": 2,
    "sdxl_num_inference_steps": 50,
    "sdxl_guidance_scale": 7.5,
    "sdxl_revision": "462165984030d82259a11f4367a4eed129e94a7b",
    "evaluator": "gemma-creativity",
    "gemma_model": "google/gemma-4-26B-A4B-it",
    "gemma_revision": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
    "gemma_max_new_tokens": 64,
    "gemma_image_token_budget": 140,
    "gemma_batch_size": 2,
}


def clean_commit(project):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(project), *args], text=True
        ).strip()

    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError(
            "Generator checkout must be clean before submission or execution"
        )
    return git("rev-parse", "HEAD")


def make_plan(project, runtime, campaign, seed, workers, node, commit):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", campaign):
        raise ValueError("Campaign name: use 1-80 letters, digits, - or _")
    if workers not in (1, 2) or not 0 <= seed < 2**32:
        raise ValueError("Choose 1-2 workers and a seed between 0 and 2**32-1")
    if node and not re.fullmatch(r"[A-Za-z0-9_.-]+", node):
        raise ValueError("Node must be a single Slurm node name")
    base = runtime / "gemma_ga_outputs"
    tasks = []
    for index, (slug, prompt) in enumerate(PROMPTS):
        experiment = f"{campaign}-{slug}-p100-g30-seed{seed}"
        name = f"gemma-creativity_{experiment}"
        tasks.append(
            {
                "index": index,
                "array_index": index // workers,
                "prompt": prompt,
                "experiment_id": experiment,
                "run_name": name,
                "output_directory": str(base / "results" / "simulations" / name),
                "expected_records": 3100,
            }
        )
    return {
        "schema_version": 1,
        "campaign": campaign,
        "project": str(project),
        "python": str(Path(sys.executable).absolute()),
        "generator_commit": commit,
        "seed": seed,
        "options": dict(OPTIONS),
        "gpus_per_job": workers,
        "node": node,
        "environment": {
            "BASE_PATH": str(base),
            "HF_HOME": str(runtime / "cache" / "huggingface"),
            "HF_HUB_CACHE": str(runtime / "cache" / "huggingface" / "hub"),
            "PIP_CACHE_DIR": str(runtime / "cache" / "pip"),
            "CONDA_PKGS_DIRS": str(runtime / "cache" / "conda-pkgs"),
            "TMPDIR": str(runtime / "tmp"),
            "PYTHONUNBUFFERED": "1",
            "GENERATION_CODE_COMMIT": commit,
        },
        "tasks": tasks,
    }


def inference_command(plan, task):
    command = [
        plan["python"],
        "-u",
        str(Path(plan["project"]) / "run.py"),
        "--id",
        "gemma-creativity",
        "--directory",
        "results/simulations",
        "--experiment_id",
        task["experiment_id"],
        "--prompt",
        task["prompt"],
        "--seed",
        str(plan["seed"]),
    ]
    for key, value in plan["options"].items():
        command.extend([f"--{key}", str(value)])
    return command


def sbatch_command(plan, manifest, digest):
    workers = plan["gpus_per_job"]
    count = len(plan["tasks"]) // workers
    command = [
        "sbatch",
        "--parsable",
        "--account=dldevel",
        "--partition=gpu2",
        "--qos=gpu2",
        f"--array=0-{count - 1}%{min(4, count)}",
        "--nodes=1",
        f"--ntasks={workers}",
        f"--gres=gpu:{workers}",
        "--cpus-per-task=8",
        f"--mem={workers * 64}G",
        "--time=3-00:00:00",
        "--no-requeue",
        "--export=ALL",
        "--job-name=gemma-thesis-full",
        f"--chdir={plan['project']}",
        f"--output={manifest.parent}/job-%A_%a.out",
        f"--error={manifest.parent}/job-%A_%a.err",
    ]
    if plan["node"]:
        command.append(f"--nodelist={plan['node']}")
    command.extend(
        [
            str(Path(plan["project"]) / "cluster" / "run_thesis_full_array.sbatch"),
            plan["python"],
            str(Path(plan["project"]) / "cluster" / "thesis_full.py"),
            str(manifest),
            digest,
        ]
    )
    return command


def print_plan(plan):
    workers = plan["gpus_per_job"]
    print(
        f"FULL RUN: 6 prompts x 100 candidates x 31 generations (0-30)\n"
        f"3,100 image records per prompt; 18,600 total; seed={plan['seed']}\n"
        "SDXL: Euler, 50 steps, guidance=7.5, batch=2\n"
        "Gemma: 140 visual tokens, batch=2, greedy creativity scoring\n"
        f"{6 // workers} jobs; {workers} GPU(s)/job; "
        f"at most {min(4, 6 // workers)} jobs concurrently\n"
        f"Per worker: 8 CPUs, 64 GB host RAM; node={plan['node'] or 'any gpu2'}\n"
        f"Generator commit: {plan['generator_commit']}",
        flush=True,
    )
    for task in plan["tasks"]:
        print(f"  [{task['array_index']}] {task['prompt']}: {task['output_directory']}")


def save_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def submit(plan):
    # Claim before sbatch: rerunning this campaign cannot silently submit twice.
    directory = (
        Path(plan["environment"]["BASE_PATH"]) / "submissions" / plan["campaign"]
    )
    for task in plan["tasks"]:
        if Path(task["output_directory"]).exists():
            raise FileExistsError(f"Existing experiment: {task['output_directory']}")
    directory.mkdir(parents=True, exist_ok=False)
    for key in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "PIP_CACHE_DIR",
        "CONDA_PKGS_DIRS",
        "TMPDIR",
    ):
        Path(plan["environment"][key]).mkdir(parents=True, exist_ok=True)
    manifest = directory / "plan.json"
    save_json(manifest, plan)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    command = sbatch_command(plan, manifest, digest)
    save_json(
        directory / "submission_request.json",
        {
            "command": command,
            "manifest_sha256": digest,
        },
    )
    print(f"Manifest: {manifest}\nSubmitting: {shlex.join(command)}", flush=True)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    try:
        result = subprocess.run(
            command, env=environment, capture_output=True, text=True
        )
    except OSError as exc:
        save_json(directory / "submission.json", {"error": str(exc)})
        raise
    receipt = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    match = re.fullmatch(r"(\d+)(?:;[^\s;]+)?", result.stdout.strip())
    if result.returncode == 0 and match:
        receipt["job_id"] = match.group(1)
    save_json(directory / "submission.json", receipt)
    if "job_id" not in receipt:
        raise RuntimeError(
            f"No confirmed job ID; inspect {directory} and squeue before retrying"
        )
    with (directory / "tasks.tsv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["job_id", "prompt", "run_name", "expected_records"])
        for task in plan["tasks"]:
            writer.writerow(
                [
                    f"{receipt['job_id']}_{task['array_index']}",
                    task["prompt"],
                    task["run_name"],
                    task["expected_records"],
                ]
            )
    print(f"Submitted array: {receipt['job_id']}\nRecord: {directory}")
    print(
        f"squeue -j {receipt['job_id']}\nsacct -j {receipt['job_id']} "
        "--format=JobID,State,Elapsed,ExitCode,MaxRSS,NodeList"
    )


def read_verified_plan(path, digest):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Submission manifest changed after submission")
    plan = json.loads(content)
    if plan["schema_version"] != 1 or plan["options"] != OPTIONS:
        raise ValueError("Unexpected full-run configuration")
    if clean_commit(plan["project"]) != plan["generator_commit"]:
        raise ValueError("Generator checkout changed while this campaign was queued")
    return plan


def execute_task(plan, task):
    environment = dict(os.environ)
    environment.update(plan["environment"])
    check = (
        "import torch\n"
        "if torch.cuda.device_count() != 1:\n"
        "    raise RuntimeError('Expected exactly one visible GPU')\n"
        "p = torch.cuda.get_device_properties(0)\n"
        "print('Allocated GPU:', p.name, 'bytes:', p.total_memory, flush=True)\n"
        "if p.total_memory < 78 * 1024**3:\n"
        "    raise RuntimeError('This profile needs an 80 GB GPU')\n"
    )
    subprocess.run([plan["python"], "-c", check], env=environment, check=True)
    command = inference_command(plan, task)
    print(f"Task {task['index']}: {shlex.join(command)}", flush=True)
    return subprocess.call(command, cwd=plan["project"], env=environment)


def run_group(plan, manifest, digest, group_index):
    tasks = [task for task in plan["tasks"] if task["array_index"] == group_index]
    if not tasks:
        raise ValueError(f"Unknown array index {group_index}")
    processes = []
    for task in tasks:
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
            f"--output={manifest.parent}/prompt-{task['index']}.out",
            f"--error={manifest.parent}/prompt-{task['index']}.err",
            plan["python"],
            "-u",
            str(Path(plan["project"]) / "cluster/thesis_full.py"),
            "run-task",
            "--manifest",
            str(manifest),
            "--sha256",
            digest,
            "--index",
            str(task["index"]),
        ]
        processes.append(subprocess.Popen(command))
    # Allow an independently running partner to finish even if one prompt fails.
    returncodes = [process.wait() for process in processes]
    print(
        f"Prompt indices {[task['index'] for task in tasks]}: exit codes {returncodes}"
    )
    return int(any(code != 0 for code in returncodes))


def status(manifest):
    plan = json.loads(manifest.read_text(encoding="utf-8"))
    receipt = manifest.parent / "submission.json"
    if receipt.exists():
        print(
            f"Array job: {json.loads(receipt.read_text()).get('job_id', 'unconfirmed')}"
        )
    print("Prompt           Saved rows  Finished generations  Config")
    for task in plan["tasks"]:
        stem = Path(task["output_directory"]) / task["run_name"]
        config_file = Path(f"{stem}.json")
        config_ok = "not started"
        if config_file.exists():
            config = json.loads(config_file.read_text(encoding="utf-8"))
            config_ok = (
                "100 / 30"
                if (
                    config.get("population_size") == 100
                    and config.get("num_generations") == 30
                    and config.get("prompt") == task["prompt"]
                )
                else "MISMATCH"
            )
        counts = {}
        csv_file = Path(f"{stem}.csv")
        if csv_file.exists():
            with csv_file.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    gen = row["generation"]
                    counts[gen] = counts.get(gen, 0) + 1
        complete = sum(counts.get(str(gen), 0) == 100 for gen in range(31))
        print(
            f"{task['prompt']:16} {sum(counts.values()):4}/3100   "
            f"{complete:2}/31                 {config_ok}"
        )
    print(f"Logs: {manifest.parent}/prompt-*.out and prompt-*.err")
    print(
        "Counts indicate progress. Confirm Slurm exit status and audit artifacts "
        "at completion."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("preview", "submit"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--runtime-root", type=Path, required=True)
        sub.add_argument("--campaign", required=True)
        sub.add_argument("--seed", type=int, default=2025)
        sub.add_argument("--gpus-per-job", type=int, choices=(1, 2), default=1)
        sub.add_argument(
            "--node",
            default="gpu30-022",
            help="Known A100 node; empty string allows any gpu2 node",
        )
    subparsers.add_parser("status").add_argument("--manifest", type=Path, required=True)
    for action in ("run-group", "run-task"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--sha256", required=True)
        sub.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.action in ("preview", "submit"):
        project = Path(__file__).resolve().parents[1]
        plan = make_plan(
            project,
            args.runtime_root.resolve(),
            args.campaign,
            args.seed,
            args.gpus_per_job,
            args.node,
            clean_commit(project),
        )
        print_plan(plan)
        if args.action == "submit":
            submit(plan)
        else:
            print("Preview only. No jobs submitted.")
    elif args.action == "status":
        status(args.manifest)
    else:
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("GPU workers must run inside a Slurm allocation")
        plan = read_verified_plan(args.manifest, args.sha256)
        if args.action == "run-group":
            return run_group(plan, args.manifest, args.sha256, args.index)
        if not 0 <= args.index < len(plan["tasks"]):
            raise ValueError("Task index out of range")
        return execute_task(plan, plan["tasks"][args.index])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
