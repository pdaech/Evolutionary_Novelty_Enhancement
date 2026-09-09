import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "cluster" / "thesis_full.py"
SPEC = importlib.util.spec_from_file_location("thesis_full", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def plan(tmp_path, workers=1):
    return launcher.make_plan(
        tmp_path / "code",
        tmp_path / "runtime",
        "full-test",
        2025,
        workers,
        "gpu30-022",
        "a" * 40,
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_smoke_environment_cannot_override_full_run(tmp_path, monkeypatch, workers):
    for key, value in {
        "NUM_GENERATIONS": "1",
        "POPULATION_SIZE": "4",
        "RUN_SEED": "123",
        "GENERATION_PROMPT": "cat",
        "SDXL_GUIDANCE_SCALE": "7.0",
        "GEMMA_IMAGE_TOKEN_BUDGET": "280",
        "SDXL_BATCH_SIZE": "1",
    }.items():
        monkeypatch.setenv(key, value)
    configuration = plan(tmp_path, workers)
    assert sum(task["expected_records"] for task in configuration["tasks"]) == 18600
    assert len({task["output_directory"] for task in configuration["tasks"]}) == 6
    for task in configuration["tasks"]:
        command = launcher.inference_command(configuration, task)
        options = dict(zip(command[3::2], command[4::2], strict=True))
        assert options["--population_size"] == "100"
        assert options["--num_generations"] == "30"
        assert options["--seed"] == "2025"
        assert options["--prompt"] == launcher.PROMPTS[task["index"]][1]
        assert options["--sdxl_guidance_scale"] == "7.5"
        assert options["--gemma_image_token_budget"] == "140"
        assert options["--batch_size"] == "2"


@pytest.mark.parametrize(
    "workers,array,memory", [(1, "0-5%4", "64G"), (2, "0-2%3", "128G")]
)
def test_resource_requests_cover_isolated_workers(tmp_path, workers, array, memory):
    configuration = plan(tmp_path, workers)
    command = launcher.sbatch_command(configuration, tmp_path / "plan.json", "digest")
    assert f"--array={array}" in command
    assert f"--gres=gpu:{workers}" in command
    assert f"--ntasks={workers}" in command
    assert f"--mem={memory}" in command
    assert "--partition=gpu2" in command
    assert "--nodelist=gpu30-022" in command


def test_manifest_precedes_submission_and_duplicate_is_refused(tmp_path, monkeypatch):
    configuration = plan(tmp_path)
    directory = (
        Path(configuration["environment"]["BASE_PATH"]) / "submissions/full-test"
    )
    calls = []
    monkeypatch.setenv("SBATCH_ARRAY_INX", "0-1")

    def sbatch(command, **kwargs):
        assert (directory / "plan.json").is_file()
        assert (directory / "submission_request.json").is_file()
        assert "SBATCH_ARRAY_INX" not in kwargs["env"]
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="1234;cluster\n", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", sbatch)
    launcher.submit(configuration)
    assert json.loads((directory / "submission.json").read_text())["job_id"] == "1234"
    assert len((directory / "tasks.tsv").read_text().splitlines()) == 7
    with pytest.raises(FileExistsError):
        launcher.submit(configuration)
    assert len(calls) == 1


def test_submission_failure_preserves_receipt_without_retry(tmp_path, monkeypatch):
    configuration = plan(tmp_path)
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="QOS limit\n")

    monkeypatch.setattr(launcher.subprocess, "run", sbatch)
    with pytest.raises(RuntimeError, match="No confirmed job ID"):
        launcher.submit(configuration)
    receipt = (
        Path(configuration["environment"]["BASE_PATH"])
        / "submissions/full-test/submission.json"
    )
    assert json.loads(receipt.read_text())["stderr"] == "QOS limit\n"
    with pytest.raises(FileExistsError):
        launcher.submit(configuration)
    assert len(calls) == 1


def test_manifest_and_source_changes_are_rejected(tmp_path, monkeypatch):
    manifest = tmp_path / "plan.json"
    configuration = plan(tmp_path)
    launcher.save_json(manifest, configuration)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(launcher, "clean_commit", lambda project: "b" * 40)
    with pytest.raises(ValueError, match="checkout changed"):
        launcher.read_verified_plan(manifest, digest)
    monkeypatch.setattr(launcher, "clean_commit", lambda project: "a" * 40)
    assert launcher.read_verified_plan(manifest, digest) == configuration
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        launcher.read_verified_plan(manifest, digest)


def test_workers_launch_before_wait_and_propagate_failure(tmp_path, monkeypatch):
    configuration = plan(tmp_path, workers=2)
    events = []

    class Process:
        def __init__(self, command):
            self.index = int(command[-1])
            assert "--gres=gpu:1" in command
            assert "--ntasks=1" in command
            assert "--mem=64G" in command
            assert "--exclusive" in command
            events.append(("launch", self.index))

        def wait(self):
            events.append(("wait", self.index))
            return 1 if self.index == 0 else 0

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    assert launcher.run_group(configuration, tmp_path / "plan.json", "hash", 0) == 1
    assert events == [("launch", 0), ("launch", 1), ("wait", 0), ("wait", 1)]


def test_wrong_gpu_visibility_stops_before_inference(tmp_path, monkeypatch):
    configuration = plan(tmp_path)
    calls = []

    def check(command, **kwargs):
        calls.append("check")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(launcher.subprocess, "run", check)
    monkeypatch.setattr(
        launcher.subprocess, "call", lambda *a, **kw: calls.append("inference")
    )
    with pytest.raises(subprocess.CalledProcessError):
        launcher.execute_task(configuration, configuration["tasks"][0])
    assert calls == ["check"]


def test_status_counts_full_run_and_flags_smoke_config(tmp_path, capsys):
    configuration = plan(tmp_path)
    manifest = tmp_path / "plan.json"
    launcher.save_json(manifest, configuration)
    full, smoke = configuration["tasks"][:2]
    for task, size, generations in ((full, 100, 30), (smoke, 4, 1)):
        directory = Path(task["output_directory"])
        directory.mkdir(parents=True)
        stem = directory / task["run_name"]
        launcher.save_json(
            Path(f"{stem}.json"),
            {
                "population_size": size,
                "num_generations": generations,
                "prompt": task["prompt"],
            },
        )
        with Path(f"{stem}.csv").open("w", encoding="utf-8") as handle:
            handle.write("generation\n")
            for gen in range(generations + 1):
                handle.write(f"{gen}\n" * size)
    launcher.status(manifest)
    output = capsys.readouterr().out
    assert "3100/3100" in output
    assert "31/31" in output
    assert "8/3100" in output
    assert "MISMATCH" in output
