import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
from src.gemma_constructs import ADJECTIVES, scoring_prompt

CLUSTER = Path(__file__).resolve().parents[1] / "cluster"
sys.path.insert(0, str(CLUSTER))
SPEC = importlib.util.spec_from_file_location(
    "construct_replicates", CLUSTER / "thesis_replicates.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_prompts_change_only_the_adjective():
    assert list(ADJECTIVES) == [
        "novelty",
        "unusualness",
        "uncommonness",
        "uniqueness",
        "originality",
        "innovation",
    ]
    template = (
        "How {adjective} do you find the image? Use a continuous score from 1 "
        "to 5, where higher scores mean that you find the image more "
        '{adjective}. Return only valid JSON in this exact shape: {{"score": <number>}}'
    )
    for construct, adjective in ADJECTIVES.items():
        assert scoring_prompt(construct) == template.format(adjective=adjective)
    with pytest.raises(ValueError, match="Unknown"):
        scoring_prompt("creativity")


@pytest.mark.parametrize("construct", ADJECTIVES)
def test_construct_plan_matches_three_seeds_and_isolates_fitness(tmp_path, construct):
    plan = launcher.make_construct_plan(
        tmp_path / "code",
        tmp_path / "runtime",
        f"thesis6-{construct}-v1",
        [2025, 2026, 2027],
        "a" * 40,
        construct,
    )
    assert len(plan["tasks"]) == 18
    assert len({task["output_directory"] for task in plan["tasks"]}) == 18
    assert plan["options"] == dict(
        launcher.full.OPTIONS,
        evaluator="gemma-construct",
        gemma_fitness_construct=construct,
    )
    for lane in range(6):
        tasks = [task for task in plan["tasks"] if task["lane"] == lane]
        assert [task["seed"] for task in tasks] == [2025, 2026, 2027]
        assert len({task["prompt"] for task in tasks}) == 1
        for task in tasks:
            command = launcher.full.inference_command(
                dict(plan, seed=task["seed"]), launcher.attempt_task(task, 1)
            )
            options = dict(zip(command[3::2], command[4::2], strict=True))
            assert options["--evaluator"] == "gemma-construct"
            assert options["--gemma_fitness_construct"] == construct
            assert options["--id"] == f"gemma-{construct}"
            assert options["--prompt"] == task["prompt"]
            assert options["--seed"] == str(task["seed"])
            assert options["--num_generations"] == "30"
            for attempt in (1, 2):
                current = launcher.attempt_task(task, attempt)
                current_command = launcher.full.inference_command(plan, current)
                current_options = dict(
                    zip(current_command[3::2], current_command[4::2], strict=True)
                )
                actual_output = (
                    Path(plan["environment"]["BASE_PATH"])
                    / current_options["--directory"]
                    / f"{current_options['--id']}_{current_options['--experiment_id']}"
                )
                assert Path(current["output_directory"]) == actual_output
    slurm = launcher.sbatch_command(plan, tmp_path, tmp_path / "batch-001", "hash")
    assert "--array=0-2%3" in slurm
    assert "--gres=gpu:2" in slurm
    assert f"--job-name=gemma-{construct}" in slurm


def test_plan_tampering_or_wrong_scoring_prompt_is_rejected(tmp_path, monkeypatch):
    plan = launcher.make_construct_plan(
        tmp_path / "code",
        tmp_path / "runtime",
        "thesis6-novelty-v1",
        [2025, 2026, 2027],
        "a" * 40,
        "novelty",
    )
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(launcher.full, "clean_commit", lambda _: "a" * 40)
    assert launcher.verified_plan(campaign)["construct"] == "novelty"
    plan["scoring_prompt"] = scoring_prompt("originality")
    (campaign / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt changed"):
        launcher.verified_plan(campaign)


def test_eight_gpu_dispatch_and_fourth_parent(tmp_path, monkeypatch):
    plan = launcher.make_construct_plan(
        tmp_path / "code",
        tmp_path / "runtime",
        "eight",
        [2025, 2026, 2027],
        "a" * 40,
        "novelty",
        gpus=8,
    )
    monkeypatch.setattr(launcher.full, "clean_commit", lambda _: "a" * 40)
    launcher.full.save_json(tmp_path / "plan.json", plan)
    assert launcher.verified_plan(tmp_path) == plan
    command = launcher.sbatch_command(plan, tmp_path, tmp_path / "batch", "hash")
    assert "--array=0-3%4" in command and "--gres=gpu:2" in command
    calls = []
    monkeypatch.setattr(launcher, "lane_lock", lambda *args: nullcontext())
    monkeypatch.setattr(launcher, "run_task", lambda _, __, task: calls.append(task["key"]) or 0)
    for lane in range(8):
        assert launcher.run_lane(plan, tmp_path, lane) == 0
    assert len(calls) == len(set(calls)) == 18
    assert set(calls) == {task["key"] for task in plan["tasks"]}
    assert [sum(task["lane"] == lane for task in plan["tasks"]) for lane in range(8)] == [
        3,
        3,
        2,
        2,
        2,
        2,
        2,
        2,
    ]

    batch = tmp_path / "batches/batch-001"
    launcher.atomic_json(batch / "submission.json", {"job_id": "1234"})
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "1234")
    events = []

    class Process:
        def __init__(self, command):
            self.lane = int(command[-1])
            events.append(("launch", self.lane))

        def wait(self):
            events.append(("wait", self.lane))
            return 0

    def barrier(path, deadline, poll, groups):
        assert groups == 4
        assert events == [("launch", 6), ("launch", 7), ("wait", 6), ("wait", 7)]
        assert launcher.read_json(path / "group-3.json")["failed"] is False
        launcher.atomic_json(path / "release.json", {"failed": False})
        return False

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher, "wait_for_groups", barrier)
    assert (
        launcher.run_group(tmp_path, launcher.sha256(tmp_path / "plan.json"), "batch-001", 3) == 0
    )
    plan["tasks"][0]["lane"] = 8
    launcher.atomic_json(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="scheduling changed"):
        launcher.verified_plan(tmp_path)
