import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

CLUSTER = Path(__file__).resolve().parents[1] / "cluster"
sys.path.insert(0, str(CLUSTER))
SPEC = importlib.util.spec_from_file_location(
    "construct_smoke", CLUSTER / "smoke_fitness_construct.py"
)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


@pytest.fixture
def plan(tmp_path):
    if sys.platform == "win32":
        tmp_path = Path("\\\\?\\" + str(tmp_path))
    return smoke.make_plan(
        tmp_path / "code", tmp_path / "runtime", "novelty-smoke-v1", "novelty", "a" * 40
    )


def test_smoke_requests_one_gpu_and_uses_full_model_settings(plan, tmp_path):
    task = plan["tasks"][0]
    assert task["prompt"] == "a cat"
    assert task["seed"] == 2025
    assert plan["options"]["population_size"] == 4
    assert plan["options"]["num_generations"] == 1
    assert plan["options"]["gemma_image_token_budget"] == 140
    assert plan["options"]["sdxl_num_inference_steps"] == 50
    assert plan["options"]["gemma_fitness_construct"] == "novelty"
    command = smoke.full.inference_command(plan, task)
    options = dict(zip(command[3::2], command[4::2], strict=True))
    assert options["--evaluator"] == "gemma-construct"
    assert options["--gemma_fitness_construct"] == "novelty"
    assert options["--population_size"] == "4"
    assert options["--num_generations"] == "1"
    actual_output = (
        Path(plan["environment"]["BASE_PATH"])
        / options["--directory"]
        / f"{options['--id']}_{options['--experiment_id']}"
    )
    assert Path(task["output_directory"]) == actual_output
    slurm = smoke.sbatch_command(plan, tmp_path / "campaign", "hash")
    assert "--gres=gpu:1" in slurm
    assert "--mem=64G" in slurm
    assert "--time=02:00:00" in slurm
    assert "--partition=gpu2" in slurm


def write_artifacts(plan):
    task = plan["tasks"][0]
    directory = Path(task["output_directory"])
    directory.mkdir(parents=True)
    stem = directory / task["run_name"]
    config = {
        "prompt": "a cat",
        "random_seed": 2025,
        "population_size": 4,
        "num_generations": 1,
        "generation_code_commit": plan["generator_commit"],
        "fitness_aggregation": "latest",
        "evaluator": "GemmaConstructEvaluator",
        "evaluator_config": {
            "construct": "novelty",
            "prompt": plan["scoring_prompt"],
            "prompt_sha256": hashlib.sha256(plan["scoring_prompt"].encode("utf-8")).hexdigest(),
            "model_id": plan["options"]["gemma_model"],
            "requested_revision": plan["options"]["gemma_revision"],
        },
    }
    Path(f"{stem}.json").write_text(json.dumps(config), encoding="utf-8")
    image = io.BytesIO()
    Image.new("RGB", (1024, 1024), "white").save(image, format="JPEG")
    rows = []
    with zipfile.ZipFile(f"{stem}.zip", "w") as archive:
        for generation in (0, 1):
            for candidate in range(4):
                name = f"g{generation}_idn_{candidate:04d}_f4.0"
                rows.append(
                    {
                        "generation": generation,
                        "candidate_id": f"n_{candidate:04d}",
                        "file_name": name,
                        "fitness": "4.0",
                        "score_value": "4.0",
                        "score_name": "Gemma4Novelty",
                        "fitness_parse_error": "",
                    }
                )
                archive.writestr(f"/images/{name}.JPEG", image.getvalue())
                archive.writestr(f"/noise/{name}.pt", b"noise")
    with Path(f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return stem, config


def test_smoke_audits_scores_question_images_and_noise(plan):
    stem, config = write_artifacts(plan)
    assert smoke.audit(plan)["rows"] == 8
    config["evaluator_config"]["prompt"] = "How creative do you find the image?"
    Path(f"{stem}.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Wrong scoring question"):
        smoke.audit(plan)


@pytest.fixture
def legacy_campaign(plan, monkeypatch):
    plan["generator_commit"] = "c45162a93971be8ec3d565356915b29d067c8552"
    stem, _ = write_artifacts(plan)
    campaign = Path(plan["environment"]["BASE_PATH"]) / "smoke" / plan["campaign"]
    campaign.mkdir(parents=True)
    plan["tasks"][0]["output_directory"] = str(campaign / "results" / plan["tasks"][0]["run_name"])
    smoke.full.save_json(campaign / "plan.json", plan)
    smoke.full.save_json(
        campaign / "submission_request.json",
        {"command": smoke.sbatch_command(plan, campaign, smoke.sha256(campaign / "plan.json"))},
    )
    monkeypatch.setattr(smoke.full, "clean_commit", lambda _: plan["generator_commit"])

    def forbidden_inference(*args, **kwargs):
        pytest.fail("Audit recovery must not rerun inference")

    monkeypatch.setattr(smoke.full, "execute_task", forbidden_inference)
    return campaign, stem


def test_recover_existing_outputs_preserves_original_plan_and_hashes(legacy_campaign):
    campaign, stem = legacy_campaign
    original = (campaign / "plan.json").read_bytes()
    smoke.recover_audit(campaign)
    report = smoke.read_json(campaign / "audit-recovery.json")
    assert report["rows"] == 8
    assert report["actual_output_directory"] == str(stem.parent)
    assert report["original_plan_sha256"] == hashlib.sha256(original).hexdigest()
    assert (campaign / "plan.json").read_bytes() == original
    assert not (campaign / "completion.json").exists()
    smoke.recover_audit(campaign)  # Repeat rechecks the files, without overwriting.
    with Path(f"{stem}.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="artifacts changed"):
        smoke.recover_audit(campaign)


def test_recovery_rejects_changed_plan(legacy_campaign):
    campaign, _ = legacy_campaign
    with (campaign / "plan.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="plan changed"):
        smoke.recover_audit(campaign)
    assert not (campaign / "audit-recovery.json").exists()


def test_recovery_does_not_accept_incomplete_artifacts(legacy_campaign):
    campaign, stem = legacy_campaign
    with zipfile.ZipFile(f"{stem}.zip", "w"):
        pass
    with pytest.raises(ValueError, match="coverage differs"):
        smoke.recover_audit(campaign)
    assert not (campaign / "audit-recovery.json").exists()
