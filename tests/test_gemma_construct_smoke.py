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
    slurm = smoke.sbatch_command(plan, tmp_path / "campaign", "hash")
    assert "--gres=gpu:1" in slurm
    assert "--mem=64G" in slurm
    assert "--time=02:00:00" in slurm
    assert "--partition=gpu2" in slurm


def test_smoke_audits_scores_question_images_and_noise(plan, tmp_path):
    task = plan["tasks"][0]
    task["output_directory"] = str(tmp_path / "results")
    task["run_name"] = "fixture"
    directory = Path(task["output_directory"])
    directory.mkdir()
    stem = directory / task["run_name"]
    config = {
        "prompt": "a cat",
        "random_seed": 2025,
        "population_size": 4,
        "num_generations": 1,
        "generation_code_commit": "a" * 40,
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
    assert smoke.audit(plan)["rows"] == 8
    config["evaluator_config"]["prompt"] = "How creative do you find the image?"
    Path(f"{stem}.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Wrong scoring question"):
        smoke.audit(plan)
