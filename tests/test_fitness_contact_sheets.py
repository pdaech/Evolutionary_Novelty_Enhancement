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

SCRIPT = Path(__file__).resolve().parents[1] / "cluster/export_fitness_contact_sheets.py"
SPEC = importlib.util.spec_from_file_location("fitness_contact_sheets", SCRIPT)
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def make_source(tmp_path):
    if sys.platform == "win32":
        tmp_path = Path("\\\\?\\" + str(tmp_path))
    image = io.BytesIO()
    Image.new("RGB", (1024, 1024), "#2468a8").save(image, "JPEG")
    novelty = tmp_path / "novelty"
    remaining = tmp_path / "remaining"
    for campaign, constructs, kind in (
        (novelty, ("novelty",), "six-gpu-fitness-construct"),
        (remaining, exporter.CONSTRUCTS[1:], "multi-construct-work-queue"),
    ):
        (campaign / "completed").mkdir(parents=True)
        tasks = []
        index = 0
        for construct in constructs:
            for prompt, slug in exporter.PROMPTS.items():
                for seed in exporter.SEEDS:
                    key = f"{construct}-seed{seed}-{slug}"
                    directory = tmp_path / campaign.name / "artifacts" / str(index)
                    directory.mkdir(parents=True)
                    run_name = "accepted"
                    stem = directory / run_name
                    planned = {"key": key, "prompt": prompt, "seed": seed}
                    if construct != "novelty":
                        planned["construct"] = construct
                    tasks.append(planned)
                    accepted = dict(planned, output_directory=str(directory), run_name=run_name)
                    rows = []
                    with zipfile.ZipFile(f"{stem}.zip", "w") as archive:
                        for generation in range(31):
                            for candidate in range(2):
                                filename = f"g{generation}_idn_{candidate:04d}_f4.0"
                                rows.append(
                                    {
                                        "generation": generation,
                                        "candidate_id": f"n_{candidate:04d}",
                                        "file_name": filename,
                                        "fitness": "4.5" if candidate else "3.5",
                                        "fitness_parse_error": "",
                                    }
                                )
                                if generation in exporter.GENERATIONS:
                                    archive.writestr(f"/images/{filename}.JPEG", image.getvalue())
                    with Path(f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    Path(f"{stem}.json").write_text(
                        json.dumps(
                            {
                                "prompt": prompt,
                                "random_seed": seed,
                                "population_size": 100,
                                "num_generations": 30,
                                "generation_code_commit": "a" * 40,
                                "evaluator_config": {"construct": construct},
                            }
                        ),
                        encoding="utf-8",
                    )
                    files = {
                        suffix: {
                            "path": str(Path(f"{stem}.{suffix}")),
                            "sha256": exporter.sha256_file(f"{stem}.{suffix}"),
                        }
                        for suffix in ("csv", "json", "zip")
                    }
                    (campaign / "completed" / f"{key}.json").write_text(
                        json.dumps(
                            {
                                "status": "complete",
                                "rows": 3100,
                                "generations": list(range(31)),
                                "task": accepted,
                                "files": files,
                            }
                        ),
                        encoding="utf-8",
                    )
                    index += 1
        plan = {"kind": kind, "tasks": tasks, "generator_commit": "a" * 40}
        if kind == "six-gpu-fitness-construct":
            plan["construct"] = "novelty"
        (campaign / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (campaign / "completion.json").write_text('{"status":"complete"}', encoding="utf-8")
    return novelty, remaining, tmp_path


def test_complete_contact_package_covers_every_condition_run_and_image(tmp_path):
    novelty, remaining, root = make_source(tmp_path)
    destination = root / "package.zip"
    exporter.build_package(novelty, remaining, destination, population=2)
    with zipfile.ZipFile(destination) as package:
        assert len(package.namelist()) == 436
        assert package.testzip() is None
        metadata = json.loads(package.read("metadata.json"))
        assert metadata["run_count"] == 108
        assert metadata["sheet_count"] == 432
        assert metadata["indexed_images"] == 864
        with io.TextIOWrapper(package.open("index.csv"), encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 864
        assert {row["construct"] for row in rows} == set(exporter.CONSTRUCTS)
        assert {int(row["generation"]) for row in rows} == set(exporter.GENERATIONS)
        assert {int(row["seed"]) for row in rows} == set(exporter.SEEDS)
        assert len({row["sheet_path"] for row in rows}) == 432
        first = next(row for row in rows if row["construct"] == "novelty")
        assert first["rank"] == "1" and first["fitness"] == "4.5"
        with Image.open(io.BytesIO(package.read(first["sheet_path"]))) as sheet:
            assert sheet.format == "JPEG" and sheet.width == 1360
            sheet.load()
        source = Path(first["source_zip"])
        with zipfile.ZipFile(source) as archive:
            original = archive.read(first["source_zip_member"])
        assert hashlib.sha256(original).hexdigest() == first["image_sha256"]
        assert first["sheet_path"] in package.read("index.html").decode("utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        exporter.build_package(novelty, remaining, destination, population=2)


def test_export_rejects_changed_accepted_csv_and_leaves_no_package(tmp_path):
    novelty, remaining, root = make_source(tmp_path)
    first = next((novelty / "completed").glob("*.json"))
    csv_path = Path(json.loads(first.read_text(encoding="utf-8"))["files"]["csv"]["path"])
    with csv_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    destination = root / "bad.zip"
    with pytest.raises(ValueError, match="Accepted csv changed"):
        exporter.build_package(novelty, remaining, destination, population=2)
    assert not destination.exists() and not Path(str(destination) + ".partial").exists()


def test_export_rejects_duplicate_source_zip_members(tmp_path):
    novelty, remaining, root = make_source(tmp_path)
    first = next((novelty / "completed").glob("*.json"))
    path = Path(json.loads(first.read_text(encoding="utf-8"))["files"]["zip"]["path"])
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("/images/g0_idn_0000_f4.0.JPEG", b"duplicate")
    with pytest.raises(ValueError, match="Duplicate ZIP members"):
        exporter.build_package(novelty, remaining, root / "bad.zip", population=2)
