"""Build a portable, compact review ZIP for the six verified fitness conditions.

The ZIP contains 432 sheets of 100 thumbnails, an HTML browser and a row-level
index. Original full-resolution JPEGs and noise tensors stay in the source ZIPs.
No model inference is performed.
"""

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

GENERATIONS = (0, 10, 20, 30)
SEEDS = (2025, 2026, 2027)
PROMPTS = {
    "a cat": "cat",
    "a dog": "dog",
    "a chair": "chair",
    "a coffee mug": "coffee-mug",
    "boredom": "boredom",
    "creativity": "creativity",
}
CONSTRUCTS = (
    "novelty",
    "unusualness",
    "uncommonness",
    "uniqueness",
    "originality",
    "innovation",
)
INDEX_FIELDS = (
    "construct",
    "generation_prompt",
    "seed",
    "generation",
    "rank",
    "fitness",
    "candidate_id",
    "file_name",
    "source_zip",
    "source_zip_sha256_from_receipt",
    "source_zip_member",
    "image_sha256",
    "sheet_path",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect_runs(novelty, remaining):
    found = {}
    sources = []
    for campaign, expected_kind, count in (
        (novelty, "six-gpu-fitness-construct", 18),
        (remaining, "multi-construct-work-queue", 90),
    ):
        plan_path = campaign / "plan.json"
        plan = read_json(plan_path)
        require(plan.get("kind") == expected_kind, f"Wrong campaign kind: {campaign}")
        require(len(plan["tasks"]) == count, f"Wrong task count: {campaign}")
        require(
            (campaign / "completion.json").is_file(),
            f"No campaign completion: {campaign}",
        )
        sources.append(
            {
                "campaign": str(campaign),
                "plan_sha256": sha256_file(plan_path),
                "completion_sha256": sha256_file(campaign / "completion.json"),
                "generator_commit": plan["generator_commit"],
            }
        )
        for planned in plan["tasks"]:
            construct = planned.get("construct", plan.get("construct"))
            prompt = planned["prompt"]
            seed = planned["seed"]
            key = planned["key"]
            require(construct in CONSTRUCTS, f"Unexpected condition: {key}")
            require(prompt in PROMPTS and seed in SEEDS, f"Unexpected prompt/seed: {key}")
            require(
                key == f"{construct}-seed{seed}-{PROMPTS[prompt]}",
                f"Unexpected task identity: {key}",
            )
            identity = (construct, prompt, seed)
            require(identity not in found, f"Duplicate run: {identity}")
            receipt_path = campaign / "completed" / f"{key}.json"
            receipt = read_json(receipt_path)
            accepted = receipt["task"]
            require(
                receipt.get("status") == "complete"
                and receipt.get("rows") == 3100
                and receipt.get("generations") == list(range(31)),
                f"Run has no complete audit: {key}",
            )
            require(
                accepted["key"] == key
                and accepted["prompt"] == prompt
                and accepted["seed"] == seed
                and accepted.get("construct", construct) == construct,
                f"Accepted attempt differs from plan: {key}",
            )
            stem = Path(accepted["output_directory"]) / accepted["run_name"]
            paths = {suffix: Path(f"{stem}.{suffix}") for suffix in ("csv", "json", "zip")}
            require(
                all(
                    str(path) == receipt["files"][suffix]["path"] for suffix, path in paths.items()
                ),
                f"Accepted artifact paths differ: {key}",
            )
            require(
                all(paths[suffix].is_file() for suffix in paths),
                f"Missing accepted artifact: {key}",
            )
            for suffix in ("csv", "json"):
                require(
                    sha256_file(paths[suffix]) == receipt["files"][suffix]["sha256"],
                    f"Accepted {suffix} changed: {key}",
                )
            config = read_json(paths["json"])
            require(
                config.get("prompt") == prompt
                and config.get("random_seed") == seed
                and config.get("population_size") == 100
                and config.get("num_generations") == 30
                and config.get("generation_code_commit") == plan["generator_commit"]
                and config.get("evaluator_config", {}).get("construct") == construct,
                f"Saved scientific settings differ: {key}",
            )
            found[identity] = {
                "key": key,
                "construct": construct,
                "prompt": prompt,
                "seed": seed,
                "paths": paths,
                "source_zip_sha256": receipt["files"]["zip"]["sha256"],
                "receipt_sha256": sha256_file(receipt_path),
            }
    expected = {
        (condition, prompt, seed)
        for condition in CONSTRUCTS
        for prompt in PROMPTS
        for seed in SEEDS
    }
    require(
        set(found) == expected,
        f"Missing or extra run identities: {expected ^ set(found)}",
    )
    return [found[key] for key in sorted(found)], sources


def selected_rows(csv_path, key, population=100):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == population * 31, f"Wrong source row count: {key}")
    require(
        Counter(int(row["generation"]) for row in rows)
        == {generation: population for generation in range(31)},
        f"Missing generation or population: {key}",
    )
    selected = {}
    for generation in GENERATIONS:
        group = [row for row in rows if int(row["generation"]) == generation]
        for row in group:
            name = row["file_name"]
            require(
                name.startswith(f"g{generation}_id") and not any(char in name for char in "/\\:"),
                f"Unsafe or mismatched image name: {key} {name}",
            )
            score = float(row["fitness"])
            require(math.isfinite(score) and 1 <= score <= 5, f"Invalid score: {key}")
            require(not row["fitness_parse_error"].strip(), f"Invalid rating: {key}")
        selected[generation] = sorted(
            group, key=lambda row: (-float(row["fitness"]), row["file_name"])
        )
    require(
        len({row["file_name"] for group in selected.values() for row in group})
        == population * len(GENERATIONS),
        f"Duplicate milestone filename: {key}",
    )
    return selected


def render_sheet(source, rows, run, generation, population=100):
    require(len(rows) == population, "Sheet has wrong population")
    cols = 10
    cell_w, cell_h = 136, 154
    header_h = 76
    sheet = Image.new(
        "RGB",
        (cols * cell_w, header_h + math.ceil(population / cols) * cell_h),
        "#f7f7f7",
    )
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.load_default(size=20)
    label_font = ImageFont.load_default(size=14)
    draw.text(
        (8, 6),
        f"{run['construct']} | {run['prompt']} | seed {run['seed']} | generation {generation:02d}",
        fill="#111111",
        font=title_font,
    )
    draw.text(
        (8, 42),
        f"{population} images · sorted by Gemma fitness (highest first)"
        " · rank and score below each image",
        fill="#333333",
        font=label_font,
    )
    indexed = []
    for rank, row in enumerate(rows, 1):
        member = f"/images/{row['file_name']}.JPEG"
        raw = source.read(member)  # ZipFile checks the selected member's CRC.
        with Image.open(io.BytesIO(raw)) as original:
            require(
                original.format == "JPEG" and original.size == (1024, 1024),
                f"Bad source image: {member}",
            )
            original.load()
            thumb = original.convert("RGB")
        thumb.thumbnail((128, 128))
        x = ((rank - 1) % cols) * cell_w + 4
        y = header_h + ((rank - 1) // cols) * cell_h + 2
        sheet.paste(thumb, (x + (128 - thumb.width) // 2, y))
        draw.text(
            (x, y + 130),
            f"{rank:03d}  {float(row['fitness']):.2f}",
            fill="#111111",
            font=label_font,
        )
        indexed.append((rank, row, member, sha256_bytes(raw)))
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=82, subsampling=0)
    return buffer.getvalue(), indexed


def html_index(sheet_paths):
    lines = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<title>Gemma fitness milestone contact sheets</title>",
        (
            "<style>body{font:16px system-ui,sans-serif;max-width:1120px;"
            "margin:32px auto;padding:0 12px}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:7px}"
            "tr:nth-child(even){background:#f5f5f5}a{color:#0755a0}</style>"
        ),
        "<h1>Gemma fitness milestone contact sheets</h1>",
        (
            "<p>Six fitness conditions, six image prompts, three seeds. Each sheet shows all 100 images"
            " from one generation, sorted by score. Click a generation to open its sheet."
            " The number below an image is its rank and score; index.csv maps it"
            " to the original image.</p>"
        ),
        (
            "<table><thead><tr><th>Fitness</th><th>Image prompt</th><th>Seed</th>"
            "<th>00</th><th>10</th><th>20</th><th>30</th></tr></thead><tbody>"
        ),
    ]
    for condition in CONSTRUCTS:
        for prompt in PROMPTS:
            for seed in SEEDS:
                parts = [
                    f"<tr><td>{html.escape(condition)}</td>",
                    f"<td>{html.escape(prompt)}</td><td>{seed}</td>",
                ]
                for generation in GENERATIONS:
                    path = sheet_paths[(condition, prompt, seed, generation)]
                    parts.append(f"<td><a href='{html.escape(path, quote=True)}'>Open</a></td>")
                lines.append("".join(parts) + "</tr>")
    lines.append("</tbody></table></html>")
    return "\n".join(lines)


def build_package(novelty, remaining, output, population=100):
    require(not output.exists(), f"Refusing to overwrite: {output}")
    partial = output.with_name(output.name + ".partial")
    require(not partial.exists(), f"Previous partial export needs inspection: {partial}")
    runs, sources = collect_runs(novelty, remaining)
    output.parent.mkdir(parents=True, exist_ok=True)
    index_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(index_buffer, fieldnames=INDEX_FIELDS, lineterminator="\n")
    writer.writeheader()
    sheet_paths = {}
    run_receipts = {}
    try:
        with zipfile.ZipFile(
            partial, "x", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as target:
            for number, run in enumerate(runs, 1):
                selected = selected_rows(run["paths"]["csv"], run["key"], population)
                run_receipts[run["key"]] = run["receipt_sha256"]
                with zipfile.ZipFile(run["paths"]["zip"]) as source:
                    names = source.namelist()
                    require(
                        len(names) == len(set(names)),
                        f"Duplicate ZIP members: {run['key']}",
                    )
                    available = set(names)
                    for generation, rows in selected.items():
                        require(
                            all(f"/images/{row['file_name']}.JPEG" in available for row in rows),
                            f"Missing milestone image: {run['key']} generation {generation}",
                        )
                        sheet_path = (
                            f"sheets/{run['construct']}/{PROMPTS[run['prompt']]}"
                            f"/seed-{run['seed']}/generation-{generation:02d}.jpg"
                        )
                        require(
                            sheet_path not in sheet_paths.values(),
                            "Repeated sheet path",
                        )
                        sheet, entries = render_sheet(source, rows, run, generation, population)
                        target.writestr(sheet_path, sheet)
                        sheet_paths[(run["construct"], run["prompt"], run["seed"], generation)] = (
                            sheet_path
                        )
                        for rank, row, member, image_hash in entries:
                            writer.writerow(
                                {
                                    "construct": run["construct"],
                                    "generation_prompt": run["prompt"],
                                    "seed": run["seed"],
                                    "generation": generation,
                                    "rank": rank,
                                    "fitness": row["fitness"],
                                    "candidate_id": row["candidate_id"],
                                    "file_name": row["file_name"],
                                    "source_zip": str(run["paths"]["zip"]),
                                    "source_zip_sha256_from_receipt": run["source_zip_sha256"],
                                    "source_zip_member": member,
                                    "image_sha256": image_hash,
                                    "sheet_path": sheet_path,
                                }
                            )
                print(f"CONTACT SHEETS {number}/108: {run['key']}", flush=True)
            require(len(sheet_paths) == 108 * 4, "Wrong contact-sheet count")
            metadata = {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "script_sha256": sha256_file(__file__),
                "source_campaigns": sources,
                "accepted_receipt_sha256": run_receipts,
                "run_count": 108,
                "generations": list(GENERATIONS),
                "images_per_generation": population,
                "sheet_count": len(sheet_paths),
                "indexed_images": 108 * 4 * population,
                "notes": "Contact sheets only; original JPEGs and noise tensors"
                " remain in source ZIPs."
                " Selected image members were read with CRC checks."
                " Source ZIP SHA-256 values are copied from verified completion receipts,"
                " not recomputed during this export.",
            }
            target.writestr(
                "index.csv", index_buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED
            )
            target.writestr(
                "index.html",
                html_index(sheet_paths),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            target.writestr(
                "metadata.json",
                json.dumps(metadata, indent=2) + "\n",
                compress_type=zipfile.ZIP_DEFLATED,
            )
            target.writestr(
                "README.txt",
                "Unzip this package and open index.html. Each link opens one sheet of 100 images.\n"
                "The number below each thumbnail is its rank and Gemma fitness score.\n"
                "Ranks restart at 1 in each generation and are sorted from highest score.\n"
                "Use index.csv to find each original filename, source ZIP member, score"
                " and SHA-256.\n"
                "Contact sheets are reduced previews; use the original source ZIP"
                " for full resolution.\n",
            )
        with zipfile.ZipFile(partial) as check:
            require(len(check.namelist()) == 436, "Wrong package member count")
            require(check.testzip() is None, "Review package ZIP CRC failure")
        os.link(partial, output)  # Atomic publish without replacing an existing ZIP.
        partial.unlink()
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f"EXPORT OK: {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    print("432 contact sheets; 43,200 indexed images; six fitness conditions, 108 runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--novelty-campaign", type=Path, required=True)
    parser.add_argument("--remaining-campaign", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()
    build_package(args.novelty_campaign, args.remaining_campaign, args.output_zip)


if __name__ == "__main__":
    os.umask(0o002)
    main()
