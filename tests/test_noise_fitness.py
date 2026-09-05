import io
import zipfile

import torch
from PIL import Image

from src.models.noise import Noise


def test_latest_fitness_uses_current_image_score_only():
    candidate = Noise(id="n_0001", initial_noise=torch.zeros(1, 4, 2, 2))
    candidate.evaluation_scores = [
        {"name": "old", "score": 1.0},
        {"name": "current", "score": 4.5},
    ]

    candidate.calculate_fitness("latest")

    assert candidate.fitness == 4.5


def test_legacy_mean_history_fitness_is_preserved():
    candidate = Noise(id="n_0001", initial_noise=torch.zeros(1, 4, 2, 2))
    candidate.evaluation_scores = [
        {"name": "first", "score": 1.0},
        {"name": "second", "score": 5.0},
    ]

    candidate.calculate_fitness()

    assert candidate.fitness == 3.0


def test_scored_jpeg_is_written_without_reencoding():
    candidate = Noise(
        id="n_0001",
        initial_noise=torch.zeros(1, 4, 2, 2),
        fitness=4.5,
        end_generation=0,
    )
    candidate.set_scored_jpeg(Image.new("RGB", (8, 8), color=(12, 34, 56)))

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        candidate.add_to_zip(archive)
    with zipfile.ZipFile(io.BytesIO(archive_buffer.getvalue())) as archive:
        archived = archive.read(f"/images/{candidate.filename}.JPEG")

    assert archived == candidate.jpeg_artifact
    with Image.open(io.BytesIO(archived)) as image:
        assert list(image.convert("RGB").getdata()) == list(
            candidate.pil_image.getdata()
        )
