import torch

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
