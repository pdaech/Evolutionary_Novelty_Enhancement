import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from src.factorys import NoiseFactory
from src.models import Noise
from src.pipelines.genetic_algorithm import GeneticAlgorithmPipeline


class DummyEvaluator:
    @classmethod
    def need(cls):
        return None

    def config_metadata(self):
        return {}


def _pipeline(tmp_path, monkeypatch, experiment_id="test-run"):
    monkeypatch.setenv("BASE_PATH", str(tmp_path))
    return GeneticAlgorithmPipeline(
        generative_model=object(),
        embedding_model=None,
        crossover_operation=object(),
        noise_factory=NoiseFactory(device="cpu"),
        mutator=object(),
        selector=object(),
        evaluator=DummyEvaluator(),
        global_evaluator=None,
        prompt="cat",
        experiment_id=experiment_id,
        num_generations=1,
        population_size=2,
        initial_mutation_rate=0.0,
        initial_crossover_rate=0.5,
        compute_auxiliary_metrics=False,
    )


def test_existing_experiment_directory_is_rejected(tmp_path, monkeypatch):
    _pipeline(tmp_path, monkeypatch)

    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        _pipeline(tmp_path, monkeypatch)


def test_copy_child_records_direct_parent_and_clears_stale_state(tmp_path, monkeypatch):
    pipeline = _pipeline(tmp_path, monkeypatch, experiment_id="lineage")
    parent = Noise(
        id="n_parent",
        initial_noise=torch.zeros(1, 4, 2, 2),
        fitness=5.0,
        evaluation_scores=[{"name": "old", "score": 5.0}],
        start_generation=0,
        end_generation=0,
        parent_1="n_grandparent",
        parent_2="n_other",
    )
    pipeline.population = [parent]
    pipeline.population_size = 1
    pipeline.elite_size = 0
    pipeline.crossover_rate = -1.0
    pipeline.initial_mutation_rate = -1.0
    pipeline.selection_function = SimpleNamespace(
        select=lambda population: population[0]
    )

    pipeline.evolve()

    child = pipeline.population[0]
    assert child.id != parent.id
    assert child.parent_1 == parent.id
    assert child.parent_2 == ""
    assert child.start_generation == 1
    assert child.end_generation == 1
    assert child.evaluation_scores == []
    assert child.fitness is None
    assert child.crossover is False
    assert child.mutate is False


def test_invalid_fitness_response_is_persisted_before_failure(tmp_path, monkeypatch):
    pipeline = _pipeline(tmp_path, monkeypatch, experiment_id="invalid-audit")
    candidate = Noise(
        id="n_0007",
        initial_noise=torch.zeros(1, 4, 2, 2),
        jpeg_artifact=b"exact archived image bytes",
    )
    pipeline.population = [candidate]
    pipeline.generations_done = 3

    pipeline.save_fitness_failures(
        [
            {
                "name": "Gemma4Creativity",
                "score": 0.0,
                "raw_response": '{"score": 0.0}',
                "parse_error": "score_out_of_range",
            }
        ]
    )

    records = [
        json.loads(line)
        for line in pipeline.fitness_failure_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(records) == 1
    assert records[0]["generation"] == 3
    assert records[0]["candidate_id"] == "n_0007"
    assert records[0]["raw_response"] == '{"score": 0.0}'
    assert records[0]["parse_error"] == "score_out_of_range"
    assert (
        records[0]["content_sha256"]
        == hashlib.sha256(candidate.jpeg_artifact).hexdigest()
    )
