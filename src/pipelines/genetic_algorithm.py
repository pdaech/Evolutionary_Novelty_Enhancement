import gc
import hashlib
import json
import logging
import os
import random
import zipfile
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch

from src.crossover.base_crossover import CrossoverFunction
from src.evaluators.base_evaluator import Evaluator
from src.evaluators.diversity_evaluator import DiversityEvaluator
from src.evaluators.prompt_fidelity import PromptFidelityEvaluator
from src.factorys import NoiseFactory
from src.huggingface_models.base_strategy import (
    CaptionModelStrategy,
    EmbeddingModelStrategy,
    GenerativModelStrategy,
)
from src.models import Noise
from src.mutators.base_mutator import MutationFunction
from src.selector_functions.base_selectorf_unction import SelectorFunction

from .pipeline import Pipeline

logger = logging.getLogger(__name__)


def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class GeneticAlgorithmPipeline(Pipeline):
    def __init__(
        self,
        generative_model: GenerativModelStrategy,
        embedding_model: EmbeddingModelStrategy | None,
        crossover_operation: CrossoverFunction,
        noise_factory: NoiseFactory,
        mutator: MutationFunction,
        selector: SelectorFunction,
        evaluator: Evaluator,
        global_evaluator: Evaluator | None,
        prompt: str,
        experiment_id: str,
        num_generations: int,
        population_size: int,
        initial_mutation_rate: float,
        initial_crossover_rate: float,
        elite_size: int = 0,
        batch_size: int = 1,
        caption_model: CaptionModelStrategy | None = None,
        initial_generation: list[Noise] | None = None,
        result_path: str = "0_results/simulations",
        fitness_aggregation: str = "mean_history",
        compute_auxiliary_metrics: bool = True,
    ):
        self.population = None
        self.generative_model = generative_model
        self.embedding_model = embedding_model
        self.crossover_operation = crossover_operation
        self.noise_factory = noise_factory
        self.crossover_rate = initial_crossover_rate
        self.elite_size = elite_size
        self.caption_model = caption_model
        self.evaluator = evaluator
        self.global_evaluator = global_evaluator
        self.initial_mutation_rate = initial_mutation_rate
        self.prompt = prompt
        self.num_generations = num_generations
        self.population_size = population_size
        self.selection_function = selector
        self.mutator = mutator
        self.initial_generation = initial_generation
        self.batch_size = batch_size
        if fitness_aggregation not in {"latest", "mean_history"}:
            raise ValueError(
                "fitness_aggregation must be either 'latest' or 'mean_history'"
            )
        self.fitness_aggregation = fitness_aggregation
        self.compute_auxiliary_metrics = compute_auxiliary_metrics
        self.generations_done = 0
        self.prompt_fidelity_evaluator = (
            PromptFidelityEvaluator(prompt) if compute_auxiliary_metrics else None
        )
        self.diversity_evaluator = (
            DiversityEvaluator()
            if compute_auxiliary_metrics and embedding_model is not None
            else None
        )
        base_path = Path(os.environ["BASE_PATH"])
        self.name = f"{experiment_id}"

        self.result_path = base_path / result_path / self.name
        self.zip_path = self.result_path / f"{self.name}.zip"
        self.state_path = self.result_path / f"{self.name}.csv"
        self.config_path = self.result_path / f"{self.name}.json"
        self.fitness_failure_path = (
            self.result_path / f"{self.name}.fitness_failures.jsonl"
        )
        self.generation_timing_path = (
            self.result_path / f"{self.name}.generation_timings.jsonl"
        )

        try:
            self.result_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to reuse experiment output directory {self.result_path}. "
                "Choose a new experiment id; automatic resume is not implemented."
            ) from exc

    def create_batches(self, embeds: list[Any]) -> Generator[list[Any], None, None]:
        num_samples = len(embeds)
        for i in range(0, num_samples, self.batch_size):
            yield embeds[i : i + self.batch_size]

    def save_config(self):
        config = {
            "num_generations": self.num_generations,
            "population_size": self.population_size,
            "batch_size": self.batch_size,
            "prompt": self.prompt,
            "generative_model": type(self.generative_model).__name__,
            "generative_model_config": (
                self.generative_model.config_metadata()
                if hasattr(self.generative_model, "config_metadata")
                else {}
            ),
            "selector": type(self.selection_function).__name__,
            "selector_config": {
                "tournament_size": getattr(
                    self.selection_function, "tournament_size", None
                ),
            },
            "mutator": type(self.mutator).__name__,
            "mutator_config": {
                "mutation_rate": getattr(self.mutator, "mutation_rate", None),
                "mutation_strength": getattr(
                    self.mutator, "mutation_strength", None
                ),
            },
            "crossover_function": type(self.crossover_operation).__name__,
            "crossover_config": {
                "swap_rate": getattr(self.crossover_operation, "swap_rate", None),
            },
            "noise_factory_config": {
                "distribution": "standard_normal",
                "latent_shape_per_candidate": [1, 4, 128, 128],
                "image_size": [1024, 1024],
                "init_noise_sigma": 1.0,
                "apply_pink_noise_filter": getattr(
                    self.noise_factory, "apply_pink_noise_filter", None
                ),
                "dtype": str(getattr(self.noise_factory, "dtype", None)),
                "device": str(getattr(self.noise_factory, "device", None)),
            },
            "initial_mutation_rate": self.initial_mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elitism_count": self.elite_size,
            "no_crossover_policy": "copy_fitter_selected_parent",
            "evaluator": type(self.evaluator).__name__,
            "evaluator_config": self.evaluator.config_metadata(),
            "global_evaluator": (
                type(self.global_evaluator).__name__
                if self.global_evaluator is not None
                else None
            ),
            "fitness_aggregation": self.fitness_aggregation,
            "compute_auxiliary_metrics": self.compute_auxiliary_metrics,
            "generation_code_commit": os.environ.get("GENERATION_CODE_COMMIT"),
            "random_seed": torch.initial_seed(),
            "timestamp": datetime.now(UTC).isoformat(),
            "result_path": str(self.result_path),
            "stat_path": str(self.state_path),
            "generation_timing_path": str(self.generation_timing_path),
            "generation_timing_schema_version": 1,
        }

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=4)

    def save_states(self):
        columns = [
            "generation",
            "candidate_id",
            "start_gen",
            "end_gen",
            "parent_1",
            "parent_2",
            "fitness",
            "global_score",
            "prompt_fidelty",
            "diversity_score",
            "score_name",
            "score_value",
            "caption",
            "file_name",
            "fitness_raw_response",
            "fitness_parse_error",
        ]
        data_rows = []
        for candidate in self.population:
            first_score = candidate.evaluation_scores[0]
            current_score = candidate.evaluation_scores[-1]
            if isinstance(candidate.global_score, dict):
                global_score = candidate.global_score.get("score")
            else:
                global_score = candidate.global_score
            row = [
                self.generations_done,
                candidate.id,
                candidate.start_generation,
                candidate.end_generation,
                candidate.parent_1,
                candidate.parent_2,
                candidate.fitness,
                global_score,
                candidate.prompt_fidelity,
                candidate.diversity_score,
                first_score["name"],
                first_score["score"],
                getattr(candidate, "caption", ""),
                candidate.filename,
                current_score.get("raw_response"),
                current_score.get("parse_error"),
            ]
            data_rows.append(row)

        new_data_df = pd.DataFrame(data_rows, columns=columns)
        if not self.state_path.exists():
            header_needed = True
        else:
            header_needed = False
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        new_data_df.to_csv(self.state_path, mode="a", header=header_needed, index=False)

    def save_fitness_failures(self, scores: list[dict[str, Any]]) -> None:
        """Persist unusable VLM responses before failing the generation."""
        timestamp = datetime.now(UTC).isoformat()
        with self.fitness_failure_path.open("a", encoding="utf-8") as handle:
            for candidate, score in zip(self.population, scores, strict=True):
                if score.get("parse_error") is None and score.get("score") is not None:
                    continue
                image_hash = (
                    hashlib.sha256(candidate.jpeg_artifact).hexdigest()
                    if candidate.jpeg_artifact is not None
                    else None
                )
                record = {
                    "timestamp": timestamp,
                    "generation": self.generations_done,
                    "candidate_id": candidate.id,
                    "score_name": score.get("name"),
                    "score": score.get("score"),
                    "raw_response": score.get("raw_response"),
                    "parse_error": score.get("parse_error") or "missing_score",
                    "content_sha256": image_hash,
                }
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )

    def save_generation_timing(self, seconds: dict[str, float]) -> None:
        """Append auditable stage timings for one successfully evaluated generation."""
        record = {
            "schema_version": 1,
            "timestamp": datetime.now(UTC).isoformat(),
            "generation": self.generations_done,
            "population_size": len(self.population),
            "sdxl_batch_size": self.batch_size,
            "evaluator_batch_size": getattr(self.evaluator, "batch_size", None),
            "image_token_budget": getattr(self.evaluator, "image_token_budget", None),
            "seconds": {
                name: round(float(duration), 6) for name, duration in seconds.items()
            },
        }
        with self.generation_timing_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def one_generation(self):

        one_generation_started = perf_counter()
        image_generation_started = perf_counter()

        pils = []

        batches = self.create_batches(
            [candidate.initial_noise for candidate in self.population]
        )
        for batch_index, batch in enumerate(batches):
            pil_images = self.generative_model.generate_batch(batch, self.prompt)
            pils.extend(pil_images)
            print(f"Batch {batch_index} generiert")
        print(f"Menge der Biler {len(pils)}")
        if len(pils) != len(self.population):
            raise RuntimeError(
                f"Generator returned {len(pils)} images for "
                f"{len(self.population)} candidates"
            )
        image_generation_seconds = perf_counter() - image_generation_started
        pre_evaluation_started = perf_counter()

        evaluator_need = self.evaluator.need()
        if evaluator_need is None:
            for candidate, image in zip(self.population, pils, strict=True):
                candidate.set_scored_jpeg(image)
            pils = [candidate.pil_image for candidate in self.population]
        else:
            for candidate, image in zip(self.population, pils, strict=True):
                candidate.set_pil_image(image)

        embeddings = []
        if self.embedding_model is not None:
            for batch in self.create_batches(pils):
                batch_embeddings = self.embedding_model.batch_image_features_extraction(
                    batch
                )
                embeddings.extend(batch_embeddings)
            print(f"Menge der Embeddings {len(embeddings)}")

        if self.prompt_fidelity_evaluator is not None:
            fidelity_scores = []
            for batch in self.create_batches(pils):
                fidelity_scores.extend(
                    self.prompt_fidelity_evaluator.evaluate_batch(batch)
                )
        else:
            fidelity_scores = [None] * len(pils)

        # Release temporary SDXL allocations before the large VLM starts decoding.
        cleanup_memory()

        if evaluator_need is None:
            evaluation_inputs = pils
        else:
            if not embeddings:
                raise RuntimeError(
                    f"{type(self.evaluator).__name__} requires image embeddings, "
                    "but no embedding model was configured"
                )
            evaluation_inputs = embeddings

        pre_evaluation_seconds = perf_counter() - pre_evaluation_started
        fitness_evaluation_started = perf_counter()
        self.evaluator.update(evaluation_inputs)
        scores = self.evaluator.evaluate_batch(evaluation_inputs)
        fitness_evaluation_seconds = perf_counter() - fitness_evaluation_started
        post_evaluation_started = perf_counter()

        if len(scores) != len(self.population):
            raise RuntimeError(
                f"Evaluator returned {len(scores)} scores for "
                f"{len(self.population)} candidates"
            )
        invalid_scores = [
            score
            for score in scores
            if score.get("parse_error") is not None or score.get("score") is None
        ]
        if invalid_scores:
            self.save_fitness_failures(scores)
            reasons = sorted(
                {
                    str(score.get("parse_error") or "missing_score")
                    for score in invalid_scores
                }
            )
            raise RuntimeError(
                "Gemma creativity evaluation returned unusable scores; "
                f"details were written to {self.fitness_failure_path}. "
                f"Reasons: {', '.join(reasons)}"
            )

        if self.global_evaluator is not None:
            if not embeddings:
                raise RuntimeError("The global evaluator requires image embeddings")
            if self.generations_done == 0:
                self.global_evaluator.update(embeddings)
            global_scores = self.global_evaluator.evaluate_batch(embeddings)
        else:
            global_scores = [None] * len(pils)
        print(f"Menge der Scores {len(scores)}")

        diversity_score = (
            self.diversity_evaluator.evalute_batch(embeddings)
            if self.diversity_evaluator is not None
            else None
        )
        captions = []
        if self.caption_model is not None:
            for batch in self.create_batches(pils):
                batch_captions = self.caption_model.caption_batch(batch)
                captions.extend(batch_captions)

        for i, candidate in enumerate(self.population):
            candidate.blip2_embedding = embeddings[i] if embeddings else None
            if self.fitness_aggregation == "latest":
                candidate.evaluation_scores = [scores[i]]
            else:
                candidate.evaluation_scores.append(scores[i])
            candidate.global_score = global_scores[i]
            candidate.prompt_fidelity = fidelity_scores[i]
            candidate.diversity_score = diversity_score
            candidate.calculate_fitness(self.fitness_aggregation)
            if self.caption_model is not None:
                candidate.caption = captions[i]

        post_evaluation_seconds = perf_counter() - post_evaluation_started
        self.save_generation_timing(
            {
                "image_generation": image_generation_seconds,
                "pre_evaluation": pre_evaluation_seconds,
                "fitness_evaluation": fitness_evaluation_seconds,
                "post_evaluation": post_evaluation_seconds,
                "one_generation_total": perf_counter() - one_generation_started,
            }
        )

    def evolve(self):
        self.generations_done += 1
        new_gen: list[Noise] = []
        self.population = sorted(
            self.population, key=lambda noise: noise.fitness, reverse=True
        )
        new_gen.extend(self.population[: self.elite_size])

        action_log = []
        for candidate in new_gen:
            candidate.end_generation = self.generations_done
        while len(new_gen) < self.population_size:
            parent1 = self.selection_function.select(self.population)
            parent2 = self.selection_function.select(self.population)

            if random.random() > self.crossover_rate:
                copy_parent = max((parent1, parent2), key=lambda parent: parent.fitness)
                child = self.noise_factory.create_noise_from_noise(
                    copy_parent.initial_noise.clone()
                )
                child.start_generation = self.generations_done
                child.end_generation = self.generations_done
                child.parent_1 = copy_parent.id
                child.parent_2 = ""
                child.crossover = False
                child.mutate = False
                log_entry = "Copy"
                if random.random() < self.initial_mutation_rate:
                    child.initial_noise = self.mutator.mutate(child.initial_noise)
                    child.mutate = True
                    log_entry += "+Mut"
                new_gen.append(child)
                action_log.append(log_entry)
            else:
                child_noise = self.crossover_operation.crossover(
                    parent1.initial_noise, parent2.initial_noise
                )
                child = self.noise_factory.create_noise_from_noise(child_noise)
                child.end_generation = self.generations_done
                child.start_generation = self.generations_done
                child.parent_1 = parent1.id
                child.parent_2 = parent2.id
                child.crossover = True
                child.mutate = False
                log_entry = "Cross"
                if random.random() < self.initial_mutation_rate:
                    child.initial_noise = self.mutator.mutate(child.initial_noise)
                    child.mutate = True
                    log_entry += "+Mut"
                new_gen.append(child)
                action_log.append(log_entry)

        self.population = new_gen
        print(f"Gen {self.generations_done}: " + "; ".join(action_log))

    def initial_population(self):
        self.population = self.noise_factory.create_batch(self.population_size)
        self.one_generation()
        for candidate in self.population:
            candidate.end_generation = 0
            candidate.start_generation = 0

    def save_generation(self):
        self.result_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            self.zip_path, "a", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for candidate in self.population:
                candidate.add_to_zip(zf, folder_prefix="")
                # candidate.save_pil_image(self.result_path / "images")
                # candidate.save_blip2(self.result_path / "blip2")
                # candidate.save_noise(self.result_path / "initial_noise")
                # candidate.save_noise_to_rgb(self.result_path / "initial_noise_rgb")

    def run(self):
        if self.initial_generation:
            print("Use pregenerated start generation")
            self.population = self.initial_generation
            self.one_generation()

        else:
            print("Generate fresh start generation")
            self.initial_population()
        self.save_generation()
        self.save_states()
        while self.generations_done < self.num_generations:
            print(f"Next Generattion: {self.generations_done + 1}")
            self.evolve()
            cleanup_memory()
            self.one_generation()
            self.save_generation()
            self.save_states()

        return self.population
