import copy
import logging
import zipfile
import os
import gc
import random
from pathlib import Path
from typing import Generator, Any
import pandas as pd
import torch
import json
from datetime import datetime
from src.models import Noise
from .pipeline import Pipeline
from src.crossover.base_crossover import CrossoverFunction
from src.evaluators.base_evaluator import Evaluator
from src.factorys import NoiseFactory
from src.huggingface_models.base_strategy import (
    GenerativModelStrategy,
    EmbeddingModelStrategy,
    CaptionModelStrategy,
)
from src.mutators.base_mutator import MutationFunction
from src.selector_functions.base_selectorf_unction import SelectorFunction
from src.evaluators.prompt_fidelity import PromptFidelityEvaluator
from src.evaluators.diversity_evaluator import DiversityEvaluator

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
        embedding_model: EmbeddingModelStrategy,
        crossover_operation: CrossoverFunction,
        noise_factory: NoiseFactory,
        mutator: MutationFunction,
        selector: SelectorFunction,
        evaluator: Evaluator,
        global_evaluator: Evaluator,
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
        self.generations_done = 0
        self.prompt_fidelity_evaluator = PromptFidelityEvaluator(prompt)
        self.diversity_evaluator = DiversityEvaluator()
        base_path = Path(os.environ["BASE_PATH"])
        self.name = f"{experiment_id}"

        self.result_path = base_path / result_path / self.name
        self.zip_path = self.result_path / f"{self.name}.zip"
        self.state_path = self.result_path / f"{self.name}.csv"
        self.config_path = self.result_path / f"{self.name}.json"

        if not self.initial_generation and self.zip_path.exists():
            logger.warning(f"Lösche alte Ergebnis-Datei: {self.zip_path}")
            os.remove(self.zip_path)

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
            "selector": type(self.selection_function).__name__,
            "mutator": type(self.mutator).__name__,
            "crossover_function": type(self.crossover_operation).__name__,
            "initial_mutation_rate": self.initial_mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elitism_count": self.elite_size,
            "evaluator": type(self.evaluator).__name__,
            "random_seed": torch.initial_seed(),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "result_path": str(self.result_path),
            "stat_path": str(self.state_path),
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
        ]
        data_rows = []
        for candidate in self.population:
            row = [
                self.generations_done,
                candidate.id,
                candidate.start_generation,
                candidate.end_generation,
                candidate.parent_1,
                candidate.parent_2,
                candidate.fitness,
                candidate.global_score["score"],
                candidate.prompt_fidelity,
                candidate.diversity_score,
                candidate.evaluation_scores[0]["name"],
                candidate.evaluation_scores[0]["score"],
                getattr(candidate, "caption", ""),
                candidate.filename,
            ]
            data_rows.append(row)

        new_data_df = pd.DataFrame(data_rows, columns=columns)
        if not self.state_path.exists():
            header_needed = True
        else:
            header_needed = False
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            new_data_df.to_csv(
                self.state_path, mode="a", header=header_needed, index=False
            )
        except Exception as e:
            logging.error(f"FEHLER beim Schreiben der CSV-Datei: {e}")

    def one_generation(self):

        pils = []

        bc = 0
        for batch in self.create_batches(
            [candidate.initial_noise for candidate in self.population]
        ):

            pil_images = self.generative_model.generate_batch(batch, self.prompt)
            pils.extend(pil_images)
            print(f"Batch {bc} generiert")
            bc += 1
        print(f"Menge der Biler {len(pils)}")
        embeddings = []
        fidelity_scores = []
        for batch in self.create_batches(pils):
            batch_embeddings = self.embedding_model.batch_image_features_extraction(
                batch
            )

            batch_scores = self.prompt_fidelity_evaluator.evaluate_batch(batch)

            embeddings.extend(batch_embeddings)
            fidelity_scores.extend(batch_scores)
        print(embeddings[0].shape)
        print(f"Menge der Embeddings {len(embeddings)}")

        self.evaluator.update(embeddings)
        scores = self.evaluator.evaluate_batch(embeddings)

        if self.generations_done == 0:
            self.global_evaluator.update(embeddings)

        global_scores = self.global_evaluator.evaluate_batch(embeddings)
        print(f"Menge der Scores {len(scores)}")

        diversity_score = self.diversity_evaluator.evalute_batch(embeddings)
        captions = []
        if self.caption_model is not None:

            for batch in self.create_batches(pils):
                batch_captions = self.caption_model.caption_batch(batch)
                captions.extend(batch_captions)

        for i, candidate in enumerate(self.population):
            candidate.pil_image = pils[i]
            candidate.blip2_embedding = embeddings[i]
            candidate.evaluation_scores.append(scores[i])
            candidate.global_score = global_scores[i]
            candidate.prompt_fidelity = fidelity_scores[i]
            candidate.diversity_score = diversity_score
            candidate.calculate_fitness()
            if self.caption_model is not None:
                candidate.caption = captions[i]

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

                child = copy.deepcopy(parent1)
                child.end_generation = self.generations_done
                child.id = self.noise_factory._create_id()
                log_entry = "Cross"
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
