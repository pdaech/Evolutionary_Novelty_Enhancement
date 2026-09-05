import logging
import os
import random
from pathlib import Path

import torch
from diffusers.utils.logging import disable_progress_bar
from dotenv import load_dotenv

from src.crossover import UniformCrossover
from src.evaluators.gemma_creativity_evaluator import (
    DEFAULT_MODEL_REVISION,
    GemmaCreativityEvaluator,
    validate_gemma_runtime,
)
from src.evaluators.local_max_mean_divergence_evaluator import (
    LocalMaxMeanDivergenceEvaluator,
)
from src.factorys import NoiseFactory
from src.huggingface_models import ModelLoader
from src.mutators.uniform_gaussian_mutator import UniformGaussianMutator
from src.pipelines.genetic_algorithm import GeneticAlgorithmPipeline
from src.selector_functions.tournament_selector import TournamentSelector
from src.utils.arg_parser import args

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def _huggingface_cache_dir() -> str | None:
    """Resolve the Hub repository cache without shadowing ``$HF_HOME/hub``."""
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        return hub_cache
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return str(Path(hf_home) / "hub")
    legacy_cache = os.environ.get("HF_CACHE")
    return legacy_cache or None


def main(
    experiment_id: str,
    directory: str,
    base_id: str,
    prompt: str,
    num_generations: int,
    population_size: int,
    batch_size: int,
    evaluator_name: str,
    seed: int,
    gemma_model: str,
    gemma_revision: str,
    gemma_max_new_tokens: int,
):

    normalized_evaluator = evaluator_name.strip().lower().replace("_", "-")
    if normalized_evaluator in {"gemma", "gemma4", "gemma-creativity"}:
        validate_gemma_runtime(torch)

    selector = TournamentSelector(tournament_size=3)
    mutator = UniformGaussianMutator(mutation_rate=0.1, mutation_strengh=0.2)
    crossover = UniformCrossover()

    noise_factory = NoiseFactory()
    cache_dir = _huggingface_cache_dir()
    ml = ModelLoader(cache_dir=cache_dir)

    sdxl = ml.load_sdxl()

    if normalized_evaluator in {"gemma", "gemma4", "gemma-creativity"}:
        evaluator = GemmaCreativityEvaluator(
            model_id=gemma_model,
            revision=gemma_revision,
            cache_dir=cache_dir or None,
            max_new_tokens=gemma_max_new_tokens,
        )
        global_evaluator = None
        embed = None
        caption_model = None
        fitness_aggregation = "latest"
        compute_auxiliary_metrics = False
    elif normalized_evaluator in {"novelty", "local-max-mean-divergence"}:
        evaluator = LocalMaxMeanDivergenceEvaluator(prompt, "cuda")
        global_evaluator = LocalMaxMeanDivergenceEvaluator(prompt, "cuda")
        embed = ml.load_blip2_embeddings()
        caption_model = ml.load_blip2_captions()
        fitness_aggregation = "mean_history"
        compute_auxiliary_metrics = True
    else:
        raise ValueError(
            f"Unknown evaluator {evaluator_name!r}; use 'novelty' or 'gemma-creativity'"
        )

    unique_experiment_id = f"{base_id}_{experiment_id}"

    print(f"Start Next: {unique_experiment_id}")
    pipe = GeneticAlgorithmPipeline(
        generative_model=sdxl,
        prompt=prompt,
        embedding_model=embed,
        crossover_operation=crossover,
        selector=selector,
        mutator=mutator,
        noise_factory=noise_factory,
        evaluator=evaluator,
        global_evaluator=global_evaluator,
        experiment_id=unique_experiment_id,
        num_generations=num_generations,
        population_size=population_size,
        initial_mutation_rate=0.05,
        initial_crossover_rate=0.9,
        elite_size=1,
        batch_size=batch_size,
        caption_model=caption_model,
        result_path=directory,
        fitness_aggregation=fitness_aggregation,
        compute_auxiliary_metrics=compute_auxiliary_metrics,
    )

    pipe.save_config()
    pipe.run()


if __name__ == "__main__":
    load_dotenv()
    disable_progress_bar()

    parsed_args = args()

    experiment_id = parsed_args.experiment_id
    directory = parsed_args.directory
    base_id = parsed_args.id
    seed = parsed_args.seed
    prompt = parsed_args.prompt
    num_generations = parsed_args.num_generations

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    main(
        experiment_id=experiment_id,
        directory=directory,
        base_id=base_id,
        prompt=prompt,
        num_generations=num_generations,
        population_size=parsed_args.population_size,
        batch_size=parsed_args.batch_size,
        evaluator_name=parsed_args.evaluator,
        seed=seed,
        gemma_model=parsed_args.gemma_model,
        gemma_revision=parsed_args.gemma_revision or DEFAULT_MODEL_REVISION,
        gemma_max_new_tokens=parsed_args.gemma_max_new_tokens,
    )
