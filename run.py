import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from src.utils.arg_parser import args
from src.crossover import (
    UniformCrossover,
    NPointEqualCrossoverFunction,
    NPointEqualPixelCrossoverFunction,
    NPointRandomCrossoverFunction,
    ArithmeticRenormedCrossoverFunction,
    UniformPixelCrossover,
    NPointRandomPixelCrossoverFunction,
)
from src.evaluators.local_kernel_density_evaluator import LocalKernelDensityEvaluator
from src.evaluators.local_kNN_evaluator import (
    LocalkNNEvaluator,
    LocalArchivkNNEvaluator,
)
from src.evaluators.global_max_mean_divergence_evaluator import (
    GlobalMaxMeanDivergenceEvaluator,
)
from src.evaluators.local_max_mean_divergence_evaluator import (
    LocalMaxMeanDivergenceEvaluator,
    LocalArchivMaxMeanDivergenceEvaluator,
)
from src.huggingface_models import ModelLoader
from src.mutators.uniform_gaussian_mutator import UniformGaussianMutator
from src.pipelines.genetic_algorithm import GeneticAlgorithmPipeline
from src.selector_functions.tournament_selector import TournamentSelector
from src.factorys import NoiseFactory
from diffusers.utils.logging import disable_progress_bar
import random
import torch

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def main(experiment_id, directory, base_id, prompt, num_generations):

    selector = TournamentSelector(tournament_size=3)
    mutator = UniformGaussianMutator(mutation_rate=0.1, mutation_strengh=0.2)
    crossover = UniformCrossover()

    evaluator = LocalMaxMeanDivergenceEvaluator(prompt, "cuda")
    global_evaluator = LocalMaxMeanDivergenceEvaluator(prompt, "cuda")

    noise_factory = NoiseFactory()
    ml = ModelLoader(cache_dir=os.environ.get("HF_CACHE", ""))

    sdxl = ml.load_sdxl()
    embed = ml.load_blip2_embeddings()
    caption_model = ml.load_blip2_captions()

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
        population_size=100,
        initial_mutation_rate=0.05,
        initial_crossover_rate=0.9,
        elite_size=1,
        batch_size=5,
        caption_model=caption_model,
        result_path=directory,
    )

    pipe.save_config()
    pipe.run()


if __name__ == "__main__":
    load_dotenv()

    parsed_args = args()

    experiment_id = parsed_args.experiment_id
    directory = parsed_args.directory
    base_id = parsed_args.id
    seed = parsed_args.seed
    prompt = parsed_args.prompt
    num_generations = parsed_args.num_generations

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    main(
        experiment_id=experiment_id,
        directory=directory,
        base_id=base_id,
        prompt=prompt,
        num_generations=num_generations,
    )
