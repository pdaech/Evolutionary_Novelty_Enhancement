from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src import evaluators
from src.crossover.base_crossover import CrossoverFunction
from src.evaluators.base_evaluator import Evaluator
from src.factory import NoiseFactory
from src.huggingface_models.base_strategy import (
    GenerativModelStrategy,
    EmbeddingModelStrategy,
    CaptionModelStrategy,
)
from src.mutators.base_mutator import MutationFunction
from src.pipelines.genetic_algorithm import GeneticAlgorithmPipeline
from src.pipelines.pipeline import Pipeline
from src.selector_functions.base_selectorf_unction import SelectorFunction


@dataclass
class PipelineBuilder(ABC):

    model: GenerativModelStrategy = None
    embedding_model: EmbeddingModelStrategy = None
    caption_model: CaptionModelStrategy = None
    crossover: CrossoverFunction = None
    selector: SelectorFunction = None
    mutator: MutationFunction = None

    population_size: int = None
    num_generations: int = None
    elite_size: int = None
    initial_mutation_rate: float = None
    initial_crossover_rate: float = None
    prompt: str = None

    def set_generativ_model(self, model: GenerativModelStrategy) -> "PipelineBuilder":
        self.model = model
        return self

    def set_embedding_model(self, model: EmbeddingModelStrategy) -> "PipelineBuilder":
        self.embedding_model = model
        return self

    def set_captioin_model(self, model: CaptionModelStrategy) -> "PipelineBuilder":
        self.captio_model = model
        return self

    def set_crossover(self, crossover: CrossoverFunction) -> "PipelineBuilder":
        self.crossover = crossover
        return self

    def set_selector(self, selector: SelectorFunction) -> "PipelineBuilder":
        self.selector = selector
        return self

    def set_mutator(self, mutator: MutationFunction) -> "PipelineBuilder":
        self.mutator = mutator
        return self

    def set_population_size(self, size: int) -> "PipelineBuilder":
        self.population_size = size
        return self

    def set_num_generations(self, num_generations: int) -> "PipelineBuilder":
        self.num_generations = num_generations
        return self

    def set_initial_mutation_rate(self, rate: float) -> "PipelineBuilder":
        self.initial_mutation_rate = rate
        return self

    def set_initial_crossover_rate(self, rate: float) -> "PipelineBuilder":
        self.initial_crossover_rate = rate
        return self

    def set_elite_size(self, size: int) -> "PipelineBuilder":
        self.elite_size = size
        return self

    def set_prompt(self, prompt: str) -> "PipelineBuilder":
        self.prompt = prompt
        return self

    @abstractmethod
    def build(self) -> Pipeline:
        raise NotImplementedError


class GeneticAlgorithmPipelineBuilder(PipelineBuilder):

    def set_evaluator(self, evaluator: Evaluator) -> "PipelineBuilder":
        self.evaluators.append(evaluator)
        return self

    def build(self) -> Pipeline:

        noise_factory = NoiseFactory()
        if evaluators[0].need():
            assert isinstance(
                self.embedding_model, evaluators[0].need()
            ), "Required Model is missing"

        ga = GeneticAlgorithmPipeline(
            generative_model=self.model,
            embedding_model=self.embedding_model,
            caption_model=self.caption_model,
            noise_factory=noise_factory,
            num_generations=self.num_generations,
            population_size=self.population_size,
            initial_mutation_rate=self.initial_mutation_rate,
            initial_crossover_rate=self.initial_crossover_rate,
            elite_size=self.elite_size,
            mutator=self.mutator,
            selector=self.selector,
            crossover_operation=self.crossover,
            evaluator=self.evaluators[0],
        )
        return ga


class NSGAIIPipelineBuilder(PipelineBuilder):
    embedding_models: list[EmbeddingModelStrategy] = None
    evaluators: list[Evaluator] = field(default_factory=list)
    evaluator_weights: list[float] = field(default_factory=list)

    def add_evaluator(
        self, evaluator: Evaluator, evaluator_weight: float
    ) -> "PipelineBuilder":
        self.evaluators.append(evaluator)
        self.evaluator_weights.append(evaluator_weight)
        return self

    def build(self) -> Pipeline:

        has_instance = []

        for evaluator in self.evaluators:
            if evaluator.need():
                has_instance.append(
                    any(
                        [
                            isinstance(model, evaluator.need())
                            for model in self.embedding_models
                        ]
                    )
                )

        assert all(has_instance), "One or more required models are missing"

        sum_evaluator_weights = sum(self.evaluator_weights)
        assert sum_evaluator_weights > 1.0, ""
        assert sum_evaluator_weights < 1.0, ""

        noise_factory = NoiseFactory()
        ga = NSGAIIPipelineBuilder(
            generative_model=self.model,
            embedding_model=self.embedding_model,
            caption_model=self.caption_model,
            noise_factory=noise_factory,
            num_generations=self.num_generations,
            population_size=self.population_size,
            initial_mutation_rate=self.initial_mutation_rate,
            initial_crossover_rate=self.initial_crossover_rate,
            elite_size=self.elite_size,
            mutator=self.mutator,
            selector=self.selector,
            crossover_operation=self.crossover,
            evaluator=self.evaluators[0],
        )
        return ga
