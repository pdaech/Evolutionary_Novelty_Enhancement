"""Gemma image scoring for one preregistered fitness construct."""

from src.evaluators.gemma_creativity_evaluator import GemmaCreativityEvaluator
from src.gemma_constructs import ADJECTIVES, scoring_prompt


class GemmaConstructEvaluator(GemmaCreativityEvaluator):
    """Preserve the validated inference path while varying only score wording."""

    def __init__(self, construct: str, **kwargs) -> None:
        if construct not in ADJECTIVES:
            raise ValueError(f"Unknown Gemma fitness construct: {construct!r}")
        if "prompt" in kwargs:
            raise ValueError("Construct prompt is fixed by the registered condition")
        super().__init__(prompt=scoring_prompt(construct), **kwargs)
        self.construct = construct
        self.name = f"Gemma4{construct.title()}"

    def config_metadata(self) -> dict:
        result = super().config_metadata()
        result["objective"] = f"maximize_current_image_{self.construct}"
        result["construct"] = self.construct
        return result
