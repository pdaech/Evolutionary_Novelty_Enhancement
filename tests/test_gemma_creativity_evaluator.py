import pytest
from PIL import Image

from src.evaluators.gemma_creativity_evaluator import (
    DEFAULT_CREATIVITY_PROMPT,
    GemmaCreativityEvaluator,
    _parse_score,
)


def test_parse_score_accepts_json_and_fenced_json():
    assert _parse_score('{"score": 4.25}', 1, 5) == (4.25, None)
    assert _parse_score('```json\n{"score": 3}\n```', 1, 5) == (3.0, None)


def test_parse_score_rejects_out_of_range_value():
    assert _parse_score('{"score": 0}', 1, 5) == (0.0, "score_out_of_range")


def test_evaluate_routes_greedy_generation_options_and_image():
    calls = {}

    class RecordingPipeline:
        def __call__(self, **kwargs):
            calls.update(kwargs)
            return [{"generated_text": '{"score": 4.5}'}]

    evaluator = object.__new__(GemmaCreativityEvaluator)
    evaluator.seed = 123
    evaluator.prompt = DEFAULT_CREATIVITY_PROMPT
    evaluator.max_new_tokens = 64
    evaluator._pipeline = RecordingPipeline()
    seeds = []
    evaluator._set_seed = seeds.append

    result = evaluator.evaluate(Image.new("RGB", (4, 4), color="white"))

    assert result["score"] == 4.5
    assert result["raw_response"] == '{"score": 4.5}'
    assert seeds == [123]
    assert calls["generate_kwargs"] == {"max_new_tokens": 64, "do_sample": False}
    content = calls["text"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[1] == {"type": "text", "text": DEFAULT_CREATIVITY_PROMPT}


def test_evaluate_fails_closed_on_invalid_rating():
    class InvalidPipeline:
        def __call__(self, **kwargs):
            return [{"generated_text": '{"score": 0}'}]

    evaluator = object.__new__(GemmaCreativityEvaluator)
    evaluator.seed = 123
    evaluator.prompt = DEFAULT_CREATIVITY_PROMPT
    evaluator.max_new_tokens = 64
    evaluator._pipeline = InvalidPipeline()
    evaluator._set_seed = lambda seed: None

    with pytest.raises(ValueError, match="score_out_of_range"):
        evaluator.evaluate(Image.new("RGB", (4, 4), color="white"))
