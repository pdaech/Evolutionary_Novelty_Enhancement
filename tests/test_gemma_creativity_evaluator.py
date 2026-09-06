import random
from types import SimpleNamespace

import pytest
from PIL import Image

from src.evaluators.gemma_creativity_evaluator import (
    DEFAULT_CREATIVITY_PROMPT,
    DEFAULT_MODEL_REVISION,
    GemmaCreativityEvaluator,
    _configure_batch_padding,
    _configure_image_token_budget,
    _parse_score,
    validate_gemma_runtime,
)
from src.gemma_options import (
    DEFAULT_IMAGE_TOKEN_BUDGET,
    validate_image_token_budget,
)


def _fake_torch(version: str, capability: tuple[int, int]):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda index: capability,
    )
    return SimpleNamespace(__version__=version, cuda=cuda)


def test_a100_rejects_torch_28_before_model_loading():
    with pytest.raises(RuntimeError, match="A100/SM80 requires PyTorch >=2.9"):
        validate_gemma_runtime(_fake_torch("2.8.0+cu128", (8, 0)))


def test_a100_accepts_torch_29():
    validate_gemma_runtime(_fake_torch("2.9.1+cu128", (8, 0)))


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
    evaluator.prompt = DEFAULT_CREATIVITY_PROMPT
    evaluator.max_new_tokens = 64
    evaluator._pipeline = RecordingPipeline()

    random.seed(9876)
    random_state = random.getstate()
    result = evaluator.evaluate(Image.new("RGB", (4, 4), color="white"))

    assert result["score"] == 4.5
    assert result["raw_response"] == '{"score": 4.5}'
    assert calls["generate_kwargs"] == {"max_new_tokens": 64, "do_sample": False}
    content = calls["text"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[1] == {"type": "text", "text": DEFAULT_CREATIVITY_PROMPT}
    assert random.getstate() == random_state


def test_evaluate_batch_preserves_order_across_pipeline_chunks():
    calls = []
    raw_responses = iter(['{"score": 1}', '{"score": 2}', '{"score": 3}'])

    class RecordingPipeline:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return [[{"generated_text": next(raw_responses)}] for _ in kwargs["text"]]

    evaluator = object.__new__(GemmaCreativityEvaluator)
    evaluator.prompt = DEFAULT_CREATIVITY_PROMPT
    evaluator.max_new_tokens = 64
    evaluator.batch_size = 2
    evaluator._pipeline = RecordingPipeline()

    results = evaluator.evaluate_batch(
        [Image.new("RGB", (4, 4), color=index) for index in range(3)]
    )

    assert [result["score"] for result in results] == [1.0, 2.0, 3.0]
    assert [len(call["text"]) for call in calls] == [2, 1]
    assert [call["batch_size"] for call in calls] == [2, 2]
    assert all(
        call["generate_kwargs"] == {"max_new_tokens": 64, "do_sample": False}
        for call in calls
    )


def test_evaluate_returns_invalid_rating_for_pipeline_audit():
    class InvalidPipeline:
        def __call__(self, **kwargs):
            return [{"generated_text": '{"score": 0}'}]

    evaluator = object.__new__(GemmaCreativityEvaluator)
    evaluator.prompt = DEFAULT_CREATIVITY_PROMPT
    evaluator.max_new_tokens = 64
    evaluator._pipeline = InvalidPipeline()

    result = evaluator.evaluate(Image.new("RGB", (4, 4), color="white"))

    assert result == {
        "name": "Gemma4Creativity",
        "score": 0.0,
        "raw_response": '{"score": 0}',
        "parse_error": "score_out_of_range",
    }


def test_scientific_default_pins_validated_model_revision():
    assert DEFAULT_MODEL_REVISION == "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"


def test_gemma_image_token_budget_defaults_to_validated_setting():
    assert DEFAULT_IMAGE_TOKEN_BUDGET == 280


def test_configure_image_token_budget_updates_chat_pipeline_processor():
    image_processor = SimpleNamespace(max_soft_tokens=280)
    inference_pipeline = SimpleNamespace(
        processor=SimpleNamespace(image_processor=image_processor)
    )

    _configure_image_token_budget(inference_pipeline, 140)

    assert image_processor.max_soft_tokens == 140


def test_configure_batch_padding_aligns_feature_extractor_to_tokenizer():
    inference_pipeline = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="left"),
        feature_extractor=SimpleNamespace(padding_side="right"),
    )

    padding = _configure_batch_padding(inference_pipeline, batch_size=2)

    assert inference_pipeline.tokenizer.padding_side == "left"
    assert inference_pipeline.feature_extractor.padding_side == "left"
    assert padding == {"tokenizer": "left", "feature_extractor": "left"}


def test_configure_batch_padding_does_not_mutate_serial_pipeline():
    inference_pipeline = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="left"),
        feature_extractor=SimpleNamespace(padding_side="right"),
    )

    padding = _configure_batch_padding(inference_pipeline, batch_size=1)

    assert inference_pipeline.feature_extractor.padding_side == "right"
    assert padding == {"tokenizer": "left", "feature_extractor": "right"}


@pytest.mark.parametrize("value", [0, 69, 100, 281, 2048])
def test_gemma_image_token_budget_rejects_unsupported_values(value):
    with pytest.raises(ValueError, match="must be one of"):
        validate_image_token_budget(value)
