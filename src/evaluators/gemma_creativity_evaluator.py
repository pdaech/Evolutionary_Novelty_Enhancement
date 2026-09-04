from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from PIL import Image

from src.evaluators.base_evaluator import Evaluator

DEFAULT_MODEL_ID = "google/gemma-4-26B-A4B-it"
DEFAULT_CREATIVITY_PROMPT = (
    "How creative do you find the image? "
    "Use a continuous score from 1 to 5, where higher scores mean that you find "
    "the image more creative. Return only valid JSON in this exact shape: "
    '{"score": <number>}'
)


class GemmaCreativityEvaluator(Evaluator):
    """Use Gemma 4's image creativity rating as a maximization objective."""

    name = "Gemma4Creativity"
    score_min = 1.0
    score_max = 5.0

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        cache_dir: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 64,
        seed: int = 2025,
        prompt: str = DEFAULT_CREATIVITY_PROMPT,
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import pipeline, set_seed
        except ImportError as exc:
            raise RuntimeError(
                "Gemma creativity fitness requires torch, transformers>=5.5, and accelerate"
            ) from exc

        if not hasattr(torch, dtype):
            raise ValueError(f"Unknown torch dtype: {dtype}")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")

        options: dict[str, Any] = {"device_map": device_map}
        if revision:
            options["revision"] = revision
        if cache_dir:
            options["model_kwargs"] = {"cache_dir": cache_dir}
        dtype_parameter = (
            "dtype"
            if "dtype" in inspect.signature(pipeline).parameters
            else "torch_dtype"
        )
        options[dtype_parameter] = getattr(torch, dtype)

        self.model_id = model_id
        self.requested_revision = revision
        self.cache_dir = cache_dir
        self.dtype = dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.prompt = prompt
        self._set_seed = set_seed
        self._pipeline = pipeline("image-text-to-text", model=model_id, **options)
        self._resolved_revision = _resolved_revision(self._pipeline)
        self._hardware = _hardware_metadata(self._pipeline, torch)
        self._software = {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "accelerate": _package_version("accelerate"),
            "cuda_runtime": (
                str(torch.version.cuda) if torch.version.cuda is not None else None
            ),
            "cudnn": (
                str(torch.backends.cudnn.version())
                if torch.cuda.is_available()
                and torch.backends.cudnn.version() is not None
                else None
            ),
        }

    def evaluate(self, image: Image.Image, *args, **kwargs) -> dict[str, Any]:
        self._set_seed(self.seed)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        output = self._pipeline(
            text=messages,
            generate_kwargs={
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            },
        )
        raw_response = _extract_text(output)
        score, parse_error = _parse_score(
            raw_response,
            score_min=self.score_min,
            score_max=self.score_max,
        )
        if parse_error is not None:
            raise ValueError(
                f"Gemma creativity response could not be used ({parse_error}): "
                f"{raw_response!r}"
            )
        return {
            "name": self.name,
            "score": score,
            "raw_response": raw_response,
            "parse_error": None,
        }

    def evaluate_batch(
        self, images: list[Image.Image], *args, **kwargs
    ) -> list[dict[str, Any]]:
        # Evaluate serially to keep the combined SDXL + Gemma memory footprint predictable.
        scores = []
        for index, image in enumerate(images):
            try:
                scores.append(self.evaluate(image))
            except Exception as exc:
                raise RuntimeError(
                    f"Gemma creativity evaluation failed for batch item {index}"
                ) from exc
        return scores

    @classmethod
    def need(cls) -> None:
        """Gemma consumes generated PIL images, not embedding tensors."""
        return

    def config_metadata(self) -> dict[str, Any]:
        return {
            "objective": "maximize_current_image_creativity",
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self._resolved_revision,
            "dtype": self.dtype,
            "device_map": self.device_map,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "prompt": self.prompt,
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
            "generation": {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            },
            "software": self._software,
            "hardware": self._hardware,
        }


def _extract_text(output: Any) -> str:
    """Normalize output shapes used by recent Transformers image-text pipelines."""
    value = output[0] if isinstance(output, list) and output else output
    if isinstance(value, dict):
        value = value.get("generated_text", value.get("text", value))
    if isinstance(value, list):
        assistants = [
            item
            for item in value
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        value = assistants[-1].get("content") if assistants else value[-1]
    if isinstance(value, list):
        text_parts = [item.get("text", "") for item in value if isinstance(item, dict)]
        value = "".join(text_parts)
    if isinstance(value, dict):
        value = value.get("text", value.get("content", value))
    return str(value)


def _parse_score(
    text: str, score_min: float, score_max: float
) -> tuple[float | None, str | None]:
    if not text or not text.strip():
        return None, "empty_response"

    candidates = [text.strip()]
    candidates.extend(
        re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    )
    candidates.extend(re.findall(r"\{.*?\}", text, flags=re.DOTALL))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and "score" in payload:
            return _validate_score(payload["score"], score_min, score_max)

    match = re.search(
        r"(?:^|[\s\"'])score[\s\"']*[:=]\s*([-+]?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return _validate_score(match.group(1), score_min, score_max)
    return None, "score_not_found"


def _validate_score(
    value: Any, score_min: float, score_max: float
) -> tuple[float | None, str | None]:
    if isinstance(value, bool):
        return None, "score_not_numeric"
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None, "score_not_numeric"
    if not math.isfinite(score):
        return None, "score_not_finite"
    if not score_min <= score <= score_max:
        return score, "score_out_of_range"
    return score, None


def _resolved_revision(inference_pipeline: Any) -> str | None:
    model = getattr(inference_pipeline, "model", None)
    config = getattr(model, "config", None)
    commit = getattr(config, "_commit_hash", None)
    return str(commit) if commit is not None else None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _hardware_metadata(inference_pipeline: Any, torch: Any) -> dict[str, Any]:
    model = getattr(inference_pipeline, "model", None)
    device = getattr(model, "device", getattr(inference_pipeline, "device", None))
    device_map = getattr(model, "hf_device_map", None)
    cuda_available = bool(torch.cuda.is_available())
    gpus = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            gpus.append(
                {
                    "index": index,
                    "name": str(properties.name),
                    "capability": f"{capability[0]}.{capability[1]}",
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "device": str(device) if device is not None else "unknown",
        "device_map": _json_safe(device_map),
        "cuda_available": cuda_available,
        "cuda_device_count": len(gpus),
        "gpus": gpus,
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
