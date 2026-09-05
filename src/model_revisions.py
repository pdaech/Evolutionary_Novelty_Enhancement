"""Immutable model snapshots without importing GPU libraries at module import time."""

import re
from pathlib import Path
from typing import Any

DEFAULT_SDXL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"


def full_model_revision(value: str) -> str:
    """Require an immutable Hugging Face commit rather than a movable branch/tag."""
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(
            "Model revision must be a full 40-character lowercase commit hash"
        )
    return value


def load_pinned_pipeline(
    pipeline_class: Any,
    *,
    model_id: str,
    revision: str,
    cache_dir: str | None,
    dtype: Any,
) -> tuple[Any, dict[str, str]]:
    """Download only pipeline assets, then load the verified snapshot entirely locally."""
    full_model_revision(revision)
    # Diffusers filters the repository to the required pipeline files and reuses its cache.
    snapshot = Path(
        pipeline_class.download(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            use_safetensors=True,
        )
    ).absolute()
    if snapshot.parent.name != "snapshots" or snapshot.name != revision:
        raise RuntimeError(
            f"Requested model revision {revision}, but download returned {snapshot}. "
            "Cannot verify the model snapshot; refusing to load it."
        )
    if not (snapshot / "model_index.json").is_file():
        raise RuntimeError(f"Model snapshot is missing model_index.json: {snapshot}")

    pipeline = pipeline_class.from_pretrained(
        str(snapshot),
        local_files_only=True,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    return pipeline, {
        "requested_revision": revision,
        "resolved_revision": snapshot.name,
        "revision_source": "huggingface_cache_snapshot",
        "snapshot_path": str(snapshot),
    }
