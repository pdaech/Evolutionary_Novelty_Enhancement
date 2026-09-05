from types import SimpleNamespace

import pytest
import torch

from src.huggingface_models.model_loader import ModelLoader
from src.huggingface_models.text_to_image.stable_diffusion_xl import (
    StableDiffusionXLModel,
    StableDiffusionXLPipeline,
)
from src.model_revisions import DEFAULT_SDXL_REVISION


def test_sdxl_metadata_records_verified_snapshot(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshots" / DEFAULT_SDXL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    pipeline = SimpleNamespace(
        config={},
        to=lambda **kwargs: None,
        set_progress_bar_config=lambda **kwargs: None,
    )
    monkeypatch.setattr(
        StableDiffusionXLPipeline, "download", lambda *a, **kw: str(snapshot)
    )
    monkeypatch.setattr(
        StableDiffusionXLPipeline, "from_pretrained", lambda *a, **kw: pipeline
    )
    model = StableDiffusionXLModel(
        "cpu", torch.float32, str(tmp_path), revision=DEFAULT_SDXL_REVISION
    )

    metadata = model.config_metadata()
    assert metadata["requested_revision"] == DEFAULT_SDXL_REVISION
    assert metadata["resolved_revision"] == DEFAULT_SDXL_REVISION
    assert metadata["revision_source"] == "huggingface_cache_snapshot"
    assert metadata["snapshot_path"] == str(snapshot)


def test_loader_refuses_to_reuse_different_sdxl_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(ModelLoader, "_instances", {})
    monkeypatch.setattr(
        "src.huggingface_models.model_loader.StableDiffusionXLModel",
        lambda *a, revision: SimpleNamespace(requested_revision=revision),
    )
    loader = ModelLoader(cache_dir=str(tmp_path))
    first = loader.load_sdxl(revision=DEFAULT_SDXL_REVISION)
    assert first.requested_revision == DEFAULT_SDXL_REVISION
    assert loader.load_sdxl(revision=DEFAULT_SDXL_REVISION) is first
    reused = ModelLoader(cache_dir=str(tmp_path))
    assert reused is loader
    assert reused.load_sdxl(revision=DEFAULT_SDXL_REVISION) is first
    with pytest.raises(ValueError, match="different requested revision"):
        reused.load_sdxl(revision="a" * 40)
    with pytest.raises(ValueError, match="different cache_dir"):
        ModelLoader(cache_dir=str(tmp_path / "other-cache"))
