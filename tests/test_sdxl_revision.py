from types import SimpleNamespace

import pytest
import torch
from src.huggingface_models.model_loader import ModelLoader
from src.huggingface_models.text_to_image.stable_diffusion_xl import (
    StableDiffusionXLModel,
    StableDiffusionXLPipeline,
)
from src.model_revisions import DEFAULT_SDXL_REVISION
from src.sdxl_options import (
    DEFAULT_SDXL_GUIDANCE_SCALE,
    DEFAULT_SDXL_NUM_INFERENCE_STEPS,
)


class EulerDiscreteScheduler:
    def __init__(self):
        self.config = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "prediction_type": "epsilon",
            "timestep_spacing": "leading",
        }


def test_sdxl_metadata_records_verified_snapshot(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshots" / DEFAULT_SDXL_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    pipeline = SimpleNamespace(
        config={},
        scheduler=EulerDiscreteScheduler(),
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
    assert metadata["num_inference_steps"] == 50
    assert metadata["guidance_scale"] == 7.5
    assert metadata["scheduler_class"] == "EulerDiscreteScheduler"
    assert metadata["scheduler_config"]["prediction_type"] == "epsilon"


def test_loader_refuses_to_reuse_different_sdxl_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(ModelLoader, "_instances", {})
    monkeypatch.setattr(
        "src.huggingface_models.model_loader.StableDiffusionXLModel",
        lambda *a, revision, num_inference_steps, guidance_scale: SimpleNamespace(
            requested_revision=revision,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ),
    )
    loader = ModelLoader(cache_dir=str(tmp_path))
    first = loader.load_sdxl(revision=DEFAULT_SDXL_REVISION)
    assert first.requested_revision == DEFAULT_SDXL_REVISION
    assert loader.load_sdxl(revision=DEFAULT_SDXL_REVISION) is first
    reused = ModelLoader(cache_dir=str(tmp_path))
    assert reused is loader
    assert reused.load_sdxl(revision=DEFAULT_SDXL_REVISION) is first
    with pytest.raises(ValueError, match="different requested configuration"):
        reused.load_sdxl(revision="a" * 40)
    with pytest.raises(ValueError, match="different requested configuration"):
        reused.load_sdxl(
            revision=DEFAULT_SDXL_REVISION,
            guidance_scale=DEFAULT_SDXL_GUIDANCE_SCALE + 0.5,
        )
    with pytest.raises(ValueError, match="different requested configuration"):
        reused.load_sdxl(
            revision=DEFAULT_SDXL_REVISION,
            num_inference_steps=DEFAULT_SDXL_NUM_INFERENCE_STEPS + 1,
        )
    with pytest.raises(ValueError, match="different cache_dir"):
        ModelLoader(cache_dir=str(tmp_path / "other-cache"))
