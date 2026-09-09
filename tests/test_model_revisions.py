from types import SimpleNamespace

import pytest
from src.gemma_options import DEFAULT_IMAGE_TOKEN_BUDGET
from src.model_revisions import DEFAULT_SDXL_REVISION, load_pinned_pipeline
from src.sdxl_options import (
    DEFAULT_SDXL_GUIDANCE_SCALE,
    DEFAULT_SDXL_NUM_INFERENCE_STEPS,
)
from src.utils.arg_parser import args


def _snapshot(tmp_path, revision=DEFAULT_SDXL_REVISION):
    snapshot = tmp_path / "models--stabilityai--sdxl" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    return snapshot


def test_load_uses_verified_local_snapshot_without_pipeline_commit(tmp_path):
    snapshot = _snapshot(tmp_path)
    calls = {}
    model = SimpleNamespace(config={})  # Diffusers need not retain _commit_hash.

    class FakePipeline:
        @staticmethod
        def download(model_id, **kwargs):
            calls["download"] = (model_id, kwargs)
            return str(snapshot)

        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["load"] = (path, kwargs)
            return model

    result, metadata = load_pinned_pipeline(
        FakePipeline,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        revision=DEFAULT_SDXL_REVISION,
        cache_dir=str(tmp_path),
        dtype="fake-float16",
    )

    assert result is model
    assert calls["download"] == (
        "stabilityai/stable-diffusion-xl-base-1.0",
        {
            "revision": DEFAULT_SDXL_REVISION,
            "cache_dir": str(tmp_path),
            "use_safetensors": True,
        },
    )
    assert calls["load"] == (
        str(snapshot),
        {
            "local_files_only": True,
            "torch_dtype": "fake-float16",
            "use_safetensors": True,
        },
    )
    assert metadata["requested_revision"] == DEFAULT_SDXL_REVISION
    assert metadata["resolved_revision"] == DEFAULT_SDXL_REVISION
    assert metadata["snapshot_path"] == str(snapshot)
    assert metadata["revision_source"] == "huggingface_cache_snapshot"


@pytest.mark.parametrize("revision", ["main", "v1.0", "4621659", "", "x" * 40])
def test_mutable_or_malformed_revision_rejected_before_download(revision):
    with pytest.raises(ValueError, match="full 40-character"):
        load_pinned_pipeline(
            object(), model_id="sdxl", revision=revision, cache_dir=None, dtype=None
        )


@pytest.mark.parametrize("fault", ["wrong_commit", "wrong_layout", "missing_index"])
def test_unverifiable_snapshot_is_not_loaded(tmp_path, fault):
    snapshot = _snapshot(
        tmp_path,
        revision="a" * 40 if fault == "wrong_commit" else DEFAULT_SDXL_REVISION,
    )
    if fault == "wrong_layout":
        snapshot = tmp_path / DEFAULT_SDXL_REVISION
        snapshot.mkdir()
        (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    elif fault == "missing_index":
        (snapshot / "model_index.json").unlink()
    # Deliberately no from_pretrained: failures must be detected before loading.
    pipeline = SimpleNamespace(download=lambda *a, **kw: str(snapshot))

    with pytest.raises(RuntimeError, match="refusing to load|missing model_index"):
        load_pinned_pipeline(
            pipeline,
            model_id="sdxl",
            revision=DEFAULT_SDXL_REVISION,
            cache_dir=None,
            dtype=None,
        )


def test_cli_defaults_to_smoke_snapshot_and_accepts_explicit_commit(monkeypatch):
    argv = ["run.py", "--prompt", "cat", "--experiment_id", "unit"]
    monkeypatch.setattr("sys.argv", argv)
    assert args().sdxl_revision == DEFAULT_SDXL_REVISION
    monkeypatch.setattr("sys.argv", [*argv, "--sdxl_revision", "b" * 40])
    assert args().sdxl_revision == "b" * 40


def test_cli_defaults_to_thesis_sdxl_settings_and_accepts_overrides(monkeypatch):
    argv = ["run.py", "--prompt", "a cat", "--experiment_id", "unit"]
    monkeypatch.setattr("sys.argv", argv)
    parsed = args()
    assert parsed.sdxl_num_inference_steps == DEFAULT_SDXL_NUM_INFERENCE_STEPS
    assert parsed.sdxl_guidance_scale == DEFAULT_SDXL_GUIDANCE_SCALE

    monkeypatch.setattr(
        "sys.argv",
        [
            *argv,
            "--sdxl_num_inference_steps",
            "25",
            "--sdxl_guidance_scale",
            "6.0",
        ],
    )
    parsed = args()
    assert parsed.sdxl_num_inference_steps == 25
    assert parsed.sdxl_guidance_scale == 6.0


def test_cli_defaults_to_280_image_tokens_and_accepts_140(monkeypatch):
    argv = ["run.py", "--prompt", "cat", "--experiment_id", "unit"]
    monkeypatch.setattr("sys.argv", argv)
    assert args().gemma_image_token_budget == DEFAULT_IMAGE_TOKEN_BUDGET
    monkeypatch.setattr("sys.argv", [*argv, "--gemma_image_token_budget", "140"])
    assert args().gemma_image_token_budget == 140


def test_cli_rejects_unsupported_image_token_budget(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "--prompt",
            "cat",
            "--experiment_id",
            "unit",
            "--gemma_image_token_budget",
            "100",
        ],
    )
    with pytest.raises(SystemExit) as error:
        args()
    assert error.value.code == 2


def test_cli_accepts_positive_gemma_batch_size_and_rejects_zero(monkeypatch):
    argv = ["run.py", "--prompt", "cat", "--experiment_id", "unit"]
    monkeypatch.setattr("sys.argv", [*argv, "--gemma_batch_size", "2"])
    assert args().gemma_batch_size == 2

    monkeypatch.setattr("sys.argv", [*argv, "--gemma_batch_size", "0"])
    with pytest.raises(SystemExit) as error:
        args()
    assert error.value.code == 2


def test_cli_rejects_mutable_revision(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "--prompt",
            "cat",
            "--experiment_id",
            "unit",
            "--sdxl_revision",
            "main",
        ],
    )
    with pytest.raises(SystemExit) as error:
        args()
    assert error.value.code == 2
