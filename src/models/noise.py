from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from typing import Optional, Dict
import io
import zipfile


def _save_pil(
    pil_image: Image.Image,
    full_path: Path,
    file_format: str = "JPEG",
) -> None:
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if pil_image is not None:
        pil_image.save(full_path, format=file_format)
    else:
        raise ValueError()


@dataclass
class Noise:
    """
    Attributes:
    id (str): Unique identifier of the noise instance
    initial_noise (Optional[torch.Tensor]): Initial Gaussian noise
    initial_seed (int): Initial seed
    latent_representation (Optional[torch.Tensor]): Latent embedding of the generated image
    pil_image (Optional[PIL.Image.Image]): PIL image of the generated image
    blip2_embedding (Optional[torch.Tensor]): BLIP2 Image Encoder embedding vector of the generated image
    clip_embedding (Optional[torch.Tensor]): CLIP embedding vector of the generated image
    fitness (Optional[float]): Fitness score of the generated image
    evaluation_scores (Dict[str, float]): Individual evaluation scores from the evaluators
    start_generation (Optional[int]): Generation number of the first appearance of the noise instance
    end_generation (Optional[int]): Generation number of the last appearance of the noise instance
    """

    id: str
    initial_noise: torch.Tensor
    initial_seed: int = None
    latent_representation: torch.Tensor = None
    pil_image: Image.Image = None
    blip2_embedding: torch.Tensor = None
    clip_embedding: torch.Tensor = None
    fitness: float = None
    evaluation_scores: list[dict[str, float]] = field(default_factory=list)
    global_score: float = None
    prompt_fidelity: float = None
    diversity_score: float = None
    caption: str = None
    start_generation: int = None
    end_generation: int = None
    parent_1: str = ""
    parent_2: str = ""
    crossover: bool = None
    mutate: bool = None

    @property
    def filename(self):
        fitness = 0.0 if self.fitness is None else self.fitness
        return f"g{self.end_generation}_id{self.id}_f{fitness}"

    def save_pil_image(
        self,
        filepath: str,
        filename: str | None = None,
        file_format: str = "JPEG",
    ) -> None:

        try:
            path = Path(filepath)
            if filename:
                file = f"{filename}.{file_format}"
            else:
                file = f"{self.filename}.{file_format}"
            full_path = path / file
            _save_pil(
                pil_image=self.pil_image, full_path=full_path, file_format=file_format
            )
        except ValueError as e:
            raise ValueError(f"No pil image available for noise {self.id}: {e}") from e

    def save_noise_to_rgb(
        self,
        filepath: str,
        file_format: str = "JPEG",
    ) -> None:

        try:
            weights = (
                (60, -60, 25, -70),
                (60, -5, 15, -50),
                (60, 10, -5, -35),
            )

            weights_tensor = torch.t(
                torch.tensor(weights, dtype=self.initial_noise.dtype).to(
                    self.initial_noise.device
                )
            )
            biases_tensor = torch.tensor(
                (150, 140, 130), dtype=self.initial_noise.dtype
            ).to(self.initial_noise.device)
            rgb_tensor = torch.einsum(
                "...lxy,lr -> ...rxy", self.initial_noise, weights_tensor
            ) + biases_tensor.unsqueeze(-1).unsqueeze(-1)

            if rgb_tensor.dim() == 4 and rgb_tensor.shape[0] == 1:
                rgb_tensor = rgb_tensor.squeeze(0)

            if rgb_tensor.dim() == 3:
                image_array = (
                    rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
                )
            else:
                raise ValueError(
                    f"Unexpected tensor shape: {rgb_tensor.shape}. Expected [C, H, W] or [1, C, H, W]"
                )

            image = Image.fromarray(image_array)
        except Exception as e:
            raise Exception(f"Failed to convert initial noise to rgb image: {e}")

        try:
            path = Path(filepath)
            file = f"{self.filename}.{file_format}"
            full_path = path / file
            _save_pil(pil_image=image, full_path=full_path, file_format=file_format)
        except ValueError as e:
            raise ValueError(f"Error while saving image to RGB: {e}")

    def save_noise(
        self,
        filepath: str,
    ) -> None:
        path = Path(filepath)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.initial_noise, f"{filepath}/{self.filename}.pt")

    def save_representation(
        self,
        filepath: str,
    ) -> None:
        path = Path(filepath)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.latent_representation, f"{filepath}/img_{self.filename}.pt")

    def save_blip2(
        self,
        filepath: str,
    ) -> None:
        path = Path(filepath)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.blip2_embedding, f"{filepath}/blip2_{self.filename}.pt")

    def save_clip(
        self,
        filepath: str,
    ) -> None:
        path = Path(filepath)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.clip_embedding, f"{filepath}/clip_{self.filename}.pt")

    def calculate_fitness(self) -> None:

        if len(self.evaluation_scores) == 0:
            self.fitness = 0.0
        elif len(self.evaluation_scores) == 1:
            self.fitness = self.evaluation_scores[0]["score"]
        else:
            score_values = [d["score"] for d in self.evaluation_scores]
            self.fitness = sum(score_values) / len(score_values)

    def add_to_zip(self, zf: zipfile.ZipFile, folder_prefix: str = ""):
        """
        Schreibt Bild und Embeddings direkt in das übergebene ZipFile Objekt.
        """

        if self.pil_image is not None:

            img_buffer = io.BytesIO()
            self.pil_image.save(img_buffer, format="JPEG")
            zip_path_img = f"{folder_prefix}/images/{self.filename}.JPEG"
            zf.writestr(zip_path_img, img_buffer.getvalue())

        if self.blip2_embedding is not None:

            emb_buffer = io.BytesIO()
            torch.save(self.blip2_embedding, emb_buffer)
            zip_path_blip = f"{folder_prefix}/blip2/blip2_{self.filename}.pt"
            zf.writestr(zip_path_blip, emb_buffer.getvalue())

        if self.initial_noise is not None:

            noise_buffer = io.BytesIO()
            torch.save(self.initial_noise, noise_buffer)
            zip_path_noise = f"{folder_prefix}/noise/{self.filename}.pt"
            zf.writestr(zip_path_noise, noise_buffer.getvalue())
