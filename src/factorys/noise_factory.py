import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import torch

from src.models import Noise


def _create_latent_shape(
    batch_size: int = 1,
    num_channels_latents: int = 4,
    height: int = 1024,
    width: int = 1024,
    vae_scale_factor: int = 8,
) -> Tuple[int, int, int, int]:
    """

    :param batch_size:
    :param num_channels_latents:
    :param height:
    :param width:
    :param vae_scale_factor:
    :return:
    """
    return (
        batch_size,
        num_channels_latents,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )


@dataclass
class NoiseFactory:
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    seed_clamp: Tuple[int, int] = (1, 2**32 - 1)
    id_count: int = 0
    num_channel_latents: int = 4
    vae_scale_factor: int = 8
    apply_pink_noise_filter: bool = False

    def _generate_seed(self):
        return random.randint(*self.seed_clamp)

    def _apply_pink_noise_filter(self, white_noise: torch.Tensor) -> torch.Tensor:
        B, C, H, W = white_noise.shape

        fft_noise = torch.fft.fft2(white_noise.float(), norm="ortho")

        freq_h = torch.fft.fftfreq(H, device=self.device).view(-1, 1)
        freq_w = torch.fft.fftfreq(W, device=self.device).view(1, -1)

        freq = torch.sqrt(freq_h**2 + freq_w**2)
        freq[0, 0] = 1.0

        filter_1_f = 1.0 / freq
        filter_1_f[0, 0] = 0.0

        pink_fft = fft_noise * filter_1_f

        pink_noise = torch.fft.ifft2(pink_fft, norm="ortho").real

        pink_noise = pink_noise / pink_noise.std(dim=(-2, -1), keepdim=True)

        return pink_noise.to(self.dtype)

    def _generate_initial_noise(
        self,
        seed: int,
        latents_shape: Tuple[int, int, int, int],
        init_noise_sigma: float = 1.0,
    ) -> torch.Tensor:

        generator = torch.Generator(device=self.device).manual_seed(seed)

        initial_noise = torch.randn(
            latents_shape, generator=generator, device=self.device, dtype=self.dtype
        )
        if self.apply_pink_noise_filter:
            initial_noise = self._apply_pink_noise_filter(initial_noise)
        initial_noise = initial_noise * init_noise_sigma

        return initial_noise

    def _create_id(self) -> str:
        noise_id = f"n_{self.id_count:04}"
        self.id_count += 1
        return noise_id

    def create_noise_from_noise(
        self,
        noise: torch.Tensor,
    ):
        noise_id: str = self._create_id()

        return Noise(id=noise_id, initial_noise=noise)

    def create_noise(
        self,
        num_channels_latents: int = 4,
        height: int = 1024,
        width: int = 1024,
        vae_scale_factor: int = 8,
        init_noise_sigma: float = 1.0,
        seed: int | None = None,
    ) -> Noise:

        latents_shape = _create_latent_shape(
            1, num_channels_latents, height, width, vae_scale_factor
        )

        if seed is None:
            seed = self._generate_seed()

        initial_noise = self._generate_initial_noise(
            seed, latents_shape, init_noise_sigma
        )

        noise_id: str = self._create_id()
        return Noise(
            id=noise_id,
            initial_noise=initial_noise,
            initial_seed=seed,
        )

    def create_batch(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 4,
        height: int = 1024,
        width: int = 1024,
        vae_scale_factor: int = 8,
        init_noise_sigma: float = 1.0,
    ) -> list[Noise]:

        latents_shape = _create_latent_shape(
            batch_size, num_channels_latents, height, width, vae_scale_factor
        )
        seed: int = self._generate_seed()

        initial_noises: torch.Tensor = self._generate_initial_noise(
            seed, latents_shape, init_noise_sigma
        )
        noises: List[Noise] = []
        for i in range(batch_size):

            noise_id: str = self._create_id()

            initial_noise = initial_noises[i]
            initial_noise.unsqueeze_(0)
            noise = Noise(
                id=noise_id,
                initial_noise=initial_noise,
            )
            noises.append(noise)

        return noises

    def create_from_files(
        self,
        base_path: Path,
        prompt: str,
        experiment_id: int,
        generation: int | None = None,
    ) -> list[Noise]:
        path = (
            base_path
            / "results"
            / "simulations"
            / f"{prompt.replace(' ', '_')}_{experiment_id}"
        )

        initial_noise_dir = path / "initial_noise"

        blip2_dir = path / "blip2"
        output = []

        if generation is None:
            glob = initial_noise_dir.glob("*.pt")
        else:
            filter_query = f"g{generation}_*.pt"

            glob = initial_noise_dir.glob(filter_query)

        for i in glob:

            raw = str(i.name).split("_")

            noise = Noise(
                id=f"n_{raw[2]}",
                initial_noise=torch.load(i, map_location=torch.device(self.device)),
                fitness=float(raw[3].replace("f", "").replace(".pt", "")),
                start_generation=int(raw[0].replace("g", "")),
                end_generation=int(raw[0].replace("g", "")),
            )

            output.append(noise)

        output.sort(key=lambda x: x.id)
        return output
