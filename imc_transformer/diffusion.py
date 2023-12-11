import math
from typing import Optional, Callable, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.activations import Swish
from tqdm import tqdm

from util import _initialize, mask_loss, compress, EPS


@torch.jit.script
def _gather(consts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Gather the constants for time t.
    """
    # c = torch.gather(consts, -1, t)
    # return torch.reshape(c, -1, 1, 1, 1)
    c = torch.gather(consts, 0, t)
    return torch.reshape(c, t.shape[0], 1, 1, 1)


@torch.jit.script
def _poisson_div(
    lmbda1: torch.Tensor, lmbda2: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """
    Poisson divergence between two Poisson distributions.
    """
    return (
        lmbda1 * (torch.log(lmbda1.clamp(min=eps)) - torch.log(lmbda2.clamp(min=eps)))
        + lmbda2
        - lmbda1
    )


@torch.jit.script
def _stable_division(
    x: torch.Tensor, y: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """
    Stable division.
    """
    return x / (y.clamp(min=eps))


@torch.jit.script
def _jump_loss(
    pred_x0: torch.Tensor,
    x0: torch.Tensor,
    deltas: torch.Tensor,
    t: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Calculate the JUMP loss. (poisson kl)
    """
    delta = _gather(deltas, t)
    pred_x0 = (pred_x0) * delta
    x0 = (x0) * delta
    return _poisson_div(x0, pred_x0, eps)
    # return _poisson_div(pred_x0, x0, eps)


def _linear_schedule(beta_start: float, beta_end: float, n_steps: int) -> torch.Tensor:
    """
    Linear Beta schedule.
    """
    return (
        torch.linspace(beta_start**0.5, beta_end**0.5, n_steps, dtype=torch.float32)
        ** 2
    )


def _cosine_schedule(beta_start: float, beta_end: float, n_steps: int) -> torch.Tensor:
    # Ref: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py#L45
    betas = []
    for i in range(n_steps):
        t1 = i / n_steps
        t2 = (i + 1) / n_steps
        abar1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2
        abar2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2
        betas.append(min(1 - abar2 / abar1, beta_end))
    return torch.tensor(betas, dtype=torch.float32)


class DiffusionIMC(nn.Module):
    """
    A diffusion model for IMC data.
    Heavily draws from https://nn.labml.ai/diffusion/ddpm.html for the implementation details.
    """

    def __init__(
        self,
        compressed_channels: int,
        uncompressed_channels: int,
        unet: "DiffusionUnet",
        n_steps: int,
        raw_counts: bool = True,
        sampler: str = "jump",
        compile: bool = True,
    ):
        """
        :param compressed_channels: The number of compressed channels.
        :param uncompressed_channels: The number of uncompressed channels.
        :param unet_factory: A factory function that returns a U-Net given the number of input image channels to use and output channels.
        :param n_steps: The number of diffusion steps.
        :param raw_counts: Whether to use raw counts or log counts.
        :param sampler: Either 'ddim' (Denoising Diffusion Implicit Model), 'ddpm' (Denoising Diffusion Probabilistic Model), 'jump' (Thinning and Thickening Latent Counts), or 'overdispersed_jump' sampling schemes.
            NOTE: 'jump' and 'overdispersed_jump' require raw_counts=True. Additionally, overdispersed_jump requires a doubling of the output channels in the U-Net output.
        :param loss_function: The loss function to use.
        """
        super().__init__()
        self.compressed_channels = compressed_channels
        self.uncompressed_channels = uncompressed_channels
        self.raw_counts = raw_counts
        # U-Net
        self.unet_backbone = unet  # U-Net conditioned on compressed image
        if sampler == "ddim":
            self.sampler = DDIMSampler(n_steps)
        elif sampler == "ddpm":
            self.sampler = DDPMSampler(n_steps)
        elif sampler == "jump" or sampler == "overdispersed_jump":
            self.sampler = JUMPSampler(
                n_steps,
                # beta_start=0.001,
                # beta_end=0.2,
                lmbda=25.0,
                beta_schedule="linear",
                overdispersed=sampler == "overdispersed_jump",
            )
            assert raw_counts, "JUMP sampler requires raw counts"
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        if compile:
            self.unet_backbone = torch.compile(
                self.unet_backbone,
                backend="aot_eager" if not torch.cuda.is_available() else "inductor",
            )  # MPS backend

    @property
    def device(self):
        return next(self.parameters()).device

    def sample_q(
        self,
        x0: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample the q(x_t | x_0, c) distribution. This is the "forward" process.
        :param x0: The initial image.
        :param c: The compressed image.
        :param t: The time step to sample at.
        :return: A sample from the q distribution.
        """
        return self.sampler.sample_q(self, x0, c, t, eps)

    def sample_p(self, xt: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
        """
        Sample the p(x_t-1 | x_t, c) distribution. This is the "reverse" process.
        :param xt: The image at time t.
        :param c: The compressed image.
        :param t: The time.
        :return: The sampled image from the p distribution.
        """
        return self.sampler.sample_p(self, xt, c, t)

    @torch.no_grad()
    def sample_image(
        self, c: torch.Tensor, output_intermediates: bool = False
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Sample from the p(x_t-1 | x_t, c) distribution to get a full image.
        :param c: The compressed image.
        :param output_intermediates: Whether to output the intermediate images.
        :return: The final sampled image. If output_intermediates is True, will instead return all the intermediate images.
        """
        img = self.sampler.sample_image(self, c, output_intermediates)
        if not self.raw_counts:
            if output_intermediates:
                img = [torch.expm1(x) for x in img]
            else:
                img = torch.expm1(img)
        return img

    def predict_eps(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
        """
        This uses the model to predict the noise for the time step.
        :param x: The image at time t.
        :param c: The compressed image.
        :param t: The time.
        :return: The noise prediction.
        """
        return self.sampler.predict_eps(self, x, c, t)

    def random_noise(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Sample from N(0, 1) noise.
        :param x0: The initial image.
        :param t: The time step.
        :return: The noise.
        """
        return self.sampler.random_noise(self, x0, t)

    def random_time_steps(self, x0: torch.Tensor):
        """
        Sample random time steps.
        :param x0: The initial image.
        :return: The time steps.
        """
        return self.sampler.random_time_steps(self, x0)

    def loss(
        self,
        x0: torch.Tensor,
        c: torch.Tensor,
        mask: torch.Tensor,
        codebook: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        Calculate the loss by comparing predicted noise to the ground truth noise. This is a simplified loss that does not
        require sampling from the full model.
        :param x0: The initial image minibatch.
        :param c: The compressed data image minibatch.
        :param mask: The mask for the data.
        :param codebook: The codebook used for compressing the data.
        :param noise: The ground truth noise. If None, will be sampled from N(0, 1).
        :return The loss.
        """
        if not self.raw_counts:
            x0 = torch.log1p(x0)

        return self.sampler.loss(self, x0, c, mask, codebook, noise)


class DiffusionSampler(nn.Module):
    """
    Base class for sampling from a diffusion model.
    """

    def __init__(self, n_steps: int):
        super().__init__()
        self.n_steps = n_steps

    def random_noise(self, model: DiffusionIMC, x0: torch.Tensor, t: torch.Tensor):
        """
        Sample from N(0, 1) noise.
        :param model: The model.
        :param x0: The initial image.
        :param t: The time step.
        :return: The noise.
        """
        return torch.randn_like(x0, device=x0.device)

    def random_time_steps(self, model: DiffusionIMC, x0: torch.Tensor):
        """
        Sample random time steps.
        :param model: The model.
        :param x0: The initial image.
        :return: The time steps.
        """
        return torch.randint(
            0, self.n_steps, (x0.shape[0],), device=x0.device, dtype=torch.long
        )

    def predict_eps(
        self, model: DiffusionIMC, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor
    ):
        """
        This uses the model to predict the noise for the time step.
        :param model: The model.
        :param x: The image at time t.
        :param c: The compressed image.
        :param t: The time.
        :return: The noise prediction.
        """
        return model.unet_backbone(x, c, t)

    def sample_image(
        self, model: DiffusionIMC, c: torch.Tensor, output_intermediates: bool = False
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Sample from the p(x_t-1 | x_t, c) distribution to get a full image.
        :param model: The model.
        :param c: The compressed image.
        :param output_intermediates: Whether to output the intermediate images.
        :return: The final sampled image. If output_intermediates is True, will instead return all the intermediate images.
        """
        pass

    def sample_p(
        self, model: DiffusionIMC, xt: torch.Tensor, c: torch.Tensor, t: torch.Tensor
    ):
        """
        Sample the p(x_t-1 | x_t, c) distribution. This is the "reverse" process.
        :param model: The model.
        :param xt: The image at time t.
        :param c: The compressed image.
        :param t: The time.
        :return: The sampled image from the p distribution.
        """
        pass

    def sample_q(
        self,
        model: DiffusionIMC,
        x0: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample the q(x_t | x_0, c) distribution. This is the "forward" process.
        :param model: The model.
        :param x0: The initial image.
        :param c: The compressed image.
        :param t: The time step to sample at.
        :return: A sample from the q distribution.
        """
        pass

    def loss(
        self,
        model: DiffusionIMC,
        x0: torch.Tensor,
        c: torch.Tensor,
        mask: torch.Tensor,
        codebook: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        Calculate the loss by comparing predicted noise to the ground truth noise. This is a simplified loss that does not
        require sampling from the full model.
        :param model: The model.
        :param x0: The initial image minibatch.
        :param c: The compressed data image minibatch.
        :param mask: The mask for the data.
        :param codebook: The codebook used for compressing the data.
        :param noise: The ground truth noise. If None, will be sampled from N(0, 1).
        :return The loss.
        """
        # Randomly sample a time step
        t = self.random_time_steps(model, x0)

        if noise is None:
            noise = self.random_noise(model, x0, t)

        # assert x0[~mask].sum() == 0

        xt = self.sample_q(model, x0, c, t, noise)

        predicted_noise = self.predict_eps(model, xt, c, t)

        # Calculate the loss accounting for masks
        if model.raw_counts:
            loss = _poisson_div(predicted_noise, noise)
        else:
            loss = F.mse_loss(predicted_noise, noise, reduction="none")

        loss = mask_loss(loss, mask, normalize=True, reduce=True)
        return loss


class JUMPSampler(DiffusionSampler):
    """
    Implementation of the JUMP sampler for poisson diffusion.
    Ref: https://arxiv.org/pdf/2305.18375.pdf
    Ref: https://colab.research.google.com/drive/1o6KMadCjlFSgeDEMtNrT6MtkoRdxq_Hl
    """

    # Defaults are based on recommendations from the paper (Appendix A): https://arxiv.org/pdf/2305.18375.pdf
    def __init__(
        self,
        n_steps: int,
        beta_start: float = 0.001,
        beta_end: float = 0.055,
        beta_schedule: str = "linear",
        alphas: list[float] = None,
        lmbda: float = 10.0,
        overdispersed: bool = False,
    ):
        """
        :param n_steps: The number of diffusion steps.
        :param beta_start: The starting variance. Same as in DDPM.
        :param beta_end: The ending variance. Same as in DDPM.
        :param beta_schedule: The schedule to use for time. Either "linear" or "cosine".
        :param alphas: The "thinning coefficients" for each time step, representing the noise variance. Must approach 0 as t -> inf.
        :param lmbda: The scaling parameter. This is the rate parameter for the Poisson distribution.
        :param overdispersed: Whether to use the overdispersed JUMP sampler. This requires doubling the output channels in the U-Net.
        """
        super().__init__(n_steps)
        self.overdispersed = overdispersed
        if beta_schedule == "linear":
            # Linear variance schedule
            self.register_buffer(
                "beta",
                _linear_schedule(beta_start, beta_end, n_steps).requires_grad_(False),
            )
        elif beta_schedule == "cosine":
            # Cosine variance schedule
            self.register_buffer(
                "beta",
                _cosine_schedule(beta_start, beta_end, n_steps).requires_grad_(False),
            )

        if alphas is None:
            # Ref: https://github.com/tqch/poisson-jump/blob/d08e08b1e18e795538fbaf4abb3df5d8ca68120b/poisson_jump/schedules.py#L92
            self.register_buffer(
                "alpha",
                torch.cumprod(1.0 - self.beta, dim=0).sqrt().requires_grad_(False),
            )
        else:
            self.register_buffer("alpha", torch.tensor(alphas, requires_grad=False))
        self.register_buffer(
            "alpha_prev",
            torch.cat(
                [
                    torch.tensor(
                        [
                            1.0,
                        ]
                    ),
                    self.alpha[:-1],
                ],
                dim=0,
            ).requires_grad_(False),
        )

        self.register_buffer("lmbda", torch.tensor(lmbda, requires_grad=False))
        # https://github.com/tqch/poisson-jump/blob/d08e08b1e18e795538fbaf4abb3df5d8ca68120b/poisson_jump/diffusions/jump.py#L54
        self.register_buffer(
            "deltas",
            ((self.alpha_prev - self.alpha) * self.lmbda).requires_grad_(False),
        )

    def predict_eps(
        self, model: DiffusionIMC, z: torch.Tensor, c: torch.Tensor, t: torch.Tensor
    ):
        """
        This uses the model to predict the noise for the time step.
        :param model: The model.
        :param z: The *NOISE* at time t.
        :param c: The compressed image.
        :param t: The time.
        :return: The noise prediction.
        """
        raise AssertionError("Not used")
        return torch.nn.functional.softplus(model.unet_backbone(z, c, t))

    def random_noise(self, model: DiffusionIMC, x0: torch.Tensor, t: torch.Tensor):
        """
        Sample from Poisson noise.
        :param model: The model.
        :param x0: The initial image.
        :return: The noise.
        """
        rate = self.lmbda * _gather(self.alpha, t)
        return torch.poisson(rate * x0)

    def sample_image(
        self, model: DiffusionIMC, c: torch.Tensor, output_intermediates: int = False
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        if output_intermediates:
            tracked_images = []
        z_t = torch.zeros(
            (c.shape[0], model.uncompressed_channels, *c.shape[2:]),
            device=c.device,
        )
        for t in tqdm(
            reversed(range(self.n_steps)),
            desc="Thickening image...",
            total=self.n_steps,
        ):
            # Sample from the p distribution
            z_t, pred_x0 = self.sample_p(
                model, z_t, c, torch.tensor([t], device=c.device, dtype=torch.long)
            )

            if output_intermediates:
                # z_t is the current process, pred_x0 is what the model thinks is the final result
                # TODO: Add option to output both?
                tracked_images.append((z_t / self.lmbda).detach().cpu())

        if output_intermediates:
            return tracked_images
        return (z_t / self.lmbda).detach().cpu()

    def sample_p(
        self, model: DiffusionIMC, xt: torch.Tensor, c: torch.Tensor, t: torch.Tensor
    ):
        """
        Sample the p(x_t-1 | x_t, c) distribution. This is the "reverse" process.
        :param model: The model.
        :param xt: The image at time t.
        :param c: The compressed image.
        :param t: The time.
        :return: The sampled image from the p distribution.
        """
        # Rescale for numerical stability
        scaled_xt = xt / (self.lmbda * self.alpha[t])

        predicted_x0 = torch.nn.functional.softplus(
            model.unet_backbone(
                scaled_xt,
                c,
                torch.full((xt.shape[0],), t.item(), device=xt.device),
            )
        ).clamp(min=0)
        if self.overdispersed:
            # Convert params to lambda
            predicted_x0_alpha, predicted_x0_beta = torch.chunk(predicted_x0, 2, dim=1)
            predicted_x0 = _stable_division(
                predicted_x0_alpha, predicted_x0_beta
            )  # Expected value of a Gamma distribution
        rate = _gather(self.deltas, t) * predicted_x0
        z_prev = torch.poisson(rate) + xt
        return z_prev, predicted_x0
        # Ref: https://github.com/tqch/poisson-jump/blob/d08e08b1e18e795538fbaf4abb3df5d8ca68120b/poisson_jump/diffusions/jump.py#L116
        # eps = self.predict_eps(model, xt, c, t)
        #
        # rate = _gather(self.alpha, t) * self.lmbda
        # # https://github.com/tqch/poisson-jump/blob/d08e08b1e18e795538fbaf4abb3df5d8ca68120b/poisson_jump/diffusions/jump.py#L101
        # pred_x0 = (xt.div(rate) - eps.div(rate.sqrt())).clamp(min=0)
        # prev_rate = _gather(self.deltas, t) * pred_x0
        # prev_x = torch.poisson(prev_rate) + xt
        # return prev_x

        # alpha = _gather(self.alpha, t)
        # next_x_rate = alpha * self.lmbda
        # next_x = xt.div(next_x_rate) - eps.div(next_x_rate.sqrt())
        # next_alpha = _gather(self.alpha, t - 1)
        # rate = self.lmbda * (next_alpha - alpha) * next_x
        # return xt + torch.poisson(rate.clamp(1e-6))

    def sample_q(
        self,
        model: DiffusionIMC,
        x0: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample the q(x_t | x_0, c) distribution. This is the "forward" process.
        :param model: The model.
        :param x0: The initial image.
        :param c: The compressed image.
        :param t: The time step to sample at.
        :param eps: THIS IS IGNORED.
        :return: A sample from the q distribution.
        """

        rate = self.lmbda * _gather(self.alpha, t)
        noise = torch.poisson(rate * x0)

        # predicted_eps = self.predict_eps(model, eps, c, t)
        # rate = (_gather(self.alpha, t) * self.lmbda).clamp(min=1e-12)
        # https://github.com/tqch/poisson-jump/blob/d08e08b1e18e795538fbaf4abb3df5d8ca68120b/poisson_jump/diffusions/jump.py#L101
        # return eps.div(rate) - predicted_eps.div(rate.sqrt())

        predicted_x0 = torch.nn.functional.softplus(
            model.unet_backbone(noise / rate, c, t)
        ).clamp(min=0)
        return predicted_x0

    def loss(
        self,
        model: DiffusionIMC,
        x0: torch.Tensor,
        c: torch.Tensor,
        mask: torch.Tensor,
        codebook: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        Calculate the loss by comparing predicted noise to the ground truth noise. This is a simplified loss that does not
        require sampling from the full model.
        :param model: The model.
        :param x0: The initial image minibatch.
        :param c: The compressed data image minibatch.
        :param mask: The mask for the data.
        :param codebook: The codebook used for compressing the data.
        :param noise: THIS IS IGNORED.
        :return The loss.
        """
        # Randomly sample a time step
        t = self.random_time_steps(model, x0)

        pred_x0 = self.sample_q(model, x0, c, t)

        if self.overdispersed:
            # Convert params to lambda
            pred_x0_alpha, pred_x0_beta = torch.chunk(pred_x0, 2, dim=1)
            pred_x0 = _stable_division(
                pred_x0_alpha, pred_x0_beta
            )  # Expected value of a Gamma distribution
        original_reconstruction_loss = _jump_loss(pred_x0, x0, self.deltas, t)
        # reconstruction_loss = mask_loss(
        #     original_reconstruction_loss, mask, normalize=False, reduce=True
        # )
        compressed_reconstruction_loss = _jump_loss(
            compress(pred_x0, codebook), c, self.deltas, t
        )
        compressed_mask = compress(mask, codebook) > 0
        reconstruction_loss = mask_loss(
            original_reconstruction_loss, mask, normalize=False, reduce=True
        ) * mask_loss(
            compressed_reconstruction_loss,
            compressed_mask,
            normalize=False,
            reduce=True,
        )

        return reconstruction_loss.sum()


class DDPMSampler(DiffusionSampler):
    """
    Basic DDPM sampler.
    """

    def __init__(
        self,
        n_steps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine",
    ):
        super().__init__(n_steps)
        if beta_schedule == "linear":
            # Linear variance schedule
            self.register_buffer(
                "beta",
                _linear_schedule(beta_start, beta_end, n_steps).requires_grad_(False),
            )
        elif beta_schedule == "cosine":
            # Cosine variance schedule
            self.register_buffer(
                "beta",
                _cosine_schedule(beta_start, beta_end, n_steps).requires_grad_(False),
            )

        # Noise schedule
        self.register_buffer("alpha", (1 - self.beta).requires_grad_(False))
        self.register_buffer(
            "cumprod_alpha", torch.cumprod(self.alpha, dim=0).requires_grad_(False)
        )

    def sample_q(
        self,
        model: DiffusionIMC,
        x0: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mu = _gather(self.cumprod_alpha, t) ** 0.5 * x0
        var = 1 - _gather(self.cumprod_alpha, t)

        if eps is None:
            eps = self.random_noise(model, x0, t)

        return mu + (var**0.5) * eps

    def sample_p(
        self, model: DiffusionIMC, xt: torch.Tensor, c: torch.Tensor, t: torch.Tensor
    ):
        # Predict the noise for the next time step given the compressed image.
        predicted_noise = self.predict_eps(model, xt, c, t)
        cumprod_alpha = _gather(self.cumprod_alpha, t)
        alpha = _gather(self.alpha, t)
        noise_coef = (1 / alpha) / (1 - cumprod_alpha) ** 0.5
        mu = 1 / (alpha**0.5) * (xt - noise_coef * predicted_noise)
        var = _gather(self.beta, t)
        noise = self.random_noise(model, xt, t)
        return mu + (var**0.5) * noise

    def sample_image(
        self, model: DiffusionIMC, c: torch.Tensor, output_intermediates: int = False
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        if output_intermediates:
            tracked_images = []
        # Sample from the prior
        x = torch.randn(
            (c.shape[0], model.uncompressed_channels, *c.shape[2:]),
            device=c.device,
        )
        for t in tqdm(
            reversed(range(self.n_steps)), desc="Denoising image...", total=self.n_steps
        ):
            # Sample from the p distribution
            x = self.sample_p(
                model, x, c, x.new_full((x.shape[0],), t, dtype=torch.long)
            )
            if output_intermediates:
                tracked_images.append(x.detach().cpu())

        if output_intermediates:
            return tracked_images
        return x.detach().cpu()


class DDIMSampler(DiffusionSampler):
    """
    Basic DDIM sampler. This may improve performance over DDPM, but is slower.
    """

    def __init__(
        self,
        n_steps: int,
        s_steps: int = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.2,
        schedule: str = "linear",
        eta: float = 0,
    ):
        """
        :param n_steps: Number of diffusion steps.
        :param s_steps: Number of sampling steps.
        :param beta_start: The starting variance. Same as in DDPM.
        :param beta_end: The ending variance. Same as in DDPM.
        :param schedule: The schedule to use for time. Either "linear" or "quadratic".
        :param eta: Represents the amount of noise to add to the variance. Higher values will make the process less deterministic.
        """
        super().__init__(n_steps)
        assert schedule in ("linear", "quadratic")
        self.schedule = schedule
        self.eta = eta
        self.s_steps = (n_steps // 2) if s_steps is None else s_steps

        # Using this to extract some constants
        dummy_ddpm = DDPMSampler(n_steps, beta_start, beta_end)

        if schedule == "linear":
            # Linear time step schedule
            self.register_buffer(
                "time_steps",
                (
                    torch.arange(
                        0, n_steps, self.n_steps // self.s_steps, dtype=torch.long
                    )
                    + 1
                ).requires_grad_(False),
            )
        elif schedule == "quadratic":
            # Quadratic time step schedule
            # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html#section-7
            self.register_buffer(
                "time_steps",
                (
                    (
                        (
                            torch.linspace(
                                0,
                                torch.sqrt(torch.tensor(self.n_steps * 0.8)),
                                self.s_steps,
                            )
                        )
                        ** 2
                    ).long()
                    + 1
                ).requires_grad_(False),
            )

        self.register_buffer(
            "alpha",
            dummy_ddpm.cumprod_alpha[self.time_steps - 1].clone().requires_grad_(False),
        )
        self.register_buffer("alpha_sqrt", torch.sqrt(self.alpha).requires_grad_(False))
        self.register_buffer(
            "alpha_prev",
            torch.cat(
                [
                    dummy_ddpm.cumprod_alpha[0:1],
                    dummy_ddpm.cumprod_alpha[self.time_steps[:-1]],
                ]
            ).requires_grad_(False),
        )
        self.register_buffer(
            "sigma",
            (
                self.eta
                * (
                    (1 - self.alpha_prev)
                    / (1 - self.alpha)
                    * (1 - self.alpha / self.alpha_prev)
                )
                ** 0.5
            ).requires_grad_(False),
        )

        self.register_buffer(
            "sqrt_beta", torch.sqrt(1 - self.alpha).requires_grad_(False)
        )

    def sample_q(
        self,
        model: DiffusionIMC,
        x0: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mu = _gather(self.alpha_sqrt, t) * x0
        var = _gather(self.sqrt_beta, t)

        if eps is None:
            eps = self.random_noise(model, x0, t)

        return mu + var * eps

    def sample_p(
        self,
        model: DiffusionIMC,
        xt: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        ti: int,
    ):
        eps = self.predict_eps(model, xt, c, t)

        x_prev, pred_x0 = self._get_prev_x_and_predicted_x0(
            model, eps, xt, c, torch.tensor(ti).to(t.device, non_blocking=True)
        )

        return x_prev

    def _get_prev_x_and_predicted_x0(
        self,
        model: DiffusionIMC,
        eps: torch.Tensor,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
    ):
        alpha = _gather(self.alpha, t)
        alpha_prev = _gather(self.alpha_prev, t)
        sigma = _gather(self.sigma, t)
        sqrt_beta = _gather(self.sqrt_beta, t)
        pred_x0 = (x - sqrt_beta * eps) / alpha.sqrt()
        xt_direction = (1 - alpha_prev - sigma**2).sqrt() * eps

        if sigma == 0:
            noise = 0  # No variance
        else:
            noise = self.random_noise(model, x, t)

        x_prev = (alpha_prev.sqrt() * x) + xt_direction + (sigma * noise)

        return x_prev, pred_x0

    def sample_image(
        self, model: DiffusionIMC, c: torch.Tensor, output_intermediates: bool = False
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        if output_intermediates:
            tracked_images = []

        x = torch.randn(
            (c.shape[0], model.uncompressed_channels, *c.shape[2:]), device=c.device
        )

        time_steps = torch.flip(self.time_steps, dims=(0,))

        for i, tstep in tqdm(
            enumerate(time_steps), desc="Denoising image...", total=time_steps.shape[0]
        ):
            # Sample from the p distribution
            ti = time_steps.shape[0] - i - 1  # Reverse timestep
            t = c.new_full((c.shape[0],), tstep, dtype=torch.long)

            x = self.sample_p(model, x, c, t, ti)

            if output_intermediates:
                tracked_images.append(x.detach().cpu())

        if output_intermediates:
            return tracked_images
        return x.detach().cpu()


class DiffusionUnet(nn.Module):
    """
    U-Net backbone of the diffusion model.
    """

    def __init__(
        self,
        image_channels: int,
        conditional_channels: int,
        out_image_channels: int,
        feature_channels: int = 32,
        feature_mult: tuple[int, ...] = (1, 2, 2, 4),
        use_attn: tuple[bool, ...] = (False, False, True, True),
        context_attn: bool = True,
        transformer_blocks: int = 0,
        n_blocks: int = 2,
        n_groups: int = 32,
        dropout: float = 0,
        activation: type[nn.Module] = Swish,
        n_heads: int = 1,
        normalize_condition: bool = True,
        embedding_type: Union[
            Literal["shift"], Literal["scale"], Literal["shift_and_scale"]
        ] = "shift",
    ):
        """
        :param image_channels: The input image channels.
        :param conditional_channels: The number of channels used for the conditional embedding (i.e. the compressed image).
        :param out_image_channels: The output image channels.
        :param feature_channels: The number of features to embed the image into.
        :param feature_mult: The feature multiplier at each resolution.
        :param use_attn: Whether to apply attention at each resolution.
        :param context_attn: Whether to use context attention inspired by ContextDiffusion: https://arxiv.org/pdf/2312.03584.pdf, else, use self-attention exclusively.
        :param transformer_blocks: If greater than 0, we will apply a these many transformer blocks, rather than using just attention. The transformer blocks follow the SpatialTransformer architecture of StableDiffusion.
        :param n_blocks: The number of residual blocks at each resolution.
        :param n_groups: The number of groups for group normalization.
        :param dropout: The dropout rate.
        :param activation: The nonlinearity to apply.
        :param n_heads: The number of heads for attention.
        :param normalize_condition: Whether to normalize the condition embedding.
        :param embedding_type: How to embed the time and condition channels. Either 'shift', 'scale', or 'shift_and_scale'. Shift and scale is used in Learning to Jump
        """
        super().__init__()
        n_resolutions = len(feature_mult)
        self.context_attn = context_attn
        self.initial_projection = nn.Conv2d(
            image_channels, feature_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.time_embedding = TimeEmbedding(
            feature_channels, embedding_mult=4, dropout=dropout, act=activation
        )

        self.condition_embedding = nn.Sequential(
            nn.BatchNorm2d(conditional_channels)
            if normalize_condition
            else nn.Identity(),  # Normalizes the input count data
            # https://towardsdatascience.com/replace-manual-normalization-with-batch-normalization-in-vision-ai-models-e7782e82193c
            nn.Conv2d(
                conditional_channels,
                feature_channels,
                kernel_size=(1, 1),
            ),
            nn.GroupNorm(n_groups, feature_channels),
            activation(),
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=(1, 1),
            ),
        )

        # Build the blocks
        down = []
        out_channels = in_channels = feature_channels * (
            1 if context_attn else 2
        )  # Multiply 2 since the first half of channels are from the image and the second half are from the condition embedding when context_attn is False
        for i in range(n_resolutions):
            out_channels = in_channels * feature_mult[i]
            for j in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        self.time_embedding.time_channels,
                        use_attn[i],
                        n_groups=n_groups,
                        dropout=dropout,
                        act=activation,
                        n_heads=n_heads,
                        context_attn=context_attn,
                        transformer_blocks=transformer_blocks,
                        embedding_type=embedding_type,
                    )
                )
                in_channels = out_channels
            if i < n_resolutions - 1:  # Downsample all but last resolution
                down.append(Downsample(out_channels))
        self.down = nn.ModuleList(down)
        self.middle = MiddleBlock(
            out_channels,
            self.time_embedding.time_channels,
            n_groups=n_groups,
            dropout=dropout,
            act=activation,
            n_heads=n_heads,
            context_attn=context_attn,
            transformer_blocks=transformer_blocks,
            embedding_type=embedding_type,
        )

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for j in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        self.time_embedding.time_channels,
                        use_attn[i],
                        n_groups=n_groups,
                        dropout=dropout,
                        act=activation,
                        n_heads=n_heads,
                        context_attn=context_attn,
                        transformer_blocks=transformer_blocks,
                        embedding_type=embedding_type,
                    )
                )
            out_channels = in_channels // feature_mult[i]
            # Reduce the channels
            up.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    self.time_embedding.time_channels,
                    use_attn[i],
                    n_groups=n_groups,
                    dropout=dropout,
                    act=activation,
                    n_heads=n_heads,
                    context_attn=context_attn,
                    transformer_blocks=transformer_blocks,
                    embedding_type=embedding_type,
                )
            )
            in_channels = out_channels

            # Upsample all but last resolution
            if i > 0:
                up.append(Upsample(out_channels))

        self.up = nn.ModuleList(up)
        self.norm = nn.GroupNorm(n_groups // 4, out_channels)
        self.act = activation()
        self.final_projection = nn.Conv2d(
            in_channels, out_image_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.initial_projection)
        _initialize(self.condition_embedding)
        _initialize(self.final_projection)

    def forward(self, x, c, t):
        # Get the time embedding
        t = self.time_embedding(t)
        # Get the class embedding
        c = self.condition_embedding(c)
        # Project the image
        x = self.initial_projection(x)
        if not self.context_attn:
            # Concatenate the condition embedding for future self-attention
            x = torch.cat([x, c], dim=1)
        # Store the skip connections
        h = [x]
        # Downsample half
        for block in self.down:
            x, c = block(x, t, c)
            h.append(x)
        # Middle
        x, c = self.middle(x, t, c)
        # Upsample half
        for block in self.up:
            if isinstance(block, Upsample):
                x, c = block(x, t, c)
            else:
                x, c = block(torch.cat([x, h.pop()], dim=1), t, c)
        # Final projection
        y = self.final_projection(self.act(self.norm(x)))
        return y


class TimePositionalEncoding(nn.Module):
    """
    Sinusoidal positional time encoding for the diffusion model.
    """

    def __init__(self, n_channels: int, dropout: float = 0):
        """
        :param n_channels: The number of time channels to embed into, ideally an even number.
        :param dropout: The dropout rate.
        """
        super().__init__()
        self.n_channels = n_channels
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        half_dim = self.n_channels // 2
        self.register_buffer(
            "pos_embed",
            torch.exp(torch.arange(half_dim) * -(math.log(10_000) / (half_dim - 1)))[
                None, :
            ],
        )
        self.pos_embed.requires_grad_(False)

    def forward(self, x):
        x = x[:, None] * self.pos_embed.detach()
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class TimeEmbedding(nn.Module):
    """
    Embedding layer for "time". This represents the number of diffusion steps.
    """

    def __init__(
        self,
        n_channels: int,
        embedding_mult: int = 4,
        dropout: float = 0,
        act: type[nn.Module] = Swish,
    ):
        """
        :param n_channels: The number of time channels to use.
        :param embedding_mult: The multiplier for embedding the time channels.
        :param dropout: Dropout rate for the positional encoding.
        :param act: The non-linearity to use.
        """
        super().__init__()
        self.time_channels = n_channels * embedding_mult
        self.embedding = nn.Sequential(
            TimePositionalEncoding(n_channels, dropout),
            nn.Linear(n_channels, n_channels * embedding_mult),
            act(),
            nn.Linear(n_channels * embedding_mult, n_channels * embedding_mult),
        )
        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.embedding)

    def forward(self, x):
        return self.embedding(x)


class ResidualBlock(nn.Module):
    """
    A block with two Convolutional layers and group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_channels: int,
        time_channels: int,
        embedding_style: Union[
            Literal["shift"], Literal["scale"], Literal["shift_and_scale"]
        ] = "shift",  # Shift is the default
        n_groups: int = 32,
        dropout: float = 0,
        act: type[nn.Module] = Swish,
        self_attn: bool = False,
        n_heads: int = 1,
    ):
        """
        :param in_channels: The input channels.
        :param out_channels: The output channels.
        :param time_channels: The time channels to use.
        :param conditional_channels: The number of channels for the conditional embedding.
        :param embedding_style: The style of embedding to use for the conditional embedding.
        :param n_groups: The number of groups for group normalization.
        :param dropout: The dropout rate.
        :param act: The non-linearity to use.
        :param self_attn: Should this block include a self-attention step?
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_channels = condition_channels
        self.has_shift = embedding_style in ("shift", "shift_and_scale")
        self.has_scale = embedding_style in ("scale", "shift_and_scale")
        assert self.has_shift or self.has_scale

        self.conv1 = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            act(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.condition_projection = nn.Sequential(
            nn.GroupNorm(n_groups, condition_channels),
            act(),
            nn.Conv2d(
                condition_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
            ),
        )

        self.time_decoder = nn.Sequential(
            act(),
            nn.Linear(
                time_channels,
                out_channels * (2 if self.has_scale and self.has_shift else 1),
            ),
        )

        self.scale_norm = nn.GroupNorm(n_groups, out_channels)

        self.conv2 = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            act(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
        )

        if self_attn:
            self.attn = SelfAttentionBlock(out_channels, n_heads, dropout=dropout)
        else:
            self.attn = None

        if in_channels != out_channels:
            # Projection for the skip connection
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.projection = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.conv1)
        _initialize(self.conv2, scale=0.0)
        _initialize(self.projection)
        _initialize(self.time_decoder)
        _initialize(self.condition_projection)

    def forward(self, x, time, c):
        z = self.conv1(x)
        c_proj = self.condition_projection(c)

        # Condition on time
        t_embed = self.time_decoder(time)[:, :, None, None]
        # https://github.com/tqch/poisson-jump/blob/d08e08b1e18e795538fbaf4abb3df5d8ca68120b/poisson_jump/nets/unet.py#L104
        if self.has_scale and self.has_shift:
            t_shift, t_scale = torch.chunk(t_embed, 2, dim=1)
        elif self.has_shift:
            t_shift = t_embed
            t_scale = 1
        else:
            t_shift = 0
            t_scale = t_embed

        if self.has_scale:
            t_scale = self.scale_norm(t_scale) / 2

        z = z * t_scale + t_shift

        z = self.conv2(z)

        if self.attn is not None:
            z, c_proj = self.attn(z, time, c_proj)

        return z + self.projection(x), c_proj


class AttentionModule(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        head_dim: int = None,
        dropout: float = 0,
        pre_norm: bool = True,  # https://arxiv.org/pdf/2002.04745.pdf
        compute_q: bool = True,
        compute_k: bool = True,
        compute_v: bool = True,
    ):
        """
        :param n_channels: Number of channels.
        :param n_heads: The number of heads.
        :param head_dim: The dimension of each head. By default, uses the number of channels.
        :param dropout: The dropout rate.
        :param pre_norm: Whether to apply normalization before the attention block.
        :param compute_q: Whether to compute Q if it is not provided.
        :param compute_k: Whether to compute K if it is not provided.
        :param compute_v: Whether to compute V if it is not provided.

        Note: For self-attention, compute_q, compute_k, and compute_v should all be True.
        """
        super().__init__()
        if head_dim is None:
            head_dim = n_channels
        self.pre_norm = pre_norm
        if self.pre_norm:
            self.norm1 = nn.LayerNorm(n_channels)
            self.norm2 = nn.LayerNorm([n_heads, head_dim])
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if compute_q:
            self.projection_q = nn.Linear(n_channels, n_heads * head_dim)
        else:
            self.projection_q = nn.Identity()
        if compute_k:
            self.projection_k = nn.Linear(n_channels, n_heads * head_dim)
        else:
            self.projection_k = nn.Identity()
        if compute_v:
            self.projection_v = nn.Linear(n_channels, n_heads * head_dim)
        else:
            self.projection_v = nn.Identity()

        self.out = nn.Linear(n_heads * head_dim, n_channels)
        self.scale = head_dim**-0.5  # Scale factor for QK^T
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.projection_q)
        _initialize(self.projection_k)
        _initialize(self.projection_v)
        _initialize(self.out, scale=0.0)

    def forward(
        self,
        x: torch.Tensor,  # (B, C, H, W)
        q: torch.Tensor = None,  # (B, HW, HEADS, HEAD_DIM)
        k: torch.Tensor = None,  # (B, HW, HEADS, HEAD_DIM)
        v: torch.Tensor = None,  # (B, HW, HEADS, HEAD_DIM)
        channels_first: bool = True,
    ) -> torch.Tensor:
        x_resid = x
        # Move channels last and get the original shape
        if channels_first:
            batch_size, n_channels, height, width = x.shape
            x = x.permute(0, 2, 3, 1)
        else:
            batch_size, height, width, n_channels = x.shape
        # Collapse HW
        x = x.view(batch_size, -1, n_channels)  # (B, HW, C)
        # Apply normalization
        x = self.norm1(x)

        if q is None:
            q = self.projection_q(x)
        if k is None:
            k = self.projection_k(x)
        if v is None:
            v = self.projection_v(x)

        # Reshape
        q = q.view(batch_size, -1, self.n_heads, self.head_dim)  # (B, HW, H, C)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim)  # (B, HW, H, C)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim)  # (B, HW, H, C)

        # Apply attention
        res = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, dropout_p=self.dropout if self.training else 0
        )
        # Apply normalization
        res = self.norm2(res)
        # Collapse the heads
        res = res.view(batch_size, -1, self.n_heads * self.head_dim)
        # Project back to the original dimension
        res = self.out(res)
        # Reshape to original shape
        res = res.view(batch_size, height, width, n_channels)
        if channels_first:
            res = res.permute(0, 3, 1, 2)
        # Apply residual connection
        res += x_resid
        return res


class SelfAttentionModule(nn.Module):
    """
    Multi-headed self-attention block, may use flash attention.
    """

    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        head_dim: int = None,
        dropout: float = 0,
    ):
        """
        :param n_channels: Number of channels.
        :param n_heads: The number of heads.
        :param head_dim: The dimension of each head. By default, uses the number of channels.
        :param dropout: The dropout rate.
        """
        super().__init__()
        if head_dim is None:
            head_dim = n_channels

        self.attn = AttentionModule(
            n_channels,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            pre_norm=True,
            # Compute Self-Attention
            compute_q=True,
            compute_k=True,
            compute_v=True,
        )

    def forward(self, x: torch.Tensor, channels_first: bool = True):
        return self.attn(x, channels_first=channels_first)


class CrossAttentionModule(nn.Module):
    """
    Multi-headed cross-attention block, may use flash attention.
    """

    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        head_dim: int = None,
        dropout: float = 0,
        compute_q: bool = False,
    ):
        """
        :param n_channels: Number of channels.
        :param n_heads: The number of heads.
        :param head_dim: The dimension of each head. By default, uses the number of channels.
        :param dropout: The dropout rate.
        :param compute_q: Whether to compute Q if it is not provided.
        """
        super().__init__()
        if head_dim is None:
            head_dim = n_channels

        self.attn = AttentionModule(
            n_channels,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            pre_norm=True,
            # Compute Cross-Attention by providing Q, K, and V
            compute_q=compute_q,
            compute_k=False,
            compute_v=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        channels_first: bool = True,
    ):
        return self.attn(x, q, k, v, channels_first=channels_first)


class SelfAttentionBlock(nn.Module):
    """
    Basic Self-attention used in U-Nets.
    """

    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        head_dim: int = None,
        dropout: float = 0,
    ):
        """
        :param n_channels: Number of channels.
        :param n_heads: The number of heads.
        :param head_dim: The dimension of each head. By default, uses the number of channels.
        :param dropout: The dropout rate.
        :param act: The non-linearity to use.
        """
        super().__init__()
        self.attn = SelfAttentionModule(
            n_channels,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        # Self-attention ignores the time embedding and the conditional embedding
        return self.attn(x, channels_first=True), c


class ContextAttentionBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_context_channels: int,
        n_heads: int = 1,
        head_dim: int = None,
        dropout: float = 0,
        activation: type[nn.Module] = Swish,
        n_groups: int = 32,
    ):
        """
        Inspired by the ContextDiffusion paper: https://arxiv.org/pdf/2312.03584.pdf
        The idea, apply self-attention to the image, then cross attention to the image with context embeddings from the
            compressed image.
        :param n_channels: Number of channels.
        :param n_context_channels: The number of channels for the context embedding.
        :param n_heads: The number of heads.
        :param head_dim: The dimension of each head. By default, uses the number of channels.
        :param dropout: The dropout rate.
        :param activation: The non-linearity to use for context projections.
        :param n_groups: The number of groups for group normalization.
        """
        super().__init__()
        self.n_channels = n_channels

        if head_dim is None:
            head_dim = n_channels

        self.self_attn = SelfAttentionModule(
            n_channels,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.cross_attn = CrossAttentionModule(
            n_channels,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            compute_q=True,
        )

        # Projection for the context image to match the number of channels
        self.context_embedding = nn.Sequential(
            nn.Conv2d(
                n_context_channels,
                n_channels,
                kernel_size=(1, 1),
            ),
            nn.GroupNorm(n_groups, n_channels),
            activation(),
            nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=(1, 1),
            ),
        )

        self.context_proj = nn.Linear(n_channels, n_heads * head_dim)

        # One final projection
        self.out = nn.Sequential(
            nn.LayerNorm(n_channels),
            nn.Linear(n_channels, n_channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.context_embedding)
        _initialize(self.context_proj)
        _initialize(self.out)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        # TODO: Should we embed time?
        # Channel last
        x = x.permute(0, 2, 3, 1)

        # In the paper, The Q in the cross attention is the output of the self-attention
        x = self.self_attn(x, channels_first=False)
        c_embed = self.context_embedding(c)
        # Channel Last, then collapse HW
        c_embed = c_embed.permute(0, 2, 3, 1).view(c.shape[0], -1, self.n_channels)
        # Project to the k,v dimensions required for multi-head attention
        c_embed = self.context_proj(c_embed)
        x = self.cross_attn(x, q=None, k=c_embed, v=c_embed, channels_first=False)
        x = self.out(x)
        # Channel first
        x = x.permute(0, 3, 1, 2)
        # Reshape context embedding
        c_embed = c_embed.permute(0, 2, 1).view(
            c.shape[0], self.n_channels, *c.shape[2:]
        )
        return x, c_embed  #  TODO: Return original or projected context embedding?


# Gated GeLU https://paperswithcode.com/method/geglu
class GeGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int = None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.proj)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class BaseTransformerModule(nn.Module):
    """
    Basic Building block for the Transformer.
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int = None,
        n_heads: int = 1,
        head_dim: int = None,
        context_attn: bool = True,
        dropout: float = 0,
        activation: type[nn.Module] = Swish,
        n_groups: int = 32,
    ):
        super().__init__()
        if cond_channels is None:
            cond_channels = in_channels
        if head_dim is None:
            head_dim = in_channels

        self.context_attn = context_attn
        if context_attn:
            self.block = ContextAttentionBlock(
                in_channels,
                cond_channels,
                n_heads=n_heads,
                head_dim=head_dim,
                dropout=dropout,
                activation=activation,
                n_groups=n_groups,
            )
        else:
            self.attn1 = SelfAttentionModule(
                in_channels,
                n_heads=n_heads,
                head_dim=head_dim,
                dropout=dropout,
            )

            self.attn2 = CrossAttentionModule(
                in_channels,
                n_heads=n_heads,
                head_dim=head_dim,
                dropout=dropout,
                compute_q=True,
            )

        self.out = nn.Sequential(
            nn.LayerNorm(in_channels),
            GeGELU(in_channels, in_channels * 4),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4, in_channels),
        )
        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.out)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        if self.context_attn:
            x, c = self.block(x, t, c)
        else:
            x = self.attn1(x, channels_first=True)
            x = self.attn2(x, None, c, c, channels_first=True)
        # Channel first -> last
        x = x.permute(0, 2, 3, 1)
        x = x + self.out(x)
        # Channel last -> first
        x = x.permute(0, 3, 1, 2)
        return x, c


class TransformerBlock(nn.Module):
    """
    Instead of just using attention, we can use an entire transformer block (just like Stable Diffusion).
    Based on the SpatialTransformer architecture from Stable Diffusion. https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int = None,
        n_heads: int = 1,
        n_blocks: int = 1,
        head_dim: int = None,
        context_attn: bool = True,
        dropout: float = 0,
        activation: type[nn.Module] = Swish,
        n_groups: int = 32,
    ):
        super().__init__()
        if cond_channels is None:
            cond_channels = in_channels
        if head_dim is None:
            head_dim = in_channels

        self.norm = nn.GroupNorm(n_groups, in_channels)

        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        self.transformer = nn.ModuleList(
            [
                BaseTransformerModule(
                    in_channels,
                    cond_channels=cond_channels,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    context_attn=context_attn,
                    dropout=dropout,
                    activation=activation,
                    n_groups=n_groups,
                )
                for _ in range(n_blocks)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.proj_in)
        _initialize(self.proj_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        x_init = x
        x = self.proj_in(self.norm(x))
        for block in self.transformer:
            x, c = block(x, t, c)
        x = self.proj_out(x)
        return x + x_init, c


class DownBlock(nn.Module):
    """
    Building block for the downsampling half of the U-Net.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        use_attention: bool,
        n_groups: int = 32,
        dropout: float = 0,
        act: type[nn.Module] = Swish,
        n_heads: int = 1,
        context_attn=False,
        transformer_blocks=0,
        embedding_type: Union[
            Literal["shift"], Literal["scale"], Literal["shift_and_scale"]
        ] = "shift",
    ):
        """
        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param time_channels: The number of time embedding channels.
        :param use_attention: Whether to apply attention.
        :param n_groups: The number of groups for group normalization.
        :param dropout: The dropout rate.
        :param act: The non-linearity to use.
        :param n_heads: The number of heads for attention.
        :param context_attn: Whether to use context attention.
        :param transformer_blocks: Use this many transformer blocks instead of a single attention.
        :param embedding_type: How to embed the time and condition channels. Either 'shift', 'scale', or 'shift_and_scale'.
        """
        super().__init__()
        self.residual = ResidualBlock(
            in_channels,
            out_channels,
            in_channels,
            time_channels,
            n_groups=n_groups,
            dropout=dropout,
            act=act,
            embedding_style=embedding_type,
            self_attn=use_attention,
            n_heads=n_heads,
        )
        if use_attention:
            if transformer_blocks > 0:
                self.attn = TransformerBlock(
                    out_channels,
                    cond_channels=out_channels,
                    n_heads=n_heads,
                    n_blocks=transformer_blocks,
                    context_attn=context_attn,
                    dropout=dropout,
                    activation=act,
                    n_groups=n_groups,
                )
            else:
                if not context_attn:
                    self.attn = SelfAttentionBlock(
                        out_channels,
                        dropout=dropout,
                        n_heads=n_heads,
                    )
                else:
                    self.attn = ContextAttentionBlock(
                        out_channels,
                        in_channels,
                        dropout=dropout,
                        n_heads=n_heads,
                        activation=act,
                        n_groups=n_groups,
                    )
        else:
            self.attn = None

    def forward(self, x, time, c):
        x, c = self.residual(x, time, c)
        if self.attn:
            x, c = self.attn(x, time, c)
        return x, c


class UpBlock(nn.Module):
    """
    Building block for the upsampling half of the U-Net.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        use_attention: bool,
        n_groups: int = 32,
        dropout: float = 0,
        act: type[nn.Module] = Swish,
        n_heads: int = 1,
        context_attn=False,
        transformer_blocks=0,
        embedding_type: Union[
            Literal["shift"], Literal["scale"], Literal["shift_and_scale"]
        ] = "shift",
    ):
        """
        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param time_channels: The number of time embedding channels.
        :param condition_channels: The number of channels for the conditional embedding.
        :param use_attention: Whether to apply attention.
        :param n_groups: The number of groups for group normalization.
        :param dropout: The dropout rate.
        :param act: The non-linearity to use.
        :param n_heads: The number of heads for attention.
        :param context_attn: Whether to use context attention.
        :param transformer_blocks: Use this many transformer blocks instead of a single attention.
        :param embedding_type: How to embed the time and condition channels. Either 'shift', 'scale', or 'shift_and_scale'.
        """
        super().__init__()
        self.residual = ResidualBlock(
            in_channels + out_channels,
            out_channels,
            in_channels,
            time_channels,
            n_groups=n_groups,
            dropout=dropout,
            act=act,
            embedding_style=embedding_type,
            self_attn=use_attention,
            n_heads=n_heads,
        )
        if use_attention:
            if transformer_blocks > 0:
                self.attn = TransformerBlock(
                    out_channels,
                    cond_channels=out_channels,
                    n_heads=n_heads,
                    n_blocks=transformer_blocks,
                    context_attn=context_attn,
                    dropout=dropout,
                    activation=act,
                    n_groups=n_groups,
                )
            else:
                if not context_attn:
                    self.attn = SelfAttentionBlock(
                        out_channels,
                        dropout=dropout,
                        n_heads=n_heads,
                    )
                else:
                    self.attn = ContextAttentionBlock(
                        out_channels,
                        in_channels,
                        dropout=dropout,
                        n_heads=n_heads,
                        activation=act,
                        n_groups=n_groups,
                    )
        else:
            self.attn = None

    def forward(self, x, time, c):
        x, c = self.residual(x, time, c)
        if self.attn:
            x, c = self.attn(x, time, c)
        return x, c


class MiddleBlock(nn.Module):
    """
    Final additional processing at the lowest resolution of the U-net.
    """

    def __init__(
        self,
        n_channels: int,
        time_channels: int,
        n_groups: int = 32,
        dropout: float = 0,
        act: type[nn.Module] = Swish,
        n_heads: int = 1,
        context_attn=False,
        transformer_blocks=0,
        embedding_type: Union[
            Literal["shift"], Literal["scale"], Literal["shift_and_scale"]
        ] = "shift",
    ):
        """
        :param n_channels: The number of input channels.
        :param time_channels: The number of time embedding channels.
        :param n_groups: The number of groups for group normalization.
        :param dropout: The dropout rate.
        :param act: The non-linearity to use.
        :param context_attn: Whether to use context attention.
        :param transformer_blocks: Use this many transformer blocks instead of a single attention.
        :param embedding_type: How to embed the time and condition channels. Either 'shift', 'scale', or 'shift_and_scale'.
        """
        super().__init__()
        self.residual1 = ResidualBlock(
            n_channels,
            n_channels,
            n_channels,
            time_channels,
            n_groups=n_groups,
            dropout=dropout,
            act=act,
            embedding_style=embedding_type,
            self_attn=True,
            n_heads=n_heads,
        )
        if transformer_blocks > 0:
            self.attn = TransformerBlock(
                n_channels,
                cond_channels=n_channels,
                n_heads=n_heads,
                n_blocks=transformer_blocks,
                context_attn=context_attn,
                dropout=dropout,
                activation=act,
                n_groups=n_groups,
            )
        else:
            if not context_attn:
                self.attn = SelfAttentionBlock(
                    n_channels,
                    dropout=dropout,
                    n_heads=n_heads,
                )
            else:
                self.attn = ContextAttentionBlock(
                    n_channels,
                    n_channels,
                    dropout=dropout,
                    n_heads=n_heads,
                    activation=act,
                    n_groups=n_groups,
                )
        self.residual2 = ResidualBlock(
            n_channels,
            n_channels,
            n_channels,
            time_channels,
            n_groups=n_groups,
            dropout=dropout,
            act=act,
            embedding_style=embedding_type,
            self_attn=True,
            n_heads=n_heads,
        )

    def forward(self, x, t, c):
        x, c = self.residual1(x, t, c)
        x, c = self.attn(x, t, c)
        x, c = self.residual2(x, t, c)
        return x, c


class Upsample(nn.Module):
    """
    2x upsampling block.
    """

    def __init__(self, n_channels: int):
        """
        :param n_channels: The number of input channels.
        """
        super().__init__()
        self.deconv_img = nn.ConvTranspose2d(
            n_channels, n_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
        )
        self.deconv_context = nn.ConvTranspose2d(
            n_channels, n_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.deconv_img)
        _initialize(self.deconv_context)

    def forward(self, x, t, c):
        return self.deconv_img(x), self.deconv_context(c)


class Downsample(nn.Module):
    """
    2x downsampling block.
    """

    def __init__(self, n_channels: int):
        """
        :param n_channels: The number of input channels.
        """
        super().__init__()
        self.conv_img = nn.Conv2d(
            n_channels, n_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.conv_context = nn.Conv2d(
            n_channels, n_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.conv_img)
        _initialize(self.conv_context)

    def forward(self, x, t, c):
        return self.conv_img(x), self.conv_context(c)
