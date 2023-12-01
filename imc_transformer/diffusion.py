import math
from typing import Optional, Callable, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.activations import Swish
from tqdm import tqdm

from imc_transformer.util import _initialize


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
    lmbda1: torch.Tensor, lmbda2: torch.Tensor, eps: float = 1e-12
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
def _jump_loss(
    pred_x0: torch.Tensor,
    x0: torch.Tensor,
    deltas: torch.Tensor,
    t: torch.Tensor,
    offset: int = 1,  # FIXME: This is a hack to avoid numerical issues
) -> torch.Tensor:
    """
    Calculate the JUMP loss. (poisson kl)
    """
    delta = _gather(deltas, t)
    pred_x0 = (pred_x0 + offset) * delta
    x0 = (x0 + offset) * delta
    return _poisson_div(x0, pred_x0)
    # return _poisson_div(pred_x0, x0)


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
    Inspired by Palette: https://arxiv.org/pdf/2111.05826.pdf as we build a denoising diffusion model where the task
    is to generate the original image from noise conditioned on the compressed image.
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
        :param sampler: Either 'ddim' (Denoising Diffusion Implicit Model), 'ddpm' (Denoising Diffusion Probabilistic Model), or 'jump' (Thinning and Thickening Latent Counts) sampling schemes.
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
        elif sampler == "jump":
            self.sampler = JUMPSampler(
                n_steps,
                # beta_start=0.001,
                # beta_end=0.2,
                lmbda=100.0,
                beta_schedule="linear",
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
        # if not self.raw_counts:
        c = torch.log1p(c + 1)  # Map the embedding to a more manageable range
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
        noise: Optional[torch.Tensor] = None,
    ):
        """
        Calculate the loss by comparing predicted noise to the ground truth noise. This is a simplified loss that does not
        require sampling from the full model.
        :param x0: The initial image minibatch.
        :param c: The compressed data image minibatch.
        :param mask: The mask for the data.
        :param noise: The ground truth noise. If None, will be sampled from N(0, 1).
        :return The loss.
        """
        c = torch.log1p(c + 1)
        if not self.raw_counts:
            x0 = torch.log1p(x0 + 1)

        return self.sampler.loss(self, x0, c, mask, noise)


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
        noise: Optional[torch.Tensor] = None,
    ):
        """
        Calculate the loss by comparing predicted noise to the ground truth noise. This is a simplified loss that does not
        require sampling from the full model.
        :param model: The model.
        :param x0: The initial image minibatch.
        :param c: The compressed data image minibatch.
        :param mask: The mask for the data.
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
        loss = (loss * mask.unsqueeze(1).float()).sum() / mask.sum()
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
    ):
        """
        :param n_steps: The number of diffusion steps.
        :param beta_start: The starting variance. Same as in DDPM.
        :param beta_end: The ending variance. Same as in DDPM.
        :param beta_schedule: The schedule to use for time. Either "linear" or "cosine".
        :param alphas: The "thinning coefficients" for each time step, representing the noise variance. Must approach 0 as t -> inf.
        :param lmbda: The scaling parameter. This is the rate parameter for the Poisson distribution.
        """
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
        pred_x0 = torch.zeros(
            (c.shape[0], model.uncompressed_channels, *c.shape[2:]),
            device=c.device,
        )
        z_t = torch.zeros_like(pred_x0)
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
                scaled_xt, c, torch.full((xt.shape[0],), t.item(), device=xt.device)
            )
        ).clamp(min=0)
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
        noise: Optional[torch.Tensor] = None,
    ):
        """
        Calculate the loss by comparing predicted noise to the ground truth noise. This is a simplified loss that does not
        require sampling from the full model.
        :param model: The model.
        :param x0: The initial image minibatch.
        :param c: The compressed data image minibatch.
        :param mask: The mask for the data.
        :param noise: THIS IS IGNORED.
        :return The loss.
        """
        # Randomly sample a time step
        t = self.random_time_steps(model, x0)

        pred_x0 = self.sample_q(model, x0, c, t)

        # Calculate the loss accounting for masks
        loss = _jump_loss(pred_x0, x0, self.deltas, t)
        indices = list(range(1, loss.ndim))
        # Loss per image
        mask = mask.unsqueeze(1).float()
        loss = (loss * mask).sum(dim=indices) / (mask.sum(dim=indices))  # Total loss
        loss = loss.sum()
        return loss


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
        n_blocks: int = 2,
        n_groups: int = 32,
        dropout: float = 0,
        activation: type[nn.Module] = Swish,
        n_heads: int = 1,
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
        :param n_blocks: The number of residual blocks at each resolution.
        :param n_groups: The number of groups for group normalization.
        :param dropout: The dropout rate.
        :param activation: The nonlinearity to apply.
        :param n_heads: The number of heads for attention.
        :param embedding_type: How to embed the time and condition channels. Either 'shift', 'scale', or 'shift_and_scale'.
        """
        super().__init__()
        n_resolutions = len(feature_mult)
        self.initial_projection = nn.Conv2d(
            image_channels, feature_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.time_embedding = TimeEmbedding(
            feature_channels, embedding_mult=4, dropout=dropout, act=activation
        )
        self.class_embedding = nn.Sequential(
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
        out_channels = in_channels = (
            feature_channels * 2
        )  # Multiply 2 since the first half of channels are from the image and the second half are from the condition embedding
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
        _initialize(self.class_embedding)
        _initialize(self.final_projection)

    def forward(self, x, c, t):
        # Get the time embedding
        t = self.time_embedding(t)
        # Get the class embedding
        c = self.class_embedding(c)
        # Project the image
        x = self.initial_projection(x)
        # Concatenate the condition embedding
        xc = torch.cat([x, c], dim=1)
        # Store the skip connections
        h = [xc]
        # Downsample half
        for block in self.down:
            xc = block(xc, t)
            h.append(xc)
        # Middle
        xc = self.middle(xc, t)
        # Upsample half
        for block in self.up:
            if isinstance(block, Upsample):
                xc = block(xc, t)
            else:
                xc = block(torch.cat([xc, h.pop()], dim=1), t)
        # Final projection
        y = self.final_projection(self.act(self.norm(xc)))
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
        time_channels: int,
        embedding_style: Union[
            Literal["shift"], Literal["scale"], Literal["shift_and_scale"]
        ] = "shift",  # Shift is the default
        n_groups: int = 32,
        dropout: float = 0,
        act: type[nn.Module] = Swish,
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
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_shift = embedding_style in ("shift", "shift_and_scale")
        self.has_scale = embedding_style in ("scale", "shift_and_scale")
        assert self.has_shift or self.has_scale

        self.conv1 = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            act(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
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

    def forward(self, x, time):
        z = self.conv1(x)

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
        return z + self.projection(x)


class AttentionBlock(nn.Module):
    """
    Multi-headed attention block, using flash attention.
    """

    def __init__(
        self,
        n_channels: int,
        n_heads: int = 1,
        head_dim: int = None,
        n_groups: int = 32,
        dropout: float = 0,
    ):
        """
        :param n_channels: Number of channels.
        :param n_heads: The number of heads.
        :param head_dim: The dimension of each head. By default, uses the number of channels.
        :param n_groups: The number of groups for group normalization.
        :param dropout: The dropout rate.
        """
        super().__init__()
        if head_dim is None:
            head_dim = n_channels

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * head_dim * 3)  # Projects QKV
        self.out = nn.Linear(n_heads * head_dim, n_channels)
        self.scale = head_dim**-0.5  # Scale factor for QK^T
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.projection)
        _initialize(self.out, scale=0.0)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ):
        # Add the conditional embedding
        xc_orig = x

        batch_size, n_channels, height, width = xc_orig.shape
        # Move the channels to the end and collapse the spatial dimensions
        xc = xc_orig.view(batch_size, n_channels, -1).permute(0, 2, 1)  # (B, HW, C)
        # Project to QKV
        qkv = self.projection(xc).view(
            batch_size, -1, self.n_heads, self.head_dim * 3
        )  # Split QKV to separate heads
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # (B, HW, H, C)
        # Apply attention
        res = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, dropout_p=self.dropout if self.training else 0
        )
        # Collapse the heads
        res = res.view(batch_size, -1, self.n_heads * self.head_dim)
        # Project back to the original dimension
        res = self.out(res)
        # Reshape to original shape
        res = res.permute(0, 2, 1).view(xc_orig.shape)
        # Apply residual connection
        res += xc_orig
        return res


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
        :param embedding_type: How to embed the time and condition channels. Either 'shift', 'scale', or 'shift_and_scale'.
        """
        super().__init__()
        self.residual = ResidualBlock(
            in_channels,
            out_channels,
            time_channels,
            n_groups=n_groups,
            dropout=dropout,
            act=act,
            embedding_style=embedding_type,
        )
        if use_attention:
            self.attn = AttentionBlock(
                out_channels,
                n_groups=n_groups,
                dropout=dropout,
                n_heads=n_heads,
            )
        else:
            self.attn = None

    def forward(self, x, time):
        x = self.residual(x, time)
        if self.attn:
            x = self.attn(x, time)
        return x


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
        :param embedding_type: How to embed the time and condition channels. Either 'shift', 'scale', or 'shift_and_scale'.
        """
        super().__init__()
        self.residual = ResidualBlock(
            in_channels + out_channels,
            out_channels,
            time_channels,
            n_groups=n_groups,
            dropout=dropout,
            act=act,
            embedding_style=embedding_type,
        )
        if use_attention:
            self.attn = AttentionBlock(
                out_channels,
                n_groups=n_groups,
                dropout=dropout,
                n_heads=n_heads,
            )
        else:
            self.attn = None

    def forward(self, x, time):
        x = self.residual(x, time)
        if self.attn:
            x = self.attn(x, time)
        return x


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
        :param embedding_type: How to embed the time and condition channels. Either 'shift', 'scale', or 'shift_and_scale'.
        """
        super().__init__()
        self.residual1 = ResidualBlock(
            n_channels,
            n_channels,
            time_channels,
            n_groups=n_groups,
            dropout=dropout,
            act=act,
            embedding_style=embedding_type,
        )
        self.attn = AttentionBlock(n_channels, n_groups=n_groups, dropout=dropout)
        self.residual2 = ResidualBlock(
            n_channels,
            n_channels,
            time_channels,
            n_groups=n_groups,
            dropout=dropout,
            act=act,
            embedding_style=embedding_type,
        )

    def forward(self, x, t):
        x = self.residual1(x, t)
        x = self.attn(x, t)
        x = self.residual2(x, t)
        return x


class Upsample(nn.Module):
    """
    2x upsampling block.
    """

    def __init__(self, n_channels: int):
        """
        :param n_channels: The number of input channels.
        """
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            n_channels, n_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.deconv)

    def forward(self, x, t):
        return self.deconv(x)


class Downsample(nn.Module):
    """
    2x downsampling block.
    """

    def __init__(self, n_channels: int):
        """
        :param n_channels: The number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            n_channels, n_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        _initialize(self.conv)

    def forward(self, x, t):
        return self.conv(x)
