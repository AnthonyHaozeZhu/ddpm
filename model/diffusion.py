from typing import Tuple, Optional
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class DenoiseDiffusion:
    """
    ## Denoise Diffusion
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model

        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.device = device

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def q_xt_x0(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)

        return mean, var

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
    ):
        if eps is None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)

        return mean + (var**0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.eps_model(xt, t)

        alpha_bar = gather(self.alpha_bar, t)

        alpha = gather(self.alpha, t)

        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha**0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)

        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var**0.5) * eps

    def sample(self, n_samples, n_steps, image_channels, image_size):
        """
        ### Sample images
        n_samples: The number of the samples to generate
        n_steps: Time steps
        image_channels: The channels of the image
        """
        with torch.no_grad():
            x = torch.randn(
                [n_samples, image_channels, *image_size], device=self.device
            )

            range_bar = tqdm(range(n_steps), desc="Generating")

            for t_ in range_bar:
                t = n_steps - t_ - 1
                x = self.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))

            return x

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        # Get batch size
        batch_size = x0.shape[0]
        # Get random t for each sample in the batch
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )

        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)

        # MSE loss
        return F.mse_loss(noise, eps_theta)
