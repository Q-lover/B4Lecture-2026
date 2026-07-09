# -*- coding: utf-8 -*-
"""
Ex8 Diffusion Model — 解答例
diffusion_skeleton.py の #TODO を埋めた完全実装。
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from tqdm import tqdm


class DiffusionModel(nn.Module):
    """Denoising Diffusion Probabilistic Model (DDPM) — 完全実装版"""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        num_timesteps: int,
        noise_schedule: str,
        noise_schedule_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.num_timesteps = num_timesteps

        if noise_schedule == "linear":
            beta = torch.linspace(
                noise_schedule_kwargs["start"],
                noise_schedule_kwargs["end"],
                num_timesteps,
            )
            alpha = 1.0 - beta
            alpha_prod = alpha.cumprod(dim=0)
            self.register_buffer("beta", beta)
            self.register_buffer("alpha", alpha)
            self.register_buffer("alpha_prod", alpha_prod)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ノイズ推定モデルを呼び出す。"""
        return self.model(x, t).sample

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """順拡散過程: x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε  (DDPM Eq. 4)"""
        a = self.alpha_prod[t].view(-1, 1, 1, 1)
        x_t = x0 * a.sqrt() + noise * (1 - a).sqrt()
        return x_t

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """逆拡散過程: x_{t-1} を推定する  (DDPM Algorithm 2, Eq. 11)"""
        with torch.no_grad():
            noise_pred = self.forward(x, t)
            alpha_t = self.alpha[t].view(-1, 1, 1, 1)
            alpha_prod_t = self.alpha_prod[t].view(-1, 1, 1, 1)
            x = (
                x - noise_pred * (1 - alpha_t) / (1 - alpha_prod_t).sqrt()
            ) / alpha_t.sqrt()
            if t[0].item() != 0:
                x = x + torch.randn_like(x) * (1 - alpha_t).sqrt()
            return x

    def training_step(self, images: torch.Tensor) -> torch.Tensor:
        """1バッチの損失を計算する  (DDPM Algorithm 1)"""
        device = images.device
        t = torch.randint(
            0, self.num_timesteps, (images.size(0),), device=device
        ).long()
        noise = torch.randn_like(images)
        noisy_images = self.q_sample(images, t, noise)
        noise_pred = self.forward(noisy_images, t)
        criterion = self.criterion(noise_pred, noise)
        return criterion

    def generate(self, num_timesteps: int, shape: tuple) -> torch.Tensor:
        """サンプルを生成する。"""
        device = self.alpha.device
        x = torch.randn(shape, device=device)
        for step in tqdm(range(num_timesteps - 1, -1, -1)):
            t = torch.full((x.size(0),), step, dtype=torch.long, device=device)
            x = self.p_sample(x, t)
        return x
