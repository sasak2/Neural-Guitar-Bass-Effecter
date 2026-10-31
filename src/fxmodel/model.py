"""FiLM-conditioned dilated TCN (WaveNet style, non-causal) for dry -> wet conversion."""

from __future__ import annotations

import torch
from torch import nn


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, cond_hidden: int, kernel: int = 3):
        super().__init__()
        assert kernel % 2 == 1, "kernel must be odd for 'same' padding"
        self.conv = nn.Conv1d(
            channels, 2 * channels, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2
        )
        self.film = nn.Linear(cond_hidden, 4 * channels)  # gamma, beta for both gate halves
        self.res = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.film.weight)  # 처음에는 조건 없는 WaveNet처럼 시작
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.film(c).unsqueeze(-1).chunk(2, dim=1)
        z = self.conv(x) * (1.0 + gamma) + beta
        a, g = z.chunk(2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(g)
        return x + self.res(h), self.skip(h)


class FiLMTCN(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        channels: int = 32,
        layers: int = 11,
        stacks: int = 2,
        kernel: int = 3,
        hidden: int = 128,
    ):
        super().__init__()
        self.hparams = dict(
            cond_dim=cond_dim,
            channels=channels,
            layers=layers,
            stacks=stacks,
            kernel=kernel,
            hidden=hidden,
        )
        self.dilations = [2**i for i in range(layers)] * stacks
        self.kernel = kernel
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.inp = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList(FiLMBlock(channels, d, hidden, kernel) for d in self.dilations)
        self.out = nn.Sequential(
            nn.ReLU(), nn.Conv1d(channels, channels, 1), nn.ReLU(), nn.Conv1d(channels, 1, 1)
        )
        nn.init.zeros_(self.out[-1].weight)  # 시작 시 출력 = 입력 (항등)
        nn.init.zeros_(self.out[-1].bias)

    def receptive_field(self) -> int:
        return 1 + sum(d * (self.kernel - 1) for d in self.dilations)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, T) dry, cond: (B, cond_dim) -> (B, 1, T) wet."""
        c = self.cond_mlp(cond)
        h = self.inp(x)
        skip = torch.zeros_like(h)
        for block in self.blocks:
            h, s = block(h, c)
            skip = skip + s
        return x + self.out(skip)
