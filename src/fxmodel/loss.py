"""Multi-resolution STFT loss (spectral convergence + log-magnitude L1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MRSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(512, 1024, 2048), hop_ratio: float = 0.25, eps: float = 1e-7):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.hop_ratio = hop_ratio
        self.eps = eps

    def _mag(self, x: torch.Tensor, n: int) -> torch.Tensor:
        window = torch.hann_window(n, device=x.device, dtype=x.dtype)
        spec = torch.stft(
            x, n, hop_length=int(n * self.hop_ratio), win_length=n, window=window,
            return_complex=True,
        )
        # abs() 대신 sqrt(re^2 + im^2 + eps): 0 근처에서 기울기가 NaN이 되는 것을 막는다
        return torch.sqrt(spec.real**2 + spec.imag**2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = pred.squeeze(1).float(), target.squeeze(1).float()
        total = pred.new_zeros(())
        for n in self.fft_sizes:
            p, t = self._mag(pred, n), self._mag(target, n)
            sc = torch.linalg.norm(t - p) / (torch.linalg.norm(t) + self.eps)
            mag = F.l1_loss(torch.log(p + self.eps), torch.log(t + self.eps))
            total = total + sc + mag
        return total / len(self.fft_sizes)


def total_loss(crit: MRSTFTLoss, pred: torch.Tensor, target: torch.Tensor, time_weight: float):
    return crit(pred, target) + time_weight * F.l1_loss(pred.float(), target.float())
