"""Flow-matching network: a small transformer that predicts the velocity field over mel frames.

입력: 노이즈가 섞인 멜 x_t, 시각 t, 프레임별 연주 특징, 참조 클립에서 뽑은 음색 벡터.
출력: 속도 v (x_t를 진짜 멜 쪽으로 옮기는 방향).

- 연주 특징(음높이/세기)은 프레임마다 x_t 옆에 붙여 넣는다 -> 시간 정렬이 구조적으로 유지된다.
- 음색 벡터와 t는 모든 층의 정규화(AdaLN)를 조절한다 -> 전체에 고르게 작용하는 "톤" 성격.
- 음색 벡터를 가끔 지우고 학습해서(null 벡터) 샘플링 때 classifier-free guidance를 쓸 수 있다.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .features import N_FEAT
from .mel import N_MELS

PITCH_BANDS = 8


def pitch_encoding(feat: torch.Tensor) -> torch.Tensor:
    """(B, T, 4) -> (B, T, 2*PITCH_BANDS + 4). 음높이를 여러 주파수의 sin/cos로 펼친다.

    스칼라 하나로는 신경망이 몇 센트 차이를 구분하기 어렵다. 가장 빠른 성분은 약 0.47반음마다 한 바퀴 돈다.
    """
    p, voiced = feat[..., 0:1], feat[..., 1:2]
    k = 2.0 ** torch.arange(PITCH_BANDS, device=feat.device, dtype=feat.dtype)
    ang = 2.0 * math.pi * p * k
    enc = torch.cat([ang.sin(), ang.cos()], dim=-1) * voiced
    return torch.cat([enc, feat], dim=-1)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None].float() * 1000.0 * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class RefEncoder(nn.Module):
    """참조 멜 (B, Tr, 128) -> 음색 벡터 (B, out_dim). 시간축을 주의(attention) 가중 평균으로 없앤다."""

    def __init__(self, hidden: int = 256, out_dim: int = 192):
        super().__init__()
        self.inp = nn.Conv1d(N_MELS, hidden, 1)
        self.convs = nn.ModuleList(
            nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU())
            for _ in range(4))
        self.score = nn.Conv1d(hidden, 1, 1)
        self.out = nn.Linear(2 * hidden, out_dim)

    def forward(self, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask[:, None].to(ref.dtype)
        h = self.inp(ref.transpose(1, 2)) * m
        for conv in self.convs:
            h = (h + conv(h)) * m
        w = self.score(h).masked_fill(~mask[:, None], -1e4).softmax(dim=-1)
        mean = (h * w).sum(-1)
        std = ((h - mean[..., None]).pow(2) * w).sum(-1).clamp_min(1e-5).sqrt()
        return self.out(torch.cat([mean, std], dim=-1))


class Rotary(nn.Module):
    """회전 위치 인코딩: 프레임 사이의 상대 거리를 주의 계산에 넣는다. 학습 때보다 긴 입력에도 쓸 수 있다."""

    def __init__(self, head_dim: int):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv", inv, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.arange(x.shape[-2], device=x.device, dtype=self.inv.dtype)
        ang = t[:, None] * self.inv[None]
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class AdaNorm(nn.Module):
    def __init__(self, dim: int, cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(g)[:, None].chunk(2, dim=-1)
        return self.norm(x) * (1.0 + gamma) + beta


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, cond: int):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = AdaNorm(dim, cond), AdaNorm(dim, cond)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.rot = Rotary(dim // heads)

    def forward(self, x: torch.Tensor, g: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x, g)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(self.rot(q), self.rot(k), v, attn_mask=attn_mask)
        x = x + self.proj(h.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.n2(x, g))


class FlowNet(nn.Module):
    def __init__(self, dim: int = 384, depth: int = 8, heads: int = 6, timbre_dim: int = 192):
        super().__init__()
        self.hparams = dict(dim=dim, depth=depth, heads=heads, timbre_dim=timbre_dim)
        cond = dim
        self.ref_enc = RefEncoder(out_dim=timbre_dim)
        self.null_timbre = nn.Parameter(torch.zeros(timbre_dim))
        self.t_mlp = nn.Sequential(nn.Linear(dim, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.timbre_mlp = nn.Sequential(nn.Linear(timbre_dim, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.inp = nn.Linear(N_MELS + 2 * PITCH_BANDS + N_FEAT, dim)
        self.local = nn.Conv1d(dim, dim, 31, padding=15, groups=16)   # 가까운 프레임 문맥
        self.blocks = nn.ModuleList(Block(dim, heads, cond) for _ in range(depth))
        self.out_norm = AdaNorm(dim, cond)
        self.out = nn.Linear(dim, N_MELS)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def timbre(self, ref: torch.Tensor, ref_mask: torch.Tensor) -> torch.Tensor:
        return self.ref_enc(ref, ref_mask)

    def forward(self, x: torch.Tensor, t: torch.Tensor, feat: torch.Tensor,
                timbre: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 128), t: (B,), feat: (B, T, 4), timbre: (B, D) 또는 None(조건 없음), mask: (B, T) bool."""
        if timbre is None:
            timbre = self.null_timbre.expand(x.shape[0], -1)
        g = self.t_mlp(timestep_embedding(t, self.hparams["dim"]).to(x.dtype)) + self.timbre_mlp(timbre)
        m = mask[..., None].to(x.dtype)
        h = self.inp(torch.cat([x, pitch_encoding(feat)], dim=-1)) * m
        h = h + self.local(h.transpose(1, 2)).transpose(1, 2) * m
        attn_mask = mask[:, None, None, :]
        for blk in self.blocks:
            h = blk(h, g, attn_mask)
        return self.out(self.out_norm(h, g)) * m
