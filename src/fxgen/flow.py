"""Conditional flow matching (rectified flow): training loss and Euler sampler.

노이즈 x0 ~ N(0, I)와 진짜 멜 x1을 직선으로 잇는다: x_t = (1 - t) x0 + t x1.
신경망은 그 직선의 속도 v = x1 - x0 를 맞히도록 배우고, 샘플링 때는 t=0의 노이즈에서 출발해
dx/dt = v(x, t) 를 적분해 t=1의 멜에 도착한다.
"""

from __future__ import annotations

import torch

from .model import FlowNet


def fm_loss(model: FlowNet, mel: torch.Tensor, feat: torch.Tensor, mask: torch.Tensor,
            ref: torch.Tensor, ref_mask: torch.Tensor, p_drop: float = 0.15,
            t: torch.Tensor | None = None, noise: torch.Tensor | None = None) -> torch.Tensor:
    b = mel.shape[0]
    if t is None:
        t = torch.sigmoid(torch.randn(b, device=mel.device))      # 중간 시각을 더 자주 뽑는다 (logit-normal)
    if noise is None:
        noise = torch.randn_like(mel)
    tt = t[:, None, None].to(mel.dtype)
    xt = (1.0 - tt) * noise + tt * mel
    timbre = model.timbre(ref, ref_mask)
    if p_drop > 0:
        drop = torch.rand(b, device=mel.device) < p_drop
        timbre = torch.where(drop[:, None], model.null_timbre.to(timbre.dtype)[None], timbre)
    v = model(xt, t, feat, timbre, mask)
    err = ((v.float() - (mel - noise).float()) ** 2).mean(dim=-1)
    m = mask.float()
    return (err * m).sum() / m.sum().clamp_min(1.0)


@torch.no_grad()
def sample(model: FlowNet, feat: torch.Tensor, timbre: torch.Tensor, mask: torch.Tensor | None = None,
           steps: int = 32, cfg: float = 2.0, known: torch.Tensor | None = None,
           known_mask: torch.Tensor | None = None,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """feat: (B, T, 4), timbre: (B, D) -> 정규화된 멜 (B, T, 128).

    cfg: 1이면 guidance 없음. 크게 줄수록 참조 음색을 더 강하게 따른다.
    known/known_mask: 이미 정해진 프레임(예: 앞 구간과 겹치는 부분). 매 스텝마다 그 프레임을
    "정답에 노이즈를 섞은 값"으로 덮어써서 나머지가 자연스럽게 이어지도록 한다.
    """
    b, t_len, _ = feat.shape
    if mask is None:
        mask = torch.ones(b, t_len, dtype=torch.bool, device=feat.device)
    noise = torch.randn(b, t_len, model.out.out_features, device=feat.device, generator=generator)
    x = noise.clone()
    ts = torch.linspace(0.0, 1.0, steps + 1, device=feat.device)
    for i in range(steps):
        t = ts[i].expand(b)
        if known is not None:
            km = known_mask[..., None]
            x = torch.where(km, (1.0 - ts[i]) * noise + ts[i] * known, x)
        v = model(x, t, feat, timbre, mask)
        if cfg != 1.0:
            v0 = model(x, t, feat, None, mask)
            v = v0 + cfg * (v - v0)
        x = x + (ts[i + 1] - ts[i]) * v
    if known is not None:
        x = torch.where(known_mask[..., None], known, x)
    return x * mask[..., None]
