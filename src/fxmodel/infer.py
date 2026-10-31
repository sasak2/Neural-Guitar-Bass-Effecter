"""Apply a trained model to an audio file with chosen knob settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from fxdata.render import load_mono

from .cond import chain_from_settings, cond_from_chain
from .model import FiLMTCN


def load_model(ckpt_path, device: str = "cpu") -> tuple[FiLMTCN, dict[str, Any]]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FiLMTCN(**ck["hparams"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["cfg"]


@torch.no_grad()
def apply_model(
    model: FiLMTCN,
    cfg: dict[str, Any],
    audio: np.ndarray,
    cond: list[float],
    device: str = "cpu",
    chunk_seconds: float = 8.0,
) -> np.ndarray:
    """Run the model on a mono float32 signal at cfg['sample_rate'] (chunked with context)."""
    sr = int(cfg["sample_rate"])
    target_peak = 10.0 ** (float(cfg.get("dry_peak_db", -6.0)) / 20.0)
    peak = float(np.max(np.abs(audio)) + 1e-12)
    x = (audio * (target_peak / peak)).astype(np.float32)  # 학습 때와 같은 입력 레벨

    c = torch.tensor(cond, dtype=torch.float32, device=device)[None]
    chunk = int(chunk_seconds * sr)
    ctx = model.receptive_field() // 2 + 1
    out = np.zeros_like(x)
    for start in range(0, len(x), chunk):
        end = min(start + chunk, len(x))
        lo, hi = max(0, start - ctx), min(len(x), end + ctx)
        seg = torch.from_numpy(x[lo:hi])[None, None].to(device)
        y = model(seg, c)[0, 0].cpu().numpy()
        out[start:end] = y[start - lo : start - lo + (end - start)]
    return out


def process_file(
    ckpt_path, input_path, output_path, settings: dict[str, float], device: str = "cpu"
) -> Path:
    model, cfg = load_model(ckpt_path, device)
    chain = chain_from_settings(cfg, settings)
    audio = load_mono(Path(input_path), int(cfg["sample_rate"]))
    out = apply_model(model, cfg, audio, cond_from_chain(chain, cfg), device)
    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        print(f"[warn] 출력 피크 {peak:.2f} > 0.99, 클리핑 방지를 위해 낮춘다")
        out = out * (0.99 / peak)
    sf.write(str(output_path), out, int(cfg["sample_rate"]), subtype="FLOAT")
    return Path(output_path)
