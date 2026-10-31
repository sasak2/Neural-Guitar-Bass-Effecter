"""Cut dry recordings into clips and render randomly processed wet versions.

출력 구조:
    out_dir/{train,val,test}/dry/<clip_id>.wav
    out_dir/{train,val,test}/wet/<clip_id>_v<k>.wav
    out_dir/{train,val,test}/meta.jsonl      # 클립마다 노브 값(정규화) 기록

train/val/test 분할은 "원본 dry 파일" 단위로 해시로 정해서, 같은 연주가 여러 분할에 섞이지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .chain import build_board, physical_values, sample_knobs

AUDIO_EXTS = (".wav", ".flac", ".aif", ".aiff", ".ogg")


def _hash01(key: str) -> float:
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def split_of(key: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> str:
    """Deterministic split by source-file key."""
    u = _hash01(key)
    if u < ratios[0]:
        return "train"
    if u < ratios[0] + ratios[1]:
        return "val"
    return "test"


def load_mono(path: Path, sr: int) -> np.ndarray:
    x, file_sr = sf.read(str(path), always_2d=True, dtype="float32")
    x = x.mean(axis=1)
    if int(file_sr) != sr:
        g = gcd(int(file_sr), int(sr))
        x = resample_poly(x, sr // g, int(file_sr) // g).astype(np.float32)
    return x


def _rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)


def _peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)) + 1e-12)


def render_file(
    path: Path,
    key: str,
    out_dir: Path,
    cfg: dict[str, Any],
    variants: int,
    seed: int,
) -> int:
    """Render all clips of one dry file. Returns the number of wet files written."""
    sr = int(cfg["sample_rate"])
    clip = int(cfg["clip_seconds"] * sr)
    hop = int(cfg.get("hop_seconds", cfg["clip_seconds"]) * sr)
    min_rms = float(cfg.get("min_rms_db", -50.0))
    target_peak = 10.0 ** (float(cfg.get("dry_peak_db", -6.0)) / 20.0)

    x = load_mono(path, sr)
    split = split_of(key)
    clip_prefix = key.replace("/", "__").replace("\\", "__")
    rng = np.random.default_rng([seed, int(_hash01(key) * 2**31)])

    dry_dir = out_dir / split / "dry"
    wet_dir = out_dir / split / "wet"
    dry_dir.mkdir(parents=True, exist_ok=True)
    wet_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_dir / split / "meta.jsonl", "a", encoding="utf-8") as meta:
        for idx, start in enumerate(range(0, max(len(x) - clip, 0) + 1, hop)):
            seg = x[start : start + clip]
            if len(seg) < clip or _rms_db(seg) < min_rms:
                continue
            seg = (seg * (target_peak / _peak(seg))).astype(np.float32)
            clip_id = f"{clip_prefix}_{idx:04d}"
            sf.write(str(dry_dir / f"{clip_id}.wav"), seg, sr, subtype="FLOAT")

            for k in range(variants):
                chain = sample_knobs(rng, cfg)
                wet = build_board(cfg, chain)(seg, sr).astype(np.float32)
                scale = 1.0
                if _peak(wet) > 0.99:  # 클리핑 방지 (스케일은 메타에 기록)
                    scale = 0.99 / _peak(wet)
                    wet = wet * scale
                wet_id = f"{clip_id}_v{k}"
                sf.write(str(wet_dir / f"{wet_id}.wav"), wet, sr, subtype="FLOAT")
                meta.write(
                    json.dumps(
                        {
                            "id": wet_id,
                            "split": split,
                            "source": key,
                            "start_sample": start,
                            "sample_rate": sr,
                            "dry": f"{split}/dry/{clip_id}.wav",
                            "wet": f"{split}/wet/{wet_id}.wav",
                            "wet_scale": scale,
                            "chain": chain,
                            "physical": physical_values(cfg, chain),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
    return written


def render_dataset(
    dry_dir: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    variants: int = 4,
    seed: int = 0,
    max_files: int | None = None,
) -> int:
    dry_dir, out_dir = Path(dry_dir), Path(out_dir)
    files = sorted(p for p in dry_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if max_files is not None:
        files = files[:max_files]
    total = 0
    for p in files:
        key = p.relative_to(dry_dir).with_suffix("").as_posix()
        total += render_file(p, key, out_dir, cfg, variants, seed)
    return total
