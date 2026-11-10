"""Turn audio files into cached training items (mel + performance features) and an index.

출력 구조:
    <out>/feats/<domain>/<name>.npz   mel (frames, 128) float16 (정규화 전 log-mel), feat (frames, 4) float16
    <out>/index.jsonl                 {"file", "group", "domain", "frames", "split"}

group은 "같은 음색"의 단위다(가수 한 명, 기타 연주자 한 명). 참조 클립은 같은 group의 다른 파일에서 뽑는다.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from .features import extract_features
from .mel import HOP, SR, LogMel

MIN_FRAMES = 64


def load_audio(path: Path, peak: float = 0.95) -> np.ndarray:
    """모노 44.1 kHz float32로 읽고, 최대값이 peak가 되게 맞춘다."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    m = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if m > 1e-4:
        wav = wav * (peak / m)
    return wav[: (len(wav) // HOP) * HOP].astype(np.float32)


def group_of(path: Path, domain: str) -> str:
    """파일 이름에서 음색 묶음을 정한다. download_data.py가 `<묶음>__<원래이름>.wav`로 풀어 둔다."""
    stem = path.name.split("__")[0]
    if domain == "voice":
        m = re.fullmatch(r"(female|male)\d+", stem)
        return f"voice_{stem}" if m else f"voice_{stem or 'unknown'}"
    m = re.match(r"(P\d+)", stem)
    return f"{domain}_{m.group(1) if m else stem}"


def split_of(name: str, group: str, test_groups: set[str], val_frac: float) -> str:
    if group in test_groups:
        return "test"
    h = int(hashlib.sha1(name.encode()).hexdigest(), 16) % 10000
    return "val" if h < val_frac * 10000 else "train"


def prepare(sources: list[tuple[str, Path]], out_dir: Path, device: str = "auto",
            f0_method: str = "crepe", crepe_model: str = "full",
            test_groups: tuple[str, ...] = ("voice_female9", "voice_male11"),
            val_frac: float = 0.04, max_files: int | None = None) -> list[dict]:
    """sources: [(domain, 폴더)]. 이미 처리된 파일은 건너뛰므로 중간에 끊겨도 다시 실행하면 된다."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir)
    mel_fn = LogMel().to(device)
    rows: list[dict] = []
    for domain, folder in sources:
        files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".wav", ".flac"))
        if max_files:
            files = files[:: max(1, len(files) // max_files)][:max_files]
        print(f"[{domain}] {len(files)} files from {folder}", flush=True)
        for i, p in enumerate(files):
            rel = Path("feats") / domain / (p.stem + ".npz")
            dst = out_dir / rel
            group = group_of(p, domain)
            if dst.exists():
                with np.load(dst) as z:
                    frames = int(z["mel"].shape[0])
            else:
                wav = load_audio(p)
                if len(wav) // HOP < MIN_FRAMES:
                    continue
                with torch.no_grad():
                    mel = mel_fn(torch.from_numpy(wav)[None].to(device))[0].T.cpu().numpy()
                feat = extract_features(wav, device=device, method=f0_method, crepe_model=crepe_model)
                frames = min(len(mel), len(feat))
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(".tmp.npz")
                np.savez(tmp, mel=mel[:frames].astype(np.float16), feat=feat[:frames].astype(np.float16))
                tmp.replace(dst)
            rows.append({"file": str(rel), "group": group, "domain": domain, "frames": frames,
                         "split": split_of(p.name, group, set(test_groups), val_frac)})
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(files)}", flush=True)
    (out_dir / "index.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    hours = sum(r["frames"] for r in rows) * HOP / SR / 3600
    print(f"index: {len(rows)} files, {hours:.2f} h -> {out_dir / 'index.jsonl'}")
    return rows
