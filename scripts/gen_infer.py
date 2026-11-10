#!/usr/bin/env python3
"""Play your performance in the timbre of a reference clip.

예: python scripts/gen_infer.py --ckpt runs/gen1/best.pt --input riff.wav --ref singer.wav --output out.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxgen.infer import convert  # noqa: E402
from fxgen.mel import SR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="연주 파일 (단선율 권장)")
    ap.add_argument("--ref", type=Path, required=True, help="원하는 음색의 참조 파일 (5~10초 정도)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--transpose", type=float, default=0.0, help="반음 단위 이동 (예: 12 = 한 옥타브 위)")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0, help="참조 음색을 따르는 세기 (1 = guidance 없음)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocoder", default="bigvgan", choices=["bigvgan", "griffinlim"])
    ap.add_argument("--f0-method", default="crepe", choices=["crepe", "pyin"])
    ap.add_argument("--crepe-model", default="full", choices=["full", "tiny"])
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.device == "auto":
        import torch
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    r = convert(a.ckpt, a.input, a.ref, a.device, a.vocoder, a.transpose, a.steps, a.cfg, a.seed, a.f0_method,
                a.crepe_model)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(a.output), r["wav"], SR, subtype="PCM_16")
    print(f"wrote {a.output} ({len(r['wav']) / SR:.1f} s)")
    print("pitch (입력 연주 대비 출력):", json.dumps(r["pitch"]))


if __name__ == "__main__":
    main()
