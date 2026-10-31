#!/usr/bin/env python3
"""기타 DI에서 연주(음높이/세기)를 뽑아 모음 소리로 합성하고, 음높이가 보존됐는지 잰다.

예:
    python scripts/voice_demo.py --input di.wav --out-dir demo --seconds 20 --vowel a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fxdata.render import load_mono  # noqa: E402
from fxdata.voice import VOWELS, extract_performance, pitch_error_cents, render_voice  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--vowel", default="a", choices=list(VOWELS))
    args = ap.parse_args()

    x = load_mono(args.input, args.sr)
    a, b = int(args.start * args.sr), int((args.start + args.seconds) * args.sr)
    x = x[a:b]
    perf = extract_performance(x, args.sr)
    y = render_voice(perf, args.sr, len(x), vowel=args.vowel)
    perf_out = extract_performance(y, args.sr)

    print(f"유성 프레임 비율: {perf.voiced.mean():.2f}")
    print("입력 vs 합성 음높이 오차:", pitch_error_cents(perf, perf_out))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    sf.write(str(args.out_dir / f"{stem}_dry.wav"), x / (abs(x).max() + 1e-9) * 0.5, args.sr, subtype="PCM_16")
    sf.write(str(args.out_dir / f"{stem}_voice_{args.vowel}.wav"), y, args.sr, subtype="PCM_16")
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
