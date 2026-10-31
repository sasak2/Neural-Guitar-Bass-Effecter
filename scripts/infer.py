#!/usr/bin/env python3
"""Apply a trained checkpoint to an audio file.

예:
    python scripts/infer.py --ckpt runs/exp1/best.pt --input riff.wav --output riff_fx.wav \
        --set drive.gain_db=0.8 --set drive.level_db=0.6 --set tone.tilt_db=0.7

`effect.knob=값(0~1)` 을 주면 그 이펙트가 켜진다. 지정하지 않은 이펙트는 꺼진 상태.
이펙트: comp(threshold_db, ratio), drive(gain_db, level_db), tone(tilt_db), reverb(room_size, wet_level)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fxmodel.infer import process_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--set", dest="settings", action="append", default=[], metavar="effect.knob=값")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    settings: dict[str, float] = {}
    for item in args.settings:
        key, _, value = item.partition("=")
        settings[key.strip()] = float(value)
    out = process_file(args.ckpt, args.input, args.output, settings, args.device)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
