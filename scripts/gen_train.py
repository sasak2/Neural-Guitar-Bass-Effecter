#!/usr/bin/env python3
"""Train the generative (flow matching) timbre model. 같은 --out-dir 로 다시 실행하면 이어서 학습한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxgen.trainer import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True, help="gen_prepare.py 의 --out-dir")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--crop", type=int, default=384, help="학습 구간 길이(프레임). 384 = 약 4.5초")
    ap.add_argument("--ref-crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--p-drop", type=float, default=0.15, help="음색 조건을 지우는 비율 (guidance용)")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-items", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    train(**vars(ap.parse_args()))


if __name__ == "__main__":
    main()
