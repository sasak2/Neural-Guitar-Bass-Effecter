#!/usr/bin/env python3
"""Train the knob-conditioned effect model on rendered dry/wet pairs.

같은 --out-dir 로 다시 실행하면 last.pt 에서 이어서 학습한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fxmodel.trainer import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True, help="render_dataset.py 의 --out-dir")
    ap.add_argument("--out-dir", type=Path, required=True, help="체크포인트/로그/샘플 저장 위치")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--crop", type=int, default=32768, help="학습 시 잘라 쓰는 길이(샘플)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--layers", type=int, default=11)
    ap.add_argument("--stacks", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--val-items", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=None, help="이 시간이 지나면 저장하고 멈춤")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--time-weight", type=float, default=0.5)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--final-items", type=int, default=None,
                    help="마지막 val/test 평가에 쓸 쌍 수 (기본: 전부, 클립 전체 길이)")
    args = vars(ap.parse_args())
    train(**args)


if __name__ == "__main__":
    main()
