#!/usr/bin/env python3
"""Precompute mel + performance features for the generative model.

예: python scripts/gen_prepare.py --voice data/vocalset --guitar data/guitar_techs_di --out-dir gen_data
중간에 끊겨도 같은 명령을 다시 실행하면 이미 만든 파일은 건너뛴다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fxgen.prep import prepare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--voice", type=Path, action="append", default=[], help="보컬 wav 폴더")
    ap.add_argument("--guitar", type=Path, action="append", default=[], help="기타 DI wav 폴더")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--f0-method", default="crepe", choices=["crepe", "pyin"])
    ap.add_argument("--crepe-model", default="full", choices=["full", "tiny"])
    ap.add_argument("--test-groups", nargs="*", default=["voice_female9", "voice_male11"],
                    help="학습에서 통째로 빼 둘 음색 묶음 (처음 듣는 음색에 대한 평가용)")
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--max-files", type=int, default=None, help="도메인마다 이 개수만 처리 (점검용)")
    a = ap.parse_args()
    sources = [("voice", p) for p in a.voice] + [("guitar", p) for p in a.guitar]
    if not sources:
        raise SystemExit("--voice 또는 --guitar 폴더를 하나 이상 줘야 한다.")
    prepare(sources, a.out_dir, a.device, a.f0_method, a.crepe_model, tuple(a.test_groups),
            a.val_frac, a.max_files)


if __name__ == "__main__":
    main()
