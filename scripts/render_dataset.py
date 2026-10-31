#!/usr/bin/env python3
"""Render wet clips from a folder of dry guitar recordings.

사용 예:
    python scripts/render_dataset.py --dry-dir data/guitar_techs_di --out-dir rendered \
        --variants 4 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fxdata import render_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "effects.yaml")
    ap.add_argument("--variants", type=int, default=4, help="dry 클립당 만들 wet 변형 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-files", type=int, default=None, help="디버깅용: 앞에서 N개 파일만")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    n = render_dataset(args.dry_dir, args.out_dir, cfg, args.variants, args.seed, args.max_files)
    print(f"wrote {n} wet clips to {args.out_dir}")


if __name__ == "__main__":
    main()
