#!/usr/bin/env python3
"""체크포인트를 학습과 별개로 전체 분할(클립 전체 길이)에서 평가한다.

예:
    python scripts/evaluate.py --data-dir rendered --ckpt runs/exp1/best.pt --split test

학습이 도는 중에도 best.pt/last.pt 로 따로 돌려 볼 수 있다. 손실이 큰 쌍도 같이 보여 준다.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fxmodel.data import PairDataset  # noqa: E402
from fxmodel.infer import load_model  # noqa: E402
from fxmodel.loss import MRSTFTLoss  # noqa: E402
from fxmodel.trainer import _device, per_item_losses  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--max-items", type=int, default=None, help="기본: 분할 전체")
    ap.add_argument("--worst", type=int, default=5, help="손실이 큰 쌍을 몇 개 보여 줄지")
    ap.add_argument("--time-weight", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = _device(args.device)
    model, cfg = load_model(args.ckpt, str(dev))
    clip_len = int(cfg["clip_seconds"] * cfg["sample_rate"])
    ds = PairDataset(args.data_dir, args.split, cfg, crop=clip_len,
                     deterministic=True, max_items=args.max_items)
    if len(ds) == 0:
        raise SystemExit(f"{args.data_dir}/{args.split}/meta.jsonl 에 쌍이 없다.")
    crit = MRSTFTLoss().to(dev)
    res = per_item_losses(model, ds, crit, dev, args.time_weight)

    m = [a for a, _ in res]
    b = [c for _, c in res]
    print(f"{args.split}: {len(res)}쌍, 클립 전체 길이")
    print(f"  모델     평균 {statistics.mean(m):.4f}  중앙값 {statistics.median(m):.4f}  최대 {max(m):.4f}")
    print(f"  identity 평균 {statistics.mean(b):.4f}  중앙값 {statistics.median(b):.4f}")
    print(f"  모델이 identity보다 나쁜 쌍: {sum(a > c for a, c in res)}개")
    print(f"\n손실이 큰 쌍 상위 {args.worst}개:")
    for i in sorted(range(len(res)), key=lambda k: -m[k])[: args.worst]:
        r = ds.rows[i]
        phys = r.get("physical", {})
        on = {n: {k: round(v, 1) for k, v in phys.get(n, {}).items()}
              for n, c in r["chain"].items() if c["enabled"]}
        print(f"  {r['id']}  모델 {m[i]:.3f}  identity {b[i]:.3f}  켜진 이펙트: {on or '없음'}")


if __name__ == "__main__":
    main()
