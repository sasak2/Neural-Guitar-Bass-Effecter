"""Training loop for the flow-matching model: resumable, time-limited, EMA weights, fixed-noise validation."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import GenDataset
from .flow import fm_loss
from .model import FlowNet


def _save(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _forever(loader):
    while True:
        yield from loader


@torch.no_grad()
def validate(model: FlowNet, ds: GenDataset, dev: torch.device, batch_size: int = 16) -> float:
    """검증 손실. t와 노이즈를 고정해서 스텝끼리 비교할 수 있게 한다 (그냥 뽑으면 값이 크게 출렁인다)."""
    model.eval()
    g = torch.Generator().manual_seed(1234)
    tot, n = 0.0, 0
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
        mel, feat, mask, ref, ref_mask = (x.to(dev) for x in batch)
        b = mel.shape[0]
        t = ((torch.arange(b) + torch.rand(1, generator=g)) / b).to(dev) * 0.98 + 0.01
        noise = torch.randn(mel.shape, generator=g).to(dev)
        loss = fm_loss(model, mel, feat, mask, ref, ref_mask, p_drop=0.0, t=t, noise=noise)
        tot += float(loss) * b
        n += b
    return tot / max(n, 1)


def train(data_dir, out_dir, steps: int = 100000, batch_size: int = 32, crop: int = 384,
          ref_crop: int = 256, lr: float = 3e-4, dim: int = 384, depth: int = 8, heads: int = 6,
          p_drop: float = 0.15, ema_decay: float = 0.999, val_every: int = 2000, save_every: int = 500,
          val_items: int = 128, num_workers: int = 2, max_minutes: float | None = None,
          device: str = "auto", amp: bool | None = None, seed: int = 0, log_every: int = 100,
          resume: bool = True) -> dict[str, Any]:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = GenDataset(data_dir, "train", crop, ref_crop, items=10**9)
    if not train_ds.rows:
        raise SystemExit(f"{data_dir}/index.jsonl 에 train 항목이 없다.")
    val_ds = GenDataset(data_dir, "val", crop, ref_crop, deterministic=True, items=val_items)
    if not val_ds.rows:
        print("[warn] val 분할이 비어 있다. train으로 대신한다 (검증 손실이 낙관적이다).")
        val_ds = GenDataset(data_dir, "train", crop, ref_crop, deterministic=True, items=val_items)
    hours = sum(r["frames"] for r in train_ds.rows) * 512 / 44100 / 3600
    print(f"train files={len(train_ds.rows)} ({hours:.2f} h) groups={len(train_ds.by_group)} "
          f"| val files={len(val_ds.rows)} | device={dev}")

    loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, drop_last=True,
                        pin_memory=dev.type == "cuda", persistent_workers=num_workers > 0,
                        sampler=torch.utils.data.RandomSampler(train_ds, replacement=True,
                                                               num_samples=10**7))
    model = FlowNet(dim=dim, depth=depth, heads=heads).to(dev)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01)

    def lr_lambda(s: int) -> float:
        warm = min(1.0, (s + 1) / 1000)
        return warm * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(s / max(steps, 1), 1.0))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = dev.type == "cuda" if amp is None else (amp and dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, best_val = 0, float("inf")
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        if ck["hparams"] != model.hparams:
            raise SystemExit(f"체크포인트와 모델 설정이 다르다: {ck['hparams']} vs {model.hparams}")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        step, best_val = int(ck["step"]), float(ck["best_val"])
        print(f"resumed from step {step} (best_val={best_val:.4f})")

    def payload(best_only: bool = False) -> dict[str, Any]:
        d = {"ema": ema.state_dict(), "hparams": model.hparams, "step": step,
             "crop": crop, "ref_crop": ref_crop}
        if not best_only:
            d.update(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                     scaler=scaler.state_dict(), best_val=best_val)
        return d

    log_file = open(out_dir / "train_log.jsonl", "a", encoding="utf-8")
    data_iter = _forever(loader)
    t0, running, count, stopped_early = time.time(), 0.0, 0, False
    model.train()
    while step < steps:
        mel, feat, mask, ref, ref_mask = (x.to(dev, non_blocking=True) for x in next(data_iter))
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            loss = fm_loss(model, mel, feat, mask, ref, ref_mask, p_drop=p_drop)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        # EMA: 가중치의 이동 평균. 샘플링은 이 평균 가중치로 한다 (학습 중 흔들림이 줄어 소리가 안정적이다)
        decay = min(ema_decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1.0 - decay)
        running += loss.item()
        count += 1

        if step % log_every == 0:
            msg = {"step": step, "train_loss": running / count, "lr": sched.get_last_lr()[0],
                   "elapsed_min": (time.time() - t0) / 60}
            print(json.dumps(msg), flush=True)
            log_file.write(json.dumps(msg) + "\n")
            log_file.flush()
            running, count = 0.0, 0

        if step % val_every == 0 or step == steps:
            val = validate(ema, val_ds, dev)
            print(f"[val] step={step} loss={val:.4f} (best={min(best_val, val):.4f})", flush=True)
            log_file.write(json.dumps({"step": step, "val_loss": val}) + "\n")
            log_file.flush()
            if val < best_val:
                best_val = val
                _save(best_path, payload(best_only=True))
            model.train()

        if step % save_every == 0 or step == steps:
            _save(last_path, payload())

        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            print(f"[stop] 시간 제한({max_minutes}분). step={step}에서 저장하고 멈춘다. "
                  "같은 명령을 다시 실행하면 이어서 학습한다.")
            stopped_early = True
            break

    _save(last_path, payload())
    log_file.close()
    if not best_path.exists():
        _save(best_path, payload(best_only=True))
    metrics = {"steps": step, "finished": (not stopped_early) and step >= steps, "best_val": best_val}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[final]", json.dumps(metrics))
    return metrics
