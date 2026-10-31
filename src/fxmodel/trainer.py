"""Training loop: resumable, time-limited, with validation, audio samples and a final test report."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from .cond import cond_dim
from .data import PairDataset, load_render_config
from .loss import MRSTFTLoss, total_loss
from .model import FiLMTCN


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _pick(data_dir, cfg, split, crop, max_items) -> PairDataset:
    ds = PairDataset(data_dir, split, cfg, crop=crop, deterministic=True, max_items=max_items)
    if len(ds) == 0 and split != "train":
        print(f"[warn] '{split}' 분할이 비어 있다. train의 일부로 대신한다 (평가가 낙관적일 수 있음).")
        ds = PairDataset(data_dir, "train", cfg, crop=crop, deterministic=True, max_items=max_items)
    return ds


def _save(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # 저장 도중 세션이 끊겨도 기존 체크포인트가 깨지지 않게


def _forever(loader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model, ds, crit, dev, time_weight, identity=False, batch_size=8) -> float:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    tot, n = 0.0, 0
    for dry, wet, cond in loader:
        dry, wet, cond = dry.to(dev), wet.to(dev), cond.to(dev)
        pred = dry if identity else model(dry, cond)
        tot += float(total_loss(crit, pred, wet, time_weight)) * dry.size(0)
        n += dry.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def write_samples(model, ds, dev, out_dir: Path, step: int, sr: int) -> None:
    d = out_dir / "samples" / f"step{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    model.eval()
    for i in range(len(ds)):
        dry, wet, cond = ds[i]
        pred = model(dry[None].to(dev), cond[None].to(dev))[0, 0].cpu().numpy()
        for tag, arr in (("dry", dry[0].numpy()), ("target", wet[0].numpy()), ("pred", pred)):
            sf.write(str(d / f"{i}_{tag}.wav"), arr, sr, subtype="FLOAT")


def train(
    data_dir,
    out_dir,
    steps: int = 20000,
    batch_size: int = 16,
    crop: int = 32768,
    lr: float = 1e-3,
    channels: int = 32,
    layers: int = 11,
    stacks: int = 2,
    val_every: int = 1000,
    save_every: int = 500,
    val_items: int = 128,
    num_workers: int = 2,
    max_minutes: float | None = None,
    device: str = "auto",
    amp: bool | None = None,
    seed: int = 0,
    log_every: int = 100,
    time_weight: float = 0.5,
    resume: bool = True,
) -> dict[str, Any]:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_render_config(data_dir)
    dev = _device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    sr = int(cfg["sample_rate"])

    train_ds = PairDataset(data_dir, "train", cfg, crop=crop)
    if len(train_ds) == 0:
        raise SystemExit(f"{data_dir}/train/meta.jsonl 에 학습 데이터가 없다.")
    val_ds = _pick(data_dir, cfg, "val", crop, val_items)
    test_ds = _pick(data_dir, cfg, "test", crop, val_items)
    sample_ds = _pick(data_dir, cfg, "val", int(cfg["clip_seconds"] * sr), 3)
    print(f"pairs: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} | device={dev}")

    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=len(train_ds) >= batch_size, pin_memory=dev.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    hparams = dict(cond_dim=cond_dim(cfg), channels=channels, layers=layers, stacks=stacks)
    model = FiLMTCN(**hparams).to(dev)
    print(f"params={sum(p.numel() for p in model.parameters()):,} "
          f"receptive_field={model.receptive_field()} samples ({1000 * model.receptive_field() / sr:.0f} ms)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    def lr_lambda(s: int) -> float:
        warm = min(1.0, (s + 1) / 200)
        cos = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(s / max(steps, 1), 1.0)))
        return warm * cos

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = dev.type == "cuda" if amp is None else (amp and dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = MRSTFTLoss().to(dev)

    step, best_val = 0, float("inf")
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        if ck["hparams"] != model.hparams:
            raise SystemExit(f"체크포인트와 모델 설정이 다르다: {ck['hparams']} vs {model.hparams}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        step, best_val = int(ck["step"]), float(ck["best_val"])
        print(f"resumed from step {step} (best_val={best_val:.4f})")
    else:
        print(f"identity baseline (dry를 그대로 출력) val loss = "
              f"{evaluate(model, val_ds, crit, dev, time_weight, identity=True):.4f}")

    def payload(best_only: bool = False) -> dict[str, Any]:
        d = {"model": model.state_dict(), "hparams": model.hparams, "cfg": cfg, "step": step}
        if not best_only:
            d.update(opt=opt.state_dict(), sched=sched.state_dict(),
                     scaler=scaler.state_dict(), best_val=best_val)
        return d

    log_file = open(out_dir / "train_log.jsonl", "a", encoding="utf-8")
    data_iter = _forever(loader)
    t0, running, count = time.time(), 0.0, 0
    stopped_early = False
    model.train()
    while step < steps:
        dry, wet, cond = (t.to(dev, non_blocking=True) for t in next(data_iter))
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            pred = model(dry, cond)
        loss = total_loss(crit, pred.float(), wet, time_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
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
            val = evaluate(model, val_ds, crit, dev, time_weight)
            print(f"[val] step={step} loss={val:.4f} (best={min(best_val, val):.4f})", flush=True)
            log_file.write(json.dumps({"step": step, "val_loss": val}) + "\n")
            if val < best_val:
                best_val = val
                _save(best_path, payload(best_only=True))
            write_samples(model, sample_ds, dev, out_dir, step, sr)
            model.train()

        if step % save_every == 0 or step == steps:
            _save(last_path, payload())

        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            print(f"[stop] 시간 제한({max_minutes}분)에 도달. step={step}에서 저장하고 멈춘다. "
                  "같은 명령을 다시 실행하면 이어서 학습한다.")
            stopped_early = True
            break

    _save(last_path, payload())
    log_file.close()
    if not best_path.exists():
        _save(best_path, payload(best_only=True))

    ck = torch.load(best_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    metrics = {
        "steps": step,
        "finished": (not stopped_early) and step >= steps,
        "best_val": best_val,
        "val_identity": evaluate(model, val_ds, crit, dev, time_weight, identity=True),
        "test_model": evaluate(model, test_ds, crit, dev, time_weight),
        "test_identity": evaluate(model, test_ds, crit, dev, time_weight, identity=True),
        "receptive_field_samples": model.receptive_field(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[final]", json.dumps(metrics))
    return metrics
