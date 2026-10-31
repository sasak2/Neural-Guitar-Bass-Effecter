"""Dataset over rendered dry/wet pairs (see fxdata.render for the layout)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .cond import cond_from_chain


def load_render_config(data_dir: Path) -> dict[str, Any]:
    p = Path(data_dir) / "render_config.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} 가 없다. scripts/render_dataset.py 로 먼저 데이터를 렌더링해야 한다."
        )
    return json.loads(p.read_text(encoding="utf-8"))


class PairDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        cfg: dict[str, Any],
        crop: int = 32768,
        deterministic: bool = False,
        max_items: int | None = None,
    ):
        self.root = Path(root)
        self.cfg = cfg
        self.sr = int(cfg["sample_rate"])
        self.clip_len = int(cfg["clip_seconds"] * self.sr)
        self.crop = min(int(crop), self.clip_len)
        self.deterministic = deterministic

        meta = self.root / split / "meta.jsonl"
        rows: list[dict[str, Any]] = []
        if meta.exists():
            rows = [json.loads(line) for line in meta.read_text(encoding="utf-8").splitlines() if line]
        if max_items is not None and len(rows) > max_items:
            step = max(1, len(rows) // max_items)
            rows = rows[::step][:max_items]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        slack = self.clip_len - self.crop
        if slack == 0:
            start = 0
        elif self.deterministic:
            start = slack // 2
        else:
            start = int(torch.randint(0, slack + 1, (1,)))
        dry, _ = sf.read(str(self.root / r["dry"]), start=start, frames=self.crop, dtype="float32")
        wet, _ = sf.read(str(self.root / r["wet"]), start=start, frames=self.crop, dtype="float32")
        cond = torch.tensor(cond_from_chain(r["chain"], self.cfg), dtype=torch.float32)
        return (
            torch.from_numpy(np.ascontiguousarray(dry))[None],
            torch.from_numpy(np.ascontiguousarray(wet))[None],
            cond,
        )
