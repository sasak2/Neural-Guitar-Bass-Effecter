"""Dataset: (target mel crop, its performance features, a reference mel crop from the same group)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .mel import MEL_MEAN, MEL_STD, N_MELS
from .features import N_FEAT


def load_index(root: Path, split: str) -> list[dict]:
    p = Path(root) / "index.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} 가 없다. scripts/gen_prepare.py 를 먼저 실행해야 한다.")
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r["split"] == split]


class GenDataset(Dataset):
    """deterministic=True면 항목마다 항상 같은 구간/같은 참조를 낸다 (검증용)."""

    def __init__(self, root: Path, split: str, crop: int = 384, ref_crop: int = 256,
                 deterministic: bool = False, items: int | None = None):
        self.root = Path(root)
        self.rows = load_index(root, split)
        self.crop, self.ref_crop = crop, ref_crop
        self.deterministic = deterministic
        self.by_group: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(self.rows):
            self.by_group[r["group"]].append(i)
        frames = np.array([r["frames"] for r in self.rows], dtype=np.float64)
        self.prob = frames / frames.sum() if len(frames) else frames
        # 한 "에폭"의 길이: 겹치지 않게 자르면 나오는 구간 수
        self.n = items if items is not None else int(max(1, frames.sum() // crop)) if len(frames) else 0

    def __len__(self) -> int:
        return self.n

    def _crop(self, arr: np.ndarray, length: int, rng: np.random.Generator):
        n = arr.shape[0]
        start = int(rng.integers(0, n - length + 1)) if n > length else 0
        seg = arr[start : start + length].astype(np.float32)
        mask = np.zeros(length, dtype=bool)
        mask[: len(seg)] = True
        if len(seg) < length:
            seg = np.concatenate([seg, np.zeros((length - len(seg), arr.shape[1]), np.float32)])
        return seg, mask

    def __getitem__(self, i: int):
        rng = np.random.default_rng(i) if self.deterministic else np.random.default_rng()
        fi = int(rng.choice(len(self.rows), p=self.prob))
        row = self.rows[fi]
        with np.load(self.root / row["file"]) as z:
            mel, feat = z["mel"], z["feat"]
        both, mask = self._crop(np.concatenate([mel, feat], axis=1), self.crop, rng)
        mel_c = (both[:, :N_MELS] - MEL_MEAN) / MEL_STD
        feat_c = both[:, N_MELS : N_MELS + N_FEAT]
        mel_c[~mask] = 0.0

        # 참조: 같은 group의 다른 파일. (같은 파일이면 내용이 새어 들어가 음색 대신 내용을 베낄 수 있다)
        others = [j for j in self.by_group[row["group"]] if j != fi] or [fi]
        rj = others[int(rng.integers(0, len(others)))]
        with np.load(self.root / self.rows[rj]["file"]) as z:
            ref, ref_mask = self._crop(z["mel"], self.ref_crop, rng)
        ref = (ref - MEL_MEAN) / MEL_STD
        ref[~ref_mask] = 0.0
        return (torch.from_numpy(mel_c), torch.from_numpy(feat_c), torch.from_numpy(mask),
                torch.from_numpy(ref), torch.from_numpy(ref_mask))
