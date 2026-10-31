import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from fxdata import render_dataset, split_of

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "effects.yaml").read_text(encoding="utf-8"))


def _pluck(freq: float, seconds: float, sr: int) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 6))
    return (x * np.exp(-3.0 * t)).astype(np.float32)


def _make_dry(dry_dir: Path, sr: int) -> None:
    dry_dir.mkdir(parents=True)
    for i, f in enumerate([110.0, 164.8, 220.0]):
        sig = np.concatenate([_pluck(f, 2.0, sr) for _ in range(5)])  # 10 s
        sf.write(str(dry_dir / f"take{i}.wav"), sig, sr)


def _read_meta(out: Path) -> list[dict]:
    rows = []
    for p in sorted(out.glob("*/meta.jsonl")):
        rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    return rows


def test_split_is_deterministic():
    assert split_of("a/b") == split_of("a/b")
    assert {split_of(f"k{i}") for i in range(200)} == {"train", "val", "test"}


def test_render_writes_pairs_and_meta(tmp_path):
    sr = CFG["sample_rate"]
    _make_dry(tmp_path / "dry", sr)
    n = render_dataset(tmp_path / "dry", tmp_path / "out", CFG, variants=3, seed=1)
    rows = _read_meta(tmp_path / "out")
    assert n == len(rows) > 0
    for r in rows:
        dry, _ = sf.read(str(tmp_path / "out" / r["dry"]))
        wet, _ = sf.read(str(tmp_path / "out" / r["wet"]))
        assert len(dry) == len(wet) == int(CFG["clip_seconds"] * sr)
        assert np.max(np.abs(wet)) <= 1.0
        if any(r["chain"][e]["enabled"] for e in r["chain"]):
            assert not np.allclose(dry, wet)


def test_same_seed_same_chains(tmp_path):
    sr = CFG["sample_rate"]
    _make_dry(tmp_path / "dry", sr)
    render_dataset(tmp_path / "dry", tmp_path / "a", CFG, variants=2, seed=7)
    render_dataset(tmp_path / "dry", tmp_path / "b", CFG, variants=2, seed=7)
    ca = [r["chain"] for r in _read_meta(tmp_path / "a")]
    cb = [r["chain"] for r in _read_meta(tmp_path / "b")]
    assert ca == cb
