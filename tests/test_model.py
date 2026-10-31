import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

torch = pytest.importorskip("torch")

from fxdata import render_dataset  # noqa: E402
from fxmodel.cond import chain_from_settings, cond_dim, cond_from_chain  # noqa: E402
from fxmodel.infer import process_file  # noqa: E402
from fxmodel.loss import MRSTFTLoss  # noqa: E402
from fxmodel.model import FiLMTCN  # noqa: E402
from fxmodel.trainer import train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "effects_mvp.yaml").read_text(encoding="utf-8"))


def _pluck(freq, seconds, sr):
    t = np.arange(int(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 6))
    return (x * np.exp(-3.0 * t)).astype(np.float32)


def _make_dry(dry_dir: Path, sr: int) -> None:
    dry_dir.mkdir(parents=True)
    for i, f in enumerate([110.0, 146.8, 164.8, 196.0, 220.0, 246.9]):
        sig = np.concatenate([_pluck(f, 2.0, sr) for _ in range(5)])  # 10 s
        sf.write(str(dry_dir / f"take{i}.wav"), sig, sr)


def test_cond_vector_zeroes_disabled_knobs():
    chain = chain_from_settings(CFG, {"drive.gain_db": 0.8})
    v = cond_from_chain(chain, CFG)
    assert len(v) == cond_dim(CFG) == 11
    assert v[:4] == [0.0, 1.0, 0.0, 0.0]  # comp, drive, tone, reverb
    assert v[4:6] == [0.0, 0.0]            # comp 노브는 꺼져서 0
    assert v[6] == pytest.approx(0.8)      # drive.gain_db


def test_chain_from_settings_rejects_bad_input():
    with pytest.raises(ValueError):
        chain_from_settings(CFG, {"fuzz.gain": 0.5})
    with pytest.raises(ValueError):
        chain_from_settings(CFG, {"drive.nope": 0.5})
    with pytest.raises(ValueError):
        chain_from_settings(CFG, {"drive.gain_db": 1.5})


def test_model_starts_as_identity_and_uses_condition():
    m = FiLMTCN(cond_dim=11, channels=8, layers=4, stacks=1)
    x = torch.randn(2, 1, 2048) * 0.1
    c = torch.rand(2, 11)
    assert torch.allclose(m(x, c), x)  # 영(0) 초기화라 처음에는 항등
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    y1, y2 = m(x, c), m(x, c.flip(0) * 0.0)
    assert y1.shape == x.shape
    assert not torch.allclose(y1, y2)  # 조건이 출력에 영향을 줘야 한다
    assert m.receptive_field() == 1 + 2 * (1 + 2 + 4 + 8)


def test_mrstft_loss_is_zero_for_identical_and_finite_for_silence():
    crit = MRSTFTLoss()
    x = torch.randn(2, 1, 8192)
    assert float(crit(x, x)) < 1e-4
    z = torch.zeros(2, 1, 8192, requires_grad=True)
    loss = crit(z, x)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


def test_train_resume_and_infer(tmp_path):
    sr = CFG["sample_rate"]
    _make_dry(tmp_path / "dry", sr)
    render_dataset(tmp_path / "dry", tmp_path / "rendered", CFG, variants=3, seed=1)
    assert (tmp_path / "rendered" / "render_config.json").exists()

    # 다시 렌더링해도 메타가 중복으로 쌓이지 않는다
    n1 = sum(len(p.read_text().splitlines()) for p in (tmp_path / "rendered").glob("*/meta.jsonl"))
    render_dataset(tmp_path / "dry", tmp_path / "rendered", CFG, variants=3, seed=1)
    n2 = sum(len(p.read_text().splitlines()) for p in (tmp_path / "rendered").glob("*/meta.jsonl"))
    assert n1 == n2 > 0

    kw = dict(batch_size=4, crop=4096, channels=8, layers=4, stacks=1, val_every=20,
              save_every=20, val_items=8, num_workers=0, device="cpu", log_every=10)
    out = tmp_path / "run"
    m1 = train(tmp_path / "rendered", out, steps=20, **kw)
    assert m1["finished"] and (out / "best.pt").exists() and (out / "last.pt").exists()
    assert list((out / "samples").glob("step*/0_pred.wav"))
    assert np.isfinite([m1["test_model"], m1["test_identity"]]).all()

    m2 = train(tmp_path / "rendered", out, steps=40, **kw)  # 이어서 학습
    assert torch.load(out / "last.pt", weights_only=False)["step"] == 40
    assert m2["steps"] == 40

    inp = tmp_path / "in.wav"
    sf.write(str(inp), _pluck(196.0, 3.0, 44100), 44100)  # 샘플레이트가 달라도 처리돼야 한다
    res = process_file(out / "best.pt", inp, tmp_path / "out.wav", {"drive.gain_db": 0.7})
    y, ysr = sf.read(str(res))
    assert ysr == sr and len(y) == int(3.0 * sr) and np.isfinite(y).all()
    assert json.loads((out / "metrics.json").read_text())["steps"] == 40
