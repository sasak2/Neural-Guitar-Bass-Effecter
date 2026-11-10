"""Checks for the generative package on synthetic signals (no downloads, CPU only)."""

import numpy as np
import pytest
import soundfile as sf
import torch

from fxgen.data import GenDataset
from fxgen.features import extract_features, feat_f0_hz, pitch_error_cents, transpose
from fxgen.flow import fm_loss, sample
from fxgen.mel import HOP, N_MELS, SR, LogMel, frame_rms_db
from fxgen.model import FlowNet, pitch_encoding
from fxgen.prep import prepare
from fxgen.trainer import train
from fxgen.vocoder import griffin_lim_decode


def tone(freq: float, seconds: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t) + 0.3 * amp * np.sin(4 * np.pi * freq * t)).astype(np.float32)


def test_mel_shape_and_peak():
    wav = tone(1000.0, 1.0)
    mel = LogMel()(torch.from_numpy(wav)[None])
    assert mel.shape == (1, N_MELS, len(wav) // HOP)
    # 1 kHz 성분이 있는 멜 대역이 가장 커야 한다
    import librosa
    centers = librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0, fmax=SR / 2)[1:-1]
    assert abs(centers[int(mel[0].mean(-1).argmax())] - 1000.0) < 80.0


def test_frame_rms_matches_level():
    db = frame_rms_db(tone(440.0, 0.5, amp=0.5))
    assert len(db) == int(SR * 0.5) // HOP
    assert abs(np.median(db) - 20 * np.log10(0.5 * np.sqrt((1 + 0.09) / 2))) < 0.5


@pytest.mark.parametrize("method", ["pyin", "crepe"])
def test_features_follow_pitch_and_silence(method):
    if method == "crepe":
        pytest.importorskip("torchcrepe")
    wav = np.concatenate([tone(220.0, 0.8), np.zeros(int(SR * 0.4), np.float32), tone(330.0, 0.8)])
    wav = wav[: len(wav) // HOP * HOP]
    feat = extract_features(wav, method=method, crepe_model="tiny")
    assert feat.shape == (len(wav) // HOP, 4)
    f0 = feat_f0_hz(feat)
    fps = SR / HOP
    a, gap, b = f0[int(0.2 * fps):int(0.6 * fps)], f0[int(0.9 * fps):int(1.1 * fps)], f0[int(1.4 * fps):int(1.8 * fps)]
    assert abs(1200 * np.log2(np.median(a[a > 0]) / 220.0)) < 30
    assert abs(1200 * np.log2(np.median(b[b > 0]) / 330.0)) < 30
    assert (gap == 0).all()                       # 무음 구간은 무성
    assert feat[int(1.0 * fps), 3] < 0.2 < feat[int(0.4 * fps), 3]

    up = transpose(feat, 12)
    err = pitch_error_cents(feat, up)
    assert abs(err["median_abs"] - 1200) < 1 and err["octave_err_frac"] == 1.0


def test_pitch_encoding_distinguishes_small_steps():
    f = torch.zeros(1, 2, 4)
    f[0, :, 1] = 1.0
    f[0, 0, 0], f[0, 1, 0] = 0.5, 0.5 + 0.2 / 60     # 20센트 차이
    e = pitch_encoding(f)
    assert (e[0, 0] - e[0, 1]).abs().max() > 0.1
    f[0, :, 1] = 0.0                                  # 무성이면 음높이 성분이 0
    assert pitch_encoding(f)[..., :16].abs().max() == 0


def small_model():
    torch.manual_seed(0)
    return FlowNet(dim=64, depth=2, heads=2, timbre_dim=32)


def test_model_shapes_mask_and_null_timbre():
    m = small_model()
    for p in (m.out.weight, m.out.bias):
        torch.nn.init.normal_(p, std=0.1)
    b, t = 2, 40
    x, feat = torch.randn(b, t, N_MELS), torch.rand(b, t, 4)
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[1, 30:] = False
    ref, ref_mask = torch.randn(b, 50, N_MELS), torch.ones(b, 50, dtype=torch.bool)
    tim = m.timbre(ref, ref_mask)
    v = m(x, torch.rand(b), feat, tim, mask)
    assert v.shape == (b, t, N_MELS) and v[1, 30:].abs().max() == 0
    # 패딩 구간의 내용은 유효 구간의 출력에 영향을 주지 않아야 한다
    x2 = x.clone()
    x2[1, 30:] = 99.0
    t_ = torch.tensor([0.3, 0.7])
    assert torch.allclose(m(x, t_, feat, tim, mask)[1, :30], m(x2, t_, feat, tim, mask)[1, :30], atol=1e-4)
    # 참조의 패딩도 음색 벡터에 영향을 주지 않아야 한다
    rm = ref_mask.clone()
    rm[:, 40:] = False
    r2 = ref.clone()
    r2[:, 40:] = 5.0
    assert torch.allclose(m.timbre(ref, rm), m.timbre(r2, rm), atol=1e-4)
    assert m(x, t_, feat, None, mask).shape == v.shape


def test_sampler_keeps_known_frames_and_is_seeded():
    m = small_model()
    torch.nn.init.normal_(m.out.weight, std=0.05)
    feat, tim = torch.rand(1, 32, 4), torch.randn(1, 32)
    known = torch.randn(1, 32, N_MELS)
    km = torch.zeros(1, 32, dtype=torch.bool)
    km[0, :8] = True
    g = lambda: torch.Generator().manual_seed(3)  # noqa: E731
    a = sample(m, feat, tim, steps=4, cfg=1.5, known=known, known_mask=km, generator=g())
    b = sample(m, feat, tim, steps=4, cfg=1.5, known=known, known_mask=km, generator=g())
    assert torch.equal(a, b) and torch.equal(a[0, :8], known[0, :8])
    assert not torch.allclose(a[0, 8:], known[0, 8:])


def test_flow_loss_learns_pitch_dependent_target():
    """음높이에 따라 달라지는 멜을 조건만 보고 맞힐 수 있어야 한다 (조건이 실제로 쓰이는지 확인)."""
    m = small_model()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)

    def batch(n=16, t=24):
        p = torch.rand(n, 1, 1) * 0.8 + 0.1
        feat = torch.cat([p.expand(n, t, 1), torch.ones(n, t, 1), torch.ones(n, t, 1), torch.full((n, t, 1), 0.8)], -1)
        bins = torch.linspace(0, 1, N_MELS)[None, None]
        mel = 2.0 * torch.exp(-((bins - p) ** 2) / 0.005) - 1.0
        mel = mel.expand(n, t, N_MELS).contiguous()
        return mel, feat, torch.ones(n, t, dtype=torch.bool), torch.zeros(n, 8, N_MELS), torch.ones(n, 8, dtype=torch.bool)

    first = None
    for i in range(200):
        loss = fm_loss(m, *batch())
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or loss.item()
    assert loss.item() < 0.6 * first   # 작은 모델(64차원)은 128차원 노이즈를 다 못 실어서 0까지는 안 간다
    mel, feat, mask, ref, rmask = batch(4)
    out = sample(m, feat, m.timbre(ref, rmask), steps=8, cfg=1.0, generator=torch.Generator().manual_seed(0))
    peak_err = (out.mean(1).argmax(-1) - mel.mean(1).argmax(-1)).abs().float().mean()
    assert peak_err < 6             # 128개 대역 중 봉우리 위치가 맞는다 (참조는 0이라 음높이 조건만으로)


def test_prepare_dataset_and_short_training(tmp_path):
    raw = tmp_path / "raw"
    (raw / "v").mkdir(parents=True)
    (raw / "g").mkdir()
    for i, f in enumerate([200.0, 260.0, 310.0, 240.0]):
        sf.write(raw / "v" / f"female{1 + i % 2}__a{i}.wav", tone(f, 1.6), SR)
    for i, f in enumerate([110.0, 150.0]):
        sf.write(raw / "g" / f"P1_music__directinput_{i}.wav", np.stack([tone(f, 1.6)] * 2, 1), 48000)
    rows = prepare([("voice", raw / "v"), ("guitar", raw / "g")], tmp_path / "d", device="cpu",
                   f0_method="pyin", test_groups=(), val_frac=0.0)
    assert len(rows) == 6 and {r["group"] for r in rows} == {"voice_female1", "voice_female2", "guitar_P1"}
    assert len(prepare([("voice", raw / "v")], tmp_path / "d", device="cpu", f0_method="pyin",
                       test_groups=("voice_female2",), val_frac=0.0)) == 4     # 캐시 재사용

    prepare([("voice", raw / "v"), ("guitar", raw / "g")], tmp_path / "d", device="cpu",
            f0_method="pyin", test_groups=(), val_frac=0.0)
    ds = GenDataset(tmp_path / "d", "train", crop=160, ref_crop=64, deterministic=True, items=8)
    mel, feat, mask, ref, ref_mask = ds[0]
    assert mel.shape == (160, N_MELS) and feat.shape == (160, 4) and ref.shape == (64, N_MELS)
    assert mask.sum() < 160 and mel[~mask].abs().max() == 0      # 1.6초(137프레임)라 뒤는 패딩
    assert all(torch.equal(a, b) for a, b in zip(ds[3], ds[3]))

    kw = dict(batch_size=4, crop=64, ref_crop=32, dim=64, depth=2, heads=2, val_every=10, save_every=10,
              val_items=8, num_workers=0, device="cpu", log_every=5)
    m1 = train(tmp_path / "d", tmp_path / "run", steps=10, **kw)
    assert m1["finished"] and (tmp_path / "run" / "best.pt").exists()
    m2 = train(tmp_path / "d", tmp_path / "run", steps=20, **kw)           # 이어서 학습
    assert m2["steps"] == 20


def test_griffin_lim_fallback_keeps_length_and_pitch():
    wav = tone(330.0, 1.0)
    wav = wav[: len(wav) // HOP * HOP]
    mel = LogMel()(torch.from_numpy(wav)[None])[0].T
    y = griffin_lim_decode(mel, n_iter=8)
    assert len(y) == len(wav)
    err = pitch_error_cents(extract_features(wav, method="pyin"), extract_features(y, method="pyin"))
    assert err["median_abs"] < 30
