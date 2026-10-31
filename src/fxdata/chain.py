"""Random guitar effect chains whose knobs live in [0, 1].

노브 값은 [0, 1]로 정규화해서 저장하고, configs/effects.yaml 의 범위로 실제 값으로 바꾼다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pedalboard import (
    Compressor,
    Distortion,
    Gain,
    HighShelfFilter,
    LowShelfFilter,
    Pedalboard,
    Reverb,
)

ORDER = ("comp", "drive", "tone", "reverb")


def lerp(lo: float, hi: float, u: float) -> float:
    return lo + (hi - lo) * u


def sample_knobs(rng: np.random.Generator, cfg: dict[str, Any]) -> dict[str, Any]:
    """Draw one random chain: per effect an on/off flag and normalized knobs."""
    out: dict[str, Any] = {}
    for name in ORDER:
        spec = cfg["effects"][name]
        out[name] = {
            "enabled": bool(rng.random() < spec["p_enable"]),
            "knobs": {k: float(rng.random()) for k in spec["knobs"]},
        }
    return out


def physical_values(cfg: dict[str, Any], sample: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Normalized knobs -> physical values (for logging / debugging)."""
    vals: dict[str, dict[str, float]] = {}
    for name in ORDER:
        ranges = cfg["effects"][name]["knobs"]
        vals[name] = {k: lerp(*ranges[k], sample[name]["knobs"][k]) for k in ranges}
    return vals


def build_board(cfg: dict[str, Any], sample: dict[str, Any]) -> Pedalboard:
    phys = physical_values(cfg, sample)
    plugins = []
    for name in ORDER:
        if not sample[name]["enabled"]:
            continue
        v = phys[name]
        if name == "comp":
            plugins.append(
                Compressor(
                    threshold_db=v["threshold_db"],
                    ratio=v["ratio"],
                    attack_ms=10.0,
                    release_ms=120.0,
                )
            )
        elif name == "drive":
            plugins.append(Distortion(drive_db=v["gain_db"]))
            plugins.append(Gain(gain_db=v["level_db"]))
        elif name == "tone":
            plugins.append(LowShelfFilter(cutoff_frequency_hz=250.0, gain_db=-v["tilt_db"]))
            plugins.append(HighShelfFilter(cutoff_frequency_hz=2500.0, gain_db=v["tilt_db"]))
        elif name == "reverb":
            plugins.append(
                Reverb(
                    room_size=v["room_size"],
                    damping=0.5,
                    wet_level=v["wet_level"],
                    dry_level=1.0 - v["wet_level"],
                    width=1.0,
                )
            )
    return Pedalboard(plugins)
