"""Condition vectors: effect on/off flags + normalized knob values.

조건 벡터 = [이펙트별 켜짐 플래그 4개] + [노브 값 7개 (꺼진 이펙트의 노브는 0)].
"""

from __future__ import annotations

from typing import Any

from fxdata.chain import ORDER


def cond_dim(cfg: dict[str, Any]) -> int:
    return len(ORDER) + sum(len(cfg["effects"][n]["knobs"]) for n in ORDER)


def cond_from_chain(chain: dict[str, Any], cfg: dict[str, Any]) -> list[float]:
    flags = [1.0 if chain[n]["enabled"] else 0.0 for n in ORDER]
    knobs: list[float] = []
    for name, flag in zip(ORDER, flags):
        for k in cfg["effects"][name]["knobs"]:
            knobs.append(flag * float(chain[name]["knobs"][k]))
    return flags + knobs


def chain_from_settings(cfg: dict[str, Any], settings: dict[str, float]) -> dict[str, Any]:
    """Build a chain from {"drive.gain_db": 0.8, "tone.enabled": 1, ...}.

    - 모든 이펙트는 기본 꺼짐, 노브 기본값 0.5
    - `effect.knob=값(0~1)`을 주면 그 이펙트는 자동으로 켜진다 (`effect.enabled`를 따로 주면 그 값이 우선)
    """
    chain = {
        n: {"enabled": False, "knobs": {k: 0.5 for k in cfg["effects"][n]["knobs"]}} for n in ORDER
    }
    explicit: set[str] = set()
    for key, value in settings.items():
        try:
            effect, name = key.split(".", 1)
        except ValueError as e:
            raise ValueError(f"'{key}' 형식이 아니다. 예: drive.gain_db=0.8") from e
        if effect not in chain:
            raise ValueError(f"알 수 없는 이펙트 '{effect}'. 사용 가능: {list(ORDER)}")
        if name == "enabled":
            chain[effect]["enabled"] = bool(float(value))
            explicit.add(effect)
        elif name in chain[effect]["knobs"]:
            v = float(value)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{key}는 0~1 범위여야 한다 (받은 값: {v})")
            chain[effect]["knobs"][name] = v
            if effect not in explicit:
                chain[effect]["enabled"] = True
        else:
            raise ValueError(
                f"'{effect}'에 노브 '{name}'가 없다. 사용 가능: {list(chain[effect]['knobs'])}"
            )
    return chain
