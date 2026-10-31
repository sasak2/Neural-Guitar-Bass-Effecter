#!/usr/bin/env python3
"""Download helper for the dry datasets listed in DATA_LICENSES.md.

데이터 파일은 레포에 올리지 않는다. 이 스크립트로 각자 내려받는다.
URL을 직접 확인하지 못한 항목은 추측으로 받지 않고 안내만 출력한다.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# EGFxSet: Zenodo record 7044411 (CC BY 4.0). clean 음원 파일명은 RemFx README 기준이라
# 받기 전에 https://zenodo.org/record/7044411 에서 실제 파일명을 확인할 것.
EGFXSET_CLEAN_URL = "https://zenodo.org/record/7044411/files/Clean.zip?download=1"


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}\n  -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (고정 URL만 사용)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=["egfxset_clean", "guitar_techs"])
    ap.add_argument("--dest", type=Path, default=Path("data"))
    args = ap.parse_args()

    if args.dataset == "egfxset_clean":
        fetch(EGFXSET_CLEAN_URL, args.dest / "EGFxSet_Clean.zip")
        print("압축을 풀고 dry 폴더로 지정하세요: unzip data/EGFxSet_Clean.zip -d data/egfxset_clean")
    else:
        print(
            "Guitar-TECHS(CC BY 4.0)는 공식 사이트 guitar-techs.github.io 에서 내려받으세요.\n"
            "파일 구조를 확인한 뒤 direct input(DI) 오디오만 data/guitar_techs_di/ 에 두세요."
        )


if __name__ == "__main__":
    main()
