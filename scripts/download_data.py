#!/usr/bin/env python3
"""Download the dry datasets listed in DATA_LICENSES.md.

데이터 파일은 레포에 올리지 않는다. 이 스크립트로 각자 내려받는다.

guitar_techs: Zenodo record 14963133 (CC BY 4.0)의 압축 파일에서 direct input(DI) wav만 꺼낸다.
              압축 안의 구조는 `<아카이브>/audio/directinput/*.wav` (48kHz 24bit, 좌우 동일한 모노).
egfxset_clean: Zenodo record 7044411 (CC BY 4.0)의 Clean.zip (431MB).
"""

from __future__ import annotations

import argparse
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

GT_RECORD = "https://zenodo.org/records/14963133/files"
# 용량(MB): P3_music 130, P1/P2_techniques 326/396, P1/P2_scales 453/471, chords는 1GB 이상
GT_DEFAULT_PARTS = ["P3_music", "P1_techniques", "P2_techniques", "P1_scales", "P2_scales"]
GT_ALL_PARTS = GT_DEFAULT_PARTS + [
    "P1_chords", "P2_chords", "P1_singlenotes", "P2_singlenotes",
]
EGFXSET_CLEAN_URL = "https://zenodo.org/record/7044411/files/Clean.zip?download=1"


def download(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f"downloading {url}\n  -> {dest}", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "neural-guitar-bass-effecter"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:  # noqa: S310
                total = int(r.headers.get("Content-Length", 0))
                done, last = 0, time.time()
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 15:
                        pct = f" ({100 * done / total:.0f}%)" if total else ""
                        print(f"  {done / 1e6:.0f} MB{pct}", flush=True)
                        last = time.time()
            return
        except Exception as e:  # 네트워크 오류는 재시도
            print(f"  attempt {attempt} failed: {e}", flush=True)
            if attempt == retries:
                raise
            time.sleep(5 * attempt)


def is_di_member(name: str) -> bool:
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    if not parts or parts[0] == "__MACOSX" or parts[-1].startswith("._"):
        return False
    return parts[-1].lower().endswith(".wav") and any(p.lower() == "directinput" for p in parts[:-1])


def extract_di(zip_path: Path, out_dir: Path) -> int:
    """zip 안의 directinput/*.wav만 out_dir에 평평하게 풀어 쓴다. 풀어 쓴 파일 수를 반환."""
    n = 0
    stem = zip_path.stem
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            if not is_di_member(m):
                continue
            parts = [p for p in m.split("/") if p]
            idx = [p.lower() for p in parts].index("directinput")
            rel = "__".join(parts[idx + 1 :])
            target = out_dir / f"{stem}__{rel}"
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    if n == 0:
        with zipfile.ZipFile(zip_path) as z:
            sample = [x for x in z.namelist() if not x.startswith("__MACOSX")][:15]
        print(f"[warn] {zip_path.name}: directinput/*.wav 를 찾지 못했다. 압축 내부 예시:")
        for x in sample:
            print("   ", x)
    return n


def get_guitar_techs(dest: Path, parts: list[str], keep_zips: bool) -> None:
    out_dir = dest / "guitar_techs_di"
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for part in parts:
        existing = list(out_dir.glob(f"{part}__*.wav"))
        if existing:
            print(f"{part}: 이미 {len(existing)}개 있음, 건너뜀")
            total += len(existing)
            continue
        zip_path = dest / "_zips" / f"{part}.zip"
        download(f"{GT_RECORD}/{part}.zip?download=1", zip_path)
        n = extract_di(zip_path, out_dir)
        print(f"{part}: DI wav {n}개 추출")
        total += n
        if not keep_zips:
            zip_path.unlink(missing_ok=True)
    if total == 0:
        raise SystemExit("DI 파일을 하나도 얻지 못했다. 위의 경고를 확인할 것.")
    print(f"완료: {total}개 파일 -> {out_dir}")


def get_egfxset_clean(dest: Path) -> None:
    zip_path = dest / "_zips" / "EGFxSet_Clean.zip"
    download(EGFXSET_CLEAN_URL, zip_path)
    out_dir = dest / "egfxset_clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            parts = [p for p in m.split("/") if p]
            if not parts or parts[0] == "__MACOSX" or parts[-1].startswith("._"):
                continue
            if parts[-1].lower().endswith(".wav"):
                with z.open(m) as src, open(out_dir / "__".join(parts), "wb") as dst:
                    shutil.copyfileobj(src, dst)
    zip_path.unlink(missing_ok=True)
    print(f"완료 -> {out_dir} ({len(list(out_dir.glob('*.wav')))}개)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dataset", choices=["guitar_techs", "egfxset_clean"])
    ap.add_argument("--dest", type=Path, default=Path("data"))
    ap.add_argument(
        "--parts", nargs="+", default=GT_DEFAULT_PARTS, choices=GT_ALL_PARTS,
        help="guitar_techs에서 받을 아카이브 (기본: music + techniques + scales)",
    )
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()

    if args.dataset == "guitar_techs":
        get_guitar_techs(args.dest, args.parts, args.keep_zips)
    else:
        get_egfxset_clean(args.dest)


if __name__ == "__main__":
    main()
