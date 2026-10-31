# 뉴럴 기타베이스 이펙터

참조 이펙트를 조건으로 받아 기타 소리를 변환하는 모델을 만들기 위한 실험 레포. 현재는 **1단계: 노브 값을 조건으로 받는 블랙박스 변환 모델**을 목표로 데이터 파이프라인부터 만든다.

## 단계

1. **노브 조건 모델 (현재)**: `dry 오디오 + 노브 값 -> wet 오디오`. 합성 데이터로 학습한다.
2. 참조음 임베딩 조건: 노브 값 자리에 참조음 인코더의 출력을 넣는다.
3. 처음 보는 참조음(하드웨어 페달 등)으로 일반화 평가.

오프라인(파일 입력, 파일 출력)으로 먼저 검증하고, 실시간화와 경량화는 나중에 생각한다.

## 데이터

dry 기타 DI 녹음을 공개 데이터셋에서 받아서 `pedalboard`로 랜덤 이펙트 체인(comp, drive, tone, reverb)을 걸어 wet을 만든다. 노브 값(0~1 정규화)은 `meta.jsonl`에 같이 저장한다. 데이터셋 선택 기준과 라이선스는 [DATA_LICENSES.md](DATA_LICENSES.md) 참고.

- train/val/test는 **원본 dry 파일 단위**로 나눈다. 같은 연주가 양쪽에 섞이지 않게 하기 위해서다.
- 데이터와 렌더링 결과는 레포에 올리지 않는다.

## 사용법

```bash
pip install -r requirements.txt
pytest -q

# 데이터 받기 (Guitar-TECHS는 안내만 출력)
python scripts/download_data.py guitar_techs
python scripts/download_data.py egfxset_clean --dest data

# wet 렌더링
python scripts/render_dataset.py --dry-dir data/guitar_techs_di --out-dir rendered --variants 4 --seed 0
```

Colab에서는 `notebooks/00_colab_quickstart.ipynb`를 참고.

## 구조

```
configs/effects.yaml        이펙트 체인과 노브 범위
src/fxdata/chain.py         랜덤 체인 샘플링, pedalboard 체인 생성
src/fxdata/render.py        클립 분할, 렌더링, 메타 기록, 분할
scripts/                    다운로드, 렌더링 CLI
tests/                      합성 신호로 하는 검증
```

## 해야 할 일

- [ ] Guitar-TECHS 파일 구조 확인 후 DI 폴더 지정
- [ ] 학습 코드 (TCN 또는 WaveNet류 + FiLM 조건, 멀티 해상도 STFT 손실)
- [ ] 학습에 안 쓴 노브 조합과 안 쓴 dry 연주로 평가
- [ ] 코드 라이선스 결정 (아직 미정)
