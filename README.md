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

### Colab에서 원클릭 학습 (권장)

1. `notebooks/01_train_oneclick.ipynb`를 Colab에서 연다.
2. 런타임을 GPU(T4)로 바꾼다.
3. (레포가 private일 때만) 읽기 권한이 있는 GitHub 토큰을 Colab Secrets에 `GH_TOKEN`으로 저장한다.
4. "모두 실행"을 누른다. 데이터 다운로드, 렌더링, 학습, 샘플 듣기까지 이어서 진행된다.

체크포인트는 Google Drive(`MyDrive/neural-guitar-bass-effecter/runs/<RUN_NAME>`)에 저장된다. 세션이 끊기거나 시간 제한(`MAX_MINUTES`)에 걸려도 같은 이름으로 다시 실행하면 이어서 학습한다. 흐름만 먼저 점검하고 싶으면 노트북 설정에서 `SMOKE_TEST = True`로 둔다.

### 로컬에서

```bash
pip install -r requirements.txt
pytest -q

# 1. 데이터 (Guitar-TECHS의 DI 오디오만 추출)
python scripts/download_data.py guitar_techs --dest data

# 2. wet 렌더링 (MVP 설정: 48 kHz, comp/drive/tone)
python scripts/render_dataset.py --dry-dir data/guitar_techs_di --out-dir rendered \
    --config configs/effects_mvp.yaml --variants 3 --seed 0

# 3. 학습 (같은 --out-dir로 다시 실행하면 이어서 학습)
python scripts/train.py --data-dir rendered --out-dir runs/exp1 --steps 30000

# 4. 내 파일에 적용 (노브는 0~1, 지정한 이펙트만 켜진다)
python scripts/infer.py --ckpt runs/exp1/best.pt --input riff.wav --output riff_fx.wav \
    --set drive.gain_db=0.8 --set drive.level_db=0.6 --set tone.tilt_db=0.6
```

학습이 끝나면 `metrics.json`에 모델과 identity(dry를 그대로 내보내는 기준선)의 val/test 손실이 같이 기록된다. 모델이 identity보다 확실히 낮아야 의미가 있다.

## 모델

- 입력: dry 모노 오디오 + 노브 조건 벡터(이펙트 켜짐 4개 + 노브 7개 = 11차원)
- 구조: dilated 비인과 TCN(WaveNet식 gated activation) + FiLM 조건 + skip 연결, 출력은 `입력 + 보정`이라 처음엔 항등
- 손실: 멀티 해상도 STFT 손실 + 시간영역 L1
- MVP 설정(`configs/effects_mvp.yaml`)은 reverb를 끈다. 현재 구조의 수용 영역이 약 0.2초라 리버브 꼬리를 배울 수 없기 때문이다.

## 구조

```
configs/effects.yaml        이펙트 체인과 노브 범위 (44.1 kHz, reverb 포함)
configs/effects_mvp.yaml    MVP 학습용 설정 (48 kHz, reverb 제외)
src/fxdata/                 랜덤 체인 샘플링, 렌더링, 메타 기록, 분할
src/fxmodel/                조건 벡터, FiLM TCN, 손실, 데이터셋, 학습, 추론
scripts/                    download_data, render_dataset, train, infer
notebooks/                  Colab 원클릭 학습
tests/                      합성 신호로 하는 검증
```

## 해야 할 일

- [x] Guitar-TECHS 파일 구조 확인 후 DI 폴더 지정
- [x] 학습 코드 (TCN + FiLM 조건, 멀티 해상도 STFT 손실)
- [x] Colab 원클릭 학습 노트북
- [ ] 실제 GPU 학습 결과 확인 (Colab에서 아직 안 돌려 봄)
- [ ] 학습에 안 쓴 노브 조합과 안 쓴 dry 연주로 평가
- [ ] 리버브/딜레이처럼 긴 꼬리를 위한 더 큰 수용 영역 구조
- [ ] 베이스 지원, 2단계(참조음 임베딩)
- [ ] 코드 라이선스 결정 (아직 미정)
