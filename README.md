# 뉴럴 기타베이스 이펙터

연주(기타/베이스)에 참조 소리의 톤을 입혀 주는 신경망 이펙터를 만들기 위한 실험 레포. 목표는 "목표 소리를 들려주면 그 톤으로 바뀌고, 연주자의 터치(세기 변화, 어택, 뮤트)는 남는" 모델이다. 지금은 가장 쉬운 부분(노브 조건 이펙터)으로 학습 파이프라인을 검증한 단계다.

## 목표와 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 노브 값을 조건으로 받는 이펙터: `dry + 노브 -> wet` (합성 데이터) | 파이프라인 검증 완료 |
| 2 | 데이터 강화: 센 디스토션/퍼즈, 입력 레벨/잡음/픽업 EQ 증강, 속주 변환 | 계획 |
| 3 | 참조음 임베딩 조건: 노브 자리에 참조음 인코더 출력. 가까운 목표(같은 악기군의 음색 교환, 예: 프레시전에서 재즈 베이스 톤)부터 | 계획 |
| 4 | 먼 목표(목소리 등 완전히 다른 소리): 연주 특징(음높이/세기/어택)을 입력으로 받는 생성형 | 프로토타입(특징 추출, 합성 정답) |

오프라인(파일 입력, 파일 출력)으로 먼저 검증하고, 실시간화와 경량화는 나중에 생각한다.

### 설계 원칙

- **터치 보존은 구조가 정한다.** 입력 파형을 직접 변환하는 구조(1, 3단계)는 미세한 세기 변화와 어택이 자연스럽게 남는다. 음높이/세기만 뽑아 재합성하는 구조(4단계)는 그 둘에 안 담긴 터치가 사라진다. 그래서 가까운 목표는 파형 변환, 먼 목표만 생성형으로 간다.
- **정답 쌍이 병목이다.** 같은 연주를 두 악기/두 톤으로 동시에 녹음한 쌍은 거의 없다. 합성 데이터(이펙트 체인, 픽업 변화 DSP)로 먼저 확인하고, 실제 쌍은 오디오 인터페이스가 생기면 직접 녹음한다. 합성에서 실제 연주로 가면 변환이 약해질 수 있다(비슷한 연구에서 보고됨).
- **못 친 소리도 필요하다.** 이펙터는 입력에 따라 반응이 달라지는 비선형 시스템이라, 줄 긁는 소리, 잡음, 들쭉날쭉한 세기, 서툰 연주도 학습/평가에 들어가야 한다.
- **dry가 없는 wet 데이터도 쓴다.** 이미 이펙트가 걸린 소리에서 연주 특징을 뽑아 자기 복원으로 학습하면 dry 없이도 쓸 수 있다(4단계 후보).

## 현재 결과 (1단계, 합성 이펙트, 중간 체크포인트 기준)

Guitar-TECHS DI 46개 파일로 만든 쌍(train 4338 / val 360 / test 906)을 T4에서 약 7천 스텝 학습한 중간 체크포인트를 전체 test 906쌍, 클립 전체 길이로 평가했다. 모델 평균 손실 0.283, dry를 그대로 내보내는 기준선(identity) 2.158. 손실이 큰 쌍은 컴프레서가 강하게 걸린 조합에 몰렸다. 주관 평가(소리 듣기)는 아직 안 했고, 합성 이펙트가 순한 편이라는 한계가 있다.

## 데이터

dry 기타 DI 녹음을 공개 데이터셋에서 받아서 `pedalboard`로 랜덤 이펙트 체인(comp, drive, tone, reverb)을 걸어 wet을 만든다. 노브 값(0~1 정규화)은 `meta.jsonl`에 같이 저장한다. 데이터셋 선택 기준과 라이선스, 검토했지만 쓰지 않는 출처는 [DATA_LICENSES.md](DATA_LICENSES.md) 참고.

- train/val/test는 **원본 dry 파일 단위**로 나눈다. 같은 연주가 양쪽에 섞이지 않게 하기 위해서다.
- 데이터와 렌더링 결과는 레포에 올리지 않는다.

### 필요한 데이터

| 필요한 것 | 현황 |
|---|---|
| 속주/다양한 주법의 기타 DI | 공개 데이터 중 라이선스가 명확한 것을 못 찾음. 직접 녹음 예정 |
| 베이스 DI (슬랩, 핑거, 피크 등) | 직접 녹음 + 베이스 커뮤니티에 자발적 기여 요청 계획 |
| 서툰/날것의 연주, 잡음이 섞인 입력 | 직접 녹음, 합성 증강 |
| 같은 연주의 두 악기/두 톤 쌍 | 합성(픽업 변화 DSP) 후 실제 녹음으로 확인 |
| 이펙트가 걸린 소리(wet-only) | 이용 조건이 확실한 것만. 자기 복원 학습용 |

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

학습이 끝나면 `metrics.json`에 모델과 identity(dry를 그대로 내보내는 기준선)의 val/test 손실이 같이 기록된다. 마지막 평가는 전체 val/test를 클립 전체 길이로 한다. 모델이 identity보다 확실히 낮아야 의미가 있다.

```bash
# 체크포인트를 학습과 따로 전체 test로 평가하고, 손실이 큰 쌍을 확인
python scripts/evaluate.py --data-dir rendered --ckpt runs/exp1/best.pt --split test --worst 5

# (실험) 기타 DI에서 음높이/세기를 뽑아 모음 소리로 합성하고 음높이 보존도를 잰다
python scripts/voice_demo.py --input di.wav --out-dir demo --seconds 15 --vowel a
```

## 모델

- 입력: dry 모노 오디오 + 노브 조건 벡터(이펙트 켜짐 4개 + 노브 7개 = 11차원)
- 구조: dilated 비인과 TCN(WaveNet식 gated activation) + FiLM 조건 + skip 연결, 출력은 `입력 + 보정`이라 처음엔 항등
- 손실: 멀티 해상도 STFT 손실 + 시간영역 L1
- MVP 설정(`configs/effects_mvp.yaml`)은 reverb를 끈다. 현재 구조의 수용 영역이 약 0.2초라 리버브 꼬리를 배울 수 없기 때문이다.

## 구조

```
configs/effects.yaml        이펙트 체인과 노브 범위 (44.1 kHz, reverb 포함)
configs/effects_mvp.yaml    MVP 학습용 설정 (48 kHz, reverb 제외)
src/fxdata/                 랜덤 체인 샘플링, 렌더링, 메타 기록, 분할, 연주 특징/모음 합성 프로토타입(voice.py)
src/fxmodel/                조건 벡터, FiLM TCN, 손실, 데이터셋, 학습, 추론
scripts/                    download_data, render_dataset, train, evaluate, infer, voice_demo
notebooks/                  Colab 원클릭 학습
tests/                      합성 신호로 하는 검증
```

## 해야 할 일

- [x] Guitar-TECHS 파일 구조 확인 후 DI 폴더 지정
- [x] 학습 코드 (TCN + FiLM 조건, 멀티 해상도 STFT 손실), Colab 원클릭 노트북
- [x] GPU 학습으로 파이프라인 확인, 전체 test 평가 스크립트
- [x] 연주 특징 추출과 합성 정답 프로토타입
- [ ] 센 디스토션/퍼즈 체인, 오버샘플링(에일리어싱 방지)
- [ ] 입력 레벨/잡음/픽업 EQ 증강, 속주 변환(클립 가속, 겹치기)
- [ ] 악기 설정 일반화(음높이 범위, 이펙트 체인, 샘플레이트를 설정으로), 베이스용 설정
- [ ] 슬랩 같은 타악기성 소리를 위한 어택 특징
- [ ] 합성 픽업 변화 데이터와 참조 임베딩 조건 모델
- [ ] 베이스 커뮤니티에 올릴 데이터 기여 요청 글
- [ ] 오디오 인터페이스로 직접 녹음(속주, 서툰 연주, DI/리앰프 쌍)
- [ ] 리버브/딜레이처럼 긴 꼬리를 위한 더 큰 수용 영역 구조
- [ ] 코드 라이선스 결정 (아직 미정)

## 관련 연구

- [Style Transfer of Audio Effects with Differentiable Signal Processing](https://export.arxiv.org/abs/2207.08759): 참조 녹음에서 이펙트 파라미터를 예측해 입력에 적용
- [Demo of Zero-Shot Guitar Amplifier Modelling](https://arxiv.org/abs/2410.04702): 참조 녹음의 톤 임베딩으로 앰프 톤 모델링
- [Distortion Recovery](https://arxiv.org/html/2407.16639v1): 이펙트가 걸린 기타 소리에서 dry 복원
- [clean-to-effect 기타 변환 (DAFx26 데모)](https://www.dafx.de/paper-archive/2026/papers/DAFx26_demo_58.pdf): 클린 기타와 이펙트 라벨로 변환, 코드 공개
- [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler): DI와 리앰프 쌍으로 장비 캡처
