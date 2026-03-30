# 심리 성향 예측 해커톤

개인의 심리 성향 데이터를 기반으로 투표 참여 여부(voted)를 예측하는 머신러닝 프로젝트.

- 대회: [심리 성향 예측 AI 해커톤](https://dacon.io/competitions/official/236705/data)
- 평가 지표: ROC-AUC

---

## Project Structure

```
psychological_voted/
├── data/               # 데이터셋 (Git 제외)
│   ├── train.csv
│   ├── test_x.csv
│   └── sample_submission.csv
│
├── models/             # OOF / test 예측값 .npy (Git 제외)
│
├── outputs/            # submission csv (Git 제외)
│
├── notebooks/          # EDA 및 실험 노트북
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_baseline_model.ipynb
│
├── src/
│   ├── config.py       # 경로 · 컬럼 상수 · 하이퍼파라미터
│   ├── preprocess.py   # 전처리 파이프라인
│   ├── train.py        # 모델 학습 + CV + submission 저장
│   └── inference.py    # 저장된 .npy로 앙상블 submission 생성
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 실행 방법

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

[데이콘](https://dacon.io/competitions/official/236705/data)에서 다운로드 후 `data/` 에 배치.

```
data/
├── train.csv
├── test_x.csv
└── sample_submission.csv
```

### 3. 학습

```bash
python -m src.train --model lgbm       # LightGBM
python -m src.train --model xgb        # XGBoost
python -m src.train --model catboost   # CatBoost
python -m src.train --model both       # 3개 학습 후 앙상블
```

학습 완료 시 `models/` 에 `{model}_oof.npy`, `{model}_test.npy` 저장.

### 4. 앙상블 submission 생성

```bash
python -m src.inference                          # models/ 의 전체 모델 앙상블
python -m src.inference --models lgbm catboost  # 지정 모델만 앙상블
```

결과는 `outputs/submission_ensemble_<timestamp>.csv` 로 저장.

---

## Feature Engineering

| 피처 | 설명 |
|------|------|
| `mach_score` | MACH-IV 20문항 평균 (역문항 10개 보정 후) |
| `q_response_std` | 20문항 응답 표준편차 |
| `q_extreme_ratio` | 극단 응답(1 또는 5) 비율 |
| `vocab_real` | 실존 단어 인식 수 (wr_01~13 합계) |
| `vocab_fake` | 허구 단어에 속은 수 (wf_01~03 합계) |
| `vocab_score` | vocab_real - vocab_fake × 2 |
| `vocab_accuracy` | real / (real + fake + ε) |
| `delay_root10` | 전체 응답시간 합의 0.1승 (QE raw 20개 대체) |
| `tp_notapplicable_cnt` | tp 해당없음(7) 응답 수 |
| `tp_missing_cnt` | tp 무응답(0) 수 |
| `familysize` | log1p 변환 (이상치 >50 제거) |

**역문항 10개** (Spearman 상관분석 기반): QaA, QdA, QeA, QfA, QgA, QiA, QkA, QnA, QqA, QrA

### 실험 결론

| 실험 | 결과 | 판단 |
|------|------|------|
| 역문항 4개 → 10개 보정 | mach_score 정확도 향상 | ✅ 적용 |
| QE → delay_root10 단일 피처 | 5개 요약통계보다 효과적 | ✅ 적용 |
| familysize log1p + 이상치 제거 | 분포 정규화 | ✅ 적용 |
| tp 파생 피처 생성 | 결측 패턴 정보 보존 | ✅ 적용 |
| 상관계수 기반 피처 제거 | OOF AUC -0.019049 | ❌ 금지 |
| QA PCA 치환 | item-level 패턴 소실 | ❌ 금지 |
| Big5 추가 FE | OOF AUC -0.000331 | ❌ 금지 |

---

## 데이터 변수 설명

### Q_A (a~t) — MACH-IV 마키아벨리즘 문항 (1=동의 안함 ~ 5=동의함)

| 문항 | 내용 |
|------|------|
| Qa | 비밀 |
| Qb | 대부분의 범죄자와 일반인의 가장 큰 차이점은 범죄자들은 잡힐 만큼 어리석다는 것 |
| Qc | 다른 사람을 완전히 신뢰하는 사람은 문제를 자초하는 것 |
| Qd | 비밀 |
| Qe | PT 바넘이 "세상엔 매분마다 속기 쉬운 사람이 태어난다"고 말한 건 틀린 말 |
| Qf | 다른 사람에게 거짓말하는 것은 절대 용납될 수 없음 |
| Qg | 비밀 |
| Qh | 대부분의 사람들은 재산 손실보다 부모님의 죽음을 더 쉽게 잊음 |
| Qi | 비밀 |
| Qj | 모든 사람에게는 악의적인 면이 있고, 기회가 주어지면 드러날 것 |
| Qk | 중요한 인물이 되려 부정직하게 구는 것보다 겸손하고 정직한 것이 더 나음 |
| Ql | 비밀 |
| Qm | 요령을 부리지 않고서는 성공하기 어려움 |
| Qo | 비밀 |
| Qp | 사람들을 다루는 가장 좋은 방법은 그들이 듣고 싶어하는 말을 해주는 것 |
| Qq | 대부분의 사람들은 기본적으로 선하고 친절함 |
| Qr | 도덕적으로 옳다고 확신할 때만 행동해야 함 |
| Qs | 중요한 사람들에게 아첨하는 것은 현명한 일 |
| Qt | 비밀 |

### Q_E (a~t) — 각 문항 응답에 걸린 시간 (ms)

### tp01~tp10 — Big Five 성격 문항 (1=강하게 반대 ~ 7=강하게 동의, 0=무응답)

| 문항 | 내용 |
|------|------|
| tp01 | 외향적이고 열정적 |
| tp02 | 비판적이고 다투기를 좋아함 |
| tp03 | 믿음직스럽고 자기 절제력이 뛰어남 |
| tp04 | 불안해하고 쉽게 화를 냄 |
| tp05 | 새로운 경험에 열려있고 복잡한 성격 |
| tp06 | 조용하고 한적한 곳을 좋아함 |
| tp07 | 공감 능력이 뛰어나고 따뜻함 |
| tp08 | 정리정돈이 안 되어 있고 부주의함 |
| tp09 | 차분하고 감정적으로 안정적 |
| tp10 | 틀에 박힌, 창의성이 부족함 |

### 인구통계 변수

| 변수 | 설명 |
|------|------|
| `age_group` | 연령대 |
| `education` | 1=고졸미만 2=고졸 3=학사 4=대학원, 0=무응답 |
| `engnat` | 모국어 영어 여부 (1=Yes 2=No, 0=무응답) |
| `familysize` | 형제자매 수 |
| `gender` | Female / Male |
| `hand` | 1=오른손 2=왼손 3=양손, 0=무응답 |
| `married` | 1=미혼 2=기혼 3=이전기혼, 0=무응답 |
| `race` | Asian / Arab / Black / White / Other 등 |
| `religion` | Agnostic / Atheist / Buddhist / Christian 등 |
| `urban` | 1=시골 2=교외 3=도시, 0=무응답 |
| `wr_01~13` | 실존 단어 인식 여부 (1=예 0=아니오) |
| `wf_01~03` | 허구 단어 인식 여부 (1=예 0=아니오) |

### voted (타겟)

- 1 = 투표함
- 2 = 투표 안함 → 전처리 후 0으로 변환

---

## 참고

- [심리 성향 예측 AI 해커톤](https://dacon.io/competitions/official/236705/data)
- [데이터 변수 설명](https://www.dacon.io/competitions/official/235647/talkboard/401534)
- [openpsychometrics.org](https://openpsychometrics.org/)
