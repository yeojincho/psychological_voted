"""
src/catboost_test5.py
─────────────────────
CatBoost 실험 #5 — 노이즈 변수 제거

test1이 최고 성능이지만, 해커톤 특성상 출제자가
의도적으로 노이즈 변수를 삽입했을 가능성이 있음.

제거 대상 및 근거:
  1. hand (손잡이)
       → 투표 행동과 심리·인구통계적 연관성 없음
       → "흥미로운 변수처럼 보이는 noise" 패턴
  2. engnat (영어 모국어 여부)
       → 온라인 심리 설문 메타 정보 (접속 국가/언어 proxy)
       → 데이터 수집 아티팩트일 가능성
  3. wf_01~03 raw (가짜 단어 개별 항목)
       → vocab_fake 집계 피처로 이미 요약됨
       → 개별 raw 항목은 집계에 노이즈 추가
  4. q_response_std (응답 표준편차)
       → 응답 스타일 편향 (tendency bias) — 측정 아티팩트
  5. q_extreme_ratio (극단 응답 비율)
       → 반응 스타일 변수, 투표 예측력보다 개인 응답 습관 반영
  6. delay_root10 (QE 응답 시간 합산)
       → 인터넷 속도·기기 성능 영향, 심리 신호보다 환경 노이즈

유지 항목 (test1의 핵심 신호원):
  - QA raw 20개 (역문항 보정) — MACH 개별 문항 패턴
  - tp01~10 raw — Big5 개별 문항 패턴
  - wr_01~13 raw — 실제 단어 인식 개별 패턴
  - mach_score — MACH 전체 점수 (집계 신호)
  - vocab_real/fake/score/accuracy — 어휘력 집계
  - tp_notapplicable_cnt, tp_missing_cnt — 설문 참여 성실도
  - CAT_COLS 7개: age_group/education/gender/married/race/religion/urban

실행:
    python -m src.catboost_test5

출력 (신기록 시에만):
    outputs/catboost_test5_submission.csv
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

from src.tracker import try_log, show_leaderboard

# ================================================================
# 설정
# ================================================================
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test_x.csv"
OUT_PATH   = "outputs/catboost_test5_submission.csv"
SEED       = 42
N_SPLITS   = 5

Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
QE_COLS = [f"Q{chr(ord('a')+i)}E" for i in range(20)]
TP_COLS = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]

# ※ hand, engnat 제거 (test1 대비 핵심 변경)
CAT_COLS = ["age_group", "education", "gender", "married", "race", "religion", "urban"]

# 제거할 노이즈 컬럼
NOISE_COLS = ["hand", "engnat"]

# ================================================================
# 하이퍼파라미터 (test1과 동일)
# ================================================================
PARAMS = dict(
    objective        = "Logloss",
    eval_metric      = "AUC",
    iterations       = 3000,
    learning_rate    = 0.03,
    depth            = 6,
    l2_leaf_reg      = 5,
    bootstrap_type   = "Bernoulli",
    subsample        = 0.8,
    scale_pos_weight = 1.20665,
    random_seed      = SEED,
    allow_writing_files = False,
)
FIT_PARAMS = dict(
    early_stopping_rounds = 200,
    verbose               = 200,
)


# ================================================================
# 피처 엔지니어링
# ================================================================
def preprocess(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train, test = train_raw.copy(), test_raw.copy()

    for df in [train, test]:
        if "index" in df.columns:
            df.drop(columns=["index"], inplace=True)

    # ── 노이즈 컬럼 제거 ────────────────────────────────────────
    for df in [train, test]:
        df.drop(columns=[c for c in NOISE_COLS if c in df.columns], inplace=True)

    # familysize 이상치 제거 (train만)
    outlier_idx = train[train["familysize"] > 50].index
    if len(outlier_idx):
        print(f"familysize 이상치 제거: {len(outlier_idx)}행")
        train = train.drop(index=outlier_idx).reset_index(drop=True)

    # ── 1. tp 파생 피처 (NaN 변환 전) ───────────────────────────
    avail_tp = [c for c in TP_COLS if c in train.columns]
    for df in [train, test]:
        df["tp_notapplicable_cnt"] = (df[avail_tp] == 7).sum(axis=1)
        df["tp_missing_cnt"]       = (df[avail_tp] == 0).sum(axis=1)
    for col in avail_tp:
        train[col] = train[col].replace({0: np.nan, 7: np.nan})
        test[col]  = test[col].replace({0: np.nan, 7: np.nan})

    # ── 2. 인구통계 무응답 → NaN ──────────────────────────────
    # hand/engnat 제거됐으므로 남은 항목만 처리
    for col in ["education", "married", "urban"]:
        if col in train.columns:
            train[col] = train[col].replace(0, np.nan)
            test[col]  = test[col].replace(0, np.nan)

    # ── 3. 역문항 리버스 코딩 ─────────────────────────────────
    for col in REVERSE_Q:
        if col in train.columns:
            train[col] = 6 - train[col]
            test[col]  = 6 - test[col]

    # ── 4. MACH 집계 피처 (QA raw 유지) ──────────────────────
    #    q_response_std, q_extreme_ratio 제거 (응답 스타일 노이즈)
    avail_q = [c for c in Q_COLS if c in train.columns]
    for df in [train, test]:
        df["mach_score"] = df[avail_q].mean(axis=1)
        # ※ q_response_std, q_extreme_ratio 의도적으로 미생성

    # ── 5. 어휘력 피처 ───────────────────────────────────────
    #    wf raw 제거, wr raw 유지
    avail_wr = [c for c in WR_COLS if c in train.columns]
    avail_wf = [c for c in WF_COLS if c in train.columns]
    for df in [train, test]:
        df["vocab_real"]     = df[avail_wr].sum(axis=1)
        df["vocab_fake"]     = df[avail_wf].sum(axis=1)
        df["vocab_score"]    = df["vocab_real"] - df["vocab_fake"] * 2
        df["vocab_accuracy"] = df["vocab_real"] / (df["vocab_real"] + df["vocab_fake"] + 1e-6)

    # wf raw 제거 (집계로 요약됨, 개별 항목은 노이즈)
    train = train.drop(columns=[c for c in avail_wf if c in train.columns])
    test  = test.drop(columns=[c for c in avail_wf if c in test.columns])

    # ── 6. QE 전체 제거 (응답 시간 = 환경 노이즈) ─────────────
    avail_qe = [c for c in QE_COLS if c in train.columns]
    train = train.drop(columns=avail_qe)
    test  = test.drop(columns=avail_qe)

    # ── 7. familysize log1p ───────────────────────────────────
    for df in [train, test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # ── 8. 범주형 → 문자열 (CatBoost용) ─────────────────────
    for col in CAT_COLS:
        if col in train.columns:
            train[col] = train[col].astype(str)
            test[col]  = test[col].astype(str)

    return train, test


# ================================================================
# CV 학습
# ================================================================
def train_cv(X_train, y_train, X_test, cat_idx):
    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n── Fold {fold}/{N_SPLITS} ──────────────────────────")
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train[tr_idx]
        y_val = y_train[val_idx]

        train_pool = Pool(X_tr,  label=y_tr,  cat_features=cat_idx)
        val_pool   = Pool(X_val, label=y_val, cat_features=cat_idx)
        test_pool  = Pool(X_test, cat_features=cat_idx)

        model = CatBoostClassifier(**PARAMS, **FIT_PARAMS)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        vp = model.predict_proba(val_pool)[:, 1]
        oof_preds[val_idx] = vp
        test_preds += model.predict_proba(test_pool)[:, 1] / N_SPLITS

        print(f"Fold {fold} AUC: {roc_auc_score(y_val, vp):.5f}")

    oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"\n[OOF AUC] {oof_auc:.5f}")
    return oof_preds, test_preds, oof_auc


# ================================================================
# Main
# ================================================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw  = pd.read_csv(TEST_PATH)
    test_index = test_raw["index"] if "index" in test_raw.columns \
                 else pd.RangeIndex(len(test_raw))

    train_raw["voted"] = (train_raw["voted"] == 1).astype(int)
    train_proc, test_proc = preprocess(train_raw, test_raw)

    y_train = train_proc["voted"].values
    X_train = train_proc.drop(columns=["voted"])
    X_test  = test_proc.drop(columns=["voted"], errors="ignore")

    common  = [c for c in X_train.columns if c in X_test.columns]
    X_train, X_test = X_train[common], X_test[common]

    cat_idx = [i for i, c in enumerate(X_train.columns) if c in CAT_COLS]

    print(f"피처: {X_train.shape[1]}개 | 범주형: {len(cat_idx)}개 | 샘플: {len(y_train)}")
    print(f"제거된 노이즈: hand, engnat, wf raw 3개, QE 20개(delay_root10 포함), "
          f"q_response_std, q_extreme_ratio\n")

    oof_preds, test_preds, oof_auc = train_cv(X_train, y_train, X_test, cat_idx)

    os.makedirs("outputs", exist_ok=True)
    recorded = try_log(
        "catboost", oof_auc, PARAMS, FIT_PARAMS,
        notes=(
            "test5 [vs test1: 노이즈 의심 컬럼 제거 실험] "
            "사용컬럼: QA_raw20+tp_raw10+wr_raw13+인구통계7개 | "
            "파생: mach_score만(q_response_std/q_extreme_ratio제거)+vocab4개 | "
            "제거: hand(도메인무관)+engnat(메타정보)+wf_raw3(vocab_fake집계로충분)+QE_raw20전체(환경노이즈)+q_response_std+q_extreme_ratio(응답스타일편향)"
        ),
    )
    if recorded:
        pd.DataFrame({"index": test_index, "voted": test_preds}).to_csv(OUT_PATH, index=False)
        print(f"submission → {OUT_PATH}")
    else:
        print("[tracker] 신기록 아님 → submission 저장 생략")

    show_leaderboard()


if __name__ == "__main__":
    main()
