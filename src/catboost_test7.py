"""
src/catboost_test7.py
─────────────────────
CatBoost 실험 #7 — 레퍼런스(cat_b_delay_root10_only) 충실 재현

전략: 팀원 레퍼런스 코드를 프로젝트 구조에 맞게 이식.
      파생 피처 없이 raw 최대 보존 + delay_root10만 추가.

레퍼런스 대비 차이점:
  - voted 인코딩 수정: (voted==1).astype(int)  [레퍼런스는 voted-1로 반전됨]
  - scale_pos_weight=1.20665 추가 (클래스 불균형 보정)
  - tracker 연동 (신기록 시에만 submission 저장)
  - allow_writing_files=False 추가

레퍼런스와 동일하게 유지하는 것:
  - tp 0/7을 NaN으로 변환하지 않음 (raw 그대로)
  - QA 역문항 리버스 코딩 없음
  - mach_score, vocab, Big5 등 파생 피처 일절 없음
  - hand 컬럼 CAT_COLS에 포함하지 않음
  - familysize log1p + 범주형 문자열 변환만 적용

실행:
    python -m src.catboost_test7

출력 (신기록 시에만):
    outputs/catboost_test7_submission.csv
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
OUT_PATH   = "outputs/catboost_test7_submission.csv"
SEED       = 42
N_SPLITS   = 5

QUESTION_KEYS = list("abcdefghijklmnopqrst")
QE_COLS = [f"Q{k}E" for k in QUESTION_KEYS]

# 레퍼런스와 동일 — hand 미포함
CAT_COLS = [
    "age_group", "education", "engnat", "gender",
    "married", "race", "religion", "urban",
]

# ================================================================
# 하이퍼파라미터 (레퍼런스와 동일)
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
# 피처 엔지니어링 (레퍼런스 충실 재현)
# ================================================================
def preprocess(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train, test = train_raw.copy(), test_raw.copy()

    for df in [train, test]:
        if "index" in df.columns:
            df.drop(columns=["index"], inplace=True)

    # familysize 이상치 제거 (train만)
    outlier_idx = train[train["familysize"] > 50].index
    if len(outlier_idx):
        print(f"familysize 이상치 제거: {len(outlier_idx)}행")
        train = train.drop(index=outlier_idx).reset_index(drop=True)

    # ── delay_root10 (QE raw 제거 전) ───────────────────────
    for df in [train, test]:
        delay_sum = df[QE_COLS].clip(lower=0).sum(axis=1)
        df["delay_root10"] = np.power(delay_sum, 0.1)

    train = train.drop(columns=QE_COLS, errors="ignore")
    test  = test.drop(columns=QE_COLS, errors="ignore")

    # ── familysize log1p ──────────────────────────────────
    for df in [train, test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # ── 범주형 → 문자열 (CatBoost용) ─────────────────────
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

    # voted 인코딩 수정 (레퍼런스의 voted-1은 방향 반전 버그)
    train_raw["voted"] = (train_raw["voted"] == 1).astype(int)

    train_proc, test_proc = preprocess(train_raw, test_raw)

    y_train = train_proc["voted"].values
    X_train = train_proc.drop(columns=["voted"])
    X_test  = test_proc.drop(columns=["voted"], errors="ignore")

    common  = [c for c in X_train.columns if c in X_test.columns]
    X_train, X_test = X_train[common], X_test[common]

    cat_idx = [i for i, c in enumerate(X_train.columns) if c in CAT_COLS]

    print(f"피처: {X_train.shape[1]}개 | 범주형: {len(cat_idx)}개 | 샘플: {len(y_train)}")
    print(f"전략: 레퍼런스 충실 재현 — tp/QA raw 그대로, 파생피처 없음, delay_root10만\n")

    oof_preds, test_preds, oof_auc = train_cv(X_train, y_train, X_test, cat_idx)

    os.makedirs("outputs", exist_ok=True)
    recorded = try_log(
        "catboost", oof_auc, PARAMS, FIT_PARAMS,
        notes=(
            "test7 [vs test1: 팀원레퍼런스재현, 전처리최소화] "
            "사용컬럼: QA_raw20+tp_raw10+wr_raw13+wf_raw3+인구통계8개(hand제외) | "
            "파생: delay_root10+familysize_log1p만 | "
            "변경: tp(0/7→NaN변환없음, test1과다름)+QA역문항코딩없음+mach_score등파생피처없음 | "
            "제거: QE_raw20, hand"
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
