"""
src/catboost_test3.py
─────────────────────
CatBoost 실험 #3

test1/2 대비 변경 사항:
  - QA raw 20개 유지 (test2는 제거) + MACH 파생 피처 추가
  - QE: log1p 개별 변환 후 mean/std/max 요약 (delay_root10 미사용)
  - tp → Big5 5개 점수 직접 계산 (IPIP 역문항 보정 포함)
  - Big5 × mach_score 상호작용 피처 (외향성×마키아벨리즘 등)
  - wr/wf raw 유지 + vocab 파생 피처 추가
  - 인구통계 전부 유지 (engnat, hand 포함)

Big5 스코어링 (IPIP 10문항 기준):
  extraversion  = tp01 + (6 - tp06)
  agreeableness = tp07 + (6 - tp02)
  conscientiousness = tp03 + (6 - tp08)
  neuroticism   = (6 - tp04) + (6 - tp09)
  openness      = tp05 + (6 - tp10)

실행:
    python -m src.catboost_test3

출력 (신기록 시에만):
    outputs/catboost_test3_submission.csv
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
OUT_PATH   = "outputs/catboost_test3_submission.csv"
SEED       = 42
N_SPLITS   = 5

Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
QE_COLS = [f"Q{chr(ord('a')+i)}E" for i in range(20)]
TP_COLS = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]

# MACH 세부 차원
MACH_TACTICS  = ["QbA","QjA","QmA","QpA","QsA"]
MACH_MORALITY = ["QeA","QfA","QkA","QqA","QrA"]
MACH_CYNICISM = ["QcA","QhA","QaA","QdA","QgA","QiA"]

# 범주형 (engnat, hand 포함 — test2에서 제거했던 것 복원)
CAT_COLS = ["age_group","education","engnat","gender",
            "hand","married","race","religion","urban"]

# ================================================================
# 하이퍼파라미터
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

    # familysize 이상치 제거 (train만)
    outlier_idx = train[train["familysize"] > 50].index
    if len(outlier_idx):
        print(f"familysize 이상치 제거: {len(outlier_idx)}행")
        train = train.drop(index=outlier_idx).reset_index(drop=True)

    # ── 1. tp 파생 피처 (NaN 변환 전) ───────────────────────
    avail_tp = [c for c in TP_COLS if c in train.columns]
    for df in [train, test]:
        df["tp_notapplicable_cnt"] = (df[avail_tp] == 7).sum(axis=1)
        df["tp_missing_cnt"]       = (df[avail_tp] == 0).sum(axis=1)
    for col in avail_tp:
        train[col] = train[col].replace({0: np.nan, 7: np.nan})
        test[col]  = test[col].replace({0: np.nan, 7: np.nan})

    # ── 2. Big5 점수 (IPIP 10문항 역문항 보정) ──────────────
    # tp에 NaN이 있으므로 fillna(median)으로 보정 후 계산
    for df in [train, test]:
        tp = {c: df[c].fillna(df[c].median()) for c in avail_tp}
        df["big5_extraversion"]      = tp["tp01"] + (6 - tp["tp06"])
        df["big5_agreeableness"]     = tp["tp07"] + (6 - tp["tp02"])
        df["big5_conscientiousness"] = tp["tp03"] + (6 - tp["tp08"])
        df["big5_neuroticism"]       = (6 - tp["tp04"]) + (6 - tp["tp09"])
        df["big5_openness"]          = tp["tp05"] + (6 - tp["tp10"])

    # ── 3. 인구통계 무응답 → NaN ─────────────────────────────
    for col in ["education","engnat","hand","married","urban"]:
        if col in train.columns:
            train[col] = train[col].replace(0, np.nan)
            test[col]  = test[col].replace(0, np.nan)

    # ── 4. 역문항 리버스 코딩 ────────────────────────────────
    for col in REVERSE_Q:
        if col in train.columns:
            train[col] = 6 - train[col]
            test[col]  = 6 - test[col]

    # ── 5. MACH 파생 피처 (QA raw 유지) ─────────────────────
    avail_q = [c for c in Q_COLS if c in train.columns]
    for df in [train, test]:
        df["mach_score"]      = df[avail_q].mean(axis=1)
        df["q_response_std"]  = df[avail_q].std(axis=1)
        df["q_extreme_ratio"] = (
            (df[avail_q] == 1) | (df[avail_q] == 5)
        ).sum(axis=1) / len(avail_q)
        df["mach_high_ratio"] = (df[avail_q] >= 4).sum(axis=1) / len(avail_q)

        # 세부 차원
        for name, cols in [("mach_tactics",  MACH_TACTICS),
                            ("mach_morality", MACH_MORALITY),
                            ("mach_cynicism", MACH_CYNICISM)]:
            c = [x for x in cols if x in df.columns]
            if c: df[name] = df[c].mean(axis=1)

    # ── 6. Big5 × mach_score 상호작용 ───────────────────────
    big5_cols = ["big5_extraversion","big5_agreeableness",
                 "big5_conscientiousness","big5_neuroticism","big5_openness"]
    for df in [train, test]:
        for b in big5_cols:
            df[f"{b}_x_mach"] = df[b] * df["mach_score"]

    # ── 7. 어휘력 피처 (raw 유지) ────────────────────────────
    avail_wr = [c for c in WR_COLS if c in train.columns]
    avail_wf = [c for c in WF_COLS if c in train.columns]
    for df in [train, test]:
        df["vocab_real"]     = df[avail_wr].sum(axis=1)
        df["vocab_fake"]     = df[avail_wf].sum(axis=1)
        df["vocab_score"]    = df["vocab_real"] - df["vocab_fake"] * 2
        df["vocab_accuracy"] = df["vocab_real"] / (df["vocab_real"] + df["vocab_fake"] + 1e-6)

    # ── 8. QE: log1p 개별 변환 후 mean/std/max 요약 ─────────
    # delay_root10(sum 기반) 대신 개별 항목 log 변환 후 통계
    avail_qe = [c for c in QE_COLS if c in train.columns]
    for df in [train, test]:
        qe_log = np.log1p(df[avail_qe].clip(lower=0))
        df["qe_log_mean"] = qe_log.mean(axis=1)
        df["qe_log_std"]  = qe_log.std(axis=1)
        df["qe_log_max"]  = qe_log.max(axis=1)
        df["qe_log_min"]  = qe_log.min(axis=1)
        # 응답 속도 변동계수
        df["time_cv"] = df["qe_log_std"] / (df["qe_log_mean"] + 1e-6)
    train = train.drop(columns=avail_qe)
    test  = test.drop(columns=avail_qe)

    # ── 9. familysize log1p ──────────────────────────────────
    for df in [train, test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # ── 10. 범주형 → 문자열 (CatBoost용) ────────────────────
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
    print(f"피처 목록: {list(X_train.columns)}\n")

    oof_preds, test_preds, oof_auc = train_cv(X_train, y_train, X_test, cat_idx)

    # 신기록일 때만 submission 저장 + 실험 기록
    os.makedirs("outputs", exist_ok=True)
    recorded = try_log(
        "catboost", oof_auc, PARAMS, FIT_PARAMS,
        notes=(
            "test3 [vs test1: tp→Big5압축, QE방식변경] "
            "사용컬럼: QA_raw20+wr_raw13+wf_raw3+인구통계9개 | "
            "파생: mach_score+mach_tactics+mach_morality+mach_cynicism+mach_high_ratio+q_response_std+q_extreme_ratio+vocab4개+big5_E/A/C/N/O+big5×mach교호작용5개+qe_log_mean/std/max/min+time_cv | "
            "변경: tp_raw10→Big5_5개로압축, QE→log1p개별변환후통계(test1의delay_root10대신) | "
            "제거: tp_raw10, QE_raw20"
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
