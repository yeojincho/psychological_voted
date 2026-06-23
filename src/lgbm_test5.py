"""
src/lgbm_test5.py
─────────────────
LightGBM 실험 #5 — 덜 중요한 피처 제거 (B)

팀 공통 FE 컬럼 정의 기준 (실제 데이터 컬럼명).

lgbm_test4 대비 제거:
  [제거] rank 변환 4개 (mach_rank, vocab_rank, delay_rank, familysize_rank)
       → bin/cat 피처가 있으면 중복 정보 가능성
  [제거] bin category 2개 (mach_bin, vocab_bin)
       → num_leaves=255 leaf-wise가 이미 비선형 분기 처리
  [제거] cross_cat 2개 (married_x_urban, gender_x_edu) → 3개로 축소
       → 핵심 3개(age_x_edu, gender_x_urban, race_x_religion)만 유지
  [제거] qe_cv, qe_max_ratio → qe_std만 유지
       → qe_std와 상관도 높은 파생 정리
  [제거] tp_range → tp_mean, tp_std만 유지

가설: 피처 수가 많아 colsample이 핵심 피처를 덜 선택할 수 있음 → 정제로 품질↑

실행:
    python -m src.lgbm_test5
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from src.tracker import try_log, show_leaderboard

# ================================================================
# 설정 — 팀 공통 FE 컬럼 정의 (실제 데이터 기준)
# ================================================================
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test_x.csv"
OUT_PATH   = "outputs/lgbm_test5_submission.csv"
SEED       = 42
N_SPLITS   = 5

QUESTION_KEYS = list("abcdefghijklmnopqrst")
QA_COLS = [f"Q{k}A" for k in QUESTION_KEYS]
QE_COLS = [f"Q{k}E" for k in QUESTION_KEYS]
TP_COLS = [f"tp0{i}" for i in range(1, 10)] + ["tp10"]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]

CAT_COLS = ["age_group","education","engnat","gender",
            "married","race","religion","urban"]

# cross_cat 핵심 3개만 (married_x_urban, gender_x_edu 제거)
CROSS_COLS   = ["age_x_edu", "gender_x_urban", "race_x_religion"]
ALL_CAT_COLS = CAT_COLS + CROSS_COLS

# ================================================================
# 하이퍼파라미터
# ================================================================
PARAMS = dict(
    objective         = "binary",
    metric            = "auc",
    n_estimators      = 4000,
    learning_rate     = 0.02,
    num_leaves        = 255,
    max_depth         = -1,
    min_child_samples = 10,
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.7,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    cat_smooth        = 10,
    path_smooth       = 0.1,
    scale_pos_weight  = 1.20665,
    random_state      = SEED,
    verbose           = -1,
)
FIT_PARAMS = dict(
    callbacks = [
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=200),
    ],
)


# ================================================================
# 피처 엔지니어링
# ================================================================
def preprocess(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train, test = train_raw.copy(), test_raw.copy()

    for df in [train, test]:
        if "index" in df.columns:
            df.drop(columns=["index"], inplace=True)
    for df in [train, test]:
        df.drop(columns=["hand"], errors="ignore", inplace=True)

    outlier_idx = train[train["familysize"] > 50].index
    if len(outlier_idx):
        print(f"familysize 이상치 제거: {len(outlier_idx)}행")
        train = train.drop(index=outlier_idx).reset_index(drop=True)

    # 1. tp 파생 (NaN 변환 전)
    avail_tp = [c for c in TP_COLS if c in train.columns]
    for df in [train, test]:
        df["tp_notapplicable_cnt"] = (df[avail_tp] == 7).sum(axis=1)
        df["tp_missing_cnt"]       = (df[avail_tp] == 0).sum(axis=1)
    for col in avail_tp:
        train[col] = train[col].replace({0: np.nan, 7: np.nan})
        test[col]  = test[col].replace({0: np.nan, 7: np.nan})
    for df in [train, test]:
        df["tp_mean"] = df[avail_tp].mean(axis=1)
        df["tp_std"]  = df[avail_tp].std(axis=1)
        # tp_range 제거

    # 2. 인구통계 무응답 → NaN + 서수 수치
    for col in ["education","engnat","married","urban"]:
        if col in train.columns:
            train[col] = train[col].replace(0, np.nan)
            test[col]  = test[col].replace(0, np.nan)
    edu_map          = {1:1, 2:2, 3:3, 4:4, 5:5}
    age_midpoint_map = {1:15, 2:21, 3:29, 4:39, 5:49, 6:59, 7:70}
    for df in [train, test]:
        df["edu_num"] = df["education"].map(edu_map)
        df["age_num"] = df["age_group"].map(age_midpoint_map)

    # 3. 역문항 리버스 코딩
    for col in REVERSE_Q:
        if col in train.columns:
            train[col] = 6 - train[col]
            test[col]  = 6 - test[col]

    # 4. MACH 피처
    avail_q = [c for c in QA_COLS if c in train.columns]
    for df in [train, test]:
        df["mach_score"]      = df[avail_q].mean(axis=1)
        df["q_response_std"]  = df[avail_q].std(axis=1)
        df["q_extreme_ratio"] = ((df[avail_q]==1)|(df[avail_q]==5)).sum(axis=1) / len(avail_q)
        df["mach_x_edu"]      = df["mach_score"] * df["edu_num"].fillna(3)

    # 5. 어휘력 피처
    avail_wr = [c for c in WR_COLS if c in train.columns]
    avail_wf = [c for c in WF_COLS if c in train.columns]
    for df in [train, test]:
        df["vocab_real"]     = df[avail_wr].sum(axis=1)
        df["vocab_fake"]     = df[avail_wf].sum(axis=1)
        df["vocab_score"]    = df["vocab_real"] - df["vocab_fake"] * 2
        df["vocab_accuracy"] = df["vocab_real"] / (df["vocab_real"] + df["vocab_fake"] + 1e-6)
        df["edu_x_vocab"]    = df["edu_num"].fillna(3) * df["vocab_score"]
        df["age_x_mach"]     = df["age_num"].fillna(39) * df["mach_score"]

    # 6. QE → delay + qe_std만 유지 (qe_cv, qe_max_ratio 제거)
    avail_qe = [c for c in QE_COLS if c in train.columns]
    for df in [train, test]:
        qe_vals = df[avail_qe].clip(lower=0)
        df["delay_root10"] = np.power(qe_vals.sum(axis=1), 0.1)
        df["qe_std"]       = qe_vals.std(axis=1)
    train = train.drop(columns=avail_qe)
    test  = test.drop(columns=avail_qe)

    # 7. familysize log1p
    for df in [train, test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # 8. 교차 category 핵심 3개
    for df in [train, test]:
        df["age_x_edu"]       = (df["age_group"].astype(str)+"_"+df["education"].astype(str)).astype("category")
        df["gender_x_urban"]  = (df["gender"].astype(str)+"_"+df["urban"].astype(str)).astype("category")
        df["race_x_religion"] = (df["race"].astype(str)+"_"+df["religion"].astype(str)).astype("category")

    # 9. 범주형 → category dtype
    for col in CAT_COLS:
        if col in train.columns:
            train[col] = train[col].astype("category")
            test[col]  = test[col].astype("category")

    return train, test


# ================================================================
# CV 학습
# ================================================================
def train_cv(X_train, y_train, X_test):
    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n── Fold {fold}/{N_SPLITS} ──────────────────────────")
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(
            X_train.iloc[tr_idx], y_train[tr_idx],
            eval_set=[(X_train.iloc[val_idx], y_train[val_idx])],
            **FIT_PARAMS,
        )
        vp = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        oof_preds[val_idx] = vp
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        print(f"Fold {fold} AUC: {roc_auc_score(y_train[val_idx], vp):.5f}")

    oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"\n[OOF AUC] {oof_auc:.5f}")
    return oof_preds, test_preds, oof_auc


# ================================================================
# Main
# ================================================================
def main():
    random.seed(SEED); np.random.seed(SEED)

    train_raw  = pd.read_csv(TRAIN_PATH)
    test_raw   = pd.read_csv(TEST_PATH)
    test_index = test_raw["index"] if "index" in test_raw.columns else pd.RangeIndex(len(test_raw))

    train_raw["voted"] = (train_raw["voted"] == 1).astype(int)
    train_proc, test_proc = preprocess(train_raw, test_raw)

    y_train = train_proc["voted"].values
    X_train = train_proc.drop(columns=["voted"])
    X_test  = test_proc.drop(columns=["voted"], errors="ignore")
    common  = [c for c in X_train.columns if c in X_test.columns]
    X_train, X_test = X_train[common], X_test[common]

    print(f"피처: {X_train.shape[1]}개 | 샘플: {len(y_train)}")
    print(f"제거: rank4개+bin2개+cross_cat2개+qe_cv+qe_max_ratio+tp_range\n")

    oof_preds, test_preds, oof_auc = train_cv(X_train, y_train, X_test)

    os.makedirs("outputs", exist_ok=True)
    recorded = try_log(
        "lgbm", oof_auc, PARAMS, FIT_PARAMS,
        notes=(
            "lgbm_test5 [test4에서 중복/저기여 피처 제거] "
            "사용컬럼: QA_raw20+tp_raw10+wr_raw13+wf_raw3+인구통계8개(hand제거) | "
            "파생: mach_score+q_response_std+q_extreme_ratio+vocab4개+delay_root10 | "
            "       qe_std+tp_mean+tp_std+mach_x_edu+edu_x_vocab+age_x_mach+edu_num+age_num | "
            "제거: rank4개(mach/vocab/delay/familysize_rank) | "
            "       bin2개(mach_bin/vocab_bin) | "
            "       cross_cat2개(married_x_urban/gender_x_edu)→3개유지 | "
            "       qe_cv+qe_max_ratio+tp_range"
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
