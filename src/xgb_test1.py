"""
src/xgb_test1.py
────────────────
XGBoost 베이스라인 실험 #1

catboost_test1과 동일한 FE:
  - QA raw 20개 (역문항 10개 보정)
  - tp raw 10개 (0/7 → NaN)
  - wr raw 13개, wf raw 3개
  - mach_score, q_response_std, q_extreme_ratio
  - vocab_real, vocab_fake, vocab_score, vocab_accuracy
  - delay_root10 (QE 20개 압축)
  - familysize log1p
  - 인구통계 9개

XGBoost 전용 처리:
  - 범주형 → 수치형 label encoding (XGBoost는 category 직접 지원 안 함)
  - enable_categorical=True (XGBoost 1.6+에서 실험적 지원, fallback: label enc)
  - scale_pos_weight으로 클래스 불균형 처리
  - tree_method="hist" (LGBM 방식과 유사, 빠름)

실행:
    python -m src.xgb_test1

출력 (신기록 시에만):
    outputs/xgb_test1_submission.csv
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder

from src.tracker import try_log, show_leaderboard

# ================================================================
# 설정
# ================================================================
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test_x.csv"
OUT_PATH   = "outputs/xgb_test1_submission.csv"
SEED       = 42
N_SPLITS   = 5

Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
QE_COLS = [f"Q{chr(ord('a')+i)}E" for i in range(20)]
TP_COLS = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]
CAT_COLS  = ["age_group","education","engnat","gender",
             "hand","married","race","religion","urban"]

# ================================================================
# 하이퍼파라미터
# ================================================================
PARAMS = dict(
    objective             = "binary:logistic",
    eval_metric           = "auc",
    tree_method           = "hist",
    n_estimators          = 3000,
    learning_rate         = 0.03,
    max_depth             = 6,
    min_child_weight      = 5,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    reg_alpha             = 0.1,
    reg_lambda            = 5.0,
    scale_pos_weight      = 1.20665,
    random_state          = SEED,
    early_stopping_rounds = 200,   # 최신 XGBoost: 생성자에 전달
)
FIT_PARAMS = dict(
    verbose = 200,
)


# ================================================================
# 피처 엔지니어링 (catboost_test1과 동일)
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

    # 1. tp 파생 피처 (NaN 변환 전)
    avail_tp = [c for c in TP_COLS if c in train.columns]
    for df in [train, test]:
        df["tp_notapplicable_cnt"] = (df[avail_tp] == 7).sum(axis=1)
        df["tp_missing_cnt"]       = (df[avail_tp] == 0).sum(axis=1)
    for col in avail_tp:
        train[col] = train[col].replace({0: np.nan, 7: np.nan})
        test[col]  = test[col].replace({0: np.nan, 7: np.nan})

    # 2. 인구통계 무응답 → NaN
    for col in ["education","engnat","hand","married","urban"]:
        if col in train.columns:
            train[col] = train[col].replace(0, np.nan)
            test[col]  = test[col].replace(0, np.nan)

    # 3. 역문항 리버스 코딩
    for col in REVERSE_Q:
        if col in train.columns:
            train[col] = 6 - train[col]
            test[col]  = 6 - test[col]

    # 4. MACH 피처
    avail_q = [c for c in Q_COLS if c in train.columns]
    for df in [train, test]:
        df["mach_score"]      = df[avail_q].mean(axis=1)
        df["q_response_std"]  = df[avail_q].std(axis=1)
        df["q_extreme_ratio"] = (
            (df[avail_q] == 1) | (df[avail_q] == 5)
        ).sum(axis=1) / len(avail_q)

    # 5. 어휘력 피처
    avail_wr = [c for c in WR_COLS if c in train.columns]
    avail_wf = [c for c in WF_COLS if c in train.columns]
    for df in [train, test]:
        df["vocab_real"]     = df[avail_wr].sum(axis=1)
        df["vocab_fake"]     = df[avail_wf].sum(axis=1)
        df["vocab_score"]    = df["vocab_real"] - df["vocab_fake"] * 2
        df["vocab_accuracy"] = df["vocab_real"] / (df["vocab_real"] + df["vocab_fake"] + 1e-6)

    # 6. QE → delay_root10 (raw 20개 제거)
    avail_qe = [c for c in QE_COLS if c in train.columns]
    for df in [train, test]:
        df["delay_root10"] = np.power(df[avail_qe].sum(axis=1).clip(lower=0), 0.1)
    train = train.drop(columns=avail_qe)
    test  = test.drop(columns=avail_qe)

    # 7. familysize log1p
    for df in [train, test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # 8. 범주형 → OrdinalEncoder (XGBoost는 category dtype 불안정)
    #    train만으로 fit → test는 transform만 (누수 방지)
    #    train에 없는 test 카테고리는 -1로 인코딩
    existing_cats = [c for c in CAT_COLS if c in train.columns]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-1)
    enc.fit(train[existing_cats].astype(str))
    train[existing_cats] = enc.transform(train[existing_cats].astype(str))
    test[existing_cats]  = enc.transform(test[existing_cats].astype(str))

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
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train[tr_idx]
        y_val = y_train[val_idx]

        model = xgb.XGBClassifier(**PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            **FIT_PARAMS,
        )

        vp = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = vp
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

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

    print(f"피처: {X_train.shape[1]}개 | 샘플: {len(y_train)}")
    print(f"범주형: {[c for c in CAT_COLS if c in X_train.columns]}\n")

    oof_preds, test_preds, oof_auc = train_cv(X_train, y_train, X_test)

    os.makedirs("outputs", exist_ok=True)
    recorded = try_log(
        "xgb", oof_auc, PARAMS, FIT_PARAMS,
        notes=(
            "xgb_test1 [XGBoost 베이스라인, catboost_test1과 동일한 FE] "
            "사용컬럼: QA_raw20+tp_raw10+wr_raw13+wf_raw3+인구통계9개 | "
            "파생: mach_score+q_response_std+q_extreme_ratio+vocab4개+delay_root10 | "
            "전처리: 역문항10개보정+tp(0/7→NaN)+familysize_log1p+범주형OrdinalEncoder(-1=NaN) | "
            "제거: QE_raw20 | "
            "하이퍼파라미터: tree_method=hist+depth=6+lr=0.03+l2=5.0"
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
