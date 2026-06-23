"""
src/team_catboost_test5.py
──────────────────────────
team_catboost 기반 실험 D: A + B + C 전체 종합

team_catboost 대비 변경 (test2+3+4 전부 합산):
  [A] tp 0(결측) → NaN, 7(해당없음) → NaN
      tp_notapplicable_cnt, tp_missing_cnt 파생 추가
  [B] QA 역문항 10개 리버스 코딩 (6 - x)
      대상: QaA, QdA, QeA, QfA, QgA, QiA, QkA, QnA, QqA, QrA
  [C] mach_score, q_response_std, q_extreme_ratio (QA raw 위에 추가)
      vocab_real, vocab_fake, vocab_score, vocab_accuracy (wr/wf raw 위에 추가)

나머지는 team_catboost와 완전 동일:
  - Optuna best params 그대로
  - delay_root10, familysize log1p
  - QA raw 20개, wr/wf raw 유지
"""

from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fe_v2 import load_common, save_npy, QE_COLS, CATEGORICAL_COLS
from tracker import try_log, show_leaderboard

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
OUTPUT_DIR  = BASE_DIR / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH      = DATA_DIR / "train.csv"
TEST_PATH       = DATA_DIR / "test_x.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

SEED     = 42
N_SPLITS = 5

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": SEED,
    "allow_writing_files": False,
    "verbose": 200,
    "early_stopping_rounds": 200,
    "iterations": 2500,
    "learning_rate": 0.011353066986375095,
    "depth": 7,
    "l2_leaf_reg": 3.427353847207771,
    "random_strength": 1.4620769648916006,
    "min_data_in_leaf": 10,
    "border_count": 132,
    "bagging_temperature": 1.2166796843617005,
}

Q_COLS    = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
TP_COLS   = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS   = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS   = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]
REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]
CAT_COLS  = [c for c in CATEGORICAL_COLS]


def prepare(train_df, test_df):
    from common_fe_v2 import _add_delay_root10

    X_train = train_df.copy()
    X_test  = test_df.copy()

    X_train = _add_delay_root10(X_train)
    X_test  = _add_delay_root10(X_test)

    for df in [X_train, X_test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # [A] tp 파생 피처 (NaN 변환 전)
    avail_tp = [c for c in TP_COLS if c in X_train.columns]
    for df in [X_train, X_test]:
        df["tp_notapplicable_cnt"] = (df[avail_tp] == 7).sum(axis=1)
        df["tp_missing_cnt"]       = (df[avail_tp] == 0).sum(axis=1)
    for col in avail_tp:
        X_train[col] = X_train[col].replace({0: np.nan, 7: np.nan})
        X_test[col]  = X_test[col].replace({0: np.nan, 7: np.nan})

    # [B] QA 역문항 리버스 코딩
    for col in REVERSE_Q:
        if col in X_train.columns:
            X_train[col] = 6 - X_train[col]
            X_test[col]  = 6 - X_test[col]

    # [C] mach 파생 피처 (QA raw 위에 추가, 역문항 보정 후)
    avail_q = [c for c in Q_COLS if c in X_train.columns]
    for df in [X_train, X_test]:
        df["mach_score"]      = df[avail_q].mean(axis=1)
        df["q_response_std"]  = df[avail_q].std(axis=1)
        df["q_extreme_ratio"] = (
            (df[avail_q] == 1) | (df[avail_q] == 5)
        ).sum(axis=1) / len(avail_q)

    # [C] vocab 파생 피처 (wr/wf raw 위에 추가)
    avail_wr = [c for c in WR_COLS if c in X_train.columns]
    avail_wf = [c for c in WF_COLS if c in X_train.columns]
    for df in [X_train, X_test]:
        df["vocab_real"]     = df[avail_wr].sum(axis=1)
        df["vocab_fake"]     = df[avail_wf].sum(axis=1)
        df["vocab_score"]    = df["vocab_real"] - df["vocab_fake"] * 2
        df["vocab_accuracy"] = df["vocab_real"] / (df["vocab_real"] + df["vocab_fake"] + 1e-6)

    for col in CAT_COLS:
        for df in [X_train, X_test]:
            if col in df.columns:
                df[col] = df[col].astype(str)

    drop_base      = QE_COLS + ["voted", "index"]
    drop_test_base = QE_COLS + ["index"]
    X_train = X_train.drop(columns=[c for c in drop_base      if c in X_train.columns])
    X_test  = X_test.drop(columns=[c for c in drop_test_base  if c in X_test.columns])

    cat_indices = [X_train.columns.get_loc(c) for c in CAT_COLS if c in X_train.columns]
    print(f"FE 완료 | 피처 {X_train.shape[1]}개 | cat {len(cat_indices)}개")
    return X_train, X_test, cat_indices


train_df, test_df, y, sample_sub = load_common(TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH)
X_train, X_test, cat_features    = prepare(train_df, test_df)
y_int = y.astype(int)

skf       = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_pred  = np.zeros(len(X_train), dtype=np.float64)
test_pred = np.zeros(len(X_test),  dtype=np.float64)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_int), 1):
    print(f"\n{'='*70}\nFold {fold}/{N_SPLITS}\n{'='*70}")
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        X_train.iloc[tr_idx], y_int[tr_idx],
        eval_set=(X_train.iloc[val_idx], y_int[val_idx]),
        cat_features=cat_features,
        use_best_model=True,
    )
    val_pred = model.predict_proba(X_train.iloc[val_idx])[:, 1]
    oof_pred[val_idx] = val_pred
    test_pred += model.predict_proba(X_test)[:, 1] / N_SPLITS
    print(f"Fold {fold} AUC: {roc_auc_score(y_int[val_idx], val_pred):.6f}")

oof_auc = roc_auc_score(y_int, oof_pred)
print(f"\n{'='*50}\nFINAL OOF AUC: {oof_auc:.6f}\n{'='*50}")

save_npy(oof_pred,  OUTPUT_DIR / "team_catboost_test5_oof.npy")
save_npy(test_pred, OUTPUT_DIR / "team_catboost_test5_test.npy")

try_log(
    "catboost", oof_auc, CATBOOST_PARAMS, {},
    notes=(
        "team_catboost_test5 [vs team_catboost: A+B+C 전체 종합] "
        "사용컬럼: QA_raw20+tp_raw10+wr_raw13+wf_raw3+인구통계8개 | "
        "파생: delay_root10+tp_notapplicable_cnt+tp_missing_cnt | "
        "       mach_score+q_response_std+q_extreme_ratio+vocab4개 | "
        "변경A: tp(0→NaN+7→NaN)+카운트2개 | "
        "변경B: 역문항10개→6-x보정 | "
        "변경C: mach_score+q_response_std+q_extreme_ratio+vocab4개(역문항보정후계산) | "
        "하이퍼파라미터: Optuna최적값그대로(lr=0.01135+depth=7+l2=3.43)"
    ),
)
show_leaderboard()
