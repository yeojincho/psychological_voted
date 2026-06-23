"""
src/team_catboost_test4.py
──────────────────────────
team_catboost 기반 실험 C: mach_score + vocab 파생 피처 추가

team_catboost 대비 변경:
  - mach_score (QA raw 평균)
  - q_response_std (QA 응답 분산)
  - q_extreme_ratio (극단 응답 비율)
  - vocab_real, vocab_fake, vocab_score, vocab_accuracy (어휘력 4개)
  → raw 위에 요약 신호 병행 — CatBoost가 raw 패턴 + 요약 신호 둘 다 활용

나머지는 team_catboost와 완전 동일:
  - Optuna best params 그대로
  - delay_root10, familysize log1p
  - QA raw 20개 유지 (역문항 보정 없음)
  - tp 처리 없음 (0/7 그대로 유지)
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

Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]
CAT_COLS = [c for c in CATEGORICAL_COLS]


def prepare(train_df, test_df):
    from common_fe_v2 import _add_delay_root10

    X_train = train_df.copy()
    X_test  = test_df.copy()

    X_train = _add_delay_root10(X_train)
    X_test  = _add_delay_root10(X_test)

    for df in [X_train, X_test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # [변경 C] mach 파생 피처 (QA raw 위에 추가)
    avail_q = [c for c in Q_COLS if c in X_train.columns]
    for df in [X_train, X_test]:
        df["mach_score"]      = df[avail_q].mean(axis=1)
        df["q_response_std"]  = df[avail_q].std(axis=1)
        df["q_extreme_ratio"] = (
            (df[avail_q] == 1) | (df[avail_q] == 5)
        ).sum(axis=1) / len(avail_q)

    # [변경 C] vocab 파생 피처 (wr/wf raw 위에 추가)
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

save_npy(oof_pred,  OUTPUT_DIR / "team_catboost_test4_oof.npy")
save_npy(test_pred, OUTPUT_DIR / "team_catboost_test4_test.npy")

try_log(
    "catboost", oof_auc, CATBOOST_PARAMS, {},
    notes=(
        "team_catboost_test4 [vs team_catboost: mach_score+vocab 파생 추가] "
        "사용컬럼: QA_raw20+tp_raw10+wr_raw13+wf_raw3+인구통계8개 | "
        "파생: delay_root10+mach_score+q_response_std+q_extreme_ratio+vocab4개 | "
        "변경: raw 위에 요약파생 7개 병행추가(역문항보정없음) | "
        "하이퍼파라미터: Optuna최적값그대로(lr=0.01135+depth=7+l2=3.43)"
    ),
)
show_leaderboard()
