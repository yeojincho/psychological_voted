"""
src/optuna_catboost_test6.py
────────────────────────────
team_catboost_test6 FE 기반 Optuna 하이퍼파라미터 튜닝

FE: team_catboost_test6과 완전 동일
  - tp 0/7 → NaN + tp_notapplicable_cnt, tp_missing_cnt
  - mach_score, q_response_std, q_extreme_ratio (역문항 보정 없이 raw 기반)
  - vocab_real, vocab_fake, vocab_score, vocab_accuracy
  - delay_root10, familysize log1p

탐색 파라미터:
  - iterations: 500~3000
  - learning_rate: 0.005~0.1 (log scale)
  - depth: 4~10
  - l2_leaf_reg: 1~10
  - random_strength: 0.1~5
  - bagging_temperature: 0~2
  - border_count: 32~255
  - min_data_in_leaf: 1~50

실행:
    python -m src.optuna_catboost_test6

결과:
    outputs/optuna_catboost_test6_results.csv  — 전체 trial 기록
    터미널에 best params 출력
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
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fe_v2 import load_common, QE_COLS, CATEGORICAL_COLS

# ================================================================
# 설정
# ================================================================
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
OUTPUT_DIR      = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH    = OUTPUT_DIR / "optuna_catboost_test6_results.csv"

TRAIN_PATH      = DATA_DIR / "train.csv"
TEST_PATH       = DATA_DIR / "test_x.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

SEED     = 42
N_SPLITS = 5
N_TRIALS = 100   # 시간이 부족하면 50으로 줄여도 됨

Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
TP_COLS = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]
CAT_COLS = [c for c in CATEGORICAL_COLS]


# ================================================================
# FE (team_catboost_test6과 동일)
# ================================================================
def prepare(train_df, test_df):
    from common_fe_v2 import _add_delay_root10

    X_train = train_df.copy()
    X_test  = test_df.copy()

    X_train = _add_delay_root10(X_train)
    X_test  = _add_delay_root10(X_test)

    for df in [X_train, X_test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    avail_tp = [c for c in TP_COLS if c in X_train.columns]
    for df in [X_train, X_test]:
        df["tp_notapplicable_cnt"] = (df[avail_tp] == 7).sum(axis=1)
        df["tp_missing_cnt"]       = (df[avail_tp] == 0).sum(axis=1)
    for col in avail_tp:
        X_train[col] = X_train[col].replace({0: np.nan, 7: np.nan})
        X_test[col]  = X_test[col].replace({0: np.nan, 7: np.nan})

    avail_q = [c for c in Q_COLS if c in X_train.columns]
    for df in [X_train, X_test]:
        df["mach_score"]      = df[avail_q].mean(axis=1)
        df["q_response_std"]  = df[avail_q].std(axis=1)
        df["q_extreme_ratio"] = (
            (df[avail_q] == 1) | (df[avail_q] == 5)
        ).sum(axis=1) / len(avail_q)

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

    drop_train = QE_COLS + ["voted", "index"]
    drop_test  = QE_COLS + ["index"]
    X_train = X_train.drop(columns=[c for c in drop_train if c in X_train.columns])
    X_test  = X_test.drop(columns=[c for c in drop_test  if c in X_test.columns])

    cat_indices = [X_train.columns.get_loc(c) for c in CAT_COLS if c in X_train.columns]
    return X_train, X_test, cat_indices


# ================================================================
# Objective
# ================================================================
def objective(trial: optuna.Trial) -> float:
    params = {
        "loss_function":      "Logloss",
        "eval_metric":        "AUC",
        "random_seed":        SEED,
        "allow_writing_files": False,
        "verbose":            0,
        # 탐색 파라미터
        "iterations":         trial.suggest_int("iterations", 500, 3000, step=100),
        "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "depth":              trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_strength":    trial.suggest_float("random_strength", 0.1, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count":       trial.suggest_int("border_count", 32, 255),
        "min_data_in_leaf":   trial.suggest_int("min_data_in_leaf", 1, 50),
    }

    skf      = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(X_train_global))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_global, y_global), 1):
        model = CatBoostClassifier(**params)
        model.fit(
            X_train_global.iloc[tr_idx], y_global[tr_idx],
            eval_set=(X_train_global.iloc[val_idx], y_global[val_idx]),
            cat_features=cat_features_global,
            use_best_model=True,
            early_stopping_rounds=100,
        )
        oof_pred[val_idx] = model.predict_proba(X_train_global.iloc[val_idx])[:, 1]

    return roc_auc_score(y_global, oof_pred)


# ================================================================
# Main
# ================================================================
def main():
    global X_train_global, y_global, cat_features_global

    train_df, test_df, y, _ = load_common(TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH)
    X_train_global, _, cat_features_global = prepare(train_df, test_df)
    y_global = y.astype(int)

    print(f"피처: {X_train_global.shape[1]}개 | 샘플: {len(y_global)}")
    print(f"Optuna 시작 — {N_TRIALS} trials\n")

    sampler = TPESampler(seed=SEED)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # 결과 저장
    df_results = study.trials_dataframe()
    df_results.to_csv(RESULTS_PATH, index=False)
    print(f"\n전체 trial 저장 → {RESULTS_PATH}")

    # Best params 출력
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best OOF AUC : {best.value:.6f}")
    print(f"Best Params  :")
    for k, v in best.params.items():
        print(f"  {k:<25}: {v}")
    print(f"{'='*60}")
    print("\n아래 파라미터를 team_catboost_test6 기반 학습 파일에 복붙하세요:")
    print("CATBOOST_PARAMS = {")
    print(f'    "loss_function": "Logloss",')
    print(f'    "eval_metric": "AUC",')
    print(f'    "random_seed": {SEED},')
    print(f'    "allow_writing_files": False,')
    print(f'    "verbose": 200,')
    print(f'    "early_stopping_rounds": 200,')
    for k, v in best.params.items():
        print(f'    "{k}": {v},')
    print("}")


if __name__ == "__main__":
    main()
