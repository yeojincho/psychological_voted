"""
train_catboost.py — CatBoost 학습 코드 (boosting.py 정확 재현)
═══════════════════════════════════════════════════════════════
common_fe_v2.py를 import하여 FE 수행
Optuna best params + 학습 로직은 boosting.py와 100% 동일

현재 최고 기록: OOF AUC ≈ 0.772888

핵심 설정 (변경 시 주의):
  - Optuna best params (lr=0.01135, depth=7, l2=3.43, iter=2500)
  - cat_features: age_group, education, engnat, gender, married, race, religion, urban
  - QA summary 안 씀 — delay_root10만 추가
  - early_stopping_rounds=200
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
from common_fe_v2 import load_common, prepare_for_catboost, save_npy
from tracker import try_log, show_leaderboard


# =========================================================
# 0. Path
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test_x.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_SUB_PATH = BASE_DIR / "submission_catboost.csv"


# =========================================================
# 1. Config (boosting.py Optuna best params 그대로)
# =========================================================
SEED = 42
N_SPLITS = 5

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": SEED,
    "allow_writing_files": False,
    "verbose": 200,
    "early_stopping_rounds": 200,
    "iterations": 2500,
    # Optuna best
    "learning_rate": 0.011353066986375095,
    "depth": 7,
    "l2_leaf_reg": 3.427353847207771,
    "random_strength": 1.4620769648916006,
    "min_data_in_leaf": 10,
    "border_count": 132,
    "bagging_temperature": 1.2166796843617005,
}


# =========================================================
# 2. Data
# =========================================================
train_df, test_df, y, sample_sub = load_common(
    TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH
)
X_train, X_test, cat_features = prepare_for_catboost(train_df, test_df)
y_int = y.astype(int)


# =========================================================
# 3. Train — 5-Fold CV
# =========================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_pred = np.zeros(len(X_train), dtype=np.float64)
test_pred = np.zeros(len(X_test), dtype=np.float64)

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


# =========================================================
# 4. Final
# =========================================================
oof_auc = roc_auc_score(y_int, oof_pred)
print(f"\n{'='*50}\nFINAL OOF AUC: {oof_auc:.6f}\n{'='*50}")

save_npy(oof_pred, OUTPUT_DIR / "catboost_oof.npy")
save_npy(test_pred, OUTPUT_DIR / "catboost_test.npy")

submission = sample_sub.copy()
submission["voted"] = test_pred
submission.to_csv(OUTPUT_SUB_PATH, index=False)
print(f"Submission: {OUTPUT_SUB_PATH}")

try_log(
    "catboost", oof_auc, CATBOOST_PARAMS, {},
    notes=(
        "team_catboost [팀원코드, Optuna튜닝결과] "
        "사용컬럼: QA_raw20+tp_raw10+wr_raw13+wf_raw3+인구통계8개(hand제외) | "
        "파생: delay_root10만 | "
        "하이퍼파라미터: lr=0.01135+depth=7+l2=3.43+random_strength=1.46+min_data_in_leaf=10+border_count=132+bagging_temperature=1.22(Optuna최적화) | "
        "voted인코딩: voted-1방식(1=미투표)"
    ),
)
show_leaderboard()
