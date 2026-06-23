"""
train_linear.py — Linear 학습 코드 (liner.py 정확 재현)
════════════════════════════════════════════════════════
common_fe_v2.py를 import하여 FE 수행
모델/학습 로직은 liner.py와 100% 동일

현재 최고 기록: OOF AUC ≈ 0.764888

핵심 설정:
  - LogisticRegression(C=1.0, solver='lbfgs', max_iter=3000)
  - Fold별 StandardScaler fit
  - QA features: QA_mean, QA_std, QA_top_mean, A_low_ratio, A_high_ratio
  - TP features: TP_top_mean
  - Interaction: edu_married_group, edu_age_group
  - OHE: 기본 8개 범주형 + interaction 2개
"""

from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fe_v2 import load_common, prepare_for_linear, save_npy


# =========================================================
# 0. Path
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test_x.csv"
OUTPUT_SUB_PATH = BASE_DIR / "submission_linear.csv"


# =========================================================
# 1. Config
# =========================================================
SEED = 42
N_SPLITS = 5


# =========================================================
# 2. Data
# =========================================================
train_df, test_df, y, _ = load_common(TRAIN_PATH, TEST_PATH)
X_train, X_test = prepare_for_linear(train_df, test_df)

# y를 int로
y_int = y.astype(int)


# =========================================================
# 3. Train — 5-Fold CV + fold별 StandardScaler
# =========================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_pred = np.zeros(len(X_train))
test_pred = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_int), 1):

    scaler = StandardScaler()

    X_tr = scaler.fit_transform(X_train.iloc[tr_idx])
    y_tr = y_int[tr_idx]

    X_val = scaler.transform(X_train.iloc[val_idx])
    y_val = y_int[val_idx]

    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        C=1.0,
        max_iter=3000,
        n_jobs=-1,
        solver="lbfgs",
        random_state=SEED,
    )
    model.fit(X_tr, y_tr)

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_pred[val_idx] = val_pred
    test_pred += model.predict_proba(X_test_scaled)[:, 1] / N_SPLITS

    print(f"Fold {fold} AUC: {roc_auc_score(y_val, val_pred):.6f}")


# =========================================================
# 4. Final
# =========================================================
oof_auc = roc_auc_score(y_int, oof_pred)
print(f"\n{'='*50}\nLogisticRegression OOF AUC: {oof_auc:.6f}\n{'='*50}")

save_npy(oof_pred, OUTPUT_DIR / "linear_oof.npy")
save_npy(test_pred, OUTPUT_DIR / "linear_test.npy")
print("Done.")
