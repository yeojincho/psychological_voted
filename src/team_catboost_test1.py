"""
src/team_catboost_test1.py
──────────────────────────
team_catboost.py 기반 노이즈 컬럼 최소 제거 실험

team_catboost 대비 변경점:
  1. hand 제거
       - CATEGORICAL_COLS에 없어서 수치형으로 방치되어 있음
       - 손잡이(오른손잡이/왼손잡이)는 투표 행동과 도메인 관련성 없음
       - 오히려 트리 분기에 노이즈로 작용할 가능성
  2. engnat 제거
       - 영어 모국어 여부 → 온라인 설문 메타 정보
       - 투표 행동보다 데이터 수집 환경 반영 가능성

나머지는 team_catboost와 완전 동일:
  - Optuna best params 그대로
  - delay_root10, familysize log1p
  - QA raw 20개, tp raw 10개, wr/wf raw 유지

실행:
    python src/team_catboost_test1.py

출력:
    outputs/models/team_catboost_test1_oof.npy
    outputs/models/team_catboost_test1_test.npy
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


# =========================================================
# 1. Config (team_catboost Optuna best params 그대로)
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
    "learning_rate": 0.011353066986375095,
    "depth": 7,
    "l2_leaf_reg": 3.427353847207771,
    "random_strength": 1.4620769648916006,
    "min_data_in_leaf": 10,
    "border_count": 132,
    "bagging_temperature": 1.2166796843617005,
}

# hand, engnat 제거
NOISE_COLS = ["hand", "engnat"]

# engnat 제거했으므로 cat_features에서도 제외
CAT_COLS = [c for c in CATEGORICAL_COLS if c not in NOISE_COLS]


# =========================================================
# 2. FE (prepare_for_catboost 인라인 재현 + 노이즈 제거)
# =========================================================
def prepare(train_df, test_df):
    from common_fe_v2 import _add_delay_root10

    X_train = train_df.copy()
    X_test = test_df.copy()

    # delay_root10
    X_train = _add_delay_root10(X_train)
    X_test  = _add_delay_root10(X_test)

    # familysize log1p
    for df in [X_train, X_test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # 범주형 → 문자열
    for col in CAT_COLS:
        for df in [X_train, X_test]:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # 불필요 컬럼 제거: QE raw + index + voted + 노이즈
    drop_base = QE_COLS + ["voted", "index"] + NOISE_COLS
    drop_test_base = QE_COLS + ["index"] + NOISE_COLS
    X_train = X_train.drop(columns=[c for c in drop_base if c in X_train.columns])
    X_test  = X_test.drop(columns=[c for c in drop_test_base if c in X_test.columns])

    cat_indices = [X_train.columns.get_loc(c) for c in CAT_COLS if c in X_train.columns]

    print(f"\nFE 완료 (노이즈 제거: {NOISE_COLS})")
    print(f"  피처 {X_train.shape[1]}개 | cat_features {len(cat_indices)}개")

    return X_train, X_test, cat_indices


# =========================================================
# 3. Data
# =========================================================
train_df, test_df, y, sample_sub = load_common(
    TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH
)
X_train, X_test, cat_features = prepare(train_df, test_df)
y_int = y.astype(int)


# =========================================================
# 4. Train — 5-Fold CV
# =========================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

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


# =========================================================
# 5. Final
# =========================================================
oof_auc = roc_auc_score(y_int, oof_pred)
print(f"\n{'='*50}\nFINAL OOF AUC: {oof_auc:.6f}\n{'='*50}")

save_npy(oof_pred, OUTPUT_DIR / "team_catboost_test1_oof.npy")
save_npy(test_pred, OUTPUT_DIR / "team_catboost_test1_test.npy")

try_log(
    "catboost", oof_auc, CATBOOST_PARAMS, {},
    notes=(
        "team_catboost_test1 [vs team_catboost: hand+engnat 제거] "
        "사용컬럼: QA_raw20+tp_raw10+wr_raw13+wf_raw3+인구통계6개 | "
        "파생: delay_root10만 | "
        "제거: hand(수치형방치+도메인무관)+engnat(설문메타정보) | "
        "하이퍼파라미터: Optuna최적값그대로(lr=0.01135+depth=7+l2=3.43) | "
        "voted인코딩: voted-1방식(1=미투표)"
    ),
)
show_leaderboard()
