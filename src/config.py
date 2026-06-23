"""
src/config.py
─────────────
경로, 컬럼 상수, 하이퍼파라미터 설정.
"""

# ================================================================
# 경로
# ================================================================
TRAIN_PATH  = "data/train.csv"
TEST_PATH   = "data/test_x.csv"
OUTPUT_DIR  = "outputs"
MODEL_DIR   = "models"

# ================================================================
# 학습 설정
# ================================================================
SEED     = 42
N_SPLITS = 5

# ================================================================
# 컬럼 그룹
# ================================================================
Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
QE_COLS = [f"Q{chr(ord('a')+i)}E" for i in range(20)]
TP_COLS = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

# 역문항 10개 (Spearman 상관분석 기반)
REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]

# CatBoost에 문자열로 넘길 범주형 컬럼
CAT_COLS = ["age_group","education","engnat","gender",
            "hand","married","race","religion","urban"]

# ================================================================
# 모델 하이퍼파라미터
# ================================================================
LGBM_PARAMS = dict(
    objective        = "binary",
    metric           = "auc",
    boosting_type    = "gbdt",
    n_estimators     = 3000,
    learning_rate    = 0.02,
    num_leaves       = 127,
    min_child_samples= 15,
    subsample        = 0.8,
    subsample_freq   = 1,
    colsample_bytree = 0.8,
    colsample_bynode = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    scale_pos_weight = 1.20665,
    random_state     = SEED,
    n_jobs           = -1,
    verbose          = -1,
)
LGBM_FIT = dict(early_stopping_rounds=150, log_eval=200)

XGB_PARAMS = dict(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    n_estimators     = 3000,
    learning_rate    = 0.02,
    max_depth        = 5,
    min_child_weight = 3,
    gamma            = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    colsample_bylevel= 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    scale_pos_weight = 1.20665,
    tree_method      = "hist",
    random_state     = SEED,
    n_jobs           = -1,
    verbosity        = 0,
)
XGB_FIT = dict(early_stopping_rounds=150, verbose=200)

CATBOOST_PARAMS = dict(
    objective            = "Logloss",
    eval_metric          = "AUC",
    iterations           = 3000,
    learning_rate        = 0.01,
    depth                = 6,
    min_data_in_leaf     = 15,
    l2_leaf_reg          = 5,
    bootstrap_type       = "Bernoulli",
    subsample            = 0.8,
    scale_pos_weight     = 1.20665,
    random_seed          = SEED,
    allow_writing_files  = False,
    task_type            = "CPU",
)
CATBOOST_FIT = dict(early_stopping_rounds=200, verbose=200)
