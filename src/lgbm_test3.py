"""
src/lgbm_test3.py
─────────────────
LightGBM 실험 #3 — LightGBM 구조에 최적화된 FE

LightGBM 고유 특성과 그에 따른 FE 전략:

  ① Leaf-wise tree growth (vs XGBoost depth-wise)
     → 비대칭 분기로 복잡한 다항 관계를 자체 포착 → polynomial 피처 불필요
     → num_leaves=255로 확장 (depth와 독립 제어)

  ② Native categorical (Fisher's optimal partitioning)
     → OrdinalEncoder보다 훨씬 강력 — 고카디널리티 교차 category 적극 활용
     → cross_cat 종류를 test2(3개)에서 5개로 확장
       (married_x_urban, gender_x_edu 추가)
     → mach_bin, vocab_bin: 연속형을 분위 기반 bin → category로 변환
       (LGBM Fisher 알고리즘이 최적 분할점 찾음)

  ③ 히스토그램 기반 분기 → percentile rank 변환
     → 연속형 피처의 값 분포가 왜곡(skewed)되면 히스토그램 버킷 낭비
     → rank 변환으로 균등 분포화 → 버킷 품질 향상
     → mach_score, vocab_score, delay_root10, familysize에 적용

  ④ LGBM 전용 파라미터 활용
     → min_child_samples=10 (리프 최소 샘플, leaf-wise에서 작게 가능)
     → cat_smooth=10 (범주형 과적합 방지 smoothing)
     → path_smooth=0.1 (리프 경로 regularization)

test2 대비 변경 요약:
  [추가] cross_cat 확장: married_x_urban, gender_x_edu (5개로)
  [추가] mach_bin, vocab_bin (분위 기반 → category)
  [추가] mach_rank, vocab_rank, delay_rank, familysize_rank (percentile rank)
  [제거] polynomial (mach_score_sq 등) → leaf-wise가 자체 포착
  [조정] num_leaves 127→255, min_child_samples 20→10
         cat_smooth=10, path_smooth=0.1 추가

실행:
    python -m src.lgbm_test3

출력 (신기록 시에만):
    outputs/lgbm_test3_submission.csv
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
# 설정
# ================================================================
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test_x.csv"
OUT_PATH   = "outputs/lgbm_test3_submission.csv"
SEED       = 42
N_SPLITS   = 5

Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
QE_COLS = [f"Q{chr(ord('a')+i)}E" for i in range(20)]
TP_COLS = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]

CAT_COLS = ["age_group","education","engnat","gender",
            "married","race","religion","urban"]

# cross_cat 5개 (test2: 3개에서 확장)
CROSS_COLS = [
    "age_x_edu",
    "gender_x_urban",
    "race_x_religion",
    "married_x_urban",   # [NEW] 결혼+거주지 (생활 환경 복합)
    "gender_x_edu",      # [NEW] 성별+교육 (사회경제적 지위 복합)
]

# bin category 피처
BIN_COLS = ["mach_bin", "vocab_bin"]

# LGBM에 category dtype으로 전달할 모든 컬럼
ALL_CAT_COLS = CAT_COLS + CROSS_COLS + BIN_COLS

# ================================================================
# 하이퍼파라미터 (LightGBM 최적화)
# ================================================================
PARAMS = dict(
    objective         = "binary",
    metric            = "auc",
    n_estimators      = 4000,
    learning_rate     = 0.02,
    num_leaves        = 255,          # leaf-wise: depth와 독립, 넓은 탐색
    max_depth         = -1,
    min_child_samples = 10,           # leaf-wise에서 작게 허용 → rare 패턴 포착
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.7,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    cat_smooth        = 10,           # LGBM 전용: 범주형 과적합 방지 smoothing
    path_smooth       = 0.1,          # LGBM 전용: 리프 경로 regularization
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
# rank 변환 헬퍼 (train 기준 fit, test transform — 누수 방지)
# ================================================================
def rank_transform(train_col: pd.Series, test_col: pd.Series):
    """train 기준 percentile rank → [0, 1] 범위"""
    n = len(train_col)
    # train: 자기 자신 순위
    train_rank = train_col.rank(method="average", na_option="keep") / n
    # test: train 분포 기준 보간
    sorted_train = train_col.dropna().sort_values().values
    def _to_rank(val):
        if pd.isna(val):
            return np.nan
        idx = np.searchsorted(sorted_train, val, side="left")
        return idx / n
    test_rank = test_col.map(_to_rank)
    return train_rank.astype(np.float32), test_rank.astype(np.float32)


# ================================================================
# 피처 엔지니어링
# ================================================================
def preprocess(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train, test = train_raw.copy(), test_raw.copy()

    for df in [train, test]:
        if "index" in df.columns:
            df.drop(columns=["index"], inplace=True)

    # hand 제거
    for df in [train, test]:
        df.drop(columns=["hand"], errors="ignore", inplace=True)

    # familysize 이상치 제거 (train만)
    outlier_idx = train[train["familysize"] > 50].index
    if len(outlier_idx):
        print(f"familysize 이상치 제거: {len(outlier_idx)}행")
        train = train.drop(index=outlier_idx).reset_index(drop=True)

    # ── 1. tp 파생 피처 (NaN 변환 전) ───────────────────────────
    avail_tp = [c for c in TP_COLS if c in train.columns]
    for df in [train, test]:
        df["tp_notapplicable_cnt"] = (df[avail_tp] == 7).sum(axis=1)
        df["tp_missing_cnt"]       = (df[avail_tp] == 0).sum(axis=1)
    for col in avail_tp:
        train[col] = train[col].replace({0: np.nan, 7: np.nan})
        test[col]  = test[col].replace({0: np.nan, 7: np.nan})

    # tp 일관성 피처
    for df in [train, test]:
        df["tp_mean"]  = df[avail_tp].mean(axis=1)
        df["tp_std"]   = df[avail_tp].std(axis=1)
        df["tp_range"] = df[avail_tp].max(axis=1) - df[avail_tp].min(axis=1)

    # ── 2. 인구통계 무응답 → NaN ─────────────────────────────────
    for col in ["education","engnat","married","urban"]:
        if col in train.columns:
            train[col] = train[col].replace(0, np.nan)
            test[col]  = test[col].replace(0, np.nan)

    # 서수형 수치 인코딩
    edu_map          = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    age_midpoint_map = {1: 15, 2: 21, 3: 29, 4: 39, 5: 49, 6: 59, 7: 70}
    for df in [train, test]:
        df["edu_num"] = df["education"].map(edu_map)
        df["age_num"] = df["age_group"].map(age_midpoint_map)

    # ── 3. 역문항 리버스 코딩 ────────────────────────────────────
    for col in REVERSE_Q:
        if col in train.columns:
            train[col] = 6 - train[col]
            test[col]  = 6 - test[col]

    # ── 4. MACH 피처 ─────────────────────────────────────────────
    avail_q = [c for c in Q_COLS if c in train.columns]
    for df in [train, test]:
        df["mach_score"]      = df[avail_q].mean(axis=1)
        df["q_response_std"]  = df[avail_q].std(axis=1)
        df["q_extreme_ratio"] = (
            (df[avail_q] == 1) | (df[avail_q] == 5)
        ).sum(axis=1) / len(avail_q)

    # mach × edu 교호작용
    for df in [train, test]:
        df["mach_x_edu"] = df["mach_score"] * df["edu_num"].fillna(3)

    # ── 5. 어휘력 피처 ───────────────────────────────────────────
    avail_wr = [c for c in WR_COLS if c in train.columns]
    avail_wf = [c for c in WF_COLS if c in train.columns]
    for df in [train, test]:
        df["vocab_real"]     = df[avail_wr].sum(axis=1)
        df["vocab_fake"]     = df[avail_wf].sum(axis=1)
        df["vocab_score"]    = df["vocab_real"] - df["vocab_fake"] * 2
        df["vocab_accuracy"] = df["vocab_real"] / (df["vocab_real"] + df["vocab_fake"] + 1e-6)

    # 교호작용
    for df in [train, test]:
        df["edu_x_vocab"] = df["edu_num"].fillna(3) * df["vocab_score"]
        df["age_x_mach"]  = df["age_num"].fillna(39) * df["mach_score"]

    # ── 6. QE → delay_root10 + 분산 피처 ────────────────────────
    avail_qe = [c for c in QE_COLS if c in train.columns]
    for df in [train, test]:
        qe_vals    = df[avail_qe].clip(lower=0)
        qe_sum     = qe_vals.sum(axis=1)
        qe_mean    = qe_vals.mean(axis=1)
        qe_std_val = qe_vals.std(axis=1)
        df["delay_root10"] = np.power(qe_sum, 0.1)
        df["qe_std"]       = qe_std_val
        df["qe_cv"]        = qe_std_val / (qe_mean + 1)
        df["qe_max_ratio"] = qe_vals.max(axis=1) / (qe_mean + 1)
    train = train.drop(columns=avail_qe)
    test  = test.drop(columns=avail_qe)

    # ── 7. familysize log1p ──────────────────────────────────────
    for df in [train, test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # ── 8. [NEW] Percentile rank 변환 (train 기준 fit, 누수 방지) ──
    # 히스토그램 버킷 균등 분포화 → LGBM 분기 품질 향상
    rank_targets = ["mach_score", "vocab_score", "delay_root10", "familysize"]
    for col in rank_targets:
        if col in train.columns:
            tr_rank, te_rank = rank_transform(train[col], test[col])
            train[f"{col}_rank"] = tr_rank
            test[f"{col}_rank"]  = te_rank

    # ── 9. [NEW] 분위 기반 bin → category (LGBM Fisher 알고리즘 활용) ──
    # train 기준 분위수로 bin 생성, test에 적용 (누수 방지)
    for col, bin_col in [("mach_score", "mach_bin"), ("vocab_score", "vocab_bin")]:
        if col in train.columns:
            _, bin_edges = pd.qcut(train[col], q=5, retbins=True, duplicates="drop")
            bin_edges[0]  = -np.inf
            bin_edges[-1] =  np.inf
            train[bin_col] = pd.cut(train[col], bins=bin_edges, labels=False).astype("category")
            test[bin_col]  = pd.cut(test[col],  bins=bin_edges, labels=False).astype("category")

    # ── 10. [NEW] 교차 category 확장 (5개) ──────────────────────
    # LGBM native categorical → Fisher 최적 분할 → 고카디널리티도 OK
    for df in [train, test]:
        df["age_x_edu"]       = (df["age_group"].astype(str) + "_" + df["education"].astype(str)).astype("category")
        df["gender_x_urban"]  = (df["gender"].astype(str)    + "_" + df["urban"].astype(str)).astype("category")
        df["race_x_religion"] = (df["race"].astype(str)      + "_" + df["religion"].astype(str)).astype("category")
        df["married_x_urban"] = (df["married"].astype(str)   + "_" + df["urban"].astype(str)).astype("category")
        df["gender_x_edu"]    = (df["gender"].astype(str)    + "_" + df["education"].astype(str)).astype("category")

    # ── 11. 기존 범주형 → category dtype ─────────────────────────
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
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train[tr_idx]
        y_val = y_train[val_idx]

        model = lgb.LGBMClassifier(**PARAMS)
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
    print(f"LGBM 전용 추가:")
    print(f"  - cross_cat 5개: age_x_edu / gender_x_urban / race_x_religion")
    print(f"                   married_x_urban / gender_x_edu (Fisher 최적분할)")
    print(f"  - bin category: mach_bin, vocab_bin (분위5→category)")
    print(f"  - rank 변환: mach/vocab/delay/familysize_rank (히스토그램 균등화)")
    print(f"  - 하이퍼파라미터: num_leaves=255, min_child_samples=10")
    print(f"                    cat_smooth=10, path_smooth=0.1\n")

    oof_preds, test_preds, oof_auc = train_cv(X_train, y_train, X_test)

    os.makedirs("outputs", exist_ok=True)
    recorded = try_log(
        "lgbm", oof_auc, PARAMS, FIT_PARAMS,
        notes=(
            "lgbm_test3 [LGBM 구조 최적화 FE] "
            "사용컬럼: QA_raw20+tp_raw10+wr_raw13+wf_raw3+인구통계8개(hand제거) | "
            "파생(test2유지): mach_score+q_response_std+q_extreme_ratio+vocab4개 | "
            "       delay_root10+qe_std+qe_cv+qe_max_ratio+tp_mean+tp_std+tp_range | "
            "       mach_x_edu+edu_x_vocab+age_x_mach+edu_num+age_num | "
            "LGBM전용추가: cross_cat5개(age_x_edu+gender_x_urban+race_x_religion+married_x_urban+gender_x_edu) | "
            "       mach_bin+vocab_bin(분위5→category,Fisher최적분할) | "
            "       mach_rank+vocab_rank+delay_rank+familysize_rank(percentile,히스토그램균등화) | "
            "하이퍼파라미터: num_leaves=255+min_child_samples=10+cat_smooth=10+path_smooth=0.1"
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
