"""
src/catboost_test4.py
─────────────────────
CatBoost 실험 #4 — AUC 향상 집중

점수가 안 오르는 근본 원인 분석:
  - 피처 수 증가만으로는 한계 → 예측력 높은 상호작용 필요
  - 인구통계(나이/교육)가 투표여부와 높은 상관 → 심리 피처와 교차
  - QA 응답 패턴 중 '중립(3) 비율', '엔트로피' 등 미개척

핵심 변경점:
  1. age_group 수치화 → age_num (10대=1, 20대=2, ...) 연속 피처로 추가
  2. 인구통계 × 심리 상호작용:
       age_num × mach_score
       education × vocab_accuracy
       age_num × vocab_score
  3. QA 응답 패턴 심화:
       q_neutral_ratio  : 중립(3) 응답 비율 (우유부단/무관심 지표)
       q_entropy        : 응답 분포 엔트로피 (다양성 지표)
       q_agree_ratio    : 동의(4,5) 응답 비율
  4. 전체 결측 패턴:
       total_missing    : 전체 피처의 결측 수 (참여 회피 성향)
  5. 하이퍼파라미터 강화:
       depth 6→7, min_data_in_leaf 15→10
       random_strength 추가, bagging_temperature 조정

실행:
    python -m src.catboost_test4

출력 (신기록 시에만):
    outputs/catboost_test4_submission.csv
"""

import os
import random
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

from src.tracker import try_log, show_leaderboard

# ================================================================
# 설정
# ================================================================
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test_x.csv"
OUT_PATH   = "outputs/catboost_test4_submission.csv"
SEED       = 42
N_SPLITS   = 5

Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
QE_COLS = [f"Q{chr(ord('a')+i)}E" for i in range(20)]
TP_COLS = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]

CAT_COLS = ["age_group","education","engnat","gender",
            "hand","married","race","religion","urban"]

# age_group 수치 매핑 (투표율은 고령일수록 높음)
AGE_MAP = {
    "1": 1,  # 10대 미만
    "2": 2,  # 10대
    "3": 3,  # 20대
    "4": 4,  # 30대
    "5": 5,  # 40대
    "6": 6,  # 50대
    "7": 7,  # 60대 이상
}

# ================================================================
# 하이퍼파라미터 (강화)
# ================================================================
PARAMS = dict(
    objective           = "Logloss",
    eval_metric         = "AUC",
    iterations          = 3000,
    learning_rate       = 0.02,       # 0.03 → 0.02 (더 정밀하게)
    depth               = 7,          # 6 → 7
    min_data_in_leaf    = 10,         # 15 → 10
    l2_leaf_reg         = 3,          # 5 → 3
    random_strength     = 1.0,        # 추가: 분할 랜덤성
    bagging_temperature = 0.5,        # 추가: 가중치 샘플링 온도
    bootstrap_type      = "Bayesian", # Bernoulli → Bayesian (depth 7과 궁합)
    scale_pos_weight    = 1.20665,
    random_seed         = SEED,
    allow_writing_files = False,
)
FIT_PARAMS = dict(
    early_stopping_rounds = 200,
    verbose               = 200,
)


# ================================================================
# 피처 엔지니어링
# ================================================================
def _q_entropy(row: pd.Series) -> float:
    """QA 20문항 응답값(1~5) 분포의 엔트로피. 높을수록 다양한 응답."""
    counts = np.bincount(row.astype(int).clip(1, 5), minlength=6)[1:]
    return float(scipy_entropy(counts + 1e-9))  # 라플라스 스무딩


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

    # ── 1. tp 파생 피처 (NaN 변환 전) ───────────────────────
    avail_tp = [c for c in TP_COLS if c in train.columns]
    for df in [train, test]:
        df["tp_notapplicable_cnt"] = (df[avail_tp] == 7).sum(axis=1)
        df["tp_missing_cnt"]       = (df[avail_tp] == 0).sum(axis=1)
    for col in avail_tp:
        train[col] = train[col].replace({0: np.nan, 7: np.nan})
        test[col]  = test[col].replace({0: np.nan, 7: np.nan})

    # ── 2. 인구통계 무응답 → NaN ─────────────────────────────
    for col in ["education","engnat","hand","married","urban"]:
        if col in train.columns:
            train[col] = train[col].replace(0, np.nan)
            test[col]  = test[col].replace(0, np.nan)

    # ── 3. age_group 수치화 (연속 피처 추가) ─────────────────
    for df in [train, test]:
        if "age_group" in df.columns:
            df["age_num"] = df["age_group"].astype(str).map(AGE_MAP).fillna(3)

    # ── 4. education 수치화 (1~4, NaN=0으로 임시) ────────────
    for df in [train, test]:
        if "education" in df.columns:
            df["edu_num"] = pd.to_numeric(df["education"], errors="coerce").fillna(0)

    # ── 5. 역문항 리버스 코딩 ────────────────────────────────
    for col in REVERSE_Q:
        if col in train.columns:
            train[col] = 6 - train[col]
            test[col]  = 6 - test[col]

    # ── 6. MACH 피처 + 응답 패턴 심화 ───────────────────────
    avail_q = [c for c in Q_COLS if c in train.columns]
    for df in [train, test]:
        df["mach_score"]      = df[avail_q].mean(axis=1)
        df["q_response_std"]  = df[avail_q].std(axis=1)
        df["q_extreme_ratio"] = (
            (df[avail_q] == 1) | (df[avail_q] == 5)
        ).sum(axis=1) / len(avail_q)

        # 신규: 중립(3) 비율 — 높을수록 우유부단/무관심
        df["q_neutral_ratio"] = (df[avail_q] == 3).sum(axis=1) / len(avail_q)

        # 신규: 동의(4,5) 비율 — 마키아벨리즘 동의 성향
        df["q_agree_ratio"]   = (df[avail_q] >= 4).sum(axis=1) / len(avail_q)

        # 신규: 응답 엔트로피 — 높을수록 응답이 고르게 분포
        df["q_entropy"] = df[avail_q].apply(_q_entropy, axis=1)

    # ── 7. 어휘력 피처 ───────────────────────────────────────
    avail_wr = [c for c in WR_COLS if c in train.columns]
    avail_wf = [c for c in WF_COLS if c in train.columns]
    for df in [train, test]:
        df["vocab_real"]     = df[avail_wr].sum(axis=1)
        df["vocab_fake"]     = df[avail_wf].sum(axis=1)
        df["vocab_score"]    = df["vocab_real"] - df["vocab_fake"] * 2
        df["vocab_accuracy"] = df["vocab_real"] / (df["vocab_real"] + df["vocab_fake"] + 1e-6)

    # ── 8. 인구통계 × 심리 상호작용 (핵심 신규) ─────────────
    for df in [train, test]:
        # 나이 × 마키아벨리즘 (고령+저마키아벨리즘 = 높은 투표율 가설)
        df["age_x_mach"]     = df["age_num"] * df["mach_score"]

        # 나이 × 어휘력 (고령+고어휘 = 높은 참여 성향)
        df["age_x_vocab"]    = df["age_num"] * df["vocab_score"]

        # 교육 × 어휘 정확도 (고학력+고어휘 = 높은 투표율 가설)
        df["edu_x_vocab_acc"] = df["edu_num"] * df["vocab_accuracy"]

        # 교육 × mach_score
        df["edu_x_mach"]     = df["edu_num"] * df["mach_score"]

        # 중립 응답 × 나이 (고령+중립 응답 = 낮은 참여 가설)
        df["age_x_neutral"]  = df["age_num"] * df["q_neutral_ratio"]

    # ── 9. 전체 결측 패턴 ────────────────────────────────────
    # 설문 회피 성향 지표 (tp + 인구통계 무응답 합산)
    for df in [train, test]:
        df["total_missing"] = df[avail_tp].isna().sum(axis=1) + \
                              df[["education","engnat","hand","married","urban"]].isna().sum(axis=1)

    # ── 10. QE → delay_root10 + time_cv ─────────────────────
    avail_qe = [c for c in QE_COLS if c in train.columns]
    for df in [train, test]:
        qe_clip = df[avail_qe].clip(lower=0)
        qe_mean = qe_clip.mean(axis=1)
        qe_std  = qe_clip.std(axis=1)
        df["delay_root10"] = np.power(qe_clip.sum(axis=1), 0.1)
        df["time_cv"]      = qe_std / (qe_mean + 1e-6)
    train = train.drop(columns=avail_qe)
    test  = test.drop(columns=avail_qe)

    # ── 11. familysize log1p ─────────────────────────────────
    for df in [train, test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # ── 12. 범주형 → 문자열 (CatBoost용) ────────────────────
    for col in CAT_COLS:
        if col in train.columns:
            train[col] = train[col].astype(str)
            test[col]  = test[col].astype(str)

    return train, test


# ================================================================
# CV 학습
# ================================================================
def train_cv(X_train, y_train, X_test, cat_idx):
    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n── Fold {fold}/{N_SPLITS} ──────────────────────────")
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train[tr_idx]
        y_val = y_train[val_idx]

        train_pool = Pool(X_tr,  label=y_tr,  cat_features=cat_idx)
        val_pool   = Pool(X_val, label=y_val, cat_features=cat_idx)
        test_pool  = Pool(X_test, cat_features=cat_idx)

        model = CatBoostClassifier(**PARAMS, **FIT_PARAMS)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        vp = model.predict_proba(val_pool)[:, 1]
        oof_preds[val_idx] = vp
        test_preds += model.predict_proba(test_pool)[:, 1] / N_SPLITS

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

    cat_idx = [i for i, c in enumerate(X_train.columns) if c in CAT_COLS]

    print(f"피처: {X_train.shape[1]}개 | 범주형: {len(cat_idx)}개 | 샘플: {len(y_train)}")
    print(f"신규 피처: age_num, edu_num, age_x_mach, age_x_vocab, edu_x_vocab_acc, "
          f"edu_x_mach, age_x_neutral, q_neutral_ratio, q_agree_ratio, q_entropy, "
          f"total_missing, time_cv\n")

    oof_preds, test_preds, oof_auc = train_cv(X_train, y_train, X_test, cat_idx)

    os.makedirs("outputs", exist_ok=True)
    recorded = try_log(
        "catboost", oof_auc, PARAMS, FIT_PARAMS,
        notes=(
            "test4 [vs test1: 인구통계수치화+교호작용+응답패턴피처추가, 하이퍼파라미터강화] "
            "사용컬럼: QA_raw20+tp_raw10+wr_raw13+wf_raw3+인구통계9개 | "
            "추가파생: age_num+edu_num(수치화)+age_x_mach+age_x_vocab+edu_x_vocab_acc+edu_x_mach+age_x_neutral(교호작용)+q_neutral_ratio+q_agree_ratio+q_entropy+total_missing | "
            "하이퍼파라미터변경: depth6→7+min_data_in_leaf15→10+l2_5→3+random_strength1.0추가+bagging_temperature0.5+bootstrap_type_Bayesian | "
            "제거: QE_raw20"
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
