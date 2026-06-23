"""
src/lgbm_test2.py
─────────────────
LightGBM 실험 #2 — LGBM 특성 맞춤 FE 강화

lgbm_test1 대비 변경:
  [추가] 카테고리 교차 피처 (age_x_edu, gender_x_urban, race_x_religion)
       → LGBM은 CatBoost처럼 내부 카테고리 임베딩이 없으므로 명시적 교차항 필요
  [추가] 서수형 수치 인코딩 (edu_num, age_num)
       → 순서 정보를 숫자로 명시해 트리 분기 품질 향상
  [추가] QE 분산 피처 (qe_std, qe_cv, qe_max_ratio)
       → delay_root10(합계)만으로는 응답 시간 패턴 내 분산 정보 손실
  [추가] tp 일관성 피처 (tp_mean, tp_std, tp_range)
       → tp_missing/notapplicable만 있고 응답 수준/분산 미활용
  [추가] mach_x_edu (MACH 점수 × 교육 수준 교호작용 수치 피처)
  [제거] hand (도메인 무관 노이즈)
  [조정] 하이퍼파라미터: num_leaves 63→127, lr 0.03→0.02, n_estimators 3000→4000

LGBM vs CatBoost 핵심 차이:
  - CatBoost: 카테고리 임베딩 + symmetric tree → raw categorical 강함
  - LGBM: leaf-wise tree, category dtype은 단순 label encoding 수준
  → 교차 피처/서수 수치 인코딩이 LGBM AUC 향상에 효과적

실행:
    python -m src.lgbm_test2

출력 (신기록 시에만):
    outputs/lgbm_test2_submission.csv
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
OUT_PATH   = "outputs/lgbm_test2_submission.csv"
SEED       = 42
N_SPLITS   = 5

Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
QE_COLS = [f"Q{chr(ord('a')+i)}E" for i in range(20)]
TP_COLS = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]

# hand 제거 — 나머지 8개 유지
CAT_COLS = ["age_group","education","engnat","gender",
            "married","race","religion","urban"]

# 교차 피처 (카테고리로 처리)
CROSS_CAT_COLS = ["age_x_edu", "gender_x_urban", "race_x_religion"]

# ================================================================
# 하이퍼파라미터
# ================================================================
PARAMS = dict(
    objective         = "binary",
    metric            = "auc",
    n_estimators      = 4000,
    learning_rate     = 0.02,
    num_leaves        = 127,
    max_depth         = -1,
    min_child_samples = 20,
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.7,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
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
# 피처 엔지니어링
# ================================================================
def preprocess(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train, test = train_raw.copy(), test_raw.copy()

    for df in [train, test]:
        if "index" in df.columns:
            df.drop(columns=["index"], inplace=True)

    # hand 제거 (도메인 무관 노이즈)
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

    # [NEW] tp 일관성 피처 (NaN 변환 후)
    for df in [train, test]:
        df["tp_mean"]  = df[avail_tp].mean(axis=1)   # 평균 Big5 응답 수준
        df["tp_std"]   = df[avail_tp].std(axis=1)    # 응답 일관성 (낮을수록 일관됨)
        df["tp_range"] = df[avail_tp].max(axis=1) - df[avail_tp].min(axis=1)

    # ── 2. 인구통계 무응답 → NaN ─────────────────────────────────
    for col in ["education","engnat","married","urban"]:
        if col in train.columns:
            train[col] = train[col].replace(0, np.nan)
            test[col]  = test[col].replace(0, np.nan)

    # [NEW] 서수형 수치 인코딩
    # education: 1=고졸미만 2=고졸 3=대학중퇴 4=대졸 5=대학원
    edu_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    # age_group: 1=<18 2=18-24 3=25-34 4=35-44 5=45-54 6=55-64 7=65+
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

    # [NEW] mach × edu 교호작용 수치 피처
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

    # ── 6. QE → delay_root10 + [NEW] 분산 피처 ──────────────────
    avail_qe = [c for c in QE_COLS if c in train.columns]
    for df in [train, test]:
        qe_vals    = df[avail_qe].clip(lower=0)
        qe_sum     = qe_vals.sum(axis=1)
        qe_mean    = qe_vals.mean(axis=1)
        qe_std_val = qe_vals.std(axis=1)
        df["delay_root10"] = np.power(qe_sum, 0.1)
        df["qe_std"]       = qe_std_val                           # 항목 간 시간 분산
        df["qe_cv"]        = qe_std_val / (qe_mean + 1)          # 변동계수 (정규화된 분산)
        df["qe_max_ratio"] = qe_vals.max(axis=1) / (qe_mean + 1) # 가장 느린 항목의 상대적 크기
    train = train.drop(columns=avail_qe)
    test  = test.drop(columns=avail_qe)

    # ── 7. familysize log1p ──────────────────────────────────────
    for df in [train, test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # ── 8. [NEW] 카테고리 교차 피처 ─────────────────────────────
    for df in [train, test]:
        df["age_x_edu"]       = df["age_group"].astype(str) + "_" + df["education"].astype(str)
        df["gender_x_urban"]  = df["gender"].astype(str)    + "_" + df["urban"].astype(str)
        df["race_x_religion"] = df["race"].astype(str)      + "_" + df["religion"].astype(str)

    # ── 9. 범주형 → category dtype (LightGBM용) ─────────────────
    all_cats = CAT_COLS + CROSS_CAT_COLS
    for col in all_cats:
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
    print(f"추가 피처: age_x_edu / gender_x_urban / race_x_religion (교차cat)")
    print(f"           edu_num / age_num (서수수치)")
    print(f"           qe_std / qe_cv / qe_max_ratio (QE분산)")
    print(f"           tp_mean / tp_std / tp_range (tp패턴)")
    print(f"           mach_x_edu (교호작용수치)\n")

    oof_preds, test_preds, oof_auc = train_cv(X_train, y_train, X_test)

    os.makedirs("outputs", exist_ok=True)
    recorded = try_log(
        "lgbm", oof_auc, PARAMS, FIT_PARAMS,
        notes=(
            "lgbm_test2 [vs test1: interaction+QE분산+tp패턴+서수수치+하이퍼파라미터조정] "
            "사용컬럼: QA_raw20+tp_raw10+wr_raw13+wf_raw3+인구통계8개(hand제거) | "
            "파생: mach_score+q_response_std+q_extreme_ratio+vocab4개+delay_root10 | "
            "추가파생: age_x_edu+gender_x_urban+race_x_religion(교차cat) | "
            "       edu_num+age_num(서수수치) | "
            "       qe_std+qe_cv+qe_max_ratio(QE응답시간분산) | "
            "       tp_mean+tp_std+tp_range(tp응답패턴)+mach_x_edu(교호작용) | "
            "제거: QE_raw20+hand | "
            "하이퍼파라미터: num_leaves=127+lr=0.02+n_estimators=4000+colsample=0.7"
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
