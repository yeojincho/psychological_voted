"""
src/catboost_test2.py
─────────────────────
CatBoost 실험 #2

test1 대비 변경 사항:
  - QA raw 20개 제거 → MACH 세부 차원 점수 + 추가 파생 피처로 대체
  - QE raw 20개 제거 → delay_root10 + time_cv(응답 속도 변동계수)로 대체
  - 투표와 무관한 컬럼 제거: engnat, hand
  - education, married 명시적으로 유지 (무응답 0 → NaN 처리 후 CatBoost에 전달)

MACH-IV 세부 차원 (표준 구분 기준):
  - mach_tactics  : 조작 전술 (Qb, Qj, Qm, Qp, Qs)
  - mach_morality : 도덕성 (Qe, Qf, Qk, Qq, Qr) ← 역문항 다수
  - mach_cynicism : 냉소적 세계관 (Qc, Qh, Qo 외)
  - 나머지 비공개(secret) 문항은 mach_score 전체에 반영됨

실행:
    python -m src.catboost_test2

출력 (신기록 시에만):
    outputs/catboost_test2_submission.csv
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

from src.tracker import try_log, show_leaderboard

# ================================================================
# 설정
# ================================================================
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test_x.csv"
OUT_PATH   = "outputs/catboost_test2_submission.csv"
SEED       = 42
N_SPLITS   = 5

Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
QE_COLS = [f"Q{chr(ord('a')+i)}E" for i in range(20)]
TP_COLS = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]

# MACH-IV 세부 차원 문항 (역문항 보정 후 기준)
MACH_TACTICS  = ["QbA","QjA","QmA","QpA","QsA"]           # 조작 전술
MACH_MORALITY = ["QeA","QfA","QkA","QqA","QrA"]           # 도덕성 (역문항 포함)
MACH_CYNICISM = ["QcA","QhA","QaA","QdA","QgA","QiA"]     # 냉소적 세계관

# 범주형 컬럼 (CatBoost 문자열 처리)
CAT_COLS = ["age_group","education","married","gender","race","religion","urban"]
# ※ engnat, hand 제거 (투표 예측과 낮은 연관성)

# 제거할 컬럼
DROP_COLS = ["engnat", "hand"]

# ================================================================
# 하이퍼파라미터 (여기를 수정해서 실험)
# ================================================================
PARAMS = dict(
    objective        = "Logloss",
    eval_metric      = "AUC",
    iterations       = 3000,
    learning_rate    = 0.03,
    depth            = 6,
    l2_leaf_reg      = 5,
    bootstrap_type   = "Bernoulli",
    subsample        = 0.8,
    scale_pos_weight = 1.20665,
    random_seed      = SEED,
    allow_writing_files = False,
)
FIT_PARAMS = dict(
    early_stopping_rounds = 200,
    verbose               = 200,
)

# ================================================================
# 피처 엔지니어링
# ================================================================
def preprocess(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train, test = train_raw.copy(), test_raw.copy()

    # index 제거
    for df in [train, test]:
        if "index" in df.columns:
            df.drop(columns=["index"], inplace=True)

    # 투표와 무관한 컬럼 제거
    for df in [train, test]:
        df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # familysize 이상치 제거 (train만)
    outlier_idx = train[train["familysize"] > 50].index
    if len(outlier_idx):
        print(f"familysize 이상치 제거: {len(outlier_idx)}행")
        train = train.drop(index=outlier_idx).reset_index(drop=True)

    # ── tp 파생 피처 (NaN 변환 전) ──────────────────────────
    avail_tp = [c for c in TP_COLS if c in train.columns]
    for df in [train, test]:
        df["tp_notapplicable_cnt"] = (df[avail_tp] == 7).sum(axis=1)
        df["tp_missing_cnt"]       = (df[avail_tp] == 0).sum(axis=1)
    for col in avail_tp:
        train[col] = train[col].replace({0: np.nan, 7: np.nan})
        test[col]  = test[col].replace({0: np.nan, 7: np.nan})

    # ── 인구통계 무응답 → NaN ────────────────────────────────
    for col in ["education","married","urban"]:
        if col in train.columns:
            train[col] = train[col].replace(0, np.nan)
            test[col]  = test[col].replace(0, np.nan)

    # ── 역문항 리버스 코딩 (10개) ───────────────────────────
    for col in REVERSE_Q:
        if col in train.columns:
            train[col] = 6 - train[col]
            test[col]  = 6 - test[col]

    # ── QA 파생 피처 (raw 제거 전) ──────────────────────────
    avail_q = [c for c in Q_COLS if c in train.columns]
    for df in [train, test]:
        # 전체 MACH 점수
        df["mach_score"]      = df[avail_q].mean(axis=1)
        df["q_response_std"]  = df[avail_q].std(axis=1)
        df["q_extreme_ratio"] = (
            (df[avail_q] == 1) | (df[avail_q] == 5)
        ).sum(axis=1) / len(avail_q)

        # 고마키아벨리즘 응답 비율 (4 이상)
        df["mach_high_ratio"] = (df[avail_q] >= 4).sum(axis=1) / len(avail_q)

        # MACH 세부 차원 점수
        tactics  = [c for c in MACH_TACTICS  if c in df.columns]
        morality = [c for c in MACH_MORALITY if c in df.columns]
        cynicism = [c for c in MACH_CYNICISM if c in df.columns]
        if tactics:  df["mach_tactics"]  = df[tactics].mean(axis=1)
        if morality: df["mach_morality"] = df[morality].mean(axis=1)
        if cynicism: df["mach_cynicism"] = df[cynicism].mean(axis=1)

    # QA raw 20개 제거
    train = train.drop(columns=[c for c in avail_q if c in train.columns])
    test  = test.drop(columns=[c for c in avail_q if c in test.columns])

    # ── 어휘력 피처 ──────────────────────────────────────────
    avail_wr = [c for c in WR_COLS if c in train.columns]
    avail_wf = [c for c in WF_COLS if c in train.columns]
    for df in [train, test]:
        df["vocab_real"]     = df[avail_wr].sum(axis=1)
        df["vocab_fake"]     = df[avail_wf].sum(axis=1)
        df["vocab_score"]    = df["vocab_real"] - df["vocab_fake"] * 2
        df["vocab_accuracy"] = df["vocab_real"] / (df["vocab_real"] + df["vocab_fake"] + 1e-6)

    # wr/wf raw 제거
    train = train.drop(columns=avail_wr + avail_wf)
    test  = test.drop(columns=avail_wr + avail_wf)

    # ── QE 파생 피처 (raw 제거 전) ──────────────────────────
    avail_qe = [c for c in QE_COLS if c in train.columns]
    for df in [train, test]:
        qe_clipped = df[avail_qe].clip(lower=0)
        qe_sum  = qe_clipped.sum(axis=1)
        qe_mean = qe_clipped.mean(axis=1)
        qe_std  = qe_clipped.std(axis=1)

        # 총 응답시간 압축 (heavy-tail 보정)
        df["delay_root10"] = np.power(qe_sum, 0.1)

        # 응답 속도 변동계수 (일관성 측정): std/mean, 낮을수록 일정한 속도
        df["time_cv"] = qe_std / (qe_mean + 1e-6)

    # QE raw 20개 제거
    train = train.drop(columns=[c for c in avail_qe if c in train.columns])
    test  = test.drop(columns=[c for c in avail_qe if c in test.columns])

    # ── familysize log1p ─────────────────────────────────────
    for df in [train, test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # ── 범주형 → 문자열 (CatBoost용) ────────────────────────
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
    print(f"피처 목록: {list(X_train.columns)}\n")

    oof_preds, test_preds, oof_auc = train_cv(X_train, y_train, X_test, cat_idx)

    # 신기록일 때만 submission 저장 + 실험 기록
    os.makedirs("outputs", exist_ok=True)
    recorded = try_log(
        "catboost", oof_auc, PARAMS, FIT_PARAMS,
        notes=(
            "test2 [vs test1: QA/WR/WF raw 제거, engnat/hand 제거] "
            "사용컬럼: tp_raw10+인구통계7개(engnat/hand제거) | "
            "파생: mach_score+mach_tactics+mach_morality+mach_cynicism+mach_high_ratio+q_response_std+q_extreme_ratio+vocab4개+delay_root10+time_cv | "
            "제거: QA_raw20→MACH차원으로압축, WR_raw13+WF_raw3→vocab집계, QE_raw20, engnat, hand"
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
