"""
src/catboost_test8.py
─────────────────────
CatBoost 실험 #8 — 인구통계 중심 구조

전략:
  - age_group + education이 투표율의 핵심 예측 변수
    (고령일수록, 고학력일수록 투표율 높음 — 실증 연구 일관된 결과)
  - QE 전체 제거 (환경 노이즈)
  - QA는 mach_score 1개만 (약한 보조 역할, 20개 raw로 과대 대표 방지)

인구통계 강화:
  age_num       : age_group 수치화 (1~7) — 연속 신호
  edu_num       : education 수치화 (1~4) — 연속 신호
  age_x_edu     : age_num × edu_num — 핵심 교호작용 (고령+고학력)
  age_edu_cat   : age_group + "_" + education 결합 범주 — CatBoost 패턴 학습용
  familysize    : log1p 변환 (가족 구성원 수 — 사회적 연결망 proxy)

심리 (보조):
  mach_score    : MACH 전체 평균 1개만 (QA 20개 raw 대신 요약)
  big5_E/A/C/N/O: Big5 5개 점수 (tp 10개 → 5개로 압축)

어휘력 (보조):
  real_accept_rate, fake_accept_rate, disc_ratio

기타 인구통계 (범주형 유지):
  gender, married, race, religion, urban, engnat

실행:
    python -m src.catboost_test8

출력 (신기록 시에만):
    outputs/catboost_test8_submission.csv
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
OUT_PATH   = "outputs/catboost_test8_submission.csv"
SEED       = 42
N_SPLITS   = 5

Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
QE_COLS = [f"Q{chr(ord('a')+i)}E" for i in range(20)]
TP_COLS = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]

# age_group → 수치 (투표율과 단조 관계)
AGE_MAP = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
# education → 수치 (1=초졸, 2=중졸, 3=고졸, 4=대졸 이상)
EDU_MAP = {"1": 1, "2": 2, "3": 3, "4": 4}

CAT_COLS = [
    "age_group", "education",        # 핵심
    "gender", "married", "race",     # 보조 인구통계
    "religion", "urban", "engnat",   # 보조 인구통계
    "age_edu_cat",                   # 결합 범주
]

# ================================================================
# 하이퍼파라미터
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

    for df in [train, test]:
        if "index" in df.columns:
            df.drop(columns=["index"], inplace=True)

    # familysize 이상치 제거 (train만)
    outlier_idx = train[train["familysize"] > 50].index
    if len(outlier_idx):
        print(f"familysize 이상치 제거: {len(outlier_idx)}행")
        train = train.drop(index=outlier_idx).reset_index(drop=True)

    # ── 1. QE 전체 제거 (노이즈) ────────────────────────────
    for df in [train, test]:
        df.drop(columns=[c for c in QE_COLS if c in df.columns], inplace=True)

    # ── 2. 인구통계 강화 ─────────────────────────────────────
    for df in [train, test]:
        # 수치형 연속 변수
        df["age_num"] = df["age_group"].astype(str).map(AGE_MAP).fillna(3)
        df["edu_num"] = df["education"].replace(0, np.nan)
        df["edu_num"] = pd.to_numeric(df["edu_num"], errors="coerce").fillna(2)

        # 핵심 교호작용: 고령 × 고학력
        df["age_x_edu"] = df["age_num"] * df["edu_num"]

        # 결합 범주 (CatBoost 패턴 학습)
        df["age_edu_cat"] = df["age_group"].astype(str) + "_" + df["education"].astype(str)

    # ── 3. 인구통계 무응답 → NaN ─────────────────────────────
    for col in ["education", "married", "urban", "engnat"]:
        if col in train.columns:
            train[col] = train[col].replace(0, np.nan)
            test[col]  = test[col].replace(0, np.nan)

    # ── 4. tp 파생 피처 → Big5 (NaN 변환 전 카운트) ──────────
    avail_tp = [c for c in TP_COLS if c in train.columns]
    for df in [train, test]:
        df["tp_notapplicable_cnt"] = (df[avail_tp] == 7).sum(axis=1)
        df["tp_missing_cnt"]       = (df[avail_tp] == 0).sum(axis=1)
    for col in avail_tp:
        train[col] = train[col].replace({0: np.nan, 7: np.nan})
        test[col]  = test[col].replace({0: np.nan, 7: np.nan})

    for df in [train, test]:
        tp = {c: df[c].fillna(df[c].median()) for c in avail_tp}
        df["big5_E"] = tp["tp01"] + (6 - tp["tp06"])
        df["big5_A"] = tp["tp07"] + (6 - tp["tp02"])
        df["big5_C"] = tp["tp03"] + (6 - tp["tp08"])
        df["big5_N"] = (6 - tp["tp04"]) + (6 - tp["tp09"])
        df["big5_O"] = tp["tp05"] + (6 - tp["tp10"])

    # tp raw 제거
    train = train.drop(columns=avail_tp)
    test  = test.drop(columns=avail_tp)

    # ── 5. QA → mach_score 1개만 (약한 보조) ────────────────
    avail_q = [c for c in Q_COLS if c in train.columns]
    for col in REVERSE_Q:
        if col in train.columns:
            train[col] = 6 - train[col]
            test[col]  = 6 - test[col]
    for df in [train, test]:
        df["mach_score"] = df[avail_q].mean(axis=1)

    # QA raw 20개 제거
    train = train.drop(columns=[c for c in avail_q if c in train.columns])
    test  = test.drop(columns=[c for c in avail_q if c in test.columns])

    # ── 6. 어휘력 (보조) ─────────────────────────────────────
    avail_wr = [c for c in WR_COLS if c in train.columns]
    avail_wf = [c for c in WF_COLS if c in train.columns]
    for df in [train, test]:
        vocab_real = df[avail_wr].sum(axis=1)
        vocab_fake = df[avail_wf].sum(axis=1)
        df["real_accept_rate"] = vocab_real / len(avail_wr)
        df["fake_accept_rate"] = vocab_fake / len(avail_wf)
        df["disc_ratio"]       = df["real_accept_rate"] - df["fake_accept_rate"]

    train = train.drop(columns=avail_wr + avail_wf)
    test  = test.drop(columns=avail_wr + avail_wf)

    # ── 7. familysize log1p ───────────────────────────────────
    for df in [train, test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # ── 8. 범주형 → 문자열 ────────────────────────────────────
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
    print(f"핵심: age_num, edu_num, age_x_edu, age_edu_cat")
    print(f"보조: mach_score(1개), big5×5, real/fake_accept_rate, disc_ratio")
    print(f"제거: QE 전체, QA raw 20개, tp raw 10개, WR/WF raw\n")

    oof_preds, test_preds, oof_auc = train_cv(X_train, y_train, X_test, cat_idx)

    os.makedirs("outputs", exist_ok=True)
    recorded = try_log(
        "catboost", oof_auc, PARAMS, FIT_PARAMS,
        notes=(
            "test8 [vs test1: 인구통계중심, QA/tp/WR/WF/QE raw 전부 제거] "
            "사용컬럼: 인구통계8개(hand제외)+age_edu_cat결합범주 | "
            "파생: age_num+edu_num+age_x_edu(핵심교호작용)+mach_score1개+big5_E/A/C/N/O+tp_notapplicable/missing_cnt+real/fake_accept_rate+disc_ratio | "
            "제거: QA_raw20→mach_score1개, tp_raw10→Big5압축, WR_raw13+WF_raw3→accept_rate, QE_raw20전체, hand"
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
