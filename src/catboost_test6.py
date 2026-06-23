"""
src/catboost_test6.py
─────────────────────
CatBoost 실험 #6 — 완전 압축형 구조

전략: 모든 원시 컬럼을 의미 단위로 압축 → 노이즈 최소화 + 해석 가능성 최대화

피처 구성:
  [MACH / QA]
    mach_T          : Tactics 차원 평균 (QbA, QjA, QmA, QpA, QsA)
    mach_V          : Values/Morality 차원 평균 (QeA, QfA, QkA, QqA, QrA)
    mach_M          : Machinations/Cynicism 차원 평균 (QcA, QhA, QaA, QdA, QgA, QiA)
    mach_score      : 전체 MACH 평균
    A_extreme_ratio : 극단 응답(1·5) 비율 — 응답 확신도
    A_neutral_ratio : 중립 응답(3) 비율 — 우유부단/무관심 지표

  [Big5 / TP]
    big5_E          : Extraversion   (tp01 + (6-tp06))
    big5_A          : Agreeableness  (tp07 + (6-tp02))
    big5_C          : Conscientiousness (tp03 + (6-tp08))
    big5_N          : Neuroticism    ((6-tp04) + (6-tp09))
    big5_O          : Openness       (tp05 + (6-tp10))
    tp_std          : tp 응답 표준편차 — 성격 일관성 지표
    tp_notapplicable_cnt, tp_missing_cnt : 설문 성실도

  [응답 시간 / QE]
    delay_root10    : 총 응답시간 압축 (sum^0.1)
    qe_std          : 응답시간 표준편차 — 집중도 변동
    qe_fatigue      : 후반부(Q11~20) 평균 - 전반부(Q1~10) 평균 → 양수=피로, 음수=워밍업

  [어휘력 / WR·WF]
    real_accept_rate : 실제 단어 인식률 (vocab_real / 13)
    fake_accept_rate : 가짜 단어 수락률 (vocab_fake / 3) — 낮을수록 주의 깊음
    disc_ratio       : 변별력 (real_accept_rate - fake_accept_rate)

  [인구통계]
    age_group, education, gender, married, race, religion, urban
    edu_age_group   : education × age_group 결합 범주 (고학력+고령 등 교호작용 캡처)

실행:
    python -m src.catboost_test6

출력 (신기록 시에만):
    outputs/catboost_test6_submission.csv
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
OUT_PATH   = "outputs/catboost_test6_submission.csv"
SEED       = 42
N_SPLITS   = 5

Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
QE_COLS = [f"Q{chr(ord('a')+i)}E" for i in range(20)]
TP_COLS = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]

# MACH-IV 세부 차원 (역문항 보정 후 기준)
MACH_TACTICS  = ["QbA","QjA","QmA","QpA","QsA"]
MACH_VALUES   = ["QeA","QfA","QkA","QqA","QrA"]       # Morality/Values
MACH_CYNICISM = ["QcA","QhA","QaA","QdA","QgA","QiA"] # Machinations/Cynicism

# Big5 역문항 (tp raw 기준)
BIG5_REVERSE = {"tp02", "tp04", "tp06", "tp08", "tp09", "tp10"}

# 기본 범주형 (hand·engnat 제외)
CAT_COLS = ["age_group", "education", "gender", "married",
            "race", "religion", "urban", "edu_age_group"]

# ================================================================
# 하이퍼파라미터 (test1 기준 유지)
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

    # ── 1. tp 파생 피처 (NaN 변환 전) ──────────────────────────
    avail_tp = [c for c in TP_COLS if c in train.columns]
    for df in [train, test]:
        df["tp_notapplicable_cnt"] = (df[avail_tp] == 7).sum(axis=1)
        df["tp_missing_cnt"]       = (df[avail_tp] == 0).sum(axis=1)
    for col in avail_tp:
        train[col] = train[col].replace({0: np.nan, 7: np.nan})
        test[col]  = test[col].replace({0: np.nan, 7: np.nan})

    # ── 2. Big5 점수 + tp_std ───────────────────────────────────
    for df in [train, test]:
        tp = {c: df[c].fillna(df[c].median()) for c in avail_tp}
        df["big5_E"] = tp["tp01"] + (6 - tp["tp06"])
        df["big5_A"] = tp["tp07"] + (6 - tp["tp02"])
        df["big5_C"] = tp["tp03"] + (6 - tp["tp08"])
        df["big5_N"] = (6 - tp["tp04"]) + (6 - tp["tp09"])
        df["big5_O"] = tp["tp05"] + (6 - tp["tp10"])
        # tp 응답 표준편차 (역문항 보정 후)
        tp_corrected = pd.DataFrame({
            c: (6 - df[c]) if c in BIG5_REVERSE else df[c]
            for c in avail_tp
        })
        df["tp_std"] = tp_corrected.std(axis=1)

    # tp raw 제거 (Big5로 압축됨)
    train = train.drop(columns=avail_tp)
    test  = test.drop(columns=avail_tp)

    # ── 3. 인구통계 무응답 → NaN ───────────────────────────────
    for col in ["education", "married", "urban"]:
        if col in train.columns:
            train[col] = train[col].replace(0, np.nan)
            test[col]  = test[col].replace(0, np.nan)

    # ── 4. edu_age_group 결합 범주 ────────────────────────────
    for df in [train, test]:
        edu = df["education"].astype(str).fillna("nan")
        age = df["age_group"].astype(str).fillna("nan")
        df["edu_age_group"] = edu + "_" + age

    # ── 5. 역문항 리버스 코딩 ─────────────────────────────────
    for col in REVERSE_Q:
        if col in train.columns:
            train[col] = 6 - train[col]
            test[col]  = 6 - test[col]

    # ── 6. MACH 압축 피처 ─────────────────────────────────────
    avail_q = [c for c in Q_COLS if c in train.columns]
    for df in [train, test]:
        t_cols = [c for c in MACH_TACTICS  if c in df.columns]
        v_cols = [c for c in MACH_VALUES   if c in df.columns]
        m_cols = [c for c in MACH_CYNICISM if c in df.columns]

        df["mach_T"]          = df[t_cols].mean(axis=1) if t_cols else np.nan
        df["mach_V"]          = df[v_cols].mean(axis=1) if v_cols else np.nan
        df["mach_M"]          = df[m_cols].mean(axis=1) if m_cols else np.nan
        df["mach_score"]      = df[avail_q].mean(axis=1)
        df["A_extreme_ratio"] = ((df[avail_q] == 1) | (df[avail_q] == 5)).sum(axis=1) / len(avail_q)
        df["A_neutral_ratio"] = (df[avail_q] == 3).sum(axis=1) / len(avail_q)

    # QA raw 제거 (3차원 + 전체 점수로 압축됨)
    train = train.drop(columns=[c for c in avail_q if c in train.columns])
    test  = test.drop(columns=[c for c in avail_q if c in test.columns])

    # ── 7. 어휘력 압축 피처 ───────────────────────────────────
    avail_wr = [c for c in WR_COLS if c in train.columns]
    avail_wf = [c for c in WF_COLS if c in train.columns]
    for df in [train, test]:
        vocab_real = df[avail_wr].sum(axis=1)
        vocab_fake = df[avail_wf].sum(axis=1)
        df["real_accept_rate"] = vocab_real / len(avail_wr)          # 실제 단어 인식률
        df["fake_accept_rate"] = vocab_fake / len(avail_wf)          # 가짜 단어 수락률
        df["disc_ratio"]       = df["real_accept_rate"] - df["fake_accept_rate"]  # 변별력

    # WR/WF raw 제거
    train = train.drop(columns=avail_wr + avail_wf)
    test  = test.drop(columns=avail_wr + avail_wf)

    # ── 8. QE 압축 피처 ───────────────────────────────────────
    avail_qe = [c for c in QE_COLS if c in train.columns]
    for df in [train, test]:
        qe_clip = df[avail_qe].clip(lower=0)
        first_half = [c for c in avail_qe[:10]]
        second_half = [c for c in avail_qe[10:]]

        df["delay_root10"] = np.power(qe_clip.sum(axis=1), 0.1)
        df["qe_std"]       = qe_clip.std(axis=1)
        # 피로도: 후반 평균 - 전반 평균 (양수 = 후반으로 갈수록 느려짐)
        df["qe_fatigue"]   = (
            df[second_half].clip(lower=0).mean(axis=1) -
            df[first_half].clip(lower=0).mean(axis=1)
        )

    # QE raw 제거
    train = train.drop(columns=avail_qe)
    test  = test.drop(columns=avail_qe)

    # ── 9. familysize log1p ───────────────────────────────────
    for df in [train, test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # ── 10. 범주형 → 문자열 (CatBoost용) ─────────────────────
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

    os.makedirs("outputs", exist_ok=True)
    recorded = try_log(
        "catboost", oof_auc, PARAMS, FIT_PARAMS,
        notes=(
            "test6 [vs test1: 모든 raw 제거 후 완전 압축형] "
            "사용컬럼: 인구통계7개+edu_age_group결합범주 | "
            "파생: mach_T+mach_V+mach_M+mach_score+A_extreme_ratio+A_neutral_ratio+big5_E/A/C/N/O+tp_std+delay_root10+qe_std+qe_fatigue+real_accept_rate+fake_accept_rate+disc_ratio | "
            "제거: QA_raw20→3차원압축, tp_raw10→Big5압축, WR_raw13+WF_raw3→accept_rate압축, QE_raw20→3개압축, hand+engnat"
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
