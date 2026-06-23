"""
src/preprocess.py
─────────────────
부스팅 모델 전용 전처리 파이프라인.

파이프라인 순서 주의:
  1. tp 파생 피처 생성  ← NaN 변환 전에 먼저
  2. tp 0/7 → NaN
  3. 인구통계 0 → NaN
  4. 역문항 리버스 코딩
  5. mach_score, q_response_std, q_extreme_ratio
  6. 어휘력 피처
  7. QE → delay_root10 (raw 20개 제거)
  8. familysize log1p
  9. 범주형 → 문자열 (CatBoost용)

실험 결론:
  - delay_root10 단일 피처 > QE 5개 요약통계
  - 상관계수 기반 피처 제거 금지 (OOF -0.019)
  - QA PCA 치환 금지 (item-level 패턴 소실)
  - Big5 추가 FE 금지 (OOF -0.0003)
"""

import numpy as np
import pandas as pd
from src.config import Q_COLS, QE_COLS, TP_COLS, WR_COLS, WF_COLS, REVERSE_Q, CAT_COLS


def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    train, test = train.copy(), test.copy()

    # index 제거
    for df in [train, test]:
        if "index" in df.columns:
            df.drop(columns=["index"], inplace=True)

    # familysize 이상치 제거 (train만)
    outlier_idx = train[train["familysize"] > 50].index
    if len(outlier_idx):
        train = train.drop(index=outlier_idx).reset_index(drop=True)

    # 1. tp 파생 피처 (NaN 변환 전)
    avail_tp = [c for c in TP_COLS if c in train.columns]
    for df in [train, test]:
        df["tp_notapplicable_cnt"] = (df[avail_tp] == 7).sum(axis=1)
        df["tp_missing_cnt"]       = (df[avail_tp] == 0).sum(axis=1)
    for col in avail_tp:
        train[col] = train[col].replace({0: np.nan, 7: np.nan})
        test[col]  = test[col].replace({0: np.nan, 7: np.nan})

    # 2. 인구통계 무응답 → NaN
    for col in ["education","engnat","hand","married","urban"]:
        if col in train.columns:
            train[col] = train[col].replace(0, np.nan)
            test[col]  = test[col].replace(0, np.nan)

    # 3. 역문항 리버스 코딩
    for col in REVERSE_Q:
        if col in train.columns:
            train[col] = 6 - train[col]
            test[col]  = 6 - test[col]

    # 4. MACH 피처
    avail_q = [c for c in Q_COLS if c in train.columns]
    for df in [train, test]:
        df["mach_score"]      = df[avail_q].mean(axis=1)
        df["q_response_std"]  = df[avail_q].std(axis=1)
        df["q_extreme_ratio"] = (
            (df[avail_q] == 1) | (df[avail_q] == 5)
        ).sum(axis=1) / len(avail_q)

    # 5. 어휘력 피처
    avail_wr = [c for c in WR_COLS if c in train.columns]
    avail_wf = [c for c in WF_COLS if c in train.columns]
    for df in [train, test]:
        df["vocab_real"]     = df[avail_wr].sum(axis=1)
        df["vocab_fake"]     = df[avail_wf].sum(axis=1)
        df["vocab_score"]    = df["vocab_real"] - df["vocab_fake"] * 2
        df["vocab_accuracy"] = df["vocab_real"] / (df["vocab_real"] + df["vocab_fake"] + 1e-6)

    # 6. QE → delay_root10 (raw 20개 제거)
    avail_qe = [c for c in QE_COLS if c in train.columns]
    for df in [train, test]:
        df["delay_root10"] = np.power(df[avail_qe].sum(axis=1).clip(lower=0), 0.1)
    train = train.drop(columns=avail_qe)
    test  = test.drop(columns=avail_qe)

    # 7. familysize log1p
    for df in [train, test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # 8. 범주형 → 문자열 (CatBoost용)
    for col in CAT_COLS:
        if col in train.columns:
            train[col] = train[col].astype(str)
            test[col]  = test[col].astype(str)

    return train, test
