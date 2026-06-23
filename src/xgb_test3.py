"""
src/xgb_test3.py
────────────────
XGBoost 실험 #3 — XGBoost 구조에 최적화된 FE

XGBoost 고유 특성과 그에 따른 FE 전략:

  ① depth-wise tree (vs LGBM leaf-wise)
     → max_depth=8로 더 깊은 분기 허용
     → gamma(분기 최소 이득 임계값) 추가로 불필요한 분기 억제

  ② NaN을 자체 방향으로 학습 (go_left/go_right)
     → tp 항목별 결측 플래그 10개 추가
       (tp_missing_cnt는 개수만 — 어느 항목이 missing인지 패턴 손실)
       e.g. tp03만 missing인 사람 vs tp07만 missing인 사람은 다른 패턴

  ③ 피처 서브샘플링: colsample_bytree + colsample_bylevel 동시 적용
     → 상관된 피처가 많을수록 효과적

  ④ 임계값 분기 → 다항 관계 표현 약함
     → mach_score², vocab_score² polynomial 추가
     → XGBoost는 x²를 스스로 찾기 위해 분기 2개가 필요하지만,
       명시하면 1개 분기로 처리 가능

  ⑤ 응답 품질 피처 (직선형 응답 감지)
     → qa_unique_cnt: QA 응답에서 사용한 고유값 수 (1~5)
       (1이면 모두 같은 값 → 불성실 응답 가능성)
     → qa_straightline: 동일 값 연속 최대 길이 / 20
       (0.5 이상이면 10문항 연속 동일값 → 응답 신뢰도 낮음)

  ⑥ 핵심 교호작용 수치 피처 2개 추가
     → age_x_mach: 나이대 × MACH 점수 (나이에 따른 마키아벨리즘 표현 차이)
     → edu_x_vocab: 교육수준 × vocab_score (교육+어휘력 복합 신호)

test2 대비 변경 요약:
  [추가] tp01~tp10 항목별 결측 플래그 (10개 binary)
  [추가] qa_unique_cnt, qa_straightline (응답 품질)
  [추가] mach_score², vocab_score² (polynomial)
  [추가] age_x_mach, edu_x_vocab (교호작용)
  [조정] max_depth 7→8, gamma=0.05, colsample_bylevel=0.8 추가

실행:
    python -m src.xgb_test3

출력 (신기록 시에만):
    outputs/xgb_test3_submission.csv
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

from src.tracker import try_log, show_leaderboard

# ================================================================
# 설정
# ================================================================
TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test_x.csv"
OUT_PATH   = "outputs/xgb_test3_submission.csv"
SEED       = 42
N_SPLITS   = 5

Q_COLS  = [f"Q{chr(ord('a')+i)}A" for i in range(20)]
QE_COLS = [f"Q{chr(ord('a')+i)}E" for i in range(20)]
TP_COLS = [f"tp{str(i).zfill(2)}" for i in range(1, 11)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

REVERSE_Q = ["QaA","QdA","QeA","QfA","QgA","QiA","QkA","QnA","QqA","QrA"]

CAT_COLS   = ["age_group","education","engnat","gender",
              "married","race","religion","urban"]
CROSS_COLS = ["age_x_edu", "gender_x_urban", "race_x_religion"]

# ================================================================
# 하이퍼파라미터 (XGBoost 최적화)
# ================================================================
PARAMS = dict(
    objective             = "binary:logistic",
    eval_metric           = "auc",
    tree_method           = "hist",
    n_estimators          = 4000,
    learning_rate         = 0.02,
    max_depth             = 8,            # depth-wise → test2(7)보다 깊게
    min_child_weight      = 5,
    gamma                 = 0.05,         # XGBoost 전용: 분기 최소 이득 임계값
    subsample             = 0.8,
    colsample_bytree      = 0.7,
    colsample_bylevel     = 0.8,          # XGBoost 전용: 레벨별 추가 서브샘플링
    reg_alpha             = 0.1,
    reg_lambda            = 5.0,
    scale_pos_weight      = 1.20665,
    random_state          = SEED,
    early_stopping_rounds = 200,
)
FIT_PARAMS = dict(
    verbose = 200,
)


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

    # [NEW] tp 항목별 결측 플래그 (NaN 변환 전에 만들어야 함)
    # tp_missing_cnt는 개수만 → 어느 항목이 missing인지 패턴 정보 손실
    for col in avail_tp:
        flag_name = f"{col}_miss"
        for df in [train, test]:
            df[flag_name] = (df[col] == 0).astype(np.float32)

    for col in avail_tp:
        train[col] = train[col].replace({0: np.nan, 7: np.nan})
        test[col]  = test[col].replace({0: np.nan, 7: np.nan})

    # tp 일관성 피처 (NaN 변환 후)
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

    # [NEW] QA 응답 품질 피처 (직선형 응답 감지)
    # XGBoost는 단일 피처 임계값 분기 → 불성실 응답 패턴을 명시해야 포착 가능
    for df in [train, test]:
        qa_vals = df[avail_q]
        # 몇 가지 다른 값을 사용했는가 (1이면 모두 같은 값 → 불성실)
        df["qa_unique_cnt"] = qa_vals.nunique(axis=1).astype(np.float32)
        # 동일 값 연속 최대 길이 / 전체 문항 수 (row-wise 계산)
        def max_run_ratio(row):
            vals = row.values
            max_run, cur_run = 1, 1
            for i in range(1, len(vals)):
                if vals[i] == vals[i-1]:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 1
            return max_run / len(vals)
        df["qa_straightline"] = qa_vals.apply(max_run_ratio, axis=1).astype(np.float32)

    # [NEW] polynomial 피처 (XGBoost: x² 표현에 분기 2개 필요 → 명시하면 1개로 처리)
    for df in [train, test]:
        df["mach_score_sq"]  = df["mach_score"] ** 2

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

    # [NEW] polynomial + 교호작용
    for df in [train, test]:
        df["vocab_score_sq"] = df["vocab_score"] ** 2
        df["edu_x_vocab"]    = df["edu_num"].fillna(3) * df["vocab_score"]
        df["age_x_mach"]     = df["age_num"].fillna(39) * df["mach_score"]

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

    # ── 8. 카테고리 교차 피처 ────────────────────────────────────
    for df in [train, test]:
        df["age_x_edu"]       = df["age_group"].astype(str) + "_" + df["education"].astype(str)
        df["gender_x_urban"]  = df["gender"].astype(str)    + "_" + df["urban"].astype(str)
        df["race_x_religion"] = df["race"].astype(str)      + "_" + df["religion"].astype(str)

    # ── 9. 범주형 → OrdinalEncoder (train만 fit, 누수 방지) ──────
    all_cats      = CAT_COLS + CROSS_COLS
    existing_cats = [c for c in all_cats if c in train.columns]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-1)
    enc.fit(train[existing_cats].astype(str))
    train[existing_cats] = enc.transform(train[existing_cats].astype(str))
    test[existing_cats]  = enc.transform(test[existing_cats].astype(str))

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

        model = xgb.XGBClassifier(**PARAMS)
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
    print(f"XGBoost 전용 추가:")
    print(f"  - tp항목별결측플래그: tp01_miss~tp10_miss (10개)")
    print(f"  - 응답품질: qa_unique_cnt, qa_straightline")
    print(f"  - polynomial: mach_score_sq, vocab_score_sq")
    print(f"  - 교호작용: age_x_mach, edu_x_vocab")
    print(f"  - 하이퍼파라미터: gamma=0.05, colsample_bylevel=0.8, max_depth=8\n")

    oof_preds, test_preds, oof_auc = train_cv(X_train, y_train, X_test)

    os.makedirs("outputs", exist_ok=True)
    recorded = try_log(
        "xgb", oof_auc, PARAMS, FIT_PARAMS,
        notes=(
            "xgb_test3 [XGBoost 구조 최적화 FE] "
            "사용컬럼: QA_raw20+tp_raw10+wr_raw13+wf_raw3+인구통계8개(hand제거) | "
            "파생(test2 유지): age_x_edu+gender_x_urban+race_x_religion+edu_num+age_num | "
            "       qe_std+qe_cv+qe_max_ratio+tp_mean+tp_std+tp_range+mach_x_edu | "
            "XGBoost전용추가: tp01~tp10_miss(항목별결측플래그10개) | "
            "       qa_unique_cnt+qa_straightline(불성실응답감지) | "
            "       mach_score_sq+vocab_score_sq(polynomial) | "
            "       age_x_mach+edu_x_vocab(교호작용) | "
            "하이퍼파라미터: depth=8+gamma=0.05+colsample_bylevel=0.8"
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
