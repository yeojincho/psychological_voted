"""
common_fe_v2.py  —  실제 코드 기반 공통 FE 파이프라인
═══════════════════════════════════════════════════════
nn.py (AUC 0.7745), boosting.py (AUC 0.7729), liner.py (AUC 0.7649)
세 코드를 분석하여 정확히 재현한 공통 + 모델별 파이프라인

사용법:
  from common_fe_v2 import (
      load_common,          # 공통 데이터 로드 + 이상치 제거
      prepare_for_nn,       # NN 전용 FE (nn.py 재현)
      prepare_for_catboost, # CatBoost 전용 FE (boosting.py 재현)
      prepare_for_linear,   # Linear 전용 FE (liner.py 재현)
      OOF_FOLDS_5,          # CatBoost/Linear용 5-fold
      save_npy, load_npy,   # npy 유틸리티
      check_diversity,      # 앙상블 다양성 분석
  )
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────
# 상수 (세 코드 공통)
# ─────────────────────────────────────────────────────────
SEED = 42
QUESTION_KEYS = list("abcdefghijklmnopqrst")
QA_COLS = [f"Q{k}A" for k in QUESTION_KEYS]
QE_COLS = [f"Q{k}E" for k in QUESTION_KEYS]
TP_COLS = [f"tp0{i}" for i in range(1, 10)] + ["tp10"]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(0, 8)]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(0, 8)]

CATEGORICAL_COLS = [
    "age_group", "education", "engnat",
    "gender", "married", "race", "religion", "urban",
]

# NN 전용
NN_DROP_LIST = QE_COLS + ["index", "hand"]
NN_CONT_STANDARDIZE_COLS = [
    "delay_root10", "A_mean", "A_std",
    "A_extreme_ratio", "A_neutral_ratio",

    # ===================================================================================================
    # [추가] WR/WF SUMMARY 표준화 대상 컬럼
    # raw_plus_summary baseline 반영
    # =====================================================
    "wr_mean", "wr_sum", "wf_mean", "wf_sum",
    "disc_score", "overclaim_ratio", "wr_std", "wf_std",
]

# Linear 전용
LINEAR_TOP_QA = ["QqA", "QtA", "QbA", "QpA", "QjA", "QmA", "QoA", "QkA"]
LINEAR_TOP_TP = ["tp07", "tp03", "tp09"]

# Fold 설정
OOF_FOLDS_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# ═════════════════════════════════════════════════════════
# 1. 공통 로드
# ═════════════════════════════════════════════════════════
def load_common(train_path, test_path, sample_sub_path=None):
    """
    공통 데이터 로드 + familysize > 50 이상치 제거 + target 생성

    Returns:
        train_df, test_df, y (ndarray, float32), sample_sub (or None)
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path) if sample_sub_path else None

    # target: voted 1/2 → 0/1
    y = (train["voted"] - 1).astype(np.float32).to_numpy()

    # familysize 이상치 제거 (세 코드 공통)
    drop_idx = train[train["familysize"] > 50].index
    train = train.drop(index=drop_idx).reset_index(drop=True)
    y = np.delete(y, drop_idx.to_numpy())

    print(f"데이터 로드 완료")
    print(f"  Train: {train.shape} (familysize>50 {len(drop_idx)}건 제거)")
    print(f"  Test:  {test.shape}")
    print(f"  Target: 0(투표)={int((y==0).sum())}, 1(미투표)={int((y==1).sum())}")

    return train, test, y, sample_sub


# ═════════════════════════════════════════════════════════
# 2. NN 전용 FE  (nn.py 정확 재현)
# ═════════════════════════════════════════════════════════
def prepare_for_nn(train_df, test_df):
    """
    nn.py의 FE를 정확히 재현.

    핵심 특징 (이전 common_fe.py와 다른 점):
      - QA 역채점 안 함 → (x-3)/2 고정 스케일링
      - OCEAN trait 합산 안 함
      - 버킷 피처 안 씀
      - hand, index 제거
      - delay_root10, A_mean, A_std, A_extreme_ratio, A_neutral_ratio 추가
      - 연속형 summary는 fold 내부에서 표준화 (여기서 안 함)
      - OHE도 fold 내부에서 fit (여기서 안 함)

    Returns:
        X_train_df, X_test_df: FE 적용된 DataFrame (스케일링/OHE 전)
        numeric_cols: 수치형 컬럼 리스트
        existing_cat_cols: 범주형 컬럼 리스트
    """
    X_train = train_df.copy()
    X_test = test_df.copy()

    # (1) delay_root10 추가
    X_train = _add_delay_root10(X_train)
    X_test = _add_delay_root10(X_test)

    # (2) QA summary 추가
    X_train = _add_qa_summary(X_train)
    X_test = _add_qa_summary(X_test)

    # ===========================================================================================================
    # [추가] WR/WF SUMMARY 생성 - kdu
    # raw_plus_summary baseline 반영
    # =====================================================
    X_train = _add_wrwf_summary(X_train)
    X_test = _add_wrwf_summary(X_test)

    # (3) 불필요 컬럼 제거 + voted 제거
    drop_train = [c for c in NN_DROP_LIST + ["voted"] if c in X_train.columns]
    drop_test = [c for c in NN_DROP_LIST if c in X_test.columns]
    X_train = X_train.drop(columns=drop_train)
    X_test = X_test.drop(columns=drop_test)

    # (4) 범주형/수치형 분리
    existing_cat = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    numeric = [c for c in X_train.columns if c not in existing_cat]

    # (5) 범주형을 문자열로 변환
    for col in existing_cat:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # (6) 수치형 고정 변환 (nn.py 방식: 학습 통계 불필요)
    X_train[numeric] = _apply_nn_fixed_transforms(X_train[numeric])
    X_test[numeric] = _apply_nn_fixed_transforms(X_test[numeric])

    print(f"\nNN FE 완료:")
    print(f"  수치형 {len(numeric)}개, 범주형 {len(existing_cat)}개")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, numeric, existing_cat


def build_nn_fold_matrices(X_train_df, X_test_df, numeric_cols,
                           existing_cat_cols, train_idx, valid_idx):
    """
    nn.py의 build_fold_matrices() 정확 재현.
    fold train에만 OHE fit + 연속형 summary 표준화.

    Returns:
        X_tr_np, X_va_np, X_te_np (numpy float32 arrays), input_dim
    """
    X_tr = X_train_df.iloc[train_idx].copy()
    X_va = X_train_df.iloc[valid_idx].copy()
    X_te = X_test_df.copy()

    # 수치형
    X_tr_num = X_tr[numeric_cols].copy()
    X_va_num = X_va[numeric_cols].copy()
    X_te_num = X_te[numeric_cols].copy()

    # fold train 기준 연속형 summary 표준화
    fold_std_cols = [c for c in NN_CONT_STANDARDIZE_COLS if c in X_tr_num.columns]
    for col in fold_std_cols:
        mean_ = X_tr_num[col].mean()
        std_ = X_tr_num[col].std()
        if pd.isna(std_) or std_ < 1e-6:
            std_ = 1.0
        X_tr_num[col] = (X_tr_num[col] - mean_) / std_
        X_va_num[col] = (X_va_num[col] - mean_) / std_
        X_te_num[col] = (X_te_num[col] - mean_) / std_

    # 범주형 OHE (fold train에만 fit)
    if existing_cat_cols:
        encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32,
        )
        X_tr_cat = encoder.fit_transform(X_tr[existing_cat_cols])
        X_va_cat = encoder.transform(X_va[existing_cat_cols])
        X_te_cat = encoder.transform(X_te[existing_cat_cols])
    else:
        n_tr, n_va, n_te = len(X_tr), len(X_va), len(X_te)
        X_tr_cat = np.empty((n_tr, 0), dtype=np.float32)
        X_va_cat = np.empty((n_va, 0), dtype=np.float32)
        X_te_cat = np.empty((n_te, 0), dtype=np.float32)

    X_tr_np = np.concatenate([X_tr_num.to_numpy(np.float32), X_tr_cat], axis=1)
    X_va_np = np.concatenate([X_va_num.to_numpy(np.float32), X_va_cat], axis=1)
    X_te_np = np.concatenate([X_te_num.to_numpy(np.float32), X_te_cat], axis=1)

    return X_tr_np, X_va_np, X_te_np, X_tr_np.shape[1]


# ═════════════════════════════════════════════════════════
# 3. CatBoost 전용 FE  (boosting.py 정확 재현)
# ═════════════════════════════════════════════════════════
def prepare_for_catboost(train_df, test_df):
    """
    boosting.py의 FE를 정확히 재현.

    핵심 특징:
      - QA summary 피처 안 씀! delay_root10만 추가
      - hand 컬럼 유지 (NN과 다름)
      - 범주형은 문자열로 변환 → cat_features로 전달
      - familysize → log1p

    Returns:
        X_train, X_test (DataFrames), cat_feature_indices
    """
    X_train = train_df.copy()
    X_test = test_df.copy()

    # (1) delay_root10 추가
    X_train = _add_delay_root10(X_train)
    X_test = _add_delay_root10(X_test)

    # (2) familysize → log1p
    for df in [X_train, X_test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # (3) 범주형 → 문자열
    for col in CATEGORICAL_COLS:
        for df in [X_train, X_test]:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # (4) 불필요 컬럼 제거 (QE raw + index + voted)
    drop_train = [c for c in QE_COLS + ["voted", "index"] if c in X_train.columns]
    drop_test = [c for c in QE_COLS + ["index"] if c in X_test.columns]
    X_train = X_train.drop(columns=drop_train, errors="ignore")
    X_test = X_test.drop(columns=drop_test, errors="ignore")

    # (5) cat_features 인덱스
    cat_indices = [X_train.columns.get_loc(c) for c in CATEGORICAL_COLS
                   if c in X_train.columns]

    print(f"\nCatBoost FE 완료:")
    print(f"  피처 {X_train.shape[1]}개 (QA summary 없음, delay_root10만 추가)")
    print(f"  cat_features {len(cat_indices)}개")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, cat_indices


# ═════════════════════════════════════════════════════════
# 4. Linear 전용 FE  (liner.py 정확 재현)
# ═════════════════════════════════════════════════════════
def prepare_for_linear(train_df, test_df):
    """
    liner.py의 FE를 정확히 재현.

    핵심 특징:
      - QA_mean, QA_std, QA_top_mean, A_low_ratio, A_high_ratio
      - TP_top_mean
      - interaction: edu_married_group, edu_age_group
      - OHE (기본 범주형 + interaction 범주형)
      - familysize → log1p

    Returns:
        X_train, X_test (OHE 적용된 DataFrames)
    """
    X_train = train_df.copy()
    X_test = test_df.copy()

    # (1) delay_root10
    X_train = _add_delay_root10(X_train)
    X_test = _add_delay_root10(X_test)

    # (2) QA features (liner.py 방식)
    for df in [X_train, X_test]:
        df["QA_mean"] = df[QA_COLS].mean(axis=1)
        df["QA_std"] = df[QA_COLS].std(axis=1)
        df["QA_top_mean"] = df[LINEAR_TOP_QA].mean(axis=1)
        df["A_low_ratio"] = (df[QA_COLS] <= 2).sum(axis=1) / len(QA_COLS)
        df["A_high_ratio"] = (df[QA_COLS] >= 4).sum(axis=1) / len(QA_COLS)

    # (3) TP features
    for df in [X_train, X_test]:
        df["TP_top_mean"] = df[LINEAR_TOP_TP].mean(axis=1)

    # (4) Interaction features
    for df in [X_train, X_test]:
        df["edu_married_group"] = (
            df["education"].astype(str) + "_" + df["married"].astype(str)
        )
        df["edu_age_group"] = (
            df["education"].astype(str) + "_" + df["age_group"].astype(str)
        )

    # (5) familysize → log1p
    for df in [X_train, X_test]:
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0))

    # (6) 불필요 컬럼 제거
    drop_train = [c for c in QE_COLS + ["voted", "index"] if c in X_train.columns]
    drop_test = [c for c in QE_COLS + ["index"] if c in X_test.columns]
    X_train = X_train.drop(columns=drop_train, errors="ignore")
    X_test = X_test.drop(columns=drop_test, errors="ignore")

    # (7) OHE (기본 범주형 + interaction 범주형)
    all_cat = CATEGORICAL_COLS + ["edu_married_group", "edu_age_group"]
    existing_cat = [c for c in all_cat if c in X_train.columns]
    X_train = pd.get_dummies(X_train, columns=existing_cat)
    X_test = pd.get_dummies(X_test, columns=existing_cat)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"\nLinear FE 완료:")
    print(f"  피처 {X_train.shape[1]}개 (OHE + interaction 포함)")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test


# ═════════════════════════════════════════════════════════
# 내부 헬퍼 함수
# ═════════════════════════════════════════════════════════
def _add_delay_root10(df):
    """QE 20개 합산 → 10제곱근 (세 코드 공통)"""
    df = df.copy()
    qe_present = [c for c in QE_COLS if c in df.columns]
    if qe_present:
        delay_sum = df[qe_present].sum(axis=1)
        df["delay_root10"] = np.power(delay_sum.clip(lower=0), 0.1)
    return df


def _add_qa_summary(df):
    """NN용 QA summary 4개 (nn.py 방식)"""
    df = df.copy()
    qa_present = [c for c in QA_COLS if c in df.columns]
    if len(qa_present) == len(QA_COLS):
        qa_frame = df[qa_present]
        df["A_mean"] = qa_frame.mean(axis=1)
        df["A_std"] = qa_frame.std(axis=1)
        df["A_extreme_ratio"] = ((qa_frame == 1) | (qa_frame == 5)).mean(axis=1)
        df["A_neutral_ratio"] = (qa_frame == 3).mean(axis=1)
    return df

# ===============================================================================================================
# [추가] WR/WF SUMMARY 생성 함수
# raw_plus_summary baseline 반영
# =====================================================
def _add_wrwf_summary(df):
    df = df.copy()

    wr_present = sorted([c for c in df.columns if c.startswith("wr_")])
    wf_present = sorted([c for c in df.columns if c.startswith("wf_")])

    if len(wr_present) == 0 or len(wf_present) == 0:
        return df

    wr_frame = df[wr_present]
    wf_frame = df[wf_present]

    df["wr_mean"] = wr_frame.mean(axis=1)
    df["wr_sum"] = wr_frame.sum(axis=1)
    df["wf_mean"] = wf_frame.mean(axis=1)
    df["wf_sum"] = wf_frame.sum(axis=1)

    df["disc_score"] = df["wr_mean"] - df["wf_mean"]
    df["overclaim_ratio"] = df["wf_sum"] / (df["wr_sum"] + 1.0)

    df["wr_std"] = wr_frame.std(axis=1)
    df["wf_std"] = wf_frame.std(axis=1)

    return df

def _apply_nn_fixed_transforms(df):
    """
    nn.py의 고정 수치 변환 (학습 통계 불필요).
    - QA: (x - 3) / 2
    - TP: (x - 3.5) / 3.5
    - familysize: log1p
    """
    df = df.copy()
    for col in QA_COLS:
        if col in df.columns:
            df[col] = (df[col].astype(np.float32) - 3.0) / 2.0
    for col in TP_COLS:
        if col in df.columns:
            df[col] = (df[col].astype(np.float32) - 3.5) / 3.5
    if "familysize" in df.columns:
        df["familysize"] = np.log1p(df["familysize"].clip(lower=0).astype(np.float32))
    return df


# ═════════════════════════════════════════════════════════
# npy 유틸리티 (멤버 C용)
# ═════════════════════════════════════════════════════════
def save_npy(preds, path):
    """npy 저장 + 검증 출력"""
    np.save(path, preds)
    print(f"  Saved: {path} | shape={preds.shape} | "
          f"mean={preds.mean():.4f} | min={preds.min():.4f} | max={preds.max():.4f}")


def load_npy(model_names, load_dir="."):
    """여러 모델 OOF/test npy 로드 → 메타 피처 행렬"""
    oof_list, test_list = [], []
    for name in model_names:
        oof = np.load(Path(load_dir) / f"{name}_oof.npy")
        test = np.load(Path(load_dir) / f"{name}_test.npy")
        oof_list.append(oof)
        test_list.append(test)
        print(f"  {name}: OOF mean={oof.mean():.4f}, Test mean={test.mean():.4f}")
    return np.column_stack(oof_list), np.column_stack(test_list)


def check_diversity(model_names, load_dir="."):
    """모델 간 OOF 상관계수 분석"""
    from scipy.stats import pearsonr
    preds = {n: np.load(Path(load_dir) / f"{n}_oof.npy") for n in model_names}
    names = list(preds.keys())
    print("\n모델 간 OOF 상관계수:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr, _ = pearsonr(preds[names[i]], preds[names[j]])
            tag = "좋음(다양성 O)" if corr < 0.90 else "높음(다양성 부족)"
            print(f"  {names[i]} vs {names[j]}: {corr:.4f} — {tag}")


# ═════════════════════════════════════════════════════════
# 실행 예시
# ═════════════════════════════════════════════════════════
if __name__ == "__main__":
    TRAIN_PATH = "data/train.csv"
    TEST_PATH = "data/test_x.csv"
    SAMPLE_SUB = "data/sample_submission.csv"

    train_df, test_df, y, sample_sub = load_common(TRAIN_PATH, TEST_PATH, SAMPLE_SUB)

    print("\n" + "=" * 60)
    print("멤버 A: NN 준비")
    print("=" * 60)
    X_nn, X_nn_test, num_cols, cat_cols = prepare_for_nn(train_df, test_df)

    print("\n" + "=" * 60)
    print("멤버 B-1: CatBoost 준비")
    print("=" * 60)
    X_cat, X_cat_test, cat_idx = prepare_for_catboost(train_df, test_df)

    print("\n" + "=" * 60)
    print("멤버 B-2: Linear 준비")
    print("=" * 60)
    X_lr, X_lr_test = prepare_for_linear(train_df, test_df)

    # Fold 확인
    print("\n" + "=" * 60)
    print("5-Fold 인덱스 (CatBoost/Linear 공통)")
    for i, (tr, va) in enumerate(OOF_FOLDS_5.split(X_cat, y)):
        print(f"  Fold {i}: train={len(tr)}, val={len(va)}")

    print("\nNN은 5repeat × 7fold 사용 (nn.py 내부에서 처리)")
