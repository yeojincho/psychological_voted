"""
train_mlp_b.py — MLP-B "전체투입형" (앙상블 다양성용)
═══════════════════════════════════════════════════════
베이스라인 nn.py와 다른 점 3가지:
  1. 피처: 베이스라인 전체 + 추가 엔지니어링 (wr_sum, A_low/high_ratio 등)
  2. 아키텍처: Narrow Deep 128→64→32→1, SiLU, 높은 dropout
  3. 손실: Focal Loss (gamma=2.0) — 어려운 샘플에 집중

설계 의도:
  베이스라인 NN이 모든 샘플을 동등하게 학습하는 반면,
  MLP-B는 경계 근처 어려운 샘플(20대, edu=2)에 집중한다.
  Focal Loss가 이미 맞추기 쉬운 60대/10대 확실한 케이스의
  loss를 줄이고, 판별 어려운 케이스에 학습을 집중시킨다.

출력:
  outputs/models/mlp_b_oof.npy
  outputs/models/mlp_b_test.npy
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fe_v2 import (
    load_common, save_npy,
    QA_COLS, QE_COLS, TP_COLS, WR_COLS,
    CATEGORICAL_COLS, NN_CONT_STANDARDIZE_COLS,
)


# ─────────────────────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # 프로젝트 루트
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────
SEED = 42
N_REPEAT = 5
N_SKFOLD = 7
N_EPOCH = 48
BATCH_SIZE = 64    # 작은 배치 → 더 많은 gradient noise → 정규화 효과

# MLP-B: Narrow Deep
HIDDEN_DIMS = [128, 64, 32]
DROPOUT_INPUT = 0.05
DROPOUT_HIDDEN = 0.40

LR = 2e-3
WEIGHT_DECAY = 1e-1
ETA_MIN = 2e-4

# Focal Loss
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.5


# ─────────────────────────────────────────────────────────
# 2. Reproducibility
# ─────────────────────────────────────────────────────────
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"DEVICE: {DEVICE}")

PIN_MEMORY = torch.cuda.is_available()
loader_params = {"batch_size": BATCH_SIZE, "num_workers": 0, "pin_memory": PIN_MEMORY}


# ─────────────────────────────────────────────────────────
# 3. Focal Loss
# ─────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    어려운 샘플(확률 ~0.5)의 loss를 키우고,
    쉬운 샘플(확률 ~0 or ~1)의 loss를 줄인다.
    gamma=0이면 BCE와 동일, gamma 높을수록 어려운 샘플에 집중.
    """
    def __init__(self, gamma=2.0, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ─────────────────────────────────────────────────────────
# 4. Model — Narrow Deep + SiLU + BatchNorm
# ─────────────────────────────────────────────────────────
class MLP_B(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        layers = [nn.Dropout(DROPOUT_INPUT)]
        prev_dim = input_dim
        for hdim in HIDDEN_DIMS:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.SiLU(inplace=True),
                nn.Dropout(DROPOUT_HIDDEN),
            ])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ─────────────────────────────────────────────────────────
# 5. Feature Engineering — 전체투입 + 추가 엔지니어링
# ─────────────────────────────────────────────────────────
def prepare_mlp_b(train_df, test_df):
    """
    MLP-B 전용 FE: 베이스라인 전체 피처 + 추가 피처.
    베이스라인 nn.py 대비 추가되는 것:
      - wr_sum (실험 로그에서 보조 효과 확인)
      - A_low_ratio, A_high_ratio (liner.py에서 사용)
      - TP_mean (tp 전체 요약)
    """
    X_train = train_df.copy()
    X_test = test_df.copy()

    for df in [X_train, X_test]:
        # (1) QA summary (nn.py와 동일)
        qa_frame = df[QA_COLS]
        df["A_mean"] = qa_frame.mean(axis=1)
        df["A_std"] = qa_frame.std(axis=1)
        df["A_extreme_ratio"] = ((qa_frame == 1) | (qa_frame == 5)).mean(axis=1)
        df["A_neutral_ratio"] = (qa_frame == 3).mean(axis=1)

        # (2) QE summary
        qe_present = [c for c in QE_COLS if c in df.columns]
        delay_sum = df[qe_present].sum(axis=1)
        df["delay_root10"] = np.power(delay_sum.clip(lower=0), 0.1)

        # (3) 추가 피처 — 베이스라인에 없는 것들
        wr_present = [c for c in WR_COLS if c in df.columns]
        if wr_present:
            df["wr_sum"] = df[wr_present].sum(axis=1)

        df["A_low_ratio"] = (qa_frame <= 2).sum(axis=1) / len(QA_COLS)
        df["A_high_ratio"] = (qa_frame >= 4).sum(axis=1) / len(QA_COLS)

        tp_present = [c for c in TP_COLS if c in df.columns]
        if tp_present:
            df["TP_mean"] = df[tp_present].mean(axis=1)

        # (4) familysize
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0).astype(np.float32))

    # 사용할 컬럼 제거: QE raw + hand + index + voted
    drop_train = [c for c in QE_COLS + ["hand", "index", "voted"] if c in X_train.columns]
    drop_test = [c for c in QE_COLS + ["hand", "index"] if c in X_test.columns]
    X_train = X_train.drop(columns=drop_train, errors="ignore")
    X_test = X_test.drop(columns=drop_test, errors="ignore")

    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    numeric_cols = [c for c in X_train.columns if c not in cat_cols]

    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # nn.py 고정 변환 (QA, TP)
    for col in QA_COLS:
        if col in X_train.columns:
            X_train[col] = (X_train[col].astype(np.float32) - 3.0) / 2.0
            X_test[col] = (X_test[col].astype(np.float32) - 3.0) / 2.0
    for col in TP_COLS:
        if col in X_train.columns:
            X_train[col] = (X_train[col].astype(np.float32) - 3.5) / 3.5
            X_test[col] = (X_test[col].astype(np.float32) - 3.5) / 3.5

    # fold 내 표준화 대상 (summary 계열 + 추가 피처)
    fold_std_cols = [c for c in
        NN_CONT_STANDARDIZE_COLS + ["wr_sum", "A_low_ratio", "A_high_ratio", "TP_mean"]
        if c in numeric_cols
    ]

    print(f"\nMLP-B FE 완료:")
    print(f"  수치형 {len(numeric_cols)}개, 범주형 {len(cat_cols)}개")
    print(f"  추가 피처: wr_sum, A_low/high_ratio, TP_mean")
    print(f"  fold 내 표준화: {fold_std_cols}")

    return X_train, X_test, numeric_cols, cat_cols, fold_std_cols


def build_mlp_b_fold(X_train_df, X_test_df, numeric_cols, cat_cols,
                     fold_std_cols, train_idx, valid_idx):
    """fold 내부에서 표준화 + OHE"""
    X_tr = X_train_df.iloc[train_idx]
    X_va = X_train_df.iloc[valid_idx]
    X_te = X_test_df

    X_tr_num = X_tr[numeric_cols].copy()
    X_va_num = X_va[numeric_cols].copy()
    X_te_num = X_te[numeric_cols].copy()

    for col in fold_std_cols:
        if col in X_tr_num.columns:
            m = X_tr_num[col].mean()
            s = X_tr_num[col].std()
            if pd.isna(s) or s < 1e-6:
                s = 1.0
            X_tr_num[col] = (X_tr_num[col] - m) / s
            X_va_num[col] = (X_va_num[col] - m) / s
            X_te_num[col] = (X_te_num[col] - m) / s

    if cat_cols:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        tr_cat = enc.fit_transform(X_tr[cat_cols])
        va_cat = enc.transform(X_va[cat_cols])
        te_cat = enc.transform(X_te[cat_cols])
    else:
        tr_cat = np.empty((len(X_tr), 0), dtype=np.float32)
        va_cat = np.empty((len(X_va), 0), dtype=np.float32)
        te_cat = np.empty((len(X_te), 0), dtype=np.float32)

    X_tr_np = np.concatenate([X_tr_num.to_numpy(np.float32), tr_cat], axis=1)
    X_va_np = np.concatenate([X_va_num.to_numpy(np.float32), va_cat], axis=1)
    X_te_np = np.concatenate([X_te_num.to_numpy(np.float32), te_cat], axis=1)

    return X_tr_np, X_va_np, X_te_np, X_tr_np.shape[1]


# ─────────────────────────────────────────────────────────
# 6. Data
# ─────────────────────────────────────────────────────────
train_df, test_df, y, sample_sub = load_common(
    DATA_DIR / "train.csv",
    DATA_DIR / "test_x.csv",
    DATA_DIR / "sample_submission.csv",
)

X_train_df, X_test_df, numeric_cols, cat_cols, fold_std_cols = prepare_mlp_b(train_df, test_df)


# ─────────────────────────────────────────────────────────
# 7. Training loop
# ─────────────────────────────────────────────────────────
oof_pred = np.zeros(len(X_train_df), dtype=np.float32)
test_pred = np.zeros(len(X_test_df), dtype=np.float32)

for repeat in range(N_REPEAT):
    skf = StratifiedKFold(n_splits=N_SKFOLD, random_state=SEED + repeat, shuffle=True)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_df, y), 1):
        X_tr_np, X_va_np, X_te_np, input_dim = build_mlp_b_fold(
            X_train_df, X_test_df, numeric_cols, cat_cols, fold_std_cols,
            train_idx, valid_idx,
        )

        X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32)
        X_va_t = torch.tensor(X_va_np, dtype=torch.float32)
        X_te_t = torch.tensor(X_te_np, dtype=torch.float32)
        y_tr_t = torch.tensor(y[train_idx], dtype=torch.float32)
        y_va_t = torch.tensor(y[valid_idx], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), shuffle=True, drop_last=True, **loader_params)
        valid_loader = DataLoader(TensorDataset(X_va_t, y_va_t), shuffle=False, **loader_params)
        test_loader = DataLoader(TensorDataset(X_te_t), shuffle=False, **loader_params)

        model = MLP_B(input_dim=input_dim).to(DEVICE)

        # Focal Loss (베이스라인과 가장 큰 차이)
        criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)

        # validation 평가용 BCE (Focal Loss로 AUC 비교하면 불공정)
        eval_criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(N_EPOCH // 6, 1), eta_min=ETA_MIN,
        )

        best_val_auc = -np.inf
        best_val_loss = np.inf
        best_valid_prob = np.zeros(len(valid_idx), dtype=np.float32)
        best_state = None

        for epoch in tqdm(range(N_EPOCH), desc=f"R{repeat+1} F{fold}/{N_SKFOLD}"):
            model.train()
            for step, (xx, yy) in enumerate(train_loader):
                xx, yy = xx.to(DEVICE), yy.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(xx), yy)
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + step / max(len(train_loader), 1))

            model.eval()
            val_probs, rloss, rn = [], 0.0, 0
            with torch.no_grad():
                for xx, yy in valid_loader:
                    xx, yy = xx.to(DEVICE), yy.to(DEVICE)
                    pred = model(xx)
                    rloss += eval_criterion(pred, yy).item() * len(yy)
                    rn += len(yy)
                    val_probs.append(torch.sigmoid(pred).cpu().numpy())

            vl = rloss / rn
            vp = np.concatenate(val_probs)
            va = roc_auc_score(y[valid_idx], vp)

            if va > best_val_auc or (np.isclose(va, best_val_auc) and vl < best_val_loss):
                best_val_loss, best_val_auc = vl, va
                best_valid_prob = vp.copy()
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        oof_pred[valid_idx] += best_valid_prob / N_REPEAT

        model.load_state_dict(best_state)
        model.eval()
        tp = []
        with torch.no_grad():
            for (xx,) in test_loader:
                tp.append(torch.sigmoid(model(xx.to(DEVICE))).cpu().numpy())
        test_pred += np.concatenate(tp) / (N_REPEAT * N_SKFOLD)

        print(f"[R{repeat+1} F{fold}] AUC={best_val_auc:.6f}")


# ─────────────────────────────────────────────────────────
# 8. Results + Save
# ─────────────────────────────────────────────────────────
oof_auc = roc_auc_score(y, oof_pred)
print(f"\n{'='*50}")
print(f"MLP-B FINAL OOF AUC: {oof_auc:.6f}")
print(f"{'='*50}")

save_npy(oof_pred, OUTPUT_DIR / "mlp_b_oof.npy")
save_npy(test_pred, OUTPUT_DIR / "mlp_b_test.npy")
print("MLP-B Done.")
