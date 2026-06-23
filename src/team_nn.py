"""
train_nn.py — NN 학습 코드 (nn.py 정확 재현)
═══════════════════════════════════════════════
common_fe_v2.py를 import하여 FE 수행
모델 아키텍처/학습 로직은 nn.py와 100% 동일

현재 최고 기록: OOF AUC ≈ 0.7745 / Public LB ≈ 0.7812

핵심 설정 (변경 시 주의):
  - 5 repeat x 7 fold x 48 epoch
  - RefNN: Dropout(0.05)->180->LeakyReLU->Dropout(0.50)->32->ReLU->1
  - AdamW lr=5e-3, wd=7.8e-2
  - CosineAnnealingWarmRestarts T_0=epoch//6
  - pos_weight: fold train 기준 자동 계산
  - OHE + 연속형 표준화: fold 내부에서만 fit
"""

from pathlib import Path
import sys
import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fe_v2 import load_common, prepare_for_nn, build_nn_fold_matrices, save_npy


# =========================================================
# 0. Path
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test_x.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_SUB_PATH = BASE_DIR / "submission_nn.csv"


# =========================================================
# 1. Reproducibility
# =========================================================
SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print("DEVICE:", DEVICE)


# =========================================================
# 2. Config (nn.py 원본 동일)
# =========================================================
N_REPEAT = 5
N_SKFOLD = 7
N_EPOCH = 48
BATCH_SIZE = 72
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()


# =========================================================
# 3. Data
# =========================================================
train_df, test_df, y, sample_sub = load_common(
    TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH
)
X_train_df, X_test_df, numeric_cols, existing_cat_cols = prepare_for_nn(
    train_df, test_df
)


# =========================================================
# 4. Model (nn.py 원본 동일)
# =========================================================
class RefNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear(input_dim, 180, bias=False),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(0.50),
            nn.Linear(180, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


# =========================================================
# 5. Train
# =========================================================
loader_params = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "pin_memory": PIN_MEMORY,
}

oof_pred = np.zeros(len(X_train_df), dtype=np.float32)
test_pred = np.zeros(len(X_test_df), dtype=np.float32)
fold_records = []

for repeat in range(N_REPEAT):
    skf = StratifiedKFold(
        n_splits=N_SKFOLD,
        random_state=SEED + repeat,
        shuffle=True,
    )

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_df, y), 1):

        X_tr_np, X_va_np, X_te_np, input_dim = build_nn_fold_matrices(
            X_train_df, X_test_df,
            numeric_cols, existing_cat_cols,
            train_idx, valid_idx,
        )

        X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32)
        X_va_t = torch.tensor(X_va_np, dtype=torch.float32)
        X_te_t = torch.tensor(X_te_np, dtype=torch.float32)
        y_tr_t = torch.tensor(y[train_idx], dtype=torch.float32)
        y_va_t = torch.tensor(y[valid_idx], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            shuffle=True, drop_last=True, **loader_params,
        )
        valid_loader = DataLoader(
            TensorDataset(X_va_t, y_va_t),
            shuffle=False, drop_last=False, **loader_params,
        )
        test_loader = DataLoader(
            TensorDataset(X_te_t),
            shuffle=False, drop_last=False, **loader_params,
        )

        model = RefNN(input_dim=input_dim).to(DEVICE)

        pos_count = float(y[train_idx].sum())
        neg_count = float(len(train_idx) - pos_count)
        pos_weight_value = neg_count / max(pos_count, 1.0)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [pos_weight_value], dtype=torch.float32, device=DEVICE
            )
        )
        optimizer = optim.AdamW(
            model.parameters(), lr=5e-3, weight_decay=7.8e-2,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(N_EPOCH // 6, 1), eta_min=4e-4,
        )

        best_val_loss = np.inf
        best_val_auc = -np.inf
        best_valid_prob = np.zeros(len(valid_idx), dtype=np.float32)
        best_model_state = None

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
            vp_list, rl, rc = [], 0.0, 0
            with torch.no_grad():
                for xx, yy in valid_loader:
                    xx, yy = xx.to(DEVICE), yy.to(DEVICE)
                    pred = model(xx)
                    rl += criterion(pred, yy).item() * len(yy)
                    rc += len(yy)
                    vp_list.append(torch.sigmoid(pred).cpu().numpy())

            val_loss = rl / rc
            valid_prob = np.concatenate(vp_list)
            val_auc = roc_auc_score(y[valid_idx], valid_prob)

            if val_auc > best_val_auc or (
                np.isclose(val_auc, best_val_auc) and val_loss < best_val_loss
            ):
                best_val_loss = val_loss
                best_val_auc = val_auc
                best_valid_prob = valid_prob.copy()
                best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        oof_pred[valid_idx] += best_valid_prob / N_REPEAT

        model.load_state_dict(best_model_state)
        model.eval()
        tp = []
        with torch.no_grad():
            for (xx,) in test_loader:
                tp.append(torch.sigmoid(model(xx.to(DEVICE))).cpu().numpy())
        test_pred += np.concatenate(tp) / (N_REPEAT * N_SKFOLD)

        fold_records.append({
            "repeat": repeat+1, "fold": fold,
            "input_dim": input_dim, "pos_weight": pos_weight_value,
            "best_val_loss": best_val_loss, "best_val_auc": best_val_auc,
        })
        print(f"[R{repeat+1} F{fold}] auc={best_val_auc:.6f} loss={best_val_loss:.6f}")


# =========================================================
# 6. Final
# =========================================================
oof_auc = roc_auc_score(y, oof_pred)
print(f"\n{'='*50}\nFINAL OOF AUC: {oof_auc:.6f}\n{'='*50}")

save_npy(oof_pred, OUTPUT_DIR / "nn_oof.npy")
save_npy(test_pred, OUTPUT_DIR / "nn_test.npy")

submission = sample_sub.copy()
submission["voted"] = test_pred
submission.to_csv(OUTPUT_SUB_PATH, index=False)
pd.DataFrame(fold_records).to_csv(OUTPUT_DIR / "nn_fold_records.csv", index=False)

print(f"Submission: {OUTPUT_SUB_PATH}")
print(f"pred: {submission['voted'].min():.4f} ~ {submission['voted'].max():.4f}")
