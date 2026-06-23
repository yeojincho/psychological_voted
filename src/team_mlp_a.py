"""
train_mlp_a.py — MLP-A "해석형" (앙상블 다양성용)
═══════════════════════════════════════════════════
베이스라인 nn.py와 다른 점 3가지:
  1. 피처: summary + 인구통계만 (~20개, raw QA/TP/wf/wr 제거)
  2. 아키텍처: Wide Shallow 256→128→1, SiLU, 낮은 dropout
  3. 손실: BCE (pos_weight 없음) — 모든 샘플 동등 취급

설계 의도:
  베이스라인 NN이 raw 문항 수준 패턴을 보는 반면,
  MLP-A는 요약 통계와 인구통계 조합만으로 "큰 그림"을 본다.
  같은 데이터에서 다른 관점 → 앙상블 다양성 확보.

출력:
  outputs/models/mlp_a_oof.npy
  outputs/models/mlp_a_test.npy
"""

from pathlib import Path
import sys
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

# src/ 디렉토리를 경로에 추가 (common_fe_v2 import용)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fe_v2 import load_common, save_npy, QA_COLS, QE_COLS, TP_COLS, CATEGORICAL_COLS


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
BATCH_SIZE = 128   # 피처 적으니 배치 키워서 안정적 학습

# MLP-A: Wide Shallow
HIDDEN_1 = 256
HIDDEN_2 = 128
DROPOUT_1 = 0.03   # 입력 dropout 낮게 (피처 적으니까)
DROPOUT_2 = 0.35   # 히든 dropout 적당히

LR = 3e-3
WEIGHT_DECAY = 5e-2
ETA_MIN = 3e-4


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
# 3. Model — Wide Shallow + SiLU
# ─────────────────────────────────────────────────────────
class MLP_A(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(DROPOUT_1),
            nn.Linear(input_dim, HIDDEN_1),
            nn.BatchNorm1d(HIDDEN_1),
            nn.SiLU(inplace=True),
            nn.Dropout(DROPOUT_2),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.BatchNorm1d(HIDDEN_2),
            nn.SiLU(inplace=True),
            nn.Linear(HIDDEN_2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ─────────────────────────────────────────────────────────
# 4. Feature Engineering — Summary 중심
# ─────────────────────────────────────────────────────────
def prepare_mlp_a(train_df, test_df):
    """
    MLP-A 전용 FE: summary + 인구통계 + 버킷만 사용.
    raw QA 20개, TP 10개, wf/wr 16개 전부 제거.
    """
    X_train = train_df.copy()
    X_test = test_df.copy()

    for df in [X_train, X_test]:
        # QA summary (nn.py와 동일)
        qa_frame = df[QA_COLS]
        df["A_mean"] = qa_frame.mean(axis=1)
        df["A_std"] = qa_frame.std(axis=1)
        df["A_extreme_ratio"] = ((qa_frame == 1) | (qa_frame == 5)).mean(axis=1)
        df["A_neutral_ratio"] = (qa_frame == 3).mean(axis=1)

        # QE summary (실험 로그 검증됨)
        qe_present = [c for c in QE_COLS if c in df.columns]
        delay_sum = df[qe_present].sum(axis=1)
        df["delay_root10"] = np.power(delay_sum.clip(lower=0), 0.1)

        # 버킷 피처 (NN에서 비선형 점프 포착)
        age_map = {"10s": 1, "20s": 2, "30s": 3, "40s": 4, "50s": 5, "60s": 6, "70s": 7}
        if "age_group" in df.columns and df["age_group"].dtype == "object":
            age_num = df["age_group"].map(age_map).fillna(0).astype(int)
        else:
            age_num = df.get("age_group", pd.Series(0, index=df.index))
        df["is_teen"] = (age_num == 1).astype(np.float32)
        df["is_young"] = (age_num <= 2).astype(np.float32)
        df["is_senior"] = (age_num >= 6).astype(np.float32)
        df["is_student"] = (df["education"] == 1).astype(np.float32) if "education" in df.columns else 0

        # familysize
        if "familysize" in df.columns:
            df["familysize"] = np.log1p(df["familysize"].clip(lower=0).astype(np.float32))

    # 사용할 피처 선별 (summary + 버킷 + 인구통계)
    summary_cols = ["A_mean", "A_std", "A_extreme_ratio", "A_neutral_ratio", "delay_root10"]
    bucket_cols = ["is_teen", "is_young", "is_senior", "is_student"]
    demo_num_cols = ["familysize"]

    numeric_cols = summary_cols + bucket_cols + demo_num_cols
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]

    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # 수치형: summary는 fold 내 표준화, 버킷/familysize는 그대로
    fold_std_cols = summary_cols

    print(f"\nMLP-A FE 완료:")
    print(f"  수치형 {len(numeric_cols)}개: {numeric_cols}")
    print(f"  범주형 {len(cat_cols)}개 (OHE 예정)")
    print(f"  fold 내 표준화 대상: {fold_std_cols}")

    return X_train, X_test, numeric_cols, cat_cols, fold_std_cols


def build_mlp_a_fold(X_train_df, X_test_df, numeric_cols, cat_cols,
                     fold_std_cols, train_idx, valid_idx):
    """fold 내부에서 표준화 + OHE"""
    X_tr = X_train_df.iloc[train_idx]
    X_va = X_train_df.iloc[valid_idx]
    X_te = X_test_df

    # 수치형
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

    # OHE
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
# 5. Data
# ─────────────────────────────────────────────────────────
train_df, test_df, y, sample_sub = load_common(
    DATA_DIR / "train.csv",
    DATA_DIR / "test_x.csv",
    DATA_DIR / "sample_submission.csv",
)

X_train_df, X_test_df, numeric_cols, cat_cols, fold_std_cols = prepare_mlp_a(train_df, test_df)


# ─────────────────────────────────────────────────────────
# 6. Training loop
# ─────────────────────────────────────────────────────────
oof_pred = np.zeros(len(X_train_df), dtype=np.float32)
test_pred = np.zeros(len(X_test_df), dtype=np.float32)

for repeat in range(N_REPEAT):
    skf = StratifiedKFold(n_splits=N_SKFOLD, random_state=SEED + repeat, shuffle=True)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train_df, y), 1):
        X_tr_np, X_va_np, X_te_np, input_dim = build_mlp_a_fold(
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

        model = MLP_A(input_dim=input_dim).to(DEVICE)

        # BCE 순수 (pos_weight 없음 — 베이스라인과 다른 점)
        criterion = nn.BCEWithLogitsLoss()

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
                    rloss += criterion(pred, yy).item() * len(yy)
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
# 7. Results + Save
# ─────────────────────────────────────────────────────────
oof_auc = roc_auc_score(y, oof_pred)
print(f"\n{'='*50}")
print(f"MLP-A FINAL OOF AUC: {oof_auc:.6f}")
print(f"{'='*50}")

save_npy(oof_pred, OUTPUT_DIR / "mlp_a_oof.npy")
save_npy(test_pred, OUTPUT_DIR / "mlp_a_test.npy")
print("MLP-A Done.")
