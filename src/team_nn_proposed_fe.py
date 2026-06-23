"""
train_nn_proposed_fe.py — 제안 FE 28개 기반 NN
═══════════════════════════════════════════════
기존 nn.py (raw 70개, AUC 0.7745) 대비:
  - QA 20개 raw → QA_T, QA_V, QA_M, QA_Mach, A_extreme, A_neutral (6개)
  - TP 10개 raw → big5_E/A/C/S/O, TP_std (6개)
  - QE 20개 raw → delay_root10, qe_std, qe_fatigue (3개)
  - WR 13 + WF 3 raw → real_accept_rate, fake_accept_rate, disc_ratio (3개)
  - demographic → 기본 + edu_age_group (10개)
  총 28개 파생변수만 사용 (raw 전부 제거)

목적: NN에서 "노이즈 줄이기 + 신호 보존" 균형점 검증
      + 앙상블 다양성용 (LGBM은 raw, 이 NN은 파생만)

아키텍처: nn.py와 동일 (180→32, LeakyReLU+ReLU)
CV: 5 repeat × 7 fold × 48 epoch

출력:
  outputs/models/nn_proposed_oof.npy
  outputs/models/nn_proposed_test.npy
  submission_nn_proposed.csv
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


# ─────────────────────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # 프로젝트 루트
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test_x.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"


# ─────────────────────────────────────────────────────────
# 1. Config (nn.py 동일)
# ─────────────────────────────────────────────────────────
SEED = 42
N_REPEAT = 5
N_SKFOLD = 7
N_EPOCH = 48
BATCH_SIZE = 72
NUM_WORKERS = 0

LR = 5e-3
WEIGHT_DECAY = 7.8e-2
ETA_MIN = 4e-4

# 상수
QUESTION_KEYS = list("abcdefghijklmnopqrst")
QA_COLS = [f"Q{k}A" for k in QUESTION_KEYS]
QE_COLS = [f"Q{k}E" for k in QUESTION_KEYS]
TP_COLS = [f"tp0{i}" for i in range(1, 10)] + ["tp10"]
WR_COLS = [f"wr_{str(i).zfill(2)}" for i in range(1, 14)]
WF_COLS = [f"wf_{str(i).zfill(2)}" for i in range(1, 4)]

CATEGORICAL_COLS = [
    "education", "engnat", "gender",
    "married", "race", "religion", "urban",
]

# MACH-IV 하위 척도
MACH_POS = ["QbA", "QcA", "QhA", "QjA", "QmA", "QoA", "QsA"]
MACH_NEG = ["QeA", "QfA", "QkA", "QqA", "QrA"]
TRUST_ITEMS = ["QeA", "QfA", "QkA", "QqA", "QrA"]
VIEW_ITEMS = ["QjA", "QqA", "QbA"]
MANIP_ITEMS = ["QoA", "QsA", "QmA", "QcA"]

# fold 내부 표준화 대상 (연속형 파생변수)
FOLD_STD_COLS = [
    "QA_T", "QA_V", "QA_M", "QA_Mach",
    "A_extreme_ratio", "A_neutral_ratio",
    "big5_E", "big5_A", "big5_C", "big5_S", "big5_O", "TP_std",
    "delay_root10", "qe_std", "qe_fatigue",
    "real_accept_rate", "fake_accept_rate", "disc_ratio",
    "familysize", "edu_age_group",
]


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
loader_params = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "pin_memory": PIN_MEMORY,
}


# ─────────────────────────────────────────────────────────
# 3. Model (nn.py RefNN 동일)
# ─────────────────────────────────────────────────────────
class RefNN(nn.Module):
    def __init__(self, input_dim: int):
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


# ─────────────────────────────────────────────────────────
# 4. Feature Engineering — 제안 FE 28개
# ─────────────────────────────────────────────────────────
def build_proposed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 raw 전부 제거, 파생변수 28개만 생성.

    QA → QA_T(불신), QA_V(인간관), QA_M(조작), QA_Mach(종합),
         A_extreme_ratio, A_neutral_ratio
    TP → big5_E/A/C/S/O, TP_std
    QE → delay_root10, qe_std, qe_fatigue
    WR/WF → real_accept_rate, fake_accept_rate, disc_ratio
    demo → age_num, education, engnat, gender, married, race,
           religion, urban, familysize, edu_age_group
    """
    out = pd.DataFrame(index=df.index)

    # ── QA 파생 (MACH-IV 하위 척도) ──
    out["QA_T"] = (6 - df[TRUST_ITEMS]).mean(axis=1)
    out["QA_V"] = df[VIEW_ITEMS].mean(axis=1)
    out["QA_M"] = df[MANIP_ITEMS].mean(axis=1)
    out["QA_Mach"] = (
        df[MACH_POS].mean(axis=1) + (6 - df[MACH_NEG]).mean(axis=1)
    ) / 2

    qa_frame = df[QA_COLS]
    out["A_extreme_ratio"] = ((qa_frame == 1) | (qa_frame == 5)).mean(axis=1)
    out["A_neutral_ratio"] = (qa_frame == 3).mean(axis=1)

    # ── TP 파생 (TIPI Big Five) ──
    out["big5_E"] = (df["tp01"] + (7 - df["tp06"])) / 2
    out["big5_A"] = (df["tp07"] + (7 - df["tp02"])) / 2
    out["big5_C"] = (df["tp03"] + (7 - df["tp08"])) / 2
    out["big5_S"] = (df["tp09"] + (7 - df["tp04"])) / 2
    out["big5_O"] = (df["tp05"] + (7 - df["tp10"])) / 2
    out["TP_std"] = df[TP_COLS].std(axis=1)

    # ── QE 파생 ──
    qe_frame = df[QE_COLS]
    qe_sum = qe_frame.sum(axis=1)
    out["delay_root10"] = np.power(qe_sum.clip(lower=0), 0.1)
    out["qe_std"] = qe_frame.std(axis=1)

    first_half = [f"Q{c}E" for c in "abcdefghij"]
    second_half = [f"Q{c}E" for c in "klmnopqrst"]
    out["qe_fatigue"] = (
        qe_frame[second_half].mean(axis=1)
        / qe_frame[first_half].mean(axis=1).clip(lower=1)
    )

    # ── WR/WF 파생 ──
    wr_sum = df[WR_COLS].sum(axis=1)
    wf_sum = df[WF_COLS].sum(axis=1)
    out["real_accept_rate"] = wr_sum / 13
    out["fake_accept_rate"] = wf_sum / 3
    out["disc_ratio"] = out["real_accept_rate"] / (out["fake_accept_rate"] + 0.01)

    # ── Demographic ──
    age_map = {"10s": 1, "20s": 2, "30s": 3, "40s": 4, "50s": 5, "60s": 6, "+70s": 7}
    if df["age_group"].dtype == "object":
        out["age_num"] = df["age_group"].map(age_map).fillna(0).astype(int)
    else:
        out["age_num"] = df["age_group"]

    out["education"] = df["education"]
    out["engnat"] = df["engnat"]
    out["married"] = df["married"]
    out["urban"] = df["urban"]

    if "familysize" in df.columns:
        out["familysize"] = np.log1p(df["familysize"].clip(lower=0).astype(np.float32))

    out["edu_age_group"] = out["education"] * 10 + out["age_num"]

    # 범주형 (OHE 대상)
    for col in ["gender", "race", "religion"]:
        if col in df.columns:
            out[col] = df[col].astype(str)

    return out


def prepare_proposed_fe(train_df, test_df):
    """
    제안 FE 적용 후 수치형/범주형 분리.
    OHE와 표준화는 fold 내부에서 수행.
    """
    X_train = build_proposed_features(train_df)
    X_test = build_proposed_features(test_df)

    cat_cols = ["gender", "race", "religion"]
    cat_cols = [c for c in cat_cols if c in X_train.columns]
    numeric_cols = [c for c in X_train.columns if c not in cat_cols]

    print(f"\n제안 FE NN 준비 완료:")
    print(f"  수치형 {len(numeric_cols)}개, 범주형 {len(cat_cols)}개 (OHE 예정)")
    print(f"  총 피처: {X_train.shape[1]}개 (raw 전부 제거)")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, numeric_cols, cat_cols


# ─────────────────────────────────────────────────────────
# 5. Fold matrix builder (nn.py 방식)
# ─────────────────────────────────────────────────────────
def build_fold_matrices(X_train_df, X_test_df, numeric_cols, cat_cols,
                        train_idx, valid_idx):
    """
    fold train에만 OHE fit + 연속형 표준화.
    nn.py의 build_fold_matrices()와 동일 구조.
    """
    X_tr = X_train_df.iloc[train_idx]
    X_va = X_train_df.iloc[valid_idx]
    X_te = X_test_df

    # 수치형
    X_tr_num = X_tr[numeric_cols].copy()
    X_va_num = X_va[numeric_cols].copy()
    X_te_num = X_te[numeric_cols].copy()

    # fold train 기준 표준화
    fold_std = [c for c in FOLD_STD_COLS if c in X_tr_num.columns]
    for col in fold_std:
        mean_ = X_tr_num[col].mean()
        std_ = X_tr_num[col].std()
        if pd.isna(std_) or std_ < 1e-6:
            std_ = 1.0
        X_tr_num[col] = (X_tr_num[col] - mean_) / std_
        X_va_num[col] = (X_va_num[col] - mean_) / std_
        X_te_num[col] = (X_te_num[col] - mean_) / std_

    # 범주형 OHE (fold train에만 fit)
    if cat_cols:
        encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32,
        )
        X_tr_cat = encoder.fit_transform(X_tr[cat_cols])
        X_va_cat = encoder.transform(X_va[cat_cols])
        X_te_cat = encoder.transform(X_te[cat_cols])
    else:
        X_tr_cat = np.empty((len(X_tr), 0), dtype=np.float32)
        X_va_cat = np.empty((len(X_va), 0), dtype=np.float32)
        X_te_cat = np.empty((len(X_te), 0), dtype=np.float32)

    X_tr_np = np.concatenate([X_tr_num.to_numpy(np.float32), X_tr_cat], axis=1)
    X_va_np = np.concatenate([X_va_num.to_numpy(np.float32), X_va_cat], axis=1)
    X_te_np = np.concatenate([X_te_num.to_numpy(np.float32), X_te_cat], axis=1)

    return X_tr_np, X_va_np, X_te_np, X_tr_np.shape[1]


# ─────────────────────────────────────────────────────────
# 6. Data
# ─────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH) if SAMPLE_SUB_PATH.exists() else None

# familysize 이상치 제거
drop_idx = train_df[train_df["familysize"] > 50].index
train_df = train_df.drop(index=drop_idx).reset_index(drop=True)

# target
y = (train_df["voted"] - 1).astype(np.float32).to_numpy()

print(f"Train: {train_df.shape}, Test: {test_df.shape}")
print(f"Target: 0={int((y==0).sum())}, 1={int((y==1).sum())}")

# FE 적용
X_train_df, X_test_df, numeric_cols, cat_cols = prepare_proposed_fe(train_df, test_df)


# ─────────────────────────────────────────────────────────
# 7. Training loop (5 repeat × 7 fold × 48 epoch)
# ─────────────────────────────────────────────────────────
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
        # fold별 행렬 생성
        X_tr_np, X_va_np, X_te_np, input_dim = build_fold_matrices(
            X_train_df, X_test_df,
            numeric_cols, cat_cols,
            train_idx, valid_idx,
        )

        # Tensor
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

        # 모델
        model = RefNN(input_dim=input_dim).to(DEVICE)

        # fold train 기준 pos_weight
        pos_count = float(y[train_idx].sum())
        neg_count = float(len(train_idx) - pos_count)
        pos_weight_value = neg_count / max(pos_count, 1.0)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=DEVICE)
        )

        optimizer = optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(N_EPOCH // 6, 1), eta_min=ETA_MIN,
        )

        # 학습
        best_val_auc = -np.inf
        best_val_loss = np.inf
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

            # Validation
            model.eval()
            val_probs, running_loss, running_n = [], 0.0, 0
            with torch.no_grad():
                for xx, yy in valid_loader:
                    xx, yy = xx.to(DEVICE), yy.to(DEVICE)
                    pred = model(xx)
                    running_loss += criterion(pred, yy).item() * len(yy)
                    running_n += len(yy)
                    val_probs.append(torch.sigmoid(pred).cpu().numpy())

            val_loss = running_loss / running_n
            valid_prob = np.concatenate(val_probs)
            val_auc = roc_auc_score(y[valid_idx], valid_prob)

            if (val_auc > best_val_auc) or (
                np.isclose(val_auc, best_val_auc) and val_loss < best_val_loss
            ):
                best_val_loss = val_loss
                best_val_auc = val_auc
                best_valid_prob = valid_prob.copy()
                best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        # OOF
        oof_pred[valid_idx] += best_valid_prob / N_REPEAT

        # Test (best model)
        model.load_state_dict(best_model_state)
        model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for (xx,) in test_loader:
                xx = xx.to(DEVICE)
                prob = torch.sigmoid(model(xx)).cpu().numpy()
                fold_test_preds.append(prob)

        test_pred += np.concatenate(fold_test_preds) / (N_REPEAT * N_SKFOLD)

        fold_records.append({
            "repeat": repeat + 1,
            "fold": fold,
            "input_dim": input_dim,
            "best_val_auc": best_val_auc,
            "best_val_loss": best_val_loss,
            "pos_weight": pos_weight_value,
        })

        print(
            f"[R{repeat+1} F{fold}] "
            f"AUC={best_val_auc:.6f} | loss={best_val_loss:.6f} | "
            f"dim={input_dim}"
        )


# ─────────────────────────────────────────────────────────
# 8. Results
# ─────────────────────────────────────────────────────────
oof_auc = roc_auc_score(y, oof_pred)

print(f"\n{'='*60}")
print(f"PROPOSED FE NN — FINAL OOF AUC: {oof_auc:.6f}")
print(f"{'='*60}")
print(f"(비교: 기존 nn.py baseline ≈ 0.7745)")
print(f"(차이: {oof_auc - 0.7745:+.6f})")


# ─────────────────────────────────────────────────────────
# 9. Save
# ─────────────────────────────────────────────────────────
np.save(OUTPUT_DIR / "nn_proposed_oof.npy", oof_pred)
np.save(OUTPUT_DIR / "nn_proposed_test.npy", test_pred)
print(f"\nOOF saved: {OUTPUT_DIR / 'nn_proposed_oof.npy'}")
print(f"TEST saved: {OUTPUT_DIR / 'nn_proposed_test.npy'}")

if sample_sub is not None:
    submission = sample_sub.copy()
    submission["voted"] = test_pred
    sub_path = BASE_DIR / "submission_nn_proposed.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path}")
    print(f"  min={submission['voted'].min():.4f}, max={submission['voted'].max():.4f}")
    print(f"  nan={submission['voted'].isna().sum()}")

fold_df = pd.DataFrame(fold_records)
fold_df.to_csv(OUTPUT_DIR / "nn_proposed_fold_records.csv", index=False)

print(f"\nFold 평균 AUC: {fold_df['best_val_auc'].mean():.6f} ± {fold_df['best_val_auc'].std():.6f}")
print("Done.")
