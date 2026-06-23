"""
search_all_blends.py
────────────────────
5개 npy 예측 결과를 이용해
- 모든 조합(subset)
- raw blend
- rank blend
를 탐색하고
Optuna로 최적 가중치를 찾는 코드

출력
- best submission csv
- blend result csv
"""

from pathlib import Path
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata


# =========================================================
# 0. Path
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # 프로젝트 루트
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "outputs" / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "blend_search"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

BEST_SUB_PATH = BASE_DIR / "outputs" / "submission_best_blend.csv"
RESULT_CSV_PATH = OUTPUT_DIR / "blend_search_results.csv"


# =========================================================
# 1. 모델 파일명 설정 --> 모델 추가시 추가!!
#    여기만 네 파일명에 맞게 수정하면 됨
# =========================================================
MODEL_FILES = {
    "nn":      ("nn_oof.npy",       "nn_test.npy"),
    "cat":     ("catboost_oof.npy", "catboost_test.npy"),
    "linear":  ("linear_oof.npy",   "linear_test.npy"),
    "mlp_a":   ("mlp_a_oof.npy",    "mlp_a_test.npy"),
    # 아직 생성 안 된 파일은 주석 처리
    #"cat_seed":  ("cat_seed_oof.npy",   "cat_seed_test.npy"),
    #"nn_strong": ("nn_strong_oof.npy",  "nn_strong_test.npy"),
    #"mlp_b":     ("mlp_b_oof.npy",      "mlp_b_test.npy"),
}


# =========================================================
# 2. Load target
# =========================================================
train = pd.read_csv(TRAIN_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

drop_idx = train[train["familysize"] > 50].index
train = train.drop(index=drop_idx).reset_index(drop=True)

y = (train["voted"] - 1).astype(int).to_numpy()

print("train shape after filter:", train.shape)
print("target distribution:", np.bincount(y))


# =========================================================
# 3. Load predictions
# =========================================================
oof_dict = {}
test_dict = {}

for model_name, (oof_file, test_file) in MODEL_FILES.items():
    oof_path = MODEL_DIR / oof_file
    test_path = MODEL_DIR / test_file

    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF file: {oof_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing TEST file: {test_path}")

    oof_pred = np.load(oof_path)
    test_pred = np.load(test_path)

    assert len(oof_pred) == len(y), f"{model_name} OOF length mismatch"
    assert len(test_pred) == len(sample_sub), f"{model_name} TEST length mismatch"

    oof_dict[model_name] = oof_pred.astype(np.float64)
    test_dict[model_name] = test_pred.astype(np.float64)

print("\n===== SINGLE MODEL OOF =====")
for k in oof_dict:
    print(f"{k:8s}: {roc_auc_score(y, oof_dict[k]):.6f}")


# =========================================================
# 4. Blend helpers
# =========================================================
def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, 1e-12, None)
    return w / w.sum()


def make_rank(x: np.ndarray) -> np.ndarray:
    return rankdata(x, method="average") / len(x)


def blend_predictions(pred_list, weights, mode="raw"):
    weights = normalize_weights(weights)

    if mode == "raw":
        blended = np.zeros_like(pred_list[0], dtype=np.float64)
        for p, w in zip(pred_list, weights):
            blended += w * p
        return blended

    elif mode == "rank":
        blended = np.zeros_like(pred_list[0], dtype=np.float64)
        for p, w in zip(pred_list, weights):
            blended += w * make_rank(p)
        return blended

    elif mode == "power_raw":
        # 확률 분포를 약간 펴거나 눌러보는 탐색
        # 각 모델에 동일 power를 적용
        raise NotImplementedError("power_raw handled separately")
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =========================================================
# 5. Optuna search function
# =========================================================
def search_best_weights(model_names, mode="raw", n_trials=300):
    pred_list_oof = [oof_dict[m] for m in model_names]
    pred_list_test = [test_dict[m] for m in model_names]
    n_models = len(model_names)

    def objective(trial):
        raw_weights = np.array(
            [trial.suggest_float(f"w_{i}", 1e-3, 1.0) for i in range(n_models)],
            dtype=np.float64
        )
        weights = normalize_weights(raw_weights)

        if mode in ("raw", "rank"):
            oof_blend = blend_predictions(pred_list_oof, weights, mode=mode)

        elif mode == "power_raw":
            power = trial.suggest_float("power", 0.7, 1.4)
            transformed = [np.power(np.clip(p, 1e-8, 1 - 1e-8), power) for p in pred_list_oof]
            oof_blend = blend_predictions(transformed, weights, mode="raw")

        else:
            raise ValueError(mode)

        return roc_auc_score(y, oof_blend)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_auc = study.best_value

    raw_weights = np.array([best_params[f"w_{i}"] for i in range(n_models)], dtype=np.float64)
    best_weights = normalize_weights(raw_weights)

    if mode in ("raw", "rank"):
        best_oof = blend_predictions(pred_list_oof, best_weights, mode=mode)
        best_test = blend_predictions(pred_list_test, best_weights, mode=mode)

    elif mode == "power_raw":
        power = best_params["power"]
        oof_transformed = [np.power(np.clip(p, 1e-8, 1 - 1e-8), power) for p in pred_list_oof]
        test_transformed = [np.power(np.clip(p, 1e-8, 1 - 1e-8), power) for p in pred_list_test]
        best_oof = blend_predictions(oof_transformed, best_weights, mode="raw")
        best_test = blend_predictions(test_transformed, best_weights, mode="raw")
    else:
        raise ValueError(mode)

    return {
        "model_names": model_names,
        "mode": mode,
        "best_auc": best_auc,
        "best_weights": best_weights,
        "best_oof": best_oof,
        "best_test": best_test,
        "best_params": best_params,
    }


# =========================================================
# 6. Search all subsets
# =========================================================
all_model_names = list(MODEL_FILES.keys())

results = []
best_global = None

# 조합 개수 줄이고 싶으면 min_subset_size = 2, max_subset_size = 4 등으로 바꾸면 됨
min_subset_size = 2
max_subset_size = len(all_model_names)

search_modes = ["raw", "rank", "power_raw"]

for r in range(min_subset_size, max_subset_size + 1):
    for subset in itertools.combinations(all_model_names, r):
        subset = list(subset)
        print(f"\n{'='*80}")
        print(f"Searching subset: {subset}")
        print(f"{'='*80}")

        for mode in search_modes:
            print(f"  mode = {mode}")

            try:
                out = search_best_weights(subset, mode=mode, n_trials=250)
            except Exception as e:
                print(f"  failed: {e}")
                continue

            row = {
                "subset": "+".join(subset),
                "n_models": len(subset),
                "mode": mode,
                "best_auc": out["best_auc"],
            }

            for i, model_name in enumerate(subset):
                row[f"w_{model_name}"] = out["best_weights"][i]

            if mode == "power_raw":
                row["power"] = out["best_params"]["power"]

            results.append(row)

            print(f"  best_auc = {out['best_auc']:.6f}")
            print("  weights  =", {m: round(w, 6) for m, w in zip(subset, out["best_weights"])})

            if (best_global is None) or (out["best_auc"] > best_global["best_auc"]):
                best_global = out


# =========================================================
# 7. Save result table
# =========================================================
result_df = pd.DataFrame(results).sort_values("best_auc", ascending=False).reset_index(drop=True)
result_df.to_csv(RESULT_CSV_PATH, index=False)

print(f"\nSaved search results: {RESULT_CSV_PATH}")

print("\n===== TOP 10 RESULTS =====")
print(result_df.head(10).to_string(index=False))


# =========================================================
# 8. Save best submission
# =========================================================
print(f"\n{'='*80}")
print("BEST GLOBAL BLEND")
print(f"{'='*80}")
print("subset   :", best_global["model_names"])
print("mode     :", best_global["mode"])
print("best_auc :", round(best_global["best_auc"], 6))
print("weights  :", {m: round(w, 6) for m, w in zip(best_global["model_names"], best_global["best_weights"])})

best_submission = sample_sub.copy()
best_submission["voted"] = best_global["best_test"]
best_submission.to_csv(BEST_SUB_PATH, index=False)

print(f"\nSaved best submission: {BEST_SUB_PATH}")
print("prediction min/max:", best_submission["voted"].min(), best_submission["voted"].max())
print(best_submission.head())

# =========================================================
# 9. Append current TOP-1 to Excel history
# =========================================================
from datetime import datetime

TOP1_HISTORY_XLSX_PATH = OUTPUT_DIR / "best_blend_top1_history.xlsx"

best_row = {
    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "subset": "+".join(best_global["model_names"]),
    "n_models": len(best_global["model_names"]),
    "mode": best_global["mode"],
    "best_auc": float(best_global["best_auc"]),
}

# weight 저장
for model_name, w in zip(best_global["model_names"], best_global["best_weights"]):
    best_row[f"w_{model_name}"] = float(w)

# power_raw일 때만 저장
if "power" in best_global["best_params"]:
    best_row["power"] = float(best_global["best_params"]["power"])

new_row_df = pd.DataFrame([best_row])

if TOP1_HISTORY_XLSX_PATH.exists():
    old_df = pd.read_excel(TOP1_HISTORY_XLSX_PATH)
    history_df = pd.concat([old_df, new_row_df], ignore_index=True)
else:
    history_df = new_row_df.copy()

history_df.to_excel(TOP1_HISTORY_XLSX_PATH, index=False)

print(f"\n[APPENDED] Current run TOP-1 saved to history: {TOP1_HISTORY_XLSX_PATH}")
print(history_df.tail(5))