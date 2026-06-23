"""
src/train.py
────────────
모델 학습 + OOF 예측 + submission 저장.

실행:
    python src/train.py --model lgbm
    python src/train.py --model xgb
    python src/train.py --model catboost
    python src/train.py --model both     # 3개 학습 후 앙상블
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

from src.config import (
    TRAIN_PATH, TEST_PATH, OUTPUT_DIR, MODEL_DIR,
    SEED, N_SPLITS, CAT_COLS,
    LGBM_PARAMS, LGBM_FIT,
    XGB_PARAMS, XGB_FIT,
    CATBOOST_PARAMS, CATBOOST_FIT,
)
from src.preprocess import preprocess
from src.tracker import try_log, show_leaderboard


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


# ================================================================
# 모델별 fold 학습
# ================================================================
def _train_fold_lgbm(params, X_tr, y_tr, X_val, y_val, X_test, fit_cfg):
    p = params.copy()
    n_est = p.pop("n_estimators", 3000)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    m = lgb.train(
        p, dtrain, num_boost_round=n_est,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(fit_cfg["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(fit_cfg["log_eval"]),
        ],
    )
    vp = m.predict(X_val,  num_iteration=m.best_iteration)
    tp = m.predict(X_test, num_iteration=m.best_iteration)
    return vp, tp


def _train_fold_xgb(params, X_tr, y_tr, X_val, y_val, X_test, fit_cfg):
    p = params.copy()
    n_est = p.pop("n_estimators", 3000)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)
    m = xgb.train(
        p, dtrain, num_boost_round=n_est,
        evals=[(dval, "val")],
        early_stopping_rounds=fit_cfg["early_stopping_rounds"],
        verbose_eval=fit_cfg["verbose"],
    )
    vp = m.predict(xgb.DMatrix(X_val),
                   iteration_range=(0, m.best_iteration + 1))
    tp = m.predict(xgb.DMatrix(X_test),
                   iteration_range=(0, m.best_iteration + 1))
    return vp, tp


def _train_fold_catboost(params, X_tr, y_tr, X_val, y_val, X_test, cat_idx, fit_cfg):
    train_pool = Pool(X_tr,  label=y_tr,  cat_features=cat_idx)
    val_pool   = Pool(X_val, label=y_val, cat_features=cat_idx)
    test_pool  = Pool(X_test, cat_features=cat_idx)
    m = CatBoostClassifier(
        **params,
        early_stopping_rounds=fit_cfg["early_stopping_rounds"],
        verbose=fit_cfg["verbose"],
    )
    m.fit(train_pool, eval_set=val_pool, use_best_model=True)
    vp = m.predict_proba(val_pool)[:, 1]
    tp = m.predict_proba(test_pool)[:, 1]
    return vp, tp


# ================================================================
# CV 학습
# ================================================================
def train_cv(model_type, X_train, y_train, X_test):
    skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    cat_idx = [i for i, c in enumerate(X_train.columns) if c in CAT_COLS]

    if model_type == "lgbm":
        params, fit_cfg = LGBM_PARAMS, LGBM_FIT
    elif model_type == "xgb":
        params, fit_cfg = XGB_PARAMS, XGB_FIT
    elif model_type == "catboost":
        params, fit_cfg = CATBOOST_PARAMS, CATBOOST_FIT

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"── Fold {fold}/{N_SPLITS} ──────────────────")
        X_tr  = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        if model_type == "lgbm":
            vp, tp = _train_fold_lgbm(params, X_tr, y_tr, X_val, y_val, X_test, fit_cfg)
        elif model_type == "xgb":
            vp, tp = _train_fold_xgb(params, X_tr, y_tr, X_val, y_val, X_test, fit_cfg)
        elif model_type == "catboost":
            vp, tp = _train_fold_catboost(params, X_tr, y_tr, X_val, y_val,
                                           X_test, cat_idx, fit_cfg)

        oof_preds[val_idx] = vp
        test_preds        += tp / N_SPLITS
        print(f"Fold {fold} AUC: {roc_auc_score(y_val, vp):.5f}")

    oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"\n[{model_type.upper()} OOF AUC] {oof_auc:.5f}")
    return oof_preds, test_preds, oof_auc


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lgbm",
                        choices=["lgbm","xgb","catboost","both"])
    parser.add_argument("--notes", default="", help="실험 메모 (선택)")
    args = parser.parse_args()

    seed_everything(SEED)

    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw  = pd.read_csv(TEST_PATH)
    test_index = test_raw["index"] if "index" in test_raw.columns \
                 else pd.RangeIndex(len(test_raw))

    train_raw["voted"] = (train_raw["voted"] == 1).astype(int)
    train_proc, test_proc = preprocess(train_raw, test_raw)

    y_train = train_proc["voted"]
    X_train = train_proc.drop(columns=["voted"])
    X_test  = test_proc.drop(columns=["voted"], errors="ignore")

    common  = [c for c in X_train.columns if c in X_test.columns]
    X_train, X_test = X_train[common], X_test[common]

    print(f"피처: {X_train.shape[1]}개 | 샘플: {len(y_train)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    models_to_run = ["lgbm","xgb","catboost"] if args.model == "both" else [args.model]
    all_preds = {}

    for mt in models_to_run:
        print(f"\n{'='*45}")
        print(f"  {mt.upper()}")
        print(f"{'='*45}")
        oof, test_preds, oof_auc = train_cv(mt, X_train, y_train, X_test)

        # OOF / test 저장
        np.save(os.path.join(MODEL_DIR, f"{mt}_oof.npy"),  oof)
        np.save(os.path.join(MODEL_DIR, f"{mt}_test.npy"), test_preds)

        # submission 저장
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        sub_path = os.path.join(OUTPUT_DIR, f"submission_{mt}_{ts}.csv")
        pd.DataFrame({"index": test_index, "voted": test_preds}).to_csv(sub_path, index=False)
        print(f"submission → {sub_path}")
        all_preds[mt] = test_preds

        # 실험 기록 (동일 파라미터면 생략, 신기록일 때만 저장)
        params_map  = {"lgbm": LGBM_PARAMS,     "xgb": XGB_PARAMS,     "catboost": CATBOOST_PARAMS}
        fit_map     = {"lgbm": LGBM_FIT,         "xgb": XGB_FIT,        "catboost": CATBOOST_FIT}
        try_log(mt, oof_auc, params_map[mt], fit_map[mt], notes=args.notes)

    # 실험 리더보드 출력
    show_leaderboard()

    # 앙상블
    if args.model == "both" and len(all_preds) > 1:
        ensemble = np.mean(list(all_preds.values()), axis=0)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ens_path = os.path.join(OUTPUT_DIR, f"submission_ensemble_{ts}.csv")
        pd.DataFrame({"index": test_index, "voted": ensemble}).to_csv(ens_path, index=False)
        print(f"\n앙상블 submission → {ens_path}")


if __name__ == "__main__":
    main()
