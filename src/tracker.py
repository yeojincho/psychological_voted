"""
src/tracker.py
──────────────
실험 결과 기록 유틸리티.

기록 조건:
  1. 동일한 파라미터로 이미 실행한 적이 있으면 → 기록 안 함
  2. 새로운 파라미터이고 이전 최고 OOF AUC를 넘으면 → 기록

기록 파일: outputs/experiments.csv
  - timestamp, model, oof_auc, params_hash, params (JSON), notes
"""

import os
import json
import hashlib
import pandas as pd
from datetime import datetime

EXPERIMENTS_PATH = "outputs/experiments.csv"

COLUMNS = [
    "timestamp",
    "model",
    "oof_auc",
    "is_best",
    "params_hash",
    "params",
    "notes",
]


def _hash_params(model: str, params: dict, fit_params: dict, notes: str = "") -> str:
    """모델명 + 파라미터 + notes를 정규화하여 SHA256 해시 반환."""
    combined = {"model": model, "params": params, "fit": fit_params, "notes": notes}
    serialized = json.dumps(combined, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def _load() -> pd.DataFrame:
    if os.path.exists(EXPERIMENTS_PATH):
        return pd.read_csv(EXPERIMENTS_PATH)
    return pd.DataFrame(columns=COLUMNS)


def _best_auc(df: pd.DataFrame, model: str) -> float:
    """모델별 최고 AUC 반환."""
    if df.empty:
        return 0.0
    model_df = df[df["model"] == model]
    if model_df.empty:
        return 0.0
    return model_df["oof_auc"].max()


def try_log(
    model: str,
    oof_auc: float,
    params: dict,
    fit_params: dict,
    notes: str = "",
) -> bool:
    """
    조건에 맞을 때만 기록하고, 기록 여부(bool)를 반환.

    Parameters
    ----------
    model      : 모델명 (lgbm / xgb / catboost)
    oof_auc    : OOF AUC 점수
    params     : 모델 하이퍼파라미터 dict
    fit_params : fit 관련 파라미터 dict (early_stopping 등)
    notes      : 자유 메모 (선택)
    """
    os.makedirs(os.path.dirname(EXPERIMENTS_PATH), exist_ok=True)

    h = _hash_params(model, params, fit_params, notes)
    df = _load()

    # 1. 동일 파라미터 → 기록 안 함
    if not df.empty and (df["params_hash"] == h).any():
        print(f"[tracker] 동일 파라미터 실행 → 기록 생략  (hash={h})")
        return False

    # 2. 새 파라미터 → 무조건 기록 (전체 이력 보존)
    best = _best_auc(df, model)
    is_best = oof_auc > best

    row = pd.DataFrame([{
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":       model,
        "oof_auc":     round(oof_auc, 6),
        "is_best":     "True" if is_best else "",
        "params_hash": h,
        "params":      json.dumps({**params, **fit_params}, default=str),
        "notes":       notes,
    }])

    updated = pd.concat([df, row], ignore_index=True)
    updated = updated.sort_values(["model", "oof_auc"], ascending=[True, False]).reset_index(drop=True)
    updated.to_csv(EXPERIMENTS_PATH, index=False)

    if is_best:
        print(f"[tracker] ★ [{model}] 신기록! OOF AUC {oof_auc:.5f}  (이전 최고: {best:.5f})")
    else:
        print(f"[tracker] [{model}] OOF AUC {oof_auc:.5f}  (현재 최고: {best:.5f}) → 기록은 저장")
    print(f"[tracker] 기록 저장 → {EXPERIMENTS_PATH}")
    return is_best


def show_leaderboard() -> None:
    """모델별 OOF AUC 내림차순으로 출력."""
    df = _load()
    if df.empty:
        print("[tracker] 아직 기록된 실험이 없습니다.")
        return

    print("\n── 실험 기록 (모델별 OOF AUC 순위) ─────────────────────────────")
    for model_name, group in df.groupby("model"):
        group = group.sort_values("oof_auc", ascending=False).reset_index(drop=True)
        group.index = range(1, len(group) + 1)
        print(f"\n  [{model_name}]")
        display = group[["timestamp","oof_auc","is_best","notes","params_hash"]].copy()
        print(display.to_string())
    print()


def show_params(rank: int = 1) -> None:
    """특정 순위 실험의 전체 파라미터 출력."""
    df = _load()
    if df.empty or rank > len(df):
        print("[tracker] 해당 순위의 기록이 없습니다.")
        return
    row = df.iloc[rank - 1]
    print(f"\n── 순위 {rank} | {row['model']} | OOF AUC {row['oof_auc']} ───")
    print(f"timestamp : {row['timestamp']}")
    params = json.loads(row["params"])
    for k, v in params.items():
        print(f"  {k:<25}: {v}")
    if row["notes"]:
        print(f"notes     : {row['notes']}")
    print()
