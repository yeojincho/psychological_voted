"""
src/inference.py
────────────────
저장된 OOF/test .npy 파일로 앙상블 submission 생성.

실행:
    python src/inference.py                    # models/ 의 모든 *_test.npy 평균 앙상블
    python src/inference.py --models lgbm xgb  # 지정 모델만 앙상블
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from src.config import TEST_PATH, OUTPUT_DIR, MODEL_DIR


def make_submission(test_index, preds, filename=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"submission_ensemble_{ts}.csv"
    path = os.path.join(OUTPUT_DIR, filename)
    pd.DataFrame({"index": test_index, "voted": preds}).to_csv(path, index=False)
    print(f"submission → {path}  (min={preds.min():.4f}, max={preds.max():.4f})")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="앙상블할 모델명 (예: lgbm xgb catboost)")
    args = parser.parse_args()

    test_raw   = pd.read_csv(TEST_PATH)
    test_index = test_raw["index"] if "index" in test_raw.columns \
                 else pd.RangeIndex(len(test_raw))

    if args.models:
        model_names = args.models
    else:
        # models/ 폴더의 *_test.npy 전부
        npy_files   = [f for f in os.listdir(MODEL_DIR) if f.endswith("_test.npy")]
        model_names = [f.replace("_test.npy", "") for f in sorted(npy_files)]

    if not model_names:
        raise FileNotFoundError(f"{MODEL_DIR}/ 에 *_test.npy 파일이 없습니다.")

    preds_list = []
    for name in model_names:
        path = os.path.join(MODEL_DIR, f"{name}_test.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"파일 없음: {path}")
        arr = np.load(path)
        preds_list.append(arr)
        print(f"  로드: {name}  (AUC용 OOF: {os.path.join(MODEL_DIR, name+'_oof.npy')})")

    ensemble = np.mean(preds_list, axis=0)
    print(f"\n앙상블 모델: {model_names}")
    make_submission(test_index, ensemble)


if __name__ == "__main__":
    main()
