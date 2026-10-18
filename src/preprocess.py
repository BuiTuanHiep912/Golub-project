"""Threshold, log10, lọc gen; transformer GeneFilter theo chuẩn scikit-learn; ghi tầng curated.

Công việc: 1.2 — Phụ trách: TV3 (kế hoạch mục 2.6).

Quy trình (Dudoit, Fridlyand & Speed, 2002):
  1. Threshold: giá trị < 100 gán 100, > 16000 gán 16000.
  2. log10.  Hai bước 1–2 là phép biến đổi cố định từng phần tử, không học tham số từ dữ liệu
     nên áp cho cả 72 mẫu không gây rò rỉ -> X_log10.parquet (đầu vào của mọi mô hình).
  3. Lọc gen: giữ gen có max/min > 5 và max − min > 500 (tính trên thang gốc).  Bước này HỌC
     tập gen từ dữ liệu nên khi làm mô hình phải nằm trong Pipeline (GeneFilter) và chỉ fit
     trên phần huấn luyện.  X_log10_filtered.parquet lọc trên 72 mẫu, CHỈ dùng cho khám phá.
  4. Chuẩn hóa: dùng sklearn.preprocessing.StandardScaler trong Pipeline (xem make_preprocess).

Đầu ra data/curated/:
  - X_log10.parquet          : 72 x 7129
  - X_log10_filtered.parquet : 72 x ~3571
  - samples.parquet          : bảng mẫu của tầng cleansed (load_golub("curated") cần file này)
  - preprocess_params.json   : ngưỡng + số gen còn lại khi lọc trên 72 mẫu và trên 38 mẫu train
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.config import CURATED_DIR
    from src.load import load_golub
except ModuleNotFoundError:  # chạy trực tiếp `python src/preprocess.py`
    from config import CURATED_DIR
    from load import load_golub

FLOOR, CEIL = 100, 16000
MIN_RATIO, MIN_DIFF = 5, 500


def threshold_log10(X: pd.DataFrame) -> pd.DataFrame:
    """Bước 1–2: cắt về [FLOOR, CEIL] rồi log10. Giữ nguyên index/cột."""
    return np.log10(X.clip(FLOOR, CEIL).astype(float))


class GeneFilter(BaseEstimator, TransformerMixin):
    """Bộ lọc gen Dudoit trên dữ liệu ĐÃ log10: giữ gen có max/min > ratio và max − min > diff
    (max, min tính trên thang gốc 10**X).  fit chỉ nhìn dữ liệu được truyền vào (y bị bỏ qua),
    transform chỉ chọn cột đã học -> đặt trong Pipeline để không rò rỉ."""

    def __init__(self, ratio: float = MIN_RATIO, diff: float = MIN_DIFF):
        self.ratio = ratio
        self.diff = diff

    def fit(self, X, y=None):
        raw = 10 ** np.asarray(X, dtype=float)
        mx, mn = raw.max(axis=0), raw.min(axis=0)
        self.mask_ = (mx / mn > self.ratio) & (mx - mn > self.diff)
        self.n_features_in_ = raw.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        if np.shape(X)[1] != self.n_features_in_:
            raise ValueError(f"GeneFilter fit với {self.n_features_in_} gen, nhận {np.shape(X)[1]}")
        return X.loc[:, self.mask_] if hasattr(X, "loc") else np.asarray(X)[:, self.mask_]

    def get_feature_names_out(self, input_features=None):
        names = input_features if input_features is not None else getattr(
            self, "feature_names_in_", np.array([f"x{i}" for i in range(self.n_features_in_)]))
        return np.asarray(names, dtype=object)[self.mask_]


def make_preprocess(ratio: float = MIN_RATIO, diff: float = MIN_DIFF):
    """Bước 3–4 cho Pipeline mô hình (TV3 2.3, TV4, TV5): đầu vào là X_log10."""
    return make_pipeline(GeneFilter(ratio, diff), StandardScaler())


def main() -> None:
    X, samples = load_golub("cleansed")
    Xl = threshold_log10(X)
    Xf = GeneFilter().fit_transform(Xl)
    train = samples["split"] == "train"
    n_train = int(GeneFilter().fit(Xl[train]).mask_.sum())

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    Xl.to_parquet(CURATED_DIR / "X_log10.parquet")
    Xf.to_parquet(CURATED_DIR / "X_log10_filtered.parquet")
    samples.to_parquet(CURATED_DIR / "samples.parquet")
    params = {
        "threshold": [FLOOR, CEIL],
        "transform": "log10",
        "filter": {"max_over_min_gt": MIN_RATIO, "max_minus_min_gt": MIN_DIFF, "scale": "raw (10**X)"},
        "n_genes_in": X.shape[1],
        "n_genes_filtered_all72": Xf.shape[1],
        "n_genes_filtered_train38": n_train,
        "note": "X_log10_filtered chỉ cho khám phá; mô hình đọc X_log10 và dùng make_preprocess() trong Pipeline",
    }
    (CURATED_DIR / "preprocess_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"X_log10 {Xl.shape} [{Xl.min().min():.3f}, {Xl.max().max():.3f}]; "
          f"filtered (72 mẫu) {Xf.shape}; lọc trên 38 train còn {n_train} gen")


if __name__ == "__main__":
    main()
