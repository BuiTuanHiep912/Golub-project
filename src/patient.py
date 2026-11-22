"""Hồ sơ một bệnh nhân: gợi ý chẩn đoán, độ tin cậy, gen bỏ phiếu, bệnh nhân tương tự.

Công việc: 2.1 — Phụ trách: TV1 — Kế hoạch giai đoạn 2, mục 2.1 (ADR-014).
Dùng cho: TV6 (màn hình Hồ sơ bệnh nhân).

Quy ước: dữ liệu tham chiếu = 38 bệnh nhân tập huấn luyện. 34 bệnh nhân tập kiểm tra
đóng vai bệnh nhân mới; không dùng chúng để huấn luyện, chọn gen hay tính tham số nào.

Giao diện đã thống nhất, KHÔNG đổi tên khi chưa báo nhóm:
    patient_profile(sample_id=52)      # một bệnh nhân có sẵn trong dữ liệu
    patient_profile(values=series)     # mẫu mới, đã qua validate_new_sample() của TV2
    -> dict với các khóa:
       'prediction'  : 'ALL' | 'AML' | 'uncertain'
       'ps'          : float, độ tin cậy
       'votes'       : DataFrame [probe_id, description, value, vote, toward]
       'similar'     : DataFrame [sample_id, correlation, class]  (5 người giống nhất)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import STD_DIR
from src.golub_wv import GolubWeightedVoting
from src.load import load_golub
from src.preprocess import threshold_log10

N_SIMILAR = 5     # số bệnh nhân tương tự trả về
N_GENES = 50      # số gen bỏ phiếu, như bài báo

# nạp giữ liệu tham chiếu
@lru_cache(maxsize=1)
def _reference():
    """Dữ liệu tham chiếu: X log10 (72 mẫu), bảng mẫu, mã 38 mẫu train, mô tả gen."""
    X, samples = load_golub("curated")
    train_ids = samples.index[samples["split"] == "train"]
    desc = pd.read_parquet(STD_DIR / "genes.parquet").set_index("probe_id")["description"]
    return X, samples, train_ids, desc

@lru_cache(maxsize=None)
def _model(exclude: int | None = None) -> GolubWeightedVoting:
    """Mô hình học trên 38 mẫu train; nếu exclude là một mẫu train thì bỏ mẫu đó (37 mẫu)."""
    X, samples, train_ids, _ = _reference()
    ids = train_ids.drop(exclude) if exclude is not None else train_ids
    return GolubWeightedVoting(n_genes=N_GENES).fit(X.loc[ids], samples.loc[ids, "class"])

def _to_log10(values, columns: pd.Index) -> pd.Series:
    """Mẫu mới từ validate_new_sample() ở thang MAS4 gốc -> threshold + log10 như tầng curated."""
    v = pd.Series(values, dtype=float)
    missing = columns.difference(v.index)
    if len(missing):
        raise ValueError(f"Mẫu mới thiếu {len(missing)} gen, ví dụ {list(missing[:3])}")
    return threshold_log10(v[columns])

def _similar(x: pd.Series, clf: GolubWeightedVoting, exclude: int | None) -> pd.DataFrame:
    """5 bệnh nhân tham chiếu có tương quan Pearson cao nhất trên các gen bỏ phiếu (đã z-score)."""
    X, samples, train_ids, _ = _reference()
    ids = train_ids.drop(exclude) if exclude is not None else train_ids
    g = clf.genes_
    Z = (X.loc[ids, g] - clf.mean_[g]) / clf.sd_[g]
    z = (x[g] - clf.mean_[g]) / clf.sd_[g]
    top = Z.corrwith(z, axis=1).nlargest(N_SIMILAR)
    return pd.DataFrame({"sample_id": top.index, "correlation": top.to_numpy(),
                         "class": samples.loc[top.index, "class"].to_numpy()})

def patient_profile(sample_id: int | None = None, values=None) -> dict:
    """Trả về hồ sơ của một bệnh nhân có sẵn (sample_id) hoặc một mẫu mới (values)."""
    if (sample_id is None) == (values is None):
        raise ValueError("Truyền đúng một trong hai: sample_id hoặc values")
    X, samples, train_ids, desc = _reference()

    if sample_id is not None:
        if sample_id not in X.index:
            raise KeyError(f"Không có bệnh nhân {sample_id} trong dữ liệu")
        x = X.loc[sample_id]
        exclude = sample_id if sample_id in train_ids else None
    else:
        x = _to_log10(values, X.columns)
        exclude = None

    clf = _model(exclude)
    row = x.to_frame().T
    v = clf.votes(row).iloc[0]
    votes = pd.DataFrame({
        "probe_id": v.index,
        "description": desc.reindex(v.index).to_numpy(),
        "value": x[v.index].to_numpy(),
        "vote": v.to_numpy(),
        "toward": np.where(v.to_numpy() > 0, "ALL", "AML"),
    }).sort_values("vote", key=np.abs, ascending=False, ignore_index=True)

    return {"prediction": clf.predict(row)[0],
            "ps": float(clf.prediction_strength(row)[0]),
            "votes": votes,
            "similar": _similar(x, clf, exclude)}

if __name__ == "__main__":
    p = patient_profile(sample_id=52)
    print(p["prediction"], round(p["ps"], 2))
    print(p["votes"].head())
    print(p["similar"])
