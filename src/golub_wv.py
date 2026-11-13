"""Classifier weighted voting và prediction strength của Golub (1999).

Công việc: 2.1 — Phụ trách: TV1 — Kế hoạch giai đoạn 2, mục 2.1 (ADR-014).
Dùng cho: TV6 (màn hình Hồ sơ bệnh nhân, qua src/patient.py), TV5 (PS cho Fig. 3A).

Giao diện đã thống nhất, KHÔNG đổi tên khi chưa báo nhóm:
    clf = GolubWeightedVoting(n_genes=50, ps_threshold=0.3).fit(X_train, y_train)
    clf.predict(X)              -> array gồm 'ALL', 'AML' hoặc 'uncertain'
    clf.predict_proba(X)        -> array (n, 2), cột theo clf.classes_
    clf.prediction_strength(X)  -> array PS trong [0, 1]
"""
from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin

try:
    from src.config import PS_THRESHOLD
except ModuleNotFoundError:
    from config import PS_THRESHOLD


class GolubWeightedVoting(ClassifierMixin, BaseEstimator):
    """Weighted voting: n/2 gen SNR dương lớn nhất + n/2 gen SNR âm nhỏ nhất, chọn trong fit."""

    def __init__(self, n_genes: int = 50, ps_threshold: float = PS_THRESHOLD):
        self.n_genes = n_genes
        self.ps_threshold = ps_threshold

    def fit(self, X, y):
        raise NotImplementedError("TV1 - công việc 2.1")

    def predict(self, X):
        raise NotImplementedError("TV1 - công việc 2.1")

    def predict_proba(self, X):
        raise NotImplementedError("TV1 - công việc 2.1")

    def prediction_strength(self, X):
        raise NotImplementedError("TV1 - công việc 2.1")


if __name__ == "__main__":
    print("Chưa cài đặt: TV1 - công việc 2.1")
