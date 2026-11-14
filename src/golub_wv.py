"""Classifier weighted voting và prediction strength của Golub (1999).

Công việc: 2.1 — Phụ trách: TV1
Công thức và ví dụ tính tay: docs/report/5.1_mo_hinh_golub.md (bài báo: thân bài tr. 532, ghi chú 18–21).

Quy ước: lớp 1 = ALL (phiếu dương), lớp 2 = AML (phiếu âm). Mọi tham số (chuẩn hóa, SNR,
chọn gen, a, b) học trong `fit` từ đúng dữ liệu được truyền vào, nên LOOCV không rò rỉ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from src.config import PS_THRESHOLD

CLASSES = ("ALL", "AML")  # lớp 1 (dấu dương), lớp 2 (dấu âm)
UNCERTAIN = "uncertain"


class GolubWeightedVoting(BaseEstimator, ClassifierMixin):
    """Weighted voting của Golub et al. (1999).

    Tham số:
        n_genes      : số gen bỏ phiếu, chẵn; n/2 gen SNR lớn nhất + n/2 gen SNR nhỏ nhất (ghi chú 18)
        ps_threshold : PS dưới ngưỡng này thì trả "uncertain" (ghi chú 21, mặc định 0,3)
        standardize  : z-score từng gen theo mean/SD của tập train trước khi bỏ phiếu (ghi chú 19).
                       Đặt False chỉ để đối chiếu với ví dụ tính tay; không đổi tập gen chọn được.

    Thuộc tính học được:
        mean_, sd_ : mean/SD từng gen của tập train (dùng để chuẩn hóa mẫu mới)
        snr_       : SNR P(g) của mọi gen
        genes_     : tên n_genes probe được chọn (n/2 cao ở ALL rồi n/2 cao ở AML)
        a_, b_     : trọng số a_g = P(g) và ranh giới b_g của các gen được chọn
    """

    def __init__(self, n_genes: int = 50, ps_threshold: float = PS_THRESHOLD,
                 standardize: bool = True):
        self.n_genes = n_genes
        self.ps_threshold = ps_threshold
        self.standardize = standardize

    # ------------------------------------------------------------------ học
    def fit(self, X: pd.DataFrame, y):
        """Học SNR, chọn gen và tham số (a, b) từ tập train (Phần A, công thức 1–4)."""
        _check_frame(X)
        y = np.asarray(y)
        if not isinstance(self.n_genes, (int, np.integer)) or self.n_genes <= 0 or self.n_genes % 2:
            raise ValueError(f"n_genes phải là số nguyên chẵn dương, nhận {self.n_genes!r}")
        if len(y) != len(X):
            raise ValueError(f"X có {len(X)} mẫu nhưng y có {len(y)} nhãn")
        if set(y) != set(CLASSES):
            raise ValueError(f"y phải gồm đúng hai lớp {CLASSES}, nhận {sorted(set(y))}")
        if self.n_genes > X.shape[1]:
            raise ValueError(f"n_genes={self.n_genes} lớn hơn số gen {X.shape[1]}")

        # Chuẩn hóa theo train; gen hằng (SD = 0) giữ nguyên thang để tránh chia 0
        if self.standardize:
            self.mean_ = X.mean()
            self.sd_ = X.std(ddof=1).replace(0, 1.0)
        else:
            self.mean_ = pd.Series(0.0, index=X.columns)
            self.sd_ = pd.Series(1.0, index=X.columns)
        Z = (X - self.mean_) / self.sd_

        # Công thức 1, 2: trung bình và SD (chia n − 1) của từng lớp
        is_all = y == CLASSES[0]
        mu1, mu2 = Z[is_all].mean(), Z[~is_all].mean()
        s1, s2 = Z[is_all].std(ddof=1), Z[~is_all].std(ddof=1)

        # Công thức 3: SNR; mẫu số 0 (gen không dao động trong cả hai lớp) thì P = 0
        denom = s1 + s2
        self.snr_ = ((mu1 - mu2) / denom).where(denom > 0, 0.0)

        # Ghi chú 18: n/2 gen cao ở ALL + n/2 gen cao ở AML
        half = self.n_genes // 2
        self.genes_ = self.snr_.nlargest(half).index.append(self.snr_.nsmallest(half).index)

        # Ghi chú 19 / công thức 4
        self.a_ = self.snr_[self.genes_]
        self.b_ = ((mu1 + mu2) / 2)[self.genes_]
        self.classes_ = np.array(CLASSES)
        self.n_features_in_ = X.shape[1]
        return self

    # --------------------------------------------------------------- dự đoán
    def votes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Phiếu v_g = a_g (z_g − b_g) của từng gen (công thức 5); dương bầu ALL, âm bầu AML.

        Trả về DataFrame (mẫu × gen được chọn), giữ index của X.
        """
        check_is_fitted(self, "genes_")
        _check_frame(X)
        missing = self.genes_.difference(X.columns)
        if len(missing):
            raise ValueError(f"X thiếu {len(missing)} gen đã chọn, ví dụ {list(missing[:3])}")
        Z = (X[self.genes_] - self.mean_[self.genes_]) / self.sd_[self.genes_]
        return (Z - self.b_) * self.a_

    def vote_totals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Tổng phiếu V_ALL, V_AML của từng mẫu (công thức 6, 7)."""
        v = self.votes(X)
        return pd.DataFrame({"V_ALL": v.clip(lower=0).sum(axis=1),
                             "V_AML": (-v).clip(lower=0).sum(axis=1)}, index=X.index)

    def prediction_strength(self, X: pd.DataFrame) -> np.ndarray:
        """PS = (V_win − V_lose) / (V_win + V_lose) ∈ [0, 1] (công thức 9, ghi chú 20)."""
        t = self.vote_totals(X)
        total = (t.V_ALL + t.V_AML).to_numpy()
        diff = (t.V_ALL - t.V_AML).abs().to_numpy()
        return np.divide(diff, total, out=np.zeros_like(total), where=total > 0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Tỷ lệ phiếu [V_ALL, V_AML] / (V_ALL + V_AML), cột theo `classes_`.

        Đây là tỷ lệ phiếu, KHÔNG phải xác suất đã hiệu chỉnh; mẫu không có phiếu nào được 0,5/0,5.
        """
        t = self.vote_totals(X)
        total = (t.V_ALL + t.V_AML).to_numpy()
        p_all = np.divide(t.V_ALL.to_numpy(), total, out=np.full_like(total, 0.5), where=total > 0)
        return np.column_stack([p_all, 1 - p_all])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """'ALL' / 'AML' theo bên nhiều phiếu hơn; PS < ps_threshold thì 'uncertain' (công thức 8, 10)."""
        t = self.vote_totals(X)
        pred = np.where(t.V_ALL.to_numpy() > t.V_AML.to_numpy(), CLASSES[0], CLASSES[1]).astype(object)
        pred[self.prediction_strength(X) < self.ps_threshold] = UNCERTAIN
        return pred


def _check_frame(X) -> None:
    """Gen được chọn và tra theo tên probe nên X phải là DataFrame có tên cột."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X phải là pandas DataFrame (cột = probe id), nhận {type(X).__name__}")


if __name__ == "__main__":
    from src.load import load_golub

    X, s = load_golub("curated")
    tr, te = s.split == "train", s.split == "test"
    clf = GolubWeightedVoting().fit(X[tr], s.loc[tr, "class"])
    pred = clf.predict(X[te])
    print(pd.crosstab(s.loc[te, "class"], pred, rownames=["thật"], colnames=["dự đoán"]))
    print(f"PS trung vị trên tập test: {np.median(clf.prediction_strength(X[te])):.2f}")
