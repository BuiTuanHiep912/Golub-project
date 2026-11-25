"""Kiểm thử GolubWeightedVoting (TV1, công việc 2.1).

Dùng dữ liệu tổng hợp nên không cần dữ liệu thật. Ví dụ tính tay lấy từ
docs/report/5.1_mo_hinh_golub.md, mục 5.1.4.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.golub_wv import GolubWeightedVoting  # noqa: E402

Y4 = ["ALL", "ALL", "AML", "AML"]


def _hand_example():
    """4 mẫu train × 2 gen của mục 5.1.4."""
    return pd.DataFrame({"gA": [3.0, 3.4, 2.0, 2.2], "gB": [2.0, 2.2, 3.0, 3.6]}), Y4


def _synthetic(n_all: int, n_aml: int, seed: int, p: int = 200, n_inf: int = 10, shift: float = 2.0):
    """n_inf gen đầu cao ở ALL, n_inf gen kế tiếp cao ở AML, còn lại là nhiễu."""
    rng = np.random.default_rng(seed)
    X = rng.normal(3.0, 0.3, size=(n_all + n_aml, p))
    X[:n_all, :n_inf] += shift
    X[n_all:, n_inf:2 * n_inf] += shift
    y = np.array(["ALL"] * n_all + ["AML"] * n_aml)
    return pd.DataFrame(X, columns=[f"g{i}" for i in range(p)]), y


# ------------------------------------------------------------- ví dụ tính tay
def test_hand_example_parameters():
    X, y = _hand_example()
    clf = GolubWeightedVoting(n_genes=2, standardize=False).fit(X, y)
    assert clf.snr_["gA"] == pytest.approx(2.5927, abs=1e-4)
    assert clf.snr_["gB"] == pytest.approx(-2.1213, abs=1e-4)
    assert clf.b_["gA"] == pytest.approx(2.65)
    assert clf.b_["gB"] == pytest.approx(2.70)


def test_hand_example_prediction():
    X, y = _hand_example()
    clf = GolubWeightedVoting(n_genes=2, standardize=False).fit(X, y)
    new = pd.DataFrame({"gA": [2.8], "gB": [3.0]})
    v = clf.votes(new).iloc[0]
    assert v["gA"] == pytest.approx(0.3889, abs=1e-4)
    assert v["gB"] == pytest.approx(-0.6364, abs=1e-4)
    assert clf.prediction_strength(new)[0] == pytest.approx(0.2414, abs=1e-4)
    assert clf.predict(new)[0] == "uncertain"  # AML thắng nhưng PS < 0,3
    assert clf.set_params(ps_threshold=0.2).predict(new)[0] == "AML"


def test_standardize_keeps_snr_and_genes():
    """z-score không đổi P(g) nên không đổi tập gen chọn được (mục 5.1.3)."""
    X, y = _synthetic(27, 11, seed=0)
    raw = GolubWeightedVoting(n_genes=20, standardize=False).fit(X, y)
    std = GolubWeightedVoting(n_genes=20, standardize=True).fit(X, y)
    pd.testing.assert_series_equal(raw.snr_, std.snr_, atol=1e-10)
    assert list(raw.genes_) == list(std.genes_)


# ------------------------------------------------------------- chọn gen
def test_selects_half_from_each_class():
    X, y = _synthetic(27, 11, seed=1)
    clf = GolubWeightedVoting(n_genes=20).fit(X, y)
    assert len(clf.genes_) == 20 and clf.genes_.is_unique
    assert (clf.a_ > 0).sum() == 10 and (clf.a_ < 0).sum() == 10
    # gen thông tin được cài sẵn phải được chọn đúng phía
    assert set(clf.genes_[:10]) == {f"g{i}" for i in range(10)}
    assert set(clf.genes_[10:]) == {f"g{i}" for i in range(10, 20)}


def test_constant_gene_never_selected():
    X, y = _synthetic(27, 11, seed=2, p=30)
    X["const"] = 2.0
    clf = GolubWeightedVoting(n_genes=28).fit(X, y)
    assert clf.snr_["const"] == 0.0
    assert np.isfinite(clf.snr_).all()


def test_no_leakage_from_test_set():
    """Tham số chỉ phụ thuộc train: hai tập test khác nhau không đổi gen, a, b."""
    X_tr, y_tr = _synthetic(27, 11, seed=3)
    X_te1, _ = _synthetic(17, 17, seed=4)
    clf = GolubWeightedVoting(n_genes=20).fit(X_tr, y_tr)
    genes, a, b = clf.genes_.copy(), clf.a_.copy(), clf.b_.copy()
    clf.predict(X_te1)
    clf.predict(X_te1 * 10)
    assert list(clf.genes_) == list(genes)
    pd.testing.assert_series_equal(clf.a_, a)
    pd.testing.assert_series_equal(clf.b_, b)


# ------------------------------------------------------------- dự đoán
def test_separable_data_is_perfect_and_confident():
    X_tr, y_tr = _synthetic(27, 11, seed=5, shift=3.0)
    X_te, y_te = _synthetic(20, 14, seed=6, shift=3.0)
    clf = GolubWeightedVoting(n_genes=20).fit(X_tr, y_tr)
    assert (clf.predict(X_te) == y_te).all()
    assert clf.prediction_strength(X_te).min() > 0.5


def test_ps_range_and_threshold():
    X_tr, y_tr = _synthetic(27, 11, seed=7, shift=0.3)  # tín hiệu yếu → có PS thấp
    X_te, _ = _synthetic(20, 14, seed=8, shift=0.3)
    clf = GolubWeightedVoting(n_genes=20).fit(X_tr, y_tr)
    ps = clf.prediction_strength(X_te)
    assert ((ps >= 0) & (ps <= 1)).all()
    pred = clf.predict(X_te)
    assert ((pred == "uncertain") == (ps < 0.3)).all()
    assert "uncertain" not in clf.set_params(ps_threshold=0.0).predict(X_te)
    assert (clf.set_params(ps_threshold=1.01).predict(X_te) == "uncertain").all()


def test_predict_proba_consistent():
    X_tr, y_tr = _synthetic(27, 11, seed=9)
    X_te, _ = _synthetic(20, 14, seed=10)
    clf = GolubWeightedVoting(n_genes=20, ps_threshold=0.0).fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    assert proba.shape == (34, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # PS = |P(ALL) − P(AML)|; lớp có tỷ lệ phiếu cao hơn là lớp dự đoán
    assert np.allclose(clf.prediction_strength(X_te), np.abs(proba[:, 0] - proba[:, 1]))
    assert (clf.classes_[proba.argmax(axis=1)] == clf.predict(X_te)).all()


def test_column_order_does_not_matter():
    X_tr, y_tr = _synthetic(27, 11, seed=11)
    X_te, _ = _synthetic(20, 14, seed=12)
    clf = GolubWeightedVoting(n_genes=20).fit(X_tr, y_tr)
    shuffled = X_te[X_te.columns[::-1]]
    assert (clf.predict(shuffled) == clf.predict(X_te)).all()


# ------------------------------------------------------------- tương thích sklearn và lỗi
def test_sklearn_compatible():
    clf = GolubWeightedVoting(n_genes=10, ps_threshold=0.4)
    assert clone(clf).get_params() == {"n_genes": 10, "ps_threshold": 0.4, "standardize": True}
    with pytest.raises(NotFittedError):
        clf.predict(pd.DataFrame({"g0": [1.0]}))


@pytest.mark.parametrize("n_genes, y, match", [
    (3, Y4, "chẵn"),
    (0, Y4, "chẵn"),
    (2, ["ALL"] * 4, "hai lớp"),
    (2, ["ALL", "ALL", "AML", "XYZ"], "hai lớp"),
    (4, Y4, "lớn hơn số gen"),
])
def test_invalid_input_raises(n_genes, y, match):
    X, _ = _hand_example()
    with pytest.raises(ValueError, match=match):
        GolubWeightedVoting(n_genes=n_genes).fit(X, y)


def test_requires_dataframe_and_selected_genes():
    X, y = _hand_example()
    with pytest.raises(TypeError):
        GolubWeightedVoting(n_genes=2).fit(X.to_numpy(), y)
    clf = GolubWeightedVoting(n_genes=2).fit(X, y)
    with pytest.raises(ValueError, match="thiếu"):
        clf.predict(X[["gA"]])
