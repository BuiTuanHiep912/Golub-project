"""Kiểm tra tiền xử lý không nhìn thấy dữ liệu test (TV3, công việc 1.2).

Dùng dữ liệu tổng hợp cùng kích thước Golub (38 train / 34 test) nên không cần dữ liệu thật.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import GeneFilter, make_preprocess, threshold_log10  # noqa: E402


def _raw(n: int, seed: int, p: int = 500) -> pd.DataFrame:
    """Giá trị kiểu MAS4: có âm, có < 100, có > 16000."""
    rng = np.random.default_rng(seed)
    X = rng.lognormal(mean=6, sigma=1.5, size=(n, p)) - 200
    X[:, p // 2:] = rng.normal(1000, 50, size=(n, p - p // 2))  # nửa sau gần hằng -> bị lọc
    return pd.DataFrame(X, columns=[f"g{i}" for i in range(p)])


def _split():
    """X_train 38 mẫu; hai phiên bản test 34 mẫu hoàn toàn khác nhau."""
    return (threshold_log10(_raw(38, 0)), threshold_log10(_raw(34, 1)),
            threshold_log10(_raw(34, 2) * 50))


def test_threshold_log10_range():
    Xl = threshold_log10(_raw(72, 3))
    assert Xl.shape == (72, 500)
    assert Xl.min().min() == 2.0
    assert np.isclose(Xl.max().max(), np.log10(16000))


def test_filter_fitted_on_train_only():
    X_tr, X_te1, X_te2 = _split()
    gf = GeneFilter().fit(X_tr)
    genes = gf.get_feature_names_out().copy()
    assert 0 < len(genes) < X_tr.shape[1]
    # transform tập test không làm đổi tập gen đã học
    assert list(gf.transform(X_te1).columns) == list(genes)
    assert list(gf.transform(X_te2).columns) == list(genes)
    assert (gf.get_feature_names_out() == genes).all()
    # tập gen khớp tính tay trên riêng train
    raw = 10 ** X_tr
    expect = raw.columns[(raw.max() / raw.min() > 5) & (raw.max() - raw.min() > 500)]
    assert list(genes) == list(expect)


def test_scaler_params_from_train_only():
    X_tr, X_te1, X_te2 = _split()
    pipe = make_preprocess().fit(X_tr)
    scaler = pipe[-1]
    mean, scale = scaler.mean_.copy(), scaler.scale_.copy()
    pipe.transform(X_te1)
    pipe.transform(X_te2)
    assert np.array_equal(scaler.mean_, mean) and np.array_equal(scaler.scale_, scale)
    kept = pipe[0].transform(X_tr)
    assert np.allclose(mean, kept.mean())
    assert np.allclose(scale, kept.std(ddof=0))
    # train sau chuẩn hóa có mean 0; test thì không (vì dùng tham số của train)
    assert np.allclose(pipe.transform(X_tr).mean(axis=0), 0)
    assert not np.allclose(pipe.transform(X_te2).mean(axis=0), 0)
