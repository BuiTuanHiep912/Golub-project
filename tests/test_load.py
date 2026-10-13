"""Kiểm tra toàn bộ Bảng 2.4 của tài liệu kế hoạch (TV2 - công việc 1.1).

Chạy sau khi đã có tầng standardized/cleansed:
    python -m src.load && python -m src.quality && pytest tests/
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Cho phép chạy `pytest tests/test_load.py` mà không cần cài package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import STD_DIR  # noqa: E402
from src.load import load_golub  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    req = {n: STD_DIR / f"{n}.parquet" for n in ("expression", "samples", "genes")}
    missing = [f.name for f in req.values() if not f.exists()]
    if missing:
        # Fail chứ không skip: đây là test nghiệm thu Bảng 2.4, không có dữ liệu mà báo
        # xanh thì người nghiệm thu tưởng đã kiểm xong.
        pytest.fail(f"Chưa có tầng standardized ({', '.join(missing)}): chạy `make standardize` trước.")
    X = pd.read_parquet(req["expression"])
    samples = pd.read_parquet(req["samples"])
    genes = pd.read_parquet(req["genes"])
    X.index = X.index.astype(int)
    samples.index = samples.index.astype(int)
    return X, samples, genes


# --- Mức 1: cấu trúc --------------------------------------------------------
def test_expression_shape_72x7129(tables):
    X, _, _ = tables
    assert X.shape == (72, 7129)


def test_samples_shape_72x6(tables):
    _, samples, _ = tables
    assert samples.shape == (72, 6)
    assert list(samples.columns) == ["split", "class", "subtype", "tissue", "source", "sex"]


def test_sample_id_unique(tables):
    _, samples, _ = tables
    assert samples.index.is_unique and samples.index.is_monotonic_increasing


def test_expression_and_samples_share_sample_id(tables):
    """Bước 7 mục 2.1.2: hai file dùng chung chỉ mục sample_id, cùng thứ tự."""
    X, samples, _ = tables
    assert X.index.equals(samples.index)
    assert list(samples.index) == list(range(1, 73))


def test_probe_unique(tables):
    X, _, _ = tables
    assert X.columns.is_unique


# --- Mức 2: phân bố mẫu theo Bảng 2.4 --------------------------------------
def test_class_counts_47_25(tables):
    _, samples, _ = tables
    assert samples["class"].value_counts().to_dict() == {"ALL": 47, "AML": 25}


def test_split_class(tables):
    _, samples, _ = tables
    xt = pd.crosstab(samples["split"], samples["class"])
    assert xt.loc["train", "ALL"] == 27 and xt.loc["train", "AML"] == 11
    assert xt.loc["test", "ALL"] == 20 and xt.loc["test", "AML"] == 14


def test_split_tissue(tables):
    _, samples, _ = tables
    xt = pd.crosstab(samples["split"], samples["tissue"])
    assert xt.loc["train"].to_dict() == {"BM": 38, "PB": 0}
    assert xt.loc["test"].to_dict() == {"BM": 24, "PB": 10}


def test_train_source(tables):
    """ALL ở train đến từ DFCI, AML ở train đến từ CALGB (confounding, H6)."""
    _, samples, _ = tables
    tr = samples[samples["split"] == "train"]
    xt = pd.crosstab(tr["class"], tr["source"])
    assert xt.loc["ALL", "DFCI"] == 27 and xt.loc["AML", "CALGB"] == 11


def test_test_source(tables):
    _, samples, _ = tables
    te = samples[samples["split"] == "test"]
    xt = pd.crosstab(te["class"], te["source"])
    assert xt.loc["ALL", "DFCI"] == 17 and xt.loc["ALL", "St-Jude"] == 3
    assert xt.loc["AML", "CALGB"] == 4 and xt.loc["AML", "St-Jude"] == 5 and xt.loc["AML", "CCG"] == 5


def test_subtype_sums(tables):
    _, samples, _ = tables
    assert samples["subtype"].value_counts().to_dict() == {"B-ALL": 38, "T-ALL": 9, "AML": 25}
    assert samples["subtype"].notna().all()


def test_label_consistency_72_72(tables):
    """Nhãn Kaggle nhất quán với subtype OpenIntro (Bảng 2.4, hàng cuối)."""
    _, samples, _ = tables
    ok = ((samples["class"] == "ALL") & samples["subtype"].isin(["B-ALL", "T-ALL"])) | \
         ((samples["class"] == "AML") & samples["subtype"].eq("AML"))
    assert int(ok.sum()) == len(samples) == 72


# --- Mức 3: định dạng và danh mục -------------------------------------------
def test_genes_meta(tables):
    _, _, genes = tables
    assert len(genes) == 7129
    assert {"probe_id", "description", "is_control"}.issubset(genes.columns)
    affx = genes["probe_id"].str.startswith("AFFX")
    assert (genes["is_control"] == affx).all()
    assert affx.sum() > 0


def test_expression_integer(tables):
    X, _, _ = tables
    assert pd.api.types.is_integer_dtype(X.dtypes.iloc[0])
    assert np.isfinite(X.to_numpy()).all()


def test_calls_pam(tables):
    calls = pd.read_parquet(STD_DIR / "calls.parquet")
    assert calls.shape == (72, 7129)
    assert set(calls.to_numpy().ravel()) <= {"P", "A", "M", np.nan}


# --- Tầng cleansed + API cho nhóm -------------------------------------------
def test_load_golub_cleansed(tables):
    X, samples, _ = tables
    Xc, sc = load_golub(layer="cleansed")
    assert Xc.shape == (72, 7129)
    assert sc.shape == (72, 7)
    assert "qc_outlier" in sc.columns
    assert Xc.index.equals(sc.index)
    assert sc["class"].value_counts().equals(samples["class"].value_counts())


@pytest.mark.parametrize("layer", ["cleanesd", "raw", ""])
def test_load_golub_rejects_unknown_layer(layer):
    # ROOT CAUSE:
    #
    # load_golub() từng chọn tầng bằng `cleansed if layer == "cleansed" else standardized`,
    # nên gõ nhầm tên tầng (hoặc gọi "curated" khi chưa hỗ trợ) vẫn chạy và lặng lẽ trả về
    # tầng standardized - người dùng không biết mình đang đọc sai tầng.
    #
    # Sửa bằng bảng LAYER_FILES và ValueError cho tên không có trong bảng.
    with pytest.raises(ValueError, match="layer phải là một trong"):
        load_golub(layer=layer)