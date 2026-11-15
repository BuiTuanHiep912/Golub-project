"""Điểm nối duy nhất giữa dashboard và dữ liệu (công việc 2.6, TV6).

Các trang KHÔNG import trực tiếp src.ingest / src.patient / src.gene_explorer; chúng chỉ gọi module này.
Một biến môi trường quyết định dữ liệu đến từ đâu:

    GOLUB_APP_DATA=mock   (mặc định) dữ liệu giả đúng định dạng, xem mock_data.py
    GOLUB_APP_DATA=real   hàm thật của TV1/TV2/TV4 và dữ liệu thật qua load_golub()

Chuyển sang dữ liệu thật khi các mục trên bàn giao (kế hoạch mục 2.6, "Phụ thuộc"):
    Windows (cmd):  set GOLUB_APP_DATA=real && make app        Linux/macOS: GOLUB_APP_DATA=real make app
Nhánh `real` chỉ gọi hàm theo đúng giao diện đã khóa trong tests/test_interfaces.py; chưa chạy thử được
khi các hàm đó còn NotImplementedError.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

MODE = os.environ.get("GOLUB_APP_DATA", "mock").strip().lower()
if MODE not in ("mock", "real"):
    raise ValueError(f"GOLUB_APP_DATA phải là 'mock' hoặc 'real', nhận {MODE!r}")
IS_MOCK = MODE == "mock"

BATCH_SOURCES = ("DFCI", "CALGB", "St-Jude", "CCG")
PS_THRESHOLD = 0.3

if IS_MOCK:
    from app.components import mock_data as _impl

    ingest_batches = _impl.ingest_batches
    validate_new_sample = _impl.validate_new_sample
    patient_profile = _impl.patient_profile
    find_gene = _impl.find_gene
    gene_stats = _impl.gene_stats
    gene_values = _impl.gene_values
    disease_or_source = _impl.disease_or_source
    ranking_table = _impl.ranking_table
    annotate = _impl.annotate
    sample_warnings = _impl.sample_warnings
    make_sample_file = _impl.make_sample_file
    _reference_data = _impl.reference_data
else:
    from src.gene_explorer import (annotate, disease_or_source, find_gene,  # noqa: F401
                                   gene_stats, gene_values, ranking_table)
    from src.ingest import ingest_batches, validate_new_sample  # noqa: F401
    from src.load import load_golub
    from src.patient import patient_profile  # noqa: F401

    make_sample_file = None   # file thử chỉ có ở chế độ dữ liệu giả

    def _reference_data():
        X, samples = load_golub("curated")
        _, cleansed = load_golub("cleansed")
        samples = samples.join(cleansed[["qc_outlier"]]) if "qc_outlier" in cleansed else samples.assign(qc_outlier=False)
        samples["qc_reason"] = samples["qc_outlier"].map({True: "gắn cờ qc_outlier (xem quality_report.md)", False: ""})
        return X, samples

    def sample_warnings() -> pd.DataFrame:
        _, s = _reference_data()
        w = s[s["qc_outlier"]].reset_index()[["sample_id", "source", "qc_reason"]]
        return w.rename(columns={"qc_reason": "reason"})


@st.cache_resource(show_spinner="Đang đọc dữ liệu…")
def reference_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(X, samples): log10 sau threshold, 72 × p; samples có thêm qc_outlier, qc_reason. Chỉ ĐỌC, không sửa."""
    return _reference_data()


@st.cache_data(show_spinner=False)
def batches() -> pd.DataFrame:
    return ingest_batches()


@st.cache_data(show_spinner=False)
def warnings_table() -> pd.DataFrame:
    return sample_warnings()


@st.cache_data(show_spinner=False)
def pca_all() -> tuple[pd.DataFrame, list]:
    """PCA 72 bệnh nhân (hàm compute_pca của viz.py, việc 1.4)."""
    from src import viz
    X, _ = reference_data()
    scores, ratio = viz.compute_pca(X, n_components=5)
    return scores, list(ratio)
