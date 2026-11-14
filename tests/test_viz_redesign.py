"""TV5 - tests/test_viz_redesign.py (Kiểm thử đơn vị cho src/viz_redesign.py)."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import REPORT_FIG_DIR, SLIDE_FIG_DIR
from src.viz_redesign import fit_som_labels, plot_fig2_som_redesign, plot_fig4_error_analysis


@pytest.fixture(scope="module")
def mock_discovery_data():
    """Tạo dữ liệu nhỏ mô phỏng: 20 mẫu (12 ALL, 8 AML) x 50 probe."""
    rng = np.random.default_rng(42)
    n, p = 20, 50
    ids = np.arange(1, n + 1)
    cls = np.array(["ALL"] * 12 + ["AML"] * 8)
    sub = np.array(["B-ALL"] * 8 + ["T-ALL"] * 4 + ["AML"] * 8)
    split = np.array(["train"] * 12 + ["test"] * 8)
    samples = pd.DataFrame({
        "split": split,
        "class": cls,
        "subtype": sub,
        "tissue": ["BM"] * 16 + ["PB"] * 4,
        "source": ["DFCI"] * 10 + ["CALGB"] * 5 + ["CCG"] * 5
    }, index=pd.Index(ids, name="sample_id"))

    # Tạo biểu hiện gen có tín hiệu
    X = rng.normal(2.5, 0.4, (n, p))
    X[cls == "AML", :25] += 0.8
    X[cls == "ALL", 25:] += 0.8
    X_df = pd.DataFrame(X, index=samples.index, columns=[f"probe_{i}" for i in range(p)])
    return X_df, samples


@pytest.fixture(scope="module")
def mock_predictions(mock_discovery_data):
    """Bảng dự đoán kiểm tra cho các mẫu test."""
    _, samples = mock_discovery_data
    test_idx = samples[samples["split"] == "test"].index
    preds = pd.DataFrame({
        "y_true": samples.loc[test_idx, "class"],
        "y_pred": samples.loc[test_idx, "class"],
        "score": [0.15, 0.22, 0.18, 0.25, 0.85, 0.90, 0.78, 0.45]
    }, index=test_idx)
    # Giả lập mẫu cuối bị sai/ranh giới
    preds.iloc[-1, preds.columns.get_loc("y_pred")] = "ALL"
    return preds


def test_fit_som_labels_shape(mock_discovery_data):
    X_df, _ = mock_discovery_data
    labels = fit_som_labels(X_df.values, (1, 2), seed=42, n_iter=100)
    assert len(labels) == len(X_df)
    assert set(np.unique(labels)).issubset({0, 1})


def test_plot_fig2_som_redesign_returns_figure(mock_discovery_data):
    X_df, samples = mock_discovery_data
    for mode in ("report", "slide"):
        fig = plot_fig2_som_redesign(X_df, samples, mode=mode, seed=42)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_plot_fig4_error_analysis_returns_figure(mock_discovery_data, mock_predictions):
    _, samples = mock_discovery_data
    for mode in ("report", "slide"):
        fig = plot_fig4_error_analysis(mock_predictions, samples, mode=mode)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_fig_styles_reject_invalid_mode(mock_discovery_data, mock_predictions):
    X_df, samples = mock_discovery_data
    with pytest.raises(ValueError):
        plot_fig2_som_redesign(X_df, samples, mode="invalid")

    with pytest.raises(ValueError):
        plot_fig4_error_analysis(mock_predictions, samples, mode="invalid")


def test_redesigned_figures_files_exist():
    assert (REPORT_FIG_DIR / "F02_som_redesign_report.png").exists()
    assert (SLIDE_FIG_DIR / "F02_som_redesign_slide.png").exists()
    assert (REPORT_FIG_DIR / "F04_error_analysis_report.png").exists()
    assert (SLIDE_FIG_DIR / "F04_error_analysis_slide.png").exists()
