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
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import REPORT_FIG_DIR, SLIDE_FIG_DIR
from src.viz_redesign import (
    SomPanels, compute_fig4, fit_som_labels, plot_fig2_neighborhood, plot_fig4_som_redesign,
    plot_fig3a_prediction_strength, plot_fig4_error_analysis
)


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


@pytest.fixture(scope="module")
def mock_ps_df():
    """Tạo bảng Prediction Strength mô phỏng cho test."""
    return pd.DataFrame({
        "sample_id": [1, 2, 12, 35, 66, 67, 39, 40],
        "true": ["ALL", "ALL", "ALL", "AML", "AML", "ALL", "AML", "ALL"],
        "pred": ["ALL", "ALL", "uncertain", "uncertain", "ALL", "uncertain", "AML", "ALL"],
        "ps": [0.95, 0.85, 0.05, 0.15, 0.30004, 0.12, 0.75, 0.90],
        "set": ["loocv", "loocv", "loocv", "loocv", "test", "test", "test", "test"],
        "source": ["DFCI", "DFCI", "DFCI", "CALGB", "CCG", "DFCI", "CALGB", "DFCI"],
        "tissue": ["BM", "BM", "BM", "BM", "BM", "PB", "BM", "BM"],
        "subtype": ["B-ALL", "T-ALL", "B-ALL", "AML", "AML", "T-ALL", "AML", "B-ALL"]
    })


def test_fit_som_labels_shape(mock_discovery_data):
    X_df, _ = mock_discovery_data
    labels = fit_som_labels(X_df.values, (1, 2), seed=42, n_iter=100)
    assert len(labels) == len(X_df)
    assert set(np.unique(labels)).issubset({0, 1})


@pytest.fixture
def som_panels():
    pairs = ("B1–B2", "B1–B3", "B1–B4", "B2–B3", "B2–B4", "B3–B4")
    scores = pd.DataFrame({"true": ["ALL", "AML"], "pred": ["ALL", "uncertain"], "ps": [0.7, 0.2]})
    return SomPanels(
        50, 42, pd.DataFrame({"ALL": [5, 1], "AML": [1, 4]}),
        pd.DataFrame({"B-ALL": [4, 1, 1, 0], "T-ALL": [0, 3, 1, 0], "AML": [0, 0, 2, 4]}),
        np.array([0.4, 0.8]), np.array([0.3, 0.7]), np.array([[0.12, 0.20], [0.15, 0.24]]),
        dict.fromkeys(pairs, scores),
    )


def test_plot_fig2_uses_only_train(mock_discovery_data):
    X, samples = mock_discovery_data
    samples = samples.copy()
    samples.loc[[13, 14], "split"] = "train"
    test = samples["split"].eq("test")
    shifted = X.copy()
    shifted.loc[test] += 1000
    for mode in ("report", "slide"):
        first = plot_fig2_neighborhood(X, samples, mode=mode, n_permutations=5)
        second = plot_fig2_neighborhood(shifted, samples, mode=mode, n_permutations=5)
        assert len(first.axes) == 2
        for a, b in zip(first.axes, second.axes):
            assert len(a.lines) == 3
            for line_a, line_b in zip(a.lines, b.lines):
                np.testing.assert_array_equal(line_a.get_ydata(), line_b.get_ydata())
        plt.close(first)
        plt.close(second)


def test_fig4_cluster_validation_keeps_test_out_of_training(mock_discovery_data, monkeypatch):
    X, samples = mock_discovery_data
    samples = samples.copy()
    samples.loc[9:12, "split"] = "test"
    samples.loc[13:20, "split"] = "train"
    train = samples["split"].eq("train")

    def fixed_clusters(values, shape, **kwargs):
        assert len(values) == train.sum()
        return np.repeat(np.arange(np.prod(shape)), len(values) // np.prod(shape))

    monkeypatch.setattr("src.viz_redesign.fit_som_labels", fixed_clusters)
    first = compute_fig4(X, samples, n_permutations=2)
    changed = X.copy()
    changed.loc[~train, X.columns[::2]] += 0.5
    second = compute_fig4(changed, samples, n_permutations=2)

    assert first.null_medians.shape == (2, 2)
    assert len(first.ps_test) == 4
    assert len(first.pairwise) == 6
    pd.testing.assert_frame_equal(first.two_clusters, second.two_clusters)
    pd.testing.assert_frame_equal(first.four_clusters, second.four_clusters)
    np.testing.assert_allclose(first.ps_cv, second.ps_cv, rtol=1e-13, atol=1e-14)
    assert not np.allclose(first.ps_test, second.ps_test)
    for pair, scores in first.pairwise.items():
        assert len(scores) == 8
        pd.testing.assert_frame_equal(scores, second.pairwise[pair], rtol=1e-13, atol=1e-14)


def test_plot_fig4_som_redesign_returns_all_panels(som_panels):
    for mode in ("report", "slide"):
        fig = plot_fig4_som_redesign(som_panels, mode=mode)
        assert [ax.get_title()[:3] for ax in fig.axes] == ["(A)", "(B)", "(C)", "(D)"]
        assert len(fig.axes[3].get_xticklabels()) == 6
        assert "50 probe" in fig.texts[-1].get_text()
        plt.close(fig)


def test_plot_fig3a_prediction_strength_returns_figure(mock_ps_df):
    for mode in ("report", "slide"):
        fig = plot_fig3a_prediction_strength(mock_ps_df, mode=mode)
        assert isinstance(fig, plt.Figure)
        test_text = "\n".join(text.get_text() for text in fig.axes[1].texts)
        assert "Chắc chắn: 3/4 (75.0%)" in test_text
        assert "Đúng trong số chắc chắn: 2/3 (66.7%)" in test_text
        assert "PB: 0/1 đúng, 1 không chắc chắn" in test_text
        plt.close(fig)


def test_plot_fig4_error_analysis_returns_figure(mock_discovery_data, mock_predictions):
    _, samples = mock_discovery_data
    for mode in ("report", "slide"):
        fig = plot_fig4_error_analysis(mock_predictions, samples, mode=mode)
        assert isinstance(fig, plt.Figure)
        assert "1/8 mẫu phân loại sai" in fig.axes[0].get_title(loc="left")
        assert "PB (Máu ngoại vi, n=4)" in [text.get_text() for text in fig.axes[1].get_legend().get_texts()]
        plt.close(fig)


def test_fig_styles_reject_invalid_mode(mock_discovery_data, mock_predictions, mock_ps_df, som_panels):
    X_df, samples = mock_discovery_data
    with pytest.raises(ValueError):
        plot_fig4_som_redesign(som_panels, mode="invalid")
    with pytest.raises(ValueError):
        plot_fig2_neighborhood(X_df, samples, mode="invalid")

    with pytest.raises(ValueError):
        plot_fig3a_prediction_strength(mock_ps_df, mode="invalid")

    with pytest.raises(ValueError):
        plot_fig4_error_analysis(mock_predictions, samples, mode="invalid")


def test_redesigned_figures_are_exported_in_both_modes():
    stems = (
        "F02_neighborhood_redesign", "F02_som_redesign",
        "F03a_prediction_strength", "F04_error_analysis",
    )
    for mode, folder, dpi in (("report", REPORT_FIG_DIR, 300), ("slide", SLIDE_FIG_DIR, 150)):
        for stem in stems:
            with Image.open(folder / f"{stem}_{mode}.png") as image:
                assert image.info["dpi"] == pytest.approx((dpi, dpi), abs=1)
