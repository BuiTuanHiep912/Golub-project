"""Kiểm tra src/viz.py (TV6, công việc 1.4) bằng dữ liệu tổng hợp nhỏ — không cần dữ liệu thật."""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import viz  # noqa: E402
from src.config import PALETTE  # noqa: E402


@pytest.fixture(scope="module")
def data():
    """30 mẫu × 400 probe kiểu MAS4: 18 ALL (12 B-ALL, 6 T-ALL), 12 AML; 3 nguồn, 2 loại mô."""
    rng = np.random.default_rng(0)
    n, p = 30, 400
    ids = np.arange(1, n + 1)
    cls = np.array(["ALL"] * 18 + ["AML"] * 12)
    sub = np.array(["B-ALL"] * 12 + ["T-ALL"] * 6 + ["AML"] * 12)
    src = np.array(["DFCI"] * 18 + ["CALGB"] * 6 + ["St-Jude"] * 6)
    samples = pd.DataFrame({"split": ["train"] * 20 + ["test"] * 10, "class": cls, "subtype": sub,
                            "tissue": ["BM"] * 24 + ["PB"] * 6, "source": src, "sex": "M",
                            "qc_outlier": [i == 5 for i in ids]}, index=pd.Index(ids, name="sample_id"))
    L = rng.normal(5, 1, p)[None, :] + rng.normal(0, 0.3, (n, p))
    L[:, :60] += np.outer((cls == "AML").astype(float) - 0.5, rng.normal(0, 1, 60))
    raw = pd.DataFrame(np.round(np.exp(L)) - 30, index=samples.index, columns=[f"g{i}" for i in range(p)])
    Xl = np.log10(raw.clip(100, 16000))
    Xf = Xl.loc[:, Xl.std() > Xl.std().median()]
    return raw, Xl, Xf, samples


def test_class_palette_matches_config_and_style_guide():
    assert viz.CLASS_COLORS == PALETTE == {"ALL": "#0072B2", "AML": "#E69F00"}
    assert viz.CLASS_MARKERS == {"ALL": "o", "AML": "^"}
    assert viz.SOURCE_COLORS["DFCI"] == "#009E73"          # khớp F4d của TV1


def test_font_sizes_meet_style_guide():
    assert viz.style_rc("report")["font.size"] >= 10
    assert viz.style_rc("slide")["font.size"] >= 16
    with pytest.raises(ValueError):
        viz.style_rc("poster")


def test_compute_pca_shapes_and_variance(data):
    _, _, Xf, _ = data
    scores, ratio = viz.compute_pca(Xf, 10)
    assert scores.shape == (30, 10) and list(scores.index) == list(Xf.index)
    assert np.all(np.diff(ratio) <= 1e-12) and 0 < ratio.sum() <= 1


def test_log_reduces_skewness(data):
    raw, Xl, _, _ = data
    s = viz.skewness_summary(raw.clip(100, 16000), Xl)
    assert abs(s["skew_median_gene_after"]) < abs(s["skew_median_gene_before"])


def test_pc_class_separation_finds_signal(data):
    _, _, Xf, samples = data
    scores, _ = viz.compute_pca(Xf, 5)
    sep = viz.pc_class_separation(scores, samples)
    assert sep["auc"].between(0.5, 1.0).all() and sep["auc"].max() > 0.9


def test_sample_linkage_is_valid(data):
    _, _, Xf, _ = data
    Z = viz.sample_linkage(np.corrcoef(viz.zscore(Xf)))
    assert Z.shape == (29, 4)


@pytest.mark.parametrize("mode", ["report", "slide"])
def test_render_eda_writes_named_files(data, tmp_path, mode):
    raw, Xl, Xf, samples = data
    paths = viz.render_eda(raw, Xl, Xf, samples, modes=(mode,), base_dir=tmp_path,
                           embedding_methods=("pca", "tsne"))
    sub = "report" if mode == "report" else "slides"
    names = sorted(p.name for p in paths)
    assert names == sorted(f"{stem}_{mode}.png" for stem in (
        "F02_hist_before_after_log", "F02b_gene_variance", "F03_box_per_sample", "F04a_pca_scree",
        "F04b_pca_by_class", "F04c_tsne_umap", "F04e_sample_correlation"))
    assert all(p.parent == tmp_path / sub and p.stat().st_size > 5_000 for p in paths)
    plt.close("all")


def test_pca_scatter_accepts_ax_and_returns_it(data):
    _, _, Xf, samples = data
    fig, ax = plt.subplots()
    out = viz.pca_scatter(Xf, samples, color="source", shape="class", ax=ax)
    assert out is ax
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert {"DFCI", "CALGB", "St-Jude", "ALL", "AML"} <= set(labels)   # màu theo nguồn + hình theo lớp
    plt.close(fig)


def test_eda_summary_keys_and_consistency(data):
    raw, Xl, Xf, samples = data
    d = viz.eda_summary(raw, Xl, Xf, samples)
    assert d["class_counts"] == {"ALL": 18, "AML": 12}
    assert d["qc_outlier_samples"] == [5]
    assert d["pca"]["pcs_to_reach_50pct"] <= d["pca"]["pcs_to_reach_80pct"]
    assert d["gene_filter"]["n_kept"] + d["gene_filter"]["n_removed"] == 400


def test_plots_reject_misaligned_samples(data):
    _, Xl, Xf, samples = data
    with pytest.raises(ValueError, match="cùng sample_id"):
        viz.plot_box_per_sample(Xl, samples.iloc[::-1])
    with pytest.raises(ValueError, match="cùng sample_id"):
        viz.pca_scatter(Xf, samples.iloc[::-1])


def test_eda_summary_reports_measurement_range(data):
    raw, Xl, Xf, samples = data
    mr = viz.eda_summary(raw, Xl, Xf, samples)["measurement_range"]
    assert 0 <= mr["frac_negative"] <= mr["frac_below_100"] <= 1 and 0 <= mr["frac_above_16000"] <= 1
