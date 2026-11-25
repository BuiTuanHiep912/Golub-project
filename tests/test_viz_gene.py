"""Kiểm tra src/viz_gene.py (TV4, công việc 2.2) trên dữ liệu tổng hợp, không cần dữ liệu thật.

Chỉ kiểm tra hình vẽ được, đúng số lớp/nhóm và báo lỗi rõ khi thiếu cột; không so sánh điểm ảnh.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import viz_gene as vg  # noqa: E402


@pytest.fixture
def ranking():
    rng = np.random.default_rng(1)
    n = 120
    snr = rng.normal(0, 0.6, n)
    return pd.DataFrame({
        "rank": np.argsort(np.argsort(-np.abs(snr))) + 1,
        "probe_id": [f"P{i}_at" for i in range(n)],
        "gene_symbol": [f"G{i}" if i % 3 else "" for i in range(n)],
        "snr": snr,
        "q_value": np.clip(np.exp(-6 * np.abs(snr)), 1e-8, 1),
    })


@pytest.fixture
def values():
    rows = []
    sid = 1
    for split, cls, src, tissue, k in [("train", "ALL", "DFCI", "BM", 10), ("train", "AML", "CALGB", "BM", 6),
                                       ("test", "ALL", "St-Jude", "BM", 3), ("test", "AML", "St-Jude", "PB", 4),
                                       ("test", "AML", "CCG", "BM", 3)]:
        for _ in range(k):
            rows.append({"sample_id": sid, "value": 2.2 + (0.8 if cls == "AML" else 0) + 0.05 * sid % 0.3,
                         "class": cls, "subtype": "AML" if cls == "AML" else "B-ALL", "source": src,
                         "tissue": tissue, "split": split, "qc_outlier": sid == 3})
            sid += 1
    return pd.DataFrame(rows)


@pytest.fixture
def consistency():
    df = pd.DataFrame({
        "rank": [1, 2, 3, 4, 5],
        "probe_id": ["A_at", "B_at", "C_at", "D_at", "E_at"],
        "gene_symbol": ["GA", "GB", "", "GD", "GE"],
        "snr": [1.2, 0.9, -1.1, -0.8, 0.5],
        "train_diff": [0.8, 0.6, -0.7, -0.5, 0.3],
        "batch_diff": [0.9, -0.4, -0.6, 0.02, 0.0],
        "same_direction": pd.array([True, False, True, False, False], dtype="boolean"),
        "status": ["Giữ chiều", "Đảo chiều", "Giữ chiều", "Chênh lệch nhỏ", "Chênh lệch nhỏ"],
    })
    df.attrs.update(n_ALL=3, n_AML=5, note="lô thử")
    return df


# ---------------------------------------------------------------- Altair

altair_only = pytest.mark.skipif(vg.alt is None, reason="chưa cài altair")


@altair_only
def test_volcano_chart_builds_and_highlights(ranking):
    spec = vg.volcano_chart(ranking, highlight="P5_at").to_dict()
    assert len(spec["layer"]) == 4                            # điểm, ngưỡng, vòng, tên gen


@altair_only
def test_volcano_chart_unknown_highlight_is_ignored(ranking):
    assert len(vg.volcano_chart(ranking, highlight="KHONGCO").to_dict()["layer"]) == 2


@altair_only
@pytest.mark.parametrize("facet", ["source", "tissue"])
def test_gene_expression_chart_builds(values, facet):
    spec = vg.gene_expression_chart(values, facet=facet).to_dict()
    assert spec["facet"]["column"]["field"] == facet


@altair_only
def test_gene_expression_chart_without_qc_column(values):
    vg.gene_expression_chart(values.drop(columns="qc_outlier")).to_dict()


@altair_only
def test_consistency_chart_lists_flipped_genes(consistency):
    spec = vg.consistency_chart(consistency).to_dict()
    assert "GB" in spec["title"]["subtitle"][-1]                 # dòng cuối: tên gen đảo chiều
    assert len(spec["title"]["text"]) < 60                       # tiêu đề ngắn, note dài nằm ở phụ đề


@altair_only
def test_snr_bar_chart_balanced_and_unique_labels(ranking):
    df = vg._snr_frame(ranking, 10)
    assert (df["direction"] == "ALL").sum() == 10 and (df["direction"] == "AML").sum() == 10
    assert df["label"].is_unique
    vg.snr_bar_chart(ranking, 10).to_dict()


# ---------------------------------------------------------------- matplotlib

@pytest.mark.parametrize("mode", ["report", "slide"])
def test_static_plots_return_figures(ranking, values, consistency, mode):
    figs = [vg.plot_volcano(ranking, highlight="P5_at", mode=mode),
            vg.plot_gene_expression(values, facet="source", mode=mode),
            vg.plot_gene_expression(values, facet="tissue", mode=mode),
            vg.plot_consistency(consistency, mode=mode)]
    for fig in figs:
        assert isinstance(fig, Figure)
    assert len(figs[1].axes) == 1                             # một trục chung, không chia ô
    assert len(figs[1].axes[0].get_xticks()) == 5             # DFCI ALL, CALGB AML, St-Jude ALL+AML, CCG AML
    assert len(figs[2].axes[0].get_xticks()) == 3             # BM ALL+AML, PB AML
    import matplotlib.pyplot as plt
    for fig in figs:
        plt.close(fig)


def test_gene_expression_only_groups_with_samples(values):
    fig = vg.plot_gene_expression(values, facet="source")
    ax = fig.axes[0]
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert ticks[0] == "ALL\nn=10" and ticks[1] == "AML\nn=6"   # DFCI chỉ có ALL, CALGB chỉ có AML: không cột n=0
    headers = [t.get_text() for t in ax.texts]
    assert "DFCI" in headers and "St-Jude · đủ 2 lớp" in headers and "CCG" in headers
    assert any("mức sàn" in h for h in headers) is False      # dữ liệu giả không chạm mức sàn: không vẽ vạch sàn


def test_expression_groups_positions(values):
    groups = vg._expression_groups(values, "source")
    assert [(lv, c) for lv, c, _ in groups] == [("DFCI", "ALL"), ("CALGB", "AML"), ("St-Jude", "ALL"),
                                                ("St-Jude", "AML"), ("CCG", "AML")]
    xs = [x for *_, x in groups]
    assert xs[3] - xs[2] == 1.0 and xs[2] - xs[1] == 1.0 + vg.GROUP_GAP   # cùng lab sát nhau, khác lab có khoảng


def test_gene_expression_by_subtype_and_ref_line(values):
    fig = vg.plot_gene_expression(values, facet="subtype", ref_line=(2.5, "mốc"))
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts if t.get_text() in ("B-ALL", "AML")] == ["B-ALL", "AML"]
    assert any(t.get_text() == "mốc" for t in ax.texts)
    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.fixture
def sex():
    return pd.DataFrame({"sample_id": [1, 2, 3, 4], "rps4y": [2.0, 3.3, 3.2, 2.05], "xist": [2.7, 2.2, 2.1, 2.6],
                         "sex_recorded": ["F", "M", "F", None], "sex_inferred": ["F", "M", "M", "F"],
                         "status": ["Khớp", "Khớp", "Lệch metadata", "Metadata thiếu"],
                         "source": ["DFCI"] * 4})


def test_plot_sex_check_labels_mismatch(sex):
    fig = vg.plot_sex_check(sex)
    assert any(t.get_text() == "mẫu 3" for t in fig.axes[0].texts)
    import matplotlib.pyplot as plt
    plt.close(fig)
    with pytest.raises(ValueError):
        vg.plot_sex_check(sex.drop(columns="xist"))


@altair_only
def test_sex_check_chart_builds(sex):
    assert len(vg.sex_check_chart(sex).to_dict()["layer"]) == 2


def test_swarm_offsets_spread_ties_and_stay_inside():
    off = vg._swarm_offsets(np.full(30, 2.0), y_span=1.0, step=0.075, half_width=0.4)
    assert len(np.unique(off.round(6))) > 10                  # 30 mẫu cùng mức sàn được dàn ra, không chồng
    assert np.abs(off).max() <= 0.4 + 1e-9
    assert vg._swarm_offsets(np.array([1.0]), 1.0, 0.075, 0.4).tolist() == [0.0]


# ---------------------------------------------------------------- kiểm tra đầu vào

def test_volcano_requires_columns():
    with pytest.raises(ValueError):
        vg.plot_volcano(pd.DataFrame({"probe_id": ["a"], "snr": [1.0]}))


def test_consistency_requires_marker_consistency_output():
    with pytest.raises(ValueError):
        vg.plot_consistency(pd.DataFrame({"probe_id": ["a"]}))


def test_consistency_status_derived_when_missing(consistency):
    df = vg._consistency_frame(consistency.drop(columns="status"))
    assert df.set_index("probe_id").loc["B_at", "status"] == "Đảo chiều"


def test_gene_expression_bad_facet(values):
    with pytest.raises(ValueError):
        vg.plot_gene_expression(values, facet="class")
