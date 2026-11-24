"""Kiểm tra hình F08 (src/viz_ingest.py, TV2 2.4, ADR-031) trên đầu ra thật của ingest.

Cần tầng standardized: chạy `make data` trước (báo lỗi, không skip — ADR-006).
"""
from __future__ import annotations

import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import ingest_batches, sample_quality  # noqa: E402
from src.viz import FIG_STYLES  # noqa: E402
from src.viz_ingest import BLOCKED_COLOR, plot_batch_quality  # noqa: E402


def _texts(fig) -> list[str]:
    return [t.get_text() for ax in fig.axes for t in ax.texts]


def test_f08_marks_the_qc_outlier_once_and_labels_every_batch():
    batches, qc = ingest_batches(), sample_quality()

    fig = plot_batch_quality(batches, qc, mode="report")

    assert len(fig.axes) == 2
    ticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert ticks == ["DFCI\nn = 44", "CALGB\nn = 15", "St-Jude\nn = 8", "CCG\nn = 5"]
    assert _texts(fig).count("mẫu 21 (qc_outlier)") == 1          # chỉ ghi ở panel đầu
    assert fig._suptitle.get_text().startswith("4/4 lô đạt; St-Jude có tỷ lệ call P thấp nhất")
    sizes = [t.get_fontsize() for ax in fig.axes for t in ax.texts + ax.get_xticklabels()]
    assert min(sizes) >= FIG_STYLES["report"]["fontsize"]


def test_f08_highlights_a_blocked_batch_with_the_single_accent_color():
    batches, qc = ingest_batches(), sample_quality()
    # Giả lập lô CCG bị chặn (cùng dữ liệu, chỉ đổi trạng thái) để kiểm cách mã hóa trên hình.
    batches.loc[batches["source"] == "CCG", "status"] = "blocked"
    qc.loc[qc["batch_id"] == "B4-CCG", "batch_status"] = "blocked"

    fig = plot_batch_quality(batches, qc, mode="report")

    ticks = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert ticks[-1] == "CCG\nn = 5 · bị chặn"
    colors = {tuple(c) for coll in fig.axes[0].collections for c in coll.get_facecolors()}
    assert tuple(matplotlib.colors.to_rgba(BLOCKED_COLOR, alpha=0.85)) in colors
    assert fig._suptitle.get_text().startswith("3/4 lô đạt, 1 lô bị chặn")
