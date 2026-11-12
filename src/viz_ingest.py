"""Hình F8 — chất lượng dữ liệu theo lô (phòng lab), vẽ từ đầu ra của `make ingest`.

Công việc: 2.4 — Phụ trách: TV2 (ADR-031). Theo ADR-014, hàm vẽ mới để trong file riêng
`src/viz_<chủ đề>.py`; kiểu dáng (cỡ chữ, dpi, màu trung tính, lưu 2 bản) lấy từ `src/viz.py`.

Câu hỏi của hình: bốn lô có chất lượng tương đương không, lô hay mẫu nào lệch phần còn lại?
  - Panel trái: tương quan trung vị của từng mẫu với mẫu các lô khác (batch effect, chip lỗi).
  - Panel phải: tỷ lệ probe có call "P" của từng mẫu (lượng RNA / độ nhạy của chip).
  Mỗi điểm là một mẫu, vạch đen là trung vị của lô. Điểm xám = lô đạt; màu nhấn duy nhất dành
  cho lô bị chặn. Mẫu gắn cờ qc_outlier được khoanh đen và ghi mã (style guide mục 3).

Chạy: python -m src.viz_ingest   (sau `make ingest`; ghi F08_chat_luong_theo_lo_{report,slide}.png)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.config import CLEANSED_DIR, RANDOM_SEED
    from src.ingest import BATCHES_FILE, QC_FILE
    from src.viz import FIG_STYLES, MODES, NEUTRAL, figsize, save_figure, styled
except ModuleNotFoundError:  # chạy trực tiếp `python src/viz_ingest.py`
    from config import CLEANSED_DIR, RANDOM_SEED
    from ingest import BATCHES_FILE, QC_FILE
    from viz import FIG_STYLES, MODES, NEUTRAL, figsize, save_figure, styled

FIG_STEM = "F08_chat_luong_theo_lo"
# Màu nhấn duy nhất của hình: lô bị chặn (vermillion Okabe–Ito, như F10).
BLOCKED_COLOR = "#D55E00"
PANELS = (("median_corr_other", "Tương quan trung vị\nvới mẫu các lô khác"),
          ("present_rate", "Tỷ lệ probe có call P"))


@styled
def plot_batch_quality(batches: pd.DataFrame, qc: pd.DataFrame, mode: str = "report") -> plt.Figure:
    """F08 — strip plot chất lượng theo mẫu, nhóm theo lô (cột batch_id của qc khớp batches).

    batches: bảng ingest_batches() (dim_batch); qc: bảng sample_quality() (fact_qc).
    Lô hỏng cấu trúc không có hàng trong qc nên chỉ hiện ở nhãn trục (n mẫu, "bị chặn").
    """
    st = FIG_STYLES[mode]
    order = list(batches["batch_id"])
    fig, axes = plt.subplots(1, len(PANELS), figsize=figsize(mode, 0.55), sharex=True)
    rng = np.random.default_rng(RANDOM_SEED)
    status = batches.set_index("batch_id")["status"]
    for k, (ax, (col, label)) in enumerate(zip(axes, PANELS)):
        for i, bid in enumerate(order):
            part = qc[qc["batch_id"] == bid]
            vals = part[col].to_numpy(dtype=float)
            if len(vals) == 0 or np.isnan(vals).all():
                continue
            color = BLOCKED_COLOR if status[bid] == "blocked" else NEUTRAL
            x = i + rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(x, vals, s=st["marker"] * 0.6, color=color, alpha=0.85, edgecolor="black",
                       linewidths=0.3, zorder=3)
            med = float(np.nanmedian(vals))
            ax.plot([i - 0.28, i + 0.28], [med, med], color="black", lw=st["lw"], zorder=4)
            # Mẫu bất thường: khoanh tròn đen ở mọi panel để dò được cùng một mẫu; mã mẫu chỉ ghi ở
            # panel đầu (cờ qc_outlier dựa trên tương quan), panel sau ghi lại sẽ đè lên các điểm.
            for xi, (_, r) in zip(x, part.iterrows()):
                if r["qc_outlier"]:
                    ax.scatter([xi], [r[col]], s=st["marker"] * 3, facecolor="none", edgecolor="black",
                               linewidths=1.2, zorder=5)
                    if k == 0:
                        ax.annotate(f"mẫu {int(r['sample_id'])} (qc_outlier)", (xi, r[col]),
                                    xytext=(8, 0), textcoords="offset points", va="center")
        ax.set_ylabel(label)
        ax.set_xticks(range(len(order)))
    ticks = []
    for _, b in batches.iterrows():
        mark = " · bị chặn" if b["status"] == "blocked" else ""
        ticks.append(f"{b['source']}\nn = {b['n_samples']}{mark}")
    for ax in axes:
        ax.set_xticklabels(ticks)
        ax.set_xlim(-0.6, len(order) - 0.4)

    passed = batches[batches["status"] == "passed"]
    low = passed.loc[passed["present_rate"].idxmin()]
    high = passed.loc[passed["present_rate"].idxmax()]
    n_blocked = int((batches["status"] == "blocked").sum())
    head = f"{len(passed)}/{len(batches)} lô đạt" + (f", {n_blocked} lô bị chặn" if n_blocked else "")
    fig.suptitle(f"{head}; {low['source']} có tỷ lệ call P thấp nhất "
                 f"({low['present_rate']:.0%} so với {high['present_rate']:.0%} ở {high['source']})")
    fig.tight_layout()
    return fig


def main() -> None:
    paths = [CLEANSED_DIR / BATCHES_FILE, CLEANSED_DIR / QC_FILE]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu {missing} trong {CLEANSED_DIR} — chạy `make ingest` trước.")
    batches, qc = (pd.read_parquet(p) for p in paths)
    for mode in MODES:
        fig = plot_batch_quality(batches, qc, mode=mode)
        print("Đã ghi", save_figure(fig, FIG_STEM, mode))
        plt.close(fig)


if __name__ == "__main__":
    main()
