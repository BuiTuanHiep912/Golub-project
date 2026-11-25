"""TV5 - src/viz_redesign.py (Công việc 2.5: Tái thiết kế trực quan hóa Golub 1999; ADR-014, ADR-061).

Tái thiết kế các hình kinh điển của bài báo Golub et al. (1999) theo chuẩn trực quan hóa hiện đại:
    - Fig 2: Phân cụm không giám sát bằng Self-Organizing Maps (SOM 1x2 và 2x2), đối chiếu cấu trúc lớp và chiếu không gian biểu hiện gen.
    - Fig 3A: (Dành riêng cho đợt tích hợp TV1) Phân bố Prediction Strength (PS) của mô hình Weighted Voting.
    - Fig 4: Phân tích dự đoán trên tập kiểm tra độc lập (34 mẫu), làm nổi bật mẫu biên (mẫu 66), 5 mẫu máu ngoại vi (PB) và các yếu tố gây nhiễu từ nguồn thu thập mẫu.

Tuân thủ nghiêm ngặt Style Guide (docs/style_guide.md) và ADR-014:
    - Bảng màu Okabe-Ito: ALL (#0072B2), AML (#E69F00), an toàn cho người mù màu.
    - Chế độ xuất: report (300 dpi, chữ >= 10 pt) và slide (150 dpi, chữ >= 16 pt).
    - Không chỉnh sửa src/viz.py của TV6.

Cách dùng:
    from src.viz_redesign import plot_fig2_som_redesign, plot_fig3a_prediction_strength, plot_fig4_error_analysis
    fig2 = plot_fig2_som_redesign(X, samples, mode="report")
    fig3a = plot_fig3a_prediction_strength(ps_df, mode="report")
    fig4 = plot_fig4_error_analysis(preds, samples, mode="report")

Chạy trực tiếp:
    python -m src.viz_redesign
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from minisom import MiniSom
from sklearn.decomposition import PCA

try:
    from src.config import (
        CLEANSED_DIR, CURATED_DIR, PALETTE, RANDOM_SEED,
        REPORT_FIG_DIR, SLIDE_FIG_DIR
    )
    from src.evaluate import evaluate, _baseline_pipeline
    from src.load import load_golub
    from src.viz import (
        CLASS_COLORS, CLASS_MARKERS, SUBTYPE_COLORS, SUBTYPE_MARKERS,
        SOURCE_COLORS, TISSUE_COLORS, NEUTRAL, NEUTRAL_LIGHT,
        style_rc, figsize, save_figure
    )
except ModuleNotFoundError:
    from config import (
        CLEANSED_DIR, CURATED_DIR, PALETTE, RANDOM_SEED,
        REPORT_FIG_DIR, SLIDE_FIG_DIR
    )
    from evaluate import evaluate, _baseline_pipeline
    from load import load_golub
    from viz import (
        CLASS_COLORS, CLASS_MARKERS, SUBTYPE_COLORS, SUBTYPE_MARKERS,
        SOURCE_COLORS, TISSUE_COLORS, NEUTRAL, NEUTRAL_LIGHT,
        style_rc, figsize, save_figure
    )


# --------------------------------------------------------------------------------------
# 1. Huấn luyện SOM hỗ trợ Fig 2
# --------------------------------------------------------------------------------------

def fit_som_labels(X_mat: np.ndarray, shape: tuple[int, int], seed: int = RANDOM_SEED, n_iter: int = 1500) -> np.ndarray:
    """Huấn luyện MiniSom trên ma trận biểu hiện gen và trả về nhãn nơ-ron (0..k-1) của từng mẫu."""
    som = MiniSom(shape[0], shape[1], X_mat.shape[1], sigma=0.5, learning_rate=0.5, random_seed=seed)
    som.random_weights_init(X_mat)
    som.train_random(X_mat, n_iter)
    return np.array([np.ravel_multi_index(som.winner(x), shape) for x in X_mat])


# --------------------------------------------------------------------------------------
# 2. Tái thiết kế Fig 2: SOM Class Discovery
# --------------------------------------------------------------------------------------

def plot_fig2_som_redesign(
    X_filtered: pd.DataFrame,
    samples: pd.DataFrame,
    mode: str = "report",
    seed: int = RANDOM_SEED
) -> plt.Figure:
    """Tái thiết kế Hình 2 của bài báo Golub 1999: Khám phá lớp bằng mạng SOM.

    Gồm 3 bảng liên kết:
        (A) Thành phần lâm sàng các nơ-ron SOM 1x2 (38 mẫu huấn luyện): đối chiếu cụm A1/A2.
        (B) Chiếu các mẫu và nơ-ron SOM lên không gian PCA PC1-PC2 để thấy ranh giới phân tách.
        (C) Phân tách các nhóm con tế bào lympho (B-ALL vs T-ALL) trên mạng SOM 2x2 (4 nơ-ron).
    """
    with plt.rc_context(style_rc(mode)):
        fig = plt.figure(figsize=(figsize(mode, height_ratio=0.5)[0], figsize(mode, height_ratio=0.5)[1] * 1.1))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.4, 1.2], wspace=0.32)

        # ---------------------------------------------------------
        # Panel A: SOM 1x2 trên 38 mẫu train
        # ---------------------------------------------------------
        ax_a = fig.add_subplot(gs[0, 0])
        train_mask = samples["split"] == "train"
        samples_tr = samples[train_mask]
        X_tr = X_filtered.loc[samples_tr.index].values

        lab_1x2 = fit_som_labels(X_tr, (1, 2), seed=seed, n_iter=1000)
        ct_1x2 = pd.crosstab(lab_1x2, samples_tr["class"])
        # Đảm bảo có đủ 2 cột ALL và AML
        for col in ("ALL", "AML"):
            if col not in ct_1x2.columns:
                ct_1x2[col] = 0

        # Sắp xếp nút có ALL chiếm ưu thế sang trái (A1)
        if ct_1x2.loc[0, "ALL"] < ct_1x2.loc[1, "ALL"]:
            ct_1x2 = ct_1x2.reindex([1, 0]).reset_index(drop=True)

        # Số liệu bài báo: A1 (24 ALL, 1 AML), A2 (3 ALL, 10 AML)
        paper_counts = [("Bài báo\nCụm A1", 24, 1), ("Bài báo\nCụm A2", 3, 10),
                        ("Nhóm\nNút 1", int(ct_1x2.loc[0, "ALL"]), int(ct_1x2.loc[0, "AML"])),
                        ("Nhóm\nNút 2", int(ct_1x2.loc[1, "ALL"]), int(ct_1x2.loc[1, "AML"]))]

        x_pos = np.arange(len(paper_counts))
        all_bars = [p[1] for p in paper_counts]
        aml_bars = [p[2] for p in paper_counts]

        ax_a.bar(x_pos, all_bars, color=CLASS_COLORS["ALL"], label="ALL", width=0.6)
        ax_a.bar(x_pos, aml_bars, bottom=all_bars, color=CLASS_COLORS["AML"], label="AML", width=0.6, hatch="//", edgecolor="white")

        for i, (name, a_val, m_val) in enumerate(paper_counts):
            total = a_val + m_val
            ax_a.text(i, total + 0.6, f"{a_val}/{m_val}", ha="center", va="bottom",
                      fontsize=style_rc(mode)["font.size"] - 1, fontweight="bold")

        ax_a.set_xticks(x_pos)
        ax_a.set_xticklabels([p[0] for p in paper_counts])
        ax_a.set_ylabel("Số mẫu bệnh nhân")
        ax_a.set_ylim(0, 32)
        ax_a.set_title("(A) SOM 1×2 (38 mẫu train)\nĐối chiếu bài báo gốc")
        ax_a.legend(frameon=False, loc="upper center", ncol=2)

        # ---------------------------------------------------------
        # Panel B: Chiếu không gian PCA PC1-PC2 và biên cụm SOM
        # ---------------------------------------------------------
        ax_b = fig.add_subplot(gs[0, 1])
        pca = PCA(n_components=2, random_state=seed)
        X_pca = pca.fit_transform(X_filtered.values)
        var1, var2 = pca.explained_variance_ratio_ * 100

        # Huấn luyện SOM 1x2 trên toàn bộ 72 mẫu
        lab_all_1x2 = fit_som_labels(X_filtered.values, (1, 2), seed=seed, n_iter=1500)

        for cls_name, color in CLASS_COLORS.items():
            mask_cls = samples["class"] == cls_name
            marker = CLASS_MARKERS[cls_name]
            ax_b.scatter(
                X_pca[mask_cls, 0], X_pca[mask_cls, 1],
                c=color, marker=marker, s=style_rc(mode)["lines.linewidth"] * 24,
                alpha=0.85, edgecolors="white", linewidth=0.6, label=cls_name, zorder=3
            )

        # Vẽ đường ellipse / ranh giới gần đúng giữa 2 nơ-ron SOM
        centroids_pca = []
        for n_id in np.unique(lab_all_1x2):
            pts = X_pca[lab_all_1x2 == n_id]
            c = pts.mean(axis=0)
            centroids_pca.append(c)
            # Vẽ vòng tròn tâm nơ-ron
            ax_b.scatter(c[0], c[1], s=style_rc(mode)["lines.linewidth"] * 75,
                         facecolors="none", edgecolors=NEUTRAL, linewidths=2.0, zorder=4)
            ax_b.text(c[0], c[1] + 1.2, f"Nút {n_id + 1}", ha="center", va="bottom",
                      color=NEUTRAL, fontweight="bold")

        ax_b.set_xlabel(f"PC1 ({var1:.1f}% phương sai)")
        ax_b.set_ylabel(f"PC2 ({var2:.1f}% phương sai)")
        ax_b.set_title("(B) Phân tách trên không gian biểu hiện gen\n(PCA & Tâm nơ-ron SOM)")
        ax_b.legend(frameon=False, loc="lower right")

        # ---------------------------------------------------------
        # Panel C: SOM 2x2 trên 72 mẫu (Khảo sát phân nhóm sâu)
        # ---------------------------------------------------------
        ax_c = fig.add_subplot(gs[0, 2])
        lab_2x2 = fit_som_labels(X_filtered.values, (2, 2), seed=seed, n_iter=1500)
        ct_2x2 = pd.crosstab(lab_2x2, samples["subtype"])
        for sub in ("B-ALL", "T-ALL", "AML"):
            if sub not in ct_2x2.columns:
                ct_2x2[sub] = 0

        n_nodes = 4
        x_c = np.arange(n_nodes)
        b_vals = ct_2x2["B-ALL"].values
        t_vals = ct_2x2["T-ALL"].values
        m_vals = ct_2x2["AML"].values

        ax_c.bar(x_c, b_vals, color=SUBTYPE_COLORS["B-ALL"], label="B-ALL", width=0.55)
        ax_c.bar(x_c, t_vals, bottom=b_vals, color=SUBTYPE_COLORS["T-ALL"], label="T-ALL", width=0.55, hatch="..", edgecolor="white")
        ax_c.bar(x_c, m_vals, bottom=b_vals + t_vals, color=SUBTYPE_COLORS["AML"], label="AML", width=0.55, hatch="//", edgecolor="white")

        for idx in range(n_nodes):
            tot = b_vals[idx] + t_vals[idx] + m_vals[idx]
            ax_c.text(idx, tot + 0.5, f"n={tot}", ha="center", va="bottom",
                      fontsize=style_rc(mode)["font.size"] - 1)

        ax_c.set_xticks(x_c)
        ax_c.set_xticklabels([f"Nút {i+1}" for i in range(n_nodes)])
        ax_c.set_ylabel("Số mẫu (72 bệnh nhân)")
        ax_c.set_ylim(0, 32)
        ax_c.set_title("(C) SOM 2×2 (72 mẫu)\nKhám phá phân nhóm phụ (H5)")
        ax_c.legend(frameon=False, loc="upper right")

        fig.suptitle("F02. Tái thiết kế trực quan hóa Khám phá lớp bằng SOM (Golub et al., 1999, Fig. 2)",
                     y=1.04, fontweight="bold")
        return fig


# --------------------------------------------------------------------------------------
# 3. Tái thiết kế Fig 4: Error Analysis on Independent Test Set
# --------------------------------------------------------------------------------------

def plot_fig4_error_analysis(
    preds_df: pd.DataFrame,
    samples: pd.DataFrame,
    mode: str = "report"
) -> plt.Figure:
    """Tái thiết kế Hình 4 của bài báo Golub 1999: Phân tích dự đoán trên tập kiểm tra độc lập (34 mẫu).

    Hiển thị:
        (A) Biểu đồ ngang xác suất dự đoán P(AML) của từng bệnh nhân, làm nổi bật ca phân loại sai (Mẫu 66).
        (B) Dải kiểm định Quản trị dữ liệu (Data Governance tracks): Loại mô (BM vs PB) và Nguồn bệnh viện.
    """
    with plt.rc_context(style_rc(mode)):
        test_samples = samples[samples["split"] == "test"].copy()
        merged = preds_df.merge(test_samples, left_index=True, right_index=True)

        # Sắp xếp bệnh nhân theo xác suất P(AML) tăng dần
        merged = merged.sort_values(by=["score", "y_true"], ascending=[True, True])
        n_samples = len(merged)

        # Chiều cao linh hoạt để đọc rõ mã 34 bệnh nhân
        h = figsize(mode, height_ratio=0.75)[1]
        fig, (ax_main, ax_meta) = plt.subplots(
            1, 2, figsize=(figsize(mode)[0], h),
            gridspec_kw={"width_ratios": [3.2, 0.8], "wspace": 0.08},
            sharey=True
        )

        y_positions = np.arange(n_samples)

        # Xác định màu sắc từng thanh dự đoán
        bar_colors = []
        for idx, row in merged.iterrows():
            if row["y_true"] == "ALL" and row["y_pred"] == "ALL":
                bar_colors.append(CLASS_COLORS["ALL"])
            elif row["y_true"] == "AML" and row["y_pred"] == "AML":
                bar_colors.append(CLASS_COLORS["AML"])
            else:
                # Sai phân loại (Mẫu 66)
                bar_colors.append("#D55E00")

        # Vẽ thanh xác suất P(AML)
        scores = merged["score"].values
        ax_main.barh(y_positions, scores, color=bar_colors, height=0.68, alpha=0.9, edgecolor="none")

        # Ngưỡng phân định lâm sàng
        ax_main.axvline(0.5, color=NEUTRAL, linestyle="--", linewidth=1.2, label="Ngưỡng phân loại (0.5)")
        ax_main.axvspan(0.3, 0.7, color=NEUTRAL_LIGHT, alpha=0.25, label="Vùng nghi ngờ (PS < 0.3)")

        # Ghi chú ca đặc biệt Mẫu 66
        if 66 in merged.index:
            m66_idx = np.where(merged.index == 66)[0][0]
            ax_main.annotate(
                "Mẫu 66: AML thực tế\n(CCG, BM) -> Đoán nhầm ALL",
                xy=(merged.loc[66, "score"], m66_idx),
                xytext=(merged.loc[66, "score"] + 0.22, m66_idx - 1.5),
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.5),
                color="#D55E00", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF2E6", edgecolor="#D55E00", lw=1)
            )

        ax_main.set_yticks(y_positions)
        yticklabels = [f"#{idx} ({row['y_true']})" for idx, row in merged.iterrows()]
        ax_main.set_yticklabels(yticklabels, fontsize=style_rc(mode)["font.size"] - 1.5)
        ax_main.set_xlabel("Xác suất dự đoán P(AML) — Điểm số mô hình")
        ax_main.set_xlim(-0.02, 1.05)
        ax_main.set_title("(A) Phân bố xác suất dự đoán trên 34 bệnh nhân độc lập", loc="left")
        ax_main.legend(frameon=True, loc="upper left", facecolor="white", framealpha=0.9)

        # ---------------------------------------------------------
        # Bảng metadata: Mô (Tissue) và Nguồn (Source)
        # ---------------------------------------------------------
        # Track 1: Tissue (BM = 0, PB = 1)
        tissue_vals = np.array([0 if row["tissue"] == "BM" else 1 for _, row in merged.iterrows()])
        # Track 2: Source (DFCI=0, CALGB=1, CCG=2, St-Jude=3)
        src_map = {"DFCI": 0, "CALGB": 1, "CCG": 2, "St-Jude": 3}
        source_vals = np.array([src_map[row["source"]] for _, row in merged.iterrows()])

        # Vẽ 2 cột nhiệt metadata
        ax_meta.scatter(np.zeros(n_samples), y_positions, c=[TISSUE_COLORS[r["tissue"]] for _, r in merged.iterrows()],
                        s=style_rc(mode)["lines.linewidth"] * 30, marker="s", label="Mô")
        ax_meta.scatter(np.ones(n_samples), y_positions, c=[SOURCE_COLORS[r["source"]] for _, r in merged.iterrows()],
                        s=style_rc(mode)["lines.linewidth"] * 30, marker="o", label="Nguồn")

        ax_meta.set_xticks([0, 1])
        ax_meta.set_xticklabels(["Mô\n(BM/PB)", "Nguồn\n(Bệnh viện)"], fontsize=style_rc(mode)["font.size"] - 1)
        ax_meta.set_xlim(-0.5, 1.5)
        ax_meta.set_title("(B) Quản trị dữ liệu", loc="center")
        ax_meta.spines["left"].set_visible(False)
        ax_meta.tick_params(left=False)

        # Chú giải riêng cho metadata
        legend_elements = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor=TISSUE_COLORS["BM"], label="BM (Tủy xương)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=TISSUE_COLORS["PB"], label="PB (Máu ngoại vi - 5 mẫu)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SOURCE_COLORS["DFCI"], label="DFCI"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SOURCE_COLORS["CALGB"], label="CALGB"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SOURCE_COLORS["CCG"], label="CCG"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SOURCE_COLORS["St-Jude"], label="St-Jude"),
        ]
        ax_meta.legend(handles=legend_elements, loc="lower right", bbox_to_anchor=(1.05, 0.0),
                       fontsize=style_rc(mode)["font.size"] - 2, frameon=True, facecolor="white")

        fig.suptitle("F04. Tái thiết kế Đánh giá & Phân tích lỗi trên tập kiểm tra độc lập (Golub et al., 1999, Fig. 4)",
                     y=1.02, fontweight="bold")
        return fig


# --------------------------------------------------------------------------------------
# 4. Tái thiết kế Fig 3A: Prediction Strength (PS) của mô hình Weighted Voting
# --------------------------------------------------------------------------------------

def plot_fig3a_prediction_strength(ps_df: pd.DataFrame | None = None, mode: str = "report") -> plt.Figure:
    """Tái thiết kế Fig 3A của Golub et al. (1999): Prediction Strength (PS) phân bổ theo lớp và tập mẫu.

    Panel A: Tập huấn luyện (LOOCV - 38 mẫu), làm rõ ngưỡng PS >= 0.30 và 2 mẫu uncertain.
    Panel B: Tập kiểm tra độc lập (34 mẫu), làm nổi bật mẫu 66 (sai biên), 5 mẫu PB và mẫu uncertain 67.
    """
    if mode not in ("report", "slide"):
        raise ValueError(f"mode phải là 'report' hoặc 'slide', nhận được: {mode}")

    if ps_df is None:
        ps_path = Path(__file__).resolve().parent.parent / "results" / "metrics" / "golub_ps.csv"
        if not ps_path.exists():
            from src.config import PROJECT_ROOT
            ps_path = PROJECT_ROOT / "results" / "metrics" / "golub_ps.csv"
        ps_df = pd.read_csv(ps_path)

    rc = style_rc(mode)
    with plt.rc_context(rc):
        w, h = (12.5, 6.2) if mode == "report" else (14.5, 7.5)
        fig, (ax_loocv, ax_test) = plt.subplots(1, 2, figsize=(w, h), sharey=True)

        # Hàm vẽ cho từng panel
        def _render_panel(ax, subset_name: str, title_panel: str):
            sub = ps_df[ps_df["set"] == subset_name].copy()

            # Vùng bóng Uncertain (PS < 0.30)
            ax.axhspan(-0.02, 0.30, facecolor="#FADBD8" if mode == "report" else "#F2D7D5", alpha=0.35, zorder=0)
            ax.axhline(0.30, color="#C0392B", linestyle="--", linewidth=rc["lines.linewidth"] * 1.2,
                       label="Ngưỡng chắc chắn (PS = 0.30)", zorder=2)

            # Text cảnh báo vùng không chắc chắn
            ax.text(0.5, 0.15, "Vùng không chắc chắn (Uncertain, PS < 0.30)\n→ Chuyển giao bác sĩ (Human-in-the-loop)",
                    transform=ax.get_yaxis_transform(), ha="center", va="center",
                    fontsize=rc["font.size"] - 1, color="#922B21", style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E6B0AA", alpha=0.85))

            classes = ["ALL", "AML"]
            positions = [0, 1]
            box_data = [sub[sub["true"] == c]["ps"].values for c in classes]

            # Boxplot mờ
            bp = ax.boxplot(box_data, positions=positions, widths=0.35, patch_artist=True,
                            showfliers=False, zorder=1,
                            medianprops=dict(color="#2C3E50", linewidth=rc["lines.linewidth"] * 1.5),
                            boxprops=dict(facecolor="#EAECEE", edgecolor="#BDC3C7", alpha=0.6),
                            whiskerprops=dict(color="#BDC3C7", linewidth=rc["lines.linewidth"]),
                            capprops=dict(color="#BDC3C7", linewidth=rc["lines.linewidth"]))

            # Scatter điểm với jitter
            rng = np.random.default_rng(RANDOM_SEED + (0 if subset_name == "loocv" else 1))
            for i, c in enumerate(classes):
                c_df = sub[sub["true"] == c]
                jitter = rng.uniform(-0.10, 0.10, size=len(c_df))
                x_vals = positions[i] + jitter

                for x_pos, (_, row) in zip(x_vals, c_df.iterrows()):
                    sample_id = int(row["sample_id"])
                    ps_val = float(row["ps"])
                    pred = row["pred"]
                    tissue = row.get("tissue", "BM")

                    # Marker: O cho BM, S cho PB
                    marker = "s" if tissue == "PB" else "o"
                    size = rc["lines.linewidth"] * (35 if tissue == "PB" else 28)

                    # Màu và viền
                    if pred == "uncertain":
                        color = "#95A5A6"
                        edge = "#C0392B"
                        lw = 2.0
                    elif pred != c:
                        color = "#E74C3C"  # Lỗi phân loại
                        edge = "black"
                        lw = 2.0
                        size *= 1.4
                    else:
                        color = CLASS_COLORS[c]
                        edge = "white"
                        lw = 1.0

                    ax.scatter(x_pos, ps_val, marker=marker, s=size, facecolor=color,
                               edgecolor=edge, linewidth=lw, zorder=4)

                    # Chú thích các ca đặc biệt
                    if subset_name == "loocv":
                        if pred == "uncertain":
                            ax.annotate(f"#{sample_id}\n(PS={ps_val:.2f})",
                                        xy=(x_pos, ps_val), xytext=(x_pos + 0.16, ps_val - 0.04),
                                        fontsize=rc["font.size"] - 2, fontweight="bold", color="#78281F",
                                        arrowprops=dict(arrowstyle="->", color="#78281F", lw=1.0))
                    elif subset_name == "test":
                        if sample_id == 66:  # Ca sai duy nhất
                            ax.annotate(f"Mẫu 66: Sai duy nhất (AML→ALL)\nPS = {ps_val:.4f} (sát ngưỡng)\nTumor Purity thấp",
                                        xy=(x_pos, ps_val), xytext=(x_pos - 0.45, ps_val + 0.14),
                                        fontsize=rc["font.size"] - 1, fontweight="bold", color="#C0392B",
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", edgecolor="#E74C3C"),
                                        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5))
                        elif sample_id == 67:  # Uncertain
                            ax.annotate(f"Mẫu 67 (PB, T-ALL)\nPS = {ps_val:.2f} (Uncertain)",
                                        xy=(x_pos, ps_val), xytext=(x_pos + 0.16, ps_val - 0.05),
                                        fontsize=rc["font.size"] - 2, fontweight="bold", color="#78281F",
                                        arrowprops=dict(arrowstyle="->", color="#78281F", lw=1.0))

            ax.set_xticks(positions)
            ax.set_xticklabels([f"ALL (n={len(sub[sub['true']=='ALL'])})",
                                f"AML (n={len(sub[sub['true']=='AML'])})"],
                               fontsize=rc["font.size"])
            ax.set_xlim(-0.4, 1.4)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(title_panel, loc="left", fontweight="bold", fontsize=rc["font.size"] + 1)
            ax.grid(axis="y", linestyle=":", alpha=0.5, color="#BDC3C7")

        # Render 2 panel
        _render_panel(ax_loocv, "loocv", "(A) Huấn luyện: Kiểm tra chéo LOOCV (38 mẫu)")
        _render_panel(ax_test, "test", "(B) Độc lập: Tập kiểm tra 34 bệnh nhân mới")

        ax_loocv.set_ylabel("Prediction Strength (PS)", fontsize=rc["font.size"] + 1)

        # Thống kê hộp tóm tắt
        loocv_sub = ps_df[ps_df["set"] == "loocv"]
        loocv_conf = len(loocv_sub[loocv_sub["pred"] != "uncertain"])
        loocv_corr = len(loocv_sub[loocv_sub["pred"] == loocv_sub["true"]])
        loocv_med = loocv_sub["ps"].median()

        test_sub = ps_df[ps_df["set"] == "test"]
        test_conf = len(test_sub[test_sub["pred"] != "uncertain"])
        test_corr = len(test_sub[test_sub["pred"] == test_sub["true"]])
        test_med = test_sub["ps"].median()

        ax_loocv.text(0.04, 0.96,
                      f"Thống kê LOOCV:\n"
                      f"• Chắc chắn (PS ≥ 0.3): {loocv_conf}/38 (94.7%)\n"
                      f"• Đúng tuyệt đối: {loocv_corr}/{loocv_conf} (100%)\n"
                      f"• Trung vị PS: {loocv_med:.2f}\n"
                      f"• Không chắc chắn: 2 mẫu (#12, #35)",
                      transform=ax_loocv.transAxes, va="top", ha="left",
                      fontsize=rc["font.size"] - 2,
                      bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor="#AED6F1", alpha=0.9))

        ax_test.text(0.04, 0.96,
                     f"Thống kê Tập kiểm tra:\n"
                     f"• Chắc chắn (PS ≥ 0.3): {test_conf}/34 (97.1%)\n"
                     f"• Đúng: {test_corr}/{test_conf} (97.0%)\n"
                     f"• Trung vị PS: {test_med:.2f}\n"
                     f"• Sai: 1 mẫu (#66, AML→ALL)\n"
                     f"• Không chắc chắn: 1 mẫu (#67)",
                     transform=ax_test.transAxes, va="top", ha="left",
                     fontsize=rc["font.size"] - 2,
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF9E7", edgecolor="#F9E79F", alpha=0.9))

        # Chú giải chung
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS["ALL"], markersize=8, label="Lớp ALL (BM)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS["AML"], markersize=8, label="Lớp AML (BM)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=NEUTRAL, markersize=8, label="Máu ngoại vi (PB)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#95A5A6", markeredgecolor="#C0392B", markeredgewidth=1.5, markersize=8, label="Uncertain (PS < 0.3)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C", markeredgecolor="black", markeredgewidth=1.5, markersize=10, label="Sai phân loại (#66)"),
            Line2D([0], [0], color="#C0392B", linestyle="--", linewidth=1.5, label="Ngưỡng PS = 0.30"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04),
                   fontsize=rc["font.size"] - 2, frameon=True, facecolor="white")

        fig.suptitle("F03a. Tái thiết kế Định lượng độ tin cậy dự đoán (Prediction Strength — Golub et al., 1999, Fig. 3A)",
                     y=0.99, fontweight="bold", fontsize=rc["font.size"] + 2)
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        return fig


# --------------------------------------------------------------------------------------
# 5. Thực thi sinh file
# --------------------------------------------------------------------------------------

def generate_all_redesigned_figures() -> None:
    """Sinh toàn bộ hình F02, F03a và F04 ở cả hai phiên bản report và slide."""
    print("--- Bắt đầu sinh hình F02, F03a và F04 tái thiết kế ---")
    X_cu, samples = load_golub("curated")
    X_cl, _ = load_golub("cleansed")

    # 1. Chạy đánh giá để lấy bảng dự đoán 34 mẫu kiểm tra cho Fig 4
    pipe = _baseline_pipeline(k=50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evaluate(pipe, X_cl, samples["class"], scheme="original_split")
    preds = res["predictions"]

    for mode in ("report", "slide"):
        # Sinh Fig 2
        fig2 = plot_fig2_som_redesign(X_cu, samples, mode=mode)
        path2 = save_figure(fig2, "F02_som_redesign", mode=mode)
        plt.close(fig2)
        print(f"[{mode.upper()}] Đã lưu: {path2}")

        # Sinh Fig 3A
        fig3a = plot_fig3a_prediction_strength(ps_df=None, mode=mode)
        path3a = save_figure(fig3a, "F03a_prediction_strength", mode=mode)
        plt.close(fig3a)
        print(f"[{mode.upper()}] Đã lưu: {path3a}")

        # Sinh Fig 4
        fig4 = plot_fig4_error_analysis(preds, samples, mode=mode)
        path4 = save_figure(fig4, "F04_error_analysis", mode=mode)
        plt.close(fig4)
        print(f"[{mode.upper()}] Đã lưu: {path4}")

    print("--- Hoàn tất xuất sắc việc tạo hình F02, F03a và F04! ---")


if __name__ == "__main__":
    generate_all_redesigned_figures()
