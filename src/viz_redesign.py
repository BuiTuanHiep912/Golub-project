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
    from src.viz_redesign import plot_fig2_som_redesign, plot_fig4_error_analysis
    fig2 = plot_fig2_som_redesign(X, samples, mode="report")
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
# 4. Thực thi sinh file
# --------------------------------------------------------------------------------------

def generate_all_redesigned_figures() -> None:
    """Sinh toàn bộ hình F02 và F04 ở cả hai phiên bản report và slide."""
    print("--- Bắt đầu sinh hình F02 và F04 tái thiết kế ---")
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

        # Sinh Fig 4
        fig4 = plot_fig4_error_analysis(preds, samples, mode=mode)
        path4 = save_figure(fig4, "F04_error_analysis", mode=mode)
        plt.close(fig4)
        print(f"[{mode.upper()}] Đã lưu: {path4}")

    print("--- Hoàn tất xuất sắc việc tạo hình F02 và F04! ---")


if __name__ == "__main__":
    generate_all_redesigned_figures()
