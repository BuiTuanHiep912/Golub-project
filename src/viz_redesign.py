"""TV5 - src/viz_redesign.py (Công việc 2.5: Tái thiết kế trực quan hóa Golub 1999; ADR-014, ADR-061).

Tái thiết kế các hình kinh điển của bài báo Golub et al. (1999) theo chuẩn trực quan hóa hiện đại:
    - Fig 2: Phân tích lân cận ALL/AML với 400 lần hoán vị nhãn.
    - Fig 4: Phân cụm không giám sát bằng Self-Organizing Maps (SOM 1x2 và 2x2), đánh giá dự đoán nhãn cụm và so sánh từng cặp.
    - Fig 3A: Phân bố Prediction Strength (PS) của mô hình Weighted Voting.
    - F04 (hình bổ sung của nhóm): Phân tích lỗi Logistic Regression trên tập kiểm tra độc lập theo loại mô và nguồn thu thập.

Tuân thủ nghiêm ngặt Style Guide (docs/style_guide.md) và ADR-014:
    - Bảng màu Okabe-Ito: ALL (#0072B2), AML (#E69F00), an toàn cho người mù màu.
    - Chế độ xuất: report (300 dpi, chữ >= 10 pt) và slide (150 dpi, chữ >= 16 pt).
    - Không chỉnh sửa src/viz.py của TV6.

Cách dùng:
    from src.viz_redesign import compute_fig4, plot_fig4_som_redesign, plot_fig3a_prediction_strength, plot_fig4_error_analysis
    fig_som = plot_fig4_som_redesign(compute_fig4(X, samples), mode="report")
    fig3a = plot_fig3a_prediction_strength(ps_df, mode="report")
    fig4 = plot_fig4_error_analysis(preds, samples, mode="report")

Chạy trực tiếp:
    python -m src.viz_redesign
"""
from __future__ import annotations


from dataclasses import dataclass
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from minisom import MiniSom

try:
    from src.config import (
        PROJECT_ROOT, PS_THRESHOLD, RANDOM_SEED
    )
    from src.evaluate import evaluate, _baseline_pipeline
    from src.golub_wv import GolubWeightedVoting, loocv
    from src.load import load_golub
    from src.preprocess import GeneFilter
    from src.viz import (
        CLASS_COLORS, SUBTYPE_COLORS,
        SOURCE_COLORS, TISSUE_COLORS, NEUTRAL, NEUTRAL_LIGHT,
        style_rc, figsize, save_figure
    )
except ModuleNotFoundError:
    from config import (
        PROJECT_ROOT, PS_THRESHOLD, RANDOM_SEED
    )
    from evaluate import evaluate, _baseline_pipeline
    from golub_wv import GolubWeightedVoting, loocv
    from load import load_golub
    from preprocess import GeneFilter
    from viz import (
        CLASS_COLORS, SUBTYPE_COLORS,
        SOURCE_COLORS, TISSUE_COLORS, NEUTRAL, NEUTRAL_LIGHT,
        style_rc, figsize, save_figure
    )


# --------------------------------------------------------------------------------------
# 1. Huấn luyện SOM hỗ trợ Fig 4
# --------------------------------------------------------------------------------------

def fit_som_labels(X_mat: np.ndarray, shape: tuple[int, int], seed: int = RANDOM_SEED, n_iter: int = 1500) -> np.ndarray:
    """Huấn luyện MiniSom trên ma trận biểu hiện gen và trả về nhãn nơ-ron (0..k-1) của từng mẫu."""
    som = MiniSom(shape[0], shape[1], X_mat.shape[1], sigma=0.5, learning_rate=0.5, random_seed=seed)
    som.random_weights_init(X_mat)
    som.train_random(X_mat, n_iter)
    return np.array([np.ravel_multi_index(som.winner(x), shape) for x in X_mat])


def plot_fig2_neighborhood(
    X: pd.DataFrame, samples: pd.DataFrame, mode: str = "report",
    seed: int = RANDOM_SEED, n_permutations: int = 400,
) -> plt.Figure:
    """Đếm probe vượt ngưỡng SNR ở 38 mẫu train, theo từng phía ALL/AML."""
    rc = style_rc(mode)
    train = samples.loc[X.index, "split"].eq("train")
    is_all = samples.loc[X.index[train], "class"].eq("ALL").to_numpy()
    values = X.loc[train].to_numpy()
    if min(is_all.sum(), (~is_all).sum()) < 2 or n_permutations < 1:
        raise ValueError("Fig. 2 cần ít nhất hai mẫu mỗi lớp và một lượt hoán vị.")
    thresholds = np.linspace(0, 1.8, 91)

    def counts(mask):
        a, b = values[mask], values[~mask]
        spread = a.std(axis=0, ddof=1) + b.std(axis=0, ddof=1)
        delta = a.mean(axis=0) - b.mean(axis=0)
        snr = np.divide(delta, spread, out=np.zeros_like(delta), where=spread > 0)
        return np.stack([
            len(snr) - np.searchsorted(np.sort(scores), thresholds, side="right")
            for scores in (snr, -snr)
        ])

    observed = counts(is_all)
    rng = np.random.default_rng(seed)
    controls = np.stack([counts(rng.permutation(is_all)) for _ in range(n_permutations)])
    band_95, band_99 = np.percentile(controls, [95, 99], axis=0)
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=figsize(mode, 0.63), sharey=True, layout="constrained")
        for i, (ax, label) in enumerate(zip(axes, ("ALL", "AML"))):
            ax.plot(thresholds, observed[i], color=CLASS_COLORS[label], label="Nhãn thật")
            ax.plot(thresholds, band_95[i], "--", color=NEUTRAL, label="Phân vị 95%")
            ax.plot(thresholds, band_99[i], ":", color="black", label="Phân vị 99%")
            ax.set(title=f"Cao hơn ở {label}", xlabel="Ngưỡng |SNR|", ylim=(0, None))
        axes[0].set_ylabel("Số probe vượt ngưỡng")
        axes[1].legend(frameon=False)
        fig.suptitle(f"Fig. 2 · Phân tích lân cận ALL/AML\n{len(values)} mẫu train · {n_permutations} hoán vị nhãn")
        return fig


@dataclass
class SomPanels:
    probe_count: int
    seed: int
    two_clusters: pd.DataFrame
    four_clusters: pd.DataFrame
    ps_cv: np.ndarray
    ps_test: np.ndarray
    null_medians: np.ndarray
    pairwise: dict[str, pd.DataFrame]


def compute_fig4(
    X: pd.DataFrame, samples: pd.DataFrame, seed: int = RANDOM_SEED, n_permutations: int = 100,
) -> SomPanels:
    """Fit và lọc SOM trên train. LOOCV kiểm tra khả năng dự đoán nhãn cụm cố định."""
    samples = samples.loc[X.index]
    train = samples["split"].eq("train")
    X_train, X_test = X.loc[train], X.loc[samples["split"].eq("test")]

    # Ghi chú 33 của bài gốc dùng ngưỡng biến thiên năm lần.
    X_som = GeneFilter(ratio=5, diff=0).fit_transform(X_train)
    X_values = X_som.to_numpy()
    two = fit_som_labels(X_values, (1, 2), seed=seed, n_iter=1000)
    four = fit_som_labels(X_values, (2, 2), seed=seed, n_iter=1500)
    two_counts = pd.crosstab(two, samples.loc[train, "class"]).reindex(
        index=range(2), columns=["ALL", "AML"], fill_value=0,
    ).sort_values("ALL", ascending=False)
    four_counts = pd.crosstab(four, samples.loc[train, "subtype"]).reindex(
        index=range(4), columns=["B-ALL", "T-ALL", "AML"], fill_value=0,
    )
    if min(two_counts.sum(axis=1)) < 3 or n_permutations < 1:
        raise ValueError("Fig. 4B cần hai cụm có ít nhất ba mẫu và một lượt hoán vị.")

    # API Weighted Voting dùng hai giá trị ALL/AML làm mã nội bộ cho nhãn cụm.
    labels = np.where(two == two_counts.index[0], "ALL", "AML")
    n_genes = 20

    def scores(y):
        ps_cv = loocv(X_train, y, n_genes=n_genes)["ps"].to_numpy()
        model = GolubWeightedVoting(n_genes=n_genes).fit(X_train, y)
        return ps_cv, model.prediction_strength(X_test)

    ps_cv, ps_test = scores(labels)
    rng = np.random.default_rng(seed)
    null_medians = np.array([
        [np.median(ps) for ps in scores(rng.permutation(labels))]
        for _ in range(n_permutations)
    ])

    pairwise = {}
    for left, right in combinations(range(4), 2):
        pair = f"B{left + 1}–B{right + 1}"
        if min(four_counts.loc[[left, right]].sum(axis=1)) < 3:
            # Với ít hơn ba mẫu, một lượt LOOCV không ước lượng được SD của lớp.
            pairwise[pair] = pd.DataFrame(columns=["true", "pred", "ps"])
            continue
        keep = np.isin(four, [left, right])
        y = np.where(four[keep] == left, "ALL", "AML")
        pairwise[pair] = loocv(X_train.loc[keep], y, n_genes=n_genes)

    return SomPanels(X_som.shape[1], seed, two_counts, four_counts, ps_cv, ps_test, null_medians, pairwise)


def plot_fig4_som_redesign(result: SomPanels, mode: str = "report") -> plt.Figure:
    """Vẽ bốn panel khám phá lớp của Fig. 4."""
    with plt.rc_context(style_rc(mode)):
        fig, ((ax_a, ax_b), (ax_c, ax_d)) = plt.subplots(
            2, 2, figsize=figsize(mode, 1.02 if mode == "report" else 0.7), layout="constrained",
        )
        fig.suptitle(f"Fig. 4 · Khám phá lớp bằng SOM\n{result.two_clusters.to_numpy().sum()} mẫu train · {result.probe_count:,} probe · seed {result.seed}")

        def composition(ax, counts, positions, colors):
            bottom = np.zeros(len(counts))
            for label, hatch in zip(counts.columns, ("", "//", "..")):
                ax.bar(positions, counts[label], bottom=bottom, color=colors[label],
                       label=label, width=0.62, hatch=hatch, edgecolor="white")
                bottom += counts[label].to_numpy()
            ax.set_ylim(0, 1.5 * bottom.max())
            ax.set_ylabel("Số mẫu")
            ax.legend(frameon=False)

        paper = pd.DataFrame({"ALL": [24, 3], "AML": [1, 10]})
        composition(ax_a, pd.concat([paper, result.two_clusters]), [0, 1, 2.5, 3.5], CLASS_COLORS)
        ax_a.set(xticks=[0, 1, 2.5, 3.5], xticklabels=["A1", "A2", "A1", "A2"], xlabel="Bài báo     |     Nhóm")
        ax_a.set_title("(A) SOM hai cụm")
        composition(ax_c, result.four_clusters, range(4), SUBTYPE_COLORS)
        ax_c.set(xticks=range(4), xticklabels=["B1", "B2", "B3", "B4"], title="(C) SOM bốn cụm")

        groups = [result.ps_cv, result.ps_test, result.null_medians[:, 0], result.null_medians[:, 1]]
        ax_b.boxplot(groups, widths=0.53, medianprops={"color": CLASS_COLORS["ALL"]})
        ax_b.set(xticks=range(1, 5), xticklabels=["SOM\nLOOCV", "SOM\nTest", "Xáo nhãn\nLOOCV", "Xáo nhãn\nTest"])
        ax_b.set_title("(B) PS của mô hình dự đoán cụm")
        ax_b.set_xlabel(f"Xáo nhãn: trung vị từ {len(result.null_medians)} lượt")

        rng = np.random.default_rng(result.seed)
        for pos, scores in enumerate(result.pairwise.values()):
            if scores.empty:
                ax_d.text(pos, 0.52, "Thiếu mẫu", rotation=90, ha="center", va="center")
                continue
            x = pos + rng.uniform(-0.15, 0.15, size=len(scores))
            color = np.select(
                [scores["pred"].eq("uncertain"), scores["pred"].eq(scores["true"])],
                ["#777777", CLASS_COLORS["ALL"]], default="#D55E00",
            )
            ax_d.scatter(x, scores["ps"], c=color, s=13, alpha=0.8)
            ax_d.hlines(scores["ps"].median(), pos - 0.21, pos + 0.21, color="black")
        ax_d.set(xticks=range(len(result.pairwise)), xticklabels=list(result.pairwise),
                 title="(D) PS theo từng cặp cụm (LOOCV)", xlim=(-0.45, len(result.pairwise) - 0.55))
        ax_d.tick_params(axis="x", labelrotation=32)
        legend = [
            Line2D([], [], marker="o", linestyle="none", color=color, label=label)
            for color, label in ((CLASS_COLORS["ALL"], "Đúng"), ("#D55E00", "Sai"), ("#777777", "Bất định"))
        ]
        ax_d.legend(handles=legend, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.23), frameon=False)
        for ax in (ax_b, ax_d):
            ax.axhline(PS_THRESHOLD, color=NEUTRAL, linestyle="--", alpha=0.8)
            ax.set(ylim=(0, 1.02), ylabel="Prediction Strength")
        return fig


# --------------------------------------------------------------------------------------
# 3. F04: Phân tích lỗi Logistic Regression trên tập kiểm tra độc lập
# --------------------------------------------------------------------------------------

def plot_fig4_error_analysis(
    preds_df: pd.DataFrame,
    samples: pd.DataFrame,
    mode: str = "report"
) -> plt.Figure:
    """Trực quan hóa kết quả Logistic Regression trên tập kiểm tra độc lập.

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
        wrong = merged["y_pred"].ne(merged["y_true"])
        n_pb = merged["tissue"].eq("PB").sum()

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

        # Ngưỡng phân loại của mô hình
        ax_main.axvline(0.5, color=NEUTRAL, linestyle="--", linewidth=1.2, label="Ngưỡng phân loại (0.5)")
        ax_main.axvspan(0.3, 0.7, color=NEUTRAL_LIGHT, alpha=0.25, label="Dải điểm AML 0,3–0,7")

        # Ghi chú ca đặc biệt Mẫu 66
        if 66 in merged.index and wrong.loc[66]:
            m66_idx = np.where(merged.index == 66)[0][0]
            ax_main.annotate(
                f"Mẫu 66: AML → ALL\nĐiểm AML = {merged.loc[66, 'score']:.3f}",
                xy=(merged.loc[66, "score"], m66_idx),
                xytext=(merged.loc[66, "score"] + 0.22, m66_idx - 1.5),
                arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.5),
                color="#D55E00", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF2E6", edgecolor="#D55E00", lw=1)
            )

        ax_main.set_yticks(y_positions)
        yticklabels = [f"#{idx} ({row['y_true']})" for idx, row in merged.iterrows()]
        ax_main.set_yticklabels(yticklabels, fontsize=style_rc(mode)["font.size"] - 1.5)
        ax_main.set_xlabel("Xác suất lớp AML do Logistic Regression ước tính")
        ax_main.set_xlim(-0.02, 1.05)
        ax_main.set_title(f"(A) Tập kiểm tra: {wrong.sum()}/{n_samples} mẫu phân loại sai", loc="left")
        ax_main.legend(
            handles=[
                *ax_main.get_legend_handles_labels()[0],
                Patch(facecolor=CLASS_COLORS["ALL"], label="ALL đúng"),
                Patch(facecolor=CLASS_COLORS["AML"], label="AML đúng"),
                Patch(facecolor="#D55E00", label="Sai phân loại"),
            ],
            loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, frameon=False,
        )

        # ---------------------------------------------------------
        # Bảng metadata: Mô (Tissue) và Nguồn (Source)
        # ---------------------------------------------------------
        # Vẽ 2 cột nhiệt metadata
        ax_meta.scatter(np.zeros(n_samples), y_positions, c=[TISSUE_COLORS[r["tissue"]] for _, r in merged.iterrows()],
                        s=style_rc(mode)["lines.linewidth"] * 30, marker="s", label="Mô")
        ax_meta.scatter(np.ones(n_samples), y_positions, c=[SOURCE_COLORS[r["source"]] for _, r in merged.iterrows()],
                        s=style_rc(mode)["lines.linewidth"] * 30, marker="o", label="Nguồn")

        ax_meta.set_xticks([0, 1])
        ax_meta.set_xticklabels(["Mô", "Nguồn"], fontsize=style_rc(mode)["font.size"] - 1)
        ax_meta.set_xlim(-0.5, 1.5)
        ax_meta.set_title("(B) Quản trị dữ liệu", loc="center")
        ax_meta.spines["left"].set_visible(False)
        ax_meta.tick_params(left=False)

        # Chú giải riêng cho metadata
        legend_elements = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor=TISSUE_COLORS["BM"], label="BM (Tủy xương)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=TISSUE_COLORS["PB"], label=f"PB (Máu ngoại vi, n={n_pb})"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SOURCE_COLORS["DFCI"], label="DFCI"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SOURCE_COLORS["CALGB"], label="CALGB"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SOURCE_COLORS["CCG"], label="CCG"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=SOURCE_COLORS["St-Jude"], label="St-Jude"),
        ]
        ax_meta.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(-0.10, -0.16),
                       fontsize=style_rc(mode)["font.size"] - 2, frameon=False, ncol=2)

        fig.suptitle("F04. Phân tích lỗi Logistic Regression trên tập kiểm tra độc lập",
                     y=1.02, fontweight="bold")
        return fig


# --------------------------------------------------------------------------------------
# 4. Tái thiết kế Fig 3A: Prediction Strength (PS) của mô hình Weighted Voting
# --------------------------------------------------------------------------------------

def plot_fig3a_prediction_strength(ps_df: pd.DataFrame | None = None, mode: str = "report") -> plt.Figure:
    """Tái thiết kế Fig 3A của Golub et al. (1999): Prediction Strength (PS) phân bổ theo lớp và tập mẫu.

    Hai panel trình bày PS trên LOOCV và tập kiểm tra độc lập.
    """
    rc = style_rc(mode)
    if ps_df is None:
        ps_df = pd.read_csv(PROJECT_ROOT / "results" / "metrics" / "golub_ps.csv")
    with plt.rc_context(rc):
        w, h = (12.5, 6.2) if mode == "report" else (14.5, 7.5)
        fig, (ax_loocv, ax_test) = plt.subplots(1, 2, figsize=(w, h), sharey=True)

        # Hàm vẽ cho từng panel
        def _render_panel(ax, subset_name: str, title_panel: str):
            sub = ps_df.loc[ps_df["set"].eq(subset_name)]

            # Vùng bóng Uncertain (PS < 0.30)
            ax.axhspan(-0.02, PS_THRESHOLD, facecolor="#FADBD8" if mode == "report" else "#F2D7D5", alpha=0.35, zorder=0)
            ax.axhline(PS_THRESHOLD, color="#C0392B", linestyle="--", linewidth=rc["lines.linewidth"] * 1.2,
                       label=f"Ngưỡng PS = {PS_THRESHOLD:.2f}", zorder=2)

            classes = ["ALL", "AML"]
            positions = [0, 1]
            box_data = [sub[sub["true"] == c]["ps"].values for c in classes]

            # Boxplot mờ
            ax.boxplot(box_data, positions=positions, widths=0.35, patch_artist=True,
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
                        if sample_id == 66 and pred not in (c, "uncertain"):
                            ax.annotate(f"Mẫu {sample_id}: {c} → {pred}\nPS = {ps_val:.5f}",
                                        xy=(x_pos, ps_val), xytext=(0.12, ps_val + 0.14),
                                        fontsize=rc["font.size"] - 1, fontweight="bold", color="#C0392B",
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", edgecolor="#E74C3C"),
                                        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5))
                        elif sample_id == 67 and pred == "uncertain":
                            ax.annotate(f"Mẫu {sample_id} ({tissue})\nPS = {ps_val:.2f} (uncertain)",
                                        xy=(x_pos, ps_val), xytext=(x_pos + 0.16, ps_val - 0.05),
                                        fontsize=rc["font.size"] - 2, fontweight="bold", color="#78281F",
                                        arrowprops=dict(arrowstyle="->", color="#78281F", lw=1.0))

            ax.set_xticks(positions)
            ax.set_xticklabels([f"ALL (n={len(sub[sub['true']=='ALL'])})",
                                f"AML (n={len(sub[sub['true']=='AML'])})"],
                               fontsize=rc["font.size"])
            ax.set_xlim(-0.4, 1.4)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{title_panel} ({len(sub)} mẫu)", loc="left", fontweight="bold", fontsize=rc["font.size"] + 1)
            ax.grid(axis="y", linestyle=":", alpha=0.5, color="#BDC3C7")

            confident = sub["pred"].ne("uncertain")
            correct = confident & sub["pred"].eq(sub["true"])
            n_conf = confident.sum()
            n_correct = correct.sum()
            coverage = f"{n_conf / len(sub):.1%}" if len(sub) else "—"
            accuracy = f"{n_correct / n_conf:.1%}" if n_conf else "—"
            summary = [
                f"Chắc chắn: {n_conf}/{len(sub)} ({coverage})",
                f"Đúng trong số chắc chắn: {n_correct}/{n_conf} ({accuracy})",
                f"Trung vị PS: {sub['ps'].median():.2f}" if len(sub) else "Trung vị PS: —",
                f"Sai: {n_conf - n_correct} · Không chắc chắn: {(~confident).sum()}",
            ]
            if subset_name == "test" and "tissue" in sub:
                pb = sub["tissue"].eq("PB")
                summary.append(f"PB: {(pb & correct).sum()}/{pb.sum()} đúng, {(pb & ~confident).sum()} không chắc chắn")
            ax.text(
                0.01, -0.15, "\n".join(summary), transform=ax.transAxes, va="top",
                fontsize=rc["font.size"] - 2,
            )

        _render_panel(ax_loocv, "loocv", "(A) Kiểm tra chéo LOOCV")
        _render_panel(ax_test, "test", "(B) Tập kiểm tra độc lập")
        ax_loocv.set_ylabel("Prediction Strength (PS)", fontsize=rc["font.size"] + 1)

        # Chú giải chung
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS["ALL"], markersize=8, label="Lớp ALL (BM)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS["AML"], markersize=8, label="Lớp AML (BM)"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=NEUTRAL, markersize=8, label="Máu ngoại vi (PB)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#95A5A6", markeredgecolor="#C0392B", markeredgewidth=1.5, markersize=8, label="Không chắc chắn"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C", markeredgecolor="black", markeredgewidth=1.5, markersize=10, label="Sai phân loại"),
            Patch(facecolor="#FADBD8", edgecolor="#C0392B", label=f"Vùng PS < {PS_THRESHOLD:.2f}"),
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
    """Sinh Fig. 2, Fig. 3A, Fig. 4 và F04 bổ sung ở cả hai định dạng."""
    print("Đang tính các panel SOM và sinh hình TV5...")
    X_cu, samples = load_golub("curated")
    som = compute_fig4(X_cu, samples)

    # 1. Chạy đánh giá để lấy bảng dự đoán 34 mẫu kiểm tra cho Fig 4
    pipe = _baseline_pipeline(k=50)
    res = evaluate(pipe, X_cu, samples["class"], scheme="original_split")
    preds = res["predictions"]

    for mode in ("report", "slide"):
        fig2 = plot_fig2_neighborhood(X_cu, samples, mode=mode)
        print(f"[{mode.upper()}] Đã lưu: {save_figure(fig2, 'F02_neighborhood_redesign', mode)}")
        plt.close(fig2)

        fig_som = plot_fig4_som_redesign(som, mode=mode)
        path_som = save_figure(fig_som, "F02_som_redesign", mode=mode)
        plt.close(fig_som)
        print(f"[{mode.upper()}] Đã lưu: {path_som}")

        # Sinh Fig 3A
        fig3a = plot_fig3a_prediction_strength(ps_df=None, mode=mode)
        path3a = save_figure(fig3a, "F03a_prediction_strength", mode=mode)
        plt.close(fig3a)
        print(f"[{mode.upper()}] Đã lưu: {path3a}")

        # Sinh hình bổ sung F04
        fig4 = plot_fig4_error_analysis(preds, samples, mode=mode)
        path4 = save_figure(fig4, "F04_error_analysis", mode=mode)
        plt.close(fig4)
        print(f"[{mode.upper()}] Đã lưu: {path4}")




if __name__ == "__main__":
    generate_all_redesigned_figures()
