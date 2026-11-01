"""Các hàm vẽ theo style guide (docs/style_guide.md); chạy `python -m src.viz` để vẽ lại các hình EDA.

Công việc: 1.4 / 2.6 — Phụ trách: TV6 (kế hoạch mục 4.1, 4.2, 5.2).

Hình do module này vẽ (bản 1, công việc 1.4):
  F02  histogram trước/sau log10          F04a scree plot
  F02b phân bố độ biến thiên của gen      F04b PCA PC1–PC2 theo lớp và nhóm con
  F03  boxplot theo từng mẫu              F04c so sánh PCA / t-SNE / UMAP
                                          F04e tương quan giữa các mẫu + phân cụm phân cấp
Hình do người khác vẽ (sẽ chuyển sang module này ở công việc 2.6, giữ nguyên tên file):
  F01, F04d (TV1, notebooks/02_design_batch.ipynb) · F10, F11 (TV2, src/evaluate.py)
  F05–F09, F12 chưa có (chờ kết quả giai đoạn 2).

Quy ước dùng chung:
  - Mọi hàm `plot_*` / `pca_scatter` nhận `mode="report"` (chữ 10 pt, 300 dpi) hoặc `mode="slide"`
    (chữ 16 pt, 150 dpi) và trả về `matplotlib.figure.Figure`.
  - Màu, hình dạng điểm, cỡ chữ lấy từ các hằng số bên dưới — KHÔNG ghi cứng màu trong notebook.
  - Hàm vẽ chỉ nhận DataFrame, không tự đọc dữ liệu; việc đọc là của `load_golub()` (xem `main`).
"""
from __future__ import annotations

import functools
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from src.config import (CURATED_DIR, METRICS_DIR, PALETTE, RANDOM_SEED, REPORT_FIG_DIR,
                            SLIDE_FIG_DIR)
    from src.load import load_golub
    from src.preprocess import GeneFilter
except ModuleNotFoundError:  # chạy trực tiếp `python src/viz.py`
    from config import (CURATED_DIR, METRICS_DIR, PALETTE, RANDOM_SEED, REPORT_FIG_DIR,
                        SLIDE_FIG_DIR)
    from load import load_golub
    from preprocess import GeneFilter

# --------------------------------------------------------------------------------------
# 1. Hằng số kiểu dáng (nguồn duy nhất; khớp docs/style_guide.md)
# --------------------------------------------------------------------------------------
# Bảng màu Okabe–Ito (an toàn với người mù màu). ALL/AML lấy từ src.config.PALETTE.
CLASS_COLORS = dict(PALETTE)                                   # ALL xanh dương, AML cam
CLASS_MARKERS = {"ALL": "o", "AML": "^"}                       # hình dạng đi kèm màu (mã hóa dư thừa)
SUBTYPE_COLORS = {"B-ALL": "#0072B2", "T-ALL": "#332288", "AML": "#E69F00"}
SUBTYPE_MARKERS = {"B-ALL": "o", "T-ALL": "s", "AML": "^"}
# Khớp với hình F4d của TV1: một nguồn mẫu luôn có một màu trong mọi hình.
SOURCE_COLORS = {"DFCI": "#009E73", "CALGB": "#CC79A7", "CCG": "#56B4E9", "St-Jude": "#D55E00"}
TISSUE_COLORS = {"BM": "#7F7F7F", "PB": "#D55E00"}
NEUTRAL = "#4D4D4D"          # đối tượng không mang nghĩa lớp (cột scree, đường tham chiếu…)
NEUTRAL_LIGHT = "#BDBDBD"
DIVERGING_CMAP = "RdBu_r"    # heatmap z-score: xanh–trắng–đỏ, tâm 0; KHÔNG dùng rainbow/jet
SEQUENTIAL_CMAP = "cividis"  # thang tuần tự an toàn với người mù màu (ma trận tương quan)

COLOR_MAPS = {"class": CLASS_COLORS, "subtype": SUBTYPE_COLORS, "source": SOURCE_COLORS,
              "tissue": TISSUE_COLORS}
MARKER_MAPS = {"class": CLASS_MARKERS, "subtype": SUBTYPE_MARKERS}
LABELS = {"class": "Lớp", "subtype": "Nhóm con", "source": "Nguồn mẫu", "tissue": "Loại mô"}
_EXTRA_MARKERS = ["o", "^", "s", "D", "v", "P"]

FIG_STYLES = {
    # fontsize: cỡ chữ tối thiểu của MỌI chữ trong hình; width tính theo inch ở kích thước in thật
    "report": dict(fontsize=10, dpi=300, width=6.5, marker=38, lw=1.4, subdir="report"),
    "slide": dict(fontsize=16, dpi=150, width=12.0, marker=95, lw=2.4, subdir="slides"),
}
MODES = tuple(FIG_STYLES)


def style_rc(mode: str = "report") -> dict:
    """rcParams của một chế độ; dùng qua `with plt.rc_context(style_rc(mode))`."""
    if mode not in FIG_STYLES:
        raise ValueError(f"mode phải là một trong {MODES}, nhận {mode!r}")
    s = FIG_STYLES[mode]
    fs = s["fontsize"]
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],  # có đủ dấu tiếng Việt
        "font.size": fs,
        "axes.titlesize": fs + 1,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "legend.title_fontsize": fs,
        "figure.titlesize": fs + 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "lines.linewidth": s["lw"],
        "figure.dpi": 100,
        "savefig.dpi": s["dpi"],
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    }


def styled(fn):
    """Bọc hàm vẽ trong rc_context(style_rc(mode)) để mọi chữ/đường theo đúng chế độ."""
    @functools.wraps(fn)
    def wrapper(*args, mode: str = "report", **kwargs):
        with plt.rc_context(style_rc(mode)):
            return fn(*args, mode=mode, **kwargs)
    return wrapper


def figsize(mode: str, height_ratio: float = 0.62, width_ratio: float = 1.0) -> tuple[float, float]:
    """Kích thước (inch). Báo cáo rộng 6.5 in = bề rộng văn bản A4, nên chữ 10 pt là 10 pt khi in
    (vẽ hình to rồi thu nhỏ sẽ làm chữ nhỏ hơn 10 pt)."""
    w = FIG_STYLES[mode]["width"] * width_ratio
    return (w, w * height_ratio)


def save_figure(fig: plt.Figure, stem: str, mode: str, base_dir: Path | None = None) -> Path:
    """Lưu `<stem>_<mode>.png` vào results/figures/report|slides/ (hoặc base_dir/report|slides)."""
    if base_dir is None:
        out = REPORT_FIG_DIR if mode == "report" else SLIDE_FIG_DIR
    else:
        out = Path(base_dir) / FIG_STYLES[mode]["subdir"]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}_{mode}.png"
    fig.savefig(path, dpi=FIG_STYLES[mode]["dpi"], bbox_inches="tight", facecolor="white")
    return path


# --------------------------------------------------------------------------------------
# 2. Tính toán dùng chung cho hình và cho bảng tóm tắt
# --------------------------------------------------------------------------------------
def zscore(X: pd.DataFrame) -> np.ndarray:
    """Chuẩn hóa mỗi gen về trung bình 0, độ lệch chuẩn 1 (chỉ cho EDA; mô hình dùng Pipeline)."""
    return StandardScaler().fit_transform(X.to_numpy(dtype=float))


def compute_pca(X: pd.DataFrame, n_components: int = 10, seed: int = RANDOM_SEED):
    """PCA trên z-score từng gen. Trả (scores: DataFrame PC1.. cùng index với X, ratio: ndarray)."""
    k = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=k, random_state=seed)
    scores = pd.DataFrame(pca.fit_transform(zscore(X)), index=X.index,
                          columns=[f"PC{i + 1}" for i in range(k)])
    return scores, pca.explained_variance_ratio_


def compute_embeddings(X: pd.DataFrame, seed: int = RANDOM_SEED,
                       methods: tuple[str, ...] = ("pca", "tsne", "umap")) -> dict[str, pd.DataFrame]:
    """Tọa độ 2D của PCA, t-SNE, UMAP (cùng z-score đầu vào, cùng seed). Thiếu umap-learn thì bỏ UMAP."""
    Z = zscore(X)
    n = Z.shape[0]
    out: dict[str, pd.DataFrame] = {}
    cols = ["dim1", "dim2"]
    if "pca" in methods:
        out["PCA"] = pd.DataFrame(PCA(2, random_state=seed).fit_transform(Z), index=X.index, columns=cols)
    if "tsne" in methods:
        from sklearn.manifold import TSNE
        perp = float(min(15, max(2, (n - 1) // 3)))
        out["t-SNE"] = pd.DataFrame(
            TSNE(2, perplexity=perp, init="pca", learning_rate="auto", random_state=seed).fit_transform(Z),
            index=X.index, columns=cols)
    if "umap" in methods:
        try:
            import umap
        except ImportError:  # pragma: no cover
            warnings.warn("Không có umap-learn: bỏ panel UMAP (pip install umap-learn).")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                emb = umap.UMAP(n_neighbors=min(10, n - 1), min_dist=0.3, random_state=seed).fit_transform(Z)
            out["UMAP"] = pd.DataFrame(emb, index=X.index, columns=cols)
    return out


def skewness_summary(X_raw: pd.DataFrame, X_log: pd.DataFrame) -> dict:
    """Độ lệch (skewness) và kurtosis dư trước/sau log10. X_raw là giá trị đã threshold [100, 16000].
    Trả trung vị theo gen (mỗi gen có 72 giá trị) và giá trị gộp mọi ô."""
    def per_gene(M, f):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmedian(f(M.to_numpy(float), axis=0)))
    return {
        "skew_median_gene_before": per_gene(X_raw, stats.skew),
        "skew_median_gene_after": per_gene(X_log, stats.skew),
        "kurtosis_median_gene_before": per_gene(X_raw, stats.kurtosis),
        "kurtosis_median_gene_after": per_gene(X_log, stats.kurtosis),
        "skew_pooled_before": float(stats.skew(X_raw.to_numpy(float).ravel())),
        "skew_pooled_after": float(stats.skew(X_log.to_numpy(float).ravel())),
    }


def pc_class_separation(scores: pd.DataFrame, samples: pd.DataFrame, n_pcs: int = 5) -> pd.DataFrame:
    """Mỗi PC tách ALL/AML tốt đến đâu: AUC (Mann–Whitney, lấy max(AUC, 1−AUC)) và p. Chỉ để mô tả."""
    y = (samples.loc[scores.index, "class"] == "AML").to_numpy()
    rows = []
    for pc in scores.columns[:n_pcs]:
        v = scores[pc].to_numpy()
        u, p = stats.mannwhitneyu(v[y], v[~y], alternative="two-sided")
        auc = u / (y.sum() * (~y).sum())
        rows.append({"PC": pc, "auc": float(max(auc, 1 - auc)), "p_mannwhitney": float(p)})
    return pd.DataFrame(rows)


def sample_median_outliers(X_log: pd.DataFrame, thresh: float = 3.5) -> pd.DataFrame:
    """Đối chiếu với cờ qc_outlier: modified z-score (median/MAD) của trung vị biểu hiện từng mẫu."""
    med = X_log.median(axis=1)
    mad = float((med - med.median()).abs().median()) or np.nan
    z = 0.6745 * (med - med.median()) / mad
    return pd.DataFrame({"median_log10": med, "modified_z": z, "flag": z.abs() > thresh})


def eda_summary(X_raw: pd.DataFrame, X_log: pd.DataFrame, X_filt: pd.DataFrame, samples: pd.DataFrame,
                seed: int = RANDOM_SEED) -> dict:
    """Mọi con số cho 'Nhận xét bắt buộc' của 1.4 và mục 3.1 báo cáo (ghi ra eda_summary.json)."""
    Xt = X_raw.clip(100, 16000)
    keep = GeneFilter().fit(X_log).mask_
    scores, ratio_full = compute_pca(X_filt, X_filt.shape[0], seed)   # đủ mọi PC để tính số PC tích lũy
    ratio, cum = ratio_full[:10], np.cumsum(ratio_full)
    sep = pc_class_separation(scores, samples)
    lab = lambda col, pcs: float(silhouette_score(scores[list(pcs)].to_numpy(), samples.loc[scores.index, col]))
    sd = X_log.std(axis=0, ddof=1)
    med_out = sample_median_outliers(X_log)
    return {
        "n_samples": int(X_log.shape[0]), "n_probes": int(X_log.shape[1]),
        "n_genes_filtered": int(X_filt.shape[1]),
        "class_counts": samples["class"].value_counts().to_dict(),
        "measurement_range": {"frac_below_100": float((X_raw.to_numpy(float) < 100).mean()),
                              "frac_negative": float((X_raw.to_numpy(float) < 0).mean()),
                              "frac_above_16000": float((X_raw.to_numpy(float) > 16000).mean()),
                              "min": float(X_raw.to_numpy(float).min()), "max": float(X_raw.to_numpy(float).max())},
        "skewness": skewness_summary(Xt, X_log),
        "gene_filter": {"n_kept": int(keep.sum()), "n_removed": int((~keep).sum()),
                        "sd_log10_median_kept": float(sd[keep].median()),
                        "sd_log10_median_removed": float(sd[~keep].median())},
        "pca": {"explained_ratio": [float(r) for r in ratio],
                "cumulative_pc1_pc2": float(ratio[:2].sum()),
                "cumulative_pc1_pc3": float(ratio[:3].sum()),
                "pcs_to_reach_50pct": int(np.searchsorted(cum, 0.5) + 1),
                "pcs_to_reach_80pct": int(np.searchsorted(cum, 0.8) + 1),
                "class_separation_by_pc": sep.assign(auc=sep["auc"].round(4)).to_dict(orient="records"),
                "silhouette_class_pc1_pc2": lab("class", ("PC1", "PC2")),
                "silhouette_subtype_pc1_pc2": lab("subtype", ("PC1", "PC2"))},
        "qc_outlier_samples": [int(i) for i in samples.index[samples.get("qc_outlier", False).astype(bool)]]
        if "qc_outlier" in samples else [],
        "median_based_outlier_samples": [int(i) for i in med_out.index[med_out["flag"]]],
    }


# --------------------------------------------------------------------------------------
# 3. Thành phần vẽ dùng lại
# --------------------------------------------------------------------------------------
def _marker_for(col: str, values: list) -> dict:
    m = MARKER_MAPS.get(col)
    if m and all(v in m for v in values):
        return m
    return {v: _EXTRA_MARKERS[i % len(_EXTRA_MARKERS)] for i, v in enumerate(values)}


def _color_for(col: str, values: list) -> dict:
    c = COLOR_MAPS.get(col)
    if c and all(v in c for v in values):
        return c
    cycle = list(SOURCE_COLORS.values()) + list(CLASS_COLORS.values())
    return {v: cycle[i % len(cycle)] for i, v in enumerate(values)}


def legend_handles(samples: pd.DataFrame, color: str, shape: str | None, mode: str) -> list:
    """Chú giải: một nhóm khi màu và hình cùng theo một cột, hai nhóm khi khác cột."""
    ms = 6 if mode == "report" else 11
    cv = [v for v in COLOR_MAPS.get(color, {}) if v in set(samples[color])] or sorted(set(samples[color]))
    cmap = _color_for(color, cv)
    if shape is None or shape == color:
        mk = _marker_for(color, cv)
        return [Line2D([], [], ls="", marker=mk[v], color=cmap[v], mec="black", mew=0.5, ms=ms, label=str(v))
                for v in cv]
    sv = [v for v in MARKER_MAPS.get(shape, {}) if v in set(samples[shape])] or sorted(set(samples[shape]))
    smap = _marker_for(shape, sv)
    h = [Patch(facecolor=cmap[v], edgecolor="black", linewidth=0.5, label=str(v)) for v in cv]
    h += [Line2D([], [], ls="", marker=smap[v], mfc="white", mec="black", ms=ms, label=str(v)) for v in sv]
    return h


def _check_aligned(X: pd.DataFrame, samples: pd.DataFrame) -> None:
    """Cùng sample_id và cùng thứ tự: tránh gán nhầm nhãn/metadata cho mẫu (mục 2.1.2)."""
    if X.empty or samples.empty:
        raise ValueError("X và samples không được rỗng")
    if not X.index.equals(samples.index):
        raise ValueError("X và samples phải có cùng sample_id, cùng thứ tự")
    if not np.isfinite(X.to_numpy(float)).all():
        raise ValueError("X chứa giá trị không hữu hạn")


def _title(fig: plt.Figure, text: str, y: float = 1.02) -> None:
    fig.suptitle(text, y=y, x=0.5, ha="center")


# --------------------------------------------------------------------------------------
# 4. Các hình
# --------------------------------------------------------------------------------------
@styled
def pca_scatter(X: pd.DataFrame | None, samples: pd.DataFrame, color: str = "class",
                shape: str | None = "class", pcs: tuple[int, int] = (1, 2), mode: str = "report",
                scores: pd.DataFrame | None = None, ratio=None, ax: plt.Axes | None = None,
                legend: bool = True, label_outliers: bool = True):
    """PCA scatter PC_i–PC_j: màu theo cột `color`, hình theo cột `shape` (mã hóa dư thừa).

    Demo nghiệm thu: pca_scatter(Xf, samples, color='class', shape='class').
    Truyền `scores`/`ratio` (từ compute_pca) để khỏi tính lại; truyền `ax` để vẽ vào panel có sẵn
    (khi đó trả về ax, không trả Figure). Mẫu có cờ qc_outlier được khoanh tròn và ghi mã mẫu.
    """
    if scores is None:
        _check_aligned(X, samples)
        scores, ratio = compute_pca(X, max(pcs))
    elif not scores.index.isin(samples.index).all():
        raise ValueError("scores có sample_id không có trong samples")
    s = FIG_STYLES[mode]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=figsize(mode, 0.75 if mode == "report" else 0.6, 0.8))
    else:
        fig = ax.figure
    meta = samples.loc[scores.index]
    cvals = [v for v in COLOR_MAPS.get(color, {}) if v in set(meta[color])] or sorted(set(meta[color]))
    cmap = _color_for(color, cvals)
    if shape is None:
        mk_of = lambda row: "o"
    else:
        svals = [v for v in MARKER_MAPS.get(shape, {}) if v in set(meta[shape])] or sorted(set(meta[shape]))
        smap = _marker_for(shape, svals)
        mk_of = lambda row: smap[row]
    xi, yi = f"PC{pcs[0]}", f"PC{pcs[1]}"
    by = [color] + ([shape] if shape and shape != color else [])
    for key, grp in meta.groupby(by if len(by) > 1 else by[0], observed=True):
        idx = grp.index
        key = key if isinstance(key, tuple) else (key,)
        c_key = key[0]
        m_key = key[-1] if shape else None
        ax.scatter(scores.loc[idx, xi], scores.loc[idx, yi], c=cmap[c_key],
                   marker=mk_of(m_key) if shape else "o", s=s["marker"], edgecolors="black",
                   linewidths=0.5, alpha=0.85, zorder=3)
    if label_outliers and "qc_outlier" in meta:
        for sid in meta.index[meta["qc_outlier"].astype(bool)]:
            ax.scatter(scores.loc[sid, xi], scores.loc[sid, yi], s=s["marker"] * 4, facecolors="none",
                       edgecolors="black", linewidths=1.2, zorder=4)
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            right_half = scores.loc[sid, xi] > (x0 + x1) / 2
            top_part = scores.loc[sid, yi] > y0 + 0.8 * (y1 - y0)   # gần mép trên thì ghi nhãn bên dưới điểm
            ax.annotate(str(sid), (scores.loc[sid, xi], scores.loc[sid, yi]),
                        textcoords="offset points", xytext=(-9 if right_half else 9, -12 if top_part else 8),
                        ha="right" if right_half else "left", va="top" if top_part else "bottom",
                        fontsize=s["fontsize"], bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.8), zorder=6)
    pct = (lambda k: f" ({ratio[k - 1]:.1%})") if ratio is not None else (lambda k: "")
    ax.set_xlabel(f"{xi}{pct(pcs[0])}")
    ax.set_ylabel(f"{yi}{pct(pcs[1])}")
    ax.axhline(0, color=NEUTRAL_LIGHT, lw=0.8, zorder=1)
    ax.axvline(0, color=NEUTRAL_LIGHT, lw=0.8, zorder=1)
    if legend:
        handles = legend_handles(meta, color, shape, mode)
        if label_outliers and "qc_outlier" in meta and meta["qc_outlier"].astype(bool).any():
            handles.append(Line2D([], [], ls="", marker="o", mfc="none", mec="black", mew=1.2,
                                  ms=9 if mode == "report" else 14, label="qc_outlier"))
        # chú giải đặt DƯỚI trục để không che điểm dữ liệu
        ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3),
                  ncol=min(len(handles), 4), handletextpad=0.2, columnspacing=0.8, borderaxespad=0.0)
    return fig if own else ax


@styled
def plot_hist_log(X_raw: pd.DataFrame, X_log: pd.DataFrame, mode: str = "report"):
    """F02 — Vì sao cần threshold + log10? Trái: giá trị MAS4 gốc (có cả giá trị âm), trục y log, đường đứt
    tại 100 và 16000; phải: log10 sau khi cắt về [100, 16000], trục y log.

    X_raw là giá trị GỐC (chưa threshold). Trên dữ liệu Golub ~47% giá trị < 100 (gồm ~29% âm) nên cột ở mức
    sàn 100 rất cao; trục y log để vẫn thấy phần đuôi. Độ lệch (skewness) gộp mọi ô ghi trong hình.
    """
    v = X_raw.to_numpy(float).ravel()
    lg = X_log.to_numpy(float).ravel()
    summ = skewness_summary(X_raw.clip(100, 16000), X_log)
    below, above = float(np.mean(v < 100)), float(np.mean(v > 16000))
    lo, hi = np.floor(np.percentile(v, 0.5) / 1000) * 1000, np.ceil(np.percentile(v, 99.5) / 1000) * 1000
    fig, axes = plt.subplots(1, 2, figsize=figsize(mode, 0.42 if mode == "report" else 0.4))
    axes[0].hist(v[(v >= lo) & (v <= hi)], bins=70, color=NEUTRAL_LIGHT, edgecolor="white", linewidth=0.3, log=True)
    for x in (100, 16000):
        axes[0].axvline(x, color=NEUTRAL, ls="--", lw=1.2)
    axes[0].text(0.73, 0.95, f"{below:.1%} < 100\n{above:.1%} > 16000".replace(".", ","),
                 transform=axes[0].transAxes, ha="right", va="top")
    axes[0].set_title("MAS4 gốc")
    axes[0].set_xlabel(f"Giá trị biểu hiện (hiển thị {lo:,.0f} đến {hi:,.0f})".replace(",", " "))
    axes[0].set_ylabel("Số giá trị (trục log)")
    axes[1].hist(lg, bins=60, color=NEUTRAL_LIGHT, edgecolor="white", linewidth=0.3, log=True)
    axes[1].text(0.97, 0.95, f"độ lệch {summ['skew_pooled_after']:.2f}\n(sau threshold, trước log: {summ['skew_pooled_before']:.2f})".replace(".", ","),
                 transform=axes[1].transAxes, ha="right", va="top")
    axes[1].set_title("Sau threshold và log10")
    axes[1].set_xlabel("log10(giá trị cắt về [100, 16000])")
    _title(fig, "F2. Phân phối biểu hiện trước và sau tiền xử lý")
    fig.tight_layout()
    return fig


@styled
def plot_gene_variance(X_log: pd.DataFrame, mode: str = "report"):
    """F02b — Độ biến thiên của gen và bộ lọc Dudoit: SD log10 của 7129 gen, tô gen giữ/loại."""
    keep = GeneFilter().fit(X_log).mask_
    sd = X_log.std(axis=0, ddof=1).to_numpy()
    mean = X_log.mean(axis=0).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=figsize(mode, 0.45 if mode == "report" else 0.42))
    bins = np.linspace(0, np.nanmax(sd), 50)
    axes[0].hist([sd[~keep], sd[keep]], bins=bins, stacked=True, color=[NEUTRAL_LIGHT, NEUTRAL],
                 edgecolor="white", linewidth=0.3)
    axes[0].set_xlabel("Độ lệch chuẩn của log10 (qua 72 mẫu)")
    axes[0].set_ylabel("Số gen")
    axes[0].text(0.97, 0.92, f"giữ {keep.sum():,}".replace(",", "."), color=NEUTRAL, transform=axes[0].transAxes, ha="right", va="top")
    axes[0].text(0.97, 0.80, f"loại {(~keep).sum():,}".replace(",", "."), color="#7F7F7F", transform=axes[0].transAxes, ha="right", va="top")
    axes[1].scatter(mean[~keep], sd[~keep], s=3 if mode == "report" else 6, color=NEUTRAL_LIGHT, alpha=0.6, rasterized=True)
    axes[1].scatter(mean[keep], sd[keep], s=3 if mode == "report" else 6, color=NEUTRAL, alpha=0.6, rasterized=True)
    axes[1].set_xlabel("Trung bình log10")
    axes[1].set_ylabel("Độ lệch chuẩn log10")
    _title(fig, "F2b. Độ biến thiên của gen và kết quả lọc (đậm: giữ, nhạt: loại)")
    fig.tight_layout()
    return fig


def _sample_order(samples: pd.DataFrame) -> pd.DataFrame:
    return samples.reset_index().sort_values(["split", "source", "class", "sample_id"], ascending=[False, True, True, True])


@styled
def plot_box_per_sample(X_log: pd.DataFrame, samples: pd.DataFrame, mode: str = "report"):
    """F03 — Có mẫu bất thường (chip lỗi) hoặc khác biệt theo nguồn? 72 hộp, màu theo nguồn, gạch chéo = AML.

    Hộp sắp theo (tập, nguồn, lớp) để các khối nguồn liền nhau; râu = 1.5 IQR; không vẽ điểm ngoại lai
    (mỗi mẫu có 7129 giá trị). Mẫu có cờ qc_outlier được chú thích trực tiếp.
    """
    _check_aligned(X_log, samples)
    meta = _sample_order(samples)
    fig, ax = plt.subplots(figsize=figsize(mode, 0.5 if mode == "report" else 0.45))
    pos, gap, prev, group_spans = 0, 1.0, None, []
    xs, tops = {}, {}
    for _, r in meta.iterrows():
        key = (r["split"], r["source"])
        if prev is not None and key != prev:
            pos += gap
        if not group_spans or group_spans[-1][0] != key:
            group_spans.append([key, pos, pos])
        group_spans[-1][2] = pos
        xs[r["sample_id"]] = pos
        pos += 1
        prev = key
    for sid, x in xs.items():
        v = X_log.loc[sid].to_numpy(float)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        lo = v[v >= q1 - 1.5 * iqr].min()
        hi = v[v <= q3 + 1.5 * iqr].max()
        src, cls = samples.loc[sid, "source"], samples.loc[sid, "class"]
        flagged = "qc_outlier" in samples and bool(samples.loc[sid, "qc_outlier"])
        ax.bxp([dict(med=med, q1=q1, q3=q3, whislo=lo, whishi=hi, fliers=[])], positions=[x], widths=0.8,
               patch_artist=True, showfliers=False, manage_ticks=False,
               boxprops=dict(facecolor=SOURCE_COLORS.get(src, NEUTRAL_LIGHT), edgecolor="black",
                             linewidth=2.2 if flagged else 0.5, hatch="////" if cls == "AML" else ""),
               medianprops=dict(color="black", linewidth=1.0),
               whiskerprops=dict(linewidth=1.6 if flagged else 0.5), capprops=dict(linewidth=1.6 if flagged else 0.5))
        tops[sid] = hi
    ytop = ax.get_ylim()[1]
    ax.set_ylim(ax.get_ylim()[0], ytop + 0.45)
    if "qc_outlier" in samples:
        for sid in samples.index[samples["qc_outlier"].astype(bool)]:
            ax.annotate(f"mẫu {sid} (qc_outlier)", (xs[sid], tops[sid]), xytext=(xs[sid], ytop + 0.3),
                        ha="center", va="bottom", fontsize=FIG_STYLES[mode]["fontsize"],
                        arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_xticks([(a + b) / 2 for _, a, b in group_spans], [k[1] for k, _, _ in group_spans])
    ax.tick_params(axis="x", length=0)
    # nhãn tập (train/test) bên dưới nhãn nguồn: gộp các khối cùng tập
    for split in ("train", "test"):
        spans = [(a, b) for (sp, _), a, b in group_spans if sp == split]
        if spans:
            a, b = min(s[0] for s in spans), max(s[1] for s in spans)
            ax.annotate("", xy=(a - 0.4, -0.16), xytext=(b + 0.4, -0.16), xycoords=("data", "axes fraction"),
                        textcoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="-", lw=1.0, color=NEUTRAL))
            ax.text((a + b) / 2, -0.2, f"tập {split}", transform=ax.get_xaxis_transform(), ha="center", va="top")
    ax.set_ylabel("log10 biểu hiện")
    ax.set_yticks(np.arange(2.0, 4.01, 0.5))
    ax.set_xlim(-1, pos)
    handles = [Patch(facecolor="white", edgecolor="black", label="ALL"),
               Patch(facecolor="white", edgecolor="black", hatch="////", label="AML")]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              borderaxespad=0.1, title=None)
    _title(fig, "F3. Phân phối biểu hiện (log10) của từng mẫu, màu theo nguồn", y=1.07)
    fig.tight_layout()
    return fig


@styled
def plot_scree(ratio, mode: str = "report", n_show: int = 10):
    """F04a — Scree plot: phương sai giải thích từng PC (cột, từ 0) và tích lũy (đường, nhãn trực tiếp)."""
    ratio = np.asarray(ratio)[:n_show]
    k = np.arange(1, len(ratio) + 1)
    fig, ax = plt.subplots(figsize=figsize(mode, 0.6 if mode == "report" else 0.5, 0.8 if mode == "report" else 0.75))
    ax.bar(k, ratio * 100, color=NEUTRAL, width=0.7)
    for x, r in zip(k[:3], ratio[:3]):
        ax.text(x, r * 100 + 0.6, f"{r:.1%}", ha="center", va="bottom")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(k, np.cumsum(ratio) * 100, color=NEUTRAL_LIGHT, marker="o", ms=4, zorder=1)
    ax2.set_ylim(0, 100)
    ax2.text(k[-1], np.cumsum(ratio)[-1] * 100 + 3, "tích lũy", ha="right", va="bottom", color=NEUTRAL)
    ax.set_xticks(k, [f"PC{i}" for i in k], rotation=45 if mode == "report" else 0)
    ax.set_ylabel("Phương sai giải thích (%)")
    ax2.set_ylabel("Tích lũy (%)")
    _title(fig, "F4a. Phương sai giải thích bởi các thành phần chính")
    fig.tight_layout()
    return fig


@styled
def plot_pca_by_class(scores: pd.DataFrame, ratio, samples: pd.DataFrame, mode: str = "report"):
    """F04b — PCA PC1–PC2 có cấu trúc lớp không? Trái: theo lớp (ALL/AML); phải: theo nhóm con (B/T-ALL/AML)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize(mode, 0.5 if mode == "report" else 0.45), sharex=True, sharey=True)
    for ax, col in zip(axes, ("class", "subtype")):
        pca_scatter(None, samples, color=col, shape=col, scores=scores, ratio=ratio, ax=ax, mode=mode,
                    label_outliers=(col == "class"))
        ax.set_title({"class": "Theo lớp", "subtype": "Theo nhóm con"}[col])
    axes[1].set_ylabel("")
    _title(fig, "F4b. PCA (PC1–PC2) của 72 mẫu trên các gen đã lọc")
    fig.tight_layout()
    return fig


@styled
def plot_embeddings(embeddings: dict[str, pd.DataFrame], samples: pd.DataFrame, mode: str = "report"):
    """F04c — Cấu trúc lớp có bền qua các phương pháp giảm chiều? PCA / t-SNE / UMAP, cùng màu và hình."""
    n = len(embeddings)
    fig, axes = plt.subplots(1, n, figsize=figsize(mode, 0.36 if mode == "report" else 0.38), squeeze=False)
    axes = axes[0]
    s = FIG_STYLES[mode]
    for ax, (name, emb) in zip(axes, embeddings.items()):
        meta = samples.loc[emb.index]
        for cls in ("ALL", "AML"):
            idx = meta.index[meta["class"] == cls]
            ax.scatter(emb.loc[idx, "dim1"], emb.loc[idx, "dim2"], c=CLASS_COLORS[cls], marker=CLASS_MARKERS[cls],
                       s=s["marker"], edgecolors="black", linewidths=0.5, alpha=0.85)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("PC1" if name == "PCA" else "")
        if name == "PCA":
            ax.set_ylabel("PC2")
    fig.legend(handles=legend_handles(samples, "class", "class", mode), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    fig.text(0.5, -0.19, "Khoảng cách giữa các cụm trên t-SNE và UMAP không có ý nghĩa định lượng.",
             ha="center", va="top", fontsize=s["fontsize"])
    _title(fig, "F4c. PCA, t-SNE và UMAP của 72 mẫu")
    fig.tight_layout()
    return fig


def sample_linkage(corr: np.ndarray, method: str = "average") -> np.ndarray:
    """Phân cụm phân cấp theo khoảng cách 1 − Pearson, liên kết average (mục 3.1.2)."""
    d = 1.0 - np.asarray(corr, float)
    np.fill_diagonal(d, 0.0)
    d = np.clip((d + d.T) / 2, 0, None)
    return linkage(squareform(d, checks=False), method=method)


def plot_sample_correlation(X: pd.DataFrame, samples: pd.DataFrame, mode: str = "report", standardize: bool = True):
    """F04e — Các mẫu cùng lớp/nguồn có giống nhau hơn? Ma trận tương quan Pearson giữa 72 mẫu,
    sắp theo cây phân cấp (1 − r, average); thanh màu trên: lớp, trái: nguồn mẫu.

    standardize=True: z-score từng gen trước khi tính tương quan (nếu không, mọi cặp mẫu đều r ≈ 0.9
    vì mức nền của gen chi phối)."""
    import seaborn as sns
    _check_aligned(X, samples)
    M = zscore(X) if standardize else X.to_numpy(float)
    corr = np.corrcoef(M)
    Z = sample_linkage(corr)
    meta = samples.loc[X.index]
    colors = pd.DataFrame({"Lớp": meta["class"].map(CLASS_COLORS).to_numpy(),
                           "Nguồn": meta["source"].map(SOURCE_COLORS).to_numpy()}, index=X.index)
    off = corr[~np.eye(len(corr), dtype=bool)]
    cdf = pd.DataFrame(np.where(np.eye(len(corr), dtype=bool), np.nan, corr), index=X.index, columns=X.index)
    vmin = float(np.floor(off.min() * 20) / 20)   # thang theo các cặp khác nhau, không tính đường chéo (r = 1)
    vmax = float(np.ceil(off.max() * 20) / 20)
    with plt.rc_context(style_rc(mode)):
        side = FIG_STYLES[mode]["width"] * (0.85 if mode == "report" else 0.6)
        g = sns.clustermap(cdf, row_linkage=Z, col_linkage=Z, cmap=SEQUENTIAL_CMAP, vmin=vmin, vmax=vmax,
                           col_colors=colors["Lớp"], row_colors=colors["Nguồn"], xticklabels=False,
                           yticklabels=False, figsize=(side, side), dendrogram_ratio=0.13, colors_ratio=0.035,
                           cbar_pos=(0.0, 0.84, 0.025, 0.12), cbar_kws={"label": "Pearson r"})
        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")
        handles = ([Patch(facecolor=c, label=k) for k, c in CLASS_COLORS.items()]
                   + [Patch(facecolor=c, label=k) for k, c in SOURCE_COLORS.items()])
        g.fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.12),
                     title="Thanh màu trên: lớp · thanh màu trái: nguồn mẫu")
        g.fig.suptitle("F4e. Tương quan giữa các mẫu, sắp theo phân cụm phân cấp", y=1.02)
    return g.fig


# --------------------------------------------------------------------------------------
# 5. Vẽ lại toàn bộ hình EDA và ghi bảng tóm tắt
# --------------------------------------------------------------------------------------
def render_eda(Xraw: pd.DataFrame, Xl: pd.DataFrame, Xf: pd.DataFrame, samples: pd.DataFrame,
               modes: tuple[str, ...] = MODES, base_dir: Path | None = None, seed: int = RANDOM_SEED,
               embedding_methods: tuple[str, ...] = ("pca", "tsne", "umap")) -> list[Path]:
    """Vẽ F02, F02b, F03, F04a, F04b, F04c, F04e cho các chế độ; trả về danh sách file đã ghi."""
    scores, ratio = compute_pca(Xf, 10, seed)
    emb = compute_embeddings(Xf, seed, embedding_methods)
    builders = {
        "F02_hist_before_after_log": lambda m: plot_hist_log(Xraw, Xl, mode=m),
        "F02b_gene_variance": lambda m: plot_gene_variance(Xl, mode=m),
        "F03_box_per_sample": lambda m: plot_box_per_sample(Xl, samples, mode=m),
        "F04a_pca_scree": lambda m: plot_scree(ratio, mode=m),
        "F04b_pca_by_class": lambda m: plot_pca_by_class(scores, ratio, samples, mode=m),
        "F04c_tsne_umap": lambda m: plot_embeddings(emb, samples, mode=m),
        "F04e_sample_correlation": lambda m: plot_sample_correlation(Xf, samples, mode=m),
    }
    paths = []
    for mode in modes:
        for stem, build in builders.items():
            fig = build(mode)
            paths.append(save_figure(fig, stem, mode, base_dir))
            plt.close(fig)
    return paths


def main() -> None:
    plt.switch_backend("Agg")
    Xraw, samples = load_golub("cleansed")          # giá trị gốc, có cờ qc_outlier
    Xl, _ = load_golub("curated")                   # threshold + log10 (đầu vào của mọi mô hình)
    Xf = pd.read_parquet(CURATED_DIR / "X_log10_filtered.parquet")
    Xf.index = Xf.index.astype(int)
    assert Xf.index.equals(samples.index), "Xf và samples lệch thứ tự mẫu"
    paths = render_eda(Xraw, Xl, Xf, samples)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "eda_summary.json"
    out.write_text(json.dumps(eda_summary(Xraw, Xl, Xf, samples), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã vẽ {len(paths)} hình EDA (F02, F02b, F03, F04a, F04b, F04c, F04e) và ghi {out.name}.")
    print("Chưa thuộc module này: F01, F04d (TV1, notebook 02), F10, F11 (TV2, evaluate.py); "
          "F05–F09, F12 sẽ thêm ở công việc 2.6.")


if __name__ == "__main__":
    main()
