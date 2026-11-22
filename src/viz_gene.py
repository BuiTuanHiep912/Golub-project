"""TV4 - src/viz_gene.py (công việc 2.2, Khám phá gen; ADR-014, ADR-052).

Hàm vẽ cho màn hình Khám phá gen và cho báo cáo. Theo ADR-014, hàm vẽ mới của TV4 nằm ở file riêng này,
dùng màu, hình dạng điểm và cỡ chữ từ src/viz.py (TV6), không sửa viz.py.

Mỗi hình trả lời một câu hỏi:
    Volcano                  Gen nào khác biệt mạnh VÀ có ý nghĩa thống kê?
    Biểu hiện một gen        Gen này cao ở lớp nào, ở phòng lab / loại mô nào?
    Nhất quán của lô         Trong một lô mới (ví dụ St-Jude), gen dấu ấn có giữ chiều ALL/AML không?
    Thanh SNR hai phía       Những gen nào dẫn đầu mỗi phía?

Hai bản cho mỗi hình:
    *_chart(...)   -> altair.Chart, tương tác (di chuột), cho dashboard của TV6
    plot_*(...)    -> matplotlib Figure, mode="report" | "slide", cho báo cáo và slide

Hàm vẽ chỉ nhận DataFrame từ src/gene_explorer.py, không tự đọc dữ liệu:
    volcano_chart(ge.ranking_table(top_n=None), highlight="M23197_at")
    gene_expression_chart(ge.gene_values("M23197_at"), facet="source")
    consistency_chart(ge.marker_consistency(source="St-Jude"))
    snr_bar_chart(ge.ranking_table(top_n=None), n_per_side=25)

Vẽ lại hình cho báo cáo và slide: python -m src.viz_gene
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

try:
    import altair as alt
except ModuleNotFoundError:          # altair đi kèm streamlit; thiếu thì chỉ bản matplotlib dùng được
    alt = None

try:
    from src.viz import (CLASS_COLORS, CLASS_MARKERS, FIG_STYLES, NEUTRAL, NEUTRAL_LIGHT, SOURCE_COLORS,
                         figsize, save_figure, styled)
except ModuleNotFoundError:
    from viz import (CLASS_COLORS, CLASS_MARKERS, FIG_STYLES, NEUTRAL, NEUTRAL_LIGHT, SOURCE_COLORS,
                     figsize, save_figure, styled)

Q_THRESHOLD = 0.05
CLASS_ORDER = ["ALL", "AML"]
FACET_ORDER = {"source": ["DFCI", "CALGB", "St-Jude", "CCG"], "tissue": ["BM", "PB"]}
FACET_LABELS = {"source": "Phòng lab", "tissue": "Loại mô"}
DIRECTION_LABELS = {"ALL": "Cao ở ALL", "AML": "Cao ở AML"}
FLIPPED_COLOR = "#000000"            # gen đảo chiều: đen, hình chữ X (khác hẳn chấm màu theo lớp)
ALTAIR_SHAPES = {"ALL": "circle", "AML": "triangle-up"}


def _need_altair():
    if alt is None:
        raise ImportError("Cần altair (cài kèm streamlit) để vẽ bản tương tác")


def _class_scale():
    return alt.Scale(domain=CLASS_ORDER, range=[CLASS_COLORS[c] for c in CLASS_ORDER])


def _label(row) -> str:
    sym = row.get("gene_symbol")
    return sym if isinstance(sym, str) and sym else row["probe_id"]


def _volcano_frame(ranking: pd.DataFrame, q_threshold: float) -> pd.DataFrame:
    for col in ("probe_id", "snr", "q_value"):
        if col not in ranking.columns:
            raise ValueError(f"ranking cần cột {col!r} (dùng ge.ranking_table(top_n=None))")
    df = ranking.copy()
    df["neg_log10_q"] = -np.log10(df["q_value"].clip(lower=1e-300))
    df["direction"] = np.where(df["snr"] > 0, "ALL", "AML")
    df["significant"] = df["q_value"] < q_threshold
    df["group"] = np.where(df["significant"], df["direction"].map(DIRECTION_LABELS), f"q ≥ {q_threshold:g}")
    df["label"] = df.apply(_label, axis=1)
    return df


def _split_counts(values: pd.DataFrame, facet: str) -> pd.DataFrame:
    return values.groupby([facet, "class"], observed=True).size().rename("n").reset_index()


# ======================================================================== bản tương tác (Altair)

def volcano_chart(ranking: pd.DataFrame, highlight: str | None = None,
                  q_threshold: float = Q_THRESHOLD) -> "alt.Chart":
    """Volcano: trục ngang SNR (dương cao ở ALL), trục dọc −log10(q). Gen q ≥ ngưỡng tô xám.

    highlight: probe_id của gen đang chọn, vẽ vòng đen và tên gen.
    """
    _need_altair()
    df = _volcano_frame(ranking, q_threshold)
    groups = [DIRECTION_LABELS["ALL"], DIRECTION_LABELS["AML"], f"q ≥ {q_threshold:g}"]
    colors = [CLASS_COLORS["ALL"], CLASS_COLORS["AML"], NEUTRAL_LIGHT]
    tooltip = [alt.Tooltip("label:N", title="Gen"), alt.Tooltip("probe_id:N", title="Probe"),
               alt.Tooltip("rank:Q", title="Hạng"), alt.Tooltip("snr:Q", title="SNR", format=".2f"),
               alt.Tooltip("q_value:Q", title="q", format=".2e")]
    base = alt.Chart(df)
    pts = base.mark_circle(size=28, opacity=0.75).encode(
        x=alt.X("snr:Q", title="SNR (dương: cao ở ALL, âm: cao ở AML)"),
        y=alt.Y("neg_log10_q:Q", title="−log10(q)"),
        color=alt.Color("group:N", scale=alt.Scale(domain=groups, range=colors), title=None,
                        legend=alt.Legend(orient="top")),
        tooltip=tooltip)
    thr = alt.Chart(pd.DataFrame({"y": [-np.log10(q_threshold)]})).mark_rule(
        color=NEUTRAL, strokeDash=[4, 4]).encode(y="y:Q")
    layers = [pts, thr]
    if highlight is not None:
        sel = df[df["probe_id"].eq(highlight)]
        if not sel.empty:
            layers.append(alt.Chart(sel).mark_point(size=220, color="black", strokeWidth=2.5).encode(
                x="snr:Q", y="neg_log10_q:Q", tooltip=tooltip))
            layers.append(alt.Chart(sel).mark_text(dy=-16, fontWeight="bold").encode(
                x="snr:Q", y="neg_log10_q:Q", text="label:N"))
    return alt.layer(*layers).properties(width="container", height=380,
                                         title="Gen khác biệt giữa ALL và AML (38 mẫu huấn luyện)")


def gene_expression_chart(values: pd.DataFrame, facet: str = "source", title: str = "") -> "alt.Chart":
    """Biểu hiện (log10) của một gen theo lớp, tách theo phòng lab hoặc loại mô.

    Chấm đặc: mẫu huấn luyện; chấm rỗng: mẫu kiểm tra; viền đen dày: mẫu gắn cờ chất lượng (qc_outlier);
    vạch đen: trung vị; số dưới mỗi cột: số mẫu.
    """
    _need_altair()
    if facet not in FACET_ORDER:
        raise ValueError(f"facet phải là một trong {tuple(FACET_ORDER)}")
    df = values.copy()
    df["qc_outlier"] = df["qc_outlier"].fillna(False).astype(bool) if "qc_outlier" in df else False
    df["Tập"] = df["split"].map({"train": "Huấn luyện", "test": "Kiểm tra"})
    rng = np.random.default_rng(0)
    df["jitter"] = rng.uniform(-0.6, 0.6, len(df))
    counts = _split_counts(df, facet)
    order = [f for f in FACET_ORDER[facet] if f in set(df[facet])]
    return _faceted_expression(df, counts, facet, order, title)


def _faceted_expression(df, counts, facet, order, title):
    """Altair cần chung một nguồn dữ liệu để chia cột; gộp điểm và nhãn n vào một bảng."""
    lab = counts.rename(columns={"n": "n_count"})
    lab["is_label"] = True
    pts_df = df.assign(is_label=False)
    data = pd.concat([pts_df, lab], ignore_index=True)
    x = alt.X("class:N", title=None, sort=CLASS_ORDER, axis=alt.Axis(labelAngle=0))
    base = alt.Chart().encode(x=x)
    pts = base.transform_filter("!datum.is_label").mark_point(size=70, strokeWidth=1.6, opacity=0.9).encode(
        xOffset=alt.XOffset("jitter:Q"),
        y=alt.Y("value:Q", title="Biểu hiện (log10)", scale=alt.Scale(zero=False)),
        shape=alt.Shape("class:N", scale=alt.Scale(domain=CLASS_ORDER,
                                                    range=[ALTAIR_SHAPES[c] for c in CLASS_ORDER]), legend=None),
        stroke=alt.condition("datum.qc_outlier", alt.value("black"),
                             alt.Color("class:N", scale=_class_scale(), legend=None)),
        fill=alt.condition("datum.split == 'train'", alt.Color("class:N", scale=_class_scale(), legend=None),
                           alt.value("white")),
        tooltip=[alt.Tooltip("sample_id:N", title="Mẫu"), alt.Tooltip("class:N", title="Lớp"),
                 alt.Tooltip("subtype:N", title="Nhóm con"), alt.Tooltip("source:N", title="Phòng lab"),
                 alt.Tooltip("tissue:N", title="Loại mô"), alt.Tooltip("Tập:N"),
                 alt.Tooltip("value:Q", title="log10", format=".2f")])
    med = base.transform_filter("!datum.is_label").mark_tick(color="black", thickness=2.5, size=40).encode(
        y=alt.Y("median(value):Q"))
    n_text = base.transform_filter("datum.is_label").mark_text(baseline="bottom", color=NEUTRAL).encode(
        y=alt.value(318), text=alt.Text("n_count:Q", format="d"))
    return (alt.layer(pts, med, n_text, data=data).properties(width=130, height=320)
            .facet(column=alt.Column(f"{facet}:N", sort=order, title=FACET_LABELS[facet],
                                     header=alt.Header(labelOrient="bottom", titleOrient="bottom")))
            .properties(title=title or "Biểu hiện của gen theo lớp (số dưới cột: số mẫu)"))


def consistency_chart(consistency: pd.DataFrame, title: str | None = None) -> "alt.Chart":
    """Gen dấu ấn có giữ chiều trong lô mới không (đầu vào: ge.marker_consistency(...)).

    Trục ngang: chênh lệch ALL − AML trên tập huấn luyện; trục dọc: chênh lệch trong lô. Vùng nền xám là
    giữ chiều (nghiêng về do bệnh). Chấm màu: giữ chiều rõ; chữ X đen: đảo chiều rõ; chấm rỗng xám: chênh
    lệch trong lô quá nhỏ (< MIN_DIFF_LOG10, thường do gần mức sàn) nên không nói được chiều.
    """
    _need_altair()
    df = _consistency_frame(consistency)
    tooltip = [alt.Tooltip("label:N", title="Gen"), alt.Tooltip("probe_id:N", title="Probe"),
               alt.Tooltip("train_diff:Q", title="Huấn luyện ALL − AML", format=".2f"),
               alt.Tooltip("batch_diff:Q", title="Trong lô ALL − AML", format=".2f"),
               alt.Tooltip("status:N", title="Kết quả")]
    lim = float(np.nanmax(np.abs(df[["train_diff", "batch_diff"]].to_numpy()))) * 1.1 or 1.0
    quad = pd.DataFrame({"x": [0, -lim], "x2": [lim, 0], "y": [0, -lim], "y2": [lim, 0]})
    shade = alt.Chart(quad).mark_rect(color=NEUTRAL_LIGHT, opacity=0.25).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-lim, lim])), x2="x2", y=alt.Y("y:Q", scale=alt.Scale(domain=[-lim, lim])),
        y2="y2")
    zero_x = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color=NEUTRAL).encode(x="v:Q")
    zero_y = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color=NEUTRAL).encode(y="v:Q")
    kept = alt.Chart(df[df["status"].eq("Giữ chiều")]).mark_point(size=70, filled=True, opacity=0.85).encode(
        x=alt.X("train_diff:Q", title="Tập huấn luyện: ALL − AML (log10)"),
        y=alt.Y("batch_diff:Q", title="Trong lô: ALL − AML (log10)"),
        color=alt.Color("direction:N", scale=_class_scale(), title="Gen cao ở",
                        legend=alt.Legend(orient="top")),
        shape=alt.Shape("direction:N", scale=alt.Scale(domain=CLASS_ORDER,
                                                        range=[ALTAIR_SHAPES[c] for c in CLASS_ORDER]),
                        legend=None),
        tooltip=tooltip)
    small = alt.Chart(df[df["status"].eq("Chênh lệch nhỏ")]).mark_point(
        size=60, color=NEUTRAL, filled=False, strokeWidth=1.4).encode(
        x="train_diff:Q", y="batch_diff:Q", tooltip=tooltip)
    flipped = alt.Chart(df[df["status"].eq("Đảo chiều")]).mark_point(
        shape="cross", size=150, color=FLIPPED_COLOR, filled=True).encode(
        x="train_diff:Q", y="batch_diff:Q", tooltip=tooltip)
    n_all, n_aml = consistency.attrs.get("n_ALL"), consistency.attrs.get("n_AML")
    head = title or (f"Gen dấu ấn trong lô ({n_all} ALL, {n_aml} AML)" if n_all is not None
                     else "Gen dấu ấn trong lô")
    note = consistency.attrs.get("note", "")
    lines = [s.strip().rstrip(".") + "." for s in note.split(". ") if s.strip()]   # note dài: mỗi câu một dòng
    lines += [_flipped_text(df) + ". Chấm rỗng xám: chênh lệch quá nhỏ để nói chiều. Di chuột để xem tên gen."]
    return alt.layer(shade, zero_x, zero_y, kept, small, flipped).properties(
        width="container", height=380, title=alt.TitleParams(head, subtitle=lines))


def snr_bar_chart(ranking: pd.DataFrame, n_per_side: int = 25) -> "alt.Chart":
    """Thanh SNR hai phía: n gen hạng cao nhất mỗi phía, dài theo |SNR|, màu theo lớp gen cao hơn."""
    _need_altair()
    df = _snr_frame(ranking, n_per_side)
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("snr:Q", title="SNR (dương: cao ở ALL, âm: cao ở AML)"),
        y=alt.Y("label:N", sort=alt.EncodingSortField("snr", order="descending"), title=None),
        color=alt.Color("direction:N", scale=_class_scale(), title="Gen cao ở", legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("label:N", title="Gen"), alt.Tooltip("probe_id:N", title="Probe"),
                 alt.Tooltip("rank:Q", title="Hạng"), alt.Tooltip("snr:Q", title="SNR", format=".2f"),
                 alt.Tooltip("q_value:Q", title="q", format=".2e")],
    ).properties(width="container", height=max(300, 13 * len(df)),
                 title=f"{n_per_side} gen dẫn đầu mỗi phía (38 mẫu huấn luyện)")


def _consistency_frame(consistency: pd.DataFrame) -> pd.DataFrame:
    need = ("probe_id", "train_diff", "batch_diff", "same_direction")
    miss = [c for c in need if c not in consistency.columns]
    if miss:
        raise ValueError(f"cần đầu ra của ge.marker_consistency(); thiếu cột {miss}")
    df = consistency.copy()
    df["label"] = df.apply(_label, axis=1)
    df["direction"] = np.where(df["train_diff"] > 0, "ALL", "AML")
    if "status" not in df.columns:      # bảng cũ không có status: suy từ dấu
        df["status"] = np.where(df["same_direction"].isna(), "Không so được",
                                np.where(df["same_direction"].fillna(False).astype(bool), "Giữ chiều", "Đảo chiều"))
    return df


def _flipped_text(df: pd.DataFrame) -> str:
    names = df.loc[df["status"].eq("Đảo chiều"), "label"].tolist()
    return ("Đảo chiều rõ: " + ", ".join(names)) if names else "Không gen nào đảo chiều rõ"


def _snr_frame(ranking: pd.DataFrame, n_per_side: int) -> pd.DataFrame:
    up = ranking[ranking["snr"] > 0].nsmallest(n_per_side, "rank")
    down = ranking[ranking["snr"] < 0].nsmallest(n_per_side, "rank")
    df = pd.concat([up, down]).copy()
    df["direction"] = np.where(df["snr"] > 0, "ALL", "AML")
    base = df.apply(_label, axis=1)
    dup = base.duplicated(keep=False)
    df["label"] = np.where(dup, base + " (" + df["probe_id"] + ")", base)
    return df


# ======================================================================== bản tĩnh (matplotlib)

def _ms(mode: str) -> float:
    return FIG_STYLES[mode]["marker"]


@styled
def plot_volcano(ranking: pd.DataFrame, highlight: str | None = None, q_threshold: float = Q_THRESHOLD,
                 mode: str = "report") -> plt.Figure:
    """Volcano cho báo cáo/slide. Trả lời: gen nào khác biệt mạnh và có ý nghĩa thống kê?"""
    df = _volcano_frame(ranking, q_threshold)
    fig, ax = plt.subplots(figsize=figsize(mode, 0.7))
    ns = df[~df["significant"]]
    ax.scatter(ns["snr"], ns["neg_log10_q"], s=_ms(mode) * 0.35, color=NEUTRAL_LIGHT, lw=0,
               label=f"q ≥ {q_threshold:g} ({len(ns)} gen)")
    for d in CLASS_ORDER:
        sub = df[df["significant"] & df["direction"].eq(d)]
        ax.scatter(sub["snr"], sub["neg_log10_q"], s=_ms(mode) * 0.45, color=CLASS_COLORS[d], lw=0, alpha=0.8,
                   marker=CLASS_MARKERS[d], label=f"{DIRECTION_LABELS[d]} ({len(sub)} gen)")
    ax.axhline(-np.log10(q_threshold), color=NEUTRAL, ls="--", lw=1)
    ax.axvline(0, color=NEUTRAL, lw=0.8)
    if highlight is not None and highlight in set(df["probe_id"]):
        r = df[df["probe_id"].eq(highlight)].iloc[0]
        ax.scatter([r["snr"]], [r["neg_log10_q"]], s=_ms(mode) * 3, facecolors="none", edgecolors="black", lw=2)
        ax.annotate(r["label"], (r["snr"], r["neg_log10_q"]), xytext=(8, 6), textcoords="offset points",
                    fontweight="bold")
    ax.set_xlabel("SNR (dương: cao ở ALL, âm: cao ở AML)")
    ax.set_ylabel("−log10(q)")
    ax.set_title("Gen khác biệt giữa ALL và AML (38 mẫu huấn luyện)")
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    return fig


@styled
def plot_gene_expression(values: pd.DataFrame, facet: str = "source", title: str | None = None,
                         mode: str = "report") -> plt.Figure:
    """Biểu hiện một gen theo lớp, mỗi ô một phòng lab (hoặc loại mô).

    Chấm đặc: huấn luyện; chấm rỗng: kiểm tra; vòng đen: mẫu gắn cờ chất lượng; vạch đen: trung vị.
    Ô có cả hai lớp được tô nền nhạt: chỉ ở đó mới so được ALL với AML trong cùng một phòng lab.
    """
    if facet not in FACET_ORDER:
        raise ValueError(f"facet phải là một trong {tuple(FACET_ORDER)}")
    order = [f for f in FACET_ORDER[facet] if f in set(values[facet])]
    qc = values["qc_outlier"].fillna(False).astype(bool) if "qc_outlier" in values else pd.Series(False, values.index)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(order), figsize=figsize(mode, 0.55), sharey=True, squeeze=False)
    for ax, level in zip(axes[0], order):
        sub = values[values[facet].eq(level)]
        both = sub["class"].nunique() == 2
        if both:
            ax.set_facecolor("#F2F2F2")
        for i, cls in enumerate(CLASS_ORDER):
            s = sub[sub["class"].eq(cls)]
            if s.empty:
                continue
            xs = i + rng.uniform(-0.18, 0.18, len(s))
            train = s["split"].eq("train").to_numpy()
            color = CLASS_COLORS[cls]
            ax.scatter(xs[train], s["value"][train], s=_ms(mode), marker=CLASS_MARKERS[cls], color=color,
                       edgecolors=color, lw=1)
            ax.scatter(xs[~train], s["value"][~train], s=_ms(mode), marker=CLASS_MARKERS[cls],
                       facecolors="white", edgecolors=color, lw=1.4)
            flagged = qc.loc[s.index].to_numpy()
            if flagged.any():
                ax.scatter(xs[flagged], s["value"][flagged], s=_ms(mode) * 3.2, facecolors="none",
                           edgecolors="black", lw=1.8)
            ax.hlines(s["value"].median(), i - 0.3, i + 0.3, color="black", lw=2.2)
        ax.set_xticks([0, 1])
        counts = sub["class"].value_counts()
        ax.set_xticklabels([f"{c}\nn={counts.get(c, 0)}" for c in CLASS_ORDER])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(level + (" (đủ 2 lớp)" if both and facet == "source" else ""),
                     fontweight="bold" if both else "normal")
    axes[0][0].set_ylabel("Biểu hiện (log10)")
    handles = [Line2D([], [], ls="", marker="o", color=NEUTRAL, label="Huấn luyện"),
               Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor=NEUTRAL,
                      label="Kiểm tra")]
    if qc.any():
        handles.append(Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor="black",
                              markersize=11, label="Gắn cờ chất lượng"))
    handles.append(Line2D([], [], color="black", lw=2.2, label="Trung vị"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(title or "Biểu hiện của gen theo lớp")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


@styled
def plot_consistency(consistency: pd.DataFrame, title: str | None = None, mode: str = "report") -> plt.Figure:
    """Gen dấu ấn có giữ chiều trong lô mới không. Đầu vào: ge.marker_consistency(...)."""
    df = _consistency_frame(consistency)
    fig, ax = plt.subplots(figsize=figsize(mode, 0.6))
    vals = df[["train_diff", "batch_diff"]].to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(vals))) * 1.12 if np.isfinite(vals).any() else 1.0
    ax.fill_between([0, lim], 0, lim, color=NEUTRAL_LIGHT, alpha=0.3, lw=0)
    ax.fill_between([-lim, 0], -lim, 0, color=NEUTRAL_LIGHT, alpha=0.3, lw=0)
    ax.axhline(0, color=NEUTRAL, lw=0.8)
    ax.axvline(0, color=NEUTRAL, lw=0.8)
    for d in CLASS_ORDER:
        sub = df[df["status"].eq("Giữ chiều") & df["direction"].eq(d)]
        ax.scatter(sub["train_diff"], sub["batch_diff"], s=_ms(mode), marker=CLASS_MARKERS[d],
                   color=CLASS_COLORS[d], alpha=0.85, label=f"Giữ chiều, gen cao ở {d} ({len(sub)})")
    small = df[df["status"].eq("Chênh lệch nhỏ")]
    ax.scatter(small["train_diff"], small["batch_diff"], s=_ms(mode), marker="o", facecolors="none",
               edgecolors=NEUTRAL, lw=1.2, label=f"Chênh lệch quá nhỏ để nói chiều ({len(small)})")
    flip = df[df["status"].eq("Đảo chiều")]
    ax.scatter(flip["train_diff"], flip["batch_diff"], s=_ms(mode) * 1.8, marker="X", color=FLIPPED_COLOR,
               label=f"Đảo chiều rõ ({len(flip)})")
    ax.text(0.03, 0.97, _flipped_text(df), transform=ax.transAxes, va="top", ha="left")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Tập huấn luyện: ALL − AML (log10)")
    ax.set_ylabel("Trong lô: ALL − AML (log10)")
    n_all, n_aml = consistency.attrs.get("n_ALL"), consistency.attrs.get("n_AML")
    ax.set_title(title or (f"Gen dấu ấn trong lô ({n_all} ALL, {n_aml} AML)" if n_all is not None
                           else "Gen dấu ấn trong lô"))
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.03, 0.5))
    fig.tight_layout()
    return fig


# ======================================================================== vẽ lại hình cho báo cáo

def main() -> None:
    """Vẽ hình của TV4 cho báo cáo và slide vào results/figures/{report,slides}/."""
    try:
        from src import gene_explorer as ge
    except ModuleNotFoundError:
        import gene_explorer as ge

    ranking = ge.ranking_table(top_n=None)
    cd33 = "M23197_at"
    stjude = ge.marker_consistency(source="St-Jude")
    note = (f"St-Jude: gen dấu ấn có giữ chiều? ({stjude.attrs['n_ALL']} ALL, {stjude.attrs['n_AML']} AML)")
    for mode in ("report", "slide"):
        figs = {
            "gene_volcano": plot_volcano(ranking, highlight=cd33, mode=mode),
            "gene_expression_CD33": plot_gene_expression(ge.gene_values(cd33), facet="source",
                                                         title="CD33 theo lớp và phòng lab", mode=mode),
            "gene_consistency_StJude": plot_consistency(stjude, title=note, mode=mode),
        }
        for stem, fig in figs.items():
            print("Đã ghi", save_figure(fig, stem, mode))
            plt.close(fig)


if __name__ == "__main__":
    main()
