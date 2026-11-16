"""Biểu đồ tương tác (Altair) cho dashboard, theo style guide (docs/style_guide.md).

Màu, hình điểm lấy từ src/viz.py (CLASS_COLORS, SOURCE_COLORS, ...), không ghi cứng. Hàm nhận DataFrame
và trả về đối tượng Altair; việc đọc dữ liệu là của backend.py.

Hình tĩnh F03 (boxplot từng mẫu) và F04e (tương quan giữa các mẫu) dùng thẳng hàm matplotlib của viz.py.
Hàm vẽ biểu hiện một gen của TV4 (công việc 2.2) sẽ thay `gene_strip` khi bàn giao.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src import viz

LABELS = dict(viz.LABELS)                       # class/subtype/source/tissue -> nhãn tiếng Việt
LABELS["split"] = "Tập"
SPLIT_NAMES = {"train": "huấn luyện", "test": "kiểm tra"}
CLASS_SHAPES = {"ALL": "circle", "AML": "triangle-up"}   # mã hóa dư thừa: màu + hình (style guide, mục 1)
INK = viz.NEUTRAL                                        # màu nhấn trung tính
GRID = viz.NEUTRAL_LIGHT


def scale(col: str) -> alt.Scale:
    """Thang màu của một cột (class, subtype, source, tissue) lấy từ viz.COLOR_MAPS."""
    m = viz.COLOR_MAPS[col]
    return alt.Scale(domain=list(m), range=list(m.values()))


def _shape_scale() -> alt.Scale:
    return alt.Scale(domain=list(CLASS_SHAPES), range=list(CLASS_SHAPES.values()))


def _tooltips(df: pd.DataFrame) -> list:
    cols = [("sample_id", "Mẫu"), ("class", "Lớp"), ("subtype", "Nhóm con"), ("source", "Nguồn"),
            ("tissue", "Loại mô"), ("split", "Tập")]
    return [alt.Tooltip(f"{c}:N", title=t) for c, t in cols if c in df]


# ------------------------------------------------------------------------------------------------
# PCA tương tác
# ------------------------------------------------------------------------------------------------
def pca_chart(scores: pd.DataFrame, ratio, samples: pd.DataFrame, color_by: str = "class",
              pcs: tuple[int, int] = (1, 2), patient_xy: tuple[float, float] | None = None,
              height: int = 420) -> alt.Chart:
    """PCA 72 bệnh nhân: màu theo `color_by`, hình theo lớp; di chuột để xem thông tin mẫu.

    Mẫu qc_outlier được khoanh vòng đen. `patient_xy` đánh dấu riêng một bệnh nhân (màn hình Hồ sơ).
    """
    xi, yi = f"PC{pcs[0]}", f"PC{pcs[1]}"
    df = scores[[xi, yi]].join(samples).reset_index()
    df["sample_id"] = df["sample_id"].astype(str)
    xt, yt = f"{xi} ({ratio[pcs[0] - 1]:.1%})", f"{yi} ({ratio[pcs[1] - 1]:.1%})"
    enc_x = alt.X(f"{xi}:Q", title=xt, axis=alt.Axis(grid=False))
    enc_y = alt.Y(f"{yi}:Q", title=yt, axis=alt.Axis(grid=False))
    pts = alt.Chart(df).mark_point(filled=True, size=95, stroke="black", strokeWidth=0.5, opacity=0.85).encode(
        x=enc_x, y=enc_y,
        color=alt.Color(f"{color_by}:N", scale=scale(color_by), title=LABELS[color_by]),
        shape=alt.Shape("class:N", scale=_shape_scale(), title=LABELS["class"]),
        tooltip=_tooltips(df))
    layers = [pts]
    if "qc_outlier" in df and df["qc_outlier"].any():
        out = df[df["qc_outlier"]]
        layers.append(alt.Chart(out).mark_point(size=380, filled=False, stroke="black", strokeWidth=1.4)
                      .encode(x=enc_x, y=enc_y))
        layers.append(alt.Chart(out).mark_text(align="right", dx=-14, dy=-12, fontSize=12, color="black")
                      .encode(x=enc_x, y=enc_y, text=alt.Text("label:N"))
                      .transform_calculate(label="'mẫu ' + datum.sample_id + ' (qc_outlier)'"))
    if patient_xy is not None:
        pt = pd.DataFrame({xi: [patient_xy[0]], yi: [patient_xy[1]], "label": ["bệnh nhân này"]})
        layers.append(alt.Chart(pt).mark_point(shape="diamond", size=420, filled=False, stroke="black",
                                               strokeWidth=3).encode(x=enc_x, y=enc_y))
        layers.append(alt.Chart(pt).mark_text(align="left", dx=16, dy=-4, fontSize=13, fontWeight="bold",
                                              color="black").encode(x=enc_x, y=enc_y, text="label:N"))
    return alt.layer(*layers).properties(height=height).interactive()


def project_patient(X_ref: pd.DataFrame, x_new: pd.Series, n_components: int = 2):
    """PCA của các bệnh nhân tham chiếu (z-score theo tham chiếu) và vị trí của một mẫu mới trên cùng bản đồ.

    Trả (scores_ref DataFrame, ratio, (pc1, pc2) của mẫu mới). Mẫu mới không tham gia việc fit.
    """
    mu, sd = X_ref.mean(), X_ref.std().replace(0, np.nan)
    Z = ((X_ref - mu) / sd).fillna(0).to_numpy()
    z = ((x_new.reindex(X_ref.columns) - mu) / sd).fillna(0).to_numpy()[None, :]
    pca = PCA(n_components=n_components, random_state=viz.RANDOM_SEED)
    sc = pd.DataFrame(pca.fit_transform(Z), index=X_ref.index, columns=[f"PC{i + 1}" for i in range(n_components)])
    new = pca.transform(z)[0]
    return sc, list(pca.explained_variance_ratio_), (float(new[0]), float(new[1]))


# ------------------------------------------------------------------------------------------------
# Khám phá gen
# ------------------------------------------------------------------------------------------------
def _jitter(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.28, 0.28, n)


def gene_strip(values: pd.DataFrame, facet: str = "source", title: str = "") -> alt.Chart:
    """Biểu hiện (log10) của một gen theo lớp, tách theo `facet` (source hoặc tissue); vạch ngang = trung vị."""
    df = values.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["jitter"] = _jitter(len(df))
    base = alt.Chart().encode(x=alt.X("class:N", title=None, axis=alt.Axis(labelAngle=0)))
    pts = base.mark_point(filled=True, size=70, stroke="black", strokeWidth=0.5, opacity=0.85).encode(
        xOffset=alt.XOffset("jitter:Q"),
        y=alt.Y("value:Q", title="Biểu hiện (log10)", scale=alt.Scale(zero=False)),
        color=alt.Color("class:N", scale=scale("class"), legend=None),
        shape=alt.Shape("class:N", scale=_shape_scale(), legend=None),
        tooltip=_tooltips(df) + [alt.Tooltip("value:Q", title="log10", format=".2f")])
    med = base.mark_tick(color="black", thickness=2.5, size=46).encode(
        y=alt.Y("median_v:Q")).transform_joinaggregate(median_v="median(value)", groupby=["class", facet])
    return (alt.layer(pts, med, data=df).properties(width=120, height=320)
            .facet(column=alt.Column(f"{facet}:N", title=LABELS[facet], header=alt.Header(labelOrient="bottom",
                                                                                          titleOrient="bottom")))
            .properties(title=title))


# ------------------------------------------------------------------------------------------------
# Chất lượng dữ liệu
# ------------------------------------------------------------------------------------------------
BATCH_METRICS = {"median_log10": "Trung vị (log10)", "iqr_log10": "Độ trải IQR (log10)",
                 "mean_corr": "Tương quan trung bình trong lô"}


def batch_metric_dots(batches: pd.DataFrame) -> alt.Chart:
    """Small multiples: mỗi panel một chỉ số chất lượng, mỗi chấm một phòng lab (màu = nguồn, hình = trạng thái)."""
    long = batches.melt(id_vars=["source", "status"], value_vars=list(BATCH_METRICS), var_name="metric",
                        value_name="value")
    long["metric_label"] = long["metric"].map(BATCH_METRICS)
    long["status_label"] = long["status"].map({"passed": "đạt", "blocked": "bị chặn"})
    src_scale = alt.Scale(domain=list(viz.SOURCE_COLORS), range=list(viz.SOURCE_COLORS.values()))
    ch = alt.Chart(long).mark_point(filled=True, size=190, stroke="black", strokeWidth=0.7, opacity=0.95).encode(
        y=alt.Y("source:N", title=None, sort=list(viz.SOURCE_COLORS)),
        x=alt.X("value:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color("source:N", scale=src_scale, legend=None),
        shape=alt.Shape("status_label:N", title="Trạng thái lô",
                        scale=alt.Scale(domain=["đạt", "bị chặn"], range=["circle", "cross"])),
        tooltip=[alt.Tooltip("source:N", title="Nguồn"), alt.Tooltip("metric_label:N", title="Chỉ số"),
                 alt.Tooltip("value:Q", title="Giá trị", format=".3f"),
                 alt.Tooltip("status_label:N", title="Trạng thái")])
    return ch.properties(width=190, height=140).facet(column=alt.Column("metric_label:N", title=None)) \
             .resolve_scale(x="independent")


# ------------------------------------------------------------------------------------------------
# Hồ sơ bệnh nhân
# ------------------------------------------------------------------------------------------------
def _gene_label(probe_id: str, description: str, n: int = 26) -> str:
    d = str(description)
    return f"{probe_id} · {d[:n]}{'…' if len(d) > n else ''}"


def votes_chart(votes: pd.DataFrame, top: int = 10) -> alt.Chart:
    """Các gen bỏ phiếu mạnh nhất: thanh sang phải = phiếu cho ALL, sang trái = phiếu cho AML."""
    v = votes.reindex(votes["vote"].abs().sort_values(ascending=False).index).head(top).copy()
    v["gene"] = [_gene_label(p, d, 14) for p, d in zip(v["probe_id"], v["description"])]
    v["toward_label"] = "bỏ phiếu cho " + v["toward"]
    cmap = viz.CLASS_COLORS
    return alt.Chart(v).mark_bar(stroke="black", strokeWidth=0.5).encode(
        y=alt.Y("gene:N", sort=list(v["gene"]), title=None, axis=alt.Axis(labelLimit=190)),
        x=alt.X("vote:Q", title="Phiếu bầu (dương: ALL · âm: AML)"),
        color=alt.Color("toward:N", scale=alt.Scale(domain=list(cmap), range=list(cmap.values())),
                        title="Bỏ phiếu cho"),
        tooltip=[alt.Tooltip("probe_id:N", title="Probe"), alt.Tooltip("description:N", title="Mô tả"),
                 alt.Tooltip("value:Q", title="Biểu hiện bệnh nhân", format=".2f"),
                 alt.Tooltip("vote:Q", title="Phiếu", format=".2f"), alt.Tooltip("toward_label:N", title="")],
    ).properties(height=26 * len(v) + 20)


def patient_vs_groups(votes: pd.DataFrame, X_ref: pd.DataFrame, samples_ref: pd.DataFrame,
                      top: int = 6) -> alt.Chart:
    """Mức biểu hiện của bệnh nhân (vạch đen đứt) đặt cạnh phân bố của nhóm ALL và nhóm AML tham chiếu."""
    top_v = votes.reindex(votes["vote"].abs().sort_values(ascending=False).index).head(top)
    rows, lines = [], []
    for p, d, val in zip(top_v["probe_id"], top_v["description"], top_v["value"]):
        lab = _gene_label(p, d, 20)
        for sid, v in X_ref[p].items():
            rows.append({"gene": lab, "class": samples_ref.loc[sid, "class"], "value": float(v), "sample_id": str(sid)})
        lines.append({"gene": lab, "value": float(val)})
    pts_df, line_df = pd.DataFrame(rows), pd.DataFrame(lines)
    pts_df["jitter"] = _jitter(len(pts_df), seed=3)
    order = [_gene_label(p, d, 20) for p, d in zip(top_v["probe_id"], top_v["description"])]
    return _facet_layers(pts_df, line_df, order)


def _facet_layers(pts_df: pd.DataFrame, line_df: pd.DataFrame, order: list) -> alt.Chart:
    base = alt.Chart().encode()
    pts = base.mark_point(filled=True, size=45, stroke="black", strokeWidth=0.4, opacity=0.8).encode(
        x=alt.X("class:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("jitter:Q"),
        y=alt.Y("value:Q", title="log10", scale=alt.Scale(zero=False)),
        color=alt.Color("class:N", scale=scale("class"), legend=None),
        shape=alt.Shape("class:N", scale=_shape_scale(), legend=None),
        tooltip=[alt.Tooltip("sample_id:N", title="Mẫu tham chiếu"), alt.Tooltip("class:N", title="Lớp"),
                 alt.Tooltip("value:Q", title="log10", format=".2f")])
    # Vạch bệnh nhân: nhúng giá trị vào chính bảng điểm (cột patient_value) để một dataset duy nhất cho facet
    pv = line_df.set_index("gene")["value"]
    df = pts_df.assign(patient_value=pts_df["gene"].map(pv))
    rule = base.mark_rule(color="black", strokeDash=[5, 3], strokeWidth=2).encode(
        y=alt.Y("patient_value:Q"), tooltip=[alt.Tooltip("patient_value:Q", title="Bệnh nhân này", format=".2f")])
    return (alt.layer(pts, rule, data=df).properties(width=150, height=150)
            .facet(facet=alt.Facet("gene:N", sort=order, title=None,
                                   header=alt.Header(labelFontSize=11, labelLimit=170)), columns=3)
            .resolve_scale(y="independent"))
