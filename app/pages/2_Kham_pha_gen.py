"""Màn hình Khám phá gen (nhà nghiên cứu).

Giao diện: TV6 (2.6). Dữ liệu: src/gene_explorer.py của TV4 (2.2), chú giải gen của TV5 (2.5) — qua backend.py.
Nội dung theo kế hoạch mục D.2: ô tìm gen, biểu hiện của gen đã chọn (tách theo phòng lab / loại mô),
bảng xếp hạng gen, PCA tương tác, tương quan giữa các mẫu.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import backend, charts, ui  # noqa: E402
from src import viz  # noqa: E402

ui.page_header("Khám phá gen", "Nhà nghiên cứu",
               "Gen nào khác biệt giữa ALL và AML, và khác biệt đó do bệnh hay do phòng lab?")

X, samples = backend.reference_data()

tab_gene, tab_rank, tab_pca, tab_corr = st.tabs(
    ["Tra cứu gen", "Xếp hạng gen", "PCA tương tác", "Tương quan giữa các mẫu"])

# ---- Tra cứu một gen ----
with tab_gene:
    query = st.text_input("Tên hoặc mã gen", value="CD33", key="gene_query",
                          placeholder="ví dụ CD33, M23197_at, zyxin")
    hits = backend.find_gene(query) if query.strip() else pd.DataFrame()
    if hits.empty:
        st.warning("Không tìm thấy gen nào khớp. Thử ký hiệu gen (CD33), mã probe (M23197_at) hoặc một phần tên gen.")
    else:
        def _fmt(pid: str) -> str:
            r = hits.set_index("probe_id").loc[pid]
            name = r["gene_symbol"] or str(r["description"])[:48]
            rank = f"hạng {int(r['rank'])}" if pd.notna(r["rank"]) else "không có trong xếp hạng"
            return f"{pid} · {name} · {rank}"

        probe = (hits["probe_id"].iloc[0] if len(hits) == 1 else
                 st.selectbox(f"Có {len(hits)} probe khớp, chọn một probe", hits["probe_id"], format_func=_fmt))
        row = hits.set_index("probe_id").loc[probe]
        ann = backend.annotate(hits.set_index("probe_id").loc[[probe]].reset_index())

        m1, m2, m3 = st.columns(3)
        m1.metric("Probe", probe)
        m2.metric("Hạng (38 mẫu huấn luyện)", int(row["rank"]) if pd.notna(row["rank"]) else "—")
        m3.metric("Số probe cùng mô tả", int(row["n_probes_same_description"]))
        st.markdown(f"**Mô tả:** {row['description']}")
        func = ann["function"].iloc[0] if "function" in ann else None
        if isinstance(func, str) and func:
            st.markdown(f"**Chức năng:** {func}  \n**Dòng tế bào:** {ann['cell_lineage'].iloc[0]}")
        else:
            st.caption("Chưa có chú giải chức năng cho gen này (đang chờ gene_annotations.csv của TV5).")

        facet_label = st.radio("Tách biểu hiện theo", ["Phòng lab", "Loại mô"], horizontal=True)
        facet = "source" if facet_label == "Phòng lab" else "tissue"
        vals = backend.gene_values(probe)
        st.altair_chart(charts.gene_strip(vals, facet=facet), width="content")
        st.caption("Mỗi điểm là một bệnh nhân; vạch đen là trung vị. Nếu hai lớp tách nhau ở mọi phòng lab thì "
                   "khác biệt do bệnh; nếu chỉ tách giữa các phòng lab thì nghi do nguồn mẫu.")

        c_stats, c_dos = st.columns(2)
        with c_stats:
            st.markdown("**Thống kê theo nhóm**")
            st.dataframe(backend.gene_stats(probe, by=("class", facet)), hide_index=True, width="stretch",
                         column_config={"class": "Lớp", "source": "Phòng lab", "tissue": "Loại mô", "n": "n",
                                        "mean": st.column_config.NumberColumn("Trung bình", format="%.2f"),
                                        "median": st.column_config.NumberColumn("Trung vị", format="%.2f"),
                                        "sd": st.column_config.NumberColumn("Độ lệch chuẩn", format="%.2f")})
        with c_dos:
            st.markdown("**Do bệnh hay do nguồn mẫu?**")
            dos = backend.disease_or_source(probe)
            dos["comparable"] = dos["comparable"].map({True: "Có", False: "Không"})
            st.dataframe(dos[["source", "n_ALL", "n_AML", "diff", "comparable"]], hide_index=True, width="stretch",
                         column_config={"source": "Phòng lab", "n_ALL": "n ALL", "n_AML": "n AML",
                                        "diff": st.column_config.NumberColumn("ALL − AML", format="%.2f"),
                                        "comparable": "So sánh được"})
            st.caption("Chỉ so sánh ALL với AML được trong phòng lab có đủ cả hai lớp (mỗi lớp từ 3 mẫu).")

# ---- Bảng xếp hạng gen ----
with tab_rank:
    r1, r2 = st.columns([2, 3])
    dir_label = r1.selectbox("Hướng khác biệt", ["Cả hai", "Cao ở ALL", "Cao ở AML"])
    top_n = r2.slider("Số gen hiển thị", 10, 100, 50, step=10)
    direction = {"Cả hai": "both", "Cao ở ALL": "ALL", "Cao ở AML": "AML"}[dir_label]
    rk = backend.annotate(backend.ranking_table(top_n=top_n, direction=direction))
    note = rk.attrs.get("note")
    if note:
        st.info(note, icon="ℹ️")
    st.dataframe(
        rk[["rank", "probe_id", "gene_symbol", "description", "snr", "q_value", "direction", "function"]],
        hide_index=True, width="stretch", height=480,
        column_config={"rank": "Hạng", "probe_id": "Probe", "gene_symbol": "Gen", "description": "Mô tả",
                       "snr": st.column_config.NumberColumn("SNR", format="%.2f",
                                                            help="Dương: cao ở ALL. Âm: cao ở AML."),
                       "q_value": st.column_config.NumberColumn("q (ý nghĩa)", format="%.2e",
                                                                help="p đã hiệu chỉnh đa kiểm định (BH)."),
                       "direction": "Cao ở", "function": "Chức năng"})
    st.caption("Muốn xem biểu hiện của một gen: chép mã probe vào ô tìm ở thẻ Tra cứu gen.")

# ---- PCA tương tác ----
with tab_pca:
    scores, ratio = backend.pca_all()
    p1, p2, p3 = st.columns(3)
    color_label = p1.selectbox("Tô màu theo", [viz.LABELS[c] for c in ("class", "subtype", "source", "tissue")])
    color_by = {v: k for k, v in viz.LABELS.items()}[color_label]
    px = p2.selectbox("Trục ngang", [1, 2, 3, 4, 5], index=0, format_func=lambda i: f"PC{i}")
    py = p3.selectbox("Trục dọc", [1, 2, 3, 4, 5], index=1, format_func=lambda i: f"PC{i}")
    st.altair_chart(charts.pca_chart(scores, ratio, samples, color_by=color_by, pcs=(px, py)),
                    width="stretch")
    st.caption("Mỗi điểm là một bệnh nhân (72 người); di chuột để xem thông tin mẫu, cuộn để phóng to. "
               "Hình dạng cho biết lớp (tròn: ALL, tam giác: AML). Vòng đen: mẫu gắn cờ qc_outlier.")

# ---- Tương quan giữa các mẫu (F04e, hàm của viz.py) ----
with tab_corr:
    fig = viz.plot_sample_correlation(X, samples, mode="slide")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Ma trận tương quan Pearson giữa 72 mẫu (sau chuẩn hóa từng gen), sắp theo cây phân cụm. "
               "Thấy các nhóm bệnh nhân giống nhau.")
