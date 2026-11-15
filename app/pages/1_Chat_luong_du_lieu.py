"""Màn hình Chất lượng dữ liệu (kỹ thuật viên phòng lab).

Giao diện: TV6 (2.6). Dữ liệu: src/ingest.py của TV2 (2.4), kho của TV3 (2.3) — qua app/components/backend.py.
Nội dung theo kế hoạch mục D.1: bộ lọc, ô số liệu tổng quan, boxplot từng mẫu (F3), chỉ số theo phòng lab,
danh sách cảnh báo, tải mẫu mới.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import backend, charts, ui  # noqa: E402
from src import viz  # noqa: E402

ui.page_header("Chất lượng dữ liệu", "Kỹ thuật viên phòng lab",
               "Lô nào lỗi? Phòng lab nào đo lệch? Mẫu nào bất thường? Mẫu mới có hợp lệ không?")

X, samples = backend.reference_data()
batches = backend.batches()
warns = backend.warnings_table()

# ---- Bộ lọc: xem riêng từng nguồn ----
f1, f2 = st.columns([3, 2])
labs = f1.multiselect("Phòng lab", list(backend.BATCH_SOURCES), default=list(backend.BATCH_SOURCES))
split_label = f2.radio("Tập dữ liệu", ["Tất cả", "Huấn luyện", "Kiểm tra"], horizontal=True)
split_val = {"Huấn luyện": "train", "Kiểm tra": "test"}.get(split_label)

mask = samples["source"].isin(labs)
if split_val:
    mask &= samples["split"] == split_val
sel = samples[mask]
sel_batches = batches[batches["source"].isin(labs)]

# ---- Ô số liệu tổng quan ----
k1, k2, k3, k4 = st.columns(4)
k1.metric("Tổng số mẫu", len(sel))
n_pass = int((sel_batches["status"] == "passed").sum())
k2.metric("Lô đạt kiểm tra", f"{n_pass}/{len(sel_batches)}")
k3.metric("Mẫu bị gắn cờ", int(sel["qc_outlier"].sum()))
if len(sel_batches):
    w = sel_batches["n_samples"]
    k4.metric("Tỷ lệ giá trị âm", ui.fmt_pct(float((sel_batches["pct_negative"] * w).sum() / w.sum())))
else:
    k4.metric("Tỷ lệ giá trị âm", "—")
st.caption("Chỉ số theo lô (lô đạt, giá trị âm) tính trên toàn bộ lô của các phòng lab đã chọn; "
           "số mẫu và mẫu bị gắn cờ tính theo cả bộ lọc tập dữ liệu.")

n_warn = len(warns[warns["source"].isin(labs)]) + int((sel_batches["status"] == "blocked").sum())
tab_box, tab_lab, tab_warn, tab_up = st.tabs(
    ["Phân bố từng mẫu", "Chỉ số theo phòng lab", f"Cảnh báo ({n_warn})", "Tải mẫu mới"])

# ---- Boxplot từng mẫu (F3, hàm của viz.py) ----
with tab_box:
    if sel.empty:
        st.info("Không có mẫu nào khớp bộ lọc.")
    else:
        fig = viz.plot_box_per_sample(X.loc[sel.index], sel, mode="slide")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Mỗi hộp là một chip (7129 gen). Màu theo phòng lab, gạch chéo là AML. "
                   "Hộp lệch hẳn so với cùng nguồn là dấu hiệu chip hỏng hoặc phòng lab đo lệch.")

# ---- Chỉ số theo phòng lab ----
with tab_lab:
    if sel_batches.empty:
        st.info("Chọn ít nhất một phòng lab.")
    else:
        st.altair_chart(charts.batch_metric_dots(sel_batches), width="content")
        show = sel_batches.assign(
            status=sel_batches["status"].map({"passed": "✅ Đạt", "blocked": "⛔ Bị chặn"}))[
            ["source", "n_samples", "status", "reason", "median_log10", "iqr_log10", "mean_corr",
             "pct_negative", "n_flagged"]]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "source": "Phòng lab", "n_samples": "Số mẫu", "status": "Trạng thái", "reason": "Lý do",
            "median_log10": st.column_config.NumberColumn("Trung vị (log10)", format="%.2f"),
            "iqr_log10": st.column_config.NumberColumn("Độ trải IQR", format="%.2f"),
            "mean_corr": st.column_config.NumberColumn("Tương quan TB", format="%.2f"),
            "pct_negative": st.column_config.NumberColumn("Giá trị âm", format="percent"),
            "n_flagged": "Mẫu gắn cờ"})

# ---- Danh sách cảnh báo ----
with tab_warn:
    blocked = sel_batches[sel_batches["status"] == "blocked"]
    flagged = warns[warns["source"].isin(labs)]
    if blocked.empty and flagged.empty:
        st.success("Không có cảnh báo nào cho các phòng lab đã chọn.")
    for _, r in blocked.iterrows():
        st.error(f"**Lô {r['batch_id']} bị chặn.** {r['reason']}", icon="⛔")
    for _, r in flagged.iterrows():
        st.warning(f"**Mẫu {r['sample_id']}** ({r['source']}): {r['reason']}. Nên kiểm tra lại chip này.", icon="⚠️")

# ---- Tải mẫu mới ----
with tab_up:
    st.write("Tải file kết quả chip của một mẫu mới. Hệ thống kiểm tra định dạng (đủ 7129 gen, đúng mã gen, "
             "không ô trống) và chất lượng trước khi cho vào.")
    up = st.file_uploader("File kết quả chip (CSV/TSV: cột probe_id, value)", type=["csv", "tsv", "txt"])
    if up is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / Path(up.name).name
            path.write_bytes(up.getvalue())
            result = backend.validate_new_sample(path)
        if result["ok"]:
            st.success(f"**{up.name}** hợp lệ, đã đưa vào hệ thống.", icon="✅")
            for msg in result["warnings"]:
                st.warning(msg, icon="⚠️")
            st.session_state["new_sample"] = {"name": up.name, "values": result["values"],
                                              "warnings": result["warnings"]}
            ui.page_link("pages/3_Ho_so_benh_nhan.py", "Xem hồ sơ bệnh nhân của mẫu này", "🩺")
        else:
            st.error(f"**{up.name}** bị chặn. Lý do:", icon="⛔")
            for msg in result["errors"]:
                st.markdown(f"- {msg}")
            st.session_state.pop("new_sample", None)
    if backend.IS_MOCK and backend.make_sample_file is not None:
        with st.expander("File thử (dữ liệu giả)"):
            st.caption("Dùng để thử nút tải lên khi chưa có file thật của TV2.")
            d1, d2, d3 = st.columns(3)
            d1.download_button("File hợp lệ", backend.make_sample_file("valid"), "mau_hop_le.csv", "text/csv")
            d2.download_button("File thiếu gen", backend.make_sample_file("missing_genes"), "mau_thieu_gen.csv",
                               "text/csv")
            d3.download_button("File có ô trống", backend.make_sample_file("blank"), "mau_o_trong.csv", "text/csv")
