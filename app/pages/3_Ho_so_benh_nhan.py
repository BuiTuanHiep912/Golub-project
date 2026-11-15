"""Màn hình Hồ sơ bệnh nhân (bác sĩ).

Giao diện: TV6 (2.6). Dữ liệu: src/patient.py của TV1 (2.1), validate_new_sample của TV2 (2.4) — qua backend.py.
Nội dung theo kế hoạch mục D.3: thông tin mẫu, gợi ý chẩn đoán kèm độ tin cậy, lý do, vị trí trên bản đồ PCA,
5 bệnh nhân tương tự, lưu ý sử dụng.
Quy ước: 38 bệnh nhân huấn luyện là dữ liệu tham chiếu; 34 bệnh nhân kiểm tra đóng vai bệnh nhân mới.
Chẩn đoán thật ẩn mặc định, chỉ hiện khi bấm "Hiện chẩn đoán thật".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.components import backend, charts, ui  # noqa: E402

ui.page_header("Hồ sơ bệnh nhân", "Bác sĩ",
               "Bệnh nhân này giống nhóm bệnh nào, vì sao, và kết luận đáng tin đến đâu?")

X, samples = backend.reference_data()
ref_ids = samples.index[samples["split"] == "train"]          # 38 bệnh nhân tham chiếu
X_ref, S_ref = X.loc[ref_ids], samples.loc[ref_ids]
test_ids = [int(i) for i in samples.index[samples["split"] == "test"]]

# ---- Chọn bệnh nhân ----
new = st.session_state.get("new_sample")
options = ["Bệnh nhân kiểm tra có sẵn"] + (["Mẫu mới đã tải lên"] if new else [])
c_src, c_pick = st.columns([2, 3])
mode = c_src.radio("Nguồn bệnh nhân", options, horizontal=True)
sample_id, values, key = None, None, ""
if mode == options[0]:
    sample_id = c_pick.selectbox("Mã bệnh nhân (34 người, đóng vai bệnh nhân mới)", test_ids,
                                 index=test_ids.index(52) if 52 in test_ids else 0)
    key = f"id{sample_id}"
else:
    values = new["values"]
    c_pick.write(f"File: **{new['name']}**")
    key = f"new:{new['name']}"
if not new:
    st.caption("Muốn xem hồ sơ của một mẫu mới: tải file ở màn hình Chất lượng dữ liệu, mẫu hợp lệ sẽ xuất hiện ở đây.")

profile = backend.patient_profile(sample_id=sample_id) if sample_id is not None else backend.patient_profile(values=values)
pred, ps = profile["prediction"], float(profile["ps"])
x_pat = X.loc[sample_id] if sample_id is not None else values

# ---- Thông tin mẫu và cảnh báo ----
st.subheader("Thông tin mẫu")
i1, i2, i3 = st.columns(3)
if sample_id is not None:
    meta = samples.loc[sample_id]
    i1.metric("Nguồn", meta["source"])
    i2.metric("Loại mô", {"BM": "Tủy xương (BM)", "PB": "Máu ngoại vi (PB)"}[meta["tissue"]])
    i3.metric("Kiểm tra chất lượng", "Gắn cờ" if meta["qc_outlier"] else "Đạt")
    if meta["qc_outlier"]:
        st.warning(f"Mẫu này bị gắn cờ chất lượng: {meta['qc_reason']}. Độ tin cậy có thể giảm.", icon="⚠️")
    if meta["source"] == "St-Jude":
        st.warning("Mẫu từ St-Jude, phòng lab có quy trình xử lý khác dữ liệu tham chiếu. Độ tin cậy có thể giảm.",
                   icon="⚠️")
    if meta["tissue"] == "PB":
        st.warning("Mẫu máu ngoại vi; dữ liệu tham chiếu chủ yếu là tủy xương. Độ tin cậy có thể giảm.", icon="⚠️")
else:
    i1.metric("Nguồn", "Chưa rõ")
    i2.metric("Loại mô", "Chưa rõ")
    i3.metric("Kiểm tra chất lượng", "Đạt" if not new["warnings"] else "Có cảnh báo")
    for msg in new["warnings"]:
        st.warning(msg, icon="⚠️")

# ---- Gợi ý chẩn đoán ----
st.subheader("Gợi ý chẩn đoán")
with st.container(border=True):
    d1, d2 = st.columns([1, 2])
    with d1:
        st.markdown(ui.class_badge(pred), unsafe_allow_html=True)
    with d2:
        st.metric("Độ tin cậy (PS)", ui.fmt_num(ps), help="Prediction strength của Golub weighted voting, từ 0 đến 1.")
        st.progress(min(max(ps, 0.0), 1.0))
        st.caption(f"Dưới ngưỡng {ui.fmt_num(backend.PS_THRESHOLD, 1)} thì hệ thống không kết luận.")
    if pred == "uncertain":
        st.warning("**Không chắc chắn, cần xét nghiệm thêm.**", icon="⚠️")
    else:
        st.write(f"Gợi ý **{pred}** (độ tin cậy {ui.fmt_num(ps)}).")
    # Chẩn đoán thật: ẩn mặc định
    if sample_id is not None:
        if st.button("Hiện chẩn đoán thật", key=f"reveal_btn_{key}"):
            st.session_state["revealed"] = key
        if st.session_state.get("revealed") == key:
            truth = samples.loc[sample_id, "class"]
            verdict = "trùng với gợi ý" if truth == pred else ("hệ thống không kết luận" if pred == "uncertain"
                                                               else "KHÁC với gợi ý")
            st.info(f"Chẩn đoán thật: **{truth}** ({verdict}).", icon="🔓")
    else:
        st.caption("Mẫu mới tải lên không có chẩn đoán thật để đối chiếu.")

# ---- Lý do ----
st.subheader("Lý do")
st.caption("Các gen quan trọng nhất: mức biểu hiện của bệnh nhân (vạch đen) đặt cạnh phân bố của nhóm ALL và nhóm AML; "
           "mỗi gen “bỏ phiếu” cho một bệnh.")
st.altair_chart(charts.patient_vs_groups(profile["votes"], X_ref, S_ref, top=6), width="content")
st.markdown("**Các gen bỏ phiếu mạnh nhất**")
st.altair_chart(charts.votes_chart(profile["votes"], top=10), width="stretch")

# ---- Vị trí trên bản đồ ----
st.subheader("Vị trí trên bản đồ")
sc_ref, ratio, xy = charts.project_patient(X_ref, x_pat)
st.altair_chart(charts.pca_chart(sc_ref, ratio, S_ref, color_by="class", patient_xy=xy), width="stretch")
st.caption("PCA của 38 bệnh nhân tham chiếu; hình thoi đen là bệnh nhân này. Nằm giữa một nhóm thì kết luận chắc hơn; "
           "ở vùng lẫn lộn thì nên thận trọng.")

# ---- Bệnh nhân tương tự ----
st.subheader("5 bệnh nhân tương tự nhất")
sim = profile["similar"]
st.dataframe(sim, hide_index=True, width="stretch",
             column_config={"sample_id": "Mã bệnh nhân", "correlation": st.column_config.NumberColumn(
                 "Tương quan", format="%.2f"), "class": "Chẩn đoán"})

# ---- Lưu ý sử dụng ----
st.info("**Lưu ý sử dụng.** Công cụ hỗ trợ nghiên cứu, không thay thế chẩn đoán lâm sàng. "
        "Dữ liệu tham chiếu chỉ có ALL trẻ em.", icon="ℹ️")
