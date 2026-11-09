"""Trang chủ dashboard: chọn vai trò người dùng.

Công việc: 2.6 — Phụ trách: TV6 — Kế hoạch giai đoạn 2, mục 2.6 (ADR-014).
Chạy: make app  (Windows: python run_all.py app)

Quy tắc: các trang trong app/ chỉ GỌI hàm của src/ (ingest, gene_explorer, patient,
warehouse), không tự tính toán lại.
"""
import streamlit as st

st.set_page_config(page_title="Golub ALL/AML", layout="wide")
st.title("Dữ liệu biểu hiện gen bạch cầu cấp ALL/AML")
st.write("Chọn màn hình theo vai trò ở thanh bên trái.")
st.markdown(
    "- **Kỹ thuật viên phòng lab:** Chất lượng dữ liệu\n"
    "- **Nhà nghiên cứu:** Khám phá gen\n"
    "- **Bác sĩ:** Hồ sơ bệnh nhân"
)
st.caption("Công cụ hỗ trợ nghiên cứu, không thay thế chẩn đoán lâm sàng.")
