"""Trang chủ dashboard: chọn vai trò người dùng.

Công việc: 2.6 — Phụ trách: TV6 — Kế hoạch giai đoạn 2, mục 2.6 (ADR-014).
Chạy: make app  (Windows: python run_all.py app)

Quy tắc: các trang trong app/ chỉ GỌI hàm của src/ (ingest, gene_explorer, patient, warehouse) qua
app/components/backend.py, không tự tính toán lại. Dữ liệu giả hay thật do biến GOLUB_APP_DATA quyết định.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # để import `src` và `app` khi chạy streamlit run

import streamlit as st  # noqa: E402

from app.components import backend, ui  # noqa: E402

st.set_page_config(page_title="Golub ALL/AML", page_icon="🧬", layout="wide")


def home() -> None:
    st.markdown(ui._CSS, unsafe_allow_html=True)
    st.title("Dữ liệu biểu hiện gen bạch cầu cấp ALL/AML")
    st.write("Trung tâm xét nghiệm nhận mẫu từ 4 phòng lab (DFCI, CALGB, St-Jude, CCG). "
             "Mọi màn hình đọc cùng một kho dữ liệu đã kiểm tra. Chọn màn hình theo vai trò của bạn.")
    if backend.IS_MOCK:
        st.warning("**Đang dùng DỮ LIỆU GIẢ**: đây là khung giao diện. Đặt `GOLUB_APP_DATA=real` để dùng hàm thật "
                   "khi các việc 2.1–2.5 đã bàn giao.", icon="⚠️")

    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.subheader("Kỹ thuật viên phòng lab")
        st.write("**Chất lượng dữ liệu.** Lô nào lỗi, phòng lab nào đo lệch, mẫu nào bất thường; "
                 "tải mẫu mới lên hệ thống.")
        st.page_link("pages/1_Chat_luong_du_lieu.py", label="Mở màn hình Chất lượng dữ liệu", icon="🧪")
    with c2, st.container(border=True):
        st.subheader("Nhà nghiên cứu")
        st.write("**Khám phá gen.** Gen nào khác biệt giữa ALL và AML, và khác biệt đó do bệnh hay do phòng lab.")
        st.page_link("pages/2_Kham_pha_gen.py", label="Mở màn hình Khám phá gen", icon="🔬")
    with c3, st.container(border=True):
        st.subheader("Bác sĩ")
        st.write("**Hồ sơ bệnh nhân.** Bệnh nhân giống nhóm bệnh nào, vì sao, và kết luận đáng tin đến đâu.")
        st.page_link("pages/3_Ho_so_benh_nhan.py", label="Mở màn hình Hồ sơ bệnh nhân", icon="🩺")

    st.caption("Mô hình gợi ý chẩn đoán chỉ xuất hiện ở màn hình Hồ sơ bệnh nhân. " + ui.DISCLAIMER)


pages = [
    st.Page(home, title="Trang chủ", icon="🏠", url_path="trang-chu", default=True),
    st.Page("pages/1_Chat_luong_du_lieu.py", title="Chất lượng dữ liệu", icon="🧪", url_path="chat-luong"),
    st.Page("pages/2_Kham_pha_gen.py", title="Khám phá gen", icon="🔬", url_path="kham-pha-gen"),
    st.Page("pages/3_Ho_so_benh_nhan.py", title="Hồ sơ bệnh nhân", icon="🩺", url_path="ho-so-benh-nhan"),
]
st.navigation(pages).run()
