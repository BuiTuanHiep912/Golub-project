"""Thành phần giao diện dùng chung của 3 màn hình (công việc 2.6, TV6)."""
from __future__ import annotations

import streamlit as st

from app.components import backend
from src import viz

DISCLAIMER = "Công cụ hỗ trợ nghiên cứu, không thay thế chẩn đoán lâm sàng."

_CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
.role-tag {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.85rem;
           border:1px solid #BDBDBD; color:#4D4D4D;}
.dx-badge {display:inline-block; padding:6px 18px; border-radius:8px; font-size:1.6rem; font-weight:700;
           border:2px solid #000; color:#000;}
</style>
"""


def page_header(title: str, role: str, question: str) -> None:
    """Tiêu đề trang + vai trò người dùng + câu hỏi màn hình trả lời; cảnh báo khi đang dùng dữ liệu giả."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(f'<span class="role-tag">{role}</span>', unsafe_allow_html=True)
    st.title(title)
    st.caption(question)
    if backend.IS_MOCK:
        st.warning("**Đang dùng DỮ LIỆU GIẢ** (khung giao diện, chờ hàm thật của TV1–TV5). "
                   "Số liệu trên màn hình này không phải kết quả thật.", icon="⚠️")


def page_link(path: str, label: str, icon: str | None = None) -> None:
    """st.page_link nhưng không làm vỡ trang khi chạy riêng lẻ (không qua Home.py), ví dụ trong kiểm thử."""
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:  # noqa: BLE001 - trang chưa đăng ký trong st.navigation
        st.caption(f"→ {label}")


def class_badge(label: str) -> str:
    """Nhãn chẩn đoán dạng HTML: màu Okabe–Ito kèm ký hiệu hình (màu không là tín hiệu duy nhất)."""
    if label in viz.CLASS_COLORS:
        mark = "●" if label == "ALL" else "▲"
        bg = viz.CLASS_COLORS[label]
        fg = "#FFFFFF" if label == "ALL" else "#000000"
        return f'<span class="dx-badge" style="background:{bg}; color:{fg}">{mark} {label}</span>'
    return f'<span class="dx-badge" style="background:{viz.NEUTRAL_LIGHT}">? Không chắc chắn</span>'


def fmt_pct(x: float, digits: int = 1) -> str:
    """Số thập phân kiểu Việt Nam: 24,7%."""
    return f"{x * 100:.{digits}f}%".replace(".", ",")


def fmt_num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")
