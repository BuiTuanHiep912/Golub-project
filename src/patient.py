"""Hồ sơ một bệnh nhân: gợi ý chẩn đoán, độ tin cậy, gen bỏ phiếu, bệnh nhân tương tự.

Công việc: 2.1 — Phụ trách: TV1 — Kế hoạch giai đoạn 2, mục 2.1 (ADR-014).
Dùng cho: TV6 (màn hình Hồ sơ bệnh nhân).

Quy ước: dữ liệu tham chiếu = 38 bệnh nhân tập huấn luyện. 34 bệnh nhân tập kiểm tra
đóng vai bệnh nhân mới; không dùng chúng để huấn luyện, chọn gen hay tính tham số nào.

Giao diện đã thống nhất, KHÔNG đổi tên khi chưa báo nhóm:
    patient_profile(sample_id=52)      # một bệnh nhân có sẵn trong dữ liệu
    patient_profile(values=series)     # mẫu mới, đã qua validate_new_sample() của TV2
    -> dict với các khóa:
       'prediction'  : 'ALL' | 'AML' | 'uncertain'
       'ps'          : float, độ tin cậy
       'votes'       : DataFrame [probe_id, description, value, vote, toward]
       'similar'     : DataFrame [sample_id, correlation, class]  (5 người giống nhất)
"""
from __future__ import annotations

N_SIMILAR = 5


def patient_profile(sample_id: int | None = None, values=None) -> dict:
    """Trả về hồ sơ của một bệnh nhân có sẵn (sample_id) hoặc một mẫu mới (values)."""
    raise NotImplementedError("TV1 - công việc 2.1")


if __name__ == "__main__":
    print("Chưa cài đặt: TV1 - công việc 2.1")
