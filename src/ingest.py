"""Nhập dữ liệu theo lô (mỗi phòng lab một lô), chất lượng theo lô, kiểm tra file mẫu mới.

Công việc: 2.4 — Phụ trách: TV2 — Kế hoạch giai đoạn 2, mục 2.4 (ADR-014).
Dùng cho: TV3 (bảng fact_qc, dim_batch trong kho), TV6 (màn hình Chất lượng dữ liệu,
nút tải mẫu), TV1 (mẫu mới đã kiểm tra cho patient_profile).

Giao diện đã thống nhất, KHÔNG đổi tên khi chưa báo nhóm:
    ingest_batches()          -> DataFrame chất lượng theo lô, một hàng mỗi lô:
                                 [batch_id, source, n_samples, status ('passed'|'blocked'),
                                  reason, ...chỉ số chất lượng...]
    validate_new_sample(path) -> dict {'ok': bool, 'errors': list[str],
                                       'warnings': list[str], 'values': Series | None}

Chạy: make ingest  (Windows: python run_all.py ingest)
"""
from __future__ import annotations

BATCH_SOURCES = ("DFCI", "CALGB", "St-Jude", "CCG")


def ingest_batches():
    """Chia dữ liệu thành 4 lô theo nguồn, kiểm tra từng lô, chặn lô lỗi, ghi nhật ký."""
    raise NotImplementedError("TV2 - công việc 2.4")


def validate_new_sample(path) -> dict:
    """Kiểm tra file kết quả chip của một mẫu mới: đủ 7129 gen, đúng mã gen, không ô trống."""
    raise NotImplementedError("TV2 - công việc 2.4")


if __name__ == "__main__":
    print("Chưa cài đặt: TV2 - công việc 2.4")
