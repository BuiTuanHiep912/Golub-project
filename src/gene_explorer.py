"""Tra cứu gen và thống kê biểu hiện theo bệnh, nguồn mẫu, loại mô.

Công việc: 2.2 — Phụ trách: TV4 — Kế hoạch giai đoạn 2, mục 2.2 (ADR-014).
Dùng cho: TV6 (màn hình Khám phá gen).

Giao diện đã thống nhất, KHÔNG đổi tên khi chưa báo nhóm:
    find_gene('CD33')        -> DataFrame [probe_id, description]
    gene_stats(probe_id)     -> DataFrame biểu hiện của gen theo class, source, tissue
Hàm vẽ mới của TV4 viết trong src/viz_gene.py (không sửa src/viz.py, ADR-014).
"""
from __future__ import annotations


def find_gene(query: str):
    """Tìm probe theo tên gen hoặc mã probe."""
    raise NotImplementedError("TV4 - công việc 2.2")


def gene_stats(probe_id: str):
    """Thống kê biểu hiện của một probe theo bệnh, nguồn mẫu và loại mô."""
    raise NotImplementedError("TV4 - công việc 2.2")


if __name__ == "__main__":
    print("Chưa cài đặt: TV4 - công việc 2.2")
