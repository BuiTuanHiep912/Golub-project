"""Kho dữ liệu DuckDB theo lược đồ hình sao và truy vấn OLAP.

Công việc: 2.3 — Phụ trách: TV3 — Kế hoạch giai đoạn 2, mục 2.3 (ADR-014).
Dùng cho: TV1, TV2, TV4, TV6.

Giao diện đã thống nhất, KHÔNG đổi tên khi chưa báo nhóm:
    WAREHOUSE_PATH            = data/warehouse/golub.duckdb  (không đưa lên Git)
    build_warehouse()         dựng lại kho từ các tầng Parquet
    connect(read_only=True)   -> kết nối duckdb để truy vấn
Bảng: fact_expression, dim_sample, dim_gene, dim_batch, fact_qc (hai bảng cuối lấy từ TV2).
Câu truy vấn OLAP để trong thư mục sql/.

Chạy: make warehouse  (Windows: python run_all.py warehouse)
"""
from __future__ import annotations

try:
    from src.config import DATA_DIR
except ModuleNotFoundError:
    from config import DATA_DIR

WAREHOUSE_PATH = DATA_DIR / "warehouse" / "golub.duckdb"
TABLES = ("fact_expression", "dim_sample", "dim_gene", "dim_batch", "fact_qc")


def build_warehouse():
    """Dựng data/warehouse/golub.duckdb từ các tầng Parquet."""
    raise NotImplementedError("TV3 - công việc 2.3")


def connect(read_only: bool = True):
    """Mở kết nối tới kho; báo lỗi rõ ràng nếu kho chưa được dựng."""
    raise NotImplementedError("TV3 - công việc 2.3")


if __name__ == "__main__":
    print("Chưa cài đặt: TV3 - công việc 2.3")
