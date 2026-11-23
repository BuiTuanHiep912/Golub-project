"""Kho dữ liệu DuckDB theo lược đồ hình sao và truy vấn OLAP.

Công việc: 2.3 — Phụ trách: TV3 — Kế hoạch giai đoạn 2, mục 2.3 (ADR-014, ADR-040, ADR-041).
Dùng cho: TV1, TV2, TV4, TV6.

Giao diện đã thống nhất, KHÔNG đổi tên khi chưa báo nhóm:
    WAREHOUSE_PATH            = data/warehouse/golub.duckdb  (không đưa lên Git)
    build_warehouse()         dựng lại kho từ các tầng Parquet
    connect(read_only=True)   -> kết nối duckdb để truy vấn
Bảng: fact_expression, dim_sample, dim_gene, dim_batch, fact_qc (hai bảng cuối lấy từ TV2).
Câu truy vấn OLAP để trong thư mục sql/.

Lược đồ (ADR-040, sửa theo ADR-041 để khớp src/ingest.py của TV2):
    fact_expression(sample_id, probe_id, value_raw, value_log10)   72 × 7129 = 513 288 dòng
        value_raw   : giá trị MAS4 gốc của tầng cleansed (có âm, chưa threshold)
        value_log10 : giá trị tầng curated (threshold [100, 16000] rồi log10)
    dim_sample(sample_id, split, class, subtype, tissue, source, sex, qc_outlier, batch_id)
    dim_gene(probe_id, description, is_control)
    dim_batch : batches.parquet của TV2, một hàng mỗi lô (batch_id = B1-DFCI, B2-CALGB, ...),
                có status 'passed' | 'blocked' và lý do
    fact_qc   : qc_samples.parquet của TV2, một hàng mỗi mẫu (chỉ số chất lượng, qc_outlier)
    log_ingest: ingest_log.csv của TV2, mỗi lần `make ingest` thêm một dòng mỗi lô (nếu có file)
Chỉ mẫu thuộc lô 'passed' mới vào dim_sample và fact_expression; lô bị chặn vẫn nằm trong
dim_batch/fact_qc để màn hình Chất lượng dữ liệu hiện được lý do.

Chạy: make ingest rồi make warehouse  (Windows: python run_all.py ingest warehouse)
"""
from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd

try:
    from src.config import CLEANSED_DIR, DATA_DIR, STD_DIR
    from src.ingest import BATCHES_FILE, LOG_FILE, QC_FILE, ingest_batches, sample_quality
    from src.load import load_golub
except ModuleNotFoundError:  # chạy trực tiếp `python src/warehouse.py`
    from config import CLEANSED_DIR, DATA_DIR, STD_DIR
    from ingest import BATCHES_FILE, LOG_FILE, QC_FILE, ingest_batches, sample_quality
    from load import load_golub

WAREHOUSE_PATH = DATA_DIR / "warehouse" / "golub.duckdb"
TABLES = ("fact_expression", "dim_sample", "dim_gene", "dim_batch", "fact_qc")


def _fact_expression(raw: pd.DataFrame, log10: pd.DataFrame) -> pd.DataFrame:
    """Ma trận rộng (mẫu × probe) -> bảng dài, một dòng mỗi cặp (mẫu, probe)."""
    if not (raw.index.equals(log10.index) and raw.columns.equals(log10.columns)):
        raise ValueError("Tầng cleansed và curated không cùng mẫu/probe")
    n, p = raw.shape
    return pd.DataFrame({
        "sample_id": np.repeat(raw.index.to_numpy(), p),
        "probe_id": np.tile(raw.columns.to_numpy(dtype=object), n),
        "value_raw": raw.to_numpy(dtype=float).ravel(),
        "value_log10": log10.to_numpy(dtype=float).ravel(),
    })


def _batch_tables(cleansed_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str]:
    """(dim_batch, fact_qc, log_ingest, nguồn) — đọc đầu ra `make ingest`; chưa có thì tính tại chỗ."""
    b, q, log = (cleansed_dir / f for f in (BATCHES_FILE, QC_FILE, LOG_FILE))
    if b.exists() and q.exists():
        return (pd.read_parquet(b), pd.read_parquet(q),
                pd.read_csv(log) if log.exists() else None, f"{b.name}, {q.name}")
    return ingest_batches(), sample_quality(), None, "ingest_batches() (chưa chạy make ingest)"


def build_warehouse(path=WAREHOUSE_PATH, cleansed_dir=CLEANSED_DIR) -> dict:
    """Dựng data/warehouse/golub.duckdb từ các tầng Parquet. Trả về số dòng mỗi bảng.

    Ghi ra file tạm rồi mới thay file cũ, để lỗi giữa chừng không để lại kho dở dang.
    """
    raw, samples = load_golub("cleansed")
    log10, _ = load_golub("curated")
    genes = pd.read_parquet(STD_DIR / "genes.parquet")
    dim_batch, fact_qc, log_ingest, origin = _batch_tables(cleansed_dir)

    # Lô của mỗi mẫu lấy từ fact_qc; chỉ giữ mẫu thuộc lô đã qua kiểm tra
    passed = fact_qc.loc[fact_qc["batch_status"] == "passed", ["sample_id", "batch_id"]]
    keep = samples.index.intersection(passed["sample_id"])
    dropped = samples.index.difference(keep)
    if len(dropped):
        print(f"Không nạp {len(dropped)} mẫu thuộc lô bị chặn hoặc không có trong fact_qc: "
              f"{list(dropped)[:10]}")
    dim_sample = samples.loc[keep].join(passed.set_index("sample_id")["batch_id"]).reset_index()

    tables = {
        "fact_expression": _fact_expression(raw.loc[keep], log10.loc[keep, raw.columns]),
        "dim_sample": dim_sample,
        "dim_gene": genes[["probe_id", "description", "is_control"]],
        "dim_batch": dim_batch,
        "fact_qc": fact_qc,
    }
    if log_ingest is not None:
        tables["log_ingest"] = log_ingest

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.duckdb")
    tmp.unlink(missing_ok=True)
    with duckdb.connect(str(tmp)) as con:
        for name, df in tables.items():
            con.register("df", df)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
            con.unregister("df")
        con.execute("ALTER TABLE dim_sample ADD PRIMARY KEY (sample_id)")
        con.execute("ALTER TABLE dim_gene ADD PRIMARY KEY (probe_id)")
        con.execute("ALTER TABLE dim_batch ADD PRIMARY KEY (batch_id)")
        counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
    os.replace(tmp, path)
    print(f"dim_batch, fact_qc lấy từ: {origin}")
    return counts


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Mở kết nối tới kho; báo lỗi rõ ràng nếu kho chưa được dựng."""
    if not WAREHOUSE_PATH.exists():
        raise FileNotFoundError(f"Chưa có {WAREHOUSE_PATH} — chạy `make warehouse` "
                                "(Windows: python run_all.py warehouse) trước.")
    return duckdb.connect(str(WAREHOUSE_PATH), read_only=read_only)


if __name__ == "__main__":
    for table, n in build_warehouse().items():
        print(f"{table:<16} {n:>8,} dòng")
    print(f"Đã ghi {WAREHOUSE_PATH}")
