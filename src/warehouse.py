"""Kho dữ liệu DuckDB theo lược đồ hình sao và truy vấn OLAP.

Công việc: 2.3 — Phụ trách: TV3 — Kế hoạch giai đoạn 2, mục 2.3 (ADR-014, ADR-040).
Dùng cho: TV1, TV2, TV4, TV6.

Giao diện đã thống nhất, KHÔNG đổi tên khi chưa báo nhóm:
    WAREHOUSE_PATH            = data/warehouse/golub.duckdb  (không đưa lên Git)
    build_warehouse()         dựng lại kho từ các tầng Parquet
    connect(read_only=True)   -> kết nối duckdb để truy vấn
Bảng: fact_expression, dim_sample, dim_gene, dim_batch, fact_qc (hai bảng cuối lấy từ TV2).
Câu truy vấn OLAP để trong thư mục sql/.

Lược đồ (ADR-040):
    fact_expression(sample_id, probe_id, value_raw, value_log10)   72 × 7129 = 513 288 dòng
        value_raw   : giá trị MAS4 gốc của tầng cleansed (có âm, chưa threshold)
        value_log10 : giá trị tầng curated (threshold [100, 16000] rồi log10)
    dim_sample(sample_id, split, class, subtype, tissue, source, sex, qc_outlier, batch_id)
    dim_gene(probe_id, description, is_control)
    dim_batch(batch_id, source, n_samples, n_train, n_test, n_all, n_aml, n_bm, n_pb)
        mỗi phòng lab là một lô: batch_id = tên nguồn (DFCI, CALGB, St-Jude, CCG)
    fact_qc  : kết quả ingest_batches() của TV2, một hàng mỗi lô; chỉ có khi TV2 đã cài đặt.

Chạy: make warehouse  (Windows: python run_all.py warehouse)
"""
from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd

try:
    from src.config import DATA_DIR, STD_DIR
    from src.load import load_golub
except ModuleNotFoundError:  # chạy trực tiếp `python src/warehouse.py`
    from config import DATA_DIR, STD_DIR
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


def _dim_batch(samples: pd.DataFrame) -> pd.DataFrame:
    g = samples.groupby("source")
    return pd.DataFrame({
        "batch_id": g.size().index,
        "source": g.size().index,
        "n_samples": g.size().to_numpy(),
        "n_train": g["split"].apply(lambda s: (s == "train").sum()).to_numpy(),
        "n_test": g["split"].apply(lambda s: (s == "test").sum()).to_numpy(),
        "n_all": g["class"].apply(lambda s: (s == "ALL").sum()).to_numpy(),
        "n_aml": g["class"].apply(lambda s: (s == "AML").sum()).to_numpy(),
        "n_bm": g["tissue"].apply(lambda s: (s == "BM").sum()).to_numpy(),
        "n_pb": g["tissue"].apply(lambda s: (s == "PB").sum()).to_numpy(),
    })


def _fact_qc() -> pd.DataFrame | None:
    """Bảng chất lượng theo lô của TV2; None nếu src/ingest.py chưa cài đặt."""
    try:
        from src.ingest import ingest_batches
        return ingest_batches()
    except NotImplementedError:
        return None


def build_warehouse(path=WAREHOUSE_PATH) -> dict:
    """Dựng data/warehouse/golub.duckdb từ các tầng Parquet. Trả về số dòng mỗi bảng.

    Ghi ra file tạm rồi mới thay file cũ, để lỗi giữa chừng không để lại kho dở dang.
    """
    raw, samples = load_golub("cleansed")
    log10, _ = load_golub("curated")
    genes = pd.read_parquet(STD_DIR / "genes.parquet")

    dim_sample = samples.assign(batch_id=samples["source"]).reset_index()
    tables = {
        "fact_expression": _fact_expression(raw, log10.loc[raw.index, raw.columns]),
        "dim_sample": dim_sample,
        "dim_gene": genes[["probe_id", "description", "is_control"]],
        "dim_batch": _dim_batch(samples),
    }
    qc = _fact_qc()
    if qc is not None:
        tables["fact_qc"] = qc
    else:
        print("Bỏ qua fact_qc: src/ingest.py (TV2, 2.4) chưa cài đặt ingest_batches().")

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
