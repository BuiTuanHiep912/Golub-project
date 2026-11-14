"""Kiểm tra kho DuckDB (TV3, công việc 2.3): dựng vào thư mục tạm từ dữ liệu thật."""
from __future__ import annotations

import os
import sys

import duckdb
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import warehouse  # noqa: E402
from src.config import CURATED_DIR, PROJECT_ROOT  # noqa: E402

pytestmark = pytest.mark.skipif(not (CURATED_DIR / "X_log10.parquet").exists(),
                                reason="Chưa có tầng curated — chạy `make data` trước")

# Bảng 2.4 của kế hoạch: (split, class, source) -> số mẫu
TABLE_2_4 = {
    ("train", "ALL", "DFCI"): 27, ("train", "AML", "CALGB"): 11,
    ("test", "ALL", "DFCI"): 17, ("test", "ALL", "St-Jude"): 3,
    ("test", "AML", "CALGB"): 4, ("test", "AML", "CCG"): 5, ("test", "AML", "St-Jude"): 5,
}


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    path = tmp_path_factory.mktemp("wh") / "golub.duckdb"
    counts = warehouse.build_warehouse(path)
    assert counts["fact_expression"] == 72 * 7129
    with duckdb.connect(str(path), read_only=True) as c:
        yield c


def test_counts_match_table_2_4(con):
    sql = (PROJECT_ROOT / "sql" / "01_mau_theo_lop_nguon.sql").read_text(encoding="utf-8")
    got = {(s, c, src): n for s, c, src, n in con.execute(sql).fetchall()}
    assert got == TABLE_2_4


def test_dimensions_and_keys(con):
    one = lambda q: con.execute(q).fetchone()[0]  # noqa: E731
    assert one("SELECT count(*) FROM dim_sample") == 72
    assert one("SELECT count(*) FROM dim_gene") == 7129
    assert one("SELECT count(*) FROM dim_batch") == 4
    assert one("SELECT sum(n_samples) FROM dim_batch") == 72
    # mọi dòng sự kiện đều nối được với bảng chiều
    assert one("SELECT count(*) FROM fact_expression f ANTI JOIN dim_sample USING (sample_id)") == 0
    assert one("SELECT count(*) FROM fact_expression f ANTI JOIN dim_gene USING (probe_id)") == 0
    assert one("SELECT count(*) FROM dim_sample ANTI JOIN dim_batch USING (batch_id)") == 0
    assert one("SELECT count(DISTINCT (sample_id, probe_id)) FROM fact_expression") == 72 * 7129


def test_values_match_layers(con):
    lo, hi, n_neg = con.execute(
        "SELECT min(value_log10), max(value_log10), count(*) FILTER (value_raw < 0) "
        "FROM fact_expression").fetchone()
    assert lo == 2.0 and abs(hi - 4.2041) < 1e-3
    assert n_neg == 147_174  # số giá trị âm của tầng cleansed (handover 1.1)


def test_connect_missing_warehouse(tmp_path, monkeypatch):
    monkeypatch.setattr(warehouse, "WAREHOUSE_PATH", tmp_path / "none.duckdb")
    with pytest.raises(FileNotFoundError, match="make warehouse"):
        warehouse.connect()
