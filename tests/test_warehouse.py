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


def test_batch_and_qc_tables(con):
    one = lambda q: con.execute(q).fetchone()[0]  # noqa: E731
    assert sorted(r[0] for r in con.execute("SELECT batch_id FROM dim_batch").fetchall()) == [
        "B1-DFCI", "B2-CALGB", "B3-St-Jude", "B4-CCG"]
    assert one("SELECT count(*) FROM dim_batch WHERE status <> 'passed'") == 0
    assert one("SELECT count(*) FROM fact_qc") == 72
    assert one("SELECT count(*) FROM fact_qc ANTI JOIN dim_sample USING (sample_id, batch_id)") == 0
    # cờ qc_outlier của TV2 trùng cờ của tầng cleansed (chỉ mẫu 21)
    assert con.execute("SELECT sample_id FROM fact_qc WHERE qc_outlier").fetchall() == [(21,)]
    assert one("SELECT count(*) FROM dim_sample s JOIN fact_qc q USING (sample_id) "
               "WHERE s.qc_outlier <> q.qc_outlier") == 0
    sql = (PROJECT_ROOT / "sql" / "02_chat_luong_theo_lo.sql").read_text(encoding="utf-8")
    assert [r[2] for r in con.execute(sql).fetchall()] == [44, 15, 8, 5]


def test_blocked_batch_not_loaded(tmp_path):
    """Lô bị chặn vẫn có trong dim_batch/fact_qc nhưng mẫu của nó không vào dim_sample, fact_expression."""
    from src.ingest import ingest_batches, sample_quality
    batches, qc = ingest_batches(), sample_quality()
    batches.loc[batches["batch_id"] == "B4-CCG", ["status", "reason"]] = ["blocked", "lỗi giả lập"]
    qc.loc[qc["batch_id"] == "B4-CCG", "batch_status"] = "blocked"
    batches.to_parquet(tmp_path / "batches.parquet", index=False)
    qc.to_parquet(tmp_path / "qc_samples.parquet", index=False)

    path = tmp_path / "golub.duckdb"
    counts = warehouse.build_warehouse(path, cleansed_dir=tmp_path)
    assert counts["dim_sample"] == 67 and counts["fact_expression"] == 67 * 7129
    assert counts["dim_batch"] == 4 and counts["fact_qc"] == 72
    with duckdb.connect(str(path), read_only=True) as c:
        assert c.execute("SELECT count(*) FROM dim_sample WHERE source = 'CCG'").fetchone()[0] == 0


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
