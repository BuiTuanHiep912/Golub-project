"""Kiểm tra nhanh dashboard (TV6, công việc 2.6) bằng dữ liệu giả: 3 màn hình mở được, không lỗi.

Không cần dữ liệu thật hay hàm của TV1–TV5. Khi chuyển sang dữ liệu thật (GOLUB_APP_DATA=real), test
định dạng ở dưới vẫn áp dụng cho hàm thật.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GOLUB_APP_DATA", "mock")

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.components import mock_data as m  # noqa: E402

PAGES = {
    "chat_luong": "app/pages/1_Chat_luong_du_lieu.py",
    "kham_pha_gen": "app/pages/2_Kham_pha_gen.py",
    "ho_so": "app/pages/3_Ho_so_benh_nhan.py",
}


def _run(rel: str) -> AppTest:
    return AppTest.from_file(str(ROOT / rel), default_timeout=120).run()


@pytest.mark.parametrize("name", PAGES)
def test_page_runs_without_error(name):
    at = _run(PAGES[name])
    assert not at.exception, [e.value for e in at.exception]


def test_home_runs():
    at = _run("app/Home.py")
    assert not at.exception


def test_quality_page_kpis():
    at = _run(PAGES["chat_luong"])
    kpi = {mt.label: mt.value for mt in at.metric}
    assert kpi["Tổng số mẫu"] == "72"
    assert kpi["Lô đạt kiểm tra"] == "3/4"
    assert kpi["Mẫu bị gắn cờ"] == "1"


def test_gene_page_default_cd33():
    at = _run(PAGES["kham_pha_gen"])
    assert {mt.label: mt.value for mt in at.metric}["Probe"] == "M23197_at"


def test_gene_page_not_found_message():
    at = _run(PAGES["kham_pha_gen"])
    at.text_input(key="gene_query").set_value("khong_co_gen_nay").run()
    assert not at.exception
    assert any("Không tìm thấy" in w.value for w in at.warning)


def test_patient_page_hides_truth_until_button():
    at = _run(PAGES["ho_so"])
    assert not any("Chẩn đoán thật" in i.value for i in at.info)
    next(b for b in at.button if b.label == "Hiện chẩn đoán thật").click().run()
    assert any("Chẩn đoán thật" in i.value for i in at.info)


def test_patient_page_uncertain_case():
    at = _run(PAGES["ho_so"])
    at.selectbox[0].set_value(66).run()          # bệnh nhân tín hiệu yếu trong dữ liệu giả
    assert any("Không chắc chắn" in w.value for w in at.warning)


# ---- Định dạng hàm (áp dụng cho dữ liệu giả; hàm thật phải trả cùng định dạng, mục 4 kế hoạch) ----
def test_mock_ingest_batches_format():
    df = m.ingest_batches()
    assert {"batch_id", "source", "n_samples", "status", "reason"} <= set(df.columns)
    assert set(df["status"]) <= {"passed", "blocked"}
    assert list(df["batch_id"]) == list(m.BATCH_SOURCES)


def test_mock_validate_new_sample():
    ok = m.validate_new_sample(io.BytesIO(m.make_sample_file("valid")))
    assert ok["ok"] and ok["errors"] == [] and len(ok["values"]) == m.N_GENES
    for kind in ("missing_genes", "blank"):
        bad = m.validate_new_sample(io.BytesIO(m.make_sample_file(kind)))
        assert not bad["ok"] and bad["errors"] and bad["values"] is None
    assert set(ok) == {"ok", "errors", "warnings", "values"}


def test_mock_patient_profile_format():
    p = m.patient_profile(sample_id=52)
    assert set(p) >= {"prediction", "ps", "votes", "similar"}
    assert p["prediction"] in {"ALL", "AML", "uncertain"}
    assert list(p["votes"].columns) == ["probe_id", "description", "value", "vote", "toward"]
    assert list(p["similar"].columns) == ["sample_id", "correlation", "class"]
    assert len(p["similar"]) == m.N_SIMILAR


def test_mock_patient_profile_from_new_sample_matches_existing():
    v = m.validate_new_sample(io.BytesIO(m.make_sample_file("valid", 52)))["values"]
    assert m.patient_profile(values=v)["prediction"] == m.patient_profile(sample_id=52)["prediction"]


def test_mock_reference_follows_convention():
    X, s = m.reference_data()
    assert X.shape == (72, m.N_GENES)
    assert (s["split"] == "train").sum() == 38 and (s["split"] == "test").sum() == 34


def test_mock_gene_functions_format():
    hits = m.find_gene("CD33")
    assert list(hits.columns) == m.FIND_COLUMNS and hits["probe_id"].iloc[0] == "M23197_at"
    stats = m.gene_stats("M23197_at")
    assert {"class", "source", "tissue", "n", "mean", "median", "sd"} <= set(stats.columns)
    assert m.ranking_table(10).shape[0] == 10
