"""Kiểm tra src/ingest.py (TV2 - công việc 2.4, ADR-014, ADR-030) trên dữ liệu thật.

Cần tầng standardized (và cleansed cho một test đối chiếu cờ): chạy `make data` trước. Thiếu
dữ liệu thì test báo lỗi, không skip — giống tests/test_load.py (ADR-006).
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CLEANSED_DIR, STD_DIR  # noqa: E402
from src.ingest import (  # noqa: E402
    BATCH_COLS, BATCH_SOURCES, BATCHES_FILE, LOG_FILE, QC_FILE, ingest_batches, sample_quality,
    validate_new_sample,
)
from src.load import SAMPLE_COLS  # noqa: E402

NEW_PATIENT = 52  # bệnh nhân tập test, đóng vai bệnh nhân mới (ADR-014; ví dụ của patient_profile)


@pytest.fixture(scope="module")
def std():
    """(expression, samples, probe chuẩn) của tầng standardized."""
    path = STD_DIR / "expression.parquet"
    assert path.exists(), f"Chưa có {path} — chạy `make data` trước khi chạy test (ADR-006)"
    X = pd.read_parquet(path)
    samples = pd.read_parquet(STD_DIR / "samples.parquet")
    X.index, samples.index = X.index.astype(int), samples.index.astype(int)
    probes = pd.read_parquet(STD_DIR / "genes.parquet")["probe_id"].tolist()
    return X, samples, probes


@pytest.fixture(scope="module")
def real_batches():
    return ingest_batches()


def _chip_file(tmp_path, probes, values, name="mau_moi.csv") -> str:
    """Ghi file chip một mẫu theo định dạng `probe_id,value`."""
    path = tmp_path / name
    pd.DataFrame({"probe_id": probes, "value": values}).to_csv(path, index=False)
    return str(path)


def _batch_file(tmp_path, X, samples, ids, name, new_ids=None) -> str:
    """Ghi file lô thêm: sample_id + bảng mẫu + 7129 cột probe, lấy từ các mẫu `ids`."""
    df = samples.loc[ids, SAMPLE_COLS].copy()
    df = pd.concat([df, X.loc[ids]], axis=1)
    df.insert(0, "sample_id", new_ids if new_ids is not None else ids)
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# ingest_batches / sample_quality
# ---------------------------------------------------------------------------
def test_real_data_is_split_into_four_source_batches_that_all_pass(real_batches):
    # Bảng 2.4: DFCI = 27 train + 17 test, CALGB = 11 + 4, St-Jude = 0 + 8, CCG = 0 + 5.
    b = real_batches.set_index("source")
    assert list(real_batches.columns) == BATCH_COLS
    assert list(real_batches["source"]) == list(BATCH_SOURCES)
    assert list(real_batches["batch_id"]) == ["B1-DFCI", "B2-CALGB", "B3-St-Jude", "B4-CCG"]
    assert b["n_samples"].to_dict() == {"DFCI": 44, "CALGB": 15, "St-Jude": 8, "CCG": 5}
    assert b["n_train"].to_dict() == {"DFCI": 27, "CALGB": 11, "St-Jude": 0, "CCG": 0}
    assert b["n_test"].to_dict() == {"DFCI": 17, "CALGB": 4, "St-Jude": 8, "CCG": 5}
    assert (real_batches["status"] == "passed").all()
    assert (real_batches["reason"] == "").all()


def test_quality_metrics_are_filled_for_every_real_batch(real_batches):
    for col in ("pct_below_100", "pct_above_16000", "present_rate", "median_corr_other"):
        assert real_batches[col].between(0, 1).all(), col
    assert real_batches["n_qc_outlier"].sum() == 1
    assert "mẫu gắn cờ qc_outlier: [21]" in real_batches.set_index("source").loc["DFCI", "warnings"]


def test_sample_flags_match_cleansed_layer():
    # Cờ qc_outlier của bảng fact_qc phải trùng tầng cleansed (cùng quy tắc quality.sample_qc),
    # nếu không thì kho của TV3 và quality_report.md sẽ mâu thuẫn nhau.
    path = CLEANSED_DIR / "samples.parquet"
    assert path.exists(), f"Chưa có {path} — chạy `make data` trước khi chạy test (ADR-006)"
    cleansed = pd.read_parquet(path)
    cleansed.index = cleansed.index.astype(int)
    qc = sample_quality().set_index("sample_id")

    assert list(qc.index) == list(range(1, 73))
    assert qc["qc_outlier"].equals(cleansed["qc_outlier"].reindex(qc.index))
    assert qc.index[qc["qc_outlier"]].tolist() == [21]
    assert (qc["batch_status"] == "passed").all()


def test_write_outputs_and_log_is_appended_not_overwritten(tmp_path):
    ingest_batches(write=True, out_dir=tmp_path)
    ingest_batches(write=True, out_dir=tmp_path)

    assert len(pd.read_parquet(tmp_path / BATCHES_FILE)) == 4
    assert len(pd.read_parquet(tmp_path / QC_FILE)) == 72
    log = pd.read_csv(tmp_path / LOG_FILE)
    assert len(log) == 8                                   # 2 lần chạy × 4 lô
    assert list(log.columns) == ["run_at", "batch_id", "source", "n_samples", "status", "reason"]


def test_default_call_does_not_write_files(tmp_path):
    # Dashboard (TV6) gọi ingest_batches() trực tiếp: không được sinh file phụ mỗi lần tải trang.
    ingest_batches(out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_broken_extra_batch_is_blocked_without_blocking_real_batches(std, tmp_path):
    X, samples, _ = std
    ids = [50, 51, 53]
    path = _batch_file(tmp_path, X, samples, ids, "lo_loi", new_ids=[101, 102, 103])
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.drop(columns=[X.columns[100]])                 # thiếu 1 probe
    df.loc[0, X.columns[5]] = ""                           # 1 ô trống ở mẫu 101
    df.loc[1, "subtype"] = "T-ALL"                         # mẫu 102 là AML nhưng ghi T-ALL
    df.to_csv(path, index=False)

    out = ingest_batches(extra_files=[path]).set_index("batch_id")

    assert out.loc["lo_loi", "status"] == "blocked"
    reason = out.loc["lo_loi", "reason"]
    assert "thiếu 1 probe" in reason
    assert "1 ô biểu hiện trống ở mẫu [101]" in reason
    assert "class lệch subtype ở mẫu [102]" in reason
    assert np.isnan(out.loc["lo_loi", "present_rate"])     # hỏng cấu trúc: không có chỉ số
    assert (out.drop(index="lo_loi")["status"] == "passed").all()


def test_extra_batch_reusing_an_existing_sample_id_is_blocked(std, tmp_path):
    X, samples, _ = std
    path = _batch_file(tmp_path, X, samples, [NEW_PATIENT], "lo_trung_ma")

    out = ingest_batches(extra_files=[path]).set_index("batch_id")

    assert out.loc["lo_trung_ma", "status"] == "blocked"
    assert out.loc["lo_trung_ma", "reason"] == f"sample_id đã có ở lô khác: [{NEW_PATIENT}]"


def test_valid_extra_batch_passes_and_leaves_real_flags_unchanged(std, tmp_path):
    X, samples, _ = std
    path = _batch_file(tmp_path, X, samples, [57, 58], "lo_moi", new_ids=[201, 202])

    out = ingest_batches(extra_files=[path]).set_index("batch_id")
    qc = sample_quality(extra_files=[path]).set_index("sample_id")

    assert out.loc["lo_moi", "status"] == "passed"
    assert out.loc["lo_moi", "n_samples"] == 2
    assert qc.loc[[201, 202], "batch_id"].tolist() == ["lo_moi", "lo_moi"]
    assert qc.index[qc["qc_outlier"]].tolist() == [21]     # lô thêm không làm đổi cờ mẫu thật


# ---------------------------------------------------------------------------
# validate_new_sample
# ---------------------------------------------------------------------------
def test_valid_chip_file_returns_values_in_reference_order(std, tmp_path):
    # ROOT CAUSE:
    #
    # Bản ở nhánh 2.4-ingest nhận dict metadata và trả {valid, errors, warnings, normalized},
    # lệch giao diện đã khóa ở ADR-014 (validate_new_sample(path) -> {ok, errors, warnings,
    # values}), nên TV6 không gọi được từ nút tải file và TV1 không có values cho patient_profile.
    #
    # Nay hàm đọc file chip `probe_id,value` và trả values theo đúng thứ tự genes.parquet.
    X, _, probes = std
    shuffled = X.loc[NEW_PATIENT].sample(frac=1, random_state=0)     # thứ tự dòng bị xáo
    path = _chip_file(tmp_path, shuffled.index, shuffled.to_numpy())

    res = validate_new_sample(path)

    assert set(res) == {"ok", "errors", "warnings", "values"}
    assert res["ok"] is True
    assert res["errors"] == []
    assert res["warnings"] == []
    assert res["values"].index.tolist() == probes
    assert res["values"].to_numpy().tolist() == X.loc[NEW_PATIENT].astype(float).tolist()


def test_uploaded_file_object_is_accepted(std):
    # Nút tải file của Streamlit đưa vào một đối tượng file, không phải đường dẫn.
    X, _, _ = std
    buf = io.BytesIO(X.loc[NEW_PATIENT].rename("value").rename_axis("probe_id").to_csv().encode())
    buf.name = "benh_nhan_52.csv"

    res = validate_new_sample(buf)

    assert res["ok"] is True
    assert res["values"].name == "benh_nhan_52"


@pytest.mark.parametrize("mutate, expected", [
    (lambda df: df.drop(index=[0, 1, 2]), "thiếu 3/7129 gen"),
    (lambda df: df.assign(probe_id=df["probe_id"].replace({"X95735_at": "X95735_AT"})),
     "1 mã gen không có trong 7129 probe chuẩn: [X95735_AT]"),
    (lambda df: pd.concat([df, df.iloc[[10]]]), "1 mã gen lặp lại"),
    (lambda df: df.assign(value=df["value"].astype(str).where(df.index != 7, "")), "1 ô giá trị trống"),
    (lambda df: df.assign(value=df["value"].astype(str).where(df.index != 8, "abc")), "1 giá trị không phải số"),
    (lambda df: df.rename(columns={"probe_id": "Gene Accession Number"}), "'probe_id,value'"),
])
def test_broken_chip_file_is_rejected(std, tmp_path, mutate, expected):
    X, _, _ = std
    df = X.loc[NEW_PATIENT].rename("value").rename_axis("probe_id").reset_index()
    path = tmp_path / "loi.csv"
    mutate(df).to_csv(path, index=False)

    res = validate_new_sample(path)

    assert res["ok"] is False
    assert res["values"] is None
    assert any(expected in e for e in res["errors"]), res["errors"]


def test_log_scaled_file_is_rejected(std, tmp_path):
    X, _, probes = std
    logged = np.log10(X.loc[NEW_PATIENT].clip(100, 16000))
    res = validate_new_sample(_chip_file(tmp_path, probes, logged.to_numpy()))

    assert res["ok"] is False
    assert "có vẻ file đã log hóa" in res["errors"][0]


def test_chip_unlike_reference_samples_passes_with_warning(std, tmp_path):
    # Xáo giá trị giữa các gen: đủ 7129 gen hợp lệ về cấu trúc nhưng không còn giống mẫu thật nào.
    X, _, probes = std
    rng = np.random.default_rng(0)
    res = validate_new_sample(_chip_file(tmp_path, probes, rng.permutation(X.loc[NEW_PATIENT].to_numpy())))

    assert res["ok"] is True
    assert any("tương quan trung vị với 38 mẫu tham chiếu thấp bất thường" in w for w in res["warnings"])


@pytest.mark.parametrize("content, expected", [(None, "không tìm thấy file"), ("", "file rỗng")])
def test_missing_or_empty_file_is_rejected(tmp_path, content, expected):
    path = tmp_path / "khong_co.csv"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    res = validate_new_sample(path)

    assert res["ok"] is False
    assert expected in res["errors"][0]
