"""Kiểm tra src/quality.py trên dữ liệu tổng hợp (TV2 - công việc 1.1).

Không cần dữ liệu Golub thật: mỗi test tự dựng một tầng standardized giả có đúng phân bố
Bảng 2.4 trong thư mục tạm, rồi gọi run_quality() như `make quality`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quality import QualityCheckError, run_quality  # noqa: E402

N_PROBES = 7129


def _samples_table_2_4() -> pd.DataFrame:
    """72 mẫu có phân bố split/class/source/tissue đúng Bảng 2.4."""
    rows = (
        [("train", "ALL", "DFCI", "BM")] * 27 + [("train", "AML", "CALGB", "BM")] * 11
        + [("test", "ALL", "DFCI", "BM")] * 17 + [("test", "ALL", "St-Jude", "BM")] * 3
        + [("test", "AML", "CALGB", "BM")] * 4 + [("test", "AML", "St-Jude", "PB")] * 5
        + [("test", "AML", "CCG", "PB")] * 5
    )
    samples = pd.DataFrame(rows, columns=["split", "class", "source", "tissue"],
                           index=pd.Index(range(1, 73), name="sample_id"))
    samples["subtype"] = np.where(samples["class"] == "AML", "AML", "B-ALL")
    samples.loc[samples.index[:9], "subtype"] = "T-ALL"  # 9 T-ALL như dữ liệu thật
    samples["sex"] = (["M", "F", None] * 24)[:72]
    return samples[["split", "class", "subtype", "tissue", "source", "sex"]]


def _write_std_layer(std_dir, corrupt_id: int | None = None) -> None:
    """Ghi expression/samples/genes giả; mẫu `corrupt_id` là chip hỏng không liên quan mẫu khác."""
    rng = np.random.default_rng(0)
    base = rng.uniform(2.0, 4.2, N_PROBES)                  # mức biểu hiện log10 chung của từng gen
    logged = base + rng.normal(0.0, 0.15, (72, N_PROBES))   # 72 chip đo cùng một mô hình biểu hiện
    ids = np.arange(1, 73)
    if corrupt_id is not None:
        logged[corrupt_id - 1] = rng.uniform(2.0, 4.2, N_PROBES)  # chip lỗi: tín hiệu ngẫu nhiên
    probes = [f"AFFX-C{i}_at" for i in range(58)] + [f"G{i}_at" for i in range(N_PROBES - 58)]
    X = pd.DataFrame(np.round(10 ** logged).astype(int), index=pd.Index(ids, name="sample_id"), columns=probes)
    genes = pd.DataFrame({"probe_id": probes, "description": [f"gene {i // 2}" for i in range(N_PROBES)]})
    genes["is_control"] = genes["probe_id"].str.startswith("AFFX")

    std_dir.mkdir(parents=True, exist_ok=True)
    X.to_parquet(std_dir / "expression.parquet")
    _samples_table_2_4().to_parquet(std_dir / "samples.parquet")
    genes.to_parquet(std_dir / "genes.parquet", index=False)


def test_qc_outlier_flags_the_corrupted_sample_by_sample_id(tmp_path):
    # ROOT CAUSE:
    #
    # Hai lỗi chồng nhau khiến qc_outlier không bao giờ đúng:
    # 1. z-score tính trên TOÀN BỘ ma trận (một mean, một SD cho mọi giá trị), nên trên dữ
    #    liệu thật |z| lớn nhất chỉ ~2.5 < ngưỡng 4: không mẫu nào có thể bị gắn cờ.
    # 2. Khi có mẫu bị gắn cờ, cột được gán bằng
    #        samples.index.isin(flagged.index)
    #    trong đó flagged là Series có GIÁ TRỊ là sample_id và index là 0..n-1, nên cờ của
    #    mẫu 50 rơi vào "mẫu 0" (không tồn tại); cờ của 3 mẫu rơi vào mẫu 1 và 2.
    #
    # Sửa bằng tương quan giữa các mẫu + boxplot theo mẫu với modified z-score (Bảng 2.9),
    # và gán cờ theo đúng tập sample_id trả về:
    #        samples.index.isin(list(reasons))
    std, out = tmp_path / "standardized", tmp_path / "cleansed"
    _write_std_layer(std, corrupt_id=50)

    run_quality(std, out)

    cleansed = pd.read_parquet(out / "samples.parquet")
    assert cleansed.shape == (72, 7)
    assert cleansed.index[cleansed["qc_outlier"]].tolist() == [50]
    report = (out / "quality_report.md").read_text(encoding="utf-8")
    assert "[INFO] Mẫu gắn cờ qc_outlier: 1" in report
    assert "50: tương quan trung vị với các mẫu khác thấp" in report


def test_clean_layer_has_no_flag_and_passes_all_four_levels(tmp_path):
    std, out = tmp_path / "standardized", tmp_path / "cleansed"
    _write_std_layer(std)

    run_quality(std, out)

    cleansed = pd.read_parquet(out / "samples.parquet")
    assert not cleansed["qc_outlier"].any()
    report = (out / "quality_report.md").read_text(encoding="utf-8")
    for level in ("Value", "Value-set", "Record", "Relation"):
        assert f"### Mức {level}" in report
    assert "[FAIL]" not in report
    assert "[PASS] Nhãn Kaggle khớp subtype OpenIntro: 72/72" in report
    assert "Nhiều probe cho cùng một gen: 7128 probe thuộc 3564 mô tả trùng" in report


def test_failed_check_stops_pipeline_and_removes_stale_cleansed(tmp_path):
    """[FAIL] phải chặn pipeline: không ghi (và xóa bản cũ của) tầng cleansed, vẫn ghi báo cáo."""
    std, out = tmp_path / "standardized", tmp_path / "cleansed"
    _write_std_layer(std)
    run_quality(std, out)                       # lần chạy tốt để lại tầng cleansed cũ
    assert (out / "samples.parquet").exists()

    samples = pd.read_parquet(std / "samples.parquet")
    samples.loc[1, "subtype"] = "AML"           # mẫu ALL nhưng subtype AML -> lệch nhãn
    samples.to_parquet(std / "samples.parquet")

    with pytest.raises(QualityCheckError, match="Nhãn Kaggle khớp subtype OpenIntro: 71/72"):
        run_quality(std, out)

    assert not (out / "samples.parquet").exists()
    assert not (out / "expression.parquet").exists()
    report = (out / "quality_report.md").read_text(encoding="utf-8")
    assert "KHÔNG ĐẠT" in report and "[INFO] mẫu lệch: [1]" in report
