"""
TV2 - GĐ1: Mục 1.1 Kiểm tra chất lượng dữ liệu (mục 2.5, Bảng 2.9).

Bốn mức kiểm tra theo phân loại vấn đề chất lượng:
  Mức 1 - Record    : cấu trúc (72x7129, không trùng sample_id/probe, quan hệ một-một)
  Mức 2 - Relation  : nhãn Kaggle nhất quán với subtype metadata (72/72, Bảng 2.4)
  Mức 3 - Value     : NaN, giá trị <100 / >16000 (đặc thù MAS4), bão hòa; value-set
                      giá trị metadata ngoài danh mục (tissue/source/subtype/sex)
  Mức 4 - Record    : outlier mẫu (chip lỗi) - gắn cờ qc_outlier, KHÔNG xóa

Đầu ra: data/cleansed/expression.parquet, samples.parquet (+qc_outlier),
        data/cleansed/quality_report.md (định dạng như demo nghiệm thu).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from src.config import STD_DIR, CLEANSED_DIR, VALID_TISSUE, VALID_SOURCE, VALID_SEX
except ModuleNotFoundError:
    from config import STD_DIR, CLEANSED_DIR, VALID_TISSUE, VALID_SOURCE, VALID_SEX

REPORT = CLEANSED_DIR / "quality_report.md"
EXPR_HI = 16000
EXPR_LO = 100


# ---------------------------------------------------------------------------
# Các khối kiểm tra
# ---------------------------------------------------------------------------
def check_structure(report, X, samples, genes) -> None:
    """Mức 1 - Record/cấu trúc."""
    ok_shape = X.shape == (72, 7129)
    no_dup_sample = X.index.is_unique and samples.index.is_unique
    no_dup_probe = X.columns.is_unique
    one_to_one = samples.index.is_unique
    report.append(f"[{'PASS' if ok_shape else 'FAIL'}] 72 mẫu, 7129 probe, kích thước X = {X.shape}")
    report.append(f"[{'PASS' if no_dup_sample else 'FAIL'}] không trùng sample_id")
    report.append(f"[{'PASS' if no_dup_probe else 'FAIL'}] không trùng probe id")
    report.append(f"[{'PASS' if one_to_one else 'FAIL'}] ghép metadata/nhãn một-một theo sample_id")
    n_affx = int(genes["is_control"].sum()) if "is_control" in genes.columns else 0
    report.append(f"[INFO] số probe AFFX (is_control): {n_affx}")


def check_labels(report, samples) -> None:
    """Mức 2 - Relation: nhãn Kaggle vs subtype metadata (Bảng 2.4)."""
    sub = samples["subtype"].dropna()
    mism = ((samples["class"] == "ALL") & ~samples["subtype"].isin(["B-ALL", "T-ALL"])) | \
           ((samples["class"] == "AML") & ~samples["subtype"].eq("AML"))
    n_ok = len(samples) - int(mism.sum())
    report.append(f"[{'PASS' if n_ok == 72 else 'FAIL'}] Nhãn Kaggle khớp subtype metadata: {n_ok}/72")
    if mism.sum():
        report.append(f"[INFO] mẫu lệch: {samples.index[mism].tolist()}")


def check_values(report, X, samples) -> None:
    """Mức 3 - Value + value-set."""
    vals = X.values.astype(float)
    n_nan = int(np.isnan(vals).sum())
    n_lo = int((vals < EXPR_LO).sum())
    n_hi = int((vals > EXPR_HI).sum())
    n_neg = int((vals < 0).sum())
    report.append(f"[INFO] Số giá trị < {EXPR_LO}: {n_lo}; > {EXPR_HI}: {n_hi}; "
                  f"NaN: {n_nan}; giá trị âm: {n_neg}")
    report.append(f"[{'WARN' if n_nan else 'PASS'}] không có giá trị khuyết trong biểu hiện")

    ok_tissue = set(samples["tissue"].dropna().unique()) <= VALID_TISSUE
    ok_source = set(samples["source"].dropna().unique()) <= VALID_SOURCE
    ok_sex = set(samples["sex"].dropna().unique()) <= VALID_SEX
    bad_sub = set(samples["subtype"].dropna().unique()) - {"B-ALL", "T-ALL", "AML"}
    report.append(f"[{'PASS' if ok_tissue else 'WARN'}] tissue ∈ {sorted(VALID_TISSUE)}")
    report.append(f"[{'PASS' if ok_source else 'WARN'}] source ∈ {sorted(VALID_SOURCE)}")
    report.append(f"[{'PASS' if ok_sex else 'WARN'}] sex ∈ {{M, F}} (NaN được phép)")
    report.append(f"[{'WARN' if bad_sub else 'PASS'}] subtype ∈ {{B-ALL, T-ALL, AML}}")
    n_sex_miss = int(samples["sex"].isna().sum())
    report.append(f"[INFO] Số mẫu thiếu sex: {n_sex_miss}/72 (metadata gốc thiếu - không tự điền)")


def check_outliers(report, X, samples) -> None:
    """Mức 4 - Record: mẫu nghi chip lỗi bằng z-score toàn cục trên log10; gắn cờ, không xóa."""
    vals = np.clip(X.values.astype(float), 1.0, None)
    g = np.log10(vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (g - g.mean()) / g.std()
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    n_extreme = (np.abs(z) > 4).sum(axis=1)
    flagged = samples.index[n_extreme > 50].tolist()
    report.append(f"[INFO] Mẫu có >50 gen |z|>4 (nghi lệch phân phối): {flagged if flagged else 'không có'}")
    samples["qc_outlier"] = samples.index.isin(flagged)
    report.append(f"[INFO] Mẫu gắn cờ qc_outlier: {flagged if flagged else 'không có'} "
                  "(lý do: nhiều gen bất thường so với tổng thể; không xóa nếu không có căn cứ)")


# ---------------------------------------------------------------------------
def run_quality() -> pd.DataFrame:
    """Chạy 4 mức trên tầng standardized, ghi cleansed + quality_report.md."""
    req = (STD_DIR / "expression.parquet", STD_DIR / "samples.parquet", STD_DIR / "genes.parquet")
    for f in req:
        if not f.exists():
            raise FileNotFoundError(f"Thiếu {f.name}: chạy `python3 load.py` (hay `make data`) trước.")
    X = pd.read_parquet(STD_DIR / "expression.parquet")
    samples = pd.read_parquet(STD_DIR / "samples.parquet")
    genes = pd.read_parquet(STD_DIR / "genes.parquet")
    X.index = X.index.astype(int)
    samples.index = samples.index.astype(int)

    report = ["# Quality Report - Golub 1999", "",
              "## Tổng quan", "",
              f"- Số mẫu: {X.shape[0]}", f"- Số probe: {X.shape[1]}", "",
              "## Kết quả 4 mức", ""]
    check_structure(report, X, samples, genes)
    check_labels(report, samples)
    check_values(report, X, samples)
    check_outliers(report, X, samples)
    report += ["", "Báo cáo sinh bởi src/quality.py (TV2 - mục 1.1), tầng standardized → cleansed."]

    CLEANSED_DIR.mkdir(parents=True, exist_ok=True)
    X.to_parquet(CLEANSED_DIR / "expression.parquet", index=True)
    samples.to_parquet(CLEANSED_DIR / "samples.parquet", index=True)

    msg = "\n".join(report)
    REPORT.write_text(msg, encoding="utf-8")
    print(REPORT)
    print(msg)
    return samples


if __name__ == "__main__":
    run_quality()