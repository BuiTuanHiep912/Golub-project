"""Kiểm tra chất lượng dữ liệu 4 mức (mục 2.5), ghi tầng cleansed + quality_report.md.

TV2 - công việc 1.1 (GĐ1). Chạy: `python -m src.quality` hoặc `make quality`.

Bốn mức kiểm tra theo phân loại vấn đề chất lượng:
  Mức 1 - Record    : cấu trúc (72 x 7129, không trùng sample_id/probe, quan hệ 1-1)
  Mức 2 - Relation  : nhãn Kaggle nhất quán với subtype OpenIntro (72/72, Bảng 2.4)
  Mức 3 - Value     : NaN, giá trị < 100 / > 16000 (đặc thù MAS4), giá trị âm; value-set
                      của metadata ngoài danh mục (tissue / source / subtype / sex)
  Mức 4 - Record    : mẫu nghi lỗi chip (z-score toàn cục) - gắn cờ qc_outlier, KHÔNG xóa

Đầu ra:
  data/cleansed/expression.parquet   - copy nguyên vẹn của tầng standardized
  data/cleansed/samples.parquet      - thêm cột qc_outlier
  data/cleansed/quality_report.md    - báo cáo 4 mức
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from src.config import (
        STD_DIR, CLEANSED_DIR, VALID_TISSUE, VALID_SOURCE, VALID_SEX,
    )
except ModuleNotFoundError:  # chạy trực tiếp `python src/quality.py`
    from config import (
        STD_DIR, CLEANSED_DIR, VALID_TISSUE, VALID_SOURCE, VALID_SEX,
    )

REPORT = CLEANSED_DIR / "quality_report.md"
EXPR_LO, EXPR_HI = 100, 16000      # ngưỡng đặc thù thang đo MAS4
OUTLIER_Z = 4.0                    # |z| của một gen để tính là bất thường
OUTLIER_MIN_GEN = 50               # ngưỡng số gen bất thường để gắn cờ một mẫu


# ---------------------------------------------------------------------------
# Các khối kiểm tra
# ---------------------------------------------------------------------------
def check_structure(report: list[str], X, samples, genes) -> None:
    """Mức 1 - Record: cấu trúc và khóa."""
    ok_shape = X.shape == (72, 7129)
    report.append(f"[{'PASS' if ok_shape else 'FAIL'}] 72 mẫu, 7129 probe, kích thước X = {X.shape}")
    ok_sample = X.index.is_unique and samples.index.is_unique
    report.append(f"[{'PASS' if ok_sample else 'FAIL'}] không trùng sample_id")
    ok_probe = X.columns.is_unique
    report.append(f"[{'PASS' if ok_probe else 'FAIL'}] không trùng probe id")
    same_keys = list(X.index) == list(samples.index)
    report.append(f"[{'PASS' if same_keys else 'FAIL'}] biểu hiện và metadata cùng thứ tự sample_id")
    n_affx = int(genes["is_control"].sum()) if "is_control" in genes.columns else 0
    report.append(f"[INFO] số probe AFFX (is_control): {n_affx}")


def check_labels(report: list[str], samples) -> None:
    """Mức 2 - Relation: nhãn Kaggle vs subtype OpenIntro (Bảng 2.4)."""
    mism = ((samples["class"] == "ALL") & ~samples["subtype"].isin(["B-ALL", "T-ALL"])) | \
           ((samples["class"] == "AML") & ~samples["subtype"].eq("AML"))
    n_ok = len(samples) - int(mism.sum())
    report.append(f"[{'PASS' if n_ok == len(samples) else 'FAIL'}] "
                  f"Nhãn Kaggle khớp subtype OpenIntro: {n_ok}/{len(samples)}")
    if int(mism.sum()):
        report.append(f"[INFO] mẫu lệch: {samples.index[mism].tolist()}")


def check_values(report: list[str], X, samples) -> None:
    """Mức 3 - Value và value-set."""
    vals = X.to_numpy(dtype=float)
    n_nan, n_lo, n_hi, n_neg = (int(np.isnan(vals).sum()), int((vals < EXPR_LO).sum()),
                               int((vals > EXPR_HI).sum()), int((vals < 0).sum()))
    report.append(f"[INFO] Số giá trị < {EXPR_LO}: {n_lo}; > {EXPR_HI}: {n_hi}; "
                  f"NaN: {n_nan}; giá trị âm: {n_neg}")
    report.append(f"[{'WARN' if n_nan else 'PASS'}] không có giá trị khuyết trong biểu hiện")

    ok_tissue = set(samples["tissue"].dropna().unique()) <= VALID_TISSUE
    ok_source = set(samples["source"].dropna().unique()) <= VALID_SOURCE
    ok_sex = set(samples["sex"].dropna().unique()) <= VALID_SEX
    bad_sub = set(samples["subtype"].dropna().unique()) - {"B-ALL", "T-ALL", "AML"}
    report.append(f"[{'PASS' if ok_tissue else 'WARN'}] tissue ∈ {sorted(VALID_TISSUE)}")
    report.append(f"[{'PASS' if ok_source else 'WARN'}] source ∈ {sorted(VALID_SOURCE)}")
    report.append(f"[{'PASS' if ok_sex else 'WARN'}] sex ∈ {{F, M}} (NaN được phép)")
    report.append(f"[{'WARN' if bad_sub else 'PASS'}] subtype ∈ {{B-ALL, T-ALL, AML}}")
    n_sex_miss = int(samples["sex"].isna().sum())
    report.append(f"[INFO] Số mẫu thiếu sex: {n_sex_miss}/{len(samples)} "
                  "(OpenIntro thiếu - không tự điền)")


def check_outliers(report: list[str], X, samples) -> pd.Series:
    """Mức 4 - Record: mẫu nghi lỗi chip; z-score toàn cục trên log10, chỉ gắn cờ."""
    logged = np.log10(np.clip(X.to_numpy(dtype=float), 1.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (logged - logged.mean()) / logged.std()
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    n_extreme = (np.abs(z) > OUTLIER_Z).sum(axis=1)
    flagged = pd.Series(samples.index[n_extreme > OUTLIER_MIN_GEN], name="qc_outlier")
    report.append(f"[INFO] Mẫu có >{OUTLIER_MIN_GEN} gen |z|>{OUTLIER_Z:g}: "
                  f"{flagged.tolist() if len(flagged) else 'không có'}")
    report.append(f"[INFO] Mẫu gắn cờ qc_outlier: {len(flagged)} "
                  "(không xóa nếu chưa có căn cứ - xem docs/decisions.md)")
    return flagged


# ---------------------------------------------------------------------------
def run_quality() -> pd.DataFrame:
    """Chạy 4 mức trên tầng standardized, ghi tầng cleansed + quality_report.md."""
    req = {n: STD_DIR / f"{n}.parquet" for n in ("expression", "samples", "genes")}
    missing = [f for f in req.values() if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu tầng standardized: {[f.name for f in missing]} — chạy `make standardize` trước."
        )
    X = pd.read_parquet(req["expression"])
    samples = pd.read_parquet(req["samples"])
    genes = pd.read_parquet(req["genes"])
    X.index = X.index.astype(int)
    samples.index = samples.index.astype(int)

    report = ["# Quality Report - Golub 1999 (tầng standardized → cleansed)", "",
              "## Tổng quan", "",
              f"- Số mẫu: {X.shape[0]}",
              f"- Số probe: {X.shape[1]}",
              f"- Kiểu dữ liệu biểu hiện: {X.dtypes.iloc[0]}", "",
              "## Kết quả 4 mức (mục 2.5)", "",
              "### Mức 1 - Record / cấu trúc", ""]
    check_structure(report, X, samples, genes)
    report += ["", "### Mức 2 - Relation / nhãn", ""]
    check_labels(report, samples)
    report += ["", "### Mức 3 - Value và value-set", ""]
    check_values(report, X, samples)
    report += ["", "### Mức 4 - Record / mẫu bất thường", ""]
    flagged = check_outliers(report, X, samples)
    samples["qc_outlier"] = samples.index.isin(flagged.index)

    CLEANSED_DIR.mkdir(parents=True, exist_ok=True)
    X.to_parquet(CLEANSED_DIR / "expression.parquet", index=True)
    samples.to_parquet(CLEANSED_DIR / "samples.parquet", index=True)

    report += ["", "## Ghi chú", "",
               "- Tầng cleansed KHÔNG xóa hay sửa giá trị biểu hiện; chỉ gắn cờ `qc_outlier`.",
               "- Ngưỡng [100, 16000] chỉ để báo cáo; phép threshold/log10 thực hiện ở tầng curated "
               "(công việc 1.2 của TV3).",
               "- Sinh bởi `python -m src.quality` (TV2 - công việc 1.1)."]

    msg = "\n".join(report)
    REPORT.write_text(msg, encoding="utf-8")
    print(f"Đã ghi: {REPORT}\n")
    print(msg)
    return samples


if __name__ == "__main__":
    run_quality()