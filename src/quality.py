"""Kiểm tra chất lượng dữ liệu 4 mức (mục 2.5, Bảng 2.9), ghi tầng cleansed + quality_report.md.

TV2 - công việc 1.1 (GĐ1). Chạy: `python -m src.quality` hoặc `make quality`.

Bốn mức theo đúng thứ tự Bảng 2.9:
  Value     : NaN; giá trị < 100 / > 16000 (đặc thù MAS4); giá trị âm
  Value-set : probe điều khiển AFFX; nhiều probe cho cùng một gen; metadata ngoài danh mục
  Record    : kích thước, trùng sample_id/probe; mẫu trùng lặp; mẫu có phân phối bất thường
              (tương quan giữa các mẫu + boxplot theo mẫu) -> gắn cờ qc_outlier, KHÔNG xóa
  Relation  : X và samples cùng sample_id; nhãn Kaggle khớp subtype OpenIntro; phân bố Bảng 2.4

Đầu ra (chỉ khi không có [FAIL]):
  data/cleansed/expression.parquet   - copy nguyên vẹn của tầng standardized
  data/cleansed/samples.parquet      - thêm cột qc_outlier
  data/cleansed/quality_report.md    - báo cáo 4 mức (luôn được ghi, kể cả khi FAIL)

Có [FAIL] thì không ghi tầng cleansed và thoát mã 1, để `make data` dừng lại thay vì
chạy tiếp trên dữ liệu chưa đạt (dữ liệu chỉ chảy một chiều khi đã kiểm tra xong, mục 2.2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import (
        STD_DIR, CLEANSED_DIR, VALID_SUBTYPE, VALID_TISSUE, VALID_SOURCE, VALID_SEX,
    )
except ModuleNotFoundError:  # chạy trực tiếp `python src/quality.py`
    from config import (
        STD_DIR, CLEANSED_DIR, VALID_SUBTYPE, VALID_TISSUE, VALID_SOURCE, VALID_SEX,
    )

EXPR_LO, EXPR_HI = 100, 16000      # ngưỡng đặc thù thang đo MAS4 (mục 2.6)

# NOTICE: ngưỡng modified z-score 3.5 theo Iglewicz & Hoaglin (1993), "How to Detect and
# Handle Outliers". Dùng median/MAD thay cho mean/SD vì chính mẫu lỗi sẽ kéo mean/SD về
# phía nó và tự che mình. Đổi ngưỡng thì ghi vào docs/decisions.md (ADR-004).
MODZ_THRESHOLD = 3.5
# Hai mẫu có tương quan (log10) cao hơn mức này được coi là trùng lặp (cùng chip / chép nhầm).
DUP_CORR = 0.99

# Bảng 2.4: số mẫu kỳ vọng lấy từ bài báo, phải khớp tuyệt đối.
EXPECTED_SPLIT_CLASS_SOURCE = {
    ("train", "ALL", "DFCI"): 27, ("train", "AML", "CALGB"): 11,
    ("test", "ALL", "DFCI"): 17, ("test", "ALL", "St-Jude"): 3,
    ("test", "AML", "CALGB"): 4, ("test", "AML", "St-Jude"): 5, ("test", "AML", "CCG"): 5,
}
EXPECTED_SPLIT_TISSUE = {("train", "BM"): 38, ("test", "BM"): 24, ("test", "PB"): 10}


class QualityCheckError(RuntimeError):
    """Có ít nhất một kiểm tra [FAIL]; tầng cleansed không được ghi."""


def _line(report: list[str], ok: bool, text: str) -> None:
    report.append(f"[{'PASS' if ok else 'FAIL'}] {text}")


def modified_z(values: np.ndarray) -> np.ndarray:
    """Modified z-score 0.6745·(x − median)/MAD (Iglewicz & Hoaglin).

    MAD = 0 (mọi giá trị như nhau) thì không có gì bất thường -> trả về toàn 0
    thay vì chia cho 0.
    """
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - med) / mad


# ---------------------------------------------------------------------------
# Bốn mức kiểm tra (Bảng 2.9)
# ---------------------------------------------------------------------------
def check_value(report: list[str], X: pd.DataFrame) -> dict[str, int]:
    """Mức Value: giá trị thiếu, âm, ngoài vùng đo tin cậy [100, 16000]."""
    vals = X.to_numpy(dtype=float)
    counts = {
        "nan": int(np.isnan(vals).sum()),
        "lo": int((vals < EXPR_LO).sum()),
        "hi": int((vals > EXPR_HI).sum()),
        "neg": int((vals < 0).sum()),
    }
    _line(report, counts["nan"] == 0, f"không có giá trị khuyết trong biểu hiện (NaN: {counts['nan']})")
    report.append(f"[INFO] Số giá trị < {EXPR_LO}: {counts['lo']}; > {EXPR_HI}: {counts['hi']}; "
                  f"giá trị âm: {counts['neg']} — xử lý bằng threshold ở tầng curated (TV3, 1.2)")
    report.append(f"[INFO] Kiểu dữ liệu biểu hiện: {X.dtypes.iloc[0]}")
    return counts


def check_value_set(report: list[str], samples: pd.DataFrame, genes: pd.DataFrame) -> int:
    """Mức Value-set: probe điều khiển, nhiều probe cho một gen, danh mục metadata."""
    n_affx = int(genes["is_control"].sum())
    report.append(f"[INFO] Probe điều khiển AFFX (is_control): {n_affx} — giữ lại, đánh dấu ở genes.parquet")

    # Nhiều probe cùng một mô tả gen: không phải lỗi, nhưng TV4/TV5 phải biết khi đếm "số gen"
    # và khi diễn giải (một gen có thể xuất hiện nhiều lần trong danh sách chọn).
    dup = genes["description"].duplicated(keep=False)
    report.append(f"[INFO] Nhiều probe cho cùng một gen: {int(dup.sum())} probe thuộc "
                  f"{genes.loc[dup, 'description'].nunique()} mô tả trùng — ghi chú khi diễn giải gen")

    valid_subtype = set().union(*VALID_SUBTYPE.values())
    catalogs = {"tissue": VALID_TISSUE, "source": VALID_SOURCE, "subtype": valid_subtype, "sex": VALID_SEX}
    for col, allowed in catalogs.items():
        bad = sorted(set(samples[col].dropna().unique()) - allowed)
        _line(report, not bad, f"{col} ∈ {sorted(allowed)}" + (f" — ngoài danh mục: {bad}" if bad else ""))
    n_sex_miss = int(samples["sex"].isna().sum())
    report.append(f"[INFO] Số mẫu thiếu sex: {n_sex_miss}/{len(samples)} (OpenIntro thiếu — không tự điền)")
    return n_affx


def check_record(report: list[str], X: pd.DataFrame, samples: pd.DataFrame) -> dict[int, list[str]]:
    """Mức Record: cấu trúc, mẫu trùng lặp, mẫu có phân phối bất thường.

    Trả về {sample_id: [lý do, ...]} cho các mẫu cần gắn cờ qc_outlier.
    """
    _line(report, X.shape == (72, 7129), f"72 mẫu, 7129 probe — kích thước X = {X.shape}")
    _line(report, X.index.is_unique and samples.index.is_unique, "không trùng sample_id")
    _line(report, X.columns.is_unique, "không trùng probe id")

    # Đo trên log10 sau threshold [100, 16000] chỉ để so sánh các mẫu với nhau; giá trị
    # ghi ra tầng cleansed vẫn là giá trị gốc (phép biến đổi thật nằm ở tầng curated).
    logged = np.log10(np.clip(X.to_numpy(dtype=float), EXPR_LO, EXPR_HI))
    ids = X.index.to_numpy()
    reasons: dict[int, list[str]] = {}

    def flag(i: int, why: str) -> None:
        reasons.setdefault(int(ids[i]), []).append(why)

    corr = np.corrcoef(logged)
    np.fill_diagonal(corr, np.nan)  # bỏ tương quan của mẫu với chính nó

    # Mẫu trùng lặp: hai chip gần như giống hệt nhau.
    dup_i, dup_j = np.where(np.triu(np.nan_to_num(corr) > DUP_CORR, k=1))
    for i, j in zip(dup_i, dup_j):
        flag(i, f"trùng lặp với mẫu {ids[j]} (r = {corr[i, j]:.3f})")
        flag(j, f"trùng lặp với mẫu {ids[i]} (r = {corr[i, j]:.3f})")
    _line(report, len(dup_i) == 0, f"không có cặp mẫu trùng lặp (tương quan > {DUP_CORR})")

    # Tương quan giữa các mẫu: chip lỗi tương quan kém với mọi mẫu còn lại -> chỉ xét phía thấp.
    med_corr = np.nanmedian(corr, axis=1)
    z_corr = modified_z(med_corr)
    for i in np.where(z_corr < -MODZ_THRESHOLD)[0]:
        flag(i, f"tương quan trung vị với các mẫu khác thấp (r = {med_corr[i]:.3f}, z = {z_corr[i]:.1f})")

    # Boxplot theo mẫu: trung vị và IQR của từng mẫu lệch hẳn so với phần còn lại (cả hai phía).
    q25, med, q75 = np.percentile(logged, [25, 50, 75], axis=1)
    for name, stat in (("trung vị", med), ("IQR", q75 - q25)):
        z = modified_z(stat)
        for i in np.where(np.abs(z) > MODZ_THRESHOLD)[0]:
            flag(i, f"{name} log10 bất thường ({stat[i]:.3f}, z = {z[i]:.1f})")

    report.append(f"[INFO] Tương quan trung vị giữa các mẫu: thấp nhất {np.min(med_corr):.3f} "
                  f"(mẫu {ids[np.argmin(med_corr)]}), trung vị {np.median(med_corr):.3f}")
    detail = "; ".join(f"{sid}: {', '.join(why)}" for sid, why in sorted(reasons.items()))
    report.append(f"[INFO] Mẫu gắn cờ qc_outlier: {len(reasons)}"
                  + (f" (lý do: {detail})" if reasons else "")
                  + f" — quy tắc modified z > {MODZ_THRESHOLD}; chỉ gắn cờ, không xóa (ADR-004)")
    return reasons


def check_relation(report: list[str], X: pd.DataFrame, samples: pd.DataFrame) -> int:
    """Mức Relation: khóa ghép, nhãn Kaggle vs subtype OpenIntro, phân bố Bảng 2.4."""
    _line(report, X.index.equals(samples.index), "biểu hiện và metadata cùng chỉ mục sample_id, cùng thứ tự")

    # class thiếu hoặc lạ -> map ra NaN -> tính là lệch (không phải set nên không khớp được)
    allowed = samples["class"].map(VALID_SUBTYPE)
    ok = pd.Series([isinstance(al, set) and sub in al for sub, al in zip(samples["subtype"], allowed)],
                   index=samples.index)
    n_ok = int(ok.sum())
    _line(report, n_ok == len(samples), f"Nhãn Kaggle khớp subtype OpenIntro: {n_ok}/{len(samples)}")
    if n_ok != len(samples):
        report.append(f"[INFO] mẫu lệch: {samples.index[~ok].tolist()}")

    observed = samples.groupby(["split", "class", "source"]).size()
    observed = {k: int(v) for k, v in observed.items() if v}
    _line(report, observed == EXPECTED_SPLIT_CLASS_SOURCE,
          "split × class × source khớp Bảng 2.4 (train: 27 ALL DFCI, 11 AML CALGB; "
          "test: ALL 17 DFCI + 3 St-Jude, AML 4 CALGB + 5 St-Jude + 5 CCG)")
    if observed != EXPECTED_SPLIT_CLASS_SOURCE:
        report.append(f"[INFO] thực tế: {observed}")

    observed = samples.groupby(["split", "tissue"]).size()
    observed = {k: int(v) for k, v in observed.items() if v}
    _line(report, observed == EXPECTED_SPLIT_TISSUE, "split × tissue khớp Bảng 2.4 (train: 38 BM; test: 24 BM, 10 PB)")
    if observed != EXPECTED_SPLIT_TISSUE:
        report.append(f"[INFO] thực tế: {observed}")

    sub = samples["subtype"].value_counts().to_dict()
    ok_sub = (sub.get("B-ALL", 0) + sub.get("T-ALL", 0) == 47 and sub.get("AML", 0) == 25
              and samples["subtype"].notna().all())
    _line(report, ok_sub, f"subtype: B-ALL + T-ALL = 47, AML = 25, không thiếu (thực tế {sub})")
    return n_ok


# ---------------------------------------------------------------------------
def run_quality(std_dir: Path = STD_DIR, out_dir: Path = CLEANSED_DIR) -> pd.DataFrame:
    """Chạy 4 mức trên tầng standardized; ghi tầng cleansed + quality_report.md.

    Luôn ghi quality_report.md để xem lý do. Nếu có [FAIL] thì xóa expression/samples cũ
    của tầng cleansed (tránh để người sau đọc nhầm bản cũ tưởng là mới) và ném
    QualityCheckError.
    """
    req = {n: std_dir / f"{n}.parquet" for n in ("expression", "samples", "genes")}
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

    sections: dict[str, list[str]] = {name: [] for name in ("Value", "Value-set", "Record", "Relation")}
    counts = check_value(sections["Value"], X)
    n_affx = check_value_set(sections["Value-set"], samples, genes)
    reasons = check_record(sections["Record"], X, samples)
    n_label_ok = check_relation(sections["Relation"], X, samples)
    failed = [ln for lines in sections.values() for ln in lines if ln.startswith("[FAIL]")]

    # Định dạng tóm tắt đúng như phần "Trích quality_report.md" trong demo nghiệm thu (trang 20).
    no_dup_ids = X.shape == (72, 7129) and X.index.is_unique and samples.index.is_unique
    summary = [
        f"[{'PASS' if no_dup_ids else 'FAIL'}] {X.shape[0]} mẫu, {X.shape[1]} probe, không trùng sample_id",
        f"[{'PASS' if n_label_ok == len(samples) else 'FAIL'}] Nhãn Kaggle khớp subtype OpenIntro: "
        f"{n_label_ok}/{len(samples)}",
        f"[INFO] Số giá trị < {EXPR_LO}: {counts['lo']} Số giá trị > {EXPR_HI}: {counts['hi']} Probe AFFX: {n_affx}",
        f"[INFO] Mẫu gắn cờ qc_outlier: {len(reasons)}"
        + (f" (lý do: xem mức Record — mẫu {sorted(reasons)})" if reasons else " (lý do: không có mẫu vượt ngưỡng)"),
    ]
    report = ["# Quality Report - Golub 1999 (tầng standardized → cleansed)", "",
              "## Tóm tắt", "", *summary, "",
              f"Kết luận: {'ĐẠT' if not failed else f'KHÔNG ĐẠT ({len(failed)} kiểm tra FAIL)'}", "",
              "## Kết quả 4 mức (mục 2.5, Bảng 2.9)", ""]
    for name, lines in sections.items():
        report += [f"### Mức {name}", "", *lines, ""]
    report += ["## Ghi chú", "",
               "- Tầng cleansed KHÔNG xóa hay sửa giá trị biểu hiện; chỉ gắn cờ `qc_outlier`.",
               "- Ngưỡng [100, 16000] chỉ để báo cáo và để so sánh các mẫu; phép threshold/log10 "
               "thực hiện ở tầng curated (công việc 1.2 của TV3).",
               "- Sinh bởi `python -m src.quality` (TV2 - công việc 1.1)."]

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "quality_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"\nĐã ghi: {report_path}")

    if failed:
        for name in ("expression.parquet", "samples.parquet"):
            (out_dir / name).unlink(missing_ok=True)
        raise QualityCheckError("Kiểm tra chất lượng thất bại:\n  " + "\n  ".join(failed))

    samples["qc_outlier"] = samples.index.isin(list(reasons))
    X.to_parquet(out_dir / "expression.parquet", index=True)
    samples.to_parquet(out_dir / "samples.parquet", index=True)
    return samples


if __name__ == "__main__":
    try:
        run_quality()
    except QualityCheckError as err:
        print(f"\n{err}", file=sys.stderr)
        sys.exit(1)
