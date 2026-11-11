"""Ingest dữ liệu Golub: nhập 4 lô nguồn, sinh bảng chất lượng, kiểm tra mẫu mới.

Công việc mở rộng cho 2.4/1.1: hỗ trợ kiểm tra nhanh 4 lô dữ liệu thô (raw) và
hàm validate_new_sample() để demo/kiểm tra mẫu mới trước khi đưa vào pipeline.

Bốn lô dữ liệu (data/raw/):
  1. kaggle/data_set_ALL_AML_train.csv      (biểu hiện + call, 38 mẫu train)
  2. kaggle/data_set_ALL_AML_independent.csv (biểu hiện + call, 34 mẫu test)
  3. kaggle/actual.csv                       (nhãn ALL/AML theo patient)
  4. openintro/golub.csv                     (6 cột mô tả metadata)

Chạy: python -m src.ingest
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.config import (
        DATA_DIR, KAGGLE_DIR, OPEN_INTRO_DIR, RAW_DIR,
        VALID_SEX, VALID_SOURCE, VALID_SUBTYPE, VALID_TISSUE,
    )
    from src.load import (
        ACTUAL_FILE, KAGGLE_INDEP, KAGGLE_TRAIN, META_CANDIDATES,
        META_COL_RENAME, META_COLS, SUBTYPE_MAP, md5_file,
    )
except ModuleNotFoundError:
    from config import (
        DATA_DIR, KAGGLE_DIR, OPEN_INTRO_DIR, RAW_DIR,
        VALID_SEX, VALID_SOURCE, VALID_SUBTYPE, VALID_TISSUE,
    )
    from load import (
        ACTUAL_FILE, KAGGLE_INDEP, KAGGLE_TRAIN, META_CANDIDATES,
        META_COL_RENAME, META_COLS, SUBTYPE_MAP, md5_file,
    )

# Các lô dữ liệu thô theo đúng nguồn trong kế hoạch
LOT_DEFS = [
    {"lot_id": "lot1_train", "name": "Kaggle train expression", "path": KAGGLE_TRAIN, "kind": "kaggle_expr", "expected_rows": 7129},
    {"lot_id": "lot2_independent", "name": "Kaggle independent expression", "path": KAGGLE_INDEP, "kind": "kaggle_expr", "expected_rows": 7129},
    {"lot_id": "lot3_actual", "name": "Kaggle actual labels", "path": ACTUAL_FILE, "kind": "labels", "expected_rows": 72},
    {"lot_id": "lot4_openintro", "name": "OpenIntro golub metadata", "path": META_CANDIDATES[0], "kind": "metadata", "expected_rows": 72},
]

SAMPLE_REQUIRED = ["sample_id", "split", "class", "subtype", "tissue", "source"]


def _safe_read_csv(path: Path, **kwargs) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return pd.read_csv(path, **kwargs), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def _check_kaggle_expr(df: pd.DataFrame) -> list[str]:
    issues = []
    if "Gene Accession Number" not in df.columns:
        issues.append("thiếu cột 'Gene Accession Number'")
    if "Gene Description" not in df.columns:
        issues.append("thiếu cột 'Gene Description'")
    call_cols = [c for c in df.columns if "call" in str(c).lower()]
    expr_cols = [c for c in df.columns if c not in {"Gene Accession Number", "Gene Description"} and "call" not in str(c).lower()]
    if not expr_cols:
        issues.append("không tìm thấy cột biểu hiện")
    if not call_cols:
        issues.append("không tìm thấy cột call (P/A/M)")
    if df["Gene Accession Number"].duplicated().any() if "Gene Accession Number" in df.columns else False:
        issues.append("probe_id trùng")
    return issues


def _check_labels(df: pd.DataFrame) -> list[str]:
    issues = []
    cols = {c.strip().lower() for c in df.columns}
    if "patient" not in cols:
        issues.append("thiếu cột 'patient'")
    if "cancer" not in cols and "class" not in cols:
        issues.append("thiếu cột nhãn (cancer/class)")
    if "patient" in cols:
        p = df[[c for c in df.columns if c.strip().lower() == "patient"][0]]
        if p.duplicated().any():
            issues.append("patient id trùng")
    label_col = "cancer" if "cancer" in cols else ("class" if "class" in cols else None)
    if label_col:
        c = df[[x for x in df.columns if x.strip().lower() == label_col][0]].astype(str).str.upper().str.strip()
        bad = sorted(set(c.unique()) - {"ALL", "AML"})
        if bad:
            issues.append(f"nhãn ngoài danh mục: {bad}")
    return issues


def _check_metadata(df: pd.DataFrame) -> list[str]:
    issues = []
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        issues.append(f"thiếu cột mô tả: {missing}")
        return issues
    if df["Samples"].duplicated().any():
        issues.append("sample_id (Samples) trùng")
    sub = df["cancer"].map(SUBTYPE_MAP)
    if sub.isna().any():
        bad = sorted(set(df.loc[sub.isna(), "cancer"].astype(str).unique()))
        issues.append(f"subtype không map được: {bad}")
    for col, allowed, src in [
        ("BM.PB", VALID_TISSUE, "tissue"),
        ("Source", VALID_SOURCE, "source"),
        ("Gender", VALID_SEX | {float("nan")}, "sex"),
    ]:
        vals = df[col].dropna().unique()
        if src == "sex":
            bad = sorted(set(str(v) for v in vals) - {"M", "F", "nan"})
        else:
            bad = sorted(set(vals) - allowed)
        if bad:
            issues.append(f"{src} ngoài danh mục: {bad}")
    return issues


def ingest_lots(lot_defs: list[dict] | None = None) -> pd.DataFrame:
    """Đọc 4 lô dữ liệu và trả về bảng chất lượng (quality table) theo từng lô."""
    lot_defs = lot_defs or LOT_DEFS
    rows = []
    for lot in lot_defs:
        path = Path(lot["path"])
        rec: dict[str, Any] = {
            "lot_id": lot["lot_id"],
            "name": lot["name"],
            "path": str(path),
            "kind": lot["kind"],
            "exists": path.exists(),
            "n_rows": None,
            "n_cols": None,
            "md5": None,
            "status": "FAIL",
            "issues": "",
        }
        if not path.exists():
            rec["issues"] = "file không tồn tại"
            rows.append(rec)
            continue
        rec["md5"] = md5_file(path)
        kwargs = {}
        if lot["kind"] == "metadata":
            # đọc nhanh để kiểm tra cột, không cần toàn bộ 7129 cột biểu hiện
            try:
                df = pd.read_csv(path, usecols=lambda c: c in META_COLS)
            except Exception:
                df, err = _safe_read_csv(path)
                if err:
                    rec["issues"] = f"lỗi đọc: {err}"
                    rows.append(rec)
                    continue
        else:
            df, err = _safe_read_csv(path)
            if err:
                rec["issues"] = f"lỗi đọc: {err}"
                rows.append(rec)
                continue
        rec["n_rows"] = int(len(df))
        rec["n_cols"] = int(len(df.columns))
        issues = []
        exp = lot.get("expected_rows")
        if exp is not None and rec["n_rows"] != exp:
            issues.append(f"số dòng {rec['n_rows']} != kỳ vọng {exp}")
        if lot["kind"] == "kaggle_expr":
            issues.extend(_check_kaggle_expr(df))
        elif lot["kind"] == "labels":
            issues.extend(_check_labels(df))
        elif lot["kind"] == "metadata":
            issues.extend(_check_metadata(df))
        rec["issues"] = "; ".join(issues) if issues else ""
        rec["status"] = "PASS" if not issues else "FAIL"
        rows.append(rec)
    return pd.DataFrame(rows)


def validate_new_sample(sample: dict | pd.Series, genes_ref: list[str] | None = None) -> dict[str, Any]:
    """Kiểm tra tính hợp lệ của một mẫu mới trước khi đưa vào pipeline.

    Parameters
    ----------
    sample : dict hoặc pd.Series
        Có thể chứa metadata (sample_id, split, class, subtype, tissue, source, sex)
        và/hoặc vector biểu hiện gen (key là probe_id).
    genes_ref : list[str], optional
        Danh sách probe_id chuẩn (7129 probe). Nếu cung cấp sẽ kiểm tra thiếu/thừa gen.

    Returns
    -------
    dict với keys: valid (bool), errors (list), warnings (list), normalized (dict)
    """
    s = dict(sample)
    errors, warnings = [], []
    norm = {}

    # Chuẩn hóa khóa thường gặp
    key_map = {"Samples": "sample_id", "BM.PB": "tissue", "Gender": "sex", "Source": "source", "cancer": "subtype"}
    for k, v in list(s.items()):
        norm[key_map.get(k, k)] = v

    for col in SAMPLE_REQUIRED:
        if col not in norm or pd.isna(norm[col]) or str(norm[col]).strip() == "":
            errors.append(f"thiếu trường bắt buộc: {col}")

    if "class" in norm and str(norm["class"]).upper() not in {"ALL", "AML"}:
        errors.append(f"class không hợp lệ: {norm['class']} (chỉ ALL/AML)")
    if "split" in norm and str(norm["split"]) not in {"train", "test"}:
        errors.append(f"split không hợp lệ: {norm['split']} (chỉ train/test)")

    st = norm.get("subtype")
    if st in SUBTYPE_MAP:
        norm["subtype"] = SUBTYPE_MAP[st]
        st = norm["subtype"]
    valid_subtypes = set().union(*VALID_SUBTYPE.values())
    if st is not None and st not in valid_subtypes:
        errors.append(f"subtype không hợp lệ: {st}")

    # Kiểm tra quan hệ class-subtype
    cls = str(norm.get("class", "")).upper()
    if cls in VALID_SUBTYPE and st in valid_subtypes and st not in VALID_SUBTYPE.get(cls, set()):
        errors.append(f"class={cls} không khớp subtype={st}")

    if "tissue" in norm and norm["tissue"] not in VALID_TISSUE:
        errors.append(f"tissue không hợp lệ: {norm['tissue']}")
    if "source" in norm and norm["source"] not in VALID_SOURCE:
        errors.append(f"source không hợp lệ: {norm['source']}")
    if "sex" in norm and not pd.isna(norm["sex"]) and norm["sex"] not in VALID_SEX:
        errors.append(f"sex không hợp lệ: {norm['sex']} (chỉ M/F hoặc bỏ trống)")
    if "sex" not in norm or pd.isna(norm.get("sex")):
        warnings.append("sex thiếu (cho phép, nhưng nên ghi rõ)")

    # Kiểm tra vector biểu hiện nếu có
    meta_keys = set(SAMPLE_REQUIRED) | {"sex", "tissue_mf"}
    expr = {k: v for k, v in norm.items() if k not in meta_keys}
    if genes_ref is not None:
        ref = set(genes_ref)
        have = set(expr.keys())
        missing = ref - have
        extra = have - ref
        if missing:
            errors.append(f"thiếu {len(missing)} probe so với danh sách chuẩn ({len(ref)})")
        if extra:
            warnings.append(f"thừa {len(extra)} trường không thuộc probe chuẩn")
        # kiểm tra giá trị số
        bad_vals = 0
        for g in (ref & have):
            try:
                float(expr[g])
            except Exception:
                bad_vals += 1
        if bad_vals:
            errors.append(f"{bad_vals} giá trị biểu hiện không phải số")
    elif expr:
        # có biểu hiện nhưng không có ref -> cảnh báo
        non_numeric = 0
        for v in expr.values():
            try:
                float(v)
            except Exception:
                non_numeric += 1
        if non_numeric:
            errors.append(f"{non_numeric} giá trị biểu hiện không phải số")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings, "normalized": norm}


def save_quality_table(out_dir: Path | None = None) -> Path:
    """Chạy ingest 4 lô và lưu bảng chất lượng ra CSV/MD."""
    out_dir = Path(out_dir) if out_dir else (DATA_DIR / "sandbox" / "ingest_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl = ingest_lots()
    csv_path = out_dir / "ingest_quality_table.csv"
    md_path = out_dir / "ingest_quality_table.md"
    tbl.to_csv(csv_path, index=False)
    try:
        md_table = tbl.to_markdown(index=False)
    except Exception:
        # fallback nếu thiếu tabulate
        header = "| " + " | ".join(tbl.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(tbl.columns)) + " |"
        rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in tbl.to_numpy()]
        md_table = "\n".join([header, sep] + rows)
    lines = ["# Bảng chất lượng 4 lô dữ liệu (ingest)", "", md_table, ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path


def main() -> None:
    print("=== Ingest 4 lô dữ liệu Golub ===\n")
    tbl = ingest_lots()
    # In bảng gọn
    cols = ["lot_id", "name", "exists", "n_rows", "n_cols", "status", "issues"]
    print(tbl[cols].to_string(index=False))
    out_dir = DATA_DIR / "sandbox" / "ingest_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_dir / "ingest_quality_table.csv", index=False)
    try:
        md_table = tbl.to_markdown(index=False)
    except Exception:
        header = "| " + " | ".join(tbl.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(tbl.columns)) + " |"
        rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in tbl.to_numpy()]
        md_table = "\n".join([header, sep] + rows)
    (out_dir / "ingest_quality_table.md").write_text("# Bảng chất lượng 4 lô dữ liệu (ingest)\n\n" + md_table + "\n", encoding="utf-8")
    print(f"\nĐã ghi: {out_dir / 'ingest_quality_table.csv'}")
    # Demo validate_new_sample
    demo = {"sample_id": 999, "split": "test", "class": "AML", "subtype": "AML", "tissue": "BM", "source": "CALGB", "sex": "F"}
    res = validate_new_sample(demo)
    print(f"\nDemo validate_new_sample (hợp lệ): valid={res['valid']}, errors={res['errors']}, warnings={res['warnings']}")
    bad = {"sample_id": 1000, "split": "val", "class": "ALL", "subtype": "AML", "tissue": "XX", "source": "ABC"}
    res2 = validate_new_sample(bad)
    print(f"Demo validate_new_sample (lỗi): valid={res2['valid']}, errors={res2['errors']}")


if __name__ == "__main__":
    main()
