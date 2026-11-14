"""
TV4 - Danh mục mô tả toàn bộ 7129 probe của tầng raw (phụ lục cho docs/report/description.txt).

Đọc trực tiếp hai file Kaggle (data/raw/kaggle/), ghép 72 bệnh nhân theo tên cột, rồi với mỗi
probe ghi: định danh, tên gen tách từ mô tả, loại probe, nhóm điều khiển, và thống kê mô tả
trên thang gốc MAS 4.0. Không dùng nhãn lớp -> đây chỉ là mô tả, không phải phân tích gen khác biệt.

Đầu ra: docs/report/gene_catalog.csv (UTF-8 có BOM để mở thẳng bằng Excel).
Cách chạy: python -m src.gene_catalog
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

try:
    from src.config import DOCS_DIR, KAGGLE_DIR
except ModuleNotFoundError:  # chạy trực tiếp `python src/gene_catalog.py`
    from config import DOCS_DIR, KAGGLE_DIR

RAW_FILES = [KAGGLE_DIR / "data_set_ALL_AML_train.csv",
             KAGGLE_DIR / "data_set_ALL_AML_independent.csv"]
OUT_FILE = DOCS_DIR / "report" / "gene_catalog.csv"

# Ngưỡng giống src/preprocess.py (Dudoit et al., 2002)
FLOOR, CEIL = 100, 16000
MIN_RATIO, MIN_DIFF = 5, 500

PROBE_TYPES = {  # hậu tố -> ý nghĩa theo quy ước Affymetrix
    "_s_at": "dùng chung cho nhiều bản phiên mã/gen cùng họ",
    "_f_at": "thuộc họ gen, có thể lai chéo với gen tương tự",
    "_r_at": "thiết kế với quy tắc nới lỏng, độ đặc hiệu thấp hơn",
    "_st": "probe mạch xuôi (sense), đối chứng âm",
    "_at": "probe tiêu chuẩn, đặc hiệu cho một bản phiên mã",
}
CONTROL_GROUPS = {  # tên trong mã AFFX -> nhóm đối chứng
    "BioB": "lai (E. coli)", "BioC": "lai (E. coli)", "BioDn": "lai (E. coli)",
    "CreX": "lai (phage P1)",
    "DapX": "poly-A (B. subtilis)", "LysX": "poly-A (B. subtilis)", "PheX": "poly-A (B. subtilis)",
    "ThrX": "poly-A (B. subtilis)", "TrpnX": "poly-A (B. subtilis)",
}
# Ký hiệu gen HUGO: chữ in hoa + số, có thể có '-' (RPS11, HOXA9, KIAA0008, HLA-B)
SYMBOL_RE = re.compile(r"^([A-Z][A-Z0-9-]{1,14})\s+(.+)$")


def read_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trả về (bảng mô tả probe, biểu hiện probe x 72, call probe x 72), cột = mã bệnh nhân."""
    expr, calls = [], []
    for f in RAW_FILES:
        d = pd.read_csv(f)
        # cột biểu hiện là mã bệnh nhân; cột "call.k" đi ngay sau cột biểu hiện tương ứng
        cols = list(d.columns)
        pid = [c for c in cols if c.isdigit()]
        expr.append(d[pid].set_axis([int(c) for c in pid], axis=1))
        calls.append(d[[cols[cols.index(c) + 1] for c in pid]].set_axis([int(c) for c in pid], axis=1))
        desc = d[["Gene Accession Number", "Gene Description"]]
    expr = pd.concat(expr, axis=1).sort_index(axis=1)
    calls = pd.concat(calls, axis=1).sort_index(axis=1)
    desc = desc.set_axis(["probe_id", "description"], axis=1)
    expr.index = calls.index = desc["probe_id"]
    return desc.set_index("probe_id"), expr, calls


def split_symbol(description: str) -> tuple[str, str]:
    """Tách 'RPS11 Ribosomal protein S11' -> ('RPS11', 'Ribosomal protein S11').

    Chỉ nhận là ký hiệu gen khi phần còn lại không viết hoa toàn bộ, để không cắt nhầm
    tên viết hoa như 'PROTO-ONCOGENE TYROSINE-PROTEIN KINASE SRC'.
    """
    if description.startswith("GB DEF = "):
        return "", description[len("GB DEF = "):]
    m = SYMBOL_RE.match(description)
    if m and not m.group(1).startswith("AFFX") and m.group(2) != m.group(2).upper():
        # 'KIAA0207 gene', 'YPT3 mRNA': phần sau không phải tên gen -> giữ nguyên mô tả
        name = description if m.group(2) in ("gene", "mRNA") else m.group(2)
        return m.group(1), name
    return "", description


def accession_info(probe_id: str) -> tuple[str, str, str]:
    """Trả về (mã truy cập gốc, cơ sở dữ liệu, loại probe)."""
    suffix = next(s for s in PROBE_TYPES if probe_id.endswith(s))
    core = probe_id[: -len(suffix)]
    if probe_id.startswith("AFFX"):
        gb = re.search(r"/([A-Z]+\d+)", core)  # AFFX-HUMGAPDH/M33197_5 -> M33197
        return (gb.group(1) if gb else ""), "Affymetrix control", suffix
    if core.startswith("HG"):
        return core, "Affymetrix/TIGR gene cluster", suffix
    return core, "GenBank", suffix


def control_group(probe_id: str) -> str:
    if not probe_id.startswith("AFFX"):
        return ""
    name = re.match(r"AFFX-(?:HUM|HS)?([A-Za-z0-9]+)", probe_id).group(1)
    return CONTROL_GROUPS.get(name, "gen giữ nhà của người")


def build_catalog() -> pd.DataFrame:
    desc, expr, calls = read_raw()
    cat = desc.copy()
    cat[["gene_symbol", "gene_name"]] = [split_symbol(s) for s in cat["description"]]
    cat[["accession", "accession_db", "probe_type"]] = [accession_info(p) for p in cat.index]
    cat["probe_type_meaning"] = cat["probe_type"].map(PROBE_TYPES)
    cat["is_control"] = cat.index.str.startswith("AFFX")
    cat["control_group"] = [control_group(p) for p in cat.index]
    cat["n_probes_same_description"] = cat.groupby("description")["description"].transform("size")

    cat["expr_min"] = expr.min(axis=1)
    cat["expr_median"] = expr.median(axis=1)
    cat["expr_max"] = expr.max(axis=1)
    cat["pct_negative"] = (expr < 0).mean(axis=1).mul(100).round(1)
    for c, name in (("P", "pct_present"), ("M", "pct_marginal"), ("A", "pct_absent")):
        cat[name] = (calls == c).mean(axis=1).mul(100).round(1)

    clipped = expr.clip(FLOOR, CEIL)
    hi, lo = clipped.max(axis=1), clipped.min(axis=1)
    cat["pass_filter_72"] = (hi / lo > MIN_RATIO) & (hi - lo > MIN_DIFF)
    return cat.reset_index()


def main() -> None:
    cat = build_catalog()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {OUT_FILE} ({len(cat)} probe x {cat.shape[1]} cột)")
    print(f"  có ký hiệu gen: {(cat['gene_symbol'] != '').sum()}"
          f" | probe điều khiển: {cat['is_control'].sum()}"
          f" | qua bộ lọc (72 mẫu): {cat['pass_filter_72'].sum()}")


if __name__ == "__main__":
    main()
