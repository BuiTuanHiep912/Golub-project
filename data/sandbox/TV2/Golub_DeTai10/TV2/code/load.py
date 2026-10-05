"""
TV2 - GĐ1: Mục 1.1 Nạp, ghép dữ liệu và kiểm tra chất lượng.

Quy trình ghép theo mục 2.1.2: ghép theo MÃ BỆNH NHÂN, không bao giờ theo thứ tự dòng.

Nguồn (data/raw/):
  - kaggle/data_set_ALL_AML_train.csv      : ma trận biểu hiện + call, patient 1..38
  - kaggle/data_set_ALL_AML_independent.csv: ma trận biểu hiện + call, patient 39..72
  - kaggle/actual.csv                      : nhãn ALL/AML (theo mã bệnh nhân) - NGUỒN NHÃN CHÍNH
  - openintro/golub_metadata.csv           : metadata mẫu (xem docs/decisions.md ADR-001)

Đầu ra data/standardized/:
  - expression.parquet : 72 x 7129, index = sample_id (int = mã bệnh nhân), cột = probe, int
  - calls.parquet      : 72 x 7129, giá trị P/A/M (có thể NA) - để tham khảo, không dùng mô hình
  - samples.parquet    : 72 x 6 (split, class, subtype, tissue, source, sex), index = sample_id
  - genes.parquet      : probe_id, description, is_control (tiền tố AFFX) - cho TV4/TV5
  - data/raw/SOURCES.md: MD5 từng file raw + nguồn tải
"""
from __future__ import annotations

import hashlib

import pandas as pd

try:
    from src.config import (
        DATA_DIR, RAW_DIR, STD_DIR, KAGGLE_DIR, OPEN_INTRO_DIR,
    )
except ModuleNotFoundError:  # module chạy như bản làm việc trong TV2/code
    from config import (
        DATA_DIR, RAW_DIR, STD_DIR, KAGGLE_DIR, OPEN_INTRO_DIR,
    )

KAGGLE_TRAIN = KAGGLE_DIR / "data_set_ALL_AML_train.csv"
KAGGLE_INDEP = KAGGLE_DIR / "data_set_ALL_AML_independent.csv"
ACTUAL_FILE = KAGGLE_DIR / "actual.csv"

# 6 cột mô tả lấy từ metadata gốc (dataset_more.csv -> golub_metadata.csv)
META_FILE = OPEN_INTRO_DIR / "golub_metadata.csv"
META_COL_RENAME = {
    "Samples": "sample_id",
    "BM.PB": "tissue",
    "Gender": "sex",
    "Source": "source",
    "cancer": "subtype",       # allB / allT / aml
    "tissue.mf": "morphology",  # tài liệu, không dùng trong mô hình
}
SUBTYPE_MAP = {"allB": "B-ALL", "allT": "T-ALL", "aml": "AML"}

# Thứ tự cột chuẩn của samples.parquet (File 1, Bảng 2.3a)
SAMPLE_COLS = ["split", "class", "subtype", "tissue", "source", "sex"]

DESC_COL = "Gene Description"
PROBE_COL = "Gene Accession Number"


def md5_file(path) -> str:
    """MD5 checksum cho SOURCES.md (mục 2.1)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sources() -> None:
    """Ghi data/raw/SOURCES.md kèm MD5 và nguồn tải từng file (mục 2.2, Bảng 2.5)."""
    sources = _source_notes()
    lines = ["# SOURCES.md - Nguồn dữ liệu và checksum", ""]
    for f in sorted(RAW_DIR.rglob("*")):
        if f.is_file() and f.suffix in {".csv", ".txt", ".md"}:
            rel = f.relative_to(RAW_DIR)
            note = sources.get(f.name, "")
            lines.append(f"- `{rel}` MD5: `{md5_file(f)}`  {note}")
    lines += [
        "",
        "## Quy ước",
        "- Nhãn ALL/AML lấy theo `actual.csv` (Kaggle) - bộ dữ liệu chính (mục 2.1.2).",
        "- Metadata mẫu lấy từ `golub_metadata.csv`, mapping theo ADR-001 (docs/decisions.md).",
        "- Chi tiết từng cột: docs/data_dictionary.md; phân bố mẫu đối chiếu Golub 1999 Bảng 1.",
        "- Bài báo gốc (PDF tham khảo): docs/nguon/Golub_1999_Molecular_Classification_of_Cancer.pdf"
        " (MD5 1d65f9500df35892ce169297b5209c4b, ADR-005).",
    ]
    (RAW_DIR / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")


def _source_notes() -> dict:
    return {
        "data_set_ALL_AML_train.csv": "(Kaggle: Golub leukemia ALL_AML)",
        "data_set_ALL_AML_independent.csv": "(Kaggle: Golub leukemia ALL_AML)",
        "actual.csv": "(Kaggle, nhãn patient->cancer)",
        "golub_metadata.csv": "(metadata mẫu, nguồn: dataset_more.csv - xem ADR-001)",
    }


# ---------------------------------------------------------------------------
# Đọc và định hình lại file Kaggle (bước 1-2 của mục 2.1.2)
# ---------------------------------------------------------------------------
def _read_kaggle(path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc 1 file Kaggle -> (expr-T, calls-T, gene_meta).

    File Kaggle dạng probe-theo-HÀNG, mỗi bệnh nhân 2 cột xen kẽ: `k` (biểu hiện) và
    `k(call)` (hiện diện P/A/M). Bước 1: bỏ 2 cột mô tả + mọi cột call, chuyển vị để
    mỗi HÀNG là một bệnh nhân, tên cột đổi thành mã probe.
    """
    raw = pd.read_csv(path)
    desc_cols = {DESC_COL, PROBE_COL}
    expr_cols = [c for c in raw.columns if c not in desc_cols and "call" not in c.lower()]
    call_cols = [c for c in raw.columns if c not in desc_cols and "call" in c.lower()]

    probe_ids = raw[PROBE_COL].astype(str)
    if not probe_ids.is_unique:
        raise ValueError(f"probe trùng trong {path.name}")

    expr = raw[expr_cols].T.copy()
    expr.columns = probe_ids.values
    expr.index = pd.Index([int(c) for c in expr_cols], name="sample_id")

    calls = raw[call_cols].T.copy()
    calls.columns = probe_ids.values
    calls.index = expr.index

    gene_meta = pd.DataFrame({"probe_id": probe_ids.values, "description": raw[DESC_COL].values})
    return expr, calls, gene_meta


def _load_kaggle_expression() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gộp train + independent thành 1 bảng 72 hàng, sắp xếp theo mã bệnh nhân (bước 1-2)."""
    frames, call_frames, meta_frames = [], [], []
    for path, split in ((KAGGLE_TRAIN, "train"), (KAGGLE_INDEP, "test")):
        if not path.exists():
            raise FileNotFoundError(
                f"Thiếu {path}. Chạy `make data` (sao chép dataset/ sang data/raw/) trước."
            )
        e, c, m = _read_kaggle(path)
        e["split"] = split
        frames.append(e)
        call_frames.append(c)
        meta_frames.append(m)

    expr = pd.concat(frames).sort_index()
    calls = pd.concat(call_frames).sort_index()
    gene_meta = meta_frames[0]
    for m in meta_frames[1:]:
        if not m["probe_id"].equals(gene_meta["probe_id"]):
            raise ValueError("hai file Kaggle không cùng danh sách probe")
    gene_meta = gene_meta.reset_index(drop=True)
    gene_meta["is_control"] = gene_meta["probe_id"].str.startswith("AFFX")
    return expr, calls, gene_meta


def _load_labels() -> pd.Series:
    """Nhãn từ actual.csv, theo mã bệnh nhân (='patient')."""
    lab = pd.read_csv(ACTUAL_FILE)
    lab.columns = [c.strip().lower() for c in lab.columns]
    col = "cancer" if "cancer" in lab.columns else "class"
    out = lab.set_index("patient")[col].astype(str).str.upper().str.strip().rename("class")
    out.index.name = "sample_id"
    return out


def _load_metadata() -> pd.DataFrame:
    """Đọc metadata 6 cột mô tả + map subtype (bước 4). Không chấp nhận file thiếu."""
    if not META_FILE.exists():
        raise FileNotFoundError(
            f"Thiếu metadata {META_FILE} (nên là bản copy của dataset_more.csv). "
            "Xem docs/decisions.md ADR-001."
        )
    meta = pd.read_csv(META_FILE).rename(columns=META_COL_RENAME)
    missing = [c for c in META_COL_RENAME.values() if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata thiếu cột: {missing}")
    meta = meta.set_index("sample_id")
    meta["subtype"] = meta["subtype"].map(SUBTYPE_MAP)
    return meta


def build() -> pd.DataFrame:
    """Toàn bộ quy trình ghép (bước 1-7) -> sinh 4 parquet tại data/standardized."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    OPEN_INTRO_DIR.mkdir(parents=True, exist_ok=True)
    STD_DIR.mkdir(parents=True, exist_ok=True)

    # 1-2. Biểu hiện + split + calls + gene_meta từ Kaggle
    expr, calls, gene_meta = _load_kaggle_expression()
    split = expr["split"]
    X = expr.drop(columns=["split"]).astype(int)
    calls = calls.astype(object)

    # 3. Nhãn theo mã bệnh nhân
    labels = _load_labels()

    # 4-5. Metadata + ghép left-join một-một, validate one_to_one
    meta = _load_metadata()
    samples = meta.join(labels, how="outer", validate="one_to_one")

    samples["split"] = split
    samples = samples[SAMPLE_COLS].sort_index()
    samples.index.name = "sample_id"

    # 6. Kiểm tra sau ghép (Bảng 2.4) - báo lỗi rõ để nhóm xử lý, không sửa tay
    _assert_post_merge(None, samples)

    # Tách ma trận biểu hiện và ghi
    # Ghi
    if X.shape != (72, 7129):
        raise ValueError(f"kích thước biểu hiện lệch: {X.shape}, kỳ vọng (72, 7129)")
    if X.isna().any().any():
        raise ValueError("ma trận biểu hiện có NaN (kiểm tra file Kaggle gốc)")

    X.to_parquet(STD_DIR / "expression.parquet", index=True)
    calls.to_parquet(STD_DIR / "calls.parquet", index=True)
    samples.to_parquet(STD_DIR / "samples.parquet", index=True)
    gene_meta.to_parquet(STD_DIR / "genes.parquet", index=False)
    write_sources()

    print(f"Đã ghi 4 parquet vào {STD_DIR}")
    print(f"- expression {X.shape} (int)  | calls {calls.shape}")
    print(f"- samples {samples.shape}: {list(samples.columns)}")
    print(f"- genes {gene_meta.shape} (AFFX control: {int(gene_meta['is_control'].sum())})")
    _print_summary(samples)
    return samples


def _assert_post_merge(_unused, samples: pd.DataFrame) -> None:
    """Các kiểm tra Bắt buộc sau khi ghép → nếu lệch, in rõ và ghi decisions.md (2.1.2)."""
    checks = {
        "Số mẫu": len(samples) == 72,
        "Không mẫu thiếu class": samples["class"].notna().all(),
        "Không mẫu thiếu subtype": samples["subtype"].notna().all(),
        "Một-một giữa metadata và nhãn": samples.index.is_unique,
    }
    miss = {k for k, ok in checks.items() if not ok}
    if miss:
        raise ValueError(f"Kiểm tra sau ghép thất bại: {miss} — xem docs/decisions.md")


def _print_summary(samples: pd.DataFrame) -> None:
    print("\nCrosstab split x class:")
    print(pd.crosstab(samples["split"], samples["class"]).to_string())
    print("\nCrosstab split x tissue:")
    print(pd.crosstab(samples["split"], samples["tissue"]).to_string())
    print("\nCrosstab split x class x source:")
    print(pd.crosstab([samples["split"], samples["class"]], samples["source"]).to_string())


# ---------------------------------------------------------------------------
# API cho cả nhóm
# ---------------------------------------------------------------------------
def load_golub(layer: str = "cleansed") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nạp expression + samples từ tầng cleansed (mặc định) hoặc standardized.

    Returns:
        X      : (72, 7129), index = sample_id (int), cột = probe id
        samples: bảng mô tả mẫu cùng chỉ mục sample_id
    """
    base = DATA_DIR / "cleansed" if layer == "cleansed" else STD_DIR
    X = pd.read_parquet(base / "expression.parquet")
    samples = pd.read_parquet(base / "samples.parquet")
    X.index = X.index.astype(int)
    samples.index = samples.index.astype(int)
    return X, samples


if __name__ == "__main__":
    build()