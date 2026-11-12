"""Nhập dữ liệu theo lô (mỗi phòng lab một lô), chất lượng theo lô, kiểm tra file mẫu mới.

Công việc: 2.4 — Phụ trách: TV2 — Kế hoạch giai đoạn 2, mục 2.4 (ADR-014, ADR-030).
Dùng cho: TV3 (bảng fact_qc, dim_batch trong kho), TV6 (màn hình Chất lượng dữ liệu,
nút tải mẫu), TV1 (mẫu mới đã kiểm tra cho patient_profile).

Giao diện đã thống nhất, KHÔNG đổi tên khi chưa báo nhóm:
    ingest_batches()          -> DataFrame chất lượng theo lô, một hàng mỗi lô:
                                 [batch_id, source, n_samples, status ('passed'|'blocked'),
                                  reason, ...chỉ số chất lượng...]
    validate_new_sample(path) -> dict {'ok': bool, 'errors': list[str],
                                       'warnings': list[str], 'values': Series | None}
Thêm (ADR-030):
    sample_quality()          -> DataFrame chất lượng theo mẫu (bảng fact_qc của kho)

Mô hình lô (ADR-030):
  - Một lô = mọi mẫu do một phòng lab gửi (cột `source`). Bốn lô thật B1-DFCI, B2-CALGB,
    B3-St-Jude, B4-CCG tách từ tầng standardized; có thể nhập thêm lô từ file CSV
    (`extra_files`, cùng định dạng bảng mẫu + 7129 cột probe) để demo lô lỗi bị chặn.
  - Lô bị chặn ('blocked') khi vi phạm ràng buộc cấu trúc hoặc quan hệ: thiếu/thừa probe, ô
    trống, giá trị không phải số, sample_id trùng, metadata thiếu hoặc ngoài danh mục, class lệch
    subtype, giá trị call ngoài P/A/M. Chặn một lô không ảnh hưởng các lô khác.
  - Chỉ số chất lượng (tương quan, tỷ lệ P, tỷ lệ ngoài [100, 16000], qc_outlier) chỉ để báo
    cáo, KHÔNG chặn lô: cờ của vài mẫu không đủ căn cứ loại cả lô (cùng tinh thần ADR-004).
  - Dữ liệu tham chiếu: chỉ số của mẫu ở lô thêm được đo so với các lô thật, để lô demo không
    làm thay đổi cờ qc_outlier của 72 mẫu thật (cờ này phải trùng tầng cleansed).

Đầu ra của `make ingest` (write=True), đặt ở tầng cleansed cạnh quality_report.md:
  batches.parquet     -> bảng dim_batch của kho (TV3): một hàng mỗi lô
  qc_samples.parquet  -> bảng fact_qc của kho (TV3): một hàng mỗi mẫu
  ingest_log.csv      -> nhật ký nhập: mỗi lần chạy GHI THÊM một dòng mỗi lô, không ghi đè

Bộ file lỗi mẫu để demo (ADR-031): write_demo_files() sinh từ dữ liệu thật vào data/demo/
(không đưa lên Git, tạo lại được) các file chip một mẫu và file lô, mỗi file một tình huống, kèm
README.md ghi kết quả mong đợi.

Chạy: make ingest  (Windows: python run_all.py ingest)
      python -m src.ingest <file_lo_them.csv ...>   # nhập thêm lô (ghi cả vào bảng và nhật ký)
      python -m src.ingest --demo                   # sinh data/demo/ và chạy thử, không ghi bảng
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import CLEANSED_DIR, DATA_DIR, STD_DIR, VALID_SEX, VALID_SOURCE, VALID_SUBTYPE, VALID_TISSUE
    from src.load import REQUIRED_COLS, SAMPLE_COLS
    from src.quality import EXPR_HI, EXPR_LO, MODZ_THRESHOLD, sample_correlation, sample_qc
except ModuleNotFoundError:  # chạy trực tiếp `python src/ingest.py`
    from config import CLEANSED_DIR, DATA_DIR, STD_DIR, VALID_SEX, VALID_SOURCE, VALID_SUBTYPE, VALID_TISSUE
    from load import REQUIRED_COLS, SAMPLE_COLS
    from quality import EXPR_HI, EXPR_LO, MODZ_THRESHOLD, sample_correlation, sample_qc

BATCH_SOURCES = ("DFCI", "CALGB", "St-Jude", "CCG")

# Tên file đầu ra (tương đối với out_dir); TV3 import các hằng này thay vì ghi cứng đường dẫn.
BATCHES_FILE = "batches.parquet"
QC_FILE = "qc_samples.parquet"
LOG_FILE = "ingest_log.csv"

# Bộ file demo nằm trong data/ nên theo .gitignore không lên Git; sinh lại bằng --demo.
DEMO_DIR = DATA_DIR / "demo"
# Bệnh nhân tập kiểm tra đóng vai bệnh nhân mới (ADR-014); 52 là ví dụ của patient_profile.
DEMO_PATIENT = 52

VALID_SPLIT = {"train", "test"}
VALID_CLASS = set(VALID_SUBTYPE)
VALID_CALLS = {"P", "A", "M"}

# File chip của một mẫu mới: CSV đúng hai cột này, mỗi dòng một probe.
SAMPLE_FILE_COLS = ("probe_id", "value")

# NOTICE: ngưỡng phát hiện file đã log hóa / sai thang đo. Trên 72 chip thật, tỷ lệ giá trị
# < 100 của từng mẫu nằm trong [0.40, 0.56] (thang MAS4 gốc). File đã log10 có ~100% giá trị
# < 100 (log10 của 16000 chỉ là 4.2), nên ngưỡng 0.9 tách hai trường hợp với biên rất rộng.
# Đổi ngưỡng thì ghi vào docs/decisions.md (ADR-030).
LOG_SCALE_SHARE = 0.9

BATCH_COLS = ["batch_id", "source", "n_samples", "status", "reason", "warnings", "n_train", "n_test",
              "n_ALL", "n_AML", "pct_below_100", "pct_above_16000", "present_rate", "median_corr_other",
              "n_qc_outlier", "origin"]
QC_COLS = ["sample_id", "batch_id", "source", "split", "class", "median_log10", "iqr_log10", "median_corr",
           "median_corr_other", "present_rate", "pct_below_100", "pct_above_16000", "qc_outlier", "qc_reason",
           "batch_status"]


def _preview(items, limit: int = 5) -> str:
    """Liệt kê tối đa `limit` phần tử cho thông báo lỗi ngắn gọn.

    Before:
    - [3, 7, 9, 11, 12, 15, 20]

    After:
    - "[3, 7, 9, 11, 12, …(+2)]"
    """
    items = list(items)
    shown = ", ".join(str(x) for x in items[:limit])
    more = f", …(+{len(items) - limit})" if len(items) > limit else ""
    return f"[{shown}{more}]"


# ---------------------------------------------------------------------------
# Đọc các lô
# ---------------------------------------------------------------------------
def _real_batches(std_dir: Path) -> tuple[list[dict], pd.Index]:
    """Tách tầng standardized thành các lô theo source; trả về (lô, danh sách probe chuẩn)."""
    req = {n: std_dir / f"{n}.parquet" for n in ("expression", "calls", "samples", "genes")}
    missing = [p.name for p in req.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu tầng standardized: {missing} trong {std_dir} — chạy `make standardize` trước.")
    X = pd.read_parquet(req["expression"])
    calls = pd.read_parquet(req["calls"])
    samples = pd.read_parquet(req["samples"])
    ref_probes = pd.Index(pd.read_parquet(req["genes"])["probe_id"])
    for frame in (X, calls, samples):
        frame.index = frame.index.astype(int)

    # Nguồn ngoài BATCH_SOURCES (hoặc thiếu) vẫn thành một lô riêng để bị chặn ở bước kiểm
    # danh mục, thay vì lặng lẽ bị bỏ khỏi quá trình nhập.
    source = samples["source"].fillna("(thiếu)")
    order = [s for s in BATCH_SOURCES if s in set(source)] + sorted(set(source) - set(BATCH_SOURCES))
    batches = []
    for i, src in enumerate(order, start=1):
        ids = samples.index[source == src]
        batches.append({"batch_id": f"B{i}-{src}", "source": src, "origin": "standardized", "real": True,
                        "meta": samples.loc[ids], "X": X.loc[ids], "calls": calls.loc[ids]})
    return batches, ref_probes


def _file_batch(path) -> dict:
    """Đọc một lô thêm từ CSV: cột sample_id + bảng mẫu (SAMPLE_COLS) + các cột probe.

    Đọc mọi ô dưới dạng chuỗi để phân biệt được ô trống với giá trị không phải số; việc kiểm
    tra nằm ở _check_batch, hàm này chỉ định hình lại.
    """
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    meta_cols = [c for c in SAMPLE_COLS if c in df.columns]
    probe_cols = [c for c in df.columns if c != "sample_id" and c not in SAMPLE_COLS]
    ids = df["sample_id"] if "sample_id" in df.columns else pd.Series([""] * len(df))
    meta = df[meta_cols].replace("", np.nan).copy()
    meta.insert(0, "sample_id_raw", ids.to_numpy())
    sources = sorted(meta["source"].dropna().unique()) if "source" in meta.columns else []
    if len(sources) == 1:
        source = sources[0]
    else:
        source = "(thiếu)" if not sources else "(nhiều nguồn)"
    return {"batch_id": path.stem, "source": source,
            "origin": path.name, "real": False, "meta": meta, "X": df[probe_cols], "calls": None}


# ---------------------------------------------------------------------------
# Kiểm tra một lô
# ---------------------------------------------------------------------------
def _numeric_cells(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(giá trị số, mặt nạ ô trống, mặt nạ ô không phải số) — làm theo mảng để nhanh với 7129 cột."""
    if all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        num = X.astype(float)
        return num, num.isna().to_numpy(), np.zeros(X.shape, dtype=bool)
    arr = X.to_numpy(dtype=object)
    text = np.char.strip(arr.astype(str))
    empty = pd.isna(arr) | (text == "")
    num = pd.to_numeric(pd.Series(text.ravel()), errors="coerce").to_numpy().reshape(arr.shape)
    return pd.DataFrame(num, index=X.index, columns=X.columns), empty, np.isnan(num) & ~empty


def _check_batch(batch: dict, ref_probes: pd.Index, seen_ids: set) -> tuple[list[str], list[str], pd.DataFrame | None]:
    """Kiểm tra ràng buộc của một lô -> (lỗi chặn lô, cảnh báo, X số theo probe chuẩn | None).

    X trả về None khi không tính được chỉ số chất lượng: biểu hiện hỏng cấu trúc (thiếu/thừa
    probe, ô trống, không phải số) hoặc sample_id không dùng làm khóa được (trống, trùng).
    Các dòng của meta và X luôn cùng thứ tự, nên mặt nạ theo dòng dùng chung cho cả hai.
    """
    problems, warns = [], []
    meta, X = batch["meta"], batch["X"]
    if len(meta) == 0:
        return ["lô rỗng (không có mẫu)"], warns, None

    # Mã mẫu từng dòng: lô thật đã có chỉ mục số nguyên; lô từ file phải tự kiểm.
    shown = pd.Series(meta.index.to_numpy()) if batch["real"] else meta["sample_id_raw"].reset_index(drop=True)
    ids = pd.to_numeric(shown, errors="coerce")
    bad_id = (ids.isna() | (ids % 1 != 0)).to_numpy()

    def who(rows) -> str:
        return _preview(shown[np.asarray(rows)])

    if bad_id.any():
        problems.append(f"sample_id trống hoặc không phải số nguyên ở {int(bad_id.sum())} dòng")
    valid_ids = ids[~bad_id].astype(int)
    dup = sorted(set(valid_ids[valid_ids.duplicated()]))
    if dup:
        problems.append(f"sample_id trùng trong lô: {_preview(dup)}")
    reused = sorted(set(valid_ids) & seen_ids)
    if reused:
        problems.append(f"sample_id đã có ở lô khác: {_preview(reused)}")

    # Metadata bắt buộc và danh mục (Bảng 2.2); sex được phép thiếu (ADR-001).
    for col in REQUIRED_COLS:
        if col not in meta.columns:
            problems.append(f"thiếu cột {col}")
        elif meta[col].isna().any():
            problems.append(f"thiếu {col} ở mẫu {who(meta[col].isna())}")
    catalogs = {"split": VALID_SPLIT, "class": VALID_CLASS, "subtype": set().union(*VALID_SUBTYPE.values()),
                "tissue": VALID_TISSUE, "source": VALID_SOURCE, "sex": VALID_SEX}
    for col, allowed in catalogs.items():
        if col in meta.columns:
            bad_vals = sorted(set(meta[col].dropna()) - allowed)
            if bad_vals:
                problems.append(f"{col} ngoài danh mục: {bad_vals}")
    if "source" in meta.columns and meta["source"].nunique() > 1:
        problems.append(f"lô chứa nhiều nguồn: {sorted(meta['source'].dropna().unique())}")
    if {"class", "subtype"} <= set(meta.columns):
        # class thiếu hoặc lạ -> không có tập subtype hợp lệ -> tính là lệch
        mismatch = [s not in VALID_SUBTYPE.get(c, set()) for c, s in zip(meta["class"], meta["subtype"])]
        if any(mismatch):
            problems.append(f"class lệch subtype ở mẫu {who(mismatch)}")
    if "sex" in meta.columns and meta["sex"].isna().any():
        warns.append(f"thiếu sex ở {int(meta['sex'].isna().sum())}/{len(meta)} mẫu (cho phép, ADR-001)")

    # Cấu trúc biểu hiện: đúng bộ probe chuẩn, không ô trống, toàn số.
    cols = pd.Index(X.columns)
    structural_ok = True
    if cols.duplicated().any():
        problems.append(f"probe trùng: {_preview(sorted(set(cols[cols.duplicated()])))}")
        structural_ok = False
    missing, extra = ref_probes.difference(cols), cols.difference(ref_probes)
    if len(missing):
        problems.append(f"thiếu {len(missing)} probe so với {len(ref_probes)} probe chuẩn: {_preview(missing)}")
        structural_ok = False
    if len(extra):
        problems.append(f"thừa {len(extra)} cột không phải probe chuẩn: {_preview(extra)}")
        structural_ok = False
    num, empty, non_numeric = _numeric_cells(X)
    for mask, what in ((empty, "ô biểu hiện trống"), (non_numeric, "giá trị biểu hiện không phải số")):
        if mask.any():
            problems.append(f"{int(mask.sum())} {what} ở mẫu {who(mask.any(axis=1))}")
            structural_ok = False
    if batch["calls"] is not None:
        seen_calls = pd.Series(batch["calls"].to_numpy().ravel()).dropna().unique()
        bad_calls = sorted(set(seen_calls) - VALID_CALLS)
        if bad_calls:
            problems.append(f"giá trị call ngoài P/A/M: {bad_calls}")

    if not structural_ok or bad_id.any() or dup or reused:
        return problems, warns, None
    num = num[list(ref_probes)]
    num.index = pd.Index(valid_ids.to_numpy(), name="sample_id")
    return problems, warns, num


# ---------------------------------------------------------------------------
# Chỉ số chất lượng và bảng kết quả
# ---------------------------------------------------------------------------
def _assess(extra_files=None, std_dir: Path = STD_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kiểm tra mọi lô (thật + thêm) -> (bảng theo lô, bảng theo mẫu). Không ghi file."""
    batches, ref_probes = _real_batches(std_dir)
    batches += [_file_batch(p) for p in (extra_files or [])]

    # seen: sample_id đã thuộc một lô trước đó -> lô sau dùng lại mã đó bị chặn.
    seen: set = set()
    checked = []
    for b in batches:
        problems, warns, num = _check_batch(b, ref_probes, seen)
        if b["real"]:
            seen |= set(b["meta"].index)
        elif num is not None:
            seen |= set(num.index)
        checked.append((b, problems, warns, num))

    # Chỉ số theo mẫu. Lô thật đo với nhau (cờ khớp đúng tầng cleansed); mỗi lô thêm đo so với
    # các lô thật để lô demo không làm đổi cờ của 72 mẫu thật.
    real_parts = [num for b, _, _, num in checked if b["real"] and num is not None]
    if not real_parts:
        raise ValueError("Không lô thật nào đọc được cấu trúc biểu hiện — kiểm tra tầng standardized")
    real_X = pd.concat(real_parts)
    real_qc, real_corr = sample_qc(real_X), sample_correlation(real_X)
    per_sample = {}
    for b, _, _, num in checked:
        if num is None:
            continue
        if b["real"]:
            q, corr = real_qc.loc[num.index].copy(), real_corr
        else:
            both = pd.concat([real_X, num])
            q, corr = sample_qc(both).loc[num.index].copy(), sample_correlation(both)
        # Tương quan trung vị với mẫu của các lô THẬT khác: lô lệch hẳn phần còn lại (batch effect,
        # quy trình lab khác) có giá trị thấp ở mọi mẫu của nó.
        others = real_X.index.difference(num.index)
        q["median_corr_other"] = corr.loc[num.index, others].median(axis=1, skipna=True)
        q["pct_below_100"] = (num < EXPR_LO).mean(axis=1)
        q["pct_above_16000"] = (num > EXPR_HI).mean(axis=1)
        q["present_rate"] = (b["calls"].loc[num.index] == "P").mean(axis=1) if b["calls"] is not None else np.nan
        per_sample[b["batch_id"]] = q

    batch_rows, qc_rows = [], []
    for b, problems, warns, num in checked:
        status = "blocked" if problems else "passed"
        meta = b["meta"]
        q = per_sample.get(b["batch_id"])
        if q is not None and q["qc_outlier"].any():
            warns = warns + [f"mẫu gắn cờ qc_outlier: {_preview(q.index[q['qc_outlier']])}"]

        def count(col: str, val: str) -> int:
            return int((meta[col] == val).sum()) if col in meta.columns else 0

        batch_rows.append({
            "batch_id": b["batch_id"], "source": b["source"], "n_samples": len(meta), "status": status,
            "reason": "; ".join(problems), "warnings": "; ".join(warns),
            "n_train": count("split", "train"), "n_test": count("split", "test"),
            "n_ALL": count("class", "ALL"), "n_AML": count("class", "AML"),
            # Lô hỏng cấu trúc không có chỉ số: để NaN chứ không phải 0, tránh đọc nhầm là "tốt".
            "pct_below_100": float(q["pct_below_100"].mean()) if q is not None else np.nan,
            "pct_above_16000": float(q["pct_above_16000"].mean()) if q is not None else np.nan,
            "present_rate": float(q["present_rate"].median()) if q is not None else np.nan,
            "median_corr_other": float(q["median_corr_other"].median()) if q is not None else np.nan,
            "n_qc_outlier": int(q["qc_outlier"].sum()) if q is not None else 0,
            "origin": b["origin"],
        })
        if q is None:
            continue  # không có chỉ số theo mẫu để đưa vào fact_qc
        rows = q.reset_index(names="sample_id")
        meta_by_id = meta.set_index(num.index)  # cùng thứ tự dòng với num (xem _check_batch)
        for col in ("source", "split", "class"):
            rows[col] = meta_by_id[col].reindex(rows["sample_id"]).to_numpy() if col in meta.columns else np.nan
        rows["batch_id"], rows["batch_status"] = b["batch_id"], status
        qc_rows.append(rows)

    batches_df = pd.DataFrame(batch_rows, columns=BATCH_COLS)
    qc_df = pd.concat(qc_rows, ignore_index=True)[QC_COLS].sort_values("sample_id", ignore_index=True)
    return batches_df, qc_df


def ingest_batches(extra_files=None, write: bool = False, std_dir: Path = STD_DIR,
                   out_dir: Path = CLEANSED_DIR) -> pd.DataFrame:
    """Chia dữ liệu thành 4 lô theo nguồn, kiểm tra từng lô, chặn lô lỗi, ghi nhật ký.

    Trả về bảng theo lô (cột BATCH_COLS): status 'passed' | 'blocked', reason là các lỗi chặn
    lô (rỗng khi passed), warnings là ghi chú không chặn. Mặc định không ghi file để dashboard
    gọi trực tiếp được; `write=True` (make ingest) ghi batches.parquet, qc_samples.parquet và
    GHI THÊM vào ingest_log.csv trong out_dir.

    extra_files: các file CSV lô thêm (demo), định dạng: sample_id, split, class, subtype,
    tissue, source, sex rồi 7129 cột probe.
    """
    batches, qc = _assess(extra_files, std_dir)
    if write:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        batches.to_parquet(out_dir / BATCHES_FILE, index=False)
        qc.to_parquet(out_dir / QC_FILE, index=False)
        log = batches[["batch_id", "source", "n_samples", "status", "reason"]].copy()
        log.insert(0, "run_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        log_path = out_dir / LOG_FILE
        log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    return batches


def sample_quality(extra_files=None, std_dir: Path = STD_DIR) -> pd.DataFrame:
    """Chất lượng theo mẫu (bảng fact_qc), một hàng mỗi mẫu của các lô đọc được cấu trúc.

    Cờ qc_outlier của 72 mẫu thật trùng cột qc_outlier của tầng cleansed (cùng quy tắc
    quality.sample_qc). Mẫu của lô hỏng cấu trúc không có hàng nào.
    """
    return _assess(extra_files, std_dir)[1]


# ---------------------------------------------------------------------------
# Kiểm tra file chip của một mẫu mới
# ---------------------------------------------------------------------------
def validate_new_sample(path, std_dir: Path = STD_DIR) -> dict:
    """Kiểm tra file kết quả chip của một mẫu mới: đủ 7129 gen, đúng mã gen, không ô trống.

    path: đường dẫn hoặc đối tượng file (ví dụ file tải lên từ Streamlit). File là CSV có đúng
    hai cột `probe_id,value`, mỗi dòng một probe, giá trị ở thang MAS4 gốc như data/raw
    (chưa threshold, chưa log; được phép âm). Thứ tự dòng không quan trọng: ghép theo mã gen.

    Trả về {'ok', 'errors', 'warnings', 'values'}:
      ok=False khi có ít nhất một lỗi; values=None.
      ok=True thì values là Series float, chỉ mục probe_id theo đúng thứ tự genes.parquet,
      dùng thẳng cho patient_profile(values=...) của TV1. warnings không chặn (ví dụ chip có
      tương quan thấp với 38 mẫu tham chiếu).
    Tầng standardized là bắt buộc (danh sách probe chuẩn và mẫu tham chiếu); thiếu thì báo
    FileNotFoundError vì đó là lỗi cài đặt, không phải lỗi của file mẫu.
    """
    errors: list[str] = []
    warns: list[str] = []
    fail = lambda: {"ok": False, "errors": errors, "warnings": warns, "values": None}  # noqa: E731

    std_dir = Path(std_dir)
    for name in ("genes", "expression", "samples"):
        if not (std_dir / f"{name}.parquet").exists():
            raise FileNotFoundError(f"Thiếu {name}.parquet trong {std_dir} — chạy `make standardize` trước.")
    ref_probes = pd.Index(pd.read_parquet(std_dir / "genes.parquet")["probe_id"])
    name = Path(str(path if isinstance(path, (str, Path)) else getattr(path, "name", "new_sample"))).stem

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except FileNotFoundError:
        errors.append(f"không tìm thấy file: {path}")
        return fail()
    except pd.errors.EmptyDataError:
        errors.append("file rỗng")
        return fail()
    except (UnicodeDecodeError, pd.errors.ParserError) as err:
        errors.append(f"không đọc được file CSV: {str(err).splitlines()[0]}")
        return fail()

    df.columns = [str(c).strip() for c in df.columns]
    lacking = [c for c in SAMPLE_FILE_COLS if c not in df.columns]
    if lacking:
        errors.append(f"thiếu cột {lacking}: dòng tiêu đề phải là 'probe_id,value' (file có {_preview(df.columns)})")
        return fail()
    extra_cols = [c for c in df.columns if c not in SAMPLE_FILE_COLS]
    if extra_cols:
        warns.append(f"bỏ qua cột thừa {_preview(extra_cols)}")

    probe = df["probe_id"].str.strip()
    raw = df["value"].str.strip()
    if len(df) != len(ref_probes):
        errors.append(f"file có {len(df)} dòng gen, cần đúng {len(ref_probes)}")
    # Số dòng trong file = chỉ mục + 2 (dòng tiêu đề là dòng 1) để kỹ thuật viên tìm đúng chỗ.
    blank_probe = probe == ""
    if blank_probe.any():
        errors.append(f"{int(blank_probe.sum())} dòng trống mã gen (dòng {_preview(df.index[blank_probe] + 2)})")
    blank_val = raw == ""
    if blank_val.any():
        errors.append(f"{int(blank_val.sum())} ô giá trị trống (gen {_preview(probe[blank_val])})")
    values = pd.to_numeric(raw, errors="coerce")
    non_num = values.isna() & ~blank_val
    if non_num.any():
        examples = [f"{p}='{v}'" for p, v in zip(probe[non_num][:3], raw[non_num][:3])]
        errors.append(f"{int(non_num.sum())} giá trị không phải số (ví dụ {', '.join(examples)})")
    named = probe[~blank_probe]
    dup = named[named.duplicated(keep=False)]
    if len(dup):
        errors.append(f"{dup.nunique()} mã gen lặp lại: {_preview(sorted(set(dup)))}")
    unknown = pd.Index(named.unique()).difference(ref_probes)
    if len(unknown):
        errors.append(f"{len(unknown)} mã gen không có trong {len(ref_probes)} probe chuẩn: {_preview(unknown)}")
    missing = ref_probes.difference(pd.Index(named))
    if len(missing):
        errors.append(f"thiếu {len(missing)}/{len(ref_probes)} gen: {_preview(missing)}")
    if errors:
        return fail()

    values = pd.Series(values.to_numpy(dtype=float), index=pd.Index(probe, name="probe_id"), name=name)
    values = values.reindex(ref_probes)

    # Thang đo: so với 38 mẫu tham chiếu (tập huấn luyện, quy ước ADR-014).
    X = pd.read_parquet(std_dir / "expression.parquet")
    samples = pd.read_parquet(std_dir / "samples.parquet")
    X.index, samples.index = X.index.astype(int), samples.index.astype(int)
    ref = X.loc[samples.index[samples["split"] == "train"], list(ref_probes)]
    share_lo = float((values < EXPR_LO).mean())
    if share_lo > LOG_SCALE_SHARE:
        errors.append(f"{share_lo:.0%} giá trị < {EXPR_LO}: có vẻ file đã log hóa hoặc sai thang đo; "
                      "cần giá trị MAS4 gốc (như data/raw)")
        return fail()
    # Chip lỗi: cùng ba thước đo và cùng quy tắc với cờ qc_outlier (quality.sample_qc), nhưng mẫu
    # mới được đặt vào phân bố của mẫu tham chiếu thay vì tự tính median/MAD trên chính nó.
    # (Không dùng khoảng min–max của 38 mẫu: bệnh nhân thật 52 đã nằm ngoài khoảng đó.)
    def robust_z(x: float, ref_values: np.ndarray) -> float:
        center = np.median(ref_values)
        mad = np.median(np.abs(ref_values - center))
        return 0.6745 * (x - center) / mad if mad else 0.0

    ref_log = np.log10(np.clip(ref.to_numpy(dtype=float), EXPR_LO, EXPR_HI))
    new_log = np.log10(np.clip(values.to_numpy(), EXPR_LO, EXPR_HI))
    ref_med_corr = np.nanmedian(sample_correlation(ref).to_numpy(), axis=1)
    r_new = float(np.median([np.corrcoef(new_log, row)[0, 1] for row in ref_log]))
    z = robust_z(r_new, ref_med_corr)
    if z < -MODZ_THRESHOLD:
        warns.append(f"tương quan trung vị với {len(ref)} mẫu tham chiếu thấp bất thường (r = {r_new:.3f}, "
                     f"z = {z:.1f}): chip có thể lỗi, nên đo lại; kết quả dự đoán cần thận trọng")
    ref_q = np.percentile(ref_log, [25, 50, 75], axis=1)
    new_q = np.percentile(new_log, [25, 50, 75])
    for label, x, ref_values in (("trung vị", new_q[1], ref_q[1]), ("IQR", new_q[2] - new_q[0], ref_q[2] - ref_q[0])):
        z = robust_z(float(x), ref_values)
        if abs(z) > MODZ_THRESHOLD:
            warns.append(f"{label} log10 của mẫu ({x:.3f}) lệch hẳn {len(ref)} mẫu tham chiếu (z = {z:.1f})")
    return {"ok": True, "errors": errors, "warnings": warns, "values": values}


# ---------------------------------------------------------------------------
# Bộ file lỗi mẫu để demo (tuần 2 của lịch giai đoạn 2, ADR-031)
# ---------------------------------------------------------------------------
def write_demo_files(out_dir: Path = DEMO_DIR, std_dir: Path = STD_DIR) -> pd.DataFrame:
    """Sinh bộ file demo từ dữ liệu thật; trả về bảng mô tả (cũng ghi ra out_dir/README.md).

    Mỗi file một tình huống. Cột `expect` là kết quả mong đợi, máy đọc được để test kiểm lại:
      file chip một mẫu (validate_new_sample): "ok", "ok+warning:<đoạn chữ>", "error:<đoạn chữ>"
      file lô (ingest_batches(extra_files=...)): "passed", "blocked:<đoạn chữ>"
    Ghi đè các file cùng tên; không đụng tới tầng dữ liệu nào.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X = pd.read_parquet(std_dir / "expression.parquet")
    samples = pd.read_parquet(std_dir / "samples.parquet")
    X.index, samples.index = X.index.astype(int), samples.index.astype(int)
    base = X.loc[DEMO_PATIENT].rename("value").rename_axis("probe_id").reset_index()
    rng = np.random.default_rng(0)
    rows = []

    def chip(name: str, df: pd.DataFrame | None, case: str, expect: str, raw_text: str | None = None) -> None:
        path = out_dir / name
        if raw_text is not None:
            path.write_text(raw_text, encoding="utf-8")
        else:
            df.to_csv(path, index=False)
        rows.append({"file": name, "kind": "mẫu", "case": case, "expect": expect})

    chip("mau_hop_le_52.csv", base, f"chip hợp lệ của bệnh nhân {DEMO_PATIENT} (tập kiểm tra)", "ok")
    chip("loi_thieu_gen.csv", base.iloc[:-29], "file bị cắt cụt, thiếu 29 dòng cuối", "error:thiếu 29/7129 gen")
    typo = base.copy()
    typo.loc[[100, 200, 300], "probe_id"] = typo.loc[[100, 200, 300], "probe_id"].str.upper()
    chip("loi_sai_ma_gen.csv", typo, "3 mã gen gõ sai (đổi chữ hoa)", "error:3 mã gen không có trong")
    chip("loi_trung_gen.csv", pd.concat([base, base.iloc[[10, 20]]]), "2 dòng gen bị lặp", "error:2 mã gen lặp lại")
    blank = base.astype({"value": str})
    blank.loc[[5, 50, 500, 5000], "value"] = ""
    chip("loi_o_trong.csv", blank, "4 ô giá trị trống", "error:4 ô giá trị trống")
    text = base.astype({"value": str})
    text.loc[[7, 70, 700], "value"] = ["abc", "1,234", "N/A"]
    chip("loi_khong_phai_so.csv", text, "giá trị chữ, dấu phẩy hàng nghìn, N/A", "error:3 giá trị không phải số")
    chip("loi_sai_tieu_de.csv", base.rename(columns={"probe_id": "Gene Accession Number", "value": "52"}),
         "tiêu đề kiểu Kaggle thay vì probe_id,value", "error:'probe_id,value'")
    logged = base.assign(value=np.log10(base["value"].clip(100, 16000)).round(4))
    chip("loi_da_log10.csv", logged, "giá trị đã threshold + log10", "error:có vẻ file đã log hóa")
    chip("loi_file_rong.csv", None, "file rỗng (0 byte)", "error:file rỗng", raw_text="")
    chip("canh_bao_chip_lech.csv", base.assign(value=rng.permutation(base["value"].to_numpy())),
         "đủ 7129 gen nhưng giá trị bị xáo giữa các gen (chip lỗi)", "ok+warning:tương quan trung vị")

    def batch(name: str, ids: list[int], new_ids: list[int], case: str, expect: str, damage=None) -> None:
        df = pd.concat([samples.loc[ids, SAMPLE_COLS], X.loc[ids]], axis=1)
        df.insert(0, "sample_id", new_ids)
        df = df.astype({c: str for c in df.columns if c not in SAMPLE_COLS + ["sample_id"]})
        if damage is not None:
            df = damage(df.reset_index(drop=True))
        df.to_csv(out_dir / name, index=False)
        rows.append({"file": name, "kind": "lô", "case": case, "expect": expect})

    batch("lo_hop_le_StJude.csv", [57, 58], [801, 802], "lô mới 2 mẫu của St-Jude, mã mẫu mới", "passed")

    def break_batch(df: pd.DataFrame) -> pd.DataFrame:
        df["source"] = "XYZ"                          # phòng lab ngoài danh mục
        df.loc[1, "subtype"] = "T-ALL"                # mẫu 902 là AML nhưng ghi T-ALL
        df.loc[0, X.columns[5]] = ""                  # 1 ô trống ở mẫu 901
        return df.drop(columns=[X.columns[100]])      # thiếu 1 probe

    batch("lo_loi_XYZ.csv", [50, 51, 53], [901, 902, 903],
          "lô 3 mẫu: nguồn lạ, thiếu 1 probe, 1 ô trống, class lệch subtype",
          "blocked:source ngoài danh mục", damage=break_batch)
    batch("lo_trung_ma.csv", [DEMO_PATIENT], [DEMO_PATIENT], f"gửi lại mẫu {DEMO_PATIENT} với mã cũ",
          f"blocked:sample_id đã có ở lô khác: [{DEMO_PATIENT}]")

    manifest = pd.DataFrame(rows)
    readme = ["# Bộ file lỗi mẫu để demo (TV2, công việc 2.4, ADR-031)", "",
              "Sinh bởi `python -m src.ingest --demo` từ dữ liệu thật; không sửa tay, không đưa lên Git.", "",
              "- File **mẫu**: một chip, định dạng `probe_id,value` → kiểm bằng `validate_new_sample(path)`",
              "  (nút tải file ở màn hình Chất lượng dữ liệu / Hồ sơ bệnh nhân).",
              "- File **lô**: `sample_id` + bảng mẫu + 7129 cột probe → `ingest_batches(extra_files=[...])`.", "",
              "| File | Loại | Tình huống | Kết quả mong đợi |", "|---|---|---|---|"]
    readme += [f"| `{r.file}` | {r.kind} | {r.case} | `{r.expect}` |" for r in manifest.itertuples()]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Nhập dữ liệu theo lô và ghi bảng chất lượng (TV2, 2.4)")
    ap.add_argument("extra", nargs="*", help="file CSV lô thêm, định dạng xem ingest_batches")
    ap.add_argument("--demo", action="store_true",
                    help=f"sinh bộ file demo vào {DEMO_DIR} rồi chạy thử; không ghi bảng/nhật ký")
    args = ap.parse_args(argv)

    if args.demo:
        manifest = write_demo_files()
        print(f"Đã sinh {len(manifest)} file demo vào {DEMO_DIR} (xem README.md)\n")
        for r in manifest[manifest["kind"] == "mẫu"].itertuples():
            res = validate_new_sample(DEMO_DIR / r.file)
            detail = "; ".join(res["errors"] or res["warnings"]) or "—"
            print(f"[{'OK ' if res['ok'] else 'LỖI'}] {r.file:<26} {detail}")
        lots = [DEMO_DIR / f for f in manifest.loc[manifest["kind"] == "lô", "file"]]
        out = ingest_batches(extra_files=lots)
        print()
        for row in out.itertuples():
            print(f"[{row.status:<7}] {row.batch_id:<16} {row.reason or '—'}")
        return

    batches = ingest_batches(extra_files=args.extra, write=True)
    show = ["batch_id", "n_samples", "status", "present_rate", "median_corr_other", "n_qc_outlier"]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(batches[show].to_string(index=False))
    for _, row in batches.iterrows():
        if row["reason"]:
            print(f"[BLOCKED] {row['batch_id']}: {row['reason']}")
        if row["warnings"]:
            print(f"[WARN]    {row['batch_id']}: {row['warnings']}")
    print(f"\nĐã ghi {CLEANSED_DIR / BATCHES_FILE}, {CLEANSED_DIR / QC_FILE}; "
          f"ghi thêm {len(batches)} dòng vào {CLEANSED_DIR / LOG_FILE}")


if __name__ == "__main__":
    main()
