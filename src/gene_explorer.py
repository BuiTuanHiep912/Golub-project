"""TV4 - src/gene_explorer.py (công việc 2.2, Khám phá gen; ADR-014, ADR-051).

Tra cứu gen, thống kê biểu hiện theo nhóm, xếp hạng gen khác biệt, và trả lời câu hỏi
"gen này khác biệt do bệnh hay do nguồn mẫu" ở mức mô tả. Module chỉ tính toán, không vẽ hình
(hàm vẽ của TV4 nằm trong src/viz_gene.py) và không ghi file.

Giao diện đã khóa trong tests/test_interfaces.py, KHÔNG đổi tên khi chưa báo nhóm:
    find_gene('CD33')        -> DataFrame có probe_id, description (và các cột thêm)
    gene_stats(probe_id)     -> thống kê biểu hiện theo class, source, tissue

Đầu vào:
    data/warehouse/golub.duckdb (TV3, ADR-040)    -> fact_expression.value_log10, dim_sample
      (chưa dựng kho thì đọc load_golub("curated"), cùng số liệu; xem data_source())
    data/standardized/genes.parquet               -> probe_id, description, is_control
    data/standardized/calls.parquet               -> P/A/M của chip, chỉ để đánh giá độ tin cậy số đo
    results/gene_lists/gene_ranking.csv           -> xếp hạng gen trên 38 mẫu huấn luyện
    results/gene_lists/golub50_comparison.csv     -> (tùy chọn) đối chiếu 50 gen của Golub
    results/gene_lists/gene_annotations.csv       -> (tùy chọn, TV5) chú giải gen

Ba nhóm hàm:
    Tra cứu một gen   : find_gene, gene_values, gene_stats, disease_or_source
    Độ tin cậy, độ vững: gene_reliability (P/A, mức sàn theo lớp), robustness (bỏ mẫu QC, B/T-ALL, BM, St-Jude)
    Xếp hạng, lô mới  : ranking_table, collapse_by_symbol, annotate,
                        marker_consistency (lô có nhãn: gen dấu ấn có giữ chiều không),
                        batch_anomalies (lô chưa nhãn: mẫu nào có gen dấu ấn ngoài khoảng tham chiếu)
Quy ước chống rò rỉ: gen dấu ấn và khoảng tham chiếu chỉ lấy từ 38 mẫu huấn luyện; lô mới chỉ dùng
để kiểm chứng, không dùng để xếp hạng lại hay chọn gen.

Bộ nhớ đệm: dữ liệu được đọc một lần và giữ lại cho các lần gọi sau (khóa theo đường dẫn và
thời điểm sửa file, nên file sinh lại thì tự đọc lại). Gọi clear_cache() để xóa thủ công.
Các DataFrame trả về là bản sao, sửa thoải mái không ảnh hưởng bộ nhớ đệm.

Cách dùng:
    from src import gene_explorer as ge
    ge.find_gene("CD33")
    ge.gene_stats("M23197_at", by=("class", "source"))
    ge.disease_or_source("M23197_at")
    ge.ranking_table(top_n=50, direction="AML")
    ge.collapse_by_symbol(ge.ranking_table(top_n=200))
    ge.annotate(ge.ranking_table(top_n=50))
    vals = ge.gene_values("M23197_at")
    ge.gene_reliability("M23197_at"); ge.robustness("M23197_at")
    ge.marker_consistency(source="St-Jude")      # 46/50 gen dấu ấn giữ đúng chiều
    ge.batch_anomalies(source="CCG")

Chạy trực tiếp: python -m src.gene_explorer  (in ra một tra cứu mẫu)
"""
from __future__ import annotations

import itertools
import math
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import CURATED_DIR, DATA_DIR, GENE_LISTS_DIR, RANDOM_SEED, STD_DIR
    from src.gene_catalog import split_symbol
    from src.load import load_golub
except ModuleNotFoundError:
    from config import CURATED_DIR, DATA_DIR, GENE_LISTS_DIR, RANDOM_SEED, STD_DIR
    from gene_catalog import split_symbol
    from load import load_golub

RANKING_PATH = GENE_LISTS_DIR / "gene_ranking.csv"
GOLUB50_PATH = GENE_LISTS_DIR / "golub50_comparison.csv"
ANNOTATION_PATH = GENE_LISTS_DIR / "gene_annotations.csv"
GENES_PATH = STD_DIR / "genes.parquet"
CALLS_PATH = STD_DIR / "calls.parquet"
CURATED_FILES = (CURATED_DIR / "X_log10.parquet", CURATED_DIR / "samples.parquet")
WAREHOUSE_PATH = DATA_DIR / "warehouse" / "golub.duckdb"   # trùng src.warehouse.WAREHOUSE_PATH (TV3)
USE_WAREHOUSE = True                     # False: luôn đọc Parquet (dùng khi nghi kho cũ hơn tầng curated)

FLOOR_LOG10 = 2.0                        # log10(100): mức sàn của threshold [100, 16000]
PRESENT_HIGH = 0.5                       # lớp được xem là "có mặt" khi ≥ 50% mẫu có call P
MIN_DIFF_LOG10 = 0.1                     # chênh lệch nhỏ hơn mức này (≈ 1,26 lần) coi là quá nhỏ để nói chiều
CONSISTENCY_STATUS = ("Giữ chiều", "Đảo chiều", "Chênh lệch nhỏ", "Không so được")
PERM_EXACT_MAX = 5000                    # số cách xếp nhãn trong lô ≤ mức này thì duyệt hết (St-Jude: C(8,3) = 56)
N_PERM_RANDOM = 2000                     # lô lớn hơn: xáo nhãn ngẫu nhiên chừng này lần
CLASSES = ("ALL", "AML")

RANKING_NOTE = "Xếp hạng tính trên 38 mẫu huấn luyện, chỉ dùng để khám phá"
MIN_N_PER_GROUP = 3                      # ngưỡng cho cờ comparable của disease_or_source
GROUP_COLUMNS = ("class", "subtype", "source", "tissue", "split")
DIRECTIONS = ("ALL", "AML", "both")
ANNOTATION_COLUMNS = ("probe_id", "gene_symbol", "function", "cell_lineage", "source_url", "verified")
GOLUB50_COLUMNS = ("kết quả", "probe tốt nhất")
FIND_COLUMNS = ["probe_id", "gene_symbol", "description", "n_probes_same_description",
                "rank", "match_type"]


# ---------------------------------------------------------------- bộ nhớ đệm

def _mtime(path: Path) -> int | None:
    """Thời điểm sửa file (ns), None nếu chưa có; dùng làm khóa bộ nhớ đệm."""
    path = Path(path)
    return path.stat().st_mtime_ns if path.exists() else None


@lru_cache(maxsize=2)
def _expression_cached(source: str, _key: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source == "duckdb":
        return _read_warehouse(WAREHOUSE_PATH)
    return load_golub("curated")


def _read_warehouse(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc X (log10) và bảng mẫu từ kho DuckDB của TV3, cùng định dạng với load_golub("curated")."""
    import duckdb                        # chỉ cần khi có kho; không bắt buộc để import module

    con = duckdb.connect(str(path), read_only=True)
    try:
        long = con.execute("SELECT sample_id, probe_id, value_log10 FROM fact_expression").df()
        samples = con.execute("SELECT * FROM dim_sample ORDER BY sample_id").df()
    finally:
        con.close()
    samples = samples.set_index("sample_id")
    samples.index = samples.index.astype(int)
    X = long.pivot(index="sample_id", columns="probe_id", values="value_log10")
    X.index = X.index.astype(int)
    X = X.reindex(index=samples.index)
    X.columns.name = None
    return X, samples


@lru_cache(maxsize=2)
def _calls_cached(path: str, _mtime_key: int | None) -> pd.DataFrame:
    calls = pd.read_parquet(path)
    calls.index = calls.index.astype(int)
    return calls


@lru_cache(maxsize=4)
def _genes_cached(path: str, _mtime_key: int | None) -> pd.DataFrame:
    genes = pd.read_parquet(path)
    genes["gene_symbol"] = [split_symbol(d)[0] for d in genes["description"].fillna("")]
    return genes


@lru_cache(maxsize=4)
def _ranking_cached(path: str, _mtime_key: int | None, g50_path: str, _g50_mtime: int | None) -> pd.DataFrame:
    rank = pd.read_csv(path)
    rank["gene_symbol"] = [split_symbol(d)[0] for d in rank["description"].fillna("")]
    rank["in_golub50"] = _golub50_flags(rank["probe_id"], Path(g50_path))
    return rank


def clear_cache() -> None:
    """Xóa bộ nhớ đệm (ví dụ sau khi sinh lại dữ liệu trong cùng một phiên)."""
    _expression_cached.cache_clear()
    _genes_cached.cache_clear()
    _ranking_cached.cache_clear()
    _calls_cached.cache_clear()


# ---------------------------------------------------------------- đọc dữ liệu

def data_source() -> str:
    """Nguồn dữ liệu biểu hiện đang dùng: "duckdb" (kho của TV3) hoặc "parquet" (tầng curated)."""
    return "duckdb" if USE_WAREHOUSE and WAREHOUSE_PATH.exists() else "parquet"


def _load_expression() -> tuple[pd.DataFrame, pd.DataFrame]:
    """X (72 x p, log10) và samples, cùng index sample_id. Không sửa trực tiếp."""
    source = data_source()
    files = (WAREHOUSE_PATH,) if source == "duckdb" else CURATED_FILES
    return _expression_cached(source, tuple(_mtime(p) for p in files))


def _load_calls() -> pd.DataFrame | None:
    """Call P/A/M của chip (72 x 7129); None khi chưa có file."""
    if not CALLS_PATH.exists():
        return None
    return _calls_cached(str(CALLS_PATH), _mtime(CALLS_PATH))


def _load_genes() -> pd.DataFrame:
    """Danh sách probe, thêm cột gene_symbol (chuỗi rỗng khi mô tả không có ký hiệu)."""
    return _genes_cached(str(GENES_PATH), _mtime(GENES_PATH)).copy()


def _load_ranking() -> pd.DataFrame:
    """gene_ranking.csv, thêm gene_symbol và cột in_golub50."""
    return _ranking_cached(str(RANKING_PATH), _mtime(RANKING_PATH),
                           str(GOLUB50_PATH), _mtime(GOLUB50_PATH)).copy()


def _golub50_flags(probe_ids: pd.Series, path: Path) -> pd.Series:
    """True khi probe là probe tốt nhất của một gen Golub và kết quả là 'trùng'.

    Không có file, hoặc file thiếu cột cần dùng, thì trả về NA (kèm cảnh báo khi thiếu cột).
    """
    na = pd.Series(pd.NA, index=probe_ids.index, dtype="boolean")
    if not path.exists():
        return na
    g50 = pd.read_csv(path)
    missing = [c for c in GOLUB50_COLUMNS if c not in g50.columns]
    if missing:
        warnings.warn(f"{path.name} thiếu cột {missing}; in_golub50 để trống.", stacklevel=2)
        return na
    matched = set(g50.loc[g50["kết quả"].eq("trùng"), "probe tốt nhất"].dropna())
    return probe_ids.isin(matched).astype("boolean")


def _check_probe(probe_id: str, X: pd.DataFrame) -> None:
    if probe_id not in X.columns:
        raise KeyError(f"Không có probe {probe_id!r}; dùng find_gene() để tìm mã đúng.")


def _is_blank(s: pd.Series) -> pd.Series:
    """True khi giá trị là NA hoặc chuỗi rỗng."""
    return s.isna() | s.astype(object).where(s.notna(), "").astype(str).str.strip().eq("")


# ---------------------------------------------------------------- tra cứu

def find_gene(query: str, limit: int = 20) -> pd.DataFrame:
    """Tìm probe theo mã probe, ký hiệu gen hoặc chuỗi trong mô tả (đều không phân biệt hoa thường).

    Thứ tự ưu tiên: probe > symbol > text. Không khớp thì trả về DataFrame rỗng.
    Cột đầu ra: probe_id, gene_symbol, description, n_probes_same_description, rank, match_type.
    rank là hạng trong gene_ranking.csv; để trống khi probe không nằm trong xếp hạng
    (probe điều khiển AFFX hoặc gen bị lọc ở công việc 1.6).
    """
    q = str(query).strip()
    if not q:
        raise ValueError("query rỗng")
    genes = _load_genes()
    counts = genes["description"].value_counts()
    ql = q.lower()

    is_probe = genes["probe_id"].str.lower().eq(ql)
    is_symbol = genes["gene_symbol"].str.lower().eq(ql)
    is_text = genes["description"].fillna("").str.lower().str.contains(ql, regex=False)

    hits = genes.loc[is_probe | is_symbol | is_text].copy()
    hits["match_type"] = "text"
    hits.loc[is_symbol.loc[hits.index], "match_type"] = "symbol"
    hits.loc[is_probe.loc[hits.index], "match_type"] = "probe"
    hits["n_probes_same_description"] = hits["description"].map(counts)

    if RANKING_PATH.exists():
        ranks = _load_ranking().set_index("probe_id")["rank"]
        hits["rank"] = hits["probe_id"].map(ranks).astype("Int64")
    else:
        hits["rank"] = pd.Series(pd.NA, index=hits.index, dtype="Int64")

    order = {"probe": 0, "symbol": 1, "text": 2}
    hits["_o"] = hits["match_type"].map(order)
    hits = hits.sort_values(["_o", "probe_id"]).head(limit)
    return hits[FIND_COLUMNS].reset_index(drop=True)


def gene_values(probe_id: str) -> pd.DataFrame:
    """Một dòng mỗi mẫu: sample_id, value (log10), class, subtype, source, tissue, split, qc_outlier.

    qc_outlier (mẫu bị gắn cờ ở tầng cleansed, ví dụ mẫu 21) chỉ có khi bảng mẫu có cột này.
    """
    X, samples = _load_expression()
    _check_probe(probe_id, X)
    out = samples.copy()
    out["value"] = X[probe_id].reindex(out.index)
    out.index.name = "sample_id"
    extra = ["qc_outlier"] if "qc_outlier" in out.columns else []
    return out.reset_index()[["sample_id", "value", *GROUP_COLUMNS, *extra]]


def gene_stats(probe_id: str, by: tuple[str, ...] = ("class", "source", "tissue")) -> pd.DataFrame:
    """Thống kê mô tả (n, mean, median, sd) của một probe, nhóm theo `by`.

    Mặc định nhóm theo class, source, tissue (giao diện thống nhất, ADR-014).
    """
    by = tuple(by)
    if not by or any(b not in GROUP_COLUMNS for b in by):
        raise ValueError(f"by phải là tổ hợp của {GROUP_COLUMNS}, nhận {by}")
    df = gene_values(probe_id)
    agg = (df.groupby(list(by), observed=True)["value"]
             .agg(n="count", mean="mean", median="median", sd="std")
             .reset_index())
    return agg


def disease_or_source(probe_id: str) -> pd.DataFrame:
    """So sánh ALL và AML bên trong từng nguồn, trên cả 72 mẫu, để thấy khác biệt do bệnh hay do nguồn.

    Các số liệu (n_ALL, n_AML, mean_ALL, mean_AML, diff) luôn được trả về đầy đủ. Ngưỡng
    MIN_N_PER_GROUP chỉ dùng cho cờ comparable: True khi cả hai nhóm có ít nhất MIN_N_PER_GROUP
    mẫu trong nguồn đó. Với dữ liệu Golub, DFCI chỉ có ALL, CALGB và CCG chỉ có AML, nên chỉ
    St-Jude (3 ALL, 5 AML, đều ở tập kiểm tra) là so sánh được.
    """
    df = gene_values(probe_id)
    rows = []
    for source, sub in df.groupby("source", observed=True):
        all_v = sub.loc[sub["class"].eq("ALL"), "value"]
        aml_v = sub.loc[sub["class"].eq("AML"), "value"]
        n_all, n_aml = int(all_v.count()), int(aml_v.count())
        mean_all = all_v.mean() if n_all else float("nan")
        mean_aml = aml_v.mean() if n_aml else float("nan")
        rows.append({
            "source": source,
            "n_ALL": n_all,
            "n_AML": n_aml,
            "mean_ALL": mean_all,
            "mean_AML": mean_aml,
            "diff": mean_all - mean_aml,
            "comparable": n_all >= MIN_N_PER_GROUP and n_aml >= MIN_N_PER_GROUP,
        })
    return pd.DataFrame(rows, columns=["source", "n_ALL", "n_AML", "mean_ALL", "mean_AML",
                                       "diff", "comparable"])


# ---------------------------------------------------------------- xếp hạng

def ranking_table(top_n: int | None = 50, direction: str = "both", min_abs_snr: float = 0.0) -> pd.DataFrame:
    """Bảng xếp hạng gen từ gene_ranking.csv.

    top_n: số dòng; None lấy toàn bộ (ví dụ cho volcano plot).
    direction: "ALL" (SNR > 0, cao ở ALL), "AML" (SNR < 0) hoặc "both".
    Cột đầu ra: rank, probe_id, gene_symbol, description, snr, t_stat, q_value, direction, in_golub50.
    Thuộc tính df.attrs["note"] chứa RANKING_NOTE.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction phải là một trong {DIRECTIONS}, nhận {direction!r}")
    rank = _load_ranking()
    if direction == "ALL":
        rank = rank[rank["snr"] > 0]
    elif direction == "AML":
        rank = rank[rank["snr"] < 0]
    rank = rank[rank["snr"].abs() >= min_abs_snr]
    rank = rank.sort_values("rank")
    rank = (rank if top_n is None else rank.head(top_n)).copy()
    rank["direction"] = rank["snr"].map(lambda s: "ALL" if s > 0 else "AML")
    cols = ["rank", "probe_id", "gene_symbol", "description", "snr", "t_stat", "q_value",
            "direction", "in_golub50"]
    out = rank[cols].reset_index(drop=True)
    out.attrs["note"] = RANKING_NOTE
    return out


def collapse_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Gộp các probe cùng gen. Giữ probe có |SNR| lớn nhất, thêm cột n_probes.

    Khóa gộp là gene_symbol; khi symbol rỗng thì dùng description, rỗng nữa thì dùng probe_id.
    """
    if "snr" not in df.columns:
        raise ValueError("collapse_by_symbol cần cột 'snr'")
    work = df.copy()
    key = work["probe_id"].astype(str) if "probe_id" in work.columns else pd.Series(work.index.astype(str), index=work.index)
    for col in ("description", "gene_symbol"):           # ưu tiên tăng dần: probe < description < symbol
        if col in work.columns:
            key = work[col].where(~_is_blank(work[col]), key)
    work["_key"] = key
    work["n_probes"] = work.groupby("_key")["_key"].transform("size").astype(int)
    work["_abs"] = work["snr"].abs()
    best = work.sort_values("_abs", ascending=False).drop_duplicates("_key")
    best = best.sort_values("rank") if "rank" in best.columns else best.sort_values("_abs", ascending=False)
    return best.drop(columns=["_key", "_abs"]).reset_index(drop=True)


# ---------------------------------------------------------------- độ tin cậy và độ vững của một gen

def gene_reliability(probe_id: str) -> pd.DataFrame:
    """Số đo của gen này có đáng tin không, tính riêng cho từng lớp.

    Cột đầu ra: class, n, median, pct_floor (tỉ lệ mẫu ở mức sàn log10 = 2, tức ≤ 100 trên thang gốc),
    pct_present / pct_marginal / pct_absent (call P/M/A của chip; NA nếu chưa có calls.parquet).
    df.attrs["pattern"] là "on_off", "level", "weak" hoặc "unknown"; df.attrs["note"] giải thích bằng chữ.

    Phải tính theo lớp: gen dấu ấn tốt thường "tắt" ở một lớp (call A, ở mức sàn) và "bật" ở lớp kia,
    nên tỉ lệ P chung cho 72 mẫu sẽ thấp mà không có nghĩa là số đo kém (ví dụ CD33: P ở 19% ALL, 92% AML).
    """
    X, samples = _load_expression()
    _check_probe(probe_id, X)
    calls = _load_calls()
    has_calls = calls is not None and probe_id in calls.columns
    rows = []
    for cls in CLASSES:
        idx = samples.index[samples["class"].eq(cls)]
        v = X.loc[idx, probe_id]
        row = {"class": cls, "n": len(idx), "median": v.median(),
               "pct_floor": float((v <= FLOOR_LOG10 + 1e-9).mean()) if len(idx) else np.nan}
        if has_calls:
            c = calls[probe_id].reindex(idx)
            row.update(pct_present=float(c.eq("P").mean()), pct_marginal=float(c.eq("M").mean()),
                       pct_absent=float(c.eq("A").mean()))
        else:
            row.update(pct_present=np.nan, pct_marginal=np.nan, pct_absent=np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["pattern"], out.attrs["note"] = _reliability_note(out)
    return out


def _reliability_note(rel: pd.DataFrame) -> tuple[str, str]:
    if rel["pct_present"].isna().all():
        return "unknown", "Chưa có call P/A của chip (calls.parquet); chỉ xem được tỉ lệ mẫu ở mức sàn."
    high = rel.set_index("class")["pct_present"] >= PRESENT_HIGH
    pct = rel.set_index("class")["pct_present"]
    if not high.any():
        return "weak", (f"Tín hiệu yếu: chip ghi nhận gen có mặt ở {pct['ALL']:.0%} mẫu ALL và {pct['AML']:.0%} "
                        "mẫu AML (đều dưới một nửa). Khác biệt có thể chỉ là nhiễu quanh ngưỡng đo.")
    if high.all():
        return "level", (f"Gen có mặt ở cả hai lớp ({pct['ALL']:.0%} ALL, {pct['AML']:.0%} AML): khác biệt là "
                         "về mức độ biểu hiện, không phải bật/tắt.")
    on = high.idxmax()
    off = "AML" if on == "ALL" else "ALL"
    return "on_off", (f"Kiểu bật/tắt: gen có mặt ở {pct[on]:.0%} mẫu {on} nhưng chỉ {pct[off]:.0%} mẫu {off}. "
                      f"Mức thấp ở {off} phần lớn là mức sàn của chip, nên so sánh định tính đáng tin hơn con số.")


def _class_compare(df: pd.DataFrame, all_mask: pd.Series, aml_mask: pd.Series) -> dict:
    a, m = df.loc[all_mask, "value"], df.loc[aml_mask, "value"]
    n_all, n_aml = int(a.count()), int(m.count())
    mean_all = a.mean() if n_all else np.nan
    mean_aml = m.mean() if n_aml else np.nan
    diff = mean_all - mean_aml
    return {"n_ALL": n_all, "n_AML": n_aml, "mean_ALL": mean_all, "mean_AML": mean_aml, "diff": diff,
            "fold_change": 10 ** diff if pd.notna(diff) else np.nan,
            "comparable": n_all >= MIN_N_PER_GROUP and n_aml >= MIN_N_PER_GROUP}


def robustness(probe_id: str) -> pd.DataFrame:
    """Khác biệt ALL/AML của gen có còn giữ khi đổi cách so sánh không.

    Mỗi dòng là một phép so sánh: tập huấn luyện (mốc), toàn bộ 72 mẫu, bỏ mẫu gắn cờ QC, B-ALL so với AML,
    T-ALL so với AML, chỉ tủy xương (BM), trong St-Jude (cùng phòng lab).
    Cột: check, n_ALL, n_AML, mean_ALL, mean_AML, diff (ALL − AML, log10), fold_change (= 10^diff, ALL gấp
    bao nhiêu lần AML trên thang gốc), comparable (mỗi bên ≥ MIN_N_PER_GROUP mẫu), same_direction (cùng
    dấu với tập huấn luyện; NA khi một bên không có mẫu).
    Chỉ là mô tả trên dữ liệu đã có nhãn, không dùng để chọn gen.
    """
    df = gene_values(probe_id)
    is_all, is_aml = df["class"].eq("ALL"), df["class"].eq("AML")
    every = pd.Series(True, index=df.index)
    checks = [
        ("Tập huấn luyện (mốc so sánh)", df["split"].eq("train")),
        ("Toàn bộ 72 mẫu", every),
    ]
    if "qc_outlier" in df.columns:
        checks.append(("Bỏ mẫu gắn cờ chất lượng", ~df["qc_outlier"].fillna(False).astype(bool)))
    rows = []
    for label, mask in checks:
        rows.append({"check": label, **_class_compare(df, mask & is_all, mask & is_aml)})
    rows.append({"check": "B-ALL so với AML", **_class_compare(df, df["subtype"].eq("B-ALL"), is_aml)})
    rows.append({"check": "T-ALL so với AML", **_class_compare(df, df["subtype"].eq("T-ALL"), is_aml)})
    bm = df["tissue"].eq("BM")
    rows.append({"check": "Chỉ tủy xương (BM)", **_class_compare(df, bm & is_all, bm & is_aml)})
    sj = df["source"].eq("St-Jude")
    rows.append({"check": "Trong St-Jude (cùng phòng lab)", **_class_compare(df, sj & is_all, sj & is_aml)})

    out = pd.DataFrame(rows)
    ref = np.sign(out.loc[0, "diff"])
    same = np.sign(out["diff"]).eq(ref)
    out["same_direction"] = same.astype("boolean").where(out["diff"].notna(), pd.NA)
    ok = out.iloc[1:]
    ok = ok[ok["comparable"]]
    n_same = int(ok["same_direction"].fillna(False).sum())
    out.attrs["note"] = (f"Giữ đúng chiều của tập huấn luyện ở {n_same}/{len(ok)} phép so sánh đủ mẫu "
                         f"(mỗi bên từ {MIN_N_PER_GROUP} mẫu).")
    return out


# ---------------------------------------------------------------- lô mẫu mới

def _marker_genes(n_per_side: int) -> pd.DataFrame:
    """Gen dấu ấn theo kiểu Golub: n gen hạng cao nhất mỗi phía (SNR > 0 cao ở ALL, SNR < 0 cao ở AML)."""
    rank = _load_ranking()
    up = rank[rank["snr"] > 0].nsmallest(n_per_side, "rank")
    down = rank[rank["snr"] < 0].nsmallest(n_per_side, "rank")
    return pd.concat([up, down]).sort_values("rank").reset_index(drop=True)


def _batch_index(samples: pd.DataFrame, source: str | None, sample_ids, only_test: bool) -> pd.Index:
    if (source is None) == (sample_ids is None):
        raise ValueError("Truyền đúng một trong hai: source (tên phòng lab) hoặc sample_ids")
    if sample_ids is not None:
        mask = samples.index.isin(list(sample_ids))
    else:
        if source not in set(samples["source"]):
            raise ValueError(f"Không có nguồn {source!r}; các nguồn: {sorted(set(samples['source']))}")
        mask = samples["source"].eq(source).to_numpy()
    if only_test:
        mask = mask & samples["split"].eq("test").to_numpy()
    idx = samples.index[mask]
    if len(idx) == 0:
        raise ValueError("Lô rỗng (kiểm tra source/sample_ids, hoặc only_test=True mà lô toàn mẫu huấn luyện)")
    return idx


def _label_masks(is_all: np.ndarray) -> np.ndarray:
    """Các cách xếp nhãn giữ nguyên số mẫu ALL; mỗi hàng là một mặt nạ True = ALL.

    Ít cách xếp (≤ PERM_EXACT_MAX) thì liệt kê đủ, trong đó có cách xếp thật, nên p là chính xác.
    Nhiều hơn thì hàng đầu là nhãn thật, sau đó N_PERM_RANDOM lần xáo ngẫu nhiên (seed cố định).
    """
    n, k = len(is_all), int(is_all.sum())
    n_ways = math.comb(n, k)
    if n_ways <= PERM_EXACT_MAX:
        masks = np.zeros((n_ways, n), dtype=bool)
        for row, idx in enumerate(itertools.combinations(range(n), k)):
            masks[row, list(idx)] = True
        return masks
    rng = np.random.default_rng(RANDOM_SEED)
    return np.vstack([is_all] + [rng.permutation(is_all) for _ in range(N_PERM_RANDOM)])


def _perm_same_direction(Xb: np.ndarray, is_all: np.ndarray, train_sign: np.ndarray) -> tuple[float, int]:
    """Kiểm định hoán vị trong lô cho cả bộ gen dấu ấn.

    Thống kê: số gen cùng dấu ALL − AML với tập huấn luyện. Giả thuyết không: nhãn trong lô không liên quan
    đến gen, nên xáo nhãn (giữ số ALL/AML) cho số gen cùng dấu tương đương. Xáo cả bộ gen một lúc nên tương
    quan giữa các gen được giữ nguyên. Trả về (p một phía, số cách xếp đã dùng).
    """
    masks = _label_masks(is_all).astype(float)
    k = is_all.sum()
    diff = masks @ Xb / k - (1.0 - masks) @ Xb / (len(is_all) - k)
    same = (np.sign(diff) == train_sign).sum(axis=1)
    observed = (np.sign(Xb[is_all].mean(axis=0) - Xb[~is_all].mean(axis=0)) == train_sign).sum()
    return float((same >= observed).mean()), len(masks)


def marker_consistency(source: str | None = None, sample_ids=None, n_per_side: int = 25,
                       only_test: bool = True) -> pd.DataFrame:
    """Lô mới CÓ chẩn đoán: các gen dấu ấn có giữ chiều ALL/AML như ở tập huấn luyện không.

    Gen dấu ấn: n_per_side gen hạng cao nhất mỗi phía (mặc định 25 + 25, như Golub). only_test=True chỉ lấy
    mẫu kiểm tra của lô, để không so tập huấn luyện với chính nó.
    Cột: rank, probe_id, gene_symbol, snr, train_diff, batch_mean_ALL, batch_mean_AML, batch_diff,
    batch_diff_sd (batch_diff chia cho độ lệch chuẩn trong lớp của gen trên tập huấn luyện, để so mức chênh
    giữa các gen; σ lấy từ 38 mẫu huấn luyện vì σ của lô vài mẫu không ổn định),
    same_direction (cùng dấu với tập huấn luyện; NA khi lô thiếu một lớp),
    status: "Giữ chiều" / "Đảo chiều" khi |batch_diff| ≥ MIN_DIFF_LOG10; "Chênh lệch nhỏ" khi dưới ngưỡng
    (thường do cả hai lớp ở gần mức sàn, không nói được chiều); "Không so được" khi lô thiếu một lớp.
    df.attrs: n_ALL, n_AML (số mẫu của lô), n_genes, n_same (cùng dấu), n_kept, n_flipped, n_small,
    p_perm, n_perm (kiểm định hoán vị trong lô cho n_same; NaN và 0 khi không so được), comparable, note.
    Ví dụ: marker_consistency(source="St-Jude") -> 46/50 cùng dấu; 41 giữ chiều rõ, 0 đảo chiều rõ.
    """
    X, samples = _load_expression()
    batch = _batch_index(samples, source, sample_ids, only_test)
    genes = _marker_genes(n_per_side)
    genes = genes[genes["probe_id"].isin(X.columns)]
    train = samples.index[samples["split"].eq("train")]

    def subset(idx, cls):
        sub = idx[samples.loc[idx, "class"].eq(cls).to_numpy()]
        return X.loc[sub, genes["probe_id"]]

    tr_all, tr_aml = subset(train, "ALL"), subset(train, "AML")
    b_all_x, b_aml_x = subset(batch, "ALL"), subset(batch, "AML")
    n_all, n_aml = len(b_all_x), len(b_aml_x)
    b_all, b_aml = b_all_x.mean(), b_aml_x.mean()
    n1, n2 = len(tr_all), len(tr_aml)
    pooled_sd = np.sqrt(((n1 - 1) * tr_all.var() + (n2 - 1) * tr_aml.var()) / (n1 + n2 - 2)).replace(0, np.nan)

    out = genes[["rank", "probe_id", "gene_symbol", "snr"]].copy()
    out["train_diff"] = (tr_all.mean() - tr_aml.mean()).to_numpy()
    out["batch_mean_ALL"] = b_all.to_numpy()
    out["batch_mean_AML"] = b_aml.to_numpy()
    out["batch_diff"] = (b_all - b_aml).to_numpy()
    out["batch_diff_sd"] = out["batch_diff"].to_numpy() / pooled_sd.to_numpy()
    comparable = n_all > 0 and n_aml > 0
    same = np.sign(out["batch_diff"]).eq(np.sign(out["train_diff"])).astype("boolean")
    out["same_direction"] = same if comparable else pd.Series(pd.NA, index=out.index, dtype="boolean")
    small = out["batch_diff"].abs() < MIN_DIFF_LOG10
    out["status"] = np.where(not comparable, "Không so được",
                             np.where(small, "Chênh lệch nhỏ",
                                      np.where(out["same_direction"].fillna(False).astype(bool),
                                               "Giữ chiều", "Đảo chiều")))
    n_same = int(out["same_direction"].fillna(False).sum())
    n_kept, n_flipped, n_small = (int(out["status"].eq(k).sum()) for k in CONSISTENCY_STATUS[:3])
    label = source if source is not None else f"{len(batch)} mẫu đã chọn"
    p_perm, n_perm = np.nan, 0
    if comparable:
        Xb = pd.concat([b_all_x, b_aml_x]).to_numpy(dtype=float)
        is_all_b = np.r_[np.ones(n_all, dtype=bool), np.zeros(n_aml, dtype=bool)]
        p_perm, n_perm = _perm_same_direction(Xb, is_all_b, np.sign(out["train_diff"].to_numpy()))
        how = (f"duyệt đủ {n_perm} cách xếp nhãn" if n_perm <= PERM_EXACT_MAX
               else f"{n_perm - 1} lần xáo nhãn ngẫu nhiên")
        note = (f"{label}: {n_same}/{len(out)} gen dấu ấn cùng chiều ALL/AML với tập huấn luyện. Xét chênh lệch "
                f"≥ {MIN_DIFF_LOG10:g} log10: {n_kept} giữ chiều, {n_flipped} đảo chiều, {n_small} quá nhỏ để "
                f"kết luận (lô có {n_all} ALL, {n_aml} AML). Kiểm định hoán vị trong lô ({how}): "
                f"p = {p_perm:.3f}, nhỏ nhất có thể là {1 / n_perm:.3f}.")
        if min(n_all, n_aml) < MIN_N_PER_GROUP:
            note += f" Một lớp có dưới {MIN_N_PER_GROUP} mẫu nên kết luận yếu."
    else:
        have = "ALL" if n_all else "AML"
        note = (f"{label}: lô chỉ có {have}, không so được chiều ALL/AML. "
                "Dùng batch_anomalies() để kiểm tra không cần nhãn.")
    out.attrs.update(n_ALL=n_all, n_AML=n_aml, n_genes=len(out), n_same=n_same, n_kept=n_kept,
                     n_flipped=n_flipped, n_small=n_small, p_perm=p_perm, n_perm=n_perm,
                     comparable=comparable, note=note)
    return out


def batch_anomalies(source: str | None = None, sample_ids=None, n_per_side: int = 25,
                    only_test: bool = True) -> pd.DataFrame:
    """Lô mới CHƯA có chẩn đoán: mẫu nào có gen dấu ấn nằm ngoài khoảng của mọi mẫu tham chiếu.

    Khoảng tham chiếu của mỗi gen là [min, max] trên 38 mẫu huấn luyện (gộp ALL và AML). Không dùng nhãn của
    lô mới. Mức lệch (severity) = khoảng vượt ra ngoài / độ lệch chuẩn của gen trên tập huấn luyện.
    Cột: sample_id, source, tissue, probe_id, gene_symbol, value, ref_min, ref_max, side, severity, reason;
    sắp mức lệch giảm dần. Bảng rỗng nghĩa là không mẫu nào lệch khỏi tham chiếu ở các gen dấu ấn.
    Đây là kiểm tra mức gen cho nhà nghiên cứu; kiểm tra định dạng và chất lượng chip là việc của TV2 (2.4).
    """
    X, samples = _load_expression()
    batch = _batch_index(samples, source, sample_ids, only_test)
    genes = _marker_genes(n_per_side)
    genes = genes[genes["probe_id"].isin(X.columns)].set_index("probe_id")
    train = samples.index[samples["split"].eq("train")]
    ref = X.loc[train, genes.index]
    lo, hi, sd = ref.min(), ref.max(), ref.std().replace(0, np.nan)
    vals = X.loc[batch, genes.index]
    below, above = vals.lt(lo, axis=1), vals.gt(hi, axis=1)
    rows = []
    for sid in batch:
        for pid in genes.index:
            if not (below.at[sid, pid] or above.at[sid, pid]):
                continue
            v = vals.at[sid, pid]
            side = "thấp hơn" if below.at[sid, pid] else "cao hơn"
            gap = (lo[pid] - v) if side == "thấp hơn" else (v - hi[pid])
            name = genes.at[pid, "gene_symbol"] or pid
            bound = lo[pid] if side == "thấp hơn" else hi[pid]
            rows.append({
                "sample_id": int(sid), "source": samples.at[sid, "source"], "tissue": samples.at[sid, "tissue"],
                "probe_id": pid, "gene_symbol": genes.at[pid, "gene_symbol"], "value": v,
                "ref_min": lo[pid], "ref_max": hi[pid], "side": side,
                "severity": gap / sd[pid] if pd.notna(sd[pid]) else np.nan,
                "reason": (f"{name} của mẫu {int(sid)} {side} mọi mẫu tham chiếu "
                           f"({v:.2f} so với {'min' if side == 'thấp hơn' else 'max'} {bound:.2f})"),
            })
    cols = ["sample_id", "source", "tissue", "probe_id", "gene_symbol", "value", "ref_min", "ref_max",
            "side", "severity", "reason"]
    out = pd.DataFrame(rows, columns=cols)
    out = out.sort_values("severity", ascending=False).reset_index(drop=True)
    n_samples = out["sample_id"].nunique()
    out.attrs["note"] = (f"{n_samples}/{len(batch)} mẫu có ít nhất một gen dấu ấn ngoài khoảng tham chiếu "
                         f"(tổng {len(out)} trường hợp, xét {len(genes)} gen).")
    return out


# ---------------------------------------------------------------- chú giải

def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Gắn chức năng gen. Đọc gene_annotations.csv nếu có, không thì dùng description.

    Thêm các cột: function, cell_lineage, source_url, verified, function_source.
    function_source là "gene_annotations.csv" khi lấy từ file chú giải, "description" khi không.
    gene_symbol rỗng trong df được điền từ file chú giải nếu file có.
    df phải có cột probe_id.
    """
    if "probe_id" not in df.columns:
        raise ValueError("annotate cần cột probe_id")
    out = df.copy()
    if "description" not in out.columns:
        genes = _load_genes()[["probe_id", "description"]]
        out = out.merge(genes, on="probe_id", how="left")

    if ANNOTATION_PATH.exists():
        ann = pd.read_csv(ANNOTATION_PATH)
        missing = [c for c in ANNOTATION_COLUMNS if c not in ann.columns]
        if missing:
            raise ValueError(f"gene_annotations.csv thiếu cột {missing}")
        ann = ann[list(ANNOTATION_COLUMNS)].drop_duplicates("probe_id")
        ann = ann.rename(columns={c: f"_ann_{c}" for c in ANNOTATION_COLUMNS if c != "probe_id"})
        out = out.merge(ann, on="probe_id", how="left")
        matched = ~_is_blank(out["_ann_function"])
        out["function"] = out["_ann_function"].where(matched, out["description"])
        out["function_source"] = matched.map({True: "gene_annotations.csv", False: "description"})
        for col in ("gene_symbol", "cell_lineage", "source_url", "verified"):
            cur = out[col] if col in out.columns else pd.Series(pd.NA, index=out.index, dtype=object)
            out[col] = cur.where(~_is_blank(cur), out[f"_ann_{col}"])
        out = out.drop(columns=[c for c in out.columns if c.startswith("_ann_")])
    else:
        out["function"] = out["description"]
        out["function_source"] = "description"
        for col in ("cell_lineage", "source_url", "verified"):
            out[col] = pd.NA
    return out


if __name__ == "__main__":
    print(find_gene("CD33"))
    print(gene_stats("M23197_at", by=("class", "source")))
    print(disease_or_source("M23197_at"))
    print(ranking_table(top_n=10, direction="AML"))
    print("Nguồn dữ liệu:", data_source())
    rel = gene_reliability("M23197_at")
    print(rel, "\n", rel.attrs["note"])
    rob = robustness("M23197_at")
    print(rob[["check", "n_ALL", "n_AML", "diff", "fold_change", "same_direction"]], "\n", rob.attrs["note"])
    print(marker_consistency(source="St-Jude").attrs["note"])
    anom = batch_anomalies(source="CCG")
    print(anom.attrs["note"])
    print(anom.head(5)[["reason", "severity"]])
