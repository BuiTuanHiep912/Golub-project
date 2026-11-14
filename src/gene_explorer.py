"""TV4 - src/gene_explorer.py (công việc 2.2, Khám phá gen; ADR-014, ADR-051).

Tra cứu gen, thống kê biểu hiện theo nhóm, xếp hạng gen khác biệt, và trả lời câu hỏi
"gen này khác biệt do bệnh hay do nguồn mẫu" ở mức mô tả. Module chỉ tính toán, không vẽ hình
(hàm vẽ của TV4 nằm trong src/viz_gene.py) và không ghi file.

Giao diện đã khóa trong tests/test_interfaces.py, KHÔNG đổi tên khi chưa báo nhóm:
    find_gene('CD33')        -> DataFrame có probe_id, description (và các cột thêm)
    gene_stats(probe_id)     -> thống kê biểu hiện theo class, source, tissue

Đầu vào:
    load_golub("curated")                         -> X (72 x p), samples (72 x 6)
    data/standardized/genes.parquet               -> probe_id, description, is_control
    results/gene_lists/gene_ranking.csv           -> xếp hạng gen trên 38 mẫu huấn luyện
    results/gene_lists/golub50_comparison.csv     -> (tùy chọn) đối chiếu 50 gen của Golub
    results/gene_lists/gene_annotations.csv       -> (tùy chọn, TV5) chú giải gen

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

Chạy trực tiếp: python -m src.gene_explorer  (in ra một tra cứu mẫu)
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import pandas as pd

try:
    from src.config import CURATED_DIR, GENE_LISTS_DIR, STD_DIR
    from src.gene_catalog import split_symbol
    from src.load import load_golub
except ModuleNotFoundError:
    from config import CURATED_DIR, GENE_LISTS_DIR, STD_DIR
    from gene_catalog import split_symbol
    from load import load_golub

RANKING_PATH = GENE_LISTS_DIR / "gene_ranking.csv"
GOLUB50_PATH = GENE_LISTS_DIR / "golub50_comparison.csv"
ANNOTATION_PATH = GENE_LISTS_DIR / "gene_annotations.csv"
GENES_PATH = STD_DIR / "genes.parquet"
CURATED_FILES = (CURATED_DIR / "X_log10.parquet", CURATED_DIR / "samples.parquet")

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
def _expression_cached(_key: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_golub("curated")


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


# ---------------------------------------------------------------- đọc dữ liệu

def _load_expression() -> tuple[pd.DataFrame, pd.DataFrame]:
    """X (72 x p, log10) và samples (72 x 6), cùng index sample_id. Không sửa trực tiếp."""
    return _expression_cached(tuple(_mtime(p) for p in CURATED_FILES))


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
    """Một dòng mỗi mẫu: sample_id, value (log10), class, subtype, source, tissue, split."""
    X, samples = _load_expression()
    _check_probe(probe_id, X)
    out = samples.copy()
    out["value"] = X[probe_id].reindex(out.index)
    out.index.name = "sample_id"
    return out.reset_index()[["sample_id", "value", *GROUP_COLUMNS]]


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

def ranking_table(top_n: int = 50, direction: str = "both", min_abs_snr: float = 0.0) -> pd.DataFrame:
    """Bảng xếp hạng gen từ gene_ranking.csv.

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
    rank = rank.sort_values("rank").head(top_n).copy()
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
