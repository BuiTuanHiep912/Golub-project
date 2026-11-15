"""Dữ liệu GIẢ đúng định dạng, để dựng khung dashboard khi các việc 2.1–2.5 chưa bàn giao.

Công việc: 2.6 — Phụ trách: TV6 — Kế hoạch giai đoạn 2, mục 2.6 và mục 4 (giao diện dùng chung).

Mọi hàm ở đây TRÙNG TÊN, TRÙNG CHỮ KÝ và TRẢ VỀ CÙNG ĐỊNH DẠNG với hàm thật của các thành viên:
    ingest_batches(), validate_new_sample(path)          ~ src/ingest.py        (TV2)
    patient_profile(sample_id=None, values=None)         ~ src/patient.py       (TV1)
    find_gene(), gene_stats(), gene_values(),
    disease_or_source(), ranking_table(), annotate()     ~ src/gene_explorer.py (TV4)
    reference_data()                                     ~ src/load.load_golub  (đọc dữ liệu)
Nhờ vậy khi hàm thật xong chỉ cần đổi GOLUB_APP_DATA=real (xem backend.py), các trang không phải sửa.

Cái gì là thật, cái gì là giả:
  - Tên probe, mô tả gen, SNR và hạng của 3541 gen lấy từ results/gene_lists/gene_ranking.csv (kết quả thật
    của giai đoạn 1). Thêm 3588 probe "MOCK_*" cho đủ 7129 gen.
  - Giá trị biểu hiện của 72 bệnh nhân là SỐ NGẪU NHIÊN (seed cố định), chỉ mô phỏng khác biệt ALL/AML theo SNR.
    Bảng mẫu (tập huấn luyện/kiểm tra, nguồn, loại mô) mô phỏng cấu trúc Golub; mẫu 21 gắn cờ qc_outlier.
  - Dự đoán của patient_profile() là Golub weighted voting tính trên dữ liệu giả; KHÔNG phải kết quả của TV1.
  - Chú giải chức năng chỉ có cho vài gen quen thuộc; còn lại để trống (việc của TV5).
KHÔNG dùng bất kỳ con số nào từ dữ liệu giả này trong báo cáo hay slide.
"""
from __future__ import annotations

from functools import lru_cache
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.gene_catalog import split_symbol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANKING_CSV = PROJECT_ROOT / "results" / "gene_lists" / "gene_ranking.csv"

SEED = 7
N_GENES = 7129
FLOOR, CEIL = 100, 16000          # ngưỡng của tầng curated (threshold rồi log10)
PS_THRESHOLD = 0.3
N_SIMILAR = 5
N_VOTE_GENES = 50                 # 25 gen cao ở ALL + 25 gen cao ở AML (bài báo Golub)
IS_MOCK = True
RANKING_NOTE = "Xếp hạng tính trên 38 mẫu huấn luyện, chỉ dùng để khám phá"
BATCH_SOURCES = ("DFCI", "CALGB", "St-Jude", "CCG")
OUTLIER_ID = 21                   # ví dụ mục D.1: mẫu 21, tương quan thấp
AMBIGUOUS_TEST_IDS = (47, 66)     # hai bệnh nhân kiểm tra có tín hiệu yếu, để thấy ca "không chắc chắn"
BLOCKED_BATCH = "CCG"             # một lô bị chặn, để thấy trạng thái lỗi trên dashboard

# Chú giải chỉ cho vài gen quen thuộc; phần còn lại là việc của TV5 (gene_annotations.csv)
_ANNOTATIONS = {
    "CD33": ("Kháng nguyên biệt hóa dòng tủy, dấu ấn bề mặt của tế bào tủy", "dòng tủy"),
    "MPO": ("Myeloperoxidase, enzym của bạch cầu hạt và tế bào tủy", "dòng tủy"),
    "CST3": ("Cystatin C, chất ức chế protease cystein", "không đặc hiệu"),
    "ZYX": ("Zyxin, protein bám dính ổ (focal adhesion)", "không đặc hiệu"),
    "LTC4S": ("Leukotriene C4 synthase, tổng hợp chất trung gian viêm", "dòng tủy"),
    "ELA2": ("Elastase bạch cầu trung tính", "dòng tủy"),
    "CD19": ("Kháng nguyên bề mặt tế bào B", "dòng lympho B"),
    "MYB": ("c-myb, yếu tố phiên mã của tạo máu", "tạo máu"),
}


# ---------------------------------------------------------------------------------------------------
# 1. Bảng gen, bảng mẫu, ma trận biểu hiện
# ---------------------------------------------------------------------------------------------------
def _split_symbol(desc: str) -> str:
    """Ký hiệu gen tách từ mô tả, dùng đúng hàm của TV4 (src/gene_catalog.py) để khớp bản thật."""
    return split_symbol(str(desc))[0]


@lru_cache(maxsize=1)
def _genes() -> pd.DataFrame:
    """7129 probe: 3541 probe thật trong gene_ranking.csv + probe MOCK_*."""
    rk = pd.read_csv(RANKING_CSV)
    n_fill = N_GENES - len(rk)
    fill = pd.DataFrame({
        "probe_id": [f"MOCK_{i:05d}_at" for i in range(n_fill)],
        "description": "(probe giả lập, không có trong dữ liệu thật)",
        "snr": 0.0,
    })
    genes = pd.concat([rk[["probe_id", "description", "snr"]], fill], ignore_index=True)
    genes = genes.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # trộn cho khỏi lộ ranh giới
    genes["gene_symbol"] = genes["description"].map(_split_symbol)
    genes["is_control"] = genes["probe_id"].str.startswith("AFFX")
    return genes


@lru_cache(maxsize=1)
def _ranking() -> pd.DataFrame:
    rk = pd.read_csv(RANKING_CSV).copy()
    rk["gene_symbol"] = rk["description"].map(_split_symbol)
    rk["in_golub50"] = False
    return rk


@lru_cache(maxsize=1)
def _samples_base() -> pd.DataFrame:
    """72 bệnh nhân theo cấu trúc Golub: 38 train (27 ALL DFCI, 11 AML) và 34 test (20 ALL, 14 AML)."""
    ids = np.arange(1, 73)
    split = np.where(ids <= 38, "train", "test")
    klass = np.where((ids <= 27) | ((ids >= 39) & (ids <= 58)), "ALL", "AML")
    source = np.empty(72, dtype=object)
    for i in ids:
        if i <= 27:
            source[i - 1] = "DFCI"
        elif i <= 38:                                  # 11 AML huấn luyện: 8 CALGB, 3 CCG
            source[i - 1] = "CALGB" if i <= 35 else "CCG"
        elif i <= 58:                                  # 20 ALL kiểm tra: 3 St-Jude, 17 DFCI
            source[i - 1] = "St-Jude" if i in (40, 41, 42) else "DFCI"
        else:                                          # 14 AML kiểm tra: 5 St-Jude, 6 CALGB, 3 CCG
            source[i - 1] = ("St-Jude" if i in (59, 60, 61, 62, 63) else "CALGB" if i <= 69 else "CCG")
    subtype = np.where(klass == "AML", "AML", "B-ALL")
    subtype[[6, 7, 8, 9, 10, 11, 39, 40, 41]] = "T-ALL"   # sample_id 7–12 và 40–42
    tissue = np.full(72, "BM", dtype=object)
    tissue[[28, 32, 36, 44, 59, 62, 64, 67]] = "PB"       # sample_id 29, 33, 37, 45, 60, 63, 65, 68
    rng = np.random.default_rng(SEED)
    sex = rng.choice(["M", "F", None], size=72, p=[0.5, 0.4, 0.1])
    df = pd.DataFrame({"split": split, "class": klass, "subtype": subtype, "tissue": tissue,
                       "source": source, "sex": sex}, index=pd.Index(ids, name="sample_id"))
    df["qc_outlier"] = df.index == OUTLIER_ID
    df["qc_reason"] = np.where(df["qc_outlier"], "tương quan thấp với các mẫu cùng nguồn", "")
    return df


@lru_cache(maxsize=1)
def _matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(raw MAS4 có giá trị âm, log10 sau threshold [100, 16000]); 72 × 7129, mỗi hàng một bệnh nhân."""
    rng = np.random.default_rng(SEED)
    genes, samples = _genes(), _samples_base()
    n, p = len(samples), len(genes)
    snr = genes["snr"].to_numpy()
    in_rank = snr != 0
    base = np.where(in_rank, rng.uniform(2.2, 3.5, p), rng.uniform(0.9, 2.3, p))   # probe giả thường thấp
    sd = rng.uniform(0.12, 0.30, p)
    y = np.where(samples["class"].to_numpy() == "ALL", 0.5, -0.5)                  # ALL +, AML −
    strength = np.ones(n)
    strength[[i - 1 for i in AMBIGUOUS_TEST_IDS]] = 0.12                           # tín hiệu yếu
    L = base[None, :] + (y * strength)[:, None] * snr[None, :] * 2 * sd[None, :]   # snr = Δ / (s1 + s2)
    L += rng.normal(0, 1, (n, p)) * sd[None, :]
    L += rng.normal(0, 0.04, (n, 1))                                               # độ lệch mỗi chip
    # hiệu ứng lô nhỏ: máu ngoại vi và St-Jude lệch ở ~10% gen
    shift_idx = rng.choice(p, size=p // 10, replace=False)
    L[np.ix_((samples["tissue"] == "PB").to_numpy(), shift_idx)] += 0.10
    L[np.ix_((samples["source"] == "St-Jude").to_numpy(), shift_idx)] -= 0.08
    # mẫu 21: chip kém, mất tín hiệu và nhiễu nhiều
    o = int(np.flatnonzero(samples.index == OUTLIER_ID)[0])
    L[o] = base + rng.normal(0, 1, p) * sd * 1.15 + rng.normal(0, 0.05)
    raw = 10.0 ** L
    raw = np.where(L < 1.6, -raw * rng.uniform(0.2, 1.0, (n, p)), raw)             # giá trị thấp -> âm như MAS4
    raw = np.round(raw)
    raw_df = pd.DataFrame(raw, index=samples.index, columns=genes["probe_id"])
    log_df = np.log10(raw_df.clip(FLOOR, CEIL))
    return raw_df, log_df


@lru_cache(maxsize=1)
def reference_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(X, samples): X = log10 sau threshold (72 × 7129), samples có thêm qc_outlier và qc_reason."""
    _, X = _matrices()
    return X.copy(), _samples_base().copy()


# ---------------------------------------------------------------------------------------------------
# 2. TV2 — ingest_batches(), validate_new_sample()
# ---------------------------------------------------------------------------------------------------
def _batch_corr(X: pd.DataFrame, idx) -> float:
    """Tương quan Pearson trung bình giữa các mẫu của một lô (trên log10, chưa chuẩn hóa từng gen)."""
    c = np.corrcoef(X.loc[idx].to_numpy())
    return float(c[~np.eye(len(c), dtype=bool)].mean())


@lru_cache(maxsize=1)
def _ingest_batches() -> pd.DataFrame:
    raw, X = _matrices()
    s = _samples_base()
    rows = []
    for src in BATCH_SOURCES:
        idx = s.index[s["source"] == src]
        v = X.loc[idx]
        blocked = src == BLOCKED_BATCH
        rows.append({
            "batch_id": src, "source": src, "n_samples": len(idx),
            "status": "blocked" if blocked else "passed",
            "reason": "Chỉ có 7117/7129 gen (thiếu 12 gen), lô bị chặn" if blocked else "",
            "median_log10": float(np.median(v.to_numpy())),
            "iqr_log10": float(np.subtract(*np.percentile(v.to_numpy(), [75, 25]))),
            "mean_corr": _batch_corr(X, idx),
            "pct_negative": float((raw.loc[idx].to_numpy() < 0).mean()),
            "n_flagged": int(s.loc[idx, "qc_outlier"].sum()),
        })
    return pd.DataFrame(rows)


def ingest_batches() -> pd.DataFrame:
    """DataFrame chất lượng theo lô, một hàng mỗi lô (đúng cột đã thống nhất với TV2)."""
    return _ingest_batches().copy()


def sample_warnings() -> pd.DataFrame:
    """Danh sách cảnh báo: [sample_id, source, reason]. (Nguồn của lý do do TV2 chốt; ở đây lấy qc_reason.)"""
    s = _samples_base()
    w = s[s["qc_outlier"]].reset_index()[["sample_id", "source", "qc_reason"]]
    return w.rename(columns={"qc_reason": "reason"})


def _read_sample_file(path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def validate_new_sample(path) -> dict:
    """Kiểm tra file một mẫu mới (CSV/TSV hai cột probe_id, value; giá trị gốc MAS4).

    Trả về {'ok', 'errors', 'warnings', 'values'}; `values` là Series log10 (đã threshold [100, 16000])
    theo thứ tự probe của tầng curated, hoặc None khi có lỗi.
    """
    errors: list[str] = []
    warnings: list[str] = []
    genes = _genes()
    try:
        df = _read_sample_file(path)
    except Exception as exc:  # noqa: BLE001 - file hỏng đủ kiểu, chỉ cần báo lý do
        return {"ok": False, "errors": [f"Không đọc được file: {type(exc).__name__}"], "warnings": [], "values": None}
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"probe_id", "value"} <= set(df.columns):
        return {"ok": False, "errors": ["Thiếu cột 'probe_id' hoặc 'value'"], "warnings": [], "values": None}
    df = df[["probe_id", "value"]]
    if len(df) != N_GENES:
        errors.append(f"Số gen là {len(df)}, cần đúng {N_GENES}")
    unknown = sorted(set(df["probe_id"].astype(str)) - set(genes["probe_id"]))
    if unknown:
        errors.append(f"{len(unknown)} mã gen không có trong chip (ví dụ {unknown[0]})")
    if df["probe_id"].duplicated().any():
        errors.append("Có mã gen bị lặp")
    if df["value"].isna().any():
        errors.append(f"Có {int(df['value'].isna().sum())} ô trống")
    num = pd.to_numeric(df["value"], errors="coerce")
    if (num.isna() & df["value"].notna()).any():
        errors.append("Có giá trị không phải số")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "values": None}
    vals = pd.Series(num.to_numpy(), index=df["probe_id"].astype(str)).reindex(genes["probe_id"])
    n_neg = int((vals < 0).sum())
    if n_neg:
        warnings.append(f"{n_neg} giá trị âm ({n_neg / N_GENES:.1%}), sẽ được đưa về ngưỡng {FLOOR}")
    log = np.log10(vals.clip(FLOOR, CEIL))
    ref = reference_data()[0]
    r = float(np.corrcoef(log.to_numpy(), ref.mean().to_numpy())[0, 1])
    if r < 0.6:
        warnings.append(f"Tương quan với hồ sơ trung bình của dữ liệu tham chiếu thấp ({r:.2f})")
    return {"ok": True, "errors": [], "warnings": warnings, "values": log}


def make_sample_file(kind: str = "valid", sample_id: int = 52) -> bytes:
    """File CSV mẫu để thử nút tải lên: 'valid' (hợp lệ), 'missing_genes' (thiếu gen), 'blank' (ô trống)."""
    raw, _ = _matrices()
    df = pd.DataFrame({"probe_id": raw.columns, "value": raw.loc[sample_id].to_numpy()})
    if kind == "missing_genes":
        df = df.iloc[: N_GENES - 120]
    elif kind == "blank":
        df.loc[df.sample(25, random_state=1).index, "value"] = np.nan
    elif kind != "valid":
        raise ValueError(kind)
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------------------------------
# 3. TV4 — find_gene(), gene_values(), gene_stats(), disease_or_source(), ranking_table(), annotate()
# ---------------------------------------------------------------------------------------------------
FIND_COLUMNS = ["probe_id", "gene_symbol", "description", "n_probes_same_description", "rank", "match_type"]
GROUP_COLUMNS = ("class", "subtype", "source", "tissue", "split")


def find_gene(query: str, limit: int = 20) -> pd.DataFrame:
    q = str(query).strip()
    if not q:
        raise ValueError("query rỗng")
    genes = _genes()
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
    ranks = _ranking().set_index("probe_id")["rank"]
    hits["rank"] = hits["probe_id"].map(ranks).astype("Int64")
    hits["_o"] = hits["match_type"].map({"probe": 0, "symbol": 1, "text": 2})
    hits = hits.sort_values(["_o", "probe_id"]).head(limit)
    return hits[FIND_COLUMNS].reset_index(drop=True)


def gene_values(probe_id: str) -> pd.DataFrame:
    X, samples = reference_data()
    if probe_id not in X.columns:
        raise KeyError(probe_id)
    out = samples.copy()
    out["value"] = X[probe_id]
    return out.reset_index()[["sample_id", "value", *GROUP_COLUMNS]]


def gene_stats(probe_id: str, by: tuple[str, ...] = ("class", "source", "tissue")) -> pd.DataFrame:
    by = tuple(by)
    if not by or any(b not in GROUP_COLUMNS for b in by):
        raise ValueError(f"by phải là tổ hợp của {GROUP_COLUMNS}, nhận {by}")
    df = gene_values(probe_id)
    return (df.groupby(list(by), observed=True)["value"]
              .agg(n="count", mean="mean", median="median", sd="std").reset_index())


def disease_or_source(probe_id: str) -> pd.DataFrame:
    df = gene_values(probe_id)
    rows = []
    for source, sub in df.groupby("source", observed=True):
        a = sub.loc[sub["class"] == "ALL", "value"]
        m = sub.loc[sub["class"] == "AML", "value"]
        ma = a.mean() if len(a) else float("nan")
        mm = m.mean() if len(m) else float("nan")
        rows.append({"source": source, "n_ALL": len(a), "n_AML": len(m), "mean_ALL": ma, "mean_AML": mm,
                     "diff": ma - mm, "comparable": len(a) >= 3 and len(m) >= 3})
    return pd.DataFrame(rows)


def ranking_table(top_n: int = 50, direction: str = "both", min_abs_snr: float = 0.0) -> pd.DataFrame:
    if direction not in ("ALL", "AML", "both"):
        raise ValueError(direction)
    rk = _ranking()
    if direction == "ALL":
        rk = rk[rk["snr"] > 0]
    elif direction == "AML":
        rk = rk[rk["snr"] < 0]
    rk = rk[rk["snr"].abs() >= min_abs_snr].sort_values("rank").head(top_n).copy()
    rk["direction"] = rk["snr"].map(lambda s: "ALL" if s > 0 else "AML")
    cols = ["rank", "probe_id", "gene_symbol", "description", "snr", "t_stat", "q_value", "direction", "in_golub50"]
    out = rk[cols].reset_index(drop=True)
    out.attrs["note"] = RANKING_NOTE
    return out


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột function, cell_lineage (chỉ vài gen có chú giải giả; còn lại trống)."""
    out = df.copy()
    sym = out["gene_symbol"] if "gene_symbol" in out else pd.Series("", index=out.index)
    out["function"] = sym.map(lambda s: _ANNOTATIONS.get(s, (None, None))[0])
    out["cell_lineage"] = sym.map(lambda s: _ANNOTATIONS.get(s, (None, None))[1])
    out.attrs.update(df.attrs)
    return out


# ---------------------------------------------------------------------------------------------------
# 4. TV1 — patient_profile()
# ---------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _wv_model() -> dict:
    """Golub weighted voting trên 38 mẫu train của dữ liệu giả: 25 gen cao ở ALL + 25 gen cao ở AML."""
    X, s = reference_data()
    tr = s.index[s["split"] == "train"]
    a, m = X.loc[tr[s.loc[tr, "class"] == "ALL"]], X.loc[tr[s.loc[tr, "class"] == "AML"]]
    snr = (a.mean() - m.mean()) / (a.std() + m.std())
    use = list(snr.nlargest(N_VOTE_GENES // 2).index) + list(snr.nsmallest(N_VOTE_GENES // 2).index)
    return {"probes": use, "snr": snr[use], "mid": ((a.mean() + m.mean()) / 2)[use], "all_snr": snr}


def _vote(values: pd.Series, exclude_id: int | None = None) -> dict:
    mdl = _wv_model()
    g = _genes().set_index("probe_id")["description"]
    x = values[mdl["probes"]]
    v = mdl["snr"] * (x - mdl["mid"])
    v_all, v_aml = v[v > 0].sum(), -v[v < 0].sum()
    win, lose = max(v_all, v_aml), min(v_all, v_aml)
    ps = float((win - lose) / (win + lose))
    pred = "uncertain" if ps < PS_THRESHOLD else ("ALL" if v_all > v_aml else "AML")
    votes = pd.DataFrame({"probe_id": mdl["probes"], "description": g.reindex(mdl["probes"]).to_numpy(),
                          "value": x.to_numpy(), "vote": v.to_numpy(),
                          "toward": np.where(v.to_numpy() > 0, "ALL", "AML")})
    votes = votes.reindex(votes["vote"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    X, s = reference_data()
    ref = s.index[(s["split"] == "train") & (s.index != exclude_id)]
    top200 = list(mdl["all_snr"].abs().nlargest(200).index)       # tương quan trên 200 gen phân biệt nhất
    Xs, xs = X[top200], values[top200]
    Z, z = (Xs - Xs.mean()) / Xs.std(), (xs - Xs.mean()) / Xs.std()
    corr = Z.loc[ref].apply(lambda r: np.corrcoef(r.to_numpy(), z.to_numpy())[0, 1], axis=1)
    top = corr.nlargest(N_SIMILAR)
    similar = pd.DataFrame({"sample_id": top.index, "correlation": top.to_numpy(),
                            "class": s.loc[top.index, "class"].to_numpy()})
    return {"prediction": pred, "ps": ps, "votes": votes, "similar": similar}


def patient_profile(sample_id: int | None = None, values=None) -> dict:
    """Hồ sơ một bệnh nhân có sẵn (sample_id) hoặc một mẫu mới (values: Series log10 theo probe_id)."""
    if (sample_id is None) == (values is None):
        raise ValueError("Truyền đúng một trong sample_id hoặc values")
    X, _ = reference_data()
    if sample_id is not None:
        return _vote(X.loc[sample_id], exclude_id=sample_id)
    return _vote(pd.Series(values).reindex(X.columns))
