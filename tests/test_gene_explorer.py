"""Kiểm tra src/gene_explorer.py (TV4, công việc 2.2) trên dữ liệu tổng hợp.

Dữ liệu giả cùng cỡ Golub: 47 ALL + 25 AML; phân bố nguồn như handover 1.1:
  train: DFCI 27 ALL | CALGB 11 AML
  test : DFCI 17 ALL | St-Jude 3 ALL + 5 AML | CCG 9 AML
Test thật với dữ liệu curated được bỏ qua khi chưa có file.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import gene_explorer as ge  # noqa: E402
from src.config import CURATED_DIR  # noqa: E402

PROBES = {
    "M23197_at": "CD33 CD33 antigen (differentiation antigen)",
    "X95735_at": "ZYX Zyxin",
    "U12345_at": "ZYX Zyxin",                  # cùng mô tả với probe trên
    "AFFX-BioB-5_at": "AFFX-BioB-5_at (endogenous control)",
}


def _samples() -> pd.DataFrame:
    rows = []
    sid = 1
    spec = [
        ("train", "ALL", "B-ALL", "BM", "DFCI", 27),
        ("train", "AML", "AML", "BM", "CALGB", 11),
        ("test", "ALL", "B-ALL", "BM", "DFCI", 17),
        ("test", "ALL", "B-ALL", "BM", "St-Jude", 3),
        ("test", "AML", "AML", "BM", "St-Jude", 5),
        ("test", "AML", "AML", "BM", "CCG", 9),
    ]
    for split, cls, sub, tissue, src, n in spec:
        for _ in range(n):
            rows.append({"sample_id": sid, "split": split, "class": cls, "subtype": sub,
                         "tissue": tissue, "source": src})
            sid += 1
    return pd.DataFrame(rows).set_index("sample_id")


def _X(samples: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(2.0, 0.1, size=(len(samples), len(PROBES))),
                     index=samples.index, columns=list(PROBES))
    aml = samples["class"].eq("AML").to_numpy()
    X.loc[aml, "M23197_at"] += 1.0            # CD33 cao ở AML trong mọi nguồn
    return X


@pytest.fixture
def env(tmp_path, monkeypatch):
    samples = _samples()
    X = _X(samples)
    monkeypatch.setattr(ge, "_load_expression", lambda: (X, samples))

    genes_path = tmp_path / "genes.parquet"
    pd.DataFrame({
        "probe_id": list(PROBES),
        "description": list(PROBES.values()),
        "is_control": [p.startswith("AFFX") for p in PROBES],
    }).to_parquet(genes_path)
    monkeypatch.setattr(ge, "GENES_PATH", genes_path)

    ranking_path = tmp_path / "gene_ranking.csv"
    pd.DataFrame({
        "probe_id": ["M23197_at", "X95735_at", "U12345_at", "L00001_at", "L00002_at"],
        "description": ["CD33 CD33 antigen (differentiation antigen)", "ZYX Zyxin", "ZYX Zyxin",
                        "TCF3 transcription factor 3", "CD19 CD19 antigen"],
        "snr": [-1.41, -1.60, 0.90, 1.20, 0.30],
        "t_stat": [-7.3, -9.6, 4.1, 5.0, 1.2],
        "p_value": [1e-6, 1e-10, 1e-4, 1e-5, 0.2],
        "q_value": [2e-4, 4e-7, 1e-3, 5e-4, 0.3],
        "rank": [4, 2, 6, 1, 3],
    }).to_csv(ranking_path, index=False)
    monkeypatch.setattr(ge, "RANKING_PATH", ranking_path)

    g50_path = tmp_path / "golub50_comparison.csv"
    pd.DataFrame({
        "phía": ["AML", "ALL"],
        "gen Golub": ["CD33", "CD19"],
        "probe tốt nhất": ["M23197_at", "L00002_at"],
        "rank của nhóm": [4, 3],
        "kết quả": ["trùng", "trùng"],
    }).to_csv(g50_path, index=False)
    monkeypatch.setattr(ge, "GOLUB50_PATH", g50_path)

    monkeypatch.setattr(ge, "ANNOTATION_PATH", tmp_path / "missing_annotations.csv")
    return {"samples": samples, "X": X, "tmp": tmp_path}


# ---------------------------------------------------------------- tra cứu

def test_find_gene_by_symbol_is_first(env):
    out = ge.find_gene("CD33")
    assert out.iloc[0]["probe_id"] == "M23197_at"
    assert out.iloc[0]["match_type"] == "symbol"


def test_find_gene_probe_and_case_insensitive_symbol(env):
    assert ge.find_gene("U12345_at").iloc[0]["match_type"] == "probe"
    assert set(ge.find_gene("zyxin")["probe_id"]) == {"X95735_at", "U12345_at"}
    assert ge.find_gene("zyx").iloc[0]["match_type"] == "symbol"   # 'zyx' khớp symbol 'ZYX' không phân biệt hoa thường


def test_find_gene_counts_probes_sharing_description(env):
    out = ge.find_gene("Zyxin")
    assert (out["n_probes_same_description"] == 2).all()


def test_find_gene_no_match_returns_empty(env):
    out = ge.find_gene("KHONGCO")
    assert out.empty
    assert list(out.columns) == ge.FIND_COLUMNS


def test_find_gene_probe_is_case_insensitive(env):
    out = ge.find_gene("m23197_AT")
    assert out.iloc[0]["probe_id"] == "M23197_at"
    assert out.iloc[0]["match_type"] == "probe"


def test_find_gene_reports_rank_or_blank(env):
    assert ge.find_gene("CD33").iloc[0]["rank"] == 4
    affx = ge.find_gene("AFFX-BioB-5_at")
    assert pd.isna(affx.iloc[0]["rank"])                    # probe điều khiển không có hạng


def test_loaders_are_cached(env):
    ge.clear_cache()
    ge.find_gene("CD33")
    before = ge._genes_cached.cache_info().hits
    ge.find_gene("Zyxin")
    assert ge._genes_cached.cache_info().hits > before


def test_cached_result_is_not_mutated_by_caller(env):
    out = ge.find_gene("CD33")
    out.loc[0, "gene_symbol"] = "SUA"
    assert ge.find_gene("CD33").iloc[0]["gene_symbol"] == "CD33"


def test_find_gene_empty_query_raises(env):
    with pytest.raises(ValueError):
        ge.find_gene("   ")


# ---------------------------------------------------------------- biểu hiện

def test_gene_values_has_one_row_per_sample(env):
    vals = ge.gene_values("M23197_at")
    assert len(vals) == 72
    assert {"sample_id", "value", "class", "subtype", "source", "tissue", "split"} <= set(vals.columns)


def test_gene_values_unknown_probe_raises(env):
    with pytest.raises(KeyError):
        ge.gene_values("NOPE_at")


def test_gene_stats_counts_match_golub_groups(env):
    st = ge.gene_stats("M23197_at", by=("class",)).set_index("class")
    assert st.loc["ALL", "n"] == 47
    assert st.loc["AML", "n"] == 25
    assert st["n"].sum() == 72


def test_gene_stats_default_groups_by_class_source_tissue(env):
    st = ge.gene_stats("M23197_at")                         # giao diện khóa: gene_stats(probe_id)
    assert {"class", "source", "tissue"} <= set(st.columns)
    assert st["n"].sum() == 72


def test_gene_stats_rejects_unknown_group(env):
    with pytest.raises(ValueError):
        ge.gene_stats("M23197_at", by=("outcome",))


# ---------------------------------------------------------------- bệnh hay nguồn

def test_disease_or_source_comparable_only_where_both_classes_present(env):
    out = ge.disease_or_source("M23197_at").set_index("source")
    assert bool(out.loc["St-Jude", "comparable"]) is True     # 3 ALL, 5 AML
    assert bool(out.loc["DFCI", "comparable"]) is False       # chỉ ALL
    assert bool(out.loc["CALGB", "comparable"]) is False      # chỉ AML
    assert bool(out.loc["CCG", "comparable"]) is False        # chỉ AML


def test_disease_or_source_diff_sign(env):
    out = ge.disease_or_source("M23197_at").set_index("source")
    # CD33 cao hơn 1.0 ở AML, nên ALL - AML < 0 trong nguồn có cả hai nhóm
    assert out.loc["St-Jude", "diff"] < 0


# ---------------------------------------------------------------- xếp hạng

def test_ranking_table_direction_filters(env):
    aml = ge.ranking_table(top_n=10, direction="AML")
    assert (aml["snr"] < 0).all() and set(aml["direction"]) == {"AML"}
    allo = ge.ranking_table(top_n=10, direction="ALL")
    assert (allo["snr"] > 0).all() and set(allo["direction"]) == {"ALL"}


def test_ranking_table_sorted_and_note(env):
    out = ge.ranking_table(top_n=10)
    assert out["rank"].is_monotonic_increasing
    assert out.attrs["note"] == ge.RANKING_NOTE


def test_ranking_table_bad_direction_raises(env):
    with pytest.raises(ValueError):
        ge.ranking_table(direction="up")


def test_ranking_table_in_golub50_flag(env):
    out = ge.ranking_table(top_n=10).set_index("probe_id")
    assert bool(out.loc["M23197_at", "in_golub50"]) is True
    assert bool(out.loc["X95735_at", "in_golub50"]) is False


def test_ranking_table_top_n(env):
    assert len(ge.ranking_table(top_n=2)) == 2


def test_golub50_file_missing_columns_gives_na_and_warns(env, monkeypatch):
    bad = env["tmp"] / "golub50_bad.csv"
    pd.DataFrame({"gene": ["CD33"], "probe": ["M23197_at"]}).to_csv(bad, index=False)
    monkeypatch.setattr(ge, "GOLUB50_PATH", bad)
    with pytest.warns(UserWarning):
        out = ge.ranking_table(top_n=10)
    assert out["in_golub50"].isna().all()


def test_collapse_by_symbol_falls_back_to_probe_when_all_blank(env):
    df = pd.DataFrame({"probe_id": ["a_at", "b_at"], "gene_symbol": ["", ""],
                       "description": ["", ""], "snr": [1.0, -2.0]})
    out = ge.collapse_by_symbol(df)
    assert len(out) == 2                                    # không gộp nhầm hai probe khác nhau


def test_collapse_by_symbol_keeps_strongest_and_counts(env):
    ranking = ge.ranking_table(top_n=10)
    out = ge.collapse_by_symbol(ranking).set_index("gene_symbol")
    assert out.loc["ZYX", "probe_id"] == "X95735_at"         # |-1.60| > |0.90|
    assert out.loc["ZYX", "n_probes"] == 2
    assert not out.index.duplicated().any()


def test_collapse_by_symbol_requires_snr(env):
    with pytest.raises(ValueError):
        ge.collapse_by_symbol(pd.DataFrame({"probe_id": ["a"]}))


# ---------------------------------------------------------------- chú giải

def test_annotate_falls_back_to_description(env):
    out = ge.annotate(ge.ranking_table(top_n=3))
    assert (out["function_source"] == "description").all()
    assert out["function"].notna().all()


def test_annotate_uses_file_when_present(env):
    path = env["tmp"] / "annotations.csv"
    pd.DataFrame({
        "probe_id": ["M23197_at"],
        "gene_symbol": ["CD33"],
        "function": ["Thụ thể bề mặt tế bào tủy"],
        "cell_lineage": ["tủy"],
        "source_url": ["https://www.ncbi.nlm.nih.gov/gene/945"],
        "verified": ["yes"],
    }).to_csv(path, index=False)
    ge.ANNOTATION_PATH = path
    try:
        out = ge.annotate(ge.ranking_table(top_n=10)).set_index("probe_id")
    finally:
        ge.ANNOTATION_PATH = ge.GENE_LISTS_DIR / "gene_annotations.csv"
    assert out.loc["M23197_at", "function_source"] == "gene_annotations.csv"
    assert out.loc["X95735_at", "function_source"] == "description"


def test_annotate_fills_blank_symbol_from_file(env, monkeypatch):
    path = env["tmp"] / "annotations_symbol.csv"
    pd.DataFrame({
        "probe_id": ["X95735_at"], "gene_symbol": ["ZYX"], "function": ["Protein bám dính"],
        "cell_lineage": ["tủy"], "source_url": ["https://www.ncbi.nlm.nih.gov/gene/7791"],
        "verified": ["yes"],
    }).to_csv(path, index=False)
    monkeypatch.setattr(ge, "ANNOTATION_PATH", path)
    df = pd.DataFrame({"probe_id": ["X95735_at"], "gene_symbol": [""], "description": ["Zyxin"]})
    out = ge.annotate(df)
    assert out.loc[0, "gene_symbol"] == "ZYX"
    assert out.loc[0, "function_source"] == "gene_annotations.csv"


def test_annotate_rejects_file_missing_columns(env):
    path = env["tmp"] / "bad.csv"
    pd.DataFrame({"probe_id": ["M23197_at"], "function": ["x"]}).to_csv(path, index=False)
    ge.ANNOTATION_PATH = path
    try:
        with pytest.raises(ValueError):
            ge.annotate(ge.ranking_table(top_n=3))
    finally:
        ge.ANNOTATION_PATH = ge.GENE_LISTS_DIR / "gene_annotations.csv"


# ---------------------------------------------------------------- dữ liệu thật

@pytest.mark.skipif(not (CURATED_DIR / "X_log10.parquet").exists(),
                    reason="chưa có tầng curated")
def test_real_data_cd33_smoke():
    hit = ge.find_gene("CD33")
    assert not hit.empty
    stats = ge.gene_stats(hit.iloc[0]["probe_id"], by=("class",))
    assert stats["n"].sum() == 72
