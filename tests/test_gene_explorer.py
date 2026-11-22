"""Kiểm tra src/gene_explorer.py (TV4, công việc 2.2) trên dữ liệu tổng hợp.

Dữ liệu giả cùng cỡ Golub: 47 ALL + 25 AML; phân bố nguồn như handover 1.1:
  train: DFCI 27 ALL (24 B-ALL + 3 T-ALL) | CALGB 11 AML
  test : DFCI 17 ALL | St-Jude 3 ALL + 5 AML | CCG 9 AML
Mẫu 2 gắn cờ qc_outlier; mẫu 72 (CCG) có CD33 cao hơn mọi mẫu tham chiếu (để thử batch_anomalies).
Call P/A: CD33 bật ở AML, tắt ở ALL; X95735_at tắt ở cả hai lớp; U12345_at bật ở cả hai lớp.
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
        ("train", "ALL", "B-ALL", "BM", "DFCI", 24),
        ("train", "ALL", "T-ALL", "BM", "DFCI", 3),
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
    df = pd.DataFrame(rows).set_index("sample_id")
    df["qc_outlier"] = df.index == 2
    return df


def _X(samples: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(2.0, 0.1, size=(len(samples), len(PROBES))),
                     index=samples.index, columns=list(PROBES))
    aml = samples["class"].eq("AML").to_numpy()
    X.loc[aml, "M23197_at"] += 1.0            # CD33 cao ở AML trong mọi nguồn
    X.loc[72, "M23197_at"] = 5.0              # mẫu CCG bất thường, ngoài mọi khoảng tham chiếu
    return X


def _calls(samples: pd.DataFrame) -> pd.DataFrame:
    aml = samples["class"].eq("AML")
    return pd.DataFrame({
        "M23197_at": aml.map({True: "P", False: "A"}),
        "X95735_at": "A",
        "U12345_at": "P",
        "AFFX-BioB-5_at": "A",
    }, index=samples.index)


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

    calls_path = tmp_path / "calls.parquet"
    _calls(samples).to_parquet(calls_path)
    monkeypatch.setattr(ge, "CALLS_PATH", calls_path)

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


# ---------------------------------------------------------------- độ tin cậy, độ vững

def test_gene_values_carries_qc_flag(env):
    vals = ge.gene_values("M23197_at").set_index("sample_id")
    assert bool(vals.loc[2, "qc_outlier"]) is True
    assert vals["qc_outlier"].sum() == 1


def test_reliability_on_off_pattern(env):
    rel = ge.gene_reliability("M23197_at").set_index("class")
    assert rel.loc["AML", "pct_present"] == 1.0 and rel.loc["ALL", "pct_present"] == 0.0
    assert rel.attrs["pattern"] == "on_off"
    assert "AML" in rel.attrs["note"]


def test_reliability_weak_and_level(env):
    assert ge.gene_reliability("X95735_at").attrs["pattern"] == "weak"
    assert ge.gene_reliability("U12345_at").attrs["pattern"] == "level"


def test_reliability_without_calls_file(env, monkeypatch):
    monkeypatch.setattr(ge, "CALLS_PATH", env["tmp"] / "khong_co.parquet")
    rel = ge.gene_reliability("M23197_at")
    assert rel["pct_present"].isna().all()
    assert rel.attrs["pattern"] == "unknown"
    assert rel["pct_floor"].notna().all()


def test_robustness_rows_and_direction(env):
    rob = ge.robustness("M23197_at").set_index("check")
    assert len(rob) == 7
    assert rob.loc["Bỏ mẫu gắn cờ chất lượng", "n_ALL"] == 46       # bỏ mẫu 2
    assert rob.loc["T-ALL so với AML", "n_ALL"] == 3
    assert rob.loc["Trong St-Jude (cùng phòng lab)", "n_ALL"] == 3
    assert rob["same_direction"].fillna(True).all()                  # CD33 luôn cao ở AML
    assert rob.loc["Toàn bộ 72 mẫu", "fold_change"] < 1              # ALL thấp hơn AML


def test_robustness_without_qc_column(env, monkeypatch):
    X, samples = env["X"], env["samples"].drop(columns="qc_outlier")
    monkeypatch.setattr(ge, "_load_expression", lambda: (X, samples))
    assert len(ge.robustness("M23197_at")) == 6


# ---------------------------------------------------------------- lô mới

def test_marker_consistency_stjude(env):
    c = ge.marker_consistency(source="St-Jude")
    assert set(c["probe_id"]) == {"M23197_at", "X95735_at", "U12345_at"}   # chỉ gen dấu ấn có trong X
    assert c.attrs["n_ALL"] == 3 and c.attrs["n_AML"] == 5
    row = c.set_index("probe_id").loc["M23197_at"]
    assert row["status"] == "Giữ chiều"
    assert set(c["status"]) <= set(ge.CONSISTENCY_STATUS)
    assert c.attrs["n_kept"] + c.attrs["n_flipped"] + c.attrs["n_small"] == len(c)


def test_marker_consistency_single_class_batch(env):
    c = ge.marker_consistency(source="CCG")
    assert c.attrs["comparable"] is False
    assert c["same_direction"].isna().all()
    assert (c["status"] == "Không so được").all()
    assert "batch_anomalies" in c.attrs["note"]
    assert np.isnan(c.attrs["p_perm"]) and c.attrs["n_perm"] == 0


def test_marker_consistency_permutation_and_sd_column(env):
    X, samples = env["X"], env["samples"]
    c = ge.marker_consistency(source="St-Jude")
    assert c.attrs["n_perm"] == 56                                     # C(8, 3): duyệt đủ
    assert 1 / 56 <= c.attrs["p_perm"] <= 1
    assert "p = " in c.attrs["note"]
    # batch_diff_sd = batch_diff / độ lệch chuẩn gộp trong lớp trên tập huấn luyện
    tr = samples["split"].eq("train")
    a = X.loc[tr & samples["class"].eq("ALL"), "M23197_at"]
    m = X.loc[tr & samples["class"].eq("AML"), "M23197_at"]
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(m) - 1) * m.var()) / (len(a) + len(m) - 2))
    row = c.set_index("probe_id").loc["M23197_at"]
    assert row["batch_diff_sd"] == pytest.approx(row["batch_diff"] / pooled)


def test_label_masks_exact_and_random():
    is_all = np.r_[np.ones(3, bool), np.zeros(5, bool)]
    exact = ge._label_masks(is_all)
    assert exact.shape == (56, 8) and (exact.sum(axis=1) == 3).all()
    assert len({tuple(r) for r in exact}) == 56                        # không trùng
    assert any((r == is_all).all() for r in exact)                     # có cách xếp thật
    big = np.r_[np.ones(20, bool), np.zeros(20, bool)]                 # C(40, 20) quá lớn: xáo ngẫu nhiên
    rand = ge._label_masks(big)
    assert rand.shape == (ge.N_PERM_RANDOM + 1, 40)
    assert (rand[0] == big).all() and (rand.sum(axis=1) == 20).all()


def test_perm_same_direction_signal_and_opposite():
    rng = np.random.default_rng(1)
    is_all = np.r_[np.ones(3, bool), np.zeros(5, bool)]
    Xb = rng.normal(0, 0.1, size=(8, 20))
    Xb[is_all] += 2.0                                                  # mọi gen cao ở ALL
    p_same, n = ge._perm_same_direction(Xb, is_all, np.ones(20))
    p_opp, _ = ge._perm_same_direction(Xb, is_all, -np.ones(20))
    assert n == 56
    assert 1 / 56 <= p_same < 0.5                                      # tín hiệu thật: ít cách xếp bằng được
    assert p_opp == 1.0                                                # ngược chiều: 0 gen cùng dấu, mọi cách xếp ≥


def test_marker_consistency_only_test_excludes_train(env):
    with pytest.raises(ValueError):
        ge.marker_consistency(source="CALGB")                  # CALGB chỉ có mẫu huấn luyện
    c = ge.marker_consistency(source="CALGB", only_test=False)
    assert c.attrs["n_AML"] == 11


def test_marker_consistency_argument_checks(env):
    with pytest.raises(ValueError):
        ge.marker_consistency()
    with pytest.raises(ValueError):
        ge.marker_consistency(source="St-Jude", sample_ids=[40])
    with pytest.raises(ValueError):
        ge.marker_consistency(source="KHONGCO")


def test_batch_anomalies_flags_outlier_sample(env):
    a = ge.batch_anomalies(source="CCG")
    top = a.iloc[0]
    assert top["sample_id"] == 72 and top["probe_id"] == "M23197_at"
    assert top["side"] == "cao hơn" and top["severity"] > 0
    assert "mẫu 72" in top["reason"]
    assert a["severity"].is_monotonic_decreasing


def test_batch_anomalies_does_not_use_batch_labels(env, monkeypatch):
    X, samples = env["X"], env["samples"].copy()
    before = ge.batch_anomalies(source="CCG")
    samples.loc[samples["source"].eq("CCG"), "class"] = "ALL"     # đổi nhãn của lô: kết quả không đổi
    monkeypatch.setattr(ge, "_load_expression", lambda: (X, samples))
    after = ge.batch_anomalies(source="CCG")
    pd.testing.assert_frame_equal(before, after)


def test_ranking_table_all_rows(env):
    assert len(ge.ranking_table(top_n=None)) == 5


# ---------------------------------------------------------------- dữ liệu thật

@pytest.mark.skipif(not (CURATED_DIR / "X_log10.parquet").exists(),
                    reason="chưa có tầng curated")
def test_real_data_cd33_smoke():
    hit = ge.find_gene("CD33")
    assert not hit.empty
    stats = ge.gene_stats(hit.iloc[0]["probe_id"], by=("class",))
    assert stats["n"].sum() == 72


@pytest.mark.skipif(not (CURATED_DIR / "X_log10.parquet").exists(), reason="chưa có tầng curated")
def test_real_data_stjude_consistency():
    ge.clear_cache()
    c = ge.marker_consistency(source="St-Jude")
    assert c.attrs["n_same"] == 46 and len(c) == 50          # khớp kết quả 1.6
    assert c.attrs["n_flipped"] == 0                          # không gen nào đảo chiều rõ
    assert c.attrs["n_perm"] == 56 and c.attrs["p_perm"] == pytest.approx(2 / 56)   # 2/56 cách xếp đạt ≥ 46


@pytest.mark.skipif(not (CURATED_DIR / "X_log10.parquet").exists(), reason="chưa có tầng curated")
def test_real_data_warehouse_matches_parquet(tmp_path):
    duckdb = pytest.importorskip("duckdb")  # noqa: F841
    from src.load import load_golub
    from src.warehouse import WAREHOUSE_PATH, build_warehouse
    assert ge.WAREHOUSE_PATH == WAREHOUSE_PATH
    path = tmp_path / "golub.duckdb"
    build_warehouse(path)
    Xd, sd = ge._read_warehouse(path)
    Xp, sp = load_golub("curated")
    pd.testing.assert_frame_equal(Xd[Xp.columns], Xp, check_dtype=False)
    for col in sp.columns:
        assert sd[col].astype(str).tolist() == sp[col].astype(str).tolist(), col
