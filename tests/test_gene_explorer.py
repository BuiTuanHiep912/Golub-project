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
    assert bool(top["notable"]) is True                           # CD33 = 5,0: vượt xa ngưỡng của mẫu train
    assert a.attrs["n_notable_samples"] >= 1 and a.attrs["baseline_n"] == 38
    assert "Mức nền" in a.attrs["note"] and "72" in a.attrs["note"]


def test_reference_baseline_leave_one_out():
    ref = pd.DataFrame({"g": [0.0, 1.0, 2.0, 3.0, 10.0]}, index=[1, 2, 3, 4, 5])
    base = ge._reference_baseline(ref)
    assert base["n"] == 5 and base["n_flagged"] == 2              # chỉ mẫu đầu (0) và mẫu cuối (10) ngoài khoảng
    gap_last = 7.0 / ref["g"].iloc[:4].std()                      # 10 so với max 3, chia SD của 4 mẫu còn lại
    assert base["cutoff"] == pytest.approx(np.quantile([1.0 / ref["g"].iloc[1:].std(), 0, 0, 0, gap_last],
                                                       ge.BASELINE_QUANTILE))


def test_out_of_range_severity_zero_inside():
    ref = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
    vals = pd.DataFrame({"a": [2.5, 4.0], "b": [5.0, 6.0]})
    sev = ge._out_of_range_severity(vals, ref)
    assert sev.loc[0, "a"] == 0 and sev.loc[1, "a"] == pytest.approx(1.0)   # (4 − 3) / SD 1
    assert (sev["b"] == 0).all()                                  # SD = 0: không chia được, coi là 0


def test_batch_anomalies_does_not_use_batch_labels(env, monkeypatch):
    X, samples = env["X"], env["samples"].copy()
    before = ge.batch_anomalies(source="CCG")
    samples.loc[samples["source"].eq("CCG"), "class"] = "ALL"     # đổi nhãn của lô: kết quả không đổi
    monkeypatch.setattr(ge, "_load_expression", lambda: (X, samples))
    after = ge.batch_anomalies(source="CCG")
    pd.testing.assert_frame_equal(before, after)


# ---------------------------------------------------------------- yếu tố khác ngoài ALL/AML

@pytest.fixture
def factors(monkeypatch):
    """Dữ liệu giả có gen kiểu B-ALL, gen giới tính, gen bạch cầu hạt và gen khác theo phòng lab."""
    samples = _samples()
    rng = np.random.default_rng(3)
    n = len(samples)
    sub, src, cls = samples["subtype"], samples["source"], samples["class"]
    sex = np.where(samples.index % 2 == 0, "M", "F").astype(object)
    sex[samples.index.get_indexer([5, 6, 7])] = None             # 3 mẫu không ghi giới tính
    samples["sex"] = sex
    true_male = samples["sex"].eq("M").to_numpy().copy()
    true_male[samples.index.get_loc(9)] = True                   # mẫu 9 ghi nữ nhưng thực ra nam
    true_male[samples.index.get_indexer([5, 6, 7])] = [True, False, True]
    noise = lambda: rng.normal(0, 0.03, n)                       # noqa: E731
    cols = {
        "B_ONLY_at": np.where(sub.eq("B-ALL"), 3.0, 2.0) + noise(),   # như IGB
        "BOTH_at": np.where(cls.eq("ALL"), 3.0, 2.0) + noise(),
        "AMLHI_at": np.where(cls.eq("AML"), 3.0, 2.0) + noise(),
        ge.SEX_Y_PROBE: np.where(true_male, 3.3, 2.0) + noise(),
        ge.SEX_XIST_PROBE: np.where(true_male, 2.2, 2.7) + noise(),
        "LAB_at": np.where(cls.eq("AML") & src.eq("St-Jude"), 2.8, 2.0) + noise(),
    }
    gran = np.full(n, 2.0)
    gran[samples.index.get_loc(70)] = 3.8                        # mẫu CCG (kiểm tra) lẫn bạch cầu hạt
    gran[samples.index.get_loc(46)] = 3.6                        # mẫu St-Jude B-ALL (kiểm tra)
    for p in ge.GRANULOCYTE_PROBES:
        cols[p] = gran + noise()
    cols["BOTH_at"] = cols["BOTH_at"] + np.where(cls.eq("ALL"), (gran - 2.0), 0)   # gen dấu ấn đi theo điểm
    for i in range(30):
        cols[f"N{i}_at"] = 2.5 + rng.normal(0, 0.1, n)
    X = pd.DataFrame(cols, index=samples.index)
    monkeypatch.setattr(ge, "_load_expression", lambda: (X, samples))
    probes = ["B_ONLY_at", "BOTH_at", "AMLHI_at", "LAB_at"] + [f"N{i}_at" for i in range(30)]
    ranking = pd.DataFrame({"probe_id": probes, "rank": range(1, len(probes) + 1),
                            "snr": [1.2, 1.1, -1.3, 0.1] + [0.01] * 30,
                            "gene_symbol": ["IGBX", "BOTH", "AMLG", "LABG"] + [""] * 30})
    monkeypatch.setattr(ge, "_load_ranking", lambda: ranking.copy())
    return {"X": X, "samples": samples}


def test_subtype_check_flags_b_only_gene(factors):
    s = ge.subtype_check(n_per_side=2).set_index("probe_id")
    assert s.loc["B_ONLY_at", "pattern"] == "Chỉ B-ALL"
    assert s.loc["B_ONLY_at", "pct_floor_T"] > 0.5 or s.loc["B_ONLY_at", "mean_T"] < 2.1
    assert s.loc["BOTH_at", "pattern"] == "Cả B-ALL và T-ALL"
    assert s.loc["AMLHI_at", "pattern"] == "Cả B-ALL và T-ALL" and s.loc["AMLHI_at", "direction"] == "AML"
    assert "IGBX" in s.attrs["note"] and s.attrs["counts"]["Chỉ B-ALL"] == 1


def test_subtype_check_single_probe_and_unknown(factors):
    assert ge.subtype_check(probes="B_ONLY_at").iloc[0]["pattern"] == "Chỉ B-ALL"
    with pytest.raises(KeyError):
        ge.subtype_check(probes=["KHONGCO_at"])


def test_subtype_contrast_finds_b_cell_gene(factors):
    c = ge.subtype_contrast(n_perm=50)
    assert c.iloc[0]["probe_id"] == "B_ONLY_at" and c.iloc[0]["snr"] < 0      # cao ở B-ALL, thấp ở T-ALL
    assert (c.attrs["n_T"], c.attrs["n_B"]) == (3, 44)
    assert [r["threshold"] for r in c.attrs["thresholds"]] == [0.5, 1.0, 1.5]
    assert all(r["observed"] >= 1 for r in c.attrs["thresholds"])     # ít nhất gen kiểu B-ALL vượt mọi ngưỡng
    assert c["snr"].abs().is_monotonic_decreasing                      # so với xáo nhãn: kiểm tra trên dữ liệu thật


def test_marker_stability_frequencies(factors):
    st = ge.marker_stability(n_boot=20, n_per_side=2)
    assert st["freq"].between(0, 1).all() and st["freq"].is_monotonic_decreasing
    assert set(st["probe_id"]) == set(ge._marker_genes(2)["probe_id"])
    assert 0 < st.attrs["mean_jaccard"] <= 1 and st.attrs["min_jaccard"] <= st.attrs["mean_jaccard"]
    assert st.set_index("probe_id").loc["B_ONLY_at", "freq"] >= 0.8      # gen tách lớp rõ thì luôn được chọn lại


def test_marker_correlation_groups_and_errors(factors):
    c = ge.marker_correlation("train", n_per_side=2)
    assert {"mean_abs_r", "pc1_share", "n_effective", "modules"} <= set(c.attrs)
    assert 1 <= c.attrs["n_effective"] <= 3 and 0 < c.attrs["pc1_share"] <= 1
    assert (c["module_size"] >= 3).all()
    with pytest.raises(ValueError):
        ge.marker_correlation("B-ALL")


def test_sex_check_mismatch_missing_and_no_metadata_change(factors):
    before = factors["samples"]["sex"].copy()
    sx = ge.sex_check().set_index("sample_id")
    assert sx.loc[9, "status"] == "Lệch metadata" and sx.loc[9, "sex_inferred"] == "M"
    assert (sx.loc[[5, 6, 7], "status"] == "Metadata thiếu").all()
    assert sx.loc[[5, 6, 7], "sex_inferred"].tolist() == ["M", "F", "M"]
    assert sx.attrs["n_mismatch"] == 1 and sx.attrs["n_missing"] == 3
    assert sx.iloc[0].name == 9                                   # mẫu lệch đứng đầu bảng
    pd.testing.assert_series_equal(factors["samples"]["sex"], before)   # không sửa metadata


def test_lab_effect_finds_lab_gene_within_aml(factors):
    le = ge.lab_effect("AML", n_per_side=2)
    assert le.attrs["labs"] == ["CALGB", "St-Jude", "CCG"]
    assert le.iloc[0]["probe_id"] == "LAB_at" and le.iloc[0]["q_value"] < 0.05
    assert le.attrs["n_sig"] >= 1 and "LAB_at" not in le.attrs["markers_affected"]
    with pytest.raises(ValueError):
        ge.lab_effect("T-ALL")                                    # chỉ một lab (DFCI)


def test_bh_matches_manual():
    q = ge._bh(np.array([0.01, 0.04, 0.03, np.nan]))
    assert np.isnan(q[3])
    assert q[:3] == pytest.approx([0.03, 0.04, 0.04])


def test_purity_check_flags_contaminated_test_samples(factors):
    p = ge.purity_check(n_per_side=2)
    assert set(p.columns) >= {"sample_id", "value", "class", "source", "split", "above_train"}
    assert p.attrs["flagged"] == [70, 46]                         # sắp điểm giảm dần
    assert (p.loc[p["above_train"], "split"] == "test").all()
    assert "BOTH_at" in p.attrs["tracking_markers"]


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
def test_real_data_other_factors():
    ge.clear_cache()
    st = ge.subtype_check()
    only_b = set(st.loc[st["pattern"].eq("Chỉ B-ALL"), "gene_symbol"])
    assert only_b == {"IGB", "IRF2", "TCL1", "CD24"}    # CD19 không qua bộ lọc trên train
    sx = ge.sex_check()
    assert (sx.attrs["n_match"], sx.attrs["n_mismatch"], sx.attrs["n_missing"]) == (47, 2, 23)
    assert sorted(sx.loc[sx["status"].eq("Lệch metadata"), "sample_id"]) == [18, 60]
    aml = ge.lab_effect("AML")
    assert aml.attrs["labs"] == ["CALGB", "St-Jude", "CCG"] and len(aml.attrs["markers_affected"]) == 2
    stab = ge.marker_stability()
    assert stab.attrs["n_stable"] == 5 and stab.attrs["mean_jaccard"] == pytest.approx(0.36, abs=0.01)
    assert ge.marker_correlation("train").attrs["n_effective"] == pytest.approx(3.1, abs=0.1)
    assert ge.marker_correlation("ALL").attrs["n_effective"] > 9                 # trong cùng lớp: nhiều tín hiệu hơn
    con = ge.subtype_contrast()
    assert [r["observed"] for r in con.attrs["thresholds"]] == [415, 72, 23]
    assert con.attrs["thresholds"][1]["perm_max"] <= 2                # |SNR| > 1: tối đa 2 gen khi xáo nhãn
    assert con.iloc[0]["probe_id"] == "X03934_at"                     # CD3 delta, cao ở T-ALL
    pur = ge.purity_check()
    assert pur.attrs["flagged"] == [60, 43, 59, 56]
    assert pur.attrs["tracking_markers"] == ["M27783_s_at"]     # ELA2, gen dấu ấn AML hạng 8


@pytest.mark.skipif(not (CURATED_DIR / "X_log10.parquet").exists(), reason="chưa có tầng curated")
def test_real_data_batch_anomalies_baseline():
    ge.clear_cache()
    ccg = ge.batch_anomalies(source="CCG")
    assert ccg["sample_id"].nunique() == 3                    # 3/5 mẫu có gen ngoài khoảng...
    assert (ccg.attrs["baseline_flagged"], ccg.attrs["baseline_n"]) == (18, 38)   # ...mẫu train cũng 47%
    assert ccg.attrs["n_notable_samples"] == 0                # nên CCG không có gì đáng chú ý
    dfci = ge.batch_anomalies(source="DFCI")
    assert dfci.loc[dfci["notable"], "sample_id"].unique().tolist() == [42]   # LTC4S ở mức sàn


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
