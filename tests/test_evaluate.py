"""Kiểm tra src/evaluate.py (TV2 - công việc 2.4) trên dữ liệu tổng hợp, không cần dữ liệu thật."""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import evaluate, mcnemar_test, wilson_ci  # noqa: E402

PLAN_KEYS = ["bal_acc_mean", "bal_acc_sd", "auc_mean", "per_fold", "config", "predictions"]


def _labels() -> pd.Series:
    """47 ALL / 25 AML, train 1..38 (27/11) và test 39..72 (20/14) như Bảng 2.4."""
    y = ["ALL"] * 27 + ["AML"] * 11 + ["ALL"] * 20 + ["AML"] * 14
    return pd.Series(y, index=pd.Index(range(1, 73), name="sample_id"), name="class")


def _split() -> pd.Series:
    return pd.Series(["train"] * 38 + ["test"] * 34, index=pd.Index(range(1, 73), name="sample_id"))


def _data(signal: float, n_genes: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """10 gen đầu lệch `signal` độ lệch chuẩn giữa AML và ALL; còn lại là nhiễu thuần."""
    rng = np.random.default_rng(seed)
    y = _labels()
    X = rng.normal(size=(72, n_genes))
    X[:, :10] += signal * (y.to_numpy() == "AML")[:, None]
    return pd.DataFrame(X, index=y.index, columns=[f"g{i}" for i in range(n_genes)]), y


def _pipe(k: int = 10) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("select", SelectKBest(f_classif, k=k)),
                     ("clf", LogisticRegression(max_iter=5000))])


def test_nested_cv_returns_plan_keys_and_defined_auc():
    # ROOT CAUSE:
    #
    # Bản làm việc cũ in "nested_cv ... auc=nan" dù mỗi fold stratified luôn có đủ hai lớp.
    # AUC chỉ được bỏ trống khi fold thiếu một lớp; ở đây mọi fold phải có AUC.
    X, y = _data(signal=3.0)
    res = evaluate(_pipe(), X, y, "nested_cv", seeds=range(2))

    assert list(res) == PLAN_KEYS
    assert len(res["per_fold"]) == 2 * 5               # 2 seed × outer_folds = 5 (experiment_config.yaml)
    assert res["per_fold"]["auc"].notna().all()
    assert res["bal_acc_mean"] > 0.9
    assert res["config"]["seeds"] == [0, 1]
    assert len(res["predictions"]) == 2 * 72            # mỗi mẫu được dự đoán đúng một lần mỗi seed


def test_loocv_pools_predictions_over_38_train_samples():
    # ROOT CAUSE:
    #
    # Bản làm việc cũ báo "loocv bal_acc=0.395±0.489": nó lấy trung bình balanced accuracy
    # của từng fold, mà mỗi fold LOOCV chỉ có 1 mẫu nên chỉ số mỗi fold là 0 hoặc 1 (hoặc
    # NaN), con số trung bình không có nghĩa.
    #
    # Sửa bằng cách gộp dự đoán của cả 38 fold rồi tính chỉ số một lần (một dòng "pooled").
    X, y = _data(signal=3.0)
    res = evaluate(_pipe(), X, y, "loocv", split=_split())

    assert len(res["per_fold"]) == 1
    row = res["per_fold"].iloc[0]
    assert row["fold"] == "pooled" and row["n_test"] == 38
    assert np.isnan(res["bal_acc_sd"])
    assert res["bal_acc_mean"] == 1.0 and res["auc_mean"] == 1.0
    assert list(res["predictions"].index) == list(range(1, 39))


def test_original_split_trains_on_train_and_reports_wilson_ci():
    X, y = _data(signal=3.0)
    res = evaluate(_pipe(), X, y, "original_split", split=_split())

    assert list(res["predictions"].index) == list(range(39, 73))
    row = res["per_fold"].iloc[0]
    assert row["n_test"] == 34
    assert row["acc_ci_low"] <= row["accuracy"] <= row["acc_ci_high"]
    assert row["sensitivity"] == pytest.approx((res["predictions"].query("y_true == 'AML'")["y_pred"] == "AML").mean())


def test_wrong_cv_inflates_accuracy_on_pure_noise():
    """Cơ chế H3: không có tín hiệu thật, CV đúng quanh 0.5 còn CV sai cao bất thường."""
    X, y = _data(signal=0.0, n_genes=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correct = evaluate(_pipe(k=20), X, y, "nested_cv", seeds=range(3))
        wrong = evaluate(_pipe(k=20), X, y, "wrong_cv", seeds=range(3))

    assert correct["bal_acc_mean"] < 0.65
    assert wrong["bal_acc_mean"] > 0.75


def test_wilson_ci_for_perfect_score_on_34_samples():
    lo, hi = wilson_ci(34, 34)
    assert lo == pytest.approx(34 / (34 + 1.959964 ** 2), abs=1e-4)   # công thức Wilson khi p̂ = 1
    assert hi == pytest.approx(1.0)


def test_mcnemar_exact_counts_only_discordant_samples():
    y = ["ALL"] * 6 + ["AML"] * 4
    a = list(y)                                  # mô hình A đúng cả 10
    b = ["ALL"] * 10                             # mô hình B sai cả 4 mẫu AML
    b[0] = "AML"                                 # và sai thêm 1 mẫu ALL
    res = mcnemar_test(y, a, b)
    assert (res["a_only_correct"], res["b_only_correct"]) == (5, 0)
    assert res["p_value"] == pytest.approx(2 * 0.5 ** 5)   # nhị thức chính xác, hai phía


def test_uncertain_prediction_counts_as_error():
    res = mcnemar_test(["ALL", "AML"], ["ALL", "AML"], ["ALL", "uncertain"])
    assert res["b_correct"] == 1


@pytest.mark.parametrize("scheme", ["cv", "LOOCV", ""])
def test_unknown_scheme_raises(scheme):
    X, y = _data(signal=1.0)
    with pytest.raises(ValueError, match="scheme phải là một trong"):
        evaluate(_pipe(), X, y, scheme)


def test_wrong_cv_requires_a_selection_step():
    X, y = _data(signal=1.0)
    no_select = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
    with pytest.raises(ValueError, match="bước chọn gen"):
        evaluate(no_select, X, y, "wrong_cv", seeds=[0])
