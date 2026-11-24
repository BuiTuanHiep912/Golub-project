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

from src.evaluate import (  # noqa: E402
    _experiment_config, _nested_param_grid, confusion_counts, evaluate, mcnemar_test, wilson_ci,
)

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


def _test_labels() -> pd.Series:
    """Nhãn thật của 34 mẫu test (sample_id 39..72) như cột y_true của evaluate()["predictions"]."""
    return pd.Series(["ALL"] * 20 + ["AML"] * 14, index=pd.Index(range(39, 73), name="sample_id"))


def test_mcnemar_aligns_predictions_by_sample_id():
    # ROOT CAUSE:
    #
    # mcnemar_test() đưa cả ba đầu vào qua np.asarray, tức bỏ chỉ mục sample_id rồi so theo VỊ
    # TRÍ. Dự đoán của hai mô hình thường lấy từ hai lần chạy hoặc hai file (vd. predictions_test.csv
    # của TV3) nên thứ tự dòng có thể khác: hai mô hình cùng đúng 34/34 nhưng một bên xếp ngược
    # thì hàm báo B chỉ đúng 6/34 và p = 0.0.
    #
    # Sửa bằng cách xếp pred_a/pred_b theo chỉ mục của y_true khi chúng là Series.
    y = _test_labels()
    b = y.sort_index(ascending=False)          # cùng dự đoán, chỉ khác thứ tự dòng

    res = mcnemar_test(y, y.copy(), b)

    assert res["b_correct"] == 34
    assert res["p_value"] == 1.0


def test_mcnemar_rejects_predictions_from_another_test_set():
    y = _test_labels()
    with pytest.raises(ValueError, match="sample_id"):
        mcnemar_test(y, y, y.drop(72))
    with pytest.raises(ValueError, match="cùng số mẫu"):
        mcnemar_test(list(y), list(y), list(y)[:-1])


def test_confusion_counts_keeps_uncertain_predictions():
    # ROOT CAUSE:
    #
    # plot_roc_cm() gọi confusion_matrix(t, p, labels=[0, 1]), nên dự đoán "uncertain" (mã -1, có ở
    # weighted voting của TV1) bị bỏ mà không báo: 2 mẫu "uncertain" thì ma trận F11 chỉ cộng ra
    # 32/34.
    #
    # Sửa bằng confusion_counts(): thêm cột "uncertain" khi có, để tổng luôn bằng số mẫu.
    y = list(_test_labels())
    pred = list(y)
    pred[0], pred[25] = "uncertain", "uncertain"   # một mẫu ALL, một mẫu AML

    cm = confusion_counts(y, pred)

    assert int(cm.to_numpy().sum()) == 34
    assert list(cm.columns) == ["ALL", "AML", "uncertain"]
    assert cm.loc["ALL", "uncertain"] == 1 and cm.loc["AML", "uncertain"] == 1
    assert cm.loc["ALL", "ALL"] == 19 and cm.loc["AML", "AML"] == 13


def test_confusion_counts_is_2x2_when_model_is_never_uncertain():
    y = list(_test_labels())
    pred = list(y)
    pred[0] = "AML"                                 # một mẫu ALL bị đoán AML

    cm = confusion_counts(y, pred)

    assert list(cm.columns) == ["ALL", "AML"]
    assert cm.to_numpy().tolist() == [[19, 1], [0, 14]]


def test_nested_param_grid_tunes_k_from_experiment_config():
    # ROOT CAUSE:
    #
    # main() (và notebook 08) gọi evaluate(..., "nested_cv") không kèm param_grid, nên không có
    # vòng trong: dòng "nested_cv" trong summary_evaluation.csv thực ra là repeated CV với k = 50,
    # C = 1 cố định, trái mục 3.5.1 (vòng trong chọn k và siêu tham số).
    #
    # Sửa bằng _nested_param_grid(): k lấy từ nhân tố B của experiment_config.yaml, C theo mục 3.5.2.
    grid = _nested_param_grid()

    assert grid["select__k"] == [int(k) for k in _experiment_config()["k"]]
    assert grid["clf__C"] == [0.01, 0.1, 1, 10]


def test_nested_cv_with_param_grid_runs_an_inner_loop():
    X, y = _data(signal=3.0)

    res = evaluate(_pipe(), X, y, "nested_cv", seeds=[0], param_grid={"select__k": [5, 10], "clf__C": [0.1, 1]})

    assert res["config"]["inner_folds"] == 5
    assert res["config"]["param_grid"] == {"select__k": [5, 10], "clf__C": [0.1, 1]}
    assert len(res["per_fold"]) == 5
    assert res["bal_acc_mean"] > 0.9


def _sb_runs(n_perm: int) -> pd.DataFrame:
    """Bảng kết quả selection bias giả: 3 seed nhãn thật, n_perm lần hoán vị."""
    rows = [{"labels": "real", "cv": cv, "rep": r, "seed": r, "bal_acc": 0.95, "auc": 0.99}
            for r in range(3) for cv in ("correct", "wrong")]
    rows += [{"labels": "permuted", "cv": cv, "rep": r, "seed": r, "bal_acc": acc, "auc": 0.5}
             for r in range(n_perm) for cv, acc in (("correct", 0.5), ("wrong", 0.8))]
    return pd.DataFrame(rows)


def test_f10_counts_permutations_from_runs_instead_of_hard_coding():
    # ROOT CAUSE:
    #
    # Tiêu đề F10 ghi cứng "100 lần hoán vị nhãn", nên chạy thử `--n-permutations 7` vẫn in 100.
    #
    # Sửa bằng cách đếm số lần hoán vị và lấy trung bình "hoán vị × CV sai" từ chính `runs`.
    from src.evaluate import _selection_bias_figure

    fig = _selection_bias_figure(_sb_runs(7), mode="report")
    texts = [fig._suptitle.get_text()] + [ax.get_title(loc="left") for ax in fig.axes]

    assert any("7 lần hoán vị nhãn, 3 seed nhãn thật" in t for t in texts)
    assert "nhãn hoán vị vẫn đạt 0.80" in fig._suptitle.get_text()


def test_f10_follows_style_guide_sizes(tmp_path):
    # ROOT CAUSE:
    #
    # F10/F11 bản 1 dùng cỡ chữ 9 pt và 220 dpi riêng của evaluate.py, dưới mức tối thiểu của
    # docs/style_guide.md (báo cáo: chữ >= 10 pt, 300 dpi; slide: chữ >= 16 pt).
    #
    # Sửa bằng cách vẽ qua src/viz.py (styled, figsize, save_figure).
    from matplotlib.image import imread

    from src.evaluate import _selection_bias_figure, plot_selection_bias
    from src.viz import FIG_STYLES

    for mode in ("report", "slide"):
        fig = _selection_bias_figure(_sb_runs(5), mode=mode)
        sizes = [t.get_fontsize() for ax in fig.axes for t in ax.texts + ax.get_xticklabels() + ax.get_yticklabels()]
        assert min(sizes) >= FIG_STYLES[mode]["fontsize"], mode

    paths = plot_selection_bias(_sb_runs(5), base_dir=tmp_path)
    assert [p.name for p in paths] == ["F10_selection_bias_report.png", "F10_selection_bias_slide.png"]
    width_px = imread(paths[0]).shape[1]
    assert width_px >= 0.9 * FIG_STYLES["report"]["width"] * FIG_STYLES["report"]["dpi"]   # ~300 dpi ở 6,5 in
