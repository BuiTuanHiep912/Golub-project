"""Khung đánh giá dùng chung: 4 sơ đồ CV, chỉ số, khoảng tin cậy, kiểm định, selection bias.

Công việc: 2.4 — Phụ trách: TV2. TV1 và TV3 dùng hàm evaluate() từ tuần 5.

    from src.evaluate import evaluate, mcnemar_test, wilson_ci, selection_bias_experiment
    res = evaluate(pipe, X, y, scheme="nested_cv", seeds=range(10))
    res.keys()  # bal_acc_mean, bal_acc_sd, auc_mean, per_fold, config, predictions

Bốn sơ đồ (mục 3.5.1 của kế hoạch):
  original_split : huấn luyện trên 38 mẫu train, kiểm tra trên 34 mẫu test (giống bài báo)
  loocv          : LOOCV trên 38 mẫu train; mọi bước của pipeline (kể cả chọn gen) fit lại ở
                   mỗi fold (note 22 của bài báo); chỉ số tính trên dự đoán GỘP của 38 fold
  nested_cv      : stratified K-fold trên 72 mẫu, mỗi seed là một lần lặp; mọi bước nằm trong
                   pipeline nên đây là "CV đúng"; có param_grid thì thêm vòng trong GridSearchCV
  wrong_cv       : bước chọn gen (và mọi bước trước nó) fit trên TOÀN BỘ 72 mẫu rồi mới CV
                   phần còn lại — "CV sai", chỉ để minh họa selection bias (H3)

Quy ước: AML là lớp dương (config.POSITIVE_CLASS). Số fold và seed đọc từ
config/experiment_config.yaml.

Chạy `python -m src.evaluate` (hay `make evaluate`) để đánh giá pipeline tạm (log10 → chuẩn hóa →
50 gen → logistic) bằng cả 4 sơ đồ, kiểm định McNemar với baseline lớp đa số, và chạy thí
nghiệm selection bias; ghi results/metrics/summary_evaluation.csv,
summary_selection_bias.csv, selection_bias_runs.csv + hình F10 (selection bias) và
F11 (ROC + ma trận nhầm lẫn) vào results/figures/report|slides/ (Bảng 2.9 của thuyết minh TV2).
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, SelectorMixin, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from src.config import (
    CLEANSED_DIR, CONFIG_DIR, METRICS_DIR, N_PERMS_EVAL, NEGATIVE_CLASS, POSITIVE_CLASS,
    RANDOM_SEED, REPORT_FIG_DIR, SLIDE_FIG_DIR,
)
from src.load import load_golub

SCHEMES = ("original_split", "loocv", "nested_cv", "wrong_cv")

# Hai "bản" của mỗi hình theo Bảng 2.9: report = nhỏ, chữ 9pt (in trong báo cáo);
# slide = chữ lớn (thông số do TV6 chốt trong style_guide, sẽ thay bằng src/viz.py khi có).
_FIG_STYLES = {
    "report": dict(figsize=(7.2, 4.0), fontsize=9, dpi=220),
    "slide": dict(figsize=(9.0, 5.2), fontsize=17, dpi=150),
}


# ---------------------------------------------------------------------------
# Chỉ số và kiểm định
# ---------------------------------------------------------------------------
def wilson_ci(n_correct: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Khoảng tin cậy Wilson (mặc định 95%) cho một tỷ lệ, ví dụ accuracy trên tập test.

    Dùng Wilson thay cho xấp xỉ chuẩn vì n nhỏ (34 mẫu test) và accuracy gần 1: khoảng
    chuẩn khi đó vượt quá 1 hoặc co về độ rộng 0.
    """
    lo, hi = proportion_confint(n_correct, n, alpha=alpha, method="wilson")
    # NOTICE: theo công thức, khi đúng hết (hoặc sai hết) thì cận trên (cận dưới) Wilson bằng
    # đúng 1 (0), nhưng statsmodels trả về 0.9999999999999999 do làm tròn - khiến khoảng tin
    # cậy không chứa chính accuracy = 1. Gán lại giá trị chính xác ở hai đầu mút.
    if n_correct == n:
        hi = 1.0
    if n_correct == 0:
        lo = 0.0
    return float(lo), float(hi)


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Kiểm định McNemar (bản chính xác, nhị thức) so sánh hai mô hình trên CÙNG các mẫu test.

    Chỉ các mẫu mà đúng một trong hai mô hình đoán đúng mới mang thông tin; với 34 mẫu số
    cặp lệch nhỏ nên dùng bản chính xác thay cho chi bình phương. Dự đoán "không chắc
    chắn" tính là sai.
    """
    t = _encode(y_true)
    ok_a, ok_b = _encode(pred_a) == t, _encode(pred_b) == t
    only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    if only_a + only_b == 0:
        p_value = 1.0  # hai mô hình đúng/sai y hệt nhau: không có bằng chứng khác biệt
    else:
        table = [[int((ok_a & ok_b).sum()), only_a], [only_b, int((~ok_a & ~ok_b).sum())]]
        p_value = float(mcnemar(table, exact=True).pvalue)
    return {"n": len(t), "a_correct": int(ok_a.sum()), "b_correct": int(ok_b.sum()),
            "a_only_correct": only_a, "b_only_correct": only_b, "p_value": p_value}


def _encode(labels) -> np.ndarray:
    """Mã hóa nhãn hoặc dự đoán về số để tính chỉ số.

    Before:
    - ["ALL", "AML", "uncertain"]   (nhãn chuỗi; "uncertain" từ weighted voting của TV1)
    - [0, 1, 1]                     (nhãn đã mã hóa sẵn, 1 = AML)

    After:
    - [0, 1, -1]                    (-1 = không chắc chắn, luôn tính là sai)
    - [0, 1, 1]
    """
    arr = np.asarray(labels)
    if arr.dtype.kind in "iub":
        return arr.astype(int)
    s = arr.astype(str)
    return np.where(s == POSITIVE_CLASS, 1, np.where(s == NEGATIVE_CLASS, 0, -1))


def _metrics(y_true, y_pred, score) -> dict:
    """Chỉ số trên một tập dự đoán; AML là lớp dương, dự đoán không chắc chắn tính là sai."""
    t, p = _encode(y_true), _encode(y_pred)
    pos, neg = t == 1, t == 0
    sens = float((p[pos] == 1).mean()) if pos.any() else np.nan
    spec = float((p[neg] == 0).mean()) if neg.any() else np.nan
    n, k = len(t), int((p == t).sum())
    lo, hi = wilson_ci(k, n)
    # AUC chỉ xác định khi tập có đủ hai lớp; stratified fold luôn có đủ, chỉ fold 1 mẫu thì không.
    auc = float(roc_auc_score(t, score)) if pos.any() and neg.any() else np.nan
    return {"n_test": n, "bal_acc": (sens + spec) / 2, "auc": auc, "sensitivity": sens,
            "specificity": spec, "accuracy": k / n, "acc_ci_low": lo, "acc_ci_high": hi,
            "n_uncertain": int((p == -1).sum())}


def _positive_score(model, X) -> np.ndarray:
    """Điểm của lớp AML để tính AUC: cột AML của predict_proba, không có thì decision_function."""
    classes = list(model.classes_)
    positive = POSITIVE_CLASS if POSITIVE_CLASS in classes else 1
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, classes.index(positive)]
    # decision_function nhị phân dương nghĩa là classes_[1]; đảo dấu nếu AML đứng đầu.
    score = model.decision_function(X)
    return score if classes.index(positive) == 1 else -score


# ---------------------------------------------------------------------------
# Sơ đồ đánh giá
# ---------------------------------------------------------------------------
def _experiment_config() -> dict:
    with open(CONFIG_DIR / "experiment_config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _seeded(estimator, seed: int):
    """Bản sao chưa fit, với mọi tham số random_state (kể cả lồng trong Pipeline) gán bằng seed.

    Nhờ vậy mô hình có yếu tố ngẫu nhiên (Random Forest, SVM-RFE…) chạy lại cho đúng số cũ,
    và seed được ghi lại trong kết quả (nguyên tắc randomization, mục 1.5.2).
    """
    est = clone(estimator)
    keys = [k for k in est.get_params(deep=True) if k == "random_state" or k.endswith("__random_state")]
    if keys:
        est.set_params(**{k: seed for k in keys})
    return est


def _split_at_selector(pipeline: Pipeline) -> tuple[Pipeline, Pipeline]:
    """Tách pipeline thành (các bước đến hết bước chọn gen, các bước còn lại) cho wrong_cv.

    Bước chọn gen là bước tên "select" (quy ước trong mục 3.5.2); không có thì lấy bước
    cuối cùng là SelectorMixin của scikit-learn.
    """
    if not isinstance(pipeline, Pipeline):
        raise ValueError("wrong_cv cần một sklearn Pipeline có bước chọn gen")
    names = [name for name, _ in pipeline.steps]
    if "select" in names:
        cut = names.index("select")
    else:
        selectors = [i for i, (_, step) in enumerate(pipeline.steps) if isinstance(step, SelectorMixin)]
        if not selectors:
            raise ValueError("wrong_cv cần pipeline có bước chọn gen (tên 'select' hoặc SelectorMixin)")
        cut = selectors[-1]
    if cut == len(pipeline.steps) - 1:
        raise ValueError("wrong_cv cần ít nhất một bước (bộ phân loại) sau bước chọn gen")
    return Pipeline(pipeline.steps[:cut + 1]), Pipeline(pipeline.steps[cut + 1:])


def _resolve_split(X: pd.DataFrame, split) -> pd.Series:
    """Cột split (train/test) khớp chỉ mục X; mặc định đọc từ tầng cleansed theo sample_id."""
    if split is None:
        # X của mọi tầng đều có chỉ mục sample_id, nên lấy split đúng từ bảng mẫu tầng cleansed.
        split = pd.read_parquet(CLEANSED_DIR / "samples.parquet")["split"]
        split.index = split.index.astype(int)
    split = pd.Series(split).reindex(X.index)
    if split.isna().any() or not split.isin(["train", "test"]).all():
        raise ValueError("split phải có giá trị 'train'/'test' cho mọi mẫu của X")
    return split


def evaluate(pipeline, X: pd.DataFrame, y, scheme: str, seeds=None, split=None, param_grid=None) -> dict:
    """Đánh giá một pipeline scikit-learn theo một trong 4 sơ đồ ở mục 3.5.1.

    pipeline : estimator chưa fit (thường là Pipeline: tiền xử lý → chọn gen → phân loại).
               Mọi bước học từ dữ liệu phải nằm TRONG pipeline để được fit lại ở từng fold.
    X, y     : X có chỉ mục sample_id; y là nhãn "ALL"/"AML" (hoặc 0/1, 1 = AML) cùng thứ tự X.
    scheme   : "original_split" | "loocv" | "nested_cv" | "wrong_cv".
    seeds    : mỗi seed là một lần lặp (chia fold + random_state của mô hình). Mặc định:
               `seeds` trong experiment_config.yaml cho nested_cv/wrong_cv; seed đầu tiên
               cho original_split/loocv (cách chia cố định).
    split    : cột "train"/"test" theo sample_id cho original_split/loocv; mặc định lấy từ
               samples.parquet tầng cleansed.
    param_grid: lưới siêu tham số cho vòng trong (GridSearchCV, inner_folds trong YAML);
               không hỗ trợ với wrong_cv.

    Trả về dict:
        bal_acc_mean, bal_acc_sd, auc_mean : trung bình / độ lệch chuẩn qua các dòng per_fold
        per_fold    : DataFrame mỗi dòng một fold (loocv: một dòng gộp mỗi seed) với bal_acc,
                      auc, sensitivity, specificity, accuracy + Wilson CI, n_uncertain
        config      : cấu hình đã chạy (để ghi vào runs.parquet)
        predictions : dự đoán theo mẫu (sample_id, y_true, y_pred, score) — cần cho McNemar,
                      ROC (F11) và phân tích lỗi (TV5)

    bal_acc_sd là NaN khi chỉ có một ước lượng (original_split/loocv với một seed).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme phải là một trong {SCHEMES}, nhận {scheme!r}")
    if scheme == "wrong_cv" and param_grid is not None:
        raise ValueError("wrong_cv chỉ dùng để minh họa selection bias; không hỗ trợ param_grid")

    cfg = _experiment_config()
    outer_folds, inner_folds = int(cfg["cv"]["outer_folds"]), int(cfg["cv"]["inner_folds"])
    if seeds is None:
        seeds = list(cfg["seeds"]) if scheme in ("nested_cv", "wrong_cv") else [int(cfg["seeds"][0])]
    seeds = [int(s) for s in seeds]
    y = pd.Series(np.asarray(y), index=X.index) if not isinstance(y, pd.Series) else y.reindex(X.index)
    if y.isna().any():
        raise ValueError("y thiếu nhãn cho một số mẫu của X (kiểm tra chỉ mục sample_id)")
    if not np.isin(_encode(y), [0, 1]).all():
        raise ValueError(f"y chỉ được chứa {NEGATIVE_CLASS}/{POSITIVE_CLASS} hoặc 0/1")

    def fit(train_X, train_y, seed):
        model = _seeded(pipeline, seed)
        if param_grid is not None:
            inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
            model = GridSearchCV(model, param_grid, cv=inner, scoring="balanced_accuracy")
        return model.fit(train_X, train_y)

    rows, preds = [], []

    def record(seed, fold, model, test_X, test_y):
        pred, score = model.predict(test_X), _positive_score(model, test_X)
        rows.append({"seed": seed, "fold": fold, **_metrics(test_y, pred, score)})
        preds.append(pd.DataFrame({"seed": seed, "fold": fold, "y_true": test_y.to_numpy(),
                                   "y_pred": pred, "score": score}, index=test_y.index))

    for seed in seeds:
        if scheme == "original_split":
            sp = _resolve_split(X, split)
            tr, te = (sp == "train").to_numpy(), (sp == "test").to_numpy()
            record(seed, 0, fit(X[tr], y[tr], seed), X[te], y[te])

        elif scheme == "loocv":
            sp = _resolve_split(X, split)
            Xtr, ytr = X[(sp == "train").to_numpy()], y[(sp == "train").to_numpy()]
            fold_preds = []
            for tr, te in LeaveOneOut().split(Xtr):
                model = fit(Xtr.iloc[tr], ytr.iloc[tr], seed)
                fold_preds.append((model.predict(Xtr.iloc[te])[0], _positive_score(model, Xtr.iloc[te])[0]))
            # Một fold chỉ có 1 mẫu nên balanced accuracy/AUC theo fold vô nghĩa: gộp dự đoán
            # của cả 38 fold rồi mới tính chỉ số một lần.
            # để numpy tự suy kiểu (chuỗi hay số) - mảng object chứa số sẽ bị _encode đọc thành chuỗi
            pred = np.asarray([p for p, _ in fold_preds])
            score = np.asarray([s for _, s in fold_preds], dtype=float)
            rows.append({"seed": seed, "fold": "pooled", **_metrics(ytr, pred, score)})
            preds.append(pd.DataFrame({"seed": seed, "fold": np.arange(len(ytr)), "y_true": ytr.to_numpy(),
                                       "y_pred": pred, "score": score}, index=ytr.index))

        else:
            outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
            if scheme == "wrong_cv":
                pre, post = _split_at_selector(_seeded(pipeline, seed))
                # CỐ Ý RÒ RỈ: chọn gen nhìn thấy cả 72 mẫu (kể cả mẫu sẽ nằm ở fold kiểm tra).
                X_cv = pd.DataFrame(pre.fit_transform(X, y), index=X.index)
            for fold, (tr, te) in enumerate(outer.split(X, y)):
                if scheme == "wrong_cv":
                    model = clone(post).fit(X_cv.iloc[tr], y.iloc[tr])
                    record(seed, fold, model, X_cv.iloc[te], y.iloc[te])
                else:
                    record(seed, fold, fit(X.iloc[tr], y.iloc[tr], seed), X.iloc[te], y.iloc[te])

    per_fold = pd.DataFrame(rows)
    predictions = pd.concat(preds)
    predictions.index.name = "sample_id"
    config = {"scheme": scheme, "seeds": seeds, "n_samples": len(X), "positive_class": POSITIVE_CLASS,
              "outer_folds": outer_folds if scheme in ("nested_cv", "wrong_cv") else None,
              "inner_folds": inner_folds if param_grid is not None else None,
              "param_grid": param_grid, "pipeline": str(pipeline)}
    return {
        "bal_acc_mean": float(per_fold["bal_acc"].mean()),
        "bal_acc_sd": float(per_fold["bal_acc"].std(ddof=1)) if len(per_fold) > 1 else np.nan,
        "auc_mean": float(per_fold["auc"].mean()),
        "per_fold": per_fold,
        "config": config,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Thí nghiệm selection bias (mục 3.5.2, H3)
# ---------------------------------------------------------------------------
def selection_bias_experiment(pipeline, X: pd.DataFrame, y, n_permutations: int = N_PERMS_EVAL,
                              seeds=None, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """Bảng 2×2 của mục 5.3 (2.4): nhãn thật/hoán vị × CV đúng (nested_cv)/CV sai (wrong_cv).

    - Nhãn thật: mỗi seed (mặc định 10 seed trong YAML) một lần CV cho mỗi cách.
    - Nhãn hoán vị: n_permutations lần xáo nhãn (xóa mọi quan hệ thật giữa gen và nhãn);
      mỗi lần chạy CẢ HAI cách trên cùng nhãn đã xáo, với seed chia fold = số thứ tự lần lặp.

    Trả về mỗi dòng một lần chạy: labels ("real"|"permuted"), cv ("correct"|"wrong"), rep,
    seed, bal_acc, auc. Với nhãn hoán vị, CV đúng phải quanh 0.5; CV sai cao hơn hẳn là
    bằng chứng của selection bias.
    """
    cfg_seeds = list(_experiment_config()["seeds"]) if seeds is None else list(seeds)
    y = pd.Series(np.asarray(y), index=X.index) if not isinstance(y, pd.Series) else y.reindex(X.index)
    rng = np.random.default_rng(random_state)
    runs = [("real", rep, seed, y) for rep, seed in enumerate(cfg_seeds)]
    runs += [("permuted", rep, rep, pd.Series(rng.permutation(y.to_numpy()), index=y.index))
             for rep in range(n_permutations)]
    rows = []
    for labels, rep, seed, y_run in runs:
        for cv_name, scheme in (("correct", "nested_cv"), ("wrong", "wrong_cv")):
            res = evaluate(pipeline, X, y_run, scheme, seeds=[seed])
            rows.append({"labels": labels, "cv": cv_name, "rep": rep, "seed": seed,
                         "bal_acc": res["bal_acc_mean"], "auc": res["auc_mean"]})
    return pd.DataFrame(rows)


def summarize_selection_bias(runs: pd.DataFrame) -> pd.DataFrame:
    """Gộp kết quả selection_bias_experiment thành bảng 4 dòng (TB ± SD) để đưa vào báo cáo."""
    out = (runs.groupby(["labels", "cv"], sort=False)["bal_acc"]
           .agg(bal_acc_mean="mean", bal_acc_sd="std", n_runs="count").reset_index())
    return out


# ---------------------------------------------------------------------------
# Hình F10, F11 (sản phẩm bàn giao 2.4; tên file theo thuyết minh TV2)
# ---------------------------------------------------------------------------
def _save_both(fig, name: str) -> list:
    """Lưu một hình theo 2 bản: report (chữ nhỏ) và slide (chữ lớn)."""
    paths = []
    for style, out_dir in (("report", REPORT_FIG_DIR), ("slide", SLIDE_FIG_DIR)):
        cfg = _FIG_STYLES[style]
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.set_size_inches(cfg["figsize"])
        for item in fig.get_axes():
            if item is None:
                continue
            item.title.set_fontsize(cfg["fontsize"] + 2)
            item.xaxis.label.set_fontsize(cfg["fontsize"])
            item.yaxis.label.set_fontsize(cfg["fontsize"])
            item.tick_params(labelsize=cfg["fontsize"])
            for txt in item.texts:                    # nhãn in trực tiếp (vd. "0.50")
                txt.set_fontsize(cfg["fontsize"])
            if item.get_legend() is not None:
                for txt in item.get_legend().get_texts():
                    txt.set_fontsize(cfg["fontsize"])
        if fig._suptitle is not None:
            fig._suptitle.set_fontsize(cfg["fontsize"] + 4)
        path = out_dir / f"{name}_{style}.png"
        fig.savefig(path, dpi=cfg["dpi"], bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_selection_bias(runs: pd.DataFrame) -> list:
    """Hình F10: phân phối balanced accuracy của 4 tổ hợp (nhãn × cách CV), đường 0.5.

    Kỳ vọng (mục 3.5.2): nhãn thật thì cả hai cách đều cao; nhãn hoán vị thì CV đúng
    quanh 0.50 còn CV sai cao bất thường — đó là bằng chứng H3 (selection bias).
    """
    order = [("real", "correct"), ("real", "wrong"), ("permuted", "correct"), ("permuted", "wrong")]
    labels = ["Thật\nĐúng", "Thật\nSai", "Hoán vị\nĐúng", "Hoán vị\nSai"]
    colors = {"correct": "#2b7bba", "wrong": "#c0392b"}

    fig, ax = plt.subplots()
    frame = runs.copy()
    frame["combo"] = [f"{a}|{b}" for a, b in zip(frame["labels"], frame["cv"])]
    combo_order = [f"{a}|{b}" for a, b in order]
    sns.violinplot(data=frame, x="combo", y="bal_acc", order=combo_order,
                   hue="combo", hue_order=combo_order, legend=False,
                   inner=None, cut=0, linewidth=1.0, palette=[colors[c] for _, c in order],
                   alpha=0.35, ax=ax)
    rng = np.random.default_rng(RANDOM_SEED)
    for i, key in enumerate(order):
        vals = frame.loc[frame["combo"] == f"{key[0]}|{key[1]}", "bal_acc"].to_numpy()
        ax.scatter(i + rng.uniform(-0.09, 0.09, len(vals)), vals, s=14, color=colors[key[1]],
                   alpha=0.75, zorder=3, linewidths=0)
    ax.axhline(0.5, color="0.35", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(len(order) - 0.45, 0.505, "0.50", color="0.35", fontsize=_FIG_STYLES["report"]["fontsize"])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Nhãn × cách cross-validation")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Thí nghiệm selection bias (H3): 100 lần hoán vị nhãn × {CV đúng, CV sai}")
    ax.set_ylim(max(0.3, float(frame["bal_acc"].min()) - 0.05), 1.05)
    fig.tight_layout()
    return _save_both(fig, "F10_selection_bias")


def plot_roc_cm(y_true, y_pred, score, model_name: str = "baseline") -> list:
    """Hình F11: đường ROC + ma trận nhầm lẫn của mô hình (AML là lớp dương).

    Bản 1 vẽ cho pipeline tạm trên tập test 34 mẫu (original_split); khi nhóm chốt mô
    hình cuối (TV3, công việc 2.3) thì gọi lại với dự đoán của mô hình đó.
    """
    t, p = _encode(y_true), _encode(y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fpr, tpr, _ = roc_curve(t, score)
    auc = roc_auc_score(t, score)
    ax1.plot(fpr, tpr, color="#2b7bba", linewidth=1.8, label=f"AUC = {auc:.3f}")
    ax1.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1.0, label="chance")
    ax1.set_xlabel("False positive rate (1 − specificity)")
    ax1.set_ylabel("True positive rate (sensitivity)")
    ax1.set_title("ROC — AML là lớp dương")
    ax1.legend(loc="lower right", fontsize=_FIG_STYLES["report"]["fontsize"])

    cm = confusion_matrix(t, p, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                xticklabels=["ALL (dự đoán)", "AML (dự đoán)"],
                yticklabels=["ALL (thật)", "AML (thật)"], ax=ax2,
                annot_kws={"fontsize": _FIG_STYLES["report"]["fontsize"] + 2})
    ax2.set_title("Ma trận nhầm lẫn — tập test 34 mẫu")

    fig.suptitle(f"Mô hình cuối ({model_name})", fontsize=_FIG_STYLES["report"]["fontsize"] + 3)
    fig.tight_layout()
    return _save_both(fig, "F11_final_model_roc_cm")


# ---------------------------------------------------------------------------
# Chạy trên dữ liệu thật: python -m src.evaluate
# ---------------------------------------------------------------------------
def _threshold_log10(X):
    """Threshold [100, 16000] rồi log10 — hai bước cố định đầu tiên của mục 2.6."""
    return np.log10(np.clip(np.asarray(X, dtype=float), 100, 16000))


def _baseline_pipeline(k: int = 50) -> Pipeline:
    # NOTICE:
    # Bước "log10" là bản tạm thay cho tầng curated (X_log10.parquet) của TV3, vì
    # src/preprocess.py (công việc 1.2) chưa có. Đây là phép biến đổi cố định từng phần tử nên
    # đặt trong Pipeline không gây rò rỉ (mục 2.6). Khi TV3 giao X_log10: bỏ bước này, gọi
    # load_golub("curated") và dùng pipeline của src/models.py.
    return Pipeline([
        ("log10", FunctionTransformer(_threshold_log10)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("clf", LogisticRegression(max_iter=5000)),
    ])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Đánh giá pipeline tạm bằng 4 sơ đồ + thí nghiệm selection bias")
    ap.add_argument("--n-permutations", type=int, default=N_PERMS_EVAL,
                    help=f"số lần hoán vị nhãn (mặc định {N_PERMS_EVAL}; giảm khi chạy thử)")
    args = ap.parse_args(argv)

    # NOTICE:
    # Sau threshold, nhiều gen là hằng số trong một fold nên f_classif chia cho 0 và cảnh báo
    # hàng nghìn lần. SelectKBest tự xếp các điểm NaN xuống cuối nên kết quả không đổi; tắt
    # riêng hai cảnh báo này cho đầu ra đọc được. Xóa khi bước lọc gen của TV3 (GeneFilter)
    # đứng trước bước chọn gen.
    # Danh sách chỉ số gen trong thông điệp có xuống dòng nên dùng [\s\S] thay cho "."
    warnings.filterwarnings("ignore", message=r"Features [\s\S]* are constant", category=UserWarning)
    warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

    X, samples = load_golub("cleansed")
    y = samples["class"]
    pipe = _baseline_pipeline()
    print("Pipeline tạm: threshold+log10 → StandardScaler → SelectKBest(f_classif, k=50) → LogisticRegression\n")

    summary, results = [], {}
    for scheme in SCHEMES:
        res = evaluate(pipe, X, y, scheme, split=samples["split"])
        results[scheme] = res
        pf = res["per_fold"]
        summary.append({"scheme": scheme, "bal_acc_mean": res["bal_acc_mean"], "bal_acc_sd": res["bal_acc_sd"],
                        "auc_mean": res["auc_mean"], "sensitivity_mean": pf["sensitivity"].mean(),
                        "specificity_mean": pf["specificity"].mean(), "n_rows": len(pf)})
        sd = "(1 ước lượng)" if np.isnan(res["bal_acc_sd"]) else f"±{res['bal_acc_sd']:.3f}"
        print(f"{scheme:<15} bal_acc={res['bal_acc_mean']:.3f} {sd:<13} "
              f"auc={res['auc_mean']:.3f}  sens={pf['sensitivity'].mean():.3f}  "
              f"spec={pf['specificity'].mean():.3f}  ({len(pf)} dòng per_fold)")

    test = results["original_split"]
    row = test["per_fold"].iloc[0]
    print(f"\nTập test 34 mẫu: accuracy={row['accuracy']:.3f}, Wilson 95% [{row['acc_ci_low']:.3f}, "
          f"{row['acc_ci_high']:.3f}]")
    # Đối chứng (nguyên tắc control, mục 1.5.2): baseline luôn đoán lớp đa số.
    majority = evaluate(Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]), X, y,
                        "original_split", split=samples["split"])
    mc = mcnemar_test(test["predictions"]["y_true"], test["predictions"]["y_pred"],
                      majority["predictions"]["y_pred"])
    print(f"McNemar (LogReg 50 gen vs lớp đa số) trên test: đúng {mc['a_correct']} vs {mc['b_correct']}/"
          f"{mc['n']}, p = {mc['p_value']:.2g}")

    print(f"\nThí nghiệm selection bias: {args.n_permutations} lần hoán vị nhãn × {{CV đúng, CV sai}} …")
    runs = selection_bias_experiment(pipe, X, y, n_permutations=args.n_permutations)
    sb = summarize_selection_bias(runs)
    names = {("real", "correct"): "Thật     Đúng", ("real", "wrong"): "Thật     Sai ",
             ("permuted", "correct"): "Hoán vị  Đúng", ("permuted", "wrong"): "Hoán vị  Sai "}
    print("Nhãn     CV    Balanced accuracy (TB ± SD)   số lần")
    for _, r in sb.iterrows():
        print(f"{names[(r['labels'], r['cv'])]}   {r['bal_acc_mean']:.3f} ± {r['bal_acc_sd']:.3f}"
              f"                 {r['n_runs']}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(METRICS_DIR / "summary_evaluation.csv", index=False)
    sb.to_csv(METRICS_DIR / "summary_selection_bias.csv", index=False)
    runs.to_csv(METRICS_DIR / "selection_bias_runs.csv", index=False)
    print(f"\nĐã ghi {METRICS_DIR}/summary_evaluation.csv, summary_selection_bias.csv, selection_bias_runs.csv")

    figs = plot_selection_bias(runs)
    figs += plot_roc_cm(test["predictions"]["y_true"], test["predictions"]["y_pred"],
                        test["predictions"]["score"], model_name="LogReg 50 gen (pipeline tạm)")
    print("Đã ghi hình F10/F11 (bản 1):\n  " + "\n  ".join(str(p) for p in figs))


if __name__ == "__main__":
    main()
