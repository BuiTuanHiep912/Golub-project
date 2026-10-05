"""
TV2 - GĐ2: Mục 2.4 Khung đánh giá mô hình (src/evaluate.py).

evaluate(pipeline, X, y, scheme) với 4 sơ đồ theo mục 3.5.1:
  - original_split : tách gốc 38/34 giống bài báo (dùng cột split hoặc 38 mẫu đầu)
  - loocv          : LOOCV ĐÚNG CÁCH trên 38 mẫu train; chọn gen lặp lại mỗi fold
                     (selection NẰM TRONG pipeline) — note 22 của bài báo
  - nested_cv      : nested repeated stratified CV trên 72 mẫu (GridSearchCV trong
                     vòng trong nếu truyền param_grid)
  - wrong_cv       : CV SAI — fit scale/chọn gen trên TOÀN BỘ rồi mới CV (minh hoạ H3)

Chỉ số: balanced accuracy, AUC, sensitivity/specificity (AML = lớp dương), F1,
Wilson interval cho accuracy tập test; McNemar cho cặp mô hình trên cùng test set.
Mọi bước tiền xử lý (scale, chọn gen) phải NẰM TRONG pipeline để tránh rò rỉ.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, LeaveOneOut, RepeatedStratifiedKFold, StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCHEMES = ("original_split", "loocv", "nested_cv", "wrong_cv")


@dataclass
class EvalResult:
    scheme: str
    bal_acc_mean: float
    bal_acc_sd: float
    auc_mean: float
    auc_sd: float
    per_fold: list
    config: dict = field(default_factory=dict)
    sensitivity: float = np.nan
    specificity: float = np.nan
    f1: float = np.nan
    accuracy: float = np.nan
    acc_ci_low: float = np.nan
    acc_ci_high: float = np.nan

    def keys(self):
        return ["bal_acc_mean", "bal_acc_sd", "auc_mean", "per_fold", "config"]


def wilson_ci(pos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson cho tỷ lệ (mục 3.5.3)."""
    if n == 0:
        return (np.nan, np.nan)
    p_hat = pos / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def mcnemar(y_true, y_a, y_b) -> tuple[float, float]:
    """Kiểm định McNemar (hiệu chỉnh liên tục) cho 2 mô hình trên cùng test set.
    Trả về (chi2, p)."""
    b = int(np.sum((y_a == y_true) & (y_b != y_true)))
    c = int(np.sum((y_a != y_true) & (y_b == y_true)))
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else np.nan
    p = stats.chi2.sf(chi2, df=1) if not np.isnan(chi2) else 1.0
    return chi2, p


def _binary_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Chỉ số nhị phân với AML = lớp dương (mục 3.5.3)."""
    y_true = (np.asarray(y_true) == "AML").astype(int)
    y_pred = (np.asarray(y_pred) == "AML").astype(int)
    ba = balanced_accuracy_score(y_true, y_pred)
    acc = float(np.mean(y_true == y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    if set(np.unique(y_true)) == {0, 1} and set(np.unique(y_pred)) == {0, 1}:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        tn, fp, fn, tp = 0.0, 0.0, 0.0, float(np.sum(y_pred == 1))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    auc_val = roc_auc_score(y_true, y_prob, labels=[0, 1]) \
        if y_prob is not None and set(np.unique(y_true)) == {0, 1} else np.nan
    return {"bal_acc": ba, "acc": acc, "sens": sens, "spec": spec, "f1": f1, "auc": auc_val}


def _split_index(X: pd.DataFrame, split) -> tuple[np.ndarray, np.ndarray]:
    """Chỉ mục train/test: ưu tiên cột split (Series cùng index), ngược lại 38 đầu."""
    if split is not None:
        s = pd.Series(split).reindex(X.index).fillna("test")
        return np.where(s == "train")[0], np.where(s == "test")[0]
    return np.arange(38), np.arange(38, len(X))


def evaluate(pipeline, X: pd.DataFrame, y, scheme: str = "nested_cv",
             param_grid: dict | None = None, split=None,
             seeds=range(10), inner_splits: int = 5) -> EvalResult:
    """Chạy 1 trong 4 sơ đồ đánh giá (mục 3.5.1).

    pipeline MUST chứa mọi bước tiền xử lý (scale, chọn gen) trong Pipeline để mỗi
    fold chỉ fit trên phần train. param_grid dùng cho nested_cv khi cần tối ưu.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme phải ∈ {SCHEMES}")

    y = np.asarray(y)
    tr_idx, te_idx = _split_index(X, split)

    if scheme == "original_split":
        est = clone(pipeline).fit(X.iloc[tr_idx], y[tr_idx])
        pred = est.predict(X.iloc[te_idx])
        prob = est.predict_proba(X.iloc[te_idx])[:, 1] if hasattr(est, "predict_proba") else None
        m = _binary_metrics(y[te_idx], pred, prob)
        lo, hi = wilson_ci(int(np.sum(y[te_idx] == pred)), len(te_idx))
        per_fold = [(str(te_idx[i]), pred[i]) for i in range(len(te_idx))]
        return EvalResult(
            scheme, m["bal_acc"], 0.0, m["auc"], 0.0, per_fold,
            {"n_train": int(len(tr_idx)), "n_test": int(len(te_idx))},
            sensitivity=m["sens"], specificity=m["spec"], f1=m["f1"], accuracy=m["acc"],
            acc_ci_low=lo, acc_ci_high=hi,
        )

    if scheme == "loocv":
        # Đúng cách: chỉ trên 38 mẫu train, selection lặp lại trong từng fold
        Xtr = X.iloc[tr_idx]
        ytr = y[tr_idx]
        per_fold, fold_auc = [], []
        for f_in, f_out in LeaveOneOut().split(Xtr, ytr):
            est = clone(pipeline).fit(Xtr.iloc[f_in], ytr[f_in])
            pred = est.predict(Xtr.iloc[f_out])
            prob = est.predict_proba(Xtr.iloc[f_out])[:, 1] if hasattr(est, "predict_proba") else None
            m = _binary_metrics(ytr[f_out], pred, prob)
            per_fold.append(m["bal_acc"])
            fold_auc.append(m["auc"])
        n = len(per_fold)
        val = np.array([v for v in per_fold if not np.isnan(v)])
        au = np.array([v for v in fold_auc if not np.isnan(v)])
        return EvalResult(
            scheme, float(val.mean()) if len(val) else np.nan,
            float(val.std()) if len(val) > 1 else 0.0,
            float(au.mean()) if len(au) else np.nan,
            float(au.std()) if len(au) > 1 else 0.0,
            per_fold, {"folds": n},
        )

    if scheme == "wrong_cv":
        # SAI (H3): fit scale/chọn gen trên TOÀN BỘ rồi mới CV. Classifier tách riêng.
        preproc = [clone(s) for _, s in pipeline.steps[:-1]]
        clf = clone(pipeline.steps[-1][1])
        X_leak = X
        for step in preproc:
            X_leak = step.fit_transform(X_leak, y)   # rò rỉ: học trên cả test
        leaky = Pipeline([("c", clf)])
        out = []
        for seed in seeds:
            cv = RepeatedStratifiedKFold(n_splits=inner_splits, n_repeats=1, random_state=int(seed))
            scores = cross_val_score(leaky, X_leak, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
            out.extend(scores.tolist())
        out = np.asarray(out)
        return EvalResult(
            scheme, float(out.mean()), float(out.std()), np.nan, np.nan, out.tolist(),
            {"leakage": True, "n": int(len(out))},
        )

    if scheme == "nested_cv":
        outer = RepeatedStratifiedKFold(n_splits=inner_splits, n_repeats=len(seeds), random_state=0)
        if param_grid:
            inner = StratifiedKFold(inner_splits, shuffle=True, random_state=1)
            estimator = GridSearchCV(clone(pipeline), param_grid, cv=inner,
                                     scoring="balanced_accuracy", n_jobs=-1)
        else:
            estimator = clone(pipeline)
        scores = cross_val_score(estimator, X, y, cv=outer, scoring="balanced_accuracy", n_jobs=-1)
        return EvalResult(
            scheme, float(scores.mean()), float(scores.std()), np.nan, np.nan, scores.tolist(),
            {"outer": outer.get_n_splits(), "tuned": bool(param_grid),
             "inner_cv": inner_splits if param_grid else None},
        )
    raise NotImplementedError(scheme)


# ---------------------------------------------------------------------------
# Demo chạy thử
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Chạy thử trên dữ liệu mẫu (72x200 ngẫu nhiên)...")
    rng = np.random.default_rng(0)
    Xd = pd.DataFrame(rng.normal(size=(72, 200)))
    yd = np.array(["ALL"] * 47 + ["AML"] * 25)
    idx = rng.permutation(72)
    Xd = Xd.iloc[idx].reset_index(drop=True)
    yd = yd[idx]
    split_s = pd.Series(["train"] * 38 + ["test"] * 34)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(k=20)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    for s in SCHEMES:
        r = evaluate(pipe, Xd, pd.Series(yd), scheme=s, split=split_s)
        extra = f"  acc={r.accuracy:.3f}" if not np.isnan(r.accuracy) else ""
        print(f"{s:14s} bal_acc={r.bal_acc_mean:.3f}±{r.bal_acc_sd:.3f}  "
              f"auc={r.auc_mean:.3f}{extra}")