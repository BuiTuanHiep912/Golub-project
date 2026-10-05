"""
TV2 - GĐ2: Mục 2.4 ROC + confusion matrix mô hình cuối (hình F11).

Chạy default với dữ liệu thật (threshold+log10 từ standardized, tách 38/34 theo cột
split, pipeline StandardScaler + SelectKBest(50) + LogisticRegression) để sinh ngay
hình F11 thật. Có thể truyền kết quả từ src/evaluate cho mô hình/ước lượng khác.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.config import STD_DIR, REPORT_FIG_DIR, PALETTE, RANDOM_SEED
except ModuleNotFoundError:
    from config import STD_DIR, REPORT_FIG_DIR, PALETTE, RANDOM_SEED


def roc_cm(y_true, y_pred, y_prob: np.ndarray | None, model_name: str = "final_model") -> str:
    """Vẽ ROC + confusion matrix (AML = lớp dương), lưu results/figures/report/F11_*.png."""
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score as _auc

    matplotlib.use("Agg")
    yt = (np.asarray(y_true) == "AML").astype(int)
    yp = (np.asarray(y_pred) == "AML").astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(yt, y_prob)
        auc_val = _auc(yt, y_prob)
        ax1.plot(fpr, tpr, color=PALETTE["AML"], lw=2, label=f"AUC = {auc_val:.3f}")
        ax1.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["AML"])
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate",
            ylabel="True positive rate", title=f"ROC - {model_name}")
    ax1.legend(loc="lower right")

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    im = ax2.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax2.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["ALL", "AML"],
            yticklabels=["ALL", "AML"], title="Confusion matrix",
            xlabel="Dự đoán", ylabel="Thực tế")
    ax2.set_xticklabels(["ALL", "AML"], rotation=0)

    out = REPORT_FIG_DIR / "F11_roc_cm.png"
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"{model_name} · test set 38/34 (n = {len(yt)})", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print("Đã lưu:", out)
    return str(out)


def _default_figure() -> str:
    """Sinh F11 từ dữ liệu thật: preprocess cố định, huấn luyện trên 38, đánh giá trên 34."""
    X = pd.read_parquet(STD_DIR / "expression.parquet").clip(lower=100, upper=16000)
    X = np.log10(X)
    samples = pd.read_parquet(STD_DIR / "samples.parquet")
    tr = samples["split"] == "train"
    y = samples["class"]
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=50)),
        ("clf", LogisticRegression(penalty="l2", max_iter=5000)),
    ])
    model = pipe.fit(X[tr], y[tr])
    pred = model.predict(X[~tr])
    prob = model.predict_proba(X[~tr])[:, 1]
    return roc_cm(y[~tr], pred, prob, "GolubWV-like · 50 genes + LogReg")


if __name__ == "__main__":
    _default_figure()