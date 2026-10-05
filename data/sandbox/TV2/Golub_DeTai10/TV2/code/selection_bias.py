"""
TV2 - GĐ2: Mục 2.4 Thí nghiệm selection bias (mục 3.5.2 kế hoạch) → hình F10.

Pipeline chuẩn (đúng như kế hoạch): StandardScaler + SelectKBest(f_classif) +
LogisticRegression(l2). Chọn gen NẰM TRONG pipeline.

Bảng 4 tổ hợp: {nhãn thật, hoán vị} × {CV đúng, CV sai}
  - Thật/Đúng  : cao
  - Thật/Sai   : cao, có thể cao hơn CV đúng
  - Hoán vị/Đúng: ≈ 0.50                       -> hiệu năng thật khi không có tín hiệu
  - Hoán vị/Sai : cao bất thường               -> BẰNG CHỨNG H3 (chọn gen trước khi CV)

CV "đúng": SelectKBest fit trong từng fold (scikit pipeline, cross_val_score).
CV "sai" : fit Scale + SelectKBest trên TOÀN BỘ (dính test) rồi mới CV classifier.

Đầu ra: results/metrics/selection_bias_summary.csv (4 dòng) +
         results/metrics/selection_bias_permutations.csv (chi tiết từng hoán vị,
         để TV6 vẽ phân phối F10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.config import (CURATED_DIR, STD_DIR, METRICS_DIR, N_PERMS_EVAL, RANDOM_SEED)
except ModuleNotFoundError:
    from config import (CURATED_DIR, STD_DIR, METRICS_DIR, N_PERMS_EVAL, RANDOM_SEED)


def build_pipeline() -> Pipeline:
    """Selection NẰM TRONG pipeline (đúng cách)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=50)),
        ("clf", LogisticRegression(penalty="l2", max_iter=5000, C=1.0)),
    ])


def cv_correct(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray,
               n_splits: int = 5, seed: int = RANDOM_SEED) -> np.ndarray:
    """CV đúng: mọi bước (scale+select+clf) được refit trong từng fold."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_score(pipeline, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)


def cv_wrong(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray,
             n_splits: int = 5, seed: int = RANDOM_SEED) -> np.ndarray:
    """CV sai (H3): fit scale+chọn gen trên TOÀN BỘ rồi mới CV classifier trên phần đã chọn."""
    steps = [("c", pipeline.named_steps["scale"]), ("s", pipeline.named_steps["select"])]
    X_leak = Pipeline(steps).fit_transform(X, y)          # rò rỉ: học trên cả test
    clf = Pipeline([("c", pipeline.named_steps["clf"])])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_score(clf, X_leak, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)


def experiment(X: pd.DataFrame, y, n_perm: int = N_PERMS_EVAL,
               n_features: int = 2000) -> pd.DataFrame:
    """Chạy bảng 4 tổ hợp. n_features: giảm số gen trước thí nghiệm cho nhanh
    (giữ gen có phương sai cao, như TV1/CV thường làm)."""
    y_true = np.asarray(y)
    if X.shape[1] > n_features:
        top = np.argsort(X.var(0).values)[-n_features:]
        Xs = X.iloc[:, top].copy()
        print(f"Rút gọn {X.shape[1]} -> {n_features} gen (top variance) cho thí nghiệm.")
    else:
        Xs = X

    pipe = build_pipeline()
    rng = np.random.default_rng(RANDOM_SEED)
    long_rows = []
    combos = {}

    def _run(tag: str, yv: np.ndarray, cv_func, seed: int = RANDOM_SEED):
        return cv_func(pipe, Xs, yv, seed=seed)

    # 1. Nhãn thật
    for cv_name, cv_func in (("đúng", cv_correct), ("sai", cv_wrong)):
        s = _run("thật", y_true, cv_func)
        combos[("thật", cv_name)] = s
        long_rows += [{"label": "thật", "cv": cv_name, "perm_id": -1, "bal_acc": float(v)} for v in s]
        print(f"Thật/{cv_name:<5}: {s.mean():.3f} ± {s.std():.3f}")

    # 2. Nhãn hoán vị (xóa mọi quan hệ gen-nhãn)
    perm = {"đúng": np.zeros(n_perm), "sai": np.zeros(n_perm)}
    for i in range(n_perm):
        y_perm = y_true[rng.permutation(len(y_true))]
        s_c = cv_correct(pipe, Xs, y_perm, seed=RANDOM_SEED + i)
        s_w = cv_wrong(pipe, Xs, y_perm, seed=RANDOM_SEED + i)
        perm["đúng"][i] = s_c.mean()
        perm["sai"][i] = s_w.mean()
        long_rows.append({"label": "hoán vị", "cv": "đúng", "perm_id": i, "bal_acc": float(s_c.mean())})
        long_rows.append({"label": "hoán vị", "cv": "sai", "perm_id": i, "bal_acc": float(s_w.mean())})

    rows = [{
        "label": lb, "cv": cv,
        "bal_acc_mean": float(np.mean(v)), "bal_acc_sd": float(np.std(v)),
        "kỳ_vọng": _expect(lb, cv),
    } for (lb, cv), v in sorted(combos.items())]
    rows += [
        {"label": "hoán vị", "cv": "đúng",
         "bal_acc_mean": float(perm["đúng"].mean()), "bal_acc_sd": float(perm["đúng"].std()),
         "kỳ_vọng": "≈0.50"},
        {"label": "hoán vị", "cv": "sai",
         "bal_acc_mean": float(perm["sai"].mean()), "bal_acc_sd": float(perm["sai"].std()),
         "kỳ_vọng": "cao bất thường (H3)"},
    ]
    summary = pd.DataFrame(rows)
    detail = pd.DataFrame(long_rows)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(METRICS_DIR / "selection_bias_summary.csv", index=False)
    detail.to_csv(METRICS_DIR / "selection_bias_permutations.csv", index=False)
    print("\nBảng selection bias (F10):")
    print(summary.to_string(index=False))
    print(f"\nĐã ghi: {METRICS_DIR/'selection_bias_summary.csv'} và "
          f"{METRICS_DIR/'selection_bias_permutations.csv'}")
    return summary


def _expect(label: str, cv: str) -> str:
    if label == "thật":
        return "cao" if cv == "đúng" else "cao, có thể cao hơn CV đúng"
    return "≈ 0.50" if cv == "đúng" else "cao bất thường (bằng chứng H3)"


def load_expression() -> tuple[pd.DataFrame, np.ndarray]:
    """Đọc X-log10 từ curated nếu có (TV3), ngược lại tự làm threshold+log10 từ standardized
    (phép biến đổi CỐ ĐỊNH [100,16000]+log10, cho phép áp lên cả 72 mẫu - mục 2.6)."""
    if (CURATED_DIR / "X_log10.parquet").exists():
        X = pd.read_parquet(CURATED_DIR / "X_log10.parquet")
        y = pd.read_parquet(CURATED_DIR / "samples.parquet")["class"]
        print("Đọc từ tầng curated (X_log10).")
        return X, np.asarray(y)
    X = pd.read_parquet(STD_DIR / "expression.parquet")
    y = pd.read_parquet(STD_DIR / "samples.parquet")["class"]
    X = X.clip(lower=100, upper=16000)          # bước 1: threshold
    X = np.log10(X)                             # bước 2: log10
    print("Curated chưa có -> dùng threshold+log10 từ standardized (mục 2.6).")
    return X, np.asarray(y)


if __name__ == "__main__":
    X, y = load_expression()
    experiment(X, y, n_perm=min(N_PERMS_EVAL, 10) if X.shape[1] < 5000 else N_PERMS_EVAL)