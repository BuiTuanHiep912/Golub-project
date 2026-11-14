"""Golden-file test cho src/evaluate.py (TV2 - công việc 2.4).

File này KHÔNG chứa test case nào. Test case nằm trong dữ liệu:

    tests/input/testcaseN/mota.txt       mô tả tiếng Việt: mục đích, đầu vào, kỳ vọng, vì sao quan trọng
    tests/input/testcaseN/params.json    "kind" chọn hàm cần gọi + tham số của lần gọi + "desc":
        wilson_ci           n_correct, n
        mcnemar             y_true, pred_a, pred_b; tuỳ chọn index_true / index_a / index_b là danh
                            sách sample_id -> đầu vào tương ứng thành pd.Series có chỉ mục đó
        confusion_counts    y_true, y_pred
        encode              labels
        evaluate            scheme, data {signal, n_genes, seed}, k, seeds?, param_grid?
                            (dữ liệu tổng hợp 72 mẫu, nhãn và split đúng Bảng 2.4; 10 gen đầu lệch
                            `signal` độ lệch chuẩn giữa AML và ALL, còn lại là nhiễu)
        split_at_selector   steps [[tên, loại]] với loại ∈ scale / select / clf / noop
    tests/expect/testcaseN/result.json   kết quả kỳ vọng (số thực làm tròn 6 chữ số)
    tests/expect/testcaseN/error.json    {"type", "message_contains"} nếu kỳ vọng raise
    tests/actual/testcaseN/...           sinh ra khi chạy rồi đem so với expect/ (nằm trong .gitignore)

Thêm test case = thêm thư mục, không sửa file Python này.

Chạy:  python -m pytest tests/test_golden_evaluate.py -v
       python -m unittest tests.test_golden_evaluate -v

Không có lời gọi mạng nào: evaluate() chỉ đọc config/experiment_config.yaml; split luôn truyền
thẳng nên không đọc dữ liệu thật của tầng cleansed.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))

from src.evaluate import (  # noqa: E402
    _encode, _split_at_selector, confusion_counts, evaluate, mcnemar_test, wilson_ci,
)

INPUT, EXPECT, ACTUAL = BASE / "input", BASE / "expect", BASE / "actual"


# --------------------------------------------------------------------------- #
# dựng đầu vào từ params.json
# --------------------------------------------------------------------------- #
def _series_or_list(values, index):
    """Có danh sách sample_id thì dựng pd.Series (như cột predictions của evaluate), không thì giữ list."""
    return pd.Series(values, index=pd.Index(index, name="sample_id")) if index is not None else values


def _synthetic(data: dict) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """72 mẫu: train 1..38 (27 ALL, 11 AML), test 39..72 (20 ALL, 14 AML) như Bảng 2.4."""
    ids = pd.Index(range(1, 73), name="sample_id")
    y = pd.Series(["ALL"] * 27 + ["AML"] * 11 + ["ALL"] * 20 + ["AML"] * 14, index=ids, name="class")
    split = pd.Series(["train"] * 38 + ["test"] * 34, index=ids)
    rng = np.random.default_rng(data["seed"])
    X = rng.normal(size=(72, data["n_genes"]))
    X[:, :10] += data["signal"] * (y.to_numpy() == "AML")[:, None]
    return pd.DataFrame(X, index=ids, columns=[f"g{i}" for i in range(data["n_genes"])]), y, split


def _step(kind: str):
    return {"scale": StandardScaler, "select": lambda: SelectKBest(f_classif, k=5),
            "clf": lambda: LogisticRegression(max_iter=5000), "noop": FunctionTransformer}[kind]()


def _num(x):
    """Số thực -> làm tròn 6 chữ số để so được giữa các máy; NaN -> null."""
    x = float(x)
    return None if np.isnan(x) else round(x, 6)


# --------------------------------------------------------------------------- #
# chạy 1 testcase, ghi mọi thứ vào actual/
# --------------------------------------------------------------------------- #
def run_case(name: str) -> Path:
    params = json.loads((INPUT / name / "params.json").read_text(encoding="utf-8"))
    out = ACTUAL / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    kind, result = params["kind"], None
    try:
        if kind == "wilson_ci":
            result = [_num(v) for v in wilson_ci(params["n_correct"], params["n"])]
        elif kind == "mcnemar":
            res = mcnemar_test(_series_or_list(params["y_true"], params.get("index_true")),
                               _series_or_list(params["pred_a"], params.get("index_a")),
                               _series_or_list(params["pred_b"], params.get("index_b")))
            result = {k: (_num(v) if k == "p_value" else v) for k, v in res.items()}
        elif kind == "confusion_counts":
            cm = confusion_counts(params["y_true"], params["y_pred"])
            result = {"index": list(cm.index), "columns": list(cm.columns), "counts": cm.to_numpy().tolist()}
        elif kind == "encode":
            result = _encode(params["labels"]).tolist()
        elif kind == "evaluate":
            X, y, split = _synthetic(params["data"])
            pipe = Pipeline([("scale", StandardScaler()), ("select", SelectKBest(f_classif, k=params["k"])),
                             ("clf", LogisticRegression(max_iter=5000))])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = evaluate(pipe, X, y, params["scheme"], seeds=params.get("seeds"), split=split,
                               param_grid=params.get("param_grid"))
            pf, pred = res["per_fold"], res["predictions"]
            result = {"keys": list(res), "per_fold_rows": len(pf), "n_test": [int(v) for v in pf["n_test"]],
                      "bal_acc_mean": _num(res["bal_acc_mean"]), "bal_acc_sd": _num(res["bal_acc_sd"]),
                      "auc_mean": _num(res["auc_mean"]), "inner_folds": res["config"]["inner_folds"],
                      "n_predictions": len(pred), "prediction_ids": [int(pred.index.min()), int(pred.index.max())]}
        elif kind == "split_at_selector":
            pre, post = _split_at_selector(Pipeline([(n, _step(k)) for n, k in params["steps"]]))
            result = {"pre": [n for n, _ in pre.steps], "post": [n for n, _ in post.steps]}
        else:
            raise KeyError(f"kind lạ trong params.json: {kind}")
    except Exception as exc:  # noqa: BLE001 - ghi lại để so với expect/error.json
        dump(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
    if result is not None:
        dump(out / "result.json", result)
    return out


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# so actual/ với expect/
# --------------------------------------------------------------------------- #
def compare(tc: unittest.TestCase, name: str, exp: Path, act: Path) -> None:
    """Mọi file trong expect/ phải có trong actual/ và khớp; lỗi không kỳ vọng thì fail."""
    exp_files = sorted(p for p in exp.glob("*.json")) if exp.is_dir() else []
    # Thiếu expect/ thì vòng so bên dưới không chạy lần nào và testcase xanh mà không kiểm gì.
    tc.assertTrue(exp_files, f"{name}: expect/ rỗng hoặc không có — testcase không kiểm gì")

    if not (exp / "error.json").exists() and (act / "error.json").exists():
        raised = json.loads((act / "error.json").read_text(encoding="utf-8"))
        tc.fail(f"{name}: không kỳ vọng lỗi nhưng raise {raised['type']}: {raised['message']}")

    for exp_file in exp_files:
        act_file = act / exp_file.name
        tc.assertTrue(act_file.exists(), f"{name}: actual/ thiếu {exp_file.name}")
        want = json.loads(exp_file.read_text(encoding="utf-8"))
        got = json.loads(act_file.read_text(encoding="utf-8"))
        if exp_file.name == "error.json":
            tc.assertEqual(want["type"], got["type"], f"{name}: sai loại exception")
            tc.assertIn(want["message_contains"], got["message"], f"{name}: message thiếu thông tin")
        else:
            tc.assertEqual(want, got, f"{name}: {exp_file.name} khác expect")


# --------------------------------------------------------------------------- #
# sinh test method từ thư mục input/
# --------------------------------------------------------------------------- #
class TestGoldenEvaluate(unittest.TestCase):
    maxDiff = None


def make_test(name: str):
    def test(self):
        compare(self, name, EXPECT / name, run_case(name))
    desc = json.loads((INPUT / name / "params.json").read_text(encoding="utf-8")).get("desc", "")
    test.__doc__ = f"{name}: {desc}"
    return test


def natural_key(path: Path) -> int:
    return int(re.search(r"\d+", path.name).group())


if not INPUT.is_dir():
    raise RuntimeError(f"không thấy thư mục testcase: {INPUT}")

for _dir in sorted(INPUT.glob("testcase*"), key=natural_key):
    setattr(TestGoldenEvaluate, f"test_{_dir.name}", make_test(_dir.name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
