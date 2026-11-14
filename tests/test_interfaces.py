"""Khóa giao diện dùng chung của giai đoạn 2 (ADR-014).

Test chỉ kiểm tra tên module và tên hàm tồn tại, không kiểm tra kết quả. Ai muốn đổi
tên thì báo nhóm, ghi vào docs/decisions.md, rồi mới sửa test này.
"""
import importlib

import pytest

INTERFACES = {
    "src.golub_wv": ["GolubWeightedVoting"],   # 2.1, TV1
    "src.patient": ["patient_profile"],        # 2.1, TV1
    "src.gene_explorer": ["find_gene", "gene_stats"],  # 2.2, TV4
    "src.warehouse": ["WAREHOUSE_PATH", "build_warehouse", "connect"],  # 2.3, TV3
    "src.ingest": ["ingest_batches", "validate_new_sample"],  # 2.4, TV2
}


@pytest.mark.parametrize("module, names", INTERFACES.items())
def test_interface_names_exist(module, names):
    mod = importlib.import_module(module)
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing, f"{module} thiếu {missing}"


def test_golub_wv_methods():
    from src.golub_wv import GolubWeightedVoting
    clf = GolubWeightedVoting(n_genes=50, ps_threshold=0.3)
    for name in ("fit", "predict", "predict_proba", "prediction_strength"):
        assert callable(getattr(clf, name))
