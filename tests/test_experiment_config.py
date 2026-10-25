import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CONFIG_DIR  # noqa: E402


def _cfg():
    with open(CONFIG_DIR / "experiment_config.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_252_cau_hinh():
    cfg = _cfg()
    assert len(cfg["selection"]) * len(cfg["k"]) * len(cfg["classifiers"]) == 252


def test_khong_trung_lap_va_k_tang_dan():
    cfg = _cfg()
    for key in ("selection", "k", "classifiers", "seeds"):
        assert len(set(cfg[key])) == len(cfg[key]), f"{key} có giá trị trùng"
    assert cfg["k"] == sorted(cfg["k"]) and all(k > 0 for k in cfg["k"])


def test_outer_repeats_bang_so_seed():
    cfg = _cfg()
    assert cfg["cv"]["outer_repeats"] == len(cfg["seeds"])
    assert cfg["cv"]["stratified"] is True