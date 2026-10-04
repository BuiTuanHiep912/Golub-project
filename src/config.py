"""Cấu hình dùng chung cho cả project (TV2 phụ trách)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW, STANDARDIZED, CLEANSED, CURATED = (DATA / p for p in ["raw", "standardized", "cleansed", "curated"])
RESULTS = ROOT / "results"
FIGURES_REPORT = RESULTS / "figures" / "report"
FIGURES_SLIDES = RESULTS / "figures" / "slides"
METRICS = RESULTS / "metrics"
GENE_LISTS = RESULTS / "gene_lists"
EXPERIMENT_CONFIG = ROOT / "config" / "experiment_config.yaml"

SEED = 42

# Bảng màu an toàn với người mù màu (Okabe–Ito); dùng thống nhất trong mọi hình
CLASS_COLORS = {"ALL": "#0072B2", "AML": "#E69F00"}
CLASS_MARKERS = {"ALL": "o", "AML": "^"}

# Tiền xử lý theo Dudoit et al. (2002)
THRESHOLD_MIN, THRESHOLD_MAX = 100, 16000
FILTER_FOLD, FILTER_DIFF = 5, 500
