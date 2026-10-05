"""
TV2 - src/config.py (mục 2.2 kiến trúc lưu trữ theo tầng, Bảng 2.5/2.6).

Gốc đường dẫn là thư mục repo `golub-project` (chứa src/, data/, results/, docs/).
Cách dùng chung cho mọi module:

    from src.config import DATA_DIR, RAW_DIR, STD_DIR, ...

Khi chạy trực tiếp một module (`python src/load.py`) thì `src` chưa nằm trên
sys.path, nên các module dùng try/except:

    try:
        from src.config import ...
    except ModuleNotFoundError:
        from config import ...
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/ -> golub-project

# Tầng dữ liệu (mục 2.2, Bảng 2.5)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                      # Tầng 1: dữ liệu gốc, không sửa
KAGGLE_DIR = RAW_DIR / "kaggle"                 # 3 file Kaggle
OPEN_INTRO_DIR = RAW_DIR / "openintro"          # metadata 6 cột mô tả
STD_DIR = DATA_DIR / "standardized"             # Tầng 2: cùng cấu trúc, chưa kiểm tra
CLEANSED_DIR = DATA_DIR / "cleansed"            # Tầng 3: đã kiểm tra chất lượng
CURATED_DIR = DATA_DIR / "curated"              # Tầng 4: sẵn sàng phân tích (TV3)
SANDBOX_DIR = DATA_DIR / "sandbox"              # Tầng 5: nháp từng người

# Các thư mục còn lại (Bảng 2.6)
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
REPORT_FIG_DIR = FIG_DIR / "report"
SLIDE_FIG_DIR = FIG_DIR / "slides"
METRICS_DIR = RESULTS_DIR / "metrics"
GENE_LISTS_DIR = RESULTS_DIR / "gene_lists"
DOCS_DIR = PROJECT_ROOT / "docs"
TESTS_DIR = PROJECT_ROOT / "tests"

# Hằng số chung
RANDOM_SEED = 42
PS_THRESHOLD = 0.3                       # ngưỡng prediction strength (bài báo, note 21)
POSITIVE_CLASS = "AML"                   # lớp dương khi tính sensitivity/specificity

# Bảng màu ALL/AML (style guide, Bảng 4.1): an toàn với người mù màu
PALETTE = {"ALL": "#0072B2", "AML": "#E69F00"}
POS_NEG = {"negative": "#0072B2", "positive": "#E69F00"}

# Danh mục giá trị hợp lệ (dùng cho kiểm tra value-set, mục 2.5)
VALID_SUBTYPE = {"ALL": {"B-ALL", "T-ALL"}, "AML": {"AML"}}
VALID_TISSUE = {"BM", "PB"}
VALID_SOURCE = {"DFCI", "CALGB", "CCG", "St-Jude"}
VALID_SEX = {"M", "F"}  # NaN được phép (gender thiếu trong metadata gốc)

# Khóa thí nghiệm (mục 3.5): giảm số lần lặp khi chạy thử
N_PERMS_EVAL = 100       # số lần hoán vị selection bias
CV_REPEATS = 10          # số lần lặp repeated stratified CV


def ensure_dirs() -> None:
    """Tạo các thư mục dữ liệu/kết quả nếu chưa có."""
    for d in (RAW_DIR, KAGGLE_DIR, OPEN_INTRO_DIR, STD_DIR, CLEANSED_DIR,
              SANDBOX_DIR, METRICS_DIR, REPORT_FIG_DIR, SLIDE_FIG_DIR, GENE_LISTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    for name in ("DATA_DIR", "RAW_DIR", "STD_DIR", "CURATED_DIR", "RESULTS_DIR"):
        print(f"{name:<12} = {globals()[name]}")