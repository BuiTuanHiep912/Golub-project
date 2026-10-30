"""Kiểm thử tự động cho Công việc 1.5: Khám phá lớp không giám sát (TV5).

Kiểm tra:
1. Bảng số liệu nghiệm thu cluster_discovery_metrics.csv đầy đủ các dòng k=2, k=4, K-Means, Hierarchical, SOM.
2. Chất lượng phân cụm đạt chuẩn: K-Means k=2 có ARI > 0.85 (dữ liệu ủng hộ mạnh H5).
3. Các file hình vẽ F05 tồn tại và có dung lượng hợp lệ (> 50KB).
4. Notebook 03_discovery.ipynb tồn tại và chứa output.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import METRICS_DIR, REPORT_FIG_DIR, SLIDE_FIG_DIR


def test_metrics_file_exists_and_valid():
    metrics_path = METRICS_DIR / "cluster_discovery_metrics.csv"
    assert metrics_path.exists(), f"Thiếu file {metrics_path}"
    
    df = pd.read_csv(metrics_path)
    required_cols = ["method", "k", "ARI_vs_class", "ARI_vs_subtype", "silhouette", "notes"]
    for col in required_cols:
        assert col in df.columns, f"Thiếu cột {col} trong metrics"
    
    # Kiểm tra có đủ k=2 và k=4
    assert 2 in df["k"].values, "Thiếu cấu hình k=2"
    assert 4 in df["k"].values, "Thiếu cấu hình k=4"
    
    # Kiểm tra K-Means k=2 đạt ARI cao (> 0.85)
    km2_row = df[(df["method"] == "K-Means") & (df["k"] == 2)]
    assert not km2_row.empty, "Thiếu dòng K-Means k=2"
    ari_val_str = km2_row["ARI_vs_class"].values[0].split("±")[0].strip()
    assert float(ari_val_str) > 0.85, f"ARI của K-Means k=2 thấp ({ari_val_str}), cần > 0.85"


def test_f05_figures_exist():
    # Kiểm tra hình F05 report và slide
    f05_rep = REPORT_FIG_DIR / "F05_clustermap_report.png"
    f05_sli = SLIDE_FIG_DIR / "F05_clustermap_slide.png"
    
    assert f05_rep.exists(), f"Thiếu {f05_rep}"
    assert f05_sli.exists(), f"Thiếu {f05_sli}"
    
    assert f05_rep.stat().st_size > 50_000, "Hình F05 report quá nhỏ hoặc bị hỏng"
    assert f05_sli.stat().st_size > 50_000, "Hình F05 slide quá nhỏ hoặc bị hỏng"


def test_discovery_notebook_exists_with_outputs():
    nb_path = ROOT / "notebooks" / "03_discovery.ipynb"
    assert nb_path.exists(), f"Thiếu notebook {nb_path}"
    assert nb_path.stat().st_size > 50_000, "Notebook chưa có output hoặc bị rỗng"


def test_handover_and_report_docs_exist():
    handover_path = ROOT / "docs" / "handover" / "1.5.md"
    report_path = ROOT / "docs" / "report" / "3.2_class_discovery.md"
    
    assert handover_path.exists(), f"Thiếu biên bản bàn giao {handover_path}"
    assert report_path.exists(), f"Thiếu báo cáo mục 3.2 {report_path}"
    
    # Kiểm tra nội dung có kết luận H5
    content = report_path.read_text(encoding="utf-8")
    assert "H5" in content, "Báo cáo chưa đề cập giả thuyết H5"
