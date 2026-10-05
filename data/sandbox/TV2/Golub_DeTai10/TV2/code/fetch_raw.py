"""
TV2 - Makefile `make data` -> sao chép dữ liệu thô từ kho môn học sang data/raw/.

Kho môn học (KHÔNG CÓ TRÊN GIT): `Quản trị dữ liệu và trực quan hóa/dataset/`
  - data_set_ALL_AML_train.csv / data_set_ALL_AML_independent.csv / actual.csv
  - dataset_more.csv -> golub_metadata.csv (metadata 6 cột, ADR-001)

Nếu đường dẫn đặt tại nơi khác, truyền qua biến môi trường GOLUB_DATASET_DIR.
Chỉ sao chép + tính MD5; không sửa nội dung file.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

try:
    from src.config import RAW_DIR, KAGGLE_DIR, OPEN_INTRO_DIR, STD_DIR
    from src.load import md5_file, write_sources
except ModuleNotFoundError:
    from config import RAW_DIR, KAGGLE_DIR, OPEN_INTRO_DIR, STD_DIR
    from load import md5_file, write_sources


CANDIDATE_ROOTS = [
    Path(os.environ.get("GOLUB_DATASET_DIR", "")).expanduser() if os.environ.get("GOLUB_DATASET_DIR") else None,
    Path(__file__).resolve().parents[4] / "dataset",      # code -> TV2 -> Golub_DeTai10 -> subject -> dataset
    Path(__file__).resolve().parents[3] / "dataset",       # dự phòng
]

FILES_KAGGLE = ["data_set_ALL_AML_train.csv", "data_set_ALL_AML_independent.csv", "actual.csv"]
META_SOURCE = "dataset_more.csv"


def _find_dataset() -> Path:
    for cand in CANDIDATE_ROOTS:
        if cand and (cand / "actual.csv").exists() and (cand / FILES_KAGGLE[0]).exists():
            return cand
    raise FileNotFoundError(
        "Không tìm thấy thư mục dataset (actual.csv, data_set_ALL_AML_*). "
        "Truyền GOLUB_DATASET_DIR=/duong/dan/dataset"
    )


def _find_meta() -> Path:
    subj = Path(__file__).resolve().parents[3]  # gốc môn học
    for cand in [subj / META_SOURCE, _find_dataset() / META_SOURCE]:
        if cand and cand.exists():
            return cand
    raise FileNotFoundError(
        f"Không tìm thấy {META_SOURCE} (gốc môn học hoặc dataset/). "
        "Truyền GOLUB_DATASET_DIR=/duong/dan/chua/dataset_more.csv"
    )


def fetch() -> None:
    root = _find_dataset()
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    OPEN_INTRO_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name in FILES_KAGGLE:
        shutil.copy(root / name, KAGGLE_DIR / name)
    shutil.copy(_find_meta(), OPEN_INTRO_DIR / "golub_metadata.csv")

    print(f"Đã chép 3 file Kaggle + metadata từ: {root}")
    for f in sorted(RAW_DIR.rglob("*.*")):
        if f.is_file() and f.suffix in {".csv", ".txt", ".md"}:
            print(f"  {f.relative_to(RAW_DIR)}  MD5={md5_file(f)}")
    write_sources()
    print("Đã ghi: data/raw/SOURCES.md")


if __name__ == "__main__":
    fetch()