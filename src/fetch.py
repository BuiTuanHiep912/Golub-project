"""Tải dữ liệu gốc vào tầng raw và đối chiếu MD5 với data/raw/SOURCES.md.

Chạy: `make fetch` (Windows: `python run_all.py fetch`). Không cần curl/unzip hay tài khoản Kaggle.

  - kaggle/  : 3 file CSV, giải nén từ gói zip của Kaggle "crawford/gene-expression"
  - openintro/golub.csv

File đã có và đúng MD5 thì bỏ qua (chạy lại không tải thừa). MD5 lệch thì xóa file vừa tải
và báo lỗi, không để file sai nằm trong tầng raw. MD5 chuẩn chỉ ghi ở một chỗ là bảng
trong SOURCES.md (ADR-005); hàm này đọc bảng đó, không khai lại.
"""
from __future__ import annotations

import io
import re
import sys
import urllib.request
import zipfile

try:
    from src.config import RAW_DIR
    from src.load import md5_file
except ModuleNotFoundError:  # chạy trực tiếp `python src/fetch.py`
    from config import RAW_DIR
    from load import md5_file

KAGGLE_ZIP_URL = "https://www.kaggle.com/api/v1/datasets/download/crawford/gene-expression"
OPENINTRO_URL = "https://www.openintro.org/data/csv/golub.csv"
KAGGLE_FILES = ["data_set_ALL_AML_train.csv", "data_set_ALL_AML_independent.csv", "actual.csv"]


class FetchError(RuntimeError):
    """Tải hỏng hoặc MD5 không khớp SOURCES.md."""


def expected_md5(sources_md=RAW_DIR / "SOURCES.md") -> dict[str, str]:
    """{đường dẫn tương đối trong raw/: MD5} lấy từ các dòng bảng `| kaggle/x.csv | ... | <md5> |`."""
    text = sources_md.read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*([\w./-]+\.csv)\s*\|.*\|\s*([0-9a-f]{32})\s*\|\s*$", text, flags=re.M)
    return dict(rows)


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "golub-project-fetch"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _is_valid(rel: str, expected: dict[str, str]) -> bool:
    path = RAW_DIR / rel
    return path.exists() and md5_file(path) == expected[rel]


def fetch() -> None:
    expected = expected_md5()
    wanted = [f"kaggle/{n}" for n in KAGGLE_FILES] + ["openintro/golub.csv"]
    missing_md5 = [rel for rel in wanted if rel not in expected]
    if missing_md5:
        raise FetchError(f"SOURCES.md thiếu MD5 của {missing_md5}")

    todo = [rel for rel in wanted if not _is_valid(rel, expected)]
    for rel in sorted(set(wanted) - set(todo)):
        print(f"[OK] {rel} đã có, MD5 khớp — bỏ qua")

    if any(rel.startswith("kaggle/") for rel in todo):
        print(f"Tải gói Kaggle: {KAGGLE_ZIP_URL}")
        with zipfile.ZipFile(io.BytesIO(_download(KAGGLE_ZIP_URL))) as zf:
            for name in KAGGLE_FILES:
                (RAW_DIR / "kaggle").mkdir(parents=True, exist_ok=True)
                (RAW_DIR / "kaggle" / name).write_bytes(zf.read(name))
    if "openintro/golub.csv" in todo:
        print(f"Tải OpenIntro: {OPENINTRO_URL}")
        (RAW_DIR / "openintro").mkdir(parents=True, exist_ok=True)
        (RAW_DIR / "openintro" / "golub.csv").write_bytes(_download(OPENINTRO_URL))

    bad = [rel for rel in todo if not _is_valid(rel, expected)]
    for rel in bad:
        (RAW_DIR / rel).unlink(missing_ok=True)
    if bad:
        raise FetchError("MD5 không khớp SOURCES.md (đã xóa file vừa tải): " + ", ".join(bad)
                         + "\n→ nguồn có thể đã đổi bản; báo nhóm và ghi vào docs/decisions.md")
    for rel in todo:
        print(f"[OK] {rel} đã tải, MD5 khớp")
    print("Tầng raw đủ 4 file. Chạy tiếp: make data")


if __name__ == "__main__":
    try:
        fetch()
    except FetchError as err:
        print(f"\n{err}", file=sys.stderr)
        sys.exit(1)
