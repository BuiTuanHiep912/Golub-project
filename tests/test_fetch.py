"""Kiểm tra src/fetch.py đọc đúng MD5 từ SOURCES.md (không tải mạng)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fetch import KAGGLE_FILES, expected_md5  # noqa: E402


def test_sources_md_has_md5_for_every_raw_file():
    md5 = expected_md5()
    wanted = {f"kaggle/{n}" for n in KAGGLE_FILES} | {"openintro/golub.csv"}
    assert set(md5) == wanted
    assert all(len(v) == 32 for v in md5.values())
