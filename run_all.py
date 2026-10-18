"""Thay thế Makefile cho Windows. Cách dùng: python run_all.py [fetch|data|experiments|evaluate|figures|all] ...

Có thể ghi nhiều bước liền nhau, ví dụ: python run_all.py fetch data
"""
import subprocess, sys

STEPS = {
    "fetch": [
        [sys.executable, "-m", "src.fetch"],
    ],
    "data": [
        [sys.executable, "-m", "src.load"],
        [sys.executable, "-m", "src.quality"],
        [sys.executable, "-m", "src.preprocess"],
        [sys.executable, "-m", "pytest", "tests/"],
    ],
    "experiments": [
        [sys.executable, "-m", "src.models"],
    ],
    "evaluate": [
        [sys.executable, "-m", "src.evaluate"],
    ],
    "figures": [
        [sys.executable, "-m", "src.viz"],
    ],
}
STEPS["all"] = STEPS["data"] + STEPS["experiments"] + STEPS["evaluate"] + STEPS["figures"]

targets = sys.argv[1:] or ["all"]
unknown = [t for t in targets if t not in STEPS]
if unknown:
    sys.exit(f"Bước không hợp lệ: {unknown}. Chọn trong {list(STEPS)}")
for target in targets:
    for cmd in STEPS[target]:
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
