"""Thay thế Makefile cho Windows. Cách dùng: python run_all.py [fetch|data|ingest|warehouse|evaluate|figures|app|all] ...

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
    "ingest": [
        [sys.executable, "-m", "src.ingest"],
    ],
    "warehouse": [
        [sys.executable, "-m", "src.warehouse"],
    ],
    "evaluate": [
        [sys.executable, "-m", "src.evaluate"],
    ],
    "figures": [
        [sys.executable, "-m", "src.viz"],
    ],
    "app": [
        [sys.executable, "-m", "streamlit", "run", "app/Home.py"],
    ],
}
STEPS["all"] = STEPS["data"] + STEPS["ingest"] + STEPS["warehouse"] + STEPS["evaluate"] + STEPS["figures"]

targets = sys.argv[1:] or ["all"]
unknown = [t for t in targets if t not in STEPS]
if unknown:
    sys.exit(f"Bước không hợp lệ: {unknown}. Chọn trong {list(STEPS)}")
for target in targets:
    for cmd in STEPS[target]:
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
