"""Thay thế Makefile cho Windows. Cách dùng: python run_all.py [data|experiments|evaluate|figures|all]"""
import subprocess, sys

STEPS = {
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

target = sys.argv[1] if len(sys.argv) > 1 else "all"
for cmd in STEPS[target]:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
