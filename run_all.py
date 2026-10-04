"""Thay thế Makefile cho Windows. Cách dùng: python run_all.py [data|experiments|figures|all]"""
import subprocess, sys

STEPS = {
    "data": [["python", "-m", "src.load"], ["python", "-m", "src.quality"],
             ["python", "-m", "src.preprocess"], ["pytest", "tests/"]],
    "experiments": [["python", "-m", "src.models"]],
    "figures": [["python", "-m", "src.viz"]],
}
STEPS["all"] = STEPS["data"] + STEPS["experiments"] + STEPS["figures"]

target = sys.argv[1] if len(sys.argv) > 1 else "all"
for cmd in STEPS[target]:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
