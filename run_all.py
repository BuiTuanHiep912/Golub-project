"""Thay thế Makefile cho Windows. Cách dùng: python run_all.py [data|experiments|evaluate|figures|all]"""
import subprocess, sys

STEPS = {
    "data": [["python", "-m", "src.load"], ["python", "-m", "src.quality"],
             ["python", "-m", "src.preprocess"], ["python", "-m", "pytest", "tests/"]],
    "experiments": [["python", "-m", "src.models"]],
    "evaluate": [["python", "-m", "src.evaluate"]],
    "figures": [["python", "-m", "src.viz"]],
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
