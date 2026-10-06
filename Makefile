# Chạy các bước của pipeline. Trên Windows có thể dùng: python run_all.py
PY ?= python3

standardize:   ## raw/ -> standardized/
	$(PY) -m src.load
quality:       ## standardized/ -> cleansed/ + quality_report.md
	$(PY) -m src.quality
curate:        ## cleansed/ -> curated/
	$(PY) -m src.preprocess
test:          ## dùng đúng python của môi trường, không lấy nhầm pytest hệ thống
	$(PY) -m pytest tests/
data: standardize quality curate test

experiments:   ## chạy thí nghiệm giai thừa, ghi results/metrics/runs.parquet
	$(PY) -m src.models
figures:       ## vẽ lại toàn bộ F1–F12
	$(PY) -m src.viz

all: data experiments figures
.PHONY: standardize quality curate test data experiments figures all
