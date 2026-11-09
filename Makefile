# Chạy các bước của pipeline. Trên Windows có thể dùng: python run_all.py
PY ?= python3

fetch:         ## tải dữ liệu gốc vào raw/, đối chiếu MD5 với SOURCES.md
	$(PY) -m src.fetch
standardize:   ## raw/ -> standardized/
	$(PY) -m src.load
quality:       ## standardized/ -> cleansed/ + quality_report.md
	$(PY) -m src.quality
curate:        ## cleansed/ -> curated/
	$(PY) -m src.preprocess
test:          ## dùng đúng python của môi trường, không lấy nhầm pytest hệ thống
	$(PY) -m pytest tests/
data: standardize quality curate test

ingest:        ## nhập dữ liệu theo lô, chất lượng theo lô (TV2, 2.4)
	$(PY) -m src.ingest
warehouse:     ## dựng data/warehouse/golub.duckdb (TV3, 2.3)
	$(PY) -m src.warehouse
evaluate:      ## 4 sơ đồ đánh giá + selection bias, ghi results/metrics/summary_*.csv (TV2, 2.4)
	$(PY) -m src.evaluate
figures:       ## vẽ lại toàn bộ F1–F12
	$(PY) -m src.viz
app:           ## mở dashboard Streamlit (TV6, 2.6)
	$(PY) -m streamlit run app/Home.py

all: data ingest warehouse evaluate figures
.PHONY: fetch standardize quality curate test data ingest warehouse evaluate figures app all
