# Makefile golub-project (Bảng 2.6) - chạy tại gốc repo Golub_DeTai10.
#   make all : cleansed + test (Bảng 2.4) + đánh giá + hình F10/F11
# Các module đang ở TV2/code; khi merge vào src/ chỉ cần đổi TV2/code -> src
# và PYTHONPATH (các module đã viết 2 chế độ `from src.config` / `from config`).

PY       ?= python3
CODEDIR  := TV2/code

.PHONY: all data fetch standardized cleansed quality test evaluate selection-bias roc-cm figures clean clean_data

all: data test evaluate figures

data: cleansed

fetch:
	cd $(CODEDIR) && $(PY) fetch_raw.py

standardized: fetch
	cd $(CODEDIR) && $(PY) load.py

cleansed: standardized
	cd $(CODEDIR) && $(PY) quality.py

quality: cleansed   # alias

test:
	cd $(CODEDIR) && $(PY) -m pytest test_load.py -v

evaluate:
	cd $(CODEDIR) && $(PY) evaluate.py

selection-bias:
	cd $(CODEDIR) && $(PY) selection_bias.py

roc-cm:
	cd $(CODEDIR) && $(PY) roc_cm.py

figures: selection-bias roc-cm

clean_data:
	@echo "Xoá data/ và results/ (tạo lại bằng make all)"
	rm -rf data results

clean: clean_data
	find . -name __pycache__ -type d -prune -exec rm -rf {} +