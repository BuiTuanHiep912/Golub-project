# TV2 – code (mục 1.1 + 2.4)

## Các module
| File | Công việc |
|---|---|
| `config.py` | Đường dẫn repo (root = `Golub_DeTai10/`), `PALETTE`, tập giá trị hợp lệ (`VALID_TISSUE/SOURCE/SEX`), `RANDOM_SEED` |
| `fetch_raw.py` | Sao chép 3 file Kaggle + `dataset_more.csv` → `data/raw/`; ghi MD5 |
| `load.py` | Nạp/ghép theo mục 2.1.2 → `expression/calls/samples/genes.parquet`; ghi `data/raw/SOURCES.md` |
| `quality.py` | Kiểm tra 4 mức (mục 2.5) → `data/cleansed/quality_report.md` |
| `test_load.py` | 15 test bảng 2.4 |
| `evaluate.py` | `evaluate(pipe, X, y, scheme)` với scheme ∈ `original_split`, `loocv`, `nested_cv`, `wrong_cv`; balanced acc, AUC, Wilson CI |
| `selection_bias.py` | Hoán vị nhãn × {CV đúng, sai} → `results/metrics/selection_bias_{summary,permutations}.csv` (F10) |
| `roc_cm.py` | ROC + confusion matrix → `results/figures/report/F11_roc_cm.png` |

## Cách chạy
```bash
# Tại gốc repo Golub_DeTai10/ (Makefile) - chạy trọn gói:
make data          # fetch -> standardized -> cleansed
make test          # pytest test_load.py (15 tests)
make all           # data + test + evaluate + figures

# Hoặc chạy trực tiếp từ code/:
cd code && python3 load.py && python3 quality.py
python3 -m pytest test_load.py
python3 evaluate.py && python3 selection_bias.py && python3 roc_cm.py
```

## Demo nghiệm thu (đã chạy thật trên dữ liệu Golub)
```bash
$ make data && make test
test_load.py ............ 15 passed
```
```python
>>> from load import load_golub       # chạy trong code/; config.py nằm cùng thư mục
>>> X, samples = load_golub(layer='cleansed')
>>> X.shape, samples.shape            # (72, 7129), (72, 7)
>>> samples['class'].value_counts()   # ALL 47, AML 25 (khớp Bảng 2.4)
```
Selection bias (100 hoán vị): hoán vị/đúng 0.510 ≈ 0.50; hoán vị/sai 0.743 → bằng chứng H3.
F11: ROC/CM LogReg 50 gen, AUC thật trên split 38/34 → `results/figures/report/F11_roc_cm.png`.

## Ghi chú
- Code chạy 2 chế độ import: nếu đặt trong bố cục `src/` (`from src.config import …`) hoặc chạy rời từ `code/` (`from config import …`) — dùng `try/except ModuleNotFoundError`. Khi merge nhóm về `src/` không phải sửa import.
- Nguồn metadata là `dataset_more.csv` (ADR-001); `openintro/golub.csv` không tồn tại trong kho nên không dùng.
- Dữ liệu thô nằm ngoài git (dataset/ môn học); tái tạo bằng `make data`.