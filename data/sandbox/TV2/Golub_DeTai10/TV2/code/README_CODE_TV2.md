# TV2 – code (công việc 1.1 + 2.4)

Code của TV2 nằm trong cấu trúc chuẩn của repo, **không** nằm trong thư mục này. Thư mục
`Golub_DeTai10/TV2/code/` chỉ còn file hướng dẫn này (bản làm việc cũ đã gộp hết vào `src/`).

## Các module
| File | Công việc | Nội dung |
|---|---|---|
| `src/config.py` | 1.1 | Đường dẫn các tầng dữ liệu, `RANDOM_SEED`, lớp dương `AML`, `PALETTE`, danh mục giá trị hợp lệ, `N_PERMS_EVAL` |
| `src/load.py` | 1.1 | Ghép 7 bước mục 2.1.2 (theo `sample_id`, left join, `validate="one_to_one"`) → `data/standardized/{expression,calls,samples,genes}.parquet`; đối chiếu MD5 với `SOURCES.md`; `load_golub(layer)` |
| `src/quality.py` | 1.1 | 4 mức Bảng 2.9 → `data/cleansed/` + `quality_report.md`; cờ `qc_outlier` |
| `src/evaluate.py` | 2.4 | `evaluate(pipeline, X, y, scheme, seeds)` với `original_split`, `loocv`, `nested_cv`, `wrong_cv`; `wilson_ci`, `mcnemar_test`, `selection_bias_experiment` |
| `tests/test_load.py` | 1.1 | Toàn bộ Bảng 2.4 trên dữ liệu thật |
| `tests/test_quality.py` | 1.1 | Kiểm tra chất lượng trên dữ liệu tổng hợp (gắn cờ đúng mẫu, FAIL dừng pipeline) |
| `tests/test_evaluate.py` | 2.4 | 4 sơ đồ, Wilson, McNemar, cơ chế selection bias trên dữ liệu tổng hợp |

Nguồn metadata là `data/raw/openintro/golub.csv` (6 cột mô tả, ADR-001); nhãn lấy từ Kaggle
`actual.csv` (ADR-002).

## Cách chạy (tại gốc repo)
```bash
conda env create -f environment.yml && conda activate golub   # hoặc pip install -r requirements.txt
# tải dữ liệu thô: xem lệnh trong data/raw/SOURCES.md, đối chiếu MD5
make standardize   # python -m src.load      raw/ -> standardized/
make quality       # python -m src.quality   standardized/ -> cleansed/ + quality_report.md
make test          # python -m pytest tests/
make evaluate      # python -m src.evaluate  4 sơ đồ + selection bias -> results/metrics/summary_*.csv + hình F10/F11
make data          # standardize + quality + curate (TV3) + test
make figures       # python -m src.viz       vẽ lại toàn bộ F1–F12 (của TV6, chưa có code)
```
Trên Windows: `python run_all.py data` / `python run_all.py evaluate`.

## Demo nghiệm thu
```python
>>> from src.load import load_golub
>>> X, samples = load_golub(layer='cleansed')
>>> X.shape, samples.shape            # (72, 7129), (72, 7)
>>> samples['class'].value_counts()   # ALL 47, AML 25 (khớp Bảng 2.4)
>>> from src.evaluate import evaluate
>>> res = evaluate(pipe, X, samples['class'], scheme='nested_cv', seeds=range(10))
>>> res.keys()   # bal_acc_mean, bal_acc_sd, auc_mean, per_fold, config, predictions
```
Sản phẩm khi chạy `make evaluate` (07/10/2026): `results/metrics/{summary_evaluation,
summary_selection_bias,selection_bias_runs}.csv` + 4 hình
`results/figures/{report,slides}/F10_selection_bias_*.png`, `F11_final_model_roc_cm_*.png`.
Kết quả chạy thật (06–07/10/2026) và trạng thái từng việc: xem
`data/sandbox/TV2/Golub_DeTai10/TV2/Cong_viec_chi_tiet_TV2.md`.

## Ghi chú
- `src/load.py` và `src/quality.py` chạy được cả bằng `python -m src.<module>` lẫn
  `python src/<module>.py` (import thử `src.config` rồi mới tới `config`).
- Dữ liệu (`data/`) và báo cáo chất lượng không đưa lên Git; tạo lại bằng các lệnh trên.
- Đã làm ngày 07/10/2026: hình F10/F11 (bản 1, `make evaluate` tự sinh vào
  `results/figures/{report,slides}/` — ADR-009) và notebook `notebooks/08_evaluation.ipynb`
  (15 cell, chạy thật 0 lỗi). Tổng hợp toàn bộ công việc: `notebooks/TV2_tong_quan_cong_viec.ipynb`.
- Chưa làm: `docs/leakage_review.md` (chờ code TV1/TV3/TV4); F11 bản chính thức và chỉnh
  kiểu dáng F10/F11 theo `src/viz.py` của TV6 (T4 14/10).
