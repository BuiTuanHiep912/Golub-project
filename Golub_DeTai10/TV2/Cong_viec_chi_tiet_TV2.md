# TV2 – Công việc chi tiết (theo Bảng phân công 5.1)

> Nguồn: `Ke_hoach_trien_khai_project_Golub.pdf`, mục 5.1 (Bảng 5.2) + mô tả chi tiết mục 5.2
> (công việc 1.1) và 5.3 (công việc 2.4). Cập nhật 06/10/2026 trên nhánh `tv2-review-fixes`
> theo `Golub_DeTai10/TV2/Danh_gia_TV2.md`.

TV2 phụ trách **2 công việc kỹ thuật**, mỗi giai đoạn một việc (trưởng nhóm là TV1):

| Giai đoạn | Công việc | Mạch liên kết |
|---|---|---|
| GĐ1 (tuần 1–4) | **1.1. Nạp, ghép dữ liệu và kiểm tra chất lượng** (deadline T7 10/10) | Người dựng hạ tầng dữ liệu dựng tiếp hạ tầng đánh giá dùng chung |
| GĐ2 (tuần 5–7) | **2.4. Khung đánh giá và thí nghiệm selection bias** | |

Mọi code nằm trong cấu trúc chuẩn của repo (`src/`, `tests/`, `docs/`, `Makefile`).
Thư mục `Golub_DeTai10/TV2/` chỉ giữ tài liệu của TV2: file này, `Danh_gia_TV2.md`,
`code/README_CODE_TV2.md` và script `tao_thuyet_minh_pdf.py` sinh `Thuyet_minh_nhiem_vu_TV2.pdf`.

---

## Phần A. GĐ1 – Công việc 1.1. Nạp, ghép dữ liệu và kiểm tra chất lượng

### Mục tiêu
Đưa bộ dữ liệu Golub từ Kaggle và metadata OpenIntro vào dạng dùng được cho cả nhóm, đáng tin cậy và tái lập được.

### Công việc cụ thể
| # | Việc | Trạng thái |
|---|---|---|
| 1 | Tải 3 file Kaggle và `golub.csv` của OpenIntro vào `data/raw/{kaggle,openintro}/`; `data/raw/SOURCES.md` ghi link, ngày tải, MD5 (ADR-005) | ✅ MD5 khớp 4/4 |
| 2 | Dựng repo, cấu trúc thư mục (mục 2.2), `environment.yml` + `requirements.txt` **cố định phiên bản** (ADR-007), `Makefile`, `run_all.py`, `src/config.py` | ✅ |
| 3 | `src/load.py` theo 7 bước mục 2.1.2: left join vào bảng mẫu Kaggle với `validate="one_to_one"`, so tập `sample_id` của 3 nguồn, kiểm tra sau ghép (đủ metadata, class khớp subtype, `tissue.mf` khớp tissue + sex) → 4 parquet; `load_golub(layer)` cho `standardized` / `cleansed` / `curated`, tên tầng sai thì báo lỗi | ✅ |
| 4 | `src/quality.py` kiểm tra **đúng 4 mức Bảng 2.9** (Value, Value-set, Record, Relation) → tầng `data/cleansed/` + `quality_report.md`; có FAIL thì dừng pipeline (ADR-006) | ✅ |
| 5 | `tests/test_load.py` (toàn bộ Bảng 2.4, báo lỗi khi chưa có dữ liệu) + `tests/test_quality.py` (dữ liệu tổng hợp); `docs/data_dictionary.md`, `docs/decisions.md` | ✅ |
| 6 | Viết mục 2.1–2.5, 2.7 của báo cáo (`docs/report/`) | ⬜ chưa làm |
| 7 | Giao `load_golub()` bản tạm cho TV3 (CN 11/10); giao `samples.parquet` chính thức cho TV1 (CN 18/10) | ⬜ đến hạn |
| 8 | Ghi chú bàn giao `docs/handover/1.1.md`, trình bày 10 phút + demo (CN 01/11) | ⬜ đến hạn |

### Sản phẩm bàn giao
- `src/config.py`, `src/load.py`, `src/quality.py`, `tests/test_load.py`, `tests/test_quality.py`
- `data/standardized/`, `data/cleansed/` (sinh tự động, không commit), `data/raw/SOURCES.md`
- `docs/data_dictionary.md`, `docs/decisions.md`; mục 2.1–2.5, 2.7 của báo cáo

### Demo nghiệm thu (đã chạy thật ngày 06/10/2026, Python 3.11, đúng bộ phiên bản đã ghim)
```bash
$ make standardize      # python -m src.load
[OK] Kiểm tra sau ghép: 72 mẫu, đủ metadata, class khớp subtype 72/72, tissue.mf khớp tissue + sex 72/72
[OK] MD5 trong SOURCES.md khớp 4 file raw
$ make quality          # python -m src.quality  ->  Kết luận: ĐẠT
$ make test             # python -m pytest tests/
33 passed, 1 skipped    # 1 skipped = test_no_leakage.py của TV3 (công việc 1.2)
```
```python
>>> from src.load import load_golub
>>> X, samples = load_golub(layer='cleansed')
>>> X.shape, samples.shape
((72, 7129), (72, 7))        # 6 cột mô tả + cột cờ qc_outlier
>>> samples['class'].value_counts()
ALL 47
AML 25
>>> pd.crosstab(samples['split'], samples['class'])
class ALL AML
split
test  20  14
train 27  11
>>> pd.crosstab([samples['split'], samples['class']], samples['source'])
source      CALGB CCG DFCI St-Jude
split class
test  ALL     0   0   17       3
      AML     4   5    0       5
train ALL     0   0   27       0
      AML    11   0    0       0
```
Trích `data/cleansed/quality_report.md` (phần tóm tắt, đúng định dạng demo trang 20):
```text
[PASS] 72 mẫu, 7129 probe, không trùng sample_id
[PASS] Nhãn Kaggle khớp subtype OpenIntro: 72/72
[INFO] Số giá trị < 100: 241357 Số giá trị > 16000: 4259 Probe AFFX: 58
[INFO] Mẫu gắn cờ qc_outlier: 1 (lý do: xem mức Record — mẫu [21])
```
Chi tiết đáng chú ý trong báo cáo:
- **Mẫu 21** (train, ALL, DFCI) bị gắn cờ: tương quan trung vị với các mẫu khác 0.756 so
  với trung vị chung 0.832 (modified z = −4.7). Chỉ gắn cờ, không xóa (ADR-004); TV1/TV6 đối
  chiếu trên boxplot F3 và PCA.
- 926 probe dùng chung 424 mô tả gen → khi đếm "số gen" hoặc diễn giải phải gộp theo mô tả.
- 147 174 giá trị âm, thiếu `sex` 23/72 (thiếu ở nguồn, không tự điền).

---

## Phần B. GĐ2 – Công việc 2.4. Khung đánh giá và thí nghiệm selection bias (tuần 5–7)

### Mục tiêu
Đảm bảo mọi con số hiệu năng là trung thực và chứng minh bằng thực nghiệm tác hại của đánh giá sai (giả thuyết H3).

### Công việc cụ thể
| # | Việc | Trạng thái |
|---|---|---|
| 1 | `src/evaluate.py`: `evaluate(pipeline, X, y, scheme, seeds)` cho 4 sơ đồ `original_split` (38/34), `loocv` (38 mẫu train, dự đoán gộp), `nested_cv`, `wrong_cv`; số fold và seed đọc từ `config/experiment_config.yaml` (TV1) | ✅ bản 1 (ADR-008) |
| 2 | Balanced accuracy, AUC, **sensitivity, specificity** (AML dương), khoảng tin cậy **Wilson**; kiểm định **McNemar** cho cặp mô hình trên tập test | ✅ |
| 3 | Thí nghiệm selection bias: 100 lần hoán vị nhãn × {CV đúng, CV sai} + nhãn thật × {đúng, sai} → số liệu `results/metrics/` | ✅ số liệu |
| 3b | Vẽ **F10** (violin/strip plot) bằng `src/viz.py` của TV6; notebook `08_evaluation.ipynb` | ⬜ chờ `viz.py` (TV6 giao T4 14/10) |
| 4 | **F11**: ROC + ma trận nhầm lẫn cho mô hình cuối | ⬜ chờ mô hình cuối (TV3, 2.3) |
| 5 | Review code TV1/TV3/TV4 phát hiện rò rỉ dữ liệu → `docs/leakage_review.md` | ⬜ chưa có code để review (`golub_wv.py`, `preprocess.py`, `features.py`, `models.py` còn trống) |
| 6 | Mục 3.5 của báo cáo | ⬜ |

`evaluate.py` không phải chờ TV3: hàm nhận bất kỳ pipeline nào. Trong lúc tầng curated chưa có,
`python -m src.evaluate` dùng pipeline tạm threshold+log10 → chuẩn hóa → 50 gen (f_classif) →
logistic trên tầng cleansed; khi TV3 giao `X_log10.parquet` thì đổi sang `load_golub("curated")`.

### Sản phẩm bàn giao (theo Bảng 2.6 — không tạo module riêng ngoài cấu trúc)
- `src/evaluate.py` (gồm cả thí nghiệm selection bias), `tests/test_evaluate.py`
- `notebooks/08_evaluation.ipynb`; hình `results/figures/report/F10_selection_bias_report.png`,
  `results/figures/slides/F10_selection_bias_slide.png`, `F11_final_model_roc_cm_{report,slide}.png`
- `results/metrics/summary_evaluation.csv`, `summary_selection_bias.csv`, `selection_bias_runs.csv`
- `docs/leakage_review.md`; mục 3.5 của báo cáo

### Demo nghiệm thu (đã chạy thật ngày 06/10/2026 — `make evaluate`, ~45 giây)
```python
>>> from src.evaluate import evaluate
>>> res = evaluate(pipe, X, y, scheme='nested_cv', seeds=range(10))
>>> res.keys()
dict_keys(['bal_acc_mean', 'bal_acc_sd', 'auc_mean', 'per_fold', 'config', 'predictions'])
```
`predictions` (dự đoán theo mẫu) là khóa thêm so với demo của kế hoạch vì McNemar, ROC và phân
tích lỗi của TV5 cần nó (ADR-008).

```text
$ make evaluate
original_split  bal_acc=0.964 (1 ước lượng) auc=0.993  sens=0.929  spec=1.000  (1 dòng per_fold)
loocv           bal_acc=0.955 (1 ước lượng) auc=1.000  sens=0.909  spec=1.000  (1 dòng per_fold)
nested_cv       bal_acc=0.960 ±0.049        auc=0.997  sens=0.956  spec=0.963  (50 dòng per_fold)
wrong_cv        bal_acc=0.964 ±0.043        auc=0.998  sens=0.960  spec=0.968  (50 dòng per_fold)

Tập test 34 mẫu: accuracy=0.971, Wilson 95% [0.851, 0.995]
McNemar (LogReg 50 gen vs lớp đa số) trên test: đúng 33 vs 20/34, p = 0.00024
```

Bảng selection bias (100 hoán vị, dữ liệu thật, pipeline tạm ở trên):

| Nhãn | Cách CV | Balanced accuracy (TB ± SD) | Số lần | Kỳ vọng (kế hoạch) | Kết luận |
|---|---|---|---|---|---|
| Thật | Đúng | 0.960 ± 0.014 | 10 | cao | đạt |
| Thật | Sai | 0.964 ± 0.006 | 10 | cao, có thể cao hơn CV đúng | đạt |
| Hoán vị | Đúng | 0.501 ± 0.065 | 100 | xấp xỉ 0.50 | đạt |
| Hoán vị | Sai | 0.807 ± 0.059 | 100 | cao bất thường | **bằng chứng H3** |

Với nhãn thật, CV sai chỉ cao hơn CV đúng chút ít vì tín hiệu ALL/AML rất mạnh; selection bias
lộ rõ ở nhãn hoán vị: không có quan hệ thật nào mà CV sai vẫn báo ~0.81.

---

## Hướng triển khai code

| File | Nội dung | Trạng thái |
|---|---|---|
| `src/config.py` | đường dẫn, seed, lớp dương, `PALETTE`, danh mục giá trị hợp lệ | ✅ |
| `src/load.py` | ghép 7 bước mục 2.1.2, 4 parquet, đối chiếu MD5, `load_golub()` | ✅ |
| `src/quality.py` | 4 mức Bảng 2.9, `quality_report.md`, cờ `qc_outlier` | ✅ |
| `src/evaluate.py` | 4 sơ đồ, chỉ số, Wilson, McNemar, selection bias | ✅ bản 1 |
| `tests/test_load.py`, `test_quality.py`, `test_evaluate.py` | Bảng 2.4; QC trên dữ liệu tổng hợp; khung đánh giá | ✅ |
| `docs/data_dictionary.md`, `docs/decisions.md` | ADR-001…008 | ✅ |
| `notebooks/08_evaluation.ipynb`, F10, F11, `docs/leakage_review.md` | | ⬜ |
