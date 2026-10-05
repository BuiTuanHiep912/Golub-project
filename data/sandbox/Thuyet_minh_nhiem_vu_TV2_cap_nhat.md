# THUYẾT MINH NHIỆM VỤ TV2 – CẬP NHẬT (bám sát code hiện tại)

> Bản cập nhật dựa trên `src/evaluate.py`, `src/quality.py`, `src/config.py` và kế hoạch `Ke_hoach_trien_khai_project_Golub.pdf`.  
> File gốc: `Thuyet_minh_nhiem_vu_TV2.pdf` (đã backup tại `/tmp/Thuyet_minh_nhiem_vu_TV2.pdf.bak`)

## 1. Nhiệm vụ TV2

TV2 phụ trách:
- **Giai đoạn 1 (1.1)**: Nạp, ghép dữ liệu, kiểm tra chất lượng (QC) theo 4 mức Bảng 2.9.
- **Giai đoạn 2 (2.4)**: Khung đánh giá và thí nghiệm *selection bias* (H3).

## 2. Giai đoạn 1 – 1.1. Nạp, ghép dữ liệu và kiểm tra chất lượng

### 2.1. Đầu vào (tầng `data/raw/`)
| File | Ghi chú |
|---|---|
| `data/raw/kaggle/data_set_ALL_AML_train.csv` | 7131 dòng, probe theo hàng, cặp `k, call` cho bệnh nhân 1..38 (train) |
| `data/raw/kaggle/data_set_ALL_AML_independent.csv` | 7131 dòng, cặp `k, call` cho bệnh nhân 39..72 (test) |
| `data/raw/kaggle/actual.csv` | `patient,cancer` – nguồn nhãn chính `class` (ALL/AML) |
| `data/raw/openintro/golub.csv` | Chỉ dùng **6 cột mô tả**: `Samples,BM.PB,Gender,Source,tissue.mf,cancer`. Thứ tự hàng **không** theo mã bệnh nhân. |

### 2.2. `src/load.py` – quy trình ghép (theo mục 2.1.2)
Thứ tự thực thi trong `build()`:
1. `_load_kaggle_expression()` gọi `_read_kaggle(path)` cho từng file Kaggle
   - Tách cột biểu hiện `k` và cột `k(call)` (P/A/M)
   - **Chuyển vị**: mỗi hàng = 1 bệnh nhân (`sample_id`), mỗi cột = 1 probe
   - Gộp `train + test` theo `sample_id` → `expression`, `calls`
2. `_load_labels()` đọc `actual.csv` → `class` (`ALL`/`AML`)
3. `_load_metadata()` đọc **chỉ 6 cột** từ `openintro/golub.csv`, map `allB/allT/aml` → `B-ALL/T-ALL/AML`, ghép theo `sample_id`

**Sản phẩm tầng `data/standardized/`:**
| File | Kích thước | Mô tả |
|---|---|---|
| `expression.parquet` | (72, 7129) | Ma trận biểu hiện gốc thang MAS4 (int64), index `sample_id` 1..72 |
| `calls.parquet` | (72, 7129) | P/A/M – **chỉ để kiểm tra, không dùng cho mô hình** |
| `samples.parquet` | (72, 6) | [`split`, `class`, `subtype`, `tissue`, `source`, `sex`] |
| `genes.parquet` | (7129, 3) | [`probe_id`, `description`, `is_control`] (58 probe AFFX) |

### 2.3. `src/quality.py` – kiểm tra 4 mức (Bảng 2.9)
Thứ tự gọi trong `run_quality()`:
| Hàm | Mức | Nội dung |
|---|---|---|
| `check_value()` | **Value** | Kiểm tra NaN (→ `[FAIL]`), giá trị < 100, > 16000, giá trị âm. Chỉ báo cáo. Threshold `[100,16000]` + log10 thực hiện ở tầng **curated** (TV3). |
| `check_value_set()` | **Value-set** | Probe AFFX (`is_control` = 58), nhiều probe cùng `description` (926 probe thuộc 424 mô tả – ghi chú), kiểm tra danh mục `tissue/source/subtype/sex` theo `src/config.py`. `sex` thiếu 23/72 (không tự điền). Ngoài danh mục → `[FAIL]`. |
| `check_record()` | **Record** | Kích thước (72,7129), duy nhất `sample_id`/probe. Trên log10 sau clip `[100,16000]`, tính tương quan 72x72, phát hiện cặp trùng (r > 0.99), tương quan trung vị với các mẫu khác + trung vị/IQR theo mẫu. Dùng **modified z-score (median/MAD), ngưỡng 3.5** → gắn cờ `qc_outlier`. Kết quả: **sample 21** (r=0.756, z=-4.7). **Chỉ gắn cờ, không xóa**. |
| `check_relation()` | **Relation** | `X.index == samples.index`. `class` (Kaggle) khớp `subtype` (OpenIntro) 72/72. `split × class × source` và `split × tissue` khớp **Bảng 2.4**. `B-ALL+T-ALL=47`, `AML=25`. |

**Sản phẩm tầng `data/cleansed/`:**
- `expression.parquet` – **bản sao nguyên vẹn** của `standardized/expression.parquet` (không sửa giá trị)
- `samples.parquet` – thêm cột `qc_outlier` (bool)
- `quality_report.md` – báo cáo đầy đủ 4 mức, có `[Tóm tắt]` và `Kết luận: ĐẠT/KHÔNG ĐẠT`

## 3. Giai đoạn 2 – 2.4. Khung đánh giá và thí nghiệm *selection bias*

**Module:** `src/evaluate.py` | **Test:** `tests/test_evaluate.py` | **Notebook:** `notebooks/08_evaluation.ipynb`

### 3.1. Hàm chính
| Hàm | Mô tả |
|---|---|
| `evaluate(pipeline, X, y, scheme, seeds=..., split=...)` | Thực hiện đánh giá theo 1 trong 4 sơ đồ. Trả về dict gồm `bal_acc_mean`, `bal_acc_sd`, `auc_mean`, `per_fold`, `config`, `predictions`. |
| `wilson_ci(n_correct, n, alpha=0.05)` | Khoảng tin cậy Wilson cho tỷ lệ (phù hợp n nhỏ 34 mẫu test). |
| `mcnemar_test(y_true, pred_a, pred_b)` | McNemar exact test so sánh 2 mô hình trên **cùng** tập test. `uncertain` được coi là sai. |
| `_encode()` | Mã hóa nhãn/dự đoán: `AML`->1, `ALL`->0, `uncertain`-> -1 (luôn tính sai). |
| `_metrics()` | Tính `balanced accuracy`, `AUC`, `sensitivity`, `specificity`, `accuracy`, Wilson CI, `n_uncertain`. |
| `selection_bias_experiment(pipe, X, y, n_permutations=100)` | Chạy `{labels: real|permuted} × {cv: correct|wrong}` với hoán vị nhãn. Ghi `selection_bias_runs.csv`. |
| `summarize_selection_bias(runs)` | Gom trung bình ± SD balanced accuracy theo nhóm. |

### 3.2. 4 sơ đồ đánh giá
| Scheme | Mô tả |
|---|---|
| `original_split` | Dùng đúng split `train/test` gốc của Golub (38/34) |
| `loocv` | Leave-One-Out CV trên toàn bộ 72 mẫu |
| `nested_cv` | **Nested CV** (outer/inner theo `config/experiment_config.yaml`) |
| `wrong_cv` | CV sai: **chọn biến trên toàn bộ fold** (fit selector trên train+validation) trước khi chia CV |

### 3.3. Thí nghiệm *selection bias* (H3)
- `n_permutations = N_PERMS_EVAL` (mặc định 100)
- Với mỗi hoán vị: chạy `CV đúng` và `CV sai` trên nhãn **hoán vị ngẫu nhiên**
- Kỳ vọng: `CV đúng + nhãn thật` cao, `CV đúng + nhãn hoán vị` ~ 0.50, `CV sai` cao bất thường ngay cả với nhãn hoán vị → bằng chứng H3

### 3.4. Sản phẩm đầu ra
| File/Hình | Vị trí |
|---|---|
| `summary_evaluation.csv`, `summary_selection_bias.csv`, `selection_bias_runs.csv` | `results/metrics/` |
| `F10_selection_bias_report.png`, `F10_selection_bias_slide.png` | `results/figures/report/`, `results/figures/slides/` |
| `F11_final_model_roc_cm_report.png`, `F11_final_model_roc_cm_slide.png` | `results/figures/report/`, `results/figures/slides/` |

### 3.5. Lưu ý triển khai quan trọng
- **Pipeline tạm trong `main()`**: Hiện tại `src/evaluate.py` dựng pipeline tạm để demo. Khi tích hợp chính thức, **phải thay bằng pipeline thật** từ TV3. **Không dùng pipeline tạm cho báo cáo cuối.**
- **Quy ước bước `select`**: `_split_at_selector()` phụ thuộc tên `select` hoặc `SelectorMixin`. Cần thống nhất với TV3/TV4.
- **`uncertain`**: Giữ logic coi `uncertain` là sai.
- **Seeds**: Luôn gán `random_state` qua `_seeded()` để tái lập.
