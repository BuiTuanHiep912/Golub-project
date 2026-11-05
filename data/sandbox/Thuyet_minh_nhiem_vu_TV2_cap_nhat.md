# THUYẾT MINH NHIỆM VỤ TV2 – CẬP NHẬT (bám sát code hiện tại)

> Cập nhật 05/11/2026. Căn cứ: kế hoạch `Ke_hoach_trien_khai_project_Golub.md` (mục 2.1.2, Bảng 2.4,
> 2.9, mục 3.5, 5.2 công việc 1.1, 5.3 công việc 2.4) và code tại:
> - `main` `c039b19`: `src/load.py`, `src/quality.py`, `src/config.py`, `src/preprocess.py` (TV3);
> - nhánh `2.4-evaluate` `a536dbd` (**chờ PR vào `main`**): `src/evaluate.py`, `tests/test_evaluate.py`,
>   `tests/test_golden_evaluate.py`, `notebooks/08_evaluation.ipynb`, `results/`, ADR-013.
>
> Chỗ nào chỉ đúng sau khi merge `2.4-evaluate` thì ghi **[2.4-evaluate]**.

## 1. Nhiệm vụ TV2

TV2 phụ trách (Bảng 5.2 — "người dựng hạ tầng dữ liệu dựng tiếp hạ tầng đánh giá dùng chung"):
- **Giai đoạn 1 (1.1)**: Nạp, ghép dữ liệu, kiểm tra chất lượng (QC) theo 4 mức Bảng 2.9.
- **Giai đoạn 2 (2.4, tuần 5–7, tức 02/11–22/11)**: Khung đánh giá và thí nghiệm *selection bias* (H3);
  hàm đánh giá được TV1 và TV3 dùng từ tuần 5.

## 2. Giai đoạn 1 – 1.1. Nạp, ghép dữ liệu và kiểm tra chất lượng

### 2.1. Đầu vào (tầng `data/raw/`)
| File | Ghi chú |
|---|---|
| `data/raw/kaggle/data_set_ALL_AML_train.csv` | 7131 dòng, probe theo hàng, cặp `k, call` cho bệnh nhân 1..38 (train) |
| `data/raw/kaggle/data_set_ALL_AML_independent.csv` | 7131 dòng, cặp `k, call` cho bệnh nhân 39..72 (test) |
| `data/raw/kaggle/actual.csv` | `patient,cancer` – nguồn nhãn chính `class` (ALL/AML) |
| `data/raw/openintro/golub.csv` | Chỉ dùng **6 cột mô tả**: `Samples,BM.PB,Gender,Source,tissue.mf,cancer`. Thứ tự hàng **không** theo mã bệnh nhân. |

MD5 của cả 4 file ghi trong `data/raw/SOURCES.md`; `make fetch` tải lại và đối chiếu.

### 2.2. `src/load.py` – quy trình ghép (theo mục 2.1.2)
Thứ tự thực thi trong `build()` (`make standardize`):
1. `_load_kaggle_expression()` gọi `_read_kaggle(path)` cho từng file Kaggle
   - Tách cột biểu hiện `k` và cột `k(call)` (P/A/M)
   - **Chuyển vị**: mỗi hàng = 1 bệnh nhân (`sample_id`), mỗi cột = 1 probe
   - Gắn cột `split` theo file (train/test), gộp `train + test`, sắp theo `sample_id` → `expression`, `calls`
   - Báo lỗi nếu thiếu file, probe trùng trong một file, hoặc hai file không cùng danh sách probe
2. `_load_labels()` đọc `actual.csv` → `class` (`ALL`/`AML`)
3. `_load_metadata()` đọc **chỉ 6 cột** từ `openintro/golub.csv` (`usecols`), đổi tên cột, map
   `allB/allT/aml` → `B-ALL/T-ALL/AML`, đặt `sample_id` làm chỉ mục
4. `_assert_same_ids()` — **trước khi ghép**: tập `sample_id` của Kaggle, `actual.csv` và OpenIntro phải
   trùng nhau (left join sẽ lặng lẽ bỏ mẫu chỉ có ở một nguồn); lệch thì liệt kê mã thừa/thiếu và dừng
5. Ghép: bảng mẫu gốc là cột `split` của Kaggle → `.join(labels)` rồi `.join(meta)`, đều là **left join theo
   `sample_id`** với `validate="one_to_one"`; **không bao giờ ghép theo thứ tự dòng**
6. `_assert_post_merge()` — **kiểm tra sau ghép**: đủ 72 mẫu; biểu hiện và bảng mẫu cùng chỉ mục; không thiếu
   `split/class/subtype/tissue/source` (`sex` được phép thiếu); `class` khớp `subtype` 72/72; `tissue.mf` khớp
   `tissue` + `sex` 72/72 (Bảng 2.2), rồi bỏ cột `tissue_mf`
7. Kiểm tra (72, 7129) và không có NaN → ghi 4 parquet → `write_sources()` đối chiếu MD5 (không ghi đè
   `SOURCES.md`, ADR-005) → in các bảng chéo để đối chiếu Bảng 2.4

**Sản phẩm tầng `data/standardized/`:**
| File | Kích thước | Mô tả |
|---|---|---|
| `expression.parquet` | (72, 7129) | Ma trận biểu hiện gốc thang MAS4 (int64), index `sample_id` 1..72 |
| `calls.parquet` | (72, 7129) | P/A/M – **chỉ để kiểm tra, không dùng cho mô hình** |
| `samples.parquet` | (72, 6) | [`split`, `class`, `subtype`, `tissue`, `source`, `sex`] |
| `genes.parquet` | (7129, 3) | [`probe_id`, `description`, `is_control`] (58 probe AFFX) |

`load_golub(layer)` đọc `standardized` / `cleansed` / `curated` (tầng curated là `X_log10.parquet` của TV3);
tên tầng sai thì báo lỗi.

### 2.3. `src/quality.py` – kiểm tra 4 mức (Bảng 2.9)
Thứ tự gọi trong `run_quality()`:
| Hàm | Mức | Nội dung |
|---|---|---|
| `check_value()` | **Value** | Kiểm tra NaN (→ `[FAIL]`), giá trị < 100, > 16000, giá trị âm. Chỉ báo cáo. Threshold `[100,16000]` + log10 thực hiện ở tầng **curated** (TV3). |
| `check_value_set()` | **Value-set** | Probe AFFX (`is_control` = 58), nhiều probe cùng `description` (926 probe thuộc 424 mô tả – ghi chú), kiểm tra danh mục `tissue/source/subtype/sex` theo `src/config.py`. `sex` thiếu 23/72 (không tự điền). Ngoài danh mục → `[FAIL]`. |
| `check_record()` | **Record** | Kích thước (72,7129), duy nhất `sample_id`/probe. Trên log10 sau clip `[100,16000]`, tính tương quan 72x72, phát hiện cặp trùng (r > 0.99), tương quan trung vị với các mẫu khác + trung vị/IQR theo mẫu. Dùng **modified z-score (median/MAD), ngưỡng 3.5** → gắn cờ `qc_outlier`. Kết quả: **sample 21** (r=0.756, z=-4.7). **Chỉ gắn cờ, không xóa**. |
| `check_relation()` | **Relation** | `X.index == samples.index`. `class` (Kaggle) khớp `subtype` (OpenIntro) 72/72. `split × class × source` và `split × tissue` khớp **Bảng 2.4**. `B-ALL+T-ALL=47`, `AML=25`. |

Có `[FAIL]` → vẫn ghi `quality_report.md`, không ghi (và xóa bản cũ của) tầng cleansed, thoát mã 1 để
`make data` dừng (ADR-006).

**Sản phẩm tầng `data/cleansed/`:**
- `expression.parquet` – **bản sao nguyên vẹn** của `standardized/expression.parquet` (không sửa giá trị)
- `samples.parquet` – thêm cột `qc_outlier` (bool)
- `quality_report.md` – báo cáo đầy đủ 4 mức, có `[Tóm tắt]` và `Kết luận: ĐẠT/KHÔNG ĐẠT`

## 3. Giai đoạn 2 – 2.4. Khung đánh giá và thí nghiệm *selection bias*

**Module:** `src/evaluate.py` | **Test:** `tests/test_evaluate.py` (11 test trên `main`, 17 ở **[2.4-evaluate]**),
`tests/test_golden_evaluate.py` (19 testcase golden-file **[2.4-evaluate]**) | **Notebook:** `notebooks/08_evaluation.ipynb`

### 3.1. Hàm chính
| Hàm | Mô tả |
|---|---|
| `evaluate(pipeline, X, y, scheme, seeds=None, split=None, param_grid=None)` | Đánh giá theo 1 trong 4 sơ đồ. Trả về dict gồm `bal_acc_mean`, `bal_acc_sd`, `auc_mean`, `per_fold`, `config`, `predictions` (khóa thứ 6 thêm so với demo kế hoạch — ADR-008). Số fold, seed đọc từ `config/experiment_config.yaml`; mỗi seed là một lần lặp CV. |
| `wilson_ci(n_correct, n, alpha=0.05)` | Khoảng tin cậy Wilson cho tỷ lệ (phù hợp n nhỏ 34 mẫu test); đúng hết → cận trên đúng 1, sai hết → cận dưới đúng 0. |
| `mcnemar_test(y_true, pred_a, pred_b)` | McNemar exact (nhị thức) so sánh 2 mô hình trên **cùng** tập test. `uncertain` được coi là sai. **[2.4-evaluate]** `pd.Series` được ghép theo `sample_id` (không theo thứ tự dòng); khác tập mẫu hoặc khác độ dài → `ValueError`. |
| `confusion_counts(y_true, y_pred)` **[2.4-evaluate]** | Ma trận nhầm lẫn dạng bảng (hàng ALL/AML thật); thêm cột `uncertain` khi mô hình trả về, để tổng luôn bằng số mẫu. Dùng cho F11. |
| `_encode()` | Mã hóa nhãn/dự đoán: `AML`->1, `ALL`->0, mọi giá trị khác (kể cả `uncertain`, `aml`) -> -1 (luôn tính sai). |
| `_metrics()` | Tính `balanced accuracy`, `AUC`, `sensitivity`, `specificity`, `accuracy`, Wilson CI, `n_uncertain`. |
| `_positive_score()` | Điểm của lớp AML cho AUC: cột AML của `predict_proba`, không có thì `decision_function`. Mô hình phải có `classes_`. |
| `_split_at_selector()` | Cắt pipeline cho `wrong_cv` ngay sau bước tên `select` (hoặc bước `SelectorMixin` cuối cùng); không có bước chọn gen → `ValueError`. |
| `_seeded()` | Gán `random_state` của mọi bước bằng seed để chạy lại ra đúng số. |
| `_nested_param_grid()` **[2.4-evaluate]** | Lưới vòng trong của `nested_cv` trong `main()`: `select__k` theo nhân tố B của `experiment_config.yaml`, `clf__C` ∈ {0.01, 0.1, 1, 10} (mục 3.5.2). |
| `selection_bias_experiment(pipe, X, y, n_permutations=100)` | Chạy `{labels: real|permuted} × {cv: correct|wrong}`. Ghi `selection_bias_runs.csv` (qua `main()`). |
| `summarize_selection_bias(runs)` | Gom trung bình ± SD balanced accuracy theo nhóm. |
| `plot_selection_bias(runs)`, `plot_roc_cm(...)` | Vẽ F10, F11 (bản report + slide), ADR-009. |

### 3.2. 4 sơ đồ đánh giá
| Scheme | Mô tả |
|---|---|
| `original_split` | Dùng đúng split `train/test` gốc của Golub (38/34) |
| `loocv` | Leave-One-Out CV trên **38 mẫu train**; mọi bước của pipeline (kể cả chọn gen) fit lại mỗi fold (note 22); chỉ số tính trên dự đoán **gộp** của 38 fold |
| `nested_cv` | Stratified K-fold lặp lại trên 72 mẫu (`outer_folds`, `seeds` trong YAML). **Có `param_grid`** thì thêm vòng trong `GridSearchCV` (`inner_folds`) → nested CV thật; **không có** thì chỉ là repeated CV với tham số cố định ("CV đúng") |
| `wrong_cv` | CV sai: bước chọn gen **và mọi bước trước nó** fit trên **cả 72 mẫu, kể cả fold kiểm tra**, rồi mới CV phần còn lại của pipeline — chỉ để minh họa H3 |

### 3.3. Thí nghiệm *selection bias* (H3)
- **Nhãn thật**: mỗi seed trong YAML (10 seed) chạy một lần CV đúng và một lần CV sai.
- **Nhãn hoán vị**: `n_permutations = N_PERMS_EVAL` (mặc định 100) lần xáo nhãn; mỗi lần chạy cả CV đúng và
  CV sai trên cùng nhãn đã xáo.
- "CV đúng" ở đây là `nested_cv` **không** có lưới (k cố định): H3 chỉ cần so chọn gen trong/ngoài CV.
- Kỳ vọng: nhãn thật thì cả hai cao; nhãn hoán vị thì CV đúng ~ 0.50 còn CV sai cao bất thường → bằng chứng H3.

Kết quả chạy thật (`make evaluate`, pipeline tạm, 100 hoán vị — số liệu không đổi sau ADR-013):

| Nhãn | Cách CV | Balanced accuracy (TB ± SD) | Số lần | Kỳ vọng (kế hoạch §5.3) |
|---|---|---|---|---|
| Thật | Đúng | 0.960 ± 0.014 | 10 | cao |
| Thật | Sai | 0.964 ± 0.006 | 10 | cao, có thể cao hơn CV đúng |
| Hoán vị | Đúng | 0.501 ± 0.065 | 100 | xấp xỉ 0.50 |
| Hoán vị | Sai | **0.807 ± 0.059** | 100 | cao bất thường (bằng chứng H3) |

### 3.4. Sản phẩm đầu ra
| File/Hình | Vị trí |
|---|---|
| `summary_evaluation.csv` (có cột `inner_folds` **[2.4-evaluate]**), `summary_selection_bias.csv`, `selection_bias_runs.csv` | `results/metrics/` |
| `F10_selection_bias_report.png`, `F10_selection_bias_slide.png` | `results/figures/report/`, `results/figures/slides/` |
| `F11_final_model_roc_cm_report.png`, `F11_final_model_roc_cm_slide.png` | `results/figures/report/`, `results/figures/slides/` |

Bảng tổng hợp 4 sơ đồ (pipeline tạm, **[2.4-evaluate]**):

| Sơ đồ | Balanced accuracy | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| `original_split` | 0.964 (1 ước lượng) | 0.993 | 0.929 | 1.000 |
| `loocv` | 0.955 (1 ước lượng) | 1.000 | 0.909 | 1.000 |
| `nested_cv` (vòng trong 5 fold) | 0.968 ± 0.055 | 0.996 | 0.948 | 0.987 |
| `wrong_cv` | 0.964 ± 0.043 | 0.998 | 0.960 | 0.968 |

Tập test 34 mẫu: accuracy 0.971, Wilson 95% [0.851, 0.995]; McNemar (pipeline tạm vs lớp đa số): đúng 33 vs
20/34, p = 0.00024.

### 3.5. Lưu ý triển khai quan trọng
- **Pipeline tạm trong `main()`**: threshold+log10 → StandardScaler → SelectKBest(k=50) → LogisticRegression
  trên tầng cleansed. **Không dùng số liệu và hình của pipeline tạm cho báo cáo cuối.** TV3 đã có
  `src/preprocess.py` (`X_log10.parquet`, `make_preprocess()`), nên bước tiếp theo là chuyển sang
  `load_golub("curated")` + `make_preprocess()`, rồi tới pipeline của TV3 (`src/models.py`, chưa có).
- **Bẫy log hai lần**: chạy `_baseline_pipeline()` trên `X_log10` thì bước log10 kẹp mọi giá trị về 100, cả
  7129 cột thành hằng số, bal_acc = 0.50. Khi chuyển sang curated phải bỏ bước `log10`.
- **Quy ước bước `select`**: `_split_at_selector()` phụ thuộc tên `select` hoặc `SelectorMixin`. Cần thống nhất
  với TV3/TV4; `GeneFilter` của TV3 phải đứng **trước** bước `select`.
- **`uncertain`**: Giữ logic coi `uncertain` là sai; báo cáo kèm `n_uncertain` vì cách tính của bài báo
  (36/36 trong số mẫu chắc chắn) khác cách này.
- **Weighted voting của TV1** (`golub_wv`, chưa có) cần `classes_ = ['ALL', 'AML']` và cột `predict_proba` theo
  đúng thứ tự đó để AUC lấy đúng cột AML.
- **Seeds**: Luôn gán `random_state` qua `_seeded()` để tái lập.
- **Thời gian chạy** **[2.4-evaluate]**: `make evaluate` ~4,5 phút (nested_cv có vòng trong, 24 tổ hợp k × C).
- **Nhóm con** (ghi chú của TV1 trong YAML): nên báo cáo riêng kết quả test trên AML trẻ em (St-Jude + CCG,
  n = 10), St-Jude (n = 8), PB (n = 10) — ghép `predictions` với `samples` theo `sample_id`.

## 4. Nghiệm thu công việc 2.4

### 4.1. Đối chiếu sản phẩm bàn giao (kế hoạch §5.3)
| Sản phẩm / việc | Trạng thái 05/11 | Còn phải làm |
|---|---|---|
| `src/evaluate.py` — 4 sơ đồ, chỉ số, Wilson, McNemar | ✅ bản 1; sửa 3 lỗi ở **[2.4-evaluate]** | Merge `2.4-evaluate` vào `main` qua PR |
| Thí nghiệm selection bias 100 hoán vị × {đúng, sai}; F10 | ✅ (bảng 3.3) | Chạy lại trên tầng curated |
| F11 — ROC + ma trận nhầm lẫn **cho mô hình cuối** | ⬜ đang vẽ cho pipeline tạm | Chờ mô hình cuối + `predictions_test.csv` của TV3 (2.3), gọi lại `plot_roc_cm` |
| Notebook `08_evaluation` | ✅ chạy sạch (0 lỗi) | Chạy lại sau khi đổi sang curated / mô hình cuối |
| Bảng tổng hợp chỉ số (`results/metrics/summary_*.csv`) | ✅ (bảng 3.4, pipeline tạm) | Chạy lại sau khi đổi pipeline |
| Review rò rỉ dữ liệu code TV1, TV3, TV4 | ⬜ | `docs/leakage_review.md`: review được ngay `src/preprocess.py` + `tests/test_no_leakage.py` (TV3, 1.2); `golub_wv.py` (TV1), `features.py` (TV4), `models.py` (TV3) khi có code |
| Mục 3.5 của báo cáo | ⬜ | `docs/report/3.5_danh_gia.md` (theo cách đặt tên các mục đã có trong `docs/report/`) |

### 4.2. Demo nghiệm thu (chạy trước trưởng nhóm)
```bash
make data && make test           # tầng raw → curated; toàn bộ test xanh
make evaluate                    # 4 sơ đồ + 100 hoán vị → results/metrics/summary_*.csv, F10, F11
```
```python
>>> from src.evaluate import evaluate
>>> res = evaluate(pipe, X, y, scheme='nested_cv', seeds=range(10))
>>> res.keys()
dict_keys(['bal_acc_mean', 'bal_acc_sd', 'auc_mean', 'per_fold', 'config', 'predictions'])
```
Kèm bảng selection bias ở mục 3.3 (đã điền số thật). Báo trước trưởng nhóm: `res.keys()` có 6 khóa thay vì
5 như demo kế hoạch, vì thêm `predictions` (ADR-008).

### 4.3. Mốc
- Tuần 5 (02/11–08/11): hàm `evaluate` sẵn cho TV1 (2.1) và TV3 (2.3) — cần merge `2.4-evaluate` vào `main`.
- Tuần 5–7: review rò rỉ khi có code; chuyển sang tầng curated.
- Hết tuần 7 (CN 22/11): F11 cho mô hình cuối, notebook 08, bảng chỉ số và mục 3.5 hoàn chỉnh.
