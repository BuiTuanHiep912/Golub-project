# THUYẾT MINH NHIỆM VỤ TV2 – CẬP NHẬT (bám sát code hiện tại)

> Cập nhật 12/11/2026 (bản trước: 05/11/2026). Căn cứ: kế hoạch `Ke_hoach_trien_khai_project_Golub_.md`
> (mục 2.1.2, Bảng 2.4, 2.9, mục 3.5, 5.2 công việc 1.1, 5.3 công việc 2.4), **ADR-014** (đổi hướng giai
> đoạn 2, 09/11/2026), lịch giai đoạn 2 trong `moretask.jpg` (commit `e86bc08`), và code tại:
> - `origin/main` `d5c9824`: khung giai đoạn 2 (ADR-014), `src/load.py`, `src/quality.py`, `src/preprocess.py` (TV3);
> - `2.4-evaluate` `c41b373` (đã rebase lên `main`, **đã mở PR**): `src/evaluate.py` sửa 3 lỗi (ADR-013);
> - **nhánh mới, xếp chồng, chưa push** (mỗi nhánh tạo từ nhánh cũ tương ứng rồi rebase lên nhánh trước):
>   `2.4-ingest-v2` `4c7722a` ← `2.4-f8-f10-demo-v2` `ac24fe0` ← `2.4-report-draft-v2` `1bac4ec`.
>
> Chỗ nào chỉ đúng sau khi merge một nhánh thì ghi tên nhánh trong ngoặc vuông, ví dụ **[2.4-ingest-v2]**.

## 1. Nhiệm vụ TV2

TV2 phụ trách (Bảng 5.2 — "người dựng hạ tầng dữ liệu dựng tiếp hạ tầng đánh giá dùng chung"):
- **Giai đoạn 1 (1.1)**: Nạp, ghép dữ liệu, kiểm tra chất lượng (QC) theo 4 mức Bảng 2.9 — ✅ xong.
- **Giai đoạn 2 (2.4)** — kế hoạch gốc: khung đánh giá và thí nghiệm *selection bias* (H3).
  **Từ 09/11/2026 (ADR-014)** đổi thành **"Nhập theo lô, chất lượng"**, vẫn giữ thí nghiệm selection bias.
  ADR-014 cũng bỏ H2, H4 và thí nghiệm giai thừa, và đặt quy ước: **38 mẫu huấn luyện là dữ liệu tham chiếu;
  34 mẫu kiểm tra đóng vai bệnh nhân mới, không dùng để huấn luyện, chọn gen hay tính tham số**.

Lịch giai đoạn 2 của TV2 (`moretask.jpg`) và trạng thái 12/11:

| Tuần | Việc | Trạng thái 12/11 |
|---|---|---|
| 1 (đến 15/11) | Merge `2.4-evaluate` (hạn 11/11) | ✅ đã rebase lên `main`, đã mở PR — chờ duyệt |
| 1 (đến 15/11) | `src/ingest.py` nhập 4 lô, bảng chất lượng, `validate_new_sample()` | ✅ **[2.4-ingest-v2]** |
| 2 (đến 22/11) | F8, F10 bản cuối; bộ file lỗi mẫu để demo | ✅ **[2.4-f8-f10-demo-v2]** (F8: xem giả định ở 5.1) |
| 3 (đến 29/11) | Nháp mục báo cáo | ✅ nháp **[2.4-report-draft-v2]** — chờ dàn ý cuối |
| 4 (đến 06/12) | Sửa theo góp ý | ⬜ |

## 2. Giai đoạn 1 – 1.1. Nạp, ghép dữ liệu và kiểm tra chất lượng

### 2.1. Đầu vào (tầng `data/raw/`)
| File | Ghi chú |
|---|---|
| `data/raw/kaggle/data_set_ALL_AML_train.csv` | 7131 dòng, probe theo hàng, cặp `k, call` cho bệnh nhân 1..38 (train) |
| `data/raw/kaggle/data_set_ALL_AML_independent.csv` | 7131 dòng, cặp `k, call` cho bệnh nhân 39..72 (test) |
| `data/raw/kaggle/actual.csv` | `patient,cancer` – nguồn nhãn chính `class` (ALL/AML) |
| `data/raw/openintro/golub.csv` | Chỉ dùng **6 cột mô tả**: `Samples,BM.PB,Gender,Source,tissue.mf,cancer`. Thứ tự hàng **không** theo mã bệnh nhân. |

MD5 của cả 4 file ghi trong `data/raw/SOURCES.md`; `make fetch` tải lại và đối chiếu (chạy lại 12/11: 4/4 khớp).

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
| `calls.parquet` | (72, 7129) | P/A/M – **chỉ để kiểm tra, không dùng cho mô hình** (từ 2.4 mới: tính tỷ lệ call P theo lô) |
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

**[2.4-ingest-v2]** Quy tắc gắn cờ theo mẫu được tách thành hàm dùng chung `sample_correlation()` /
`sample_qc()` để `src/ingest.py` gắn cờ đúng như tầng cleansed. Đã đối chiếu: `quality_report.md` và cột
`qc_outlier` của `cleansed/samples.parquet` **giống hệt** trước khi tách (ADR-030).

Có `[FAIL]` → vẫn ghi `quality_report.md`, không ghi (và xóa bản cũ của) tầng cleansed, thoát mã 1 để
`make data` dừng (ADR-006).

**Sản phẩm tầng `data/cleansed/`:**
- `expression.parquet` – **bản sao nguyên vẹn** của `standardized/expression.parquet` (không sửa giá trị)
- `samples.parquet` – thêm cột `qc_outlier` (bool)
- `quality_report.md` – báo cáo đầy đủ 4 mức, có `[Tóm tắt]` và `Kết luận: ĐẠT/KHÔNG ĐẠT`
- **[2.4-ingest-v2]** `batches.parquet`, `qc_samples.parquet`, `ingest_log.csv` — đầu ra của `make ingest` (mục 4)

## 3. Giai đoạn 2 – 2.4 (phần giữ lại). Khung đánh giá và thí nghiệm *selection bias*

**Module:** `src/evaluate.py` | **Test:** `tests/test_evaluate.py` (17 test ở `2.4-evaluate`, 19 ở
**[2.4-f8-f10-demo-v2]**), `tests/test_golden_evaluate.py` (19 testcase golden-file) | **Notebook:**
`notebooks/08_evaluation.ipynb`

### 3.1. Hàm chính
| Hàm | Mô tả |
|---|---|
| `evaluate(pipeline, X, y, scheme, seeds=None, split=None, param_grid=None)` | Đánh giá theo 1 trong 4 sơ đồ. Trả về dict gồm `bal_acc_mean`, `bal_acc_sd`, `auc_mean`, `per_fold`, `config`, `predictions` (khóa thứ 6 thêm so với demo kế hoạch — ADR-008). Số fold, seed đọc từ `config/experiment_config.yaml`; mỗi seed là một lần lặp CV. |
| `wilson_ci(n_correct, n, alpha=0.05)` | Khoảng tin cậy Wilson cho tỷ lệ (phù hợp n nhỏ 34 mẫu test); đúng hết → cận trên đúng 1, sai hết → cận dưới đúng 0. |
| `mcnemar_test(y_true, pred_a, pred_b)` | McNemar exact (nhị thức) so sánh 2 mô hình trên **cùng** tập test. `uncertain` được coi là sai. `pd.Series` được ghép theo `sample_id` (không theo thứ tự dòng); khác tập mẫu hoặc khác độ dài → `ValueError` (ADR-013). |
| `confusion_counts(y_true, y_pred)` | Ma trận nhầm lẫn dạng bảng (hàng ALL/AML thật); thêm cột `uncertain` khi mô hình trả về, để tổng luôn bằng số mẫu. Dùng cho F11. |
| `selection_bias_experiment(pipe, X, y, n_permutations=100)` | Chạy `{labels: real|permuted} × {cv: correct|wrong}`. Ghi `selection_bias_runs.csv` (qua `main()`). |
| `summarize_selection_bias(runs)` | Gom trung bình ± SD balanced accuracy theo nhóm. |
| `plot_selection_bias(runs, base_dir=None)`, `plot_roc_cm(..., base_dir=None)` | Vẽ F10, F11 (bản report + slide). **[2.4-f8-f10-demo-v2]** vẽ qua `src/viz.py` theo style guide; số lần hoán vị trên hình đếm từ `runs` (ADR-031). |
| `_baseline_pipeline(k=50)` | **[2.4-f8-f10-demo-v2]** Pipeline minh họa trên tầng curated: `make_preprocess()` (TV3) → `SelectKBest` → logistic. Bỏ bước log10 tạm và bộ lọc cảnh báo. |
| `_nested_param_grid()` | Lưới vòng trong của `nested_cv`: `select__k` theo nhân tố B của YAML, `clf__C` ∈ {0.01, 0.1, 1, 10}. |
| `_encode()`, `_metrics()`, `_positive_score()`, `_split_at_selector()`, `_seeded()` | Mã hóa nhãn (`uncertain` → −1, luôn sai), tính chỉ số, điểm lớp AML cho AUC, cắt pipeline tại bước `select` cho `wrong_cv`, gán seed cho mọi `random_state`. |

### 3.2. 4 sơ đồ đánh giá
| Scheme | Mô tả | Dữ liệu trong `make evaluate` **[2.4-f8-f10-demo-v2]** |
|---|---|---|
| `original_split` | Đúng split `train/test` gốc của Golub | học 38 → chấm 34 |
| `loocv` | Leave-One-Out CV trên **38 mẫu train**; mọi bước fit lại mỗi fold (note 22); chỉ số tính trên dự đoán **gộp** | 38 |
| `nested_cv` | Stratified K-fold lặp lại; **có `param_grid`** thì thêm vòng trong `GridSearchCV` → nested CV thật | 38 (trước ADR-031: 72) |
| `wrong_cv` | CV sai: bước chọn gen **và mọi bước trước nó** fit trên **toàn bộ X**, kể cả fold kiểm tra | 38 (trước ADR-031: 72) |

### 3.3. Kết quả `make evaluate` — **[2.4-f8-f10-demo-v2]**, 12/11/2026, tầng curated, 100 hoán vị

| Sơ đồ | Balanced accuracy | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| `original_split` (học 38 → chấm 34) | 0.964 (1 ước lượng) | 0.996 | 0.929 | 1.000 |
| `loocv` (38) | 0.955 (1 ước lượng) | 1.000 | 0.909 | 1.000 |
| `nested_cv` (38, vòng trong 5 fold) | 0.922 ± 0.137 | 0.998 | 0.843 | 1.000 |
| `wrong_cv` (38) | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 |

Tập kiểm tra 34 mẫu: đúng 33/34 (sai mẫu 66, AML/BM/CCG → đoán ALL), accuracy 0.971, Wilson 95%
[0.851, 0.995]; McNemar (pipeline minh họa vs lớp đa số): đúng 33 vs 20/34, p = 0.00024.

| Nhãn | Cách CV | Balanced accuracy (TB ± SD) | Số lần | Kỳ vọng (kế hoạch §5.3) | Trước ADR-031 (72 mẫu, cleansed) |
|---|---|---|---|---|---|
| Thật | Đúng | 0.951 ± 0.019 | 10 | cao | 0.960 ± 0.014 |
| Thật | Sai | 1.000 ± 0.000 | 10 | cao, có thể cao hơn CV đúng | 0.964 ± 0.006 |
| Hoán vị | Đúng | 0.510 ± 0.089 | 100 | xấp xỉ 0.50 | 0.501 ± 0.065 |
| Hoán vị | Sai | **0.826 ± 0.065** | 100 | cao bất thường (bằng chứng H3) | 0.807 ± 0.059 |

→ **H3 vẫn được xác nhận** khi chỉ học trên 38 mẫu. `make evaluate` chạy 6 phút, không còn cảnh báo
nào (chạy với `-W error::UserWarning`).

### 3.4. Lưu ý triển khai
- **Bẫy log hai lần** (đã xử lý ở **[2.4-f8-f10-demo-v2]**): `_baseline_pipeline()` nay nhận X_log10 của
  tầng curated và không có bước log10 nào; đưa tầng cleansed (thang gốc) vào là sai.
- **Quy ước bước `select`**: `_split_at_selector()` cắt tại bước tên `select` (hoặc `SelectorMixin` cuối);
  bước lọc gen của TV3 nằm trong `make_preprocess()` đứng trước `select`.
- **`uncertain`**: coi là sai; báo cáo kèm `n_uncertain` vì cách tính của bài báo (36/36 trên mẫu chắc chắn) khác.
- **Weighted voting của TV1** cần `classes_ = ['ALL', 'AML']` và `predict_proba` theo đúng thứ tự đó.
- **F11 bản cuối** chuyển cho TV1 (lịch mới, tuần 3), vẽ bằng `plot_roc_cm()`.

## 4. Giai đoạn 2 – 2.4 (phần mới, ADR-014). Nhập theo lô, chất lượng — **[2.4-ingest-v2]**

### 4.1. Giao diện (khóa ở `tests/test_interfaces.py`)
| Hàm | Mô tả |
|---|---|
| `ingest_batches(extra_files=None, write=False)` | Chia dữ liệu thành 4 lô theo nguồn, kiểm tra từng lô, chặn lô lỗi. Trả về bảng một hàng mỗi lô: `batch_id, source, n_samples, status ('passed'|'blocked'), reason, warnings, n_train, n_test, n_ALL, n_AML, pct_below_100, pct_above_16000, present_rate, median_corr_other, n_qc_outlier, origin`. `write=True` (`make ingest`) ghi `data/cleansed/batches.parquet` (→ `dim_batch`), `qc_samples.parquet` (→ `fact_qc`) và **ghi thêm** `ingest_log.csv`. Mặc định không ghi file để dashboard gọi trực tiếp. |
| `validate_new_sample(path)` | File chip `probe_id,value` (đường dẫn hoặc file tải lên từ Streamlit) → `{ok, errors, warnings, values}`; `values` là Series theo thứ tự `genes.parquet`, dùng cho `patient_profile(values=...)` của TV1. |
| `sample_quality(extra_files=None)` | Bảng chất lượng theo mẫu (`fact_qc`). Thêm ngoài giao diện khóa (ADR-030). |
| `write_demo_files()` | **[2.4-f8-f10-demo-v2]** Sinh bộ file lỗi mẫu vào `data/demo/` (mục 4.4). |

### 4.2. Quy tắc (ADR-030)
- **Chặn lô**: thiếu/thừa/trùng probe so với 7129 probe chuẩn; ô trống; giá trị không phải số; `sample_id`
  trống/trùng/đã thuộc lô khác; metadata bắt buộc thiếu hoặc ngoài danh mục; lô nhiều nguồn; `class` lệch
  `subtype`; call ngoài P/A/M. Chặn một lô không ảnh hưởng lô khác.
- **Chỉ báo cáo** (không chặn): tương quan với mẫu các lô khác, tỷ lệ call P, tỷ lệ ô ngoài [100, 16000],
  `qc_outlier` (cùng tinh thần ADR-004).
- **`validate_new_sample`**: lỗi khi sai tiêu đề, số dòng ≠ 7129, thiếu/thừa/trùng mã gen, ô trống, không
  phải số, file rỗng, hoặc > 90% giá trị < 100 (đã log hóa — 72 chip thật chỉ 40–56%); cảnh báo khi chip
  lệch 38 mẫu tham chiếu (modified z ngưỡng 3.5 của tương quan trung vị, trung vị, IQR). Cả 34 bệnh nhân
  tập kiểm tra đều qua, **không có cảnh báo nhầm**. (Quy tắc min–max ban đầu đã bỏ vì báo nhầm bệnh nhân 52.)

### 4.3. Kết quả trên 4 lô thật (`make ingest`, 12/11/2026)
| Lô | Số mẫu (train/test) | ALL/AML | Trạng thái | Tỷ lệ call P | Tương quan với lô khác | `qc_outlier` |
|---|---|---|---|---:|---:|---|
| B1-DFCI | 44 (27/17) | 44/0 | passed | 0.300 | 0.818 | 1 (mẫu 21) |
| B2-CALGB | 15 (11/4) | 0/15 | passed | 0.275 | 0.822 | 0 |
| B3-St-Jude | 8 (0/8) | 3/5 | passed | 0.236 | 0.826 | 0 |
| B4-CCG | 5 (0/5) | 0/5 | passed | 0.272 | 0.827 | 0 |

St-Jude có tỷ lệ call P thấp nhất (khớp ghi chú của bài báo về quy trình lab khác). DFCI toàn ALL, CALGB toàn
AML nên chỉ số của hai lô này lẫn với khác biệt lớp (confounding).

### 4.4. Bộ file lỗi mẫu và hình F8 — **[2.4-f8-f10-demo-v2]** (ADR-031)
- `python -m src.ingest --demo` sinh **13 file** từ dữ liệu thật vào `data/demo/` (không lên Git) + `README.md`
  ghi kết quả mong đợi: 1 chip hợp lệ (bệnh nhân 52), 8 chip lỗi (thiếu 29 gen, sai mã gen, trùng gen, ô trống,
  không phải số, sai tiêu đề, đã log10, file rỗng), 1 chip cảnh báo (giá trị bị xáo, r = 0.004), 3 lô (1 hợp lệ
  `passed`, 2 lỗi `blocked`). Test kiểm từng file. Bộ cũ `data/sandbox/demo_errors/` (dict JSON, 3 dòng CSV,
  thêm bằng `git add -f`) đã gỡ.
- **F8 = "Chất lượng dữ liệu theo lô"** (`src/viz_ingest.py`, `python -m src.viz_ingest` sau `make ingest`):
  strip plot theo lô của tương quan với các lô khác và tỷ lệ call P; mẫu 21 khoanh đen; lô bị chặn là màu nhấn.
  **Giả định cần TV1 xác nhận** — tài liệu "Kế hoạch giai đoạn 2 (hướng mới)" chưa có trong repo; F8 cũ của
  kế hoạch gốc (hiệu năng theo k) thuộc H2 đã bỏ.

## 5. Nghiệm thu công việc 2.4

### 5.1. Rà soát 3 nhánh cũ (12/11/2026) — đều **chưa đạt**, đã làm lại trên nhánh `-v2`
| Nhánh cũ | Vấn đề (đã kiểm chứng bằng chạy thật) | Nhánh mới |
|---|---|---|
| `2.4-ingest` `7717555` | `ingest_lots()` chia "lô" theo 4 **file raw** thay vì 4 **nguồn**; `validate_new_sample(sample: dict, genes_ref)` trả `{valid, errors, warnings, normalized}` — lệch giao diện ADR-014; đặt lên nền hiện tại thì `tests/test_interfaces.py` báo `src.ingest thiếu ['ingest_batches']`; không có test | `2.4-ingest-v2` |
| `2.4-f8-f10-demo` `562ec9c` | Không có F8; F10 vẫn bản 1; file demo là dict metadata / CSV 3 dòng, không khớp `validate_new_sample(path)`; thêm vào `data/sandbox/` bằng `git add -f` (trái `.gitignore`) | `2.4-f8-f10-demo-v2` |
| `2.4-report-draft` `1d4ec8f` | Số liệu **không khớp** CSV ở hầu hết các ô (vd `original_split` 0.975 vs thật 0.964; SD hoán vị-đúng 0.110 vs 0.065; McNemar "34 đúng" vs thật 33 trong khi accuracy ghi 0.971); vẫn hứa F11 "mô hình cuối của TV3" dù ADR-014 đã bỏ | `2.4-report-draft-v2` |

### 5.2. Kiểm chứng
- Môi trường ảo riêng (Python 3.11, đúng phiên bản `environment.yml`), tạo trong thư mục tạm và **đã xóa** sau
  khi chạy; dữ liệu tải lại bằng `make fetch` (MD5 4/4 khớp) và dựng lại bằng `make data`.
- `pytest tests/`: `2.4-evaluate` **83 passed** → `2.4-ingest-v2` **103 passed** → `2.4-f8-f10-demo-v2`
  **108 passed** (thêm `tests/test_ingest.py`, `tests/test_viz_ingest.py`, 2 test hồi quy F10).
- `notebooks/08_evaluation.ipynb` và `notebooks/TV2_tong_quan_cong_viec.ipynb` chạy lại sạch bằng
  `jupyter nbconvert --execute` (ô `make -n` của notebook tổng quan nay tự đọc Makefile khi máy không có `make`).

### 5.3. Sản phẩm bàn giao
| Sản phẩm | Vị trí | Nhánh |
|---|---|---|
| `src/evaluate.py` 4 sơ đồ, Wilson, McNemar, selection bias | `src/` | `2.4-evaluate` (PR) |
| `src/ingest.py`, `quality.sample_qc()`, `tests/test_ingest.py`, data dictionary, ADR-030 | `src/`, `tests/`, `docs/` | `2.4-ingest-v2` |
| Bộ file lỗi mẫu (`--demo`), F8, F10 bản cuối, notebook 08, ADR-031 | `src/`, `results/`, `notebooks/` | `2.4-f8-f10-demo-v2` |
| Nháp báo cáo: `docs/report/2.8_nhap_theo_lo.md`, `docs/report/3.5_danh_gia.md` | `docs/report/` | `2.4-report-draft-v2` |

### 5.4. Demo nghiệm thu (chạy trước trưởng nhóm)
```bash
make fetch && make data          # raw → curated; toàn bộ test xanh
make ingest                      # 4 lô passed; ghi batches/qc_samples/ingest_log
python -m src.ingest --demo      # 13 file demo: 1 hợp lệ, 8 lỗi, 1 cảnh báo, lô lỗi bị chặn
python -m src.viz_ingest         # F8
make evaluate                    # 4 sơ đồ + 100 hoán vị → summary_*.csv, F10, F11
```
```python
>>> from src.ingest import ingest_batches, validate_new_sample
>>> ingest_batches()[["batch_id", "n_samples", "status"]]
>>> validate_new_sample("data/demo/loi_thieu_gen.csv")["errors"]
>>> from src.evaluate import evaluate
>>> res = evaluate(pipe, X_tr, y_tr, scheme='nested_cv', seeds=range(10))
>>> res.keys()   # 6 khóa: thêm 'predictions' (ADR-008)
```

### 5.5. Việc còn lại
- Push 3 nhánh `-v2`, mở PR theo thứ tự `2.4-evaluate` → `2.4-ingest-v2` → `2.4-f8-f10-demo-v2` →
  `2.4-report-draft-v2` (mỗi PR chỉ còn commit riêng sau khi PR trước được merge).
- TV1 xác nhận định nghĩa F8; TV3 đọc `batches.parquet`/`qc_samples.parquet` qua hằng `BATCHES_FILE`, `QC_FILE`
  vào `dim_batch`/`fact_qc`; TV6 gọi `ingest_batches()` và `validate_new_sample(file_tải_lên)` từ màn hình
  Chất lượng dữ liệu; TV1 nhận `values` cho `patient_profile`.
- Lệnh `make figures` (của TV6) chưa gọi `src.viz_ingest`; báo nhóm trước khi sửa Makefile (ADR-014).
- Review rò rỉ dữ liệu code TV1 (`golub_wv.py`, `patient.py`) khi có.
- Phần báo cáo giai đoạn 1 của TV2 (mục 2.1–2.5, 2.7) chưa có file trong `docs/report/`.
