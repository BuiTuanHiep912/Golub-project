# Rà soát rò rỉ dữ liệu (data leakage) — công việc 2.4

> TV2 · Kế hoạch mục 2.6 ("Nguyên tắc chống rò rỉ dữ liệu") và mục 5.3, công việc 2.4 ("Review code của
> TV1, TV3, TV4 để phát hiện rò rỉ dữ liệu") · quy ước dữ liệu ADR-014.
>
> Rà soát ngày 18/11/2026 trên `origin/main` `fa6f8a8` (số dòng `file:dòng` tính trên commit đó; code của
> TV2 tính trên nhánh `2.4-report-v3`). Cách làm: đọc code và test của từng người. **Chưa chạy lại** trên
> máy rà soát vì máy không có môi trường Python có scikit-learn; kết luận "an toàn" dựa trên code và test
> hiện có của chính chủ file.

## 1. Tiêu chí

1. Mọi bước **học tham số từ dữ liệu** (lọc gen, chuẩn hóa, chọn gen, chọn siêu tham số, huấn luyện) chỉ
   fit trên phần huấn luyện. Khi đánh giá bằng cross-validation thì các bước này nằm trong scikit-learn
   `Pipeline` để fit lại ở mỗi fold (kế hoạch mục 2.6).
2. **34 mẫu tập kiểm tra là bệnh nhân mới**: không dùng để huấn luyện, chọn gen hay tính tham số (ADR-014).
3. **Phép biến đổi cố định từng phần tử** (threshold [100, 16000], log10) áp cho cả 72 mẫu được, vì không
   học gì từ dữ liệu.

Thống kê mô tả hoặc khám phá trên cả 72 mẫu **không** phải rò rỉ, miễn là kết quả không quay lại làm đầu
vào cho mô hình hay làm "mức tham chiếu" để chấm bệnh nhân mới.

## 2. Tóm tắt

| File | Chủ | Kết luận |
|---|---|---|
| `src/preprocess.py`, `tests/test_no_leakage.py` | TV3 | **An toàn.** Lọc gen và chuẩn hóa học trong `Pipeline`; có test chứng minh |
| `src/golub_wv.py` | TV1 | **An toàn.** Mọi tham số học trong `fit` từ đúng dữ liệu truyền vào; có test chứng minh |
| `src/patient.py` | TV1 | **Chưa rà được**: mới có giao diện (`NotImplementedError`). Mục 3.3 liệt kê điểm sẽ kiểm |
| Code tái lập LOOCV của bài báo | TV1 | **Chưa rà được**: chưa có trên `main` (lịch tuần 2, hạn 22/11) |
| `src/gene_explorer.py` | TV4 | **An toàn cho khám phá.** Một lưu ý khi dashboard dùng làm mức tham chiếu (mục 3.4) |
| `notebooks/04_de_genes.ipynb` → `gene_ranking.csv` | TV4 | **Rò rỉ nhẹ, không dùng nhãn**: tập gen ứng viên lọc trên 72 mẫu (mục 3.5) |
| `src/warehouse.py` | TV3 | **Không học tham số, không rò rỉ.** Có một lỗi tích hợp với TV2 về `batch_id` (mục 4.1) |
| `src/evaluate.py`, `src/ingest.py`, `src/quality.py` | TV2 | **An toàn** (tự rà, mục 3.7); `wrong_cv` rò rỉ có chủ đích |

## 3. Chi tiết

### 3.1. TV3 — `src/preprocess.py`, `tests/test_no_leakage.py`

- `threshold_log10()` `:41–43` là phép biến đổi cố định, áp cho 72 mẫu được (tiêu chí 3).
- `GeneFilter` `:46–72`: `fit` chỉ nhìn dữ liệu được truyền vào (không dùng nhãn); `transform` chỉ chọn
  các cột đã học. `make_preprocess()` `:75–77` = `GeneFilter` → `StandardScaler`, dùng trong `Pipeline`.
  `_baseline_pipeline()` của TV2 dùng đúng hàm này nên lọc gen và chuẩn hóa được fit lại ở mỗi fold.
- `main()` `:83` ghi `X_log10_filtered.parquet` lọc trên 72 mẫu. File này được ghi rõ là **chỉ để khám
  phá** (docstring `:9–11`, `docs/data_dictionary.md`, ADR-011, `docs/handover/1.2.md`). Rủi ro nằm ở người
  dùng file, xem mục 3.5.
- Test: `test_filter_fitted_on_train_only` `:39` và `test_scaler_params_from_train_only` `:54` kiểm tham số
  học được không đổi khi thay hai tập test khác hẳn nhau.

### 3.2. TV1 — `src/golub_wv.py`

- `fit()` `:56–96` học trung bình/SD để chuẩn hóa (`:70–72`), SNR (`:84–85`), chọn gen (`:88–89`), trọng số
  `a_`, ranh giới `b_` (`:92–93`), tất cả từ đúng `X`, `y` truyền vào. Mẫu mới được chuẩn hóa bằng tham số
  của train (`:109`).
- `__main__` `:152–155` fit trên 38 mẫu train rồi dự đoán 34 mẫu test, đúng ADR-014.
- Test: `tests/test_golub_wv.py:89` `test_no_leakage_from_test_set` (hai tập test khác nhau không làm đổi
  gen, `a`, `b`).
- Tương thích với `evaluate()`: `classes_ = ['ALL', 'AML']` (`:94`), `predict` trả `'uncertain'` (`:139`)
  và `evaluate()` tính câu này là sai. Dùng được với `original_split`, `loocv`, `nested_cv`. Không dùng
  được với `wrong_cv` vì bước chọn gen nằm bên trong `fit`, không có bước `select` riêng; `evaluate()` báo
  lỗi rõ ràng. H3 không cần cặp này.
- **Lưu ý cho phần tái lập LOOCV (lịch tuần 2):** phải fit lại cả bước chọn 50 gen ở mỗi lượt, tức truyền
  nguyên `GolubWeightedVoting()` vào `evaluate(..., "loocv")`. Chọn 50 gen một lần trên 38 mẫu rồi mới
  LOOCV chính là `wrong_cv`, và sẽ cho kết quả lạc quan như thí nghiệm H3 đã chỉ ra.

### 3.3. TV1 — `src/patient.py` (chưa cài đặt)

`patient_profile()` `:23–25` mới `raise NotImplementedError`. Khi có code, TV2 sẽ kiểm:

1. Mô hình và "5 bệnh nhân giống nhất" chỉ lấy 38 mẫu train làm tham chiếu (docstring `:6–7` đã ghi đúng
   quy ước này).
2. `patient_profile(sample_id=k)` với `k` thuộc tập train: mô hình đã học chính bệnh nhân đó nên PS sẽ lạc
   quan. Nên fit lại mô hình bỏ bệnh nhân `k` (kiểu LOOCV), hoặc ghi chú trên dashboard.
3. `values` trả về từ `validate_new_sample()` ở **thang MAS4 gốc** (như `data/raw`), trong khi
   `GolubWeightedVoting` học trên `X_log10`. Phải gọi `threshold_log10()` trước khi dự đoán. Đây không
   phải rò rỉ nhưng là bẫy thang đo dễ mắc.
4. Tương quan để tìm bệnh nhân giống nhất không được tính trên tập gen chọn bằng nhãn của cả 72 mẫu.

### 3.4. TV4 — `src/gene_explorer.py`

- Module không fit mô hình. `gene_values()` `:186`, `gene_stats()` `:196`, `disease_or_source()` `:211` tính
  thống kê mô tả trên cả 72 mẫu của tầng curated. Đây là khám phá, không phải rò rỉ.
  `disease_or_source()` ghi rõ so sánh trong St-Jude chỉ gồm mẫu tập kiểm tra (`:216–217`).
- `ranking_table()` `:242` đọc `gene_ranking.csv`, kèm ghi chú `RANKING_NOTE` `:57` "tính trên 38 mẫu huấn
  luyện".
- **Lưu ý:** nếu dashboard (TV6) hoặc `patient.py` (TV1) dùng `gene_stats()` làm "khoảng tham chiếu" để
  chấm bệnh nhân mới, thì 34 bệnh nhân mới đã lọt vào mức tham chiếu, trái ADR-014. Khi đó gọi
  `gene_stats(probe_id, by=("split", "class"))` và chỉ lấy hàng `split == "train"`.

### 3.5. TV4 — `notebooks/04_de_genes.ipynb` → `results/gene_lists/gene_ranking.csv` (công việc 1.6)

- SNR, t, q được tính trên **38 mẫu train** (cell 9), đúng.
- Nhưng đầu vào là `X_log10_filtered.parquet` (cell 2), tức tập gen ứng viên đã được **lọc trên 72 mẫu**,
  có nhìn giá trị (không nhìn nhãn) của 34 mẫu test. Theo `data/curated/preprocess_params.json`, lọc trên
  72 mẫu giữ 3571 gen, lọc trên 38 mẫu train chỉ giữ 3052 gen. Bộ lọc dùng max/min trên các mẫu, nên thêm
  mẫu chỉ làm tập gen lớn lên: **519 gen qua được bộ lọc chỉ nhờ giá trị của tập kiểm tra.** Notebook tự
  ghi nhận một phần chuyện này (cell 6: bỏ gen hằng trên train nhưng "chỉ thay đổi ở tập test nên vẫn qua
  bộ lọc 72 mẫu").
- **Mức độ: nhẹ.** Bộ lọc không dùng nhãn, và từ ADR-014 xếp hạng chỉ dùng để khám phá (`features.py` đã
  bị xóa, không còn mô hình nào dùng xếp hạng này). Nhưng nó trái chữ của ADR-014 ("không dùng 34 mẫu để
  chọn gen").
- **Đề xuất cho TV4** (một trong hai):
  - lọc lại bằng `GeneFilter().fit(X_log10[train])` rồi tính lại xếp hạng (còn 3052 gen ứng viên); hoặc
  - giữ nguyên và ghi hạn chế này trong mục báo cáo của 1.6 / 2.2.

### 3.6. TV3 — `src/warehouse.py`

Kho chỉ chép dữ liệu, không học tham số: `fact_expression` `:44–54` chứa cả 72 mẫu kèm `dim_sample.split`
để người dùng tự lọc. `docs/handover/2.3.md` đã nhắc lọc `WHERE split = 'train'` khi tính tham số.
Không có rò rỉ. Lỗi tích hợp ở mục 4.1.

### 3.7. TV2 tự rà — `src/evaluate.py`, `src/ingest.py`, `src/quality.py`

- `evaluate()`: `original_split` chỉ fit trên train; `loocv` và `nested_cv` fit lại cả pipeline ở mỗi fold;
  vòng trong `GridSearchCV` chỉ chạy trên phần train của fold ngoài. `wrong_cv` **cố ý** fit bước chọn gen
  trên toàn bộ dữ liệu (ghi chú ở chỗ code), chỉ dùng cho thí nghiệm H3. Từ nhánh `2.4-f8-f10-demo-v2`,
  `make evaluate` chỉ học trên 38 mẫu (`src/evaluate.py:517`, `:527`, `:556`).
- `validate_new_sample()`: mẫu tham chiếu để cảnh báo chip lệch là 38 mẫu train (`src/ingest.py:459`).
- `ingest_batches()`, `sample_qc()` và `check_record()` tính tương quan trên cả 72 mẫu. Đây là kiểm tra
  chất lượng: cờ `qc_outlier` không loại mẫu nào và không đi vào mô hình (ADR-004), nên không phải rò rỉ.

## 4. Phát hiện ngoài rò rỉ (gặp khi rà)

### 4.1. `batch_id` của TV2 và TV3 không khớp — **cần chốt**

- `src/ingest.py:130` (TV2) đặt `batch_id = "B1-DFCI"`, `"B2-CALGB"`, `"B3-St-Jude"`, `"B4-CCG"`.
- `src/warehouse.py` (TV3) đặt `batch_id` = tên nguồn (`"DFCI"`, …) cho `dim_batch` (`:57–69`) và
  `dim_sample` (`:90`), rồi nạp nguyên `ingest_batches()` làm `fact_qc` (`:72–78`, `:97–99`). Kết quả:
  `fact_qc.batch_id` **không join được** với `dim_batch.batch_id`. `docs/handover/2.3.md` có dòng "TV2 cần
  giữ `batch_id` = tên nguồn".
- ADR-030 của TV2 lại ghi `batches.parquet` → `dim_batch`, `qc_samples.parquet` → `fact_qc`, khác cách TV3
  đang nạp. Cần thống nhất một trong hai.
- Hai cách sửa:
  - (a) TV2 đổi `batch_id` của 4 lô thật thành tên nguồn. Lô thêm từ file vẫn lấy tên file, vì một nguồn có
    thể gửi nhiều lô. Phải sửa thêm test, F8 và bảng trong mục 2.8.
  - (b) TV3 join qua cột `source`, có sẵn ở cả hai bảng.

### 4.2. Thang đo của `values` từ `validate_new_sample()`

Như mục 3.3, điểm 3: `values` ở thang MAS4 gốc; mô hình học trên `X_log10`. Đã ghi trong
`docs/handover/2.4.md` cho TV1 và TV6.

## 5. Việc cần làm

| Việc | Ai | Khi nào |
|---|---|---|
| Gửi code tái lập LOOCV và `patient_profile()` để TV2 rà theo mục 3.2, 3.3 | TV1 → TV2 | khi xong (lịch tuần 2–3) |
| Chọn lọc lại trên 38 mẫu hay ghi hạn chế cho `gene_ranking.csv` | TV4 | trước khi nộp nháp báo cáo (29/11) |
| Chốt cách join `batch_id` (mục 4.1) | TV2 + TV3 | sớm, trước khi TV3 làm F7, F9 |
| Khoảng tham chiếu trên dashboard chỉ lấy 38 mẫu train | TV6 | khi nối hàm thật (lịch tuần 3) |
