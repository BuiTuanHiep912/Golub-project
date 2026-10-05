# Review rò rỉ dữ liệu (TV2 - mục 2.4, nhiệm vụ 5)

Nội dung: rà soát code của nhóm để phát hiện data leakage — nhất là **chọn gen / chuẩn
hóa trước khi tách train-test**, nguyên nhân chính của selection bias (H3).
Kết hợp với `tests/test_no_leakage.py` của TV3 khi có pipeline chính thức.

## Nguyên tắc đối chiếu (mục 2.6 kế hoạch)
1. Mọi bước lọc/chọn gen và chuẩn hóa phải NẰM TRONG scikit `Pipeline`, chỉ `fit`
   trên phần train của từng fold.
2. Mô hình đọc `X_log10.parquet` (chưa lọc), KHÔNG đọc bản đã lọc theo toàn bộ dữ liệu.
3. Không được fit một lần (fit_once) trên toàn bộ rồi tách fold.

## Kết quả đã rà soát

### TV1 - `golub_wv.py` (weighted voting) ✅ Đạt
- Chọn gen bằng |SNR| nằm trong `fit()` của `GolubWeightedVoting` (dòng 40-45) →
  mỗi lần `fit` lại trên chỉ train fold. Đúng mô hình khi dùng trong CV.
- Ghi chú nhỏ (không phải leak):
  - `predict_proba` là map tuyến tính từ PS sang [0,1], chỉ dùng vẽ ROC.
  - Giả định `np.unique` sắp nhãn: 'ALL', 'AML' — đúng với dữ liệu hiện tại; nên
    truyền `classes` tường minh cho chắc.

### TV1 - `reproduce_golub.py` ✅ Đạt
- LOOCV chạy trên 38 mẫu train, fit lại `GolubWeightedVoting` từng fold (dòng 21).
- Fit 38 train → dự đoán 34 test (dòng 45-46). Không dùng data test để học.
- X đọc từ `load_golub(layer="cleansed")` — chưa lọc gen toàn cục ✓.

### TV1 - `design_batch.py` ✅ Đạt (EDA)
- PCA/Kruskal-Wallis/mann-whitney là phân tích mô tả (H6, batch effect), không dùng
  để chọn đặc trưng cho mô hình — không vi phạm.
- Ghi chú: phụ thuộc `src.preprocess.threshold_log10` (TV3) — chưa có module.

### TV2 - `evaluate.py` ✅ Đạt (thiết kế chống leak từ đầu)
- `loocv`: `clone(pipeline)` mỗi fold → scale/chọn gen học trên 37 train.
- `nested_cv`: `GridSearchCV` vòng trong + `cross_val_score` vòng ngoài.
- `wrong_cv`: chủ ý gây leak (fit scale+chọn gen trên toàn bộ) chỉ để minh hoạ H3;
  kết quả không được dùng làm con số báo cáo.
- `selection_bias.py`: đúng cách = pipeline trong `cross_val_score`; sai cách = fit
  trước toàn bộ — dùng để chứng minh khác biệt.

## Khuyến nghị cho nhóm (cần làm khi TV3 đưa pipeline thật)
1. Viết `tests/test_no_leakage.py`: rằng buộc pipeline huấn luyện không gọi
   `fit_transform`/`fit` trên tập test toàn bộ trước CV; kiểm tra bằng cách train
   trên dữ liệu đã hoán vị nhãn phải cho hiệu năng ≈ 0.50 trong CV đúng.
2. `preprocess.py` (TV3): chỉ viết `X_log10.parquet` (threshold+log10 - phép cố định);
   tuyệt đối không ghi `X_log10_filtered` ra trước CV để mô hình đọc.
3. Notebook không đọc tầng filtered để huấn luyện; chỉ dùng trong EDA/giải thích.
4. Mọi tham số tái lập (seed, n_genes, C) ghi vào `config/experiment_config.yaml`
   và Dim_Run (Bảng 2.6).

## Bằng chứng thực nghiệm H3 (kết quả `selection_bias_summary.csv`)
| Nhãn | Cách CV | Balanced acc (TB ± SD) |
|---|---|---|
| Thật | Đúng | 0.97 ± 0.04 |
| Thật | Sai | 0.97 ± 0.04 |
| Hoán vị | Đúng | 0.50 ± 0.07 |
| Hoán vị | Sai | 0.72 ± 0.08 |

Nhãn hoán vị + CV sai cho 0.72 (chênh 0.22 so với 0.50 đúng) → bằng chứng rõ cho
selection bias. Không có tín hiệu thật, CV sai vẫn "thắng" nhờ chọn gen rỗi rị.