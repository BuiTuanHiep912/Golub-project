# TV2 – Khái quát công việc

**Đề tài:** Đề tài số 10 – Golub 1999 (ALL/AML, High-Dimensional Data Analysis)

**Phân công (mục 5.1, Ke_hoach_trien_khai_project_Golub.pdf):**

| Giai đoạn | Công việc | Vai trò trong nhóm |
|---|---|---|
| GĐ1 (tuần 1–4) | **1.1. Nạp, ghép dữ liệu và kiểm tra chất lượng** | Dựng hạ tầng dữ liệu dùng chung |
| GĐ2 (tuần 5–8) | **2.4. Khung đánh giá và thí nghiệm selection bias** | Dựng hạ tầng đánh giá dùng chung |

## Nhiệm vụ chính

**TV2 là "người dựng hạ tầng":** tạo bộ dữ liệu sạch chuẩn cho cả nhóm (nạp 3 file
Kaggle + `golub.csv` của OpenIntro làm metadata → parquet), đảm bảo chất lượng (4 mức kiểm
tra), sau đó dựng khung đánh giá mô hình dùng chung (4 sơ đồ CV) và làm thí nghiệm
selection bias chứng minh tác hại của "chọn gen trước rồi mới CV".

## Các nội dung thực hiện
- Nạp/ghép dữ liệu: quy trình mục 2.1.2 (left join theo `sample_id`) → `expression.parquet`,
  `calls.parquet`, `samples.parquet`, `genes.parquet`. Nguồn metadata: `golub.csv` của OpenIntro
  (ADR-001, `docs/decisions.md`); nhãn từ Kaggle `actual.csv` (ADR-002).
- Kiểm tra chất lượng: đúng 4 mức Bảng 2.9 (Value, Value-set, Record, Relation) →
  `quality_report.md`, tầng `cleansed`, cờ `qc_outlier`.
- Đánh giá: `evaluate(pipe, X, y, scheme)` với `original_split`, `loocv`, `nested_cv`, `wrong_cv`;
  balanced accuracy, AUC, sensitivity/specificity, Wilson CI, McNemar.
- Thí nghiệm selection bias: 100 lần hoán vị nhãn × {CV đúng, CV sai}; review rò rỉ dữ liệu
  TV1/TV3/TV4 → `docs/leakage_review.md` (**chưa làm**: code các TV này còn `# TODO`).

## Sản phẩm bàn giao
- `src/{config,load,quality,evaluate}.py` + `tests/{test_load,test_quality,test_evaluate}.py`
- `Makefile`, `environment.yml` (gốc repo); `data/cleansed/` + `quality_report.md`
- `docs/data_dictionary.md`, `docs/decisions.md` (ADR-001..009), `docs/handover/1.1.md`
- `notebooks/08_evaluation.ipynb`; hình `results/figures/{report,slides}/F10_selection_bias_*.png`,
  `F11_final_model_roc_cm_*.png`; mục 2.1–2.5 + 3.5 của báo cáo.

## Hiện trạng (đã chạy trên dữ liệu thật)
- `make data && pytest tests/` → **33 passed, 1 skipped** (skip = test của TV3); ma trận
  72×7129 khớp 100% Bảng 2.4.
- Selection bias (100 hoán vị, ngày 07/10): thật/đúng 0.960 · thật/sai 0.964 ·
  hoán vị/đúng 0.501 · hoán vị/sai 0.807 → bằng chứng H3.
- `notebooks/08_evaluation.ipynb` chạy thật: 0 lỗi, 0 warning, 2 ảnh nhúng.

## Chi tiết
Xem `Cong_viec_chi_tiet_TV2.md` (mô tả đầy đủ + demo nghiệm thu) và `code/`.