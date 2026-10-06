# Nhật ký quyết định

Mọi quyết định ảnh hưởng tới dữ liệu hoặc cách đánh giá đều ghi ở đây để kết quả
tái lập được và truy ngược về nguồn.

## ADR-001 — Metadata lấy 6 cột từ `openintro/golub.csv`
- **Ngày:** 05/10/2026 · **Người đề xuất:** TV2 · **Người quyết định:** TV2
- **Bối cảnh:** Kaggle có biểu hiện + nhãn nhưng **không có** thông tin mẫu (mô, giới
  tính, ngân hàng mẫu, nhóm miễn dịch). OpenIntro `golub.csv` có đủ 6 cột này.
- **Quyết định:** `golub.csv` là nguồn metadata chính; đọc bằng `usecols` để chỉ lấy
  6 cột (bỏ 7129 cột biểu hiện của OpenIntro, vì biểu hiện lấy từ Kaggle).
- **Ánh xạ cột:** `Samples→sample_id`, `BM.PB→tissue`, `Gender→sex`, `Source→source`,
  `cancer→subtype` (`allB→B-ALL`, `allT→T-ALL`, `aml→AML`). `tissue.mf` (tổ hợp mô và
  giới tính, không phải "morphology") chỉ dùng để kiểm tra chéo trong `load.py` rồi bỏ,
  đúng Bảng 2.2 — sửa ngày 06/10/2026, khớp 72/72.
- **Hệ quả:** `Gender` thiếu 23/72 là thiếu thật ở nguồn → giữ NaN, không tự điền.

## ADR-002 — Nhãn `class` lấy theo Kaggle (`actual.csv`)
- **Ngày:** 05/10/2026 · **Người quyết định:** TV2
- **Lý do:** mục 2.1.2 của kế hoạch — nhãn theo bộ dữ liệu chính, metadata chỉ dùng
  cho phân tích phụ. Đã kiểm tra khớp `subtype` 72/72 (xem `tests/test_load.py`).

## ADR-003 — `split` gán theo file Kaggle (1..38 train, 39..72 test)
- **Ngày:** 05/10/2026 · **Người quyết định:** TV2
- **Lý do:** `data_set_ALL_AML_train.csv` chứa patient 1..38, file independent chứa
  39..72. Thứ tự cột trong file không bảo đảm theo số thứ tự → mọi lần ghép đều theo
  mã bệnh nhân, không theo vị trí.

## ADR-004 — Tầng cleansed chỉ gắn cờ `qc_outlier`, không xóa mẫu
- **Ngày:** 05/10/2026, sửa 06/10/2026 · **Người quyết định:** TV2
- **Lý do:** mức Record của mục 2.5 chỉ phát hiện, không tự quyết xóa dữ liệu. Cần review
  của nhóm trước khi loại mẫu.
- **Cách phát hiện (sửa 06/10/2026):** bản cũ dùng z-score toàn ma trận (|z| lớn nhất chỉ
  ~2.5 nên không thể vượt ngưỡng 4) và gán cờ theo vị trí thay vì `sample_id`. Bản mới
  theo Bảng 2.9: tương quan giữa các mẫu và boxplot theo mẫu (trung vị, IQR) trên log10,
  modified z-score median/MAD ngưỡng 3.5 (Iglewicz & Hoaglin, 1993), cộng phát hiện trùng
  lặp (tương quan > 0.99).
- **Kết quả:** mẫu 21 (train, ALL, DFCI) bị gắn cờ do tương quan trung vị thấp (0.756,
  z = −4.7). Vẫn giữ trong mọi phân tích; TV1/TV6 đối chiếu với boxplot F3 và PCA.

## ADR-005 — `SOURCES.md` do người tải ghi tay; code chỉ đối chiếu MD5
- **Ngày:** 05/10/2026 · **Người quyết định:** TV2
- **Lý do:** file ghi thêm ngày tải và người tải (thông tin ngoài phạm vi code).
  `src.load.write_sources()` in MD5 thực tế, cảnh báo nếu lệch, không ghi đè.

## ADR-006 — Kiểm tra chất lượng thất bại thì dừng pipeline
- **Ngày:** 06/10/2026 · **Người quyết định:** TV2
- **Quyết định:** có `[FAIL]` thì `src.quality` vẫn ghi `quality_report.md` nhưng không ghi
  (và xóa bản cũ của) `expression/samples.parquet` tầng cleansed, rồi thoát mã 1 để
  `make data` dừng. `tests/test_load.py` báo lỗi (không skip) khi chưa có dữ liệu, để test
  nghiệm thu không thể xanh mà không kiểm gì.

## ADR-007 — Cố định phiên bản thư viện
- **Ngày:** 06/10/2026 · **Người quyết định:** TV2
- **Quyết định:** `environment.yml` và `requirements.txt` ghi cùng một bộ phiên bản, đã chạy
  `make standardize quality` + pytest xanh trên Python 3.11 (mục 2.7).
- **Ghi chú:** `pyarrow` ghim 24.0.0 vì `streamlit 1.65.0` loại trừ đúng bản 25.0.0 và
  conda-forge chưa có 25.0.1; `gseapy` chỉ cài qua pip vì không có trên conda-forge.

## ADR-008 — Quy ước của `src/evaluate.py` (công việc 2.4, bản 1)
- **Ngày:** 06/10/2026 · **Người quyết định:** TV2
- **Kết quả trả về:** đủ 5 khóa của demo nghiệm thu (`bal_acc_mean`, `bal_acc_sd`, `auc_mean`,
  `per_fold`, `config`) và thêm `predictions` (dự đoán theo mẫu), vì McNemar, ROC (F11) và
  phân tích lỗi của TV5 đều cần dự đoán từng mẫu.
- **Trung bình ± SD:** tính qua các dòng `per_fold` (fold × seed), giống `cross_val_score` ở
  mục 3.5.2. LOOCV gộp dự đoán của 38 fold rồi tính chỉ số một lần (fold 1 mẫu không có
  balanced accuracy/AUC); original_split và LOOCV có SD = NaN khi chỉ có một seed.
- **Seed và số fold** đọc từ `config/experiment_config.yaml`: mỗi seed là một lần lặp CV;
  `random_state` của mọi bước trong pipeline được gán bằng seed để chạy lại ra đúng số.
- **wrong_cv:** cắt pipeline tại bước tên `select` (hoặc bước chọn gen cuối cùng), fit phần
  trước trên cả 72 mẫu rồi mới CV phần sau.
- **Dự đoán "uncertain"** (weighted voting, PS < 0.3) tính là sai trong mọi chỉ số; số lượng
  ghi ở cột `n_uncertain`.
- **Tạm thời:** `python -m src.evaluate` dùng pipeline threshold+log10 → chuẩn hóa → 50 gen
  → logistic trên tầng cleansed, vì tầng curated của TV3 chưa có (xem NOTICE trong code).

## Nhật ký nhanh (mục 5.2 kế hoạch)

| Ngày | Quyết định | Lý do | Người quyết định |
|---|---|---|---|
| 05/10/2026 | Dùng Kaggle (biểu hiện gen + nhãn) + 6 cột mô tả của OpenIntro | Kaggle thiếu metadata; bản biểu hiện OpenIntro đã chuẩn hóa nên không dùng | TV2 |
| 05/10/2026 | Cấu trúc tầng raw → standardized → cleansed → curated → sandbox | Mục 2.2, Bảng 2.5 | Cả nhóm |
| 06/10/2026 | Viết lại kiểm tra chất lượng theo đúng 4 mức Bảng 2.9; mẫu 21 gắn cờ `qc_outlier` | ADR-004 | TV2 |
| 06/10/2026 | Ghép theo left join vào bảng mẫu Kaggle, so tập `sample_id` của 3 nguồn trước khi ghép | Mục 2.1.2 bước 5–6 | TV2 |
| 06/10/2026 | Kiểm tra FAIL dừng pipeline; cố định phiên bản thư viện | ADR-006, ADR-007 | TV2 |
| 06/10/2026 | `evaluate.py` bản 1: 4 sơ đồ, Wilson, McNemar, selection bias; thêm khóa `predictions` | ADR-008 | TV2 |