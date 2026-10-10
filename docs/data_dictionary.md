# Data dictionary (TV2 — công việc 1.1)

Quy ước: mọi tầng trong `data/` đều sinh tự động bằng `make data`; **không sửa tay**
file nào. Chi tiết quyết định nguồn/ánh xạ cột: `docs/decisions.md`.

## 1. Tầng raw — `data/raw/` (tải tay, chỉ đọc, có MD5 trong `SOURCES.md`)

| File | Cột | Kiểu / đơn vị | Nguồn | Ghi chú |
|---|---|---|---|---|
| `kaggle/data_set_ALL_AML_train.csv` | `Gene Description`, `Gene Accession Number`, rồi xen kẽ `k` (biểu hiện) và `k(call)` | int (thang MAS4), `call` ∈ P/A/M | Kaggle *Gene expression dataset (Golub et al.)* | probe theo **hàng**, bệnh nhân theo **cột** (1..38) |
| `kaggle/data_set_ALL_AML_independent.csv` | như trên | int, P/A/M | Kaggle | bệnh nhân 39..72 (tập test) |
| `kaggle/actual.csv` | `patient`, `cancer` | int, str | Kaggle | **nguồn nhãn chính** |
| `openintro/golub.csv` | `Samples`, `BM.PB`, `Gender`, `Source`, `tissue.mf`, `cancer` + 7129 cột biểu hiện | str + float | OpenIntro (`openintro.org/data` → golub) | chỉ dùng **6 cột mô tả** (ADR-001); `tissue.mf` chỉ để kiểm tra chéo |

Đặc thù thang đo MAS4 của bộ dữ liệu: có **giá trị âm** (147 174 giá trị), giá trị
`< 100` và `> 16000` được xem là ngoài vùng đo tin cậy (báo cáo ở mức 3, mục 2.5).
Nhóm probe điều khiển `AFFX-*` (58 probe) không phải gen người → đánh dấu `is_control`.

## 2. Tầng standardized — `data/standardized/` (sinh bởi `python -m src.load`)

| File | Nội dung | Kích thước / kiểu |
|---|---|---|
| `expression.parquet` | ma trận biểu hiện, index = `sample_id` (1..72), cột = probe | (72, 7129), int64, không NaN |
| `calls.parquet` | hiện diện probe P/A/M, cùng index/cột | (72, 7129), object, có thể NA — **không dùng cho mô hình** |
| `samples.parquet` | Dim_Sample, index = `sample_id` | (72, 6) |
| `genes.parquet` | Dim_Gene | (7129, 3) |

### `samples.parquet` — 6 cột mô tả

| Cột | Kiểu | Giá trị | Nguồn | Ghi chú |
|---|---|---|---|---|
| `split` | str | `train` (1..38) / `test` (39..72) | theo file Kaggle | không gán lại theo thứ tự cột trong file |
| `class` | str | `ALL` (47), `AML` (25) | `actual.csv` (Kaggle) | nhãn chính, mục 2.1.2 |
| `subtype` | str | `B-ALL` (38), `T-ALL` (9), `AML` (25) | `golub.csv: cancer` | map `allB→B-ALL`, `allT→T-ALL`, `aml→AML` |
| `tissue` | str | `BM` (tủy xương), `PB` (máu ngoại vi) | `golub.csv: BM.PB` | train 38 BM; test 24 BM + 10 PB |
| `source` | str | `DFCI`, `CALGB`, `CCG`, `St-Jude` | `golub.csv: Source` | xem Bảng 2.4 |
| `sex` | str | `M`, `F`, NaN | `golub.csv: Gender` | **thiếu 23/72** ở nguồn — không tự điền |

`golub.csv: tissue.mf` (`BM:f`, `PB:NA`, …) là tổ hợp của `tissue` và `sex`; `src/load.py`
dựng lại nó từ hai cột kia để kiểm tra chéo (khớp 72/72) rồi bỏ, không ghi ra (Bảng 2.2).

Phân bố chuẩn đối chiếu Bảng 2.4 (kiểm bằng `pytest tests/test_load.py`):
train = 27 ALL (toàn DFCI) + 11 AML (toàn CALGB); test = 20 ALL (17 DFCI + 3 St-Jude)
+ 14 AML (4 CALGB + 5 St-Jude + 5 CCG). Nhãn `class` khớp `subtype` 72/72.

### `genes.parquet`

| Cột | Kiểu | Mô tả |
|---|---|---|
| `probe_id` | str | Affymetrix accession (`AFFX-BioB-5_at`, `X95735_at`, …) |
| `description` | str | mô tả gen lấy từ Kaggle `Gene Description` |
| `is_control` | bool | True khi `probe_id` bắt đầu bằng `AFFX` (58 probe) |

Một gen có thể có nhiều probe: 926 probe dùng chung 424 mô tả. Khi đếm "số gen" hoặc diễn
giải danh sách gen (TV4, TV5) phải gộp theo `description`, không theo `probe_id`.

## 3. Tầng cleansed — `data/cleansed/` (sinh bởi `python -m src.quality`)

| File | Nội dung |
|---|---|
| `expression.parquet` | **copy nguyên vẹn** của standardized — không sửa giá trị biểu hiện |
| `samples.parquet` | như trên **+ cột `qc_outlier`** (bool) |
| `quality_report.md` | báo cáo 4 mức chất lượng (mục 2.5) |

`qc_outlier = True` khi mẫu (đo trên log10 sau threshold [100, 16000]) có một trong các dấu hiệu
(modified z-score theo median/MAD, ngưỡng 3.5 — ADR-004):

- tương quan trung vị với các mẫu còn lại thấp bất thường (z < −3.5);
- trung vị hoặc IQR của mẫu lệch hẳn so với các mẫu khác (|z| > 3.5) — tương ứng boxplot theo mẫu;
- trùng lặp với một mẫu khác (tương quan > 0.99).

Kết quả hiện tại: **mẫu 21** bị gắn cờ (tương quan trung vị 0.756 so với trung vị chung 0.832,
z = −4.7). Lý do từng mẫu ghi trong `quality_report.md`; **không xóa mẫu** khi chưa có căn cứ.

## 4. Các phép biến đổi ở tầng curated (công việc 1.2 — TV3)

1. Threshold `[100, 16000]` → 2. `log10` → `X_log10.parquet` (mọi mô hình dùng bản này).
3. Lọc gen (max/min > 5 và max − min > 500) → `X_log10_filtered.parquet` — **chỉ dùng
   cho EDA**, không dùng làm đầu vào mô hình (tránh selection bias, xem mục 2.6 của kế hoạch).
4. Chuẩn hóa từng gen (mean 0, sd 1) phải nằm **trong** `Pipeline` của từng fold CV.

## 5. Tham khảo

- Golub et al. (1999), *Science* 286:531–537 — bảng mẫu và ngưỡng prediction strength.
- `data/cleansed/quality_report.md` — kết quả kiểm tra chất lượng hiện tại.