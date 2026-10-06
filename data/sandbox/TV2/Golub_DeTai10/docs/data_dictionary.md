# Data Dictionary - Golub 1999 (TV2 - mục 2.7, Bảng 2.6)

Quy ước: mọi file trong `data/` sinh tự động từ `data/raw/` bằng `make data`
(không ai sửa tay). Tổng quan kiến trúc tầng: mục 2.2 kế hoạch.

## 1. Tầng raw (`data/raw/`, chỉ đọc)

| File | Cột | Kiểu/đơn vị | Nguồn | Ghi chú |
|---|---|---|---|---|
| `kaggle/data_set_ALL_AML_train.csv` | `Gene Description`; `Gene Accession Number`; cặp `k` + `k call` cho k = patient 1..38 | int (MAS4), `call`: P/A/M | Kaggle *Golub leukemia* | patient 1..38 = train |
| `kaggle/data_set_ALL_AML_independent.csv` | như trên, patient 39..72 | int | Kaggle | patient 39..72 = test |
| `kaggle/actual.csv` | `patient`, `cancer` (ALL/AML) | int, str | Kaggle | **Nguồn nhãn chính** |
| `openintro/golub_metadata.csv` | `Samples, BM.PB, Gender, Source, tissue.mf, cancer` (+7129 gen) | xem bên dưới | `dataset_more.csv` (ADR-001) | metadata mẫu |
| `SOURCES.md` | — | — | TV2 | link + MD5 từng file |

Ghi chú dữ liệu biểu hiện (giá trị thô, kiểu int):
- Thuật toán MAS4 của Affymetrix; bộ dữ liệu có giá trị ÂM (biểu hiện sau trừ nền).
- Ngưỡng reference `[100, 16000]`: <100 hoặc >16000 gán cụt ở bước tiền xử lý (mục 2.6).
- Nhóm probe điều khiển AFFX-... (58 probe) làm nhiễu, đánh dấu `is_control`.

## 2. Tầng standardized (`data/standardized/`)

### `expression.parquet`
- 72 hàng × 7129 cột; index = `sample_id` (int, = mã bệnh nhân 1..72).
- Cột = probe id (Affymetrix accession, VD `AFFX-BioB-5_at`, `X95735_at`).
- Giá trị int, chưa chuyển đổi → có thể âm; không có NaN.

### `calls.parquet`
- 72 × 7129, giá trị `P`/`A`/`M` (present/absent/marginal), có thể NA.
- Chỉ để tham khảo; KHÔNG dùng trong mô hình.

### `samples.parquet` (Dim_Sample) - 6 cột, index = `sample_id`

| Cột | Mô tả | Giá trị | Ghi chú |
|---|---|---|---|
| `split` | tập huấn luyện/độc lập | `train` / `test` | theo file Kaggle (1..38 / 39..72) |
| `class` | nhãn chính (Kaggle) | `ALL`, `AML` | ALL 47, AML 25 |
| `subtype` | nhóm miễn dịch (metadata) | `B-ALL`, `T-ALL`, `AML` | 38/9/25; khớp nhãn 72/72 |
| `tissue` | loại mô | `BM` (tủy xương), `PB` (máu ngoại vi) | train: 38 BM; test: 24 BM + 10 PB |
| `source` | ngân hàng mẫu | `DFCI`, `CALGB`, `CCG`, `St-Jude` | xem Bảng 2.4 |
| `sex` | giới tính | `M`, `F`, NaN | **23/72 thiếu** (metadata gốc, không tự điền) |

Phân bố chuẩn (Bảng 2.4): train 27 ALL (DFCI) + 11 AML (CALGB);
test 20 ALL (17 DFCI + 3 St-Jude) + 14 AML (4 CALGB + 5 St-Jude + 5 CCG).

### `genes.parquet` (Dim_Gene)
- `probe_id` (str, khóa), `description` (mô tả gen từ Kaggle), `is_control` (bool, AFFX).

## 3. Tầng cleansed (`data/cleansed/`)
- `expression.parquet`, `samples.parquet` (thêm cột `qc_outlier` bool), `quality_report.md`
- `qc_outlier` = cờ nghi chip lỗi (z-score>4 ở >50 gen, mức 4); KHÔNG xóa dữ liệu.

## 4. Phép biến đổi (sẽ làm ở tầng curated - TV3, mục 2.6)
1. Threshold `[100, 16000]` → 2. `log10` → `X_log10.parquet` (mọi mô hình).
3. Lọc gen (max/min>5 và max−min>500) → `X_log10_filtered.parquet` (chỉ EDA).
4. Chuẩn hóa mỗi gen (mean 0, sd 1) tùy mô hình — LUÔN nằm trong Pipeline của từng fold.

## 5. Tham khảo
- Golub et al. (1999), *Science* 286:531-537. PDF: `docs/nguon/Golub_1999_Molecular_Classification_of_Cancer.pdf`.
- Các quyết định về nguồn/ánh xạ cột: `docs/decisions.md`.
- Kiểm tra chất lượng 4 mức: `data/cleansed/quality_report.md`.