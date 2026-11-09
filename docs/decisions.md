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

## ADR-009 — Hình F10/F11 vẽ từ `evaluate.py`, chưa chờ `viz.py` của TV6
- **Ngày:** 07/10/2026 · **Người quyết định:** TV2
- **Bối cảnh:** thuyết minh TV2 quy định TV2 bàn giao hình F10 (selection bias) và F11 (ROC +
  ma trận nhầm lẫn) với tên file `F10_selection_bias_{report,slide}.png`,
  `F11_final_model_roc_cm_{report,slide}.png`; kế hoạch lại giao `src/viz.py` cho TV6 (T4 14/10).
- **Quyết định:** `src/evaluate.py` có `plot_selection_bias()` và `plot_roc_cm()`, chạy sau
  `make evaluate`, ghi cả hai bản report (9pt, dpi 220) và slide (17pt, dpi 150) vào
  `results/figures/report|slides/`. Khi TV6 giao `style_guide` + `viz.py` thì chuyển phần
  kiểu dáng sang đó, giữ nguyên tên file và nơi sinh số liệu.
- **F11 bản 1:** vẽ cho pipeline tạm (50 gen → logistic) trên tập test 34 mẫu
  (`original_split`); khi nhóm chốt mô hình cuối (TV3, công việc 2.3) thì gọi lại
  `plot_roc_cm` với dự đoán của mô hình đó — ghi chú này cũng nằm trong `notebooks/08_evaluation.ipynb`.

## ADR-010 — Batch effect kiểm định trong từng lớp; không hiệu chỉnh batch (công việc 1.3)
- **Ngày:** 25/10/2026 · **Người quyết định:** TV1
- **Bối cảnh:** trong train, lớp ≡ tuổi ≡ nguồn (ALL = trẻ em = DFCI, AML = người lớn = CALGB;
  chú thích 13, 23 của bài báo). Kruskal–Wallis theo source trên cả 72 mẫu (PC2 p = 7,7e-07,
  PC3 p = 9,8e-06) lẫn với hiệu ứng lớp nên không dùng để kết luận H6.
- **Quyết định:** kiểm định nguồn / loại mô / St-Jude vs còn lại **trong từng lớp** trên PC1–PC3
  (Kruskal–Wallis, hiệu chỉnh BH, `results/metrics/batch_effect_tests.csv`). Kết quả 0/15 có ý
  nghĩa → không thêm bước hiệu chỉnh batch (ComBat…) vào pipeline.
- **Hệ quả:** báo cáo riêng kết quả test trên AML trẻ em (n = 10), St-Jude (n = 8), PB (n = 10);
  `experiment_config.yaml` giữ nguyên giá trị, thêm chú thích và `tests/test_experiment_config.py`
  (`outer_repeats` phải bằng số seed vì `evaluate.py` lặp theo `seeds`).

## ADR-011 — Style guide và `src/viz.py` (công việc 1.4)
- **Ngày:** 28/10/2026 · **Người quyết định:** TV6
- **Quyết định:** màu/hình dạng/cỡ chữ nằm ở `src/viz.py` (nguồn duy nhất) và mô tả trong `docs/style_guide.md`. Báo cáo: chữ ≥ 10 pt, hình rộng 6,5 in, 300 dpi; slide: chữ ≥ 16 pt, 12 in, 150 dpi.
  Phải vẽ ở **kích thước in thật** (hình 11 in thu về 6,5 in làm chữ 10 pt thành ~6 pt).
- **Màu:** Okabe–Ito. ALL `#0072B2` tròn, AML `#E69F00` tam giác (đúng `src.config.PALETTE`); T-ALL `#332288` vuông; nguồn mẫu và loại mô giữ màu TV1 đã dùng ở F4d.
- **Hình của 1.4:** F2, F2b, F3, F4a (scree), F4b (PCA theo lớp và nhóm con), F4c (PCA/t-SNE/UMAP), F4e (tương quan giữa mẫu). **F04d giữ nguyên là hình batch effect của TV1**; `viz.py` không vẽ lại F04d
  và không vẽ PCA theo nguồn/loại mô (trùng việc 1.3).
- **F2:** vẽ histogram đếm với trục y log trên giá trị MAS4 **gốc** (47% < 100, có số âm) cạnh giá trị sau threshold và log10; density thường bị cột ở sàn 100 chi phối nên không dùng.
- **EDA dùng `X_log10_filtered`** (3 571 gen, lọc trên 72 mẫu, không dùng nhãn) và z-score từng gen; chỉ để khám phá, không để đánh giá mô hình (mục 2.6).
- **Hệ quả:** F10/F11 (TV2) và F01/F04d (TV1) chưa theo style guide (màu, cỡ chữ, kích thước); chuyển sang `viz.py` ở công việc 2.6 (xem ADR-009).

## ADR-012 — Sửa công việc 1.5: bảng chéo theo `sample_id`, SOM qua 10 seed, giữ average làm kết quả chính thức (công việc 1.5)
- **Ngày:** 01/11/2026 · **Người quyết định:** TV1 (rà soát nghiệm thu cuối giai đoạn 1)
- **Bối cảnh:** `notebooks/03_discovery.ipynb` bản 31/10 dựng bảng chéo từ `pd.Series(labels)` không có index trong
  khi `samples` có index `sample_id` 1–72 → nhãn cụm lệch một mẫu, mất 1 mẫu; `cluster_discovery_metrics.csv` và
  F05/F05b/F05c do script nháp `scripts/build_nb03.py` (đã xóa) sinh ra nên không tái lập được; báo cáo đọc nhầm
  "24 of 25 samples" của bài báo thành "24/25 mẫu ALL".
- **Quyết định:**
  - Bảng chéo gắn nhãn cụm theo `sample_id` (hàm `bang_cheo`); notebook tự ghi CSV và cả ba hình (report + slide, theo `src/viz.py`).
  - SOM (`minisom`, σ = 0.5, lr = 0.5, giữ nguyên tham số của TV5) báo cáo trung bình ± độ lệch chuẩn qua 10 seed;
    **không** dò tham số SOM theo ARI vì như vậy là dùng nhãn để chọn mô hình không giám sát.
  - Hierarchical: báo cáo **cả** average (kế hoạch chỉ định; ARI −0.013 do mẫu 21 thành cụm 1 mẫu) **và** complete
    (ARI 0.690); không thay average bằng complete trong kết luận.
  - Kết luận H5 đổi từ "ủng hộ mạnh" thành "ủng hộ ở mức ALL/AML, không ủng hộ ở mức B-ALL/T-ALL".
- **Hệ quả:** xóa `F5_clustermap_{report,slide}.png` (bản trùng cũ của F05). Cập nhật `docs/report/3.2_class_discovery.md`, `docs/handover/1.5.md`.

## ADR-014 — Đổi hướng giai đoạn 2: quản trị dữ liệu và dashboard, thu gọn học máy
- **Ngày:** 09/11/2026 · **Người đề xuất:** TV1 · **Người quyết định:** giảng viên duyệt, TV1
- **Bối cảnh:** giảng viên góp ý đề tài thiên về học máy; học phần Quản trị dữ liệu và trực quan hóa
  (IT5425) yêu cầu đẩy mạnh quản trị dữ liệu và trực quan hóa có tính thực tế.
- **Quyết định:**
  - Bỏ thí nghiệm giai thừa 252 cấu hình và giả thuyết H2, H4; xóa `src/features.py`, `src/models.py`
    (còn trống) và lệnh `make experiments`. `config/experiment_config.yaml` giữ lại làm tài liệu giai
    đoạn 1, giai đoạn 2 không dùng.
  - Thêm: nhập theo lô (`src/ingest.py`, TV2), kho DuckDB (`src/warehouse.py`, `sql/`, TV3), khám phá
    gen (`src/gene_explorer.py`, TV4), hồ sơ bệnh nhân (`src/patient.py`, TV1), thiết kế lại hình của
    bài báo (TV5), dashboard Streamlit bắt buộc chạy được (`app/`, TV6). Thêm `duckdb==1.5.6`.
  - Giữ mô hình Golub weighted voting (TV1) và thí nghiệm selection bias (TV2).
  - Quy ước dữ liệu: 38 mẫu huấn luyện là dữ liệu tham chiếu; 34 mẫu kiểm tra đóng vai bệnh nhân mới,
    không dùng để huấn luyện, chọn gen hay tính tham số.
- **Quy tắc tránh conflict khi làm song song:**
  - Mỗi file một chủ, không sửa file của người khác; không sửa notebook của người khác.
  - `src/viz.py` chỉ TV6 sửa; hàm vẽ mới để trong file riêng `src/viz_<chủ đề>.py`.
  - `Makefile`, `run_all.py`, `environment.yml`, `requirements.txt`: báo nhóm trước khi sửa.
  - Số ADR giữ trước: ADR-013 cho nhánh `2.4-evaluate` (TV2); sau đó TV1 020–029, TV2 030–039,
    TV3 040–049, TV4 050–059, TV5 060–069, TV6 070–079.
  - `tests/test_interfaces.py` khóa tên hàm dùng chung; đổi tên thì ghi ADR và sửa test cùng lúc.
- **Hệ quả:** kế hoạch, phân công và hạn giai đoạn 2 nằm trong tài liệu "Kế hoạch giai đoạn 2 (hướng mới)".

## Nhật ký nhanh (mục 5.2 kế hoạch)

| Ngày | Quyết định | Lý do | Người quyết định |
|---|---|---|---|
| 05/10/2026 | Dùng Kaggle (biểu hiện gen + nhãn) + 6 cột mô tả của OpenIntro | Kaggle thiếu metadata; bản biểu hiện OpenIntro đã chuẩn hóa nên không dùng | TV2 |
| 05/10/2026 | Cấu trúc tầng raw → standardized → cleansed → curated → sandbox | Mục 2.2, Bảng 2.5 | Cả nhóm |
| 06/10/2026 | Viết lại kiểm tra chất lượng theo đúng 4 mức Bảng 2.9; mẫu 21 gắn cờ `qc_outlier` | ADR-004 | TV2 |
| 06/10/2026 | Ghép theo left join vào bảng mẫu Kaggle, so tập `sample_id` của 3 nguồn trước khi ghép | Mục 2.1.2 bước 5–6 | TV2 |
| 06/10/2026 | Kiểm tra FAIL dừng pipeline; cố định phiên bản thư viện | ADR-006, ADR-007 | TV2 |
| 06/10/2026 | `evaluate.py` bản 1: 4 sơ đồ, Wilson, McNemar, selection bias; thêm khóa `predictions` | ADR-008 | TV2 |
| 07/10/2026 | `make evaluate` sinh thêm hình F10/F11 (bản 1) và `notebooks/08_evaluation.ipynb`; 100 lần hoán vị xác nhận H3 (CV đúng 0.50 vs CV sai 0.81) | ADR-009 | TV2 |
| 25/10/2026 | 1.3: H6 kiểm định trong từng lớp (0/15 có ý nghĩa sau BH), không hiệu chỉnh batch; chú thích + test cho `experiment_config.yaml` | ADR-010 | TV1 |
| 28/10/2026 | 1.4: `src/viz.py` (F2, F2b, F3, F4a–c, F4e), `style_guide.md`, `notebooks/01_eda.ipynb`, `tests/test_viz.py`; F04d giữ nguyên của TV1 | ADR-011 | TV6 |
| 01/11/2026 | 1.5: sửa bảng chéo lệch `sample_id`, SOM qua 10 seed, notebook tự sinh CSV + F05/F05b/F05c; H5 chỉ ủng hộ ở mức ALL/AML | ADR-012 | TV1 |
| 09/11/2026 | Đổi hướng giai đoạn 2 sang quản trị dữ liệu và dashboard; dựng khung file, lệnh `make`, giữ trước số ADR | ADR-014 | TV1 (giảng viên duyệt) |
