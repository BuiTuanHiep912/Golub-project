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

## ADR-013 — Sửa 3 lỗi của `src/evaluate.py` trước khi dùng số liệu cho báo cáo (công việc 2.4)
- **Ngày:** 04/11/2026 · **Người quyết định:** TV2
- **`mcnemar_test`** từng so ba đầu vào theo vị trí (bỏ chỉ mục `sample_id`): hai mô hình cùng đúng
  34/34 nhưng một bên xếp ngược thì báo p = 0.0. Nay `pd.Series` được xếp lại theo chỉ mục của
  `y_true`; khác tập mẫu hoặc khác độ dài thì báo lỗi.
- **F11 / `confusion_counts`**: ma trận nhầm lẫn từng dùng `labels=[0, 1]` nên bỏ mất dự đoán
  "uncertain" của weighted voting (2 mẫu "uncertain" thì chỉ cộng ra 32/34). Nay có thêm cột
  "uncertain" khi mô hình trả về nó; tiêu đề lấy đúng số mẫu thay vì ghi cứng 34.
- **Dòng `nested_cv` của `main()` và notebook 08** từng chạy không có lưới nên không lồng vòng nào
  (repeated CV, k = 50, C = 1 cố định). Nay vòng trong (5 fold) chọn k theo nhân tố B của
  `experiment_config.yaml` và C ∈ {0.01, 0.1, 1, 10} theo mục 3.5.2. Kết quả: 0.968 ± 0.055 (trước
  0.960 ± 0.049); `make evaluate` lâu hơn (~4,5 phút so với ~45 giây).
- **Không đổi:** "CV đúng" của thí nghiệm selection bias vẫn là `nested_cv` không lưới (k cố định),
  để 200 lần chạy không nặng gấp hàng chục lần; bảng selection bias giữ nguyên từng chữ số.
- **Kiểm thử:** 6 test hồi quy trong `tests/test_evaluate.py` + bộ golden-file 19 testcase
  (`tests/test_golden_evaluate.py`, `tests/input|expect/`; `tests/actual/` trong `.gitignore`).

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

## ADR-020 — Lựa chọn cài đặt mô hình Golub weighted voting (công việc 2.1)
- **Ngày:** 15/11/2026 · **Người quyết định:** TV1
- **Bối cảnh:** bài báo (thân bài tr. 532, ghi chú 18–21) không nêu đủ chi tiết để cài đặt duy nhất một cách; các lựa
  chọn nhỏ có thể đổi kết luận: mẫu 66 có PS = 0,30004, chỉ vừa qua ngưỡng 0,3. Công thức và ví dụ tính tay:
  `docs/report/5.1_mo_hinh_golub.md`.
- **Quyết định** (`src/golub_wv.py`):
  - **D1 — Chuẩn hóa:** z-score từng gen theo mean/SD của tập train trước khi bỏ phiếu, đúng ghi chú 19
    (`standardize=True`, mặc định). Thực hiện trong `fit`, nên mỗi lần lặp LOOCV tự tính lại trên phần huấn luyện.
    z-score không đổi SNR, nên không đổi tập gen chọn được; chỉ đổi độ lớn phiếu. `standardize=False` chỉ dùng để
    đối chiếu với ví dụ tính tay.
  - **D2 — Độ lệch chuẩn:** chia n − 1 (`ddof=1`, mặc định của pandas và hàm `sd()` của R); bài báo không nêu.
  - **D3 — Số gen lẻ:** báo `ValueError`, vì ghi chú 18 định nghĩa theo n/2 và thí nghiệm lưới k (có k = 5) đã bỏ
    theo ADR-014. Thay cho ghi chú "TV1 chốt ở 2.1" trong `config/experiment_config.yaml`.
  - **Gen hằng:** gen có SD = 0 trên train (1050 gen ở ngưỡng sàn) giữ SD = 1 khi chuẩn hóa; gen có σ_ALL + σ_AML = 0
    được gán SNR = 0 nên không bao giờ được chọn.
  - **Ngưỡng:** PS < `PS_THRESHOLD` (0,3, `src/config.py`) thì trả `"uncertain"`; PS đúng bằng 0,3 tính là chắc chắn.
  - **`predict_proba`:** trả tỷ lệ phiếu V_ALL / (V_ALL + V_AML), không phải xác suất hiệu chỉnh; ghi rõ trong docstring.
- **Hệ quả:** LOOCV 38 mẫu: 36 dự đoán chắc chắn, 36 đúng, khớp bài báo. Tập test 34 mẫu: 33 chắc chắn, 32 đúng,
  PS trung vị 0,73; bài báo 29/29 (`notebooks/05_golub_reproduction.ipynb`). Bản 15/11 ghi "32 chắc chắn, 31 đúng" là
  cộng nhầm từ bảng nhầm lẫn; số đúng là 33/32. Dự đoán sai duy nhất là mẫu 66 (PS = 0,30004). Chưa đổi lựa chọn nào.
- **Cập nhật 23/11 (tuần 2):** `patient_profile()` dùng mô hình 38 mẫu cho mẫu test và mẫu mới; với mẫu train dùng
  mô hình bỏ chính mẫu đó (37 mẫu) để PS trùng LOOCV. Mẫu mới nhận ở thang gốc, đi qua `threshold_log10` như tầng
  curated. Bệnh nhân tương tự: tương quan Pearson trên 50 gen bỏ phiếu đã z-score, chỉ so với 38 mẫu tham chiếu.

## ADR-050 — Quy ước của `src/features.py` (công việc 2.2, bản 1; đã thay thế bởi ADR-051)
- **Ngày:** 01/11/2026 · **Người quyết định:** TV4
- **Giao diện:** 6 bộ chọn `snr, ttest, lasso, enet, rf, svm_rfe` (đúng tên trong `experiment_config.yaml`),
  tạo bằng `make_selector(tên, k, **params)`. Mỗi bộ chọn kế thừa `SelectorMixin` + `BaseEstimator` để
  `clone()` được và để `evaluate.py` cắt Pipeline tại bước chọn gen (`wrong_cv`). Đặt ở bước tên `select`:
  `GeneFilter → StandardScaler → select → bộ phân loại`. Nhãn nhận `"ALL"/"AML"` hoặc 0/1 (1 = AML).
- **Luôn trả về đúng k gen** (k > số gen thì giữ tất cả), để mỗi ô của thí nghiệm giai thừa so cùng số gen.
- **snr, ttest chọn cân bằng hai phía** như Golub: ⌈k/2⌉ gen cao ở ALL + ⌊k/2⌋ gen cao ở AML; k = 5 → 3 ALL + 2 AML
  (ALL không đồng nhất B/T-ALL nên lấy phần dư). Lấy top-k theo |SNR| chung sẽ lệch về AML (32/50, xem 1.6).
  `golub_wv` (TV1, 2.1) nên dùng cùng quy tắc làm tròn.
- **ttest:** xếp theo p của Welch tăng dần, hòa thì |t| lớn hơn; Benjamini–Hochberg chỉ ghi `qvalues_`, không đổi thứ tự.
- **lasso, enet:** logistic có phạt, đi theo lưới C = 10⁻²…10² (9 mức) từ phạt mạnh đến yếu, dừng khi có ≥ k hệ số khác 0.
  Xếp gen: vào mô hình sớm hơn → |hệ số| → |t| Welch. LASSO chỉ chọn tối đa ~n gen khác 0, nên với k lớn (100, 200)
  phần còn lại xếp theo |t| — cần nhắc khi so sánh kết quả lasso ở k lớn. enet: `l1_ratio = 0.5`, solver saga.
  Dùng `l1_ratio` thay cho `penalty` (đã bỏ ở scikit-learn 1.8; nhóm ghim 1.9.1, ADR-007).
- **rf:** 500 cây, độ quan trọng Gini; gen điểm 0 xếp theo |t|. **svm_rfe:** `LinearSVC(C=1)`, mỗi vòng bỏ 10% số gen
  ban đầu (`step = 0.1`) để chạy đủ nhanh cho 252 cấu hình.
- **Gen hằng số** trong dữ liệu được fit (σ = 0, hay gặp ở mức sàn trong một fold): SNR = 0, t = 0, p = 1.
- **Thời gian** (38 mẫu train, 3032 gen sau `GeneFilter`): snr/ttest/lasso/svm_rfe < 0,15 s; enet ≤ 1,2 s; rf ~0,5 s.
- **Kiểm tra:** `tests/test_features.py` (46 test, dữ liệu tổng hợp): đúng k với mọi k trong YAML, tìm lại gen có tín hiệu,
  cân bằng hai phía, gen hằng số, nhãn chuỗi/số, `clone` + seed, không rò rỉ, chạy được trong `evaluate()` (nested_cv, wrong_cv).
  Bộ chọn snr trên 38 mẫu train trùng 19/20 gen đầu với `gene_ranking.csv` (1.6); lệch 1 gen do lọc gen trên train (3032 gen)
  thay vì trên 72 mẫu (3541 gen).

## ADR-051 — Công việc 2.2 đổi sang Khám phá gen; quy ước của `src/gene_explorer.py`
- **Ngày:** 12/11/2026 · **Người quyết định:** TV4 (theo ADR-014 của TV1 và *Kế hoạch giai đoạn 2 (hướng mới)*)
- **Bối cảnh:** kế hoạch mới chuyển trọng tâm sang quản trị dữ liệu và trực quan hóa. Thí nghiệm giai thừa 252 cấu hình,
  giả thuyết H2, H4 và hình F7 (UpSet) bị bỏ. Công việc 2.2 đổi từ *Chọn biến và độ ổn định* sang *Khám phá gen*,
  cung cấp logic cho màn hình Khám phá gen của dashboard (TV6, 2.6).
- **Thay thế ADR-050:** `src/features.py` và `tests/test_features.py` bị xóa theo ADR-014; bản đã làm không đưa vào repo,
  ADR-050 chỉ còn là ghi chép thiết kế. Không làm bootstrap, Jaccard, `final_genes.csv`, F7 cũ. Hình F7 mới (sơ đồ truy vết) thuộc TV3.
  Đánh số 050–051 theo dải số dành cho TV4 trong ADR-014 (ADR-013 dành cho TV2).
- **Giao diện** (`src/gene_explorer.py`), TV6 chỉ gọi các hàm này, không tự tính lại:
  `find_gene(query)`, `gene_values(probe_id)`, `gene_stats(probe_id, by)`, `disease_or_source(probe_id)`,
  `ranking_table(top_n, direction, min_abs_snr)`, `collapse_by_symbol(df)`, `annotate(df)`, `clear_cache()`.
  `find_gene` và `gene_stats` là tên đã khóa trong `tests/test_interfaces.py`; `gene_stats(probe_id)` mặc định nhóm theo
  `class, source, tissue` như giao diện thống nhất. Hàm vẽ của TV4 đặt trong `src/viz_gene.py`, không sửa `src/viz.py`.
- **Dữ liệu:** đọc qua `load_golub("curated")` (log10). Xếp hạng lấy nguyên `gene_ranking.csv` của 1.6, tức chỉ tính trên
  38 mẫu train; bảng trả về mang `attrs["note"]` để giao diện hiện cảnh báo. Biểu đồ và thống kê mô tả dùng cả 72 mẫu,
  chỉ để hiển thị, không dùng chọn gen hay ước lượng tham số.
- **`disease_or_source`:** luôn trả về đủ `n_ALL, n_AML, mean_ALL, mean_AML, diff` theo từng nguồn. Ngưỡng
  `MIN_N_PER_GROUP = 3` chỉ dùng cho cờ `comparable`, không lọc số liệu. Với dữ liệu Golub chỉ St-Jude (3 ALL + 5 AML,
  tập test) có `comparable = True`; DFCI chỉ có ALL, CALGB và CCG chỉ có AML.
- **`collapse_by_symbol`:** gộp probe cùng gen, giữ probe có |SNR| lớn nhất, thêm `n_probes`. Khóa gộp theo thứ tự
  `gene_symbol` → `description` → `probe_id` (bỏ qua giá trị rỗng). Tiêu chí |SNR| chỉ dùng số liệu train nên không rò rỉ.
  **Còn mở:** cần TV1 (đối chiếu 50 gen Golub theo probe), TV5 (chú giải theo gen), TV6 (bảng hiện theo gen hay probe) xác nhận.
- **Ký hiệu gen:** lấy bằng `split_symbol` của `src/gene_catalog.py`; rỗng khi mô tả không có dạng `KÝHIỆU tên`
  (4206/7129 probe, 1971/3541 dòng xếp hạng, ví dụ Zyxin). `find_gene` vẫn tìm được qua chuỗi trong mô tả.
- **Hợp đồng với TV5:** `results/gene_lists/gene_annotations.csv` có đúng các cột
  `probe_id, gene_symbol, function, cell_lineage, source_url, verified` (`verified` ∈ `yes`/`no`, mỗi dòng có liên kết
  NCBI Gene hoặc GeneCards). Thiếu cột thì `annotate` báo lỗi; chưa có file thì dùng `description` và ghi
  `function_source = "description"`. Ký hiệu rỗng trong bảng được điền từ file chú giải.
- **Bộ nhớ đệm:** dữ liệu đọc một lần, khóa theo đường dẫn và thời điểm sửa file (sinh lại file thì tự đọc lại).
  Lần gọi đầu 0,2–0,7 s, các lần sau < 0,02 s (trước khi có bộ nhớ đệm: 0,5–1 s mỗi lần).
- **Kiểm tra:** `tests/test_gene_explorer.py` (30 test, dữ liệu tổng hợp cùng cỡ nhóm Golub + 1 test trên dữ liệu thật).

## ADR-052 — Khám phá gen bản 2: độ tin cậy, độ vững, lô mới, đọc kho DuckDB, `src/viz_gene.py`
- **Ngày:** 17/11/2026 · **Người quyết định:** TV4 (công việc 2.2, tuần 2)
- **Bối cảnh:** màn hình Khám phá gen mới trả lời "gen này biểu hiện thế nào", chưa trả lời được câu chính của nhà
  nghiên cứu: số đo có tin được không, khác biệt có vững không, và khi có lô mẫu mới thì có gì bất thường.
- **Hàm mới trong `src/gene_explorer.py`** (không đổi tên hàm đã khóa):
  - `gene_reliability(probe_id)`: theo **từng lớp**, tỉ lệ call P/M/A của chip (`standardized/calls.parquet`) và tỉ lệ
    mẫu ở mức sàn (log10 = 2). Phải tính theo lớp vì gen dấu ấn tốt thường tắt ở một lớp: CD33 có P ở 19% ALL, 92% AML;
    tính chung 72 mẫu sẽ trông như tín hiệu yếu. Kiểu: `on_off`, `level`, `weak` (lớp "có mặt" khi ≥ 50% call P).
  - `robustness(probe_id)`: chênh lệch ALL − AML qua 7 phép so sánh (huấn luyện làm mốc, 72 mẫu, bỏ mẫu gắn cờ QC,
    B-ALL, T-ALL, chỉ tủy xương, trong St-Jude) kèm `fold_change` = 10^diff và cờ cùng chiều. Chỉ mô tả.
  - `marker_consistency(source | sample_ids)`: lô **có** nhãn. Gen dấu ấn = 25 + 25 gen hạng cao nhất mỗi phía (như
    Golub; top 50 theo hạng chung cho 44/50 thay vì 46/50). `only_test=True` mặc định để không so tập huấn luyện với chính nó.
  - `batch_anomalies(source | sample_ids)`: lô **chưa** nhãn. Mẫu có gen dấu ấn ngoài [min, max] của 38 mẫu huấn
    luyện; mức lệch tính theo độ lệch chuẩn của gen. Không dùng nhãn của lô (có test khóa điều này).
  - `ranking_table(top_n=None)` lấy toàn bộ gen; `gene_values` trả thêm `qc_outlier`.
- **Ngưỡng chênh lệch tối thiểu `MIN_DIFF_LOG10 = 0,1`** (≈ 1,26 lần): chênh lệch trong lô nhỏ hơn mức này xếp
  "Chênh lệch nhỏ", không tính là giữ hay đảo chiều. Lý do: 4 gen "đảo chiều" theo dấu ở St-Jude đều sát mức sàn,
  chênh 0,00–0,05 (TCF12 bằng đúng 0); ngược lại có gen "giữ chiều" chỉ chênh 0,002. Kết quả St-Jude: 46/50 cùng dấu
  (khớp 1.6); với ngưỡng: **41 giữ chiều, 0 đảo chiều, 9 quá nhỏ**. Không gen dấu ấn nào đảo chiều rõ trong cùng một
  phòng lab, ủng hộ khác biệt do bệnh; giới hạn: St-Jude chỉ có 3 ALL, 5 AML.
- **Đọc kho DuckDB của TV3** (`fact_expression.value_log10`, `dim_sample`) khi `data/warehouse/golub.duckdb` tồn tại,
  không thì đọc `load_golub("curated")`; `data_source()` cho biết đang dùng nguồn nào, `USE_WAREHOUSE = False` để ép
  đọc Parquet. Đã kiểm tra: X và bảng mẫu đọc từ kho **giống hệt** tầng curated (test dựng kho vào thư mục tạm).
  `WAREHOUSE_PATH` khai lại trong module (không import `src.warehouse` để module không bắt buộc cài duckdb) và có test
  kiểm tra trùng với hằng số của TV3.
- **`src/viz_gene.py`** (file riêng theo ADR-014, dùng màu/hình điểm/cỡ chữ của `src/viz.py`): volcano, biểu hiện một
  gen (chấm đặc huấn luyện, chấm rỗng kiểm tra, vòng đen mẫu gắn cờ, ghi n, tô nền ô đủ hai lớp), nhất quán của lô,
  thanh SNR hai phía. Mỗi hình có bản Altair (`*_chart`, cho dashboard) và bản matplotlib (`plot_*`, mode
  report/slide). `python -m src.viz_gene` ghi `gene_volcano`, `gene_expression_CD33`, `gene_consistency_StJude`
  vào `results/figures/{report,slides}/`. Chưa thêm vào `Makefile`/`run_all.py` (file chung, cần báo nhóm).
- **Ranh giới, không chồng việc:** kiểm tra định dạng và chất lượng chip của lô mới là của TV2 (2.4); vị trí một bệnh
  nhân trên PCA và gen bỏ phiếu là của TV1 (2.1); đặt hình lên trang là của TV6 (2.6). TV4 chỉ cung cấp hàm.
- **Kiểm tra:** `tests/test_gene_explorer.py` 45 test (thêm dữ liệu giả có T-ALL, mẫu gắn cờ, call P/A, mẫu CCG bất
  thường; khóa 46/50 và 0 đảo chiều trên dữ liệu thật), `tests/test_viz_gene.py` 14 test. Toàn bộ repo 178 test đạt.
- **Bổ sung 18/11 (độ bất định cho `marker_consistency`):**
  - *Vì sao vẫn dùng hiệu trung bình, không dùng SNR trong lô:* hàm chỉ xét chiều, mà dấu SNR luôn trùng dấu hiệu trung
    bình (mẫu số dương). σ tính từ 3–5 mẫu của lô không ổn định và gần 0 ở mức sàn, làm SNR bị thổi phồng đúng ở các gen
    cần loại. Hiệu log10 có đơn vị dễ đọc (0,1 ≈ 1,26 lần).
  - *Thêm kiểm định hoán vị trong lô* (`attrs["p_perm"]`, `attrs["n_perm"]`): xáo nhãn các mẫu của lô (giữ số ALL/AML),
    đếm lại số gen cùng dấu với tập huấn luyện; p = tỉ lệ cách xếp cho số gen ≥ nhãn thật. Kiểm định cho **cả bộ** gen,
    không từng gen (3 mẫu thì kiểm định từng gen gần như vô nghĩa), và giữ tương quan giữa các gen. Lô ít mẫu thì duyệt đủ
    (≤ 5000 cách), lô lớn thì xáo 2000 lần với seed cố định. **St-Jude: duyệt đủ C(8,3) = 56 cách, p = 2/56 ≈ 0,036**
    (nhỏ nhất có thể là 1/56 ≈ 0,018); toàn bộ 34 mẫu kiểm tra: 50/50 cùng dấu, p ≈ 0,002. Với 8 mẫu, p không thể nhỏ
    hơn 0,018, nên kết quả St-Jude chỉ ở mức ủng hộ vừa phải.
  - *Thêm cột `batch_diff_sd`* = `batch_diff` / độ lệch chuẩn gộp trong lớp của gen trên 38 mẫu huấn luyện, để so mức chênh
    giữa các gen theo cùng đơn vị. σ lấy từ tập huấn luyện vì ổn định hơn σ của lô. `status` vẫn theo `MIN_DIFF_LOG10`
    để không đổi số liệu đã báo cáo; 9 gen "Chênh lệch nhỏ" ở St-Jude đều có |batch_diff_sd| ≤ 0,58.
  - Thêm 3 test (liệt kê cách xếp, kiểm định trên dữ liệu có tín hiệu và ngược chiều, cột chuẩn hóa) và khóa p = 2/56 trên
    dữ liệu thật. Toàn bộ repo 181 test đạt.

## ADR-040 — Lược đồ kho DuckDB `golub.duckdb` (công việc 2.3, bản 1)
- **Ngày:** 15/11/2026 · **Người quyết định:** TV3 (theo ADR-014 và *Kế hoạch giai đoạn 2 (hướng mới)*)
- **Bối cảnh:** TV1, TV2, TV4, TV6 cần một nơi truy vấn chung thay vì tự đọc Parquet; lược đồ hình sao ở
  Bảng 2.7 mới chỉ có trên giấy.
- **Quyết định:** `src/warehouse.py` dựng `data/warehouse/golub.duckdb` (không lên Git) từ `load_golub("cleansed")`,
  `load_golub("curated")` và `standardized/genes.parquet`:
  - `fact_expression(sample_id, probe_id, value_raw, value_log10)`: dạng dài, 72 × 7129 = 513 288 dòng. Giữ cả giá trị
    MAS4 gốc (tầng cleansed, có âm) và giá trị log10 (tầng curated) để mọi con số đối chiếu được với tầng gốc.
  - `dim_sample(sample_id, split, class, subtype, tissue, source, sex, qc_outlier, batch_id)`, khóa chính `sample_id`.
  - `dim_gene(probe_id, description, is_control)`, khóa chính `probe_id`.
  - `dim_batch(batch_id, source, n_samples, n_train, n_test, n_all, n_aml, n_bm, n_pb)`: **mỗi phòng lab là một lô,
    `batch_id` = tên nguồn** (`DFCI`, `CALGB`, `St-Jude`, `CCG`, khớp `BATCH_SOURCES` của `src/ingest.py`).
  - `fact_qc`: nạp nguyên kết quả `ingest_batches()` của TV2; **TV2 giữ `batch_id` = tên nguồn** để nối được với
    `dim_batch`. Khi `ingest_batches()` chưa cài đặt thì bỏ qua bảng này và in thông báo, các bảng khác vẫn dựng.
- **Cách ghi:** ghi ra file tạm rồi thay file cũ, lỗi giữa chừng không để lại kho dở dang. Dựng lại mất khoảng 4 giây.
- **Truy vấn OLAP:** mỗi câu hỏi nghiệp vụ là một file `sql/NN_<chủ đề>.sql`, có chú thích câu hỏi ở đầu file.
- **Kiểm tra:** `tests/test_warehouse.py` dựng kho vào thư mục tạm, đối chiếu Bảng 2.4, khóa ngoại, khoảng giá trị
  và số giá trị âm (147 174).
- **Đã sửa bởi ADR-041:** `dim_batch`, `fact_qc` và `batch_id` theo `src/ingest.py` của TV2.

## ADR-041 — Kho bản 2: `dim_batch`, `fact_qc` lấy từ `src/ingest.py`, chỉ nạp mẫu của lô đạt (công việc 2.3)
- **Ngày:** 23/11/2026 · **Người quyết định:** TV3
- **Bối cảnh:** TV2 cài đặt `src/ingest.py` (ADR-030) với `batch_id` dạng `B1-DFCI`, `B2-CALGB`, `B3-St-Jude`, `B4-CCG`,
  và tách hai đầu ra: `batches.parquet` (một hàng mỗi lô) và `qc_samples.parquet` (một hàng mỗi mẫu). Quy ước
  `batch_id` = tên nguồn và `fact_qc` theo lô của ADR-040 không còn khớp.
- **Quyết định (thay phần tương ứng của ADR-040):**
  - `dim_batch` = nguyên `batches.parquet` (có `status`, `reason`, `warnings`, chỉ số chất lượng theo lô); bỏ các cột
    tự tính `n_all, n_aml, n_bm, n_pb` của bản 1. Khóa chính `batch_id` theo TV2.
  - `fact_qc` = nguyên `qc_samples.parquet`: **một hàng mỗi mẫu**, nối với `dim_sample` qua `(sample_id, batch_id)`.
  - `log_ingest` = `ingest_log.csv` (nhật ký mọi lần `make ingest`), có khi file tồn tại; dùng cho sơ đồ truy vết.
  - `dim_sample.batch_id` lấy từ `fact_qc`, không tự suy từ `source`.
  - **Chỉ mẫu thuộc lô `passed` mới vào `dim_sample` và `fact_expression`.** Lô bị chặn vẫn nằm trong `dim_batch`
    và `fact_qc` để màn hình Chất lượng dữ liệu hiện lý do chặn.
  - Đọc file của `make ingest`; nếu chưa chạy thì gọi `ingest_batches()`, `sample_quality()` tại chỗ (không ghi
    file, không ghi nhật ký). Tên file lấy từ hằng `BATCHES_FILE`, `QC_FILE`, `LOG_FILE` của `src/ingest.py`.
- **Thứ tự chạy:** `make data` → `make ingest` → `make warehouse` (đúng thứ tự của `make all`).
- **Kiểm tra:** `tests/test_warehouse.py` thêm 2 test: khóa ngoại `fact_qc`–`dim_sample`, cờ `qc_outlier` của TV2 trùng
  tầng cleansed (chỉ mẫu 21), và giả lập lô B4-CCG bị chặn thì 5 mẫu CCG không vào kho.

## ADR-030 — Nhập theo lô và kiểm tra file mẫu mới (`src/ingest.py`, công việc 2.4)
- **Ngày:** 12/11/2026 · **Người quyết định:** TV2
- **Bối cảnh:** nhánh `2.4-ingest` (commit `7717555`) có `ingest_lots()` chia "lô" theo 4 file raw và
  `validate_new_sample(sample: dict)` trả `{valid, errors, warnings, normalized}` — lệch giao diện khóa
  ở ADR-014 (`ingest_batches()`, `validate_new_sample(path)` → `{ok, errors, warnings, values}`),
  `tests/test_interfaces.py` báo `src.ingest thiếu ['ingest_batches']`. Viết lại trên nhánh `2.4-ingest-v2`.
- **Quyết định:**
  - Lô = mọi mẫu của một phòng lab (`source`): B1-DFCI, B2-CALGB, B3-St-Jude, B4-CCG, tách từ tầng
    standardized. Nguồn lạ hoặc thiếu vẫn thành một lô riêng để bị chặn, không lặng lẽ bỏ qua.
  - **Chặn lô** khi vi phạm ràng buộc cấu trúc/quan hệ: thiếu/thừa/trùng probe, ô trống, giá trị không
    phải số, `sample_id` trống/trùng/đã thuộc lô khác, metadata bắt buộc thiếu hoặc ngoài danh mục, lô
    nhiều nguồn, `class` lệch `subtype`, call ngoài P/A/M. Chặn một lô không ảnh hưởng lô khác.
  - **Không chặn** theo chỉ số chất lượng (tương quan, tỷ lệ P, `qc_outlier`): chỉ báo cáo, cùng tinh
    thần ADR-004. Quy tắc gắn cờ theo mẫu tách thành `src.quality.sample_qc()` dùng chung cho tầng
    cleansed và ingest (quality_report.md và cờ của cleansed không đổi, đã đối chiếu).
  - Lô thêm từ file CSV (`ingest_batches(extra_files=...)`) để demo lô lỗi; chỉ số của lô thêm đo so
    với các lô thật nên không làm đổi cờ của 72 mẫu thật.
  - Đầu ra (`write=True`, `make ingest`) ở `data/cleansed/`: `batches.parquet` (→ `dim_batch`),
    `qc_samples.parquet` (→ `fact_qc`), `ingest_log.csv` ghi thêm mỗi lần chạy. Mặc định
    `ingest_batches()` không ghi file để dashboard gọi trực tiếp. Thêm `sample_quality()` (bảng theo mẫu).
  - `validate_new_sample(path)`: CSV `probe_id,value`, đủ 7129 probe chuẩn, không trùng/trống/không số;
    nhận cả đối tượng file (nút tải của Streamlit). Có > 90% giá trị < 100 thì báo lỗi "đã log hóa / sai
    thang" (72 chip thật: 40–56%). Chip lệch 38 mẫu tham chiếu (modified z của tương quan trung vị,
    trung vị, IQR; ngưỡng 3.5 như ADR-004) chỉ cảnh báo. Đã chạy cả 34 bệnh nhân tập test: đều `ok`,
    không cảnh báo nhầm.
- **Kiểm thử:** `tests/test_ingest.py` (20 test trên dữ liệu thật; báo lỗi, không skip khi thiếu dữ liệu).

## ADR-070 — Vẽ lại F01, F04d bằng `src/viz.py`; đổi màu loại mô (công việc 2.6)
- **Ngày:** 22/11/2026 · **Người đề xuất / quyết định:** TV6
- **Bối cảnh:** F01, F04d (TV1, notebook 02) vẽ rộng 10–11 in nên chữ khi in nhỏ hơn 10 pt, màu ghi cứng
  trong notebook (ADR-011). Trong F04d gốc, `#D55E00` vừa là St-Jude (panel trái) vừa là PB (panel phải),
  vi phạm quy tắc "một màu một nghĩa trong một hình" của style guide.
- **Quyết định:**
  - Thêm `viz.design_counts`, `viz.plot_sample_design` (F01) và `viz.plot_pca_batch` (F04d) vào
    `render_eda`; `make figures` vẽ cả hai, giữ nguyên tên file `F01_thiet_ke_mau_*`, `F04d_pca_batch_*`.
  - F04d dùng chung tọa độ PCA với F04b (`compute_pca`, cùng z-score và seed như notebook 02: PC1 16,1 %,
    PC2 11,4 %); chú giải màu là ô vuông, chú giải hình dạng (lớp) dùng chung ở trên; mẫu 21 khoanh tròn.
  - `TISSUE_COLORS`: BM `#BDBDBD`, PB `#000000` (trước: `#7F7F7F` / `#D55E00`). Áp dụng cho cả dashboard.
  - F01: hai panel cùng trục, cột từ 0, nhãn số trực tiếp, AML gạch `////` viền đen (in đen trắng vẫn đọc được).
- **Hệ quả:** notebook 02 (TV1) vẫn còn hàm vẽ cũ; chạy lại notebook sẽ ghi đè hình bằng bản cũ. Theo
  ADR-014 TV6 không sửa notebook của người khác: TV1 thay ô vẽ bằng lời gọi `viz` (xem `docs/handover/2.6.md`).

## ADR-031 — Bộ file lỗi mẫu, F8 và F10 bản cuối (công việc 2.4, tuần 2 của lịch giai đoạn 2)
- **Ngày:** 12/11/2026 · **Người quyết định:** TV2
- **Bối cảnh:** nhánh `2.4-f8-f10-demo` (commit `562ec9c`) chỉ có 4 file JSON/CSV vài dòng trong
  `data/sandbox/demo_errors/` (thêm bằng `git add -f`, trái `.gitignore` `data/**`), dạng dict metadata
  không khớp `validate_new_sample(path)` (file chip 7129 gen); chưa có F8; F10 vẫn là bản 1 (pipeline tạm
  trên tầng cleansed, chữ 9 pt / 220 dpi, tiêu đề ghi cứng "100 lần hoán vị"). Làm lại trên `2.4-f8-f10-demo-v2`.
- **Quyết định:**
  - **Bộ file lỗi mẫu** sinh bằng code (`write_demo_files()`, `python -m src.ingest --demo`) từ dữ liệu
    thật vào `data/demo/` (không lên Git, tạo lại được): 10 file chip một mẫu + 3 file lô, `README.md` ghi
    kết quả mong đợi; `tests/test_ingest.py` kiểm từng file. Gỡ `data/sandbox/demo_errors/`.
  - **F8 = "Chất lượng dữ liệu theo lô"** (`src/viz_ingest.py`, `F08_chat_luong_theo_lo_{report,slide}.png`):
    strip plot theo lô của tương quan với mẫu các lô khác và tỷ lệ call P; mẫu `qc_outlier` khoanh đen.
    **Giả định cần TV1 xác nhận:** tài liệu "Kế hoạch giai đoạn 2 (hướng mới)" chưa có trong repo; F8 cũ
    (hiệu năng theo k) phục vụ H2 đã bỏ ở ADR-014, còn lịch mới giao F8 cho việc 2.4 "Nhập theo lô, chất lượng".
  - **F10 bản cuối:** đầu vào tầng curated qua `make_preprocess()` của TV3 (bỏ bước log10 tạm và bộ lọc
    cảnh báo của ADR-008 — chạy với `-W error::UserWarning` không còn cảnh báo nào); hình vẽ qua `src/viz.py`
    (10/16 pt, 300/150 dpi, xám + một màu nhấn cho "CV sai", nhãn trực tiếp; bản slide bỏ chú thích phụ);
    số lần hoán vị và trung bình trên hình tính từ dữ liệu. F11 vẽ cùng kiểu; bản cuối do TV1 vẽ (lịch mới).
  - **Theo quy ước dữ liệu ADR-014**, mọi bước học của `make evaluate` chỉ dùng 38 mẫu huấn luyện:
    `nested_cv`, `wrong_cv` và thí nghiệm selection bias chạy trên 38 mẫu (trước đây 72);
    `original_split` học 38 rồi chấm 34; `loocv` không đổi. Cột `data` của `summary_evaluation.csv` ghi rõ.
- **Kết quả (`make evaluate`, 100 hoán vị, 12/11/2026):** nhãn thật — CV đúng 0.951 ± 0.019, CV sai
  1.000 ± 0.000; nhãn hoán vị — CV đúng 0.510 ± 0.089, CV sai **0.826 ± 0.065** → H3 vẫn được xác nhận.
  `original_split` 0.964 (AUC 0.996; 33/34 đúng, Wilson 95% [0.851, 0.995]; McNemar với lớp đa số
  p = 0.00024); `nested_cv` trên 38 mẫu 0.922 ± 0.137 (fold chỉ ~8 mẫu nên SD lớn).
- **Kiểm thử:** `tests/test_viz_ingest.py` (2), 2 test hồi quy F10 trong `tests/test_evaluate.py`, test bộ demo.

## ADR-060 — Chú giải 50 gen đặc trưng phân biệt ALL/AML (`gene_annotations.csv`)
- **Ngày:** 14/11/2026 · **Người quyết định:** TV5 (theo hợp đồng ADR-051 với TV4)
- **Bối cảnh:** TV4 (`src/gene_explorer.py`, ADR-051) và TV6 (dashboard) cần dữ liệu chú giải chức năng sinh học, dòng tế bào và liên kết cơ sở dữ liệu cho các gen xếp hạng hàng đầu để người dùng hiểu ý nghĩa sinh học thay vì chỉ nhìn mã probe Affymetrix hoặc tên mô tả thô.
- **Quyết định:** Tạo `results/gene_lists/gene_annotations.csv` cho top 50 gen phân biệt ALL/AML từ `gene_ranking.csv` (1.6), tuân thủ đúng hợp đồng 6 cột của ADR-051:
  `probe_id, gene_symbol, function, cell_lineage, source_url, verified`.
- **Nguyên tắc chú giải:**
  1. **Nguồn gốc:** Tra cứu chính xác từ cơ sở dữ liệu NCBI Gene / GeneCards; `source_url` trỏ về trang thông tin định danh gen chuẩn (ví dụ `https://www.ncbi.nlm.nih.gov/gene/<GeneID>`).
  2. **Ký hiệu gen:** Cập nhật ký hiệu chuẩn HGNC cho các probe có mô tả thô (ví dụ: `X95735_at` -> `ZYX`, `U50136_rna1_at` -> `LTC4S`, `M84526_at` -> `CFD`, `U05259_rna1_at` -> `CD79A`, `X82240_rna1_at` -> `TCL1A`).
  3. **Chức năng sinh học (`function`):** Mô tả ngắn gọn cơ chế hoạt động, phân loại phân tử, vai trò trong bệnh học bạch cầu (leukemia) và ý nghĩa phân biệt ALL/AML.
  4. **Dòng tế bào (`cell_lineage`):** Xác định thuộc dòng tủy (`myeloid`) hay dòng lympho (`lymphoid`), giải thích trực tiếp tại sao gen biểu hiện lệch mạnh giữa AML và ALL.
  5. **Tính xác thực (`verified`):** Gán `yes` cho 49 gen đã kiểm chứng rõ ràng qua NCBI Gene; gán `no` cho các gen cảm ứng chưa đặc trưng rõ (`L47738_at`), không tự ý đoán chức năng theo nguyên tắc quản trị dữ liệu trung thực.
- **Kiểm thử:** Đã kiểm tra tương thích 100% với hàm `annotate()` của `src/gene_explorer.py` và toàn bộ 113 bài kiểm thử `pytest tests/ -v` đều đạt `PASSED`.

## ADR-061 — Tái thiết kế trực quan hóa Golub et al. (1999) Fig 2 và Fig 4 (`src/viz_redesign.py`)
- **Ngày:** 15/11/2026 · **Người quyết định:** TV5 (theo ADR-014 của TV1 và Kế hoạch Giai đoạn 2)
- **Bối cảnh:** Các hình gốc trong bài báo Golub et al. (Science, 1999) sử dụng công nghệ đồ họa cũ (thang xám, thiếu phân tách thành phần lâm sàng, biểu đồ thanh mật độ thô khó quan sát chi tiết từng bệnh nhân). Mục tiêu môn học yêu cầu trực quan hóa hiện đại, an toàn cho người mù màu và tích hợp góc nhìn Quản trị dữ liệu (Data Governance).
- **Quyết định:** Tạo module `src/viz_redesign.py` và notebook `notebooks/05_viz_redesign.ipynb` độc lập với `src/viz.py` của TV6:
  1. **Fig 2 (SOM Class Discovery):**
     - Bảng A: Biểu đồ thanh đối chiếu thành phần mẫu ALL/AML trong 2 nơ-ron SOM 1×2 (38 mẫu train) giữa bài báo gốc và thực nghiệm của nhóm.
     - Bảng B: Chiếu không gian biểu hiện gen PCA (PC1 vs PC2) kết hợp tọa độ tâm nơ-ron SOM, làm rõ cấu trúc ranh giới cụm tự nhiên.
     - Bảng C: Khám phá phân nhóm sâu trên mạng SOM 2×2 (4 nơ-ron, 72 mẫu), chứng minh T-ALL không tách khỏi B-ALL ở cấp độ không giám sát (ủng hộ H5 ở mức ALL/AML, bác bỏ ở mức subtype).
  2. **Fig 4 (Independent Test Set & Error Analysis):**
     - Bảng A: Biểu đồ ngang xác suất dự đoán P(AML) cho toàn bộ 34 bệnh nhân tập kiểm tra độc lập, phân màu theo kết quả phân loại (ALL đúng, AML đúng, Sai phân loại), vạch rõ ngưỡng quyết định 0.5 và vùng ranh giới/nghi ngờ ([0.3, 0.7]).
     - Bảng B: Dải kiểm định Quản trị dữ liệu (Governance Tracks) gồm Loại mô (BM vs PB) và Nguồn bệnh viện (DFCI, CALGB, CCG, St-Jude).
     - Điểm nhấn lâm sàng: Làm nổi bật mẫu 66 (mẫu AML duy nhất bị mô hình đoán nhầm thành ALL, khớp với độ tin cậy thấp PS = 0.11 trong bài báo gốc); khẳng định 5 mẫu máu ngoại vi (PB) vẫn được mô hình phân loại chính xác.
  3. **Quy chuẩn đồ họa:** Đủ 2 phiên bản `report` (300 dpi, chữ >= 10 pt) và `slide` (150 dpi, chữ >= 16 pt) lưu tại `results/figures/report/` và `results/figures/slides/`. Bảng màu Okabe-Ito: ALL (`#0072B2`), AML (`#E69F00`).
- **Kiểm thử:** Đã bổ sung `tests/test_viz_redesign.py` (5 test) kiểm tra việc sinh hình, kích thước, định dạng và tính hợp lệ; toàn bộ 118 test suite đạt `PASSED`.

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
| 04/11/2026 | 2.4: `mcnemar_test` ghép theo `sample_id`; F11 giữ cột "uncertain"; `nested_cv` có vòng trong chọn k, C; thêm bộ golden-file 19 testcase | ADR-013 | TV2 |
| 09/11/2026 | Đổi hướng giai đoạn 2 sang quản trị dữ liệu và dashboard; dựng khung file, lệnh `make`, giữ trước số ADR | ADR-014 | TV1 |
| 01/11/2026 | 2.2: `src/features.py` bản 1 — 6 bộ chọn gen + `make_selector`, luôn đúng k gen, snr/ttest cân bằng hai phía | ADR-050 | TV4 |
| 12/11/2026 | 2.2 đổi sang Khám phá gen theo kế hoạch mới: `src/gene_explorer.py`, thay thế ADR-050; schema `gene_annotations.csv` cho TV5 | ADR-051 | TV4 |
| 12/11/2026 | 2.4: `src/ingest.py` theo giao diện ADR-014 — 4 lô theo nguồn, chặn lô lỗi, nhật ký, `validate_new_sample(path)`; tách `quality.sample_qc()` dùng chung | ADR-030 | TV2 |
| 15/11/2026 | 2.1: Golub weighted voting chuẩn hóa z-score theo train, SD chia n − 1, n lẻ báo lỗi, PS < 0,3 là uncertain | ADR-020 | TV1 |
| 23/11/2026 | 2.1 tuần 2: `loocv()` (36/36 khớp bài báo), test 33 chắc chắn / 32 đúng (sửa số cộng nhầm 32/31), `patient_profile()` | ADR-020 | TV1 |
| 22/11/2026 | 2.6: vẽ lại F01, F04d bằng `viz.py` (6,5 in, chữ ≥ 10 pt); loại mô BM xám nhạt, PB đen | ADR-070 | TV6 |
| 17/11/2026 | 2.2 bản 2: độ tin cậy (P/A theo lớp), độ vững, lô mới (`marker_consistency`, `batch_anomalies`), đọc kho DuckDB, `src/viz_gene.py`; St-Jude 41 giữ chiều, 0 đảo chiều | ADR-052 | TV4 |
| 18/11/2026 | 2.2: `marker_consistency` thêm kiểm định hoán vị trong lô (St-Jude p = 2/56 ≈ 0,036) và cột `batch_diff_sd` | ADR-052 | TV4 |
| 15/11/2026 | 2.3 bản 1: kho DuckDB `golub.duckdb` (`fact_expression` dạng dài, `dim_sample`, `dim_gene`, `dim_batch`); `sql/01` khớp Bảng 2.4 | ADR-040 | TV3 |
| 23/11/2026 | 2.3 bản 2: `dim_batch`, `fact_qc` (theo mẫu), `log_ingest` lấy từ `src/ingest.py`; chỉ nạp mẫu của lô đạt; `sql/02` chất lượng theo lô | ADR-041 | TV3 |
| 12/11/2026 | 2.4: bộ file lỗi mẫu sinh bằng code (`--demo`), F8 chất lượng theo lô, F10 bản cuối trên tầng curated theo style guide; mọi bước học chỉ dùng 38 mẫu huấn luyện | ADR-031 | TV2 |
| 14/11/2026 | 2.5: Chú giải 50 gen đặc trưng phân biệt ALL/AML (`gene_annotations.csv`) theo hợp đồng ADR-051 | ADR-060 | TV5 |
| 15/11/2026 | 2.5: Tái thiết kế trực quan hóa Golub et al. (1999) Fig 2 và Fig 4 (`src/viz_redesign.py`) | ADR-061 | TV5 |
