# Đánh giá công việc TV2 theo kế hoạch triển khai

> Xuất từ cuộc trò chuyện với Claude Code ngày 06/10/2026, nhánh `tv2-review-fixes`.
> Căn cứ: `Ke_hoach_trien_khai_project_Golub.pdf` (mục 1.1 và 2.4 của TV2).
> Tài liệu được đánh giá: `Thuyet_minh_nhiem_vu_TV2.pdf`,
> `Golub_DeTai10/TV2/Cong_viec_chi_tiet_TV2.md`, `Golub_DeTai10/TV2/code/README_CODE_TV2.md`
> và code trong `src/`, `tests/` tại commit `330ce48`.

## Phần I. Câu trả lời lần 1 — đánh giá (nguyên văn)

### Tin nhắn tiến độ trong lúc đánh giá

- "Đã đọc xong kế hoạch (27 trang); đang đọc thuyết minh TV2, file công việc chi tiết, README code và mã nguồn."
- "Đã đọc hết tài liệu và code. Giờ đang tải dữ liệu Golub công khai vào scratchpad (đối chiếu MD5 với `SOURCES.md`) và dựng venv tạm để chạy thật pipeline — không ghi gì vào repo."
- "Tìm được bản sao 3 file Kaggle trên GitHub; đang tải về scratchpad và đối chiếu MD5 với `SOURCES.md` trước khi dùng."
- "Cả 4 file khớp MD5 trong `SOURCES.md`. Giờ chạy pipeline trên bản sao sạch của commit HEAD trong scratchpad."
- "Pipeline GĐ1 chạy được: 4 parquet, quality report và `15 passed, 1 skipped`. Giờ kiểm tra sâu vài điểm nghi lỗi trong `quality.py` và cách ghép theo `sample_id`."
- "Phát hiện đáng chú ý: chỉ 17/72 mẫu Kaggle tương quan cao nhất với đúng `sample_id` bên OpenIntro. Cần xác minh đây là do OpenIntro đã chuẩn hóa khác hay do khoá ghép sai."
- "Khoá ghép được xác minh đúng: Spearman theo từng mẫu giữa Kaggle và OpenIntro (cùng `sample_id`) = 1.0 ở cả 72/72 mẫu. Con số 17/72 lúc trước là do tôi so hai thang đo khác nhau. Tiếp theo: kiểm tra demo của kế hoạch, cách gọi trực tiếp, và đối chiếu các số GĐ2 mà tài liệu công bố."

### Tóm tắt

- **Phần 1.1 (GĐ1) chạy được.** Kết quả khớp đủ các con số bắt buộc ở Bảng 2.4. Tuy vậy `src/quality.py` có 2 lỗi thật, và phần kiểm tra chưa đủ 4 mức như Bảng 2.9 yêu cầu.
- **Phần 2.4 (GĐ2) chưa có code.** `src/evaluate.py` vẫn chỉ là chỗ để trống (`# TODO`). Các con số GĐ2 ghi trong `Cong_viec_chi_tiet_TV2.md` không kiểm chứng được và tự mâu thuẫn nhau.
- **`README_CODE_TV2.md` đã lỗi thời hoàn toàn.** Nó mô tả 8 file không tồn tại ở nhánh nào.

**Cách tôi kiểm:** Thư mục `data/` trong repo đang trống, nên tôi tải bản Kaggle chính thức và `golub.csv` của OpenIntro về scratchpad. Cả 4 file khớp MD5 ghi trong `SOURCES.md`. Tôi chạy trên một bản sao của commit HEAD (Python 3.11, pandas 3.0.6, scikit-learn 1.9.1). Tôi không ghi gì vào repo.

### 1. Code GĐ1: chạy được, khớp demo nghiệm thu

| Lệnh | Kết quả |
|---|---|
| `make standardize` | Ra đủ 4 parquet; MD5 khớp 4/4; các bảng chéo đúng Bảng 2.4 |
| `make quality` | Ra đúng các con số ghi trong tài liệu (241357 / 4259 / 147174, thiếu sex 23/72) |
| `make test` | `15 passed, 1 skipped` (test bị bỏ qua là của TV3) |
| `load_golub(layer='cleansed')` | `(72, 7129), (72, 7)`, đúng như demo trang 20 |

Tôi kiểm thêm khoá ghép `sample_id`. Với từng mẫu, giá trị biểu hiện của OpenIntro tương quan Spearman bằng 1.0 với Kaggle ở cả 72/72 mẫu. Vậy là ghép đúng bệnh nhân, không chỉ đúng phân bố.

### 2. Lỗi trong code cần sửa trước hạn 10/10

1. **Gắn cờ sai mẫu** (`src/quality.py:131`). Code dùng `isin(flagged.index)`, tức vị trí 0, 1, 2…, chứ không phải mã mẫu. Tôi thử giả lập: cờ cho mẫu 50 thì không gắn vào mẫu nào; cờ cho các mẫu [10, 20, 30] thì lại gắn vào mẫu [1, 2]. Cần đổi thành `flagged.values`.
2. **Kiểm tra mẫu bất thường không bao giờ báo được gì** (`src/quality.py:89-93`). Code tính z-score trên toàn bộ ma trận thay vì theo từng gen. Giá trị |z| lớn nhất thực tế là 2.55, trong khi ngưỡng là 4, nên "0 mẫu bị cờ" là do cách tính chứ không phải do dữ liệu sạch. Bảng 2.9 yêu cầu dùng tương quan giữa các mẫu và boxplot theo mẫu. Đề xuất: tính tương quan trung vị của mỗi mẫu với các mẫu khác, dùng z-score chống ngoại lai (MAD), và phát hiện mẫu trùng khi tương quan gần bằng 1.
3. **Chưa đúng 4 mức của Bảng 2.9.** Code đang chia thành Record, Relation, Value gộp Value-set, rồi Record thêm lần nữa. Ngoài ra còn thiếu:
   - Đếm các probe dùng chung một gen: thực tế có 926 probe trùng mô tả, thuộc 424 mô tả.
   - Đối chiếu chéo `tissue.mf` như Bảng 2.2 yêu cầu. Tôi đã thử: ghép lại từ tissue và sex thì khớp 72/72, nên thêm rất dễ.
   - Tên `morphology` (`load.py:50`) là sai nghĩa: `tissue.mf` là loại mô kèm giới tính (m/f).
4. **`load_golub` âm thầm trả nhầm tầng** (`load.py:279`). Gọi `layer='curated'` hoặc gõ nhầm `'cleanesd'` đều không báo lỗi mà trả về tầng standardized. Cần báo lỗi khi tên tầng không hợp lệ. Việc này nên làm trước khi giao bản tạm cho TV3 vào CN 11/10.
5. **`load.py` còn thiếu các kiểm tra sau ghép của bước 6**: chưa kiểm `X.index.equals(samples.index)`, chưa kiểm `split` và `tissue`/`source` không bị thiếu.
6. **Lỗi không chặn pipeline.** Khi `quality.py` gặp kiểm tra `[FAIL]`, nó vẫn ghi tầng cleansed và thoát với mã 0. Còn khi chưa có dữ liệu, `pytest` cho `16 skipped` mà vẫn báo xanh.
7. **`environment.yml` chưa cố định phiên bản thư viện**, trong khi mục 2.7 yêu cầu cố định. Nên ghi đúng các phiên bản tôi vừa chạy được.
8. **`SOURCES.md` thiếu link Kaggle** (`kaggle.com/datasets/crawford/gene-expression`). Ngày `05102026` nên viết lại là `05/10/2026`.

### 3. Đánh giá 3 tài liệu so với kế hoạch

**`Thuyet_minh_nhiem_vu_TV2.pdf`**: nội dung chính bám đúng kế hoạch. Các chỗ cần sửa:
- Đường dẫn hình đang ghi `reports/figures/…`. Kế hoạch dùng `results/figures/report|slides/`, tên file dạng `F10_<tên>_report.png`.
- Câu "tầng làm sạch… sẵn sàng cho trích xuất đặc trưng" là sai. Tầng cleansed giữ nguyên giá trị và chỉ thêm cờ; tầng sẵn sàng phân tích là curated (của TV3).
- Thí nghiệm selection bias trong kế hoạch là bảng 2×2: nhãn thật/hoán vị × CV đúng/sai.
- Test cần bao phủ toàn bộ Bảng 2.4.
- Còn thiếu:
  - Các mốc giao việc: `load_golub` bản tạm cho TV3 (11/10), `samples.parquet` chính thức cho TV1 (18/10), `evaluate.py` bản 1 cho TV1/TV3 từ tuần 5, ghi chú bàn giao (01/11).
  - Chữ ký hàm `evaluate(..., seeds=)` và các khoá kết quả trả về.
  - Phụ thuộc vào `experiment_config.yaml`.

**`Cong_viec_chi_tiet_TV2.md`**:
- **Phần A**: các con số đều đúng, tôi chạy lại ra y hệt. Nhưng có 3 chỗ sai:
  - "TV2 cùng TV1 là trưởng nhóm" không đúng: kế hoạch chỉ ghi TV1 là trưởng nhóm.
  - Trạng thái "đã xong" là quá lời, vì các mục 2.1–2.5 và 2.7 của báo cáo chưa viết.
  - Thư mục `Golub_DeTai10` được mô tả là chứa tài liệu của TV1/TV6 và code cũ, nhưng thực tế không có.
- **Phần B** có vấn đề nghiêm trọng:
  - Viện dẫn code và `docs/leakage_review.md`, nhưng cả hai không tồn tại ở nhánh nào. Code của TV1 cũng chỉ là chỗ để trống, nên chưa có gì để review.
  - Các con số tự mâu thuẫn: `nested_cv bal_acc=0.555` trong khi bảng selection bias ghi "Thật/Đúng 0.970". Còn `auc=nan` với CV 5 fold là dấu hiệu có lỗi.
  - `loocv 0.395±0.489` là tính trung bình chỉ số trên từng fold chỉ có 1 mẫu, cách làm này sai. Phải gộp dự đoán của cả 38 fold rồi mới tính chỉ số. Tôi thử nhanh một pipeline đúng (log10 → chọn 50 gen → LogReg) thì được: tách gốc 38/34 có bal_acc 0.964, AUC 0.993; LOOCV gộp có bal_acc 0.955.
  - Danh sách việc thiếu sensitivity/specificity và kiểm định McNemar.
  - Đề xuất thêm `src/selection_bias.py` và `src/roc_cm.py` lệch với Bảng 2.6. Theo kế hoạch, phần này nằm trong `evaluate.py`, phần vẽ dùng `viz.py`, kèm notebook `08_evaluation`.
  - `make evaluate` chưa có trong Makefile.
  - Ghi "đang chờ TV3" là trái kế hoạch: TV1 và TV3 cần dùng `evaluate.py` từ tuần 5. Có thể viết và test ngay bằng pipeline tạm như đoạn mã mẫu ở mục 3.5.2.

**`README_CODE_TV2.md`**: nên viết lại từ đầu hoặc xoá.
- 8 module trong bảng không tồn tại. Tôi đã kiểm cả `main`, `nhap`, `origin/TV2` và `origin/1.1-load-data`.
- Các thông tin gốc thư mục `Golub_DeTai10/`, `fetch_raw.py`, `dataset_more.csv` mâu thuẫn với ADR-001 và với `load.py` hiện tại.
- Câu "openintro/golub.csv không tồn tại" là sai.
- `docs/data_dictionary.md:66` cũng đang trỏ tới `leakage_review.md`, file này không tồn tại.

### 4. Đề xuất thứ tự làm

1. **Trước 10/10:** sửa các mục 1–5 ở phần 2 (`quality.py`, `load_golub`, các kiểm tra sau ghép), cố định phiên bản trong `environment.yml`.
2. **Trước khi nộp:** sửa ba tài liệu theo phần 3. Riêng Phần B và README nên ghi rõ là chưa làm, thay vì đưa số liệu.
3. **Trước tuần 5:** viết `evaluate.py` bản 1 đúng API kế hoạch, gồm đủ 4 sơ đồ, sensitivity/specificity, Wilson, McNemar và thí nghiệm selection bias.

Tôi có thể sửa các lỗi code ở phần 2 trên một nhánh riêng, kèm test tái hiện lỗi `qc_outlier`, nếu anh/chị đồng ý. Tôi cũng có thể làm bản đánh giá này thành một trang để gửi nhóm nếu cần.

## Phần II. Cập nhật sau khi sửa

_Cập nhật 07/10/2026. Phần I giữ nguyên làm bản đánh giá gốc tại commit `330ce48`._

### 1. Các lỗi code ở Phần I, mục 2 — đã sửa hết (kiểm lại ngày 07/10)

| # | Lỗi | Trạng thái | Bằng chứng |
|---|---|---|---|
| 1 | Gắn cờ `qc_outlier` theo vị trí thay vì `sample_id` | ✅ | `tests/test_quality.py::test_qc_outlier_flags_the_corrupted_sample_by_sample_id` tái hiện lỗi cũ và xác nhận cờ rơi đúng mẫu (gộp với lỗi 2 ở sửa đổi ADR-004) |
| 2 | Z-score toàn ma trận → không bao giờ báo được | ✅ | `quality.py` đổi sang Bảng 2.9: tương quan giữa các mẫu + boxplot theo mẫu, modified z (median/MAD) ngưỡng 3.5; dữ liệu thật phát hiện mẫu 21 (r = 0.756, z = −4.7) |
| 3 | Chưa đúng 4 mức Bảng 2.9; thiếu đếm trùng mô tả, thiếu `tissue.mf`, tên `morphology` sai nghĩa | ✅ | 4 mức `check_value/check_value_set/check_record/check_relation`; 926 probe thuộc 424 mô tả trùng; `tissue.mf` khớp `tissue`+`sex` 72/72 và chỉ dùng để kiểm tra chéo; cột đổi tên `tissue_mf` |
| 4 | `load_golub` âm thầm trả nhầm tầng | ✅ | Bảng `LAYER_FILES` + `ValueError` cho tên tầng lạ; test `test_load_golub_rejects_unknown_layer` (3 tên sai) |
| 5 | Thiếu kiểm tra sau ghép (bước 6) | ✅ | `_assert_same_ids` (so tập `sample_id` 3 nguồn) + `_assert_post_merge` (72 mẫu, đủ metadata, class↔subtype, tissue.mf) |
| 6 | FAIL không chặn pipeline; chưa có dữ liệu thì test vẫn xanh | ✅ | `QualityCheckError` → không ghi + xóa tầng cleansed cũ + thoát mã 1 (`ADR-006`); fixture của `test_load.py` `pytest.fail` thay vì skip; test `test_failed_check_stops_pipeline_and_removes_stale_cleansed` |
| 7 | `environment.yml` chưa ghim phiên bản | ✅ | Đã ghim `environment.yml` = `requirements.txt` (ADR-007); `make data` xanh trên Python 3.10 máy hiện tại |
| 8 | `SOURCES.md` thiếu link Kaggle, ngày sai định dạng | ✅ | Đã có link `kaggle.com/datasets/crawford/gene-expression`, ngày `05/10/2026`, kèm lệnh tải lại + đối chiếu MD5 |

### 2. Tài liệu ở Phần I, mục 3 — đã sửa (các commit `82dee9d`, `ef5206a`)

- **Thuyết minh** (`Thuyet_minh_nhiem_vu_TV2.pdf`): sửa đường dẫn hình sang
  `results/figures/report|slides/` với tên `F10_selection_bias_{report,slide}.png`,
  `F11_final_model_roc_cm_{report,slide}.png`; câu "tầng cleansed sẵn sàng cho trích xuất đặc
  trưng" → tầng sẵn sàng là `curated` (TV3); thêm bảng 2×2 selection bias; thêm các mốc giao
  việc (11/10, 18/10, 01/11), chữ ký `evaluate(..., seeds=)`, 6 khoá trả về (kèm `predictions`),
  phụ thuộc `experiment_config.yaml`.
- **`Cong_viec_chi_tiet_TV2.md`**: Phần B bỏ số liệu tự mâu thuẫn, mô tả đúng 4 sơ đồ
  (LOOCV gộp dự đoán, không `auc=nan`), thêm sensitivity/specificity + McNemar, bỏ đề xuất
  `selection_bias.py`/`roc_cm.py` (nay nằm trong `evaluate.py` + notebook `08`), thêm
  `make evaluate` vào Makefile; các số liệu Bảng 2×2 chạy lại khớp thật.
- **`README_CODE_TV2.md`**: viết lại — bảng module giờ trỏ đúng `src/` + `tests/`, bỏ thông tin
  `fetch_raw.py`/`dataset_more.csv` sai với ADR-001, không còn câu "golub.csv không tồn tại".
- `docs/data_dictionary.md` không còn trỏ tới `docs/leakage_review.md` (file chưa tồn tại).

### 3. Thứ tự làm ở Phần I, mục 4

1. ✅ Trước 10/10: sửa 8 mục lỗi code + ghim phiên bản (xong 06/10, kiểm lại 07/10).
2. ✅ Sửa 3 tài liệu theo Phần 3 (các commit `82dee9d`, `ef5206a`).
3. ✅ Trước tuần 5: `src/evaluate.py` bản 1 đúng API kế hoạch — 4 sơ đồ, sensitivity/
   specificity, Wilson, McNemar, selection bias 100 hoán vị; có `tests/test_evaluate.py`
   (11 test, dữ liệu tổng hợp) chạy xanh.

### 4. Việc làm thêm ngày 07/10/2026 (ngoài danh sách đánh giá, theo thuyết minh)

- `make evaluate` chạy đủ 100 hoán vị, ghi `results/metrics/summary_evaluation.csv`,
  `summary_selection_bias.csv`, `selection_bias_runs.csv`.
- Thêm `plot_selection_bias()` + `plot_roc_cm()` vào `src/evaluate.py` → tự sinh 4 hình
  `F10_selection_bias_{report,slide}.png`, `F11_final_model_roc_cm_{report,slide}.png`
  (quyết định ADR-009; F11 bản 1 vẽ bằng pipeline tạm, chờ mô hình cuối của TV3).
- Viết `notebooks/08_evaluation.ipynb` (15 cell) và chạy thật: 0 lỗi, 0 warning, 2 ảnh nhúng.
- Soạn `docs/handover/1.1.md` (mốc CN 01/11).
- `.gitignore` thêm `dataset/`, `dataset_more.csv`.

### 5. Còn lại (chưa làm, có lý do khách quan)

| Việc | Lý do |
|---|---|
| Mục 2.1–2.5, 2.7 và 3.5 của báo cáo (`docs/report/` đang trống) | Viết khi các GĐ khác giao số liệu cuối |
| `docs/leakage_review.md` (review rò rỉ code TV1/TV3/TV4) | `golub_wv.py`, `preprocess.py`, `features.py`, `models.py` vẫn là `# TODO` — chưa có code để review |
| F11 bản chính thức + làm lại kiểu dáng F10/F11 theo `viz.py` | Chờ mô hình cuối (TV3, 2.3) và style guide của TV6 (T4 14/10) — xem ADR-009 |
| Test `tests/test_no_leakage.py` (đang 1 skipped) | Của TV3, công việc 1.2 |

**Kết luận:** cả 8 lỗi code và các lỗi tài liệu nêu ở Phần I đã được sửa và kiểm chứng lại;
phần còn lại phụ thuộc công việc của TV1/TV3/TV6 và nội dung báo cáo cuối kỳ.

## Phần III. Đánh giá cấu trúc thư mục sau commit `363fc5f` (07/10/2026)

> Ghi lại ngày 09/10/2026 (bản ghi ngày 07/10 chưa commit nên đã mất). Kiểm lại hiện trạng ngày
> 09/10: 2 file PDF vẫn chưa có trên đĩa; 25 file trong `data/sandbox/` vẫn được git theo dõi;
> `Golub_DeTai10/TV2/__pycache__/` vẫn còn ở gốc. Các nhận định dưới đây vẫn đúng.

Căn cứ: cây thư mục mục 2.2 và Bảng 2.5 của kế hoạch. Đã kiểm trên đĩa, trong `git ls-files`,
và chạy lại pipeline + test trên commit `363fc5f`: `make quality` ĐẠT, `33 passed, 1 skipped`.

### Kết luận: hợp lý hơn ở gốc repo, chưa hợp lý ở cách dùng sandbox

**Phần tốt hơn**
- Gốc repo giờ khớp đúng cây mục 2.2 (`config/ data/ docs/ notebooks/ results/ src/ tests/` + file
  môi trường); thư mục lạ `Golub_DeTai10/` không còn được git theo dõi.
- Bản làm việc cũ được giữ làm nháp thay vì xoá hẳn.

**Phần chưa hợp lý**
1. **Mất 2 file PDF.** `Ke_hoach_trien_khai_project_Golub.pdf` (đặc tả của cả dự án) và
   `Thuyet_minh_nhiem_vu_TV2.pdf` bị xoá khỏi git và **không có ở đâu trên đĩa**, kể cả
   `data/sandbox/`. Khôi phục được: `git show 82dee9d:<tên file> > <tên file>`.
2. **Sandbox bị đưa lên git, trái kế hoạch.** Mục 2.2 ghi `data/` nằm trong `.gitignore` và sandbox là
   "nháp riêng, không kết quả nào trong báo cáo lấy từ đây". `.gitignore` vẫn chặn `data/**`,
   nhưng 25 file trong `data/sandbox/TV2/` đã bị ép vào git. Hệ quả: file mới trong sandbox bị bỏ qua,
   còn file cũ thì vẫn bị theo dõi, nên trạng thái không nhất quán.
3. **Hai nguồn sự thật mâu thuẫn trong git.**
   - `data/sandbox/.../docs/decisions.md` ghi ADR-001 = `dataset_more.csv`, ngược với `docs/decisions.md` (golub.csv).
   - `.../docs/leakage_review.md` chấm "✅ Đạt" cho `golub_wv.py` của TV1, trong khi `src/golub_wv.py` vẫn là `# TODO`.
   - `.../results/metrics/selection_bias_summary.csv` còn số cũ không kiểm chứng (0.970 / 0.743).
   - Có 8 file `.py` cũ (`load.py`, `quality.py`, `evaluate.py`…) song song với `src/`, mang các lỗi đã sửa
     (z-score toàn cục, `dataset_more.csv`).
4. **Tài liệu chính thức của TV2 bị đặt vào chỗ nháp.** `Danh_gia_TV2.md`,
   `Cong_viec_chi_tiet_TV2.md`, `tao_thuyet_minh_pdf.py` là tài liệu bàn giao, nên thuộc `docs/`.
5. **Hỏng do đổi chỗ.**
   - `tao_thuyet_minh_pdf.py` tính `OUT = parents[2]`, giờ trỏ vào `data/sandbox/TV2/` (bị ignore)
     thay vì gốc repo.
   - `README_CODE_TV2.md` ghi "thư mục code chỉ còn file này" trong khi bên cạnh có 8 file `.py`.
   - Các tài liệu vẫn ghi đường dẫn `Golub_DeTai10/TV2/...`.
6. **Lặt vặt.**
   - Tên thư mục là `TV2`, kế hoạch dùng `tv2/`.
   - `Golub_DeTai10/TV2/__pycache__/` còn sót ở gốc (chỉ trên đĩa).
   - Bài báo Golub 1999 (476 KB) được commit: nếu repo GitHub công khai thì nên chỉ ghi DOI.
   - `results/` (PNG, CSV) được commit, trong khi mục 2.2 chỉ đưa mã nguồn, cấu hình và tài liệu lên
     GitHub. Thêm nữa, F11 hiện vẽ từ pipeline tạm.
   - Commit `363fc5f` gộp nhiều việc khác nhau với message chỉ là tên file.

### Chỗ `07102026.md` tóm tắt sai hoặc thiếu
- Ghi "chưa commit gì trong ngày": thực tế đã commit `363fc5f` và đẩy lên `origin/nhap`.
- Ghi "file ở sandbox không bị git theo dấu": thực tế 25 file đang được theo dõi.
- Ghi "2 PDF chuyển vào `data/sandbox/`": thực tế không có trên đĩa.
- Ghi "`Golub_DeTai10/` gồm `TV1/`, `TV6/`, `data/`": thực tế sandbox không có các thư mục này.
- Không nhắc tới: bài báo PDF, 8 file code cũ, `docs/` cũ mâu thuẫn, `results/` cũ, đường dẫn
  `tao_thuyet_minh_pdf.py` bị hỏng, `__pycache__` sót ở gốc.

### Cấu trúc đề xuất (chưa thực hiện — chờ quyết định)
```
docs/
├── Ke_hoach_trien_khai_project_Golub.pdf     # khôi phục từ 82dee9d
└── tv2/                                       # tài liệu bàn giao của TV2 (trong git)
    ├── Thuyet_minh_nhiem_vu_TV2.pdf           # khôi phục, sinh lại bằng script bên cạnh
    ├── tao_thuyet_minh_pdf.py                 # sửa OUT trỏ về docs/tv2/
    ├── Cong_viec_chi_tiet_TV2.md
    ├── Danh_gia_TV2.md
    └── nhat_ky/07102026.md
data/sandbox/tv2/                              # chỉ trên máy (git rm --cached), đúng mục 2.2
└── Golub_DeTai10/ (code, docs, results cũ)
```
Cần chốt trước khi làm: có commit `results/` không; có giữ bài báo PDF trong repo không.

## Phần IV. TV2 đã xong cả 2 giai đoạn chưa? (07/10/2026)

> Ghi lại ngày 09/10/2026. Kiểm lại hiện trạng ngày 09/10: `golub_wv.py`, `preprocess.py`,
> `features.py`, `models.py`, `viz.py` vẫn là `# TODO`; `docs/report/` vẫn trống. Bảng dưới vẫn đúng.

**Chưa.** Phần code của GĐ1 đã xong. GĐ2 mới có bản 1 làm sớm, phần còn lại phụ thuộc code của
TV1/TV3/TV4/TV6.

| Giai đoạn | Đã xong | Còn lại |
|---|---|---|
| GĐ1 – 1.1 (hạn T7 10/10) | `load.py`, `quality.py`, `tests/`, `data_dictionary.md`, `decisions.md`, nháp `docs/handover/1.1.md` | Mục 2.1–2.5, 2.7 của báo cáo (`docs/report/` trống); giao `load_golub()` cho TV3 (CN 11/10), `samples.parquet` cho TV1 (CN 18/10); nghiệm thu với TV1 và trình bày (CN 01/11); khôi phục 2 PDF đã mất (Phần III) |
| GĐ2 – 2.4 (tuần 5–7) | `evaluate.py` bản 1 (4 sơ đồ, Wilson, McNemar), selection bias 100 hoán vị, F10, notebook 08 | Chạy lại trên `X_log10` của TV3; F11 cho mô hình cuối của TV3 (hiện vẽ từ pipeline tạm); McNemar giữa các mô hình thật; review rò rỉ TV1/TV3/TV4 (`docs/leakage_review.md`); vẽ lại F10/F11 theo `viz.py` của TV6; mục 3.5 của báo cáo |

Việc nên làm ngay trước 10/10: viết các mục báo cáo của GĐ1 và khôi phục 2 PDF.

## Phần V. Cập nhật `notebooks/TV2_tong_quan_cong_viec.ipynb` (09/10/2026)

Sửa ô `fefd3cf0` (mục 4, bảng §5.2), chỉ thay đúng một dòng nguồn; các ô khác và kết quả chạy
giữ nguyên từng byte; `nbformat.validate` đạt, hàng "Công việc cụ thể" vẫn render đúng 2 ô.
Mọi tên hàm, tên test và con số đối chiếu với code tại HEAD `0edc240` (không đổi từ `8d3f364`).

1. **Thêm mô tả quá trình** (khuôn: dữ liệu vào → hàm/trình tự → kết quả) cho:
   - `src/quality.py`: đầu vào 3 parquet tầng standardized; `run_quality()` gọi `check_value`,
     `check_value_set`, `check_record`, `check_relation` theo thứ tự Bảng 2.9, kèm số liệu thật
     (241357 / 4259 / 147174; 58 AFFX; 926 probe / 424 mô tả; mẫu 21, r = 0.756, z = −4.7); FAIL
     dừng pipeline (ADR-006); `test_quality.py`; đầu ra 3 file `data/cleansed/`.
   - `tests/test_load.py`: fixture `tables` (thiếu dữ liệu → fail); 9 test ứng với 8 dòng
     Bảng 2.4 (có tên test và số kỳ vọng); 10 test định dạng/API; kết quả `19 passed`.
   - `docs/data_dictionary.md`: nguồn thông tin, cách viết theo chiều chảy dữ liệu, nội dung 5
     mục, kết quả (84 dòng).
2. **Sửa phần `src/load.py` do TV2 viết:**
   - Lỗi gõ `golgle/golub.csv` → `openintro/golub.csv`.
   - Câu "`_load_metadata()` … ghép theo `sample_id`" sửa thành "đặt `sample_id` làm chỉ mục":
     việc ghép nằm trong `build()`, không trong `_load_metadata()`.
   - Thêm các bước kiểm tra `build()` thật sự chạy: lỗi file/probe ở `_load_kaggle_expression()`;
     `_assert_same_ids()` trước khi ghép; ghép left join `validate="one_to_one"`;
     `_assert_post_merge()` sau ghép (72 mẫu, cùng chỉ mục, đủ metadata, class khớp subtype,
     `tissue.mf` khớp tissue + sex); kiểm tra (72,7129) và NaN trước khi ghi; `write_sources()`
     đối chiếu MD5 (ADR-005); in bảng chéo đối chiếu Bảng 2.4.
