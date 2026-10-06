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

_Đang thực hiện trên nhánh `tv2-review-fixes`; mục này được cập nhật khi xong từng việc._
