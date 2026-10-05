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
  `cancer→subtype` (`allB→B-ALL`, `allT→T-ALL`, `aml→AML`), `tissue.mf→morphology`
  (giữ để tài liệu, không dùng trong mô hình).
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
- **Ngày:** 05/10/2026 · **Người quyết định:** TV2
- **Lý do:** mức 4 của mục 2.5 chỉ phát hiện, không tự quyết xóa dữ liệu. Cần review
  của nhóm trước khi loại mẫu. Hiện chưa có mẫu nào bị gắn cờ.

## ADR-005 — `SOURCES.md` do người tải ghi tay; code chỉ đối chiếu MD5
- **Ngày:** 05/10/2026 · **Người quyết định:** TV2
- **Lý do:** file ghi thêm ngày tải và người tải (thông tin ngoài phạm vi code).
  `src.load.write_sources()` in MD5 thực tế, cảnh báo nếu lệch, không ghi đè.

## Nhật ký nhanh (mục 5.2 kế hoạch)

| Ngày | Quyết định | Lý do | Người quyết định |
|---|---|---|---|
| 05/10/2026 | Dùng Kaggle (biểu hiện gen + nhãn) + 6 cột mô tả của OpenIntro | Kaggle thiếu metadata; bản biểu hiện OpenIntro đã chuẩn hóa nên không dùng | TV2 |
| 05/10/2026 | Cấu trúc tầng raw → standardized → cleansed → curated → sandbox | Mục 2.2, Bảng 2.5 | Cả nhóm |