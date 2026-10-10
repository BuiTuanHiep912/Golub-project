# Decision Log (docs/decisions.md - TV2, mục 2.7)

Nhật ký quyết định của project. Mọi quyết định ảnh hưởng dữ liệu đều ghi ở đây để
kết quả tái lập được và truy ngược về nguồn.

## ADR-001 — Nguồn metadata mẫu là `dataset_more.csv`, không phải `openintro/golub.csv`
- **Ngày:** 2026-10-03 · **Người đề xuất:** TV2 (rà soát dữ liệu thật)
- **Bối cảnh:** Kế hoạch yêu cầu 6 cột mô tả từ `openintro/golub.csv` (OpenIntro).
  Không tìm thấy file này trong kho môn học.
- **Quyết định:** Dùng `dataset_more.csv` làm nguồn metadata (chứa đủ các cột:
  `Samples`=mã bệnh nhân, `BM.PB`=tissue, `Gender`=sex, `Source`, `cancer`=subtype,
  `tissue.mf`=morphology). Sao chép thành `data/raw/openintro/golub_metadata.csv`.
- **Ánh xạ cột:** `Samples→sample_id`, `BM.PB→tissue`, `Gender→sex`, `Source→source`,
  `cancer→subtype` (allB→B-ALL, allT→T-ALL, aml→AML), `tissue.mf→morphology`.
- **Hệ quả:** Số `sex` thiếu 23/72 là thiếu thật trong nguồn — giữ nguyên NaN, không
  tự điền (quality báo INFORMATION).

## ADR-002 — Nhãn ALL/AML lấy theo Kaggle (`actual.csv`)
- Nguyên tắc kế hoạch (2.1.2): nhãn theo bộ dữ liệu chính (Kaggle); metadata chỉ
  cho phân tích phụ. Đã xác minh khớp subtype metadata 72/72 (Bảng 2.4).

## ADR-003 — Gán `split` theo file Kaggle (patient 1..38 train, 39..72 test)
- `data_set_ALL_AML_train.csv` chứa patient 1..38; file independent chứa 39..72.
  Không gán lại theo thứ tự cột của file (thứ tự cột trong file không chuẩn 1..38).

## ADR-004 — Cờ `qc_outlier` chỉ gắn cờ, không xóa mẫu
- Mức 4 (2.5): mẫu có >50 gen |z-score toàn cục|>4 thì gắn cờ; chỉ xóa khi có căn cứ
  và qua review nhóm.

## ADR-005 — Bản PDF bài báo (Golub 1999) tải từ mirror
- Trang `proteome.gs.washington.edu` mất kết nối (timeout 07/2026). Tải từ
  `https://biostatistics.dk/teaching/advtopicsA/data/golub1999-...pdf`
  (ResearchGate export, 8 trang). MD5 `1d65f9…`. Lưu `docs/nguon/`.

## ADR-006 — Ngưỡng giá trị [100, 16000] chỉ báo cáo ở tầng cleansed
- Quality báo SỐ LƯỢNG giá trị ngoài khoảng; việc threshold/log10 thực hiện ở tầng
  curated (TV3, mục 2.6), không thay đổi dữ liệu cleansed.