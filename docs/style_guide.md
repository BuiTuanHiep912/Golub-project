# Style guide hình vẽ

Phụ trách: TV6 (công việc 1.4) · Áp dụng cho mọi hình F1–F12 · Mã nguồn: `src/viz.py` (các hằng số `*_COLORS`, `FIG_STYLES`).
Nguyên tắc nguồn: kế hoạch mục 4.1 (Mackinlay; Cleveland & McGill; Wilke 2019).

**Quy tắc vàng:** mỗi hình trả lời đúng **một câu hỏi** (cột "Câu hỏi" ở kế hoạch Bảng 4.2); màu và hình dạng điểm luôn **nhất quán** giữa các hình; không ghi cứng màu trong notebook, lấy từ `src/viz.py`.

## 1. Màu

Bảng Okabe–Ito, an toàn với người mù màu.

| Đối tượng | Giá trị | Mã màu | Hình điểm | Ghi chú |
|---|---|---|---|---|
| Lớp | ALL | `#0072B2` (xanh dương) | tròn `o` | `src.config.PALETTE` |
| Lớp | AML | `#E69F00` (cam) | tam giác `^` | AML là lớp dương |
| Nhóm con | B-ALL | `#0072B2` | tròn `o` | cùng màu ALL |
| Nhóm con | T-ALL | `#332288` (chàm) | vuông `s` | |
| Nhóm con | AML | `#E69F00` | tam giác `^` | |
| Nguồn mẫu | DFCI | `#009E73` | — | khớp F4d (TV1) |
| Nguồn mẫu | CALGB | `#CC79A7` | — | |
| Nguồn mẫu | CCG | `#56B4E9` | — | |
| Nguồn mẫu | St-Jude | `#D55E00` | — | |
| Loại mô | BM / PB | `#7F7F7F` / `#D55E00` | — | |
| Trung tính | cột, đường tham chiếu | `#4D4D4D` đậm, `#BDBDBD` nhạt | — | không mang nghĩa lớp |

- **Mã hóa dư thừa:** màu luôn đi kèm hình dạng (điểm) hoặc kiểu gạch (hộp, cột) để in đen trắng vẫn đọc được. Hộp/cột: ALL trơn, AML gạch chéo `////`.
- **Một màu nổi bật:** mỗi hình/slide chỉ có một màu nhấn cho thông điệp chính (mô hình tốt nhất, mẫu bất thường); phần còn lại xám.
- **Heatmap z-score:** thang phân kỳ `RdBu_r` (xanh–trắng–đỏ), tâm tại 0, `vmin = −vmax`. **Không** dùng rainbow/jet.
- **Thang tuần tự** (tương quan, mật độ): `cividis`. Khi thang không bắt đầu từ 0 phải ghi rõ trên thanh màu.
- Một màu chỉ mang **một nghĩa trong một hình**. Trong F4d màu là nguồn mẫu; trong F4b màu là lớp.

## 2. Chữ, kích thước, độ phân giải

| | Báo cáo (`_report`) | Slide (`_slide`) |
|---|---|---|
| Cỡ chữ tối thiểu (mọi chữ: trục, chú giải, nhãn) | **10 pt** | **16 pt** |
| Tiêu đề hình (`suptitle`) | 12 pt | 18 pt |
| Bề rộng hình | **6,5 in** (bằng bề rộng văn bản A4) | 12 in |
| Độ phân giải | 300 dpi | 150 dpi |
| Kích thước điểm (`s`) | 38 | 95 |
| Độ dày đường | 1,4 | 2,4 |

- Font: **DejaVu Sans** (đủ dấu tiếng Việt). Không dùng font không có dấu.
- **Vẽ ở kích thước in thật.** Hình vẽ rộng 11 in rồi thu về 6,5 in trong báo cáo làm chữ 10 pt thành ~6 pt, vi phạm quy tắc. `figsize(mode)` trong `viz.py` đã đặt sẵn đúng kích thước.
- Bản slide **đơn giản hóa** (bớt chú thích phụ), không chỉ phóng to bản báo cáo.
- Bỏ khung trên và phải (`axes.spines`), không dùng lưới nền, không 3D, không biểu đồ tròn.

## 3. Quy tắc theo loại biểu đồ

| Tình huống | Quy tắc |
|---|---|
| Cột (bar) | **Luôn bắt đầu từ 0** (proportional ink). Muốn phóng vùng 0,9–1,0 thì dùng dot plot |
| So sánh số chính xác | Ưu tiên dot plot / bar hơn diện tích hay màu (Cleveland & McGill) |
| Điểm chồng lấn | `alpha ≈ 0.85`, viền đen mảnh; strip plot thêm jitter nhẹ; nhiều điểm thì 2D density |
| So sánh nhiều phương pháp | **Small multiples** (các panel cùng trục), không chồng 6 đường |
| Nhãn | **Nhãn trực tiếp** thay chú giải khi được; chú giải riêng cho màu và cho hình khi hai đại lượng khác nhau |
| Mẫu bất thường | Khoanh tròn đen và ghi mã mẫu, ví dụ "mẫu 21 (qc_outlier)" |
| t-SNE/UMAP | Ẩn số trên trục; ghi chú "khoảng cách giữa các cụm không có ý nghĩa định lượng" |
| Độ bất định | Dải sai số hoặc khoảng tin cậy; nói rõ là SD hay CI 95% |
| Đường tham chiếu | Đường đứt xám (ví dụ 0,50 cho nhãn hoán vị) kèm giá trị |

## 4. Tên hình và tên file

- Tiêu đề hình: `F<số>. <câu hỏi hoặc mô tả trung tính>`. **Không đặt tiêu đề khẳng định kết quả trước khi có số liệu** (ví dụ dùng "Phân phối biểu hiện trước và sau log10", không dùng "Log10 làm hết lệch").
- File: `F<NN>[chữ]_<mô_tả_ngắn>_<report|slide>.png` trong `results/figures/report/` và `results/figures/slides/`.
- Mã hình đã dùng:

| Mã | Nội dung | Người vẽ | Nơi vẽ |
|---|---|---|---|
| F01 | Thiết kế mẫu | TV1 | `notebooks/02_design_batch.ipynb` |
| F02, F02b | Histogram trước/sau log10; độ biến thiên gen và lọc | TV6 | `src/viz.py` |
| F03 | Boxplot từng mẫu | TV6 | `src/viz.py` |
| F04a, F04b, F04c | Scree; PCA theo lớp và nhóm con; PCA/t-SNE/UMAP | TV6 | `src/viz.py` |
| F04d | PCA tô theo nguồn và loại mô (batch effect) | TV1 | `notebooks/02_design_batch.ipynb` |
| F04e | Tương quan giữa các mẫu + phân cụm phân cấp | TV6 | `src/viz.py` |
| F05–F09, F12 | chưa có | | chờ giai đoạn 2 |
| F10, F11 | Selection bias; ROC + ma trận nhầm lẫn | TV2 | `src/evaluate.py` |

F04d giữ nguyên tên TV1 đã đặt (không trùng với F04a–c, F04e của 1.4).

## 5. Cách dùng `src/viz.py`

```python
from src.viz import pca_scatter, plot_hist_log, save_figure, compute_pca

fig = pca_scatter(Xf, samples, color="class", shape="class")             # bản báo cáo
fig = pca_scatter(Xf, samples, color="source", shape="class", mode="slide")
save_figure(fig, "F04x_ten_ngan", "report")                              # -> results/figures/report/F04x_ten_ngan_report.png
```
- Mọi hàm vẽ nhận `mode="report"|"slide"` và trả về `Figure`; truyền `ax=` để vẽ vào panel có sẵn (khi đó trả về `ax`).
- Vẽ hình mới: viết hàm mới trong `viz.py` có decorator `@styled` (tự áp cỡ chữ theo `mode`), lấy màu từ `COLOR_MAPS`, rồi thêm vào `render_eda` hoặc hàm vẽ riêng của giai đoạn 2. Thêm test vào `tests/test_viz.py`.
- `python -m src.viz` (hoặc `make figures`) vẽ lại các hình EDA từ dữ liệu tầng `curated`.

## 6. Checklist nghiệm thu một hình (TV1 đánh dấu)

- [ ] Hình trả lời đúng một câu hỏi; tiêu đề đúng mẫu mục 4
- [ ] Mọi chữ ≥ 10 pt (báo cáo) / 16 pt (slide) ở kích thước in thật; vẽ bằng `viz.py`
- [ ] Trục có tên và đơn vị; bar bắt đầu từ 0
- [ ] Màu đúng bảng mục 1; thang màu đúng loại (phân kỳ/tuần tự/định tính)
- [ ] Có mã hóa dư thừa (hình dạng hoặc gạch) và còn đọc được khi in đen trắng
- [ ] Nhãn trực tiếp ở nơi có thể; chỉ một màu nổi bật
- [ ] Có đủ hai bản `_report` và `_slide`, đúng quy ước tên file

## 7. Việc còn lại / không nhất quán cần xử lý ở công việc 2.6

- `F10`, `F11` (TV2) đang dùng màu `#2b7bba`/`#c0392b`, chữ 9 pt (báo cáo) và 17 pt (slide): khác bảng ở mục 1–2. Sẽ chuyển sang `viz.py` (ADR-009).
- `F01`, `F04d` (TV1) vẽ rộng 10–11 in với chữ 10 pt, nên khi chèn vào báo cáo chữ nhỏ hơn 10 pt. Cần vẽ lại bằng `figsize("report")`.
