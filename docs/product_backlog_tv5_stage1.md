# Product Backlog — Thành viên 5: Giai đoạn 1 (Class Discovery)

**Dự án:** High-Dimensional Data Analysis — Phân loại bạch cầu cấp ALL/AML (Golub et al., 1999)  
**Vai trò:** Thành viên 5 (TV5)  
**Mã công việc:** 1.5 — Khám phá lớp không giám sát (Unsupervised Class Discovery & Clustering)  
**Thời gian thực hiện:** Tuần 3–4 (hoàn thành trước mốc nghiệm thu CN 01/11)  
**Mục tiêu nghiên cứu:** Kiểm chứng giả thuyết **H5**: *"Dữ liệu biểu hiện gen có cấu trúc cụm tự nhiên trùng khớp với hai nhóm bệnh ALL/AML và các phân nhóm subtype (B-ALL, T-ALL) mà không cần dùng nhãn lâm sàng?"*

---

## 1. Tổng quan Epic & User Stories

### Epic: Unsupervised Class Discovery on Microarray Gene Expression
Là một nhà khoa học dữ liệu / nghiên cứu sinh học tính toán, tôi muốn khám phá cấu trúc tự nhiên của dữ liệu biểu hiện gen microarray trên 72 bệnh nhân mà không sử dụng nhãn chẩn đoán lâm sàng, nhằm tái lập phát hiện lịch sử của Golub et al. (1999), đánh giá độ ổn định và định lượng mức độ tách biệt của các phân nhóm bệnh.

---

### User Story 1 (US-1): Chuẩn bị và kiểm tra dữ liệu đầu vào cho phân cụm
* **Mô tả:** Đọc dữ liệu từ tầng `curated` (`X_log10_filtered.parquet` với 3571 gen đã qua lọc Dudoit và `samples.parquet`) thông qua API `load_golub()`.
* **Tiêu chí chấp nhận (Acceptance Criteria):**
  - [x] Đọc ma trận $X$ kích thước $72 \times 3571$ và $samples$ kích thước $72 \times 7$.
  - [x] Kiểm tra tính toàn vẹn: không có giá trị NaN, khớp chỉ mục `sample_id` từ 1 đến 72.
  - [x] Tách biệt tập nhãn đối chứng (`class`: ALL/AML; `subtype`: B-ALL, T-ALL, AML).

### User Story 2 (US-2): Phân cụm K-Means & Đánh giá độ ổn định
* **Mô tả:** Cài đặt thuật toán K-Means cho $k=2$ (phân biệt ALL vs AML) và $k=4$ (phân biệt AML, B-ALL, T-ALL). Đánh giá độ ổn định qua 10 random seeds khác nhau.
* **Tiêu chí chấp nhận (Acceptance Criteria):**
  - [x] Chạy K-Means trên 10 seeds ($0 \dots 9$), ghi nhận Adjusted Rand Index (ARI) và Silhouette Score.
  - [x] Lập bảng chéo (`crosstab`) giữa cụm phân bổ và nhãn thực tế.
  - [x] Báo cáo giá trị trung bình $\pm$ độ lệch chuẩn của ARI và Silhouette.

### User Story 3 (US-3): Phân cụm phân cấp (Hierarchical Clustering)
* **Mô tả:** Cài đặt Hierarchical Clustering với khoảng cách $1 - \text{Pearson correlation}$ và phương pháp liên kết `average linkage` (theo chuẩn Golub 1999), đồng thời thử nghiệm liên kết `complete` và `ward` để so sánh hiện tượng cô lập ngoại lai (chaining effect).
* **Tiêu chí chấp nhận (Acceptance Criteria):**
  - [x] Xây dựng ma trận khoảng cách tương quan Pearson giữa 72 mẫu bệnh nhân.
  - [x] Phân cụm tại $k=2$ và $k=4$ với các phương pháp liên kết (`average`, `complete`, `ward`).
  - [x] Phân tích hiện tượng mẫu ngoại lai 21 bị tách riêng ở `average linkage` và sự cải thiện vượt trội khi dùng `complete` / `ward`.
  - [x] Tính toán chỉ số ARI và Silhouette Score cho từng cấu hình.

### User Story 4 (US-4): Tái lập mô hình Self-Organizing Map (SOM) của Golub 1999
* **Mô tả:** Dùng thư viện `minisom` để huấn luyện lưới SOM 1D ($1 \times 2$ nơ-ron) cho 2 cụm và 2D ($2 \times 2$ nơ-ron) cho 4 cụm trên cả tập 38 mẫu train và toàn bộ 72 mẫu.
* **Tiêu chí chấp nhận (Acceptance Criteria):**
  - [x] Huấn luyện SOM $1 \times 2$ nơ-ron, tính ARI và lập bảng chéo phân loại cụm.
  - [x] Đối chiếu chính xác với con số của bài báo gốc: SOM 2 cụm đạt 24/25 ALL ở cụm $A_1$ và 10/13 AML ở cụm $A_2$.
  - [x] Huấn luyện SOM $2 \times 2$ nơ-ron để kiểm tra khả năng phát hiện các phân nhóm B-ALL và T-ALL.

### User Story 5 (US-5): Trực quan hóa chuẩn mực — Hình F5 (Clustermap) & Chiếu giảm chiều
* **Mô tả:** Thiết kế Heatmap biểu hiện gen kèm Clustermap hai chiều (mẫu và gen) theo đúng hướng dẫn của `docs/style_guide.md` (màu ALL/AML, bảng màu phân kỳ z-score, nhãn trực tiếp).
* **Tiêu chí chấp nhận (Acceptance Criteria):**
  - [x] Xuất `results/figures/report/F05_clustermap_report.png` (độ phân giải cao 300 dpi, font $\ge 10\,\text{pt}$).
  - [x] Xuất `results/figures/slides/F05_clustermap_slide.png` (tối ưu trình chiếu slide, font $\ge 16\,\text{pt}$).
  - [x] Xuất hình F05b chiếu PCA PC1–PC2 tô màu theo cụm không giám sát so với nhãn thực tế.

### User Story 6 (US-6): Hoàn thiện Notebook, Báo cáo mục 3.2 và Biên bản bàn giao 1.5
* **Mô tả:** Tích hợp toàn bộ phân tích vào `notebooks/03_discovery.ipynb`, viết tài liệu báo cáo `docs/report/3.2_class_discovery.md` và ghi chú bàn giao `docs/handover/1.5.md`.
* **Tiêu chí chấp nhận (Acceptance Criteria):**
  - [x] Notebook chạy thông suốt từ đầu đến cuối (`Restart & Run All` không lỗi).
  - [x] Xuất file số liệu `results/metrics/cluster_discovery_metrics.csv`.
  - [x] Viết mục 3.2 của báo cáo với đầy đủ lý thuyết, phân tích số liệu, bảng so sánh và diễn giải sinh học.
  - [x] Hoàn thiện `docs/handover/1.5.md` chuẩn form và đưa ra kết luận rõ ràng về giả thuyết H5.

---

## 2. Bảng phân rã công việc (Task Breakdown & Acceptance Criteria)

| Mã Task | Hạng mục công việc | Output / File sinh ra | Trạng thái |
|---|---|---|---|
| **TSK-1.5.1** | Kiểm tra môi trường, cài đặt `minisom`, nạp dữ liệu từ `CURATED_DIR` | Môi trường Python sẵn sàng, test dữ liệu PASS | Đã sẵn sàng |
| **TSK-1.5.2** | Thực nghiệm K-Means ($k=2, k=4$) trên 10 seeds, tính ARI, Silhouette, độ ổn định | Ma trận kết quả, crosstab | Đã sẵn sàng |
| **TSK-1.5.3** | Thực nghiệm Hierarchical clustering với khoảng cách $1-\text{Pearson}$ và so sánh liên kết | Bảng so sánh linkages, phân tích mẫu 21 | Đã sẵn sàng |
| **TSK-1.5.4** | Huấn luyện SOM ($1\times2$ và $2\times2$) bằng MiniSom, đối chiếu bài báo gốc | Bảng so sánh với số liệu Golub 1999 | Đã sẵn sàng |
| **TSK-1.5.5** | Tạo file tổng hợp chỉ số định lượng | `results/metrics/cluster_discovery_metrics.csv` | Đã sẵn sàng |
| **TSK-1.5.6** | Trực quan hóa Clustermap F5 và PCA projection (bản report và slide) | `results/figures/report/F05*`, `results/figures/slides/F05*` | Đã sẵn sàng |
| **TSK-1.5.7** | Xây dựng Jupyter Notebook hoàn chỉnh có chú thích | `notebooks/03_discovery.ipynb` | Đã sẵn sàng |
| **TSK-1.5.8** | Soạn thảo chương 3.2 trong tài liệu báo cáo | `docs/report/3.2_class_discovery.md` | Đã sẵn sàng |
| **TSK-1.5.9** | Viết biên bản bàn giao nghiệm thu giai đoạn 1 | `docs/handover/1.5.md` | Đã sẵn sàng |

---

## 3. Định nghĩa hoàn thành (Definition of Done - DoD)
1. Tất cả các notebook, script chạy độc lập, tái lập 100% kết quả từ dữ liệu `curated`.
2. Bảng kết quả định lượng khớp định dạng yêu cầu của kế hoạch đề tài.
3. Hình vẽ tuân thủ nghiêm ngặt Style Guide của nhóm (bảng màu thân thiện với người mù màu, độ phân giải cao).
4. Giả thuyết nghiên cứu H5 được kết luận rành mạch dựa trên bằng chứng thực nghiệm và đối chiếu lịch sử.
5. Biên bản bàn giao `1.5.md` được ghi nhận đầy đủ để phục vụ buổi nghiệm thu ngày 01/11.
