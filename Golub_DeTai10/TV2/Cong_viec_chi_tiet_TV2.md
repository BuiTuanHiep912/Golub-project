# TV2 – Công việc chi tiết (theo Bảng phân công 5.1)

> Nguồn: `Ke_hoach_trien_khai_project_Golub.pdf`, mục 5.1 (Bảng 5.2) + mô tả chi tiết mục 5.2 (1.1) và 5.3 (2.4).

TV2 (cùng TV1 là trưởng nhóm về hạ tầng) phụ trách **2 công việc kỹ thuật**, mỗi giai đoạn một việc:

| Giai đoạn | Công việc | Mạch liên kết |
|---|---|---|
| GĐ1 (tuần 1–4) | **1.1. Nạp, ghép dữ liệu và kiểm tra chất lượng** | Người dựng hạ tầng dữ liệu dựng tiếp hạ tầng đánh giá dùng chung |
| GĐ2 (tuần 5–8) | **2.4. Khung đánh giá và thí nghiệm selection bias** | |

---

## Phần A. GĐ1 – Mục 1.1. Nạp, ghép dữ liệu và kiểm tra chất lượng (deadline 10/10)

> **Lưu ý 06/10/2026:** code đã chuyển theo cấu trúc repo chuẩn `golub-project`
> (`src/`, `tests/`, `docs/`, `Makefile`, `environment.yml`). Thư mục `Golub_DeTai10/`
> giữ lại **tài liệu phân công** (README/Công việc chi tiết của TV1, TV2, TV6) và bản
> code làm việc cũ; nguồn chuẩn dùng để merge là `src/` + `tests/`.

### Mục tiêu
Đưa bộ dữ liệu Golub từ Kaggle và metadata OpenIntro vào dạng dùng được cho cả nhóm, đáng tin cậy và tái lập được.

### Công việc cụ thể (trạng thái: **đã xong**)
1. ✅ Tải 3 file Kaggle (`data_set_ALL_AML_train.csv`, `data_set_ALL_AML_independent.csv`, `actual.csv`) và `golub.csv` của OpenIntro vào `data/raw/{kaggle,openintro}/`; `data/raw/SOURCES.md` ghi MD5 từng file (ADR-005: người tải ghi tay ngày tải/người tải, code chỉ đối chiếu).
2. ✅ Dựng cấu trúc repo (mục 2.2): `src/`, `tests/`, `docs/`, `config/`, `notebooks/`, `results/`, `data/` 5 tầng; `environment.yml`, `requirements.txt`, `Makefile`, `run_all.py`, `src/config.py`.
3. ✅ `src/load.py` theo quy trình ghép mục 2.1.2 → **4 parquet**: `expression.parquet` (72×7129, int), `calls.parquet` (72×7129, P/A/M), `samples.parquet` (72×6), `genes.parquet` (7129×3).
4. ✅ `src/quality.py` kiểm tra **4 mức chất lượng** (mục 2.5) → tầng `data/cleansed/` + `quality_report.md`.
5. ✅ `tests/test_load.py` — **15 test** cho toàn bộ Bảng 2.4; `docs/data_dictionary.md` + `docs/decisions.md` (ADR-001…005).

### Sản phẩm bàn giao
- `src/config.py`, `src/load.py`, `src/quality.py`, `tests/test_load.py`
- `data/standardized/`, `data/cleansed/` (sinh tự động, không commit), `data/raw/SOURCES.md`
- `docs/data_dictionary.md`, `docs/decisions.md`; mục 2.1–2.5, 2.7 của báo cáo

### Demo nghiệm thu (đã chạy thật tại gốc repo `golub-project/`)
```bash
$ make standardize      # python3 -m src.load
$ make quality          # python3 -m src.quality
$ make test             # pytest tests/
15 passed, 1 skipped    # 1 skipped = test_no_leakage.py của TV3 (công việc 1.2)
```
```python
>>> from src.load import load_golub
>>> X, samples = load_golub(layer='cleansed')
>>> X.shape, samples.shape
((72, 7129), (72, 7))        # 6 cột mô tả + cột cờ qc_outlier
>>> samples['class'].value_counts()
ALL 47
AML 25
>>> pd.crosstab(samples['split'], samples['class'])
class ALL AML
split
test  20  14
train 27  11
>>> pd.crosstab([samples['split'], samples['class']], samples['source'])
source      CALGB CCG DFCI St-Jude
split class
test  ALL     0   0   17       3
      AML     4   5    0       5
train ALL     0   0   27       0
      AML    11   0    0       0
```
Trích `data/cleansed/quality_report.md`:
```text
[PASS] 72 mẫu, 7129 probe, kích thước X = (72, 7129)
[PASS] Nhãn Kaggle khớp subtype OpenIntro: 72/72
[INFO] Số giá trị < 100: 241357; > 16000: 4259; NaN: 0; giá trị âm: 147174
[INFO] Số mẫu thiếu sex: 23/72 (OpenIntro thiếu - không tự điền)
[INFO] Mẫu gắn cờ qc_outlier: 0 (không xóa nếu chưa có căn cứ)
```

---

## Phần B. GĐ2 – Mục 2.4. Khung đánh giá và thí nghiệm selection bias (tuần 5–7)

> **Trạng thái:** mục 2.4 chưa merge vào `src/` của repo chuẩn (đang chờ TV3 có
> `src/preprocess.py` + pipeline thật). Bản làm việc đã chạy trên dữ liệu thật nằm ở
> `Golub_DeTai10/TV2/code/`; số liệu demo dưới đây lấy từ bản đó.

### Mục tiêu
Đảm bảo mọi con số hiệu năng là trung thực và chứng minh bằng thực nghiệm tác hại của đánh giá sai (giả thuyết H3).

### Công việc cụ thể
1. Viết `src/evaluate.py` với hàm `evaluate(pipe, X, y, scheme)` cho **4 sơ đồ**: `original_split` (38/34), `loocv`, `nested_cv`, `wrong_cv`.
2. Tính **balanced accuracy, AUC**, khoảng tin cậy **Wilson** cho scheme `original_split`.
3. Thí nghiệm **selection bias**: 100 lần hoán vị nhãn × {CV đúng, CV sai}; vẽ hình **F10**.
4. Vẽ **ROC** và **ma trận nhầm lẫn** cho mô hình cuối (hình **F11**).
5. Review code TV1/TV3/TV4 phát hiện rò rỉ dữ liệu (đã xong TV1 → `docs/leakage_review.md`; TV3/TV4 khi có code).

### Sản phẩm bàn giao
- `src/evaluate.py`, `src/selection_bias.py`, `src/roc_cm.py`
- `docs/leakage_review.md`; hình F10, F11; số liệu `results/metrics/selection_bias_{summary,permutations}.csv`; mục 3.5 báo cáo

### Demo nghiệm thu
```bash
$ make all            # gốc repo: data → cleansed → test → evaluate → figures
$ make evaluate
original_split bal_acc=0.644  auc=0.589   acc=0.647
loocv          bal_acc=0.395±0.489  auc=nan     (mỗi fold chỉ 1 mẫu - AUC không xác định)
nested_cv      bal_acc=0.555±0.138  auc=nan
wrong_cv       bal_acc=0.739±0.089  auc=nan
```
Bảng selection bias (F10 - đã chạy 100 hoán vị, dữ liệu thật):
| Nhãn | Cách CV | Balanced accuracy (TB ± SD) | Kết luận |
|---|---|---|---|
| Thật | Đúng | 0.970 ± 0.040 | cao |
| Thật | Sai | 0.969 ± 0.041 | cao, có thể cao hơn CV đúng |
| Hoán vị | Đúng | 0.510 ± 0.072 | xấp xỉ 0.50 |
| Hoán vị | Sai | 0.743 ± 0.077 | cao bất thường → **bằng chứng H3** |

---

## Hướng triển khai code (mục 1.1 đã merge, mục 2.4 đang chờ TV3/TV6)
Trong repo chuẩn `golub-project/`:
- `src/config.py` — đường dẫn repo, `PALETTE`, tập giá trị hợp lệ. ✅
- `src/load.py` — quy trình ghép mục 2.1.2, sinh 4 parquet, đối chiếu MD5. ✅
- `src/quality.py` — 4 mức kiểm tra, sinh `quality_report.md`. ✅
- `tests/test_load.py` — 15 test cho Bảng 2.4. ✅
- `docs/data_dictionary.md`, `docs/decisions.md` (ADR-001…005). ✅

Còn lại cho GĐ2 (mục 2.4, khi TV3 có `src/preprocess.py` + pipeline thật):
- `src/evaluate.py` — 4 sơ đồ CV (`original_split`, `loocv`, `nested_cv`, `wrong_cv`) + chỉ số (balanced acc, AUC, Wilson CI). Bản làm việc: `Golub_DeTai10/TV2/code/evaluate.py`.
- `src/selection_bias.py` — thí nghiệm hoán vị nhãn × {CV đúng, sai}, xuất số liệu cho F10.
- `src/roc_cm.py` — ROC + confusion matrix cho mô hình cuối (F11).
- `docs/leakage_review.md` — review rò rỉ dữ liệu (bản nháp: `Golub_DeTai10/docs/leakage_review.md`).