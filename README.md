# Golub Leukemia — High-Dimensional Data Analysis

Mini project: phân loại bạch cầu cấp ALL/AML từ dữ liệu biểu hiện gen (Golub et al., 1999).
72 mẫu × 7129 probe (p >> n). Chi tiết: xem tài liệu kế hoạch triển khai.

## Cài đặt
```bash
git clone <link-repo>
cd golub-project
conda env create -f environment.yml && conda activate golub
# hoặc: python -m venv .venv && pip install -r requirements.txt
```

## Dữ liệu
1. Tải 3 file từ Kaggle "Gene expression dataset (Golub et al.)" vào `data/raw/kaggle/`
2. Tải `golub.csv` từ OpenIntro vào `data/raw/openintro/`
3. Ghi ngày tải và MD5 vào `data/raw/SOURCES.md`
4. Chạy `make data` (Windows: `python run_all.py data`)

## Cấu trúc
| Thư mục | Nội dung |
|---|---|
| data/ | raw → standardized → cleansed → curated (không đưa lên Git), sandbox/ nháp cá nhân |
| config/ | experiment_config.yaml |
| src/ | code dùng chung (load, quality, preprocess, golub_wv, features, models, evaluate, viz) |
| notebooks/ | 01_eda … 09_interpretation |
| results/ | metrics/, figures/report/, figures/slides/, gene_lists/ |
| tests/ | kiểm thử dữ liệu và chống rò rỉ |
| docs/ | data dictionary, style guide, nhật ký quyết định, biên bản, bàn giao, báo cáo |

## Quy tắc làm việc
- Mỗi công việc một nhánh: `git checkout -b 1.1-load-data`
- Mở pull request, điền mẫu có sẵn (kèm demo nghiệm thu), cần 1 người review chéo duyệt
- Không commit dữ liệu; không sửa tay file trong data/
