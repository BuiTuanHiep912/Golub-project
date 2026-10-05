# Nguồn dữ liệu gốc (tầng raw)

Không chỉnh sửa các file trong data/raw/. Ghi lại thông tin mỗi khi tải.

| File | Nguồn | Ngày tải | Người tải      | MD5                              |
|---|---|----------|----------------|----------------------------------|
| kaggle/data_set_ALL_AML_train.csv | Kaggle "Gene expression dataset (Golub et al.)": https://www.kaggle.com/datasets/crawford/gene-expression | 05/10/2026 | nguyenlamvu123 | 8391cdcbc4d7d8618a0587ff92fa4d80 |
| kaggle/data_set_ALL_AML_independent.csv | Kaggle (như trên) | 05/10/2026 | nguyenlamvu123 | 4b4f0dfc941b0662594e8848ec2d8b06 |
| kaggle/actual.csv | Kaggle (như trên) | 05/10/2026 | nguyenlamvu123 | b8d0bd345a4c32fbd81629924cffe804 |
| openintro/golub.csv | https://www.openintro.org/data/index.php?data=golub | 05/10/2026 | nguyenlamvu123 | cb8d3b313e6537f8e52f44a489c3dc2a |

Tính MD5: `md5sum <file>` (Linux/Mac) hoặc `certutil -hashfile <file> MD5` (Windows).

Tải lại đúng các file trên (đã đối chiếu MD5 khớp cả 4 file ngày 06/10/2026):

```bash
curl -L -o gene-expression.zip https://www.kaggle.com/api/v1/datasets/download/crawford/gene-expression
unzip gene-expression.zip -d data/raw/kaggle/
curl -L -o data/raw/openintro/golub.csv https://www.openintro.org/data/csv/golub.csv
md5sum data/raw/kaggle/*.csv data/raw/openintro/golub.csv   # so với bảng trên
```
