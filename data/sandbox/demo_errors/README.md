# Bộ file lỗi mẫu để demo (TV2 / 2.4)

Dùng để minh họa kiểm tra chất lượng/validate đầu vào trước khi chạy pipeline.

- `sample_missing_fields.json`: thiếu trường bắt buộc (split, source, sex).
- `sample_invalid_values.json`: giá trị ngoài danh mục/không khớp (class-subtype, tissue, source, split).
- `sample_bad_expression.csv`: ví dụ vector biểu hiện có giá trị không phải số/thiếu probe.
- `labels_bad.csv`: nhãn ngoài {ALL,AML} và trùng patient (demo lỗi lô actual).

Có thể dùng kèm `src/ingest.py`:
```bash
python -m src.ingest
python -c "import json; from src.ingest import validate_new_sample; print(validate_new_sample(json.load(open('data/sandbox/demo_errors/sample_invalid_values.json'))))"
```
