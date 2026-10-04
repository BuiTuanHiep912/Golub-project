# Data dictionary (TV2 hoàn thiện, hạn CN 25/10)

## standardized/samples.parquet
| Cột | Kiểu | Giá trị | Nguồn |
|---|---|---|---|
| sample_id | int | 1–72 | Kaggle (patient), OpenIntro (Samples) |
| split | str | train / test | File Kaggle |
| class | str | ALL / AML | Kaggle actual.csv |
| subtype | str | B-ALL / T-ALL / AML | OpenIntro cancer |
| tissue | str | BM / PB | OpenIntro BM.PB |
| source | str | CALGB / CCG / DFCI / St-Jude | OpenIntro Source |
| sex | str | F / M / NA | OpenIntro Gender |
