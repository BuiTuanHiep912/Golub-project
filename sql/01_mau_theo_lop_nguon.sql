-- Câu hỏi: mỗi tập (train/test), mỗi lớp bệnh có bao nhiêu mẫu từ mỗi phòng lab?
-- Đối chiếu: Bảng 2.4 của kế hoạch (train 27 ALL DFCI + 11 AML CALGB;
-- test ALL 17 DFCI + 3 St-Jude, AML 4 CALGB + 5 St-Jude + 5 CCG).
SELECT split, class, source, count(*) AS n_samples
FROM dim_sample
GROUP BY split, class, source
ORDER BY split DESC, class, source;
