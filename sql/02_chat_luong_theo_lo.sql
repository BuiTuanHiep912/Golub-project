-- Câu hỏi: lô (phòng lab) nào đạt kiểm tra, có bao nhiêu mẫu bị gắn cờ, chất lượng chip
-- trung bình ra sao? Dùng cho màn hình Chất lượng dữ liệu và hình F8.
SELECT b.batch_id,
       b.status,
       count(q.sample_id)                          AS n_samples,
       sum(q.qc_outlier::INT)                      AS n_qc_outlier,
       round(median(q.median_corr_other), 3)       AS median_corr_other,
       round(avg(q.present_rate), 3)               AS mean_present_rate,
       round(avg(q.pct_below_100), 3)              AS mean_pct_below_100
FROM dim_batch b
LEFT JOIN fact_qc q USING (batch_id)
GROUP BY b.batch_id, b.status
ORDER BY b.batch_id;
