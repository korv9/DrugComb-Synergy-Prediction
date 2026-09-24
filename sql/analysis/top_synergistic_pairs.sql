-- Pairs that are synergistic *consistently* across many cell lines,
-- not just once: require >= 10 cell lines and rank by mean ZIP.
SELECT
    pair_name,
    COUNT(DISTINCT cell_key)                            AS cell_lines,
    ROUND(AVG(zip), 2)                                  AS mean_zip,
    ROUND(STDDEV_SAMP(zip), 2)                          AS sd_zip,
    ROUND(AVG((synergy_class = 'synergistic')::INT), 3) AS share_synergistic
FROM v_combination
GROUP BY pair_name
HAVING COUNT(DISTINCT cell_key) >= 10
ORDER BY mean_zip DESC
LIMIT 25;
