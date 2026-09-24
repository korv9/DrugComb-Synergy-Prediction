-- Screens pooled in DrugCombDB: size, scope and synergy level per study.
SELECT
    study,
    COUNT(*)                                            AS combinations,
    COUNT(DISTINCT pair_key)                            AS drug_pairs,
    COUNT(DISTINCT cell_key)                            AS cell_lines,
    ROUND(AVG(zip), 2)                                  AS mean_zip,
    ROUND(STDDEV_SAMP(zip), 2)                          AS sd_zip,
    ROUND(QUANTILE_CONT(zip, 0.25), 2)                  AS q25,
    ROUND(MEDIAN(zip), 2)                               AS median_zip,
    ROUND(QUANTILE_CONT(zip, 0.75), 2)                  AS q75,
    ROUND(AVG((synergy_class = 'synergistic')::INT), 4) AS share_synergistic
FROM fact_combination
GROUP BY study
HAVING COUNT(*) >= 100
ORDER BY combinations DESC;
