-- ZIP distribution per tissue lineage (lineages with >= 500 combinations).
SELECT
    lineage,
    COUNT(*)                                  AS combinations,
    COUNT(DISTINCT cell_key)                  AS cell_lines,
    ROUND(AVG(zip), 2)                        AS mean_zip,
    ROUND(QUANTILE_CONT(zip, 0.25), 2)        AS q25,
    ROUND(MEDIAN(zip), 2)                     AS median_zip,
    ROUND(QUANTILE_CONT(zip, 0.75), 2)        AS q75,
    ROUND(AVG((synergy_class = 'synergistic')::INT), 4) AS share_synergistic
FROM v_combination
GROUP BY lineage
HAVING COUNT(*) >= 500
ORDER BY median_zip DESC;
