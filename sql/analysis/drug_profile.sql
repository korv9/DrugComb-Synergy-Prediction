-- Per-drug profile: how widely it was screened and how it tends to combine.
WITH long AS (
    SELECT drug_1 AS drug_key, drug_2 AS partner, cell_key, zip FROM fact_combination
    UNION ALL
    SELECT drug_2, drug_1, cell_key, zip FROM fact_combination
)
SELECT
    d.drug_name,
    COUNT(*)                       AS combinations,
    COUNT(DISTINCT partner)        AS partners,
    COUNT(DISTINCT cell_key)       AS cell_lines,
    ROUND(AVG(zip), 2)             AS mean_zip,
    ROUND(AVG((zip > 10)::INT), 3) AS share_synergistic
FROM long l
JOIN dim_drug d USING (drug_key)
GROUP BY d.drug_name
ORDER BY combinations DESC
LIMIT 50;
