-- How much do the four synergy reference models agree?
SELECT 'ZIP vs Bliss' AS pair, ROUND(CORR(zip, bliss), 3) AS pearson FROM fact_combination
UNION ALL SELECT 'ZIP vs Loewe', ROUND(CORR(zip, loewe), 3) FROM fact_combination
UNION ALL SELECT 'ZIP vs HSA',   ROUND(CORR(zip, hsa), 3)   FROM fact_combination
UNION ALL SELECT 'Bliss vs Loewe', ROUND(CORR(bliss, loewe), 3) FROM fact_combination
UNION ALL SELECT 'Bliss vs HSA', ROUND(CORR(bliss, hsa), 3) FROM fact_combination
UNION ALL SELECT 'Loewe vs HSA', ROUND(CORR(loewe, hsa), 3) FROM fact_combination;
