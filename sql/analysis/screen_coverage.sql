-- How densely is the drug x drug x cell space sampled?
WITH d AS (SELECT COUNT(*) AS n FROM dim_drug),
     c AS (SELECT COUNT(DISTINCT cell_key) AS n FROM fact_combination)
SELECT
    (SELECT COUNT(*) FROM fact_combination)                       AS observed,
    (d.n * (d.n - 1) / 2) * c.n                                   AS possible,
    ROUND((SELECT COUNT(*) FROM fact_combination) / ((d.n * (d.n - 1) / 2.0) * c.n), 8)
                                                                  AS density,
    (SELECT COUNT(*) FROM (SELECT pair_key FROM fact_combination
                           GROUP BY pair_key HAVING COUNT(*) = 1)) AS pairs_in_one_cell_line
FROM d, c;
