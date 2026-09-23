-- Headline numbers for the dataset.
SELECT
    COUNT(*)                                              AS combinations,
    COUNT(DISTINCT pair_key)                              AS drug_pairs,
    (SELECT COUNT(*) FROM dim_drug)                       AS drugs,
    COUNT(DISTINCT cell_key)                              AS cell_lines,
    COUNT(DISTINCT lineage) FILTER (WHERE lineage <> 'Unknown') AS lineages,
    ROUND(AVG(zip), 2)                                    AS mean_zip,
    ROUND(AVG((synergy_class = 'synergistic')::INT), 4)   AS share_synergistic,
    ROUND(AVG((synergy_class = 'antagonistic')::INT), 4)  AS share_antagonistic,
    ROUND(AVG(has_both_structures::INT), 4)               AS share_with_structures,
    ROUND(AVG(has_rna::INT), 4)                           AS share_with_rna
FROM v_combination;
