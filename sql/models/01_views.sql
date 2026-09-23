-- Enriched analytical view on top of the star schema.
CREATE OR REPLACE VIEW v_combination AS
SELECT
    f.*,
    d1.drug_name                         AS drug_1_name,
    d2.drug_name                         AS drug_2_name,
    d1.drug_name || ' + ' || d2.drug_name AS pair_name,
    c.cell_name,
    c.model_id,
    c.lineage,
    c.primary_disease,
    d1.has_structure AND d2.has_structure AS has_both_structures,
    COALESCE(c.has_rna, FALSE)           AS has_rna
FROM fact_combination f
JOIN dim_drug d1 ON d1.drug_key = f.drug_1
JOIN dim_drug d2 ON d2.drug_key = f.drug_2
JOIN dim_cell c  ON c.cell_key  = f.cell_key;
