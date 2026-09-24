import numpy as np
import pandas as pd
import pytest

from drugsyn.download import resolve_depmap_release
from drugsyn.drugs import display_name, name_variants, standardise
from drugsyn.features import history_features, tanimoto
from drugsyn.ingest import norm_cell, norm_drug, stage_measurements
from drugsyn.splits import make_folds


def test_name_normalisation():
    assert norm_drug("  5–FU ") == "5-fu"
    assert norm_drug(float("nan")) is None
    assert norm_drug("5-fluoro-2\\'-deoxyuridine") == "5-fluoro-2'-deoxyuridine"
    assert norm_drug("(+\\\\/-)-sulfinpyrazone") == "(+/-)-sulfinpyrazone"
    assert norm_cell("NCI-H460") == "ncih460"
    assert norm_cell("786-O") == "786o"


def test_stage_measurements_funnel():
    raw = pd.DataFrame({
        "Drug1": ["a", "b", None, "c"], "Drug2": ["b", "c", "d", "d"],
        "Cell line": ["x", "x", "x", "x"], "ZIP": ["1.0", "oops", "2", "999"],
    })
    staged, funnel = stage_measurements(raw)
    assert [s["rows"] for s in funnel] == [4, 3, 2, 1]
    assert staged["zip"].tolist() == [1.0]


def test_salt_stripping_merges_structures():
    _, smi_a, key_a = standardise("O=c1[nH]cc(F)c(=O)[nH]1")
    _, smi_b, key_b = standardise("[Na+].O=c1[nH]cc(F)c(=O)[n-]1")
    assert key_a == key_b and smi_a == smi_b
    assert standardise("not a smiles") == (None, None, None)


def test_name_variants_strip_salts():
    assert "imatinib" in name_variants("imatinib mesylate")
    assert display_name("doxorubicin hydrochloride") == "doxorubicin"
    assert display_name("carfilzomib (pr-171)") == "carfilzomib"
    assert display_name("temozolomide") == "temozolomide"


def test_tanimoto():
    a = np.array([[1, 1, 0, 0]], np.uint8)
    b = np.array([[1, 0, 1, 0]], np.uint8)
    assert tanimoto(a, b)[0] == pytest.approx(1 / 3)


def _toy_fact(n=400, seed=0):
    rng = np.random.default_rng(seed)
    drugs = [f"d{i}" for i in range(12)]
    d = np.sort(np.stack([rng.choice(drugs, n), rng.choice(drugs, n)], 1), axis=1)
    fact = pd.DataFrame({"drug_1": d[:, 0], "drug_2": d[:, 1],
                         "cell_key": rng.choice([f"c{i}" for i in range(6)], n)})
    fact["pair_key"] = fact["drug_1"] + "|" + fact["drug_2"]
    return fact


@pytest.mark.parametrize("scheme", ["random", "cold_pair", "cold_drug", "cold_cell"])
def test_split_schemes_are_disjoint(scheme):
    fact = _toy_fact()
    for _, tr, te in make_folds(fact, scheme, n_folds=4):
        assert len(np.intersect1d(tr, te)) == 0 and len(te) > 0
        ftr, fte = fact.iloc[tr], fact.iloc[te]
        if scheme == "cold_pair":
            assert not set(ftr["pair_key"]) & set(fte["pair_key"])
        if scheme == "cold_cell":
            assert not set(ftr["cell_key"]) & set(fte["cell_key"])
        if scheme == "cold_drug":
            train_drugs = set(ftr["drug_1"]) | set(ftr["drug_2"])
            assert all((a not in train_drugs) or (b not in train_drugs)
                       for a, b in zip(fte["drug_1"], fte["drug_2"]))


def test_history_features_use_only_training_labels():
    fact = _toy_fact()
    y = np.zeros(len(fact))
    tr, te = fact.iloc[:300], fact.iloc[300:].copy()
    h1 = history_features(tr, y[:300], te, m=5)
    # changing test labels must not change test features
    h2 = history_features(tr, y[:300], te.assign(zip=1e6), m=5)
    pd.testing.assert_frame_equal(h1, h2)
    assert np.allclose(h1.filter(like="te_"), 0)


def test_unseen_keys_fall_back_to_prior():
    tr = pd.DataFrame({"drug_1": ["a"], "drug_2": ["b"], "cell_key": ["x"], "pair_key": ["a|b"]})
    te = pd.DataFrame({"drug_1": ["c"], "drug_2": ["d"], "cell_key": ["y"], "pair_key": ["c|d"]})
    h = history_features(tr, np.array([7.0]), te, m=1)
    assert np.allclose(h.filter(like="te_"), 7.0)
    assert np.allclose(h.filter(like="n_"), 0.0)


def test_depmap_release_resolution():
    index = pd.DataFrame({
        "release": ["R2", "R2", "R1", "R1", "R0"],
        "release_date": ["2025-05-01", "2025-05-01", "2024-11-01", "2024-11-01", "2024-05-01"],
        "filename": ["Model.csv", "Expr.csv", "Model.csv", "Expr.csv", "Model.csv"],
        "url": ["u1", "u2", "u3", "u4", "u5"],
    })
    rel, urls = resolve_depmap_release(index, ["Model.csv", "Expr.csv"], None)
    assert rel == "R2" and urls == {"Model.csv": "u1", "Expr.csv": "u2"}
    rel, _ = resolve_depmap_release(index, ["Model.csv", "Expr.csv"], "R1")
    assert rel == "R1"
    with pytest.raises(ValueError):
        resolve_depmap_release(index, ["Model.csv", "Expr.csv"], "R0")
