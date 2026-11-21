import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from iavalss.simulate import make_catalogue, truth_masks
from iavalss.pipeline import run_pipeline
from iavalss import evaluate as ev
from iavalss.explain import build_candidate_table

DF = make_catalogue(n=4000, seed=1)
RES = run_pipeline(DF, n_mc=4, shortlist_per_kind=150, seed=0)


def test_catalogue_has_known_truth():
    true_rare, art = truth_masks(DF)
    assert true_rare.sum() > 0 and art.sum() > 0


def test_probabilities_are_valid():
    assert np.all((RES.P_anom >= 0) & (RES.P_anom <= 1))
    assert np.all((RES.p_art >= 0) & (RES.p_art <= 1))
    assert np.all(np.isfinite(RES.aware))


def test_aware_ranking_beats_random_at_top():
    true_rare, _ = truth_masks(DF)
    p = ev.precision_at_k(RES.aware, true_rare, 50)
    assert p > 3 * true_rare.mean()


def test_candidate_table_has_explanations():
    t = build_candidate_table(RES, n=10)
    assert len(t) == 10 and t["top_projections"].str.len().min() > 0
