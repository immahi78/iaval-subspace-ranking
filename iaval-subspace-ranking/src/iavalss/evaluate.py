"""Injection-recovery metrics, ablations and rank stability (all on SIMULATED data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import perturb_catalogue, truth_masks
from .baselines import all_baselines
from .pipeline import run_pipeline, aware_from_weights, PipelineResult
from .subspace import SCORE_NAMES

KS = (100, 200, 500, 1000)


def ranking(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(scores), kind="stable")


def precision_at_k(scores, truth, k: int) -> float:
    return float(np.asarray(truth)[ranking(scores)[:k]].mean())


def recall_at_k(scores, truth, k: int) -> float:
    truth = np.asarray(truth)
    return float(truth[ranking(scores)[:k]].sum() / max(truth.sum(), 1))


def precision_curve(scores, truth, ks) -> np.ndarray:
    hit = np.cumsum(np.asarray(truth)[ranking(scores)])
    return np.array([hit[k - 1] / k for k in ks])


def topk_jaccard(a, b, k: int) -> float:
    sa, sb = set(ranking(a)[:k].tolist()), set(ranking(b)[:k].tolist())
    return len(sa & sb) / len(sa | sb)


def metrics_table(scores: dict, df: pd.DataFrame, ks=KS, runtimes: dict | None = None) -> pd.DataFrame:
    true, art = truth_masks(df)
    rows = []
    for name, s in scores.items():
        row = {"method": name}
        for k in ks:
            row[f"P@{k}"] = precision_at_k(s, true, k)
        for k in ks:
            row[f"R@{k}"] = recall_at_k(s, true, k)
        row["artefact@500"] = precision_at_k(s, art, 500)
        if runtimes and name in runtimes:
            row["runtime_s"] = runtimes[name]
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def ablation_scores(res: PipelineResult) -> dict:
    """Variants of the aware method (same fitted reference, same MC cache)."""
    w = res.weights
    return {
        "full (MC + DQ)": res.aware,
        "no uncertainty stage (DQ only)": aware_from_weights(res, w, use_mc=False)["aware"],
        "no data-quality stage (MC only)": aware_from_weights(res, w, use_dq=False)["aware"],
        "neither (plain I)": aware_from_weights(res, w, use_mc=False, use_dq=False)["aware"],
    }


def weight_variants(n_random: int = 20, seed: int = 0) -> dict:
    out = {"equal": (1, 1, 1)}
    for i, s in enumerate(SCORE_NAMES):
        w = [0, 0, 0]
        w[i] = 1
        out[f"only {s}"] = tuple(w)
    rng = np.random.default_rng(seed)
    for r in range(n_random):
        out[f"random_{r}"] = tuple(rng.dirichlet(np.ones(3)))
    return out


def weight_sensitivity(res: PipelineResult, df: pd.DataFrame, k: int = 500, n_random: int = 20,
                       seed: int = 0) -> pd.DataFrame:
    true, art = truth_masks(df)
    rows = []
    for name, w in weight_variants(n_random, seed).items():
        a = aware_from_weights(res, w)
        rows.append({"weights": name, "w_density": w[0] / sum(w), "w_dist": w[1] / sum(w),
                     "w_lof": w[2] / sum(w), f"P@{k}_aware": precision_at_k(a["aware"], true, k),
                     f"P@{k}_plainI": precision_at_k(a["I"], true, k),
                     f"artefact@{k}_aware": precision_at_k(a["aware"], art, k)})
    return pd.DataFrame(rows).set_index("weights")


def rank_stability(df: pd.DataFrame, base_scores: dict, n_real: int = 2, k: int = 500,
                   seed: int = 100, **pipeline_kw) -> pd.DataFrame:
    """Top-k Jaccard overlap between the original ranking and rankings recomputed on
    catalogues re-drawn within the stated errors. Random ordering is the floor."""
    from .pipeline import prepare_data
    vals = {m: [] for m in base_scores}
    for r in range(n_real):
        d2 = perturb_catalogue(df, seed + r)
        X2, _ = prepare_data(d2)
        sc2, _ = all_baselines(X2, seed=seed + r)
        res2 = run_pipeline(d2, seed=seed + r, **pipeline_kw)
        sc2["iavalss aware (full)"] = res2.aware
        sc2["iavalss plain I"] = res2.I
        for m in base_scores:
            if m == "random":
                continue
            vals[m].append(topk_jaccard(base_scores[m], sc2[m], k))
    rows = [{"method": m, f"jaccard@{k}": float(np.mean(v)), "n_realisations": len(v)}
            for m, v in vals.items() if v]
    return pd.DataFrame(rows).set_index("method")
