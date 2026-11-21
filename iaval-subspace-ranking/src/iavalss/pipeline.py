"""Orchestrates the four stages: data engine -> subspace scoring -> uncertainty -> explain."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .simulate import PROPERTIES
from .subspace import SubspaceScorer, combine, normalise_weights
from . import uncertainty as unc


@dataclass
class PipelineResult:
    df: pd.DataFrame
    X: np.ndarray
    sigma: np.ndarray
    scorer: SubspaceScorer
    weights: np.ndarray
    top_m: int
    anomaly_fraction: float
    n_mc: int
    I: np.ndarray            # object score from measured values
    I_proj: np.ndarray       # (N, n_proj) per-projection scores
    tau: float
    shortlist: np.ndarray
    mc_comp: np.ndarray      # (n_mc, M, n_proj, n_scores)
    P_anom: np.ndarray       # (N,) zero outside the shortlist
    p_art: np.ndarray        # (N,)
    aware: np.ndarray        # (N,)
    timing: dict = field(default_factory=dict)


def prepare_data(df: pd.DataFrame, props=PROPERTIES):
    """Stage 1 (data engine): numeric matrices, error matrix, basic validation."""
    X = df[list(props)].to_numpy(dtype=float)
    sigma = df[[f"{p}_err" for p in props]].to_numpy(dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(sigma).all():
        raise ValueError("non-finite values in properties/errors")
    if (sigma < 0).any():
        raise ValueError("negative measurement errors")
    return X, sigma


def threshold(I: np.ndarray, anomaly_fraction: float) -> float:
    return float(np.quantile(I, 1.0 - anomaly_fraction))


def run_pipeline(df: pd.DataFrame, weights=None, anomaly_fraction: float = 0.03,
                 n_mc: int = 16, shortlist_per_kind: int = 800, top_m: int = 3,
                 seed: int = 0, props=PROPERTIES) -> PipelineResult:
    t0 = time.perf_counter()
    X, sigma = prepare_data(df, props)
    w = normalise_weights(weights)
    scorer = SubspaceScorer(props).fit(X)
    t1 = time.perf_counter()
    I, I_proj = combine(scorer.S_ref_, w, top_m)
    tau = threshold(I, anomaly_fraction)
    shortlist = unc.select_shortlist(scorer.S_ref_, w, top_m, shortlist_per_kind)
    rng = np.random.default_rng(seed)
    comp = unc.mc_component_scores(scorer, X, sigma, shortlist, n_mc, rng)
    P = np.zeros(len(X))
    P[shortlist] = unc.anomaly_probability(comp, w, top_m, tau)
    t2 = time.perf_counter()
    p_art = unc.artefact_probability(df["log_snr"].to_numpy(), df["dq_flag"].to_numpy(), I >= tau)
    aware = unc.aware_score(P, p_art, I)
    t3 = time.perf_counter()
    return PipelineResult(df, X, sigma, scorer, w, top_m, anomaly_fraction, n_mc, I, I_proj,
                          tau, shortlist, comp, P, p_art, aware,
                          {"subspace_scoring_s": t1 - t0, "monte_carlo_s": t2 - t1,
                           "artefact_stage_s": t3 - t2, "total_s": t3 - t0})


def aware_from_weights(res: PipelineResult, weights, use_mc: bool = True, use_dq: bool = True):
    """Recompute the ranking scores for other weights from cached component scores.

    Returns dict(I, P, p_art, aware). The shortlist is held fixed (it contains the top
    objects of each single score, so it covers any convex weighting approximately).
    """
    w = normalise_weights(weights)
    I, _ = combine(res.scorer.S_ref_, w, res.top_m)
    tau = threshold(I, res.anomaly_fraction)
    P = np.zeros(len(I))
    if use_mc:
        P[res.shortlist] = unc.anomaly_probability(res.mc_comp, w, res.top_m, tau)
    p_art = (unc.artefact_probability(res.df["log_snr"].to_numpy(), res.df["dq_flag"].to_numpy(),
                                      I >= tau) if use_dq else np.zeros(len(I)))
    base = P if use_mc else I
    aware = (1.0 - p_art) * base + unc.TIE_EPS * I
    return {"I": I, "P": P, "p_art": p_art, "aware": aware}
