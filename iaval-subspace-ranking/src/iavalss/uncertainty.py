"""Stage 3: uncertainty propagation and artefact down-weighting.

Monte Carlo: each shortlisted object is resampled n_mc times within its stated
(Gaussian) 1-sigma errors, every resample is scored with the SAME fitted reference
(ECDF-normalised, original object excluded from its own neighbourhood), and

    P(anomaly | x, sigma) = fraction of resamples whose object score I exceeds a
                            threshold tau (the (1 - anomaly_fraction) quantile of the
                            reference object scores).

Caveat: the catalogue values already contain one noise draw, so resampling around
them adds a second one (the usual "double-noise" approximation). P is therefore a
conservative stability measure, not a calibrated posterior.

Artefact probability (data-driven, no truth labels): the anomaly rate among
high-S/N, unflagged objects is taken as the baseline. In a low-S/N bin (or among
flagged objects) with anomaly rate r above that baseline b, the fraction
1 - b/r of the anomalies is attributed to noise / data quality:

    p_snr(bin) = max(0, 1 - b / r_bin),   p_flag = max(0, 1 - b / r_flagged)
    p_art      = 1 - (1 - p_snr)(1 - p_flag)

Aware score = (1 - p_art) * P(anomaly) + 1e-3 * I (the last term only breaks ties).
"""
from __future__ import annotations

import numpy as np

from .subspace import combine, SCORE_NAMES

TIE_EPS = 1e-3


def select_shortlist(S_ref: np.ndarray, weights, top_m: int, per_kind: int) -> np.ndarray:
    """Union of the top ``per_kind`` objects by the combined score and by each single score.

    Including every single-score ranking keeps the shortlist valid for other weightings
    (used by the weight-sensitivity ablation).
    """
    keep = set()
    I, _ = combine(S_ref, weights, top_m)
    keep.update(np.argsort(-I)[:per_kind].tolist())
    for s in range(len(SCORE_NAMES)):
        w = np.zeros(len(SCORE_NAMES))
        w[s] = 1.0
        Is, _ = combine(S_ref, w, top_m)
        keep.update(np.argsort(-Is)[:per_kind].tolist())
    return np.array(sorted(keep))


def mc_component_scores(scorer, X: np.ndarray, sigma: np.ndarray, idx: np.ndarray,
                        n_mc: int, rng: np.random.Generator) -> np.ndarray:
    """Scores of Monte Carlo resamples: array (n_mc, len(idx), n_proj, n_scores)."""
    Xs = np.tile(X[idx], (n_mc, 1))
    Xs = Xs + rng.normal(0.0, 1.0, Xs.shape) * np.tile(sigma[idx], (n_mc, 1))
    S = scorer.transform(Xs, self_idx=np.tile(idx, n_mc))
    return S.reshape(n_mc, len(idx), S.shape[1], S.shape[2])


def anomaly_probability(comp: np.ndarray, weights, top_m: int, tau: float) -> np.ndarray:
    """P(anomaly) per shortlisted object from MC component scores; always in [0, 1]."""
    n_mc, m = comp.shape[:2]
    flat = comp.reshape(n_mc * m, comp.shape[2], comp.shape[3])
    I, _ = combine(flat, weights, top_m)
    return (I.reshape(n_mc, m) > tau).mean(axis=0)


def artefact_probability(log_snr: np.ndarray, dq_flag: np.ndarray, anomalous: np.ndarray,
                         n_bins: int = 10, min_rate: float = 1e-4) -> np.ndarray:
    """Probability that an anomalous object's status is explained by low S/N / quality flag."""
    log_snr = np.asarray(log_snr, float)
    flagged = np.asarray(dq_flag) > 0
    anomalous = np.asarray(anomalous, bool)
    edges = np.quantile(log_snr, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, log_snr, side="right") - 1, 0, n_bins - 1)
    hi = (b >= n_bins // 2) & ~flagged
    base = max(anomalous[hi].mean() if hi.any() else min_rate, min_rate)
    rate_bin = np.array([anomalous[b == i].mean() if (b == i).any() else base
                         for i in range(n_bins)])
    p_snr = np.clip(1.0 - base / np.maximum(rate_bin, min_rate), 0.0, 1.0)[b]
    rate_flag = anomalous[flagged].mean() if flagged.any() else base
    p_flag = np.where(flagged, np.clip(1.0 - base / max(rate_flag, min_rate), 0.0, 1.0), 0.0)
    return 1.0 - (1.0 - p_snr) * (1.0 - p_flag)


def aware_score(P_full: np.ndarray, p_art: np.ndarray, I: np.ndarray) -> np.ndarray:
    return (1.0 - p_art) * P_full + TIE_EPS * I
