"""Stage 2: subspace (2D projection) scoring.

For every pair of properties we compute three interpretable, rank-normalised
scores per object, each in [0, 1]:

density_contrast
    log ratio of the local density at a small kNN scale to that at a larger kNN
    scale. High = the object sits in a compact clump relative to its surroundings
    (what a small rare subclass looks like).
distribution_deviation
    |pointwise mutual information - its catalogue median| where
    PMI = log p(x,y) / (p(x) p(y)); the joint density is a kNN estimate, the
    marginals are equal-count histograms. High = the (x, y) combination is far
    from what the two marginals alone would suggest (a histogram-divergence style
    deviation from marginal independence, resolved per object).
local_outlyingness
    A local-outlier-factor-like ratio: the object's mean kNN distance over the
    mean kNN distance of its own neighbours. High = isolated relative to the
    local neighbourhood.

Raw scores are mapped to [0, 1] with the empirical CDF of the reference catalogue
so that new (e.g. Monte Carlo resampled) points can be scored on the same scale.
The per-projection interest is I_proj = sum_k w_k s_k (weights normalised to sum 1)
and the object-level score is the mean of its top-m projection scores.
"""
from __future__ import annotations

import itertools
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

SCORE_NAMES = ("density_contrast", "distribution_deviation", "local_outlyingness")
DEFAULT_WEIGHTS = {"density_contrast": 1.0, "distribution_deviation": 1.0,
                   "local_outlyingness": 1.0}


def normalise_weights(weights: Mapping[str, float] | Sequence[float] | None) -> np.ndarray:
    """Return non-negative weights in SCORE_NAMES order, rescaled to sum to 1."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if isinstance(weights, Mapping):
        unknown = set(weights) - set(SCORE_NAMES)
        if unknown:
            raise KeyError(f"unknown score names: {sorted(unknown)}")
        w = np.array([float(weights.get(s, 0.0)) for s in SCORE_NAMES])
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(SCORE_NAMES),):
            raise ValueError(f"need {len(SCORE_NAMES)} weights")
    if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and not all zero")
    return w / w.sum()


def rank_normalise(x: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] by (average) rank: 0 = smallest, 1 = largest."""
    from scipy.stats import rankdata
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.zeros(1)
    return (rankdata(x, method="average") - 1.0) / (x.size - 1.0)


def robust_standardise(X: np.ndarray):
    med = np.median(X, axis=0)
    scale = 1.4826 * np.median(np.abs(X - med), axis=0)
    scale[scale == 0] = 1.0
    return (X - med) / scale, med, scale


def _drop_self(d: np.ndarray, idx: np.ndarray, self_idx: np.ndarray, k: int):
    """From k+1 neighbours drop the one equal to self (or the farthest if absent)."""
    is_self = idx == self_idx[:, None]
    no_self = ~is_self.any(axis=1)
    is_self[no_self, -1] = True
    order = np.argsort(is_self, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(d, order, 1), np.take_along_axis(idx, order, 1)


def combine(S: np.ndarray, weights=None, top_m: int = 3):
    """I_proj = sum_k w_k s_k per projection; object score = mean of top-m I_proj.

    S has shape (n_obj, n_proj, n_scores). Returns (I, I_proj).
    """
    w = normalise_weights(weights)
    proj = S @ w
    P = proj.shape[1]
    m = max(1, min(top_m, P))
    top = np.partition(proj, P - m, axis=1)[:, P - m:]
    return top.mean(axis=1), proj


class SubspaceScorer:
    """Fit on a reference catalogue; score reference or new points in every 2D projection."""

    def __init__(self, prop_names: Sequence[str], k_small: int = 10, k_large: int = 60,
                 k_pmi: int = 20, k_lof: int = 20, n_marg_bins: int = 100, workers: int = -1):
        self.prop_names = list(prop_names)
        self.pairs = list(itertools.combinations(range(len(self.prop_names)), 2))
        self.k_small, self.k_large, self.k_pmi, self.k_lof = k_small, k_large, k_pmi, k_lof
        self.n_marg_bins = n_marg_bins
        self.workers = workers

    # ------------------------------------------------------------------ helpers
    def pair_name(self, p: int) -> str:
        i, j = self.pairs[p]
        return f"{self.prop_names[i]} vs {self.prop_names[j]}"

    def _marginal_logdens(self, j: int, v: np.ndarray) -> np.ndarray:
        edges, dens = self._edges[j], self._dens[j]
        b = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(dens) - 1)
        return np.log(dens[b])

    def _pmi(self, p: int, Q: np.ndarray, d: np.ndarray) -> np.ndarray:
        i, j = self.pairs[p]
        rk = d[:, self.k_pmi - 1] + 1e-9
        log_rho2 = np.log(self.k_pmi / (np.pi * rk ** 2 * self.n_))
        return log_rho2 - self._marginal_logdens(i, Q[:, 0]) - self._marginal_logdens(j, Q[:, 1])

    def _raw(self, p: int, Q: np.ndarray, d: np.ndarray, idx: np.ndarray) -> np.ndarray:
        eps = 1e-9
        rs, rl = d[:, self.k_small - 1] + eps, d[:, self.k_large - 1] + eps
        dc = np.log(self.k_small / self.k_large) + 2.0 * np.log(rl / rs)
        dev = np.abs(self._pmi(p, Q, d) - self._pmi_med[p])
        lof = (np.log(d[:, :self.k_lof].mean(axis=1) + eps)
               - np.log(self._ref_md[p][idx[:, :self.k_lof]].mean(axis=1) + eps))
        return np.column_stack([dc, dev, lof])

    def _query(self, p: int, Zq: np.ndarray, self_idx):
        i, j = self.pairs[p]
        Q = Zq[:, [i, j]]
        kq = self.k_large + (1 if self_idx is not None else 0)
        d, idx = self.trees_[p].query(Q, k=kq, workers=self.workers)
        if self_idx is not None:
            d, idx = _drop_self(d, idx, np.asarray(self_idx), self.k_large)
        return Q, d, idx

    # ---------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray) -> "SubspaceScorer":
        X = np.asarray(X, dtype=float)
        self.n_, n_prop = X.shape
        if n_prop != len(self.prop_names):
            raise ValueError("X has wrong number of columns")
        if self.n_ <= self.k_large + 2:
            raise ValueError("catalogue too small for k_large")
        Z, self.med_, self.scale_ = robust_standardise(X)
        self._Zref = Z
        self._edges, self._dens = [], []
        for j in range(n_prop):
            e = np.unique(np.quantile(Z[:, j], np.linspace(0, 1, self.n_marg_bins + 1)))
            cnt, _ = np.histogram(Z[:, j], bins=e)
            self._edges.append(e)
            self._dens.append(np.maximum(cnt, 1) / self.n_ / np.diff(e))
        self.trees_, self._ref_md, self._pmi_med = [], {}, {}
        self._sorted = []
        S = np.empty((self.n_, len(self.pairs), len(SCORE_NAMES)), dtype=np.float32)
        ref_idx = np.arange(self.n_)
        for p, (i, j) in enumerate(self.pairs):
            self.trees_.append(cKDTree(Z[:, [i, j]]))
            Q, d, idx = self._query(p, Z, ref_idx)
            self._ref_md[p] = d[:, :self.k_lof].mean(axis=1)
            self._pmi_med[p] = np.median(self._pmi(p, Q, d))
            raw = self._raw(p, Q, d, idx)
            self._sorted.append([np.sort(raw[:, s]) for s in range(raw.shape[1])])
            S[:, p, :] = self._ecdf(p, raw)
        self.S_ref_ = S
        return self

    def _ecdf(self, p: int, raw: np.ndarray) -> np.ndarray:
        out = np.empty(raw.shape, dtype=np.float32)
        for s in range(raw.shape[1]):
            ref = self._sorted[p][s]
            out[:, s] = (np.searchsorted(ref, raw[:, s], "left")
                         + np.searchsorted(ref, raw[:, s], "right")) / (2.0 * len(ref))
        return out

    # --------------------------------------------------------------- transform
    def transform(self, X: np.ndarray, self_idx=None) -> np.ndarray:
        """Rank-normalised scores (n, n_proj, n_scores) for arbitrary points.

        If the points are perturbed copies of reference objects, pass ``self_idx``
        (the reference index of each) so the original is excluded from its own
        neighbourhood.
        """
        Z = (np.asarray(X, dtype=float) - self.med_) / self.scale_
        S = np.empty((len(Z), len(self.pairs), len(SCORE_NAMES)), dtype=np.float32)
        for p in range(len(self.pairs)):
            Q, d, idx = self._query(p, Z, self_idx)
            raw = self._raw(p, Q, d, idx)
            S[:, p, :] = self._ecdf(p, raw)
        return S
