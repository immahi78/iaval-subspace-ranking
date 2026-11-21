"""Baseline rankers: random, Mahalanobis distance, IsolationForest (same inputs)."""
from __future__ import annotations

import time

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest

from .subspace import robust_standardise


def random_scores(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def mahalanobis_scores(X: np.ndarray) -> np.ndarray:
    Z, _, _ = robust_standardise(X)
    return EmpiricalCovariance().fit(Z).mahalanobis(Z)


def isolation_forest_scores(X: np.ndarray, seed: int = 0, n_estimators: int = 200) -> np.ndarray:
    Z, _, _ = robust_standardise(X)
    model = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(Z)
    return -model.score_samples(Z)  # higher = more anomalous


def all_baselines(X: np.ndarray, seed: int = 0):
    """Return (scores dict, runtime dict in seconds)."""
    out, rt = {}, {}
    for name, fn in (("random", lambda: random_scores(len(X), seed)),
                     ("mahalanobis", lambda: mahalanobis_scores(X)),
                     ("isolation_forest", lambda: isolation_forest_scores(X, seed))):
        t = time.perf_counter()
        out[name] = fn()
        rt[name] = time.perf_counter() - t
    return out, rt
