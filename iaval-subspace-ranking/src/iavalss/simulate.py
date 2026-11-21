"""Simulated quasar-like catalogue with injected rare populations.

EVERYTHING GENERATED HERE IS SIMULATED. It is a toy stand-in for a spectroscopic
quasar catalogue and contains no DESI (or any other real survey) data. The
distributions, correlations and error model are simple assumptions chosen only so
that the ranking layer can be exercised and tested against known ground truth.

Ground-truth labels (column ``sim_label``):
    normal          ordinary objects
    dusty_reddened  rare subclass shifted redder in several colours together
    bal_like        rare subclass with weaker line EW and broader line width
    artefact        low-S/N objects with spurious extreme values (not real)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROPERTIES = [
    "z",          # redshift
    "g_r",        # colour (mag)
    "r_z",        # colour (mag)
    "z_W1",       # optical-to-WISE colour (mag)
    "W1_W2",      # mid-IR colour (mag)
    "log_ew",     # log10 emission-line equivalent width (A)
    "log_fwhm",   # log10 emission-line FWHM (km/s)
    "log_lbol",   # log10 bolometric luminosity proxy (erg/s)
    "alpha_nu",   # continuum spectral slope
    "log_snr",    # log10 continuum S/N per pixel
]
ERR_COLS = [f"{p}_err" for p in PROPERTIES]
TRUE_LABELS = ("dusty_reddened", "bal_like")
ARTEFACT_LABEL = "artefact"

# Base 1-sigma measurement error at high S/N; it grows as sqrt(1 + (S0/snr)^2).
BASE_ERR = {
    "z": 0.0005, "g_r": 0.04, "r_z": 0.05, "z_W1": 0.08, "W1_W2": 0.10,
    "log_ew": 0.04, "log_fwhm": 0.03, "log_lbol": 0.05, "alpha_nu": 0.08,
    "log_snr": 0.0,
}
SNR_KNEE = 4.0


def error_scale(snr: np.ndarray) -> np.ndarray:
    """Multiplicative growth of errors at low S/N."""
    return np.sqrt(1.0 + (SNR_KNEE / np.asarray(snr)) ** 2)


def _base_population(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """True (noise-free) properties of ordinary objects, with mild correlations."""
    z = 0.5 + 3.3 * rng.beta(2.2, 3.0, n)
    log_lbol = 45.5 + 0.35 * (z - 2.0) + rng.normal(0, 0.35, n)
    alpha = rng.normal(-0.5, 0.30, n)
    g_r = 0.25 + 0.10 * (z - 2.0) + 0.5 * (alpha + 0.5) + rng.normal(0, 0.10, n)
    r_z = 0.20 + 0.04 * (z - 2.0) + 0.5 * (alpha + 0.5) + rng.normal(0, 0.10, n)
    z_w1 = 1.80 + 0.25 * (z - 2.0) + 0.9 * (alpha + 0.5) + rng.normal(0, 0.25, n)
    w1_w2 = 0.95 + 0.08 * (z - 2.0) + 0.3 * (alpha + 0.5) + rng.normal(0, 0.12, n)
    log_ew = 1.75 - 0.15 * (log_lbol - 45.5) + rng.normal(0, 0.15, n)
    log_fwhm = 3.55 + 0.20 * (log_lbol - 45.5) + rng.normal(0, 0.10, n)
    return {"z": z, "g_r": g_r, "r_z": r_z, "z_W1": z_w1, "W1_W2": w1_w2,
            "log_ew": log_ew, "log_fwhm": log_fwhm, "log_lbol": log_lbol,
            "alpha_nu": alpha}


def make_catalogue(
    n: int = 50_000,
    seed: int = 42,
    frac_dusty: float = 0.010,
    frac_bal: float = 0.008,
    frac_artefact: float = 0.004,
    flag_rate_normal: float = 0.03,
    flag_rate_artefact: float = 0.60,
) -> pd.DataFrame:
    """Build the simulated catalogue (observed values + per-object errors + labels).

    Observed value = true value + Gaussian noise with the stated per-object sigma.
    The sigma grows at low continuum S/N. Artefacts are low-S/N objects whose
    observed values in 2-3 properties are replaced by spurious extreme values.
    """
    rng = np.random.default_rng(seed)
    n_dusty = int(round(n * frac_dusty))
    n_bal = int(round(n * frac_bal))
    n_art = int(round(n * frac_artefact))
    n_norm = n - n_dusty - n_bal - n_art
    if n_norm <= 0:
        raise ValueError("fractions leave no normal objects")

    true = _base_population(n, rng)
    labels = np.array(["normal"] * n_norm + ["dusty_reddened"] * n_dusty
                      + ["bal_like"] * n_bal + [ARTEFACT_LABEL] * n_art, dtype=object)
    sl_dusty = slice(n_norm, n_norm + n_dusty)
    sl_bal = slice(n_norm + n_dusty, n_norm + n_dusty + n_bal)
    sl_art = slice(n_norm + n_dusty + n_bal, n)

    # --- dust-reddened subclass: correlated redward shift of colours -------------
    ebv = rng.lognormal(np.log(0.45), 0.30, n_dusty)
    true["g_r"][sl_dusty] += 1.5 * ebv
    true["r_z"][sl_dusty] += 1.1 * ebv
    true["z_W1"][sl_dusty] += 1.8 * ebv
    true["W1_W2"][sl_dusty] += 0.2 * ebv
    true["alpha_nu"][sl_dusty] += 0.4 * ebv

    # --- BAL-like subclass: weaker EW, broader width, only at z > 1.6 ------------
    true["z"][sl_bal] = 1.6 + 1.9 * rng.beta(2.0, 2.5, n_bal)
    true["log_ew"][sl_bal] -= rng.normal(0.50, 0.10, n_bal)
    true["log_fwhm"][sl_bal] += rng.normal(0.40, 0.08, n_bal)
    true["g_r"][sl_bal] += 0.08
    true["alpha_nu"][sl_bal] += 0.05

    # --- continuum S/N -----------------------------------------------------------
    log_snr = np.clip(rng.normal(0.50, 0.25, n), -0.2, 1.5)
    log_snr[sl_art] = np.clip(rng.normal(0.0, 0.15, n_art), -0.2, 0.35)  # low S/N
    snr = 10.0 ** log_snr
    scale = error_scale(snr)

    obs, err = {}, {}
    for p in PROPERTIES[:-1]:
        err[p] = BASE_ERR[p] * scale
        obs[p] = true[p] + rng.normal(0.0, 1.0, n) * err[p]
    obs["log_snr"] = log_snr
    err["log_snr"] = np.zeros(n)

    # --- artefacts: spurious extreme values in 2-3 properties --------------------
    spur_props = ["g_r", "r_z", "z_W1", "W1_W2", "log_ew", "log_fwhm", "alpha_nu"]
    mad = {p: 1.4826 * np.median(np.abs(obs[p][:n_norm] - np.median(obs[p][:n_norm])))
           for p in spur_props}
    med = {p: np.median(obs[p][:n_norm]) for p in spur_props}
    for i in range(n_norm + n_dusty + n_bal, n):
        k = rng.integers(2, 4)
        for p in rng.choice(spur_props, size=k, replace=False):
            sign = rng.choice([-1.0, 1.0])
            obs[p][i] = med[p] + sign * rng.uniform(3.0, 6.0) * mad[p]

    dq = np.zeros(n, dtype=int)
    dq[labels != ARTEFACT_LABEL] = rng.random((labels != ARTEFACT_LABEL).sum()) < flag_rate_normal
    dq[sl_art] = rng.random(n_art) < flag_rate_artefact

    df = pd.DataFrame({p: obs[p] for p in PROPERTIES})
    for p in PROPERTIES:
        df[f"{p}_err"] = err[p]
    df["dq_flag"] = dq
    df["sim_label"] = labels
    # shuffle so that labels are not ordered in the table
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "obj_id", np.arange(len(df)))
    df.attrs["simulated"] = True
    return df


def perturb_catalogue(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Another plausible realisation: observed values re-drawn within stated errors."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for p in PROPERTIES:
        out[p] = df[p].to_numpy() + rng.normal(0.0, 1.0, len(df)) * df[f"{p}_err"].to_numpy()
    return out


def truth_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(true rare population mask, artefact mask)."""
    lab = df["sim_label"].to_numpy()
    return np.isin(lab, TRUE_LABELS), lab == ARTEFACT_LABEL
