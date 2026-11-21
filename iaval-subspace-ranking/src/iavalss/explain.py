"""Stage 4: plain-language explanations and export (CSV for TOPCAT, VOTable via astropy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io.votable.tree import Info
from astropy.table import Table

from .simulate import PROPERTIES
from .subspace import SCORE_NAMES, normalise_weights, robust_standardise
from .pipeline import PipelineResult

SCORE_TEXT = {
    "density_contrast": "sits in a compact over-dense clump relative to its surroundings",
    "distribution_deviation": "has a combination of values unexpected from the two properties' marginal distributions",
    "local_outlyingness": "is isolated relative to its local neighbours (local-outlier-factor-like)",
}


def explain_object(res: PipelineResult, i: int, n_proj: int = 3, z_med_sig=None) -> dict:
    """Structured and plain-language explanation for object index ``i``."""
    sc = res.scorer
    props = sc.prop_names
    w = normalise_weights(res.weights)
    order = np.argsort(-res.I_proj[i])[:n_proj]
    Z, _, _ = robust_standardise(res.X)
    proj_txt, driver_weight = [], {}
    for p in order:
        comp = sc.S_ref_[i, p] * w                     # weighted contribution of each score
        top = int(np.argmax(comp))
        a, b = sc.pairs[p]
        proj_txt.append(f"{sc.pair_name(p)} (score {res.I_proj[i, p]:.2f}; mainly "
                        f"{SCORE_NAMES[top].replace('_', ' ')}: it {SCORE_TEXT[SCORE_NAMES[top]]})")
        for j in (a, b):
            driver_weight[j] = driver_weight.get(j, 0.0) + float(res.I_proj[i, p])
    drivers = sorted(driver_weight, key=lambda j: (-driver_weight[j], -abs(Z[i, j])))[:3]
    drv_txt = []
    for j in drivers:
        err = res.sigma[i, j]
        drv_txt.append(f"{props[j]}={res.X[i, j]:.3g} ({Z[i, j]:+.1f} robust-sigma from the "
                       f"catalogue median" + (f"; measured to +/-{err:.2g})" if err > 0 else ")"))
    flags = []
    if res.p_art[i] > 0.5:
        flags.append("likely instrumental/low-quality artefact")
    if res.df["dq_flag"].iloc[i] > 0:
        flags.append("data-quality flag set")
    text = (f"Ranked on {len(order)} projections: " + "; ".join(proj_txt)
            + ". Properties driving it: " + ", ".join(drv_txt)
            + f". P(anomaly | measurement errors)={res.P_anom[i]:.2f}; "
              f"artefact probability={res.p_art[i]:.2f}"
            + (" [" + "; ".join(flags) + "]" if flags else "") + ".")
    return {"top_projections": [sc.pair_name(p) for p in order],
            "driver_properties": [props[j] for j in drivers], "explanation": text}


def build_candidate_table(res: PipelineResult, n: int = 200, include_truth: bool = True) -> pd.DataFrame:
    """Top-n objects by aware score with scores, explanations and the original columns."""
    top = np.argsort(-res.aware, kind="stable")[:n]
    rows = []
    for rank, i in enumerate(top, start=1):
        e = explain_object(res, int(i))
        rows.append({
            "rank": rank, "obj_id": int(res.df["obj_id"].iloc[i]),
            "aware_score": float(res.aware[i]), "I_score": float(res.I[i]),
            "P_anomaly": float(res.P_anom[i]), "p_artefact": float(res.p_art[i]),
            "top_projections": " | ".join(e["top_projections"]),
            "driver_properties": ",".join(e["driver_properties"]),
            "explanation": e["explanation"],
        })
    out = pd.DataFrame(rows)
    cols = res.df.iloc[top].reset_index(drop=True)
    keep = ["obj_id"] + PROPERTIES + [f"{p}_err" for p in PROPERTIES] + ["dq_flag"]
    if include_truth:
        keep.append("sim_label")
    out = out.merge(cols[keep], on="obj_id", how="left")
    if include_truth:
        out = out.rename(columns={"sim_label": "sim_truth_label_not_used_by_method"})
    return out


def export_csv(table: pd.DataFrame, path: str) -> None:
    """Plain CSV: header row, comma separated, no index (TOPCAT reads it directly)."""
    table.to_csv(path, index=False)


def export_votable(table: pd.DataFrame, path: str, provenance: dict | None = None) -> None:
    """VOTable (astropy) with column descriptions and provenance INFO elements."""
    t = Table.from_pandas(table)
    descr = {"rank": "Rank by aware score (1 = most interesting)",
             "aware_score": "(1 - p_artefact) * P_anomaly + 1e-3 * I_score",
             "I_score": "Weighted subspace interest score from measured values, in [0,1]",
             "P_anomaly": "Monte Carlo fraction of resamples above anomaly threshold",
             "p_artefact": "Probability anomaly is explained by low S/N / quality flag",
             "explanation": "Plain-language reason for flagging"}
    for c, d in descr.items():
        if c in t.colnames:
            t[c].description = d
    from astropy.io.votable import from_table
    vot = from_table(t)
    res0 = vot.resources[0]
    prov = {"DATA_ORIGIN": "SIMULATED catalogue - not DESI or any real survey data",
            "SOFTWARE": "iavalss prototype"}
    prov.update(provenance or {})
    for k, v in prov.items():
        res0.infos.append(Info(name=k, value=str(v)))
    vot.to_xml(path)
