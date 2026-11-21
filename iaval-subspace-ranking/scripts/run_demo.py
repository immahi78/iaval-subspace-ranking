#!/usr/bin/env python
"""End-to-end demo on a SIMULATED catalogue. Writes figures, results.txt, candidates CSV/VOTable.

Usage: python scripts/run_demo.py [--n 50000] [--seed 42]
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iavalss import evaluate as ev
from iavalss.baselines import all_baselines
from iavalss.explain import build_candidate_table, export_csv, export_votable
from iavalss.pipeline import run_pipeline
from iavalss.simulate import PROPERTIES, make_catalogue, truth_masks

COL = {"aware": "#0072B2", "isolation_forest": "#E69F00", "mahalanobis": "#009E73",
       "random": "#7f7f7f", "plain I": "#CC79A7"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120})
SIM = "SIMULATED data - not DESI"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t_start = time.perf_counter()
    figdir, outdir = ROOT / "docs" / "figures", ROOT / "outputs"
    figdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    df = make_catalogue(n=args.n, seed=args.seed)
    true, art = truth_masks(df)
    res = run_pipeline(df, seed=args.seed)
    base, base_rt = all_baselines(res.X, seed=args.seed)
    scores = {**base, "iavalss aware (full)": res.aware, "iavalss plain I": res.I}
    runtimes = {**base_rt, "iavalss aware (full)": res.timing["total_s"],
                "iavalss plain I": res.timing["subspace_scoring_s"]}
    metrics = ev.metrics_table(scores, df, runtimes=runtimes)
    abl = ev.metrics_table(ev.ablation_scores(res), df)
    wsens = ev.weight_sensitivity(res, df)
    stab = ev.rank_stability(df, scores, n_real=2, k=500, seed=args.seed + 100)

    # ---- candidates + exports --------------------------------------------------
    cand = build_candidate_table(res, n=200)
    export_csv(cand, str(outdir / "candidates_topcat.csv"))
    export_votable(cand, str(outdir / "candidates.vot"),
                   {"SEED": args.seed, "N_OBJECTS": args.n})
    top100 = np.argsort(-res.aware, kind="stable")[:100]
    proj_counter = Counter()
    for t in cand.head(100)["top_projections"]:
        proj_counter.update(t.split(" | "))

    # ---- figures ----------------------------------------------------------------
    ks = np.unique(np.concatenate([np.arange(20, 200, 20), np.arange(200, 2001, 100)]))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, s in scores.items():
        key = "aware" if "aware" in name else ("plain I" if "plain" in name else name)
        ax.plot(ks, ev.precision_curve(s, true, ks), label=name, color=COL[key],
                lw=2.2 if key == "aware" else 1.5, ls="--" if key == "plain I" else "-")
    ax.axhline(true.mean(), color="k", lw=0.6, ls=":")
    ax.set(xlabel="k (top-k ranked objects)", ylabel="precision@k (true rare populations)",
           title=f"Injection-recovery precision@k ({SIM})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "precision_at_k.png"); plt.close(fig)

    # ranked-projection plot
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
    common = proj_counter.most_common(12)[::-1]
    a1.barh([c[0] for c in common], [c[1] for c in common], color=COL["aware"])
    a1.set(xlabel="times among top-3 projections (top-100 aware candidates)",
           title="Which projections drive the candidates")
    a1.tick_params(axis="y", labelsize=8)
    best = common[-1][0] if common else None
    if best:
        px, py = best.split(" vs ")
        ax = a2
        ax.scatter(df[px], df[py], s=1, c="#cccccc", rasterized=True, label="all objects")
        lab = df["sim_label"].to_numpy()
        for nm, c in (("dusty_reddened", "#D55E00"), ("bal_like", "#0072B2"), ("artefact", "#000000")):
            sel = np.intersect1d(top100, np.where(lab == nm)[0])
            ax.scatter(df[px].to_numpy()[sel], df[py].to_numpy()[sel], s=14, c=c, label=f"top-100: {nm} ({len(sel)})")
        ax.set(xlabel=px, ylabel=py, title=f"Most-used projection ({SIM})")
        ax.legend(frameon=False, fontsize=7, markerscale=1.2)
    fig.tight_layout(); fig.savefig(figdir / "ranked_projections.png"); plt.close(fig)

    # rank stability
    fig, ax = plt.subplots(figsize=(6, 3.6))
    names = list(stab.index)
    ax.bar(names, stab["jaccard@500"], color=[COL.get("aware" if "aware" in n else ("plain I" if "plain" in n else n)) for n in names])
    ax.set(ylabel="top-500 Jaccard overlap", title="Rank stability under error resampling (simulated)")
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.tight_layout(); fig.savefig(figdir / "rank_stability.png"); plt.close(fig)

    # ablation
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(abl)); wd = 0.38
    ax.bar(x - wd / 2, abl["P@500"], wd, label="precision@500 (true rare)", color="#0072B2")
    ax.bar(x + wd / 2, abl["artefact@500"], wd, label="artefact fraction in top 500", color="#D55E00")
    ax.set_xticks(x); ax.set_xticklabels(abl.index, rotation=15, ha="right", fontsize=8)
    ax.set(title=f"Ablation ({SIM})"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "ablation.png"); plt.close(fig)

    total = time.perf_counter() - t_start

    # ---- results.txt ------------------------------------------------------------
    pd.set_option("display.width", 200)
    rnd = wsens[wsens.index.str.startswith("random_")]
    lines = [
        "IAVAL subspace-ranking prototype - RESULTS ON SIMULATED DATA (NOT DESI)",
        f"seed={args.seed}  N={args.n}  properties={len(PROPERTIES)}  projections={len(res.scorer.pairs)}",
        f"injected: dusty_reddened={int((df.sim_label=='dusty_reddened').sum())}, "
        f"bal_like={int((df.sim_label=='bal_like').sum())}, artefact={int(art.sum())}; "
        f"true rare = {int(true.sum())} ({true.mean():.4f} of catalogue = random-ordering precision)",
        f"anomaly threshold tau={res.tau:.4f} (top {res.anomaly_fraction:.0%} of I); n_mc={res.n_mc}; "
        f"MC shortlist size={len(res.shortlist)}",
        "", "== Precision/recall vs baselines (truth = dusty_reddened + bal_like) ==",
        metrics.round(4).to_string(),
        "", "== Ablation ==", abl.round(4).to_string(),
        "", "== Weight sensitivity (aware pipeline, shortlist fixed) ==",
        wsens.round(4).to_string(),
        f"random Dirichlet weights (n={len(rnd)}): P@500_aware mean={rnd['P@500_aware'].mean():.4f} "
        f"std={rnd['P@500_aware'].std():.4f} min={rnd['P@500_aware'].min():.4f} max={rnd['P@500_aware'].max():.4f}",
        "", "== Rank stability (top-500 Jaccard vs original; 2 re-drawn catalogues) ==",
        stab.round(4).to_string(),
        f"(random-ordering expectation ~ {500/(2*args.n-500):.4f})",
        "", "== Mean p_artefact and P_anomaly by simulated label ==",
        df.assign(p_artefact=res.p_art, P_anomaly=res.P_anom).groupby("sim_label")[["p_artefact", "P_anomaly"]].mean().round(4).to_string(),
        "", "== Composition of aware top-100 / top-500 ==",
        pd.DataFrame({"top100": df["sim_label"].iloc[top100].value_counts(),
                      "top500": df["sim_label"].iloc[np.argsort(-res.aware, kind='stable')[:500]].value_counts()}).fillna(0).astype(int).to_string(),
        "", "== Runtime (s, this machine) ==",
        ", ".join(f"{k}={v:.1f}" for k, v in res.timing.items()),
        f"whole demo wall-clock = {total:.1f}",
        "", "== Example explanation (rank 1) ==", cand.loc[0, "explanation"],
    ]
    (outdir / "results.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
