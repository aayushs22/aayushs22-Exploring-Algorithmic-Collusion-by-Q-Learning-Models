"""Quantify collusion between two Q-learners in the Prisoner's Dilemma.

Builds on qlearning_games.py. For many independent runs it measures

  * the collusion index  Δ = (π̄ − π_Nash) / (π_coop − π_Nash)
        0 = competitive (Nash) payoff, 1 = full cooperation payoff,
  * whether the long-run outcome is a one-shot Nash equilibrium,
  * whether a forced one-shot deviation is PUNISHED (the signature of collusion),

and classifies every run:

  competitive   long-run outcome is the one-shot Nash equilibrium (D, D)
  sustained     Δ >= threshold, converged, not Nash, and a deviation is punished
  unpunished    Δ >= threshold, converged, not Nash, but a deviation is NOT punished
  other         everything else (cycles with low Δ, runs that never converged, ...)

"Sustained" only requires that the deviation is punished. Whether play then returns to the
settled outcome (a forgiving strategy) or not (grim-trigger style) is reported separately.

The deviation test is run from the first state of the settled cycle.
All rates come with 95% Wilson confidence intervals.

Examples
--------
    python collusion_report.py --G 1 --L 1 --gamma 0.9 --runs 100
    python collusion_report.py --G 1 --L 1 --gamma 0.9 --runs 100 --curve 0 0.5 0.8 0.9 0.95 \
        --csv curve.csv --plot curve.png
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Sequence, Tuple

import numpy as np

from qlearning_games import Game, deviation_test, prisoners_dilemma, simulate

CLASSES = ("sustained", "unpunished", "competitive", "other")


def pd_benchmarks(game: Game) -> Tuple[float, float]:
    """(π_Nash, π_coop): per-player payoff at (D,D) and at (C,C). Only defined for the PD."""
    if game.name != "pd":
        raise ValueError("collusion index is only defined for the Prisoner's Dilemma")
    return float(game.payoffs[1, 1].mean()), float(game.payoffs[0, 0].mean())


def collusion_index(game: Game, run) -> float:
    nash, coop = pd_benchmarks(game)
    return (float(np.mean(run.avg_payoff)) - nash) / (coop - nash)


def classify(game: Game, run, threshold: float = 0.5, horizon: int = 8) -> Dict:
    delta = collusion_index(game, run)
    out = {"delta": delta, "cls": "other", "returns": False}
    if run.is_nash:
        out["cls"] = "competitive"
    elif run.converged and delta >= threshold:
        dev = deviation_test(game, run, horizon)
        out["cls"] = "sustained" if dev["punished"] else "unpunished"
        out["returns"] = bool(dev["returns_to_cycle"])
    return out


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _one(args):
    game, kw, seed, threshold, horizon = args
    return classify(game, simulate(game, seed=seed, **kw), threshold, horizon)


def analyse(game: Game, runs: int = 100, base_seed: int = 0, threshold: float = 0.5,
            horizon: int = 8, jobs: int = 1, **kw) -> Dict:
    tasks = [(game, kw, base_seed + i, threshold, horizon) for i in range(runs)]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            res = list(ex.map(_one, tasks, chunksize=max(1, runs // (4 * jobs))))
    else:
        res = [_one(t) for t in tasks]
    counts = {c: sum(r["cls"] == c for r in res) for c in CLASSES}
    sustained = [r for r in res if r["cls"] == "sustained"]
    return {
        "runs": runs,
        "counts": counts,
        "rates": {c: counts[c] / runs for c in CLASSES},
        "ci": {c: wilson(counts[c], runs) for c in CLASSES},
        "mean_delta": float(np.mean([r["delta"] for r in res])),
        "forgiving_share": (sum(r["returns"] for r in sustained) / len(sustained)) if sustained else None,
    }


def fmt_rate(a: Dict, c: str, pad: bool = True) -> str:
    lo, hi = a["ci"][c]
    s = f"{a['rates'][c]:.0%} [{lo:.0%}-{hi:.0%}]"
    return s.rjust(15) if pad else s


def print_table(rows: List[Tuple[float, Dict]]) -> None:
    print(f"{'gamma':>6} {'sustained collusion':>22} {'unpunished high':>18} {'competitive':>18} {'other':>8} {'mean Δ':>8}")
    for g, a in rows:
        print(f"{g:>6} {fmt_rate(a, 'sustained'):>22} {fmt_rate(a, 'unpunished'):>18} "
              f"{fmt_rate(a, 'competitive'):>18} {a['rates']['other']:>8.0%} {a['mean_delta']:>8.2f}")


def plot(rows: List[Tuple[float, Dict]], path: str, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (pip install matplotlib)")
        return
    colors = {"sustained": "#2a78d6", "unpunished": "#1baf7a", "competitive": "#eb6834", "other": "#b4b2aa"}
    gammas = [g for g, _ in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    left = np.zeros(len(rows))
    for c in CLASSES:
        v = np.array([a["rates"][c] for _, a in rows])
        ax1.barh(range(len(rows)), v, left=left, color=colors[c], label=c.replace("_", " "), edgecolor="white", height=0.7)
        left += v
    ax1.set_yticks(range(len(rows)), [f"γ = {g}" for g in gammas])
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("share of runs")
    ax1.set_title("What the learners end up doing")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=8)
    rate = np.array([a["rates"]["sustained"] for _, a in rows])
    lo = np.array([a["ci"]["sustained"][0] for _, a in rows])
    hi = np.array([a["ci"]["sustained"][1] for _, a in rows])
    ax2.fill_between(gammas, lo, hi, color=colors["sustained"], alpha=0.18, linewidth=0)
    ax2.plot(gammas, rate, color=colors["sustained"], marker="o", label="sustained-collusion rate (95% CI)")
    ax2.plot(gammas, [np.clip(a["mean_delta"], 0, 1) for _, a in rows], color=colors["unpunished"], marker="o", label="mean Δ")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("discount factor γ")
    ax2.set_title("Collusion appears as the future starts to matter")
    ax2.legend(frameon=False, fontsize=8)
    for ax in (ax1, ax2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.9, help="patient setting to compare with gamma = 0")
    ap.add_argument("--curve", type=float, nargs="+", help="gamma values for a curve (overrides --gamma comparison)")
    ap.add_argument("--runs", type=int, default=100, help="runs per gamma (100+ recommended)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Δ above which an outcome counts as 'high'")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eps-decay", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel processes")
    ap.add_argument("--csv")
    ap.add_argument("--plot")
    a = ap.parse_args()

    game = prisoners_dilemma(a.G, a.L)
    if a.G - a.L >= 1:
        print(f"warning: G - L = {a.G - a.L:g} >= 1, so taking turns exploiting each other beats mutual "
              "cooperation; (C,C) is no longer the joint-payoff maximiser and Δ is hard to interpret.")
    gammas = a.curve if a.curve else sorted({0.0, a.gamma})
    rows = []
    for g in gammas:
        rows.append((g, analyse(game, a.runs, a.seed, a.threshold, jobs=a.jobs, alpha=a.alpha,
                                gamma=g, steps=a.steps, eps_decay=a.eps_decay)))
    print(f"\nPrisoner's Dilemma G={a.G}, L={a.L}; {a.runs} runs per gamma; 'high' means Δ >= {a.threshold}\n")
    print_table(rows)

    base = next((r for g, r in rows if g == 0), None)
    top = rows[-1][1] if rows[-1][0] != 0 else None
    if base is not None and top is not None:
        g = rows[-1][0]
        print(f"\nHeadline: with gamma = {g}, {fmt_rate(top, 'sustained', False)} of {a.runs} runs sustain collusion, "
              f"versus {fmt_rate(base, 'sustained', False)} at gamma = 0.")
        if top["forgiving_share"] is not None:
            fs = top["forgiving_share"]
            print(f"Of the collusive runs, {fs:.0%} return to the settled outcome after punishing (forgiving) "
                  f"and {1 - fs:.0%} do not (grim-trigger style).")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gamma", "runs", *[f"{c}_rate" for c in CLASSES], "sustained_ci_lo", "sustained_ci_hi", "mean_delta"])
            for g, r in rows:
                w.writerow([g, r["runs"], *[round(r["rates"][c], 4) for c in CLASSES],
                            round(r["ci"]["sustained"][0], 4), round(r["ci"]["sustained"][1], 4), round(r["mean_delta"], 4)])
        print(f"saved {a.csv}")
    if a.plot:
        plot(rows, a.plot, f"Prisoner's Dilemma, G={a.G}, L={a.L}, {a.runs} runs per γ")


if __name__ == "__main__":
    main()
