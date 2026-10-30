"""Independent Q-learners playing repeated 2-player games.

A cleaned-up, unified version of prisoners dilemma.py / hawk_dove.py /
corporation.py. Two tabular Q-learning agents repeatedly play a 2x2 (or 3x3)
matrix game and we look at what they converge to: a Nash equilibrium, a
collusive outcome, or a cycle.

Differences from the original scripts
-------------------------------------
* One engine for all games (PD, Hawk-Dove, Coordination, Stag Hunt, Bilingual).
* Actions are pure actions (C/D, H/D, ...). The old version used a grid of
  mixing probabilities, but payoffs are linear in your own probability, so the
  best response is always p=0 or p=1 and the grid only added noise.
* Discount factor `gamma` is a real parameter (the old scripts hard-coded 0, so
  the repeated-game / punishment effects could never appear).
* Random tie-breaking and per-action initial Q-values (expected payoff against a
  uniformly random opponent), instead of zeros + argmax's first-index bias
  towards action 0.
* A single long repeated game per run with exponentially declining exploration
  (as in Calvano et al., 2020) and a convergence criterion, instead of 5000
  restarted episodes from random states.
* Many seeds per parameter setting, with outcome frequencies reported.
* The long-run outcome is checked against the stage-game Nash equilibria, and a
  one-shot deviation test shows whether the opponent punishes.
* No global state; everything is reproducible from `--seed`.

Examples
--------
    python qlearning_games.py pd --G 1 --L 1 --gamma 0 --seeds 20
    python qlearning_games.py pd --G 1 --L 1 --gamma 0.9 --seeds 20 --deviation
    python qlearning_games.py hawk_dove --G 0.1 0.3 0.5 0.7 0.9 --gamma 0
    python qlearning_games.py pd --G 0.5 1 1.5 2 --L 0.5 1 1.5 2 --gamma 0.9 \
        --csv pd.csv --heatmap pd.png
"""
from __future__ import annotations

import argparse
import csv
import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Games
# --------------------------------------------------------------------------- #
@dataclass(eq=False)
class Game:
    """Two-player game. payoffs[a1, a2] = (reward to player 1, reward to player 2)."""

    name: str
    actions: Tuple[str, ...]
    payoffs: np.ndarray
    focal: str = ""  # outcome label whose frequency is reported / plotted

    @property
    def n(self) -> int:
        return len(self.actions)

    def label(self, state: int) -> str:
        a1, a2 = divmod(state, self.n)
        return self.actions[a1] + self.actions[a2]

    def pure_nash(self) -> List[int]:
        """Joint actions (as state indices) that are pure Nash equilibria."""
        eq = []
        for a1, a2 in itertools.product(range(self.n), repeat=2):
            u1, u2 = self.payoffs[a1, a2]
            best1 = u1 >= self.payoffs[:, a2, 0].max() - 1e-12
            best2 = u2 >= self.payoffs[a1, :, 1].max() - 1e-12
            if best1 and best2:
                eq.append(a1 * self.n + a2)
        return eq


def _game(name, actions, table, focal=""):
    return Game(name, tuple(actions), np.array(table, dtype=float), focal)


def prisoners_dilemma(G: float, L: float) -> Game:
    return _game("pd", "CD",
                 [[(1, 1), (-L, 1 + G)],
                  [(1 + G, -L), (0, 0)]], focal="CC")


def hawk_dove(G: float) -> Game:
    # Anti-coordination: pure NE are (H,D) and (D,H) for 0<G<1; mixed NE has P(H)=G.
    return _game("hawk_dove", "HD",
                 [[(0, 0), (1 + G, 1 - G)],
                  [(1 - G, 1 + G), (1, 1)]], focal="DD")


def coordination(G: float) -> Game:
    """Pure coordination game exactly as drawn on the poster (miscoordination pays 0,0)."""
    return _game("coordination", "AB",
                 [[(1 + G, 1 + G), (0, 0)],
                  [(0, 0), (1, 1)]], focal="AA")


def stag_hunt(G: float) -> Game:
    """What corporation.py actually implemented: 'D' pays 1 no matter what, so
    D is the safe action and C the risky, payoff-dominant one. C is risk-dominant
    only when G > 1 (the threshold seen in the results)."""
    return _game("stag_hunt", "CD",
                 [[(1 + G, 1 + G), (0, 1)],
                  [(1, 0), (1, 1)]], focal="CC")


def bilingual(G: float, c: float) -> Game:
    """Bilingual coordination from the poster; 'AB' = play both, at cost c."""
    return _game("bilingual", ["A", "B", "X"],  # X stands for the 'AB' option
                 [[(1 + G, 1 + G), (0, 1), (1 + G, 1 + G - c)],
                  [(1, 0), (1, 1), (1, 1 - c)],
                  [(1 + G - c, 1 + G), (1 - c, 1), (1 + G - c, 1 + G - c)]],
                 focal="AA")


GAMES = {
    "pd": (prisoners_dilemma, ("G", "L")),
    "hawk_dove": (hawk_dove, ("G",)),
    "coordination": (coordination, ("G",)),
    "stag_hunt": (stag_hunt, ("G",)),
    "bilingual": (bilingual, ("G", "c")),
}


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class QAgent:
    """Tabular Q-learner. State = last joint action (memory-1)."""

    def __init__(self, q0: np.ndarray, n_states: int, alpha: float, gamma: float,
                 rng: np.random.Generator):
        # q0: per-action initial value, shape (n_actions,), tiled over states.
        self.Q = np.tile(q0, (n_states, 1)).astype(float)
        self.alpha, self.gamma, self.rng = alpha, gamma, rng

    def act(self, s: int, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.Q.shape[1]))
        row = self.Q[s]
        best = np.flatnonzero(row >= row.max() - 1e-12)  # random tie-breaking
        return int(self.rng.choice(best))

    def update(self, s: int, a: int, r: float, s2: int) -> None:
        target = r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def policy(self) -> np.ndarray:
        return self.Q.argmax(axis=1)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    policies: Tuple[np.ndarray, np.ndarray]
    cycle: Tuple[int, ...]          # long-run cycle of states under greedy play
    avg_payoff: Tuple[float, float]
    is_nash: bool                   # cycle is a single stage-game Nash equilibrium
    converged: bool
    steps: int
    final_state: int
    extra: Dict = field(default_factory=dict)


def follow_greedy(policies, n: int, start: int) -> Tuple[List[int], Tuple[int, ...]]:
    """Deterministic greedy play from `start`; returns (path, cycle) where cycle
    is rotated so that it starts at its smallest state (canonical form)."""
    seen: Dict[int, int] = {}
    path: List[int] = []
    s = start
    while s not in seen:
        seen[s] = len(path)
        path.append(s)
        s = int(policies[0][s]) * n + int(policies[1][s])
    cyc = path[seen[s]:]
    k = cyc.index(min(cyc))
    return path, tuple(cyc[k:] + cyc[:k])


def simulate(game: Game, alpha: float = 0.1, gamma: float = 0.0, steps: int = 30000,
             eps_decay: float = 3e-4, stable_window: int = 2000, conv_eps: float = 0.05,
             seed: int = 0) -> Run:
    """One run of two independent Q-learners.

    Exploration: eps_t = exp(-eps_decay * t). Training stops early once eps is
    below `conv_eps` and both greedy policies have not changed for
    `stable_window` periods.
    """
    rng = np.random.default_rng(seed)
    n, P = game.n, game.payoffs
    nS = n * n
    # Initial Q: expected own payoff vs. a uniformly random opponent, discounted
    q0 = lambda v: v / (1.0 - gamma) if gamma < 1 else v
    agents = (QAgent(q0(P[:, :, 0].mean(axis=1)), nS, alpha, gamma, rng),
              QAgent(q0(P[:, :, 1].mean(axis=0)), nS, alpha, gamma, rng))

    s = int(rng.integers(nS))
    prev = (agents[0].policy(), agents[1].policy())
    stable, converged, t = 0, False, 0
    check = 100
    for t in range(1, steps + 1):
        eps = np.exp(-eps_decay * t)
        a1, a2 = agents[0].act(s, eps), agents[1].act(s, eps)
        r1, r2 = P[a1, a2]
        s2 = a1 * n + a2
        agents[0].update(s, a1, r1, s2)
        agents[1].update(s, a2, r2, s2)
        s = s2
        if t % check == 0 and eps < conv_eps:
            cur = (agents[0].policy(), agents[1].policy())
            if all(np.array_equal(c, p) for c, p in zip(cur, prev)):
                stable += check
            else:
                stable = 0
            prev = cur
            if stable >= stable_window:
                converged = True
                break

    pol = (agents[0].policy(), agents[1].policy())
    _, cyc = follow_greedy(pol, n, s)
    avg = tuple(float(np.mean([P[divmod(c, n)][i] for c in cyc])) for i in (0, 1))
    nash = len(cyc) == 1 and cyc[0] in set(game.pure_nash())
    return Run(pol, cyc, avg, nash, converged, t, s, extra={"agents": agents})


def cycle_label(game: Game, cycle: Tuple[int, ...]) -> str:
    return ">".join(game.label(c) for c in cycle)


def deviation_test(game: Game, run: Run, horizon: int = 8) -> Dict:
    """Force player 1 to take its best one-shot deviation from the long-run
    outcome, then let both follow their greedy policies. If player 2 punishes,
    the path leaves the cycle for a while before (maybe) returning."""
    n, P = game.n, game.payoffs
    s0 = run.cycle[0]
    a1, a2 = divmod(s0, n)
    dev = max((a for a in range(n) if a != a1), key=lambda a: P[a, a2, 0])
    s = dev * n + a2
    path = [s]
    for _ in range(horizon):
        s = int(run.policies[0][s]) * n + int(run.policies[1][s])
        path.append(s)
    return {
        "start": game.label(s0),
        "deviation": game.actions[dev] + game.actions[a2],
        "path": " ".join(game.label(x) for x in path),
        "deviation_profit": float(P[dev, a2, 0] - P[a1, a2, 0]),
        "punished": any(x not in run.cycle for x in path[1:]),
        "returns_to_cycle": path[-1] in run.cycle,
    }


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def run_many(game: Game, seeds: int, base_seed: int, **kw) -> List[Run]:
    return [simulate(game, seed=base_seed + i, **kw) for i in range(seeds)]


def summarise(game: Game, runs: List[Run]) -> Dict:
    counts = Counter(cycle_label(game, r.cycle) for r in runs)
    n = len(runs)
    return {
        "outcomes": {k: v / n for k, v in counts.most_common()},
        "focal_freq": counts.get(game.focal, 0) / n,
        "nash_freq": sum(r.is_nash for r in runs) / n,
        "converged_freq": sum(r.converged for r in runs) / n,
        "mean_payoff_1": float(np.mean([r.avg_payoff[0] for r in runs])),
        "mean_payoff_2": float(np.mean([r.avg_payoff[1] for r in runs])),
    }


def print_policy(game: Game, run: Run) -> None:
    print("  greedy policy (state = last joint action -> action played):")
    for s in range(game.n ** 2):
        print(f"    after {game.label(s)}: P1 -> {game.actions[run.policies[0][s]]}, "
              f"P2 -> {game.actions[run.policies[1][s]]}")


def heatmap(rows: List[Dict], xkey: str, ykey: str, path: str, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping heatmap (pip install matplotlib)")
        return
    xs = sorted({r[xkey] for r in rows})
    ys = sorted({r[ykey] for r in rows})
    grid = np.zeros((len(ys), len(xs)))
    for r in rows:
        grid[ys.index(r[ykey]), xs.index(r[xkey])] = r["focal_freq"]
    fig, ax = plt.subplots()
    im = ax.imshow(grid, origin="lower", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set_yticks(range(len(ys)), [str(y) for y in ys])
    ax.set_xlabel(xkey)
    ax.set_ylabel(ykey)
    ax.set_title(title)
    fig.colorbar(im, label="frequency of focal outcome")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", choices=GAMES)
    ap.add_argument("--G", type=float, nargs="+", default=[1.0])
    ap.add_argument("--L", type=float, nargs="+", default=[1.0],
                    help="sucker's payoff (pd); cost c for bilingual")
    ap.add_argument("--gamma", type=float, default=0.0, help="discount factor")
    ap.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eps-decay", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--show-policy", action="store_true", help="print greedy policy of seed 0")
    ap.add_argument("--deviation", action="store_true", help="run one-shot deviation test (seed 0)")
    ap.add_argument("--csv", help="write results to CSV")
    ap.add_argument("--heatmap", help="save a G x L heatmap of focal-outcome frequency (png)")
    args = ap.parse_args(argv)

    builder, params = GAMES[args.game]
    second = params[1] if len(params) == 2 else None
    grid = list(itertools.product(args.G, args.L if second else [None]))

    rows = []
    for G, L in grid:
        game = builder(G, L) if second else builder(G)
        runs = run_many(game, args.seeds, args.seed, alpha=args.alpha, gamma=args.gamma,
                        steps=args.steps, eps_decay=args.eps_decay)
        s = summarise(game, runs)
        tag = f"G={G}" + (f", {second}={L}" if second else "")
        print(f"\n{game.name}  {tag}  gamma={args.gamma}  "
              f"pure NE={[game.label(x) for x in game.pure_nash()]}")
        print("  outcomes: " + ", ".join(f"{k} {v:.0%}" for k, v in s["outcomes"].items()))
        print(f"  nash={s['nash_freq']:.0%}  converged={s['converged_freq']:.0%}  "
              f"mean payoff=({s['mean_payoff_1']:.2f}, {s['mean_payoff_2']:.2f})")
        if args.show_policy:
            print_policy(game, runs[0])
        if args.deviation:
            d = deviation_test(game, runs[0])
            print(f"  deviation test (seed {args.seed}): {d['start']} -> force {d['deviation']} "
                  f"(+{d['deviation_profit']:.2f} for deviator)\n"
                  f"    path: {d['path']}\n"
                  f"    punished={d['punished']}  returns_to_cycle={d['returns_to_cycle']}")
        rows.append({"game": game.name, "G": G, second or "L": L, "gamma": args.gamma,
                     "focal": game.focal, **{k: v for k, v in s.items() if k != "outcomes"},
                     "outcomes": "; ".join(f"{k}:{v:.2f}" for k, v in s["outcomes"].items())})

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved {args.csv}")
    if args.heatmap and second:
        heatmap(rows, "G", second, args.heatmap,
                f"{args.game}: frequency of {rows[0]['focal']} (gamma={args.gamma})")


if __name__ == "__main__":
    main()
