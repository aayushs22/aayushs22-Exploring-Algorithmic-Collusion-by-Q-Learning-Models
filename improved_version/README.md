# Improved version: Q-learning in repeated games

An interactive explainer plus a cleaned-up Python implementation of the SURGE '24 project.

| File | What it is |
|---|---|
| `index.html` | **Open this in a browser.** Self-explaining page: choose a game, change parameters, watch two Q-learners play. No server, no internet, no install needed. |
| `qlearning.js` | The simulation engine (JavaScript port of the Python file) used by the page. |
| `app.js` | Charts, controls and explanations for the page. |
| `qlearning_games.py` | Command-line version: many seeds, outcome frequencies, Nash check, deviation test, CSV/heatmap export. Needs `numpy`. |
| `collusion_report.py` | **Quantifies collusion** in the Prisoner's Dilemma: collusion index Δ, run classification, 95% confidence intervals, γ = 0 vs γ > 0 comparison and γ curve. Needs `numpy`. |

## The page
1. Pick a game (Prisoner's Dilemma, Hawk-Dove, Stag Hunt, Pure Coordination, Bilingual Coordination) and set G, L / c, the discount factor γ.
2. See the payoff matrix with its Nash equilibria and what to expect.
3. Watch one run: how often each player picks the first action over time, and the strategy (policy table) they end up with.
4. Many runs: how often each final outcome occurs.
5. Deviation test: force one player to cheat once and see whether the other punishes (the signature of collusion).
6. **Measuring collusion** (Prisoner's Dilemma): classify every run and report rates with confidence intervals, for γ = 0 up to patient learners.
7. Parameter sweep: where does the behaviour flip?
Preset buttons ("Price war", "Patient learners", ...) jump to the interesting cases. Open `index.html#sweep` or `index.html#collusion` to load the page with that analysis already run.

## Python
```
python qlearning_games.py pd --G 1 --L 1 --gamma 0.9 --seeds 20 --deviation
python qlearning_games.py stag_hunt --G 0.5 1.5 --gamma 0
python qlearning_games.py pd --G 0.5 1 1.5 2 --L 0.5 1 1.5 2 --gamma 0.9 --csv pd.csv --heatmap pd.png
python collusion_report.py --G 1 --L 1 --gamma 0.9 --runs 100 --curve 0 0.5 0.8 0.9 0.95 --csv curve.csv --plot curve.png
```

## How collusion is measured
"Different from Nash" is necessary but not sufficient (in the Stag Hunt, (C,C) is itself an equilibrium). For the Prisoner's Dilemma each run gets:

* **Collusion index** `Δ = (π̄ − π_Nash) / (π_coop − π_Nash)`: 0 = competitive, 1 = full cooperation.
* **Class**: *competitive* (ends at the one-shot Nash outcome), *sustained collusion* (converged, Δ ≥ threshold, not Nash, and a forced one-time deviation is **punished**), *unpunished high outcome* (same but no punishment), *other*.
* Rates over many runs come with 95% Wilson confidence intervals, compared across discount factors (γ = 0 is the no-collusion baseline).

Example (100 runs per γ, G = L = 1): 0% sustained collusion for γ ≤ 0.8, about 25% at γ = 0.9, about 60% at γ = 0.95 (Python and JS agree within a few points). These depend on the settings and learning parameters; re-run before quoting.

The JS and Python engines implement the same algorithm but use different random number generators, so exact percentages differ slightly.
