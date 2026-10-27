# Improved version: Q-learning in repeated games

An interactive explainer plus a cleaned-up Python implementation of the SURGE '24 project.

| File | What it is |
|---|---|
| `index.html` | **Open this in a browser.** Self-explaining page: choose a game, change parameters, watch two Q-learners play. No server, no internet, no install needed. |
| `qlearning.js` | The simulation engine (JavaScript port of the Python file) used by the page. |
| `app.js` | Charts, controls and explanations for the page. |
| `qlearning_games.py` | Command-line version: many seeds, outcome frequencies, Nash check, deviation test, CSV/heatmap export. Needs `numpy`. |

## The page
1. Pick a game (Prisoner's Dilemma, Hawk-Dove, Stag Hunt, Pure Coordination, Bilingual Coordination) and set G, L / c, the discount factor γ.
2. See the payoff matrix with its Nash equilibria and what to expect.
3. Watch one run: how often each player picks the first action over time, and the strategy (policy table) they end up with.
4. Many runs: how often each final outcome occurs.
5. Deviation test: force one player to cheat once and see whether the other punishes (the signature of collusion).
6. Parameter sweep: where does the behaviour flip?
Preset buttons ("Price war", "Patient learners", ...) jump to the interesting cases. Open `index.html#sweep` to load the page with the sweep already run.

## Python
```
python qlearning_games.py pd --G 1 --L 1 --gamma 0.9 --seeds 20 --deviation
python qlearning_games.py stag_hunt --G 0.5 1.5 --gamma 0
python qlearning_games.py pd --G 0.5 1 1.5 2 --L 0.5 1 1.5 2 --gamma 0.9 --csv pd.csv --heatmap pd.png
```
The JS and Python engines implement the same algorithm but use different random number generators, so exact percentages differ slightly.
