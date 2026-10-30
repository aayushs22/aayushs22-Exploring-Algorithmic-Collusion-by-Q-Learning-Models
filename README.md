## SURGE Intern | Prof. Srinivas Arigapudi, IIT Kanpur

**May'24 - July'24**

**Objective**

* To determine whether algorithmic pricing models can engage in collusion and the factors influencing its extent.

**Approach**

* Implemented Q-learning, a reinforcement learning algorithm, using Python and numerical computation libraries.
* Developed core logic to simulate player behavior and learning in classic game scenarios like Prisoner's Dilemma and Hawk-Dove.
* Defined key parameters and used an epsilon-greedy approach for strategy exploration.
* Conducted simulations and analyzed varying cooperation probabilities to understand strategic dynamics.

**Impact**

* Determined the effectiveness of Q-learning in modelling player behaviour and predicting optimal strategies in complex games.
* Revealed how different reward structures and cooperation probabilities influence the emergence or breakdown of cooperation.
* Offered a deeper understanding of strategic decision-making processes in classic game theory scenarios.

---

## Updated implementation: `improved_version/` (open `improved_version/index.html` for the interactive explainer)

A cleaned-up, unified rewrite of `prisoners dilemma.py`, `hawk_dove.py` and `corporation.py` (the originals are kept for reference).

**What changed**
* One engine for Prisoner's Dilemma, Hawk-Dove, Coordination, Stag Hunt (what `corporation.py` actually implemented) and Bilingual Coordination.
* Discount factor `--gamma` is a parameter (the originals fixed it at 0, so punishment/repeated-game effects could not appear).
* Pure actions instead of a probability grid (payoffs are linear in your own mixing probability, so best responses are always pure).
* Random tie-breaking and sensible initial Q-values (the old `argmax` + zeros biased toward action 0).
* One long repeated game per run, with exponentially decaying exploration and a convergence criterion.
* Many seeds per setting, with outcome frequencies, a check against the stage-game Nash equilibria, and a one-shot deviation/punishment test.
* No globals; reproducible via `--seed`. Optional CSV and heatmap output.
* `collusion_report.py` quantifies collusion in the Prisoner's Dilemma: collusion index Δ, run classification (competitive / sustained collusion / unpunished / other) and 95% confidence intervals across discount factors.

**Usage** (needs `numpy`; `matplotlib` only for `--heatmap`)
```
python improved_version/qlearning_games.py pd --G 1 --L 1 --gamma 0 --seeds 20            # -> DD (dominant strategy)
python improved_version/qlearning_games.py pd --G 1 --L 1 --gamma 0.9 --seeds 20 --deviation
python improved_version/qlearning_games.py hawk_dove --G 0.1 0.5 0.9
python improved_version/qlearning_games.py stag_hunt --G 0.5 1.5                           # D if G<1, C if G>1
python improved_version/qlearning_games.py pd --G 0.5 1 1.5 2 --L 0.5 1 1.5 2 --gamma 0.9 --csv pd.csv --heatmap pd.png
cd improved_version && python collusion_report.py --gamma 0.9 --runs 100 --curve 0 0.5 0.8 0.9 0.95
```
