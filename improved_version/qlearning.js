/* Q-learning engine for 2-player repeated games.
 * JavaScript port of qlearning_games.py so the web page can run the experiments
 * live in the browser. Works in the browser (window.QL) and in Node (module.exports). */
(function (root) {
  "use strict";

  // ---- seedable RNG (mulberry32) ------------------------------------------------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---- game definitions ---------------------------------------------------------
  // table(p)[a1][a2] = [reward to player 1, reward to player 2]
  const GAMES = {
    pd: {
      name: "Prisoner's Dilemma",
      actions: ["C", "D"], names: ["Cooperate", "Defect"],
      params: [
        { key: "G", label: "Temptation G", min: 0.1, max: 3, step: 0.1, def: 1 },
        { key: "L", label: "Sucker's loss L", min: 0.1, max: 3, step: 0.1, def: 1 },
      ],
      focal: "CC", colors: { CC: 1, DD: 2, CD: 3, DC: 4 },
      table: (p) => [[[1, 1], [-p.L, 1 + p.G]], [[1 + p.G, -p.L], [0, 0]]],
    },
    hawk_dove: {
      name: "Hawk-Dove",
      actions: ["H", "D"], names: ["Hawk", "Dove"],
      params: [{ key: "G", label: "Prize G", min: 0.1, max: 0.9, step: 0.1, def: 0.5 }],
      focal: "DD", colors: { HD: 1, DH: 2, DD: 3, HH: 4 },
      table: (p) => [[[0, 0], [1 + p.G, 1 - p.G]], [[1 - p.G, 1 + p.G], [1, 1]]],
    },
    stag_hunt: {
      name: "Stag Hunt (coordination)",
      actions: ["C", "D"], names: ["Risky (C)", "Safe (D)"],
      params: [{ key: "G", label: "Bonus G", min: 0.1, max: 3, step: 0.1, def: 1.5 }],
      focal: "CC", colors: { CC: 1, DD: 2, CD: 3, DC: 4 },
      table: (p) => [[[1 + p.G, 1 + p.G], [0, 1]], [[1, 0], [1, 1]]],
    },
    coordination: {
      name: "Pure Coordination",
      actions: ["A", "B"], names: ["Option A", "Option B"],
      params: [{ key: "G", label: "Bonus G", min: 0.1, max: 3, step: 0.1, def: 0.5 }],
      focal: "AA", colors: { AA: 1, BB: 2, AB: 3, BA: 4 },
      table: (p) => [[[1 + p.G, 1 + p.G], [0, 0]], [[0, 0], [1, 1]]],
    },
    bilingual: {
      name: "Bilingual Coordination",
      actions: ["A", "B", "AB"], names: ["A", "B", "Both (AB)"],
      params: [
        { key: "G", label: "Bonus G", min: 0.1, max: 1, step: 0.1, def: 0.5 },
        { key: "c", label: "Cost c of AB", min: 0.1, max: 2, step: 0.1, def: 0.5 },
      ],
      focal: "A/A", colors: { "A/A": 1, "B/B": 2, "AB/AB": 3 },
      table: (p) => [
        [[1 + p.G, 1 + p.G], [0, 1], [1 + p.G, 1 + p.G - p.c]],
        [[1, 0], [1, 1], [1, 1 - p.c]],
        [[1 + p.G - p.c, 1 + p.G], [1 - p.c, 1], [1 + p.G - p.c, 1 + p.G - p.c]],
      ],
    },
  };

  function makeGame(key, p) {
    const d = GAMES[key];
    const n = d.actions.length;
    const wide = d.actions.some((a) => a.length > 1);
    const g = {
      key, name: d.name, actions: d.actions, names: d.names, n, P: d.table(p),
      focal: d.focal, colors: d.colors, params: p,
      label(s) {
        const a1 = Math.floor(s / n), a2 = s % n;
        return wide || n > 2 ? d.actions[a1] + "/" + d.actions[a2] : d.actions[a1] + d.actions[a2];
      },
      pureNash() {
        const out = [];
        for (let a1 = 0; a1 < n; a1++) for (let a2 = 0; a2 < n; a2++) {
          let b1 = true, b2 = true;
          for (let k = 0; k < n; k++) {
            if (this.P[k][a2][0] > this.P[a1][a2][0] + 1e-12) b1 = false;
            if (this.P[a1][k][1] > this.P[a1][a2][1] + 1e-12) b2 = false;
          }
          if (b1 && b2) out.push(a1 * n + a2);
        }
        return out;
      },
    };
    return g;
  }

  // ---- one run ------------------------------------------------------------------
  function argmax(row, off, n) {
    let b = 0;
    for (let i = 1; i < n; i++) if (row[off + i] > row[off + b]) b = i;
    return b;
  }

  function followGreedy(pol, n, start) {
    const seen = new Map(), path = [];
    let s = start;
    while (!seen.has(s)) {
      seen.set(s, path.length);
      path.push(s);
      s = pol[0][s] * n + pol[1][s];
    }
    let cyc = path.slice(seen.get(s));
    const k = cyc.indexOf(Math.min(...cyc));
    cyc = cyc.slice(k).concat(cyc.slice(0, k));
    return { path, cycle: cyc };
  }

  const DEFAULTS = { alpha: 0.1, gamma: 0, steps: 30000, epsDecay: 3e-4, stableWindow: 2000, convEps: 0.05 };

  function simulate(game, opts, seed, trace) {
    const o = Object.assign({}, DEFAULTS, opts);
    const rng = mulberry32(seed * 7919 + 13);
    const n = game.n, nS = n * n, P = game.P;
    const disc = o.gamma < 1 ? 1 / (1 - o.gamma) : 1;
    const Q = [new Float64Array(nS * n), new Float64Array(nS * n)];
    for (let a = 0; a < n; a++) {
      let m1 = 0, m2 = 0;
      for (let b = 0; b < n; b++) { m1 += P[a][b][0] / n; m2 += P[b][a][1] / n; }
      for (let s = 0; s < nS; s++) { Q[0][s * n + a] = m1 * disc; Q[1][s * n + a] = m2 * disc; }
    }
    function act(q, s, eps) {
      if (rng() < eps) return Math.floor(rng() * n);
      let mx = -Infinity;
      for (let i = 0; i < n; i++) if (q[s * n + i] > mx) mx = q[s * n + i];
      const best = [];
      for (let i = 0; i < n; i++) if (q[s * n + i] >= mx - 1e-12) best.push(i);
      return best[Math.floor(rng() * best.length)];
    }
    function upd(q, s, a, r, s2) {
      let mx = -Infinity;
      for (let i = 0; i < n; i++) if (q[s2 * n + i] > mx) mx = q[s2 * n + i];
      const idx = s * n + a;
      q[idx] += o.alpha * (r + o.gamma * mx - q[idx]);
    }
    const policyOf = () => [0, 1].map((p) => {
      const out = new Array(nS);
      for (let s = 0; s < nS; s++) out[s] = argmax(Q[p], s * n, n);
      return out;
    });

    const A1 = trace ? new Uint8Array(o.steps) : null;
    const A2 = trace ? new Uint8Array(o.steps) : null;
    let s = Math.floor(rng() * nS);
    let prev = policyOf(), stable = 0, converged = false, t = 0;
    for (t = 1; t <= o.steps; t++) {
      const eps = Math.exp(-o.epsDecay * t);
      const a1 = act(Q[0], s, eps), a2 = act(Q[1], s, eps);
      const r = P[a1][a2], s2 = a1 * n + a2;
      upd(Q[0], s, a1, r[0], s2);
      upd(Q[1], s, a2, r[1], s2);
      if (trace) { A1[t - 1] = a1; A2[t - 1] = a2; }
      s = s2;
      if (t % 100 === 0 && eps < o.convEps) {
        const cur = policyOf();
        const same = cur[0].every((v, i) => v === prev[0][i]) && cur[1].every((v, i) => v === prev[1][i]);
        stable = same ? stable + 100 : 0;
        prev = cur;
        if (stable >= o.stableWindow) { converged = true; break; }
      }
    }
    t = Math.min(t, o.steps);
    const pol = policyOf();
    const { cycle } = followGreedy(pol, n, s);
    const avg = [0, 1].map((i) => cycle.reduce((acc, c) => acc + P[Math.floor(c / n)][c % n][i], 0) / cycle.length);
    const nash = new Set(game.pureNash());
    const res = {
      policies: pol, cycle, avg, converged, steps: t, finalState: s,
      isNash: cycle.length === 1 && nash.has(cycle[0]),
    };
    if (trace) { res.a1 = A1.subarray(0, t); res.a2 = A2.subarray(0, t); res.epsDecay = o.epsDecay; }
    return res;
  }

  const cycleLabel = (game, cyc) => cyc.map((c) => game.label(c)).join(">");

  function deviationTest(game, run, horizon) {
    horizon = horizon || 8;
    const n = game.n, P = game.P;
    const s0 = run.cycle[0];
    const a1 = Math.floor(s0 / n), a2 = s0 % n;
    let dev = -1, best = -Infinity;
    for (let a = 0; a < n; a++) if (a !== a1 && P[a][a2][0] > best) { best = P[a][a2][0]; dev = a; }
    let s = dev * n + a2;
    const path = [s];
    for (let i = 0; i < horizon; i++) {
      s = run.policies[0][s] * n + run.policies[1][s];
      path.push(s);
    }
    const inCycle = new Set(run.cycle);
    return {
      start: s0, forced: path[0], path,
      profit: P[dev][a2][0] - P[a1][a2][0],
      punished: path.slice(1).some((x) => !inCycle.has(x)),
      returns: inCycle.has(path[path.length - 1]),
    };
  }

  function runMany(game, opts, seeds, base) {
    const runs = [];
    for (let i = 0; i < seeds; i++) runs.push(simulate(game, opts, (base || 0) + i, false));
    return runs;
  }

  function summarise(game, runs) {
    const counts = new Map();
    runs.forEach((r) => {
      const k = cycleLabel(game, r.cycle);
      counts.set(k, (counts.get(k) || 0) + 1);
    });
    const N = runs.length;
    const outcomes = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([label, c]) => ({
      label, freq: c / N, isCycle: label.indexOf(">") >= 0,
    }));
    const mean = (f) => runs.reduce((a, r) => a + f(r), 0) / N;
    return {
      outcomes, N,
      focal: (counts.get(game.focal) || 0) / N,
      nash: mean((r) => (r.isNash ? 1 : 0)),
      converged: mean((r) => (r.converged ? 1 : 0)),
      payoff: [mean((r) => r.avg[0]), mean((r) => r.avg[1])],
    };
  }

  // ---- collusion measurement (Prisoner's Dilemma only) ----------------------------
  // Collusion index  Δ = (π̄ − π_Nash) / (π_coop − π_Nash): 0 = Nash payoff, 1 = full cooperation.
  // Classes: competitive (one-shot Nash outcome), sustained (Δ >= thr, converged, not Nash and a forced
  // deviation is punished), unpunished (same but no punishment), other.
  const CLASSES = ["sustained", "unpunished", "competitive", "other"];

  function pdBench(game) {
    if (game.key !== "pd") return null;
    const P = game.P;
    return { nash: (P[1][1][0] + P[1][1][1]) / 2, coop: (P[0][0][0] + P[0][0][1]) / 2 };
  }
  function collusionIndex(game, run) {
    const b = pdBench(game);
    return ((run.avg[0] + run.avg[1]) / 2 - b.nash) / (b.coop - b.nash);
  }
  function classify(game, run, thr, horizon) {
    thr = thr == null ? 0.5 : thr;
    const delta = collusionIndex(game, run);
    const out = { delta, cls: "other", returns: false };
    if (run.isNash) out.cls = "competitive";
    else if (run.converged && delta >= thr) {
      const d = deviationTest(game, run, horizon || 8);
      out.cls = d.punished ? "sustained" : "unpunished";
      out.returns = d.returns;
    }
    return out;
  }
  function wilson(k, n) {            // 95% Wilson score interval
    if (n === 0) return [0, 1];
    const z = 1.96, p = k / n, d = 1 + (z * z) / n;
    const c = (p + (z * z) / (2 * n)) / d;
    const hw = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
    return [Math.max(0, c - hw), Math.min(1, c + hw)];
  }
  function tally(results) {          // results: array of classify() outputs
    const N = results.length, counts = {}, rates = {}, ci = {};
    CLASSES.forEach((c) => {
      counts[c] = results.filter((r) => r.cls === c).length;
      rates[c] = counts[c] / N;
      ci[c] = wilson(counts[c], N);
    });
    const sus = results.filter((r) => r.cls === "sustained");
    return {
      N, counts, rates, ci,
      meanDelta: results.reduce((a, r) => a + r.delta, 0) / N,
      forgiving: sus.length ? sus.filter((r) => r.returns).length / sus.length : null,
    };
  }

  const QL = { GAMES, makeGame, simulate, runMany, summarise, deviationTest, cycleLabel, DEFAULTS, mulberry32,
    CLASSES, pdBench, collusionIndex, classify, wilson, tally };
  if (typeof module !== "undefined" && module.exports) module.exports = QL;
  else root.QL = QL;
})(typeof window !== "undefined" ? window : globalThis);
