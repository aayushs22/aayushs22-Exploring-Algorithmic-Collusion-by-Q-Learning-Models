/* UI for the Q-learning games explainer. Depends on qlearning.js (window.QL). */
(function () {
  "use strict";
  const QL = window.QL;
  const $ = (s) => document.querySelector(s);
  const SVGNS = "http://www.w3.org/2000/svg";

  function h(tag, attrs, parent, text) {
    const e = document.createElement(tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function s(tag, attrs, parent, text) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  const pct = (x) => Math.round(x * 100) + "%";
  const num = (x) => String(Number(x.toFixed(2)));

  // ---------------------------------------------------------------- state
  const state = { key: "pd", params: { G: 1, L: 1 }, gamma: 0.9, alpha: 0.1, steps: 30000, decay: 3e-4, seeds: 20, seed: 18 };
  let cur = null;       // current results {game, single, sum, runs}
  let sweepRes = null;  // last sweep

  const INFO = {
    pd: {
      what: "Two firms each pick <b>Cooperate</b> (keep prices high) or <b>Defect</b> (undercut). Whatever the other does, defecting pays you more: defection is a <i>dominant strategy</i>. So the only Nash equilibrium is mutual defection (D, D), even though (C, C) would leave both better off. <b>G</b> is the temptation to undercut and <b>L</b> is the loss if you cooperate and the other defects.",
      expect: "With γ = 0 expect (D, D). With patient learners (γ close to 1), some runs may keep cooperating because cheating gets punished.",
    },
    hawk_dove: {
      what: "Two players contest a prize. <b>Hawk</b> = fight, <b>Dove</b> = back down. Two Hawks clash and both get 0. Being the lone Hawk pays most. The pure equilibria are asymmetric (one Hawk, one Dove), and there is also a mixed equilibrium where each plays Hawk with probability G.",
      expect: "Expect one Hawk and one Dove, or the players taking turns. With a small G the edge of being Hawk is tiny, so many runs settle on both-Dove.",
    },
    stag_hunt: {
      what: "The <b>Safe</b> action D always pays 1. The <b>risky</b> action C pays 1+G only if the other also plays C, and 0 otherwise. Two equilibria: (C, C) is best for both, but (D, D) is safer. When G &lt; 1, D is <i>risk-dominant</i>.",
      expect: "Expect (D, D) when G &lt; 1 and mostly (C, C) when G &gt; 1: trust only pays off when the prize is big enough.",
    },
    coordination: {
      what: "Both players win only if they pick the <b>same</b> option: (A, A) pays 1+G, (B, B) pays 1, and a mismatch pays 0 to both. Two equilibria: the better one (A, A) and the lesser (B, B).",
      expect: "Since mismatching is costless to learn from, expect most runs to land on the better option A.",
    },
    bilingual: {
      what: "Players can pick <b>A</b>, <b>B</b>, or <b>AB</b> (be compatible with both, but pay a cost <b>c</b>). Higher G favours coordinating on A; a high cost c makes AB unattractive.",
      expect: "Low cost c → (A, A) is chosen. High c → players avoid AB and fall back on the safe B. (Payoff table is the one from the poster.)",
    },
  };

  const SWEEP = {
    pd: { G: [0.5, 1, 1.5, 2, 2.5, 3], L: [0.5, 1, 1.5, 2, 2.5, 3] },
    hawk_dove: { G: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] },
    stag_hunt: { G: [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3] },
    coordination: { G: [0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3] },
    bilingual: { G: [0.2, 0.4, 0.6, 0.8, 1], c: [0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2] },
  };

  const PRESETS = [
    { label: "Price war (γ = 0)", key: "pd", params: { G: 1, L: 1 }, gamma: 0,
      note: "<b>What to notice:</b> short-sighted learners (γ = 0) end at <code>DD</code> in essentially every run. Defecting is dominant, so there is no reason to cooperate. This is the textbook Nash prediction." },
    { label: "Patient learners (γ = 0.9)", key: "pd", params: { G: 1, L: 1 }, gamma: 0.9,
      note: "<b>What to notice:</b> once the future matters, some runs end at <code>CC</code>, which is <i>not</i> a one-shot equilibrium. Scroll to section 5 to see whether a cheater gets punished: that is what makes it collusion." },
    { label: "Stag hunt, G = 0.5", key: "stag_hunt", params: { G: 0.5 }, gamma: 0,
      note: "<b>What to notice:</b> with G &lt; 1 the safe action D is risk-dominant, and the learners settle on <code>DD</code>." },
    { label: "Stag hunt, G = 1.5", key: "stag_hunt", params: { G: 1.5 }, gamma: 0,
      note: "<b>What to notice:</b> with G &gt; 1 the risky cooperative outcome <code>CC</code> wins nearly every run. Run the sweep (section 6) to see the flip at G = 1." },
    { label: "Hawk-Dove", key: "hawk_dove", params: { G: 0.5 }, gamma: 0,
      note: "<b>What to notice:</b> outcomes are asymmetric (<code>HD</code> or <code>DH</code>), or the players alternate (<code>HD&gt;DH</code>). Roles are decided by chance early in learning." },
    { label: "Bilingual, costly AB", key: "bilingual", params: { G: 0.5, c: 1 }, gamma: 0,
      note: "<b>What to notice:</b> with a high cost, both avoid the <code>AB</code> option. Lower c in the controls and watch the outcome change." },
  ];

  // ---------------------------------------------------------------- tooltip
  const tip = $("#tip");
  function showTip(title, rows, x, y) {
    tip.textContent = "";
    if (title) h("div", { class: "t" }, tip, title);
    rows.forEach((r) => {
      const d = h("div", { class: "r" }, tip);
      if (r.color) { const k = h("i", {}, d); k.style.background = r.color; }
      h("b", {}, d, r.value);
      h("span", {}, d, r.name || "");
    });
    tip.style.display = "block";
    const w = tip.offsetWidth, hh = tip.offsetHeight;
    let left = x + 14, top = y + 14;
    if (left + w > window.innerWidth - 8) left = x - w - 14;
    if (top + hh > window.innerHeight - 8) top = y - hh - 14;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }
  const hideTip = () => { tip.style.display = "none"; };

  // ---------------------------------------------------------------- generic line chart
  function lineChart(host, cfg) {
    host.textContent = "";
    const W = 760, H = 320, m = { l: 52, r: 120, t: 14, b: 46 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const xs = cfg.xs, n = xs.length, x0 = xs[0], x1 = xs[n - 1];
    const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * iw;
    const Y = (v) => m.t + (1 - v) * ih;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": cfg.aria, tabindex: "0" }, host);

    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      s("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: v === 0 ? "axisline" : "gridline" }, svg);
      s("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, svg, pct(v));
    });
    const ticks = cfg.xTicks || Array.from({ length: 5 }, (_, i) => x0 + ((x1 - x0) * i) / 4);
    ticks.forEach((v) => s("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle" }, svg, cfg.fmtX(v)));
    s("text", { x: m.l + iw / 2, y: H - 6, "text-anchor": "middle", class: "lbl" }, svg, cfg.xLabel);
    s("text", { x: 14, y: m.t + ih / 2, "text-anchor": "middle", class: "lbl", transform: `rotate(-90 14 ${m.t + ih / 2})` }, svg, cfg.yLabel);

    if (cfg.vline != null && cfg.vline >= x0 && cfg.vline <= x1) {
      s("line", { x1: X(cfg.vline), x2: X(cfg.vline), y1: m.t, y2: m.t + ih, class: "axisline" }, svg);
      s("text", { x: X(cfg.vline) - 6, y: m.t + ih * 0.62, "text-anchor": "end" }, svg, cfg.vlineLabel);
    }
    cfg.series.forEach((se) => {
      const d = se.values.map((v, i) => (i ? "L" : "M") + X(xs[i]).toFixed(1) + " " + Y(v).toFixed(1)).join(" ");
      s("path", { d, class: "line " + se.cls }, svg);
      if (cfg.markers) se.values.forEach((v, i) => s("circle", { cx: X(xs[i]), cy: Y(v), r: 4, fill: se.fill, class: "dot" }, svg));
    });

    // direct labels with simple de-collision
    const labs = cfg.series.map((se) => ({ se, y: Y(se.values[n - 1]) })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < labs.length; i++) if (labs[i].y < labs[i - 1].y + 15) labs[i].y = labs[i - 1].y + 15;
    labs.forEach((l) => {
      s("line", { x1: W - m.r + 8, x2: W - m.r + 22, y1: l.y, y2: l.y, class: "line " + l.se.cls }, svg);
      s("text", { x: W - m.r + 27, y: l.y + 4, class: "lbl" }, svg, l.se.name);
    });

    // hover layer
    const cross = s("line", { y1: m.t, y2: m.t + ih, class: "cross", visibility: "hidden" }, svg);
    const dots = cfg.series.map((se) => s("circle", { r: 4.5, fill: se.fill, class: "dot", visibility: "hidden" }, svg));
    const overlay = s("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
    let idx = n - 1;
    function show(i, cx, cy) {
      idx = Math.max(0, Math.min(n - 1, i));
      const px = X(xs[idx]);
      cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
      cfg.series.forEach((se, k) => {
        dots[k].setAttribute("cx", px); dots[k].setAttribute("cy", Y(se.values[idx])); dots[k].setAttribute("visibility", "visible");
      });
      if (cx == null) { const r = svg.getBoundingClientRect(); cx = r.left + (px / W) * r.width; cy = r.top + (m.t / H) * r.height + 20; }
      showTip(cfg.tipTitle(xs[idx]), cfg.series.map((se) => ({ color: se.color, value: se.fmt(se.values[idx]), name: se.name })), cx, cy);
    }
    function hide() { cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); hideTip(); }
    overlay.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect();
      const vx = ((e.clientX - r.left) / r.width) * W;
      const xv = x0 + ((vx - m.l) / iw) * (x1 - x0);
      let best = 0;
      for (let i = 1; i < n; i++) if (Math.abs(xs[i] - xv) < Math.abs(xs[best] - xv)) best = i;
      show(best, e.clientX, e.clientY);
    });
    overlay.addEventListener("pointerleave", hide);
    svg.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") { show(idx - 1); e.preventDefault(); }
      if (e.key === "ArrowRight") { show(idx + 1); e.preventDefault(); }
      if (e.key === "Escape") hide();
    });
    svg.addEventListener("blur", hide);
  }

  // ---------------------------------------------------------------- game description
  function renderMatrix(game) {
    const host = $("#matrix");
    host.textContent = "";
    const t = h("table", { class: "matrix" }, host);
    const hr = h("tr", {}, t);
    h("th", {}, hr, "P1 ↓ / P2 →");
    game.names.forEach((nm) => h("th", {}, hr, nm));
    const ne = new Set(game.pureNash());
    for (let a1 = 0; a1 < game.n; a1++) {
      const r = h("tr", {}, t);
      h("th", {}, r, game.names[a1]);
      for (let a2 = 0; a2 < game.n; a2++) {
        const c = h("td", {}, r, `${num(game.P[a1][a2][0])}, ${num(game.P[a1][a2][1])}`);
        if (ne.has(a1 * game.n + a2)) { c.className = "ne"; h("span", { class: "tag" }, c, "Nash equilibrium"); }
      }
    }
    const info = INFO[game.key];
    const g = $("#gameText");
    let extra = "";
    if (game.key === "hawk_dove") extra = ` With G = ${num(game.params.G)}, the mixed equilibrium has each player choosing Hawk ${pct(game.params.G)} of the time.`;
    if (game.key === "stag_hunt") extra = game.params.G < 1
      ? ` Here G = ${num(game.params.G)} &lt; 1, so the safe D is risk-dominant.`
      : game.params.G > 1 ? ` Here G = ${num(game.params.G)} &gt; 1, so the risky C beats the safe 1 at 50/50 odds.` : " Here G = 1 is exactly the threshold.";
    g.innerHTML = `<h3 style="margin-top:0">${game.name}</h3><p>${info.what}${extra}</p><p class="muted"><b>What to expect:</b> ${info.expect}</p>`;
  }

  // ---------------------------------------------------------------- single run
  function renderRun(game, run) {
    $("#firstAct").textContent = `${game.names[0]} = "${game.actions[0]}"`;
    const T = run.steps;
    const W = Math.max(50, Math.min(300, Math.floor(T / 3)));
    const c1 = new Uint32Array(T + 1), c2 = new Uint32Array(T + 1);
    for (let i = 0; i < T; i++) { c1[i + 1] = c1[i] + (run.a1[i] === 0 ? 1 : 0); c2[i + 1] = c2[i] + (run.a2[i] === 0 ? 1 : 0); }
    const xs = [], p1 = [], p2 = [], ep = [];
    const push = (t) => { xs.push(t); p1.push((c1[t] - c1[t - W]) / W); p2.push((c2[t] - c2[t - W]) / W); ep.push(Math.exp(-run.epsDecay * t)); };
    for (let t = W; t <= T; t += 100) push(t);
    if (xs[xs.length - 1] !== T) push(T);

    const pc = (v) => pct(v);
    lineChart($("#runChart"), {
      xs, aria: "Share of recent periods in which each player chooses the first action, and exploration rate, over time",
      series: [
        { name: "Player 1", cls: "st1", color: "var(--s1)", fill: "var(--s1)", values: p1, fmt: pc },
        { name: "Player 2", cls: "st2", color: "var(--s2)", fill: "var(--s2)", values: p2, fmt: pc },
        { name: "exploration ε", cls: "stm", color: "var(--muted)", fill: "var(--muted)", values: ep, fmt: pc },
      ],
      xLabel: "Period (time)", yLabel: `Share playing "${game.actions[0]}" (last ${W} periods)`,
      fmtX: (v) => (v >= 1000 ? Math.round(v / 100) / 10 + "k" : String(v)),
      tipTitle: (x) => "Period " + Math.round(x).toLocaleString(),
      vline: run.converged ? T : null, vlineLabel: "strategies stable",
    });
    const lg = $("#runLegend");
    lg.innerHTML = '<span><i class="k" style="background:var(--s1)"></i>Player 1</span><span><i class="k" style="background:var(--s2)"></i>Player 2</span><span><i class="k" style="background:var(--muted)"></i>exploration rate ε (share of random moves)</span>';

    // outcome text
    const lab = QL.cycleLabel(game, run.cycle);
    const nashTxt = run.isNash ? "a pure Nash equilibrium of the one-shot game"
      : run.cycle.length > 1 ? "a repeating cycle, not a single equilibrium" : "<b>not</b> a Nash equilibrium of the one-shot game";
    $("#runText").innerHTML = `<b>Long-run outcome: <code>${lab}</code></b>, ${nashTxt}. Average payoff per period: Player 1 = ${num(run.avg[0])}, Player 2 = ${num(run.avg[1])}. ` +
      (run.converged ? `Both strategies stopped changing after ${T.toLocaleString()} periods.` : "Strategies were still changing when the period limit was reached.");

    // policy table
    const host = $("#policy");
    host.textContent = "";
    const t = h("table", { class: "tbl" }, host);
    const hr = h("tr", {}, t);
    ["If last round was…", "Player 1 plays", "Player 2 plays", "…so next round is"].forEach((x) => h("th", {}, hr, x));
    const onPath = new Set(run.cycle);
    for (let st = 0; st < game.n * game.n; st++) {
      const r = h("tr", onPath.has(st) ? { class: "on-path" } : {}, t);
      const a1 = run.policies[0][st], a2 = run.policies[1][st];
      h("td", {}, r, game.label(st));
      h("td", {}, r, game.names[a1]);
      h("td", {}, r, game.names[a2]);
      h("td", {}, r, game.label(a1 * game.n + a2));
    }

    // table view of chart data
    const rt = $("#runTable");
    rt.textContent = "";
    const tt = h("table", { class: "tbl" }, rt);
    const th = h("tr", {}, tt);
    ["Period", "Player 1", "Player 2", "Exploration ε"].forEach((x) => h("th", {}, th, x));
    const stepIdx = Math.max(1, Math.floor(xs.length / 12));
    for (let i = 0; i < xs.length; i += stepIdx) {
      const r = h("tr", {}, tt);
      [xs[i].toLocaleString(), pct(p1[i]), pct(p2[i]), pct(ep[i])].forEach((x) => h("td", {}, r, x));
    }
  }

  // ---------------------------------------------------------------- many runs
  function renderMany(game, sum, runs) {
    const tiles = $("#tiles");
    tiles.textContent = "";
    const T = (v, l) => { const d = h("div", { class: "tile" }, tiles); h("div", { class: "v" }, d, v); h("div", { class: "l" }, d, l); };
    T(pct(sum.focal), `of runs end at ${game.focal} (focal outcome)`);
    T(pct(sum.nash), "end at a pure Nash equilibrium");
    T(`${num(sum.payoff[0])} / ${num(sum.payoff[1])}`, "avg payoff per period (P1 / P2)");
    T(pct(sum.converged), `of ${sum.N} runs fully converged`);

    // stacked bar: coloured single-state outcomes + one grey "cycles/other" segment
    const host = $("#stack");
    host.textContent = "";
    const segs = [];
    let other = 0;
    sum.outcomes.forEach((o) => {
      const c = !o.isCycle && game.colors[o.label];
      if (c) segs.push({ label: o.label, freq: o.freq, c }); else other += o.freq;
    });
    if (other > 0) segs.push({ label: "cycles / other", freq: other, c: 0 });
    const bar = h("div", { class: "stack", role: "img", "aria-label": "Share of runs by final outcome" }, host);
    segs.forEach((g) => {
      const d = h("div", { class: `seg c${g.c} ${g.c === 4 ? "c4" : ""}`, tabindex: "0" }, bar, g.freq >= 0.07 ? `${g.label} ${pct(g.freq)}` : "");
      d.style.flex = String(g.freq);
      const f = (e) => { const r = d.getBoundingClientRect(); showTip(null, [{ value: pct(g.freq), name: `of runs → ${g.label}` }], e.clientX || r.left + r.width / 2, e.clientY || r.top); };
      d.addEventListener("pointermove", f); d.addEventListener("focus", f);
      d.addEventListener("pointerleave", hideTip); d.addEventListener("blur", hideTip);
    });

    // table (also the table-view fallback for the colours)
    const ot = $("#outTable");
    ot.textContent = "";
    const t = h("table", { class: "tbl" }, ot);
    const hr = h("tr", {}, t);
    ["", "Final outcome", "Share of runs", "Avg payoff (P1, P2)", "Nash equilibrium?"].forEach((x) => h("th", {}, hr, x));
    sum.outcomes.forEach((o) => {
      const run = runs.find((r) => QL.cycleLabel(game, r.cycle) === o.label);
      const r = h("tr", {}, t);
      const c = !o.isCycle && game.colors[o.label];
      const sw = h("td", {}, r);
      h("span", { class: `sw c${c || 0}`, style: "display:inline-block;width:12px;height:12px;border-radius:3px" }, sw);
      h("td", {}, r, o.label + (o.isCycle ? "  (cycle)" : ""));
      h("td", {}, r, pct(o.freq));
      h("td", {}, r, `${num(run.avg[0])}, ${num(run.avg[1])}`);
      h("td", {}, r, run.isNash ? "yes" : "no");
    });
  }

  function manyNarrative(game, sum) {
    const top = sum.outcomes[0];
    const g = state.gamma, G = game.params.G;
    let t = `The most common result is <code>${top.label}</code> (${pct(top.freq)} of ${sum.N} runs). `;
    t += `${pct(sum.nash)} of runs end at a Nash equilibrium of the one-shot game. `;
    if (game.key === "pd") {
      if (g === 0) t += "With γ = 0 the learners are purely short-sighted, so defection is dominant. Raise γ to see cooperation emerge.";
      else if (sum.focal > 0) t += `${pct(sum.focal)} of runs end at <code>CC</code>, which is <b>not</b> an equilibrium of the one-shot game. The next section checks whether it is held up by punishment.`;
      else t += `Even with γ = ${num(g)}, no run reached CC. Try a larger γ, or a smaller temptation G.`;
    } else if (game.key === "stag_hunt") {
      t += G < 1 ? "G &lt; 1 makes the safe action risk-dominant, so the learners mostly avoid the risky cooperative option." : "G &gt; 1 makes the cooperative outcome attractive enough that learners mostly find it.";
    } else if (game.key === "hawk_dove") {
      t += "Runs that do not end at a Nash equilibrium (both Dove, or cycles) are usually <i>self-confirming</i>: once exploration fades, the Hawk option is rarely retried, so its pessimistic early estimate sticks.";
    } else if (game.key === "bilingual") {
      t += "Compare with a different cost c to see where the preference flips.";
    }
    $("#manyText").innerHTML = t;
  }

  // ---------------------------------------------------------------- deviation test
  function renderDev(game, run) {
    const d = QL.deviationTest(game, run, 6);
    const host = $("#dev");
    host.textContent = "";
    const inCyc = new Set(run.cycle);
    const path = h("div", { class: "path" }, host);
    const mk = (cls, big, small) => { const p = h("div", { class: "pill " + cls }, path); h("b", {}, p, big); h("span", {}, p, small); };
    mk("cyc", QL.cycleLabel(game, run.cycle), "settled outcome");
    h("div", { class: "arrow" }, path, "→");
    mk("forced", game.label(d.forced), "P1 cheats once");
    d.path.slice(1).forEach((st, i) => {
      h("div", { class: "arrow" }, path, "→");
      mk(inCyc.has(st) ? "cyc" : "", game.label(st), `t+${i + 1}${inCyc.has(st) ? "" : " · off-path"}`);
    });
    let msg;
    if (d.profit <= 1e-9) {
      msg = `Cheating would not even pay in the first period (gain ${num(d.profit)}), as expected at a Nash equilibrium. There is nothing to punish.`;
    } else if (d.punished && d.returns) {
      msg = `Cheating gains ${num(d.profit)} for one period, but then play <b>leaves the settled outcome</b> (retaliation) and later <b>returns to it</b>. That is reward-and-punishment behaviour: the outcome is sustained by a threat, not by chance.`;
    } else if (d.punished) {
      msg = `Cheating gains ${num(d.profit)} once, and play moves off the settled outcome and does not return within 6 periods. The learned strategies react to the deviation, but not by simply forgiving.`;
    } else {
      msg = `Cheating gains ${num(d.profit)} and the other player does <b>not</b> react: play goes straight back to the settled outcome. This outcome is not backed by punishment.`;
    }
    h("div", { class: "takeaway" }, host).innerHTML = msg;
    if (game.key === "pd" && QL.cycleLabel(game, run.cycle) !== "CC")
      h("p", { class: "small muted" }, host, "Tip: use the preset “Patient learners (γ = 0.9)” and try a few seeds in section 3 until the settled outcome is CC.");
  }

  // ---------------------------------------------------------------- heatmap
  function heatmap(host, cfg) {
    host.textContent = "";
    const W = 760, m = { l: 66, r: 16, t: 10, b: 52 };
    const nx = cfg.xs.length, ny = cfg.ys.length;
    const cw = (W - m.l - m.r) / nx, ch = Math.min(56, cw);
    const H = m.t + ch * ny + m.b;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": cfg.aria }, host);
    cfg.ys.forEach((yv, j) => s("text", { x: m.l - 8, y: m.t + (ny - 1 - j) * ch + ch / 2 + 4, "text-anchor": "end" }, svg, String(yv)));
    cfg.xs.forEach((xv, i) => s("text", { x: m.l + i * cw + cw / 2, y: m.t + ny * ch + 18, "text-anchor": "middle" }, svg, String(xv)));
    s("text", { x: m.l + (nx * cw) / 2, y: H - 8, "text-anchor": "middle", class: "lbl" }, svg, cfg.xLabel);
    s("text", { x: 14, y: m.t + (ny * ch) / 2, "text-anchor": "middle", class: "lbl", transform: `rotate(-90 14 ${m.t + (ny * ch) / 2})` }, svg, cfg.yLabel);
    cfg.ys.forEach((yv, j) => cfg.xs.forEach((xv, i) => {
      const v = cfg.vals[j][i];
      const bin = Math.min(6, Math.floor(v * 7));
      const r = s("rect", { x: m.l + i * cw, y: m.t + (ny - 1 - j) * ch, width: cw, height: ch, class: "hm-cell", fill: `var(--seq${bin})`, tabindex: "0",
        "aria-label": `${cfg.xLabel} ${xv}, ${cfg.yLabel} ${yv}: ${pct(v)}` }, svg);
      const f = (e) => { const b = r.getBoundingClientRect(); showTip(`${cfg.xLabel} = ${xv}, ${cfg.yLabel} = ${yv}`, [{ value: pct(v), name: cfg.metricText }], e.clientX || b.left + b.width / 2, e.clientY || b.top); };
      r.addEventListener("pointermove", f); r.addEventListener("focus", f);
      r.addEventListener("pointerleave", hideTip); r.addEventListener("blur", hideTip);
    }));
    const lg = h("div", { class: "legend" }, host);
    h("span", {}, lg, "Share of runs:");
    for (let k = 0; k < 7; k++) {
      const sp = h("span", {}, lg);
      h("i", { class: "sw", style: `background:var(--seq${k})` }, sp);
      sp.appendChild(document.createTextNode(`${Math.round((k / 7) * 100)}–${Math.round(((k + 1) / 7) * 100)}%`));
    }
  }

  // ---------------------------------------------------------------- sweep
  function opts() { return { alpha: state.alpha, gamma: state.gamma, steps: state.steps, epsDecay: state.decay }; }

  async function runSweep() {
    const def = QL.GAMES[state.key];
    const keys = def.params.map((p) => p.key);
    const sw = SWEEP[state.key];
    const xs = sw[keys[0]];
    const ys = keys.length > 1 ? sw[keys[1]] : [null];
    const seeds = keys.length > 1 ? Math.min(state.seeds, 12) : Math.min(state.seeds, 30);
    const total = xs.length * ys.length;
    const prog = $("#prog");
    prog.style.display = "inline-block"; prog.max = total; prog.value = 0;
    $("#sweepBtn").disabled = true;
    const cells = ys.map(() => []);
    let done = 0;
    for (let j = 0; j < ys.length; j++) {
      for (let i = 0; i < xs.length; i++) {
        const p = Object.assign({}, state.params);
        p[keys[0]] = xs[i];
        if (keys.length > 1) p[keys[1]] = ys[j];
        const game = QL.makeGame(state.key, p);
        const sum = QL.summarise(game, QL.runMany(game, opts(), seeds, 500));
        cells[j].push({ focal: sum.focal, nash: sum.nash });
        prog.value = ++done;
        $("#sweepStatus").textContent = `${done} / ${total} settings`;
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    prog.style.display = "none";
    $("#sweepBtn").disabled = false;
    $("#sweepStatus").textContent = `Done: ${seeds} runs per setting, γ = ${num(state.gamma)}. With so few runs each percentage is only accurate to about ±15 points; increase “Runs” for smoother results.`;
    sweepRes = { key: state.key, keys, xs, ys, cells, seeds, gamma: state.gamma, focal: QL.GAMES[state.key].focal };
    renderSweep();
  }

  function renderSweep() {
    if (!sweepRes) return;
    const r = sweepRes, host = $("#sweepChart"), tbl = $("#sweepTable"), txt = $("#sweepText");
    const metric = $("#metric").value;
    const metricText = metric === "focal" ? `of runs end at ${r.focal}` : "of runs end at a Nash equilibrium";
    const label = (k) => (k === "G" ? "G" : k === "L" ? "L" : k);
    txt.style.display = "block";
    tbl.textContent = "";
    if (r.keys.length > 1) {
      const vals = r.ys.map((_, j) => r.xs.map((_, i) => r.cells[j][i][metric]));
      heatmap(host, { xs: r.xs, ys: r.ys, vals, xLabel: label(r.keys[0]), yLabel: label(r.keys[1]), metricText,
        aria: `Heatmap of the share of runs ${metricText} across ${r.keys[0]} and ${r.keys[1]}` });
      let best = { v: -1 };
      vals.forEach((row, j) => row.forEach((v, i) => { if (v > best.v) best = { v, i, j }; }));
      txt.innerHTML = `Highest share: <b>${pct(best.v)}</b> ${metricText}, at ${r.keys[0]} = ${r.xs[best.i]}, ${r.keys[1]} = ${r.ys[best.j]}.`;
      if ($("#sweepTbl").checked) {
        const t = h("table", { class: "tbl" }, tbl);
        const hr = h("tr", {}, t);
        h("th", {}, hr, `${r.keys[1]} \\ ${r.keys[0]}`);
        r.xs.forEach((x) => h("th", {}, hr, String(x)));
        r.ys.forEach((y, j) => { const row = h("tr", {}, t); h("th", {}, row, String(y)); r.xs.forEach((_, i) => h("td", {}, row, pct(vals[j][i]))); });
      }
    } else {
      const f = r.cells[0].map((c) => c.focal), n = r.cells[0].map((c) => c.nash);
      const pc = (v) => pct(v);
      lineChart(host, {
        xs: r.xs, markers: true, aria: `Share of runs by outcome as ${r.keys[0]} varies`,
        xTicks: r.xs.length > 8 ? r.xs.filter((_, i) => i % 2 === 0) : r.xs,
        series: [
          { name: `ends at ${r.focal}`, cls: "st1", color: "var(--s1)", fill: "var(--s1)", values: f, fmt: pc },
          { name: "ends at Nash", cls: "st2", color: "var(--s2)", fill: "var(--s2)", values: n, fmt: pc },
        ],
        xLabel: `${label(r.keys[0])} (payoff parameter)`, yLabel: "Share of runs", fmtX: (v) => String(v), tipTitle: (x) => `${r.keys[0]} = ${x}`,
      });
      let cross = null;
      for (let i = 1; i < f.length; i++) if (f[i - 1] < 0.5 && f[i] >= 0.5) { cross = [r.xs[i - 1], r.xs[i]]; break; }
      txt.innerHTML = cross ? `The share ending at ${r.focal} climbs from below 50% at ${r.keys[0]} = ${cross[0]} to ${pct(f[r.xs.indexOf(cross[1])])} at ${cross[1]}: that is where the learners' behaviour flips.`
        : `The share ending at ${r.focal} never crosses 50% in this range (min ${pct(Math.min(...f))}, max ${pct(Math.max(...f))}).`;
      if ($("#sweepTbl").checked) {
        const t = h("table", { class: "tbl" }, tbl);
        const hr = h("tr", {}, t);
        [r.keys[0], `ends at ${r.focal}`, "ends at Nash"].forEach((x) => h("th", {}, hr, x));
        r.xs.forEach((x, i) => { const row = h("tr", {}, t); [String(x), pct(f[i]), pct(n[i])].forEach((v) => h("td", {}, row, v)); });
      }
    }
  }

  function resetSweep() {
    sweepRes = null;
    $("#sweepChart").textContent = ""; $("#sweepTable").textContent = ""; $("#sweepText").style.display = "none";
    $("#sweepStatus").textContent = "Settings changed. Click the button to run a sweep for them.";
  }

  // ---------------------------------------------------------------- orchestration
  function runAll() {
    const game = QL.makeGame(state.key, state.params);
    const single = QL.simulate(game, opts(), state.seed, true);
    const runs = QL.runMany(game, opts(), state.seeds, 1000);
    const sum = QL.summarise(game, runs);
    cur = { game, single, sum, runs };
    renderMatrix(game);
    renderRun(game, single);
    renderMany(game, sum, runs);
    manyNarrative(game, sum);
    renderDev(game, single);
  }

  let timer = null;
  function schedule() { clearTimeout(timer); timer = setTimeout(() => { runAll(); resetSweep(); }, 120); }

  // ---------------------------------------------------------------- controls
  function buildParams() {
    const box = $("#paramBox");
    box.textContent = "";
    QL.GAMES[state.key].params.forEach((p) => {
      const d = h("div", {}, box);
      const l = h("label", { for: "p_" + p.key }, d, p.label + " ");
      const v = h("span", { class: "val" }, l, num(state.params[p.key]));
      const inp = h("input", { type: "range", id: "p_" + p.key, min: p.min, max: p.max, step: p.step }, d);
      inp.value = state.params[p.key];
      inp.addEventListener("input", () => { state.params[p.key] = parseFloat(inp.value); v.textContent = num(state.params[p.key]); schedule(); });
    });
  }
  function syncInputs() {
    $("#gameSel").value = state.key;
    buildParams();
    $("#gamma").value = state.gamma; $("#gammaV").textContent = num(state.gamma);
    $("#seeds").value = state.seeds; $("#seed").value = state.seed;
    $("#alpha").value = state.alpha; $("#alphaV").textContent = num(state.alpha);
    $("#steps").value = state.steps; $("#decay").value = state.decay; $("#decayV").textContent = state.decay.toExponential(1);
  }
  function bind() {
    const gs = $("#gameSel");
    Object.keys(QL.GAMES).forEach((k) => h("option", { value: k }, gs, QL.GAMES[k].name));
    gs.addEventListener("change", () => {
      state.key = gs.value;
      state.params = {};
      QL.GAMES[state.key].params.forEach((p) => (state.params[p.key] = p.def));
      $("#presetNote").style.display = "none";
      syncInputs(); schedule();
    });
    const link = (id, key, parse, after) => $(id).addEventListener("input", (e) => {
      const v = parse(e.target.value);
      if (Number.isNaN(v)) return;
      state[key] = v; if (after) after(v); schedule();
    });
    link("#gamma", "gamma", parseFloat, (v) => ($("#gammaV").textContent = num(v)));
    link("#alpha", "alpha", parseFloat, (v) => ($("#alphaV").textContent = num(v)));
    link("#decay", "decay", parseFloat, (v) => ($("#decayV").textContent = v.toExponential(1)));
    link("#steps", "steps", (x) => Math.max(5000, parseInt(x, 10)));
    link("#seeds", "seeds", (x) => Math.max(5, Math.min(200, parseInt(x, 10))));
    $("#seed").addEventListener("input", (e) => { const v = parseInt(e.target.value, 10); if (!Number.isNaN(v)) { state.seed = v; schedule(); } });
    $("#newSeed").addEventListener("click", () => { state.seed = Math.floor(Math.random() * 1000); $("#seed").value = state.seed; runAll(); });
    $("#sweepBtn").addEventListener("click", runSweep);
    $("#metric").addEventListener("change", renderSweep);
    $("#sweepTbl").addEventListener("change", renderSweep);
    $("#theme").addEventListener("click", () => {
      const de = document.documentElement;
      const eff = de.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      de.dataset.theme = eff === "dark" ? "light" : "dark";
    });
    const pr = $("#presets");
    PRESETS.forEach((p) => {
      const b = h("button", { type: "button", class: "chip" }, pr, p.label);
      b.addEventListener("click", () => {
        state.key = p.key; state.params = Object.assign({}, p.params); state.gamma = p.gamma;
        syncInputs();
        const n = $("#presetNote"); n.innerHTML = p.note; n.style.display = "block";
        runAll(); resetSweep();
      });
    });
  }

  bind();
  syncInputs();
  runAll();
  resetSweep();
  $("#sweepStatus").textContent = "Click the button to run a sweep (takes a few seconds).";
  if (location.hash === "#sweep") runSweep();   // deep link: open the page with the sweep already run
})();
