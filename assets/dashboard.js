/* dashboard.js — the home page (index.html). A switch at the top shows everything or one part of the course
   (Foundations · ML · Deep Learning · LLMs & Agents).
   Layout: left column (hello + today's goal, streaks, the big gauge, review) · right column (activity year grid,
   "your progress" cards with what to continue, weak spots, all categories on demand). Data: data/progress.js. */
(function () {
  const main = $("#content"), PR = window.PROGRESS;
  if (!main || !PR) return;
  const GOAL = 3;                                                         // actions a day for the "today" ring
  const ICON = (d, cls = "cat-ico") => `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true">${d}</svg>`;
  const I = {
    turtle: '<path d="M4 14c0-3.9 3.6-7 8-7s8 3.1 8 7H4z"/><path d="M20 12.5h1.5a1 1 0 0 0 0-2H19"/><path d="M7 14v3M17 14v3M11 14v3"/><path d="M9 7.4 10.5 10l1.5-2.6L13.5 10 15 7.4"/>',
    snake: '<path d="M3 17.5c2.5 0 2.5-3.5 5-3.5s2.5 3.5 5 3.5 2.5-3.5 5-3.5c1.5 0 2-1.6 2-3.2V8.5a2.5 2.5 0 0 0-5 0"/><path d="M17.8 8.4h.01"/>',
    flame: '<path d="M12 22c4 0 7-2.7 7-6.8 0-3.6-2.4-6-4.6-8.7-.5 2.5-1.8 3.8-3.2 4.4C11.6 7.4 10 4.5 8 2c.3 3.6-3 6.2-3 11.2C5 18.6 8 22 12 22z"/>',
    trophy: '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0V4z"/><path d="M7 6H4a3 3 0 0 0 3 4M17 6h3a3 3 0 0 1-3 4"/>',
    pulse: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    layers: '<path d="m12 3 9 5-9 5-9-5 9-5z"/><path d="m3 13 9 5 9-5"/>',
    arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    chev: '<path d="m6 9 6 6 6-6"/>',
    help: '<circle cx="12" cy="12" r="9"/><path d="M9.6 9.3a2.5 2.5 0 0 1 4.8.9c0 1.7-2.4 2.2-2.4 3.6"/><path d="M12 17h.01"/>',
    gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    book: '<path d="M2.5 5.5h6.5a3 3 0 0 1 3 3V20a2.2 2.2 0 0 0-2.2-2.2H2.5z"/><path d="M21.5 5.5H15a3 3 0 0 0-3 3V20a2.2 2.2 0 0 1 2.2-2.2h7.3z"/>',
    braces: '<path d="M8.5 4C6.5 4 6.5 5 6.5 7v2.2c0 1.6-.9 2.8-2.3 2.8 1.4 0 2.3 1.2 2.3 2.8V17c0 2 0 3 2 3"/><path d="M15.5 4c2 0 2 1 2 3v2.2c0 1.6.9 2.8 2.3 2.8-1.4 0-2.3 1.2-2.3 2.8V17c0 2 0 3-2 3"/>',
    sigma: '<path d="M17.5 6.5V4h-11l6 8-6 8h11v-2.5"/>',
    chart: '<path d="M3 20.5h18"/><path d="M6.5 20.5V12"/><path d="M12 20.5V5.5"/><path d="M17.5 20.5v-6"/>',
    net: '<circle cx="5" cy="6.5" r="2"/><circle cx="5" cy="17.5" r="2"/><circle cx="12" cy="12" r="2.2"/><circle cx="19" cy="12" r="2"/><path d="m6.8 7.5 3.4 3M6.8 16.5l3.4-3M14.2 12H17"/>',
    chat: '<path d="M4 5h16v11H10l-5 4v-4H4z"/><path d="M8.5 10.5h.01M12 10.5h.01M15.5 10.5h.01"/>',
    bot: '<rect x="4" y="8" width="16" height="11.5" rx="3.2"/><path d="M12 4.5V8"/><circle cx="12" cy="3.6" r="1.1"/><path d="M9 13v1.6M15 13v1.6"/>',
    flask: '<path d="M9 3h6M10 3v6L4.8 18a2 2 0 0 0 1.7 3h11a2 2 0 0 0 1.7-3L14 9V3"/><path d="M7.3 14.5h9.4"/>',
  };
  const SITES = {
    all: { name: "All", mascot: I.layers },
    found: { name: "Foundations", mascot: I.sigma },
    ml: { name: "ML", mascot: I.chart },
    dl: { name: "Deep Learning", mascot: I.net },
    llm: { name: "LLMs & Agents", mascot: I.chat },
  };
  let cur = "all", showCats = false, year = new Date().getFullYear(), pop = null;
  try { const v = sessionStorage.getItem(lsKey("dash:view")); if (SITES[v]) cur = v; } catch (e) {}
  const isDue = (it) => !!(it && it.status && it.next && it.next <= Date.now() + 60000);
  const DOC_ICON = { math: I.sigma, pyml: I.snake, stats: I.chart, ml: I.chart, cml: I.sigma, dl: I.net, tf: I.layers, llm: I.chat, ag: I.bot };
  const HELP = {
    found: ["<strong>Math for ML</strong>: the maths every later page uses (vectors, gradients, probability), each idea with a picture and a few lines of NumPy.",
      "<strong>Python for ML</strong>: NumPy, pandas, matplotlib and scikit-learn. Press <strong>Edit &amp; run</strong> on any example to change it and run it right here."],
    ml: ["<strong>Machine Learning</strong>: classic models (regression, trees, SVMs, clustering, PCA), how to evaluate them, and the traps (leakage, overfitting)."],
    dl: ["<strong>Deep Learning</strong>: neurons to CNNs and LSTMs, with backprop worked out by hand and a tiny autograd engine you can step through."],
    llm: ["<strong>Transformers</strong>: tokens, attention and the transformer block, built from scratch in NumPy.",
      "<strong>LLMs</strong>: pretraining, fine-tuning, RLHF, decoding, prompting and RAG.",
      "<strong>Agentic AI</strong>: tools, ReAct, memory, multi-agent patterns, MCP and evaluation, with a runnable toy agent."],
  };

  /* ---------- one part of the course: its gauge, courses, groups and an index of its items ---------- */
  function part(pt, st) {
    const docs = SITE.docs.filter((d) => d.part === pt), prac = PR.prac.filter((p) => PR.part[p.cat] === pt);
    const index = {};                                                    // key → title + link (for review / weak spots)
    docs.forEach((d) => (PR.docs[d.kind] || []).forEach((x) => (index[d.kind + ":" + x.id] = { title: x.title, href: d.href + "#" + x.id })));
    prac.forEach((p) => (index["prac:" + p.id] = { title: p.t, href: "practice.html#p/" + p.id }));
    const courses = docs.map((d) => {
      const secs = PR.docs[d.kind] || [], nx = secs.find((x) => !st(d.kind + ":" + x.id));
      return { kicker: "Learn", name: d.title, done: secs.filter((x) => st(d.kind + ":" + x.id)).length, total: secs.length, next: nx && { t: nx.title, href: d.href + "#" + nx.id } };
    });
    const pd = prac.filter((p) => st("prac:" + p.id)).length, pn = prac.find((p) => !st("prac:" + p.id));
    const practice = { kicker: "Practice", name: "Practice", done: pd, total: prac.length, next: pn && { t: pn.t, href: "practice.html#p/" + pn.id } };
    // groups: each doc page's sidebar groups, then this part's practice categories
    const groups = [];
    docs.forEach((d) => (PR.docs[d.kind] || []).forEach((x) => {
      const name = d.title + (x.group ? " · " + x.group : "");
      let G = groups.find((g) => g.name === name);
      if (!G) { G = { name, icon: DOC_ICON[d.kind], href: d.href + "#" + x.id, keys: [] }; groups.push(G); }
      G.keys.push(d.kind + ":" + x.id);
    }));
    PR.cats.filter((c) => PR.part[c] === pt).forEach((c) => {
      const all = prac.filter((p) => p.cat === c);
      if (all.length) groups.push({ name: "Practice · " + c, icon: PR.icons[c], href: "practice.html#list", cat: c, keys: all.map((p) => "prac:" + p.id) });
    });
    groups.forEach((g) => { g.done = g.keys.filter(st).length; g.total = g.keys.length; });
    const done = courses.reduce((a, c) => a + c.done, 0) + pd, total = courses.reduce((a, c) => a + c.total, 0) + prac.length;
    const tones = ["accent", "green", "purple"];
    return { index, done, total, courses, practice, groups, help: HELP[pt],
      gauge: { title: SITE.parts[pt], done, total, unit: "done", rows: [...courses.map((c, i) => [c.name.replace(" for ML", ""), c.done, c.total, tones[i % 3]]), ["Practice", pd, prac.length, "amber"]] } };
  }

  /* ---------- everything the dashboard shows for the chosen view ---------- */
  function model(view) {
    const d = Store.load(), st = (k) => (d.items[k] || {}).status;
    const P = Object.fromEntries(Object.keys(SITE.parts).map((k) => [k, part(k, st)]));
    const parts = view === "all" ? Object.values(P) : [P[view]];
    const index = Object.assign({}, ...parts.map((p) => p.index));
    let gauge = parts[0].gauge, courses = [...parts[0].courses, parts[0].practice];
    if (view === "all") {
      const tone = { found: "green", ml: "accent", dl: "amber", llm: "purple" }, short = { found: "Basics", ml: "ML", dl: "DL", llm: "LLMs" };
      const sum = (f) => parts.reduce((a, p) => a + f(p), 0);
      gauge = { title: "Everything", done: sum((p) => p.done), total: sum((p) => p.total), unit: "done", rows: Object.entries(P).map(([k, p]) => [short[k], p.done, p.total, tone[k]]) };
      const pr = PR.prac, pn = pr.find((p) => !st("prac:" + p.id));
      courses = [...parts.flatMap((p) => p.courses), { kicker: "Practice", name: "Practice", done: pr.filter((p) => st("prac:" + p.id)).length, total: pr.length, next: pn && { t: pn.t, href: "practice.html#p/" + pn.id } }];
    }
    const m = {
      gauge, courses,
      groups: parts.flatMap((p) => p.groups),
      help: [...parts.flatMap((p) => p.help),
        "<strong>Practice</strong>: implement softmax, k-means, backprop, attention, a ReAct loop… yourself. <strong>Try it yourself</strong> checks your code against tests.",
        "Mark sections <strong>learned</strong> and problems <strong>solved</strong>; they come back in <strong>Revision</strong> after 1, 3, 7, 14, 30 and 60 days.",
        "Write <strong>notes</strong> on any section or problem: the key insight or the mistake you made."],
    };
    // review: what is due, in the order it fell due
    const due = Object.entries(d.items).filter(([k, it]) => index[k] && isDue(it)).sort(([, a], [, z]) => (a.next || 0) - (z.next || 0)).map(([k]) => ({ k, ...index[k] }));
    // weak spots: groups where things are marked "needs revision" or were forgotten in reviews
    const weak = m.groups.map((g) => {
      const its = g.keys.map((k) => d.items[k] || {}), revise = its.filter((it) => it.status === "revise").length, lapses = its.reduce((a, it) => a + (it.lapses || 0), 0);
      return { ...g, revise, lapses, score: revise * 2 + lapses };
    }).filter((g) => g.score > 0).sort((a, z) => z.score - a.score).slice(0, 4);
    return { ...m, act: d.activity, due, weak, streak: Store.streak(), best: Store.bestStreak(), todayN: d.activity[today()] || 0 };
  }

  /* ---------- pieces ---------- */
  const greet = () => { const h = new Date().getHours(); return h < 5 || h >= 22 ? "Late-night grind" : h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening"; };
  const ring = (frac, cls) => `<circle class="${cls}" cx="29" cy="29" r="24" pathLength="100" stroke-dasharray="${Math.max(0, Math.min(100, frac * 100)).toFixed(1)} 100"/>`;
  function gauge(G) {
    const tone = { green: "var(--green)", amber: "var(--amber)", red: "var(--red)", accent: "var(--accent)" };
    let off = 0;
    const arcs = G.rows.filter((r) => r[1] > 0).map(([, n, , col]) => { const len = (n / G.total) * 75, s = `<circle class="g-arc" cx="60" cy="60" r="48" pathLength="100" style="stroke:${tone[col]};stroke-dasharray:${len.toFixed(2)} 100;stroke-dashoffset:${(-off).toFixed(2)}"/>`; off += len; return s; }).join("");
    return `<section class="card dash-card"><div class="dash-h">${esc(G.title)}</div>
      <div class="dash-gauge-row"><div class="rm-diff">${G.rows.map(([l, n, t, col]) => `<div><span class="text-${col}">${l}</span><b>${n}<small>/${t}</small></b></div>`).join("")}</div>
      <div class="rm-gauge"><svg viewBox="0 0 120 120" aria-hidden="true"><circle class="g-track" cx="60" cy="60" r="48" pathLength="100"/>${arcs}</svg>
        <div class="rm-gauge-t"><b class="stat">${G.done}</b><span>/${G.total} ${G.unit}</span></div></div></div></section>`;
  }
  function heat(act) {
    const thisYear = new Date().getFullYear();
    const years = [...new Set(Object.keys(act).filter((k) => act[k]).map((k) => +k.slice(0, 4)).concat(thisYear))].sort((a, z) => z - a);
    if (!years.includes(year)) year = thisYear;
    const first = new Date(year, 0, 1), start = new Date(year, 0, 1 - first.getDay()), tk = today();
    const cell = 11, step = 14, left = 28, top = 18;
    let s = "", total = 0, days = 0, months = "", lastM = -1;
    for (let w = 0; w < 54; w++) for (let dw = 0; dw < 7; dw++) {
      const t = new Date(start.getFullYear(), start.getMonth(), start.getDate() + w * 7 + dw);
      if (t.getFullYear() !== year) continue;
      const key = today(t.getTime()), v = act[key] || 0;
      if (v) { total += v; days++; }
      const lvl = v === 0 ? 0 : v < 3 ? 1 : v < 6 ? 2 : v < 10 ? 3 : 4;
      s += `<rect class="hm-c l${lvl}${key > tk ? " fut" : ""}${key === tk ? " now" : ""}" x="${left + w * step}" y="${top + dw * step}" width="${cell}" height="${cell}" rx="2.5"><title>${t.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}: ${v} action${v === 1 ? "" : "s"}</title></rect>`;
      if (t.getDate() === 1 && t.getMonth() !== lastM) { lastM = t.getMonth(); months += `<text class="hm-m" x="${left + w * step}" y="11">${t.toLocaleString(undefined, { month: "short" })}</text>`; }
    }
    ["Mon", "Wed", "Fri"].forEach((l, i) => (s += `<text class="hm-m" x="0" y="${top + (1 + i * 2) * step + 9}">${l}</text>`));
    return { svg: `<svg class="hm-svg" viewBox="0 0 ${left + 54 * step} ${top + 7 * step}" preserveAspectRatio="xMinYMin meet" role="img" aria-label="Activity in ${year}">${months}${s}</svg>`, total, days, years };
  }

  /* ---------- the page ---------- */
  function render() {
    const M = model(cur), H = heat(M.act), goalFrac = Math.min(1, M.todayN / GOAL);
    const due = M.due;
    main.innerHTML = `
      <div class="dash-top">
        <div class="view-switch dash-switch" role="tablist" aria-label="Show">${Object.keys(SITES).map((k) => `<button type="button" role="tab" aria-selected="${k === cur}" class="${k === cur ? "on" : ""}" data-site="${k}">${ICON(SITES[k].mascot)}${SITES[k].name}</button>`).join("")}</div>
        <div class="dash-tools">
          <button type="button" class="icon-btn" data-pop="help" aria-label="How to use this site" title="How to use this site">${ICON(I.help)}</button>
          <button type="button" class="icon-btn" data-pop="gear" aria-label="Backup, import or reset progress" title="Backup / import / reset">${ICON(I.gear)}</button>
        </div>
      </div>
      <div class="dash">
        <aside class="dash-left">
          <section class="card dash-card dash-hello">
            <span class="dash-mascot">${ICON(SITES[cur].mascot)}</span>
            <div class="flex-1"><div class="small muted">${greet()}</div><div class="dash-hello-t">${M.todayN >= GOAL ? "Goal done for today. Nice work!" : "Ready for today's session?"}</div></div>
            <div class="dash-goal" title="Today: ${M.todayN} of ${GOAL} actions"><svg viewBox="0 0 58 58" aria-hidden="true">${ring(1, "goal-track")}${goalFrac > 0 ? ring(goalFrac, "goal-arc") : ""}</svg>
              <div><b>${Math.min(M.todayN, GOAL)}/${GOAL}</b><span>today</span></div></div>
          </section>
          <div class="dash-pair">
            <section class="card dash-card"><div class="small muted">Current streak</div><div class="dash-big">${ICON(I.flame, "cat-ico ico-amber")}<span class="stat">${M.streak} day${M.streak === 1 ? "" : "s"}</span></div></section>
            <section class="card dash-card"><div class="small muted">Best streak</div><div class="dash-big">${ICON(I.trophy, "cat-ico ico-amber")}<span class="stat">${M.best} day${M.best === 1 ? "" : "s"}</span></div></section>
          </div>
          ${gauge(M.gauge)}
          <section class="card dash-card">
            <div class="row-between"><div class="dash-h">Review</div>${due.length ? `<span class="badge purple">${due.length} due today</span>` : `<span class="badge easy">all caught up</span>`}</div>
            <div class="small muted dash-due">${due.length ? due.slice(0, 3).map((x) => esc(x.title)).join(" · ") + (due.length > 3 ? " …" : "") : "Nothing to review right now. Learn something new below."}</div>
            ${due.length ? `<a class="btn primary dash-cta" href="revision.html">Start review ${ICON(I.arrow)}</a>` : (M.courses.find((c) => c.next) ? `<a class="btn dash-cta" href="${M.courses.find((c) => c.next).next.href}">Continue learning ${ICON(I.arrow)}</a>` : `<a class="btn dash-cta" href="practice.html#roadmap">Open the practice roadmap ${ICON(I.arrow)}</a>`)}
          </section>
        </aside>

        <div class="dash-main">
          <section class="card dash-card hm" id="hm">
            <div class="hm-head"><b>${ICON(I.pulse, "cat-ico ico-green")}${H.total} action${H.total === 1 ? "" : "s"} in ${year}</b>
              <span class="hm-stats"><span>Active days <b>${H.days}</b></span><span>Current streak <b>${M.streak}</b></span><span>Max streak <b>${M.best}</b></span>
              <select id="hmYear" aria-label="Year">${H.years.map((y) => `<option ${y === year ? "selected" : ""}>${y}</option>`).join("")}</select></span></div>
            ${H.svg}
            <div class="hm-foot"><span>Each square is a day; greener means you did more that day.</span><span class="hm-legend">Less ${[0, 1, 2, 3, 4].map((l) => `<i class="l${l}"></i>`).join("")} More</span></div>
          </section>

          <section>
            <h2 class="dash-h2">Your progress</h2>
            <div class="dash-courses">${M.courses.map((c) => `<a class="card dash-course" href="${c.next ? c.next.href : "#"}">
              <span class="dash-kicker">${c.kicker}</span>
              <span class="row-between"><b class="dash-course-t">${c.name}</b><span class="ct-n">${c.done} / ${c.total}</span></span>
              <span class="ct-bar"><span style="width: ${c.total ? ((c.done / c.total) * 100).toFixed(1) : 0}%"></span></span>
              <span class="small muted">${c.next ? `Continue: <span class="text-accent fw-700">${esc(c.next.t)}</span>` : icon("party") + " All done"}</span></a>`).join("")}</div>
          </section>

          ${M.weak.length ? `<section class="card dash-card">
            <div class="dash-h">${ICON(I.target, "cat-ico ico-red")}Weak spots</div>
            <div class="dash-chips">${M.weak.map((g) => `<a class="dash-chip" href="${g.href}"${g.cat ? ` data-cat="${esc(g.cat)}" data-s="${g.revise ? "revise" : ""}"` : ""}><b>${esc(g.name)}</b><span class="muted">${[g.revise ? g.revise + " need revision" : "", g.lapses ? "forgot " + g.lapses + "×" : ""].filter(Boolean).join(" · ")}</span><span class="text-accent fw-700">Practice</span></a>`).join("")}</div>
          </section>` : ""}

          <section>
            <button type="button" class="dash-more" data-cats aria-expanded="${showCats}"><span>${showCats ? "Hide" : "Show all"} ${M.groups.length} groups and practice topics</span>${ICON(I.chev)}</button>
            ${showCats ? `<div class="cat-grid dash-cats">${M.groups.map((g) => `<a class="cat-tile${g.done === g.total ? " done" : ""}" href="${g.href}"${g.cat ? ` data-cat="${esc(g.cat)}"` : ""}><span class="cat-badge">${ICON(g.icon)}</span>
              <span class="ct-body"><span class="ct-top"><b>${esc(g.name)}</b><span class="ct-n">${g.done === g.total ? icon("check") + " " : ""}${g.done}/${g.total}</span></span><span class="ct-bar"><span style="width: ${g.total ? ((g.done / g.total) * 100).toFixed(1) : 0}%"></span></span></span></a>`).join("")}</div>` : ""}
          </section>
        </div>
      </div>
      <div class="dash-pop card" id="dashPop" hidden></div>`;
    const yr = $("#hmYear");
    if (yr) yr.onchange = () => { year = +yr.value; render(); };
  }

  /* ---------- popovers: how-to and backup ---------- */
  function openPop(kind, btn) {
    const el = $("#dashPop");
    if (pop === kind) { closePop(); return; }
    pop = kind;
    el.innerHTML = kind === "help"
      ? `<div class="dash-h">How to use this site</div><ol class="clean">${model(cur).help.map((x) => `<li>${x}</li>`).join("")}</ol><p class="small muted">Press <kbd>Ctrl</kbd> <kbd>K</kbd> to find anything, or <kbd>?</kbd> for all keyboard shortcuts.</p>`
      : `<div class="dash-h">Your progress</div><p class="small muted">Your progress, notes and bookmarks are saved only in this browser (separately from the DSA site). Export it now and then, and import it on another device.</p>
         <div class="btn-row"><button type="button" class="btn primary" data-do="export">${icon("download")} Export</button><label class="btn">${icon("upload")} Import<input type="file" data-do="import" accept=".json,application/json" hidden></label><button type="button" class="btn on-red" data-do="reset">Reset</button></div>`;
    const r = btn.getBoundingClientRect(), mr = main.getBoundingClientRect();
    el.style.top = r.bottom - mr.top + 8 + "px"; el.style.right = mr.right - r.right + "px";
    el.hidden = false;
  }
  function closePop() { pop = null; const el = $("#dashPop"); if (el) el.hidden = true; }
  function resetAll() {
    if (!confirm("Delete ALL progress, bookmarks and notes for the AI Study Hub in this browser? (The DSA site is not touched.) Export first if unsure.")) return;
    Store.reset(); toast("Progress reset"); closePop(); render();
  }

  main.addEventListener("click", (e) => {
    const sw = e.target.closest("[data-site]");
    if (sw && sw.dataset.site !== cur) { cur = sw.dataset.site; showCats = false; try { sessionStorage.setItem(lsKey("dash:view"), cur); } catch (x) {} closePop(); render(); if (window.FX && FX.rise) FX.rise(); return; }
    const p = e.target.closest("[data-pop]");
    if (p) { e.stopPropagation(); openPop(p.dataset.pop, p); return; }
    if (e.target.closest("[data-cats]")) { showCats = !showCats; render(); return; }
    const doIt = e.target.closest("[data-do]");
    if (doIt && doIt.dataset.do === "export") Store.exportJSON();
    if (doIt && doIt.dataset.do === "reset") resetAll();
    const cat = e.target.closest("[data-cat]");          // open the Practice list already filtered to that category
    if (cat) { try { sessionStorage.setItem(lsKey("prac:filters"), JSON.stringify({ cat: cat.dataset.cat, s: cat.dataset.s || "" })); } catch (x) {} }
  });
  main.addEventListener("change", (e) => { if (e.target.matches('[data-do="import"]') && e.target.files[0]) { Store.importJSON(e.target.files[0], render); closePop(); } });
  document.addEventListener("click", (e) => { if (pop && !e.target.closest("#dashPop")) closePop(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && pop) closePop(); });
  render();
})();
