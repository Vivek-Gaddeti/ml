/* =====================================================================
   viz.js — step-by-step visualizers (engine from the DSA site, plus sliders and a small plotting helper)
   Each visualizer: { title, input?, help?, params?, gen(inputString, P) -> steps[], kind? }
   kind "rows"  : step = { rows:[{label, vals, cls:{i:'hl'|'inwin'|'done'|'dim'|'bad'}, ptr:{i:'L'}, arrows:[]}], side:{}, msg }
   kind "grid"  : step = { grid, cls:{'r,c':'hl'|'done'|'water'|'land'}, head:{rows:[],cols:[]}, side, msg }
   kind "svg"   : step = { svg: '<svg…>', side, msg }        (any step with .svg is drawn as SVG)
   params       : [{ id, label, min, max, step, value, fmt? }] → sliders; P = { id: number } is passed to gen.
                  Moving a slider re-runs gen and keeps you on the same step (slide: "end" jumps to the last step instead).
   A visualizer that returns a single step hides Play / Prev / Next (a "live" picture driven by its sliders).
   Plot (window.Plot) draws small charts as SVG strings: Plot.chart({ x:[a,b], y:[c,d] }, [layers…]).
   ===================================================================== */
(function () {
  const nums = (s) => s.split(/[\s,]+/).filter(Boolean).map(Number);
  const split2 = (s) => { const [a, b] = s.split("|"); return [a.trim(), (b || "").trim()]; };
  const range = (a, b) => { const r = []; for (let i = a; i <= b; i++) r.push(i); return r; };
  const C = (pairs) => { const o = {}; pairs.forEach(([idxs, c]) => idxs.forEach((i) => (o[i] = c))); return o; };
  const r3 = (v) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : String(+v.toFixed(3)));

  /* =================================================================== Plot: tiny SVG charts
     Plot.chart(o, layers): o = { W, H, x:[min,max], y:[min,max], xl, yl (axis labels), grid (default true), ticks }
     layers are functions (S) => svg, where S.sx / S.sy map data → pixels. Ready-made layers:
       Plot.fn(f, cls)            curve y = f(x) (gaps where it leaves the box)
       Plot.line(pts, cls)        polyline through [[x,y],…]
       Plot.pts(pts, cls, r)      dots; a point can carry its own class: [x, y, "a"]
       Plot.seg(x1,y1,x2,y2,cls,arrow)   Plot.dot(x,y,cls,r)   Plot.text(x,y,s,cls,anchor)
       Plot.vline(x,cls)   Plot.hline(y,cls)   Plot.regions(fn(x,y) -> class, n)  (shaded decision regions)
     Colours come from CSS classes (.pl-c1 … .pl-c5, .pl-a / .pl-b for two classes), so light and dark mode both work. */
  function niceTicks(a, b, n = 5) {
    const span = b - a, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    const step = (f >= 5 ? 10 : f >= 2 ? 5 : f >= 1 ? 2 : 1) * mag, out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const Plot = {
    chart(o, layers = []) {
      const W = o.W || 520, H = o.H || 300, P = Object.assign({ l: 44, r: 14, t: 14, b: o.xl ? 40 : 28 }, o.pad || {});
      const [x0, x1] = o.x, [y0, y1] = o.y;
      const sx = (v) => P.l + ((v - x0) / (x1 - x0)) * (W - P.l - P.r), sy = (v) => H - P.b - ((v - y0) / (y1 - y0)) * (H - P.t - P.b);
      const S = { sx, sy, x0, x1, y0, y1, W, H, P };
      let g = "";
      if (o.grid !== false) {
        niceTicks(x0, x1, o.ticks || 6).forEach((t) => (g += `<line class="pl-grid" x1="${sx(t)}" y1="${P.t}" x2="${sx(t)}" y2="${H - P.b}"/><text class="pl-tick" x="${sx(t)}" y="${H - P.b + 15}" text-anchor="middle">${+t.toFixed(4)}</text>`));
        niceTicks(y0, y1, o.yticks || 5).forEach((t) => (g += `<line class="pl-grid" x1="${P.l}" y1="${sy(t)}" x2="${W - P.r}" y2="${sy(t)}"/><text class="pl-tick" x="${P.l - 6}" y="${sy(t) + 4}" text-anchor="end">${+t.toFixed(4)}</text>`));
      }
      g += `<rect class="pl-frame" x="${P.l}" y="${P.t}" width="${W - P.l - P.r}" height="${H - P.t - P.b}"/>`;
      if (o.xl) g += `<text class="pl-lbl" x="${(P.l + W - P.r) / 2}" y="${H - 6}" text-anchor="middle">${esc(o.xl)}</text>`;
      if (o.yl) g += `<text class="pl-lbl" x="12" y="${(P.t + H - P.b) / 2}" text-anchor="middle" transform="rotate(-90 12 ${(P.t + H - P.b) / 2})">${esc(o.yl)}</text>`;
      const id = "pc" + Math.random().toString(36).slice(2, 8);
      const body = layers.map((L) => (typeof L === "function" ? L(S) : L || "")).join("");
      return `<svg class="plot" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">
        <defs><clipPath id="${id}"><rect x="${P.l}" y="${P.t}" width="${W - P.l - P.r}" height="${H - P.t - P.b}"/></clipPath>
        <marker id="${id}a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="pl-arrowhead"/></marker></defs>
        ${g}<g clip-path="url(#${id})" data-arrow="${id}a">${body.replace(/url\(#ARROW\)/g, `url(#${id}a)`)}</g></svg>`;
    },
    fn: (f, cls = "pl-c1", n = 240) => (S) => {
      let d = "", pen = false;
      for (let i = 0; i <= n; i++) {
        const x = S.x0 + ((S.x1 - S.x0) * i) / n, y = f(x);
        if (!isFinite(y) || y < S.y0 - (S.y1 - S.y0) || y > S.y1 + (S.y1 - S.y0)) { pen = false; continue; }
        d += (pen ? "L" : "M") + S.sx(x).toFixed(1) + "," + S.sy(y).toFixed(1); pen = true;
      }
      return `<path class="pl-line ${cls}" d="${d}"/>`;
    },
    line: (pts, cls = "pl-c1") => (S) => pts.length ? `<path class="pl-line ${cls}" d="${pts.map(([x, y], i) => (i ? "L" : "M") + S.sx(x).toFixed(1) + "," + S.sy(y).toFixed(1)).join("")}"/>` : "",
    pts: (pts, cls = "pl-c1", r = 4.5) => (S) => pts.map(([x, y, c]) => `<circle class="pl-pt ${c ? "pl-" + c : cls}" cx="${S.sx(x).toFixed(1)}" cy="${S.sy(y).toFixed(1)}" r="${r}"/>`).join(""),
    dot: (x, y, cls = "pl-c1", r = 6) => (S) => `<circle class="pl-pt ${cls}" cx="${S.sx(x).toFixed(1)}" cy="${S.sy(y).toFixed(1)}" r="${r}"/>`,
    seg: (x1, y1, x2, y2, cls = "pl-c2", arrow) => (S) => `<line class="pl-line ${cls}" x1="${S.sx(x1).toFixed(1)}" y1="${S.sy(y1).toFixed(1)}" x2="${S.sx(x2).toFixed(1)}" y2="${S.sy(y2).toFixed(1)}"${arrow ? ' marker-end="url(#ARROW)"' : ""}/>`,
    text: (x, y, s, cls = "", anchor = "middle") => (S) => `<text class="pl-note ${cls}" x="${S.sx(x).toFixed(1)}" y="${S.sy(y).toFixed(1)}" text-anchor="${anchor}">${esc(s)}</text>`,
    vline: (x, cls = "pl-dash") => (S) => `<line class="pl-line ${cls}" x1="${S.sx(x)}" y1="${S.P.t}" x2="${S.sx(x)}" y2="${S.H - S.P.b}"/>`,
    hline: (y, cls = "pl-dash") => (S) => `<line class="pl-line ${cls}" x1="${S.P.l}" y1="${S.sy(y)}" x2="${S.W - S.P.r}" y2="${S.sy(y)}"/>`,
    regions: (f, n = 60) => (S) => {
      let s = "";
      const dx = (S.x1 - S.x0) / n, dy = (S.y1 - S.y0) / n;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const c = f(S.x0 + (i + 0.5) * dx, S.y0 + (j + 0.5) * dy);
        if (c) s += `<rect class="pl-reg pl-${c}" shape-rendering="crispEdges" x="${S.sx(S.x0 + i * dx).toFixed(1)}" y="${S.sy(S.y0 + (j + 1) * dy).toFixed(1)}" width="${(S.sx(dx) - S.sx(0) + 0.6).toFixed(1)}" height="${(S.sy(0) - S.sy(dy) + 0.6).toFixed(1)}"/>`;
      }
      return s;
    },
  };

  const V = {};

  /* ---------- Gradient descent on a 1-D loss: the learning rate decides everything ---------- */
  V.gd1d = {
    title: "Gradient descent on a 1-D loss",
    slide: "end",
    params: [
      { id: "lr", label: "learning rate η", min: 0.02, max: 1.05, step: 0.01, value: 0.1 },
      { id: "x0", label: "start x", min: -4, max: 4, step: 0.1, value: -3.5 },
    ],
    gen(_, P) {
      const f = (x) => x * x, g = (x) => 2 * x, lr = P.lr, S = [];
      let x = P.x0;
      const path = [[x, f(x)]];
      const frame = (msg, extra = []) => ({
        svg: Plot.chart({ x: [-4.5, 4.5], y: [-1, 20], xl: "x (the parameter)", yl: "loss f(x) = x²" }, [
          Plot.fn(f, "pl-c1"),
          Plot.line(path, "pl-c3 pl-thin"),
          Plot.pts(path.slice(0, -1), "pl-c3", 3.5),
          ...extra,
          Plot.dot(path[path.length - 1][0], Math.min(path[path.length - 1][1], 19.6), "pl-c2", 6.5),
        ]),
        side: { step: path.length - 1, x: r3(x), "loss f(x)": r3(f(x)), "gradient f′(x)": r3(g(x)) },
        msg,
      });
      const tangent = (x) => { const m = g(x), y = f(x), w = 1.2; return Plot.seg(x - w, y - m * w, x + w, y + m * w, "pl-c5 pl-thin"); };
      S.push(frame(`Start at x = ${r3(x)}. The loss is ${r3(f(x))}. The slope (gradient) there is ${r3(g(x))}: the purple tangent line. We step **against** the slope.`, [tangent(x)]));
      for (let i = 0; i < 25; i++) {
        const gx = g(x), nx = x - lr * gx;
        const note = Math.abs(nx) > 60 ? "The steps keep growing: with this learning rate gradient descent **diverges**." :
          Math.abs(nx) > Math.abs(x) ? `|x| grew from ${r3(Math.abs(x))} to ${r3(Math.abs(nx))}: the step overshot the minimum and landed higher up the other side. η is too big.` :
          Math.sign(nx) !== Math.sign(x) && x !== 0 ? "We jumped over the minimum to the other side (zig-zag), but got closer. η is on the large side." :
          Math.abs(nx - x) < 1e-3 ? "The steps are tiny now: we have **converged** to the minimum at x = 0." : "Moving downhill towards the minimum at x = 0.";
        x = nx; path.push([x, f(x)]);
        S.push(frame(`x ← x − η·f′(x) = ${r3(path[path.length - 2][0])} − ${lr}·${r3(gx)} = **${r3(x)}**. ${note}`, [tangent(x)]));
        if (Math.abs(x) > 60 || Math.abs(nx - path[path.length - 2][0]) < 1e-4) break;
      }
      S.push(frame(Math.abs(x) > 4.5 ? `Done: the loss **exploded** (x = ${r3(x)}). For f(x) = x², any η above 1.0 diverges. Try η around 0.1–0.5.` :
        Math.abs(x) < 0.01 ? `Done: converged to x ≈ ${r3(x)} in ${path.length - 1} steps. Try a tiny η (0.02) to see it crawl, or η = 0.95 to see it zig-zag.` :
        `Stopped after ${path.length - 1} steps at x = ${r3(x)}. Small η means slow progress; a bigger η gets there faster (until it's too big).`));
      return S;
    },
  };

  /* =================================================================== Renderer */
  function renderRows(st) {
    return (st.rows || []).map((row) => {
      const cells = row.vals.map((v, i) => {
        const c = row.cls && row.cls[i] ? row.cls[i] : "";
        const arrow = row.arrows && i < row.vals.length - 1 ? `<div class="viz-cell"><div class="ptr"></div><div class="box gap">${row.arrows[i] || ""}</div><div class="idx"></div></div>` : "";
        return `<div class="viz-cell"><div class="ptr">${row.ptr && row.ptr[i] != null ? esc(row.ptr[i]) : ""}</div><div class="box ${c}">${esc(v === "" ? "" : v)}</div><div class="idx">${row.idx === false ? "" : i}</div></div>${arrow}`;
      }).join("");
      return `<div class="viz-row">${row.label ? `<div class="small muted viz-label">${esc(row.label)}</div>` : ""}<div class="viz-cells">${cells}</div></div>`;
    }).join("");
  }
  function renderGrid(st) {
    const cols = st.grid[0].length;
    return `<div class="text-center"><div class="viz-grid" style="grid-template-columns:repeat(${cols},38px)">${st.grid.map((row, i) => row.map((v, j) => `<div class="box ${(st.cls && st.cls[i + "," + j]) || ""}">${esc(v)}</div>`).join("")).join("")}</div></div>`;
  }
  function renderStep(def, st) {
    let body = st.svg != null ? `<div class="text-center viz-svg">${st.svg}</div>` : def.kind === "grid" ? renderGrid(st) : renderRows(st);
    if (st.side) body += `<div class="viz-side">${Object.entries(st.side).map(([k, v]) => `<span class="chip"><span class="muted">${esc(k)}:</span> ${esc(v)}</span>`).join("")}</div>`;
    return body;
  }

  function mount(el) {
    const name = el.dataset.viz, def = V[name];
    if (!def || el.dataset.ready) return;
    el.dataset.ready = 1;
    const params = def.params || [], P = Object.fromEntries(params.map((p) => [p.id, p.value]));
    const pfmt = (p, v) => (p.fmt ? p.fmt(v) : String(+(+v).toFixed(4)));
    el.innerHTML = `<div class="viz-head"><span class="title">${icon("play")} ${esc(def.title)}</span>
        ${def.input ? `<input type="text" value="${esc(def.input)}" title="${esc(def.help || "")}" aria-label="input"><button class="btn sm" data-a="run">Run</button>` : ""}</div>
      ${params.length ? `<div class="viz-params">${params.map((p) => `<label class="viz-param"><span class="vp-l">${esc(p.label)}</span><input type="range" data-p="${p.id}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.value}"><b class="vp-v">${esc(pfmt(p, p.value))}</b></label>`).join("")}</div>` : ""}
      <div class="viz-stage"></div><div class="viz-msg"></div>
      <div class="viz-ctrl"><button class="btn sm" data-a="reset">${icon("first")}</button><button class="btn sm" data-a="prev">${icon("prev")} Prev</button><button class="btn sm primary" data-a="play">${swapLabel([icon("play") + " Play", icon("pause") + " Pause"])}</button><button class="btn sm" data-a="next">Next ${icon("play")}</button>${def.help ? `<span class="hide-sm">Input: ${esc(def.help)}</span>` : ""}<span class="step"></span></div>`;
    const stage = el.querySelector(".viz-stage"), msg = el.querySelector(".viz-msg"), stepEl = el.querySelector(".step"), inp = el.querySelector(".viz-head input"), playB = el.querySelector('[data-a="play"]'), ctrl = el.querySelector(".viz-ctrl");
    let steps = [], k = 0, timer = null;
    const draw = () => { const st = steps[k]; if (!st) return; stage.innerHTML = renderStep(def, st); msg.innerHTML = fmt(st.msg || ""); stepEl.textContent = `step ${k + 1} / ${steps.length}`; };
    const stop = () => { clearInterval(timer); timer = null; setSwap(playB, 0); };
    const run = (keep) => {
      if (!keep) stop();
      try { steps = def.gen(inp ? inp.value : "", P); } catch (e) { console.error(e); steps = [{ rows: [], msg: "Could not parse that input. Format: " + (def.help || "") }]; }
      k = keep === "end" ? steps.length - 1 : keep ? Math.min(k, steps.length - 1) : 0;
      ctrl.hidden = steps.length <= 1;
      draw();
    };
    el.addEventListener("click", (e) => {
      const a = e.target.closest("[data-a]"); if (!a) return;
      const act = a.dataset.a;
      if (act === "run") run();
      if (act === "reset") { stop(); k = 0; draw(); }
      if (act === "prev") { stop(); k = Math.max(0, k - 1); draw(); }
      if (act === "next") { stop(); k = Math.min(steps.length - 1, k + 1); draw(); }
      if (act === "play") {
        if (timer) return stop();
        if (k >= steps.length - 1) k = 0;
        setSwap(playB, 1);
        timer = setInterval(() => { if (k >= steps.length - 1) return stop(); k++; draw(); }, def.speed || 1100);
      }
    });
    // sliders: re-run on every move (one redraw per frame), staying on the same step
    let raf = 0;
    el.addEventListener("input", (e) => {
      const r = e.target.closest("[data-p]"); if (!r) return;
      const p = params.find((x) => x.id === r.dataset.p);
      P[p.id] = +r.value;
      r.parentNode.querySelector(".vp-v").textContent = pfmt(p, r.value);
      cancelAnimationFrame(raf); raf = requestAnimationFrame(() => run(def.slide === "end" ? "end" : true));
    });
    if (inp) inp.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
    run();
  }
  window.Plot = Plot;
  window.Viz = { defs: V, Plot, mountAll: (root = document) => root.querySelectorAll(".viz[data-viz]").forEach(mount), helpers: { nums, split2, range, C, r3 } };
})();
