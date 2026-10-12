/* Transformers: visualizers, diagrams and glossary terms.
   Loaded on every doc page and on Practice, after assets/viz.js. Every number shown is computed, not faked. */
(function () {
  const V = Viz.defs, Plot = Viz.Plot, { r3 } = Viz.helpers;

  /* ------------------------------------------------------------------ private helpers */
  const rng = (seed) => { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  const gauss = (r) => { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const hash = (s) => { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; };
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const f2 = (v, d = 2) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : +v).toFixed(d);
  const rmat = (r, n, m, sc, q = 0.1) => Array.from({ length: n }, () => Array.from({ length: m }, () => Math.round((gauss(r) * sc) / q) * q));
  const mm = (A, B) => A.map((a) => B[0].map((_, j) => sum(a.map((v, k) => v * B[k][j]))));
  const tr = (A) => A[0].map((_, j) => A.map((r) => r[j]));
  const sm = (v) => { const m = Math.max(...v); const e = v.map((x) => Math.exp(x - m)); const s = sum(e); return e.map((x) => x / s); };
  const smRows = (M) => M.map(sm);
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const ln = (x) => { const m = sum(x) / x.length, v = sum(x.map((t) => (t - m) ** 2)) / x.length; return x.map((t) => (t - m) / Math.sqrt(v + 1e-5)); };
  const gelu = (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
  const norm = (x) => Math.sqrt(sum(x.map((t) => t * t)));
  const HEADC = ["var(--accent)", "var(--amber)", "var(--green)", "var(--purple)", "var(--red)", "var(--cls-a)", "var(--cls-b)", "var(--muted)"];

  /* heatmap of a matrix as SVG. o: rl (row labels), cl (col labels), cell, title, mode "pos" (0..1) or "div" (-max..max), max,
     hlRow, fmt, masked(i,j), dimRows. Returns { w, h, draw(x, y) } so panels can be laid out side by side. */
  function hm(M, o = {}) {
    const cell = o.cell || 32, LW = o.rl ? 46 : 0, CH = o.cl ? 16 : 0, TH = o.title ? 20 : 0, n = M.length, m = M[0].length;
    const w = LW + m * cell, h = TH + CH + n * cell;
    const mx = o.max || Math.max(1e-9, ...M.flat().filter(isFinite).map(Math.abs));
    const fmt = o.fmt || ((v) => f2(v));
    return {
      w, h, draw(x0, y0) {
        let s = "";
        if (o.title) s += `<text x="${x0 + LW + (m * cell) / 2}" y="${y0 + 12}" text-anchor="middle" style="font-size:12px;font-weight:600;fill:var(--text)">${esc(o.title)}</text>`;
        if (o.cl) o.cl.forEach((c, j) => (s += `<text x="${x0 + LW + j * cell + cell / 2}" y="${y0 + TH + 12}" text-anchor="middle" style="font-size:10.5px;fill:var(--muted)">${esc(c)}</text>`));
        for (let i = 0; i < n; i++) {
          if (o.rl) s += `<text x="${x0 + LW - 6}" y="${y0 + TH + CH + i * cell + cell / 2 + 4}" text-anchor="end" style="font-size:11px;fill:${o.hlRow === i ? "var(--text)" : "var(--muted)"};${o.hlRow === i ? "font-weight:700" : ""}">${esc(o.rl[i])}</text>`;
          for (let j = 0; j < m; j++) {
            const v = M[i][j], bad = o.masked && o.masked(i, j);
            let fill;
            if (bad) fill = "var(--panel)";
            else if (o.mode === "div") fill = `color-mix(in srgb, ${v >= 0 ? "var(--accent)" : "var(--amber)"} ${Math.min(75, (Math.abs(v) / mx) * 75).toFixed(0)}%, var(--panel-2))`;
            else fill = `color-mix(in srgb, var(--accent) ${Math.min(75, (v / mx) * 75).toFixed(0)}%, var(--panel-2))`;
            const dim = o.dimRows && o.dimRows.includes(i) ? "opacity:.35;" : "";
            s += `<rect x="${x0 + LW + j * cell + 1}" y="${y0 + TH + CH + i * cell + 1}" width="${cell - 2}" height="${cell - 2}" rx="3" style="fill:${fill};stroke:var(--border);${dim}"/>`;
            if (!o.nonum) s += `<text x="${x0 + LW + j * cell + cell / 2}" y="${y0 + TH + CH + i * cell + cell / 2 + 4}" text-anchor="middle" style="font-size:${o.fs || 10}px;fill:${bad ? "var(--muted)" : "var(--text)"};${dim}">${bad ? (o.maskTxt || "−∞") : esc(fmt(v))}</text>`;
          }
        }
        if (o.hlRow != null) s += `<rect x="${x0 + LW}" y="${y0 + TH + CH + o.hlRow * cell}" width="${m * cell}" height="${cell}" rx="4" style="fill:none;stroke:var(--amber);stroke-width:2.5"/>`;
        return s;
      },
    };
  }
  // place panels left to right; ops are small texts between panels ("×", "=")
  function layout(items, gap = 16, pad = 10) {
    let x = pad, H = 0, s = "";
    items.forEach((it) => {
      if (typeof it === "string") { s += `<text x="${x + 6}" y="${pad + 20 + Math.max(...items.filter((q) => typeof q !== "string").map((q) => q.h)) / 2}" text-anchor="middle" style="font-size:18px;fill:var(--muted)">${esc(it)}</text>`; x += 24; return; }
      s += it.draw(x, pad); x += it.w + gap; H = Math.max(H, it.h);
    });
    const W = x - gap + pad;
    return `<svg viewBox="0 0 ${W} ${H + 2 * pad}" width="${W}" height="${H + 2 * pad}" xmlns="http://www.w3.org/2000/svg" role="img">${s}</svg>`;
  }
  const bars = (vals, lo, hi, cls, outline) => (S) => {
    const bw = (S.sx(1) - S.sx(0)) * 0.62;
    return vals.map((v, i) => {
      const c = Math.max(lo, Math.min(hi, v)), y0 = S.sy(0), y1 = S.sy(c);
      const ol = outline ? `<rect x="${(S.sx(i + 0.5) - bw / 2).toFixed(1)}" y="${Math.min(S.sy(0), S.sy(Math.max(lo, Math.min(hi, outline[i])))).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.abs(S.sy(Math.max(lo, Math.min(hi, outline[i]))) - S.sy(0)).toFixed(1)}" style="fill:none;stroke:var(--muted);stroke-dasharray:3 3"/>` : "";
      return `<rect x="${(S.sx(i + 0.5) - bw / 2).toFixed(1)}" y="${Math.min(y0, y1).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(1, Math.abs(y1 - y0)).toFixed(1)}" style="fill:${cls};fill-opacity:.8;stroke:${cls}"/>${ol}`;
    }).join("");
  };

  /* ================================================================== 1. BPE merges, step by step */
  V.tf_bpe = {
    title: "Byte-Pair Encoding: watch the merges happen",
    input: "low:5 lower:2 newest:6 widest:3",
    help: "words with counts, e.g. low:5 newest:6, or just repeat words",
    gen(input) {
      const freq = new Map();
      input.trim().split(/\s+/).filter(Boolean).slice(0, 12).forEach((t) => { const [w, c] = t.split(":"); if (w) freq.set(w.slice(0, 12), (freq.get(w.slice(0, 12)) || 0) + (+c > 0 ? Math.min(+c, 99) : 1)); });
      let words = [...freq].map(([w, c]) => ({ s: [...w].concat(["_"]), c }));
      const draw = (merged, pair) => {
        const rowH = 30, W = 520, H = words.length * rowH + 14;
        let s = "";
        words.forEach((wd, i) => {
          const y = 8 + i * rowH; s += `<text x="70" y="${y + 19}" text-anchor="end" style="font-size:12px;fill:var(--muted)">${esc(wd.s.join("").replace(/_$/, ""))} ×${wd.c}</text>`;
          let x = 82;
          wd.s.forEach((sy, k) => {
            const w = Math.max(24, 9 * sy.length + 12), isNew = merged && sy === merged, isPair = pair && k < wd.s.length - 1 && wd.s[k] === pair[0] && wd.s[k + 1] === pair[1];
            s += `<rect x="${x}" y="${y}" width="${w}" height="24" rx="5" style="fill:${isNew ? "var(--green-soft)" : isPair ? "var(--hl)" : "var(--panel-2)"};stroke:${isNew ? "var(--green)" : isPair ? "var(--amber)" : "var(--border)"};stroke-width:${isNew || isPair ? 2 : 1}"/><text x="${x + w / 2}" y="${y + 16}" text-anchor="middle" style="font-size:12.5px;font-family:var(--mono);fill:var(--text)">${esc(sy)}</text>`;
            x += w + 3;
          });
        });
        return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">${s}</svg>`;
      };
      const vocabSize = () => new Set(words.flatMap((w) => w.s)).size;
      const S = [{ svg: draw(), side: { merges: 0, "symbols so far": vocabSize() }, msg: "Start: every word is a row of single characters plus the end marker `_`. The vocabulary is just those characters. Press Next to count neighbouring pairs and merge the most frequent." }];
      for (let step = 1; step <= 14; step++) {
        const counts = new Map();
        words.forEach((w) => { for (let k = 0; k < w.s.length - 1; k++) { const key = w.s[k] + "\u0001" + w.s[k + 1]; counts.set(key, (counts.get(key) || 0) + w.c); } });
        if (!counts.size) break;
        let best = null; counts.forEach((c, k) => { if (!best || c > best[1]) best = [k, c]; });
        if (best[1] < 2 && step > 1) { S.push({ svg: draw(), side: { merges: step - 1, "symbols so far": vocabSize() }, msg: "Every remaining pair occurs only once, so more merges would just memorise single words. Stop here." }); break; }
        const pair = best[0].split("\u0001"), top = [...counts].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k, c]) => `(${k.split("\u0001").join(", ")}) ${c}`).join(" · ");
        const before = draw(null, pair);
        words = words.map((w) => { const out = []; for (let k = 0; k < w.s.length; k++) { if (k < w.s.length - 1 && w.s[k] === pair[0] && w.s[k + 1] === pair[1]) { out.push(pair[0] + pair[1]); k++; } else out.push(w.s[k]); } return { s: out, c: w.c }; });
        S.push({ svg: before, side: { merges: step - 1, "most frequent": `(${pair[0]}, ${pair[1]}) ×${best[1]}` }, msg: `**Count pairs** (weighted by word frequency). Top pairs: ${top}. The winner is **(${pair[0]}, ${pair[1]})**, highlighted in amber.` });
        S.push({ svg: draw(pair[0] + pair[1], null), side: { merges: step, "new symbol": pair[0] + pair[1], "symbols so far": vocabSize() }, msg: `**Merge #${step}**: every \`${pair[0]}\` next to a \`${pair[1]}\` is glued into the new symbol \`${pair[0] + pair[1]}\` (green) and the merge is recorded. Next round counts pairs again, now including the new symbol.` });
      }
      return S;
    },
  };

  /* ================================================================== 2. Attention step by step */
  const E_DIM = 4;
  const embedWord = (w) => { const r = rng(hash(w)); return Array.from({ length: E_DIM }, () => Math.round(gauss(r) * 9) / 10); };
  const Wq4 = rmat(rng(11), 4, 4, 0.9), Wk4 = rmat(rng(12), 4, 4, 0.9), Wv4 = rmat(rng(13), 4, 4, 0.9);
  V.tf_attention = {
    title: "Self-attention, step by step: Q·Kᵀ → scale → softmax → weighted sum of V",
    input: "the cat sat on the mat",
    help: "a short sentence (up to 8 words)",
    params: [{ id: "q", label: "look at token #", min: 1, max: 8, step: 1, value: 3 }],
    gen(input, P) {
      const toks = input.trim().split(/\s+/).filter(Boolean).slice(0, 8);
      if (toks.length < 2) return [{ svg: "", msg: "Type at least two words." }];
      const T = toks.length, qi = Math.min(P.q, T) - 1, d = E_DIM;
      const X = toks.map(embedWord), Q = mm(X, Wq4), K = mm(X, Wk4), Vv = mm(X, Wv4);
      const S0 = mm(Q, tr(K)), S1 = S0.map((r) => r.map((v) => v / Math.sqrt(d))), A = smRows(S1), O = mm(A, Vv);
      const dl = ["d₁", "d₂", "d₃", "d₄"], cell = T > 6 ? 30 : 34;
      const lab = toks.map((t, i) => t);
      const st = (items) => layout(items, 18);
      const S = [];
      S.push({ svg: st([hm(X, { rl: lab, cl: dl, title: "X: one row per token", mode: "div", cell })]), side: { tokens: T, "d (width)": d }, msg: `Each token starts as a vector (here only ${d} numbers; real models use thousands). Same word, same row: *the* appears twice and gets the same vector. Self-attention will let every row look at all the others.` });
      S.push({ svg: st([hm(Q, { rl: lab, cl: dl, title: "Q = X·W_Q", mode: "div", cell, hlRow: qi }), hm(K, { cl: dl, title: "K = X·W_K", mode: "div", cell, hlRow: qi }), hm(Vv, { cl: dl, title: "V = X·W_V", mode: "div", cell, hlRow: qi })], 14), side: { tokens: T }, msg: `Three learned matrices turn every token into a **query** (what am I looking for?), a **key** (what do I offer?) and a **value** (what do I hand over if chosen?). The amber row is "${toks[qi]}".` });
      S.push({ svg: st([hm(S0, { rl: lab, cl: lab, title: "scores = Q·Kᵀ", mode: "div", cell, hlRow: qi })]), side: { "score[q,k]": `q·k = ${f2(S0[qi][qi])}` }, msg: `Row *i*, column *j* is the dot product of query *i* with key *j*: how well token *i*'s question matches token *j*'s offer. Row "${toks[qi]}" is highlighted. Bigger = better match.` });
      S.push({ svg: st([hm(S1, { rl: lab, cl: lab, title: "scores / √d", mode: "div", cell, hlRow: qi })]), side: { "√d": r3(Math.sqrt(d)) }, msg: `Divide by $\\sqrt{d_k}=\\sqrt{${d}}=${r3(Math.sqrt(d))}$. Without this the dot products grow with the width, the softmax saturates to one-hot and gradients vanish. See [scaled dot-product attention](transformers.html#scaled-dot-product).` });
      S.push({ svg: st([hm(A, { rl: lab, cl: lab, title: "A = softmax(rows)", mode: "pos", max: 1, cell, hlRow: qi })]), side: { "row sum": r3(sum(A[qi])), "strongest": toks[A[qi].indexOf(Math.max(...A[qi]))] }, msg: `Softmax turns each **row** into weights that are positive and sum to 1. Token "${toks[qi]}" pays most attention to **${toks[A[qi].indexOf(Math.max(...A[qi]))]}** (${f2(Math.max(...A[qi]))}). With random weights the pattern is arbitrary; training makes it meaningful.` });
      const parts = A[qi].map((a, j) => `${f2(a)}·v(${toks[j]})`).join(" + ");
      S.push({ svg: st([hm(A, { rl: lab, cl: lab, title: "A", mode: "pos", max: 1, cell, hlRow: qi }), "×", hm(Vv, { cl: dl, title: "V", mode: "div", cell }), "=", hm(O, { cl: dl, title: "output", mode: "div", cell, hlRow: qi })], 10), side: { "out[q]": "[" + O[qi].map((v) => f2(v)).join(", ") + "]" }, msg: `Each output row is a **weighted average of the value rows**: ${T <= 5 ? `"${toks[qi]}" gets ${parts}` : `row "${toks[qi]}" of A mixes all the rows of V`}. The result is a new vector for "${toks[qi]}" that now contains information from the tokens it attended to.` });
      return S;
    },
  };

  /* ================================================================== 3. Causal / padding mask */
  V.tf_mask = {
    title: "Masks: which tokens may look at which",
    params: [
      { id: "T", label: "tokens", min: 3, max: 8, step: 1, value: 5 },
      { id: "causal", label: "causal mask", min: 0, max: 1, step: 1, value: 1, fmt: (v) => (+v ? "on" : "off") },
      { id: "pad", label: "padding tokens", min: 0, max: 3, step: 1, value: 0 },
    ],
    gen(_, P) {
      const T = P.T, pad = Math.min(P.pad, T - 1), r = rng(5), cell = T > 6 ? 30 : 34;
      const raw = Array.from({ length: T }, () => Array.from({ length: T }, () => Math.round(gauss(r) * 12) / 10));
      const toks = Array.from({ length: T }, (_, i) => (i >= T - pad ? "<pad>" : "t" + (i + 1)));
      const masked = (i, j) => (P.causal && j > i) || j >= T - pad;
      const A = raw.map((row, i) => { const e = row.map((v, j) => (masked(i, j) ? 0 : Math.exp(v))); const s = sum(e); return e.map((x) => (s ? x / s : 0)); });
      const dimRows = toks.map((t, i) => (i >= T - pad ? i : -1)).filter((i) => i >= 0);
      const svg = layout([
        hm(raw, { rl: toks, cl: toks, title: "scores + mask (−∞)", mode: "div", cell, masked, dimRows }),
        "→",
        hm(A, { cl: toks, title: "softmax = attention weights", mode: "pos", max: 1, cell, dimRows, maskTxt: "0", masked: (i, j) => masked(i, j) }),
      ], 12);
      const msgs = [];
      if (P.causal) msgs.push("With the **causal mask** the upper triangle is set to −∞ before the softmax, so it becomes exactly 0: token *i* sees only tokens $\\le i$. Row 1 can only look at itself (weight 1.0). This is what lets a GPT train on every position of a sentence at once without cheating.");
      else msgs.push("No causal mask: every token may look at every other token (the **bidirectional** setting of BERT-style encoders).");
      if (pad) msgs.push(`The last ${pad} column(s) are \`<pad>\` filler: they are masked as **keys** so nobody attends to them. Their own rows (faded) are computed but ignored by the loss.`);
      return [{ svg, side: { "allowed pairs": sum(Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) => (masked(i, j) ? 0 : 1))).map(sum)), "of": T * T }, msg: msgs.join(" ") }];
    },
  };

  /* ================================================================== 4. Sinusoidal positional encoding */
  const pe = (pos, j, d) => { const i = Math.floor(j / 2), a = pos / Math.pow(10000, (2 * i) / d); return j % 2 === 0 ? Math.sin(a) : Math.cos(a); };
  V.tf_sinusoid = {
    title: "Sinusoidal positional encoding: a clock for every pair of dimensions",
    params: [
      { id: "d", label: "model width d", min: 8, max: 64, step: 8, value: 32 },
      { id: "j", label: "highlight dimension", min: 0, max: 63, step: 1, value: 0 },
    ],
    gen(_, P) {
      const d = P.d, j = Math.min(P.j, d - 1), T = 32, cw = Math.min(14, 300 / d), rh = 7;
      let hmSvg = "";
      for (let p = 0; p < T; p++) for (let k = 0; k < d; k++) {
        const v = pe(p, k, d);
        hmSvg += `<rect x="${(34 + k * cw).toFixed(1)}" y="${28 + p * rh}" width="${cw.toFixed(1)}" height="${rh}" style="fill:color-mix(in srgb, ${v >= 0 ? "var(--accent)" : "var(--amber)"} ${(Math.abs(v) * 80).toFixed(0)}%, var(--panel-2))"/>`;
      }
      const W = 34 + d * cw + 10, H = 28 + T * rh + 24;
      hmSvg += `<rect x="${34 + j * cw}" y="26" width="${cw}" height="${T * rh + 4}" style="fill:none;stroke:var(--text);stroke-width:1.5"/>`;
      hmSvg += `<text x="${34 + (d * cw) / 2}" y="14" text-anchor="middle" style="font-size:12px;font-weight:600;fill:var(--text)">PE[pos, dim]</text><text x="${34 + (d * cw) / 2}" y="${H - 4}" text-anchor="middle" style="font-size:11px;fill:var(--muted)">dimension →</text><text x="28" y="${28 + (T * rh) / 2}" text-anchor="end" style="font-size:11px;fill:var(--muted)">pos ↓</text>`;
      const heat = `<svg viewBox="0 0 ${W.toFixed(0)} ${H}" width="${W.toFixed(0)}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">${hmSvg}</svg>`;
      const pts = []; for (let p = 0; p <= 100; p += 0.5) pts.push([p, pe(p, j, d)]);
      const wl = (2 * Math.PI) / (1 / Math.pow(10000, (2 * Math.floor(j / 2)) / d));
      const chart = Plot.chart({ x: [0, 100], y: [-1.1, 1.1], W: 330, H: 230, xl: "position", yl: `dim ${j}` }, [Plot.line(pts, "pl-c1"), Plot.hline(0, "pl-dash")]);
      return [{ svg: `<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:10px;align-items:flex-start">${heat}${chart}</div>`, side: { d, "dimension": j + " (" + (j % 2 ? "cos" : "sin") + ")", wavelength: wl > 1e5 ? wl.toExponential(1) : f2(wl, 1) + " positions" },
        msg: `Each row is the position vector of one token. Left columns (small dimension numbers) are fast waves, right columns are slow ones, like the seconds hand versus the hour hand. The line plot shows column **${j}** across positions 0 to 100: a ${j % 2 ? "cosine" : "sine"} with wavelength ${f2(wl, 1)}. Together, the columns give every position a unique fingerprint.` }];
    },
  };

  /* ================================================================== 5. Multi-head attention shapes */
  V.tf_multihead = {
    title: "Multi-head attention: split, attend in parallel, concatenate",
    params: [{ id: "lh", label: "heads h", min: 0, max: 3, step: 1, value: 2, fmt: (v) => String(Math.pow(2, +v)) }],
    gen(_, P) {
      const h = Math.pow(2, P.lh), T = 4, d = 8, dk = d / h, r = rng(21), cell = 30;
      const X = rmat(r, T, d, 0.8, 0.1), Wq = rmat(r, d, d, 0.45), Wk = rmat(r, d, d, 0.45), Wv = rmat(r, d, d, 0.45), Wo = rmat(r, d, d, 0.45);
      const Q = mm(X, Wq), K = mm(X, Wk), Vv = mm(X, Wv);
      const heads = Array.from({ length: h }, (_, a) => {
        const sl = (M) => M.map((row) => row.slice(a * dk, (a + 1) * dk));
        const q = sl(Q), k = sl(K), v = sl(Vv), A = smRows(mm(q, tr(k)).map((row) => row.map((x) => x / Math.sqrt(dk))));
        return { A, out: mm(A, v) };
      });
      const cat = Array.from({ length: T }, (_, i) => heads.flatMap((hd) => hd.out[i]));
      const Y = mm(cat, Wo), tok = ["t1", "t2", "t3", "t4"];
      const grid = (M, title, color, x0 = 10, y0 = 24, c = 26) => {
        let s = `<text x="${x0 + (M[0].length * c) / 2}" y="${y0 - 8}" text-anchor="middle" style="font-size:12px;font-weight:600;fill:var(--text)">${esc(title)}</text>`;
        M.forEach((row, i) => row.forEach((v, j) => (s += `<rect x="${x0 + j * c}" y="${y0 + i * c}" width="${c - 2}" height="${c - 2}" rx="3" style="fill:color-mix(in srgb, ${color(j)} ${Math.min(70, 15 + Math.abs(v) * 35)}%, var(--panel-2));stroke:var(--border)"/>`)));
        return s;
      };
      const wrap = (W, H, s) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">${s}</svg>`;
      const hc = (j) => HEADC[Math.floor(j / dk) % 8];
      const S = [];
      S.push({ svg: wrap(280, 150, grid(X, `X: shape (${T}, ${d})`, () => "var(--accent)")), side: { shape: `(${T}, ${d})`, heads: h, "d_k = d/h": dk }, msg: `Input: ${T} tokens of width $d_{model}=${d}$. We will use **h = ${h}** head(s), each working on ${dk} dimensions. Change h with the slider.` });
      S.push({ svg: wrap(640, 150, grid(Q, "Q = X·W_Q", () => "var(--accent)", 10) + grid(K, "K = X·W_K", () => "var(--accent)", 230) + grid(Vv, "V = X·W_V", () => "var(--accent)", 450)), side: { "each": `(${T}, ${d})` }, msg: "One big matrix multiplication creates Q, K and V for **all heads at once**. Each is still (tokens, d_model)." });
      S.push({ svg: wrap(640, 170, grid(Q, `Q columns coloured by head`, hc, 10) + grid(K, "K", hc, 230) + grid(Vv, "V", hc, 450)), side: { reshape: `(${T}, ${d}) → (${T}, ${h}, ${dk}) → (${h}, ${T}, ${dk})` }, msg: `**Split**: the ${d} columns are cut into ${h} group(s) of ${dk}. Each colour is one head. In code this is a reshape to $(T, h, d_k)$ followed by a transpose to $(h, T, d_k)$ so that all heads become a batch dimension. No new numbers are created, they are just regrouped.` });
      const per = 4 * 18 + 24;
      let sv = ""; heads.forEach((hd, a) => { const col = HEADC[a % 8], x0 = 10 + (a % 4) * per, y0 = 26 + Math.floor(a / 4) * 112; sv += `<text x="${x0 + 36}" y="${y0 - 10}" text-anchor="middle" style="font-size:11.5px;font-weight:600;fill:${col}">head ${a + 1}</text>`; hd.A.forEach((row, i) => row.forEach((v, j) => (sv += `<rect x="${x0 + j * 18}" y="${y0 + i * 18}" width="16" height="16" rx="2" style="fill:color-mix(in srgb, ${col} ${(v * 85).toFixed(0)}%, var(--panel-2));stroke:var(--border)"/>`))); });
      S.push({ svg: wrap(Math.min(h, 4) * per + 10, h > 4 ? 250 : 130, sv), side: { "attention maps": `${h} × (${T}, ${T})` }, msg: `Every head runs its own scaled dot-product attention on its slice: scores $(${T},${T})$ → softmax → weights times V. The ${h} map(s) differ because each head has its own Q and K slice, so heads can specialise (e.g. look at the previous token, or at a matching word).` });
      const outs = heads.map(() => 0);
      S.push({ svg: wrap(640, 150, grid(cat, `concat of heads: (${T}, ${d})`, hc, 10) + grid(Y, `Y = concat · W_O: (${T}, ${d})`, () => "var(--green)", 330)), side: { "output shape": `(${T}, ${d})` }, msg: `**Merge**: the per-head results (each ${dk} wide) are concatenated back to width ${d} (the transpose + reshape undone) and multiplied by an output matrix $W_O$, which lets the heads' findings mix. The output has the same shape as the input, so the block can be stacked.` });
      return S;
    },
  };

  /* ================================================================== 6. Residual stream through one block */
  V.tf_residual = {
    title: "The residual stream through one pre-LN transformer block",
    params: [{ id: "t", label: "show token #", min: 1, max: 3, step: 1, value: 3 }],
    gen(_, P) {
      const T = 3, d = 8, r = rng(33), ti = P.t - 1, toks = ["the", "cat", "sat"];
      const X0 = rmat(r, T, d, 1.0, 0.1), Wq = rmat(r, d, d, 0.45), Wk = rmat(r, d, d, 0.45), Wv = rmat(r, d, d, 0.45), Wo = rmat(r, d, d, 0.45), W1 = rmat(r, d, 2 * d, 0.4), W2 = rmat(r, 2 * d, d, 0.3);
      const H1 = X0.map(ln), Q = mm(H1, Wq), K = mm(H1, Wk), Vv = mm(H1, Wv);
      const Sc = mm(Q, tr(K)).map((row, i) => row.map((v, j) => (j > i ? -Infinity : v / Math.sqrt(d)))), A = smRows(Sc), att = mm(mm(A, Vv), Wo);
      const X1 = X0.map((row, i) => add(row, att[i])), H2 = X1.map(ln), M = mm(H2, W1).map((row) => row.map(gelu)), mlp = mm(M, W2), X2 = X1.map((row, i) => add(row, mlp[i]));
      const mk = (vals, color, label, outline) => Plot.chart({ x: [0, d], y: [-3, 3], W: 540, H: 210, xl: "dimension of the vector (d = 8)", yl: label, ticks: 8 }, [bars(vals, -3, 3, color, outline), Plot.hline(0, "pl-dash")]);
      const S = [], ac = "var(--accent)";
      const nm = (v) => ({ "‖vector‖": r3(norm(v)) });
      S.push({ svg: mk(X0[ti], ac, "x⁰", null), side: nm(X0[ti]), msg: `The **residual stream** for token "${toks[ti]}": its embedding plus position, ${d} numbers. Everything in the block *reads* from this vector and *adds* to it; the stream itself is never replaced.` });
      S.push({ svg: mk(H1[ti], "var(--muted)", "LN(x⁰)", null), side: nm(H1[ti]), msg: `**LayerNorm** (pre-LN: before the sublayer). Re-scales the vector to mean 0 and variance 1 (mean ${f2(sum(H1[ti]) / d)}). This normalised copy is what attention reads. The stream itself stays un-normalised.` });
      S.push({ svg: mk(att[ti], "var(--amber)", "attn out", null), side: Object.assign({ "attends to": toks.slice(0, ti + 1).map((t, j) => t + " " + f2(A[ti][j])).join(", ") }, nm(att[ti])), msg: `**Multi-head attention** (one head here, causal): "${toks[ti]}" mixes the values of ${ti === 0 ? "itself only" : "the tokens up to itself"} with weights ${toks.slice(0, ti + 1).map((t, j) => `${t} ${f2(A[ti][j])}`).join(", ")}, then projects with $W_O$. This is only a *correction* to the stream, usually smaller than the stream.` });
      S.push({ svg: mk(X1[ti], ac, "x¹ = x⁰ + attn", X0[ti]), side: nm(X1[ti]), msg: "**Residual add**: $x^1 = x^0 + \\text{attn}$. The dashed outlines are the old stream, the filled bars the new one. Mostly the same, nudged by the attention output. This addition is also the path the gradient takes straight back to the input." });
      S.push({ svg: mk(H2[ti], "var(--muted)", "LN(x¹)", null), side: nm(H2[ti]), msg: "A second **LayerNorm** prepares the vector for the feed-forward network." });
      S.push({ svg: mk(mlp[ti], "var(--green)", "MLP out", null), side: nm(mlp[ti]), msg: `**Feed-forward network (MLP)**: expand ${d} → ${2 * d} dimensions, apply GELU, shrink back to ${d}. Unlike attention, it works on each token alone, with no mixing between tokens.` });
      S.push({ svg: mk(X2[ti], ac, "x² = x¹ + MLP", X1[ti]), side: nm(X2[ti]), msg: "**Residual add** again: $x^2 = x^1 + \\text{MLP}(\\text{LN}(x^1))$. This is the block's output. Stack N such blocks and the stream picks up one attention correction and one MLP correction per layer. See [the transformer block](transformers.html#transformer-block)." });
      return S;
    },
  };

  /* ================================================================== 7. RoPE rotation */
  const RQ = [1.0, 0.5, -0.8, 0.6, 0.3, -0.9, 0.7, 0.2], RK = [0.9, -0.3, 0.4, 0.8, -0.6, 0.5, 0.2, 1.0];
  const ropeV = (x, pos) => { const out = x.slice(); for (let i = 0; i < x.length / 2; i++) { const th = Math.pow(10000, (-2 * i) / x.length), c = Math.cos(pos * th), s = Math.sin(pos * th); out[2 * i] = x[2 * i] * c - x[2 * i + 1] * s; out[2 * i + 1] = x[2 * i] * s + x[2 * i + 1] * c; } return out; };
  const dot = (a, b) => sum(a.map((v, i) => v * b[i]));
  V.tf_rope = {
    title: "RoPE: rotate queries and keys by an angle that grows with position",
    params: [
      { id: "m", label: "query position m", min: 0, max: 30, step: 1, value: 6 },
      { id: "n", label: "key position n", min: 0, max: 30, step: 1, value: 3 },
      { id: "i", label: "dimension pair", min: 0, max: 3, step: 1, value: 0 },
    ],
    gen(_, P) {
      const d = 8, i = P.i, th = Math.pow(10000, (-2 * i) / d), qp = [RQ[2 * i], RQ[2 * i + 1]], kp = [RK[2 * i], RK[2 * i + 1]];
      const rot = (v, a) => [v[0] * Math.cos(a) - v[1] * Math.sin(a), v[0] * Math.sin(a) + v[1] * Math.cos(a)];
      const qr = rot(qp, P.m * th), kr = rot(kp, P.n * th), circ = []; for (let t = 0; t <= 64; t++) circ.push([Math.cos((t / 64) * 2 * Math.PI), Math.sin((t / 64) * 2 * Math.PI)]);
      const c1 = Plot.chart({ x: [-1.4, 1.4], y: [-1.4, 1.4], W: 300, H: 300, xl: `pair ${i}: (x${2 * i}, x${2 * i + 1}), θ = ${th < 0.01 ? th.toExponential(1) : r3(th)} rad per position`, ticks: 4, yticks: 4 }, [
        Plot.line(circ, "pl-dash"),
        Plot.seg(0, 0, qp[0], qp[1], "pl-c1 pl-dash", true), Plot.seg(0, 0, kp[0], kp[1], "pl-c2 pl-dash", true),
        Plot.seg(0, 0, qr[0], qr[1], "pl-c1", true), Plot.seg(0, 0, kr[0], kr[1], "pl-c2", true),
        Plot.text(qr[0] * 1.12, qr[1] * 1.12 + 0.05, "q@" + P.m, "", "middle"), Plot.text(kr[0] * 1.12, kr[1] * 1.12 - 0.1, "k@" + P.n, "", "middle"),
      ]);
      const rel = []; for (let dlt = -30; dlt <= 30; dlt++) rel.push([dlt, dot(ropeV(RQ, dlt), ropeV(RK, 0))]);
      const ys = rel.map((p) => p[1]), lo = Math.floor(Math.min(...ys)) - 0.5, hi = Math.ceil(Math.max(...ys)) + 0.5, cur = P.m - P.n;
      const c2 = Plot.chart({ x: [-30, 30], y: [lo, hi], W: 330, H: 300, xl: "relative offset m − n", yl: "q·k (all 4 pairs)" }, [Plot.line(rel, "pl-c3"), Plot.vline(cur, "pl-dash"), Plot.dot(cur, dot(ropeV(RQ, P.m), ropeV(RK, P.n)), "pl-c4", 6)]);
      const sc = dot(ropeV(RQ, P.m), ropeV(RK, P.n)), sh = dot(ropeV(RQ, P.m + 9), ropeV(RK, P.n + 9)), plain = dot(RQ, RK);
      return [{ svg: `<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:8px">${c1}${c2}</div>`, side: { "score q@m · k@n": r3(sc), "both moved +9": r3(sh), "no rotation": r3(plain), "m − n": cur },
        msg: `Dashed arrows: the original query and key pair. Solid arrows: after rotating by (position × θ). The score is a dot product, and the angle **between** the two solid arrows is $(m-n)\\theta$, so it depends only on the offset. Check: shifting both positions by 9 gives the same score (${r3(sc)} vs ${r3(sh)}). The green curve shows the full 8-dimensional score for every offset.` }];
    },
  };

  /* ================================================================== static diagrams */
  const arr = (id) => `<defs><marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker></defs>`;
  const wrapD = (W, H, inner, id = "tfah") => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">${arr(id)}${inner}</svg>`;
  const bx = (x, y, w, h, label, cls = "node", sublab) => `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="8"/><text class="val" x="${x + w / 2}" y="${y + h / 2 + (sublab ? -1 : 5)}" text-anchor="middle" style="font-size:13px">${esc(label)}</text>` + (sublab ? `<text class="lbl" x="${x + w / 2}" y="${y + h / 2 + 14}" text-anchor="middle" style="font-size:11px">${esc(sublab)}</text>` : "");
  const ar = (x1, y1, x2, y2, id = "tfah", dash) => `<line class="edge" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#${id})"${dash ? ' style="stroke-dasharray:4 4"' : ""}/>`;
  const tx = (x, y, s, style = "", anchor = "middle") => `<text class="lbl" x="${x}" y="${y}" text-anchor="${anchor}" style="${style}">${esc(s)}</text>`;

  D.tf_embed = () => {
    const ids = [3, 7, 3], rows = 10, rh = 20, tx0 = 190, ty0 = 30;
    let s = tx(80, 16, "token IDs") + tx(tx0 + 60, 16, "embedding matrix E  (V × d)") + tx(520, 16, "output X  (3 × d)");
    ids.forEach((id, k) => { const y = 40 + k * 56; s += bx(40, y, 80, 34, "id " + id, "node hl") + ar(122, y + 17, tx0 - 2, ty0 + id * rh + rh / 2); });
    for (let r = 0; r < rows; r++) {
      const hot = r === 3 || r === 7;
      s += `<rect x="${tx0}" y="${ty0 + r * rh}" width="120" height="${rh - 2}" rx="3" style="fill:${hot ? "var(--hl)" : "var(--panel-2)"};stroke:${hot ? "var(--amber)" : "var(--border)"}"/><text class="lbl" x="${tx0 - 6}" y="${ty0 + r * rh + 14}" text-anchor="end" style="font-size:11px">row ${r}</text>`;
      for (let c = 0; c < 4; c++) s += `<text class="lbl" x="${tx0 + 18 + c * 28}" y="${ty0 + r * rh + 14}" text-anchor="middle" style="font-size:10.5px">${((Math.sin(r * 7 + c * 3) * 1.6)).toFixed(1)}</text>`;
    }
    ids.forEach((id, k) => { const y = 40 + k * 56; s += ar(tx0 + 124, ty0 + id * rh + rh / 2, 470, y + 17); s += `<rect x="472" y="${y}" width="120" height="34" rx="6" style="fill:var(--hl);stroke:var(--amber)"/>`; for (let c = 0; c < 4; c++) s += `<text class="lbl" x="${490 + c * 28}" y="${y + 21}" text-anchor="middle" style="font-size:10.5px">${((Math.sin(id * 7 + c * 3) * 1.6)).toFixed(1)}</text>`; });
    return wrapD(620, 238, s);
  };

  D.tf_qkv = () => {
    let s = tx(310, 16, "one query looks through all the keys, then collects the matching values", "font-size:12px");
    s += bx(10, 70, 100, 60, "query q", "node hl", "what I need");
    const ks = [["key 1: cat", 0.1], ["key 2: sat", 0.2], ["key 3: mat", 0.7]];
    ks.forEach(([k, w], i) => {
      const y = 34 + i * 60;
      s += bx(180, y, 110, 40, k, "node") + ar(112, 100, 178, y + 20);
      s += tx(330, y + 25, "q·k →  " + w.toFixed(1), "font-size:11px") + ar(372, y + 20, 408, y + 20);
      s += bx(410, y, 100, 40, "value " + (i + 1), i === 2 ? "node ok" : "node");
      s += `<text class="lbl" x="525" y="${y + 25}" style="font-size:11px">× ${w.toFixed(1)}</text>`;
    });
    s += `<path class="edge" d="M580,54 L596,54 L596,174 L580,174" style="fill:none"/>` + ar(596, 114, 612, 114);
    s += bx(614, 90, 70, 48, "Σ", "node ok", "output");
    return wrapD(700, 214, s);
  };

  D.tf_block = () => {
    const cx = 150, w = 130;
    let s = "";
    const ys = { in: 440, ln1: 372, att: 316, add1: 262, ln2: 202, mlp: 148, add2: 94, out: 28 };
    s += tx(cx, 466, "x (from the layer below)", "font-size:11px");
    s += ar(cx, 450, cx, ys.ln1 + 30);
    s += bx(cx - 50, ys.ln1, 100, 30, "LayerNorm", "node");
    s += ar(cx, ys.ln1, cx, ys.att + 40);
    s += bx(cx - w / 2, ys.att, w, 40, "Multi-head", "node hl", "self-attention");
    s += ar(cx, ys.att, cx, ys.add1 + 16);
    s += `<circle cx="${cx}" cy="${ys.add1}" r="14" class="node ok"/><text class="val" x="${cx}" y="${ys.add1 + 5}" text-anchor="middle">+</text>`;
    s += `<path class="edge hl" d="M${cx},${ys.in - 4} L${cx + 110},${ys.in - 4} L${cx + 110},${ys.add1} L${cx + 15},${ys.add1}" marker-end="url(#tfah)" style="fill:none;stroke:var(--accent)"/>`;
    s += tx(cx + 118, 340, "residual", "font-size:11px;fill:var(--accent)", "start");
    s += ar(cx, ys.add1 - 14, cx, ys.ln2 + 30);
    s += bx(cx - 50, ys.ln2, 100, 30, "LayerNorm", "node");
    s += ar(cx, ys.ln2, cx, ys.mlp + 40);
    s += bx(cx - w / 2, ys.mlp, w, 40, "MLP", "node hl", "expand · GELU · shrink");
    s += ar(cx, ys.mlp, cx, ys.add2 + 14);
    s += `<circle cx="${cx}" cy="${ys.add2}" r="14" class="node ok"/><text class="val" x="${cx}" y="${ys.add2 + 5}" text-anchor="middle">+</text>`;
    s += `<path class="edge hl" d="M${cx + 110},${ys.add1} L${cx + 110},${ys.add2} L${cx + 15},${ys.add2}" marker-end="url(#tfah)" style="fill:none;stroke:var(--accent)"/>`;
    s += ar(cx, ys.add2 - 14, cx, ys.out + 16);
    s += tx(cx, ys.out + 8, "x′ (to the layer above)", "font-size:11px");
    // right-hand legend
    const lx = 330;
    s += tx(lx, 60, "Pre-LN block (used by GPT-2 and most modern LLMs):", "font-weight:700;fill:var(--text);font-size:12px", "start");
    s += tx(lx, 86, "x ← x + Attention(LayerNorm(x))", "font-family:var(--mono);font-size:12px", "start");
    s += tx(lx, 108, "x ← x + MLP(LayerNorm(x))", "font-family:var(--mono);font-size:12px", "start");
    s += tx(lx, 150, "Attention mixes information BETWEEN tokens.", "font-size:12px", "start");
    s += tx(lx, 170, "The MLP transforms each token on its own.", "font-size:12px", "start");
    s += tx(lx, 210, "Both only ADD to the residual stream (blue path).", "font-size:12px", "start");
    s += tx(lx, 230, "Output shape = input shape, so blocks stack N times.", "font-size:12px", "start");
    return wrapD(640, 480, s);
  };

  D.tf_archs = () => {
    const grid = (x, y, n, causal, c = 16) => { let s = ""; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const on = !causal || j <= i; s += `<rect x="${x + j * c}" y="${y + i * c}" width="${c - 2}" height="${c - 2}" rx="2" style="fill:${on ? "color-mix(in srgb, var(--accent) 55%, var(--panel-2))" : "var(--panel-2)"};stroke:var(--border)"/>`; } return s; };
    let s = "";
    s += tx(110, 20, "Encoder-only", "font-weight:700;fill:var(--text);font-size:13px") + tx(110, 36, "BERT, RoBERTa", "font-size:11px");
    s += grid(62, 50, 6, false) + tx(110, 160, "every token sees every token", "font-size:11px") + tx(110, 176, "(bidirectional)", "font-size:11px");
    s += tx(110, 200, "output: a vector per token", "font-size:11px") + tx(110, 216, "tasks: classify, embed, tag", "font-size:11px");
    s += tx(330, 20, "Decoder-only", "font-weight:700;fill:var(--text);font-size:13px") + tx(330, 36, "GPT, Llama, Claude-style LLMs", "font-size:11px");
    s += grid(282, 50, 6, true) + tx(330, 160, "each token sees itself and the past", "font-size:11px") + tx(330, 176, "(causal mask)", "font-size:11px");
    s += tx(330, 200, "output: next-token probabilities", "font-size:11px") + tx(330, 216, "tasks: generate text", "font-size:11px");
    s += tx(560, 20, "Encoder–decoder", "font-weight:700;fill:var(--text);font-size:13px") + tx(560, 36, "original Transformer, T5, BART", "font-size:11px");
    s += grid(470, 50, 4, false, 14) + grid(590, 50, 4, true, 14);
    s += tx(498, 120, "encoder", "font-size:11px") + tx(618, 120, "decoder", "font-size:11px");
    s += ar(528, 78, 586, 78) + tx(558, 70, "cross", "font-size:10.5px");
    s += tx(560, 160, "decoder reads the encoder's output", "font-size:11px") + tx(560, 176, "through cross-attention", "font-size:11px");
    s += tx(560, 200, "output: a new sequence", "font-size:11px") + tx(560, 216, "tasks: translate, summarise", "font-size:11px");
    return wrapD(680, 232, s);
  };

  D.tf_gqa = () => {
    const sec = (x, title, sub, kv) => {
      let s = tx(x + 70, 20, title, "font-weight:700;fill:var(--text);font-size:13px") + tx(x + 70, 36, sub, "font-size:11px");
      for (let i = 0; i < 8; i++) s += `<rect class="node" x="${x + i * 17}" y="52" width="14" height="22" rx="3"/>`;
      s += tx(x + 70, 92, "8 query heads", "font-size:11px");
      const per = 8 / kv, w = 136 / kv;
      for (let g = 0; g < kv; g++) {
        const gx = x + g * w + (w - 14) / 2 + (kv === 8 ? 0 : 0);
        for (let i = 0; i < per; i++) s += `<line class="edge" x1="${x + (g * per + i) * 17 + 7}" y1="76" x2="${x + g * 17 * per + (per * 17 - 3) / 2}" y2="124" style="stroke-width:1.2"/>`;
        s += `<rect x="${x + g * 17 * per + (per * 17 - 3) / 2 - 7}" y="124" width="14" height="22" rx="3" style="fill:var(--green-soft);stroke:var(--green);stroke-width:2"/>`;
      }
      return s + tx(x + 70, 164, kv + " key/value head" + (kv > 1 ? "s" : ""), "font-size:11px") + tx(x + 70, 180, "KV cache: " + (kv / 8 === 1 ? "1×" : kv === 2 ? "¼" : "⅛") + " size", "font-size:11px;fill:var(--accent)");
    };
    return wrapD(640, 194, sec(10, "MHA", "multi-head", 8) + sec(222, "GQA", "grouped-query (2 groups)", 2) + sec(434, "MQA", "multi-query", 1));
  };

  D.tf_moe = () => {
    let s = tx(70, 24, "token vector", "font-size:11px") + bx(20, 34, 100, 36, "x", "node hl");
    s += ar(120, 52, 170, 52) + bx(172, 30, 90, 44, "Router", "node", "softmax · top-2");
    const ex = [["Expert 1", false, "0.00"], ["Expert 2", true, "0.62"], ["Expert 3", false, "0.00"], ["Expert 4", true, "0.38"]];
    ex.forEach(([n, on, w], i) => { const y = 14 + i * 50; s += ar(262, 52, 328, y + 20, "tfah", !on) + bx(330, y, 110, 38, n, on ? "node ok" : "node", on ? "weight " + w : "skipped"); if (on) s += ar(442, y + 19, 500, 112); });
    s += `<circle cx="516" cy="112" r="16" class="node ok"/><text class="val" x="516" y="117" text-anchor="middle">Σ</text>` + ar(534, 112, 574, 112) + tx(606, 116, "output", "font-size:11px");
    s += tx(380, 224, "only 2 of 4 experts run for this token: more parameters, same compute per token", "font-size:11.5px");
    return wrapD(660, 238, s);
  };

  D.tf_vit = () => {
    let s = tx(70, 16, "image", "font-size:11px");
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += `<rect x="${20 + j * 40}" y="${26 + i * 40}" width="38" height="38" rx="2" style="fill:color-mix(in srgb, var(--accent) ${15 + ((i * 3 + j) * 37) % 55}%, var(--panel-2));stroke:var(--border)"/><text class="lbl" x="${39 + j * 40}" y="${50 + i * 40}" text-anchor="middle" style="font-size:11px">${i * 3 + j + 1}</text>`;
    s += tx(80, 158, "cut into 3×3 patches", "font-size:11px") + ar(144, 86, 186, 86) + tx(166, 74, "flatten", "font-size:10.5px");
    s += bx(188, 60, 84, 52, "Linear", "node hl", "patch → vector") + ar(274, 86, 306, 86);
    // sequence
    s += tx(440, 16, "sequence of tokens (+ position embeddings)", "font-size:11px");
    ["[CLS]", "p1", "p2", "p3", "…", "p9"].forEach((t, i) => (s += `<rect class="${i === 0 ? "node ok" : "node"}" x="${308 + i * 44}" y="62" width="40" height="48" rx="6"/><text class="val" x="${328 + i * 44}" y="91" text-anchor="middle" style="font-size:11px">${t}</text>`));
    s += ar(440, 112, 440, 138) + bx(360, 140, 160, 34, "Transformer encoder", "node hl") + ar(440, 176, 440, 198) + tx(440, 214, "[CLS] vector → classifier", "font-size:11px");
    // multimodal
    s += tx(440, 246, "Multimodal LLM: image tokens and text tokens share one sequence", "font-weight:700;fill:var(--text);font-size:12px");
    ["img", "img", "img", "img", "What", "is", "this", "?"].forEach((t, i) => (s += `<rect class="${i < 4 ? "node ok" : "node"}" x="${180 + i * 44}" y="258" width="40" height="30" rx="6"/><text class="val" x="${200 + i * 44}" y="278" text-anchor="middle" style="font-size:11px">${t}</text>`));
    return wrapD(680, 304, s);
  };

  D.tf_gpt = () => {
    const items = [["token IDs", "(T,)"], ["Embedding + position", "(T, d)"], ["Block 1 … N", "(T, d)"], ["Final LayerNorm", "(T, d)"], ["LM head  (· Eᵀ)", "(T, V)"], ["softmax of last row", "(V,)"]];
    let s = "";
    items.forEach(([a, b], i) => { const y = 8 + i * 52; s += bx(110, y, 190, 34, a, i === 2 ? "node hl" : "node") + tx(320, y + 22, b, "font-family:var(--mono);font-size:11.5px", "start"); if (i < 5) s += ar(205, y + 36, 205, y + 50); });
    s += tx(360, 70, "", "") + tx(420, 150, "← the only place where tokens interact: attention", "font-size:11px;fill:var(--accent)", "start");
    return wrapD(640, 326, s);
  };

  D.tf_flash = () => {
    let s = bx(20, 20, 150, 60, "HBM (GPU memory)", "node", "big, slow");
    s += bx(20, 130, 150, 60, "SRAM (on-chip)", "node hl", "tiny, very fast");
    s += ar(80, 82, 80, 128) + ar(110, 128, 110, 82);
    s += tx(250, 56, "Standard: write the full T×T score matrix to HBM,", "font-size:12px", "start") + tx(250, 74, "read it back for softmax, write again, read again.", "font-size:12px", "start");
    s += tx(250, 150, "FlashAttention: load a tile of Q, K, V into SRAM,", "font-size:12px;fill:var(--accent)", "start") + tx(250, 168, "do scores, running softmax and the V-sum there,", "font-size:12px;fill:var(--accent)", "start") + tx(250, 186, "write only the final output. Same result, far less traffic.", "font-size:12px;fill:var(--accent)", "start");
    return wrapD(640, 210, s);
  };

  /* ================================================================== glossary */
  Object.assign(window.GLOSSARY, {
    "subword": "A piece of a word (like un, happi, ness) used as a token, so rare words can be built from common parts.",
    "vocabulary": "The fixed list of all tokens a model knows. Each entry has an integer ID.",
    "bpe": "Byte-Pair Encoding: build a subword vocabulary by repeatedly merging the most frequent pair of neighbouring symbols.",
    "byte-pair encoding": "Build a subword vocabulary by repeatedly merging the most frequent pair of neighbouring symbols.",
    "wordpiece": "BERT's subword method: like BPE, but merges the pair that most improves the likelihood; pieces inside a word start with ##.",
    "sentencepiece": "A tokenizer library that works on raw text including spaces (shown as ▁), so it needs no language-specific word splitting.",
    "special tokens": "Reserved tokens with a job instead of a meaning, such as start-of-text, end-of-text, padding or chat role markers.",
    "positional encoding": "Extra information that tells the model where each token sits in the sequence, because attention alone ignores order.",
    "rope": "Rotary position embedding: rotate each query and key by an angle proportional to its position, so attention scores depend on relative distance.",
    "query": "In attention, the vector a token uses to ask: which other tokens are relevant to me?",
    "key": "In attention, the vector a token shows to others so they can decide whether it matches their query.",
    "value": "In attention, the content a token hands over when other tokens attend to it.",
    "self-attention": "Attention where queries, keys and values all come from the same sequence: every token looks at every other token in it.",
    "cross-attention": "Attention where the queries come from one sequence and the keys and values from another (e.g. decoder asking about the encoder's output).",
    "attention weights": "The softmax outputs of attention: for each token, non-negative numbers that sum to 1, saying how much to read from each other token.",
    "causal mask": "A mask that blocks attention to later tokens, so each token sees only itself and the past. Needed for next-token prediction.",
    "padding mask": "A mask that stops attention from reading <pad> filler tokens added to make batch sequences the same length.",
    "multi-head attention": "Several attention computations run in parallel on different slices of the vector, then joined. Lets the model look for different relations at once.",
    "residual connection": "Adding a layer's input to its output (x + f(x)), so information and gradients can flow past the layer unchanged.",
    "residual stream": "The vector each token carries up through the layers; every attention and MLP sublayer reads from it and adds to it.",
    "layernorm": "Normalising each token's vector to mean 0 and variance 1 (then rescaling with learned numbers). Keeps activations stable.",
    "feed-forward network": "The MLP inside each transformer block: expand the vector, apply a nonlinearity, shrink back. Works on each token separately.",
    "encoder": "The part of a transformer that reads a whole input with bidirectional attention and produces a vector per token (BERT).",
    "decoder": "The part of a transformer that generates tokens one at a time with a causal mask (GPT).",
    "kv cache": "Stored keys and values of past tokens, reused during generation so each new token costs one step instead of recomputing everything.",
    "flashattention": "An exact attention algorithm that works in fast on-chip memory tiles and never writes the full score matrix, making it faster and lighter on memory.",
    "mqa": "Multi-query attention: all heads share a single key and value head, which shrinks the KV cache a lot.",
    "gqa": "Grouped-query attention: query heads are split into groups that each share one key/value head. A middle ground between MHA and MQA.",
    "moe": "Mixture of experts: many parallel feed-forward 'experts' and a router that sends each token to only a few of them.",
    "router": "In a mixture-of-experts layer, the small network that scores the experts and picks which ones process each token.",
    "patch": "A small square piece of an image (e.g. 16×16 pixels) turned into one token by a Vision Transformer.",
    "weight tying": "Reusing the input embedding matrix (transposed) as the output layer, saving parameters.",
    "lm head": "The final linear layer of a language model that turns each token's vector into one score (logit) per vocabulary entry.",
  });
})();
