/* LLMs part 1 (training and inference): visualizers, diagrams and glossary terms.
   Loaded on every doc page and on Practice, after assets/viz.js. Every number shown is computed, not faked. */
(function () {
  const V = Viz.defs, Plot = Viz.Plot, { r3 } = Viz.helpers;

  /* ------------------------------------------------------------------ private helpers */
  const rng = (seed) => { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  const gauss = (r) => { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const f2 = (v, d = 2) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : +v).toFixed(d);
  const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";
  const sm = (v) => { const m = Math.max(...v); const e = v.map((x) => Math.exp(x - m)); const s = sum(e); return e.map((x) => x / s); };
  const ent = (p) => -sum(p.map((x) => (x > 0 ? x * Math.log(x) : 0)));
  const wrap = (W, H, inner) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">${inner}</svg>`;
  const T = (x, y, s, style = "", anchor = "middle") => `<text x="${x}" y="${y}" text-anchor="${anchor}" style="font-size:11.5px;fill:var(--muted);${style}">${esc(s)}</text>`;
  const big = (n) => (n >= 1e9 ? (n / 1e9).toFixed(2) + " B" : n >= 1e6 ? (n / 1e6).toFixed(2) + " M" : n >= 1e3 ? (n / 1e3).toFixed(1) + " K" : String(n));

  /* vertical probability bars. items: [{label, p, kind: "keep"|"cut"|"plain", p2?}]  p2 = renormalised value drawn solid, p = original outline */
  function barChart(items, o = {}) {
    const W = o.W || 560, H = o.H || 250, L = 40, B = 46, Tp = 18, n = items.length, bw = Math.min(46, (W - L - 12) / n - 8), mx = o.max || Math.max(0.05, ...items.map((i) => Math.max(i.p, i.p2 || 0))) * 1.1;
    const sy = (v) => H - B - (v / mx) * (H - B - Tp);
    let s = "";
    for (let k = 0; k <= 4; k++) { const v = (mx * k) / 4; s += `<line x1="${L}" y1="${sy(v)}" x2="${W - 8}" y2="${sy(v)}" style="stroke:var(--border);stroke-width:1"/>` + T(L - 6, sy(v) + 4, f2(v, 2), "", "end"); }
    items.forEach((it, i) => {
      const cx = L + 8 + (i + 0.5) * ((W - L - 12) / n), cut = it.kind === "cut";
      const col = cut ? "var(--muted)" : it.kind === "keep" ? "var(--green)" : "var(--accent)";
      if (it.p2 != null) {
        s += `<rect x="${cx - bw / 2}" y="${sy(it.p)}" width="${bw}" height="${Math.max(0, sy(0) - sy(it.p))}" style="fill:none;stroke:${col};stroke-dasharray:3 3;opacity:${cut ? 0.6 : 0.9}"/>`;
        if (it.p2 > 0) s += `<rect x="${cx - bw / 2}" y="${sy(it.p2)}" width="${bw}" height="${sy(0) - sy(it.p2)}" rx="2" style="fill:${col};fill-opacity:.8"/>` + T(cx, sy(it.p2) - 4, f2(it.p2, 2), "fill:var(--text);font-size:10.5px");
        else s += T(cx, sy(0) - 4, "cut", "font-size:10px");
      } else {
        s += `<rect x="${cx - bw / 2}" y="${sy(it.p)}" width="${bw}" height="${Math.max(0, sy(0) - sy(it.p))}" rx="2" style="fill:${col};fill-opacity:${cut ? 0.35 : 0.8}"/>` + T(cx, sy(it.p) - 4, f2(it.p, 2), "fill:var(--text);font-size:10.5px");
      }
      s += T(cx, H - B + 16, it.label, `font-size:11.5px;${cut ? "" : "fill:var(--text)"}`);
      if (it.sub != null) s += T(cx, H - B + 30, it.sub, "font-size:10px");
    });
    return wrap(W, H, s);
  }

  /* ================================================================== 1. Bigram language model */
  V.llm1_bigram = {
    title: "A bigram language model: count pairs, then sample",
    input: "the cat sat on the mat . the dog sat on the log . the cat ate the dog .",
    help: "any short text; words are tokens (at most 10 different ones)",
    params: [{ id: "alpha", label: "add-α smoothing", min: 0, max: 1, step: 0.1, value: 0 }],
    gen(input, P) {
      let toks = input.trim().toLowerCase().replace(/\./g, " . ").split(/\s+/).filter(Boolean).slice(0, 60);
      const vocab = [];
      const keep = [];
      for (const t of toks) { if (!vocab.includes(t)) { if (vocab.length >= 10) break; vocab.push(t); } keep.push(t); }
      toks = keep;
      const n = vocab.length, idx = (t) => vocab.indexOf(t), a = P.alpha;
      if (n < 2) return [{ svg: wrap(300, 40, T(150, 24, "type a few words")), side: {}, msg: "Type at least two different words." }];
      const cnt = Array.from({ length: n }, () => Array(n).fill(0));
      for (let i = 0; i + 1 < toks.length; i++) cnt[idx(toks[i])][idx(toks[i + 1])]++;
      const prob = cnt.map((r) => { const z = sum(r) + a * n; return r.map((c) => (z > 0 ? (c + a) / z : 1 / n)); });
      let nll = 0, m = 0;
      for (let i = 0; i + 1 < toks.length; i++) { nll -= Math.log(Math.max(prob[idx(toks[i])][idx(toks[i + 1])], 1e-12)); m++; }
      const ppl = Math.exp(nll / m);
      const cell = Math.min(34, Math.floor(300 / n)), LW = 54, CH = 34, W = LW + n * cell + 26 + 230, H = Math.max(CH + n * cell + 14, 210);
      const mx = Math.max(1, ...cnt.flat());
      const draw = (row, col, seq) => {
        let s = "";
        vocab.forEach((v, j) => { s += T(LW + j * cell + cell / 2, CH - 8, v.length > 4 ? v.slice(0, 4) + "." : v, "font-size:10.5px"); });
        s += T(LW + (n * cell) / 2, 11, "next token →", "font-size:10.5px");
        vocab.forEach((v, i) => {
          s += T(LW - 6, CH + i * cell + cell / 2 + 4, v.length > 6 ? v.slice(0, 5) + "." : v, `font-size:11px;${row === i ? "fill:var(--text);font-weight:700" : ""}`, "end");
          for (let j = 0; j < n; j++) {
            const c = cnt[i][j];
            s += `<rect x="${LW + j * cell + 1}" y="${CH + i * cell + 1}" width="${cell - 2}" height="${cell - 2}" rx="3" style="fill:color-mix(in srgb, var(--accent) ${Math.round((c / mx) * 75)}%, var(--panel-2));stroke:${col === j && row === i ? "var(--amber)" : "var(--border)"};stroke-width:${col === j && row === i ? 2.5 : 1}"/>` + (c || cell > 26 ? T(LW + j * cell + cell / 2, CH + i * cell + cell / 2 + 4, c, "fill:var(--text);font-size:10.5px") : "");
          }
        });
        s += T(10, CH + (n * cell) / 2, "previous", "font-size:10.5px", "start").replace("<text", `<text transform="rotate(-90 10 ${CH + (n * cell) / 2})"`);
        if (row != null) {
          s += `<rect x="${LW}" y="${CH + row * cell}" width="${n * cell}" height="${cell}" rx="4" style="fill:none;stroke:var(--amber);stroke-width:2"/>`;
          const bx = LW + n * cell + 26, bw = 200, bh = 15, by = CH + 4;
          s += T(bx, by - 6, `P(next | "${vocab[row]}")`, "font-size:11.5px;fill:var(--text);font-weight:600", "start");
          vocab.forEach((v, j) => {
            const y = by + 4 + j * (bh + 3), w = prob[row][j] * (bw - 60);
            s += T(bx + 38, y + 11, v, "font-size:10.5px", "end") + `<rect x="${bx + 44}" y="${y}" width="${Math.max(0, w)}" height="${bh}" rx="2" style="fill:${col === j ? "var(--amber)" : "var(--accent)"};fill-opacity:.8"/>` + T(bx + 48 + w, y + 11, f2(prob[row][j], 2), "font-size:10.5px;fill:var(--text)", "start");
          });
        }
        if (seq) s += T(LW, CH + n * cell + 18, "generated: " + seq.join(" "), "fill:var(--text);font-size:12px", "start");
        return wrap(W, H + (seq ? 14 : 0), s);
      };
      const S = [{ svg: draw(null, null, null), side: { tokens: vocab.length, "word pairs": m, perplexity: f2(ppl, 2), "smoothing α": a },
        msg: `**Training** a bigram model is just counting. Row = the previous token, column = the next token, number = how often that pair occurs in your text. The ${m} pairs are all the model ever sees. Dividing a row by its sum gives $P(\\text{next}\\mid\\text{previous})$. Perplexity on this same text: **${f2(ppl, 2)}** (roughly "how many tokens the model is torn between"). Try α = 0 versus 1, then press Next to generate.` }];
      const r = rng(7);
      let cur = idx(toks[0]); const seq = [vocab[cur]];
      for (let k = 0; k < 9; k++) {
        const row = prob[cur]; let u = r(), nx = 0, acc = 0;
        for (let j = 0; j < n; j++) { acc += row[j]; if (u <= acc) { nx = j; break; } nx = j; }
        const u0 = u;
        seq.push(vocab[nx]);
        S.push({ svg: draw(cur, nx, seq.slice()), side: { previous: vocab[cur], "random u": f2(u0, 3), sampled: vocab[nx], "P(sampled)": f2(row[nx], 3) },
          msg: `Look at row **${vocab[cur]}**. Draw a random number $u=${f2(u0, 3)}$ and walk along the row adding probabilities until the running total passes $u$: that lands on **${vocab[nx]}** (probability ${f2(row[nx], 3)}). Append it and repeat with the new token as the "previous" one.` });
        cur = nx;
      }
      return S;
    },
  };

  /* ================================================================== 2. Softmax with temperature */
  const WORDS = ["cat", "dog", "man", "car", "idea", "sat", "the", "zebra"], LOG = [3.0, 2.7, 2.0, 1.4, 0.6, 0.2, -0.5, -1.5];
  V.llm1_temperature = {
    title: "Softmax with temperature: sharpen or flatten the distribution",
    params: [{ id: "T", label: "temperature T", min: 0.1, max: 3, step: 0.05, value: 1 }],
    gen(_, P) {
      const p = sm(LOG.map((z) => z / P.T)), p1 = sm(LOG), H = ent(p), top = p.indexOf(Math.max(...p));
      const svg = barChart(WORDS.map((w, i) => ({ label: w, p: p1[i], p2: p[i], sub: "z=" + f2(LOG[i], 1) })), { max: 1 });
      return [{ svg, side: { T: P.T, "top token": WORDS[top], "P(top)": f2(p[top], 3), "entropy (nats)": f2(H, 3), "effective choices e^H": f2(Math.exp(H), 2) },
        msg: `The model outputs raw scores (logits, shown as z). The probability is $p_i = \\dfrac{e^{z_i/T}}{\\sum_j e^{z_j/T}}$. Dashed outlines are the probabilities at $T=1$, solid bars are at the current $T$. **Low T** (towards 0) piles all the mass on the top token (greedy). **High T** flattens everything towards uniform (here $\\ln 8 = ${f2(Math.log(8), 3)}$ nats). The ranking of tokens never changes, only how peaked it is.` }];
    },
  };

  /* ================================================================== 3. Top-k and top-p */
  const WORDS2 = ["the", "a", "cat", "dog", "sat", "mat", "ran", "on", "zebra", "qux"], LOG2 = [3.2, 2.8, 2.5, 2.2, 1.6, 1.1, 0.5, 0.0, -1.0, -2.5];
  V.llm1_topkp = {
    title: "Top-k and top-p (nucleus) filtering",
    params: [
      { id: "T", label: "temperature T", min: 0.3, max: 2, step: 0.05, value: 1 },
      { id: "k", label: "top-k", min: 1, max: 10, step: 1, value: 10 },
      { id: "p", label: "top-p", min: 0.1, max: 1, step: 0.05, value: 0.9 },
    ],
    gen(_, P) {
      const p0 = sm(LOG2.map((z) => z / P.T)), n = p0.length;
      const kept1 = p0.map((_, i) => i < P.k);                       // already sorted descending
      const z1 = sum(p0.filter((_, i) => kept1[i])), p1 = p0.map((v, i) => (kept1[i] ? v / z1 : 0));
      let cum = 0, kept2 = p1.map(() => false), cumv = [];
      for (let i = 0; i < n; i++) { if (!kept1[i]) { cumv.push(null); continue; } cumv.push(cum + p1[i]); if (cum < P.p - 1e-12) kept2[i] = true; cum += p1[i]; }
      const z2 = sum(p1.filter((_, i) => kept2[i])), p2 = p1.map((v, i) => (kept2[i] ? v / z2 : 0));
      const items = WORDS2.map((w, i) => ({ label: w, p: p0[i], p2: p2[i], kind: kept2[i] ? "keep" : "cut", sub: cumv[i] == null ? "" : "cum " + f2(cumv[i], 2) }));
      const nk = kept2.filter(Boolean).length;
      return [{ svg: barChart(items, { max: Math.max(...p0, ...p2) }), side: { "tokens kept": nk + " of " + n, "mass kept": f2(z1 * z2, 3), "top-k kept": kept1.filter(Boolean).length, "entropy before": f2(ent(p0), 3), "entropy after": f2(ent(p2), 3) },
        msg: `Order: temperature → **top-k** (keep the k most likely) → renormalise → **top-p** (keep the smallest top set whose cumulative probability reaches p) → renormalise → sample. Dashed = original probability, solid = what you would actually sample from. The "cum" row is the running total. Top-k always keeps k tokens; top-p keeps **few** tokens when the model is confident and **many** when it is unsure. Set k=1 to see greedy decoding.` }];
    },
  };

  /* ================================================================== 4. Greedy vs beam search */
  const BTAB = { "": [0.5, 0.4, 0.1], a: [0.4, 0.3, 0.3], b: [0.1, 0.8, 0.1], c: [0.3, 0.3, 0.4] }, BTOK = ["a", "b", "c"];
  V.llm1_beam = {
    title: "Greedy vs beam search on a toy next-token table",
    params: [{ id: "w", label: "beam width", min: 1, max: 4, step: 1, value: 2 }],
    gen(_, P) {
      const w = P.w, depth = 3, levels = [];
      let beams = [{ seq: "", lp: 0, id: "root" }];
      const rowH = 22, X = (l) => 14 + l * 150;
      for (let l = 1; l <= depth; l++) {
        const cand = [];
        beams.forEach((b) => BTOK.forEach((t, i) => cand.push({ seq: b.seq + t, lp: b.lp + Math.log(BTAB[b.seq.slice(-1)][i]), parent: b.seq, tok: t, p: BTAB[b.seq.slice(-1)][i] })));
        cand.sort((x, y) => y.lp - x.lp);
        cand.forEach((c, i) => (c.kept = i < w));
        levels.push(cand);
        beams = cand.filter((c) => c.kept);
      }
      // greedy path
      let gs = "", gp = 1;
      for (let l = 0; l < depth; l++) { const row = BTAB[gs.slice(-1)]; const i = row.indexOf(Math.max(...row)); gs += BTOK[i]; gp *= row[i]; }
      const best = levels[depth - 1][0];
      const mkSvg = (upto, final) => {
        const H = 40 + Math.max(3, ...levels.slice(0, upto).map((c) => c.length)) * rowH + 22, W = X(depth) + 140;
        let s = `<rect class="node" x="${X(0)}" y="${H / 2 - 12}" width="40" height="24" rx="6"/>` + T(X(0) + 20, H / 2 + 4, "start", "font-size:11px");
        const pos = { "": [X(0) + 40, H / 2] };
        levels.slice(0, upto).forEach((cand, li) => {
          s += T(X(li + 1) + 55, 18, "token " + (li + 1), "font-size:11px");
          cand.forEach((c, i) => {
            const y = 34 + i * rowH, x = X(li + 1) + 10;
            pos[c.seq] = [x + 124, y + 10];
            const par = pos[c.parent] || [X(0) + 40, H / 2];
            const isBest = final && c.seq === best.seq && li === depth - 1, isGreedy = final && c.seq === gs && li === depth - 1;
            s += `<line x1="${par[0]}" y1="${par[1]}" x2="${x}" y2="${y + 10}" style="stroke:${c.kept ? "var(--muted)" : "var(--border)"};stroke-width:${c.kept ? 1.6 : 1};${c.kept ? "" : "stroke-dasharray:3 3"}"/>`;
            s += `<rect class="node${c.kept ? (isBest ? " ok" : "") : ""}" x="${x}" y="${y}" width="124" height="20" rx="5" style="${c.kept ? "" : "opacity:.45"}${isGreedy && !isBest ? ";stroke:var(--amber);stroke-width:3" : ""}"/>`;
            s += `<text x="${x + 6}" y="${y + 14}" style="font-size:11.5px;font-weight:600;fill:var(--text);${c.kept ? "" : "opacity:.6"}">${c.seq}</text><text x="${x + 118}" y="${y + 14}" text-anchor="end" style="font-size:11px;fill:var(--muted)">logP ${f2(c.lp, 2)}</text>`;
          });
        });
        return wrap(W, H, s);
      };
      const S = [];
      for (let l = 1; l <= depth; l++) {
        const kept = levels[l - 1].filter((c) => c.kept).map((c) => c.seq + " (" + f2(Math.exp(c.lp), 3) + ")").join(", ");
        S.push({ svg: mkSvg(l, false), side: { "beam width": w, step: l, "candidates": levels[l - 1].length, "kept": levels[l - 1].filter((c) => c.kept).length },
          msg: l === 1 ? `Expand the start into its 3 possible tokens. Each node shows the **cumulative log-probability** of the sequence. Keep only the best **${w}** (solid); the rest are pruned (faded, dashed).` : `Expand every kept sequence by all 3 tokens, giving ${levels[l - 1].length} candidates. Sort by cumulative log-probability and keep the top **${w}**: ${kept}.` });
      }
      S.push({ svg: mkSvg(depth, true), side: { "beam best": best.seq + "  P=" + f2(Math.exp(best.lp), 3), "greedy": gs + "  P=" + f2(gp, 3), "beam finds better?": Math.exp(best.lp) > gp + 1e-12 ? "yes" : "same" },
        msg: Math.exp(best.lp) > gp + 1e-12 ? `Beam search returns **${best.seq}** with probability ${f2(Math.exp(best.lp), 3)}. Greedy (amber outline) walked into **${gs}** with only ${f2(gp, 3)}: its first choice **a** looked best (0.5) but led to a flat, uncertain continuation, while **b** (0.4) was followed by a very confident **b** (0.8). Greedy cannot recover from an early choice; a wider beam can.` : `With width ${w} the beam and greedy agree on **${best.seq}** (probability ${f2(Math.exp(best.lp), 3)}). Increase the width to 2 to see beam search beat greedy on this table.` });
      return S;
    },
  };

  /* ================================================================== 5. KV cache */
  const KVW = "the quick brown fox jumps over the lazy dog today and tomorrow".split(" ");
  V.llm1_kvcache = {
    title: "The KV cache: generate token by token without recomputing",
    params: [{ id: "n", label: "tokens to generate", min: 4, max: 12, step: 1, value: 8 }],
    gen(_, P) {
      const n = P.n, cw = 40, X0 = 100, W = Math.max(X0 + n * cw + 20, 600), H = 190, S = [];
      let cumNo = 0, cumYes = 0;
      for (let t = 0; t < n; t++) {
        cumNo += t + 1; cumYes += 1;
        let s = T(X0 + (n * cw) / 2, 14, "keys and values (K, V) per position", "font-size:11.5px;fill:var(--text)");
        const strip = (y0, mode, title) => {
          let o = T(X0 - 10, y0 + 28, title, "font-size:11.5px;fill:var(--text)", "end") + T(X0 - 10, y0 + 43, mode === "no" ? "recompute all" : "reuse + 1 new", "font-size:10.5px", "end");
          for (let i = 0; i < n; i++) {
            const x = X0 + i * cw + 2, future = i > t;
            let fill, stroke;
            if (future) { fill = "var(--panel)"; stroke = "var(--border)"; }
            else if (mode === "yes" && i < t) { fill = "var(--green-soft)"; stroke = "var(--green)"; }
            else { fill = "var(--hl)"; stroke = "var(--amber)"; }
            o += `<rect x="${x}" y="${y0}" width="${cw - 4}" height="22" rx="4" style="fill:${fill};stroke:${stroke};${future ? "stroke-dasharray:3 3" : ""}"/><text x="${x + (cw - 4) / 2}" y="${y0 + 15}" text-anchor="middle" style="font-size:11px;fill:${future ? "var(--muted)" : "var(--text)"}">K${i}</text>`;
            o += `<rect x="${x}" y="${y0 + 26}" width="${cw - 4}" height="22" rx="4" style="fill:${fill};stroke:${stroke};${future ? "stroke-dasharray:3 3" : ""}"/><text x="${x + (cw - 4) / 2}" y="${y0 + 41}" text-anchor="middle" style="font-size:11px;fill:${future ? "var(--muted)" : "var(--text)"}">V${i}</text>`;
          }
          return o;
        };
        s += strip(30, "no", "No cache") + strip(100, "yes", "KV cache");
        for (let i = 0; i < n; i++) s += T(X0 + i * cw + (cw - 4) / 2 + 2, 172, KVW[i], i === t ? "fill:var(--text);font-weight:700;font-size:11px" : "font-size:11px");
        s += T(W / 2, 186, "amber = computed now   green = read from cache   dashed = not generated yet", "font-size:10.5px");
        S.push({ svg: wrap(W, H, s), side: { "new token": KVW[t], "K/V rows computed (no cache)": t + 1, "K/V rows computed (cache)": 1, "total so far (no cache)": cumNo, "total so far (cache)": cumYes, "work saved": pct(1 - cumYes / cumNo, 0) },
          msg: t === 0 ? `Generating **${KVW[0]}**: nothing is cached yet, so both ways compute one K and V pair.` : `Step ${t + 1}: the new token **${KVW[t]}** needs K and V for itself, but the **${t}** earlier tokens have not changed, so the cached version just reads their rows. Without a cache the model would project all ${t + 1} tokens again. (Attention still compares the new query with all ${t + 1} keys, so the cache removes the recomputation, not the reading.)` });
      }
      return S;
    },
  };

  /* ================================================================== 6. Quantization */
  const QW = (() => { const r = rng(5); return Array.from({ length: 24 }, () => Math.round(gauss(r) * 0.5 * 1000) / 1000); })();
  V.llm1_quant = {
    title: "Quantization: float weights to integer buckets and back",
    params: [
      { id: "bits", label: "bits per weight", min: 2, max: 8, step: 1, value: 4 },
      { id: "out", label: "outlier size", min: 0, max: 6, step: 0.5, value: 0 },
    ],
    gen(_, P) {
      const w = QW.slice(); if (P.out > 0) w[3] = P.out * 0.5;
      const b = P.bits, qmax = Math.pow(2, b - 1) - 1, amax = Math.max(...w.map(Math.abs)), scale = amax / qmax;
      const q = w.map((v) => Math.max(-qmax, Math.min(qmax, Math.round(v / scale)))), dq = q.map((v) => v * scale), err = w.map((v, i) => dq[i] - v);
      const mse = sum(err.map((e) => e * e)) / w.length, mxe = Math.max(...err.map(Math.abs));
      const n = w.length, Wd = 560, L = 44, ym = amax * 1.12, hT = 190, top = 16;
      const cw = (Wd - L - 10) / n, sy = (v) => top + hT / 2 - (v / ym) * (hT / 2);
      let s = "";
      if (qmax <= 20) for (let k = -qmax; k <= qmax; k++) s += `<line x1="${L}" y1="${sy(k * scale)}" x2="${Wd - 6}" y2="${sy(k * scale)}" style="stroke:var(--border);stroke-width:1;stroke-dasharray:${k === 0 ? "0" : "2 4"}"/>`;
      s += `<line x1="${L}" y1="${sy(0)}" x2="${Wd - 6}" y2="${sy(0)}" style="stroke:var(--muted)"/>`;
      s += T(L - 6, sy(amax) + 4, f2(amax, 2), "", "end") + T(L - 6, sy(-amax) + 4, f2(-amax, 2), "", "end") + T(L - 6, sy(0) + 4, "0", "", "end");
      w.forEach((v, i) => {
        const x = L + (i + 0.5) * cw;
        s += `<line x1="${x}" y1="${sy(v)}" x2="${x}" y2="${sy(dq[i])}" style="stroke:var(--red);stroke-width:2"/>` +
          `<circle cx="${x}" cy="${sy(v)}" r="4.5" style="fill:var(--accent)"/><rect x="${x - 4}" y="${sy(dq[i]) - 4}" width="8" height="8" style="fill:var(--amber);stroke:var(--panel)"/>`;
      });
      s += T(Wd / 2, top + hT + 22, "weight index  (round dot = original float, square = after quantize + dequantize, vertical line = error)", "font-size:11px");
      // integer codes row
      const y2 = top + hT + 40;
      w.forEach((v, i) => { s += T(L + (i + 0.5) * cw, y2, q[i], "fill:var(--text);font-size:10px"); });
      s += T(L - 6, y2, "int", "font-size:10.5px", "end");
      return [{ svg: wrap(Wd, y2 + 10, s), side: { bits: b, "integer range": `-${qmax} … ${qmax}`, ["scale = max|w| / " + qmax]: r3(scale), "max |error|": f2(mxe, 4), "bound scale/2": f2(scale / 2, 4), MSE: r3(mse), "memory vs fp16": pct(b / 16, 1) },
        msg: `Symmetric absmax quantization: $s=\\max|w|/${qmax}$, store $q=\\mathrm{round}(w/s)$ as a ${b}-bit integer, and recover $\\hat w = q\\,s$. Dashed lines are the only values $\\hat w$ can take (${2 * qmax + 1} of them). The error never exceeds $s/2$. Drop the bits and the grid gets coarse. Now raise **outlier size**: one huge weight stretches the scale, so every ordinary weight gets squeezed into just a few buckets and the error rises. This is why real schemes quantize small blocks or groups of weights, each with its own scale.` }];
    },
  };

  /* ================================================================== 7. LoRA shapes and parameter count */
  V.llm1_lora = {
    title: "LoRA: a low-rank update instead of a full weight update",
    params: [
      { id: "d", label: "hidden size d", min: 256, max: 8192, step: 256, value: 4096 },
      { id: "r", label: "rank r", min: 1, max: 64, step: 1, value: 8 },
      { id: "L", label: "layers (4 matrices each)", min: 1, max: 80, step: 1, value: 32 },
    ],
    gen(_, P) {
      const { d, r, L } = P, full = d * d, lora = 2 * d * r, S0 = 130, wr = Math.max(5, (S0 * r) / d);
      let s = "";
      const x0 = 14, y0 = 40;
      s += T(x0 + S0 / 2, 20, `W  (d×d)`, "font-size:12px;fill:var(--text)") + `<rect x="${x0}" y="${y0}" width="${S0}" height="${S0}" rx="6" style="fill:var(--panel-2);stroke:var(--muted)"/>` + T(x0 + S0 / 2, y0 + S0 / 2 + 4, "frozen", "font-size:12px") + T(x0 + S0 / 2, y0 + S0 + 18, `${full.toLocaleString("en-US")} numbers`, "font-size:11px");
      s += T(x0 + S0 + 22, y0 + S0 / 2 + 6, "+", "font-size:22px");
      const bx = x0 + S0 + 40;
      s += T(bx + wr / 2, 20, "B (d×r)", "font-size:12px;fill:var(--text)") + `<rect x="${bx}" y="${y0}" width="${wr}" height="${S0}" rx="3" style="fill:color-mix(in srgb, var(--accent) 35%, transparent);stroke:var(--accent);stroke-width:2"/>`;
      const ax = bx + wr + 26;
      s += T(ax + 30 + S0 / 2, 20, "A (r×d)", "font-size:12px;fill:var(--text)") + `<rect x="${ax + 30}" y="${y0}" width="${S0}" height="${wr}" rx="3" style="fill:color-mix(in srgb, var(--green) 35%, transparent);stroke:var(--green);stroke-width:2"/>`;
      s += T(bx + wr + 13, y0 + S0 / 2 + 6, "×", "font-size:20px");
      s += T(bx + (wr + 26 + 30 + S0) / 2, y0 + S0 + 18, `${lora.toLocaleString("en-US")} trainable numbers`, "font-size:11px;fill:var(--text)");
      s += T(bx + (wr + 26 + 30 + S0) / 2, y0 + S0 + 34, "(drawn to scale; r/d is tiny, so B and A are thin)", "font-size:10.5px");
      const W = Math.max(ax + 30 + S0 + 20, 520), H = y0 + S0 + 44;
      const totFull = full * 4 * L, totLora = lora * 4 * L;
      return [{ svg: wrap(W, H, s), side: { "per matrix: full": big(full), "per matrix: LoRA": big(lora), "fraction": pct(lora / full, 2), [`all ${4 * L} attention matrices: full`]: big(totFull), "all: LoRA": big(totLora) },
        msg: `Instead of learning a full update $\\Delta W$ ($d\\times d$), LoRA learns $\\Delta W = B A$ with $B\\in\\mathbb{R}^{d\\times r}$ and $A\\in\\mathbb{R}^{r\\times d}$. The layer computes $y = Wx + \\tfrac{\\alpha}{r} B(Ax)$. Trainable numbers per matrix: $2dr$ instead of $d^2$, so a fraction $2r/d$ = **${pct(lora / full, 2)}**. Slide the rank up and the fraction grows linearly. ($B$ starts at zero, so training begins exactly at the base model.)` }];
    },
  };

  /* ================================================================== 8. Illustrative scaling law */
  const CH = { E: 1.7, A: 600, B: 400, a: 0.34, b: 0.28 };   // round, made-up constants of the Chinchilla functional form (not the published fit)
  const chin = (N, D) => CH.E + CH.A / Math.pow(N, CH.a) + CH.B / Math.pow(D, CH.b);
  V.llm1_scaling = {
    title: "Scaling law (ILLUSTRATIVE): for a fixed compute budget, what size model is best?",
    params: [{ id: "c", label: "compute budget log₁₀ C (FLOPs)", min: 18, max: 26, step: 0.5, value: 23 }],
    gen(_, P) {
      const C = Math.pow(10, P.c), pts = [];
      let bestN = 0, bestL = 99;
      for (let x = 6.5; x <= 12.5; x += 0.05) { const N = Math.pow(10, x), D = C / (6 * N), l = chin(N, D); if (D >= 1e6) { pts.push([x, l]); if (l < bestL) { bestL = l; bestN = N; } } }
      const Dbest = C / (6 * bestN), N20 = Math.sqrt(C / 120), L20 = chin(N20, 20 * N20);
      const lo = Math.floor(bestL * 10) / 10 - 0.05, hi = lo + 1.2;
      const clip = pts.filter((p) => p[1] <= hi);
      const chart = Plot.chart({ x: [6.5, 12.5], y: [lo, hi], W: 560, H: 300, xl: "model size, log₁₀ parameters N  (tokens D = C / 6N)", yl: "loss (illustrative)" },
        [Plot.line(clip, "pl-c1"), Plot.dot(Math.log10(bestN), bestL, "pl-c3", 6), Plot.dot(Math.log10(N20), L20, "pl-c2", 5),
          Plot.text(Math.log10(bestN), bestL + 0.1, "minimum", "", "middle")]);
      return [{ svg: chart, side: { "best N": big(bestN), "its tokens D": big(Dbest), "tokens per param": f2(Dbest / bestN, 1), "loss at the minimum": f2(bestL, 3), "loss at 20 tokens/param": f2(L20, 3) },
        msg: `**Illustrative only.** The curve uses the functional form $L(N,D)=E+A/N^{\\alpha}+B/D^{\\beta}$ with round, made-up constants (the published fits use different ones, and real labs fit their own); the numbers here do not predict any real model. With compute $C\\approx 6ND$ fixed, a model that is too small is limited by its size and one that is too big by too little data, so there is a sweet spot (green). The amber dot is the "20 tokens per parameter" rule of thumb. Slide the budget: the best model size and the amount of data grow together.` }];
    },
  };

  /* ================================================================== static diagrams */
  const ahead = (id) => `<defs><marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker></defs>`;
  const bx = (x, y, w, h, label, cls = "node", sub) => `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="8"/><text class="val" x="${x + w / 2}" y="${y + h / 2 + (sub ? -2 : 5)}" text-anchor="middle" style="font-size:13px">${esc(label)}</text>` + (sub ? `<text class="lbl" x="${x + w / 2}" y="${y + h / 2 + 14}" text-anchor="middle" style="font-size:11px">${esc(sub)}</text>` : "");
  const ar = (x1, y1, x2, y2, id, dash) => `<line class="edge" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#${id})"${dash ? ' style="stroke-dasharray:4 4"' : ""}/>`;
  const tt = (x, y, s, style = "", anchor = "middle") => `<text class="lbl" x="${x}" y="${y}" text-anchor="${anchor}" style="${style}">${esc(s)}</text>`;

  D.llm1_pipeline = () => {
    const id = "l1p";
    let s = ahead(id);
    const cols = [
      ["1. Pretraining", "next-token prediction", "trillions of tokens of raw text", "Base model: fluent, knows a lot,\nbut just continues text", "node hl"],
      ["2. Supervised fine-tuning", "imitate good answers", "thousands to millions of\n(prompt, answer) pairs", "Instruction model: follows\ninstructions, answers in a format", "node"],
      ["3. Preference tuning", "RLHF or DPO", "human or AI judgments:\nanswer A is better than B", "Chat model: helpful, polite,\nsafer, matches preferences", "node ok"],
    ];
    cols.forEach((c, i) => {
      const x = 14 + i * 214;
      s += bx(x, 10, 190, 44, c[0], c[4]) + tt(x + 95, 72, c[1], "font-size:12px;fill:var(--accent);font-weight:600");
      s += tt(x + 95, 96, "data:", "font-size:11px;font-weight:600") + c[2].split("\n").map((l, k) => tt(x + 95, 111 + k * 14, l, "font-size:11px")).join("");
      s += tt(x + 95, 150, "result:", "font-size:11px;font-weight:600") + c[3].split("\n").map((l, k) => tt(x + 95, 165 + k * 14, l, "font-size:11px")).join("");
      if (i < 2) s += ar(x + 192, 32, x + 212, 32, id);
    });
    s += tt(320, 205, "compute: almost all of it goes into step 1; steps 2 and 3 are comparatively tiny", "font-size:11px;font-style:italic");
    return `<svg viewBox="0 0 650 215" width="650" height="215" xmlns="http://www.w3.org/2000/svg" role="img">${s}</svg>`;
  };

  D.llm1_specdec = () => {
    const id = "l1s";
    let s = ahead(id);
    s += tt(10, 18, "1. Draft model proposes 4 tokens (cheap, one at a time)", "font-size:12px;font-weight:600;fill:var(--text)", "start");
    const tk = ["the", "cat", "sat", "on"];
    tk.forEach((t, i) => { s += bx(20 + i * 70, 28, 62, 30, t, "node"); if (i) s += ar(20 + i * 70 - 8, 43, 20 + i * 70, 43, id); });
    s += tt(10, 90, "2. Big model checks all 4 positions in ONE forward pass (parallel, like prefill)", "font-size:12px;font-weight:600;fill:var(--text)", "start");
    const ok = [true, true, true, false], fix = ["the", "cat", "sat", "in"];
    tk.forEach((t, i) => {
      s += bx(20 + i * 70, 100, 62, 30, ok[i] ? t : t, ok[i] ? "node ok" : "node bad") + tt(51 + i * 70, 148, ok[i] ? "accept" : "reject", `font-size:11px;fill:var(--${ok[i] ? "green" : "red"})`);
    });
    s += tt(10, 178, "3. Keep the 3 accepted tokens, plus the big model's own token at the first mismatch:", "font-size:12px;font-weight:600;fill:var(--text)", "start");
    ["the", "cat", "sat", "in"].forEach((t, i) => { s += bx(20 + i * 70, 188, 62, 30, t, i < 3 ? "node ok" : "node hl"); });
    s += tt(310, 208, "4 tokens produced for the price of 1 big-model pass", "font-size:12px;fill:var(--text)", "start");
    s += tt(310, 226, "(worst case: 1 token, same as normal decoding)", "font-size:11px", "start");
    s += tt(310, 50, "The output distribution is exactly the big model's:", "font-size:11px", "start") + tt(310, 66, "a rejection-sampling rule guarantees it.", "font-size:11px", "start");
    return `<svg viewBox="0 0 640 238" width="640" height="238" xmlns="http://www.w3.org/2000/svg" role="img">${s}</svg>`;
  };

  /* ================================================================== glossary */
  Object.assign(window.GLOSSARY, {
    "bigram": "A pair of neighbouring tokens. A bigram language model predicts the next token using only the one token before it.",
    "language model": "A model that assigns a probability to the next token (or to a whole text) given what came before.",
    "perplexity": "exp of the average negative log-likelihood per token. Roughly the number of equally likely tokens the model is choosing between at each step; lower is better.",
    "next-token prediction": "The pretraining task of LLMs: given the tokens so far, predict the next one. Trained with cross-entropy on huge amounts of text.",
    "pretraining": "The first, most expensive training stage: learning to predict the next token on a very large text collection. Produces a base model.",
    "base model": "A model after pretraining only. It continues text fluently but does not reliably follow instructions.",
    "scaling law": "A smooth, predictable relationship (often a power law) between loss and model size, data or compute.",
    "chinchilla": "A 2022 study showing that, for a fixed compute budget, model size and training tokens should grow together, at roughly 20 tokens per parameter.",
    "compute-optimal": "The model size and data amount that reach the lowest loss for a given training compute budget.",
    "sft": "Supervised fine-tuning: training a pretrained model on (prompt, good answer) pairs so it learns to follow instructions.",
    "instruction tuning": "Fine-tuning on many tasks written as instructions with answers, so the model follows new instructions.",
    "peft": "Parameter-efficient fine-tuning: adapt a model by training a tiny number of extra parameters while the big weights stay frozen.",
    "qlora": "LoRA applied on top of a base model stored in 4-bit precision, so fine-tuning fits on much smaller GPUs.",
    "low-rank": "Describes a matrix that can be written as the product of two thin matrices. It has far fewer free numbers than a full matrix.",
    "reward model": "A model trained on human preference comparisons that outputs a single score for how good an answer is.",
    "bradley-terry": "A model of pairwise preferences: the chance that A beats B is the sigmoid of the difference of their scores.",
    "ppo": "Proximal Policy Optimization: a reinforcement-learning algorithm that improves the policy in small, clipped steps. Used in classic RLHF.",
    "dpo": "Direct Preference Optimization: trains on preference pairs with a simple classification-style loss, with no separate reward model or RL loop.",
    "rlaif": "Reinforcement learning from AI feedback: preference labels come from another model instead of humans.",
    "constitutional ai": "Anthropic's approach of guiding a model with a written list of principles; the model critiques and revises its own answers and AI feedback replaces many human labels.",
    "kl penalty": "A term that keeps the tuned model close to a reference model, so optimisation cannot drift into strange text that fools the reward.",
    "reward hacking": "When a model finds ways to get a high reward score that do not match what people actually wanted.",
    "chain of thought": "Writing out intermediate reasoning steps before the final answer. It often improves accuracy on multi-step problems.",
    "test-time compute": "Extra computation spent when answering (longer reasoning, several samples, search) rather than during training.",
    "distillation": "Training a small student model to imitate a bigger teacher, usually by matching the teacher's output probabilities or generated text.",
    "greedy decoding": "Always pick the single most likely next token. Fast and deterministic, but can get stuck in dull or repetitive text.",
    "beam search": "Keeps the best B partial sequences at every step and extends them all, so it can find a sequence with higher total probability than greedy decoding.",
    "top-k": "Sampling only from the k most likely next tokens, after renormalising their probabilities.",
    "top-p": "Nucleus sampling: sample only from the smallest set of top tokens whose probabilities add up to p.",
    "nucleus sampling": "Another name for top-p sampling.",
    "prefill": "The first phase of generation: the whole prompt is processed in parallel to fill the KV cache. Compute-bound.",
    "decode": "The second phase of generation: one token at a time, reading the cache. Usually limited by memory bandwidth.",
    "continuous batching": "A serving technique that lets new requests join and finished ones leave a running batch at every step, keeping the GPU busy.",
    "speculative decoding": "A small draft model proposes several tokens and the big model verifies them in one pass, giving the same output distribution faster.",
    "paged attention": "Storing the KV cache in small fixed-size blocks, like virtual memory pages, to avoid wasted GPU memory.",
    "int8": "8-bit integer storage of numbers (256 possible values), commonly used to shrink model weights.",
    "scale factor": "The number that converts between integer codes and real values in quantization: real is about code times scale.",
    "absmax": "Quantization that sets the scale from the largest absolute value in a tensor or block.",
    "gptq": "A post-training quantization method that rounds weights one group at a time while correcting the remaining weights to reduce the output error.",
    "awq": "Activation-aware weight quantization: protects the weights that matter most for the activations before rounding.",
    "long context": "Models and techniques that handle very long inputs (hundreds of thousands of tokens or more).",
    "lost in the middle": "The tendency of models to use information at the start and end of a long prompt better than information in the middle.",
    "ntk scaling": "A family of tricks that stretch RoPE frequencies so a model can handle sequences longer than it was trained on.",
  });
})();
