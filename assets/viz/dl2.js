/* Deep Learning, part 2 (architectures + in practice): visualizers, diagrams and glossary terms.
   Loaded on every doc page and on Practice, after assets/viz.js. Every number shown is computed, not faked. */
(function () {
  const V = Viz.defs, Plot = Viz.Plot, { r3 } = Viz.helpers;

  /* ------------------------------------------------------------------ private helpers */
  const rng = (seed) => { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  const gauss = (r) => { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const fx = (v, d = 2) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : +v).toFixed(d);
  const pr = (v, d = 0) => (v < 0 ? `(${fx(v, d)})` : fx(v, d));
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const soft = (v, p = 16) => `color-mix(in srgb, var(--${v}) ${p}%, transparent)`;
  const HALO = "paint-order:stroke;stroke:var(--panel);stroke-width:4px;stroke-linejoin:round;";
  const svgWrap = (W, H, inner) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">${inner}</svg>`;
  const T = (x, y, s, style = "", anchor = "middle") => `<text x="${x}" y="${y}" text-anchor="${anchor}" style="font-size:12px;fill:var(--muted);${style}">${esc(s)}</text>`;
  // a grid of numbers: vals[r][c], top-left (x0, y0), cell size cs. fill(v, r, c) -> css colour or null, cls(r, c) -> extra style
  function numGrid(vals, x0, y0, cs, fmt, fill, extra) {
    let s = "";
    vals.forEach((row, r) => row.forEach((v, c) => {
      const f = fill ? fill(v, r, c) : null, ex = extra ? extra(r, c) : null;
      s += `<rect x="${x0 + c * cs}" y="${y0 + r * cs}" width="${cs}" height="${cs}" style="fill:${f || "var(--panel)"};stroke:${ex && ex.stroke ? ex.stroke : "var(--border)"};stroke-width:${ex && ex.sw ? ex.sw : 1};${ex && ex.dash ? "stroke-dasharray:3 3;" : ""}"/>`;
      if (v !== null && v !== undefined) s += `<text x="${x0 + c * cs + cs / 2}" y="${y0 + r * cs + cs / 2 + 4}" text-anchor="middle" style="font-size:${cs < 28 ? 10.5 : 12.5}px;font-weight:600;fill:${ex && ex.dim ? "var(--muted)" : "var(--text)"}">${fmt(v)}</text>`;
    }));
    return s;
  }
  const mix = (v, lo, hi, col = "accent", max = 52) => `color-mix(in srgb, var(--${col}) ${Math.round(6 + (max * (v - lo)) / (hi - lo || 1))}%, var(--panel))`;
  const signMix = (v, m) => (v >= 0 ? `color-mix(in srgb, var(--accent) ${Math.round(6 + (48 * v) / (m || 1))}%, var(--panel))` : `color-mix(in srgb, var(--red) ${Math.round(6 + (48 * -v) / (m || 1))}%, var(--panel))`);

  /* ================================================================== 1. Convolution: a kernel slides over an image */
  const IMG = [[1, 1, 1, 1, 1, 1], [1, 9, 9, 9, 9, 1], [1, 1, 1, 9, 9, 1], [1, 1, 9, 9, 1, 1], [1, 9, 9, 1, 1, 1], [1, 1, 1, 1, 1, 1]];
  const KERS = [
    { name: "vertical edges", k: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], div: 1, note: "it subtracts the left column from the right column, so it lights up where the brightness changes from left to right" },
    { name: "all edges", k: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], div: 1, note: "centre pixel minus its eight neighbours: zero on flat areas, large at outlines" },
    { name: "blur", k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]], div: 9, note: "the average of the nine pixels, so everything gets smoother" },
    { name: "sharpen", k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], div: 1, note: "5 times the centre minus its four neighbours: it boosts the centre against its surroundings" },
  ];
  V.dl2_conv = {
    title: "Convolution: slide a 3x3 kernel over the image",
    params: [
      { id: "ker", label: "kernel", min: 0, max: 3, step: 1, value: 0, fmt: (v) => KERS[+v].name },
      { id: "stride", label: "stride", min: 1, max: 3, step: 1, value: 1 },
      { id: "pad", label: "padding", min: 0, max: 1, step: 1, value: 0 },
    ],
    gen(_, P) {
      const K = KERS[+P.ker], s = +P.stride, p = +P.pad, n = 6, m = n + 2 * p;
      const pad = Array.from({ length: m }, (_, r) => Array.from({ length: m }, (_, c) => (r >= p && r < p + n && c >= p && c < p + n ? IMG[r - p][c - p] : 0)));
      const o = Math.floor((m - 3) / s) + 1;
      const out = Array.from({ length: o }, () => Array(o).fill(null));
      const cs = m > 7 ? 29 : 32, gx = 14, gy = 64, kx = 14 + m * cs + 38, ox = kx + 3 * cs + 46;
      const W = ox + o * 32 + 14, H = Math.max(m * cs, o * 32) + gy + 40;
      const val = (r, c) => { let t = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) t += pad[r * s + i][c * s + j] * K.k[i][j]; return t / K.div; };
      const allV = []; for (let r = 0; r < o; r++) for (let c = 0; c < o; c++) allV.push(val(r, c));
      const mabs = Math.max(1, ...allV.map(Math.abs));
      const fmtO = (v) => (K.div === 1 ? String(Math.round(v)) : fx(v, 1));
      const S = [];
      const frame = (k, msg) => {
        const cur = k >= 0 ? [Math.floor(k / o), k % o] : null;
        for (let q = 0; q <= k; q++) out[Math.floor(q / o)][q % o] = allV[q];
        for (let q = k + 1; q < o * o; q++) out[Math.floor(q / o)][q % o] = null;
        let g = T(gx + (m * cs) / 2, gy - 24, p ? "input, padded with zeros" : "input image", "font-weight:700;fill:var(--text)");
        g += numGrid(pad, gx, gy, cs, (v) => v, (v, r, c) => (r >= p && r < p + n && c >= p && c < p + n ? mix(v, 0, 9) : "var(--panel-2)"), (r, c) => ({ dim: !(r >= p && r < p + n && c >= p && c < p + n), dash: !(r >= p && r < p + n && c >= p && c < p + n) }));
        if (cur) g += `<rect x="${gx + cur[1] * s * cs}" y="${gy + cur[0] * s * cs}" width="${3 * cs}" height="${3 * cs}" style="fill:${soft("amber", 14)};stroke:var(--amber);stroke-width:2.6"/>`;
        g += T(kx + 1.5 * cs, gy - 24, "kernel (weights)", "font-weight:700;fill:var(--text)");
        const ky = gy + (m * cs - 3 * cs) / 2;
        g += numGrid(K.k, kx, ky, cs, (v) => v, (v) => signMix(v, 8));
        if (K.div !== 1) g += T(kx + 1.5 * cs, ky + 3 * cs + 18, "each weight is 1, then divide by 9", "font-size:11px");
        g += `<text x="${kx - 19}" y="${ky + 1.5 * cs + 6}" text-anchor="middle" style="font-size:20px;fill:var(--muted)">×</text><text x="${ox - 23}" y="${ky + 1.5 * cs + 6}" text-anchor="middle" style="font-size:20px;fill:var(--muted)">=</text>`;
        const oy = gy + (m * cs - o * 32) / 2;
        g += T(ox + (o * 32) / 2, gy - 24, `output ${o}×${o}`, "font-weight:700;fill:var(--text)");
        g += numGrid(out, ox, oy, 32, fmtO, (v) => (v === null ? "var(--panel-2)" : signMix(v, mabs)), (r, c) => (cur && cur[0] === r && cur[1] === c ? { stroke: "var(--amber)", sw: 2.8 } : null));
        return { svg: svgWrap(W, H, g), side: { "output size": `${o}×${o}`, "stride": s, "padding": p, "positions done": `${Math.max(0, k + 1)} / ${o * o}` }, msg };
      };
      S.push(frame(-1, `A **convolution** slides a small grid of weights, the **kernel**, over the image. At each position it multiplies the 3×3 patch of pixels by the kernel, element by element, and adds everything up: that one number becomes one cell of the output. Current kernel: **${K.name}**: ${K.note}. With stride ${s} and padding ${p} the output is ${o}×${o}, from $\\lfloor (${n}+2\\cdot${p}-3)/${s}\\rfloor+1 = ${o}$. Press Next to compute the cells one by one (or change the sliders).`));
      for (let q = 0; q < o * o; q++) {
        const r = Math.floor(q / o), c = q % o;
        const terms = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) terms.push(`${pad[r * s + i][c * s + j]}·${pr(K.k[i][j])}`);
        const raw = allV[q] * K.div;
        let msg = `Output cell (${r + 1}, ${c + 1}): the window sits at rows ${r * s + 1}–${r * s + 3}, columns ${c * s + 1}–${c * s + 3} of the ${p ? "padded " : ""}input. Multiply and add: \`${terms.join(" + ")}\` = **${raw}**${K.div === 1 ? "" : `, divided by 9 = **${fx(allV[q], 2)}**`}.`;
        if (q === o * o - 1) msg += ` That was the last position. Same kernel everywhere (**weight sharing**), so this layer has only 9 weights no matter how big the image is.`;
        S.push(frame(q, msg));
      }
      return S;
    },
  };

  /* ================================================================== 2. Pooling */
  const FMAP = [[1, 3, 2, 1, 0, 2], [4, 6, 5, 2, 1, 1], [2, 1, 7, 8, 3, 0], [0, 2, 9, 4, 2, 5], [1, 0, 3, 3, 6, 7], [5, 2, 1, 2, 8, 4]];
  V.dl2_pool = {
    title: "Pooling: keep the strongest response in each window",
    params: [
      { id: "size", label: "window (stride = window)", min: 2, max: 3, step: 1, value: 2 },
      { id: "type", label: "type", min: 0, max: 1, step: 1, value: 0, fmt: (v) => (+v ? "average" : "max") },
    ],
    gen(_, P) {
      const w = +P.size, avg = +P.type === 1, n = 6, o = n / w, cs = 36, gx = 14, gy = 52, ox = gx + n * cs + 70, W = ox + o * 44 + 20, H = n * cs + gy + 24;
      const val = (r, c) => { const a = []; for (let i = 0; i < w; i++) for (let j = 0; j < w; j++) a.push(FMAP[r * w + i][c * w + j]); return avg ? a.reduce((x, y) => x + y, 0) / a.length : Math.max(...a); };
      const out = Array.from({ length: o }, () => Array(o).fill(null));
      const S = [];
      const frame = (k, msg) => {
        const cur = k >= 0 ? [Math.floor(k / o), k % o] : null;
        for (let q = 0; q < o * o; q++) out[Math.floor(q / o)][q % o] = q <= k ? val(Math.floor(q / o), q % o) : null;
        let g = T(gx + (n * cs) / 2, gy - 22, "feature map 6×6", "font-weight:700;fill:var(--text)");
        g += numGrid(FMAP, gx, gy, cs, (v) => v, (v) => mix(v, 0, 9));
        if (cur) {
          g += `<rect x="${gx + cur[1] * w * cs}" y="${gy + cur[0] * w * cs}" width="${w * cs}" height="${w * cs}" style="fill:${soft("amber", 14)};stroke:var(--amber);stroke-width:2.6"/>`;
          if (!avg) { let best = -1, bi = 0, bj = 0; for (let i = 0; i < w; i++) for (let j = 0; j < w; j++) { const v = FMAP[cur[0] * w + i][cur[1] * w + j]; if (v > best) { best = v; bi = i; bj = j; } } g += `<rect x="${gx + (cur[1] * w + bj) * cs + 2}" y="${gy + (cur[0] * w + bi) * cs + 2}" width="${cs - 4}" height="${cs - 4}" style="fill:none;stroke:var(--green);stroke-width:2.6"/>`; }
        }
        g += `<text x="${gx + n * cs + 35}" y="${gy + (n * cs) / 2 + 5}" text-anchor="middle" style="font-size:22px;fill:var(--muted)">→</text>`;
        g += T(ox + (o * 44) / 2, gy - 22, `${avg ? "average" : "max"} pool → ${o}×${o}`, "font-weight:700;fill:var(--text)");
        const oy = gy + (n * cs - o * 44) / 2;
        g += numGrid(out, ox, oy, 44, (v) => (avg ? fx(v, 1) : v), (v) => (v === null ? "var(--panel-2)" : mix(v, 0, 9, "green")), (r, c) => (cur && cur[0] === r && cur[1] === c ? { stroke: "var(--amber)", sw: 2.8 } : null));
        return { svg: svgWrap(W, H, g), side: { window: `${w}×${w}`, stride: w, "output size": `${o}×${o}`, "numbers kept": `${o * o} of ${n * n}` }, msg };
      };
      S.push(frame(-1, `**Pooling** shrinks a feature map by summarising each ${w}×${w} window with a single number. **Max pooling** keeps the largest value ("was this feature seen anywhere in this window?"); **average pooling** keeps the mean. Unlike a convolution it has **no weights to learn**. The ${w}×${w} windows do not overlap (stride ${w}), so ${n}×${n} becomes ${o}×${o}. Press Next.`));
      for (let q = 0; q < o * o; q++) {
        const r = Math.floor(q / o), c = q % o, a = []; for (let i = 0; i < w; i++) for (let j = 0; j < w; j++) a.push(FMAP[r * w + i][c * w + j]);
        S.push(frame(q, `Window ${q + 1}: the values are \`${a.join(", ")}\`. ${avg ? `Their average is ${a.reduce((x, y) => x + y, 0)} / ${a.length} = **${fx(val(r, c), 2)}**.` : `The largest is **${val(r, c)}** (green box): only the strongest response survives, so a feature that moves by one pixel inside the window gives the same output.`}`));
      }
      return S;
    },
  };

  /* ================================================================== 3. Receptive field growth */
  V.dl2_receptive = {
    title: "Receptive field: what one deep unit can see",
    params: [
      { id: "layers", label: "conv layers", min: 1, max: 4, step: 1, value: 3 },
      { id: "k", label: "kernel size", min: 3, max: 5, step: 2, value: 3 },
      { id: "s", label: "stride", min: 1, max: 2, step: 1, value: 1 },
    ],
    gen(_, P) {
      const L = +P.layers, k = +P.k, s = +P.s;
      const jump = [1], rf = [1];
      for (let l = 1; l <= 4; l++) { rf[l] = rf[l - 1] + (k - 1) * jump[l - 1]; jump[l] = jump[l - 1] * s; }
      const base = Math.max(rf[4] + 2, 13), Nn = base % 2 ? base : base + 1, c = (Nn - 1) / 2;
      const W = 600, cw = (W - 80) / Nn, x0 = 70, rowH = 62, H = (L + 1) * rowH + 8;
      const px = (pos) => x0 + (pos + c) * cw + cw / 2;      // pos is an offset from the centre
      const rows = [];                                                // rows[l] = list of offsets drawn; dep[l] = Set of offsets in the dependency tree
      const dep = [];
      dep[L] = new Set([0]);
      for (let l = L; l >= 1; l--) { dep[l - 1] = new Set(); dep[l].forEach((u) => { for (let d = 0; d < k; d++) dep[l - 1].add(u + (d - (k - 1) / 2) * jump[l - 1]); }); }
      for (let l = 0; l <= 4; l++) { rows[l] = []; for (let m = -Math.floor(c / jump[l]); m <= Math.floor(c / jump[l]); m++) rows[l].push(m * jump[l]); }
      const y = (l) => 22 + (L - l) * rowH;
      let g = "";
      for (let l = L; l >= 1; l--) dep[l].forEach((u) => { for (let d = 0; d < k; d++) { const ch = u + (d - (k - 1) / 2) * jump[l - 1]; g += `<line x1="${px(u)}" y1="${y(l) + 8}" x2="${px(ch)}" y2="${y(l - 1) - 8}" style="stroke:var(--accent);stroke-width:1;opacity:.55"/>`; } });
      for (let l = 0; l <= 4; l++) {
        const on = l <= L;
        g += `<text x="12" y="${y(l) + 4}" style="font-size:12px;font-weight:700;fill:${on ? "var(--text)" : "var(--muted)"}">${l === 0 ? "input" : "layer " + l}</text>`;
        rows[l].forEach((u) => {
          const hit = on && dep[l].has(u), top = l === L && u === 0, sz = Math.min(11, cw - 2);
          if (l > L) return;
          g += `<rect x="${px(u) - sz / 2}" y="${y(l) - sz / 2}" width="${sz}" height="${sz}" rx="2" style="fill:${top ? "var(--amber)" : hit ? "var(--accent)" : "var(--panel-2)"};stroke:${hit ? "var(--accent)" : "var(--border)"};stroke-width:1"/>`;
        });
      }
      g += `<text x="${W - 8}" y="${H - 8}" text-anchor="end" style="font-size:12px;fill:var(--muted)">one axis of the image shown; the picture is the same in the other direction</text>`;
      return [{
        svg: svgWrap(W, H, g),
        side: { "layers": L, "kernel": `${k}×${k}`, "stride": s, "receptive field": `${rf[L]}×${rf[L]} pixels`, "weights per layer": `${k * k} (per channel pair)` },
        msg: `The amber square is **one unit** in layer ${L}. Each blue square is a unit that can influence it. A unit in layer 1 sees ${k} input pixels; a unit in layer 2 sees ${k} units of layer 1, which together cover ${rf[2] > 0 ? rf[2] : "-"} pixels, and so on. After ${L} layer${L > 1 ? "s" : ""} its **receptive field** is **${rf[L]} pixels** wide, using the recurrence $r_l = r_{l-1} + (k-1)\\cdot j_{l-1}$ where $j$ (the "jump") doubles at every stride-2 layer. Try stride 2: the field grows much faster, which is why CNNs downsample.`,
      }];
    },
  };

  /* ================================================================== 4. RNN unrolled through time */
  const WORDS = { the: [0.2, 0.1], a: [0.2, 0.0], cat: [1.0, 0.3], dog: [0.9, 0.4], sat: [0.1, 1.0], ran: [0.2, 0.9], mat: [0.8, -0.5], not: [-1.0, 0.0], good: [0.6, 0.6], bad: [-0.6, -0.6], very: [0.3, -0.3], movie: [0.7, 0.1], i: [0.0, 0.2], love: [1.0, 1.0], hate: [-1.0, -1.0] };
  const wordVec = (w) => { if (WORDS[w]) return WORDS[w]; let h = 7; for (const ch of w) h = (h * 31 + ch.charCodeAt(0)) % 997; const r = rng(h); return [+(r() * 2 - 1).toFixed(1), +(r() * 2 - 1).toFixed(1)]; };
  V.dl2_rnn = {
    title: "An RNN unrolled through time: the hidden state is the memory",
    input: "the cat sat on mat",
    help: "up to 6 words (unknown words get a made-up 2-number vector)",
    params: [{ id: "mem", label: "recurrent strength (scale of W_h)", min: 0, max: 1.5, step: 0.1, value: 1 }],
    gen(inp, P) {
      const words = inp.toLowerCase().split(/\s+/).filter(Boolean).slice(0, 6);
      if (!words.length) words.push("the");
      const Wx = [[0.8, -0.4], [0.3, 0.9]], Wh0 = [[0.6, -0.5], [0.5, 0.6]], b = [0.1, -0.1], a = +P.mem;
      const Wh = Wh0.map((r) => r.map((v) => v * a));
      const xs = words.map(wordVec), hs = [[0, 0]], zs = [];
      xs.forEach((x, t) => {
        const hp = hs[t], z = [0, 1].map((i) => Wx[i][0] * x[0] + Wx[i][1] * x[1] + Wh[i][0] * hp[0] + Wh[i][1] * hp[1] + b[i]);
        zs.push(z); hs.push(z.map(Math.tanh));
      });
      const n = words.length, cw = 88, x0 = 56, W = x0 + n * cw + 120, H = 330;
      const S = [];
      const frame = (k, msg) => {
        let g = `<defs><marker id="rnA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker></defs>`;
        g += T(W / 2, 20, "the same weights W_x, W_h, b are reused at every step", "font-weight:700;fill:var(--accent)");
        for (let t = 0; t <= n; t++) {
          const cx = x0 + t * cw, done = t <= k, cur = t === k && t > 0, hv = hs[t];
          const st = done ? (cur ? "fill:" + soft("amber", 22) + ";stroke:var(--amber);stroke-width:2.4" : "fill:var(--node);stroke:var(--node-stroke);stroke-width:2") : "fill:var(--panel-2);stroke:var(--border);stroke-width:1.5;stroke-dasharray:4 3";
          g += `<rect x="${cx - 36}" y="110" width="72" height="84" rx="10" style="${st}"/>`;
          g += `<text x="${cx}" y="128" text-anchor="middle" style="font-size:13px;font-weight:700;fill:var(--text)">h<tspan dy="3" style="font-size:9px">${t}</tspan></text>`;
          if (done) {
            [0, 1].forEach((i) => {
              const by = 143 + i * 24, v = hv[i], half = 24;
              g += `<line x1="${cx}" y1="${by - 8}" x2="${cx}" y2="${by + 8}" style="stroke:var(--muted);stroke-width:1"/>`;
              g += `<rect x="${v >= 0 ? cx : cx + v * half}" y="${by - 5}" width="${Math.abs(v) * half}" height="10" style="fill:${v >= 0 ? "var(--accent)" : "var(--red)"}"/>`;
              g += `<text x="${cx + (v >= 0 ? -3 : 3)}" y="${by + 3.5}" text-anchor="${v >= 0 ? "end" : "start"}" style="font-size:10.5px;fill:var(--text);font-weight:600">${fx(v, 2)}</text>`;
            });
          } else g += `<text x="${cx}" y="160" text-anchor="middle" style="font-size:12px;fill:var(--muted)">not yet</text>`;
          if (t > 0) {
            g += `<line x1="${cx - cw + 38}" y1="152" x2="${cx - 38}" y2="152" style="stroke:var(--${done ? "accent" : "border"});stroke-width:2.4" marker-end="url(#rnA)"/>`;
            const x = xs[t - 1];
            g += `<line x1="${cx}" y1="${244}" x2="${cx}" y2="${198}" style="stroke:var(--${done ? "accent" : "border"});stroke-width:2.2" marker-end="url(#rnA)"/>`;
            g += `<rect x="${cx - 34}" y="246" width="68" height="26" rx="6" style="fill:var(--node);stroke:var(--node-stroke);stroke-width:1.5"/><text x="${cx}" y="263" text-anchor="middle" style="font-size:11.5px;font-weight:600;fill:var(--text)">[${fx(x[0], 1)}, ${fx(x[1], 1)}]</text>`;
            g += `<text x="${cx}" y="292" text-anchor="middle" style="font-size:13px;font-weight:700;fill:var(--text)">${esc(words[t - 1])}</text><text x="${cx}" y="308" text-anchor="middle" style="font-size:11px;fill:var(--muted)">x<tspan dy="3" style="font-size:8px">${t}</tspan></text>`;
          } else g += `<text x="${cx}" y="213" text-anchor="middle" style="font-size:11px;fill:var(--muted)">start: zeros</text>`;
        }
        g += `<line x1="${x0 + (n) * cw + 38}" y1="152" x2="${x0 + n * cw + 62}" y2="152" style="stroke:var(--muted);stroke-width:1.6;stroke-dasharray:4 3" marker-end="url(#rnA)"/><text x="${x0 + n * cw + 54}" y="140" text-anchor="middle" style="font-size:11px;fill:var(--muted)">→ output</text>`;
        return { svg: svgWrap(W, H, g), side: { step: k === 0 ? "start" : `t = ${k}`, "W_h scale": a, "h now": k >= 0 ? `[${fx(hs[k][0], 2)}, ${fx(hs[k][1], 2)}]` : "-" }, msg };
      };
      S.push(frame(0, `A recurrent network reads a sequence one item at a time and keeps a small **hidden state** $h$ as its memory. We start with $h_0 = [0, 0]$. Each word is turned into a 2-number vector $x_t$ (made up here, in real models it is a learned [embedding](transformers.html#token-embeddings)). Press Next to run step 1. The "recurrent strength" slider scales $W_h$, the matrix that carries the old memory forward: at 0 the network has no memory at all.`));
      xs.forEach((x, t) => {
        const hp = hs[t], z = zs[t];
        S.push(frame(t + 1, `**Step ${t + 1}, word "${words[t]}".** $h_${t + 1} = \\tanh(W_x x_${t + 1} + W_h h_${t} + b)$. First unit: \`0.8·${pr(x[0], 1)} + (-0.4)·${pr(x[1], 1)} + ${fx(Wh[0][0], 2)}·${pr(hp[0], 2)} + ${pr(Wh[0][1], 2)}·${pr(hp[1], 2)} + 0.1\` = ${fx(z[0], 3)}, and tanh gives **${fx(hs[t + 1][0], 3)}**. Second unit: z = ${fx(z[1], 3)} → **${fx(hs[t + 1][1], 3)}**. The new state mixes the new word with the old memory, then is passed to the next step.${t === n - 1 ? " The last state $h_" + n + "$ is a summary of the whole sentence; a classifier could read it." : ""}`));
      });
      return S;
    },
  };

  /* ================================================================== 5. LSTM gates for one step */
  V.dl2_lstm = {
    title: "One LSTM step: forget, input and output gates with numbers",
    params: [
      { id: "x", label: "input x", min: -2, max: 2, step: 0.1, value: 1 },
      { id: "h", label: "previous hidden h", min: -1, max: 1, step: 0.1, value: 0.5 },
      { id: "c", label: "previous cell c", min: -2, max: 2, step: 0.1, value: 1 },
    ],
    gen(_, P) {
      const x = +P.x, hp = +P.h, cp = +P.c;
      const Wf = [0.5, 0.4, 1.0], Wi = [0.8, 0.3, 0.0], Wg = [1.0, 0.5, 0.0], Wo = [0.6, 0.2, 0.0];
      const lin = (w) => w[0] * x + w[1] * hp + w[2];
      const zf = lin(Wf), zi = lin(Wi), zg = lin(Wg), zo = lin(Wo);
      const f = sig(zf), i = sig(zi), g = Math.tanh(zg), o = sig(zo);
      const keep = f * cp, write = i * g, c = keep + write, tc = Math.tanh(c), h = o * tc;
      const W = 640, H = 340, S = [];
      const frame = (k, msg) => {
        const ln = (x1, y1, x2, y2, on, arrow) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" style="stroke:var(--${on ? "accent" : "border"});stroke-width:${on ? 2.4 : 1.8}" ${arrow ? 'marker-end="url(#lsA)"' : ""}/>`;
        const op = (cx, cy, sym, on) => `<circle cx="${cx}" cy="${cy}" r="15" style="fill:${on ? soft("amber", 28) : "var(--panel-2)"};stroke:${on ? "var(--amber)" : "var(--border)"};stroke-width:2"/><text x="${cx}" y="${cy + 6}" text-anchor="middle" style="font-size:18px;font-weight:700;fill:var(--text)">${sym}</text>`;
        const gate = (cx, name, act, val, z, on) => `<rect x="${cx - 40}" y="188" width="80" height="62" rx="9" style="fill:${on ? "var(--node)" : "var(--panel-2)"};stroke:${on ? "var(--accent)" : "var(--border)"};stroke-width:2"/><text x="${cx}" y="206" text-anchor="middle" style="font-size:12px;font-weight:700;fill:var(--text)">${name}</text><text x="${cx}" y="222" text-anchor="middle" style="font-size:10.5px;fill:var(--muted)">${act}${on ? "(" + fx(z, 2) + ")" : ""}</text><text x="${cx}" y="241" text-anchor="middle" style="font-size:14px;font-weight:700;fill:var(--accent)">${on ? fx(val, 3) : "?"}</text>`;
        let s = `<defs><marker id="lsA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker></defs>`;
        const cellOn = k >= 1;
        // cell state highway
        s += ln(60, 70, 590, 70, true, true);
        s += `<text x="22" y="66" text-anchor="middle" style="font-size:12px;font-weight:700;fill:var(--text)">c<tspan dy="3" style="font-size:8px">t-1</tspan></text><text x="22" y="86" text-anchor="middle" style="font-size:12px;fill:var(--accent);font-weight:700">${fx(cp, 2)}</text>`;
        s += `<text x="612" y="66" text-anchor="middle" style="font-size:12px;font-weight:700;fill:var(--text)">c<tspan dy="3" style="font-size:8px">t</tspan></text><text x="612" y="86" text-anchor="middle" style="font-size:12px;font-weight:700;fill:var(--accent)">${k >= 3 ? fx(c, 3) : "?"}</text>`;
        s += T(330, 28, "cell state: the long-term memory \"conveyor belt\"", "font-weight:700;fill:var(--accent)");
        // forget
        s += ln(110, 188, 110, 94, k >= 1, true) + op(110, 70, "×", k >= 1);
        if (k >= 1) s += `<text x="110" y="52" text-anchor="middle" style="font-size:11.5px;font-weight:700;fill:var(--text);${HALO}">keep ${fx(keep, 3)}</text>`;
        // input * candidate then add
        s += ln(250, 188, 250, 150, k >= 2) + ln(360, 188, 360, 150, k >= 2) + ln(250, 150, 305, 150, k >= 2) + ln(360, 150, 305, 150, k >= 2);
        s += op(305, 128, "×", k >= 2) + ln(305, 113, 305, 94, k >= 2, true) + op(305, 70, "+", k >= 3);
        if (k >= 2) s += `<text x="344" y="124" style="font-size:11.5px;font-weight:700;fill:var(--text);${HALO}">write ${fx(write, 3)}</text>`;
        // output path
        s += ln(440, 70, 440, 118, k >= 4, false) + `<rect x="418" y="96" width="44" height="26" rx="6" style="fill:${k >= 4 ? "var(--node)" : "var(--panel-2)"};stroke:${k >= 4 ? "var(--accent)" : "var(--border)"};stroke-width:2"/><text x="440" y="114" text-anchor="middle" style="font-size:12px;font-weight:700;fill:var(--text)">tanh</text>`;
        s += ln(462, 109, 505, 109, k >= 4) + op(520, 109, "×", k >= 4) + ln(520, 188, 520, 124, k >= 4, true) + ln(535, 109, 590, 109, k >= 4, true);
        s += `<text x="612" y="106" text-anchor="middle" style="font-size:12px;font-weight:700;fill:var(--text)">h<tspan dy="3" style="font-size:8px">t</tspan></text><text x="612" y="126" text-anchor="middle" style="font-size:12px;font-weight:700;fill:var(--accent)">${k >= 4 ? fx(h, 3) : "?"}</text>`;
        if (k >= 4) s += `<text x="440" y="140" text-anchor="middle" style="font-size:10.5px;fill:var(--muted)">tanh(c) = ${fx(tc, 3)}</text>`;
        // gates
        s += gate(110, "forget gate", "σ", f, zf, k >= 1) + gate(250, "input gate", "σ", i, zi, k >= 2) + gate(360, "candidate", "tanh", g, zg, k >= 2) + gate(520, "output gate", "σ", o, zo, k >= 4);
        // inputs bus
        s += ln(60, 290, 580, 290, true, false);
        [110, 250, 360, 520].forEach((cx) => (s += ln(cx, 290, cx, 252, true, true)));
        s += `<text x="320" y="316" text-anchor="middle" style="font-size:13px;font-weight:700;fill:var(--text)">inputs to every gate: x<tspan dy="3" style="font-size:9px">t</tspan><tspan dy="-3"> = ${fx(x, 1)}   and   h</tspan><tspan dy="3" style="font-size:9px">t-1</tspan><tspan dy="-3"> = ${fx(hp, 1)}</tspan></text>`;
        return { svg: svgWrap(W, H, s), side: { "forget f": k >= 1 ? fx(f, 3) : "-", "input i": k >= 2 ? fx(i, 3) : "-", "candidate g̃": k >= 2 ? fx(g, 3) : "-", "output o": k >= 4 ? fx(o, 3) : "-", "c_t": k >= 3 ? fx(c, 3) : "-", "h_t": k >= 4 ? fx(h, 3) : "-" }, msg };
      };
      S.push(frame(0, `An **LSTM** keeps two things: a hidden state $h$ and a **cell state** $c$, a memory lane that only gets small, gated edits. Three **gates** (numbers between 0 and 1, from a sigmoid) decide how much to keep, write and reveal. Here everything is a single number so you can follow the arithmetic: inputs $x_t=${fx(x, 1)}$, $h_{t-1}=${fx(hp, 1)}$, $c_{t-1}=${fx(cp, 1)}$. Drag the sliders, then press Next.`));
      S.push(frame(1, `**Forget gate.** $f = \\sigma(${Wf[0]}\\cdot${pr(x, 1)} + ${Wf[1]}\\cdot${pr(hp, 1)} + ${Wf[2]}) = \\sigma(${fx(zf, 2)}) = ${fx(f, 3)}$. It multiplies the old memory: $f\\cdot c_{t-1} = ${fx(f, 3)}\\cdot${pr(cp, 2)} = ${fx(keep, 3)}$. A value near 1 means "remember this", near 0 means "forget it". (The forget bias starts at +1 so the cell remembers by default.)`));
      S.push(frame(2, `**Input gate and candidate.** The candidate memory is $\\tilde c = \\tanh(${fx(zg, 2)}) = ${fx(g, 3)}$ (what we could write) and the input gate is $i = \\sigma(${fx(zi, 2)}) = ${fx(i, 3)}$ (how much of it to allow). What gets written is $i\\cdot\\tilde c = ${fx(write, 3)}$.`));
      S.push(frame(3, `**Cell update.** Add the kept old memory and the new write: $c_t = f\\cdot c_{t-1} + i\\cdot\\tilde c = ${fx(keep, 3)} + ${pr(write, 3)} = ${fx(c, 3)}$. This is an addition, not a squashing function, so gradients can flow back along this lane with little shrinking: that is how the LSTM fights [vanishing gradients](dl.html#vanishing-gradients).`));
      S.push(frame(4, `**Output gate.** $o = \\sigma(${fx(zo, 2)}) = ${fx(o, 3)}$ decides how much of the memory to reveal: $h_t = o\\cdot\\tanh(c_t) = ${fx(o, 3)}\\cdot${fx(tc, 3)} = ${fx(h, 3)}$. The pair $(h_t, c_t)$ goes on to the next time step; $h_t$ is also what the next layer or classifier sees. Now go back and move a slider: set the previous cell very large and see how the tanh limits $h_t$.`));
      return S;
    },
  };

  /* ================================================================== 6. Word embeddings in 2-D */
  const EMB = {
    man: [-1.0, 0.2], woman: [1.0, 0.1], king: [-1.1, 2.1], queen: [0.9, 2.2], boy: [-0.9, -0.9], girl: [1.1, -1.0], prince: [-1.0, 1.3], princess: [1.0, 1.2],
    uncle: [-1.4, 0.7], aunt: [1.4, 0.6],
    paris: [-3.2, 2.6], france: [-3.7, 1.8], rome: [-3.0, 3.6], italy: [-3.5, 2.9],
    apple: [3.6, -1.6], banana: [3.9, -2.3], orange: [3.3, -2.1], car: [-3.2, -2.4], truck: [-3.8, -1.8], bus: [-2.9, -3.1],
  };
  V.dl2_embed = {
    title: "Word embeddings in 2-D: directions carry meaning",
    input: "king - man + woman",
    help: "words from the map joined by + and - (e.g. paris - france + italy)",
    gen(inp) {
      const toks = inp.toLowerCase().replace(/([+-])/g, " $1 ").split(/\s+/).filter(Boolean);
      let sign = 1, vec = [0, 0], used = [], bad = [], seq = [];
      toks.forEach((t) => { if (t === "+") sign = 1; else if (t === "-") sign = -1; else if (EMB[t]) { vec = [vec[0] + sign * EMB[t][0], vec[1] + sign * EMB[t][1]]; used.push(t); seq.push([sign, t]); sign = 1; } else bad.push(t); });
      const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
      const ranked = Object.keys(EMB).filter((w) => !used.includes(w)).map((w) => [w, dist(vec, EMB[w])]).sort((a, b) => a[1] - b[1]);
      const best = ranked[0], nn = ranked.slice(0, 3);
      const pts = Object.entries(EMB).map(([w, p]) => [p[0], p[1], used.includes(w) ? "b" : "a"]);
      const layers = [
        (S) => { let s = ""; // arrows tracing the arithmetic from the first word
          let cur = [0, 0]; seq.forEach(([sg, w], idx) => { const nxt = [cur[0] + sg * EMB[w][0], cur[1] + sg * EMB[w][1]]; s += Plot.seg(cur[0], cur[1], nxt[0], nxt[1], sg > 0 ? "pl-c3" : "pl-c4", true)(S); cur = nxt; }); return s; },
        Plot.pts(pts, "pl-c1", 5),
        ...Object.entries(EMB).map(([w, p]) => Plot.text(p[0] + 0.1, p[1] + 0.2, w, "", "start")),
        ...(used.length ? [Plot.dot(vec[0], vec[1], "pl-c2", 8), Plot.text(vec[0] + 0.12, vec[1] - 0.35, "result", "", "start")] : []),
      ];
      const svg = Plot.chart({ W: 560, H: 380, x: [-4.5, 5.6], y: [-4, 4.5], xl: "dimension 1", yl: "dimension 2", ticks: 6, yticks: 6 }, layers);
      const msg = bad.length ? `Unknown word${bad.length > 1 ? "s" : ""}: ${bad.map((b) => "`" + b + "`").join(", ")}. Use words visible on the map.`
        : !used.length ? "Type an expression like `king - man + woman`."
        : `Every word is a point (a vector). Green arrows add a word's vector, red arrows subtract it. \`${inp.trim()}\` lands at (${fx(vec[0], 2)}, ${fx(vec[1], 2)}), the orange dot. The nearest word (leaving out the words you typed) is **${best[0]}**, at distance ${fx(best[1], 2)}. Real systems usually rank by [cosine similarity](math.html#dot-product) instead of distance. The map was built by hand with a "gender" direction (left to right) and a "royalty" direction (bottom to top), which is exactly the kind of structure that real word2vec vectors discover on their own in 300 dimensions.`;
      return [{ svg, side: { result: `(${fx(vec[0], 2)}, ${fx(vec[1], 2)})`, "nearest": best ? best[0] : "-", "3 nearest (distance)": nn.map((n) => `${n[0]} ${fx(n[1], 2)}`).join(", ") }, msg }];
    },
  };

  /* ================================================================== 7. Diffusion: forward noising */
  const DIF = (() => {
    const r = rng(11), pts = [], ring = (cx, cy, rad, n, a0 = 0, a1 = 2 * Math.PI) => { for (let i = 0; i < n; i++) { const t = a0 + ((a1 - a0) * i) / n; pts.push([cx + rad * Math.cos(t), cy + rad * Math.sin(t)]); } };
    ring(0, 0, 1.6, 70); ring(-0.6, 0.55, 0.12, 6); ring(0.6, 0.55, 0.12, 6); ring(0, -0.1, 0.9, 26, 1.15 * Math.PI, 1.85 * Math.PI);
    const eps = pts.map(() => [gauss(r), gauss(r)]);
    const pix = []; // 16x16 smiley, values -1 (background) / +1 (ink)
    for (let i = 0; i < 16; i++) { const row = []; for (let j = 0; j < 16; j++) { const x = j - 7.5, y = i - 7.5, d = Math.hypot(x, y); let v = Math.abs(d - 6.3) < 0.8 ? 1 : -1; if (Math.hypot(x + 2.4, y + 2) < 1.1 || Math.hypot(x - 2.4, y + 2) < 1.1) v = 1; if (d < 4.2 && y > 1 && Math.abs(d - 3.6) < 0.8) v = 1; row.push(v); } pix.push(row); }
    const peps = pix.map((row) => row.map(() => gauss(r)));
    const T = 1000, abar = [1]; for (let t = 1; t <= T; t++) abar.push(abar[t - 1] * (1 - (1e-4 + ((0.02 - 1e-4) * (t - 1)) / (T - 1))));
    return { pts, eps, pix, peps, abar };
  })();
  V.dl2_diffusion = {
    title: "Diffusion: slowly turning data into pure noise",
    params: [{ id: "t", label: "timestep t (0 = clean, 1000 = pure noise)", min: 0, max: 1000, step: 10, value: 100 }],
    gen(_, P) {
      const t = +P.t, ab = DIF.abar[t], sa = Math.sqrt(ab), sn = Math.sqrt(1 - ab);
      const W = 640, H = 330;
      let g = T(160, 22, "2-D data points", "font-weight:700;fill:var(--text)") + T(480, 22, "a 16×16 image", "font-weight:700;fill:var(--text)");
      const px = (v) => 20 + ((v + 3.2) / 6.4) * 280, py = (v) => 40 + ((3.2 - v) / 6.4) * 280;
      g += `<rect x="20" y="40" width="280" height="280" style="fill:none;stroke:var(--border)"/>`;
      DIF.pts.forEach((p, i) => { const x = sa * p[0] + sn * DIF.eps[i][0], y = sa * p[1] + sn * DIF.eps[i][1]; if (Math.abs(x) < 3.2 && Math.abs(y) < 3.2) g += `<circle cx="${px(x).toFixed(1)}" cy="${py(y).toFixed(1)}" r="3.2" style="fill:var(--accent);opacity:.85"/>`; });
      const cs = 17.5, ox = 340, oy = 40;
      DIF.pix.forEach((row, i) => row.forEach((v, j) => { const q = sa * v + sn * DIF.peps[i][j], u = Math.max(0, Math.min(1, (q + 1.5) / 3)); g += `<rect x="${ox + j * cs}" y="${oy + i * cs}" width="${cs}" height="${cs}" style="fill:color-mix(in srgb, var(--text) ${Math.round(u * 92)}%, var(--panel))"/>`; }));
      const snr = ab / (1 - ab || 1e-12);
      return [{
        svg: svgWrap(W, H, g),
        side: { "timestep t": t, "ᾱ_t": ab >= 0.01 ? fx(ab, 3) : ab.toExponential(1), "signal weight √ᾱ": fx(sa, 3), "noise weight √(1−ᾱ)": fx(sn, 3), "signal/noise (ᾱ/(1−ᾱ))": snr > 1000 ? snr.toExponential(1) : fx(snr, 3) },
        msg: `The **forward process** adds a little Gaussian noise at each of 1000 steps. Thanks to a neat property we can jump straight to any step $t$: $x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon$, where $\\varepsilon$ is standard Gaussian noise (the same noise for every slider position, so you see one smooth path). At $t=0$ the shape is clean; around $t=100$ it is still easy to see; by $t=250$ it is mostly lost; at $t=1000$ the signal weight is only ${fx(Math.sqrt(DIF.abar[1000]), 3)}, so nothing is left but noise. A diffusion model is trained to **undo one small step of this**; running those undo steps from pure noise generates new data.`,
      }];
    },
  };

  /* ================================================================== static diagrams */
  const mk = (id) => `<defs><marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker><marker id="${id}r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--red)"/></marker><marker id="${id}a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--accent)"/></marker></defs>`;
  const wrap = (W, H, inner, id) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">${mk(id)}${inner}</svg>`;
  const bx = (x, y, w, h, label, cls = "node", sub) => `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="8"/><text class="val" x="${x + w / 2}" y="${y + h / 2 + (sub ? 0 : 5)}" text-anchor="middle" style="font-size:12.5px">${esc(label)}</text>${sub ? `<text class="lbl" x="${x + w / 2}" y="${y + h / 2 + 15}" text-anchor="middle" style="font-size:11px">${esc(sub)}</text>` : ""}`;
  const tx = (x, y, s, style = "", anchor = "middle") => `<text class="lbl" x="${x}" y="${y}" text-anchor="${anchor}" style="${style}">${esc(s)}</text>`;
  const arr = (x1, y1, x2, y2, id, col) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" style="stroke:var(--${col === "red" ? "red" : col === "accent" ? "accent" : "muted"});stroke-width:1.8" marker-end="url(#${id}${col === "red" ? "r" : col === "accent" ? "a" : ""})"/>`;
  // a stack of feature maps drawn as offset rectangles
  const stack = (x, y, w, h, n, cls = "node") => { let s = ""; for (let i = n - 1; i >= 0; i--) s += `<rect class="${cls}" x="${x + i * 5}" y="${y - i * 5}" width="${w}" height="${h}" rx="2" style="stroke-width:1.5"/>`; return s; };

  // a typical CNN: image -> (conv, relu, pool) x2 -> flatten -> fully connected -> softmax
  D.dl2_cnn = () => {
    const id = "dc2";
    let s = "";
    s += stack(14, 80, 62, 62, 3) + tx(40, 168, "image") + tx(40, 183, "32×32×3", "font-size:11px");
    s += arr(96, 100, 128, 100, id);
    s += stack(132, 80, 56, 56, 6, "node hl") + tx(158, 168, "conv 3×3 + ReLU") + tx(158, 183, "32×32×6", "font-size:11px");
    s += arr(172 + 24, 100, 232, 100, id);
    s += stack(236, 92, 32, 32, 6, "node ok") + tx(252, 168, "max pool") + tx(252, 183, "16×16×6", "font-size:11px");
    s += arr(290, 100, 322, 100, id);
    s += stack(326, 84, 34, 34, 12, "node hl") + tx(350, 168, "conv 3×3 + ReLU") + tx(350, 183, "16×16×12", "font-size:11px");
    s += arr(386, 100, 418, 100, id);
    s += stack(422, 96, 18, 18, 12, "node ok") + tx(436, 168, "max pool") + tx(436, 183, "8×8×12", "font-size:11px");
    s += arr(480, 100, 508, 100, id);
    s += `<rect class="node" x="512" y="40" width="16" height="120" rx="3" style="stroke-width:1.5"/>` + tx(520, 178, "flatten") + tx(520, 192, "768", "font-size:11px");
    s += arr(534, 100, 560, 100, id);
    s += `<rect class="node bad" x="564" y="62" width="16" height="76" rx="3" style="stroke-width:1.5"/>` + tx(572, 156, "dense") + tx(572, 170, "+ softmax", "font-size:11px");
    s += arr(586, 100, 606, 100, id);
    ["cat", "dog", "car"].forEach((c, i) => (s += `<text x="612" y="${92 + i * 14}" style="font-size:11px;fill:var(--text);font-weight:600">${c}</text>`));
    s += tx(250, 24, "feature extractor: convolutions find patterns, pooling shrinks", "font-size:12.5px;font-weight:700;fill:var(--accent)");
    s += tx(100, 218, "←  few weights, shared across the image  →", "font-size:11.5px", "start");
    s += tx(496, 218, "classifier head: all-to-all weights", "font-size:11.5px", "start");
    s += `<line x1="14" y1="205" x2="486" y2="205" style="stroke:var(--accent);stroke-width:2"/><line x1="506" y1="205" x2="640" y2="205" style="stroke:var(--red);stroke-width:2"/>`;
    return wrap(660, 236, s, id);
  };

  // a residual block: y = relu(F(x) + x)
  D.dl2_resblock = () => {
    const id = "dr2";
    let s = "";
    s += bx(150, 20, 100, 30, "x (input)", "node") + arr(200, 52, 200, 70, id);
    s += bx(130, 72, 140, 34, "conv 3×3", "node hl") + arr(200, 108, 200, 124, id);
    s += bx(130, 126, 140, 30, "BatchNorm + ReLU", "node") + arr(200, 158, 200, 174, id);
    s += bx(130, 176, 140, 34, "conv 3×3", "node hl") + arr(200, 212, 200, 228, id);
    s += bx(130, 230, 140, 30, "BatchNorm", "node") + arr(200, 262, 200, 280, id);
    s += `<circle class="node ok" cx="200" cy="296" r="16"/><text class="val" x="200" y="302" text-anchor="middle" style="font-size:20px">+</text>`;
    s += arr(200, 313, 200, 332, id) + bx(160, 334, 80, 28, "ReLU", "node") + arr(200, 364, 200, 380, id) + tx(200, 396, "output", "font-size:12px;fill:var(--text);font-weight:700");
    s += `<path d="M250 36 C 360 36, 360 296, 218 296" fill="none" style="stroke:var(--green);stroke-width:2.6" marker-end="url(#${id}a)"/>`;
    s += tx(372, 168, "skip connection", "font-size:12.5px;font-weight:700;fill:var(--green)", "start") + tx(372, 184, "copies x unchanged", "font-size:11.5px", "start");
    s += tx(14, 168, "F(x): the layers", "font-size:12px;fill:var(--accent);font-weight:700", "start") + tx(14, 184, "learn a correction", "font-size:11.5px", "start");
    s += tx(372, 300, "output = ReLU( F(x) + x )", "font-size:12.5px;fill:var(--text);font-weight:700", "start");
    return wrap(600, 412, s, id);
  };

  // encoder-decoder (seq2seq) without attention
  D.dl2_encdec = () => {
    const id = "de2", src = ["I", "am", "a", "student"], tgt = ["je", "suis", "étudiant", "<end>"], inp = ["<start>", "je", "suis", "étudiant"];
    let s = "";
    s += tx(180, 18, "ENCODER reads the whole input", "font-size:12.5px;font-weight:700;fill:var(--accent)") + tx(500, 18, "DECODER writes the output one word at a time", "font-size:12.5px;font-weight:700;fill:var(--green)");
    src.forEach((w, i) => {
      const x = 20 + i * 86;
      s += bx(x, 96, 70, 40, "RNN", "node hl") + tx(x + 35, 170, w, "font-size:13px;font-weight:700;fill:var(--text)") + arr(x + 35, 164 - 12, x + 35, 140, id);
      if (i < 3) s += arr(x + 72, 116, x + 84, 116, id);
    });
    s += arr(20 + 3 * 86 + 72, 116, 372, 116, id);
    s += `<rect class="node bad" x="376" y="82" width="70" height="68" rx="10"/><text class="val" x="411" y="110" text-anchor="middle" style="font-size:12.5px">context</text><text class="val" x="411" y="127" text-anchor="middle" style="font-size:12.5px">vector</text><text class="lbl" x="411" y="143" text-anchor="middle" style="font-size:10.5px">one fixed size</text>`;
    s += arr(448, 116, 482, 116, id);
    tgt.forEach((w, i) => {
      const x = 486 + i * 86;
      s += bx(x, 96, 70, 40, "RNN", "node ok") + arr(x + 35, 94, x + 35, 62, id) + tx(x + 35, 52, w, "font-size:13px;font-weight:700;fill:var(--text)");
      s += tx(x + 35, 170, inp[i], "font-size:12px;fill:var(--muted)") + arr(x + 35, 158, x + 35, 140, id);
      if (i < 3) s += arr(x + 72, 116, x + 84, 116, id);
    });
    s += tx(300, 214, "the encoder squeezes the whole sentence into ONE vector; the decoder must produce every output word from it", "font-size:12px;fill:var(--text)");
    s += tx(300, 232, "the decoder's input is its own previous output (or the true previous word while training: teacher forcing)", "font-size:12px");
    return wrap(860, 250, s, id);
  };

  // seq2seq with attention: decoder looks at all encoder states
  D.dl2_attnseq = () => {
    const id = "da2", src = ["I", "am", "a", "student"], wts = [0.05, 0.1, 0.1, 0.75];
    let s = tx(300, 18, "decoder is about to write the 3rd word (\"étudiant\"): where should it look?", "font-size:12.5px;font-weight:700;fill:var(--text)");
    src.forEach((w, i) => {
      const x = 40 + i * 120;
      s += bx(x, 190, 80, 34, "h" + (i + 1), "node hl") + tx(x + 40, 246, w, "font-size:13px;font-weight:700;fill:var(--text)");
      s += `<line x1="${x + 40}" y1="188" x2="300" y2="106" style="stroke:var(--accent);stroke-width:${(1 + wts[i] * 9).toFixed(1)};opacity:${(0.25 + wts[i]).toFixed(2)}"/>`;
      s += `<text x="${x + 40 + (300 - x - 40) * 0.38}" y="${188 + (106 - 188) * 0.38 - 6}" text-anchor="middle" style="font-size:12px;font-weight:700;fill:var(--text);${HALO}">${wts[i].toFixed(2)}</text>`;
    });
    s += bx(240, 68, 120, 38, "context c₃", "node bad") + arr(300, 66, 300, 40, id);
    s += `<rect class="node ok" x="430" y="68" width="110" height="38" rx="8"/><text class="val" x="485" y="92" text-anchor="middle" style="font-size:12.5px">decoder state s₃</text>`;
    s += arr(428, 87, 362, 87, id, "accent") + tx(395, 78, "asks", "font-size:11px");
    s += tx(300, 276, "attention weights (they sum to 1): c₃ = 0.05·h₁ + 0.10·h₂ + 0.10·h₃ + 0.75·h₄  (illustrative numbers)", "font-size:12px;fill:var(--text)");
    return wrap(600, 292, s, id);
  };

  // overview of autoencoder / VAE / GAN
  D.dl2_genoverview = () => {
    const id = "dg2";
    const trap = (x, y, wl, wr, h, cls) => `<polygon class="${cls}" points="${x},${y - wl / 2} ${x + 70},${y - wr / 2} ${x + 70},${y + wr / 2} ${x},${y + wl / 2}" style="stroke-width:1.8"/>`;
    let s = "";
    // autoencoder
    s += tx(10, 22, "Autoencoder", "font-size:13px;font-weight:700;fill:var(--text)", "start");
    s += bx(20, 40, 56, 40, "x") + arr(78, 60, 96, 60, id) + trap(98, 60, 64, 22, 0, "node hl") + tx(133, 108, "encoder", "font-size:11px") + arr(170, 60, 188, 60, id);
    s += bx(190, 46, 44, 28, "z", "node bad") + arr(236, 60, 254, 60, id) + trap(256, 60, 22, 64, 0, "node ok") + tx(291, 108, "decoder", "font-size:11px") + arr(328, 60, 346, 60, id) + bx(348, 40, 56, 40, "x̂") + tx(420, 66, "z is small: it keeps only the essentials", "font-size:12px", "start");
    // VAE
    s += tx(10, 150, "Variational autoencoder (VAE)", "font-size:13px;font-weight:700;fill:var(--text)", "start");
    s += bx(20, 168, 56, 40, "x") + arr(78, 188, 96, 188, id) + trap(98, 188, 64, 22, 0, "node hl") + tx(133, 236, "encoder", "font-size:11px");
    s += arr(170, 180, 186, 168, id) + arr(170, 196, 186, 208, id) + bx(188, 150, 44, 24, "μ", "node") + bx(188, 202, 44, 24, "σ", "node");
    s += arr(234, 162, 258, 182, id) + arr(234, 214, 258, 194, id) + bx(260, 172, 96, 32, "z = μ + σ·ε", "node bad") + tx(308, 222, "ε random noise", "font-size:11px");
    s += arr(358, 188, 376, 188, id) + trap(378, 188, 22, 64, 0, "node ok") + tx(413, 236, "decoder", "font-size:11px") + arr(450, 188, 468, 188, id) + bx(470, 168, 56, 40, "x̂") + tx(540, 192, "z is a distribution: sample new z", "font-size:12px", "start");
    // GAN
    s += tx(10, 278, "GAN", "font-size:13px;font-weight:700;fill:var(--text)", "start");
    s += bx(20, 296, 56, 40, "noise z") + arr(78, 316, 96, 316, id) + trap(98, 316, 22, 64, 0, "node hl") + tx(133, 360, "generator", "font-size:11px") + arr(170, 316, 188, 316, id) + bx(190, 296, 56, 40, "fake", "node bad");
    s += bx(190, 352, 56, 36, "real", "node ok") + arr(248, 316, 274, 330, id) + arr(248, 370, 274, 346, id);
    s += trap(276, 338, 64, 22, 0, "node") + tx(311, 384, "discriminator", "font-size:11px") + arr(348, 338, 366, 338, id) + bx(368, 322, 92, 32, "real or fake?", "node hl");
    s += arr(414, 322, 414, 300, id, "red") + `<path d="M414 296 C 330 262, 150 262, 134 284" fill="none" style="stroke:var(--red);stroke-width:1.8" marker-end="url(#${id}r)"/>` + tx(500, 300, "generator learns from the", "font-size:12px", "start") + tx(500, 316, "discriminator's mistakes", "font-size:12px", "start");
    return wrap(760, 400, s, id);
  };

  // diffusion as a chain: add noise forward, learn to remove it backward
  D.dl2_diffchain = () => {
    const id = "dd2", labels = ["x₀", "x₁", "x₂", "…", "x_T"], nz = [0, 0.25, 0.5, 0.75, 1];
    let s = tx(310, 16, "forward: add a little noise each step (fixed, no learning)", "font-size:12.5px;font-weight:700;fill:var(--red)");
    labels.forEach((l, i) => {
      const x = 30 + i * 118;
      s += `<rect class="node" x="${x}" y="40" width="64" height="64" rx="6"/>`;
      // a few dots whose spread grows with the noise level
      for (let q = 0; q < 14; q++) { const a = (q / 14) * 2 * Math.PI, rr = 18 * (1 - nz[i]) + 3, jx = ((q * 37) % 11 - 5) * nz[i] * 2.6, jy = ((q * 53) % 11 - 5) * nz[i] * 2.6; s += `<circle cx="${x + 32 + rr * Math.cos(a) + jx}" cy="${72 + rr * Math.sin(a) + jy}" r="2.4" style="fill:var(--accent)"/>`; }
      s += tx(x + 32, 124, l, "font-size:13px;font-weight:700;fill:var(--text)");
      if (i < 4) s += arr(x + 68, 62, x + 114, 62, id, "red") + arr(x + 114, 84, x + 68, 84, id, "accent");
    });
    s += tx(310, 160, "reverse: a neural network learns to remove one step of noise (the part we train)", "font-size:12.5px;font-weight:700;fill:var(--accent)");
    s += tx(310, 182, "generate: start from pure noise x_T and apply the learned denoiser T times", "font-size:12px");
    return wrap(620, 196, s, id);
  };

  // skip-gram: predict the context words from the centre word
  D.dl2_skipgram = () => {
    const id = "dk2", ws = ["the", "quick", "brown", "fox", "jumps", "over"];
    let s = tx(300, 18, "skip-gram, window = 2: the centre word predicts each neighbour", "font-size:12.5px;font-weight:700;fill:var(--text)");
    ws.forEach((w, i) => {
      const x = 14 + i * 96, c = i === 3;
      s += bx(x, 44, 80, 34, w, c ? "node hl" : i >= 1 && i <= 5 && Math.abs(i - 3) <= 2 ? "node ok" : "node");
    });
    [1, 2, 4, 5].forEach((i) => (s += `<path d="M${14 + 3 * 96 + 40} 80 C ${14 + 3 * 96 + 40} 130, ${14 + i * 96 + 40} 110, ${14 + i * 96 + 40} 82" fill="none" style="stroke:var(--accent);stroke-width:1.8" marker-end="url(#${id}a)"/>`));
    s += tx(300, 150, "training pairs from this window: (fox → quick), (fox → brown), (fox → jumps), (fox → over)", "font-size:12px;fill:var(--text)");
    s += tx(300, 168, "words that appear in similar contexts end up with similar vectors", "font-size:12px");
    return wrap(600, 182, s, id);
  };

  /* ------------------------------------------------------------------ glossary */
  const G = window.GLOSSARY;
  const add = (o) => Object.keys(o).forEach((k) => { if (!(k in G)) G[k] = o[k]; });
  add({
    "convolution": "Sliding a small grid of weights (the kernel) over an image and, at each position, multiplying and adding to get one output number.",
    "kernel": "The small grid of learned weights (for example 3×3) that a convolution slides over the input. Also called a filter.",
    "filter": "Another name for a convolution kernel: a small grid of weights that detects one pattern.",
    "stride": "How many pixels the kernel jumps between positions. Stride 2 halves the output size.",
    "padding": "Extra border (usually zeros) added around the input so the kernel can sit on the edge pixels and the output keeps its size.",
    "feature map": "The output of one filter: a grid showing where in the image that filter's pattern was found.",
    "channel": "One layer of a stacked image or feature map, for example the red, green and blue channels of a colour image.",
    "receptive field": "The part of the input that can influence one unit. It grows as you stack convolution layers.",
    "weight sharing": "Using the same weights at every position, as a convolution does. It gives far fewer parameters and finds a pattern wherever it appears.",
    "pooling": "Shrinking a feature map by summarising each small window (its maximum or average) with a single number. It has no learned weights.",
    "max pooling": "Pooling that keeps the largest value in each window, so a feature counts as found if it appears anywhere in the window.",
    "global average pooling": "Averaging each whole feature map down to one number, giving a vector with one value per channel.",
    "cnn": "Convolutional neural network: a network built from convolution layers, the standard choice for images.",
    "resnet": "A network built from residual blocks with skip connections, which made networks with 50 to 1000 layers trainable.",
    "skip connection": "A shortcut that adds a layer's input to its output, so the layers only learn a correction and gradients can flow straight through.",
    "im2col": "A trick that turns a convolution into one big matrix multiplication by copying every patch of the image into a column.",
    "rnn": "Recurrent neural network: reads a sequence one item at a time and carries a hidden state forward as memory.",
    "hidden state": "The vector a recurrent network carries from one time step to the next. It is the network's memory of what it has read so far.",
    "time step": "One position in a sequence, such as one word. A recurrent network does one update per time step.",
    "bptt": "Backpropagation through time: unrolling a recurrent network over the sequence and backpropagating through all the steps.",
    "lstm": "Long short-term memory: a recurrent cell with a cell state and three gates (forget, input, output) that lets it remember over long sequences.",
    "gru": "Gated recurrent unit: a simpler cousin of the LSTM with two gates (update and reset) and no separate cell state.",
    "gate": "A number between 0 and 1 (from a sigmoid) that multiplies a signal, letting a fraction of it through.",
    "cell state": "The LSTM's long-term memory lane, which is edited only by small gated additions.",
    "seq2seq": "Sequence-to-sequence: an encoder reads an input sequence and a decoder writes an output sequence, as in translation.",
    "teacher forcing": "Training a decoder by feeding it the true previous word instead of its own earlier guess.",
    "context vector": "In seq2seq, the summary of the input that the decoder uses to write the output.",
    "word2vec": "A method that learns word vectors by predicting a word from its neighbours (or neighbours from a word) in a large text.",
    "glove": "A word-vector method that fits vectors to the co-occurrence counts of words over a whole corpus.",
    "skip-gram": "The word2vec task of predicting the surrounding words from a centre word.",
    "negative sampling": "A cheap way to train word2vec: for each real word pair, also push the centre word away from a few random wrong words.",
    "autoencoder": "A network that squeezes its input into a small code and then rebuilds the input from it, learning a compact representation.",
    "latent space": "The hidden space of the code z inside an autoencoder or generator, where each point can be turned into an output.",
    "bottleneck": "A deliberately narrow layer that forces a network to keep only the most important information.",
    "vae": "Variational autoencoder: an autoencoder whose code is a smooth probability distribution, so you can sample new, realistic outputs.",
    "gan": "Generative adversarial network: a generator makes fakes and a discriminator tries to spot them; they improve by competing.",
    "generator": "In a GAN, the network that turns random noise into a fake sample.",
    "discriminator": "In a GAN, the network that decides whether a sample is real or made by the generator.",
    "diffusion": "A generative method that learns to remove noise a little at a time; starting from pure noise it gradually produces a sample.",
    "transfer learning": "Starting from a network already trained on a big dataset and adapting it to a new, usually smaller, task.",
    "fine-tuning": "Continuing to train a pretrained model on your own data, usually with a small learning rate.",
    "pretrained": "A model already trained on a large dataset, whose weights you reuse as a starting point.",
    "dataloader": "A PyTorch helper that serves the dataset in shuffled mini-batches.",
    "requires_grad": "A PyTorch flag on a tensor: when True, operations on it are recorded so gradients can be computed.",
    "overfit one batch": "A debugging test: train on a single small batch; if the loss cannot reach almost zero, the code or model has a bug.",
  });
})();
