/* Math for ML: visualizers, diagrams and glossary terms.
   Loaded on every doc page and on Practice, after assets/viz.js. Every number shown is computed, not faked. */
(function () {
  const V = Viz.defs, Plot = Viz.Plot, { r3 } = Viz.helpers;

  /* ------------------------------------------------------------------ private helpers */
  const f2 = (v, d = 2) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : +v).toFixed(d);
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const DEG = Math.PI / 180;
  const vec = (x, y) => `(${f2(x, 1)}, ${f2(y, 1)})`;
  // an arrow layer from the origin (or from a point) with a CSS class
  const arrow = (x1, y1, x2, y2, cls = "pl-c1") => Plot.seg(x1, y1, x2, y2, cls, true);
  // a custom Plot layer: filled polygon in data coordinates
  const poly = (pts, style) => (S) => `<polygon points="${pts.map(([x, y]) => S.sx(x).toFixed(1) + "," + S.sy(y).toFixed(1)).join(" ")}" style="${style}"/>`;
  const soft = (v, p = 16) => `color-mix(in srgb, var(--${v}) ${p}%, transparent)`;
  // a chart whose unit is the same on both axes (so angles look like angles)
  const square = (r, extra = {}) => Object.assign({ W: 430, H: 414, x: [-r, r], y: [-r, r], ticks: 6, yticks: 6 }, extra);
  const tag = (x, y, s, style = "", anchor = "middle") => (S) => `<text x="${S.sx(x).toFixed(1)}" y="${S.sy(y).toFixed(1)}" text-anchor="${anchor}" style="font-size:12px;fill:var(--muted);${style}">${esc(s)}</text>`;
  const erf = (x) => { const s = Math.sign(x), t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; };
  const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  const L2 = (v) => (v > 0 ? v * Math.log2(v) : 0);

  // Jacobi eigen-decomposition of a small symmetric matrix: returns { vals (descending), vecs (columns as rows) }
  function jacobiEig(Sym) {
    const n = Sym.length, A = Sym.map((r) => r.slice()), Vv = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 80; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-14) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b; }
        for (let k = 0; k < n; k++) { const a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b; }
        for (let k = 0; k < n; k++) { const a = Vv[k][p], b = Vv[k][q]; Vv[k][p] = c * a - s * b; Vv[k][q] = s * a + c * b; }
      }
    }
    const idx = A.map((_, i) => i).sort((i, j) => A[j][j] - A[i][i]);
    return { vals: idx.map((i) => A[i][i]), vecs: idx.map((i) => Vv.map((r) => r[i])) };
  }

  /* ================================================================== 1. Vector addition */
  V.math_vec = {
    title: "Vectors: add them tip to tail",
    params: [
      { id: "ax", label: "a, x part", min: -4, max: 4, step: 0.5, value: 3 },
      { id: "ay", label: "a, y part", min: -4, max: 4, step: 0.5, value: 1 },
      { id: "bx", label: "b, x part", min: -4, max: 4, step: 0.5, value: 1 },
      { id: "by", label: "b, y part", min: -4, max: 4, step: 0.5, value: 3 },
    ],
    gen(_, P) {
      const sx = P.ax + P.bx, sy = P.ay + P.by, len = Math.hypot(sx, sy);
      const svg = Plot.chart(square(6), [
        poly([[0, 0], [P.ax, P.ay], [sx, sy], [P.bx, P.by]], `fill:${soft("accent", 9)};stroke:none`),
        Plot.seg(P.ax, P.ay, sx, sy, "pl-c3 pl-dash pl-thin"), Plot.seg(P.bx, P.by, sx, sy, "pl-c1 pl-dash pl-thin"),
        arrow(0, 0, P.ax, P.ay, "pl-c1"), arrow(0, 0, P.bx, P.by, "pl-c3"), arrow(0, 0, sx, sy, "pl-c4"),
        tag(P.ax * 0.5 + 0.1, P.ay * 0.5 - 0.5, "a", "fill:var(--accent);font-weight:700"), tag(P.bx * 0.5 - 0.4, P.by * 0.5 + 0.2, "b", "fill:var(--green);font-weight:700"),
        tag(sx * 0.5 + 0.4, sy * 0.5 + 0.3, "a + b", "fill:var(--red);font-weight:700"),
      ]);
      return [{
        svg, side: { a: vec(P.ax, P.ay), b: vec(P.bx, P.by), "a + b": vec(sx, sy), "length of a + b": r3(len) },
        msg: `Adding vectors means adding their numbers **position by position**: (${P.ax} + ${P.bx}, ${P.ay} + ${P.by}) = (${sx}, ${sy}). Drawn as arrows: walk along **a**, then along **b**; you end where **a + b** points (the red arrow). The dashed sides show the same trip taken in the other order.`,
      }];
    },
  };

  /* ================================================================== 2. Dot product and projection */
  V.math_dot = {
    title: "Dot product: how much do two vectors point the same way?",
    params: [
      { id: "ang", label: "angle of a (degrees)", min: -180, max: 180, step: 5, value: 20 },
      { id: "la", label: "length of a", min: 1, max: 4.5, step: 0.5, value: 3 },
      { id: "bng", label: "angle of b (degrees)", min: -180, max: 180, step: 5, value: 100 },
      { id: "lb", label: "length of b", min: 1, max: 4.5, step: 0.5, value: 3 },
    ],
    gen(_, P) {
      const a = [P.la * Math.cos(P.ang * DEG), P.la * Math.sin(P.ang * DEG)], b = [P.lb * Math.cos(P.bng * DEG), P.lb * Math.sin(P.bng * DEG)];
      const dot = a[0] * b[0] + a[1] * b[1], cos = dot / (P.la * P.lb), k = dot / (P.la * P.la), pr = [k * a[0], k * a[1]];
      let th = Math.abs(P.bng - P.ang) % 360; if (th > 180) th = 360 - th;
      const rel = Math.abs(dot) < 0.05 ? "**zero**: the vectors are perpendicular (nothing in common)." : dot > 0 ? "**positive**: they point roughly the same way." : "**negative**: they point roughly opposite ways.";
      const svg = Plot.chart(square(5), [
        Plot.seg(-5 * Math.cos(P.ang * DEG), -5 * Math.sin(P.ang * DEG), 5 * Math.cos(P.ang * DEG), 5 * Math.sin(P.ang * DEG), "pl-c1 pl-dash pl-thin"),
        Plot.seg(b[0], b[1], pr[0], pr[1], "pl-c2 pl-dash pl-thin"),
        arrow(0, 0, pr[0], pr[1], "pl-c2"), arrow(0, 0, a[0], a[1], "pl-c1"), arrow(0, 0, b[0], b[1], "pl-c3"),
        Plot.dot(pr[0], pr[1], "pl-c2", 4),
        tag(a[0] * 1.08, a[1] * 1.08 + 0.3, "a", "fill:var(--accent);font-weight:700"), tag(b[0] * 1.08, b[1] * 1.08 + 0.3, "b", "fill:var(--green);font-weight:700"),
      ]);
      return [{
        svg, side: { "a · b": r3(dot), "angle between": Math.round(th) + "°", "cos(angle)": r3(cos), "projection of b on a (length)": r3(dot / P.la) },
        msg: `$a\\cdot b = |a|\\,|b|\\cos\\theta = ${r3(P.la)}\\times${r3(P.lb)}\\times${r3(cos)} = ${r3(dot)}$, ${rel} The amber arrow is the **shadow of b on the line of a** (its projection): its length is $a\\cdot b/|a|$. Cosine similarity is just the dot product with both lengths divided out, so it only looks at the angle.`,
      }];
    },
  };

  /* ================================================================== 3. A 2x2 matrix transforms the plane */
  V.math_matrix = {
    title: "A matrix transforms the plane (and eigenvectors stay on their line)",
    params: [
      { id: "a", label: "a  (top left)", min: -2, max: 2, step: 0.1, value: 2 },
      { id: "b", label: "b  (top right)", min: -2, max: 2, step: 0.1, value: 1 },
      { id: "c", label: "c  (bottom left)", min: -2, max: 2, step: 0.1, value: 1 },
      { id: "d", label: "d  (bottom right)", min: -2, max: 2, step: 0.1, value: 2 },
      { id: "t", label: "test vector angle (deg)", min: 0, max: 355, step: 5, value: 20 },
    ],
    gen(_, P) {
      const { a, b, c, d } = P, tr = a + d, det = a * d - b * c, disc = tr * tr / 4 - det;
      const M = (x, y) => [a * x + b * y, c * x + d * y];
      const v = [Math.cos(P.t * DEG), Math.sin(P.t * DEG)], Mv = M(v[0], v[1]);
      const layers = [
        poly([[0, 0], [1, 0], [1, 1], [0, 1]], "fill:none;stroke:var(--muted);stroke-dasharray:4 4;stroke-width:1.2"),
        poly([[0, 0], [a, c], [a + b, c + d], [b, d]], `fill:${soft("accent", 14)};stroke:var(--accent);stroke-width:1.5`),
      ];
      let eig = "none (the matrix rotates everything: no direction keeps its line)";
      if (disc >= -1e-9) {
        const s = Math.sqrt(Math.max(disc, 0)), ls = [tr / 2 + s, tr / 2 - s], evs = [];
        ls.forEach((lam) => {
          let e = Math.abs(b) > 1e-9 ? [b, lam - a] : Math.abs(c) > 1e-9 ? [lam - d, c] : (Math.abs(lam - a) < 1e-9 ? [1, 0] : [0, 1]);
          const n = Math.hypot(e[0], e[1]); e = [e[0] / n, e[1] / n]; evs.push(e);
          layers.push(Plot.seg(-8 * e[0], -8 * e[1], 8 * e[0], 8 * e[1], "pl-c5 pl-dash pl-thin"));
        });
        eig = `λ = ${f2(ls[0])} (direction ${vec(evs[0][0], evs[0][1])}) and λ = ${f2(ls[1])} (direction ${vec(evs[1][0], evs[1][1])})`;
      }
      layers.push(arrow(0, 0, a, c, "pl-c1"), arrow(0, 0, b, d, "pl-c3"), arrow(0, 0, v[0], v[1], "pl-c2"), arrow(0, 0, Mv[0], Mv[1], "pl-c4"),
        tag(a, c + 0.35, "A·î", "fill:var(--accent);font-weight:700"), tag(b, d + 0.35, "A·ĵ", "fill:var(--green);font-weight:700"));
      const svg = Plot.chart(square(4), layers);
      const volume = Math.abs(det) < 0.05 ? "The square is squashed flat onto a line: the matrix is **singular** (no inverse)." : det < 0 ? "A negative determinant means the plane got **flipped** (mirrored)." : "The determinant is the factor by which areas are scaled.";
      return [{
        svg, side: { "det(A)": r3(det), "trace": r3(tr), "eigenvalues": disc >= -1e-9 ? `${f2(tr / 2 + Math.sqrt(Math.max(disc, 0)))}, ${f2(tr / 2 - Math.sqrt(Math.max(disc, 0)))}` : "complex" },
        msg: `The **columns** of the matrix say where the basis arrows go: î = (1,0) lands on (${a}, ${c}) (blue) and ĵ = (0,1) lands on (${b}, ${d}) (green). The dashed unit square becomes the shaded parallelogram, so its area is scaled by det = ${r3(det)}. ${volume} The amber test vector goes to the red one. Eigenvectors (purple dashed lines): ${eig}. Turn the test vector onto a purple line and the red arrow lines up with it, only longer or shorter by λ.`,
      }];
    },
  };

  /* ================================================================== 4. Derivative as a tangent line */
  const FT = (x) => x ** 3 / 4 - x + 1, FTd = (x) => 0.75 * x * x - 1;
  V.math_tangent = {
    title: "Derivative = slope of the tangent line",
    params: [
      { id: "x", label: "point x", min: -3, max: 3, step: 0.1, value: 1.5 },
      { id: "h", label: "step h for the secant line", min: 0.01, max: 2, step: 0.01, value: 1 },
    ],
    gen(_, P) {
      const x = P.x, h = P.h, y = FT(x), m = FTd(x), sec = (FT(x + h) - y) / h;
      const line = (slope, cls, name) => Plot.line([[x - 4, y - 4 * slope], [x + 4, y + 4 * slope]], cls);
      const svg = Plot.chart({ W: 520, H: 320, x: [-3.5, 3.5], y: [-4, 6.5], xl: "x", yl: "f(x) = x³/4 − x + 1" }, [
        Plot.fn(FT, "pl-c1"),
        Plot.line([[x - 4, y - 4 * sec], [x + 4, y + 4 * sec]], "pl-c2 pl-dash pl-thin"),
        line(m, "pl-c4 pl-thin"),
        Plot.dot(x + h, FT(x + h), "pl-c2", 4.5), Plot.dot(x, y, "pl-c4", 6),
      ]);
      const dir = Math.abs(m) < 0.05 ? "The slope is about zero: a **flat spot** (a minimum, maximum or saddle). Gradient descent stops here." : m > 0 ? "Slope positive: the function is going **up** to the right, so to reduce f we would move left." : "Slope negative: the function is going **down** to the right, so to reduce f we would move right.";
      return [{
        svg, side: { "f(x)": r3(y), "f′(x) exact": r3(m), "secant slope": r3(sec), "gap": r3(sec - m) },
        msg: `The amber dashed line is the **secant** through x and x + h: slope = (f(x+h) − f(x)) / h = ${r3(sec)}. Shrink h and it rotates onto the red **tangent line**, whose slope is the derivative f′(x) = ¾x² − 1 = ${r3(m)}. ${dir}`,
      }];
    },
  };

  /* ================================================================== 5. Gradient descent on a 2-D loss */
  const GA = [[3, 1], [1, 1]], GM = [1, 0.5];
  const GL = (x, y) => { const u = x - GM[0], v = y - GM[1]; return 0.5 * (GA[0][0] * u * u + 2 * GA[0][1] * u * v + GA[1][1] * v * v); };
  const GG = (x, y) => { const u = x - GM[0], v = y - GM[1]; return [GA[0][0] * u + GA[0][1] * v, GA[0][1] * u + GA[1][1] * v]; };
  const GE = jacobiEig(GA);
  V.math_gradpath = {
    title: "The gradient points uphill: follow its negative downhill",
    params: [
      { id: "lr", label: "learning rate η", min: 0.05, max: 0.62, step: 0.01, value: 0.2 },
      { id: "sx", label: "start w₁", min: -3, max: 4, step: 0.5, value: -3 },
      { id: "sy", label: "start w₂", min: -2, max: 3, step: 0.5, value: 2.5 },
    ],
    gen(_, P) {
      const S = [], path = [[P.sx, P.sy]];
      const base = (cur, step) => {
        const lay = [];
        [0.4, 1.2, 2.5, 4.5, 7, 10, 14].forEach((c) => {
          const pts = []; for (let i = 0; i <= 72; i++) { const t = (i / 72) * 2 * Math.PI; let x = GM[0], y = GM[1];
            for (let k = 0; k < 2; k++) { const r = Math.sqrt(2 * c / GE.vals[k]) * (k ? Math.sin(t) : Math.cos(t)); x += r * GE.vecs[k][0]; y += r * GE.vecs[k][1]; }
            pts.push([x, y]); }
          lay.push(Plot.line(pts, "pl-c1 pl-thin"));
        });
        for (let gx = -3.5; gx <= 4.5; gx += 1) for (let gy = -2.5; gy <= 3.5; gy += 1) {
          const g = GG(gx, gy), n = Math.hypot(g[0], g[1]) || 1;
          lay.push((Sx) => `<line class="pl-line pl-c1" x1="${Sx.sx(gx).toFixed(1)}" y1="${Sx.sy(gy).toFixed(1)}" x2="${Sx.sx(gx - 0.42 * g[0] / n).toFixed(1)}" y2="${Sx.sy(gy - 0.42 * g[1] / n).toFixed(1)}" marker-end="url(#ARROW)" style="opacity:.28"/>`);
        }
        lay.push(Plot.line(path, "pl-c3"), Plot.pts(path.slice(0, -1), "pl-c3", 3), Plot.dot(GM[0], GM[1], "pl-c5", 5));
        if (step) lay.push(arrow(cur[0], cur[1], cur[0] + step[0], cur[1] + step[1], "pl-c2"));
        lay.push(Plot.dot(cur[0], cur[1], "pl-c4", 6));
        return Plot.chart({ W: 480, H: 370, x: [-4, 5], y: [-3, 4], ticks: 9, yticks: 7 }, lay);
      };
      const frame = (msg, cur, step) => ({ svg: base(cur, step), side: { step: path.length - 1, "w₁": r3(cur[0]), "w₂": r3(cur[1]), "loss": r3(GL(cur[0], cur[1])), "|gradient|": r3(Math.hypot(...GG(cur[0], cur[1]))) }, msg });
      let cur = path[0], g = GG(...cur);
      S.push(frame(`The grey rings are lines of **equal loss** and the faint arrows show $-\\nabla L$ everywhere: always **perpendicular to the ring**, pointing to the purple minimum. Start at (${cur[0]}, ${cur[1]}) with loss ${r3(GL(...cur))}. The amber arrow is the step $-\\eta\\nabla L$.`, cur, [-P.lr * g[0], -P.lr * g[1]]));
      for (let i = 0; i < 30; i++) {
        g = GG(...cur); const nx = [cur[0] - P.lr * g[0], cur[1] - P.lr * g[1]];
        path.push(nx); cur = nx; g = GG(...cur);
        const L = GL(...cur), n = Math.hypot(...g);
        if (L > 1e3) { S.push(frame(`The loss is **exploding** (${r3(L)}): η is too large for the steep direction, so every step overshoots by more than it gained. Try a smaller η.`, [Math.max(-3.9, Math.min(4.9, cur[0])), Math.max(-2.9, Math.min(3.9, cur[1]))])); break; }
        const done = n < 0.02;
        S.push(frame(done ? `Converged: the gradient is almost zero (${r3(n)}), so we are at the bottom.` :
          `New position (${r3(cur[0])}, ${r3(cur[1])}), loss ${r3(L)}. Notice the zig-zag across the narrow valley: the gradient points straight across the rings, not along the valley.`, cur, done ? null : [-P.lr * g[0], -P.lr * g[1]]));
        if (done) break;
      }
      return S;
    },
  };

  /* ================================================================== 6. Normal distribution */
  V.math_normal = {
    title: "The normal distribution: area = probability",
    params: [
      { id: "mu", label: "mean μ", min: -2, max: 2, step: 0.1, value: 0 },
      { id: "sg", label: "std deviation σ", min: 0.5, max: 2.5, step: 0.1, value: 1 },
      { id: "lo", label: "shade from", min: -6, max: 6, step: 0.1, value: -1 },
      { id: "hi", label: "shade to", min: -6, max: 6, step: 0.1, value: 1 },
    ],
    gen(_, P) {
      const { mu, sg } = P, lo = Math.min(P.lo, P.hi), hi = Math.max(P.lo, P.hi);
      const pdf = (x) => Math.exp(-0.5 * ((x - mu) / sg) ** 2) / (sg * Math.sqrt(2 * Math.PI));
      const area = Phi((hi - mu) / sg) - Phi((lo - mu) / sg);
      const pts = [[lo, 0]]; for (let i = 0; i <= 60; i++) { const x = lo + ((hi - lo) * i) / 60; pts.push([x, pdf(x)]); } pts.push([hi, 0]);
      const svg = Plot.chart({ W: 520, H: 300, x: [-6, 6], y: [0, 0.85], xl: "x", yl: "density p(x)" }, [
        poly(pts, `fill:${soft("accent", 30)};stroke:none`), Plot.fn(pdf, "pl-c1"), Plot.vline(mu, "pl-c4 pl-dash pl-thin"),
        tag(mu, 0.04, "μ", "fill:var(--red);font-weight:700"),
      ]);
      const zl = (lo - mu) / sg, zh = (hi - mu) / sg;
      return [{
        svg, side: { "P(from ≤ X ≤ to)": (area * 100).toFixed(1) + "%", "z-scores": `${r3(zl)} to ${r3(zh)}`, "peak height": r3(pdf(mu)) },
        msg: `The curve is a **density**: probabilities are **areas** under it, not heights. The shaded area from ${r3(lo)} to ${r3(hi)} is **${(area * 100).toFixed(1)}%**. μ moves the bell sideways; σ widens it (and lowers the peak, because the total area must stay 1). Set the shading to μ ± σ, μ ± 2σ, μ ± 3σ to see the 68 / 95 / 99.7 rule.`,
      }];
    },
  };

  /* ================================================================== 7. Bayes and the base rate */
  V.math_bayes = {
    title: "Bayes' theorem: a 99% accurate test and a rare disease",
    params: [
      { id: "prev", label: "prevalence (% who have it)", min: 0.1, max: 50, step: 0.1, value: 1 },
      { id: "acc", label: "test accuracy (%)", min: 50, max: 99.9, step: 0.1, value: 99 },
    ],
    gen(_, P) {
      const prev = P.prev / 100, sens = P.acc / 100, N = 10000;
      const sick = N * prev, tp = sick * sens, fp = (N - sick) * (1 - sens), post = tp / (tp + fp);
      const cells = []; const nSick = Math.round(post * 100);
      for (let i = 0; i < 100; i++) {
        const r = Math.floor(i / 10), c = i % 10, on = i < nSick;
        cells.push(`<rect x="${290 + c * 25}" y="${60 + r * 25}" width="21" height="21" rx="4" style="fill:${on ? "var(--red)" : "var(--panel-2)"};stroke:${on ? "var(--red)" : "var(--border)"}"/>`);
      }
      const T = (x, y, s, st = "") => `<text x="${x}" y="${y}" style="font-size:12.5px;fill:var(--text);${st}">${esc(s)}</text>`;
      const svg = `<svg viewBox="0 0 560 330" width="560" height="330" xmlns="http://www.w3.org/2000/svg" role="img">
        ${T(14, 22, "Out of 10,000 people:", "font-weight:700")}
        ${T(14, 52, `${f2(sick, sick < 10 ? 1 : 0)} are sick  (${r3(P.prev)}%)`)}
        ${T(34, 74, `${f2(tp, tp < 10 ? 1 : 0)} test positive  (true positives)`, "fill:var(--red)")}
        ${T(14, 112, `${f2(N - sick, 0)} are healthy`)}
        ${T(34, 134, `${f2(fp, fp < 10 ? 1 : 0)} test positive  (false positives)`, "fill:var(--accent)")}
        ${T(14, 176, `All positives: ${f2(tp + fp, tp + fp < 10 ? 1 : 0)}`, "font-weight:700")}
        ${T(14, 198, `Truly sick among them: ${f2(tp, tp < 10 ? 1 : 0)} / ${f2(tp + fp, tp + fp < 10 ? 1 : 0)}`)}
        ${T(14, 236, `P(sick | positive) = ${(post * 100).toFixed(1)}%`, "font-size:16px;font-weight:700;fill:var(--red)")}
        ${T(290, 28, "100 people who tested positive:", "font-weight:700")}
        ${T(290, 46, "red = really sick, grey = false alarm", "fill:var(--muted);font-size:11.5px")}
        ${cells.join("")}
        ${T(290, 322, `${nSick} of 100 are really sick`, "fill:var(--muted);font-size:11.5px")}
      </svg>`;
      const verdict = post < 0.5 ? "Most positives are **false alarms**, because the healthy group is so much bigger than the sick group that even a small error rate on it produces many mistakes." : post > 0.9 ? "Now a positive result is very convincing." : "A positive result is useful but far from certain.";
      return [{
        svg, side: { "prior P(sick)": r3(P.prev) + "%", "P(+ | sick)": P.acc + "%", "posterior P(sick | +)": (post * 100).toFixed(1) + "%" },
        msg: `$P(\\text{sick}\\mid +) = \\dfrac{P(+\\mid\\text{sick})\\,P(\\text{sick})}{P(+)} = \\dfrac{${f2(tp, 1)}}{${f2(tp + fp, 1)}} = ${(post * 100).toFixed(1)}\\%$. ${verdict} Slide the prevalence up and the same test becomes trustworthy: the **prior** matters as much as the evidence.`,
      }];
    },
  };

  /* ================================================================== 8. Entropy, cross-entropy, KL */
  V.math_entropy = {
    title: "Entropy, cross-entropy and KL divergence (in bits)",
    params: [
      { id: "p1", label: "true p: weight of A", min: 0.05, max: 1, step: 0.05, value: 0.8 },
      { id: "p2", label: "true p: weight of B", min: 0.05, max: 1, step: 0.05, value: 0.15 },
      { id: "p3", label: "true p: weight of C", min: 0.05, max: 1, step: 0.05, value: 0.05 },
      { id: "q1", label: "model q: weight of A", min: 0.05, max: 1, step: 0.05, value: 0.4 },
      { id: "q2", label: "model q: weight of B", min: 0.05, max: 1, step: 0.05, value: 0.4 },
      { id: "q3", label: "model q: weight of C", min: 0.05, max: 1, step: 0.05, value: 0.2 },
    ],
    gen(_, P) {
      const pw = [P.p1, P.p2, P.p3], qw = [P.q1, P.q2, P.q3], sp = sum(pw), sq = sum(qw);
      const p = pw.map((v) => v / sp), q = qw.map((v) => v / sq);
      const H = -sum(p.map(L2)), CE = -sum(p.map((v, i) => v * Math.log2(q[i]))), KL = CE - H;
      const bw = 38, X0 = 56, base = 190, hs = 150;
      let s = `<svg viewBox="0 0 560 330" width="560" height="330" xmlns="http://www.w3.org/2000/svg" role="img">`;
      s += `<line x1="40" y1="${base}" x2="270" y2="${base}" style="stroke:var(--border)"/>`;
      ["A", "B", "C"].forEach((nm, i) => {
        const x = X0 + i * 72;
        s += `<rect x="${x}" y="${base - p[i] * hs}" width="${bw}" height="${p[i] * hs}" rx="3" style="fill:var(--accent)"/><text x="${x + bw / 2}" y="${base - p[i] * hs - 5}" text-anchor="middle" style="font-size:11px;fill:var(--text)">${f2(p[i])}</text>`;
        s += `<rect x="${x + bw + 3}" y="${base - q[i] * hs}" width="${bw - 10}" height="${q[i] * hs}" rx="3" style="fill:var(--amber)"/><text x="${x + bw + 3 + (bw - 10) / 2}" y="${base - q[i] * hs - 5}" text-anchor="middle" style="font-size:11px;fill:var(--text)">${f2(q[i])}</text>`;
        s += `<text x="${x + bw / 2 + 10}" y="${base + 16}" text-anchor="middle" style="font-size:12px;fill:var(--muted)">${nm}</text>`;
      });
      s += `<rect x="300" y="18" width="12" height="12" style="fill:var(--accent)"/><text x="318" y="29" style="font-size:12px;fill:var(--text)">p: the truth</text><rect x="300" y="38" width="12" height="12" style="fill:var(--amber)"/><text x="318" y="49" style="font-size:12px;fill:var(--text)">q: the model's guess</text>`;
      const scale = Math.max(3, CE * 1.08), W2 = 230, px = (v) => (v / scale) * W2;
      s += `<text x="300" y="100" style="font-size:12px;fill:var(--text);font-weight:700">Cross-entropy H(p, q) = ${r3(CE)} bits</text>`;
      s += `<rect x="300" y="108" width="${px(H)}" height="26" style="fill:var(--accent)"/><rect x="${300 + px(H)}" y="108" width="${Math.max(px(KL), 0)}" height="26" style="fill:var(--red)"/>`;
      s += `<text x="300" y="152" style="font-size:12px;fill:var(--accent)">entropy H(p) = ${r3(H)}</text><text x="300" y="170" style="font-size:12px;fill:var(--red)">+ KL(p ‖ q) = ${r3(KL)}  (the penalty for q ≠ p)</text>`;
      s += `<text x="300" y="206" style="font-size:12px;fill:var(--muted)">max entropy for 3 outcomes: ${r3(Math.log2(3))} bits</text>`;
      s += `<text x="40" y="236" style="font-size:12px;fill:var(--muted)">surprise of each outcome under q:  −log₂ q = ${q.map((v) => r3(-Math.log2(v))).join("  ·  ")}</text>`;
      s += `</svg>`;
      const note = KL < 0.005 ? "q matches p, so the KL penalty is zero and cross-entropy bottoms out at the entropy." : KL > 1 ? "q is badly wrong about something that matters (it gives little probability to a likely outcome), and the penalty is large." : "q is somewhat off, so you pay a KL penalty on top of the unavoidable entropy.";
      return [{
        svg: s, side: { "H(p)": r3(H), "H(p, q)": r3(CE), "KL(p‖q)": r3(KL) },
        msg: `Entropy $H(p)=-\\sum p\\log_2 p$ is the average surprise when outcomes follow p. Cross-entropy $H(p,q)=-\\sum p\\log_2 q$ is the average surprise if you *believe* q. They differ by exactly the KL divergence: $H(p,q)=H(p)+\\mathrm{KL}(p\\|q)$ and KL ≥ 0. ${note} Training a classifier with cross-entropy pushes q toward p.`,
      }];
    },
  };

  /* ================================================================== 9. SVD: low-rank approximation of a small image */
  const IMG = (() => {
    const n = 12, A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
      let v = Math.exp(-(((i - 5.5) ** 2) + ((j - 5.5) ** 2)) / 14);                    // a soft blob
      if (Math.abs(i - j) <= 1) v += 0.5;                                                  // a diagonal band
      if (i >= 8 && j <= 3) v += 0.55;                                                     // a corner patch
      return v;
    }));
    const AtA = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => sum(A.map((r) => r[i] * r[j])))), E = jacobiEig(AtA);
    const sv = E.vals.map((v) => Math.sqrt(Math.max(v, 0)));
    const Uc = E.vecs.map((vv, k) => A.map((r) => (sv[k] > 1e-9 ? sum(r.map((a, j) => a * vv[j])) / sv[k] : 0)));   // k-th left vector
    return { n, A, sv, Vr: E.vecs, Uc };
  })();
  V.math_svdrank = {
    title: "SVD: rebuild a picture from its top k singular values",
    params: [{ id: "k", label: "rank k", min: 1, max: 12, step: 1, value: 2 }],
    gen(_, P) {
      const { n, A, sv, Vr, Uc } = IMG, k = P.k;
      const R = Array.from({ length: n }, () => new Array(n).fill(0));
      for (let t = 0; t < k; t++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) R[i][j] += sv[t] * Uc[t][i] * Vr[t][j];
      let err = 0, tot = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { err += (A[i][j] - R[i][j]) ** 2; tot += A[i][j] ** 2; }
      const cs = 15, heat = (M, ox, oy) => M.map((r, i) => r.map((v, j) => `<rect x="${ox + j * cs}" y="${oy + i * cs}" width="${cs - 1}" height="${cs - 1}" style="fill:color-mix(in srgb, var(--accent) ${Math.round(Math.max(0, Math.min(1, v / 1.4)) * 100)}%, var(--panel-2))"/>`).join("")).join("");
      const e2 = sv.map((v) => v * v), tot2 = sum(e2), cum = sum(e2.slice(0, k)) / tot2, mx = sv[0];
      let s = `<svg viewBox="0 0 640 260" width="640" height="260" xmlns="http://www.w3.org/2000/svg" role="img">`;
      s += `<text x="10" y="16" style="font-size:12px;fill:var(--text);font-weight:700">original A (12 × 12)</text>${heat(A, 10, 26)}`;
      s += `<text x="215" y="16" style="font-size:12px;fill:var(--text);font-weight:700">rank-${k} rebuild</text>${heat(R, 215, 26)}`;
      s += `<text x="420" y="16" style="font-size:12px;fill:var(--text);font-weight:700">singular values σ</text>`;
      sv.forEach((v, i) => { const h = (v / mx) * 150, x = 424 + i * 17; s += `<rect x="${x}" y="${190 - h}" width="13" height="${h}" rx="2" style="fill:${i < k ? "var(--accent)" : "var(--panel-2)"};stroke:var(--accent)"/><text x="${x + 6.5}" y="206" text-anchor="middle" style="font-size:9.5px;fill:var(--muted)">${i + 1}</text>`; });
      s += `<text x="10" y="238" style="font-size:12px;fill:var(--muted)">numbers stored: ${k}×(12+12+1) = ${k * 25} instead of 144</text></svg>`;
      return [{
        svg: s, side: { "rank k": k, "energy kept": (cum * 100).toFixed(1) + "%", "relative error": r3(Math.sqrt(err / tot)), "σ₁ : σ₂": r3(sv[0] / sv[1]) },
        msg: `$A=\\sum_i \\sigma_i\\,u_i v_i^{\\top}$: a sum of simple rank-1 layers, biggest first. Keeping only the top ${k} layers keeps ${(cum * 100).toFixed(1)}% of the "energy" $\\sum\\sigma^2$. The singular values drop fast, so a few layers already look like the original. At k = 12 the rebuild is exact. This is why PCA and LoRA work with low-rank pieces.`,
      }];
    },
  };

  /* ================================================================== 10. Saddle points: curvature in two directions */
  V.math_saddle = {
    title: "Curvature decides: minimum, saddle point or flat valley",
    params: [{ id: "lam", label: "curvature λ along y", min: -1.5, max: 2, step: 0.1, value: -0.8 }],
    gen(_, P) {
      const lam = Math.abs(P.lam) < 1e-9 ? 0 : P.lam, f = (x, y) => x * x + lam * y * y;
      const eta = 0.1; let x = 1, y = 0.05; for (let i = 0; i < 25; i++) { x -= eta * 2 * x; y -= eta * 2 * lam * y; }
      const svg = Plot.chart({ W: 520, H: 300, x: [-2, 2], y: [-4, 4], xl: "distance from the critical point along an axis", yl: "f" }, [
        Plot.fn((t) => t * t, "pl-c1"), Plot.fn((t) => lam * t * t, "pl-c4"), Plot.hline(0, "pl-dash"),
        tag(1.45, 3.4, "slice along x:  x²", "fill:var(--accent)"), tag(1.45, lam > 0.3 ? 3.4 * 0.6 * lam : lam * 2.6 + (lam < 0 ? 0.1 : -0.5) , "slice along y:  " + r3(lam) + "·y²", "fill:var(--red)"),
      ]);
      const kind = lam > 1e-9 ? ["a **minimum**", "Every direction curves up, like a bowl."] : lam < -1e-9 ? ["a **saddle point**", "It curves up along x but **down** along y: a minimum in one direction, a maximum in another, like a horse saddle. The gradient is zero here, yet it is not a minimum."] : ["a **flat valley**", "No curvature along y: a whole line of equally good points."];
      return [{
        svg, side: { "Hessian eigenvalues": `2, ${r3(2 * lam)}`, "gradient at (0,0)": "(0, 0)", "after 25 GD steps": `y = ${Math.abs(y) > 1e4 ? y.toExponential(1) : r3(y)}` },
        msg: `$f(x,y)=x^2+${r3(lam)}y^2$ has zero gradient at (0, 0). The sign of the curvature in each direction (the Hessian eigenvalues 2 and ${r3(2 * lam)}) says what it is: ${kind[0]}. ${kind[1]} Gradient descent started almost on the axis (y = 0.05) ends with y = ${Math.abs(y) > 1e4 ? y.toExponential(1) : r3(y)} after 25 steps: it ${lam < -1e-9 ? "slides **away** from the saddle once the tiny y part is amplified" : "settles down"}.`,
      }];
    },
  };

  /* ------------------------------------------------------------------ diagrams */
  const mk = (id) => `<defs><marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker><marker id="${id}r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--red)"/></marker></defs>`;
  const wrap = (W, H, inner, id) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">${mk(id)}${inner}</svg>`;
  const bx = (x, y, w, h, label, cls = "node") => `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="8"/><text class="val" x="${x + w / 2}" y="${y + h / 2 + 5}" text-anchor="middle" style="font-size:13px">${esc(label)}</text>`;
  const tx = (x, y, s, style = "", anchor = "middle") => `<text class="lbl" x="${x}" y="${y}" text-anchor="${anchor}" style="${style}">${esc(s)}</text>`;

  // L = (w·x + b − y)²  with x=2, w=3, b=1, y=5: forward values above, backward gradients below
  D.math_chain = () => {
    const id = "mch", nodes = [["x = 2", 0], ["× w", 1], ["+ b", 2], ["− y", 3], ["( )²", 4]];
    const fwd = ["2", "6", "7", "2", "4"], X = [60, 170, 280, 390, 500];
    let s = tx(300, 18, "forward pass (values)", "font-size:12px;font-weight:700;fill:var(--accent)");
    nodes.forEach(([t], i) => { s += bx(X[i] - 42, 60, 84, 44, t, i === 4 ? "node ok" : "node"); s += tx(X[i], 50, i === 0 ? "x" : i === 4 ? "L = 4" : "= " + fwd[i], "font-size:12px;fill:var(--accent)"); if (i < 4) s += `<line class="edge" x1="${X[i] + 44}" y1="82" x2="${X[i + 1] - 46}" y2="82" marker-end="url(#${id})"/>`; });
    s += tx(300, 150, "backward pass (each arrow multiplies by a local derivative)", "font-size:12px;font-weight:700;fill:var(--red)");
    const g = ["∂L/∂x = 12", "∂L/∂(wx) = 4", "∂L/∂p = 4", "∂L/∂e = 4", "∂L/∂L = 1"], loc = ["× w = 3", "× 1", "× 1", "× 2e = 4", ""];
    nodes.forEach((_, i) => { s += tx(X[i], 128, "", ""); s += tx(X[i], 178, g[i], "font-size:11.5px;fill:var(--red)"); if (i < 4) s += `<line x1="${X[i + 1] - 46}" y1="196" x2="${X[i] + 44}" y2="196" style="stroke:var(--red);stroke-width:1.6" marker-end="url(#${id}r)"/>` + tx((X[i] + X[i + 1]) / 2, 214, loc[i], "font-size:11px;fill:var(--red)"); });
    s += tx(300, 246, "also: ∂L/∂w = 4 × x = 8   and   ∂L/∂b = 4 × 1 = 4", "font-size:12px;fill:var(--text)");
    return wrap(600, 262, s, id);
  };

  // a probability picture: a box of all outcomes with events A and B
  D.math_prob = () => {
    const id = "mpr";
    let s = `<rect x="10" y="22" width="280" height="190" rx="8" style="fill:var(--panel);stroke:var(--border)"/>` + tx(150, 16, "all possible outcomes (area = 1)", "font-size:11.5px");
    s += `<circle cx="110" cy="118" r="62" style="fill:${soft("accent", 22)};stroke:var(--accent)"/><circle cx="190" cy="118" r="62" style="fill:${soft("green", 22)};stroke:var(--green)"/>`;
    s += tx(78, 122, "A", "font-size:15px;font-weight:700;fill:var(--accent)") + tx(222, 122, "B", "font-size:15px;font-weight:700;fill:var(--green)") + tx(150, 122, "A and B", "font-size:11.5px;fill:var(--text)");
    s += tx(450, 70, "P(A | B) = P(A and B) / P(B)", "font-size:13px;font-weight:700;fill:var(--text)");
    s += tx(450, 98, "Once we know B happened, the world shrinks", "font-size:12px") + tx(450, 116, "to the green circle. What fraction of it", "font-size:12px") + tx(450, 134, "is also inside A? That is P(A | B).", "font-size:12px");
    s += tx(450, 170, "A and B independent  ⇔  P(A and B) = P(A) P(B)", "font-size:12px;fill:var(--muted)");
    return wrap(620, 226, s, id);
  };

  // the SVD of a 2x2 matrix: rotate, stretch, rotate
  D.math_svd = () => {
    const id = "msv"; let s = "";
    const panel = (cx, title, rot, sx, sy, extra = "") => {
      let o = `<g transform="translate(${cx},120)">` + `<line x1="-70" y1="0" x2="70" y2="0" style="stroke:var(--border)"/><line x1="0" y1="-70" x2="0" y2="70" style="stroke:var(--border)"/>`;
      o += `<g transform="rotate(${rot})"><ellipse rx="${sx}" ry="${sy}" style="fill:${soft("accent", 12)};stroke:var(--accent);stroke-width:1.6"/>`;
      o += `<line x1="0" y1="0" x2="${sx}" y2="0" style="stroke:var(--red);stroke-width:2.2"/><line x1="0" y1="0" x2="0" y2="${-sy}" style="stroke:var(--green);stroke-width:2.2"/></g></g>`;
      return o + tx(cx, 22, title, "font-size:12px;font-weight:700");
    };
    s += panel(80, "unit circle", 0, 42, 42) + panel(250, "1. rotate (Vᵀ)", -32, 42, 42) + panel(420, "2. stretch (Σ)", -32, 62, 20) + panel(590, "3. rotate (U)", 18, 62, 20);
    [[145, 175], [315, 345], [485, 515]].forEach(([a, b]) => { s += `<line class="edge" x1="${a}" y1="120" x2="${b}" y2="120" marker-end="url(#${id})"/>`; });
    s += tx(335, 214, "Any matrix = rotate, then stretch along the axes, then rotate. The stretch amounts are the singular values σ₁ ≥ σ₂ ≥ 0.", "font-size:12px;fill:var(--muted)");
    return wrap(670, 230, s, id);
  };

  // 1-D loss landscapes: convex bowl vs a bumpy one with a saddle-like flat spot
  D.math_landscape = () => {
    const id = "mla"; let s = "";
    const curve = (f, x0, y0, w, h, n = 120) => { let d = ""; for (let i = 0; i <= n; i++) { const t = i / n; d += (i ? "L" : "M") + (x0 + t * w).toFixed(1) + "," + (y0 + h - f(t) * h).toFixed(1); } return d; };
    const bowl = (t) => 0.1 + 3.2 * (t - 0.5) ** 2 * 1.0;
    const bumpy = (t) => 0.55 + 0.22 * Math.sin(t * 17 + 0.3) * 0.75 + 0.5 * (t - 0.62) ** 2 - 0.3 * Math.exp(-((t - 0.78) ** 2) / 0.004) + 0.25 * (t - 0.3) ** 3 * 4;
    s += `<g><rect x="10" y="30" width="270" height="150" rx="8" style="fill:var(--panel);stroke:var(--border)"/><path d="${curve(bowl, 20, 40, 250, 130)}" style="fill:none;stroke:var(--accent);stroke-width:2.2"/>`;
    s += `<circle cx="145" cy="${40 + 130 - bowl(0.5) * 130}" r="5.5" style="fill:var(--green)"/>` + tx(145, 22, "convex: one valley, so one answer", "font-size:12px;font-weight:700") + tx(145, 198, "global minimum = the only minimum", "font-size:11.5px;fill:var(--green)") + `</g>`;
    const ys = (t) => 40 + 130 - bumpy(t) * 130;
    s += `<g><rect x="320" y="30" width="290" height="150" rx="8" style="fill:var(--panel);stroke:var(--border)"/><path d="${curve(bumpy, 330, 40, 270, 130)}" style="fill:none;stroke:var(--accent);stroke-width:2.2"/>`;
    let best = 0, bi = 0; const mins = [];
    for (let i = 2; i < 118; i++) { const a = bumpy((i - 1) / 120), b = bumpy(i / 120), c = bumpy((i + 1) / 120); if (b < a && b < c) mins.push(i / 120); }
    mins.forEach((t, k) => { s += `<circle cx="${(330 + t * 270).toFixed(1)}" cy="${ys(t).toFixed(1)}" r="5.5" style="fill:${k === 0 ? "var(--amber)" : "var(--green)"}"/>`; });
    s += tx(465, 22, "non-convex: many valleys and flat spots", "font-size:12px;font-weight:700") + tx(465, 198, "local minima (dots): descent can stop in any of them", "font-size:11.5px;fill:var(--amber)") + `</g>`;
    return wrap(620, 212, s, id);
  };

  // shapes: y = f(x), J has one row per output and one column per input
  D.math_shapes = () => {
    const id = "msh";
    let s = tx(310, 16, "Jacobian J: how every output reacts to every input", "font-size:12px;font-weight:700");
    s += bx(20, 40, 60, 100, "x", "node") + tx(50, 156, "n inputs", "font-size:11px") + tx(105, 95, "→  f  →", "font-size:12px");
    s += bx(140, 50, 42, 80, "y", "node hl") + tx(161, 146, "m outputs", "font-size:11px");
    s += tx(230, 95, "J =", "font-size:15px;font-weight:700");
    for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) s += `<rect x="${262 + c * 38}" y="${44 + r * 34}" width="34" height="30" rx="4" style="fill:var(--panel-2);stroke:var(--border)"/><text class="lbl" x="${279 + c * 38}" y="${64 + r * 34}" text-anchor="middle" style="font-size:10.5px">${"∂y" + (r + 1) + "/∂x" + (c + 1)}</text>`;
    s += tx(358, 160, "n columns (one per input)", "font-size:11px") + `<text class="lbl" x="250" y="95" text-anchor="end" style="font-size:11px" transform="rotate(-90 250 95)">m rows</text>`;
    s += tx(520, 70, "shape of J: (m × n)", "font-size:13px;font-weight:700;fill:var(--accent)") + tx(520, 96, "a tiny change dx in the input", "font-size:12px") + tx(520, 114, "changes the output by J · dx", "font-size:12px");
    s += tx(520, 144, "for a scalar loss (m = 1) J is a", "font-size:12px;fill:var(--muted)") + tx(520, 160, "single row: the gradient (transposed)", "font-size:12px;fill:var(--muted)");
    return wrap(660, 176, s, id);
  };

  /* ------------------------------------------------------------------ glossary (plain-English hover definitions) */
  Object.assign(window.GLOSSARY, {
    "vector": "An ordered list of numbers, like (3, 1). You can picture it as an arrow from the origin, or as a point. In ML it is one example's features, an embedding, or a set of weights.",
    "scalar": "A single ordinary number (as opposed to a vector or a matrix).",
    "dot product": "Multiply two vectors position by position and add up the results. It is large and positive when they point the same way, zero when perpendicular, negative when opposed.",
    "norm": "The length of a vector. The usual one (L2) is the square root of the sum of squares of its entries.",
    "unit vector": "A vector of length 1. Dividing any vector by its length gives a unit vector pointing the same way.",
    "cosine similarity": "The cosine of the angle between two vectors: their dot product divided by both lengths. 1 means the same direction, 0 unrelated, -1 opposite. It ignores length.",
    "orthogonal": "Perpendicular. Two vectors are orthogonal when their dot product is zero.",
    "projection": "The shadow of one vector onto another: the part of it that points along that direction.",
    "basis": "A set of vectors that you can combine (with weights) to reach any point. The standard basis in 2-D is (1, 0) and (0, 1).",
    "matrix": "A rectangular grid of numbers. It can be a table of data, or a machine that turns one vector into another (a linear transformation).",
    "transpose": "A matrix flipped over its diagonal: rows become columns. Written with a small T.",
    "identity matrix": "The matrix of ones on the diagonal and zeros elsewhere. Multiplying by it changes nothing, like multiplying a number by 1.",
    "inverse": "The matrix that undoes another one: A⁻¹A = I. Only square matrices that do not squash space flat have one.",
    "determinant": "A single number saying how much a matrix scales area (or volume). Zero means it squashes space flat, so there is no inverse.",
    "rank": "The number of independent directions a matrix really has. A rank-1 matrix is a single outer product, and a low-rank matrix can be stored very compactly.",
    "eigenvector": "A direction that a matrix only stretches (or flips), never turns. The matrix sends v to λv.",
    "eigenvalue": "The stretch factor λ that goes with an eigenvector: how much the matrix scales that direction.",
    "singular value": "A non-negative number from the SVD saying how much a matrix stretches along one of its main directions. Big ones carry the signal, small ones are often noise.",
    "svd": "Singular value decomposition: any matrix can be written as rotate, stretch along axes, rotate. It exposes the matrix's most important directions and underlies PCA and low-rank methods.",
    "linear transformation": "A function on vectors that keeps straight lines straight and the origin fixed. Every one is multiplication by some matrix.",
    "derivative": "How fast a function changes at a point: the slope of its tangent line. It answers 'if I nudge the input a little, how much does the output move?'",
    "tangent line": "The straight line that just touches a curve at one point and has the same slope as the curve there.",
    "partial derivative": "The derivative with respect to one input while all the other inputs are held fixed.",
    "chain rule": "To differentiate a function of a function, multiply the derivatives along the chain. It is the heart of backpropagation.",
    "jacobian": "The matrix of all partial derivatives of a vector-valued function: one row per output, one column per input.",
    "hessian": "The matrix of second derivatives of a function. It describes the curvature of a loss surface.",
    "critical point": "A point where the gradient is zero. It can be a minimum, a maximum or a saddle point.",
    "saddle point": "A point where the gradient is zero but the surface curves up in some directions and down in others, like a horse saddle.",
    "local minimum": "A point lower than everything near it, but not necessarily the lowest point overall.",
    "convex": "A bowl-shaped function: the line between any two points on its graph lies above the graph. It has no false valleys, so any local minimum is the global one.",
    "probability": "A number between 0 and 1 saying how likely an event is. 0 means impossible, 1 means certain.",
    "conditional probability": "The probability of one event given that another has happened, written P(A | B).",
    "independent": "Two events are independent when knowing one tells you nothing about the other, so P(A and B) = P(A)P(B).",
    "bayes' theorem": "The rule that flips a conditional probability: P(A | B) = P(B | A) P(A) / P(B). It tells you how to update a belief after seeing evidence.",
    "prior": "What you believe about something before seeing the new evidence.",
    "posterior": "What you believe after updating the prior with the evidence (via Bayes' theorem).",
    "likelihood": "How probable the observed data is under a particular setting of the model's parameters.",
    "base rate": "How common something is in the whole population. Ignoring it is the classic reason a 'accurate' test gives mostly false alarms.",
    "random variable": "A quantity whose value depends on chance, like the roll of a die or the height of a random person.",
    "expectation": "The average value of a random variable, weighted by how likely each value is. Also called the mean or expected value.",
    "variance": "How spread out a random variable is: the average squared distance from its mean.",
    "standard deviation": "The square root of the variance: a spread measured in the same units as the data.",
    "normal distribution": "The bell curve, described by its mean and standard deviation. Sums of many small independent effects tend to look normal.",
    "bernoulli": "The distribution of a single yes/no event that is 'yes' with probability p, like one coin flip.",
    "categorical distribution": "A distribution over a few possible outcomes with a probability for each, adding to 1. A softmax output is one.",
    "i.i.d.": "Independent and identically distributed: each example is drawn separately from the same distribution. The usual assumption about training data.",
    "maximum likelihood": "Choose the parameters that make the observed data most probable. Minimising squared error and cross-entropy are special cases.",
    "mle": "Maximum likelihood estimation: pick the parameters under which the data you saw is most likely.",
    "negative log-likelihood": "The minus log of the likelihood. Turning a product of probabilities into a sum makes it easy to minimise, and it is what cross-entropy loss computes.",
    "entropy": "How surprising a distribution is on average, in bits or nats. A fair coin has 1 bit; a certain outcome has 0.",
    "kl divergence": "A measure of how different distribution q is from p: the extra surprise you pay for believing q when the truth is p. Never negative, and zero only when they match.",
    "nat": "The unit of information when logarithms are natural (base e). With base 2 the unit is the bit.",
  });
})();
