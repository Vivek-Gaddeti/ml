/* Machine Learning, part 1 (foundations, regression, classification): visualizers, diagrams and glossary terms.
   Loaded on every doc page and on Practice, after assets/viz.js. Every number shown is computed, not faked. */
(function () {
  const V = Viz.defs, Plot = Viz.Plot, { r3 } = Viz.helpers;

  /* ------------------------------------------------------------------ private helpers */
  const rng = (seed) => { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  const gauss = (r) => { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const mean = (a) => sum(a) / a.length;
  const mse = (a, b) => mean(a.map((v, i) => (v - b[i]) ** 2));
  const f2 = (v, d = 2) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : +v).toFixed(d);
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const pct = (v) => Math.round(v * 100) + "%";
  // put several SVGs (or charts) next to each other; they wrap on narrow screens
  const row = (...parts) => `<div style="display:flex;flex-wrap:wrap;justify-content:center;align-items:flex-start;gap:6px">${parts.join("")}</div>`;
  const col = (...parts) => `<div style="display:flex;flex-direction:column;align-items:center;gap:4px">${parts.join("")}</div>`;
  // a custom Plot layer: rectangle in data coordinates with a CSS style
  const rect = (x0, y0, x1, y1, style) => (S) => `<rect x="${S.sx(Math.min(x0, x1)).toFixed(1)}" y="${S.sy(Math.max(y0, y1)).toFixed(1)}" width="${Math.abs(S.sx(x1) - S.sx(x0)).toFixed(1)}" height="${Math.abs(S.sy(y1) - S.sy(y0)).toFixed(1)}" style="${style}"/>`;
  const ring = (x, y, r, style) => (S) => `<circle cx="${S.sx(x).toFixed(1)}" cy="${S.sy(y).toFixed(1)}" r="${r}" style="fill:none;${style}"/>`;
  const tag = (x, y, s, style = "", anchor = "middle", size = 11) => (S) => `<text x="${S.sx(x).toFixed(1)}" y="${S.sy(y).toFixed(1)}" text-anchor="${anchor}" style="font-size:${size}px;fill:var(--muted);${style}">${esc(s)}</text>`;
  // step function through sorted [x, y] knots (for tree predictions on 1-D data)
  const stepLine = (f, x0, x1, n = 400) => { const p = []; for (let i = 0; i <= n; i++) { const x = x0 + ((x1 - x0) * i) / n; p.push([x, f(x)]); } return p; };

  // least squares: minimise ||A w − y||² + lam·||w||² (lam on the columns where pen[j] is true), via Householder QR
  function lstsq(A, y, lam = 0, pen) {
    const p = A[0].length, M = A.map((r) => r.slice()), b = y.slice();
    if (lam > 0) for (let j = 0; j < p; j++) { if (pen && !pen[j]) continue; const r = new Array(p).fill(0); r[j] = Math.sqrt(lam); M.push(r); b.push(0); }
    const m = M.length;
    for (let k = 0; k < Math.min(p, m); k++) {
      let nrm = 0; for (let i = k; i < m; i++) nrm += M[i][k] * M[i][k]; nrm = Math.sqrt(nrm);
      if (nrm < 1e-14) continue;
      const al = M[k][k] > 0 ? -nrm : nrm, v = new Array(m).fill(0);
      for (let i = k; i < m; i++) v[i] = M[i][k]; v[k] -= al;
      let vv = 0; for (let i = k; i < m; i++) vv += v[i] * v[i];
      if (vv < 1e-30) continue;
      for (let j = k; j < p; j++) { let s = 0; for (let i = k; i < m; i++) s += v[i] * M[i][j]; s = (2 * s) / vv; for (let i = k; i < m; i++) M[i][j] -= s * v[i]; }
      let s = 0; for (let i = k; i < m; i++) s += v[i] * b[i]; s = (2 * s) / vv; for (let i = k; i < m; i++) b[i] -= s * v[i];
    }
    const w = new Array(p).fill(0);
    for (let k = Math.min(p, m) - 1; k >= 0; k--) { let s = b[k]; for (let j = k + 1; j < p; j++) s -= M[k][j] * w[j]; w[k] = Math.abs(M[k][k]) < 1e-12 ? 0 : s / M[k][k]; }
    return w;
  }
  // Chebyshev features T_0..T_d of t = 2x − 1 (x in [0, 1]): the same curves as 1, x, …, x^d but numerically stable
  const cheb = (x, d) => { const t = 2 * x - 1, T = [1, t]; for (let k = 2; k <= d; k++) T.push(2 * t * T[k - 1] - T[k - 2]); return T.slice(0, d + 1); };
  const polyFit = (xs, ys, d, lam = 1e-12) => { const w = lstsq(xs.map((x) => cheb(x, d)), ys, lam); return (x) => { const T = cheb(x, d); let s = 0; for (let k = 0; k <= d; k++) s += w[k] * T[k]; return s; }; };

  // 1-D regression tree (squared error): returns a predict function. idx = training indices (may repeat: bootstrap)
  function regTree(xs, ys, idx, maxDepth) {
    const build = (ids, depth) => {
      const m = mean(ids.map((i) => ys[i]));
      if (depth >= maxDepth || ids.length < 2) return { v: m };
      const s = ids.slice().sort((a, b) => xs[a] - xs[b]), n = s.length;
      let tot = 0, tot2 = 0; s.forEach((i) => { tot += ys[i]; tot2 += ys[i] * ys[i]; });
      let best = null, ls = 0, ls2 = 0;
      for (let k = 0; k < n - 1; k++) {
        const v = ys[s[k]]; ls += v; ls2 += v * v;
        if (xs[s[k]] === xs[s[k + 1]]) continue;
        const nl = k + 1, nr = n - nl, rs = tot - ls, rs2 = tot2 - ls2;
        const sse = ls2 - (ls * ls) / nl + rs2 - (rs * rs) / nr;
        if (!best || sse < best.sse - 1e-12) best = { sse, t: (xs[s[k]] + xs[s[k + 1]]) / 2, k };
      }
      if (!best || best.sse >= tot2 - (tot * tot) / n - 1e-12) return { v: m };
      return { t: best.t, l: build(s.slice(0, best.k + 1), depth + 1), r: build(s.slice(best.k + 1), depth + 1) };
    };
    const root = build(idx, 0);
    return (x) => { let nd = root; while (nd.t != null) nd = x <= nd.t ? nd.l : nd.r; return nd.v; };
  }

  /* ================================================================== 1. Linear regression: residuals and MSE */
  const LIN = (() => {
    const r = rng(7), xs = [], ys = [];
    for (let i = 0; i < 12; i++) { const x = 0.6 + i * 0.78 + (r() - 0.5) * 0.5; xs.push(+x.toFixed(2)); ys.push(+(0.8 * x + 2 + gauss(r) * 1.1).toFixed(2)); }
    const mx = mean(xs), my = mean(ys);
    const w = sum(xs.map((x, i) => (x - mx) * (ys[i] - my))) / sum(xs.map((x) => (x - mx) ** 2)), b = my - w * mx;
    return { xs, ys, w, b, best: mse(xs.map((x) => w * x + b), ys) };
  })();
  V.ml1_linfit = {
    title: "Fit a line by hand: residuals and mean squared error",
    params: [
      { id: "w", label: "slope w", min: -1, max: 2.5, step: 0.05, value: 0.2 },
      { id: "b", label: "intercept b", min: -3, max: 8, step: 0.1, value: 1.5 },
      { id: "sq", label: "draw squares", min: 0, max: 1, step: 1, value: 1, fmt: (v) => (+v ? "on" : "off") },
    ],
    gen(_, P) {
      const { xs, ys } = LIN, pred = xs.map((x) => P.w * x + P.b), res = ys.map((y, i) => y - pred[i]), L = mean(res.map((e) => e * e));
      const squares = (S) => xs.map((x, i) => { const y0 = S.sy(pred[i]), y1 = S.sy(ys[i]), s = Math.abs(y1 - y0), px = S.sx(x);
        return `<rect x="${px.toFixed(1)}" y="${Math.min(y0, y1).toFixed(1)}" width="${s.toFixed(1)}" height="${s.toFixed(1)}" style="fill:color-mix(in srgb, var(--red) 13%, transparent);stroke:color-mix(in srgb, var(--red) 45%, transparent);stroke-width:1"/>`; }).join("");
      const svg = Plot.chart({ x: [0, 10.5], y: [-1, 14], xl: "x (e.g. house size)", yl: "y (e.g. price)", H: 320 }, [
        P.sq ? squares : "",
        ...xs.map((x, i) => Plot.seg(x, pred[i], x, ys[i], "pl-c4 pl-thin")),
        Plot.fn((x) => LIN.w * x + LIN.b, "pl-dash"),
        Plot.fn((x) => P.w * x + P.b, "pl-c1"),
        Plot.pts(xs.map((x, i) => [x, ys[i]]), "pl-c1", 4.5),
      ]);
      const gap = L - LIN.best;
      const hint = gap < 0.02 ? "That is the **least-squares line**: no other slope and intercept give a smaller MSE." :
        gap < 0.5 ? "Very close to the best line (dashed). Fine-tune w and b to squeeze out the last bit." :
          res.filter((e) => e > 0).length > 9 ? "Most points are **above** the line: raise it (bigger b or w)." :
            res.filter((e) => e < 0).length > 9 ? "Most points are **below** the line: lower it." :
              "Tilt the line (w) and shift it (b) until the red residuals are small overall.";
      return [{
        svg, side: { "your MSE": r3(L), "best MSE": r3(LIN.best), "best w": r3(LIN.w), "best b": r3(LIN.b) },
        msg: `Each red segment is a **residual** $y_i - \\hat y_i$ (actual minus predicted). Squaring them${P.sq ? " (the red squares)" : ""} and averaging gives the MSE = ${r3(L)}. ${hint}`,
      }];
    },
  };

  /* ================================================================== 2. 2-D gradient descent: plain vs momentum on a narrow valley */
  V.ml1_gd2d = {
    title: "Gradient descent vs momentum on a long, narrow valley",
    params: [
      { id: "lr", label: "learning rate η", min: 0.01, max: 0.105, step: 0.005, value: 0.09 },
      { id: "beta", label: "momentum β", min: 0, max: 0.95, step: 0.05, value: 0.7 },
    ],
    gen(_, P) {
      // L(w1, w2) = ½(w1² + 20·w2²): gentle along w1, steep along w2 (like unscaled features)
      const A = 1, B = 20, L = (x, y) => 0.5 * (A * x * x + B * y * y), G = (x, y) => [A * x, B * y];
      const start = [-4.5, 1.6], N = 40, gd = [start.slice()], mo = [start.slice()];
      let v = [0, 0];
      for (let t = 0; t < N; t++) {
        const [x, y] = gd[t], g = G(x, y); gd.push([x - P.lr * g[0], y - P.lr * g[1]]);
        const [a, c] = mo[t], h = G(a, c); v = [P.beta * v[0] + h[0], P.beta * v[1] + h[1]]; mo.push([a - P.lr * v[0], c - P.lr * v[1]]);
      }
      const levels = [0.05, 0.3, 1, 2.5, 5, 9, 15, 24];
      const contours = levels.map((l) => { const pts = []; for (let i = 0; i <= 120; i++) { const t = (i / 120) * 2 * Math.PI; pts.push([Math.sqrt((2 * l) / A) * Math.cos(t), Math.sqrt((2 * l) / B) * Math.sin(t)]); } return Plot.line(pts, "pl-dash"); });
      const clip = (p) => p.map(([x, y]) => [Math.max(-9, Math.min(9, x)), Math.max(-9, Math.min(9, y))]);
      const S = [];
      for (let k = 0; k <= N; k++) {
        const g = clip(gd.slice(0, k + 1)), m = clip(mo.slice(0, k + 1)), lg = L(...gd[k]), lm = L(...mo[k]);
        const svg = Plot.chart({ x: [-5, 5], y: [-2.2, 2.2], W: 560, H: 290, xl: "w₁ (gentle direction)", yl: "w₂ (steep)" }, [
          ...contours, Plot.dot(0, 0, "pl-c3", 4),
          Plot.line(g, "pl-c2 pl-thin"), Plot.pts(g, "pl-c2", 3),
          Plot.line(m, "pl-c5 pl-thin"), Plot.pts(m, "pl-c5", 3),
          Plot.text(-4.9, 2.0, "plain GD", "", "start"), (S2) => `<circle cx="${S2.sx(-3.55)}" cy="${S2.sy(2.05)}" r="4" style="fill:var(--amber)"/>`,
          Plot.text(-2.9, 2.0, "momentum", "", "start"), (S2) => `<circle cx="${S2.sx(-1.4)}" cy="${S2.sy(2.05)}" r="4" style="fill:var(--purple)"/>`,
        ]);
        let msg;
        const div = !isFinite(lg) || lg > 1e3 || Math.abs(gd[k][1]) > 5;
        if (k === 0) msg = `Both start at (−4.5, 1.6). The loss $L = \\tfrac12(w_1^2 + 20w_2^2)$ is a narrow valley: **20× steeper** across (w₂) than along (w₁). Press Play or Next.`;
        else if (div) msg = `Plain GD is **diverging**: in the steep direction a step of η·20 = ${r3(P.lr * 20)} > 2 overshoots further every time. Lower η.`;
        else if (k < 6) msg = `Plain GD (amber) bounces across the valley because the steep direction dominates its gradient, while it barely moves along w₁. Momentum (purple) keeps a running sum of past gradients: $v \\leftarrow \\beta v + \\nabla L$, $w \\leftarrow w - \\eta v$.`;
        else if (k < 20) msg = `The zig-zags of momentum cancel out (they point opposite ways each step), while the small, consistent push along w₁ **adds up**, so momentum races along the valley floor.`;
        else msg = `After ${k} steps: plain GD loss ${r3(lg)}, momentum loss ${r3(lm)}. ${lm < lg ? "Momentum is ahead." : "Plain GD is ahead here: with this β the momentum path overshoots and circles."} Try β = 0 (momentum becomes plain GD) or β = 0.95 (too much: it swings around).`;
        S.push({ svg, side: { step: k, "GD loss": r3(lg), "momentum loss": r3(lm) }, msg });
      }
      return S;
    },
  };

  /* ================================================================== 3. Polynomial degree: under- and overfitting */
  const POLY = (() => {
    const r = rng(11), f = (x) => Math.sin(2 * Math.PI * x), tx = [], ty = [], vx = [], vy = [];
    for (let i = 0; i < 12; i++) { const x = (i + 0.2 + 0.6 * r()) / 12; tx.push(x); ty.push(f(x) + 0.25 * gauss(r)); }
    for (let i = 0; i < 40; i++) { const x = (i + r()) / 40; vx.push(x); vy.push(f(x) + 0.25 * gauss(r)); }
    return { f, tx, ty, vx, vy };
  })();
  V.ml1_poly = {
    title: "Polynomial degree: from underfitting to overfitting",
    params: [{ id: "d", label: "degree d", min: 0, max: 11, step: 1, value: 1 }],
    gen(_, P) {
      const { f, tx, ty, vx, vy } = POLY, d = P.d, h = polyFit(tx, ty, d);
      const tr = mse(tx.map(h), ty), te = mse(vx.map(h), vy);
      const verdict = d <= 1 ? "underfitting" : d <= 5 ? "about right" : d <= 8 ? "starting to overfit" : "overfitting";
      const svg = Plot.chart({ x: [0, 1], y: [-1.9, 1.9], xl: "x", yl: "y", H: 300 }, [
        Plot.fn(f, "pl-dash"), Plot.pts(vx.map((x, i) => [x, vy[i], "b"]), "", 3.2), Plot.fn(h, "pl-c1"), Plot.pts(tx.map((x, i) => [x, ty[i], "a"]), "", 5),
      ]);
      const msgs = {
        underfitting: `Degree ${d} is too simple: a ${d === 0 ? "flat line" : "straight line"} can't follow the wave. Training **and** test error are both high (**high bias**).`,
        "about right": `Degree ${d} follows the true curve (dashed) without chasing the noise. Test error is ${r3(te)}: close to the noise level (0.25² ≈ 0.06).`,
        "starting to overfit": `Degree ${d}: training error keeps falling (${r3(tr)}), but the curve starts to wiggle between the training points and test error rises.`,
        overfitting: `Degree ${d} with only 12 training points: the curve almost passes through every blue point (train MSE ${r3(tr)}) but swings wildly between them. It memorised the noise, so test MSE is ${r3(te)} (**high variance**).`,
      };
      return [{ svg, side: { degree: d, "train MSE": r3(tr), "test MSE": r3(te), verdict }, msg: msgs[verdict] + " Blue = 12 training points, orange = 40 test points." }];
    },
  };

  /* ================================================================== 4. Bias–variance: refit on many training sets */
  V.ml1_bvsim = {
    title: "Bias and variance: the same model refit on 25 different training sets",
    params: [{ id: "d", label: "degree d", min: 0, max: 9, step: 1, value: 1 }],
    gen(_, P) {
      const f = (x) => Math.sin(2 * Math.PI * x), d = P.d, G = 60, grid = Array.from({ length: G }, (_, i) => (i + 0.5) / G);
      const fits = [];
      for (let s = 0; s < 25; s++) {
        const r = rng(100 + s), xs = [], ys = [];
        for (let i = 0; i < 15; i++) { const x = r(); xs.push(x); ys.push(f(x) + 0.3 * gauss(r)); }
        fits.push(polyFit(xs, ys, d, 1e-9));
      }
      const preds = fits.map((h) => grid.map(h)), avg = grid.map((_, j) => mean(preds.map((p) => p[j])));
      const bias2 = mean(grid.map((x, j) => (avg[j] - f(x)) ** 2)), vr = mean(grid.map((_, j) => mean(preds.map((p) => (p[j] - avg[j]) ** 2))));
      const svg = Plot.chart({ x: [0, 1], y: [-2, 2], xl: "x", yl: "y", H: 300 }, [
        ...fits.map((h) => Plot.fn(h, "pl-c1 pl-thin\" style=\"opacity:.28")),
        Plot.fn(f, "pl-c3"), Plot.line(grid.map((x, j) => [x, avg[j]]), "pl-c2"),
      ]);
      const msg = d <= 1 ? `Degree ${d}: every fit (thin blue) looks almost the same, so **variance is low**. But their average (amber) is far from the truth (green): **high bias**. The model is too simple, whatever data it gets.` :
        d <= 4 ? `Degree ${d}: the average fit (amber) sits close to the truth (green) and the individual fits stay close together. Both bias² and variance are small: the sweet spot.` :
          `Degree ${d}: the average fit is still close to the truth (**low bias**), but each individual fit swings wildly depending on which 15 points it happened to get: **high variance**.`;
      return [{ svg, side: { "bias²": r3(bias2), variance: r3(vr), "bias² + variance": r3(bias2 + vr), "noise σ²": "0.09" }, msg: msg + " Each training set has 15 noisy points from the same curve." }];
    },
  };

  /* ================================================================== 5. Regularization: ridge / lasso on a degree-9 polynomial */
  const REG = (() => {
    const r = rng(5), f = (x) => Math.sin(2 * Math.PI * x), tx = [], ty = [], vx = [], vy = [];
    for (let i = 0; i < 14; i++) { const x = (i + 0.15 + 0.7 * r()) / 14; tx.push(x); ty.push(f(x) + 0.25 * gauss(r)); }
    for (let i = 0; i < 40; i++) { const x = (i + r()) / 40; vx.push(x); vy.push(f(x) + 0.25 * gauss(r)); }
    const D = 9, raw = (x) => Array.from({ length: D }, (_, k) => Math.pow(x, k + 1));
    const R = tx.map(raw), mu = Array.from({ length: D }, (_, j) => mean(R.map((q) => q[j])));
    const sd = Array.from({ length: D }, (_, j) => Math.sqrt(mean(R.map((q) => (q[j] - mu[j]) ** 2))));
    const feat = (x) => raw(x).map((v, j) => (v - mu[j]) / sd[j]);
    return { f, tx, ty, vx, vy, D, feat, X: tx.map(feat), ybar: mean(ty) };
  })();
  function lassoCD(X, yc, alpha, D) {
    const n = X.length, w = new Array(D).fill(0), res = yc.slice();
    for (let it = 0; it < 30000; it++) {
      let mx = 0;
      for (let j = 0; j < D; j++) {
        let rho = 0; for (let i = 0; i < n; i++) rho += X[i][j] * (res[i] + X[i][j] * w[j]); rho /= n;   // columns have mean 0, variance 1
        const nw = Math.sign(rho) * Math.max(Math.abs(rho) - alpha, 0), dw = nw - w[j];
        if (dw) { for (let i = 0; i < n; i++) res[i] -= X[i][j] * dw; w[j] = nw; mx = Math.max(mx, Math.abs(dw)); }
      }
      if (mx < 1e-9) break;
    }
    return w;
  }
  V.ml1_ridge = {
    title: "Regularization strength: shrinking the weights of a degree-9 polynomial",
    params: [
      { id: "la", label: "log₁₀ α", min: -7, max: 2, step: 0.25, value: -7, fmt: (v) => "α = " + (+v <= -7 ? "1e-7" : r3(Math.pow(10, +v))) },
      { id: "l1", label: "penalty", min: 0, max: 1, step: 1, value: 0, fmt: (v) => (+v ? "L1 (lasso)" : "L2 (ridge)") },
    ],
    gen(_, P) {
      const { f, tx, ty, vx, vy, D, feat, X, ybar } = REG, alpha = Math.pow(10, P.la), yc = ty.map((y) => y - ybar);
      const w = P.l1 ? lassoCD(X, yc, alpha, D) : lstsq(X, yc, alpha);
      const h = (x) => ybar + feat(x).reduce((s, v, j) => s + v * w[j], 0);
      const tr = mse(tx.map(h), ty), te = mse(vx.map(h), vy), nz = w.filter((v) => Math.abs(v) > 1e-8).length, norm = Math.sqrt(sum(w.map((v) => v * v)));
      const fit = Plot.chart({ x: [0, 1], y: [-1.9, 1.9], xl: "x", yl: "y", W: 380, H: 270 }, [
        Plot.fn(f, "pl-dash"), Plot.pts(vx.map((x, i) => [x, vy[i], "b"]), "", 3), Plot.fn(h, "pl-c1"), Plot.pts(tx.map((x, i) => [x, ty[i], "a"]), "", 4.5),
      ]);
      // weights on a signed log scale so both 3000 and 0.3 are visible: s(v) = sign(v)·log10(1 + |v|)
      const sl = (v) => Math.sign(v) * Math.log10(1 + Math.abs(v));
      const bars = Plot.chart({ x: [0.3, D + 0.7], y: [-4.2, 4.2], W: 250, H: 270, xl: "weight w₁ … w₉ (signed log scale)", grid: false, pad: { l: 40 } }, [
        (S) => [-4, -2, 0, 2, 4].map((t) => `<line class="pl-grid" x1="${S.P.l}" x2="${S.W - S.P.r}" y1="${S.sy(t)}" y2="${S.sy(t)}"/><text class="pl-tick" x="${S.P.l - 4}" y="${S.sy(t) + 4}" text-anchor="end">${t === 0 ? "0" : (t < 0 ? "−" : "") + "10" + (Math.abs(t) === 2 ? "²" : "⁴")}</text>`).join(""),
        ...w.map((v, j) => rect(j + 0.62, 0, j + 1.38, sl(v), Math.abs(v) < 1e-8 ? "fill:var(--muted)" : v > 0 ? "fill:var(--accent)" : "fill:var(--amber)")),
        ...w.map((v, j) => (Math.abs(v) < 1e-8 ? tag(j + 1, 0.25, "0", "font-weight:700") : "")),
      ]);
      const pen = P.l1 ? "L1" : "L2";
      const msg = alpha < 1e-4 ? `Almost no penalty: the 9 weights are huge (up to ${r3(Math.max(...w.map(Math.abs)))}) and cancel each other out, so the curve wiggles through the training points. Train MSE ${r3(tr)}, test MSE ${r3(te)}: overfitting.` :
        te < 0.12 ? `A moderate ${pen} penalty keeps the weights small, so the curve is smooth and close to the truth. Test MSE ${r3(te)}.${P.l1 ? ` Lasso has set ${D - nz} of the 9 weights to **exactly zero**.` : ""}` :
          norm < 0.05 ? `The penalty is so strong that every weight is ~0: the model predicts the mean of y (a flat line). Underfitting.` :
            `As α grows the weights shrink and the curve flattens. ${P.l1 ? `${D - nz} weights are exactly zero (lasso does that, ridge never does).` : "Ridge shrinks every weight but never makes one exactly zero."} Test MSE ${r3(te)}.`;
      return [{ svg: row(fit, bars), side: { "‖w‖": r3(norm), "non-zero weights": nz + " / " + D, "train MSE": r3(tr), "test MSE": r3(te) }, msg }];
    },
  };

  /* ================================================================== 6. Logistic regression: boundary + sigmoid */
  const LOG = (() => {
    const r = rng(21), pts = [];
    for (let i = 0; i < 18; i++) pts.push([-1.2 + gauss(r) * 0.95, -0.6 + gauss(r) * 0.95, 0]);
    for (let i = 0; i < 18; i++) pts.push([1.1 + gauss(r) * 0.95, 0.9 + gauss(r) * 0.95, 1]);
    // the best fit (minimum mean log loss, no penalty) by Newton's method
    let w = [0, 0, 0];
    for (let it = 0; it < 50; it++) {
      const g = [0, 0, 0], H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      pts.forEach(([a, b, y]) => { const x = [a, b, 1], p = sig(w[0] * a + w[1] * b + w[2]); for (let i = 0; i < 3; i++) { g[i] += (p - y) * x[i]; for (let j = 0; j < 3; j++) H[i][j] += p * (1 - p) * x[i] * x[j] + (i === j ? 1e-6 : 0); } });
      const step = lstsq(H, g); w = w.map((v, i) => v - step[i]);
    }
    return { pts, best: w };
  })();
  const bce = (pts, w) => mean(pts.map(([a, b, y]) => { const p = Math.min(1 - 1e-12, Math.max(1e-12, sig(w[0] * a + w[1] * b + w[2]))); return -(y * Math.log(p) + (1 - y) * Math.log(1 - p)); }));
  V.ml1_logreg = {
    title: "Logistic regression: a line, a sigmoid and the log loss",
    params: [
      { id: "w1", label: "w₁", min: -4, max: 4, step: 0.1, value: 0.5 },
      { id: "w2", label: "w₂", min: -4, max: 4, step: 0.1, value: 1.5 },
      { id: "b", label: "bias b", min: -4, max: 4, step: 0.1, value: 0 },
    ],
    gen(_, P) {
      const { pts, best } = LOG, w = [P.w1, P.w2, P.b], z = (a, b) => w[0] * a + w[1] * b + w[2];
      const L = bce(pts, w), acc = mean(pts.map(([a, b, y]) => +((z(a, b) >= 0 ? 1 : 0) === y)));
      const shade = (S) => { let s = ""; const n = 36, dx = (S.x1 - S.x0) / n, dy = (S.y1 - S.y0) / n;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const p = sig(z(S.x0 + (i + 0.5) * dx, S.y0 + (j + 0.5) * dy)), o = Math.abs(p - 0.5) * 0.6;
          s += `<rect x="${S.sx(S.x0 + i * dx).toFixed(1)}" y="${S.sy(S.y0 + (j + 1) * dy).toFixed(1)}" width="${(S.sx(dx) - S.sx(0) + 0.6).toFixed(1)}" height="${(S.sy(0) - S.sy(dy) + 0.6).toFixed(1)}" style="fill:var(${p >= 0.5 ? "--cls-b" : "--cls-a"});opacity:${o.toFixed(3)}"/>`; }
        return s; };
      // boundary w1 x + w2 y + b = 0
      const bnd = Math.abs(w[1]) > 1e-9 ? Plot.fn((x) => -(w[0] * x + w[2]) / w[1], "pl-c5") : Math.abs(w[0]) > 1e-9 ? Plot.vline(-w[2] / w[0], "pl-c5") : "";
      const plane = Plot.chart({ x: [-4, 4], y: [-4, 4], W: 320, H: 300, xl: "x₁", yl: "x₂" }, [shade, bnd, Plot.pts(pts.map(([a, b, y]) => [a, b, y ? "b" : "a"]), "", 4)]);
      const zs = pts.map(([a, b]) => z(a, b)), zr = Math.max(6, Math.min(16, Math.ceil(Math.max(...zs.map(Math.abs)))));
      const curve = Plot.chart({ x: [-zr, zr], y: [-0.08, 1.08], W: 320, H: 300, xl: "z = w₁x₁ + w₂x₂ + b", yl: "p = σ(z)" }, [
        Plot.hline(0.5), Plot.vline(0), Plot.fn(sig, "pl-c5"),
        Plot.pts(pts.map(([a, b, y], i) => [Math.max(-zr, Math.min(zr, zs[i])), y, y ? "b" : "a"]), "", 3.8),
        Plot.pts(pts.map((_, i) => [Math.max(-zr, Math.min(zr, zs[i])), sig(zs[i])]), "pl-c5", 2.2),
      ]);
      const gap = L - bce(pts, best);
      return [{
        svg: row(plane, curve),
        side: { "log loss": r3(L), accuracy: pct(acc), "best possible loss": r3(bce(pts, best)) },
        msg: `Left: the purple line is where $z = 0$, i.e. $p = 0.5$; the shading gets stronger as the model gets more sure. Right: every point's score $z$ goes through the sigmoid. ` +
          (gap < 0.01 ? `This is the **best fit** (minimum log loss): w₁ ≈ ${r3(best[0])}, w₂ ≈ ${r3(best[1])}, b ≈ ${r3(best[2])}.` :
            `Gradient descent would move you to w₁ ≈ ${r3(best[0])}, w₂ ≈ ${r3(best[1])}, b ≈ ${r3(best[2])}. Try doubling all three: accuracy stays the same but the loss changes, because the loss cares about **confidence**, not just the side of the line.`),
      }];
    },
  };

  /* ================================================================== 7. k-NN: neighbours vote */
  const KNN = (() => {
    const r = rng(33), pts = [];
    for (let i = 0; i < 16; i++) pts.push([+(3.2 + gauss(r) * 1.5).toFixed(2), +(3.6 + gauss(r) * 1.5).toFixed(2), 0]);
    for (let i = 0; i < 16; i++) pts.push([+(6.6 + gauss(r) * 1.5).toFixed(2), +(6.2 + gauss(r) * 1.5).toFixed(2), 1]);
    return pts.map(([a, b, c]) => [Math.max(0.3, Math.min(9.7, a)), Math.max(0.3, Math.min(9.7, b)), c]);
  })();
  const knnVote = (qx, qy, k) => {
    const d = KNN.map(([a, b, c], i) => ({ i, c, d: Math.hypot(a - qx, b - qy) })).sort((p, q) => p.d - q.d).slice(0, k);
    const votes = [0, 0]; d.forEach((n) => votes[n.c]++);
    const cls = votes[0] !== votes[1] ? (votes[1] > votes[0] ? 1 : 0) : d[0].c;   // tie → the single nearest neighbour decides
    return { nb: d, votes, cls, tie: votes[0] === votes[1] };
  };
  V.ml1_knn = {
    title: "k-nearest neighbours: who are the query point's neighbours?",
    params: [
      { id: "k", label: "k", min: 1, max: 21, step: 1, value: 5 },
      { id: "qx", label: "query x₁", min: 0.5, max: 9.5, step: 0.1, value: 5.2 },
      { id: "qy", label: "query x₂", min: 0.5, max: 9.5, step: 0.1, value: 4.6 },
    ],
    gen(_, P) {
      const { nb, votes, cls, tie } = knnVote(P.qx, P.qy, P.k), R = nb[nb.length - 1].d;
      const circle = (S) => `<circle cx="${S.sx(P.qx).toFixed(1)}" cy="${S.sy(P.qy).toFixed(1)}" r="${(S.sx(R) - S.sx(0)).toFixed(1)}" style="fill:color-mix(in srgb, var(--amber) 8%, transparent);stroke:var(--amber);stroke-width:1.4;stroke-dasharray:4 3"/>`;
      const svg = Plot.chart({ x: [0, 10], y: [0, 10], W: 420, H: 400, xl: "x₁", yl: "x₂" }, [
        Plot.regions((x, y) => (knnVote(x, y, P.k).cls ? "b" : "a"), 50), circle,
        ...nb.map((n) => Plot.seg(P.qx, P.qy, KNN[n.i][0], KNN[n.i][1], "pl-c2 pl-thin")),
        Plot.pts(KNN.map(([a, b, c]) => [a, b, c ? "b" : "a"]), "", 4.5),
        ...nb.map((n) => ring(KNN[n.i][0], KNN[n.i][1], 7.5, "stroke:var(--text);stroke-width:1.6")),
        (S) => `<rect x="${(S.sx(P.qx) - 6).toFixed(1)}" y="${(S.sy(P.qy) - 6).toFixed(1)}" width="12" height="12" transform="rotate(45 ${S.sx(P.qx).toFixed(1)} ${S.sy(P.qy).toFixed(1)})" style="fill:var(--amber);stroke:var(--panel);stroke-width:1.5"/>`,
      ]);
      const name = (c) => (c ? "orange" : "blue");
      return [{
        svg, side: { k: P.k, "blue votes": votes[0], "orange votes": votes[1], prediction: name(cls) },
        msg: `The diamond is a new point. Its ${P.k} nearest training points (circled) vote: ${votes[0]} blue, ${votes[1]} orange → **${name(cls)}**.` +
          (tie ? " It's a **tie**, so the single nearest neighbour breaks it (an odd k avoids ties with two classes)." : "") +
          (P.k <= 2 ? " With k this small the shaded regions are jagged islands around single points: low bias, high variance." : P.k >= 15 ? " With a big k the boundary is very smooth, and the majority class starts to win everywhere: high bias." : " The shading shows what k-NN would predict at every location."),
      }];
    },
  };

  /* ================================================================== 8. Gaussian naive Bayes in 1-D */
  const GNB = (() => {
    const r = rng(41), A = [], B = [];
    for (let i = 0; i < 20; i++) A.push(2.4 + gauss(r) * 1.0);
    for (let i = 0; i < 20; i++) B.push(5.4 + gauss(r) * 1.4);
    const st = (a) => { const m = mean(a); return [m, Math.sqrt(mean(a.map((v) => (v - m) ** 2)))]; };
    return { A, B, a: st(A), b: st(B) };
  })();
  const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  V.ml1_gnb = {
    title: "Naive Bayes in one dimension: likelihood × prior → posterior",
    params: [
      { id: "pb", label: "prior P(orange)", min: 0.05, max: 0.95, step: 0.05, value: 0.5 },
      { id: "x", label: "new x", min: -1, max: 10, step: 0.1, value: 4.0 },
    ],
    gen(_, P) {
      const { A, B, a, b } = GNB, pa = 1 - P.pb, pb = P.pb;
      const ja = (x) => npdf(x, a[0], a[1]) * pa, jb = (x) => npdf(x, b[0], b[1]) * pb, post = (x) => jb(x) / (ja(x) + jb(x));
      const top = Math.max(0.05, ...Array.from({ length: 100 }, (_, i) => Math.max(ja(-1 + i * 0.11), jb(-1 + i * 0.11)))) * 1.15;
      const c1 = Plot.chart({ x: [-1, 10], y: [0, top], W: 560, H: 200, xl: "", yl: "P(x | class)·P(class)", pad: { b: 22 } }, [
        Plot.fn(ja, "pl-c1"), Plot.fn(jb, "pl-c2"), Plot.vline(P.x),
        Plot.pts(A.map((v) => [v, top * 0.03, "a"]), "", 3), Plot.pts(B.map((v) => [v, top * 0.08, "b"]), "", 3),
        Plot.dot(P.x, ja(P.x), "pl-c1", 5), Plot.dot(P.x, jb(P.x), "pl-c2", 5),
      ]);
      // decision boundaries: where the posterior crosses 0.5
      const cross = []; for (let i = 0; i < 1100; i++) { const x0 = -1 + i * 0.01, x1 = x0 + 0.01; if ((post(x0) - 0.5) * (post(x1) - 0.5) < 0) cross.push((x0 + x1) / 2); }
      const c2 = Plot.chart({ x: [-1, 10], y: [0, 1], W: 560, H: 170, xl: "x", yl: "P(orange | x)" }, [
        Plot.hline(0.5), ...cross.map((c) => Plot.vline(c, "pl-c5 pl-thin")), Plot.fn(post, "pl-c3"), Plot.vline(P.x), Plot.dot(P.x, post(P.x), "pl-c3", 5),
      ]);
      const pp = post(P.x);
      return [{
        svg: col(c1, c2),
        side: { "P(x|blue)P(blue)": r3(ja(P.x)), "P(x|orange)P(orange)": r3(jb(P.x)), "P(orange | x)": r3(pp), prediction: pp >= 0.5 ? "orange" : "blue" },
        msg: `Training fitted one bell curve per class: blue mean ${r3(a[0])}, sd ${r3(a[1])}; orange mean ${r3(b[0])}, sd ${r3(b[1])} (the dots at the bottom are the training data). ` +
          `At x = ${r3(P.x)}, Bayes' rule divides the orange height by the sum of both heights: **P(orange | x) = ${r3(pp)}**. ` +
          `Move the prior: a rarer class needs stronger evidence, so the purple decision boundary${cross.length > 1 ? "s shift" : " shifts"}. With several features, naive Bayes simply multiplies one such curve per feature.`,
      }];
    },
  };

  /* ================================================================== 9. SVM: maximum margin, support vectors and C (solved with SMO) */
  const SVMD = (() => {
    const r = rng(52), pts = [];
    for (let i = 0; i < 11; i++) pts.push([2.6 + gauss(r) * 0.85, 3.0 + gauss(r) * 0.85, -1]);
    for (let i = 0; i < 11; i++) pts.push([6.4 + gauss(r) * 0.85, 6.2 + gauss(r) * 0.85, 1]);
    return pts;
  })();
  function smo(X, y, C, tol = 1e-4) {
    const n = X.length, K = X.map((a) => X.map((b) => a[0] * b[0] + a[1] * b[1])), al = new Array(n).fill(0);
    let b = 0;
    const f = (i) => { let s = b; for (let j = 0; j < n; j++) if (al[j]) s += al[j] * y[j] * K[j][i]; return s; };
    const take = (i, j, Ei) => {
      if (i === j) return false;
      const Ej = f(j) - y[j], ai = al[i], aj = al[j];
      let L, H;
      if (y[i] !== y[j]) { L = Math.max(0, aj - ai); H = Math.min(C, C + aj - ai); } else { L = Math.max(0, ai + aj - C); H = Math.min(C, ai + aj); }
      if (H - L < 1e-12) return false;
      const eta = 2 * K[i][j] - K[i][i] - K[j][j];
      if (eta >= 0) return false;
      let nj = aj - (y[j] * (Ei - Ej)) / eta; nj = Math.min(H, Math.max(L, nj));
      if (Math.abs(nj - aj) < 1e-9 * (nj + aj + 1e-9)) return false;
      const ni = ai + y[i] * y[j] * (aj - nj);
      const b1 = b - Ei - y[i] * (ni - ai) * K[i][i] - y[j] * (nj - aj) * K[i][j];
      const b2 = b - Ej - y[i] * (ni - ai) * K[i][j] - y[j] * (nj - aj) * K[j][j];
      al[i] = ni; al[j] = nj;
      b = ni > 0 && ni < C ? b1 : nj > 0 && nj < C ? b2 : (b1 + b2) / 2;
      return true;
    };
    for (let pass = 0; pass < 3000; pass++) {
      let changed = 0;
      for (let i = 0; i < n; i++) {
        const Ei = f(i) - y[i];
        if ((y[i] * Ei < -tol && al[i] < C) || (y[i] * Ei > tol && al[i] > 0)) {
          let jb = -1, best = -1;
          for (let j = 0; j < n; j++) if (j !== i) { const d = Math.abs(Ei - (f(j) - y[j])); if (d > best) { best = d; jb = j; } }
          if (take(i, jb, Ei)) { changed++; continue; }
          for (let j = 0; j < n; j++) if (j !== jb && take(i, j, Ei)) { changed++; break; }
        }
      }
      if (!changed) break;
    }
    const w = [0, 0]; for (let i = 0; i < n; i++) { w[0] += al[i] * y[i] * X[i][0]; w[1] += al[i] * y[i] * X[i][1]; }
    const free = al.map((a, i) => i).filter((i) => al[i] > 1e-7 && al[i] < C - 1e-7);
    if (free.length) b = mean(free.map((i) => y[i] - (w[0] * X[i][0] + w[1] * X[i][1])));
    return { w, b, al };
  }
  V.ml1_svm = {
    title: "Support vector machine: the widest street between two classes",
    params: [
      { id: "lc", label: "log₁₀ C", min: -2, max: 2, step: 0.25, value: 2, fmt: (v) => "C = " + r3(Math.pow(10, +v)) },
      { id: "out", label: "outlier", min: 0, max: 1, step: 1, value: 0, fmt: (v) => (+v ? "on" : "off") },
    ],
    gen(_, P) {
      const pts = SVMD.concat(P.out ? [[3.4, 3.9, 1]] : []), X = pts.map((p) => [p[0], p[1]]), y = pts.map((p) => p[2]), C = Math.pow(10, P.lc);
      const { w, b, al } = smo(X, y, C), nw = Math.hypot(w[0], w[1]) || 1e-9, margin = 2 / nw;
      const fx = (x, k) => (k - b - w[0] * x) / w[1];
      const sv = al.map((a, i) => i).filter((i) => al[i] > 1e-6);
      const errs = pts.filter((p) => p[2] * (w[0] * p[0] + w[1] * p[1] + b) < 0).length, inside = pts.filter((p) => p[2] * (w[0] * p[0] + w[1] * p[1] + b) < 1 - 1e-6).length;
      const svg = Plot.chart({ x: [0, 9.5], y: [0, 9.5], W: 420, H: 400, xl: "x₁", yl: "x₂" }, [
        Plot.regions((a, c) => (w[0] * a + w[1] * c + b >= 0 ? "b" : "a"), 40),
        Plot.fn((x) => fx(x, 1), "pl-dash"), Plot.fn((x) => fx(x, -1), "pl-dash"), Plot.fn((x) => fx(x, 0), "pl-c5"),
        Plot.pts(pts.map(([a, c, l]) => [a, c, l > 0 ? "b" : "a"]), "", 4.5),
        ...sv.map((i) => ring(X[i][0], X[i][1], 8.5, "stroke:var(--text);stroke-width:1.8")),
      ]);
      const hard = C >= 30 && !P.out;
      return [{
        svg, side: { C: r3(C), "margin width 2/‖w‖": r3(margin), "support vectors": sv.length, "inside margin": inside, "training errors": errs },
        msg: `The purple line is $w \\cdot x + b = 0$; the dashed lines are $w \\cdot x + b = \\pm 1$, the edges of the "street", whose width is $2/\\lVert w \\rVert$ = ${r3(margin)}. Circled points are the **support vectors** (α > 0): ` +
          (hard ? "with a large C nothing may sit inside the street, so only the few points touching its edges matter. Move any other point and the line would not change." :
            P.out && C >= 10 ? "the outlier forces a big C to bend the line (or misclassify it at a high cost). Lower C to let the model ignore it and keep a wide street." :
              `a smaller C makes margin violations cheap, so the street gets wider and more points end up inside it (and become support vectors). ${inside} point${inside === 1 ? "" : "s"} sit inside the street now.`),
      }];
    },
  };

  /* ================================================================== 10. Decision tree: grow axis-aligned splits */
  const TREE = (() => {
    const r = rng(64), pts = [];
    for (let i = 0; i < 46; i++) {
      const a = +(0.4 + r() * 9.2).toFixed(1), b = +(0.4 + r() * 9.2).toFixed(1);
      let c = (a > 5.5 && b > 4) || (a < 3 && b < 3.2) ? 1 : 0;
      if (i === 7 || i === 30) c = 1 - c;   // two noisy labels: deep trees will chase them
      pts.push([a, b, c]);
    }
    return pts;
  })();
  const gini = (n0, n1) => { const n = n0 + n1; return n ? 1 - (n0 / n) ** 2 - (n1 / n) ** 2 : 0; };
  function bestSplit(ids) {
    const n = ids.length, c1 = ids.filter((i) => TREE[i][2]).length;
    let best = null;
    for (const f of [0, 1]) {
      const s = ids.slice().sort((p, q) => TREE[p][f] - TREE[q][f]);
      let l1 = 0;
      for (let k = 0; k < n - 1; k++) {
        l1 += TREE[s[k]][2];
        if (TREE[s[k]][f] === TREE[s[k + 1]][f]) continue;
        const nl = k + 1, nr = n - nl, g = (nl * gini(nl - l1, l1) + nr * gini(nr - (c1 - l1), c1 - l1)) / n;
        if (!best || g < best.g - 1e-12) best = { f, t: +((TREE[s[k]][f] + TREE[s[k + 1]][f]) / 2).toFixed(2), g };
      }
    }
    return best;
  }
  V.ml1_tree = {
    title: "Decision tree: grow one split at a time (Gini impurity)",
    params: [{ id: "md", label: "max depth", min: 1, max: 6, step: 1, value: 3 }],
    gen(_, P) {
      const all = TREE.map((_, i) => i), cnt = (ids) => { const c1 = ids.filter((i) => TREE[i][2]).length; return [ids.length - c1, c1]; };
      const root = { ids: all, box: [0, 10, 0, 10], depth: 0 };
      let leaves = [root];
      const splits = [], S = [];
      const acc = () => { let ok = 0; leaves.forEach((L) => { const [a, b] = cnt(L.ids); ok += Math.max(a, b); }); return ok / TREE.length; };
      const draw = (cur, msg, side) => {
        const layers = leaves.map((L) => { const [a, b] = cnt(L.ids), [x0, x1, y0, y1] = L.box; return rect(x0, y0, x1, y1, `fill:var(${b > a ? "--cls-b" : "--cls-a"});opacity:${(0.06 + 0.16 * Math.abs(a - b) / Math.max(1, a + b)).toFixed(3)}`); });
        splits.forEach((s) => layers.push(s.f === 0 ? Plot.seg(s.t, s.box[2], s.t, s.box[3], s === cur ? "pl-c2" : "pl-c5 pl-thin") : Plot.seg(s.box[0], s.t, s.box[1], s.t, s === cur ? "pl-c2" : "pl-c5 pl-thin")));
        if (cur) { const [x0, x1, y0, y1] = cur.box; layers.push(rect(x0, y0, x1, y1, "fill:none;stroke:var(--amber);stroke-width:2;stroke-dasharray:5 3")); }
        layers.push(Plot.pts(TREE.map(([a, b, c]) => [a, b, c ? "b" : "a"]), "", 4.2));
        S.push({ svg: Plot.chart({ x: [0, 10], y: [0, 10], W: 420, H: 400, xl: "x₁", yl: "x₂" }, layers), side: Object.assign({ leaves: leaves.length, "train accuracy": pct(acc()) }, side), msg });
      };
      const [r0, r1] = cnt(all);
      draw(null, `The root holds all ${TREE.length} points: ${r0} blue, ${r1} orange. Gini impurity $1 - p_{blue}^2 - p_{orange}^2$ = ${r3(gini(r0, r1))}. The tree will try **every** threshold on x₁ and on x₂ and keep the one whose children are purest.`, { depth: 0, "node Gini": r3(gini(r0, r1)) });
      const queue = [root];
      while (queue.length) {
        const nd = queue.shift(), [a, b] = cnt(nd.ids);
        if (nd.depth >= P.md || !a || !b) continue;
        const sp = bestSplit(nd.ids);
        if (!sp) continue;
        const left = nd.ids.filter((i) => TREE[i][sp.f] <= sp.t), right = nd.ids.filter((i) => TREE[i][sp.f] > sp.t);
        const [x0, x1, y0, y1] = nd.box;
        const L = { ids: left, depth: nd.depth + 1, box: sp.f === 0 ? [x0, sp.t, y0, y1] : [x0, x1, y0, sp.t] };
        const R = { ids: right, depth: nd.depth + 1, box: sp.f === 0 ? [sp.t, x1, y0, y1] : [x0, x1, sp.t, y1] };
        const s = { f: sp.f, t: sp.t, box: nd.box };
        splits.push(s); leaves = leaves.flatMap((q) => (q === nd ? [L, R] : [q])); queue.push(L, R);
        const g0 = gini(a, b), [la, lb] = cnt(left), [ra, rb] = cnt(right);
        draw(s, `Depth ${nd.depth + 1}: split the dashed box (${a} blue, ${b} orange, Gini ${r3(g0)}) on **${sp.f ? "x₂" : "x₁"} ≤ ${sp.t}**. Left/below: ${la} blue + ${lb} orange; right/above: ${ra} blue + ${rb} orange. Weighted Gini ${r3(g0)} → ${r3(sp.g)}: a gain of **${r3(g0 - sp.g)}**.`, { depth: nd.depth + 1, "node Gini": r3(g0), "gain": r3(g0 - sp.g) });
      }
      const pure = leaves.every((L) => { const [a, b] = cnt(L.ids); return !a || !b; });
      draw(null, `Done (max depth ${P.md}): ${leaves.length} leaves, training accuracy ${pct(acc())}. ` + (pure ? "Every leaf is pure, including tiny boxes around the two noisy points: that is **overfitting**. Limit the depth (or min samples per leaf) to stop it." :
        `Each leaf predicts its majority class (the shading). ${P.md <= 2 ? "Increase max depth to carve out more boxes." : "Deeper trees fit the training data more and more exactly, noise included."}`), { depth: Math.max(...leaves.map((L) => L.depth)) });
      return S;
    },
  };

  /* ================================================================== shared 1-D regression data for bagging and boosting */
  const ONED = (() => {
    const r = rng(77), f = (x) => 1.8 * Math.sin(x) + 0.25 * x, xs = [], ys = [], vx = [], vy = [];
    for (let i = 0; i < 40; i++) { const x = r() * 10; xs.push(x); ys.push(f(x) + 0.55 * gauss(r)); }
    for (let i = 0; i < 300; i++) { const x = r() * 10; vx.push(x); vy.push(f(x) + 0.55 * gauss(r)); }
    return { f, xs, ys, vx, vy };
  })();

  /* ================================================================== 11. Bagging: average many bootstrapped trees */
  V.ml1_bagging = {
    title: "Bagging: average many trees, each trained on a bootstrap sample",
    params: [
      { id: "B", label: "number of trees", min: 1, max: 60, step: 1, value: 1 },
      { id: "md", label: "tree depth", min: 1, max: 8, step: 1, value: 8 },
    ],
    gen(_, P) {
      const { f, xs, ys, vx, vy } = ONED, n = xs.length, trees = [];
      for (let t = 0; t < P.B; t++) { const r = rng(1000 + t), idx = Array.from({ length: n }, () => Math.floor(r() * n)); trees.push(regTree(xs, ys, idx, P.md)); }
      const avg = (x) => mean(trees.map((h) => h(x)));
      const single = mse(vx.map(trees[0]), vy), bag = mse(vx.map(avg), vy);
      const svg = Plot.chart({ x: [0, 10], y: [-3.5, 5], H: 300, xl: "x", yl: "y" }, [
        ...trees.slice(0, 25).map((h) => Plot.line(stepLine(h, 0, 10), "pl-c1 pl-thin\" style=\"opacity:" + (P.B === 1 ? 0 : 0.22))),
        Plot.fn(f, "pl-dash"), Plot.pts(xs.map((x, i) => [x, ys[i]]), "pl-c1", 3.2), Plot.line(stepLine(avg, 0, 10, 600), "pl-c2"),
      ]);
      return [{
        svg, side: { trees: P.B, "test MSE, one tree": r3(single), [`test MSE, average of ${P.B}`]: r3(bag) },
        msg: P.B === 1 ? `One ${P.md >= 6 ? "deep" : ""} tree trained on a bootstrap sample (n draws **with replacement**): its prediction (amber) is a jumpy staircase that chases the noise. Add trees.` :
          `${P.B} trees (thin blue, up to 25 drawn), each grown on a different bootstrap sample, so each one overfits **differently**. Their average (amber) is much smoother: test MSE ${r3(single)} → ${r3(bag)}. ${P.md <= 2 ? "Shallow trees are biased, and averaging can't fix bias: raise the depth." : "Averaging cancels variance but not bias, which is why bagging uses deep, low-bias trees."}`,
      }];
    },
  };

  /* ================================================================== 12. Gradient boosting: fit the residuals, one small tree at a time */
  V.ml1_boost = {
    title: "Gradient boosting: each new tree fits what is still wrong",
    params: [
      { id: "lr", label: "learning rate η", min: 0.05, max: 1, step: 0.05, value: 0.3 },
      { id: "md", label: "tree depth", min: 1, max: 3, step: 1, value: 1 },
    ],
    gen(_, P) {
      const { f, xs, ys, vx, vy } = ONED, n = xs.length, M = 40, all = Array.from({ length: n }, (_, i) => i);
      const F0 = mean(ys), trees = [];
      const F = (x, m) => { let s = F0; for (let k = 0; k < m; k++) s += P.lr * trees[k](x); return s; };
      const S = [];
      const frame = (m, h, res, msg) => {
        const top = Plot.chart({ x: [0, 10], y: [-3.5, 5], W: 560, H: 220, xl: "", yl: "y and model F", pad: { b: 22 } }, [
          Plot.fn(f, "pl-dash"), Plot.pts(xs.map((x, i) => [x, ys[i]]), "pl-c1", 3.2),
          m > 0 ? Plot.line(stepLine((x) => F(x, m - 1), 0, 10), "pl-c2 pl-thin\" style=\"opacity:.45") : "", Plot.line(stepLine((x) => F(x, m), 0, 10), "pl-c2"),
        ]);
        const bot = res ? Plot.chart({ x: [0, 10], y: [-3.5, 3.5], W: 560, H: 170, xl: "x", yl: "residual y − F" }, [
          Plot.hline(0), ...xs.map((x, i) => Plot.seg(x, 0, x, res[i], "pl-c4 pl-thin")), Plot.pts(xs.map((x, i) => [x, res[i]]), "pl-c4", 3), h ? Plot.line(stepLine(h, 0, 10), "pl-c5") : "",
        ]) : "";
        S.push({ svg: col(top, bot), side: { round: m, "train MSE": r3(mse(xs.map((x) => F(x, m)), ys)), "test MSE": r3(mse(vx.map((x) => F(x, m)), vy)) }, msg });
      };
      frame(0, null, ys.map((y) => y - F0), `Round 0: the model is just the average of y (${r3(F0)}), a flat line. The red bars below are the **residuals** $y - F(x)$: what is still unexplained.`);
      for (let m = 1; m <= M; m++) {
        const res = xs.map((x, i) => ys[i] - F(x, m - 1)), h = regTree(xs, res, all, P.md);
        trees.push(h);
        const res2 = xs.map((x, i) => ys[i] - F(x, m));
        const msg = m === 1 ? `Round 1: fit a tiny tree (depth ${P.md}, purple) to the residuals, then add η = ${P.lr} times it to the model: $F_1 = F_0 + \\eta\\,h_1$. The amber line moves towards the data.` :
          m < 6 ? `Round ${m}: compute the new residuals, fit another small tree to them (purple), add η·h. Each tree only has to fix what the previous ones got wrong.` :
            m < M ? `Round ${m}: the residuals are shrinking, so each new tree makes a smaller correction.${P.lr >= 0.8 ? " With a large η the model fits fast, and soon starts fitting the noise." : ""}` :
              `After ${M} rounds of tiny trees the model follows the curve. Train MSE keeps falling with more rounds; test MSE eventually stops improving: that is when **early stopping** would halt training.`;
        frame(m, h, res2, msg);
      }
      return S;
    },
  };

  /* ================================================================== static diagrams */
  const box = (x, y, w, h, label, cls = "node", sub) => `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="9"/>` +
    `<text class="val" x="${x + w / 2}" y="${y + h / 2 + (sub ? -2 : 5)}" text-anchor="middle" style="font-size:13px">${esc(label)}</text>` +
    (sub ? `<text class="lbl" x="${x + w / 2}" y="${y + h / 2 + 14}" text-anchor="middle" style="font-size:11px">${esc(sub)}</text>` : "");
  const arrow = (x1, y1, x2, y2, hl) => `<line class="edge${hl ? " hl" : ""}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#${hl ? "ml1ahA" : "ml1ah"})"/>`;
  const svgWrap = (W, H, inner) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img"><defs>` +
    `<marker id="ml1ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker>` +
    `<marker id="ml1ahA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--accent)"/></marker></defs>${inner}</svg>`;
  const txt = (x, y, s, style = "", anchor = "middle", cls = "lbl") => `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}" style="${style}">${esc(s)}</text>`;

  /* traditional programming vs machine learning */
  D.ml1_paradigm = () => {
    let s = txt(10, 22, "Traditional programming", "font-weight:700;fill:var(--text)", "start");
    s += box(10, 34, 110, 40, "Rules") + txt(135, 59, "+", "font-size:16px") + box(150, 34, 110, 40, "Data") + arrow(266, 54, 300, 54) + box(304, 34, 120, 40, "Program", "node hl") + arrow(430, 54, 464, 54) + box(468, 34, 130, 40, "Answers", "node ok");
    s += txt(10, 112, "Machine learning", "font-weight:700;fill:var(--text)", "start");
    s += box(10, 124, 110, 40, "Data") + txt(135, 149, "+", "font-size:16px") + box(150, 124, 110, 40, "Answers", "node ok") + arrow(266, 144, 300, 144, true) + box(304, 124, 120, 40, "Learning", "node hl") + arrow(430, 144, 464, 144, true) + box(468, 124, 130, 40, "Model", "node", "(the rules)");
    s += txt(10, 202, "Then:", "font-weight:700;fill:var(--text)", "start") + box(60, 186, 110, 34, "New data") + arrow(176, 203, 214, 203) + box(218, 186, 100, 34, "Model") + arrow(324, 203, 362, 203) + box(366, 186, 130, 34, "Predictions", "node ok");
    return svgWrap(610, 232, s);
  };

  /* the ML workflow */
  D.ml1_workflow = () => {
    const w = 112, h = 50, y1 = 20, y2 = 130;
    const top = [["1 Define", "the problem"], ["2 Collect", "the data"], ["3 Explore", "and clean"], ["4 Split", "train/val/test"], ["5 Features", "and scaling"]];
    const bot = [["6 Baseline", "simplest model"], ["7 Train", "and tune"], ["8 Test", "once, at the end"], ["9 Deploy", "and monitor"]];
    let s = "";
    top.forEach(([a, b], i) => { const x = 8 + i * 126; s += box(x, y1, w, h, a, i === 3 ? "node hl" : "node", b); if (i < 4) s += arrow(x + w + 2, y1 + h / 2, x + 124, y1 + h / 2); });
    s += `<path class="edge" d="M${8 + 4 * 126 + w / 2},${y1 + h + 2} L${8 + 4 * 126 + w / 2},${y1 + h + 22} L${71 + 3 * 126 + w / 2},${y1 + h + 22} L${71 + 3 * 126 + w / 2},${y2 - 3}" marker-end="url(#ml1ah)"/>`;
    bot.forEach(([a, b], i) => { const x = 71 + (3 - i) * 126; s += box(x, y2, w, h, a, i === 2 ? "node ok" : i === 1 ? "node hl" : "node", b); if (i < 3) s += arrow(x - 2, y2 + h / 2, x - 12, y2 + h / 2); });
    // loops: tune ↔ features, monitor → collect
    const tx = 71 + 2 * 126 + w / 2;
    s += `<path class="edge hl" d="M${tx},${y2 + h + 2} C${tx},${y2 + h + 34} ${8 + 4 * 126 + w + 6},${y2 + h + 34} ${8 + 4 * 126 + w + 6},${y1 + h / 2} L${8 + 4 * 126 + w + 2},${y1 + h / 2}" marker-end="url(#ml1ahA)" style="fill:none"/>`;
    s += txt(tx + 120, y2 + h + 36, "iterate: new features, other models", "fill:var(--accent);font-size:11px");
    const mx = 71 + w / 2;
    s += `<path class="edge" d="M${mx - 40},${y2 + h / 2} C${-20},${y2 + h / 2} ${-20},${y1 + h + 30} ${8 + 126 + w / 2},${y1 + h + 3}" marker-end="url(#ml1ah)" style="fill:none;stroke-dasharray:4 4"/>`;
    s += txt(4, y2 + h + 20, "data drifts → retrain", "font-size:11px", "start");
    return svgWrap(640, 232, s);
  };

  /* train / validation / test split, plus k-fold */
  D.ml1_split = () => {
    const X = 20, W = 600;
    let s = `<rect x="${X}" y="20" width="${W * 0.6}" height="42" rx="6" style="fill:color-mix(in srgb, var(--accent) 30%, transparent);stroke:var(--accent)"/>` +
      `<rect x="${X + W * 0.6}" y="20" width="${W * 0.2}" height="42" rx="6" style="fill:color-mix(in srgb, var(--amber) 30%, transparent);stroke:var(--amber)"/>` +
      `<rect x="${X + W * 0.8}" y="20" width="${W * 0.2}" height="42" rx="6" style="fill:color-mix(in srgb, var(--green) 30%, transparent);stroke:var(--green)"/>`;
    s += txt(X + W * 0.3, 46, "Training set (≈ 60–80%)", "font-weight:700;fill:var(--text);font-size:13px") + txt(X + W * 0.7, 46, "Validation", "font-weight:700;fill:var(--text);font-size:13px") + txt(X + W * 0.9, 46, "Test", "font-weight:700;fill:var(--text);font-size:13px");
    s += txt(X + W * 0.3, 80, "the model learns its parameters here") + txt(X + W * 0.7, 80, "choose settings,") + txt(X + W * 0.7, 94, "compare models") + txt(X + W * 0.9, 80, "final score,") + txt(X + W * 0.9, 94, "used once");
    s += txt(X, 128, "Small dataset? Use k-fold cross-validation on the training part instead of one validation set:", "font-size:12px;fill:var(--text)", "start");
    for (let k = 0; k < 5; k++) {
      const y = 140 + k * 20;
      for (let j = 0; j < 5; j++) s += `<rect x="${X + j * 96}" y="${y}" width="92" height="15" rx="3" style="fill:${j === k ? "color-mix(in srgb, var(--amber) 45%, transparent)" : "color-mix(in srgb, var(--accent) 22%, transparent)"}"/>`;
      s += txt(X + 5 * 96 + 8, y + 12, `fold ${k + 1}: validate on the amber part`, "font-size:11px", "start");
    }
    return svgWrap(640, 246, s);
  };

  /* bias–variance trade-off (illustrative curves) */
  D.ml1_biasvar = () => {
    const b2 = (x) => 0.95 * Math.exp(-0.42 * x) + 0.02, vr = (x) => 0.02 * Math.exp(0.36 * x), nz = 0.12, tot = (x) => b2(x) + vr(x) + nz;
    let best = 0, bv = 9; for (let x = 0; x <= 10; x += 0.01) if (tot(x) < bv) { bv = tot(x); best = x; }
    return Plot.chart({ x: [0, 10], y: [0, 1.25], W: 560, H: 300, xl: "model complexity (degree, depth, fewer constraints …) →", yl: "error", grid: false }, [
      Plot.hline(nz), Plot.fn(b2, "pl-c1"), Plot.fn(vr, "pl-c4"), Plot.fn(tot, "pl-c5"), Plot.vline(best),
      Plot.text(1.4, 0.93, "bias²", "", "start"), Plot.text(8.9, 0.62, "variance", "", "end"), Plot.text(5.6, 0.62, "total error", "", "middle"), Plot.text(9.8, nz + 0.04, "irreducible noise σ²", "", "end"),
      Plot.text(1.6, 1.16, "← underfitting", "", "middle"), Plot.text(best, 1.16, "sweet spot", "", "middle"), Plot.text(8.4, 1.16, "overfitting →", "", "middle"),
      Plot.dot(best, bv, "pl-c5", 5),
    ]);
  };

  /* loss functions: "reg" (vs residual) or "clf" (vs margin) */
  D.ml1_losses = (kind) => {
    if (kind === "clf") return Plot.chart({ x: [-3, 3], y: [0, 4], W: 520, H: 280, xl: "margin m = y·f(x)   (negative = wrong side)", yl: "loss" }, [
      Plot.fn((m) => (m < 0 ? 1 : 0), "pl-c3"), Plot.fn((m) => Math.log2(1 + Math.exp(-m)), "pl-c1"), Plot.fn((m) => Math.max(0, 1 - m), "pl-c2"),
      Plot.text(-2.9, 1.18, "0–1 loss", "", "start"), Plot.text(-1.9, 3.3, "log loss", "", "end"), Plot.text(-2.6, 3.85, "hinge", "", "end"),
    ]);
    const hub = (r, d = 1) => (Math.abs(r) <= d ? 0.5 * r * r : d * (Math.abs(r) - 0.5 * d));
    return Plot.chart({ x: [-3, 3], y: [0, 4], W: 520, H: 280, xl: "residual r = y − ŷ", yl: "loss" }, [
      Plot.fn((r) => r * r, "pl-c1"), Plot.fn((r) => Math.abs(r), "pl-c2"), Plot.fn((r) => hub(r), "pl-c3"),
      Plot.text(-1.75, 3.5, "squared r²", "", "end"), Plot.text(-2.95, 2.75, "absolute |r|", "", "start"), Plot.text(2.95, 1.6, "Huber (δ = 1)", "", "end"),
    ]);
  };

  /* why L1 gives exact zeros: loss contours touch the diamond at a corner */
  D.ml1_l1l2 = () => {
    const c = [1.6, 2.4], H = [[2.2, 1.2], [1.2, 1.4]];                // loss Q(w) = (w − c)ᵀ H (w − c), unconstrained minimum at c
    const Q = (w) => { const a = w[0] - c[0], b = w[1] - c[1]; return H[0][0] * a * a + 2 * H[0][1] * a * b + H[1][1] * b * b; };
    // eigen-decomposition of the 2×2 H for drawing ellipses
    const tr = H[0][0] + H[1][1], det = H[0][0] * H[1][1] - H[0][1] ** 2, l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr / 2 - Math.sqrt(tr * tr / 4 - det);
    const u1 = (() => { const v = [H[0][1], l1 - H[0][0]], n = Math.hypot(...v); return [v[0] / n, v[1] / n]; })(), u2 = [-u1[1], u1[0]];
    const ell = (lev) => { const p = []; for (let i = 0; i <= 120; i++) { const t = (i / 120) * 2 * Math.PI, a = Math.sqrt(lev / l1) * Math.cos(t), b = Math.sqrt(lev / l2) * Math.sin(t); p.push([c[0] + a * u1[0] + b * u2[0], c[1] + a * u1[1] + b * u2[1]]); } return p; };
    const panel = (title, boundary) => {
      let best = null; boundary.forEach((w) => { const q = Q(w); if (!best || q < best.q) best = { w, q }; });
      return Plot.chart({ x: [-1.6, 3.2], y: [-1.6, 3.2], W: 300, H: 300, xl: "w₁", yl: "w₂" }, [
        Plot.hline(0, "pl-thin\" style=\"stroke:var(--muted)"), Plot.vline(0, "pl-thin\" style=\"stroke:var(--muted)"),
        (S) => `<path d="${boundary.map((w, i) => (i ? "L" : "M") + S.sx(w[0]).toFixed(1) + "," + S.sy(w[1]).toFixed(1)).join("")}Z" style="fill:color-mix(in srgb, var(--accent) 20%, transparent);stroke:var(--accent);stroke-width:2"/>`,
        ...[best.q * 3.2, best.q * 2, best.q * 1.35].map((l) => Plot.line(ell(l), "pl-dash")), Plot.line(ell(best.q), "pl-c2 pl-thin"),
        Plot.dot(c[0], c[1], "pl-c4", 4.5), Plot.text(c[0] + 0.12, c[1] + 0.16, "ŵ (no penalty)", "", "start"),
        Plot.dot(best.w[0], best.w[1], "pl-c2", 6), Plot.text(-1.5, 2.95, title, "", "start"),
      ]);
    };
    const N = 2000, t = 1.25;
    const diamond = Array.from({ length: N }, (_, i) => { const a = (i / N) * 2 * Math.PI, ca = Math.cos(a), sa = Math.sin(a), s = t / (Math.abs(ca) + Math.abs(sa)); return [s * ca, s * sa]; });
    const circle = Array.from({ length: N }, (_, i) => { const a = (i / N) * 2 * Math.PI; return [t * Math.cos(a), t * Math.sin(a)]; });
    return row(panel("L1: |w₁| + |w₂| ≤ t", diamond), panel("L2: w₁² + w₂² ≤ t²", circle));
  };

  /* the kernel idea: 1-D points that no threshold separates become separable after x → (x, x²) */
  D.ml1_kernel = () => {
    const A = [-1.1, -0.6, -0.2, 0.3, 0.7, 1.2], B = [-3, -2.5, -2.1, 2, 2.4, 2.9];
    const one = Plot.chart({ x: [-3.5, 3.5], y: [-1, 1], W: 300, H: 150, xl: "x (1 feature): no single threshold works", grid: false }, [
      Plot.hline(0, "pl-thin\" style=\"stroke:var(--muted)"), Plot.pts(A.map((x) => [x, 0, "a"]), "", 6), Plot.pts(B.map((x) => [x, 0, "b"]), "", 6),
    ]);
    const two = Plot.chart({ x: [-3.5, 3.5], y: [-0.5, 9.5], W: 300, H: 260, xl: "x", yl: "x²" }, [
      Plot.fn((x) => x * x, "pl-thin\" style=\"stroke:var(--border)"), Plot.hline(2.6, "pl-c5"), Plot.pts(A.map((x) => [x, x * x, "a"]), "", 6), Plot.pts(B.map((x) => [x, x * x, "b"]), "", 6),
      Plot.text(3.4, 3.2, "a straight line now separates them", "", "end"),
    ]);
    return row(one, `<div style="align-self:center;font-size:22px;color:var(--muted);padding:0 4px">→ φ(x) = (x, x²) →</div>`, two);
  };

  /* a real depth-2 tree on the iris data (numbers from scikit-learn, random_state=0) */
  D.ml1_treechart = () => {
    const q = (x, y, w, l1, l2, cls = "node") => `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="46" rx="9"/>` + txt(x + w / 2, y + 19, l1, "font-size:13px;font-weight:700;fill:var(--text)") + txt(x + w / 2, y + 36, l2, "font-size:11px");
    let s = "";
    s += q(215, 10, 210, "petal width ≤ 0.80 cm ?", "150 flowers · Gini 0.667", "node hl");
    s += arrow(270, 58, 135, 98) + txt(186, 78, "yes", "font-size:11px;font-weight:700;fill:var(--accent)") + arrow(370, 58, 445, 98) + txt(425, 78, "no", "font-size:11px;font-weight:700;fill:var(--accent)");
    s += q(40, 100, 190, "→ setosa", "50 flowers: 50 / 0 / 0 · Gini 0", "node ok");
    s += q(340, 100, 220, "petal width ≤ 1.75 cm ?", "100 flowers · Gini 0.5", "node hl");
    s += arrow(400, 148, 300, 188) + txt(335, 170, "yes", "font-size:11px;font-weight:700;fill:var(--accent)") + arrow(500, 148, 540, 188) + txt(538, 170, "no", "font-size:11px;font-weight:700;fill:var(--accent)");
    s += q(180, 190, 220, "→ versicolor", "54 flowers: 0 / 49 / 5 · Gini 0.168", "node ok");
    s += q(420, 190, 210, "→ virginica", "46 flowers: 0 / 1 / 45 · Gini 0.043", "node ok");
    return svgWrap(640, 246, s);
  };

  /* bagging (parallel) vs boosting (sequential) */
  D.ml1_ensembles = () => {
    let s = txt(10, 20, "Bagging / random forest: independent trees, in parallel", "font-weight:700;fill:var(--text);font-size:13px", "start");
    s += box(10, 66, 90, 40, "Data");
    [0, 1, 2].forEach((i) => { const y = 32 + i * 40; s += arrow(102, 86, 150, y + 16) + box(152, y, 120, 32, `bootstrap ${i + 1}`) + arrow(274, y + 16, 300, y + 16) + box(302, y, 80, 32, `tree ${i + 1}`, "node hl") + arrow(384, y + 16, 448, 86); });
    s += box(450, 66, 170, 40, "vote / average", "node ok");
    const y0 = 172;
    s += txt(10, y0, "Boosting: each tree fixes the previous trees' errors, in sequence", "font-weight:700;fill:var(--text);font-size:13px", "start");
    s += box(10, y0 + 14, 70, 36, "Data");
    [0, 1, 2].forEach((i) => { const x = 104 + i * 150; s += arrow(x - 22, y0 + 32, x - 2, y0 + 32) + box(x, y0 + 14, 74, 36, `tree ${i + 1}`, "node hl"); if (i < 2) s += arrow(x + 76, y0 + 32, x + 96, y0 + 32) + txt(x + 112, y0 + 26, "errors", "font-size:10px") + box(x + 98, y0 + 34, 30, 2, "", "node"); });
    s += arrow(480, y0 + 32, 500, y0 + 32) + box(502, y0 + 14, 118, 36, "weighted sum", "node ok");
    return svgWrap(640, 236, s);
  };

  /* ================================================================== glossary */
  Object.assign(window.GLOSSARY, {
    "residual": "The gap between the true value and the model's prediction for one example: y − ŷ.",
    "mean squared error": "The average of the squared residuals. The standard loss for regression; big mistakes count a lot.",
    "least squares": "Choosing the parameters that make the sum of squared residuals as small as possible.",
    "normal equation": "The closed-form solution of linear least squares: w = (XᵀX)⁻¹Xᵀy. Exact, but slow when there are very many features.",
    "intercept": "The constant term b in a linear model: the prediction when every feature is 0. Also called the bias term.",
    "empirical risk": "The average loss over the training examples: the number training actually minimises, as a stand-in for the error on all future data.",
    "baseline": "The simplest sensible model (predict the mean, or the most common class). A real model must beat it to be worth anything.",
    "stratified": "A split that keeps the class proportions the same in every part (e.g. 10% positives in train and in test).",
    "momentum": "An optimizer trick: keep a running sum of past gradients and step along it, so steps speed up along consistent directions and zig-zags cancel.",
    "polynomial features": "New columns made from powers and products of the original features (x², x₁x₂ …) so a linear model can fit curves.",
    "ridge": "Linear regression with an L2 penalty (α·Σw²) that shrinks all weights towards zero.",
    "lasso": "Linear regression with an L1 penalty (α·Σ|w|) that shrinks weights and sets some exactly to zero, so it also selects features.",
    "elasticnet": "Linear regression with a mix of the L1 (lasso) and L2 (ridge) penalties.",
    "decision boundary": "The line or surface where a classifier switches from predicting one class to another.",
    "log-odds": "log(p / (1 − p)). Logistic regression models the log-odds as a straight-line function of the features.",
    "log loss": "Binary cross-entropy: −[y·log p + (1 − y)·log(1 − p)]. It punishes confident wrong predictions very hard.",
    "hinge loss": "max(0, 1 − y·f(x)): the SVM loss. Zero once a point is on the right side with a margin of at least 1.",
    "huber loss": "Squared error for small residuals, absolute error for large ones: less sensitive to outliers than MSE.",
    "k-nn": "k-nearest neighbours: predict by a vote (or average) of the k most similar training examples.",
    "lazy learner": "A model that does no work at training time and just stores the data (like k-NN); all the work happens at prediction time.",
    "curse of dimensionality": "In many dimensions, points are all far apart and distances become similar, so 'nearest' neighbours stop being meaningfully near.",
    "laplace smoothing": "Adding a small count (often 1) to every feature-class count so that an unseen word doesn't make a probability exactly zero.",
    "support vectors": "The training points on or inside the SVM margin. They alone decide where the boundary goes.",
    "margin": "In an SVM, the width of the empty street between the decision boundary and the closest points of each class.",
    "kernel trick": "Computing dot products in a high-dimensional feature space with a cheap function k(x, z), without ever building those features.",
    "gini impurity": "1 − Σ pₖ²: the chance that two random picks from a node have different classes. 0 means the node is pure.",
    "information gain": "How much a split lowers impurity: the parent's impurity minus the weighted impurity of the children.",
    "bootstrap": "A random sample of n rows drawn with replacement from n rows: some rows appear twice, about 37% don't appear at all.",
    "bagging": "Bootstrap aggregating: train many models on bootstrap samples and average their predictions to cut variance.",
    "out-of-bag": "The rows a bootstrapped tree never saw. Scoring each tree on them gives a free validation estimate.",
    "random forest": "Many decision trees, each grown on a bootstrap sample and choosing each split from a random subset of features; their votes are averaged.",
    "boosting": "Building models one after another, each focusing on the errors the previous ones still make, and adding them up.",
    "weak learner": "A model only a little better than guessing, such as a depth-1 tree (a stump). Boosting combines many of them.",
    "early stopping": "Stop training when the validation score stops improving, before the model starts to overfit.",
    "multicollinearity": "When features are strongly correlated with each other; linear-model coefficients then become unstable and hard to interpret.",
  });
})();
