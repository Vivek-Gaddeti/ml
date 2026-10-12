/* Classical ML, part 1 (discriminants, kernels, probabilistic models): visualizers, diagrams and glossary terms.
   Loaded on every doc page and on Practice, after assets/viz.js. Every name starts with cml1_. */
(function () {
  const V = Viz.defs, Plot = Viz.Plot;

  /* ============================================================ private helpers */
  const rng = (seed) => { let a = (seed >>> 0) || 1; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  const gauss = (r) => { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const COL = ["var(--accent)", "var(--amber)", "var(--green)", "var(--red)", "var(--purple)"];
  const col = (j) => COL[((j % COL.length) + COL.length) % COL.length];
  const soft = (c, p) => `color-mix(in srgb, ${c} ${p || 18}%, transparent)`;
  const fx = (v, n) => (+v).toFixed(n == null ? 2 : n);
  const pct = (v) => (100 * v).toFixed(1) + "%";
  const SQ = { W: 398, H: 394 };
  const row = (...svgs) => `<div style="display:flex;flex-wrap:wrap;gap:10px 14px;justify-content:center;align-items:center">${svgs.join("")}</div>`;
  const colm = (...svgs) => `<div style="display:flex;flex-direction:column;gap:6px;align-items:center">${svgs.join("")}</div>`;
  const D2R = Math.PI / 180;

  function chart(o, layers, outside) {
    let SS = null;
    const s = Plot.chart(o, [(S) => { SS = S; return ""; }].concat(layers));
    return outside ? s.replace(/<\/svg>\s*$/, outside(SS) + "</svg>") : s;
  }
  const circles = (pts, r0) => (S) => pts.map((p) => `<circle cx="${S.sx(p[0]).toFixed(1)}" cy="${S.sy(p[1]).toFixed(1)}" r="${p[3] || r0 || 4.5}" style="${p[2]}"/>`).join("");
  const fillDot = (c, extra) => `fill:${c};stroke:var(--panel);stroke-width:1.2${extra || ""}`;
  const pathOf = (S, pts) => pts.map(([x, y], i) => (i ? "L" : "M") + S.sx(x).toFixed(1) + "," + S.sy(y).toFixed(1)).join("");
  const curve = (pts, c, w, dash) => (S) => `<path d="${pathOf(S, pts)}" style="fill:none;stroke:${c};stroke-width:${w || 2.4};stroke-linejoin:round;stroke-linecap:round${dash ? ";stroke-dasharray:" + dash : ""}"/>`;
  const area = (pts, c, p) => (S) => `<path d="${pathOf(S, pts)}Z" style="fill:${soft(c, p || 16)};stroke:none"/>`;
  const segPx = (S, x1, y1, x2, y2, style) => `<line x1="${S.sx(x1).toFixed(1)}" y1="${S.sy(y1).toFixed(1)}" x2="${S.sx(x2).toFixed(1)}" y2="${S.sy(y2).toFixed(1)}" style="${style}"/>`;
  const txt = (x, y, s, o) => { o = o || {}; return `<text x="${(+x).toFixed(1)}" y="${(+y).toFixed(1)}" text-anchor="${o.a || "middle"}" style="font-size:${o.fs || 12}px;fill:${o.c || "var(--text)"}${o.b ? ";font-weight:700" : ""}">${esc(s)}</text>`;};
  const svgWrap = (W, H, inner) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img"><defs><marker id="cml1ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker></defs>${inner}</svg>`;
  const box = (x, y, w, h, c, p, r) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r == null ? 7 : r}" style="fill:${soft(c, p || 14)};stroke:${c};stroke-width:1.5"/>`;
  const arrow = (x1, y1, x2, y2, c, w, dash) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" style="stroke:${c || "var(--muted)"};stroke-width:${w || 1.8}${dash ? ";stroke-dasharray:" + dash : ""}" marker-end="url(#cml1ah)"/>`;

  /* ---- tiny linear algebra ---- */
  const chol = (A) => {                                // lower Cholesky factor of a symmetric positive-definite matrix
    const n = A.length, L = A.map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
    }
    return L;
  };
  const fwd = (L, b) => { const y = []; for (let i = 0; i < L.length; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y.push(s / L[i][i]); } return y; };
  const bwd = (L, y) => { const n = L.length, x = new Array(n); for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; } return x; };
  const m2inv = (a) => { const d = a[0][0] * a[1][1] - a[0][1] * a[1][0]; return [[a[1][1] / d, -a[0][1] / d], [-a[1][0] / d, a[0][0] / d]]; };
  const m2det = (a) => a[0][0] * a[1][1] - a[0][1] * a[1][0];
  const m2v = (a, v) => [a[0][0] * v[0] + a[0][1] * v[1], a[1][0] * v[0] + a[1][1] * v[1]];
  const mean2 = (P) => [P.reduce((s, p) => s + p[0], 0) / P.length, P.reduce((s, p) => s + p[1], 0) / P.length];
  const scatter2 = (P, m) => { const S = [[0, 0], [0, 0]]; P.forEach((p) => { const a = p[0] - m[0], b = p[1] - m[1]; S[0][0] += a * a; S[0][1] += a * b; S[1][0] += a * b; S[1][1] += b * b; }); return S; };
  const ellipsePts = (m, Sg, k, n) => {               // the k-sigma contour of a 2-D Gaussian
    const L = chol(Sg), out = [];
    for (let i = 0; i <= (n || 48); i++) { const t = (2 * Math.PI * i) / (n || 48), u = Math.cos(t) * k, v = Math.sin(t) * k; out.push([m[0] + L[0][0] * u, m[1] + L[1][0] * u + L[1][1] * v]); }
    return out;
  };
  const ellipse = (m, Sg, k, c, fillp, w) => (S) => `<path d="${pathOf(S, ellipsePts(m, Sg, k))}Z" style="fill:${fillp ? soft(c, fillp) : "none"};stroke:${c};stroke-width:${w || 2}"/>`;
  const through = (deg, R, style) => (S) => { const t = deg * D2R; return segPx(S, -R * Math.cos(t), -R * Math.sin(t), R * Math.cos(t), R * Math.sin(t), style); };
  const angDiff = (a, b) => { const d = Math.abs((((a - b) % 180) + 180) % 180); return Math.min(d, 180 - d); };

  /* ============================================================ LDA vs PCA */
  const LDA_DATA = (() => {
    const r = rng(11), th = 25 * D2R, main = [Math.cos(th), Math.sin(th)], perp = [-Math.sin(th), Math.cos(th)], A = [], B = [];
    for (let c = 0; c < 2; c++) for (let i = 0; i < 45; i++) {
      const u = gauss(r) * 1.9, v = gauss(r) * 0.55, s = c ? 1 : -1;
      const p = [u * main[0] + v * perp[0] + s * (0.9 * perp[0] + 0.3 * main[0]), u * main[1] + v * perp[1] + s * (0.9 * perp[1] + 0.3 * main[1])];
      (c ? B : A).push(p);
    }
    const all = A.concat(B), m = mean2(all);
    const cen = (P) => P.map((p) => [p[0] - m[0], p[1] - m[1]]);
    return { A: cen(A), B: cen(B) };
  })();
  const LDA_FIT = (() => {
    const { A, B } = LDA_DATA, mA = mean2(A), mB = mean2(B), SA = scatter2(A, mA), SB = scatter2(B, mB);
    const Sw = [[SA[0][0] + SB[0][0], SA[0][1] + SB[0][1]], [SA[1][0] + SB[1][0], SA[1][1] + SB[1][1]]];
    const w = m2v(m2inv(Sw), [mB[0] - mA[0], mB[1] - mA[1]]), nr = Math.hypot(w[0], w[1]);
    let ang = (Math.atan2(w[1], w[0]) / D2R + 360) % 180;
    const all = A.concat(B), T = scatter2(all, mean2(all)), tr = T[0][0] + T[1][1], dsc = Math.sqrt(((T[0][0] - T[1][1]) / 2) ** 2 + T[0][1] ** 2);
    let pca = (0.5 * Math.atan2(2 * T[0][1], T[0][0] - T[1][1]) / D2R + 360) % 180;
    return { mA, mB, ang, pca, w: [w[0] / nr, w[1] / nr] };
  })();
  function fisherJ(deg) {                              // (difference of projected means)² / (sum of projected scatters)
    const { A, B } = LDA_DATA, t = deg * D2R, u = [Math.cos(t), Math.sin(t)];
    const pa = A.map((p) => p[0] * u[0] + p[1] * u[1]), pb = B.map((p) => p[0] * u[0] + p[1] * u[1]);
    const ma = pa.reduce((s, v) => s + v, 0) / pa.length, mb = pb.reduce((s, v) => s + v, 0) / pb.length;
    const sa = pa.reduce((s, v) => s + (v - ma) ** 2, 0), sb = pb.reduce((s, v) => s + (v - mb) ** 2, 0);
    return { J: (mb - ma) ** 2 / (sa + sb), pa, pb, ma, mb };
  }
  V.cml1_lda = {
    title: "LDA vs PCA: which line should the data be squashed onto?",
    params: [{ id: "deg", label: "direction θ (degrees)", min: 0, max: 179, step: 1, value: 28 }],
    gen(_, P) {
      const { A, B } = LDA_DATA, F = fisherJ(P.deg), phi = LDA_FIT.ang, pca = LDA_FIT.pca;
      const Js = []; let Jmax = 0; for (let d = 0; d < 180; d += 2) { const j = fisherJ(d).J; Js.push([d, j]); Jmax = Math.max(Jmax, j); }
      const Jbest = fisherJ(phi).J, Jpca = fisherJ(pca).J, t = P.deg * D2R;
      const tag = (S, deg, s, c) => { let x = 5.3 * Math.cos(deg * D2R), y = 5.3 * Math.sin(deg * D2R); if (y < -5) { x = -x; y = -y; } return txt(S.sx(x), S.sy(y) + 4, s, { fs: 12, b: 1, c }); };
      const left = chart({ ...SQ, x: [-6, 6], y: [-6, 6], xl: "feature 1", yl: "feature 2" }, [
        through(pca, 9, "stroke:var(--purple);stroke-width:1.6;stroke-dasharray:6 4"),
        through(phi, 9, "stroke:var(--green);stroke-width:1.6;stroke-dasharray:6 4"),
        through(P.deg, 9, "stroke:var(--amber);stroke-width:2.6"),
        circles(A.map((p) => [p[0], p[1], fillDot("var(--cls-a)")]), 4),
        circles(B.map((p) => [p[0], p[1], fillDot("var(--cls-b)")]), 4),
      ], (S) => tag(S, pca, "PCA", "var(--purple)") + tag(S, phi, "LDA", "var(--green)"));
      const nb = 28, lo = -7, hi = 7, bw = (hi - lo) / nb, ca = new Array(nb).fill(0), cb = new Array(nb).fill(0);
      const bin = (v) => Math.max(0, Math.min(nb - 1, Math.floor((v - lo) / bw)));
      F.pa.forEach((v) => ca[bin(v)]++); F.pb.forEach((v) => cb[bin(v)]++);
      const top = Math.max(8, ...ca, ...cb) * 1.12;
      const bars = (cnt, c) => (S) => cnt.map((n, i) => n ? `<rect x="${S.sx(lo + i * bw).toFixed(1)}" y="${S.sy(n).toFixed(1)}" width="${(S.sx(lo + bw) - S.sx(lo) - 1).toFixed(1)}" height="${(S.sy(0) - S.sy(n)).toFixed(1)}" style="fill:${c};opacity:.62"/>` : "").join("");
      const hist = chart({ W: 262, H: 188, x: [lo, hi], y: [0, top], xl: "position along the amber line", yl: "count", ticks: 4, yticks: 4 }, [bars(ca, "var(--cls-a)"), bars(cb, "var(--cls-b)")]);
      const jc = chart({ W: 262, H: 188, x: [0, 180], y: [0, Jmax * 1.15], xl: "direction θ (°)", yl: "Fisher score J", ticks: 4, yticks: 4 }, [
        Plot.line(Js, "pl-c2"), Plot.vline(P.deg), Plot.dot(phi, Jbest, "pl-c3", 5.5), Plot.dot(pca, Jpca, "pl-c5", 5.5), Plot.dot(P.deg, F.J, "pl-c2", 6.5),
      ]);
      const atL = angDiff(P.deg, phi) <= 2, atP = angDiff(P.deg, pca) <= 2;
      const msg = atL ? `This is the **LDA direction** (green). The two classes land in separate heaps on the line: the class means are far apart compared with the spread inside each class. Fisher's score $J$ = ${fx(F.J, 3)} is the highest possible.`
        : atP ? `This is the **PCA direction** (purple): the line along which the whole cloud is most spread out. But that spread is *inside* each class, so after projecting the two heaps sit on top of each other. $J$ = ${fx(F.J, 3)}, far below the best ${fx(Jbest, 3)}.`
        : `Every point is dropped onto the amber line and its position along it is histogrammed. $J$ = (gap between the two class means)² ÷ (total scatter inside the classes) = **${fx(F.J, 3)}**. Slide θ to maximise it: the best is the green LDA line (J = ${fx(Jbest, 3)}), while PCA's purple line only gets ${fx(Jpca, 3)}.`;
      return [{ svg: row(left, colm(hist, jc)), side: { "θ": P.deg + "°", "Fisher score J": fx(F.J, 3), "LDA direction": phi.toFixed(0) + "° (J " + fx(Jbest, 3) + ")", "PCA direction": pca.toFixed(0) + "° (J " + fx(Jpca, 3) + ")" }, msg }];
    },
  };

  /* ============================================================ QDA vs LDA */
  const QDA_Z = (() => { const r = rng(23), out = { A: [], B: [] }; for (let i = 0; i < 70; i++) { out.A.push([gauss(r), gauss(r)]); out.B.push([gauss(r), gauss(r)]); } return out; })();
  V.cml1_qda = {
    title: "QDA: a curved boundary when the classes have different spreads",
    params: [{ id: "s", label: "spread of the orange class (× blue)", min: 0.7, max: 2.6, step: 0.05, value: 2.0 }],
    gen(_, P) {
      const L = [[0.8, 0], [0.35, 0.55]], mk = (z, m, k) => z.map((q) => [m[0] + k * (L[0][0] * q[0] + L[0][1] * q[1]), m[1] + k * (L[1][0] * q[0] + L[1][1] * q[1])]);
      const A = mk(QDA_Z.A, [-1.0, -0.4], 1), B = mk(QDA_Z.B, [0.9, 0.5], P.s);
      const mA = mean2(A), mB = mean2(B), sA = scatter2(A, mA), sB = scatter2(B, mB), n = A.length;
      const cov = (S) => [[S[0][0] / (n - 1), S[0][1] / (n - 1)], [S[1][0] / (n - 1), S[1][1] / (n - 1)]];
      const CA = cov(sA), CB = cov(sB), CP = [[(CA[0][0] + CB[0][0]) / 2, (CA[0][1] + CB[0][1]) / 2], [(CA[1][0] + CB[1][0]) / 2, (CA[1][1] + CB[1][1]) / 2]];
      const IA = m2inv(CA), IB = m2inv(CB), IP = m2inv(CP), lA = Math.log(m2det(CA)), lB = Math.log(m2det(CB));
      const quad = (I, m, p) => { const d = [p[0] - m[0], p[1] - m[1]], t = m2v(I, d); return d[0] * t[0] + d[1] * t[1]; };
      const qda = (x, y) => (-0.5 * lA - 0.5 * quad(IA, mA, [x, y])) - (-0.5 * lB - 0.5 * quad(IB, mB, [x, y]));   // > 0 → blue
      const lda = (x, y) => (-0.5 * quad(IP, mA, [x, y])) - (-0.5 * quad(IP, mB, [x, y]));
      const acc = (f) => (A.filter((p) => f(p[0], p[1]) > 0).length + B.filter((p) => f(p[0], p[1]) <= 0).length) / (2 * n);
      // LDA boundary is a straight line: f(x,y) = 0; find it from f at three points
      const f0 = lda(0, 0), fx1 = lda(1, 0) - f0, fy1 = lda(0, 1) - f0;
      const lineY = (x) => -(f0 + fx1 * x) / (fy1 || 1e-9);
      const left = chart({ ...SQ, x: [-5, 5], y: [-4, 5], xl: "feature 1", yl: "feature 2" }, [
        Plot.regions((x, y) => (qda(x, y) > 0 ? "a" : "b"), 50),
        (S) => {
          const pts = []; for (let i = 0; i <= 40; i++) { const x = -5 + i * 0.25, y = lineY(x); if (y > -4.2 && y < 5.2) pts.push([x, y]); }
          return pts.length > 1 ? `<path d="${pathOf(S, pts)}" style="fill:none;stroke:var(--text);stroke-width:2;stroke-dasharray:7 4"/>` : "";
        },
        Plot.pts(A.map((p) => [p[0], p[1], "a"]), "pl-a", 3.8), Plot.pts(B.map((p) => [p[0], p[1], "b"]), "pl-b", 3.8),
      ]);
      const near = Math.abs(P.s - 1) < 0.12;
      const msg = near ? `Both classes have about the same spread, so the best boundary really is a straight line and QDA's curve almost coincides with LDA's dashed line. Equal covariances is exactly the assumption LDA makes.`
        : `The orange class is ${fx(P.s, 2)}× more spread out. A tight class inside a wide one cannot be cut off by a straight line, so the **shaded regions** (QDA) bend around the blue class while the **dashed line** (LDA) cannot. Training accuracy: QDA ${pct(acc(qda))}, LDA ${pct(acc(lda))}.`;
      return [{ svg: left, side: { "orange spread": fx(P.s) + "×", "QDA accuracy": pct(acc(qda)), "LDA accuracy": pct(acc(lda)), "|Σ_orange| / |Σ_blue|": fx(m2det(CB) / m2det(CA), 1) }, msg }];
    },
  };

  /* ============================================================ Bayesian linear regression */
  const BLR_PTS = (() => {
    const r = rng(5), xs = [-0.8, 0.6, -0.2, 0.9, -0.5, 0.2, 0.75, -0.95, 0.0, 0.4], w = [-0.3, 0.5];
    return xs.map((x) => [x, w[0] + w[1] * x + 0.2 * gauss(r)]);
  })();
  const BLR_Z = (() => { const r = rng(9), z = []; for (let i = 0; i < 6; i++) z.push([gauss(r), gauss(r)]); return z; })();
  V.cml1_bayeslin = {
    title: "Bayesian linear regression: the posterior tightens as data arrives",
    params: [{ id: "alpha", label: "prior precision α (bigger = stronger belief that w ≈ 0)", min: 0.2, max: 12, step: 0.2, value: 2 }],
    gen(_, P) {
      const beta = 25, a = P.alpha, w0 = [-0.3, 0.5], S = [];
      for (let n = 0; n <= BLR_PTS.length; n++) {
        const pts = BLR_PTS.slice(0, n);
        let A = [[a, 0], [0, a]], b = [0, 0];                                   // A = S_N⁻¹ = αI + β ΦᵀΦ,  b = β Φᵀ y
        pts.forEach(([x, y]) => { A[0][0] += beta; A[0][1] += beta * x; A[1][0] += beta * x; A[1][1] += beta * x * x; b[0] += beta * y; b[1] += beta * x * y; });
        const SN = m2inv(A), mN = m2v(SN, b), Lc = chol(SN);
        const predSd = (x) => Math.sqrt(1 / beta + SN[0][0] + 2 * SN[0][1] * x + SN[1][1] * x * x);
        const xs = []; for (let i = 0; i <= 40; i++) xs.push(-1.1 + i * 0.055);
        const upper = xs.map((x) => [x, mN[0] + mN[1] * x + 2 * predSd(x)]), lower = xs.map((x) => [x, mN[0] + mN[1] * x - 2 * predSd(x)]).reverse();
        const lines = BLR_Z.map((z) => { const w = [mN[0] + Lc[0][0] * z[0], mN[1] + Lc[1][0] * z[0] + Lc[1][1] * z[1]]; return (Sx) => segPx(Sx, -1.1, w[0] - 1.1 * w[1], 1.1, w[0] + 1.1 * w[1], "stroke:var(--purple);stroke-width:1;opacity:.55"); });
        const left = chart({ W: 400, H: 330, x: [-1.1, 1.1], y: [-1.8, 1.4], xl: "x", yl: "y" }, [
          area(upper.concat(lower), "var(--accent)", 16), ...lines,
          curve([[-1.1, mN[0] - 1.1 * mN[1]], [1.1, mN[0] + 1.1 * mN[1]]], "var(--accent)", 2.6),
          circles(pts.map(([x, y]) => [x, y, fillDot("var(--amber)")]), 5),
        ]);
        const pr = ellipsePts(mN, SN, 1, 48), pr2 = ellipsePts(mN, SN, 2, 48), pp = ellipsePts([0, 0], [[1 / a, 0], [0, 1 / a]], 1, 48);
        const right = chart({ W: 300, H: 330, x: [-1.5, 1.5], y: [-1.5, 1.5], xl: "intercept w₀", yl: "slope w₁" }, [
          curve(pp, "var(--muted)", 1.5, "5 4"),
          (Sx) => `<path d="${pathOf(Sx, pr2)}Z" style="fill:${soft("var(--accent)", 10)};stroke:var(--accent);stroke-width:1.3;opacity:.8"/><path d="${pathOf(Sx, pr)}Z" style="fill:${soft("var(--accent)", 22)};stroke:var(--accent);stroke-width:2"/>`,
          Plot.dot(w0[0], w0[1], "pl-c3", 5.5), Plot.dot(mN[0], mN[1], "pl-c2", 5),
        ]);
        const msg = n === 0 ? `**Before any data** we only have the prior: every line through the plot is possible, weights are expected near 0 (dashed grey circle, radius 1/√α = ${fx(1 / Math.sqrt(a))}). The purple lines are random draws from this belief and the blue band is the uncertainty.`
          : `After **${n}** point${n > 1 ? "s" : ""}: the posterior is a Gaussian over $(w_0, w_1)$ with mean (${fx(mN[0])}, ${fx(mN[1])}). The right picture shows it shrinking onto the true weights (green dot, (−0.3, 0.5)). The band on the left is ±2 predictive standard deviations: it is widest far from the data and never below the noise level 1/√β = ${fx(1 / Math.sqrt(beta))}.`;
        S.push({ svg: row(left, right), side: { "points seen": n, "posterior mean w": `(${fx(mN[0])}, ${fx(mN[1])})`, "posterior sd of w": `(${fx(Math.sqrt(SN[0][0]))}, ${fx(Math.sqrt(SN[1][1]))})`, "prior α": a }, msg });
      }
      return S;
    },
  };

  /* ============================================================ Gaussian process regression */
  const GP_X = [-4, -2.8, -1.5, 0.3, 1.6, 3.2], GP_Y = GP_X.map((x, i) => Math.sin(x) + [0.1, -0.08, 0.05, -0.1, 0.06, 0.04][i]);
  const GP_G = (() => { const g = []; for (let i = 0; i <= 70; i++) g.push(-5 + i * (10 / 70)); return g; })();
  const GP_Z = (() => { const r = rng(4), z = []; for (let s = 0; s < 3; s++) z.push(GP_G.map(() => gauss(r))); return z; })();
  V.cml1_gp = {
    title: "Gaussian process regression: a smooth fit with honest error bars",
    params: [
      { id: "ell", label: "length-scale ℓ", min: 0.2, max: 3, step: 0.05, value: 1.2 },
      { id: "noise", label: "noise sd σₙ", min: 0.02, max: 0.6, step: 0.02, value: 0.1 },
    ],
    gen(_, P) {
      const l = P.ell, sn = P.noise, k = (a, b) => Math.exp(-((a - b) ** 2) / (2 * l * l)), n = GP_X.length;
      const K = GP_X.map((a, i) => GP_X.map((b, j) => k(a, b) + (i === j ? sn * sn : 0))), L = chol(K);
      const alpha = bwd(L, fwd(L, GP_Y));
      const Ks = GP_G.map((g) => GP_X.map((x) => k(g, x)));
      const mu = Ks.map((r) => r.reduce((s, v, i) => s + v * alpha[i], 0));
      const V_ = Ks.map((r) => fwd(L, r));                                   // L⁻¹ k*
      const sd = V_.map((v, i) => Math.sqrt(Math.max(1e-9, 1 - v.reduce((s, q) => s + q * q, 0))));
      const covPost = GP_G.map((a, i) => GP_G.map((b, j) => k(a, b) - V_[i].reduce((s, q, t) => s + q * V_[j][t], 0) + (i === j ? 1e-4 : 0)));
      const Lp = chol(covPost);
      const samples = GP_Z.map((z) => GP_G.map((_, i) => { let s = mu[i]; for (let j = 0; j <= i; j++) s += Lp[i][j] * z[j]; return [GP_G[i], s]; }));
      const lml = -0.5 * GP_Y.reduce((s, y, i) => s + y * alpha[i], 0) - L.reduce((s, r, i) => s + Math.log(r[i]), 0) - (n / 2) * Math.log(2 * Math.PI);
      const up = GP_G.map((g, i) => [g, mu[i] + 2 * sd[i]]), lo = GP_G.map((g, i) => [g, mu[i] - 2 * sd[i]]).reverse();
      const svg = chart({ W: 600, H: 340, x: [-5, 5], y: [-2.6, 2.6], xl: "x", yl: "f(x)" }, [
        area(up.concat(lo), "var(--accent)", 18),
        curve(GP_G.map((g) => [g, Math.sin(g)]), "var(--muted)", 1.6, "6 4"),
        ...samples.map((s) => curve(s, "var(--purple)", 1, null)).map((f) => (S) => `<g style="opacity:.5">${f(S)}</g>`),
        curve(GP_G.map((g, i) => [g, mu[i]]), "var(--accent)", 2.8),
        circles(GP_X.map((x, i) => [x, GP_Y[i], fillDot("var(--amber)")]), 5.5),
      ], (S) => txt(S.sx(-4.85), S.sy(2.35), "dashed grey = true sin(x) · blue = mean ± 2σ · purple = random draws", { a: "start", fs: 11, c: "var(--muted)" }));
      const msg = l < 0.5 ? `**ℓ = ${fx(l)} is too short**: the function is allowed to wiggle fast, so between the points it forgets everything and the band balloons back to the prior (±2). The fit chases the data and the evidence (log marginal likelihood ${fx(lml, 1)}) is poor.`
        : l > 2.2 ? `**ℓ = ${fx(l)} is too long**: the function is forced to be very smooth, so it cannot bend to follow sin(x) and the mean misses the data. Evidence ${fx(lml, 1)}.`
        : `ℓ = ${fx(l)} gives a smooth curve through the data. The band is thin near the amber points and widens where there are none, exactly the honest uncertainty a plain regression line cannot give. Log marginal likelihood ${fx(lml, 1)}: the data is more probable under good hyperparameters.`;
      return [{ svg, side: { "length-scale ℓ": fx(l), "noise σₙ": fx(sn), "log marginal likelihood": fx(lml, 2), "width of band at x=0": fx(4 * sd[35], 2) }, msg }];
    },
  };

  /* ============================================================ Gaussian mixture fitted by EM */
  const GMM_PTS = (() => {
    const r = rng(17), out = [];
    [[[-3.2, -1.6], [[1.0, 0], [0.6, 0.5]], 45], [[2.8, 2.6], [[0.6, 0], [-0.3, 0.9]], 40], [[3.2, -2.4], [[0.8, 0], [0, 0.5]], 35]].forEach(([m, L, n]) => {
      for (let i = 0; i < n; i++) { const a = gauss(r), b = gauss(r); out.push([m[0] + L[0][0] * a + L[0][1] * b, m[1] + L[1][0] * a + L[1][1] * b]); }
    });
    return out;
  })();
  const gpdf = (p, m, S) => { const d = [p[0] - m[0], p[1] - m[1]], I = m2inv(S), t = m2v(I, d); return Math.exp(-0.5 * (d[0] * t[0] + d[1] * t[1])) / (2 * Math.PI * Math.sqrt(m2det(S))); };
  function gmmEM(X, seed, K) {
    const r = rng(seed * 101 + 7), idx = []; while (idx.length < K) { const i = Math.floor(r() * X.length); if (!idx.includes(i)) idx.push(i); }
    let th = { mu: idx.map((i) => X[i].slice()), S: idx.map(() => [[2.5, 0], [0, 2.5]]), pi: idx.map(() => 1 / K) };
    const estep = (t) => { let ll = 0; const R = X.map((p) => { const w = t.mu.map((m, k) => t.pi[k] * gpdf(p, m, t.S[k])), s = w.reduce((a, b) => a + b, 0) || 1e-300; ll += Math.log(s); return w.map((v) => v / s); }); return { R, ll }; };
    const mstep = (R) => {
      const n = X.length, mu = [], S = [], pi = [];
      for (let k = 0; k < K; k++) {
        const Nk = R.reduce((s, q) => s + q[k], 0) || 1e-9, m = [0, 0]; X.forEach((p, i) => { m[0] += R[i][k] * p[0]; m[1] += R[i][k] * p[1]; }); m[0] /= Nk; m[1] /= Nk;
        const C = [[0, 0], [0, 0]]; X.forEach((p, i) => { const a = p[0] - m[0], b = p[1] - m[1], w = R[i][k]; C[0][0] += w * a * a; C[0][1] += w * a * b; C[1][0] += w * a * b; C[1][1] += w * b * b; });
        mu.push(m); S.push([[C[0][0] / Nk + 1e-3, C[0][1] / Nk], [C[1][0] / Nk, C[1][1] / Nk + 1e-3]]); pi.push(Nk / n);
      }
      return { mu, S, pi };
    };
    const frames = [{ kind: "init", th, R: null, ll: estep(th).ll }];
    for (let it = 1; it <= 25; it++) {
      const e = estep(th); frames.push({ kind: "E", it, th, R: e.R, ll: e.ll });
      th = mstep(e.R); const e2 = estep(th); frames.push({ kind: "M", it, th, R: e.R, ll: e2.ll });
      if (Math.abs(e2.ll - e.ll) < 1e-3) break;
    }
    return frames;
  }
  const mixColor = (q) => { const s12 = q[1] + q[2] || 1e-9; return `color-mix(in srgb, ${COL[0]} ${(q[0] * 100).toFixed(0)}%, color-mix(in srgb, ${COL[1]} ${((q[1] / s12) * 100).toFixed(0)}%, ${COL[2]}))`; };
  V.cml1_gmm = {
    title: "Gaussian mixture by EM: soft assignments, then refit",
    params: [{ id: "seed", label: "random start", min: 1, max: 6, step: 1, value: 1 }],
    gen(_, P) {
      const X = GMM_PTS, F = gmmEM(X, P.seed, 3), lls = F.filter((f) => f.kind !== "E").map((f) => f.ll), lo = Math.min(...lls), hi = Math.max(...lls), S = [];
      F.forEach((f, fi) => {
        const th = f.th, hist = F.slice(0, fi + 1).filter((g) => g.kind === "init" || g.kind === "M").map((g, i) => [i, g.ll]);
        const left = chart({ ...SQ, x: [-7, 7], y: [-6, 6], xl: "feature 1", yl: "feature 2" }, [
          ...th.mu.map((m, k) => ellipse(m, th.S[k], 2, col(k), 14, 2.2)),
          circles(X.map((p, i) => [p[0], p[1], `fill:${f.R ? mixColor(f.R[i]) : "var(--muted)"};stroke:var(--panel);stroke-width:1`]), 4),
          circles(th.mu.map((m, k) => [m[0], m[1], `fill:${col(k)};stroke:var(--text);stroke-width:2`]), 6),
        ]);
        const nIt = F.filter((g) => g.kind === "M").length;
        const right = chart({ W: 250, H: 394, x: [-0.4, Math.max(nIt, 4) + 0.4], y: [lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.1], xl: "EM iteration", yl: "log-likelihood", ticks: 4, yticks: 5 }, [Plot.line(hist, "pl-c3"), Plot.pts(hist, "pl-c3", 4)]);
        const last = fi === F.length - 1;
        const msg = f.kind === "init" ? `**Initial guess.** Three components, each a round Gaussian centred on a randomly chosen data point, with equal weights 1/3. The points are grey because nothing has been assigned yet. Log-likelihood ${fx(f.ll, 1)}.`
          : f.kind === "E" ? `**E-step ${f.it}.** Holding the Gaussians fixed, every point computes its *responsibility* for each component: the posterior probability that this component produced it. The colour of a point is the blend of the component colours by those probabilities, so points between two clouds are visibly mixed.`
          : `**M-step ${f.it}.** Holding the responsibilities fixed, each component is refit using *weighted* data: its mean is the weighted average of the points, its covariance the weighted scatter, its weight π the share of responsibility it received. Log-likelihood rose to ${fx(f.ll, 1)}${last ? ". It has stopped changing: **converged**. Try another random start to see a different local optimum." : "."}`;
        S.push({ svg: row(left, right), side: { step: f.kind === "init" ? "start" : f.kind + " " + f.it, "log-likelihood": fx(f.ll, 2), "weights π": th.pi.map((p) => fx(p, 2)).join(" / "), "sizes N_k": th.pi.map((p) => (p * X.length).toFixed(0)).join(" / ") }, msg });
      });
      return S;
    },
  };

  /* ============================================================ HMM: Viterbi trellis */
  const HMM = { st: ["Rainy", "Sunny"], obs: ["walk", "shop", "clean"], pi: [0.6, 0.4], A: [[0.7, 0.3], [0.4, 0.6]], B: [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]] };
  V.cml1_viterbi = {
    title: "Viterbi decoding: the most likely weather behind what Alice did",
    input: "walk shop clean", help: "what Alice did each day (walk, shop, clean), up to 5 days",
    gen(inp) {
      let o = (inp || "").toLowerCase().split(/[\s,]+/).filter((w) => HMM.obs.includes(w)).slice(0, 5); if (!o.length) o = ["walk", "shop", "clean"];
      const T = o.length, oi = o.map((w) => HMM.obs.indexOf(w)), d = [], ps = [];
      d.push([0, 1].map((s) => HMM.pi[s] * HMM.B[s][oi[0]])); ps.push([-1, -1]);
      for (let t = 1; t < T; t++) {
        const dt = [], pt = [];
        for (let s = 0; s < 2; s++) { const c = [0, 1].map((q) => d[t - 1][q] * HMM.A[q][s]), b = c[1] > c[0] ? 1 : 0; dt.push(c[b] * HMM.B[s][oi[t]]); pt.push(b); }
        d.push(dt); ps.push(pt);
      }
      const bestEnd = d[T - 1][1] > d[T - 1][0] ? 1 : 0, path = new Array(T); path[T - 1] = bestEnd; for (let t = T - 1; t > 0; t--) path[t - 1] = ps[t][path[t]];
      const X0 = 130, DX = 100, Y = [96, 186], W = 640, H = 260;
      const fv = (v) => (v >= 0.001 ? v.toFixed(4) : v.toExponential(1));
      // st: upto = number of columns computed; win = which arrows to emphasise; bt = how many path nodes from the end are lit
      function draw(upto, showWin, btFrom) {
        let s = "";
        for (let t = 0; t < T; t++) s += txt(X0 + t * DX, 36, o[t], { b: 1, fs: 13 }) + txt(X0 + t * DX, 22, "day " + (t + 1), { fs: 10.5, c: "var(--muted)" });
        for (let q = 0; q < 2; q++) s += txt(12, Y[q] + 4, HMM.st[q], { a: "start", b: 1, c: col(q === 0 ? 0 : 1) });
        for (let t = 1; t < upto; t++) for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
          const win = ps[t][b] === a, onPath = btFrom != null && t >= btFrom + 1 && path[t - 1] === a && path[t] === b;
          const stl = onPath ? "stroke:var(--green);stroke-width:4" : showWin && win && t === upto - 1 ? "stroke:var(--amber);stroke-width:3" : win ? "stroke:var(--muted);stroke-width:1.8" : "stroke:var(--border);stroke-width:1;stroke-dasharray:3 3";
          s += `<line x1="${X0 + (t - 1) * DX + 38}" y1="${Y[a]}" x2="${X0 + t * DX - 38}" y2="${Y[b]}" style="${stl}"/>`;
        }
        for (let t = 0; t < T; t++) for (let q = 0; q < 2; q++) {
          const x = X0 + t * DX, y = Y[q];
          if (t >= upto) { s += `<rect x="${x - 38}" y="${y - 22}" width="76" height="44" rx="8" style="fill:none;stroke:var(--border);stroke-width:1;stroke-dasharray:3 3"/>`; continue; }
          const lit = btFrom != null && t >= btFrom && path[t] === q;
          s += `<rect x="${x - 38}" y="${y - 22}" width="76" height="44" rx="8" style="fill:${soft(lit ? "var(--green)" : q === 0 ? "var(--accent)" : "var(--amber)", lit ? 30 : 14)};stroke:${lit ? "var(--green)" : q === 0 ? "var(--accent)" : "var(--amber)"};stroke-width:${lit ? 3 : 1.5}"/>` + txt(x, y + 5, fv(d[t][q]), { fs: 12.5, b: lit });
        }
        return svgWrap(W, H, s + txt(W / 2, 236, "each box = probability of the best path that ends in that state on that day · grey solid arrow = the winning predecessor", { fs: 11, c: "var(--muted)" }));
      }
      const Sx = [], fvv = (v) => v.toFixed(4);
      Sx.push({ svg: draw(1, false), side: { day: 1, observed: o[0] }, msg: `**Day 1.** No history yet, so the score of each state is: start probability × chance of seeing "${o[0]}" from that state. Rainy: ${HMM.pi[0]} × ${HMM.B[0][oi[0]]} = **${fvv(d[0][0])}**. Sunny: ${HMM.pi[1]} × ${HMM.B[1][oi[0]]} = **${fvv(d[0][1])}**.` });
      for (let t = 1; t < T; t++) {
        const m = [0, 1].map((s) => { const c = [0, 1].map((q) => d[t - 1][q] * HMM.A[q][s]); return `${HMM.st[s]}: best of (Rainy→ ${fvv(c[0])}, Sunny→ ${fvv(c[1])}) = ${fvv(Math.max(...c))} from ${HMM.st[ps[t][s]]}, × ${HMM.B[s][oi[t]]} = **${fvv(d[t][s])}**`; });
        Sx.push({ svg: draw(t + 1, true), side: { day: t + 1, observed: o[t] }, msg: `**Day ${t + 1}, saw "${o[t]}".** For each state keep only the best way of arriving: $\\delta_t(s)=\\max_q \\delta_{t-1}(q)\\,a_{qs}\\; b_s(o_t)$. ${m.join(". ")}. The amber arrows remember which predecessor won (the backpointers).` });
      }
      Sx.push({ svg: draw(T, false, T - 1), side: { "best final state": HMM.st[bestEnd], "probability of best path": d[T - 1][bestEnd].toExponential(2) }, msg: `**Finish.** On the last day the larger score is **${HMM.st[bestEnd]}** (${fvv(d[T - 1][bestEnd])}). That number is the probability of the single most likely hidden sequence together with the observations. Now follow the backpointers backwards.` });
      for (let t = T - 2; t >= 0; t--) Sx.push({ svg: draw(T, false, t), side: { day: t + 1, "best state": HMM.st[path[t]] }, msg: `**Backtrack to day ${t + 1}:** the winning predecessor of ${HMM.st[path[t + 1]]} on day ${t + 2} was **${HMM.st[path[t]]}**.${t === 0 ? ` The decoded weather is **${path.map((q) => HMM.st[q]).join(" → ")}**.` : ""}` });
      return Sx;
    },
  };

  /* ============================================================ MCMC: Metropolis-Hastings random walk */
  const tpdf = (x) => 0.5 * Math.exp(-0.5 * ((x + 2) / 0.7) ** 2) / (0.7 * Math.sqrt(2 * Math.PI)) + 0.5 * Math.exp(-0.5 * ((x - 2) / 1.0) ** 2) / (1.0 * Math.sqrt(2 * Math.PI));
  V.cml1_mcmc = {
    title: "Metropolis-Hastings: a random walk that samples a bumpy target",
    params: [{ id: "sd", label: "proposal step size", min: 0.05, max: 8, step: 0.05, value: 1.0 }],
    slide: "end",
    gen(_, P) {
      const r = rng(31), N = 3000, xs = [-6], acc = [], info = [];
      for (let i = 0; i < N; i++) {
        const x = xs[i], xp = x + P.sd * gauss(r), ratio = tpdf(xp) / tpdf(x), u = r(), ok = u < ratio;
        info.push({ x, xp, ratio, u, ok }); acc.push(ok ? 1 : 0); xs.push(ok ? xp : x);
      }
      const S = [];
      [1, 2, 3, 5, 10, 30, 100, 300, 1000, 3000].forEach((n) => {
        const ch = xs.slice(0, n + 1), samp = xs.slice(1, n + 1), rate = acc.slice(0, n).reduce((a, b) => a + b, 0) / n;
        const lo = -8, hi = 8, nb = 40, bw = (hi - lo) / nb, cnt = new Array(nb).fill(0);
        samp.forEach((v) => { const b = Math.floor((v - lo) / bw); if (b >= 0 && b < nb) cnt[b]++; });
        const mx = Math.max(...cnt.map((c) => c / (n * bw)), 0.4) * 1.1;
        const trace = chart({ W: 330, H: 320, x: [0, Math.max(n, 5)], y: [-8, 8], xl: "iteration", yl: "x", ticks: 4 }, [
          Plot.hline(-2, "pl-dash"), Plot.hline(2, "pl-dash"),
          curve(ch.map((v, i) => [i, v]), "var(--accent)", n > 300 ? 1 : 1.6),
          n <= 10 ? circles(info.slice(0, n).map((t, i) => [i + 1, t.xp, t.ok ? fillDot("var(--green)") : `fill:none;stroke:var(--red);stroke-width:1.8`]), 4.5) : "",
        ]);
        const hist = chart({ W: 330, H: 320, x: [lo, hi], y: [0, mx], xl: "x", yl: "density", ticks: 4, yticks: 4 }, [
          (Sx) => cnt.map((c, i) => c ? `<rect x="${Sx.sx(lo + i * bw).toFixed(1)}" y="${Sx.sy(c / (n * bw)).toFixed(1)}" width="${(Sx.sx(lo + bw) - Sx.sx(lo) - 1).toFixed(1)}" height="${(Sx.sy(0) - Sx.sy(c / (n * bw))).toFixed(1)}" style="fill:${soft("var(--accent)", 45)};stroke:var(--accent);stroke-width:1"/>` : "").join(""),
          Plot.fn(tpdf, "pl-c2", 200),
        ]);
        const m = samp.reduce((a, b) => a + b, 0) / n;
        let msg;
        if (n <= 5) { const t = info[n - 1]; msg = `**Iteration ${n}.** From x = ${fx(t.x)} propose x′ = ${fx(t.xp)}. Ratio $p(x')/p(x)$ = ${t.ratio.toFixed(3)}; draw $u$ = ${t.u.toFixed(3)}. ${t.ok ? "u < ratio, so **accept** (green dot): move." : "u ≥ ratio, so **reject** (red ring): stay at " + fx(t.x) + " and count it again."}`; }
        else {
          const tip = rate < 0.1 ? "Acceptance is very low: the proposals are too bold, they usually land in a low-probability spot and are rejected, so the chain sticks." : rate > 0.8 ? "Acceptance is very high but the steps are tiny, so the chain barely moves: it takes ages to explore (and to leave the starting point)." : rate >= 0.2 && rate <= 0.6 ? "Healthy: it accepts roughly a quarter to a half of its proposals and explores both bumps." : "Usable, but see if a different step size mixes faster.";
          msg = `After **${n}** iterations the histogram of samples (blue) ${n < 300 ? "is still rough and biased by the start at x = −6 (burn-in)" : "hugs the orange target density"}. Acceptance rate **${pct(rate)}**. ${tip}`;
        }
        S.push({ svg: row(trace, hist), side: { iterations: n, "acceptance rate": pct(rate), "sample mean": fx(m), "target mean": "0.00" }, msg });
      });
      return S;
    },
  };

  /* ============================================================ static diagrams */
  D.cml1_kernellift = () => {
    const W = 600, H = 230; let s = "";
    const A = [-1.1, -0.5, 0, 0.6, 1.0], B = [-3.0, -2.4, 2.2, 3.0];
    s += txt(95, 24, "1-D: no single cut separates them", { b: 1, c: "var(--muted)" });
    s += `<line x1="15" y1="150" x2="180" y2="150" style="stroke:var(--muted);stroke-width:1.6"/>`;
    const px = (v) => 97 + v * 26;
    A.forEach((v) => (s += `<circle cx="${px(v)}" cy="150" r="6" style="fill:var(--cls-a);stroke:var(--panel);stroke-width:1.5"/>`));
    B.forEach((v) => (s += `<circle cx="${px(v)}" cy="150" r="6" style="fill:var(--cls-b);stroke:var(--panel);stroke-width:1.5"/>`));
    s += txt(97, 178, "x", { c: "var(--muted)" });
    s += arrow(205, 120, 262, 120, "var(--accent)", 2.4) + txt(233, 106, "φ(x) = (x, x²)", { fs: 12, b: 1, c: "var(--accent)" }) + txt(233, 140, "lift", { fs: 11, c: "var(--muted)" });
    s += txt(430, 24, "2-D: a straight line now works", { b: 1, c: "var(--muted)" });
    const qx = (v) => 430 + v * 40, qy = (v) => 200 - v * 17;
    s += `<line x1="290" y1="200" x2="575" y2="200" style="stroke:var(--muted);stroke-width:1.4"/><line x1="430" y1="200" x2="430" y2="40" style="stroke:var(--muted);stroke-width:1.4"/>`;
    let par = ""; for (let i = 0; i <= 40; i++) { const v = -3.3 + (6.6 * i) / 40; par += (i ? "L" : "M") + qx(v).toFixed(1) + "," + qy(v * v).toFixed(1); }
    s += `<path d="${par}" style="fill:none;stroke:var(--border);stroke-width:1.4;stroke-dasharray:3 3"/>`;
    s += `<line x1="300" y1="${qy(3.2)}" x2="570" y2="${qy(3.2)}" style="stroke:var(--green);stroke-width:2.4"/>` + txt(540, qy(3.2) - 6, "a line separates", { fs: 11, c: "var(--green)" });
    A.forEach((v) => (s += `<circle cx="${qx(v)}" cy="${qy(v * v)}" r="6" style="fill:var(--cls-a);stroke:var(--panel);stroke-width:1.5"/>`));
    B.forEach((v) => (s += `<circle cx="${qx(v)}" cy="${qy(v * v)}" r="6" style="fill:var(--cls-b);stroke:var(--panel);stroke-width:1.5"/>`));
    s += txt(574, 216, "x", { c: "var(--muted)" }) + txt(442, 48, "x²", { c: "var(--muted)" });
    return svgWrap(W, H, s);
  };

  D.cml1_emloop = () => {
    const W = 600, H = 250; let s = "";
    s += box(20, 100, 120, 50, "var(--muted)", 10) + txt(80, 121, "initial guess", { b: 1 }) + txt(80, 138, "μ, Σ, π (random)", { fs: 11, c: "var(--muted)" });
    s += box(210, 30, 190, 62, "var(--accent)", 16) + txt(305, 54, "E-step", { b: 1, fs: 14, c: "var(--accent)" }) + txt(305, 74, "responsibilities r = P(k | x)", { fs: 11.5 });
    s += box(210, 158, 190, 62, "var(--amber)", 16) + txt(305, 182, "M-step", { b: 1, fs: 14, c: "var(--amber)" }) + txt(305, 202, "refit μ, Σ, π with weights r", { fs: 11.5 });
    s += arrow(140, 118, 208, 72, "var(--muted)", 2) + arrow(305, 94, 305, 156, "var(--accent)", 2.4) + `<path d="M402,190 C470,190 470,60 402,60" style="fill:none;stroke:var(--amber);stroke-width:2.4" marker-end="url(#cml1ah)"/>`;
    s += txt(318, 130, "new r gives weights", { a: "start", fs: 11, c: "var(--muted)" }) + txt(478, 128, "new parameters", { a: "start", fs: 11, c: "var(--muted)" });
    s += txt(305, 242, "repeat until the log-likelihood stops rising: it can never go down", { fs: 11.5, c: "var(--muted)" });
    return svgWrap(W, H, s);
  };

  D.cml1_hmmgraph = () => {
    const W = 600, H = 210; let s = "";
    for (let t = 0; t < 4; t++) {
      const x = 90 + t * 140;
      s += `<circle cx="${x}" cy="60" r="26" style="fill:${soft("var(--accent)", 14)};stroke:var(--accent);stroke-width:1.8"/>` + txt(x, 65, "z" + (t + 1), { b: 1, fs: 14 });
      s += arrow(x, 87, x, 128, "var(--muted)", 1.8);
      s += `<circle cx="${x}" cy="152" r="26" style="fill:${soft("var(--amber)", 30)};stroke:var(--amber);stroke-width:2.2"/>` + txt(x, 157, "x" + (t + 1), { b: 1, fs: 14 });
      if (t < 3) s += arrow(x + 27, 60, x + 113, 60, "var(--accent)", 2);
    }
    s += txt(158, 36, "A: transition", { fs: 11, c: "var(--accent)" }) + txt(100, 114, "B: emission", { a: "start", fs: 11, c: "var(--muted)" }) + txt(560, 66, "hidden", { fs: 11, c: "var(--muted)", a: "end" }) + txt(560, 158, "observed", { fs: 11, c: "var(--muted)", a: "end" });
    s += txt(300, 200, "the next hidden state depends only on the current one; each observation depends only on its own hidden state", { fs: 11, c: "var(--muted)" });
    return svgWrap(W, H, s);
  };

  D.cml1_weather = () => {
    const W = 600, H = 250; let s = "";
    s += `<circle cx="150" cy="105" r="42" style="fill:${soft("var(--accent)", 16)};stroke:var(--accent);stroke-width:2"/>` + txt(150, 110, "Rainy", { b: 1, fs: 14 });
    s += `<circle cx="400" cy="105" r="42" style="fill:${soft("var(--amber)", 20)};stroke:var(--amber);stroke-width:2"/>` + txt(400, 110, "Sunny", { b: 1, fs: 14 });
    s += `<path d="M130,65 C100,15 200,15 170,65" style="fill:none;stroke:var(--accent);stroke-width:1.8" marker-end="url(#cml1ah)"/>` + txt(150, 12, "0.7", { fs: 12 });
    s += `<path d="M380,65 C350,15 450,15 420,65" style="fill:none;stroke:var(--amber);stroke-width:1.8" marker-end="url(#cml1ah)"/>` + txt(400, 12, "0.6", { fs: 12 });
    s += `<path d="M190,88 C260,60 300,60 358,88" style="fill:none;stroke:var(--muted);stroke-width:1.8" marker-end="url(#cml1ah)"/>` + txt(274, 62, "0.3", { fs: 12 });
    s += `<path d="M358,125 C300,155 260,155 190,125" style="fill:none;stroke:var(--muted);stroke-width:1.8" marker-end="url(#cml1ah)"/>` + txt(274, 160, "0.4", { fs: 12 });
    s += txt(150, 190, "walk 0.1 · shop 0.4 · clean 0.5", { fs: 12, c: "var(--accent)" }) + txt(400, 190, "walk 0.6 · shop 0.3 · clean 0.1", { fs: 12, c: "var(--amber)" });
    s += txt(300, 225, "arrows = transition probabilities A · lists = emission probabilities B · start: Rainy 0.6, Sunny 0.4", { fs: 11, c: "var(--muted)" });
    return svgWrap(W, H, s);
  };

  D.cml1_ldaplate = () => {
    const W = 600, H = 250; let s = "";
    const node = (x, y, lab, obs) => `<circle cx="${x}" cy="${y}" r="21" style="fill:${obs ? soft("var(--amber)", 40) : "var(--panel)"};stroke:${obs ? "var(--amber)" : "var(--accent)"};stroke-width:2"/>` + txt(x, y + 5, lab, { b: 1, fs: 14 });
    s += `<rect x="100" y="48" width="300" height="130" rx="10" style="fill:none;stroke:var(--muted);stroke-width:1.4"/>` + txt(386, 172, "D documents", { a: "end", fs: 11, c: "var(--muted)" });
    s += `<rect x="225" y="62" width="165" height="100" rx="10" style="fill:none;stroke:var(--muted);stroke-width:1.4"/>` + txt(380, 156, "N words", { a: "end", fs: 11, c: "var(--muted)" });
    s += `<rect x="430" y="62" width="100" height="100" rx="10" style="fill:none;stroke:var(--muted);stroke-width:1.4"/>` + txt(520, 156, "K topics", { a: "end", fs: 11, c: "var(--muted)" });
    s += node(45, 112, "α") + node(145, 112, "θ") + node(265, 112, "z") + node(355, 112, "w", true) + node(480, 112, "φ") + node(570, 112, "β");
    s += arrow(66, 112, 122, 112, "var(--muted)") + arrow(166, 112, 242, 112, "var(--muted)") + arrow(286, 112, 332, 112, "var(--muted)") + arrow(459, 112, 378, 112, "var(--muted)") + arrow(549, 112, 503, 112, "var(--muted)");
    s += txt(145, 82, "topic mix of doc", { fs: 10.5, c: "var(--muted)" }) + txt(265, 80, "topic of word", { fs: 10.5, c: "var(--muted)" }) + txt(355, 80, "word", { fs: 10.5, c: "var(--muted)" }) + txt(480, 82, "words of topic", { fs: 10.5, c: "var(--muted)" });
    s += txt(300, 210, "shaded = observed (the words) · α, β = Dirichlet priors · θ, φ, z are hidden", { fs: 11, c: "var(--muted)" });
    s += txt(300, 232, "story: draw θ ~ Dir(α) per document; for each word draw a topic z ~ θ, then the word w ~ φ_z", { fs: 11, c: "var(--text)" });
    return svgWrap(W, H, s);
  };

  /* ============================================================ glossary */
  Object.assign(window.GLOSSARY, {
    "linear discriminant analysis": "A classifier (and dimensionality reducer) that models each class as a Gaussian with a shared covariance. The decision boundary is a straight line, and projecting onto the discriminant directions best separates the classes.",
    "lda": "Two different algorithms share this abbreviation: Linear Discriminant Analysis (a classifier / projection) and Latent Dirichlet Allocation (a topic model). Context tells you which.",
    "fisher criterion": "The score J(w) = (distance between projected class means)² ÷ (spread inside the classes after projecting). Fisher's LDA picks the direction w that makes it as large as possible.",
    "scatter matrix": "The covariance matrix without the division by n: the sum of outer products (x − mean)(x − mean)ᵀ. Within-class scatter adds this up inside each class; between-class scatter measures how far the class means are from the overall mean.",
    "covariance matrix": "A square table whose entry (i, j) says how much feature i and feature j vary together. The diagonal holds the variances.",
    "qda": "Quadratic Discriminant Analysis: like LDA but every class gets its own covariance matrix, which makes the decision boundary curved (a quadratic).",
    "generative classifier": "A classifier that models how each class generates its inputs, p(x | class), and then uses Bayes' theorem to get p(class | x). LDA, QDA and Naive Bayes are examples.",
    "conjugate prior": "A prior that, combined with a given likelihood, gives a posterior of the same family (a Gaussian prior with a Gaussian likelihood gives a Gaussian posterior), so the update has a closed form.",
    "posterior predictive": "The prediction you get by averaging over all parameter values weighted by their posterior probability, instead of plugging in one best guess. It includes the uncertainty about the parameters.",
    "marginal likelihood": "The probability of the data with the parameters averaged out (integrated over their prior). Also called the evidence; it can be used to compare models and tune hyperparameters.",
    "kernel function": "A function k(x, x′) that returns the dot product of two points after an (often implicit) feature map. It measures similarity and lets algorithms work in huge feature spaces cheaply.",
    "gram matrix": "The n × n table of kernel values K[i, j] = k(xᵢ, xⱼ) between all pairs of training points.",
    "rbf kernel": "The Gaussian kernel k(x, x′) = exp(−‖x − x′‖² / 2ℓ²): similarity decays smoothly with distance. ℓ is the length-scale.",
    "length-scale": "How far apart two inputs can be before a Gaussian process (or RBF kernel) treats them as unrelated. Small = wiggly, large = smooth.",
    "kernel ridge regression": "Ridge regression done in a kernel's feature space. The model is a weighted sum of kernels centred on the training points, fit with one linear solve.",
    "kernel pca": "PCA carried out in a kernel feature space, by eigen-decomposing the centred Gram matrix. It finds curved directions of variation.",
    "gaussian process": "A distribution over functions: any finite set of function values is jointly Gaussian. Conditioning on training data gives a prediction with a mean and an uncertainty at every input.",
    "mixture model": "A model that says each data point came from one of several simple distributions (the components), but we do not know which.",
    "gaussian mixture model": "A mixture of Gaussians: a weighted sum of several bell-shaped components, each with its own mean and covariance. Used for soft clustering and density estimation.",
    "gmm": "Gaussian Mixture Model: a weighted sum of Gaussian components, usually fitted with the EM algorithm.",
    "responsibility": "In a mixture model, the posterior probability that a given component generated a given point. Each point's responsibilities sum to 1.",
    "latent variable": "A variable that is part of the model but never observed, such as which cluster a point came from.",
    "log-likelihood": "The logarithm of the probability the model assigns to the data. Higher is better; logs turn the product over examples into a sum.",
    "em algorithm": "Expectation-Maximisation: fits models with hidden variables by alternating an E-step (fill in the hidden variables with probabilities) and an M-step (refit the parameters). The likelihood never decreases.",
    "e-step": "The expectation step of EM: compute the posterior probabilities of the hidden variables under the current parameters.",
    "m-step": "The maximisation step of EM: re-estimate the parameters to maximise the expected complete-data log-likelihood, using the E-step's weights.",
    "jensen's inequality": "For a concave function such as log, the log of an average is at least the average of the logs. It is what makes EM's lower bound valid.",
    "elbo": "Evidence Lower Bound: a quantity that is always at most the log-likelihood and that EM (and variational inference) pushes up.",
    "markov chain": "A sequence of random states in which the next state depends only on the current one, not on how you got there.",
    "markov property": "The future depends on the past only through the present state.",
    "hidden markov model": "A Markov chain of hidden states that we cannot see, each emitting an observation we can see. Used for speech, tagging and sequences in general.",
    "hmm": "Hidden Markov Model: hidden states that follow a Markov chain, each producing an observation.",
    "transition probability": "In a Markov chain or HMM, the probability of moving from one state to another in one step.",
    "emission probability": "In an HMM, the probability that a hidden state produces a particular observation.",
    "forward algorithm": "Dynamic programming that computes the probability of an observation sequence under an HMM by summing over all hidden paths, in O(T·K²) time.",
    "viterbi algorithm": "Dynamic programming that finds the single most likely hidden state sequence in an HMM: like the forward algorithm but with max instead of sum, plus backpointers.",
    "baum-welch": "The EM algorithm for HMMs: it uses the forward-backward algorithm as the E-step to learn the transition and emission probabilities from unlabelled sequences.",
    "dynamic programming": "Solving a problem by storing and reusing answers to overlapping sub-problems, so an exponential search becomes a polynomial one.",
    "topic model": "A model that explains a collection of documents as mixtures of a few hidden topics, where each topic is a distribution over words.",
    "dirichlet distribution": "A distribution over probability vectors (non-negative numbers that sum to 1). It is the natural prior for the parameters of categorical choices such as topic mixes.",
    "latent dirichlet allocation": "A topic model: each document has a Dirichlet-distributed mix of topics, each topic a Dirichlet-distributed mix of words, and each word is generated by first picking a topic.",
    "gibbs sampling": "An MCMC method that resamples one variable at a time from its conditional distribution given all the others. Every proposal is accepted.",
    "mcmc": "Markov chain Monte Carlo: build a Markov chain whose long-run distribution is the one you want, then use its states as (correlated) samples.",
    "metropolis-hastings": "An MCMC algorithm: propose a move, accept it with probability min(1, p(new)/p(old)) (for a symmetric proposal), otherwise stay put. It only needs p up to a constant.",
    "burn-in": "The first samples of an MCMC run, which still remember the starting point and are usually thrown away.",
    "acceptance rate": "The fraction of proposed moves that a Metropolis-Hastings sampler accepts. Around 20 to 50 percent is typical for a good step size.",
    "stationary distribution": "A distribution over states that a Markov chain leaves unchanged. MCMC is designed so that this is the target distribution.",
    "proposal distribution": "In Metropolis-Hastings, the distribution used to suggest the next candidate state given the current one.",
  });
})();
