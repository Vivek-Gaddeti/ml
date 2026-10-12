/* Classical ML, part 2 (factorization, manifolds, other classics): visualizers, diagrams and glossary terms.
   Loaded on every doc page and on Practice, after assets/viz.js. Every name starts with cml2_. */
(function () {
  const V = Viz.defs, Plot = Viz.Plot;

  /* ============================================================ private helpers */
  const rng = (seed) => { let a = (seed >>> 0) || 1; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  const gauss = (r) => { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const COL = ["var(--accent)", "var(--amber)", "var(--green)", "var(--red)", "var(--purple)"];
  const col = (j) => COL[((j % COL.length) + COL.length) % COL.length];
  const soft = (c, p) => `color-mix(in srgb, ${c} ${p || 18}%, transparent)`;
  const fx = (v, n) => (+v).toFixed(n == null ? 2 : n);
  const SQ = { W: 360, H: 330 };
  const row = (...svgs) => `<div style="display:flex;flex-wrap:wrap;gap:10px 14px;justify-content:center;align-items:center">${svgs.join("")}</div>`;
  const colm = (...svgs) => `<div style="display:flex;flex-direction:column;gap:6px;align-items:center">${svgs.join("")}</div>`;
  const D2R = Math.PI / 180;

  function chart(o, layers) { return Plot.chart(o, layers); }
  const circles = (pts, r0) => (S) => pts.map((p) => `<circle cx="${S.sx(p[0]).toFixed(1)}" cy="${S.sy(p[1]).toFixed(1)}" r="${p[3] || r0 || 4.5}" style="${p[2]}"/>`).join("");
  const fillDot = (c, extra) => `fill:${c};stroke:var(--panel);stroke-width:1.2${extra || ""}`;
  const pathOf = (S, pts) => pts.map(([x, y], i) => (i ? "L" : "M") + S.sx(x).toFixed(1) + "," + S.sy(y).toFixed(1)).join("");
  const curve = (pts, c, w, dash) => (S) => `<path d="${pathOf(S, pts)}" style="fill:none;stroke:${c};stroke-width:${w || 2.2};stroke-linejoin:round;stroke-linecap:round${dash ? ";stroke-dasharray:" + dash : ""}"/>`;
  const band = (lo, hi, c, p) => (S) => `<path d="${pathOf(S, lo)}${hi.slice().reverse().map(([x, y]) => "L" + S.sx(x).toFixed(1) + "," + S.sy(y).toFixed(1)).join("")}Z" style="fill:${soft(c, p || 20)};stroke:none"/>`;
  const segPx = (S, x1, y1, x2, y2, style) => `<line x1="${S.sx(x1).toFixed(1)}" y1="${S.sy(y1).toFixed(1)}" x2="${S.sx(x2).toFixed(1)}" y2="${S.sy(y2).toFixed(1)}" style="${style}"/>`;
  const txt = (x, y, s, o) => { o = o || {}; return `<text x="${(+x).toFixed(1)}" y="${(+y).toFixed(1)}" text-anchor="${o.a || "middle"}" style="font-size:${o.fs || 12}px;fill:${o.c || "var(--text)"}${o.b ? ";font-weight:700" : ""}">${esc(s)}</text>`; };
  const svgWrap = (W, H, inner) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img"><defs><marker id="cml2ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker></defs>${inner}</svg>`;
  const box = (x, y, w, h, c, p, r) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r == null ? 7 : r}" style="fill:${soft(c, p || 14)};stroke:${c};stroke-width:1.5"/>`;
  const arrow = (x1, y1, x2, y2, c, w, dash) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" style="stroke:${c || "var(--muted)"};stroke-width:${w || 1.8}${dash ? ";stroke-dasharray:" + dash : ""}" marker-end="url(#cml2ah)"/>`;
  const sum = (a) => a.reduce((s, v) => s + v, 0);

  /* ---- tiny linear algebra ---- */
  function solveSmall(A, b) {                         // Gaussian elimination with partial pivoting (small systems)
    const n = b.length, M = A.map((r, i) => r.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let k = i + 1; k < n; k++) s -= M[i][k] * x[k]; x[i] = s / M[i][i]; }
    return x;
  }
  function jacobiEig(A0) {                            // all eigenpairs of a symmetric matrix, ascending eigenvalues
    const n = A0.length, A = A0.map((r) => r.slice()), Vm = A.map((_, i) => A.map((__, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-18) break;
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-14) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const kp = A[k][p], kq = A[k][q]; A[k][p] = c * kp - s * kq; A[k][q] = s * kp + c * kq; }
        for (let k = 0; k < n; k++) { const pk = A[p][k], qk = A[q][k]; A[p][k] = c * pk - s * qk; A[q][k] = s * pk + c * qk; }
        for (let k = 0; k < n; k++) { const kp = Vm[k][p], kq = Vm[k][q]; Vm[k][p] = c * kp - s * kq; Vm[k][q] = s * kp + c * kq; }
      }
    }
    const idx = A.map((_, i) => i).sort((a, b) => A[a][a] - A[b][b]);
    return { vals: idx.map((i) => A[i][i]), vecs: idx.map((i) => Vm.map((r) => r[i])) };   // vecs[k] = k-th eigenvector
  }

  /* ============================================================ ICA: cocktail party */
  const ICA = (() => {
    const N = 240, t = [], s1 = [], s2 = [];
    for (let i = 0; i < N; i++) { const u = i / N; t.push(u); s1.push(Math.sin(2 * Math.PI * 3 * u) >= 0 ? 1 : -1); s2.push((((5 * u) % 1) * 2) - 1); }
    const A = [[1, 0.55], [0.4, 1]], x1 = [], x2 = [];
    for (let i = 0; i < N; i++) { x1.push(A[0][0] * s1[i] + A[0][1] * s2[i]); x2.push(A[1][0] * s1[i] + A[1][1] * s2[i]); }
    const m1 = sum(x1) / N, m2 = sum(x2) / N, c1 = x1.map((v) => v - m1), c2 = x2.map((v) => v - m2);
    const a = sum(c1.map((v) => v * v)) / N, b = sum(c1.map((v, i) => v * c2[i])) / N, d = sum(c2.map((v) => v * v)) / N;
    const tr = a + d, det = a * d - b * b, l1 = tr / 2 + Math.sqrt(tr * tr / 4 - det), l2 = tr / 2 - Math.sqrt(tr * tr / 4 - det);
    const e1 = Math.abs(b) > 1e-12 ? [l1 - d, b] : [1, 0], n1 = Math.hypot(e1[0], e1[1]), E1 = [e1[0] / n1, e1[1] / n1], E2 = [-E1[1], E1[0]];
    const z1 = c1.map((v, i) => (v * E1[0] + c2[i] * E1[1]) / Math.sqrt(l1)), z2 = c1.map((v, i) => (v * E2[0] + c2[i] * E2[1]) / Math.sqrt(l2));
    const proj = (th) => z1.map((v, i) => Math.cos(th * D2R) * v + Math.sin(th * D2R) * z2[i]);
    const kurt = (y) => sum(y.map((v) => v ** 4)) / N - 3;
    const curveK = []; for (let th = 0; th <= 180; th += 2) curveK.push([th, Math.abs(kurt(proj(th)))]);
    let best = curveK.reduce((p, q) => (q[1] > p[1] ? q : p));
    return { N, t, s1, s2, x1, x2, z1, z2, proj, kurt, curveK, best: best[0] };
  })();
  V.cml2_ica = {
    title: "ICA: rotate the whitened data until the signals are as non-Gaussian as possible",
    params: [{ id: "ang", label: "rotation angle θ (degrees)", min: 0, max: 180, step: 2, value: 40 }],
    gen(_, P) {
      const th = P.ang, y = ICA.proj(th), y2 = ICA.proj(th + 90), k1 = ICA.kurt(y), k2 = ICA.kurt(y2);
      const scatter = ICA.z1.map((v, i) => [v, ICA.z2[i], fillDot("var(--accent)", ";opacity:.75"), 2.6]);
      const left = chart({ ...SQ, x: [-2.6, 2.6], y: [-2.6, 2.6], xl: "whitened mic 1", yl: "whitened mic 2", ticks: 4 }, [
        circles(scatter, 2.6),
        (S) => segPx(S, -3.3 * Math.cos(th * D2R), -3.3 * Math.sin(th * D2R), 3.3 * Math.cos(th * D2R), 3.3 * Math.sin(th * D2R), "stroke:var(--amber);stroke-width:2.6"),
        (S) => segPx(S, -3.3 * Math.cos((th + 90) * D2R), -3.3 * Math.sin((th + 90) * D2R), 3.3 * Math.cos((th + 90) * D2R), 3.3 * Math.sin((th + 90) * D2R), "stroke:var(--purple);stroke-width:2.6"),
      ]);
      const kc = chart({ W: 360, H: 150, x: [0, 180], y: [0, 2.2], xl: "θ (degrees)", yl: "|excess kurtosis|", ticks: 4, yticks: 3 }, [
        curve(ICA.curveK, "var(--muted)", 2), Plot.dot(th, Math.min(2.2, Math.abs(k1)), "pl-c2", 5.5), Plot.vline(ICA.best, "pl-dash"),
      ]);
      const tr = (arr, c, lab) => chart({ W: 360, H: 78, x: [0, 1], y: [-3, 3], yl: lab, ticks: 4, yticks: 2, pad: { t: 6, b: 18 } }, [curve(ICA.t.map((u, i) => [u, Math.max(-3, Math.min(3, arr[i]))]), c, 1.8)]);
      const right = colm(kc, tr(y, "var(--amber)", "axis 1"), tr(y2, "var(--purple)", "axis 2"));
      const dd = (((th - ICA.best) % 90) + 90) % 90, near = dd < 4 || dd > 86;
      return [{
        svg: row(left, right),
        side: { "θ": th + "°", "kurtosis, axis 1": fx(k1, 2), "kurtosis, axis 2": fx(k2, 2), "best θ": ICA.best + "° (and +90°)" },
        msg: near
          ? `**Unmixed.** The two axes now line up with the sides of the (rotated) square of the whitened cloud. Axis 1 recovers a clean sawtooth or square wave, axis 2 the other one, and the |kurtosis| curve is at its peak. A Gaussian would have kurtosis 0; these sources are flat-topped, so their kurtosis is far below zero.`
          : `The data are already **whitened**: uncorrelated with unit variance, so only a rotation is left to find. The two crossing lines are candidate unmixing axes, and the traces on the right are the data projected on them. At this angle each trace is still a blend of the sawtooth and the square wave, so it looks more Gaussian (kurtosis closer to 0). Slide θ towards the peaks of the curve (about ${ICA.best}° and ${ICA.best + 90}° modulo 180).`,
      }];
    },
  };

  /* ============================================================ NMF: parts-based */
  const NMFD = (() => {
    const r = rng(5), n = 8, p = 16, parts = [[0, 4, 8, 12], [0, 1, 2, 3], [5, 6, 9, 10]];   // left column, top row, centre block (4x4 images)
    const W0 = []; for (let i = 0; i < n; i++) W0.push(parts.map((_, k) => ((i + k) % 3 === 0 || r() < 0.4 ? 0.3 + 0.7 * r() : 0)));
    W0.forEach((w) => { if (!w.some((v) => v > 0)) w[0] = 0.8; });
    const Vm = W0.map((w) => { const v = new Array(p).fill(0); w.forEach((c, k) => parts[k].forEach((px) => { v[px] += c; })); return v; });
    return { n, p, Vm };
  })();
  function nmfFit(Vm, k, iters) {                     // Lee-Seung multiplicative updates, deterministic start
    const n = Vm.length, p = Vm[0].length, r = rng(3 + k), e = 1e-9;
    let W = Array.from({ length: n }, () => Array.from({ length: k }, () => 0.2 + r())), H = Array.from({ length: k }, () => Array.from({ length: p }, () => 0.2 + r()));
    const mul = (A, B) => A.map((row) => B[0].map((_, j) => sum(row.map((v, t) => v * B[t][j]))));
    for (let it = 0; it < iters; it++) {
      const WH = mul(W, H);
      for (let a = 0; a < k; a++) for (let j = 0; j < p; j++) { let nu = 0, de = 0; for (let i = 0; i < n; i++) { nu += W[i][a] * Vm[i][j]; de += W[i][a] * WH[i][j]; } H[a][j] *= nu / (de + e); }
      const WH2 = mul(W, H);
      for (let i = 0; i < n; i++) for (let a = 0; a < k; a++) { let nu = 0, de = 0; for (let j = 0; j < p; j++) { nu += Vm[i][j] * H[a][j]; de += WH2[i][j] * H[a][j]; } W[i][a] *= nu / (de + e); }
    }
    const R = mul(W, H); let err = 0; Vm.forEach((rw, i) => rw.forEach((v, j) => { err += (v - R[i][j]) ** 2; }));
    return { W, H, R, err: Math.sqrt(err) };
  }
  const NMF_ERR = [1, 2, 3, 4, 5].map((k) => nmfFit(NMFD.Vm, k, 400).err);
  const NMF_FIT = {};
  const tile = (img, x0, y0, c, vmax) => {            // a 4x4 image as small squares
    let s = "";
    for (let i = 0; i < 16; i++) { const v = Math.max(0, Math.min(1, img[i] / vmax)), cx = x0 + (i % 4) * 10, cy = y0 + Math.floor(i / 4) * 10;
      s += `<rect x="${cx}" y="${cy}" width="9.2" height="9.2" style="fill:color-mix(in srgb, ${c} ${(v * 100).toFixed(0)}%, var(--panel-2));stroke:var(--border);stroke-width:.5"/>`; }
    return s;
  };
  V.cml2_nmf = {
    title: "NMF: images as sums of non-negative parts",
    params: [{ id: "k", label: "rank k (number of parts)", min: 1, max: 5, step: 1, value: 3 }],
    gen(_, P) {
      const k = P.k, F = NMF_FIT[k] || (NMF_FIT[k] = nmfFit(NMFD.Vm, k, 400)), vmax = Math.max(...NMFD.Vm.map((r) => Math.max(...r)));
      let s = "", W = 640, H = 330;
      s += txt(6, 14, "data V: 8 images of 4×4 pixels", { a: "start", c: "var(--muted)", fs: 11.5 });
      NMFD.Vm.forEach((img, i) => { s += tile(img, 6 + i * 52, 22, "var(--accent)", vmax); });
      s += txt(6, 92, "reconstruction W·H with " + k + " part" + (k > 1 ? "s" : ""), { a: "start", c: "var(--muted)", fs: 11.5 });
      F.R.forEach((img, i) => { s += tile(img, 6 + i * 52, 100, "var(--amber)", vmax); });
      s += txt(6, 170, "the learned parts (rows of H), each pixel ≥ 0", { a: "start", c: "var(--muted)", fs: 11.5 });
      const hmax = Math.max(...F.H.map((r) => Math.max(...r)));
      F.H.forEach((img, a) => { s += tile(img, 6 + a * 62, 178, col(a + 2), hmax); s += txt(26 + a * 62, 230, "part " + (a + 1), { fs: 11, c: "var(--muted)" }); });
      s += txt(6, 252, "how much of each part image 1 uses (row 1 of W)", { a: "start", c: "var(--muted)", fs: 11.5 });
      F.W[0].forEach((w, a) => { s += `<rect x="${6 + a * 62}" y="${258}" width="52" height="14" style="fill:${soft(col(a + 2), 14)};stroke:${col(a + 2)}"/><rect x="${6 + a * 62}" y="258" width="${Math.min(52, w / 1.6 * 52).toFixed(1)}" height="14" style="fill:${col(a + 2)}"/>` + txt(32 + a * 62, 290, fx(w, 2), { fs: 11 }); });
      const bars = chart({ W: 250, H: 250, x: [0.4, 5.6], y: [0, Math.max(...NMF_ERR) * 1.1], xl: "rank k", yl: "‖V − WH‖", ticks: 5, yticks: 4 },
        [(S) => NMF_ERR.map((e, i) => `<rect x="${S.sx(i + 1 - 0.32).toFixed(1)}" y="${S.sy(e).toFixed(1)}" width="${(S.sx(i + 1 + 0.32) - S.sx(i + 1 - 0.32)).toFixed(1)}" height="${(S.sy(0) - S.sy(e)).toFixed(1)}" style="fill:${i + 1 === k ? "var(--amber)" : soft("var(--accent)", 45)}"/>`).join("")]);
      return [{
        svg: row(svgWrap(W - 150, H, s), bars),
        side: { rank: k, "error": fx(F.err, 3), "pixels in H": "all ≥ 0" },
        msg: k < 3 ? `With only **${k}** part${k > 1 ? "s" : ""} the images cannot be rebuilt: each part has to cover several of the true building blocks at once, so the reconstruction is blurry and the error is large.`
          : k === 3 ? `**Rank 3 is the true rank** of this toy data (a left column, a top row and a centre block). The three learned parts look like those blocks, and the reconstruction is almost exact (error ${fx(F.err, 3)}). Because nothing may be negative, parts can only be **added**, never cancelled, so NMF is pushed towards pieces you can recognise.`
          : `Extra parts beyond 3 barely lower the error: there are no more real building blocks in the data. Extra parts tend to split or duplicate existing ones, which is why you choose the rank by the elbow of this curve.`,
      }];
    },
  };

  /* ============================================================ ALS: filling a rating grid */
  const ALSD = (() => {
    const r = rng(8), nu = 6, ni = 6, K = 2, U = [], It = [];
    for (let u = 0; u < nu; u++) U.push([gauss(r) * 0.3 + (u < 3 ? 1 : -1) * 1.1, gauss(r) * 0.6]);
    for (let i = 0; i < ni; i++) It.push([gauss(r) * 0.3 + (i < 3 ? 1 : -1) * 1.1, gauss(r) * 0.6]);
    const T = [], mask = [];
    for (let u = 0; u < nu; u++) { T.push([]); mask.push([]); for (let i = 0; i < ni; i++) { T[u].push(Math.max(1, Math.min(5, Math.round(3 + 1.0 * (U[u][0] * It[i][0] + U[u][1] * It[i][1]) + 0.25 * gauss(r))))); mask[u].push(r() > 0.33); } }
    return { nu, ni, T, mask };
  })();
  function alsRun(k, lam, rounds) {
    const { nu, ni, T, mask } = ALSD, r = rng(21), frames = [];
    let Pu = Array.from({ length: nu }, () => Array.from({ length: k }, () => 0.1 * gauss(r))), Q = Array.from({ length: ni }, () => Array.from({ length: k }, () => 0.5 * gauss(r)));
    const pred = () => Pu.map((p) => Q.map((q) => 3 + sum(p.map((v, a) => v * q[a]))));
    const rmse = (obs) => { let s = 0, c = 0; for (let u = 0; u < nu; u++) for (let i = 0; i < ni; i++) if (mask[u][i] === obs) { s += (T[u][i] - Math.max(1, Math.min(5, pred()[u][i]))) ** 2; c++; } return Math.sqrt(s / c); };
    const snap = (kind, i0) => frames.push({ kind, it: i0, P: Pu.map((p) => p.slice()), Q: Q.map((q) => q.slice()), pr: pred(), tr: rmse(true), te: rmse(false) });
    snap("init", 0);
    for (let it = 1; it <= rounds; it++) {
      for (let u = 0; u < nu; u++) { const A = Array.from({ length: k }, (_, a) => Array.from({ length: k }, (__, b) => (a === b ? lam : 0))), bb = new Array(k).fill(0);
        for (let i = 0; i < ni; i++) if (mask[u][i]) { for (let a = 0; a < k; a++) { bb[a] += (T[u][i] - 3) * Q[i][a]; for (let b = 0; b < k; b++) A[a][b] += Q[i][a] * Q[i][b]; } }
        Pu[u] = solveSmall(A, bb); }
      snap("user", it);
      for (let i = 0; i < ni; i++) { const A = Array.from({ length: k }, (_, a) => Array.from({ length: k }, (__, b) => (a === b ? lam : 0))), bb = new Array(k).fill(0);
        for (let u = 0; u < nu; u++) if (mask[u][i]) { for (let a = 0; a < k; a++) { bb[a] += (T[u][i] - 3) * Pu[u][a]; for (let b = 0; b < k; b++) A[a][b] += Pu[u][a] * Pu[u][b]; } }
        Q[i] = solveSmall(A, bb); }
      snap("item", it);
    }
    return frames;
  }
  V.cml2_als = {
    title: "ALS: alternately solve for the user factors and the item factors",
    params: [{ id: "k", label: "latent factors k", min: 1, max: 3, step: 1, value: 2 }, { id: "lam", label: "regularization λ", min: 0.1, max: 4, step: 0.1, value: 1 }],
    gen(_, P) {
      const F = alsRun(P.k, P.lam, 5), { nu, ni, T, mask } = ALSD, S = [], cw = 58, ch = 36, x0 = 40, y0 = 30;
      F.forEach((f) => {
        let s = "";
        for (let i = 0; i < ni; i++) s += txt(x0 + i * cw + cw / 2, y0 - 10, "item " + (i + 1), { fs: 11, c: "var(--muted)" });
        for (let u = 0; u < nu; u++) {
          s += txt(x0 - 6, y0 + u * ch + ch / 2 + 4, "u" + (u + 1), { a: "end", fs: 11, c: "var(--muted)" });
          for (let i = 0; i < ni; i++) {
            const x = x0 + i * cw, y = y0 + u * ch, obs = mask[u][i], v = Math.max(1, Math.min(5, f.pr[u][i]));
            s += `<rect x="${x + 1}" y="${y + 1}" width="${cw - 2}" height="${ch - 2}" rx="4" style="fill:${obs ? soft("var(--accent)", 10 + 8 * T[u][i]) : soft("var(--amber)", f.kind === "init" ? 4 : 8 + 7 * v)};stroke:${obs ? "var(--accent)" : "var(--amber)"};stroke-width:${obs ? 1 : 1.2};${obs ? "" : "stroke-dasharray:4 3"}"/>`;
            s += txt(x + cw / 2, y + ch / 2 + 4.5, obs ? String(T[u][i]) : f.kind === "init" ? "?" : fx(v, 1), { b: obs, fs: obs ? 14 : 13, c: obs ? "var(--text)" : "var(--amber)" });
            if (!obs) s += txt(x + cw - 5, y + 11, "(" + T[u][i] + ")", { a: "end", fs: 9, c: "var(--muted)" });
          }
        }
        s += txt(x0, y0 + nu * ch + 18, "solid: observed rating · dashed: prediction (true value)", { a: "start", fs: 11, c: "var(--muted)" });
        const grid = svgWrap(x0 + ni * cw + 8, y0 + nu * ch + 28, s);
        const lat = P.k >= 2 ? chart({ W: 270, H: 260, x: [-2.5, 2.5], y: [-2.5, 2.5], xl: "factor 1", yl: "factor 2", ticks: 4, yticks: 4 }, [
          circles(f.P.map((p, u) => [p[0], p[1], fillDot("var(--accent)"), 6])), circles(f.Q.map((q) => [q[0], q[1], fillDot("var(--amber)"), 5])),
          (S2) => f.P.map((p, u) => `<text x="${S2.sx(p[0]) + 7}" y="${S2.sy(p[1]) - 6}" style="font-size:10px;fill:var(--accent)">u${u + 1}</text>`).join("") + f.Q.map((q, i) => `<text x="${S2.sx(q[0]) + 7}" y="${S2.sy(q[1]) + 12}" style="font-size:10px;fill:var(--amber)">i${i + 1}</text>`).join(""),
        ]) : "";
        const msg = f.kind === "init" ? `**Start.** Item factors get small random values, user factors are zero, so every prediction is the global mean 3. The aim is to find vectors $\\mathbf p_u,\\mathbf q_i\\in\\mathbb R^{${P.k}}$ so that $3+\\mathbf p_u^\\top\\mathbf q_i$ matches every *observed* rating.`
          : f.kind === "user" ? `**Round ${f.it}, user step.** Freeze all item vectors. Each user's best vector is now an ordinary **ridge regression** on just the items that user rated: $\\mathbf p_u=(Q_u^\\top Q_u+\\lambda I)^{-1}Q_u^\\top\\mathbf r_u$. The users are independent of each other, so all of them can be solved at once. Training RMSE ${fx(f.tr, 3)}.`
          : `**Round ${f.it}, item step.** Now freeze the users and solve the same kind of ridge regression for every item, using only the users who rated it. The dashed cells (never seen during training) are predictions; compare them with the true value in brackets. Training RMSE ${fx(f.tr, 3)}, hidden-rating RMSE ${fx(f.te, 3)}.`;
        S.push({ svg: row(grid, lat), side: { step: f.kind === "init" ? "start" : f.kind + " " + f.it, "train RMSE": fx(f.tr, 3), "RMSE on hidden cells": f.kind === "init" ? "-" : fx(f.te, 3) }, msg });
      });
      return S;
    },
  };

  /* ============================================================ t-SNE */
  function tsneData(nc) {
    const r = rng(14), cen = [[0, 0, 0, 0], [5, 4, -3, 1], [-4, 5, 4, -3]], X = [], lab = [];
    for (let c = 0; c < nc; c++) for (let i = 0; i < 16; i++) { X.push(cen[c].map((m) => m + gauss(r) * 1.3)); lab.push(c); }
    return { X, lab };
  }
  function tsneRun(X, perp, frames) {
    const n = X.length, r = rng(3), D = X.map((a) => X.map((b) => sum(a.map((v, k) => (v - b[k]) ** 2)))), P = X.map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {                       // binary search for sigma_i so that the entropy matches log(perplexity)
      let lo = 1e-9, hi = 1e9, beta = 1, row_ = new Array(n).fill(0);
      for (let it = 0; it < 60; it++) {
        let Z = 0; for (let j = 0; j < n; j++) { row_[j] = j === i ? 0 : Math.exp(-D[i][j] * beta); Z += row_[j]; }
        Z = Z || 1e-12; let H = 0; for (let j = 0; j < n; j++) if (row_[j] > 0) { const p = row_[j] / Z; H -= p * Math.log(p); }
        if (H > Math.log(perp)) { lo = beta; beta = hi > 1e8 ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = (beta + lo) / 2; }
      }
      let Z = 0; for (let j = 0; j < n; j++) { row_[j] = j === i ? 0 : Math.exp(-D[i][j] * beta); Z += row_[j]; }
      for (let j = 0; j < n; j++) P[i][j] = row_[j] / (Z || 1);
    }
    const Ps = P.map((row2, i) => row2.map((v, j) => (v + P[j][i]) / (2 * n)));
    let Y = X.map(() => [gauss(r) * 1e-2, gauss(r) * 1e-2]), vel = Y.map(() => [0, 0]);
    const out = {}, want = new Set(frames);
    const snap = (it, kl) => { out[it] = { Y: Y.map((y) => y.slice()), kl }; };
    snap(0, NaN);
    for (let it = 1; it <= Math.max(...frames); it++) {
      const ex = it <= 60 ? 4 : 1; let Zq = 0; const num = Y.map((a) => Y.map((b) => 1 / (1 + (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)));
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) Zq += num[i][j];
      const g = Y.map(() => [0, 0]); let kl = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) {
        const q = num[i][j] / Zq, c = (ex * Ps[i][j] - q) * num[i][j];
        g[i][0] += 4 * c * (Y[i][0] - Y[j][0]); g[i][1] += 4 * c * (Y[i][1] - Y[j][1]);
        if (Ps[i][j] > 0) kl += Ps[i][j] * Math.log(Ps[i][j] / Math.max(q, 1e-12));
      }
      const mom = it < 60 ? 0.5 : 0.8;
      for (let i = 0; i < n; i++) for (let a = 0; a < 2; a++) { vel[i][a] = mom * vel[i][a] - 40 * g[i][a]; Y[i][a] += vel[i][a]; }
      const my = [sum(Y.map((y) => y[0])) / n, sum(Y.map((y) => y[1])) / n]; Y.forEach((y) => { y[0] -= my[0]; y[1] -= my[1]; });
      if (want.has(it)) snap(it, kl);
    }
    return out;
  }
  V.cml2_tsne = {
    title: "t-SNE: points slide in 2-D until neighbours in the original space are neighbours here",
    params: [{ id: "perp", label: "perplexity", min: 2, max: 30, step: 1, value: 10 }, { id: "nc", label: "clusters in the data", min: 2, max: 3, step: 1, value: 3 }],
    gen(_, P) {
      const { X, lab } = tsneData(P.nc), frames = [0, 10, 30, 60, 100, 200, 300], F = tsneRun(X, Math.min(P.perp, X.length - 2), frames), S = [];
      frames.forEach((it) => {
        const f = F[it], ext = Math.max(1e-6, ...f.Y.map((y) => Math.max(Math.abs(y[0]), Math.abs(y[1])))) * 1.15;
        const svg = chart({ W: 500, H: 330, x: [-ext, ext], y: [-ext, ext], xl: "t-SNE 1", yl: "t-SNE 2", ticks: 4, yticks: 4 }, [circles(f.Y.map((y, i) => [y[0], y[1], fillDot(col(lab[i])), 5.2]))]);
        const msg = it === 0 ? `**Start.** All ${X.length} points sit in a tiny random blob. Before this, t-SNE has turned the original distances into probabilities $p_{ij}$ ("how likely are $i$ and $j$ to be neighbours?") using a Gaussian whose width is tuned per point so that each has about **perplexity ${P.perp}** effective neighbours.`
          : it <= 60 ? `**Iteration ${it}, early exaggeration.** For the first 60 iterations the $p_{ij}$ are multiplied by 4, which makes the attraction between true neighbours so strong that clusters can slide around freely and pack tightly. KL divergence (cost) ${fx(f.kl, 2)}.`
          : `**Iteration ${it}.** Gradient descent keeps moving points: neighbours attract, and the heavy tails of the Student-t kernel let non-neighbours repel at a distance. Cost (KL divergence between $P$ and $Q$) ${fx(f.kl, 3)}.${it === 300 ? " Try a very small perplexity (2 to 3: many tiny islands) and a large one (30: clusters pull together)." : ""}`;
        S.push({ svg, side: { iteration: it, perplexity: P.perp, "KL cost": it === 0 ? "-" : fx(f.kl, 3) }, msg });
      });
      return S;
    },
  };

  /* ============================================================ Spectral clustering on two moons */
  const MOONS = (() => {
    const r = rng(6), n = 50, X = [], truth = [];
    for (let i = 0; i < n; i++) { const a = Math.PI * i / (n - 1); X.push([Math.cos(a) + 0.07 * gauss(r), Math.sin(a) + 0.07 * gauss(r)]); truth.push(0); }
    for (let i = 0; i < n; i++) { const a = Math.PI * i / (n - 1); X.push([1 - Math.cos(a) + 0.07 * gauss(r), 0.5 - Math.sin(a) + 0.07 * gauss(r)]); truth.push(1); }
    return { X, truth };
  })();
  function kmeans2(X) {
    let i0 = 0, j0 = 0, best = -1;
    X.forEach((a, i) => X.forEach((b, j) => { const d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2; if (d > best) { best = d; i0 = i; j0 = j; } }));
    let C = [X[i0].slice(), X[j0].slice()], lab = X.map(() => 0);
    for (let it = 0; it < 30; it++) {
      lab = X.map((p) => ((p[0] - C[0][0]) ** 2 + (p[1] - C[0][1]) ** 2 <= (p[0] - C[1][0]) ** 2 + (p[1] - C[1][1]) ** 2 ? 0 : 1));
      C = [0, 1].map((c) => { const q = X.filter((_, i) => lab[i] === c); return q.length ? [sum(q.map((v) => v[0])) / q.length, sum(q.map((v) => v[1])) / q.length] : C[c]; });
    }
    return { lab, C };
  }
  function spectral(X, kn) {
    const n = X.length, d2 = X.map((a) => X.map((b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)), A = X.map(() => new Array(n).fill(0)), edges = [];
    const med = []; for (let i = 0; i < n; i++) { const o = d2[i].map((v, j) => [v, j]).sort((a, b) => a[0] - b[0]).slice(1, kn + 1); o.forEach(([v, j]) => { med.push(v); }); }
    med.sort((a, b) => a - b); const sig2 = med[Math.floor(med.length / 2)] || 0.05;
    for (let i = 0; i < n; i++) { d2[i].map((v, j) => [v, j]).sort((a, b) => a[0] - b[0]).slice(1, kn + 1).forEach(([v, j]) => { const w = Math.exp(-v / sig2); A[i][j] = w; A[j][i] = w; }); }
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (A[i][j] > 0) edges.push([i, j]);
    const deg = A.map((r) => sum(r)), L = A.map((r, i) => r.map((v, j) => (i === j ? 1 : 0) - v / Math.sqrt(deg[i] * deg[j])));
    const E = jacobiEig(L), emb = X.map((_, i) => { const a = E.vecs[0][i], b = E.vecs[1][i], nr = Math.hypot(a, b) || 1; return [a / nr, b / nr]; });   // Ng-Jordan-Weiss: rows of the 2 smallest eigenvectors, scaled to length 1
    const km = kmeans2(emb);
    return { edges, emb, lab: km.lab, vals: E.vals.slice(0, 4) };
  }
  V.cml2_spectral = {
    title: "Spectral clustering: build a graph, embed with Laplacian eigenvectors, then run k-means",
    params: [{ id: "kn", label: "neighbours per point", min: 4, max: 14, step: 1, value: 8 }],
    gen(_, P) {
      const { X, truth } = MOONS, km = kmeans2(X), sp = spectral(X, P.kn), S = [];
      const acc = (lab) => { const m = truth.reduce((s, t, i) => s + (t === lab[i] ? 1 : 0), 0); return Math.max(m, X.length - m) / X.length; };
      const base = (lab, extra) => chart({ W: 460, H: 300, x: [-1.4, 2.4], y: [-1.1, 1.5], xl: "x₁", yl: "x₂", ticks: 5, yticks: 5 }, (extra || []).concat([circles(X.map((p, i) => [p[0], p[1], fillDot(lab ? col(lab[i]) : "var(--muted)"), 4.4]))]));
      S.push({ svg: base(null), side: { points: X.length, "true clusters": 2 }, msg: "**The data**: two interleaved half-moons. A human sees two clusters at once, but they are not blobs: each is a long curved shape, and the two come close together in the middle." });
      S.push({ svg: base(km.lab, [(Sx) => `<circle cx="${Sx.sx(km.C[0][0])}" cy="${Sx.sy(km.C[0][1])}" r="7" style="fill:none;stroke:var(--text);stroke-width:2.4"/><circle cx="${Sx.sx(km.C[1][0])}" cy="${Sx.sy(km.C[1][1])}" r="7" style="fill:none;stroke:var(--text);stroke-width:2.4"/>`]), side: { method: "k-means", accuracy: (acc(km.lab) * 100).toFixed(0) + "%" }, msg: "**k-means fails.** It draws a straight boundary halfway between two centroids, because it can only make round blobs. It slices each moon in half. Spectral clustering will fix this by changing what 'close' means." });
      S.push({ svg: base(null, [(Sx) => sp.edges.map(([i, j]) => segPx(Sx, X[i][0], X[i][1], X[j][0], X[j][1], "stroke:var(--accent);stroke-width:1;opacity:.55")).join("")]), side: { "neighbours k": P.kn, edges: sp.edges.length }, msg: `**Step 1: a similarity graph.** Connect every point to its ${P.kn} nearest neighbours (edge weight $e^{-d^2/\\sigma^2}$, big for close points). Points on the same moon are linked in a long chain; only a few thin edges bridge the gap between moons. Clusters are now 'groups that are well connected inside and weakly connected to the outside'.` });
      const ext = Math.max(...sp.emb.map((e) => Math.max(Math.abs(e[0]), Math.abs(e[1])))) * 1.15;
      S.push({ svg: chart({ W: 460, H: 300, x: [-ext, ext], y: [-ext, ext], xl: "eigenvector 1 (normalised row)", yl: "eigenvector 2", ticks: 4, yticks: 4 }, [circles(X.map((_, i) => [sp.emb[i][0], sp.emb[i][1], fillDot(col(truth[i])), 4.4]))]), side: { "smallest eigenvalues": sp.vals.map((v) => fx(v, 3)).join(", ") }, msg: "**Step 2: spectral embedding.** Compute the two smallest eigenvectors of the normalised graph Laplacian $L=I-D^{-1/2}WD^{-1/2}$ and give each point its two entries as new coordinates (scaled to length 1). Each moon is a well-connected group, so all of its points land in (almost) the same spot: two clumps, here two dots, one per moon, perfectly separated. On noisier data they would be small clouds. (Colours show the true moons, only to confirm the embedding separates them.)" });
      S.push({ svg: base(sp.lab), side: { method: "spectral", accuracy: (acc(sp.lab) * 100).toFixed(0) + "%" }, msg: `**Step 3: k-means in the embedding.** Plain k-means now has an easy job on two clumps, and mapping the labels back to the original points gives the right moons (accuracy ${(acc(sp.lab) * 100).toFixed(0)}%). With very few neighbours the graph can break into several pieces and with many it blurs the gap; try the slider.` });
      return S;
    },
  };

  /* ============================================================ AdaBoost */
  const ADA = (() => {
    const r = rng(17), pts = [];
    for (let i = 0; i < 32; i++) { const x = r(), y = r(); let lab = x + y > 1 ? 1 : -1; if (i % 11 === 0) lab = -lab; pts.push({ x, y, lab }); }
    return pts;
  })();
  function adaRun(T) {
    const n = ADA.length; let w = new Array(n).fill(1 / n); const stumps = [], frames = [];
    const F = (x, y, upto) => sum(stumps.slice(0, upto).map((s) => s.alpha * s.h(x, y)));
    frames.push({ round: 0, w: w.slice(), stumps: [], acc: 0.5 });
    for (let t = 1; t <= T; t++) {
      let best = null;
      for (const f of [0, 1]) for (let q = 0.04; q < 1; q += 0.04) for (const sgn of [1, -1]) {
        const err = sum(ADA.map((p, i) => ((sgn * ((f ? p.y : p.x) > q ? 1 : -1)) !== p.lab ? w[i] : 0)));
        if (!best || err < best.err) best = { err, f, q, sgn };
      }
      const e = Math.max(best.err, 1e-9), alpha = 0.5 * Math.log((1 - e) / e), { f, q, sgn } = best;
      const h = (x, y) => sgn * ((f ? y : x) > q ? 1 : -1), wrong = ADA.map((p) => h(p.x, p.y) !== p.lab);
      const wOld = w.slice();
      w = w.map((v, i) => v * Math.exp(-alpha * ADA[i].lab * h(ADA[i].x, ADA[i].y))); const Z = sum(w); w = w.map((v) => v / Z);
      stumps.push({ f, q, sgn, alpha, h, err: e });
      const acc = sum(ADA.map((p) => ((F(p.x, p.y, t) >= 0 ? 1 : -1) === p.lab ? 1 : 0))) / n;
      frames.push({ round: t, wOld, w: w.slice(), wrong, stump: stumps[t - 1], stumps: stumps.slice(), acc, Fn: (x, y) => F(x, y, t) });
    }
    return frames;
  }
  V.cml2_adaboost = {
    title: "AdaBoost: each round trains a stump on re-weighted points",
    params: [{ id: "T", label: "rounds", min: 2, max: 12, step: 1, value: 8 }],
    slide: "end",
    gen(_, P) {
      const FR = adaRun(P.T), S = [], mw = (ws) => Math.max(...ws);
      FR.forEach((f) => {
        const sizes = (f.round === 0 ? new Array(ADA.length).fill(1 / ADA.length) : f.wOld), scale = (v) => 3 + 14 * Math.sqrt(v / 0.1);
        const left = chart({ W: 360, H: 340, x: [0, 1], y: [0, 1], xl: "x₁", yl: "x₂", ticks: 5, yticks: 5 }, [
          (Sx) => f.round > 0 ? (f.stump.f ? `<rect x="${Sx.P.l}" y="${Sx.sy(1)}" width="${Sx.W - Sx.P.l - Sx.P.r}" height="${(Sx.sy(f.stump.q) - Sx.sy(1)).toFixed(1)}" style="fill:${soft(f.stump.sgn > 0 ? "var(--accent)" : "var(--amber)", 14)}"/><rect x="${Sx.P.l}" y="${Sx.sy(f.stump.q)}" width="${Sx.W - Sx.P.l - Sx.P.r}" height="${(Sx.sy(0) - Sx.sy(f.stump.q)).toFixed(1)}" style="fill:${soft(f.stump.sgn > 0 ? "var(--amber)" : "var(--accent)", 14)}"/>`
            : `<rect x="${Sx.sx(0)}" y="${Sx.P.t}" width="${(Sx.sx(f.stump.q) - Sx.sx(0)).toFixed(1)}" height="${Sx.H - Sx.P.t - Sx.P.b}" style="fill:${soft(f.stump.sgn > 0 ? "var(--amber)" : "var(--accent)", 14)}"/><rect x="${Sx.sx(f.stump.q)}" y="${Sx.P.t}" width="${(Sx.sx(1) - Sx.sx(f.stump.q)).toFixed(1)}" height="${Sx.H - Sx.P.t - Sx.P.b}" style="fill:${soft(f.stump.sgn > 0 ? "var(--accent)" : "var(--amber)", 14)}"/>`) : "",
          circles(ADA.map((p, i) => [p.x, p.y, fillDot(p.lab > 0 ? "var(--accent)" : "var(--amber)") + (f.round > 0 && f.wrong[i] ? ";stroke:var(--red);stroke-width:2.6" : ""), scale(sizes[i])])),
        ]);
        const right = chart({ W: 360, H: 340, x: [0, 1], y: [0, 1], xl: "x₁", yl: "x₂", ticks: 5, yticks: 5 }, [
          f.round > 0 ? Plot.regions((x, y) => (f.Fn(x, y) >= 0 ? "a" : "b"), 40) : "",
          circles(ADA.map((p) => [p.x, p.y, fillDot(p.lab > 0 ? "var(--accent)" : "var(--amber)"), 4.6])),
        ]);
        const msg = f.round === 0 ? "**Start.** Every one of the 32 points has the same weight $1/32$ (dot size shows the weight). Two classes, separated roughly by a diagonal, with a few noisy labels. A single vertical or horizontal cut (a **stump**) can never separate them; boosting will combine many."
          : `**Round ${f.round}.** Using the current weights (left), the best stump is "${f.stump.f ? "x₂" : "x₁"} ${f.stump.sgn > 0 ? ">" : "<"} ${fx(f.stump.q, 2)}". Its weighted error is $\\varepsilon=${fx(f.stump.err, 3)}$, so it earns the vote $\\alpha=\\tfrac12\\ln\\frac{1-\\varepsilon}{\\varepsilon}=${fx(f.stump.alpha, 3)}$. The circled (red-ringed) points were misclassified: their weights are multiplied by $e^{\\alpha}$ and the correct ones by $e^{-\\alpha}$, then everything is renormalised. The next round will focus on them. Right: the combined classifier $\\mathrm{sign}(\\sum_t\\alpha_t h_t)$, training accuracy ${(f.acc * 100).toFixed(0)}%.`;
        S.push({ svg: row(left, right), side: { round: f.round, "ε (weighted error)": f.round ? fx(f.stump.err, 3) : "-", "α (vote)": f.round ? fx(f.stump.alpha, 3) : "-", "ensemble accuracy": f.round ? (f.acc * 100).toFixed(0) + "%" : "-" }, msg });
      });
      return S;
    },
  };

  /* ============================================================ Isolation Forest: random cuts */
  const ISO = (() => {
    const r = rng(23), pts = [];
    for (let i = 0; i < 60; i++) pts.push([gauss(r) * 1.1 + 5, gauss(r) * 1.1 + 5]);
    pts.push([9.3, 8.6]);                              // index 60: the outlier
    return pts;
  })();
  const cFn = (n) => (n <= 1 ? 0 : n === 2 ? 1 : 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1)) / n);
  function isoPath(target, seed) {
    const r = rng(seed * 977 + target * 31 + 5), steps = []; let cell = { x0: 0, x1: 10, y0: 0, y1: 10 }, idx = ISO.map((_, i) => i);
    while (idx.length > 1 && steps.length < 24) {
      const f = r() < 0.5 ? 0 : 1, vals = idx.map((i) => ISO[i][f]), lo = Math.min(...vals), hi = Math.max(...vals), q = lo + (hi - lo) * r();
      const left = ISO[target][f] <= q, parent = { ...cell };
      if (f === 0) { if (left) cell.x1 = q; else cell.x0 = q; } else { if (left) cell.y1 = q; else cell.y0 = q; }
      idx = idx.filter((i) => (ISO[i][f] <= q) === left);
      steps.push({ f, q, cell: { ...cell }, parent, left: idx.length, idxSnap: idx.slice() });
    }
    return steps;
  }
  function isoAvg(target) { let s = 0; const T = 120; for (let t = 1; t <= T; t++) { const p = isoPath(target, t * 3 + 1); const last = p.length ? p[p.length - 1].left : 61; s += p.length + cFn(last); } return s / T; }
  const ISO_AVG = { 60: isoAvg(60), 0: isoAvg(0), 1: isoAvg(1) };
  V.cml2_iforest = {
    title: "Isolation Forest: random cuts isolate an outlier in very few steps",
    params: [{ id: "who", label: "point to isolate (0 = outlier, 1 = typical, 2 = another typical)", min: 0, max: 2, step: 1, value: 0 }, { id: "seed", label: "random tree number", min: 1, max: 8, step: 1, value: 1 }],
    gen(_, P) {
      const target = P.who === 0 ? 60 : P.who === 1 ? 0 : 1, path = isoPath(target, P.seed), S = [];
      const avg = ISO_AVG[target], sc = Math.pow(2, -avg / cFn(61));
      for (let s = 0; s <= path.length; s++) {
        const cur = s ? path[s - 1] : null, cell = cur ? cur.cell : { x0: 0, x1: 10, y0: 0, y1: 10 };
        const cuts = path.slice(0, s).map((c) => (Sx) => (c.f === 0 ? segPx(Sx, c.q, c.parent.y0, c.q, c.parent.y1, "stroke:var(--muted);stroke-width:1.4;stroke-dasharray:4 3") : segPx(Sx, c.parent.x0, c.q, c.parent.x1, c.q, "stroke:var(--muted);stroke-width:1.4;stroke-dasharray:4 3")));
        const inCell = (p) => p[0] >= cell.x0 && p[0] <= cell.x1 && p[1] >= cell.y0 && p[1] <= cell.y1;
        const svg = chart({ W: 480, H: 330, x: [0, 10], y: [0, 10], xl: "x₁", yl: "x₂", ticks: 5, yticks: 5 }, [
          (Sx) => `<rect x="${Sx.sx(cell.x0)}" y="${Sx.sy(cell.y1)}" width="${(Sx.sx(cell.x1) - Sx.sx(cell.x0)).toFixed(1)}" height="${(Sx.sy(cell.y0) - Sx.sy(cell.y1)).toFixed(1)}" style="fill:${soft("var(--green)", 16)};stroke:var(--green);stroke-width:1.6"/>`,
          ...cuts,
          circles(ISO.map((p, i) => [p[0], p[1], i === target ? "fill:var(--red);stroke:var(--text);stroke-width:2" : fillDot(inCell(p) ? "var(--accent)" : "var(--muted)", inCell(p) ? "" : ";opacity:.45"), i === target ? 7 : 4.2])),
        ]);
        const left = s ? path[s - 1].left : ISO.length;
        const done = left <= 1;
        const msg = s === 0 ? `**Start.** 61 points: a dense cloud plus one far-away point. The red point is the one we try to isolate. Each step picks a random feature and a random split value between the min and max of the points *still in the green box*, and keeps the side that contains the red point.`
          : done ? `**Isolated after ${s} cut${s > 1 ? "s" : ""}.** The red point is alone in its box, so its path length is $h=${s}$. ${target === 60 ? "An outlier sits far from everyone, so a random cut very often puts it alone straight away. Try other tree numbers: it stays short." : "A typical point is surrounded by neighbours, so it takes many cuts to separate it from all of them. Try other tree numbers: it stays long."}`
          : `**Cut ${s}.** Random split on ${cur.f ? "x₂" : "x₁"} at ${fx(cur.q, 2)}. ${left} point${left > 1 ? "s" : ""} remain in the box with the red point.`;
        S.push({ svg, side: { cuts: s, "points left in box": left, "avg path (120 trees)": fx(avg, 1), "anomaly score": fx(sc, 2) }, msg });
      }
      return S;
    },
  };

  /* ============================================================ Kalman filter */
  const KAL = (() => {
    const r = rng(31), T = 30, truth = [], z = [];
    for (let t = 0; t <= T; t++) { const x = 10 + 0.8 * t + 5 * Math.sin(t / 5); truth.push(x); z.push(x + 1.5 * gauss(r)); }
    return { T, truth, z };
  })();
  function kalmanRun(q, Rm) {
    const { T, z } = KAL, Fm = [[1, 1], [0, 1]], Q = [[q / 4, q / 2], [q / 2, q]];
    let x = [z[0], 0], Pm = [[Rm, 0], [0, 4]]; const fr = [{ t: 0, kind: "init", x: x.slice(), P: Pm.map((r) => r.slice()), prior: null }];
    for (let t = 1; t <= T; t++) {
      const xp = [x[0] + x[1], x[1]], FP = [[Pm[0][0] + Pm[1][0], Pm[0][1] + Pm[1][1]], [Pm[1][0], Pm[1][1]]];
      const Pp = [[FP[0][0] + FP[0][1] + Q[0][0], FP[0][1] + Q[0][1]], [FP[1][0] + FP[1][1] + Q[1][0], FP[1][1] + Q[1][1]]];
      fr.push({ t, kind: "predict", x: xp.slice(), P: Pp.map((r) => r.slice()) });
      const Sx = Pp[0][0] + Rm, K = [Pp[0][0] / Sx, Pp[1][0] / Sx], inn = z[t] - xp[0];
      x = [xp[0] + K[0] * inn, xp[1] + K[1] * inn];
      Pm = [[(1 - K[0]) * Pp[0][0], (1 - K[0]) * Pp[0][1]], [Pp[1][0] - K[1] * Pp[0][0], Pp[1][1] - K[1] * Pp[0][1]]];
      fr.push({ t, kind: "update", x: x.slice(), P: Pm.map((r) => r.slice()), K, inn, prior: xp[0], priorSd: Math.sqrt(Pp[0][0]) });
    }
    return fr;
  }
  V.cml2_kalman = {
    title: "Kalman filter: predict, measure, correct",
    params: [{ id: "lq", label: "process noise q (how much the motion can change)", min: 0.001, max: 1, step: 0.001, value: 0.05 }, { id: "lr", label: "measurement noise R (how noisy the sensor is)", min: 0.2, max: 12, step: 0.1, value: 2.25 }],
    gen(_, P) {
      const FR = kalmanRun(P.lq, P.lr), { T, truth, z } = KAL, S = [];
      FR.forEach((f, fi) => {
        const upto = f.t, ests = FR.slice(0, fi + 1).filter((g) => g.kind === "update" || g.kind === "init").map((g) => g.t),
          estPts = FR.slice(0, fi + 1).filter((g) => g.kind === "update" || g.kind === "init").map((g) => [g.t, g.x[0]]);
        const bandLo = FR.slice(0, fi + 1).filter((g) => g.kind === "update" || g.kind === "init").map((g) => [g.t, g.x[0] - 2 * Math.sqrt(g.P[0][0])]);
        const bandHi = FR.slice(0, fi + 1).filter((g) => g.kind === "update" || g.kind === "init").map((g) => [g.t, g.x[0] + 2 * Math.sqrt(g.P[0][0])]);
        const showMeas = f.kind === "update" || f.kind === "init";
        const sd = Math.sqrt(f.P[0][0]);
        const svg = chart({ W: 620, H: 330, x: [0, T], y: [-6, 40], xl: "time step", yl: "position", ticks: 6, yticks: 5 }, [
          band(bandLo, bandHi, "var(--green)", 16),
          curve(truth.slice(0, T + 1).map((v, i) => [i, v]), "var(--muted)", 1.6, "5 4"),
          circles(z.slice(0, upto + (f.kind === "update" || f.kind === "init" ? 1 : 0)).map((v, i) => [i, v, "fill:var(--amber);stroke:var(--panel);stroke-width:1", 3.6])),
          curve(estPts, "var(--green)", 2.6),
          f.kind === "predict" ? (Sx) => `<line x1="${Sx.sx(f.t)}" y1="${Sx.sy(f.x[0] - 2 * sd)}" x2="${Sx.sx(f.t)}" y2="${Sx.sy(f.x[0] + 2 * sd)}" style="stroke:var(--purple);stroke-width:3"/><circle cx="${Sx.sx(f.t)}" cy="${Sx.sy(f.x[0])}" r="6" style="fill:var(--panel);stroke:var(--purple);stroke-width:2.6"/>` : "",
          f.kind === "update" ? (Sx) => `<line x1="${Sx.sx(f.t)}" y1="${Sx.sy(f.prior - 2 * f.priorSd)}" x2="${Sx.sx(f.t)}" y2="${Sx.sy(f.prior + 2 * f.priorSd)}" style="stroke:var(--purple);stroke-width:2;stroke-dasharray:3 3"/><circle cx="${Sx.sx(f.t)}" cy="${Sx.sy(f.prior)}" r="5" style="fill:var(--panel);stroke:var(--purple);stroke-width:2"/><circle cx="${Sx.sx(f.t)}" cy="${Sx.sy(f.x[0])}" r="6" style="fill:var(--green);stroke:var(--panel);stroke-width:1.5"/>` : "",
          (Sx) => `<g style="font-size:11px"><circle cx="${Sx.P.l + 12}" cy="${Sx.P.t + 12}" r="4" style="fill:var(--amber)"/><text x="${Sx.P.l + 22}" y="${Sx.P.t + 16}" style="fill:var(--text)">measurement</text><line x1="${Sx.P.l + 110}" x2="${Sx.P.l + 130}" y1="${Sx.P.t + 12}" y2="${Sx.P.t + 12}" style="stroke:var(--green);stroke-width:2.6"/><text x="${Sx.P.l + 136}" y="${Sx.P.t + 16}" style="fill:var(--text)">estimate ± 2σ</text><line x1="${Sx.P.l + 235}" x2="${Sx.P.l + 255}" y1="${Sx.P.t + 12}" y2="${Sx.P.t + 12}" style="stroke:var(--muted);stroke-width:1.6;stroke-dasharray:5 4"/><text x="${Sx.P.l + 261}" y="${Sx.P.t + 16}" style="fill:var(--text)">true position</text><circle cx="${Sx.P.l + 350}" cy="${Sx.P.t + 12}" r="4.5" style="fill:var(--panel);stroke:var(--purple);stroke-width:2"/><text x="${Sx.P.l + 359}" y="${Sx.P.t + 16}" style="fill:var(--text)">prediction</text></g>`,
        ]);
        const msg = f.kind === "init" ? `**Start.** We have only the first reading, so the position estimate is that reading, the velocity is a guess of 0, and the uncertainty is wide. The model: position moves by velocity each step; the sensor adds noise of variance R=${fx(P.lr, 2)}; the motion can change by process noise q=${fx(P.lq, 3)}.`
          : f.kind === "predict" ? `**Time ${f.t}, predict.** Push the last estimate forward with the motion model, $\\hat x^- = F\\hat x$. The prior position is ${fx(f.x[0], 2)} and, because the motion model is not perfect, the uncertainty has *grown* (± 2σ = ${fx(2 * sd, 2)}): $P^-=FPF^\\top+Q$.`
          : `**Time ${f.t}, update.** The sensor says ${fx(z[f.t], 2)} (amber), the prediction was ${fx(f.prior, 2)}. The gap (innovation) is ${fx(f.inn, 2)}; the Kalman gain $K=${fx(f.K[0], 2)}$ decides how far to move toward the measurement. ${f.K[0] > 0.5 ? "A large gain: the filter trusts the sensor." : "A small gain: the filter trusts its own prediction more."} The new estimate ${fx(f.x[0], 2)} is better than either source and the band shrinks (± 2σ = ${fx(2 * sd, 2)}).`;
        S.push({ svg, side: { time: f.t, phase: f.kind, "estimate": fx(f.x[0], 2), "σ of position": fx(sd, 2), "gain K": f.K ? fx(f.K[0], 3) : "-" }, msg });
      });
      return S;
    },
  };

  /* ============================================================ Time series: AR(2) and its ACF */
  const ARN = (() => { const r = rng(41), e = []; for (let i = 0; i < 260; i++) e.push(gauss(r)); return e; })();
  V.cml2_ar = {
    title: "AR(2): simulate a time series and read its autocorrelation function",
    params: [{ id: "p1", label: "φ₁", min: -1.9, max: 1.9, step: 0.05, value: 0.7 }, { id: "p2", label: "φ₂ (0 gives AR(1))", min: -0.95, max: 0.95, step: 0.05, value: 0 }],
    gen(_, P) {
      const p1 = P.p1, p2 = P.p2, N = 130, burn = 100, y = new Array(N + burn).fill(0);
      for (let t = 2; t < N + burn; t++) y[t] = p1 * y[t - 1] + p2 * y[t - 2] + ARN[t];
      const s = y.slice(burn), stationary = p2 < 1 + 0 && p2 + p1 < 1 && p2 - p1 < 1 && Math.abs(p2) < 1;
      const m = sum(s) / N, c0 = sum(s.map((v) => (v - m) ** 2)), L = 20, acf = [];
      for (let k = 0; k <= L; k++) acf.push(sum(s.slice(k).map((v, i) => (v - m) * (s[i] - m))) / c0);
      const th = [1]; if (stationary) { th.push(p1 / (1 - p2)); for (let k = 2; k <= L; k++) th.push(p1 * th[k - 1] + p2 * th[k - 2]); }
      const ext = Math.max(3, ...s.map(Math.abs)) * 1.05;
      const ser = chart({ W: 560, H: 190, x: [0, N], y: [-ext, ext], xl: "time t", yl: "yₜ", ticks: 6, yticks: 4 }, [curve(s.map((v, i) => [i, v]), "var(--accent)", 1.8), Plot.hline(0, "pl-dash")]);
      const ac = chart({ W: 560, H: 180, x: [-0.6, L + 0.6], y: [-1.05, 1.05], xl: "lag k", yl: "autocorrelation", ticks: 10, yticks: 4 }, [
        Plot.hline(0, "pl-thin"), Plot.hline(1.96 / Math.sqrt(N), "pl-dash"), Plot.hline(-1.96 / Math.sqrt(N), "pl-dash"),
        (Sx) => acf.map((v, k) => (k ? `<line x1="${Sx.sx(k)}" x2="${Sx.sx(k)}" y1="${Sx.sy(0)}" y2="${Sx.sy(Math.max(-1.05, Math.min(1.05, v)))}" style="stroke:var(--accent);stroke-width:4"/>` : "")).join(""),
        stationary ? (Sx) => th.map((v, k) => (k ? `<circle cx="${Sx.sx(k)}" cy="${Sx.sy(v)}" r="3.6" style="fill:none;stroke:var(--amber);stroke-width:2"/>` : "")).join("") : "",
      ]);
      const kind = p2 === 0 ? "AR(1)" : "AR(2)", complex = stationary && p1 * p1 + 4 * p2 < 0;
      return [{
        svg: colm(ser, ac), side: { model: kind, stationary: stationary ? "yes" : "NO", "ρ₁ (sample)": fx(acf[1], 2), "ρ₁ (theory)": stationary ? fx(th[1], 2) : "-" },
        msg: !stationary ? `**Not stationary.** With these coefficients the characteristic roots are not all outside the unit circle, so the series wanders off or explodes, and the autocorrelation has no meaning. The stationarity region of an AR(2) is the triangle $\\varphi_2+\\varphi_1<1$, $\\varphi_2-\\varphi_1<1$, $|\\varphi_2|<1$.`
          : p2 === 0 ? `**AR(1).** Each value is ${fx(p1, 2)} times the previous one plus fresh noise. The ACF (blue bars, sample; amber rings, theory $\\rho_k=\\varphi_1^k$) decays geometrically${p1 < 0 ? ", flipping sign every lag because φ₁ is negative" : ""}. Closer to 1, memory is longer and the series is smoother; the dashed lines are the ±1.96/√n noise band.`
          : `**AR(2).** $y_t=${fx(p1, 2)}\\,y_{t-1}${p2 < 0 ? "" : "+"}${fx(p2, 2)}\\,y_{t-2}+\\varepsilon_t$. ${complex ? "The roots are complex, so the ACF decays as a damped wave: the series has a pseudo-cycle." : "The roots are real, so the ACF decays without oscillating (or with a simple sign flip)."} Amber rings are the theoretical ACF from the Yule-Walker recursion $\\rho_k=\\varphi_1\\rho_{k-1}+\\varphi_2\\rho_{k-2}$.`,
      }];
    },
  };

  /* ============================================================ diagrams */
  D.cml2_ica = () => {
    const W = 640, H = 210; let s = "";
    s += box(14, 24, 110, 64, "var(--accent)", 14) + txt(69, 50, "sources s", { b: 1 }) + txt(69, 68, "voice A, voice B", { fs: 11, c: "var(--muted)" });
    s += arrow(126, 56, 188, 56, "var(--muted)", 2) + txt(157, 46, "mix A", { fs: 11, c: "var(--muted)" });
    s += box(190, 24, 130, 64, "var(--amber)", 14) + txt(255, 50, "microphones x", { b: 1 }) + txt(255, 68, "x = A s, two blends", { fs: 11, c: "var(--muted)" });
    s += arrow(322, 56, 384, 56, "var(--muted)", 2) + txt(353, 46, "whiten", { fs: 11, c: "var(--muted)" });
    s += box(386, 24, 110, 64, "var(--purple)", 14) + txt(441, 50, "whitened z", { b: 1 }) + txt(441, 68, "uncorrelated", { fs: 11, c: "var(--muted)" });
    s += arrow(498, 56, 540, 56, "var(--muted)", 2) + txt(519, 46, "rotate", { fs: 11, c: "var(--muted)" });
    s += box(542, 24, 84, 64, "var(--green)", 14) + txt(584, 50, "ŝ", { b: 1, fs: 15 }) + txt(584, 68, "recovered", { fs: 11, c: "var(--muted)" });
    s += txt(320, 130, "PCA stops after whitening: it only makes the signals uncorrelated.", { fs: 12 });
    s += txt(320, 152, "ICA adds one more step: find the rotation that makes the signals independent,", { fs: 12 });
    s += txt(320, 170, "which means as non-Gaussian as possible (central limit theorem: mixing makes things more Gaussian).", { fs: 12, c: "var(--muted)" });
    return svgWrap(W, H, s);
  };
  D.cml2_nmffactor = () => {
    const W = 600, H = 190; let s = "";
    const mat = (x, y, w, h, c, lab, sub) => box(x, y, w, h, c, 16, 4) + txt(x + w / 2, y + h / 2 + 4, lab, { b: 1, fs: 15 }) + txt(x + w / 2, y + h + 16, sub, { fs: 11, c: "var(--muted)" });
    s += mat(20, 30, 130, 100, "var(--accent)", "V", "n × p  (data, ≥ 0)") + txt(176, 84, "≈", { fs: 26 });
    s += mat(200, 30, 50, 100, "var(--amber)", "W", "n × k") + txt(276, 86, "×", { fs: 22 });
    s += mat(300, 55, 130, 50, "var(--green)", "H", "k × p  (parts)");
    s += txt(500, 60, "row i of W: how much of", { a: "start", fs: 11.5 }) + txt(500, 76, "each part sample i uses", { a: "start", fs: 11.5 }) + txt(500, 104, "row of H: one part", { a: "start", fs: 11.5 }) + txt(500, 120, "(a basis 'image')", { a: "start", fs: 11.5 });
    s += txt(300, 176, "all entries non-negative, so parts only add and never cancel", { fs: 12, c: "var(--muted)" });
    return svgWrap(W, H, s);
  };
  D.cml2_manifold = () => {
    const W = 600, H = 230; let s = "";
    const sp = []; for (let i = 0; i <= 40; i++) { const a = 0.4 + (i / 40) * 4.4, rr = 18 + 22 * a; sp.push([200 + rr * Math.cos(a) * 0.55 + 10, 118 - rr * Math.sin(a) * 0.55 + 20]); }
    s += `<path d="${sp.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join("")}" style="fill:none;stroke:var(--accent);stroke-width:4;stroke-linecap:round"/>`;
    const a = sp[3], b = sp[37];
    s += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" style="stroke:var(--red);stroke-width:2;stroke-dasharray:6 4"/>` + txt((a[0] + b[0]) / 2 + 40, (a[1] + b[1]) / 2 - 6, "straight-line (Euclidean) distance", { fs: 11.5, c: "var(--red)" });
    s += `<circle cx="${a[0]}" cy="${a[1]}" r="5" style="fill:var(--red)"/><circle cx="${b[0]}" cy="${b[1]}" r="5" style="fill:var(--red)"/>`;
    s += txt(300, 26, "data lie on a curled sheet (a manifold)", { fs: 12.5, b: 1 }) + txt(150, 215, "geodesic distance: walk along the sheet (blue)", { fs: 11.5, c: "var(--accent)" });
    s += arrow(380, 120, 450, 120, "var(--muted)", 2.2) + txt(415, 108, "unroll", { fs: 11, c: "var(--muted)" });
    s += `<line x1="470" y1="120" x2="580" y2="120" style="stroke:var(--accent);stroke-width:4;stroke-linecap:round"/><circle cx="470" cy="120" r="5" style="fill:var(--red)"/><circle cx="580" cy="120" r="5" style="fill:var(--red)"/>` + txt(525, 146, "now the distances agree", { fs: 11.5, c: "var(--muted)" });
    return svgWrap(W, H, s);
  };
  D.cml2_kalmanloop = () => {
    const W = 600, H = 230; let s = "";
    s += box(40, 20, 220, 70, "var(--purple)", 14) + txt(150, 44, "1. PREDICT", { b: 1, c: "var(--purple)", fs: 14 }) + txt(150, 64, "x⁻ = F x     P⁻ = F P Fᵀ + Q", { fs: 12 }) + txt(150, 80, "uncertainty grows", { fs: 11, c: "var(--muted)" });
    s += box(340, 20, 220, 70, "var(--amber)", 14) + txt(450, 44, "2. MEASURE", { b: 1, c: "var(--amber)", fs: 14 }) + txt(450, 64, "z = H x + noise (variance R)", { fs: 12 }) + txt(450, 80, "a new noisy reading arrives", { fs: 11, c: "var(--muted)" });
    s += box(150, 140, 300, 72, "var(--green)", 14) + txt(300, 163, "3. UPDATE (correct)", { b: 1, c: "var(--green)", fs: 14 }) + txt(300, 182, "K = P⁻Hᵀ (H P⁻Hᵀ + R)⁻¹     x = x⁻ + K (z − H x⁻)", { fs: 11.5 }) + txt(300, 200, "P = (I − K H) P⁻   : uncertainty shrinks", { fs: 11.5 });
    s += arrow(262, 55, 338, 55, "var(--muted)", 2) + arrow(450, 92, 380, 138, "var(--muted)", 2) + arrow(150, 92, 220, 138, "var(--muted)", 2);
    s += `<path d="M152,214 C60,236 8,150 40,70" style="fill:none;stroke:var(--green);stroke-width:2.2" marker-end="url(#cml2ah)"/>` + txt(40, 150, "next time step", { fs: 10.5, c: "var(--muted)", a: "end" });
    return svgWrap(W, H, s);
  };
  D.cml2_lattice = () => {
    const W = 600, H = 250; let s = "";
    const node = (x, y, lab, st) => `<rect x="${x - 26}" y="${y - 13}" width="52" height="26" rx="6" style="fill:${st === "x" ? soft("var(--red)", 14) : soft("var(--green)", 16)};stroke:${st === "x" ? "var(--red)" : "var(--green)"};stroke-width:1.5${st === "x" ? ";stroke-dasharray:4 3" : ""}"/>` + txt(x, y + 4, lab, { fs: 12 });
    const L0 = [[300, 24, "{}", ""]], L1 = [[90, 84, "{A}", ""], [210, 84, "{B}", ""], [330, 84, "{C}", ""], [450, 84, "{D}", "x"]];
    const L2 = [[60, 150, "{A,B}", ""], [150, 150, "{A,C}", ""], [240, 150, "{B,C}", ""], [330, 150, "{A,D}", "x"], [420, 150, "{B,D}", "x"], [510, 150, "{C,D}", "x"]];
    const L3 = [[130, 214, "{A,B,C}", ""], [330, 214, "{A,B,D}", "x"], [470, 214, "{...D}", "x"]];
    const link = (a, b, dashed) => `<line x1="${a[0]}" y1="${a[1] + 13}" x2="${b[0]}" y2="${b[1] - 13}" style="stroke:var(--muted);stroke-width:1;opacity:${dashed ? .35 : .8}${dashed ? ";stroke-dasharray:3 3" : ""}"/>`;
    L1.forEach((n) => { s += link(L0[0], n, n[3] === "x"); });
    [[0, 0], [0, 1], [1, 2], [0, 3], [1, 4], [2, 5]].forEach(([i, j]) => {});
    const edges = [[0, 0], [1, 0], [0, 1], [2, 1], [1, 2], [2, 2], [0, 3], [3, 3], [1, 4], [3, 4], [2, 5], [3, 5]];
    edges.forEach(([i, j]) => { s += link(L1[i], L2[j], L2[j][3] === "x"); });
    [[0, 0], [1, 0], [2, 0], [0, 1], [3, 1], [4, 1], [1, 2], [4, 2], [5, 2]].forEach(([j, k]) => { s += link(L2[j], L3[k], L3[k][3] === "x"); });
    [L0, L1, L2, L3].forEach((L) => L.forEach((n) => { s += node(n[0], n[1], n[2], n[3]); }));
    s += txt(590, 44, "infrequent: {D} fails minsup", { a: "end", fs: 11, c: "var(--red)" }) + txt(590, 62, "so every superset is pruned", { a: "end", fs: 11, c: "var(--red)" });
    return svgWrap(W, H, s);
  };

  /* ============================================================ glossary */
  const G = {
    "independent component analysis": "A method that separates a mixed signal into statistically independent, non-Gaussian sources (like voices at a party recorded by several microphones).",
    "ica": "Independent Component Analysis: unmix signals into statistically independent, non-Gaussian sources.",
    "cocktail party problem": "Several people speak at once and several microphones record blends of their voices; the task is to recover each voice. ICA solves it.",
    "whitening": "A linear transform after which the data have zero mean, unit variance in every direction and no correlations (covariance equal to the identity).",
    "kurtosis": "A measure of how heavy the tails and how peaked a distribution is. A Gaussian has excess kurtosis 0; ICA uses it as one way to measure non-Gaussianity.",
    "negentropy": "A measure of non-Gaussianity: how much less entropy a variable has than a Gaussian with the same variance. ICA maximises it.",
    "fastica": "A fast fixed-point algorithm for ICA: whiten the data, then repeatedly update each unmixing vector with a simple non-linear rule until it converges.",
    "nmf": "Non-negative Matrix Factorization: write a non-negative matrix as the product of two smaller non-negative matrices, giving additive, parts-based features.",
    "non-negative matrix factorization": "Approximating a non-negative matrix V by W times H where both W and H are non-negative. Parts can only add up, never cancel, so the factors tend to be interpretable.",
    "multiplicative update": "The classic NMF update rule (Lee and Seung): multiply each entry by a ratio of non-negative terms. It keeps entries non-negative automatically and never increases the error.",
    "matrix factorization": "Approximating a big matrix as a product of two thin ones. In recommenders, user factors times item factors predict ratings.",
    "latent factor": "A hidden numeric trait learned from data, such as how much a film is a comedy. Users and items each get a vector of such factors.",
    "als": "Alternating Least Squares: fix one factor matrix, solve a linear least-squares problem for the other, and swap. Each step is a convex ridge regression.",
    "alternating least squares": "Fit a matrix factorization by alternately solving for the user factors (items fixed) and the item factors (users fixed). Each half is a ridge regression.",
    "cold start": "The recommender problem that a brand-new user or item has no ratings yet, so there is nothing to learn their factors from.",
    "implicit feedback": "Signals such as clicks, plays or purchases that suggest preference without an explicit rating. Absence of a signal is not the same as dislike.",
    "perplexity": "In t-SNE: a smooth number of effective neighbours each point considers (typically 5 to 50). In language models it means something different: exp of the average cross-entropy.",
    "student-t distribution": "A bell-shaped distribution with heavier tails than a Gaussian. t-SNE uses it in the low-dimensional space so that far-apart points can repel without being squashed together.",
    "crowding problem": "In high dimensions a point has many equidistant neighbours that cannot all fit around it in 2-D. t-SNE's heavy-tailed kernel is the fix.",
    "early exaggeration": "A trick in t-SNE: multiply the high-dimensional similarities for the first iterations so clusters form tight and can move around easily.",
    "manifold": "A curved surface of low dimension sitting inside a higher-dimensional space, such as a sheet of paper rolled up in 3-D.",
    "manifold learning": "Methods (Isomap, t-SNE, UMAP, spectral embedding) that unroll data lying on a curved low-dimensional surface into a flat low-dimensional picture.",
    "geodesic distance": "The distance measured along the surface itself rather than straight through space; on a rolled-up sheet it is the walking distance along the paper.",
    "multidimensional scaling": "A family of methods that place points in a low-dimensional space so that their distances match a given table of pairwise distances as well as possible.",
    "mds": "Multidimensional Scaling: find low-dimensional coordinates whose pairwise distances match a given distance matrix.",
    "isomap": "MDS applied to geodesic distances: build a nearest-neighbour graph, take shortest-path distances through it, then embed with classical MDS.",
    "stress": "In MDS, a measure of how badly the low-dimensional distances disagree with the target distances. Smaller is better.",
    "graph laplacian": "The matrix L = D − W built from a similarity graph (D holds the degrees). Its smallest eigenvectors tell you how to cut the graph into well-connected groups.",
    "spectral clustering": "Cluster by building a similarity graph, embedding the points with the smallest eigenvectors of its Laplacian, then running k-means on that embedding.",
    "fiedler vector": "The eigenvector of the graph Laplacian with the second-smallest eigenvalue; its signs give a good two-way cut of the graph.",
    "normalized cut": "A graph-partition score that is small when few edge weights are cut relative to the total connection of each group. Spectral clustering relaxes it.",
    "adaboost": "Adaptive Boosting: train weak classifiers one after another, re-weighting the training points so that each new one focuses on the previous ones' mistakes, then take a weighted vote.",
    "decision stump": "A decision tree with a single split: one feature compared with one threshold. The classic weak learner for AdaBoost.",
    "exponential loss": "The loss exp(−y·f(x)). AdaBoost can be derived as stagewise minimisation of it.",
    "isolation forest": "An anomaly detector that builds random trees; anomalies are separated from the rest in far fewer random splits than ordinary points.",
    "anomaly detection": "Finding the rare observations that do not fit the pattern of the rest of the data, such as fraud or faulty sensors.",
    "path length": "In an isolation tree, the number of splits needed to reach a point's leaf. Short average path lengths mean anomalies.",
    "local outlier factor": "An anomaly score that compares a point's local density with the density of its neighbours; points in much sparser spots than their neighbours are outliers.",
    "lof": "Local Outlier Factor: ratio of the neighbours' density to a point's own density. About 1 is normal; much bigger than 1 is an outlier.",
    "one-class svm": "An SVM trained only on normal data that draws a boundary around it in kernel feature space; points outside are flagged as anomalies.",
    "novelty detection": "Training on clean normal data only and then flagging anything new that looks different (as opposed to outlier detection, where the training data may already contain outliers).",
    "association rule": "An if-then pattern found in transaction data, such as 'people who buy bread and butter often buy milk', measured by support, confidence and lift.",
    "apriori": "A classic algorithm for finding frequent itemsets: grow itemsets one item at a time and discard any whose subset is already infrequent.",
    "frequent itemset": "A set of items that appears together in at least a minimum fraction (the minimum support) of all transactions.",
    "support": "In association rules, the fraction of transactions that contain a given itemset.",
    "confidence": "In association rules, P(consequent | antecedent): of the transactions with the left side, the share that also have the right side.",
    "lift": "Confidence divided by the baseline support of the consequent. Above 1 means the items occur together more than chance would predict.",
    "kalman filter": "A recursive estimator that tracks a hidden state over time by alternating a prediction from a motion model with a correction from each noisy measurement, weighting them by their uncertainties.",
    "state space model": "A model with a hidden state that evolves over time and noisy measurements of it. Kalman filters and HMMs are both examples.",
    "kalman gain": "The weight in the Kalman update that decides how much to trust the new measurement versus the prediction. Near 1 means trust the sensor, near 0 means trust the model.",
    "innovation": "In a Kalman filter, the surprise: measurement minus predicted measurement.",
    "process noise": "In a Kalman filter, the variance Q that describes how much the true state can wander away from the motion model between steps.",
    "measurement noise": "In a Kalman filter, the variance R of the sensor error.",
    "time series": "Data recorded in time order, where neighbouring values are usually related. Shuffling destroys the information.",
    "stationary": "A time series whose mean, variance and autocorrelations do not change over time. Most classical time-series models assume it.",
    "autocorrelation": "The correlation of a series with a lagged copy of itself. The plot of it against the lag is the ACF.",
    "acf": "Autocorrelation function: correlation between a series and itself shifted by k steps, for k = 1, 2, 3 ...",
    "pacf": "Partial autocorrelation function: the correlation at lag k after removing the effect of the shorter lags in between. It cuts off after lag p for an AR(p) process.",
    "ar model": "Autoregressive model: each value is a weighted sum of the previous p values plus noise.",
    "ma model": "Moving-average model: each value is the current noise plus a weighted sum of the previous q noise terms.",
    "arima": "AutoRegressive Integrated Moving Average: an ARMA model fitted after differencing the series d times to make it stationary. Written ARIMA(p, d, q).",
    "differencing": "Replacing a series by the change from one step to the next, yₜ − yₜ₋₁. It removes a trend and is the 'I' in ARIMA.",
    "yule-walker": "Equations that link the autoregressive coefficients to the autocorrelations, so an AR model can be fitted from the ACF.",
    "white noise": "A series of independent random values with constant mean and variance; there is nothing left to predict.",
  };
  const have = window.GLOSSARY;
  Object.keys(G).forEach((k) => { if (!(k in have)) have[k] = G[k]; });
})();
