/* Machine Learning, part 2 (unsupervised, evaluation, practical): visualizers, diagrams and glossary terms.
   Loaded on every doc page and on Practice, after assets/viz.js. Every name starts with ml2_. */
(function () {
  const V = Viz.defs, Plot = Viz.Plot;

  /* ============================================================ private helpers */
  // seeded random numbers: every picture is identical on every load
  const rng = (seed) => { let a = (seed >>> 0) || 1; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  const gauss = (r) => { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const COL = ["var(--accent)", "var(--amber)", "var(--green)", "var(--red)", "var(--purple)"];
  const col = (j) => COL[((j % COL.length) + COL.length) % COL.length];
  const soft = (c, p) => `color-mix(in srgb, ${c} ${p || 18}%, transparent)`;
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  const fx = (v, n) => (+v).toFixed(n == null ? 2 : n);
  const pct = (v) => (100 * v).toFixed(1) + "%";
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const SQ = { W: 398, H: 394 };                       // a square 340 × 340 px plotting area (with axis labels)
  const row = (...svgs) => `<div style="display:flex;flex-wrap:wrap;gap:10px 14px;justify-content:center;align-items:center">${svgs.join("")}</div>`;

  /* Plot.chart plus an "outside" layer drawn on top without clipping (tick labels, legends) */
  function chart(o, layers, outside) {
    let SS = null;
    const s = Plot.chart(o, [(S) => { SS = S; return ""; }].concat(layers));
    return outside ? s.replace(/<\/svg>\s*$/, outside(SS) + "</svg>") : s;
  }
  // layers that take their colours as CSS (so light and dark both work)
  const circles = (pts, r0) => (S) => pts.map((p) => `<circle cx="${S.sx(p[0]).toFixed(1)}" cy="${S.sy(p[1]).toFixed(1)}" r="${p[3] || r0 || 4.5}" style="${p[2]}"/>`).join("");
  const fillDot = (c, extra) => `fill:${c};stroke:var(--panel);stroke-width:1.2${extra || ""}`;
  const diamond = (S, x, y, c, s, dash) => {
    const X = S.sx(x), Y = S.sy(y); s = s || 8;
    return `<path d="M${(X - s).toFixed(1)},${Y.toFixed(1)}L${X.toFixed(1)},${(Y - s).toFixed(1)}L${(X + s).toFixed(1)},${Y.toFixed(1)}L${X.toFixed(1)},${(Y + s).toFixed(1)}Z" style="${dash ? `fill:none;stroke:${c};stroke-width:1.6;stroke-dasharray:3 2` : `fill:${c};stroke:var(--text);stroke-width:1.6`}"/>`;
  };
  const pathOf = (S, pts) => pts.map(([x, y], i) => (i ? "L" : "M") + S.sx(x).toFixed(1) + "," + S.sy(y).toFixed(1)).join("");
  const curve = (pts, c, w, dash) => (S) => `<path d="${pathOf(S, pts)}" style="fill:none;stroke:${c};stroke-width:${w || 2.4};stroke-linejoin:round;stroke-linecap:round${dash ? ";stroke-dasharray:" + dash : ""}"/>`;
  const area = (pts, c, p) => (S) => `<path d="${pathOf(S, pts)}Z" style="fill:${soft(c, p || 16)};stroke:none"/>`;
  const segPx = (S, x1, y1, x2, y2, style, arrow, shrink) => {
    let X1 = S.sx(x1), Y1 = S.sy(y1), X2 = S.sx(x2), Y2 = S.sy(y2);
    if (shrink) { const L = Math.hypot(X2 - X1, Y2 - Y1) || 1; X2 -= ((X2 - X1) / L) * shrink; Y2 -= ((Y2 - Y1) / L) * shrink; }
    return `<line x1="${X1.toFixed(1)}" y1="${Y1.toFixed(1)}" x2="${X2.toFixed(1)}" y2="${Y2.toFixed(1)}" style="${style}"${arrow ? ' marker-end="url(#ARROW)"' : ""}/>`;
  };
  const cross = (S, x, y, c) => { const X = S.sx(x), Y = S.sy(y); return `<path d="M${X - 4},${Y - 4}L${X + 4},${Y + 4}M${X + 4},${Y - 4}L${X - 4},${Y + 4}" style="stroke:${c || "var(--muted)"};stroke-width:2;stroke-linecap:round"/>`; };
  const txt = (x, y, s, o) => { o = o || {}; return `<text x="${(+x).toFixed(1)}" y="${(+y).toFixed(1)}" text-anchor="${o.a || "middle"}" style="font-size:${o.fs || 12}px;fill:${o.c || "var(--text)"}${o.b ? ";font-weight:700" : ""}"${o.rot ? ` transform="rotate(${o.rot} ${(+x).toFixed(1)} ${(+y).toFixed(1)})"` : ""}>${esc(s)}</text>`; };
  // a legend that wraps onto a second line if needed: items = [[label, colour, dash?]]
  const legend = (S, items, y) => {
    let x = S.P.l, yy = y, s = "";
    items.forEach(([lab, c, dash]) => {
      const w = 30 + lab.length * 6.2;
      if (x + w > S.W - 4 && x > S.P.l) { x = S.P.l; yy += 16; }
      s += `<line x1="${x}" y1="${yy}" x2="${x + 18}" y2="${yy}" style="stroke:${c};stroke-width:3${dash ? ";stroke-dasharray:5 3" : ""}"/>` + txt(x + 23, yy + 4, lab, { a: "start", fs: 11.5, c: "var(--muted)" });
      x += w + 8;
    });
    return s;
  };
  const svgWrap = (W, H, inner) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img"><defs><marker id="ml2ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z"/></marker></defs>${inner}</svg>`;
  const box = (x, y, w, h, c, p, r) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r == null ? 7 : r}" style="fill:${soft(c, p || 14)};stroke:${c};stroke-width:1.5"/>`;
  const arrowLine = (x1, y1, x2, y2, c) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" style="stroke:${c || "var(--muted)"};stroke-width:1.8" marker-end="url(#ml2ah)"/>`;

  /* a 2×2 confusion matrix picture (textbook layout: rows = actual, columns = predicted) */
  function cmSvg(TP, FN, FP, TN, title) {
    const W = 290, H = 214, x0 = 104, y0 = 52, cw = 88, ch = 66;
    const cell = (x, y, n, lab, c) => `<rect x="${x}" y="${y}" width="${cw - 4}" height="${ch - 4}" rx="7" style="fill:${soft(c, 20)};stroke:${c};stroke-width:1.5"/>` +
      txt(x + (cw - 4) / 2, y + 31, String(n), { fs: 20, b: 1 }) + txt(x + (cw - 4) / 2, y + 50, lab, { fs: 11, c: "var(--muted)" });
    return svgWrap(W, H,
      (title ? txt(W / 2, 14, title, { fs: 12, b: 1, c: "var(--muted)" }) : "") +
      txt(x0 + cw / 2 - 2, y0 - 10, "predicted +", { fs: 11.5, c: "var(--muted)" }) + txt(x0 + cw * 1.5 - 2, y0 - 10, "predicted −", { fs: 11.5, c: "var(--muted)" }) +
      txt(x0 - 10, y0 + ch / 2 + 2, "actual +", { a: "end", fs: 11.5, c: "var(--muted)" }) + txt(x0 - 10, y0 + ch * 1.5 + 2, "actual −", { a: "end", fs: 11.5, c: "var(--muted)" }) +
      cell(x0, y0, TP, "TP", "var(--green)") + cell(x0 + cw, y0, FN, "FN (missed)", "var(--amber)") +
      cell(x0, y0 + ch, FP, "FP (false alarm)", "var(--red)") + cell(x0 + cw, y0 + ch, TN, "TN", "var(--accent)"));
  }

  /* ------------------------------------------------------------ k-means core */
  function assign(X, C) { return X.map((p) => { let b = 0, bd = Infinity; C.forEach((c, j) => { const d = d2(p, c); if (d < bd) { bd = d; b = j; } }); return b; }); }
  function update(X, lab, C) { return C.map((c, j) => { let sx = 0, sy = 0, n = 0; X.forEach((p, i) => { if (lab[i] === j) { sx += p[0]; sy += p[1]; n++; } }); return n ? [sx / n, sy / n] : c.slice(); }); }
  function inertia(X, lab, C) { return X.reduce((s, p, i) => s + d2(p, C[lab[i]]), 0); }
  function kmeansRun(X, k, r) {                        // k-means++ start, then Lloyd's iterations
    const C = [X[Math.floor(r() * X.length)].slice()];
    while (C.length < k) {
      const D = X.map((p) => Math.min(...C.map((c) => d2(p, c)))), tot = D.reduce((a, b) => a + b, 0);
      let u = r() * tot, i = 0;
      while (u > D[i] && i < X.length - 1) { u -= D[i]; i++; }
      C.push(X[i].slice());
    }
    let lab = assign(X, C), M = C;
    for (let t = 0; t < 100; t++) { M = update(X, lab, M); const nl = assign(X, M); if (nl.every((v, i) => v === lab[i])) break; lab = nl; }
    return { lab, C: M, J: inertia(X, lab, M) };
  }
  function kmeansBest(X, k, seed, n) { const r = rng(seed); let best = null; for (let i = 0; i < (n || 8); i++) { const m = kmeansRun(X, k, r); if (!best || m.J < best.J - 1e-9) best = m; } return best; }

  const BLOBS = (() => {                               // 4 blobs of 15 points in the box [0, 10]²
    const r = rng(7), C = [[2.3, 2.5], [7.5, 2.7], [3.4, 7.6], [7.9, 7.8]], pts = [];
    for (let i = 0; i < 15; i++) C.forEach((c) => pts.push([clamp(c[0] + gauss(r) * 0.9, 0.3, 9.7), clamp(c[1] + gauss(r) * 0.9, 0.3, 9.7)]));
    return pts;
  })();

  /* ============================================================ k-means visualizer */
  V.ml2_kmeans = {
    title: "k-Means: assign every point, then move every centroid",
    params: [
      { id: "k", label: "clusters k", min: 2, max: 5, step: 1, value: 3 },
      { id: "seed", label: "random start", min: 1, max: 12, step: 1, value: 1 },
    ],
    gen(_, P) {
      const X = BLOBS, k = P.k, r = rng(1000 * P.seed + 17 * k), S = [], idx = [];
      while (idx.length < k) { const i = Math.floor(r() * X.length); if (!idx.includes(i)) idx.push(i); }
      let C = idx.map((i) => X[i].slice()), lab = null, J = null;
      const best = kmeansBest(X, k, 99, 12).J;
      const draw = (o) => chart({ ...SQ, x: [0, 10], y: [0, 10], xl: "feature x₁", yl: "feature x₂" }, [
        (Sc) => (o.lines && lab ? X.map((p, i) => segPx(Sc, p[0], p[1], C[lab[i]][0], C[lab[i]][1], `stroke:${col(lab[i])};stroke-width:1;opacity:.45`)).join("") : ""),
        circles(X.map((p, i) => [p[0], p[1], lab ? fillDot(col(lab[i])) : "fill:var(--muted);opacity:.6"])),
        (Sc) => (o.old || []).map((c, j) => diamond(Sc, c[0], c[1], col(j), 8, true) + (d2(c, C[j]) > 1e-4 ? segPx(Sc, c[0], c[1], C[j][0], C[j][1], "stroke:var(--text);stroke-width:1.6", true, 9) : "")).join(""),
        (Sc) => C.map((c, j) => diamond(Sc, c[0], c[1], col(j), 8.5)).join(""),
      ]);
      S.push({ svg: draw({}), side: { k, iteration: 0, inertia: "–" },
        msg: `Start: ${k} data points picked at random become the first **centroids** (the diamonds). No point belongs to a cluster yet.` });
      let it = 1, done = false;
      for (; it <= 25; it++) {
        const nl = assign(X, C), changed = lab ? nl.filter((v, i) => v !== lab[i]).length : X.length;
        lab = nl; J = inertia(X, lab, C);
        if (changed === 0) { done = true; break; }
        S.push({ svg: draw({ lines: true }), side: { k, iteration: it, step: "assign", inertia: fx(J, 1), "points that switched": changed },
          msg: it === 1 ? `**Assign**: every point joins its nearest centroid (the thin lines). Inertia, the sum of squared distances from each point to its centroid, is **${fx(J, 1)}**.`
            : `**Assign** (iteration ${it}): with the centroids in their new places, **${changed}** point${changed > 1 ? "s" : ""} found a nearer centroid and switched. Inertia is now **${fx(J, 1)}**.` });
        const old = C, counts = C.map((_, j) => lab.filter((v) => v === j).length);
        C = update(X, lab, C);
        const J2 = inertia(X, lab, C), empty = counts.filter((n) => n === 0).length;
        S.push({ svg: draw({ old }), side: { k, iteration: it, step: "update", inertia: fx(J2, 1) },
          msg: `**Update**: each centroid jumps to the **mean** of its points (arrows). Inertia falls from ${fx(J, 1)} to **${fx(J2, 1)}**: both steps can only lower it.` + (empty ? ` (${empty} centroid${empty > 1 ? "s have" : " has"} no points, so it stays where it is.)` : "") });
        J = J2;
      }
      S.push({ svg: draw({ lines: true }), side: { k, iterations: it - 1, inertia: fx(J, 1), "best of 12 starts": fx(best, 1) },
        msg: (done ? `**Converged** after ${it - 1} updates: no point switched, so the centroids would not move again. Final inertia **${fx(J, 1)}**. ` : `Stopped after 25 iterations. `) +
          (J - best < 0.05 ? "That is the best this k can do (it matches the best of 12 k-means++ runs). " : `Another start reaches **${fx(best, 1)}**: k-means only finds a *local* minimum, which is why libraries run several starts (\`n_init\`). Try other random starts. `) +
          (k === 4 ? "" : "The data has 4 blobs: try k = 4.") });
      return S;
    },
  };

  /* ============================================================ DBSCAN visualizer */
  const MOONS = (() => {                               // two interleaved moons, a small blob and 7 lonely points
    const r = rng(11), pts = [];
    for (let i = 0; i < 30; i++) { const t = (Math.PI * (i + 0.5)) / 30; pts.push([2.6 + 2.4 * Math.cos(t) + gauss(r) * 0.15, 4.0 + 2.4 * Math.sin(t) + gauss(r) * 0.15]); }
    for (let i = 0; i < 30; i++) { const t = (Math.PI * (i + 0.5)) / 30; pts.push([2.6 + 2.4 * (1 - Math.cos(t)) + gauss(r) * 0.15, 4.0 + 2.4 * (0.5 - Math.sin(t)) + gauss(r) * 0.15]); }
    for (let i = 0; i < 12; i++) pts.push([8.3 + gauss(r) * 0.4, 8.0 + gauss(r) * 0.4]);
    [[0.8, 8.9], [5.0, 8.7], [9.3, 1.0], [6.4, 7.3], [0.6, 1.0], [9.4, 4.8], [4.4, 1.0]].forEach((p) => pts.push(p));
    return pts;
  })();
  function dbscan(X, eps, minPts) {
    const n = X.length, e2 = eps * eps;
    const nb = X.map((p) => X.map((q, j) => (d2(p, q) <= e2 ? j : -1)).filter((j) => j >= 0));   // includes the point itself
    const core = nb.map((a) => a.length >= minPts), lab = new Array(n).fill(-1), layers = [];
    let c = -1;
    for (let i = 0; i < n; i++) {
      if (lab[i] !== -1 || !core[i]) continue;
      c++; lab[i] = c; layers.push({ c, pts: [i], from: [], seed: i });
      let frontier = [i];
      while (frontier.length) {
        const next = [], from = frontier.filter((p) => core[p]);
        from.forEach((p) => nb[p].forEach((q) => { if (lab[q] === -1) { lab[q] = c; next.push(q); } }));
        if (next.length) layers.push({ c, pts: next, from });
        frontier = next;
      }
    }
    return { nb, core, lab, layers, nc: c + 1 };
  }
  V.ml2_dbscan = {
    title: "DBSCAN: grow clusters from dense points",
    slide: "end",
    params: [
      { id: "eps", label: "eps (radius)", min: 0.3, max: 2, step: 0.05, value: 0.8 },
      { id: "m", label: "minPts", min: 2, max: 8, step: 1, value: 4 },
    ],
    gen(_, P) {
      const X = MOONS, eps = P.eps, m = P.m, R = dbscan(X, eps, m), S = [];
      const nCore = R.core.filter(Boolean).length;
      const isBorder = (i) => R.lab[i] >= 0 && !R.core[i];
      const nBorder = X.filter((_, i) => isBorder(i)).length, nNoise = R.lab.filter((v) => v < 0).length;
      const draw = (lab, o) => chart({ ...SQ, x: [0, 10], y: [0, 10], xl: "feature x₁", yl: "feature x₂" }, [
        (Sc) => { const rr = Sc.sx(eps) - Sc.sx(0); return (o.discs ? X.map((p) => `<circle cx="${Sc.sx(p[0]).toFixed(1)}" cy="${Sc.sy(p[1]).toFixed(1)}" r="${rr.toFixed(1)}" style="fill:${soft("var(--accent)", 7)};stroke:none"/>`).join("") : "") +
          (o.rings || []).map((i) => `<circle cx="${Sc.sx(X[i][0]).toFixed(1)}" cy="${Sc.sy(X[i][1]).toFixed(1)}" r="${rr.toFixed(1)}" style="fill:${soft(col(lab[i]), 8)};stroke:${col(lab[i])};stroke-width:1.2;stroke-dasharray:4 3"/>`).join(""); },
        (Sc) => X.map((p, i) => {
          const x = Sc.sx(p[0]).toFixed(1), y = Sc.sy(p[1]).toFixed(1);
          if (o.final && lab[i] < 0) return cross(Sc, p[0], p[1]);
          if (lab[i] < 0) return R.core[i] ? `<circle cx="${x}" cy="${y}" r="4.5" style="fill:var(--text);opacity:.7"/>` : `<circle cx="${x}" cy="${y}" r="3.6" style="fill:none;stroke:var(--muted);stroke-width:1.4"/>`;
          const c = col(lab[i]), fresh = (o.fresh || []).includes(i);
          if (R.core[i]) return `<circle cx="${x}" cy="${y}" r="${fresh ? 6.5 : 5.2}" style="${fillDot(c)}"/>`;
          return `<circle cx="${x}" cy="${y}" r="${fresh ? 5 : 3.8}" style="fill:${soft(c, 45)};stroke:${c};stroke-width:1.6"/>`;
        }).join(""),
      ]);
      const side = (o) => Object.assign({ eps: fx(eps), minPts: m }, o);
      S.push({ svg: draw(R.lab.map(() => -1), { discs: true }), side: side({ "core points": nCore }),
        msg: `Every point counts the points within **eps = ${fx(eps)}** of it (the shaded discs), itself included. **${nCore}** of the ${X.length} points have at least minPts = ${m} and are **core points** (filled); the hollow ones are not dense enough to start a cluster.` });
      // group the growth layers so there are at most ~30 steps
      const L = R.layers, g = Math.max(1, Math.ceil(L.length / 30)), lab = R.lab.map(() => -1);
      for (let s = 0; s < L.length; s += g) {
        const chunk = L.slice(s, s + g), fresh = [], rings = [];
        chunk.forEach((ly) => { ly.pts.forEach((i) => { lab[i] = ly.c; fresh.push(i); }); ly.from.forEach((i) => rings.push(i)); });
        const first = chunk[0], cs = [...new Set(chunk.map((ly) => ly.c + 1))];
        const nb = fresh.filter((i) => !R.core[i]).length;
        const msg = g === 1 && first.from.length === 0
          ? `**Cluster ${first.c + 1}** starts at the first unvisited core point. Everything within eps of a core point joins its cluster.`
          : `Cluster${cs.length > 1 ? "s " + cs.join(", ") : " " + cs[0]} grow${cs.length > 1 ? "" : "s"}: the core points on the edge (dashed circles) reach **${fresh.length}** new point${fresh.length > 1 ? "s" : ""}.` +
            (nb ? ` ${nb} of them ${nb > 1 ? "are" : "is a"} **border point${nb > 1 ? "s" : ""}** (in reach, but not dense themselves), so growth stops there.` : "");
        S.push({ svg: draw(lab.slice(), { rings: rings.slice(0, 40), fresh }), side: side({ clusters: Math.max(...lab) + 1 }), msg });
      }
      const hint = R.nc === 0 ? "eps is so small that no point is dense: everything is noise. Make eps bigger." :
        nNoise > X.length / 3 ? "Lots of noise: eps is on the small side (or minPts too big)." :
        R.nc === 1 ? "Everything merged into one cluster: eps is too big, so the gaps between groups are bridged." :
        R.nc <= 3 && nNoise <= 9 ? "Both moons and the blob come out as clusters, shapes and all, which k-means cannot do." : "Some groups break into pieces: try a slightly larger eps.";
      S.push({ svg: draw(R.lab, { final: true }), side: side({ clusters: R.nc, core: nCore, border: nBorder, noise: nNoise }),
        msg: `Done: **${R.nc}** cluster${R.nc === 1 ? "" : "s"}, ${nCore} core points (big dots), ${nBorder} border points (rings) and **${nNoise} noise** points (×), which belong to no cluster. ${hint}` });
      return S;
    },
  };

  /* ============================================================ PCA visualizer */
  const PCA_PTS = (() => {
    const r = rng(5), a = (32 * Math.PI) / 180, pts = [];
    for (let i = 0; i < 40; i++) { const u = gauss(r) * 1.8, v = gauss(r) * 0.6; pts.push([u * Math.cos(a) - v * Math.sin(a), u * Math.sin(a) + v * Math.cos(a)]); }
    const mx = mean(pts.map((p) => p[0])), my = mean(pts.map((p) => p[1]));
    return pts.map((p) => [p[0] - mx, p[1] - my]);
  })();
  const angDiff = (x, y) => { const d = (((x - y) % 180) + 180) % 180; return Math.min(d, 180 - d); };
  V.ml2_pca = {
    title: "PCA: find the direction with the most spread",
    params: [{ id: "deg", label: "direction θ (degrees)", min: 0, max: 179, step: 1, value: 110 }],
    gen(_, P) {
      const X = PCA_PTS, n = X.length;
      const a = X.reduce((s, p) => s + p[0] * p[0], 0) / (n - 1), c = X.reduce((s, p) => s + p[1] * p[1], 0) / (n - 1), b = X.reduce((s, p) => s + p[0] * p[1], 0) / (n - 1);
      const tot = a + c, disc = Math.sqrt(((a - c) / 2) ** 2 + b * b), l1 = tot / 2 + disc, l2 = tot / 2 - disc;
      let phi = (0.5 * Math.atan2(2 * b, a - c) * 180) / Math.PI; if (phi < 0) phi += 180;
      const vAt = (deg) => { const t = (deg * Math.PI) / 180, cs = Math.cos(t), sn = Math.sin(t); return a * cs * cs + 2 * b * cs * sn + c * sn * sn; };
      const th = (P.deg * Math.PI) / 180, u = [Math.cos(th), Math.sin(th)], v = vAt(P.deg);
      const proj = X.map((p) => { const s = p[0] * u[0] + p[1] * u[1]; return [s * u[0], s * u[1]]; });
      const through = (deg, style) => (S) => { const t = (deg * Math.PI) / 180; return segPx(S, -9 * Math.cos(t), -9 * Math.sin(t), 9 * Math.cos(t), 9 * Math.sin(t), style); };
      const tagAt = (S, deg, s, c) => { const t = (deg * Math.PI) / 180, R = 4.55; let x = R * Math.cos(t), y = R * Math.sin(t); if (y < -4.4) { x = -x; y = -y; } return txt(S.sx(x), S.sy(y) + 4, s, { fs: 12, b: 1, c }); };
      const left = chart({ ...SQ, x: [-5, 5], y: [-5, 5], xl: "feature 1 (centred)", yl: "feature 2 (centred)" }, [
        through(phi, "stroke:var(--green);stroke-width:1.6;stroke-dasharray:6 4"),
        through(phi + 90, "stroke:var(--purple);stroke-width:1.6;stroke-dasharray:6 4"),
        (S) => X.map((p, i) => segPx(S, p[0], p[1], proj[i][0], proj[i][1], "stroke:var(--muted);stroke-width:1;opacity:.6")).join(""),
        through(P.deg, "stroke:var(--amber);stroke-width:2.6"),
        circles(X.map((p) => [p[0], p[1], fillDot("var(--accent)")]), 4.3),
        circles(proj.map((p) => [p[0], p[1], "fill:var(--amber);stroke:var(--panel);stroke-width:.8"]), 3),
      ], (S) => tagAt(S, phi, "PC1", "var(--green)") + tagAt(S, phi + 90, "PC2", "var(--purple)"));
      const right = chart({ W: 250, H: 394, x: [0, 180], y: [0, l1 * 1.18], xl: "direction θ (°)", yl: "variance of the shadows", ticks: 4 }, [
        Plot.fn(vAt, "pl-c2"),
        Plot.vline(P.deg),
        Plot.dot(phi, l1, "pl-c3", 5), Plot.dot((phi + 90) % 180, l2, "pl-c5", 5),
        Plot.dot(P.deg, v, "pl-c2", 6.5),
      ], (S) => txt(S.sx(phi), S.sy(l1) - 9, "PC1", { b: 1, c: "var(--green)" }) + txt(S.sx((phi + 90) % 180), S.sy(l2) - 9, "PC2", { b: 1, c: "var(--purple)" }));
      const near = angDiff(P.deg, phi), near2 = angDiff(P.deg, phi + 90);
      const msg = near <= 1.5 ? `You found **PC1**: the shadows are as spread out as they can be. Their variance is ${fx(v)}, the largest eigenvalue λ₁ = ${fx(l1)} of the covariance matrix. Keeping just this one number per point keeps **${pct(l1 / tot)}** of the total variance (${fx(tot)}).`
        : near2 <= 1.5 ? `This is **PC2**, at right angles to PC1: the shadows are squeezed together (variance ${fx(v)} = λ₂, the smallest eigenvalue). Dropping this direction loses only ${pct(l2 / tot)} of the variance.`
        : `Every point drops a shadow (amber) onto the line at θ = ${P.deg}°. The shadows' variance is **${fx(v)}**, which is ${pct(v / tot)} of the total. Rotate the line to spread the shadows as much as possible: the best direction is PC1 (green, θ ≈ ${phi.toFixed(0)}°). The right chart shows this variance for every angle.`;
      return [{ svg: row(left, right), side: { "θ": P.deg + "°", "variance along θ": fx(v), "λ₁ (PC1)": fx(l1) + " at " + phi.toFixed(0) + "°", "λ₂ (PC2)": fx(l2), "share of total": pct(v / tot) }, msg }];
    },
  };

  /* ============================================================ threshold → confusion matrix, precision / recall, ROC point */
  const SCORES = (() => {                              // 20 negatives and 20 positives with model scores in (0, 1)
    const r = rng(21), sig = (z) => 1 / (1 + Math.exp(-z)), out = [];
    for (let i = 0; i < 20; i++) out.push({ s: +sig(gauss(r) * 1.25 - 1.0).toFixed(2), y: 0, j: r() });
    for (let i = 0; i < 20; i++) out.push({ s: +sig(gauss(r) * 1.25 + 1.0).toFixed(2), y: 1, j: r() });
    return out;
  })();
  function rocPoints(D) {                              // empirical ROC: one point per distinct threshold (rule: score ≥ t → positive)
    const P = D.filter((d) => d.y).length, N = D.length - P, ts = [...new Set(D.map((d) => d.s))].sort((a, b) => b - a), pts = [[0, 0]];
    ts.forEach((t) => { const tp = D.filter((d) => d.y && d.s >= t).length, fp = D.filter((d) => !d.y && d.s >= t).length; pts.push([fp / N, tp / P]); });
    let auc = 0; for (let i = 1; i < pts.length; i++) auc += ((pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1])) / 2;
    return { pts, auc };
  }
  V.ml2_threshold = {
    title: "Move the decision threshold",
    params: [{ id: "t", label: "threshold", min: 0, max: 1, step: 0.01, value: 0.5 }],
    gen(_, P) {
      const t = P.t, D = SCORES, roc = rocPoints(D);
      let TP = 0, FP = 0, FN = 0, TN = 0;
      D.forEach((d) => { const p = d.s >= t; if (p && d.y) TP++; else if (p) FP++; else if (d.y) FN++; else TN++; });
      const prec = TP + FP ? TP / (TP + FP) : null, rec = TP / (TP + FN), fpr = FP / (FP + TN), acc = (TP + TN) / D.length;
      const f1 = prec != null && prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
      const outc = (d) => (d.s >= t ? (d.y ? "var(--green)" : "var(--red)") : d.y ? "var(--amber)" : "var(--accent)");
      const strip = chart({ W: 620, H: 168, x: [0, 1], y: [0, 1], xl: "model score (predicted probability of positive)", grid: false, pad: { l: 104, r: 16, t: 24, b: 40 } }, [
        (S) => `<rect x="${S.sx(t).toFixed(1)}" y="${S.P.t}" width="${Math.max(0, S.sx(1) - S.sx(t)).toFixed(1)}" height="${S.H - S.P.t - S.P.b}" style="fill:${soft("var(--text)", 7)}"/>`,
        Plot.hline(0.5, "pl-dash"),
        circles(D.map((d) => [d.s, (d.y ? 0.73 : 0.27) + (d.j - 0.5) * 0.3, fillDot(outc(d))]), 5),
        (S) => segPx(S, t, 0, t, 1, "stroke:var(--text);stroke-width:2"),
      ], (S) => [0, 0.2, 0.4, 0.6, 0.8, 1].map((v) => txt(S.sx(v), S.H - S.P.b + 15, String(v), { fs: 11, c: "var(--muted)" })).join("") +
        txt(S.P.l - 8, S.sy(0.73) + 4, "actual positive", { a: "end", fs: 11.5, c: "var(--muted)" }) + txt(S.P.l - 8, S.sy(0.27) + 4, "actual negative", { a: "end", fs: 11.5, c: "var(--muted)" }) +
        txt(S.sx(t) - 6, S.P.t - 8, "← predict negative", { a: "end", fs: 11, c: "var(--muted)" }) + txt(S.sx(t) + 6, S.P.t - 8, "predict positive →", { a: "start", fs: 11, c: "var(--muted)" }));
      const rocC = chart({ W: 290, H: 250, x: [0, 1], y: [0, 1], xl: "false positive rate", yl: "true positive rate (recall)", ticks: 5 }, [
        area(roc.pts.concat([[1, 0]]), "var(--accent)", 12),
        Plot.seg(0, 0, 1, 1, "pl-dash"),
        curve(roc.pts.concat([[1, 1]]), "var(--accent)", 2.4),
        Plot.dot(fpr, rec, "pl-c2", 6.5),
      ]);
      const hint = t >= 0.75 ? "A strict threshold: very few false alarms, but many positives are missed." :
        t <= 0.25 ? "A lenient threshold: almost every positive is caught, at the price of many false alarms." :
        "Moving the threshold right trades recall for precision; the amber dot on the ROC curve slides down and left.";
      return [{ svg: strip + row(cmSvg(TP, FN, FP, TN), rocC),
        side: { threshold: fx(t), accuracy: fx(acc, 3), precision: prec == null ? "n/a" : fx(prec, 3), "recall (TPR)": fx(rec, 3), F1: fx(f1, 3), FPR: fx(fpr, 3), "ROC-AUC": fx(roc.auc, 3) },
        msg: `Scores ≥ **${fx(t)}** are called positive: ${TP + FP} of the 40 examples. ` +
          (prec == null ? "Nothing is predicted positive, so precision is undefined (scikit-learn reports 0 with a warning) and recall is 0. " :
            `Precision = TP / (TP + FP) = ${TP}/${TP + FP} = **${fx(prec, 2)}**; recall = TP / (TP + FN) = ${TP}/${TP + FN} = **${fx(rec, 2)}**. `) + hint }];
    },
  };

  /* ============================================================ ROC vs PR: separation and class balance */
  function erfc(x) {                                   // Numerical Recipes erfcc: fractional error < 1.2e-7 everywhere
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }
  const Q = (x) => 0.5 * erfc(x / Math.SQRT2);         // P(Z > x) for a standard normal Z
  const phiN = (x) => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
  V.ml2_roc = {
    title: "ROC curve vs precision–recall curve",
    params: [
      { id: "d", label: "class separation", min: 0, max: 4, step: 0.1, value: 1.5 },
      { id: "pi", label: "positives (%)", min: 1, max: 50, step: 1, value: 10 },
    ],
    gen(_, P) {
      const d = P.d, pi = P.pi / 100, T = [];
      for (let i = 0; i <= 600; i++) T.push(d + 8 - (i * (16 + d)) / 600);      // thresholds from high to low
      const rocP = [[0, 0]], prP = [];
      let ap = 0, lastR = 0;
      T.forEach((t) => {
        const tpr = Q(t - d), fpr = Q(t), prec = (pi * tpr) / (pi * tpr + (1 - pi) * fpr || 1);
        rocP.push([fpr, tpr]);
        if (tpr > 1e-4) prP.push([tpr, prec]);
        ap += (tpr - lastR) * prec; lastR = tpr;
      });
      rocP.push([1, 1]);
      const auc = 1 - Q(d / Math.SQRT2);                // = Φ(d / √2) for two unit-variance normals
      const dens = chart({ W: 620, H: 150, x: [-4, 8], y: [0, 0.46], xl: "model score", yl: "density", yticks: 2 }, [
        area([[-4, 0]].concat(Array.from({ length: 121 }, (_, i) => { const x = -4 + i * 0.1; return [x, phiN(x)]; }), [[8, 0]]), "var(--accent)", 16),
        area([[-4, 0]].concat(Array.from({ length: 121 }, (_, i) => { const x = -4 + i * 0.1; return [x, phiN(x - d)]; }), [[8, 0]]), "var(--amber)", 18),
        Plot.fn(phiN, "pl-c1"), Plot.fn((x) => phiN(x - d), "pl-c2"),
      ], (S) => txt(S.sx(-0.35), S.sy(0.38), "negatives", { a: "end", fs: 11.5, b: 1, c: "var(--accent)" }) + txt(S.sx(d + 0.35), S.sy(0.38), "positives", { a: "start", fs: 11.5, b: 1, c: "var(--amber)" }));
      const rocC = chart({ W: 300, H: 260, x: [0, 1], y: [0, 1], xl: "false positive rate", yl: "true positive rate", ticks: 5 }, [
        area(rocP.concat([[1, 0]]), "var(--accent)", 14), Plot.seg(0, 0, 1, 1, "pl-dash"), curve(rocP, "var(--accent)", 2.6),
        Plot.text(0.62, 0.2, "AUC = " + fx(auc, 3)),
      ]);
      const prC = chart({ W: 300, H: 260, x: [0, 1], y: [0, 1.02], xl: "recall", yl: "precision", ticks: 5 }, [
        area([[0, 0]].concat(prP, [[1, 0]]), "var(--amber)", 14), Plot.hline(pi, "pl-dash"), curve(prP, "var(--amber)", 2.6),
        Plot.text(0.5, Math.min(0.92, pi + 0.07), "baseline = " + fx(pi, 2)), Plot.text(0.3, 0.08 + (pi > 0.3 ? 0 : 0), "AP = " + fx(ap, 3)),
      ]);
      return [{ svg: dens + row(rocC, prC), side: { "ROC-AUC": fx(auc, 3), "AP (area under PR)": fx(ap, 3), "PR baseline (positive share)": fx(pi, 2) },
        msg: (d === 0 ? "With no separation the scores are useless: ROC is the diagonal (AUC 0.5) and precision equals the positive share at every threshold. "
          : `Separation ${fx(d, 1)}: a random positive outscores a random negative with probability **${fx(auc, 3)}** (that is what ROC-AUC means). `) +
          `The ROC curve does not depend on the class balance at all, but precision does: with **${P.pi}% positives** the average precision is **${fx(ap, 3)}** against a baseline of ${fx(pi, 2)}. Lower the positive share and watch the PR curve sink while the ROC curve stays put.` }];
    },
  };

  /* ============================================================ k-fold cross-validation */
  const KF = (() => { const r = rng(31), pts = []; for (let i = 0; i < 20; i++) { const x = 0.5 + r() * 9; pts.push([x, 1 + 1.1 * x + gauss(r) * 1.6]); } return pts; })();
  function linfit(P) { const mx = mean(P.map((p) => p[0])), my = mean(P.map((p) => p[1])); let sxy = 0, sxx = 0; P.forEach(([x, y]) => { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; }); const w = sxy / sxx; return [w, my - w * mx]; }
  V.ml2_kfold = {
    title: "k-fold cross-validation, one fold at a time",
    params: [{ id: "k", label: "folds k", min: 2, max: 10, step: 1, value: 5 }],
    gen(_, P) {
      const n = KF.length, k = P.k, base = Math.floor(n / k), extra = n % k, fold = [], S = [];
      for (let f = 0, i = 0; f < k; f++) { const sz = base + (f < extra ? 1 : 0); for (let j = 0; j < sz; j++) fold[i++] = f; }
      const res = [];
      for (let f = 0; f < k; f++) {
        const tr = KF.filter((_, i) => fold[i] !== f), va = KF.filter((_, i) => fold[i] === f), [w, b] = linfit(tr);
        res.push({ w, b, mse: mean(va.map(([x, y]) => (y - (w * x + b)) ** 2)), nv: va.length });
      }
      const ys = KF.map((p) => p[1]), y0 = Math.floor(Math.min(...ys) - 1), y1 = Math.ceil(Math.max(...ys) + 1);
      const grid = (cur) => {                         // cur: -1 = intro, 0..k-1 = that fold, k = summary
        const cw = 21, x0 = 76, top = 26, rh = 22, H = top + k * rh + (cur === k ? 30 : 8);
        let s = KF.map((_, i) => txt(x0 + i * cw + 10, 15, String(i), { fs: 10, c: "var(--muted)" })).join("");
        for (let f = 0; f < k; f++) {
          const y = top + f * rh, state = cur === -1 ? "todo" : f < cur || cur === k ? "done" : f === cur ? "cur" : "todo";
          s += `<g style="opacity:${state === "todo" ? 0.35 : 1}">` + txt(x0 - 10, y + 13, "fold " + (f + 1), { a: "end", fs: 11.5, b: state === "cur" ? 1 : 0, c: state === "cur" ? "var(--text)" : "var(--muted)" });
          for (let i = 0; i < n; i++) {
            const isVal = fold[i] === f;
            s += `<rect x="${x0 + i * cw}" y="${y}" width="${cw - 3}" height="${rh - 5}" rx="3" style="fill:${isVal ? soft("var(--amber)", 75) : soft("var(--accent)", 20)};stroke:${isVal ? "var(--amber)" : soft("var(--accent)", 55)};stroke-width:1.2"/>`;
          }
          if (state === "done" || state === "cur") s += txt(x0 + n * cw + 10, y + 13, "MSE " + fx(res[f].mse), { a: "start", fs: 11.5, b: state === "cur" ? 1 : 0 });
          s += "</g>";
        }
        if (cur === k) { const ms = res.map((r) => r.mse), mu = mean(ms), sd = Math.sqrt(mean(ms.map((v) => (v - mu) ** 2))); s += txt(x0 + n * cw + 10, top + k * rh + 16, `mean ${fx(mu)} ± ${fx(sd)}`, { a: "start", fs: 12, b: 1, c: "var(--accent)" }); }
        return svgWrap(620, H, s);
      };
      const scatter = (cur) => chart({ W: 620, H: 240, x: [0, 10], y: [y0, y1], xl: "x", yl: "y" }, [
        (Sc) => (cur === k ? res.map((r) => segPx(Sc, 0, r.b, 10, r.w * 10 + r.b, "stroke:var(--accent);stroke-width:1.3;opacity:.55")).join("") : ""),
        cur >= 0 && cur < k ? Plot.fn((x) => res[cur].w * x + res[cur].b, "pl-c1") : "",
        (Sc) => (cur >= 0 && cur < k ? KF.map((p, i) => (fold[i] === cur ? segPx(Sc, p[0], p[1], p[0], res[cur].w * p[0] + res[cur].b, "stroke:var(--red);stroke-width:1.6") : "")).join("") : ""),
        circles(KF.map((p, i) => [p[0], p[1], cur < 0 || cur === k ? fillDot("var(--muted)") : fold[i] === cur ? fillDot("var(--amber)") : fillDot("var(--accent)"), cur >= 0 && cur < k && fold[i] === cur ? 6.5 : 4.5])),
      ]);
      const sizes = [...new Set(res.map((r) => r.nv))].join(" or ");
      S.push({ svg: grid(-1) + scatter(-1), side: { k, examples: n, "per fold": sizes },
        msg: `${n} examples (already shuffled, numbered 0–${n - 1}) are cut into **k = ${k} folds** of ${sizes}. Each fold will take one turn as the **validation set** (amber) while the model trains on all the others (blue).` });
      for (let f = 0; f < k; f++) {
        S.push({ svg: grid(f) + scatter(f), side: { fold: `${f + 1} / ${k}`, train: n - res[f].nv, validate: res[f].nv, "validation MSE": fx(res[f].mse) },
          msg: `**Fold ${f + 1}**: fit the line on the ${n - res[f].nv} blue examples only, then score it on the ${res[f].nv} held-out amber ones it has never seen. Mean squared error of the red gaps: **${fx(res[f].mse)}**.` });
      }
      const ms = res.map((r) => r.mse), mu = mean(ms), sd = Math.sqrt(mean(ms.map((v) => (v - mu) ** 2)));
      S.push({ svg: grid(k) + scatter(k), side: { k, "mean MSE": fx(mu), "std across folds": fx(sd), "lowest / highest": fx(Math.min(...ms)) + " / " + fx(Math.max(...ms)) },
        msg: `CV estimate: **${fx(mu)} ± ${fx(sd)}** (mean ± standard deviation of the ${k} fold scores). Every example was used for validation exactly once and for training ${k - 1} times. The ${k} fitted lines differ a little: that spread is why one single split can be lucky or unlucky.` });
      return S;
    },
  };

  /* ============================================================ class imbalance: accuracy vs F1 */
  V.ml2_imbalance = {
    title: "Class imbalance: why accuracy lies",
    params: [
      { id: "pi", label: "positives (%)", min: 0.5, max: 50, step: 0.5, value: 2 },
      { id: "tpr", label: "model recall", min: 0.3, max: 1, step: 0.01, value: 0.8 },
      { id: "fpr", label: "model false-positive rate", min: 0, max: 0.3, step: 0.01, value: 0.05 },
    ],
    gen(_, P) {
      const pi = P.pi / 100, R = P.tpr, F = P.fpr, lo = Math.log10(0.005), hi = Math.log10(0.5);
      const rates = (p) => { const tp = p * R, fp = (1 - p) * F, prec = tp + fp > 0 ? tp / (tp + fp) : 0; return { acc: tp + (1 - p) * (1 - F), prec, f1: prec + R > 0 ? (2 * prec * R) / (prec + R) : 0 }; };
      const xs = Array.from({ length: 161 }, (_, i) => Math.pow(10, lo + (i * (hi - lo)) / 160));
      const line = (f) => xs.map((p) => [Math.log10(p), f(p)]);
      const TK = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5];
      const ch = chart({ W: 400, H: 300, x: [lo, hi], y: [0, 1.02], xl: "share of positives (log scale)", grid: false, pad: { l: 44, r: 12, t: 46, b: 40 } }, [
        (S) => TK.map((v) => `<line class="pl-grid" x1="${S.sx(Math.log10(v))}" y1="${S.P.t}" x2="${S.sx(Math.log10(v))}" y2="${S.H - S.P.b}"/>`).join("") + [0.2, 0.4, 0.6, 0.8, 1].map((v) => `<line class="pl-grid" x1="${S.P.l}" y1="${S.sy(v)}" x2="${S.W - S.P.r}" y2="${S.sy(v)}"/>`).join(""),
        curve(line((p) => 1 - p), "var(--red)", 2.2, "6 4"),
        curve(line((p) => rates(p).acc), "var(--accent)", 2.4),
        curve(line((p) => rates(p).prec), "var(--purple)", 2.2),
        curve(line((p) => rates(p).f1), "var(--amber)", 3),
        Plot.vline(Math.log10(pi)),
        (S) => { const r = rates(pi), x = Math.log10(pi); return [[1 - pi, "var(--red)"], [r.acc, "var(--accent)"], [r.prec, "var(--purple)"], [r.f1, "var(--amber)"]].map(([y, c]) => `<circle cx="${S.sx(x).toFixed(1)}" cy="${S.sy(y).toFixed(1)}" r="4.5" style="${fillDot(c)}"/>`).join(""); },
      ], (S) => TK.map((v) => txt(S.sx(Math.log10(v)), S.H - S.P.b + 15, (100 * v) + "%", { fs: 10.5, c: "var(--muted)" })).join("") +
        [0, 0.5, 1].map((v) => txt(S.P.l - 6, S.sy(v) + 4, String(v), { a: "end", fs: 11, c: "var(--muted)" })).join("") +
        legend(S, [["accuracy of always 'no'", "var(--red)", 1], ["model accuracy", "var(--accent)"], ["precision", "var(--purple)"], ["F1", "var(--amber)"]], 12));
      const N = 10000, pos = Math.round(N * pi), neg = N - pos, TP = Math.round(pos * R), FN = pos - TP, FP = Math.round(neg * F), TN = neg - FP;
      const acc = (TP + TN) / N, prec = TP + FP ? TP / (TP + FP) : 0, f1 = prec + R > 0 ? (2 * prec * (TP / pos)) / (prec + TP / pos) : 0, dummy = neg / N;
      return [{ svg: row(ch, cmSvg(TP, FN, FP, TN, "the model, per 10,000 examples")),
        side: { "always-'no' accuracy": pct(dummy), "model accuracy": pct(acc), precision: fx(prec, 3), recall: fx(TP / pos, 3), F1: fx(f1, 3), "balanced accuracy": fx((TP / pos + TN / neg) / 2, 3) },
        msg: `With **${P.pi}% positives**, a "model" that always answers *no* scores **${pct(dummy)} accuracy** and catches nobody (F1 = 0). ` +
          `The real model (recall ${fx(R, 2)}, false-positive rate ${fx(F, 2)}) has accuracy ${pct(acc)}${acc < dummy ? ", **lower than the useless one**" : ""}, yet it finds ${TP} of the ${pos} positives. ` +
          `Its precision is only ${fx(prec, 2)} because its ${FP} false alarms come from a huge pool of ${neg} negatives. F1 = ${fx(f1, 2)} tells the honest story.` }];
    },
  };

  /* ============================================================ regression metrics: one big miss */
  const RM = (() => { const r = rng(41), pts = []; for (let i = 0; i < 12; i++) { const x = 0.6 + i * 0.8; pts.push([x, 1 + 0.7 * x + gauss(r) * 0.5]); } return pts; })();
  V.ml2_regmetrics = {
    title: "MAE vs RMSE: one prediction goes badly wrong",
    params: [{ id: "out", label: "extra error on one point", min: 0, max: 12, step: 0.5, value: 4 }],
    gen(_, P) {
      const pred = (x) => 1 + 0.7 * x, K = 8;
      const met = (Y) => {
        const e = Y.map(([x, y]) => y - pred(x)), ae = e.map(Math.abs).sort((a, b) => a - b), my = mean(Y.map((p) => p[1]));
        const mse = mean(e.map((v) => v * v)), sst = Y.reduce((s, p) => s + (p[1] - my) ** 2, 0);
        return { mae: mean(ae), mse, rmse: Math.sqrt(mse), med: (ae[5] + ae[6]) / 2, r2: 1 - (mse * Y.length) / sst };
      };
      const Y = RM.map((p, i) => [p[0], p[1] + (i === K ? P.out : 0)]), m0 = met(RM), m = met(Y);
      const sc = chart({ W: 380, H: 300, x: [0, 10], y: [0, 20], xl: "x", yl: "y" }, [
        Plot.fn(pred, "pl-c3"),
        (S) => Y.map((p, i) => segPx(S, p[0], p[1], p[0], pred(p[0]), `stroke:${i === K && P.out > 0 ? "var(--red)" : "var(--muted)"};stroke-width:${i === K && P.out > 0 ? 2 : 1.3}`)).join(""),
        circles(Y.map((p, i) => [p[0], p[1], fillDot(i === K && P.out > 0 ? "var(--red)" : "var(--accent)")])),
      ], (S) => txt(S.sx(9.6), S.sy(pred(9.6)) + 18, "predictions", { a: "end", fs: 11.5, c: "var(--green)", b: 1 }));
      const ymax = Math.max(2, m.rmse * 1.25), W = 230, H = 300, bt = 250, tp = 30, bw = 46;
      const bars = [["MAE", m.mae, m0.mae, "var(--accent)"], ["RMSE", m.rmse, m0.rmse, "var(--amber)"], ["median AE", m.med, m0.med, "var(--green)"]];
      const yv = (v) => bt - (v / ymax) * (bt - tp);
      let bs = `<line x1="16" y1="${bt}" x2="${W - 10}" y2="${bt}" style="stroke:var(--border)"/>`;
      bars.forEach(([lab, v, v0, c], i) => {
        const x = 26 + i * 68;
        bs += `<rect x="${x}" y="${yv(v).toFixed(1)}" width="${bw}" height="${(bt - yv(v)).toFixed(1)}" rx="4" style="fill:${soft(c, 55)};stroke:${c};stroke-width:1.5"/>` +
          `<line x1="${x - 5}" y1="${yv(v0).toFixed(1)}" x2="${x + bw + 5}" y2="${yv(v0).toFixed(1)}" style="stroke:var(--text);stroke-width:1.5;stroke-dasharray:4 3"/>` +
          txt(x + bw / 2, yv(v) - 7, fx(v), { fs: 12, b: 1 }) + txt(x + bw / 2, bt + 16, lab, { fs: 11.5, c: "var(--muted)" });
      });
      bs += txt(W / 2, 14, "dashed = without the miss", { fs: 11, c: "var(--muted)" });
      return [{ svg: row(sc, svgWrap(W, H, bs)), side: { MAE: fx(m.mae), MSE: fx(m.mse), RMSE: fx(m.rmse), "median AE": fx(m.med), "R²": fx(m.r2, 3) },
        msg: P.out === 0 ? `No big miss: every error is small, so MAE (${fx(m.mae)}) and RMSE (${fx(m.rmse)}) are close. Slide to make one of the 12 predictions badly wrong.`
          : `One of the 12 points is now off by an extra **${P.out}**. MAE went from ${fx(m0.mae)} to ${fx(m.mae)} (×${fx(m.mae / m0.mae, 1)}), RMSE from ${fx(m0.rmse)} to **${fx(m.rmse)}** (×${fx(m.rmse / m0.rmse, 1)}). Squaring lets a single big error dominate RMSE; the median absolute error hardly moves (${fx(m0.med)} → ${fx(m.med)}).` }];
    },
  };

  /* ============================================================ scaling before k-means */
  const AGEINC = (() => { const r = rng(51), pts = []; for (let i = 0; i < 20; i++) pts.push([27 + gauss(r) * 3, 52000 + gauss(r) * 15000, 0]); for (let i = 0; i < 20; i++) pts.push([52 + gauss(r) * 4, 56000 + gauss(r) * 15000, 1]); return pts; })();
  V.ml2_scaling = {
    title: "Scaling before k-means: age vs income",
    params: [{ id: "z", label: "standardize features", min: 0, max: 1, step: 1, value: 0, fmt: (v) => (+v ? "on" : "off") }],
    gen(_, P) {
      const on = +P.z === 1, A = AGEINC;
      const mu = [0, 1].map((j) => mean(A.map((p) => p[j]))), sd = [0, 1].map((j) => Math.sqrt(mean(A.map((p) => (p[j] - mu[j]) ** 2))));
      const Z = A.map((p) => (on ? [(p[0] - mu[0]) / sd[0], (p[1] - mu[1]) / sd[1]] : [p[0], p[1]]));
      const km = kmeansBest(Z, 2, 3, 8), lab = km.lab;
      const agree = Math.max(mean(lab.map((v, i) => +(v === A[i][2]))), mean(lab.map((v, i) => +(v !== A[i][2]))));
      const shape = (S, x, y, g, c) => g ? `<rect x="${(S.sx(x) - 4.2).toFixed(1)}" y="${(S.sy(y) - 4.2).toFixed(1)}" width="8.4" height="8.4" style="${fillDot(c)}"/>` : `<circle cx="${S.sx(x).toFixed(1)}" cy="${S.sy(y).toFixed(1)}" r="4.6" style="${fillDot(c)}"/>`;
      const R = 42000, left = on
        ? chart({ W: 278, H: 274, x: [-3, 3], y: [-3, 3], xl: "age (z-score)", yl: "income (z-score)", ticks: 4, yticks: 4 }, [(S) => Z.map((p, i) => shape(S, p[0], p[1], A[i][2], col(lab[i] ? 1 : 0))).join("")])
        : chart({ W: 278, H: 274, x: [mu[0] - R, mu[0] + R], y: [mu[1] - R, mu[1] + R], xl: "age (years)", yl: "income ($)", grid: false }, [(S) => Z.map((p, i) => shape(S, p[0], p[1], A[i][2], col(lab[i] ? 1 : 0))).join("")],
          (S) => txt(S.P.l + 8, S.P.t + 16, "same scale on both axes:", { a: "start", fs: 11, c: "var(--muted)" }) + txt(S.P.l + 8, S.P.t + 30, "1 year = $1", { a: "start", fs: 11, c: "var(--muted)" }));
      const right = chart({ W: 330, H: 274, x: [15, 65], y: [0, 100], xl: "age (years)", yl: "income ($ thousands)", ticks: 5 }, [(S) => A.map((p, i) => shape(S, p[0], p[1] / 1000, p[2], col(lab[i] ? 1 : 0))).join("")],
        (S) => txt(S.W - S.P.r - 6, S.P.t + 14, "circles: under 40 · squares: over 40", { a: "end", fs: 10.5, c: "var(--muted)" }));
      return [{ svg: row(left, right), side: { standardized: on ? "yes" : "no", "matches the age groups": pct(agree), "income spread": "$" + Math.round(sd[1]).toLocaleString("en-US"), "age spread": fx(sd[0], 1) + " years" },
        msg: on ? `After **standardizing** (subtract each feature's mean, divide by its standard deviation) a step of one standard deviation counts the same in age and income. The two age groups become two separate blobs and k-means finds them: **${pct(agree)}** agreement.`
          : `Raw units: income varies by about $${Math.round(sd[1] / 1000)},000 while age varies by about ${Math.round(sd[0])} years, so distances are almost pure income. On a common scale (left) every point sits on one vertical line, and k-means splits **high vs low income** instead of the real age groups: only **${pct(agree)}** agreement (50% is a coin flip). Turn standardizing on.` }];
    },
  };

  /* ============================================================ static diagrams */
  // elbow method: real k-means inertia on the 4-blob data for k = 1..8
  D.ml2_elbow = () => {
    const J = [1, 2, 3, 4, 5, 6, 7, 8].map((k) => [k, kmeansBest(BLOBS, k, 5, 10).J]);
    return chart({ W: 480, H: 260, x: [0.5, 8.5], y: [0, J[0][1] * 1.1], xl: "number of clusters k", yl: "inertia (within-cluster SS)", ticks: 8 }, [
      Plot.line(J, "pl-c1"), Plot.pts(J, "pl-c1", 4.5), Plot.dot(4, J[3][1], "pl-c2", 7),
      (S) => segPx(S, 5.3, J[3][1] + J[0][1] * 0.22, 4.15, J[3][1] + J[0][1] * 0.03, "stroke:var(--muted);stroke-width:1.5", true),
      Plot.text(5.4, J[3][1] + J[0][1] * 0.25, "elbow at k = 4", "", "start"),
    ]);
  };

  // dendrogram: average-linkage agglomerative clustering of 8 labelled points
  D.ml2_dendrogram = () => {
    const pts = [[1.0, 1.2], [1.7, 0.9], [1.3, 2.0], [4.2, 4.6], [4.9, 4.1], [4.5, 5.4], [8.0, 1.6], [8.7, 2.5]], names = "ABCDEFGH".split("");
    const dist = (a, b) => Math.sqrt(d2(a, b));
    let clusters = pts.map((p, i) => ({ id: i, members: [i], h: 0 })), nodes = clusters.slice(), merges = [];
    while (clusters.length > 1) {
      let best = null;
      for (let i = 0; i < clusters.length; i++) for (let j = i + 1; j < clusters.length; j++) {
        let s = 0; clusters[i].members.forEach((a) => clusters[j].members.forEach((b) => (s += dist(pts[a], pts[b]))));
        const d = s / (clusters[i].members.length * clusters[j].members.length);
        if (!best || d < best.d) best = { i, j, d };
      }
      const A = clusters[best.i], B = clusters[best.j], N = { id: nodes.length, members: A.members.concat(B.members), h: best.d, kids: [A, B] };
      nodes.push(N); merges.push(N); clusters = clusters.filter((_, k) => k !== best.i && k !== best.j).concat([N]);
    }
    const root = clusters[0], order = [];
    (function walk(n) { if (!n.kids) order.push(n.id); else n.kids.forEach(walk); })(root);
    const cut = (merges[merges.length - 3].h + merges[merges.length - 2].h) / 2;   // cut between the last 3 merges → 3 clusters
    const groupOf = {}; let g = 0;
    (function tag(n) { if (n.h <= cut) { n.members.forEach((m) => (groupOf[m] = g)); g++; } else n.kids.forEach(tag); })(root);
    // left: the points
    const sx = (x) => 20 + x * 24, sy = (y) => 200 - y * 30;
    let L = `<rect x="8" y="16" width="232" height="196" rx="8" style="fill:none;stroke:var(--border)"/>` + txt(124, 12, "the points", { fs: 11.5, c: "var(--muted)" });
    pts.forEach((p, i) => { L += `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="6" style="${fillDot(col(groupOf[i]))}"/>` + txt(sx(p[0]) + 10, sy(p[1]) - 6, names[i], { a: "start", fs: 12, b: 1 }); });
    // right: the dendrogram
    const ox = 290, W = 330, base = 200, top = 26, hmax = root.h, xs = {}, yh = (h) => base - (h / hmax) * (base - top);
    order.forEach((id, k) => (xs[id] = ox + 30 + k * ((W - 50) / (order.length - 1))));
    let R = `<line x1="${ox + 10}" y1="${top}" x2="${ox + 10}" y2="${base}" style="stroke:var(--border)"/>` + txt(ox + 4, top - 8, "merge distance", { a: "start", fs: 11, c: "var(--muted)" });
    const pos = (n) => [xs[n.id] != null ? xs[n.id] : (xs[n.id] = (pos(n.kids[0])[0] + pos(n.kids[1])[0]) / 2), yh(n.h)];
    merges.forEach((n) => {
      const [x1, y1] = pos(n.kids[0]), [x2, y2] = pos(n.kids[1]), y = yh(n.h), c = n.h <= cut ? col(groupOf[n.members[0]]) : "var(--muted)";
      R += `<path d="M${x1},${y1}V${y}H${x2}V${y2}" style="fill:none;stroke:${c};stroke-width:2.2"/>`;
    });
    order.forEach((id) => (R += txt(xs[id], base + 16, names[id], { fs: 12, b: 1 })));
    R += `<line x1="${ox + 10}" y1="${yh(cut)}" x2="${ox + W}" y2="${yh(cut)}" style="stroke:var(--text);stroke-width:1.4;stroke-dasharray:5 4"/>` + txt(ox + W, yh(cut) - 6, "cut here → 3 clusters", { a: "end", fs: 11, c: "var(--text)" });
    return svgWrap(630, 226, L + R);
  };

  // precision and recall against the threshold (300 toy scores, 20% positives)
  D.ml2_prtradeoff = () => {
    const r = rng(61), sig = (z) => 1 / (1 + Math.exp(-z)), Dt = [];
    for (let i = 0; i < 240; i++) Dt.push([sig(gauss(r) * 1.3 - 1.3), 0]);
    for (let i = 0; i < 60; i++) Dt.push([sig(gauss(r) * 1.3 + 1.0), 1]);
    const P = 60, pr = [], rc = [], f1 = [];
    let best = [0, 0];
    for (let t = 0.02; t <= 0.981; t += 0.01) {
      const tp = Dt.filter((d) => d[1] && d[0] >= t).length, fp = Dt.filter((d) => !d[1] && d[0] >= t).length;
      if (tp + fp === 0) continue;
      const p = tp / (tp + fp), q = tp / P, f = p + q ? (2 * p * q) / (p + q) : 0;
      pr.push([t, p]); rc.push([t, q]); f1.push([t, f]); if (f > best[1]) best = [t, f];
    }
    return chart({ W: 520, H: 280, x: [0, 1], y: [0, 1.02], xl: "decision threshold", yl: "score", pad: { l: 44, r: 14, t: 34, b: 40 } }, [
      curve(pr, "var(--purple)", 2.6), curve(rc, "var(--green)", 2.6), curve(f1, "var(--amber)", 2.2, "6 4"),
      Plot.vline(best[0]), Plot.dot(best[0], best[1], "pl-c2", 5.5),
    ], (S) => legend(S, [["precision", "var(--purple)"], ["recall", "var(--green)"], ["F1", "var(--amber)", 1]], 14) + txt(S.sx(best[0]) + 6, S.sy(0.08), "best F1 at t ≈ " + best[0].toFixed(2), { a: "start", fs: 11.5 }));
  };

  // the confusion matrix explained
  D.ml2_cm = () => {
    const x0 = 140, y0 = 60, cw = 150, ch = 78;
    const cell = (x, y, a, b, c) => `<rect x="${x}" y="${y}" width="${cw - 6}" height="${ch - 6}" rx="8" style="fill:${soft(c, 18)};stroke:${c};stroke-width:1.6"/>` + txt(x + (cw - 6) / 2, y + 30, a, { fs: 15, b: 1 }) + txt(x + (cw - 6) / 2, y + 52, b, { fs: 11.5, c: "var(--muted)" });
    let s = txt(x0 + cw - 3, 18, "PREDICTED", { fs: 11, b: 1, c: "var(--muted)" }) + txt(x0 + cw / 2 - 3, y0 - 12, "positive", { fs: 12, c: "var(--muted)" }) + txt(x0 + cw * 1.5 - 3, y0 - 12, "negative", { fs: 12, c: "var(--muted)" });
    s += txt(28, y0 + ch, "ACTUAL", { fs: 11, b: 1, c: "var(--muted)", rot: -90 }) + txt(x0 - 12, y0 + ch / 2, "positive", { a: "end", fs: 12, c: "var(--muted)" }) + txt(x0 - 12, y0 + ch * 1.5, "negative", { a: "end", fs: 12, c: "var(--muted)" });
    s += cell(x0, y0, "TP", "true positive: a hit", "var(--green)") + cell(x0 + cw, y0, "FN", "false negative: a miss", "var(--amber)");
    s += cell(x0, y0 + ch, "FP", "false positive: a false alarm", "var(--red)") + cell(x0 + cw, y0 + ch, "TN", "true negative", "var(--accent)");
    // precision = left column, recall = top row
    s += `<rect x="${x0 - 4}" y="${y0 - 4}" width="${cw + 2}" height="${2 * ch + 2}" rx="10" style="fill:none;stroke:var(--purple);stroke-width:2;stroke-dasharray:6 4"/>`;
    s += `<rect x="${x0 - 8}" y="${y0 - 8}" width="${2 * cw + 10}" height="${ch + 10}" rx="12" style="fill:none;stroke:var(--green);stroke-width:2;stroke-dasharray:2 3"/>`;
    s += txt(x0 + 2 * cw + 18, y0 + ch / 2 - 4, "recall = TP / (TP + FN)", { a: "start", fs: 12.5, b: 1, c: "var(--green)" }) + txt(x0 + 2 * cw + 18, y0 + ch / 2 + 13, "the actual-positive row", { a: "start", fs: 11, c: "var(--muted)" });
    s += txt(x0 + cw / 2, y0 + 2 * ch + 22, "precision = TP / (TP + FP)", { fs: 12.5, b: 1, c: "var(--purple)" }) + txt(x0 + cw / 2, y0 + 2 * ch + 38, "the predicted-positive column", { fs: 11, c: "var(--muted)" });
    s += txt(x0 + 2 * cw + 18, y0 + ch * 1.5 - 4, "accuracy = (TP + TN) / all", { a: "start", fs: 12.5, b: 1 }) + txt(x0 + 2 * cw + 18, y0 + ch * 1.5 + 13, "the diagonal", { a: "start", fs: 11, c: "var(--muted)" });
    return svgWrap(630, 262, s);
  };

  // the ROC staircase for a handful of scores: D.ml2_rocsteps([scores], [labels])
  D.ml2_rocsteps = (scores, labels) => {
    const idx = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]), P = labels.filter(Boolean).length, N = labels.length - P;
    const pts = [[0, 0]]; let x = 0, y = 0;
    idx.forEach((i) => { if (labels[i]) y += 1 / P; else x += 1 / N; pts.push([x, y]); });
    let auc = 0; for (let i = 1; i < pts.length; i++) auc += ((pts[i][0] - pts[i - 1][0]) * (pts[i][1] + pts[i - 1][1])) / 2;
    return chart({ W: 360, H: 320, x: [0, 1], y: [0, 1.04], xl: "FPR (one step right per negative)", yl: "TPR (one step up per positive)", ticks: 3, yticks: 3 }, [
      area(pts.concat([[1, 0]]), "var(--accent)", 14), Plot.seg(0, 0, 1, 1, "pl-dash"), Plot.line(pts, "pl-c1"),
      (S) => idx.map((i, k) => { const [a, b] = pts[k + 1]; const up = labels[i]; return `<circle cx="${S.sx(a).toFixed(1)}" cy="${S.sy(b).toFixed(1)}" r="4.5" style="${fillDot(up ? "var(--amber)" : "var(--accent)")}"/>` + txt(S.sx(a) + (up ? -8 : 4), S.sy(b) + (up ? 4 : -8), String(scores[i]), { a: up ? "end" : "start", fs: 11 }); }).join(""),
      Plot.text(0.68, 0.3, "AUC = " + fx(auc, 3)),
    ]);
  };

  // leakage: preprocessing before vs after the split
  D.ml2_leakage = () => {
    const bw = 112, bh = 46, gap = 26, x0 = 12;
    const rowOf = (y, items, bad) => items.map(([a, b, c], i) => {
      const x = x0 + i * (bw + gap);
      return box(x, y, bw, bh, c, 14) + txt(x + bw / 2, y + 19, a, { fs: 11.5, b: 1 }) + txt(x + bw / 2, y + 35, b, { fs: 10.5, c: "var(--muted)" }) + (i < items.length - 1 ? arrowLine(x + bw + 2, y + bh / 2, x + bw + gap - 3, y + bh / 2) : "");
    }).join("") + txt(x0, y - 9, bad ? "✗ Leaky: the test rows influence the preprocessing" : "✓ Honest: fit everything on the training rows only", { a: "start", fs: 12, b: 1, c: bad ? "var(--red)" : "var(--green)" });
    return svgWrap(640, 196,
      rowOf(28, [["all data", "train + test", "var(--muted)"], ["fit scaler", "mean, std of ALL", "var(--red)"], ["then split", "train | test", "var(--muted)"], ["train model", "on train", "var(--accent)"], ["test score", "too optimistic", "var(--red)"]], true) +
      rowOf(130, [["all data", "train + test", "var(--muted)"], ["split first", "train | test", "var(--muted)"], ["fit scaler", "on train only", "var(--green)"], ["train model", "scaled train", "var(--accent)"], ["test score", "transform test, score", "var(--green)"]], false));
  };

  // KFold vs GroupKFold vs TimeSeriesSplit on 12 samples
  D.ml2_cvschemes = () => {
    const n = 12, cw = 26, x0 = 100, rh = 20, panel = (y, title, sub, rows, tags) => {
      let s = txt(12, y, title, { a: "start", fs: 12.5, b: 1 }) + txt(12 + title.length * 7.6 + 6, y, sub, { a: "start", fs: 11, c: "var(--muted)" });
      if (tags) tags.forEach((t, i) => (s += txt(x0 + i * cw + (cw - 3) / 2, y + 16, t, { fs: 9.5, c: "var(--muted)" })));
      rows.forEach((r, k) => {
        const yy = y + (tags ? 22 : 10) + k * rh;
        s += txt(x0 - 10, yy + 12, "split " + (k + 1), { a: "end", fs: 11, c: "var(--muted)" });
        r.forEach((c, i) => {
          const st = c === "v" ? `fill:${soft("var(--amber)", 75)};stroke:var(--amber)` : c === "t" ? `fill:${soft("var(--accent)", 20)};stroke:${soft("var(--accent)", 55)}` : `fill:none;stroke:var(--border);stroke-dasharray:2 2`;
          s += `<rect x="${x0 + i * cw}" y="${yy}" width="${cw - 3}" height="${rh - 5}" rx="3" style="${st};stroke-width:1.2"/>`;
        });
      });
      return s;
    };
    const kf = [0, 1, 2, 3].map((f) => Array.from({ length: n }, (_, i) => (Math.floor(i / 3) === f ? "v" : "t")));
    const groups = ["P1", "P1", "P2", "P3", "P2", "P1", "P4", "P3", "P4", "P2", "P3", "P4"];
    const gk = ["P1", "P2", "P3", "P4"].map((g) => groups.map((x) => (x === g ? "v" : "t")));
    const ts = [0, 1, 2, 3].map((s) => Array.from({ length: n }, (_, i) => (i < 4 + 2 * s ? "t" : i < 6 + 2 * s ? "v" : "-")));
    let s = panel(18, "KFold", "(4 folds; StratifiedKFold also keeps the class mix in every fold)", kf);
    s += panel(126, "GroupKFold", "(all rows of one patient stay together)", gk, groups);
    s += panel(246, "TimeSeriesSplit", "(time runs left → right; never train on the future)", ts);
    s += box(x0, 352, 14, 12, "var(--accent)", 20, 3) + txt(x0 + 20, 362, "train", { a: "start", fs: 11, c: "var(--muted)" }) + `<rect x="${x0 + 70}" y="352" width="14" height="12" rx="3" style="fill:${soft("var(--amber)", 75)};stroke:var(--amber)"/>` + txt(x0 + 90, 362, "validate", { a: "start", fs: 11, c: "var(--muted)" }) +
      `<rect x="${x0 + 160}" y="352" width="14" height="12" rx="3" style="fill:none;stroke:var(--border);stroke-dasharray:2 2"/>` + txt(x0 + 180, 362, "not used in this split", { a: "start", fs: 11, c: "var(--muted)" });
    return svgWrap(520, 372, s);
  };

  // SMOTE: new minority points on the segments between a point and its minority neighbours
  D.ml2_smote = () => {
    const r = rng(71), maj = [], mino = [[6.2, 6.6], [7.0, 5.6], [7.6, 6.9], [6.4, 7.8], [8.3, 6.0], [7.2, 8.1]];
    for (let i = 0; i < 55; i++) { const p = [2 + r() * 7.5, 1 + r() * 8]; if (Math.hypot(p[0] - 7.2, p[1] - 6.8) > 1.9) maj.push(p); }
    const base = mino[1], nn = mino.filter((p) => p !== base).sort((a, b) => d2(a, base) - d2(b, base)).slice(0, 3), lam = [0.35, 0.6, 0.5];
    const synth = nn.map((p, i) => [base[0] + lam[i] * (p[0] - base[0]), base[1] + lam[i] * (p[1] - base[1])]);
    return chart({ W: 440, H: 300, x: [1, 10], y: [0.5, 9.5], grid: false, xl: "feature 1", pad: { l: 20, r: 14, t: 14, b: 40 } }, [
      circles(maj.map((p) => [p[0], p[1], fillDot("var(--accent)", ";opacity:.55")]), 4),
      (S) => nn.map((p) => segPx(S, base[0], base[1], p[0], p[1], "stroke:var(--amber);stroke-width:1.5;stroke-dasharray:4 3")).join(""),
      circles(mino.map((p) => [p[0], p[1], fillDot("var(--amber)")]), 5.5),
      (S) => `<circle cx="${S.sx(base[0])}" cy="${S.sy(base[1])}" r="9" style="fill:none;stroke:var(--text);stroke-width:1.5"/>`,
      circles(synth.map((p) => [p[0], p[1], "fill:var(--green);stroke:var(--text);stroke-width:1.2"]), 5),
    ], (S) => txt(S.sx(base[0]) + 12, S.sy(base[1]) + 20, "chosen minority point", { a: "start", fs: 11 }) +
      legend(S, [["majority", "var(--accent)"], ["minority", "var(--amber)"], ["new synthetic points", "var(--green)"]], S.H - 8));
  };

  // a first-model flowchart
  D.ml2_modelmap = () => {
    const B = (x, y, w, h, c, lines, bold) => box(x, y, w, h, c, 13) + lines.map((l, i) => txt(x + w / 2, y + 18 + i * 15 - ((lines.length - 1) * 15) / 2 + (h - 36) / 2, l, { fs: i === 0 && bold ? 12 : 11, b: i === 0 && bold ? 1 : 0, c: i === 0 && bold ? "var(--text)" : "var(--muted)" })).join("");
    const lab = (x, y, s) => txt(x, y, s, { fs: 11, b: 1, c: "var(--accent)" });
    let s = B(220, 8, 200, 38, "var(--accent)", ["Do you have labels (a target)?"], true);
    s += arrowLine(250, 46, 130, 80) + lab(172, 64, "no") + arrowLine(390, 46, 470, 80) + lab(444, 64, "yes");
    s += B(10, 82, 240, 76, "var(--purple)", ["Unsupervised", "find groups: k-means, DBSCAN, hierarchical", "compress: PCA · look: t-SNE / UMAP", "spot oddities: Isolation Forest"], true);
    s += B(350, 82, 260, 38, "var(--accent)", ["What does one example look like?"], true);
    s += arrowLine(420, 120, 330, 160) + lab(360, 146, "") + arrowLine(540, 120, 540, 160);
    s += B(210, 162, 200, 60, "var(--amber)", ["Images, audio, free text", "deep learning: fine-tune a", "pretrained CNN / transformer"], true);
    s += B(430, 162, 200, 60, "var(--green)", ["Rows and columns", "(tabular data)"], true);
    s += arrowLine(530, 222, 530, 246);
    s += B(300, 248, 330, 92, "var(--green)", ["Tabular recipe", "1. dummy baseline (most frequent / mean)", "2. linear or logistic regression (scaled)", "3. gradient boosting (HistGB, XGBoost, LightGBM)", "4. tune with cross-validation, then test once"], true);
    s += B(10, 248, 270, 92, "var(--muted)", ["Special needs", "must explain each decision → linear / small tree", "tiny data (< 100s of rows) → simple models", "huge data, fast → linear + SGD, LightGBM", "text with few labels → TF-IDF + logistic"], true);
    return svgWrap(640, 348, s);
  };

  /* ============================================================ glossary */
  Object.assign(window.GLOSSARY, {
    "unsupervised learning": "Learning from data that has no labels: the goal is to find structure, such as groups or a simpler description of the data.",
    "clustering": "Grouping examples so that similar ones end up together, without being told the groups in advance.",
    "centroid": "The centre point of a cluster: the average of all the points assigned to it.",
    "inertia": "In k-means, the sum of squared distances from every point to its cluster's centroid. Lower means tighter clusters.",
    "k-means": "A clustering algorithm that alternates two steps: assign each point to the nearest of k centres, then move each centre to the mean of its points.",
    "k-means++": "A smart way to pick k-means' starting centres: each new centre is chosen far from the ones already picked, which gives better, more stable results.",
    "silhouette score": "A clustering quality score from −1 to 1: how much closer each point is to its own cluster than to the next nearest cluster.",
    "dbscan": "A clustering algorithm that grows clusters from dense regions (points with many neighbours within a radius eps) and labels isolated points as noise.",
    "core point": "In DBSCAN, a point with at least minPts points (itself included) within distance eps. Only core points can grow a cluster.",
    "border point": "In DBSCAN, a point that is close to a core point but does not have enough neighbours to be a core point itself.",
    "dendrogram": "A tree diagram that shows the order in which hierarchical clustering merged points and groups, and at what distance.",
    "linkage": "In hierarchical clustering, the rule for the distance between two groups: closest pair (single), farthest pair (complete), average, or Ward.",
    "pca": "Principal Component Analysis: rotate the data onto new axes ordered by how much the data varies along them, then keep only the first few.",
    "principal component": "One of the new axes found by PCA: a direction in feature space. The first one points where the data is most spread out.",
    "explained variance": "The share of the data's total variance captured by a principal component (or by the first few together).",
    "dimensionality reduction": "Describing each example with fewer numbers while keeping as much of the useful information as possible (PCA, t-SNE, UMAP…).",
    "t-sne": "A non-linear method that squashes high-dimensional data into 2-D for plotting while keeping close neighbours close. For looking, not for features.",
    "umap": "A non-linear dimensionality-reduction method like t-SNE, usually faster, often used to make 2-D maps of embeddings.",
    "confusion matrix": "A table counting a classifier's results by actual class and predicted class: true positives, false positives, false negatives and true negatives.",
    "true positive": "A positive example that the model correctly predicted as positive (a hit).",
    "false positive": "A negative example that the model wrongly predicted as positive (a false alarm, or Type I error).",
    "false negative": "A positive example that the model wrongly predicted as negative (a miss, or Type II error).",
    "true negative": "A negative example that the model correctly predicted as negative.",
    "precision": "In classification: of everything the model flagged as positive, the fraction that really is positive. (For numbers, precision means how many digits or bits are kept.)",
    "recall": "Of all the truly positive examples, the fraction the model found. Also called sensitivity or the true positive rate.",
    "f1 score": "The harmonic mean of precision and recall: high only when both are high.",
    "specificity": "Of all the truly negative examples, the fraction the model correctly called negative (the true negative rate).",
    "decision threshold": "The cut-off on a model's score or probability above which an example is called positive. 0.5 is only a default.",
    "roc curve": "A plot of true positive rate against false positive rate for every possible decision threshold.",
    "roc-auc": "The area under the ROC curve: the probability that a random positive gets a higher score than a random negative. 0.5 is random guessing, 1 is perfect.",
    "average precision": "The area under the precision–recall curve (roughly the mean precision over all recall levels). Its baseline is the share of positives.",
    "mae": "Mean absolute error: the average size of the errors, in the target's units. Not very sensitive to outliers.",
    "rmse": "Root mean squared error: the square root of the average squared error. In the target's units, and it punishes big errors heavily.",
    "mape": "Mean absolute percentage error: the average error as a percentage of the true value. Breaks down when true values are near zero.",
    "cross-validation": "Estimating performance by training and testing on several different splits of the data (e.g. 5 folds) and averaging the scores.",
    "k-fold": "Cross-validation that cuts the data into k parts; each part is the validation set once while the model trains on the other k − 1.",
    "grid search": "Trying every combination of a list of hyperparameter values and keeping the one with the best cross-validation score.",
    "nested cross-validation": "Cross-validation inside cross-validation: the inner loop tunes hyperparameters, the outer loop measures how well the whole procedure works.",
    "standardization": "Rescaling a feature to mean 0 and standard deviation 1: subtract the mean, divide by the standard deviation.",
    "min-max scaling": "Rescaling a feature to the range 0 to 1: subtract the minimum, divide by (maximum − minimum).",
    "one-hot encoding": "Turning a category column into one 0/1 column per category, e.g. colour = red → [1, 0, 0].",
    "ordinal encoding": "Replacing each category with an integer (small = 0, medium = 1, large = 2). Right when the categories have a natural order.",
    "target encoding": "Replacing each category with the average target value of the training rows in that category (smoothed, and computed out-of-fold to avoid leakage).",
    "class imbalance": "When one class is much rarer than the others (fraud, disease, defects), so a model can look accurate while ignoring it.",
    "class weight": "A multiplier on the loss for examples of a class. Giving the rare class a bigger weight makes mistakes on it cost more.",
    "oversampling": "Adding copies (or synthetic versions) of rare-class examples to the training set so the classes are more balanced.",
    "undersampling": "Throwing away some majority-class training examples so the classes are more balanced.",
    "smote": "Synthetic Minority Over-sampling: creates new rare-class examples on the straight line between a rare example and one of its rare-class neighbours.",
    "balanced accuracy": "The average of the recall of each class. A model that always predicts the majority class gets 0.5 in a two-class problem, however imbalanced.",
  });
})();
