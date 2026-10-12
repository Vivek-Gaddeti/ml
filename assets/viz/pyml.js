/* Python for ML: visualizers, diagrams and glossary terms.
   Loaded on every doc page and on Practice, after assets/viz.js. Every name starts with pyml_. All numbers are computed. */
(function () {
  const V = Viz.defs, r3 = Viz.helpers.r3;

  /* ------------------------------------------------------------------ helpers */
  const soft = (v, p = 18) => `color-mix(in srgb, var(--${v}) ${p}%, transparent)`;
  const shapeStr = (s) => (s.length === 0 ? "()" : s.length === 1 ? `(${s[0]},)` : `(${s.join(", ")})`);
  const parseShape = (txt) => (txt.match(/\d+/g) || []).map((v) => Math.min(999, +v));
  const rect = (x, y, w, h, fill, stroke, dash, rx = 6) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" style="fill:${fill};stroke:${stroke};stroke-width:1.6${dash ? ";stroke-dasharray:4 3" : ""}"/>`;
  const txt = (x, y, s, o = {}) => `<text x="${x}" y="${y}" text-anchor="${o.a || "middle"}" style="font-size:${o.s || 13}px;${o.w ? "font-weight:" + o.w + ";" : ""}fill:${o.c || "var(--text)"}${o.mono ? ";font-family:var(--mono)" : ""}">${esc(s)}</text>`;
  const svgWrap = (w, h, inner) => `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" xmlns="http://www.w3.org/2000/svg">${inner}</svg>`;

  /* ================================================================== 1. Broadcasting */
  // one row of dimension boxes. status per column: "pending" | "ok" | "stretch" | "bad"
  function dimBoard(A, B, n, upto, res) {
    const pa = Array(n - A.length).fill(null).concat(A), pb = Array(n - B.length).fill(null).concat(B);
    const cw = 76, x0 = 96, W = x0 + n * (cw + 10) + 10, H = 196;
    let s = "";
    const rows = [["A", pa, 44], ["B", pb, 92], ["result", res, 150]];
    for (let j = 0; j < n; j++) {
      const x = x0 + j * (cw + 10), fromRight = n - j;
      s += txt(x + cw / 2, 26, "dim −" + fromRight, { s: 11, c: "var(--muted)" });
    }
    rows.forEach(([name, arr, y]) => {
      s += txt(86, y + 22, name, { a: "end", w: 700, s: 13, c: name === "A" ? "var(--accent)" : name === "B" ? "var(--green)" : "var(--text)" });
      for (let j = 0; j < n; j++) {
        const x = x0 + j * (cw + 10), v = arr[j], done = n - j <= upto; // columns are processed from the right
        if (name === "result") {
          if (!done || v == null) { s += rect(x, y, cw, 34, "var(--panel)", "var(--border)", true); continue; }
          const bad = v === "x";
          s += rect(x, y, cw, 34, bad ? soft("red", 28) : soft("accent", 14), bad ? "var(--red)" : "var(--accent)");
          s += txt(x + cw / 2, y + 22, bad ? "error" : String(v), { w: 700, mono: true, c: bad ? "var(--red)" : "var(--text)" });
          continue;
        }
        const a = pa[j] == null ? 1 : pa[j], b = pb[j] == null ? 1 : pb[j];
        if (v == null) { const st = done && a !== b; s += rect(x, y, cw, 34, st ? soft("amber", 30) : "transparent", st ? "var(--amber)" : "var(--border)", true); s += txt(x + cw / 2, y + 22, "(1)", { c: "var(--muted)", mono: true }); continue; }
        let fill = "var(--panel)", stroke = "var(--border)";
        if (done) {
          if (a === b) { fill = soft("green", 18); stroke = "var(--green)"; }
          else if (a === 1 || b === 1) { const stretched = v === 1; fill = stretched ? soft("amber", 30) : soft("green", 18); stroke = stretched ? "var(--amber)" : "var(--green)"; }
          else { fill = soft("red", 26); stroke = "var(--red)"; }
        }
        s += rect(x, y, cw, 34, fill, stroke);
        s += txt(x + cw / 2, y + 22, String(v), { w: 700, mono: true });
      }
    });
    return { s, W, H };
  }

  // little grids showing how each operand is stretched to the result shape (only for 2-D results of modest size)
  function stretchPicture(A, B, R) {
    const pad2 = (s) => (s.length === 1 ? [1, s[0]] : s.length === 0 ? [1, 1] : s);
    const a = pad2(A), b = pad2(B), r = pad2(R), cs = 18;
    if (r[0] > 8 || r[1] > 8) return "";
    const gw = r[1] * cs, gh = r[0] * cs, gap = 46, x0 = 14, W = x0 * 2 + gw * 3 + gap * 2, H = gh + 54;
    const grid = (shape, x, cls) => {
      let g = "";
      for (let i = 0; i < r[0]; i++) for (let j = 0; j < r[1]; j++) {
        const orig = cls !== "r" && i < shape[0] && j < shape[1];
        const c = cls === "a" ? "accent" : cls === "b" ? "green" : "purple";
        g += `<rect x="${x + j * cs}" y="${30 + i * cs}" width="${cs - 2}" height="${cs - 2}" rx="3" style="fill:${orig || cls === "r" ? soft(c, 55) : soft(c, 12)};stroke:var(--${c});stroke-width:1.2${orig || cls === "r" ? "" : ";stroke-dasharray:3 2"}"/>`;
      }
      return g;
    };
    const xs = [x0, x0 + gw + gap, x0 + 2 * (gw + gap)];
    let s = grid(a, xs[0], "a") + grid(b, xs[1], "b") + grid(r, xs[2], "r");
    s += txt(xs[0] + gw / 2, 18, "A stretched", { s: 12, c: "var(--accent)", w: 600 }) + txt(xs[1] + gw / 2, 18, "B stretched", { s: 12, c: "var(--green)", w: 600 }) + txt(xs[2] + gw / 2, 18, "A + B", { s: 12, w: 600 });
    s += txt(xs[0] + gw + gap / 2 - 1, 30 + gh / 2 + 5, "+", { s: 20, c: "var(--muted)" }) + txt(xs[1] + gw + gap / 2 - 1, 30 + gh / 2 + 5, "=", { s: 20, c: "var(--muted)" });
    s += txt(W / 2, H - 6, "solid = real data, dashed = repeated copies", { s: 11, c: "var(--muted)" });
    return svgWrap(W, H, s);
  }

  V.pyml_broadcast = {
    title: "Broadcasting: which shapes can be combined?",
    input: "3,4 | 4",
    help: "two shapes separated by |  e.g. 3,4 | 4   or   5,1 | 1,4   or   3,4 | 3",
    gen(input) {
      const parts = input.split("|");
      if (parts.length !== 2) throw new Error("need two shapes");
      const A = parseShape(parts[0]), B = parseShape(parts[1]);
      if (A.length > 4 || B.length > 4) throw new Error("at most 4 dims");
      const n = Math.max(A.length, B.length, 1), S = [];
      const res = Array(n).fill(null);
      const sideBase = { "shape of A": shapeStr(A), "shape of B": shapeStr(B) };
      const pa = Array(n - A.length).fill(1).concat(A), pb = Array(n - B.length).fill(1).concat(B);
      const frame = (upto, msg, side, picture) => {
        const d = dimBoard(A, B, n, upto, res);
        return { svg: svgWrap(d.W, d.H, d.s) + (picture || ""), side: Object.assign({}, sideBase, side || {}), msg };
      };
      S.push(frame(0, `Two arrays: **A** has shape ${shapeStr(A)} and **B** has shape ${shapeStr(B)}. NumPy compares the shapes **from the right-hand end**, one dimension at a time. Press Next.`));
      if (A.length !== B.length) {
        const short = A.length < B.length ? "A" : "B";
        S.push(frame(0, `Step 1: the shorter shape is padded with 1s **on the left** until both have ${n} dimensions. So **${short}** is treated as ${shapeStr(short === "A" ? pa : pb)}. (Dashed boxes are the padded dimensions.)`, { "padded A": shapeStr(pa), "padded B": shapeStr(pb) }));
      } else {
        S.push(frame(0, `Step 1: both shapes already have ${n} dimension${n > 1 ? "s" : ""}, so no padding is needed.`));
      }
      let failed = false;
      for (let k = 1; k <= n && !failed; k++) {
        const j = n - k, a = pa[j], b = pb[j];
        let msg;
        if (a === b) { res[j] = a; msg = `Dimension −${k}: ${a} and ${b} are **equal**, so the result keeps ${a}.`; }
        else if (a === 1) { res[j] = b; msg = `Dimension −${k}: A has size **1**, so it is stretched (repeated) ${b} times to match B. Result: ${b}.`; }
        else if (b === 1) { res[j] = a; msg = `Dimension −${k}: B has size **1**, so it is stretched (repeated) ${a} times to match A. Result: ${a}.`; }
        else { res[j] = "x"; failed = true; msg = `Dimension −${k}: ${a} versus ${b}. They are different and neither is 1, so NumPy cannot line them up: **ValueError: operands could not be broadcast together with shapes ${shapeStr(A).replace(", ", ",")} ${shapeStr(B).replace(", ", ",")}**.`; }
        S.push(frame(k, msg, failed ? { result: "error" } : {}));
      }
      if (!failed) {
        const R = res.slice();
        const pic = stretchPicture(A, B, R);
        const stretchedA = pa.map((v, j) => v !== R[j]).some(Boolean), stretchedB = pb.map((v, j) => v !== R[j]).some(Boolean);
        S.push(frame(n, `Done: the result has shape **${shapeStr(R)}** (${R.reduce((x, y) => x * y, 1)} numbers). ${stretchedA || stretchedB ? "No data is really copied: NumPy just re-uses the size-1 dimension over and over." : "Same shapes, so this is an ordinary element-wise operation."}` + (pic ? "" : " (The picture is skipped for big or 3-D+ shapes.)"), { result: shapeStr(R) }, pic));
      } else {
        S.push(frame(n, "Rule of thumb: from the right, every pair of sizes must be **equal or contain a 1**. Fix a mismatch with reshape, `[:, None]` (add an axis), or by transposing.", { result: "error" }));
      }
      return S;
    },
  };

  /* ================================================================== 2. Axis reductions */
  const AXM = [[2, 7, 1, 5], [4, 3, 8, 6], [9, 0, 2, 3]];
  const AXOP = [
    ["sum", (a) => a.reduce((s, v) => s + v, 0)],
    ["mean", (a) => a.reduce((s, v) => s + v, 0) / a.length],
    ["max", (a) => Math.max(...a)],
    ["min", (a) => Math.min(...a)],
  ];
  V.pyml_axis = {
    title: "Axes: which dimension collapses?",
    params: [
      { id: "axis", label: "axis", min: 0, max: 2, step: 1, value: 0, fmt: (v) => (+v === 2 ? "None (all)" : "axis=" + v) },
      { id: "op", label: "operation", min: 0, max: 3, step: 1, value: 0, fmt: (v) => AXOP[+v][0] },
    ],
    gen(_, P) {
      const [opName, f] = AXOP[P.op], R = 3, C = 4, cw = 58, ch = 40, x0 = 40, y0 = 50;
      const axis = P.axis, S = [];
      const slots = axis === 0 ? C : axis === 1 ? R : 1;
      const results = [];
      for (let s = 0; s < slots; s++) {
        const cells = [];
        for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) if (axis === 2 || (axis === 0 && j === s) || (axis === 1 && i === s)) cells.push([i, j]);
        results.push({ cells, val: f(cells.map(([i, j]) => AXM[i][j])) });
      }
      const fmt = (v) => (Number.isInteger(v) ? String(v) : r3(v));
      const shape = axis === 2 ? "()" : axis === 0 ? "(4,)" : "(3,)";
      const call = axis === 2 ? `X.${opName}()` : `X.${opName}(axis=${axis})`;
      // resX/resY: where a result cell for slot s sits
      const pos = (s) => (axis === 0 ? [x0 + s * cw, y0 + R * ch + 26] : axis === 1 ? [x0 + C * cw + 26, y0 + s * ch] : [x0 + C * cw + 26, y0 + R * ch + 26]);
      const W = x0 + C * cw + 26 + (axis === 0 ? 0 : cw) + 30, H = y0 + R * ch + 26 + ch + 26;
      const draw = (active, shown) => {
        let g = txt(x0 + (C * cw) / 2, 18, "X  (shape (3, 4))", { s: 12, c: "var(--muted)", w: 600 });
        for (let j = 0; j < C; j++) g += txt(x0 + j * cw + cw / 2, y0 - 8, "col " + j, { s: 11, c: "var(--muted)" });
        for (let i = 0; i < R; i++) g += txt(x0 - 6, y0 + i * ch + ch / 2 + 4, "row " + i, { s: 11, c: "var(--muted)", a: "end" });
        const hot = new Set(active == null ? [] : results[active].cells.map(([i, j]) => i + "," + j));
        const old = new Set();
        for (let s = 0; s < shown; s++) if (s !== active) results[s].cells.forEach(([i, j]) => old.add(i + "," + j));
        for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
          const k = i + "," + j, isHot = hot.has(k);
          g += rect(x0 + j * cw + 2, y0 + i * ch + 2, cw - 4, ch - 4, isHot ? soft("amber", 38) : old.has(k) ? soft("green", 12) : "var(--panel)", isHot ? "var(--amber)" : old.has(k) ? "var(--green)" : "var(--border)");
          g += txt(x0 + j * cw + cw / 2, y0 + i * ch + ch / 2 + 5, String(AXM[i][j]), { w: 600, mono: true });
        }
        for (let s = 0; s < slots; s++) {
          const [px, py] = pos(s), isNow = s === active, has = s < shown;
          g += rect(px + 2, py + 2, cw - 4, ch - 4, has ? soft(isNow ? "amber" : "green", isNow ? 38 : 20) : "transparent", has ? (isNow ? "var(--amber)" : "var(--green)") : "var(--border)", !has);
          if (has) g += txt(px + cw / 2, py + ch / 2 + 5, fmt(results[s].val), { w: 700, mono: true });
        }
        if (axis === 0) g += txt(x0 + (C * cw) / 2, y0 + R * ch + 26 + ch + 22, `result shape ${shape}: one number per column`, { s: 12, c: "var(--muted)" });
        else if (axis === 1) g += txt(x0 + C * cw + 26 + cw / 2, y0 - 8, "result", { s: 11, c: "var(--muted)" }) + txt(x0 + (C * cw) / 2, y0 + R * ch + 24, `result shape ${shape}: one number per row`, { s: 12, c: "var(--muted)" });
        else g += txt(x0 + (C * cw) / 2, y0 + R * ch + 26 + ch + 22, "result is a single number (shape ())", { s: 12, c: "var(--muted)" });
        return svgWrap(W, H, g);
      };
      const side = (shown) => ({ call, "result shape": shape, filled: `${shown} / ${slots}` });
      S.push({ svg: draw(null, 0), side: side(0), msg: axis === 2
        ? `\`${call}\` with no axis combines **all 12 numbers** into one. Press Next.`
        : `\`${call}\`: the dimension named by \`axis=${axis}\` is the one that **disappears**. ${axis === 0 ? "Axis 0 runs down the rows, so each column gets squashed into one number." : "Axis 1 runs across the columns, so each row gets squashed into one number."} Press Next.` });
      for (let s = 0; s < slots; s++) {
        const vals = results[s].cells.map(([i, j]) => AXM[i][j]);
        const who = axis === 0 ? `column ${s}` : axis === 1 ? `row ${s}` : "the whole matrix";
        S.push({ svg: draw(s, s + 1), side: side(s + 1), msg: `${opName} of ${who}: ${opName}([${vals.join(", ")}]) = **${fmt(results[s].val)}**.` });
      }
      S.push({ svg: draw(null, slots), side: side(slots), msg: axis === 2
        ? "No axis: everything collapsed, result shape `()`. Add `keepdims=True` if you want to keep shape (1, 1)."
        : `Done. Shape (3, 4) became **${shape}**: the axis you named (${axis}) is the one that vanished, the other one survived. Tip: \`keepdims=True\` keeps the collapsed axis with size 1 (shape ${axis === 0 ? "(1, 4)" : "(3, 1)"}), which broadcasts back against X.` });
      return S;
    },
  };

  /* ================================================================== 3. Boolean masks and fancy indexing */
  const MA = [7, 2, 9, 4, 6, 1, 8, 3];
  V.pyml_mask = {
    title: "Boolean masks and fancy indexing",
    kind: "rows",
    input: "0 3 3 5 -1",
    help: "integer indices for the fancy-indexing step (negatives count from the end)",
    params: [{ id: "t", label: "threshold t", min: 0, max: 10, step: 1, value: 5, fmt: (v) => "a > " + v }],
    gen(input, P) {
      const t = P.t, S = [], n = MA.length;
      const mask = MA.map((v) => v > t), keep = MA.filter((v) => v > t);
      const hl = (arr) => Object.fromEntries(arr.map((b, i) => [i, b ? "hl" : "dim"]));
      S.push({ rows: [{ label: "a = np.array([7, 2, 9, 4, 6, 1, 8, 3])", vals: MA }], side: { "a.shape": "(8,)" }, msg: "An array of 8 numbers. We will pick items out of it in two ways: with a **boolean mask** (a True/False per item) and with a **list of positions** (fancy indexing)." });
      S.push({ rows: [{ label: "a", vals: MA, cls: hl(mask) }, { label: `mask = a > ${t}`, vals: mask.map((b) => (b ? "T" : "F")), cls: hl(mask) }], side: { "mask dtype": "bool", "True count": mask.filter(Boolean).length }, msg: `\`a > ${t}\` is an **element-wise comparison**: it returns a new array of the same shape, with True where the test passes. Nothing has been removed yet.` });
      S.push({ rows: [{ label: "a", vals: MA, cls: hl(mask) }, { label: `a[mask]  (a copy)`, vals: keep.length ? keep : ["(empty)"], cls: Object.fromEntries(keep.map((_, i) => [i, "done"])) }], side: { "a[mask].shape": `(${keep.length},)`, "mask.sum()": keep.length, "mean of picked": keep.length ? r3(keep.reduce((s, v) => s + v, 0) / keep.length) : "nan" }, msg: keep.length ? `\`a[mask]\` keeps the items where the mask is True, in the original order. The result is a **new, shorter array** (a copy, not a view). \`mask.sum()\` counts the True values because True is 1.` : `No item is bigger than ${t}, so \`a[mask]\` is an empty array (shape (0,)). Move the slider to see it fill up.` });
      const w = MA.map((v) => (v > t ? v : 0));
      S.push({ rows: [{ label: "a", vals: MA, cls: hl(mask) }, { label: `np.where(a > ${t}, a, 0)`, vals: w, cls: Object.fromEntries(w.map((v, i) => [i, mask[i] ? "done" : "dim"])) }], side: { "shape kept": "(8,)" }, msg: "`np.where(cond, x, y)` keeps the **shape** and chooses per position: x where the condition is True, y elsewhere. Use it to clip or replace values, instead of removing them." });
      const idx = (input.match(/-?\d+/g) || []).map(Number);
      const bad = idx.find((i) => i >= n || i < -n);
      if (!idx.length) throw new Error("no indices");
      if (bad !== undefined) {
        S.push({ rows: [{ label: "a", vals: MA }], side: { index: bad }, msg: `Index **${bad}** is out of range: valid positions are −${n} … ${n - 1}. NumPy raises **IndexError: index ${bad} is out of bounds for axis 0 with size ${n}**.` });
      } else {
        const pos = idx.map((i) => (i < 0 ? i + n : i));
        const cls = {};
        pos.forEach((p) => (cls[p] = "hl"));
        const dup = new Set(pos).size < pos.length;
        S.push({ rows: [{ label: "a", vals: MA, cls }, { label: `a[[${idx.join(", ")}]]  (a copy)`, vals: pos.map((p) => MA[p]), cls: Object.fromEntries(pos.map((_, i) => [i, "done"])) }], side: { "indices": `[${idx.join(", ")}]`, "result.shape": `(${idx.length},)` }, msg: `**Fancy indexing**: an integer list means "give me these positions, in this order". The result follows **your list**, not the array order${dup ? ", and a repeated index gives a repeated value" : ""}. Negative numbers count from the end (−1 is the last item). It also returns a copy. Edit the box and press Run.` });
      }
      return S;
    },
  };

  /* ================================================================== 4. groupby: split - apply - combine */
  const GB = [["Pune", 10], ["Delhi", 20], ["Pune", 30], ["Goa", 5], ["Delhi", 40], ["Pune", 20]];
  const GBOP = [
    ["sum", (a) => a.reduce((s, v) => s + v, 0)],
    ["mean", (a) => a.reduce((s, v) => s + v, 0) / a.length],
    ["count", (a) => a.length],
    ["max", (a) => Math.max(...a)],
  ];
  V.pyml_groupby = {
    title: "groupby: split, apply, combine",
    params: [{ id: "op", label: "aggregation", min: 0, max: 3, step: 1, value: 1, fmt: (v) => GBOP[+v][0] }],
    gen(_, P) {
      const [opName, f] = GBOP[P.op], keys = ["Delhi", "Goa", "Pune"], colr = { Delhi: "purple", Goa: "amber", Pune: "green" };
      const groups = Object.fromEntries(keys.map((k) => [k, GB.filter((r) => r[0] === k).map((r) => r[1])]));
      const res = Object.fromEntries(keys.map((k) => [k, f(groups[k])]));
      const fmt = (v) => (Number.isInteger(v) ? String(v) : r3(v));
      const rh = 26, gy = {}, gh = {};
      let y = 20;
      keys.forEach((k) => { gh[k] = groups[k].length * rh + 34; gy[k] = y; y += gh[k] + 10; });
      const H = y + 4, W = 640;
      const draw = (stage) => {
        let g = txt(14, 14, "df", { a: "start", w: 700, s: 12, c: "var(--muted)" });
        g += rect(10, 24, 170, rh + 4 + GB.length * rh, "var(--panel)", "var(--border)", false, 6);
        g += txt(50, 42, "city", { w: 700, s: 12 }) + txt(145, 42, "sales", { w: 700, s: 12 });
        GB.forEach(([k, v], i) => {
          const yy = 50 + i * rh, c = colr[k];
          if (stage >= 1) g += rect(14, yy, 162, rh - 3, soft(c, 28), "var(--" + c + ")", false, 4);
          g += txt(50, yy + 17, k, { s: 12 }) + txt(145, yy + 17, String(v), { s: 12, mono: true });
        });
        if (stage >= 1) {
          g += txt(300, 14, "split: one mini-table per key", { s: 11, c: "var(--muted)" });
          keys.forEach((k) => {
            const c = colr[k], yy = gy[k] + 20;
            g += `<path d="M184,${100} C 215,${100} 215,${yy + gh[k] / 2} 232,${yy + gh[k] / 2}" style="fill:none;stroke:var(--${c});stroke-width:1.4;stroke-dasharray:4 3"/>`;
            g += rect(236, yy, 190, gh[k], soft(c, 10), "var(--" + c + ")", false, 8) + txt(250, yy + 19, k, { a: "start", w: 700, s: 12 });
            groups[k].forEach((v, i) => { g += rect(262 + i * 40, yy + 28, 34, rh - 2, soft(c, 30), "var(--" + c + ")", false, 4) + txt(279 + i * 40, yy + 28 + 17, String(v), { s: 12, mono: true }); });
            if (stage >= 2) g += txt(414, yy + gh[k] - 8, `${opName} = ${fmt(res[k])}`, { a: "end", w: 700, s: 12, c: "var(--" + c + ")" });
          });
        }
        if (stage >= 3) {
          g += txt(555, 14, "combine", { s: 11, c: "var(--muted)" });
          g += rect(470, 24, 160, 34 + keys.length * rh, "var(--panel)", "var(--border)", false, 6);
          g += txt(500, 42, "city", { w: 700, s: 12 }) + txt(595, 42, "sales", { w: 700, s: 12 });
          keys.forEach((k, i) => {
            const yy = 52 + i * rh;
            g += rect(474, yy, 152, rh - 3, soft(colr[k], 28), "var(--" + colr[k] + ")", false, 4) + txt(500, yy + 17, k, { s: 12 }) + txt(595, yy + 17, fmt(res[k]), { s: 12, mono: true });
          });
          keys.forEach((k) => { g += `<path d="M430,${gy[k] + 20 + gh[k] / 2} C 450,${gy[k] + 20 + gh[k] / 2} 450,${64 + keys.indexOf(k) * rh} 468,${64 + keys.indexOf(k) * rh}" style="fill:none;stroke:var(--${colr[k]});stroke-width:1.4"/>`; });
        }
        return svgWrap(W, H + 24, g);
      };
      const code = `df.groupby("city")["sales"].${opName}()`;
      const msgs = [
        `A small table: six sales, each with a city. We want one number per city. The code is \`${code}\`.`,
        "**Split**: pandas collects the rows that share the same key (\"city\") into separate groups. Groups are sorted by key by default (Delhi, Goa, Pune).",
        `**Apply**: the aggregation \`${opName}\` runs **inside each group on its own**: Delhi ${fmt(res.Delhi)}, Goa ${fmt(res.Goa)}, Pune ${fmt(res.Pune)}.`,
        `**Combine**: the per-group answers are stitched into a new table whose index is the group key (a Series here, because we picked one column). Try another aggregation with the slider: the split never changes, only the apply step.`,
      ];
      return msgs.map((m, i) => ({ svg: draw(i), side: { code, groups: keys.length, rows: GB.length }, msg: m }));
    },
  };

  /* ------------------------------------------------------------------ glossary */
  Object.assign(window.GLOSSARY, {
    "ndarray": "NumPy's array type: a block of same-typed numbers with a shape, such as (3, 4) for 3 rows and 4 columns.",
    "dtype": "The data type of every element in a NumPy array or pandas column, e.g. int64, float64, bool. It decides memory use and how arithmetic behaves.",
    "shape": "A tuple with the size of each dimension of an array: (3, 4) means 3 rows and 4 columns; (5,) is a flat list of 5.",
    "view": "A NumPy array that shares memory with another one. Changing the view changes the original too. Basic slicing makes views.",
    "boolean mask": "An array of True/False values used as an index: a[a > 0] keeps only the items where the test is True.",
    "fancy indexing": "Indexing an array with a list or array of integers, e.g. a[[3, 0, 3]]. It picks those positions in that order and returns a copy.",
    "axis": "One dimension of an array. In a 2-D array axis 0 runs down the rows and axis 1 runs across the columns. sum(axis=0) collapses the rows.",
    "ufunc": "A universal function: a NumPy function like np.exp or np.add that works element by element over a whole array, in fast compiled code.",
    "dataframe": "pandas' table type: labelled columns (each a Series with its own dtype) and a row index.",
    "series": "pandas' one-dimensional labelled array: a column of values plus an index.",
    "groupby": "A pandas operation that splits rows by the value of a key, applies a function to each group, then combines the results.",
    "nan": "'Not a number': the float value NumPy and pandas use for a missing or undefined number. NaN is not equal to anything, even itself.",
    "estimator": "In scikit-learn, any object that learns from data with .fit(), e.g. a model or a scaler.",
    "pipeline": "A scikit-learn object that chains preprocessing steps and a final model so they are fitted and applied together.",
    "random seed": "A number that starts a random number generator, so the 'random' numbers come out the same every run.",
    "virtual environment": "An isolated folder of Python packages for one project, so versions of libraries do not clash between projects.",
  });
})();
