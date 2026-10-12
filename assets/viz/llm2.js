/* LLMs part 2 (using LLMs): visualizers, diagrams and glossary terms.
   Loaded on every doc page and on Practice, after assets/viz.js. Every number shown is computed, not faked. */
(function () {
  const V = Viz.defs, Plot = Viz.Plot;

  /* ------------------------------------------------------------------ private helpers */
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const f2 = (v, d = 2) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : +v).toFixed(d);
  const wrap = (W, H, inner) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">${inner}</svg>`;
  const T = (x, y, s, style = "", anchor = "middle") => `<text x="${x}" y="${y}" text-anchor="${anchor}" style="font-size:11.5px;fill:var(--muted);${style}">${esc(s)}</text>`;
  const TX = "fill:var(--text);";
  const wrapText = (s, n) => { const out = []; let cur = ""; for (const w of s.split(/\s+/)) { if (cur && (cur + " " + w).length > n) { out.push(cur); cur = w; } else cur = cur ? cur + " " + w : w; } if (cur) out.push(cur); return out; };
  const tokenize = (s) => (s.toLowerCase().match(/[a-z0-9]+/g) || []);
  const STOP = new Set("the a an of to is are and in on for if be it its how does do what which at by with or as my i can until than this that after".split(" "));
  const terms = (s) => tokenize(s).filter((w) => !STOP.has(w));

  /* ------------------------------------------------------------------ the toy corpus (8 short documents) */
  const DOCS = [
    "Nimbus bikes come with a two year warranty covering the frame and the motor.",
    "To reset the display hold the power button for ten seconds until the screen blinks.",
    "The battery takes four hours to charge fully and lasts about sixty kilometres per charge.",
    "Returns are accepted within thirty days if the bike is unused and in its original box.",
    "Tyre pressure should be kept between fifty and sixty five psi for city riding.",
    "Free shipping applies to orders above two hundred euros and takes five working days.",
    "The motor is rated at two hundred fifty watts, the legal limit for e-bikes in the EU.",
    "Clean the chain every month and apply lubricant after riding in the rain.",
  ];
  const DT = DOCS.map(terms);
  const N = DOCS.length;
  const vocab = [...new Set(DT.flat())].sort();
  const df = {}; vocab.forEach((w) => { df[w] = DT.filter((d) => d.includes(w)).length; });
  const idf = (w) => Math.log((1 + N) / (1 + (df[w] || 0))) + 1;           // smoothed idf (as in scikit-learn)
  const tfidf = (toks) => { const v = {}; toks.forEach((w) => { if (df[w]) v[w] = (v[w] || 0) + 1; }); Object.keys(v).forEach((w) => { v[w] *= idf(w); }); const n = Math.sqrt(sum(Object.values(v).map((x) => x * x))) || 1; Object.keys(v).forEach((w) => { v[w] /= n; }); return v; };
  const dvec = DT.map(tfidf);
  const dotS = (a, b) => sum(Object.keys(a).map((w) => (b[w] ? a[w] * b[w] : 0)));
  const snip = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

  /* ================================================================== 1. The RAG pipeline step by step */
  V.llm2_rag = {
    title: "RAG step by step: question, embed, nearest chunks, prompt, answer",
    input: "how long does the battery take to charge",
    help: "a question about the toy bike manual shown in the first step (try: warranty, shipping, tyre pressure, rain)",
    params: [{ id: "k", label: "chunks retrieved k", min: 1, max: 4, step: 1, value: 2 }],
    gen(input, P) {
      const q = input.trim() || "how long does the battery take to charge";
      const qt = terms(q), qv = tfidf(qt);
      const sc = DOCS.map((_, i) => dotS(qv, dvec[i]));
      const order = sc.map((s, i) => i).sort((a, b) => sc[b] - sc[a] || a - b);
      const k = P.k, top = order.slice(0, k).filter((i) => sc[i] > 0);
      const S = [];
      /* step 0: the knowledge base and the question */
      let s0 = T(10, 16, "Knowledge base: 8 chunks (the model has never seen them)", TX + "font-weight:600", "start");
      DOCS.forEach((d, i) => { s0 += T(10, 38 + i * 19, `d${i + 1}`, "fill:var(--accent);font-weight:600", "start") + T(34, 38 + i * 19, d, "", "start"); });
      s0 += T(10, 212, "Question:", TX + "font-weight:600", "start") + T(78, 212, q, "fill:var(--amber);font-weight:600", "start");
      S.push({ svg: wrap(620, 226, s0), side: { chunks: N, "vocabulary terms": vocab.length, "question terms kept": qt.length },
        msg: `RAG starts from text the model does not know. Here: 8 short chunks of a toy bike manual and **your question**. Common words ("the", "does", "to") are dropped, leaving the terms **${qt.join(", ") || "(none)"}**. Everything that follows is plain, deterministic search plus one prompt.` });
      /* step 1: embed the question */
      const ws = Object.keys(qv).sort((a, b) => qv[b] - qv[a]);
      let s1 = T(10, 16, "Question vector (TF-IDF, length 1). Only terms that occur somewhere in the 8 chunks count.", TX + "font-weight:600", "start");
      ws.forEach((w, i) => { const y = 34 + i * 24, wd = qv[w] * 330; s1 += T(120, y + 13, w, TX, "end") + `<rect x="128" y="${y}" width="${wd}" height="17" rx="3" style="fill:var(--accent);fill-opacity:.8"/>` + T(134 + wd, y + 13, `${f2(qv[w], 3)}   (idf ${f2(idf(w), 2)}, in ${df[w]} chunk${df[w] > 1 ? "s" : ""})`, "", "start"); });
      const unk = qt.filter((w) => !df[w]);
      if (!ws.length) s1 += T(10, 50, "No question term occurs in the knowledge base: nothing to match.", "", "start");
      if (unk.length) s1 += T(10, 40 + ws.length * 24 + 14, "ignored (not in any chunk): " + unk.join(", "), "", "start");
      S.push({ svg: wrap(620, Math.max(90, 70 + ws.length * 24), s1), side: { "non-zero entries": ws.length, "of vocabulary": vocab.length },
        msg: `**Embed**: turn the question into a vector. This toy uses TF-IDF: each term gets weight $\\text{tf}\\times\\text{idf}$, where rare terms (small document frequency) get a larger idf, then the vector is scaled to length 1. A real system uses a neural embedding model ([embeddings](llm.html#embeddings-search)) and the same chunks were embedded in advance, offline.` });
      /* step 2: score every chunk */
      let s2 = T(10, 16, "Cosine similarity between the question and each chunk, best first", TX + "font-weight:600", "start");
      const mx = Math.max(0.01, ...sc);
      order.forEach((i, r) => { const y = 30 + r * 24, wd = (sc[i] / mx) * 200, on = top.includes(i);
        s2 += T(24, y + 13, `d${i + 1}`, TX + "font-weight:600", "end") + `<rect x="30" y="${y}" width="${Math.max(1, wd)}" height="17" rx="3" style="fill:${on ? "var(--green)" : "var(--muted)"};fill-opacity:${on ? 0.85 : 0.35}"/>` + T(36 + wd, y + 13, f2(sc[i], 3), TX, "start") + T(300, y + 13, snip(DOCS[i], 66), on ? TX : "", "start"); });
      S.push({ svg: wrap(620, 30 + N * 24 + 8, s2), side: { "top chunk": top.length ? "d" + (top[0] + 1) : "none", "its cosine": top.length ? f2(sc[top[0]], 3) : "0", "retrieved (k)": top.length, "lowest score": f2(Math.min(...sc), 3) },
        msg: `**Nearest chunks**: the cosine of two unit vectors is just their [dot product](math.html#dot-product), so scoring all chunks is one matrix-vector product. The best ${top.length ? "**" + top.length + "**" : "0"} (green) go on to the next step. Chunks that share no term with the question score exactly 0. A real system searches millions of chunks with an approximate index, but the idea is identical.` });
      /* step 3: the prompt */
      const lines = [["System: Answer using only the context. If the answer is not there, say you do not know. Cite chunk ids.", "var(--purple)"], ["", ""], ["Context:", "var(--green)"]];
      top.forEach((i) => wrapText(`[d${i + 1}] ${DOCS[i]}`, 88).forEach((l) => lines.push([l, "var(--green)"])));
      if (!top.length) lines.push(["(nothing retrieved)", "var(--muted)"]);
      lines.push(["", ""], [`Question: ${q}`, "var(--amber)"]);
      const flat = []; lines.forEach(([l, c]) => wrapText(l || " ", 88).forEach((x) => flat.push([x, c])));
      let s3 = T(10, 16, "The prompt the LLM actually receives", TX + "font-weight:600", "start") + `<rect x="10" y="26" width="600" height="${flat.length * 17 + 14}" rx="8" style="fill:var(--panel-2);stroke:var(--border)"/>`;
      flat.forEach(([l, c], i) => { s3 += T(22, 46 + i * 17, l, `fill:${c || "var(--text)"};font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px`, "start"); });
      S.push({ svg: wrap(620, flat.length * 17 + 48, s3), side: { "chunks in prompt": top.length, "prompt characters": flat.reduce((a, [l]) => a + l.length, 0) },
        msg: `**Build the prompt**: instructions (purple), the retrieved chunks as context (green) and the question (amber). The LLM was never retrained; it just reads this text. Telling it to answer only from the context and to say "I do not know" otherwise is the main defence against [hallucination](llm.html#hallucinations).` });
      /* step 4: answer */
      const ans = top.length ? `${DOCS[top[0]]} [d${top[0] + 1}]` : "I do not know: the provided context does not contain the answer.";
      let s4 = T(10, 16, "Generated answer (a real LLM would paraphrase; this toy copies the best chunk)", TX + "font-weight:600", "start") + `<rect x="10" y="26" width="600" height="${wrapText(ans, 84).length * 20 + 18}" rx="8" style="fill:color-mix(in srgb, var(--green) 14%, transparent);stroke:var(--green)"/>`;
      wrapText(ans, 84).forEach((l, i) => { s4 += T(22, 50 + i * 20, l, TX + "font-size:13px", "start"); });
      S.push({ svg: wrap(620, wrapText(ans, 84).length * 20 + 56, s4), side: { grounded: top.length ? "yes, cites d" + (top[0] + 1) : "no: refuses" },
        msg: `**Answer**. With good retrieval the answer is grounded in a chunk you can show as a citation. If retrieval brings back the wrong chunk, the model usually answers confidently from it anyway, so most RAG failures are **retrieval** failures. Change the question to something the manual does not cover and watch the refusal path.` });
      return S;
    },
  };

  /* ================================================================== 2. Chunking with size and overlap */
  const PARA = "The Nimbus battery is a lithium-ion pack that sits inside the frame. Charge it at room temperature with the supplied charger only. A full charge takes about four hours and gives roughly sixty kilometres of range. Avoid leaving the battery fully empty for weeks because deep discharge shortens its life. For winter storage keep it about half charged in a cool dry place. If the battery does not charge, check the plug and the charger light before contacting support. Support can replace a faulty pack during the two year warranty.";
  const PW = PARA.split(/\s+/);
  const chunkSpans = (n, size, ovp) => { const ov = Math.min(size - 1, Math.round((size * ovp) / 100)), step = Math.max(1, size - ov), out = []; for (let s = 0; ; s += step) { out.push([s, Math.min(n, s + size)]); if (s + size >= n) break; } return { spans: out, ov, step }; };
  V.llm2_chunking = {
    title: "Chunking: size and overlap",
    params: [{ id: "size", label: "chunk size (words)", min: 10, max: 60, step: 2, value: 30 }, { id: "ov", label: "overlap (% of size)", min: 0, max: 50, step: 10, value: 20 }],
    gen(_, P) {
      const n = PW.length, { spans, ov, step } = chunkSpans(n, P.size, P.ov);
      const endsSent = (i) => /[.]$/.test(PW[i]), startsSent = (i) => i === 0 || endsSent(i - 1);
      const cut = spans.filter(([a, b]) => !startsSent(a) || !endsSent(b - 1)).length;
      const total = sum(spans.map(([a, b]) => b - a));
      /* text layout */
      const W = 620, LH = 22, x0 = 10, widths = PW.map((w) => w.length * 6.7 + 5), pos = [];
      let x = x0, line = 0;
      PW.forEach((w, i) => { if (x + widths[i] > W - 10) { x = x0; line++; } pos.push([x, line]); x += widths[i] + 4; });
      const lines = line + 1, rows = spans.length, rh = rows > 12 ? 7 : 11, barsH = rows * rh + 28, textY = barsH + 16;
      const draw = (cur) => {
        let s = T(10, 14, "Each bar is one chunk along the paragraph (left = first word, right = last word)", TX + "font-weight:600", "start");
        const bx = (i) => 10 + (i / n) * 600;
        spans.forEach(([a, b], r) => {
          const on = cur == null || cur === r, ovl = cur != null && cur > 0 && r === cur - 1;
          s += `<rect x="${bx(a)}" y="${22 + r * rh}" width="${Math.max(2, bx(b) - bx(a))}" height="${rh - 2}" rx="2" style="fill:${cur === r ? "var(--accent)" : "var(--muted)"};fill-opacity:${cur === r ? 0.9 : on ? 0.45 : 0.25}"/>`;
          if (cur != null && r === cur && cur > 0) { const pa = spans[cur - 1]; const o0 = a, o1 = Math.min(pa[1], b); if (o1 > o0) s += `<rect x="${bx(o0)}" y="${22 + r * rh}" width="${bx(o1) - bx(o0)}" height="${rh - 2}" rx="2" style="fill:var(--amber);fill-opacity:.95"/>`; }
          if (rh >= 11) s += T(bx(a) + 3, 22 + r * rh + 8, String(r + 1), "font-size:9.5px;fill:var(--panel)", "start");
        });
        PW.forEach((w, i) => { if (endsSent(i)) s += `<line x1="${bx(i + 1)}" y1="20" x2="${bx(i + 1)}" y2="${22 + rows * rh}" style="stroke:var(--red);stroke-width:1;stroke-dasharray:2 3"/>`; });
        s += T(10, 22 + rows * rh + 12, "dotted red lines = sentence ends", "font-size:10.5px", "start");
        /* text with highlights */
        const cs = cur == null ? null : spans[cur], ps = cur != null && cur > 0 ? spans[cur - 1] : null;
        PW.forEach((w, i) => {
          const [px, pl] = pos[i], inC = cs && i >= cs[0] && i < cs[1], inO = inC && ps && i < ps[1];
          if (inC) s += `<rect x="${px - 2}" y="${textY + pl * LH - 12}" width="${widths[i] + 4}" height="${LH - 3}" rx="3" style="fill:${inO ? "var(--amber)" : "var(--accent)"};fill-opacity:${inO ? 0.5 : 0.25}"/>`;
          s += `<text x="${px}" y="${textY + pl * LH + 2}" style="font-size:12.5px;fill:${inC || cur == null ? "var(--text)" : "var(--muted)"}">${esc(w)}</text>`;
        });
        return wrap(W, textY + lines * LH + 6, s);
      };
      const side0 = { words: n, chunks: spans.length, "step (words)": step, "overlap (words)": ov, "chunks cutting a sentence": cut + " of " + spans.length, "stored words / original": f2(total / n, 2) + "×" };
      const S = [{ svg: draw(null), side: side0,
        msg: `A paragraph of **${n} words** cut into windows of **${P.size}** words that advance **${step}** words at a time, so neighbouring chunks share **${ov}** words. Overlap makes it likelier that a fact cut at a boundary still appears whole in one chunk, but you store **${f2(total / n, 2)}×** the text. Notice how many chunks start or end in the middle of a sentence (${cut}). Press Next to walk through the chunks.` }];
      spans.forEach(([a, b], r) => {
        S.push({ svg: draw(r), side: { chunk: `${r + 1} of ${spans.length}`, words: `${a + 1} to ${b}`, "size": b - a, "shared with previous": r > 0 ? Math.max(0, spans[r - 1][1] - a) : 0, "starts at sentence start": startsSent(a) ? "yes" : "no", "ends at sentence end": endsSent(b - 1) ? "yes" : "no" },
          msg: `**Chunk ${r + 1}** (blue): words ${a + 1} to ${b}. ${r > 0 ? `The amber part is the overlap, ${Math.max(0, spans[r - 1][1] - a)} words repeated from chunk ${r}.` : "The first chunk has nothing before it, so no overlap."} ${(!startsSent(a) || !endsSent(b - 1)) ? "This chunk **cuts a sentence**, so its embedding mixes half-ideas." : "This chunk lines up with sentence boundaries."}` });
      });
      return S;
    },
  };

  /* ================================================================== 3. Embedding space + top-k by cosine */
  const PTS = [["cat", 14, .9], ["dog", 22, .8], ["horse", 30, .95], ["tiger", 8, .5], ["lion", 34, .6],
    ["pizza", 105, .9], ["pasta", 114, .7], ["bread", 98, .6], ["apple", 125, .85], ["soup", 118, .4],
    ["car", 195, .9], ["bus", 205, .7], ["truck", 188, .55], ["train", 214, .8], ["bike", 200, .35],
    ["soccer", 285, .9], ["tennis", 295, .7], ["golf", 275, .5], ["rugby", 305, .8], ["chess", 265, .45]]
    .map(([w, a, r]) => ({ w, x: r * Math.cos((a * Math.PI) / 180), y: r * Math.sin((a * Math.PI) / 180), grp: Math.floor(PTS_GRP(a)) }));
  function PTS_GRP(a) { return a < 70 ? 0 : a < 160 ? 1 : a < 245 ? 2 : 3; }
  V.llm2_embspace = {
    title: "Embedding space: a query and its nearest neighbours by cosine",
    params: [{ id: "ang", label: "query direction (degrees)", min: 0, max: 355, step: 5, value: 110 }, { id: "len", label: "query length", min: 0.3, max: 1.5, step: 0.05, value: 0.4 }, { id: "k", label: "neighbours k", min: 1, max: 8, step: 1, value: 4 }],
    gen(_, P) {
      const qa = (P.ang * Math.PI) / 180, qx = P.len * Math.cos(qa), qy = P.len * Math.sin(qa), qn = Math.hypot(qx, qy);
      const cos = PTS.map((p) => (p.x * qx + p.y * qy) / (Math.hypot(p.x, p.y) * qn));
      const eu = PTS.map((p) => Math.hypot(p.x - qx, p.y - qy));
      const byC = PTS.map((_, i) => i).sort((a, b) => cos[b] - cos[a]), byE = PTS.map((_, i) => i).sort((a, b) => eu[a] - eu[b]);
      const top = byC.slice(0, P.k), euTop = byE[0];
      const W = 560, H = 450, cx = W / 2, cy = H / 2 + 6, sc = 185;
      const X = (x) => cx + x * sc, Y = (y) => cy - y * sc;
      let s = `<circle cx="${cx}" cy="${cy}" r="${sc}" style="fill:none;stroke:var(--border);stroke-dasharray:3 4"/><line x1="10" y1="${cy}" x2="${W - 10}" y2="${cy}" style="stroke:var(--border)"/><line x1="${cx}" y1="14" x2="${cx}" y2="${H - 6}" style="stroke:var(--border)"/>`;
      s += T(W - 12, cy - 6, "dim 1", "font-size:10.5px", "end") + T(cx + 6, 24, "dim 2", "font-size:10.5px", "start");
      const gcol = ["var(--cls-a)", "var(--amber)", "var(--purple)", "var(--red)"];
      top.forEach((i) => { s += `<line x1="${cx}" y1="${cy}" x2="${X(PTS[i].x)}" y2="${Y(PTS[i].y)}" style="stroke:var(--green);stroke-width:1.5;opacity:.7"/>`; });
      s += `<line x1="${cx}" y1="${cy}" x2="${X(qx)}" y2="${Y(qy)}" style="stroke:var(--text);stroke-width:2"/>`;
      PTS.forEach((p, i) => { const on = top.includes(i);
        s += `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="${on ? 7 : 5}" style="fill:${gcol[p.grp]};fill-opacity:${on ? 1 : 0.55};stroke:${on ? "var(--green)" : "none"};stroke-width:3"/>` + T(X(p.x) + 9, Y(p.y) + 4, p.w, `font-size:11px;fill:var(--${on ? "text" : "muted"});${on ? "font-weight:700" : ""}`, "start"); });
      if (!top.includes(euTop)) s += `<circle cx="${X(PTS[euTop].x)}" cy="${Y(PTS[euTop].y)}" r="11" style="fill:none;stroke:var(--red);stroke-width:2;stroke-dasharray:3 2"/>`;
      s += `<polygon points="${X(qx)},${Y(qy) - 10} ${X(qx) + 9},${Y(qy) + 7} ${X(qx) - 9},${Y(qy) + 7}" style="fill:var(--text);stroke:var(--panel);stroke-width:1.5"/>` + T(X(qx) + 12, Y(qy) - 8, "query", TX + "font-weight:700", "start");
      const side = {}; top.forEach((i, r) => { side[`#${r + 1} ${PTS[i].w}`] = f2(cos[i], 3); });
      side["Euclidean nearest"] = PTS[euTop].w + (top.includes(euTop) ? " (also in top-k)" : " (not in top-k)");
      return [{ svg: wrap(W, H, s), side,
        msg: `Each word is a point (a real embedding has hundreds or thousands of dimensions; two are drawn here). Cosine similarity compares only the **angle** to the query: $\\cos\\theta=\\frac{q\\cdot d}{\\lVert q\\rVert\\lVert d\\rVert}$. The green rays are the top-${P.k} by cosine. Drag **query length** and the cosine ranking does not change at all, because only direction matters; the Euclidean-nearest point (dashed red ring, when it differs) can. That is why vector databases usually normalise embeddings. Turn the direction to land between two clusters and the neighbours come from both.` }];
    },
  };

  /* ================================================================== 4. Hybrid search with Reciprocal Rank Fusion */
  const BM_K1 = 1.5, BM_B = 0.75, avgdl = sum(DT.map((d) => d.length)) / N;
  const bidf = (w) => Math.log(1 + (N - (df[w] || 0) + 0.5) / ((df[w] || 0) + 0.5));
  const bm25 = (qt, i) => sum([...new Set(qt)].map((w) => { const tf = DT[i].filter((x) => x === w).length; if (!tf) return 0; return bidf(w) * (tf * (BM_K1 + 1)) / (tf + BM_K1 * (1 - BM_B + BM_B * DT[i].length / avgdl)); }));
  const CONCEPTS = ["battery", "charge", "warranty", "return", "shipping", "motor", "tyre", "care", "reset"];
  const LEX = {
    battery: { battery: 1, charge: .4 }, cell: { battery: .8 }, power: { battery: .6, reset: .3, motor: .2 }, lasts: { battery: .5 }, kilometres: { battery: .5 }, range: { battery: .5 }, juice: { battery: .6, charge: .4 },
    charge: { charge: 1, battery: .5 }, charged: { charge: 1, battery: .5 }, charging: { charge: 1, battery: .5 }, charger: { charge: 1, battery: .4 }, recharge: { charge: 1, battery: .5 }, refill: { charge: .9, battery: .3 }, fully: { charge: .5 }, full: { charge: .5 }, hours: { charge: .2 }, plug: { charge: .5 },
    warranty: { warranty: 1 }, guarantee: { warranty: 1 }, covering: { warranty: .6 }, covers: { warranty: .6 }, frame: { warranty: .2 }, year: { warranty: .3 }, years: { warranty: .3 },
    returns: { return: 1 }, return: { return: 1 }, refund: { return: 1 }, send: { return: .4, shipping: .2 }, back: { return: .4 }, unused: { return: .6 }, box: { return: .4, shipping: .2 }, original: { return: .4 }, accepted: { return: .4 },
    shipping: { shipping: 1 }, delivery: { shipping: 1 }, deliver: { shipping: 1 }, postage: { shipping: 1 }, orders: { shipping: .5 }, order: { shipping: .5 }, ship: { shipping: 1 }, euros: { shipping: .3 }, working: { shipping: .3 },
    days: { shipping: .3, return: .3 },
    motor: { motor: 1 }, engine: { motor: 1 }, watts: { motor: .8 }, rated: { motor: .5 }, speed: { motor: .5 }, legal: { motor: .3 }, limit: { motor: .3 },
    tyre: { tyre: 1 }, tyres: { tyre: 1 }, tire: { tyre: 1 }, tires: { tyre: 1 }, wheel: { tyre: .7 }, wheels: { tyre: .7 }, pressure: { tyre: .8 }, psi: { tyre: .8 }, air: { tyre: .6 }, pump: { tyre: .6 },
    chain: { care: 1 }, clean: { care: 1 }, lubricant: { care: 1 }, oil: { care: .9 }, rain: { care: .5 }, wash: { care: 1 }, maintenance: { care: 1 }, service: { care: .6 }, dirty: { care: .8 },
    reset: { reset: 1 }, restart: { reset: 1 }, display: { reset: .8 }, screen: { reset: .8 }, button: { reset: .6 }, blinks: { reset: .5 }, frozen: { reset: .8 }, hold: { reset: .3 },
  };
  const cvec = (toks) => { const v = CONCEPTS.map(() => 0); toks.forEach((w) => { const e = LEX[w]; if (e) CONCEPTS.forEach((c, j) => { v[j] += e[c] || 0; }); }); const n = Math.sqrt(sum(v.map((x) => x * x))); return n ? v.map((x) => x / n) : v; };
  const DC = DOCS.map((d) => cvec(tokenize(d)));
  V.llm2_hybrid = {
    title: "Hybrid search: keyword (BM25) + vector, fused with RRF",
    input: "how do I refill the power cell",
    help: "try: how do I refill the power cell (lists disagree) | how long does the power cell take to charge (they agree) | bike warranty | what is the legal motor limit | wheels air",
    params: [{ id: "rk", label: "RRF constant k", min: 1, max: 100, step: 1, value: 60 }],
    gen(input, P) {
      const q = input.trim() || "how do I refill the power cell";
      const qt = terms(q), qc = cvec(tokenize(q)), TOPN = 5;
      const kw = DOCS.map((_, i) => bm25(qt, i)), vs = DOCS.map((_, i) => sum(qc.map((x, j) => x * DC[i][j])));
      const rank = (sc) => { const o = sc.map((_, i) => i).filter((i) => sc[i] > 1e-9).sort((a, b) => sc[b] - sc[a] || a - b).slice(0, TOPN); const r = {}; o.forEach((i, p) => { r[i] = p + 1; }); return { o, r }; };
      const A = rank(kw), B = rank(vs);
      const fused = DOCS.map((_, i) => (A.r[i] ? 1 / (P.rk + A.r[i]) : 0) + (B.r[i] ? 1 / (P.rk + B.r[i]) : 0));
      const F = rank(fused);
      const rowH = 44, y0 = 60, colX = [10, 232, 454], cw = 150;
      const yOf = (p) => y0 + (p - 1) * rowH;
      let s = T(colX[0] + cw / 2, 20, "Keyword (BM25)", TX + "font-weight:700") + T(colX[1] + cw / 2, 20, "Fused (RRF)", TX + "font-weight:700") + T(colX[2] + cw / 2, 20, "Vector (concepts)", TX + "font-weight:700");
      s += T(colX[0] + cw / 2, 36, "exact words only", "font-size:10.5px") + T(colX[1] + cw / 2, 36, `score = Σ 1/(${P.rk} + rank)`, "font-size:10.5px") + T(colX[2] + cw / 2, 36, "meaning, synonyms", "font-size:10.5px");
      const line = (x1, y1, x2, y2, col) => `<path d="M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}" style="fill:none;stroke:${col};stroke-width:2;opacity:.55"/>`;
      F.o.forEach((i) => { const yf = yOf(F.r[i]) + 16;
        if (A.r[i]) s += line(colX[0] + cw, yOf(A.r[i]) + 16, colX[1], yf, "var(--accent)");
        if (B.r[i]) s += line(colX[2], yOf(B.r[i]) + 16, colX[1] + cw, yf, "var(--purple)"); });
      const box = (x, p, i, sc, col) => `<rect x="${x}" y="${yOf(p)}" width="${cw}" height="32" rx="7" style="fill:color-mix(in srgb, ${col} 16%, var(--panel-2));stroke:${col}"/>` + T(x + 8, yOf(p) + 14, `#${p}  d${i + 1}`, TX + "font-weight:700", "start") + T(x + cw - 8, yOf(p) + 14, f2(sc, sc < 0.1 ? 4 : 2), "", "end") + T(x + 8, yOf(p) + 27, snip(DOCS[i], 24), "font-size:10px", "start");
      A.o.forEach((i) => { s += box(colX[0], A.r[i], i, kw[i], "var(--accent)"); });
      B.o.forEach((i) => { s += box(colX[2], B.r[i], i, vs[i], "var(--purple)"); });
      F.o.forEach((i) => { s += box(colX[1], F.r[i], i, fused[i], "var(--green)"); });
      if (!A.o.length) s += T(colX[0] + cw / 2, y0 + 20, "no word matches", "", "middle");
      if (!B.o.length) s += T(colX[2] + cw / 2, y0 + 20, "no concept matches", "", "middle");
      const best = (i) => Math.min(A.r[i] || 99, B.r[i] || 99);
      const moved = F.o.map((i) => `d${i + 1}: best single rank ${best(i)} to fused ${F.r[i]}`).slice(0, 3).join("; ");
      const note = (A.o[0] != null && B.o[0] != null && A.o[0] !== B.o[0]) ? `The two lists **disagree** on the winner (d${A.o[0] + 1} versus d${B.o[0] + 1}); fusion cannot know which is right, but it keeps both candidates at the top, which is what a [reranker](llm.html#rag-advanced) needs.` : (A.o.length && B.o.length ? "Both lists agree on the winner." : "Only one list found anything, so the fused list is just that list.");
      return [{ svg: wrap(640, y0 + TOPN * rowH + 6, s), side: { "BM25 top": A.o.length ? "d" + (A.o[0] + 1) : "none", "vector top": B.o.length ? "d" + (B.o[0] + 1) : "none", "fused top": F.o.length ? "d" + (F.o[0] + 1) : "none", "RRF k": P.rk, "query terms": qt.join(" ") },
        msg: `Two retrievers rank the same 8 chunks: **BM25** counts matching words (no stemming, so "charged" does not match "charge"), the **vector** side here is a hand-made toy (a word-to-concept table, so "power cell" is close to "battery"). **Reciprocal Rank Fusion** ignores the raw scores, which live on different scales, and adds $1/(k+\\text{rank})$ for each list a chunk appears in. A chunk ranked well by *both* wins. ${note} Raise $k$ and the top ranks matter less; lower it and the first place counts a lot. ${moved ? "Rank changes: " + moved + "." : ""}` }];
    },
  };

  /* ================================================================== static diagrams */
  const ahead = (id) => `<defs><marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker></defs>`;
  const bxx = (x, y, w, h, label, cls = "node", sub) => `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="8"/><text class="val" x="${x + w / 2}" y="${y + h / 2 + (sub ? -2 : 5)}" text-anchor="middle" style="font-size:12.5px">${esc(label)}</text>` + (sub ? `<text class="lbl" x="${x + w / 2}" y="${y + h / 2 + 13}" text-anchor="middle" style="font-size:10.5px">${esc(sub)}</text>` : "");
  const arr = (x1, y1, x2, y2, id) => `<line class="edge" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#${id})"/>`;
  const tt = (x, y, s, style = "", anchor = "middle") => `<text class="lbl" x="${x}" y="${y}" text-anchor="${anchor}" style="${style}">${esc(s)}</text>`;

  D.llm2_prompt = () => {
    const rows = [
      ["System prompt", "Who the model is, tone, hard rules, what it must never do", "var(--purple)", "same every request: put first (cacheable)"],
      ["Instructions", "The task in plain words and the exact output format wanted", "var(--accent)", "same for a given feature"],
      ["Examples (few-shot)", "2 to 5 input/output pairs that show the format and edge cases", "var(--amber)", "same for a given feature"],
      ["Context", "Retrieved chunks, documents, tool results, conversation history", "var(--green)", "changes every request: put late"],
      ["Question / input", "The user's actual message or the item to process", "var(--red)", "changes every request: put last"],
    ];
    let s = "";
    rows.forEach((r, i) => { const y = 14 + i * 56;
      s += `<rect x="14" y="${y}" width="330" height="46" rx="8" style="fill:color-mix(in srgb, ${r[2]} 15%, var(--panel-2));stroke:${r[2]};stroke-width:1.5"/>` + tt(26, y + 19, r[0], "font-size:13px;font-weight:700;fill:var(--text)", "start") + tt(26, y + 36, r[1], "font-size:10.5px", "start") + tt(358, y + 28, r[3], "font-size:11px;fill:var(--text)", "start"); });
    s += `<path d="M352,20 L352,128" style="stroke:var(--purple);stroke-width:2;fill:none"/>` + `<path d="M352,150 L352,286" style="stroke:var(--green);stroke-width:2;fill:none"/>`;
    s += tt(14, 308, "The model then continues after the last line: its answer is the next tokens.", "font-size:11.5px;font-style:italic", "start");
    return `<svg viewBox="0 0 640 318" width="640" height="318" xmlns="http://www.w3.org/2000/svg" role="img">${s}</svg>`;
  };

  D.llm2_ragflow = () => {
    const id = "l2f";
    let s = ahead(id);
    s += tt(10, 16, "Indexing (offline, once per document change)", "font-size:12px;font-weight:700;fill:var(--text)", "start");
    [["Documents", "PDFs, pages, tickets"], ["Split", "chunks + overlap"], ["Embed", "one vector per chunk"], ["Vector store", "index + metadata"]].forEach((b, i) => { const x = 10 + i * 158; s += bxx(x, 26, 132, 46, b[0], i === 3 ? "node ok" : "node", b[1]); if (i < 3) s += arr(x + 134, 49, x + 156, 49, id); });
    s += tt(10, 108, "Querying (online, every question)", "font-size:12px;font-weight:700;fill:var(--text)", "start");
    [["Question", "user text"], ["Embed", "same model"], ["Search top-k", "nearest chunks"], ["Prompt", "rules + chunks + question"], ["LLM", "answer + sources"]].forEach((b, i) => { const x = 10 + i * 126; s += bxx(x, 118, 106, 46, b[0], i === 4 ? "node hl" : "node", b[1]); if (i < 4) s += arr(x + 108, 141, x + 124, 141, id); });
    s += `<path d="M586,72 L586,96 L270,96 L270,116" style="fill:none;stroke:var(--muted);stroke-dasharray:4 4;stroke-width:1.5" marker-end="url(#${id})"/>` + tt(436, 90, "search runs against the stored vectors", "font-size:10.5px");
    return `<svg viewBox="0 0 640 182" width="640" height="182" xmlns="http://www.w3.org/2000/svg" role="img">${s}</svg>`;
  };

  /* ================================================================== glossary */
  Object.assign(window.GLOSSARY, {
    "prompt": "The text you send to a model, including instructions, examples, context and the question. The model's answer is its continuation.",
    "system prompt": "Standing instructions given before the conversation: the model's role, rules and style. Usually kept the same across requests.",
    "few-shot": "Putting a few worked examples (input and desired output) in the prompt so the model copies the pattern. With zero examples it is called zero-shot.",
    "zero-shot": "Asking a model to do a task with instructions only and no worked examples in the prompt.",
    "in-context learning": "A model picking up a task from examples or instructions in its prompt, without any change to its weights.",
    "structured output": "Model output that follows a machine-readable format such as JSON that matches a schema, so code can use it directly.",
    "json schema": "A standard way to describe the shape of JSON: which fields exist, their types, which are required.",
    "constrained decoding": "Forcing generation to follow a grammar or schema by blocking, at every step, the tokens that would break it.",
    "vector search": "Finding the stored vectors closest to a query vector, usually by cosine similarity or dot product.",
    "vector database": "A store that keeps embedding vectors with their text and metadata and answers nearest-neighbour queries quickly.",
    "nearest neighbour": "The stored item whose vector is closest to the query. k nearest neighbours returns the k closest.",
    "ann": "Approximate nearest neighbour search: indexes that find very close (not guaranteed exact) neighbours much faster than comparing with every vector.",
    "hnsw": "A popular approximate nearest-neighbour index: a layered graph in which a search hops from node to node towards the query.",
    "bi-encoder": "An embedding model that encodes the query and each document separately, so document vectors can be precomputed. Fast, less precise than a cross-encoder.",
    "cross-encoder": "A model that reads the query and one document together and outputs a relevance score. More accurate than a bi-encoder but too slow to run on everything.",
    "reranker": "A second-stage model (often a cross-encoder) that re-orders the top results of a fast first-stage search.",
    "tf-idf": "A weight for a term in a document: how often it occurs there (tf) times how rare it is across documents (idf).",
    "bm25": "A classic keyword ranking function: like TF-IDF, but with saturating term frequency and a correction for document length.",
    "chunking": "Splitting documents into smaller pieces (chunks) before embedding them so retrieval can return just the relevant part.",
    "chunk overlap": "Letting neighbouring chunks share some text so a fact cut at a boundary still appears whole in one of them.",
    "hybrid search": "Combining keyword search (like BM25) and vector search, usually by fusing their rankings.",
    "rrf": "Reciprocal rank fusion: merge several ranked lists by giving each item the sum of 1/(k + rank) over the lists it appears in.",
    "recall@k": "Of all the relevant items, the fraction that appear in the top k results.",
    "mrr": "Mean reciprocal rank: the average of 1/rank of the first relevant result across queries.",
    "grounding": "Basing an answer on supplied source text rather than on what the model remembers, ideally with citations.",
    "citation": "A pointer from a statement in the answer to the source chunk that supports it.",
    "exact match": "An evaluation metric that counts an answer as right only if it equals the reference (after simple normalising).",
    "token f1": "Overlap between the words of an answer and a reference: the harmonic mean of word precision and recall.",
    "llm-as-a-judge": "Using a language model with a written rubric to grade another model's outputs.",
    "rubric": "A written list of criteria, with score levels, used to grade an answer consistently.",
    "jailbreak": "A prompt crafted to make a model ignore its safety training or rules.",
    "indirect prompt injection": "Instructions hidden in content the model reads (a web page, email, document) that try to take over its behaviour.",
    "red teaming": "Deliberately attacking a system, with adversarial prompts, to find safety and security failures before real users do.",
    "multimodal": "Handling more than one kind of input or output, such as text together with images, audio or video.",
    "vision-language model": "A model that takes images as well as text and answers in text.",
    "streaming": "Receiving a model's answer token by token as it is generated instead of waiting for the whole reply.",
    "rate limit": "A cap on how many requests or tokens an API account may use per minute; going over returns an error to retry later.",
    "prompt caching": "Reusing the already-processed prefix of a prompt across requests, which makes repeated long prompts cheaper and faster.",
    "retry with backoff": "Retrying a failed request after waiting, with the wait growing each time (and some randomness), so a busy server is not hammered.",
  });
})();
