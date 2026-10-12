/* Deep Learning, part 1 (basics + training): visualizers, diagrams and glossary terms.
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

  /* ================================================================== 1. Perceptron learning: AND converges, XOR never does */
  V.dl1_perceptron = {
    title: "Perceptron learning: AND works, XOR never does",
    params: [{ id: "data", label: "problem", min: 0, max: 1, step: 1, value: 0, fmt: (v) => (+v ? "XOR" : "AND") }],
    gen(_, P) {
      const xor = +P.data === 1, pts = [[0, 0], [0, 1], [1, 0], [1, 1]], ys = xor ? [0, 1, 1, 0] : [0, 0, 0, 1];
      let w1 = 0, w2 = 0, b = 0;
      const S = [];
      const frame = (cur, msg, ep, mist) => {
        const dots = pts.map(([x, y], i) => [x, y, ys[i] ? "b" : "a"]);
        const layers = [
          Plot.regions((x, y) => (w1 * x + w2 * y + b > 0 ? "b" : "a"), 40),
          Math.abs(w2) > 1e-9 ? Plot.fn((x) => -(w1 * x + b) / w2, "pl-c1") : Math.abs(w1) > 1e-9 ? Plot.vline(-b / w1, "pl-c1") : () => "",
          Plot.pts(dots, "pl-c1", 8),
          ...(cur >= 0 ? [(S2) => `<circle cx="${S2.sx(pts[cur][0])}" cy="${S2.sy(pts[cur][1])}" r="14" style="fill:none;stroke:var(--text);stroke-width:2.2"/>`] : []),
          ...pts.map(([x, y], i) => Plot.text(x + 0.17, y + 0.13, `(${x},${y}) → ${ys[i]}`, "", "start")),
        ];
        return {
          svg: Plot.chart({ W: 440, H: 380, x: [-0.5, 1.5], y: [-0.5, 1.5], xl: "x₁", yl: "x₂", ticks: 4, yticks: 4 }, layers),
          side: { epoch: ep, "w₁": fx(w1, 0), "w₂": fx(w2, 0), bias: fx(b, 0), "mistakes this pass": mist },
          msg,
        };
      };
      S.push(frame(-1, `Start with all weights 0. The neuron outputs 1 when $w_1x_1 + w_2x_2 + b > 0$, else 0, so the whole plane is predicted **0** (blue). Target: ${xor ? "**XOR** (1 when exactly one input is 1)" : "**AND** (1 only when both inputs are 1)"}. Blue dots are target 0, orange dots target 1. Press Next to show the neuron one example at a time.`, 0, "-"));
      const maxEp = xor ? 10 : 30;
      let done = false, lastEp = 0;
      for (let ep = 1; ep <= maxEp && !done; ep++) {
        let mist = 0;
        for (let i = 0; i < 4; i++) {
          const [x1, x2] = pts[i], z = w1 * x1 + w2 * x2 + b, p = z > 0 ? 1 : 0, e = ys[i] - p;
          let msg = `Epoch ${ep}, example (${x1}, ${x2}), target ${ys[i]}. Score $z = ${pr(w1)}\\cdot${x1} + ${pr(w2)}\\cdot${x2} + ${pr(b)} = ${fx(z, 0)}$, so it predicts **${p}**. `;
          if (e === 0) msg += "Correct: nothing changes.";
          else { mist++; w1 += e * x1; w2 += e * x2; b += e; msg += `Wrong (error ${e > 0 ? "+" : ""}${e}): add ${e > 0 ? "" : "minus "}the input to the weights: $w \\leftarrow w + (y - \\hat y)\\,x$, $b \\leftarrow b + (y - \\hat y)$. The line moves to $w = (${fx(w1, 0)}, ${fx(w2, 0)})$, $b = ${fx(b, 0)}$.`; }
          if (i === 3 && mist === 0) { done = true; msg += " **A whole pass with no mistakes: the perceptron has converged.** The line separates the two classes."; }
          S.push(frame(i, msg, ep, mist));
        }
        lastEp = ep;
      }
      if (!done) S.push(frame(-1, `After ${lastEp} passes the perceptron is **still making mistakes**, and it always will: no single straight line can put (0,1) and (1,0) on one side and (0,0) and (1,1) on the other. The weights just cycle. XOR is not linearly separable, which is exactly why we need hidden layers.`, lastEp, "> 0"));
      return S;
    },
  };

  /* ================================================================== 2. A 2-2-1 network: forward and backward pass with numbers */
  function netCalc(x1, x2, t, lr) {
    const W1 = [[0.5, -0.4], [0.3, 0.8]], b1 = [0.1, -0.1], W2 = [0.7, -0.6], b2 = 0.2, x = [x1, x2];
    const run = (W1, b1, W2, b2) => {
      const z1 = W1.map((r, j) => r[0] * x[0] + r[1] * x[1] + b1[j]), h = z1.map(sig), z2 = W2[0] * h[0] + W2[1] * h[1] + b2, y = sig(z2);
      return { z1, h, z2, y, L: 0.5 * (y - t) ** 2 };
    };
    const f = run(W1, b1, W2, b2);
    const dy = f.y - t, dz2 = dy * f.y * (1 - f.y), dW2 = f.h.map((v) => dz2 * v), db2 = dz2;
    const dh = W2.map((w) => dz2 * w), dz1 = dh.map((g, j) => g * f.h[j] * (1 - f.h[j])), dW1 = dz1.map((g) => x.map((xi) => g * xi)), db1 = dz1;
    const nW1 = W1.map((r, j) => r.map((w, i) => w - lr * dW1[j][i])), nb1 = b1.map((v, j) => v - lr * db1[j]);
    const nW2 = W2.map((w, j) => w - lr * dW2[j]), nb2 = b2 - lr * db2;
    return { W1, b1, W2, b2, x, t, ...f, dy, dz2, dW2, db2, dh, dz1, dW1, db1, nW1, nb1, nW2, nb2, after: run(nW1, nb1, nW2, nb2) };
  }
  V.dl1_netpass = {
    title: "A 2-2-1 network: forward pass, then backward pass, with every number",
    params: [
      { id: "x1", label: "input x₁", min: -2, max: 2, step: 0.1, value: 1 },
      { id: "x2", label: "input x₂", min: -2, max: 2, step: 0.1, value: 0.5 },
      { id: "t", label: "target t", min: 0, max: 1, step: 1, value: 1 },
      { id: "lr", label: "learning rate", min: 0.1, max: 2, step: 0.1, value: 0.5 },
    ],
    gen(_, P) {
      const N = netCalc(P.x1, P.x2, P.t, P.lr), S = [];
      const IN = [[70, 130], [70, 270]], HD = [[285, 95], [285, 305]], OUT = [[500, 200]];
      const d4 = (v) => fx(v, 4), d3 = (v) => fx(v, 3);
      const draw = (k) => {
        const red = "fill:var(--red);font-size:11.5px;" + HALO, mut = "fill:var(--muted);font-size:11.5px;" + HALO;
        let s = "";
        const edge = (a, b, w, g, nw, frac) => {
          s += `<line class="edge" x1="${a[0] + 26}" y1="${a[1]}" x2="${b[0] - 26}" y2="${b[1]}"/>`;
          const lx = a[0] + 26 + (b[0] - a[0] - 52) * frac, yy = a[1] + (b[1] - a[1]) * frac;
          s += `<text x="${lx}" y="${yy - 5}" text-anchor="middle" style="${mut}">${k >= 12 ? "w=" + d3(nw) : "w=" + d3(w)}</text>`;
          if (g !== null) s += `<text x="${lx}" y="${yy + 9}" text-anchor="middle" style="${red}">∂L/∂w=${d3(g)}</text>`;
        };
        // edges: input -> hidden (label near the input end), hidden -> output (label near the middle)
        for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) edge(IN[i], HD[j], N.W1[j][i], k >= 11 ? N.dW1[j][i] : null, N.nW1[j][i], 0.25);
        for (let j = 0; j < 2; j++) edge(HD[j], OUT[0], N.W2[j], k >= 8 ? N.dW2[j] : null, N.nW2[j], 0.5);
        const node = (p, inside, above, below, cls) => {
          s += `<circle class="${cls}" cx="${p[0]}" cy="${p[1]}" r="26" style="stroke-width:2.2"/>`;
          s += `<text x="${p[0]}" y="${p[1] + 5}" text-anchor="middle" class="val" style="font-size:14px">${inside}</text>`;
          if (above) s += `<text x="${p[0]}" y="${p[1] - 33}" text-anchor="middle" style="${mut}">${above}</text>`;
          below.forEach((t, i) => (s += `<text x="${p[0]}" y="${p[1] + 44 + i * 14}" text-anchor="middle" style="${red}">${t}</text>`));
        };
        IN.forEach((p, i) => node(p, fx(N.x[i], 2), `x${i + 1}`, [], "node"));
        HD.forEach((p, j) => {
          const grads = []; if (k >= 9) grads.push("∂L/∂h=" + d4(N.dh[j])); if (k >= 10) grads.push("∂L/∂z=" + d4(N.dz1[j]));
          node(p, k >= 2 ? d3(N.h[j]) : "h" + (j + 1), k >= 1 ? "z=" + d3(N.z1[j]) : "hidden " + (j + 1), j === 0 ? [] : grads, k >= 2 ? "node ok" : "node");
          if (j === 0) grads.forEach((t, i) => (s += `<text x="${p[0]}" y="${p[1] - 47 - i * 14}" text-anchor="middle" style="${red}">${t}</text>`));
        });
        const ob = []; if (k >= 6) ob.push("∂L/∂y=" + d4(N.dy)); if (k >= 7) ob.push("∂L/∂z=" + d4(N.dz2));
        node(OUT[0], k >= 4 ? d3(N.y) : "y", k >= 3 ? "z=" + d3(N.z2) : "output", ob, k >= 4 ? "node ok" : "node");
        if (k >= 5) s += `<text x="${OUT[0][0] + 50}" y="${OUT[0][1] + 82}" text-anchor="middle" style="fill:var(--text);font-size:12.5px;font-weight:700">loss L = ½(y−t)² = ${d4(N.L)}</text><text x="${OUT[0][0] + 50}" y="${OUT[0][1] + 98}" text-anchor="middle" style="fill:var(--muted);font-size:11.5px">target t = ${N.t}</text>`;
        if (k >= 12) s += `<text x="${OUT[0][0] + 50}" y="${OUT[0][1] + 114}" text-anchor="middle" style="fill:var(--green);font-size:12.5px;font-weight:700">after update: L = ${d4(N.after.L)}</text>`;
        s += `<text x="10" y="16" style="fill:var(--accent);font-size:12px;font-weight:700">${k <= 5 ? "forward pass →" : k <= 11 ? "← backward pass (red)" : "update"}</text>`;
        return `<svg viewBox="0 0 640 380" width="640" height="380" xmlns="http://www.w3.org/2000/svg" role="img">${s}</svg>`;
      };
      const side = (k) => {
        const o = {};
        if (k >= 1) o["z₁, z₂"] = `${d3(N.z1[0])}, ${d3(N.z1[1])}`;
        if (k >= 2) o["h₁, h₂"] = `${d3(N.h[0])}, ${d3(N.h[1])}`;
        if (k >= 4) o.y = d3(N.y);
        if (k >= 5) o.loss = d4(N.L);
        if (k >= 8) o["∂L/∂b₂"] = d4(N.db2);
        if (k >= 11) o["∂L/∂b₁"] = `${d4(N.db1[0])}, ${d4(N.db1[1])}`;
        return o;
      };
      const msgs = [
        `Tiny network: 2 inputs, 2 hidden neurons, 1 output, all with sigmoid. The weights are fixed (shown on the edges); the biases are $b_1 = (0.1, -0.1)$ and $b_2 = 0.2$. We will push one example through (forward), measure the loss, then send gradients back (backward). Press Next.`,
        `**Hidden pre-activations.** Each hidden neuron does $z_j = w_{j1}x_1 + w_{j2}x_2 + b_j$. For neuron 1: ${fx(N.W1[0][0], 1)}·${fx(N.x[0], 2)} + ${pr(N.W1[0][1], 1)}·${fx(N.x[1], 2)} + ${pr(N.b1[0], 1)} = **${d3(N.z1[0])}**. Neuron 2: **${d3(N.z1[1])}**.`,
        `**Hidden activations.** Squash each $z$ with the sigmoid $h = 1/(1+e^{-z})$: $h_1 = ${d3(N.h[0])}$, $h_2 = ${d3(N.h[1])}$.`,
        `**Output pre-activation.** $z_o = ${fx(N.W2[0], 1)}\\cdot${d3(N.h[0])} + ${pr(N.W2[1], 1)}\\cdot${d3(N.h[1])} + ${pr(N.b2, 1)} = ${d3(N.z2)}$.`,
        `**Prediction.** $y = \\sigma(z_o) = ${d3(N.y)}$. The network says "${d3(N.y)}" and the target is ${N.t}.`,
        `**Loss.** $L = \\tfrac12 (y - t)^2 = \\tfrac12(${d3(N.y)} - ${N.t})^2 = ${d4(N.L)}$. The forward pass is done; now we ask: how does L change if we nudge each weight?`,
        `**Backward, step 1.** Start at the end: $\\partial L/\\partial y = y - t = ${d3(N.y)} - ${N.t} = ${d4(N.dy)}$.`,
        `**Through the output sigmoid.** $\\partial L/\\partial z_o = \\partial L/\\partial y \\cdot y(1-y) = ${d4(N.dy)} \\cdot ${d4(N.y * (1 - N.y))} = ${d4(N.dz2)}$. This number is the "error signal" at the output neuron.`,
        `**Output weights.** $z_o$ depends on $w_{o1}$ through $h_1$, so $\\partial L/\\partial w_{o1} = \\partial L/\\partial z_o \\cdot h_1 = ${d4(N.dz2)}\\cdot${d3(N.h[0])} = ${d4(N.dW2[0])}$, and likewise $\\partial L/\\partial w_{o2} = ${d4(N.dW2[1])}$. The bias gradient is just $\\partial L/\\partial z_o = ${d4(N.db2)}$.`,
        `**Into the hidden layer.** $h_j$ reaches the loss through $w_{oj}$, so $\\partial L/\\partial h_j = \\partial L/\\partial z_o \\cdot w_{oj}$: $${d4(N.dz2)}\\cdot ${fx(N.W2[0], 1)} = ${d4(N.dh[0])}$ and $${d4(N.dz2)}\\cdot ${fx(N.W2[1], 1)} = ${d4(N.dh[1])}$.`,
        `**Through the hidden sigmoids.** $\\partial L/\\partial z_j = \\partial L/\\partial h_j \\cdot h_j(1-h_j)$. Neuron 1: ${d4(N.dh[0])} × ${d4(N.h[0] * (1 - N.h[0]))} = ${d4(N.dz1[0])}. Neuron 2: ${d4(N.dh[1])} × ${d4(N.h[1] * (1 - N.h[1]))} = ${d4(N.dz1[1])}.`,
        `**First-layer weights.** $\\partial L/\\partial w_{ji} = \\partial L/\\partial z_j \\cdot x_i$: each hidden neuron's error signal is multiplied by the input that fed the weight. All six weight gradients and three bias gradients are now known. That was one full **backpropagation**.`,
        `**Update.** $w \\leftarrow w - \\eta\\,\\partial L/\\partial w$ with $\\eta = ${P.lr}$. The loss goes from ${d4(N.L)} to **${d4(N.after.L)}**. ${N.after.L < N.L ? "It went down, as it should." : "It did not go down: the step was too large for this example, try a smaller learning rate."} Change the inputs, target or learning rate with the sliders.`,
      ];
      for (let k = 0; k < msgs.length; k++) S.push({ svg: draw(k), side: side(k), msg: msgs[k] });
      return S;
    },
  };

  /* ================================================================== 3. Activation functions with derivatives */
  const ACT = [
    { name: "sigmoid", f: sig, d: (x) => sig(x) * (1 - sig(x)), sat: true, note: "squashes to (0, 1); the slope is at most 0.25, and ~0 for large |x| (saturation)" },
    { name: "tanh", f: Math.tanh, d: (x) => 1 - Math.tanh(x) ** 2, sat: true, note: "squashes to (−1, 1), zero-centred; slope at most 1, ~0 for large |x|" },
    { name: "ReLU", f: (x) => Math.max(0, x), d: (x) => (x > 0 ? 1 : 0), note: "max(0, x); slope exactly 1 for x > 0 and exactly 0 for x < 0 (a \"dead\" neuron gets no gradient)" },
    { name: "Leaky ReLU", f: (x) => (x > 0 ? x : 0.1 * x), d: (x) => (x > 0 ? 1 : 0.1), note: "like ReLU but leaks a slope of 0.1 on the negative side, so it never goes fully dead" },
    { name: "GELU", f: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))), d: (x) => { const h = 1e-4, g = (v) => 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3))); return (g(x + h) - g(x - h)) / (2 * h); }, note: "a smooth ReLU-like curve (tanh approximation); the default in many transformers" },
    { name: "SiLU / Swish", f: (x) => x * sig(x), d: (x) => sig(x) * (1 + x * (1 - sig(x))), note: "x·sigmoid(x): smooth, slightly negative dip, used in many modern networks" },
  ];
  V.dl1_act = {
    title: "Activation functions and their derivatives",
    params: [
      { id: "fn", label: "function", min: 0, max: 5, step: 1, value: 0, fmt: (v) => ACT[+v].name },
      { id: "x", label: "input x", min: -6, max: 6, step: 0.1, value: 1.5 },
    ],
    gen(_, P) {
      const A = ACT[+P.fn], x = P.x, y = A.f(x), g = A.d(x);
      const sat = A.sat ? Math.abs(g) < 0.05 : false, dead = !A.sat && Math.abs(g) < 1e-9;
      const lo = Math.min(-1.2, ...[-6, 6].map(A.f)), hi = Math.max(1.2, A.f(6));
      const satZones = A.sat ? [(S) => { let s = ""; const a = [], b = []; for (let u = -6; u <= 6.001; u += 0.05) if (Math.abs(A.d(u)) < 0.05) (u < 0 ? a : b).push(u); [a, b].forEach((z) => { if (z.length) s += `<rect x="${S.sx(z[0])}" y="${S.P.t}" width="${S.sx(z[z.length - 1]) - S.sx(z[0])}" height="${S.H - S.P.t - S.P.b}" style="fill:${soft("red", 12)}"/>`; }); return s; }] : [];
      const svg = Plot.chart({ W: 560, H: 300, x: [-6, 6], y: [Math.floor(lo - 0.2), Math.ceil(hi + 0.2)], xl: "x", yl: "output", ticks: 6 }, [
        ...satZones, Plot.hline(0, "pl-dash"), Plot.vline(0, "pl-dash"),
        Plot.fn(A.f, "pl-c1"), Plot.fn(A.d, "pl-c4"),
        Plot.vline(x, "pl-c2 pl-thin"), Plot.dot(x, y, "pl-c1", 6), Plot.dot(x, g, "pl-c4", 6),
        (S) => `<circle cx="${S.P.l + 18}" cy="${S.P.t + 14}" r="4" style="fill:var(--accent)"/><text x="${S.P.l + 28}" y="${S.P.t + 18}" class="pl-note">f(x)</text><circle cx="${S.P.l + 78}" cy="${S.P.t + 14}" r="4" style="fill:var(--red)"/><text x="${S.P.l + 88}" y="${S.P.t + 18}" class="pl-note">f′(x), the slope</text>`,
      ]);
      const state = sat ? "**Saturated**: the curve is flat here, so the slope is almost 0 and almost no gradient flows back through this neuron (shaded red zone)." : dead ? "**Dead zone**: the slope is exactly 0 here, so this neuron gets no gradient for this input." : Math.abs(g) >= 0.9 ? "Slope near 1: gradients pass through almost unchanged." : "Healthy slope: gradients flow, but are scaled down a bit.";
      return [{ svg, side: { x: fx(x, 1), "f(x)": r3(y), "f′(x)": r3(g) }, msg: `**${A.name}**: ${A.note}. ${state} Drag the slider and watch the red dot (the slope) as the other dot (the output) moves.` }];
    },
  };

  /* ================================================================== 4. Optimizer race on a ravine */
  V.dl1_optim = {
    title: "Optimizer race on a narrow ravine",
    params: [
      { id: "lr", label: "learning rate", min: 0.01, max: 0.12, step: 0.01, value: 0.05 },
      { id: "beta", label: "momentum β", min: 0, max: 0.95, step: 0.05, value: 0.9 },
    ],
    gen(_, P) {
      // L(x, y) = ½(0.05 x² + 20 y²): very gentle along x, very steep along y
      const A = 0.05, B = 20, L = (x, y) => 0.5 * (A * x * x + B * y * y), G = (x, y) => [A * x, B * y];
      const N = 100, lr = P.lr, start = [-8, 1.5], paths = { sgd: [start.slice()], mom: [start.slice()], rms: [start.slice()], adam: [start.slice()] };
      let p = start.slice(), v = [0, 0], pm = start.slice(), s = [0, 0], pr = start.slice(), m = [0, 0], s2 = [0, 0], pa = start.slice();
      for (let t = 1; t <= N; t++) {
        let g = G(...p); p = [p[0] - lr * g[0], p[1] - lr * g[1]]; paths.sgd.push(p);
        g = G(...pm); v = [P.beta * v[0] + g[0], P.beta * v[1] + g[1]]; pm = [pm[0] - lr * v[0], pm[1] - lr * v[1]]; paths.mom.push(pm);
        g = G(...pr); s = s.map((a, i) => 0.9 * a + 0.1 * g[i] * g[i]); pr = pr.map((a, i) => a - (lr * g[i]) / (Math.sqrt(s[i]) + 1e-8)); paths.rms.push(pr);
        g = G(...pa); m = m.map((a, i) => 0.9 * a + 0.1 * g[i]); s2 = s2.map((a, i) => 0.999 * a + 0.001 * g[i] * g[i]);
        pa = pa.map((a, i) => a - (lr * (m[i] / (1 - 0.9 ** t))) / (Math.sqrt(s2[i] / (1 - 0.999 ** t)) + 1e-8)); paths.adam.push(pa);
      }
      const contours = [0.05, 0.4, 1.5, 4, 10, 20].map((l) => { const pts = []; for (let i = 0; i <= 120; i++) { const t = (i / 120) * 2 * Math.PI; pts.push([Math.sqrt((2 * l) / A) * Math.cos(t), Math.sqrt((2 * l) / B) * Math.sin(t)]); } return Plot.line(pts, "pl-dash"); });
      const clip = (a) => a.map(([x, y]) => [Math.max(-11, Math.min(11, x)), Math.max(-2.4, Math.min(2.4, y))]);
      const cols = { sgd: "pl-c2", mom: "pl-c5", rms: "pl-c3", adam: "pl-c1" }, names = { sgd: "SGD", mom: "Momentum", rms: "RMSProp", adam: "Adam" };
      const S = [];
      for (let k = 0; k <= N; k += k < 20 ? 1 : 2) {
        const layers = [...contours, Plot.dot(0, 0, "pl-c3", 3.5)];
        for (const key of ["sgd", "mom", "rms", "adam"]) { const q = clip(paths[key].slice(0, k + 1)); layers.push(Plot.line(q, cols[key] + " pl-thin"), Plot.dot(q[q.length - 1][0], q[q.length - 1][1], cols[key], 5.5)); }
        ["sgd", "mom", "rms", "adam"].forEach((key, i) => layers.push((S2) => `<circle cx="${S2.sx(-10.6) + i * 100}" cy="${S2.sy(2.0)}" r="4.5" style="fill:var(--${{ sgd: "amber", mom: "purple", rms: "green", adam: "accent" }[key]})"/><text x="${S2.sx(-10.6) + i * 100 + 9}" y="${S2.sy(2.0) + 4}" class="pl-note">${names[key]}</text>`));
        const lo = {}; for (const key in paths) lo[key] = L(...paths[key][k]);
        const best = Object.keys(lo).sort((a, b) => lo[a] - lo[b])[0], diverged = lo.sgd > 1e3;
        let msg;
        if (k === 0) msg = "All four optimizers start at the same point (−8, 1.5) with the **same learning rate**. The loss is a ravine: very steep across (y) and almost flat along (x). The minimum is the green dot. Press Play.";
        else if (diverged) msg = "SGD **diverges**: along the steep direction its step is $\\eta \\cdot 20$, and that is above 2, so each step overshoots further than the last. Lower the learning rate. The others tame the steep direction by normalising or averaging gradients.";
        else if (k < 15) msg = "SGD feels the steep wall and makes only tiny progress along the valley. Momentum accumulates the consistent downhill push. RMSProp and Adam divide each coordinate by its recent gradient size, so both directions move at a similar pace.";
        else msg = `After ${k} steps the loss is: SGD ${r3(lo.sgd)}, Momentum ${r3(lo.mom)}, RMSProp ${r3(lo.rms)}, Adam ${r3(lo.adam)}. **${names[best]}** leads. Try lr 0.1 (plain SGD starts to fail), or momentum β = 0 (then momentum equals SGD). No optimizer wins everywhere: the ranking changes with the learning rate.`;
        S.push({ svg: Plot.chart({ W: 600, H: 290, x: [-11, 11], y: [-2.4, 2.4], xl: "w₁ (flat direction)", yl: "w₂ (steep)", ticks: 8, yticks: 4 }, layers), side: { step: k, SGD: r3(lo.sgd), Momentum: r3(lo.mom), RMSProp: r3(lo.rms), Adam: r3(lo.adam) }, msg });
      }
      return S;
    },
  };

  /* ================================================================== 5. Learning-rate schedules */
  const SCHED = ["constant", "step decay", "cosine", "warmup + cosine"];
  function lrAt(kind, t, T, lr0, warm, minf) {
    const lrmin = lr0 * minf, cos = (a, b) => lrmin + 0.5 * (lr0 - lrmin) * (1 + Math.cos(Math.PI * Math.min(1, Math.max(0, (t - a) / (b - a)))));
    if (kind === 0) return lr0;
    if (kind === 1) return lr0 * Math.pow(0.5, Math.floor(t / 250));
    if (kind === 2) return cos(0, T);
    return t < warm ? lr0 * (t + 1) / warm : cos(warm, T);
  }
  V.dl1_lrsched = {
    title: "Learning-rate schedules",
    params: [
      { id: "kind", label: "schedule", min: 0, max: 3, step: 1, value: 3, fmt: (v) => SCHED[+v] },
      { id: "lr", label: "peak learning rate", min: 0.001, max: 0.1, step: 0.001, value: 0.03 },
      { id: "warm", label: "warmup steps", min: 10, max: 300, step: 10, value: 100 },
      { id: "minf", label: "final lr as fraction of peak", min: 0, max: 0.5, step: 0.05, value: 0.05 },
    ],
    gen(_, P) {
      const T = 1000, kind = +P.kind, pts = (k) => { const a = []; for (let t = 0; t <= T; t += 5) a.push([t, lrAt(k, t, T, P.lr, P.warm, P.minf)]); return a; };
      const cls = ["pl-c2", "pl-c3", "pl-c5", "pl-c1"];
      const layers = [0, 1, 2, 3].filter((k) => k !== kind).map((k) => Plot.line(pts(k), "pl-dash")).concat([Plot.line(pts(kind), cls[kind])]);
      const mark = [250, 500, 1000].map((t) => ({ t, v: lrAt(kind, t, T, P.lr, P.warm, P.minf) }));
      const descr = [
        "**Constant**: the same step size all the way. Simple, but the loss is noisy at the end because the steps never shrink.",
        "**Step decay**: halve the learning rate every 250 steps. Easy to reason about; the loss usually drops right after each cut.",
        "**Cosine**: glide smoothly from the peak to a small final value along half a cosine wave. No schedule knobs to tune except the endpoints.",
        `**Warmup + cosine**: ramp up linearly for ${P.warm} steps (early gradients and Adam's statistics are unreliable), then cosine decay. The default recipe for transformers.`,
      ][kind];
      return [{
        svg: Plot.chart({ W: 560, H: 290, x: [0, T], y: [0, 0.105], xl: "training step", yl: "learning rate", ticks: 5, yticks: 5 }, layers),
        side: { "lr @ 250": r3(mark[0].v), "lr @ 500": r3(mark[1].v), "lr @ 1000": r3(mark[2].v) },
        msg: descr + " The dashed grey lines are the other three schedules for comparison.",
      }];
    },
  };

  /* ================================================================== 6. Dropout masks */
  V.dl1_dropout = {
    title: "Dropout: random masks on one layer",
    params: [
      { id: "p", label: "drop probability p", min: 0, max: 0.9, step: 0.1, value: 0.5 },
      { id: "seed", label: "random mask #", min: 1, max: 20, step: 1, value: 1 },
    ],
    gen(_, P) {
      const base = [0.8, -0.3, 1.2, 0.5, -0.9, 0.1, 0.7, -0.6], p = P.p, r = rng(P.seed * 7919);
      const mask = base.map(() => (r() >= p ? 1 : 0)), scale = 1 / (1 - p);
      const out = base.map((v, i) => mask[i] * v * scale);
      // average of the scaled outputs over many random masks (should be close to the original)
      const rr = rng(12345), M = 4000, acc = base.map(() => 0);
      for (let m = 0; m < M; m++) base.forEach((v, i) => { if (rr() >= p) acc[i] += v * scale; });
      const avg = acc.map((a) => a / M), kept = mask.reduce((a, b) => a + b, 0);
      const cls = {}, cls2 = {}; mask.forEach((m, i) => { cls[i] = m ? "done" : "bad"; cls2[i] = m ? "hl" : "dim"; });
      return [{
        rows: [
          { label: "activations of one layer", vals: base.map((v) => fx(v, 2)), idx: false },
          { label: `mask (1 = keep, 0 = drop), drop probability ${p}`, vals: mask.map(String), cls, idx: false },
          { label: `after dropout, kept values scaled by 1/(1−p) = ${fx(scale, 2)}`, vals: out.map((v) => fx(v, 2)), cls: cls2, idx: false },
          { label: "average over 4000 random masks (≈ the original activations)", vals: avg.map((v) => fx(v, 2)), idx: false },
        ],
        side: { kept: `${kept} of ${base.length}`, "sum before": fx(base.reduce((a, b) => a + b, 0), 2), "sum after": fx(out.reduce((a, b) => a + b, 0), 2) },
        msg: p === 0 ? "With p = 0 nothing is dropped and nothing is scaled." : `Every unit is switched off with probability **${p}**; the survivors are multiplied by $1/(1-p)$ so the **expected** value of each unit stays the same (last row). Pick another mask number to see a different random subnetwork. Each training step uses a fresh mask, so the network cannot rely on any single unit. At test time dropout is simply turned off.`,
      }];
    },
  };

  /* ================================================================== 7. Vanishing / exploding gradients through depth */
  V.dl1_vanish = {
    title: "Vanishing and exploding gradients through depth",
    params: [
      { id: "depth", label: "depth (layers)", min: 2, max: 30, step: 1, value: 20 },
      { id: "gain", label: "weight scale", min: 0.5, max: 3, step: 0.1, value: 1.4 },
    ],
    gen(_, P) {
      const D = P.depth, W = 24, r = rng(42);
      const Ws = Array.from({ length: D }, () => Array.from({ length: W }, () => Array.from({ length: W }, () => (gauss(r) * P.gain) / Math.sqrt(W))));
      const x0 = Array.from({ length: W }, () => gauss(r));
      const acts = { sigmoid: [sig, (z, a) => a * (1 - a)], tanh: [Math.tanh, (z, a) => 1 - a * a], ReLU: [(z) => Math.max(0, z), (z) => (z > 0 ? 1 : 0)] };
      const curves = {};
      for (const [name, [f, df]] of Object.entries(acts)) {
        const zs = [], as = [x0];
        for (let l = 0; l < D; l++) { const z = Ws[l].map((row) => row.reduce((s, w, j) => s + w * as[l][j], 0)); zs.push(z); as.push(z.map(f)); }
        let g = Array.from({ length: W }, () => 1 / Math.sqrt(W));
        const norms = new Array(D);
        for (let l = D - 1; l >= 0; l--) {
          norms[l] = Math.sqrt(g.reduce((s, v) => s + v * v, 0));
          const gz = g.map((v, j) => v * df(zs[l][j], as[l + 1][j]));
          g = Array.from({ length: W }, (_, j) => Ws[l].reduce((s, row, i) => s + row[j] * gz[i], 0));
        }
        curves[name] = norms.map((n, l) => [l + 1, Math.log10(Math.max(n, 1e-30))]);
      }
      const ref = { sigmoid: 0, tanh: 0, ReLU: 0 };
      const cls = { sigmoid: "pl-c4", tanh: "pl-c5", ReLU: "pl-c3" }, col = { sigmoid: "red", tanh: "purple", ReLU: "green" };
      const lo = -14, hi = 6, clipc = (c) => c.map(([x, y]) => [x, Math.max(lo, Math.min(hi, y))]);
      const layers = [Plot.hline(0, "pl-dash")];
      Object.keys(curves).forEach((k, i) => layers.push(Plot.line(clipc(curves[k]), cls[k]), Plot.pts(clipc(curves[k]), cls[k], 2.8),
        (S) => `<circle cx="${S.P.l + 18 + i * 92}" cy="${S.P.t + 14}" r="4.5" style="fill:var(--${col[k]})"/><text x="${S.P.l + 28 + i * 92}" y="${S.P.t + 18}" class="pl-note">${k}</text>`));
      const first = (k) => Math.pow(10, curves[k][0][1]);
      const sg = first("sigmoid"), rl = first("ReLU"), th = first("tanh");
      const verdict = (v) => (v < 1e-3 ? "vanished" : v > 1e3 ? "exploded" : "healthy");
      return [{
        svg: Plot.chart({ W: 580, H: 300, x: [1, Math.max(D, 2)], y: [lo, hi], xl: "layer (1 = closest to the input, the last layer is the loss end)", yl: "log10 gradient size", ticks: Math.min(D, 10), yticks: 5 }, layers),
        side: { "sigmoid @ layer 1": r3(sg) + " (" + verdict(sg) + ")", "tanh @ layer 1": r3(th) + " (" + verdict(th) + ")", "ReLU @ layer 1": r3(rl) + " (" + verdict(rl) + ")" },
        msg: `The gradient reaching each layer, measured against the gradient at the last layer (the dashed line at 0 = same size). Each layer multiplies it by its weights and by the activation's slope. A sigmoid slope is at most 0.25, so a deep sigmoid stack shrinks it fast. ReLU has slope 1, so with the right weight scale (about 1.4, "He" scaling) it survives. Push the weight scale up toward 3 and everything **explodes**; push depth up and the gap widens.`,
      }];
    },
  };

  /* ================================================================== static diagrams */
  const mk = (id) => `<defs><marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker><marker id="${id}r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--red)"/></marker></defs>`;
  const wrap = (W, H, inner, id) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">${mk(id)}${inner}</svg>`;
  const bx = (x, y, w, h, label, cls = "node") => `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="8"/><text class="val" x="${x + w / 2}" y="${y + h / 2 + 5}" text-anchor="middle" style="font-size:13px">${esc(label)}</text>`;
  const tx = (x, y, s, style = "", anchor = "middle") => `<text class="lbl" x="${x}" y="${y}" text-anchor="${anchor}" style="${style}">${esc(s)}</text>`;
  const arr = (x1, y1, x2, y2, id, red) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" style="stroke:var(--${red ? "red" : "muted"});stroke-width:1.8" marker-end="url(#${id}${red ? "r" : ""})"/>`;

  // one artificial neuron: inputs, weights, sum + bias, activation, output
  D.dl1_neuron = () => {
    const id = "dn1";
    let s = "";
    [0, 1, 2].forEach((i) => {
      const y = 50 + i * 70;
      s += `<circle class="node" cx="60" cy="${y}" r="22"/><text class="val" x="60" y="${y + 5}" text-anchor="middle">x${i + 1}</text>`;
      s += arr(84, y, 232, 120 + (y - 120) * 0.35, id) + tx(150, y + (120 - y) * 0.2 - 8, "× w" + (i + 1), "font-size:12px;fill:var(--accent)");
    });
    s += `<circle class="node hl" cx="262" cy="120" r="34"/><text class="val" x="262" y="116" text-anchor="middle" style="font-size:20px">Σ</text><text class="lbl" x="262" y="136" text-anchor="middle">+ b</text>`;
    s += tx(262, 172, "z = w₁x₁ + w₂x₂ + w₃x₃ + b", "font-size:12px;fill:var(--text)");
    s += arr(298, 120, 372, 120, id);
    s += `<rect class="node ok" x="374" y="92" width="70" height="56" rx="10"/><text class="val" x="409" y="116" text-anchor="middle" style="font-size:16px">f( )</text><text class="lbl" x="409" y="136" text-anchor="middle">activation</text>`;
    s += arr(444, 120, 510, 120, id) + `<circle class="node" cx="534" cy="120" r="24"/><text class="val" x="534" y="125" text-anchor="middle">a</text>`;
    s += tx(534, 164, "output a = f(z)", "font-size:12px;fill:var(--text)");
    s += tx(60, 214, "inputs", "font-size:12px") + tx(150, 214, "weights (learned)", "font-size:12px") + tx(262, 214, "weighted sum + bias (learned)", "font-size:12px");
    return wrap(600, 232, s, id);
  };

  // a fully-connected network of given layer sizes, e.g. [3,4,4,2]
  D.dl1_mlp = (sizes) => {
    sizes = sizes || [3, 4, 4, 2];
    const id = "dm1", W = 600, H = 260, n = sizes.length, gap = (W - 120) / (n - 1), maxN = Math.max(...sizes);
    const pos = sizes.map((c, l) => Array.from({ length: c }, (_, i) => [60 + l * gap, 130 + (i - (c - 1) / 2) * Math.min(52, 190 / maxN)]));
    let s = "";
    for (let l = 0; l < n - 1; l++) pos[l].forEach((a) => pos[l + 1].forEach((b) => (s += `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" style="stroke:var(--border);stroke-width:1.3"/>`)));
    pos.forEach((layer, l) => layer.forEach(([x, y]) => (s += `<circle class="node${l === 0 ? "" : l === n - 1 ? " ok" : " hl"}" cx="${x}" cy="${y}" r="14"/>`)));
    pos.forEach((layer, l) => {
      const name = l === 0 ? "input layer" : l === n - 1 ? "output layer" : "hidden layer" + (n > 3 ? " " + l : "");
      s += tx(layer[0][0], 24, name, "font-size:12px;font-weight:700;fill:var(--text)") + tx(layer[0][0], 242, sizes[l] + (sizes[l] === 1 ? " unit" : " units"), "font-size:11.5px");
    });
    return wrap(W, H, s, id);
  };

  // the computational graph of  L = (sigmoid(w*x + b) - t)^2  with values and gradients
  D.dl1_graph = () => {
    const id = "dg1", x = 2, w = 0.5, b = -0.5, t = 1, z = w * x + b, a = sig(z), L = (a - t) ** 2;
    const dL_da = 2 * (a - t), da_dz = a * (1 - a), dz = dL_da * da_dz, dw = dz * x, db = dz, dx = dz * w;
    const c = (v) => fx(v, 3);
    const X = [50, 150, 270, 390, 510];
    let s = tx(300, 18, "forward pass: values flow left to right", "font-size:12px;font-weight:700;fill:var(--accent)");
    const forward = [["x=" + x, "input"], ["w·x", c(w * x)], ["z = w·x + b", c(z)], ["a = σ(z)", c(a)], ["L = (a−t)²", c(L)]];
    forward.forEach(([name, val], i) => {
      s += bx(X[i] - 46, 40, 92, 46, name, i === 4 ? "node ok" : "node");
      s += tx(X[i], 104, val, "font-size:12px;fill:var(--accent);font-weight:700");
      if (i < 4) s += arr(X[i] + 48, 63, X[i + 1] - 48, 63, id);
    });
    s += tx(150, 128, "w = " + w + ", b = " + b + ", t = " + t, "font-size:11.5px");
    s += tx(300, 158, "backward pass: gradients flow right to left, multiplying local slopes", "font-size:12px;font-weight:700;fill:var(--red)");
    const back = [["∂L/∂x", c(dx)], ["∂L/∂(wx)", c(dz)], ["∂L/∂z", c(dz)], ["∂L/∂a", c(dL_da)], ["∂L/∂L", "1"]];
    const loc = ["× w = " + w, "× 1", "× a(1−a) = " + c(da_dz), "× 2(a−t)", ""];
    back.forEach(([name, val], i) => {
      s += bx(X[i] - 46, 178, 92, 46, name, "node bad");
      s += tx(X[i], 246, val, "font-size:12px;fill:var(--red);font-weight:700");
      if (i < 4) s += arr(X[i + 1] - 48, 201, X[i] + 48, 201, id, true) + tx((X[i] + X[i + 1]) / 2, 194, loc[i], "font-size:10.5px;fill:var(--red)");
    });
    s += tx(300, 276, "also ∂L/∂w = ∂L/∂(wx) × x = " + c(dw) + "   and   ∂L/∂b = ∂L/∂z = " + c(db), "font-size:12px;fill:var(--text)");
    return wrap(600, 292, s, id);
  };

  // the training loop as a cycle
  D.dl1_loop = () => {
    const id = "dt1", Bx = [[20, 'Get a mini-batch', 'x, y'], [170, 'Forward pass', 'ŷ = model(x)'], [320, 'Compute loss', 'L = loss(ŷ, y)'], [470, 'Backward pass', 'gradients ∂L/∂θ']];
    let s = "";
    Bx.forEach(([x, a, b], i) => {
      s += `<rect class="node${i === 2 ? " hl" : ""}" x="${x}" y="40" width="120" height="56" rx="8"/><text class="val" x="${x + 60}" y="64" text-anchor="middle" style="font-size:13px">${esc(a)}</text><text class="lbl" x="${x + 60}" y="83" text-anchor="middle">${esc(b)}</text>`;
      if (i < 3) s += arr(x + 122, 68, Bx[i + 1][0] - 2, 68, id);
    });
    s += `<rect class="node ok" x="470" y="150" width="120" height="56" rx="8"/><text class="val" x="530" y="174" text-anchor="middle" style="font-size:13px">Update weights</text><text class="lbl" x="530" y="193" text-anchor="middle">θ ← θ − η·step</text>`;
    s += arr(530, 98, 530, 148, id);
    s += `<rect class="node" x="170" y="150" width="190" height="56" rx="8"/><text class="val" x="265" y="174" text-anchor="middle" style="font-size:13px">zero the gradients</text><text class="lbl" x="265" y="193" text-anchor="middle">then repeat with the next batch</text>`;
    s += arr(468, 178, 362, 178, id) + `<path d="M170 178 C 80 178, 80 140, 80 100" fill="none" style="stroke:var(--muted);stroke-width:1.8" marker-end="url(#${id})"/>`;
    s += tx(300, 232, "one pass through all the batches = one epoch; repeat for many epochs, checking a validation set after each", "font-size:12px;fill:var(--text)");
    return wrap(610, 250, s, id);
  };

  // learned feature hierarchy: what "deep" means
  D.dl1_hier = () => {
    const id = "dh1", Lv = [["pixels", "raw numbers"], ["edges", "layer 1"], ["textures, corners", "layer 2"], ["parts: eyes, ears", "layer 3"], ["object: cat", "last layer"]];
    let s = "";
    Lv.forEach(([a, b], i) => {
      const x = 14 + i * 118;
      s += `<rect class="node${i === 4 ? " ok" : i === 0 ? "" : " hl"}" x="${x}" y="40" width="100" height="58" rx="8"/><text class="val" x="${x + 50}" y="66" text-anchor="middle" style="font-size:12px">${esc(a)}</text><text class="lbl" x="${x + 50}" y="85" text-anchor="middle">${esc(b)}</text>`;
      if (i < 4) s += arr(x + 102, 69, x + 116, 69, id);
    });
    s += tx(300, 130, "simple features → combined into more complex ones, layer by layer (learned, not hand-designed)", "font-size:12px;fill:var(--text)");
    return wrap(600, 150, s, id);
  };

  /* ------------------------------------------------------------------ glossary */
  Object.assign(window.GLOSSARY, {
    "neuron": "The basic unit of a neural network: it takes a weighted sum of its inputs, adds a bias, and passes the result through an activation function.",
    "perceptron": "The simplest neuron: a weighted sum followed by a step function (output 1 if the sum is above 0, else 0). It can only learn straight-line boundaries.",
    "bias": "A learned constant added to a neuron's weighted sum. It shifts the activation threshold, like the intercept of a line.",
    "hidden layer": "A layer between the input and the output of a network. Its values are not data you give or answers you read; the network learns what to put there.",
    "forward pass": "Running an input through the network, layer by layer, to get a prediction (and then the loss).",
    "backward pass": "Running backpropagation: sending gradients from the loss back through the network to every weight.",
    "relu": "Rectified linear unit: f(x) = max(0, x). The default activation for hidden layers; cheap, and its slope is 1 for positive inputs.",
    "saturation": "When an activation function is flat (its slope near 0) for large inputs, so almost no gradient flows through it. Typical for sigmoid and tanh.",
    "dead relu": "A ReLU neuron whose input is negative for every example, so its output and its gradient are always 0 and it can never recover.",
    "computational graph": "The calculation written as a graph of small operations (add, multiply, sigmoid…). Backpropagation walks this graph backwards.",
    "autograd": "Automatic differentiation: software that records the operations of a calculation and computes all gradients for you.",
    "optimizer": "The rule that turns gradients into weight updates: plain SGD, momentum, RMSProp, Adam…",
    "sgd": "Stochastic gradient descent: update the weights using the gradient of a small random batch, step = learning rate × gradient.",
    "rmsprop": "An optimizer that divides each gradient by a running average of its recent size, so every parameter gets a sensible step size.",
    "adam": "The most popular optimizer: momentum (average of gradients) combined with RMSProp-style scaling, plus bias correction for the first steps.",
    "learning-rate schedule": "A rule that changes the learning rate during training, for example decaying it or warming it up.",
    "warmup": "Starting training with a very small learning rate and ramping it up over the first steps, to avoid unstable early updates.",
    "cosine decay": "A schedule that lowers the learning rate from its peak to a small value along half a cosine wave.",
    "dropout": "A regulariser that randomly switches off a fraction of the units at each training step, so the network cannot depend on any single one.",
    "weight decay": "Shrinking the weights a little at every step, which pulls them toward zero. For plain SGD it is the same as L2 regularization.",
    "batch normalization": "A layer that normalises each feature across the mini-batch to mean 0 and variance 1, then rescales it with learned parameters. It makes deep nets easier to train.",
    "batchnorm": "Short for batch normalization: normalise each feature over the mini-batch, then rescale with learned parameters.",
    "vanishing gradient": "Gradients that shrink toward 0 as they travel back through many layers, so the early layers barely learn.",
    "exploding gradient": "Gradients that grow larger and larger as they travel back through many layers, making the weights jump wildly or become NaN.",
    "gradient clipping": "Capping the size of the gradient before the update, to stop exploding gradients from wrecking training.",
    "initialization": "The starting values of the weights. Random values with the right scale (Xavier, He) keep signals and gradients from vanishing or exploding.",
    "xavier": "Xavier (Glorot) initialization: random weights with variance about 1/fan-in, designed for tanh and sigmoid layers.",
    "he initialization": "Kaiming/He initialization: random weights with variance 2/fan-in, designed for ReLU layers.",
    "deep learning": "Machine learning with neural networks that have many layers, which learn their own features from raw data.",
    "tensor": "A multi-dimensional array of numbers (a vector is 1-D, a matrix 2-D). The basic data type of deep learning libraries.",
    "iteration": "One weight update on one mini-batch. An epoch is many iterations.",
    "universal approximation": "A theorem: a network with one hidden layer and enough neurons can approximate any reasonable function. It says nothing about how easy it is to train.",
    "softplus": "A smooth version of ReLU: ln(1 + e^x).",
    "label smoothing": "Training against slightly softened targets (0.9 / 0.1 instead of 1 / 0), which discourages over-confident predictions.",
  });
})();
