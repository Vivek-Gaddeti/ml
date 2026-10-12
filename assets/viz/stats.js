/* Statistics for ML: visualizers, diagrams and glossary terms.
   Loaded on every doc page and on Practice, after assets/viz.js. Every number shown is computed (seeded random numbers), not faked. */
(function () {
  const V = Viz.defs, Plot = Viz.Plot, { r3 } = Viz.helpers;

  /* ------------------------------------------------------------------ private helpers */
  const soft = (v, p = 18) => `color-mix(in srgb, var(--${v}) ${p}%, transparent)`;
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const mean = (a) => sum(a) / a.length;
  const sd = (a) => { const m = mean(a); return Math.sqrt(sum(a.map((v) => (v - m) ** 2)) / (a.length - 1)); };
  const median = (a) => { const s = a.slice().sort((x, y) => x - y), n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
  const pct = (a, q) => { const s = a.slice().sort((x, y) => x - y), h = (s.length - 1) * q, lo = Math.floor(h); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (h - lo); };
  const f = (v, d = 3) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : +v).toFixed(d);
  const pfmt = (p) => (p < 0.0001 ? "< 0.0001" : p.toFixed(4));
  // seeded random numbers (mulberry32) so every picture is the same each time
  const mulberry = (seed) => { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  const gauss = (rnd) => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
  const expo = (rnd) => -Math.log(1 - rnd());
  // normal cdf / quantile
  const erf = (x) => { const s = Math.sign(x), t = 1 / (1 + 0.3275911 * Math.abs(x)), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; };
  const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  const phi = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  function probit(p) {            // Acklam's rational approximation of the normal quantile
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    const pl = 0.02425; let q, r;
    if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - pl) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    q = p - 0.5; r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  // Student t quantile through the regularised incomplete beta function
  function lgam(x) { const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.001208650973866179, -0.000005395239384953]; let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015; for (let j = 0; j < 6; j++) s += c[j] / ++y; return -t + Math.log(2.5066282746310005 * s / x); }
  function betacf(a, b, x) {
    const FP = 1e-300; const qab = a + b, qap = a + 1, qam = a - 1; let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < FP) d = FP; d = 1 / d; let h = d;
    for (let m = 1; m <= 200; m++) {
      const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-12) break;
    }
    return h;
  }
  function betai(a, b, x) { if (x <= 0) return 0; if (x >= 1) return 1; const bt = Math.exp(lgam(a + b) - lgam(a) - lgam(b) + a * Math.log(x) + b * Math.log(1 - x)); return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; }
  const tcdf = (t, df) => { const p = 0.5 * betai(df / 2, 0.5, df / (df + t * t)); return t > 0 ? 1 - p : p; };
  function tquant(p, df) { let lo = -200, hi = 200; for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (tcdf(m, df) < p) lo = m; else hi = m; } return (lo + hi) / 2; }

  // layers
  const poly = (pts, style) => (S) => `<polygon points="${pts.map(([x, y]) => S.sx(x).toFixed(1) + "," + S.sy(y).toFixed(1)).join(" ")}" style="${style}"/>`;
  const bars = (counts, lo, w, style) => (S) => counts.map((c, i) => (c > 0 ? `<rect x="${S.sx(lo + i * w).toFixed(1)}" y="${S.sy(c).toFixed(1)}" width="${Math.max(1, S.sx(lo + w) - S.sx(lo) - 1).toFixed(1)}" height="${(S.sy(0) - S.sy(c)).toFixed(1)}" style="${style}"/>` : "")).join("");
  const tag = (x, y, s, style = "", anchor = "middle") => (S) => `<text x="${S.sx(x).toFixed(1)}" y="${S.sy(y).toFixed(1)}" text-anchor="${anchor}" style="font-size:12px;fill:var(--muted);${style}">${esc(s)}</text>`;
  const binCounts = (vals, lo, w, nb) => { const c = new Array(nb).fill(0); vals.forEach((v) => { const i = Math.floor((v - lo) / w); if (i >= 0 && i < nb) c[i]++; }); return c; };
  const area = (fn, a, b, n = 80) => { const p = [[a, 0]]; for (let i = 0; i <= n; i++) { const x = a + ((b - a) * i) / n; p.push([x, fn(x)]); } p.push([b, 0]); return p; };

  /* ================================================================== 1. Descriptive statistics: outliers */
  V.stats_descriptive = {
    title: "Mean vs median: one outlier moves the mean, not the median",
    params: [{ id: "out", label: "value of the last person (k$)", min: 45, max: 400, step: 5, value: 60 }],
    gen(_, P) {
      const base = [31, 34, 36, 38, 40, 41, 43, 45, 47, 52, 55];
      const data = base.concat([P.out]);
      const m = mean(data), md = median(data), s = sd(data), q1 = pct(data, 0.25), q3 = pct(data, 0.75);
      const maxX = Math.max(120, P.out + 20);
      const stack = {}, pts = data.map((v, i) => { const k = Math.round(v / 3); stack[k] = (stack[k] || 0) + 1; return [v, stack[k], i === data.length - 1 ? "c2" : "c1"]; });
      const svg = Plot.chart({ W: 560, H: 230, x: [20, maxX], y: [0, 6], xl: "salary (thousand $)", yticks: 3 }, [
        poly([[q1, 0], [q1, 5.9], [q3, 5.9], [q3, 0]], `fill:${soft("accent", 12)};stroke:none`),
        Plot.vline(m, "pl-c4 pl-thin"), Plot.vline(md, "pl-c3 pl-thin"),
        Plot.pts(pts.map(([x, y, c]) => [x, y - 0.4, c === "c2" ? "a" : "b"]), "pl-c1", 6),
        tag(m, 5.5, "mean " + f(m, 1), "fill:var(--red);font-weight:700", m > (20 + maxX) / 2 ? "end" : "start"),
        tag(md, 4.9, "median " + f(md, 1), "fill:var(--green);font-weight:700", md > (20 + maxX) / 2 ? "end" : "end"),
      ]);
      return [{
        svg, side: { mean: f(m, 1), median: f(md, 1), "std dev": f(s, 1), "IQR": f(q3 - q1, 1) },
        msg: `Twelve salaries; slide the last one. The **mean** (red) is pulled toward the big value, the **median** (green) barely moves, and the shaded box is the middle half of the data (the IQR, ${f(q1, 1)} to ${f(q3, 1)}). The standard deviation explodes with an outlier, the IQR does not. For skewed or messy data report the median and IQR.`,
      }];
    },
  };

  /* ================================================================== 2. Central limit theorem */
  V.stats_clt = {
    title: "Central limit theorem: averages of a skewed population turn bell-shaped",
    params: [{ id: "n", label: "sample size n", min: 1, max: 50, step: 1, value: 5 }],
    gen(_, P) {
      const n = P.n, rnd = mulberry(100 + n), TOT = 2000, means = [];
      for (let i = 0; i < TOT; i++) { let s = 0; for (let j = 0; j < n; j++) s += expo(rnd); means.push(s / n); }
      const lo = 0, hi = 4, nb = 40, w = (hi - lo) / nb, se = 1 / Math.sqrt(n);
      const finalC = binCounts(means, lo, w, nb), ymax = Math.max(Math.max(...finalC), TOT * w * phi(0) / se * (1)) * 1.15;
      const checks = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000], steps = [];
      checks.forEach((M) => {
        const part = means.slice(0, M), c = binCounts(part, lo, w, nb);
        const curve = (x) => M * w * phi((x - 1) / se) / se, pop = (x) => M * w * Math.exp(-x);
        const ym = Math.max(3, Math.max(...c) * 1.25, M * w * phi(0) / se * 1.1);
        const svg = Plot.chart({ W: 560, H: 300, x: [lo, hi], y: [0, ym], xl: "sample mean x̄ (each bar counts samples whose mean falls there)", yl: "number of samples", yticks: 4 }, [
          bars(c, lo, w, `fill:${soft("accent", 55)};stroke:none`),
          Plot.fn(pop, "pl-c2 pl-thin pl-dash"), Plot.fn(curve, "pl-c4"),
          Plot.vline(1, "pl-dash"),
        ]);
        steps.push({
          svg, side: { "sample size n": n, "samples drawn": M, "mean of means": M > 0 ? f(mean(part), 3) : "-", "sd of means": M > 1 ? f(sd(part), 3) : "-", "theory σ/√n": f(se, 3) },
          msg: M === 1 ? `The population is **exponential**: very skewed (amber dashed curve, mean 1, std 1). We draw **one sample of ${n}** values and record its mean (${f(means[0], 2)}). One bar of height 1.` :
            `After **${M}** samples of size ${n}, each contributes its mean to the histogram. ${n === 1 ? "With n = 1 a 'mean' is just one value, so the histogram copies the skewed population." : n < 10 ? "Small n: still a bit skewed, the red normal curve is only a rough fit." : "The histogram is now close to the red normal curve with mean 1 and standard error 1/√" + n + " = " + f(se, 3) + "."} Slide n to see the bell get narrower and more symmetric.`,
        });
      });
      return steps;
    },
  };

  /* ================================================================== 3. p-value as a tail area */
  V.stats_pvalue = {
    title: "The p-value is a tail area under the null distribution",
    params: [
      { id: "z", label: "observed test statistic z", min: -4, max: 4, step: 0.05, value: 2 },
      { id: "tails", label: "tails (1 or 2)", min: 1, max: 2, step: 1, value: 2 },
    ],
    gen(_, P) {
      const z = P.z, two = P.tails === 2, a = 0.05;
      const p = two ? 2 * (1 - Phi(Math.abs(z))) : 1 - Phi(z);
      const crit = two ? probit(1 - a / 2) : probit(1 - a);
      const layers = [];
      if (two) { layers.push(poly(area(phi, Math.abs(z), 4), `fill:${soft("red", 45)};stroke:none`), poly(area(phi, -4, -Math.abs(z)), `fill:${soft("red", 45)};stroke:none`)); }
      else layers.push(poly(area(phi, Math.min(z, 4), 4), `fill:${soft("red", 45)};stroke:none`));
      layers.push(Plot.fn(phi, "pl-c1"), Plot.vline(z, "pl-c4"), Plot.vline(crit, "pl-dash"));
      if (two) layers.push(Plot.vline(-crit, "pl-dash"));
      layers.push(tag(z, 0.43, "observed z = " + f(z, 2), "fill:var(--red);font-weight:700", z > 0 ? "start" : "end"), tag(crit + 0.1, 0.13, "α = 0.05 cut-off", "", "start"));
      const svg = Plot.chart({ W: 560, H: 290, x: [-4, 4], y: [0, 0.47], xl: "test statistic if the null hypothesis is true", yl: "density", yticks: 4 }, layers);
      return [{
        svg, side: { "test statistic z": f(z, 2), "p-value": pfmt(p), [`cut-off at α = 0.05`]: (two ? "±" : "") + f(crit, 3), decision: p < a ? "reject H0" : "cannot reject H0" },
        msg: `If H0 were true, the test statistic would follow this bell curve. The **p-value** is the probability of a result **at least this extreme**: the red area, **${pfmt(p)}**${two ? " (both tails, because a 'two-sided' test counts big values in either direction)" : " (one tail only, we only care about large positive z)"}. It is **not** the probability that H0 is true. Drag z outward: the red area shrinks. Past the dashed cut-off the area drops under 5%.`,
      }];
    },
  };

  /* ================================================================== 4. Errors and power */
  V.stats_power = {
    title: "Type I error, Type II error and power",
    params: [
      { id: "d", label: "effect size d (in std devs)", min: 0, max: 1.5, step: 0.05, value: 0.5 },
      { id: "a", label: "significance level α", min: 0.01, max: 0.2, step: 0.01, value: 0.05 },
      { id: "n", label: "samples per group n", min: 5, max: 300, step: 5, value: 30 },
    ],
    gen(_, P) {
      const { d, a, n } = P, delta = d * Math.sqrt(n / 2), zc = probit(1 - a / 2);
      const beta = Phi(zc - delta) - Phi(-zc - delta), power = 1 - beta;
      const h0 = phi, h1 = (x) => phi(x - delta), X0 = -4, X1 = Math.max(6, delta + 4);
      const svg = Plot.chart({ W: 560, H: 300, x: [X0, X1], y: [0, 0.46], xl: "standardised difference between group means", yl: "density", yticks: 4 }, [
        poly(area(h1, X0, zc), `fill:${soft("amber", 38)};stroke:none`),
        poly(area(h1, Math.max(zc, X0), X1), `fill:${soft("green", 38)};stroke:none`),
        poly(area(h0, zc, X1), `fill:${soft("red", 55)};stroke:none`), poly(area(h0, X0, -zc), `fill:${soft("red", 55)};stroke:none`),
        Plot.fn(h0, "pl-c1"), Plot.fn(h1, "pl-c5"), Plot.vline(zc, "pl-dash"), Plot.vline(-zc, "pl-dash"),
        tag(-0.4, 0.43, "H0: no difference", "fill:var(--accent);font-weight:700", "end"), tag(delta + 0.4, 0.43, "H1: true difference", "fill:var(--purple);font-weight:700", "start"),
      ]);
      const nNeed = d > 0 ? Math.ceil(2 * ((zc + probit(0.8)) / d) ** 2) : Infinity;
      return [{
        svg, side: { "Type I rate (red) = α": f(a, 2), "Type II rate β (amber)": f(beta, 3), "power 1 − β (green)": f(power, 3), "n per group for 80% power": isFinite(nNeed) ? nNeed : "-" },
        msg: `Left curve (H0): the sampling distribution if there is **no** effect. Purple curve (H1): if the true effect is **d = ${f(d, 2)}** standard deviations. We call a result 'significant' beyond the dashed lines. **Red** = Type I error (false alarm, rate α). **Amber** = Type II error (a real effect that we miss, rate β = ${f(beta, 2)}). **Green** = power (we detect the real effect) = **${f(power, 2)}**. Raise n or the effect size and the curves pull apart, power rises. Lower α and the cut-offs move out, so power falls.`,
      }];
    },
  };

  /* ================================================================== 5. Confidence-interval coverage */
  V.stats_ci = {
    title: "Confidence intervals: 95% of them catch the true mean",
    params: [
      { id: "conf", label: "confidence level (%)", min: 50, max: 99, step: 1, value: 95 },
      { id: "n", label: "sample size n", min: 3, max: 60, step: 1, value: 10 },
    ],
    gen(_, P) {
      const n = P.n, C = P.conf / 100, rnd = mulberry(500 + n), tc = tquant(1 - (1 - C) / 2, n - 1), all = [];
      for (let i = 0; i < 50; i++) { const x = []; for (let j = 0; j < n; j++) x.push(gauss(rnd)); const m = mean(x), h = tc * sd(x) / Math.sqrt(n); all.push({ m, lo: m - h, hi: m + h }); }
      const R = Math.max(1, Math.min(3, Math.ceil(Math.max(...all.map((v) => Math.max(Math.abs(v.lo), Math.abs(v.hi)))) * 2) / 2));
      const steps = [];
      for (let k = 5; k <= 50; k += 5) {
        const part = all.slice(0, k), hit = part.filter((v) => v.lo <= 0 && v.hi >= 0).length, L = [Plot.vline(0, "pl-c4")];
        part.forEach((v, i) => { const y = 51 - i, ok = v.lo <= 0 && v.hi >= 0; L.push(Plot.seg(Math.max(-R, v.lo), y, Math.min(R, v.hi), y, (ok ? "pl-c3" : "pl-c4") + " pl-thin")); L.push(Plot.dot(v.m, y, ok ? "pl-c3" : "pl-c4", 2.6)); });
        const svg = Plot.chart({ W: 560, H: 380, x: [-R, R], y: [0, 52], xl: "estimate of the mean (true mean = 0, red line)", yl: "sample 1 ... 50", yticks: 5 }, L);
        steps.push({
          svg, side: { "level": P.conf + "%", "t critical value": f(tc, 3), "intervals drawn": k, "contain the true mean": `${hit} of ${k}`, "share": f(hit / k, 2) },
          msg: `Each row is a **new sample** of ${n} values from a population whose true mean is **0**; we compute x̄ ± t · s/√n. Green intervals contain 0, red ones miss. In the long run about **${P.conf}%** of intervals catch the truth (here ${hit} of ${k} so far). The level describes the **procedure**, not one interval. Raise the level and the intervals widen, raise n and they shrink.`,
        });
      }
      return steps;
    },
  };

  /* ================================================================== 6. Bootstrap */
  V.stats_bootstrap = {
    title: "The bootstrap: resample your own data with replacement",
    params: [{ id: "st", label: "statistic (0 = mean, 1 = median)", min: 0, max: 1, step: 1, value: 0 }],
    gen(_, P) {
      const rnd = mulberry(11), data = []; for (let i = 0; i < 15; i++) data.push(+(1 + Math.exp(0.7 * gauss(rnd))).toFixed(2));
      const stat = P.st ? median : mean, name = P.st ? "median" : "mean", B = 1000, r2 = mulberry(5), bs = [];
      for (let b = 0; b < B; b++) { const s = []; for (let i = 0; i < data.length; i++) s.push(data[Math.floor(r2() * data.length)]); bs.push(stat(s)); }
      const lo = Math.floor(Math.min(...bs) * 4) / 4, hi = Math.ceil(Math.max(...bs) * 4) / 4, nb = 36, w = (hi - lo) / nb;
      const obs = stat(data), steps = [];
      [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000].forEach((M) => {
        const part = bs.slice(0, M), c = binCounts(part, lo, w, nb), ymax = Math.max(3, Math.max(...c) * 1.3), L = [bars(c, lo, w, `fill:${soft("accent", 55)};stroke:none`), Plot.vline(obs, "pl-c1")];
        let ci = null; if (M >= 100) { ci = [pct(part, 0.025), pct(part, 0.975)]; L.push(Plot.vline(ci[0], "pl-c4"), Plot.vline(ci[1], "pl-c4"), poly([[ci[0], ymax * 0.97], [ci[1], ymax * 0.97], [ci[1], ymax * 0.93], [ci[0], ymax * 0.93]], `fill:${soft("red", 40)};stroke:none`)); }
        L.push(Plot.pts(data.map((v) => [v, ymax * 0.03, "b"]), "pl-c1", 3.5));
        const svg = Plot.chart({ W: 560, H: 300, x: [lo, hi], y: [0, ymax], xl: `bootstrap ${name} (dots at the bottom = your 15 data values)`, yl: "number of resamples", yticks: 4 }, L);
        steps.push({
          svg, side: { ["your sample's " + name]: f(obs, 3), "resamples": M, "bootstrap std error": M > 1 ? f(sd(part), 3) : "-", "95% percentile CI": ci ? `${f(ci[0], 2)} to ${f(ci[1], 2)}` : "need 100+" },
          msg: M === 1 ? `We have **one sample of 15 numbers** and no formula for the uncertainty of its ${name}. Trick: pretend the sample is the population. Draw 15 values **with replacement** from it, compute the ${name}, and record it. That is resample 1.` :
            `${M} resamples so far. Each is a fake 'new study' made from your data; the spread of their ${name}s estimates the **standard error** (${M > 1 ? f(sd(part), 3) : "-"}). ${ci ? "The middle 95% of the bootstrap values (red) is a **percentile confidence interval**: " + f(ci[0], 2) + " to " + f(ci[1], 2) + "." : "Keep stepping: after 100 resamples the 2.5% and 97.5% percentiles give a confidence interval."}`,
        });
      });
      return steps;
    },
  };

  /* ================================================================== 7. Permutation test */
  V.stats_permutation = {
    title: "Permutation test: shuffle the labels, build the null distribution",
    gen() {
      const A = [14.1, 15.3, 12.8, 16.2, 13.9, 15.8, 14.7, 16.5], B = [12.9, 13.4, 11.8, 14.0, 12.2, 13.1, 12.7, 14.4];
      const vals = A.concat(B), lab0 = vals.map((_, i) => (i < 8 ? 0 : 1)), obs = mean(A) - mean(B), rnd = mulberry(3), T = 2000;
      const labs = [lab0], diffs = [];
      for (let t = 0; t < T; t++) { const idx = vals.map((_, i) => i); for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; } const lb = new Array(16).fill(1); idx.slice(0, 8).forEach((i) => (lb[i] = 0)); labs.push(lb); const ma = mean(vals.filter((_, i) => lb[i] === 0)), mb = mean(vals.filter((_, i) => lb[i] === 1)); diffs.push(ma - mb); }
      const strip = (lb, dm) => {
        const W = 560, H = 140, sx = (v) => 40 + ((v - 11) / 6.5) * 500; let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img"><line x1="40" y1="108" x2="540" y2="108" style="stroke:var(--border)"/>`;
        [12, 13, 14, 15, 16, 17].forEach((t) => (s += `<line x1="${sx(t)}" y1="105" x2="${sx(t)}" y2="111" style="stroke:var(--muted)"/><text x="${sx(t)}" y="127" text-anchor="middle" style="font-size:11px;fill:var(--muted)">${t}</text>`));
        const used = {}; vals.forEach((v, i) => { const key = lb[i] + "_" + Math.round(v * 2); used[key] = (used[key] || 0) + 1; const y = (lb[i] === 0 ? 44 : 78) - (used[key] - 1) * 0; s += `<circle cx="${sx(v) + (used[key] - 1) * 9}" cy="${y}" r="7" style="fill:var(--cls-${lb[i] === 0 ? "a" : "b"});stroke:var(--panel);stroke-width:1.5"/>`; });
        const ma = mean(vals.filter((_, i) => lb[i] === 0)), mb = mean(vals.filter((_, i) => lb[i] === 1));
        s += `<line x1="${sx(ma)}" y1="28" x2="${sx(ma)}" y2="60" style="stroke:var(--cls-a);stroke-width:2.5"/><line x1="${sx(mb)}" y1="60" x2="${sx(mb)}" y2="92" style="stroke:var(--cls-b);stroke-width:2.5"/>`;
        s += `<text x="8" y="48" style="font-size:12px;fill:var(--cls-a);font-weight:700">A</text><text x="8" y="82" style="font-size:12px;fill:var(--cls-b);font-weight:700">B</text>`;
        s += `<text x="548" y="16" text-anchor="end" style="font-size:12px;fill:var(--text)">mean A − mean B = ${f(dm, 2)}</text></svg>`; return s;
      };
      const lo = -2.5, hi = 2.5, nb = 40, w = (hi - lo) / nb, steps = [];
      [0, 1, 2, 3, 5, 10, 25, 100, 500, 2000].forEach((t) => {
        const part = diffs.slice(0, t), c = binCounts(part, lo, w, nb), ymax = Math.max(3, Math.max(...c) * 1.3), ext = part.filter((d) => Math.abs(d) >= Math.abs(obs) - 1e-12).length, p = t ? ext / t : NaN;
        const L = [bars(c, lo, w, `fill:${soft("accent", 55)};stroke:none`), Plot.vline(obs, "pl-c4"), Plot.vline(-obs, "pl-c4 pl-dash")];
        if (t) L.push(tag(0, ymax * 0.95, `${ext} of ${t} shuffles are as extreme`, "fill:var(--text)"));
        const chart = Plot.chart({ W: 560, H: 230, x: [lo, hi], y: [0, ymax], xl: "difference in means after shuffling (red line = what we actually observed)", yl: "shuffles", yticks: 3 }, L);
        steps.push({
          svg: strip(labs[t], t ? diffs[t - 1] : obs) + chart,
          side: { "observed difference": f(obs, 2), "shuffles": t, "as extreme": t ? ext : "-", "permutation p-value": t ? f(p, 4) : "-" },
          msg: t === 0 ? `Real data: 8 values labelled A, 8 labelled B. A is **${f(obs, 2)}** higher on average. If the label did nothing, any other split of these 16 numbers into 8 + 8 would be just as likely. So we will **shuffle the labels** and see how big a difference pure chance makes.` :
            t === 1 ? `Shuffle 1: the same 16 numbers, labels randomly reassigned. The difference between the two means is now ${f(diffs[0], 2)}. Record it in the histogram below, then repeat.` :
            t < 100 ? `${t} shuffles. The histogram of 'chance' differences is building up, centred on 0 because the labels are meaningless. ${ext} shuffles gave a gap at least as large as ours (either direction).` :
            `${t} shuffles: this is the **null distribution** of the difference. Only ${ext} of ${t} shuffles are as extreme as the real difference, so the permutation p-value is about **${f(p, 4)}**. Chance rarely produces a gap this big, so the labels probably do matter.`,
        });
      });
      return steps;
    },
  };

  /* ================================================================== 8. Correlation and Anscombe */
  const ANS = {
    1: { x: [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], y: [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68], note: "Set I: a plain, roughly linear cloud. Here r is a fair summary." },
    2: { x: [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], y: [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74], note: "Set II: a perfect smooth curve. r is high, but a straight line is the wrong model." },
    3: { x: [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], y: [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73], note: "Set III: a perfect line plus one outlier that tilts the fitted line and changes r." },
    4: { x: [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], y: [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89], note: "Set IV: x never varies except for one extreme point, which alone creates the whole correlation." },
  };
  const corr = (x, y) => { const mx = mean(x), my = mean(y); let sxy = 0, sxx = 0, syy = 0; x.forEach((v, i) => { sxy += (v - mx) * (y[i] - my); sxx += (v - mx) ** 2; syy += (y[i] - my) ** 2; }); return sxy / Math.sqrt(sxx * syy); };
  const ols = (x, y) => { const mx = mean(x), my = mean(y); let sxy = 0, sxx = 0; x.forEach((v, i) => { sxy += (v - mx) * (y[i] - my); sxx += (v - mx) ** 2; }); const b = sxy / sxx; return [b, my - b * mx]; };
  V.stats_corr = {
    title: "Correlation: the number r, and why you must look at the plot",
    params: [
      { id: "r", label: "target correlation ρ", min: -1, max: 1, step: 0.05, value: 0.7 },
      { id: "set", label: "data (0 = random cloud, 1-4 = Anscombe)", min: 0, max: 4, step: 1, value: 0 },
    ],
    gen(_, P) {
      let x, y, note;
      if (P.set === 0) {
        const rnd = mulberry(21), n = 80; x = []; y = [];
        for (let i = 0; i < n; i++) { const a = gauss(rnd), b = gauss(rnd); x.push(a); y.push(P.r * a + Math.sqrt(1 - P.r * P.r) * b); }
        note = `A random cloud built to have correlation about ${f(P.r, 2)}. The sample value r differs slightly from the target because n = 80 is finite. r = 0 means no **linear** relationship, not no relationship at all.`;
      } else { x = ANS[P.set].x; y = ANS[P.set].y; note = ANS[P.set].note + " All four Anscombe sets have almost identical mean, variance, r ≈ 0.816 and fitted line."; }
      const r = corr(x, y), [b, a0] = ols(x, y), xs = x.concat([]), x0 = Math.min(...xs), x1 = Math.max(...xs), ys = y;
      const padx = (x1 - x0) * 0.12 + 0.5, y0 = Math.min(...ys), y1 = Math.max(...ys), pady = (y1 - y0) * 0.12 + 0.5;
      const cx = [x0 - padx, x1 + padx], cy = [y0 - pady, y1 + pady];
      const svg = Plot.chart({ W: 520, H: 340, x: cx, y: cy, xl: "x", yl: "y" }, [
        Plot.line([[cx[0], a0 + b * cx[0]], [cx[1], a0 + b * cx[1]]], "pl-c4 pl-thin"), Plot.pts(x.map((v, i) => [v, y[i], "a"]), "pl-c1", P.set ? 6 : 4),
      ]);
      return [{ svg, side: { "Pearson r": f(r, 3), "r²": f(r * r, 3), "slope": f(b, 3), n: x.length }, msg: note }];
    },
  };

  /* ================================================================== 9. Simpson's paradox */
  V.stats_simpson = {
    title: "Simpson's paradox: the trend flips when you ignore the group",
    params: [
      { id: "gap", label: "how much worse the severe group does", min: 0, max: 7, step: 0.25, value: 4 },
      { id: "col", label: "colour by patient group (0 / 1)", min: 0, max: 1, step: 1, value: 1, fmt: (v) => (+v ? "on" : "off") },
    ],
    gen(_, P) {
      const rnd = mulberry(8), xa = [], ya = [], xb = [], yb = [];
      for (let i = 0; i < 14; i++) { const x = 1 + rnd() * 4; xa.push(x); ya.push(10 + 0.8 * (x - 3) + 0.9 * gauss(rnd)); }
      for (let i = 0; i < 14; i++) { const x = 4 + rnd() * 4; xb.push(x); yb.push(10 - P.gap + 0.8 * (x - 6) + 0.9 * gauss(rnd)); }
      const xs = xa.concat(xb), ys = ya.concat(yb), [bAll, aAll] = ols(xs, ys), [bA, aA] = ols(xa, ya), [bB, aB] = ols(xb, yb);
      const L = [];
      L.push(Plot.line([[0.5, aAll + bAll * 0.5], [8.5, aAll + bAll * 8.5]], "pl-c4"));
      if (P.col) { L.push(Plot.line([[0.8, aA + bA * 0.8], [5.2, aA + bA * 5.2]], "pl-c1 pl-thin"), Plot.line([[3.8, aB + bB * 3.8], [8.2, aB + bB * 8.2]], "pl-c2 pl-thin")); }
      L.push(Plot.pts(xa.map((v, i) => [v, ya[i], P.col ? "a" : "c1"]), "pl-c1", 5), Plot.pts(xb.map((v, i) => [v, yb[i], P.col ? "b" : "c1"]), "pl-c1", 5));
      const svg = Plot.chart({ W: 540, H: 320, x: [0.5, 8.5], y: [0, 14], xl: "dose of the drug", yl: "recovery score" }, L);
      const flip = bAll * ((bA + bB) / 2) < 0;
      return [{
        svg, side: { "overall slope (red)": f(bAll, 2), "slope, mild group": f(bA, 2), "slope, severe group": f(bB, 2), "paradox?": flip ? "yes, signs differ" : "no" },
        msg: `Severe patients are given **higher doses** and also recover **worse** for reasons that have nothing to do with the drug. Inside each group a higher dose goes with better recovery (slopes ${f(bA, 2)} and ${f(bB, 2)}), but if you pool everyone the overall slope is **${f(bAll, 2)}**${flip ? ": the trend reverses!" : "."} ${P.col ? "" : "Turn the colouring on to see the groups. "}The group is a **confounder**: it influences both dose and outcome. Raise the gap slider to flip the pooled trend.`,
      }];
    },
  };

  /* ================================================================== diagrams */
  D.stats_confounder = () => {
    const node = (x, y, w, t, cls) => `<rect x="${x}" y="${y}" width="${w}" height="40" rx="10" style="fill:${cls};stroke:var(--border)"/><text x="${x + w / 2}" y="${y + 25}" text-anchor="middle" style="font-size:13px;fill:var(--text);font-weight:600">${esc(t)}</text>`;
    const arr = (x1, y1, x2, y2, col, dash) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#stcf)" style="stroke:${col};stroke-width:2.2;${dash ? "stroke-dasharray:6 5;" : ""}"/>`;
    return `<svg viewBox="0 0 560 230" width="560" height="230" xmlns="http://www.w3.org/2000/svg" role="img"><defs><marker id="stcf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" style="fill:context-stroke"/></marker></defs>
      ${node(215, 14, 130, "Confounder Z", soft("amber", 30))}${node(40, 160, 150, "Treatment X", soft("accent", 22))}${node(370, 160, 150, "Outcome Y", soft("green", 22))}
      ${arr(250, 56, 150, 156, "var(--amber)")}${arr(310, 56, 410, 156, "var(--amber)")}${arr(194, 180, 366, 180, "var(--red)", true)}
      <text x="280" y="170" text-anchor="middle" style="font-size:12px;fill:var(--red)">does X really cause Y?</text>
      <text x="100" y="100" text-anchor="end" style="font-size:12px;fill:var(--muted)">Z causes X</text><text x="470" y="100" style="font-size:12px;fill:var(--muted)">Z causes Y</text>
      <text x="280" y="222" text-anchor="middle" style="font-size:12px;fill:var(--muted)">X and Y move together even if the dashed arrow does not exist</text></svg>`;
  };

  /* ================================================================== glossary */
  const G = {
    "p-value": "If the null hypothesis were true, the probability of seeing a result at least as extreme as the one you got. It is not the probability that the null is true.",
    "null hypothesis": "The 'nothing is going on' claim that a test tries to find evidence against: no difference between groups, no effect, no relationship.",
    "alternative hypothesis": "What you suspect instead of the null hypothesis: there is a difference, an effect or a relationship.",
    "significance level": "The false-alarm rate you accept before looking at data, usually alpha = 0.05. A result with p below alpha is called statistically significant.",
    "type i error": "A false positive: rejecting the null hypothesis when it is actually true. Its rate is alpha.",
    "type ii error": "A false negative: failing to detect an effect that is really there. Its rate is beta.",
    "power": "The probability that a test detects a real effect of a given size (1 minus the Type II error rate). 80% is a common target.",
    "effect size": "How big a difference or relationship is in practical terms, for example Cohen's d (difference in means divided by the standard deviation). Unlike a p-value it does not grow with sample size.",
    "standard error": "How much an estimate (such as a sample mean) would typically wobble from sample to sample. For a mean it is s divided by the square root of n.",
    "sampling distribution": "The distribution of a statistic (like the sample mean) over many repeated samples from the same population.",
    "central limit theorem": "Averages (or sums) of many independent values are approximately normal, almost whatever the shape of the original data. Spread of the average shrinks like 1 over the square root of n.",
    "confidence interval": "A range computed from data by a procedure that captures the true value in a stated share (say 95%) of repeated studies. It is not a 95% chance that this particular range holds the truth.",
    "degrees of freedom": "The number of values that are free to vary after the constraints are used up; for a t-test on n values it is n - 1. It sets the exact shape of the t distribution.",
    "t-test": "A test that compares means using the t distribution, which accounts for estimating the spread from a small sample.",
    "chi-square test": "A test for counts in categories: it compares observed counts with the counts expected under the null hypothesis.",
    "anova": "Analysis of variance: a test that asks whether three or more group means could all be equal, by comparing between-group to within-group variation.",
    "permutation test": "A test that shuffles the group labels many times to see how big a difference chance alone produces, then compares it with the real difference.",
    "bonferroni correction": "When running m tests, use alpha divided by m for each one so that the chance of any false alarm stays at alpha. Simple but conservative.",
    "false discovery rate": "The expected share of your 'discoveries' that are false alarms. Controlled by the Benjamini-Hochberg procedure.",
    "p-hacking": "Trying many analyses, subgroups or stopping points until something gets p below 0.05, then reporting only that one. It makes false findings look real.",
    "confounder": "A third variable that influences both the cause you study and the outcome, creating an association that is not a causal effect.",
    "correlation": "A number from -1 to 1 measuring how closely two variables follow a straight-line relationship. It says nothing about cause.",
    "covariance": "The average of (x minus its mean) times (y minus its mean): positive when the two tend to rise together. Its size depends on the units.",
    "median": "The middle value when the data are sorted: half the values are below it and half above. Unlike the mean it ignores outliers.",
    "interquartile range": "The distance between the 25th and 75th percentiles: the spread of the middle half of the data.",
    "percentile": "The value below which a given percentage of the data falls. The 90th percentile has 90% of values under it.",
    "quantile": "Same idea as a percentile but on a 0-1 scale: the 0.9 quantile is the 90th percentile.",
    "histogram": "A bar chart that counts how many values fall into each range (bin), showing the shape of the data.",
    "skewness": "How lopsided a distribution is. Right-skewed data has a long tail of large values, so the mean sits above the median.",
    "contingency table": "A table of counts for combinations of two categorical variables, such as group by outcome.",
    "paired data": "Two measurements on the same unit (before and after, model A and model B on the same test set), so differences are analysed instead of two separate groups.",
    "mcnemar test": "A test for comparing two classifiers on the same test set that looks only at the examples where they disagree.",
    "peeking": "Checking an A/B test repeatedly and stopping as soon as p dips below 0.05. This inflates the false-positive rate far above 5%.",
    "simpson's paradox": "A trend that appears in every subgroup can reverse when the groups are pooled, because a confounder is unevenly spread over the groups.",
    "bayes factor": "How many times more likely the data are under one hypothesis than another; a Bayesian counterpart of a test.",
    "credible interval": "A Bayesian interval that holds the parameter with a stated probability (say 95%) given the data and the prior.",
    "r-squared": "The share of the variance in y explained by the model; for a single predictor it equals the squared correlation.",
    "heteroscedasticity": "Spread of the errors that changes with the predicted value, which breaks the usual standard errors of a regression.",
  };
  Object.keys(G).forEach((k) => { if (!(k in window.GLOSSARY)) window.GLOSSARY[k] = G[k]; });
})();
