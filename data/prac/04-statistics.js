/* Practice: Statistics (tests, resampling, multiple testing)
   Each problem: q (task), ex (examples), hint, brute (plain Python), opt (vectorized NumPy), run + out (the tests),
   dry (a dry run), expl (code explained line by line), take, mist. Tests print rounded values so floats compare exactly. */
PRAC.add("Statistics", [
{
  id: "chi-square-independence", t: "Chi-Square Test of Independence", d: "Easy", p: "Counts & contingency tables",
  q: "A website shows two button designs. For each design you count how many visitors clicked and how many did not. Is the click rate independent of the design?\n\nWrite `chi_square_test(table)` for a contingency table of counts (rows = groups, columns = outcomes). Return `(chi2, dof, p)`:\n\n1. The **expected** count of a cell is $E_{ij} = \\dfrac{(\\text{row total}) (\\text{column total})}{\\text{grand total}}$.\n2. $\\chi^2 = \\sum \\dfrac{(O - E)^2}{E}$ over all cells.\n3. The degrees of freedom are $(r-1)(c-1)$, and $p = P(\\chi^2_{\\text{dof}} \\ge \\chi^2)$.\n\nDo not use `scipy.stats.chi2_contingency` (you may use `scipy.stats.chi2.sf` for the final tail probability).",
  ex: [["table = [[120, 880], [150, 850]]", "chi2 = 3.854, dof = 1, p = 0.0496", "Expected: 135 clicks and 865 non-clicks in each row. The cells deviate by 15."],
    ["table = [[20, 30, 50], [30, 30, 40]]", "chi2 = 3.111, dof = 2, p = 0.2111", "More than two columns work the same way."]],
  c: ["`table` is a 2-D list/array of non-negative counts with every row and column total above 0", "No continuity correction: use the plain formula"],
  hint: "The whole matrix of expected counts is an **outer product**: row totals (a column vector) times column totals (a row vector), divided by the grand total. `keepdims=True` keeps the shapes right for broadcasting.",
  brute: {
    name: "Plain Python loops",
    idea: "Compute row and column totals with loops, then visit every cell, compute its expected count and add its contribution.",
    code: R`
import numpy as np
from scipy import stats

def chi_square_test(table):
    rows, cols = len(table), len(table[0])
    row_tot = [sum(table[i]) for i in range(rows)]
    col_tot = [sum(table[i][j] for i in range(rows)) for j in range(cols)]
    total = sum(row_tot)
    chi2 = 0.0
    for i in range(rows):
        for j in range(cols):
            e = row_tot[i] * col_tot[j] / total
            chi2 += (table[i][j] - e) ** 2 / e
    dof = (rows - 1) * (cols - 1)
    return chi2, dof, stats.chi2.sf(chi2, dof)`,
    tc: "O(r c)", tw: "Each cell is visited a constant number of times.", sc: "O(r + c)", sw: "Just the totals.",
    bad: "Correct and easy to follow, but the double loop is Python-slow for large tables and hides the structure: the expected counts are simply an outer product.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Row totals as a column and column totals as a row broadcast into the full matrix of expected counts in one expression, so the whole statistic is two array operations.",
    steps: ["Convert to a float array `obs`.", "`expected = row_totals * col_totals / grand_total`, using `keepdims=True` so a `(r,1)` times `(1,c)` broadcasts to `(r,c)`.", "Sum `(obs - expected)**2 / expected`.", "Look up the upper tail of the chi-square distribution with `stats.chi2.sf`."],
    code: R`
import numpy as np
from scipy import stats

def chi_square_test(table):
    obs = np.asarray(table, dtype=float)
    expected = obs.sum(axis=1, keepdims=True) * obs.sum(axis=0, keepdims=True) / obs.sum()   # (r,1) * (1,c) -> (r,c)
    chi2 = ((obs - expected) ** 2 / expected).sum()
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    return chi2, dof, stats.chi2.sf(chi2, dof)`,
    tc: "O(r c)", tw: "A few element-wise operations over the table.", sc: "O(r c)", sw: "The expected-count matrix.",
  },
  run: R`
c, d, p = chi_square_test([[120, 880], [150, 850]])
print(round(c, 3), d, round(p, 4))
c, d, p = chi_square_test([[20, 30, 50], [30, 30, 40]])
print(round(c, 3), d, round(p, 4))
t = [[10, 20, 30], [6, 9, 17], [4, 11, 2]]
print(np.isclose(chi_square_test(t)[0], stats.chi2_contingency(t, correction=False)[0]))
print(round(chi_square_test([[50, 50], [50, 50]])[0], 6))`,
  out: R`
3.854 1 0.0496
3.111 2 0.2111
True
0.0`,
  dryOn: "table = [[120, 880], [150, 850]]",
  dry: { h: ["step", "value"], r: [["row totals", "[1000, 1000]"], ["column totals", "[270, 1730]"], ["grand total", "2000"], ["expected", "[[135, 865], [135, 865]]"], ["(O - E)^2 / E", "[[1.667, 0.260], [1.667, 0.260]]"], ["chi2, dof", "**3.854, 1**"], ["p-value", "**0.0496**"]] },
  expl: [
    ["obs.sum(axis=1, keepdims=True)", "Row totals as a column vector of shape (r, 1)."],
    ["obs.sum(axis=0, keepdims=True)", "Column totals as a row vector of shape (1, c)."],
    ["... / obs.sum()", "Dividing by the grand total turns the outer product into the expected counts under independence."],
    ["((obs - expected) ** 2 / expected).sum()", "Add up the squared surprise of every cell, scaled by how many were expected there."],
    ["stats.chi2.sf(chi2, dof)", "The survival function is the upper-tail probability, i.e. the p-value."],
  ],
  take: "**Expected counts are an outer product of the margins.** If the groups behaved identically, each cell would hold its row's share of its column's total. The statistic then measures how far reality is from that picture.",
  mist: ["Feeding in percentages or rates instead of raw counts (the test needs counts).", "Using it when expected counts are tiny (below about 5); use Fisher's exact test instead.", "Forgetting that a significant result only says 'not independent', not how strong the effect is.", "Counting one visitor many times (clicks per page view) so the observations are not independent."],
},
{
  id: "welch-t-test", t: "Two-Sample Welch t-Test", d: "Medium", p: "Comparing means",
  q: "Two independent groups (say the response times of two model versions) have different sizes and different spreads. Test whether their means differ without assuming equal variances.\n\nWrite `welch_t(a, b)` returning `(t, df, p)`:\n\n$t = \\dfrac{\\bar a - \\bar b}{\\sqrt{s_a^2/n_a + s_b^2/n_b}}$\n\n$\\text{df} = \\dfrac{(s_a^2/n_a + s_b^2/n_b)^2}{\\dfrac{(s_a^2/n_a)^2}{n_a - 1} + \\dfrac{(s_b^2/n_b)^2}{n_b - 1}}$ (Welch-Satterthwaite)\n\nThe two-sided p-value is $2 \\, P(T_{\\text{df}} \\ge |t|)$. Use the **sample** variance ($n-1$). You may call `scipy.stats.t.sf` for the tail probability but not `ttest_ind`.",
  ex: [["a = [12.1, 13.4, 11.8, 14.2, 12.9], b = [10.2, 11.1, 12.0, 10.9, 11.5, 10.4]", "t = 3.62, df = 6.95, p = 0.0086", "A has the higher mean, and the spread is small enough to call it significant."],
    ["a identical to b", "t = 0, p = 1", "No difference at all."]],
  c: ["Each group has at least 2 values", "Return Python/NumPy floats; the two groups may have different lengths"],
  hint: "Work with the variance **of each mean**, `v = s^2 / n`, for both groups. The standard error is `sqrt(va + vb)` and the degrees of freedom formula is built from the same two numbers. `a.var(ddof=1)` is the sample variance.",
  brute: {
    name: "Plain Python loops",
    idea: "Compute means and sample variances with explicit sums, then apply the formulas.",
    code: R`
import numpy as np
from scipy import stats

def welch_t(a, b):
    def mean_var(x):
        n = len(x)
        m = sum(x) / n
        v = sum((xi - m) ** 2 for xi in x) / (n - 1)
        return n, m, v
    na, ma, va = mean_var(a)
    nb, mb, vb = mean_var(b)
    se2 = va / na + vb / nb
    t = (ma - mb) / se2 ** 0.5
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df, 2 * stats.t.sf(abs(t), df)`,
    tc: "O(n)", tw: "A couple of passes over each group.", sc: "O(1)", sw: "A few scalars.",
    bad: "Fine for small inputs; the per-element Python loops are slower than NumPy's reductions, and re-implementing mean and variance invites small bugs.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Let `mean` and `var(ddof=1)` do the sums, then use the variance of each group's mean as the one building block of both the statistic and the degrees of freedom.",
    steps: ["`va = a.var(ddof=1) / len(a)` and the same for `b` (the variance of each sample mean).", "`t = (a.mean() - b.mean()) / sqrt(va + vb)`.", "Degrees of freedom: `(va + vb)**2 / (va**2/(na-1) + vb**2/(nb-1))`.", "Two-sided p from the t distribution's tail: `2 * stats.t.sf(abs(t), df)`."],
    code: R`
import numpy as np
from scipy import stats

def welch_t(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)          # variance of each sample mean
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))      # Welch-Satterthwaite
    return t, df, 2 * stats.t.sf(abs(t), df)`,
    tc: "O(n)", tw: "Mean and variance of each group.", sc: "O(1)", sw: "A few scalars.",
  },
  run: R`
a = [12.1, 13.4, 11.8, 14.2, 12.9]
b = [10.2, 11.1, 12.0, 10.9, 11.5, 10.4]
t, df, p = welch_t(a, b)
print(round(t, 3), round(df, 2), round(p, 4))
r = stats.ttest_ind(a, b, equal_var=False)
print(np.allclose([t, p], [r.statistic, r.pvalue]))
print([round(float(v), 3) for v in welch_t(a, a)])
x, y = np.arange(10.0), np.arange(10.0) + 3
print([round(float(v), 4) for v in welch_t(x, y)])`,
  out: R`
3.62 6.95 0.0086
True
[0.0, 8.0, 1.0]
[-2.2156, 18.0, 0.0398]`,
  dryOn: "a = [12.1, 13.4, 11.8, 14.2, 12.9], b = [10.2, 11.1, 12.0, 10.9, 11.5, 10.4]",
  dry: { h: ["quantity", "group a", "group b"], r: [["n", "5", "6"], ["mean", "12.88", "11.02"], ["sample variance", "0.947", "0.454"], ["variance of the mean s^2/n", "0.189", "0.076"], ["t = diff / sqrt(sum)", "**(12.88 - 11.02) / sqrt(0.265) = 3.62**", ""], ["df", "**about 6.95**", ""], ["p (two-sided)", "**0.0086**", ""]] },
  expl: [
    ["va = a.var(ddof=1) / len(a)", "Sample variance divided by n: how much the group's mean wobbles."],
    ["t = (a.mean() - b.mean()) / np.sqrt(va + vb)", "The difference in means in units of its standard error (the two wobbles add as variances)."],
    ["df = (va + vb) ** 2 / (...)", "Welch-Satterthwaite: an effective sample size between min(n)-1 and na+nb-2, smaller when the variances are unequal."],
    ["2 * stats.t.sf(abs(t), df)", "Both tails of the t distribution beyond |t|: the two-sided p-value."],
  ],
  take: "**Do not pool the variances.** Welch's test builds the standard error from each group's own `s^2/n` and adjusts the degrees of freedom, so it stays valid with unequal sizes and spreads, and is the safe default.",
  mist: ["Using `ddof=0` (NumPy's default) instead of `ddof=1` for the variance.", "Using the pooled-variance Student formula when the groups clearly differ in spread.", "Using a paired situation (same users measured twice) with an unpaired test, which loses power.", "Reading a large p-value as 'no difference' when the groups were tiny."],
},
{
  id: "bootstrap-ci-seeded", t: "Bootstrap Confidence Interval", d: "Medium", p: "Resampling",
  q: "There is no tidy formula for the uncertainty of a median. Estimate it by **bootstrapping**.\n\nWrite `bootstrap_ci(x, stat, n_boot, alpha, seed)`:\n\n1. Create `rng = np.random.default_rng(seed)`.\n2. Do `n_boot` times: draw `len(x)` values from `x` **with replacement** (draw the indices with `rng.integers(0, len(x), len(x))`) and compute `stat` on the resample.\n3. Return the `alpha/2` and `1 - alpha/2` percentiles (as percentages: `100*alpha/2`, `100*(1-alpha/2)`) of the bootstrap statistics as a tuple `(low, high)`.\n\n`stat` is a function such as `np.mean` or `np.median` that accepts `axis=`.",
  ex: [["x = [3, 5, 4, 8, 6, 7, 9, 4, 5, 6], stat = np.mean", "[4.7, 6.8]", "The sample mean is 5.7; resampling shows how far it typically moves."],
    ["same x, stat = np.median", "[4.0, 7.0]", "Medians of small samples take few distinct values."]],
  c: ["`n_boot` up to a few thousand", "Same seed gives the same interval; draw the indices exactly as described so results match"],
  hint: "Make **all** the resamples at once: `rng.integers(0, n, size=(n_boot, n))` is a matrix of indices, `x[idx]` a matrix of resamples (one per row), and `stat(..., axis=1)` gives one statistic per row.",
  brute: {
    name: "Plain Python loop over resamples",
    idea: "Resample, compute, append; repeat n_boot times; then take percentiles.",
    code: R`
import numpy as np

def bootstrap_ci(x, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        boots.append(stat(x[idx]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)`,
    tc: "O(B n)", tw: "B resamples of n values, one at a time.", sc: "O(B)", sw: "The list of bootstrap statistics.",
    bad: "Correct, but it makes thousands of tiny NumPy calls from a Python loop. The vectorized version builds all resamples in a single array operation.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Draw a `(n_boot, n)` matrix of random indices, gather the data with fancy indexing, and reduce along axis 1 to get every bootstrap statistic at once.",
    steps: ["`idx = rng.integers(0, n, size=(n_boot, n))`: row b holds the indices of resample b.", "`x[idx]` gives the `(n_boot, n)` matrix of resampled data.", "`stat(x[idx], axis=1)` reduces each row to one number.", "`np.percentile` of those numbers gives the interval."],
    code: R`
import numpy as np

def bootstrap_ci(x, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))            # all resamples at once
    boots = stat(x[idx], axis=1)                                     # one statistic per resample
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)`,
    tc: "O(B n)", tw: "Same work, done in compiled code.", sc: "O(B n)", sw: "The index and resample matrices.",
  },
  run: R`
x = [3, 5, 4, 8, 6, 7, 9, 4, 5, 6]
print([round(v, 2) for v in bootstrap_ci(x, np.mean)])
print([round(v, 2) for v in bootstrap_ci(x, np.median)])
print([round(v, 2) for v in bootstrap_ci(x, np.mean, alpha=0.2)])
print(bootstrap_ci(x, np.mean, seed=1) == bootstrap_ci(x, np.mean, seed=1))
print([round(v, 2) for v in bootstrap_ci([5.0] * 8, np.mean)])`,
  out: R`
[4.7, 6.8]
[4.0, 7.0]
[5.0, 6.4]
True
[5.0, 5.0]`,
  dryOn: "x = [3, 5, 4], stat = np.mean, n_boot = 4 (tiny, with made-up random draws, to see the idea)",
  dry: { h: ["resample", "indices drawn", "values", "mean"], r: [["1", "[1, 1, 2]", "[5, 5, 4]", "4.67"], ["2", "[0, 2, 2]", "[3, 4, 4]", "3.67"], ["3", "[1, 0, 1]", "[5, 3, 5]", "4.33"], ["4", "[2, 2, 2]", "[4, 4, 4]", "4.00"], ["interval", "percentiles of the four means", "", "**the middle (1 - alpha) of them**"]] },
  expl: [
    ["rng = np.random.default_rng(seed)", "A seeded generator makes the answer reproducible."],
    ["idx = rng.integers(0, len(x), size=(n_boot, len(x)))", "Each row is one resample: n random positions, repeats allowed."],
    ["boots = stat(x[idx], axis=1)", "Fancy indexing builds a matrix of resamples; the statistic is applied to every row."],
    ["np.percentile(boots, [...])", "The middle (1 - alpha) of the bootstrap statistics is the percentile interval."],
  ],
  take: "**Resample with replacement, recompute, take percentiles.** Vectorize by building one `(n_boot, n)` index matrix; a 5-line function replaces a pile of algebra and works for any statistic.",
  mist: ["Sampling without replacement (every resample would equal the original).", "Resampling to a size other than n.", "Forgetting `axis=1`, which reduces the whole matrix to a single number.", "Bootstrapping non-independent rows (resample users, not rows, when a user has many rows)."],
},
{
  id: "benjamini-hochberg-fdr", t: "Benjamini-Hochberg FDR Adjustment", d: "Medium", p: "Multiple testing",
  q: "You ran $m$ hypothesis tests and got $m$ p-values. Controlling the **false discovery rate** at level $q$ means: sort the p-values $p_{(1)} \\le \\dots \\le p_{(m)}$ and find the largest $i$ with $p_{(i)} \\le \\dfrac{i}{m}\\, q$; then reject the $i$ smallest.\n\nThe equivalent **adjusted p-value** (q-value) of the $i$-th sorted p-value is\n\n$\\tilde p_{(i)} = \\min_{j \\ge i} \\left( \\min\\left(1, \\dfrac{m}{j}\\, p_{(j)}\\right) \\right)$\n\nand you reject test $k$ when its adjusted value is at most $q$.\n\nWrite `bh_adjust(p)` returning the adjusted p-values **in the original order**, and `bh_reject(p, q)` returning a boolean array.",
  ex: [["p = [0.001, 0.009, 0.020, 0.041, 0.045, 0.60]", "adjusted [0.006, 0.027, 0.04, 0.054, 0.054, 0.6]", "At q = 0.05 the first three are rejected."],
    ["p = [0.6, 0.02, 0.001]", "[0.6, 0.03, 0.003]", "Unsorted input must come back aligned with its original positions."]],
  c: ["`p` is a 1-D array of values in [0, 1]", "Cap adjusted values at 1; keep the original order"],
  hint: "Sort, multiply each sorted p by `m / rank`, then take a **running minimum from the largest rank downwards** (`np.minimum.accumulate` on the reversed array) so adjusted values never decrease with p. Finally undo the sort.",
  brute: {
    name: "Plain Python loops",
    idea: "Sort with indices, then walk from the largest p down to the smallest, keeping a running minimum of `m/rank * p`.",
    code: R`
import numpy as np

def bh_adjust(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):                       # from the largest p downwards
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        adj[i] = running
    return np.array(adj)

def bh_reject(p, q=0.05):
    return bh_adjust(p) <= q`,
    tc: "O(m log m)", tw: "Dominated by the sort.", sc: "O(m)", sw: "Order list and result.",
    bad: "Correct. The loop is clear, but vectorizing makes the running minimum a single call and is shorter to verify.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Scale the sorted p-values by `m / rank`, take a reversed cumulative minimum, cap at 1, and scatter the result back to the original positions.",
    steps: ["`order = np.argsort(p)` and `ps = p[order]`.", "`scaled = ps * m / np.arange(1, m + 1)`.", "Cumulative minimum from the right: `np.minimum.accumulate(scaled[::-1])[::-1]`.", "Cap at 1 and put back with `adj[order] = ...`."],
    code: R`
import numpy as np

def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)                      # p_(i) * m / i
    adj_sorted = np.minimum.accumulate(scaled[::-1])[::-1]           # enforce monotonicity from the top
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)                         # undo the sort, cap at 1
    return adj

def bh_reject(p, q=0.05):
    return bh_adjust(p) <= q`,
    tc: "O(m log m)", tw: "One sort plus linear array operations.", sc: "O(m)", sw: "A few arrays.",
  },
  run: R`
p = [0.001, 0.009, 0.020, 0.041, 0.045, 0.60]
print(np.round(bh_adjust(p), 4).tolist())
print(bh_reject(p, 0.05).tolist())
print(np.round(bh_adjust([0.6, 0.02, 0.001]), 4).tolist())
rng = np.random.default_rng(0)
big = rng.uniform(size=50) ** 3
from scipy import stats
print(np.allclose(bh_adjust(big), stats.false_discovery_control(big)))
print(int(bh_reject(big, 0.05).sum()), int((big < 0.05).sum()))`,
  out: R`
[0.006, 0.027, 0.04, 0.054, 0.054, 0.6]
[True, True, True, False, False, False]
[0.6, 0.03, 0.003]
True
9 16`,
  dryOn: "p = [0.001, 0.009, 0.020, 0.041, 0.045, 0.60], m = 6",
  dry: { h: ["sorted p", "rank i", "p * m / i", "running min from the right (adjusted)"], r: [["0.001", "1", "0.006", "0.006"], ["0.009", "2", "0.027", "0.027"], ["0.020", "3", "0.040", "0.040"], ["0.041", "4", "0.0615", "0.054"], ["0.045", "5", "0.054", "0.054"], ["0.60", "6", "0.60", "0.60"]] },
  expl: [
    ["order = np.argsort(p)", "Indices that sort the p-values; we need them to restore the original order."],
    ["scaled = p[order] * m / np.arange(1, m + 1)", "The BH threshold for rank i is i/m * q, so p <= i/m * q is the same as p * m / i <= q."],
    ["np.minimum.accumulate(scaled[::-1])[::-1]", "A reversed running minimum makes the adjusted values non-decreasing in p, matching the 'largest i that passes' rule."],
    ["adj[order] = np.minimum(adj_sorted, 1.0)", "Scatter back to the original positions; probabilities cannot exceed 1."],
  ],
  take: "**BH compares the i-th smallest p-value with i/m * q.** Adjusted values are the scaled p-values made monotone by a running minimum from the largest rank down; reject everything with an adjusted value at most q.",
  mist: ["Forgetting the running minimum, which can leave adjusted values out of order.", "Returning the adjusted values in sorted order instead of the original order.", "Using the BH procedure on correlated tests without knowing it can be anti-conservative in rare cases (use Benjamini-Yekutieli there).", "Not capping at 1."],
},
{
  id: "exact-permutation-test", t: "Exact Permutation Test", d: "Hard", p: "Resampling",
  q: "With two tiny groups you can run a permutation test **exactly**: list every possible way to split the pooled values into a group of size $n_1$ and a group of size $n_2$, and see in how many splits the difference in means is at least as extreme as the one observed.\n\nWrite `exact_perm_test(a, b)` returning the **two-sided p-value**: the fraction of all $\\binom{N}{n_1}$ splits (where $N = n_1 + n_2$) with $|\\bar x_1 - \\bar x_2| \\ge |\\bar a - \\bar b|$. Compare with a tolerance of `1e-9` to avoid floating-point trouble. No random numbers needed.",
  ex: [["a = [10, 11, 12], b = [1, 2, 3]", "p = 0.1", "20 splits; only the real split and its mirror image are this extreme: 2/20."],
    ["a = [1, 2, 3], b = [1, 2, 3]", "p = 1.0", "The observed difference is 0, so every split is at least as extreme."]],
  c: ["`len(a) + len(b)` is at most 16 (so at most 12,870 splits)", "Return a Python float"],
  hint: "`itertools.combinations(range(N), n1)` yields every choice of which pooled positions form group 1. Stacking them into an index matrix lets NumPy compute all the group means with one fancy-index and one `sum(axis=1)`. Group 2's sum is the pooled total minus group 1's.",
  brute: {
    name: "Plain Python loops over splits",
    idea: "For each combination, form the two groups as lists, compute the difference of means, and count.",
    code: R`
import numpy as np
from itertools import combinations

def exact_perm_test(a, b):
    pooled = list(a) + list(b)
    n1, N = len(a), len(a) + len(b)
    obs = abs(sum(a) / len(a) - sum(b) / len(b))
    count = total = 0
    for chosen in combinations(range(N), n1):
        g1 = [pooled[i] for i in chosen]
        g2 = [pooled[i] for i in range(N) if i not in chosen]
        d = abs(sum(g1) / n1 - sum(g2) / (N - n1))
        count += d >= obs - 1e-9
        total += 1
    return count / total`,
    tc: "O(C(N, n1) N)", tw: "Every split, each rebuilt from scratch.", sc: "O(N)", sw: "One split at a time.",
    bad: "Correct, but rebuilding both groups for every split in Python is slow even for N = 14. The vectorized version computes all the group sums in one step.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Build the matrix of all index combinations once, gather the pooled values with it, and compute every difference in means with row sums.",
    steps: ["`idx = np.array(list(combinations(range(N), n1)))` has one row per split.", "`s1 = pooled[idx].sum(axis=1)` is the sum of group 1 in every split; group 2's sum is `pooled.sum() - s1`.", "Differences of means: `s1 / n1 - (total - s1) / (N - n1)`.", "p = share of splits with `abs(diff) >= abs(observed) - 1e-9`."],
    code: R`
import numpy as np
from itertools import combinations

def exact_perm_test(a, b):
    pooled = np.concatenate([a, b]).astype(float)
    n1, N = len(a), len(a) + len(b)
    idx = np.array(list(combinations(range(N), n1)))             # every way to pick the 'group a' positions
    s1 = pooled[idx].sum(axis=1)                                 # group-1 sum in each split
    diff = s1 / n1 - (pooled.sum() - s1) / (N - n1)              # difference in means in each split
    obs = np.mean(a) - np.mean(b)
    return float(np.mean(np.abs(diff) >= abs(obs) - 1e-9))`,
    tc: "O(C(N, n1) n1)", tw: "One gather and one row sum over all splits.", sc: "O(C(N, n1) n1)", sw: "The combination index matrix.",
  },
  run: R`
print(exact_perm_test([10, 11, 12], [1, 2, 3]))
print(exact_perm_test([1, 2, 3], [1, 2, 3]))
a = [12.1, 13.4, 11.8, 14.2, 12.9]
b = [10.2, 11.1, 12.0, 10.9, 11.5]
print(round(exact_perm_test(a, b), 4))
print(round(exact_perm_test(b, a), 4))
print(round(exact_perm_test([5, 6], [1, 2, 9]), 4))`,
  out: R`
0.1
1.0
0.0159
0.0159
0.8`,
  dryOn: "a = [10, 11, 12], b = [1, 2, 3]",
  dry: { h: ["step", "value"], r: [["pooled", "[10, 11, 12, 1, 2, 3], N = 6, n1 = 3"], ["number of splits", "C(6, 3) = 20"], ["observed diff", "11 - 2 = 9"], ["split {10,11,12} vs {1,2,3}", "diff 9 (as extreme)"], ["split {1,2,3} vs {10,11,12}", "diff -9 (as extreme, two-sided)"], ["the other 18 splits", "|diff| below 9"], ["p", "**2 / 20 = 0.1**"]] },
  expl: [
    ["combinations(range(N), n1)", "All ways to choose which pooled positions belong to the first group."],
    ["pooled[idx].sum(axis=1)", "Fancy indexing gathers one row of values per split; the row sum is group 1's total."],
    ["pooled.sum() - s1", "Group 2's total comes for free from the pooled total."],
    ["np.abs(diff) >= abs(obs) - 1e-9", "Two-sided: count splits at least as extreme; the tiny tolerance protects against floating-point ties."],
  ],
  take: "**A permutation p-value is just 'how many relabelings look this extreme'.** For tiny samples you can count them all exactly; for larger ones you sample random relabelings instead, using the same logic.",
  mist: ["Comparing floats with `>=` and no tolerance, which drops tied splits.", "Counting only one tail when the question is two-sided.", "Forgetting that the observed split is itself one of the possible splits (and so p can never be 0).", "Trying this for N = 40: the number of splits explodes, so switch to random shuffles."],
},
]);
