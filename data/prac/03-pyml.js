/* Practice: NumPy Basics (Python for ML companions): broadcasting, axis reductions, vectorized windows.
   Each problem: q, ex, hint, brute (plain Python), opt (vectorized NumPy), run + out, dry, expl, take, mist. */
PRAC.add("NumPy Basics", [
{
  id: "pairwise-distance-broadcast", t: "Pairwise Euclidean Distances", d: "Medium", p: "Broadcasting",
  q: "k-NN, k-means and clustering all start from the same table: the distance between **every** point of one set and **every** point of another.\n\nWrite `pairwise_dist(A, B)` for `A` of shape `(n, d)` and `B` of shape `(m, d)` that returns an `(n, m)` array where entry `[i, j]` is the Euclidean distance between row `i` of `A` and row `j` of `B`.\n\n$D_{ij} = \\sqrt{\\sum_k (A_{ik} - B_{jk})^2}$\n\nNo Python loops in the solution: use broadcasting.",
  ex: [["A = [[0,0],[3,4]], B = [[0,0],[6,8],[3,0]]", "[[0, 10, 3], [5, 5, 4]]", "Row 1 of A is (3,4): it is 5 from (0,0), 5 from (6,8) and 4 from (3,0)."],
    ["A = B = [[1,2]]", "[[0]]", "A point is at distance 0 from itself."]],
  c: ["`A` has shape `(n, d)`, `B` has shape `(m, d)`, both float", "Return a float array of shape `(n, m)`"],
  hint: "You want a difference for every pair `(i, j)`. If `A` had shape `(n, 1, d)` and `B` had shape `(1, m, d)`, what would `A - B` look like? Which axis do you sum over?",
  brute: {
    name: "Plain Python loops",
    idea: "Three nested loops: for every pair of points, sum the squared coordinate differences and take the square root.",
    code: R`
import numpy as np

def pairwise_dist(A, B):
    n, d = A.shape
    m = B.shape[0]
    D = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(d):
                s += (A[i, k] - B[j, k]) ** 2
            D[i, j] = s ** 0.5
    return D`,
    tc: "O(n m d)", tw: "One subtraction per pair per coordinate.", sc: "O(n m)", sw: "The output.",
    bad: "Correct but slow: n m d steps executed in the Python interpreter. With thousands of points it takes minutes.",
  },
  opt: {
    name: "Vectorized with broadcasting",
    idea: "Insert a new axis so `A` becomes `(n, 1, d)` and `B` becomes `(1, m, d)`. Subtracting broadcasts to `(n, m, d)`: the difference vector of every pair. Square, sum over the last axis, take the square root.",
    steps: ["`A[:, None, :]` has shape `(n, 1, d)`; `B[None, :, :]` has shape `(1, m, d)`.", "`diff = A[:, None, :] - B[None, :, :]` has shape `(n, m, d)`.", "`(diff ** 2).sum(axis=-1)` collapses the coordinates to `(n, m)`.", "`np.sqrt` of that is the distance table."],
    code: R`
import numpy as np

def pairwise_dist(A, B):
    diff = A[:, None, :] - B[None, :, :]          # (n,1,d) - (1,m,d) -> (n,m,d)
    return np.sqrt((diff ** 2).sum(axis=-1))      # collapse the coordinate axis -> (n,m)`,
    tc: "O(n m d)", tw: "The same number of arithmetic operations, but in compiled code.", sc: "O(n m d)", sw: "The intermediate `diff` array holds n m d numbers.",
  },
  better: {
    name: "Matrix-multiply trick (no 3-D array)",
    idea: "Expand the square: $\\|a-b\\|^2 = \\|a\\|^2 - 2\\,a\\cdot b + \\|b\\|^2$. The cross term for all pairs is one matrix product `A @ B.T`. This needs only `(n, m)` memory. Rounding can make tiny negative values, so clip at 0 before the square root.",
    steps: ["Squared norms: `a2 = (A**2).sum(axis=1)[:, None]` and `b2 = (B**2).sum(axis=1)[None, :]`.", "Cross terms: `A @ B.T` has shape `(n, m)`.", "`d2 = a2 - 2 * (A @ B.T) + b2`, then `np.maximum(d2, 0)`, then `sqrt`."],
    code: R`
import numpy as np

def pairwise_dist(A, B):
    a2 = (A ** 2).sum(axis=1)[:, None]            # (n,1)
    b2 = (B ** 2).sum(axis=1)[None, :]            # (1,m)
    d2 = a2 - 2 * (A @ B.T) + b2                  # (n,m), by broadcasting
    return np.sqrt(np.maximum(d2, 0.0))           # clip tiny negatives from rounding`,
    tc: "O(n m d)", tw: "Dominated by one matrix multiply, which is very well optimised.", sc: "O(n m)", sw: "Only the (n, m) tables; no 3-D array.",
  },
  run: R`
A = np.array([[0.0, 0.0], [3.0, 4.0]])
B = np.array([[0.0, 0.0], [6.0, 8.0], [3.0, 0.0]])
print(np.round(pairwise_dist(A, B), 4).tolist())
print(np.round(pairwise_dist(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]])), 4).tolist())
rng = np.random.default_rng(0)
P = rng.normal(size=(6, 3))
D = pairwise_dist(P, P)
print(D.shape, np.round(np.diag(D), 4).tolist())
print(bool(np.allclose(D, D.T)))
Q = rng.normal(size=(4, 3))
print(pairwise_dist(P, Q).shape, round(float(pairwise_dist(P, Q)[2, 1]), 4) == round(float(np.linalg.norm(P[2] - Q[1])), 4))`,
  out: R`
[[0.0, 10.0, 3.0], [5.0, 5.0, 4.0]]
[[0.0]]
(6, 6) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
True
(6, 4) True`,
  dryOn: "A = [[0,0],[3,4]], B = [[0,0],[6,8],[3,0]]",
  dry: { h: ["step", "shape", "value"], r: [["A[:, None, :]", "(2, 1, 2)", "[[[0,0]], [[3,4]]]"], ["B[None, :, :]", "(1, 3, 2)", "[[[0,0],[6,8],[3,0]]]"], ["A - B", "(2, 3, 2)", "pair differences, e.g. (3,4)-(6,8) = (-3,-4)"], ["squared, summed over last axis", "(2, 3)", "[[0, 100, 9], [25, 25, 16]]"], ["sqrt", "(2, 3)", "**[[0, 10, 3], [5, 5, 4]]**"]] },
  expl: [
    ["A[:, None, :] - B[None, :, :]", "`None` inserts a size-1 axis. Shapes (n,1,d) and (1,m,d) broadcast to (n,m,d): entry [i,j] is A[i] - B[j]."],
    ["(diff ** 2).sum(axis=-1)", "Square every coordinate difference and add them up over the last axis (the coordinates), leaving an (n, m) table."],
    ["np.sqrt(...)", "The square root of the squared distance is the distance."],
  ],
  take: "**To compare every row of one array with every row of another, add an axis to each side (`A[:, None]`, `B[None, :]`) and let broadcasting build the pairs.** For big data, use the `A @ B.T` expansion to avoid the 3-D intermediate.",
  mist: ["Forgetting the new axes: `A - B` with shapes `(n, d)` and `(m, d)` fails, or silently pairs row i with row i when `n == m`.", "Summing over the wrong axis (`axis=0` or `1` instead of the last).", "In the matmul trick, skipping `np.maximum(d2, 0)` and getting `nan` from the square root of a tiny negative number."],
},
{
  id: "minmax-zscore-axis", t: "Scale Along an Axis: Min-Max and Z-Score", d: "Easy", p: "Axes & broadcasting",
  q: "Write `scale(X, mode, axis)` for a 2-D float array `X`:\n\n- `mode=\"minmax\"`: rescale to the range `[0, 1]`: $(x - \\min) / (\\max - \\min)$.\n- `mode=\"zscore\"`: subtract the mean and divide by the (population) standard deviation.\n\n`axis=0` scales each **column** using the statistics of that column; `axis=1` scales each **row**. If the spread is zero (a constant column or row) return zeros for it instead of dividing by zero.",
  ex: [["X = [[1,10],[2,20],[3,40]], \"minmax\", axis=0", "[[0, 0], [0.5, 0.3333], [1, 1]]", "Column 0 runs 1..3, column 1 runs 10..40."],
    ["X = [[1,10],[2,20],[3,40]], \"minmax\", axis=1", "[[0, 1], [0, 1], [0, 1]]", "Each row is scaled on its own: its smallest becomes 0 and its largest 1."]],
  c: ["`X` is a float array of shape `(n, d)`", "`mode` is `\"minmax\"` or `\"zscore\"`; `axis` is 0 or 1", "Return a new array; do not modify `X`"],
  hint: "Reductions accept `axis` and `keepdims=True`. With `keepdims=True` the statistics have shape `(1, d)` or `(n, 1)`, which broadcasts back against `X` for either axis.",
  brute: {
    name: "Plain Python loops",
    idea: "Pull out each column (or row) as a list, compute its statistics with Python, and write the scaled values back one by one.",
    code: R`
import numpy as np

def scale(X, mode="minmax", axis=0):
    X = np.asarray(X, dtype=float)
    Y = X if axis == 0 else X.T                  # work on columns of Y
    out = np.zeros(Y.shape)
    for j in range(Y.shape[1]):
        col = [Y[i, j] for i in range(Y.shape[0])]
        n = len(col)
        if mode == "minmax":
            lo, span = min(col), max(col) - min(col)
        else:
            lo = sum(col) / n
            span = (sum((v - lo) ** 2 for v in col) / n) ** 0.5
        for i in range(n):
            out[i, j] = 0.0 if span == 0 else (col[i] - lo) / span
    return out if axis == 0 else out.T`,
    tc: "O(n d)", tw: "A few passes over every entry.", sc: "O(n d)", sw: "The output.",
    bad: "Works but the loops are in Python, and the axis handling with transposes is easy to get wrong.",
  },
  opt: {
    name: "Vectorized with axis and keepdims",
    idea: "Compute the two statistics along `axis` with `keepdims=True` so they keep a size-1 axis. Broadcasting then lines them up with `X` whichever axis you chose. Replace a zero spread by 1 so constant slices become 0.",
    steps: ["For min-max: `lo = X.min(axis, keepdims=True)`, `span = X.max(...) - lo`. For z-score: `lo = X.mean(...)`, `span = X.std(...)`.", "Replace zeros in `span` by 1 with `np.where`.", "Return `(X - lo) / span`."],
    code: R`
import numpy as np

def scale(X, mode="minmax", axis=0):
    X = np.asarray(X, dtype=float)
    if mode == "minmax":
        lo = X.min(axis=axis, keepdims=True)
        span = X.max(axis=axis, keepdims=True) - lo
    else:
        lo = X.mean(axis=axis, keepdims=True)
        span = X.std(axis=axis, keepdims=True)
    span = np.where(span == 0, 1.0, span)          # constant slice: avoid 0/0
    return (X - lo) / span                         # keepdims makes this broadcast for both axes`,
    tc: "O(n d)", tw: "Two reductions and two element-wise operations.", sc: "O(n d)", sw: "The output.",
  },
  run: R`
X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 40.0]])
print(np.round(scale(X, "minmax", 0), 4).tolist())
print(np.round(scale(X, "minmax", 1), 4).tolist())
print(np.round(scale(X, "zscore", 0), 4).tolist())
C = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
print(np.round(scale(C, "zscore", 0), 4).tolist())
rng = np.random.default_rng(0)
R2 = rng.normal(3, 2, size=(50, 4))
Z = scale(R2, "zscore", 1)
print(np.round(Z.mean(axis=1)[:3], 6).tolist() == [0.0, 0.0, 0.0], np.round(Z.std(axis=1)[:3], 4).tolist())
M = scale(R2, "minmax", 0)
print(np.round(M.min(axis=0), 4).tolist(), np.round(M.max(axis=0), 4).tolist())`,
  out: R`
[[0.0, 0.0], [0.5, 0.3333], [1.0, 1.0]]
[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
[[-1.2247, -1.069], [0.0, -0.2673], [1.2247, 1.3363]]
[[0.0, -1.2247], [0.0, 0.0], [0.0, 1.2247]]
True [1.0, 1.0, 1.0]
[0.0, 0.0, 0.0, 0.0] [1.0, 1.0, 1.0, 1.0]`,
  dryOn: "X = [[1,10],[2,20],[3,40]], \"minmax\", axis=0",
  dry: { h: ["step", "column 0", "column 1"], r: [["min", "1", "10"], ["max - min (span)", "2", "30"], ["X - min", "[0, 1, 2]", "[0, 10, 30]"], ["divide by span", "**[0, 0.5, 1]**", "**[0, 0.3333, 1]**"]] },
  expl: [
    ["X.min(axis=axis, keepdims=True)", "Smallest value along the chosen axis; `keepdims` keeps shape (1, d) or (n, 1) so it broadcasts against X."],
    ["span = np.where(span == 0, 1.0, span)", "A constant column has span 0; using 1 gives 0/1 = 0 instead of nan."],
    ["(X - lo) / span", "Broadcasting stretches lo and span across the collapsed axis."],
  ],
  take: "**Pair every reduction with `keepdims=True` when you plan to broadcast the result back.** Then one piece of code serves both `axis=0` (per column, per feature) and `axis=1` (per row).",
  mist: ["Confusing the axes: features are scaled with `axis=0` (down the rows), not `axis=1`.", "Dropping `keepdims=True`: for `axis=1` the `(n,)` statistics then broadcast against the wrong axis, or raise an error.", "Using `np.std` with ddof=1 by accident and getting results that differ from the standard z-score used here (ddof=0).", "Computing the statistics on test data: fit them on the training data only."],
},
{
  id: "moving-average-vectorized", t: "Moving Average Without a Loop", d: "Medium", p: "Vectorization",
  q: "A **moving average** smooths a noisy series (a loss curve, a sensor, a stock price): each output is the mean of `k` consecutive inputs.\n\nWrite `moving_average(x, k)` for a 1-D array `x` of length `n` that returns the `n - k + 1` averages of every window of length `k` (only full windows, no padding).\n\nThe vectorized solution must not loop over the windows.",
  ex: [["x = [1,2,3,4,5], k = 3", "[2, 3, 4]", "(1+2+3)/3, (2+3+4)/3, (3+4+5)/3."],
    ["x = [4, 8], k = 1", "[4, 8]", "Windows of length 1 are the numbers themselves."]],
  c: ["`x` is a 1-D float array, `1 <= k <= len(x)`", "Return an array of length `len(x) - k + 1`"],
  hint: "Summing each window from scratch repeats work. The running total `cumsum` already contains every window sum: the sum of x[i..i+k-1] is the difference of two running totals. A zero at the front of the cumulative sum makes the indexing tidy.",
  brute: {
    name: "Plain Python loops",
    idea: "For each starting position, slice out `k` numbers and average them.",
    code: R`
import numpy as np

def moving_average(x, k):
    out = []
    for i in range(len(x) - k + 1):
        window = x[i:i + k]
        out.append(sum(window) / k)
    return np.array(out)`,
    tc: "O(n k)", tw: "Each window sums k numbers again from scratch.", sc: "O(n)", sw: "The output.",
    bad: "Correct but re-adds k numbers for every window, in the Python interpreter.",
  },
  opt: {
    name: "Vectorized with cumsum",
    idea: "Let `c` be the running total with a leading 0, so `c[i]` is the sum of the first `i` numbers. The window `x[i:i+k]` sums to `c[i+k] - c[i]`. Subtracting two shifted views of `c` gives all the window sums at once.",
    steps: ["`c = np.concatenate(([0.0], np.cumsum(x)))` has length `n + 1`.", "`c[k:] - c[:-k]` has length `n - k + 1`: entry i is the sum of `x[i:i+k]`.", "Divide by `k`."],
    code: R`
import numpy as np

def moving_average(x, k):
    c = np.concatenate(([0.0], np.cumsum(x)))     # c[i] = sum of the first i numbers
    return (c[k:] - c[:-k]) / k                   # window sum = difference of running totals`,
    tc: "O(n)", tw: "One cumulative sum and one subtraction: independent of k.", sc: "O(n)", sw: "The running total and the output.",
  },
  better: {
    name: "np.convolve",
    idea: "A moving average is a convolution with a flat kernel of k equal weights `1/k`. `mode=\"valid\"` keeps only the full windows. (Same answer; easier to read, but O(n k).)",
    steps: ["Build the kernel `np.ones(k) / k`.", "`np.convolve(x, kernel, mode=\"valid\")` slides it along `x`."],
    code: R`
import numpy as np

def moving_average(x, k):
    return np.convolve(x, np.ones(k) / k, mode="valid")`,
    tc: "O(n k)", tw: "Compiled loop, but still k multiplications per output.", sc: "O(n)", sw: "The output.",
  },
  run: R`
print(np.round(moving_average(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3), 4).tolist())
print(np.round(moving_average(np.array([4.0, 8.0]), 1), 4).tolist())
x = np.array([2.0, 4.0, 6.0, 8.0])
print(np.round(moving_average(x, 4), 4).tolist())
rng = np.random.default_rng(0)
y = rng.normal(size=1000)
m = moving_average(y, 10)
print(m.shape, round(float(m[0]), 4) == round(float(y[:10].mean()), 4), round(float(m[-1]), 4) == round(float(y[-10:].mean()), 4))
print(bool(m.std() < y.std()))`,
  out: R`
[2.0, 3.0, 4.0]
[4.0, 8.0]
[5.0]
(991,) True True
True`,
  dryOn: "x = [1,2,3,4,5], k = 3",
  dry: { h: ["step", "value"], r: [["cumsum with a leading 0 (c)", "[0, 1, 3, 6, 10, 15]"], ["c[k:] = c[3:]", "[6, 10, 15]"], ["c[:-k] = c[:-3]", "[0, 1, 3]"], ["difference (window sums)", "[6, 9, 12]"], ["divide by k = 3", "**[2, 3, 4]**"]] },
  expl: [
    ["c = np.concatenate(([0.0], np.cumsum(x)))", "Running totals, with a 0 in front so that c[i] is the sum of the first i numbers."],
    ["c[k:] - c[:-k]", "Two views of c shifted by k. Entry i is c[i+k] - c[i], the sum of x[i], ..., x[i+k-1]."],
    ["/ k", "Window sum to window mean."],
  ],
  take: "**A sliding-window sum is a difference of two running totals.** `cumsum` turns an O(n k) loop into O(n) vectorized work. One caveat: with huge numbers the subtraction of large running totals can lose precision.",
  mist: ["Off-by-one: forgetting the leading 0, so the first window is dropped or the lengths do not match.", "Returning `n` outputs by padding without being asked: the task wants only full windows (`n - k + 1`).", "Using `x[i:i+k]` slices in a list comprehension and calling it vectorized: it is still a Python loop."],
},
]);
