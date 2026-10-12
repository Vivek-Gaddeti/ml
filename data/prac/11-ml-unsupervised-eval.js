/* Practice: Clustering & PCA + Evaluation (machine learning part 2)
   Each problem: q, ex, hint, brute (plain Python), opt (NumPy), run + out (tests), dry, expl, take, mist. */
PRAC.add("Clustering & PCA", [
{
  id: "kmeans-one-iteration", t: "One k-Means Iteration", d: "Easy", p: "k-means",
  q: "Write `kmeans_step(X, C)` that performs **one** iteration of k-means: assign every point to its nearest centroid, then move every centroid to the mean of the points assigned to it.\n\nReturn `(labels, new_C)`: `labels` is an integer array of length n (the index of the nearest centroid), and `new_C` has the same shape as `C`.\n\nUse squared Euclidean distance. Break ties by the lower centroid index. If a centroid receives **no** points, leave it where it is.",
  ex: [["X = [[0,0],[0,2],[10,0],[10,2]], C = [[0,1],[10,1]]", "labels [0, 0, 1, 1], new_C [[0, 1], [10, 1]]", "Each centroid already sits at the mean of its two points, so nothing moves: converged."],
    ["X = [[0],[1],[9]], C = [[0],[1],[100]]", "labels [0, 1, 1], new_C [[0], [5], [100]]", "Point 9 is closest to centroid 1. The third centroid got no points, so it stays at 100."]],
  c: ["`X` has shape (n, d), `C` has shape (k, d), k ≥ 1", "Both are float arrays"],
  hint: "Broadcasting gives all the distances at once: `X[:, None, :] - C[None, :, :]` has shape (n, k, d). Sum the squares over the last axis, then `argmin(axis=1)`. Then loop over the k clusters with a boolean mask.",
  brute: {
    name: "Plain Python loops",
    idea: "For every point, loop over all centroids to find the nearest one. Then, for every centroid, collect its points and average them coordinate by coordinate.",
    code: R`
import numpy as np

def kmeans_step(X, C):
    n, k, d = len(X), len(C), len(X[0])
    labels = []
    for i in range(n):
        best, best_d = 0, float("inf")
        for j in range(k):
            dist = sum((X[i][t] - C[j][t]) ** 2 for t in range(d))
            if dist < best_d:                      # strict '<' keeps the lower index on ties
                best, best_d = j, dist
        labels.append(best)
    new_C = []
    for j in range(k):
        members = [X[i] for i in range(n) if labels[i] == j]
        if members:
            new_C.append([sum(m[t] for m in members) / len(members) for t in range(d)])
        else:
            new_C.append([float(v) for v in C[j]])
    return np.array(labels), np.array(new_C)`,
    tc: "O(n·k·d)", tw: "Every point against every centroid, coordinate by coordinate.", sc: "O(n·k)", sw: "Lists of members per cluster.",
    bad: "Correct, but all the distance arithmetic happens in the Python interpreter. The same table of distances is one broadcasted NumPy expression.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Broadcasting builds the (n, k) matrix of squared distances in one expression. `argmin` along the centroid axis gives the labels (and picks the lowest index on ties). The update is a masked mean per cluster.",
    steps: ["`d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)` is an (n, k) table.", "`labels = d.argmin(axis=1)`.", "For each j: `X[labels == j].mean(axis=0)` if the mask has any True, else keep `C[j]`."],
    code: R`
import numpy as np

def kmeans_step(X, C):
    X, C = np.asarray(X, float), np.asarray(C, float)
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)     # (n, k) squared distances
    labels = d.argmin(axis=1)                                  # nearest centroid, lowest index on ties
    new_C = C.copy()
    for j in range(len(C)):
        mask = labels == j
        if mask.any():                                         # an empty cluster keeps its centroid
            new_C[j] = X[mask].mean(axis=0)
    return labels, new_C`,
    tc: "O(n·k·d)", tw: "The distance table dominates.", sc: "O(n·k)", sw: "The (n, k) distance matrix.",
  },
  run: R`
X = [[0, 0], [0, 2], [10, 0], [10, 2]]
l, c = kmeans_step(X, [[0, 1], [10, 1]])
print(l.tolist(), np.round(c, 2).tolist())
l, c = kmeans_step([[0.0], [1.0], [9.0]], [[0.0], [1.0], [100.0]])
print(l.tolist(), np.round(c, 2).tolist())
X = [[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]]
l, c = kmeans_step(X, [[1, 1], [5, 7]])
print(l.tolist(), np.round(c, 3).tolist())
l, c = kmeans_step([[1.0], [3.0]], [[2.0], [2.0]])
print(l.tolist(), np.round(c, 2).tolist())`,
  out: R`
[0, 0, 1, 1] [[0.0, 1.0], [10.0, 1.0]]
[0, 1, 1] [[0.0], [5.0], [100.0]]
[0, 0, 0, 1, 1, 1, 1] [[1.833, 2.333], [4.125, 5.375]]
[0, 0] [[2.0], [2.0]]`,
  dryOn: "X = [[1,1],[1.5,2],[3,4],[5,7],[3.5,5],[4.5,5],[3.5,4.5]], C = [[1,1],[5,7]]",
  dry: { h: ["point", "dist² to C0 (1,1)", "dist² to C1 (5,7)", "label"], r: [["(1, 1)", "0", "52", "0"], ["(1.5, 2)", "1.25", "37.25", "0"], ["(3, 4)", "13", "13", "0 (tie → lower index)"], ["(5, 7)", "52", "0", "1"], ["(3.5, 5)", "22.25", "3.25", "1"], ["(4.5, 5)", "28.25", "4.25", "1"], ["(3.5, 4.5)", "18.5", "6.25", "1"]] },
  expl: [
    ["d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)", "Shapes (n,1,d) minus (1,k,d) broadcast to (n,k,d): the difference of every point from every centroid. Squaring and summing the last axis gives the squared distance."],
    ["labels = d.argmin(axis=1)", "For each row (point), the column (centroid) with the smallest distance. `argmin` returns the first minimum, which is the tie-break we want."],
    ["mask = labels == j", "A boolean vector selecting the points of cluster j."],
    ["X[mask].mean(axis=0)", "The centroid update: the coordinate-wise mean of those points."],
    ["if mask.any()", "Guards against an empty cluster, where the mean of nothing would be NaN."],
  ],
  take: "**k-means is two lines of linear algebra**: a distance table plus `argmin` (assign), then group means (update). Everything else is repeating it until the labels stop changing.",
  mist: ["Forgetting the empty-cluster case: `mean` of an empty selection gives NaN and a warning.", "Using distance (square root) in one place and squared distance in another; the argmin is the same, but mixing them in the inertia is wrong.", "Mutating `C` in place, so the next step compares against a half-updated array."],
},
{
  id: "kmeans-from-scratch", t: "k-Means from Scratch with Inertia", d: "Medium", p: "k-means",
  q: "Write `kmeans(X, k, max_iter=100)` that runs the full algorithm and returns `(labels, centroids, inertia)`.\n\nInitialise the centroids as the **first k rows** of `X` (so the result is deterministic). Repeat assign and update until the centroids stop changing (or `max_iter` iterations). An empty cluster keeps its centroid.\n\n**Inertia** is the sum of squared distances from every point to its assigned centroid, measured with the final labels and centroids.",
  ex: [["X = [[0],[1],[2],[10],[11],[12]], k = 2", "labels [0, 0, 0, 1, 1, 1], centroids [[1], [11]], inertia 4.0", "Each group has squared distances 1 + 0 + 1 = 2, so the inertia is 2 + 2 = 4."],
    ["k = 1 on the same data", "centroid [[6]], inertia 154.0", "With one cluster, the centroid is the overall mean."]],
  c: ["`X` has shape (n, d) with n ≥ k ≥ 1", "The first k rows are distinct"],
  hint: "Reuse the assign / update step from the previous problem inside a loop. After the loop, recompute the distances with the final centroids and sum each point's distance to its own centroid with `d[np.arange(n), labels].sum()`.",
  brute: {
    name: "Plain Python loops",
    idea: "The same algorithm with explicit loops over points, centroids and coordinates, and a loop to total up the inertia.",
    code: R`
import numpy as np

def kmeans(X, k, max_iter=100):
    X = [list(map(float, r)) for r in X]
    n, d = len(X), len(X[0])
    C = [r[:] for r in X[:k]]
    def assign(C):
        labels = []
        for p in X:
            dists = [sum((p[t] - c[t]) ** 2 for t in range(d)) for c in C]
            labels.append(dists.index(min(dists)))
        return labels
    for _ in range(max_iter):
        labels = assign(C)
        new_C = []
        for j in range(k):
            pts = [X[i] for i in range(n) if labels[i] == j]
            new_C.append([sum(p[t] for p in pts) / len(pts) for t in range(d)] if pts else C[j][:])
        if new_C == C:
            break
        C = new_C
    labels = assign(C)
    inertia = sum(sum((X[i][t] - C[labels[i]][t]) ** 2 for t in range(d)) for i in range(n))
    return np.array(labels), np.array(C), inertia`,
    tc: "O(iters·n·k·d)", tw: "Each iteration scans every point against every centroid.", sc: "O(n·d)", sw: "Copies of the data and centroids.",
    bad: "Right answer, but every distance is computed in interpreted Python.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "One broadcasted distance table per iteration, `argmin` for labels, masked means for the update, and `np.allclose` to detect convergence. The inertia is a fancy-indexed sum over the same table.",
    steps: ["Start with `C = X[:k].copy()`.", "Loop: distance table → labels → new centroids; stop when `np.allclose(new_C, C)`.", "Recompute the table with the final centroids; `inertia = d[np.arange(n), labels].sum()`."],
    code: R`
import numpy as np

def sq_dist(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)

def kmeans(X, k, max_iter=100):
    X = np.asarray(X, float)
    C = X[:k].copy()                                   # deterministic start: the first k rows
    for _ in range(max_iter):
        labels = sq_dist(X, C).argmin(axis=1)          # assign
        new_C = C.copy()
        for j in range(k):
            if np.any(labels == j):
                new_C[j] = X[labels == j].mean(axis=0) # update
        if np.allclose(new_C, C):                      # converged
            break
        C = new_C
    d = sq_dist(X, C)
    labels = d.argmin(axis=1)
    inertia = d[np.arange(len(X)), labels].sum()
    return labels, C, float(inertia)`,
    tc: "O(iters·n·k·d)", tw: "The distance table each iteration.", sc: "O(n·k)", sw: "The distance matrix.",
  },
  run: R`
X = [[0], [1], [2], [10], [11], [12]]
l, c, J = kmeans(X, 2)
print(l.tolist(), np.round(c, 2).tolist(), round(J, 2))
l, c, J = kmeans(X, 1)
print(l.tolist(), np.round(c, 2).tolist(), round(J, 2))
rng = np.random.default_rng(0)
blobs = np.vstack([rng.normal(0, 0.3, (20, 2)), rng.normal(5, 0.3, (20, 2)), rng.normal([0, 5], 0.3, (20, 2))])
order = np.array([0, 20, 40] + [i for i in range(60) if i not in (0, 20, 40)])
l, c, J = kmeans(blobs[order], 3)
print(np.bincount(l).tolist(), round(J, 2))
print(np.round(c[np.argsort(c[:, 0] + 10 * c[:, 1])], 2).tolist())
l, c, J = kmeans(blobs[order], 6)
print(round(J, 2) < 10)`,
  out: R`
[0, 0, 0, 1, 1, 1] [[1.0], [11.0]] 4.0
[0, 0, 0, 0, 0, 0] [[6.0]] 154.0
[20, 20, 20] 9.31
[[-0.08, 0.04], [-0.04, 5.04], [5.12, 5.06]]
True`,
  dryOn: "X = [[0],[1],[2],[10],[11],[12]], k = 2, start C = [[0],[1]]",
  dry: { setup: "Start with the first two rows as centroids: 0 and 1.", steps: [["Assign: 0→c0, 1→c1, 2→c1, 10→c1, 11→c1, 12→c1", "Distances to 0 and 1; only the point 0 is closer to c0."], ["Update: c0 = 0, c1 = mean(1, 2, 10, 11, 12) = 7.2", "c1 jumps toward the big group."], ["Assign: 0, 1, 2→c0; 10, 11, 12→c1", "Now 1 and 2 are closer to c0 (0) than c1 (7.2)."], ["Update: c0 = 1, c1 = 11", "Means of the two groups."], ["Assign again: unchanged, centroids unchanged", "Converged."]], result: "labels [0,0,0,1,1,1], centroids [1, 11], inertia 2 + 2 = 4" },
  expl: [
    ["C = X[:k].copy()", "Deterministic initialisation. `.copy()` stops later updates from modifying X."],
    ["np.allclose(new_C, C)", "Convergence test on the centroids; once the labels stop changing the centroids stop too."],
    ["d[np.arange(len(X)), labels]", "Fancy indexing picks, for each row, the distance to its own centroid."],
    ["return ..., float(inertia)", "The final distances use the final centroids, so the inertia matches the returned labels."],
  ],
  take: "**k-means never increases the inertia**: assign lowers it, update lowers it, so the loop must stop. But it stops at a *local* minimum, which is why libraries run it from several starts (`n_init`) and use k-means++.",
  mist: ["Computing the inertia with the labels from before the last update, so it does not match the returned centroids.", "Initialising from random points that happen to be duplicates, which leaves an empty cluster.", "Comparing labels instead of centroids for convergence on the very first iteration."],
},
{
  id: "pca-svd", t: "PCA with SVD and Explained Variance", d: "Medium", p: "PCA",
  q: "Write `pca(X, m)` that returns `(Z, ratio)`:\n\n- `Z`: the data projected onto the first `m` principal components, shape (n, m)\n- `ratio`: the explained variance ratio of those `m` components, shape (m,)\n\nCentre the data first, then use the SVD of the centred matrix. The variance along component j is $s_j^2/(n-1)$, and the ratio divides that by the total variance of all components.\n\nThe sign of a principal component is arbitrary, so the tests only print signs-free quantities.",
  ex: [["X = [[1,1],[2,2],[3,3]], m = 1", "ratio [1.0]", "All points lie on one line, so a single component explains 100% of the variance."],
    ["X = [[1,0],[-1,0],[0,2],[0,-2]], m = 2", "ratio [0.8, 0.2]", "The variances are 8/3 along the y-axis and 2/3 along the x-axis, so 80% and 20%."]],
  c: ["`X` has shape (n, d), n ≥ 2, 1 ≤ m ≤ min(n, d)"],
  hint: "`Xc = X - X.mean(axis=0)`. `U, s, Vt = np.linalg.svd(Xc, full_matrices=False)`. The rows of `Vt` are the components, so `Z = Xc @ Vt[:m].T`. Variances: `s**2 / (n-1)`.",
  brute: {
    name: "Covariance matrix and eigenvectors, built by hand",
    idea: "Compute the covariance matrix with explicit loops, take its eigen-decomposition, sort the eigenvalues in decreasing order, and project onto the top eigenvectors.",
    code: R`
import numpy as np

def pca(X, m):
    X = np.asarray(X, float)
    n, d = X.shape
    mean = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    Xc = np.array([[X[i][j] - mean[j] for j in range(d)] for i in range(n)])
    cov = np.zeros((d, d))
    for a in range(d):
        for b in range(d):
            cov[a][b] = sum(Xc[i][a] * Xc[i][b] for i in range(n)) / (n - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return Xc @ vecs[:, :m], vals[:m] / vals.sum()`,
    tc: "O(n·d² + d³)", tw: "Building the covariance and the eigen-decomposition.", sc: "O(d²)", sw: "The covariance matrix.",
    bad: "Correct, but forming XᵀX squares the condition number (less accurate), and the loops are slow. The SVD works on the data directly.",
  },
  opt: {
    name: "SVD of the centred data",
    idea: "The SVD of the centred matrix gives the principal directions (rows of Vt) and the singular values; squared singular values divided by n−1 are the variances. No covariance matrix needed.",
    steps: ["Centre: `Xc = X - X.mean(axis=0)`.", "`U, s, Vt = np.linalg.svd(Xc, full_matrices=False)`.", "`var = s**2 / (n-1)`, `ratio = var[:m] / var.sum()`.", "`Z = Xc @ Vt[:m].T` (equivalently `U[:, :m] * s[:m]`)."],
    code: R`
import numpy as np

def pca(X, m):
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0)                           # centre each feature
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2 / (len(X) - 1)                       # variance along each component
    Z = Xc @ Vt[:m].T                                 # scores on the first m components
    return Z, var[:m] / var.sum()`,
    tc: "O(n·d·min(n,d))", tw: "The cost of the SVD.", sc: "O(n·d)", sw: "The centred copy and the factors.",
  },
  run: R`
Z, r = pca([[1, 1], [2, 2], [3, 3]], 1)
print(np.round(r, 3).tolist(), np.round(np.abs(Z).ravel(), 3).tolist())
Z, r = pca([[1, 0], [-1, 0], [0, 2], [0, -2]], 2)
print(np.round(r, 3).tolist(), Z.shape)
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 3)) * np.array([5.0, 2.0, 0.5])
Z, r = pca(X, 2)
print(np.round(r, 3).tolist(), round(float(r.sum()), 3))
print(np.round(Z.var(axis=0, ddof=1), 2).tolist())
print(bool(abs(np.corrcoef(Z[:, 0], Z[:, 1])[0, 1]) < 1e-8))`,
  out: R`
[1.0] [1.414, 0.0, 1.414]
[0.8, 0.2] (4, 2)
[0.874, 0.119] 0.993
[29.88, 4.07]
True`,
  dryOn: "X = [[1,0],[-1,0],[0,2],[0,-2]], m = 2",
  dry: { setup: "The mean is (0, 0), so the data is already centred. Covariance = XᵀX / 3.", steps: [["XᵀX = [[2, 0], [0, 8]]", "Sum of x² is 1+1 = 2; sum of y² is 4+4 = 8; cross terms are 0."], ["Variances: 2/3 along x and 8/3 along y", "Divide by n−1 = 3."], ["Sort: component 1 = y-axis (8/3), component 2 = x-axis (2/3)", "Largest variance first."], ["Ratios: (8/3) / (10/3) = 0.8 and 0.2", "Divide by the total variance 10/3."]], result: "ratio [0.8, 0.2]" },
  expl: [
    ["Xc = X - X.mean(axis=0)", "PCA is defined on centred data; skipping this makes the first component point at the mean instead of the spread."],
    ["np.linalg.svd(Xc, full_matrices=False)", "Economy SVD: `Vt` has shape (min(n,d), d); its rows are the unit-length principal directions."],
    ["s ** 2 / (len(X) - 1)", "Singular values squared over n−1 equal the eigenvalues of the sample covariance matrix."],
    ["Xc @ Vt[:m].T", "Project each centred row onto the first m directions."],
  ],
  take: "**PCA is an SVD of the centred data**: right singular vectors are the components, squared singular values (over n−1) are the variances. Components are uncorrelated and sorted by variance.",
  mist: ["Forgetting to centre the data first.", "Forgetting to standardize features measured in different units before PCA.", "Dividing by n instead of n−1 for the variance and mismatching `sklearn` by a constant (the ratios are unaffected, the raw variances are).", "Fitting PCA on train and test together (leakage)."],
},
]);

PRAC.add("Evaluation", [
{
  id: "confusion-metrics", t: "Confusion Matrix, Precision, Recall & F1", d: "Easy", p: "Classification metrics",
  q: "Write `classification_metrics(y_true, y_pred)` for a binary problem with labels 0 and 1 (1 is the positive class). Return a tuple `(tp, fp, fn, tn, precision, recall, f1)`.\n\nWhen a denominator would be zero (for example no predicted positives), that metric is `0.0`.\n\n$\\text{precision}=\\frac{TP}{TP+FP}\\quad \\text{recall}=\\frac{TP}{TP+FN}\\quad F_1=\\frac{2PR}{P+R}$",
  ex: [["y_true = [1,1,1,0,0,0,0], y_pred = [1,1,0,1,0,0,0]", "tp 2, fp 1, fn 1, tn 3, precision 0.667, recall 0.667, f1 0.667", "Two hits, one false alarm, one miss, three correct negatives."],
    ["y_pred = all zeros", "precision 0.0, recall 0.0, f1 0.0", "No predicted positives: precision's denominator is 0, so it is reported as 0."]],
  c: ["Both inputs have the same length and contain only 0 and 1"],
  hint: "Each count is the sum of a boolean AND: `np.sum((p == 1) & (t == 1))` is TP. Divide carefully, with a small helper `safe_div(a, b)` that returns 0.0 when `b == 0`.",
  brute: {
    name: "One pass with if / elif",
    idea: "Loop over the pairs once and bump one of four counters, then compute the ratios.",
    code: R`
import numpy as np

def classification_metrics(y_true, y_pred):
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if p == 1 and t == 1: tp += 1
        elif p == 1 and t == 0: fp += 1
        elif p == 0 and t == 1: fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, tn, precision, recall, f1`,
    tc: "O(n)", tw: "One pass.", sc: "O(1)", sw: "Four counters.",
    bad: "Fine for correctness; with NumPy the four counts are four vectorised comparisons with no Python loop.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Boolean masks combined with `&`, summed to counts. A helper handles division by zero in one place.",
    steps: ["`t, p = np.asarray(y_true), np.asarray(y_pred)`.", "`tp = ((p == 1) & (t == 1)).sum()` and the same for fp, fn, tn.", "`safe_div` avoids dividing by zero.", "F1 is the harmonic mean of precision and recall."],
    code: R`
import numpy as np

def safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def classification_metrics(y_true, y_pred):
    t, p = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    tn = int(((p == 0) & (t == 0)).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return tp, fp, fn, tn, precision, recall, f1`,
    tc: "O(n)", tw: "Four vectorised passes.", sc: "O(n)", sw: "Boolean masks.",
  },
  run: R`
def show(t, p):
    r = classification_metrics(t, p)
    print(r[:4], [round(v, 3) for v in r[4:]])
show([1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0])
show([1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0])
show([1, 0, 1, 0], [1, 0, 1, 0])
show([0, 0, 0, 0], [0, 0, 0, 0])
show([1] * 5 + [0] * 95, [1, 1, 1, 1, 0] + [1] * 10 + [0] * 85)`,
  out: R`
(2, 1, 1, 3) [0.667, 0.667, 0.667]
(0, 0, 3, 4) [0.0, 0.0, 0.0]
(2, 0, 0, 2) [1.0, 1.0, 1.0]
(0, 0, 0, 4) [0.0, 0.0, 0.0]
(4, 10, 1, 85) [0.286, 0.8, 0.421]`,
  dryOn: "y_true = [1,1,1,0,0,0,0], y_pred = [1,1,0,1,0,0,0]",
  dry: { h: ["i", "true", "pred", "cell"], r: [["0", "1", "1", "TP"], ["1", "1", "1", "TP"], ["2", "1", "0", "FN"], ["3", "0", "1", "FP"], ["4", "0", "0", "TN"], ["5", "0", "0", "TN"], ["6", "0", "0", "TN"]] },
  expl: [
    ["((p == 1) & (t == 1)).sum()", "Element-wise AND of two boolean arrays; summing booleans counts the True entries."],
    ["safe_div(tp, tp + fp)", "Precision. If nothing was predicted positive the denominator is 0 and we report 0.0."],
    ["safe_div(2 * precision * recall, precision + recall)", "F1, the harmonic mean; it is 0 when either is 0."],
  ],
  take: "**All the classification metrics are ratios of four counts.** Precision looks down the predicted-positive column, recall along the actual-positive row; F1 punishes a big gap between them.",
  mist: ["Dividing by zero when there are no predicted (or no actual) positives.", "Swapping rows and columns: FP and FN get mixed up.", "Averaging precision and recall instead of taking the harmonic mean.", "Reporting accuracy on imbalanced data."],
},
{
  id: "roc-auc-scores", t: "ROC-AUC from Scores", d: "Medium", p: "Classification metrics",
  q: "Write `roc_auc(y_true, scores)` that returns the area under the ROC curve for binary labels (0/1) and real-valued scores (higher = more likely positive).\n\nUse the ranking meaning of AUC: it is the probability that a randomly chosen positive has a **higher** score than a randomly chosen negative, counting **ties as one half**.\n\n$\\text{AUC}=\\frac{1}{n_+ n_-}\\sum_{i\\in +}\\sum_{j\\in -}\\Big[\\mathbb 1(s_i>s_j)+\\tfrac12\\mathbb 1(s_i=s_j)\\Big]$",
  ex: [["y = [1,1,0,0], scores = [0.9, 0.6, 0.7, 0.2]", "0.75", "Pairs: (0.9,0.7) win, (0.9,0.2) win, (0.6,0.7) lose, (0.6,0.2) win: 3 of 4."],
    ["y = [1,0], scores = [0.5, 0.5]", "0.5", "A tie counts as half."]],
  c: ["Both classes are present", "Scores may contain ties"],
  hint: "Brute force: compare every positive with every negative. Faster: the **rank-sum** (Mann-Whitney) formula. Rank all scores (ties get the average of their ranks, starting at 1); then $\\text{AUC}=\\frac{R_+ - n_+(n_+ + 1)/2}{n_+ n_-}$ where $R_+$ is the sum of the positives' ranks.",
  brute: {
    name: "Compare every pair",
    idea: "Loop over all (positive, negative) pairs and count wins and ties. Direct from the definition, but O(n₊·n₋).",
    code: R`
import numpy as np

def roc_auc(y_true, scores):
    pos = [s for y, s in zip(y_true, scores) if y == 1]
    neg = [s for y, s in zip(y_true, scores) if y == 0]
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))`,
    tc: "O(n₊·n₋)", tw: "Up to n²/4 comparisons.", sc: "O(n)", sw: "The two score lists.",
    bad: "Correct, but quadratic: 100,000 examples means billions of comparisons. Sorting once is enough.",
  },
  opt: {
    name: "Rank-sum with average ranks",
    idea: "Sort the scores, give each distinct value the average of the ranks it occupies, then the positives' rank sum gives the number of correctly ordered pairs without comparing pairs.",
    steps: ["`uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)`.", "A group of tied scores occupying ranks `start..end` gets rank `(start+end)/2`; `ends = cumsum(counts)`, `starts = ends - counts + 1`.", "`ranks = avg[inv]`.", "`AUC = (sum of positive ranks - n₊(n₊+1)/2) / (n₊·n₋)`."],
    code: R`
import numpy as np

def roc_auc(y_true, scores):
    y = np.asarray(y_true)
    s = np.asarray(scores, float)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts + 1
    ranks = ((starts + ends) / 2)[inv]                 # tied scores share their average rank
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))`,
    tc: "O(n log n)", tw: "A sort (inside np.unique).", sc: "O(n)", sw: "Rank arrays.",
  },
  run: R`
print(round(roc_auc([1, 1, 0, 0], [0.9, 0.6, 0.7, 0.2]), 4))
print(round(roc_auc([1, 0], [0.5, 0.5]), 4))
print(round(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 4))
print(round(roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]), 4))
print(round(roc_auc([1, 1, 0, 1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2]), 4))
rng = np.random.default_rng(0)
y = rng.integers(0, 2, 300); s = rng.normal(size=300) + y
print(round(roc_auc(y, s), 4), round(roc_auc(y, np.round(s)), 4))`,
  out: R`
0.75
0.5
1.0
0.0
0.8125
0.8143 0.7981`,
  dryOn: "y = [1,1,0,0], scores = [0.9, 0.6, 0.7, 0.2]",
  dry: { h: ["positive score", "vs negative 0.7", "vs negative 0.2", "wins"], r: [["0.9", "win", "win", "2"], ["0.6", "lose", "win", "1"], ["**total**", "", "", "**3 of 4 pairs = 0.75**"]] },
  expl: [
    ["np.unique(s, return_inverse=True, return_counts=True)", "Sorted distinct scores, which group each score belongs to, and how many ties each group has."],
    ["ends = np.cumsum(counts); starts = ends - counts + 1", "A group of c tied scores occupies c consecutive ranks; these are the first and last of them (ranks start at 1)."],
    ["(starts + ends) / 2", "The average rank shared by tied scores, which is what gives ties a half credit."],
    ["ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2", "The smallest possible rank sum for the positives is n₊(n₊+1)/2; what is left counts pairs where a positive outranks a negative."],
  ],
  take: "**AUC is a ranking statistic**: the chance a random positive outscores a random negative. The rank-sum formula computes it in O(n log n) with one sort and no explicit curve.",
  mist: ["Giving tied scores arbitrary distinct ranks instead of their average (biases the AUC).", "Starting ranks at 0 instead of 1, which shifts the sum.", "Passing hard 0/1 predictions instead of scores, which collapses the curve to a single point."],
},
{
  id: "stratified-kfold", t: "Stratified k-Fold Indices", d: "Medium", p: "Cross-validation",
  q: "Write `stratified_kfold(y, k)` that splits the row indices `0..n-1` into `k` **validation folds** so that every fold has (almost) the same class proportions as `y`. Return a list of `k` lists of indices, each in increasing order.\n\nUse this deterministic rule: process the rows grouped by class (classes in increasing order, rows of a class in their original order), and deal them out one at a time to folds 0, 1, 2, ..., k-1, 0, 1, ... continuing the counter from one class to the next.\n\nEvery index must appear in exactly one fold.",
  ex: [["y = [0,0,0,0,0,0,1,1,1], k = 3", "[[0, 3, 6], [1, 4, 7], [2, 5, 8]]", "Each fold gets two of class 0 and one of class 1."],
    ["y = [0,1,0,1], k = 2", "[[0, 1], [2, 3]]", "Class 0 rows 0, 2 go to folds 0 and 1; class 1 rows 1, 3 continue the counter: fold 0 and fold 1."]],
  c: ["`y` is a 1-D array of integer labels", "2 ≤ k ≤ n"],
  hint: "Sort the row indices by class with a **stable** sort (`np.argsort(y, kind=\"stable\")`), so rows within a class stay in order. Then the i-th row in that order goes to fold `i % k`.",
  brute: {
    name: "Loop over classes",
    idea: "For each class in turn, take its row indices in order and deal them out to the folds with a running counter.",
    code: R`
import numpy as np

def stratified_kfold(y, k):
    y = list(y)
    folds = [[] for _ in range(k)]
    counter = 0
    for c in sorted(set(y)):
        for i, label in enumerate(y):
            if label == c:
                folds[counter % k].append(i)
                counter += 1
    return [sorted(f) for f in folds]`,
    tc: "O(C·n)", tw: "One scan of the labels per class.", sc: "O(n)", sw: "The folds.",
    bad: "Scans the whole label list once per class. A single stable sort groups all classes at once.",
  },
  opt: {
    name: "Stable sort and modulo",
    idea: "A stable argsort of the labels lists the rows class by class, each class in original order. Dealing positions 0, 1, 2, ... to folds is just `position % k`.",
    steps: ["`order = np.argsort(y, kind=\"stable\")`.", "`fold = np.empty(n, int); fold[order] = np.arange(n) % k`.", "Fold f is `np.where(fold == f)[0]` (already increasing)."],
    code: R`
import numpy as np

def stratified_kfold(y, k):
    y = np.asarray(y)
    n = len(y)
    order = np.argsort(y, kind="stable")           # rows grouped by class, original order kept inside a class
    fold = np.empty(n, dtype=int)
    fold[order] = np.arange(n) % k                 # deal them out round-robin
    return [np.where(fold == f)[0].tolist() for f in range(k)]`,
    tc: "O(n log n)", tw: "The sort.", sc: "O(n)", sw: "The fold id per row.",
  },
  run: R`
print(stratified_kfold([0, 0, 0, 0, 0, 0, 1, 1, 1], 3))
print(stratified_kfold([0, 1, 0, 1], 2))
y = np.array([0] * 90 + [1] * 10)
folds = stratified_kfold(y, 5)
print([len(f) for f in folds], [int(y[f].sum()) for f in folds])
print(sorted(i for f in folds for i in f) == list(range(100)))
y = np.array([0, 1, 2] * 4 + [2, 2])
folds = stratified_kfold(y, 4)
print([[int((y[f] == c).sum()) for c in range(3)] for f in folds])`,
  out: R`
[[0, 3, 6], [1, 4, 7], [2, 5, 8]]
[[0, 1], [2, 3]]
[20, 20, 20, 20, 20] [2, 2, 2, 2, 2]
True
[[1, 1, 2], [1, 1, 2], [1, 1, 1], [1, 1, 1]]`,
  dryOn: "y = [0,0,0,0,0,0,1,1,1], k = 3",
  dry: { h: ["position in sorted order", "row index", "class", "fold = position % 3"], r: [["0", "0", "0", "0"], ["1", "1", "0", "1"], ["2", "2", "0", "2"], ["3", "3", "0", "0"], ["4", "4", "0", "1"], ["5", "5", "0", "2"], ["6", "6", "1", "0"], ["7", "7", "1", "1"], ["8", "8", "1", "2"]] },
  expl: [
    ["np.argsort(y, kind=\"stable\")", "Row indices ordered by label. Stable means equal labels keep their original relative order, which keeps the result deterministic."],
    ["np.arange(n) % k", "The round-robin pattern 0,1,...,k-1,0,1,... over the sorted positions."],
    ["fold[order] = ...", "Scatter the fold numbers back to the original row positions."],
    ["np.where(fold == f)[0]", "All rows of fold f, in increasing index order."],
  ],
  take: "**Stratification = group by class, then deal round-robin.** Every fold gets the same share of each class (to within one row), so no fold ends up with almost no positives.",
  mist: ["Using plain k-fold on imbalanced data, so one fold may contain no positives and its F1 or AUC is undefined.", "Shuffling the rows differently per class and losing reproducibility (set a seed).", "Forgetting that the validation fold, not the training part, is what each list contains."],
},
]);
