/* Practice: Regression + Classification (machine learning part 1)
   Each problem: q, ex, hint, brute (plain Python), opt (NumPy), run + out (tests), dry, expl, take, mist. */
PRAC.add("Regression", [
{
  id: "linreg-normal-equation", t: "Least Squares with the Normal Equation", d: "Easy", p: "Linear regression",
  q: "Fit a linear regression $\\hat y = b + w_1 x_1 + \\dots + w_d x_d$ **in closed form**. Write `fit_linear(X, y)` that returns a 1-D array `[b, w1, ..., wd]`, and `predict(X, params)` that returns the predictions.\n\nThe least-squares weights solve the **normal equation**:\n\n$X_b^\\top X_b \\, \\mathbf{w} = X_b^\\top \\mathbf{y}$\n\nwhere $X_b$ is `X` with a column of ones added in front (so the intercept is just another weight). Do **not** compute a matrix inverse; solve the linear system instead.",
  ex: [["X = [[1],[2],[3],[4]], y = [3, 5, 7, 9]", "[1.0, 2.0]", "The points lie exactly on y = 1 + 2x: intercept 1, slope 2."],
    ["predict([[10]]) with those parameters", "[21.0]", "1 + 2·10 = 21."]],
  c: ["`X` has shape (n, d) with n > d, and its columns are not perfectly collinear", "`y` has shape (n,)"],
  hint: "Stack a column of ones in front of `X` with `np.hstack`. Then `np.linalg.solve(A, b)` solves $A\\mathbf w = \\mathbf b$ without ever forming $A^{-1}$.",
  brute: {
    name: "Plain Python loops",
    idea: "Build the augmented rows by hand, accumulate the matrix $X_b^\\top X_b$ and the vector $X_b^\\top y$ with nested loops, then solve the small system.",
    code: R`
import numpy as np

def fit_linear(X, y):
    n, d = len(X), len(X[0]) + 1
    rows = [[1.0] + [float(v) for v in row] for row in X]      # add the intercept column
    A = [[0.0] * d for _ in range(d)]                          # X_b^T X_b
    r = [0.0] * d                                              # X_b^T y
    for i in range(n):
        for a in range(d):
            r[a] += rows[i][a] * y[i]
            for b in range(d):
                A[a][b] += rows[i][a] * rows[i][b]
    return np.linalg.solve(np.array(A), np.array(r))

def predict(X, params):
    return np.array([params[0] + sum(p * v for p, v in zip(params[1:], row)) for row in X])`,
    tc: "O(n·d²)", tw: "Three nested loops over rows and the d×d matrix.", sc: "O(d²)", sw: "The Gram matrix.",
    bad: "Correct but slow: Python loops over every row and every pair of features. The same sums are one matrix product in NumPy.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Prepend a column of ones, then `Xb.T @ Xb` and `Xb.T @ y` build both sides of the normal equation in two matrix products, and `np.linalg.solve` finishes. Solving is faster and more accurate than inverting.",
    steps: ["`Xb = np.hstack([ones, X])` adds the intercept column.", "`A = Xb.T @ Xb`, `r = Xb.T @ y`.", "`np.linalg.solve(A, r)` returns `[b, w1, ..., wd]`.", "Prediction is `Xb @ params`."],
    code: R`
import numpy as np

def add_ones(X):
    return np.hstack([np.ones((len(X), 1)), X])

def fit_linear(X, y):
    Xb = add_ones(X)
    return np.linalg.solve(Xb.T @ Xb, Xb.T @ y)       # solve, don't invert

def predict(X, params):
    return add_ones(X) @ params`,
    tc: "O(n·d² + d³)", tw: "Building XᵀX costs n·d²; solving the d×d system costs d³.", sc: "O(d²)", sw: "The Gram matrix.",
  },
  run: R`
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([3.0, 5.0, 7.0, 9.0])
p = fit_linear(X, y)
print(np.round(p, 3).tolist())
print(np.round(predict(np.array([[10.0]]), p), 3).tolist())
rng = np.random.default_rng(0)
X2 = rng.normal(size=(50, 2))
y2 = 3.0 + 2.0 * X2[:, 0] - 1.5 * X2[:, 1]
print(np.round(fit_linear(X2, y2), 3).tolist())
y3 = y2 + rng.normal(0, 0.1, 50)
print(np.round(fit_linear(X2, y3), 2).tolist())`,
  out: R`
[1.0, 2.0]
[21.0]
[3.0, 2.0, -1.5]
[3.0, 2.02, -1.5]`,
  dryOn: "X = [[1],[2],[3]], y = [3, 5, 7]",
  dry: { h: ["quantity", "value"], r: [["X_b", "[[1,1],[1,2],[1,3]]"], ["X_bᵀ X_b", "[[3, 6], [6, 14]]"], ["X_bᵀ y", "[15, 34]"], ["solve", "3b + 6w = 15, 6b + 14w = 34"], ["result", "**b = 1, w = 2**"]] },
  expl: [
    ["np.hstack([np.ones((len(X), 1)), X])", "Glue a column of ones to the left of `X`, so the first weight is the intercept."],
    ["Xb.T @ Xb", "A (d+1)×(d+1) matrix of dot products between feature columns."],
    ["Xb.T @ y", "How strongly each column lines up with the target."],
    ["np.linalg.solve(A, r)", "Solves the linear system exactly; the weights that minimise the squared error."],
  ],
  take: "**Least squares = solve $X^\\top X w = X^\\top y$.** Add a ones column for the intercept and use `solve`, never an explicit inverse.",
  mist: ["Forgetting the column of ones: the line is forced through the origin.", "Using `np.linalg.inv(A) @ r`: slower and less accurate than `solve`.", "Passing collinear columns: `XᵀX` is singular and the solve fails or blows up (use ridge or `lstsq`)."],
},
{
  id: "ridge-closed-form", t: "Ridge Regression in Closed Form", d: "Medium", p: "Regularization",
  q: "Implement **ridge regression** (linear regression with an L2 penalty) with its closed-form solution. Write `ridge_fit(X, y, alpha)` returning `(w, b)`: the weight vector and the intercept.\n\nThe weights minimise $\\lVert X\\mathbf w + b - \\mathbf y\\rVert^2 + \\alpha\\lVert \\mathbf w\\rVert^2$. The intercept is **not** penalised. The standard way to handle that: centre `X` and `y` (subtract their column means), solve\n\n$\\mathbf w = (X_c^\\top X_c + \\alpha I)^{-1} X_c^\\top \\mathbf y_c$\n\nand then recover $b = \\bar y - \\bar{\\mathbf x}^\\top \\mathbf w$.",
  ex: [["X = [[1],[2],[3]], y = [2, 4, 6], alpha = 0", "w = [2.0], b = 0.0", "Without a penalty it is plain least squares: y = 2x."],
    ["same data, alpha = 2", "w = [1.0], b = 2.0", "The penalty halves the slope: On centred data w = Σxy / (Σx² + α) = 4 / (2 + 2) = 1, and b = 4 − 2·1 = 2."]],
  c: ["`alpha >= 0`", "`X` has shape (n, d), `y` has shape (n,)"],
  hint: "Centre first: `Xc = X - X.mean(axis=0)`, `yc = y - y.mean()`. Then `np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ yc)`. The intercept is `y.mean() - X.mean(axis=0) @ w`.",
  brute: {
    name: "Plain Python loops",
    idea: "Compute the column means with loops, build the centred Gram matrix entry by entry, add $\\alpha$ to its diagonal, solve, and recover the intercept.",
    code: R`
import numpy as np

def ridge_fit(X, y, alpha):
    n, d = len(X), len(X[0])
    xm = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    ym = sum(y) / n
    A = [[0.0] * d for _ in range(d)]
    r = [0.0] * d
    for i in range(n):
        for a in range(d):
            ca = X[i][a] - xm[a]
            r[a] += ca * (y[i] - ym)
            for b in range(d):
                A[a][b] += ca * (X[i][b] - xm[b])
    for a in range(d):
        A[a][a] += alpha                                   # the ridge penalty
    w = np.linalg.solve(np.array(A), np.array(r))
    b = ym - sum(xm[j] * w[j] for j in range(d))
    return w, float(b)`,
    tc: "O(n·d² + d³)", tw: "Loops for the Gram matrix, then the solve.", sc: "O(d²)", sw: "The d×d matrix.",
    bad: "Same maths, but slow Python loops over rows and feature pairs.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Centre the data, form $X_c^\\top X_c + \\alpha I$ with one matrix product, solve the system, and recover the intercept from the means. Adding $\\alpha I$ also makes the matrix invertible when features are collinear.",
    steps: ["Subtract the column means from `X` and the mean from `y`.", "Build `A = Xc.T @ Xc + alpha * np.eye(d)`.", "`w = np.linalg.solve(A, Xc.T @ yc)`.", "`b = y.mean() - x_mean @ w`."],
    code: R`
import numpy as np

def ridge_fit(X, y, alpha):
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean                        # centre: the intercept is not penalised
    d = X.shape[1]
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ yc)
    b = y_mean - x_mean @ w
    return w, float(b)`,
    tc: "O(n·d² + d³)", tw: "One matrix product and one solve.", sc: "O(d²)", sw: "The d×d matrix.",
  },
  run: R`
X = np.array([[1.0], [2.0], [3.0]])
y = np.array([2.0, 4.0, 6.0])
for a in [0.0, 2.0]:
    w, b = ridge_fit(X, y, a)
    print(np.round(w, 3).tolist(), round(b, 3))
rng = np.random.default_rng(1)
x1 = rng.normal(size=30)
X2 = np.column_stack([x1, x1 + rng.normal(0, 0.01, 30)])   # two almost identical columns
y2 = 4 * x1 + 1.0
for a in [0.01, 1.0, 100.0]:
    w, b = ridge_fit(X2, y2, a)
    print(a, np.round(w, 2).tolist(), round(b, 2))`,
  out: R`
[2.0] 0.0
[1.0] 2.0
0.01 [2.21, 1.79] 1.0
1.0 [1.96, 1.95] 0.99
100.0 [0.58, 0.58] 0.81`,
  dryOn: "X = [[1],[2],[3]], y = [2, 4, 6], alpha = 2",
  dry: { h: ["step", "value"], r: [["means", "x̄ = 2, ȳ = 4"], ["centred X, y", "[−1, 0, 1], [−2, 0, 2]"], ["XcᵀXc + α", "2 + 2 = 4"], ["Xcᵀyc", "4"], ["w", "4 / 4 = **1**"], ["b", "4 − 2·1 = **2**"]] },
  expl: [
    ["Xc, yc = X - x_mean, y - y_mean", "Centring removes the intercept from the problem, so it escapes the penalty."],
    ["alpha * np.eye(d)", "Adds α to the diagonal of XᵀX: that is the L2 penalty in matrix form."],
    ["b = y_mean - x_mean @ w", "Puts the line back through the data's centre."],
  ],
  take: "**Ridge = least squares with α added to the diagonal.** Centre the data so the intercept is not shrunk.",
  mist: ["Penalising the intercept (adding a ones column and then α·I on all of it).", "Not centring, then comparing to scikit-learn's `Ridge`, which does centre.", "Forgetting that larger α shrinks weights: α = 0 is plain least squares."],
},
{
  id: "linreg-minibatch-gd", t: "Mini-Batch Gradient Descent for Linear Regression", d: "Medium", p: "Gradient descent",
  q: "Train a linear regression with **mini-batch gradient descent**. Write `minibatch_gd(X, y, lr, epochs, batch_size, seed)` returning `(w, b)`.\n\nEach epoch: shuffle the row order (with `np.random.default_rng(seed)`, created once at the start), cut it into consecutive batches of `batch_size` rows (the last batch may be smaller), and for each batch take one step\n\n$\\mathbf w \\leftarrow \\mathbf w - \\eta \\cdot \\tfrac{2}{m} X_B^\\top(X_B\\mathbf w + b - \\mathbf y_B) \\qquad b \\leftarrow b - \\eta \\cdot \\tfrac{2}{m}\\sum(X_B\\mathbf w + b - \\mathbf y_B)$\n\nwhere $m$ is the size of that batch. Start from $\\mathbf w = 0$, $b = 0$.",
  ex: [["y = 1 + 2·x₁ − 3·x₂ exactly, lr = 0.1, 200 epochs, batch 16", "w ≈ [2, −3], b ≈ 1", "Noise-free data: the minimum of the loss is the true line."]],
  c: ["`lr > 0`, `epochs >= 1`, `1 <= batch_size <= n`", "Shuffle with `rng.permutation(n)` once per epoch"],
  hint: "Inside the epoch loop: `idx = rng.permutation(n)`, then `for s in range(0, n, batch_size): batch = idx[s:s+batch_size]`. The error is `err = X[batch] @ w + b - y[batch]`.",
  brute: {
    name: "Plain Python loops",
    idea: "For each batch, loop over its rows to accumulate the gradient of every weight, then take the step.",
    code: R`
import numpy as np

def minibatch_gd(X, y, lr, epochs, batch_size, seed):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w, b = [0.0] * d, 0.0
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch_size):
            batch = idx[s:s + batch_size]
            gw, gb = [0.0] * d, 0.0
            for i in batch:
                err = sum(w[j] * X[i][j] for j in range(d)) + b - y[i]
                for j in range(d):
                    gw[j] += 2 * err * X[i][j] / len(batch)
                gb += 2 * err / len(batch)
            w = [w[j] - lr * gw[j] for j in range(d)]
            b -= lr * gb
    return np.array(w), b`,
    tc: "O(epochs·n·d)", tw: "Every row visited once per epoch, with a loop over features.", sc: "O(d)", sw: "Weights and gradients.",
    bad: "Correct, but each row costs several Python-level operations. A batch is one small matrix product.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Per batch compute the error vector with one matrix–vector product, then both gradients with `X_B.T @ err` and `err.mean()`.",
    steps: ["Shuffle indices once per epoch.", "`Xb, yb = X[batch], y[batch]`; `err = Xb @ w + b - yb`.", "`w -= lr * 2 * Xb.T @ err / m`; `b -= lr * 2 * err.mean()`."],
    code: R`
import numpy as np

def minibatch_gd(X, y, lr, epochs, batch_size, seed):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        idx = rng.permutation(n)                           # new order every epoch
        for s in range(0, n, batch_size):
            batch = idx[s:s + batch_size]
            Xb, yb = X[batch], y[batch]
            err = Xb @ w + b - yb                          # predictions minus truth
            w = w - lr * 2 * Xb.T @ err / len(batch)       # dL/dw
            b = b - lr * 2 * err.mean()                    # dL/db
    return w, float(b)`,
    tc: "O(epochs·n·d)", tw: "Same arithmetic, in compiled code.", sc: "O(d)", sw: "Weights only.",
  },
  run: R`
rng = np.random.default_rng(0)
X = rng.normal(size=(120, 2))
y = 1.0 + 2.0 * X[:, 0] - 3.0 * X[:, 1]
w, b = minibatch_gd(X, y, 0.1, 100, 16, 0)
print(np.round(w, 2).tolist(), round(float(b), 2))
w, b = minibatch_gd(X, y, 0.05, 150, 120, 1)
print(np.round(w, 2).tolist(), round(float(b), 2))
w, b = minibatch_gd(X, y, 0.1, 100, 1, 2)
print(np.round(w, 1).tolist(), round(float(b), 1))`,
  out: R`
[2.0, -3.0] 1.0
[2.0, -3.0] 1.0
[2.0, -3.0] 1.0`,
  dryOn: "one step, X_B = [[1],[3]], y_B = [2, 6], w = 0, b = 0, lr = 0.1",
  dry: { h: ["quantity", "value"], r: [["err = X_B w + b − y_B", "[−2, −6]"], ["∂L/∂w = 2/m · X_Bᵀ err", "2/2 · (1·−2 + 3·−6) = −20"], ["∂L/∂b = 2/m · Σ err", "2/2 · (−8) = −8"], ["new w, b", "w = 0 + 0.1·20 = **2.0**, b = 0 + 0.8 = **0.8**"]] },
  expl: [
    ["idx = rng.permutation(n)", "Shuffling every epoch means the batches differ each time, which de-correlates the steps."],
    ["err = Xb @ w + b - yb", "Residuals for the whole batch in one go."],
    ["Xb.T @ err", "Sum over the batch of (residual × feature): the weight gradient."],
    ["range(0, n, batch_size)", "Walks through the shuffled indices in consecutive chunks; the last may be shorter."],
  ],
  take: "**Mini-batch GD = shuffle, slice into batches, one gradient step per batch.** Batch size trades gradient noise against speed.",
  mist: ["Not shuffling (or shuffling only once): the batches stay the same each epoch.", "Dividing by `batch_size` instead of the actual batch length on the last, smaller batch.", "Forgetting the factor 2 in the MSE gradient (it just rescales the learning rate, but breaks comparisons)."],
},
]);

PRAC.add("Classification", [
{
  id: "knn-predict", t: "k-Nearest Neighbours Classifier", d: "Easy", p: "k-NN",
  q: "Write `knn_predict(X_train, y_train, X_new, k)` that classifies every row of `X_new` by a **majority vote among its `k` nearest training points** (Euclidean distance). Return a NumPy array of predicted labels (integers 0, 1, 2, …).\n\nIf two classes tie in the vote, return the one with the **smaller label**.",
  ex: [["X_train = [[0],[1],[2],[10],[11]], y_train = [0,0,0,1,1], X_new = [[1.4]], k = 3", "[0]", "The 3 nearest points are x = 1, 2, 0: all class 0."],
    ["X_new = [[9]], k = 1", "[1]", "The nearest point is x = 10, class 1."]],
  c: ["`1 <= k <= len(X_train)`", "Labels are non-negative integers"],
  hint: "`np.argsort(dist)[:k]` gives the indices of the k smallest distances. `np.bincount(labels)` counts votes, and `np.argmax` of it returns the smallest label on ties. Compute all distances to all training points with broadcasting, no loop needed.",
  brute: {
    name: "Plain Python loops",
    idea: "For every new point, compute the distance to every training point in a loop, sort them, take the first k labels and count the votes with a dictionary.",
    code: R`
import math
import numpy as np

def knn_predict(X_train, y_train, X_new, k):
    out = []
    for x in X_new:
        dists = []
        for xt, yt in zip(X_train, y_train):
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, xt)))
            dists.append((d, int(yt)))
        dists.sort()
        votes = {}
        for _, label in dists[:k]:
            votes[label] = votes.get(label, 0) + 1
        out.append(min(votes, key=lambda c: (-votes[c], c)))     # most votes, ties -> smaller label
    return np.array(out)`,
    tc: "O(m·n·d + m·n log n)", tw: "m queries, each against n training points, plus a sort.", sc: "O(n)", sw: "Distances for one query.",
    bad: "Fine for tiny data, but a Python loop per distance is slow, and a full sort is wasted work when only k values matter.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Broadcasting computes the whole m×n distance matrix at once: `X_new[:, None, :] - X_train[None, :, :]`. Then `argsort` along each row finds the neighbours and `bincount` votes.",
    steps: ["Squared distances: `((X_new[:, None, :] - X_train[None]) ** 2).sum(axis=2)` has shape (m, n).", "`np.argsort(D, axis=1)[:, :k]` gives each query's k nearest indices.", "Look up their labels, `np.bincount` them per row, and take `argmax`."],
    code: R`
import numpy as np

def knn_predict(X_train, y_train, X_new, k):
    D = ((X_new[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)   # (m, n) squared distances
    nearest = np.argsort(D, axis=1)[:, :k]                              # indices of the k closest
    labels = y_train[nearest]                                           # (m, k) their labels
    return np.array([np.argmax(np.bincount(row)) for row in labels])    # majority vote, ties -> smaller`,
    tc: "O(m·n·d + m·n log n)", tw: "Distance matrix plus a row-wise sort (use argpartition for large n).", sc: "O(m·n)", sw: "The full distance matrix.",
  },
  run: R`
Xt = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
yt = np.array([0, 0, 0, 1, 1])
print(knn_predict(Xt, yt, np.array([[1.4], [9.0], [5.0]]), 3).tolist())
print(knn_predict(Xt, yt, np.array([[9.0]]), 1).tolist())
rng = np.random.default_rng(0)
A = rng.normal([0, 0], 1.0, size=(40, 2))
B = rng.normal([3, 3], 1.0, size=(40, 2))
Xtr = np.vstack([A, B]); ytr = np.array([0] * 40 + [1] * 40)
Xq = np.array([[0.2, -0.1], [2.8, 3.2], [1.5, 1.5], [-2.0, 0.0]])
print(knn_predict(Xtr, ytr, Xq, 5).tolist())
print(knn_predict(Xtr, ytr, Xq, 80).tolist())`,
  out: R`
[0, 1, 0]
[1]
[0, 1, 0, 0]
[0, 0, 0, 0]`,
  dryOn: "x_new = 1.4, k = 3 on [0,1,2,10,11] with labels [0,0,0,1,1]",
  dry: { h: ["train x", "label", "distance to 1.4"], r: [["0", "0", "1.4"], ["1", "0", "**0.4**"], ["2", "0", "**0.6**"], ["10", "1", "8.6"], ["11", "1", "9.6"]] },
  expl: [
    ["X_new[:, None, :] - X_train[None, :, :]", "Broadcasting: every query minus every training point, shape (m, n, d)."],
    ["np.argsort(D, axis=1)[:, :k]", "Per query, the indices sorted by distance; keep the first k."],
    ["y_train[nearest]", "Fancy indexing turns those indices into labels."],
    ["np.argmax(np.bincount(row))", "Count the votes; argmax returns the first (smallest) label on a tie."],
  ],
  take: "**k-NN = distances, take the k smallest, vote.** Broadcasting computes all distances at once.",
  mist: ["Forgetting that large `k` smooths everything towards the majority class (k = n always predicts it).", "Using unscaled features: one big-valued feature decides all distances.", "Taking `argsort(...)[-k:]` (the farthest) instead of `[:k]`."],
},
{
  id: "logreg-gradient-descent", t: "Logistic Regression with Gradient Descent", d: "Medium", p: "Logistic regression",
  q: "Train a **logistic regression** classifier from scratch. Write `fit_logreg(X, y, lr, steps)` returning `(w, b)`, trained with full-batch gradient descent on the **binary cross-entropy** loss, starting from $\\mathbf w = 0, b = 0$. Also write `predict_proba(X, w, b)` returning $\\sigma(X\\mathbf w + b)$.\n\n$p_i = \\sigma(\\mathbf w^\\top \\mathbf x_i + b) \\qquad L = -\\tfrac1n\\sum_i \\big[y_i\\ln p_i + (1-y_i)\\ln(1-p_i)\\big]$\n\nThe gradients are $\\tfrac1n X^\\top(\\mathbf p - \\mathbf y)$ for the weights and $\\tfrac1n\\sum_i (p_i - y_i)$ for the bias.",
  ex: [["X = [[-2],[-1],[1],[2]], y = [0,0,1,1]", "w > 0, b ≈ 0", "Larger x means class 1, so the weight is positive; the data is symmetric, so the bias stays at 0."]],
  c: ["`y` contains only 0 and 1", "Use the numerically safe sigmoid for the range of inputs in the tests"],
  hint: "`p = 1 / (1 + np.exp(-(X @ w + b)))`. The gradient is `X.T @ (p - y) / n`: prediction minus truth, just like linear regression.",
  brute: {
    name: "Plain Python loops",
    idea: "Each step, loop over the rows to accumulate the gradient of each weight and the bias, then update.",
    code: R`
import math
import numpy as np

def predict_proba(X, w, b):
    return np.array([1 / (1 + math.exp(-(sum(wj * xj for wj, xj in zip(w, row)) + b))) for row in X])

def fit_logreg(X, y, lr, steps):
    n, d = len(X), len(X[0])
    w, b = [0.0] * d, 0.0
    for _ in range(steps):
        gw, gb = [0.0] * d, 0.0
        for i in range(n):
            z = sum(w[j] * X[i][j] for j in range(d)) + b
            p = 1 / (1 + math.exp(-z))
            for j in range(d):
                gw[j] += (p - y[i]) * X[i][j] / n
            gb += (p - y[i]) / n
        w = [w[j] - lr * gw[j] for j in range(d)]
        b -= lr * gb
    return np.array(w), b`,
    tc: "O(steps·n·d)", tw: "Loops over rows and features each step.", sc: "O(d)", sw: "Weights and gradients.",
    bad: "Correct but slow: the same sums are two matrix products per step.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Per step: `p = sigmoid(X @ w + b)`, then the gradients `X.T @ (p - y) / n` and `(p - y).mean()`. Two matrix operations replace both loops.",
    steps: ["`p = sigmoid(X @ w + b)` for all rows at once.", "`err = p - y`.", "`w -= lr * X.T @ err / n`, `b -= lr * err.mean()`."],
    code: R`
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)

def fit_logreg(X, y, lr, steps):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        err = predict_proba(X, w, b) - y          # prediction minus truth
        w = w - lr * X.T @ err / n                # dL/dw
        b = b - lr * err.mean()                   # dL/db
    return w, float(b)`,
    tc: "O(steps·n·d)", tw: "Two matrix products per step.", sc: "O(d)", sw: "Weights.",
  },
  run: R`
X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
y = np.array([0, 0, 1, 1])
w, b = fit_logreg(X, y, 0.5, 500)
print(np.round(w, 3).tolist(), round(float(b), 3))
rng = np.random.default_rng(0)
Xa = np.vstack([rng.normal([-1.5, -1.0], 1.0, (60, 2)), rng.normal([1.5, 1.0], 1.0, (60, 2))])
ya = np.array([0] * 60 + [1] * 60)
w, b = fit_logreg(Xa, ya, 0.5, 400)
print(np.round(w, 2).tolist(), round(float(b), 2))
print(round(float(np.mean((predict_proba(Xa, w, b) > 0.5) == ya)), 3))
print(np.round(predict_proba(np.array([[0.0, 0.0], [3.0, 3.0]]), w, b), 3).tolist())`,
  out: R`
[4.875] 0.0
[2.46, 1.32] 0.31
0.942
[0.576, 1.0]`,
  dryOn: "first step, X = [[-2],[-1],[1],[2]], y = [0,0,1,1], lr = 0.5",
  dry: { h: ["quantity", "value"], r: [["p = σ(0)", "[0.5, 0.5, 0.5, 0.5]"], ["p − y", "[0.5, 0.5, −0.5, −0.5]"], ["Xᵀ(p−y)/n", "(−1 −0.5 −0.5 −1)/4 = −0.75"], ["mean(p−y)", "0"], ["new w, b", "w = 0 + 0.5·0.75 = **0.375**, b = 0"]] },
  expl: [
    ["sigmoid(X @ w + b)", "A weighted sum for every row, squeezed into (0, 1)."],
    ["err = predict_proba(X, w, b) - y", "For cross-entropy with a sigmoid, the gradient w.r.t. the score is simply p − y."],
    ["X.T @ err / n", "Average of (error × feature) over the rows: the weight gradient."],
  ],
  take: "**Logistic regression = sigmoid(linear score), trained by gradient descent; the gradient is Xᵀ(p − y)/n.**",
  mist: ["Putting the sigmoid in twice, or forgetting it, so `p` is not a probability.", "Computing the loss with `log(0)`: clip probabilities or skip the loss when training by gradients.", "Using unscaled features with a large learning rate: the weights explode."],
},
{
  id: "gini-stump", t: "Best Gini Split for a Decision Stump", d: "Hard", p: "Decision trees",
  q: "A **decision stump** is a tree with a single split. Write `best_stump(X, y)` that finds the best split of a binary-labelled dataset and returns `(feature, threshold, weighted_gini)`.\n\nA split `x[feature] <= threshold` sends rows to the left or right. Its score is the size-weighted **Gini impurity** of the two sides:\n\n$G = 1 - \\sum_k p_k^2 \\qquad \\text{score} = \\tfrac{n_L}{n}G_L + \\tfrac{n_R}{n}G_R$\n\nCandidate thresholds are the **midpoints between consecutive distinct sorted values** of a feature. Return the split with the lowest score. If several are equal, prefer the lower feature index, then the smaller threshold. Labels are 0 or 1.",
  ex: [["X = [[1],[2],[3],[4],[5],[6]], y = [0,0,0,1,1,1]", "(0, 3.5, 0.0)", "Splitting at 3.5 gives two perfectly pure sides: weighted Gini 0."],
    ["X = [[1],[2],[3],[4]], y = [0,1,0,1]", "(0, 1.5, 0.333)", "No split is perfect; cutting after the first point gives 0 and 0.444 weighted 1/4, 3/4 = 0.333."]],
  c: ["`X` has shape (n, d), n >= 2", "Each feature has at least two distinct values somewhere (a valid split exists)"],
  hint: "For a threshold, count the class-1 rows on each side. With only two classes, Gini is $2p(1-p)$. The efficient way: sort a feature, then use a running (cumulative) count of ones to get the left/right counts for every split position in one pass.",
  brute: {
    name: "Plain Python loops",
    idea: "For every feature and every candidate threshold, split the rows with a list comprehension and compute the impurities from scratch. Track the best.",
    code: R`
import numpy as np

def gini(labels):
    if len(labels) == 0:
        return 0.0
    p = sum(labels) / len(labels)
    return 2 * p * (1 - p)

def best_stump(X, y):
    n, d = X.shape
    best = (None, None, 2.0)
    for f in range(d):
        vals = sorted(set(X[:, f].tolist()))
        for lo, hi in zip(vals[:-1], vals[1:]):
            t = (lo + hi) / 2
            left = [y[i] for i in range(n) if X[i, f] <= t]
            right = [y[i] for i in range(n) if X[i, f] > t]
            score = (len(left) * gini(left) + len(right) * gini(right)) / n
            if score < best[2] - 1e-12:
                best = (f, t, score)
    return best[0], float(best[1]), float(best[2])`,
    tc: "O(d·n²)", tw: "Each of up to n thresholds rescans all n rows.", sc: "O(n)", sw: "The left / right lists.",
    bad: "Quadratic per feature. Real tree learners sort once and sweep, which is what the optimised version does.",
  },
  opt: {
    name: "Sort once, sweep with cumulative counts",
    idea: "Sort the rows by one feature. A cumulative sum of the labels gives the number of ones on the left of every split position at once; the right side is total minus left. Gini of each side follows in vectorized form. Only positions between **distinct** values are valid splits.",
    steps: ["Sort rows by the feature; `ones_left = np.cumsum(y_sorted)[:-1]`, `n_left = np.arange(1, n)`.", "`p_left = ones_left / n_left`, `p_right = (total - ones_left) / (n - n_left)`.", "Weighted Gini `= (n_left·2p_L(1−p_L) + n_right·2p_R(1−p_R)) / n`.", "Mask out positions where the next value equals the current one; take the argmin; threshold = midpoint."],
    code: R`
import numpy as np

def best_stump(X, y):
    n, d = X.shape
    total = y.sum()
    best = (None, None, 2.0)
    for f in range(d):
        order = np.argsort(X[:, f], kind="stable")
        xs, ys = X[order, f], y[order]
        n_left = np.arange(1, n)                               # left sizes 1 .. n-1
        ones_left = np.cumsum(ys)[:-1]                         # class-1 count on the left
        p_l = ones_left / n_left
        p_r = (total - ones_left) / (n - n_left)
        score = (n_left * 2 * p_l * (1 - p_l) + (n - n_left) * 2 * p_r * (1 - p_r)) / n
        score = np.where(xs[1:] > xs[:-1], score, np.inf)      # only split between distinct values
        i = int(np.argmin(score))
        if score[i] < best[2] - 1e-12:
            best = (f, float((xs[i] + xs[i + 1]) / 2), float(score[i]))
    return best[0], best[1], best[2]`,
    tc: "O(d·n log n)", tw: "One sort per feature, then a linear sweep.", sc: "O(n)", sw: "Sorted copies and cumulative counts.",
  },
  run: R`
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]); y = np.array([0, 0, 0, 1, 1, 1])
f, t, g = best_stump(X, y); print(f, t, round(g, 3))
X = np.array([[1.0], [2.0], [3.0], [4.0]]); y = np.array([0, 1, 0, 1])
f, t, g = best_stump(X, y); print(f, t, round(g, 3))
rng = np.random.default_rng(0)
noise = rng.normal(size=(40, 1))
signal = np.concatenate([rng.normal(0, 1, 20), rng.normal(3, 1, 20)])[:, None]
X = np.hstack([noise, signal]); y = np.array([0] * 20 + [1] * 20)
f, t, g = best_stump(X, y); print(f, round(t, 2), round(g, 3))
X = np.array([[1.0, 5.0], [1.0, 6.0], [2.0, 7.0], [2.0, 8.0]]); y = np.array([0, 0, 1, 1])
f, t, g = best_stump(X, y); print(f, t, round(g, 3))`,
  out: R`
0 3.5 0.0
0 1.5 0.333
1 2.15 0.13
0 1.5 0.0`,
  dryOn: "x = [1,2,3,4,5,6], y = [0,0,0,1,1,1]",
  dry: { h: ["threshold", "left ones / size", "right ones / size", "weighted Gini"], r: [["1.5", "0/1", "3/5", "0 + 5/6·0.48 = 0.4"], ["2.5", "0/2", "3/4", "4/6·0.375 = 0.25"], ["**3.5**", "0/3", "3/3", "**0.0**"], ["4.5", "1/4", "2/2", "4/6·0.375 = 0.25"], ["5.5", "2/5", "1/1", "5/6·0.48 = 0.4"]] },
  expl: [
    ["np.cumsum(ys)[:-1]", "After sorting, the running count of ones: entry i is the number of ones among the first i+1 rows, i.e. on the left of split position i."],
    ["p_r = (total - ones_left) / (n - n_left)", "The right side's ones are the total minus the left's."],
    ["2 * p * (1 - p)", "Gini for two classes: 1 − p² − (1−p)² simplifies to 2p(1−p)."],
    ["np.where(xs[1:] > xs[:-1], score, np.inf)", "Ties in the feature value cannot be separated by a threshold, so those positions are not allowed."],
  ],
  take: "**Sort once and sweep with cumulative counts**: every split's impurity is O(1), so a whole feature costs O(n log n). That is how real decision-tree libraries find splits.",
  mist: ["Splitting between equal values (a threshold cannot separate identical feature values).", "Not weighting the two sides by their sizes: tiny pure sides would look perfect.", "Using the data values themselves as thresholds without handling `<=` carefully; midpoints avoid ambiguity."],
},
]);
