/* Practice: NumPy Basics + Linear Algebra + Calculus & Probability (the maths foundations)
   Each problem: q (task), ex (examples), hint, brute (plain Python), opt (vectorized NumPy), run + out (the tests),
   dry (a dry run), expl (code explained line by line), take, mist. Tests print rounded values so floats compare exactly. */
PRAC.add("NumPy Basics", [
{
  id: "one-hot-encoding", t: "One-Hot Encoding", d: "Easy", p: "Indexing",
  q: "Class labels are stored as integers `0 .. K-1`, but models (and the cross-entropy loss) often want a **one-hot** matrix: one row per example, with a `1` in the column of its class and `0` elsewhere.\n\nWrite `one_hot(labels, n_classes)` that takes a 1-D integer array of length `n` and returns an `(n, n_classes)` float array.",
  ex: [["labels = [0, 2, 1], n_classes = 3", "[[1,0,0], [0,0,1], [0,1,0]]", "Row i has its 1 in column labels[i]."],
    ["labels = [1, 1], n_classes = 2", "[[0,1], [0,1]]", "Repeated labels give repeated rows."]],
  c: ["`labels` is a 1-D integer array with values in `[0, n_classes)`", "Return a float array of shape `(len(labels), n_classes)`"],
  hint: "Think of the identity matrix: row `k` of `np.eye(K)` is the one-hot vector for class `k`. How do you pick several rows at once with an integer array?",
  brute: {
    name: "Plain Python loops",
    idea: "Make a matrix of zeros and put a 1 at `[i, labels[i]]` for each example in a loop.",
    code: R`
import numpy as np

def one_hot(labels, n_classes):
    out = np.zeros((len(labels), n_classes))
    for i, k in enumerate(labels):
        out[i, k] = 1.0
    return out`,
    tc: "O(n)", tw: "One assignment per example.", sc: "O(n K)", sw: "The output matrix itself.",
    bad: "Correct and clear, but a Python loop touches one example at a time. With millions of labels the vectorized version is far faster.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Row `k` of the identity matrix `np.eye(K)` is exactly the one-hot vector of class `k`, so indexing the identity matrix with the label array (**fancy indexing**) gathers all the rows in one go.",
    steps: ["Build the `K x K` identity matrix with `np.eye(n_classes)`.", "Index it with the whole label array: `eye[labels]` picks row `labels[i]` for every `i`.", "The result has shape `(n, K)` already."],
    code: R`
import numpy as np

def one_hot(labels, n_classes):
    return np.eye(n_classes)[labels]      # fancy indexing: row k of the identity is the one-hot of class k`,
    tc: "O(n K)", tw: "Copying n rows of length K.", sc: "O(n K)", sw: "The output (plus the K x K identity).",
  },
  run: R`
print(one_hot(np.array([0, 2, 1]), 3).tolist())
print(one_hot(np.array([1, 1]), 2).tolist())
y = np.array([3, 0, 2, 3, 1])
Y = one_hot(y, 4)
print(Y.shape, Y.sum(axis=1).tolist())
print((Y.argmax(axis=1) == y).all())`,
  out: R`
[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
[[0.0, 1.0], [0.0, 1.0]]
(5, 4) [1.0, 1.0, 1.0, 1.0, 1.0]
True`,
  dryOn: "labels = [0, 2, 1], n_classes = 3",
  dry: { h: ["step", "value"], r: [["np.eye(3)", "[[1,0,0],[0,1,0],[0,0,1]]"], ["labels", "[0, 2, 1]"], ["eye[labels]", "rows 0, 2, 1 of the identity"], ["result", "**[[1,0,0],[0,0,1],[0,1,0]]**"]] },
  expl: [
    ["np.eye(n_classes)", "A K x K identity matrix: row k is a vector with a single 1 in position k."],
    ["[labels]", "Indexing with an integer array returns one row per label, in order, so the output has shape (n, K)."],
  ],
  take: "**Row `k` of an identity matrix is the one-hot vector for class `k`**; indexing it with the label array encodes everything at once. `argmax(axis=1)` undoes it.",
  mist: ["Forgetting `n_classes` and using `labels.max() + 1`: a batch missing the highest class would get too few columns.", "Using the wrong axis for `argmax` when decoding (it must be `axis=1`).", "Passing float labels: indexing needs an integer array."],
},
{
  id: "standardize-features", t: "Standardize Features", d: "Easy", p: "Axes & broadcasting",
  q: "Many models train better when every feature has mean 0 and standard deviation 1 (a **z-score**): $z = \\dfrac{x - \\mu}{\\sigma}$, computed **per column**.\n\nWrite `standardize(X)` for a 2-D array `X` of shape `(n_samples, n_features)` that returns the standardized array. Use the population standard deviation (divide by `n`, NumPy's default). If a column is constant (std = 0) leave it as zeros instead of dividing by zero.",
  ex: [["X = [[1, 10], [3, 10], [5, 10]]", "[[-1.2247, 0], [0, 0], [1.2247, 0]]", "Column 0: mean 3, std 1.633. Column 1 is constant, so it becomes zeros."]],
  c: ["`X` is a float array of shape `(n, d)` with `n >= 1`", "Return a new array; do not modify `X`"],
  hint: "`X.mean(axis=0)` gives one mean per column. Subtracting a `(d,)` array from an `(n, d)` array **broadcasts** along the rows. How can you avoid dividing by a std of 0?",
  brute: {
    name: "Plain Python loops",
    idea: "For each column, compute the mean and std with loops, then rewrite every entry of that column.",
    code: R`
import numpy as np

def standardize(X):
    n, d = X.shape
    out = np.zeros((n, d))
    for j in range(d):
        col = [X[i, j] for i in range(n)]
        mu = sum(col) / n
        sd = (sum((v - mu) ** 2 for v in col) / n) ** 0.5
        for i in range(n):
            out[i, j] = 0.0 if sd == 0 else (X[i, j] - mu) / sd
    return out`,
    tc: "O(n d)", tw: "Every entry is visited a few times.", sc: "O(n d)", sw: "The output and a temporary column.",
    bad: "Works, but the nested loops run in slow Python. The NumPy version computes all column statistics with one call each.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "`mean(axis=0)` and `std(axis=0)` give one value per column; broadcasting subtracts and divides them across all rows. Replace zero stds by 1 first so constant columns become 0 after the subtraction.",
    steps: ["Column means `mu = X.mean(axis=0)` and stds `sd = X.std(axis=0)`, each of shape `(d,)`.", "Replace zeros in `sd` by 1 with `np.where` (a constant column minus its mean is 0 anyway).", "Return `(X - mu) / sd`, broadcast over rows."],
    code: R`
import numpy as np

def standardize(X):
    mu = X.mean(axis=0)                 # one mean per column, shape (d,)
    sd = X.std(axis=0)                  # one std per column
    sd = np.where(sd == 0, 1.0, sd)     # constant columns: avoid dividing by zero
    return (X - mu) / sd                # broadcasting: (n,d) - (d,) and (n,d) / (d,)`,
    tc: "O(n d)", tw: "Two reductions and two element-wise operations.", sc: "O(n d)", sw: "The output array.",
  },
  run: R`
print(np.round(standardize(np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])), 4).tolist())
rng = np.random.default_rng(0)
X = rng.normal(5, 3, size=(100, 4)) * [1, 10, 0.1, 100]
Z = standardize(X)
print((np.round(Z.mean(axis=0), 6) + 0.0).tolist())
print(np.round(Z.std(axis=0), 6).tolist())
print(np.round(standardize(np.array([[2.0, 7.0]])), 4).tolist())`,
  out: R`
[[-1.2247, 0.0], [0.0, 0.0], [1.2247, 0.0]]
[0.0, 0.0, 0.0, 0.0]
[1.0, 1.0, 1.0, 1.0]
[[0.0, 0.0]]`,
  dryOn: "X = [[1, 10], [3, 10], [5, 10]]",
  dry: { h: ["step", "column 0", "column 1"], r: [["mean", "3.0", "10.0"], ["std", "1.633", "0 -> replaced by 1"], ["X - mean", "[-2, 0, 2]", "[0, 0, 0]"], ["divide by std", "**[-1.2247, 0, 1.2247]**", "**[0, 0, 0]**"]] },
  expl: [
    ["mu = X.mean(axis=0)", "Average down the rows: one number per feature."],
    ["sd = np.where(sd == 0, 1.0, sd)", "Where a column is constant, use 1 instead of 0 so the division is safe (the numerator is 0 there)."],
    ["return (X - mu) / sd", "Broadcasting stretches the (d,) vectors across all n rows."],
  ],
  take: "**Standardize per column with `axis=0` and let broadcasting do the rest.** Remember to guard against a zero standard deviation, and in a real pipeline compute `mu` and `sd` on the **training** data only.",
  mist: ["Using `axis=1`, which standardizes each row instead of each feature.", "Dividing by zero for constant columns and getting `nan`.", "Fitting the mean and std on the whole dataset including the test set (data leakage, see ML: Train / Validation / Test)."],
},
]);

PRAC.add("Linear Algebra", [
{
  id: "matmul-loops-vs-vectorized", t: "Matrix Multiply: Loops vs @", d: "Easy", p: "Matrix multiplication",
  q: "Implement matrix multiplication yourself, then see how the library does it.\n\n1. `matmul(A, B)`: multiply an `(m, n)` matrix by an `(n, p)` matrix and return the `(m, p)` result. $C_{ij} = \\sum_k A_{ik} B_{kj}$.\n2. `layer(X, W, b)`: a neural network layer for a **batch**: `X` is `(n, d_in)`, `W` is `(d_in, d_out)`, `b` is `(d_out,)`. Return `X @ W + b` with shape `(n, d_out)`.\n\nThe tests compare your `matmul` with NumPy's result.",
  ex: [["A = [[1, 2], [3, 4]], B = [[0, 1], [1, 0]]", "[[2, 1], [4, 3]]", "Row of A dotted with column of B; B swaps the columns of A."],
    ["X: (5, 3), W: (3, 2), b: (2,)", "shape (5, 2)", "One output row per example; b is broadcast over the rows."]],
  c: ["Inner sizes must match: `A.shape[1] == B.shape[0]`", "`matmul` may be written with loops or with `@`; the tests only check the values"],
  hint: "Entry `(i, j)` of the result is the dot product of **row i of A** with **column j of B**. Three nested loops (i, j, k) do it; `@` does the same in compiled code. For `layer`, remember that `b` needs to be added to every row.",
  brute: {
    name: "Three nested Python loops",
    idea: "Loop over rows `i`, columns `j`, and the shared index `k`, accumulating `A[i, k] * B[k, j]`.",
    code: R`
import numpy as np

def matmul(A, B):
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "inner sizes must match"
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

def layer(X, W, b):
    return matmul(X, W) + b`,
    tc: "O(m n p)", tw: "One multiply-add per (i, j, k).", sc: "O(m p)", sw: "The output matrix.",
    bad: "Correct, but the three loops run in the Python interpreter: easily hundreds of times slower than `@` on matrices of a few hundred rows.",
  },
  opt: {
    name: "Vectorized with @",
    idea: "`A @ B` calls an optimized BLAS routine (cache-friendly, multi-threaded). The bias `b` has shape `(d_out,)` and is **broadcast** across the rows of the `(n, d_out)` product.",
    steps: ["`matmul(A, B)` is just `A @ B`.", "`layer` multiplies `X @ W` first: shape `(n, d_out)`.", "Add `b`; NumPy broadcasts the `(d_out,)` vector to every row."],
    code: R`
import numpy as np

def matmul(A, B):
    return A @ B

def layer(X, W, b):
    return X @ W + b          # (n, d_in)(d_in, d_out) -> (n, d_out), then b is broadcast to every row`,
    tc: "O(m n p)", tw: "Same arithmetic, but in optimized compiled code.", sc: "O(m p)", sw: "The output matrix.",
  },
  run: R`
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[0.0, 1.0], [1.0, 0.0]])
print(matmul(A, B).tolist())
print(matmul(B, A).tolist())
rng = np.random.default_rng(0)
P, Q = rng.normal(size=(4, 3)), rng.normal(size=(3, 5))
print(np.allclose(matmul(P, Q), P @ Q))
X, W, b = rng.normal(size=(5, 3)), rng.normal(size=(3, 2)), np.array([0.5, -0.5])
print(layer(X, W, b).shape, np.round(layer(X, W, b)[0], 4).tolist())`,
  out: R`
[[2.0, 1.0], [4.0, 3.0]]
[[3.0, 4.0], [1.0, 2.0]]
True
(5, 2) [-0.5405, -0.6448]`,
  dryOn: "A = [[1, 2], [3, 4]], B = [[0, 1], [1, 0]]",
  dry: { h: ["entry", "row of A", "column of B", "dot product"], r: [["C[0,0]", "[1, 2]", "[0, 1]", "1*0 + 2*1 = 2"], ["C[0,1]", "[1, 2]", "[1, 0]", "1*1 + 2*0 = 1"], ["C[1,0]", "[3, 4]", "[0, 1]", "3*0 + 4*1 = 4"], ["C[1,1]", "[3, 4]", "[1, 0]", "3*1 + 4*0 = 3"]] },
  expl: [
    ["return A @ B", "Matrix multiplication: entry (i, j) is row i of A dotted with column j of B."],
    ["X @ W + b", "All n examples go through the layer at once; b has shape (d_out,) and broadcasts over the n rows."],
  ],
  take: "**Matrix multiplication is dot products of rows with columns; shapes `(m,n)(n,p) -> (m,p)`.** A layer for a whole batch is one `X @ W + b`.",
  mist: ["Using `*` (element-wise) instead of `@` (matrix product).", "Mismatched shapes: write `(m,n)(n,p)` next to the line and check the inner sizes.", "Transposing the wrong operand when the weights are stored as `(d_out, d_in)`."],
},
{
  id: "power-iteration", t: "Power Iteration for the Top Eigenvector", d: "Medium", p: "Eigenvectors",
  q: "The **power iteration** finds the eigenvector with the largest-magnitude eigenvalue of a square matrix $A$ by repeating $v \\leftarrow A v / \\lVert A v \\rVert$. It is how the top principal component of a huge covariance matrix, or PageRank, is computed.\n\nWrite `power_iteration(A, iters=200)` that starts from `v = [1, 1, ..., 1] / sqrt(n)` and returns `(lam, v)`: the estimated eigenvalue $\\lambda = v^\\top A v$ and the unit eigenvector, with its sign fixed so that the **largest-magnitude component is positive**.",
  ex: [["A = [[2, 1], [1, 2]]", "lam = 3.0, v = [0.7071, 0.7071]", "(1, 1) is an eigenvector of this matrix with eigenvalue 3."],
    ["A = [[4, 0], [0, 1]]", "lam = 4.0, v = [1.0, 0.0]", "Diagonal matrix: the top eigenvector is the first axis."]],
  c: ["`A` is square with a dominant eigenvalue (the largest |lambda| is strictly bigger than the others)", "Normalize `v` to length 1 in every iteration"],
  hint: "Multiply by `A`, then divide by the length so the numbers stay bounded. After many rounds the component along the largest eigenvector dominates. The eigenvalue estimate is `v @ A @ v` (the **Rayleigh quotient**, since `v` has length 1).",
  brute: {
    name: "Plain Python loops",
    idea: "Do the matrix-vector product and the normalization with explicit Python loops over lists.",
    code: R`
import math
import numpy as np

def power_iteration(A, iters=200):
    n = len(A)
    v = [1.0 / math.sqrt(n)] * n
    for _ in range(iters):
        w = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]     # w = A v
        norm = math.sqrt(sum(x * x for x in w))
        v = [x / norm for x in w]
    Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
    lam = sum(v[i] * Av[i] for i in range(n))
    big = max(range(n), key=lambda i: abs(v[i]))
    if v[big] < 0:
        v = [-x for x in v]
    return float(lam), np.array(v)`,
    tc: "O(iters n^2)", tw: "A dense matrix-vector product per iteration.", sc: "O(n)", sw: "A couple of length-n vectors.",
    bad: "The loops are fine for 3x3 but crawl on a 1000x1000 matrix; NumPy does each product in compiled code.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Each iteration is one matrix-vector product `A @ v` and one division by `np.linalg.norm`. The eigenvalue is `v @ A @ v`. Flip the sign at the end so the result is unique.",
    steps: ["Start with the normalized all-ones vector.", "Repeat: `v = A @ v; v = v / np.linalg.norm(v)`.", "`lam = v @ A @ v` (Rayleigh quotient).", "If the largest-magnitude component is negative, flip `v` (an eigenvector's sign is arbitrary)."],
    code: R`
import numpy as np

def power_iteration(A, iters=200):
    n = A.shape[0]
    v = np.ones(n) / np.sqrt(n)
    for _ in range(iters):
        v = A @ v
        v = v / np.linalg.norm(v)         # keep the length at 1 so nothing explodes
    lam = float(v @ A @ v)                # Rayleigh quotient = the eigenvalue estimate
    if v[np.argmax(np.abs(v))] < 0:       # fix the arbitrary sign
        v = -v
    return lam, v`,
    tc: "O(iters n^2)", tw: "One dense matrix-vector product per iteration.", sc: "O(n)", sw: "Two length-n vectors.",
  },
  run: R`
lam, v = power_iteration(np.array([[2.0, 1.0], [1.0, 2.0]]))
print(round(lam, 4), np.round(v, 4).tolist())
lam, v = power_iteration(np.array([[4.0, 0.0], [0.0, 1.0]]))
print(round(lam, 4), np.round(v, 4).tolist())
rng = np.random.default_rng(0)
M = rng.normal(size=(5, 5))
S = M @ M.T                                   # symmetric, positive eigenvalues
lam, v = power_iteration(S, iters=500)
w, V = np.linalg.eigh(S)
print(round(lam, 4), round(float(w[-1]), 4))
print(np.allclose(S @ v, lam * v, atol=1e-6))`,
  out: R`
3.0 [0.7071, 0.7071]
4.0 [1.0, 0.0]
8.0793 8.0793
True`,
  dryOn: "A = [[2, 1], [1, 2]]",
  dry: { h: ["iteration", "A v", "normalized v"], r: [["start", "", "[0.7071, 0.7071]"], ["1", "[2.1213, 2.1213]", "[0.7071, 0.7071]"], ["lam", "v^T A v = 3.0", ""]] },
  expl: [
    ["v = np.ones(n) / np.sqrt(n)", "Any starting vector with a component along the top eigenvector works; all-ones is a common choice."],
    ["v = A @ v; v = v / np.linalg.norm(v)", "Apply the matrix, then rescale to length 1. The largest eigen-direction grows fastest, so it takes over."],
    ["lam = float(v @ A @ v)", "For a unit eigenvector, v^T A v = lambda."],
    ["if v[np.argmax(np.abs(v))] < 0: v = -v", "An eigenvector and its negative are equally valid; fix the sign to get a unique answer."],
  ],
  take: "**Repeated multiplication by a matrix pulls any vector towards its dominant eigenvector.** Normalize every step; the convergence speed depends on the ratio |lambda2| / |lambda1|.",
  mist: ["Not normalizing: the vector grows (or shrinks) exponentially and overflows.", "Starting from a vector exactly orthogonal to the top eigenvector (it would never converge to it).", "Comparing eigenvectors without fixing the sign."],
},
]);

PRAC.add("Calculus & Probability", [
{
  id: "entropy-cross-entropy-kl", t: "Entropy, Cross-Entropy & KL Divergence", d: "Easy", p: "Information theory",
  q: "For two probability distributions `p` (the truth) and `q` (a model), 1-D arrays that each add up to 1, implement three functions using logarithms in **base 2** (so the unit is bits):\n\n- `entropy(p)` $= -\\sum_i p_i \\log_2 p_i$\n- `cross_entropy(p, q)` $= -\\sum_i p_i \\log_2 q_i$\n- `kl(p, q)` $= \\sum_i p_i \\log_2 (p_i / q_i)$\n\nBy convention $0 \\cdot \\log 0 = 0$, so entries with $p_i = 0$ must contribute nothing (and must not produce `nan`). You may assume `q_i > 0` wherever `p_i > 0`.",
  ex: [["p = [0.5, 0.5]", "entropy = 1.0", "A fair coin carries exactly 1 bit."],
    ["p = [1, 0], q = [0.5, 0.5]", "cross_entropy = 1.0, kl = 1.0", "The one-hot truth gives -log2 of the probability the model assigned to the true class."]],
  c: ["`p` and `q` are float arrays of the same length summing to 1", "Use `kl = cross_entropy - entropy` or compute it directly; both are fine"],
  hint: "Select only the positions where `p > 0` with a boolean mask before taking the log. KL is just cross-entropy minus entropy.",
  brute: {
    name: "Plain Python loops",
    idea: "Loop over the outcomes and skip any with `p_i == 0`.",
    code: R`
import math
import numpy as np

def entropy(p):
    return -sum(pi * math.log2(pi) for pi in p if pi > 0) + 0.0

def cross_entropy(p, q):
    return -sum(pi * math.log2(qi) for pi, qi in zip(p, q) if pi > 0)

def kl(p, q):
    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q) if pi > 0)`,
    tc: "O(K)", tw: "One pass over the K outcomes.", sc: "O(1)", sw: "Just the running sum.",
    bad: "Perfectly correct for small K. For a batch of thousands of distributions (a vocabulary-sized softmax at every token) you want NumPy.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Build a boolean mask `p > 0`, then compute the sum only over those positions, so `log(0)` is never evaluated. KL follows from the identity `H(p, q) = H(p) + KL(p || q)`.",
    steps: ["`m = p > 0` marks the outcomes that can occur.", "`entropy`: `-(p[m] * np.log2(p[m])).sum()`.", "`cross_entropy`: `-(p[m] * np.log2(q[m])).sum()`.", "`kl = cross_entropy - entropy`."],
    code: R`
import numpy as np

def entropy(p):
    p = np.asarray(p, dtype=float)
    m = p > 0                                   # 0 * log 0 is defined as 0: skip those entries
    return float(-(p[m] * np.log2(p[m])).sum()) + 0.0

def cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = p > 0
    return float(-(p[m] * np.log2(q[m])).sum())

def kl(p, q):
    return cross_entropy(p, q) - entropy(p)     # H(p, q) = H(p) + KL(p || q)`,
    tc: "O(K)", tw: "A mask and a few vectorized passes.", sc: "O(K)", sw: "The masked copies.",
  },
  run: R`
print(entropy([0.5, 0.5]), round(entropy([0.9, 0.1]), 4), entropy([1.0, 0.0]), round(entropy([0.25] * 4), 4))
print(round(cross_entropy([1.0, 0.0], [0.5, 0.5]), 4), round(kl([1.0, 0.0], [0.5, 0.5]), 4))
p, q = [0.8, 0.15, 0.05], [0.4, 0.4, 0.2]
print(round(entropy(p), 4), round(cross_entropy(p, q), 4), round(kl(p, q), 4))
print(round(kl(q, p), 4), round(kl(p, p), 6))`,
  out: R`
1.0 0.469 0.0 2.0
1.0 1.0
0.8842 1.3719 0.4877
0.566 0.0`,
  dryOn: "p = [0.8, 0.15, 0.05], q = [0.4, 0.4, 0.2]",
  dry: { h: ["quantity", "calculation", "value"], r: [["H(p)", "-(0.8 log2 0.8 + 0.15 log2 0.15 + 0.05 log2 0.05)", "0.8842"], ["H(p, q)", "-(0.8 log2 0.4 + 0.15 log2 0.4 + 0.05 log2 0.2)", "1.3719"], ["KL(p||q)", "H(p, q) - H(p)", "**0.4877**"]] },
  expl: [
    ["m = p > 0", "A boolean mask of the outcomes with non-zero probability."],
    ["-(p[m] * np.log2(p[m])).sum()", "Entropy: the average of -log2 p, weighted by p; masked entries contribute 0."],
    ["return cross_entropy(p, q) - entropy(p)", "KL is the extra bits you pay for believing q instead of p."],
  ],
  take: "**Cross-entropy = entropy + KL, and KL is never negative.** With a one-hot `p` it reduces to `-log q[true class]`: the classification loss.",
  mist: ["Evaluating `0 * log(0)` and getting `nan`: mask out zero probabilities first.", "Swapping the arguments: KL(p||q) and KL(q||p) differ.", "Mixing log bases: ML libraries use natural logs (nats), information theory textbooks often use bits."],
},
{
  id: "numerical-gradient-check", t: "Numerical Gradient Check", d: "Medium", p: "Derivatives",
  q: "Hand-derived gradients are easy to get wrong, so we check them against the slow-but-simple **central difference**:\n\n$\\dfrac{\\partial f}{\\partial x_i} \\approx \\dfrac{f(x + h e_i) - f(x - h e_i)}{2h}$\n\nwhere $e_i$ is the unit vector for coordinate $i$. Write:\n\n1. `num_grad(f, x, h=1e-5)`: the numerical gradient of a scalar function `f` at the 1-D float array `x`.\n2. `grad_check(f, grad_f, x)`: the **relative error** $\\dfrac{\\lVert g_{num} - g \\rVert}{\\lVert g_{num} \\rVert + \\lVert g \\rVert}$ between the numerical gradient and the analytic gradient `grad_f(x)`.\n\nThen the tests check a correct gradient (tiny error) and a deliberately wrong one (large error).",
  ex: [["f(x) = x0^2 + 3 x0 x1, at x = [1, 2]", "[8, 3]", "The analytic gradient is (2 x0 + 3 x1, 3 x0) = (8, 3); the numerical one agrees to about 6 digits."]],
  c: ["`f` takes a 1-D array and returns a float", "Do not modify the input `x` (work on copies)"],
  hint: "For each coordinate `i`, make a vector `e` that is zero except for `h` at position `i`. Then `(f(x + e) - f(x - e)) / (2 * h)` is that component of the gradient.",
  brute: {
    name: "Plain Python lists",
    idea: "Copy the point as a Python list for each perturbation, bump one coordinate up and down, and difference.",
    code: R`
import numpy as np

def num_grad(f, x, h=1e-5):
    g = []
    for i in range(len(x)):
        up = [float(v) for v in x]
        dn = [float(v) for v in x]
        up[i] += h
        dn[i] -= h
        g.append((f(np.array(up)) - f(np.array(dn))) / (2 * h))
    return np.array(g)

def grad_check(f, grad_f, x):
    gn, ga = num_grad(f, x), grad_f(x)
    return float(np.linalg.norm(gn - ga) / (np.linalg.norm(gn) + np.linalg.norm(ga)))`,
    tc: "O(d) calls to f", tw: "Two function evaluations per coordinate.", sc: "O(d)", sw: "Temporary copies of x.",
    bad: "Fine, but it rebuilds Python lists for every coordinate. Using one NumPy vector per coordinate is cleaner (and the function evaluations dominate the cost anyway).",
  },
  opt: {
    name: "NumPy perturbation vectors",
    idea: "Use `np.zeros_like(x)` for the perturbation `e`, set `e[i] = h`, and evaluate `f(x + e)` and `f(x - e)`. The relative error uses `np.linalg.norm`.",
    steps: ["Allocate the gradient array `g = np.zeros_like(x)`.", "For each `i` create `e` with `h` at position `i` and compute the central difference.", "`grad_check`: relative error between the numerical and analytic gradients."],
    code: R`
import numpy as np

def num_grad(f, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)      # central difference for coordinate i
    return g

def grad_check(f, grad_f, x):
    gn = num_grad(f, x)
    ga = np.asarray(grad_f(x), dtype=float)
    return float(np.linalg.norm(gn - ga) / (np.linalg.norm(gn) + np.linalg.norm(ga)))`,
    tc: "O(d) calls to f", tw: "Two evaluations of f per coordinate.", sc: "O(d)", sw: "One perturbation vector at a time.",
  },
  run: R`
f = lambda x: x[0] ** 2 + 3 * x[0] * x[1]
good = lambda x: np.array([2 * x[0] + 3 * x[1], 3 * x[0]])
bad = lambda x: np.array([2 * x[0] + 3 * x[1], 2 * x[0]])        # wrong second component
x = np.array([1.0, 2.0])
print(np.round(num_grad(f, x), 4).tolist())
print(grad_check(f, good, x) < 1e-7, grad_check(f, bad, x) > 1e-2)
g = lambda w: float(np.sum(w ** 3) + np.sin(w[0]) * w[1])
gg = lambda w: np.array([3 * w[0] ** 2 + np.cos(w[0]) * w[1], 3 * w[1] ** 2 + np.sin(w[0]), 3 * w[2] ** 2])
w = np.array([0.3, -1.2, 0.8])
print(np.round(num_grad(g, w), 4).tolist(), grad_check(g, gg, w) < 1e-7)`,
  out: R`
[8.0, 3.0]
True True
[-0.8764, 4.6155, 1.92] True`,
  dryOn: "f(x) = x0^2 + 3 x0 x1, x = [1, 2]",
  dry: { h: ["i", "f(x + e)", "f(x - e)", "(difference) / 2h"], r: [["0", "f(1.00001, 2)", "f(0.99999, 2)", "8.0"], ["1", "f(1, 2.00001)", "f(1, 1.99999)", "3.0"]] },
  expl: [
    ["e = np.zeros_like(x); e[i] = h", "A perturbation that nudges only coordinate i."],
    ["(f(x + e) - f(x - e)) / (2 * h)", "Central difference: the slope of f along coordinate i."],
    ["norm(gn - ga) / (norm(gn) + norm(ga))", "A scale-free error: about 1e-7 or smaller means the analytic gradient is right."],
  ],
  take: "**Always gradient-check hand-written derivatives with central differences.** A relative error around 1e-7 is great; 1e-2 or more means a bug.",
  mist: ["Using the one-sided difference `(f(x+h) - f(x)) / h`, which is much less accurate.", "Choosing `h` too small (like 1e-12): rounding errors dominate.", "Modifying `x` in place and forgetting to restore it."],
},
{
  id: "gaussian-mle", t: "Gaussian Maximum Likelihood", d: "Medium", p: "MLE",
  q: "Data points $x_1, \\dots, x_n$ are assumed to come from a normal distribution $\\mathcal{N}(\\mu, \\sigma^2)$. The **negative log-likelihood** is\n\n$\\text{NLL}(\\mu, \\sigma) = \\sum_i \\left[ \\tfrac{1}{2}\\log(2\\pi\\sigma^2) + \\dfrac{(x_i - \\mu)^2}{2\\sigma^2} \\right]$\n\nWrite:\n\n1. `gaussian_mle(x)`: return `(mu, sigma)` that minimise the NLL. (They have a closed form: the sample mean and the **population** standard deviation, dividing by `n`.)\n2. `nll(x, mu, sigma)`: the negative log-likelihood above.\n\nThe tests check that your MLE has a lower NLL than nearby parameter values.",
  ex: [["x = [1, 2, 3, 4, 5]", "mu = 3.0, sigma = 1.4142", "Mean 3; variance (4+1+0+1+4)/5 = 2, so sigma = sqrt 2."]],
  c: ["`x` is a 1-D float array with at least 2 distinct values", "`sigma > 0`"],
  hint: "Set the derivative of the NLL to zero: the best mean is the average, and the best variance is the average squared deviation from that mean. `np.mean` and `np.std` (with the default `ddof=0`) do exactly this.",
  brute: {
    name: "Plain Python loops",
    idea: "Compute the mean and variance with loops, and accumulate the NLL term by term.",
    code: R`
import math
import numpy as np

def gaussian_mle(x):
    n = len(x)
    mu = sum(x) / n
    var = sum((v - mu) ** 2 for v in x) / n          # divide by n, not n - 1
    return float(mu), math.sqrt(var)

def nll(x, mu, sigma):
    total = 0.0
    for v in x:
        total += 0.5 * math.log(2 * math.pi * sigma ** 2) + (v - mu) ** 2 / (2 * sigma ** 2)
    return float(total)`,
    tc: "O(n)", tw: "A couple of passes over the data.", sc: "O(1)", sw: "A few running sums.",
    bad: "Correct, but each pass is a Python loop. NumPy's mean, std and element-wise operations do the same in one line each.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "The closed-form MLE is `x.mean()` and `x.std()` (population version, `ddof=0`). The NLL is a sum of an element-wise expression.",
    steps: ["`mu = x.mean()` and `sigma = x.std()` (divides by n).", "`nll`: `0.5 * log(2 pi sigma^2)` plus the squared residuals over `2 sigma^2`, summed over all points."],
    code: R`
import numpy as np

def gaussian_mle(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std())       # std divides by n (ddof=0): that is the MLE

def nll(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * np.log(2 * np.pi * sigma ** 2) + (x - mu) ** 2 / (2 * sigma ** 2)))`,
    tc: "O(n)", tw: "Vectorized passes over the data.", sc: "O(n)", sw: "Temporary arrays.",
  },
  run: R`
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
mu, s = gaussian_mle(x)
print(round(mu, 4), round(s, 4), round(nll(x, mu, s), 4))
rng = np.random.default_rng(0)
x = rng.normal(4.0, 1.5, size=500)
mu, s = gaussian_mle(x)
print(round(mu, 3), round(s, 3))
best = nll(x, mu, s)
print(all(best < nll(x, mu + dm, s + ds) for dm, ds in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1), (0.05, 0.05)]))
print(round(best, 2))`,
  out: R`
3.0 1.4142 8.8276
3.96 1.52
True
918.96`,
  dryOn: "x = [1, 2, 3, 4, 5]",
  dry: { h: ["quantity", "calculation", "value"], r: [["mu", "(1+2+3+4+5)/5", "3.0"], ["variance", "(4+1+0+1+4)/5", "2.0"], ["sigma", "sqrt 2", "1.4142"], ["NLL", "5 * 0.5 log(2 pi 2) + 10 / (2 * 2)", "**8.8276**"]] },
  expl: [
    ["x.mean(), x.std()", "The maximum likelihood mean and standard deviation of a Gaussian (np.std defaults to dividing by n)."],
    ["0.5 * np.log(2 * np.pi * sigma ** 2)", "The normalizing term of the Gaussian density, per data point."],
    ["(x - mu) ** 2 / (2 * sigma ** 2)", "The squared-error term: this is why Gaussian noise leads to least squares."],
  ],
  take: "**Gaussian MLE = sample mean and the divide-by-n standard deviation.** The squared-error term of the NLL is why minimising MSE equals maximum likelihood under Gaussian noise.",
  mist: ["Using `ddof=1` (divide by n - 1): that is the unbiased estimate of the variance, not the maximum likelihood one.", "Forgetting the `log(sigma)` term when comparing NLLs for different sigma.", "Using a natural log in one place and log10 in another."],
},
]);
