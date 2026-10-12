/* Practice: Neural Net Basics + Training (deep learning part 1)
   Each problem: q, ex, hint, brute (plain Python), opt (NumPy), run + out (tests), dry, expl, take, mist. */
PRAC.add("Neural Net Basics", [
{
  id: "activations-derivatives", t: "Activations and Their Derivatives", d: "Easy", p: "Activations",
  q: "Every layer of a neural network applies an **activation function**, and backpropagation needs its **derivative**. Write `activate(name, x)` that returns the pair `(f, df)`: the activation values and the derivative values, both as arrays with the same shape as `x`.\n\nSupport four names:\n\n- `\"sigmoid\"`: $f(x)=1/(1+e^{-x})$, $f'(x)=f(x)(1-f(x))$\n- `\"tanh\"`: $f'(x)=1-\\tanh^2(x)$\n- `\"relu\"`: $f(x)=\\max(0,x)$, $f'(x)=1$ for $x>0$, else $0$\n- `\"leaky_relu\"` (slope 0.01): $f(x)=x$ for $x>0$, else $0.01x$; $f'(x)=1$ for $x>0$, else $0.01$\n\nThe sigmoid must not overflow for very large $|x|$ such as $\\pm1000$.",
  ex: [["activate(\"sigmoid\", [-2, 0, 3])", "f = [0.119, 0.5, 0.953], df = [0.105, 0.25, 0.045]", "σ(0) = 0.5 and its slope there is 0.25, the maximum."],
    ["activate(\"relu\", [-2, 0, 3])", "f = [0, 0, 3], df = [0, 0, 1]", "We define the slope at exactly 0 to be 0."]],
  c: ["`x` is a list or 1-D/2-D array of floats", "Return NumPy float arrays", "`name` is one of the four above"],
  hint: "Compute `f` first and reuse it for the derivative: sigmoid' needs only `f * (1 - f)`, tanh' only `1 - f**2`. For a stable sigmoid, only ever exponentiate a **non-positive** number: use `e = np.exp(-np.abs(x))`.",
  brute: {
    name: "Plain Python loops",
    idea: "Loop over the elements with the `math` module. Correct, but it handles one number at a time and needs separate code to cope with overflow.",
    code: R`
import math
import numpy as np

def activate(name, x):
    xs = np.asarray(x, dtype=float)
    f = np.zeros(xs.shape); df = np.zeros(xs.shape)
    for idx in np.ndindex(xs.shape):
        v = float(xs[idx])
        if name == "sigmoid":
            e = math.exp(-abs(v))                       # never exp of a positive number
            s = 1 / (1 + e) if v >= 0 else e / (1 + e)
            f[idx], df[idx] = s, s * (1 - s)
        elif name == "tanh":
            t = math.tanh(v)
            f[idx], df[idx] = t, 1 - t * t
        elif name == "relu":
            f[idx], df[idx] = (v, 1.0) if v > 0 else (0.0, 0.0)
        elif name == "leaky_relu":
            f[idx], df[idx] = (v, 1.0) if v > 0 else (0.01 * v, 0.01)
        else:
            raise ValueError(name)
    return f, df`,
    tc: "O(n)", tw: "One Python-level pass over n elements.", sc: "O(n)", sw: "Output arrays.",
    bad: "Fine for a handful of numbers, but a real layer has millions of activations per batch: Python loops are 100x slower than one NumPy call.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Each activation is a one-line array expression, and its derivative is written in terms of the already-computed output. The stable sigmoid uses $e=\\exp(-|x|)$, which is always in (0, 1].",
    steps: ["Convert `x` to a float array.", "`sigmoid`: `e = exp(-|x|)`, `f = where(x >= 0, 1/(1+e), e/(1+e))`, `df = f(1-f)`.", "`tanh`: `f = tanh(x)`, `df = 1 - f²`.", "`relu` / `leaky_relu`: use a boolean mask `x > 0` with `where`."],
    code: R`
import numpy as np

def activate(name, x):
    x = np.asarray(x, dtype=float)
    if name == "sigmoid":
        e = np.exp(-np.abs(x))                          # in (0, 1]: no overflow
        f = np.where(x >= 0, 1 / (1 + e), e / (1 + e))
        return f, f * (1 - f)
    if name == "tanh":
        f = np.tanh(x)
        return f, 1 - f ** 2
    if name == "relu":
        return np.maximum(0.0, x), (x > 0).astype(float)
    if name == "leaky_relu":
        return np.where(x > 0, x, 0.01 * x), np.where(x > 0, 1.0, 0.01)
    raise ValueError(name)`,
    tc: "O(n)", tw: "A constant number of vectorized passes.", sc: "O(n)", sw: "Output arrays.",
    viz: "dl1_act",
  },
  run: R`
x = np.array([-2.0, 0.0, 3.0])
for name in ["sigmoid", "tanh", "relu", "leaky_relu"]:
    f, df = activate(name, x)
    print(name, np.round(f, 3).tolist(), np.round(df, 3).tolist())
h = 1e-6
num = (activate("tanh", x + h)[0] - activate("tanh", x - h)[0]) / (2 * h)
print("tanh derivative matches finite differences:", np.allclose(num, activate("tanh", x)[1], atol=1e-6))
print("sigmoid at +-1000:", activate("sigmoid", [-1000.0, 1000.0])[0].tolist())`,
  out: R`
sigmoid [0.119, 0.5, 0.953] [0.105, 0.25, 0.045]
tanh [-0.964, 0.0, 0.995] [0.071, 1.0, 0.01]
relu [0.0, 0.0, 3.0] [0.0, 0.0, 1.0]
leaky_relu [-0.02, 0.0, 3.0] [0.01, 0.01, 1.0]
tanh derivative matches finite differences: True
sigmoid at +-1000: [0.0, 1.0]`,
  dryOn: "sigmoid, x = 0",
  dry: { h: ["step", "value"], r: [["e = exp(-|0|)", "1"], ["x >= 0, so f = 1/(1+e)", "0.5"], ["df = f(1 - f)", "0.5 · 0.5 = **0.25**"]] },
  expl: [
    ["e = np.exp(-np.abs(x))", "Always exponentiates a non-positive number, so it can never overflow."],
    ["np.where(x >= 0, 1 / (1 + e), e / (1 + e))", "Two algebraically equal forms of the sigmoid; pick the safe one for the sign of x."],
    ["f * (1 - f)", "The sigmoid's slope written with its own output: no second exp needed."],
    ["(x > 0).astype(float)", "ReLU's slope is a 0/1 mask of where the input was positive."],
  ],
  take: "**Write each derivative in terms of the forward output** (`f(1-f)`, `1-f²`, a mask): backprop already has `f` stored, so the derivative is nearly free.",
  mist: ["Using `1 / (1 + np.exp(-x))` directly: overflows for large negative `x` (a warning, and `inf` intermediate values).", "Forgetting the derivative of tanh is `1 - tanh²`, not `1 - tanh`.", "Using the pre-activation `z` instead of the output in `f(1-f)`."],
},
{
  id: "dense-layer-forward-backward", t: "A Dense Layer: Forward and Backward", d: "Medium", p: "Backpropagation",
  q: "A **dense (fully connected) layer** computes `out = X @ W + b`. Write the two halves of its implementation.\n\n`dense_forward(X, W, b)` returns `out`.\n\n`dense_backward(X, W, dout)` receives `dout`, the gradient of the loss with respect to `out`, and returns the three gradients `(dX, dW, db)`. Shapes: `X` is (n, d_in), `W` is (d_in, d_out), `b` is (d_out,), `dout` is (n, d_out).\n\nThe gradient of a quantity always has the **same shape** as the quantity. Do not average over the batch: just sum, as the chain rule says.",
  ex: [["X = [[1, 2]], W = [[1, 0, -1], [2, 1, 0]], b = [0, 0, 0]", "out = [[5, 2, -1]]", "1·1 + 2·2 = 5, 1·0 + 2·1 = 2, 1·(-1) + 2·0 = -1."],
    ["same, dout = [[1, 1, 1]]", "dX = [[0, 3]], dW = [[1, 1, 1], [2, 2, 2]], db = [1, 1, 1]", "dX = dout Wᵀ; dW = Xᵀ dout; db = column sums of dout."]],
  c: ["`X`: (n, d_in) float array", "Return `dX` of shape (n, d_in), `dW` of shape (d_in, d_out), `db` of shape (d_out,)"],
  hint: "Think of the shapes. `dW` must be (d_in, d_out): which product of `X` (n, d_in) and `dout` (n, d_out) gives that? And `dX` must be (n, d_in): `dout` times which transpose?",
  brute: {
    name: "Plain Python loops",
    idea: "Write every sum out as loops over examples and units, straight from the definition: `out[i,j] = sum_k X[i,k] W[k,j] + b[j]`, and differentiate each term by hand.",
    code: R`
import numpy as np

def dense_forward(X, W, b):
    n, d_in = X.shape
    d_out = W.shape[1]
    out = np.zeros((n, d_out))
    for i in range(n):
        for j in range(d_out):
            s = b[j]
            for k in range(d_in):
                s += X[i, k] * W[k, j]
            out[i, j] = s
    return out

def dense_backward(X, W, dout):
    n, d_in = X.shape
    d_out = W.shape[1]
    dX, dW, db = np.zeros_like(X), np.zeros_like(W), np.zeros(d_out)
    for i in range(n):
        for j in range(d_out):
            db[j] += dout[i, j]                       # out[i,j] depends on b[j] with slope 1
            for k in range(d_in):
                dW[k, j] += X[i, k] * dout[i, j]      # slope w.r.t. W[k,j] is X[i,k]
                dX[i, k] += W[k, j] * dout[i, j]      # slope w.r.t. X[i,k] is W[k,j]
    return dX, dW, db`,
    tc: "O(n·d_in·d_out)", tw: "Three nested loops in Python.", sc: "O(n·d_in)", sw: "The gradient arrays.",
    bad: "Correct and a great way to see where each term comes from, but each loop iteration costs microseconds in Python; matrix products run in optimised C/BLAS.",
  },
  opt: {
    name: "Matrix products",
    idea: "The three loops are matrix products. The forward pass is `X @ W + b`; the backward pass is `dout @ W.T`, `X.T @ dout` and `dout.sum(axis=0)`. Each shape works out automatically.",
    steps: ["Forward: `X @ W + b` (the bias broadcasts over rows).", "`dX = dout @ W.T`  (n, d_out) · (d_out, d_in) = (n, d_in).", "`dW = X.T @ dout`  (d_in, n) · (n, d_out) = (d_in, d_out).", "`db = dout.sum(axis=0)`: the bias is added to every row, so its gradient is the sum over rows."],
    code: R`
import numpy as np

def dense_forward(X, W, b):
    return X @ W + b

def dense_backward(X, W, dout):
    dX = dout @ W.T
    dW = X.T @ dout
    db = dout.sum(axis=0)
    return dX, dW, db`,
    tc: "O(n·d_in·d_out)", tw: "Three matrix products.", sc: "O(n·d_in + d_in·d_out)", sw: "The gradients.",
    viz: "dl1_netpass",
  },
  run: R`
X = np.array([[1.0, 2.0]]); W = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]); b = np.zeros(3)
print(dense_forward(X, W, b).tolist())
dX, dW, db = dense_backward(X, W, np.ones((1, 3)))
print(dX.tolist(), dW.tolist(), db.tolist())

rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3)); W = rng.normal(size=(3, 2)); b = rng.normal(size=2); C = rng.normal(size=(4, 2))
dX, dW, db = dense_backward(X, W, C)                 # for L = sum(out * C), dL/dout = C
print(dX.shape, dW.shape, db.shape)
L = lambda X, W, b: float((dense_forward(X, W, b) * C).sum())
h, numW = 1e-6, np.zeros_like(W)
for i in range(3):
    for j in range(2):
        Wp, Wm = W.copy(), W.copy(); Wp[i, j] += h; Wm[i, j] -= h
        numW[i, j] = (L(X, Wp, b) - L(X, Wm, b)) / (2 * h)
print("dW matches finite differences:", np.allclose(numW, dW, atol=1e-5))`,
  out: R`
[[5.0, 2.0, -1.0]]
[[0.0, 3.0]] [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]] [1.0, 1.0, 1.0]
(4, 3) (3, 2) (2,)
dW matches finite differences: True`,
  dryOn: "X = [[1, 2]], W = [[1, 0, -1], [2, 1, 0]], dout = [[1, 1, 1]]",
  dry: { h: ["gradient", "formula", "result"], r: [["dX", "dout @ Wᵀ = [1, 1, 1] · [[1, 2], [0, 1], [-1, 0]]", "[0, 3]"], ["dW", "Xᵀ @ dout = [[1], [2]] · [[1, 1, 1]]", "[[1, 1, 1], [2, 2, 2]]"], ["db", "sum of dout over rows", "[1, 1, 1]"]] },
  expl: [
    ["X @ W + b", "Every output neuron is a weighted sum of the inputs, plus its bias."],
    ["dout @ W.T", "Each input affects each output through the weight between them; sum those contributions: the error signal sent to the layer below."],
    ["X.T @ dout", "A weight's gradient is (its input) × (error signal at its output), summed over the batch."],
    ["dout.sum(axis=0)", "The bias is shared by every example, so add up all the error signals."],
  ],
  take: "**dX = dout·Wᵀ, dW = Xᵀ·dout, db = Σ dout.** Let the shapes tell you the order of the product: each gradient must have the shape of what it differentiates.",
  mist: ["Using `W @ dout` or forgetting a transpose: a shape error, or silently wrong when the matrix is square.", "Dividing by the batch size here: that belongs in the loss gradient, once.", "Forgetting to sum `db` over the batch axis (returning an (n, d_out) array)."],
},
{
  id: "softmax-ce-gradient", t: "Softmax + Cross-Entropy Loss and Gradient", d: "Medium", p: "Losses",
  q: "Write `softmax_ce(logits, y)` for a classifier. `logits` has shape (n, K) (raw scores) and `y` holds the integer class of each example. Return `(loss, grad)`:\n\n- `loss`: the **mean** cross-entropy $-\\frac1n\\sum_i \\ln p_{i,y_i}$ where $p=\\mathrm{softmax}(\\text{logits})$ per row.\n- `grad`: the gradient of that mean loss with respect to `logits`, same shape (n, K).\n\nThe key fact: for softmax followed by cross-entropy, $\\partial L/\\partial z = (p - \\text{onehot}(y))/n$. Your function must stay finite for huge logits such as 1000 (use the max-subtraction / log-sum-exp trick).",
  ex: [["logits = [[2, 1, 0.1]], y = [0]", "loss 0.4170, grad [[-0.3410, 0.2424, 0.0986]]", "p = [0.659, 0.242, 0.099]; the true class gets p − 1 = −0.341."],
    ["logits = [[1000, 0, -1000]], y = [0]", "loss 0.0", "A naive exp(1000) overflows; subtracting the row maximum fixes it."]],
  c: ["`logits`: (n, K) float array, `y`: (n,) integer array with values in [0, K)", "Return a Python float and an (n, K) array"],
  hint: "Subtract each row's max before `exp`. Use `log p = z - logsumexp(z)` for the loss. For the gradient, copy `p`, subtract 1 at `[i, y[i]]` with fancy indexing, then divide by n.",
  brute: {
    name: "Plain Python loops",
    idea: "For each example: stabilise, compute softmax with a loop, take the log of the true class, and build its gradient row. Same maths, one example at a time.",
    code: R`
import math
import numpy as np

def softmax_ce(logits, y):
    n, K = logits.shape
    total = 0.0
    grad = np.zeros((n, K))
    for i in range(n):
        m = max(logits[i])
        e = [math.exp(v - m) for v in logits[i]]
        s = sum(e)
        p = [v / s for v in e]
        total += -math.log(p[y[i]])
        for k in range(K):
            grad[i, k] = (p[k] - (1.0 if k == y[i] else 0.0)) / n
    return total / n, grad`,
    tc: "O(n·K)", tw: "Python loops over examples and classes.", sc: "O(n·K)", sw: "The gradient array.",
    bad: "Correct, but loops over every example and class in Python.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Subtract the row max, exponentiate, normalise to get `p`. The loss picks `p[i, y[i]]` with fancy indexing; the gradient is `p` minus a one-hot matrix, divided by n. Doing the `-1` with fancy indexing avoids building the one-hot matrix.",
    steps: ["`z = logits - logits.max(axis=1, keepdims=True)`: shift for stability.", "`p = exp(z) / sum(exp(z))` per row.", "`loss = -mean(log(p[arange(n), y]))`.", "`grad = p`, subtract 1 at the true classes, divide by n."],
    code: R`
import numpy as np

def softmax_ce(logits, y):
    n = len(y)
    z = logits - logits.max(axis=1, keepdims=True)       # shift: avoids overflow
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(p[np.arange(n), y]))
    grad = p.copy()
    grad[np.arange(n), y] -= 1                           # p - onehot
    return float(loss), grad / n`,
    tc: "O(n·K)", tw: "A few vectorized passes.", sc: "O(n·K)", sw: "p and the gradient.",
  },
  run: R`
loss, g = softmax_ce(np.array([[2.0, 1.0, 0.1]]), np.array([0]))
print(round(loss, 4), np.round(g, 4).tolist())
loss, g = softmax_ce(np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]), np.array([0, 1]))
print(round(loss, 4), np.round(g, 3).tolist())
print(round(abs(softmax_ce(np.array([[1000.0, 0.0, -1000.0]]), np.array([0]))[0]), 4))
print(np.allclose(g.sum(axis=1), 0))                     # every row of the gradient sums to 0`,
  out: R`
0.417 [[-0.341, 0.2424, 0.0986]]
0.3185 [[-0.17, 0.121, 0.049], [0.054, -0.099, 0.044]]
0.0
True`,
  dryOn: "logits = [[2, 1, 0.1]], y = [0]",
  dry: { h: ["step", "value"], r: [["subtract max (2)", "[0, -1, -1.9]"], ["exp", "[1, 0.368, 0.150]"], ["p = exp / sum (1.518)", "[0.659, 0.242, 0.099]"], ["loss = -ln p[0]", "-ln 0.659 = **0.417**"], ["grad = p - onehot", "[0.659 - 1, 0.242, 0.099] = **[-0.341, 0.242, 0.099]**"]] },
  expl: [
    ["logits - logits.max(axis=1, keepdims=True)", "Softmax is unchanged when you subtract a constant per row; this keeps every exponent at most 0."],
    ["p[np.arange(n), y]", "Picks, for each row i, the probability given to its true class."],
    ["grad[np.arange(n), y] -= 1", "Subtracting a one-hot vector without ever building it."],
    ["grad / n", "The loss is a mean, so each example's gradient carries a 1/n."],
  ],
  take: "**For softmax + cross-entropy, ∂L/∂logits = (p − onehot) / n.** It is the same 'prediction minus truth' as regression, and each row sums to 0.",
  mist: ["Forgetting the max-subtraction: `exp(1000)` overflows to `inf`.", "Computing `np.log(softmax(z))`: underflowed probabilities become `log(0) = -inf`.", "Dividing the gradient by n but not the loss (or vice versa)."],
},
{
  id: "xor-backprop", t: "Train a 2-Layer Net on XOR with Backprop", d: "Hard", p: "Backpropagation",
  q: "A single neuron cannot learn XOR, but a network with one hidden layer can. Write `train_xor(hidden=8, lr=0.5, epochs=3000, seed=0)` that trains a 2-layer network with **hand-written backprop** and returns `(probs, loss)`:\n\n- the data is the four XOR inputs `[[0,0],[0,1],[1,0],[1,1]]` with targets `[0,1,1,0]`\n- hidden layer: `tanh`; output layer: one **sigmoid** unit with binary cross-entropy loss\n- initialise `W1` (2, hidden) and `W2` (hidden, 1) with `rng.normal(size=...)` from `rng = np.random.default_rng(seed)` (in that order), biases zero\n- full-batch gradient descent: forward, loss, backward, update, `epochs` times\n\n`probs` is the predicted probability for each of the four inputs after training (shape (4,)) and `loss` is the final mean cross-entropy.",
  ex: [["train_xor()", "predicted classes [0, 1, 1, 0], loss below 0.05", "The hidden layer lets the network bend the decision boundary around the two diagonals."]],
  c: ["Use only NumPy", "For sigmoid + BCE the gradient with respect to the output pre-activation is `(p - y) / n`"],
  hint: "Backward pass: `dz2 = (p - y)/n`; `dW2 = A1.T @ dz2`; `dA1 = dz2 @ W2.T`; `dz1 = dA1 * (1 - A1**2)` (tanh'); `dW1 = X.T @ dz1`; biases are column sums.",
  brute: {
    name: "Per-example loops",
    idea: "Process the four examples one at a time inside each epoch, accumulate the gradients by hand, then update. Same full-batch gradient, written with loops.",
    code: R`
import numpy as np

def train_xor(hidden=8, lr=0.5, epochs=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [0, 1, 1, 0]
    W1 = rng.normal(size=(2, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(size=(hidden, 1)); b2 = np.zeros(1)
    sig = lambda z: 1 / (1 + np.exp(-z))
    for _ in range(epochs):
        gW1, gb1, gW2, gb2 = np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2)
        for x, y in zip(X, Y):
            x = np.array(x, dtype=float)
            a1 = np.tanh(x @ W1 + b1)
            p = sig(a1 @ W2 + b2)
            dz2 = (p - y) / 4
            gW2 += np.outer(a1, dz2); gb2 += dz2
            dz1 = (W2[:, 0] * dz2[0]) * (1 - a1 ** 2)
            gW1 += np.outer(x, dz1); gb1 += dz1
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    Xa = np.array(X, dtype=float)
    p = sig(np.tanh(Xa @ W1 + b1) @ W2 + b2).ravel()
    y = np.array(Y)
    return p, float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))`,
    tc: "O(epochs·n·hidden)", tw: "A Python loop over examples inside the epoch loop.", sc: "O(hidden)", sw: "Weights.",
    bad: "Works, but loops over examples in Python and builds many tiny arrays. With a real dataset the batch must be one matrix.",
  },
  opt: {
    name: "Vectorized full-batch backprop",
    idea: "Put all four examples in one matrix `X` of shape (4, 2). The forward and backward passes are a handful of matrix products, exactly the dense-layer formulas.",
    steps: ["Forward: `A1 = tanh(X @ W1 + b1)`, `p = sigmoid(A1 @ W2 + b2)`.", "Backward: `dz2 = (p - y)/n`, `dW2 = A1.T @ dz2`, `dz1 = (dz2 @ W2.T) * (1 - A1²)`, `dW1 = X.T @ dz1`.", "Update all parameters by `-lr * gradient`.", "Return the final probabilities and BCE loss."],
    code: R`
import numpy as np

def train_xor(hidden=8, lr=0.5, epochs=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)
    W1 = rng.normal(size=(2, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(size=(hidden, 1)); b2 = np.zeros(1)
    sig = lambda z: 1 / (1 + np.exp(-z))
    for _ in range(epochs):
        A1 = np.tanh(X @ W1 + b1)                        # forward
        p = sig(A1 @ W2 + b2).ravel()
        dz2 = ((p - y) / len(y))[:, None]                # backward
        dW2, db2 = A1.T @ dz2, dz2.sum(axis=0)
        dz1 = (dz2 @ W2.T) * (1 - A1 ** 2)
        dW1, db1 = X.T @ dz1, dz1.sum(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1                   # update
        W2 -= lr * dW2; b2 -= lr * db2
    p = sig(np.tanh(X @ W1 + b1) @ W2 + b2).ravel()
    return p, float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))`,
    tc: "O(epochs·hidden)", tw: "A few small matrix products per epoch.", sc: "O(hidden)", sw: "Weights.",
  },
  run: R`
p, loss = train_xor()
print(np.round(p).astype(int).tolist(), loss < 0.05)
p2, loss2 = train_xor(hidden=4, lr=0.5, epochs=3000, seed=1)
print(np.round(p2).astype(int).tolist(), loss2 < 0.05)
p3, _ = train_xor(epochs=0)
print("before training:", np.round(p3).astype(int).tolist() != [0, 1, 1, 0])`,
  out: R`
[0, 1, 1, 0] True
[0, 1, 1, 0] True
before training: True`,
  dryOn: "one epoch, conceptually",
  dry: { setup: "X is (4, 2); W1 (2, 4); W2 (4, 1)", steps: [["A1 = tanh(X @ W1 + b1)", "(4, 4) hidden activations"], ["p = sigmoid(A1 @ W2 + b2)", "(4,) probabilities"], ["dz2 = (p - y) / 4", "(4, 1) output error signal"], ["dW2 = A1ᵀ @ dz2", "(4, 1)"], ["dz1 = (dz2 @ W2ᵀ) * (1 - A1²)", "(4, 4): error pushed through W2 and the tanh slope"], ["dW1 = Xᵀ @ dz1", "(2, 4)"]], result: "After 3000 epochs the loss is tiny and the outputs are [0, 1, 1, 0]." },
  expl: [
    ["dz2 = ((p - y) / len(y))[:, None]", "Sigmoid + BCE gives the clean error signal `p - y`, averaged over the 4 examples."],
    ["dz1 = (dz2 @ W2.T) * (1 - A1 ** 2)", "Send the error back through the output weights, then multiply by the tanh slope."],
    ["X.T @ dz1", "Weight gradient = input × error signal, summed over examples."],
    ["W1 -= lr * dW1", "Gradient-descent step, in place."],
  ],
  take: "**Backprop for a 2-layer net is five lines**: output error `p − y`, `dW2 = A1ᵀ·dz2`, push back with `W2ᵀ`, multiply by the activation slope, `dW1 = Xᵀ·dz1`.",
  mist: ["Using the tanh output where its derivative needs `1 - A1**2` (not `1 - Z1**2`).", "Initialising all weights to zero: the hidden neurons stay identical and XOR is never solved.", "Skipping the `/ n` in `dz2`: the learning rate effectively changes with the batch size.", "Concluding the code is wrong when a tiny net (`hidden=4`, `seed=0`) gets stuck in a bad local minimum: a few hidden units can fail for an unlucky start. Try more units or another seed."],
},
{
  id: "tiny-autograd", t: "A Tiny Autograd Value", d: "Hard", p: "Autograd",
  q: "Build the core of an automatic-differentiation engine. Complete the `Value` class so that arithmetic on `Value` objects records a graph, and `backward()` fills in `.grad` for every node.\n\nSupport: `a + b`, `a * b` (each side may also be a plain number), `a.tanh()`, and `.backward()`. After `out.backward()`, every `Value` that took part has `.grad` = $\\partial\\,\\text{out}/\\partial\\,\\text{that value}$.\n\nThe backward pass must handle a value that is used **more than once** (gradients from each use are **added**).",
  ex: [["a = Value(2); b = Value(-3); c = Value(10); out = (a*b + c) * Value(-2); out.backward()", "a.grad = 6, b.grad = -4, c.grad = -2", "d = ab + c = 4, out = -2d, so ∂out/∂a = -2·b = 6."],
    ["x = Value(3); y = x*x + x; y.backward()", "x.grad = 7", "x is used three times: 2x + 1 = 7. The three contributions add."]],
  c: ["Pure Python (`math` allowed)", "`Value(data)` stores `.data`, `.grad` (starts at 0.0)", "Numbers on either side of `+` and `*` are wrapped into `Value`s"],
  hint: "Every operation makes a new Value that remembers its parents and a small `_backward` function holding the local derivatives (use `+=`!). For `backward()`, first build a list of all nodes in topological order (parents before children), set the output's grad to 1, then call each node's `_backward` in reverse order.",
  starter: R`
import math

class Value:
    def __init__(self, data, parents=()):
        self.data = data
        self.grad = 0.0
        self._parents = parents
        self._backward = lambda: None

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

    def tanh(self):
        pass

    def backward(self):
        pass
`,
  brute: {
    name: "Path-by-path recursion",
    idea: "Skip the topological sort: push the incoming gradient down to the parents recursively, adding up every path from the output to each node. Correct, but a node shared by many paths is revisited once per path (exponential time in the worst case).",
    code: R`
import math

class Value:
    def __init__(self, data, parents=(), local=()):
        self.data = data
        self.grad = 0.0
        self._parents = parents
        self._local = local                      # local derivative to each parent

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def tanh(self):
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t * t,))

    def backward(self):
        def push(v, upstream):
            v.grad += upstream                   # this path's contribution
            for p, d in zip(v._parents, v._local):
                push(p, upstream * d)
        push(self, 1.0)`,
    tc: "O(number of paths)", tw: "A shared node is expanded once per path to the output.", sc: "O(depth)", sw: "Recursion depth.",
    bad: "Right answers, but the cost grows with the number of *paths*, not nodes: a chain of n shared additions makes 2ⁿ paths. Real engines visit each node once.",
  },
  opt: {
    name: "Topological order",
    idea: "Order the graph so every node comes after its parents (depth-first search). Then walk it in reverse: when a node's `_backward` runs, every node that uses it has already added its share to `.grad`, so the gradient is complete. Each node is processed exactly once.",
    steps: ["Each op creates a new `Value` with its `parents` and a `_backward` closure that does `parent.grad += local_derivative * out.grad`.", "`backward()`: DFS to collect nodes in topological order.", "Set `self.grad = 1.0`.", "Loop over the list in reverse, calling `_backward()`."],
    code: R`
import math

class Value:
    def __init__(self, data, parents=()):
        self.data = data
        self.grad = 0.0
        self._parents = parents
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad                      # d(a+b)/da = 1
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad         # d(a*b)/da = b
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()`,
    tc: "O(nodes)", tw: "Each node is visited once forward and once backward.", sc: "O(nodes)", sw: "The graph.",
  },
  run: R`
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
out = (a * b + c) * Value(-2.0)
out.backward()
print(out.data, a.grad, b.grad, c.grad)

x = Value(3.0)
y = x * x + x
y.backward()
print(y.data, x.grad)

x1, w1, bias = Value(2.0), Value(-3.0), Value(6.8813735870195432)
o = (x1 * w1 + bias).tanh()
o.backward()
print(round(o.data, 4), round(x1.grad, 4), round(w1.grad, 4))

z = Value(0.5)
t = z * 2 + 1
print(t.data)`,
  out: R`
-8.0 6.0 -4.0 -2.0
12.0 7.0
0.7071 -1.5 1.0
2.0`,
  dryOn: "x = Value(3); y = x*x + x",
  dry: { setup: "graph: p = x*x, y = p + x", steps: [["topological order", "[x, p, y]"], ["y.grad = 1", "start"], ["y._backward", "p.grad += 1, x.grad += 1"], ["p._backward", "x.grad += x·p.grad (twice: both factors are x) = 3 + 3 = 6, so x.grad = 1 + 6"]], result: "**x.grad = 7** = 2x + 1." },
  expl: [
    ["self.grad += other.data * out.grad", "Local derivative of a product times the gradient arriving from above (the chain rule). `+=` handles reuse."],
    ["visit(v) ... order.append(v)", "Post-order DFS: a node is appended only after all its parents, so the list is a topological order."],
    ["self.grad = 1.0", "∂out/∂out = 1 seeds the backward pass."],
    ["for v in reversed(order): v._backward()", "Children first, so every node's grad is complete before it is pushed to its parents."],
  ],
  take: "**Autograd = record the graph + a local-derivative closure per op, then run the closures in reverse topological order with `+=`.** This is `loss.backward()` in PyTorch.",
  mist: ["Writing `self.grad = ...` instead of `+=`: a reused value loses all but one contribution.", "Calling `_backward` in forward (not reverse) order: some nodes push gradients before they have received theirs.", "Forgetting to wrap plain numbers (`z * 2`): `AttributeError: 'int' object has no attribute 'data'`."],
},
]);

PRAC.add("Training", [
{
  id: "inverted-dropout", t: "Inverted Dropout", d: "Easy", p: "Regularization",
  q: "Implement **inverted dropout** for a layer's activations.\n\n`dropout_forward(x, p, rng, training)` returns `(out, mask)`. In training mode, drop each element with probability `p`: draw `u = rng.random(x.shape)`, keep an element where `u >= p`, and scale the kept elements by `1/(1-p)` so the expected value does not change. `mask` is the keep-mask already including the scaling (`(u >= p) / (1 - p)`). In evaluation mode return `(x, None)` unchanged.\n\n`dropout_backward(dout, mask)` returns the gradient with respect to `x`.",
  ex: [["x = [1, 1, 1, 1], p = 0.5, a random draw u = [0.1, 0.7, 0.4, 0.9]", "out = [0, 2, 0, 2]", "Elements with u < 0.5 are dropped; survivors are doubled (1/(1-0.5))."],
    ["training = False", "out = x", "Nothing is dropped or scaled at test time."]],
  c: ["`0 <= p < 1`", "Use exactly one call `rng.random(x.shape)` per forward so results are reproducible for a seeded generator"],
  hint: "The mask and the scaling can be one array: `(rng.random(x.shape) >= p) / (1 - p)`. The forward pass is `x * mask`. What is the derivative of `x * mask` with respect to `x`?",
  brute: {
    name: "Plain Python loops",
    idea: "Loop over the elements in order, draw one random number per element and decide whether to keep it. A flat loop is fine because `rng.random(n)` and n successive draws are the same stream.",
    code: R`
import numpy as np

def dropout_forward(x, p, rng, training):
    if not training or p == 0:
        return x, None
    flat = x.ravel()
    mask = np.zeros(flat.shape)
    for i in range(len(flat)):
        u = rng.random()
        mask[i] = (1.0 / (1 - p)) if u >= p else 0.0
    mask = mask.reshape(x.shape)
    return x * mask, mask

def dropout_backward(dout, mask):
    return dout * mask`,
    tc: "O(n)", tw: "A Python loop over the elements.", sc: "O(n)", sw: "The mask.",
    bad: "Correct, but one `rng.random()` call per element is slow; draw the whole array at once.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Draw all the random numbers in one call, compare to `p`, and fold the `1/(1-p)` scaling into the mask. The backward pass multiplies by the same mask, because the layer is element-wise `out = x * mask`.",
    steps: ["If not training: return `x, None`.", "`mask = (rng.random(x.shape) >= p) / (1 - p)`.", "`out = x * mask`.", "Backward: `dout * mask` (dropped units get zero gradient, survivors are scaled the same way)."],
    code: R`
import numpy as np

def dropout_forward(x, p, rng, training):
    if not training or p == 0:
        return x, None
    mask = (rng.random(x.shape) >= p) / (1 - p)      # 0 for dropped, 1/(1-p) for kept
    return x * mask, mask

def dropout_backward(dout, mask):
    return dout * mask`,
    tc: "O(n)", tw: "One random array and one multiply.", sc: "O(n)", sw: "The mask.",
    viz: "dl1_dropout",
  },
  run: R`
rng = np.random.default_rng(0)
x = np.ones((2, 4))
out, mask = dropout_forward(x, 0.5, rng, True)
print(out.tolist())
print(dropout_backward(np.ones((2, 4)), mask).tolist())
print(dropout_forward(x, 0.5, rng, False)[0].tolist())
big = np.ones(100000)
o, _ = dropout_forward(big, 0.3, np.random.default_rng(1), True)
print(round(float(o.mean()), 2), round(float((o == 0).mean()), 2))`,
  out: R`
[[2.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]
[[2.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]
[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
1.0 0.3`,
  dryOn: "x = [1, 1, 1, 1], p = 0.5, u = [0.1, 0.7, 0.4, 0.9]",
  dry: { h: ["i", "u", "u ≥ p ?", "mask", "out"], r: [["0", "0.1", "no", "0", "0"], ["1", "0.7", "yes", "2", "2"], ["2", "0.4", "no", "0", "0"], ["3", "0.9", "yes", "2", "2"]] },
  expl: [
    ["rng.random(x.shape) >= p", "True with probability 1-p: each unit survives independently."],
    ["/ (1 - p)", "Scale survivors up in training, so test time needs no change (this is the 'inverted' part)."],
    ["x * mask", "Dropped units output 0; kept units are scaled."],
    ["dout * mask", "The derivative of `x * mask` w.r.t. `x` is `mask`: no gradient flows through dropped units."],
  ],
  take: "**Inverted dropout multiplies by `mask/(1-p)` in training and does nothing at test time**; the backward pass reuses the same mask.",
  mist: ["Forgetting the `1/(1-p)` scale: activations shrink in training and the model must be rescaled at test time.", "Drawing a new mask in the backward pass: it must be the **same** mask as the forward pass.", "Applying dropout at evaluation time."],
},
{
  id: "sgd-momentum-adam", t: "SGD with Momentum and Adam Steps", d: "Medium", p: "Optimizers",
  q: "Write the single-step update rules of two optimizers. Both take NumPy arrays and **return** the new state (do not modify the inputs).\n\n`sgd_momentum_step(w, g, v, lr, beta)` returns `(w_new, v_new)` with $v \\leftarrow \\beta v + g$, $w \\leftarrow w - \\text{lr}\\cdot v$.\n\n`adam_step(w, g, m, s, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8)` returns `(w_new, m_new, s_new)`. $t$ is the step number starting at **1**:\n\n$m \\leftarrow b_1 m + (1-b_1) g, \\quad s \\leftarrow b_2 s + (1-b_2) g^2$\n\n$\\hat m = m/(1-b_1^t), \\quad \\hat s = s/(1-b_2^t), \\quad w \\leftarrow w - \\text{lr}\\,\\hat m/(\\sqrt{\\hat s}+\\epsilon)$",
  ex: [["adam_step(w=[0, 0], g=[0.001, 1000], m=0, s=0, t=1, lr=0.1)", "w_new ≈ [-0.1, -0.1]", "On step 1 the update is lr·g/|g|: about lr for every parameter, however different the gradient sizes."],
    ["sgd_momentum_step(w=[1], g=[2], v=[0], lr=0.1, beta=0.9)", "w = [0.8], v = [2]", "v = 0.9·0 + 2 = 2; w = 1 − 0.1·2 = 0.8."]],
  c: ["Arrays have the same shape", "Do not mutate the arguments", "Bias correction uses the step number `t` (1-based)"],
  hint: "Compute the new `m` and `s` first, then the corrected `m_hat`, `s_hat` using `t`, then the step. Momentum is just two lines.",
  brute: {
    name: "Plain Python loops",
    idea: "Update each parameter separately with scalar maths inside a loop over the elements.",
    code: R`
import math
import numpy as np

def sgd_momentum_step(w, g, v, lr, beta):
    w_new, v_new = np.zeros(len(w)), np.zeros(len(w))
    for i in range(len(w)):
        v_new[i] = beta * v[i] + g[i]
        w_new[i] = w[i] - lr * v_new[i]
    return w_new, v_new

def adam_step(w, g, m, s, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    n = len(w)
    w_new, m_new, s_new = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        m_new[i] = b1 * m[i] + (1 - b1) * g[i]
        s_new[i] = b2 * s[i] + (1 - b2) * g[i] ** 2
        m_hat = m_new[i] / (1 - b1 ** t)
        s_hat = s_new[i] / (1 - b2 ** t)
        w_new[i] = w[i] - lr * m_hat / (math.sqrt(s_hat) + eps)
    return w_new, m_new, s_new`,
    tc: "O(n)", tw: "A Python loop over the parameters.", sc: "O(n)", sw: "Three new arrays.",
    bad: "Fine for 2 parameters, hopeless for 100 million: optimizers run on whole arrays at once.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "The rules are element-wise, so they translate directly to array expressions. Adam's bias correction divides by `1 - b**t`, which corrects the zero-initialised averages in the first steps.",
    steps: ["Momentum: `v_new = beta * v + g`; `w_new = w - lr * v_new`.", "Adam: update `m` and `s` with exponential moving averages.", "Bias-correct both: `m / (1 - b1**t)`, `s / (1 - b2**t)`.", "Step: `w - lr * m_hat / (sqrt(s_hat) + eps)`."],
    code: R`
import numpy as np

def sgd_momentum_step(w, g, v, lr, beta):
    v_new = beta * v + g
    return w - lr * v_new, v_new

def adam_step(w, g, m, s, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    m_new = b1 * m + (1 - b1) * g                     # average of gradients
    s_new = b2 * s + (1 - b2) * g ** 2                # average of squared gradients
    m_hat = m_new / (1 - b1 ** t)                     # bias correction
    s_hat = s_new / (1 - b2 ** t)
    return w - lr * m_hat / (np.sqrt(s_hat) + eps), m_new, s_new`,
    tc: "O(n)", tw: "A few vectorized operations.", sc: "O(n)", sw: "New state arrays.",
    viz: "dl1_optim",
  },
  run: R`
w = np.array([1.0, -2.0]); v = np.zeros(2)
for _ in range(3):
    w, v = sgd_momentum_step(w, 2 * w, v, lr=0.1, beta=0.9)      # gradient of w^2 is 2w
print(np.round(w, 4).tolist(), np.round(v, 4).tolist())

w = np.array([1.0, -2.0]); m = np.zeros(2); s = np.zeros(2)
for t in range(1, 4):
    w, m, s = adam_step(w, 2 * w, m, s, t, lr=0.1)
print(np.round(w, 4).tolist())

w1, _, _ = adam_step(np.zeros(2), np.array([0.001, 1000.0]), np.zeros(2), np.zeros(2), 1, lr=0.1)
print(np.round(w1, 4).tolist())
print(sgd_momentum_step(np.array([1.0]), np.array([2.0]), np.array([0.0]), 0.1, 0.9)[0].tolist())`,
  out: R`
[0.062, -0.124] [3.98, -7.96]
[0.7016, -1.7006]
[-0.1, -0.1]
[0.8]`,
  dryOn: "momentum, w = 1, g = 2, v = 0, lr = 0.1, β = 0.9",
  dry: { h: ["step", "v", "w"], r: [["start", "0", "1"], ["v = 0.9·0 + 2", "2", "1 − 0.1·2 = 0.8"]] },
  expl: [
    ["beta * v + g", "Keep 90% of the old velocity and add the new gradient: consistent gradients accumulate."],
    ["b1 * m + (1 - b1) * g", "Exponential moving average of the gradient (the 'momentum' part)."],
    ["b2 * s + (1 - b2) * g ** 2", "Moving average of the squared gradient (the 'RMSProp' part): how big gradients usually are."],
    ["m / (1 - b1 ** t)", "m and s start at zero, so early on they are biased small; this rescales them."],
    ["m_hat / (np.sqrt(s_hat) + eps)", "Divide by the typical gradient size, so every parameter moves at about the same pace."],
  ],
  take: "**Momentum averages gradients; Adam additionally divides by the running size of the gradient** (and bias-corrects the early steps). On step 1 every parameter moves by about lr.",
  mist: ["Starting `t` at 0: `1 - b1**0 = 0` divides by zero. The step counter is 1-based.", "Forgetting the bias correction: the first steps are far too small.", "Mutating `w` in place and also returning it, so the caller's old copy is silently changed."],
},
{
  id: "batchnorm-forward", t: "Batch Normalization Forward Pass", d: "Medium", p: "Normalization",
  q: "Implement the **training-mode forward pass of batch normalization** for a 2-D input.\n\n`batchnorm_forward(x, gamma, beta, run_mean, run_var, momentum=0.1, eps=1e-5)` takes a batch `x` of shape (n, d), scale `gamma` (d,), shift `beta` (d,), and the running statistics `run_mean`, `run_var` (d,). It returns `(out, new_run_mean, new_run_var)`:\n\n1. per-feature batch mean and **biased** variance (divide by n)\n2. `x_hat = (x - mean) / sqrt(var + eps)`, `out = gamma * x_hat + beta`\n3. running update: `new = (1 - momentum) * old + momentum * batch_stat`, where the running variance uses the **unbiased** batch variance (`var * n / (n - 1)`), as PyTorch does",
  ex: [["x = [[1, 2], [3, 6], [5, 10]], gamma = [1, 1], beta = [0, 0]", "out ≈ [[-1.2247, -1.2247], [0, 0], [1.2247, 1.2247]]", "Both features have mean [3, 6]; after normalising each column to mean 0, std 1 they become identical."],
    ["gamma = [2, 2], beta = [1, 1]", "out = 2·x_hat + 1", "gamma and beta rescale and shift after normalising."]],
  c: ["`n >= 2`", "Normalise with the biased variance; update the running variance with the unbiased one", "Do not modify the inputs"],
  hint: "`x.mean(axis=0)` and `x.var(axis=0)` (NumPy's `var` is biased by default) give per-feature statistics for a (n, d) batch. Remember the `n/(n-1)` factor only in the running variance.",
  brute: {
    name: "Plain Python loops",
    idea: "Loop over the features, compute that column's mean and variance with explicit sums, then normalise the column.",
    code: R`
import math
import numpy as np

def batchnorm_forward(x, gamma, beta, run_mean, run_var, momentum=0.1, eps=1e-5):
    n, d = x.shape
    out = np.zeros((n, d))
    new_mean, new_var = np.zeros(d), np.zeros(d)
    for j in range(d):
        col = [x[i, j] for i in range(n)]
        mean = sum(col) / n
        var = sum((v - mean) ** 2 for v in col) / n
        for i in range(n):
            out[i, j] = gamma[j] * (col[i] - mean) / math.sqrt(var + eps) + beta[j]
        new_mean[j] = (1 - momentum) * run_mean[j] + momentum * mean
        new_var[j] = (1 - momentum) * run_var[j] + momentum * var * n / (n - 1)
    return out, new_mean, new_var`,
    tc: "O(n·d)", tw: "Python loops over every element.", sc: "O(n·d)", sw: "Output.",
    bad: "Correct, but recomputes everything with Python loops. Reductions along an axis are one NumPy call.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Reduce over the batch axis (`axis=0`): one call gives all d means and variances. Broadcasting then normalises every column at once.",
    steps: ["`mean = x.mean(axis=0)`, `var = x.var(axis=0)` (biased).", "`x_hat = (x - mean) / sqrt(var + eps)`.", "`out = gamma * x_hat + beta`.", "Update the running statistics with momentum; use `var * n/(n-1)` for the variance."],
    code: R`
import numpy as np

def batchnorm_forward(x, gamma, beta, run_mean, run_var, momentum=0.1, eps=1e-5):
    n = x.shape[0]
    mean = x.mean(axis=0)                              # per-feature, over the batch
    var = x.var(axis=0)                                # biased (divide by n)
    x_hat = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    new_mean = (1 - momentum) * run_mean + momentum * mean
    new_var = (1 - momentum) * run_var + momentum * var * n / (n - 1)
    return out, new_mean, new_var`,
    tc: "O(n·d)", tw: "A few vectorized passes.", sc: "O(n·d)", sw: "Output.",
  },
  run: R`
x = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
out, rm, rv = batchnorm_forward(x, np.ones(2), np.zeros(2), np.zeros(2), np.ones(2))
print(np.round(out, 4).tolist())
print(np.round(rm, 4).tolist(), np.round(rv, 4).tolist())
out2, _, _ = batchnorm_forward(x, np.array([2.0, 2.0]), np.array([1.0, 1.0]), np.zeros(2), np.ones(2))
print(np.round(out2, 4).tolist())
rng = np.random.default_rng(0)
big = rng.normal(loc=[5.0, -3.0], scale=[10.0, 0.2], size=(1000, 2))
o, _, _ = batchnorm_forward(big, np.ones(2), np.zeros(2), np.zeros(2), np.ones(2))
print(np.allclose(o.mean(axis=0), 0), np.round(o.std(axis=0), 3).tolist())`,
  out: R`
[[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]]
[0.3, 0.6] [1.3, 2.5]
[[-1.4495, -1.4495], [1.0, 1.0], [3.4495, 3.4495]]
True [1.0, 1.0]`,
  dryOn: "x = [[1, 2], [3, 6], [5, 10]]",
  dry: { h: ["feature", "mean", "biased var", "x_hat (3 rows)"], r: [["0", "3", "(4+0+4)/3 = 2.667", "[-1.2247, 0, 1.2247]"], ["1", "6", "(16+0+16)/3 = 10.667", "[-1.2247, 0, 1.2247]"]] },
  expl: [
    ["x.mean(axis=0)", "Average down the batch axis: one mean per feature."],
    ["x.var(axis=0)", "Biased variance (÷ n): what the layer normalises with."],
    ["(x - mean) / np.sqrt(var + eps)", "Broadcast: every column gets mean 0 and std 1. `eps` avoids dividing by zero."],
    ["gamma * x_hat + beta", "Learned scale and shift: the network can undo the normalisation if that is best."],
    ["var * n / (n - 1)", "The running variance is an unbiased estimate (this is the convention PyTorch uses)."],
  ],
  take: "**BatchNorm = standardise each feature over the batch, then apply a learned scale and shift;** keep running averages of the statistics for inference time.",
  mist: ["Normalising over the wrong axis (`axis=1` standardises each example, which is LayerNorm).", "Forgetting `eps`: a constant feature gives 0/0 = NaN.", "Using batch statistics at inference instead of the running averages (and forgetting to call `model.eval()` in PyTorch)."],
},
]);
