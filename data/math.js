/* Math for ML: linear algebra, calculus, probability, information theory, optimization
   Outputs written as "?" are filled in by running the code: node tools/fill.js data/math.js */
MATH.push(
{
  id: "why-math", group: "Start Here", title: "Why Math for ML (and How Much You Need)",
  sub: "Machine learning is a small set of maths ideas used again and again. This page teaches exactly those, with pictures and a few lines of NumPy each.",
  tags: ["roadmap"],
  blocks: [
    ["p", "You do not need a maths degree to use machine learning, but you do need to be comfortable with a handful of ideas. The good news: the same few ideas appear everywhere. Once you see them here with small numbers, the later pages (gradient descent, neural networks, attention, loss functions) stop feeling like magic and start feeling like bookkeeping."],
    ["h", "Five pieces, five jobs"],
    ["table", ["Piece of maths", "The job it does in ML", "Where you will meet it"], [
      ["**Linear algebra** (vectors, matrices)", "Holds the data and the model: one example is a vector, a dataset is a matrix, a layer is a matrix multiply.", "[Linear regression](ml.html#linear-regression), [MLP forward pass](dl.html#mlp), [attention](transformers.html#scaled-dot-product)"],
      ["**Calculus** (derivatives, chain rule)", "Tells you how to change the model's numbers to make the error smaller.", "[Gradient descent](ml.html#gradient-descent), [backprop](dl.html#backprop)"],
      ["**Probability**", "Describes uncertainty: noisy data, predictions as probabilities, what a model 'believes'.", "[Naive Bayes](ml.html#naive-bayes), [logistic regression](ml.html#logistic-regression), [sampling](llm.html#sampling)"],
      ["**Information theory**", "Gives the standard loss for classification and language models.", "[Cross-entropy loss](dl.html#dl-losses), [pretraining](llm.html#pretraining)"],
      ["**Optimization**", "Explains when training works, where it can get stuck and why.", "[Optimizers](dl.html#optimizers), [vanishing gradients](dl.html#vanishing-gradients)"],
    ]],
    ["h", "How much is enough?"],
    ["p", "For this site you need three skills, not a hundred theorems:"],
    ["list", [
      "**Read the notation.** Know what $\\sum$, $x^\\top$, $\\hat y$, $\\partial L/\\partial w$ and $\\mathbb{E}[X]$ mean, and what shape each thing has.",
      "**Do the small examples by hand.** A 2x2 matrix, a 3-number vector, a coin flip. If you can do those, bigger ones are the same thing repeated.",
      "**Check yourself in NumPy.** Every idea below has a few lines you can run. If the code and your hand calculation agree, you understood it.",
    ]],
    ["p", "Here is a taste of how the pieces fit together. A tiny model predicts a price from two features. Linear algebra computes the prediction (a dot product), calculus gives the direction to improve, and the loss is a number we want to shrink:"],
    ["code", R`
import numpy as np

x = np.array([2.0, 1.0])        # one example: 2 features (say size and rooms)
w = np.array([0.5, -1.0])       # the model's weights
y = 3.0                         # the true answer

y_hat = w @ x                   # linear algebra: a dot product -> the prediction
loss = (y_hat - y) ** 2         # how wrong we are (one number)
grad = 2 * (y_hat - y) * x      # calculus: dloss/dw (chain rule)
w_new = w - 0.1 * grad          # optimization: one gradient-descent step
print("prediction:", y_hat, " loss:", loss)
print("gradient:", grad)
print("new loss:", round(float((w_new @ x - y) ** 2), 3))`, "prediction: 0.0  loss: 9.0\ngradient: [-12.  -6.]\nnew loss: 0.0"],
    ["p", "The loss dropped from 9.0 to (nearly) 0 in a single step. That was a lucky learning rate on one example; real training takes many small steps over many examples. Every line above is explained on this page."],
    ["h", "Suggested order"],
    ["steps", [
      "**Linear algebra** (6 sections): vectors → dot product → matrices → inverse → eigen → SVD.",
      "**Calculus** (4 sections): derivative → gradient → chain rule → Jacobians and shapes.",
      "**Probability** (5 sections): basics → Bayes → expectation and variance → distributions → maximum likelihood.",
      "**Information theory and optimization**: entropy and cross-entropy, then convexity and saddle points.",
    ]],
    ["tip", "If you are short on time, the highest-value sections are [Dot Product](math.html#dot-product), [Matrices](math.html#matrices), [Gradients](math.html#gradients), [The Chain Rule](math.html#chain-rule), [Bayes](math.html#bayes) and [Entropy](math.html#entropy). If NumPy itself is new, skim [Python for ML](pyml.html#numpy-arrays) first."],
  ],
},
{
  id: "vectors", group: "Linear Algebra", title: "Vectors",
  sub: "An ordered list of numbers that you can picture as an arrow: the basic unit of data in machine learning.",
  tags: ["linear algebra"],
  blocks: [
    ["p", "A **vector** is just a list of numbers in a fixed order, like $(3, 1)$ or $(0.2, -1.5, 4)$. The numbers are its **components** and the count of them is its **dimension**. That is the entire definition; the rest is how you think about it."],
    ["h", "Three ways to see the same thing"],
    ["table", ["View", "Example", "Useful when"], [
      ["A list of numbers", "A house: (120 m², 3 rooms, 2 baths) = (120, 3, 2)", "Describing one data example: its **features**"],
      ["A point in space", "(3, 1) is 3 steps right and 1 step up from the origin", "Seeing clusters and distances between examples"],
      ["An arrow from the origin", "The arrow from (0, 0) to (3, 1), with a direction and a length", "Adding vectors and thinking about angles"],
    ]],
    ["p", "In ML nearly everything is a vector: one example's features, a model's weights, a word's **embedding** (a list of, say, 768 numbers that captures its meaning) and the gradient that says how to improve the model."],
    ["h", "Adding and scaling"],
    ["p", "Two things you can do with vectors, both done **position by position**:"],
    ["list", [
      "**Add**: $(3,1)+(1,3)=(4,4)$. Geometrically, walk along the first arrow and then along the second.",
      "**Scale** by a number (a **scalar**): $2\\cdot(3,1)=(6,2)$. The arrow keeps its direction and gets twice as long; a negative scalar flips it.",
    ]],
    ["viz", "math_vec"],
    ["p", "Combining these gives the **linear combination**: scale some vectors and add them up, $w_1 v_1 + w_2 v_2 + \\dots$. A model's prediction is a linear combination of features, and 'the set of all points you can reach' is the idea of a **span**."],
    ["h", "In NumPy"],
    ["code", R`
import numpy as np

a = np.array([3.0, 1.0])
b = np.array([1.0, 3.0])

print("a + b      =", a + b)            # add position by position
print("2 * a      =", 2 * a)            # scale every component
print("a - b      =", a - b)            # the arrow from b to a
print("shape, dim =", a.shape, a.ndim)  # a vector is a 1-D array of length 2

# a linear combination: 0.5*a + 2*b
print("0.5a + 2b  =", 0.5 * a + 2 * b)

# the 'mean example' of a dataset is just an average of vectors
X = np.array([[120, 3, 2], [80, 2, 1], [100, 3, 1]], dtype=float)   # 3 houses, 3 features
print("average house:", X.mean(axis=0))`, "a + b      = [4. 4.]\n2 * a      = [6. 2.]\na - b      = [ 2. -2.]\nshape, dim = (2,) 1\n0.5a + 2b  = [3.5 6.5]\naverage house: [100.           2.66666667   1.33333333]"],
    ["note", "NumPy has no separate 'row vector' and 'column vector' for 1-D arrays: `a` above has shape `(2,)`. Maths texts write vectors as columns, so a vector with 3 entries is a 3x1 matrix. When we need the difference in code we reshape explicitly (see [Matrices](math.html#matrices))."],
    ["warn", "A Python list is not a vector: `[1, 2] + [3, 4]` glues the lists into `[1, 2, 3, 4]`. Convert to a NumPy array first and `+` adds position by position. See [vectorization](pyml.html#vectorization)."],
    ["p", "Next we need a way to say how *similar* two vectors are. That is the [dot product](math.html#dot-product)."],
  ],
},
{
  id: "dot-product", group: "Linear Algebra", title: "Dot Product, Norms & Cosine Similarity",
  sub: "Multiply matching entries and add: one number that says how much two vectors agree. It powers neurons, similarity search and attention.",
  tags: ["linear algebra", "similarity"],
  blocks: [
    ["p", "The **dot product** of two vectors multiplies them position by position and adds up the results:"],
    ["math", R`a\cdot b = a_1 b_1 + a_2 b_2 + \dots + a_n b_n = \sum_{i=1}^{n} a_i b_i`],
    ["p", "A tiny example: $(1,2,3)\\cdot(4,5,6)=1\\cdot4+2\\cdot5+3\\cdot6=4+10+18=32$. One number comes out, not a vector."],
    ["h", "What the number means"],
    ["p", "The dot product has a second, geometric face that explains why it is everywhere:"],
    ["math", R`a\cdot b = \lVert a\rVert \, \lVert b\rVert \cos\theta`, "θ is the angle between the two arrows"],
    ["list", [
      "Pointing the **same way** (θ near 0°): cos θ ≈ 1, big positive number.",
      "**Perpendicular** (θ = 90°): cos θ = 0, the dot product is exactly 0. We call such vectors **orthogonal**: they share nothing.",
      "Pointing **opposite ways** (θ near 180°): cos θ ≈ -1, big negative number.",
    ]],
    ["p", "Another way to see it: $a\\cdot b/\\lVert a\\rVert$ is the length of the **shadow** (projection) of $b$ on the line of $a$. Play with the sliders and watch the number and the shadow."],
    ["viz", "math_dot"],
    ["h", "Length: the norm"],
    ["p", "The **norm** $\\lVert a\\rVert$ is the length of the arrow. The usual one is the L2 (Euclidean) norm, $\\sqrt{a\\cdot a}=\\sqrt{a_1^2+a_2^2+\\dots}$. For $(3,4)$ it is $\\sqrt{9+16}=5$. Dividing a vector by its norm gives a **unit vector** of length 1 that only keeps the direction."],
    ["table", ["Norm", "Formula", "Where ML uses it"], [
      ["L2", "$\\sqrt{\\sum a_i^2}$", "Distances, [ridge regularization](ml.html#regularization), weight decay"],
      ["L1", "$\\sum |a_i|$", "[Lasso](ml.html#regularization) (pushes weights to exactly 0), MAE"],
      ["L∞ (max)", "$\\max_i |a_i|$", "Gradient clipping, adversarial robustness"],
    ]],
    ["h", "Cosine similarity"],
    ["p", "Raw dot products grow with vector length: a long document vector scores high against everything. To compare only the **direction**, divide both lengths out:"],
    ["math", R`\cos(a,b) = \frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert} \in [-1, 1]`],
    ["code", R`
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print("dot (loop)   :", sum(x * y for x, y in zip(a, b)))
print("dot (a @ b)  :", a @ b)
print("norm of a    :", np.linalg.norm(a).round(3), " (sqrt 14 =", round(14 ** 0.5, 3), ")")
print("L1, Linf     :", np.linalg.norm(a, 1), np.linalg.norm(a, np.inf))

def cosine(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

print("cos(a, b)    :", round(cosine(a, b), 4))
print("cos(a, 10a)  :", round(cosine(a, 10 * a), 4), " <- scaling does not change direction")
print("cos(a, -a)   :", round(cosine(a, -a), 4))
print("perpendicular:", round(cosine(np.array([1.0, 0.0]), np.array([0.0, 5.0])), 4))`, "dot (loop)   : 32.0\ndot (a @ b)  : 32.0\nnorm of a    : 3.742  (sqrt 14 = 3.742 )\nL1, Linf     : 6.0 3.0\ncos(a, b)    : 0.9746\ncos(a, 10a)  : 1.0  <- scaling does not change direction\ncos(a, -a)   : -1.0\nperpendicular: 0.0"],
    ["h", "Where ML uses it"],
    ["list", [
      "**A neuron / linear model**: the prediction is $w\\cdot x+b$. A large positive dot product means the input 'matches' the pattern stored in $w$.",
      "**Similarity search and RAG**: embeddings of texts are compared with cosine similarity. Practice it in [Cosine Similarity Search](practice.html#p/cosine-similarity).",
      "**Attention**: every query is dotted with every key to get match scores, see [scaled dot-product attention](transformers.html#scaled-dot-product).",
    ]],
    ["h", "Why attention divides by the square root of d"],
    ["p", "If the entries of two vectors are random with mean 0 and variance 1, their dot product is a sum of $d$ random terms, so its **spread grows like $\\sqrt{d}$**. Big scores make [softmax](dl.html#activations) saturate, so attention divides by $\\sqrt{d}$ to keep them in a sane range. You can see it:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
for d in [4, 64, 1024]:
    q = rng.standard_normal((20000, d))
    k = rng.standard_normal((20000, d))
    dots = (q * k).sum(axis=1)                  # 20000 dot products of length-d vectors
    print(f"d={d:5d}  std of q.k = {dots.std():6.2f}   after / sqrt(d): {(dots / np.sqrt(d)).std():.2f}")`, "d=    4  std of q.k =   2.00   after / sqrt(d): 1.00\nd=   64  std of q.k =   8.03   after / sqrt(d): 1.00\nd= 1024  std of q.k =  31.99   after / sqrt(d): 1.00"],
    ["tip", "Cosine similarity of two vectors that were already scaled to length 1 is just their dot product. That is why many vector databases normalise embeddings once and then use a fast plain dot product."],
    ["warn", "Cosine similarity ignores length on purpose. If length carries meaning (for example, how confident a signal is), it will be thrown away. Also, a zero vector has no direction: the formula divides by zero."],
  ],
},
{
  id: "matrices", group: "Linear Algebra", title: "Matrices & Matrix Multiplication",
  sub: "A grid of numbers that is both a table of data and a machine that turns vectors into other vectors.",
  tags: ["linear algebra"],
  blocks: [
    ["p", "A **matrix** is a rectangular grid of numbers. A matrix with $m$ rows and $n$ columns has **shape** $m\\times n$. It plays two roles in ML, and you should keep both in mind:"],
    ["list", [
      "**A table of data.** A dataset $X$ of 1000 houses with 3 features is a $1000\\times3$ matrix: one row per example, one column per feature.",
      "**A machine on vectors.** Multiplying a matrix by a vector produces a new vector. A neural network layer is exactly this.",
    ]],
    ["h", "Matrix times vector"],
    ["p", "Each output entry is the dot product of one **row** of the matrix with the vector:"],
    ["math", R`\begin{pmatrix}1&2\\3&4\end{pmatrix}\begin{pmatrix}5\\6\end{pmatrix}=\begin{pmatrix}1\cdot5+2\cdot6\\3\cdot5+4\cdot6\end{pmatrix}=\begin{pmatrix}17\\39\end{pmatrix}`],
    ["p", "There is a second, geometric reading that is worth far more. Look at the **columns**: $A v = v_1\\cdot(\\text{column 1}) + v_2\\cdot(\\text{column 2})$. The matrix says where the basis arrows $\\hat\\imath=(1,0)$ and $\\hat\\jmath=(0,1)$ land (the two columns), and every other point follows. The result is a **linear transformation** of the whole plane: rotate, stretch, shear, flip or squash. Change the four numbers and watch:"],
    ["viz", "math_matrix"],
    ["h", "Matrix times matrix"],
    ["p", "To multiply $A$ ($m\\times n$) by $B$ ($n\\times p$), the inner sizes must match, and the result is $m\\times p$. Entry $(i,j)$ of the result is row $i$ of $A$ dotted with column $j$ of $B$:"],
    ["math", R`(AB)_{ij} = \sum_{k} A_{ik} B_{kj}`, "shape: (m×n)(n×p) = (m×p)"],
    ["math", R`\begin{pmatrix}1&2\\3&4\end{pmatrix}\begin{pmatrix}0&1\\1&0\end{pmatrix}=\begin{pmatrix}2&1\\4&3\end{pmatrix}`, "multiplying by this matrix swaps the columns"],
    ["p", "Doing $A$ then $B$ means applying one transformation after another, so matrix multiplication is **composition**. Order matters: $AB\\neq BA$ in general (put on your socks, then your shoes; not the other way round)."],
    ["code", R`
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])
v = np.array([5, 6])

print("A @ v =", A @ v)
print("A @ B =\n", A @ B)
print("B @ A =\n", B @ A, " <- different: order matters")

# from scratch: three loops
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

print("loops match @:", np.allclose(matmul(A, B), A @ B))`, "A @ v = [17 39]\nA @ B =\n [[2 1]\n [4 3]]\nB @ A =\n [[3 4]\n [1 2]]  <- different: order matters\nloops match @: True"],
    ["h", "Where ML uses it"],
    ["p", "A whole dataset goes through a linear model or a layer in one multiply: with $X$ of shape (n_examples, n_features) and weights $W$ of shape (n_features, n_outputs), $XW$ gives all predictions at once. This is why GPUs matter for deep learning: they are very good at matrix multiplies."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 3))        # 5 examples, 3 features
W = rng.normal(size=(3, 2))        # a layer that maps 3 features -> 2 outputs
b = np.array([0.1, -0.1])          # one bias per output

out = X @ W + b                    # shapes: (5,3)(3,2) -> (5,2), then b is broadcast
print("X:", X.shape, " W:", W.shape, " out:", out.shape)
print(out.round(2))`, "X: (5, 3)  W: (3, 2)  out: (5, 2)\n[[ 0.72 -0.31]\n [ 0.57 -0.42]\n [-1.89 -0.33]\n [ 1.27  0.33]\n [ 0.57  1.24]]"],
    ["table", ["Operation", "NumPy", "Shape rule"], [
      ["Dot product", "`a @ b`", "(n,) and (n,) → a scalar"],
      ["Matrix · vector", "`A @ v`", "(m,n) and (n,) → (m,)"],
      ["Matrix · matrix", "`A @ B`", "(m,n) and (n,p) → (m,p)"],
      ["Element-wise product", "`A * B`", "same shape (or broadcastable); **not** matrix multiplication"],
    ]],
    ["danger", "`A * B` multiplies element by element, `A @ B` is the matrix product. Mixing them up is the most common NumPy linear-algebra bug. When a shape error appears, write the shapes down: (m,n)(n,p) must match in the middle."],
    ["tip", "Practice: [Matrix Multiply: Loops vs @](practice.html#p/matmul-loops-vs-vectorized). More on the NumPy side in [Matrix Operations & @](pyml.html#numpy-linalg) and [Broadcasting](pyml.html#broadcasting)."],
  ],
},
{
  id: "matrix-ops", group: "Linear Algebra", title: "Transpose, Inverse & Identity",
  sub: "Flip a matrix, undo a matrix, and the 'do nothing' matrix: the toolkit behind solving equations and the normal equation.",
  tags: ["linear algebra"],
  blocks: [
    ["h", "Transpose"],
    ["p", "The **transpose** $A^\\top$ flips a matrix over its diagonal: rows become columns. A $2\\times3$ matrix becomes $3\\times2$."],
    ["math", R`\begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}^{\top}=\begin{pmatrix}1&4\\2&5\\3&6\end{pmatrix}`],
    ["p", "Two rules you will use constantly: $(A^\\top)^\\top=A$ and $(AB)^\\top=B^\\top A^\\top$ (the order reverses). The dot product itself is a transpose in disguise: $a\\cdot b=a^\\top b$. A matrix with $A^\\top=A$ is **symmetric**; covariance matrices and $X^\\top X$ are always symmetric."],
    ["h", "Identity"],
    ["p", "The **identity matrix** $I$ has ones on the diagonal and zeros elsewhere. It does nothing: $IA=AI=A$ and $Iv=v$. It is the matrix version of the number 1."],
    ["h", "Inverse"],
    ["p", "The **inverse** $A^{-1}$ is the matrix that undoes $A$: $A^{-1}A=AA^{-1}=I$. If $A$ turns the plane one way, $A^{-1}$ turns it back. For a $2\\times2$ matrix there is a formula:"],
    ["math", R`\begin{pmatrix}a&b\\c&d\end{pmatrix}^{-1}=\frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}`, "ad − bc is the determinant"],
    ["p", "The **determinant** $\\det A=ad-bc$ is the factor by which $A$ scales area (you saw it in the matrix picture above). If it is **zero**, $A$ squashes the plane flat onto a line: information is lost, so nothing can undo it and **there is no inverse**. Such a matrix is called **singular**."],
    ["p", "Why care? To solve $Ax=b$ for $x$, in theory you compute $x=A^{-1}b$. Linear regression's [normal equation](ml.html#linear-regression) is exactly this: $w=(X^\\top X)^{-1}X^\\top y$."],
    ["code", R`
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 3.0]])
b = np.array([5.0, 10.0])

print("det(A)      =", round(np.linalg.det(A), 3), "  (2*3 - 1*1 = 5)")
A_inv = np.linalg.inv(A)
print("A_inv =\n", A_inv.round(3))
print("A_inv @ A =\n", (A_inv @ A).round(3) + 0.0)          # the identity

x1 = A_inv @ b                  # works, but see the warning below
x2 = np.linalg.solve(A, b)      # preferred: solves A x = b directly
print("x via inv   :", x1.round(3))
print("x via solve :", x2.round(3), "  check A@x =", A @ x2)

S = np.array([[1.0, 2.0], [2.0, 4.0]])               # second row = 2 * first row
print("det(S)      =", np.linalg.det(S), "  -> singular, no inverse")
try:
    np.linalg.inv(S)
except np.linalg.LinAlgError as e:
    print("inv(S) fails:", e)`, "det(A)      = 5.0   (2*3 - 1*1 = 5)\nA_inv =\n [[ 0.6 -0.2]\n [-0.2  0.4]]\nA_inv @ A =\n [[1. 0.]\n [0. 1.]]\nx via inv   : [1. 3.]\nx via solve : [1. 3.]   check A@x = [ 5. 10.]\ndet(S)      = 0.0   -> singular, no inverse\ninv(S) fails: Singular matrix"],
    ["warn", "In real code almost never form the inverse. `np.linalg.solve(A, b)` is faster and more accurate than `inv(A) @ b`. For regression, `np.linalg.lstsq` or a library solver also copes with nearly singular matrices. Nearly-singular (not exactly zero determinant) is the practical problem: tiny rounding errors get hugely amplified."],
    ["h", "When the inverse does not exist: the pseudo-inverse"],
    ["p", "For non-square or singular matrices, the **pseudo-inverse** $A^{+}$ gives the best possible answer in the least-squares sense. It is computed from the [SVD](math.html#svd). `np.linalg.pinv` does it, and it is what `lstsq` is based on."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(6, 2))                # 6 equations, 2 unknowns: no exact solution
w_true = np.array([2.0, -1.0])
y = X @ w_true + rng.normal(0, 0.05, 6)    # a little noise

w_normal = np.linalg.inv(X.T @ X) @ X.T @ y     # normal equation
w_pinv = np.linalg.pinv(X) @ y                  # pseudo-inverse
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]  # the recommended way
print(w_normal.round(3), w_pinv.round(3), w_lstsq.round(3))
print("same answer:", np.allclose(w_normal, w_lstsq) and np.allclose(w_pinv, w_lstsq))`, "[ 2.    -1.003] [ 2.    -1.003] [ 2.    -1.003]\nsame answer: True"],
    ["table", ["Concept", "Meaning", "ML example"], [
      ["Transpose $A^\\top$", "Swap rows and columns", "$X^\\top X$ in the normal equation"],
      ["Identity $I$", "The 'do nothing' matrix", "Ridge adds $\\lambda I$: $(X^\\top X+\\lambda I)^{-1}$ is always invertible"],
      ["Inverse $A^{-1}$", "Undo $A$ (needs $\\det A\\neq0$)", "Solving linear systems"],
      ["Determinant", "Area scaling; 0 means singular", "Detecting redundant (collinear) features"],
    ]],
    ["tip", "A matrix has no inverse when its columns are redundant (one column is a combination of the others). In a dataset this means **multicollinear** features, e.g. a price in dollars and the same price in cents. Ridge regularization fixes it, see [Regularization](ml.html#regularization)."],
  ],
},
{
  id: "eigen", group: "Linear Algebra", title: "Eigenvalues & Eigenvectors",
  sub: "The special directions a matrix only stretches and never turns: they reveal what a matrix really does, and they power PCA.",
  tags: ["linear algebra", "PCA"],
  blocks: [
    ["p", "Most vectors get both rotated and stretched when you multiply them by a matrix. But some special directions are only stretched (or flipped), staying on the same line. Those directions are the **eigenvectors** and the stretch factors are the **eigenvalues**:"],
    ["math", R`A v = \lambda v`, "v: eigenvector (a direction) · λ: eigenvalue (how much it is stretched)"],
    ["p", "Go back to the matrix picture above ([Matrices](math.html#matrices)): the purple dashed lines are the eigen-directions. Put the test vector on one of them and the red output lines up with it, longer by exactly λ."],
    ["viz", "math_matrix"],
    ["h", "A worked example"],
    ["p", "Take $A=\\begin{pmatrix}2&1\\\\1&2\\end{pmatrix}$. Try $v=(1,1)$: $Av=(2+1,\\ 1+2)=(3,3)=3\\cdot(1,1)$. So $(1,1)$ is an eigenvector with $\\lambda=3$. Try $v=(1,-1)$: $Av=(2-1,\\ 1-2)=(1,-1)=1\\cdot(1,-1)$, so $\\lambda=1$. To find them in general, require $(A-\\lambda I)v=0$ to have a non-zero solution, i.e. $\\det(A-\\lambda I)=0$:"],
    ["math", R`\det\begin{pmatrix}2-\lambda&1\\1&2-\lambda\end{pmatrix}=(2-\lambda)^2-1=0\;\Rightarrow\;\lambda=3\text{ or }1`],
    ["code", R`
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 2.0]])
vals, vecs = np.linalg.eigh(A)          # eigh: for symmetric matrices (real, sorted ascending)
print("eigenvalues :", vals)
print("eigenvectors (columns):\n", vecs.round(3))

v = vecs[:, 1]                           # the one for the larger eigenvalue
print("A v     =", (A @ v).round(3))
print("lam * v =", (vals[1] * v).round(3))
print("trace = sum of eigenvalues:", np.trace(A), vals.sum())
print("det = product of eigenvalues:", round(np.linalg.det(A), 3), vals.prod())`, "eigenvalues : [1. 3.]\neigenvectors (columns):\n [[-0.707  0.707]\n [ 0.707  0.707]]\nA v     = [2.121 2.121]\nlam * v = [2.121 2.121]\ntrace = sum of eigenvalues: 4.0 4.0\ndet = product of eigenvalues: 3.0 3.0"],
    ["p", "Use `eigh` for symmetric matrices (covariance matrices, $X^\\top X$): it is faster and returns real values with **orthogonal** eigenvectors. Use `eig` for general square matrices, where eigenvalues can even be complex (a pure rotation has none that are real)."],
    ["h", "Why they matter: repeated application"],
    ["p", "If you apply $A$ over and over, the eigenvector with the **largest** $|\\lambda|$ wins: every starting vector gets pulled towards it. This is the **power iteration** idea, and it is how the top eigenvector is found for huge matrices (and how PageRank works)."],
    ["code", R`
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 2.0]])
v = np.array([1.0, 0.0])                  # start anywhere (not exactly on another eigenvector)
for i in range(8):
    v = A @ v
    v = v / np.linalg.norm(v)             # keep the length at 1
    lam = v @ A @ v                       # Rayleigh quotient: estimate of the eigenvalue
    print(i, v.round(4), round(float(lam), 4))`, "0 [0.8944 0.4472] 2.8\n1 [0.7809 0.6247] 2.9756\n2 [0.7328 0.6805] 2.9973\n3 [0.7158 0.6983] 2.9997\n4 [0.71   0.7042] 3.0\n5 [0.7081 0.7061] 3.0\n6 [0.7074 0.7068] 3.0\n7 [0.7072 0.707 ] 3.0"],
    ["h", "Where ML uses it: PCA"],
    ["p", "The covariance matrix of your data is symmetric. Its **eigenvectors** are the directions along which the data varies most, and the **eigenvalues** say how much variance each direction holds. [PCA](ml.html#pca) is exactly: centre the data, take the top eigenvectors of the covariance matrix, and project onto them."],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
Z = rng.normal(size=(300, 2)) * [3.0, 0.8]          # stretched along x
th = np.deg2rad(30)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
X = Z @ R.T                                          # rotate by 30 degrees

Xc = X - X.mean(axis=0)                              # centre the data
C = Xc.T @ Xc / (len(Xc) - 1)                        # covariance matrix (2x2, symmetric)
vals, vecs = np.linalg.eigh(C)
vals, vecs = vals[::-1], vecs[:, ::-1]               # largest first
print("variance along each principal axis:", vals.round(2))
print("first principal axis:", np.abs(vecs[:, 0]).round(3), "(sign is arbitrary; true direction [0.866, 0.5])")
print("explained variance ratio:", (vals / vals.sum()).round(3))

plt.figure(figsize=(5, 4))
plt.scatter(Xc[:, 0], Xc[:, 1], s=8, alpha=0.5)
for k in range(2):
    d = vecs[:, k] * 2 * np.sqrt(vals[k])
    plt.plot([0, d[0]], [0, d[1]], lw=3, label=f"axis {k+1}")
plt.axis("equal"); plt.legend(); plt.title("Eigenvectors of the covariance = principal axes")
plt.show()`, "variance along each principal axis: [9.08 0.63]\nfirst principal axis: [0.864 0.504] (sign is arbitrary; true direction [0.866, 0.5])\nexplained variance ratio: [0.936 0.064]"],
    ["tip", "Practice: [Power Iteration for the Top Eigenvector](practice.html#p/power-iteration). Next: [SVD](math.html#svd) extends this idea to any matrix, not just square symmetric ones."],
    ["note", "Eigenvalues also describe stability. If a recurrent network multiplies by the same matrix at every step, eigenvalues above 1 make signals explode and below 1 make them vanish. See [vanishing & exploding gradients](dl.html#vanishing-gradients)."],
  ],
},
{
  id: "svd", group: "Linear Algebra", title: "Singular Value Decomposition (SVD)",
  sub: "Any matrix is a rotation, a stretch along the axes, and another rotation. Keep only the big stretches and you get compression, PCA and LoRA.",
  tags: ["linear algebra", "PCA", "LoRA"],
  blocks: [
    ["p", "Eigenvectors only exist for square matrices, and even then they can be awkward. The **SVD** works for **every** matrix, of any shape, and says something just as clean:"],
    ["math", R`A = U\,\Sigma\,V^{\top}`, "U, V: rotations (orthogonal matrices) · Σ: diagonal stretch with the singular values σ₁ ≥ σ₂ ≥ … ≥ 0"],
    ["diagram", "math_svd", [], "The SVD of a matrix acting on the unit circle: Vᵀ turns it, Σ stretches it into an ellipse, U turns the ellipse. The red and green lines are the ellipse's axes."],
    ["p", "Read it right to left on a vector: first $V^\\top$ rotates it, then $\\Sigma$ stretches each axis by a singular value, then $U$ rotates again. The singular values $\\sigma_i$ say how important each direction is."],
    ["h", "A sum of simple layers"],
    ["p", "Multiplying it out turns the matrix into a **sum of rank-1 layers**, each an outer product of one column of $U$ and one of $V$, weighted by its singular value:"],
    ["math", R`A=\sigma_1 u_1 v_1^{\top}+\sigma_2 u_2 v_2^{\top}+\dots+\sigma_r u_r v_r^{\top}`],
    ["p", "The first layers carry most of the information; the later ones (small $\\sigma$) are fine detail or noise. Drop them and you get the best possible approximation of lower rank (the Eckart-Young theorem: no other rank-$k$ matrix is closer). Slide the rank to rebuild a little picture:"],
    ["viz", "math_svdrank"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
A = rng.normal(size=(6, 4))
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("shapes U, s, Vt:", U.shape, s.shape, Vt.shape)
print("singular values:", s.round(3))

print("exact rebuild:", np.allclose(U @ np.diag(s) @ Vt, A))

k = 2
Ak = (U[:, :k] * s[:k]) @ Vt[:k]                  # keep the top 2 layers
err = np.linalg.norm(A - Ak)                      # Frobenius norm of the error
print("rank-2 error     :", round(float(err), 4))
print("sqrt(dropped s^2):", round(float(np.sqrt((s[k:] ** 2).sum())), 4), " <- exactly equal")

# singular values squared are the eigenvalues of A^T A
print("eig(A^T A)  :", np.sort(np.linalg.eigvalsh(A.T @ A))[::-1].round(3))
print("sigma^2     :", (s ** 2).round(3))`, "shapes U, s, Vt: (6, 4) (4,) (4, 4)\nsingular values: [3.036 2.034 1.856 0.895]\nexact rebuild: True\nrank-2 error     : 2.061\nsqrt(dropped s^2): 2.061  <- exactly equal\neig(A^T A)  : [9.218 4.135 3.446 0.802]\nsigma^2     : [9.218 4.135 3.446 0.802]"],
    ["h", "Compressing a real picture"],
    ["p", "The 8x8 digit images that ship with scikit-learn are small matrices. How many layers do you need before the digit is readable?"],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

img = load_digits().images[3].astype(float)        # an 8x8 picture of a digit
U, s, Vt = np.linalg.svd(img, full_matrices=False)
energy = np.cumsum(s ** 2) / np.sum(s ** 2)
print("singular values:", s.round(1))
print("energy kept by top 1, 2, 3 layers:", energy[:3].round(3))

fig, ax = plt.subplots(1, 5, figsize=(8, 2))
for a, k in zip(ax, [1, 2, 3, 5, 8]):
    approx = (U[:, :k] * s[:k]) @ Vt[:k]
    a.imshow(approx, cmap="gray_r"); a.set_title(f"rank {k}"); a.axis("off")
plt.show()`, "singular values: [46.5 22.4 12.8  9.8  5.2  1.8  0.   0. ]\nenergy kept by top 1, 2, 3 layers: [0.732 0.902 0.957]"],
    ["h", "Where ML uses it"],
    ["table", ["Use", "How SVD helps"], [
      ["[PCA](ml.html#pca)", "SVD of the centred data matrix $X=U\\Sigma V^\\top$: the rows of $V^\\top$ are the principal axes and $\\sigma_i^2/(n-1)$ are the variances. This is how scikit-learn computes PCA (more stable than forming the covariance matrix)."],
      ["[LoRA](llm.html#lora)", "A fine-tuning update to a big weight matrix is approximated by a **low-rank** product $BA$. Instead of $d\\times d$ numbers you train $2\\,d\\,r$ with $r$ small, which works because useful updates have a few big singular values."],
      ["Recommenders / embeddings", "Factorising a user-item matrix gives a short vector per user and per item (latent factors)."],
      ["Pseudo-inverse and conditioning", "$A^{+}=V\\Sigma^{+}U^\\top$. The ratio $\\sigma_{\\max}/\\sigma_{\\min}$ (the condition number) tells you how numerically fragile a linear system is."],
    ]],
    ["note", "Eigen vs SVD in one line: eigenvectors answer 'which directions does this *square* matrix keep?'; singular vectors answer 'which input directions get stretched the most, and to which output directions?' for *any* matrix. For a symmetric matrix with non-negative eigenvalues they coincide."],
    ["tip", "Practice: [PCA via SVD](practice.html#p/pca-svd)."],
  ],
},
{
  id: "derivatives", group: "Calculus", title: "Derivatives",
  sub: "The slope of a curve at a point: how much the output changes when you nudge the input. The compass of every learning algorithm.",
  tags: ["calculus"],
  blocks: [
    ["p", "Training a model means changing its numbers to make the loss smaller. To do that we need to know: **if I nudge this number a tiny bit up, does the loss go up or down, and by how much?** That answer is the **derivative**."],
    ["h", "Slope at a point"],
    ["p", "For a straight line the slope is 'rise over run' and is the same everywhere. For a curve the slope changes from point to point. To measure it at one point, draw a line through that point and a second point a small distance $h$ away (a **secant** line), and shrink $h$ towards zero. The secant turns into the **tangent line** that just touches the curve, and its slope is the derivative:"],
    ["math", R`f'(x)=\frac{df}{dx}=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}`],
    ["p", "Move the point and shrink $h$ in this picture. Watch the amber secant rotate onto the red tangent:"],
    ["viz", "math_tangent"],
    ["list", [
      "**$f'(x)>0$**: the function is rising; to make $f$ smaller, move left.",
      "**$f'(x)<0$**: the function is falling; to make $f$ smaller, move right.",
      "**$f'(x)=0$**: a flat spot: a minimum, a maximum, or a saddle. Gradient descent ($x\\leftarrow x-\\eta f'(x)$) stops here. See [Gradient Descent](ml.html#gradient-descent).",
    ]],
    ["h", "Rules you need"],
    ["p", "You will almost never compute a limit by hand. A short table of rules is enough:"],
    ["table", ["Function", "Derivative", "Note"], [
      ["constant $c$", "$0$", "constants do not change"],
      ["$x^n$", "$n\\,x^{n-1}$", "$x^2\\to2x$, $\\ \\sqrt{x}\\to\\frac{1}{2\\sqrt x}$"],
      ["$e^x$", "$e^x$", "the one function equal to its own slope"],
      ["$\\ln x$", "$1/x$", "appears in every log-loss"],
      ["$\\sin x$", "$\\cos x$", ""],
      ["$f+g$", "$f'+g'$", "derivative of a sum is the sum of derivatives"],
      ["$c\\cdot f$", "$c\\cdot f'$", "constants come out"],
      ["$f\\cdot g$", "$f'g+fg'$", "product rule"],
      ["$f(g(x))$", "$f'(g(x))\\cdot g'(x)$", "the [chain rule](math.html#chain-rule)"],
      ["sigmoid $\\sigma(x)=\\frac{1}{1+e^{-x}}$", "$\\sigma(x)(1-\\sigma(x))$", "never above 0.25"],
      ["ReLU $\\max(0,x)$", "$1$ if $x>0$ else $0$", "see [activations](dl.html#activations)"],
    ]],
    ["h", "Checking a derivative numerically"],
    ["p", "A formula you derived can be checked by brute force: compute the slope with a small $h$. The **central difference** $\\frac{f(x+h)-f(x-h)}{2h}$ is far more accurate than the one-sided version. This **gradient check** is the standard way to catch bugs in hand-written backprop."],
    ["code", R`
import numpy as np

f = lambda x: x ** 3 - 2 * x            # f'(x) = 3x^2 - 2
fprime = lambda x: 3 * x ** 2 - 2
x = 1.5

for h in [1.0, 0.1, 0.01, 1e-4, 1e-8, 1e-13]:
    fwd = (f(x + h) - f(x)) / h                  # forward difference
    ctr = (f(x + h) - f(x - h)) / (2 * h)        # central difference
    print(f"h={h:<7} forward {fwd:10.6f}   central {ctr:10.6f}")
print("exact f'(1.5) =", fprime(x))

# sigmoid and its derivative sigma(1 - sigma)
sigma = lambda z: 1 / (1 + np.exp(-z))
z = np.array([-4.0, 0.0, 4.0])
print("sigmoid'(z) =", (sigma(z) * (1 - sigma(z))).round(4), " max possible 0.25")`, "h=1.0     forward  10.250000   central   5.750000\nh=0.1     forward   5.210000   central   4.760000\nh=0.01    forward   4.795100   central   4.750100\nh=0.0001  forward   4.750450   central   4.750000\nh=1e-08   forward   4.750000   central   4.750000\nh=1e-13   forward   4.747314   central   4.747314\nexact f'(1.5) = 4.75\nsigmoid'(z) = [0.0177 0.25   0.0177]  max possible 0.25"],
    ["note", "Smaller is not always better: with $h=10^{-13}$ the rounding error of floating-point subtraction starts to dominate. About $10^{-5}$ to $10^{-4}$ is the sweet spot for central differences in double precision."],
    ["h", "Derivative as a way to find the minimum"],
    ["p", "For $f(x)=(x-3)^2$ the derivative is $2(x-3)$, which is zero at $x=3$: the bottom of the bowl. When no formula solves $f'(x)=0$ we **walk downhill** instead:"],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: (x - 3) ** 2
df = lambda x: 2 * (x - 3)

x, lr = -2.0, 0.2
path = [x]
for _ in range(15):
    x = x - lr * df(x)             # step against the slope
    path.append(x)
print("x after 5 steps :", round(path[5], 4))
print("x after 15 steps:", round(path[15], 6), "(true minimum at 3)")

xs = np.linspace(-3, 6, 100)
plt.figure(figsize=(6, 3.2))
plt.plot(xs, f(xs), label="f(x) = (x-3)^2")
plt.plot(path, [f(p) for p in path], "o-", ms=4, label="gradient descent")
plt.xlabel("x"); plt.legend(); plt.title("Following the slope downhill")
plt.show()`, "x after 5 steps : 2.6112\nx after 15 steps: 2.997649 (true minimum at 3)"],
    ["tip", "A derivative also tells you the *sensitivity*: if $f'(x)=3$, nudging $x$ by 0.01 changes $f$ by about 0.03. Backpropagation is nothing but a very organised way of computing these sensitivities for every weight. Next: what if there are many inputs? See [Partial Derivatives & Gradients](math.html#gradients)."],
  ],
},
{
  id: "gradients", group: "Calculus", title: "Partial Derivatives & Gradients",
  sub: "With many inputs, take one slope per input and stack them into a vector that points straight uphill.",
  tags: ["calculus"],
  blocks: [
    ["p", "A real model has thousands to billions of parameters, so the loss is a function of many inputs, $L(w_1,w_2,\\dots,w_n)$. We ask the derivative question once per input."],
    ["h", "Partial derivatives"],
    ["p", "The **partial derivative** $\\partial L/\\partial w_i$ is the ordinary derivative with respect to $w_i$ while **every other input is frozen** like a constant. Example: $f(x,y)=x^2+3xy$."],
    ["list", [
      "$\\partial f/\\partial x$: treat $y$ as a number: $2x+3y$.",
      "$\\partial f/\\partial y$: treat $x$ as a number: $3x$.",
    ]],
    ["h", "The gradient"],
    ["p", "Stack all the partial derivatives into one vector. This is the **gradient**:"],
    ["math", R`\nabla f=\begin{pmatrix}\partial f/\partial x\\ \partial f/\partial y\end{pmatrix}=\begin{pmatrix}2x+3y\\3x\end{pmatrix}\quad\Rightarrow\quad \nabla f(1,2)=\begin{pmatrix}8\\3\end{pmatrix}`],
    ["p", "At the point $(1,2)$ the loss grows by about 8 per unit step in $x$ and 3 per unit step in $y$. The gradient is the direction of **steepest ascent**, and its length is how steep that is. Why is that the steepest direction? The rate of change of $f$ if you walk in a unit direction $u$ is the dot product $\\nabla f\\cdot u=\\lVert\\nabla f\\rVert\\cos\\theta$ (see [Dot Product](math.html#dot-product)); it is largest when $u$ points along $\\nabla f$ itself, and smallest when it points opposite. So to go **downhill fastest**, step along $-\\nabla f$:"],
    ["math", R`w\;\leftarrow\;w-\eta\,\nabla L(w)`, "gradient descent, now for any number of parameters"],
    ["viz", "math_gradpath"],
    ["p", "In the picture the rings are lines of equal loss ('contour lines', like a hiking map). The gradient is always **perpendicular to the ring** it sits on, and the whole field of arrows points to the bottom. Note the zig-zag: when the valley is narrow the gradient points mostly across it. That is why [momentum and Adam](dl.html#optimizers) help."],
    ["h", "A real gradient: linear regression"],
    ["p", "For predictions $\\hat y=Xw$ and loss $L=\\frac1n\\lVert Xw-y\\rVert^2$, the gradient (derived in [The Chain Rule](math.html#chain-rule) and [Jacobians](math.html#jacobians)) is $\\nabla_w L=\\frac2n X^\\top(Xw-y)$. Always check a gradient formula against the slow numerical one:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(50, 3))
y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 0.1, 50)

def loss(w):
    return np.mean((X @ w - y) ** 2)

def grad(w):                                   # analytic gradient
    return 2 / len(y) * X.T @ (X @ w - y)

def num_grad(f, w, h=1e-5):                    # numerical gradient: one central difference per parameter
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w); e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2 * h)
    return g

w = np.array([0.0, 0.0, 0.0])
print("analytic :", grad(w).round(4))
print("numerical:", num_grad(loss, w).round(4))
print("max difference:", np.abs(grad(w) - num_grad(loss, w)).max() < 1e-6)

for step in range(51):
    w = w - 0.1 * grad(w)                      # gradient descent
    if step % 25 == 0:
        print(f"step {step:2d}  loss {loss(w):.4f}  w {w.round(3)}")`, "analytic : [-5.0599  1.1258 -1.2954]\nnumerical: [-5.0599  1.1258 -1.2954]\nmax difference: True\nstep  0  loss 3.5099  w [ 0.506 -0.113  0.13 ]\nstep 25  loss 0.0092  w [ 2.014 -0.99   0.504]\nstep 50  loss 0.0087  w [ 2.018 -1.015  0.502]"],
    ["tip", "Practice: [Numerical Gradient Check](practice.html#p/numerical-gradient-check)."],
    ["warn", "The gradient only tells you the *local* slope. A step that is too large can overshoot and end up higher. This is why the learning rate $\\eta$ matters, and why the surface shape matters ([Convexity, Minima & Saddle Points](math.html#optimization))."],
  ],
},
{
  id: "chain-rule", group: "Calculus", title: "The Chain Rule",
  sub: "To differentiate a function built from other functions, multiply the slopes along the chain. This single rule is backpropagation.",
  tags: ["calculus", "backprop"],
  blocks: [
    ["p", "A neural network is a long chain: input → layer → layer → … → loss. To learn, we need $\\partial L/\\partial w$ for a weight deep inside. The **chain rule** says how a change travels along the chain."],
    ["h", "The idea"],
    ["p", "If $y$ depends on $u$, and $u$ depends on $x$, then nudging $x$ nudges $u$, which nudges $y$. The effects **multiply**:"],
    ["math", R`\frac{dy}{dx}=\frac{dy}{du}\cdot\frac{du}{dx}`],
    ["p", "Gears are a good picture: if gear 1 turns gear 2 at 3 turns per turn, and gear 2 turns gear 3 at 2 turns per turn, then gear 1 turns gear 3 at $3\\times2=6$. Worked example: $y=\\sin(x^2)$. Let $u=x^2$ so $y=\\sin u$. Then $dy/du=\\cos u$ and $du/dx=2x$, so $dy/dx=\\cos(x^2)\\cdot2x$."],
    ["h", "A computation graph"],
    ["p", "Take the smallest 'model' there is: a prediction $\\hat y=wx+b$ and a squared error $L=(\\hat y-y)^2$. Break it into small steps (a **computation graph**). Run it **forward** to get the values, then **backward**, multiplying local derivatives:"],
    ["diagram", "math_chain", [], "With x = 2, w = 3, b = 1, y = 5. Forward: wx = 6, ŷ = 7, error e = 2, L = 4. Backward: ∂L/∂e = 2e = 4; this passes unchanged through the + and the −; at the × it is multiplied by x (for w) or by w (for x)."],
    ["p", "So $\\partial L/\\partial w=\\underbrace{2e}_{4}\\cdot\\underbrace{1}_{\\partial e/\\partial\\hat y}\\cdot\\underbrace{x}_{\\partial\\hat y/\\partial w}=4\\cdot2=8$. Each node only needs its own **local derivative**; the chain rule glues them together. If one value feeds several places, the contributions from each path are **added**."],
    ["code", R`
import numpy as np

x, w, b, y = 2.0, 3.0, 1.0, 5.0

# forward pass
p = w * x           # 6
y_hat = p + b       # 7
e = y_hat - y       # 2
L = e ** 2          # 4

# backward pass (multiply local derivatives, from the loss backwards)
dL_de = 2 * e               # d(e^2)/de
dL_dyhat = dL_de * 1        # e = y_hat - y
dL_db = dL_dyhat * 1        # y_hat = p + b
dL_dp = dL_dyhat * 1
dL_dw = dL_dp * x           # p = w * x
dL_dx = dL_dp * w
print("L =", L, " dL/dw =", dL_dw, " dL/db =", dL_db, " dL/dx =", dL_dx)

# check with finite differences
def loss(w, b, x):
    return (w * x + b - y) ** 2
h = 1e-6
print("numerical dL/dw:", round((loss(w + h, b, x) - loss(w - h, b, x)) / (2 * h), 4))
print("numerical dL/dx:", round((loss(w, b, x + h) - loss(w, b, x - h)) / (2 * h), 4))`, "L = 4.0  dL/dw = 8.0  dL/db = 4.0  dL/dx = 12.0\nnumerical dL/dw: 8.0\nnumerical dL/dx: 12.0"],
    ["h", "Adding a nonlinearity: one sigmoid neuron"],
    ["p", "Now put a sigmoid in the chain: $z=wx+b$, $a=\\sigma(z)$, $L=(a-y)^2$. Three links, three factors: $\\dfrac{\\partial L}{\\partial w}=\\underbrace{2(a-y)}_{\\partial L/\\partial a}\\cdot\\underbrace{a(1-a)}_{\\partial a/\\partial z}\\cdot\\underbrace{x}_{\\partial z/\\partial w}$."],
    ["code", R`
import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))

x, y = 1.5, 1.0
w, b = 0.8, -0.5

z = w * x + b
a = sigmoid(z)
L = (a - y) ** 2

dL_da = 2 * (a - y)
da_dz = a * (1 - a)             # derivative of the sigmoid
dL_dw = dL_da * da_dz * x       # chain rule: multiply the three links
dL_db = dL_da * da_dz * 1

h = 1e-6
Lw = lambda w_: (sigmoid(w_ * x + b) - y) ** 2
print("a =", round(a, 4), " L =", round(L, 4))
print("dL/dw analytic :", round(dL_dw, 6))
print("dL/dw numerical:", round((Lw(w + h) - Lw(w - h)) / (2 * h), 6))`, "a = 0.6682  L = 0.1101\ndL/dw analytic : -0.220701\ndL/dw numerical: -0.220701"],
    ["h", "Why very deep networks are hard to train"],
    ["p", "Backprop through $n$ layers multiplies $n$ local factors. The sigmoid slope is at most 0.25, so a product of many of them shrinks towards zero (**vanishing gradients**); factors above 1 blow up instead (**exploding gradients**):"],
    ["code", R`
for n in [1, 5, 10, 20]:
    print(f"{n:2d} layers: all factors 0.25 -> {0.25 ** n:.2e}   all factors 1.5 -> {1.5 ** n:.1f}")`, " 1 layers: all factors 0.25 -> 2.50e-01   all factors 1.5 -> 1.5\n 5 layers: all factors 0.25 -> 9.77e-04   all factors 1.5 -> 7.6\n10 layers: all factors 0.25 -> 9.54e-07   all factors 1.5 -> 57.7\n20 layers: all factors 0.25 -> 9.09e-13   all factors 1.5 -> 3325.3"],
    ["tip", "That is the reason for ReLU, careful [initialization](dl.html#initialization), [normalization](dl.html#normalization) and residual connections. The full picture, with vectors and matrices instead of single numbers, is in [Backpropagation by Hand](dl.html#backprop) and [Build a Tiny Autograd Engine](dl.html#autograd)."],
    ["note", "PyTorch and TensorFlow do exactly this for you: they record the computation graph on the forward pass and apply the chain rule backwards when you call `loss.backward()`."],
  ],
},
{
  id: "jacobians", group: "Calculus", title: "Shapes, Jacobians & Matrix Calculus",
  sub: "The bookkeeping of derivatives with vectors and matrices: what shape each gradient has, and the few formulas that cover almost all of deep learning.",
  tags: ["calculus", "shapes"],
  blocks: [
    ["p", "The chain rule with single numbers is easy. In practice $x$, $y$ and the weights are vectors and matrices, and the main difficulty is **shapes**: which way do things multiply, and when do you transpose? A good rule keeps you safe: **the gradient of a scalar loss has exactly the same shape as the thing you differentiate with respect to.**"],
    ["table", ["Differentiate the loss w.r.t.", "Shape of that thing", "Shape of the gradient"], [
      ["a scalar $b$", "()", "()"],
      ["a vector $w$", "(n,)", "(n,)"],
      ["a matrix $W$", "(m, n)", "(m, n)"],
    ]],
    ["h", "The Jacobian"],
    ["p", "If a function maps $n$ inputs to $m$ outputs, $y=f(x)$, we need to know how every output reacts to every input. The grid of all those partial derivatives is the **Jacobian** $J$, with $m$ rows (outputs) and $n$ columns (inputs): $J_{ij}=\\partial y_i/\\partial x_j$."],
    ["diagram", "math_shapes", [], "The Jacobian has one row per output and one column per input. For a scalar function (m = 1) it is a single row: the gradient."],
    ["p", "Then the chain rule becomes a matrix multiplication of Jacobians: for $x\\to y\\to L$ with scalar $L$, $\\;\\nabla_x L=J^\\top\\nabla_y L$. Backpropagation is exactly this repeated: **multiply the incoming gradient by the transposed Jacobian of each layer**."],
    ["h", "The formulas worth remembering"],
    ["table", ["Forward", "Gradient w.r.t. the input", "Gradient w.r.t. the weights"], [
      ["$y=Wx$ (vector $x$)", "$\\nabla_x L=W^\\top\\,\\nabla_y L$", "$\\nabla_W L=\\nabla_y L\\;x^\\top$ (an outer product, shape of $W$)"],
      ["$Y=XW$ (batch $X$: n×d)", "$\\nabla_X L=\\nabla_Y L\\;W^\\top$", "$\\nabla_W L=X^\\top\\,\\nabla_Y L$"],
      ["$s=w\\cdot x$", "$\\nabla_x s=w$", "$\\nabla_w s=x$"],
      ["$L=\\lVert Xw-y\\rVert^2$", "", "$\\nabla_w L=2X^\\top(Xw-y)$"],
      ["softmax then cross-entropy", "", "$\\nabla_{\\text{logits}}L=p-y_{\\text{onehot}}$"],
    ]],
    ["p", "The last row is worth remembering: for a classifier trained with [cross-entropy](math.html#entropy), the gradient with respect to the raw scores (logits) is simply **predicted probabilities minus the one-hot truth**. It is clean, bounded and a big reason cross-entropy trains so well."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3))          # batch of 4 examples, 3 features
W = rng.normal(size=(3, 2))          # layer: 3 -> 2
t = rng.normal(size=(4, 2))          # targets

Y = X @ W                            # forward: (4,3)(3,2) -> (4,2)
L = np.sum((Y - t) ** 2)             # a scalar loss

dY = 2 * (Y - t)                     # dL/dY has the shape of Y: (4,2)
dW = X.T @ dY                        # (3,4)(4,2) -> (3,2): same shape as W
dX = dY @ W.T                        # (4,2)(2,3) -> (4,3): same shape as X
print("shapes: dY", dY.shape, " dW", dW.shape, " dX", dX.shape)

# check one entry of dW numerically
h = 1e-6
Wp = W.copy(); Wp[1, 0] += h
Wm = W.copy(); Wm[1, 0] -= h
num = (np.sum((X @ Wp - t) ** 2) - np.sum((X @ Wm - t) ** 2)) / (2 * h)
print("dW[1,0] analytic:", round(dW[1, 0], 5), " numerical:", round(num, 5))`, "shapes: dY (4, 2)  dW (3, 2)  dX (4, 3)\ndW[1,0] analytic: -9.61964  numerical: -9.61964"],
    ["h", "Softmax + cross-entropy: p − y"],
    ["code", R`
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

z = np.array([2.0, 1.0, -1.0])       # logits for 3 classes
y = np.array([0.0, 1.0, 0.0])        # the true class is class 1 (one-hot)

p = softmax(z)
L = -np.sum(y * np.log(p))
print("p        :", p.round(4), " loss:", round(L, 4))
print("gradient :", (p - y).round(4), "  <- predicted minus true")

h = 1e-6                              # numerical check, one logit at a time
num = []
for i in range(3):
    zp, zm = z.copy(), z.copy(); zp[i] += h; zm[i] -= h
    num.append((-np.log(softmax(zp))[1] + np.log(softmax(zm))[1]) / (2 * h))
print("numerical:", np.round(num, 4))`, "p        : [0.7054 0.2595 0.0351]  loss: 1.349\ngradient : [ 0.7054 -0.7405  0.0351]   <- predicted minus true\nnumerical: [ 0.7054 -0.7405  0.0351]"],
    ["h", "The Hessian"],
    ["p", "The second derivative of a many-input function is a matrix, the **Hessian** $H_{ij}=\\partial^2L/\\partial w_i\\partial w_j$. It describes the **curvature** of the loss surface. Its eigenvalues ([Eigenvalues](math.html#eigen)) say how sharply the surface curves along each direction, and their signs separate minima from saddle points ([Convexity & Saddle Points](math.html#optimization))."],
    ["tip", "When a shape error appears in backprop code, write the shapes of each tensor next to the line. If $dW$ must have the shape of $W$, only one ordering of $X^\\top$ and $dY$ makes the multiplication legal. In PyTorch you will never write these formulas (autograd does), but you will *read* shape errors every day."],
  ],
},
{
  id: "probability", group: "Probability", title: "Probability Basics & Conditional Probability",
  sub: "A number between 0 and 1 for how likely something is, and how that number changes once you know something else.",
  tags: ["probability"],
  blocks: [
    ["p", "Machine learning data is noisy and models are never certain, so probability is the language of ML. A classifier does not say 'spam'; it says 'spam with probability 0.93'."],
    ["h", "Outcomes, events, probability"],
    ["list", [
      "The **sample space** is the set of all possible outcomes (a die: 1 to 6).",
      "An **event** is a set of outcomes we care about ('the roll is even' = {2, 4, 6}).",
      "The **probability** $P(A)$ is a number from 0 (impossible) to 1 (certain). With equally likely outcomes it is (outcomes in A) / (all outcomes): $P(\\text{even})=3/6$.",
      "All outcomes together add up to 1, so $P(\\text{not }A)=1-P(A)$.",
    ]],
    ["p", "Two events can overlap: $P(A\\text{ or }B)=P(A)+P(B)-P(A\\text{ and }B)$ (subtract the overlap, or you count it twice)."],
    ["h", "Conditional probability"],
    ["p", "**Conditional probability** answers: given that $B$ has happened, how likely is $A$? Knowing $B$ shrinks the world to just the outcomes in $B$, and we ask what fraction of *that* is also in $A$:"],
    ["math", R`P(A\mid B)=\frac{P(A\text{ and }B)}{P(B)}`],
    ["diagram", "math_prob", [], "Everything is a region of area 1. Given B, we only look inside B's circle, and ask what share of it also lies in A."],
    ["p", "Rearranged, this is the **product rule**: $P(A\\text{ and }B)=P(A\\mid B)\\,P(B)$. If knowing $B$ tells you nothing about $A$, the events are **independent** and $P(A\\mid B)=P(A)$, so $P(A\\text{ and }B)=P(A)P(B)$. Two coin flips are independent; 'it rains' and 'I carry an umbrella' are not."],
    ["p", "The **law of total probability** splits an event over cases: $P(A)=P(A\\mid B)P(B)+P(A\\mid\\neg B)P(\\neg B)$. We will need it for [Bayes' theorem](math.html#bayes)."],
    ["h", "Two dice, by counting and by simulation"],
    ["p", "Roll two dice. What is the probability that the sum is 8, **given** that the first die is even? Enumerate all 36 outcomes and count; then check by simulation."],
    ["code", R`
import numpy as np
from itertools import product

outcomes = list(product(range(1, 7), repeat=2))          # all 36 (first, second) pairs
n = len(outcomes)

A = [o for o in outcomes if sum(o) == 8]                 # event A: sum is 8
B = [o for o in outcomes if o[0] % 2 == 0]               # event B: first die is even
AB = [o for o in A if o in B]

print("P(A)       =", len(A), "/", n, "=", round(len(A) / n, 4))
print("P(B)       =", len(B), "/", n, "=", round(len(B) / n, 4))
print("P(A and B) =", len(AB), "/", n, "=", round(len(AB) / n, 4))
print("P(A | B)   =", round((len(AB) / n) / (len(B) / n), 4), "  = 3/18")

rng = np.random.default_rng(0)
d = rng.integers(1, 7, size=(200000, 2))
first_even = d[:, 0] % 2 == 0
print("simulated P(A | B) =", round(float(((d.sum(axis=1) == 8) & first_even).sum() / first_even.sum()), 3))`, "P(A)       = 5 / 36 = 0.1389\nP(B)       = 18 / 36 = 0.5\nP(A and B) = 3 / 36 = 0.0833\nP(A | B)   = 0.1667   = 3/18\nsimulated P(A | B) = 0.166"],
    ["h", "Where ML uses it"],
    ["list", [
      "A classifier models $P(y\\mid x)$: the probability of each label **given** the features. See [logistic regression](ml.html#logistic-regression) and [softmax](dl.html#activations).",
      "A language model models $P(\\text{next token}\\mid\\text{previous tokens})$. The product rule lets it score a whole sentence as a product of such steps, see [pretraining](llm.html#pretraining).",
      "'Independent' is the assumption in [Naive Bayes](ml.html#naive-bayes) (features independent given the class) and the i.i.d. assumption about training data.",
    ]],
    ["warn", "$P(A\\mid B)$ and $P(B\\mid A)$ are different things. 'Most people with the flu have a fever' does not mean 'most people with a fever have the flu'. Getting from one to the other is the job of the next section."],
  ],
},
{
  id: "bayes", group: "Probability", title: "Bayes' Theorem",
  sub: "How to update a belief when you see evidence: and why a '99% accurate' test can still be wrong most of the time.",
  tags: ["probability"],
  blocks: [
    ["p", "Bayes' theorem flips a conditional probability. Often we know how likely the evidence is *if* the hypothesis is true, but we want the opposite: how likely the hypothesis is *given* the evidence."],
    ["h", "Derivation in two lines"],
    ["p", "The product rule works in both directions: $P(A\\text{ and }B)=P(A\\mid B)P(B)=P(B\\mid A)P(A)$. Divide by $P(B)$:"],
    ["math", R`P(H\mid E)=\frac{P(E\mid H)\;P(H)}{P(E)}`, "H: hypothesis · E: evidence"],
    ["table", ["Part", "Name", "Meaning"], [
      ["$P(H)$", "**prior**", "belief in H before seeing the evidence"],
      ["$P(E\\mid H)$", "**likelihood**", "how likely this evidence is if H is true"],
      ["$P(E)$", "evidence (normaliser)", "how likely the evidence is overall, $=\\sum_H P(E\\mid H)P(H)$"],
      ["$P(H\\mid E)$", "**posterior**", "updated belief after the evidence"],
    ]],
    ["h", "The famous surprise"],
    ["p", "A disease affects 1% of people. A test catches 99% of the sick and wrongly flags 1% of the healthy. You test positive. How worried should you be? Most people say '99%'. Count instead: out of 10,000 people, 100 are sick and 99 of them test positive; of the 9,900 healthy ones, 99 also test positive. So of the 198 positives only 99 are really sick: **50%**, not 99%."],
    ["viz", "math_bayes"],
    ["p", "The healthy group is so much larger than the sick group that even a small false-alarm rate produces as many false positives as there are true ones. This **base-rate effect** is why screening rare conditions is hard, why a spam filter must know how common spam is, and why a 'fraud detector with 99% accuracy' on rare fraud can be useless. Drag the prevalence up and down to see how much the prior matters."],
    ["code", R`
def posterior(prior, sensitivity, specificity):
    """P(sick | positive test) by Bayes' theorem."""
    p_pos_sick = sensitivity                 # P(+ | sick)
    p_pos_healthy = 1 - specificity          # P(+ | healthy) = false-alarm rate
    evidence = p_pos_sick * prior + p_pos_healthy * (1 - prior)      # P(+), total probability
    return p_pos_sick * prior / evidence

for prior in [0.001, 0.01, 0.1, 0.5]:
    print(f"prevalence {prior:6.1%} -> P(sick | +) = {posterior(prior, 0.99, 0.99):.1%}")

# updating twice: a second independent positive test, using the first posterior as the new prior
p1 = posterior(0.01, 0.99, 0.99)
p2 = posterior(p1, 0.99, 0.99)
print("after one positive :", round(p1, 3))
print("after two positives:", round(p2, 3))`, "prevalence   0.1% -> P(sick | +) = 9.0%\nprevalence   1.0% -> P(sick | +) = 50.0%\nprevalence  10.0% -> P(sick | +) = 91.7%\nprevalence  50.0% -> P(sick | +) = 99.0%\nafter one positive : 0.5\nafter two positives: 0.99"],
    ["p", "The last lines show Bayes' theorem as **learning**: yesterday's posterior is today's prior. Each new piece of evidence updates the belief."],
    ["h", "Where ML uses it"],
    ["list", [
      "**[Naive Bayes](ml.html#naive-bayes)** classifies by $P(\\text{class}\\mid\\text{words})\\propto P(\\text{words}\\mid\\text{class})P(\\text{class})$.",
      "**Priors in regularization.** An L2 penalty on weights is the same as a Gaussian prior belief that weights are small, see [maximum likelihood](math.html#mle) and [Regularization](ml.html#regularization).",
      "**Calibration and class imbalance.** The prior shifts when the class balance changes between training and deployment, see [Imbalanced Data](ml.html#imbalanced).",
      "**Bayesian updating** appears in recommender systems, A/B testing and in how people reason about evidence.",
    ]],
    ["tip", "Always ask 'what is the base rate?' before trusting a percentage. Natural frequencies ('out of 10,000 people...') are the easiest way to get Bayes' theorem right without symbols."],
  ],
},
{
  id: "expectation", group: "Probability", title: "Expectation & Variance",
  sub: "The long-run average of a random quantity, and how much it typically strays from that average.",
  tags: ["probability", "statistics"],
  blocks: [
    ["p", "A **random variable** is a number whose value depends on chance: the roll of a die, the height of a random person, the loss on a random training example. Two numbers summarise most of what we need to know about it: where it is centred and how spread out it is."],
    ["h", "Expectation (the mean)"],
    ["p", "The **expectation** $\\mathbb{E}[X]$ is the average of all possible values, each weighted by its probability:"],
    ["math", R`\mathbb{E}[X]=\sum_x x\,P(X=x)`],
    ["p", "For a fair die: $\\frac16(1+2+3+4+5+6)=3.5$. The expectation need not be a value you can actually roll. It is the long-run average. Two properties make it easy to use: it is **linear**, $\\mathbb{E}[aX+b]=a\\,\\mathbb{E}[X]+b$ and $\\mathbb{E}[X+Y]=\\mathbb{E}[X]+\\mathbb{E}[Y]$, even if $X$ and $Y$ are dependent."],
    ["h", "Variance and standard deviation"],
    ["p", "The **variance** is the expected squared distance from the mean, and the **standard deviation** is its square root (same units as $X$):"],
    ["math", R`\mathrm{Var}(X)=\mathbb{E}\big[(X-\mu)^2\big]=\mathbb{E}[X^2]-\mu^2,\qquad \sigma=\sqrt{\mathrm{Var}(X)}`],
    ["p", "For the die: $\\mathbb{E}[X^2]=\\frac{91}{6}$, so $\\mathrm{Var}=\\frac{91}{6}-3.5^2=\\frac{35}{12}\\approx2.92$ and $\\sigma\\approx1.71$. Why square the distances? So that positive and negative misses do not cancel, and big misses count extra. (It is also what makes the maths smooth.)"],
    ["code", R`
import numpy as np

faces = np.arange(1, 7)
p = np.full(6, 1 / 6)

mean = (faces * p).sum()
ex2 = (faces ** 2 * p).sum()
var = ex2 - mean ** 2
print("E[X] =", mean, " E[X^2] =", round(ex2, 4), " Var =", round(var, 4), " std =", round(var ** 0.5, 4))

# linearity: E[2X + 1] = 2 E[X] + 1
print("E[2X+1] =", ((2 * faces + 1) * p).sum(), "=", 2 * mean + 1)
# variance: Var(aX + b) = a^2 Var(X)
print("Var(2X+1) =", round((((2 * faces + 1) - (2 * mean + 1)) ** 2 * p).sum(), 4), "=", round(4 * var, 4))`, "E[X] = 3.5  E[X^2] = 15.1667  Var = 2.9167  std = 1.7078\nE[2X+1] = 8.0 = 8.0\nVar(2X+1) = 11.6667 = 11.6667"],
    ["h", "Averages converge: the law of large numbers"],
    ["p", "You rarely know the true probabilities. Instead you average over samples, and the **sample mean** gets closer to the expectation as you collect more. Its spread shrinks like $\\sigma/\\sqrt{n}$: to halve the noise you need four times the data."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
rolls = rng.integers(1, 7, size=100000)
for n in [10, 100, 1000, 100000]:
    print(f"mean of first {n:6d} rolls = {rolls[:n].mean():.3f}")

# spread of the sample mean: roughly sigma / sqrt(n)
means = rng.integers(1, 7, size=(5000, 100)).mean(axis=1)      # 5000 experiments of 100 rolls
print("std of the mean of 100 rolls:", round(means.std(), 3), " theory:", round(1.708 / 10, 3))`, "mean of first     10 rolls = 2.800\nmean of first    100 rolls = 3.620\nmean of first   1000 rolls = 3.600\nmean of first 100000 rolls = 3.498\nstd of the mean of 100 rolls: 0.17  theory: 0.171"],
    ["h", "Covariance and correlation"],
    ["p", "For two variables, the **covariance** $\\mathrm{Cov}(X,Y)=\\mathbb{E}[(X-\\mu_X)(Y-\\mu_Y)]$ is positive when they tend to be above their means together, negative when one is high while the other is low, and 0 if unrelated. Dividing by both standard deviations gives the **correlation** in $[-1,1]$, which does not depend on units. Put all the pairwise covariances in a matrix and you get the covariance matrix used by [PCA](ml.html#pca) (see [Eigenvalues](math.html#eigen))."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(1)
height = rng.normal(170, 10, 500)
weight = 0.9 * (height - 170) + 70 + rng.normal(0, 6, 500)     # related to height
noise = rng.normal(0, 1, 500)                                  # unrelated

X = np.column_stack([height, weight, noise])
print("means:", X.mean(axis=0).round(2))
print("variance (ddof=1):", X.var(axis=0, ddof=1).round(1))
print("correlation matrix:\n", np.corrcoef(X.T).round(2))

# standardising = subtract the mean, divide by the std: every column now has mean 0 and std 1
Z = (X - X.mean(axis=0)) / X.std(axis=0)
print("after standardising:", Z.mean(axis=0).round(2) + 0.0, Z.std(axis=0).round(2))`, "means: [ 1.6964e+02  6.9240e+01 -3.0000e-02]\nvariance (ddof=1): [ 83.5 107.7   1.1]\ncorrelation matrix:\n [[1.   0.79 0.03]\n [0.79 1.   0.08]\n [0.03 0.08 1.  ]]\nafter standardising: [0. 0. 0.] [1. 1. 1.]"],
    ["note", "NumPy's `np.var` divides by $n$ by default (the variance of the data you have). For an *estimate* of a population's variance from a sample, divide by $n-1$: `ddof=1`. The difference is small for big data and large for tiny samples."],
    ["h", "Where ML uses it"],
    ["list", [
      "**The loss is an expectation.** We would love to minimise $\\mathbb{E}[\\text{loss}]$ over all possible data; we minimise the average over the training set instead (**empirical risk**, see [Loss Functions](ml.html#loss-functions)).",
      "**Mini-batch gradients** are noisy averages: their expectation equals the full-data gradient, and their variance shrinks with batch size. See [the training loop](dl.html#training-loop).",
      "**Standardisation and normalisation** use the mean and standard deviation, see [Feature Engineering](ml.html#feature-engineering) and [LayerNorm](dl.html#normalization).",
      "**Bias-variance** is a statement about expectation and variance of a model's predictions over different training sets, see [Bias-Variance](ml.html#bias-variance).",
    ]],
  ],
},
{
  id: "distributions", group: "Probability", title: "Common Distributions",
  sub: "A few named shapes of randomness (coin flip, die, bell curve) cover most of what ML models assume about data and noise.",
  tags: ["probability"],
  blocks: [
    ["p", "A **distribution** says how likely each value of a random variable is. You only need a handful of them, and they map directly to ML tasks."],
    ["table", ["Distribution", "What it models", "Parameters", "In ML"], [
      ["**Bernoulli**", "One yes/no event (a coin flip)", "$p$ = chance of yes", "Output of [logistic regression](ml.html#logistic-regression): $P(y=1\\mid x)$"],
      ["**Binomial**", "Number of yeses in $n$ independent tries", "$n,\\ p$", "Number of correct predictions out of $n$; A/B test counts"],
      ["**Categorical**", "One outcome out of $K$ options", "probabilities $p_1..p_K$ adding to 1", "The **softmax output** of a classifier; picking the next token in an LLM"],
      ["**Uniform**", "All values in a range equally likely", "range $[a,b]$", "Random initialisation, random sampling"],
      ["**Normal (Gaussian)**", "Sums of many small independent effects; noise", "mean $\\mu$, std $\\sigma$", "Noise in [linear regression](ml.html#linear-regression), weight initialisation, VAEs"],
    ]],
    ["h", "The normal distribution"],
    ["p", "The bell curve has the density $p(x)=\\dfrac{1}{\\sigma\\sqrt{2\\pi}}\\exp\\!\\Big(-\\dfrac{(x-\\mu)^2}{2\\sigma^2}\\Big)$. For a continuous variable the curve's height is a **density**, not a probability: probabilities are **areas** under it. Move $\\mu$, $\\sigma$ and the shaded range:"],
    ["viz", "math_normal"],
    ["p", "Remember the **68-95-99.7 rule**: about 68% of the mass lies within one $\\sigma$ of the mean, 95% within two, 99.7% within three. A value more than three $\\sigma$ out is rare, which is why 'z-scores' $z=(x-\\mu)/\\sigma$ are used to spot outliers."],
    ["h", "Why the bell curve is everywhere: the central limit theorem"],
    ["p", "Add up (or average) many independent random pieces and the result looks normal, **whatever the shape of the pieces**. That is why measurement noise, heights and exam scores are roughly bell-shaped, and why Gaussian noise is a reasonable default assumption."],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
u = rng.uniform(0, 1, size=(20000, 30))      # 20000 experiments, each with 30 uniform numbers (flat, not bell-shaped!)
means = u.mean(axis=1)

print("one uniform: mean", round(u[:, 0].mean(), 3), " std", round(u[:, 0].std(), 3))
print("mean of 30 : mean", round(means.mean(), 3), " std", round(means.std(), 3),
      " theory std", round((1 / 12) ** 0.5 / 30 ** 0.5, 3))
print("within 1 std of the mean:", round(np.mean(np.abs(means - means.mean()) < means.std()), 3), "(normal: 0.683)")
print("within 2 std of the mean:", round(np.mean(np.abs(means - means.mean()) < 2 * means.std()), 3), "(normal: 0.954)")

plt.figure(figsize=(6, 3.2))
plt.hist(means, bins=50, density=True)
xs = np.linspace(0.3, 0.7, 200)
s = means.std()
plt.plot(xs, np.exp(-0.5 * ((xs - 0.5) / s) ** 2) / (s * np.sqrt(2 * np.pi)), "r", lw=2)
plt.title("Averages of 30 uniform numbers follow a bell curve")
plt.show()`, "one uniform: mean 0.497  std 0.288\nmean of 30 : mean 0.5  std 0.053  theory std 0.053\nwithin 1 std of the mean: 0.681 (normal: 0.683)\nwithin 2 std of the mean: 0.954 (normal: 0.954)"],
    ["h", "Sampling from distributions in NumPy"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
n = 100000

coin = rng.binomial(1, 0.3, n)                           # Bernoulli(0.3): 1 with probability 0.3
print("Bernoulli(0.3): mean", coin.mean().round(3), "  theory", 0.3, "  var", coin.var().round(3), "  theory", 0.21)

heads = rng.binomial(10, 0.3, n)                         # Binomial(10, 0.3)
print("Binomial(10,0.3): mean", heads.mean().round(2), "  theory", 3.0)

probs = np.array([0.7, 0.2, 0.1])                        # a categorical distribution over 3 classes
draws = rng.choice(3, size=n, p=probs)
print("Categorical:", (np.bincount(draws) / n).round(3), "  theory", probs)

g = rng.normal(5, 2, n)                                  # Normal(mu=5, sigma=2)
print("Normal(5,2): mean", g.mean().round(2), " std", g.std().round(2))`, "Bernoulli(0.3): mean 0.299   theory 0.3   var 0.21   theory 0.21\nBinomial(10,0.3): mean 2.99   theory 3.0\nCategorical: [0.7   0.201 0.1  ]   theory [0.7 0.2 0.1]\nNormal(5,2): mean 5.0  std 2.0"],
    ["tip", "An LLM's output layer produces a **categorical** distribution over the vocabulary (via softmax); generating text means sampling from it. Temperature and top-p change that distribution before sampling, see [Sampling](llm.html#sampling)."],
    ["warn", "Do not assume everything is normal. Prices, word counts and response times are skewed with heavy tails, where 'rare' events happen more often than the 3-sigma rule suggests. Plot a histogram before you trust a Gaussian assumption."],
  ],
},
{
  id: "mle", group: "Probability", title: "Maximum Likelihood Estimation",
  sub: "Choose the parameters that make the data you saw as probable as possible: the principle behind squared error, cross-entropy and most training losses.",
  tags: ["probability", "loss"],
  blocks: [
    ["p", "Suppose you flip a coin 10 times and get 7 heads. What is the chance of heads, $p$? Your gut says 0.7. **Maximum likelihood** turns that gut feeling into a method: pick the $p$ under which what you actually saw is most probable."],
    ["h", "Likelihood"],
    ["p", "The **likelihood** $\\mathcal{L}(p)$ is the probability of the observed data viewed as a function of the parameter. For 7 heads and 3 tails in a particular order, it is $p^7(1-p)^3$. At $p=0.5$ it is $0.5^{10}\\approx0.00098$; at $p=0.7$ it is about $0.0022$, more than twice as big. So $0.7$ explains the data better than $0.5$."],
    ["h", "Take the log"],
    ["p", "Likelihoods of many independent examples are products of many small numbers: they underflow to 0 and are awkward to differentiate. The log turns products into sums and keeps the same maximiser (log is increasing). Maximising the **log-likelihood** is the same as minimising the **negative log-likelihood (NLL)**, the thing we call a loss:"],
    ["math", R`\hat\theta=\arg\max_\theta\prod_{i=1}^n p(x_i\mid\theta)=\arg\min_\theta\ -\sum_{i=1}^n\log p(x_i\mid\theta)`],
    ["p", "For the coin, $\\log\\mathcal{L}=7\\log p+3\\log(1-p)$. Setting the derivative to zero, $\\frac7p-\\frac3{1-p}=0$, gives $\\hat p=\\frac{7}{10}$: the fraction of heads, as your gut said. See it on a curve:"],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt

heads, tails = 7, 3
p = np.linspace(0.01, 0.99, 99)
loglik = heads * np.log(p) + tails * np.log(1 - p)

best = p[np.argmax(loglik)]
print("p with highest log-likelihood:", round(float(best), 2))
print("log-lik at p=0.5:", round(float(heads * np.log(0.5) + tails * np.log(0.5)), 3))
print("log-lik at p=0.7:", round(float(heads * np.log(0.7) + tails * np.log(0.3)), 3))

plt.figure(figsize=(6, 3.2))
plt.plot(p, loglik)
plt.axvline(best, color="r", ls="--", label=f"max at p = {best:.2f}")
plt.xlabel("p (chance of heads)"); plt.ylabel("log-likelihood"); plt.legend()
plt.title("7 heads in 10 flips")
plt.show()`, "p with highest log-likelihood: 0.7\nlog-lik at p=0.5: -6.931\nlog-lik at p=0.7: -6.109"],
    ["h", "Gaussian data: mean and variance"],
    ["p", "If data are normal with unknown $\\mu$ and $\\sigma^2$, the maximum likelihood estimates are the sample mean $\\hat\\mu=\\frac1n\\sum x_i$ and the variance $\\hat\\sigma^2=\\frac1n\\sum(x_i-\\hat\\mu)^2$ (with $1/n$, not $1/(n-1)$). A good sanity check is to find them by brute-force maximisation:"],
    ["code", R`
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(0)
x = rng.normal(loc=4.0, scale=1.5, size=200)

def nll(theta):                          # negative log-likelihood of a normal
    mu, log_sigma = theta
    s = np.exp(log_sigma)                # parametrise log(sigma) so that sigma stays positive
    return np.sum(0.5 * np.log(2 * np.pi * s ** 2) + (x - mu) ** 2 / (2 * s ** 2))

res = minimize(nll, x0=[0.0, 0.0])
print("numerical MLE:", res.x[0].round(3), np.exp(res.x[1]).round(3))
print("closed form  :", x.mean().round(3), x.std().round(3), " (np.std uses 1/n)")`, "numerical MLE: 4.023 1.442\nclosed form  : 4.023 1.442  (np.std uses 1/n)"],
    ["h", "Why this explains our loss functions"],
    ["p", "Different assumptions about the noise give different losses, all through the same recipe 'minimise the NLL':"],
    ["table", ["Assumption about the data", "NLL becomes", "Loss you know"], [
      ["$y=f(x)+$ Gaussian noise", "$\\sum(y_i-f(x_i))^2$ (plus constants)", "**Mean squared error**: [linear regression](ml.html#linear-regression)"],
      ["$y$ is yes/no with $P(y=1)=\\sigma(f(x))$ (Bernoulli)", "$-\\sum[y\\log p+(1-y)\\log(1-p)]$", "**Binary cross-entropy / log loss**: [logistic regression](ml.html#logistic-regression)"],
      ["$y$ is one of $K$ classes with softmax probabilities (Categorical)", "$-\\sum\\log p_{y_i}$", "**Cross-entropy**: [classification losses](dl.html#dl-losses)"],
      ["Laplace noise (heavier tails)", "$\\sum|y_i-f(x_i)|$", "Mean absolute error"],
      ["a Gaussian **prior** on weights too (MAP)", "NLL $+\\ \\lambda\\lVert w\\rVert^2$", "**L2 regularization** / weight decay, see [Regularization](ml.html#regularization)"],
    ]],
    ["p", "So 'minimising MSE' is not an arbitrary choice: it is maximum likelihood under Gaussian noise. We can verify that numerically: the line that maximises the Gaussian likelihood is the least-squares line."],
    ["code", R`
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(1)
x = rng.uniform(0, 10, 100)
y = 2.0 * x + 1.0 + rng.normal(0, 1.0, 100)
A = np.column_stack([x, np.ones_like(x)])

def nll(theta):                                  # Gaussian NLL with sigma fixed at 1
    resid = y - A @ theta
    return 0.5 * np.sum(resid ** 2)              # + constants that do not depend on theta

w_mle = minimize(nll, x0=[0.0, 0.0]).x
w_ls = np.linalg.lstsq(A, y, rcond=None)[0]
print("MLE (slope, intercept)         :", w_mle.round(3))
print("least squares (slope, intercept):", w_ls.round(3))

# cross-entropy as NLL: probabilities the model gave to the TRUE classes
p_true = np.array([0.9, 0.8, 0.3, 0.95])
print("mean NLL of 4 predictions:", round(float(-np.log(p_true).mean()), 4))`, "MLE (slope, intercept)         : [2.002 0.931]\nleast squares (slope, intercept): [2.002 0.931]\nmean NLL of 4 predictions: 0.3959"],
    ["tip", "Practice: [Gaussian MLE](practice.html#p/gaussian-mle). Then see how the same NLL idea trains language models in [Pretraining](llm.html#pretraining)."],
    ["warn", "MLE can overfit with little data: seeing 3 heads in 3 flips gives $\\hat p=1$, which says tails is impossible. Priors (Bayesian estimation, regularization, smoothing as in [Naive Bayes](ml.html#naive-bayes)) are the fix."],
  ],
},
{
  id: "entropy", group: "Information Theory", title: "Entropy, Cross-Entropy & KL Divergence",
  sub: "A way to measure surprise, and the reason cross-entropy is the standard loss for classifiers and language models.",
  tags: ["information theory", "loss"],
  blocks: [
    ["p", "Information theory starts with a simple question: how **surprised** are you by an outcome? A coin landing heads when it is fair is mildly surprising; a lottery win is hugely surprising; the sun rising tomorrow is not surprising at all. Surprise should depend only on how likely the event was."],
    ["h", "Surprise and entropy"],
    ["p", "The surprise (or **information**) of an event with probability $p$ is $-\\log p$. Rare events get large values, and a certain event ($p=1$) gets 0. With $\\log_2$ the unit is **bits**; with the natural log it is **nats** (what deep-learning libraries use). **Entropy** is the average surprise of a distribution:"],
    ["math", R`H(p)=-\sum_i p_i\log p_i`, "the average number of bits needed to describe an outcome drawn from p"],
    ["table", ["Distribution", "Entropy (bits)", "Meaning"], [
      ["Fair coin (0.5, 0.5)", "1.0", "maximally unpredictable for 2 outcomes"],
      ["Biased coin (0.9, 0.1)", "0.469", "mostly predictable"],
      ["Certain (1, 0)", "0", "no surprise at all"],
      ["Fair die", "2.585", "$\\log_2 6$: uniform over $K$ outcomes has the maximum, $\\log_2 K$"],
    ]],
    ["h", "Cross-entropy and KL divergence"],
    ["p", "Now suppose the truth is $p$ but your model believes $q$. The average surprise **under your wrong belief** is the **cross-entropy**. It is always at least the entropy, and the gap is the **KL divergence**:"],
    ["math", R`H(p,q)=-\sum_i p_i\log q_i=H(p)+\mathrm{KL}(p\,\|\,q),\qquad \mathrm{KL}(p\,\|\,q)=\sum_i p_i\log\frac{p_i}{q_i}\ \ge 0`],
    ["p", "KL is zero exactly when $q=p$ and grows as the model's belief drifts away. It is not symmetric: $\\mathrm{KL}(p\\|q)\\neq\\mathrm{KL}(q\\|p)$, so it is a 'divergence', not a distance. Change the true distribution $p$ and the model's $q$ and watch the three numbers:"],
    ["viz", "math_entropy"],
    ["h", "Why cross-entropy is the classification loss"],
    ["p", "In classification, the truth $p$ is **one-hot** (all probability on the correct class), so $H(p)=0$ and the cross-entropy collapses to $-\\log q_{\\text{true class}}$: *minus the log of the probability the model gave the right answer.* That is exactly the negative log-likelihood from [MLE](math.html#mle). Confident and right costs almost nothing; confident and wrong costs a lot:"],
    ["code", R`
import numpy as np

def entropy(p):
    p = np.asarray(p, float)
    return float(-np.sum(p[p > 0] * np.log2(p[p > 0])))

def cross_entropy(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(-np.sum(p[p > 0] * np.log2(q[p > 0])))

def kl(p, q):
    return cross_entropy(p, q) - entropy(p)

print("fair coin   :", entropy([0.5, 0.5]))
print("biased coin :", round(entropy([0.9, 0.1]), 3))
print("fair die    :", round(entropy([1 / 6] * 6), 3), " log2(6) =", round(np.log2(6), 3))

p = [0.8, 0.15, 0.05]
q = [0.4, 0.4, 0.2]
print("H(p) =", round(entropy(p), 3), " H(p,q) =", round(cross_entropy(p, q), 3), " KL(p||q) =", round(kl(p, q), 3))
print("KL(q||p) =", round(kl(q, p), 3), "  <- not symmetric")
print("KL(p||p) =", kl(p, p))

# classification loss: one-hot truth, in nats like PyTorch
onehot = np.array([0.0, 1.0, 0.0])
for probs in ([0.1, 0.8, 0.1], [0.3, 0.4, 0.3], [0.8, 0.1, 0.1], [0.01, 0.01, 0.98]):
    print(f"model says {probs} -> loss {-np.log(probs[1]):.3f}")`, "fair coin   : 1.0\nbiased coin : 0.469\nfair die    : 2.585  log2(6) = 2.585\nH(p) = 0.884  H(p,q) = 1.372  KL(p||q) = 0.488\nKL(q||p) = 0.566   <- not symmetric\nKL(p||p) = 0.0\nmodel says [0.1, 0.8, 0.1] -> loss 0.223\nmodel says [0.3, 0.4, 0.3] -> loss 0.916\nmodel says [0.8, 0.1, 0.1] -> loss 2.303\nmodel says [0.01, 0.01, 0.98] -> loss 4.605"],
    ["h", "Perplexity"],
    ["p", "Language models report **perplexity** $=e^{\\text{cross-entropy}}$ (with the loss in nats). It can be read as 'the model is as unsure as if it were choosing uniformly among this many tokens'. A perplexity of 1 means perfect prediction; a vocabulary-sized perplexity means guessing. See [Pretraining](llm.html#pretraining)."],
    ["code", R`
import numpy as np

# probabilities a tiny language model gave to the 5 correct next tokens
p_correct = np.array([0.5, 0.25, 0.1, 0.6, 0.05])
ce = -np.log(p_correct).mean()                 # cross-entropy in nats (per token)
print("cross-entropy per token:", round(float(ce), 3), "nats")
print("perplexity:", round(float(np.exp(ce)), 2))

# a model that is uniform over a 50,000-token vocabulary
print("uniform over 50k tokens: perplexity =", round(float(np.exp(-np.log(1 / 50000))), 1))`, "cross-entropy per token: 1.578 nats\nperplexity: 4.84\nuniform over 50k tokens: perplexity = 50000.0"],
    ["tip", "In PyTorch, `nn.CrossEntropyLoss` takes raw logits and applies a stable log-softmax itself, so never put a softmax in front of it. Its gradient with respect to the logits is the clean $p-y$ from [Jacobians & Matrix Calculus](math.html#jacobians). Practice: [Entropy, Cross-Entropy and KL](practice.html#p/entropy-cross-entropy-kl)."],
    ["note", "KL divergence shows up well beyond classification: in VAEs (keep the latent distribution near a prior), in RLHF/DPO (stay near the original model, see [RLHF](llm.html#rlhf)) and in distillation (match the teacher's distribution, see [Distillation](llm.html#distillation))."],
  ],
},
{
  id: "optimization", group: "Optimization", title: "Convexity, Minima & Saddle Points",
  sub: "When gradient descent is guaranteed to find the best answer, and what can go wrong in the bumpy landscapes of deep learning.",
  tags: ["optimization"],
  blocks: [
    ["p", "Gradient descent walks downhill until the slope is zero. Whether that leads to the *best* answer depends entirely on the shape of the loss surface. Two shapes matter: bowls, and everything else."],
    ["h", "Convex: one valley"],
    ["p", "A function is **convex** if the straight line between any two points on its graph lies on or above the graph. Intuitively it is a bowl: a **single** valley, no false bottoms. For a convex loss, every local minimum is the global minimum, and gradient descent with a small enough learning rate reaches it. Nice cases: [linear regression](ml.html#linear-regression) with squared error, [logistic regression](ml.html#logistic-regression), SVMs. In one dimension, convex means $f''(x)\\ge0$ everywhere (the slope only ever increases)."],
    ["diagram", "math_landscape", [], "Left: a convex loss has one valley. Right: a non-convex loss has several valleys (local minima) and flat spots where descent can stall or stop."],
    ["h", "Non-convex: neural networks"],
    ["p", "The loss of a neural network is **non-convex**: with many valleys of different depth, gradient descent ends in one that depends on where it started. Run it from different starting points on a simple two-valley function:"],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x ** 4 - 3 * x ** 2 + x            # two valleys of different depth
df = lambda x: 4 * x ** 3 - 6 * x + 1

def descend(x, lr=0.02, steps=300):
    path = [x]
    for _ in range(steps):
        x -= lr * df(x)
        path.append(x)
    return path

xs = np.linspace(-2.2, 2.2, 300)
plt.figure(figsize=(6, 3.2))
plt.plot(xs, f(xs), "k")
for x0, c in [(-2.0, "tab:blue"), (0.1, "tab:green"), (2.0, "tab:red")]:
    path = descend(x0)
    print(f"start {x0:5.1f} -> ends at x = {path[-1]:6.3f}, f = {f(path[-1]):6.3f}")
    plt.plot(path[0], f(path[0]), "o", c=c); plt.plot(path[-1], f(path[-1]), "*", c=c, ms=14)
plt.title("Different starts reach different minima")
plt.show()`, "start  -2.0 -> ends at x = -1.301, f = -3.514\nstart   0.1 -> ends at x = -1.301, f = -3.514\nstart   2.0 -> ends at x =  1.131, f = -1.070"],
    ["h", "Saddle points"],
    ["p", "A place with zero gradient is a **critical point**, and it need not be a minimum. At a **saddle point** the surface curves up in some directions and down in others (a mountain pass, or a horse saddle). The tool to tell them apart is the **Hessian** ([Jacobians & Matrix Calculus](math.html#jacobians)) of second derivatives:"],
    ["table", ["Eigenvalues of the Hessian at a critical point", "What it is"], [
      ["all positive", "local **minimum** (curves up everywhere)"],
      ["all negative", "local **maximum**"],
      ["mixed signs", "**saddle point**"],
      ["some zero", "flat directions: undecided without more information"],
    ]],
    ["viz", "math_saddle"],
    ["p", "Why do saddles matter? In a space with millions of dimensions, for a critical point to be a true minimum **every** direction must curve up, while a saddle only needs one direction to curve down. So saddles are far more common than bad local minima, and descent can crawl across their flat regions. Starting exactly on the saddle's axis you never leave it; but any tiny bit of noise gets amplified and you slide off. This is one reason **stochastic** gradients and [momentum](dl.html#optimizers) help in practice."],
    ["code", R`
import numpy as np

# f(x, y) = x^2 - y^2 has a critical point at (0, 0): gradient (2x, -2y) = 0 there
H = np.array([[2.0, 0.0], [0.0, -2.0]])            # its Hessian (constant)
print("Hessian eigenvalues:", np.linalg.eigvalsh(H), "-> mixed signs: a saddle point")

grad = lambda p: np.array([2 * p[0], -2 * p[1]])
for y0 in [0.0, 1e-6]:
    p = np.array([1.0, y0])
    for _ in range(60):
        p = p - 0.1 * grad(p)
    print(f"start y = {y0:g}: after 60 steps (x, y) = ({p[0]:.4f}, {p[1]:.4f})")

# a bowl: positive eigenvalues
Hb = np.array([[2.0, 0.5], [0.5, 1.0]])
print("bowl Hessian eigenvalues:", np.linalg.eigvalsh(Hb).round(3), "-> minimum")`, "Hessian eigenvalues: [-2.  2.] -> mixed signs: a saddle point\nstart y = 0: after 60 steps (x, y) = (0.0000, 0.0000)\nstart y = 1e-06: after 60 steps (x, y) = (0.0000, 0.0563)\nbowl Hessian eigenvalues: [0.793 2.207] -> minimum"],
    ["h", "Shape of the bowl: conditioning"],
    ["p", "Even for a convex bowl, speed depends on its shape. If it is round, gradient descent heads straight to the bottom. If it is a long narrow valley (curvature very different in different directions; a large ratio of the largest to smallest Hessian eigenvalue, the **condition number**) it zig-zags, as in [Gradients](math.html#gradients). Feature scaling makes the bowl rounder, and adaptive optimizers rescale each direction ([Optimizers](dl.html#optimizers))."],
    ["tip", "In practice you cannot compute the Hessian of a big network (it has parameters² entries), so you watch the loss curve instead. Flat loss early on may be a plateau or saddle, not a bug; a loss that never goes down is usually a learning-rate or data problem. See [Debugging Training](dl.html#debugging-training)."],
    ["note", "Surprisingly, large networks usually train well despite non-convexity: many local minima found by gradient descent are nearly as good as the global one. Full theory is still active research; a safe takeaway is 'good initialization, a sensible learning rate and noisy gradients usually work'."],
  ],
},

);
