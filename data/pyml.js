/* Python for ML: setup, NumPy, pandas, plotting, scikit-learn
   Outputs written as "?" are filled in by running the code: node tools/fill.js data/pyml.js */
PYML.push(
{
  id: "setup", group: "Start Here", title: "Python for ML: Setup & Tools",
  sub: "The handful of libraries every ML project uses, how to install them in a clean environment, and the Python you need to read the rest of the site.",
  tags: ["setup"],
  blocks: [
    ["p", "Almost all machine learning code is Python glued to a few fast libraries. Python itself is slow, but the libraries do the heavy work in compiled C, C++ or Fortran, so a Python line like `X @ w` runs at native speed. Your job is to learn the **vocabulary** of those libraries: what shape things have, which function does what, and which mistakes bite."],
    ["table", ["Library", "What it is for", "You will use it to"], [
      ["**NumPy**", "Fast arrays and maths on them", "Hold features as a matrix, do linear algebra, write algorithms from scratch"],
      ["**pandas**", "Tables with labelled columns", "Load a CSV, clean it, filter, group, join"],
      ["**matplotlib**", "Plots", "Look at data and learning curves"],
      ["**scikit-learn**", "Classic ML models and tools", "Split data, scale, train, evaluate, tune"],
      ["**PyTorch**", "Tensors with automatic gradients (GPU)", "Deep learning: see [PyTorch Basics](dl.html#pytorch-basics)"],
    ]],
    ["h", "A clean environment"],
    ["p", "Install packages into a **virtual environment** (a private folder of packages for one project), not into your system Python. Then two projects can use different library versions without fighting. These commands run in a terminal, not in this page:"],
    ["code", R`
# create and activate an environment named .venv
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell / cmd)
source .venv/bin/activate       # macOS / Linux`, "", "bash"],
    ["code", R`
# install the core stack, then start Jupyter
python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn jupyterlab
jupyter lab                      # opens a notebook in your browser`, "", "bash"],
    ["code", R`
# record exact versions so someone else (or you, in 6 months) can rebuild it
pip freeze > requirements.txt
pip install -r requirements.txt`, "", "bash"],
    ["tip", "No installation at all? **Google Colab** (colab.research.google.com) gives a free notebook in the browser with NumPy, pandas and scikit-learn already installed, plus a GPU for deep learning. The runnable code blocks on this very site also run in your browser, with no setup."],
    ["h", "Notebook or script?"],
    ["p", "A **Jupyter notebook** mixes code, outputs and notes in cells; it is the best place to explore data and draw plots. A plain **`.py` script** is better for anything you want to rerun or ship. A good habit: explore in a notebook, then move the stable parts into functions in `.py` files."],
    ["list", [
      "In a notebook, the last expression of a cell is printed automatically. In a script you need `print(...)`.",
      "Run cells **top to bottom** before trusting a result. Hidden state (a variable defined in a cell you later deleted) is the classic notebook bug. *Restart kernel and Run All* is the cure.",
      "`?` is built-in help: `np.mean?` in Jupyter shows the docstring. In any Python, `help(np.mean)` does the same.",
    ]],
    ["h", "The imports you will see everywhere"],
    ["p", "Three short names are universal conventions. Everyone writes them, so you can read anyone's code:"],
    ["code", R`
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = np.array([1, 2, 3])
df = pd.DataFrame({"x": a, "y": a ** 2})
print(type(a).__name__, type(df).__name__)
print(df)`, "ndarray DataFrame\n   x  y\n0  1  1\n1  2  4\n2  3  9"],
    ["h", "The Python you need (a quick refresher)"],
    ["p", "If you can read this block, you know enough Python for the rest of the page. Python lists are general containers; NumPy arrays (next sections) are what ML code actually computes with."],
    ["code", R`
nums = [3, 1, 4, 1, 5]
squares = [n ** 2 for n in nums]              # list comprehension
evens = [n for n in nums if n % 2 == 0]       # with a filter
pairs = list(zip(nums, squares))              # walk two lists together
for i, (n, s) in enumerate(pairs[:2]):        # index + unpacking
    print(f"item {i}: {n} squared is {s}")
first, *rest = nums                           # star-unpacking
d = {"lr": 0.1, "epochs": 20}                 # dict: name -> value
print(squares, evens, first, rest, d["lr"])
print(7 / 2, 7 // 2, 7 % 2)                   # true division, floor division, remainder`, "item 0: 3 squared is 9\nitem 1: 1 squared is 1\n[9, 1, 16, 1, 25] [4] 3 [1, 4, 1, 5] 0.1\n3.5 3 1"],
    ["danger", "Python lists and NumPy arrays look alike but behave differently: `[1, 2] + [3]` joins lists into `[1, 2, 3]`, while `np.array([1, 2]) + 3` adds 3 to every element. When a result looks strangely long, check whether you accidentally have a list."],
    ["note", "Maths background for the NumPy sections lives in [Math for ML](math.html#why-math): [vectors](math.html#vectors) and [matrices](math.html#matrices) are exactly NumPy's 1-D and 2-D arrays. After this page, you are ready for [the ML workflow](ml.html#workflow)."],
  ],
},
{
  id: "numpy-arrays", group: "NumPy", title: "Arrays, dtypes & Shapes",
  sub: "A NumPy array is a block of same-typed numbers with a shape. Knowing the shape and dtype of everything is half of debugging ML code.",
  tags: ["numpy"],
  blocks: [
    ["p", "A Python list can hold anything, so every item carries extra baggage. A NumPy **ndarray** holds numbers of **one type** packed side by side in memory. That makes it small and lets compiled code loop over it at full speed. Everything in ML (a dataset, an image, the weights of a layer) is an ndarray, so you will look at three properties constantly: **shape**, **dtype** and **ndim**."],
    ["code", R`
import numpy as np

a = np.array([1, 2, 3])                       # from a list
M = np.array([[1, 2, 3], [4, 5, 6]])          # list of lists -> 2-D
print(a.shape, a.ndim, a.size, a.dtype)
print(M.shape, M.ndim, M.size, M.dtype)

print(np.zeros((2, 3)))                       # all zeros
print(np.ones(3))                             # all ones
print(np.full((2, 2), 7))                     # a constant
print(np.arange(0, 10, 3))                    # like range(), but an array
print(np.linspace(0, 1, 5))                   # 5 evenly spaced points, endpoints included
print(np.eye(3))                              # identity matrix`, "(3,) 1 3 int64\n(2, 3) 2 6 int64\n[[0. 0. 0.]\n [0. 0. 0.]]\n[1. 1. 1.]\n[[7 7]\n [7 7]]\n[0 3 6 9]\n[0.   0.25 0.5  0.75 1.  ]\n[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]"],
    ["p", "The **shape** is a tuple with the size along each dimension, outermost first. `(2, 3)` means 2 rows and 3 columns. A shape like `(3,)` is a flat list of 3 numbers (one dimension). The **ndim** is just `len(shape)`."],
    ["h", "Shapes you will meet in ML"],
    ["table", ["Thing", "Typical shape", "Meaning"], [
      ["Feature matrix `X`", "`(n, d)`", "n examples (rows), d features (columns)"],
      ["Labels `y`", "`(n,)`", "one answer per example"],
      ["Weights of a linear model", "`(d,)`", "one number per feature"],
      ["A batch of grey images", "`(B, H, W)`", "batch, height, width"],
      ["A batch of colour images", "`(B, H, W, 3)` or `(B, 3, H, W)`", "the last/first axis is the colour channel"],
      ["A batch of token embeddings", "`(B, T, D)`", "batch, sequence length, embedding size"],
    ]],
    ["tip", "When code misbehaves, print `x.shape` and `x.dtype` of every variable first. Most bugs in ML code are a shape that is not what you thought."],
    ["h", "reshape: same numbers, new shape"],
    ["code", R`
import numpy as np
a = np.arange(6)                  # [0 1 2 3 4 5], shape (6,)
print(a.reshape(2, 3))            # 2 rows, 3 columns
print(a.reshape(3, -1).shape)     # -1 means "work it out": 6 / 3 = 2
col = a.reshape(-1, 1)            # a column vector
print(a.shape, col.shape)
print(a[:, None].shape)           # same thing, spelled with an extra axis
print(col.ravel().shape)          # flatten back to 1-D`, "[[0 1 2]\n [3 4 5]]\n(3, 2)\n(6,) (6, 1)\n(6, 1)\n(6,)"],
    ["p", "`reshape` never changes the numbers or their order (read left to right, top to bottom); it only changes how they are grouped. The product of the shape must stay the same: 6 numbers fit in `(2, 3)`, `(3, 2)`, `(6, 1)` but not `(4, 2)`."],
    ["danger", "**Shape `(n,)` is not shape `(n, 1)`.** Subtracting a column from a flat array does not give element-wise differences; it silently builds an `(n, n)` table (see [Broadcasting](pyml.html#broadcasting)). Many scikit-learn outputs are `(n,)` while many hand-made arrays are `(n, 1)`. When you compute a loss, make both sides the same shape."],
    ["code", R`
import numpy as np
y_true = np.array([1.0, 2.0, 3.0])          # shape (3,)
y_pred = np.array([[1.5], [2.0], [2.0]])    # shape (3, 1) - a column
wrong = y_pred - y_true                     # broadcasts to (3, 3)!
print(wrong.shape)
right = y_pred.ravel() - y_true             # same shape -> (3,)
print(right.shape, float(np.mean(right ** 2)))`, "(3, 3)\n(3,) 0.4166666666666667"],
    ["h", "dtypes: what kind of number is it?"],
    ["p", "Every array has one **dtype**. The common ones: `int64` (whole numbers), `float64` (decimals, NumPy's default), `float32` (half the memory; the usual choice for deep learning), `bool` and `uint8` (0 to 255, used for image pixels). If you mix types NumPy **upcasts** to the one that can hold everything."],
    ["code", R`
import numpy as np
print(np.array([1, 2, 3]).dtype)                 # ints
print(np.array([1, 2.5, 3]).dtype)               # one float makes all float
print(np.array([1, 2, 3], dtype=np.float32).dtype)
x = np.array([1.9, -1.9])
print(x.astype(int))                             # astype truncates toward zero
print((np.array([1, 2, 3]) / 2).dtype)           # / always gives floats
print(np.array([7, 8]) // 2, 7 % 3)              # // is floor division

pix = np.array([200], dtype=np.uint8)
print(pix + pix)                                 # uint8 wraps around at 256!
big = np.zeros((1000, 1000))
print(big.nbytes, big.astype(np.float32).nbytes) # bytes used`, "int64\nfloat64\nfloat32\n[ 1 -1]\nfloat64\n[3 4] 1\n[144]\n8000000 4000000"],
    ["warn", "Integer arrays wrap around silently when they overflow (`uint8` 200 + 200 = 144). Convert images to float before arithmetic: `img.astype(np.float32) / 255`. And remember that `/` is true division (`7 / 2 = 3.5`) while `//` drops the fraction (`7 // 2 = 3`); mixing them up is a classic bug in indices and batch sizes."],
    ["h", "Views and copies"],
    ["p", "Slicing does not copy the data; it gives you a **view**, a new window onto the same memory. Changing the view changes the original. That is fast and memory-friendly, but surprising the first time."],
    ["code", R`
import numpy as np
a = np.arange(6)
v = a[1:4]               # a view: shares memory with a
v[0] = 99
print(a)                 # a changed too!

c = a[1:4].copy()        # an independent copy
c[0] = -1
print(a, c)
print(np.shares_memory(a, v), np.shares_memory(a, c))

b = a.reshape(2, 3)      # reshape usually returns a view as well
b[0, 0] = 7
print(a)`, "[ 0 99  2  3  4  5]\n[ 0 99  2  3  4  5] [-1  2  3]\nTrue False\n[ 7 99  2  3  4  5]"],
    ["tip", "Rule of thumb: **basic slicing and reshape give views; boolean masks and integer-list (fancy) indexing give copies** ([next section](pyml.html#numpy-indexing)). When you need to modify data without touching the original, call `.copy()` explicitly. A function that edits its input array in place is a common source of mysterious bugs."],
    ["note", "Arrays are the computer form of [vectors](math.html#vectors) and [matrices](math.html#matrices). Practise the ideas in [Standardize Features](practice.html#p/standardize-features) and [One-Hot Encoding](practice.html#p/one-hot-encoding)."],
  ],
},
{
  id: "numpy-indexing", group: "NumPy", title: "Indexing, Slicing & Boolean Masks",
  sub: "Pick elements, rows, columns or anything that passes a test, without writing a single loop.",
  tags: ["numpy"],
  blocks: [
    ["p", "Selecting data is what you do all day in ML: take the first 80% of rows, one column, every example of class 1, the 5 highest scores. NumPy has three tools for it: **slices** (ranges), **boolean masks** (a True/False per item) and **fancy indexing** (a list of positions)."],
    ["h", "Index and slice"],
    ["code", R`
import numpy as np
M = np.arange(12).reshape(3, 4)
print(M)
print(M[1, 2])          # row 1, column 2 -> a single number
print(M[0])             # row 0 (a 1-D array)
print(M[:, 1])          # all rows, column 1
print(M[1:, :2])        # rows 1.., first two columns
print(M[:, ::2])        # every 2nd column
print(M[-1])            # last row
print(M[::-1, 0])       # column 0 with the rows reversed`, "[[ 0  1  2  3]\n [ 4  5  6  7]\n [ 8  9 10 11]]\n6\n[0 1 2 3]\n[1 5 9]\n[[4 5]\n [8 9]]\n[[ 0  2]\n [ 4  6]\n [ 8 10]]\n[ 8  9 10 11]\n[8 4 0]"],
    ["p", "A slice is `start:stop:step` where **stop is excluded**, and each dimension has its own slice, separated by commas. A single number removes that dimension (`M[0]` is 1-D); a slice keeps it (`M[0:1]` is still 2-D, shape `(1, 4)`)."],
    ["h", "Boolean masks"],
    ["p", "A comparison like `a > 5` does not give a single True/False. It gives an **array of booleans**, one per item. Using that array as an index keeps only the positions that are True. Play with the threshold and the list of positions:"],
    ["viz", "pyml_mask"],
    ["code", R`
import numpy as np
a = np.array([7, 2, 9, 4, 6, 1, 8, 3])
mask = a > 5
print(mask)
print(a[mask])                    # items where the mask is True
print(a[(a > 2) & (a < 8)])       # combine tests with & (and), | (or), ~ (not)
print(a[~(a % 2 == 0)])           # odd numbers
print(mask.sum(), mask.mean())    # count and fraction of True
a[a > 8] = 8                      # assign through a mask: clip in place
print(a)`, "[ True False  True False  True False  True False]\n[7 9 6 8]\n[7 4 6 3]\n[7 9 1 3]\n4 0.5\n[7 2 8 4 6 1 8 3]"],
    ["danger", "Combine masks with **`&`, `|`, `~`** and put **each test in parentheses**: `(a > 2) & (a < 8)`. Writing `a > 2 and a < 8` raises \"truth value of an array is ambiguous\", and `a > 2 & a < 8` is parsed as `a > (2 & a) < 8` because `&` binds tighter than `>`."],
    ["h", "Masks on a feature matrix"],
    ["p", "In ML the mask usually has one entry per **row**, so it filters examples. The same mask on `y` filters the labels, keeping `X` and `y` aligned."],
    ["code", R`
import numpy as np
X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
y = np.array([0, 1, 1, 0])
X1 = X[y == 1]                        # rows of class 1
print(X1)
print(X[:, 0] > 2)                    # mask built from the first feature
print(X[X[:, 0] > 2, 1])              # feature 1 of rows whose feature 0 > 2
print(X[:, [1, 0]])                   # reorder columns with a list
print(np.where(y == 1)[0])            # the row numbers where y == 1
print(np.where(X[:, 0] > 2, "big", "small"))   # choose per row`, "[[ 2. 20.]\n [ 3. 30.]]\n[False False  True  True]\n[30. 40.]\n[[10.  1.]\n [20.  2.]\n [30.  3.]\n [40.  4.]]\n[1 2]\n['small' 'small' 'big' 'big']"],
    ["h", "Fancy indexing: a list of positions"],
    ["code", R`
import numpy as np
a = np.array([10, 20, 30, 40, 50])
print(a[[0, 3, 3, 4]])                # any order, repeats allowed
print(a[np.array([-1, -2])])          # negatives count from the end

scores = np.array([0.2, 0.9, 0.4, 0.7, 0.1])
order = np.argsort(scores)[::-1]      # positions from highest to lowest score
print(order, scores[order])
print(order[:2])                      # top-2 positions
print(np.argmax(scores), scores.argmin())

rows = np.array([0, 1, 2])
cols = np.array([2, 0, 1])
M = np.arange(9).reshape(3, 3)
print(M[rows, cols])                  # picks (0,2), (1,0), (2,1) pairwise`, "[10 40 40 50]\n[50 40]\n[1 3 2 0 4] [0.9 0.7 0.4 0.2 0.1]\n[1 3]\n1 4\n[2 3 7]"],
    ["table", ["Kind of index", "Example", "Result is"], [
      ["Integer", "`a[2]`, `M[1, 2]`", "a single item (or a row)"],
      ["Slice", "`a[1:4]`, `M[:, :2]`", "a **view** (shares memory)"],
      ["Boolean mask", "`a[a > 0]`", "a **copy**, 1-D when the mask is full-shape"],
      ["Integer list / array", "`a[[3, 0]]`", "a **copy**, in the order you asked"],
    ]],
    ["warn", "Assigning *into* a slice or mask changes the original (`a[a > 8] = 8`). But chaining two selections does not: `a[mask][0] = 5` modifies a temporary copy and `a` stays as it was. Do the selection and the assignment in **one** bracket: `a[mask] = ...`."],
    ["tip", "A mask is also the cleanest way to count: `(y_pred == y_true).mean()` is the accuracy. True counts as 1, so `sum()` counts and `mean()` gives the fraction."],
  ],
},
{
  id: "vectorization", group: "NumPy", title: "Vectorization & ufuncs",
  sub: "Replace Python loops with whole-array operations: shorter code and often 10 to 100 times faster.",
  tags: ["numpy", "speed"],
  blocks: [
    ["p", "A Python `for` loop pays a small tax on every iteration (looking up types, calling functions). With a million numbers that tax dominates. **Vectorized** code hands the whole array to NumPy once and a compiled loop does the work. The rule for ML code: if you are looping over the elements of an array, there is probably a function for it."],
    ["code", R`
import numpy as np
import time

rng = np.random.default_rng(0)
x = rng.normal(size=200_000)

def sum_sq_loop(x):
    total = 0.0
    for v in x:                      # Python-level loop
        total += v * v
    return total

t0 = time.perf_counter(); r1 = sum_sq_loop(x); t_loop = time.perf_counter() - t0
t0 = time.perf_counter(); r2 = np.sum(x * x);  t_vec = time.perf_counter() - t0
print(np.isclose(r1, r2))                       # same answer
print("vectorized is faster:", t_vec < t_loop)  # timings vary, the ordering does not`, "True\nvectorized is faster: True"],
    ["p", "The vectorized line `np.sum(x * x)` does the same maths. The code is also one line and reads like the formula $\\sum x_i^2$. We only print that the vectorized version is faster, because the actual times change from machine to machine."],
    ["h", "ufuncs: functions that work element by element"],
    ["p", "A **ufunc** (universal function) applies an operation to every element, with [broadcasting](pyml.html#broadcasting) built in. The arithmetic operators `+ - * / ** %` are ufuncs, and so are `np.exp`, `np.log`, `np.sqrt`, `np.abs`, `np.maximum`, `np.sin`..."],
    ["table", ["Group", "Examples", "ML use"], [
      ["Arithmetic", "`+ - * / ** //`, `np.add`, `np.power`", "Linear models, losses"],
      ["Exp / log", "`np.exp`, `np.log`, `np.log1p`", "Softmax, cross-entropy, log-likelihood"],
      ["Comparison", "`> == !=`, `np.isclose`, `np.isnan`", "Masks, accuracy, missing data"],
      ["Element-wise max / min", "`np.maximum(a, 0)`, `np.clip`", "ReLU, clipping gradients"],
      ["Choosing", "`np.where(cond, a, b)`", "Piecewise functions, labels from scores"],
    ]],
    ["p", "Here are three activation functions written without a single loop. Each is a direct translation of its formula, and works on a number or an array of any shape:"],
    ["code", R`
import numpy as np
def sigmoid(z):  return 1 / (1 + np.exp(-z))
def relu(z):     return np.maximum(0, z)
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))    # subtract max for stability
    return e / e.sum(axis=-1, keepdims=True)

z = np.array([-2.0, 0.0, 3.0])
print(sigmoid(z).round(3))
print(relu(z))
print(softmax(z).round(3), softmax(z).sum())

Z = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])    # a batch: 2 rows of scores
print(softmax(Z).round(3))                          # one softmax per row`, "[0.119 0.5   0.953]\n[0. 0. 3.]\n[0.006 0.047 0.946] 1.0\n[[0.09  0.245 0.665]\n [0.333 0.333 0.333]]"],
    ["note", "Why subtract the maximum in softmax? `np.exp(1000)` overflows to infinity. Subtracting the largest score changes nothing mathematically (the common factor cancels) but keeps every exponent at or below 0. See [softmax in the practice set](practice.html#p/softmax)."],
    ["h", "Vectorize the whole algorithm, not just one line"],
    ["p", "Loops over *examples* are the ones to remove. Here is mean squared error and a linear model's predictions for all examples at once:"],
    ["code", R`
import numpy as np
rng = np.random.default_rng(1)
X = rng.normal(size=(5, 3))            # 5 examples, 3 features
w = np.array([2.0, -1.0, 0.5])
b = 0.1

# loop version: one example at a time
pred_loop = [sum(X[i, j] * w[j] for j in range(3)) + b for i in range(5)]
# vectorized version: one matrix-vector product
pred = X @ w + b
print(np.allclose(pred_loop, pred))

y = pred + 0.5
mse = np.mean((pred - y) ** 2)
print(round(float(mse), 3))            # every error is -0.5, so mean square = 0.25`, "True\n0.25"],
    ["h", "Gotchas"],
    ["list", [
      "**`np.vectorize` is not vectorization.** It wraps a Python function in a loop for convenience; it is no faster than a `for`. Rewrite the maths with ufuncs instead.",
      "**In-place operators need matching dtypes.** `a += 1.5` on an integer array raises an error, because the result cannot be stored as integers. Use `a = a + 1.5` or make `a` a float array first.",
      "**Intermediate arrays cost memory.** `np.exp(x) * 2 + 1` creates temporary arrays. For giant data use `out=` or chunks. For ordinary sizes, ignore this.",
      "**Some things stay loops.** Anything where step *t* depends on step *t-1* (a recurrent update, a running optimizer) cannot be vectorized over steps. Vectorize over the examples *inside* each step instead.",
    ]],
    ["code", R`
import numpy as np
a = np.arange(3)                      # int64
try:
    a += 1.5                          # in-place float add on ints
except TypeError as e:
    print(type(e).__name__)
a = a + 1.5                           # new float array: fine
print(a)`, "UFuncTypeError\n[1.5 2.5 3.5]"],
    ["tip", "Practise: [matmul loops vs vectorized](practice.html#p/matmul-loops-vs-vectorized) and the [vectorized moving average](practice.html#p/moving-average-vectorized)."],
  ],
},
{
  id: "broadcasting", group: "NumPy", title: "Broadcasting",
  sub: "NumPy's rule for combining arrays of different shapes by stretching size-1 dimensions, with no copying.",
  tags: ["numpy"],
  blocks: [
    ["p", "What should `[[1, 2, 3], [4, 5, 6]] + [10, 20, 30]` mean? Mathematically the shapes differ, but you obviously want to add the small row to **each row** of the big matrix. NumPy does exactly that, and calls it **broadcasting**. It is how you centre every column by subtracting its mean, scale features, add a bias to every example, and build tables of pairwise values, all without loops."],
    ["h", "The rule, in three steps"],
    ["steps", [
      "**Line up the shapes by their right-hand ends.** If one shape is shorter, pad it with 1s **on the left**.",
      "**Compare each pair of sizes.** They are compatible if they are **equal**, or **one of them is 1**.",
      "**The result size** in each position is the bigger of the two. A size-1 dimension is stretched (conceptually repeated) to match. If any pair is incompatible, NumPy raises a ValueError.",
    ]],
    ["p", "Type any two shapes below, separated by `|`, and step through. Try `3,4 | 4` (works), `3,1 | 1,4` (works: an outer product table), `3,4 | 3` (fails: 4 vs 3), and `5,3,4 | 3,1`."],
    ["viz", "pyml_broadcast"],
    ["table", ["Shape A", "Shape B", "Result", "Why"], [
      ["`(3, 4)`", "`(4,)`", "`(3, 4)`", "`(4,)` becomes `(1, 4)`, stretched over the 3 rows"],
      ["`(3, 4)`", "`(3, 1)`", "`(3, 4)`", "the column is stretched across the 4 columns"],
      ["`(3, 1)`", "`(1, 4)`", "`(3, 4)`", "both stretch: every row-value pairs with every column-value"],
      ["`(3, 4)`", "`(3,)`", "error", "right ends: 4 vs 3, neither is 1"],
      ["`(5, 1, 4)`", "`(3, 1)`", "`(5, 3, 4)`", "pad to `(1, 3, 1)`, then stretch every size-1"],
      ["`(3,)`", "`()` (a scalar)", "`(3,)`", "a scalar broadcasts against anything"],
    ]],
    ["h", "The everyday uses"],
    ["code", R`
import numpy as np
X = np.array([[1.0, 200.0], [2.0, 400.0], [3.0, 600.0]])   # (3, 2)
mu = X.mean(axis=0)                  # (2,): one mean per column
sd = X.std(axis=0)                   # (2,)
print(mu, sd.round(3))
Z = (X - mu) / sd                    # (3,2) - (2,) -> (3,2): standardize each column
print(Z.round(3))
print(Z.mean(axis=0).round(6), Z.std(axis=0))

b = np.array([0.5, -0.5])
print((X @ np.eye(2) + b).shape)     # add a bias row to every example`, "[  2. 400.] [  0.816 163.299]\n[[-1.225 -1.225]\n [ 0.     0.   ]\n [ 1.225  1.225]]\n[0. 0.] [1. 1.]\n(3, 2)"],
    ["p", "To combine a row with a column, you need a **column shape** `(n, 1)`. The tool is `[:, None]` (or `np.newaxis`, or `reshape(-1, 1)`), which inserts a size-1 axis:"],
    ["code", R`
import numpy as np
r = np.array([1, 2, 3])               # (3,)
c = np.array([10, 20, 30, 40])        # (4,)
table = c[:, None] * r                # (4,1) * (3,) -> (4,3)
print(table)

P = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
rowsum = P.sum(axis=1)                # (2,)
try:
    print(P / rowsum)                 # (2,3) / (2,) -> right ends 3 vs 2 : error
except ValueError as e:
    print("ValueError:", e)
print((P / rowsum[:, None]).round(3))             # fix with an added axis
print((P / P.sum(axis=1, keepdims=True)).round(3))  # or keepdims=True, tidier`, "[[ 10  20  30]\n [ 20  40  60]\n [ 30  60  90]\n [ 40  80 120]]\nValueError: operands could not be broadcast together with shapes (2,3) (2,) \n[[0.2 0.3 0.5]\n [0.1 0.1 0.8]]\n[[0.2 0.3 0.5]\n [0.1 0.1 0.8]]"],
    ["h", "Pairwise differences: the broadcasting showpiece"],
    ["p", "Given $n$ points $A$ and $m$ points $B$ (each row a point in $d$ dimensions), the difference between every pair is a 3-D array. Insert an axis on each side so the shapes become `(n, 1, d)` and `(1, m, d)`:"],
    ["code", R`
import numpy as np
A = np.array([[0.0, 0.0], [3.0, 4.0]])             # 2 points
B = np.array([[0.0, 1.0], [6.0, 8.0], [3.0, 0.0]])  # 3 points
diff = A[:, None, :] - B[None, :, :]               # (2,1,2) - (1,3,2) -> (2,3,2)
print(diff.shape)
D = np.sqrt((diff ** 2).sum(axis=-1))              # Euclidean distance for every pair
print(D.round(3))`, "(2, 3, 2)\n[[ 1.    10.     3.   ]\n [ 4.243  5.     4.   ]]"],
    ["warn", "Broadcasting makes the *intermediate* array real: `diff` here holds $n \\cdot m \\cdot d$ numbers. For 10,000 points against 10,000 points in 100 dimensions that is 10^10 floats, tens of gigabytes. Use the algebra trick $\\|a-b\\|^2 = \\|a\\|^2 - 2a\\cdot b + \\|b\\|^2$ with a matrix multiply, or `scipy.spatial.distance.cdist`. Try the exercise: [pairwise distances](practice.html#p/pairwise-distance-broadcast)."],
    ["danger", "Broadcasting never complains when the shapes happen to be compatible, even when you did not mean it. `(n, 1) - (n,)` quietly gives `(n, n)`. Whenever a result has a surprising shape, or a loss is suspiciously wrong, print the shapes of both operands. Use `keepdims=True` and `[:, None]` on purpose, and `.ravel()` when you want to remove a stray axis."],
    ["note", "This is the same idea as adding a scalar to a [matrix](math.html#matrices) or scaling a [vector](math.html#vectors), extended to arrays of any dimension. Next: how reductions like `sum` and `mean` choose which axis to squash, in [Axes & Reductions](pyml.html#axes)."],
  ],
},
{
  id: "axes", group: "NumPy", title: "Axes & Reductions",
  sub: "sum, mean, max and friends squash one dimension of an array. The axis argument says which one.",
  tags: ["numpy"],
  blocks: [
    ["p", "A **reduction** turns many numbers into fewer: the sum of a column, the largest score in each row, the mean of everything. On a 2-D array there are three useful questions: reduce over all numbers, reduce each column, or reduce each row. The `axis` argument answers it, and the single rule to remember is:"],
    ["tip", "The axis you name is the one that **disappears** from the shape. `(3, 4)` reduced over `axis=0` becomes `(4,)`; over `axis=1` it becomes `(3,)`.", "One rule"],
    ["p", "So for a matrix of shape `(rows, columns)`: **`axis=0` collapses the rows** (you walk down each column, result has one number per column) and **`axis=1` collapses the columns** (you walk across each row, one number per row). Choose the axis and the operation, and watch which cells feed which result:"],
    ["viz", "pyml_axis"],
    ["code", R`
import numpy as np
X = np.array([[2, 7, 1, 5],
              [4, 3, 8, 6],
              [9, 0, 2, 3]])
print(X.shape)
print(X.sum())                        # everything -> a single number
print(X.sum(axis=0), X.sum(axis=0).shape)   # per column -> shape (4,)
print(X.sum(axis=1), X.sum(axis=1).shape)   # per row    -> shape (3,)
print(X.mean(axis=0).round(2))
print(X.max(axis=1), X.argmax(axis=1))      # biggest value in each row, and where it is
print(X.sum(axis=1, keepdims=True).shape)   # keep the collapsed axis with size 1`, "(3, 4)\n50\n[15 10 11 14] (4,)\n[15 21 14] (3,)\n[5.   3.33 3.67 4.67]\n[7 8 9] [1 2 0]\n(3, 1)"],
    ["h", "keepdims: staying broadcast-ready"],
    ["p", "After `X.sum(axis=1)` the shape is `(3,)`, which can no longer be broadcast against `X` of shape `(3, 4)` (right ends: 4 vs 3). With `keepdims=True` the shape is `(3, 1)`, which divides the rows nicely. This is how you normalise each row:"],
    ["code", R`
import numpy as np
counts = np.array([[1, 3, 0, 4], [2, 2, 2, 2]])
probs = counts / counts.sum(axis=1, keepdims=True)    # each row now sums to 1
print(probs.round(3))
print(probs.sum(axis=1))`, "[[0.125 0.375 0.    0.5  ]\n [0.25  0.25  0.25  0.25 ]]\n[1. 1.]"],
    ["h", "The reductions you will use"],
    ["table", ["Function", "Result", "Typical ML use"], [
      ["`sum`, `mean`", "total / average", "Loss over a batch, feature means"],
      ["`std`, `var`", "spread", "Standardizing features"],
      ["`min`, `max`", "extremes", "Min-max scaling, range checks"],
      ["`argmax`, `argmin`", "**position** of the extreme", "Predicted class = `argmax` of scores"],
      ["`any`, `all`", "is any / is every one True?", "Checks like `np.isnan(X).any()`"],
      ["`cumsum`", "running total (**same shape**, not a reduction)", "Cumulative explained variance"],
    ]],
    ["h", "Three or more dimensions"],
    ["p", "The rule never changes: the axis you name disappears. A batch of 2 sequences, each with 3 time steps of 4 features, has shape `(2, 3, 4)`. Averaging over time (axis 1) leaves one vector per sequence:"],
    ["code", R`
import numpy as np
B = np.arange(24, dtype=float).reshape(2, 3, 4)   # (batch, time, features)
print(B.mean(axis=1).shape)              # average over time      -> (2, 4)
print(B.mean(axis=(0, 1)).shape)         # over batch and time    -> (4,)
print(B.sum(axis=-1).shape)              # last axis (features)   -> (2, 3)

logits = np.array([[2.0, 0.1, 0.3], [0.2, 0.2, 3.0], [1.0, 0.9, 0.0]])
y = np.array([0, 2, 1])
pred = logits.argmax(axis=1)             # predicted class per example
print(pred, (pred == y).mean().round(3)) # accuracy`, "(2, 4)\n(4,)\n(2, 3)\n[0 2 0] 0.667"],
    ["warn", "**`np.std` and pandas `.std()` disagree by default.** NumPy divides by $n$ (`ddof=0`, the population formula); pandas divides by $n-1$ (`ddof=1`, the sample formula). scikit-learn's `StandardScaler` uses `ddof=0`. Match them explicitly with `ddof=` when you compare results."],
    ["code", R`
import numpy as np
import pandas as pd

v = np.array([1.0, 2.0, 3.0, 4.0])
print(np.std(v), np.std(v, ddof=1), pd.Series(v).std())`, "1.118033988749895 1.2909944487358056 1.2909944487358056"],
    ["danger", "Mixing up the axes is the most common NumPy bug. If a result has the wrong length, or normalised rows do not sum to 1, check which axis you reduced. For `np.concatenate` and `np.stack` the word axis means something different (which dimension grows or gets created), so read their docs once."],
    ["note", "Practise with [per-column min-max and z-score](practice.html#p/minmax-zscore-axis). Standardizing is explained in [Feature Engineering & Scaling](ml.html#feature-engineering); the pandas equivalents are in [groupby](pyml.html#pandas-groupby)."],
  ],
},
{
  id: "numpy-linalg", group: "NumPy", title: "Matrix Operations & @",
  sub: "Matrix multiply, transpose, solving linear systems and decompositions: the linear algebra of the Math page, as one-liners.",
  tags: ["numpy", "linear algebra"],
  blocks: [
    ["p", "The `@` operator is **matrix multiplication**, the operation behind every linear model and neural-network layer (see [Matrices](math.html#matrices)). The `*` operator is something else entirely: it multiplies element by element. Confusing the two gives wrong numbers with no error message, so it is worth seeing them side by side."],
    ["code", R`
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])
print(A * B)          # element-wise: [[1*0, 2*1], [3*1, 4*0]]
print(A @ B)          # matrix product: rows of A dot columns of B
print(A.T)            # transpose: rows become columns
print(np.matmul(A, B).tolist() == (A @ B).tolist())   # @ is np.matmul

x = np.array([1, 2])
print(A @ x)          # matrix times vector -> shape (2,)
print(x @ x)          # vector dot vector -> a scalar (5)
print(np.outer(x, x)) # outer product -> a (2, 2) matrix`, "[[0 2]\n [3 0]]\n[[2 1]\n [4 3]]\n[[1 3]\n [2 4]]\nTrue\n[ 5 11]\n5\n[[1 2]\n [2 4]]"],
    ["h", "Shape arithmetic for @"],
    ["p", "The inner sizes must match and disappear; the outer sizes remain: `(n, d) @ (d, k)` gives `(n, k)`. A 1-D array is treated as a vector, and the dimension it adds is removed from the result again."],
    ["table", ["Left", "Right", "Result", "Example"], [
      ["`(n, d)`", "`(d,)`", "`(n,)`", "predictions `X @ w`"],
      ["`(n, d)`", "`(d, k)`", "`(n, k)`", "a layer: inputs @ weights"],
      ["`(d,)`", "`(d,)`", "scalar", "dot product, similarity"],
      ["`(B, n, d)`", "`(d, k)`", "`(B, n, k)`", "a batch of matrices, same weights"],
      ["`(B, n, d)`", "`(B, d, k)`", "`(B, n, k)`", "batched multiply (attention scores)"],
    ]],
    ["code", R`
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3))          # 4 examples, 3 features
W = rng.normal(size=(3, 2))          # a layer with 2 outputs
print((X @ W).shape)
Q = rng.normal(size=(2, 5, 3))       # batch of 2 sequences, 5 tokens, 3 dims
K = rng.normal(size=(2, 5, 3))
scores = Q @ K.transpose(0, 2, 1)    # (2,5,3) @ (2,3,5) -> (2,5,5)
print(scores.shape)
try:
    X @ X                            # (4,3) @ (4,3): inner sizes 3 vs 4
except ValueError:
    print("shapes (4,3) and (4,3) are not aligned")`, "(4, 2)\n(2, 5, 5)\nshapes (4,3) and (4,3) are not aligned"],
    ["h", "Solving, not inverting"],
    ["p", "To solve $Ax = b$ you could compute $A^{-1}b$, but explicit inverses are slower and less accurate. Use `np.linalg.solve`. For least squares (more equations than unknowns, the usual ML case) use `np.linalg.lstsq`. Here is [linear regression](ml.html#linear-regression) in a few lines: add a column of ones for the intercept, then solve the **normal equation** $X^\\top X\\,w = X^\\top y$."],
    ["code", R`
import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(0, 10, size=50)
y = 2.0 + 3.0 * x + rng.normal(scale=0.5, size=50)    # true line: intercept 2, slope 3
X = np.column_stack([np.ones_like(x), x])             # (50, 2): a ones column + the feature
w = np.linalg.solve(X.T @ X, X.T @ y)                 # normal equation
print(w.round(3))
w2, *_ = np.linalg.lstsq(X, y, rcond=None)            # same thing, more robust
print(np.allclose(w, w2))
print(np.round(np.mean((X @ w - y) ** 2), 3))         # mean squared error`, "[1.879 3.026]\nTrue\n0.248"],
    ["h", "Other tools in np.linalg"],
    ["code", R`
import numpy as np
A = np.array([[2.0, 1.0], [1.0, 3.0]])
print(round(float(np.linalg.det(A)), 3))                             # determinant
print(np.linalg.inv(A).round(3))                       # inverse (use sparingly)
print(np.linalg.norm([3.0, 4.0]))                      # length of a vector (L2 norm)
vals, vecs = np.linalg.eigh(A)                         # eigen of a symmetric matrix
print(vals.round(3))
U, S, Vt = np.linalg.svd(A)                            # singular value decomposition
print(S.round(3), np.allclose(U @ np.diag(S) @ Vt, A))
print(np.allclose(A @ np.linalg.inv(A), np.eye(2)))    # compare floats with allclose`, "5.0\n[[ 0.6 -0.2]\n [-0.2  0.4]]\n5.0\n[1.382 3.618]\n[3.618 1.382] True\nTrue"],
    ["table", ["Function", "What it does", "Learn the maths"], [
      ["`np.linalg.norm`", "length of a vector / size of a matrix", "[Norms](math.html#dot-product)"],
      ["`np.linalg.inv`, `solve`", "inverse / solve $Ax=b$", "[Inverse](math.html#matrix-ops)"],
      ["`np.linalg.eigh`, `eig`", "eigenvalues and eigenvectors", "[Eigen](math.html#eigen)"],
      ["`np.linalg.svd`", "singular value decomposition (PCA, low-rank)", "[SVD](math.html#svd)"],
      ["`np.einsum`", "any sum-of-products written with index letters", "`'ij,jk->ik'` is `A @ B`"],
    ]],
    ["danger", "Three classic slips. (1) `*` instead of `@`: shapes may even broadcast, so no error. (2) `.T` on a 1-D array does nothing (there is no second axis to swap); reshape to `(n, 1)` first. (3) Comparing floats with `==`: `A @ inv(A) == np.eye(2)` is usually False because of rounding; use `np.allclose`."],
    ["tip", "Practise this: [matmul with loops vs vectorized](practice.html#p/matmul-loops-vs-vectorized) and [the normal equation](practice.html#p/linreg-normal-equation)."],
  ],
},
{
  id: "numpy-random", group: "NumPy", title: "Random Numbers & Seeds",
  sub: "Shuffle, sample and add noise, and make every run give the same answer.",
  tags: ["numpy", "reproducibility"],
  blocks: [
    ["p", "ML is full of randomness: initial weights, shuffling the data, picking a validation set, bootstrap samples, dropout. Computers produce **pseudo-random** numbers: a deterministic formula started from a number called the **seed**. Same seed, same sequence. That is what lets you reproduce an experiment, or debug it."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)        # a random number generator with seed 0
print(rng.random(3).round(3))         # uniform in [0, 1)
print(rng.normal(loc=5, scale=2, size=3).round(3))   # Gaussian: mean 5, std 2
print(rng.integers(0, 10, size=5))    # integers 0..9 (the top is excluded)

rng2 = np.random.default_rng(0)       # same seed -> the same numbers again
print(rng2.random(3).round(3))        # identical to the first line`, "[0.637 0.27  0.041]\n[5.21  3.929 5.723]\n[5 6 9 7 6]\n[0.637 0.27  0.041]"],
    ["p", "Use the modern `np.random.default_rng(seed)` generator and call methods on it. You will still see the older global style `np.random.seed(0); np.random.rand(3)` in many tutorials; it works, but one hidden global generator is harder to control than an explicit object you pass around."],
    ["table", ["Method", "Gives you", "Use"], [
      ["`rng.random(size)`", "uniform [0, 1)", "Random masks (dropout), probabilities"],
      ["`rng.normal(mu, sd, size)`", "Gaussian numbers", "Weight initialisation, synthetic noise"],
      ["`rng.integers(lo, hi, size)`", "random whole numbers", "Random indices, labels"],
      ["`rng.permutation(n)`", "a shuffled `0..n-1`", "Shuffling, splits"],
      ["`rng.choice(a, size, replace, p)`", "samples from `a`, optionally weighted", "Bootstrap, sampling classes"],
      ["`rng.shuffle(arr)`", "shuffles `arr` **in place**", "Mixing a dataset"],
    ]],
    ["h", "Shuffling and splitting by hand"],
    ["p", "A train/test split is just a shuffled index. Shuffle the **indices**, then use them on both `X` and `y` so the rows stay matched. (In practice use `train_test_split`, shown in [the scikit-learn pipelines section](pyml.html#sklearn-pipelines).)"],
    ["code", R`
import numpy as np
rng = np.random.default_rng(42)
X = np.arange(20).reshape(10, 2)
y = np.arange(10) % 2
perm = rng.permutation(len(X))               # shuffled row numbers
cut = int(0.8 * len(X))
train_idx, test_idx = perm[:cut], perm[cut:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(train_idx, test_idx)
print(X_train.shape, X_test.shape, np.all(X_test[:, 0] // 2 == test_idx))   # rows stayed aligned`, "[5 6 0 7 3 2 4 9] [1 8]\n(8, 2) (2, 2) True"],
    ["h", "Sampling with and without replacement"],
    ["code", R`
import numpy as np
rng = np.random.default_rng(1)
data = np.array([10, 20, 30, 40, 50])
print(rng.choice(data, size=3, replace=False))     # no repeats: like dealing cards
boot = rng.choice(data, size=5, replace=True)      # a bootstrap sample: repeats allowed
print(boot)

picks = rng.choice(3, size=1000, p=[0.7, 0.2, 0.1])   # weighted sampling
print(np.bincount(picks) / 1000)                       # close to 0.7, 0.2, 0.1

noise = rng.normal(scale=0.1, size=10_000)
print(round(float(noise.mean()), 3), round(float(noise.std()), 3))`, "[30 20 40]\n[10 50 50 20 20]\n[0.687 0.206 0.107]\n-0.001 0.1"],
    ["p", "A **bootstrap** sample draws $n$ items from $n$ *with* replacement, so some appear twice and about a third are left out. [Random forests](ml.html#random-forests) train each tree on one."],
    ["h", "Reproducibility checklist"],
    ["list", [
      "Create **one** generator near the top, `rng = np.random.default_rng(0)`, and pass it to functions that need randomness.",
      "In scikit-learn, give every random thing a **`random_state`**: `train_test_split(..., random_state=0)`, `RandomForestClassifier(random_state=0)`, `KFold(shuffle=True, random_state=0)`. Without it, each run can differ.",
      "Deep learning has its own seeds (`torch.manual_seed(0)`), and GPU kernels can still be non-deterministic. Fixing seeds gets you close, not always bit-identical.",
      "Report results over **several seeds** (mean and spread). A single lucky seed can fake a 1% improvement.",
    ]],
    ["danger", "Re-creating the generator with the same seed *inside a loop* gives the same \"random\" numbers every iteration (every fold, every epoch). Create it once, outside. And never choose the seed that makes your test score look best: that is [data leakage](ml.html#splits) of your own judgement."],
    ["note", "The random numbers here follow the [distributions](math.html#distributions) from the Math page: `normal`, `uniform`, `binomial`... are all methods on `rng`."],
  ],
},
{
  id: "pandas-basics", group: "pandas", title: "Series & DataFrame",
  sub: "A DataFrame is a table with labelled columns; a Series is one column. Get data in, look at it, and hand it to NumPy.",
  tags: ["pandas"],
  blocks: [
    ["p", "NumPy arrays have one dtype and no names. Real datasets have columns of different types (numbers, text, dates) and meaningful names. **pandas** adds exactly that. A **DataFrame** is a table: each column is a **Series** (a labelled 1-D array), and rows are labelled by an **index** (0, 1, 2... by default)."],
    ["code", R`
import pandas as pd

df = pd.DataFrame({
    "city": ["Pune", "Delhi", "Pune", "Goa"],
    "temp": [31.5, 38.0, 29.0, 33.5],
    "rainy": [True, False, True, False],
})
print(df)
print(df.shape)
print(df.dtypes)
print(df["temp"].mean(), df["city"].nunique())
print(df.describe().round(2))`, "    city  temp  rainy\n0   Pune  31.5   True\n1  Delhi  38.0  False\n2   Pune  29.0   True\n3    Goa  33.5  False\n(4, 3)\ncity      object\ntemp     float64\nrainy       bool\ndtype: object\n33.0 3\n        temp\ncount   4.00\nmean   33.00\nstd     3.81\nmin    29.00\n25%    30.88\n50%    32.50\n75%    34.62\nmax    38.00"],
    ["p", "The first three commands to run on any new dataset: `df.head()` (first rows), `df.info()` (column types and missing counts) and `df.describe()` (summary statistics of the numeric columns). Real data usually comes from a file; `read_csv` works on files and on any text:"],
    ["code", R`
import io
import pandas as pd

csv = """name,age,score
Asha,23,81.5
Ben,31,
Chen,27,92.0
"""
df = pd.read_csv(io.StringIO(csv))          # normally: pd.read_csv("data.csv")
print(df.dtypes.astype(str).to_dict())
print(df.isna().sum().to_dict())
print(df.head(2))
print(df.columns.tolist())`, "{'name': 'object', 'age': 'int64', 'score': 'float64'}\n{'name': 0, 'age': 0, 'score': 1}\n   name  age  score\n0  Asha   23   81.5\n1   Ben   31    NaN\n['name', 'age', 'score']"],
    ["h", "Columns are vectorized, like NumPy"],
    ["code", R`
import pandas as pd

df = pd.DataFrame({"height_cm": [150, 160, 170], "weight_kg": [50, 64, 85]})
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2   # new column, no loop
df["tall"] = df["height_cm"] > 155                            # boolean column
print(df.round(1))
print(df.sort_values("bmi", ascending=False).index.tolist())
print(df["tall"].value_counts().to_dict())                    # counts per value`, "   height_cm  weight_kg   bmi   tall\n0        150         50  22.2  False\n1        160         64  25.0   True\n2        170         85  29.4   True\n[2, 1, 0]\n{True: 2, False: 1}"],
    ["h", "Series are aligned by label, not position"],
    ["p", "This is the pandas feature that surprises NumPy users. When you combine two Series, pandas first matches their **index labels**; anything without a partner becomes NaN (missing):"],
    ["code", R`
import pandas as pd

a = pd.Series([1, 2, 3], index=["x", "y", "z"])
b = pd.Series([10, 20, 30], index=["y", "z", "w"])
print(a + b)                     # only y and z overlap; x and w become NaN
print(a.to_numpy() + b.to_numpy())   # NumPy ignores labels: adds by position`, "w     NaN\nx     NaN\ny    12.0\nz    23.0\ndtype: float64\n[11 22 33]"],
    ["h", "From pandas to NumPy and back"],
    ["code", R`
import pandas as pd

df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "label": [0, 1, 0]})
X = df[["a", "b"]].to_numpy()        # a (3, 2) float array; double brackets = a DataFrame
y = df["label"].to_numpy()           # a (3,) array
print(X.shape, y.shape, X.dtype)
back = pd.DataFrame(X, columns=["a", "b"])   # NumPy -> DataFrame, naming the columns
print(back.sum().to_dict())`, "(3, 2) (3,) float64\n{'a': 6.0, 'b': 15.0}"],
    ["table", ["You write", "You get"], [
      ["`df[\"a\"]`", "a **Series** (one column)"],
      ["`df[[\"a\", \"b\"]]`", "a **DataFrame** (list of columns)"],
      ["`df.to_numpy()`", "a plain 2-D array (loses the names; the dtype becomes the most general one)"],
      ["`df.dtypes`, `df.info()`", "the type of every column"],
    ]],
    ["tip", "scikit-learn accepts DataFrames directly. But NumPy is faster for heavy maths, so a typical flow is: clean with pandas, then `.to_numpy()` for the model. Next: [selecting and filtering rows](pyml.html#pandas-select)."],
  ],
},
{
  id: "pandas-select", group: "pandas", title: "Selecting & Filtering",
  sub: "Pick columns, rows by label or position, and rows that pass a test, and change values without hitting the SettingWithCopy trap.",
  tags: ["pandas"],
  blocks: [
    ["p", "Selecting in pandas is the NumPy ideas from [Indexing & Masks](pyml.html#numpy-indexing) with labels added. Three tools cover almost everything: **`df[...]`** for columns and boolean row filters, **`.loc`** for selecting by **label**, and **`.iloc`** for selecting by **position**."],
    ["code", R`
import pandas as pd

df = pd.DataFrame({
    "name": ["Asha", "Ben", "Chen", "Dia", "Eli"],
    "age": [23, 31, 27, 45, 19],
    "city": ["Pune", "Delhi", "Pune", "Goa", "Delhi"],
    "score": [81.5, 70.0, 92.0, 64.5, 88.0],
}, index=["a", "b", "c", "d", "e"])

print(df["age"].tolist())                  # one column -> Series
print(df[["name", "score"]].head(2))       # several columns -> DataFrame
print(df.loc["b", "score"])                # row label "b", column "score"
print(df.loc["b":"d", ["name", "age"]])    # label slices INCLUDE the end
print(df.iloc[1:3, 0:2])                   # position slices EXCLUDE the end
print(df.iloc[-1].to_dict())               # last row`, "[23, 31, 27, 45, 19]\n   name  score\na  Asha   81.5\nb   Ben   70.0\n70.0\n   name  age\nb   Ben   31\nc  Chen   27\nd   Dia   45\n   name  age\nb   Ben   31\nc  Chen   27\n{'name': 'Eli', 'age': 19, 'city': 'Delhi', 'score': 88.0}"],
    ["table", ["", "`.loc`", "`.iloc`"], [
      ["Selects by", "labels (index values, column names)", "integer positions"],
      ["Slice end", "**included**", "excluded (like Python)"],
      ["Takes a boolean mask", "yes", "only as a plain array (`mask.to_numpy()`)"],
      ["Use when", "the index means something", "you need \"the first 100 rows\""],
    ]],
    ["h", "Filtering rows"],
    ["code", R`
import pandas as pd

df = pd.DataFrame({
    "name": ["Asha", "Ben", "Chen", "Dia", "Eli"],
    "age": [23, 31, 27, 45, 19],
    "city": ["Pune", "Delhi", "Pune", "Goa", "Delhi"],
    "score": [81.5, 70.0, 92.0, 64.5, 88.0],
})
print(df[df["score"] > 80]["name"].tolist())                     # one condition
print(df[(df["city"] == "Delhi") & (df["age"] < 30)]["name"].tolist())   # & | ~ with parentheses
print(df[df["city"].isin(["Pune", "Goa"])]["name"].tolist())     # membership
print(df[df["age"].between(20, 30)]["name"].tolist())            # inclusive range
print(df[df["name"].str.startswith("A")]["name"].tolist())       # string methods via .str
print(df.query("score > 80 and city == 'Pune'")["name"].tolist())  # the same as a string
print(df.nlargest(2, "score")["name"].tolist())                  # top-2 by a column`, "['Asha', 'Chen', 'Eli']\n['Eli']\n['Asha', 'Chen', 'Dia']\n['Asha', 'Chen']\n['Asha']\n['Asha', 'Chen']\n['Chen', 'Eli']"],
    ["p", "The pattern is exactly NumPy's: a comparison makes a boolean Series, and `df[mask]` keeps rows where it is True. Combine tests with `&`, `|`, `~` and **parentheses** (never `and`/`or`)."],
    ["h", "Changing values: the SettingWithCopy trap"],
    ["p", "Suppose you want to set `score` to 0 for everyone under 25. The intuitive way, `df[df.age < 25][\"score\"] = 0`, does **nothing** to `df`. It makes two separate selections in a row (*chained indexing*). The first, `df[df.age < 25]`, may hand back a temporary copy, and you then modify that copy. pandas warns with `SettingWithCopyWarning`. The cure is a single `.loc` with the row filter and the column together:"],
    ["code", R`
import warnings
import pandas as pd

warnings.simplefilter("ignore")            # silence the SettingWithCopyWarning for this demo
df = pd.DataFrame({"age": [23, 31, 19], "score": [81.5, 70.0, 88.0]})

df[df["age"] < 25]["score"] = 0            # chained: modifies a temporary copy
print(df["score"].tolist())                # unchanged!

df.loc[df["age"] < 25, "score"] = 0        # one .loc call: modifies df itself
print(df["score"].tolist())

young = df[df["age"] < 25].copy()          # want an independent subset? say .copy()
young["score"] = 99
print(df["score"].tolist(), young["score"].tolist())`, "[81.5, 70.0, 88.0]\n[0.0, 70.0, 0.0]\n[0.0, 70.0, 0.0] [99, 99]"],
    ["danger", "**Chained assignment** (`df[mask][\"col\"] = v`) is the classic pandas bug: it silently fails or warns. Always write `df.loc[mask, \"col\"] = v`. And when you take a subset you plan to edit, call `.copy()` so pandas knows it is yours."],
    ["h", "Handy extras"],
    ["code", R`
import pandas as pd

df = pd.DataFrame({"g": ["u", "v", "u"], "x": [3, 1, 2], "y": [9, 8, 7]})
print(df.drop(columns="y").columns.tolist())              # drop a column (returns a new df)
print(df.rename(columns={"x": "feature"}).columns.tolist())
print(df.sort_values("x").reset_index(drop=True))         # sort, then renumber the rows
print(df["g"].map({"u": 0, "v": 1}).tolist())             # recode values with a dict
print(df[["x", "y"]].apply(lambda col: col - col.mean()).round(1).values.tolist())  # per column
print(df.assign(z=df["x"] * 2)["z"].tolist())             # add a column without editing df`, "['g', 'x']\n['g', 'feature', 'y']\n   g  x  y\n0  v  1  8\n1  u  2  7\n2  u  3  9\n[0, 1, 0]\n[[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]\n[6, 2, 4]"],
    ["tip", "Most pandas methods return a **new** object and leave the original alone (`drop`, `sort_values`, `rename`...). You must assign the result: `df = df.drop(columns=\"y\")`. Calling `df.drop(columns=\"y\")` on its own line changes nothing."],
  ],
},
{
  id: "pandas-groupby", group: "pandas", title: "groupby, merge & pivot",
  sub: "Split a table by a key, summarise each part, combine the results; and glue two tables together on a shared column.",
  tags: ["pandas"],
  blocks: [
    ["p", "Most questions about a table are \"per something\": average spend per customer, accuracy per class, revenue per month. `groupby` answers them in three moves: **split** the rows by a key, **apply** a function to each group, **combine** the answers into a new table. Step through it and change the aggregation:"],
    ["viz", "pyml_groupby"],
    ["code", R`
import pandas as pd

df = pd.DataFrame({
    "city": ["Pune", "Delhi", "Pune", "Goa", "Delhi", "Pune"],
    "sales": [10, 20, 30, 5, 40, 20],
    "units": [1, 2, 3, 1, 4, 2],
})
print(df.groupby("city")["sales"].mean())               # one number per city
print(df.groupby("city").agg(total=("sales", "sum"),    # named aggregations
                             avg_units=("units", "mean"),
                             n=("sales", "size")))
print(df.groupby("city").size().to_dict())              # rows per group
print(df.groupby("city")["sales"].mean().reset_index()) # turn the key back into a column`, "city\nDelhi    30.0\nGoa       5.0\nPune     20.0\nName: sales, dtype: float64\n       total  avg_units  n\ncity                      \nDelhi     60        3.0  2\nGoa        5        1.0  1\nPune      60        2.0  3\n{'Delhi': 2, 'Goa': 1, 'Pune': 3}\n    city  sales\n0  Delhi   30.0\n1    Goa    5.0\n2   Pune   20.0"],
    ["p", "The result is indexed by the group key (sorted). Call `.reset_index()` when you want the key back as an ordinary column. Group by several keys with a list: `df.groupby([\"city\", \"year\"])`."],
    ["h", "transform: a group statistic for every row"],
    ["p", "`agg` returns **one row per group**. `transform` returns **one value per original row** (the group's statistic copied onto each member), so the result lines up with the table. This is how you centre a feature within its group, or fill missing values with the group mean:"],
    ["code", R`
import pandas as pd

df = pd.DataFrame({
    "city": ["Pune", "Delhi", "Pune", "Goa", "Delhi", "Pune"],
    "sales": [10, 20, 30, 5, 40, 20],
})
df["city_mean"] = df.groupby("city")["sales"].transform("mean")
df["vs_city"] = df["sales"] - df["city_mean"]
print(df)`, "    city  sales  city_mean  vs_city\n0   Pune     10       20.0    -10.0\n1  Delhi     20       30.0    -10.0\n2   Pune     30       20.0     10.0\n3    Goa      5        5.0      0.0\n4  Delhi     40       30.0     10.0\n5   Pune     20       20.0      0.0"],
    ["warn", "Group statistics computed on the **whole** dataset before splitting into train and test leak information from the test set. This matters for target encoding and group means: compute them on the training rows only, as covered in [Encoding Categorical Data](ml.html#encoding) and [Data Leakage](ml.html#splits)."],
    ["h", "merge: joining tables"],
    ["p", "A **merge** (a database *join*) lines up rows of two tables by a shared key column. Which rows survive depends on the join type:"],
    ["code", R`
import pandas as pd

users = pd.DataFrame({"uid": [1, 2, 3], "name": ["Asha", "Ben", "Chen"]})
orders = pd.DataFrame({"uid": [1, 1, 3, 4], "amount": [20, 35, 15, 99]})
print(users.merge(orders, on="uid", how="inner"))      # keys present in both
print(users.merge(orders, on="uid", how="left"))       # every user, orders where they exist
print(len(users.merge(orders, on="uid", how="outer"))) # every key from either side
total = orders.groupby("uid")["amount"].sum().rename("total")
print(users.merge(total, on="uid", how="left"))        # aggregate first, then join`, "   uid  name  amount\n0    1  Asha      20\n1    1  Asha      35\n2    3  Chen      15\n   uid  name  amount\n0    1  Asha    20.0\n1    1  Asha    35.0\n2    2   Ben     NaN\n3    3  Chen    15.0\n5\n   uid  name  total\n0    1  Asha   55.0\n1    2   Ben    NaN\n2    3  Chen   15.0"],
    ["table", ["`how=`", "Keeps", "Missing partner becomes"], [
      ["`inner`", "keys found in **both** tables", "(row dropped)"],
      ["`left`", "every row of the left table", "NaN on the right columns"],
      ["`right`", "every row of the right table", "NaN on the left columns"],
      ["`outer`", "all keys from either table", "NaN on both sides"],
    ]],
    ["danger", "Always check `len()` before and after a merge. If the key is **not unique** on both sides the rows multiply: user 1 above has two orders, so a merge gives two rows for them. A surprise jump in row count is the sign of a duplicated key. Also pass `how=` explicitly, so readers know what you intended."],
    ["h", "pivot_table, crosstab and concat"],
    ["code", R`
import pandas as pd

df = pd.DataFrame({
    "city": ["Pune", "Delhi", "Pune", "Goa", "Delhi", "Pune"],
    "kind": ["a", "a", "b", "b", "b", "a"],
    "sales": [10, 20, 30, 5, 40, 20],
})
print(df.pivot_table(index="city", columns="kind", values="sales", aggfunc="sum", fill_value=0))
print(pd.crosstab(df["city"], df["kind"]))               # counts of each combination
top = pd.concat([df.head(2), df.tail(1)], ignore_index=True)   # stack rows
print(top.shape)`, "kind    a   b\ncity         \nDelhi  20  40\nGoa     0   5\nPune   30  30\nkind   a  b\ncity       \nDelhi  1  1\nGoa    0  1\nPune   2  1\n(3, 3)"],
    ["tip", "`pivot_table` is `groupby` with one key down the side and another across the top. Quick reference: **stack rows** with `pd.concat([a, b])`, **add columns by key** with `merge`, **summarise per group** with `groupby`."],
  ],
},
{
  id: "pandas-missing", group: "pandas", title: "Missing Values & Cleaning",
  sub: "Find, understand and handle NaN and messy values before they reach a model.",
  tags: ["pandas", "data cleaning"],
  blocks: [
    ["p", "Real data has holes: a sensor failed, a user skipped a question, a join found no partner. pandas marks a missing number as **NaN** (\"not a number\"). Most scikit-learn models refuse NaN, and NumPy statistics return NaN if even one value is missing. So handling missing data comes before modelling."],
    ["code", R`
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "age": [25, np.nan, 31, 40, np.nan],
    "income": [50.0, 62.0, np.nan, 80.0, 58.0],
    "city": ["Pune", "Goa", None, "Pune", "Goa"],
})
print(df.isna().sum())                  # missing count per column
print(df.isna().mean().round(2).to_dict())   # missing fraction
print(df.dropna().shape)                # drop every row that has any NaN
print(df.dropna(subset=["income"]).shape)    # only look at one column
print(np.nan == np.nan, np.isnan(np.nan))    # NaN is never equal to anything
print(np.mean([1.0, np.nan]), np.nanmean([1.0, np.nan]), df["age"].mean())`, "age       2\nincome    1\ncity      1\ndtype: int64\n{'age': 0.4, 'income': 0.2, 'city': 0.2}\n(2, 3)\n(4, 3)\nFalse True\nnan 1.0 32.0"],
    ["p", "Two reminders from the output: you must test for NaN with `isna()` / `np.isnan`, never with `==`; and pandas statistics skip NaN by default (the `mean` of the age column ignores the two holes) while plain NumPy does not (`np.mean` gives nan; use `np.nanmean`)."],
    ["h", "Three ways to deal with it"],
    ["table", ["Strategy", "How", "Good when", "Watch out"], [
      ["**Drop rows**", "`dropna()`", "few rows affected, missing at random", "may throw away a lot, and bias the sample"],
      ["**Drop the column**", "`drop(columns=...)`", "mostly empty (say over 60% missing)", "may lose a useful signal"],
      ["**Impute** (fill in)", "`fillna(median)`, `SimpleImputer`", "most cases", "shrinks variance; compute the fill value on **training data only**"],
      ["**Flag + impute**", "add `x_missing = x.isna()`", "missing itself carries information", "adds a column per feature"],
    ]],
    ["code", R`
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "age": [25, np.nan, 31, 40, np.nan],
    "city": ["Pune", "Goa", None, "Pune", "Goa"],
})
df["age_missing"] = df["age"].isna().astype(int)          # keep a flag first
df["age"] = df["age"].fillna(df["age"].median())          # numeric: median is robust to outliers
df["city"] = df["city"].fillna(df["city"].mode()[0])      # category: most frequent value
print(df)
print(df.isna().sum().sum())                              # no holes left`, "    age  city  age_missing\n0  25.0  Pune            0\n1  31.0   Goa            1\n2  31.0   Goa            0\n3  40.0  Pune            0\n4  31.0   Goa            1\n0"],
    ["warn", "Computing the fill value on the **full** dataset before splitting is [data leakage](ml.html#splits): the test rows influenced the training data. Split first, fit the imputer on the training part, and apply it to the test part. `SimpleImputer` inside a pipeline does this for you ([Pipelines](pyml.html#sklearn-pipelines))."],
    ["h", "Cleaning messy values"],
    ["p", "Missing data hides in many disguises: the text `\"N/A\"`, the number -999, an empty string, a number typed with a comma. Convert them to real NaN, fix the types, and remove duplicates:"],
    ["code", R`
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "price": ["10", "12.5", "N/A", "-999", "12.5"],
    "name": [" Ada ", "bob", "Cy", "bob", "bob"],
})
df["price"] = pd.to_numeric(df["price"], errors="coerce")  # "N/A" becomes NaN
df["price"] = df["price"].replace(-999, np.nan)            # a sentinel value becomes NaN
df["name"] = df["name"].str.strip().str.lower()            # tidy text
print(df)
print(df.duplicated().sum(), len(df.drop_duplicates()))    # exact duplicate rows
print(df["price"].clip(upper=12).tolist())                 # cap extreme values`, "   price name\n0   10.0  ada\n1   12.5  bob\n2    NaN   cy\n3    NaN  bob\n4   12.5  bob\n1 4\n[10.0, 12.0, nan, nan, 12.0]"],
    ["danger", "Integer columns cannot hold NaN: one missing value turns the column into `float64` (ids like 3 become 3.0). If you need integers with holes use the nullable type `Int64`: `df[\"id\"].astype(\"Int64\")`. Also remember that sentinel values like -999 are *not* NaN to pandas: they silently pull your averages down until you convert them."],
    ["note", "Cleaning feeds directly into [Feature Engineering & Scaling](ml.html#feature-engineering) and [the ML workflow](ml.html#workflow). Next up: seeing the data with [matplotlib](pyml.html#matplotlib)."],
  ],
},
{
  id: "matplotlib", group: "Plotting", title: "Plotting with matplotlib",
  sub: "Look at your data and your model: histograms, scatter plots, learning curves, heatmaps and decision boundaries.",
  tags: ["matplotlib", "visualization"],
  blocks: [
    ["p", "You cannot debug what you cannot see. Before training, plot the distribution of each feature and the relationship between features and the target. After training, plot the loss curve and the errors. **matplotlib** is the standard tool; pandas and seaborn plots are built on top of it."],
    ["p", "A figure has a **Figure** (the canvas) holding one or more **Axes** (the individual plots with their own x/y axis). The tidy way to write it, called the object-oriented style, is to create both explicitly and draw on the axes:"],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, np.cos(x), "--", label="cos(x)")       # "--" = dashed line
ax.set_xlabel("x")
ax.set_ylabel("value")
ax.set_title("Two curves on one axes")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
print(len(x), round(float(np.sin(x).max()), 3))`, "100 1.0"],
    ["p", "The quick style `plt.plot(...)`, `plt.xlabel(...)` works on an invisible \"current axes\" and is fine for one-off plots. Use `plt.subplots` as soon as you want several panels. A figure always needs a label on each axis, a title or caption, and a legend if there is more than one line."],
    ["h", "The plots ML uses every day"],
    ["table", ["Question", "Plot", "Call"], [
      ["What does this feature look like?", "histogram", "`ax.hist(x, bins=20)`"],
      ["Do two features relate? Are classes separable?", "scatter coloured by class", "`ax.scatter(x1, x2, c=y)`"],
      ["Is training working?", "line plot of loss per epoch", "`ax.plot(losses)`"],
      ["Which classes get confused?", "heatmap of the confusion matrix", "`ax.imshow(cm)`"],
      ["Compare groups", "bar chart", "`ax.bar(names, values)`"],
    ]],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
a = rng.normal(0, 1, 300)                    # class 0
b = rng.normal(2, 1, 300)                    # class 1
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

axes[0].hist(a, bins=20, alpha=0.6, label="class 0")
axes[0].hist(b, bins=20, alpha=0.6, label="class 1")
axes[0].set_title("Histogram of a feature")
axes[0].legend()

X = np.column_stack([np.r_[a, b], rng.normal(size=600)])
y = np.r_[np.zeros(300), np.ones(300)]
axes[1].scatter(X[:, 0], X[:, 1], c=y, s=8, cmap="coolwarm")
axes[1].set_title("Scatter, coloured by class")
axes[1].set_xlabel("feature 1")
axes[1].set_ylabel("feature 2")

epochs = np.arange(1, 21)
axes[2].plot(epochs, 1 / epochs + 0.05, label="train")
axes[2].plot(epochs, 1 / epochs + 0.05 + 0.002 * epochs ** 1.5, label="validation")
axes[2].set_title("Learning curve")
axes[2].set_xlabel("epoch")
axes[2].legend()
fig.tight_layout()
plt.show()
print(round(float(a.mean()), 2), round(float(b.mean()), 2))`, "-0.04 1.99"],
    ["p", "In the third panel the validation curve turns upward while the training loss keeps falling: a sign of [overfitting](ml.html#bias-variance). Reading plots like this is a core ML skill."],
    ["h", "Heatmaps and decision boundaries"],
    ["p", "`imshow` draws a matrix as coloured cells: ideal for a confusion matrix, an attention map or a correlation matrix. To see what a 2-D classifier has learned, evaluate it on a grid of points covering the plane and colour each point by the predicted class:"],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier

X, y = make_moons(n_samples=200, noise=0.25, random_state=0)
clf = KNeighborsClassifier(n_neighbors=7).fit(X, y)

xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]                 # (40000, 2): every grid point
zz = clf.predict(grid).reshape(xx.shape)             # predicted class at each point

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")  # coloured regions
ax.scatter(X[:, 0], X[:, 1], c=y, s=14, cmap="coolwarm", edgecolor="k", linewidth=0.3)
ax.set_title("k-NN decision boundary (k=7)")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()
print(round(clf.score(X, y), 3))`, "0.935"],
    ["tip", "The same grid trick works for any classifier from [scikit-learn](pyml.html#sklearn-api). Compare how smooth the boundary is for different `n_neighbors` to see [bias and variance](ml.html#bias-variance) with your own eyes."],
    ["h", "Habits that save time"],
    ["list", [
      "**Always label the axes** and say what the units are. A plot you made last week should still make sense today.",
      "Plot the thing **before and after** a transformation (raw vs log-scaled, before vs after scaling).",
      "Use `ax.set_yscale(\"log\")` for loss curves and anything spanning orders of magnitude.",
      "In scripts, call `plt.savefig(\"fig.png\", dpi=150, bbox_inches=\"tight\")` *before* `plt.show()`; afterwards the figure may be empty.",
      "In a notebook, a figure left open from an earlier cell can draw on top of the next. Start each plot with a fresh `fig, ax = plt.subplots()`.",
    ]],
    ["danger", "Mixing the two styles in one plot is a common source of titles and labels landing on the wrong panel. With several subplots, use only the `ax.` methods (`ax.set_title`, `ax.set_xlabel`), not `plt.title` / `plt.xlabel`, which act on whichever axes happens to be current."],
  ],
},
{
  id: "sklearn-api", group: "scikit-learn", title: "The scikit-learn API",
  sub: "Every model and preprocessing step follows the same few verbs: fit, predict, transform, score.",
  tags: ["scikit-learn"],
  blocks: [
    ["p", "**scikit-learn** is the toolbox of classical machine learning: linear models, trees, forests, SVMs, k-NN, clustering, PCA, metrics and more. Its great strength is a **uniform interface**. Once you know the pattern, every one of its 100+ models works the same way, so swapping a model is a one-line change."],
    ["h", "The pattern: construct, fit, predict"],
    ["steps", [
      "**Construct** the object with its hyperparameters (the settings you choose): `model = KNeighborsClassifier(n_neighbors=5)`. Nothing is learned yet.",
      "**`fit(X, y)`** learns from the training data. It stores what it learned in attributes whose names **end in an underscore**, e.g. `model.coef_`.",
      "**`predict(X_new)`** returns predictions for new rows. For classifiers `predict_proba(X_new)` gives class probabilities.",
      "**`score(X, y)`** returns a default metric (accuracy for classifiers, $R^2$ for regressors).",
    ]],
    ["code", R`
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
print(X.shape, y.shape, np.unique(y))                  # 150 flowers, 4 features, 3 classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)

model = KNeighborsClassifier(n_neighbors=5)            # 1. construct
model.fit(X_train, y_train)                            # 2. fit (learn from training data)
pred = model.predict(X_test)                           # 3. predict on unseen rows
print(pred[:8], y_test[:8])
print(round(model.score(X_test, y_test), 3))           # 4. accuracy on the test set
print(model.predict_proba(X_test[:2]).round(2))        # class probabilities`, "(150, 4) (150,) [0 1 2]\n[2 2 0 0 1 0 1 2] [2 2 0 0 1 0 1 2]\n1.0\n[[0.  0.  1. ]\n [0.  0.2 0.8]]"],
    ["h", "Two kinds of object"],
    ["table", ["Kind", "Key methods", "Examples", "Output"], [
      ["**Predictor** (a model)", "`fit`, `predict`, `score`", "`LinearRegression`, `LogisticRegression`, `RandomForestClassifier`, `KNeighborsClassifier`", "predictions"],
      ["**Transformer** (preprocessing)", "`fit`, `transform`, `fit_transform`", "`StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `PCA`", "a new, transformed `X`"],
    ]],
    ["p", "A transformer **learns something from the data** (the mean and standard deviation of each column) in `fit`, then **applies** it in `transform`. The learned numbers are what you must reuse on new data:"],
    ["code", R`
import numpy as np
from sklearn.preprocessing import StandardScaler

X_train = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
X_test = np.array([[4.0, 250.0]])

sc = StandardScaler()
sc.fit(X_train)                           # learn mean_ and scale_ from TRAINING data only
print(sc.mean_, sc.scale_.round(3))
print(sc.transform(X_train).round(3).tolist())
print(sc.transform(X_test).round(3).tolist())    # reuse the training mean/std on new data
Z = StandardScaler().fit_transform(X_train)      # shortcut: fit then transform in one call
print(np.allclose(Z, sc.transform(X_train)))`, "[  2. 200.] [ 0.816 81.65 ]\n[[-1.225, -1.225], [0.0, 0.0], [1.225, 1.225]]\n[[2.449, 0.612]]\nTrue"],
    ["danger", "**Fit on the training data only.** `scaler.fit(X_all)` or `fit_transform(X_test)` lets test information leak into training (see [Data Leakage](ml.html#splits)). The rule: `fit` / `fit_transform` on **train**, `transform` only on validation and test. A [Pipeline](pyml.html#sklearn-pipelines) makes it automatic."],
    ["h", "What a fitted model looks like inside"],
    ["code", R`
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=(40, 1))                  # (n, 1): X must be 2-D
y = 4.0 + 2.5 * X[:, 0] + rng.normal(scale=0.3, size=40)
reg = LinearRegression().fit(X, y)                    # fit returns the model itself
print(reg.coef_.round(3), round(float(reg.intercept_), 3))   # learned: end with underscore
print(reg.predict([[2.0], [10.0]]).round(2))
print(round(reg.score(X, y), 3))                      # R^2
print(reg.get_params()["fit_intercept"])              # hyperparameters are visible too

try:
    LinearRegression().fit(X[:, 0], y)                # a flat (n,) array for X
except ValueError as e:
    print("ValueError:", str(e).split("\n")[0])`, "[2.514] 4.015\n[ 9.04 29.16]\n0.998\nTrue\nValueError: Expected 2D array, got 1D array instead:"],
    ["danger", "**X must be 2-D, shape `(n_samples, n_features)`; y is 1-D, `(n_samples,)`.** A single feature still needs `X.reshape(-1, 1)` or `X[:, None]`; otherwise scikit-learn raises \"Expected 2D array, got 1D array instead\". Predicting one row also needs two brackets: `predict([[2.0]])`."],
    ["h", "Swap models by changing one line"],
    ["p", "Because every classifier speaks the same language, comparing models is a loop. (This also shows why a single train/test split is a noisy yardstick; see [cross-validation](ml.html#cross-validation) for the careful way.)"],
    ["code", R`
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
models = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "decision tree": DecisionTreeClassifier(max_depth=4, random_state=0),
    "random forest": RandomForestClassifier(n_estimators=100, random_state=0),
}
for name, m in models.items():
    m.fit(Xtr, ytr)
    print(f"{name:20s} test accuracy {m.score(Xte, yte):.3f}")`, "logistic regression  test accuracy 0.958\nk-NN                 test accuracy 0.951\ndecision tree        test accuracy 0.909\nrandom forest        test accuracy 0.944"],
    ["tip", "Meet the models properly in the Machine Learning page: [linear regression](ml.html#linear-regression), [logistic regression](ml.html#logistic-regression), [k-NN](ml.html#knn), [decision trees](ml.html#decision-trees), [random forests](ml.html#random-forests). Unsupervised ones (`KMeans`, `PCA`) use `fit` and `transform` / `labels_` instead of `predict` with labels: see [k-Means](ml.html#kmeans) and [PCA](ml.html#pca)."],
    ["note", "Always pass `random_state=` to anything random (`train_test_split`, forests, `KMeans`, ...), as in [Random Numbers & Seeds](pyml.html#numpy-random). Forgetting it makes results change from run to run."],
  ],
},
{
  id: "sklearn-pipelines", group: "scikit-learn", title: "Pipelines, Splits & Cross-Validation in scikit-learn",
  sub: "Chain preprocessing and a model into one object, split data correctly, and evaluate and tune without leaking.",
  tags: ["scikit-learn", "evaluation"],
  blocks: [
    ["p", "A real model is rarely just a model. It is: fill missing values, scale numbers, encode categories, then fit. If you run those steps by hand it is easy to fit a scaler on the test set by accident, or to forget a step at prediction time. A **Pipeline** bundles the steps into one estimator that you `fit` and `predict` like any other, and it guarantees each step is fitted on training data only."],
    ["h", "Step 1: split first"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)    # stratify keeps class ratios
print(X_tr.shape, X_te.shape)
print(y.mean().round(3), y_tr.mean().round(3), y_te.mean().round(3))   # same class balance`, "(455, 30) (114, 30)\n0.627 0.626 0.632"],
    ["p", "`stratify=y` makes the class proportions the same in both parts, which matters for imbalanced data ([Imbalanced Data](ml.html#imbalanced)). Split **before** looking at statistics, scaling, or imputing. For the reasoning see [Train / Validation / Test](ml.html#splits)."],
    ["h", "Step 2: the leaky way vs the pipeline way"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# WRONG: the scaler sees the test rows
leaky = StandardScaler().fit(X)
print("feature-0 mean, all rows:", leaky.mean_[0].round(3), "| train rows only:", StandardScaler().fit(X_tr).mean_[0].round(3))

# RIGHT: the pipeline fits every step on the training data only
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_tr, y_tr)                        # scaler.fit_transform, then clf.fit
print(round(pipe.score(X_te, y_te), 3))     # scaler.transform, then clf.score
print(pipe.named_steps["clf"].coef_.shape)  # reach inside by step name`, "feature-0 mean, all rows: 14.127 | train rows only: 14.141\n0.982\n(1, 30)"],
    ["h", "Different columns, different treatment"],
    ["p", "Tabular data mixes numbers and categories. **ColumnTransformer** applies one recipe to the numeric columns and another to the categorical ones, then glues the results side by side:"],
    ["code", R`
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
    "age": [25, np.nan, 31, 40, 52, 36],
    "income": [50.0, 62.0, 58.0, np.nan, 90.0, 75.0],
    "city": ["Pune", "Goa", "Pune", "Delhi", "Goa", "Pune"],
})
y = np.array([0, 0, 1, 0, 1, 1])

prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age", "income"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])
model = Pipeline([("prep", prep), ("clf", LogisticRegression())])
model.fit(df, y)
print(prep.fit_transform(df).shape)                 # 2 numeric + 3 one-hot columns
print(prep.get_feature_names_out().tolist())
print(model.predict(pd.DataFrame({"age": [30], "income": [60.0], "city": ["Mumbai"]})))   # unseen city is fine`, "(6, 5)\n['num__age', 'num__income', 'cat__city_Delhi', 'cat__city_Goa', 'cat__city_Pune']\n[0]"],
    ["h", "Step 3: cross-validation through the pipeline"],
    ["p", "One split gives one noisy number. **k-fold cross-validation** splits the training data into k parts, trains on k-1 and validates on the remaining one, k times, and averages. Passing the *pipeline* (not pre-scaled data) to `cross_val_score` means the scaler is refitted inside every fold: no leakage. The full idea is in [Cross-Validation & Hyperparameter Tuning](ml.html#cross-validation)."],
    ["code", R`
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy")
print(scores.round(3))
print(round(scores.mean(), 3), "+/-", round(scores.std(), 3))`, "[0.989 0.967 0.978 0.967 0.989]\n0.978 +/- 0.01"],
    ["h", "Step 4: tune hyperparameters"],
    ["p", "**GridSearchCV** tries every combination of the settings you list, scores each one with cross-validation, then refits the best on all the training data. Parameters of a pipeline step are written `stepname__parameter`:"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
grid = GridSearchCV(
    pipe,
    param_grid={"clf__C": [0.01, 0.1, 1, 10]},          # stepname__param
    cv=StratifiedKFold(5, shuffle=True, random_state=0),
    scoring="accuracy",
)
grid.fit(X_tr, y_tr)
print(grid.best_params_, round(grid.best_score_, 3))     # chosen using validation folds only
print(round(grid.score(X_te, y_te), 3))                  # the test set is used once, at the end`, "{'clf__C': 1} 0.978\n0.982"],
    ["table", ["Tool", "What it does", "Remember"], [
      ["`train_test_split`", "one random split", "set `random_state`; `stratify=y` for classification"],
      ["`Pipeline` / `make_pipeline`", "chain steps; fit them on training data only", "the last step is the model"],
      ["`ColumnTransformer`", "different steps for different columns", "list the columns per step"],
      ["`cross_val_score`", "k scores from k-fold CV", "pass the pipeline, not pre-processed data"],
      ["`GridSearchCV`", "try every hyperparameter combination with CV", "`step__param` names; test set stays untouched"],
    ]],
    ["danger", "Three leakage traps: (1) scaling, imputing or selecting features on the full dataset before splitting; (2) tuning hyperparameters while looking at the **test** score (use the cross-validation score; the test set is touched once); (3) shuffling a **time series** randomly, so the model trains on the future. For time data use `TimeSeriesSplit`."],
    ["note", "You now have the toolbox. Continue with [What is Machine Learning?](ml.html#what-is-ml) and [The ML Workflow](ml.html#workflow), or build the algorithms from scratch with the exercises in [Practice](practice.html)."],
  ],
},
);
