/* Practice: NumPy Basics + Linear Algebra
   Each problem: q (task), ex (examples), hint, brute (plain Python), opt (vectorized NumPy), run + out (the tests),
   dry (a dry run), expl (code explained line by line), take, mist. Tests print rounded values so floats compare exactly. */
PRAC.add("NumPy Basics", [
{
  id: "softmax", t: "Stable Softmax", d: "Easy", p: "Vectorization",
  q: "Write `softmax(z)` that turns a 1-D NumPy array of scores (**logits**) into probabilities:\n\n$\\text{softmax}(z)_i = \\dfrac{e^{z_i}}{\\sum_j e^{z_j}}$\n\nThe result must be positive and add up to 1. It must also work for huge scores like `[1000, 1000]`, where `np.exp(1000)` overflows to `inf`.",
  ex: [["z = [1.0, 2.0, 3.0]", "[0.09, 0.2447, 0.6652]", "e¹, e², e³ ≈ 2.72, 7.39, 20.09; each divided by their sum 30.19."],
    ["z = [1000.0, 1000.0]", "[0.5, 0.5]", "Equal scores share the probability equally, even when they are huge."]],
  c: ["`z` is a 1-D float array with at least one element", "Return a NumPy array of the same length"],
  hint: "Softmax doesn't change if you subtract the same number from every score: the $e^{-c}$ cancels in the fraction. Which number should you subtract so that the biggest exponent is $e^0 = 1$?",
  brute: {
    name: "Plain Python loops",
    idea: "Find the maximum, subtract it from every score, take `math.exp` of each, add them up, and divide each one by the total.",
    code: R`
import math
import numpy as np

def softmax(z):
    m = max(z)                                # the largest score
    exps = [math.exp(v - m) for v in z]       # every exponent is now <= e^0 = 1
    total = sum(exps)
    return np.array([e / total for e in exps])`,
    tc: "O(n)", tw: "One pass for the max, one for the exponents, one to divide.", sc: "O(n)", sw: "The list of exponentials.",
    bad: "Correct, but a Python loop runs one number at a time. On a batch of 50,000-token vocabularies (what an LLM does every step) NumPy is 50–100× faster.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Do the same three things on the whole array at once: `z - np.max(z)` shifts every score, `np.exp` exponentiates every element, and dividing by `.sum()` normalises. Subtracting the max is the standard **numerically stable** trick: the largest exponent becomes $e^0 = 1$, so nothing overflows, and the answer is mathematically unchanged.",
    steps: ["Shift: `z - np.max(z)` (the largest score becomes 0, the rest negative).", "Exponentiate every element with `np.exp`: all values are now in (0, 1].", "Divide by their sum so they add up to 1."],
    code: R`
import numpy as np

def softmax(z):
    e = np.exp(z - np.max(z))     # shift so the largest score is 0: exp never overflows
    return e / e.sum()`,
    tc: "O(n)", tw: "Three vectorized passes over the array.", sc: "O(n)", sw: "The array of exponentials.",
  },
  run: R`
print(np.round(softmax(np.array([1.0, 2.0, 3.0])), 4).tolist())
print(np.round(softmax(np.array([1000.0, 1000.0])), 4).tolist())
print(np.round(softmax(np.array([5.0, -2.0, 0.5])), 4).tolist())
print(round(float(softmax(np.array([0.3, -1.2, 4.0, 2.2])).sum()), 6))`,
  out: R`
[0.09, 0.2447, 0.6652]
[0.5, 0.5]
[0.9881, 0.0009, 0.011]
1.0`,
  dryOn: "z = [1.0, 2.0, 3.0]",
  dry: { h: ["step", "value"], r: [["max(z)", "3.0"], ["z − max", "[−2, −1, 0]"], ["exp", "[0.1353, 0.3679, 1.0]"], ["sum", "1.5032"], ["divide", "**[0.09, 0.2447, 0.6652]**"]] },
  expl: [
    ["e = np.exp(z - np.max(z))", "Subtract the largest score from every element (broadcasting), then exponentiate element by element. The biggest value is exactly 1, so no overflow."],
    ["return e / e.sum()", "Divide every element by the total so the probabilities add up to 1."],
  ],
  take: "**Softmax = exp, then normalise. Always subtract the max first**: it changes nothing mathematically and stops `exp` from overflowing.",
  mist: ["Forgetting the max shift: `np.exp(1000)` is `inf`, and `inf / inf` gives `nan`.", "Subtracting the *mean* instead of the max: large scores can still overflow.", "Using `sum(e)` (Python's sum) instead of `e.sum()`: it works, but it's a slow Python loop."],
},
]);

PRAC.add("Linear Algebra", [
{
  id: "cosine-similarity", t: "Cosine Similarity Search", d: "Easy", p: "Dot product",
  q: "Embeddings are compared with **cosine similarity**, the cosine of the angle between two vectors:\n\n$\\cos(a, b) = \\dfrac{a \\cdot b}{\\lVert a \\rVert \\, \\lVert b \\rVert}$\n\nIt is 1 for vectors pointing the same way, 0 for perpendicular ones and −1 for opposite ones. Write two functions:\n\n1. `cosine_similarity(a, b)` for two 1-D vectors, returning a Python float.\n2. `most_similar(query, M)`: `M` is a matrix with one document vector per row. Return the **index** of the row most similar to `query`. This is the core of semantic search and RAG.",
  ex: [["a = [1, 2, 3], b = [2, 4, 6]", "1.0", "b is a scaled copy of a: same direction, so the angle is 0."],
    ["a = [1, 0], b = [0, 1]", "0.0", "Perpendicular vectors."],
    ["query = [0, 0.2, 1], M = [[1,0,0], [0.6,0.8,0], [0,0.1,0.9]]", "2", "Row 2 points almost the same way as the query."]],
  c: ["Vectors are non-zero", "`M` has shape (n_docs, d) and `query` has shape (d,)"],
  hint: "`a @ b` is the dot product and `np.linalg.norm(a)` is the length. For `most_similar`, `M @ query` gives every row's dot product with the query in one go, and `np.linalg.norm(M, axis=1)` gives every row's length.",
  brute: {
    name: "Plain Python loops",
    idea: "Compute the dot product and both lengths with loops. For the search, compute the similarity of each row in turn and remember the best index.",
    code: R`
import math
import numpy as np

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return float(dot / (na * nb))

def most_similar(query, M):
    best, best_i = -2.0, -1
    for i, row in enumerate(M):
        s = cosine_similarity(row, query)
        if s > best:
            best, best_i = s, i
    return best_i`,
    tc: "O(n·d)", tw: "n rows, each costing O(d) for the dot product and norms.", sc: "O(1)", sw: "Just running totals.",
    bad: "Fine for 3 documents. A real vector search compares a query against millions of rows: one matrix–vector product in NumPy (or on a GPU) is enormously faster than a Python loop.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "`a @ b` and `np.linalg.norm` do the single-pair case. For the search, one matrix–vector product `M @ query` computes **every** row's dot product at once; dividing by every row's norm (`axis=1`) times the query's norm gives all similarities, and `np.argmax` picks the best.",
    steps: ["`M @ query` → a vector of n dot products.", "`np.linalg.norm(M, axis=1)` → a vector of n row lengths.", "Divide element-wise (and by the query's length) → n cosine similarities.", "`np.argmax` → the index of the largest."],
    code: R`
import numpy as np

def cosine_similarity(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def most_similar(query, M):
    sims = M @ query / (np.linalg.norm(M, axis=1) * np.linalg.norm(query))
    return int(np.argmax(sims))`,
    tc: "O(n·d)", tw: "Same amount of arithmetic, but done in fast compiled code.", sc: "O(n)", sw: "The vector of n similarities.",
  },
  run: R`
print(round(cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 4))
print(round(cosine_similarity(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])), 4))
print(round(cosine_similarity(np.array([1.0, 2.0]), np.array([-1.0, -2.0])), 4))
print(round(cosine_similarity(np.array([3.0, 4.0]), np.array([4.0, 3.0])), 4))
docs = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.1, 0.9]])
print(most_similar(np.array([0.0, 0.2, 1.0]), docs))
print(most_similar(np.array([2.0, 2.5, 0.0]), docs))`,
  out: R`
0.0
1.0
-1.0
0.96
2
1`,
  dryOn: "a = [3, 4], b = [4, 3]",
  dry: { h: ["quantity", "value"], r: [["a · b", "3·4 + 4·3 = 24"], ["‖a‖", "√(9 + 16) = 5"], ["‖b‖", "√(16 + 9) = 5"], ["cos", "24 / 25 = **0.96**"]] },
  expl: [
    ["a @ b", "`@` is matrix multiplication; for two 1-D arrays it is the dot product."],
    ["np.linalg.norm(a)", "The vector's length √(Σ aᵢ²)."],
    ["M @ query", "Matrix × vector: row i of the result is (row i of M) · query, for every row at once."],
    ["np.linalg.norm(M, axis=1)", "The length of each row (axis=1 means: reduce across the columns)."],
    ["int(np.argmax(sims))", "Index of the largest similarity, as a plain Python int."],
  ],
  take: "**Cosine similarity = dot product of the two vectors divided by their lengths.** For search, stack the documents in a matrix and do one `M @ query`.",
  mist: ["Forgetting to divide by the norms: the raw dot product favours long vectors, not similar directions.", "Using `axis=0` for the row norms: that gives column lengths.", "Returning a NumPy integer instead of `int(...)`: it prints the same but isn't a plain Python int."],
},
]);
