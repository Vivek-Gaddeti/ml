/* Practice: Classical ML, part 2 (ALS update for matrix factorization, AdaBoost round, Apriori, Kalman step)
   Each problem: q, ex, hint, brute, opt, run + out (tests), dry, expl, take, mist. */
PRAC.add("Classical ML", [
{
  id: "als-user-update", t: "ALS Update for One User", d: "Easy", p: "recsys-mf",
  q: "Write `als_update_user(Q, r, observed, lam)` that performs one **user step of Alternating Least Squares** for matrix factorization.\n\n- `Q` is an (m, k) array of item factor vectors (row i is $\\mathbf q_i$).\n- `r` is a length-m array of this user's ratings, and `observed` is a length-m boolean array that is `True` where the user actually rated the item (the other entries of `r` are meaningless).\n- `lam` is the regularization strength $\\lambda$.\n\nReturn the length-k user vector that minimises $\\sum_{i\\,\\text{observed}}(r_i-\\mathbf p^\\top\\mathbf q_i)^2+\\lambda\\|\\mathbf p\\|^2$, which is\n\n$$\\mathbf p=(\\mathbf Q_o^\\top\\mathbf Q_o+\\lambda\\mathbf I)^{-1}\\mathbf Q_o^\\top\\mathbf r_o$$\n\nwhere $\\mathbf Q_o,\\mathbf r_o$ keep only the observed rows. If the user rated nothing, return a vector of zeros.",
  ex: [["Q = [[1], [2]], r = [3, 5], both observed, lam = 1", "[2.167]", "$Q^\\top Q=5$, $Q^\\top r=13$, so $p=13/(5+1)$."],
    ["same, but only item 0 observed", "[1.5]", "$Q^\\top Q=1$, $Q^\\top r=3$, so $p=3/(1+1)$."]],
  c: ["`Q` has shape (m, k) with k >= 1", "`lam` >= 0 (use `lam > 0` to guarantee the system is solvable)"],
  hint: "Mask the rows first: `Qo = Q[observed]`, `ro = r[observed]`. Then form the k×k matrix and solve a linear system with `np.linalg.solve` instead of inverting it.",
  brute: {
    name: "Plain Python loops",
    idea: "Accumulate the k×k matrix $\\sum\\mathbf q_i\\mathbf q_i^\\top+\\lambda\\mathbf I$ and the vector $\\sum r_i\\mathbf q_i$ entry by entry over the observed items, then invert the matrix explicitly.",
    code: R`
import numpy as np

def als_update_user(Q, r, observed, lam):
    Q = np.asarray(Q, float); r = np.asarray(r, float)
    m, k = Q.shape
    A = lam * np.eye(k)
    b = np.zeros(k)
    for i in range(m):
        if observed[i]:
            for a in range(k):
                b[a] += r[i] * Q[i, a]
                for c in range(k):
                    A[a, c] += Q[i, a] * Q[i, c]
    return np.linalg.inv(A) @ b`,
    tc: "O(m·k² + k³)", tw: "Triple loop over items and factor pairs, then an inversion.", sc: "O(k²)", sw: "The k×k matrix.",
    bad: "Correct, but the sum of outer products is exactly `Qo.T @ Qo`, and `inv(A) @ b` is slower and less accurate than `solve(A, b)`.",
  },
  opt: {
    name: "Vectorized ridge regression",
    idea: "The user step is a ridge regression of the observed ratings on the item factors. Select the observed rows with a boolean mask, form `Qo.T @ Qo + lam*I` and `Qo.T @ ro`, and solve.",
    steps: ["`Qo = Q[observed]`, `ro = r[observed]`.", "`A = Qo.T @ Qo + lam * np.eye(k)` and `b = Qo.T @ ro`.", "`return np.linalg.solve(A, b)`. With no observed items, `A = lam*I` and `b = 0`, so the answer is 0 (if lam > 0)."],
    code: R`
import numpy as np

def als_update_user(Q, r, observed, lam):
    Q = np.asarray(Q, float); r = np.asarray(r, float)
    observed = np.asarray(observed, bool)
    Qo, ro = Q[observed], r[observed]                      # only the rated items
    k = Q.shape[1]
    A = Qo.T @ Qo + lam * np.eye(k)                        # k x k normal equations with the ridge term
    return np.linalg.solve(A, Qo.T @ ro)                   # no explicit inverse`,
    tc: "O(m·k² + k³)", tw: "One matrix product and one k×k solve.", sc: "O(m·k)", sw: "The masked copy of Q.",
    viz: "cml2_als",
  },
  run: R`
print(np.round(als_update_user([[1], [2]], [3, 5], [True, True], 1.0), 3).tolist())
print(np.round(als_update_user([[1], [2]], [3, 5], [True, False], 1.0), 3).tolist())
Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
r = np.array([2.0, 3.0, 5.0])
p = als_update_user(Q, r, [True, True, True], 0.0)
print(np.round(p, 3).tolist(), np.round(Q @ p, 3).tolist())
print(np.round(als_update_user(Q, r, [True, True, True], 5.0), 3).tolist())
print(np.round(als_update_user(Q, r, [False, False, False], 1.0), 3).tolist())
print(np.round(als_update_user(Q, [9, 3, 5], [False, True, True], 0.5), 3).tolist())`,
  out: R`
[2.167]
[1.5]
[2.0, 3.0] [2.0, 3.0, 5.0]
[0.854, 1.021]
[0.0, 0.0]
[1.636, 2.545]`,
  dryOn: "Q = [[1], [2]], r = [3, 5], lam = 1, both items observed",
  dry: { h: ["quantity", "value"], r: [["Qo", "[[1], [2]]"], ["Qoᵀ Qo", "1² + 2² = 5"], ["A = Qoᵀ Qo + λ I", "5 + 1 = 6"], ["b = Qoᵀ ro", "1·3 + 2·5 = 13"], ["p = A⁻¹ b", "**13 / 6 = 2.167**"]] },
  expl: [
    ["Qo, ro = Q[observed], r[observed]", "A boolean mask drops the rows the user never rated: unobserved cells must not count as data."],
    ["Qo.T @ Qo + lam * np.eye(k)", "The ridge normal-equation matrix. The λI term keeps it invertible and shrinks the vector."],
    ["Qo.T @ ro", "Right-hand side: the observed ratings weighted by the item factors."],
    ["np.linalg.solve(A, ...)", "Solves A p = b directly; faster and more accurate than forming A⁻¹."],
  ],
  take: "**An ALS step is just a ridge regression** on the rows of the other factor matrix that correspond to observed ratings; every user (and then every item) is solved independently.",
  mist: ["Using all rows of `Q` and treating missing ratings as 0, which teaches the model that unseen items are disliked.", "Forgetting the `lam * I` term, so a user with fewer than k ratings gives a singular system.", "Calling `np.linalg.inv` instead of `solve`."],
},
{
  id: "adaboost-round", t: "One Round of AdaBoost", d: "Medium", p: "adaboost",
  q: "Write `adaboost_round(y, pred, w)` that carries out the bookkeeping of one AdaBoost round, **after** the weak learner has already made its predictions.\n\n- `y` and `pred` are arrays of labels in {-1, +1}; `w` is an array of positive sample weights that sums to 1.\n\nReturn a tuple `(eps, alpha, new_w)`:\n\n1. the weighted error $\\varepsilon=\\sum_i w_i\\,[\\,pred_i\\ne y_i\\,]$,\n2. the learner's vote $\\alpha=\\tfrac12\\ln\\frac{1-\\varepsilon}{\\varepsilon}$,\n3. the updated weights $w_i\\leftarrow w_i\\,e^{-\\alpha\\,y_i\\,pred_i}$, normalised to sum to 1.\n\nTo avoid division by zero and log of zero, clip $\\varepsilon$ to the range [1e-10, 1 - 1e-10] before computing $\\alpha$ (return the clipped value as `eps`).",
  ex: [["y = [1, 1, -1, -1, 1], pred = [1, 1, -1, -1, -1], w = [0.2] * 5", "eps 0.2, alpha 0.693, new_w [0.125, 0.125, 0.125, 0.125, 0.5]", "One mistake out of five equal weights. The mistake's weight doubles, the others halve, and after normalising the mistake holds half of the total weight."]],
  c: ["labels are exactly -1 or +1", "`w` positive and sums to 1", "a perfect or perfectly wrong learner must not crash"],
  hint: "`y * pred` is +1 for a correct prediction and -1 for a mistake, so the update factor is `np.exp(-alpha * y * pred)`. The error is `w[pred != y].sum()`.",
  brute: {
    name: "Plain Python loops",
    idea: "Loop over the samples to add up the weight of mistakes, then loop again to rescale each weight and a third time to normalise.",
    code: R`
import math
import numpy as np

def adaboost_round(y, pred, w):
    n = len(y)
    eps = 0.0
    for i in range(n):
        if pred[i] != y[i]:
            eps += w[i]
    eps = min(max(eps, 1e-10), 1 - 1e-10)
    alpha = 0.5 * math.log((1 - eps) / eps)
    new_w = []
    for i in range(n):
        new_w.append(w[i] * math.exp(-alpha * y[i] * pred[i]))
    total = sum(new_w)
    return eps, alpha, np.array([v / total for v in new_w])`,
    tc: "O(n)", tw: "Three passes.", sc: "O(n)", sw: "The new weights.",
    bad: "Correct and linear, but three explicit loops hide the idea; everything is a single vector expression in NumPy.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "`pred != y` is a boolean mask whose weighted sum is the error. The update factor is `exp(-alpha * y * pred)` applied to the whole weight vector, followed by one normalisation.",
    steps: ["`eps = w[pred != y].sum()`, clipped to [1e-10, 1-1e-10].", "`alpha = 0.5 * np.log((1 - eps) / eps)`.", "`new_w = w * np.exp(-alpha * y * pred)`; divide by `new_w.sum()`."],
    code: R`
import numpy as np

def adaboost_round(y, pred, w):
    y = np.asarray(y); pred = np.asarray(pred); w = np.asarray(w, float)
    eps = float(np.clip(w[pred != y].sum(), 1e-10, 1 - 1e-10))   # weighted error of the weak learner
    alpha = 0.5 * np.log((1 - eps) / eps)                          # its vote
    new_w = w * np.exp(-alpha * y * pred)                          # up-weight mistakes, down-weight hits
    return eps, float(alpha), new_w / new_w.sum()                  # renormalise`,
    tc: "O(n)", tw: "A few vectorised passes.", sc: "O(n)", sw: "The new weights.",
    viz: "cml2_adaboost",
  },
  run: R`
y = np.array([1, 1, -1, -1, 1]); pred = np.array([1, 1, -1, -1, -1]); w = np.full(5, 0.2)
eps, alpha, nw = adaboost_round(y, pred, w)
print(round(eps, 3), round(alpha, 3), np.round(nw, 3).tolist())
print(round(float(nw[pred != y].sum()), 3))
eps, alpha, nw = adaboost_round(y, y, w)
print(eps, round(alpha, 2), np.round(nw, 3).tolist())
w2 = np.array([0.1, 0.4, 0.1, 0.3, 0.1])
eps, alpha, nw = adaboost_round(y, pred, w2)
print(round(eps, 3), round(alpha, 3), np.round(nw, 3).tolist())
eps, alpha, nw = adaboost_round(y, -y, w)
print(round(eps, 3), round(alpha, 2), np.round(nw, 3).tolist())
eps, alpha, nw = adaboost_round(np.array([1, -1]), np.array([1, 1]), np.array([0.5, 0.5]))
print(round(eps, 3), round(alpha, 3), np.round(nw, 3).tolist())`,
  out: R`
0.2 0.693 [0.125, 0.125, 0.125, 0.125, 0.5]
0.5
1e-10 11.51 [0.2, 0.2, 0.2, 0.2, 0.2]
0.1 1.099 [0.056, 0.222, 0.056, 0.167, 0.5]
1.0 -11.51 [0.2, 0.2, 0.2, 0.2, 0.2]
0.5 0.0 [0.5, 0.5]`,
  dryOn: "y = [1, 1, -1, -1, 1], pred = [1, 1, -1, -1, -1], w = 0.2 each",
  dry: { h: ["sample", "correct?", "w", "w · e^(−α y pred)", "new w"], r: [["1", "yes", "0.2", "0.2 · 0.5 = 0.1", "0.1 / 0.8 = 0.125"], ["2", "yes", "0.2", "0.1", "0.125"], ["3", "yes", "0.2", "0.1", "0.125"], ["4", "yes", "0.2", "0.1", "0.125"], ["5", "**no**", "0.2", "0.2 · 2 = 0.4", "0.4 / 0.8 = **0.5**"]] },
  expl: [
    ["eps = ... w[pred != y].sum()", "Weighted error: add up the weights of the mistakes."],
    ["np.clip(..., 1e-10, 1 - 1e-10)", "Keeps the log and the division finite when the learner is perfect or always wrong."],
    ["alpha = 0.5 * np.log((1 - eps) / eps)", "Zero vote at ε = 0.5, large positive for small errors, negative if the learner is worse than chance."],
    ["np.exp(-alpha * y * pred)", "y·pred is +1 for hits (weight shrinks by e^−α) and −1 for misses (weight grows by e^α)."],
    ["new_w / new_w.sum()", "Renormalise so the weights are again a distribution."],
  ],
  take: "**After an AdaBoost update the learner that was just trained has weighted error exactly 1/2**: the misclassified points together carry half of the weight, so the next learner must do something new.",
  mist: ["Forgetting to renormalise the weights.", "Using the unweighted error rate instead of the weighted error.", "Flipping the sign in the exponent, which down-weights the mistakes and makes boosting repeat the same errors.", "Letting ε hit 0 or 1 and crashing on `log(0)`."],
},
{
  id: "apriori-rules", t: "Frequent Itemsets and Association Rules", d: "Medium", p: "apriori",
  q: "Write `apriori_rules(T, minsup, minconf)` for a list of transactions `T` (each a set of strings).\n\nReturn a pair `(frequent, rules)`:\n\n- `frequent`: a dict mapping each **frequent itemset**, as a sorted tuple of items, to its count (the number of transactions that contain it). An itemset is frequent if `count / len(T) >= minsup`.\n- `rules`: a list of tuples `(A, B, support, confidence, lift)` for every rule $A\\Rightarrow B$ made from a frequent itemset $A\\cup B$ with `A` and `B` non-empty disjoint sorted tuples, `confidence >= minconf`. Here $\\text{support}=\\text{count}(A\\cup B)/N$, $\\text{confidence}=\\text{count}(A\\cup B)/\\text{count}(A)$ and $\\text{lift}=\\text{confidence}/\\text{support}(B)$. Round support, confidence and lift to 3 decimals. Sort the rules by decreasing confidence, ties by `A` then `B`.\n\nUse the **Apriori property** (candidates need all their subsets frequent) so you do not count every possible subset.",
  ex: [["T = [{a,b}, {a,c}, {a,b,c}, {b}], minsup = 0.5, minconf = 0.7", "frequent {a:3, b:3, c:2, ab:2, ac:2}; rules include (c) → (a) with confidence 1.0", "Every transaction with c also has a, so confidence 1 and lift 1.0 / 0.75 = 1.333."]],
  c: ["items are strings; `T` has at least one transaction", "0 < minsup <= 1, 0 < minconf <= 1"],
  hint: "Keep a dict for the current level $L_k$. Build candidates by joining pairs from $L_k$ whose union has size k+1, drop any candidate with a k-subset not in $L_k$, then count the survivors. Confidence needs the counts of both A and A∪B, which you already have.",
  brute: {
    name: "Count every possible subset",
    idea: "Enumerate all non-empty subsets of the item vocabulary (2^d - 1 of them), count each in every transaction, and keep the frequent ones. Correct, but exponential in the number of distinct items.",
    code: R`
from itertools import combinations

def apriori_rules(T, minsup, minconf):
    N = len(T)
    items = sorted({i for t in T for i in t})
    frequent = {}
    for r in range(1, len(items) + 1):
        for c in combinations(items, r):                      # every subset, no pruning
            cnt = sum(set(c) <= t for t in T)
            if cnt / N >= minsup:
                frequent[c] = cnt
    rules = []
    for S, cnt in frequent.items():
        for r in range(1, len(S)):
            for A in combinations(S, r):
                B = tuple(x for x in S if x not in A)
                conf = cnt / frequent[A]
                if conf >= minconf:
                    rules.append((A, B, round(cnt / N, 3), round(conf, 3), round(conf / (frequent[B] / N), 3)))
    rules.sort(key=lambda t: (-t[3], t[0], t[1]))
    return dict(sorted(frequent.items(), key=lambda kv: (len(kv[0]), kv[0]))), rules`,
    tc: "O(2^d · N)", tw: "All subsets are counted whether or not their parts are frequent.", sc: "O(2^d)", sw: "Counts for every frequent subset.",
    bad: "Correct, but with 100 items there are about 10^30 subsets. The Apriori property says an itemset with an infrequent subset can never be frequent, so most are never worth counting.",
  },
  opt: {
    name: "Apriori: level by level with pruning",
    idea: "Start with frequent single items. Join frequent k-itemsets into (k+1)-candidates, discard any candidate with an infrequent k-subset, count the rest, keep the frequent ones, and repeat until a level is empty. Then generate rules by splitting each frequent itemset.",
    steps: ["Count single items and keep `count >= minsup * N`.", "Candidates: unions of two frequent k-itemsets with size k+1; prune those with a k-subset that is not frequent.", "Count the survivors, keep the frequent ones; loop until empty.", "For each frequent itemset and each non-empty proper subset A: confidence = count(S) / count(A); lift = confidence / (count(B) / N)."],
    code: R`
from itertools import combinations

def apriori_rules(T, minsup, minconf):
    N = len(T)
    count = lambda s: sum(set(s) <= t for t in T)
    items = sorted({i for t in T for i in t})
    Lk = {(i,): count((i,)) for i in items}
    Lk = {s: c for s, c in Lk.items() if c / N >= minsup}
    frequent = {}
    while Lk:
        frequent.update(Lk)
        k = len(next(iter(Lk)))
        cands = {tuple(sorted(set(a) | set(b))) for a in Lk for b in Lk if len(set(a) | set(b)) == k + 1}
        cands = {c for c in cands if all(sub in Lk for sub in combinations(c, k))}     # Apriori pruning
        Lk = {c: count(c) for c in cands}
        Lk = {c: n for c, n in Lk.items() if n / N >= minsup}
    rules = []
    for S, cnt in frequent.items():
        for r in range(1, len(S)):
            for A in combinations(S, r):
                B = tuple(x for x in S if x not in A)
                conf = cnt / frequent[A]                                                # both counts are already known
                if conf >= minconf:
                    rules.append((A, B, round(cnt / N, 3), round(conf, 3), round(conf / (frequent[B] / N), 3)))
    rules.sort(key=lambda t: (-t[3], t[0], t[1]))
    return dict(sorted(frequent.items(), key=lambda kv: (len(kv[0]), kv[0]))), rules`,
    tc: "O(N · #candidates)", tw: "Pruning keeps the number of candidates small when minsup is not tiny.", sc: "O(#frequent itemsets)", sw: "Counts of the frequent itemsets.",
  },
  run: R`
T = [{"a", "b"}, {"a", "c"}, {"a", "b", "c"}, {"b"}]
freq, rules = apriori_rules(T, 0.5, 0.7)
print(freq)
for r in rules: print(r)
B = [{"bread", "milk"}, {"bread", "diaper", "beer", "eggs"}, {"milk", "diaper", "beer", "cola"},
     {"bread", "milk", "diaper", "beer"}, {"bread", "milk", "diaper", "cola"}]
freq, rules = apriori_rules(B, 0.6, 0.75)
print(freq)
print(len(rules), rules[0])
freq, rules = apriori_rules(B, 0.4, 0.9)
print(len(freq), max(len(k) for k in freq))
print(rules[:2])`,
  out: R`
{('a',): 3, ('b',): 3, ('c',): 2, ('a', 'b'): 2, ('a', 'c'): 2}
(('c',), ('a',), 0.5, 1.0, 1.333)
{('beer',): 3, ('bread',): 4, ('diaper',): 4, ('milk',): 4, ('beer', 'diaper'): 3, ('bread', 'diaper'): 3, ('bread', 'milk'): 3, ('diaper', 'milk'): 3}
8 (('beer',), ('diaper',), 0.6, 1.0, 1.25)
17 3
[(('beer',), ('diaper',), 0.6, 1.0, 1.25), (('beer', 'bread'), ('diaper',), 0.4, 1.0, 1.25)]`,
  dryOn: "T = [{a,b}, {a,c}, {a,b,c}, {b}], minsup = 0.5 (count >= 2)",
  dry: { setup: "N = 4, so an itemset is frequent if it appears in at least 2 transactions.", steps: [["L1: a = 3, b = 3, c = 2", "all three reach the minimum"], ["candidates: ab, ac, bc (all subsets frequent)", "counts: ab = 2, ac = 2, bc = 1 → keep **ab, ac**"], ["candidates: abc needs bc frequent", "bc is not, so **abc is pruned without counting**"], ["rule c → a", "count(ac) / count(c) = 2 / 2 = 1.0, lift = 1.0 / (3/4) = 1.333"]], result: "Frequent: a, b, c, ab, ac. Strong rules include c → a (confidence 1.0)." },
  expl: [
    ["count = lambda s: sum(set(s) <= t for t in T)", "A subset test per transaction; summing the booleans gives the number of transactions containing s."],
    ["cands = {tuple(sorted(set(a) | set(b))) ...}", "Joining two frequent k-itemsets whose union has k+1 items yields a candidate; sorting makes the tuple canonical."],
    ["all(sub in Lk for sub in combinations(c, k))", "The Apriori property: if any k-subset is not frequent, the candidate cannot be, so skip it."],
    ["conf = cnt / frequent[A]", "Both count(A∪B) and count(A) are in `frequent` because subsets of frequent sets are frequent."],
  ],
  take: "**Downward closure drives Apriori**: a frequent itemset has only frequent subsets, so you never count a candidate that has an infrequent part. Rules then come for free from the stored counts.",
  mist: ["Not sorting items inside the tuples, so ('a','b') and ('b','a') are counted as different itemsets.", "Using `>` instead of `>=` for the minimum support or confidence.", "Computing lift with the support of A instead of B.", "Pruning with the wrong subset size: the check is on the k-subsets of a (k+1)-candidate."],
},
{
  id: "kalman-step", t: "One Kalman Filter Step", d: "Medium", p: "kalman",
  q: "Write `kalman_step(x, P, z, F, H, Q, R)` that performs one **predict and update** cycle of the Kalman filter and returns the new `(x, P)`.\n\n- `x` is the state mean (length n) and `P` the n×n state covariance.\n- `F` (n×n) is the transition matrix, `Q` (n×n) the process noise covariance.\n- `z` is the measurement (length m), `H` (m×n) the measurement matrix and `R` (m×m) the measurement noise covariance.\n\nPredict: $\\mathbf x^-=\\mathbf F\\mathbf x$, $\\mathbf P^-=\\mathbf F\\mathbf P\\mathbf F^\\top+\\mathbf Q$. Update: $\\mathbf S=\\mathbf H\\mathbf P^-\\mathbf H^\\top+\\mathbf R$, $\\mathbf K=\\mathbf P^-\\mathbf H^\\top\\mathbf S^{-1}$, $\\mathbf x=\\mathbf x^-+\\mathbf K(\\mathbf z-\\mathbf H\\mathbf x^-)$, $\\mathbf P=(\\mathbf I-\\mathbf K\\mathbf H)\\mathbf P^-$.\n\nReturn `x` as a length-n array and `P` as an n×n array. Do not form `S` inverse explicitly.",
  ex: [["x = [10], P = [[4]], z = [12], F = H = [[1]], Q = [[0]], R = [[1]]", "x = [11.6], P = [[0.8]]", "Gain $K=4/(4+1)=0.8$, so the estimate moves 80% of the way from 10 to 12, and the variance shrinks from 4 to 0.8."]],
  c: ["all inputs are NumPy-compatible arrays with consistent shapes", "`S` is invertible (R positive definite)"],
  hint: "The gain solves $\\mathbf K\\mathbf S=\\mathbf P^-\\mathbf H^\\top$. Since `S` is symmetric, `K = np.linalg.solve(S, H @ P_pred.T).T` gives it without an inverse (or `P_pred @ H.T @ inv(S)` in the plain version).",
  brute: {
    name: "Textbook formulas with an explicit inverse",
    idea: "Translate the equations literally, including `np.linalg.inv(S)` and the short form $\\mathbf P=(\\mathbf I-\\mathbf K\\mathbf H)\\mathbf P^-$.",
    code: R`
import numpy as np

def kalman_step(x, P, z, F, H, Q, R):
    x = np.asarray(x, float); P = np.asarray(P, float); z = np.asarray(z, float)
    F, H, Q, R = (np.asarray(a, float) for a in (F, H, Q, R))
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_new = x_pred + K @ (z - H @ x_pred)
    P_new = (np.eye(len(x)) - K @ H) @ P_pred
    return x_new, P_new`,
    tc: "O(n³ + m³)", tw: "Matrix products and an m×m inverse.", sc: "O(n²)", sw: "Covariance matrices.",
    bad: "Correct, but an explicit inverse is less accurate than a solve, and the short covariance update $(\\mathbf I-\\mathbf K\\mathbf H)\\mathbf P^-$ can lose symmetry and positive-definiteness through rounding errors over many steps.",
  },
  opt: {
    name: "Solve instead of invert, Joseph-form covariance",
    idea: "Compute the gain by solving a linear system, and update the covariance with the Joseph form $(\\mathbf I-\\mathbf K\\mathbf H)\\mathbf P^-(\\mathbf I-\\mathbf K\\mathbf H)^\\top+\\mathbf K\\mathbf R\\mathbf K^\\top$, which is mathematically equal for the optimal gain but stays symmetric and positive semi-definite.",
    steps: ["Predict: `x_pred = F @ x`, `P_pred = F @ P @ F.T + Q`.", "`S = H @ P_pred @ H.T + R`; `K = np.linalg.solve(S, H @ P_pred).T` (uses the symmetry of S and P_pred).", "`x_new = x_pred + K @ (z - H @ x_pred)`.", "Joseph form for `P_new`, then symmetrise: `(P_new + P_new.T) / 2`."],
    code: R`
import numpy as np

def kalman_step(x, P, z, F, H, Q, R):
    x = np.asarray(x, float); P = np.asarray(P, float); z = np.asarray(z, float)
    F, H, Q, R = (np.asarray(a, float) for a in (F, H, Q, R))
    x_pred = F @ x                                             # predict the mean
    P_pred = F @ P @ F.T + Q                                   # the covariance grows by Q
    S = H @ P_pred @ H.T + R                                   # innovation covariance
    K = np.linalg.solve(S, H @ P_pred).T                       # gain: K S = P_pred H^T, no inverse
    x_new = x_pred + K @ (z - H @ x_pred)                      # correct with the innovation
    I_KH = np.eye(len(x)) - K @ H
    P_new = I_KH @ P_pred @ I_KH.T + K @ R @ K.T               # Joseph form
    return x_new, (P_new + P_new.T) / 2`,
    tc: "O(n³ + m³)", tw: "Matrix products and one m×m solve.", sc: "O(n²)", sw: "Covariance matrices.",
    viz: "cml2_kalman",
  },
  run: R`
x, P = kalman_step([10.0], [[4.0]], [12.0], [[1.0]], [[1.0]], [[0.0]], [[1.0]])
print(np.round(x, 3).tolist(), np.round(P, 3).tolist())
F = np.array([[1.0, 1.0], [0.0, 1.0]]); H = np.array([[1.0, 0.0]])
Q = 0.05 * np.outer([0.5, 1.0], [0.5, 1.0]); R = np.array([[4.0]])
x, P = kalman_step([0.0, 1.0], np.diag([4.0, 1.0]), [1.5], F, H, Q, R)
print(np.round(x, 3).tolist())
print(np.round(P, 3).tolist())
print(bool(np.allclose(P, P.T)), bool((np.linalg.eigvalsh(P) > 0).all()))
x, P = kalman_step([0.0, 1.0], np.diag([4.0, 1.0]), [1.0], F, H, Q, R)
print(np.round(x, 3).tolist())
x, P = kalman_step([5.0], [[1.0]], [100.0], [[1.0]], [[1.0]], [[0.0]], [[1e6]])
print(np.round(x, 3).tolist(), np.round(P, 3).tolist())`,
  out: R`
[11.6] [[0.8]]
[1.278, 1.057]
[[2.225, 0.455], [0.455, 0.933]]
True True
[1.0, 1.0]
[5.0] [[1.0]]`,
  dryOn: "scalar case: x = 10, P = 4, z = 12, F = H = 1, Q = 0, R = 1",
  dry: { h: ["quantity", "value"], r: [["predict x⁻", "1 · 10 = 10"], ["predict P⁻", "1·4·1 + 0 = 4"], ["S = H P⁻ Hᵀ + R", "4 + 1 = 5"], ["K = P⁻ Hᵀ / S", "4 / 5 = 0.8"], ["x = x⁻ + K (z − H x⁻)", "10 + 0.8 · 2 = **11.6**"], ["P = (1 − K H) P⁻", "0.2 · 4 = **0.8**"]] },
  expl: [
    ["x_pred = F @ x; P_pred = F @ P @ F.T + Q", "Push the Gaussian belief through the motion model; the covariance grows by the process noise."],
    ["S = H @ P_pred @ H.T + R", "How much we expect the measurement to scatter around its prediction."],
    ["np.linalg.solve(S, H @ P_pred).T", "Solves K S = P H^T by transposing, using the fact that S and P are symmetric."],
    ["x_pred + K @ (z - H @ x_pred)", "Move the prediction towards the measurement by gain times innovation."],
    ["I_KH @ P_pred @ I_KH.T + K @ R @ K.T", "Joseph form: equal to (I−KH)P⁻ for the optimal K, but robust to rounding."],
  ],
  take: "**Kalman = predict (uncertainty grows), then update (uncertainty shrinks)**, with the gain K = P⁻Hᵀ S⁻¹ deciding how much to trust the measurement versus the prediction.",
  mist: ["Skipping the predict step (or applying F to P as `F @ P` instead of `F @ P @ F.T`).", "Using `H.T @ P @ H` or other wrong orderings: the shapes should be (m×n)(n×n)(n×m).", "Forgetting that the update shrinks P, so the filter keeps a stale, too-large covariance.", "Passing standard deviations instead of variances for Q and R."],
},
]);
