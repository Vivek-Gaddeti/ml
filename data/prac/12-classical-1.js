/* Practice: Classical ML, part 1 (Fisher LDA, EM for a Gaussian mixture, Viterbi, Metropolis-Hastings)
   Each problem: q, ex, hint, brute, opt, run + out (tests), dry, expl, take, mist. */
PRAC.add("Classical ML", [
{
  id: "fisher-lda-direction", t: "Fisher's LDA Direction", d: "Easy", p: "lda",
  q: "Write `fisher_direction(X, y)` that returns Fisher's linear discriminant direction for a two-class problem: the **unit vector** $\\mathbf w\\propto\\mathbf S_W^{-1}(\\boldsymbol\\mu_1-\\boldsymbol\\mu_0)$.\n\n$$\\mathbf S_W=\\sum_{k\\in\\{0,1\\}}\\sum_{\\mathbf x\\in\\text{class }k}(\\mathbf x-\\boldsymbol\\mu_k)(\\mathbf x-\\boldsymbol\\mu_k)^\\top$$\n\n`X` is an (n, d) array and `y` holds the labels 0 and 1. Return a length-d array of norm 1, signed so that it points from class 0 towards class 1 (that is, `w @ (mu1 - mu0) > 0`). Do not form an explicit matrix inverse.",
  ex: [["X0 = [[-1,0],[1,0],[0,-1],[0,1]], X1 = X0 + [3, 4]", "[0.6, 0.8]", "Both classes are round (S_W is a multiple of the identity), so w is just the direction between the means, (3, 4) / 5."],
    ["two flat clouds, spread ±2 in x and ±0.5 in y, class 1 shifted by (1, 1)", "[0.062, 0.998]", "x is very noisy inside each class, so LDA mostly ignores it and uses y, even though the means differ equally in both."]],
  c: ["`X` has shape (n, d), `y` contains exactly the labels 0 and 1, both present", "S_W is invertible"],
  hint: "Compute the two class means, centre each class by its own mean, and add the two $\\mathbf X_c^\\top\\mathbf X_c$ products. Then `np.linalg.solve(Sw, m1 - m0)` and divide by the norm.",
  brute: {
    name: "Plain Python loops",
    idea: "Build $\\mathbf S_W$ entry by entry with loops over the points and over pairs of coordinates, then invert it explicitly.",
    code: R`
import numpy as np

def fisher_direction(X, y):
    X = np.asarray(X, float); y = np.asarray(y)
    n, d = X.shape
    mu = []
    for k in (0, 1):
        pts = [X[i] for i in range(n) if y[i] == k]
        mu.append(sum(pts) / len(pts))
    Sw = np.zeros((d, d))
    for i in range(n):
        diff = X[i] - mu[int(y[i])]
        for a in range(d):
            for b in range(d):
                Sw[a, b] += diff[a] * diff[b]
    w = np.linalg.inv(Sw) @ (mu[1] - mu[0])
    return w / np.linalg.norm(w)`,
    tc: "O(n·d² + d³)", tw: "Two nested coordinate loops per point, then an inversion.", sc: "O(d²)", sw: "One scatter matrix.",
    bad: "Correct, but the scatter matrix is a sum of outer products that matrix multiplication computes in one call, and `inv(Sw) @ v` is slower and less accurate than `solve(Sw, v)`.",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "Centre every class by its own mean, so the scatter of a class is $\\mathbf X_c^\\top\\mathbf X_c$. Sum the two, and solve one linear system for the direction.",
    steps: ["`m0 = X[y == 0].mean(axis=0)`, same for `m1`.", "`Xc0 = X[y == 0] - m0`, `Xc1 = X[y == 1] - m1`; `Sw = Xc0.T @ Xc0 + Xc1.T @ Xc1`.", "`w = np.linalg.solve(Sw, m1 - m0)`, then divide by `np.linalg.norm(w)`."],
    code: R`
import numpy as np

def fisher_direction(X, y):
    X = np.asarray(X, float); y = np.asarray(y)
    X0, X1 = X[y == 0], X[y == 1]
    m0, m1 = X0.mean(axis=0), X1.mean(axis=0)
    Sw = (X0 - m0).T @ (X0 - m0) + (X1 - m1).T @ (X1 - m1)     # within-class scatter
    w = np.linalg.solve(Sw, m1 - m0)                           # Sw^-1 (m1 - m0) without an inverse
    return w / np.linalg.norm(w)`,
    tc: "O(n·d² + d³)", tw: "One matrix product per class plus a d×d solve.", sc: "O(d²)", sw: "The scatter matrix.",
    viz: "cml1_lda",
  },
  run: R`
X0 = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], float)
X = np.vstack([X0, X0 + [3, 4]]); y = [0] * 4 + [1] * 4
print(np.round(fisher_direction(X, y), 3).tolist())
F0 = np.array([[-2, -0.5], [2, -0.5], [-2, 0.5], [2, 0.5]])
X = np.vstack([F0, F0 + [1, 1]])
w = fisher_direction(X, y)
print(np.round(w, 3).tolist(), round(float(np.linalg.norm(w)), 3))
rng = np.random.default_rng(0)
A = rng.multivariate_normal([0, 0], [[3, 1.2], [1.2, 1]], 60)
B = rng.multivariate_normal([1.5, 2.5], [[3, 1.2], [1.2, 1]], 60)
X = np.vstack([A, B]); y = [0] * 60 + [1] * 60
w = fisher_direction(X, y)
print(np.round(w, 3).tolist(), bool(w @ (B.mean(axis=0) - A.mean(axis=0)) > 0))
print(np.round(fisher_direction(X[:, ::-1], y), 3).tolist())`,
  out: R`
[0.6, 0.8]
[0.062, 0.998] 1.0
[-0.162, 0.987] True
[0.987, -0.162]`,
  dryOn: "F0 = [[-2,-0.5],[2,-0.5],[-2,0.5],[2,0.5]], class 1 = F0 + [1, 1]",
  dry: { h: ["quantity", "value"], r: [["means", "μ0 = (0, 0), μ1 = (1, 1)"], ["class scatter (each)", "diag(16, 1)"], ["S_W", "diag(32, 2)"], ["S_W⁻¹(μ1 − μ0)", "(1/32, 1/2) = (0.031, 0.5)"], ["normalise", "(0.031, 0.5) / 0.501 = **(0.062, 0.998)**"]] },
  expl: [
    ["X0, X1 = X[y == 0], X[y == 1]", "Boolean masks split the rows by class."],
    ["(X0 - m0).T @ (X0 - m0)", "Centred data times itself: the sum of outer products (x − μ)(x − μ)ᵀ over the class, i.e. its scatter matrix."],
    ["np.linalg.solve(Sw, m1 - m0)", "Solves Sw · w = μ1 − μ0, which is S_W⁻¹(μ1 − μ0), without ever building the inverse."],
    ["w / np.linalg.norm(w)", "Only the direction matters, so rescale to length 1."],
  ],
  take: "**Fisher's LDA is one linear solve**: the line between the class means, corrected by the inverse within-class scatter so that noisy directions count less.",
  mist: ["Using the total covariance of all the data instead of the **within-class** scatter (centre each class by its own mean).", "Forgetting the sign convention: `solve` returns a direction whose sign depends on the order of the means.", "Calling `np.linalg.inv` instead of `solve`, and not checking that S_W is invertible when d is close to n."],
},
{
  id: "gmm-em-one-iteration", t: "One EM Iteration for a 1-D Gaussian Mixture", d: "Medium", p: "em",
  q: "Write `em_step(x, pi, mu, var)` that performs **one** EM iteration for a one-dimensional Gaussian mixture with K components.\n\n- E-step: for every point and component compute the responsibility $r_{ik}\\propto\\pi_k\\,\\mathcal N(x_i\\mid\\mu_k,\\sigma^2_k)$, normalised over $k$.\n- M-step: $N_k=\\sum_i r_{ik}$, $\\pi_k=N_k/n$, $\\mu_k=\\sum_i r_{ik}x_i/N_k$, $\\sigma_k^2=\\sum_i r_{ik}(x_i-\\mu_k^{\\text{new}})^2/N_k$.\n\nReturn `(new_pi, new_mu, new_var, loglik)` as NumPy arrays plus a float, where `loglik` is the log-likelihood $\\sum_i\\log\\sum_k\\pi_k\\mathcal N(x_i\\mid\\mu_k,\\sigma^2_k)$ of the data under the **input** parameters (the ones the E-step used).",
  ex: [["x = [0, 1, 9, 10], pi = [0.5, 0.5], mu = [0, 10], var = [1, 1]", "mu ≈ [0.5, 9.5]", "Points 0 and 1 belong (almost entirely) to component 0, points 9 and 10 to component 1; the means move to 0.5 and 9.5."]],
  c: ["`x` has n ≥ 2 points, K ≥ 1 components", "All variances are positive and the weights sum to 1", "Every component keeps a non-zero total responsibility in the tests"],
  hint: "Build an (n, K) matrix of weighted densities with broadcasting: `x[:, None]` against `mu[None, :]`. Row sums are the per-point likelihoods; dividing by them gives the responsibilities. The M-step is column sums of `r`, `r * x`, and `r * (x - new_mu)**2`.",
  brute: {
    name: "Plain Python loops",
    idea: "Loop over points to get responsibilities, then loop over components to accumulate the weighted sums.",
    code: R`
import math
import numpy as np

def em_step(x, pi, mu, var):
    n, K = len(x), len(pi)
    r = [[0.0] * K for _ in range(n)]
    ll = 0.0
    for i in range(n):
        dens = [pi[k] * math.exp(-0.5 * (x[i] - mu[k]) ** 2 / var[k]) / math.sqrt(2 * math.pi * var[k]) for k in range(K)]
        tot = sum(dens)
        ll += math.log(tot)
        for k in range(K):
            r[i][k] = dens[k] / tot
    new_pi, new_mu, new_var = [], [], []
    for k in range(K):
        Nk = sum(r[i][k] for i in range(n))
        m = sum(r[i][k] * x[i] for i in range(n)) / Nk
        v = sum(r[i][k] * (x[i] - m) ** 2 for i in range(n)) / Nk
        new_pi.append(Nk / n); new_mu.append(m); new_var.append(v)
    return np.array(new_pi), np.array(new_mu), np.array(new_var), ll`,
    tc: "O(n·K)", tw: "Python loops over points and components.", sc: "O(n·K)", sw: "The responsibility table.",
    bad: "Correct but slow for large n: every density is computed in the interpreter. Also, summing raw densities underflows for points far from every component (see the log-sum-exp trick).",
  },
  opt: {
    name: "Vectorized with NumPy",
    idea: "One (n, K) array of weighted densities from broadcasting; the E-step is a row normalisation and the M-step is column sums.",
    steps: ["`dens = pi * np.exp(-0.5 * (x[:, None] - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)` has shape (n, K).", "`tot = dens.sum(axis=1)`; the log-likelihood is `np.log(tot).sum()`; `r = dens / tot[:, None]`.", "`Nk = r.sum(axis=0)`; `new_mu = (r * x[:, None]).sum(axis=0) / Nk`; `new_var = (r * (x[:, None] - new_mu) ** 2).sum(axis=0) / Nk`."],
    code: R`
import numpy as np

def em_step(x, pi, mu, var):
    x, pi, mu, var = (np.asarray(a, float) for a in (x, pi, mu, var))
    dens = pi * np.exp(-0.5 * (x[:, None] - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)   # (n, K)
    tot = dens.sum(axis=1)
    ll = float(np.log(tot).sum())                                  # log-likelihood under the INPUT parameters
    r = dens / tot[:, None]                                        # E-step: responsibilities
    Nk = r.sum(axis=0)                                             # M-step
    new_mu = (r * x[:, None]).sum(axis=0) / Nk
    new_var = (r * (x[:, None] - new_mu) ** 2).sum(axis=0) / Nk
    return Nk / len(x), new_mu, new_var, ll`,
    tc: "O(n·K)", tw: "A few (n, K) array operations.", sc: "O(n·K)", sw: "The responsibility matrix.",
    viz: "cml1_gmm",
  },
  run: R`
p, m, v, ll = em_step([0, 1, 9, 10], [0.5, 0.5], [0, 10], [1, 1])
print(np.round(p, 3).tolist(), np.round(m, 3).tolist(), np.round(v, 3).tolist(), round(ll, 3))
x = [-1.0, 0.0, 1.0, 4.0, 5.0, 6.0]
p, m, v, ll = em_step(x, [0.7, 0.3], [0.0, 5.0], [2.0, 1.0])
print(np.round(p, 3).tolist(), np.round(m, 3).tolist(), np.round(v, 3).tolist(), round(ll, 3))
lls = []
for _ in range(6):
    p, m, v, ll = em_step(x, p, m, v); lls.append(round(ll, 3))
print(lls, bool(all(b >= a - 1e-9 for a, b in zip(lls, lls[1:]))))
rng = np.random.default_rng(0)
z = np.r_[rng.normal(0, 1, 200), rng.normal(6, 2, 100)]
p, m, v = np.array([0.5, 0.5]), np.array([1.0, 4.0]), np.array([1.0, 1.0])
for _ in range(50):
    p, m, v, ll = em_step(z, p, m, v)
print(np.round(p, 2).tolist(), np.round(m, 2).tolist(), np.round(np.sqrt(v), 2).tolist())`,
  out: R`
[0.5, 0.5] [0.5, 9.5] [0.25, 0.25] -7.448
[0.508, 0.492] [0.068, 5.016] [0.93, 0.663] -12.683
[-11.539, -11.456, -11.456, -11.456, -11.456, -11.456] True
[0.65, 0.35] [-0.02, 5.48] [0.95, 2.41]`,
  dryOn: "x = [0, 1, 9, 10], pi = [0.5, 0.5], mu = [0, 10], var = [1, 1]",
  dry: { h: ["point", "r (comp 0)", "r (comp 1)"], r: [["0", "≈ 1.000", "≈ 0.000"], ["1", "≈ 1.000", "≈ 0.000"], ["9", "≈ 0.000", "≈ 1.000"], ["10", "≈ 0.000", "≈ 1.000"], ["new π, μ", "π = [0.5, 0.5], μ ≈ **[0.5, 9.5]**"], ["new σ²", "each ≈ **0.25** (points ±0.5 from the mean)"]] },
  expl: [
    ["dens = pi * np.exp(-0.5 * (x[:, None] - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)", "Broadcasting (n,1) against (K,) gives an (n, K) table of π_k · N(x_i | μ_k, σ_k²)."],
    ["tot = dens.sum(axis=1)", "The mixture density at each point; its log, summed, is the log-likelihood (of the input parameters)."],
    ["r = dens / tot[:, None]", "Bayes' rule: normalise each row so a point's responsibilities sum to 1."],
    ["Nk = r.sum(axis=0)", "The effective number of points per component."],
    ["new_mu = (r * x[:, None]).sum(axis=0) / Nk", "A weighted average of the points, with the responsibilities as weights."],
    ["new_var = (r * (x[:, None] - new_mu) ** 2).sum(axis=0) / Nk", "A weighted variance around the **new** mean."],
  ],
  take: "**EM for a mixture is two array lines**: normalise the weighted densities to get responsibilities (E), then take weighted means, variances and shares (M). The log-likelihood can only go up.",
  mist: ["Using the **old** means when computing the new variances; the M-step uses the freshly updated means.", "Dividing by `n` instead of `Nk` for the means and variances.", "Forgetting `pi` in the E-step, which silently treats all components as equally common.", "Not guarding against a component with `Nk` near 0, which makes the variance blow up."],
},
{
  id: "viterbi-decoding", t: "Viterbi Decoding for an HMM", d: "Medium", p: "hmm",
  q: "Write `viterbi(obs, pi, A, B)` for a hidden Markov model with K states.\n\n- `obs`: list of observation indices (length T)\n- `pi`: start probabilities, shape (K,)\n- `A`: transition matrix, `A[i, j]` = P(next state j | state i)\n- `B`: emission matrix, `B[j, o]` = P(observation o | state j)\n\nReturn `(path, prob)`: the most likely sequence of hidden states (a list of ints) and its joint probability $P(\\mathbf z,\\mathbf x)$. Assume there are no ties.",
  ex: [["obs = [0, 1, 2] (walk, shop, clean), pi = [0.6, 0.4], A = [[0.7, 0.3], [0.4, 0.6]], B = [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]", "([1, 0, 0], 0.01344)", "States 0 = Rainy, 1 = Sunny: the best story is Sunny, Rainy, Rainy."]],
  c: ["1 ≤ T ≤ 8 in the tests (the brute-force version is exponential)", "Probabilities are in [0, 1]; rows of A and B sum to 1", "No ties between best paths"],
  hint: "Keep `delta[t, j]` = probability of the best path ending in state j at time t and `psi[t, j]` = the predecessor that achieved it. For t ≥ 1, `delta[t-1][:, None] * A` is a K×K table of (from, to) scores; take the max and argmax over `axis=0`.",
  brute: {
    name: "Enumerate every path",
    idea: "Try all $K^T$ state sequences, compute the joint probability of each with the product formula, and keep the best.",
    code: R`
import itertools
import numpy as np

def viterbi(obs, pi, A, B):
    pi, A, B = np.asarray(pi, float), np.asarray(A, float), np.asarray(B, float)
    K, T = len(pi), len(obs)
    best_p, best_path = -1.0, None
    for z in itertools.product(range(K), repeat=T):
        p = pi[z[0]] * B[z[0], obs[0]]
        for t in range(1, T):
            p *= A[z[t - 1], z[t]] * B[z[t], obs[t]]
        if p > best_p:
            best_p, best_path = p, list(z)
    return best_path, float(best_p)`,
    tc: "O(K^T · T)", tw: "Exponential: 2 states and 100 days is 2^100 paths.", sc: "O(T)", sw: "One path at a time.",
    bad: "Correct but hopeless beyond tiny T. The best path to a state only depends on the best paths to the previous states, so most of this work is repeated.",
  },
  opt: {
    name: "Dynamic programming",
    idea: "For each time and state keep only the best path ending there (delta) and where it came from (psi). Then backtrack from the best final state. Cost O(T·K²).",
    steps: ["`delta[0] = pi * B[:, obs[0]]`.", "For t = 1..T−1: `cand = delta[t-1][:, None] * A`; `psi[t] = cand.argmax(axis=0)`; `delta[t] = cand.max(axis=0) * B[:, obs[t]]`.", "Start at `delta[-1].argmax()` and follow `psi` backwards; reverse the path."],
    code: R`
import numpy as np

def viterbi(obs, pi, A, B):
    pi, A, B = np.asarray(pi, float), np.asarray(A, float), np.asarray(B, float)
    T, K = len(obs), len(pi)
    delta = np.zeros((T, K)); psi = np.zeros((T, K), dtype=int)
    delta[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        cand = delta[t - 1][:, None] * A                      # cand[i, j]: best path to i, then step i -> j
        psi[t] = cand.argmax(axis=0)                          # best predecessor of each state j
        delta[t] = cand.max(axis=0) * B[:, obs[t]]
    path = [int(delta[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(psi[t][path[-1]]))
    return path[::-1], float(delta[-1].max())`,
    tc: "O(T·K²)", tw: "A K×K table per time step.", sc: "O(T·K)", sw: "The delta and psi tables.",
    viz: "cml1_viterbi",
  },
  run: R`
pi = [0.6, 0.4]; A = [[0.7, 0.3], [0.4, 0.6]]; B = [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]
path, p = viterbi([0, 1, 2], pi, A, B)
print(path, round(p, 5))
path, p = viterbi([2, 2, 2, 0, 0], pi, A, B)
print(path, round(p, 6))
path, p = viterbi([1], pi, A, B)
print(path, round(p, 3))
rng = np.random.default_rng(3)
pi3 = rng.dirichlet(np.ones(3)); A3 = rng.dirichlet(np.ones(3), 3); B3 = rng.dirichlet(np.ones(4), 3)
obs = rng.integers(0, 4, 7).tolist()
path, p = viterbi(obs, pi3, A3, B3)
print(obs, path, f"{p:.3e}")`,
  out: R`
[1, 0, 0] 0.01344
[0, 0, 0, 1, 1] 0.002381
[0] 0.24
[0, 3, 2, 2, 0, 1, 0] [1, 2, 2, 2, 1, 2, 1] 2.791e-07`,
  dryOn: "obs = [0, 1, 2] (walk, shop, clean), Alice's HMM",
  dry: { h: ["day", "δ(Rainy)", "δ(Sunny)", "ψ (Rainy, Sunny)"], r: [["1  walk", "0.6·0.1 = 0.06", "0.4·0.6 = 0.24", "-"], ["2  shop", "max(0.042, 0.096)·0.4 = 0.0384", "max(0.018, 0.144)·0.3 = 0.0432", "(Sunny, Sunny)"], ["3  clean", "max(0.02688, 0.01728)·0.5 = 0.01344", "max(0.01152, 0.02592)·0.1 = 0.002592", "(Rainy, Sunny)"], ["backtrack", "best end: Rainy (0.01344)", "← Rainy ← Sunny", "**path: Sunny, Rainy, Rainy**"]] },
  expl: [
    ["delta[0] = pi * B[:, obs[0]]", "Day 1: start probability times the chance of seeing the first observation from each state."],
    ["cand = delta[t - 1][:, None] * A", "A K×K table: row i = previous state, column j = next state. Entry = best score into i times the transition i → j."],
    ["psi[t] = cand.argmax(axis=0)", "For each next state j, which previous state i gave the best score: the backpointer."],
    ["delta[t] = cand.max(axis=0) * B[:, obs[t]]", "Keep only the best score into each j, then multiply by the emission probability."],
    ["path.append(int(psi[t][path[-1]]))", "Walking back: the best predecessor of the state we are in."],
  ],
  take: "**Viterbi is the forward algorithm with max instead of sum, plus backpointers.** It turns an exponential search over K^T paths into T small K×K table lookups.",
  mist: ["Taking the argmax of each δ_t separately instead of following backpointers from the last time step.", "Multiplying plain probabilities over long sequences: they underflow, so use logs for real data.", "Indexing the transition table the wrong way round (`A[j, i]` instead of `A[i, j]`)."],
},
{
  id: "metropolis-hastings-sampler", t: "Metropolis-Hastings Sampler", d: "Hard", p: "mcmc",
  q: "Write `metropolis(logp, x0, step, n, seed)` that draws `n` samples with a random-walk Metropolis sampler.\n\n- `logp(x)` returns the **log** of an unnormalised density at the float `x`.\n- At each iteration propose `xp = x + step * eps` and accept it if `np.log(u) < logp(xp) - logp(x)`.\n- Use `rng = np.random.default_rng(seed)`, draw **all** the noise up front in this order: `eps = rng.standard_normal(n)` then `u = rng.random(n)`; iteration `i` uses `eps[i]` and `u[i]`.\n- A rejected proposal repeats the current state.\n\nReturn `(samples, acceptance_rate)`: `samples` is an array of length n holding the state after each iteration (the starting point `x0` is not included).",
  ex: [["logp = lambda x: -0.5 * x**2 (standard normal), x0 = 0, step = 2.4, n = 4000", "mean ≈ 0, std ≈ 1, acceptance ≈ 0.44", "A good step size for a 1-D target accepts roughly 40 to 50 percent of the proposals."]],
  c: ["`logp` returns a finite float for the points visited", "n ≥ 1, step > 0"],
  hint: "Keep the current log-density in a variable instead of recomputing `logp(x)` every time: it only changes when a proposal is accepted. Compare logs, never exponentiate the ratio.",
  brute: {
    name: "Direct ratio, recompute everything",
    idea: "Recompute the current log-density each iteration and compare the probability ratio `min(1, exp(Δ))` against `u`. Works for moderate targets but wastes a target evaluation per step and can overflow.",
    code: R`
import math
import numpy as np

def metropolis(logp, x0, step, n, seed):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n); u = rng.random(n)
    x = float(x0); out = np.empty(n); acc = 0
    for i in range(n):
        xp = x + step * eps[i]
        d = logp(xp) - logp(x)                                # logp(x) is evaluated again every iteration
        ratio = 1.0 if d >= 0 else math.exp(d)
        if u[i] < ratio:
            x = xp; acc += 1
        out[i] = x
    return out, acc / n`,
    tc: "O(n) target evaluations ×2", tw: "Two calls to logp per iteration.", sc: "O(n)", sw: "The samples.",
    bad: "Same samples, but it evaluates the (often expensive) target twice per step, and exponentiating a large positive difference can overflow. Comparing logs avoids both problems.",
  },
  opt: {
    name: "Log-space with a cached density",
    idea: "Remember the log-density of the current state. Each iteration costs one new target evaluation, and the accept test is `log u < Δ`, which cannot overflow or underflow.",
    steps: ["Draw `eps` then `u` up front.", "Keep `lp = logp(x)`. For each i: `xp = x + step*eps[i]`, `lpp = logp(xp)`.", "If `np.log(u[i]) < lpp - lp`: set `x, lp = xp, lpp` and count the acceptance. Store `x`."],
    code: R`
import numpy as np

def metropolis(logp, x0, step, n, seed):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n); u = rng.random(n)
    x = float(x0); lp = logp(x)                               # cache the log-density of the current state
    out = np.empty(n); acc = 0
    for i in range(n):
        xp = x + step * eps[i]                                # random-walk proposal
        lpp = logp(xp)
        if np.log(u[i]) < lpp - lp:                           # accept with probability min(1, p'/p), in log space
            x, lp = xp, lpp; acc += 1
        out[i] = x                                            # a rejection repeats x
    return out, acc / n`,
    tc: "O(n) target evaluations", tw: "One logp call per iteration.", sc: "O(n)", sw: "The samples and the pre-drawn noise.",
    viz: "cml1_mcmc",
  },
  run: R`
s, a = metropolis(lambda x: -0.5 * x**2, 0.0, 2.4, 4000, 0)
print(round(float(s.mean()), 3), round(float(s.std()), 3), round(a, 3))
print(np.round(s[:5], 3).tolist())
bimodal = lambda x: float(np.log(0.5 * np.exp(-0.5 * ((x + 2) / 0.7) ** 2) / 0.7 + 0.5 * np.exp(-0.5 * (x - 2) ** 2)))
s, a = metropolis(bimodal, -6.0, 3.0, 6000, 1)
s = s[500:]
print(round(float(s.mean()), 3), round(float(s.std()), 3), round(float((s > 0).mean()), 3), round(a, 3))
s, a = metropolis(lambda x: -0.5 * x**2, 0.0, 0.05, 1000, 2)
print(round(a, 3), round(float(s.std()), 3))
s, a = metropolis(lambda x: -0.5 * (x - 100) ** 2, 0.0, 1.0, 3000, 3)
print(round(float(s[-500:].mean()), 2), round(a, 3))`,
  out: R`
-0.068 0.98 0.433
[0.302, -0.015, -0.015, 0.236, -1.049]
-0.083 2.18 0.472 0.463
0.987 0.559
100.04 0.68`,
  dryOn: "logp(x) = -x²/2, x0 = 0, first two iterations with eps = [0.8, -1.5] (step 1), u = [0.5, 0.3]",
  dry: { setup: "Start at x = 0, so lp = 0.", steps: [["i = 0: xp = 0 + 0.8 = 0.8, lpp = −0.32", "Δ = lpp − lp = −0.32; log u = log 0.5 = −0.693 < −0.32 → **accept**, x = 0.8, lp = −0.32"], ["i = 1: xp = 0.8 − 1.5 = −0.7, lpp = −0.245", "Δ = −0.245 − (−0.32) = +0.075 > 0 → log u = −1.204 < 0.075 → **accept** (uphill moves always are), x = −0.7"]], result: "Samples so far: [0.8, −0.7]; every uphill move is accepted, downhill ones only sometimes." },
  expl: [
    ["eps = rng.standard_normal(n); u = rng.random(n)", "Draw all the randomness first, in the order the problem fixes, so results are reproducible."],
    ["lp = logp(x)", "Cache the log-density of the current state; it only changes when we move."],
    ["np.log(u[i]) < lpp - lp", "Accept with probability min(1, p'/p): log u is negative, so uphill moves (Δ > 0) always pass; a downhill move of size Δ passes with probability e^Δ."],
    ["out[i] = x", "Store the state after the decision. A rejected move stores the old x again, which is essential for correct sampling."],
  ],
  take: "**Metropolis in four lines**: propose, compare log-densities, accept with probability min(1, ratio), and **keep the old state when you reject**. Work in logs and cache the current log-density.",
  mist: ["Dropping rejected iterations instead of repeating the old state: that biases the samples towards where the walker moved easily.", "Comparing densities instead of log-densities, so a high-dimensional target underflows to 0 or overflows.", "Using a step size that gives an acceptance rate near 0 or near 1, and then trusting the samples.", "Forgetting burn-in: the first samples reflect the starting point, not the target."],
},
]);
