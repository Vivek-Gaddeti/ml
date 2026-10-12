/* Classical ML, part 2: factorization, manifolds and other classics
   Outputs written as "?" are filled in by running the code: node tools/fill.js data/cml-2.js */
CML.push(
{
  id: "ica", group: "Factorization & Manifolds", title: "Independent Component Analysis (ICA)",
  sub: "Unmix signals that were recorded as blends, by finding the directions in which the data look as non-Gaussian (as independent) as possible.",
  tags: ["unsupervised", "signals"],
  blocks: [
    ["p", "**What this page is, part 2.** The first half of this page covered discriminants, kernels and probabilistic models. This half covers two more families. **Factorization and manifolds**: take a table of numbers and rewrite it as a product of simpler pieces (ICA, NMF, recommender factorization), or squash high-dimensional data into a flat picture while keeping its shape (t-SNE, UMAP, MDS, Isomap, spectral clustering). Then **other classics** that every ML practitioner meets sooner or later: AdaBoost, anomaly detection, association rules, the Kalman filter and time-series models."],
    ["p", "Imagine a party with two people talking at once and two microphones in the room. Each microphone records a **blend** of both voices, with different loudness for each. You want the two separate voices back, knowing nothing about how the room mixed them. This is the **cocktail party problem**, and **Independent Component Analysis (ICA)** solves it."],
    ["diagram", "cml2_ica", [], "ICA: from sources to microphones and back. After whitening only a rotation is left to find."],
    ["p", "Why is this possible at all? Because of one simple fact: **a blend of independent signals looks more Gaussian (bell-shaped) than the signals themselves.** That is the central limit theorem. A square wave has two flat levels, a sawtooth is uniformly spread; both are very un-bell-like. Add them together and you get something closer to a bell. So to unmix, **rotate until the result is as far from Gaussian as possible**. Try it below: slide the angle and watch the projected signals. The curve shows how non-Gaussian the signal along axis 1 is; it peaks exactly where the original sources reappear."],
    ["viz", "cml2_ica"],
    ["h", "The maths"],
    ["p", R`The model: $d$ observed signals $\mathbf x$ are an unknown linear mix of $d$ independent sources $\mathbf s$, with an unknown mixing matrix $\mathbf A$. We look for an unmixing matrix $\mathbf W\approx\mathbf A^{-1}$.`],
    ["math", R`\mathbf x=\mathbf A\,\mathbf s\qquad\Longrightarrow\qquad \hat{\mathbf s}=\mathbf W\,\mathbf x,\quad \text{with } \hat s_1,\dots,\hat s_d \text{ as independent as possible}`, "x: what the microphones record · s: the true voices · W: the unmixing matrix we learn"],
    ["p", R`**Step 1: whiten.** Centre the data, then rotate and rescale it with the eigen-decomposition of its covariance $\mathbf C=\mathbf E\mathbf\Lambda\mathbf E^\top$ (see [eigenvectors](math.html#eigen)) so that it becomes uncorrelated with unit variance:`],
    ["math", R`\mathbf z=\mathbf\Lambda^{-1/2}\mathbf E^\top(\mathbf x-\bar{\mathbf x}),\qquad \operatorname{Cov}(\mathbf z)=\mathbf I`, "this is exactly PCA, followed by dividing each component by its standard deviation"],
    ["p", R`Independent signals are always uncorrelated, so after whitening the sources still are. Uncorrelated plus unit variance means that the only freedom left is a **rotation**: $\hat{\mathbf s}=\mathbf R\mathbf z$ with $\mathbf R^\top\mathbf R=\mathbf I$. That is why the picture above has only one slider.`],
    ["p", R`**Step 2: choose a rotation by maximising non-Gaussianity.** Two common measures of how non-Gaussian a unit-variance signal $y$ is:`],
    ["list", [
      R`**Excess kurtosis** $\operatorname{kurt}(y)=\mathbb E[y^4]-3$. It is $0$ for a Gaussian, negative for flat-topped signals (a square wave has $-2$), positive for spiky, heavy-tailed ones. Maximise $|\operatorname{kurt}|$. Simple, but very sensitive to outliers.`,
      R`**Negentropy** $J(y)\approx\big(\mathbb E[G(y)]-\mathbb E[G(\nu)]\big)^2$ with $\nu\sim\mathcal N(0,1)$ and a smooth $G$ such as $G(u)=\log\cosh u$. More robust. This is what FastICA uses by default.`,
    ]],
    ["p", R`**FastICA** finds each unmixing direction $\mathbf w$ with a fixed-point iteration (a Newton-like step on the negentropy, with $g=G'=\tanh$):`],
    ["math", R`\mathbf w\;\leftarrow\;\mathbb E\big[\mathbf z\,g(\mathbf w^\top\mathbf z)\big]-\mathbb E\big[g'(\mathbf w^\top\mathbf z)\big]\,\mathbf w,\qquad \mathbf w\leftarrow \frac{\mathbf w}{\|\mathbf w\|}`, "repeat until w stops changing (up to sign); to find the next component, also subtract its projection on the earlier ones"],
    ["note", R`Tiny sanity check of the idea. Take $s_1,s_2$ independent and uniform on $[-\sqrt3,\sqrt3]$ (variance 1, excess kurtosis $-1.2$). Their sum divided by $\sqrt2$ also has variance 1, but its excess kurtosis is $-1.2/2=-0.6$, closer to the Gaussian's 0. Mixing moved the kurtosis towards zero, so unmixing means moving it away from zero again.`],
    ["warn", "ICA can recover the sources only up to three ambiguities: their **order** (which output is voice 1 is arbitrary), their **sign** (a flipped voice is equally independent) and their **scale** (we fixed the variance to 1). It also needs **at most one Gaussian source**: a Gaussian source is rotation-invariant, so its direction cannot be identified."],
    ["h", "From scratch in NumPy"],
    ["p", "A square wave and a sawtooth are mixed by a known matrix. We whiten, then run the one-unit FastICA rule for each component with deflation. At the end every recovered signal should correlate almost perfectly (in absolute value, because of the sign ambiguity) with exactly one true source."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
t = np.linspace(0, 8, 2000)
S = np.c_[np.sign(np.sin(3 * t)), (t * 1.3) % 2 - 1]      # sources: square wave and sawtooth, shape (n, 2)
S += 0.05 * rng.standard_normal(S.shape)
A = np.array([[1.0, 0.6], [0.5, 1.0]])                     # the (unknown to the algorithm) mixing matrix
X = S @ A.T                                                # what the two microphones record

Xc = X - X.mean(axis=0)                                    # step 1: whiten
d, E = np.linalg.eigh(np.cov(Xc.T))
Z = Xc @ E / np.sqrt(d)
print("covariance after whitening:", np.cov(Z.T).round(3).tolist())

W = np.zeros((2, 2))
for k in range(2):                                         # step 2: FastICA, one component at a time
    w = rng.standard_normal(2); w /= np.linalg.norm(w)
    for it in range(200):
        wz = Z @ w
        w_new = (Z * np.tanh(wz)[:, None]).mean(axis=0) - (1 - np.tanh(wz) ** 2).mean() * w
        for j in range(k):                                 # deflation: stay orthogonal to earlier components
            w_new -= (w_new @ W[j]) * W[j]
        w_new /= np.linalg.norm(w_new)
        converged = abs(abs(w_new @ w) - 1) < 1e-9
        w = w_new
        if converged: break
    W[k] = w
    print("component", k, "converged after", it + 1, "iterations")

S_hat = Z @ W.T
C = np.corrcoef(S_hat.T, S.T)[:2, 2:]
print("|correlation| of recovered (rows) with true sources (columns):")
print(np.abs(C).round(3))`, "covariance after whitening: [[1.0, -0.0], [-0.0, 1.0]]\ncomponent 0 converged after 3 iterations\ncomponent 1 converged after 2 iterations\n|correlation| of recovered (rows) with true sources (columns):\n[[0.072 1.   ]\n [0.997 0.017]]"],
    ["p", "A matrix that is (almost) a permutation of the identity is the success signal: each recovered signal matches one source, order and sign aside. Now the same with scikit-learn, next to PCA, which also whitens but cannot do the last rotation."],
    ["code", R`
import numpy as np
from sklearn.decomposition import FastICA, PCA

rng = np.random.default_rng(0)
t = np.linspace(0, 8, 2000)
S = np.c_[np.sign(np.sin(3 * t)), (t * 1.3) % 2 - 1] + 0.05 * rng.standard_normal((2000, 2))
X = S @ np.array([[1.0, 0.6], [0.5, 1.0]]).T

def best_match(R):                                         # for each recovered signal, its best |corr| with a true source
    return np.abs(np.corrcoef(R.T, S.T)[:2, 2:]).max(axis=1).round(3).tolist()

ica = FastICA(n_components=2, whiten="unit-variance", random_state=0)
S_ica = ica.fit_transform(X)
S_pca = PCA(n_components=2).fit_transform(X)
print("FastICA best matches:", best_match(S_ica))
print("PCA     best matches:", best_match(S_pca))
print("estimated mixing matrix has shape", ica.mixing_.shape, "; sources shape", S_ica.shape)`, "FastICA best matches: [1.0, 0.999]\nPCA     best matches: [0.865, 0.891]\nestimated mixing matrix has shape (2, 2) ; sources shape (2000, 2)"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Recovers the true independent sources, not just uncorrelated directions", "Needs non-Gaussian sources (at most one Gaussian) and a **linear instant** mixture"],
      ["Fast (FastICA converges in a few iterations)", "Order, sign and scale of the outputs are arbitrary"],
      ["Used in practice: EEG / fMRI artefact removal, audio separation, feature extraction", "Needs as many (or more) sensors than sources; echoes and delays break the model"],
      ["Few settings: number of components, nonlinearity `g`", "Result can change with the random start; check by running twice"],
    ]],
    ["tip", "ICA versus [PCA](ml.html#pca): PCA finds orthogonal directions of **largest variance** and stops at uncorrelated; ICA finds directions of **independence**, which are not necessarily orthogonal in the original space and are not ordered by variance. Use PCA to compress, ICA to separate."],
    ["danger", "Skipping the whitening step, or whitening and keeping fewer components than sources. Also, expecting the output channels to come back in a meaningful order or with the original sign: always identify components afterwards by looking at them (or by correlating with a known reference)."],
  ],
},
{
  id: "nmf", group: "Factorization & Manifolds", title: "Non-Negative Matrix Factorization (NMF)",
  sub: "Write a matrix of non-negative numbers as the product of two thin non-negative matrices, so every sample is an additive mix of a few recognisable parts.",
  tags: ["unsupervised", "topics"],
  blocks: [
    ["p", "Think of a pile of LEGO models. Each model is built by adding bricks; nobody ever removes a brick. **Non-negative matrix factorization (NMF)** looks at many finished objects (images, documents, spectrograms) and discovers the **bricks**, plus a recipe saying how many of each brick every object uses. Because nothing can be subtracted, the bricks come out as meaningful parts: strokes of a letter, parts of a face, the words of a topic."],
    ["diagram", "cml2_nmffactor", [], "NMF: the data matrix V is approximated by W (recipes) times H (parts), all entries non-negative."],
    ["p", R`Compare with [PCA](ml.html#pca) and [SVD](math.html#svd). They also write $\mathbf V\approx\mathbf W\mathbf H$, but allow negative numbers, so the components cancel each other out ("add this blob, subtract that one") and look like ghostly blends. NMF forbids that, which gives **additive, parts-based** features at the price of a slightly worse fit and no closed-form solution.`],
    ["viz", "cml2_nmf"],
    ["p", "The toy data are eight 4×4 images, each a mix of three hidden parts (a left column, a top row, a centre block). Slide the rank. With rank 3 the learned parts are exactly those blocks and the error collapses; beyond that nothing is gained."],
    ["h", "The maths"],
    ["p", R`Given $\mathbf V\in\mathbb R_{\ge 0}^{n\times p}$ and a chosen **rank** $k$ (much smaller than $n$ and $p$), find $\mathbf W\in\mathbb R_{\ge0}^{n\times k}$ and $\mathbf H\in\mathbb R_{\ge0}^{k\times p}$ that minimise the squared reconstruction error:`],
    ["math", R`\min_{\mathbf W\ge0,\;\mathbf H\ge0}\;\|\mathbf V-\mathbf W\mathbf H\|_F^2=\sum_{i,j}\Big(V_{ij}-\sum_{a=1}^{k}W_{ia}H_{aj}\Big)^2`, "row i of W: the recipe for sample i · row a of H: part a"],
    ["p", R`The problem is not convex in $(\mathbf W,\mathbf H)$ jointly, but it is convex in each one alone. **Lee and Seung's multiplicative updates** exploit this: multiply every entry by a ratio of non-negative numbers,`],
    ["math", R`H_{aj}\leftarrow H_{aj}\,\frac{(\mathbf W^\top\mathbf V)_{aj}}{(\mathbf W^\top\mathbf W\mathbf H)_{aj}},\qquad W_{ia}\leftarrow W_{ia}\,\frac{(\mathbf V\mathbf H^\top)_{ia}}{(\mathbf W\mathbf H\mathbf H^\top)_{ia}}`, "the error never increases; entries stay >= 0 for free because we only multiply"],
    ["p", R`Where does the ratio come from? Take the gradient with respect to $\mathbf H$: $\nabla=-2\mathbf W^\top\mathbf V+2\mathbf W^\top\mathbf W\mathbf H$. Ordinary gradient descent subtracts a step; choosing the step size $\eta_{aj}=H_{aj}/(\mathbf W^\top\mathbf W\mathbf H)_{aj}$ separately for every entry turns "subtract" into "multiply by the ratio of the negative part to the positive part of the gradient". Where the ratio is 1 the gradient is zero and the entry stops moving.`],
    ["note", R`Tiny example with $k=1$, $\mathbf V=\begin{pmatrix}2\\4\end{pmatrix}$ (a column), $\mathbf W=\begin{pmatrix}1\\1\end{pmatrix}$ and $\mathbf H=(1)$. The update is $H\leftarrow 1\cdot\frac{\mathbf W^\top\mathbf V}{\mathbf W^\top\mathbf W H}=\frac{6}{2}=3$, which is exactly the average of the column, the least-squares value, in one step.`],
    ["list", [
      "**Other losses**: the generalised KL divergence $\\sum(V\\log\\frac{V}{WH}-V+WH)$ suits count data (word counts, photon counts) and has its own multiplicative updates. In scikit-learn: `beta_loss=\"kullback-leibler\"` with `solver=\"mu\"`.",
      "**Not unique**: scaling a column of `W` up and the matching row of `H` down gives the same product, and the order of the parts is arbitrary. Many local optima: initialisation matters. `init=\"nndsvd\"` (based on the SVD) is deterministic and usually good.",
      "**Sparsity**: adding an L1 penalty on `H` or `W` gives even more parts-like solutions (see [regularisation](ml.html#regularization)).",
      "**Choosing k**: plot the error against the rank and look for the elbow (as in the picture above), or use cross-validation on held-out entries.",
    ]],
    ["h", "From scratch in NumPy"],
    ["p", "Two hidden topics generate a small document-by-word count table. Starting from random positive matrices, a few hundred multiplicative updates recover them."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
W_true = np.array([[3, 0], [2, 0], [0, 4], [0, 1], [2, 2], [1, 3]], float)   # 6 documents x 2 topics
H_true = np.array([[5, 3, 0, 0, 1], [0, 1, 4, 5, 2]], float)               # 2 topics x 5 words
V = W_true @ H_true                                                        # the data we observe

k = 2
W = rng.uniform(0.5, 1.5, (6, k)); H = rng.uniform(0.5, 1.5, (k, 5))
eps = 1e-9
for it in range(1, 301):
    H *= (W.T @ V) / (W.T @ W @ H + eps)                                   # multiplicative updates
    W *= (V @ H.T) / (W @ H @ H.T + eps)
    if it in (1, 10, 50, 300):
        print("iteration", it, "error", round(float(np.linalg.norm(V - W @ H)), 4))

print("all entries non-negative:", bool((W >= 0).all() and (H >= 0).all()))
print("recovered topics (rows of H), rescaled so the largest entry is 1:")
print((H / H.max(axis=1, keepdims=True)).round(2))
print("true topics, same scaling:")
print((H_true / H_true.max(axis=1, keepdims=True)).round(2))`, "iteration 1 error 22.4351\niteration 10 error 5.9909\niteration 50 error 0.3345\niteration 300 error 0.0456\nall entries non-negative: True\nrecovered topics (rows of H), rescaled so the largest entry is 1:\n[[0.  0.2 0.8 1.  0.4]\n [1.  0.6 0.  0.  0.2]]\ntrue topics, same scaling:\n[[1.  0.6 0.  0.  0.2]\n [0.  0.2 0.8 1.  0.4]]"],
    ["p", "The recovered topics match the true ones, possibly in a different order and scale (we rescaled each row to compare). Now the library, applied to real text: four tiny documents about two themes."],
    ["code", R`
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["the cat chased the mouse around the house",
        "a dog and a cat played in the garden",
        "the stock market fell as investors sold shares",
        "investors bought shares when the market rose",
        "the dog chased the cat",
        "shares and stocks rose in the market"]
tfidf = TfidfVectorizer(stop_words="english")
V = tfidf.fit_transform(docs)                                  # sparse, non-negative: 6 documents x words
words = np.array(tfidf.get_feature_names_out())

nmf = NMF(n_components=2, init="nndsvd", max_iter=1000, tol=1e-3, random_state=0)
W = nmf.fit_transform(V)                                       # documents x topics
H = nmf.components_                                            # topics x words
print("reconstruction error:", round(float(nmf.reconstruction_err_), 3))
for a in range(2):
    top = words[np.argsort(H[a])[::-1][:4]]
    print("topic", a, ":", top.tolist())
print("dominant topic per document:", W.argmax(axis=1).tolist())`, "reconstruction error: 1.549\ntopic 0 : ['shares', 'market', 'rose', 'investors']\ntopic 1 : ['cat', 'dog', 'chased', 'mouse']\ndominant topic per document: [1, 1, 0, 0, 1, 0]"],
    ["table", ["", "PCA / SVD", "NMF"], [
      ["Constraint", "orthogonal components, any sign", "all entries >= 0"],
      ["Solution", "closed form (eigenvectors)", "iterative, local optimum, not unique"],
      ["Components look like", "blends with positive and negative parts", "additive parts (strokes, topics, instruments)"],
      ["Ordered by importance", "yes", "no"],
      ["Needs non-negative data", "no", "yes"],
    ]],
    ["tip", "NMF is a good first tool for topic discovery on TF-IDF or word counts, for audio spectrograms (separating instruments), and for image parts. See [LDA topic models](classical.html#topic-lda) for the probabilistic cousin. Scale the data to a sensible range first; very different magnitudes in different columns make some parts dominate."],
    ["danger", "Feeding NMF data with **negative values** (for example standardised features), which raises an error, or centring the data first. NMF needs the original non-negative scale. Also, comparing the topics of two runs by index: the order changes with the initialisation, so match them by content."],
  ],
},
{
  id: "recsys-mf", group: "Factorization & Manifolds", title: "Matrix Factorization for Recommenders (ALS)",
  sub: "Give every user and every item a short vector of hidden traits, so a missing rating is predicted by the dot product of the two.",
  tags: ["recommender systems", "optimization"],
  blocks: [
    ["p", "A streaming service has millions of users and tens of thousands of films, but each user has rated only a few dozen. The table of ratings is **almost entirely empty**, and the job is to guess the empty cells. The key idea of **matrix factorization** is that taste is not random: a handful of hidden traits (how much a film is an action film, how much it is for children, how much a user likes either) explain most ratings."],
    ["p", R`So give every user $u$ a short vector $\mathbf p_u\in\mathbb R^k$ and every item $i$ a short vector $\mathbf q_i\in\mathbb R^k$, with $k$ around 10 to 200. The predicted rating is how well the two vectors **line up**, a [dot product](math.html#dot-product):`],
    ["math", R`\hat r_{ui}=\mu+b_u+b_i+\mathbf p_u^\top\mathbf q_i`, "μ: global average rating · b_u, b_i: user and item biases (a harsh rater, a popular film) · the dot product: personal taste"],
    ["p", R`Stacking all the vectors gives the picture $\mathbf R\approx\mathbf P\mathbf Q^\top$, a [low-rank](math.html#svd) approximation, but fitted **only on the observed cells**. Plain SVD cannot be used because it needs the whole matrix.`],
    ["h", "The objective"],
    ["math", R`\min_{\mathbf P,\mathbf Q}\;\sum_{(u,i)\in\Omega}\big(r_{ui}-\mathbf p_u^\top\mathbf q_i\big)^2+\lambda\Big(\sum_u\|\mathbf p_u\|^2+\sum_i\|\mathbf q_i\|^2\Big)`, "Ω: the set of observed (user, item) pairs · λ: regularization strength (see ridge regression)"],
    ["p", R`Without the $\lambda$ term, a user with two ratings could be fitted perfectly in many ways and the model would overfit badly. The penalty is [L2 regularization](ml.html#regularization) on the factors. Two ways to minimise it:`],
    ["list", [
      R`**SGD**: loop over observed ratings, compute the error $e_{ui}=r_{ui}-\mathbf p_u^\top\mathbf q_i$ and nudge $\mathbf p_u\leftarrow\mathbf p_u+\eta(e_{ui}\mathbf q_i-\lambda\mathbf p_u)$, $\mathbf q_i\leftarrow\mathbf q_i+\eta(e_{ui}\mathbf p_u-\lambda\mathbf q_i)$. Simple, good for streaming.`,
      R`**ALS (Alternating Least Squares)**: the objective is not convex in $(\mathbf P,\mathbf Q)$ together, but if you **freeze one side the other becomes an ordinary ridge regression** with a closed-form answer. Alternate. Each user (or item) is solved independently, which makes ALS trivially parallel and the method of choice for big clusters.`,
    ]],
    ["viz", "cml2_als"],
    ["p", "Step through it. Solid cells are known ratings; dashed amber cells are the ones the model predicts (the true hidden rating is in brackets). Each user step and each item step is a small linear solve, and the training error falls quickly."],
    ["h", "The ALS update"],
    ["p", R`Freeze all item vectors. For user $u$, let $\mathbf Q_u$ be the matrix whose rows are the vectors of the items $u$ rated, and $\mathbf r_u$ the vector of those ratings. Then user $u$'s subproblem is ridge regression with design matrix $\mathbf Q_u$:`],
    ["math", R`\mathbf p_u=\big(\mathbf Q_u^\top\mathbf Q_u+\lambda\mathbf I\big)^{-1}\mathbf Q_u^\top\mathbf r_u`, "a k×k linear system per user; the item update is the same with the roles swapped"],
    ["note", R`Tiny example, $k=1$. User $u$ rated two items whose current factors are $q=(1,2)$, with ratings $r=(3,5)$, and $\lambda=1$. Then $\mathbf Q_u^\top\mathbf Q_u=1^2+2^2=5$, $\mathbf Q_u^\top\mathbf r_u=1\cdot3+2\cdot5=13$, so $p_u=\frac{13}{5+1}\approx2.17$. The predictions for those items are $2.17$ and $4.33$, close to 3 and 5; without $\lambda$ the answer would be $2.6$.`],
    ["list", [
      "**Implicit feedback** (clicks, plays, purchases instead of star ratings): there is no 'dislike', and a missing entry means 'unknown'. Hu, Koren and Volinsky's variant treats every cell as a 0/1 preference with a confidence weight $c_{ui}=1+\\alpha\\,r_{ui}$ and still solves each user in closed form. This is what the `implicit` library does.",
      "**Cold start**: a new user or item has no ratings, so its vector cannot be learned. Use side information (genre, age) or popularity until data arrive.",
      "**Evaluate** on held-out ratings (RMSE) or, for top-N recommendation, with ranking metrics, and split by **time** when you can, as in real use ([splits](ml.html#splits)).",
    ]],
    ["h", "From scratch in NumPy"],
    ["p", "A synthetic 20 users × 12 items rating matrix with two hidden traits. 35% of the cells are hidden to test the model. ALS with $k=2$ runs for ten rounds; we track the error on the cells it trained on and on the hidden ones, and compare with the dumb baseline of predicting the global average."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
nu, ni, k, lam = 20, 12, 2, 0.5
P_true, Q_true = rng.normal(size=(nu, 2)), rng.normal(size=(ni, 2))
R = 3 + P_true @ Q_true.T + 0.2 * rng.normal(size=(nu, ni))          # full ratings (we pretend to not see most)
obs = rng.random((nu, ni)) > 0.35                                      # True where the rating is known
mu = R[obs].mean()

P = 0.1 * rng.normal(size=(nu, k)); Q = 0.1 * rng.normal(size=(ni, k))
def rmse(mask): return float(np.sqrt(((R - mu - P @ Q.T)[mask] ** 2).mean()))

for it in range(1, 11):
    for u in range(nu):                                                # fix Q, solve each user (ridge regression)
        Qu = Q[obs[u]]
        P[u] = np.linalg.solve(Qu.T @ Qu + lam * np.eye(k), Qu.T @ (R[u, obs[u]] - mu))
    for i in range(ni):                                                # fix P, solve each item
        Pi = P[obs[:, i]]
        Q[i] = np.linalg.solve(Pi.T @ Pi + lam * np.eye(k), Pi.T @ (R[obs[:, i], i] - mu))
    if it in (1, 2, 5, 10):
        print("round", it, " train RMSE", round(rmse(obs), 3), " hidden RMSE", round(rmse(~obs), 3))

baseline = float(np.sqrt(((R - mu)[~obs] ** 2).mean()))
print("baseline (always predict the mean) hidden RMSE:", round(baseline, 3))
u, i = 0, int(np.argmin(obs[0]))
print("user 0, hidden item", i, ": predicted", round(float(mu + P[u] @ Q[i]), 2), " true", round(float(R[u, i]), 2))`, "round 1  train RMSE 0.897  hidden RMSE 1.177\nround 2  train RMSE 0.358  hidden RMSE 0.686\nround 5  train RMSE 0.194  hidden RMSE 0.33\nround 10  train RMSE 0.188  hidden RMSE 0.322\nbaseline (always predict the mean) hidden RMSE: 1.232\nuser 0, hidden item 4 : predicted 3.08  true 3.18"],
    ["p", "Hidden-cell error drops from the baseline's roughly 1.2 to well below 0.5 after a few rounds (the exact numbers are printed above). Notice that every inner loop is a tiny $k\\times k$ solve; that is why ALS scales."],
    ["h", "Libraries"],
    ["p", "scikit-learn has no recommender, but its `TruncatedSVD` and `NMF` can factorize a filled-in matrix. For real recommenders use a dedicated library:"],
    ["code", R`
# pip install implicit scipy
import numpy as np, scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

# user_items: sparse (users x items) matrix of counts / confidences, e.g. play counts
rows, cols = np.array([0, 0, 1, 2, 2, 3]), np.array([0, 2, 1, 0, 3, 3])
user_items = sp.csr_matrix((np.array([5., 3., 4., 2., 1., 6.]), (rows, cols)), shape=(4, 4))

model = AlternatingLeastSquares(factors=16, regularization=0.05, iterations=15, random_state=0)
model.fit(user_items)                                          # ALS for implicit feedback
ids, scores = model.recommend(0, user_items[0], N=2)           # top-2 unseen items for user 0
print(ids, scores)`, "", "implicit", "local"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Handles huge sparse matrices; ALS parallelises per user and per item", "Cold start: new users and items have no vector"],
      ["Compact model: n·k + m·k numbers instead of n·m", "Only linear interactions between user and item factors"],
      ["Easy to add biases, time effects and side information", "Needs tuning of k and λ; factors have no built-in meaning"],
      ["Closed-form steps, no learning rate (ALS)", "Slower per pass than SGD on small data, and dense factor matrices must fit in memory"],
    ]],
    ["danger", "Evaluating on the cells you trained on, or splitting ratings **at random** when the product recommends the future from the past (use a time split). Also forgetting the biases and the global mean: a big part of the signal is just 'this user rates high' and 'this film is popular', and the factors waste capacity on it."],
    ["tip", "Related: [kernel methods](classical.html#kernel-methods) and [SVD](math.html#svd) for low-rank structure, [NMF](classical.html#nmf) when ratings are counts, and [Bayesian linear regression](classical.html#bayes-linreg) for the view of λ as a prior. Practice: [the ALS user update](practice.html#p/als-user-update)."],
  ],
},
{
  id: "tsne-umap", group: "Factorization & Manifolds", title: "t-SNE & UMAP",
  sub: "Draw high-dimensional data as a 2-D map in which points that were close neighbours stay neighbours.",
  tags: ["visualization", "manifold learning"],
  blocks: [
    ["p", R`A world atlas flattens a sphere. Any flat map distorts something, so cartographers choose what to keep: shapes, areas or directions. **t-SNE** (t-distributed Stochastic Neighbour Embedding) and **UMAP** do the same for data with hundreds of features: they place every point on a 2-D sheet so that **neighbours stay neighbours**. They are the standard tools for looking at embeddings, cell types in biology, or learned features of a neural network. [PCA](ml.html#pca) is the linear alternative; t-SNE and UMAP can unroll curved structure that PCA cannot.`],
    ["viz", "cml2_tsne"],
    ["p", "Step through the iterations. Early on, all points sit in a tiny blob; then clusters separate and tighten. Change the **perplexity**: tiny values break clusters into islands, large values start to merge them. Change the number of clusters too."],
    ["h", "The maths of t-SNE"],
    ["p", R`**Step 1: similarities in the original space.** For each point $i$ turn distances into a probability distribution over the other points with a Gaussian centred on $i$:`],
    ["math", R`p_{j\mid i}=\frac{\exp\!\big(-\|\mathbf x_i-\mathbf x_j\|^2/2\sigma_i^2\big)}{\sum_{k\ne i}\exp\!\big(-\|\mathbf x_i-\mathbf x_k\|^2/2\sigma_i^2\big)},\qquad p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2n}`, "p_ij: how likely i and j are to be neighbours (symmetrised so the total over all pairs is 1)"],
    ["p", R`The width $\sigma_i$ is chosen separately for each point so that the distribution has a fixed **perplexity**, defined as $2^{H(P_i)}$ where $H$ is the [entropy](math.html#entropy) in bits. Read it as "the effective number of neighbours": a point whose probability is spread evenly over 10 others has perplexity exactly 10. Dense regions get a small $\sigma_i$, sparse regions a large one, so every point sees a similar number of neighbours. This is found by a binary search.`],
    ["p", R`**Step 2: similarities in the map.** Give each point a 2-D position $\mathbf y_i$ and compute similarities with a **Student-t** (one degree of freedom) kernel instead of a Gaussian:`],
    ["math", R`q_{ij}=\frac{\big(1+\|\mathbf y_i-\mathbf y_j\|^2\big)^{-1}}{\sum_{k\ne l}\big(1+\|\mathbf y_k-\mathbf y_l\|^2\big)^{-1}}`, "heavy tails: far-apart points still have a non-negligible q"],
    ["p", "Why heavy tails? In high dimensions a point can have many neighbours at about the same distance, but a plane has little room around a point (the **crowding problem**). With a Gaussian in the map, moderately distant points would be squeezed together. A heavy-tailed kernel lets them sit far apart for a small cost, which leaves room for the clusters to separate."],
    ["p", R`**Step 3: match them.** Minimise the Kullback-Leibler divergence between the two sets of probabilities ([KL divergence](math.html#entropy)) by gradient descent on the positions $\mathbf y_i$:`],
    ["math", R`C=\operatorname{KL}(P\,\|\,Q)=\sum_{i\ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}},\qquad \frac{\partial C}{\partial \mathbf y_i}=4\sum_{j}(p_{ij}-q_{ij})\,\frac{\mathbf y_i-\mathbf y_j}{1+\|\mathbf y_i-\mathbf y_j\|^2}`, "each pair pulls (p > q: they should be closer) or pushes (q > p: they should be farther)"],
    ["note", R`The cost is asymmetric. If two true neighbours are placed far apart ($p_{ij}$ large, $q_{ij}$ small) the term $p\log(p/q)$ is big, so that mistake is punished hard. If two non-neighbours end up close ($p$ tiny, $q$ larger) the penalty is mild. So **local** structure is preserved carefully and **global** structure (distances between clusters) is not.`],
    ["h", "From scratch in NumPy"],
    ["p", "A small exact t-SNE (no tricks beyond early exaggeration and momentum): 90 points in 10 dimensions forming three clusters. We measure how well neighbourhoods survive: the average fraction of each point's 10 nearest neighbours in the original space that are still among its 10 nearest in the 2-D map. PCA is the baseline."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
centers = rng.normal(scale=6, size=(3, 10))
X = np.vstack([c + rng.normal(size=(30, 10)) for c in centers])        # 90 points, 3 clusters, 10-D
n = len(X)
D = ((X[:, None] - X[None]) ** 2).sum(-1)                              # squared distances

def affinities(D, perp):                                               # step 1: find sigma_i by bisection on the entropy
    P = np.zeros_like(D)
    for i in range(n):
        lo, hi, beta = 0.0, np.inf, 1.0
        for _ in range(60):
            p = np.exp(-D[i] * beta); p[i] = 0; p /= p.sum()
            H = -(p[p > 0] * np.log(p[p > 0])).sum()
            if H > np.log(perp): lo = beta; beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else: hi = beta; beta = (beta + lo) / 2
        P[i] = p
    return (P + P.T) / (2 * n)

P = affinities(D, perp=15)
Y = 1e-2 * rng.normal(size=(n, 2)); vel = np.zeros_like(Y)
for it in range(1, 301):
    ex = 4.0 if it <= 60 else 1.0                                      # early exaggeration
    num = 1 / (1 + ((Y[:, None] - Y[None]) ** 2).sum(-1)); np.fill_diagonal(num, 0)
    Q = num / num.sum()
    G = 4 * ((ex * P - Q) * num)[:, :, None] * (Y[:, None] - Y[None])  # gradient contributions
    vel = (0.5 if it < 60 else 0.8) * vel - 40 * G.sum(axis=1)         # momentum + learning rate
    Y += vel; Y -= Y.mean(axis=0)
    if it in (60, 300):
        kl = float((P[P > 0] * np.log(P[P > 0] / np.maximum(Q[P > 0], 1e-12))).sum())
        print("iteration", it, "KL(P||Q) =", round(kl, 3))

def knn_overlap(A, B, k=10):                                           # share of kept neighbours
    nn = lambda M: np.argsort(((M[:, None] - M[None]) ** 2).sum(-1), axis=1)[:, 1:k + 1]
    a, b = nn(A), nn(B)
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(A))]))

Xc = X - X.mean(axis=0); Y_pca = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:2].T
print("neighbourhood kept, t-SNE:", round(knn_overlap(X, Y), 2), " PCA:", round(knn_overlap(X, Y_pca), 2))`, "iteration 60 KL(P||Q) = 0.968\niteration 300 KL(P||Q) = 0.303\nneighbourhood kept, t-SNE: 0.64  PCA: 0.48"],
    ["p", "On clustered data both methods keep clusters apart, but t-SNE preserves the fine neighbourhood inside each cluster much better than a flat projection. The library version is a one-liner:"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)
X, y = X[:300], y[:300]                                            # 300 handwritten digits, 64 features
emb = TSNE(n_components=2, perplexity=20, init="pca", random_state=0).fit_transform(X)
print("embedding shape:", emb.shape)

# does a point's nearest neighbour in the 2-D map have the same digit?
def nn_label_agree(E):
    d = ((E[:, None] - E[None]) ** 2).sum(-1); np.fill_diagonal(d, np.inf)
    return float((y[d.argmin(axis=1)] == y).mean())
print("1-NN label agreement  t-SNE:", round(nn_label_agree(emb), 3), " PCA:", round(nn_label_agree(PCA(2).fit_transform(X)), 3))`, "embedding shape: (300, 2)\n1-NN label agreement  t-SNE: 0.99  PCA: 0.657"],
    ["h", "UMAP"],
    ["p", "**UMAP** (Uniform Manifold Approximation and Projection) has the same goal and the same two-stage shape, built on a different foundation. It builds a weighted graph of each point's `n_neighbors` nearest neighbours (a fuzzy topological representation of the data's manifold), then optimises a low-dimensional layout so that its own graph matches, using a cross-entropy loss with attraction along edges and sampled repulsion for non-edges. In practice that gives three differences from t-SNE: it is **much faster** and scales to millions of points, it **keeps more global structure** (distances between clusters mean a little more), and it can **transform new points** after fitting."],
    ["table", ["Setting", "What it controls"], [
      ["t-SNE `perplexity` (5 to 50, default 30)", "Effective neighbours per point. Small: many tiny islands. Large: blurs local detail. Should be below the number of points."],
      ["UMAP `n_neighbors` (default 15)", "Size of the neighbourhood used to build the graph. Small: fine local detail. Large: more global view."],
      ["UMAP `min_dist` (default 0.1)", "How tightly points may pack in the map. Small: dense clumps. Large: spread out."],
      ["both: `random_state`", "Results depend on the random start; fix the seed and look at several runs before believing a pattern."],
    ]],
    ["code", R`
# pip install umap-learn
import umap
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)
emb = reducer.fit_transform(X)            # (1797, 2)
new_emb = reducer.transform(X[:5])        # unlike t-SNE, UMAP can embed new points
print(emb.shape, new_emb.shape)`, "", "umap-learn", "local"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Reveals cluster structure and curved manifolds that PCA flattens", "Cluster sizes, and distances between clusters, are **not reliable**"],
      ["Works with no labels, great for exploration", "Stochastic and sensitive to perplexity / neighbours; non-convex"],
      ["UMAP: fast, can transform new data, keeps more global shape", "t-SNE has no `transform` for new points and is slow on big data (use Barnes-Hut or FIt-SNE)"],
      ["Only needs a distance (any metric)", "Can show clusters in pure noise if the parameters are bad"],
    ]],
    ["danger", "Treating the picture as a measurement. The **size** of a blob, the **gap** between blobs and the **direction** of the axes in a t-SNE or UMAP map mean little. Do not run [k-means](ml.html#kmeans) on the 2-D coordinates and report it as the true clustering, and do not use the embedding as features for a model unless you fit it inside cross-validation. Always try several perplexities and seeds."],
    ["tip", "Run PCA down to 30-50 dimensions first on very wide data: it removes noise and speeds t-SNE up. For a neighbour-preserving map that is also linear and fast, stay with [PCA](ml.html#pca); for globally faithful unrolling see [MDS and Isomap](classical.html#mds-isomap)."],
  ],
},
{
  id: "mds-isomap", group: "Factorization & Manifolds", title: "MDS & Isomap",
  sub: "Rebuild a map of points from a table of distances, and unroll curved data by measuring distances along the surface instead of through space.",
  tags: ["manifold learning", "dimensionality reduction"],
  blocks: [
    ["p", "Here is a classic puzzle. You have a table of flight distances between 10 cities and no map. Can you recover a map? **Multidimensional scaling (MDS)** does exactly that: given only the table of pairwise distances, it places points in 2-D (or any dimension) so that the distances on the map match the table as well as possible. It never needs the original coordinates, only how far apart things are, so it works for any notion of dissimilarity: genetic distance, how differently two people rated films, edit distance between words."],
    ["h", "Classical MDS: the maths"],
    ["p", R`Let $\mathbf D^{(2)}$ be the matrix of **squared** distances $d_{ij}^2$ between $n$ points. If the points had coordinates $\mathbf X$ (centred at the origin), then $d_{ij}^2=\|\mathbf x_i\|^2+\|\mathbf x_j\|^2-2\mathbf x_i^\top\mathbf x_j$. The cross term is the dot-product (Gram) matrix $\mathbf B=\mathbf X\mathbf X^\top$. We can undo the squared norms by **double centring**:`],
    ["math", R`\mathbf B=-\tfrac12\,\mathbf J\,\mathbf D^{(2)}\,\mathbf J,\qquad \mathbf J=\mathbf I-\tfrac1n\mathbf 1\mathbf 1^\top`, "J subtracts row and column means (it centres the points on the origin)"],
    ["p", R`$\mathbf B$ is symmetric, so use its [eigen-decomposition](math.html#eigen) $\mathbf B=\mathbf V\mathbf\Lambda\mathbf V^\top$ and keep the $k$ largest eigenvalues. The coordinates are`],
    ["math", R`\mathbf Y=\mathbf V_k\,\mathbf\Lambda_k^{1/2}\quad(n\times k)`, "the k top eigenvectors, each scaled by the square root of its eigenvalue"],
    ["note", R`Tiny example: three points on a line at $0,1,2$, so the squared distances are $d_{01}^2=1,\;d_{12}^2=1,\;d_{02}^2=4$. Double centring gives $\mathbf B=\begin{pmatrix}1&0&-1\\0&0&0\\-1&0&1\end{pmatrix}$, whose only non-zero eigenvalue is $2$ with eigenvector $(1,0,-1)/\sqrt2$. Then $\mathbf Y=\sqrt2\cdot(1,0,-1)/\sqrt2=(1,0,-1)$: the original line, centred, up to a flip. One eigenvalue means one dimension is enough.`],
    ["p", "If the distances are ordinary Euclidean distances between points, classical MDS gives **exactly the same embedding as [PCA](ml.html#pca)** (the eigenvalues of $\mathbf B$ are $n$ times those of the covariance matrix). Its power is that it works from distances alone. **Metric MDS** (what scikit-learn's `MDS` does) is more flexible: it directly minimises the **stress**"],
    ["math", R`\text{stress}=\sum_{i<j}\big(d_{ij}-\|\mathbf y_i-\mathbf y_j\|\big)^2`, "d_ij: the target dissimilarity · the norm: the distance on the map. Minimised iteratively (SMACOF)"],
    ["h", "The problem: curved data"],
    ["diagram", "cml2_manifold", [], "On a curled sheet, the straight-line distance cuts through empty space; the geodesic distance follows the sheet."],
    ["p", "Take points on a rolled-up sheet like a Swiss roll. Two points on neighbouring layers are close in space but far apart if you walk along the sheet. MDS and PCA use the straight-line distance, so they squash the roll flat and glue different layers together. **Isomap** fixes this with one idea: use the **geodesic distance**, the length of the shortest path that stays on the data."],
    ["steps", [
      "**Neighbourhood graph**: connect each point to its $k$ nearest neighbours (edge length = Euclidean distance). Short hops are accurate even on a curved surface.",
      "**Geodesic distances**: the distance between any two points is the length of the shortest path through the graph (Dijkstra or Floyd-Warshall).",
      "**Classical MDS** on that distance matrix, exactly as above.",
    ]],
    ["h", "From scratch in NumPy"],
    ["p", "We implement classical MDS and Isomap and unroll a noisy Swiss roll. The data come with the true position `t` along the roll, so we can score how well each method recovers it: a good unrolling has a first coordinate that is almost perfectly (rank-)correlated with `t`."],
    ["code", R`
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph

X, t = make_swiss_roll(n_samples=800, noise=0.05, random_state=0)       # 3-D points, t = position along the roll

def classical_mds(D, k=2):
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J                                         # double centring
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:k]                                    # k largest eigenvalues
    return vecs[:, idx] * np.sqrt(vals[idx])

# MDS on straight-line distances (this equals PCA)
D_euc = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
Y_mds = classical_mds(D_euc)

# Isomap: graph of 10 nearest neighbours -> shortest paths -> MDS
G = kneighbors_graph(X, n_neighbors=10, mode="distance")
D_geo = shortest_path(G, method="D", directed=False)                    # geodesic distances
print("graph connected:", bool(np.isfinite(D_geo).all()))
Y_iso = classical_mds(D_geo)

for name, Y in [("MDS / PCA", Y_mds), ("Isomap", Y_iso)]:
    best = max(abs(float(spearmanr(Y[:, c], t)[0])) for c in range(2))
    print(name, ": best |rank correlation| of a coordinate with the roll position t =", round(best, 3))
print("example distance: straight line", round(float(D_euc[0, 1]), 2) , "-- geodesic", round(float(D_geo[0, 1]), 2))`, "graph connected: True\nMDS / PCA : best |rank correlation| of a coordinate with the roll position t = 0.19\nIsomap : best |rank correlation| of a coordinate with the roll position t = 1.0\nexample distance: straight line 19.67 -- geodesic 21.59"],
    ["p", "Isomap's coordinate follows the unrolled sheet; the straight-line methods cannot. The library versions:"],
    ["code", R`
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import MDS, Isomap

X, t = make_swiss_roll(n_samples=800, noise=0.05, random_state=0)

iso = Isomap(n_neighbors=10, n_components=2)
Y_iso = iso.fit_transform(X)
mds = MDS(n_components=2, n_init=1, max_iter=100, random_state=0)
Y_mds = mds.fit_transform(X)                                           # metric MDS: minimises stress by SMACOF

def score(Y): return round(max(abs(float(spearmanr(Y[:, c], t)[0])) for c in range(2)), 3)
print("Isomap  |rho| with t:", score(Y_iso), " reconstruction error:", round(float(iso.reconstruction_error()), 3))
print("MDS     |rho| with t:", score(Y_mds), " stress:", round(float(mds.stress_), 1))
new = iso.transform(X[:3]); print("Isomap can embed new points:", new.shape)`, "Isomap  |rho| with t: 1.0  reconstruction error: 10.216\nMDS     |rho| with t: 0.241  stress: 3989159.4\nIsomap can embed new points: (3, 2)"],
    ["table", ["Method", "Distances used", "Handles curved data?", "New points?"], [
      ["PCA", "implicit Euclidean (covariance)", "no (linear)", "yes"],
      ["Classical MDS", "any distance table (Euclidean gives PCA)", "no for Euclidean input", "no"],
      ["Metric / non-metric MDS", "any dissimilarity, minimises stress", "only as well as the input distances allow", "no"],
      ["Isomap", "geodesic (through the neighbour graph)", "yes, if the surface is smooth and convex", "yes (`transform`)"],
      ["[t-SNE / UMAP](classical.html#tsne-umap)", "local neighbourhoods only", "yes, but global distances unreliable", "UMAP yes, t-SNE no"],
    ]],
    ["table", ["Strengths", "Weaknesses"], [
      ["MDS needs only a distance table", "Quadratic memory: n×n matrix, an eigen-solve per run"],
      ["Isomap preserves global geometry of smooth manifolds", "**Short-circuits**: one edge across the gap between folds ruins every geodesic"],
      ["Few parameters (k and the output dimension)", "Needs a connected graph; sensitive to noise and to the choice of k"],
    ]],
    ["danger", "Picking k for Isomap too large: a single edge that jumps across layers of the roll makes distant points suddenly close and the whole embedding collapses. Pick the smallest k that gives a connected graph, and check by colouring the map by a known quantity. With a too small k the graph falls apart into disconnected pieces and the distances become infinite."],
    ["tip", "Related: [kernel PCA](classical.html#kernel-methods) is the other way to curve PCA, [spectral clustering](classical.html#spectral-clustering) uses the same neighbour-graph idea, and [eigenvectors](math.html#eigen) are the engine under all of them."],
  ],
},
{
  id: "spectral-clustering", group: "Factorization & Manifolds", title: "Spectral Clustering",
  sub: "Cluster by cutting a similarity graph: embed the points with the eigenvectors of the graph Laplacian, then run k-means on the embedding.",
  tags: ["clustering", "graphs"],
  blocks: [
    ["p", R`[k-means](ml.html#kmeans) assumes clusters are round blobs around a centre. Real clusters are often not: two crescents, concentric rings, long winding shapes. What defines them is not the distance to a centre but **connectedness**: you can walk from any point in the cluster to any other in small steps without leaving it. **Spectral clustering** captures exactly this. Build a graph where nearby points are linked, then find the way of cutting the graph into pieces that cuts as few links as possible. [DBSCAN](ml.html#dbscan) also finds such shapes by chaining; spectral clustering does it by linear algebra.`],
    ["viz", "cml2_spectral"],
    ["p", "Step through the demo: k-means slices both moons, while the graph and its eigenvectors turn each moon into a tight clump that k-means then splits correctly."],
    ["h", "The maths"],
    ["steps", [
      R`**Similarity graph.** Build an affinity matrix $\mathbf W$ with $W_{ij}$ large when points $i,j$ are close. Typical choice: link each point to its $k$ nearest neighbours, with weight $W_{ij}=\exp(-\|\mathbf x_i-\mathbf x_j\|^2/\sigma^2)$ (symmetrised).`,
      R`**Degrees and Laplacian.** $\mathbf D$ is diagonal with $D_{ii}=\sum_jW_{ij}$. The unnormalised Laplacian is $\mathbf L=\mathbf D-\mathbf W$; the symmetric normalised one is $\mathbf L_{\text{sym}}=\mathbf I-\mathbf D^{-1/2}\mathbf W\mathbf D^{-1/2}$.`,
      R`**Eigenvectors.** Take the $k$ eigenvectors of $\mathbf L_{\text{sym}}$ with the **smallest** eigenvalues and stack them as columns of $\mathbf U\in\mathbb R^{n\times k}$. Normalise each row to length 1.`,
      R`**k-means** on the $n$ rows of $\mathbf U$. Point $i$ gets the cluster of row $i$.`,
    ]],
    ["p", "Why does this work? Two facts."],
    ["list", [
      R`The quadratic form of the Laplacian measures how much a signal $f$ varies across edges: $\mathbf f^\top\mathbf L\mathbf f=\tfrac12\sum_{i,j}W_{ij}(f_i-f_j)^2$. A vector that is **constant inside each cluster** has almost zero energy, because the only edges where $f$ differs are the few weak ones between clusters. So the smallest eigenvectors are (nearly) piecewise-constant indicators of the clusters.`,
      R`If the graph has exactly $k$ connected components, the eigenvalue $0$ appears exactly $k$ times and its eigenvectors are indicator vectors of the components. With weak links between clusters the zeros become small numbers, which is why spectral clustering is robust. The first sign of how many clusters to use is an **eigengap**: a big jump after the $k$-th smallest eigenvalue.`,
    ]],
    ["p", R`The cut interpretation: minimising the **normalised cut** $\operatorname{Ncut}(A,B)=\operatorname{cut}(A,B)\big(\tfrac1{\operatorname{vol}(A)}+\tfrac1{\operatorname{vol}(B)}\big)$, which prefers balanced pieces, is NP-hard. Allowing the cluster indicator to take real values instead of two values turns it into the eigenvector problem above. The second-smallest eigenvector (the **Fiedler vector**) of $\mathbf L_{\text{sym}}$ (after multiplying by $\mathbf D^{-1/2}$) is the relaxed best two-way cut: its signs split the graph.`],
    ["note", R`Tiny example: four nodes, edges $1\text{–}2$ and $3\text{–}4$ with weight $1$, and a weak bridge $2\text{–}3$ with weight $0.1$. The Fiedler vector has one sign on nodes $\{1,2\}$ and the other on $\{3,4\}$: the cut removes only the weight-$0.1$ edge. The code below computes it.`],
    ["h", "From scratch in NumPy"],
    ["p", "First the four-node example, then the full algorithm on two noisy half-moons, scored against the truth with the Adjusted Rand Index (1 is perfect, 0 is chance). k-means is run on the raw points for comparison."],
    ["code", R`
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph

# 1) the four-node toy graph
W = np.array([[0, 1, 0, 0], [1, 0, 0.1, 0], [0, 0.1, 0, 1], [0, 0, 1, 0]], float)
L = np.diag(W.sum(1)) - W
vals, vecs = np.linalg.eigh(L)
print("eigenvalues of L:", vals.round(3).tolist())
print("Fiedler vector signs:", np.sign(vecs[:, 1]).astype(int).tolist())

# 2) two moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
A = kneighbors_graph(X, n_neighbors=10, mode="connectivity").toarray()
W = np.maximum(A, A.T)                                                   # symmetric k-NN graph
d = W.sum(1)
L_sym = np.eye(len(X)) - W / np.sqrt(np.outer(d, d))                     # I - D^-1/2 W D^-1/2
vals, vecs = np.linalg.eigh(L_sym)
print("smallest eigenvalues:", (vals[:4].round(4) + 0.0).tolist())
U = vecs[:, :2]
U = U / np.linalg.norm(U, axis=1, keepdims=True)                         # normalise rows
labels = KMeans(2, n_init=10, random_state=0).fit_predict(U)
plain = KMeans(2, n_init=10, random_state=0).fit_predict(X)
print("ARI, spectral:", round(adjusted_rand_score(y, labels), 3), " k-means:", round(adjusted_rand_score(y, plain), 3))`, "eigenvalues of L: [0.0, 0.095, 2.0, 2.105]\nFiedler vector signs: [1, 1, -1, -1]\nsmallest eigenvalues: [0.0, 0.001, 0.0056, 0.0073]\nARI, spectral: 1.0  k-means: 0.256"],
    ["p", "Now the library. `SpectralClustering` does the same steps; `affinity=\"nearest_neighbors\"` builds the k-NN graph, while the default `\"rbf\"` uses a dense Gaussian kernel and needs a good `gamma`."],
    ["code", R`
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import adjusted_rand_score

for name, (X, y) in {"moons": make_moons(300, noise=0.05, random_state=0),
                     "circles": make_circles(300, noise=0.05, factor=0.4, random_state=0)}.items():
    sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=12,
                            assign_labels="kmeans", random_state=0)
    ari_sc = adjusted_rand_score(y, sc.fit_predict(X))
    ari_km = adjusted_rand_score(y, KMeans(2, n_init=10, random_state=0).fit_predict(X))
    print(name, " spectral ARI:", round(ari_sc, 3), " k-means ARI:", round(ari_km, 3))`, "moons  spectral ARI: 1.0  k-means ARI: 0.241\ncircles  spectral ARI: 1.0  k-means ARI: -0.002"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Finds non-convex clusters (moons, rings, chains)", "Needs the number of clusters k in advance (use the eigengap)"],
      ["Few assumptions, solid theory (graph cuts)", "Dense eigen-solve is O(n³); big data needs sparse or Nyström approximations"],
      ["Works with any similarity, including graphs that are not points", "Sensitive to the graph: k or σ. Too many neighbours merges clusters, too few breaks them up"],
      ["Same idea behind many embeddings (Laplacian eigenmaps)", "No natural way to assign new points without extra work"],
    ]],
    ["danger", "Using a fully connected Gaussian affinity with a bad σ: too wide and every point looks like every other (one blur), too narrow and the graph shatters into isolated points. A sparse k-NN graph is much easier to tune. Also, forgetting to normalise the rows of the eigenvector matrix, or using eigenvectors of the unnormalised $\\mathbf L$ when cluster sizes differ a lot."],
    ["tip", "Compare with the model-based view in [Gaussian mixtures](classical.html#gmm), which also gives soft clusters but assumes blobs. The eigenvector machinery is the same as in [PCA](ml.html#pca) and [kernel PCA](classical.html#kernel-methods); only the matrix (a graph Laplacian) differs."],
  ],
},
{
  id: "adaboost", group: "Other Classics", title: "AdaBoost",
  sub: "Combine many weak classifiers by training each one on re-weighted data that emphasises what the previous ones got wrong.",
  tags: ["ensembles", "boosting"],
  blocks: [
    ["p", "A single weak rule such as \"if the message has the word FREE then spam\" is right a bit more often than a coin. Hundreds of such rules, each concentrating on the examples the others missed, then voting, can be a very accurate classifier. This is **boosting**, and **AdaBoost** (Adaptive Boosting, Freund and Schapire, 1995) is the original and clearest version. It came before [gradient boosting and XGBoost](ml.html#boosting), which generalise it."],
    ["viz", "cml2_adaboost"],
    ["p", "Watch the dot sizes. After each round the points that were misclassified (red ring) become bigger, i.e. heavier. The next stump then has to take them seriously, even if that costs it errors on points it used to get right. After a few rounds, the staircase of stumps already approximates the diagonal boundary that no single stump can draw."],
    ["h", "The algorithm"],
    ["p", R`We have $n$ training points with labels $y_i\in\{-1,+1\}$. Start with equal weights $w_i=1/n$. For rounds $t=1,\dots,T$:`],
    ["steps", [
      R`**Fit a weak learner** $h_t(\mathbf x)\in\{-1,+1\}$ (usually a decision stump) to minimise the **weighted** error $\varepsilon_t=\sum_i w_i\,\mathbf 1[h_t(\mathbf x_i)\ne y_i]$.`,
      R`**Compute its vote** $\alpha_t=\tfrac12\ln\dfrac{1-\varepsilon_t}{\varepsilon_t}$. A perfect learner ($\varepsilon\to0$) gets a huge vote, a coin flip ($\varepsilon=\tfrac12$) gets zero, and a learner worse than a coin gets a negative vote (its prediction is flipped).`,
      R`**Re-weight**: $w_i\leftarrow w_i\,e^{-\alpha_t\,y_i\,h_t(\mathbf x_i)}$, then divide by the sum so the weights add to 1. Correct points shrink by $e^{-\alpha_t}$, mistakes grow by $e^{\alpha_t}$.`,
    ]],
    ["math", R`H(\mathbf x)=\operatorname{sign}\Big(\sum_{t=1}^{T}\alpha_t\,h_t(\mathbf x)\Big)`, "the final classifier: a weighted vote of all T weak learners"],
    ["note", R`Worked example. Five points, weights $0.2$ each. The first stump misclassifies exactly one point, so $\varepsilon=0.2$ and $\alpha=\tfrac12\ln\frac{0.8}{0.2}=\tfrac12\ln4=0.693$. The wrong point's weight is multiplied by $e^{0.693}=2$ (to $0.4$) and each of the four right ones by $e^{-0.693}=0.5$ (to $0.1$). The sum is $0.4+4\cdot0.1=0.8$, so after normalising the wrong point has weight $0.5$ and each right one $0.125$. **The misclassified point now carries half of the total weight**, so the next stump will not be allowed to ignore it. This always happens: after re-weighting, the previous stump has weighted error exactly $\tfrac12$ on the new weights.`],
    ["h", "Why it works"],
    ["p", R`AdaBoost is the minimisation of the **exponential loss** $L=\sum_i e^{-y_iF(\mathbf x_i)}$ with $F=\sum_t\alpha_t h_t$, done greedily: at each round add the best $h_t$ and the best $\alpha_t$ to the current $F$ (stagewise additive modelling). Setting the derivative with respect to $\alpha$ to zero gives exactly the formula $\alpha_t=\tfrac12\ln\frac{1-\varepsilon_t}{\varepsilon_t}$. The weights $w_i\propto e^{-y_iF(\mathbf x_i)}$ are just the current losses per point, which is why hard points have large weights. [Gradient boosting](ml.html#boosting) is the same recipe with an arbitrary differentiable loss.`],
    ["p", R`There is also a clean training-error bound: $\text{error}\le\prod_t2\sqrt{\varepsilon_t(1-\varepsilon_t)}$. As long as every weak learner beats a coin by a margin ($\varepsilon_t<\tfrac12-\gamma$), the product shrinks exponentially in $T$: training error reaches zero in a number of rounds that grows only like $\log n/\gamma^2$.`],
    ["h", "From scratch in NumPy"],
    ["p", "A decision stump is found by trying a grid of thresholds on each feature and both polarities. We boost 50 stumps on noisy two-moons data and watch train and test accuracy; then compare with scikit-learn's `AdaBoostClassifier` using the same stump as its weak learner."],
    ["code", R`
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y01 = make_moons(n_samples=400, noise=0.3, random_state=0)
y = 2 * y01 - 1                                                       # labels in {-1, +1}
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)

def best_stump(X, y, w):                                              # exhaustive search over feature, threshold, sign
    best = (np.inf, 0, 0.0, 1)
    for f in range(X.shape[1]):
        for thr in np.quantile(X[:, f], np.linspace(0.05, 0.95, 19)):
            for s in (1, -1):
                pred = s * np.where(X[:, f] > thr, 1, -1)
                err = w[pred != y].sum()
                if err < best[0]: best = (err, f, thr, s)
    return best

def stump_predict(X, f, thr, s): return s * np.where(X[:, f] > thr, 1, -1)

n = len(ytr); w = np.full(n, 1 / n); stumps = []
for t in range(1, 51):
    err, f, thr, s = best_stump(Xtr, ytr, w)
    err = max(err, 1e-12)
    alpha = 0.5 * np.log((1 - err) / err)
    w *= np.exp(-alpha * ytr * stump_predict(Xtr, f, thr, s)); w /= w.sum()      # re-weight and normalise
    stumps.append((alpha, f, thr, s))
    if t in (1, 5, 20, 50):
        F = lambda Xs: sum(a * stump_predict(Xs, f_, t_, s_) for a, f_, t_, s_ in stumps)
        print("rounds", t, " train acc", round(float((np.sign(F(Xtr)) == ytr).mean()), 3), " test acc", round(float((np.sign(F(Xte)) == yte).mean()), 3), " last alpha", round(float(alpha), 3))
print("first stump: feature", stumps[0][1], "threshold", round(float(stumps[0][2]), 2))`, "rounds 1  train acc 0.82  test acc 0.8  last alpha 0.758\nrounds 5  train acc 0.86  test acc 0.855  last alpha 0.259\nrounds 20  train acc 0.885  test acc 0.905  last alpha 0.131\nrounds 50  train acc 0.895  test acc 0.9  last alpha 0.072\nfirst stump: feature 1 threshold 0.08"],
    ["code", R`
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)

stump = DecisionTreeClassifier(max_depth=1)                       # the weak learner
for T in (1, 10, 50, 200):
    ada = AdaBoostClassifier(estimator=stump, n_estimators=T, learning_rate=1.0, random_state=0).fit(Xtr, ytr)
    print("n_estimators", T, " train", round(ada.score(Xtr, ytr), 3), " test", round(ada.score(Xte, yte), 3))
ada = AdaBoostClassifier(estimator=stump, n_estimators=50, random_state=0).fit(Xtr, ytr)
print("first 3 estimator weights:", ada.estimator_weights_[:3].round(3).tolist())
print("first 3 estimator errors:", ada.estimator_errors_[:3].round(3).tolist())`, "n_estimators 1  train 0.83  test 0.795\nn_estimators 10  train 0.88  test 0.9\nn_estimators 50  train 0.905  test 0.895\nn_estimators 200  train 0.965  test 0.895\nfirst 3 estimator weights: [1.586, 0.537, 1.195]\nfirst 3 estimator errors: [0.17, 0.369, 0.232]"],
    ["p", "scikit-learn's multi-class AdaBoost (algorithm SAMME) uses $\\alpha_t=\\ln\\frac{1-\\varepsilon_t}{\\varepsilon_t}+\\ln(K-1)$, which for two classes is exactly twice the value we used. Doubling every vote does not change the sign of the sum, so the predictions are the same; only the printed weights differ by that factor."],
    ["table", ["Setting", "Effect"], [
      ["`n_estimators`", "Number of rounds. More rounds fit harder; watch the test curve."],
      ["`learning_rate`", "Shrinks each vote. Smaller values need more rounds but usually generalise better."],
      ["`estimator` depth", "Stumps (depth 1) are the classic choice; depth 2 to 4 trees capture interactions. A deep tree is already strong and boosts less."],
    ]],
    ["table", ["Strengths", "Weaknesses"], [
      ["Simple, few knobs, works with any weak learner", "**Sensitive to label noise and outliers**: they keep gaining weight"],
      ["Often resists overfitting for a long time, margin view", "Sequential: cannot be parallelised like a forest"],
      ["Built-in feature selection with stumps (each stump uses one feature)", "Usually beaten by modern gradient boosting libraries on tabular data"],
      ["Famous in vision: the Viola-Jones face detector", "Predicted scores are not calibrated probabilities"],
    ]],
    ["danger", "Boosting on data with many **mislabelled** points. The exponential loss gives those points ever-growing weights and the ensemble contorts itself to fit them. Clean the labels, use a smaller learning rate with early stopping, or move to a more robust loss (see [gradient boosting](ml.html#boosting))."],
    ["tip", "Contrast with [bagging and random forests](ml.html#random-forests): they train strong, decorrelated models **in parallel** and average to reduce variance; boosting trains weak models **in sequence** to reduce bias. Practice: [the AdaBoost round problem](practice.html#p/adaboost-round)."],
  ],
},
{
  id: "anomaly-detection", group: "Other Classics", title: "Anomaly Detection: Isolation Forest, LOF & One-Class SVM",
  sub: "Find the rare points that do not fit: by how easily they are isolated, how sparse their neighbourhood is, or how far outside a learned boundary they lie.",
  tags: ["unsupervised", "outliers"],
  blocks: [
    ["p", "Fraudulent card payments, a failing sensor, a network intrusion, a manufacturing defect: you often cannot collect many labelled examples of \"bad\", because bad is rare, varied and sometimes brand new. **Anomaly detection** flips the problem: learn what **normal** looks like (usually with no labels at all) and flag whatever does not fit. Because the examples of the rare class are so few, the ordinary classification metrics and tricks need care (see [imbalanced data](ml.html#imbalanced))."],
    ["table", ["Family", "Idea", "Example"], [
      ["Isolation", "anomalies are **few and different**, so they are easy to cut off from the rest", "Isolation Forest"],
      ["Density", "an anomaly sits in a much **sparser** area than its neighbours", "Local Outlier Factor (LOF), k-NN distance"],
      ["Boundary", "learn a region that contains the normal data; outside = anomaly", "One-Class SVM"],
      ["Reconstruction", "a model of normal data cannot rebuild an anomaly well", "[PCA](ml.html#pca) error, autoencoders, [GMM](classical.html#gmm) likelihood"],
    ]],
    ["h", "Isolation Forest"],
    ["p", "The cleverest idea in the family is also the simplest. Pick a random feature and a random split value between the minimum and maximum, and cut. Repeat inside the part that holds your point, until it is alone. A point far from everyone is almost always isolated within a few cuts; a point deep inside a crowd needs many. The number of cuts is the point's **path length**. Average it over many random trees and shorter means more anomalous."],
    ["viz", "cml2_iforest"],
    ["p", "Choose the red point (0 is the outlier, 1 and 2 are ordinary points) and change the tree number. The outlier is alone after only a couple of cuts in almost every tree; the ordinary points take many more. The side panel shows the average over 120 random trees and the resulting anomaly score."],
    ["math", R`s(\mathbf x,n)=2^{-\,\mathbb E[h(\mathbf x)]\,/\,c(n)},\qquad c(n)=2\big(\ln(n-1)+\gamma\big)-\frac{2(n-1)}{n}`, "h(x): path length of x · c(n): average path length of an unsuccessful binary-search-tree lookup among n points (γ ≈ 0.5772) · s near 1: anomaly, near 0.5 or below: normal"],
    ["p", R`The normalising constant $c(n)$ makes the score comparable across sample sizes: a path of about $c(n)$ gives $s=0.5$ (nothing special), a path much shorter gives $s\to1$ (anomaly). Each tree is built on a small random **sub-sample** (256 points by default), which both speeds it up and helps: with fewer points, anomalies are less hidden by their neighbours (the "swamping" and "masking" effects). Cost is linear in $n$.`],
    ["h", "Local Outlier Factor"],
    ["p", R`Distance to the neighbours is misleading when densities differ. A point 1.5 units from a tight cluster is an outlier, while a point 1.5 units from the others in a loose cluster is perfectly normal. LOF compares a point's **local density** with the density of its neighbours:`],
    ["math", R`\operatorname{reach}_k(a,b)=\max\{d_k(b),\,d(a,b)\},\quad \operatorname{lrd}(a)=\Big(\frac1k\sum_{b\in N_k(a)}\operatorname{reach}_k(a,b)\Big)^{-1},\quad \operatorname{LOF}(a)=\frac{1}{k}\sum_{b\in N_k(a)}\frac{\operatorname{lrd}(b)}{\operatorname{lrd}(a)}`, "d_k(b): distance from b to its k-th neighbour · lrd: local reachability density (inverse average reach distance) · LOF about 1: as dense as its neighbours; much more than 1: outlier"],
    ["h", "One-Class SVM"],
    ["p", R`Train on normal data only and find a boundary around it in a [kernel](classical.html#kernel-methods) feature space: the hyperplane that separates the data from the origin with the largest margin,`],
    ["math", R`\min_{\mathbf w,\xi,\rho}\;\tfrac12\|\mathbf w\|^2+\frac{1}{\nu n}\sum_i\xi_i-\rho\quad\text{s.t.}\quad \mathbf w^\top\phi(\mathbf x_i)\ge\rho-\xi_i,\;\xi_i\ge0`, "ν in (0, 1]: an upper bound on the fraction of training points treated as outliers, and a lower bound on the fraction of support vectors"],
    ["p", "With an RBF kernel this becomes a smooth blob around the dense part of the data. It is sensitive to the kernel width `gamma` and to feature scaling."],
    ["h", "From scratch in NumPy"],
    ["p", "First an Isolation Forest: random trees on random sub-samples, path length with the correction term $c(n)$ for leaves that still hold several points, and the score formula. The data are 250 normal 2-D points and five planted outliers (indices 250 to 254)."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(0, 1, (250, 2)), [[6, 6], [-5, 6], [7, -4], [0, 8], [-6, -6]]])   # 250 normal + 5 outliers
n = len(X)

def c(m):                                         # average path length of an unsuccessful BST search
    return 0.0 if m <= 1 else 2 * (np.log(m - 1) + 0.5772156649) - 2 * (m - 1) / m

def build(idx, depth, limit):
    if depth >= limit or len(idx) <= 1: return ("leaf", len(idx), depth)
    f = int(rng.integers(2)); lo, hi = X[idx, f].min(), X[idx, f].max()
    if lo == hi: return ("leaf", len(idx), depth)
    q = rng.uniform(lo, hi)                       # random split value
    return ("split", f, q, build(idx[X[idx, f] < q], depth + 1, limit), build(idx[X[idx, f] >= q], depth + 1, limit))

def path_length(x, node):
    if node[0] == "leaf": return node[2] + c(node[1])        # depth + correction for the unresolved points
    _, f, q, left, right = node
    return path_length(x, left if x[f] < q else right)

psi = 128                                         # sub-sample size per tree
limit = int(np.ceil(np.log2(psi)))
trees = [build(rng.choice(n, psi, replace=False), 0, limit) for _ in range(100)]
h = np.array([[path_length(x, t) for t in trees] for x in X]).mean(axis=1)
score = 2 ** (-h / c(psi))
print("c(128) =", round(c(psi), 2))
print("mean path length  normal:", round(float(h[:250].mean()), 2), " outliers:", round(float(h[250:].mean()), 2))
print("mean score        normal:", round(float(score[:250].mean()), 3), " outliers:", round(float(score[250:].mean()), 3))
print("5 highest scores are points:", sorted(np.argsort(score)[::-1][:5].tolist()))`, "c(128) = 8.86\nmean path length  normal: 10.92  outliers: 3.35\nmean score        normal: 0.428  outliers: 0.77\n5 highest scores are points: [250, 251, 252, 253, 254]"],
    ["p", "Now LOF from scratch. The data have a dense cluster, a sparse cluster, and one point that sits close to the dense cluster but outside it. A plain 'distance to the 5th neighbour' score ranks that point low (the sparse cluster's members are farther from their neighbours), but LOF catches it."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
dense = rng.normal(0, 0.3, (100, 2))
sparse = rng.normal(0, 1.5, (100, 2)) + [8, 0]
X = np.vstack([dense, sparse, [[1.4, 0.0]]])                    # last point: just outside the dense cluster
n, k = len(X), 5

D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
nn = np.argsort(D, axis=1)[:, :k]                              # k nearest neighbours of each point
kdist = D[np.arange(n), nn[:, -1]]                             # distance to the k-th neighbour
reach = np.maximum(D[np.arange(n)[:, None], nn], kdist[nn])    # reach-dist(a, b) = max(kdist(b), d(a, b))
lrd = 1 / reach.mean(axis=1)                                   # local reachability density
lof = lrd[nn].mean(axis=1) / lrd                               # neighbours' density / own density

p = n - 1
print("LOF of the suspicious point:", round(float(lof[p]), 2))
print("LOF, typical dense point:", round(float(np.median(lof[:100])), 2), " typical sparse point:", round(float(np.median(lof[100:200])), 2))
print("k-distance rank of the suspicious point (1 = most distant):", int((kdist > kdist[p]).sum()) + 1, "of", n)
print("LOF rank of the suspicious point:", int((lof > lof[p]).sum()) + 1)`, "LOF of the suspicious point: 4.22\nLOF, typical dense point: 1.05  typical sparse point: 1.04\nk-distance rank of the suspicious point (1 = most distant): 25 of 201\nLOF rank of the suspicious point: 1"],
    ["h", "The scikit-learn version"],
    ["p", "All three detectors share the same fit and predict convention: `+1` is normal, `-1` is an anomaly. Scale the features first for the distance- and kernel-based ones."],
    ["code", R`
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(0, 1, (250, 2)), [[6, 6], [-5, 6], [7, -4], [0, 8], [-6, -6]]])
Xs = StandardScaler().fit_transform(X)

detectors = {"IsolationForest": IsolationForest(n_estimators=100, contamination=0.02, random_state=0),
             "LocalOutlierFactor": LocalOutlierFactor(n_neighbors=20, contamination=0.02),
             "OneClassSVM": OneClassSVM(nu=0.02, gamma="scale")}
for name, det in detectors.items():
    flagged = np.where(det.fit_predict(Xs) == -1)[0]
    print(name, ": flagged", len(flagged), " planted outliers caught", int((flagged >= 250).sum()), "of 5")

iso = detectors["IsolationForest"]
print("score_samples (lower = more anomalous): outlier", round(float(iso.score_samples(Xs)[250]), 3), " typical", round(float(np.median(iso.score_samples(Xs)[:250])), 3))`, "IsolationForest : flagged 6  planted outliers caught 5 of 5\nLocalOutlierFactor : flagged 6  planted outliers caught 5 of 5\nOneClassSVM : flagged 14  planted outliers caught 2 of 5\nscore_samples (lower = more anomalous): outlier -0.797  typical -0.389"],
    ["table", ["", "Isolation Forest", "LOF", "One-Class SVM"], [
      ["Idea", "short random-cut paths", "lower local density than neighbours", "boundary around normal data"],
      ["Cost", "O(n) per tree, very fast", "O(n²) neighbour search (faster with trees)", "O(n²) to O(n³) training"],
      ["Handles varying density", "moderately", "**yes** (its purpose)", "depends on the kernel"],
      ["Needs scaling", "not really", "yes", "yes"],
      ["High dimensions", "ok", "distances become less meaningful", "ok with a good kernel"],
      ["Key setting", "`contamination`, `n_estimators`", "`n_neighbors`", "`nu`, `gamma`"],
    ]],
    ["danger", "Treating `contamination` (or `nu`) as something the algorithm discovers. It is a **threshold** you choose: it only decides what fraction of points gets labelled -1. The scores are what carry information; set the threshold from a validation set with known anomalies, from a precision/recall trade-off ([PR curves](ml.html#roc-auc)), or from what the business can afford to review. Another trap: LOF's `predict` on new data needs `novelty=True` (fit on clean data only)."],
    ["tip", "Evaluate with ranking metrics, not accuracy: with 0.1% anomalies, a model that flags nothing is 99.9% accurate. Use precision-recall or ROC-AUC on a labelled sample when you have one."],
  ],
},
{
  id: "apriori", group: "Other Classics", title: "Association Rules (Apriori)",
  sub: "Find items that are often bought together, and turn them into rules such as \"bread and butter → milk\" with measured support, confidence and lift.",
  tags: ["pattern mining", "market basket"],
  blocks: [
    ["p", "A supermarket has millions of till receipts. Each receipt is a **transaction**: a set of items. **Association rule mining** asks which items tend to appear together. The classic answer ('diapers and beer') is a story, but the technique is real and used for shop layout, recommendations (\"customers also bought\"), web-page click patterns and finding co-occurring symptoms or alarms. There is no label and no target; it is pure pattern discovery."],
    ["h", "Three numbers"],
    ["p", R`Let $N$ be the number of transactions, and write $\operatorname{count}(X)$ for how many transactions contain every item of the itemset $X$. For a rule $A\Rightarrow B$ ("if the basket has $A$, it often has $B$"):`],
    ["math", R`\operatorname{support}(A\Rightarrow B)=\frac{\operatorname{count}(A\cup B)}{N},\quad \operatorname{confidence}=\frac{\operatorname{support}(A\cup B)}{\operatorname{support}(A)}=P(B\mid A),\quad \operatorname{lift}=\frac{\operatorname{confidence}}{\operatorname{support}(B)}`, "support: how common · confidence: how reliable (a conditional probability) · lift: how much better than chance"],
    ["list", [
      "**Support** says how often the pattern happens at all. Rare patterns are usually noise or useless, so we keep only itemsets above a **minimum support**.",
      "**Confidence** is the probability of $B$ given $A$. It can be high just because $B$ is very common (everyone buys bread), which is why we need lift.",
      "**Lift** divides by $B$'s baseline frequency. Lift $=1$: $A$ and $B$ are independent; $>1$: they occur together more than chance; $<1$: they avoid each other.",
    ]],
    ["note", R`Example. 10 000 baskets; 2 000 have milk, 3 000 have bread, 1 500 have both. Rule $\text{bread}\Rightarrow\text{milk}$: support $=1500/10000=0.15$, confidence $=1500/3000=0.5$, lift $=0.5/0.2=2.5$. Bread buyers are two and a half times as likely as a random shopper to buy milk.`],
    ["h", "The Apriori algorithm"],
    ["p", R`With $d$ distinct items there are $2^d-1$ possible itemsets, far too many to count. **Apriori** uses one observation, the **downward-closure (Apriori) property**: *every subset of a frequent itemset is frequent*. Equivalently, if an itemset is infrequent, all its supersets are infrequent and never need to be counted.`],
    ["diagram", "cml2_lattice", [], "Itemset lattice. {D} fails the minimum support, so every superset containing D (dashed red) is pruned without counting."],
    ["steps", [
      "**Level 1**: count every single item and keep those with support at least `minsup`. Call this set $L_1$.",
      "**Generate candidates**: join frequent $k$-itemsets that share $k-1$ items to form $(k+1)$-itemsets.",
      "**Prune**: discard any candidate that has an infrequent $k$-subset (the Apriori property).",
      "**Count** the survivors in one pass over the data; keep those that reach `minsup`: $L_{k+1}$. Repeat until no candidates remain.",
      "**Make rules**: for every frequent itemset $S$ and every split $S=A\\cup B$, keep $A\\Rightarrow B$ if its confidence is at least `minconf`. (Rules from one itemset are cheap: no more data passes, just divisions of supports we already know.)",
    ]],
    ["h", "From scratch in Python"],
    ["p", "The classic five-transaction textbook example. Minimum support 0.6 (at least 3 of 5 baskets), minimum confidence 0.7."],
    ["code", R`
from itertools import combinations

T = [{"bread", "milk"},
     {"bread", "diaper", "beer", "eggs"},
     {"milk", "diaper", "beer", "cola"},
     {"bread", "milk", "diaper", "beer"},
     {"bread", "milk", "diaper", "cola"}]
N, minsup, minconf = len(T), 0.6, 0.7

def support(s): return sum(s <= t for t in T) / N                 # s <= t: s is a subset of the transaction

items = sorted({i for t in T for i in t})
Lk = {frozenset([i]): support(frozenset([i])) for i in items}
Lk = {s: v for s, v in Lk.items() if v >= minsup}
freq, k = {}, 1
while Lk:
    freq.update(Lk)
    print(f"L{k} (frequent {k}-itemsets):", sorted(tuple(sorted(s)) for s in Lk))
    cands = {a | b for a in Lk for b in Lk if len(a | b) == k + 1}                      # join
    cands = {c for c in cands if all(frozenset(sub) in Lk for sub in combinations(c, k))}   # prune (Apriori property)
    print("   candidates left after pruning:", len(cands))
    Lk = {c: support(c) for c in cands if support(c) >= minsup}
    k += 1

print("rules (antecedent -> consequent, support, confidence, lift):")
rules = []
for S in sorted(freq, key=sorted):
    if len(S) < 2: continue
    for r in range(1, len(S)):
        for A in map(frozenset, combinations(sorted(S), r)):
            B = S - A; conf = freq[S] / freq[A]
            if conf >= minconf: rules.append((sorted(A), sorted(B), freq[S], conf, conf / freq[B]))
for A, B, s, c, l in sorted(rules, key=lambda r: (-r[3], str(r[0]), str(r[1]))):
    print(" ", A, "->", B, " support", round(s, 2), " confidence", round(c, 2), " lift", round(l, 2))`, "L1 (frequent 1-itemsets): [('beer',), ('bread',), ('diaper',), ('milk',)]\n   candidates left after pruning: 6\nL2 (frequent 2-itemsets): [('beer', 'diaper'), ('bread', 'diaper'), ('bread', 'milk'), ('diaper', 'milk')]\n   candidates left after pruning: 1\nrules (antecedent -> consequent, support, confidence, lift):\n  ['beer'] -> ['diaper']  support 0.6  confidence 1.0  lift 1.25\n  ['bread'] -> ['diaper']  support 0.6  confidence 0.75  lift 0.94\n  ['bread'] -> ['milk']  support 0.6  confidence 0.75  lift 0.94\n  ['diaper'] -> ['beer']  support 0.6  confidence 0.75  lift 1.25\n  ['diaper'] -> ['bread']  support 0.6  confidence 0.75  lift 0.94\n  ['diaper'] -> ['milk']  support 0.6  confidence 0.75  lift 0.94\n  ['milk'] -> ['bread']  support 0.6  confidence 0.75  lift 0.94\n  ['milk'] -> ['diaper']  support 0.6  confidence 0.75  lift 0.94"],
    ["p", "{diaper} → {beer} has confidence 0.75 (3 of the 4 diaper baskets contain beer) and lift 1.25; the reverse rule {beer} → {diaper} has confidence 1.0 (all 3 beer baskets contain diapers) and the same lift. Rules such as bread → milk have lift below 1: bread buyers are slightly *less* likely than average to buy milk, a reminder that high confidence alone means little. Note how {bread, milk, diaper} was never counted at all as a full triple that passes: its pairs are frequent, so it became a candidate, but only 2 of 5 baskets contain all three. With realistic data the pruning saves most of the counting."],
    ["code", R`
# pip install mlxtend pandas
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth

T = [["bread", "milk"], ["bread", "diaper", "beer", "eggs"], ["milk", "diaper", "beer", "cola"],
     ["bread", "milk", "diaper", "beer"], ["bread", "milk", "diaper", "cola"]]
te = TransactionEncoder()
df = pd.DataFrame(te.fit(T).transform(T), columns=te.columns_)         # one-hot basket table
print(apriori(df, min_support=0.6, use_colnames=True))                 # frequent itemsets
print(fpgrowth(df, min_support=0.6, use_colnames=True))                # same result, faster on big data
# association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7) then turns them into rules`, "", "mlxtend", "local"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Simple, interpretable if-then rules", "Explosion of candidates when `minsup` is low or items are many"],
      ["No labels needed; the Apriori property prunes hard", "Many passes over the data (one per itemset size); FP-Growth needs only two"],
      ["Support / confidence / lift are easy to explain to non-experts", "Finds correlations, not causes; most rules are obvious or trivial"],
      ["Works on any 'basket' data: purchases, clicks, symptoms, alarms", "Ignores quantity, order and time; rare but valuable items get filtered out by `minsup`"],
    ]],
    ["danger", "Trusting **confidence** alone. If 90% of all baskets contain bread, then 'milk → bread' has confidence near 0.9 whether or not the two are related. Check **lift** (and a sensible minimum support) before believing a rule, and remember that with thousands of items some strong-looking rules appear by chance."],
    ["tip", "For big data use **FP-Growth** (builds a compressed prefix tree instead of generating candidates) or Eclat; the results are identical to Apriori's. For sequential patterns (what comes next) use sequence mining; for recommendations with ratings see [matrix factorization](classical.html#recsys-mf). Practice: [frequent itemsets and rules](practice.html#p/apriori-rules)."],
  ],
},
{
  id: "kalman", group: "Other Classics", title: "The Kalman Filter",
  sub: "Track a hidden quantity over time by predicting where it should be, measuring it with a noisy sensor, and blending the two according to how much each can be trusted.",
  tags: ["state estimation", "signals"],
  blocks: [
    ["p", "A GPS reading jumps around by several metres. Your car, though, cannot teleport: it moves smoothly according to physics. If you only believed the GPS you would get a jittery track; if you only believed the physics you would drift. The **Kalman filter** (1960) does the optimal thing: it **predicts** where the car should be from its previous position and speed, **measures** with the noisy sensor, and **combines** the two with weights that depend on how uncertain each is. It sent Apollo to the Moon and is in every phone, drone, robot and navigation system."],
    ["diagram", "cml2_kalmanloop", [], "The Kalman cycle: predict (uncertainty grows), measure, update (uncertainty shrinks), repeat."],
    ["viz", "cml2_kalman"],
    ["p", "Step through the filter. Each time step has two moves: the purple hollow dot is the **prediction** (with an error bar that has grown), the amber dot is the new **measurement**, and the green dot is the corrected **estimate** with a narrower band. Then play with the sliders: tell the filter the sensor is very noisy (large R) and it smooths heavily and lags; tell it the motion is very uncertain (large q) and it chases every measurement."],
    ["h", "The model"],
    ["p", R`The Kalman filter assumes a **linear-Gaussian state-space model**: a hidden state $\mathbf x_t$ (position and velocity) evolves linearly with Gaussian noise, and each measurement $\mathbf z_t$ is a linear function of it plus Gaussian noise:`],
    ["math", R`\mathbf x_t=\mathbf F\,\mathbf x_{t-1}+\mathbf w_t,\;\; \mathbf w_t\sim\mathcal N(\mathbf 0,\mathbf Q)\qquad\qquad \mathbf z_t=\mathbf H\,\mathbf x_t+\mathbf v_t,\;\; \mathbf v_t\sim\mathcal N(\mathbf 0,\mathbf R)`, "F: how the state moves · Q: process noise (how much the real motion deviates from F) · H: what the sensor sees · R: measurement noise"],
    ["p", R`For constant-velocity motion in one dimension with time step 1: $\mathbf x=(\text{position},\text{velocity})$, $\mathbf F=\begin{pmatrix}1&1\\0&1\end{pmatrix}$ (position += velocity), $\mathbf H=(1\;\;0)$ (the sensor sees only position). Because everything is Gaussian, our belief about the state is always a Gaussian $\mathcal N(\hat{\mathbf x},\mathbf P)$ and we only track its mean and covariance.`],
    ["h", "Predict and update"],
    ["steps", [
      R`**Predict** (push the belief through the motion model): $\hat{\mathbf x}^-=\mathbf F\hat{\mathbf x}$ and $\mathbf P^-=\mathbf F\mathbf P\mathbf F^\top+\mathbf Q$. The mean moves, the covariance grows by $\mathbf Q$.`,
      R`**Innovation**: the surprise $\mathbf y=\mathbf z-\mathbf H\hat{\mathbf x}^-$, with covariance $\mathbf S=\mathbf H\mathbf P^-\mathbf H^\top+\mathbf R$.`,
      R`**Kalman gain**: $\mathbf K=\mathbf P^-\mathbf H^\top\mathbf S^{-1}$.`,
      R`**Update**: $\hat{\mathbf x}=\hat{\mathbf x}^-+\mathbf K\mathbf y$ and $\mathbf P=(\mathbf I-\mathbf K\mathbf H)\mathbf P^-$. The covariance shrinks.`,
    ]],
    ["p", R`The gain is the heart of it. In the simplest scalar case ($H=1$) it is $K=\dfrac{P^-}{P^-+R}$, the prior's share of the total uncertainty:`],
    ["list", [
      R`If the sensor is very noisy ($R\gg P^-$): $K\to0$, the filter ignores the measurement.`,
      R`If the prediction is very uncertain ($P^-\gg R$): $K\to1$, the filter jumps to the measurement.`,
      R`In between, the estimate is a weighted average, and the new variance $(1-K)P^-$ is **smaller than both** $P^-$ and $R$: combining two noisy sources always beats either one.`,
    ]],
    ["note", R`Worked scalar example. Prior belief: position $\hat x^-=10$ with variance $P^-=4$. Sensor reads $z=12$ with variance $R=1$. Then $K=\frac{4}{4+1}=0.8$, estimate $=10+0.8(12-10)=11.6$, variance $=(1-0.8)\cdot4=0.8$. The estimate sits close to the more trusted measurement, and the variance 0.8 is smaller than both 4 and 1. This is exactly the product of two Gaussians (precision-weighted average), the same fact that makes [Bayesian updating](classical.html#bayes-linreg) with Gaussians closed-form.`],
    ["p", R`**Where it sits in the family.** The Kalman filter is exact Bayesian inference ([Bayes' theorem](math.html#bayes)) for a linear-Gaussian model, repeated at each step: predict is the prior for the next step, update is multiplying by the likelihood. It is the continuous-state cousin of the [HMM forward algorithm](classical.html#hmm). For non-linear dynamics there are the **extended** (linearise with Jacobians), **unscented** (propagate sample points) and **particle** filters (no Gaussian assumption).`],
    ["h", "From scratch in NumPy"],
    ["p", "First the scalar example above, then a full constant-velocity tracker. The true object moves with small random accelerations; the sensor sees its position plus noise of standard deviation 3 (variance R = 9). We compare the error of the raw measurements with the filtered estimates, and with two badly tuned filters."],
    ["code", R`
import numpy as np

# scalar check: prior N(10, 4), measurement 12 with variance 1
P_prior, R_s = 4.0, 1.0
K = P_prior / (P_prior + R_s)
print("scalar example: gain", K, " estimate", 10 + K * (12 - 10), " variance", round((1 - K) * P_prior, 2))

rng = np.random.default_rng(0)
T, q_true, R_true = 100, 0.01, 9.0
F = np.array([[1.0, 1.0], [0.0, 1.0]]); H = np.array([[1.0, 0.0]])
G = np.array([0.5, 1.0])                                         # an acceleration a moves (pos, vel) by (a/2, a)
x = np.zeros((T, 2)); x[0] = [0.0, 1.0]
for t in range(1, T):
    x[t] = F @ x[t - 1] + G * rng.normal(0, np.sqrt(q_true))     # truth: constant velocity + random acceleration
z = x[:, 0] + rng.normal(0, np.sqrt(R_true), T)                  # noisy position measurements

def kalman(z, q, R):
    Q = q * np.outer(G, G)
    xh = np.array([z[0], 0.0]); P = np.diag([R, 4.0]); est = [xh[0]]
    for t in range(1, len(z)):
        xh = F @ xh; P = F @ P @ F.T + Q                         # predict
        S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)      # innovation covariance and gain
        xh = xh + (K * (z[t] - (H @ xh)[0])).ravel()             # update the mean
        P = (np.eye(2) - K @ H) @ P                              # update the covariance
        est.append(xh[0])
    return np.array(est), K, P

rmse = lambda a: round(float(np.sqrt(((a - x[:, 0]) ** 2).mean())), 3)
est, K, P = kalman(z, q_true, R_true)
print("RMSE of the raw measurements:        ", rmse(z))
print("RMSE of the Kalman estimate (tuned): ", rmse(est))
print("RMSE, believes the sensor too much (R = 0.1):  ", rmse(kalman(z, q_true, 0.1)[0]))
print("RMSE, believes the model too much (q = 1e-6):  ", rmse(kalman(z, 1e-6, R_true)[0]))
print("steady-state gain on position:", round(float(K[0, 0]), 3), " position std:", round(float(np.sqrt(P[0, 0])), 3))`, "scalar example: gain 0.8  estimate 11.6  variance 0.8\nRMSE of the raw measurements:         2.896\nRMSE of the Kalman estimate (tuned):  1.823\nRMSE, believes the sensor too much (R = 0.1):   2.086\nRMSE, believes the model too much (q = 1e-6):   7.137\nsteady-state gain on position: 0.227  position std: 1.431"],
    ["p", "The tuned filter clearly beats the raw sensor (its error is about a third lower). Tuned too trusting of the sensor, it largely copies the noise. Tuned too trusting of the model, it stops following real changes in velocity. The ratio of Q to R is the one knob that matters."],
    ["code", R`
# pip install filterpy
import numpy as np
from filterpy.kalman import KalmanFilter

kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([0.0, 0.0])                                    # state: position, velocity
kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])                      # motion model
kf.H = np.array([[1.0, 0.0]])                                  # we measure position
kf.P *= 4.0                                                    # initial uncertainty
kf.R = np.array([[4.0]])                                       # measurement noise
kf.Q = 0.05 * np.outer([0.5, 1.0], [0.5, 1.0])                 # process noise

for z in [1.2, 2.5, 2.9, 4.4]:
    kf.predict()
    kf.update(z)
    print(kf.x)`, "", "filterpy", "local"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Optimal (minimum variance) for linear-Gaussian models", "Needs a model F, H and noise levels Q, R; wrong ones hurt"],
      ["Recursive: O(1) memory per step, real time, tiny cost", "Linear and Gaussian only (use EKF / UKF / particle filter otherwise)"],
      ["Gives uncertainty, not just a point estimate", "One bad outlier measurement can derail it unless you gate on the innovation"],
      ["Fuses several sensors (stack them in z) and handles missing data (skip the update)", "Q is hard to know in practice and is usually tuned by hand"],
    ]],
    ["danger", "Setting Q too small. The filter becomes sure of itself, the gain falls towards zero, and it stops listening to the data: it is smooth but wrong when the world changes. If the innovations $\\mathbf y_t$ are not roughly white with covariance $\\mathbf S_t$, the model is mis-specified."],
    ["tip", "Practice: [one Kalman predict and update step](practice.html#p/kalman-step). Related: [Gaussian processes](classical.html#gaussian-processes) are another way to smooth noisy time-indexed data, and the [HMM](classical.html#hmm) handles discrete hidden states."],
  ],
},
{
  id: "time-series", group: "Other Classics", title: "Time Series: AR, MA & ARIMA",
  sub: "Forecast a series from its own past: autoregressive models use past values, moving-average models use past errors, ARIMA adds differencing to remove trends.",
  tags: ["forecasting", "sequences"],
  blocks: [
    ["p", "Daily sales, temperature, heart rate, web traffic: a **time series** is data in order, and the order is the information. Today is related to yesterday. That breaks the assumption behind most of the other models on this site (independent rows), and it changes how you split and validate. Classical time-series models make the dependence explicit and give you forecasts with error bars from just the series itself."],
    ["list", [
      "**Components**: trend (slow drift), seasonality (a repeating cycle such as weekly), and the irregular remainder. Models like ARIMA describe the remainder after removing trend and seasonality.",
      "**Stationarity**: a series is (weakly) stationary if its mean and variance do not drift and its correlations depend only on the **lag** between two times, not on when. Most classical models need it. A trending series is not stationary; its **difference** $y_t-y_{t-1}$ often is.",
      "**White noise**: independent values with constant mean and variance. If a model's residuals are white noise, there is nothing left to predict.",
    ]],
    ["h", "AR, MA and ARIMA"],
    ["math", R`\text{AR}(p):\;\; y_t=c+\varphi_1y_{t-1}+\dots+\varphi_py_{t-p}+\varepsilon_t\qquad\qquad \text{MA}(q):\;\; y_t=\mu+\varepsilon_t+\theta_1\varepsilon_{t-1}+\dots+\theta_q\varepsilon_{t-q}`, "ε_t: white noise with variance σ² · φ_i: autoregressive coefficients · θ_j: moving-average coefficients"],
    ["p", R`An **AR** model is a linear regression of the series on its own past (so you can fit it with [least squares](ml.html#linear-regression)). An **MA** model says the series is a weighted sum of recent random shocks, so a shock echoes for exactly $q$ steps and then disappears. **ARMA**$(p,q)$ combines both. **ARIMA**$(p,d,q)$ first differences the series $d$ times and then fits an ARMA, so $d=1$ models a series that has a drifting level. Seasonal versions (SARIMA) add the same terms at the seasonal lag.`],
    ["viz", "cml2_ar"],
    ["p", R`Try it. With $\varphi_2=0$ it is an AR(1): a large positive $\varphi_1$ gives long, smooth swings and a slowly decaying ACF; a negative $\varphi_1$ makes the series zig-zag and the ACF alternate in sign. Give $\varphi_2$ a negative value with a positive $\varphi_1$ and the series acquires a pseudo-cycle (the ACF becomes a damped wave). Push the coefficients outside the triangle and the series is no longer stationary.`],
    ["h", "Stationarity and the ACF"],
    ["p", R`For an AR(1) the series is stationary when $|\varphi|<1$. Then the variance is $\sigma^2/(1-\varphi^2)$ and the autocorrelation at lag $k$ is simply $\rho_k=\varphi^k$. For $\varphi=0.8$ and $\sigma=1$: variance $=1/0.36=2.78$, $\rho_1=0.8$, $\rho_2=0.64$, $\rho_3=0.51$: a geometric decay. At $\varphi=1$ (a **random walk**) the variance grows without bound.`],
    ["p", R`The **autocorrelation function (ACF)** plots $\rho_k=\operatorname{corr}(y_t,y_{t-k})$ against the lag $k$. The **partial ACF (PACF)** gives the correlation at lag $k$ after removing the effect of the lags in between (it is the last coefficient of an AR($k$) fit). Their shapes identify the model:`],
    ["table", ["Model", "ACF", "PACF"], [
      ["AR(p)", "tails off (decays or oscillates)", "**cuts off** after lag p"],
      ["MA(q)", "**cuts off** after lag q", "tails off"],
      ["ARMA(p, q)", "tails off", "tails off"],
      ["Random walk / trend (non-stationary)", "decays very slowly, stays near 1", "one big spike at lag 1"],
    ]],
    ["p", R`For AR models the coefficients also satisfy the **Yule-Walker equations** linking them to the ACF. For AR(2): $\rho_1=\varphi_1/(1-\varphi_2)$ and $\rho_k=\varphi_1\rho_{k-1}+\varphi_2\rho_{k-2}$ for $k\ge2$ (the amber rings in the picture). For MA(1), $\rho_1=\theta/(1+\theta^2)$ and $\rho_k=0$ for $k\ge2$.`],
    ["h", "From scratch in NumPy"],
    ["p", "Simulate an AR(2), then recover its coefficients two ways: ordinary least squares on lagged columns and the Yule-Walker equations. Finally compare one-step forecasts against two baselines on the last 100 points, the way you should always evaluate: on the **end** of the series, never on shuffled rows."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
n, phi1, phi2, sigma = 600, 0.5, 0.3, 1.0
y = np.zeros(n)
for t in range(2, n):
    y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + rng.normal(0, sigma)
train, test = y[:500], y[500:]

# fit AR(2) by least squares: regress y_t on [y_{t-1}, y_{t-2}, 1]
A = np.column_stack([train[1:-1], train[:-2], np.ones(len(train) - 2)])
coef = np.linalg.lstsq(A, train[2:], rcond=None)[0]
print("least squares  phi1, phi2, c:", coef.round(3).tolist(), " (true 0.5, 0.3, 0)")

# Yule-Walker: solve [[1, r1], [r1, 1]] phi = [r1, r2] using sample autocorrelations
d = train - train.mean()
r = lambda k: (d[k:] * d[:len(d) - k]).sum() / (d * d).sum()
phi_yw = np.linalg.solve([[1, r(1)], [r(1), 1]], [r(1), r(2)])
print("Yule-Walker    phi1, phi2:   ", phi_yw.round(3).tolist())

# one-step-ahead forecasts on the test part
hist = np.r_[train, test]
pred = coef[0] * hist[499:-1] + coef[1] * hist[498:-2] + coef[2]
naive = hist[499:-1]                                              # tomorrow = today
mean_ = np.full(100, train.mean())                                # tomorrow = long-run average
rmse = lambda p: round(float(np.sqrt(((p - test) ** 2).mean())), 3)
print("one-step RMSE  AR(2):", rmse(pred), " naive:", rmse(naive), " mean:", rmse(mean_))

# multi-step forecast: feed predictions back in; it decays towards the mean
f = list(train[-2:])
for _ in range(10): f.append(coef[0] * f[-1] + coef[1] * f[-2] + coef[2])
print("10-step forecast:", np.round(f[2:], 2).tolist())`, "least squares  phi1, phi2, c: [0.47, 0.31, -0.027]  (true 0.5, 0.3, 0)\nYule-Walker    phi1, phi2:    [0.47, 0.309]\none-step RMSE  AR(2): 0.905  naive: 1.029  mean: 1.255\n10-step forecast: [-0.55, -0.42, -0.4, -0.35, -0.31, -0.28, -0.26, -0.24, -0.22, -0.2]"],
    ["p", "Both estimation methods land close to the true coefficients (with 500 points there is still sampling noise), and the AR(2) model beats the naive and mean forecasts. The multi-step forecast shrinks towards the mean: the further ahead, the less the model knows. Next: how the ACF and PACF identify a model, and what differencing does to a random walk."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(1)
n = 3000
eps = rng.normal(size=n + 1)
ma1 = eps[1:] + 0.7 * eps[:-1]                                   # MA(1) with theta = 0.7
ar2 = np.zeros(n)
for t in range(2, n): ar2[t] = 0.5 * ar2[t - 1] + 0.3 * ar2[t - 2] + eps[t]

def acf(y, K):
    d = y - y.mean(); return np.array([(d[k:] * d[:len(d) - k]).sum() / (d * d).sum() for k in range(1, K + 1)])

def pacf(y, K):                                                  # lag-k PACF = last coefficient of an AR(k) least-squares fit
    out = []
    for k in range(1, K + 1):
        A = np.column_stack([y[k - 1 - j:len(y) - 1 - j] for j in range(k)])
        out.append(np.linalg.lstsq(A, y[k:], rcond=None)[0][-1])
    return np.array(out)

print("MA(1)  ACF :", acf(ma1, 4).round(2).tolist(), " (theory lag 1:", round(0.7 / (1 + 0.49), 2), ", then 0)")
print("MA(1)  PACF:", pacf(ma1 - ma1.mean(), 4).round(2).tolist())
print("AR(2)  ACF :", acf(ar2, 4).round(2).tolist())
print("AR(2)  PACF:", pacf(ar2 - ar2.mean(), 4).round(2).tolist(), " (cuts off after lag 2)")

walk = np.cumsum(rng.normal(size=n)) + 0.02 * np.arange(n)       # random walk with drift: not stationary
print("random walk ACF, lags 1-3:", acf(walk, 3).round(2).tolist())
print("after differencing        :", acf(np.diff(walk), 3).round(2).tolist())`, "MA(1)  ACF : [0.47, -0.0, 0.01, 0.0]  (theory lag 1: 0.47 , then 0)\nMA(1)  PACF: [0.47, -0.28, 0.2, -0.15]\nAR(2)  ACF : [0.71, 0.65, 0.54, 0.46]\nAR(2)  PACF: [0.71, 0.29, 0.01, -0.0]  (cuts off after lag 2)\nrandom walk ACF, lags 1-3: [1.0, 0.99, 0.99]\nafter differencing        : [-0.01, 0.01, -0.01]"],
    ["code", R`
# pip install statsmodels
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

rng = np.random.default_rng(0)
y = np.zeros(600)
for t in range(2, 600):
    y[t] = 0.5 * y[t - 1] + 0.3 * y[t - 2] + rng.normal()

res = ARIMA(y[:500], order=(2, 0, 0)).fit()                    # ARIMA(p, d, q); d = 1 would difference first
print(res.summary())                                           # coefficients, standard errors, AIC / BIC
fc = res.get_forecast(steps=5)
print(fc.predicted_mean)                                       # point forecasts
print(fc.conf_int())                                           # 95% prediction intervals widen with the horizon`, "", "statsmodels", "local"],
    ["h", "How to fit a model in practice (Box-Jenkins)"],
    ["steps", [
      "**Plot** the series. Look for trend, seasonality, jumps, changing variance (a log transform often fixes growing variance).",
      "**Make it stationary**: difference once or twice (never more than needed), and take a seasonal difference if there is a cycle.",
      "**Identify** orders from the ACF and PACF, or fit a few small candidates and compare AIC / BIC.",
      "**Fit** by maximum likelihood (`statsmodels`). **Check the residuals**: they should look like white noise (no ACF spikes).",
      "**Forecast** and evaluate with a **rolling-origin** (time-ordered) split: train on the past, test on what follows, then slide forward.",
    ]],
    ["table", ["Strengths", "Weaknesses"], [
      ["Few parameters, works with short series, forecasts come with intervals", "Linear and (mostly) univariate; no covariates unless you add ARIMAX terms"],
      ["Fast, interpretable, strong baseline that often beats complicated models", "Needs stationarity; trend and seasonality must be handled by hand"],
      ["Theory tells you what the ACF/PACF should look like", "Order selection is partly art; multi-step forecasts revert to the mean"],
    ]],
    ["danger", "Using a **random** train/test split or ordinary [cross-validation](ml.html#cross-validation) on a time series. Future points would train a model that is tested on the past, which is [leakage](ml.html#splits) and makes forecasts look far better than they will be. Always split by time, and always include the **naive forecast** ('tomorrow equals today') as a baseline: a surprising number of fancy models fail to beat it."],
    ["tip", "For hidden-state or smoothly moving signals use the [Kalman filter](classical.html#kalman); an AR model is a special case of a state-space model. For nonlinear patterns, long seasonality or many series, gradient boosting on lag features ([boosting](ml.html#boosting)) and neural models ([RNNs](dl.html#rnn)) are the usual next steps."],
  ],
},
);
