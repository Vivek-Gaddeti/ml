/* Classical ML, part 1: discriminants, kernels and probabilistic models
   Outputs written as "?" are filled in by running the code: node tools/fill.js data/cml-1.js */
CML.push(
{
  id: "lda", group: "Discriminants & Kernels", title: "Linear Discriminant Analysis (Fisher's LDA)",
  sub: "Find the straight line (or flat boundary) that best separates classes by pushing their means apart while keeping each class tight.",
  tags: ["classification", "dimensionality reduction"],
  blocks: [
    ["p", "**What this page is.** The [Core ML](ml.html#what-is-ml) page covers the workhorse algorithms you will use every day. This page collects the **classical, maths-heavy algorithms** that sit behind them and next to them: Gaussian models, kernels, Bayesian regression, EM, hidden Markov models, topic models and MCMC. They are older than deep learning, but their ideas (probabilistic thinking, latent variables, kernels, sampling) show up everywhere in modern ML, and they are favourites in interviews and exams. Each section gives the intuition, the derivation, a from-scratch NumPy version and the library version."],
    ["p", "Let us start with a classifier that comes from two different ideas that turn out to be the same. Imagine two clouds of points, one per class. If you could only look at the data along **one line**, which line would you choose? The line along which the cloud is most spread out (that is [PCA](ml.html#pca))? Not necessarily: PCA ignores the labels, and the biggest spread can be *inside* each class. **Fisher's Linear Discriminant Analysis (LDA)** uses the labels. It picks the line where, after projecting every point onto it, the **class averages are far apart compared with the spread within each class**."],
    ["viz", "cml1_lda"],
    ["p", "Slide the direction. The amber line is what you project onto; the histogram shows where each class lands on it. Near the purple PCA line the two classes overlap almost completely. Near the green LDA line they separate cleanly. The curve in the corner is the score that LDA maximises."],
    ["h", "The maths: Fisher's criterion"],
    ["p", "Take two classes with means $\\boldsymbol\\mu_0, \\boldsymbol\\mu_1$. Project every point onto a direction $\\mathbf w$: the number $\\mathbf w^\\top\\mathbf x$ is its position along the line. Two things matter after projecting: how far apart the class means land, and how much each class is spread out around its own mean."],
    ["math", R`J(\mathbf w)=\frac{(\mathbf w^\top\boldsymbol\mu_1-\mathbf w^\top\boldsymbol\mu_0)^2}{s_0^2+s_1^2}=\frac{\mathbf w^\top \mathbf S_B\,\mathbf w}{\mathbf w^\top \mathbf S_W\,\mathbf w}`, "J: Fisher's score, bigger is better · s_k²: scatter of the projected class k"],
    ["list", [
      "**Within-class scatter** $\\mathbf S_W=\\sum_{k}\\sum_{\\mathbf x\\in k}(\\mathbf x-\\boldsymbol\\mu_k)(\\mathbf x-\\boldsymbol\\mu_k)^\\top$: add up the spread inside each class. The projected scatter of class $k$ is $s_k^2=\\mathbf w^\\top \\mathbf S_k\\mathbf w$, so the denominator equals $\\mathbf w^\\top\\mathbf S_W\\mathbf w$.",
      "**Between-class scatter** $\\mathbf S_B=(\\boldsymbol\\mu_1-\\boldsymbol\\mu_0)(\\boldsymbol\\mu_1-\\boldsymbol\\mu_0)^\\top$: how far apart the class means are. The numerator equals $\\mathbf w^\\top\\mathbf S_B\\mathbf w$.",
    ]],
    ["p", "$J$ only depends on the *direction* of $\\mathbf w$ (scaling $\\mathbf w$ cancels). Set the gradient to zero: $\\mathbf S_B\\mathbf w = J\\,\\mathbf S_W\\mathbf w$, a generalised eigenvalue problem, so the best $\\mathbf w$ is the top eigenvector of $\\mathbf S_W^{-1}\\mathbf S_B$ (see [eigenvectors](math.html#eigen)). For two classes there is a shortcut: $\\mathbf S_B\\mathbf w=(\\boldsymbol\\mu_1-\\boldsymbol\\mu_0)\\,[(\\boldsymbol\\mu_1-\\boldsymbol\\mu_0)^\\top\\mathbf w]$ always points along $\\boldsymbol\\mu_1-\\boldsymbol\\mu_0$, so"],
    ["math", R`\mathbf w \;\propto\; \mathbf S_W^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_0)`, "Fisher's linear discriminant"],
    ["p", "Read it as: the straight line between the two means, **corrected** by the inverse of the within-class scatter. $\\mathbf S_W^{-1}$ down-weights directions in which the classes are naturally very spread out (they are noisy) and boosts directions in which they are tight."],
    ["note", "Tiny example. Let $\\boldsymbol\\mu_1-\\boldsymbol\\mu_0=(1,1)$ and $\\mathbf S_W=\\mathrm{diag}(2,\\,0.5)$. Then $\\mathbf S_W^{-1}=\\mathrm{diag}(0.5,\\,2)$ and $\\mathbf w\\propto(0.5,\\,2)$. The means differ equally in both features, but feature 2 is four times less noisy, so LDA leans on it four times as hard. Plain \"point from mean to mean\" (direction $(1,1)$) ignores this."],
    ["h", "The same thing as a Gaussian classifier"],
    ["p", "There is a second route to the same formula. Suppose each class $k$ generates its data as a Gaussian with its own mean but one **shared** covariance $\\boldsymbol\\Sigma$, and has prior $\\pi_k$. Bayes' rule ([Bayes' theorem](math.html#bayes)) gives the class scores"],
    ["math", R`\delta_k(\mathbf x)=\mathbf x^\top\boldsymbol\Sigma^{-1}\boldsymbol\mu_k-\tfrac12\boldsymbol\mu_k^\top\boldsymbol\Sigma^{-1}\boldsymbol\mu_k+\log\pi_k`, "predict the class with the largest δ_k"],
    ["p", "The quadratic term $-\\frac12\\mathbf x^\\top\\boldsymbol\\Sigma^{-1}\\mathbf x$ is the same for every class, so it cancels, and each $\\delta_k$ is **linear** in $\\mathbf x$. The boundary between two classes ($\\delta_0=\\delta_1$) is a flat line or plane with normal vector $\\boldsymbol\\Sigma^{-1}(\\boldsymbol\\mu_1-\\boldsymbol\\mu_0)$, exactly Fisher's direction. So LDA is both a **projection** (Fisher) and a **generative classifier** (Gaussians with shared covariance). Compare [Naive Bayes](ml.html#naive-bayes), which also models each class with Gaussians but assumes the features are independent (a diagonal covariance)."],
    ["list", [
      "**More than two classes**: the discriminant directions are the top eigenvectors of $\\mathbf S_W^{-1}\\mathbf S_B$, and there are at most $\\min(K-1,\\,d)$ useful ones. LDA can therefore reduce 4 features to 2 for 3 classes (iris), like a supervised version of PCA.",
      "**Estimates**: class means, priors $\\pi_k=n_k/n$, and the pooled covariance $\\hat{\\boldsymbol\\Sigma}=\\mathbf S_W/(n-K)$.",
    ]],
    ["h", "From scratch in NumPy"],
    ["p", "Two Gaussian classes sharing one covariance. We compute Fisher's direction with one linear solve, threshold halfway between the projected means, and compare the Fisher score $J$ along $\\mathbf w$ and along the first principal component."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
cov = np.array([[3.0, 1.2], [1.2, 1.0]])               # shared covariance: long thin clouds
X0 = rng.multivariate_normal([0.0, 0.0], cov, 100)
X1 = rng.multivariate_normal([1.5, 2.5], cov, 100)
X = np.vstack([X0, X1]); y = np.r_[np.zeros(100), np.ones(100)]

m0, m1 = X0.mean(axis=0), X1.mean(axis=0)
Sw = (X0 - m0).T @ (X0 - m0) + (X1 - m1).T @ (X1 - m1)  # within-class scatter
w = np.linalg.solve(Sw, m1 - m0)                         # w ∝ Sw^-1 (m1 - m0), no explicit inverse
w /= np.linalg.norm(w)
print("Fisher direction:", w.round(3))

t = 0.5 * ((X0 @ w).mean() + (X1 @ w).mean())            # threshold between projected means (equal priors)
pred = (X @ w > t).astype(int)
print("training accuracy:", (pred == y).mean())

def J(v):                                                # Fisher score along direction v
    p0, p1 = X0 @ v, X1 @ v
    return (p1.mean() - p0.mean()) ** 2 / (((p0 - p0.mean()) ** 2).sum() + ((p1 - p1.mean()) ** 2).sum())

pc1 = np.linalg.eigh(np.cov(X.T))[1][:, -1]              # first principal component
print("J along Fisher w:", round(float(J(w)), 4), " J along PC1:", round(float(J(pc1)), 4))`, "Fisher direction: [-0.215  0.977]\ntraining accuracy: 0.925\nJ along Fisher w: 0.0374  J along PC1: 0.0119"],
    ["p", "The principal component follows the long axis of the clouds, which is mostly *within-class* spread, so its score is much lower."],
    ["h", "The scikit-learn version"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)
lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
print("classes:", lda.classes_.tolist(), " priors:", lda.priors_.round(3).tolist())
print("share of between-class variance per discriminant:", lda.explained_variance_ratio_.round(3).tolist())
print("projected shape:", lda.transform(X).shape, " training accuracy:", round(lda.score(X, y), 3))

# supervised vs unsupervised 2-D reduction, then 5-NN. The reducer is INSIDE the pipeline so CV is honest.
for name, red in [("PCA(2)", PCA(2)), ("LDA(2)", LinearDiscriminantAnalysis(n_components=2))]:
    pipe = make_pipeline(red, KNeighborsClassifier(5))
    print(name, "5-fold CV accuracy:", cross_val_score(pipe, X, y, cv=5).mean().round(3))

# shrinkage: blend the covariance with a diagonal one. Helps when features are many and samples few.
shr = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
print("shrinkage LDA accuracy:", round(shr.score(X, y), 3))`, "classes: [0, 1, 2]  priors: [0.333, 0.333, 0.333]\nshare of between-class variance per discriminant: [0.991, 0.009]\nprojected shape: (150, 2)  training accuracy: 0.98\nPCA(2) 5-fold CV accuracy: 0.967\nLDA(2) 5-fold CV accuracy: 0.973\nshrinkage LDA accuracy: 0.98"],
    ["table", ["Setting", "Meaning"], [
      ["`solver=\"svd\"` (default)", "Works for classification and `transform`. Does not form the covariance matrix, so it copes with many features."],
      ["`solver=\"lsqr\"` / `\"eigen\"`", "Support `shrinkage` (use `\"auto\"` or a number in [0, 1]). `eigen` can also transform."],
      ["`n_components`", "Number of discriminant directions to keep, at most min(classes − 1, features)."],
      ["`priors`", "Class priors. Defaults to the class frequencies in the training data."],
    ]],
    ["h", "When to use it"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Fast closed form, no iterations, no learning rate", "Assumes every class is Gaussian with the **same** covariance"],
      ["Few parameters, so it works well with small data", "Only linear boundaries (see [QDA](classical.html#qda) for curves)"],
      ["Doubles as a supervised dimensionality reduction", "At most (classes − 1) projection directions"],
      ["Gives probabilities and handles many classes naturally", "Sensitive to outliers; needs shrinkage when features ≥ samples"],
    ]],
    ["danger", "Fitting LDA (or any supervised reducer) on the **whole** dataset and then cross-validating the classifier on the projected data. The projection has already seen the labels of the validation rows, which is [data leakage](ml.html#splits). Put the LDA inside a pipeline so it is refit on each training fold."],
    ["warn", "$\\mathbf S_W$ must be invertible. With more features than samples, or with perfectly correlated features, it is singular and `np.linalg.solve` fails or returns nonsense. Use shrinkage (`shrinkage=\"auto\"`) or reduce dimensions first."],
    ["tip", "LDA is a great baseline: often within a point or two of logistic regression on small tabular data, and it needs no tuning. Try it before anything fancy. Practice: [Fisher's LDA Direction](practice.html#p/fisher-lda-direction)."],
  ],
},
{
  id: "qda", group: "Discriminants & Kernels", title: "QDA & Gaussian Discriminant Analysis",
  sub: "Let every class have its own Gaussian shape. The boundary between classes becomes a curve (a quadratic) instead of a straight line.",
  tags: ["classification"],
  blocks: [
    ["p", "[LDA](classical.html#lda) forces all classes to share one covariance matrix, i.e. the same cloud shape. That is often wrong: healthy patients may all look alike (a tight cloud) while the sick ones vary a lot (a wide cloud). **Quadratic Discriminant Analysis (QDA)** drops the assumption and fits a separate mean *and covariance* per class. Together, LDA, QDA and Naive Bayes form the family of **Gaussian discriminant analysis** (generative classifiers that model each class with a Gaussian)."],
    ["viz", "cml1_qda"],
    ["p", "Drag the slider. When the orange class has the same spread as the blue one, a straight line is best and QDA agrees with LDA. As the orange cloud grows, the blue class ends up as a tight island inside a wide sea: no straight line can separate them, but QDA's boundary wraps around the island."],
    ["h", "The maths"],
    ["p", "Model each class as $\\mathbf x\\mid k\\sim\\mathcal N(\\boldsymbol\\mu_k,\\boldsymbol\\Sigma_k)$ with prior $\\pi_k$ (see [Gaussian distributions](math.html#distributions)). Bayes' rule says to predict the class with the largest posterior $\\pi_k\\,p(\\mathbf x\\mid k)$. Take the log and drop constants common to all classes:"],
    ["math", R`\delta_k(\mathbf x)=-\tfrac12\log|\boldsymbol\Sigma_k|-\tfrac12(\mathbf x-\boldsymbol\mu_k)^\top\boldsymbol\Sigma_k^{-1}(\mathbf x-\boldsymbol\mu_k)+\log\pi_k`, "|Σ_k|: determinant (the volume of the class's ellipse) · the second term is the squared Mahalanobis distance"],
    ["p", "The squared Mahalanobis distance is a distance measured in units of the class's own spread: a point 2 units away is far for a tight class but ordinary for a wide one. The $-\\frac12\\log|\\boldsymbol\\Sigma_k|$ term pays a penalty for wide classes (their density is spread thin). Because each class has its own $\\boldsymbol\\Sigma_k$, the $\\mathbf x^\\top\\boldsymbol\\Sigma_k^{-1}\\mathbf x$ term no longer cancels between classes, so $\\delta_k$ is **quadratic** in $\\mathbf x$. Setting $\\boldsymbol\\Sigma_k=\\boldsymbol\\Sigma$ for all $k$ makes the quadratic parts cancel and gives back LDA."],
    ["note", "Tiny example in 1-D. Class 0 is $\\mathcal N(0,1)$ and class 1 is $\\mathcal N(0,4)$: **the same mean**. LDA sees identical means and has nothing to work with. QDA sets $\\delta_0=\\delta_1$: $-\\frac12x^2=-\\frac12\\log4-\\frac18x^2$, so $\\frac38x^2=\\log 2$ and $|x|=\\sqrt{\\tfrac83\\log2}\\approx1.36$. Inside $\\pm1.36$ predict the narrow class, outside predict the wide one. A two-sided rule from a quadratic."],
    ["table", ["Model", "Covariance", "Parameters per class (d features)", "Boundary"], [
      ["Naive Bayes (Gaussian)", "diagonal, one per class", "2d", "quadratic (axis-aligned ellipses)"],
      ["**LDA**", "full, **shared** by all classes", "d + d(d+1)/2 shared once", "linear"],
      ["**QDA**", "full, **one per class**", "d + d(d+1)/2", "quadratic (curved)"],
    ]],
    ["p", "The price of flexibility is parameters: a full covariance has $d(d+1)/2$ numbers. With 50 features that is 1275 per class, and the estimate becomes noisy or singular unless each class has many more samples than that. A middle road is **regularised discriminant analysis**: shrink each $\\hat{\\boldsymbol\\Sigma}_k$ towards the pooled covariance, $\\hat{\\boldsymbol\\Sigma}_k(\\lambda)=(1-\\lambda)\\hat{\\boldsymbol\\Sigma}_k+\\lambda\\hat{\\boldsymbol\\Sigma}$. In scikit-learn this is `reg_param`."],
    ["h", "From scratch in NumPy"],
    ["p", "Fit means, covariances and priors per class, then evaluate $\\delta_k$ with `slogdet` (the log-determinant, which avoids overflow) and `solve` (instead of an explicit inverse). We check that it agrees with scikit-learn."],
    ["code", R`
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

rng = np.random.default_rng(1)
X0 = rng.multivariate_normal([0, 0], [[1.0, 0.3], [0.3, 0.5]], 150)          # tight class
X1 = rng.multivariate_normal([1, 1], [[4.0, -1.5], [-1.5, 3.0]], 150)         # wide class
X = np.vstack([X0, X1]); y = np.r_[np.zeros(150, int), np.ones(150, int)]

class QDA:
    def fit(self, X, y):
        self.cls = np.unique(y)
        self.mu = [X[y == k].mean(axis=0) for k in self.cls]
        self.S = [np.cov(X[y == k].T) for k in self.cls]                 # one covariance per class
        self.logpi = [np.log(np.mean(y == k)) for k in self.cls]
        return self
    def scores(self, X):
        out = []
        for mu, S, lp in zip(self.mu, self.S, self.logpi):
            d = X - mu
            maha = np.einsum("ij,ij->i", d, np.linalg.solve(S, d.T).T)   # (x-mu)^T S^-1 (x-mu) per row
            out.append(-0.5 * np.linalg.slogdet(S)[1] - 0.5 * maha + lp)
        return np.stack(out, axis=1)
    def predict(self, X):
        return self.cls[self.scores(X).argmax(axis=1)]

q = QDA().fit(X, y)
print("scratch QDA accuracy:", (q.predict(X) == y).mean().round(3))
sk = QuadraticDiscriminantAnalysis().fit(X, y)
print("sklearn accuracy:    ", round(sk.score(X, y), 3))
print("same predictions:    ", bool((q.predict(X) == sk.predict(X)).all()))`, "scratch QDA accuracy: 0.863\nsklearn accuracy:     0.863\nsame predictions:     True"],
    ["h", "The scikit-learn version: LDA vs QDA vs regularised QDA"],
    ["code", R`
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(1)
def make(n):
    X0 = rng.multivariate_normal([0, 0], [[1.0, 0.3], [0.3, 0.5]], n)
    X1 = rng.multivariate_normal([1, 1], [[4.0, -1.5], [-1.5, 3.0]], n)
    return np.vstack([X0, X1]), np.r_[np.zeros(n, int), np.ones(n, int)]

for n in (15, 300):                                       # tiny vs plentiful training data
    X, y = make(n)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    res = {"LDA": LDA(), "QDA": QDA(), "QDA reg 0.3": QDA(reg_param=0.3)}
    print(f"n per class = {n}:", {k: round(m.fit(Xtr, ytr).score(Xte, yte), 3) for k, m in res.items()})`, "n per class = 15: {'LDA': 0.733, 'QDA': 0.733, 'QDA reg 0.3': 0.8}\nn per class = 300: {'LDA': 0.69, 'QDA': 0.86, 'QDA reg 0.3': 0.853}"],
    ["table", ["Choose", "When"], [
      ["**LDA**", "Few samples, many features, or classes that really look alike in shape"],
      ["**QDA**", "Enough samples per class and visibly different class spreads"],
      ["**Regularised QDA / LDA with shrinkage**", "In between. Cross-validate `reg_param` or `shrinkage`."],
      ["[Logistic regression](ml.html#logistic-regression)", "When you do not trust the Gaussian assumption at all: it models $p(y\\mid\\mathbf x)$ directly"],
    ]],
    ["danger", "Using QDA when a class has fewer samples than features (or about as many). Its covariance matrix is then singular, the log-determinant is $-\\infty$, and scikit-learn warns about collinear variables. Use `reg_param`, fewer features, or LDA."],
    ["tip", "The generative route has a bonus: you can sample new points from a fitted class (`rng.multivariate_normal(mu_k, S_k)`) and detect outliers (a tiny Mahalanobis density under every class)."],
  ],
},
{
  id: "bayes-linreg", group: "Discriminants & Kernels", title: "Bayesian Linear Regression",
  sub: "Instead of one best line, keep a whole probability distribution over lines. Predictions come with honest error bars that shrink as data arrives.",
  tags: ["regression", "bayesian"],
  blocks: [
    ["p", "Ordinary [linear regression](ml.html#linear-regression) answers \"what is the best slope and intercept?\" with a single number each. But with three data points, many different lines fit about equally well, and a single answer hides that. The **Bayesian** view keeps all plausible lines, each weighted by how well it explains the data. Start with a **prior** (what you believed about the weights before seeing data), multiply by the **likelihood** (how probable the data is for each choice of weights), and you get the **posterior**. As points arrive the posterior narrows and the uncertainty band around predictions tightens."],
    ["viz", "cml1_bayeslin"],
    ["p", "Press Next (or Play) to feed in points one at a time. Left: the data space, with the posterior mean line, ±2 standard deviations of the prediction (band) and a few lines randomly drawn from the posterior (purple). Right: the weight space, where the blue blob is the posterior over (intercept, slope); it shrinks onto the true weights (green dot). The grey dashed circle is the prior. Raise the prior precision α to pull the weights harder towards zero."],
    ["h", "The maths"],
    ["p", "Model: $y=\\mathbf w^\\top\\boldsymbol\\phi(x)+\\varepsilon$, where $\\boldsymbol\\phi(x)$ is the feature vector (for a line, $(1,x)$) and the noise is Gaussian, $\\varepsilon\\sim\\mathcal N(0,\\beta^{-1})$ with **precision** $\\beta$ (precision = 1 / variance). Put a Gaussian prior on the weights, $\\mathbf w\\sim\\mathcal N(\\mathbf 0,\\alpha^{-1}\\mathbf I)$: small weights are a priori more likely."],
    ["p", "With design matrix $\\boldsymbol\\Phi$ (one row per data point) and targets $\\mathbf y$, Bayes' rule says posterior $\\propto$ likelihood $\\times$ prior. Take logs:"],
    ["math", R`\log p(\mathbf w\mid \mathbf y)= -\frac{\beta}{2}\lVert \mathbf y-\boldsymbol\Phi\mathbf w\rVert^2-\frac{\alpha}{2}\mathbf w^\top\mathbf w+\text{const}`],
    ["p", "This is a quadratic in $\\mathbf w$, so the posterior is a Gaussian. Expand and **complete the square**: the coefficient of $-\\frac12\\mathbf w^\\top(\\cdot)\\mathbf w$ is the new precision matrix and the coefficient of the linear term gives the mean."],
    ["math", R`p(\mathbf w\mid\mathbf y)=\mathcal N(\mathbf m_N,\mathbf S_N),\qquad \mathbf S_N^{-1}=\alpha\mathbf I+\beta\,\boldsymbol\Phi^\top\boldsymbol\Phi,\qquad \mathbf m_N=\beta\,\mathbf S_N\boldsymbol\Phi^\top\mathbf y`, "precisions add up: prior precision α plus β per data point"],
    ["list", [
      "**Mean = ridge regression.** $\\mathbf m_N=(\\boldsymbol\\Phi^\\top\\boldsymbol\\Phi+\\tfrac{\\alpha}{\\beta}\\mathbf I)^{-1}\\boldsymbol\\Phi^\\top\\mathbf y$. This is exactly [ridge](ml.html#regularization) with $\\lambda=\\alpha/\\beta$. An L2 penalty *is* a Gaussian prior; the Bayesian version also gives you the covariance $\\mathbf S_N$.",
      "**Predictions have two sources of uncertainty.** For a new input $x_*$ the prediction is Gaussian with mean $\\mathbf m_N^\\top\\boldsymbol\\phi(x_*)$ and variance $\\sigma_N^2(x_*)=\\dfrac1\\beta+\\boldsymbol\\phi(x_*)^\\top\\mathbf S_N\\,\\boldsymbol\\phi(x_*)$: irreducible noise plus uncertainty about the weights. The second term shrinks with more data; the first never does. It grows when $x_*$ is far from the training data.",
      "**Sequential updating.** Today's posterior is tomorrow's prior. Adding points one by one gives exactly the same answer as fitting them all at once (checked in the code below).",
    ]],
    ["note", "Tiny example. One weight, one feature, so $\\boldsymbol\\Phi=(1,2)^\\top$, targets $\\mathbf y=(1,3)^\\top$, $\\alpha=\\beta=1$. Then $S_N^{-1}=\\alpha+\\beta(1^2+2^2)=6$, so the posterior variance is $1/6$, and $m_N=\\beta S_N(1\\cdot1+2\\cdot3)=7/6\\approx1.167$. Ridge with $\\lambda=1$ gives $(5+1)^{-1}\\cdot7=7/6$: the same."],
    ["h", "Choosing α and β: the evidence"],
    ["p", "How strong should the prior be, and how noisy is the data? Instead of cross-validation, Bayesians maximise the **marginal likelihood** (the *evidence*) $p(\\mathbf y\\mid\\alpha,\\beta)=\\int p(\\mathbf y\\mid\\mathbf w,\\beta)\\,p(\\mathbf w\\mid\\alpha)\\,d\\mathbf w$: the probability of the data averaged over all weights the prior allows. A model that is too flexible spreads its probability over many possible datasets and so gives each one little; a model that is too rigid cannot explain the data. The evidence automatically balances fit against complexity (an Occam's razor). scikit-learn's `BayesianRidge` does this with an iterative scheme."],
    ["h", "From scratch in NumPy"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
w_true, noise = np.array([-0.3, 0.5]), 0.2
x = rng.uniform(-1, 1, 12)
y = w_true[0] + w_true[1] * x + noise * rng.standard_normal(12)
Phi = np.column_stack([np.ones_like(x), x])                # features (1, x)

alpha, beta = 2.0, 1 / noise**2                            # prior precision, noise precision
def posterior(Phi, y, m0, S0inv):                          # works with any prior N(m0, S0)
    SN_inv = S0inv + beta * Phi.T @ Phi                    # precisions add
    SN = np.linalg.inv(SN_inv)
    mN = SN @ (S0inv @ m0 + beta * Phi.T @ y)
    return mN, SN

mN, SN = posterior(Phi, y, np.zeros(2), alpha * np.eye(2))
print("posterior mean:", mN.round(3), " sd of each weight:", np.sqrt(np.diag(SN)).round(3))

# sequential updating: use each posterior as the next prior
m, S = np.zeros(2), np.eye(2) / alpha
for i in range(len(x)):
    m, S = posterior(Phi[i:i + 1], y[i:i + 1], m, np.linalg.inv(S))
print("one-at-a-time gives the same:", np.allclose(m, mN) and np.allclose(S, SN))

for xs in (0.0, 3.0):                                      # prediction at a seen and an unseen region
    phi = np.array([1.0, xs])
    print(f"x* = {xs}: mean {phi @ mN:.3f}, sd {np.sqrt(1 / beta + phi @ SN @ phi):.3f}")

lam = alpha / beta                                         # equals ridge with lambda = alpha/beta
ridge = np.linalg.solve(Phi.T @ Phi + lam * np.eye(2), Phi.T @ y)
print("ridge solution equals posterior mean:", np.allclose(ridge, mN))`, "posterior mean: [-0.351  0.559]  sd of each weight: [0.058 0.085]\none-at-a-time gives the same: True\nx* = 0.0: mean -0.351, sd 0.208\nx* = 3.0: mean 1.325, sd 0.325\nridge solution equals posterior mean: True"],
    ["p", "The predictive standard deviation at $x_*=3$ (far outside the data, which lies in [−1, 1]) is clearly larger than at $x_*=0$. A plain least-squares fit would give both predictions with equal, unearned confidence."],
    ["h", "The scikit-learn version"],
    ["code", R`
import numpy as np
from sklearn.linear_model import BayesianRidge, Ridge

rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, 12)
y = -0.3 + 0.5 * x + 0.2 * rng.standard_normal(12)
X = x[:, None]                                            # BayesianRidge adds the intercept itself

br = BayesianRidge(compute_score=True).fit(X, y)
print("coef:", br.coef_.round(3), " intercept:", round(float(br.intercept_), 3))
print("noise precision alpha_:", round(float(br.alpha_), 1), " weight precision lambda_:", round(float(br.lambda_), 2))
print("estimated noise sd:", round(float(1 / np.sqrt(br.alpha_)), 3), "(true 0.2)")

mean, std = br.predict(np.array([[0.0], [3.0]]), return_std=True)
print("predictions:", mean.round(3), " std:", std.round(3))
print("ridge coef (for comparison):", Ridge(alpha=1.0).fit(X, y).coef_.round(3))`, "coef: [0.555]  intercept: -0.353\nnoise precision alpha_: 26.8  weight precision lambda_: 3.18\nestimated noise sd: 0.193 (true 0.2)\npredictions: [-0.353  1.312]  std: [0.193 0.312]\nridge coef (for comparison): [0.48]"],
    ["p", "Careful with names: in scikit-learn's `BayesianRidge`, `alpha_` is the **noise** precision (our $\\beta$) and `lambda_` is the **weight** precision (our $\\alpha$). The model learns both from the data, so there is no hyperparameter to cross-validate."],
    ["table", ["", "Least squares (MLE)", "Ridge (MAP)", "Bayesian"], [
      ["Weights", "one point estimate", "one point estimate, shrunk", "a full Gaussian distribution"],
      ["Overfits with little data?", "yes", "less", "no: the posterior stays wide"],
      ["Uncertainty on predictions", "none (only a noise estimate)", "none", "yes, grows away from the data"],
      ["Prior strength", "none", "chosen by cross-validation", "can be learned from the evidence"],
      ["Cost", "one solve", "one solve", "one solve plus a covariance"],
    ]],
    ["danger", "Trusting the band blindly. It only reflects uncertainty *within the model*: if the true relationship is curved and you fit a line, the band can be narrow and confidently wrong. Bayesian linear regression is only as good as its basis functions."],
    ["warn", "A single shared α assumes all weights have similar scale, so standardise the features ([scaling](ml.html#feature-engineering)) before fitting."],
    ["tip", "Use polynomial or radial features $\\boldsymbol\\phi(x)$ for curves; with infinitely many such features the model becomes a [Gaussian process](classical.html#gaussian-processes)."],
  ],
},
{
  id: "kernel-methods", group: "Discriminants & Kernels", title: "Kernel Methods: Kernel Trick, Kernel Ridge & Kernel PCA",
  sub: "Many algorithms only need dot products between points. Swap the dot product for a kernel and they work in a huge curved feature space at almost no extra cost.",
  tags: ["kernels"],
  blocks: [
    ["p", "A straight line cannot separate points that sit inside a ring from points outside it. But if you add a new feature, say $x^2$, the same points can be split by a straight line in the **bigger space**. That is the whole idea of kernel methods: *lift the data to a higher-dimensional space where the problem becomes linear*. The trouble is that the lifted space can be enormous (or infinite), so we would never want to compute the coordinates. The **kernel trick** is the observation that we never have to."],
    ["diagram", "cml1_kernellift", [], "Left: 1-D points where the orange ones surround the blue ones, so no single cut works. Right: after the lift φ(x) = (x, x²) a horizontal line separates them."],
    ["h", "The kernel trick"],
    ["p", "Take a feature map $\\boldsymbol\\phi$. A **kernel** is the dot product in the lifted space, computed without ever building the lifted vectors:"],
    ["math", R`k(\mathbf x,\mathbf z)=\boldsymbol\phi(\mathbf x)^\top\boldsymbol\phi(\mathbf z)`],
    ["p", "Example. In 2-D, take $\\boldsymbol\\phi(\\mathbf x)=(x_1^2,\\ \\sqrt2\\,x_1x_2,\\ x_2^2)$. Then $\\boldsymbol\\phi(\\mathbf x)^\\top\\boldsymbol\\phi(\\mathbf z)=x_1^2z_1^2+2x_1x_2z_1z_2+x_2^2z_2^2=(\\mathbf x^\\top\\mathbf z)^2$. A 3-D dot product turned into one cheap squared dot product in 2-D. For the **RBF (Gaussian) kernel** the lifted space is infinite-dimensional, yet the kernel is a one-liner:"],
    ["math", R`k_{\text{RBF}}(\mathbf x,\mathbf z)=\exp\!\big(-\gamma\lVert\mathbf x-\mathbf z\rVert^2\big)\qquad(\gamma=1/2\ell^2)`, "similarity is 1 for identical points and fades to 0 with distance; γ sets how fast"],
    ["list", [
      "**Valid kernel** (Mercer): symmetric, and every Gram matrix $K_{ij}=k(\\mathbf x_i,\\mathbf x_j)$ is positive semi-definite (no negative eigenvalues). Then a feature map $\\boldsymbol\\phi$ exists even if you never write it down.",
      "**Building kernels**: sums and products of kernels, and positive multiples, are kernels. So you can combine \"smooth\" and \"periodic\" similarity.",
      "**Common choices**: linear $\\mathbf x^\\top\\mathbf z$; polynomial $(\\gamma\\mathbf x^\\top\\mathbf z+c)^p$; RBF above; sigmoid (rarely valid). See [SVMs and kernels](ml.html#svm), where the same trick drives the support vector machine.",
    ]],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
x, z = rng.standard_normal(2), rng.standard_normal(2)

def phi(v):                                              # explicit 3-D feature map of the degree-2 kernel
    return np.array([v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])

print("dot product in feature space:", round(float(phi(x) @ phi(z)), 6))
print("kernel (x . z)^2            :", round(float((x @ z) ** 2), 6))

def rbf(A, B, gamma):                                    # Gram matrix between rows of A and rows of B
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d2)

X = rng.standard_normal((5, 2))
K = rbf(X, X, 0.5)
print(K.round(2))
print("symmetric:", np.allclose(K, K.T), "| smallest eigenvalue >= 0:", bool(np.linalg.eigvalsh(K).min() > -1e-10))`, "dot product in feature space: 0.004444\nkernel (x . z)^2            : 0.004444\n[[1.   0.16 0.26 0.95 0.17]\n [0.16 1.   0.01 0.1  0.  ]\n [0.26 0.01 1.   0.42 0.16]\n [0.95 0.1  0.42 1.   0.23]\n [0.17 0.   0.16 0.23 1.  ]]\nsymmetric: True | smallest eigenvalue >= 0: True"],
    ["h", "Kernel ridge regression"],
    ["p", "Ridge regression in the lifted space minimises $\\lVert\\mathbf y-\\boldsymbol\\Phi\\mathbf w\\rVert^2+\\lambda\\lVert\\mathbf w\\rVert^2$ (see [ridge](ml.html#regularization)). Its solution is $\\mathbf w=(\\boldsymbol\\Phi^\\top\\boldsymbol\\Phi+\\lambda\\mathbf I)^{-1}\\boldsymbol\\Phi^\\top\\mathbf y$, which needs a matrix of size (features × features): hopeless if the feature space is huge. The identity $(\\boldsymbol\\Phi^\\top\\boldsymbol\\Phi+\\lambda\\mathbf I)^{-1}\\boldsymbol\\Phi^\\top=\\boldsymbol\\Phi^\\top(\\boldsymbol\\Phi\\boldsymbol\\Phi^\\top+\\lambda\\mathbf I)^{-1}$ moves the inverse to the other side, where the matrix is (samples × samples):"],
    ["math", R`\mathbf w=\boldsymbol\Phi^\top\boldsymbol\alpha,\quad \boldsymbol\alpha=(\mathbf K+\lambda\mathbf I)^{-1}\mathbf y,\qquad f(\mathbf x)=\sum_{i=1}^n\alpha_i\,k(\mathbf x_i,\mathbf x)`, "K = Φ Φᵀ is the n×n Gram matrix; only kernel values appear"],
    ["p", "So the model is a weighted sum of bumps, one centred on each training point, and training is a single linear solve. That the weights are always a combination of training points is the **representer theorem**; it holds for any loss with an L2 penalty and is why SVMs, kernel ridge and Gaussian processes all look alike. The cost is $O(n^3)$ time and $O(n^2)$ memory, so it suits thousands of points, not millions."],
    ["code", R`
import numpy as np
from sklearn.kernel_ridge import KernelRidge

def rbf(A, B, gamma):
    return np.exp(-gamma * ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(-3, 3, 40)); y = np.sin(x) + 0.15 * rng.standard_normal(40)
X = x[:, None]; gamma, lam = 0.5, 0.1

K = rbf(X, X, gamma)
a = np.linalg.solve(K + lam * np.eye(len(X)), y)          # alpha = (K + lambda I)^-1 y
Xt = np.linspace(-3, 3, 7)[:, None]
pred = rbf(Xt, X, gamma) @ a                              # f(x) = sum_i alpha_i k(x_i, x)
print("scratch prediction:", pred.round(3))
print("true sin(x)       :", np.sin(Xt[:, 0]).round(3))

kr = KernelRidge(alpha=lam, kernel="rbf", gamma=gamma).fit(X, y)
print("same as sklearn KernelRidge:", bool(np.allclose(kr.predict(Xt), pred)))`, "scratch prediction: [-0.148 -0.682 -0.772 -0.048  0.855  0.927  0.312]\ntrue sin(x)       : [-0.141 -0.909 -0.841  0.     0.841  0.909  0.141]\nsame as sklearn KernelRidge: True"],
    ["h", "Kernel PCA"],
    ["p", "[PCA](ml.html#pca) finds directions of maximal variance, but only straight ones. Kernel PCA does PCA in the lifted space. The covariance of lifted data cannot be built, but PCA can also be done from the **Gram matrix** of dot products, and that we have. The recipe:"],
    ["steps", [
      "Build $\\mathbf K$ with your kernel.",
      "**Centre** it (the lifted data must have zero mean, and we cannot subtract a mean we cannot see): $\\tilde{\\mathbf K}=\\mathbf K-\\mathbf 1_n\\mathbf K-\\mathbf K\\mathbf 1_n+\\mathbf 1_n\\mathbf K\\mathbf 1_n$, where $\\mathbf 1_n$ is the $n\\times n$ matrix filled with $1/n$.",
      "Eigen-decompose $\\tilde{\\mathbf K}\\mathbf v_j=\\lambda_j\\mathbf v_j$ and keep the top eigenvectors ([eigenvectors](math.html#eigen)).",
      "The coordinates of the training points along component $j$ are $\\sqrt{\\lambda_j}\\,\\mathbf v_j$.",
    ]],
    ["code", R`
import numpy as np
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression

X, y = make_circles(300, factor=0.3, noise=0.05, random_state=0)   # an inner ring and an outer ring
gamma = 2.0
d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
K = np.exp(-gamma * d2); n = len(X); one = np.full((n, n), 1 / n)
Kc = K - one @ K - K @ one + one @ K @ one                  # centre the Gram matrix

vals, vecs = np.linalg.eigh(Kc)                             # ascending order
idx = vals.argsort()[::-1][:2]
vals, vecs = vals[idx], vecs[:, idx]
Z = vecs * np.sqrt(vals)                                    # coordinates of the training points
print("top eigenvalues:", vals.round(3))

sk = KernelPCA(n_components=2, kernel="rbf", gamma=gamma).fit_transform(X)
print("correlation with sklearn (sign is arbitrary):", [round(float(abs(np.corrcoef(Z[:, j], sk[:, j])[0, 1])), 4) for j in range(2)])

for name, F in [("raw features (x1, x2)", X), ("first kernel PC only", Z[:, :1])]:
    print(f"{name:22s} linear classifier accuracy:", round(LogisticRegression().fit(F, y).score(F, y), 3))`, "top eigenvalues: [45.062 36.9  ]\ncorrelation with sklearn (sign is arbitrary): [1.0, 1.0]\nraw features (x1, x2)  linear classifier accuracy: 0.51\nfirst kernel PC only   linear classifier accuracy: 1.0"],
    ["p", "Two concentric rings cannot be separated by any line in the original plane (accuracy near 0.5), yet **one** kernel principal component is enough for a linear classifier."],
    ["table", ["Method", "Plain version", "Kernel version", "Cost"], [
      ["Classification", "linear SVM", "[kernel SVM](ml.html#svm)", "O(n²)–O(n³) training"],
      ["Regression", "ridge regression", "kernel ridge (= Gaussian-process mean)", "O(n³)"],
      ["Dimensionality reduction", "[PCA](ml.html#pca)", "kernel PCA", "O(n³) eigen-problem"],
      ["Clustering", "k-means", "kernel k-means, [spectral clustering](classical.html#spectral-clustering)", "O(n²) memory"],
    ]],
    ["danger", "Forgetting to **centre** the kernel matrix in kernel PCA. Without it the first component just captures the mean of the lifted data. Also remember that a new point must be centred using the *training* statistics; use `KernelPCA.transform` rather than doing it by hand."],
    ["warn", "RBF kernels are very sensitive to $\\gamma$ and to feature scale. A large $\\gamma$ makes every point similar only to itself (the Gram matrix tends to the identity, so the model memorises); a tiny $\\gamma$ makes everything similar (nearly linear). Standardise first and tune $\\gamma$ and $\\lambda$ by [cross-validation](ml.html#cross-validation)."],
    ["tip", "Kernels need not be on vectors: there are kernels for strings, graphs and sets, as long as you can define a valid similarity. That is how SVMs once classified text and proteins without any feature engineering."],
  ],
},
{
  id: "gaussian-processes", group: "Discriminants & Kernels", title: "Gaussian Processes",
  sub: "A distribution over functions: fit a smooth curve through data and get an error bar at every input, with almost no assumptions about its shape.",
  tags: ["regression", "bayesian", "kernels"],
  blocks: [
    ["p", "A Gaussian for one number is a bell curve. A multivariate Gaussian for a few numbers is a cloud. Now imagine a Gaussian for the *value of a function at every point on a line*: that is a **Gaussian process (GP)**. Concretely, it says: take any finite set of inputs $x_1,\\dots,x_n$; the function values $f(x_1),\\dots,f(x_n)$ are jointly Gaussian, with a covariance given by a **kernel**: inputs that are close have strongly correlated outputs, so the function is smooth. Observing some points then *conditions* this Gaussian, and the result is a prediction (mean) plus an uncertainty (variance) at every other input."],
    ["viz", "cml1_gp"],
    ["p", "Try the length-scale slider. A short $\\ell$ means \"the function can change quickly\": the curve chases the points and the band balloons back to the prior between them. A long $\\ell$ means \"very smooth\": the curve cannot follow the sine wave. In between, the fit is good and the band is thin near data and wide far from it. The purple lines are random functions drawn from the posterior; the number \"log marginal likelihood\" tells you which setting the data prefers."],
    ["h", "The maths"],
    ["p", "A GP is defined by a **mean function** (usually 0 after centring) and a **kernel** $k(x,x')$ ([kernels](classical.html#kernel-methods)). For training inputs $X$ and test inputs $X_*$, the prior says"],
    ["math", R`\begin{bmatrix}\mathbf y\\ \mathbf f_*\end{bmatrix}\sim\mathcal N\!\left(\mathbf 0,\begin{bmatrix}K+\sigma_n^2I & K_*\\ K_*^\top & K_{**}\end{bmatrix}\right)`, "K = k(X,X) · K_* = k(X,X_*) · K_** = k(X_*,X_*) · σ_n²: observation noise variance"],
    ["p", "Now use the standard rule for **conditioning a Gaussian** (the same algebra as in [Bayes' theorem](math.html#bayes) for Gaussians): if $\\begin{bmatrix}a\\\\b\\end{bmatrix}\\sim\\mathcal N\\big(0,\\begin{bmatrix}A&C\\\\C^\\top&B\\end{bmatrix}\\big)$, then $b\\mid a\\sim\\mathcal N\\big(C^\\top A^{-1}a,\\ B-C^\\top A^{-1}C\\big)$. Plugging in:"],
    ["math", R`\boldsymbol\mu_*=K_*^\top(K+\sigma_n^2I)^{-1}\mathbf y,\qquad \boldsymbol\Sigma_*=K_{**}-K_*^\top(K+\sigma_n^2I)^{-1}K_*`, "the posterior over the function values at the test inputs"],
    ["list", [
      "The mean $\\boldsymbol\\mu_*=\\sum_i\\alpha_i\\,k(x_i,x_*)$ with $\\boldsymbol\\alpha=(K+\\sigma_n^2I)^{-1}\\mathbf y$: this is exactly **kernel ridge regression** with $\\lambda=\\sigma_n^2$.",
      "The variance **only subtracts**: it starts at the prior variance $k(x_*,x_*)$ and shrinks by how much the data tells us. Far from the data it returns to the prior.",
      "The covariance $\\boldsymbol\\Sigma_*$ lets you sample whole smooth functions from the posterior (the purple lines).",
    ]],
    ["note", "Tiny example. One training point $(x,y)=(0,1)$, RBF kernel with $\\ell=1$ so $k(r)=e^{-r^2/2}$, noise $\\sigma_n^2=0.1$. Then $K+\\sigma_n^2=1+0.1=1.1$. At $x_*=1$: $K_*=e^{-0.5}\\approx0.607$, so the mean is $0.607\\times1/1.1\\approx0.551$ and the variance is $1-0.607^2/1.1\\approx0.666$. The prediction is pulled towards the observation (0.55, not 0), but with plenty of doubt left (variance 0.67 out of a possible 1). At $x_*=0$ the variance would be $1-1/1.1\\approx0.09$."],
    ["h", "Learning the hyperparameters"],
    ["p", "The kernel has hyperparameters (length-scale $\\ell$, signal variance, noise $\\sigma_n^2$). The GP gives a closed-form way to pick them: maximise the **log marginal likelihood** of the data,"],
    ["math", R`\log p(\mathbf y\mid X)=\underbrace{-\tfrac12\mathbf y^\top(K+\sigma_n^2I)^{-1}\mathbf y}_{\text{data fit}}\;\underbrace{-\tfrac12\log\lvert K+\sigma_n^2I\rvert}_{\text{complexity penalty}}\;-\tfrac n2\log2\pi`],
    ["p", "The first term rewards explaining the data; the second penalises flexible models (large determinant). Together they implement an automatic Occam's razor without a validation set. The marginal likelihood is also what the visualizer prints."],
    ["h", "From scratch in NumPy"],
    ["p", "Never invert $K+\\sigma_n^2I$ directly. Take its **Cholesky factor** $K+\\sigma_n^2I=LL^\\top$ (a triangular square root), solve two triangular systems, and read the log-determinant off the diagonal: $\\log|LL^\\top|=2\\sum\\log L_{ii}$."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
Xtr = np.array([-4.0, -2.8, -1.5, 0.3, 1.6, 3.2])
ytr = np.sin(Xtr) + 0.05 * rng.standard_normal(6)

def k(a, b, ell):                                          # RBF kernel between two 1-D arrays
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)

def gp(Xtr, ytr, Xs, ell, sn):
    L = np.linalg.cholesky(k(Xtr, Xtr, ell) + sn**2 * np.eye(len(Xtr)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, ytr))   # (K + s^2 I)^-1 y with two triangular-style solves
    Ks = k(Xtr, Xs, ell)
    mu = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var = 1.0 - (v**2).sum(axis=0)                          # k(x*, x*) = 1 for this kernel
    lml = -0.5 * ytr @ alpha - np.log(np.diag(L)).sum() - 0.5 * len(ytr) * np.log(2 * np.pi)
    return mu, np.sqrt(np.maximum(var, 0)), lml

Xs = np.array([-3.5, -2.0, 0.0, 2.0, 5.0])
mu, sd, lml = gp(Xtr, ytr, Xs, ell=1.2, sn=0.1)
for x, m, s in zip(Xs, mu, sd):
    print(f"x = {x:4.1f}  mean {m:6.3f}  sd {s:.3f}  (true sin {np.sin(x):6.3f})")

for ell in (0.2, 0.6, 1.2, 2.5):
    print(f"length-scale {ell}: log marginal likelihood {gp(Xtr, ytr, Xs, ell, 0.1)[2]:.2f}")`, "x = -3.5  mean  0.385  sd 0.157  (true sin  0.351)\nx = -2.0  mean -0.908  sd 0.159  (true sin -0.909)\nx =  0.0  mean  0.019  sd 0.162  (true sin  0.000)\nx =  2.0  mean  0.822  sd 0.190  (true sin  0.909)\nx =  5.0  mean -0.157  sd 0.935  (true sin -0.959)\nlength-scale 0.2: log marginal likelihood -6.87\nlength-scale 0.6: log marginal likelihood -6.83\nlength-scale 1.2: log marginal likelihood -6.24\nlength-scale 2.5: log marginal likelihood -9.07"],
    ["p", "The standard deviation is smallest next to training points and is close to the prior value of 1 at $x=5$, far from all data: the GP knows what it does not know. And the length-scale with the highest marginal likelihood is the one that suits the data."],
    ["h", "The scikit-learn version"],
    ["code", R`
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
Xtr = np.array([-4.0, -2.8, -1.5, 0.3, 1.6, 3.2])
ytr = np.sin(Xtr) + 0.05 * rng.standard_normal(6)

kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)   # signal variance * RBF + noise
gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=3, random_state=0).fit(Xtr[:, None], ytr)
print("learned kernel:", gpr.kernel_)
print("log marginal likelihood:", round(float(gpr.log_marginal_likelihood_value_), 3))

Xs = np.array([[-3.5], [0.0], [5.0]])
mean, std = gpr.predict(Xs, return_std=True)
print("mean:", mean.round(3), " std:", std.round(3))`, "learned kernel: 1.04**2 * RBF(length_scale=1.85) + WhiteKernel(noise_level=1e-05)\nlog marginal likelihood: -5.637\nmean: [ 0.331  0.009 -0.766]  std: [0.031 0.02  0.657]"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Uncertainty is built in and well calibrated if the kernel is right", "O(n³) time and O(n²) memory: practical to a few thousand points"],
      ["Works with tiny datasets and few assumptions", "Result depends strongly on the kernel choice"],
      ["Hyperparameters learned from the marginal likelihood", "Classification needs approximations (no closed form)"],
      ["Perfect for Bayesian optimisation and surrogate models", "Poor in very high dimensions without structure"],
    ]],
    ["danger", "Confusing the two variances. The posterior variance of $f(x_*)$ is the uncertainty about the *function*; the variance of a new noisy *observation* is that plus $\sigma_n^2$. The shaded bands in the picture are for $f$. When the kernel contains a `WhiteKernel` (as above), scikit-learn's `predict(return_std=True)` reports the std of a new noisy observation, so it includes the learned noise; subtract it in quadrature if you want the std of $f$ alone."],
    ["warn", "A GP with a zero-mean prior reverts to zero away from the data. Subtract the mean of $\\mathbf y$ (`normalize_y=True`) or add a mean/linear kernel if your target has a trend."],
    ["tip", "A GP with a linear kernel is [Bayesian linear regression](classical.html#bayes-linreg); with an RBF kernel it is Bayesian regression with infinitely many bump features. Same maths, seen in weight space or in function space."],
  ],
},
{
  id: "gmm", group: "Probabilistic Models", title: "Gaussian Mixture Models",
  sub: "Cluster with soft, elliptical blobs: each point belongs to every cluster with some probability, and each cluster has its own shape.",
  tags: ["clustering", "probabilistic"],
  blocks: [
    ["p", "[k-means](ml.html#kmeans) draws round clusters and says each point belongs to exactly one. A **Gaussian mixture model (GMM)** is its probabilistic big sibling. It says the data was produced by several Gaussian blobs: to create a point, first pick a blob at random (with some probabilities), then draw the point from that blob. Blobs can be stretched and tilted, and a point near the border can honestly be \"60% blob A, 40% blob B\". The same model also works as a **density estimator**: it tells you how likely any new point is, which makes it useful for anomaly detection."],
    ["h", "The model"],
    ["math", R`p(\mathbf x)=\sum_{k=1}^{K}\pi_k\,\mathcal N(\mathbf x\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k),\qquad \pi_k\ge0,\ \sum_k\pi_k=1`, "π_k: mixing weight (how big blob k is) · μ_k: its centre · Σ_k: its shape (covariance matrix)"],
    ["p", "Behind this sits a **latent (hidden) variable** $z\\in\\{1,\\dots,K\\}$: the blob that generated the point. $p(z=k)=\\pi_k$ and $p(\\mathbf x\\mid z=k)=\\mathcal N(\\boldsymbol\\mu_k,\\boldsymbol\\Sigma_k)$. We observe $\\mathbf x$ but not $z$. Fitting the model means finding $\\pi,\\boldsymbol\\mu,\\boldsymbol\\Sigma$ that maximise the log-likelihood"],
    ["math", R`\log p(X)=\sum_{i=1}^n\log\sum_{k=1}^K\pi_k\,\mathcal N(\mathbf x_i\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)`, "the sum is inside the log, so there is no closed-form maximiser"],
    ["p", "If we knew which blob each point came from, fitting would be easy (the mean and covariance of each group). If we knew the blobs, assigning points would be easy (Bayes' rule). We know neither, so we alternate, which is the [EM algorithm](classical.html#em)."],
    ["h", "Responsibilities and the updates"],
    ["p", "The probability that blob $k$ produced point $\\mathbf x_i$, given the current parameters, is its **responsibility**, straight from [Bayes' theorem](math.html#bayes):"],
    ["math", R`r_{ik}=P(z_i=k\mid\mathbf x_i)=\frac{\pi_k\,\mathcal N(\mathbf x_i\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)}{\sum_{j=1}^K\pi_j\,\mathcal N(\mathbf x_i\mid\boldsymbol\mu_j,\boldsymbol\Sigma_j)}`, "E-step: each row of r sums to 1"],
    ["math", R`N_k=\sum_i r_{ik},\quad \boldsymbol\mu_k=\frac1{N_k}\sum_i r_{ik}\mathbf x_i,\quad \boldsymbol\Sigma_k=\frac1{N_k}\sum_i r_{ik}(\mathbf x_i-\boldsymbol\mu_k)(\mathbf x_i-\boldsymbol\mu_k)^\top,\quad \pi_k=\frac{N_k}{n}`, "M-step: weighted versions of the ordinary mean, covariance and proportion"],
    ["note", "Tiny example in 1-D. Two blobs with equal weight 0.5: $\\mathcal N(0,1)$ and $\\mathcal N(4,1)$. For the point $x=1.5$: the densities are $\\mathcal N(1.5\\mid0,1)=0.1295$ and $\\mathcal N(1.5\\mid4,1)=0.0175$. So $r_1=\\dfrac{0.5\\cdot0.1295}{0.5\\cdot0.1295+0.5\\cdot0.0175}\\approx0.88$ and $r_2\\approx0.12$: the point is mostly, but not entirely, blob 1."],
    ["viz", "cml1_gmm"],
    ["p", "Step through it. The steps alternate: in each E-step the points get coloured by their responsibilities, in each M-step the ellipses move to match the weighted points. The right-hand chart is the log-likelihood: it only goes up. Change the **random start**: most starts reach the same fit, but some end in a worse local optimum (try start 4), which is why you run EM several times."],
    ["p", "**Relation to k-means.** Give every blob the same round covariance $\\sigma^2\\mathbf I$ and let $\\sigma\\to0$. The responsibilities become hard 0/1 assignments, the mean update becomes the ordinary centroid update, and EM turns into k-means. So k-means is a GMM with spherical, equal, infinitely sharp blobs."],
    ["h", "From scratch in NumPy"],
    ["p", "The densities can be tiny, so we work with log-densities and the **log-sum-exp** trick (subtract the maximum before exponentiating). The multivariate Gaussian log-density uses the Cholesky factor $\\boldsymbol\\Sigma=LL^\\top$: $\\log\\mathcal N=-\\frac12\\lVert L^{-1}(\\mathbf x-\\boldsymbol\\mu)\\rVert^2-\\sum\\log L_{ii}-\\frac d2\\log2\\pi$."],
    ["code", R`
import numpy as np
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=[[0, 0], [5, 5], [0, 6]], cluster_std=[1.0, 1.8, 0.6], random_state=0)
n, d = X.shape; K = 3

def log_gauss(X, mu, S):                                   # log N(x | mu, S) for every row of X
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, (X - mu).T)
    return -0.5 * (z**2).sum(axis=0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)

rng = np.random.default_rng(0)
mu = X[rng.choice(n, K, replace=False)]                    # start: K random data points
S = np.stack([np.cov(X.T)] * K)                            # start: the overall covariance for every blob
pi = np.full(K, 1 / K)

prev = -np.inf
for it in range(200):
    logp = np.stack([np.log(pi[k]) + log_gauss(X, mu[k], S[k]) for k in range(K)], axis=1)   # (n, K)
    mx = logp.max(axis=1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(logp - mx).sum(axis=1))                                    # log sum_k pi_k N_k
    R = np.exp(logp - lse[:, None])                        # E-step: responsibilities
    ll = lse.sum()
    if ll - prev < 1e-8:
        break
    prev = ll
    Nk = R.sum(axis=0)                                     # M-step
    mu = (R.T @ X) / Nk[:, None]
    for k in range(K):
        diff = X - mu[k]
        S[k] = (R[:, k, None] * diff).T @ diff / Nk[k] + 1e-6 * np.eye(d)
    pi = Nk / n

o = np.argsort(mu[:, 1])                                   # list the blobs by their y position
print("iterations:", it, " log-likelihood:", round(float(ll), 2))
print("weights:", pi[o].round(3))
print("means:\n", mu[o].round(2))
print("rms std per blob:", np.sqrt(np.trace(S, axis1=1, axis2=2)[o] / d).round(2), "(true 1.0, 1.8, 0.6)")`, "iterations: 57  log-likelihood: -1182.93\nweights: [0.33  0.343 0.327]\nmeans:\n [[-0.02  0.13]\n [ 4.64  4.75]\n [-0.08  5.96]]\nrms std per blob: [1.01 1.76 0.6 ] (true 1.0, 1.8, 0.6)"],
    ["h", "The scikit-learn version and choosing K"],
    ["p", "The number of blobs $K$ is a choice. The likelihood always improves with more components, so use a criterion with a complexity penalty, the **BIC** (Bayesian Information Criterion) $=p\\ln n-2\\ln\\hat L$ where $p$ is the number of free parameters. Lower is better."],
    ["code", R`
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=300, centers=[[0, 0], [5, 5], [0, 6]], cluster_std=[1.0, 1.8, 0.6], random_state=0)

for K in range(1, 7):
    gm = GaussianMixture(n_components=K, covariance_type="full", n_init=3, random_state=0).fit(X)
    print(f"K = {K}  BIC {gm.bic(X):8.1f}")

gm = GaussianMixture(n_components=3, n_init=5, random_state=0).fit(X)
o = np.argsort(gm.means_[:, 0])
print("weights:", gm.weights_[o].round(3), " converged:", gm.converged_, "in", gm.n_iter_, "iterations")
print("point (2.5, 2.5): hard label", int(gm.predict([[2.5, 2.5]])[0]), " soft:", gm.predict_proba([[2.5, 2.5]]).round(3)[0][o])
print("log-density of an outlier (20, 20):", round(float(gm.score_samples([[20.0, 20.0]])[0]), 1))`, "K = 1  BIC   2888.2\nK = 2  BIC   2649.5\nK = 3  BIC   2463.0\nK = 4  BIC   2495.4\nK = 5  BIC   2526.4\nK = 6  BIC   2555.6\nweights: [0.332 0.331 0.337]  converged: True in 6 iterations\npoint (2.5, 2.5): hard label 2  soft: [0.    0.039 0.961]\nlog-density of an outlier (20, 20): -85.6"],
    ["table", ["covariance_type", "Blob shape", "Parameters per blob (d features)"], [
      ["`\"spherical\"`", "round, own radius", "1"],
      ["`\"diag\"`", "axis-aligned ellipse", "d"],
      ["`\"tied\"`", "all blobs share one full shape", "d(d+1)/2 shared once"],
      ["`\"full\"`", "any tilted ellipse (default)", "d(d+1)/2"],
    ]],
    ["table", ["Strengths", "Weaknesses"], [
      ["Soft assignments and a proper probability model", "Local optima: restart with `n_init`"],
      ["Elliptical clusters of different sizes", "Must choose K (use BIC or AIC)"],
      ["Density estimation, sampling (`gm.sample`), anomaly scores", "Singular covariances if a blob collapses onto a few points"],
      ["Includes k-means as a special case", "Assumes blobs are Gaussian; poor for rings or crescents"],
    ]],
    ["danger", "**Collapse onto a point.** If one blob shrinks onto a single data point its variance goes to 0 and the likelihood goes to infinity, a degenerate \"best\" solution. This is why the code adds a tiny `1e-6 * np.eye(d)` and scikit-learn has `reg_covar`. Watch for components with very few points."],
    ["warn", "Component labels are arbitrary (blob 0 in one run is blob 2 in another: **label switching**), so never compare raw labels between runs; compare the fitted distributions or sort the blobs first."],
    ["tip", "For anomaly detection, flag points whose `score_samples` (log-density) is below, say, the 1st percentile of the training scores. See also [Anomaly Detection](classical.html#anomaly-detection)."],
  ],
},
{
  id: "em", group: "Probabilistic Models", title: "The EM Algorithm",
  sub: "A general recipe for fitting models with hidden variables: guess the hidden parts (soft), refit the parameters, repeat. The likelihood can never go down.",
  tags: ["latent variables", "optimisation"],
  blocks: [
    ["p", "Many models have a part you never see. Which blob made this point? Which coin was flipped? Which topic is this word about? These hidden quantities are called **latent variables**. They make maximum likelihood ([MLE](math.html#mle)) hard, because the likelihood has a *sum over every possible hidden value inside the logarithm*, and that does not simplify. **Expectation-Maximisation (EM)** gets around this with a two-step loop that is the engine behind [Gaussian mixtures](classical.html#gmm), [HMM training](classical.html#hmm), many missing-data methods and (in variational form) topic models."],
    ["h", "The idea"],
    ["p", "Two facts make EM work. (1) If you **knew the hidden values**, fitting the parameters would be easy: just count or average, as in ordinary MLE. (2) If you **knew the parameters**, you could work out how likely each hidden value is (Bayes' rule). We know neither, so alternate:"],
    ["diagram", "cml1_emloop", [], "EM alternates between guessing the hidden variables (softly) and refitting the parameters using that guess."],
    ["list", [
      "**E-step (expectation)**: with the current parameters $\\theta^{\\text{old}}$, compute the posterior $q(\\mathbf z)=p(\\mathbf z\\mid\\mathbf x,\\theta^{\\text{old}})$ of the hidden variables for every data point. These are the *soft* guesses (responsibilities).",
      "**M-step (maximisation)**: choose new parameters that maximise the expected complete-data log-likelihood, where the expectation uses those soft guesses. This is the easy weighted-MLE problem.",
    ]],
    ["h", "Why it works: a lower bound"],
    ["p", "We want to maximise $\\ell(\\theta)=\\log p(\\mathbf x\\mid\\theta)=\\log\\sum_{\\mathbf z}p(\\mathbf x,\\mathbf z\\mid\\theta)$. For **any** distribution $q(\\mathbf z)$ we can multiply and divide by $q$ inside the sum and apply **Jensen's inequality** (the log of an average is at least the average of the logs, because log is concave):"],
    ["math", R`\ell(\theta)=\log\sum_{\mathbf z}q(\mathbf z)\frac{p(\mathbf x,\mathbf z\mid\theta)}{q(\mathbf z)}\;\ge\;\sum_{\mathbf z}q(\mathbf z)\log\frac{p(\mathbf x,\mathbf z\mid\theta)}{q(\mathbf z)}=:\mathcal L(q,\theta)`, "L is the evidence lower bound (ELBO): it is always below the true log-likelihood"],
    ["p", "The gap is exactly a KL divergence ([KL](math.html#entropy)): $\\ell(\\theta)-\\mathcal L(q,\\theta)=\\mathrm{KL}\\big(q(\\mathbf z)\\,\\|\\,p(\\mathbf z\\mid\\mathbf x,\\theta)\\big)\\ge0$. Now the two steps are just coordinate ascent on $\\mathcal L$:"],
    ["list", [
      "**E-step**: fix $\\theta^{\\text{old}}$ and pick $q$ to make the bound as high as possible. The gap is zero when $q(\\mathbf z)=p(\\mathbf z\\mid\\mathbf x,\\theta^{\\text{old}})$, so the bound **touches** the true log-likelihood at $\\theta^{\\text{old}}$.",
      "**M-step**: fix $q$ and maximise over $\\theta$. Only the numerator matters, giving $Q(\\theta\\mid\\theta^{\\text{old}})=\\sum_{\\mathbf z}q(\\mathbf z)\\log p(\\mathbf x,\\mathbf z\\mid\\theta)$.",
    ]],
    ["math", R`\ell(\theta^{\text{new}})\;\ge\;\mathcal L(q,\theta^{\text{new}})\;\ge\;\mathcal L(q,\theta^{\text{old}})\;=\;\ell(\theta^{\text{old}})`, "so every EM iteration cannot lower the log-likelihood"],
    ["p", "Read it left to right: after the M-step the true likelihood is at least the new bound, which is at least the old bound, which equals the old likelihood. So the log-likelihood is **monotone** and (being bounded) converges, though only to a *local* maximum or saddle point."],
    ["h", "Worked derivation: the Gaussian mixture M-step"],
    ["p", "For a mixture, the complete-data log-likelihood is $\\sum_i\\sum_k z_{ik}\\,[\\log\\pi_k+\\log\\mathcal N(\\mathbf x_i\\mid\\boldsymbol\\mu_k,\\boldsymbol\\Sigma_k)]$ with $z_{ik}=1$ if point $i$ came from blob $k$. The E-step replaces each $z_{ik}$ by its expectation, the responsibility $r_{ik}$. Maximising the resulting $Q$:"],
    ["list", [
      "**Means**: $\\partial Q/\\partial\\boldsymbol\\mu_k=\\sum_i r_{ik}\\boldsymbol\\Sigma_k^{-1}(\\mathbf x_i-\\boldsymbol\\mu_k)=0$ gives $\\boldsymbol\\mu_k=\\sum_i r_{ik}\\mathbf x_i\\,/\\,\\sum_i r_{ik}$: a **weighted average**.",
      "**Covariances**: differentiating with respect to $\\boldsymbol\\Sigma_k^{-1}$ gives the weighted covariance $\\boldsymbol\\Sigma_k=\\sum_i r_{ik}(\\mathbf x_i-\\boldsymbol\\mu_k)(\\mathbf x_i-\\boldsymbol\\mu_k)^\\top\\,/\\,N_k$.",
      "**Weights**: maximise $\\sum_k N_k\\log\\pi_k$ subject to $\\sum_k\\pi_k=1$ with a Lagrange multiplier; the answer is $\\pi_k=N_k/n$.",
    ]],
    ["h", "A tiny example you can do by hand: two coins"],
    ["p", "A classic. You have two coins, A and B, with unknown probabilities of heads $\\theta_A,\\theta_B$. In each of 5 rounds someone secretly picks one coin and flips it 10 times; you only see the head counts: 5, 9, 8, 4, 7 (you do **not** see which coin). If you knew the coins it would be simple counting. Here the hidden variable is the coin. E-step: for a round with $h$ heads, the posterior that coin A was used is proportional to $\\theta_A^{h}(1-\\theta_A)^{10-h}$ (equal prior; the binomial coefficient cancels). M-step: $\\theta_A$ = (expected heads from A) / (expected flips of A)."],
    ["code", R`
import numpy as np
from math import comb

heads, n = np.array([5, 9, 8, 4, 7]), 10
thA, thB = 0.6, 0.5                                        # initial guesses

coef = np.array([comb(n, h) for h in heads])               # binomial coefficients

def loglik(thA, thB):                                      # true log-likelihood of the data (equal coin priors)
    pa = coef * thA**heads * (1 - thA)**(n - heads)
    pb = coef * thB**heads * (1 - thB)**(n - heads)
    return float(np.log(0.5 * pa + 0.5 * pb).sum())

for it in range(12):
    la = thA**heads * (1 - thA)**(n - heads)               # E-step: how well each coin explains each round
    lb = thB**heads * (1 - thB)**(n - heads)
    rA = la / (la + lb); rB = 1 - rA                       # responsibilities
    thA = (rA * heads).sum() / (rA * n).sum()              # M-step: weighted head fractions
    thB = (rB * heads).sum() / (rB * n).sum()
    if it in (0, 1, 2, 11):
        print(f"iter {it + 1:2d}: thA {thA:.3f}  thB {thB:.3f}  log-likelihood {loglik(thA, thB):.4f}")
print("responsibility of coin A per round:", rA.round(2))`, "iter  1: thA 0.713  thB 0.581  log-likelihood -10.0860\niter  2: thA 0.745  thB 0.569  log-likelihood -9.9498\niter  3: thA 0.768  thB 0.550  log-likelihood -9.8546\niter 12: thA 0.797  thB 0.520  log-likelihood -9.7969\nresponsibility of coin A per round: [0.1  0.95 0.85 0.03 0.6 ]"],
    ["p", "The rounds with 9 and 8 heads are credited mostly to coin A, those with 5 and 4 mostly to coin B (7 leans towards A), so $\\theta_A$ rises and $\\theta_B$ falls, and the log-likelihood increases at every step."],
    ["h", "Seeing the monotone climb in scikit-learn"],
    ["p", "`GaussianMixture(max_iter=k, tol=0)` runs exactly $k$ EM iterations from the same random start, so we can watch the log-likelihood after each one."],
    ["code", R`
import warnings
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
X, _ = make_blobs(300, centers=3, cluster_std=[1.0, 2.0, 0.6], random_state=2)

lls = []
for it in range(1, 11):
    gm = GaussianMixture(3, init_params="random", max_iter=it, tol=0, random_state=0).fit(X)
    lls.append(gm.score(X))                                # average log-likelihood per point
print(np.round(lls, 4).tolist())
print("never decreases:", bool(np.all(np.diff(lls) >= -1e-9)))`, "[-4.6596, -4.6592, -4.6579, -4.6535, -4.6408, -4.6097, -4.558, -4.5075, -4.4776, -4.4648]\nnever decreases: True"],
    ["table", ["Aspect", "What to know"], [
      ["Convergence", "Monotone in the likelihood, to a local optimum. Often slow near the end (linear convergence)."],
      ["Initialisation", "Matters a lot. Run several random starts, or start from k-means."],
      ["Hard EM", "Replace the soft E-step by the single most likely value. For a spherical GMM this is exactly [k-means](ml.html#kmeans)."],
      ["Used in", "GMM, HMM (Baum-Welch), factor analysis, probabilistic PCA, mixtures of experts, imputing missing data, semi-supervised learning."],
      ["Variants", "MAP-EM (add a prior), generalised EM (just improve, not maximise), stochastic / online EM, variational EM (approximate the E-step)."],
    ]],
    ["danger", "Reading the EM answer as \"the\" answer. It is a local optimum that depends on the start. Restart several times and keep the run with the highest log-likelihood, and be suspicious of a run whose likelihood shoots up because a component collapsed onto one data point."],
    ["warn", "EM needs an E-step you can compute. If the posterior over hidden variables is intractable (topic models, deep latent models) you have to approximate it: variational inference or [MCMC](classical.html#mcmc). Practice: [One EM Iteration for a 1-D Mixture](practice.html#p/gmm-em-one-iteration)."],
  ],
},
{
  id: "hmm", group: "Probabilistic Models", title: "Hidden Markov Models: Forward, Viterbi & Baum-Welch",
  sub: "A hidden chain of states that you cannot see, each producing something you can see. Three algorithms answer: how likely, which states, and how to learn it.",
  tags: ["sequences", "probabilistic"],
  blocks: [
    ["p", "Alice lives far away and you only learn what she did each day: walked, shopped or cleaned. Her choice depends on the **weather**, which you cannot see. The weather itself follows a pattern: rain tends to be followed by rain. This is a **Hidden Markov Model (HMM)**: a hidden chain of states (weather) with a visible observation at every step (activity). The same structure appears in speech recognition (hidden phonemes, visible audio), part-of-speech tagging (hidden tags, visible words), gene finding and anything with a sequence behind noisy readings."],
    ["diagram", "cml1_weather", [], "Alice's toy HMM. Arrows are the transition probabilities between hidden weather states; the lists are the chance of each activity in each state."],
    ["h", "The model"],
    ["p", "Three ingredients, with $K$ hidden states and $M$ possible observations:"],
    ["list", [
      "**Start** distribution $\\pi_i=P(z_1=i)$.",
      "**Transition** matrix $A_{ij}=P(z_t=j\\mid z_{t-1}=i)$: each row sums to 1. This is the **Markov property**: the next state depends only on the current one.",
      "**Emission** matrix $B_{j}(o)=P(x_t=o\\mid z_t=j)$: each row sums to 1. Each observation depends only on the state that produced it.",
    ]],
    ["diagram", "cml1_hmmgraph", [], "The HMM as a graphical model: a chain of hidden states z, each emitting one observation x."],
    ["math", R`P(\mathbf z,\mathbf x)=\pi_{z_1}B_{z_1}(x_1)\prod_{t=2}^{T}A_{z_{t-1}z_t}\,B_{z_t}(x_t)`, "the joint probability of one particular hidden path and the observations"],
    ["p", "Three questions, three algorithms:"],
    ["table", ["Question", "Name", "Algorithm", "Cost"], [
      ["How likely is this observation sequence?", "Evaluation", "Forward", "O(T·K²)"],
      ["Which hidden path most likely produced it?", "Decoding", "Viterbi", "O(T·K²)"],
      ["Given only observations, what are π, A, B?", "Learning", "Baum-Welch (EM)", "O(T·K²) per iteration"],
    ]],
    ["p", "The naive way to answer the first two is to enumerate all $K^T$ hidden paths: with 2 states and 100 days that is $2^{100}$. **Dynamic programming** reuses partial results: the best way of reaching a state today only depends on yesterday's best ways of reaching each state."],
    ["h", "Forward algorithm: the likelihood"],
    ["p", "Let $\\alpha_t(j)=P(x_1,\\dots,x_t,\\,z_t=j)$: the probability of the observations so far and being in state $j$ now. Then"],
    ["math", R`\alpha_1(j)=\pi_jB_j(x_1),\qquad \alpha_t(j)=\Big[\sum_{i}\alpha_{t-1}(i)A_{ij}\Big]B_j(x_t),\qquad P(\mathbf x)=\sum_j\alpha_T(j)`, "sum over where you came from, then emit"],
    ["h", "Viterbi: the best hidden path"],
    ["p", "Replace the **sum** by a **max** and remember which predecessor won. $\\delta_t(j)$ is the probability of the *best single path* ending in state $j$ at time $t$, and $\\psi_t(j)$ is the state it came from:"],
    ["math", R`\delta_1(j)=\pi_jB_j(x_1),\quad \delta_t(j)=\max_i\big[\delta_{t-1}(i)A_{ij}\big]B_j(x_t),\quad \psi_t(j)=\arg\max_i\,\delta_{t-1}(i)A_{ij}`, "then start at argmax_j δ_T(j) and follow ψ backwards"],
    ["viz", "cml1_viterbi"],
    ["p", "Step through with Alice's days. Each box holds the best path probability for that state; the amber arrow marks which predecessor won, and the green path is read off by walking back from the best final box. Try your own sequence, for example `clean clean walk`."],
    ["note", "By hand for Alice (walk, shop, clean). Day 1: Rainy $0.6\\times0.1=0.06$, Sunny $0.4\\times0.6=0.24$. Day 2 (shop), Rainy: $\\max(0.06\\times0.7,\\ 0.24\\times0.4)=0.096$ (from Sunny), times $0.4$ gives $0.0384$. Sunny: $\\max(0.06\\times0.3,\\ 0.24\\times0.6)=0.144\\times0.3=0.0432$. Day 3 (clean): Rainy $\\max(0.0384\\times0.7,\\ 0.0432\\times0.4)\\times0.5=0.01344$ (from Rainy). Best path: Sunny, Rainy, Rainy."],
    ["h", "Baum-Welch: learning with EM"],
    ["p", "If you had the hidden states, learning would be counting. You do not, so use [EM](classical.html#em). The E-step needs two posteriors, computed with the forward values $\\alpha$ and the matching **backward** values $\\beta_t(i)=P(x_{t+1},\\dots,x_T\\mid z_t=i)$ (computed right-to-left, $\\beta_T=1$, $\\beta_t(i)=\\sum_jA_{ij}B_j(x_{t+1})\\beta_{t+1}(j)$):"],
    ["math", R`\gamma_t(i)=P(z_t=i\mid\mathbf x)=\frac{\alpha_t(i)\beta_t(i)}{P(\mathbf x)},\qquad \xi_t(i,j)=P(z_t=i,z_{t+1}=j\mid\mathbf x)=\frac{\alpha_t(i)A_{ij}B_j(x_{t+1})\beta_{t+1}(j)}{P(\mathbf x)}`, "γ: soft state at time t · ξ: soft transition between t and t+1"],
    ["math", R`\hat\pi_i=\gamma_1(i),\qquad \hat A_{ij}=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)},\qquad \hat B_j(o)=\frac{\sum_{t:\,x_t=o}\gamma_t(j)}{\sum_{t=1}^{T}\gamma_t(j)}`, "M-step: expected counts, normalised"],
    ["p", "Each new estimate is \"expected number of times it happened\" divided by \"expected number of times it could have happened\", exactly like counting, but with soft counts."],
    ["h", "From scratch in NumPy: forward and Viterbi"],
    ["p", "Two practical points. Probabilities of long sequences underflow to 0, so the forward pass here is **scaled** (normalised at every step, keeping the log of the scale factors) and Viterbi works in **log space**. We also verify the forward result against brute-force enumeration of all 8 paths."],
    ["code", R`
import itertools
import numpy as np

states, acts = ["Rainy", "Sunny"], ["walk", "shop", "clean"]
pi = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.4, 0.6]])
B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
o = np.array([0, 1, 2])                                     # walk, shop, clean

def forward(o, pi, A, B):
    T, K = len(o), len(pi)
    alpha, c = np.zeros((T, K)), np.zeros(T)
    a = pi * B[:, o[0]]; c[0] = a.sum(); alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[:, o[t]]                 # sum over predecessors, then emit
        c[t] = a.sum(); alpha[t] = a / c[t]                 # scale to sum 1 to avoid underflow
    return alpha, c                                         # log P(x) = sum(log c)

def viterbi(o, pi, A, B):
    T, K = len(o), len(pi)
    logd, psi = np.zeros((T, K)), np.zeros((T, K), int)
    logd[0] = np.log(pi) + np.log(B[:, o[0]])
    for t in range(1, T):
        cand = logd[t - 1][:, None] + np.log(A)             # cand[i, j]: come from i, go to j
        psi[t] = cand.argmax(axis=0)
        logd[t] = cand.max(axis=0) + np.log(B[:, o[t]])
    path = [int(logd[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(psi[t][path[-1]]))
    return path[::-1], float(np.exp(logd[-1].max()))

alpha, c = forward(o, pi, A, B)
brute = sum(pi[z[0]] * B[z[0], o[0]] * np.prod([A[z[t - 1], z[t]] * B[z[t], o[t]] for t in range(1, 3)])
            for z in itertools.product(range(2), repeat=3))
print("P(walk, shop, clean) forward:", round(float(np.exp(np.log(c).sum())), 6), " brute force:", round(float(brute), 6))
path, p = viterbi(o, pi, A, B)
print("Viterbi path:", [states[i] for i in path], " probability:", round(p, 5))`, "P(walk, shop, clean) forward: 0.033612  brute force: 0.033612\nViterbi path: ['Sunny', 'Rainy', 'Rainy']  probability: 0.01344"],
    ["h", "From scratch in NumPy: Baum-Welch"],
    ["p", "We simulate 1000 days from a known HMM, hide the states, and let Baum-Welch rediscover the parameters from the observations alone. EM only finds a local optimum, so we try five random starts and keep the one with the best likelihood. The recovered states may come out in swapped order (the labels are arbitrary), so we sort them."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
trueA = np.array([[0.9, 0.1], [0.2, 0.8]])
trueB = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
T, z = 1000, 0
obs = np.zeros(T, int)
for t in range(T):                                          # simulate: hidden state, then an emission
    obs[t] = rng.choice(3, p=trueB[z])
    z = rng.choice(2, p=trueA[z])

def forward(o, pi, A, B):
    alpha, c = np.zeros((len(o), len(pi))), np.zeros(len(o))
    a = pi * B[:, o[0]]; c[0] = a.sum(); alpha[0] = a / c[0]
    for t in range(1, len(o)):
        a = (alpha[t - 1] @ A) * B[:, o[t]]; c[t] = a.sum(); alpha[t] = a / c[t]
    return alpha, c

def baum_welch(o, K, M, iters, seed):
    rng = np.random.default_rng(seed)
    pi = np.full(K, 1 / K)
    A, B = rng.dirichlet(np.ones(K) * 5, K), rng.dirichlet(np.ones(M) * 5, K)    # random start
    lls = []
    for _ in range(iters):
        alpha, c = forward(o, pi, A, B)
        beta = np.ones_like(alpha)
        for t in range(len(o) - 2, -1, -1):                 # backward pass, same scale factors
            beta[t] = A @ (B[:, o[t + 1]] * beta[t + 1]) / c[t + 1]
        lls.append(float(np.log(c).sum()))
        gamma = alpha * beta                                # E-step: P(z_t = i | all observations)
        xi = alpha[:-1, :, None] * A[None] * (B[:, o[1:]].T * beta[1:])[:, None, :] / c[1:, None, None]
        pi = gamma[0]                                       # M-step: expected counts, normalised
        A = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, None]
        B = np.stack([gamma[o == m].sum(axis=0) for m in range(M)], axis=1) / gamma.sum(axis=0)[:, None]
    return pi, A, B, lls

runs = [baum_welch(obs, 2, 3, 80, seed=s) for s in range(5)]       # EM is local: use several random starts
print("final log-likelihood of each start:", [round(r[3][-1], 1) for r in runs])
pi, A, B, lls = max(runs, key=lambda r: r[3][-1])                  # keep the best one
order = np.argsort(-B[:, 0]); A, B = A[np.ix_(order, order)], B[order]   # name the states consistently
print("log-likelihood: first", round(lls[0], 1), "-> last", round(lls[-1], 1), " never decreases:", bool(np.all(np.diff(lls) > -1e-8)))
print("learned A:\n", A.round(2), "\ntrue A:\n", trueA)
print("learned B:\n", B.round(2), "\ntrue B:\n", trueB)`, "final log-likelihood of each start: [-946.7, -1003.5, -1003.5, -946.7, -1000.2]\nlog-likelihood: first -1147.1 -> last -946.7  never decreases: True\nlearned A:\n [[0.92 0.08]\n [0.2  0.8 ]] \ntrue A:\n [[0.9 0.1]\n [0.2 0.8]]\nlearned B:\n [[0.73 0.18 0.09]\n [0.11 0.29 0.61]] \ntrue B:\n [[0.7 0.2 0.1]\n [0.1 0.3 0.6]]"],
    ["h", "The library version (runs on your computer)"],
    ["code", R`
# pip install hmmlearn
import numpy as np
from hmmlearn.hmm import CategoricalHMM

obs = np.array([0, 1, 2, 0, 0, 1, 2, 2, 1, 0]).reshape(-1, 1)      # observations as integers, one column
model = CategoricalHMM(n_components=2, n_iter=200, random_state=0).fit(obs)   # Baum-Welch
print(model.transmat_.round(2))
print(model.predict(obs))                                          # Viterbi path
print(model.score(obs))                                            # log P(observations) from the forward algorithm`, "", "hmmlearn", "local"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Exact, fast inference (O(T·K²)) for sequences", "Strong assumptions: Markov hidden state, observations independent given the state"],
      ["Learns from unlabelled sequences", "Baum-Welch finds local optima; run several starts"],
      ["Interpretable states and probabilities", "Hard to model long-range dependence (use RNNs / [transformers](transformers.html#attention-intuition) for that)"],
      ["Works with discrete or continuous (e.g. Gaussian) emissions", "Needs you to choose the number of states K"],
    ]],
    ["danger", "Confusing the **Viterbi path** with the **most likely state at each time**. Viterbi returns the best *whole* sequence. Picking $\\arg\\max_i\\gamma_t(i)$ separately at each $t$ (posterior decoding) can give a sequence that is impossible, for instance using a transition of probability 0."],
    ["warn", "Multiplying many probabilities underflows to 0.0 after a few hundred steps. Always use log-probabilities (Viterbi) or scaling (forward-backward). Also, a probability that is exactly 0 in A or B stays 0 forever in Baum-Welch; start from strictly positive values."],
    ["tip", "Practice: [Viterbi Decoding](practice.html#p/viterbi-decoding). The forward and Viterbi recursions are the same dynamic programme with sum replaced by max; the same trick appears in [beam search](llm.html#greedy-beam) (keep only the best few paths) and in CRFs for tagging."],
  ],
},
{
  id: "topic-lda", group: "Probabilistic Models", title: "Latent Dirichlet Allocation (Topic Models)",
  sub: "Discover the hidden themes in a pile of documents by modelling each document as a mix of topics and each topic as a distribution over words.",
  tags: ["text", "probabilistic"],
  blocks: [
    ["p", "Given thousands of news articles with no labels, what are they about? A **topic model** answers with two ideas: a **topic** is a list of words that tend to appear together (for example \"ball, team, goal, match\"), and a **document** is a *mixture* of topics (an article can be 70% politics and 30% economics). **Latent Dirichlet Allocation (LDA)** is the standard probabilistic version. (This LDA is unrelated to the discriminant analysis of the [first section](classical.html#lda); the abbreviation is shared by accident.) It ignores word order: each document is a bag of words."],
    ["h", "The generative story"],
    ["p", "LDA is a recipe for *writing* documents; fitting it means running the recipe backwards. With $K$ topics, a vocabulary of $V$ words, and Dirichlet hyperparameters $\\alpha,\\beta$:"],
    ["steps", [
      "For each topic $k$: draw its word distribution $\\boldsymbol\\phi_k\\sim\\mathrm{Dirichlet}(\\beta)$ (a probability vector over the $V$ words).",
      "For each document $d$: draw its topic mixture $\\boldsymbol\\theta_d\\sim\\mathrm{Dirichlet}(\\alpha)$ (a probability vector over the $K$ topics).",
      "For each word position $n$ in the document: draw a topic $z_{dn}\\sim\\mathrm{Categorical}(\\boldsymbol\\theta_d)$, then draw the word $w_{dn}\\sim\\mathrm{Categorical}(\\boldsymbol\\phi_{z_{dn}})$.",
    ]],
    ["diagram", "cml1_ldaplate", [], "The plate diagram of LDA. Boxes (plates) mean \"repeat\": N words per document, D documents, K topics. Only the words w (shaded) are observed."],
    ["p", "A **Dirichlet** distribution is a distribution over probability vectors (see [distributions](math.html#distributions)). A small $\\alpha<1$ prefers vectors where one or two entries dominate: documents about few topics. A small $\\beta<1$ makes topics use few words. That sparsity is what makes the learned topics readable."],
    ["h", "Inference: collapsed Gibbs sampling"],
    ["p", "We observe only the words and want the topic $z_{dn}$ of every word position (from which $\\theta$ and $\\phi$ follow). The exact posterior is intractable, so we sample from it with [Gibbs sampling](classical.html#mcmc): resample one word's topic at a time given all the others. Because the Dirichlet is conjugate to the categorical, $\\boldsymbol\\theta$ and $\\boldsymbol\\phi$ can be integrated out (\"collapsed\"), leaving only counts:"],
    ["math", R`P(z_{dn}=k\mid\mathbf z_{-dn},\mathbf w)\;\propto\;\big(n_{dk}^{-dn}+\alpha\big)\cdot\frac{n_{kw}^{-dn}+\beta}{n_{k}^{-dn}+V\beta}`, "n_dk: words in document d assigned to topic k · n_kw: times word w is assigned to topic k · n_k: all words in topic k · the superscript means: do not count the word being resampled"],
    ["p", "Two factors, both sensible. **How much does this document like topic k?** ($n_{dk}+\\alpha$: if the document already has many words in topic $k$, this one probably does too.) **How much does topic k like this word?** (the share of topic $k$'s words that are $w$). After many sweeps the estimates are"],
    ["math", R`\hat\theta_{dk}=\frac{n_{dk}+\alpha}{n_d+K\alpha},\qquad \hat\phi_{kw}=\frac{n_{kw}+\beta}{n_k+V\beta}`],
    ["note", "Tiny example. $K=2$, $V=6$, $\\alpha=0.5$, $\\beta=0.1$. Resample the word \"ball\" in a document where, apart from this word, 3 words are in topic 0 and 1 in topic 1. Topic 0 has 20 words in total, 5 of them \"ball\"; topic 1 has 15 words, none of them \"ball\". Then topic 0 scores $(3+0.5)\\cdot\\frac{5+0.1}{20+0.6}=0.866$ and topic 1 scores $(1+0.5)\\cdot\\frac{0+0.1}{15+0.6}=0.0096$. So the word goes to topic 0 with probability $0.866/(0.866+0.0096)\\approx0.99$. The sampler is greedy-ish but never certain."],
    ["h", "From scratch in NumPy"],
    ["p", "A toy corpus: 10 short documents, four about football, four about cooking, and two that mix both."],
    ["code", R`
import numpy as np

docs = ["ball team goal match player score", "team player coach match win goal",
        "goal ball score win coach player", "match ball team score player win",
        "pasta sauce garlic oil boil onion", "oil garlic pan fry onion sauce",
        "pasta boil sauce onion pan fry", "garlic pasta oil pan sauce boil",
        "team goal pasta sauce match garlic", "player score oil pan win fry"]
docs = [d.split() for d in docs]
vocab = sorted({w for d in docs for w in d}); w2i = {w: i for i, w in enumerate(vocab)}
V, K, alpha, beta = len(vocab), 2, 0.5, 0.1

rng = np.random.default_rng(0)
tokens = [(d, w2i[w]) for d, doc in enumerate(docs) for w in doc]
z = rng.integers(K, size=len(tokens))                      # start with random topics
ndk, nkw, nk = np.zeros((len(docs), K)), np.zeros((K, V)), np.zeros(K)
for (d, w), k in zip(tokens, z):
    ndk[d, k] += 1; nkw[k, w] += 1; nk[k] += 1

for sweep in range(300):
    for i, (d, w) in enumerate(tokens):
        k = z[i]
        ndk[d, k] -= 1; nkw[k, w] -= 1; nk[k] -= 1          # take this word out of the counts
        p = (ndk[d] + alpha) * (nkw[:, w] + beta) / (nk + V * beta)
        k = rng.choice(K, p=p / p.sum())                    # resample its topic
        z[i] = k
        ndk[d, k] += 1; nkw[k, w] += 1; nk[k] += 1

phi = (nkw + beta) / (nk[:, None] + V * beta)
theta = (ndk + alpha) / (ndk.sum(axis=1, keepdims=True) + K * alpha)
for k in range(K):
    print(f"topic {k}:", [vocab[i] for i in np.argsort(-phi[k])[:6]])
for d in (0, 4, 8, 9):
    print(f"doc {d} ({' '.join(docs[d][:3])} ...): topic mix", theta[d].round(2))`, "topic 0: ['sauce', 'oil', 'pan', 'garlic', 'pasta', 'onion']\ntopic 1: ['player', 'match', 'win', 'goal', 'score', 'team']\ndoc 0 (ball team goal ...): topic mix [0.07 0.93]\ndoc 4 (pasta sauce garlic ...): topic mix [0.93 0.07]\ndoc 8 (team goal pasta ...): topic mix [0.5 0.5]\ndoc 9 (player score oil ...): topic mix [0.5 0.5]"],
    ["p", "Without ever being told the labels, the sampler separates football words from cooking words (topic numbers may come out swapped: they are just names). The first four documents come out almost purely one topic; the two mixed documents get a split mixture."],
    ["h", "The scikit-learn version"],
    ["p", "scikit-learn's `LatentDirichletAllocation` uses **variational Bayes** instead of Gibbs sampling: it approximates the posterior with a simpler distribution that it optimises, which is faster on large corpora. The input is a document-term count matrix."],
    ["code", R`
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

docs = ["ball team goal match player score", "team player coach match win goal",
        "goal ball score win coach player", "match ball team score player win",
        "pasta sauce garlic oil boil onion", "oil garlic pan fry onion sauce",
        "pasta boil sauce onion pan fry", "garlic pasta oil pan sauce boil",
        "team goal pasta sauce match garlic", "player score oil pan win fry"]
cv = CountVectorizer()
X = cv.fit_transform(docs)                                  # (documents, vocabulary) word counts
vocab = np.array(cv.get_feature_names_out())

lda = LatentDirichletAllocation(n_components=2, doc_topic_prior=0.5, topic_word_prior=0.1,
                                learning_method="batch", max_iter=100, random_state=0).fit(X)
for k, comp in enumerate(lda.components_):
    print(f"topic {k}:", vocab[np.argsort(-comp)[:6]].tolist())
print("topic mixture of docs 0, 4, 8, 9:\n", lda.transform(X)[[0, 4, 8, 9]].round(2))
print("perplexity:", round(float(lda.perplexity(X)), 2), "(lower is better)")`, "topic 0: ['player', 'score', 'win', 'match', 'team', 'goal']\ntopic 1: ['sauce', 'garlic', 'pasta', 'oil', 'pan', 'boil']\ntopic mixture of docs 0, 4, 8, 9:\n [[0.93 0.07]\n [0.07 0.93]\n [0.5  0.5 ]\n [0.5  0.5 ]]\nperplexity: 21.88 (lower is better)"],
    ["table", ["Model", "Documents are", "Topics are", "Notes"], [
      ["**LDA**", "mixtures of topics", "distributions over words", "Probabilistic, sparse via Dirichlet priors"],
      ["[NMF](classical.html#nmf)", "non-negative sums of topics", "non-negative word weights", "No probabilities; often sharper topics on short text"],
      ["LSA / SVD", "linear combinations", "can be negative", "Fast but hard to read"],
      ["Mixture of unigrams (cf. [Naive Bayes](ml.html#naive-bayes))", "**one** topic each", "word distributions", "Too rigid: a document is one topic"],
    ]],
    ["danger", "Skipping the preprocessing. Without removing stop words (\"the\", \"of\") and very rare words, the top words of every topic are the same common words and the topics are useless. Use `CountVectorizer(stop_words=\"english\", min_df=5, max_df=0.5)` on real text."],
    ["warn", "$K$ is not learned. Try several, then judge by topic **coherence** (do the top words make sense together?) and by held-out perplexity; the lowest perplexity is not always the most readable. LDA is also weak on very short documents (tweets) because there are too few words per document to estimate a mixture. As of 2026 embedding-based topic tools (clustering sentence embeddings) are a popular alternative, but LDA remains a fast, transparent baseline."],
    ["tip", "The same Gibbs recipe works for any model with discrete latent variables. LDA is also a clean example of \"collapsing\": integrating out parameters whose prior is conjugate, leaving only counts."],
  ],
},
{
  id: "mcmc", group: "Probabilistic Models", title: "Markov Chains & MCMC: Metropolis-Hastings & Gibbs",
  sub: "When you can't draw samples from a distribution directly, build a random walk whose long-run visits follow it.",
  tags: ["sampling", "bayesian"],
  blocks: [
    ["p", "Bayesian methods end with a posterior distribution $p(\\theta\\mid\\text{data})=\\tilde p(\\theta)/Z$, and the normaliser $Z$ is an integral you usually cannot compute. Without $Z$ you can't draw samples by the usual tricks, nor take averages like $\\mathbb E[f(\\theta)]$. **Markov chain Monte Carlo (MCMC)** gets around this: it builds a random walk that wanders over the parameter space and spends time in each region in proportion to its probability. A long walk is then a set of (correlated) samples from the target, and a histogram of them is the distribution. All you need is to evaluate $\\tilde p$, the *unnormalised* density."],
    ["h", "Markov chains and stationary distributions"],
    ["p", "A **Markov chain** moves between states with probabilities that depend only on the current state (the same Markov property as in [HMMs](classical.html#hmm)). If $\\mathbf P$ is the transition matrix, a distribution $\\boldsymbol\\pi$ is **stationary** when $\\boldsymbol\\pi\\mathbf P=\\boldsymbol\\pi$: run one step and nothing changes. Under mild conditions (the chain can get from anywhere to anywhere, and does not cycle) it has exactly one stationary distribution and the chain converges to it from any start. MCMC designs the chain so that this distribution is the target."],
    ["p", "A sufficient design rule is **detailed balance**: $p(x)\\,T(x\\to x')=p(x')\\,T(x'\\to x)$ for all pairs. Summing over $x$ shows that $p$ is then stationary. In words: in equilibrium, as much probability flows from $x$ to $x'$ as flows back. Example: the 2-state chain with $P_{12}=0.1$, $P_{21}=0.2$ has stationary $\\boldsymbol\\pi=(2/3,\\,1/3)$, and indeed $\\frac23\\cdot0.1=\\frac13\\cdot0.2$."],
    ["code", R`
import numpy as np

P = np.array([[0.9, 0.1], [0.2, 0.8]])                     # P[i, j] = probability of moving i -> j
for start in ([1.0, 0.0], [0.0, 1.0]):
    dist = np.array(start)
    for _ in range(50):
        dist = dist @ P                                    # one step of the chain
    print("start", start, "-> after 50 steps", dist.round(4))

vals, vecs = np.linalg.eig(P.T)                            # stationary = left eigenvector with eigenvalue 1
pi = np.real(vecs[:, np.argmax(np.real(vals))]); pi /= pi.sum()
print("stationary distribution:", pi.round(4), " pi @ P == pi:", bool(np.allclose(pi @ P, pi)))
print("detailed balance:", round(float(pi[0] * P[0, 1]), 4), "=", round(float(pi[1] * P[1, 0]), 4))`, "start [1.0, 0.0] -> after 50 steps [0.6667 0.3333]\nstart [0.0, 1.0] -> after 50 steps [0.6667 0.3333]\nstationary distribution: [0.6667 0.3333]  pi @ P == pi: True\ndetailed balance: 0.0667 = 0.0667"],
    ["h", "Metropolis-Hastings"],
    ["p", "Here is the recipe that builds a chain with the stationary distribution you want. At the current point $x$:"],
    ["steps", [
      "**Propose** a move $x'\\sim q(x'\\mid x)$. The simplest choice is a random walk: $x'=x+\\varepsilon$ with $\\varepsilon\\sim\\mathcal N(0,s^2)$.",
      "**Accept** it with probability $a=\\min\\!\\Big(1,\\ \\dfrac{\\tilde p(x')\\,q(x\\mid x')}{\\tilde p(x)\\,q(x'\\mid x)}\\Big)$. For a symmetric proposal the $q$'s cancel and $a=\\min(1,\\tilde p(x')/\\tilde p(x))$.",
      "If accepted, move to $x'$. **If rejected, stay at $x$ and count $x$ again.**",
    ]],
    ["p", "Uphill moves (more probable) are always accepted, downhill moves sometimes, with probability equal to how much less likely they are. That keeps the walker mostly on the high-probability ground but lets it visit the tails in the right proportion. The normaliser $Z$ cancels in the ratio, which is why $\\tilde p$ is enough. Why it works: the transition probability is $T(x\\to x')=q(x'\\mid x)\\,a(x\\to x')$, and"],
    ["math", R`\tilde p(x)\,q(x'\mid x)\,\min\!\Big(1,\tfrac{\tilde p(x')q(x\mid x')}{\tilde p(x)q(x'\mid x)}\Big)=\min\big(\tilde p(x)q(x'\mid x),\ \tilde p(x')q(x\mid x')\big)`, "the right side is symmetric in x and x′, so detailed balance holds"],
    ["viz", "cml1_mcmc"],
    ["p", "The target (orange) has two bumps. The chain starts far away at $x=-6$. Watch the first iterations (each shows the proposal, the ratio and the coin flip $u$), then the trace and histogram as samples accumulate. Now move the **step size**. Tiny steps accept nearly everything but crawl; huge steps are almost always rejected so the chain sticks. A middling step, with acceptance around 20 to 50 percent, explores both bumps."],
    ["h", "From scratch in NumPy"],
    ["code", R`
import numpy as np

def p_tilde(x):                                            # unnormalised target: 2 bumps (we never need Z)
    return 0.5 * np.exp(-0.5 * ((x + 2) / 0.7) ** 2) / 0.7 + 0.5 * np.exp(-0.5 * ((x - 2) / 1.0) ** 2) / 1.0

def metropolis(step, n=10000, x0=-6.0, seed=0):
    rng = np.random.default_rng(seed)
    x, out, accepted = x0, np.empty(n), 0
    for i in range(n):
        xp = x + step * rng.standard_normal()              # propose a random-walk move
        if rng.random() < p_tilde(xp) / p_tilde(x):        # accept with probability min(1, ratio)
            x, accepted = xp, accepted + 1
        out[i] = x                                         # a rejection repeats the old x
    return out, accepted / n

for step in (0.1, 1.0, 3.0, 20.0):
    s, acc = metropolis(step)
    s = s[1000:]                                           # discard burn-in
    print(f"step {step:5.1f}: accept {acc:.2f}  mean {s.mean():5.2f}  sd {s.std():.2f}  share in right bump {np.mean(s > 0):.2f}")
print("exact: mean 0.00  sd 2.18  share in right bump 0.50")`, "step   0.1: accept 0.96  mean -0.50  sd 1.97  share in right bump 0.33\nstep   1.0: accept 0.69  mean  0.18  sd 2.18  share in right bump 0.53\nstep   3.0: accept 0.46  mean  0.10  sd 2.21  share in right bump 0.51\nstep  20.0: accept 0.10  mean  0.04  sd 2.13  share in right bump 0.50\nexact: mean 0.00  sd 2.18  share in right bump 0.50"],
    ["p", "With the tiny step the chain accepts 96% of its proposals but moves so little that, even after 10,000 iterations, it is still wandering inside the left bump and has the wrong mean and mode weights. The moderate steps (1 and 3) recover the mean, the spread and the 50/50 split of the two bumps efficiently. The huge step still gets the summary right here, but it accepts only about one proposal in ten, so it has far fewer distinct samples: many repeated values per effective draw."],
    ["h", "Gibbs sampling"],
    ["p", "In many models the full distribution is hard but each **conditional** is easy. **Gibbs sampling** updates one variable at a time by drawing it from its conditional distribution given all the others: $x_j\\sim p(x_j\\mid\\mathbf x_{-j})$, cycling through $j$. It is Metropolis-Hastings with proposal equal to the conditional, for which the acceptance probability works out to exactly 1: no tuning and no rejections. For a standard bivariate normal with correlation $\\rho$ the conditionals are $x\\mid y\\sim\\mathcal N(\\rho y,\\,1-\\rho^2)$ and $y\\mid x\\sim\\mathcal N(\\rho x,\\,1-\\rho^2)$. (Collapsed Gibbs for [topic models](classical.html#topic-lda) is the same idea.)"],
    ["code", R`
import numpy as np

def gibbs(rho, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = y = 0.0; xs = np.empty(n); ys = np.empty(n)
    sd = np.sqrt(1 - rho**2)
    for i in range(n):
        x = rng.normal(rho * y, sd)                        # x | y
        y = rng.normal(rho * x, sd)                        # y | x
        xs[i], ys[i] = x, y
    return xs[500:], ys[500:]

for rho in (0.5, 0.99):
    xs, ys = gibbs(rho)
    r1 = np.corrcoef(xs[:-1], xs[1:])[0, 1]                # lag-1 autocorrelation of the chain
    ess = len(xs) * (1 - r1) / (1 + r1)                    # rough effective number of independent samples
    print(f"rho {rho}: sample corr {np.corrcoef(xs, ys)[0, 1]:.2f}  sd of x {xs.std():.2f}  lag-1 autocorr {r1:.2f}  effective samples ~ {ess:.0f} of {len(xs)}")`, "rho 0.5: sample corr 0.49  sd of x 1.00  lag-1 autocorr 0.25  effective samples ~ 11591 of 19500\nrho 0.99: sample corr 0.99  sd of x 1.05  lag-1 autocorr 0.98  effective samples ~ 180 of 19500"],
    ["p", "Strong correlation makes Gibbs take tiny axis-parallel steps along the narrow diagonal ridge: neighbouring samples are almost identical, so 19,500 draws are worth only a few hundred independent ones. The **effective sample size** is the honest count."],
    ["h", "Is the chain good? Diagnostics"],
    ["list", [
      "**Burn-in**: the first samples still remember the starting point. Discard them (the histogram in the visualizer is visibly biased at first).",
      "**Trace plot**: should look like a fuzzy caterpillar moving around the whole range, not a staircase stuck for long stretches.",
      "**Autocorrelation / effective sample size**: samples are correlated; ESS tells you how many independent ones they are worth.",
      "**Several chains from different starts** should agree. The $\\hat R$ statistic compares between-chain and within-chain variance; values near 1 are good.",
      "**Acceptance rate**: for random-walk Metropolis the rule of thumb is about 44% in one dimension and about 23% in high dimensions.",
    ]],
    ["table", ["Method", "Idea", "Needs", "Good for"], [
      ["Metropolis-Hastings", "propose, accept with ratio", "$\\tilde p(x)$ only", "Any target; simple to code; needs a step size"],
      ["Gibbs", "sample each variable from its conditional", "all conditionals", "Models with easy conditionals (mixtures, topic models); no tuning"],
      ["Hamiltonian MC / NUTS", "use gradients to make long, informed moves", "$\\nabla\\log\\tilde p$", "High-dimensional continuous posteriors; the default in Stan and PyMC"],
    ]],
    ["h", "The library version (runs on your computer)"],
    ["code", R`
# pip install pymc arviz
import numpy as np
import pymc as pm
import arviz as az

rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, 30)
y = -0.3 + 0.5 * x + 0.2 * rng.standard_normal(30)

with pm.Model():
    b = pm.Normal("b", 0, 1)                               # intercept prior
    w = pm.Normal("w", 0, 1)                               # slope prior
    sigma = pm.HalfNormal("sigma", 1)                      # noise sd prior
    pm.Normal("y", mu=b + w * x, sigma=sigma, observed=y)
    idata = pm.sample(1000, tune=1000, chains=4, random_seed=0)   # NUTS, a gradient-based MCMC

print(az.summary(idata))                                   # means, sds, ESS and r_hat for each parameter`, "", "PyMC", "local"],
    ["danger", "Treating MCMC samples as independent, or running one short chain and trusting it. Check the trace, the effective sample size and, ideally, several chains. A single chain can sit happily in one bump of a bimodal posterior and never discover the other (try the step size 0.1 above)."],
    ["warn", "Do not tune the proposal width *while keeping* the samples: adaptation breaks the Markov property. Tune during warm-up, then freeze the settings and sample. Also compute with $\\log\\tilde p$ and compare logs (`log u < logp_new - logp_old`): densities underflow to 0 in high dimensions. Practice: [Metropolis-Hastings Sampler](practice.html#p/metropolis-hastings-sampler)."],
    ["tip", "The same machinery explains how modern generative models are sampled and how Bayesian deep learning approximates posteriors. For LLM decoding, though, you sample directly from the model's softmax; no chain is needed, see [sampling](llm.html#sampling)."],
  ],
},
);
