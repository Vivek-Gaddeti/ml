/* Machine Learning, part 2: unsupervised learning, evaluation, practical
   Outputs written as "?" are filled in by running the code: node tools/fill.js data/ml-2.js */
ML.push(
{
  id: "kmeans", group: "Unsupervised", title: "k-Means Clustering",
  sub: "Find k groups in unlabelled data by alternating two simple steps: assign points to the nearest centre, then move each centre to the middle of its points.",
  tags: ["clustering"],
  blocks: [
    ["p", "So far every model had **labels**: the right answer for each example. In **unsupervised learning** there are no labels. You only have the data, and you want to find structure in it. The most common task is **clustering**: split the examples into groups so that points in a group are similar and points in different groups are not. Typical uses are customer segments, grouping similar documents, compressing colours in an image, and a first look at an unknown dataset."],
    ["h", "The idea"],
    ["p", "Imagine you must open k pizza shops in a city so that everyone lives close to one. Start by dropping k shops at random. Every house walks to its nearest shop (**assign**). Then each shop moves to the average position of the houses it serves (**update**). Houses may now prefer a different shop, so repeat. When nobody changes shop any more, you are done. That is **k-means**, and the shops are called **centroids**."],
    ["viz", "ml2_kmeans"],
    ["p", "Step through it once. Then change the **random start** slider: with a bad start k-means can settle in a worse arrangement. It always converges, but only to a *local* optimum."],
    ["h", "The maths"],
    ["p", "k-means minimises the **inertia**, also called the within-cluster sum of squares (WCSS): the total squared distance from every point to the centroid of its own cluster."],
    ["math", R`J = \sum_{j=1}^{k} \sum_{\mathbf x_i \in C_j} \lVert \mathbf x_i - \boldsymbol\mu_j \rVert^2 \qquad \boldsymbol\mu_j = \frac{1}{|C_j|}\sum_{\mathbf x_i \in C_j} \mathbf x_i`, "C_j: the points in cluster j · μ_j: its centroid (the mean of those points)"],
    ["list", [
      "**Assign step**: for fixed centroids, the best cluster for each point is the nearest centroid. This can only lower $J$.",
      "**Update step**: for fixed assignments, the best centre of a cluster is the mean of its points (the mean minimises squared distance). This can only lower $J$ too.",
      "$J$ never goes up and there are finitely many assignments, so the loop must stop. It stops at a local minimum, not necessarily the best one.",
    ]],
    ["h", "From scratch in NumPy"],
    ["p", "The distance from every point to every centroid is one broadcasted expression (see [broadcasting](pyml.html#broadcasting)): `X[:, None, :] - C[None, :, :]` has shape (n, k, d)."],
    ["code", R`
import numpy as np
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)

def kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)]            # start: k random data points
    for it in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)  # (n, k) squared distances
        labels = d.argmin(axis=1)                               # assign: nearest centroid
        new_C = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else C[j]
                          for j in range(k)])                   # update: mean of each cluster
        if np.allclose(new_C, C):                               # nothing moved: converged
            break
        C = new_C
    inertia = d[np.arange(len(X)), labels].sum()
    return C, labels, inertia, it

C, labels, inertia, it = kmeans(X, k=4)
print("iterations:", it)
print("cluster sizes:", np.bincount(labels).tolist())
print("centroids:\n", C.round(2))
print("inertia:", round(float(inertia), 1))`, "iterations: 15\ncluster sizes: [74, 72, 81, 73]\ncentroids:\n [[ 1.98  0.83]\n [-1.66  2.76]\n [ 0.85  4.44]\n [-1.44  7.78]]\ninertia: 369.5"],
    ["h", "Choosing k"],
    ["p", "k is *your* choice: the algorithm cannot tell you how many groups are real. Inertia always falls as k grows (with k = n it is zero), so you look for the **elbow**, the point after which extra clusters stop helping much."],
    ["diagram", "ml2_elbow", [], "Inertia against k on four blobs: a steep fall until k = 4, then a slow crawl. The bend is the elbow."],
    ["code", R`
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)

for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    sil = silhouette_score(X, km.labels_)
    print(f"k = {k}  inertia {km.inertia_:8.1f}  silhouette {sil:.3f}")`, "k = 2  inertia   1348.9  silhouette 0.496\nk = 3  inertia    708.8  silhouette 0.515\nk = 4  inertia    369.5  silhouette 0.580\nk = 5  inertia    326.3  silhouette 0.503\nk = 6  inertia    292.4  silhouette 0.430\nk = 7  inertia    260.2  silhouette 0.396"],
    ["p", "The **silhouette score** looks at each point: how close it is to its own cluster compared with the nearest other cluster. It runs from -1 (probably in the wrong cluster) to +1 (well inside its cluster), and unlike inertia it has a peak you can pick. Higher is better; pick the k where it peaks, and sanity-check the clusters by looking at them."],
    ["h", "The scikit-learn version"],
    ["code", R`
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)

km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=0).fit(X)
print("inertia:", round(float(km.inertia_), 1))
print("centroids sorted by x:\n", km.cluster_centers_[np.argsort(km.cluster_centers_[:, 0])].round(2))
print("new point (0, 3) goes to cluster", int(km.predict([[0.0, 3.0]])[0]))`, "inertia: 369.5\ncentroids sorted by x:\n [[-1.66  2.76]\n [-1.44  7.78]\n [ 0.85  4.44]\n [ 1.98  0.83]]\nnew point (0, 3) goes to cluster 0"],
    ["table", ["Setting", "Meaning"], [
      ["`n_clusters`", "k, the number of clusters. You must choose it."],
      ["`init=\"k-means++\"`", "Default. Picks spread-out starting centres: far better than purely random starts."],
      ["`n_init`", "How many times to run from different starts and keep the lowest inertia. Use 10 or more."],
      ["`max_iter`, `tol`", "Safety limits. Convergence usually takes under 30 iterations."],
    ]],
    ["h", "Scale your features first"],
    ["p", "k-means uses straight-line distance, so a feature measured in big numbers (income in dollars) drowns out one measured in small numbers (age in years). Try the toggle: with raw features the clusters split on income only; after standardizing they use both."],
    ["viz", "ml2_scaling"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Simple, fast: roughly O(n·k·d) per iteration, scales to millions of rows", "You must choose k"],
      ["Easy to explain and to use the centroids as a summary", "Assumes round, similar-sized, similar-density blobs"],
      ["Output is useful as features (distance to each centroid)", "Sensitive to scale, outliers and the random start"],
    ]],
    ["danger", "Running k-means on unscaled features, then reading meaning into clusters that are really just \"the feature with the biggest numbers\". Standardize first (see [feature engineering and scaling](ml.html#feature-engineering)). Also: a clustering always *returns* k clusters, even on pure noise, so judge them with the silhouette score and by looking at examples."],
    ["warn", "k-means fails on non-round shapes such as two crescents or rings, and on clusters with very different densities. For those try [DBSCAN](ml.html#dbscan)."],
    ["tip", "Practice: [One k-Means Iteration](practice.html#p/kmeans-one-iteration) and [k-Means from Scratch](practice.html#p/kmeans-from-scratch)."],
  ],
},
{
  id: "dbscan", group: "Unsupervised", title: "DBSCAN & Hierarchical Clustering",
  sub: "Two ways to cluster without choosing round blobs: grow groups from dense regions, or merge the closest points step by step into a tree.",
  tags: ["clustering"],
  blocks: [
    ["p", "[k-means](ml.html#kmeans) draws round clusters and forces **every** point into one. Real data is often not like that: clusters can be curved, and some points are just noise. Two classic algorithms handle this."],
    ["h", "DBSCAN: clusters are dense regions"],
    ["p", "**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) says: a cluster is a crowd. It has two settings:"],
    ["list", [
      "**eps** (ε): the radius of a neighbourhood around a point.",
      "**minPts** (`min_samples`): how many points must be inside that radius for the area to count as dense.",
    ]],
    ["p", "Every point is then one of three things. A **core point** has at least minPts points (itself included) within eps. A **border point** is within eps of a core point but is not core itself. Everything else is **noise** (label -1). A cluster is a set of core points that can reach each other by hopping through neighbourhoods, plus their border points."],
    ["viz", "ml2_dbscan"],
    ["p", "Change eps and minPts. A small eps shatters the data into many tiny clusters and lots of noise; a large eps glues everything into one. Because DBSCAN follows density, it follows curved shapes too."],
    ["h", "From scratch in NumPy"],
    ["code", R`
import numpy as np
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=200, noise=0.06, random_state=0)

def dbscan(X, eps, min_pts):
    n = len(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))  # all pairwise distances
    neighbours = [np.where(D[i] <= eps)[0] for i in range(n)]        # includes the point itself
    core = np.array([len(nb) >= min_pts for nb in neighbours])
    labels = np.full(n, -1)                                          # -1 = noise
    cluster = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cluster                                          # start a new cluster at a core point
        stack = [i]
        while stack:                                                 # grow it
            p = stack.pop()
            if not core[p]:
                continue                                             # border points join but do not spread
            for q in neighbours[p]:
                if labels[q] == -1:
                    labels[q] = cluster
                    stack.append(q)
        cluster += 1
    return labels

labels = dbscan(X, eps=0.2, min_pts=5)
print("clusters found:", int(labels.max()) + 1)
print("noise points:", int((labels == -1).sum()))
print("sizes:", np.bincount(labels[labels >= 0]).tolist())`, "clusters found: 2\nnoise points: 0\nsizes: [100, 100]"],
    ["h", "k-means vs DBSCAN on two crescents"],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

X, y = make_moons(n_samples=300, noise=0.06, random_state=0)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
db = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)

print("k-means agreement with truth (ARI):", round(adjusted_rand_score(y, km), 3))
print("DBSCAN  agreement with truth (ARI):", round(adjusted_rand_score(y, db), 3))

fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
for a, lab, name in [(ax[0], km, "k-means"), (ax[1], db, "DBSCAN")]:
    a.scatter(X[:, 0], X[:, 1], c=lab, cmap="coolwarm", s=12)
    a.set_title(name)
plt.tight_layout()
plt.show()`, "k-means agreement with truth (ARI): 0.234\nDBSCAN  agreement with truth (ARI): 1.0"],
    ["p", "The **adjusted Rand index** (ARI) compares two ways of grouping the same points: 1 is identical, around 0 is random. We only use the true labels here to *grade* the clustering; DBSCAN never sees them."],
    ["h", "Choosing eps and minPts"],
    ["list", [
      "**minPts**: a common start is `2 * number_of_features` (and at least 3 to 5). Noisier data needs a larger value.",
      "**eps**: plot each point's distance to its k-th nearest neighbour (k = minPts), sorted. The curve has an elbow; the distance there is a good eps.",
      "Scale the features first: eps is one distance for all directions.",
    ]],
    ["h", "Hierarchical (agglomerative) clustering"],
    ["p", "Instead of picking k up front, build a whole **tree**. Start with every point as its own cluster. Repeatedly merge the two closest clusters until one remains. Record the merge distances; the tree is called a **dendrogram**. Cutting it at a height gives you any number of clusters you like."],
    ["diagram", "ml2_dendrogram", [], "A dendrogram of eight points. Short branches join very similar points; the dashed line cuts the tree into three clusters."],
    ["p", "\"Closest\" for two *groups* needs a rule, the **linkage**:"],
    ["table", ["Linkage", "Distance between two clusters", "Tends to produce"], [
      ["single", "closest pair of points", "long chains (can follow curves, but also bridges noise)"],
      ["complete", "farthest pair of points", "compact, similar-diameter clusters"],
      ["average", "mean of all pairwise distances", "a compromise"],
      ["ward", "increase in total within-cluster variance if merged", "round, equal-size clusters (like k-means). The usual default."],
    ]],
    ["code", R`
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=60, centers=3, cluster_std=0.7, random_state=1)

Z = linkage(X, method="ward")                  # (n-1) merges: [cluster a, cluster b, distance, size]
print("last 3 merge distances:", Z[-3:, 2].round(2).tolist())
labels = fcluster(Z, t=3, criterion="maxclust")  # cut the tree into 3 clusters
print("sizes from scipy:", np.bincount(labels)[1:].tolist())

agg = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X)
print("sizes from sklearn:", sorted(np.bincount(agg.labels_).tolist()))`, "last 3 merge distances: [3.84, 23.62, 63.07]\nsizes from scipy: [20, 20, 20]\nsizes from sklearn: [20, 20, 20]"],
    ["p", "A big jump in the last merge distances means the last merges joined far-apart groups: cut just before that jump. The three numbers above show it."],
    ["table", ["", "k-means", "DBSCAN", "Hierarchical"], [
      ["Must choose k?", "Yes", "No (choose eps, minPts)", "No (cut the tree anywhere)"],
      ["Cluster shapes", "Round blobs", "Any shape", "Depends on linkage"],
      ["Finds outliers?", "No", "Yes (label -1)", "No"],
      ["Handles different densities?", "Poorly", "Poorly (one eps)", "Moderately"],
      ["Cost", "Fast, O(n·k·d)", "O(n log n) with an index, O(n²) worst", "O(n²) memory and about O(n²) to O(n³) time: up to a few thousand points"],
      ["Gives a hierarchy?", "No", "No", "Yes"],
    ]],
    ["danger", "Using one eps on data whose clusters have very different densities (a tight cluster and a loose one): either the tight one is right and the loose one becomes noise, or the loose one is right and the tight ones merge. Also forgetting to scale the features."],
    ["tip", "Rule of thumb: try k-means first; if the clusters look unnatural (curved, uneven, noisy), try DBSCAN; if you want to *see* the grouping structure on a small dataset, use a dendrogram."],
  ],
},
{
  id: "pca", group: "Unsupervised", title: "PCA & Dimensionality Reduction",
  sub: "Rotate the data onto new axes ordered by how much it varies, then keep only the first few: fewer numbers, most of the information.",
  tags: ["dimensionality reduction"],
  blocks: [
    ["p", "A dataset with 100 features is hard to plot, slow to train on, and often full of redundancy: many features move together (height and weight, or neighbouring pixels). **Principal Component Analysis (PCA)** finds a smaller set of new features, built as mixtures of the old ones, that keep as much of the **variation** in the data as possible. Uses: visualising in 2-D, speeding up models, removing noise and correlation, and compressing data."],
    ["h", "The idea"],
    ["p", "Take a cloud of points that looks like a tilted ellipse. Describing each point with (x, y) uses two numbers, but almost all the information is along the long axis of the ellipse. If we draw a new axis along that direction and measure each point's position on it, one number per point keeps most of the story. That axis is the **first principal component**: the direction in which the data is most spread out. The second component is the best direction at a right angle to the first, and so on."],
    ["viz", "ml2_pca"],
    ["p", "Turn the direction slider. The blue point cloud is projected onto the line, and the variance of those shadows is shown. It is largest exactly along the long axis, which is what PCA finds for you. The dashed segments are the **reconstruction errors**: PCA also gives the line that is closest to all the points."],
    ["h", "The maths"],
    ["steps", [
      "**Centre** the data: subtract each feature's mean, so the cloud is around the origin.",
      "Compute the **covariance matrix** $\\Sigma = \\frac{1}{n-1}X^\\top X$ (d × d). Its diagonal holds the variances and its off-diagonal entries show which features move together.",
      "Take its **eigenvectors** (see [eigenvalues and eigenvectors](math.html#eigen)). They are the principal components: orthogonal directions. The matching **eigenvalues** are the variance along each direction.",
      "Sort by eigenvalue, keep the top $m$ eigenvectors as a matrix $W$ (d × m), and **project**: $Z = XW$.",
    ]],
    ["math", R`\Sigma\,\mathbf v_j = \lambda_j \mathbf v_j \qquad \text{explained variance ratio of component } j = \frac{\lambda_j}{\sum_i \lambda_i}`],
    ["p", "In practice libraries use the **SVD** (see [SVD](math.html#svd)) of the centred data, $X = U S V^\\top$, which is more accurate than forming the covariance. The rows of $V^\\top$ are the components, and the eigenvalues are $\\lambda_j = s_j^2/(n-1)$ with $s_j$ the singular values."],
    ["h", "From scratch in NumPy"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_iris

X = load_iris().data                       # 150 flowers x 4 measurements
Xc = X - X.mean(axis=0)                    # step 1: centre

# way 1: eigen-decomposition of the covariance matrix
cov = Xc.T @ Xc / (len(X) - 1)
vals, vecs = np.linalg.eigh(cov)           # eigh: for symmetric matrices, ascending order
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]
print("eigenvalues:", vals.round(3).tolist())
print("explained variance ratio:", (vals / vals.sum()).round(3).tolist())

# way 2: SVD of the centred data (what libraries do)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
print("variance from SVD:", (s ** 2 / (len(X) - 1)).round(3).tolist())
print("same directions?", bool(np.allclose(np.abs(Vt), np.abs(vecs.T))))   # signs may flip

Z = Xc @ Vt[:2].T                          # project onto the first 2 components
print("reduced shape:", Z.shape)
print("first row in 2-D:", Z[0].round(3).tolist())`, "eigenvalues: [4.228, 0.243, 0.078, 0.024]\nexplained variance ratio: [0.925, 0.053, 0.017, 0.005]\nvariance from SVD: [4.228, 0.243, 0.078, 0.024]\nsame directions? True\nreduced shape: (150, 2)\nfirst row in 2-D: [-2.684, -0.319]"],
    ["p", "The first component alone carries most of the variance of the iris data, so a 4-feature dataset is well described by 2 numbers."],
    ["h", "The scikit-learn version, and how many components to keep"],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)        # 1797 images of 8x8 = 64 pixel values
X = StandardScaler().fit_transform(X)

pca = PCA().fit(X)                         # keep all components to see the whole curve
cum = np.cumsum(pca.explained_variance_ratio_)
for target in [0.80, 0.90, 0.95]:
    print(f"{int(target * 100)}% of the variance needs {int(np.searchsorted(cum, target)) + 1} of 64 components")

plt.figure(figsize=(6, 3.2))
plt.plot(range(1, 65), cum, marker=".")
plt.axhline(0.95, linestyle="--")
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.title("Digits: how many components to keep")
plt.show()

pca2 = PCA(n_components=0.95).fit(X)       # a float means: keep enough for this share of variance
print("PCA(n_components=0.95) keeps:", pca2.n_components_)
print("compressed shape:", pca2.transform(X).shape)`, "80% of the variance needs 21 of 64 components\n90% of the variance needs 31 of 64 components\n95% of the variance needs 40 of 64 components\nPCA(n_components=0.95) keeps: 40\ncompressed shape: (1797, 40)"],
    ["table", ["How to choose the number of components", "When"], [
      ["Keep enough for 90 to 95% of the variance", "Compression, speeding up a model"],
      ["Look for an elbow in the scree plot (variance per component)", "Quick exploration"],
      ["Treat the number as a hyper-parameter and cross-validate the full pipeline", "When PCA feeds a supervised model"],
      ["2 or 3", "Visualisation only"],
    ]],
    ["h", "Things to know"],
    ["list", [
      "PCA is **linear** and **unsupervised**. It maximises variance, which is not always the same as predictive power: a low-variance direction can be the one that separates your classes.",
      "Components are mixtures of all features, so they are hard to interpret. The `components_` weights show which original features contribute most.",
      "To plot high-dimensional data as clusters, **t-SNE** and **UMAP** are non-linear alternatives. They are for looking at the data, not for creating model features, and distances in their pictures are not reliable.",
      "PCA can also denoise: project down, then `inverse_transform` back up. The small components, mostly noise, are dropped.",
    ]],
    ["danger", "Running PCA on unscaled features. PCA chases variance, so the feature with the biggest units wins the first component by default. Standardize first (unless all features share one unit, like pixels). And **fit the PCA on the training data only**, then transform the test data, exactly as for a scaler (see [data leakage](ml.html#splits))."],
    ["code", R`
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
for m in [5, 15, 30, 64]:
    pipe = make_pipeline(StandardScaler(), PCA(n_components=m), LogisticRegression(max_iter=2000))
    print(f"{m:2d} components -> CV accuracy {cross_val_score(pipe, X, y, cv=5).mean():.3f}")`, " 5 components -> CV accuracy 0.771\n15 components -> CV accuracy 0.892\n30 components -> CV accuracy 0.907\n64 components -> CV accuracy 0.920"],
    ["p", "Putting the scaler and PCA **inside the pipeline** means each fold fits them on its own training part only, which is the safe way. More on that in [cross-validation](ml.html#cross-validation)."],
    ["tip", "Practice: [PCA with SVD and Explained Variance](practice.html#p/pca-svd)."],
  ],
},
{
  id: "classification-metrics", group: "Evaluation", title: "Confusion Matrix, Precision, Recall & F1",
  sub: "Accuracy hides what kind of mistakes a classifier makes. The confusion matrix and the numbers built from it show exactly that.",
  tags: ["metrics"],
  blocks: [
    ["p", "Suppose 1 in 100 transactions is fraud. A model that says \"not fraud\" every single time is 99% accurate and completely useless. **Accuracy** (the share of correct predictions) is fine when the classes are balanced and every mistake costs the same. Usually neither is true, so we look at *which* mistakes the model makes."],
    ["h", "The confusion matrix"],
    ["p", "Pick one class to call **positive** (the one you care about finding: fraud, disease, spam). Each prediction then falls into one of four boxes:"],
    ["diagram", "ml2_cm", [], "Rows are the truth, columns are the prediction. The two dashed outlines show which cells precision and recall use."],
    ["table", ["Cell", "Meaning", "Example (spam filter)"], [
      ["TP true positive", "positive, predicted positive", "spam sent to the spam folder"],
      ["FP false positive", "negative, predicted positive (**false alarm**, Type I)", "a good email lost in spam"],
      ["FN false negative", "positive, predicted negative (**miss**, Type II)", "spam in the inbox"],
      ["TN true negative", "negative, predicted negative", "a good email delivered"],
    ]],
    ["h", "The metrics"],
    ["math", R`\text{precision}=\frac{TP}{TP+FP} \qquad \text{recall}=\frac{TP}{TP+FN} \qquad F_1 = \frac{2PR}{P+R}`, "precision P · recall R (also called sensitivity or true positive rate)"],
    ["list", [
      "**Precision**: of everything I flagged, how much was really positive? Low precision = many false alarms.",
      "**Recall**: of everything really positive, how much did I find? Low recall = many misses.",
      "**F1**: the harmonic mean of the two. It is high only when *both* are high (a model with precision 1.0 and recall 0.01 gets F1 of about 0.02, not 0.5).",
      "**Specificity** $= TN/(TN+FP)$: recall for the negative class. **Accuracy** $=(TP+TN)/\\text{all}$.",
    ]],
    ["p", "Which to favour depends on the cost of each mistake. Screening for a serious disease: a miss is terrible, a false alarm just means another test, so push **recall** up. A spam filter or a system that automatically blocks accounts: a false alarm hurts a real user, so push **precision** up."],
    ["h", "From scratch in NumPy"],
    ["code", R`
import numpy as np

y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y_pred = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

tp = int(np.sum((y_pred == 1) & (y_true == 1)))
fp = int(np.sum((y_pred == 1) & (y_true == 0)))
fn = int(np.sum((y_pred == 0) & (y_true == 1)))
tn = int(np.sum((y_pred == 0) & (y_true == 0)))
print("TP, FP, FN, TN =", tp, fp, fn, tn)

accuracy = (tp + tn) / len(y_true)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print("accuracy :", round(accuracy, 3))
print("precision:", round(precision, 3))
print("recall   :", round(recall, 3))
print("F1       :", round(f1, 3))

always_no = np.zeros_like(y_true)
print("accuracy of 'always predict 0':", round(float(np.mean(always_no == y_true)), 3))`, "TP, FP, FN, TN = 3 2 2 8\naccuracy : 0.733\nprecision: 0.6\nrecall   : 0.6\nF1       : 0.6\naccuracy of 'always predict 0': 0.667"],
    ["p", "Accuracy is 0.73, and even \"always predict 0\" scores 0.67, so it says little. The model found only 3 of 5 positives, and 2 of its 5 alerts were false alarms. Now the scikit-learn tools:"],
    ["code", R`
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y_pred = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

print(confusion_matrix(y_true, y_pred))      # rows = actual, columns = predicted: [[TN, FP], [FN, TP]]
print("precision:", round(precision_score(y_true, y_pred), 3), " recall:", round(recall_score(y_true, y_pred), 3),
      " F1:", round(f1_score(y_true, y_pred), 3))
print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))`, "[[8 2]\n [2 3]]\nprecision: 0.6  recall: 0.6  F1: 0.6\n              precision    recall  f1-score   support\n\n    negative       0.80      0.80      0.80        10\n    positive       0.60      0.60      0.60         5\n\n    accuracy                           0.73        15\n   macro avg       0.70      0.70      0.70        15\nweighted avg       0.73      0.73      0.73        15"],
    ["warn", "scikit-learn's `confusion_matrix` puts the **negative class first**: `[[TN, FP], [FN, TP]]`. Many textbooks draw TP in the top-left corner. Always check the layout before reading off numbers."],
    ["h", "Move the threshold"],
    ["p", "Most classifiers output a **score** or probability, and you turn it into a yes / no with a **threshold**. 0.5 is only a default. Slide the threshold: a lower one catches more positives (recall up) but raises more false alarms (precision down). The best F1 is usually not at 0.5."],
    ["viz", "ml2_threshold"],
    ["diagram", "ml2_prtradeoff", [], "Precision and recall against the threshold on 300 toy scores. They pull in opposite directions; the dot marks the threshold with the best F1."],
    ["h", "More than two classes"],
    ["p", "With many classes, compute precision and recall for each class (treating it as positive and the rest as negative) and then average:"],
    ["table", ["Average", "How", "Use when"], [
      ["macro", "Plain mean of the per-class scores", "All classes matter equally, including rare ones"],
      ["micro", "Pool all TP, FP, FN, then compute once (for single-label problems it equals accuracy)", "You care about overall performance on frequent classes"],
      ["weighted", "Mean weighted by how many examples each class has", "Report an overall figure that follows class frequencies"],
    ]],
    ["code", R`
import numpy as np
from sklearn.metrics import f1_score

y_true = np.array([0] * 8 + [1] * 8 + [2] * 2)
y_pred = np.array([0] * 8 + [1] * 7 + [0] + [0, 0])        # the rare class 2 is never found

for avg in ["macro", "micro", "weighted"]:
    print(f"{avg:9s} F1 = {f1_score(y_true, y_pred, average=avg, zero_division=0):.3f}")
print("per class:", f1_score(y_true, y_pred, average=None, zero_division=0).round(3).tolist())`, "macro     F1 = 0.592\nmicro     F1 = 0.833\nweighted  F1 = 0.789\nper class: [0.842, 0.933, 0.0]"],
    ["p", "Macro averaging exposes the failure on the rare class; micro averaging (= accuracy here) hides it."],
    ["danger", "Reporting accuracy on imbalanced data, or picking the threshold on the test set. Choose the threshold on validation data (or by cross-validation), then evaluate once on the test set. See [Imbalanced Data](ml.html#imbalanced)."],
    ["tip", "Practice: [Confusion Matrix, Precision, Recall & F1](practice.html#p/confusion-metrics). Next: [ROC-AUC and PR curves](ml.html#roc-auc), which judge a model across *all* thresholds."],
  ],
},
{
  id: "roc-auc", group: "Evaluation", title: "ROC-AUC & PR Curves",
  sub: "Judge a scoring model across every possible threshold at once, instead of at one cut-off.",
  tags: ["metrics"],
  blocks: [
    ["p", "[Precision and recall](ml.html#classification-metrics) depend on the threshold you pick. Often you want to compare *models* before choosing a threshold. Both curves in this section sweep the threshold from high to low and plot what happens."],
    ["h", "The ROC curve"],
    ["p", "The **ROC curve** (receiver operating characteristic) plots, for every threshold:"],
    ["list", [
      "**True positive rate** $TPR = TP/(TP+FN)$ = recall, on the vertical axis.",
      "**False positive rate** $FPR = FP/(FP+TN)$ = $1-$ specificity, on the horizontal axis.",
    ]],
    ["p", "A threshold above every score predicts nothing positive: the point (0, 0). A threshold below every score predicts everything positive: (1, 1). In between, the curve traces the trade-off. A perfect model reaches the top-left corner (TPR 1 at FPR 0). Random guessing follows the diagonal."],
    ["p", "To build it by hand: sort the examples by score, highest first. Walk down the list: a **positive** moves the curve one step **up**, a **negative** one step **right**."],
    ["diagram", "ml2_rocsteps", [[0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2], [1, 1, 0, 1, 0, 1, 0, 0]], "Eight examples sorted by score (amber = positive, blue = negative). Each positive steps up, each negative steps right; the shaded area is the AUC."],
    ["h", "AUC: one number for the whole curve"],
    ["p", "The **area under the ROC curve (AUC)** is 1 for a perfect ranking and 0.5 for random scores. It has a neat meaning: **the probability that a randomly chosen positive gets a higher score than a randomly chosen negative.** Counting those pairs gives the AUC directly:"],
    ["math", R`\text{AUC} = \frac{1}{n_+ n_-}\sum_{i:\,y_i=1}\ \sum_{j:\,y_j=0} \Big[\mathbb 1(s_i > s_j) + \tfrac12\,\mathbb 1(s_i = s_j)\Big]`],
    ["code", R`
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

scores = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2])
labels = np.array([1, 1, 0, 1, 0, 1, 0, 0])

# way 1: count the (positive, negative) pairs that are ranked correctly
pos, neg = scores[labels == 1], scores[labels == 0]
wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
print("pairs ranked correctly:", int(wins), "of", len(pos) * len(neg))
print("AUC by pair counting:", wins / (len(pos) * len(neg)))

# way 2: walk down the sorted scores and add up trapezoids
order = np.argsort(-scores)
tpr = np.concatenate([[0], np.cumsum(labels[order] == 1) / labels.sum()])
fpr = np.concatenate([[0], np.cumsum(labels[order] == 0) / (1 - labels).sum()])
print("AUC by trapezoids   :", round(float(np.trapezoid(tpr, fpr)), 4))

print("sklearn roc_auc_score:", roc_auc_score(labels, scores))
f, t, thr = roc_curve(labels, scores)
print("first points (FPR, TPR):", list(zip(f.round(2).tolist(), t.round(2).tolist()))[:5])`, "pairs ranked correctly: 13 of 16\nAUC by pair counting: 0.8125\nAUC by trapezoids   : 0.8125\nsklearn roc_auc_score: 0.8125\nfirst points (FPR, TPR): [(0.0, 0.0), (0.0, 0.25), (0.0, 0.5), (0.25, 0.5), (0.25, 0.75)]"],
    ["p", "Play with the picture below. More **class separation** pushes the ROC curve toward the corner. Then lower the share of **positives**: the ROC curve barely moves, but the precision-recall curve collapses. That leads to the next point."],
    ["viz", "ml2_roc"],
    ["h", "Precision-recall (PR) curves"],
    ["p", "The **PR curve** plots precision against recall over all thresholds. Its summary is **average precision (AP)**, roughly the area under it. The baseline is not 0.5 but the share of positives (a random model has precision equal to the positive rate)."],
    ["table", ["", "ROC-AUC", "PR-AUC / average precision"], [
      ["Axes", "TPR vs FPR", "precision vs recall"],
      ["Random model scores", "0.5", "the positive rate (e.g. 0.01 if 1% are positive)"],
      ["Affected by class imbalance?", "Barely. FPR divides by the huge negative count, so many false alarms still look small.", "Strongly. Precision punishes every false alarm."],
      ["Best for", "Balanced data, or comparing ranking quality in general", "Rare positives (fraud, disease, retrieval) where the positive class is what you care about"],
    ]],
    ["code", R`
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=5000, n_features=10, n_informative=4, weights=[0.97, 0.03],
                           class_sep=1.5, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)

model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
p = model.predict_proba(X_te)[:, 1]
print("positive rate in test set:", round(float(y_te.mean()), 3))
print("ROC-AUC          :", round(roc_auc_score(y_te, p), 3))
print("average precision:", round(average_precision_score(y_te, p), 3))`, "positive rate in test set: 0.034\nROC-AUC          : 0.753\naverage precision: 0.395"],
    ["p", "Judge average precision against its baseline, the positive rate (about 0.03 here): an AP near 0.4 is many times better than chance, yet far from the clean picture a perfect model would give. With 3% positives the model still makes many false alarms relative to its hits, and the PR view shows that cost directly."],
    ["tip", "AUC is about **ranking**, so it ignores whether the probabilities are well calibrated and does not choose a threshold for you. After comparing models with AUC or AP, still pick the operating threshold from the costs of FP and FN."],
    ["warn", "AUC is a summary over *all* thresholds, including ones you would never use (such as FPR above 0.5). Two models with the same AUC can behave very differently in the region you care about. Look at the curve."],
    ["danger", "Comparing a ROC-AUC of 0.95 on an imbalanced problem and concluding the model is \"excellent\". Check precision at your target recall (or AP) as well."],
    ["tip", "Practice: [ROC-AUC from Scores](practice.html#p/roc-auc-scores)."],
  ],
},
{
  id: "regression-metrics", group: "Evaluation", title: "Regression Metrics",
  sub: "How far off are the predicted numbers? MAE, MSE, RMSE, R² and when to use which.",
  tags: ["metrics"],
  blocks: [
    ["p", "For regression the question is \"how big are the errors?\". Let $e_i = y_i - \\hat y_i$ be the error (residual) of example $i$. All the common metrics are different ways to summarise the list of errors. Each one punishes big errors differently, and that choice is the whole story."],
    ["math", R`\text{MAE}=\frac1n\sum|e_i| \qquad \text{MSE}=\frac1n\sum e_i^2 \qquad \text{RMSE}=\sqrt{\text{MSE}} \qquad R^2 = 1-\frac{\sum e_i^2}{\sum (y_i-\bar y)^2}`],
    ["table", ["Metric", "Units", "Reads as", "Reaction to outliers"], [
      ["MAE (mean absolute error)", "same as y", "typical error size; easy to explain", "Linear: each unit of error counts once"],
      ["MSE (mean squared error)", "y squared", "what many models minimise; smooth to optimise", "Strong: a 10 off counts 100"],
      ["RMSE", "same as y", "like MAE but leaning toward the big errors", "Strong"],
      ["R² (coefficient of determination)", "none", "share of the variance in y the model explains", "Strong (it is built on squared errors)"],
      ["MAPE (mean absolute percentage error)", "percent", "error relative to size", "Breaks when y is 0 or tiny; favours under-prediction"],
    ]],
    ["h", "MAE vs RMSE"],
    ["p", "RMSE is always at least as large as MAE. The gap tells you how uneven the errors are: equal errors give MAE = RMSE, one huge error pulls RMSE far above MAE. Make one prediction go badly wrong and watch the two metrics:"],
    ["viz", "ml2_regmetrics"],
    ["p", "Use **MAE** when outliers are noise you do not want to dominate the score, or when you need a plain \"on average we are off by X\". Use **RMSE** when large errors are disproportionately bad (a forecast that is off by 10 is worse than two off by 5)."],
    ["h", "R², the baseline comparison"],
    ["p", "$R^2$ compares your model with the laziest model possible: always predict the mean $\\bar y$. The numerator is your squared error, the denominator the baseline's. $R^2 = 1$ is perfect, $R^2 = 0$ is no better than the mean, and **negative $R^2$ is possible**: worse than the mean, usually on test data. It does not say whether the errors are small in useful units, only the *share* of the variation explained."],
    ["code", R`
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = np.array([3.0, 5.0, 7.5, 9.0, 11.0, 14.0])
y_pred = np.array([2.5, 5.5, 7.0, 10.0, 10.0, 13.0])

e = y_true - y_pred
mae = np.mean(np.abs(e))
mse = np.mean(e ** 2)
rmse = np.sqrt(mse)
r2 = 1 - np.sum(e ** 2) / np.sum((y_true - y_true.mean()) ** 2)
print("by hand : MAE", round(float(mae), 3), " MSE", round(float(mse), 3), " RMSE", round(float(rmse), 3), " R2", round(float(r2), 3))

print("sklearn : MAE", round(mean_absolute_error(y_true, y_pred), 3), " MSE", round(mean_squared_error(y_true, y_pred), 3),
      " RMSE", round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3), " R2", round(r2_score(y_true, y_pred), 3))

mean_model = np.full_like(y_true, y_true.mean())
print("R2 of always predicting the mean:", round(r2_score(y_true, mean_model), 3))
bad = np.array([14.0, 9.0, 3.0, 1.0, 2.0, 0.0])
print("R2 of a terrible model:", round(r2_score(y_true, bad), 3))`, "by hand : MAE 0.75  MSE 0.625  RMSE 0.791  R2 0.953\nsklearn : MAE 0.75  MSE 0.625  RMSE 0.791  R2 0.953\nR2 of always predicting the mean: 0.0\nR2 of a terrible model: -5.238"],
    ["h", "Look at the residuals"],
    ["p", "A single number cannot show *how* a model is wrong. Plot residuals ($y - \\hat y$) against the predictions. Healthy residuals scatter randomly around zero with a similar spread everywhere; a curve means the model is missing a pattern (try [polynomial features](ml.html#polynomial)); a funnel shape means the error grows with the size of y (consider predicting log y)."],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 120)
y = 0.5 * x ** 2 + rng.normal(0, 2, 120)                 # the truth is a curve
model = LinearRegression().fit(x[:, None], y)            # ...but we fit a straight line
pred = model.predict(x[:, None])
resid = y - pred

plt.figure(figsize=(6, 3.2))
plt.scatter(pred, resid, s=12)
plt.axhline(0, color="k", linewidth=1)
plt.xlabel("predicted value")
plt.ylabel("residual")
plt.title("A U-shaped residual plot: the model misses the curve")
plt.show()
print("R2:", round(model.score(x[:, None], y), 3), " mean residual:", round(float(resid.mean()) + 0.0, 3))`, "R2: 0.92  mean residual: -0.0"],
    ["p", "The R² looks fine, and the mean residual is zero by construction, yet the plot reveals the straight line is wrong everywhere. That is why you plot."],
    ["table", ["Situation", "Use"], [
      ["Report to a non-technical audience", "MAE (\"off by 3.2 units on average\")"],
      ["Large errors are very costly", "RMSE"],
      ["Compare models on the same dataset", "any, but R² is handy as a unit-free score"],
      ["Targets span orders of magnitude (prices, counts)", "Metrics on log(y) (RMSLE), or MAPE with care"],
      ["Model trained with squared loss", "Also report RMSE: it is what the model actually minimised"],
    ]],
    ["danger", "Judging a model by R² on the **training** data (it only goes up as you add features), or comparing R² across datasets with different variance. Evaluate on held-out data, and always compare with the mean-prediction baseline."],
    ["tip", "These metrics are the evaluation side of the [loss functions](ml.html#loss-functions) you train with. A model trained on MSE is not necessarily best by MAE."],
  ],
},
{
  id: "cross-validation", group: "Evaluation", title: "Cross-Validation & Hyperparameter Tuning",
  sub: "Use all your data to estimate how well a model will do on new data, then use that estimate to choose settings honestly.",
  tags: ["validation"],
  blocks: [
    ["p", "A single [train / validation split](ml.html#splits) is noisy: a lucky or unlucky split changes your score, and holding out data wastes examples you could have trained on. **k-fold cross-validation** fixes both by letting every example be used for validation exactly once."],
    ["h", "k-fold cross-validation"],
    ["steps", [
      "Shuffle the data and cut it into **k folds** of about equal size (5 or 10 is typical).",
      "For each fold in turn: train a **fresh** model on the other k-1 folds, score it on the held-out fold.",
      "Report the **mean** of the k scores, and their **standard deviation** as a feeling for how stable the result is.",
    ]],
    ["viz", "ml2_kfold"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)

# from scratch: one fold held out at a time
rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
folds = np.array_split(idx, 5)
scores = []
for i in range(5):
    val = folds[i]
    train = np.concatenate([folds[j] for j in range(5) if j != i])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[train], y[train])
    scores.append(model.score(X[val], y[val]))
print("fold scores:", np.round(scores, 3).tolist())
print("mean:", round(float(np.mean(scores)), 3), " std:", round(float(np.std(scores)), 3))

# the library way
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
cv = KFold(n_splits=5, shuffle=True, random_state=0)
s = cross_val_score(model, X, y, cv=cv)
print("cross_val_score:", s.round(3).tolist(), " mean", round(float(s.mean()), 3))`, "fold scores: [0.991, 0.974, 0.982, 0.974, 0.973]\nmean: 0.979  std: 0.007\ncross_val_score: [0.965, 0.982, 0.956, 0.956, 1.0]  mean 0.972"],
    ["h", "Choosing the splitter"],
    ["diagram", "ml2_cvschemes", [], "Plain k-fold, stratified k-fold, group k-fold and time-series split. Blue trains, amber validates."],
    ["table", ["Splitter", "Use it when", "Why"], [
      ["`KFold`", "Regression, balanced classes", "Simple default. Use `shuffle=True` unless the order is meaningful."],
      ["`StratifiedKFold`", "Classification, especially imbalanced", "Every fold keeps the class proportions, so no fold ends up with almost no positives. This is the default for classifiers in `cross_val_score`."],
      ["`GroupKFold`", "Several rows come from one person / patient / user / device", "Keeps each group entirely on one side, so the model is tested on groups it has never seen."],
      ["`TimeSeriesSplit`", "Data ordered in time", "Always train on the past and validate on the future. Shuffling would leak the future."],
      ["`LeaveOneOut`", "Tiny datasets", "n folds of size 1: nearly unbiased but slow and high-variance."],
    ]],
    ["code", R`
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

y = np.array([0] * 12 + [1] * 4)                  # 25% positives
X = np.zeros((16, 1))

for name, cv in [("KFold", KFold(4)), ("StratifiedKFold", StratifiedKFold(4))]:
    print(name)
    for tr, va in cv.split(X, y):
        print("  validation indices", va.tolist(), " positives:", int(y[va].sum()), "of", len(va))`, "KFold\n  validation indices [0, 1, 2, 3]  positives: 0 of 4\n  validation indices [4, 5, 6, 7]  positives: 0 of 4\n  validation indices [8, 9, 10, 11]  positives: 0 of 4\n  validation indices [12, 13, 14, 15]  positives: 4 of 4\nStratifiedKFold\n  validation indices [0, 1, 2, 12]  positives: 1 of 4\n  validation indices [3, 4, 5, 13]  positives: 1 of 4\n  validation indices [6, 7, 8, 14]  positives: 1 of 4\n  validation indices [9, 10, 11, 15]  positives: 1 of 4"],
    ["p", "Plain `KFold` without shuffling put all four positives in a single fold. Stratification gives each fold one positive."],
    ["h", "Leakage inside cross-validation"],
    ["p", "Cross-validation is only honest if **every** data-dependent step is learned from the training part of each fold: scaling, imputing, feature selection, PCA, target encoding, oversampling. Do those steps on the full dataset first and information from the validation fold leaks into training."],
    ["diagram", "ml2_leakage", [], "Left: the scaler sees all the data before the split (leak). Right: it is fitted inside each fold on the training part only."],
    ["p", "Here is how bad it can get. The labels below are pure coin flips, and so are the 2000 features. A model cannot do better than 50%. But if we pick the 20 \"best\" features using **all** the labels before cross-validating, the selection has already peeked at the answers:"],
    ["code", R`
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2000))           # pure noise
y = rng.integers(0, 2, size=100)           # random labels: the true accuracy is about 50%

# WRONG: choose features with all labels, then cross-validate
X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
wrong = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=5).mean()

# RIGHT: selection happens inside each fold, in a pipeline
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
right = cross_val_score(pipe, X, y, cv=5).mean()

print("selection outside the folds:", round(float(wrong), 3))
print("selection inside a pipeline:", round(float(right), 3))`, "selection outside the folds: 0.78\nselection inside a pipeline: 0.59"],
    ["h", "Hyperparameter tuning"],
    ["p", "**Hyperparameters** are settings you choose before training (tree depth, regularization strength `C`, number of neighbours). Tune them by trying candidates and scoring each by cross-validation. **Grid search** tries every combination; **random search** samples combinations, which usually finds good values with far fewer fits when only a few hyperparameters really matter."],
    ["code", R`
import numpy as np
from scipy.stats import loguniform
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

pipe = Pipeline([("scale", StandardScaler()), ("svm", SVC())])
grid = GridSearchCV(pipe, {"svm__C": [0.1, 1, 10], "svm__gamma": [0.001, 0.01, 0.1]}, cv=5)
grid.fit(X_tr, y_tr)                       # tuning sees ONLY the training data
print("best params:", grid.best_params_)
print("best CV score:", round(grid.best_score_, 3))
print("test score (touched once):", round(grid.score(X_te, y_te), 3))

rand = RandomizedSearchCV(pipe, {"svm__C": loguniform(1e-2, 1e2), "svm__gamma": loguniform(1e-4, 1e0)},
                          n_iter=12, cv=5, random_state=0).fit(X_tr, y_tr)
print("random search best CV score:", round(rand.best_score_, 3), "with", {k: round(float(v), 4) for k, v in rand.best_params_.items()})`, "best params: {'svm__C': 10, 'svm__gamma': 0.01}\nbest CV score: 0.981\ntest score (touched once): 0.972\nrandom search best CV score: 0.986 with {'svm__C': 2.5766, 'svm__gamma': 0.0151}"],
    ["table", ["Method", "Cost", "Notes"], [
      ["Grid search", "all combinations × k folds", "Fine for 1 or 2 hyperparameters. Cost explodes with more."],
      ["Random search", "you choose the budget", "Better coverage of each hyperparameter. Use a log scale for things like C, gamma, learning rate."],
      ["Bayesian / successive halving", "fewer fits", "Smarter search (Optuna, `HalvingGridSearchCV`). Same rules about leakage apply."],
    ]],
    ["note", "The recipe: split off a **test set** first and lock it away. Do all scaling, selection and tuning with cross-validation on the training part (inside a `Pipeline`). Refit the best settings on all the training data. Evaluate on the test set **once**."],
    ["qa", "Going deeper: nested cross-validation", "The best CV score from a search is slightly optimistic, because you picked the best of many candidates *using that same score*. For an honest estimate of the whole procedure, wrap the search in an outer loop: an inner loop (GridSearchCV) tunes on each outer training part, and the outer folds score the winner. In code that is `cross_val_score(GridSearchCV(pipe, params, cv=3), X, y, cv=5)`. It costs more, so it is mostly used when comparing methods on small datasets."],
    ["danger", "Tuning on the test set (by trying settings, looking at the test score, and trying again), or reporting `best_score_` as if it were the test score. Once you have looked at the test set several times, it is just another validation set and your estimate is optimistic."],
    ["tip", "Practice: [Stratified k-Fold Indices](practice.html#p/stratified-kfold). Related: [pipelines in scikit-learn](pyml.html#sklearn-pipelines), [data leakage](ml.html#splits)."],
  ],
},
{
  id: "feature-engineering", group: "Practical", title: "Feature Engineering & Scaling",
  sub: "Put the raw columns into a shape a model can use: put them on comparable scales, fill the gaps, and add features that expose the pattern.",
  tags: ["preprocessing"],
  blocks: [
    ["p", "Models do not see \"age\" or \"income\"; they see numbers. How you present those numbers often matters more than which model you pick. **Feature engineering** is the work of turning raw data into inputs that make the pattern easy to learn. A big part of it is **scaling**, so that no feature wins just because of its units."],
    ["h", "Why scaling matters"],
    ["p", "Age is in tens, income is in tens of thousands. Any method that uses **distances** ([k-means](ml.html#kmeans), [kNN](ml.html#knn), [SVM](ml.html#svm)) or **gradient descent** with a shared step size ([logistic regression](ml.html#logistic-regression), neural networks) or **penalties** ([ridge / lasso](ml.html#regularization)) is dominated by the big numbers unless you rescale. Trees do not care, because a split like `age < 30` is the same whatever the unit."],
    ["viz", "ml2_scaling"],
    ["table", ["Scaler", "Formula", "Result", "Use when"], [
      ["`StandardScaler`", "(x − mean) / std", "mean 0, std 1; no fixed range", "The default for linear models, SVM, kNN, PCA, neural nets"],
      ["`MinMaxScaler`", "(x − min) / (max − min)", "inside [0, 1]", "You need a bounded range (images, some neural nets). Squashed by outliers."],
      ["`RobustScaler`", "(x − median) / IQR", "median 0; outliers stay far out", "Data with strong outliers"],
      ["`MaxAbsScaler`", "x / max|x|", "inside [−1, 1]; keeps zeros", "Sparse data (does not destroy sparsity)"],
      ["`Normalizer`", "x / ‖x‖ per row", "each row has length 1", "Row-wise: text vectors, cosine similarity"],
      ["`PowerTransformer`, `np.log1p`", "Yeo-Johnson / Box-Cox / log", "less skew", "Right-skewed values: prices, counts, durations"],
    ]],
    ["h", "From scratch, and the scikit-learn way"],
    ["code", R`
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

X = np.array([[25, 40000], [32, 52000], [47, 98000], [51, 61000], [62, 150000]], dtype=float)

# standardize by hand: subtract each column's mean, divide by its standard deviation
mu, sd = X.mean(axis=0), X.std(axis=0)
Z = (X - mu) / sd
print("by hand:\n", Z.round(3))
print("matches StandardScaler:", bool(np.allclose(Z, StandardScaler().fit_transform(X))))

mm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
print("matches MinMaxScaler:", bool(np.allclose(mm, MinMaxScaler().fit_transform(X))))
print("column means/stds after scaling:", (Z.mean(axis=0).round(3) + 0.0).tolist(), Z.std(axis=0).round(3).tolist())`, "by hand:\n [[-1.383 -1.007]\n [-0.857 -0.706]\n [ 0.271  0.446]\n [ 0.571 -0.481]\n [ 1.398  1.748]]\nmatches StandardScaler: True\nmatches MinMaxScaler: True\ncolumn means/stds after scaling: [0.0, 0.0] [1.0, 1.0]"],
    ["h", "Fit on the training data only"],
    ["p", "The scaler **learns** numbers (mean and std) from data. Those must come from the training set only. If you compute them on all the data, information about the test rows leaks into training. The pattern is always: **fit on train, transform both**."],
    ["code", R`
import numpy as np
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
train = rng.normal(loc=50, scale=10, size=(200, 1))
test = rng.normal(loc=50, scale=10, size=(50, 1))

scaler = StandardScaler().fit(train)         # learns mean and std from TRAIN only
train_s, test_s = scaler.transform(train), scaler.transform(test)
print("learned mean/std:", scaler.mean_.round(2).tolist(), scaler.scale_.round(2).tolist())
print("train mean/std after:", round(float(train_s.mean()) + 0.0, 3), round(float(train_s.std()), 3))
print("test  mean/std after:", round(float(test_s.mean()), 3), round(float(test_s.std()), 3))   # near, but not exactly, 0 and 1`, "learned mean/std: [50.15] [9.61]\ntrain mean/std after: -0.0 1.0\ntest  mean/std after: -0.11 1.261"],
    ["danger", "Calling `scaler.fit_transform(X)` on the whole dataset **before** splitting or cross-validating. The test rows influenced the mean and std. The fix is a `Pipeline`, which refits the scaler inside every fold automatically (see [data leakage](ml.html#splits) and [cross-validation](ml.html#cross-validation))."],
    ["h", "Handling real tables: missing values, numbers and categories together"],
    ["p", "A `ColumnTransformer` applies different steps to different columns, and a `Pipeline` chains them with the model, so the whole recipe is fitted only on training data. Categorical columns are covered in [Encoding Categorical Data](ml.html#encoding)."],
    ["code", R`
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rng = np.random.default_rng(0)
n = 300
df = pd.DataFrame({
    "age": rng.integers(18, 70, n).astype(float),
    "income": rng.lognormal(10.5, 0.6, n),
    "city": rng.choice(["Paris", "Rome", "Oslo"], n),
})
y = ((df["income"] > 40000) & (df["age"] > 30)).astype(int)
df.loc[rng.choice(n, 30, replace=False), "age"] = np.nan          # 10% missing ages

prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["age", "income"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])
model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
print("CV accuracy:", cross_val_score(model, df, y, cv=5).mean().round(3))
print("missing ages before:", int(df["age"].isna().sum()))
print("feature matrix width:", prep.fit_transform(df).shape[1], "(2 numeric + 3 one-hot city columns)")`, "CV accuracy: 0.887\nmissing ages before: 30\nfeature matrix width: 5 (2 numeric + 3 one-hot city columns)"],
    ["h", "Creating better features"],
    ["p", "Scaling changes the units; **engineering** adds information the model cannot easily build itself. Linear models in particular can only add features up, so give them the combinations."],
    ["table", ["Technique", "Example", "Why it helps"], [
      ["Ratios and differences", "price / area, debt / income, days since last purchase", "The ratio is the real signal; a linear model cannot divide"],
      ["Interactions and polynomials", "x₁·x₂, x² (see [polynomial features](ml.html#polynomial))", "Lets linear models bend"],
      ["Log / power transform", "`np.log1p(income)`", "Tames skew and outliers; turns multiplicative effects into additive ones"],
      ["Binning", "age → child / adult / senior", "Captures steps and non-linear effects, and tames outliers"],
      ["Date and time parts", "hour, day of week, month, is_weekend; sin/cos of hour", "Models cannot read a timestamp. Cyclical features avoid the 23 → 0 jump."],
      ["Text and counts", "length, word counts, TF-IDF", "Turns text into numbers"],
      ["Aggregates", "mean spend per customer, count of orders in last 30 days", "Summarises history, usually the most powerful features in practice"],
      ["Missing-value flags", "`age_missing` = 1 if age was NaN", "Missingness itself can be informative"],
    ]],
    ["code", R`
import numpy as np
import pandas as pd

df = pd.DataFrame({"hour": [0, 6, 12, 18, 23], "income": [30e3, 45e3, 52e3, 300e3, 1.2e6]})

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)         # 23:00 and 00:00 end up next to each other
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["log_income"] = np.log1p(df["income"])                     # squashes the 1.2 million outlier
df["income_band"] = pd.cut(df["income"], bins=[0, 40e3, 100e3, np.inf], labels=["low", "mid", "high"])
print(df.round(2).to_string(index=False))`, " hour    income  hour_sin  hour_cos  log_income income_band\n    0   30000.0      0.00      1.00       10.31         low\n    6   45000.0      1.00      0.00       10.71         mid\n   12   52000.0      0.00     -1.00       10.86         mid\n   18  300000.0     -1.00     -0.00       12.61        high\n   23 1200000.0     -0.26      0.97       14.00        high"],
    ["warn", "Keep the **target** out of your features, and be careful with anything that would not be known at prediction time (for example \"refund issued\" in a model that predicts refunds). Features that quietly encode the answer give great scores and useless models: that is [target leakage](ml.html#splits)."],
    ["danger", "Scaling before k-means or PCA is not optional: without it, the biggest-unit feature decides everything. And scaling *dummy* (0/1) columns is usually unnecessary but harmless."],
    ["tip", "Which models need scaling? **Yes**: kNN, SVM, k-means, PCA, linear / logistic regression with regularization, neural networks. **No**: decision trees, random forests, gradient boosting (see [trees](ml.html#decision-trees)), Naive Bayes."],
  ],
},
{
  id: "encoding", group: "Practical", title: "Encoding Categorical Data",
  sub: "Models need numbers. Turn categories like \"red\" or \"Paris\" into numbers without inventing an order that is not there.",
  tags: ["preprocessing"],
  blocks: [
    ["p", "A column such as `colour = red / green / blue` or `city` is **categorical**. Most models only accept numbers, so we must encode it. The way you do it matters, because the numbers carry meaning to the model whether you meant them to or not."],
    ["h", "Nominal vs ordinal"],
    ["list", [
      "**Nominal** categories have no order: colour, city, country. Encoding Paris = 1, Rome = 2, Oslo = 3 would tell a linear model that Rome is \"between\" the other two. That is nonsense.",
      "**Ordinal** categories have a real order: small < medium < large, or low < mid < high. Integers are fine here, if the order is given explicitly.",
    ]],
    ["h", "The main encoders"],
    ["table", ["Encoder", "What it does", "Good for", "Watch out"], [
      ["One-hot", "One 0/1 column per category", "Nominal data with few categories, any model", "Many categories → very wide data"],
      ["Ordinal", "Category → 0, 1, 2… in the order you give", "True ordinal data; tree models even for nominal data", "Do not use on nominal data with linear / distance models"],
      ["Frequency / count", "Category → how often it appears", "High-cardinality features, trees", "Different categories with the same count collide"],
      ["Target (mean) encoding", "Category → average target value for that category", "High-cardinality features (zip code, product id)", "Leaks the target badly if done naively; use cross-fitting"],
      ["Hashing", "Hash the category into a fixed number of columns", "Huge or streaming vocabularies", "Collisions; not reversible"],
      ["Embeddings", "Learn a small vector per category", "Neural networks with high-cardinality features", "Needs enough data per category"],
    ]],
    ["h", "One-hot encoding from scratch"],
    ["code", R`
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

colours = np.array(["red", "green", "blue", "green", "red"])

# by hand: one column per distinct category
cats = np.unique(colours)                                    # sorted: blue, green, red
onehot = (colours[:, None] == cats[None, :]).astype(int)
print("categories:", cats.tolist())
print(onehot)

enc = OneHotEncoder(handle_unknown="ignore")
print("sklearn matches:", bool((enc.fit_transform(colours[:, None]).toarray() == onehot).all()))
print("unseen 'pink' becomes all zeros:", enc.transform([["pink"]]).toarray().tolist())

sizes = np.array([["small"], ["large"], ["medium"], ["small"]])
ordinal = OrdinalEncoder(categories=[["small", "medium", "large"]])   # give the order explicitly
print("ordinal sizes:", ordinal.fit_transform(sizes).ravel().tolist())
print("pandas get_dummies columns:", pd.get_dummies(pd.Series(colours)).columns.tolist())`, "categories: ['blue', 'green', 'red']\n[[0 0 1]\n [0 1 0]\n [1 0 0]\n [0 1 0]\n [0 0 1]]\nsklearn matches: True\nunseen 'pink' becomes all zeros: [[0.0, 0.0, 0.0]]\nordinal sizes: [0.0, 2.0, 1.0, 0.0]\npandas get_dummies columns: ['blue', 'green', 'red']"],
    ["tip", "For linear models you can drop one one-hot column (`drop=\"first\"`) to avoid a perfect redundancy (the \"dummy variable trap\": the columns always sum to 1). Regularized models and tree models do not mind keeping them all."],
    ["h", "Target encoding and why it leaks"],
    ["p", "A city with 5000 values cannot be one-hot encoded sensibly. **Target encoding** replaces each category by the mean of the target for that category. It is powerful, but each row's own label is part of its category mean, so a rare category is encoded by almost exactly its own label. The model learns the answer from the feature. Here the category and the target are unrelated by construction (pure noise), so any apparent relationship is a leak:"],
    ["code", R`
import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

rng = np.random.default_rng(0)
cat = rng.integers(0, 80, 400)                 # 80 categories, about 5 rows each
y = rng.integers(0, 2, 400)                    # target unrelated to the category
cat_tr, cat_te, y_tr, y_te = cat[:300], cat[300:], y[:300], y[300:]

# naive: mean of the target per category, computed on the same rows we then encode
means = pd.Series(y_tr).groupby(cat_tr).mean()
naive_tr = pd.Series(cat_tr).map(means).to_numpy()
naive_te = pd.Series(cat_te).map(means).fillna(y_tr.mean()).to_numpy()
print("naive      corr with target  train:", round(float(np.corrcoef(naive_tr, y_tr)[0, 1]), 3),
      " test:", round(float(np.corrcoef(naive_te, y_te)[0, 1]), 3))

# scikit-learn's TargetEncoder cross-fits: each training row is encoded using the OTHER folds
te = TargetEncoder(random_state=0)
safe_tr = te.fit_transform(cat_tr[:, None], y_tr).ravel()
safe_te = te.transform(cat_te[:, None]).ravel()
print("cross-fit  corr with target  train:", round(float(np.corrcoef(safe_tr, y_tr)[0, 1]), 3),
      " test:", round(float(np.corrcoef(safe_te, y_te)[0, 1]), 3))`, "naive      corr with target  train: 0.465  test: 0.145\ncross-fit  corr with target  train: -0.077  test: 0.15"],
    ["p", "The naive encoding looks clearly predictive on the training rows but much weaker on test data: the training figure is a leak, because each row's own label is inside its category mean. The cross-fitted version shows about zero on the training rows, which is honest because there is truly no relationship (the test figure only wobbles around zero because 100 rows are few)."],
    ["h", "Which encoder, in practice"],
    ["table", ["Situation", "Choose"], [
      ["Few categories (under ~15), linear model, SVM, kNN, neural net", "One-hot"],
      ["Ordered levels", "Ordinal with the order spelled out"],
      ["Tree-based models, moderate cardinality", "Ordinal or one-hot both work; trees cope with integer codes"],
      ["High cardinality (ids, zip codes)", "Target encoding (cross-fitted), frequency, or hashing; group rare levels into \"other\""],
      ["Neural net with many categories", "Learned embeddings"],
    ]],
    ["danger", "Using plain integer codes (`LabelEncoder` or `.cat.codes`) on a nominal feature for a linear or distance-based model, and using any target-based encoder fitted on all the data. Fit encoders on the training data only (inside a `Pipeline`), and set `handle_unknown=\"ignore\"` so a new category in production does not crash the model."],
    ["note", "`LabelEncoder` is meant for the **target** column, not for features. For features use `OrdinalEncoder` or `OneHotEncoder`. LightGBM, XGBoost (with `enable_categorical`) and CatBoost can also take categorical columns directly."],
    ["tip", "See the complete table-handling recipe (imputer, scaler, encoder in one `ColumnTransformer`) in [Feature Engineering & Scaling](ml.html#feature-engineering)."],
  ],
},
{
  id: "imbalanced", group: "Practical", title: "Imbalanced Data",
  sub: "When one class is rare (fraud, disease, defects), accuracy is misleading and the model can ignore the very cases you care about.",
  tags: ["classification"],
  blocks: [
    ["p", "In fraud detection maybe 1 in 1000 transactions is fraud. A model that always says \"fine\" has 99.9% accuracy and catches nothing. Rare-event problems are everywhere, and they need different **metrics** and sometimes different **training**. Use the sliders to see how a model with decent recall and a small false-positive rate behaves as the positive class gets rarer:"],
    ["viz", "ml2_imbalance"],
    ["p", "Even a good model's **precision** collapses when positives are rare, because a small false-positive rate applied to a huge pile of negatives still produces more false alarms than there are real hits. That is not a bug in the model: it is arithmetic, and it is why accuracy and ROC-AUC alone mislead."],
    ["h", "Step 1: measure the right thing"],
    ["list", [
      "Report the [confusion matrix](ml.html#classification-metrics), **precision, recall and F1** for the rare class.",
      "Use **PR-AUC / average precision** instead of (or next to) [ROC-AUC](ml.html#roc-auc).",
      "Use **stratified** splits and [cross-validation](ml.html#cross-validation) so every fold has enough positives.",
      "Compare against the trivial baseline \"always predict the majority class\" (`DummyClassifier`).",
    ]],
    ["h", "Step 2: choose what to do about it"],
    ["table", ["Approach", "How", "Pros", "Cons"], [
      ["Class weights", "`class_weight=\"balanced\"`: errors on the rare class count more in the loss", "One argument, no new data, no leakage risk", "Not available for every model"],
      ["Move the threshold", "Keep the model, choose the cut-off from the precision-recall trade-off", "Free; often the biggest win", "Needs a validation set and a stated cost trade-off"],
      ["Random over-sampling", "Duplicate minority rows", "Simple", "Can overfit to the duplicated points"],
      ["Random under-sampling", "Drop majority rows", "Faster training", "Throws information away"],
      ["SMOTE", "Create new minority points by interpolating between neighbours", "Smoother than duplicating", "Can create unrealistic points and blur class borders; needs the `imbalanced-learn` package"],
      ["Collect more data / better features", "More positives, or features that separate them", "Addresses the root cause", "Often costly"],
    ]],
    ["diagram", "ml2_smote", [], "SMOTE picks a minority point, picks one of its minority neighbours, and creates a new point somewhere on the line between them."],
    ["h", "Class weights and thresholds in code"],
    ["code", R`
import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=6000, n_features=8, n_informative=4, weights=[0.97, 0.03],
                           class_sep=1.2, flip_y=0.01, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
print("positives in test set:", int(y_te.sum()), "of", len(y_te))

def report(name, pred, score=None):
    ap = f"  AP {average_precision_score(y_te, score):.3f}" if score is not None else ""
    print(f"{name:26s} acc {np.mean(pred == y_te):.3f}  prec {precision_score(y_te, pred, zero_division=0):.3f}"
          f"  rec {recall_score(y_te, pred):.3f}  F1 {f1_score(y_te, pred):.3f}{ap}")

report("always predict majority", DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).predict(X_te))

plain = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
p_plain = plain.predict_proba(X_te)[:, 1]
report("logistic, threshold 0.5", (p_plain > 0.5).astype(int), p_plain)

weighted = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_tr, y_tr)
p_w = weighted.predict_proba(X_te)[:, 1]
report("class_weight=balanced", (p_w > 0.5).astype(int), p_w)

# tune the threshold on TRAINING-set probabilities (a fresh validation split is better in real work)
p_tr = plain.predict_proba(X_tr)[:, 1]
cands = np.linspace(0.05, 0.9, 86)
best_t = max(cands, key=lambda t: f1_score(y_tr, (p_tr > t).astype(int)))
report(f"plain, threshold {best_t:.2f}", (p_plain > best_t).astype(int), p_plain)`, "positives in test set: 81 of 2400\nalways predict majority    acc 0.966  prec 0.000  rec 0.000  F1 0.000\nlogistic, threshold 0.5    acc 0.970  prec 0.917  rec 0.136  F1 0.237  AP 0.324\nclass_weight=balanced      acc 0.723  prec 0.072  rec 0.605  F1 0.128  AP 0.262\nplain, threshold 0.21      acc 0.969  prec 0.571  rec 0.296  F1 0.390  AP 0.324"],
    ["p", "Accuracy is nearly useless here: the \"always majority\" baseline scores 0.966 and the plain model 0.970. Recall, precision and F1 tell the real story. Class weights raise recall a lot but cost precision (and accuracy); moving the threshold reaches a similar trade-off without changing the model."],
    ["h", "SMOTE from scratch"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
minority = rng.normal(loc=[2.0, 2.0], scale=0.4, size=(10, 2))     # 10 rare points

def smote(X_min, n_new, k=3, rng=rng):
    D = np.sqrt(((X_min[:, None, :] - X_min[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)                                    # a point is not its own neighbour
    nn = np.argsort(D, axis=1)[:, :k]                              # k nearest minority neighbours
    new = []
    for _ in range(n_new):
        i = rng.integers(len(X_min))                               # pick a minority point
        j = nn[i, rng.integers(k)]                                 # pick one of its neighbours
        lam = rng.random()                                         # somewhere on the segment between them
        new.append(X_min[i] + lam * (X_min[j] - X_min[i]))
    return np.array(new)

synthetic = smote(minority, n_new=20)
print("synthetic shape:", synthetic.shape)
print("x range of originals :", minority[:, 0].min().round(2), minority[:, 0].max().round(2))
print("x range of synthetics:", synthetic[:, 0].min().round(2), synthetic[:, 0].max().round(2))
print("all synthetics inside the originals' bounding box:",
      bool((synthetic >= minority.min(axis=0)).all() and (synthetic <= minority.max(axis=0)).all()))`, "synthetic shape: (20, 2)\nx range of originals : 1.07 2.52\nx range of synthetics: 1.17 2.22\nall synthetics inside the originals' bounding box: True"],
    ["p", "In real projects use `imblearn.over_sampling.SMOTE` (the `imbalanced-learn` package, which is not part of scikit-learn) inside an `imblearn.pipeline.Pipeline`, so the resampling happens only on the training part of each fold."],
    ["danger", "**Resampling before splitting.** If you oversample the whole dataset first, copies or neighbours of the same row end up in both train and validation, and your score is wildly optimistic. Split first (stratified), resample only the training data, and evaluate on the untouched, still-imbalanced validation / test data."],
    ["warn", "Resampling and class weights change the model's predicted probabilities: they no longer match real-world frequencies. If you need calibrated probabilities (for example to compute expected cost), recalibrate on untouched data."],
    ["tip", "Order of attack: (1) the right metric, (2) a stratified split, (3) class weights or a tuned threshold, (4) resampling only if still needed. Then ask whether more positive examples or better features are available."],
    ["tip", "See also: [Confusion Matrix, Precision, Recall & F1](ml.html#classification-metrics) and [ROC-AUC & PR Curves](ml.html#roc-auc)."],
  ],
},
{
  id: "model-selection", group: "Practical", title: "Which Model When: A Cheat Sheet",
  sub: "A practical guide to choosing between the models in this course, by data type, size, and what you need from the model.",
  tags: ["cheat sheet"],
  blocks: [
    ["p", "There is no best model. The best choice depends on your data (how many rows, what kinds of columns), what you need from the model (accuracy, speed, explanations), and how much time you have. This page puts the whole course on one map. The aim is a good short list of candidates, not a final answer: you still decide by [cross-validation](ml.html#cross-validation) on your own data."],
    ["diagram", "ml2_modelmap", [], "A map of the models in this course by task. Start from what you want to predict."],
    ["h", "Start from the data"],
    ["table", ["Your data", "First try", "Then try", "Why"], [
      ["Tabular, mixed columns, a few thousand to millions of rows", "Gradient boosting ([XGBoost / LightGBM](ml.html#boosting))", "[Random forest](ml.html#random-forests), logistic / linear as a baseline", "Boosted trees win on most tabular benchmarks and need little preprocessing"],
      ["Small tabular data (under ~1000 rows)", "[Logistic / linear regression](ml.html#logistic-regression) with regularization", "[SVM](ml.html#svm), random forest, [Naive Bayes](ml.html#naive-bayes)", "Simple models overfit less when data is scarce"],
      ["Many features, few rows (genes, text)", "Regularized linear model (L1 / L2)", "Linear SVM, Naive Bayes", "More features than rows: high variance unless the model is constrained"],
      ["Text", "TF-IDF + logistic regression or linear SVM", "[Naive Bayes](ml.html#naive-bayes), then transformer embeddings", "A strong, fast baseline before reaching for neural nets"],
      ["Images, audio, video", "Pretrained deep network, fine-tuned (see [transfer learning](dl.html#transfer-learning))", "[CNNs](dl.html#cnn)", "Classical models need hand-made features here"],
      ["Sequences / time series", "Lag features + gradient boosting, or a seasonal baseline", "[RNN / LSTM](dl.html#rnn), transformers", "Check a naive \"same as last period\" baseline first"],
      ["No labels, find groups", "[k-Means](ml.html#kmeans) (scaled)", "[DBSCAN](ml.html#dbscan), hierarchical", "Depends on cluster shape and noise"],
      ["Too many features, need to compress or visualise", "[PCA](ml.html#pca)", "t-SNE / UMAP for plots only", "PCA is linear, fast and reversible"],
      ["Anomaly / outlier detection", "Isolation Forest, or DBSCAN noise points", "One-class SVM, distance to k-th neighbour", "Few or no labelled anomalies"],
    ]],
    ["h", "Compare the models directly"],
    ["table", ["Model", "Task", "Needs scaling?", "Non-linear?", "Interpretable?", "Training speed", "Typical weakness"], [
      ["[Linear regression](ml.html#linear-regression) / Ridge / Lasso", "Regression", "Yes (for penalties)", "No (add features)", "High", "Very fast", "Straight lines only"],
      ["[Logistic regression](ml.html#logistic-regression)", "Classification", "Yes", "No (add features)", "High", "Fast", "Linear boundary"],
      ["[k-NN](ml.html#knn)", "Both", "**Yes**", "Yes", "Medium", "None (lazy), slow predictions", "Slow on big data; hurt by irrelevant features"],
      ["[Naive Bayes](ml.html#naive-bayes)", "Classification", "No", "Limited", "High", "Very fast", "Independence assumption; poor probabilities"],
      ["[SVM](ml.html#svm) (kernel)", "Both", "**Yes**", "Yes", "Low", "Slow beyond ~50k rows", "Scales badly; needs tuning of C, gamma"],
      ["[Decision tree](ml.html#decision-trees)", "Both", "No", "Yes", "High (small trees)", "Fast", "Overfits; unstable"],
      ["[Random forest](ml.html#random-forests)", "Both", "No", "Yes", "Medium", "Medium; parallel", "Big and slow to predict; weaker extrapolation"],
      ["[Gradient boosting](ml.html#boosting)", "Both", "No", "Yes", "Low-medium", "Medium", "Needs tuning; sensitive to noise and leakage"],
      ["Neural network ([MLP](dl.html#mlp))", "Both", "**Yes**", "Yes", "Low", "Slow; needs a GPU at scale", "Needs lots of data and tuning"],
      ["[k-Means](ml.html#kmeans)", "Clustering", "**Yes**", "No (round blobs)", "Medium", "Fast", "Must choose k; round clusters"],
      ["[DBSCAN](ml.html#dbscan)", "Clustering", "**Yes**", "Yes (any shape)", "Medium", "Fast-medium", "Single density scale"],
      ["[PCA](ml.html#pca)", "Reduction", "**Yes**", "No", "Medium", "Fast", "Linear; ignores the target"],
    ]],
    ["h", "Start from what you need"],
    ["table", ["If you need…", "Lean toward…", "Avoid…"], [
      ["To explain every decision (regulation, medicine)", "Linear / logistic, a small tree", "Large ensembles and deep nets, unless paired with explanation tools"],
      ["The best accuracy on a table", "Gradient boosting, ensembles of a few model types", "A single decision tree"],
      ["Very fast predictions (milliseconds, on a phone)", "Linear models, small trees", "k-NN on large data, large forests"],
      ["Training on a huge dataset quickly", "Linear models with SGD, LightGBM, a neural net with mini-batches", "Kernel SVM, k-NN, hierarchical clustering"],
      ["Probabilities you can trust", "Logistic regression; calibrate others (`CalibratedClassifierCV`)", "Raw Naive Bayes or SVM scores as probabilities"],
      ["Robustness to outliers and mixed scales without much preprocessing", "Tree-based models", "Distance-based models on raw features"],
      ["Handle missing values natively", "Histogram gradient boosting (`HistGradientBoosting*`), XGBoost, LightGBM", "Linear models without imputation"],
    ]],
    ["h", "A bake-off in code"],
    ["p", "In practice: build a quick baseline, then compare a few candidates with the same cross-validation, each in a pipeline so the preprocessing is fitted inside the folds. This takes minutes and tells you far more than opinions do."],
    ["code", R`
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

models = {
    "baseline (majority)": DummyClassifier(strategy="most_frequent"),
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "k-NN (k=7)": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
    "naive Bayes": GaussianNB(),
    "decision tree": DecisionTreeClassifier(max_depth=4, random_state=0),
    "random forest": RandomForestClassifier(n_estimators=100, random_state=0),
    "gradient boosting": HistGradientBoostingClassifier(random_state=0),
}
for name, m in models.items():
    s = cross_val_score(m, X, y, cv=cv)
    print(f"{name:22s} {s.mean():.3f} +/- {s.std():.3f}")`, "baseline (majority)    0.627 +/- 0.004\nlogistic regression    0.979 +/- 0.014\nk-NN (k=7)             0.965 +/- 0.019\nSVM (RBF)              0.977 +/- 0.007\nnaive Bayes            0.939 +/- 0.020\ndecision tree          0.926 +/- 0.025\nrandom forest          0.965 +/- 0.015\ngradient boosting      0.970 +/- 0.009"],
    ["p", "Note how close the strong models are: on a small, clean, mostly linear dataset like this, careful preprocessing and a simple model are as good as anything. Differences of a fraction of a percent are within noise (see the +/- column). Prefer the simpler model when scores tie."],
    ["h", "A sensible default workflow"],
    ["steps", [
      "Define the metric that matches the cost of mistakes ([classification metrics](ml.html#classification-metrics), [regression metrics](ml.html#regression-metrics)).",
      "Split off a test set, stratified for classification, grouped or time-ordered if the data demands it.",
      "Build a trivial baseline (majority class, mean, last value).",
      "Add a simple, scaled linear model. Check the [bias-variance](ml.html#bias-variance) picture with a learning curve.",
      "Try a tree ensemble (random forest, then gradient boosting). On tables this is usually the winner.",
      "Tune the best one or two with random search inside cross-validation. Improve **features** before pushing tuning further.",
      "Evaluate once on the test set, check the errors by hand, and choose the threshold using real costs.",
    ]],
    ["danger", "Picking a model because it is famous. Deep learning is not automatically better on a 5000-row spreadsheet, and a tuned gradient-boosting model is rarely beaten there. Likewise: comparing models with different preprocessing, different splits, or after peeking at the test set."],
    ["note", "As of 2026, gradient-boosted trees (XGBoost, LightGBM, CatBoost) remain the usual strongest starting point for tabular data, while deep learning dominates images, audio, text and other unstructured data. Treat this as a guide to try first, never a rule."],
    ["tip", "Related: [The ML Workflow](ml.html#workflow), [Feature Engineering & Scaling](ml.html#feature-engineering), [Imbalanced Data](ml.html#imbalanced)."],
  ],
},
/*END*/);
