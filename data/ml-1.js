/* Machine Learning, part 1: foundations, regression, classification
   Outputs written as "?" are filled in by running the code: node tools/fill.js data/ml-1.js */
ML.push(
{
  id: "what-is-ml", group: "Foundations", title: "What is Machine Learning?",
  sub: "Instead of writing the rules by hand, you show a program examples and let it find the rules.",
  tags: ["basics"],
  blocks: [
    ["p", "In normal programming you write the rules and the computer applies them to data. In **machine learning** you flip it: you give the computer **data and the right answers**, and it works out the rules itself. The result of that search is called a **model**: a function that takes new inputs and returns predictions."],
    ["p", "Why bother? Some rules are too hard to write by hand. Nobody can list every rule that tells a spam email from a normal one, or a cat photo from a dog photo. But we can collect thousands of examples, and a learning algorithm can pick out the patterns."],
    ["diagram", "ml1_paradigm", [], "Classic programming turns rules + data into answers. Machine learning turns data + answers into rules (the model)."],
    ["h", "The vocabulary"],
    ["table", ["Word", "Meaning", "Example (house prices)"], [
      ["Example / sample / row", "One thing we have data about", "One house"],
      ["Features $X$ (inputs)", "The measurements we feed in", "Area, bedrooms, age"],
      ["Label / target $y$", "The answer we want to predict", "Sale price"],
      ["Parameters", "Numbers the model learns", "The weights of a line"],
      ["Hyperparameters", "Settings *you* choose before training", "Learning rate, tree depth"],
      ["Training", "Searching for the parameters that fit the data", "Gradient descent"],
      ["Inference", "Using the trained model on new data", "Predicting a new listing"],
    ]],
    ["h", "The three main kinds of learning"],
    ["table", ["Kind", "What the data looks like", "Goal", "Examples"], [
      ["**Supervised**", "Inputs **with** labels", "Predict the label for new inputs", "Price prediction, spam filter"],
      ["**Unsupervised**", "Inputs only, no labels", "Find structure in the data", "Customer groups, compression"],
      ["**Reinforcement**", "An agent acting and getting rewards", "Choose actions that earn the most reward", "Game playing, robot control"],
    ]],
    ["p", "Supervised learning has two flavours. **Regression** predicts a number (a price, a temperature). **Classification** predicts a category (spam or not, which digit). The next two groups of sections cover these. Many modern language models also use **self-supervised** learning: the labels come free from the data itself (predict the next word of a text), which is how they can learn from the whole internet."],
    ["h", "A first model: learn a rule from examples"],
    ["p", "Say we know how many hours each student studied and whether they passed. Instead of guessing a rule like \"pass if hours > 5\", let the computer try every possible cut-off and keep the one that gets the most students right. That is the smallest honest example of learning from data:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
hours = rng.uniform(0, 10, size=60)
# the (hidden) truth: more hours -> more likely to pass, with some luck involved
passed = (hours + rng.normal(0, 1.5, size=60) > 5).astype(int)

best_t, best_acc = None, 0.0
for t in np.linspace(0, 10, 101):             # try cut-offs 0.0, 0.1, ..., 10.0
    pred = (hours > t).astype(int)            # the rule: "pass if hours > t"
    acc = np.mean(pred == passed)             # fraction of students it gets right
    if acc > best_acc:
        best_t, best_acc = t, acc

print("learned cut-off:", round(float(best_t), 1), " training accuracy:", round(float(best_acc), 3))
print("prediction for 7.5 hours:", int(7.5 > best_t))`, "learned cut-off: 3.6  training accuracy: 0.95\nprediction for 7.5 hours: 1"],
    ["p", "Here the **model** is the rule \"hours > t\", the **parameter** is the cut-off $t$, and **training** is the loop that picked it. Real models have many more parameters and smarter search methods, but the shape is always the same: define a family of rules, define how to score them, search for the best one."],
    ["note", "Training accuracy only tells you how well the rule fits the examples it has already seen. What we actually care about is how it does on **new** data. That is the subject of [Train / Validation / Test](ml.html#splits) and [Bias–Variance](ml.html#bias-variance)."],
    ["tip", "ML is not always the answer. If a simple, known rule solves the problem, write the rule. Use ML when the rules are unknown or too many, and you have enough examples."],
  ],
},
{
  id: "workflow", group: "Foundations", title: "The ML Workflow",
  sub: "The steps of a real project, from a vague question to a model you can trust.",
  tags: ["process"],
  blocks: [
    ["p", "Most of the work in machine learning is not the model. It is everything around it: framing the question, getting clean data, checking that the result is honest. The steps are always roughly the same:"],
    ["diagram", "ml1_workflow", [], "The loop: frame → data → split → baseline → train → evaluate → iterate → deploy and monitor."],
    ["steps", [
      "**Frame the problem.** What exactly are you predicting, and what would a useful answer be worth? Regression or classification? What counts as success (one number you can measure)?",
      "**Get and inspect the data.** Look at it: sizes, types, missing values, duplicates, class balance, weird outliers. Plot it.",
      "**Split it** into train, validation and test sets *before* doing anything that learns from the data. See [splits](ml.html#splits).",
      "**Build a baseline.** Predict the average, or the most common class. Any real model must beat this, or it is not worth its complexity.",
      "**Train a simple model first** (linear or logistic regression), then try stronger ones (trees, boosting).",
      "**Evaluate** on the validation set with the metric from step 1. Look at the errors, not just the score.",
      "**Iterate.** Better features, different model, tuning of hyperparameters. Touch the test set only once, at the very end.",
      "**Deploy and monitor.** Real-world data drifts, so a model that was good last year may quietly get worse.",
    ]],
    ["h", "The whole loop in a few lines"],
    ["p", "Here is the smallest complete version, on a dataset that ships with scikit-learn. Notice the order: split first, baseline, then the real model."],
    ["code", R`
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
print("data shape:", X.shape, " share of class 1:", round(float(y.mean()), 3))

# 1. split first (stratify keeps the class ratio the same in both parts)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# 2. baseline: always predict the most common class
base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
print("baseline accuracy:", round(base.score(X_te, y_te), 3))

# 3. a real model: scale the features, then logistic regression
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_tr, y_tr)
print("model accuracy   :", round(model.score(X_te, y_te), 3))`, "data shape: (569, 30)  share of class 1: 0.627\nbaseline accuracy: 0.629\nmodel accuracy   : 0.958"],
    ["p", "The baseline already scores well here because one class is more common. This is why you always compute it: a model with 63% accuracy sounds fine until you learn that a constant guess gets 63%."],
    ["warn", "Accuracy alone can mislead, especially when classes are unbalanced. Pick the metric that matches the real cost of mistakes (see [Confusion Matrix, Precision, Recall & F1](ml.html#classification-metrics))."],
    ["tip", "Make the first version of the whole pipeline ugly but end-to-end, then improve the weakest part. A simple model that is running beats a clever one that is not."],
  ],
},
{
  id: "splits", group: "Foundations", title: "Train / Validation / Test & Data Leakage",
  sub: "Why you must test a model on data it has never seen, and the sneaky ways information leaks in.",
  tags: ["evaluation"],
  blocks: [
    ["p", "A student who memorised last year's exam answers will ace last year's exam and fail a new one. A model can do exactly the same: remember the training examples instead of learning the pattern. The only way to catch this is to measure it on examples it was **never allowed to learn from**."],
    ["diagram", "ml1_split", [], "Three roles: the train set fits the parameters, the validation set chooses between models and settings, the test set is opened once for the final honest score."],
    ["table", ["Set", "Used for", "How often you look"], [
      ["**Train** (~60–80%)", "Fitting the parameters", "Constantly"],
      ["**Validation** (~10–20%)", "Choosing the model type, hyperparameters, features", "Many times"],
      ["**Test** (~10–20%)", "One final, honest estimate of performance", "**Once**"],
    ]],
    ["p", "Why a separate validation set? Each time you look at a score and then change something, a little information about that set leaks into your choices. After a hundred rounds of tuning, the validation score is optimistic because you have fitted your *decisions* to it. The test set is the one you keep untouched so that it stays honest. With little data, [cross-validation](ml.html#cross-validation) replaces the single validation set."],
    ["h", "Splitting in code"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

# first carve off the test set, then split the rest into train / validation
X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0)

print("sizes  train/val/test:", len(X_tr), len(X_va), len(X_te))
print("class counts in train:", np.bincount(y_tr).tolist())
print("class counts in test :", np.bincount(y_te).tolist())`, "sizes  train/val/test: 90 30 30\nclass counts in train: [30, 30, 30]\nclass counts in test : [10, 10, 10]"],
    ["h", "Data leakage"],
    ["p", "**Data leakage** means information from outside the training set sneaks into training, so your validation score looks great and then the model fails in the real world. It is the most common way beginners fool themselves. The usual forms:"],
    ["list", [
      "**Preprocessing on all the data.** Computing the mean for scaling, filling missing values, or picking features using the *whole* dataset before splitting. The test set has then influenced the training.",
      "**Target leakage.** A feature that secretly contains the answer, such as \"refund issued\" when predicting whether an order was returned, or a column recorded *after* the outcome.",
      "**Duplicates and near-duplicates** across the split (the same patient or the same photo in both train and test).",
      "**Time.** Shuffling time-ordered data lets the model train on the future. For forecasting, always split by time: train on the past, test on the later period.",
    ]],
    ["p", "Here is a striking demonstration. The data below is pure noise: 100 rows, 2000 random features and random labels, so *no* model can really beat 50%. Watch what happens when we select the \"best\" features using all the data before cross-validating, versus doing it correctly inside a pipeline."],
    ["code", R`
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2000))          # pure noise
y = rng.integers(0, 2, size=100)          # random labels: the true accuracy is 50%

# WRONG: pick the 20 best-looking features using ALL rows, then cross-validate
X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
wrong = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=5).mean()

# RIGHT: the selection is part of the pipeline, so it is refit inside each training fold
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
right = cross_val_score(pipe, X, y, cv=5).mean()

print("leaky accuracy   :", round(float(wrong), 3))
print("honest accuracy  :", round(float(right), 3))`, "leaky accuracy   : 0.78\nhonest accuracy  : 0.59"],
    ["danger", "Any step that *learns* something from data (scaler, imputer, encoder, feature selector, PCA, the model) must be fitted on the training part only, then applied to validation and test. Put them all in a scikit-learn `Pipeline` and the library does this correctly for you, also inside cross-validation. See [Pipelines in scikit-learn](pyml.html#sklearn-pipelines)."],
    ["tip", "Rules of thumb: split first; stratify for classification; split by time for time series; split by group (patient, user) when rows from one group are related; and never tune on the test set."],
  ],
},
{
  id: "bias-variance", group: "Foundations", title: "Bias–Variance, Under- & Overfitting",
  sub: "Too simple a model misses the pattern; too flexible a model memorises the noise. The art is the middle.",
  tags: ["core idea"],
  blocks: [
    ["p", "Every model makes mistakes for two very different reasons. **Bias** is error from being too simple: a straight line can never follow a curve, however much data you give it. **Variance** is error from being too sensitive to the particular training set: a very wiggly curve changes completely if you collect slightly different data."],
    ["diagram", "ml1_biasvar", [], "As model flexibility grows, bias falls and variance rises. Test error is the sum plus unavoidable noise, so it is U-shaped; training error keeps falling."],
    ["table", ["", "Underfitting", "Good fit", "Overfitting"], [
      ["Model", "Too simple", "Just flexible enough", "Too flexible"],
      ["Training error", "High", "Low", "Very low"],
      ["Test error", "High", "Low", "High"],
      ["Problem", "High **bias**", "-", "High **variance**"],
      ["Fix", "More features, a richer model, less regularization", "-", "More data, simpler model, regularization, early stopping"],
    ]],
    ["h", "See it happen"],
    ["p", "The visualizer below fits polynomials of degree $d$ to points from a smooth curve, and refits the same model on 25 different random training sets, drawing all the fits at once. Slide $d$ up from 1. Low degree: the fits agree with each other but are all wrong in the same way (bias). High degree: each fit is accurate on average but they all disagree with each other (variance)."],
    ["viz", "ml1_bvsim"],
    ["h", "The maths: decomposing the error"],
    ["p", "Suppose the truth is $y = f(x) + \\varepsilon$, where $\\varepsilon$ is noise with variance $\\sigma^2$, and $\\hat f$ is our model trained on a random training set. The expected squared error at a point $x$ splits into three parts:"],
    ["math", R`\mathbb{E}\big[(y - \hat f(x))^2\big] = \underbrace{\big(\mathbb{E}[\hat f(x)] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}\big[(\hat f(x) - \mathbb{E}[\hat f(x)])^2\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}`, "the expectation is over different random training sets"],
    ["list", [
      "**Bias²**: how far the model's *average* prediction is from the truth.",
      "**Variance**: how much the prediction jumps around from one training set to another.",
      "**Noise σ²**: randomness in the data itself. No model can remove it.",
    ]],
    ["h", "Measuring it with train and test error"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)                  # the true curve
x_tr = rng.uniform(0, 1, 20);  y_tr = f(x_tr) + rng.normal(0, 0.25, 20)
x_te = rng.uniform(0, 1, 200); y_te = f(x_te) + rng.normal(0, 0.25, 200)

for degree in [1, 3, 9]:
    coef = np.polyfit(x_tr, y_tr, degree)            # least-squares polynomial fit
    tr = np.mean((np.polyval(coef, x_tr) - y_tr) ** 2)
    te = np.mean((np.polyval(coef, x_te) - y_te) ** 2)
    print(f"degree {degree}: train MSE {tr:6.3f}   test MSE {te:7.3f}")`, "degree 1: train MSE  0.302   test MSE   0.290\ndegree 3: train MSE  0.026   test MSE   0.098\ndegree 9: train MSE  0.014   test MSE   0.107"],
    ["p", "Training error only goes down as the model gets more flexible. Test error goes down and then up. That gap between the two is the signature of overfitting."],
    ["h", "What to do about it"],
    ["list", [
      "**Underfitting** (both errors high): add features, use a more flexible model, reduce regularization, train longer.",
      "**Overfitting** (train error much lower than validation error): get more data, simplify the model, add [regularization](ml.html#regularization), use fewer features, average several models ([bagging](ml.html#random-forests)).",
      "Always diagnose with the **gap**: compare training and validation scores before changing anything.",
    ]],
    ["note", "Modern deep networks can fit their training data perfectly and still generalise well, which seems to break the U-shape (the so-called double descent). For the classic models in this part of the page, the U-shaped picture is the right mental model."],
    ["warn", "Do not pick the model that has the lowest *training* error. A model that memorises always wins that contest."],
  ],
},
{
  id: "loss-functions", group: "Foundations", title: "Loss Functions & Empirical Risk",
  sub: "A loss turns \"how wrong is this prediction?\" into a number that training can push down.",
  tags: ["training"],
  blocks: [
    ["p", "A **loss function** $\\ell(y, \\hat y)$ scores one prediction: it is 0 if the model is perfectly right and bigger the worse it is. Training is the search for the parameters that make the **average loss over the training set** as small as possible. That average is called the **empirical risk**:"],
    ["math", R`\hat R(\theta) = \frac{1}{n}\sum_{i=1}^{n} \ell\big(y_i,\; f_\theta(x_i)\big) \qquad \hat\theta = \arg\min_\theta \hat R(\theta)`, "f_θ is the model with parameters θ; we minimise the average loss on the data we have"],
    ["p", "\"Empirical\" because it is measured on a finite sample. What we really want is the **true risk**, the average loss on all future data, which we can never compute. Minimising the empirical risk is a stand-in, and this is why [overfitting](ml.html#bias-variance) is possible: the stand-in can look much better than the real thing."],
    ["h", "Losses for regression"],
    ["diagram", "ml1_losses", [], "Regression losses as a function of the residual r = y − ŷ. Squared loss punishes large errors heavily; absolute loss grows steadily; Huber is quadratic near zero and linear far away."],
    ["table", ["Loss", "Formula", "Behaviour"], [
      ["**Squared error** (MSE)", "$(y - \\hat y)^2$", "Smooth; one big mistake counts a lot, so outliers pull the fit. Its best constant prediction is the **mean**."],
      ["**Absolute error** (MAE)", "$|y - \\hat y|$", "Robust to outliers; not smooth at 0. Its best constant is the **median**."],
      ["**Huber**", "$\\tfrac12 r^2$ if $|r| \\le \\delta$, else $\\delta(|r| - \\tfrac12\\delta)$", "Squared near zero, linear in the tails: smooth *and* robust."],
    ]],
    ["h", "Losses for classification"],
    ["diagram", "ml1_losses", ["clf"], "Classification losses as a function of the margin m = y·f(x) with labels in {−1, +1}. The 0–1 loss counts mistakes but cannot be optimised by gradients; log loss and hinge are smooth-ish stand-ins."],
    ["table", ["Loss", "Formula", "Used by"], [
      ["**0–1 loss**", "$1$ if wrong, else $0$", "What we *report* (accuracy), but it has no useful gradient"],
      ["**Log loss / cross-entropy**", "$-[\\,y\\log p + (1-y)\\log(1-p)\\,]$", "[Logistic regression](ml.html#logistic-regression), neural nets"],
      ["**Hinge**", "$\\max(0,\\; 1 - y\\,f(x))$", "[Support vector machines](ml.html#svm)"],
    ]],
    ["p", "Log loss is the negative log of the probability the model gave to the right answer. Giving 0.9 to the truth costs $-\\ln 0.9 \\approx 0.11$; giving it 0.01 costs $-\\ln 0.01 \\approx 4.6$. Confident and wrong is punished hard. See [entropy and cross-entropy](math.html#entropy) for the information-theory view."],
    ["h", "The loss shapes the model"],
    ["p", "Different losses give different answers on the same data. With an outlier in the targets, the squared-error fit chases it and the absolute-error fit mostly ignores it:"],
    ["code", R`
import numpy as np

y = np.array([2.0, 3.0, 3.0, 4.0, 50.0])         # one huge outlier

def mse(c):  return np.mean((y - c) ** 2)
def mae(c):  return np.mean(np.abs(y - c))
def huber(c, d=1.0):
    r = np.abs(y - c)
    return np.mean(np.where(r <= d, 0.5 * r ** 2, d * (r - 0.5 * d)))

cs = np.linspace(0, 55, 5501)                     # candidate constant predictions
for name, loss in [("MSE", mse), ("MAE", mae), ("Huber", huber)]:
    vals = [loss(c) for c in cs]
    print(f"{name:5s} best constant = {cs[int(np.argmin(vals))]:.2f}")
print("mean =", y.mean(), " median =", float(np.median(y)))

# binary cross-entropy for a few predictions of the true class
for p in [0.9, 0.6, 0.01]:
    print(f"p(true class) = {p:<4}  log loss = {-np.log(p):.3f}")`, "MSE   best constant = 12.40\nMAE   best constant = 3.00\nHuber best constant = 3.33\nmean = 12.4  median = 3.0\np(true class) = 0.9   log loss = 0.105\np(true class) = 0.6   log loss = 0.511\np(true class) = 0.01  log loss = 4.605"],
    ["tip", "Choose the loss to match what you care about: MSE if big errors are truly worse and the data is clean, MAE or Huber if outliers are junk, cross-entropy for probabilities. The metric you *report* (accuracy, F1) is often different from the loss you *train* on, because the metric may not be differentiable."],
    ["warn", "A different scale of target means a different scale of loss. Compare MSE values only between models on the same target."],
  ],
},
{
  id: "linear-regression", group: "Regression", title: "Linear Regression",
  sub: "Predict a number as a weighted sum of the features, and pick the weights that make the squared errors smallest.",
  tags: ["regression"],
  blocks: [
    ["p", "Linear regression is the simplest and most useful model in the book. It draws the best straight line (or flat plane, in more dimensions) through the data. Each feature gets a **weight** that says how much the prediction changes when that feature goes up by one; a constant **intercept** sets the starting level."],
    ["p", "\"Best\" has a precise meaning: the line that makes the sum of the **squared residuals** smallest, where a residual is the vertical gap between a point and the line. This is called **least squares**. Try it by hand below: drag the slope and intercept and watch the mean squared error. The squares are the residuals squared, and the line you are looking for is the one with the smallest total area."],
    ["viz", "ml1_linfit"],
    ["h", "The maths"],
    ["p", "With $d$ features, the prediction for one example is a weighted sum plus the intercept $b$:"],
    ["math", R`\hat y = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b = \mathbf{w}^\top \mathbf{x} + b`],
    ["p", "To handle $b$ neatly, add a column of ones to the feature matrix so that $b$ becomes just another weight. For all $n$ examples at once, with $X$ of shape $(n, d+1)$:"],
    ["math", R`\hat{\mathbf y} = X\mathbf w \qquad L(\mathbf w) = \frac{1}{n}\lVert X\mathbf w - \mathbf y\rVert^2`],
    ["p", "$L$ is a bowl-shaped (convex) function of $\\mathbf w$, so the minimum is exactly where the gradient is zero. Setting $\\nabla L = \\frac{2}{n}X^\\top(X\\mathbf w - \\mathbf y) = 0$ gives the **normal equation**:"],
    ["math", R`X^\top X\,\mathbf w = X^\top \mathbf y \quad\Longrightarrow\quad \mathbf w = (X^\top X)^{-1} X^\top \mathbf y`, "an exact answer in one step: no learning rate, no loop"],
    ["note", "In code, never form the inverse. Solve the linear system with `np.linalg.solve(X.T @ X, X.T @ y)`, or better, call `np.linalg.lstsq`, which also copes with awkward matrices. See [matrix inverse](math.html#matrix-ops)."],
    ["h", "From scratch in NumPy"],
    ["p", "We make data from a known rule, $y = 3 + 2x_1 - 1.5x_2$ plus noise, and check that least squares recovers it."],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
n = 200
X = rng.normal(size=(n, 2))                                   # two features
y = 3 + 2 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.5, n)   # true: b=3, w=(2, -1.5)

Xb = np.hstack([np.ones((n, 1)), X])                          # add the column of ones for the intercept
w = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)                      # normal equation
print("weights [b, w1, w2]:", w.round(3))

w2, *_ = np.linalg.lstsq(Xb, y, rcond=None)                   # the robust library routine
print("same with lstsq    :", w2.round(3))

pred = Xb @ w
mse = np.mean((y - pred) ** 2)
r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)   # share of variance explained
print("MSE:", round(float(mse), 3), " R^2:", round(float(r2), 3))`, "weights [b, w1, w2]: [ 2.999  1.975 -1.44 ]\nsame with lstsq    : [ 2.999  1.975 -1.44 ]\nMSE: 0.246  R^2: 0.958"],
    ["p", "The learned weights are close to the true ones; the leftover error (MSE about 0.25) is the noise we added, which no model could remove. **$R^2$** compares the model to the dumb model that always predicts the mean: 1 is perfect, 0 is no better than the mean."],
    ["h", "The scikit-learn version"],
    ["code", R`
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2))
y = 3 + 2 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.5, 200)

model = LinearRegression().fit(X, y)         # adds the intercept for you
print("intercept:", round(float(model.intercept_), 3))
print("weights  :", model.coef_.round(3))
print("R^2      :", round(model.score(X, y), 3))
print("predict x=(1, 1):", model.predict([[1.0, 1.0]]).round(3))`, "intercept: 2.999\nweights  : [ 1.975 -1.44 ]\nR^2      : 0.958\npredict x=(1, 1): [3.533]"],
    ["h", "Normal equation or gradient descent?"],
    ["table", ["", "Normal equation", "[Gradient descent](ml.html#gradient-descent)"], [
      ["Steps", "One solve", "Many small steps"],
      ["Learning rate", "None", "Must be tuned"],
      ["Cost", "About $O(nd^2 + d^3)$: slow with huge $d$", "$O(nd)$ per step: scales to huge data"],
      ["Feature scaling", "Not needed", "Needed for fast convergence"],
      ["Works for", "Linear least squares only", "Almost any model"],
    ]],
    ["h", "Pros, cons and settings"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Fast, exact, and nothing to tune", "Can only draw straight lines (see [polynomial features](ml.html#polynomial))"],
      ["Easy to interpret: each weight is an effect", "Sensitive to outliers because errors are squared"],
      ["A strong baseline for any number-prediction task", "Unstable when features are strongly correlated (collinear); fix with [regularization](ml.html#regularization)"],
    ]],
    ["table", ["Setting", "Meaning"], [
      ["`fit_intercept=True`", "Learn the constant term $b$. Turn off only if your data is already centred."],
      ["`positive=False`", "Force all weights to be non-negative if set to True."],
    ]],
    ["list", [
      "**Assumptions** worth checking: the relationship is roughly linear, residuals have no pattern and similar spread, and features are not near-duplicates. Plot residuals against predictions.",
      "**Interpretation**: a weight is the change in the prediction for a one-unit change in that feature, *holding the others fixed*. It is a statement about the model, not necessarily about cause and effect.",
    ]],
    ["danger", "Do not extrapolate far beyond the range of your training data: a line keeps going forever, the real world usually does not. And do not read weights as importances unless the features are on the same scale."],
    ["tip", "Practice this in [Least Squares with the Normal Equation](practice.html#p/linreg-normal-equation)."],
  ],
},
{
  id: "gradient-descent", group: "Regression", title: "Gradient Descent",
  sub: "How almost every model learns: measure how wrong you are, find which way is downhill, take a small step, repeat.",
  tags: ["optimizer"],
  blocks: [
    ["p", "A model has **parameters** (numbers it can change, like the slope and intercept of a line). A **loss** says how wrong its predictions are. Training means finding the parameters that make the loss small. For most models there is no formula that jumps straight to the answer, so we walk there."],
    ["h", "The idea"],
    ["p", "Picture the loss as a hilly landscape and yourself standing on it in fog. You can't see the lowest valley, but you can feel which way the ground slopes under your feet. So you take a small step downhill, feel the slope again, and repeat. The slope is the **gradient**; the size of your step is the **learning rate**."],
    ["math", R`\theta \leftarrow \theta - \eta \, \nabla_{\theta} L(\theta)`, "θ: the parameters · η (eta): the learning rate · ∇L: the gradient of the loss"],
    ["p", "The gradient $\\nabla L$ points in the direction where the loss grows fastest, so we subtract it. With one parameter it is just the derivative $L'(\\theta)$: positive slope → move left, negative slope → move right."],
    ["viz", "gd1d"],
    ["table", ["Learning rate η", "What happens", "On f(x) = x²"], [
      ["Too small", "Every step is tiny: it converges, but slowly.", "η = 0.02 needs over 100 steps"],
      ["About right", "Fast, smooth progress to the minimum.", "η = 0.1 – 0.5"],
      ["Large", "Overshoots and zig-zags, but still gets closer.", "η = 0.5 – 1.0"],
      ["Too big", "Each step lands higher than the last: the loss **explodes**.", "η > 1.0"],
    ]],
    ["h", "Fitting a line with gradient descent"],
    ["p", "Let's fit $\\hat y = w x + b$ to noisy points. The loss is the **mean squared error** $L = \\frac{1}{n}\\sum_i (\\hat y_i - y_i)^2$. Its gradients (by the chain rule) are:"],
    ["math", R`\frac{\partial L}{\partial w} = \frac{2}{n}\sum_i (\hat y_i - y_i)\,x_i \qquad \frac{\partial L}{\partial b} = \frac{2}{n}\sum_i (\hat y_i - y_i)`],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, size=50)
y = 2 * x + 1 + rng.normal(0, 1, size=50)    # the true line is y = 2x + 1, plus noise

w, b, lr = 0.0, 0.0, 0.01                    # start anywhere; learning rate 0.01
for step in range(1001):
    y_hat = w * x + b                        # predictions
    err = y_hat - y
    loss = np.mean(err ** 2)                 # mean squared error
    dw = 2 * np.mean(err * x)                # ∂L/∂w
    db = 2 * np.mean(err)                    # ∂L/∂b
    w -= lr * dw                             # step downhill
    b -= lr * db
    if step % 250 == 0:
        print(f"step {step:4d}  loss {loss:7.3f}  w {w:.3f}  b {b:.3f}")`, "step    0  loss 170.636  w 1.568  b 0.231\nstep  250  loss   0.998  w 2.073  b 0.616\nstep  500  loss   0.994  w 2.059  b 0.713\nstep  750  loss   0.993  w 2.054  b 0.743\nstep 1000  loss   0.993  w 2.053  b 0.753"],
    ["p", "The loss falls quickly at first and then flattens out as we approach the bottom. Plotting the loss per step (the **loss curve**) is the first thing to look at when training anything:"],
    ["code", R`
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, size=50)
y = 2 * x + 1 + rng.normal(0, 1, size=50)

def train(lr, steps=60):
    w = b = 0.0
    losses = []
    for _ in range(steps):
        err = w * x + b - y
        losses.append(np.mean(err ** 2))
        w -= lr * 2 * np.mean(err * x)
        b -= lr * 2 * np.mean(err)
    return losses

plt.figure(figsize=(6, 3.2))
for lr in [0.001, 0.005, 0.02]:
    plt.plot(train(lr), label=f"lr = {lr}")
plt.yscale("log")
plt.xlabel("step")
plt.ylabel("loss (log scale)")
plt.legend()
plt.title("Loss curves for three learning rates")
plt.show()
print("final loss with lr=0.02:", round(train(0.02)[-1], 3))`, "final loss with lr=0.02: 1.009"],
    ["tip", "Always plot the loss curve. Flat from the start → learning rate too small (or a bug). Jumping around or going up → too big. Smoothly down then flat → good."],
    ["h", "The same thing in PyTorch"],
    ["p", "Real projects let a library compute the gradients for you (**autograd**). This is the same line fit in PyTorch. PyTorch can't run inside the browser, so this one is read-only:"],
    ["code", R`
import torch

torch.manual_seed(0)
x = torch.rand(50) * 10
y = 2 * x + 1 + torch.randn(50)

w = torch.zeros(1, requires_grad=True)   # parameters PyTorch should track
b = torch.zeros(1, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=0.01)

for step in range(1001):
    loss = ((w * x + b - y) ** 2).mean()
    opt.zero_grad()          # clear old gradients
    loss.backward()          # autograd computes dloss/dw and dloss/db
    opt.step()               # w -= lr * w.grad, b -= lr * b.grad
    if step % 250 == 0:
        print(f"step {step:4d}  loss {loss.item():7.3f}  w {w.item():.3f}  b {b.item():.3f}")`, "step    0  loss 137.853  w 1.242  b 0.203\nstep  250  loss   1.203  w 2.096  b 0.641\nstep  500  loss   1.201  w 2.083  b 0.716\nstep  750  loss   1.201  w 2.080  b 0.735\nstep 1000  loss   1.201  w 2.079  b 0.740", "PyTorch", "local"],
    ["h", "Batch, stochastic and mini-batch"],
    ["table", ["Variant", "Gradient computed on", "Trade-off"], [
      ["Batch gradient descent", "All n examples every step", "Accurate steps, but slow when n is large"],
      ["Stochastic (SGD)", "One random example per step", "Very cheap, noisy steps; the noise can help escape bad spots"],
      ["Mini-batch", "A random batch of, say, 32–512 examples", "The standard choice: fast on GPUs, reasonably smooth"],
    ]],
    ["warn", "Gradient descent needs features on similar scales. If one feature is in the thousands and another is between 0 and 1, the loss surface becomes a long narrow valley and progress zig-zags. **Standardize** features first."],
    ["h", "Going faster: momentum"],
    ["p", "On a long, narrow valley plain gradient descent zig-zags across the valley and crawls along it. **Momentum** keeps a running average of past gradients and steps along that, so the side-to-side jumps cancel out and the steady downhill direction builds up speed. Compare the two paths below (more on this in [Optimizers](dl.html#optimizers))."],
    ["viz", "ml1_gd2d"],
  ],
},
{
  id: "polynomial", group: "Regression", title: "Polynomial Features",
  sub: "Fit curves with a linear model by feeding it powers of the inputs as extra columns.",
  tags: ["feature engineering"],
  blocks: [
    ["p", "A straight line cannot follow a curve. But there is a neat trick: keep the model linear in its *weights*, and change the *features*. If we add the columns $x^2$, $x^3$, … then \"linear regression\" can draw a parabola, a cubic, and so on, and everything you know about it (the normal equation, gradient descent, regularization) still works unchanged."],
    ["math", R`\hat y = w_0 + w_1 x + w_2 x^2 + \dots + w_d x^d`, "still linear in w, so still least squares"],
    ["p", "With several inputs, polynomial features also include the **interactions** (products like $x_1 x_2$), which let the effect of one feature depend on another. For two inputs and degree 2 the new columns are $1,\\ x_1,\\ x_2,\\ x_1^2,\\ x_1x_2,\\ x_2^2$."],
    ["h", "Slide the degree"],
    ["p", "Raise the degree and watch the fit. Degree 0 is a flat line (underfit). A middle degree follows the hidden curve. A very high degree threads through every noisy point and swings wildly between them (overfit): the training error keeps dropping while the test error climbs. This is the [bias–variance](ml.html#bias-variance) trade-off made visible."],
    ["viz", "ml1_poly"],
    ["h", "From scratch"],
    ["p", "Building the extra columns by hand is one line with `np.vander`. Then it is ordinary least squares:"],
    ["code", R`
import numpy as np

rng = np.random.default_rng(1)
x = np.sort(rng.uniform(-1, 1, 30))
y = 1 + 2 * x - 3 * x ** 2 + rng.normal(0, 0.2, 30)       # the truth is a parabola

def fit_poly(x, y, d):
    A = np.vander(x, d + 1, increasing=True)               # columns 1, x, x^2, ..., x^d
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return w

w = fit_poly(x, y, 2)
print("degree-2 weights:", w.round(2), "(true: 1, 2, -3)")

x_test = np.linspace(-1, 1, 200)
y_test = 1 + 2 * x_test - 3 * x_test ** 2
for d in [1, 2, 4, 10]:
    w = fit_poly(x, y, d)
    train = np.mean((np.vander(x, d + 1, increasing=True) @ w - y) ** 2)
    test = np.mean((np.vander(x_test, d + 1, increasing=True) @ w - y_test) ** 2)
    print(f"degree {d:2d}: train MSE {train:.3f}  error vs the true curve {test:.3f}")`, "degree-2 weights: [ 0.97  1.96 -2.93] (true: 1, 2, -3)\ndegree  1: train MSE 0.745  error vs the true curve 0.954\ndegree  2: train MSE 0.031  error vs the true curve 0.001\ndegree  4: train MSE 0.029  error vs the true curve 0.003\ndegree 10: train MSE 0.023  error vs the true curve 0.241"],
    ["h", "In scikit-learn"],
    ["p", "`PolynomialFeatures` builds the columns (including interactions for many inputs). Put it in a pipeline with a scaler, because $x^{10}$ is a huge number, and with the model. Choosing the degree is a job for [cross-validation](ml.html#cross-validation):"],
    ["code", R`
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(1)
X = rng.uniform(-1, 1, size=(60, 1))
y = np.sin(3 * X[:, 0]) + rng.normal(0, 0.2, 60)

for d in [1, 3, 5, 9, 15]:
    model = make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(), LinearRegression())
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error").mean()
    print(f"degree {d:2d}: cross-validated MSE {-score:.3f}")

print("columns for 2 inputs, degree 2:", PolynomialFeatures(2).fit(np.zeros((1, 2))).get_feature_names_out(["a", "b"]).tolist())`, "degree  1: cross-validated MSE 0.181\ndegree  3: cross-validated MSE 0.032\ndegree  5: cross-validated MSE 0.026\ndegree  9: cross-validated MSE 0.030\ndegree 15: cross-validated MSE 0.094\ncolumns for 2 inputs, degree 2: ['1', 'a', 'b', 'a^2', 'a b', 'b^2']"],
    ["h", "The number of columns explodes"],
    ["table", ["Inputs", "Degree 2", "Degree 3", "Degree 5"], [
      ["1", "2", "3", "5"],
      ["10", "65", "285", "3,002"],
      ["100", "5,150", "176,850", "96,560,645"],
    ]],
    ["p", "(Counts exclude the constant column; the formula is $\\binom{d+k}{k} - 1$ for $d$ inputs and degree $k$.) Beyond a few inputs, full polynomial expansion is expensive and overfits quickly. Trees, [kernels](ml.html#svm) and neural networks find curves without writing out every term."],
    ["tip", "Scale features before taking high powers, and combine polynomial features with [regularization](ml.html#regularization) (Ridge) so a high degree does not run wild."],
    ["danger", "Using a high degree and judging it on the training error. A degree-15 polynomial will always look best on training data and be terrible everywhere else."],
  ],
},
{
  id: "regularization", group: "Regression", title: "Regularization: L1, L2 & ElasticNet",
  sub: "Add a penalty for big weights so the model stays simple and does not memorise the noise.",
  tags: ["overfitting"],
  blocks: [
    ["p", "An overfit model usually has **huge weights**: it swings the curve hard to hit every point. **Regularization** fights that by adding a price tag to large weights. The model now has to trade off two goals: fit the data well, and keep the weights small. The knob that sets the exchange rate is the strength $\\alpha$ (sometimes called $\\lambda$)."],
    ["math", R`L(\mathbf w) = \underbrace{\frac{1}{n}\lVert X\mathbf w - \mathbf y\rVert^2}_{\text{fit the data}} \;+\; \alpha \cdot \underbrace{\Omega(\mathbf w)}_{\text{penalty}}`],
    ["table", ["Name", "Penalty $\\Omega(\\mathbf w)$", "Effect"], [
      ["**Ridge** (L2)", "$\\sum_j w_j^2$", "Shrinks all weights smoothly towards 0; none become exactly 0. Handles correlated features well."],
      ["**Lasso** (L1)", "$\\sum_j |w_j|$", "Shrinks and sets many weights **exactly to 0**, so it also does feature selection."],
      ["**ElasticNet**", "$\\rho\\sum_j |w_j| + \\tfrac{1-\\rho}{2}\\sum_j w_j^2$", "A mix of both: sparse like lasso, stable like ridge."],
    ]],
    ["note", "The intercept $b$ is normally *not* penalised: it only sets the overall level, not the wiggliness. scikit-learn handles this for you."],
    ["h", "Try it"],
    ["p", "A degree-9 polynomial fitted to noisy data. With almost no penalty ($\\alpha \\approx 0$) the curve is wild and the weights are enormous. As you increase $\\alpha$ the weights shrink and the curve calms down. Push it too far and it flattens into an underfit line. Switch to L1 and watch the weight bars hit exactly zero."],
    ["viz", "ml1_ridge"],
    ["h", "Why L1 gives exact zeros"],
    ["p", "Think of the penalty as a budget on the weights: stay inside the shaded region. The loss has elliptical contours around its unconstrained best point. The constrained solution is where the smallest ellipse first touches the region. The L2 region is a circle, which has no corners, so the touching point is usually somewhere with both weights non-zero. The L1 region is a diamond whose **corners sit on the axes**; the ellipse very often touches a corner, where one weight is exactly 0."],
    ["diagram", "ml1_l1l2", [], "Left: the L2 (ridge) budget is a disc, so the solution keeps both weights. Right: the L1 (lasso) budget is a diamond, and the solution lands on a corner, setting a weight to exactly zero."],
    ["h", "Ridge has a closed form"],
    ["p", "Ridge only adds a small term to the normal equation, and that term even makes the matrix easier to invert (it fixes the collinearity problem):"],
    ["math", R`\mathbf w = \big(X^\top X + \alpha I\big)^{-1} X^\top \mathbf y`],
    ["code", R`
import numpy as np

rng = np.random.default_rng(0)
n = 40
x1 = rng.normal(size=n)
x2 = x1 + rng.normal(0, 0.05, n)                  # almost a copy of x1: collinear
X = np.column_stack([x1, x2])
y = 3 * x1 + rng.normal(0, 0.5, n)                # only x1 really matters

def ridge(X, y, alpha):
    Xc, yc = X - X.mean(0), y - y.mean()          # centre so the intercept is not penalised
    d = X.shape[1]
    return np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ yc)

for a in [0.0, 1.0, 10.0, 100.0]:
    print(f"alpha {a:5.1f}: weights {ridge(X, y, a).round(2)}")`, "alpha   0.0: weights [2.58 0.51]\nalpha   1.0: weights [1.57 1.45]\nalpha  10.0: weights [1.29 1.28]\nalpha 100.0: weights [0.51 0.51]"],
    ["p", "Without regularization, two near-identical columns make the weights unstable: the data cannot say which one deserves the credit, so the split between them is arbitrary and swings with the noise (here 2.58 and 0.51). Ridge shares the credit evenly between the twins and keeps the weights stable, and a large alpha shrinks them together."],
    ["h", "In scikit-learn"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)                      # 10 features, 442 patients
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

models = {
    "ols": LinearRegression(),
    "ridge": Ridge(alpha=1.0),
    "lasso": Lasso(alpha=1.0),
    "elasticnet": ElasticNet(alpha=1.0, l1_ratio=0.5),
}
for name, m in models.items():
    pipe = make_pipeline(StandardScaler(), m).fit(X_tr, y_tr)    # always scale before penalising
    coef = pipe[-1].coef_
    print(f"{name:10s} test R^2 {pipe.score(X_te, y_te):.3f}   zero weights: {int(np.sum(coef == 0))}")`, "ols        test R^2 0.393   zero weights: 0\nridge      test R^2 0.392   zero weights: 0\nlasso      test R^2 0.389   zero weights: 2\nelasticnet test R^2 0.402   zero weights: 2"],
    ["h", "Choosing $\\alpha$"],
    ["p", "$\\alpha$ is a [hyperparameter](ml.html#cross-validation): too small and you overfit, too large and you underfit. Try values on a log scale and pick by validation score. `RidgeCV`, `LassoCV` and `ElasticNetCV` do this for you:"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(X, y)
print("best alpha:", round(float(model[-1].alpha_), 3))`, "best alpha: 1.0"],
    ["h", "Which one to use?"],
    ["table", ["Situation", "Pick"], [
      ["Many features, most of them useful, some correlated", "Ridge"],
      ["Many features, you believe only a few matter, want a short model", "Lasso"],
      ["Groups of correlated features and you still want sparsity", "ElasticNet"],
      ["Few features, lots of data", "Plain least squares may be enough"],
    ]],
    ["table", ["Hyperparameter", "Meaning"], [
      ["`alpha`", "Penalty strength. 0 is plain least squares; larger shrinks more. Search on a log scale."],
      ["`l1_ratio` (ElasticNet)", "The mix: 1 is pure lasso, 0 is pure ridge."],
    ]],
    ["danger", "Forgetting to **standardize** before regularizing. The penalty treats all weights alike, so a feature measured in thousands gets a tiny weight and escapes the penalty, while a 0-to-1 feature is punished hard. Scale first (in a pipeline)."],
    ["tip", "The same idea reappears everywhere: weight decay in neural networks is L2 regularization, and L1/L2 penalties appear in SVMs, logistic regression and boosting. Practice: [Ridge Regression in Closed Form](practice.html#p/ridge-closed-form)."],
  ],
},
{
  id: "logistic-regression", group: "Classification", title: "Logistic Regression",
  sub: "Despite the name, a classifier: a linear score squashed into a probability between 0 and 1.",
  tags: ["classification"],
  blocks: [
    ["p", "Linear regression predicts any number, but for classification we want a **probability** between 0 and 1: \"how likely is this email to be spam?\". Logistic regression takes the same weighted sum of features, $z = \\mathbf w^\\top \\mathbf x + b$, and passes it through the **sigmoid** function, which squeezes any number into the range (0, 1). Then you predict class 1 when the probability is above a threshold, usually 0.5."],
    ["math", R`p(y=1\mid \mathbf x) = \sigma(z) = \frac{1}{1 + e^{-z}} \qquad z = \mathbf w^\top\mathbf x + b`],
    ["p", "The sigmoid is 0.5 exactly when $z = 0$, so the **decision boundary** is the straight line (plane) $\\mathbf w^\\top \\mathbf x + b = 0$. The farther a point is from that line, the more confident the prediction. Move the weights in the visualizer to turn and shift the boundary, and watch the loss change."],
    ["viz", "ml1_logreg"],
    ["h", "Where the sigmoid comes from: log-odds"],
    ["p", "If $p$ is a probability, the **odds** are $p/(1-p)$ and the **log-odds** are $\\ln\\frac{p}{1-p}$. Logistic regression says the log-odds are a *linear* function of the features. Solve for $p$ and you get the sigmoid. So a weight $w_j$ means: one more unit of feature $j$ adds $w_j$ to the log-odds (multiplies the odds by $e^{w_j}$)."],
    ["math", R`\ln\frac{p}{1-p} = \mathbf w^\top \mathbf x + b`],
    ["h", "Training: minimise the log loss"],
    ["p", "We train by maximum likelihood (see [MLE](math.html#mle)): choose weights that make the observed labels as probable as possible. Taking the negative log of that likelihood gives the **binary cross-entropy** or log loss, averaged over examples:"],
    ["math", R`L(\mathbf w, b) = -\frac{1}{n}\sum_{i=1}^{n}\Big[\, y_i \ln p_i + (1-y_i)\ln(1-p_i) \Big]`],
    ["p", "Remarkably, its gradient has the same simple shape as in linear regression: prediction minus truth, times the feature."],
    ["math", R`\frac{\partial L}{\partial \mathbf w} = \frac{1}{n} X^\top(\mathbf p - \mathbf y) \qquad \frac{\partial L}{\partial b} = \frac{1}{n}\sum_i (p_i - y_i)`],
    ["p", "There is no closed-form solution, so we use [gradient descent](ml.html#gradient-descent). The loss is convex, so there is a single minimum and descent will find it."],
    ["h", "From scratch in NumPy"],
    ["code", R`
import numpy as np
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, centers=[[-1.5, -1.0], [1.5, 1.0]], cluster_std=1.3, random_state=0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w, b, lr = np.zeros(2), 0.0, 0.5
for step in range(501):
    p = sigmoid(X @ w + b)                                   # predicted probabilities
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) # binary cross-entropy
    w -= lr * X.T @ (p - y) / len(y)                         # dL/dw
    b -= lr * np.mean(p - y)                                 # dL/db
    if step % 100 == 0:
        print(f"step {step:3d}  log loss {loss:.4f}")

pred = (sigmoid(X @ w + b) > 0.5).astype(int)
print("weights:", w.round(3), " bias:", round(float(b), 3))
print("training accuracy:", round(float(np.mean(pred == y)), 3))
print("p(class 1) at (1, 1):", round(float(sigmoid(np.array([1.0, 1.0]) @ w + b)), 3))`, "step   0  log loss 0.6931\nstep 100  log loss 0.2417\nstep 200  log loss 0.2417\nstep 300  log loss 0.2417\nstep 400  log loss 0.2417\nstep 500  log loss 0.2417\nweights: [1.618 1.089]  bias: 0.054\ntraining accuracy: 0.91\np(class 1) at (1, 1): 0.941"],
    ["h", "The scikit-learn version"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

for C in [0.001, 0.1, 10.0]:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X_tr, y_tr)
    print(f"C = {C:<6}  train acc {model.score(X_tr, y_tr):.3f}  test acc {model.score(X_te, y_te):.3f}")

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
print("class probabilities for 2 test rows:", model.predict_proba(X_te[:2]).round(3).tolist())`, "C = 0.001   train acc 0.899  test acc 0.881\nC = 0.1     train acc 0.988  test acc 0.958\nC = 10.0    train acc 0.993  test acc 0.951\nclass probabilities for 2 test rows: [[0.004, 0.996], [1.0, 0.0]]"],
    ["h", "More than two classes"],
    ["p", "For $K$ classes, the model computes one score per class and turns them into probabilities with [softmax](math.html#entropy), trained with categorical cross-entropy. scikit-learn does this automatically when `y` has more than two values."],
    ["h", "Pros, cons and settings"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Fast, simple, hard to beat as a first baseline", "Only a **linear** boundary (unless you add features)"],
      ["Outputs probabilities, and weights you can interpret", "Sensitive to feature scale and to strongly correlated features"],
      ["Convex loss: one global optimum", "Probabilities may need calibrating if heavily regularized"],
    ]],
    ["table", ["Hyperparameter", "Meaning"], [
      ["`C`", "**Inverse** regularization strength (default 1). Small C = strong penalty = simpler model. Opposite of `alpha` in Ridge."],
      ["`penalty`", "`\"l2\"` (default) or `\"l1\"` (sparse weights; needs a solver such as `liblinear` or `saga`)."],
      ["`class_weight`", "`\"balanced\"` makes the rarer class count more (see [Imbalanced Data](ml.html#imbalanced))."],
      ["`max_iter`", "Raise it if you see a convergence warning (and scale your features)."],
    ]],
    ["danger", "Treating 0.5 as sacred. The threshold is a *choice* that depends on the cost of each kind of mistake (see [precision and recall](ml.html#classification-metrics)). Also: scikit-learn's logistic regression is regularized by default, so unscaled features and a default `C` can quietly hurt it."],
    ["tip", "Practice: [Logistic Regression with Gradient Descent](practice.html#p/logreg-gradient-descent)."],
  ],
},
{
  id: "knn", group: "Classification", title: "k-Nearest Neighbours",
  sub: "To classify a new point, look at the k most similar training points and let them vote.",
  tags: ["instance-based"],
  blocks: [
    ["p", "k-NN is the most \"human\" model there is. To judge a new house, you look at the few most similar houses nearby and see what they sold for. To label a new point, find its $k$ **nearest neighbours** in the training data and take a majority vote (classification) or an average (regression). There is no training step at all: the model *is* the stored data, which is why it is called a **lazy learner**."],
    ["p", "Drag the query point and change $k$ below. With $k=1$ the boundary is jagged and follows every noisy point (high variance). With a large $k$ it is smooth, and at the extreme it just predicts the majority class everywhere (high bias)."],
    ["viz", "ml1_knn"],
    ["h", "The algorithm"],
    ["steps", [
      "Choose $k$ and a **distance** (usually Euclidean: $d(\\mathbf a, \\mathbf b) = \\lVert \\mathbf a - \\mathbf b\\rVert$).",
      "For the new point, compute its distance to every training point.",
      "Take the $k$ smallest. Their labels vote; the most common label wins (ties: pick the closest or an odd $k$).",
    ]],
    ["h", "From scratch in NumPy"],
    ["code", R`
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

def knn_predict(X_train, y_train, X_new, k):
    preds = []
    for x in X_new:
        dist = np.sqrt(((X_train - x) ** 2).sum(axis=1))    # distance to every training point
        nearest = np.argsort(dist)[:k]                      # indices of the k closest
        votes = np.bincount(y_train[nearest])               # count the labels among them
        preds.append(np.argmax(votes))
    return np.array(preds)

for k in [1, 5, 15, 50]:
    acc = np.mean(knn_predict(X_tr, y_tr, X_te, k) == y_te)
    print(f"k = {k:2d}  test accuracy {acc:.3f}")`, "k =  1  test accuracy 0.889\nk =  5  test accuracy 0.922\nk = 15  test accuracy 0.922\nk = 50  test accuracy 0.900"],
    ["h", "The scikit-learn version"],
    ["code", R`
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

knn = KNeighborsClassifier(n_neighbors=15).fit(X_tr, y_tr)      # fit just stores the data
print("test accuracy:", round(knn.score(X_te, y_te), 3))
print("probabilities for 2 points:", knn.predict_proba(X_te[:2]).round(2).tolist())   # fraction of the k votes`, "test accuracy: 0.922\nprobabilities for 2 points: [[0.33, 0.67], [0.13, 0.87]]"],
    ["h", "Features must be on the same scale"],
    ["p", "k-NN is built on distances, so a feature with big numbers drowns out all the others. The wine dataset has features ranging from about 0.1 to over 1000. See the difference scaling makes:"],
    ["code", R`
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
raw = cross_val_score(KNeighborsClassifier(5), X, y, cv=5).mean()
scaled = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(5)), X, y, cv=5).mean()
print("feature ranges (max):", X.max(axis=0).round(1)[[0, 4, 12]], "-> one column reaches the thousands")
print("accuracy, unscaled:", round(float(raw), 3))
print("accuracy, scaled  :", round(float(scaled), 3))`, "feature ranges (max): [  14.8  162.  1680. ] -> one column reaches the thousands\naccuracy, unscaled: 0.691\naccuracy, scaled  : 0.949"],
    ["h", "Pros, cons and settings"],
    ["table", ["Strengths", "Weaknesses"], [
      ["No training, easy to understand, naturally multi-class", "**Slow at prediction**: compares with every stored point"],
      ["Boundaries of any shape", "Needs scaled features and meaningful distances"],
      ["Few assumptions", "Suffers in many dimensions (the curse of dimensionality) and with irrelevant features; memory-hungry"],
    ]],
    ["table", ["Hyperparameter", "Meaning"], [
      ["`n_neighbors` (k)", "Small = flexible/noisy, large = smooth. Pick by cross-validation; start near $\\sqrt{n}$ or with 5 to 15."],
      ["`weights`", "`\"distance\"` lets closer neighbours count more than farther ones."],
      ["`metric` / `p`", "Euclidean by default; Manhattan or cosine can suit other data."],
      ["`algorithm`", "`\"kd_tree\"` / `\"ball_tree\"` speed up search in low dimensions."],
    ]],
    ["warn", "In high dimensions every point is about equally far from every other, so \"nearest\" stops meaning \"similar\". Reduce dimensions first ([PCA](ml.html#pca)) or choose features carefully. For large data, use approximate nearest-neighbour search (the engine behind [vector search](llm.html#embeddings-search))."],
    ["tip", "Practice: [k-NN Prediction](practice.html#p/knn-predict)."],
  ],
},
{
  id: "naive-bayes", group: "Classification", title: "Naive Bayes",
  sub: "Apply Bayes' theorem with one bold shortcut: pretend the features are independent given the class.",
  tags: ["probabilistic"],
  blocks: [
    ["p", "Naive Bayes asks: given what I see, which class is most probable? It uses [Bayes' theorem](math.html#bayes) to flip the question around. It is easy to estimate \"how likely is this feature value *if* the email is spam\" by counting, so we compute that for every class and combine it with how common each class is:"],
    ["math", R`P(c \mid \mathbf x) \;\propto\; P(c)\; P(\mathbf x \mid c) \;=\; P(c)\prod_{j=1}^{d} P(x_j \mid c)`, "posterior ∝ prior × likelihood; the product is the \"naive\" independence assumption"],
    ["p", "The **naive** part is multiplying the per-feature likelihoods as if the features did not affect each other once the class is known. That is rarely exactly true (the words \"free\" and \"winner\" go together), but the classifier only needs the right class to win, and it often does. In one dimension you can see the whole story: the curves are the likelihoods, scaled by the prior, and the bigger one wins."],
    ["viz", "ml1_gnb"],
    ["h", "Variants"],
    ["table", ["Variant", "For features that are…", "Models $P(x_j\\mid c)$ as"], [
      ["**GaussianNB**", "Continuous numbers", "A normal curve with a mean and variance per class"],
      ["**MultinomialNB**", "Counts (word counts)", "Word frequencies per class; the text classic"],
      ["**BernoulliNB**", "Yes / no (word present or not)", "A coin flip per feature per class"],
    ]],
    ["h", "From scratch: Gaussian Naive Bayes"],
    ["p", "\"Training\" is just computing, for each class, its prior and each feature's mean and variance. To predict, add log-probabilities (sums are safer than long products of tiny numbers, which would underflow to 0):"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

classes = np.unique(y_tr)
prior = np.array([np.mean(y_tr == c) for c in classes])                 # P(c)
mu = np.array([X_tr[y_tr == c].mean(axis=0) for c in classes])          # mean per class and feature
var = np.array([X_tr[y_tr == c].var(axis=0) for c in classes]) + 1e-9   # variance per class and feature

def predict(X_new):
    # log P(c) + sum_j log N(x_j | mu, var)
    logp = np.log(prior) + np.stack([
        (-0.5 * np.log(2 * np.pi * var[i]) - (X_new - mu[i]) ** 2 / (2 * var[i])).sum(axis=1)
        for i in range(len(classes))], axis=1)
    return classes[np.argmax(logp, axis=1)]

print("prior:", prior.round(3).tolist())
print("class means (rows = classes):")
print(mu.round(2))
print("test accuracy:", round(float(np.mean(predict(X_te) == y_te)), 3))`, "prior: [0.333, 0.333, 0.333]\nclass means (rows = classes):\n[[5.03 3.42 1.43 0.25]\n [5.89 2.73 4.23 1.31]\n [6.68 2.97 5.61 2.04]]\ntest accuracy: 0.978"],
    ["code", R`
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
nb = GaussianNB().fit(X_tr, y_tr)
print("sklearn test accuracy:", round(nb.score(X_te, y_te), 3))
print("probabilities, first test row:", nb.predict_proba(X_te[:1]).round(3).tolist())`, "sklearn test accuracy: 0.978\nprobabilities, first test row: [[0.0, 0.0, 1.0]]"],
    ["h", "Text classification with word counts"],
    ["p", "Naive Bayes shines on text. Count how often each word appears per class. A word never seen in a class would give probability 0 and wipe out the whole product, so we add a small count to everything: **Laplace smoothing** (`alpha=1`)."],
    ["code", R`
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = ["win free money now", "free prize claim now", "lowest price free offer",
         "meeting at noon tomorrow", "lunch with the team", "project report attached"]
labels = [1, 1, 1, 0, 0, 0]                     # 1 = spam, 0 = normal

vec = CountVectorizer()
X = vec.fit_transform(texts)                    # word-count matrix
nb = MultinomialNB(alpha=1.0).fit(X, labels)    # alpha=1: Laplace smoothing

for msg in ["free money prize", "team meeting tomorrow", "free lunch meeting"]:
    p = nb.predict_proba(vec.transform([msg]))[0, 1]
    print(f"{msg!r:28} P(spam) = {p:.3f}")`, "'free money prize'           P(spam) = 0.936\n'team meeting tomorrow'      P(spam) = 0.102\n'free lunch meeting'         P(spam) = 0.476"],
    ["h", "Pros, cons and settings"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Extremely fast to train and predict, even with huge numbers of features", "The independence assumption is wrong for correlated features"],
      ["Works with very little data; great text baseline", "Probabilities are often over-confident (good for ranking, not for calibrated odds)"],
      ["Handles many classes naturally; nothing to tune but smoothing", "A decision-tree or logistic regression usually wins with enough data"],
    ]],
    ["table", ["Hyperparameter", "Meaning"], [
      ["`alpha` (Multinomial, Bernoulli)", "Smoothing amount. 1 = Laplace; smaller trusts the counts more."],
      ["`var_smoothing` (Gaussian)", "A tiny value added to variances for numerical stability."],
      ["`priors`", "Override the class priors if the training mix is not the real-world mix."],
    ]],
    ["danger", "Multiplying raw probabilities: a hundred numbers of 0.01 multiply to $10^{-200}$ and underflow. Always add **log**-probabilities. And never leave a count at exactly zero: smooth it."],
  ],
},
{
  id: "svm", group: "Classification", title: "Support Vector Machines & Kernels",
  sub: "Find the boundary with the widest possible empty street between the classes, and bend it with kernels.",
  tags: ["margin", "kernel"],
  blocks: [
    ["p", "Many lines can separate two classes. Which is best? The SVM picks the one that stays **as far as possible from both classes**: the line in the middle of the widest empty street. The street's width is the **margin**. A wide margin means the boundary is not balanced on a knife-edge, and it tends to generalise well."],
    ["p", "Only the points touching the edge of the street matter. These are the **support vectors**; move or delete any other point and the boundary does not change. Experiment below: a small $C$ allows a wide street with a few violations, a large $C$ insists on classifying every point correctly even if the street gets narrow, and an outlier shows the difference."],
    ["viz", "ml1_svm"],
    ["h", "The maths"],
    ["p", "With labels $y_i \\in \\{-1, +1\\}$ and boundary $\\mathbf w^\\top \\mathbf x + b = 0$, a point is on the right side with room to spare when $y_i(\\mathbf w^\\top\\mathbf x_i + b) \\ge 1$. The street between the two edges has width $2/\\lVert\\mathbf w\\rVert$, so maximising the margin means *minimising* $\\lVert \\mathbf w\\rVert$. Real data overlaps, so we allow violations with a price $C$:"],
    ["math", R`\min_{\mathbf w, b}\;\; \tfrac12\lVert\mathbf w\rVert^2 \;+\; C\sum_{i=1}^{n}\max\!\big(0,\; 1 - y_i(\mathbf w^\top \mathbf x_i + b)\big)`, "wide street (first term) + penalty for points inside or across it (hinge loss)"],
    ["p", "The second term is the [hinge loss](ml.html#loss-functions): zero for points safely outside the street, growing linearly for points that are inside it or wrong. $C$ is the knob between a wide street (small $C$) and few violations (large $C$)."],
    ["h", "From scratch: a linear SVM by sub-gradient descent"],
    ["code", R`
import numpy as np
from sklearn.datasets import make_blobs

X, y01 = make_blobs(n_samples=100, centers=[[-2, -2], [2, 2]], cluster_std=1.2, random_state=0)
y = 2 * y01 - 1                                       # labels in {-1, +1}

w, b, lr, C = np.zeros(2), 0.0, 0.01, 1.0
for epoch in range(300):
    margin = y * (X @ w + b)
    viol = margin < 1                                 # points inside or past the street
    # gradient of 0.5*|w|^2 + C * sum(hinge)
    grad_w = w - C * (y[viol, None] * X[viol]).sum(axis=0)
    grad_b = -C * y[viol].sum()
    w -= lr * grad_w
    b -= lr * grad_b

margin = y * (X @ w + b)
print("w:", w.round(3), " b:", round(float(b), 3))
print("accuracy:", round(float(np.mean(np.sign(X @ w + b) == y)), 3))
print("street width 2/|w| =", round(float(2 / np.linalg.norm(w)), 3), " support-ish points (margin<=1):", int(np.sum(margin <= 1.0001)))`, "w: [1.635 0.621]  b: -0.59\naccuracy: 1.0\nstreet width 2/|w| = 1.144  support-ish points (margin<=1): 4"],
    ["h", "The kernel trick: curves from straight lines"],
    ["p", "What if no straight line works? The classes below sit on a number line, with one class in the middle: impossible to split with a single cut. But map each point $x$ to the pair $(x, x^2)$ and they become separable with a horizontal line. A **kernel** gets this benefit without ever building the new coordinates."],
    ["diagram", "ml1_kernel", [], "Left: no single threshold separates the classes in 1-D. Right: after adding x² as a second feature a straight line does, and that line is a curved boundary back in the original space."],
    ["p", "The SVM only ever needs **dot products** between points. A kernel $k(\\mathbf x, \\mathbf z)$ returns the dot product in a (possibly infinite-dimensional) feature space using only the original inputs. The popular choice is the **RBF (Gaussian) kernel**, a similarity that is 1 for identical points and falls to 0 with distance:"],
    ["math", R`k(\mathbf x, \mathbf z) = \exp\!\big(-\gamma\,\lVert \mathbf x - \mathbf z\rVert^2\big)`, "γ (gamma) sets how far a point's influence reaches: large γ = short reach = wiggly boundary"],
    ["table", ["Kernel", "Boundary", "Use when"], [
      ["`linear`", "Straight line / plane", "Many features (text), roughly linear classes"],
      ["`rbf`", "Smooth, flexible curves", "The default for general problems"],
      ["`poly`", "Polynomial curves", "Occasionally; known interactions"],
    ]],
    ["h", "The scikit-learn version"],
    ["code", R`
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

for kernel in ["linear", "rbf"]:
    svm = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0)).fit(X_tr, y_tr)
    print(f"{kernel:6s} test accuracy {svm.score(X_te, y_te):.3f}   support vectors: {svm[-1].n_support_.sum()}")

print("rbf, effect of C and gamma (test accuracy):")
for C in [0.1, 1, 100]:
    row = []
    for gamma in [0.1, 1, 10, 100]:
        m = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma)).fit(X_tr, y_tr)
        row.append(round(m.score(X_te, y_te), 3))
    print(f"  C = {C:<5} gamma 0.1/1/10/100 ->", row)`, "linear test accuracy 0.856   support vectors: 76\nrbf    test accuracy 0.956   support vectors: 66\nrbf, effect of C and gamma (test accuracy):\n  C = 0.1   gamma 0.1/1/10/100 -> [0.844, 0.933, 0.778, 0.411]\n  C = 1     gamma 0.1/1/10/100 -> [0.844, 0.956, 0.989, 0.944]\n  C = 100   gamma 0.1/1/10/100 -> [0.978, 0.989, 0.956, 0.956]"],
    ["h", "Pros, cons and settings"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Strong on small and medium data, especially with many features", "Training is slow on big data (roughly $O(n^2)$ to $O(n^3)$)"],
      ["Kernels give non-linear boundaries cheaply", "Needs scaled features and careful tuning of $C$ and $\\gamma$"],
      ["Depends only on support vectors: robust, memory-light model", "No native probabilities (`probability=True` adds a slow extra step)"],
    ]],
    ["table", ["Hyperparameter", "Meaning"], [
      ["`C`", "Penalty for violations. Small = wide street, simpler. Large = fit every point, risk of overfitting."],
      ["`gamma`", "RBF reach. Small = smooth, large = each point is its own island (overfit). `\"scale\"` is a sensible default."],
      ["`kernel`", "`\"rbf\"` (default), `\"linear\"`, `\"poly\"`."],
    ]],
    ["danger", "Not scaling features. The margin and the RBF kernel are both distance-based, so one large-valued feature dominates. Also: tune $C$ and $\\gamma$ together on a log grid with [cross-validation](ml.html#cross-validation); the best pair is rarely the default. For very large datasets use `LinearSVC` or `SGDClassifier`."],
  ],
},
{
  id: "decision-trees", group: "Classification", title: "Decision Trees",
  sub: "A flowchart of yes/no questions, learned from the data by always asking the most useful question next.",
  tags: ["trees"],
  blocks: [
    ["p", "A decision tree plays twenty questions. To classify a flower it might ask \"is the petal length under 2.5 cm?\", then \"is the width under 1.8?\", and follow the answers down to a leaf that names a class. The questions are simple threshold tests on one feature, so the model is easy to read, needs no feature scaling, and handles mixed kinds of features naturally."],
    ["diagram", "ml1_treechart", [], "A small tree: each internal node asks a question, each leaf makes a prediction."],
    ["h", "How the tree chooses its questions"],
    ["p", "The tree is grown greedily from the top. At each node it tries every feature and every threshold, and keeps the split that makes the two children **as pure as possible** (mostly one class). Purity is measured by **Gini impurity**, the chance that two randomly picked points in a node have different classes:"],
    ["math", R`G = 1 - \sum_{k} p_k^2`, "p_k = share of the node's points in class k. G = 0: pure. G = 0.5: a 50/50 mix of two classes."],
    ["p", "A split is scored by the weighted average impurity of its children; the improvement over the parent is the **information gain** (the alternative measure, entropy, gives very similar trees). Step through the growth below: each click adds the best split and shades the regions by their majority class."],
    ["viz", "ml1_tree"],
    ["h", "Worked example: one split by hand"],
    ["p", "A node holds 10 points: 5 of class A and 5 of class B. Its Gini is $1 - (0.5^2 + 0.5^2) = 0.5$. A candidate split sends 4 A and 1 B left, and 1 A and 4 B right. Each child has Gini $1 - (0.8^2 + 0.2^2) = 0.32$, and the weighted average is $0.5\\cdot0.32 + 0.5\\cdot0.32 = 0.32$. The gain is $0.5 - 0.32 = 0.18$. The tree tries every possible split and picks the one with the largest gain."],
    ["h", "From scratch: the best split"],
    ["code", R`
import numpy as np

def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)                  # class proportions
    return 1 - np.sum(p ** 2)

def best_split(x, y):
    """Best threshold on one feature: returns (threshold, weighted Gini after the split)."""
    best_t, best_g = None, gini(y)
    xs = np.sort(np.unique(x))
    for t in (xs[:-1] + xs[1:]) / 2:             # candidate thresholds between neighbouring values
        left, right = y[x <= t], y[x > t]
        g = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
        if g < best_g:
            best_t, best_g = t, g
    return best_t, best_g

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
print("Gini of the whole node:", round(float(gini(y)), 3))
t, g = best_split(x, y)
print("best threshold:", t, " weighted Gini after split:", round(float(g), 3))
print("information gain:", round(float(gini(y) - g), 3))`, "Gini of the whole node: 0.5\nbest threshold: 3.5  weighted Gini after split: 0.2\ninformation gain: 0.3"],
    ["p", "A full tree just applies this recursively: split, then split each child on its own data, until a stopping rule says to make a leaf (maximum depth, too few points, no improvement)."],
    ["h", "The scikit-learn version"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris()
X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.3, stratify=iris.target, random_state=0)

tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
print(export_text(tree, feature_names=iris.feature_names))
print("test accuracy (depth 2):", round(tree.score(X_te, y_te), 3))

for depth in [1, 2, 3, None]:                    # None = grow until every leaf is pure
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
    print(f"max_depth {str(depth):4s} train {t.score(X_tr, y_tr):.3f}  test {t.score(X_te, y_te):.3f}  leaves {t.get_n_leaves()}")`, "|--- petal width (cm) <= 0.80\n|   |--- class: 0\n|--- petal width (cm) >  0.80\n|   |--- petal width (cm) <= 1.75\n|   |   |--- class: 1\n|   |--- petal width (cm) >  1.75\n|   |   |--- class: 2\n\ntest accuracy (depth 2): 0.956\nmax_depth 1    train 0.667  test 0.667  leaves 2\nmax_depth 2    train 0.962  test 0.956  leaves 3\nmax_depth 3    train 0.962  test 0.956  leaves 5\nmax_depth None train 1.000  test 0.978  leaves 8"],
    ["p", "An unlimited tree always reaches 100% training accuracy because it can keep splitting until every training point has its own leaf. That is memorisation: a single deep tree has **high variance**. This is exactly the weakness that [random forests](ml.html#random-forests) and [boosting](ml.html#boosting) fix."],
    ["h", "Regression trees"],
    ["p", "For a number target, a leaf predicts the **average** of its training targets, and a split is chosen to reduce the squared error (variance) instead of Gini. The result is a staircase-shaped function. `DecisionTreeRegressor` does this."],
    ["h", "Pros, cons and settings"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Easy to explain and to draw", "A single deep tree overfits badly (high variance)"],
      ["No scaling, handles mixed features and non-linear effects", "Small changes in data can give a very different tree"],
      ["Captures interactions automatically", "Boundaries are axis-aligned steps; poor at diagonal lines and at extrapolating"],
    ]],
    ["table", ["Hyperparameter", "Meaning"], [
      ["`max_depth`", "Maximum number of questions in a row. The main brake on overfitting."],
      ["`min_samples_leaf`", "Smallest number of points allowed in a leaf. Bigger = smoother."],
      ["`min_samples_split`", "Fewest points a node needs before it may be split."],
      ["`criterion`", "`\"gini\"` (default), `\"entropy\"`, or `\"squared_error\"` for regression."],
      ["`ccp_alpha`", "Post-pruning strength: removes branches that do not earn their keep."],
    ]],
    ["danger", "Judging a tree by training accuracy. Limit `max_depth` or `min_samples_leaf`, and pick them by validation score. Also remember that the feature-importance numbers of a single tree are unstable."],
    ["tip", "Practice: [Best Gini Split for a Decision Stump](practice.html#p/gini-stump)."],
  ],
},
{
  id: "random-forests", group: "Classification", title: "Bagging & Random Forests",
  sub: "Train many different trees on shuffled copies of the data and let them vote: the noise cancels out.",
  tags: ["ensemble"],
  blocks: [
    ["p", "One deep tree overfits, but its mistakes are fairly random: different training samples give different trees that err in different places. If you average many such trees, the random errors cancel and the signal remains. It is the wisdom of crowds: many noisy, independent guesses average to something accurate."],
    ["diagram", "ml1_ensembles", [], "Two ways to combine models. Bagging trains models in parallel on resampled data and averages them (reduces variance). Boosting trains them one after another, each fixing the last one's mistakes (reduces bias)."],
    ["h", "Bagging"],
    ["p", "**Bagging** (bootstrap aggregating) makes the different datasets by **bootstrapping**: draw $n$ rows *with replacement* from the $n$ training rows. Some rows appear twice or more, and about 37% are left out (since $(1-\\frac1n)^n \\to e^{-1} \\approx 0.37$). Train one tree per bootstrap sample, then average (regression) or vote (classification). Use the slider to add trees and watch the jagged single-tree boundary smooth out and the error drop."],
    ["viz", "ml1_bagging"],
    ["h", "Random forests"],
    ["p", "Bagged trees are still alike when one feature is dominant: they all split on it first. A **random forest** adds a second kind of randomness: at every split each tree may only consider a **random subset of the features** (about $\\sqrt{d}$ for classification). This forces the trees to be different from each other, so averaging helps more."],
    ["table", ["", "Single tree", "Bagged trees", "Random forest"], [
      ["Rows used", "All", "Bootstrap sample", "Bootstrap sample"],
      ["Features per split", "All", "All", "Random subset"],
      ["Variance", "High", "Lower", "Lowest"],
      ["Bias", "Low", "Same as one tree", "About the same"],
    ]],
    ["h", "From scratch: bagging with scikit-learn trees"],
    ["code", R`
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=400, noise=0.35, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)

rng = np.random.default_rng(0)
trees = []
for _ in range(50):
    idx = rng.integers(0, len(X_tr), len(X_tr))              # bootstrap: sample rows WITH replacement
    trees.append(DecisionTreeClassifier(random_state=0).fit(X_tr[idx], y_tr[idx]))

single = trees[0].score(X_te, y_te)
votes = np.mean([t.predict(X_te) for t in trees], axis=0)    # average the 50 votes
bagged = np.mean((votes > 0.5).astype(int) == y_te)
print("one deep tree     :", round(float(single), 3))
print("50 bagged trees   :", round(float(bagged), 3))
print("share of rows left out of one bootstrap sample:", round(float(1 - len(np.unique(idx)) / len(X_tr)), 3))`, "one deep tree     : 0.825\n50 bagged trees   : 0.875\nshare of rows left out of one bootstrap sample: 0.338"],
    ["h", "The scikit-learn version"],
    ["code", R`
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.25, stratify=data.target, random_state=0)

rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
print("test accuracy      :", round(rf.score(X_te, y_te), 3))
print("out-of-bag accuracy:", round(float(rf.oob_score_), 3), "(free validation, no split needed)")

top = np.argsort(rf.feature_importances_)[::-1][:3]
for i in top:
    print(f"  {data.feature_names[i]:22s} importance {rf.feature_importances_[i]:.3f}")`, "test accuracy      : 0.944\nout-of-bag accuracy: 0.965 (free validation, no split needed)\n  worst perimeter        importance 0.144\n  worst radius           importance 0.136\n  worst concave points   importance 0.131"],
    ["p", "The **out-of-bag (OOB)** trick: each tree never saw about 37% of the rows, so we can score every row using only the trees that did not train on it. That gives a validation estimate for free."],
    ["h", "Pros, cons and settings"],
    ["table", ["Strengths", "Weaknesses"], [
      ["Excellent accuracy with almost no tuning", "Less interpretable than one tree; big models are slow and large"],
      ["Resistant to overfitting as you add trees", "Cannot extrapolate beyond the range seen in training"],
      ["Handles mixed features; OOB score; importances", "Usually a little behind well-tuned [boosting](ml.html#boosting)"],
    ]],
    ["table", ["Hyperparameter", "Meaning"], [
      ["`n_estimators`", "Number of trees. More is never worse, only slower; 100 to 500 is typical."],
      ["`max_features`", "Features tried per split (`\"sqrt\"` by default for classification). Smaller = more diverse trees."],
      ["`max_depth`, `min_samples_leaf`", "Per-tree complexity. Defaults grow deep trees, which is fine for forests."],
      ["`class_weight`, `oob_score`", "Balance classes; compute the out-of-bag estimate."],
    ]],
    ["warn", "Impurity-based `feature_importances_` are biased toward features with many distinct values. For a more trustworthy ranking, use permutation importance (`sklearn.inspection.permutation_importance`) on held-out data."],
    ["tip", "A random forest is the best first model on tabular data: strong, fast to try, and hard to misuse. Set `n_estimators=300` and see how far you get before reaching for anything fancier."],
  ],
},
{
  id: "boosting", group: "Classification", title: "Gradient Boosting & XGBoost",
  sub: "Build trees one after another, each one correcting the mistakes the ensemble still makes.",
  tags: ["ensemble", "tabular"],
  blocks: [
    ["p", "A random forest grows many deep trees independently and averages them. **Boosting** does the opposite: it builds many *small, weak* trees **in sequence**. The first tree makes a rough prediction. The second is trained to predict what the first got *wrong*, the third fixes what is still wrong, and so on. Each tree is added with a small weight (the learning rate), so the ensemble creeps towards the target."],
    ["p", "Watch it fit a curve. The thin step tree on its own is hopeless, but after a few rounds the sum of steps follows the curve. Below the main picture, the residuals (what is left to explain) shrink every round."],
    ["viz", "ml1_boost"],
    ["h", "The algorithm (squared error)"],
    ["steps", [
      "Start with a constant prediction: the mean of $y$.",
      "Compute the **residuals** $r_i = y_i - \\hat y_i$ (how far each prediction is from the truth).",
      "Fit a small tree to predict those residuals from the features.",
      "Update: $\\hat y \\leftarrow \\hat y + \\eta \\cdot \\text{tree}(x)$ where $\\eta$ is the learning rate (for example 0.1).",
      "Repeat steps 2 to 4 for a fixed number of rounds.",
    ]],
    ["p", "Why \"gradient\" boosting? For squared error the residual is exactly the negative gradient of the loss with respect to the prediction. For other losses (log loss, absolute error) each tree is fit to that loss's negative gradient, so the same recipe works for any differentiable loss. It is [gradient descent](ml.html#gradient-descent) in the space of functions: each tree is one step downhill."],
    ["math", R`\hat y^{(m)}(x) = \hat y^{(m-1)}(x) + \eta\, h_m(x), \qquad h_m \approx -\frac{\partial L}{\partial \hat y}\Big|_{\hat y^{(m-1)}}`],
    ["h", "From scratch in NumPy (with stumps)"],
    ["code", R`
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(0)
X = rng.uniform(0, 6, size=(200, 1))
y = np.sin(X[:, 0]) + rng.normal(0, 0.2, 200)

eta, rounds = 0.1, 100
pred = np.full(len(y), y.mean())                  # start from the mean
trees = []
for m in range(rounds):
    resid = y - pred                              # what is still wrong (negative gradient of MSE)
    stump = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, resid)
    pred += eta * stump.predict(X)                # add a small correction
    trees.append(stump)
    if m in (0, 9, 49, 99):
        print(f"after {m + 1:3d} trees: training MSE {np.mean((y - pred) ** 2):.4f}")

def predict(X_new):
    return y.mean() + eta * sum(t.predict(X_new) for t in trees)
print("prediction at x = 1.5708 (true sin = 1):", round(float(predict(np.array([[1.5708]]))[0]), 3))`, "after   1 trees: training MSE 0.4458\nafter  10 trees: training MSE 0.1216\nafter  50 trees: training MSE 0.0317\nafter 100 trees: training MSE 0.0268\nprediction at x = 1.5708 (true sin = 1): 0.954"],
    ["h", "The scikit-learn version"],
    ["p", "`GradientBoostingClassifier` follows the algorithm above. `HistGradientBoostingClassifier` bins features into histograms first, which makes it far faster on big data, handles missing values itself, and is the one to use in practice."],
    ["code", R`
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.25, stratify=data.target, random_state=0)

gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=2, random_state=0).fit(X_tr, y_tr)
print("GradientBoosting     test accuracy:", round(gb.score(X_te, y_te), 3))

hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=150, random_state=0).fit(X_tr, y_tr)
print("HistGradientBoosting test accuracy:", round(hgb.score(X_te, y_te), 3))

# staged predictions show how the score evolves as trees are added
acc = [round(float((p == y_te).mean()), 3) for p in gb.staged_predict(X_te)]
print("accuracy after 1, 10, 50, 150 trees:", [acc[0], acc[9], acc[49], acc[149]])`, "GradientBoosting     test accuracy: 0.93\nHistGradientBoosting test accuracy: 0.958\naccuracy after 1, 10, 50, 150 trees: [0.629, 0.923, 0.93, 0.93]"],
    ["h", "XGBoost, LightGBM and CatBoost"],
    ["p", "**XGBoost** is a fast, carefully engineered gradient-boosting library. Compared with the plain recipe it adds regularization on the leaf values and the number of leaves, uses second-order gradient information, handles missing values, and runs in parallel. LightGBM and CatBoost are close relatives with different speed tricks (and, for CatBoost, built-in categorical handling). On tabular data these libraries are, as of 2026, still the usual winners over neural networks. XGBoost is not part of scikit-learn and does not run in the browser, so this example runs on your computer:"],
    ["code", R`
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.25, stratify=data.target, random_state=0)

model = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=0,
)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
print("XGBoost test accuracy:", round(float(model.score(X_te, y_te)), 3))`, "XGBoost test accuracy: 0.944", "XGBoost", "local"],
    ["h", "Random forest or boosting?"],
    ["table", ["", "Random forest", "Gradient boosting"], [
      ["Trees", "Deep, independent", "Shallow, sequential"],
      ["Reduces", "Variance", "Bias (and some variance, with a small learning rate)"],
      ["Tuning needed", "Little", "More; accuracy is sensitive to the settings"],
      ["Overfitting risk", "Low", "Higher: needs a validation set and early stopping"],
      ["Best accuracy", "Very good", "Often the best on tabular data"],
    ]],
    ["table", ["Hyperparameter", "Meaning"], [
      ["`learning_rate` (η)", "Size of each correction. Smaller is safer but needs more trees. Typical 0.03 to 0.2."],
      ["`n_estimators`", "Number of trees (rounds). Trade off against the learning rate; choose with early stopping."],
      ["`max_depth`", "Depth of each tree. Keep it small (2 to 6): boosting wants weak learners."],
      ["`subsample`, `colsample_bytree`", "Fraction of rows / features per tree: adds randomness, reduces overfitting."],
      ["`reg_lambda`, `reg_alpha`, `min_child_weight`", "XGBoost regularization knobs on the leaf values and on tiny leaves."],
    ]],
    ["danger", "Training for too many rounds. Unlike a forest, boosting keeps fitting the training data and will start to memorise noise. Hold out a validation set and use **early stopping** (`early_stopping_rounds` in XGBoost, `n_iter_no_change` in scikit-learn). Also never tune on the test set."],
    ["tip", "Recipe that works: a low learning rate (0.05), a few hundred trees with early stopping, depth 3 to 6, and `subsample` about 0.8. Then tune depth and the regularization with [cross-validation](ml.html#cross-validation)."],
  ],
}
);
