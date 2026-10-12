# Authoring guide: AI Study Hub content

This is the shared brief for everyone writing content for this site (several writers work **at the same time**, each on
their own files). Read it fully before writing anything.

Site folder: `C:\Users\gkviv\OneDrive\Desktop\MLDLNLP` (call it ROOT). It is an offline study website for ML, Deep Learning,
Transformers, LLMs and Agentic AI, opened by double-clicking `index.html` (file://). The engine is finished; the job now is
**content**: doc sections, visualizers and practice problems.

Look at these finished examples first and match their depth and tone:
- `data/ml.js` → section `gradient-descent` (the reference section: brief, maths, a slider visualizer, runnable NumPy with
  inline outputs, a matplotlib plot, the same thing in PyTorch, tables, tip / warn callouts).
- `data/briefs-ml.js` (the plain-English brief for that section).
- `data/prac/01-numpy-linalg.js` (two complete practice problems).
- `assets/viz.js` (the visualizer engine, the `Plot` helper and the `gd1d` visualizer).

---

## 1. Who reads this and what they like

- The learner studies for **interviews and self-study**. Write **plain, friendly English** for a beginner, but go deep:
  every section should leave them able to explain the idea, write it from scratch in NumPy, and use the library version.
- Short sentences. Concrete examples and numbers before abstractions. Intuition → picture → maths → code → pitfalls.
- **No emoji anywhere.** Plain arrows (→) and ✓ / ✗ inside text are fine.
- **No "Interview questions" sections or boxes** (the learner explicitly does not want them). Don't add Q&A quizzes either.
- Both light and dark mode must look right, so in SVG use CSS classes / CSS variables, never hard-coded black or white.
- Facts must be correct. For fast-moving topics (model names, frameworks, MCP, prices, context sizes) say "as of 2026"
  and keep claims general unless you are sure. Never invent numbers, benchmark scores, citations or program outputs.

## 2. File ownership: only touch your own files

Each writer owns a fixed set of files (listed in their assignment). **Do not edit any other file**, in particular:
`assets/common.js`, `assets/viz.js`, `assets/style.css`, `assets/py.js`, `assets/site.js`, `assets/glossary.js`,
`assets/practice.js`, any page `.html`, `tools/*`, other writers' data files. If you think the engine needs a change,
describe it in your final report instead.

Your files are:
- `data/<yourfile>.js`: your doc sections (e.g. `data/ml-2.js`).
- `data/briefs-<yourfile>.js`: the briefs for your sections (e.g. `data/briefs-ml-2.js`).
- `assets/viz/<prefix>.js`: your visualizers, diagrams and glossary terms.
- `data/prac/<NN>-<name>.js`: your practice problems.

Never run `node tools/build.js` or `node tools/lite.js` (the lead runs them once at the end, and build.js deletes and
rebuilds shared folders). Everything else in section 7 is safe to run at any time.

## 3. Doc sections (`data/<file>.js`)

```js
/* Machine Learning, part 2: unsupervised learning, evaluation, practical
   Outputs written as "?" are filled in by running the code: node tools/fill.js data/ml-2.js */
ML.push(
{
  id: "kmeans", group: "Unsupervised", title: "k-Means Clustering",
  sub: "One-line lead under the title.",
  tags: ["clustering"],                          // optional, 0–2 short tags
  blocks: [ ...blocks ],
},
{ ... }
);
```
The array name is the page's list (`MATH`, `PYML`, `ML`, `DL`, `TF`, `LLM`, `AG`). **Use exactly the ids, titles and groups
from the outline in section 10** (you may add a section if something important is missing; keep the listed ones).
Sections appear in file order; consecutive sections with the same `group` form a sidebar group. Files of a split page load
in name order (`ml-1.js` then `ml-2.js`).

**Data files must be pure data.** The build tools load them in Node, where `Plot`, `D`, `Viz`, `katex` and `document` do
not exist. So never call page functions inside a data file; refer to visualizers and diagrams by name.

### Blocks
| Block | Meaning |
|---|---|
| `"text"` or `["p", text]` | paragraph |
| `["h", text]` / `["h2", text]` | sub-heading (h3) / bigger heading (rare) |
| `["list", [...]]`, `["ol", [...]]`, `["steps", [...]]` | bullet list, numbered list, numbered steps |
| `["code", code, output, title?, local?]` | code block (see 3.2) |
| `["math", tex, caption?]` | centred formula (KaTeX), e.g. `["math", R\`\frac{a}{b}\`]` |
| `["table", headers, rows]` | table; cells use inline formatting |
| `["tip" \| "note" \| "warn" \| "danger", text, label?]` | callout. `danger` is labelled "Common mistake" |
| `["viz", name]` | an interactive visualizer from `assets/viz/*.js` |
| `["diagram", name, [args], caption]` | a static SVG drawn by `D[name](...args)` (define it in your viz file) |
| `["ex", {t, q, idea, steps, code, out, open}]` | a collapsible worked example |
| `["qa", question, blocksOrText]` | a collapsible box (use for "Going deeper" / optional detail, **not** quizzes) |
| `["html", raw]` | raw HTML (rarely needed) |

**Inline formatting** (in any text): `` `code` ``, `**bold**`, `*italic*`, `[link](page.html#id)`, and maths `$x^2$`
(no space just inside the dollars). In normal JS strings backslashes must be doubled: `"$\\frac{1}{n}\\sum_i x_i$"`.
Inside `R\`...\`` (String.raw) they are single. `R` templates can't contain a backtick and must not contain `${`.

Cross-link generously to other sections with `[text](page.html#section-id)` using the outline ids
(e.g. `[softmax](math.html#entropy)`, `[attention](transformers.html#scaled-dot-product)`).

### 3.2 Code blocks
- Write code with `R\`...\``. Python only. Output `"?"` is filled by `node tools/fill.js data/<file>.js`
  (it runs the code and writes the real output into your file). Never type outputs by hand for runnable code.
- Runnable in the browser (Pyodide) **and** locally: standard library, `numpy` (2.0.2), `pandas`, `scikit-learn` (1.6.1),
  `scipy`, `matplotlib`. Teach algorithms **from scratch in NumPy** first, then show the scikit-learn / library one-liner.
- **Deterministic**: `rng = np.random.default_rng(0)`, `random_state=0`. **Fast**: each example under ~3 s.
- Print readable, stable values: round floats (`round(x, 3)`, `arr.round(3)`), convert NumPy scalars inside lists with
  `float()` / `.tolist()` (NumPy 2 prints `np.float64(0.5)` inside lists). Avoid printing huge arrays.
- Keep examples focused: usually 8–30 lines. Comment the key lines (`# ∂L/∂w`, `# shift for stability`).
- The step-through tracer records the first 400 lines executed; small loops step through nicely, big ones are fine but
  only the start is steppable.
- **Plots**: use matplotlib, `plt.figure(figsize=(6, 3.5))`, labels and a title, end with `plt.show()`. Also print
  something (a number) so the block has text output. The build turns each figure into a picture under the code.
- **Code that can't run in a browser** (PyTorch, Hugging Face `transformers`, LLM API SDKs, LangGraph, MCP servers…):
  `["code", code, output, "PyTorch", "local"]`. It shows a "Runs on your computer" badge and is never run by the site.
  This PC has `torch 2.13 (CPU)`, `torchvision`, `transformers 4.57`, `sentence-transformers`, `xgboost`, `lightgbm`. If
  your local example runs **without downloading anything** (no model weights, no datasets, no internet), run it yourself
  (`python file.py`) and paste the **real** output. Otherwise give **no output** (use `""`). Never fabricate output.
- Real model ids for Anthropic API examples: `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`. If you write
  Anthropic / Claude API or Agent SDK code, first invoke the `claude-api` skill to get the current, correct usage.

### 3.3 Briefs (`data/briefs-<file>.js`)
```js
/* plain-English briefs shown at the top of each section ("<kind>:<section id>") */
Object.assign(window.BRIEFS, {
  "ml:kmeans": "First paragraph: what it is and why it matters, zero jargon.\n\nSecond paragraph: the one key idea or trade-off to remember.",
});
```
Every section gets a brief: two short paragraphs, no maths, no code, as if explaining to a smart friend.

### 3.4 What a good section contains (adapt, don't pad)
1. `sub`: one sentence that says what it is in plain words.
2. The intuition in 1–3 paragraphs, with an everyday analogy if it really helps.
3. A visualizer (`viz`) or diagram where a picture beats words. Not every section needs one; the key ones do.
4. The maths, introduced gently: what each symbol means, a tiny worked number example.
5. From-scratch NumPy code with outputs, then the library version (scikit-learn / PyTorch local).
6. A table comparing options or summarising (when there are options).
7. `tip` / `warn` / `danger` callouts for practical advice and the classic mistakes.
8. Links to related sections and practice problems (`practice.html#p/<id>`).

Typical size: 8–20 blocks. Concept-only sections (e.g. safety, frameworks) can skip code but should still be concrete:
examples, tables, and small runnable toys where possible (a fake LLM, a toy tool, a regex guardrail…).

## 4. Visualizers, diagrams and glossary terms (`assets/viz/<prefix>.js`)

```js
/* <Page>: visualizers, diagrams and glossary terms. Loaded on every doc page and on Practice, after assets/viz.js. */
(function () {
  const V = Viz.defs, Plot = Viz.Plot, { nums, split2, range, C, r3 } = Viz.helpers;

  V.ml2_kmeans = {                         // names MUST start with your prefix + "_"
    title: "k-Means: assign, then move the centroids",
    params: [{ id: "k", label: "clusters k", min: 2, max: 5, step: 1, value: 3 }],   // optional sliders
    // input: "1 2 3 | 4", help: "numbers | target",                                  // optional text input
    // slide: "end",                      // optional: moving a slider jumps to the last step
    gen(input, P) {                        // P = { k: 3 } from the sliders; return an array of steps
      const S = [];
      S.push({ svg: Plot.chart({ x: [0, 10], y: [0, 10], xl: "x₁", yl: "x₂" }, [Plot.pts([[1, 2, "a"], [8, 9, "b"]])]),
               side: { iteration: 0, inertia: "12.3" }, msg: "What happens in this step, in **plain words**." });
      return S;
    },
  };

  // static diagrams used with ["diagram", "ml2_elbow", [args], "caption"]
  D.ml2_elbow = (n) => `<svg viewBox="0 0 400 200" width="400" height="200" xmlns="http://www.w3.org/2000/svg">…</svg>`;

  // glossary: hover definitions for jargon on every page (lowercase keys, one plain sentence or two)
  Object.assign(window.GLOSSARY, {
    "centroid": "The centre point of a cluster: the average of all the points assigned to it.",
  });
})();
```
- Step kinds: `{ svg, side, msg }` (most visualizers), `{ rows: [{label, vals, cls, ptr}], side, msg }` (arrays / tokens),
  `{ grid, cls: {"r,c": "hl"|"done"|"water"|"land"}, side, msg }` with `kind: "grid"` on the definition (matrices,
  convolution). Cell classes for rows/grid: `hl`, `inwin`, `done`, `dim`, `bad`.
- A visualizer returning a single step hides Play / Prev / Next: a live picture driven by its sliders (good for
  "temperature", "threshold", "degree"…). Multi-step ones animate an algorithm.
- `msg` supports inline formatting (bold, code, `$maths$`).
- **Plot helper** (see the comment at the top of `assets/viz.js`): `Plot.chart({W, H, x:[a,b], y:[c,d], xl, yl}, layers)`
  with layers `Plot.fn(f, cls)`, `Plot.line(pts, cls)`, `Plot.pts(pts, cls, r)` (a point can be `[x, y, "a"|"b"]`),
  `Plot.dot`, `Plot.seg(x1,y1,x2,y2,cls,arrow)`, `Plot.text`, `Plot.vline`, `Plot.hline`, `Plot.regions(fn(x,y) → "a"|"b"|null)`.
  You can also pass your own layer function `(S) => svgString` using `S.sx(x)`, `S.sy(y)`.
- **Colours: only via classes or CSS variables** so dark mode works:
  `pl-c1` (page colour) `pl-c2` (amber) `pl-c3` (green) `pl-c4` (red) `pl-c5` (purple), `pl-a` / `pl-b` (two classes),
  `pl-thin`, `pl-dash`; in your own SVG use `style="fill:var(--accent)"`, `var(--green)`, `var(--amber)`, `var(--red)`,
  `var(--purple)`, `var(--cls-a)`, `var(--cls-b)`, `var(--muted)`, `var(--text)`, `var(--panel)`, `var(--panel-2)`,
  `var(--border)`, and for soft fills `color-mix(in srgb, var(--accent) 18%, transparent)`.
  Diagram node classes from the DSA engine also work: `class="node"` (`node hl`, `node ok`, `node bad`), `class="edge"`,
  text classes `val`, `lbl`, `acc`.
- SVGs: give a `viewBox` and `width`/`height` (max ~640 wide); they shrink on small screens. Font size 11–13 px.
- Keep each visualizer's maths **correct** (compute it, don't fake it) and its messages explanatory.
- Escape text you put in SVG with `esc()` (a global). Use helpers you define inside your own IIFE; don't add globals.
- Glossary keys: check `assets/glossary.js` for terms that already exist; add the jargon your pages introduce.

## 5. Practice problems (`data/prac/<NN>-<name>.js`)

Copy the shape of `data/prac/01-numpy-linalg.js` exactly:
```js
PRAC.add("Regression", [ { id, t, d, p, q, ex, c, hint, brute, opt, run, out, dryOn, dry, expl, take, mist }, … ]);
```
- `id` unique across the whole site (use descriptive ids like `linreg-closed-form`), `t` title, `d` `"Easy"|"Medium"|"Hard"`,
  `p` topic tag, `q` the task (paragraphs separated by a blank line; a paragraph that is only `$formula$` shows as a
  display formula; lines starting `1.` or `- ` become lists), `ex` [[input, output, why]], `c` constraints.
- `brute`: `{ name, idea, code, tc, tw, sc, sw, bad }` (usually "Plain Python loops"); `opt`: `{ name, idea, steps, code,
  tc, tw, sc, sw }` (usually "Vectorized with NumPy"); `better` optional. For problems where "loops vs vectorized" makes no
  sense (e.g. an agent loop or a JSON parser), use `brute` for a naive version and `opt` for the robust one, and name them.
- The learner's starter code is built from `opt.code`: imports are kept and every `def` gets an empty body, so put the
  imports at the top of `opt.code` and keep helper functions as `def`s.
- `run`: the tests, a few `print(...)` lines that call the solution (one print per check, each printing a short, rounded,
  deterministic value). `out`: exactly what `run` prints. Generate `out` by actually running opt + run (and check brute
  prints the same). `node tools/verify.js data/prac/<your file>.js` must pass for opt, brute and better.
- `dry`: `{ h: [headers], r: [[cells]] }` table or `{ setup, steps: [[state, why]], result }`; `expl`: [[code line, why]];
  `take`: one bold key takeaway; `mist`: 2–4 common mistakes. `opt.viz` may name a visualizer to show in the solution.
- Problems should be genuinely useful implement-from-scratch exercises, Easy → Hard within a category.

## 6. Python environment facts
- numpy 2.0.2, pandas 2.3, scikit-learn 1.6.1, scipy 1.16, matplotlib 3.10 locally (the browser has matching
  numpy / sklearn). `MPLBACKEND=Agg` is set by the tools, so `plt.show()` never opens a window.
- No internet access is assumed for examples. Never download models or datasets in runnable code; build small toy data
  in code (`np.random.default_rng(0)`, `sklearn.datasets.make_blobs / make_moons / make_classification`,
  `load_iris` / `load_digits` / `load_breast_cancer` / `load_diabetes`, which ship inside scikit-learn).

## 7. Tools you may run (from ROOT, Git Bash or PowerShell)
```
node tools/fill.js data/<file>.js                     # fill "?" outputs in your doc file
node tools/verify.js data/<file>.js                   # every example in your file prints its shown output
node tools/verify.js data/prac/<NN>-<name>.js         # your practice problems pass (opt, brute, better)
node tools/verify.js data/<file>.js --local           # also run your "local" blocks (torch etc.)
node --check assets/viz/<prefix>.js                   # syntax check your visualizers
python tools/pages.py && node tools/stamp.js          # regenerate pages so they load your new files (safe any time)
node tools/cdp.js <your-scratch-dir>/steps.js         # headless Chrome test (see below)
```
Windows notes: Python output must be UTF-8 (the tools set it). Heredocs with mixed quotes break in Git Bash: write helper
scripts to your scratchpad with the Write tool and run them. Use your own scratchpad folder for temporary files.

**Browser test** (do this before you finish): write `<scratch>/steps.js`:
```js
module.exports = async ({ go, ev, shot, sleep, log }) => {
  await go("index.html"); await ev(`localStorage.setItem("theme","light"); localStorage.setItem("anim","off")`);
  for (const id of ["kmeans", "pca"]) { await go("revision.html"); await go("ml.html#" + id); await sleep(500);
    log(id, await ev(`document.title + " | " + document.querySelector("#content").innerText.length`)); await shot("light-" + id); }
};
```
Run `node tools/cdp.js <scratch>/steps.js`; it prints `PAGE ERRORS` if any script failed (there must be none from your
files). Screenshots are saved next to steps.js; look at a few (Read tool) in light **and** dark (`theme","dark"`) to check
your visualizers and diagrams look right. Note: navigating to the same page with only a different `#hash` doesn't reload,
so go to another page first (as above). Inline outputs and Step through appear only after the lead runs build.js; until
then outputs show in a box under the code, which is expected.

## 8. Quality checklist before you finish
- [ ] Every outline section for your file exists with the exact id/title/group, in order, each with a brief.
- [ ] `node tools/fill.js` then `node tools/verify.js` on each of your files: all OK, nothing fabricated.
- [ ] Visualizers: correct maths, clear messages, sliders where they help, readable in light and dark (screenshots seen).
- [ ] Practice problems verified (opt, brute, better all print `out`).
- [ ] No emoji, no interview-question boxes, no invented facts or outputs; volatile facts marked "as of 2026".
- [ ] Cross-links point to real outline ids.

## 9. Final report (your last message)
Short: files written; number of sections / visualizers / diagrams / practice problems; verify results (counts); any
engine change you'd like the lead to make; anything you couldn't finish.

## 10. The full outline (ids are fixed: use them for sections and cross-links)

**Math for ML** (`math.html`, list `MATH`, kind `math`)
- Start Here: `why-math` Why Math for ML (and How Much You Need)
- Linear Algebra: `vectors` Vectors · `dot-product` Dot Product, Norms & Cosine Similarity · `matrices` Matrices & Matrix Multiplication · `matrix-ops` Transpose, Inverse & Identity · `eigen` Eigenvalues & Eigenvectors · `svd` Singular Value Decomposition (SVD)
- Calculus: `derivatives` Derivatives · `gradients` Partial Derivatives & Gradients · `chain-rule` The Chain Rule · `jacobians` Shapes, Jacobians & Matrix Calculus
- Probability: `probability` Probability Basics & Conditional Probability · `bayes` Bayes' Theorem · `expectation` Expectation & Variance · `distributions` Common Distributions · `mle` Maximum Likelihood Estimation
- Information Theory: `entropy` Entropy, Cross-Entropy & KL Divergence
- Optimization: `optimization` Convexity, Minima & Saddle Points

**Python for ML** (`pyml.html`, list `PYML`, kind `pyml`)
- Start Here: `setup` Python for ML: Setup & Tools
- NumPy: `numpy-arrays` Arrays, dtypes & Shapes · `numpy-indexing` Indexing, Slicing & Boolean Masks · `vectorization` Vectorization & ufuncs · `broadcasting` Broadcasting · `axes` Axes & Reductions · `numpy-linalg` Matrix Operations & @ · `numpy-random` Random Numbers & Seeds
- pandas: `pandas-basics` Series & DataFrame · `pandas-select` Selecting & Filtering · `pandas-groupby` groupby, merge & pivot · `pandas-missing` Missing Values & Cleaning
- Plotting: `matplotlib` Plotting with matplotlib
- scikit-learn: `sklearn-api` The scikit-learn API · `sklearn-pipelines` Pipelines, Splits & Cross-Validation in scikit-learn

**Statistics for ML** (`stats.html`, list `STATS`, kind `stats`, file `data/stats.js`; header: Foundations ▾ → Statistics)
- Basics: `descriptive` Descriptive Statistics · `clt` Sampling Distributions & the Central Limit Theorem · `confidence-intervals` Confidence Intervals
- Hypothesis Testing: `hypothesis-testing` Hypothesis Testing: p-values & Errors · `t-tests` t-Tests: One-Sample, Two-Sample & Paired · `chi-square` Chi-Square Tests · `anova` ANOVA · `nonparametric` Non-Parametric Tests · `permutation-bootstrap` Permutation Tests & the Bootstrap · `multiple-testing` Multiple Testing: Bonferroni & FDR
- Applied: `ab-testing` A/B Testing: Power, Sample Size & Effect Size · `correlation` Correlation & Covariance · `causation` Correlation vs Causation & Confounders · `regression-inference` Regression Inference · `bayes-vs-freq` Bayesian vs Frequentist Inference

**Machine Learning** (`ml.html`, list `ML`, kind `ml`; part 1 = `data/ml-1.js`, part 2 = `data/ml-2.js`)
- Foundations: `what-is-ml` What is Machine Learning? · `workflow` The ML Workflow · `splits` Train / Validation / Test & Data Leakage · `bias-variance` Bias–Variance, Under- & Overfitting · `loss-functions` Loss Functions & Empirical Risk
- Regression: `linear-regression` Linear Regression · `gradient-descent` Gradient Descent (exists) · `polynomial` Polynomial Features · `regularization` Regularization: L1, L2 & ElasticNet
- Classification: `logistic-regression` Logistic Regression · `knn` k-Nearest Neighbours · `naive-bayes` Naive Bayes · `svm` Support Vector Machines & Kernels · `decision-trees` Decision Trees · `random-forests` Bagging & Random Forests · `boosting` Gradient Boosting & XGBoost
- *(part 2)* Unsupervised: `kmeans` k-Means Clustering · `dbscan` DBSCAN & Hierarchical Clustering · `pca` PCA & Dimensionality Reduction
- Evaluation: `classification-metrics` Confusion Matrix, Precision, Recall & F1 · `roc-auc` ROC-AUC & PR Curves · `regression-metrics` Regression Metrics · `cross-validation` Cross-Validation & Hyperparameter Tuning
- Practical: `feature-engineering` Feature Engineering & Scaling · `encoding` Encoding Categorical Data · `imbalanced` Imbalanced Data · `model-selection` Which Model When: A Cheat Sheet

**Classical ML** (`classical.html`, list `CML`, kind `cml`; part 1 = `data/cml-1.js`, part 2 = `data/cml-2.js`; header: ML ▾ → Classical ML)
- Discriminants & Kernels: `lda` Linear Discriminant Analysis (Fisher's LDA) · `qda` QDA & Gaussian Discriminant Analysis · `bayes-linreg` Bayesian Linear Regression · `kernel-methods` Kernel Methods: Kernel Trick, Kernel Ridge & Kernel PCA · `gaussian-processes` Gaussian Processes
- Probabilistic Models: `gmm` Gaussian Mixture Models · `em` The EM Algorithm · `hmm` Hidden Markov Models: Forward, Viterbi & Baum-Welch · `topic-lda` Latent Dirichlet Allocation (Topic Models) · `mcmc` Markov Chains & MCMC: Metropolis-Hastings & Gibbs
- *(part 2)* Factorization & Manifolds: `ica` Independent Component Analysis (ICA) · `nmf` Non-Negative Matrix Factorization (NMF) · `recsys-mf` Matrix Factorization for Recommenders (ALS) · `tsne-umap` t-SNE & UMAP · `mds-isomap` MDS & Isomap · `spectral-clustering` Spectral Clustering
- Other Classics: `adaboost` AdaBoost · `anomaly-detection` Anomaly Detection: Isolation Forest, LOF & One-Class SVM · `apriori` Association Rules (Apriori) · `kalman` The Kalman Filter · `time-series` Time Series: AR, MA & ARIMA

**Deep Learning** (`dl.html`, list `DL`, kind `dl`; part 1 = `data/dl-1.js`, part 2 = `data/dl-2.js`)
- Basics: `what-is-dl` What is Deep Learning? · `perceptron` The Neuron & Perceptron · `mlp` Multi-Layer Perceptrons & the Forward Pass · `activations` Activation Functions · `dl-losses` Loss Functions for Neural Networks · `backprop` Backpropagation by Hand · `autograd` Build a Tiny Autograd Engine
- Training: `training-loop` The Mini-Batch Training Loop · `optimizers` Optimizers: SGD, Momentum, RMSProp & Adam · `lr-schedules` Learning-Rate Schedules · `initialization` Weight Initialization · `vanishing-gradients` Vanishing & Exploding Gradients · `normalization` BatchNorm & LayerNorm · `dl-regularization` Dropout, Weight Decay & Early Stopping
- *(part 2)* Architectures: `cnn` Convolutional Neural Networks · `pooling-resnet` Pooling, Classic CNNs & ResNets · `rnn` Recurrent Neural Networks · `lstm` LSTM & GRU · `seq2seq` Seq2Seq & the Birth of Attention · `word-embeddings` Word Embeddings: word2vec & GloVe · `generative` Generative Models: Autoencoders, VAEs, GANs & Diffusion
- In Practice: `pytorch-basics` PyTorch Basics · `pytorch-training` Training a Model in PyTorch · `transfer-learning` Transfer Learning & Fine-Tuning · `debugging-training` Debugging Neural Network Training

**Transformers** (`transformers.html`, list `TF`, kind `tf`, file `data/tf.js`)
- Tokens: `why-tokens` Why Tokens? · `bpe` Byte-Pair Encoding (BPE) · `tokenizers` WordPiece, SentencePiece & Byte-Level BPE
- Embeddings: `token-embeddings` Token Embeddings · `positional` Positional Encodings: Sinusoidal, Learned & RoPE
- Attention: `attention-intuition` Attention: Queries, Keys & Values · `scaled-dot-product` Scaled Dot-Product Attention · `masking` Causal & Padding Masks · `multi-head` Multi-Head Attention · `self-vs-cross` Self-Attention vs Cross-Attention
- Architecture: `transformer-block` The Transformer Block · `architectures` Encoder-Only, Decoder-Only & Encoder–Decoder · `tiny-gpt` A Tiny GPT Forward Pass in NumPy · `efficient-attention` Efficient Attention: FlashAttention, MQA & GQA · `moe` Mixture of Experts · `vit-multimodal` Vision Transformers & Multimodal Models

**LLMs** (`llm.html`, list `LLM`, kind `llm`; part 1 = `data/llm-1.js`, part 2 = `data/llm-2.js`)
- Training: `bigram-lm` A Bigram Language Model · `pretraining` Pretraining: Next-Token Prediction at Scale · `scaling-laws` Scaling Laws & Chinchilla · `sft` Supervised Fine-Tuning & Instruction Tuning · `lora` LoRA, QLoRA & PEFT · `rlhf` RLHF: Learning from Human Feedback · `dpo` DPO, RLAIF & Constitutional AI · `reasoning-models` Reasoning Models & Test-Time Compute · `distillation` Distillation & Small Models
- Inference: `greedy-beam` Greedy & Beam Search · `sampling` Sampling: Temperature, Top-k & Top-p · `kv-cache` The KV Cache · `quantization` Quantization · `serving` Serving LLMs: Batching & Speculative Decoding · `long-context` Long Context
- *(part 2)* Using LLMs: `prompt-engineering` Prompt Engineering · `structured-output` Structured Output & JSON · `embeddings-search` Embeddings & Vector Search · `rag` Retrieval-Augmented Generation (RAG) · `rag-advanced` Advanced RAG: Chunking, Hybrid Search & Reranking · `llm-evaluation` Evaluating LLMs · `hallucinations` Hallucinations · `llm-safety` Safety, Jailbreaks & Prompt Injection · `llm-apis` Using LLM APIs · `multimodal-llms` Multimodal LLMs

**Agentic AI** (`agents.html`, list `AG`, kind `ag`, file `data/ag.js`)
- Concepts: `what-is-agent` What is an Agent? · `agent-loop` The Agent Loop · `tool-use` Tool Use & Function Calling · `react` ReAct: Reason + Act · `planning` Planning & Task Decomposition · `reflection` Reflection & Self-Correction · `memory` Agent Memory · `context-engineering` Context Engineering · `agentic-rag` Agentic RAG
- Systems: `multi-agent` Multi-Agent Patterns · `mcp` Model Context Protocol (MCP) · `computer-use` Computer-Use & Coding Agents
- Production: `guardrails` Guardrails & Human-in-the-Loop · `agent-security` Agent Security · `agent-evals` Evaluating Agents · `failure-modes` Failure Modes & Debugging · `frameworks` Agent Frameworks Compared · `capstone` Capstone: Build an Agent from Scratch

**Practice categories** (in `assets/site.js`; use these exact names in `PRAC.add`): NumPy Basics · Linear Algebra ·
Calculus & Probability · Statistics · Regression · Classification · Clustering & PCA · Evaluation · Classical ML · Neural Net Basics · Training ·
CNNs & RNNs · Tokenization · Attention · Transformer Blocks · Decoding & Efficiency · RAG · Agents.
Existing problems: `softmax` (NumPy Basics), `cosine-similarity` (Linear Algebra).
