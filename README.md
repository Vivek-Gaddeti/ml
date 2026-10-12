# AI Study Hub

One offline study website for Machine Learning, Deep Learning, Transformers, LLMs and Agentic AI. It is built on the same
engine as the DSA + Python Study Hub. Open `index.html` (double-click) to start: it is the dashboard.

| Page | What's inside |
|---|---|
| `index.html` | Dashboard: today's goal, streaks, progress (All / Foundations / ML / Deep Learning / LLMs & Agents), what's due, a year of activity |
| `math.html` | Math for ML (header: **Foundations ▾ → Math for ML**) |
| `pyml.html` | Python for ML: NumPy, pandas, matplotlib, scikit-learn (**Foundations ▾ → Python for ML**) |
| `ml.html` | Machine Learning |
| `dl.html` | Deep Learning & Neural Networks |
| `transformers.html` | Transformers (**LLMs & Agents ▾ → Transformers**) |
| `llm.html` | LLMs (**LLMs & Agents ▾ → LLMs**) |
| `agents.html` | Agentic AI (**LLMs & Agents ▾ → Agentic AI**) |
| `practice.html` | Implement-from-scratch exercises as a list or a roadmap, with Try it yourself |
| `revision.html` | Spaced-repetition review of everything |

Press **Ctrl + K** (or the search icon in the header) on any page to find any section or problem.

## Run Python on the page (`assets/py.js`)
Real Python 3.12 (Pyodide) runs inside the browser. It needs internet the **first** time it starts (about 10 MB, then the
browser caches it). **numpy, pandas, scikit-learn and matplotlib** load automatically the first time code imports them
(a few more MB, once). Everything else works offline.
- **Edit & run** on every Python example; plots you draw appear under the output.
- **Python playground**: the terminal icon in the header (or press **P**). Your code is saved in this browser.
- **Try it yourself** on every practice problem: write your solution, press Run, and it is checked against the tests.
- **Step through** any example or your own code: variables, call stack and output so far. NumPy arrays show their
  shape first, e.g. `(3, 4) float64 [[…]]`.
- Blocks marked **Runs on your computer** (PyTorch, Hugging Face, LLM APIs) can't run in a browser. Copy them and run
  them with Python on your PC.

## Maths
Formulas use KaTeX, saved in `assets/vendor/katex` (MIT licence), so they work offline. In content: `["math", R`\frac{a}{b}`]`
for a centred formula, or `$x^2$` inside text.

## Editing content (optional)
Content lives in `data/*.js` (one file per page: `math.js`, `pyml.js`, `ml.js`, `dl.js`, `tf.js`, `llm.js`, `ag.js`, plus
`briefs-<page>.js`) and `data/prac/*.js` (practice problems). After editing, run (needs Python + Node, and
`pip install numpy==2.0.2 pandas scikit-learn==1.6.1 matplotlib` so results match the browser):

```
node tools/fill.js data/ml.js          # fill outputs written as "?"
node tools/verify.js                   # check every example prints its shown output (add --local to also run PyTorch blocks)
node tools/build.js                    # inline outputs, step-through traces and plots (+ runs lite.js and stamp.js)
node tools/lite.js                     # rebuild data/index.js and data/progress.js (Revision, dashboard, Ctrl+K)
node tools/stamp.js                    # refresh the ?v= version stamps so browsers never mix old and new files
python tools/pages.py                  # rewrite every page from one template (then run stamp.js)
node tools/cdp.js my-steps.js          # test the site in headless Chrome (see the comment at the top of the file)
```

Progress is saved in this browser's localStorage under its own key (`aihub:v1`), separate from the DSA site. Use the
gear icon on the dashboard to export, import or reset it.
