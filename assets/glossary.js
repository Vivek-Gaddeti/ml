/* glossary.js — plain-English definitions shown when you hover/tap a jargon word.
   glossify(element) underlines the first occurrence of each term inside that element. */
window.GLOSSARY = {
  /* ---- data & training ---- */
  "feature": "One input column the model looks at, e.g. a house's size or number of rooms. Also called a variable or attribute.",
  "label": "The answer we want the model to predict for an example (the price, 'spam' or 'not spam'…).",
  "training set": "The examples the model learns from. It sees their labels while fitting.",
  "validation set": "Examples held back to compare settings (hyperparameters) and catch overfitting. The model never trains on them.",
  "test set": "Examples locked away until the very end, used once to estimate how the model will do on new data.",
  "epoch": "One full pass over the whole training set.",
  "batch": "A small group of examples processed together in one training step (e.g. 32 rows).",
  "mini-batch": "A small random group of examples used for one gradient step; a middle ground between one example and the whole dataset.",
  "hyperparameter": "A setting you choose before training (learning rate, number of trees, k in k-NN…), not learned from the data.",
  "parameter": "A number the model learns from data, like a weight or a bias.",
  "weights": "The learned numbers that say how strongly each input affects the output.",
  "loss": "One number that says how wrong the model's predictions are. Training tries to make it small.",
  "loss function": "The formula that turns predictions and true answers into the loss, e.g. mean squared error or cross-entropy.",
  "gradient": "The vector of slopes of the loss with respect to every parameter. It points uphill; we step the opposite way.",
  "gradient descent": "Repeatedly nudge the parameters a small step against the gradient, so the loss goes down.",
  "learning rate": "How big each gradient descent step is. Too small: slow. Too big: the loss jumps around or explodes.",
  "overfitting": "When a model memorises the training data (noise included) and does worse on new data.",
  "underfitting": "When a model is too simple to capture the pattern, so it does badly even on the training data.",
  "regularization": "Anything that discourages an overly complex model (e.g. penalising large weights, dropout) to reduce overfitting.",
  "generalization": "How well a model does on data it has never seen.",
  "data leakage": "When information from the test data (or the future) sneaks into training, making results look better than they really are.",
  "vectorized": "Written as whole-array NumPy operations instead of Python loops: shorter and often 10–100× faster.",
  "broadcasting": "NumPy's rule for combining arrays of different shapes by stretching size-1 dimensions, e.g. adding a row to every row of a matrix.",
  /* ---- models & outputs ---- */
  "logits": "The raw scores a model outputs before softmax turns them into probabilities.",
  "softmax": "Turns a list of scores into probabilities that are all positive and add up to 1. Bigger score → bigger share.",
  "sigmoid": "The S-shaped function 1 / (1 + e^(−x)); squashes any number into (0, 1), so it can be read as a probability.",
  "activation function": "The non-linear function a neuron applies to its weighted sum (ReLU, sigmoid, tanh…). Without it, stacked layers collapse into one linear layer.",
  "backpropagation": "The chain rule applied backwards through a network to get the gradient of the loss for every weight.",
  "embedding": "A list of numbers (a vector) that represents a word, token, image or item, placed so that similar things end up close together.",
  "cross-entropy": "The usual classification loss: −log(probability the model gave to the correct answer). Confident and wrong costs a lot.",
  /* ---- LLMs ---- */
  "token": "A chunk of text the model reads and writes: a word, part of a word or a symbol. 'unbelievable' might be 'un' + 'believ' + 'able'.",
  "tokenizer": "The tool that splits text into tokens and maps each one to an integer id (and back).",
  "context window": "The maximum number of tokens a model can look at at once (prompt + its own output).",
  "attention": "The mechanism that lets each token look at every other token and decide how much each one matters for it.",
  "transformer": "The neural network architecture behind modern LLMs: stacked blocks of attention + small feed-forward networks.",
  "fine-tuning": "Training an already-trained model a bit more on your own, smaller dataset to specialise it.",
  "rlhf": "Reinforcement Learning from Human Feedback: people rank model answers, a reward model learns their taste, and the LLM is tuned to score well on it.",
  "lora": "Low-Rank Adaptation: fine-tune a big model by learning two small matrices per layer instead of changing all the weights.",
  "quantization": "Storing weights with fewer bits (e.g. 8 or 4 instead of 16) to save memory and speed up inference, losing a little accuracy.",
  "hallucination": "When an LLM states something false or made up, fluently and confidently.",
  "temperature": "A sampling knob: low (→ 0) makes outputs predictable, high makes them more random and creative.",
  "inference": "Using a trained model to make predictions (as opposed to training it).",
  "rag": "Retrieval-Augmented Generation: look up relevant documents first, paste them into the prompt, then let the LLM answer from them.",
  /* ---- agents ---- */
  "tool call": "When an LLM asks the program to run a function for it (search, calculator, database…) by outputting the function name and arguments as JSON.",
  "function calling": "The API feature that lets an LLM return a structured tool call (name + JSON arguments) instead of plain text.",
  "mcp": "Model Context Protocol: an open standard for connecting AI apps to tools and data sources through small 'MCP servers'.",
  "agent": "An LLM in a loop: it decides what to do, calls tools, looks at the results, and repeats until the task is done.",
  "prompt injection": "Text hidden in a web page, email or document that tries to give an LLM new instructions it shouldn't follow.",
};

window.glossify = function (root) {
  if (!root || !window.GLOSSARY) return;
  const terms = Object.keys(GLOSSARY).sort((a, b) => b.length - a.length);
  const re = new RegExp("\\b(" + terms.map((t) => t.replace(/[.*+?^${}()|[\]\\-]/g, "\\$&")).join("|") + ")\\b", "i");
  const used = new Set();
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => n.parentElement.closest("code, pre, a, button, abbr, h1, .code-wrap, .viz, .chat-panel, summary, th, .crumbs, .badge, input, textarea, svg, .katex, .math-i, .math-d")
      ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach((node) => {
    let text = node.nodeValue, m, frag = null, last = 0, guard = 0;
    const r = new RegExp(re.source, "gi");
    while ((m = r.exec(text)) && guard++ < 50) {
      const key = m[1].toLowerCase();
      if (used.has(key)) continue;
      used.add(key);
      frag = frag || document.createDocumentFragment();
      frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      const ab = document.createElement("abbr");
      ab.className = "gloss"; ab.tabIndex = 0; ab.textContent = m[1]; ab.dataset.def = GLOSSARY[key];
      frag.appendChild(ab);
      last = m.index + m[1].length;
    }
    if (frag) { frag.appendChild(document.createTextNode(text.slice(last))); node.parentNode.replaceChild(frag, node); }
  });
};

/* one floating tooltip shared by all terms (works on hover, focus and tap) */
(function () {
  const css = `abbr.gloss{text-decoration:underline dotted var(--accent);text-underline-offset:3px;cursor:help}
  .gloss-tip{position:fixed;z-index:120;max-width:320px;background:var(--text);color:var(--bg);font-size:var(--fs-sm);line-height:1.45;padding:.55rem .7rem;border-radius:var(--r-sm);box-shadow:0 8px 24px rgba(0,0,0,.25);pointer-events:none}
  .gloss-tip b{display:block;margin-bottom:.15rem;text-transform:capitalize}`;
  const st = document.createElement("style"); st.textContent = css; document.head.appendChild(st);
  let tip = null;
  const show = (el) => {
    if (!tip) { tip = document.createElement("div"); tip.className = "gloss-tip"; document.body.appendChild(tip); }
    tip.innerHTML = `<b>${el.textContent}</b>${el.dataset.def}`;
    tip.hidden = false;
    const r = el.getBoundingClientRect(), w = Math.min(320, window.innerWidth - 20);
    tip.style.left = Math.max(10, Math.min(r.left, window.innerWidth - w - 10)) + "px";
    tip.style.top = (r.bottom + 8 + 120 > window.innerHeight ? r.top - tip.offsetHeight - 8 : r.bottom + 8) + "px";
  };
  const hide = () => { if (tip) tip.hidden = true; };
  document.addEventListener("mouseover", (e) => { const a = e.target.closest("abbr.gloss"); a ? show(a) : hide(); });
  document.addEventListener("focusin", (e) => { const a = e.target.closest("abbr.gloss"); if (a) show(a); });
  document.addEventListener("focusout", hide);
  window.addEventListener("scroll", hide, { passive: true });
})();
