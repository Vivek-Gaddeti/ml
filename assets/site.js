/* Site config: one site for Machine Learning, Deep Learning, Transformers, LLMs and Agentic AI */
window.SITE = {
  storeKey: "aihub:v1",                                              // all progress, notes and bookmarks
  prefix: "ai:",                                                     // in front of this site's own localStorage keys (the DSA site shares the browser's storage)
  brand: "AI Study Hub",
  logo: "AI",
  home: "index.html",
  mascot: "bot",                                                     // walks along the bottom of the sidebar as you progress
  ring: { prefixes: ["math:", "pyml:", "ml:", "dl:", "tf:", "llm:", "ag:", "prac:"], total: 255, label: "items done" },   // progress ring around the logo
  // [href, label] or [href, label, [[href, label, subtitle], …]] for a link with a small menu
  links: [
    ["index.html", "Dashboard"],
    ["math.html", "Foundations", [
      ["math.html", "Math for ML", "Vectors, gradients, probability"],
      ["pyml.html", "Python for ML", "NumPy, pandas, plots, scikit-learn"],
      ["stats.html", "Statistics", "Hypothesis tests, CIs, A/B tests, bootstrap"],
    ]],
    ["ml.html", "ML", [
      ["ml.html", "Core ML", "Workflow, models, evaluation"],
      ["classical.html", "Classical ML", "LDA, EM, HMMs, kernels, MCMC, Kalman…"],
    ]],
    ["dl.html", "Deep Learning"],
    ["transformers.html", "LLMs & Agents", [
      ["transformers.html", "Transformers", "Tokens, attention, the transformer block"],
      ["llm.html", "LLMs", "Training, decoding, prompting, RAG"],
      ["agents.html", "Agentic AI", "Tools, ReAct, memory, MCP, evals"],
    ]],
    ["practice.html", "Practice"],
    ["revision.html", "Revision"],
  ],
  // the doc pages: Store keys are "<kind>:<section id>", sections live in window[list]
  docs: [
    { kind: "math", list: "MATH", href: "math.html", title: "Math for ML", type: "Math", part: "found", noun: "math topics" },
    { kind: "pyml", list: "PYML", href: "pyml.html", title: "Python for ML", type: "Python for ML", part: "found", noun: "Python topics" },
    { kind: "stats", list: "STATS", href: "stats.html", title: "Statistics for ML", type: "Statistics", part: "found", noun: "statistics topics" },
    { kind: "ml", list: "ML", href: "ml.html", title: "Machine Learning", type: "ML", part: "ml", noun: "ML topics" },
    { kind: "cml", list: "CML", href: "classical.html", title: "Classical ML", type: "Classical ML", part: "ml", noun: "classical algorithms" },
    { kind: "dl", list: "DL", href: "dl.html", title: "Deep Learning", type: "Deep Learning", part: "dl", noun: "deep learning topics" },
    { kind: "tf", list: "TF", href: "transformers.html", title: "Transformers", type: "Transformers", part: "llm", noun: "transformer topics" },
    { kind: "llm", list: "LLM", href: "llm.html", title: "LLMs", type: "LLMs", part: "llm", noun: "LLM topics" },
    { kind: "ag", list: "AG", href: "agents.html", title: "Agentic AI", type: "Agents", part: "llm", noun: "agent topics" },
  ],
  parts: { found: "Foundations", ml: "Machine Learning", dl: "Deep Learning", llm: "LLMs & Agents" },
};
/* this site's own localStorage keys get a prefix, so they never mix with the DSA site's */
window.lsKey = (k) => (SITE.prefix || "") + k;

/* Data containers filled by data/*.js */
SITE.docs.forEach((d) => (window[d.list] = window[d.list] || []));
window.PRAC = window.PRAC || [];
window.BRIEFS = window.BRIEFS || {};

/* Practice categories, in the suggested order, with the part of the course each belongs to */
window.PRAC_CATS = [
  "NumPy Basics", "Linear Algebra", "Calculus & Probability", "Statistics",
  "Regression", "Classification", "Clustering & PCA", "Evaluation", "Classical ML",
  "Neural Net Basics", "Training", "CNNs & RNNs",
  "Tokenization", "Attention", "Transformer Blocks", "Decoding & Efficiency", "RAG", "Agents",
];
window.PRAC_PART = {
  "NumPy Basics": "found", "Linear Algebra": "found", "Calculus & Probability": "found", "Statistics": "found",
  "Regression": "ml", "Classification": "ml", "Clustering & PCA": "ml", "Evaluation": "ml", "Classical ML": "ml",
  "Neural Net Basics": "dl", "Training": "dl", "CNNs & RNNs": "dl",
  "Tokenization": "llm", "Attention": "llm", "Transformer Blocks": "llm", "Decoding & Efficiency": "llm", "RAG": "llm", "Agents": "llm",
};
/* one small line icon per practice category (24×24, drawn with the current text colour) */
window.PRAC_ICONS = {
  "NumPy Basics": '<rect x="3" y="3" width="7.5" height="7.5" rx="1.5"/><rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5"/><rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5"/><rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5"/>',
  "Linear Algebra": '<path d="M4 20 20 4"/><path d="M14 4h6v6"/><path d="M4 20V9"/><path d="m1.8 11.2 2.2-2.2 2.2 2.2"/>',
  "Calculus & Probability": '<path d="M3 19c3 0 3.5-14 9-14s6 14 9 14"/><path d="M3 21h18"/>',
  "Statistics": '<path d="M3 20.5h18"/><path d="M4 20c3-1 4-13 8-13s5 12 8 13"/><path d="M12 7v13.5" stroke-dasharray="2 2"/>',
  "Regression": '<path d="M3 20.5h18M3.5 3v17.5"/><path d="M5 17 20 6"/><circle cx="8" cy="12" r="1.3"/><circle cx="12.5" cy="13" r="1.3"/><circle cx="16" cy="7.5" r="1.3"/>',
  "Classification": '<path d="M4 20 20 4" stroke-dasharray="2.5 2.5"/><circle cx="7" cy="8" r="1.8"/><circle cx="10" cy="5" r="1.8"/><rect x="13" y="14" width="3.4" height="3.4" rx=".6"/><rect x="16.5" y="17.5" width="3.4" height="3.4" rx=".6"/>',
  "Clustering & PCA": '<circle cx="7" cy="8" r="4"/><circle cx="16.5" cy="15.5" r="4.5"/><circle cx="6.2" cy="7.2" r=".9"/><circle cx="8" cy="9" r=".9"/><circle cx="15.5" cy="14.5" r=".9"/><circle cx="17.8" cy="16.6" r=".9"/>',
  "Evaluation": '<rect x="3" y="3" width="18" height="18" rx="2.5"/><path d="M3 12h18M12 3v18"/><path d="m5.5 7.5 1.5 1.5 3-3"/><path d="m14.5 15.5 3 3M17.5 15.5l-3 3"/>',
  "Classical ML": '<path d="M3 19c2.5 0 3-9 6-9s3.5 6 6 6 3-11 6-11"/><circle cx="9" cy="10" r="1.3"/><circle cx="15" cy="16" r="1.3"/>',
  "Neural Net Basics": '<circle cx="5" cy="7" r="2"/><circle cx="5" cy="17" r="2"/><circle cx="12" cy="12" r="2.3"/><circle cx="19" cy="12" r="2"/><path d="m6.8 7.9 3.2 2.8M6.8 16.1l3.2-2.8M14.3 12H17"/>',
  "Training": '<path d="M3 4v16.5H21"/><path d="M5.5 6c2 8 5 11 15 12"/><circle cx="20.5" cy="18" r="1.2"/>',
  "CNNs & RNNs": '<rect x="3" y="3" width="12" height="12" rx="1.5"/><rect x="9" y="9" width="12" height="12" rx="1.5"/><path d="M9 3v6M3 9h6"/>',
  "Tokenization": '<rect x="2.5" y="8" width="6" height="8" rx="1.8"/><rect x="9.5" y="8" width="5" height="8" rx="1.8"/><rect x="15.5" y="8" width="6" height="8" rx="1.8"/>',
  "Attention": '<circle cx="5" cy="5" r="1.8"/><circle cx="5" cy="12" r="1.8"/><circle cx="5" cy="19" r="1.8"/><circle cx="19" cy="12" r="2.4"/><path d="M6.8 5.6 16.8 11M6.8 12h9.8M6.8 18.4l10-5.4"/>',
  "Transformer Blocks": '<rect x="4" y="3" width="16" height="5" rx="1.5"/><rect x="4" y="10" width="16" height="5" rx="1.5"/><path d="M12 15v6M9 18l3 3 3-3"/>',
  "Decoding & Efficiency": '<path d="M12 3v5"/><path d="M12 8 6 14M12 8l6 6M6 14v6M18 14l-3 6M18 14l3 6"/>',
  "RAG": '<ellipse cx="9" cy="6" rx="6" ry="2.5"/><path d="M3 6v9c0 1.4 2.7 2.5 6 2.5M15 6v4"/><path d="M3 10.5c0 1.4 2.7 2.5 6 2.5"/><circle cx="16.5" cy="16.5" r="3.5"/><path d="m19 19 2.5 2.5"/>',
  "Agents": '<rect x="4" y="8" width="16" height="11" rx="3"/><path d="M12 4v4M9 4h6"/><circle cx="9" cy="13.5" r="1.2"/><circle cx="15" cy="13.5" r="1.2"/><path d="M2 12v3M22 12v3"/>',
};
window.pracIcon = (cat, cls = "") => window.PRAC_ICONS[cat] ? `<svg class="cat-ico ${cls}" viewBox="0 0 24 24" aria-hidden="true">${window.PRAC_ICONS[cat]}</svg>` : "";
/* where each category sits on the Practice roadmap (map units, 760 × 740) and the suggested order between them */
window.PRAC_MAP = {
  nodes: {
    "NumPy Basics": [380, 40], "Linear Algebra": [240, 125], "Calculus & Probability": [520, 125], "Statistics": [680, 210],
    "Clustering & PCA": [110, 210], "Regression": [380, 210], "Classification": [250, 295], "Neural Net Basics": [530, 295],
    "Evaluation": [100, 380], "Classical ML": [90, 465], "Training": [430, 380], "CNNs & RNNs": [660, 380], "Tokenization": [260, 465], "Attention": [500, 465],
    "Transformer Blocks": [380, 550], "Decoding & Efficiency": [230, 635], "RAG": [530, 635], "Agents": [380, 715],
  },
  edges: [["NumPy Basics", "Linear Algebra"], ["NumPy Basics", "Calculus & Probability"], ["Calculus & Probability", "Statistics"], ["Linear Algebra", "Clustering & PCA"], ["Linear Algebra", "Regression"],
    ["Calculus & Probability", "Regression"], ["Calculus & Probability", "Neural Net Basics"], ["Regression", "Classification"], ["Regression", "Neural Net Basics"],
    ["Clustering & PCA", "Evaluation"], ["Classification", "Evaluation"], ["Evaluation", "Classical ML"], ["Neural Net Basics", "Training"], ["Neural Net Basics", "CNNs & RNNs"],
    ["Training", "Tokenization"], ["Training", "Attention"], ["CNNs & RNNs", "Attention"], ["Tokenization", "Transformer Blocks"], ["Attention", "Transformer Blocks"],
    ["Transformer Blocks", "Decoding & Efficiency"], ["Transformer Blocks", "RAG"], ["Decoding & Efficiency", "Agents"], ["RAG", "Agents"]],
};
/* PRAC.add(category, [problems]) — called from data/prac/*.js */
window.PRAC.add = function (cat, list) {
  list.forEach((p) => { p.cat = cat; PRAC.push(p); });
};

/* Every trackable thing on the site, keyed the same way as Store items */
function siteIndex() {
  const idx = {};
  PRAC.forEach((p, i) => (idx["prac:" + p.id] = { type: "Problem", title: p.t, href: "practice.html#p/" + p.id, d: p.d, cat: p.cat, order: i, p }));
  SITE.docs.forEach((d) => (window[d.list] || []).forEach((s) => (idx[d.kind + ":" + s.id] = { type: d.type, title: s.title, href: d.href + "#" + s.id, cat: d.title + (s.group ? " · " + s.group : "") })));
  return idx;
}
/* a doc page's start-up: docPage("ml") draws the sidebar + sections of ML (pages are written by tools/pages.py) */
function docPage(kind) {
  const d = SITE.docs.find((x) => x.kind === kind);
  DocPage({ sections: window[d.list], kind, title: d.title, noun: d.noun });
}
function pracSorted() {
  return PRAC_CATS.flatMap((c) => PRAC.filter((p) => p.cat === c));
}
