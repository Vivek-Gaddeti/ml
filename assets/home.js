/* home.js: the reading home (index.html). Subjects grouped by part, each with its section count and a "Continue" link.
   Settings (touch layout, backup) sit in a small disclosure at the bottom. Data: data/progress.js (titles only). */
(function () {
  const main = $("#content"), PR = window.PROGRESS;
  if (!main || !PR) return;
  const SUB = { math: "Vectors, gradients, probability", pyml: "NumPy, pandas, plots, scikit-learn", stats: "Hypothesis tests, CIs, A/B tests, bootstrap",
    ml: "Workflow, models, evaluation", cml: "LDA, EM, HMMs, kernels, MCMC", dl: "Neurons to CNNs, RNNs and backprop",
    tf: "Tokens, attention, the transformer block", llm: "Training, decoding, prompting, RAG", ag: "Tools, ReAct, memory, MCP, evals" };
  function render() {
    const parts = Object.entries(SITE.parts).map(([pt, name]) => {
      const cards = SITE.docs.filter((d) => d.part === pt).map((d) => {
        const secs = PR.docs[d.kind] || [], nx = secs.find((x) => Store.item(d.kind + ":" + x.id).status !== "done");
        const done = secs.filter((x) => Store.item(d.kind + ":" + x.id).status === "done").length;
        return `<div class="card home-card">
          <a class="home-t" href="${d.href}"><h3>${esc(d.title)}</h3></a>
          <p class="small muted">${SUB[d.kind] || ""}</p>
          <div class="home-n num">${secs.length} sections${done ? ` · ${done} read` : ""}</div>
          <a class="btn sm home-go" href="${nx ? d.href + "#" + nx.id : d.href}">${nx ? (done ? "Continue: " : "Start: ") + esc(nx.title) : "All read. Open again"}</a>
        </div>`;
      }).join("");
      return `<section class="home-part"><div class="label">${esc(name)}</div><div class="home-grid">${cards}</div></section>`;
    }).join("");
    const t = Store.get("touch") || "";
    main.innerHTML = `
      <div class="home-hero"><div class="home-logo">DC<i>.</i></div><h1>Daily Checker</h1><p class="lead">Read, learn, one chapter a day.</p></div>
      ${parts}
      <details class="home-set"><summary>Settings</summary>
        <div class="home-set-b">
          <div class="label">Touch layout</div>
          <p class="small muted">Bottom bar, bigger buttons and a scratch pad on every page. Auto turns it on for tablets and phones.</p>
          <div class="seg-row" role="group" aria-label="Touch layout">${[["", "Auto"], ["on", "On"], ["off", "Off"]].map(([v, l]) => `<button type="button" class="btn sm" data-touch="${v}" aria-pressed="${t === v}">${l}</button>`).join("")}</div>
          <div class="label" style="margin-top:14px">Backup</div>
          <p class="small muted">Your reading progress and notes are saved only in this browser. Export now and then, and import on another device.</p>
          <div class="btn-row"><button type="button" class="btn" data-do="export">${icon("download")} Backup</button><label class="btn">${icon("upload")} Import<input type="file" data-do="import" accept=".json,application/json" hidden></label></div>
        </div></details>`;
  }
  main.addEventListener("click", (e) => {
    const tp = e.target.closest("[data-touch]");
    if (tp) { Store.set("touch", tp.dataset.touch || undefined); location.reload(); return; }
    const d = e.target.closest("[data-do]");
    if (d && d.dataset.do === "export") Store.exportJSON();
  });
  main.addEventListener("change", (e) => { if (e.target.matches('[data-do="import"]') && e.target.files[0]) Store.importJSON(e.target.files[0], render); });
  document.addEventListener("store:change", () => { if (!main.querySelector("details[open]")) render(); });
  render();
})();
