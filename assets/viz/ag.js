/* Agentic AI: visualizers, diagrams and glossary terms.
   Loaded on every doc page and on Practice, after assets/viz.js. All counts shown are computed, not faked. */
(function () {
  const V = Viz.defs;

  /* ------------------------------------------------------------------ private helpers */
  const sum = (a) => a.reduce((s, v) => s + v, 0);
  const wrap = (W, H, inner) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">${inner}</svg>`;
  const T = (x, y, s, style = "", anchor = "middle") => `<text x="${x}" y="${y}" text-anchor="${anchor}" style="font-size:11.5px;fill:var(--muted);${style}">${esc(s)}</text>`;
  const TX = "fill:var(--text);";
  const MONO = "font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px;";
  const wrapText = (s, n) => { const out = []; let cur = ""; for (const w of String(s).split(/\s+/)) { if (cur && (cur + " " + w).length > n) { out.push(cur); cur = w; } else cur = cur ? cur + " " + w : w; } if (cur) out.push(cur); return out.length ? out : [""]; };
  const tint = (col, pct = 16) => `color-mix(in srgb, ${col} ${pct}%, var(--panel-2))`;
  const rect = (x, y, w, h, col, extra = "", pct = 16) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" style="fill:${tint(col, pct)};stroke:${col};stroke-width:1.5;${extra}"/>`;
  const defsArrow = (id) => `<defs><marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--muted)"/></marker></defs>`;
  const line = (x1, y1, x2, y2, id, col = "var(--muted)", dash = "", w = 1.6) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" style="stroke:${col};stroke-width:${w};${dash ? "stroke-dasharray:" + dash + ";" : ""}" marker-end="url(#${id})"/>`;
  const approxTokens = (s) => Math.max(1, Math.round(String(s).length / 4));

  /* ================================================================== 1. The agent loop, step by step */
  const KIND = { system: ["System prompt", "var(--purple)"], tools: ["Tool definitions", "var(--muted)"], user: ["User task", "var(--amber)"], call: ["Model: tool call", "var(--accent)"], obs: ["Tool result", "var(--green)"], final: ["Model: final answer", "var(--accent)"] };
  V.ag_loop = {
    title: "The agent loop: think, act, observe, with the context growing",
    params: [
      { id: "mult", label: "tool result size (x)", min: 1, max: 60, step: 1, value: 1 },
      { id: "lim", label: "max loop iterations", min: 1, max: 4, step: 1, value: 4 },
    ],
    gen(_, P) {
      const WIN = 2000;
      const base = [["system", "You are a helpful agent. Use tools. Stop when you can answer.", 120], ["tools", "lookup(country, field), calculator(expression)", 180], ["user", "Population of France plus Germany, in millions?", 18]];
      const plan = [
        ["lookup(country='france', field='population_m')", "68"],
        ["lookup(country='germany', field='population_m')", "84"],
        ["calculator(expression='68 + 84')", "152"],
      ];
      const ctx = base.map(([k, t, n]) => ({ k, t, n }));
      let calls = 0, processed = 0;
      const S = [];
      const phases = ["Think", "Act", "Observe"];
      const draw = (ph, headline, detail) => {
        const total = sum(ctx.map((m) => m.n));
        let s = defsArrow("agl");
        phases.forEach((p, i) => {
          const x = 30 + i * 150, on = ph === i;
          s += rect(x, 8, 120, 34, on ? "var(--accent)" : "var(--muted)", on ? "" : "opacity:.7", on ? 30 : 8) + T(x + 60, 30, p, TX + `font-weight:${on ? 700 : 500}`);
          if (i < 2) s += line(x + 122, 25, x + 148, 25, "agl");
        });
        s += `<path d="M390,44 C390,70 90,70 90,44" style="fill:none;stroke:var(--muted);stroke-width:1.4;stroke-dasharray:4 4" marker-end="url(#agl)"/>` + T(465, 29, ph === 3 ? "loop ends: final answer" : "repeat until done", "font-size:11px", "start");
        s += T(10, 78, headline, TX + "font-weight:700", "start");
        wrapText(detail, 92).slice(0, 3).forEach((l, i) => { s += T(10, 95 + i * 15, l, MONO + "fill:var(--text)", "start"); });
        /* stacked context bar */
        const bx = 10, bw = 600, by = 150, bh = 26, scale = bw / Math.max(WIN, total);
        s += T(10, by - 7, `Context sent to the model on the next call: ${total} tokens (toy window ${WIN})`, TX + "font-weight:600", "start");
        s += `<rect x="${bx}" y="${by}" width="${WIN * scale}" height="${bh}" rx="5" style="fill:var(--panel-2);stroke:var(--border)"/>`;
        let x = bx; ctx.forEach((m) => { const w = m.n * scale; s += `<rect x="${x}" y="${by}" width="${Math.max(0.5, w)}" height="${bh}" style="fill:${KIND[m.k][1]};fill-opacity:${m.k === "tools" ? 0.5 : 0.8};stroke:var(--panel);stroke-width:0.8"/>`; x += w; });
        if (total > WIN) s += `<rect x="${bx + WIN * scale}" y="${by - 3}" width="${(total - WIN) * scale}" height="${bh + 6}" style="fill:var(--red);fill-opacity:.55"/>` + T(bx + WIN * scale + 4, by + bh + 16, "over the window: the oldest content would have to be dropped", "fill:var(--red);font-size:11px", "start");
        else s += `<line x1="${bx + WIN * scale}" y1="${by - 4}" x2="${bx + WIN * scale}" y2="${by + bh + 4}" style="stroke:var(--muted);stroke-width:1.5"/>`;
        /* legend + message list */
        let ly = by + bh + 34; const seen = new Set();
        Object.entries(KIND).forEach(([k, [lab, col]], i) => { const lx = 10 + (i % 3) * 205, yy = ly + Math.floor(i / 3) * 16; s += `<rect x="${lx}" y="${yy - 9}" width="10" height="10" rx="2" style="fill:${col};fill-opacity:.8"/>` + T(lx + 15, yy, lab, "font-size:11px", "start"); seen.add(k); });
        ly += 44;
        ctx.slice(-7).forEach((m, i) => { s += `<rect x="10" y="${ly + i * 16 - 9}" width="8" height="10" style="fill:${KIND[m.k][1]};fill-opacity:.8"/>` + T(24, ly + i * 16, `${String(m.n).padStart(4, " ")} tok  ${m.t.length > 78 ? m.t.slice(0, 77) + "…" : m.t}`, MONO + "fill:var(--text)", "start"); });
        if (ctx.length > 7) s += T(24, ly - 14, `(${ctx.length - 7} older messages not listed)`, "font-size:10px", "start");
        return wrap(640, ly + Math.min(7, ctx.length) * 16 + 6, s);
      };
      const side = () => ({ "model calls": calls, "context tokens": sum(ctx.map((m) => m.n)), "tokens read so far": processed, messages: ctx.length });
      S.push({ svg: draw(-1, "Start: the context holds only the setup", "System prompt + tool definitions + the user's task. No tool has run yet."), side: side(),
        msg: "An agent is a **loop around a model**. Before anything happens the context holds three things: the rules (system prompt), the list of tools the model may call, and the task. Press Next to run the loop." });
      let stopped = false;
      for (let i = 0; i <= plan.length && !stopped; i++) {
        const last = i === plan.length;
        if (i >= P.lim && !last) {
          S.push({ svg: draw(-1, "Stopped: iteration limit reached", `The loop ran ${P.lim} time(s) and was cut off before the model could answer.`), side: side(),
            msg: `**Max-iterations guard.** The limit (${P.lim}) was hit before the answer was ready, so the run ends with no result. Every real agent needs a cap like this, plus a plan for what to do when it triggers (return partial work, escalate, or raise the cap).` });
          stopped = true; break;
        }
        calls++; processed += sum(ctx.map((m) => m.n));
        S.push({ svg: draw(0, `Iteration ${i + 1}: THINK (model call #${calls})`, `The model reads ALL ${sum(ctx.map((m) => m.n))} tokens of context again and decides what to do next.`), side: side(),
          msg: `**Think.** One model call. The model is stateless, so the whole context is sent every time: that is **${sum(ctx.map((m) => m.n))} tokens** now, and the total read so far is **${processed}**. This is why long runs get slow and expensive.` });
        if (last) {
          ctx.push({ k: "final", t: "France and Germany together have about 152 million people.", n: 20 });
          S.push({ svg: draw(3, "FINAL ANSWER: no tool call this time", ctx[ctx.length - 1].t), side: side(),
            msg: "The model answers in plain text instead of calling a tool. That is the **stop condition**: the loop ends because the model chose to stop (or a guard stopped it)." });
          break;
        }
        const [call, res] = plan[i];
        ctx.push({ k: "call", t: call, n: 30 });
        S.push({ svg: draw(1, "ACT: the model asks for a tool", call), side: side(),
          msg: `**Act.** The model does not run anything itself. It emits a structured request: \`${call}\`. Your program receives it.` });
        const size = Math.round(12 * P.mult);
        ctx.push({ k: "obs", t: `${res}${P.mult > 1 ? "  (+ padding: a verbose tool result)" : ""}`, n: size });
        S.push({ svg: draw(2, "OBSERVE: the result goes back into the context", `${call}  ->  ${res}`), side: side(),
          msg: `**Observe.** Your code ran the tool and appended the result (**${size} tokens**) to the context. ${P.mult >= 20 ? "Verbose tool output eats the window fast: return only what the model needs." : "Next the loop goes back to Think with this new fact in view."}` });
      }
      return S;
    },
  };

  /* ================================================================== 2. ReAct trace replay */
  const SCEN = {
    1: { name: "Normal run", q: "Which is bigger, France or Germany (population), and by how much?",
      lines: [["Thought", "I need both populations. Start with France."], ["Action", "lookup(country='france', field='population_m')"], ["Observation", "68"],
        ["Thought", "Now Germany."], ["Action", "lookup(country='germany', field='population_m')"], ["Observation", "84"],
        ["Thought", "84 is larger. The gap is 84 - 68; let the calculator do it."], ["Action", "calculator(expression='84 - 68')"], ["Observation", "16"],
        ["Final", "Germany is bigger, by about 16 million."]] },
    2: { name: "Tool error, then recovery", q: "How many people live in France and Germany together?",
      lines: [["Thought", "I need two populations, then add them."], ["Action", "lookup(country='france', field='population')"], ["Observation", "ERROR: unknown field 'population'. Valid fields: capital, population_m"],
        ["Thought", "The field is called population_m. Retry with that name."], ["Action", "lookup(country='france', field='population_m')"], ["Observation", "68"],
        ["Thought", "Same for Germany."], ["Action", "lookup(country='germany', field='population_m')"], ["Observation", "84"],
        ["Action", "calculator(expression='68 + 84')"], ["Observation", "152"],
        ["Final", "About 152 million people live in the two countries together."]] },
    3: { name: "Stuck in a loop", q: "What is the capital of Atlantis?",
      lines: [["Thought", "Search for it."], ["Action", "search(query='capital of Atlantis')"], ["Observation", "no results"],
        ["Thought", "Maybe the wording is off. Search again."], ["Action", "search(query='capital of Atlantis')"], ["Observation", "no results"],
        ["Thought", "Try once more."], ["Action", "search(query='capital of Atlantis')"], ["Observation", "no results"],
        ["Thought", "Try once more."], ["Action", "search(query='capital of Atlantis')"], ["Observation", "no results"],
        ["Final", "(never reached: the same failing call repeats until a guard stops it)"]] },
  };
  const RCOL = { Thought: "var(--purple)", Action: "var(--accent)", Observation: "var(--green)", Final: "var(--amber)", Stop: "var(--red)" };
  V.ag_react = {
    title: "ReAct trace: Thought, Action, Observation, repeat",
    params: [
      { id: "scen", label: "scenario (1 normal, 2 recovers, 3 stuck)", min: 1, max: 3, step: 1, value: 1 },
      { id: "cap", label: "max tool calls", min: 1, max: 6, step: 1, value: 5 },
    ],
    gen(_, P) {
      const sc = SCEN[P.scen], S = [];
      let acts = 0, errs = 0, repeats = 0; const seen = new Set();
      const shown = [];
      const draw = () => {
        let s = T(10, 16, "Question: " + sc.q, TX + "font-weight:700", "start");
        let y = 38;
        shown.forEach(([k, t]) => {
          const ls = wrapText(t, 78);
          s += `<rect x="10" y="${y - 12}" width="4" height="${ls.length * 15 + 2}" rx="2" style="fill:${RCOL[k]}"/>` + T(22, y, k + ":", `fill:${RCOL[k]};font-weight:700;font-size:11.5px`, "start");
          ls.forEach((l, i) => { s += T(110, y + i * 15, l, MONO + "fill:var(--text)", "start"); });
          y += ls.length * 15 + 7;
        });
        return wrap(640, Math.max(90, y + 6), s);
      };
      const side = () => ({ "tool calls": acts, errors: errs, "repeated calls": repeats, scenario: sc.name });
      S.push({ svg: draw(), side: side(), msg: `Scenario **${sc.name}**. In ReAct the model alternates a short **Thought** (reasoning in words), an **Action** (a tool call) and an **Observation** (the tool's reply, written by your code, not by the model). Press Next to add one line at a time.` });
      let stopped = false;
      sc.lines.forEach(([k, t]) => {
        if (stopped) return;
        if (k === "Action") {
          if (acts >= P.cap) { shown.push(["Stop", `Tool-call limit (${P.cap}) reached. The agent is cut off here and must return what it has.`]); S.push({ svg: draw(), side: side(), msg: `**Guard triggered.** After ${P.cap} tool calls the harness stops the loop. ${P.scen === 3 ? "Without this cap the agent above would repeat the same failing search forever." : "Raise the cap if the task genuinely needs more steps."}` }); stopped = true; return; }
          acts++; if (seen.has(t)) repeats++; seen.add(t);
        }
        if (k === "Observation" && /^ERROR|no results/.test(t)) errs++;
        shown.push([k, t]);
        const note = { Thought: "The model writes its reasoning as text. This is cheap, makes the next step more deliberate, and is what you read when debugging.",
          Action: repeats ? "**The same call again.** An identical action after an identical failure is the classic loop. A guard should notice this." : "The model picks a tool and arguments. Your code, not the model, validates and runs it.",
          Observation: /^ERROR/.test(t) ? "The tool returned a **useful error message** (it lists the valid fields). That lets the model fix its own mistake. A bare 'failed' would not." : /no results/.test(t) ? "An empty result. A good tool says what to try instead; this one does not, so the model has nothing new to react to." : "The tool result is appended to the context as an observation. The model has not 'seen' this fact until now.",
          Final: "No more actions: the model writes the answer and the loop ends." }[k];
        S.push({ svg: draw(), side: side(), msg: note });
      });
      return S;
    },
  };

  /* ================================================================== 3. Multi-agent patterns */
  const PAT = [
    { name: "Routing", best: "Distinct request types needing different prompts or tools", calls: "2 (router + handler)",
      nodes: { in: [20, 100, 90, 40, "Request", "var(--amber)"], r: [160, 100, 100, 40, "Router (LLM)", "var(--purple)"], a: [330, 20, 130, 36, "Billing agent", "var(--accent)"], b: [330, 102, 130, 36, "Tech support agent", "var(--accent)"], c: [330, 184, 130, 36, "Refund agent", "var(--accent)"], out: [520, 100, 90, 40, "Answer", "var(--green)"] },
      edges: [["in", "r"], ["r", "a"], ["r", "b"], ["r", "c"], ["a", "out"], ["b", "out"], ["c", "out"]],
      steps: [[["in"], [], "A request arrives: *my card was charged twice*."], [["r"], [["in", "r"]], "The **router** is one cheap LLM call that only classifies the request. It does not try to solve it."], [["c"], [["r", "c"]], "Only the matching specialist runs, with its own focused prompt and tools. The other branches never start, so cost stays low."], [["out"], [["c", "out"]], "The specialist's answer is returned. Routing = **classify, then hand off**."]] },
    { name: "Parallelization", best: "Independent sub-tasks, or several opinions to vote on", calls: "N workers + 1 aggregator",
      nodes: { in: [20, 100, 90, 40, "Task", "var(--amber)"], a: [190, 20, 140, 36, "Worker: security", "var(--accent)"], b: [190, 102, 140, 36, "Worker: style", "var(--accent)"], c: [190, 184, 140, 36, "Worker: tests", "var(--accent)"], g: [390, 100, 110, 40, "Aggregate", "var(--purple)"], out: [540, 100, 80, 40, "Result", "var(--green)"] },
      edges: [["in", "a"], ["in", "b"], ["in", "c"], ["a", "g"], ["b", "g"], ["c", "g"], ["g", "out"]],
      steps: [[["in"], [], "One task, for example *review this pull request*."], [["a", "b", "c"], [["in", "a"], ["in", "b"], ["in", "c"]], "The task is split by aspect (**sectioning**) or sent to the same worker several times (**voting**). The workers do not talk to each other, so they run **at the same time**: latency is that of the slowest one."], [["g"], [["a", "g"], ["b", "g"], ["c", "g"]], "Plain code, or one more LLM call, combines the outputs: merge the findings, or take the majority vote."], [["out"], [["g", "out"]], "One merged result. The catch: all workers cost tokens even if one would have been enough."] ] },
    { name: "Orchestrator-workers", best: "Tasks whose sub-tasks are not known in advance", calls: "1 plan + N workers + 1 synthesis",
      nodes: { in: [20, 100, 90, 40, "Task", "var(--amber)"], o: [170, 100, 120, 40, "Orchestrator", "var(--purple)"], a: [350, 20, 130, 36, "Worker A", "var(--accent)"], b: [350, 102, 130, 36, "Worker B", "var(--accent)"], c: [350, 184, 130, 36, "Worker C", "var(--accent)"], out: [530, 100, 90, 40, "Report", "var(--green)"] },
      edges: [["in", "o"], ["o", "a"], ["o", "b"], ["o", "c"], ["a", "o2"], ["o", "out"]],
      steps: [[["in"], [], "A task like *fix the failing login test* : you cannot say in advance which files matter."], [["o"], [["in", "o"]], "The **orchestrator** (an LLM) reads the task and **decides the sub-tasks itself**. This is the difference from parallelization, where the split is fixed in code."], [["a", "b", "c"], [["o", "a"], ["o", "b"], ["o", "c"]], "It hands each sub-task to a worker (often a fresh agent with its own small context). Workers can run in parallel."], [["o"], [["a", "o"], ["b", "o"], ["c", "o"]], "Workers return short summaries. The orchestrator keeps only those, so its own context stays small."], [["out"], [["o", "out"]], "The orchestrator **synthesises** one report. It may also spawn more workers if something is still missing."]] },
    { name: "Evaluator-optimizer", best: "A clear quality bar and a draft that can be improved", calls: "2 per round, until accepted",
      nodes: { in: [20, 100, 90, 40, "Task", "var(--amber)"], g: [170, 40, 130, 40, "Generator", "var(--accent)"], e: [340, 40, 130, 40, "Evaluator", "var(--purple)"], out: [520, 100, 90, 40, "Accepted", "var(--green)"] },
      edges: [["in", "g"], ["g", "e"], ["e", "g2"], ["e", "out"]],
      steps: [[["in"], [], "A task with a checkable goal, for example a translation that must keep every number."], [["g"], [["in", "g"]], "The **generator** writes a first draft."], [["e"], [["g", "e"]], "The **evaluator** (a second prompt, or code such as unit tests) critiques it against a rubric: *two numbers were changed*. This is the same idea as [reflection](agents.html#reflection)."], [["g"], [["e", "g"]], "The critique goes back to the generator, which revises. Round 2 starts. Always cap the number of rounds."], [["e"], [["g", "e"]], "The evaluator checks the revision again."], [["out"], [["e", "out"]], "All checks pass, so the draft is accepted. If the cap were hit first, you would return the best draft with a warning."]] },
  ];
  V.ag_multi = {
    title: "Multi-agent patterns: four ways to wire LLM calls together",
    params: [{ id: "pat", label: "pattern (1 routing, 2 parallel, 3 orchestrator, 4 evaluator)", min: 1, max: 4, step: 1, value: 1 }],
    gen(_, P) {
      const pt = PAT[P.pat - 1], N = pt.nodes, S = [];
      const ctr = (id) => { const n = N[id]; return [n[0] + n[2] / 2, n[1] + n[3] / 2]; };
      const edgePath = (a, b, on, id) => {
        if (b === "o2" || b === "g2") return "";
        const A = N[a], B = N[b]; if (!A || !B) return "";
        const [ax, ay] = ctr(a), [bx, by] = ctr(b);
        const dir = bx >= ax ? 1 : -1;
        const x1 = dir > 0 ? A[0] + A[2] : A[0], x2 = dir > 0 ? B[0] : B[0] + B[2];
        let y1 = ay, y2 = by;
        if (pt.name === "Evaluator-optimizer" && a === "e" && b === "g") { y1 = A[1] + A[3]; y2 = B[1] + B[3]; return `<path d="M${ax},${y1} C${ax},${y1 + 50} ${bx},${y2 + 50} ${bx},${y2}" style="fill:none;stroke:${on ? "var(--accent)" : "var(--muted)"};stroke-width:${on ? 2.4 : 1.2};stroke-opacity:${on ? 1 : 0.4};stroke-dasharray:${on ? "" : "4 4"}" marker-end="url(#${id})"/>`; }
        return `<path d="M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}" style="fill:none;stroke:${on ? "var(--accent)" : "var(--muted)"};stroke-width:${on ? 2.4 : 1.2};stroke-opacity:${on ? 1 : 0.35}" marker-end="url(#${id})"/>`;
      };
      const draw = (act, edgeOn, round) => {
        let s = defsArrow("agm");
        const edges = pt.edges.slice();
        if (pt.name === "Evaluator-optimizer") { edges.length = 0; edges.push(["in", "g"], ["g", "e"], ["e", "g"], ["e", "out"]); }
        if (pt.name === "Orchestrator-workers") { edges.length = 0; edges.push(["in", "o"], ["o", "a"], ["o", "b"], ["o", "c"], ["o", "out"]); }
        const key = (e) => e[0] + ">" + e[1];
        const onSet = new Set(edgeOn.map(key));
        edges.forEach((e) => { s += edgePath(e[0], e[1], onSet.has(key(e)), "agm"); });
        if (pt.name === "Orchestrator-workers") edgeOn.forEach((e) => { if (e[1] === "o") { const W_ = N[e[0]], O_ = N.o, wy = W_[1] + W_[3] / 2 + 9, oy = O_[1] + O_[3] / 2 + (wy > O_[1] + O_[3] / 2 ? 8 : wy < O_[1] + O_[3] / 2 ? -8 : 0); s += `<path d="M${W_[0]},${wy} C${W_[0] - 50},${wy} ${O_[0] + O_[2] + 50},${oy} ${O_[0] + O_[2] + 2},${oy}" style="fill:none;stroke:var(--green);stroke-width:2.2;stroke-dasharray:5 3" marker-end="url(#agm)"/>`; } });
        Object.entries(N).forEach(([id, [x, y, w, h, lab, col]]) => { const on = act.includes(id); s += rect(x, y, w, h, col, on ? "" : "opacity:.5", on ? 34 : 10) + T(x + w / 2, y + h / 2 + 4, lab, TX + `font-weight:${on ? 700 : 500}`); });
        if (round) s += T(405, 112, round, "fill:var(--amber);font-weight:700");
        return wrap(640, pt.name === "Evaluator-optimizer" ? 190 : 240, s);
      };
      const side = (i) => ({ pattern: pt.name, "LLM calls": pt.calls, "best for": pt.best, step: `${i} / ${pt.steps.length}` });
      S.push({ svg: draw([], [], ""), side: side(0), msg: `**${pt.name}.** Best for: ${pt.best}. Boxes are LLM calls (or plain code); arrows are data. Press Next to follow one request through. Use the slider to switch pattern.` });
      pt.steps.forEach(([act, ed, m], i) => {
        const round = pt.name === "Evaluator-optimizer" ? (i >= 4 ? "round 2" : i >= 1 ? "round 1" : "") : "";
        S.push({ svg: draw(act, ed.map((e) => e.slice()), round), side: side(i + 1), msg: m });
      });
      return S;
    },
  };

  /* ================================================================== 4. MCP message sequence */
  V.ag_mcp = {
    title: "MCP: what the client and server actually say to each other",
    params: [{ id: "fail", label: "tool fails? (0 no, 1 yes)", min: 0, max: 1, step: 1, value: 0 }],
    gen(_, P) {
      const fail = P.fail === 1;
      const MSG = [
        { dir: 1, label: "initialize", kind: "request", note: "The client opens the session, says which protocol version it speaks and what it can do. (The version string here is an example; real clients and servers negotiate one they both support.)",
          json: ['{"jsonrpc": "2.0", "id": 1, "method": "initialize",', ' "params": {"protocolVersion": "2025-06-18",', '   "capabilities": {},', '   "clientInfo": {"name": "demo-client", "version": "0.1"}}}'] },
        { dir: -1, label: "result: capabilities", kind: "response", note: "The server answers with its own version, name and capabilities. Here it says it offers tools. Both sides now know what the other supports.",
          json: ['{"jsonrpc": "2.0", "id": 1,', ' "result": {"protocolVersion": "2025-06-18",', '   "capabilities": {"tools": {}},', '   "serverInfo": {"name": "weather-server", "version": "1.0"}}}'] },
        { dir: 1, label: "notifications/initialized", kind: "notification", note: "A notification has **no id** and no reply. It just tells the server: handshake finished, you may start working.",
          json: ['{"jsonrpc": "2.0", "method": "notifications/initialized"}'] },
        { dir: 1, label: "tools/list", kind: "request", note: "The client asks what tools exist. This is how a host learns the menu at run time, instead of having it hard-coded.",
          json: ['{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}'] },
        { dir: -1, label: "result: [get_weather]", kind: "response", note: "Each tool comes with a name, a description (the model reads it to decide when to call the tool) and a JSON Schema for its arguments. The host passes these to the model as tool definitions.",
          json: ['{"jsonrpc": "2.0", "id": 2, "result": {"tools": [{', '  "name": "get_weather",', '  "description": "Current weather for a city",', '  "inputSchema": {"type": "object",', '    "properties": {"city": {"type": "string"}},', '    "required": ["city"]}}]}}'] },
        { dir: 1, label: "tools/call get_weather", kind: "request", note: "The model chose this tool. The host sends the call to the server. The model never talks to the server directly; the host sits in between and can check or refuse the call.",
          json: ['{"jsonrpc": "2.0", "id": 3, "method": "tools/call",', ' "params": {"name": "get_weather",', '   "arguments": {"city": "' + (fail ? "Atlantis" : "Paris") + '"}}}'] },
        { dir: -1, label: fail ? "result: isError true" : "result: content", kind: "response", note: fail ? "The tool failed, but this is **not** a protocol error: the server returns a normal result with `isError: true` and a message the model can read and react to. Protocol-level problems (unknown method, bad JSON) use a separate JSON-RPC `error` object." : "The result is a list of content blocks (text here; images and other types exist). The host puts it into the context as a tool result, and the model continues.",
          json: fail ? ['{"jsonrpc": "2.0", "id": 3, "result": {', ' "content": [{"type": "text",', '   "text": "No city named Atlantis. Try a real city."}],', ' "isError": true}}'] : ['{"jsonrpc": "2.0", "id": 3, "result": {', ' "content": [{"type": "text", "text": "18C, cloudy"}],', ' "isError": false}}'] },
      ];
      const S = [], X0 = 130, X1 = 510, TOP = 54, GAP = 30;
      MSG.forEach((m, k) => {
        let s = defsArrow("agp");
        s += rect(60, 6, 140, 30, "var(--accent)", "", 22) + T(130, 26, "MCP client (in the host)", TX + "font-weight:700", "middle");
        s += rect(440, 6, 140, 30, "var(--purple)", "", 22) + T(510, 26, "MCP server", TX + "font-weight:700", "middle");
        const H = TOP + MSG.length * GAP + 4;
        s += `<line x1="${X0}" y1="38" x2="${X0}" y2="${H}" style="stroke:var(--border);stroke-width:2;stroke-dasharray:5 4"/><line x1="${X1}" y1="38" x2="${X1}" y2="${H}" style="stroke:var(--border);stroke-width:2;stroke-dasharray:5 4"/>`;
        for (let i = 0; i <= k; i++) {
          const mm = MSG[i], y = TOP + i * GAP, cur = i === k;
          const col = mm.kind === "response" ? "var(--green)" : mm.kind === "notification" ? "var(--amber)" : "var(--accent)";
          const xa = mm.dir > 0 ? X0 : X1, xb = mm.dir > 0 ? X1 - 2 : X0 + 2;
          s += `<line x1="${xa}" y1="${y}" x2="${xb}" y2="${y}" style="stroke:${col};stroke-width:${cur ? 2.6 : 1.4};stroke-opacity:${cur ? 1 : 0.55};${mm.kind === "response" ? "stroke-dasharray:6 3;" : ""}" marker-end="url(#agp)"/>`;
          s += T(320, y - 5, mm.label, `fill:${cur ? "var(--text)" : "var(--muted)"};font-weight:${cur ? 700 : 500};font-size:11.5px`);
        }
        const by = H + 8, bh = m.json.length * 14 + 16;
        s += `<rect x="20" y="${by}" width="600" height="${bh}" rx="8" style="fill:var(--panel-2);stroke:var(--border)"/>`;
        m.json.forEach((l, i) => { s += T(32, by + 18 + i * 14, l, MONO + "fill:var(--text)", "start"); });
        S.push({ svg: wrap(640, by + bh + 8, s), side: { message: `${k + 1} of ${MSG.length}`, type: m.kind, transport: "stdio or HTTP" }, msg: `**${m.label}** (${m.kind}). ${m.note}` });
      });
      return S;
    },
  };

  /* ================================================================== 5. Context window budget */
  V.ag_ctx = {
    title: "Context budget: what fits in the window",
    params: [
      { id: "win", label: "window size (k tokens)", min: 8, max: 200, step: 8, value: 64 },
      { id: "sys", label: "system prompt (k)", min: 0, max: 20, step: 1, value: 2 },
      { id: "tools", label: "tool definitions (k)", min: 0, max: 40, step: 1, value: 6 },
      { id: "hist", label: "conversation + tool results (k)", min: 0, max: 150, step: 2, value: 30 },
      { id: "docs", label: "retrieved documents (k)", min: 0, max: 100, step: 2, value: 12 },
      { id: "out", label: "reserved for the answer (k)", min: 0, max: 32, step: 1, value: 4 },
    ],
    gen(_, P) {
      const parts = [["System prompt", P.sys, "var(--purple)"], ["Tools", P.tools, "var(--muted)"], ["History", P.hist, "var(--accent)"], ["Retrieved docs", P.docs, "var(--green)"], ["Answer reserve", P.out, "var(--amber)"]];
      const total = sum(parts.map((p) => p[1])), over = Math.max(0, total - P.win), free = Math.max(0, P.win - total);
      const W = 600, x0 = 20, sc = W / Math.max(P.win, total, 1);
      let s = T(20, 18, `Window: ${P.win}k tokens   Used: ${total}k   ${over ? "OVER by " + over + "k" : "Free: " + free + "k"}`, `font-weight:700;fill:${over ? "var(--red)" : "var(--text)"};font-size:12.5px`, "start");
      const by = 34, bh = 40;
      s += `<rect x="${x0}" y="${by}" width="${P.win * sc}" height="${bh}" rx="6" style="fill:var(--panel-2);stroke:var(--border)"/>`;
      let x = x0;
      parts.forEach(([n, v, col]) => { const w = v * sc; if (w > 0) s += `<rect x="${x}" y="${by}" width="${w}" height="${bh}" style="fill:${col};fill-opacity:${over && x + w > x0 + P.win * sc ? 0.55 : 0.85};stroke:var(--panel);stroke-width:1"/>` + (w > 34 ? T(x + w / 2, by + 24, v + "k", "fill:var(--bg,#fff);font-weight:700;font-size:12px;mix-blend-mode:normal;fill:var(--panel)") : ""); x += w; });
      s += `<line x1="${x0 + P.win * sc}" y1="${by - 8}" x2="${x0 + P.win * sc}" y2="${by + bh + 8}" style="stroke:${over ? "var(--red)" : "var(--text)"};stroke-width:2.5"/>` + T(x0 + P.win * sc, by - 12, "window limit", `font-size:10.5px;fill:${over ? "var(--red)" : "var(--muted)"}`, "end");
      if (over) s += `<rect x="${x0 + P.win * sc}" y="${by}" width="${over * sc}" height="${bh}" style="fill:var(--red);fill-opacity:.35"/>` + T(x0 + P.win * sc + 4, by + bh + 20, "does not fit", "fill:var(--red);font-size:11px", "start");
      let ly = by + bh + 40;
      parts.forEach(([n, v, col], i) => { const lx = 20 + (i % 3) * 200, yy = ly + Math.floor(i / 3) * 18; s += `<rect x="${lx}" y="${yy - 10}" width="11" height="11" rx="2" style="fill:${col};fill-opacity:.85"/>` + T(lx + 17, yy, `${n}: ${v}k (${total ? Math.round((v / total) * 100) : 0}%)`, "font-size:11px", "start"); });
      const fixed = P.sys + P.tools + P.out, avail = P.win - fixed;
      const adv = over ? `Over by **${over}k**. Options: summarise or drop old history (${P.hist}k now), retrieve fewer or shorter documents, load fewer tools, or use a bigger window. Even when it fits, long contexts cost more per call and models use the middle less reliably ([long context](llm.html#long-context)).` : free < P.win * 0.1 ? `It fits, but only **${free}k** of headroom is left. The next few tool results will overflow it. Compact the history before that happens, not after.` : `Comfortable: **${free}k** free. The fixed part (system + tools + answer reserve) is ${fixed}k, leaving ${Math.max(0, avail)}k for history and documents.`;
      return [{ svg: wrap(640, ly + 44, s), side: { "used (k)": total, "free (k)": free, "over (k)": over, "history share": total ? Math.round((P.hist / total) * 100) + "%" : "0%" }, msg: adv }];
    },
  };

  /* ================================================================== static diagrams */
  const bx = (x, y, w, h, label, col, sub) => rect(x, y, w, h, col) + `<text x="${x + w / 2}" y="${y + h / 2 + (sub ? -2 : 4)}" text-anchor="middle" style="font-size:12.5px;font-weight:700;fill:var(--text)">${esc(label)}</text>` + (sub ? `<text x="${x + w / 2}" y="${y + h / 2 + 13}" text-anchor="middle" style="font-size:10.5px;fill:var(--muted)">${esc(sub)}</text>` : "");
  const svgw = (W, H, s) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">${s}</svg>`;

  D.ag_spectrum = () => {
    let s = defsArrow("sp");
    s += T(20, 18, "Code decides the path", "font-weight:700;fill:var(--text);font-size:12px", "start") + T(620, 18, "Model decides the path", "font-weight:700;fill:var(--text);font-size:12px", "end");
    s += `<line x1="20" y1="30" x2="620" y2="30" style="stroke:var(--muted);stroke-width:2" marker-end="url(#sp)"/>`;
    const cols = ["var(--muted)", "var(--accent)", "var(--accent)", "var(--purple)", "var(--amber)"];
    const items = [["Single call", "1 prompt, 1 answer", "classify, summarise"], ["Chain", "fixed steps in code", "extract, then translate"], ["Router / parallel", "code picks branches", "support triage"], ["Agent loop", "model picks tools + steps", "fix a failing test"], ["Multi-agent", "models delegate to models", "open-ended research"]];
    items.forEach((it, i) => { const x = 20 + i * 122; s += bx(x, 48, 112, 52, it[0], cols[i], it[1]) + T(x + 56, 118, "e.g. " + it[2], "font-size:10px"); });
    s += T(20, 150, "Predictability, speed, low cost", "font-size:11px;fill:var(--green);font-weight:700", "start") + T(620, 150, "Flexibility, higher cost, more ways to fail", "font-size:11px;fill:var(--red);font-weight:700", "end");
    s += T(320, 172, "Start at the left. Move right only when the task genuinely needs it.", "font-size:11.5px;font-style:italic");
    return svgw(640, 184, s);
  };

  D.ag_toolcall = () => {
    let s = defsArrow("tc");
    s += bx(10, 10, 130, 44, "Your app", "var(--accent)", "the harness / loop") + bx(260, 10, 130, 44, "Model API", "var(--purple)", "stateless") + bx(500, 10, 130, 44, "Real tool", "var(--green)", "your function");
    const row = (y, x1, x2, col, label, sub) => line(x1, y, x2, y, "tc", col, "", 2) + T((x1 + x2) / 2, y - 6, label, "font-weight:700;fill:var(--text)") + T((x1 + x2) / 2, y + 14, sub, "font-size:10px");
    s += row(86, 75, 325, "var(--accent)", "1. messages + tool definitions", "what tools exist, as JSON Schema");
    s += row(138, 325, 75, "var(--purple)", "2. reply: tool_use { name, arguments }", "the model only REQUESTS a call");
    s += row(190, 75, 565, "var(--accent)", "3. your code validates, then runs the function", "allow-list, argument checks, timeout");
    s += row(242, 565, 75, "var(--green)", "4. return value (or an error)", "plain data");
    s += row(294, 75, 325, "var(--accent)", "5. messages + tool_result", "the model reads the observation");
    s += row(346, 325, 75, "var(--purple)", "6. reply: final text (or another tool_use)", "loop again, or stop");
    for (const x of [75, 325, 565]) s += `<line x1="${x}" y1="58" x2="${x}" y2="362" style="stroke:var(--border);stroke-width:1.5;stroke-dasharray:3 4"/>`;
    return svgw(640, 372, s);
  };

  D.ag_memory = () => {
    let s = defsArrow("mm");
    s += rect(190, 8, 260, 120, "var(--accent)", "", 10) + T(320, 26, "Context window (short-term memory)", "font-weight:700;fill:var(--text);font-size:12.5px");
    s += bx(204, 38, 112, 36, "System + tools", "var(--purple)") + bx(324, 38, 112, 36, "Recent turns", "var(--accent)") + bx(204, 82, 112, 36, "Retrieved facts", "var(--green)") + bx(324, 82, 112, 36, "Scratchpad / plan", "var(--amber)");
    const ext = [["Episodic", "past runs and events", "vector store + time", 10], ["Semantic", "facts about user / world", "key-value or profile", 170], ["Procedural", "how to do things", "prompts, skills, rules", 330], ["Files / state", "work products", "disk, DB, notes file", 490]];
    ext.forEach(([a, b, c, x]) => { s += bx(x, 172, 140, 62, a, "var(--green)", b) + T(x + 70, 250, c, "font-size:10px"); s += line(x + 70, 170, 320, 130, "mm", "var(--muted)", "4 3", 1.2); });
    s += T(320, 278, "Long-term memory (outside the window; fetched on demand)", "font-weight:700;fill:var(--text);font-size:12px");
    s += T(320, 298, "write: save a note or summary    read: search by similarity, key or recency, then paste the hits into context", "font-size:11px");
    return svgw(640, 308, s);
  };

  D.ag_guard = () => {
    let s = defsArrow("gl2");
    const L = [["1  Input checks", "injection hints, size, PII", "var(--purple)"], ["2  Model + prompt", "rules, refusal training", "var(--accent)"], ["3  Tool allow-list", "only listed tools, validated args", "var(--green)"], ["4  Sandbox + least privilege", "read-only, no network, small blast radius", "var(--amber)"], ["5  Human approval", "for risky or irreversible actions", "var(--red)"], ["6  Output checks + audit log", "redact secrets, log every call", "var(--muted)"]];
    L.forEach((l, i) => { const y = 8 + i * 48; s += rect(120, y, 400, 40, l[2]) + T(134, y + 17, l[0], "font-weight:700;fill:var(--text);font-size:12.5px", "start") + T(134, y + 32, l[1], "font-size:10.5px", "start"); if (i < 5) s += line(320, y + 41, 320, y + 47, "gl2", "var(--muted)", "", 1.4); });
    s += T(60, 130, "request", "font-size:11px") + line(14, 138, 118, 138, "gl2", "var(--text)", "", 2) + T(580, 130, "effect", "font-size:11px") + line(522, 138, 628, 138, "gl2", "var(--text)", "", 2);
    s += T(320, 306, "Each layer assumes the one before it can fail. No single layer is enough.", "font-size:11.5px;font-style:italic");
    return svgw(640, 318, s);
  };

  /* ================================================================== glossary (existing keys are kept) */
  const G = {
    "agent loop": "The cycle an agent runs: the model reads its context, asks for a tool, your code runs it, the result goes back into the context, and this repeats until the model answers or a limit is hit.",
    "tool use": "Letting a model request calls to functions you provide, such as search or a calculator, and read their results.",
    "tool schema": "A JSON Schema that describes a tool: its name, what it does and the arguments it accepts. The model reads it to decide when and how to call the tool.",
    "react": "Reason + Act: a prompting pattern where the model alternates a written Thought, a tool Action and an Observation until it can answer.",
    "observation": "The result of a tool call, written by your code and appended to the context for the model to read.",
    "scratchpad": "Space where the model writes intermediate reasoning or notes before the final answer.",
    "stop condition": "The rule that ends an agent loop: the model gives a final answer, or a guard such as a step limit or budget triggers.",
    "max steps": "A hard cap on loop iterations or tool calls so an agent cannot run forever.",
    "plan-and-execute": "An agent design where the model first writes a plan, then a separate step executes the plan item by item, with optional re-planning.",
    "reflection": "Having the model critique its own output (or a tool or test result) and then revise it.",
    "short-term memory": "What the agent holds in its context window right now: recent messages and tool results.",
    "long-term memory": "Information stored outside the context window, such as in a vector store or database, and retrieved when needed.",
    "episodic memory": "Memory of specific past events or runs, for example 'last time this deploy failed because of X'.",
    "semantic memory": "Stored facts about the user or the world, such as preferences or a profile.",
    "procedural memory": "Stored knowledge of how to do things: instructions, skills, rules.",
    "context engineering": "Deciding what goes into the model's context on each step (instructions, tools, memory, retrieved data) and what stays out.",
    "compaction": "Replacing old parts of a long context with a shorter summary so the agent can keep working within the window.",
    "agentic rag": "RAG where the model decides when and what to retrieve, may search several times, and checks whether the results are good enough.",
    "orchestrator": "An LLM that breaks a task into sub-tasks, hands them to worker agents, and combines their results.",
    "subagent": "A worker agent started by another agent, usually with its own fresh context, that returns a short result.",
    "routing": "Using one cheap model call to classify a request and send it to the right specialised prompt or agent.",
    "json-rpc": "A small message format using JSON: requests with an id, method and params; responses with a result or error. MCP is built on it.",
    "mcp server": "A program that exposes tools, resources or prompts to AI applications using the Model Context Protocol.",
    "mcp client": "The part of an AI application that connects to an MCP server, lists its tools and forwards calls to it.",
    "guardrail": "A check around an agent that blocks or flags unsafe input, tool calls or output.",
    "human-in-the-loop": "Pausing the agent so a person can approve, edit or reject an action before it happens.",
    "least privilege": "Giving an agent only the permissions and tools it needs for the task, nothing more.",
    "sandbox": "An isolated environment, with limited files, network and permissions, where agent actions run so mistakes cannot spread.",
    "data exfiltration": "Getting private data out of a system, for example by tricking an agent into sending it to an attacker.",
    "confused deputy": "A trusted program that is tricked into using its own authority on behalf of an attacker.",
    "trajectory": "The full sequence of an agent run: every thought, tool call and observation, in order.",
    "tool-call accuracy": "How often the agent called the right tool with the right arguments, compared with a reference trajectory.",
    "idempotent": "Safe to repeat: doing it twice has the same effect as doing it once. Important for retried tool calls.",
    "exponential backoff": "Waiting longer after each failed attempt (1s, 2s, 4s, ...) before retrying, usually with random jitter added.",
    "computer use": "A model controlling a computer through screenshots, mouse and keyboard actions, like a person would.",
    "coding agent": "An agent that reads, edits and runs code in a repository, using tools such as file editing, search and a shell.",
    "workflow": "A system where LLM calls follow a path fixed by your code, as opposed to an agent where the model chooses the path.",
    "tool poisoning": "Hiding malicious instructions in a tool's description or output so the model follows them.",
  };
  const have = window.GLOSSARY || (window.GLOSSARY = {});
  Object.keys(G).forEach((k) => { if (!(k in have)) have[k] = G[k]; });
})();
