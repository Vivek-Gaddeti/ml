/* py.js: run Python right on the page (real CPython 3.12 compiled for the browser: Pyodide).
   • PyRun   starts Python in a background worker the first time it's needed (downloads ~10 MB once, then cached),
             runs code with a fresh namespace each time, and can be stopped (an endless loop never freezes the page).
   • Editor  a small code editor: highlighting, line numbers, Tab / Shift+Tab, auto-indent, Ctrl+/ comment, Ctrl+Enter run.
   • ✏️ Edit & run on every Python example, 🐍 playground panel (header button or P), and "Try it yourself" on
     Practice problems, which runs your solution on the problem's tests and compares line by line.
   • numpy / pandas / scikit-learn / matplotlib load automatically when the code imports them (a few MB, once),
     and every matplotlib figure the code draws is shown under the output as a picture.
   Needs internet the first time Python loads; everything else on the site works offline.                       */
(function () {
  const BASE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
  const ICO = (d, cls = "cat-ico") => `<svg class="${cls}" viewBox="0 0 24 24" aria-hidden="true">${d}</svg>`;
  const I = {
    play: '<path d="M7 4.5v15l12-7.5z"/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    reset: '<path d="M4 12a8 8 0 1 0 2.4-5.7L4 8.6"/><path d="M4 4v4.6h4.6"/>',
    term: '<rect x="2.5" y="4" width="19" height="16" rx="2.5"/><path d="m7 9 3 3-3 3M12.5 15H17"/>',
    close: '<path d="M6 6l12 12M18 6 6 18"/>',
    step: '<path d="M5 5v14"/><path d="M9 5.5v13l10-6.5z"/>',
    edit: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13.5 6.5 4 4"/>',
    out: '<path d="M14 4h6v6M20 4l-8 8"/><path d="M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4"/>',
  };
  const ls = { get: (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} },
    del: (k) => { try { localStorage.removeItem(k); } catch (e) {} } };

  /* ================================================================ PyRun: Python in a worker */
  // runs inside the worker. Kept as a real function so it reads like code; it's turned into a Blob worker below
  // (a Blob works even when the page is opened as a file, where a worker file would be blocked).
  function workerMain(BASE, WRAP) {
    importScripts(BASE + "pyodide.js");
    let py = null, traceReady = false;
    const pkgMsg = (m) => postMessage({ type: "pkg", msg: String(m) });          // "Loading numpy…" while a package downloads
    const ready = loadPyodide({ indexURL: BASE }).then(async (p) => { py = p; await py.runPythonAsync(WRAP); postMessage({ type: "ready", version: py.version }); })
      .catch((e) => postMessage({ type: "fail", err: String((e && e.message) || e) }));
    onmessage = async (e) => {
      const d = e.data;
      await ready;
      if (!py) return;
      if (d.kind === "trace") {
        let r;
        try {
          await py.loadPackagesFromImports([d.job.pre || "", d.job.code].join("\n"), { messageCallback: pkgMsg }).catch(() => {});
          if (!traceReady) {
            py.globals.set("_trace_src", d.src);
            await py.runPythonAsync('import json\n_trace_ns = {"__name__": "site_trace"}\nexec(_trace_src, _trace_ns)\ndef _site_trace(job):\n    return json.dumps(_trace_ns["run"](json.loads(job)))');
            traceReady = true;
          }
          const f = py.globals.get("_site_trace"), out = f(JSON.stringify(d.job));
          f.destroy();
          r = { trace: JSON.parse(out) };
        } catch (x) { r = { traceErr: String((x && x.message) || x) }; }
        postMessage(Object.assign({ type: "done", id: d.id }, r));
        return;
      }
      try { await py.loadPackagesFromImports([d.pre, d.code, d.after || ""].join("\n"), { messageCallback: pkgMsg }); } catch (x) {}   // e.g. numpy, if you import it
      const t = performance.now();
      let r;
      try {
        const f = py.globals.get("_site_run"), res = f(d.code, d.stdin || "", d.pre || "", d.after == null ? null : d.after);
        r = res.toJs({ dict_converter: Object.fromEntries });
        res.destroy(); f.destroy();
      } catch (x) { r = { out: "", err: String(x), test: null, test_err: "", line: null }; }
      r.ms = performance.now() - t;
      postMessage(Object.assign({ type: "done", id: d.id }, r));
    };
  }
  // runs every job: helpers (silent), your code (its prints), then the tests (their prints, compared to the expected output)
  const WRAP = String.raw`
import sys, io, os, base64, warnings, traceback, contextlib
os.environ.setdefault("MPLBACKEND", "AGG")       # draw figures off-screen; we send them back as PNG pictures
warnings.simplefilter("ignore")                # library warnings (convergence notes…) would clutter the output; the shown outputs never include them
import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)   # no "building the font cache" note in your output
_LIMIT = 200_000
def _site_figs():
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return []
    out = []
    for n in plt.get_fignums()[:6]:
        b = io.BytesIO()
        plt.figure(n).savefig(b, format="png", dpi=96, bbox_inches="tight")
        out.append(base64.b64encode(b.getvalue()).decode())
    plt.close("all")
    return out
def _site_run(src, stdin, pre, after):
    ns = {"__name__": "__main__"}
    sys.stdin = io.StringIO(stdin)
    res = {"out": "", "err": "", "test": None, "test_err": "", "line": None, "imgs": []}
    def go(code, fname):
        buf, ok = io.StringIO(), True
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(compile(code, fname, "exec"), ns)
            except SystemExit:
                pass
            except BaseException as e:
                ok = False
                tb = e.__traceback__
                while tb is not None and tb.tb_frame.f_code.co_filename not in ("<your code>", "<tests>"):
                    tb = tb.tb_next
                err = "".join(traceback.format_exception(type(e), e, tb))
                if isinstance(e, SyntaxError) and e.filename == "<your code>":
                    res["line"] = e.lineno
                else:
                    t = e.__traceback__
                    while t is not None:
                        if t.tb_frame.f_code.co_filename == "<your code>": res["line"] = t.tb_lineno
                        t = t.tb_next
                return buf.getvalue()[:_LIMIT], err, ok
        return buf.getvalue()[:_LIMIT], "", ok
    if pre:
        exec(compile(pre, "<helpers>", "exec"), ns)
    res["out"], res["err"], ok = go(src, "<your code>")
    if ok and after is not None:
        res["test"], res["test_err"], _ = go(after, "<tests>")
    try:
        res["imgs"] = _site_figs()
    except Exception:
        pass
    return res
`;
  const PyRun = (() => {
    let worker = null, ready = null, failBoot = null, seq = 0, state = "off", chain = Promise.resolve(), pkg = "";
    const pending = new Map(), subs = new Set();
    const set = (s) => { state = s; subs.forEach((f) => f(s)); };
    function boot() {
      if (ready) return ready;
      const src = `(${workerMain})(${JSON.stringify(BASE)}, ${JSON.stringify(WRAP)});`;
      worker = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
      set("loading");
      ready = new Promise((res, rej) => {
        failBoot = rej;
        worker.onmessage = (e) => {
          const d = e.data;
          if (d.type === "ready") { failBoot = null; set("ready"); res(); }
          else if (d.type === "fail") { kill(); rej(new Error(d.err)); }
          else if (d.type === "done") { const p = pending.get(d.id); if (p) { pending.delete(d.id); set("ready"); p(d); } }
          else if (d.type === "pkg") { const m = d.msg.match(/^Loading (.+)/); if (m) { pkg = m[1].replace(/\s*from .*$/, ""); set("packages"); } else if (state === "packages") set("running"); }
        };
        worker.onerror = (e) => { e.preventDefault(); kill(); rej(new Error(e.message || "Python could not start")); };
      });
      return ready;
    }
    function kill() {
      if (worker) worker.terminate();
      const f = failBoot;
      worker = null; ready = null; failBoot = null;
      if (f) f(new Error("stopped"));                                     // stopped while Python was still loading
      pending.forEach((p) => p({ stopped: true })); pending.clear();
      set("off");
    }
    function run(code, opts = {}) {
      const job = chain.then(() => boot()).then(() => new Promise((res) => {
        const id = ++seq;
        pending.set(id, res); set("running");
        worker.postMessage({ id, code, stdin: opts.stdin || "", pre: opts.pre || "", after: opts.after == null ? null : opts.after });
      }));
      chain = job.catch(() => {});
      return job;
    }
    // step-through: run the code under the same tracer the site's examples use (data/trace-src.js ← tools/trace.py)
    let traceSrc = null;
    const loadTraceSrc = () => traceSrc || (traceSrc = window.TRACE_PY ? Promise.resolve(window.TRACE_PY) : new Promise((res, rej) => {
      const sc = document.createElement("script");
      sc.src = "data/trace-src.js"; sc.charset = "utf-8";
      sc.onload = () => (window.TRACE_PY ? res(window.TRACE_PY) : rej(new Error("tracer missing")));
      sc.onerror = () => { traceSrc = null; rej(new Error("Couldn't load data/trace-src.js")); };
      document.head.appendChild(sc);
    }));
    function trace(code, opts = {}) {
      const job = chain.then(() => Promise.all([boot(), loadTraceSrc()])).then(([, src]) => new Promise((res) => {
        const id = ++seq;
        pending.set(id, res); set("running");
        worker.postMessage({ id, kind: "trace", src, job: { code, pre: opts.pre || "", stdin: opts.stdin || "", max: opts.max || 1000 } });
      }));
      chain = job.catch(() => {});
      return job;
    }
    return { run, trace, boot, stop: kill, get state() { return state; }, get pkg() { return pkg; }, on: (f) => (subs.add(f), () => subs.delete(f)) };
  })();

  /* ================================================================ Editor */
  function Editor(host, { value = "", onRun, onChange, label = "Python code" } = {}) {
    host.innerHTML = `<div class="ed"><div class="ed-gut" aria-hidden="true"></div><div class="ed-body"><pre class="ed-hl" aria-hidden="true"></pre>
      <textarea spellcheck="false" autocapitalize="off" autocomplete="off" autocorrect="off" wrap="off" aria-label="${esc(label)}"></textarea></div></div>`;
    const ta = host.querySelector("textarea"), hl = host.querySelector(".ed-hl"), gut = host.querySelector(".ed-gut");
    let errLine = null;
    const paint = () => {
      const v = ta.value;
      hl.innerHTML = hlPy(v) + (v.endsWith("\n") || !v ? " " : "");
      const n = v.split("\n").length;
      gut.innerHTML = Array.from({ length: n }, (_, i) => `<span${i + 1 === errLine ? ' class="err"' : ""}>${i + 1}</span>`).join("");
    };
    // edits go through insertText so Ctrl+Z still works
    const put = (text, a = ta.selectionStart, b = ta.selectionEnd) => {
      ta.setSelectionRange(a, b);
      if (!document.execCommand || !document.execCommand("insertText", false, text)) { ta.setRangeText(text, a, b, "end"); paint(); }
    };
    const lineStart = (i) => ta.value.lastIndexOf("\n", i - 1) + 1;
    function eachLine(fn) {                                              // change every selected line, keep them selected
      const s = lineStart(ta.selectionStart), e0 = ta.selectionEnd, e = ta.value.indexOf("\n", e0 > s && ta.value[e0 - 1] === "\n" ? e0 - 1 : e0);
      const end = e < 0 ? ta.value.length : e, lines = ta.value.slice(s, end).split("\n"), out = fn(lines).join("\n");
      put(out, s, end); ta.setSelectionRange(s, s + out.length);
    }
    const caret = (p) => ta.setSelectionRange(p, p);
    // brackets and quotes, like VS Code: auto-close, type over the closer, delete pairs, wrap a selection
    const PAIR = { "(": ")", "[": "]", "{": "}", '"': '"', "'": "'" };
    function pairs(e, v, a, b) {
      const k = e.key, prev = v[a - 1] || "", next = v[b] || "";
      if (e.ctrlKey || e.metaKey || e.altKey || k.length !== 1) return false;
      const quote = k === '"' || k === "'";
      if (a !== b && PAIR[k]) {                                           // wrap the selection
        const sel = v.slice(a, b);
        put(k + sel + PAIR[k], a, b); ta.setSelectionRange(a + 1, a + 1 + sel.length);
        return true;
      }
      if ((")]}".includes(k) || quote) && next === k && a === b) {         // the closer is already there: step over it
        if (!quote || prev !== "\\") { caret(a + 1); return true; }
      }
      if (quote) {
        if (v.slice(a - 2, a) === k + k && v[a - 3] !== k) { put(k + k + k + k); caret(a + 1); return true; }   // """ docstring """
        const prefix = /[a-zA-Z]/.test(prev) && /^[rRbBuUfF]{1,2}$/.test(v.slice(lineStart(a), a).match(/[A-Za-z_]*$/)[0]);
        if ((/\w/.test(prev) && !prefix) || /\w/.test(next) || prev === "\\") return false;   // don't, it's, \' : just one quote
        put(k + k); caret(a + 1);
        return true;
      }
      if (PAIR[k] && (!next || /[\s)\]}:,;]/.test(next))) { put(k + PAIR[k]); caret(a + 1); return true; }
      return false;
    }
    ta.addEventListener("keydown", (e) => {
      const v = ta.value, a = ta.selectionStart, b = ta.selectionEnd;
      if (pairs(e, v, a, b)) { e.preventDefault(); return; }
      if (e.key === "Backspace" && a === b && a > 0 && PAIR[v[a - 1]] && PAIR[v[a - 1]] === v[a]) { e.preventDefault(); put("", a - 1, a + 1); return; }
      if (e.key === "Enter" && !e.shiftKey && !e.altKey && !e.ctrlKey && !e.metaKey && a === b && "([{".includes(v[a - 1] || "x") && PAIR[v[a - 1]] === v[a]) {
        e.preventDefault();                                               // [|]  →  [\n    |\n]
        const ind = v.slice(lineStart(a), a).match(/^\s*/)[0];
        put("\n" + ind + "    " + "\n" + ind); caret(a + 1 + ind.length + 4);
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); onRun && onRun(); return; }
      if ((e.ctrlKey || e.metaKey) && e.key === "/") {
        e.preventDefault();
        eachLine((L) => { const all = L.filter((l) => l.trim()).every((l) => /^\s*#/.test(l)); return L.map((l) => !l.trim() ? l : all ? l.replace(/^(\s*)# ?/, "$1") : l.replace(/^(\s*)/, "$1# ")); });
        return;
      }
      if (e.key === "Tab" && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        if (e.shiftKey) eachLine((L) => L.map((l) => l.replace(/^ {1,4}/, "")));
        else if (v.slice(a, b).includes("\n")) eachLine((L) => L.map((l) => (l ? "    " + l : l)));
        else put("    ");
        return;
      }
      if (e.key === "Enter" && !e.shiftKey && !e.altKey) {
        e.preventDefault();
        const cur = v.slice(lineStart(a), a), ind = cur.match(/^\s*/)[0];
        put("\n" + ind + (/:\s*(#.*)?$/.test(cur) ? "    " : ""));
        return;
      }
      if (e.key === "Backspace" && a === b && a > 0) {                   // delete one indent level at once
        const cur = v.slice(lineStart(a), a);
        if (/^ +$/.test(cur) && cur.length % 4 === 0) { e.preventDefault(); put("", a - 4, a); }
      }
    });
    ta.addEventListener("input", () => { errLine = null; paint(); onChange && onChange(ta.value); });
    ta.value = value; paint();

    // the bracket next to the cursor and its partner get a soft box (strings and comments are skipped)
    const body = host.querySelector(".ed-body"), marks = [0, 1].map(() => { const m = document.createElement("span"); m.className = "ed-match"; m.hidden = true; body.appendChild(m); return m; });
    const OPEN = "([{", CLOSE = ")]}";
    let cw = 0;
    const charW = () => { if (!cw) { const s = document.createElement("span"); s.textContent = "0".repeat(20); s.style.cssText = "position:absolute;visibility:hidden;white-space:pre"; hl.appendChild(s); cw = s.getBoundingClientRect().width / 20; s.remove(); } return cw; };
    function partner(v, i) {
      const plain = v.replace(/#[^\n]*|"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'/g, (m) => m.replace(/[^\n]/g, " "));
      const c = plain[i], o = OPEN.indexOf(c), cl = CLOSE.indexOf(c);
      if (o < 0 && cl < 0) return -1;
      const dir = o >= 0 ? 1 : -1, me = c, them = o >= 0 ? CLOSE[o] : OPEN[cl];
      for (let j = i + dir, d = 0; j >= 0 && j < plain.length; j += dir) {
        if (plain[j] === me) d++;
        else if (plain[j] === them) { if (d === 0) return j; d--; }
      }
      return -1;
    }
    function where(v, i) {                                                // line + visual column (tabs count as 4)
      const s = v.lastIndexOf("\n", i - 1) + 1, line = v.slice(0, s).split("\n").length - 1;
      let col = 0; for (let k = s; k < i; k++) col = v[k] === "\t" ? col + 4 - (col % 4) : col + 1;
      return { line, col };
    }
    function showMatch() {
      marks.forEach((m) => (m.hidden = true));
      if (document.activeElement !== ta || ta.selectionStart !== ta.selectionEnd) return;
      const v = ta.value, a = ta.selectionStart;
      let i = OPEN.includes(v[a] || "x") || CLOSE.includes(v[a] || "x") ? a : a - 1, j = i >= 0 ? partner(v, i) : -1;
      if (j < 0 && i === a && a > 0) { i = a - 1; j = partner(v, i); }
      if (j < 0) return;
      const cs = getComputedStyle(hl), lh = parseFloat(cs.lineHeight), top = parseFloat(cs.paddingTop), w = charW();
      [i, j].forEach((p, n) => { const { line, col } = where(v, p), m = marks[n]; m.style.cssText = `top:${top + line * lh}px;left:${col * w}px;width:${w}px;height:${lh}px`; m.hidden = false; });
    }
    let mT = 0;
    const later = () => { cancelAnimationFrame(mT); mT = requestAnimationFrame(showMatch); };
    ["keyup", "click", "input", "focus"].forEach((ev) => ta.addEventListener(ev, later));
    ta.addEventListener("blur", () => marks.forEach((m) => (m.hidden = true)));
    document.addEventListener("selectionchange", () => { if (document.activeElement === ta) later(); });
    return {
      el: host, ta,
      get value() { return ta.value; },
      set value(v) { ta.value = v; errLine = null; paint(); },
      markError(n) { errLine = n || null; paint(); },
      focus() { ta.focus(); },
    };
  }

  /* ================================================================ output + status */
  const fmtMs = (ms) => (ms < 1000 ? Math.max(1, Math.round(ms)) + " ms" : (ms / 1000).toFixed(2) + " s");
  // while Python downloads, the status counts seconds so a slow connection doesn't look frozen
  let bootAt = 0;
  PyRun.on((s) => { if (s === "loading" && !bootAt) bootAt = Date.now(); if (s !== "loading") bootAt = 0; });
  setInterval(() => { if (!bootAt) return; document.querySelectorAll(".run-status .py-secs").forEach((el) => (el.textContent = loadNote())); }, 1000);
  const loadNote = () => { const t = Math.round((Date.now() - (bootAt || Date.now())) / 1000); return t < 2 ? "(first time downloads about 10 MB)" : t < 15 ? `(downloading… ${t}s)` : `(slow connection, still downloading… ${t}s. It's only this slow once.)`; };
  const status = (s) => s === "loading" ? `<span class="py-dot load"></span>Starting Python… <span class="muted py-secs">${loadNote()}</span>`
    : s === "packages" ? `<span class="py-dot load"></span>Loading ${esc(PyRun.pkg.split(/,\s*/).slice(0, 4).join(", "))}… <span class="muted">(first time only, a few MB)</span>`
    : s === "running" ? `<span class="py-dot run"></span>Running…` : "";
  // a run area: a toolbar (Run / Stop / extras) and the output under it
  function RunPane(host, { getCode, pre, after, extra = "", onResult, hint = true, editor = null } = {}) {
    host.innerHTML = `<div class="run-bar">
        <button type="button" class="btn sm primary" data-r="run" title="Run (Ctrl + Enter)">${ICO(I.play)}<span>Run</span></button>
        ${editor ? `<button type="button" class="btn sm" data-r="step" title="Run your code one line at a time and watch the variables">${ICO(I.step)}<span>Step through</span></button>` : ""}
        <button type="button" class="btn sm" data-r="stop" title="Stop the program" hidden>${ICO(I.stop)}<span>Stop</span></button>
        ${extra}
        <label class="run-stdin-tog small muted"><input type="checkbox" data-r="stdin"> Input</label>
        <span class="run-status small"></span>${hint ? `<span class="run-hint small muted"><kbd>Ctrl</kbd> <kbd>Enter</kbd></span>` : ""}
      </div>
      <textarea class="run-stdin" rows="2" placeholder="Text for input(), one line per call" spellcheck="false" hidden></textarea>
      <div class="run-out" hidden></div>`;
    const out = host.querySelector(".run-out"), st = host.querySelector(".run-status"), stopB = host.querySelector('[data-r="stop"]'), runB = host.querySelector('[data-r="run"]');
    const stdin = host.querySelector(".run-stdin"), stdTog = host.querySelector('[data-r="stdin"]');
    let mine = false;
    const off = PyRun.on((s) => { if (!document.body.contains(host)) return off(); st.innerHTML = mine || s === "loading" ? status(s) : ""; stopB.hidden = !(mine && (s === "running" || s === "loading" || s === "packages")); });
    stdTog.onchange = () => { stdin.hidden = !stdTog.checked; if (stdTog.checked) stdin.focus(); };
    stopB.onclick = () => { PyRun.stop(); };
    const val = (x) => (typeof x === "function" ? x() : x);
    function needsInput(code) {                                         // the program reads input: ask for it first
      if (!/\binput\s*\(/.test(code) || stdTog.checked) return false;
      stdTog.checked = true; stdin.hidden = false; stdin.focus();
      out.hidden = false;
      out.innerHTML = `<div class="run-msg">This code calls <code>input()</code>. Type what it should read in the box above (one line per <code>input()</code> call), then press Run again.</div>`;
      return true;
    }
    async function run() {
      const code = getCode();
      if (needsInput(code)) return;
      mine = true; runB.disabled = true; stopB.hidden = false;
      st.innerHTML = status(PyRun.state === "ready" ? "running" : "loading");
      let r;
      try { r = await PyRun.run(code, { pre: typeof pre === "function" ? pre() : pre, after: typeof after === "function" ? after() : after, stdin: stdTog.checked ? stdin.value : "" }); }
      catch (e) { r = e.message === "stopped" ? { stopped: true } : { failed: String(e.message || e) }; }
      mine = false; runB.disabled = false; stopB.hidden = true; st.innerHTML = "";
      out.hidden = false;
      if (r.failed) out.innerHTML = `<div class="run-msg bad">Couldn't start Python. It loads from the internet the first time, so check your connection and try again.<div class="small muted">${esc(r.failed)}</div></div>`;
      else if (r.stopped) out.innerHTML = `<div class="run-msg">${icon("stop")} Stopped.</div>`;
      else if (!onResult || onResult(r, out) === false) out.innerHTML = plainOut(r);
      out.insertAdjacentHTML("afterbegin", `<button type="button" class="run-x" data-r="clear-out" title="Clear output" aria-label="Clear output">${ICO(I.close)}</button>`);
      return r;
    }
    runB.onclick = run;
    const stepB = host.querySelector('[data-r="step"]');
    if (stepB) stepB.onclick = async () => {
      const code = getCode(), tests = val(after);
      if (needsInput(code)) return;
      const full = tests ? code.replace(/\s+$/, "") + "\n\n\n" + String(tests).replace(/^\n+/, "") : code;
      mine = true; runB.disabled = stepB.disabled = true; stopB.hidden = false;
      st.innerHTML = status(PyRun.state === "ready" ? "running" : "loading");
      let r;
      try { r = await PyRun.trace(full, { pre: val(pre), stdin: stdTog.checked ? stdin.value : "" }); }
      catch (e) { r = e.message === "stopped" ? { stopped: true } : { failed: String(e.message || e) }; }
      mine = false; runB.disabled = stepB.disabled = false; stopB.hidden = true; st.innerHTML = "";
      if (r.trace && r.trace.steps && r.trace.steps.length > 1) { stepView(r.trace, full); return; }
      out.hidden = false;
      // a syntax error stops the tracer before line 1: show only the part about your code, and mark the line
      const te = String(r.traceErr || ""), at = te.search(/File "<sample>", line \d+/), ln = te.match(/File "<sample>", line (\d+)/);
      const msg = at >= 0 ? te.slice(at).replace("<sample>", "<your code>") : te || (r.trace && r.trace.err) || "The program didn't run any lines.";
      out.innerHTML = r.failed ? `<div class="run-msg bad">Couldn't start Python. Check your internet connection and try again.<div class="small muted">${esc(r.failed)}</div></div>`
        : r.stopped ? `<div class="run-msg">${icon("stop")} Stopped.</div>`
        : `<div class="run-head"><span>Can't step through this: fix the error first</span></div><pre class="run-pre err">${esc(msg.replace(/\n+$/, ""))}</pre>`;
      if (ln && editor && +ln[1] <= code.split("\n").length) editor.markError(+ln[1]);
      out.insertAdjacentHTML("afterbegin", `<button type="button" class="run-x" data-r="clear-out" title="Clear output" aria-label="Clear output">${ICO(I.close)}</button>`);
    };
    // the player replaces the editor until you go back to editing
    function stepView(tr, code) {
      const view = document.createElement("div");
      view.className = "step-view";
      view.innerHTML = `<div class="step-top"><span class="small"><b>Stepping through your code</b> <span class="muted">· ${tr.steps.length - 1} steps</span></span>
          <button type="button" class="btn sm" data-r="back">${ICO(I.edit)}<span>Back to editing</span></button></div>
        ${tr.err ? `<div class="step-err small">The program stopped with an error: <code>${esc(tr.err)}</code>. Step to the end to see where.</div>` : ""}
        ${codeBlock(code)}`;
      const head = view.querySelector(".code-head");
      if (head) head.remove();
      editor.el.hidden = true; host.hidden = true;
      editor.el.after(view);
      startStepper(view.querySelector(".code-wrap"), tr);
      view.querySelector('[data-r="back"]').onclick = () => { stopStepper(view.querySelector(".code-wrap")); view.remove(); editor.el.hidden = false; host.hidden = false; editor.focus(); };
      const nx = view.querySelector('[data-a="next"]');
      if (nx) nx.focus();
    }
    out.addEventListener("click", (e) => { if (e.target.closest('[data-r="clear-out"]')) { out.hidden = true; out.innerHTML = ""; } });
    return { run, out };
  }
  const imgsOut = (r) => (r.imgs && r.imgs.length ? `<div class="plots run-plots">${r.imgs.map((b) => `<img class="plot-img" alt="your plot" src="data:image/png;base64,${b}">`).join("")}</div>` : "");
  function plainOut(r) {
    const empty = !r.out && !r.err && !(r.imgs && r.imgs.length);
    return `<div class="run-head"><span>Output</span><span class="muted">${r.err ? "stopped with an error" : "finished"} · ${fmtMs(r.ms)}</span></div>
      ${r.out ? `<pre class="run-pre">${esc(r.out.replace(/\n$/, ""))}</pre>` : ""}${imgsOut(r)}${r.err ? `<pre class="run-pre err">${esc(r.err.replace(/\n$/, ""))}</pre>` : ""}
      ${empty ? `<div class="run-msg muted">The program finished without printing anything. Add <code>print(…)</code> to see values.</div>` : ""}`;
  }

  /* ================================================================ ✏️ Edit & run on code examples */
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-try]");
    if (!b) return;
    const wrap = b.closest(".code-wrap");
    if (wrap.classList.contains("editing")) { closeTry(wrap); return; }
    openTry(wrap, b);
  });
  function openTry(wrap, b) {
    const stepB = wrap.querySelector("[data-step]");
    if (stepB && !wrap.querySelector(".stepper").hidden) stepB.click();           // close the stepper first
    const orig = window.codeSource ? codeSource(+wrap.dataset.i) : "";
    const box = document.createElement("div");
    box.className = "try-box";
    box.innerHTML = `<div class="try-ed"></div><div class="try-run"></div>`;
    wrap.querySelector(".code-lines").after(box);
    wrap.classList.add("editing");
    setSwap(b, 1);
    const pre = () => (wrap.dataset.pre && window.PRAC_HELPERS ? PRAC_HELPERS[wrap.dataset.pre] : "");
    let pane;
    const ed = Editor(box.querySelector(".try-ed"), { value: orig, onRun: () => pane.run() });
    pane = RunPane(box.querySelector(".try-run"), {
      getCode: () => ed.value, pre, editor: ed,
      extra: `<button type="button" class="btn sm" data-r="reset" title="Put the original code back">${ICO(I.reset)}<span>Reset</span></button>
        <button type="button" class="btn sm" data-r="pg" title="Copy this code into the playground">${ICO(I.out)}<span>Playground</span></button>`,
      onResult: (r, out) => { ed.markError(r.line); return false; },
    });
    box.querySelector('[data-r="reset"]').onclick = () => { ed.value = orig; ed.focus(); };
    box.querySelector('[data-r="pg"]').onclick = () => Playground.open(ed.value);
    PyRun.boot().catch(() => {});                                       // start loading while you edit
    ed.focus();
  }
  function closeTry(wrap) {
    const box = wrap.querySelector(".try-box");
    if (box) box.remove();
    wrap.classList.remove("editing");
    const b = wrap.querySelector("[data-try]");
    if (b) setSwap(b, 0);
  }

  /* ================================================================ 🐍 playground panel */
  const Playground = (() => {
    let el = null, ed = null;
    const START = `# Python playground: type any code and press Ctrl + Enter
# numpy, pandas, scikit-learn and matplotlib load the first time you import them
import numpy as np

x = np.array([2.0, 1.0, 0.1])
probs = np.exp(x) / np.exp(x).sum()      # softmax
print(probs.round(3), probs.sum())
`;
    function build() {
      el = document.createElement("aside");
      el.className = "pg"; el.setAttribute("aria-label", "Python playground"); el.hidden = true;
      el.innerHTML = `<div class="pg-head"><b>${ICO(I.term)} Python playground</b><span class="small muted">saved in this browser</span>
          <button type="button" class="icon-btn" data-pg="close" title="Close (Esc)" aria-label="Close">${ICO(I.close)}</button></div>
        <div class="pg-ed"></div><div class="pg-run"></div>`;
      document.body.appendChild(el);
      let saveT = 0;
      ed = Editor(el.querySelector(".pg-ed"), { value: ls.get(lsKey("pg:code")) ?? START, onRun: () => pane.run(), onChange: (v) => { clearTimeout(saveT); saveT = setTimeout(() => ls.set(lsKey("pg:code"), v), 300); } });
      const pane = RunPane(el.querySelector(".pg-run"), {
        getCode: () => ed.value, editor: ed,
        extra: `<button type="button" class="btn sm" data-r="clear" title="Empty the editor">${ICO(I.reset)}<span>Clear code</span></button>`,
        onResult: (r) => { ed.markError(r.line); return false; },
      });
      el.querySelector('[data-r="clear"]').onclick = () => { ed.value = ""; ls.set(lsKey("pg:code"), ""); ed.focus(); };
      el.querySelector('[data-pg="close"]').onclick = close;
    }
    function open(code) {
      if (!el) build();
      if (typeof code === "string") { ed.value = code; ls.set(lsKey("pg:code"), code); }
      el.hidden = false;
      requestAnimationFrame(() => el.classList.add("show"));
      document.body.classList.add("pg-open");
      PyRun.boot().catch(() => {});
      ed.focus();
    }
    function close() {
      if (!el || el.hidden) return;
      if (el.contains(document.activeElement)) document.activeElement.blur();   // give the keyboard back to the page
      el.classList.remove("show"); document.body.classList.remove("pg-open");
      setTimeout(() => { if (!el.classList.contains("show")) el.hidden = true; }, 260);
    }
    const toggle = () => (el && !el.hidden ? close() : open());
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && el && !el.hidden && !document.querySelector(".fd-bg")) { close(); return; }
      const t = e.target;
      if (e.ctrlKey || e.metaKey || e.altKey || (t.closest && t.closest("input, textarea, select, [contenteditable]"))) return;
      if (e.key.toLowerCase() === "p" && !e.shiftKey) { e.preventDefault(); toggle(); }
    });
    return { open, close, toggle };
  })();
  // header button, left of 🔍
  const anchor = document.getElementById("findBtn") || document.querySelector(".fx-slot") || document.getElementById("themeBtn");
  if (anchor) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "icon-btn"; b.id = "pgBtn";
    b.title = "Python playground (P)"; b.setAttribute("aria-label", "Python playground");
    b.innerHTML = ICO(I.term);
    b.onclick = () => Playground.toggle();
    anchor.before(b);
  }

  /* ================================================================ Practice: try it yourself */
  // a problem's starting code: imports kept, every function (and method) signature with an empty body
  function starter(code) {
    const L = String(code).replace(/^\n+|\s+$/g, "").split("\n"), out = [];
    for (let i = 0; i < L.length; i++) {
      const m = L[i].match(/^(\s*)def\s/);
      if (!m) { out.push(L[i]); continue; }
      const ind = m[1].length;
      let j = i;
      while (j < L.length - 1 && !/:\s*(#.*)?$/.test(L[j])) j++;            // a signature can span lines
      for (let k = i; k <= j; k++) out.push(L[k].replace(/:\s*#.*$/, ":"));
      out.push(" ".repeat(ind + 4) + "pass", "");
      i = j;
      while (i + 1 < L.length && (!L[i + 1].trim() || L[i + 1].match(/^\s*/)[0].length > ind)) i++;
    }
    return out.join("\n").replace(/\n{3,}/g, "\n\n").replace(/\s+$/, "") + "\n";
  }
  const lines = (s) => String(s || "").replace(/\r/g, "").split("\n").map((l) => l.replace(/\s+$/, "")).join("\n").replace(/^\n+|\n+$/g, "").split("\n");
  // mount(el, problem): editor with your saved code, Run tests, line-by-line comparison
  function solve(host, p, onPass) {
    const KEY = lsKey("code:prac:" + p.id), start = p.starter || starter(p.opt.code), tests = String(p.run || "").replace(/^\n+/, ""), expected = lines(p.out);
    const calls = tests.split("\n").filter((l) => /^\s*print\(/.test(l)).map((l) => l.trim().replace(/^print\((.*)\)$/, "$1"));
    host.innerHTML = `<div class="try-ed"></div><div class="try-run"></div>
      <details class="small solve-tests"><summary class="muted">Show the test code</summary>${codeBlock(tests)}</details>`;
    let saveT = 0, pane;
    const ed = Editor(host.querySelector(".try-ed"), { value: ls.get(KEY) ?? start, label: "Your solution", onRun: () => pane.run(),
      onChange: (v) => { clearTimeout(saveT); saveT = setTimeout(() => (v === start ? ls.del(KEY) : ls.set(KEY, v)), 300); } });
    pane = RunPane(host.querySelector(".try-run"), {
      getCode: () => ed.value, editor: ed, pre: p.h && window.PRAC_HELPERS ? PRAC_HELPERS[p.h] : "", after: tests || null,
      extra: `<button type="button" class="btn sm" data-r="reset" title="Start again from the empty template">${ICO(I.reset)}<span>Reset</span></button>`,
      onResult: (r, out) => {
        ed.markError(r.line);
        if (!tests || r.err || r.test == null) return false;              // no tests, or your code didn't run: show plain output
        const got = lines(r.test), n = Math.max(expected.length, got.length);
        let pass = 0;
        const rows = Array.from({ length: n }, (_, i) => {
          const ok = expected[i] === got[i]; if (ok) pass++;
          return `<tr class="${ok ? "ok" : "bad"}"><td class="tr-i">${icon(ok ? "check" : "x")}</td>${calls.length === expected.length ? `<td class="tr-call"><code>${esc(calls[i] || "")}</code></td>` : ""}<td><code>${esc(expected[i] ?? "—")}</code></td><td><code>${esc(got[i] ?? "—")}</code></td></tr>`;
        }).join("");
        const all = pass === n && !r.test_err;
        out.innerHTML = `<div class="tr-sum ${all ? "ok" : "bad"}">${all ? `${icon("check")} All ${n} checks pass` : `${icon("x")} ${pass} of ${n} checks pass`}<span class="muted small">· ${fmtMs(r.ms)}</span>
            ${all && !Store.item("prac:" + p.id).status ? `<button type="button" class="btn sm on-green" data-r="solved">Mark as solved</button>` : ""}</div>
          <div class="table-wrap"><table class="tr-table"><thead><tr><th></th>${calls.length === expected.length ? "<th>Test</th>" : ""}<th>Expected</th><th>Your output</th></tr></thead><tbody>${rows}</tbody></table></div>
          ${r.test_err ? `<pre class="run-pre err">${esc(r.test_err.replace(/\n$/, ""))}</pre>` : ""}
          ${r.out ? `<div class="run-head"><span>Your own prints</span></div><pre class="run-pre">${esc(r.out.replace(/\n$/, ""))}</pre>` : ""}${imgsOut(r)}
          ${!all && got.length !== expected.length ? `<div class="run-msg muted small">The number of lines differs: extra <code>print()</code> calls inside your solution also show up here.</div>` : ""}`;
        const sb = out.querySelector('[data-r="solved"]');
        if (sb) sb.onclick = () => { sb.remove(); onPass && onPass(); };
        return true;
      },
    });
    host.querySelector('[data-r="reset"]').onclick = () => {
      if (ed.value !== start && !confirm("Clear your code and start again from the empty template?")) return;
      ed.value = start; ls.del(KEY); ed.focus();
    };
    host.addEventListener("focusin", () => PyRun.boot().catch(() => {}), { once: true });
    return ed;
  }

  window.PyRun = PyRun;
  window.PyUI = { Editor, RunPane, Playground, solve, starter };
})();
