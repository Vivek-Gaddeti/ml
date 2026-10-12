/* Verifies every code sample on the site by actually running it with Python.
   - Practice: brute / better / opt code + p.run must print exactly p.out
   - Doc pages: every ['code', code, output] block (not the ones marked to run on a computer), every example {code, out},
     and code inside Q&A answers
   Needs numpy, pandas, scikit-learn and matplotlib installed (pip install numpy==2.0.2 pandas scikit-learn==1.6.1 matplotlib).
   Usage:  node tools/verify.js [filter …]     add --local to also run the "Runs on your computer" blocks (needs torch etc.)
   A filter is a data file (data/ml-2.js, data/prac/10-ml.js), a page kind (ml, tf…), a section / problem id, or part of one. */
const fs = require("fs"), path = require("path"), { execFileSync } = require("child_process");
// Python must print UTF-8 (samples print symbols like ←, ✓), whatever the Windows console code page is; plots draw off-screen
const PYENV = { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1", MPLBACKEND: "Agg" };
const C = require("./content");

const args = process.argv.slice(2), local = args.includes("--local"), filters = args.filter((a) => a !== "--local").map((a) => a.split("\\").join("/"));
const hit = (x, tag) => !filters.length || filters.some((f) => (x._file && x._file.endsWith(f)) || tag === f || x.id === f || (f.length > 3 && !f.endsWith(".js") && x.id.includes(f)));
const jobs = [];
C.prac.forEach((p) => {
  if (!hit(p, "prac")) return;
  const pre = p.h ? C.helpers[p.h] + "\n" : "";
  [["opt", p.opt], ["brute", p.brute], ["better", p.better]].forEach(([k, a]) => {
    if (!a || !a.code || a.run === false) return;
    const own = typeof a.run === "string"; // approach-specific driver (e.g. a smaller input for a slow version)
    jobs.push({ name: `prac:${p.id}:${k}`, code: pre + a.code + "\n\n" + (own ? a.run : p.run || ""), out: own ? a.out : p.out });
  });
});
const blocks = (secs, tag) => (secs || []).forEach((s) => (s.blocks || []).forEach((b, i) => {
  if (!s._qa && !hit(s, tag)) return;
  if (b[0] === "code" && b[2] != null && b[2] !== "" && (!b[4] || local)) jobs.push({ name: `${tag}:${s.id}#${i}`, code: b[1], out: b[2] });
  if (b[0] === "ex" && b[1].code && b[1].out != null) jobs.push({ name: `${tag}:${s.id}#${i}:${b[1].t}`, code: b[1].code, out: b[1].out });
  if (b[0] === "qa" && Array.isArray(b[2])) blocks([{ id: s.id + "#qa" + i, blocks: b[2], _qa: true }], tag);
}));
C.docs.forEach((d) => blocks(d.sections, d.kind));

const tmp = fs.mkdtempSync(path.join(require("os").tmpdir(), "ai-verify-"));      // its own folder: several runs can go at once
let bad = 0;
const t0 = Date.now();
jobs.forEach((j, n) => {
  const f = path.join(tmp, `job${n}.py`);
  fs.writeFileSync(f, j.code);
  let got;
  try { got = execFileSync("python", [f], { encoding: "utf8", timeout: 60000, cwd: tmp, stdio: ["pipe", "pipe", "pipe"], env: PYENV }); }
  catch (e) { got = (e.stdout || "") + "\n!!ERROR!! " + (e.stderr || e.message).split("\n").slice(-4).join("\n"); }
  const norm = (s) => String(s).replace(/\r/g, "").split("\n").map((l) => l.replace(/\s+$/, "")).join("\n").replace(/^\n+|\n+$/g, "");
  if (norm(got) !== norm(j.out)) { bad++; console.log(`\n✗ ${j.name}\n--- expected ---\n${norm(j.out)}\n--- got ---\n${norm(got)}`); }
});
console.log(`\n${jobs.length - bad}/${jobs.length} code samples OK` + (bad ? `, ${bad} FAILED` : "") + ` (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
process.exit(bad ? 1 : 0);
