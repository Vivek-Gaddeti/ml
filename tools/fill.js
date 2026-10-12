/* Fill placeholder outputs.
   In any data file, write a code block's output as the string "?" — e.g. ["code", R`...`, "?"]
   This tool runs each such snippet with Python and writes the real output into the file.
   Usage:  node tools/fill.js data/math.js [more files...]                */
const fs = require("fs"), path = require("path"), os = require("os"), { execFileSync } = require("child_process");
// Python must print UTF-8 (samples print symbols like ←, ✓), whatever the Windows console code page is; plots draw off-screen
const PYENV = { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1", MPLBACKEND: "Agg" };
const root = path.join(__dirname, "..");
global.window = global; global.R = String.raw; global.document = { addEventListener() {} };
require(path.join(root, "assets/site.js"));
const LISTS = SITE.docs.map((d) => d.list);

for (const rel of process.argv.slice(2)) {
  const file = path.resolve(root, rel);
  let src = fs.readFileSync(file, "utf8");
  // collect the code of every block with a "?" output, in source order
  const arrays = Object.fromEntries(LISTS.map((k) => [k, []]));
  Object.keys(arrays).forEach((k) => (global[k] = arrays[k]));
  new Function(src)();
  const codes = [];
  const walk = (blocks) => (blocks || []).forEach((b) => {
    if (!Array.isArray(b)) return;
    if (b[0] === "code" && b[2] === "?") codes.push(b[1]);
    if (b[0] === "ex" && b[1].out === "?") codes.push(b[1].code);
    if (b[0] === "qa" && Array.isArray(b[2])) walk(b[2]);
  });
  Object.values(arrays).forEach((arr) => arr.forEach((s) => walk(s.blocks)));
  // placeholders in the source text: the R`...` template followed by , "?"  (or out: "?")
  const re = /, "\?"(?=\s*[\],])|out: "\?"/g;
  const hits = src.match(re) || [];
  if (hits.length !== codes.length) { console.log(`${rel}: ${codes.length} blocks but ${hits.length} placeholders — fix manually`); continue; }
  const tmp = path.join(os.tmpdir(), `fill-${process.pid}.py`);      // per run: several fills can go at once
  let i = 0, failed = 0;
  src = src.replace(re, (m) => {
    const code = codes[i++].replace(/^\n+|\s+$/g, "");
    fs.writeFileSync(tmp, code);
    let out;
    try { out = execFileSync("python", [tmp], { encoding: "utf8", cwd: os.tmpdir(), timeout: 60000, env: PYENV }); }
    catch (e) { failed++; console.log(`\n✗ ${rel} block ${i}: ${(e.stderr || e.message).split("\n").slice(-3).join(" | ")}\n${code.slice(0, 200)}`); return m; }
    out = out.replace(/\r/g, "").replace(/\s+$/, "");
    return m.startsWith("out") ? "out: " + JSON.stringify(out) : ", " + JSON.stringify(out);
  });
  fs.writeFileSync(file, src);
  console.log(`${rel}: filled ${codes.length - failed}/${codes.length}`);
}
