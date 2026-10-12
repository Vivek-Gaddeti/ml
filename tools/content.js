/* Loads the whole site's content into this Node process, the same way the pages do (site.js first, then data files).
   Used by verify.js, build.js and lite.js:   const C = require("./content");   C.docs → [{ kind, list, sections, … }], C.prac
   Generated files (outmap, index, progress, trace-src) are skipped.                                                   */
const fs = require("fs"), path = require("path");
const root = path.join(__dirname, ".."), data = path.join(root, "data");
global.window = global; global.R = String.raw; global.document = { addEventListener() {} };
require(path.join(root, "assets/site.js"));
// every section / problem remembers which file it came from (s._file), so tools can work on one file at a time
const lists = () => [...SITE.docs.map((d) => global[d.list]), global.PRAC];
const load = (f) => {
  const before = lists().map((l) => l.length);
  new Function(fs.readFileSync(f, "utf8"))();
  lists().forEach((l, i) => l.slice(before[i]).forEach((x) => Object.defineProperty(x, "_file", { value: path.relative(root, f).split(path.sep).join("/"), enumerable: false })));
};
const GENERATED = new Set(["outmap.js", "index.js", "progress.js", "trace-src.js"]);

// data/*.js: briefs first (they only fill BRIEFS), then every doc file in name order
const top = fs.readdirSync(data).filter((f) => f.endsWith(".js") && !GENERATED.has(f)).sort((a, b) => (a.startsWith("briefs") ? 0 : 1) - (b.startsWith("briefs") ? 0 : 1) || a.localeCompare(b));
top.forEach((f) => load(path.join(data, f)));
// data/prac: helpers.js first, then the numbered problem files
const pdir = path.join(data, "prac");
if (fs.existsSync(pdir)) {
  if (fs.existsSync(path.join(pdir, "helpers.js"))) load(path.join(pdir, "helpers.js"));
  fs.readdirSync(pdir).filter((f) => /^\d.*\.js$/.test(f)).sort().forEach((f) => load(path.join(pdir, f)));
}

module.exports = {
  root, data,
  docs: SITE.docs.map((d) => Object.assign({}, d, { sections: global[d.list] || [] })),
  prac: global.PRAC,
  helpers: global.PRAC_HELPERS || {},
  // the doc files a page loads: data/<kind>.js or data/<kind>-1.js, data/<kind>-2.js …
  docFiles: (kind) => fs.readdirSync(data).filter((f) => new RegExp("^" + kind + "(-\\d+)?\\.js$").test(f)).sort(),
};
