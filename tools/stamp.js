/* Version-stamp every local script and stylesheet a page loads: src="assets/common.js?v=3f9a1c2e".
   The stamp is a short fingerprint of the file's content, so when a file changes the browser fetches the new copy
   instead of mixing a cached old file with new ones (which can leave a page blank). Unchanged files keep their stamp.
   Usage:  node tools/stamp.js      (also run automatically by tools/build.js)                                     */
const fs = require("fs"), path = require("path"), crypto = require("crypto");
const root = path.join(__dirname, "..");
const pages = fs.readdirSync(root).filter((f) => f.endsWith(".html")).map((f) => path.join(root, f));
let changed = 0;
for (const page of pages) {
  if (!fs.existsSync(page)) continue;
  const dir = path.dirname(page), html = fs.readFileSync(page, "utf8");
  const out = html.replace(/(src|href)="((?!https?:|\/\/|data:)[^"?#]+\.(?:js|css))(?:\?v=[0-9a-f]*)?"/g, (m, attr, rel) => {
    const file = path.join(dir, rel);
    if (!fs.existsSync(file)) return m;
    const v = crypto.createHash("md5").update(fs.readFileSync(file)).digest("hex").slice(0, 8);
    return `${attr}="${rel}?v=${v}"`;
  });
  if (out !== html) { fs.writeFileSync(page, out); changed++; }
}
console.log(`version stamps updated in ${changed} of ${pages.length} pages`);
