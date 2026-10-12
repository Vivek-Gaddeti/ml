/* find.js: Ctrl + K (or the 🔍 button in the header) opens one search box for the whole site:
   every page, every section of every doc page (Math, Python for ML, ML, DL, Transformers, LLMs, Agents) and every practice problem.
   ↑ ↓ to move, Enter to open, Esc to close. With nothing typed it shows the pages and what you touched last.
   Data: data/progress.js (titles and groups only, so it is small). */
(function () {
  const PR = window.PROGRESS;
  if (!PR) return;
  const ICO = (d) => `<svg class="cat-ico" viewBox="0 0 24 24" aria-hidden="true">${d}</svg>`;
  const LENS = '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/>';
  const KIND = { page: "Page", prac: "Problem" };
  SITE.docs.forEach((d) => (KIND[d.kind] = d.type));

  // everything you can jump to: [key in Store, kind, title, small text under it, link]
  const SUBS = {}; SITE.links.forEach(([href, label, sub]) => (sub || [[href, label, ""]]).forEach(([h, , note]) => (SUBS[h] = note)));
  const all = [
    ["", "page", "Dashboard", "Your progress, streak and what's due", "index.html"],
    ...SITE.docs.map((d) => ["", "page", d.title, SUBS[d.href] || "", d.href]),
    ["", "page", "Practice", "All problems as a list", "practice.html#list"],
    ["", "page", "Practice roadmap", "Practice topics as a map", "practice.html#roadmap"],
    ["", "page", "Revision", "What's due for review", "revision.html"],
    ...SITE.docs.flatMap((d) => (PR.docs[d.kind] || []).map((s) => [d.kind + ":" + s.id, d.kind, s.title, s.group, d.href + "#" + s.id])),
    ...PR.prac.map((p) => ["prac:" + p.id, "prac", p.t, p.cat, "practice.html#p/" + p.id, p.d]),
  ].map(([key, kind, title, sub, href, d]) => ({ key, kind, title, sub: sub || "", href, d, low: title.toLowerCase(), hay: (title + " " + (sub || "") + " " + KIND[kind] + " " + (d || "")).toLowerCase() }));

  // a title that starts with what you typed beats one that only contains it, and pages come first on a tie
  function score(x, q, words) {
    if (!words.every((w) => x.hay.includes(w))) return -1;
    let s = x.low === q ? 100 : x.low.startsWith(q) ? 60 : new RegExp("\\b" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).test(x.low) ? 40 : x.low.includes(q) ? 25 : 0;
    if (x.kind === "page") s += 5;
    return s - x.title.length / 100;
  }
  function results(q) {
    q = q.trim().toLowerCase();
    if (!q) {
      const items = Store.load().items;
      const recent = all.filter((x) => x.key && items[x.key] && items[x.key].updated).sort((a, z) => items[z.key].updated - items[a.key].updated).slice(0, 5);
      return [...recent.map((x) => ({ ...x, recent: true })), ...all.filter((x) => x.kind === "page")];
    }
    const words = q.split(/\s+/);
    return all.map((x) => [score(x, q, words), x]).filter(([s]) => s >= 0).sort((a, z) => z[0] - a[0]).slice(0, 40).map(([, x]) => x);
  }

  let bg = null, sel = 0, list = [];
  function row(x, i) {
    const it = x.key ? Store.item(x.key) : {};
    const mark = it.status === "done" ? `<span class="fd-st ok" title="Done">${icon("check")}</span>` : it.status === "revise" ? `<span class="fd-st rv" title="Needs revision">${icon("repeat")}</span>` : "";
    return `<a class="fd-row${i === sel ? " on" : ""}" href="${x.href}" data-i="${i}" role="option" aria-selected="${i === sel}">
      <span class="fd-kind k-${x.kind}">${x.recent ? "Recent" : KIND[x.kind]}</span>
      <span class="fd-t"><b>${esc(x.title)}</b>${x.sub ? `<small>${esc(x.sub)}</small>` : ""}</span>
      ${x.d ? `<span class="badge ${x.d.toLowerCase()}">${x.d}</span>` : ""}${mark}</a>`;
  }
  function paint() {
    const box = bg.querySelector(".fd-list"), q = bg.querySelector("input").value;
    list = results(q);
    sel = Math.min(sel, Math.max(0, list.length - 1));
    box.innerHTML = list.length ? (q.trim() ? "" : `<div class="fd-h">${list.some((x) => x.recent) ? "Recent and pages" : "Pages"}</div>`) + list.map(row).join("")
      : `<div class="fd-empty">Nothing matches “${esc(q)}”.</div>`;
  }
  function move(d) {
    if (!list.length) return;
    sel = (sel + d + list.length) % list.length;
    bg.querySelectorAll(".fd-row").forEach((r) => { const on = +r.dataset.i === sel; r.classList.toggle("on", on); r.setAttribute("aria-selected", on); if (on) r.scrollIntoView({ block: "nearest" }); });
  }
  function go(x) {
    if (!x) return;
    close();
    const u = new URL(x.href, location.href);
    if (u.pathname === location.pathname && u.hash === location.hash) return;
    location.href = x.href;
  }
  function open() {
    if (bg) { bg.querySelector("input").select(); return; }
    bg = document.createElement("div");
    bg.className = "fd-bg";
    bg.innerHTML = `<div class="fd" role="dialog" aria-label="Find anything">
      <label class="fd-in">${ICO(LENS)}<input type="search" placeholder="Find a topic, section or practice problem…" aria-label="Find" autocomplete="off" spellcheck="false"><kbd>Esc</kbd></label>
      <div class="fd-list" role="listbox"></div>
      <div class="fd-foot"><span><kbd>↑</kbd> <kbd>↓</kbd> move</span><span><kbd>Enter</kbd> open</span><span><kbd>Esc</kbd> close</span></div></div>`;
    document.body.appendChild(bg);
    const inp = bg.querySelector("input");
    sel = 0; paint();
    requestAnimationFrame(() => bg.classList.add("show"));
    inp.focus();
    inp.addEventListener("input", () => { sel = 0; paint(); });
    inp.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (e.key === "Enter") { e.preventDefault(); go(list[sel]); }
    });
    bg.addEventListener("mousedown", (e) => { if (e.target === bg) close(); });
    bg.addEventListener("mousemove", (e) => { const r = e.target.closest(".fd-row"); if (r && +r.dataset.i !== sel) { sel = +r.dataset.i; move(0); } });
    bg.addEventListener("click", (e) => { const r = e.target.closest(".fd-row"); if (r) { e.preventDefault(); go(list[+r.dataset.i]); } });
  }
  function close() {
    if (!bg) return;
    const el = bg; bg = null;
    el.classList.remove("show");
    setTimeout(() => el.remove(), 180);
  }

  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === "k") { e.preventDefault(); bg ? close() : open(); }
    else if (e.key === "Escape" && bg) { e.preventDefault(); e.stopPropagation(); close(); }
  }, true);

  // a small 🔍 button in the header, next to the theme switch
  const theme = document.getElementById("themeBtn");
  if (theme) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "icon-btn"; b.id = "findBtn";
    b.title = "Find anything (Ctrl K)"; b.setAttribute("aria-label", "Find anything");
    b.innerHTML = ICO(LENS);
    b.onclick = open;
    (document.querySelector(".fx-slot") || theme).before(b);
  }
})();
