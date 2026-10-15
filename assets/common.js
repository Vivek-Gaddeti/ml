/* =====================================================================
   common.js — shared engine (copied from the DSA Study Hub, adapted for the AI Study Hub)
   - Store: progress / bookmarks / notes / spaced-repetition (localStorage)
   - hlPy: tiny Python syntax highlighter
   - fmt / renderBlocks: content block renderer (with KaTeX maths: ["math", tex] and inline $…$)
   - D: SVG diagram generators
   - DocPage: sidebar + section reader with progress
   ===================================================================== */
(function () {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const DAY = 86400000;
  const INTERVALS = [1, 3, 7, 14, 30, 60]; // spaced repetition, in days
  const today = (t = Date.now()) => { const d = new Date(t); return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0"); };
  const R = String.raw;

  /* ------------------------------------------------------------- Icons
     One set of small line icons used everywhere instead of emoji: icon("flame"), icon("check", "my-class").
     24×24, drawn with the current text colour; a few (play, pause, stars…) are filled. */
  const DOT = (x, y) => `<circle cx="${x}" cy="${y}" r="1.15" fill="currentColor" stroke="none"/>`;
  const FILL = (d) => `<path d="${d}" fill="currentColor" stroke="none"/>`;
  const STAR = "M12 3.5l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9l-5.2 2.7 1-5.8-4.3-4.1 5.9-.9z";
  const ICONS = {
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    x: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    circle: '<circle cx="12" cy="12" r="7.5"/>',
    "check-circle": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/>',
    star: `<path d="${STAR}"/>`,
    "star-fill": FILL(STAR),
    repeat: '<path d="M4.5 11a7.5 7.5 0 0 1 13-4.2L19.5 9"/><path d="M19.5 4v5h-5"/><path d="M19.5 13a7.5 7.5 0 0 1-13 4.2L4.5 15"/><path d="M4.5 20v-5h5"/>',
    note: '<path d="M5 4.5h9M5 8.5h9M5 12.5h5"/><path d="M12.5 20.5l.6-3.1 6.3-6.3 2.5 2.5-6.3 6.3z"/>',
    doc: '<path d="M6 3h8.5L19 7.5V21H6z"/><path d="M14 3v4.5h5M9 12h7M9 15.5h7"/>',
    bulb: '<path d="M9.5 18h5M10.5 21h3"/><path d="M12 3a6 6 0 0 0-3.6 10.8c.7.6 1.1 1.4 1.1 2.2h5c0-.8.4-1.6 1.1-2.2A6 6 0 0 0 12 3z"/>',
    flame: '<path d="M12 22c4 0 7-2.7 7-6.8 0-3.6-2.4-6-4.6-8.7-.5 2.5-1.8 3.8-3.2 4.4C11.6 7.4 10 4.5 8 2c.3 3.6-3 6.2-3 11.2C5 18.6 8 22 12 22z"/>',
    trophy: '<path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0V4z"/><path d="M7 6H4a3 3 0 0 0 3 4M17 6h3a3 3 0 0 1-3 4"/>',
    medal: '<circle cx="12" cy="15" r="5.5"/><path d="M8 10.5 5.5 3h4L12 8.5 14.5 3h4L16 10.5"/><path d="M12 12.5v5"/>',
    party: '<path d="M4 20.5l4.5-12.5 8 8z"/><path d="M13.5 4.5c.4 1.6-.2 2.8-1.6 3.5M19.5 10.5c-1.6-.4-2.8.2-3.5 1.6"/>' + DOT(18, 4.5) + DOT(20.5, 7.5) + DOT(15, 2.5),
    sparkles: '<path d="M11 3.5l1.7 4.8 4.8 1.7-4.8 1.7L11 16.5l-1.7-4.8L4.5 10l4.8-1.7z"/><path d="M18.5 14.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/>',
    sparkle: '<path d="M12 3.5l2 6.5 6.5 2-6.5 2-2 6.5-2-6.5-6.5-2 6.5-2z"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>',
    moon: '<path d="M19.5 14.5A8 8 0 0 1 9.5 4.5a8 8 0 1 0 10 10z"/>',
    "cloud-sun": '<path d="M8 3.5v1.3M3.6 7.9h1.3M4.9 4.9l.9.9M11.1 4.9l-.9.9"/><path d="M5.3 11.3A3.4 3.4 0 1 1 11.3 7.6"/><path d="M8.5 19.5h9a3.5 3.5 0 0 0 .3-7 5 5 0 0 0-9.6 1.4 2.8 2.8 0 0 0 .3 5.6z"/>',
    sunset: '<path d="M3 17.5h18M7 21h10"/><path d="M7 17.5a5 5 0 0 1 10 0"/><path d="M12 7.5V4M5.3 10.8 3.8 9.3M18.7 10.8l1.5-1.5"/>',
    pumpkin: '<path d="M12 7.5c-4.4 0-8 2.8-8 6.3S7.6 20 12 20s8-2.8 8-6.2-3.6-6.3-8-6.3z"/><path d="M12 7.5c-1.8 0-3 2.8-3 6.3S10.2 20 12 20s3-2.8 3-6.2-1.2-6.3-3-6.3z"/><path d="M12 7.5V5c0-.9.7-1.5 1.5-1.5"/>',
    snowflake: '<path d="M12 2.5v19M3.8 7.3l16.4 9.4M3.8 16.7l16.4-9.4"/><path d="M9.5 4.5 12 6.5l2.5-2M9.5 19.5 12 17.5l2.5 2"/>',
    heart: '<path d="M12 20s-7.5-4.6-7.5-10A4.2 4.2 0 0 1 12 7.6a4.2 4.2 0 0 1 7.5 2.4c0 5.4-7.5 10-7.5 10z"/>',
    menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
    expand: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    shrink: '<path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/>',
    flag: '<path d="M5.5 21V4"/><path d="M5.5 4.5h12l-2.5 4 2.5 4h-12"/>',
    turtle: '<path d="M4 14c0-3.9 3.6-7 8-7s8 3.1 8 7H4z"/><path d="M20 12.5h1.5a1 1 0 0 0 0-2H19"/><path d="M7 14v3M17 14v3M11 14v3"/><path d="M9 7.4 10.5 10l1.5-2.6L13.5 10 15 7.4"/>',
    snake: '<path d="M3 17.5c2.5 0 2.5-3.5 5-3.5s2.5 3.5 5 3.5 2.5-3.5 5-3.5c1.5 0 2-1.6 2-3.2V8.5a2.5 2.5 0 0 0-5 0"/><path d="M17.8 8.4h.01"/>',
    puzzle: '<path d="M5 8h3.2a2.3 2.3 0 1 1 4.6 0H16v3.2a2.3 2.3 0 1 1 0 4.6V19h-3.2a2.3 2.3 0 1 0-4.6 0H5v-3.2a2.3 2.3 0 1 0 0-4.6z"/>',
    flask: '<path d="M9 3h6M10 3v6L4.8 18a2 2 0 0 0 1.7 3h11a2 2 0 0 0 1.7-3L14 9V3"/><path d="M7.3 14.5h9.4"/>',
    walk: '<circle cx="13" cy="4.5" r="1.8"/><path d="M9 21l2.5-5.5L14 17.5V21M11.5 15.5l1-5.5-3.2 1.6L8 15M12.5 10l2.5 3 3 .6"/>',
    rocket: '<path d="M12 15.5 8.5 12c1.5-5.5 5-8.5 12-8.5 0 7-3 10.5-8.5 12z"/><path d="M8.5 12 5 11.5l3-4h4M12 15.5l.5 3.5 4-3v-4"/><path d="M6.5 17.5c-1.5.5-2.5 2-3 3.5 1.5-.5 3-1.5 3.5-3"/><circle cx="15" cy="9" r="1.5"/>',
    search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.5 15.5 5 5"/>',
    brain: '<path d="M9.5 4A2.5 2.5 0 0 0 7 6.5 3 3 0 0 0 4.5 10 3 3 0 0 0 5 15a3 3 0 0 0 4.5 4.5H12V4z"/><path d="M14.5 4A2.5 2.5 0 0 1 17 6.5a3 3 0 0 1 2.5 3.5 3 3 0 0 1-.5 5 3 3 0 0 1-4.5 4.5H12V4z"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    timer: '<circle cx="12" cy="13.5" r="7.5"/><path d="M12 13.5V9.5M10 2.5h4M18.3 6.7l1.5-1.5"/>',
    play: FILL("M7.5 4.8v14.4a1 1 0 0 0 1.5.9l11.2-7.2a1 1 0 0 0 0-1.8L9 3.9a1 1 0 0 0-1.5.9z"),
    prev: FILL("M16.5 4.8v14.4a1 1 0 0 1-1.5.9L3.8 12.9a1 1 0 0 1 0-1.8L15 3.9a1 1 0 0 1 1.5.9z"),
    pause: '<rect x="6.5" y="5" width="3.6" height="14" rx="1.2" fill="currentColor" stroke="none"/><rect x="13.9" y="5" width="3.6" height="14" rx="1.2" fill="currentColor" stroke="none"/>',
    first: '<path d="M6 5v14"/>' + FILL("M18.5 6.2v11.6a.8.8 0 0 1-1.2.7L9 12.7a.8.8 0 0 1 0-1.4l8.3-5.8a.8.8 0 0 1 1.2.7z"),
    last: '<path d="M18 5v14"/>' + FILL("M5.5 6.2v11.6a.8.8 0 0 0 1.2.7l8.3-5.8a.8.8 0 0 0 0-1.4L6.7 5.5a.8.8 0 0 0-1.2.7z"),
    stop: '<rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor" stroke="none"/>',
    pencil: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13.5 6.5 4 4"/>',
    dice: '<rect x="4" y="4" width="16" height="16" rx="3.5"/>' + DOT(8.5, 8.5) + DOT(15.5, 15.5) + DOT(15.5, 8.5) + DOT(8.5, 15.5) + DOT(12, 12),
    calendar: '<rect x="3.5" y="5" width="17" height="15.5" rx="2.5"/><path d="M3.5 10h17M8 3v4M16 3v4"/>',
    inbox: '<path d="M3.5 13.5 6 5h12l2.5 8.5V19a1.5 1.5 0 0 1-1.5 1.5H5A1.5 1.5 0 0 1 3.5 19z"/><path d="M3.5 13.5H8l1.5 2.5h5l1.5-2.5h4.5"/>',
    download: '<path d="M12 4v11M7 10.5l5 5 5-5M5 20h14"/>',
    upload: '<path d="M12 16V5M7 9.5l5-5 5 5M5 20h14"/>',
    keyboard: '<rect x="2.5" y="6" width="19" height="12" rx="2.5"/>' + DOT(6.5, 10) + DOT(10, 10) + DOT(13.5, 10) + DOT(17, 10) + '<path d="M8 14.5h8"/>',
    gamepad: '<path d="M7 8h10a4.5 4.5 0 0 1 4.3 5.8l-1 3.3a2.5 2.5 0 0 1-4.3.8L14.5 16h-5L8 17.9a2.5 2.5 0 0 1-4.3-.8l-1-3.3A4.5 4.5 0 0 1 7 8z"/><path d="M8 11v3M6.5 12.5h3"/>' + DOT(15.5, 11.5) + DOT(17.5, 13.5),
    rainbow: '<path d="M3 17.5a9 9 0 0 1 18 0"/><path d="M6.5 17.5a5.5 5.5 0 0 1 11 0"/><path d="M10 17.5a2 2 0 0 1 4 0"/>',
    pie: '<path d="M3.5 13h17l-1.4 5.4a2 2 0 0 1-2 1.6H6.9a2 2 0 0 1-2-1.6z"/><path d="M4.5 13c1-4 4.2-6.5 7.5-6.5s6.5 2.5 7.5 6.5"/><path d="M9.5 9.5l1 1.5M14.5 9.5l-1 1.5"/>',
    cool: '<circle cx="12" cy="12" r="9"/><path d="M6 10h12"/><path d="M6.8 10c.2 2.2 3.6 2.2 3.8 0M13.4 10c.2 2.2 3.6 2.2 3.8 0M9 15.5c1.8 1.3 4.2 1.3 6 0"/>',
    hash: '<path d="M5 9h15M4 15h15M10 3.5 8 20.5M16 3.5l-2 17"/>',
    cursor: '<path d="M5.5 3.5 18 10l-5.4 1.8L10 17.5z"/><path d="M13 13l5 5"/>',
    aurora: '<path d="M3 8c3-3 6 3 9 0s6-3 9 0M3 13c3-3 6 3 9 0s6-3 9 0M3 18c3-3 6 3 9 0s6-3 9 0"/>',
    enter: '<path d="M4 4.5v6a4 4 0 0 0 4 4h12"/><path d="m15.5 10 4.5 4.5-4.5 4.5"/>',
    "arrow-right": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    return: '<path d="M9 14 4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/>',
    laugh: '<circle cx="12" cy="12" r="9"/><path d="M7.8 13.5c.9 2.3 2.4 3.5 4.2 3.5s3.3-1.2 4.2-3.5z"/>' + DOT(9, 9.5) + DOT(15, 9.5),
    up: '<path d="M12 19V5M6 11l6-6 6 6"/>',
    restart: '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3L4.5 8.9"/><path d="M4.5 4.2v4.7h4.7"/>',
    bot: '<rect x="4" y="8" width="16" height="11.5" rx="3.2"/><path d="M12 4.5V8"/><circle cx="12" cy="3.6" r="1.1"/><path d="M9 13v1.6M15 13v1.6M2.5 12.5v3M21.5 12.5v3"/>',
    sigma: '<path d="M17.5 6V4h-11l6 8-6 8h11v-2"/>',
    laptop: '<rect x="4" y="5" width="16" height="11" rx="1.8"/><path d="M2 19.5h20"/>',
  };
  const icon = (name, cls = "") => ICONS[name] ? `<svg class="ico${cls ? " " + cls : ""}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>` : "";

  /* ------------------------------------------------------------- Store */
  const Store = {
    key: (window.SITE && SITE.storeKey) || "study:v1",
    d: null,
    load() {
      if (this.d) return this.d;
      let raw = null;
      try { raw = localStorage.getItem(this.key); } catch (e) {}
      try { this.d = raw ? JSON.parse(raw) : null; } catch (e) { this.d = null; }
      if (!this.d || typeof this.d !== "object") this.d = {};
      this.d.items = this.d.items || {};
      this.d.activity = this.d.activity || {};
      return this.d;
    },
    save() {
      try { localStorage.setItem(this.key, JSON.stringify(this.d)); } catch (e) {}
      document.dispatchEvent(new CustomEvent("store:change"));
    },
    item(k) { return this.load().items[k] || {}; },
    patch(k, p) {
      const d = this.load();
      d.items[k] = Object.assign({}, d.items[k], p, { updated: Date.now() });
      this.save();
      return d.items[k];
    },
    bump() { const d = this.load(), t = today(); d.activity[t] = (d.activity[t] || 0) + 1; },
    /* status: "done" | "revise" | null */
    setStatus(k, status) {
      const it = this.item(k), p = { status };
      if (status) {
        if (!it.doneAt) { p.doneAt = Date.now(); p.level = 0; p.next = Date.now() + DAY; }
        if (status === "revise") p.next = Date.now();
        this.bump();
      } else { p.doneAt = null; p.next = null; p.level = 0; }
      const r = this.patch(k, p);
      document.dispatchEvent(new CustomEvent("item:status", { detail: { key: k, was: it.status || null, status: status || null } }));
      return r;
    },
    toggleStar(k) { return this.patch(k, { star: !this.item(k).star }); },
    setNote(k, note) { return this.patch(k, { note }); },
    /* ok=true → remembered (interval grows); ok=false → forgot (back to 1 day) */
    review(k, ok) {
      const it = this.item(k);
      const level = ok ? Math.min((it.level || 0) + 1, INTERVALS.length - 1) : 0;
      this.bump();
      return this.patch(k, {
        level, next: Date.now() + INTERVALS[level] * DAY, lastReview: Date.now(),
        reviews: (it.reviews || 0) + 1, status: ok ? "done" : "revise", lapses: (it.lapses || 0) + (ok ? 0 : 1),
        doneAt: it.doneAt || Date.now(),
      });
    },
    isDue(it) { return it && it.status && it.next && it.next <= Date.now() + 60000; },
    countDone(prefix) { return Object.entries(this.load().items).filter(([k, v]) => k.startsWith(prefix) && v.status).length; },
    streak() {
      const a = this.load().activity; let s = 0, t = Date.now();
      if (!a[today(t)]) t -= DAY; // streak still alive if you studied yesterday
      while (a[today(t)]) { s++; t -= DAY; }
      return s;
    },
    // longest run of consecutive study days ever
    bestStreak() {
      const a = this.load().activity, days = Object.keys(a).filter((d) => a[d]).sort();
      let best = 0, run = 0, prev = null;
      days.forEach((d) => { const t = new Date(d + "T00:00:00").getTime(); run = prev !== null && Math.round((t - prev) / DAY) === 1 ? run + 1 : 1; best = Math.max(best, run); prev = t; });
      return best;
    },
    exportJSON() {
      const blob = new Blob([JSON.stringify(this.load(), null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = this.key.replace(/[:]/g, "-") + "-" + today() + ".json";
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    },
    importJSON(file, cb) {
      const r = new FileReader();
      r.onload = () => {
        try {
          const d = JSON.parse(r.result);
          if (!d.items) throw new Error("bad file");
          this.d = { items: Object.assign({}, this.load().items, d.items), activity: Object.assign({}, this.load().activity, d.activity || {}) };
          this.save(); toast("Progress imported", 1800, "check"); cb && cb();
        } catch (e) { toast("Could not import that file"); }
      };
      r.readAsText(file);
    },
    reset() { this.d = { items: {}, activity: {} }; this.save(); },
  };

  /* ------------------------------------------------------------- Toast */
  let toastEl, toastT;
  function toast(msg, ms = 1800, ic = "") {
    if (!toastEl) { toastEl = document.createElement("div"); toastEl.className = "toast"; document.body.appendChild(toastEl); }
    toastEl.innerHTML = icon(ic) + `<span>${esc(msg)}</span>`; toastEl.classList.add("show");
    clearTimeout(toastT); toastT = setTimeout(() => toastEl.classList.remove("show"), ms);
  }

  /* ------------------------------------------------------------- Fixed-width button labels
     swapLabel(["Copy", "Copied ✓"], 0) stacks every label in the same spot and shows one, so the
     button is always as wide as its longest label and nothing next to it jumps when the text changes. */
  const swapLabel = (labels, s = 0) => `<span class="swap" data-s="${s}">${labels.map((l) => `<span>${l}</span>`).join("")}</span>`;
  const setSwap = (btn, s) => { const w = btn && btn.querySelector(".swap"); if (w) w.dataset.s = s; };

  /* ------------------------------------------------------------- Python highlighter */
  const KW = new Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case".split(" "));
  const BI = new Set("print len range enumerate zip map filter sorted reversed sum min max abs int str float list dict set tuple bool type isinstance issubclass super object open input iter next any all ord chr hex bin oct round pow divmod hash id repr format frozenset bytes bytearray callable getattr setattr hasattr delattr vars dir self cls slice complex memoryview globals locals Exception ValueError TypeError KeyError IndexError StopIteration ZeroDivisionError AttributeError RuntimeError NameError NotImplementedError".split(" "));
  const TOK = /(#[^\n]*)|([rRbBfFuU]{0,2}"""[\s\S]*?"""|[rRbBfFuU]{0,2}'''[\s\S]*?'''|[rRbBfFuU]{0,2}"(?:\\.|[^"\\\n])*"|[rRbBfFuU]{0,2}'(?:\\.|[^'\\\n])*')|(@[A-Za-z_][\w.]*)|(\b0[xXoObB][\da-fA-F_]+\b|\b\d[\d_]*(?:\.\d*)?(?:[eE][+-]?\d+)?j?\b)|([A-Za-z_]\w*)/g;
  function hlPy(src) {
    let out = "", last = 0, m, prevKw = "";
    TOK.lastIndex = 0;
    while ((m = TOK.exec(src))) {
      out += esc(src.slice(last, m.index));
      const t = m[0];
      if (m[1]) out += `<span class="tok-com">${esc(t)}</span>`;
      else if (m[2]) out += `<span class="tok-str">${esc(t)}</span>`;
      else if (m[3]) out += `<span class="tok-dec">${esc(t)}</span>`;
      else if (m[4]) out += `<span class="tok-num">${esc(t)}</span>`;
      else if (KW.has(t)) out += `<span class="tok-kw">${t}</span>`;
      else if (prevKw === "def" || prevKw === "class") out += `<span class="tok-fn">${t}</span>`;
      else if (BI.has(t)) out += `<span class="tok-bi">${t}</span>`;
      else out += t;
      prevKw = m[5] ? t : "";
      last = TOK.lastIndex;
    }
    return out + esc(src.slice(last));
  }

  /* ------------------------------------------------------------- Maths (KaTeX, saved in assets/vendor/katex so it works offline)
     tex("\\frac{a}{b}") inline, tex(s, true) as a centred display line. Without KaTeX it falls back to the raw TeX. */
  function tex(s, display) {
    if (window.katex) {
      try { return katex.renderToString(String(s), { displayMode: !!display, throwOnError: false, output: "html" }); } catch (e) {}
    }
    return `<code>${esc(s)}</code>`;
  }

  /* ------------------------------------------------------------- Inline text format
     `code`  **bold**  *italic*  [label](url)  $maths$ (no space just inside the dollars, so "$5 and $10" stays text) */
  const isTex = (q) => q.length > 2 && q[0] === "$" && q[q.length - 1] === "$" && q[1] !== " " && q[q.length - 2] !== " ";
  function fmt(t) {
    if (t == null) return "";
    const parts = String(t).split(/(`[^`]+`)/g);
    return parts.map((p) => {
      if (p.startsWith("`") && p.endsWith("`") && p.length > 1) return `<code>${esc(p.slice(1, -1))}</code>`;
      return p.split(/(\$[^$\n]+?\$)/g).map((q) => isTex(q) ? `<span class="math-i">${tex(q.slice(1, -1))}</span>` : esc(q)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|[\s(])\*(?!\s)(.+?)\*(?=[\s).,:;!?]|$)/g, "$1<em>$2</em>")
        .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, a, b) => `<a href="${b}" ${b.startsWith("http") ? 'target="_blank" rel="noopener"' : ""}>${a}</a>`)).join("");
    }).join("");
  }

  /* ------------------------------------------------------------- Code block */
  const CODE = [];
  /* same hash as tools/build.js: identifies a code sample's outmap + trace */
  function fnv(s) { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16).padStart(8, "0"); }
  /* split highlighted HTML into lines, closing/reopening a span that crosses a newline */
  function htmlLines(html) {
    const lines = [], re = /(<span[^>]*>)|(<\/span>)|(\n)|([^<\n]+)/g;
    let cur = "", open = null, m;
    while ((m = re.exec(html))) {
      if (m[1]) { open = m[1]; cur += m[1]; }
      else if (m[2]) { open = null; cur += m[2]; }
      else if (m[3]) { if (open) cur += "</span>"; lines.push(cur); cur = open || ""; }
      else cur += m[4];
    }
    lines.push(cur);
    return lines;
  }
  let outMode = "inline";
  try { outMode = localStorage.getItem("code:outmode") || "inline"; } catch (e) {}
  document.documentElement.classList.toggle("out-end", outMode === "end");

  /* local: code that can't run in the browser (PyTorch, Hugging Face, LLM APIs…): shown read-only with a
     "Runs on your computer" badge, and skipped by tools/verify.js and tools/build.js.
     Plots: tools/build.js saves each matplotlib figure an example draws as data/plots/<hash>-<n>.png (PLOTMAP says how many). */
  function codeBlock(code, output, title, local) {
    code = String(code).replace(/^\n+|\s+$/g, "");
    const i = CODE.push(code) - 1, h = fnv(code);
    const om = local ? null : (window.OUTMAP || {})[h], plots = local ? 0 : (window.PLOTMAP || {})[h] || 0;
    const after = {};
    if (om) om.forEach(([l, t]) => { const k = l || "end"; after[k] = after[k] != null ? after[k] + "\n" + t : t; });
    const lines = htmlLines(hlPy(code));
    const isPy = !title || /python|pytest|numpy|pandas|matplotlib|scikit/i.test(title);
    const tools = (om ? `<button class="btn" data-step="${i}" title="Run the code one line at a time and watch the variables">${swapLabel([icon("play") + " Step through", "Loading…", icon("x") + " Close stepper"])}</button>
        <button class="btn" data-outmode title="Show each output under the line that printed it, or all together at the end">${swapLabel(["Outputs: inline", "Outputs: at end"], outMode === "end" ? 1 : 0)}</button>` : "") +
      (isPy && !local ? `<button class="btn" data-try="${i}" title="Change this code and run it right here">${swapLabel([icon("pencil") + " Edit & run", icon("x") + " Close editor"])}</button>` : "") +
      `<button class="btn" data-copy="${i}">${swapLabel(["Copy", "Copied " + icon("check")])}</button>`;
    let body = lines.map((ln, k) => `<div class="cl" data-ln="${k + 1}"><span class="gut">${k + 1}</span><span class="src">${ln || " "}</span></div>` +
      (after[k + 1] != null ? `<div class="co"><span class="gut">⤷</span><pre title="output printed by line ${k + 1}">${esc(after[k + 1])}</pre></div>` : "")).join("");
    if (after.end != null) body += `<div class="co"><span class="gut">⤷</span><pre>${esc(after.end)}</pre></div>`;
    const badge = local ? `<span class="code-local" title="PyTorch, Hugging Face and LLM APIs can't run inside the browser. Copy this and run it with Python on your computer.">${icon("laptop")} Runs on your computer</span>` : "";
    let html = `<div class="code-wrap${om ? " has-om" : ""}${local ? " is-local" : ""}" data-h="${h}" data-i="${i}"><div class="code-head"><span>${esc(title || "python")}${badge}</span><span class="code-tools">${tools}</span></div><div class="code-lines${isPy ? "" : " plain"}">${body}</div>`;
    if (output != null && output !== "") {
      const o = String(output).replace(/^\n+|\s+$/g, "");
      html += `<div class="output end-out"><div class="code-head"><span>${local ? "output (on a computer)" : "output"}</span></div><pre>${esc(o)}</pre></div>`;
    }
    if (plots) html += `<div class="plots">${Array.from({ length: plots }, (_, k) => `<img class="plot-img" src="data/plots/${h}-${k}.png" alt="plot drawn by this code" loading="lazy">`).join("")}</div>`;
    return html + `<div class="stepper" hidden></div></div>`;
  }

  /* ---- output mode toggle (global, remembered) ---- */
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-outmode]");
    if (!b) return;
    outMode = outMode === "end" ? "inline" : "end";
    try { localStorage.setItem("code:outmode", outMode); } catch (x) {}
    document.documentElement.classList.toggle("out-end", outMode === "end");
    document.querySelectorAll("[data-outmode]").forEach((x) => setSwap(x, outMode === "end" ? 1 : 0));
  });

  /* ---- step-through player (Python-Tutor style) ---- */
  function loadTrace(h, cb) {
    if (window.TRACES && TRACES[h]) return cb(TRACES[h]);
    const s = document.createElement("script");
    s.charset = "utf-8";
    s.src = ((window.SITE && SITE.traceBase) || "data/traces/") + h + ".js";
    s.onload = () => cb(window.TRACES && TRACES[h]);
    s.onerror = () => cb(null);
    document.head.appendChild(s);
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-step]");
    if (!b) return;
    const wrap = b.closest(".code-wrap"), panel = wrap.querySelector(".stepper");
    if (!panel.hidden) { stopStepper(wrap); return; }
    setSwap(b, 1);
    loadTrace(wrap.dataset.h, (tr) => {
      if (!tr) { setSwap(b, 0); toast("Step-through isn't available for this example"); return; }
      setSwap(b, 2);
      startStepper(wrap, tr);
    });
  });
  function stopStepper(wrap) {
    const panel = wrap.querySelector(".stepper");
    clearInterval(panel._timer);
    panel.hidden = true;
    wrap.classList.remove("stepping");
    wrap.querySelectorAll(".cl").forEach((x) => x.classList.remove("cur", "prev"));
    const b = wrap.querySelector("[data-step]");
    if (b) setSwap(b, 0);
  }
  function startStepper(wrap, tr) {
    const panel = wrap.querySelector(".stepper"), steps = tr.steps, lines = CODE[+wrap.dataset.i].split("\n");
    let G = {};
    steps.forEach((s) => { if (s.g) G = s.g; s.G = G; });          // carry globals forward
    let k = 0;
    panel.hidden = false;
    wrap.classList.add("stepping");
    panel.innerHTML = `
      <div class="st-ctrl">
        <button class="btn sm" data-a="first" title="first step">${icon("first")}</button>
        <button class="btn sm" data-a="prev">${icon("prev")} Prev</button>
        <button class="btn sm primary" data-a="play">${swapLabel([icon("play") + " Play", icon("pause") + " Pause"])}</button>
        <button class="btn sm" data-a="next">Next ${icon("play")}</button>
        <button class="btn sm" data-a="last" title="last step">${icon("last")}</button>
        <input type="range" min="0" max="${steps.length - 1}" value="0" aria-label="step">
        <span class="st-count"></span>
      </div>
      <div class="st-msg"></div>
      <div class="st-grid"><div class="st-vars"></div><div class="st-out"><div class="st-label">Output so far</div><pre></pre></div></div>
      ${tr.truncated ? `<div class="small muted st-note">Only the first ${tr.max || 400} steps are recorded.</div>` : ""}`;
    const range = panel.querySelector("input"), playB = panel.querySelector('[data-a="play"]');
    const snippet = (l) => (lines[l - 1] || "").trim();
    const table = (title, obj, prev) => {
      const keys = Object.keys(obj || {});
      if (!keys.length) return "";
      return `<div class="st-label">${title}</div><table class="st-table">${keys.map((n) => {
        const changed = prev && prev[n] !== obj[n];
        return `<tr class="${changed ? "chg" : ""}"><td>${esc(n)}</td><td>${esc(obj[n])}</td></tr>`;
      }).join("")}</table>`;
    };
    const draw = () => {
      const s = steps[k], p = steps[k - 1];
      wrap.querySelectorAll(".cl").forEach((x) => x.classList.remove("cur", "prev"));
      if (p && p.l) { const pe = wrap.querySelector(`.cl[data-ln="${p.l}"]`); if (pe) pe.classList.add("prev"); }
      if (s.l) { const ce = wrap.querySelector(`.cl[data-ln="${s.l}"]`); if (ce) { ce.classList.add("cur"); ce.scrollIntoView({ block: "nearest" }); } }
      const fn = s.s && s.s.length ? s.s[s.s.length - 1] : "<module>";
      const sameFrame = p && p.s && s.s && p.s.length === s.s.length && p.s.join() === s.s.join();
      // what changed because of the previous line
      const changes = [];
      if (p) {
        const diff = (a, b) => Object.keys(b || {}).forEach((n) => { if (!a || a[n] !== b[n]) changes.push(a && n in a ? `<code>${esc(n)}</code>: ${esc(a[n])} → <strong>${esc(b[n])}</strong>` : `<code>${esc(n)}</code> = <strong>${esc(b[n])}</strong>`); });
        diff(p.G, s.G);
        if (sameFrame) diff(p.v, s.v);
      }
      let msg = "";
      if (s.e === "end") msg = `${icon("check-circle", "text-green")} <strong>Finished.</strong> The program ran ${steps.length - 1} steps.`;
      else if (s.e === "r") msg = `${icon("return")} <strong>${esc(fn)}()</strong> returns <code>${esc(s.r)}</code>`;
      else if (p && !sameFrame && p.s && s.s && s.s.length > p.s.length) msg = `${icon("enter")} Entered <strong>${esc(fn)}()</strong> (call depth ${s.s.length - 1}). Next line to run: <strong>line ${s.l}</strong> <code>${esc(snippet(s.l))}</code>`;
      else msg = `${icon("arrow-right")} Next line to run: <strong>line ${s.l}</strong>${fn !== "<module>" ? ` inside <strong>${esc(fn)}()</strong>` : ""} <code>${esc(snippet(s.l))}</code>`;
      if (p && p.l && s.e !== "end" || (s.e === "end" && p)) {
        if (changes.length) msg += `<div class="st-chg">Line ${p.l} just ran and changed: ${changes.slice(0, 6).join(", ")}${changes.length > 6 ? " …" : ""}</div>`;
        if (s.o > p.o) msg += `<div class="st-chg">Line ${p.l} printed: <code>${esc(tr.out.slice(p.o, s.o).trim())}</code></div>`;
      }
      panel.querySelector(".st-msg").innerHTML = msg;
      panel.querySelector(".st-vars").innerHTML =
        `<div class="st-label">Call stack</div><div class="st-stack">${(s.s && s.s.length ? s.s : ["<module>"]).map((x) => esc(x === "<module>" ? "main program" : x + "()")).join(" → ")}</div>` +
        (s.v && Object.keys(s.v).length ? table(`Local variables of ${esc(fn)}()`, s.v, sameFrame ? p.v : null) : "") +
        (table("Global variables", s.G, p ? p.G : null) || (!s.v ? '<div class="small muted">No variables yet.</div>' : ""));
      panel.querySelector(".st-out pre").textContent = tr.out.slice(0, s.o) || " ";
      panel.querySelector(".st-count").textContent = `step ${k + 1} / ${steps.length}`;
      range.value = k;
    };
    const stop = () => { clearInterval(panel._timer); panel._timer = null; setSwap(playB, 0); };
    panel.onclick = (e) => {
      const a = e.target.closest("[data-a]"); if (!a) return;
      const act = a.dataset.a;
      if (act !== "play") stop();
      if (act === "first") k = 0;
      if (act === "last") k = steps.length - 1;
      if (act === "prev") k = Math.max(0, k - 1);
      if (act === "next") k = Math.min(steps.length - 1, k + 1);
      if (act === "play") {
        if (panel._timer) return stop();
        if (k >= steps.length - 1) k = 0;
        setSwap(playB, 1);
        panel._timer = setInterval(() => { if (k >= steps.length - 1) return stop(); k++; draw(); }, 800);
      }
      draw();
    };
    range.oninput = () => { stop(); k = +range.value; draw(); };
    draw();
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-copy]");
    if (!b) return;
    const txt = CODE[+b.dataset.copy];
    const done = () => { setSwap(b, 1); clearTimeout(b._t); b._t = setTimeout(() => setSwap(b, 0), 1400); };
    if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, () => fallbackCopy(txt, done));
    else fallbackCopy(txt, done);
  });
  function fallbackCopy(t, cb) { const ta = document.createElement("textarea"); ta.value = t; document.body.appendChild(ta); ta.select(); try { document.execCommand("copy"); } catch (e) {} ta.remove(); cb(); }

  /* ------------------------------------------------------------- Block renderer
     ['p', text] ['h', text] ['h2', text] ['list', [..]] ['ol', [..]] ['steps', [..]]
     ['code', code, output?, title?, local?] ['table', headers, rows] ['tip'|'note'|'warn'|'danger', text]
     ['math', tex, caption?] ['diagram', type, args, caption] ['viz', name] ['ex', {...}] ['qa', q, a] ['html', raw]  */
  function renderBlocks(blocks) {
    return (blocks || []).map(renderBlock).join("\n");
  }
  function renderBlock(b) {
    if (typeof b === "string") return `<p>${fmt(b)}</p>`;
    const [t, a, c, d, e] = b;
    switch (t) {
      case "math": return `<div class="math-d">${tex(a, true)}${c ? `<div class="cap">${fmt(c)}</div>` : ""}</div>`;
      case "p": return `<p>${fmt(a)}</p>`;
      case "h": return `<h3>${fmt(a)}</h3>`;
      case "h2": return `<h2>${fmt(a)}</h2>`;
      case "list": return `<ul class="clean">${a.map((x) => `<li>${fmt(x)}</li>`).join("")}</ul>`;
      case "ol": return `<ol class="clean">${a.map((x) => `<li>${fmt(x)}</li>`).join("")}</ol>`;
      case "steps": return `<ol class="steps">${a.map((x) => `<li>${fmt(x)}</li>`).join("")}</ol>`;
      case "code": return codeBlock(a, c, d, e);
      case "table": return table(a, c);
      case "tip": case "note": case "warn": case "danger": {
        const label = { tip: "Tip", note: "Note", warn: "Watch out", danger: "Common mistake" }[t];
        const cls = t === "note" ? "" : t;
        return `<div class="callout ${cls}"><span class="label">${c || label}</span>${fmt(a)}</div>`;
      }
      case "diagram": return `<div class="diagram">${D[a] ? D[a](...(c || [])) : ""}${d ? `<div class="cap">${fmt(d)}</div>` : ""}</div>`;
      case "viz": return `<div class="viz" data-viz="${a}"></div>`;
      case "ex": return example(a);
      case "qa": return `<details class="box"><summary>${fmt(a)}</summary><div class="body">${Array.isArray(c) ? renderBlocks(c) : `<p>${fmt(c)}</p>`}</div></details>`;
      case "html": return a;
      default: return "";
    }
  }
  function table(head, rows) {
    return `<div class="table-wrap"><table><thead><tr>${head.map((h) => `<th>${fmt(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${fmt(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  /* worked example (used in patterns) */
  function example(e) {
    const diff = e.d ? `<span class="badge ${e.d.toLowerCase()}">${e.d}</span>` : "";
    const lc = e.lc ? `<span class="badge">LC ${e.lc}</span>` : "";
    return `<details class="box"${e.open ? " open" : ""}><summary>${fmt(e.t)} ${diff} ${lc}</summary><div class="body">
      ${e.q ? `<p><strong>Problem:</strong> ${fmt(e.q)}</p>` : ""}
      ${e.idea ? `<p><strong>How the pattern applies:</strong> ${fmt(e.idea)}</p>` : ""}
      ${e.steps ? `<ol class="steps">${e.steps.map((x) => `<li>${fmt(x)}</li>`).join("")}</ol>` : ""}
      ${e.code ? codeBlock(e.code, e.out) : ""}
      ${e.cx ? `<p class="small muted"><strong>Complexity:</strong> ${fmt(e.cx)}</p>` : ""}
    </div></details>`;
  }

  /* ------------------------------------------------------------- SVG diagrams */
  function svg(W, H, inner) {
    return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img">
      <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z"/></marker>
      <marker id="ahA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--accent)"/></marker></defs>${inner}</svg>`;
  }
  const T = (x, y, s, cls = "val", anchor = "middle", extra = "") => `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}" ${extra}>${esc(s)}</text>`;
  const cls = (o, i) => (o.hl || []).includes(i) ? "node hl" : (o.ok || []).includes(i) ? "node ok" : (o.bad || []).includes(i) ? "node bad" : "node";

  const D = {
    /* array: D.arr([1,2,3], {hl:[0], ptr:{0:'L'}, title:'nums'}) */
    arr(vals, o = {}) {
      const w = 44, g = 4, n = vals.length, lx = o.title ? 60 : 10;
      const top = o.ptr ? 30 : 10, W = lx + n * (w + g) + 10, H = top + w + (o.idx === false ? 10 : 26);
      let s = o.title ? T(lx - 10, top + w / 2 + 5, o.title, "lbl", "end") : "";
      vals.forEach((v, i) => {
        const x = lx + i * (w + g);
        s += `<rect class="${cls(o, i)}" x="${x}" y="${top}" width="${w}" height="${w}" rx="7"/>` + T(x + w / 2, top + w / 2 + 5, v);
        if (o.idx !== false) s += T(x + w / 2, top + w + 17, i, "lbl");
        if (o.ptr && o.ptr[i] != null) s += T(x + w / 2, top - 10, o.ptr[i], "acc", "middle", 'font-size="12" font-weight="700"');
      });
      return svg(W, H, s);
    },
    /* linked list: D.list([1,2,3], {cycle:1, hl:[0], ptr:{0:'head'}}) */
    list(vals, o = {}) {
      const w = 50, h = 36, g = 34, top = o.ptr ? 30 : 12, n = vals.length;
      const W = 20 + n * (w + g) + 40, H = top + h + (o.cycle != null ? 46 : 16);
      let s = "";
      vals.forEach((v, i) => {
        const x = 10 + i * (w + g);
        s += `<rect class="${cls(o, i)}" x="${x}" y="${top}" width="${w}" height="${h}" rx="7"/>` + T(x + w / 2, top + h / 2 + 5, v);
        if (o.ptr && o.ptr[i] != null) s += T(x + w / 2, top - 10, o.ptr[i], "acc", "middle", 'font-size="12" font-weight="700"');
        if (i < n - 1) s += `<line class="edge" x1="${x + w}" y1="${top + h / 2}" x2="${x + w + g - 3}" y2="${top + h / 2}" marker-end="url(#ah)"/>`;
      });
      const lx = 10 + n * (w + g) - g;
      if (o.cycle != null) {
        const tx = 10 + o.cycle * (w + g) + w / 2;
        s += `<path class="edge hl" d="M${lx - w / 2},${top + h} C${lx - w / 2},${top + h + 40} ${tx},${top + h + 40} ${tx},${top + h + 3}" marker-end="url(#ahA)"/>`;
      } else {
        s += `<line class="edge" x1="${lx}" y1="${top + h / 2}" x2="${lx + 22}" y2="${top + h / 2}" marker-end="url(#ah)"/>` + T(lx + 34, top + h / 2 + 4, "None", "lbl", "start");
      }
      return svg(W + 20, H, s);
    },
    /* binary tree in heap order (null = missing): D.tree([4,2,7,1,3,6,9], {hl:[0]}) */
    tree(a, o = {}) {
      const n = a.length, depth = Math.floor(Math.log2(n)) + 1, r = 19;
      const W = Math.max(Math.pow(2, depth - 1) * (o.gap || 54), 160), H = depth * 66 + 10;
      const pos = (i) => { const d = Math.floor(Math.log2(i + 1)), p = i - (Math.pow(2, d) - 1); return [(p + 0.5) * W / Math.pow(2, d), 28 + d * 66]; };
      let e = "", s = "";
      a.forEach((v, i) => {
        if (v == null) return;
        const [x, y] = pos(i);
        if (i > 0 && a[(i - 1) >> 1] != null) { const [px, py] = pos((i - 1) >> 1); e += `<line class="edge${(o.he || []).includes(i) ? " hl" : ""}" x1="${px}" y1="${py}" x2="${x}" y2="${y}"/>`; }
        s += `<circle class="${cls(o, i)}" cx="${x}" cy="${y}" r="${r}"/>` + T(x, y + 5, v, "val", "middle", String(v).length > 2 ? 'font-size="11"' : "");
        if (o.ptr && o.ptr[i] != null) s += T(x + r + 4, y - r + 4, o.ptr[i], "acc", "start", 'font-size="11" font-weight="700"');
      });
      return svg(W, H, e + s);
    },
    /* graph: D.graph({A:[x,y],...}, [['A','B',w?]], {directed:true, hl:['A'], he:['A-B']}) */
    graph(nodes, edges, o = {}) {
      const r = 20, xs = Object.values(nodes);
      const W = Math.max(...xs.map((p) => p[0])) + 40, H = Math.max(...xs.map((p) => p[1])) + 40;
      let e = "", s = "";
      edges.forEach(([a, b, w]) => {
        const [x1, y1] = nodes[a], [x2, y2] = nodes[b];
        const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
        const hl = (o.he || []).includes(a + "-" + b) || (!o.directed && (o.he || []).includes(b + "-" + a));
        e += `<line class="edge${hl ? " hl" : ""}" x1="${x1 + ux * r}" y1="${y1 + uy * r}" x2="${x2 - ux * (r + 3)}" y2="${y2 - uy * (r + 3)}" ${o.directed ? `marker-end="url(#${hl ? "ahA" : "ah"})"` : ""}/>`;
        if (w != null) e += `<text class="lbl" x="${(x1 + x2) / 2 - uy * 12}" y="${(y1 + y2) / 2 + ux * 12 + 4}" text-anchor="middle" font-weight="700">${esc(w)}</text>`;
      });
      Object.entries(nodes).forEach(([id, [x, y]]) => {
        const c = (o.hl || []).includes(id) ? "node hl" : (o.ok || []).includes(id) ? "node ok" : (o.bad || []).includes(id) ? "node bad" : "node";
        s += `<circle class="${c}" cx="${x}" cy="${y}" r="${r}"/>` + T(x, y + 5, id);
        if (o.lab && o.lab[id] != null) s += T(x, y - r - 7, o.lab[id], "acc", "middle", 'font-size="11" font-weight="700"');
      });
      return svg(W, H, e + s);
    },
    /* stack (bottom → top) */
    stack(vals, o = {}) {
      const w = 110, h = 34, n = vals.length, W = 260, H = n * (h + 4) + 40;
      let s = `<path class="edge" d="M60,10 L60,${H - 14} L${60 + w + 20},${H - 14} L${60 + w + 20},10"/>`;
      vals.forEach((v, i) => {
        const y = H - 18 - (i + 1) * (h + 4) + 2;
        s += `<rect class="${i === n - 1 ? "node hl" : "node"}" x="70" y="${y}" width="${w}" height="${h}" rx="6"/>` + T(70 + w / 2, y + h / 2 + 5, v);
        if (i === n - 1) s += T(70 + w + 30, y + h / 2 + 5, "← top", "acc", "start", 'font-size="12" font-weight="700"');
      });
      return svg(W, H, s);
    },
    /* queue (front ... back) */
    queue(vals) {
      return D.arr(vals, { ptr: { 0: "front", [vals.length - 1]: "back" }, idx: false });
    },
    /* hash table: buckets with chains. entries: [[key, bucket]] */
    hash(entries, m = 5) {
      const bh = 34, W = 520, H = m * (bh + 8) + 20;
      let s = "";
      for (let b = 0; b < m; b++) {
        const y = 10 + b * (bh + 8);
        s += `<rect class="node" x="70" y="${y}" width="44" height="${bh}" rx="6"/>` + T(92, y + bh / 2 + 5, b) + T(60, y + bh / 2 + 5, "bucket", "lbl", "end");
        const chain = entries.filter((e) => e[1] === b);
        chain.forEach((e, j) => {
          const x = 150 + j * 125;
          s += `<line class="edge" x1="${x - 34 + (j ? 0 : -2)}" y1="${y + bh / 2}" x2="${x - 4}" y2="${y + bh / 2}" marker-end="url(#ah)"/>`;
          s += `<rect class="node ok" x="${x}" y="${y}" width="92" height="${bh}" rx="6"/>` + T(x + 46, y + bh / 2 + 5, e[0], "val", "middle", 'font-size="12"');
        });
      }
      return svg(W, H, s);
    },
    /* trie from words */
    trie(words) {
      const root = { c: {}, end: false, ch: "" };
      words.forEach((w) => { let n = root; for (const ch of w) { n = n.c[ch] = n.c[ch] || { c: {}, end: false, ch }; } n.end = true; });
      let leaf = 0, maxD = 0; const nodes = [];
      (function lay(n, d) {
        maxD = Math.max(maxD, d);
        const kids = Object.values(n.c).sort((a, b) => a.ch.localeCompare(b.ch));
        if (!kids.length) { n.x = leaf++; } else { kids.forEach((k) => lay(k, d + 1)); n.x = (kids[0].x + kids[kids.length - 1].x) / 2; }
        n.d = d; n.kids = kids; nodes.push(n);
      })(root, 0);
      const sx = 52, W = Math.max(leaf * sx + 20, 120), H = (maxD + 1) * 58 + 10, P = (n) => [20 + n.x * sx + 6, 26 + n.d * 58];
      let e = "", s = "";
      nodes.forEach((n) => {
        const [x, y] = P(n);
        n.kids.forEach((k) => { const [kx, ky] = P(k); e += `<line class="edge" x1="${x}" y1="${y}" x2="${kx}" y2="${ky}"/>`; });
        s += `<circle class="${n.end ? "node ok" : "node"}" cx="${x}" cy="${y}" r="16"/>` + T(x, y + 5, n.ch || "•");
      });
      return svg(W, H, e + s);
    },
    /* grid / matrix: D.grid([[1,0],[0,1]], {hl:['0,0'], ok:[], head:true}) */
    grid(m, o = {}) {
      const c = 38, rows = m.length, cols = m[0].length, off = o.head ? 26 : 6;
      const W = off + cols * (c + 3) + 6, H = off + rows * (c + 3) + 6;
      let s = "";
      if (o.head) { for (let j = 0; j < cols; j++) s += T(off + j * (c + 3) + c / 2, 17, j, "lbl"); for (let i = 0; i < rows; i++) s += T(13, off + i * (c + 3) + c / 2 + 4, i, "lbl"); }
      m.forEach((row, i) => row.forEach((v, j) => {
        const k = i + "," + j, x = off + j * (c + 3), y = off + i * (c + 3);
        const k2 = (o.hl || []).includes(k) ? "node hl" : (o.ok || []).includes(k) ? "node ok" : (o.bad || []).includes(k) ? "node bad" : "node";
        s += `<rect class="${k2}" x="${x}" y="${y}" width="${c}" height="${c}" rx="5"/>` + T(x + c / 2, y + c / 2 + 5, v, "val", "middle", 'font-size="13"');
      }));
      return svg(W, H, s);
    },
    /* Big-O growth chart */
    bigo() {
      const W = 560, H = 300, L = 50, B = 260, Rr = 540, Tp = 20, N = 20, YM = 60;
      const fns = [["O(1)", () => 1, "#1f9d55"], ["O(log n)", (n) => Math.log2(n), "#0ea5e9"], ["O(n)", (n) => n, "#2f6fed"], ["O(n log n)", (n) => n * Math.log2(n), "#7c4dff"], ["O(n²)", (n) => n * n, "#e0a84a"], ["O(2ⁿ)", (n) => Math.pow(2, n), "#d64545"]];
      const X = (n) => L + (n - 1) / (N - 1) * (Rr - L), Y = (v) => B - Math.min(v, YM) / YM * (B - Tp);
      let s = `<line class="axis" x1="${L}" y1="${B}" x2="${Rr}" y2="${B}" style="stroke:var(--muted)"/><line class="axis" x1="${L}" y1="${Tp}" x2="${L}" y2="${B}" style="stroke:var(--muted)"/>`;
      s += T((L + Rr) / 2, B + 30, "input size n →", "lbl") + `<text class="lbl" x="16" y="${(Tp + B) / 2}" text-anchor="middle" transform="rotate(-90 16 ${(Tp + B) / 2})">operations →</text>`;
      fns.forEach(([name, f, col], k) => {
        let d = "";
        for (let n = 1; n <= N; n += 0.25) { const v = f(n); d += (d ? " L" : "M") + X(n).toFixed(1) + "," + Y(v).toFixed(1); if (v > YM) break; }
        s += `<path d="${d}" fill="none" stroke="${col}" stroke-width="2.5"/>`;
        s += `<rect x="${L + 14 + (k % 3) * 120}" y="${Tp + Math.floor(k / 3) * 20}" width="12" height="12" rx="3" fill="${col}"/>` + T(L + 32 + (k % 3) * 120, Tp + 10 + Math.floor(k / 3) * 20, name, "lbl", "start");
      });
      return svg(W, H, s);
    },
    /* Intervals on a timeline: D.intervals([[1,3],[2,6]], {ok:[0]}) */
    intervals(list, o = {}) {
      const mx = Math.max(...list.map((x) => x[1])) + 1, sc = 30, W = 40 + mx * sc, H = 24 + list.length * 30 + 26;
      let s = "";
      for (let t = 0; t <= mx; t++) s += `<line x1="${20 + t * sc}" y1="10" x2="${20 + t * sc}" y2="${H - 22}" style="stroke:var(--border)"/>` + T(20 + t * sc, H - 6, t, "lbl");
      list.forEach(([a, b], i) => {
        const y = 16 + i * 30;
        s += `<rect class="${cls(o, i)}" x="${20 + a * sc}" y="${y}" width="${Math.max((b - a) * sc, 4)}" height="20" rx="5"/>` + T(20 + (a + b) / 2 * sc, y + 15, `[${a},${b}]`, "lbl", "middle", 'font-weight="700"');
      });
      return svg(W, H, s);
    },
  };

  /* ------------------------------------------------------------- Pretty dropdowns
     Every <select> on the site gets a matching custom dropdown (rounded menu, soft fade, ✓ on the chosen option,
     keyboard: ↑ ↓ Enter Esc). The real <select> stays underneath, hidden, as the source of truth, so code that
     reads .value or listens for "change" keeps working. New selects (re-rendered filters) are picked up automatically. */
  const CHEV = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>';
  const TICK = '<svg class="psel-ck" viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>';
  function prettySelect(sel) {
    if (sel._pretty || sel.multiple || sel.size > 1) return;
    sel._pretty = true;
    const wrap = document.createElement("span");
    wrap.className = "psel";
    sel.parentNode.insertBefore(wrap, sel);
    wrap.appendChild(sel);
    sel.classList.add("psel-native"); sel.tabIndex = -1; sel.setAttribute("aria-hidden", "true");
    const btn = document.createElement("button"), menu = document.createElement("div");
    btn.type = "button"; btn.className = "psel-btn"; btn.setAttribute("aria-haspopup", "listbox"); btn.setAttribute("aria-expanded", "false");
    if (sel.getAttribute("aria-label") || sel.title) btn.title = sel.getAttribute("aria-label") || sel.title;
    menu.className = "psel-menu"; menu.setAttribute("role", "listbox"); menu.hidden = true;
    wrap.append(btn, menu);
    let hi = -1;
    const paint = () => { btn.innerHTML = `<span class="psel-val">${esc((sel.options[sel.selectedIndex] || {}).text || "")}</span>${CHEV}`; };
    const build = () => { menu.innerHTML = [...sel.options].map((o, i) => `<button type="button" role="option" class="psel-opt${i === sel.selectedIndex ? " on" : ""}" data-i="${i}" aria-selected="${i === sel.selectedIndex}">${esc(o.text)}${TICK}</button>`).join(""); };
    const focusHi = () => { const o = menu.children[hi]; if (o) { o.focus({ preventScroll: true }); o.scrollIntoView({ block: "nearest" }); } };
    const open = (v) => {
      if (v === !menu.hidden) return;
      if (v) { closeSelects(); build(); hi = Math.max(0, sel.selectedIndex); menu.hidden = false; focusHi(); } else menu.hidden = true;
      wrap.classList.toggle("open", v); btn.setAttribute("aria-expanded", v);
    };
    const choose = (i) => {
      open(false);
      if (i !== sel.selectedIndex) { sel.selectedIndex = i; sel.dispatchEvent(new Event("change", { bubbles: true })); }
      if (wrap.isConnected) { paint(); btn.focus(); }                 // (the page may have re-rendered its filters)
    };
    btn.onclick = () => open(menu.hidden);
    menu.onclick = (e) => { const o = e.target.closest(".psel-opt"); if (o) choose(+o.dataset.i); };
    wrap.addEventListener("keydown", (e) => {
      if (menu.hidden) { if (e.target === btn && ["ArrowDown", "ArrowUp"].includes(e.key)) { e.preventDefault(); open(true); } return; }
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); open(false); btn.focus(); }
      else if (e.key === "ArrowDown") { e.preventDefault(); hi = Math.min(sel.options.length - 1, hi + 1); focusHi(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); hi = Math.max(0, hi - 1); focusHi(); }
      else if (e.key === "Tab") open(false);
    });
    wrap._close = () => open(false);
    sel.addEventListener("change", paint);
    paint();
  }
  const closeSelects = () => document.querySelectorAll(".psel.open").forEach((w) => w._close());
  document.addEventListener("click", (e) => { if (!(e.target.closest && e.target.closest(".psel"))) closeSelects(); });
  const prettyIn = (node) => { if (node.tagName === "SELECT") prettySelect(node); else if (node.querySelectorAll) node.querySelectorAll("select").forEach(prettySelect); };
  new MutationObserver((ms) => ms.forEach((m) => m.addedNodes.forEach((n) => n.nodeType === 1 && prettyIn(n)))).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener("DOMContentLoaded", () => prettyIn(document.body));

  /* ------------------------------------------------------------- Top bar
     Built in one go (icons, colours, due badge, streak, a reserved spot for the ✨ menu), and as early as
     possible, so nothing pops in or shifts after the page appears. Each link has its own soft colour (--c). */
  const NAV_SVG = {
    hop: '<path d="M3.5 11 12 4l8.5 7"/><path d="M5.5 9.5V20h13V9.5"/><path d="M10 20v-5h4v5"/>',
    book: '<path d="M2.5 5.5h6.5a3 3 0 0 1 3 3V20a2.2 2.2 0 0 0-2.2-2.2H2.5z"/><path d="M21.5 5.5H15a3 3 0 0 0-3 3V20a2.2 2.2 0 0 1 2.2-2.2h7.3z"/>',
    puzzle: '<path d="M5 8h3.2a2.3 2.3 0 1 1 4.6 0H16v3.2a2.3 2.3 0 1 1 0 4.6V19h-3.2a2.3 2.3 0 1 0-4.6 0H5v-3.2a2.3 2.3 0 1 0 0-4.6z"/>',
    code: '<rect x="3" y="4.5" width="18" height="15" rx="2.5"/><path d="m7.5 10 3 2.2-3 2.2"/><path class="cur" d="M13 15h3.5"/>',
    repeat: '<path d="m17 3 3 3-3 3"/><path d="M4 11.5V10a4 4 0 0 1 4-4h12"/><path d="m7 21-3-3 3-3"/><path d="M20 12.5V14a4 4 0 0 1-4 4H4"/>',
    snake: '<path d="M3 17.5c2.5 0 2.5-3.5 5-3.5s2.5 3.5 5 3.5 2.5-3.5 5-3.5c1.5 0 2-1.6 2-3.2V8.5a2.5 2.5 0 0 0-5 0"/><path d="M17.8 8.4h.01"/>',
    braces: '<path d="M8.5 4C6.5 4 6.5 5 6.5 7v2.2c0 1.6-.9 2.8-2.3 2.8 1.4 0 2.3 1.2 2.3 2.8V17c0 2 0 3 2 3"/><path d="M15.5 4c2 0 2 1 2 3v2.2c0 1.6.9 2.8 2.3 2.8-1.4 0-2.3 1.2-2.3 2.8V17c0 2 0 3-2 3"/>',
    chat: '<path d="M4 5h16v11H10l-5 4v-4H4z"/><path d="M8.5 10.5h.01M12 10.5h.01M15.5 10.5h.01"/>',
    chart: '<path d="M3 20.5h18"/><path class="b" d="M6.5 20.5V12"/><path class="b" d="M12 20.5V5.5"/><path class="b" d="M17.5 20.5v-6"/>',
    graph: '<circle cx="6" cy="6.5" r="2.4"/><circle cx="18" cy="8" r="2.4"/><circle cx="10.5" cy="18" r="2.4"/><path d="m8.4 6.8 7.2.9M7 8.8l2.5 6.8M16.4 10l-4.4 6.1"/>',
    sigma: '<path d="M17.5 6.5V4h-11l6 8-6 8h11v-2.5"/>',
    net: '<circle cx="5" cy="6.5" r="2"/><circle cx="5" cy="17.5" r="2"/><circle cx="12" cy="12" r="2.2"/><circle cx="19" cy="12" r="2"/><path d="m6.8 7.5 3.4 3M6.8 16.5l3.4-3M14.2 12H17"/>',
    layers: '<path d="m12 3.5 8.5 4.5-8.5 4.5L3.5 8z"/><path d="m3.5 12 8.5 4.5 8.5-4.5"/><path d="m3.5 16 8.5 4.5 8.5-4.5"/>',
    bot: '<rect x="4" y="8" width="16" height="11.5" rx="3.2"/><path d="M12 4.5V8"/><circle cx="12" cy="3.6" r="1.1"/><path class="cur" d="M9 13v1.6M15 13v1.6"/>',
  };
  // label → [icon, hover motion]
  const NAV_ICONS = { "Dashboard": ["hop", "hop"], "Foundations": ["sigma", "tilt"], "ML": ["chart", "grow"], "Core ML": ["chart", "grow"], "Classical ML": ["sigma", "tilt"], "Deep Learning": ["net", "wiggle"],
    "LLMs & Agents": ["chat", "sway"], "Practice": ["code", "blink"], "Revision": ["repeat", "spin"],
    "Math for ML": ["sigma", "tilt"], "Python for ML": ["snake", "sway"], "Statistics": ["chart", "grow"], "Transformers": ["layers", "hop"], "LLMs": ["chat", "sway"], "Agentic AI": ["bot", "blink"],
  };
  // soft per-link colours: blue, mint, lavender, peach, pink, teal, gold
  // [colour for the pill/icons, readable shade for text & buttons in light mode, readable shade in dark mode]
  const NAV_COLORS = [["#4f7cff", "#4268d6", "#6d93ff"], ["#1fae7a", "#177f59", "#2fc28d"], ["#8b6cf0", "#785dce", "#9d84f3"],
    ["#ec8a4f", "#a35f37", "#ef9a65"], ["#e2609a", "#b34c7a", "#e874a8"], ["#22a3b6", "#197a88", "#33b5c8"], ["#d3a226", "#8a6a10", "#e0b445"]];
  const navIcon = (label) => { const ic = NAV_ICONS[label.replace("↗", "").trim()]; return ic ? `<svg class="ni ni-${ic[1]}" viewBox="0 0 24 24" aria-hidden="true">${NAV_SVG[ic[0]]}</svg>` : ""; };
  const dueCount = () => Object.values(Store.load().items).filter((it) => Store.isDue(it)).length;
  const badgeHTML = (n) => (n ? `<span class="nav-badge pulse">${n > 99 ? "99+" : n}</span>` : "");
  const THEME_SVG = (() => {
    const rays = ["M12 2.5v2", "M17.3 6.7l1.4-1.4", "M19.5 12h2", "M17.3 17.3l1.4 1.4", "M12 19.5v2", "M5.3 18.7l1.4-1.4", "M2.5 12h2", "M5.3 5.3l1.4 1.4"];
    return `<svg class="ni" viewBox="0 0 24 24" aria-hidden="true"><defs><mask id="d-tm-mask" maskUnits="userSpaceOnUse" x="-6" y="-6" width="36" height="36"><rect x="-6" y="-6" width="36" height="36" fill="#fff"/><circle class="tm-bite" cx="16.5" cy="8" r="5.6" fill="#000"/></mask></defs><circle class="tm-body" cx="12" cy="12" r="4.4" mask="url(#d-tm-mask)"/><g class="tm-rays">${rays.map((d, i) => `<path d="${d}" style="--i:${i}"/>`).join("")}</g></svg>`;
  })();
  // the thin ring around the logo fills with your progress (what it counts is set in site.js)
  function paintRing() {
    const R = window.SITE && SITE.ring, w = document.querySelector(".topbar .logo-wrap");
    if (!R || !w) return;
    const done = Object.entries(Store.load().items).filter(([k, v]) => v.status && R.prefixes.some((p) => k.startsWith(p))).length;
    const fg = w.querySelector(".fg");
    fg.setAttribute("stroke-dasharray", Math.min(100, (done / R.total) * 100).toFixed(1) + " 100");
    fg.style.opacity = done ? 1 : 0;                                   // no stray dot at 0%
    w.closest(".brand").title = `${done} / ${R.total} ${R.label} · ${dueCount()} due for review`;
  }
  // a header link with a small menu (Python ▾): opens on hover or focus, or on tap where there is no hover
  function navMenus(h) {
    h.querySelectorAll(".nav a.has-menu").forEach((a) => {
      const menu = h.querySelector(`.nav-menu[data-menu="${a.dataset.menu}"]`);
      if (!menu) return;
      let t = 0;
      const open = (on) => {
        clearTimeout(t);
        if (on === !menu.hidden) return;
        a.setAttribute("aria-expanded", on);
        if (on) {
          const r = a.getBoundingClientRect();
          menu.hidden = false;
          menu.style.left = Math.max(8, Math.min(r.left, innerWidth - menu.offsetWidth - 8)) + "px";
          menu.style.top = r.bottom + 6 + "px";
          requestAnimationFrame(() => menu.classList.add("show"));
        } else { menu.classList.remove("show"); t = setTimeout(() => (menu.hidden = true), 200); }
      };
      const later = () => { clearTimeout(t); t = setTimeout(() => open(false), 220); };
      const hover = matchMedia("(hover: hover)");
      a.addEventListener("mouseenter", () => hover.matches && open(true));
      a.addEventListener("mouseleave", later);
      menu.addEventListener("mouseenter", () => clearTimeout(t));
      menu.addEventListener("mouseleave", later);
      a.addEventListener("click", (e) => { if (!hover.matches) { e.preventDefault(); open(menu.hidden); } });
      a.addEventListener("keydown", (e) => { if (e.key === "ArrowDown") { e.preventDefault(); open(true); menu.querySelector("a").focus(); } });
      menu.addEventListener("focusout", (e) => { if (!menu.contains(e.relatedTarget) && e.relatedTarget !== a) open(false); });
      document.addEventListener("click", (e) => { if (!menu.hidden && !menu.contains(e.target) && !a.contains(e.target)) open(false); });
      document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) { open(false); a.focus(); } });
      window.addEventListener("scroll", () => !menu.hidden && open(false), { passive: true });
    });
  }
  // full-screen button. Browsers leave full screen whenever a new page loads, so the button puts the site in a
  // full-window frame and makes the outer page full screen: links then only change the frame and full screen stays on.
  // (iPhone Safari has no full screen for web pages, so the button is hidden there.)
  function fullScreenBtn() {
    const btn = $("#focusBtn"), R = document.documentElement;
    const fsEl = () => document.fullscreenElement || document.webkitFullscreenElement;
    const inFrame = window.parent !== window;
    if (inFrame) {                                                   // inside the full-screen frame: report the page, button exits
      btn.innerHTML = icon("shrink"); btn.title = "Exit full screen";
      btn.onclick = () => parent.postMessage({ fsExit: true }, "*");
      parent.postMessage({ fsPage: location.href, title: document.title }, "*");
      return;
    }
    if (!(R.requestFullscreen || R.webkitRequestFullscreen)) { btn.style.display = "none"; return; }
    btn.innerHTML = icon("expand"); btn.title = "Full screen";
    let frame = null, page = location.href;
    const exit = () => {
      if (fsEl()) try { const p = (document.exitFullscreen || document.webkitExitFullscreen).call(document); if (p && p.catch) p.catch(() => {}); } catch (e) {}
      if (!frame) return;
      frame.remove(); frame = null; document.body.style.overflow = "";
      if (page !== location.href) location.href = page;               // land on the page you browsed to
    };
    btn.onclick = () => {
      try {
        const p = (R.requestFullscreen || R.webkitRequestFullscreen).call(R);
        if (p && p.catch) p.catch(() => { if (frame) exit(); });
      } catch (e) { return; }
      page = location.href;
      frame = document.createElement("iframe");
      frame.src = location.href; frame.title = document.title;
      frame.style.cssText = "position:fixed;inset:0;width:100%;height:100%;border:0;z-index:2147483647;background:var(--bg)";
      document.body.appendChild(frame); document.body.style.overflow = "hidden";
    };
    addEventListener("message", (e) => {
      if (!frame || e.source !== frame.contentWindow || !e.data) return;
      if (e.data.fsExit) exit();
      if (e.data.fsPage) { page = e.data.fsPage; if (e.data.title) document.title = e.data.title; }
    });
    const onChange = () => { if (!fsEl() && frame) exit(); };      // Esc / swipe out of full screen
    document.addEventListener("fullscreenchange", onChange); document.addEventListener("webkitfullscreenchange", onChange);
  }
  function topbar(active) {
    const S = window.SITE || { brand: "Study", links: [] };
    const h = document.getElementById("topbar");
    if (!h || h.dataset.active === active) return;                   // already built (it's built early, see the end of this file)
    h.dataset.active = active;
    h.className = "topbar";
    const streak = Store.streak(), due = dueCount(), hr = new Date().getHours();
    // greeting + logo colour follow the time of day
    const greet = hr < 5 || hr >= 22 ? icon("moon") + " Late-night grind" : hr < 12 ? icon("sun") + " Good morning" : hr < 17 ? icon("cloud-sun") + " Good afternoon" : icon("sunset") + " Good evening";
    const mood = hr < 5 || hr >= 22 ? "night" : hr < 12 ? "morning" : hr < 17 ? "day" : "evening";
    const season = (() => { const d = new Date(), m = d.getMonth(), day = d.getDate();
      return m === 9 ? "pumpkin" : m === 11 ? "snowflake" : m === 1 && day === 14 ? "heart" : m === 0 && day === 1 ? "party" : ""; })();
    h.innerHTML = `
      <button class="icon-btn menu-btn" id="menuBtn" aria-label="Menu">${icon("menu")}</button>
      <a class="brand" href="${S.home || "index.html"}"><span class="logo-wrap" data-mood="${mood}"><span class="logo">${S.logo || "DS"}</span><svg class="logo-ring" viewBox="0 0 36 36" aria-hidden="true"><rect class="tr" x="1.5" y="1.5" width="33" height="33" rx="11"/><rect class="fg" x="1.5" y="1.5" width="33" height="33" rx="11" pathLength="100" stroke-dasharray="0 100"/></svg>${season ? `<span class="season season-${season}" aria-hidden="true">${icon(season)}</span>` : ""}</span><span class="brand-name">${S.brand}</span></a>
      <span class="hello">${greet}</span>
      <nav class="nav">${S.links.map(([href, label, sub], i) => {
        const on = href === active || (sub && sub.some(([x]) => x === active)), C = NAV_COLORS[i % NAV_COLORS.length];
        return `<a href="${href}" class="${[on ? "active" : "", sub ? "has-menu" : ""].filter(Boolean).join(" ")}" title="${label}"${sub ? ` aria-haspopup="true" aria-expanded="false" data-menu="${i}"` : ""} style="--c:${C[0]}" data-ink="${C[1]}" data-ink-dark="${C[2]}">${navIcon(label)}<span>${label}</span>${sub ? `<svg class="nav-chev" viewBox="0 0 24 24" aria-hidden="true"><path d="m7 10 5 5 5-5"/></svg>` : ""}${href === "revision.html" ? badgeHTML(due) : ""}</a>`;
      }).join("")}</nav>
      ${S.links.map(([, , sub], i) => sub ? `<div class="nav-menu" role="menu" data-menu="${i}" style="--c:${NAV_COLORS[i % NAV_COLORS.length][0]}" hidden>${sub.map(([href, label, note]) => `<a role="menuitem" href="${href}" class="${href === active ? "active" : ""}">${navIcon(label)}<span><b>${label}</b>${note ? `<small>${note}</small>` : ""}</span></a>`).join("")}</div>` : "").join("")}
      ${streak > 0 ? `<span class="d-streak" title="${streak}-day study streak"><span class="f">${icon("flame")}</span><span class="sn">${streak}</span></span>` : ""}
      <span class="fx-slot"></span>
      <button class="icon-btn" id="focusBtn" title="Full screen" aria-label="Full screen">${icon("expand")}</button>
      <button class="icon-btn" id="themeBtn" title="Toggle dark mode" aria-label="Toggle theme">${THEME_SVG}</button>`;
    fullScreenBtn();
    $("#themeBtn").onclick = () => {
      const cur = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = cur;
      try { localStorage.setItem("theme", cur); } catch (e) {}
    };
    const mb = $("#menuBtn");
    if (!document.getElementById("sidebar")) mb.style.display = "none";
    mb.onclick = () => document.body.classList.toggle("sidebar-open");
    document.addEventListener("click", (e) => {
      if (document.body.classList.contains("sidebar-open") && !e.target.closest("#sidebar") && !e.target.closest("#menuBtn")) document.body.classList.remove("sidebar-open");
    });
    navMenus(h);
    paintRing();
    // this page's own colour (from its header link) tints text selection, the cursor halo and the glow
    const act = h.querySelector(".nav a.active");
    if (act) {
      const R = document.documentElement;
      R.style.setProperty("--page-c", act.style.getPropertyValue("--c"));
      R.style.setProperty("--page-ink", act.dataset.ink); R.style.setProperty("--page-ink-dark", act.dataset.inkDark);
      R.classList.add("page-tinted");
    }
    // keep the due badge and the logo ring up to date
    document.addEventListener("store:change", () => {
      paintRing();
      const link = h.querySelector('.nav a[href="revision.html"]');
      if (!link) return;
      const n = dueCount(), b = link.querySelector(".nav-badge");
      if (!n) { if (b) b.remove(); return; }
      if (b) b.textContent = n > 99 ? "99+" : n; else link.insertAdjacentHTML("beforeend", badgeHTML(n));
    });
  }

  /* ------------------------------------------------------------- Status buttons (shared UI) */
  function actionBar(key, opts = {}) {
    const it = Store.item(key);
    const doneLabel = opts.doneLabel || "Mark as learned";
    const due = it.next ? relDay(it.next) : "";
    return `<div class="doc-actions" data-key="${key}">
      <button class="btn ${it.status === "done" ? "on-green" : ""}" data-act="done">${swapLabel([icon("circle") + " " + doneLabel, icon("check") + " " + (opts.doneOn || "Learned")], it.status === "done" ? 1 : 0)}</button>
      <button class="btn ${it.status === "revise" ? "on-red" : ""}" data-act="revise">${swapLabel([icon("repeat") + " Mark for revision", icon("repeat") + " Needs revision"], it.status === "revise" ? 1 : 0)}</button>
      <button class="btn ${it.star ? "on-amber" : ""}" data-act="star">${swapLabel([icon("star") + " Bookmark", icon("star-fill") + " Bookmarked"], it.star ? 1 : 0)}</button>
      <span class="small muted due">${it.status && due ? "Next review: " + due : ""}</span>
    </div>`;
  }
  document.addEventListener("click", (e) => {
    const b = e.target.closest(".doc-actions [data-act]");
    if (!b) return;
    const bar = b.closest(".doc-actions"), key = bar.dataset.key, it = Store.item(key), act = b.dataset.act;
    if (act === "done") { Store.setStatus(key, it.status === "done" ? null : "done"); toast(it.status === "done" ? "Unmarked" : "Nice! Added to your revision schedule"); }
    if (act === "revise") { Store.setStatus(key, it.status === "revise" ? (it.doneAt ? "done" : null) : "revise"); }
    if (act === "star") { Store.toggleStar(key); }
    // update the buttons in place (no re-render), so they keep their size and position
    const fresh = document.createElement("div");
    fresh.innerHTML = actionBar(key, JSON.parse(bar.dataset.opts || "{}"));
    bar.querySelectorAll("[data-act]").forEach((btn) => {
      const nb = fresh.querySelector(`[data-act="${btn.dataset.act}"]`);
      btn.className = nb.className;
      setSwap(btn, nb.querySelector(".swap").dataset.s);
    });
    bar.querySelector(".due").textContent = fresh.querySelector(".due").textContent;
  });
  function relDay(t) {
    const days = Math.round((new Date(today(t)) - new Date(today())) / DAY);
    if (days <= 0) return "due now";
    if (days === 1) return "tomorrow";
    return `in ${days} days`;
  }
  function notesBox(key) {
    const it = Store.item(key);
    return `<details class="box" ${it.note ? "open" : ""}><summary>${icon("note")} My notes</summary><div class="body"><textarea data-note="${key}" placeholder="Write your own summary, mistakes you made, key insight...">${esc(it.note || "")}</textarea><div class="small muted">Saved automatically in this browser.</div></div></details>`;
  }
  let noteT;
  document.addEventListener("input", (e) => {
    const t = e.target.closest("[data-note]");
    if (!t) return;
    clearTimeout(noteT);
    noteT = setTimeout(() => Store.setNote(t.dataset.note, t.value), 400);
  });

  /* ------------------------------------------------------------- DocPage: sidebar + one section at a time */
  function textOf(blocks) {
    return (blocks || []).map((b) => (typeof b === "string" ? b : b.map((x) => (typeof x === "string" ? x : Array.isArray(x) ? x.flat(2).join(" ") : x && typeof x === "object" ? Object.values(x).join(" ") : "")).join(" "))).join(" ").toLowerCase();
  }
  function DocPage(cfg) {
    const secs = cfg.sections, kind = cfg.kind;
    const sb = $("#sidebar"), main = $("#content");
    secs.forEach((s) => (s._text = (s.title + " " + (s.group || "") + " " + (s.tags || []).join(" ") + " " + textOf(s.blocks)).toLowerCase()));
    let filter = "";
    const FOLD_KEY = lsKey("sbfold:" + kind);
    const getFolds = () => { try { return JSON.parse(localStorage.getItem(FOLD_KEY) || "[]"); } catch (e) { return []; } };
    const setFolds = (f) => { try { localStorage.setItem(FOLD_KEY, JSON.stringify(f)); } catch (e) {} };
    // click a group title to fold / unfold it (animated in CSS, no re-render)
    sb.addEventListener("click", (e) => {
      const t = e.target.closest(".group-title");
      if (!t) return;
      const grp = t.closest(".sb-group"), g = grp.dataset.g, folded = !grp.classList.contains("folded");
      grp.classList.toggle("folded", folded); t.setAttribute("aria-expanded", !folded);
      if (!filter) setFolds(folded ? [...new Set([...getFolds(), g])] : getFolds().filter((x) => x !== g));
    });

    const byId = Object.fromEntries(secs.map((s) => [s.id, s]));
    window.__doc = { secs, kind, byId, title: cfg.title };                // used by assets/fx/fx-sidebar.js for hover previews
    let lastPct = null;
    function sidebar() {
      const total = secs.length, done = secs.filter((s) => Store.item(kind + ":" + s.id).status).length;
      const hashId = (location.hash || "").slice(1), curId = byId[hashId] ? hashId : secs[0].id, ci = secs.findIndex((s) => s.id === curId);
      let h = `<input type="search" class="search" id="docSearch" placeholder="Search ${cfg.noun || "topics"}…  ( / )" value="${esc(filter)}">
        <div class="sb-progress"><div class="small muted row-between"><span>Progress</span><span>${done}/${total}</span></div><div class="bar"><span style="width:${(done / total) * 100}%"></span></div></div>
        <p class="small muted sb-nomatch empty-state" hidden>No matches.</p>`;
      // the current item and its neighbours read a little stronger; learned items soften and get a small ✓
      const itemHTML = (s) => {
        const it = Store.item(kind + ":" + s.id), i = secs.indexOf(s);
        const cls = ["item", s.id === curId ? "active" : "", Math.abs(i - ci) === 1 ? "near" : "", it.status === "done" ? "is-done" : ""].filter(Boolean).join(" ");
        return `<a class="${cls}" href="#${s.id}" data-id="${s.id}"><span class="dot ${it.status === "done" ? "done" : ""}" style="${it.status === "revise" ? "background:var(--red);border-color:var(--red)" : ""}"></span><span class="tt">${esc(s.title)}</span>${it.status === "done" ? `<span class="ck" title="Learned">${icon("check")}</span>` : ""}${it.star ? `<span class="star">${icon("star-fill")}</span>` : ""}</a>`;
      };
      // groups of consecutive sections: foldable (remembered), with a thin progress line under the title
      const groups = [];
      secs.forEach((s) => { const g = s.group || ""; if (!groups.length || groups[groups.length - 1].g !== g) groups.push({ g, items: [] }); groups[groups.length - 1].items.push(s); });
      const folds = getFolds();
      groups.forEach((G) => {
        if (!G.g) { h += G.items.map(itemHTML).join(""); return; }
        const gd = G.items.filter((s) => Store.item(kind + ":" + s.id).status === "done").length;
        const folded = folds.includes(G.g);
        h += `<div class="sb-group${folded ? " folded" : ""}" data-g="${esc(G.g)}"><button type="button" class="group-title" aria-expanded="${!folded}"><span class="gt-arrow">▸</span><span class="gt-name">${esc(G.g)}</span><span class="gt-count">${gd}/${G.items.length}</span></button><div class="gt-line"><span style="width: ${((gd / G.items.length) * 100).toFixed(1)}%"></span></div><div class="sb-items"><div>${G.items.map(itemHTML).join("")}</div></div></div>`;
      });
      // a tiny mascot walks towards the 🏁 as you progress
      const pct = total ? Math.round((done / total) * 100) : 0, mpos = (x) => (x >= 100 ? 100 : x * 0.86);
      h += `<div class="sb-mascot${pct >= 100 ? " won" : ""}" title="${done} / ${total} done"><div class="sbm-track"><span class="sbm-m" style="left:${mpos(lastPct ?? pct)}%">${icon(cfg.mascot || (window.SITE && SITE.mascot) || "turtle")}</span><span class="sbm-flag">${icon("flag")}</span></div><div class="sbm-txt">${pct >= 100 ? icon("party") + " You made it!" : `${pct}% of the way`}</div></div>`;
      sb.innerHTML = h;
      if (lastPct !== null && lastPct !== pct) {
        const m = sb.querySelector(".sbm-m");
        requestAnimationFrame(() => { m.classList.add("walking"); m.style.left = mpos(pct) + "%"; setTimeout(() => m.classList.remove("walking"), 1300); });
      }
      lastPct = pct;
      applyFilter();
      const inp = $("#docSearch");
      inp.oninput = () => { filter = inp.value.trim().toLowerCase(); applyFilter(); };   // no re-render: items fold away softly
      const act = sb.querySelector("a.item.active");
      if (act && !filter) act.scrollIntoView({ block: "nearest" });
    }
    // search: non-matching items fold away, matching words get a soft highlight
    function applyFilter() {
      const f = filter;
      sb.classList.toggle("filtering", !!f);
      let any = false;
      sb.querySelectorAll("a.item").forEach((a) => {
        const s = byId[a.dataset.id], m = !f || s._text.includes(f);
        a.classList.toggle("nomatch", !m); if (m) any = true;
        const i = f ? s.title.toLowerCase().indexOf(f) : -1;
        a.querySelector(".tt").innerHTML = i >= 0 ? esc(s.title.slice(0, i)) + "<mark>" + esc(s.title.slice(i, i + f.length)) + "</mark>" + esc(s.title.slice(i + f.length)) : esc(s.title);
      });
      sb.querySelectorAll(".sb-group").forEach((g) => g.classList.toggle("nomatch-g", !!f && !g.querySelector("a.item:not(.nomatch)")));
      const nm = sb.querySelector(".sb-nomatch"); if (nm) nm.hidden = !f || any;
    }
    function show() {
      let id = (location.hash || "").slice(1);
      if (!secs.find((s) => s.id === id)) id = secs[0].id;
      const i = secs.findIndex((s) => s.id === id), s = secs[i];
      const key = kind + ":" + s.id, prev = secs[i - 1], next = secs[i + 1];
      if (s.group && getFolds().includes(s.group)) setFolds(getFolds().filter((x) => x !== s.group));
      document.title = s.title + " · " + (cfg.title || "");
      main.innerHTML = `
        <div class="crumbs">${esc(cfg.title || "")}${s.group ? " / " + esc(s.group) : ""}</div>
        <h1>${esc(s.title)}</h1>
        ${s.sub ? `<p class="lead">${fmt(s.sub)}</p>` : ""}
        ${window.BRIEFS && BRIEFS[key] ? `<div class="callout tip brief"><span class="label">${icon("bulb")} In short, in plain English</span>${String(BRIEFS[key]).split(/\n\s*\n/).map((x) => `<p>${fmt(x.trim())}</p>`).join("")}</div>` : ""}
        ${s.tags ? `<div class="btn-row my-2">${s.tags.map((t) => `<span class="badge blue">${esc(t)}</span>`).join("")}</div>` : ""}
        ${actionBar(key, cfg.actionOpts)}
        <div class="doc-body">${renderBlocks(s.blocks)}</div>
        <hr>${notesBox(key)}
        <div class="btn-row page-nav">
          ${prev ? `<a class="btn" href="#${prev.id}">← ${esc(prev.title)}</a>` : "<span></span>"}
          ${next ? `<a class="btn primary" href="#${next.id}">${esc(next.title)} →</a>` : ""}
        </div>`;
      const bar = main.querySelector(".doc-actions");
      if (bar) bar.dataset.opts = JSON.stringify(cfg.actionOpts || {});
      if (window.Viz) Viz.mountAll(main);
      if (window.glossify) glossify(main);
      sidebar();
      document.body.classList.remove("sidebar-open");
      window.scrollTo(0, 0);
    }
    window.addEventListener("hashchange", show);
    // only rebuild the sidebar when something it shows changed (status / bookmark), not on every note keystroke
    const sbSig = () => secs.map((s) => { const it = Store.item(kind + ":" + s.id); return (it.status || "-") + (it.star ? "*" : ""); }).join("");
    let lastSig = sbSig();
    document.addEventListener("store:change", () => { const sig = sbSig(); if (sig !== lastSig) { lastSig = sig; sidebar(); } });
    show();
  }

  if (document.getElementById("topbar")) topbar(location.pathname.split("/").pop() || "index.html");

  Object.assign(window, { icon, codeSource: (i) => CODE[i], startStepper, stopStepper, swapLabel, setSwap, $, $$, esc, fmt, tex, hlPy, R, Store, toast, codeBlock, renderBlocks, D, topbar, actionBar, notesBox, DocPage, today, relDay, DAY });
})();
