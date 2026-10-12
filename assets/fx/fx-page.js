/* fx-page.js: page-level effects.
   • soft page change (content dissolves out, blocks rise into focus) and sliding between sections
   • aurora background • smooth open/close of boxes • click bloom, copy bounce, jelly buttons, magnetic "Next →"
   • count-up numbers • back-to-top bubble • "Resume where you left off" • keyboard shortcuts (?) • easter eggs 🥚 */
(function () {
  const FX = window.FX;
  if (!FX) return;
  const { root, ls, motionOK, finePointer, typing, page, content, sparkle, toneOf, confetti, bigConfetti } = FX;
  /* ------------------------------------------------------------------ 1. soft fade on page/section change */
  // dir: 1 = slide in from the right (going forward), -1 = from the left (going back), 0 = soft fade
  function fadeIn(dir = 0) {
    if (!content || !motionOK()) return;
    content.classList.remove("d-in", "d-in-r", "d-in-l"); void content.offsetWidth;
    content.classList.add(dir > 0 ? "d-in-r" : dir < 0 ? "d-in-l" : "d-in");
  }
  // page open: the blocks on screen rise into focus one after another, like bubbles surfacing
  function rise() {
    if (!content || !motionOK()) return;
    root.classList.add("d-loading"); clearTimeout(rise._t); rise._t = setTimeout(() => root.classList.remove("d-loading"), 1400);
    const els = [];
    [...content.children].forEach((c) => { if (c.classList.contains("d-aurora")) return; if (c.classList.contains("grid")) els.push(...c.children); else els.push(c); });
    els.filter((e) => e.getBoundingClientRect().top < innerHeight).slice(0, 14).forEach((e, i) =>
      e.animate([{ opacity: 0, transform: "translateY(12px)", filter: "blur(5px)" }, { opacity: 1, transform: "none", filter: "blur(0)" }],
        { duration: 650, delay: i * 45, easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }));
  }
  // leaving for another page: the content dissolves (fade + soft blur) for 0.2s, then the browser navigates
  document.addEventListener("click", (e) => {
    if (e.defaultPrevented || e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey || !motionOK() || !content) return;
    const a = e.target.closest && e.target.closest("a[href]");
    if (!a || a.target === "_blank" || a.hasAttribute("download")) return;
    const href = a.getAttribute("href");
    if (!href || href.startsWith("#") || /^[a-z]+:/i.test(href)) return;
    const url = new URL(href, location.href);
    if (url.pathname === location.pathname) return;                    // same page: the hash change animates instead
    e.preventDefault();
    content.classList.add("d-leave");
    setTimeout(() => (location.href = url.href), 190);
  });
  window.addEventListener("pageshow", (e) => { if (e.persisted && content) { content.classList.remove("d-leave"); rise(); } });   // back button

  // position of a section/problem in the page's order (sidebar order, or Practice order)
  function orderOf(hash) {
    hash = decodeURIComponent(hash || "");
    if (hash.startsWith("#p/") && window.pracSorted) { const i = pracSorted().findIndex((p) => p.id === hash.slice(3)); return i; }
    const items = [...document.querySelectorAll("#sidebar a.item")].map((a) => a.getAttribute("href"));
    if (!items.length) return -1;
    return hash.length > 1 ? items.indexOf(hash) : 0;
  }
  // aurora: huge, very faint colour blobs drifting behind the whole screen. Always on dashboards;
  // on other pages only with "Aurora everywhere" (✨ menu), and fainter there.
  const aurora = document.createElement("div");
  aurora.className = "d-aurora"; aurora.setAttribute("aria-hidden", "true"); aurora.innerHTML = "<i></i><i></i><i></i>";
  document.body.prepend(aurora);
  const isDash = !!(content && content.querySelector(".stat")) && !/practice\.html$/.test(location.pathname);
  const paintAurora = () => { aurora.hidden = !(isDash || ls.get("aurora") === "on"); aurora.classList.toggle("faint", !isDash); };
  paintAurora();
  rise();
  window.addEventListener("hashchange", (e) => setTimeout(() => {    // runs after the page has re-rendered
    const a = orderOf(new URL(e.oldURL).hash), b = orderOf(new URL(e.newURL).hash);
    if (a >= 0 && b >= 0 && a !== b) fadeIn(Math.sign(b - a)); else rise();

  }, 0));
  /* ------------------------------------------------------------------ 2. smooth open/close for <details class="box"> */
  document.addEventListener("click", (e) => {
    const sum = e.target.closest && e.target.closest("details.box > summary");
    if (!sum || e.target.closest("a") || !motionOK()) return;
    const d = sum.parentElement, body = d.querySelector(":scope > .body");
    if (!body || body._anim) { if (body && body._anim) e.preventDefault(); return; }
    e.preventDefault();
    body.style.overflow = "hidden";
    const done = () => { body.style.overflow = ""; body._anim = null; };
    if (d.open) {
      const h = body.offsetHeight;
      body._anim = body.animate([{ height: h + "px", opacity: 1 }, { height: "0px", paddingBottom: "0px", opacity: 0 }], { duration: Math.min(380, 240 + h / 12), easing: "cubic-bezier(.4,0,.2,1)" });
      body._anim.onfinish = () => { d.open = false; done(); };
    } else {
      d.open = true;
      const h = body.offsetHeight;
      body._anim = body.animate([{ height: "0px", paddingBottom: "0px", opacity: 0 }, { height: h + "px", opacity: 1 }], { duration: Math.min(450, 280 + h / 10), easing: "cubic-bezier(.22,1,.36,1)" });
      body._anim.onfinish = done;
    }
  });

  /* ------------------------------------------------------------------ click bloom + copy bounce */
  document.addEventListener("pointerdown", (e) => {
    if (!motionOK() || e.button !== 0 || typing(e.target) || (e.target.closest && e.target.closest(".code-lines, pre, .d-sheet"))) return;
    const r = document.createElement("div");
    r.className = "d-bloom"; r.style.left = e.clientX + "px"; r.style.top = e.clientY + "px";
    r.style.background = `color-mix(in srgb, ${toneOf(e.target) || "var(--page-c, var(--accent))"} 32%, transparent)`;
    document.body.appendChild(r);
    r.animate([{ transform: "scale(.3)", opacity: .7 }, { transform: "scale(1.6)", opacity: 0 }], { duration: 650, easing: "cubic-bezier(.22,1,.36,1)" }).onfinish = () => r.remove();
  }, { passive: true });
  document.addEventListener("click", (e) => {
    const b = e.target.closest && e.target.closest("[data-copy]");
    if (b && motionOK()) setTimeout(() => { b.classList.remove("d-pop"); void b.offsetWidth; b.classList.add("d-pop"); }, 30);
  });

  /* ------------------------------------------------------------------ magnetic pull
     Only the big "Next →" link at the bottom of a page and the back-to-top bubble, never the small
     buttons in toolbars (so it can't shift a button under your cursor). Max 3px, slow settle. */
  const MAG = "#content > .btn-row > a.btn.primary, .d-top.show";
  let magFrame = 0, mx = 0, my = 0, pulled = [];
  document.addEventListener("mousemove", (e) => {
    mx = e.clientX; my = e.clientY;
    if (!magFrame) magFrame = requestAnimationFrame(magnet);
  }, { passive: true });
  function magnet() {
    magFrame = 0;
    const on = motionOK() && finePointer.matches && !(window.getSelection && String(window.getSelection()));
    const now = [];
    if (on) document.querySelectorAll(MAG).forEach((el) => {
      const r = el.getBoundingClientRect();
      if (!r.width) return;
      const cx = r.left + r.width / 2, cy = r.top + r.height / 2, dx = mx - cx, dy = my - cy;
      const reach = Math.max(r.width, r.height) / 2 + 28;
      if (Math.hypot(dx, dy) > reach) return;
      const k = 0.08, cap = 3;
      el.style.translate = `${Math.max(-cap, Math.min(cap, dx * k)).toFixed(1)}px ${Math.max(-cap, Math.min(cap, dy * k)).toFixed(1)}px`;
      now.push(el);
    });
    pulled.forEach((el) => { if (!now.includes(el)) el.style.translate = ""; });
    pulled = now;
  }
  document.addEventListener("mouseleave", () => { pulled.forEach((el) => (el.style.translate = "")); pulled = []; });

  /* ------------------------------------------------------------------ count-up for the big numbers
     Numbers in .stat boxes (dashboard, Practice list, progress page) roll up from 0 when a page opens. */
  function countUp() {
    const scope = document.getElementById("content");
    if (!motionOK() || !scope || document.visibilityState !== "visible") return;
    const nodes = [];
    scope.querySelectorAll(".stat").forEach((st) => st.childNodes.forEach((n) => { if (n.nodeType === 3 && /\d/.test(n.nodeValue)) nodes.push({ n, tpl: n.nodeValue }); }));
    if (!nodes.length) return;
    const t0 = performance.now(), D = 1100;
    const frame = (now) => {
      const p = Math.min(1, (now - t0) / D), e = 1 - Math.pow(1 - p, 4);       // fast start, soft landing
      nodes.forEach((o) => (o.n.nodeValue = o.tpl.replace(/\d+/g, (d) => Math.round(+d * e))));
      if (p < 1) requestAnimationFrame(frame);
    };
    nodes.forEach((o) => (o.n.nodeValue = o.tpl.replace(/\d+/g, "0")));
    requestAnimationFrame(frame);
  }
  countUp();
  window.addEventListener("hashchange", () => setTimeout(countUp, 0));

  /* ------------------------------------------------------------------ jelly buttons
     On release a button does one tiny squash-and-stretch wobble (uses the separate `scale` property,
     so it never fights the press squish or the magnetic pull). */
  document.addEventListener("pointerup", (e) => {
    if (!motionOK() || e.button !== 0) return;
    const b = e.target.closest && e.target.closest(".btn, .icon-btn");
    if (!b || !b.animate) return;
    b.animate([{ scale: "0.97 0.97" }, { scale: "1.035 0.965", offset: 0.35 }, { scale: "0.988 1.012", offset: 0.65 }, { scale: "1 1" }],
      { duration: 550, easing: "ease-out" });
  }, { passive: true });

  /* ------------------------------------------------------------------ back-to-top bubble */
  const top = document.createElement("button");
  top.className = "d-top"; top.type = "button"; top.title = "Back to top"; top.setAttribute("aria-label", "Back to top"); top.innerHTML = icon("up");
  top.onclick = () => window.scrollTo({ top: 0, behavior: motionOK() ? "smooth" : "auto" });
  document.body.appendChild(top);
  let topT = 0;
  window.addEventListener("scroll", () => { if (!topT) topT = requestAnimationFrame(() => { topT = 0; top.classList.toggle("show", scrollY > 700); }); }, { passive: true });

  /* ------------------------------------------------------------------ resume where you left off */
  const RK = "delight:resume:" + Store.key;
  function remember() {
    if (location.hash.length > 1 && location.hash !== "#") ls.set(RK, JSON.stringify({ href: page + location.hash, title: document.title.split(" · ")[0], t: Date.now() }));
  }
  let chip = null;
  function resumeChip() {
    try { if (location.hash.length > 1 || sessionStorage.getItem("delight:resume-closed")) return; } catch (e) {}
    let r = null; try { r = JSON.parse(ls.get(RK) || "null"); } catch (e) {}
    if (!r || !r.href) return;
    chip = document.createElement("div");
    chip.className = "d-resume";
    chip.innerHTML = `<span aria-hidden="true">↩</span><a href="${r.href.replace(/"/g, "&quot;")}">Resume: <b></b></a><button type="button" title="Hide" aria-label="Hide">×</button>`;
    chip.querySelector("b").textContent = r.title;
    const drop = () => { if (chip) { chip.remove(); chip = null; } document.body.classList.remove("has-resume"); };
    chip.querySelector("button").onclick = () => { drop(); try { sessionStorage.setItem("delight:resume-closed", "1"); } catch (e) {} };
    chip.querySelector("a").onclick = drop;
    chip._drop = drop;
    document.body.appendChild(chip);
    document.body.classList.add("has-resume");                       // leaves room at the bottom of the page
    const tuck = () => chip && chip.classList.add("mini");
    setTimeout(tuck, 3500);
    window.addEventListener("scroll", () => { if (scrollY > 120) tuck(); }, { passive: true });
  }
  remember(); resumeChip();
  window.addEventListener("hashchange", () => setTimeout(() => { remember(); if (chip && location.hash.length > 1) chip._drop(); }, 0));

  /* ------------------------------------------------------------------ keyboard shortcuts */
  const KEYS = [["←  →", "Previous / next section or problem"], ["S", "Mark as solved / learned"], ["R", "Mark for revision"], ["B", "Bookmark"], ["Ctrl K", "Find anything"], ["/", "Search this page's list"], ["P", "Python playground"], ["T", "Toggle dark mode"], ["G", "Back to top"], ["?", "Show this list"], ["Esc", "Close"]];
  let sheet = null;
  function toggleSheet() {
    if (sheet) { sheet.remove(); sheet = null; return; }
    sheet = document.createElement("div");
    sheet.className = "d-sheet-bg";
    sheet.innerHTML = `<div class="d-sheet" role="dialog" aria-label="Keyboard shortcuts"><h3>${icon("keyboard")} Keyboard shortcuts</h3><table>${KEYS.map(([k, d]) => `<tr><td>${k.split(/\s+/).map((x) => `<kbd>${x}</kbd>`).join(" ")}</td><td>${d}</td></tr>`).join("")}</table><p class="small muted" style="margin:.8rem 0 0">Shortcuts are ignored while you're typing in a box.</p></div>`;
    sheet.onclick = (e) => { if (e.target === sheet) toggleSheet(); };
    document.body.appendChild(sheet);
  }
  function navLink(dir) {
    const rows = document.querySelectorAll("#content .btn-row");
    const row = rows[rows.length - 1];
    if (!row) return null;
    return [...row.querySelectorAll("a.btn")].find((a) => dir < 0 ? /^\s*←/.test(a.textContent) && !a.classList.contains("sm") : /→\s*$/.test(a.textContent));
  }
  const press = (sel) => { const b = document.querySelector(sel); if (b) { b.click(); return true; } return false; };
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || typing(e.target) || e.defaultPrevented) return;
    const k = e.key;
    if (k === "Escape") { if (sheet) toggleSheet(); return; }
    if (k === "?") { e.preventDefault(); toggleSheet(); return; }
    if (sheet) return;
    if (k === "ArrowLeft" || k === "ArrowRight") {
      if (e.target.closest && e.target.closest("input[type=range], .viz, .stepper, .st-panel")) return;
      const a = navLink(k === "ArrowLeft" ? -1 : 1);
      if (a) { e.preventDefault(); a.click(); }
      return;
    }
    const low = k.toLowerCase();
    if (e.shiftKey && low !== "?") return;
    if (low === "s") { const b = document.querySelector(".doc-actions [data-act=done]"); if (b) { const r = b.getBoundingClientRect(); FX.pointer.x = r.left + r.width / 2; FX.pointer.y = r.top + r.height / 2; b.click(); } }
    else if (low === "r") press(".doc-actions [data-act=revise]");
    else if (low === "b") press(".doc-actions [data-act=star]");
    else if (low === "t") press("#themeBtn");
    else if (low === "g") top.click();
    else if (k === "/") {
      const s = document.querySelector("#docSearch, #fq, #content input[type=search]");
      if (s) { e.preventDefault(); if (s.id === "docSearch" && document.body.clientWidth < 900) document.body.classList.add("sidebar-open"); s.focus(); s.select(); }
    }
  });

  /* ------------------------------------------------------------------ easter eggs 🥚 */
  // Konami code: ↑ ↑ ↓ ↓ ← → ← → B A
  const KONAMI = ["ArrowUp", "ArrowUp", "ArrowDown", "ArrowDown", "ArrowLeft", "ArrowRight", "ArrowLeft", "ArrowRight", "b", "a"];
  let kpos = 0;
  document.addEventListener("keydown", (e) => {
    if (typing(e.target)) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    kpos = k === KONAMI[kpos] ? kpos + 1 : (k === KONAMI[0] ? 1 : 0);
    if (kpos === KONAMI.length) {
      kpos = 0;
      const logo = document.querySelector(".brand .logo");
      if (logo && motionOK()) { logo.classList.remove("d-roll"); void logo.offsetWidth; logo.classList.add("d-roll"); }
      bigConfetti(); toast("You found a secret! +30 lives for your next training run", 3500, "gamepad");
    }
  }, true);

  // On the home page, click the logo 5 times for an ML joke
  const JOKES = [
    "My model has 100% training accuracy. It also memorised your phone number",
    "Gradient descent: taking small steps downhill and hoping it's not a saddle point",
    "I asked the LLM for a joke about overfitting. It told me the same one from its training data",
    "Why did the neural network break up? It found someone with better weights",
    "Machine learning is just statistics wearing a hoodie",
    "Dropout: randomly ignoring half your neurons, like a Monday morning",
    "Attention is all you need. Also a GPU. And a lot of coffee",
    "The agent said it would book the flight. It booked three, to be safe",
    "My loss went to NaN, so I lowered the learning rate and my expectations",
    "Why do data scientists love nature? So many random forests",
  ];
  const logo = document.querySelector(".brand .logo"), brand = document.querySelector(".brand");
  let clicks = 0, clickT = 0;
  if (logo && brand) logo.addEventListener("click", (e) => {
    const home = new URL(brand.getAttribute("href"), location.href);
    if (home.pathname !== location.pathname || location.hash.length > 1) return;   // elsewhere it still goes home
    e.preventDefault();
    if (motionOK()) { logo.classList.remove("d-wiggle"); void logo.offsetWidth; logo.classList.add("d-wiggle"); }
    clearTimeout(clickT); clickT = setTimeout(() => (clicks = 0), 1500);
    if (++clicks >= 5) { clicks = 0; toast(JOKES[Math.floor(Math.random() * JOKES.length)], 4500, "laugh"); confetti(FX.pointer.x, FX.pointer.y, 10); }
  });

  // a ✨ switch changed: animations back on → replay the page's entrance; off → let go of magnetic pulls; aurora on/off
  document.addEventListener("fx:settings", () => {
    if (motionOK()) rise(); else { pulled.forEach((el) => (el.style.translate = "")); pulled = []; }
    paintAurora();
  });
  FX.rise = rise;
})();
