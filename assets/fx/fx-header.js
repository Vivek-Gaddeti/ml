/* fx-header.js: the top bar.
   • ✨ settings menu • soft day ↔ night switch with the sun/moon moment • the coloured pill that glides between links
   • peek cards under links • logo breathing when idle • 🔥 streak roll-up • December snow • header shrinks + frosts on scroll */
(function () {
  const FX = window.FX;
  if (!FX) return;
  const { root, ls, motionOK, finePointer, typing, page, content, sparkle, toneOf, confetti, bigConfetti } = FX;
  /* ------------------------------------------------------------------ top-bar ✨ menu: Animations, Glow cursor, Aurora everywhere */
  const themeBtn = document.getElementById("themeBtn");
  if (themeBtn) {
    const wrap = document.createElement("div");
    wrap.className = "d-menu-wrap";
    wrap.innerHTML = `<button class="icon-btn" id="animBtn" aria-haspopup="true" aria-expanded="false"></button>
      <div class="d-menu" role="menu" hidden>
        <label class="d-row"><span><b>${icon("sparkles")} Animations</b><small>Fades, smooth boxes, confetti</small></span><input type="checkbox" data-opt="anim"><i class="d-sw"></i></label>
        <label class="d-row"><span><b>${icon("cursor")} Glow cursor</b><small>A soft light that follows your mouse</small></span><input type="checkbox" data-opt="glow"><i class="d-sw"></i></label>
        <label class="d-row"><span><b>${icon("aurora")} Aurora everywhere</b><small>Soft colours behind every page</small></span><input type="checkbox" data-opt="aurora"><i class="d-sw"></i></label>
        <div class="d-hint small muted">Press <kbd>?</kbd> for keyboard shortcuts</div>
      </div>`;
    (document.querySelector(".fx-slot") || themeBtn.parentNode).appendChild(wrap);
    const b = wrap.querySelector("#animBtn"), menu = wrap.querySelector(".d-menu");
    const animBox = wrap.querySelector('[data-opt="anim"]'), glowBox = wrap.querySelector('[data-opt="glow"]');
    const auroraBox = wrap.querySelector('[data-opt="aurora"]');
    const paint = () => {
      const on = root.dataset.anim === "on", glowOn = ls.get("glow") === "on";
      b.innerHTML = icon(on ? "sparkles" : "sparkle");
      b.title = "Effects: animations " + (on ? "on" : "off") + ", glow cursor " + (glowOn ? "on" : "off");
      animBox.checked = on; glowBox.checked = glowOn; glowBox.disabled = !on;
      glowBox.closest(".d-row").classList.toggle("dim", !on);
      auroraBox.checked = ls.get("aurora") === "on";
    };
    const openMenu = (open) => { menu.hidden = !open; b.setAttribute("aria-expanded", open); if (open) requestAnimationFrame(() => wrap.classList.add("seen")); else wrap.classList.remove("seen"); };
    wrap.addEventListener("change", (e) => {
      const sw = e.target.closest(".d-row") && e.target.closest(".d-row").querySelector(".d-sw");
      if (!sw) return;
      sw.classList.remove("kick"); void sw.offsetWidth; sw.classList.add("kick");
      clearTimeout(sw._t); sw._t = setTimeout(() => sw.classList.remove("kick"), 600);
    });
    b.onclick = (e) => { e.stopPropagation(); openMenu(menu.hidden); };
    document.addEventListener("click", (e) => { if (!menu.hidden && !wrap.contains(e.target)) openMenu(false); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) openMenu(false); });
    animBox.onchange = () => { root.dataset.anim = animBox.checked ? "on" : "off"; ls.set("anim", root.dataset.anim); FX.settingsChanged(); };
    glowBox.onchange = () => { ls.set("glow", glowBox.checked ? "on" : "off"); FX.settingsChanged(); };
    auroraBox.onchange = () => { ls.set("aurora", auroraBox.checked ? "on" : "off"); FX.settingsChanged(); };
    document.addEventListener("fx:settings", paint);
    paint();
  }

  /* ------------------------------------------------------------------ soft day ↔ night switch
     The page softly fades into the other theme (the old look blurs a touch while the new one fades in on top).
     The cute part is on the button: going dark, the sun becomes a moon and little stars twinkle out;
     going light, the rays pop back one by one with a warm glow. Also applies to the T shortcut. */
  if (themeBtn && themeBtn.onclick) {
    const flip = themeBtn.onclick;
    themeBtn.onclick = (e) => {
      if (!motionOK()) return flip.call(themeBtn, e);
      const toDark = root.dataset.theme !== "dark";
      if (document.startViewTransition) {
        root.classList.add("d-soft");
        const vt = document.startViewTransition(() => flip.call(themeBtn, e));
        vt.ready.catch(() => {}); vt.updateCallbackDone.catch(() => {});
        vt.finished.catch(() => {}).then(() => root.classList.remove("d-soft"));
      } else {
        root.classList.add("d-theme-fade"); flip.call(themeBtn, e);
        setTimeout(() => root.classList.remove("d-theme-fade"), 700);
      }
      if (toDark) setTimeout(() => sparkle(themeBtn, ["#ffe08a", "#c7b4f5", "#ffffff"], 3), 250);
      else setTimeout(() => themeBtn.animate([{ boxShadow: "0 0 0 0 rgba(255,190,80,.55)" }, { boxShadow: "0 0 0 12px rgba(255,190,80,0)" }], { duration: 1000, easing: "ease-out" }), 300);
    };
  }

  /* ------------------------------------------------------------------ header: the soft coloured pill
     It glides calmly to the link you point at and gently takes on that link's own colour. */
  const nav = document.querySelector(".topbar .nav");
  if (nav) {
    const pill = document.createElement("span");
    pill.className = "nav-pill";
    nav.prepend(pill); nav.classList.add("has-pill");
    let lastA = null, shineT = 0;
    const place = (a) => {
      if (!a) { pill.style.opacity = 0; return; }
      if (a !== lastA && lastA && motionOK()) {                       // a soft shine once it has arrived
        clearTimeout(shineT);
        shineT = setTimeout(() => { pill.classList.remove("shine"); void pill.offsetWidth; pill.classList.add("shine"); }, 420);
      }
      lastA = a;
      pill.style.setProperty("--pc", a.style.getPropertyValue("--c") || "var(--accent)");
      pill.style.width = a.offsetWidth + "px"; pill.style.height = a.offsetHeight + "px";
      pill.style.transform = `translate(${a.offsetLeft}px, ${a.offsetTop}px)`; pill.style.opacity = 1;
    };
    var home = () => place(nav.querySelector("a.active"));
    const homeNow = () => { pill.classList.remove("ready"); home(); void pill.offsetWidth; pill.classList.add("ready"); };   // re-measure without gliding
    homeNow();
    nav.addEventListener("mouseover", (e) => { const a = e.target.closest("a"); if (a) place(a); });
    nav.addEventListener("mouseleave", home);
    nav.addEventListener("focusin", (e) => { const a = e.target.closest("a"); if (a) place(a); });
    nav.addEventListener("focusout", home);
    window.addEventListener("resize", homeNow);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(homeNow);
    document.addEventListener("store:change", () => requestAnimationFrame(homeNow));   // the due badge can change a link's width
  }
  /* ------------------------------------------------------------------ header: peek cards, logo breathing, streak roll-up, snow */
  const dueN = FX.dueCount;
  // "what's next" for each list; remembered, so pages that don't load a list can still show it
  const PK = "peek:" + Store.key;
  let peekCache = {}; try { peekCache = JSON.parse(ls.get(PK) || "{}"); } catch (e) {}
  function refreshPeekCache() {
    const lists = {}; SITE.docs.forEach((d) => (lists[d.kind] = window[d.list]));
    Object.entries(lists).forEach(([kind, arr]) => {
      if (!arr || !arr.length) return;
      const nxt = arr.find((x) => !Store.item(kind + ":" + x.id).status), done = arr.filter((x) => Store.item(kind + ":" + x.id).status).length;
      peekCache[kind] = { t: nxt ? nxt.title : "All done", n: `${done} / ${arr.length} learned` };
    });
    if (window.PRAC && PRAC.length && window.pracSorted && PRAC[0].opt) {    // the full bank (not the light index on Revision)
      const all = pracSorted(), nxt = all.find((x) => !Store.item("prac:" + x.id).status), done = all.filter((x) => Store.item("prac:" + x.id).status).length;
      peekCache.prac = { t: nxt ? `${nxt.t} · ${nxt.d}` : "All solved", n: `${done} / ${all.length} solved` };
    }
    ls.set(PK, JSON.stringify(peekCache));
  }
  refreshPeekCache();
  document.addEventListener("item:status", refreshPeekCache);
  function peekFor(href) {
    // until a page with that list has been opened once, fall back to a count from your saved progress
    const C = peekCache, due = dueN(), st = Store.streak();
    const doneOf = (k) => Object.entries(Store.load().items).filter(([key, v]) => v.status && key.startsWith(k + ":")).length;
    const nx = (k, label) => C[k] ? [label, C[k].t, C[k].n] : [label, doneOf(k) ? "Keep going" : "Let's begin", `${doneOf(k)} done so far`];
    const cards = {
      "index.html": ["Today", st ? `${st}-day streak` : "Start a streak today", `${due} due for review`],
      "practice.html": nx("prac", "Next problem"),
      "revision.html": ["Revision", due ? `${due} due today` : "All caught up", "Spaced repetition keeps it fresh"],
    };
    SITE.docs.forEach((d) => (cards[d.href] = nx(d.kind, "Next up")));
    return cards[href] || null;
  }
  if (nav && finePointer.matches) {
    const card = document.createElement("div");
    card.className = "nav-peek"; card.setAttribute("aria-hidden", "true");
    document.body.appendChild(card);
    let pT = 0, cur = null;
    const hide = () => { clearTimeout(pT); cur = null; card.classList.remove("show"); };
    nav.addEventListener("mouseover", (e) => {
      const a = e.target.closest("a");
      if (!a || a === cur) return;
      cur = a; clearTimeout(pT); card.classList.remove("show");
      pT = setTimeout(() => {
        const info = !a.classList.contains("has-menu") && peekFor(a.getAttribute("href"));   // Python ▾ opens its own menu instead
        if (!info) return;
        card.innerHTML = `<div class="pk-k">${esc(info[0])}</div><div class="pk-t">${esc(info[1])}</div>${info[2] ? `<div class="pk-s">${esc(info[2])}</div>` : ""}`;
        card.style.setProperty("--c", a.style.getPropertyValue("--c"));
        const r = a.getBoundingClientRect();
        card.style.left = Math.max(8, Math.min(r.left, innerWidth - card.offsetWidth - 8)) + "px";
        card.style.top = r.bottom + 8 + "px";
        card.classList.add("show");
      }, 420);
    });
    nav.addEventListener("mouseleave", hide);
    nav.addEventListener("click", hide);
    window.addEventListener("scroll", hide, { passive: true });
  }
  // logo breathes softly after 8 seconds of no activity
  const logoWrap = document.querySelector(".topbar .logo-wrap");
  if (logoWrap) {
    let idleT = 0, last = 0;
    const wake = () => {
      const now = Date.now(); if (now - last < 250) return; last = now;
      logoWrap.classList.remove("breathe"); clearTimeout(idleT);
      idleT = setTimeout(() => { if (motionOK()) logoWrap.classList.add("breathe"); }, 8000);
    };
    ["mousemove", "keydown", "scroll", "pointerdown"].forEach((ev) => window.addEventListener(ev, wake, { passive: true }));
    wake();
  }
  // solving something: the 🔥 number rolls up if the streak grew, and a tiny ✦ twinkles on Revision / My Progress
  document.addEventListener("item:status", (e) => {
    const { was, status } = e.detail;
    if (!status || was) return;
    const revLink = nav && [...nav.querySelectorAll("a")].find((a) => a.getAttribute("href") === "revision.html");
    if (revLink) setTimeout(() => sparkle(revLink, ["#f7a8c8", "#c7b4f5"], 2), 900);
    const n = Store.streak();
    let el = document.querySelector(".topbar .d-streak");
    if (!el && n > 0) {
      el = document.createElement("span"); el.className = "d-streak pop-in"; el.innerHTML = `<span class="f">${icon("flame")}</span><span class="sn">${n}</span>`;
      el.title = `${n}-day study streak`; (document.querySelector(".fx-slot") || themeBtn).before(el); return;
    }
    const sn = el && el.querySelector(".sn");
    if (!sn || +sn.textContent >= n) return;
    el.title = `${n}-day study streak`;
    if (!motionOK()) { sn.textContent = n; return; }
    sn.animate([{ transform: "none", opacity: 1 }, { transform: "translateY(-70%)", opacity: 0 }], { duration: 260, easing: "ease-in" }).onfinish = () => {
      sn.textContent = n;
      sn.animate([{ transform: "translateY(70%)", opacity: 0 }, { transform: "none", opacity: 1 }], { duration: 420, easing: "cubic-bezier(.22,1,.36,1)" });
    };
    el.classList.remove("flick"); void el.offsetWidth; el.classList.add("flick");
  });
  // December: a few slow snowflakes in the header
  if (new Date().getMonth() === 11 && motionOK() && document.getElementById("topbar")) {
    const sn = document.createElement("span"); sn.className = "h-snow"; sn.setAttribute("aria-hidden", "true");
    sn.innerHTML = Array.from({ length: 8 }, (_, i) => `<i style="left:${(i * 12.5 + Math.random() * 8).toFixed(1)}%;animation-duration:${(7 + Math.random() * 6).toFixed(1)}s;animation-delay:-${(Math.random() * 10).toFixed(1)}s">${icon("snowflake")}</i>`).join("");
    document.getElementById("topbar").prepend(sn);
  }

  // header shrinks a little + gets frosted when you scroll
  const bar = document.getElementById("topbar");
  let small = false, sT = 0;
  function onScroll() {
    sT = 0;
    const y = scrollY;
    if (!small && y > 70) small = true; else if (small && y < 12) small = false;   // gap avoids flicker
    root.style.setProperty("--top-h", small ? "48px" : "56px");
    if (bar) bar.classList.toggle("scrolled", y > 4);
  }
  window.addEventListener("scroll", () => { if (!sT) sT = requestAnimationFrame(onScroll); }, { passive: true });
  window.addEventListener("hashchange", () => setTimeout(onScroll, 50));
  onScroll();
})();
