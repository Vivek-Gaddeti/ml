/* fx-sidebar.js: the sidebar on Topics / Patterns / Python pages.
   • one bar that glides to the current section • hover preview card (summary + reading time)
   • a section's dot fills up when you mark it learned */
(function () {
  const FX = window.FX;
  if (!FX) return;
  const { root, ls, motionOK, finePointer, typing, page, content, sparkle, toneOf, confetti, bigConfetti } = FX;
  /* ------------------------------------------------------------------ sidebar: a gliding bar
     One small bar marks the current section. When you move to another section it glides there softly
     (and the new item's tint fades in), instead of one bar vanishing and another appearing. */
  (function () {
    const sb = document.getElementById("sidebar");
    if (!sb) return;
    let bar = null, lastHref = null, lastPos = null;
    const pos = (a) => {
      const r = a.getBoundingClientRect(), p = sb.getBoundingClientRect();
      return { y: r.top - p.top + sb.scrollTop + r.height * 0.22, h: r.height * 0.56 };
    };
    const put = (p, animate) => {
      bar.classList.toggle("glide", !!animate && motionOK());
      bar.style.transform = `translateY(${p.y.toFixed(1)}px)`; bar.style.height = p.h.toFixed(1) + "px"; bar.style.opacity = 1;
    };
    function sync() {
      if (!bar || !bar.isConnected) { bar = document.createElement("span"); bar.className = "sb-ind"; sb.prepend(bar); sb.classList.add("has-ind"); if (lastPos) put(lastPos, false); }
      const a = sb.querySelector("a.item.active");
      if (!a || !a.offsetParent) { bar.style.opacity = 0; return; }
      const href = a.getAttribute("href"), p = pos(a), moved = lastHref !== null && href !== lastHref;
      if (moved && motionOK()) { a.classList.add("sb-new"); setTimeout(() => a.classList.remove("sb-new"), 700); }
      void bar.offsetWidth;                                             // start from the old spot, then glide
      put(p, moved);
      lastHref = href; lastPos = p;
    }
    // the sidebar re-renders on every section change; this runs right after, before the screen updates
    new MutationObserver(() => { if (!bar || !bar.isConnected) sync(); }).observe(sb, { childList: true });
    // folding/unfolding a group moves the items: follow them while it animates
    // folding a group or searching moves the items: follow them for a moment
    sb.addEventListener("input", () => sb.dispatchEvent(new CustomEvent("sb:moved")));
    sb.addEventListener("click", (e) => { if (e.target.closest(".group-title")) sb.dispatchEvent(new CustomEvent("sb:moved")); });
    sb.addEventListener("sb:moved", () => {
      const t0 = performance.now();
      const follow = () => { const a = sb.querySelector("a.item.active"); if (a && bar) { if (a.offsetParent && a.getBoundingClientRect().height) { const p = pos(a); put(p, false); lastPos = p; bar.style.opacity = 1; } else bar.style.opacity = 0; } if (performance.now() - t0 < 520) requestAnimationFrame(follow); };
      requestAnimationFrame(follow);
    });
    window.addEventListener("resize", () => { lastHref = null; sync(); });
    sync();

    // hover an item for a moment: a small card with its one-line summary and reading time
    if (!finePointer.matches) return;
    const card = document.createElement("div");
    card.className = "sb-peek"; card.setAttribute("aria-hidden", "true");
    document.body.appendChild(card);
    const plain = (t) => String(t || "").replace(/\*\*|__|`|\[\[|\]\]/g, "").replace(/\s+/g, " ").trim();
    const summary = (s, key) => {
      let t = s.sub || (window.BRIEFS && BRIEFS[key]) || "";
      if (!t) { const b = (s.blocks || []).find((x) => Array.isArray(x) && x[0] === "p"); t = b ? b[1] : ""; }
      t = plain(t).split(/(?<=[.!?])\s/)[0];
      return t.length > 150 ? t.slice(0, 147) + "…" : t;
    };
    let pT = 0, cur = null;
    const hide = () => { clearTimeout(pT); cur = null; card.classList.remove("show"); };
    sb.addEventListener("mouseover", (e) => {
      const a = e.target.closest("a.item");
      if (!a || a === cur || !window.__doc || innerWidth < 900) return;
      cur = a; clearTimeout(pT); card.classList.remove("show");
      pT = setTimeout(() => {
        const D = window.__doc, s = D.byId[a.dataset.id];
        if (!s) return;
        const key = D.kind + ":" + s.id, it = Store.item(key), mins = Math.max(1, Math.round((s._text || "").split(/\s+/).length / 200));
        const state = it.status === "done" ? icon("check") + " learned" : it.status === "revise" ? icon("repeat") + " needs revision" : "not started";
        card.innerHTML = `<div class="pk-t">${esc(s.title)}</div>${summary(s, key) ? `<div class="pk-d">${esc(summary(s, key))}</div>` : ""}<div class="pk-m"><span>${icon("timer")} ${mins} min read</span><span>${state}</span></div>`;
        const r = a.getBoundingClientRect(), sr = sb.getBoundingClientRect();
        card.style.left = sr.right + 10 + "px";
        card.style.top = Math.max(64, Math.min(r.top - 6, innerHeight - card.offsetHeight - 10)) + "px";
        card.classList.add("show");
      }, 500);
    });
    sb.addEventListener("mouseleave", hide);
    sb.addEventListener("click", hide);
    sb.addEventListener("scroll", hide, { passive: true });
  })();

  // sidebar dot of the section you just marked learned
  document.addEventListener("item:status", (e) => {
    const { key, was, status } = e.detail;
    if (!motionOK() || status !== "done" || was === "done") return;
    const id = key.slice(key.indexOf(":") + 1);
    const dot = document.querySelector(`#sidebar a.item[href="#${CSS.escape(id)}"] .dot`);
    if (!dot) return;
    const k = { backgroundColor: "transparent", backgroundImage: "linear-gradient(var(--green), var(--green))", backgroundRepeat: "no-repeat", backgroundPosition: "bottom" };
    dot.animate([{ ...k, backgroundSize: "100% 0%" }, { ...k, backgroundSize: "100% 100%" }], { duration: 900, easing: "cubic-bezier(.3,.6,.3,1)" })
      .onfinish = () => dot.animate([{ scale: "1" }, { scale: "1.5" }, { scale: "1" }], { duration: 420, easing: "cubic-bezier(.34,1.5,.5,1)" });
  });
})();
