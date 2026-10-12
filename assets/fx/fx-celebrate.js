/* fx-celebrate.js: rewards for progress.
   • confetti when you solve / learn something (a gentle rain for a finished category, cleared reviews, streak milestones)
   • liquid fills: Learned (green), Needs revision (red), Bookmark (amber) pour in and drain out; NeetCode circles fill
     like a glass; Revision ✓/✗ fill then the row drains away; growing progress bars rise like water */
(function () {
  const FX = window.FX;
  if (!FX) return;
  const { root, ls, motionOK, finePointer, typing, page, content, sparkle, toneOf, confetti, bigConfetti } = FX;
  /* ------------------------------------------------------------------ celebrations: hook into Store */
  const GROUPS = { prac: [window.PRAC, "cat"] };
  SITE.docs.forEach((d) => (GROUPS[d.kind] = [window[d.list], "group"]));
  function groupComplete(key) {
    const [kind, id] = [key.slice(0, key.indexOf(":")), key.slice(key.indexOf(":") + 1)];
    const g = GROUPS[kind];
    if (!g || !g[0] || !g[0].length) return null;
    const me = g[0].find((s) => s.id === id);
    if (!me || !me[g[1]]) return null;
    const all = g[0].filter((s) => s[g[1]] === me[g[1]]);
    return all.length > 1 && all.every((s) => Store.item(kind + ":" + s.id).status) ? me[g[1]] : null;
  }
  const later = (msg, ms = 1900, ic = "party") => setTimeout(() => toast(msg, 3200, ic), ms);   // after the normal "Solved ✓" toast
  function funnyTime() {
    const d = new Date(), t = String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
    return { "03:14": ["π o'clock solve", "pie"], "15:14": ["π o'clock solve", "pie"], "13:37": ["Elite solve. 1337", "cool"], "12:34": ["1, 2, 3, 4… solved!", "hash"], "11:11": ["11:11, make a wish", "sparkles"], "00:00": ["Midnight solve. The stack will still be there tomorrow", "moon"] }[t];
  }
  function streakCheck() {
    const s = Store.streak();
    if ([3, 7, 14, 30, 50, 100, 365].includes(s) && !ls.get("delight:streak:" + Store.key + ":" + s)) {
      ls.set("delight:streak:" + Store.key + ":" + s, "1");
      setTimeout(bigConfetti, 300); later(`${s}-day streak! Keep it going`, 3800, "flame");
    }
  }
  const origSet = Store.setStatus.bind(Store);
  Store.setStatus = function (k, status) {
    const was = this.item(k).status;
    const r = origSet(k, status);
    if (status === "done" && !was) {
      confetti(FX.pointer.x, FX.pointer.y);
      const g = groupComplete(k);
      if (g) { setTimeout(bigConfetti, 250); later(`You finished all of “${g}”!`, 1900, "rainbow"); }
      else { const f = funnyTime(); if (f) later(f[0], 1900, f[1]); }
      streakCheck();
    }
    return r;
  };
  const origReview = Store.review.bind(Store);
  Store.review = function (k, ok) {
    const before = FX.dueCount();
    const r = origReview(k, ok);
    if (ok) confetti(FX.pointer.x, FX.pointer.y, 12);
    if (before > 0 && FX.dueCount() === 0) { setTimeout(bigConfetti, 250); later("All reviews cleared for today!"); }
    streakCheck();
    return r;
  };

  /* Pour: fills a button with liquid from the left (or drains it out) — wave edge, bubbles, shine when full. */
  const LIQ_COL = { green: ["var(--green-soft)", "var(--green)"], red: ["var(--red-soft)", "var(--red)"], amber: ["var(--amber-soft)", "var(--amber)"] };
  function pour(b, color, drain) {
    if (!motionOK() || !b) return;
    const id = (b._pour = (b._pour || 0) + 1);                         // a newer fill on this button takes over
    b.querySelectorAll(".d-liquid, .d-shine").forEach((x) => x.remove());
    const [soft, strong] = LIQ_COL[color] || LIQ_COL.green;
    const liq = document.createElement("span");
    liq.className = "d-liquid"; liq.style.setProperty("--liq", soft); liq.style.setProperty("--liq-edge", strong);
    if (!drain) liq.innerHTML = "<i></i><i></i><i></i>";
    Object.assign(b.style, { position: "relative", overflow: "hidden", isolation: "isolate", transition: "none" });
    if (!drain) b.style.background = "var(--panel)";               // start empty; the liquid brings the colour in
    b.prepend(liq);
    let cleaned = false;
    const done = () => {
      if (cleaned) return; cleaned = true;
      liq.remove();
      if (b._pour !== id) return;                                      // an older fill: don't touch the newer one's styles
      b.style.background = ""; b.querySelectorAll(".d-shine").forEach((x) => x.remove()); void b.offsetWidth;
      Object.assign(b.style, { position: "", overflow: "", isolation: "", transition: "" });
    };
    setTimeout(done, 2600);                                            // safety net if the animation never finishes
    liq.animate(drain ? [{ transform: "translateX(0)" }, { transform: "translateX(-106%)" }] : [{ transform: "translateX(-106%)" }, { transform: "translateX(0)" }],
      { duration: drain ? 850 : 1150, easing: drain ? "cubic-bezier(.45,0,.6,.9)" : "cubic-bezier(.25,.65,.3,1)" }).onfinish = () => {
      if (drain || b._pour !== id) return done();
      const sh = document.createElement("span"); sh.className = "d-shine"; b.prepend(sh);
      sh.animate([{ transform: "translateX(-100%)" }, { transform: "translateX(100%)" }], { duration: 700, easing: "cubic-bezier(.4,0,.2,1)" }).onfinish = () => { sh.remove(); done(); };
    };
  }

  /* ------------------------------------------------------------------ liquid fills everywhere
     • Learned / Solved (green), Needs revision (red), Bookmark (amber + a little sparkle): pour in; undo drains out
     • NeetCode list circles fill up from the bottom like a glass, then the ✓ pops in
     • sidebar dots fill up from the bottom when you mark a section learned
     • Revision tab ✓ / ✗: the button fills, then the row gently drains away
     • any progress bar that grows rises like a water level with a wavy front edge */
  const tone = (cls) => { const m = String(cls).match(/on-(green|red|amber)/); return m ? m[1] : null; };
  document.addEventListener("click", (e) => {                       // capture phase: remember the "before" state
    const t = e.target.closest && e.target.closest(".doc-actions [data-act], [data-a=tog][data-k]");
    if (!t) return;
    if (t.matches("[data-act]")) { const bar = t.closest(".doc-actions"); bar._before = [...bar.querySelectorAll("[data-act]")].map((x) => x.className); }
    else window._togBefore = { k: t.dataset.k, was: Store.item("prac:" + t.dataset.k).status || null };
  }, true);
  document.addEventListener("click", (e) => {
    if (!motionOK()) return;
    const btn = e.target.closest && e.target.closest(".doc-actions [data-act]");
    if (btn) {
      // only the button you clicked animates; the others just change colour quietly
      const bar = btn.closest(".doc-actions"), before = bar._before || [];
      const i = [...bar.querySelectorAll("[data-act]")].indexOf(btn);
      const was = tone(before[i]), now = tone(btn.className);
      if (now && now !== was) pour(btn, now); else if (was && !now) pour(btn, was, true);
      if (btn.dataset.act === "star" && btn.classList.contains("on-amber")) setTimeout(() => sparkle(btn, ["#ffd166", "#ffb347", "#fff3c4"]), 500);
      return;
    }
    const tog = e.target.closest && e.target.closest("[data-a=tog][data-k]");
    if (tog && window._togBefore) {
      const { k, was } = window._togBefore; window._togBefore = null;
      setTimeout(() => {                                               // the list has re-rendered by now
        const nb = document.querySelector(`[data-a=tog][data-k="${CSS.escape(k)}"]`), now = Store.item("prac:" + k).status || null;
        if (!nb) return;
        if (now === "done" && was !== "done") glass(nb, "var(--green)");
        else if (was === "done" && !now) glass(nb, "var(--green)", true);
      }, 0);
    }
  });
  function glass(b, color, drain) {
    const c = document.createElement("span");
    c.className = "d-cup"; c.style.setProperty("--cup", color);
    const bg = b.style.background, col = b.style.color;
    Object.assign(b.style, { position: "relative", overflow: "hidden", isolation: "isolate", transition: "color .4s ease" });
    if (!drain) { b.style.background = "var(--panel)"; b.style.color = "transparent"; }
    b.prepend(c);
    let cleaned = false;
    const finish = () => { if (cleaned) return; cleaned = true; c.remove(); b.style.background = bg; b.style.color = col; b.style.overflow = ""; b.style.isolation = ""; setTimeout(() => (b.style.transition = ""), 450); };
    setTimeout(finish, 2200);                                          // safety net
    c.animate(drain ? [{ top: "-35%" }, { top: "115%" }] : [{ top: "115%" }, { top: "-35%" }], { duration: drain ? 700 : 900, easing: "cubic-bezier(.33,.7,.3,1)" }).onfinish = finish;
  }

  // Revision tab: fill the button, let the row drain away, then do the real review
  document.addEventListener("click", (e) => {
    const b = e.target.closest && e.target.closest("[data-rev]");
    if (!b || b._go || !motionOK()) return;
    e.stopPropagation(); e.preventDefault();
    b._go = true;
    pour(b, b.dataset.ok === "1" ? "green" : "red");
    const row = b.closest("tr");
    // the real review runs on a timer, so it never depends on an animation finishing
    setTimeout(() => { if (row) row.animate([{ opacity: 1, transform: "none", filter: "blur(0)" }, { opacity: 0, transform: "translateX(14px)", filter: "blur(2px)" }],
      { duration: 450, easing: "cubic-bezier(.4,0,.2,1)", fill: "forwards" }); }, 650);
    setTimeout(() => b.click(), 1100);
  }, true);
  // progress bars: when a bar grows, its level rises like water with a wavy front edge
  const origSave = Store.save.bind(Store);
  Store.save = function () {
    // measure where each bar *visibly* is right now (it may still be gliding from a previous change)
    const before = [...document.querySelectorAll(".bar > span, .gt-line > span")].map((x) => { const p = x.parentElement.getBoundingClientRect().width; return p ? (x.getBoundingClientRect().width / p) * 100 : 0; });
    const r = origSave();
    setTimeout(() => {
      if (!motionOK()) return;
      const now = document.querySelectorAll(".bar > span, .gt-line > span");
      if (now.length !== before.length) return;
      now.forEach((x, i) => {
        const w = parseFloat(x.style.width) || 0;
        if (Math.abs(w - before[i]) < 0.05) return;
        x.animate([{ width: before[i].toFixed(2) + "%" }, { width: w + "%" }], { duration: 900, easing: "cubic-bezier(.22,1,.36,1)" });
      });
    }, 0);
    return r;
  };
})();
