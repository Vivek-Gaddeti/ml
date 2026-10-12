/* fx-cursor.js: cursor touches.
   • the optional glow that follows the mouse (✨ menu), calm and colour-aware,
     resting when still and lighting up the border of the card under it • two tiny ✦ on click when the glow is on */
(function () {
  const FX = window.FX;
  if (!FX) return;
  const { root, ls, motionOK, finePointer, typing, page, content, sparkle, toneOf, confetti, bigConfetti } = FX;
  /* ------------------------------------------------------------------ glow cursor follower (optional, off by default)
     A soft blurred light that trails a little behind the mouse. It grows over things you can click,
     dims while you press, and fades away over code and text boxes so it never gets in the way of reading.
     The loop only runs while the glow is still catching up, so it costs nothing when the mouse is still. */
  const glow = document.createElement("div");
  glow.className = "d-glow"; glow.setAttribute("aria-hidden", "true");
  document.body.appendChild(glow);
  let glowOn = ls.get("glow") === "on", gx = null, gy = 0, tx = 0, ty = 0, gs = 1, gRaf = 0, pressing = false, hoverScale = 1, restT = 0, spotEl = null;
  const glowActive = () => glowOn && motionOK() && finePointer.matches;
  const setTone = (t) => { const v = t || "var(--page-c, var(--accent))"; if (root.style.getPropertyValue("--gc") !== v) root.style.setProperty("--gc", v); };
  setTone(null);
  function glowStep() {
    gx += (tx - gx) * 0.09; gy += (ty - gy) * 0.09;                 // slow, soft follow
    const target = pressing ? hoverScale * 0.8 : hoverScale;
    gs += (target - gs) * 0.1;
    glow.style.transform = `translate(${gx.toFixed(1)}px, ${gy.toFixed(1)}px) scale(${gs.toFixed(3)})`;
    gRaf = Math.abs(tx - gx) + Math.abs(ty - gy) + Math.abs(target - gs) > 0.2 ? requestAnimationFrame(glowStep) : 0;
  }
  const kick = () => { if (!gRaf) gRaf = requestAnimationFrame(glowStep); };
  const unspot = () => { if (spotEl) { spotEl.classList.remove("d-spot"); spotEl = null; } };
  // reading areas: everything extra steps aside, leaving just the normal text cursor
  const QUIET = ".code-lines, pre, input, textarea, select, .d-sheet-bg, .d-menu, p, li, td, th, h1, h2, h3, .lead, .callout";
  const CLICKABLE = "a[href], button, summary, label, [data-act], select, [role=button]";

  document.addEventListener("mousemove", (e) => {
    const t = e.target.closest ? e.target : document.body;
    const click = t.closest(CLICKABLE), selecting = (e.buttons & 1) && String(window.getSelection && getSelection());
    setTone(toneOf(t));
    // glow
    if (!glowActive()) { unspot(); return; }
    tx = e.clientX; ty = e.clientY;
    if (gx === null) { gx = tx; gy = ty; }
    const quiet = (!click && t.closest(QUIET)) || selecting;
    hoverScale = click ? 1.35 : 1;
    glow.classList.toggle("on", !quiet);
    glow.classList.remove("rest"); clearTimeout(restT);
    restT = setTimeout(() => glow.classList.add("rest"), 1000);   // rests when the mouse is still
    // spotlight on the card under the cursor
    let card = t.closest(".card, details.box");
    if (card && getComputedStyle(card).overflow !== "visible") card = null;   // never on scrolling boxes
    if (card !== spotEl) { unspot(); if (card) { spotEl = card; card.classList.add("d-spot"); } }
    if (card) { const r = card.getBoundingClientRect(); card.style.setProperty("--mx", (e.clientX - r.left).toFixed(0) + "px"); card.style.setProperty("--my", (e.clientY - r.top).toFixed(0) + "px"); }
    kick();
  }, { passive: true });
  document.addEventListener("mousedown", (e) => {
    pressing = true; kick();
    // glow on: clicking leaves two tiny ✦ that fade quickly
    if (glowActive() && e.button === 0 && !typing(e.target)) sparkle({ getBoundingClientRect: () => ({ left: e.clientX, top: e.clientY, width: 0, height: 0 }) }, ["#ffd166", "#c7b4f5", "#9cc3ff"], 2);
  }, { passive: true });
  document.addEventListener("mouseup", () => { pressing = false; kick(); }, { passive: true });
  const allOff = () => { glow.classList.remove("on"); unspot(); };
  document.documentElement.addEventListener("mouseleave", allOff);
  window.addEventListener("blur", allOff);
  window.addEventListener("scroll", () => { glow.classList.remove("on"); }, { passive: true });   // out of the way while scrolling

  // the ✨ menu changed the glow switch (or animations): pick it up, and hide whatever is no longer allowed
  document.addEventListener("fx:settings", () => { glowOn = ls.get("glow") === "on"; if (!glowActive()) { glow.classList.remove("on"); unspot(); } });
})();
