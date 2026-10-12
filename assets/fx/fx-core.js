/* fx-core.js: shared helpers for the optional effects. Load order: fx-core → fx-page → fx-header → fx-sidebar →
   fx-cursor → fx-celebrate (each one is independent; remove a <script> tag to drop that group of effects).
   The modules talk to each other only through window.FX and the "fx:settings" event (fired when a ✨ switch changes). */
(function () {
  if (!window.Store) return;
  const root = document.documentElement;
  const ls = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  // ✨ setting: "on" / "off" chosen with the top-bar button; if never chosen, follow the system "reduce motion" setting
  function applyAnim() { const p = ls.get("anim"); root.dataset.anim = p === "on" || (p !== "off" && !reduced.matches) ? "on" : "off"; }
  applyAnim();
  if (reduced.addEventListener) reduced.addEventListener("change", () => { applyAnim(); document.dispatchEvent(new CustomEvent("fx:settings")); });
  const motionOK = () => root.dataset.anim === "on";
  const finePointer = matchMedia("(hover: hover) and (pointer: fine)");
  const typing = (el) => el && el.closest && el.closest("input, textarea, select, [contenteditable]");
  const page = location.pathname.split("/").pop() || "index.html";
  const content = document.getElementById("content");
  // where the pointer last went down (confetti starts there)
  const pointer = { x: innerWidth / 2, y: innerHeight / 2 };
  document.addEventListener("pointerdown", (e) => { pointer.x = e.clientX; pointer.y = e.clientY; }, { passive: true, capture: true });
  // tiny ✦ stars that twinkle out from an element
  function sparkle(el, colors = ["#ffd166", "#c7b4f5", "#9cc3ff"], n = 3) {
    if (!motionOK() || !el) return;
    const r = el.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    for (let i = 0; i < n; i++) {
      const st = document.createElement("span");
      st.className = "d-star"; st.textContent = "✦"; st.style.color = colors[i % colors.length];
      document.body.appendChild(st);
      const ang = -Math.PI / 2 + (i - (n - 1) / 2) * 0.9 + (Math.random() - 0.5) * 0.4, dist = 16 + Math.random() * 10;
      const x = cx + Math.cos(ang) * dist - 5, y = cy + Math.sin(ang) * dist - 6;
      st.animate([{ transform: `translate(${cx - 5}px, ${cy - 6}px) scale(0) rotate(0deg)`, opacity: 0 },
        { transform: `translate(${x}px, ${y}px) scale(1.1) rotate(90deg)`, opacity: 1, offset: 0.45 },
        { transform: `translate(${x}px, ${y - 4}px) scale(0) rotate(180deg)`, opacity: 0 }],
        { duration: 1000, delay: i * 120, easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }).onfinish = () => st.remove();
    }
  }
  // the colour of a thing you point at: green for Solved / ✓, red for revision / ✗, amber for bookmarks, a header link's own colour
  function toneOf(el) {
    if (!el || !el.closest) return null;
    if (el.closest(".on-green, [data-act=done], [data-ok='1']")) return "var(--green)";
    if (el.closest(".on-red, [data-act=revise], [data-ok='0']")) return "var(--red)";
    if (el.closest(".on-amber, [data-act=star], .star")) return "var(--amber)";
    const nl = el.closest(".nav a");
    return nl ? nl.style.getPropertyValue("--c") || null : null;
  }

  /* ------------------------------------------------------------------ confetti */
  const COLORS = ["#f4a7a7", "#f7c58b", "#f3dd8a", "#9fd8b4", "#a2c4f7", "#c7b4f5", "#f2b2d3"];
  // One paper piece: launched with velocity (vx, vy) px/s, slowed by air drag, pulled down by gravity,
  // swaying side to side while it tumbles. The path is sampled into keyframes so it moves naturally.
  function piece(x, y, vx, vy, dur, size) {
    const p = document.createElement("div");
    p.className = "d-confetti";
    p.style.width = size + "px"; p.style.height = size * (Math.random() < .5 ? .5 : 1) + "px";
    p.style.background = COLORS[Math.floor(Math.random() * COLORS.length)];
    if (Math.random() < .25) p.style.borderRadius = "50%";
    document.body.appendChild(p);
    const drag = 2.2, g = 600, sway = 6 + Math.random() * 10, w = 3 + Math.random() * 3, ph = Math.random() * 6.3;
    const spin = (Math.random() - .5) * 500, flip = 200 + Math.random() * 300, N = 18, T = dur / 1000, frames = [];
    for (let i = 0; i <= N; i++) {
      const t = (i / N) * T, e = (1 - Math.exp(-drag * t)) / drag;
      const px = x + vx * e + Math.sin(ph + t * w) * sway * Math.min(1, t * 2);
      const py = y + (vy + g / drag) * e - (g / drag) * t;
      frames.push({ transform: `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px) rotate(${(spin * t).toFixed(0)}deg) rotateX(${(flip * t).toFixed(0)}deg)`, opacity: i / N < .7 ? 1 : 1 - (i / N - .7) / .3 });
    }
    p.animate(frames, { duration: dur, easing: "linear" }).onfinish = () => p.remove();
  }
  // small burst from where you clicked
  function confetti(x, y, n = 14) {
    if (!motionOK()) return;
    for (let i = 0; i < n; i++) {
      const a = -Math.PI / 2 + (Math.random() - .5) * 1.7, v = 260 + Math.random() * 260;
      piece(x, y, Math.cos(a) * v, Math.sin(a) * v, 1700 + Math.random() * 900, 5 + Math.random() * 4);
    }
  }
  // gentle rain from the top of the screen (finished a category, cleared reviews, streaks…)
  function bigConfetti() {
    if (!motionOK()) return;
    for (let i = 0; i < 70; i++) setTimeout(() => {
      piece(Math.random() * innerWidth, -20, (Math.random() - .5) * 120, 40 + Math.random() * 80, 3200 + Math.random() * 1600, 6 + Math.random() * 5);
    }, Math.random() * 900);
  }
  const dueCount = () => Object.values(Store.load().items).filter((it) => Store.isDue(it)).length;
  // a ✨ switch changed (or the system "reduce motion" setting did): every module listens to this
  const settingsChanged = () => document.dispatchEvent(new CustomEvent("fx:settings"));
  window.FX = { root, ls, motionOK, finePointer, typing, page, content, pointer, sparkle, toneOf, confetti, bigConfetti, dueCount, applyAnim, settingsChanged };
  window.confetti = confetti;
})();
