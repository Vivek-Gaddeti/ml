/* Scratchpad: an infinite handwriting canvas for stylus, finger or mouse.
   mountScratch(host, { key, height, values, fill, onClose }) -> { el, destroy }
     key     where the sketch is saved (one sketch per lesson / problem / review card)
       fill    true = take the height of the parent instead of a fixed height
   Stylus draws. One finger drags the canvas, two fingers pinch to zoom.
   Lasso tool: circle some strokes, then drag them, recolour them or bin them.
   Mouse: left draws, right or middle drag moves, wheel moves, Ctrl+wheel zooms.
   openScratchSheet({ key, title, values }) docks a pad beside the page (right side in landscape). */
(function () {
  const LS = "aihub.scratch.v1", KEEP = 80;
  let db = {};
  try { db = JSON.parse(localStorage.getItem(LS) || "{}"); } catch (e) { db = {}; }
  const persist = () => {
    try {
      const keys = Object.keys(db);
      if (keys.length > KEEP) keys.sort((a, b) => (db[a].t || 0) - (db[b].t || 0)).slice(0, keys.length - KEEP).forEach((k) => delete db[k]);
      localStorage.setItem(LS, JSON.stringify(db));
    } catch (e) {}
  };
  const S = (p) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">${p}</svg>`;
  const I = {
    pen: S('<path d="M4 20l4-1L19 8l-3-3L5 16l-1 4z"/><path d="M14 7l3 3"/>'),
    erase: S('<path d="M9 20h11"/><path d="M5 15l9-9 5 5-9 9H7l-2-2a2 2 0 0 1 0-3z"/>'),
    lasso: S('<ellipse cx="12" cy="10" rx="9" ry="6" stroke-dasharray="3 3.2"/><path d="M8 15.5c-1.5 2-.5 4 2 5"/>'),
    undo: S('<path d="M9 14L4 9l5-5"/><path d="M4 9h10a6 6 0 0 1 0 12h-3"/>'),
    full: S('<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>'),
    small: S('<path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/>'),
    x: S('<path d="M6 6l12 12M18 6L6 18"/>'),
    boxes: S('<rect x="3" y="3" width="18" height="18"/><path d="M9 3v18M15 3v18M3 9h18M3 15h18"/>'),
    trash: S('<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>')
  };
  const TPL = ["none", "grid", "axes", "ruled"], TPL_LABEL = { none: "Template off", grid: "Square grid", axes: "Axes", ruled: "Ruled lines" };
  const MINZ = 0.2, MAXZ = 5, clampZ = (z) => Math.min(MAXZ, Math.max(MINZ, z));
  let lastPen = 0;                      // time of the last stylus event anywhere: used to ignore the resting palm

  window.mountScratch = function (host, opt = {}) {
    const key = opt.key || "scratch", saved = db[key] || {};
    // strokes are stored in canvas ("world") coordinates; v is the view: top-left world point and zoom
    const st = { s: saved.s || [], h: saved.h || opt.height || 320, tpl: TPL.includes(saved.tpl) ? saved.tpl : "none", tw: saved.tw || saved.bw || 0, v: saved.v || null };
    const oldWidth = !saved.v && saved.bw ? saved.bw : 0;   // sketches saved before the canvas could move
    let tool = "pen", color = "ink", finger = !!(window.Store && Store.get("padFinger")), undo = [];
    let cur = null, erasing = false, W = 0, H = 0, dpr = 1, frame = 0, saveT = 0;
    const touches = new Map();          // pointers that move the view (fingers, right/middle mouse)
    let gesture = null;
    let sel = null, lasso = null, moving = null;   // lasso tool: selected strokes, the loop being drawn, a drag of the selection
    const act = () => (cur ? cur.id : lasso ? lasso.id : moving ? moving.id : null);   // pointer that is drawing right now

    host.innerHTML = `<div class="sp${opt.fill ? " fill" : ""}">
      <div class="sp-bar" role="toolbar" aria-label="Scratchpad tools">
        <button class="sp-t" data-a="pen" aria-pressed="true" aria-label="Pen">${I.pen}</button>
        <button class="sp-t" data-a="erase" aria-pressed="false" aria-label="Eraser">${I.erase}</button>
        <button class="sp-t" data-a="lasso" aria-pressed="false" aria-label="Lasso select">${I.lasso}</button>
        <span class="sp-sep"></span>
        <button class="sp-t" data-c="ink" aria-pressed="true" aria-label="Black ink"><i class="sp-dot"></i></button>
        <button class="sp-t" data-c="red" aria-pressed="false" aria-label="Red ink"><i class="sp-dot"></i></button>
        <button class="sp-t" data-c="blue" aria-pressed="false" aria-label="Blue ink"><i class="sp-dot"></i></button>
        <span class="sp-sep"></span>
        <button class="sp-t" data-a="tpl">${I.boxes}</button>
        <span class="sp-grow"></span>
        <button class="sp-t" data-a="undo" aria-label="Undo">${I.undo}</button>
        <button class="sp-t" data-a="clear" aria-label="Clear the pad">${I.trash}</button>
        <button class="sp-t" data-a="full" aria-label="Full screen">${I.full}</button>
        ${opt.onClose ? `<button class="sp-t" data-a="close" aria-label="Close scratchpad">${I.x}</button>` : ""}
      </div>
      <div class="sp-area"><canvas aria-label="Drawing area"></canvas><div class="sp-hint">Write or draw here</div></div>
      <div class="sp-foot"><span class="sp-status"></span><span class="sp-grow"></span>
        <button class="sp-t txt" data-a="fit" title="Show the whole drawing">Fit</button>
        <button class="sp-t txt" data-a="finger" aria-pressed="false">Draw with finger</button></div>
    </div>`;
    const el = host.firstElementChild, area = el.querySelector(".sp-area"), cv = el.querySelector("canvas");
    const ctx = cv.getContext("2d");   // not "desynchronized": on Android that makes the canvas opaque black
    const $ = (a) => el.querySelector(`[data-a="${a}"]`);
    const status = el.querySelector(".sp-status");

    /* ---- colours follow the theme: read them from the page each time we draw ---- */
    let pal = {};
    function palette() {
      const cs = getComputedStyle(el), ink = cs.color;
      const m = ink.match(/[\d.]+/g) || [0, 0, 0], dark = (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) > 140;
      pal = { ink, red: dark ? "#FF8A8A" : "#E5484D", blue: dark ? "#8CC4FF" : "#2B7FD9", muted: cs.getPropertyValue("--muted").trim() || "#888" };
      el.querySelectorAll("[data-c]").forEach((b) => (b.firstElementChild.style.background = pal[b.dataset.c]));
    }

    function resize() {
      const w = area.clientWidth, h = area.clientHeight;
      if (!w || !h) return;
      dpr = window.devicePixelRatio || 1; W = w; H = h;
      if (!st.tw) st.tw = W;
      if (!st.v) st.v = { x: 0, y: 0, z: oldWidth && st.s.length ? clampZ(W / oldWidth) : 1 };
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
      redraw();
    }

    /* ---- view ---- */
    const setT = () => { const v = st.v, k = dpr * v.z; ctx.setTransform(k, 0, 0, k, -v.x * k, -v.y * k); };
    function paintGrid() {                // the dot grid moves and scales with the canvas
      const v = st.v; let g = 22 * v.z;
      while (g < 11) g *= 2;
      area.style.backgroundSize = `${g}px ${g}px`;
      area.style.backgroundPosition = `${(-v.x * v.z) % g}px ${(-v.y * v.z) % g}px`;
    }
    const later = () => { if (!frame) frame = requestAnimationFrame(() => { frame = 0; redraw(); }); };
    const saveSoon = () => { clearTimeout(saveT); saveT = setTimeout(save, 400); };
    const showZoom = () => (status.textContent = `Zoom ${Math.round(st.v.z * 100)}%`);
    const sp = (e) => { const r = cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    function zoomAt(p, z) { const v = st.v; z = clampZ(z); const wx = p.x / v.z + v.x, wy = p.y / v.z + v.y; st.v = { x: wx - p.x / z, y: wy - p.y / z, z }; showZoom(); }

    /* ---- drawing ---- */
    // templates are drawn in world coordinates: grid and ruled lines fill the whole view, axes sit near the origin
    const AX = { x0: 60, y0: 320, w: 520, h: 280 };
    function drawTpl() {
      if (st.tpl === "none") return;
      const v = st.v, vw = W / v.z, vh = H / v.z, g = 32;
      ctx.strokeStyle = pal.muted; ctx.fillStyle = pal.muted; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      if (st.tpl === "grid" || st.tpl === "ruled") {
        ctx.save(); ctx.globalAlpha = 0.35; ctx.lineWidth = 1 / v.z; ctx.beginPath();
        const gx0 = Math.floor(v.x / g) * g, gy0 = Math.floor(v.y / g) * g;
        if (st.tpl === "grid") for (let x = gx0; x <= v.x + vw; x += g) { ctx.moveTo(x, v.y); ctx.lineTo(x, v.y + vh); }
        for (let y = gy0; y <= v.y + vh; y += g) { ctx.moveTo(v.x, y); ctx.lineTo(v.x + vw, y); }
        ctx.stroke(); ctx.restore();
        return;
      }
      const { x0, y0, w, h } = AX;
      ctx.lineWidth = 2; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(x0, y0 - h); ctx.lineTo(x0, y0); ctx.lineTo(x0 + w, y0); ctx.stroke();
      ctx.lineWidth = 1.4; ctx.beginPath();
      for (let i = 1; i <= 10; i++) { const x = x0 + i * w / 10; ctx.moveTo(x, y0 - 5); ctx.lineTo(x, y0 + 5); }
      for (let i = 1; i <= 5; i++) { const y = y0 - i * h / 5; ctx.moveTo(x0 - 5, y); ctx.lineTo(x0 + 5, y); }
      ctx.stroke();
      ctx.font = '12px "Space Mono", monospace'; ctx.fillText("x", x0 + w + 12, y0); ctx.fillText("y", x0, y0 - h - 12);
    }
    // segment i joins the midpoints around point i-1, so incremental and full redraws look identical
    function seg(p, i, last) {
      const ax = p[(i - 1) * 3], ay = p[(i - 1) * 3 + 1], aw = p[(i - 1) * 3 + 2], bx = p[i * 3], by = p[i * 3 + 1], bw = p[i * 3 + 2];
      const mx = (ax + bx) / 2, my = (ay + by) / 2;
      const sx = i > 1 ? (p[(i - 2) * 3] + ax) / 2 : ax, sy = i > 1 ? (p[(i - 2) * 3 + 1] + ay) / 2 : ay;
      ctx.lineWidth = (aw + bw) / 2;
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.quadraticCurveTo(ax, ay, mx, my);
      if (last) ctx.lineTo(bx, by);
      ctx.stroke();
    }
    function stroke(s, done) {
      const p = s.p, n = p.length / 3;
      ctx.strokeStyle = ctx.fillStyle = pal[s.c] || pal.ink; ctx.lineCap = "round"; ctx.lineJoin = "round";
      if (n === 1) { ctx.beginPath(); ctx.arc(p[0], p[1], p[2] / 2, 0, 7); ctx.fill(); return; }
      for (let i = 1; i < n; i++) seg(p, i, done && i === n - 1);
    }
    function redraw() {
      if (!W || !st.v) return;
      palette();
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height);
      setT();
      drawTpl();
      st.s.forEach((s) => stroke(s, true));
      if (cur) stroke(cur, false);
      const z = st.v.z;
      if (sel) {
        const b = selBox(), m = 8 / z;
        ctx.strokeStyle = pal.blue; ctx.lineWidth = 1.6 / z; ctx.setLineDash([7 / z, 5 / z]);
        ctx.strokeRect(b.x0 - m, b.y0 - m, b.x1 - b.x0 + 2 * m, b.y1 - b.y0 + 2 * m);
        ctx.globalAlpha = 0.07; ctx.fillStyle = pal.blue; ctx.fillRect(b.x0 - m, b.y0 - m, b.x1 - b.x0 + 2 * m, b.y1 - b.y0 + 2 * m); ctx.globalAlpha = 1;
        ctx.setLineDash([]);
      }
      if (lasso && lasso.p.length > 2) {
        ctx.strokeStyle = pal.blue; ctx.lineWidth = 1.8 / z; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.setLineDash([6 / z, 6 / z]);
        ctx.beginPath(); ctx.moveTo(lasso.p[0], lasso.p[1]);
        for (let i = 2; i < lasso.p.length; i += 2) ctx.lineTo(lasso.p[i], lasso.p[i + 1]);
        ctx.stroke(); ctx.setLineDash([]);
      }
      paintGrid();
      el.querySelector(".sp-hint").hidden = st.s.length > 0 || !!cur || st.tpl !== "none";
    }
    const save = () => { clearTimeout(saveT); st.t = Date.now(); if (st.s.length || db[key]) { db[key] = { s: st.s, h: st.h, tpl: st.tpl, tw: st.tw, v: st.v, t: st.t }; persist(); } };
    const snap = () => { undo.push(st.s.slice()); if (undo.length > 60) undo.shift(); };

    const at = (e) => {
      const r = cv.getBoundingClientRect(), v = st.v;
      const pr = e.pointerType === "pen" && e.pressure > 0 ? e.pressure : 0.45;
      const q = (n, k) => Math.round(n * k) / k;
      return [q((e.clientX - r.left) / v.z + v.x, 10), q((e.clientY - r.top) / v.z + v.y, 10), q((1.3 + pr * 3.3) / v.z, 100)];
    };
    function eraseAt(e) {
      const [x, y] = at(e), rad = 13 / st.v.z;
      const keep = st.s.filter((s) => { for (let i = 0; i < s.p.length; i += 3) if (Math.hypot(s.p[i] - x, s.p[i + 1] - y) < rad) return false; return true; });
      if (keep.length !== st.s.length) { st.s = keep; redraw(); }
    }
    /* ---- lasso: circle strokes to select them, then drag, recolour or delete ---- */
    function selBox() {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      sel.forEach((s) => { for (let i = 0; i < s.p.length; i += 3) { x0 = Math.min(x0, s.p[i]); y0 = Math.min(y0, s.p[i + 1]); x1 = Math.max(x1, s.p[i]); y1 = Math.max(y1, s.p[i + 1]); } });
      return { x0, y0, x1, y1 };
    }
    function inside(x, y, q) {             // is the point inside the closed loop q = [x, y, x, y, ...]
      let hit = false;
      for (let i = 0, j = q.length - 2; i < q.length; j = i, i += 2)
        if ((q[i + 1] > y) !== (q[j + 1] > y) && x < (q[j] - q[i]) * (y - q[i + 1]) / (q[j + 1] - q[i + 1]) + q[i]) hit = !hit;
      return hit;
    }
    function pickLasso(q) {                // a stroke is picked when at least half of it is inside the loop
      const got = st.s.filter((s) => { let n = 0; const t = s.p.length / 3; for (let i = 0; i < s.p.length; i += 3) if (inside(s.p[i], s.p[i + 1], q)) n++; return n * 2 >= t; });
      sel = got.length ? new Set(got) : null;
      status.textContent = sel ? `${got.length} stroke${got.length > 1 ? "s" : ""} selected. Drag to move, pick a colour, or tap the bin.` : "Nothing inside the loop. Circle what you want to select.";
    }
    function fresh() {                     // swap the selected strokes for copies, so undo keeps the originals
      const next = new Set();
      st.s = st.s.map((s) => { if (!sel.has(s)) return s; const c = { c: s.c, p: s.p.slice() }; next.add(c); return c; });
      sel = next;
    }
    function dropSel() { if (sel) { sel = null; redraw(); } }
    function deleteSel() {
      if (!sel) return;
      snap(); const n = sel.size; st.s = st.s.filter((s) => !sel.has(s)); sel = null;
      redraw(); save(); status.textContent = `Deleted ${n} stroke${n > 1 ? "s" : ""}. Undo brings them back.`;
    }
    function beginLasso(e) {
      const [x, y] = at(e);
      if (sel) {
        const b = selBox(), m = 14 / st.v.z;
        if (x >= b.x0 - m && x <= b.x1 + m && y >= b.y0 - m && y <= b.y1 + m) { snap(); fresh(); moving = { id: e.pointerId, x, y, moved: false }; return; }
      }
      sel = null; lasso = { id: e.pointerId, p: [x, y] }; redraw();
    }

    function fit() {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      const add = (x, y) => { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); };
      st.s.forEach((s) => { for (let i = 0; i < s.p.length; i += 3) add(s.p[i], s.p[i + 1]); });
      if (st.tpl === "axes") { add(AX.x0 - 20, AX.y0 - AX.h - 20); add(AX.x0 + AX.w + 20, AX.y0 + 20); }
      if (x0 === Infinity) st.v = { x: 0, y: 0, z: 1 };
      else {
        const pad = 28, z = clampZ(Math.min((W - 2 * pad) / Math.max(x1 - x0, 1), (H - 2 * pad) / Math.max(y1 - y0, 1), 1.5));
        st.v = { x: (x0 + x1) / 2 - W / 2 / z, y: (y0 + y1) / 2 - H / 2 / z, z };
      }
      redraw(); save(); showZoom();
    }

    /* ---- moving the view: one pointer drags, two pinch ---- */
    function startGesture() {
      const p = [...touches.values()];
      if (!p.length) { gesture = null; return; }
      gesture = p.length === 1 ? { n: 1, v0: { ...st.v }, p0: p[0] }
        : { n: 2, v0: { ...st.v }, m0: { x: (p[0].x + p[1].x) / 2, y: (p[0].y + p[1].y) / 2 }, d0: Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y) || 1 };
    }
    function moveGesture() {
      if (!gesture) return;
      const p = [...touches.values()], v0 = gesture.v0;
      if (gesture.n === 1) st.v = { x: v0.x - (p[0].x - gesture.p0.x) / v0.z, y: v0.y - (p[0].y - gesture.p0.y) / v0.z, z: v0.z };
      else if (p.length > 1) {
        const m = { x: (p[0].x + p[1].x) / 2, y: (p[0].y + p[1].y) / 2 };
        const z = clampZ(v0.z * Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y) / gesture.d0);
        const wx = gesture.m0.x / v0.z + v0.x, wy = gesture.m0.y / v0.z + v0.y;
        st.v = { x: wx - m.x / z, y: wy - m.y / z, z }; showZoom();
      }
      later();
    }
    function beginStroke(e) {
      e.preventDefault();
      try { cv.setPointerCapture(e.pointerId); } catch (err) {}
      // the eraser end or the side button of a stylus erases without switching tools
      if (tool === "erase" || e.button === 5 || (e.buttons & 32) || (e.pointerType === "pen" && (e.buttons & 2))) { sel = null; snap(); erasing = e.pointerId; eraseAt(e); redraw(); return; }
      if (tool === "lasso") { beginLasso(e); return; }
      snap();
      if (e.pointerType === "pen") status.textContent = "Stylus detected. Press harder for thicker lines.";
      cur = { c: color, p: at(e), id: e.pointerId };
      el.querySelector(".sp-hint").hidden = true;
      palette(); setT(); stroke(cur, false);
    }

    cv.addEventListener("pointerdown", (e) => {
      if (e.pointerType === "pen") lastPen = Date.now();
      if (e.pointerType === "touch") {
        if (erasing || (act() != null && !touches.has(act()))) return;   // the stylus is writing: this is the palm
        if (Date.now() - lastPen < 900 || e.width * e.height > 1600) return;  // pen just lifted, or a palm-sized contact
        e.preventDefault();
        try { cv.setPointerCapture(e.pointerId); } catch (err) {}
        touches.set(e.pointerId, sp(e));
        if (finger && touches.size === 1) { beginStroke(e); return; }
        if (cur) { cur = null; undo.pop(); redraw(); }               // a second finger turns a finger stroke into a pinch
        if (lasso) { lasso = null; redraw(); }
        if (moving) { moving = null; sel = null; st.s = undo.pop(); redraw(); }
        startGesture(); return;
      }
      if (e.pointerType === "mouse" && (e.button === 1 || e.button === 2)) {
        e.preventDefault();
        try { cv.setPointerCapture(e.pointerId); } catch (err) {}
        touches.set(e.pointerId, sp(e)); startGesture(); return;
      }
      if (act() != null || erasing) return;
      if (e.pointerType === "pen" && gesture) { st.v = { ...gesture.v0 }; gesture = null; touches.clear(); }  // a palm landed first: undo its drag
      beginStroke(e);
    });
    cv.addEventListener("pointermove", (e) => {
      if (e.pointerType === "pen") lastPen = Date.now();
      if (touches.has(e.pointerId)) {
        touches.set(e.pointerId, sp(e));
        if (act() !== e.pointerId) { moveGesture(); return; }
      }
      if (erasing === e.pointerId) { eraseAt(e); return; }
      if (lasso && lasso.id === e.pointerId) {
        const [x, y] = at(e), n = lasso.p.length;
        if (Math.abs(x - lasso.p[n - 2]) + Math.abs(y - lasso.p[n - 1]) > 2 / st.v.z) { lasso.p.push(x, y); later(); }
        return;
      }
      if (moving && moving.id === e.pointerId) {
        const [x, y] = at(e), dx = x - moving.x, dy = y - moving.y;
        if (!dx && !dy) return;
        sel.forEach((s) => { for (let i = 0; i < s.p.length; i += 3) { s.p[i] = Math.round((s.p[i] + dx) * 10) / 10; s.p[i + 1] = Math.round((s.p[i + 1] + dy) * 10) / 10; } });
        moving.x = x; moving.y = y; moving.moved = true; later();
        return;
      }
      if (!cur || e.pointerId !== cur.id) return;
      const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
      ctx.strokeStyle = pal[cur.c]; ctx.lineCap = "round"; ctx.lineJoin = "round";
      (evs.length ? evs : [e]).forEach((ev) => {
        const p = at(ev), n = cur.p.length;
        if (Math.abs(p[0] - cur.p[n - 3]) + Math.abs(p[1] - cur.p[n - 2]) < 0.6 / st.v.z) return;
        cur.p.push(p[0], p[1], p[2]);
        seg(cur.p, cur.p.length / 3 - 1, false);
      });
    });
    function end(e) {
      if (touches.has(e.pointerId)) {
        touches.delete(e.pointerId);
        if (act() !== e.pointerId) { if (touches.size) startGesture(); else { gesture = null; saveSoon(); } return; }
      }
      if (lasso && lasso.id === e.pointerId) { const q = lasso.p; lasso = null; if (q.length >= 6) pickLasso(q); redraw(); return; }
      if (moving && moving.id === e.pointerId) { const m = moving.moved; moving = null; if (m) save(); else undo.pop(); return; }
      if (erasing === e.pointerId) { erasing = false; if (undo.length && undo[undo.length - 1].length === st.s.length) undo.pop(); save(); return; }
      if (!cur || e.pointerId !== cur.id) return;
      st.s.push({ c: cur.c, p: cur.p }); cur = null;
      redraw(); save();
    }
    cv.addEventListener("pointerup", end); cv.addEventListener("pointercancel", end);
    cv.addEventListener("contextmenu", (e) => e.preventDefault());
    cv.addEventListener("mousedown", (e) => { if (e.button === 1) e.preventDefault(); });   // no autoscroll on middle-drag
    // wheel moves the canvas when the pad fills its space; a small pad inside a page lets the page scroll
    area.addEventListener("wheel", (e) => {
      if (!(opt.fill || el.classList.contains("full") || e.ctrlKey || e.metaKey || e.shiftKey)) return;
      e.preventDefault();
      const unit = e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? H : 1;
      if (e.ctrlKey || e.metaKey) zoomAt(sp(e), st.v.z * Math.exp(-e.deltaY * unit * 0.0025));
      else {
        let dx = e.deltaX * unit, dy = e.deltaY * unit;
        if (e.shiftKey && !dx) { dx = dy; dy = 0; }
        st.v = { x: st.v.x + dx / st.v.z, y: st.v.y + dy / st.v.z, z: st.v.z };
      }
      later(); saveSoon();
    }, { passive: false });

    /* ---- toolbar ---- */
    const press = (q, on) => el.querySelectorAll(q).forEach((b) => b.setAttribute("aria-pressed", String(b === on)));
    const TOOLS = '[data-a="pen"],[data-a="erase"],[data-a="lasso"]';
    const paintTpl = () => { $("tpl").setAttribute("aria-label", "Boxes to draw on: " + TPL_LABEL[st.tpl]); $("tpl").setAttribute("aria-pressed", String(st.tpl !== "none")); };
    const paintFinger = () => {
      $("finger").setAttribute("aria-pressed", String(finger));
      status.textContent = finger ? "Finger and stylus draw. Two fingers move and zoom." : "The stylus draws. Drag with a finger to move, pinch to zoom.";
    };
    function setFull(on) {
      el.classList.toggle("full", on); document.body.classList.toggle("sp-lock", on);
      $("full").innerHTML = on ? I.small : I.full; $("full").setAttribute("aria-label", on ? "Leave full screen" : "Full screen");
    }
    el.addEventListener("click", (e) => {
      const b = e.target.closest(".sp-t"); if (!b) return;
      if (b.dataset.c) {
        color = b.dataset.c; press("[data-c]", b);
        if (tool === "lasso" && sel) { snap(); fresh(); sel.forEach((s) => (s.c = color)); redraw(); save(); return; }   // recolour the selection
        tool = "pen"; press(TOOLS, $("pen")); dropSel(); return;
      }
      const a = b.dataset.a;
      if (a === "pen" || a === "erase" || a === "lasso") {
        tool = a; press(TOOLS, b); if (a !== "lasso") dropSel();
        if (a === "lasso") status.textContent = "Circle what you want to select, then drag it to move it.";
      }
      else if (a === "tpl") { st.tpl = TPL[(TPL.indexOf(st.tpl) + 1) % TPL.length]; paintTpl(); redraw(); save(); status.textContent = TPL_LABEL[st.tpl]; }
      else if (a === "undo") { if (undo.length) { sel = null; st.s = undo.pop(); redraw(); save(); } }
      else if (a === "clear" && sel) deleteSel();                 // with a selection, the bin deletes only that
      else if (a === "clear") { if (st.s.length) { snap(); st.s = []; st.v = { x: 0, y: 0, z: 1 }; st.tw = W; redraw(); save(); } }
      else if (a === "fit") fit();
      else if (a === "full") setFull(!el.classList.contains("full"));
      else if (a === "close") { setFull(false); opt.onClose(); }
      else if (a === "finger") { finger = !finger; if (window.Store) Store.set("padFinger", finger || undefined); paintFinger(); }
    });
    const onKey = (e) => {
      if (sel && !/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName || "") && !e.target.isContentEditable) {
        if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); deleteSel(); return; }
        if (e.key === "Escape") { e.preventDefault(); dropSel(); return; }
      }
      if (e.key === "Escape" && el.classList.contains("full")) { e.preventDefault(); setFull(false); }
    };
    addEventListener("keydown", onKey);

    if (!opt.fill) area.style.height = st.h + "px";
    paintTpl(); paintFinger();
    const ro = window.ResizeObserver ? new ResizeObserver(resize) : null;
    if (ro) ro.observe(area); else addEventListener("resize", resize);
    const mq = matchMedia("(prefers-color-scheme: dark)"); const onTheme = () => redraw();
    mq.addEventListener && mq.addEventListener("change", onTheme);
    const mo = new MutationObserver(onTheme); mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(redraw);
    resize();

    return { el, destroy() { if (gesture || st.v) save(); ro && ro.disconnect(); mo.disconnect(); removeEventListener("keydown", onKey); document.body.classList.remove("sp-lock"); host.innerHTML = ""; } };
  };

    /* a pad docked beside the page: right side in landscape, bottom in portrait */
  let sheet = null;
  window.closeScratchSheet = function () { if (!sheet) return; sheet.pad.destroy(); sheet.el.remove(); sheet = null; document.body.classList.remove("sheet-open"); };
  window.openScratchSheet = function (opt = {}) {
    if (sheet) return closeScratchSheet();
    const el = document.createElement("aside");
    el.className = "sp-sheet"; el.setAttribute("aria-label", "Scratchpad");
    el.innerHTML = `<div class="sp-sheet-b"></div>`;
    document.body.append(el); document.body.classList.add("sheet-open");
    sheet = { el, pad: mountScratch(el.querySelector(".sp-sheet-b"), { key: opt.key, values: opt.values, fill: true, onClose: () => closeScratchSheet() }) };
  };
})();
