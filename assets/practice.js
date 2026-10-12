/* practice.js — the Practice bank: implement-from-scratch exercises (list, roadmap, problem pages).
   Same engine as the DSA site's NeetCode page. Problems come from data/prac/*.js (PRAC.add(category, [...])). */
(function () {
  const main = $("#content");
  const list = pracSorted();
  let F = { q: "", cat: "", d: "", s: "" };
  try { F = Object.assign(F, JSON.parse(sessionStorage.getItem(lsKey("prac:filters")) || "{}")); } catch (e) {}
  const practice = () => { try { return localStorage.getItem(lsKey("prac:practice")) === "1"; } catch (e) { return false; } };

  /* ---------------- List view ---------------- */
  function statusDot(p) {
    const it = Store.item("prac:" + p.id);
    const title = it.status === "done" ? "Solved — click to unmark" : it.status === "revise" ? "Needs revision" : "Click to mark solved";
    const style = it.status === "done" ? "background:var(--green);border-color:var(--green);color:#fff" : it.status === "revise" ? "background:var(--red-soft);border-color:var(--red);color:var(--red)" : "";
    return `<button class="icon-btn nc-dot" data-k="${p.id}" data-a="tog" title="${title}" style="${style}">${it.status === "done" ? icon("check") : it.status === "revise" ? icon("repeat") : ""}</button>`;
  }
  function match(p) {
    const it = Store.item("prac:" + p.id);
    if (F.q && !(p.t + " " + p.p + " " + p.cat).toLowerCase().includes(F.q.toLowerCase())) return false;
    if (F.cat && p.cat !== F.cat) return false;
    if (F.d && p.d !== F.d) return false;
    if (F.s === "todo" && it.status) return false;
    if (F.s === "done" && it.status !== "done") return false;
    if (F.s === "revise" && it.status !== "revise") return false;
    if (F.s === "star" && !it.star) return false;
    if (F.s === "due" && !Store.isDue(it)) return false;
    return true;
  }
  function listView() {
    document.title = "Practice · AI Study Hub";
    main.classList.remove("reading");
    const solved = list.filter((p) => Store.item("prac:" + p.id).status).length;
    const byD = (d) => { const all = list.filter((p) => p.d === d); return [all.filter((p) => Store.item("prac:" + p.id).status).length, all.length]; };
    const [e1, e2] = byD("Easy"), [m1, m2] = byD("Medium"), [h1, h2] = byD("Hard");
    const next = list.find((p) => !Store.item("prac:" + p.id).status);
    let h = `<h1>Practice</h1>
      <p class="lead">Build the core pieces of ML yourself: softmax, gradient descent, k-means, backprop, attention, a tokenizer, a ReAct loop… Each problem has a hint, a plain-Python version, a vectorized NumPy version, a dry run, and a <strong>Try it yourself</strong> box that checks your code against tests.</p>
      <div class="grid cols-4 stats">
        <div class="card"><div class="stat-label">Solved</div><div class="stat">${solved}<span class="stat-of"> / ${list.length}</span></div><div class="bar"><span style="width:${(solved / Math.max(1, list.length)) * 100}%"></span></div></div>
        <div class="card"><div class="stat-label">Easy</div><div class="stat text-green">${e1}<span class="stat-of"> / ${e2}</span></div></div>
        <div class="card"><div class="stat-label">Medium</div><div class="stat text-amber">${m1}<span class="stat-of"> / ${m2}</span></div></div>
        <div class="card"><div class="stat-label">Hard</div><div class="stat text-red">${h1}<span class="stat-of"> / ${h2}</span></div></div>
      </div>
      ${viewSwitch("list")}
      ${next ? `<div class="callout tip"><span class="label">Up next</span><a href="#p/${next.id}"><strong>${esc(next.t)}</strong></a> · ${esc(next.cat)} · <span class="badge ${next.d.toLowerCase()}">${next.d}</span></div>` : ""}
      <div class="btn-row filter-bar">
        <input type="search" id="fq" placeholder="Search problems or topics…" value="${esc(F.q)}">
        <select id="fcat"><option value="">All categories</option>${PRAC_CATS.map((c) => `<option ${F.cat === c ? "selected" : ""}>${c}</option>`).join("")}</select>
        <select id="fd"><option value="">All levels</option>${["Easy", "Medium", "Hard"].map((d) => `<option ${F.d === d ? "selected" : ""}>${d}</option>`).join("")}</select>
        <select id="fs">${[["", "Any status"], ["todo", "Not solved"], ["done", "Solved"], ["revise", "Needs revision"], ["due", "Due for review"], ["star", "Bookmarked"]].map(([v, l]) => `<option value="${v}" ${F.s === v ? "selected" : ""}>${l}</option>`).join("")}</select>
      </div>`;
    let shown = 0, n = 0;
    PRAC_CATS.forEach((cat) => {
      const all = list.filter((p) => p.cat === cat);
      const ps = all.filter(match);
      if (!ps.length) return;
      shown += ps.length;
      const d = all.filter((p) => Store.item("prac:" + p.id).status).length;
      h += `<h2 class="cat-head"><span class="cat-badge">${pracIcon(cat)}</span>${esc(cat)} <span class="small muted fw-500">${d} / ${all.length}</span><span class="bar"><span style="width:${(d / all.length) * 100}%"></span></span></h2>
        <div class="table-wrap"><table><thead><tr><th class="col-44"></th><th>Problem</th><th>Difficulty</th><th>Topic</th><th class="col-40"></th></tr></thead><tbody>`;
      ps.forEach((p) => {
        const it = Store.item("prac:" + p.id);
        h += `<tr><td>${statusDot(p)}</td>
          <td><a href="#p/${p.id}"><strong>${esc(p.t)}</strong></a>${it.best ? `<span class="solve-time" title="Your best solve time">${icon("timer")} ${mmss(it.best)}</span>` : ""}${it.note ? ` <span class="muted" title="Has notes">${icon("note")}</span>` : ""}${Store.isDue(it) ? ' <span class="badge purple">due</span>' : ""}</td>
          <td><span class="badge ${p.d.toLowerCase()}">${p.d}</span></td>
          <td><span class="small muted">${esc(p.p || "")}</span></td>
          <td><button class="icon-btn icon-plain text-amber" data-k="${p.id}" data-a="star" title="Bookmark">${icon(it.star ? "star-fill" : "star")}</button></td></tr>`;
      });
      h += `</tbody></table></div>`;
    });
    if (!shown) h += list.length ? `<p class="muted">No problems match these filters.</p>` : `<div class="card empty-state"><div class="big">${icon("flask")}</div><p class="muted">Problems arrive with each stage of the course.</p></div>`;
    main.innerHTML = h;
    const save = () => { try { sessionStorage.setItem(lsKey("prac:filters"), JSON.stringify(F)); } catch (e) {} };
    const fq = $("#fq");
    fq.oninput = () => { F.q = fq.value; save(); const pos = fq.selectionStart; listView(); const n2 = $("#fq"); n2.focus(); n2.setSelectionRange(pos, pos); };
    $("#fcat").onchange = (e) => { F.cat = e.target.value; save(); listView(); };
    $("#fd").onchange = (e) => { F.d = e.target.value; save(); listView(); };
    $("#fs").onchange = (e) => { F.s = e.target.value; save(); listView(); };
  }
  main.addEventListener("click", (e) => {
    const b = e.target.closest("[data-a][data-k]");
    if (!b) return;
    const k = "prac:" + b.dataset.k, it = Store.item(k);
    if (b.dataset.a === "tog") { Store.setStatus(k, it.status ? null : "done"); if (!it.status) toast("Solved. Review scheduled for tomorrow", 1800, "check"); }
    if (b.dataset.a === "star") Store.toggleStar(k);
    const y = window.scrollY;
    if (curView === "roadmap") roadmapView(true); else listView();
    window.scrollTo(0, y);
  });


  /* ---------------- Roadmap view ----------------
     A full-screen canvas: the practice categories as nodes joined by lines in the suggested order (assets/site.js PRAC_MAP),
     each showing its progress. Drag to pan, wheel / pinch to zoom, + − 0 keys, fit button. Click a node for its
     problems. Floating on the right (can be hidden): solved by difficulty with a gauge, and a study calendar. */
  const RM_W = 760, RM_H = 740;
  const RM_NODES = PRAC_MAP.nodes, RM_EDGES = PRAC_MAP.edges;
  const NODE_H = 19;                                    // half the node height in map units (lines start/end at node edges)
  let rmMonth = null, rmTick = 0, rmOpen = null;
  const catStats = (c) => { const all = list.filter((p) => p.cat === c); return { all, done: all.filter((p) => Store.item("prac:" + p.id).status).length }; };

  const SC = 1.35, WW = RM_W * SC, WH = RM_H * SC, NHALF = 26;         // world size (px) and half a node's height
  const view = { x: 0, y: 0, k: 1, ready: false };                     // pan / zoom (kept for this browser tab)
  try { Object.assign(view, JSON.parse(sessionStorage.getItem(lsKey("rm:view")) || "{}")); } catch (e) {}
  let sideOpen = true;
  try { const v = localStorage.getItem(lsKey("rm:side")); sideOpen = v ? v === "1" : innerWidth > 900; } catch (e) {}
  const SVG = (d) => `<svg class="cat-ico" viewBox="0 0 24 24" aria-hidden="true">${d}</svg>`;
  const sideIcon = () => `<span title="${sideOpen ? "Hide" : "Show"} stats & calendar">${SVG('<rect x="3" y="4" width="18" height="16" rx="2.5"/><path d="M15 4v16"/>' + (sideOpen ? '<path d="m8 10 2 2-2 2"/>' : '<path d="m10 10-2 2 2 2"/>'))}</span>`;

  function roadmapView(quiet) {
    document.title = "Roadmap · Practice";
    main.classList.remove("reading"); main.classList.add("rm-mode"); document.body.classList.add("rm-body");
    const solved = list.filter((p) => Store.item("prac:" + p.id).status).length;
    const by = (d) => { const all = list.filter((p) => p.d === d); return [all.filter((p) => Store.item("prac:" + p.id).status).length, all.length]; };
    const [e1, e2] = by("Easy"), [m1, m2] = by("Medium"), [h1, h2] = by("Hard");
    // lines: from the bottom of a node to the top of the next; green once the earlier category is complete
    const lines = RM_EDGES.map(([a, b], i) => {
      const [x1, y1] = RM_NODES[a].map((v) => v * SC), [x2, y2] = RM_NODES[b].map((v) => v * SC), ya = y1 + NHALF, yb = y2 - NHALF, my = (ya + yb) / 2;
      const s = catStats(a), done = s.all.length > 0 && s.done === s.all.length;
      return `<path class="${done ? "done" : ""}" style="--i:${i}" pathLength="1" d="M${x1} ${ya} C${x1} ${my}, ${x2} ${my}, ${x2} ${yb}"/>`;
    }).join("");
    const nodes = PRAC_CATS.map((c, i) => {
      const [x, y] = RM_NODES[c].map((v) => v * SC), s = catStats(c), pct = s.all.length ? (s.done / s.all.length) * 100 : 0;
      return `<button type="button" class="rm-node${s.all.length && s.done === s.all.length ? " done" : s.done ? " started" : ""}${s.all.length ? "" : " empty"}" data-cat="${esc(c)}" style="left:${x}px;top:${y}px;--i:${i}" title="${esc(c)}: ${s.all.length ? `${s.done} / ${s.all.length} solved` : "problems coming soon"}">
        <span class="rm-name">${pracIcon(c, "rm-ico")}${esc(c)}</span><span class="rm-meta"><span class="rm-bar"><span style="width: ${pct.toFixed(1)}%"></span></span><span class="rm-n">${s.done}/${s.all.length}</span></span></button>`;
    }).join("");
    // gauge: a 270° ring split into the solved Easy / Medium / Hard share of all problems
    const total = list.length || 1, seg = (v) => (v / total) * 75;
    let off = 0;
    const arcSvg = [["var(--green)", seg(e1)], ["var(--amber)", seg(m1)], ["var(--red)", seg(h1)]].filter(([, len]) => len > 0.01)
      .map(([col, len]) => { const sv = `<circle class="g-arc" cx="60" cy="60" r="48" pathLength="100" style="stroke:${col};stroke-dasharray:${len.toFixed(2)} 100;stroke-dashoffset:${(-off).toFixed(2)}"/>`; off += len; return sv; }).join("");
    main.innerHTML = `
      <div class="rmc${quiet ? " rm-static" : ""}" id="rmc" aria-label="Roadmap canvas: drag to move, scroll to zoom">
        <div class="rmc-world" style="width:${WW}px;height:${WH}px"><svg class="rm-lines" width="${WW}" height="${WH}" viewBox="0 0 ${WW} ${WH}" aria-hidden="true">${lines}</svg>${nodes}</div>
      </div>
      <div class="rmc-top">${viewSwitch("roadmap")}<div class="rmc-title"><b>Practice Roadmap</b><span>${solved} / ${list.length} solved · drag to move · scroll to zoom</span></div></div>
      <div class="rmc-ctrl" role="toolbar" aria-label="Zoom">
        <button type="button" data-z="in" title="Zoom in (+)">${SVG('<path d="M12 5v14M5 12h14"/>')}</button>
        <button type="button" data-z="out" title="Zoom out (−)">${SVG('<path d="M5 12h14"/>')}</button>
        <button type="button" data-z="fit" title="Fit to screen (0)">${SVG('<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>')}</button>
      </div>
      <button type="button" class="rmc-side-tog" data-side>${sideIcon()}</button>
      <aside class="rmc-side${sideOpen ? "" : " closed"}${quiet ? " rm-static" : ""}">
        <div class="card rm-stats">
          <div class="rm-diff">
            <div><span class="text-green">Easy</span><b>${e1}<small>/${e2}</small></b></div>
            <div><span class="text-amber">Medium</span><b>${m1}<small>/${m2}</small></b></div>
            <div><span class="text-red">Hard</span><b>${h1}<small>/${h2}</small></b></div>
          </div>
          <div class="rm-gauge"><svg viewBox="0 0 120 120" aria-hidden="true"><circle class="g-track" cx="60" cy="60" r="48" pathLength="100"/>${arcSvg}</svg>
            <div class="rm-gauge-t"><b>${solved}</b><span>/${list.length}</span><small>Solved</small></div></div>
        </div>
        <div class="card rm-cal" id="rmCal"></div>
      </aside>
      <div class="rm-drawer-bg" hidden></div><aside class="rm-drawer" hidden aria-label="Category problems"></aside>`;
    calendar();
    if (rmOpen) openDrawer(rmOpen, true);
    clearInterval(rmTick);
    rmTick = setInterval(() => { const el = document.getElementById("rmLeft"); if (el) el.textContent = timeLeft(); else clearInterval(rmTick); }, 1000);
    canvas(main.querySelector("#rmc"));
  }
  function leaveRoadmap() { main.classList.remove("rm-mode"); document.body.classList.remove("rm-body"); clearInterval(rmTick); }

  /* pan & zoom: the world is moved with one transform; the dot grid moves and scales with it, so it feels endless */
  let zoomTo = null, fitView = null;
  function canvas(cv) {
    const world = cv.firstElementChild;
    const apply = (anim) => {
      world.style.transition = anim && document.documentElement.dataset.anim === "on" ? "transform .35s cubic-bezier(.22,1,.36,1)" : "none";
      world.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.k})`;
      cv.style.backgroundSize = `${26 * view.k}px ${26 * view.k}px`;
      cv.style.backgroundPosition = `${view.x}px ${view.y}px`;
      view.ready = true;
      try { sessionStorage.setItem(lsKey("rm:view"), JSON.stringify(view)); } catch (e) {}
    };
    const clampK = (k) => Math.min(2.5, Math.max(0.3, k));
    zoomTo = (cx, cy, factor, anim) => {
      const k2 = clampK(view.k * factor), f = k2 / view.k;
      view.x = cx - (cx - view.x) * f; view.y = cy - (cy - view.y) * f; view.k = k2;
      apply(anim);
    };
    fitView = (anim) => {
      const r = cv.getBoundingClientRect(), side = sideOpen && innerWidth > 900 ? 340 : 0, top = 70, pad = 40;
      const aw = r.width - side - pad * 2, ah = r.height - top - pad * 2;
      view.k = clampK(Math.min(aw / WW, ah / WH));
      view.x = pad + (aw - WW * view.k) / 2; view.y = top + pad + (ah - WH * view.k) / 2;
      apply(anim);
    };
    if (!view.ready) fitView(false); else apply(false);
    // drag to pan (a short drag never counts as a click on a node), two fingers to pinch-zoom
    const pts = new Map();
    let moved = 0;
    cv.addEventListener("pointerdown", (e) => { if (e.button > 0) return; pts.set(e.pointerId, { x: e.clientX, y: e.clientY }); if (pts.size === 1) moved = 0; });
    cv.addEventListener("pointermove", (e) => {
      const p = pts.get(e.pointerId);
      if (!p) return;
      if (pts.size === 2) {
        const [a, b] = [...pts.values()], d0 = Math.hypot(a.x - b.x, a.y - b.y);
        p.x = e.clientX; p.y = e.clientY;
        const [a2, b2] = [...pts.values()], d1 = Math.hypot(a2.x - b2.x, a2.y - b2.y), r = cv.getBoundingClientRect();
        if (d0 > 0) zoomTo((a2.x + b2.x) / 2 - r.left, (a2.y + b2.y) / 2 - r.top, d1 / d0);
        moved = 99; return;
      }
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      moved += Math.abs(dx) + Math.abs(dy);
      if (moved > 4) {
        try { if (!cv.hasPointerCapture(e.pointerId)) cv.setPointerCapture(e.pointerId); } catch (x) {}   // keep dragging even outside the canvas
        cv.classList.add("grabbing");
        view.x += dx; view.y += dy; apply(false);
      }
    });
    const up = (e) => { pts.delete(e.pointerId); if (!pts.size) cv.classList.remove("grabbing"); };
    cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
    cv.addEventListener("click", (e) => { if (moved > 4) { e.stopPropagation(); e.preventDefault(); } }, true);
    // wheel: a mouse wheel (or trackpad pinch) zooms toward the cursor; two-finger trackpad scrolling pans
    cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect(), mouse = e.deltaMode === 1 || (Math.abs(e.deltaY) >= 50 && Math.abs(e.deltaX) < 1);
      if (e.ctrlKey || mouse) zoomTo(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0016)), mouse && !e.ctrlKey);
      else { view.x -= e.deltaX; view.y -= e.deltaY; apply(false); }
    }, { passive: false });
  }
  const timeLeft = () => { const n = new Date(), m = new Date(n); m.setHours(24, 0, 0, 0); const s = Math.floor((m - n) / 1000); return [s / 3600, (s % 3600) / 60, s % 60].map((v) => String(Math.floor(v)).padStart(2, "0")).join(":") + " left"; };
  function calendar() {
    const el = document.getElementById("rmCal");
    if (!el) return;
    const now = new Date(), act = Store.load().activity;
    if (!rmMonth) rmMonth = new Date(now.getFullYear(), now.getMonth(), 1);
    const y = rmMonth.getFullYear(), m = rmMonth.getMonth(), first = new Date(y, m, 1).getDay(), n = new Date(y, m + 1, 0).getDate();
    const started = Object.keys(act).filter((d) => act[d]).sort()[0] || today();      // "missed" only counts after your first study day
    let cells = "";
    for (let i = 0; i < first; i++) cells += "<span></span>";
    for (let d = 1; d <= n; d++) {
      const key = today(new Date(y, m, d).getTime()), isToday = key === today(), studied = !!act[key];
      const missed = !studied && !isToday && key < today() && key >= started;
      cells += `<span class="cal-d${studied ? " on" : ""}${missed ? " miss" : ""}${isToday ? " today" : ""}" title="${key}${studied ? ": studied" : missed ? ": missed" : ""}">${d}</span>`;
    }
    const same = y === now.getFullYear() && m === now.getMonth();
    el.innerHTML = `<div class="cal-head"><button type="button" class="icon-btn icon-plain" data-cal="-1" aria-label="Previous month">‹</button><b>${rmMonth.toLocaleString(undefined, { month: "long", year: "numeric" })}</b><button type="button" class="icon-btn icon-plain" data-cal="1" aria-label="Next month">›</button></div>
      ${same ? `<div class="cal-day"><b>Day ${now.getDate()}</b><span id="rmLeft">${timeLeft()}</span></div>` : ""}
      <div class="cal-grid">${"SMTWTFS".split("").map((d) => `<span class="cal-w">${d}</span>`).join("")}${cells}</div>
      <div class="cal-streaks"><div><small>Current streak</small><b>${icon("flame", "ico-amber")} ${Store.streak()}</b></div><div><small>Best streak</small><b>${icon("trophy", "ico-amber")} ${Store.bestStreak()}</b></div></div>`;
  }
  function openDrawer(cat, instant) {
    rmOpen = cat;
    const d = main.querySelector(".rm-drawer"), bg = main.querySelector(".rm-drawer-bg");
    if (!d) return;
    const s = catStats(cat), pct = s.all.length ? (s.done / s.all.length) * 100 : 0;
    d.innerHTML = `<div class="rm-dh"><span class="cat-badge">${pracIcon(cat)}</span><div class="flex-1"><b>${esc(cat)}</b><div class="small muted">${s.done} / ${s.all.length} solved</div></div><button type="button" class="icon-btn" data-close aria-label="Close">${icon("x")}</button></div>
      <div class="bar"><span style="width: ${pct.toFixed(1)}%"></span></div>
      <ul class="rm-list">${s.all.length ? "" : '<li class="muted small">Problems for this topic arrive with its stage of the course.</li>'}${s.all.map((p) => `<li>${statusDot(p)}<a href="#p/${p.id}">${esc(p.t)}</a><span class="badge ${p.d.toLowerCase()}">${p.d}</span></li>`).join("")}</ul>
      <a class="btn sm mt-4" href="#list" data-cat-filter="${esc(cat)}">Open in the list →</a>`;
    d.hidden = false; bg.hidden = false;
    if (instant) d.classList.add("no-anim");
  }
  function closeDrawer() { rmOpen = null; const d = main.querySelector(".rm-drawer"); if (d) { d.hidden = true; main.querySelector(".rm-drawer-bg").hidden = true; } }
  main.addEventListener("click", (e) => {
    const z = e.target.closest("[data-z]");
    if (z && zoomTo) { const r = main.querySelector("#rmc").getBoundingClientRect(); if (z.dataset.z === "fit") fitView(true); else zoomTo(r.width / 2, r.height / 2, z.dataset.z === "in" ? 1.25 : 0.8, true); return; }
    if (e.target.closest("[data-side]")) {
      // slide the panel in / out in place (no re-render)
      sideOpen = !sideOpen; try { localStorage.setItem(lsKey("rm:side"), sideOpen ? "1" : "0"); } catch (x) {}
      main.querySelector(".rmc-side").classList.toggle("closed", !sideOpen);
      main.querySelector("[data-side]").innerHTML = sideIcon(); return;
    }
    const node = e.target.closest(".rm-node");
    if (node) { openDrawer(node.dataset.cat); return; }
    if (e.target.closest("[data-close], .rm-drawer-bg")) { closeDrawer(); return; }
    const c = e.target.closest("[data-cal]");
    if (c) { rmMonth = new Date(rmMonth.getFullYear(), rmMonth.getMonth() + +c.dataset.cal, 1); calendar(); return; }
    const f = e.target.closest("[data-cat-filter]");
    if (f) { F = { q: "", cat: f.dataset.catFilter, d: "", s: "" }; try { sessionStorage.setItem(lsKey("prac:filters"), JSON.stringify(F)); } catch (x) {} rmOpen = null; }
    if (e.target.closest(".rm-list a")) rmOpen = null;
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && rmOpen) { closeDrawer(); return; }
    if (curView !== "roadmap" || !zoomTo || e.ctrlKey || e.metaKey || (e.target.closest && e.target.closest("input, textarea, select"))) return;
    const r = main.querySelector("#rmc") && main.querySelector("#rmc").getBoundingClientRect();
    if (!r) return;
    if (e.key === "+" || e.key === "=") zoomTo(r.width / 2, r.height / 2, 1.25, true);
    else if (e.key === "-") zoomTo(r.width / 2, r.height / 2, 0.8, true);
    else if (e.key === "0") fitView(true);
  });
  // List | Roadmap switch (remembered)
  const viewSwitch = (v) => `<div class="view-switch" role="tablist"><a href="#list" class="${v === "list" ? "on" : ""}">${ncIconRaw("list")}List</a><a href="#roadmap" class="${v === "roadmap" ? "on" : ""}">${ncIconRaw("map")}Roadmap</a></div>`;
  const ncIconRaw = (k) => `<svg class="cat-ico" viewBox="0 0 24 24" aria-hidden="true">${k === "list" ? '<path d="M8 6h12M8 12h12M8 18h12"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/>' : '<circle cx="12" cy="4.5" r="2"/><circle cx="6" cy="12" r="2"/><circle cx="18" cy="12" r="2"/><circle cx="12" cy="19.5" r="2"/><path d="M10.8 6.2 7.2 10.3M13.2 6.2l3.6 4.1M7.2 13.7l3.6 4.1M16.8 13.7l-3.6 4.1"/>'}</svg>`;
  let curView = "list";

  /* ---------------- Problem timer ----------------
     Starts when you open a problem and only counts while the page is visible on screen.
     When you mark the problem solved, the time is saved (last + best) and shown here and in the list. */
  const mmss = (sec) => { sec = Math.round(sec); const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60; return (h ? h + ":" + String(m).padStart(2, "0") : String(m).padStart(2, "0")) + ":" + String(s).padStart(2, "0"); };
  const T = { key: null, acc: 0, since: null, paused: false, iv: 0 };
  const elapsed = () => (T.acc + (T.since ? Date.now() - T.since : 0)) / 1000;
  function timerStart(key) {
    clearInterval(T.iv);
    Object.assign(T, { key, acc: 0, since: document.visibilityState === "visible" ? Date.now() : null, paused: false });
    T.iv = setInterval(timerDraw, 1000);
    timerDraw();
  }
  function timerStop() { clearInterval(T.iv); T.key = null; }
  function timerDraw() {
    const el = document.getElementById("ncTimer");
    if (!el) return;
    el.querySelector(".t").textContent = mmss(elapsed());
    el.classList.toggle("paused", T.paused);
    setSwap(el.querySelector('[data-t="pause"]'), T.paused ? 1 : 0);
  }
  document.addEventListener("visibilitychange", () => {
    if (!T.key || T.paused) return;
    if (document.visibilityState === "hidden") { T.acc += T.since ? Date.now() - T.since : 0; T.since = null; }
    else T.since = Date.now();
  });
  function timerHTML(it) {
    const extra = it.best ? `<span class="best" title="Last: ${mmss(it.last || it.best)}">best ${mmss(it.best)}</span>` : "";
    return `<span class="nc-timer" id="ncTimer" title="Time on this problem (pauses when you switch tabs)">${icon("timer")}&nbsp;<span class="t">00:00</span>
      <button class="btn sm" data-t="pause" title="Pause / resume">${swapLabel([icon("pause"), icon("play")])}</button><button class="btn sm" data-t="reset" title="Restart from 0">${icon("restart")}</button>${extra}</span>`;
  }
  main.addEventListener("click", (e) => {
    const b = e.target.closest("[data-t]");
    if (!b || !T.key) return;
    if (b.dataset.t === "pause") {
      if (T.paused) { T.paused = false; T.since = Date.now(); }
      else { T.acc += T.since ? Date.now() - T.since : 0; T.since = null; T.paused = true; }
    }
    if (b.dataset.t === "reset") { T.acc = 0; T.since = T.paused ? null : Date.now(); }
    timerDraw();
  });
  // solved → save the time
  document.addEventListener("item:status", (e) => {
    const { key, was, status } = e.detail;
    if (key !== T.key || status !== "done" || was) return;
    const sec = Math.round(elapsed());
    if (sec < 5) return;
    const it = Store.item(key);
    const best = it.best ? Math.min(it.best, sec) : sec;
    Store.patch(key, { last: sec, best });
    T.paused = true; T.acc += T.since ? Date.now() - T.since : 0; T.since = null; timerDraw();
    setTimeout(() => toast(`Solved in ${mmss(sec)}${it.best && sec < it.best ? " · new best!" : ""}`, 2600, it.best && sec < it.best ? "medal" : "timer"), 1900);
  });

  /* ---------------- Detail view ---------------- */
  // a paragraph that is only a $formula$ is shown as a centred display line; lines starting "1." or "- " become a list
  const paras = (t) => String(t || "").split(/\n\s*\n/).map((x) => {
    x = x.trim();
    const L = x.split("\n");
    if (/^\$[^$]+\$$/.test(x)) return `<div class="math-d">${tex(x.slice(1, -1), true)}</div>`;
    if (L.every((l) => /^\d+\.\s/.test(l))) return `<ol class="clean">${L.map((l) => `<li>${fmt(l.replace(/^\d+\.\s/, ""))}</li>`).join("")}</ol>`;
    if (L.every((l) => /^-\s/.test(l))) return `<ul class="clean">${L.map((l) => `<li>${fmt(l.slice(2))}</li>`).join("")}</ul>`;
    return `<p>${fmt(x)}</p>`;
  }).join("");
  function cx(o) {
    if (!o || (!o.tc && !o.sc)) return "";
    return `<div class="table-wrap measure"><table><thead><tr><th></th><th>Complexity</th><th>Why</th></tr></thead><tbody>
      <tr><td><strong>Time</strong></td><td><code>${esc(o.tc || "")}</code></td><td>${fmt(o.tw || "")}</td></tr>
      <tr><td><strong>Space</strong></td><td><code>${esc(o.sc || "")}</code></td><td>${fmt(o.sw || "")}</td></tr></tbody></table></div>`;
  }
  function dryRun(dry) {
    if (!dry) return "";
    if (dry.steps) {
      return `${dry.setup ? `<div class="callout"><span class="label">What we keep track of</span>${fmt(dry.setup)}</div>` : ""}
        <ol class="dry-steps">${dry.steps.map(([state, why]) => `<li><div class="dry-state">${fmt(state)}</div>${why ? `<div class="dry-why">↳ ${fmt(why)}</div>` : ""}</li>`).join("")}</ol>
        ${dry.result ? `<div class="callout tip"><span class="label">Result</span>${fmt(dry.result)}</div>` : ""}
        <p class="small muted">Want every single step with live variables? Press <strong>${icon("play")} Step through</strong> on the optimal code above.</p>`;
    }
    if (Array.isArray(dry)) return `<ol class="steps">${dry.map((x) => `<li>${fmt(x)}</li>`).join("")}</ol>`;
    return `<div class="table-wrap"><table><thead><tr>${dry.h.map((x) => `<th>${fmt(x)}</th>`).join("")}</tr></thead><tbody>${dry.r.map((r) => `<tr>${r.map((c) => `<td>${fmt(String(c))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  function explain(ex) {
    if (!ex) return "";
    return `<div class="table-wrap"><table><thead><tr><th class="col-42p">Code</th><th>What it does &amp; why</th></tr></thead><tbody>${ex.map(([c, t]) => `<tr><td><pre class="code bare"><code>${hlPy(c)}</code></pre></td><td>${fmt(t)}</td></tr>`).join("")}</tbody></table></div>`;
  }
  function helpers(p) {
    if (!p.h) return "";
    const H = window.PRAC_HELPERS || {};
    return `<details class="box"><summary>${icon("puzzle")} Helper code used by this example (${esc(p.h)})</summary><div class="body"><p class="small muted">Run this first if you copy the solution to your computer.</p>${codeBlock(H[p.h] || "")}</div></details>`;
  }
  const join = (code, run) => (run ? code.replace(/\s+$/, "") + "\n\n\n" + run : code);

  function detail(id) {
    const i = list.findIndex((p) => p.id === id);
    if (i < 0) { location.hash = ""; return; }
    const p = list[i], key = "prac:" + p.id, prev = list[i - 1], next = list[i + 1], pm = practice();
    document.title = p.t + " · Practice";
    main.classList.add("reading");
    const hide = pm ? "" : " open";
    let h = `<div class="btn-row justify-between"><a class="btn sm" href="#">← All problems</a>
        <label class="small muted row pointer"><input type="checkbox" id="pm" ${pm ? "checked" : ""}> Practice mode (hide solutions until I open them)</label></div>
      <div class="crumbs mt-4">${esc(p.cat)} · #${i + 1} of ${list.length}</div>
      <h1>${esc(p.t)}</h1>
      <div class="btn-row"><span class="badge ${p.d.toLowerCase()}">${p.d}</span><span class="badge blue">${esc(p.p || "")}</span><span class="flex-1"></span>${timerHTML(Store.item(key))}</div>
      ${actionBar(key, { doneLabel: "Mark as solved", doneOn: "Solved" })}

      <h2>${icon("doc")} Problem</h2>${paras(p.q)}
      ${(p.ex || []).map((e, k) => `<div class="card ex-card"><div class="small muted fw-700">Example ${k + 1}</div><div><strong>Input:</strong> <code>${esc(e[0])}</code></div><div><strong>Output:</strong> <code>${esc(e[1])}</code></div>${e[2] ? `<div class="small"><strong>Why:</strong> ${fmt(e[2])}</div>` : ""}</div>`).join("")}
      ${p.c ? `<p class="small muted"><strong>Constraints:</strong> ${p.c.map(fmt).join(" · ")}</p>` : ""}
      ${p.hint ? `<details class="box"><summary>${icon("bulb")} Hint</summary><div class="body">${paras(p.hint)}</div></details>` : ""}
      ${window.PyUI && p.run ? `<details class="box solve-box" open><summary>${icon("flask")} Try it yourself: write your solution and run the tests</summary><div class="body"><div class="solve"></div></div></details>` : ""}`;

    const bRun = typeof p.brute?.run === "string", bNo = p.brute?.run === false;
    if (p.brute) h += `<details class="box"${hide}><summary>${icon("walk")} Approach 1 — ${esc(p.brute.name || "Plain Python loops")}</summary><div class="body">${paras(p.brute.idea)}${p.brute.code ? codeBlock(join(p.brute.code, bNo ? "" : bRun ? p.brute.run : p.run), bNo ? "" : bRun ? p.brute.out : p.out) : ""}${cx(p.brute)}${p.brute.bad ? `<div class="callout warn"><span class="label">Why we vectorize</span>${fmt(p.brute.bad)}</div>` : ""}</div></details>`;
    if (p.better) h += `<details class="box"${hide}><summary>${icon("walk")} Approach 2 — ${esc(p.better.name || "Better")}</summary><div class="body">${paras(p.better.idea)}${p.better.code ? codeBlock(join(p.better.code, p.run), p.out) : ""}${cx(p.better)}</div></details>`;
    const o = p.opt;
    h += `<details class="box"${hide}><summary>${icon("rocket")} ${p.brute ? (p.better ? "Approach 3" : "Approach 2") + " — " : ""}${esc(o.name || (p.brute ? "Vectorized with NumPy" : "Solution"))}</summary><div class="body">
        <h3 class="mt-2">Intuition</h3>${paras(o.idea)}
        ${o.steps ? `<h3>Step-by-step</h3><ol class="steps">${o.steps.map((x) => `<li>${fmt(x)}</li>`).join("")}</ol>` : ""}
        ${o.viz ? `<div class="viz" data-viz="${o.viz}"></div>` : ""}
        <h3>Code</h3>${helpers(p)}${codeBlock(join(o.code, p.run), p.out)}
        <h3>Complexity</h3>${cx(o)}
      </div></details>`;
    if (p.dry) h += `<details class="box"${hide}><summary>${icon("search")} Dry run${p.dryOn ? ` — ${esc(p.dryOn)}` : ""}</summary><div class="body">${dryRun(p.dry)}</div></details>`;
    if (p.expl) h += `<details class="box"${hide}><summary>${icon("brain")} Code explained line by line</summary><div class="body">${explain(p.expl)}</div></details>`;
    if (p.alt) h += `<details class="box"><summary>${icon("plus")} ${esc(p.alt.name || "Alternative solution")}</summary><div class="body">${paras(p.alt.idea)}${p.alt.code ? codeBlock(p.alt.code) : ""}${cx(p.alt)}</div></details>`;
    if (p.take) h += `<div class="callout tip"><span class="label">Key takeaway</span>${fmt(p.take)}</div>`;
    if (p.mist) h += `<div class="callout danger"><span class="label">Common mistakes</span><ul class="clean">${p.mist.map((m) => `<li>${fmt(m)}</li>`).join("")}</ul></div>`;
    h += `<hr>${notesBox(key)}
      <div class="btn-row page-nav">
        ${prev ? `<a class="btn" href="#p/${prev.id}">← ${esc(prev.t)}</a>` : "<span></span>"}
        ${next ? `<a class="btn primary" href="#p/${next.id}">${esc(next.t)} →</a>` : ""}
      </div>`;
    main.innerHTML = h;
    main.querySelector(".doc-actions").dataset.opts = JSON.stringify({ doneLabel: "Mark as solved", doneOn: "Solved" });
    if (T.key !== key) timerStart(key); else timerDraw();
    $("#pm").onchange = (e) => { try { localStorage.setItem(lsKey("prac:practice"), e.target.checked ? "1" : "0"); } catch (x) {} detail(id); };
    Viz.mountAll(main);
    if (window.glossify) glossify(main);
    if (p.h) main.querySelectorAll(".code-wrap").forEach((w) => (w.dataset.pre = p.h));     // examples that need helper code
    const sv = main.querySelector(".solve");
    if (sv) PyUI.solve(sv, p, () => { const b = main.querySelector(".doc-actions [data-act=done]"); if (b) b.click(); });
    window.scrollTo(0, 0);
  }

  function route() {
    const h = decodeURIComponent(location.hash.slice(1));
    if (h.startsWith("p/")) { leaveRoadmap(); curView = "detail"; detail(h.slice(2)); return; }
    timerStop();
    let view = h === "roadmap" ? "roadmap" : h === "list" ? "list" : null;
    if (view) { try { localStorage.setItem(lsKey("prac:view"), view); } catch (e) {} }
    else { try { view = localStorage.getItem(lsKey("prac:view")) || "list"; } catch (e) { view = "list"; } }   // no hash: the view you used last
    curView = view;
    if (view === "roadmap") roadmapView(); else { leaveRoadmap(); listView(); }
  }
  // never leave a blank page: if a view fails, say so (with the reason) and offer a reload
  function safeRoute() {
    try { route(); }
    catch (err) {
      console.error(err);
      leaveRoadmap();
      main.innerHTML = `<div class="card empty-state"><div class="big">${icon("puzzle")}</div><h2 class="dash-h2">This page didn't load properly</h2>
        <p class="muted">Usually an old cached file. Press <kbd>Ctrl</kbd> + <kbd>F5</kbd> to reload everything.</p>
        <p class="small muted">Details: ${esc(String(err && err.message || err))}</p>
        <div class="btn-row" style="justify-content:center"><a class="btn primary" href="#list" onclick="setTimeout(()=>location.reload(),0)">Reload the list</a></div></div>`;
    }
  }
  window.addEventListener("hashchange", safeRoute);
  safeRoute();
})();
