/* revision.js — spaced-repetition review, bookmarks, completed items, notes */
(function () {
  const main = $("#content");
  let tab = "due", typeF = "";
  try { tab = sessionStorage.getItem(lsKey("rev:tab")) || "due"; } catch (e) {}

  function rows() {
    const idx = siteIndex(), items = Store.load().items;
    return Object.entries(items).filter(([k]) => idx[k]).map(([k, v]) => Object.assign({ key: k, it: v }, idx[k]));
  }
  const TABS = [
    ["due", "Due now", (r) => Store.isDue(r.it)],
    ["revise", "Needs revision", (r) => r.it.status === "revise"],
    ["star", "Bookmarked", (r) => r.it.star],
    ["done", "Completed", (r) => r.it.status],
    ["notes", "My notes", (r) => r.it.note && r.it.note.trim()],
  ];
  const ago = (t) => { if (!t) return "—"; const d = Math.round((new Date(today()) - new Date(today(t))) / DAY); return d <= 0 ? "today" : d === 1 ? "yesterday" : d + " days ago"; };
  const level = (l) => `<span title="Memory strength: ${l || 0}/5" class="level">${"●".repeat(l || 0)}<span class="off">${"●".repeat(5 - (l || 0))}</span></span>`;

  function render() {
    const all = rows();
    const counts = Object.fromEntries(TABS.map(([id, , f]) => [id, all.filter(f).length]));
    const [, label, f] = TABS.find((t) => t[0] === tab) || TABS[0];
    let list = all.filter(f).filter((r) => !typeF || r.type === typeF);
    list.sort((a, b) => tab === "due" || tab === "revise" ? (a.it.next || 0) - (b.it.next || 0) : (b.it.updated || 0) - (a.it.updated || 0));
    const upcoming = all.filter((r) => r.it.status && !Store.isDue(r.it) && r.it.next).sort((a, b) => a.it.next - b.it.next).slice(0, 6);
    const pool = all.filter((r) => r.it.status && r.type === "Problem");

    let h = `<h1>Revision</h1>
      <p class="lead">Anything you mark as solved or learned comes back for review after 1 → 3 → 7 → 14 → 30 → 60 days. If you remember it, the gap grows. If you forget, it starts again at 1 day.</p>
      <div class="btn-row my-4">${TABS.map(([id, l]) => `<button class="btn ${tab === id ? "primary" : ""}" data-tab="${id}">${l} <span class="badge" style="${tab === id ? "background:rgba(255,255,255,.25);color:#fff" : ""}">${counts[id]}</span></button>`).join("")}
        <select id="typeF" class="ml-auto"><option value="">All types</option>${["Problem", ...SITE.docs.map((d) => d.type)].map((t) => `<option ${typeF === t ? "selected" : ""}>${t}</option>`).join("")}</select></div>`;

    if (tab === "due") h += `<div class="callout tip"><span class="label">How to review</span>Open the item and re-solve it (or re-explain it) <strong>from memory</strong> before you look at the solution. Then press <strong>Got it</strong> if you remembered, or <strong>Forgot</strong> if you struggled.</div>`;

    if (!list.length) h += `<div class="card empty-state"><div class="big">${icon(tab === "due" ? "party" : "inbox")}</div><p class="muted">${tab === "due" ? "Nothing due right now. Solve new problems, or review something early from <em>Completed</em>." : "Nothing here yet."}</p></div>`;
    else if (tab === "notes") {
      h += list.map((r) => `<div class="card note-card"><div class="btn-row"><span class="badge blue">${r.type}</span><a href="${r.href}"><strong>${esc(r.title)}</strong></a><span class="small muted">${esc(r.cat || "")}</span></div><div class="pre-wrap mt-2">${esc(r.it.note)}</div></div>`).join("");
    } else {
      h += `<div class="table-wrap"><table><thead><tr><th>Item</th><th>Type</th><th>Last done</th><th>Next review</th><th>Strength</th><th>${tab === "due" || tab === "revise" ? "Did you remember?" : ""}</th></tr></thead><tbody>`;
      h += list.map((r) => `<tr>
          <td><a href="${r.href}"><strong>${esc(r.title)}</strong></a>${r.it.star ? ` <span class="text-amber">${icon("star-fill")}</span>` : ""}<div class="small muted">${esc(r.cat || "")}${r.d ? ` · <span class="badge ${r.d.toLowerCase()}">${r.d}</span>` : ""}</div></td>
          <td><span class="badge blue">${r.type}</span></td>
          <td class="small">${ago(r.it.lastReview || r.it.doneAt)}</td>
          <td class="small">${r.it.status ? (Store.isDue(r.it) ? '<span class="badge purple">due now</span>' : relDay(r.it.next)) : "—"}</td>
          <td>${r.it.status ? level(r.it.level) : ""}</td>
          <td><div class="btn-row">${r.it.status ? `<button class="btn sm on-green" data-rev="${r.key}" data-ok="1" title="Remembered — push next review further">${icon("check")} Got it</button><button class="btn sm on-red" data-rev="${r.key}" data-ok="0" title="Forgot — review again tomorrow">${icon("x")} Forgot</button>` : ""}</div></td></tr>`).join("");
      h += `</tbody></table></div>`;
    }

    h += `<div class="grid cols-2 mt-8">
      <div class="card"><h3>${icon("dice")} Random recall</h3><p class="small muted">Pick a random solved problem. Can you explain the idea and write the key lines in 2 minutes?</p><div id="rand"></div><button class="btn primary" id="randBtn" ${pool.length ? "" : "disabled"}>Give me one</button></div>
      <div class="card"><h3>${icon("calendar")} Coming up</h3>${upcoming.length ? `<ul class="clean">${upcoming.map((r) => `<li><a href="${r.href}">${esc(r.title)}</a> <span class="small muted">— ${relDay(r.it.next)}</span></li>`).join("")}</ul>` : '<p class="small muted">Nothing scheduled yet.</p>'}</div></div>`;
    main.innerHTML = h;

    main.querySelectorAll("[data-tab]").forEach((b) => (b.onclick = () => { tab = b.dataset.tab; try { sessionStorage.setItem(lsKey("rev:tab"), tab); } catch (e) {} render(); }));
    $("#typeF").onchange = (e) => { typeF = e.target.value; render(); };
    main.querySelectorAll("[data-rev]").forEach((b) => (b.onclick = () => { Store.review(b.dataset.rev, b.dataset.ok === "1"); toast(b.dataset.ok === "1" ? "Great — next review pushed further out" : "No worries — it'll come back tomorrow"); render(); }));
    const rb = $("#randBtn");
    if (rb) rb.onclick = () => {
      const r = pool[Math.floor(Math.random() * pool.length)], p = r.p;
      $("#rand").innerHTML = `<p><a href="${r.href}"><strong>${esc(r.title)}</strong></a> <span class="badge ${p.d.toLowerCase()}">${p.d}</span></p><details class="box"><summary>Reveal the key idea</summary><div class="body"><p><span class="badge blue">${esc(p.p || "")}</span></p><p class="small">${fmt(p.take || p.hint || "")}</p></div></details>`;
    };
  }
  render();
})();
