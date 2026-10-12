/* search.js: highlight any text on the page → a small "🔍 Google it" pill appears next to it.
     • click it: Google searches the highlighted text (new tab)
     • or just start typing: the pill turns into a text box for your question; Enter searches the highlighted
       text together with your question, e.g. "heapq.heapify time complexity".
       Backspace on an empty box turns it back into the pill, Esc closes it.                                  */
(function () {
  const style = document.createElement("style");
  style.textContent = `
  .gsearch{position:absolute;z-index:150;display:flex;align-items:center;margin:0;padding:.3rem;border:1px solid var(--border);
    border-radius:var(--r-pill);background:var(--panel);color:var(--text);font:600 var(--fs-lg) var(--sans);
    box-shadow:0 8px 24px rgba(0,0,0,.2);white-space:nowrap;max-width:calc(100vw - 16px);transition:border-color .25s}
  .gsearch[hidden]{display:none}
  .gsearch button{display:inline-flex;align-items:center;gap:.45rem;border:0;cursor:pointer;font:inherit;color:inherit;background:none}
  .gs-go{padding:.45rem .9rem;border-radius:var(--r-pill)}
  .gsearch:not(.asking) .gs-go:hover{color:var(--accent);background:color-mix(in srgb,var(--accent) 12%,var(--panel))}
  .gsearch:not(.asking) .gs-in,.gsearch:not(.asking) .gs-send{display:none}
  .gsearch .gs-in,.gsearch .gs-in:focus,.gsearch .gs-in:focus-visible{border:0;outline:0;box-shadow:none;background:none;border-radius:0}
  .gsearch.asking{border-color:var(--accent)}
  .gsearch.asking .gs-go{padding:.45rem .2rem .45rem .7rem;color:var(--accent);cursor:default}
  .gsearch.asking .gs-label{display:none}
  .gsearch.asking .gs-in{display:block;width:min(300px,60vw);padding:.45rem .4rem;border:0;outline:0;background:none;color:var(--text);font:500 var(--fs-lg) var(--sans)}
  .gsearch.asking .gs-send{display:inline-flex;padding:.45rem .7rem;border-radius:var(--r-pill);color:var(--accent)}
  .gsearch.asking .gs-send:hover{background:color-mix(in srgb,var(--accent) 12%,var(--panel))}
  :root[data-anim="on"] .gsearch.asking .gs-in,:root[data-anim="on"] .gsearch.asking .gs-send{animation:gs-in .25s cubic-bezier(.22,1,.36,1)}
  @keyframes gs-in{from{opacity:0;transform:translateX(-6px)}}
  .gsearch svg{width:18px;height:18px;flex:none}`;
  document.head.appendChild(style);

  const LENS = `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" fill="none" stroke="currentColor" stroke-width="2.4"/><path d="M15.5 15.5 21 21" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>`;
  const ARROW = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const bar = document.createElement("form");
  bar.className = "gsearch";
  bar.hidden = true;
  bar.innerHTML = `<button type="button" class="gs-go">${LENS}<span class="gs-label">Google it</span></button>
    <input class="gs-in" type="text" autocomplete="off" spellcheck="false" aria-label="Ask a question about the highlighted text">
    <button type="submit" class="gs-send" title="Search Google for the highlighted text + your question" aria-label="Search">${ARROW}</button>`;
  document.body.appendChild(bar);
  const go = bar.querySelector(".gs-go"), input = bar.querySelector(".gs-in");

  let text = "", rect = null;
  const MAX = 300;                                  // ignore huge selections
  const short = (t, n) => (t.length > n ? t.slice(0, n - 1) + "…" : t);
  const inBar = (el) => el instanceof Element && !!el.closest(".gsearch");
  const asking = () => bar.classList.contains("asking");

  function currentSelection() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
    const t = sel.toString().replace(/\s+/g, " ").trim();
    if (!t || t.length > MAX) return null;
    const node = sel.anchorNode && (sel.anchorNode.nodeType === 1 ? sel.anchorNode : sel.anchorNode.parentElement);
    if (node && node.closest("input, textarea, .gsearch, .ed")) return null;   // not inside boxes or the code editor
    return { t, rect: sel.getRangeAt(0).getBoundingClientRect() };
  }

  // centred above the highlighted text (below it if there's no room), kept inside the window
  function place() {
    if (!rect) return;
    const w = bar.offsetWidth, h = bar.offsetHeight;
    let left = rect.left + rect.width / 2 - w / 2 + rect.sx;
    let top = rect.top + rect.sy - h - 10;
    if (rect.top < h + 14) top = rect.bottom + rect.sy + 10;
    left = Math.max(window.scrollX + 8, Math.min(left, window.scrollX + document.documentElement.clientWidth - w - 8));
    bar.style.left = left + "px";
    bar.style.top = top + "px";
  }
  function show() {
    if (asking()) return;
    const s = currentSelection();
    if (!s || (!s.rect.width && !s.rect.height)) { hide(); return; }
    text = s.t;
    rect = { left: s.rect.left, top: s.rect.top, bottom: s.rect.bottom, width: s.rect.width, sx: window.scrollX, sy: window.scrollY };
    go.title = `Search Google for “${short(text, 60)}” · or start typing to ask a question about it`;
    input.placeholder = `Ask about “${short(text, 22)}”…`;
    bar.hidden = false;
    place();
  }
  function ask(on) {                                            // the pill ↔ the question box
    bar.classList.toggle("asking", on);
    input.value = "";
    place();
    if (on) input.focus();
  }
  function hide() { if (!bar.hidden) { bar.hidden = true; bar.classList.remove("asking"); input.value = ""; } }

  function search(q) {
    if (!q) return;
    window.open("https://www.google.com/search?q=" + encodeURIComponent(q), "_blank", "noopener");
    hide();
    const sel = window.getSelection();
    if (sel) sel.removeAllRanges();
  }

  // show after the user finishes selecting (mouse or touch / keyboard selection)
  document.addEventListener("mouseup", (e) => { if (!inBar(e.target)) setTimeout(show, 10); });
  document.addEventListener("keyup", (e) => { if (!inBar(e.target) && (e.shiftKey || e.key === "Shift")) setTimeout(show, 10); });
  document.addEventListener("selectionchange", () => {
    if (asking()) return;                                       // typing a question: keep the box
    if (!currentSelection()) hide();
    else if (matchMedia("(pointer: coarse)").matches) setTimeout(show, 300);   // phones/tablets
  });
  document.addEventListener("mousedown", (e) => { if (!inBar(e.target)) hide(); });
  window.addEventListener("resize", hide);

  // with the pill showing, just start typing: it becomes the question box and the key lands in it
  // (and doesn't trigger the page's own shortcuts like S or P)
  window.addEventListener("keydown", (e) => {
    if (bar.hidden) return;
    if (e.key === "Escape") { hide(); return; }
    if (asking()) return;
    const t = e.target;
    if (e.ctrlKey || e.metaKey || e.altKey || e.key.length !== 1 || (t.closest && t.closest("input, textarea, select, [contenteditable]"))) return;
    e.stopPropagation();
    ask(true);
  }, true);

  go.addEventListener("mousedown", (e) => e.preventDefault());   // keep the selection while clicking
  go.addEventListener("click", () => { if (asking()) input.focus(); else search(text); });
  bar.addEventListener("submit", (e) => { e.preventDefault(); search((text + " " + input.value.trim()).trim()); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); hide(); }
    else if (e.key === "Backspace" && !input.value) { e.preventDefault(); ask(false); }   // empty box: back to the pill
  });
})();
