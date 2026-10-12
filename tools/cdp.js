/* Browser test driver: opens the site as files (file://) in headless Chrome and runs a steps file against it.
   Usage:  node tools/cdp.js path/to/my-steps.js        (screenshots are saved next to the steps file)
   Steps file:  module.exports = async ({ go, ev, shot, send, sleep, log }) => { await go("index.html"); log(await ev("document.title")); await shot("home"); };
   go(page) loads a page · ev(js) runs JavaScript there and returns the value · shot(name) saves name.png
   send(method, params) is a raw DevTools call (e.g. Input.dispatchKeyEvent for real typing). Needs Chrome + Node 22+. */
const { spawn } = require("child_process"), fs = require("fs"), path = require("path"), os = require("os");
const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const SITE = "file:///" + path.join(__dirname, "..").split(path.sep).join("/") + "/";
const stepsFile = path.resolve(process.argv[2] || "steps.js"), OUT = path.dirname(stepsFile);
const prof = fs.mkdtempSync(path.join(os.tmpdir(), "cdp-"));
const PORT = 9300 + Math.floor(Math.random() * 600);                  // its own port: several tests can run at once
const ch = spawn(CHROME, ["--headless=new", "--disable-gpu", "--remote-debugging-port=" + PORT, "--user-data-dir=" + prof, "--window-size=1400,900", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  let tabs;
  for (let i = 0; i < 50; i++) { try { tabs = await (await fetch("http://127.0.0.1:" + PORT + "/json")).json(); if (tabs.length) break; } catch (e) {} await sleep(200); }
  const ws = new WebSocket(tabs.find((t) => t.type === "page").webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0; const wait = {}, logs = [];
  ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && wait[d.id]) { wait[d.id](d); delete wait[d.id]; }
    if (d.method === "Runtime.exceptionThrown") logs.push("EXC " + JSON.stringify(d.params.exceptionDetails.exception && d.params.exceptionDetails.exception.description || d.params.exceptionDetails.text).slice(0, 300));
    if (d.method === "Runtime.consoleAPICalled" && d.params.type === "error") logs.push("ERR " + d.params.args.map((a) => a.value || a.description).join(" ").slice(0, 300)); };
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; wait[i] = r; ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async (expr) => { const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); return r.result.exceptionDetails ? "EXC " + JSON.stringify(r.result.exceptionDetails.exception.description) : r.result.result.value; };
  const go = async (u) => { await send("Page.navigate", { url: SITE + u }); await sleep(1800); };
  const shot = async (name) => { const r = await send("Page.captureScreenshot", { format: "png" }); fs.writeFileSync(path.join(OUT, name + ".png"), Buffer.from(r.result.data, "base64")); };
  await send("Page.enable"); await send("Runtime.enable");
  const steps = require(stepsFile);
  await steps({ go, ev, shot, send, sleep, log: (...a) => console.log(...a), logs });
  console.log(logs.length ? "PAGE ERRORS:\n" + logs.join("\n") : "no page errors");
  ws.close(); ch.kill();
  process.exit(0);
})().catch((e) => { console.error(e); ch.kill(); process.exit(1); });
