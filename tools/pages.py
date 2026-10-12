"""Write every page (index.html, math.html, ...) from one template, so they all load the same scripts in the same order.
The doc pages come from SITE.docs in assets/site.js; each loads data/briefs-<kind>.js (if any) and data/<kind>.js
(or data/<kind>-1.js, data/<kind>-2.js ...).
Usage:  python tools/pages.py      then  node tools/stamp.js  (fresh version stamps)"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SITE_JS = open(os.path.join(ROOT, "assets", "site.js"), encoding="utf-8").read()
DOCS = re.findall(r'\{ kind: "(\w+)", list: "(\w+)", href: "([\w.-]+)", title: "([^"]+)"', SITE_JS)
PRAC = sorted(f for f in os.listdir(os.path.join(DATA, "prac")) if f[0].isdigit()) if os.path.isdir(os.path.join(DATA, "prac")) else []
BRAND = "AI Study Hub"

HEAD = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800&display=optional">
<link rel="stylesheet" href="assets/style.css">
<link rel="stylesheet" href="assets/fx/fx.css">
{extra_css}<script>try{{var h=document.documentElement,t=localStorage.getItem('theme');if(t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches)h.dataset.theme='dark';var a=localStorage.getItem('anim');h.dataset.anim=a==='on'||(a!=='off'&&!matchMedia('(prefers-reduced-motion: reduce)').matches)?'on':'off'}}catch(e){{}}</script>
</head>
<body>
<header id="topbar"></header>
'''
KATEX_CSS = '<link rel="stylesheet" href="assets/vendor/katex/katex.min.css">\n'
KATEX_JS = "assets/vendor/katex/katex.min.js"
PAGE = '<main class="page" id="content"></main>\n'
DOC = '<div class="layout">\n  <aside class="sidebar" id="sidebar"></aside>\n  <main class="content" id="content"></main>\n</div>\n'
TAIL = ["assets/fx/fx-core.js", "assets/fx/fx-page.js", "assets/fx/fx-header.js", "assets/fx/fx-sidebar.js",
        "assets/fx/fx-cursor.js", "assets/fx/fx-celebrate.js", "data/progress.js", "assets/find.js", "assets/search.js"]
# assets/viz/*.js: each part of the course adds its own visualizers, diagrams and glossary terms (loaded on every doc page + Practice)
VIZ_EXT = sorted(f for f in os.listdir(os.path.join(ROOT, "assets", "viz")) if f.endswith(".js")) if os.path.isdir(os.path.join(ROOT, "assets", "viz")) else []
CODE = ["assets/glossary.js", "assets/viz.js"] + ["assets/viz/" + f for f in VIZ_EXT]


def page(name, title, body, before, after, init, maths=False):
    # before: loaded ahead of common.js (KaTeX, outmap), after: data files, init: inline script
    scripts = ["assets/site.js"] + ([KATEX_JS] if maths else []) + before + ["assets/common.js"] + after
    h = HEAD.format(title=title, extra_css=KATEX_CSS if maths else "") + body
    h += "".join(f'<script src="{s}"></script>\n' for s in scripts)
    h += f"<script>\n  {init}\n</script>\n" if "\n" in init else f"<script>{init}</script>\n"
    extra = [s for s in TAIL if s not in scripts]
    if name == "practice.html": extra.insert(0, "assets/practice.js")
    if name == "revision.html": extra.insert(0, "assets/revision.js")
    if name == "index.html": extra.insert(0, "assets/dashboard.js")
    extra.insert(0, "assets/py.js")                    # before the page's own script: Practice uses it when it first draws
    h += "".join(f'<script src="{s}"></script>\n' for s in extra)
    h += "</body>\n</html>\n"
    open(os.path.join(ROOT, name), "w", encoding="utf-8", newline="\n").write(h)


def doc_files(kind):
    files = sorted(f for f in os.listdir(DATA) if re.fullmatch(kind + r"(-\d+)?\.js", f))
    brief = sorted(f for f in os.listdir(DATA) if re.fullmatch("briefs-" + kind + r"(-\d+)?\.js", f))
    return ["data/" + f for f in brief + files]


page("index.html", f"Dashboard · {BRAND}", PAGE, [], ["data/progress.js"], 'topbar("index.html");')
for kind, lst, href, title in DOCS:
    page(href, f"{title} · {BRAND}", DOC, ["data/outmap.js"], CODE + doc_files(kind),
         f'topbar("{href}");\n  docPage("{kind}");', maths=True)
page("practice.html", f"Practice · {BRAND}", PAGE, ["data/outmap.js"], CODE + ["data/prac/helpers.js"] + ["data/prac/" + f for f in PRAC],
     'topbar("practice.html");', maths=True)
page("revision.html", f"Revision · {BRAND}", PAGE, [], ["data/index.js"], 'topbar("revision.html");')
print(f"pages written: index, {', '.join(h for _, _, h, _ in DOCS)}, practice, revision")
