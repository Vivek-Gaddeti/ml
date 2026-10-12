"""Run one code sample under a tracer.

Usage:  python trace.py job.json result.json
job    = {"code": str, "pre": str (helper code run first, not traced),
          optional: "max": step limit (default 400), "stdin": text for input()}
Also runs inside the browser (assets/py.js loads it from data/trace-src.js, written by tools/lite.js)
for stepping through your own code.
result = {"ok": bool, "out": full stdout,
          "outmap": [[insert_after_line, text], ...],   # which line produced each output
          "steps": [...], "truncated": bool, "max": step limit, "err": "Type: message" or "",
          "imgs": [base64 PNG, ...]}                    # every matplotlib figure the code drew
NumPy arrays show their shape and dtype first, e.g.  (3, 4) float64 [[0.1 0.2 … ]]  so you can follow shapes step by step.
Each step: {"l": line, "e": "l"|"r" (line/return), "s": call stack, "o": output length so far,
            "v": locals (functions only), "g": globals (only when changed), "r": return value}
"""
import ast
import base64
import builtins
import io
import json
import os
import re
import sys
import types
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")          # figures are drawn off-screen and returned as PNG pictures
warnings.filterwarnings("ignore", message=".*non-interactive.*")

MAX_STEPS = 400
SAMPLE = "<sample>"
ADDR = re.compile(r" at 0x[0-9A-Fa-f]+")


def array_repr(v):
    """NumPy array → '(2, 3) float64 [[1. 2. 3.] [4. 5. 6.]]' (shape first, values on one line)."""
    np = sys.modules["numpy"]
    body = np.array2string(v, precision=3, suppress_small=True, threshold=24, edgeitems=2, max_line_width=10_000)
    return f"{v.shape} {v.dtype} " + " ".join(body.split())


def short(v, limit=70):
    """A readable, size-limited repr (with friendly forms for NumPy, pandas, ListNode / TreeNode)."""
    mod = type(v).__module__ or ""
    try:
        if mod == "numpy" and type(v).__name__ == "ndarray":
            s = array_repr(v)
            return s if len(s) <= 110 else s[:109] + "…"
        if mod == "numpy" and hasattr(v, "item"):          # np.float64(0.5) → 0.5
            return short(v.item(), limit)
        if mod.startswith("pandas") and type(v).__name__ == "DataFrame":
            return f"DataFrame {v.shape[0]}×{v.shape[1]} columns {list(v.columns)[:6]}"
        if mod.startswith("pandas") and type(v).__name__ == "Series":
            return f"Series len {len(v)}" + (f" '{v.name}'" if v.name is not None else "")
    except Exception:
        pass
    try:
        if not isinstance(v, type) and hasattr(v, "val") and hasattr(v, "next"):
            parts, n, k = [], v, 0
            while n is not None and k < 8:
                parts.append(repr(getattr(n, "val", "?")))
                n = getattr(n, "next", None)
                k += 1
            return "list " + "→".join(parts) + ("→…" if n is not None else "")
        if not isinstance(v, type) and hasattr(v, "val") and hasattr(v, "left") and hasattr(v, "right"):
            return f"TreeNode({v.val!r})"
        if isinstance(v, types.GeneratorType):
            return "<generator>"
        s = repr(v)
    except Exception:
        s = "<?>"
    s = ADDR.sub("", s)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def interesting(name, value):
    if name.startswith("__"):
        return False
    return not isinstance(value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType,
                                  type, types.MethodType))


def collect(d, skip=()):
    out = {}
    for k, v in list(d.items()):
        if k in skip or k == "self" or not interesting(k, v):
            continue
        r = short(v)
        if r.startswith("<__main__.") and r.endswith(" object>"):
            continue                 # e.g. the Solution() instance: no useful information
        out[k] = r
        if len(out) >= 20:
            break
    return out


def statement_ends(code):
    """Map each line to the last line of the smallest simple statement containing it,
    so multi-line statements aren't split when we insert their output."""
    ends = {}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ends
    compound = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try,
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Match)
    if hasattr(ast, "TryStar"):
        compound = compound + (ast.TryStar,)
    best = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(node, compound):
            span = node.end_lineno - node.lineno
            for ln in range(node.lineno, node.end_lineno + 1):
                if ln not in best or span < best[ln][0]:
                    best[ln] = (span, node.end_lineno)
    for ln, (_, end) in best.items():
        ends[ln] = end
    return ends


def run(job):
    code, pre = job["code"], job.get("pre") or ""
    limit = job.get("max") or MAX_STEPS
    if "stdin" in job:
        sys.stdin = io.StringIO(job["stdin"])
    g = {"__name__": "__main__", "__builtins__": builtins}
    real_out = sys.stdout
    buf = io.StringIO()
    chunks = []          # (module line, text)
    steps = []
    truncated = [False]
    last_g = [None]

    def module_line():
        f, ln = sys._getframe(1), None
        while f is not None:
            if f.f_code.co_filename == SAMPLE:
                ln = f.f_lineno            # keep going: the OUTERMOST sample frame wins
            f = f.f_back
        return ln

    class Writer:
        def write(self, s):
            if s:
                chunks.append((module_line(), s))
                buf.write(s)
            return len(s)

        def flush(self):
            pass

        def isatty(self):
            return False

    helper_names = set()
    if pre:
        exec(compile(pre, "<helpers>", "exec"), g)
        helper_names = set(g)

    def record(frame, event, arg):
        stack, f = [], frame
        while f is not None:
            if f.f_code.co_filename == SAMPLE:
                stack.append(f.f_code.co_name)
            f = f.f_back
        stack.reverse()
        step = {"l": frame.f_lineno, "e": "r" if event == "return" else "l", "s": stack, "o": buf.tell()}
        if frame.f_code.co_name != "<module>":
            step["v"] = collect(frame.f_locals)
        gl = collect(frame.f_globals, helper_names)
        if gl != last_g[0]:
            step["g"] = gl
            last_g[0] = gl
        if event == "return":
            step["r"] = short(arg)
        steps.append(step)

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != SAMPLE:
            return None
        if event in ("line", "return"):
            if len(steps) < limit:
                record(frame, event, arg)
            else:
                truncated[0] = True
                sys.settrace(None)
                return None
        return tracer

    compiled = compile(code, SAMPLE, "exec")
    ok, err = True, ""
    sys.stdout = Writer()
    sys.settrace(tracer)
    try:
        exec(compiled, g)
    except SystemExit:
        pass
    except BaseException as e:      # noqa: BLE001
        ok = False
        err = f"{type(e).__name__}: {e}"
        chunks.append((None, f"\n!!{type(e).__name__}: {e}"))
    finally:
        sys.settrace(None)
        sys.stdout = real_out

    # merge output chunks by the line after which they should appear
    ends = statement_ends(code)
    outmap = []
    for ln, text in chunks:
        at = ends.get(ln, ln) if ln else 0
        if outmap and outmap[-1][0] == at:
            outmap[-1][1] += text
        else:
            outmap.append([at, text])
    steps.append({"l": 0, "e": "end", "s": [], "o": buf.tell()})
    return {"ok": ok, "out": buf.getvalue(), "outmap": outmap, "steps": steps, "truncated": truncated[0], "max": limit, "err": err,
            "imgs": figures()}


def figures():
    """Every open matplotlib figure as a base64 PNG (then closed)."""
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return []
    out = []
    for n in plt.get_fignums()[:6]:
        b = io.BytesIO()
        plt.figure(n).savefig(b, format="png", dpi=110, bbox_inches="tight")
        out.append(base64.b64encode(b.getvalue()).decode())
    plt.close("all")
    return out


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as fh:
        job = json.load(fh)
    res = run(job)
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, separators=(",", ":"))
