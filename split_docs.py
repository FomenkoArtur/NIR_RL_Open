# split_docs.py
import re, pathlib
t = pathlib.Path("docs/work.md").read_text(encoding="utf-8")
parts = re.split(r"(?m)^(?=#{1,3} )", t)
out = pathlib.Path("docs/sections"); out.mkdir(exist_ok=True)
for i, p in enumerate(parts):
    title = p.splitlines()[0] if p.strip() else ""
    name = re.sub(r"[^\w]+", "_", title)[:50]
    if len(p) > 500:
        (out / f"{i:03d}_{name}.md").write_text(p, encoding="utf-8")