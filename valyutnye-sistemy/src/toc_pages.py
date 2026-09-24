"""Находит номера страниц заголовков в PDF первого прохода и пишет toc.json."""
import json
import re
import sys

pdf, out = sys.argv[1], sys.argv[2]
import pymupdf  # pip install pymupdf

pages = [re.sub(r"\s+", " ", pg.get_text()) for pg in pymupdf.open(pdf)]

src = open(__file__.replace("toc_pages.py", "build_docx.js"), encoding="utf-8").read()
block = src[src.index("const tocItems = ["):src.index("];", src.index("const tocItems = ["))]
items = re.findall(r'\["([^"]+)", [01]\]', block)

res, start = {}, 2  # поиск после страницы содержания
for t in items:
    key = re.sub(r"\s+", " ", t)
    probe = key if not re.match(r"^\d", key) else key
    for i in range(start, len(pages)):
        page_txt = pages[i]
        if probe.upper() in page_txt.upper()[:4000] and (probe in page_txt or probe.upper() in page_txt):
            res[t] = i + 1
            start = i
            break
    else:
        # заголовок мог перенестись на две строки — ищем по первым 30 символам
        for i in range(start, len(pages)):
            if key[:30] in pages[i]:
                res[t] = i + 1
                start = i
                break
missing = [t for t in items if t not in res]
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"найдено {len(res)}/{len(items)}", "не найдено: " + "; ".join(missing) if missing else "")
