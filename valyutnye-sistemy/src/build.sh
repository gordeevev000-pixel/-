#!/usr/bin/env bash
# Полная сборка доклада: рисунки → .docx (2 прохода для страниц в содержании) → .pdf
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="$HERE/.."
WORK="${WORK:-$(mktemp -d)}"
python3 "$HERE/figures.py" "$WORK/fig"
node "$HERE/build_docx.js" "$WORK/fig" "$WORK/pass1.docx"
soffice --headless --convert-to pdf --outdir "$WORK" "$WORK/pass1.docx" >/dev/null 2>&1
python3 "$HERE/toc_pages.py" "$WORK/pass1.pdf" "$WORK/toc.json"
node "$HERE/build_docx.js" "$WORK/fig" "$OUTDIR/doklad.docx" "$WORK/toc.json"
soffice --headless --convert-to pdf --outdir "$OUTDIR" "$OUTDIR/doklad.docx" >/dev/null 2>&1
echo "готово: $OUTDIR/doklad.docx, $OUTDIR/doklad.pdf"
