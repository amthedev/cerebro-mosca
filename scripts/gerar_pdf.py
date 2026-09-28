"""Gera artigo/preprint.pdf a partir de artigo/preprint.md (Markdown -> HTML -> PDF pelo Chrome).

uv run --with markdown python scripts/gerar_pdf.py
"""

import subprocess
from pathlib import Path

import markdown

RAIZ = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: A4; margin: 2cm 2.2cm; }
body { font-family: "Charter", "Georgia", serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; line-height: 1.25; margin-bottom: 0.3em; }
h2 { font-size: 12.5pt; margin-top: 1.4em; border-bottom: 1px solid #ccc; padding-bottom: 2px; }
h3 { font-size: 11pt; margin-top: 1.1em; }
table { border-collapse: collapse; font-size: 9pt; margin: 0.8em 0; }
th, td { border-bottom: 1px solid #ddd; padding: 3px 8px; text-align: left; }
img { max-width: 100%; display: block; margin: 0.6em auto 0.2em; }
p img + em, figure figcaption { font-size: 9pt; }
blockquote { color: #444; }
hr { border: none; border-top: 1px solid #ccc; }
code { font-size: 9pt; }
"""

md = (RAIZ / "artigo" / "preprint.md").read_text()
corpo = markdown.markdown(md, extensions=["tables"])
html = f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>{corpo}</body></html>'
arquivo_html = RAIZ / "artigo" / "preprint.html"
arquivo_html.write_text(html)
pdf = RAIZ / "artigo" / "preprint.pdf"
subprocess.run(
    [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", arquivo_html.as_uri()],
    check=True,
    capture_output=True,
)
arquivo_html.unlink()
print("PDF:", pdf)
