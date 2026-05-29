"""
Extract PDFs to markdown using pymupdf4llm.
Outputs:
  data/papers/extracted/<name>.md        — full markdown
  data/papers/extracted/<name>_chunks.json — page_chunks JSON
"""

import json
import pathlib
import pymupdf4llm

BASE = pathlib.Path("/Users/abhayapat/git-repos/rag-comparison-2/claude_rc_2")
PDF_DIR = BASE / "data" / "papers"
OUT_DIR = PDF_DIR / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pdfs = sorted(PDF_DIR.glob("*.pdf"))
print(f"Found {len(pdfs)} PDFs\n")
print(f"{'Filename':<55} {'Pages':>5} {'Chars':>10}")
print("-" * 75)

for pdf_path in pdfs:
    stem = pdf_path.stem

    # Extract with page_chunks=True
    chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    # Build full markdown by joining all page text
    full_md = "\n\n".join(chunk["text"] for chunk in chunks)

    # Write markdown
    md_out = OUT_DIR / f"{stem}.md"
    md_out.write_text(full_md, encoding="utf-8")

    # Write chunks JSON
    json_out = OUT_DIR / f"{stem}_chunks.json"
    json_out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    page_count = len(chunks)
    char_count = len(full_md)
    print(f"{pdf_path.name:<55} {page_count:>5} {char_count:>10,}")

print("\nExtraction complete.")
print(f"Output directory: {OUT_DIR}")
