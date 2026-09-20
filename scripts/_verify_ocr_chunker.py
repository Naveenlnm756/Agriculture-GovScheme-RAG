"""
Verify OCR recovery: for each of the 22 OCR'd PDFs, open the ORIGINAL
and the NEW _OCR sibling with pypdf and pymupdf, and report the
before/after character count for the pages that were flagged as
scanned in the image-modality diagnostic. Confirms the OCR pass
actually put a text layer down that our loader will see.

Read-only. Does not touch data/raw/, does not touch Chroma.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf
from pypdf import PdfReader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def _extract_chars(reader, page_indices: list[int]) -> int:
    total = 0
    for idx in page_indices:
        try:
            text = reader.pages[idx].extract_text() or ""
        except Exception:
            text = ""
        total += len(text.strip())
    return total


def _extract_chars_pymupdf(pdf_path: Path, page_indices: list[int]) -> int:
    doc = pymupdf.open(str(pdf_path))
    try:
        total = 0
        for idx in page_indices:
            try:
                total += len((doc[idx].get_text() or "").strip())
            except Exception:
                pass
        return total
    finally:
        doc.close()


def main() -> int:
    data = json.load(
        (_PROJECT_ROOT / "eval" / "results" / "image_modality" / "summary.json").open(
            encoding="utf-8"
        )
    )

    # Find each candidate PDF, compute before/after char counts on the
    # scanned pages ONLY (not the whole PDF — that would drown the
    # recovery signal in already-typed prose).
    header = (
        f"{'scheme':<10} {'scanned':>7}  "
        f"{'before_pypdf':>12} {'after_pypdf':>11}   "
        f"{'before_pymu':>11} {'after_pymu':>10}   filename"
    )
    print(header)
    print("-" * len(header))

    n_recovered = n_flat = n_missing = 0

    for pdf in data["per_pdf"]:
        if "error" in pdf:
            continue
        if "_OCR" in pdf["filename"]:
            continue
        scanned_pages = [
            p["page_num"] - 1
            for p in pdf.get("interesting_pages", [])
            if p.get("likely_scanned")
        ]
        if not scanned_pages:
            continue

        src = Path(pdf["filepath"])
        ocr = src.with_name(src.stem + "_OCR.pdf")
        if not ocr.exists():
            print(f"{pdf['scheme']:<10} {len(scanned_pages):>7}  (no _OCR sibling)   {src.name}")
            n_missing += 1
            continue

        # BEFORE — pypdf and pymupdf both applied to the ORIGINAL
        before_pypdf = _extract_chars(PdfReader(str(src)), scanned_pages)
        before_pymu = _extract_chars_pymupdf(src, scanned_pages)

        # AFTER — same, applied to the _OCR sibling
        after_pypdf = _extract_chars(PdfReader(str(ocr)), scanned_pages)
        after_pymu = _extract_chars_pymupdf(ocr, scanned_pages)

        gained = max(after_pypdf - before_pypdf, after_pymu - before_pymu)
        marker = "✓" if gained >= 200 else ("~" if gained >= 50 else "✗")
        if gained >= 200:
            n_recovered += 1
        elif gained >= 50:
            n_recovered += 1  # small letters can be <200 chars
        else:
            n_flat += 1

        print(
            f"{pdf['scheme']:<10} {len(scanned_pages):>7}  "
            f"{before_pypdf:>12} {after_pypdf:>11}   "
            f"{before_pymu:>11} {after_pymu:>10}  {marker} {src.name}"
        )

    print()
    print(f"recovered (chars gained >=50): {n_recovered}")
    print(f"flat / no gain              : {n_flat}")
    print(f"_OCR sibling missing        : {n_missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
