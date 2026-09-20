"""
Render a curated set of pages per doc as PNG (at 150 DPI) so tables
can be visually verified for ground-truth construction.

Pages chosen to sample both "candidate-rich" and "candidate-sparse"
regions of each doc so the ground truth includes both TP-visible and
FN-visible pages.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pymupdf

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = _PROJECT_ROOT / "eval" / "results" / "heading_ab" / "table_gt_pages"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS_PAGES = [
    # (doc_name, path, [pages_to_render])
    ("midh_2025",   "data/raw/MIDH/01_RAW_PDFs/MIDH_Operational_Guideline_2025_OCR.pdf", [8, 20, 45, 60, 75]),
    ("nhb_final",   "data/raw/MIDH/01_RAW_PDFs/FinalNHBOperationalGuideline_OCR.pdf",   [10, 30, 55, 75]),
    ("aif_sep2024", "data/raw/AIF/01_RAW_PDFs/AIF Revised Scheme Guidelines  September 2024_OCR.pdf", [4, 10, 15]),
    ("pmfby_2020",  "data/raw/PMFBY/01_RAW_PDFs/Revamped Operational Guidelines_17th August 2020.pdf", [15, 50, 100, 140]),
    ("rbi_kcc_2026","data/raw/KCC/01_RAW_PDFs/RBI_KCC_Directions_2026_Commercial_Banks.pdf", [5, 12, 18]),
]

def main() -> int:
    for name, rel, pages in DOCS_PAGES:
        path = _PROJECT_ROOT / rel
        doc = pymupdf.open(path)
        for pnum in pages:
            if pnum < 1 or pnum > len(doc):
                continue
            page = doc[pnum - 1]  # 0-indexed
            pix = page.get_pixmap(dpi=150)
            out = OUT_DIR / f"{name}_p{pnum:03d}.png"
            pix.save(out)
            print(f"[{name}] p{pnum} -> {out.name}  ({pix.width}x{pix.height})")
        doc.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
