"""
Categorise the 24 non-OCR PDFs into ocrmypdf candidates.

Rules:
  * KEEP  if fraction of scanned pages >= 0.5 (majority of doc is scanned)
  * KEEP  if fraction >= 0.1 AND n_pages <= 30  (small letters / notifications)
  * KEEP  if entire doc is scanned (n_scanned == n_pages)
  * INSPECT if 1-2 scanned pages inside a large doc — could be a
             front/back cover or a single inserted scan. Print the page
             numbers so owner can decide.
  * SKIP the rest (assumed false positive)

Read-only. Prints a plan. No side effects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def _bucket(n_scanned: int, n_pages: int) -> str:
    if n_scanned == 0:
        return "SKIP (no scanned pages)"
    frac = n_scanned / n_pages
    if frac >= 0.5:
        return "OCR (majority scanned)"
    if n_pages <= 30 and frac >= 0.1:
        return "OCR (small doc, meaningful scan share)"
    if n_pages <= 10 and n_scanned >= 2:
        return "OCR (small letter)"
    if n_scanned <= 2 and n_pages >= 50:
        return "INSPECT (1-2 scans in a large doc)"
    return "OCR (default keep)"


def main() -> int:
    data = json.load(
        (
            _PROJECT_ROOT / "eval" / "results" / "image_modality" / "summary.json"
        ).open(encoding="utf-8")
    )

    keep: list[dict] = []
    inspect: list[dict] = []
    skip: list[dict] = []

    for pdf in data["per_pdf"]:
        if "error" in pdf:
            continue
        if "_OCR" in pdf["filename"]:
            continue
        n_scanned = pdf.get("n_scanned_pages", 0)
        if n_scanned <= 0:
            continue
        bkt = _bucket(n_scanned, pdf["n_pages"])
        record = {
            "scheme": pdf["scheme"],
            "filename": pdf["filename"],
            "filepath": pdf["filepath"],
            "n_pages": pdf["n_pages"],
            "n_scanned": n_scanned,
            "bucket": bkt,
            "scanned_page_nums": [
                p["page_num"]
                for p in pdf.get("interesting_pages", [])
                if p.get("likely_scanned")
            ],
        }
        if bkt.startswith("OCR"):
            keep.append(record)
        elif bkt.startswith("INSPECT"):
            inspect.append(record)
        else:
            skip.append(record)

    def _dump(title: str, rows: list[dict]) -> None:
        print("=" * 108)
        print(f"{title}  ({len(rows)} PDFs, "
              f"{sum(r['n_scanned'] for r in rows)} scanned pages)")
        print("=" * 108)
        if not rows:
            print("(none)")
            return
        for r in rows:
            frac = r["n_scanned"] / max(r["n_pages"], 1)
            pages = ",".join(str(p) for p in r["scanned_page_nums"][:20])
            more = "…" if len(r["scanned_page_nums"]) > 20 else ""
            print(
                f"[{r['scheme']:<10}] {r['n_scanned']:>3}/{r['n_pages']:<4} "
                f"({frac:.0%})  scanned_pages=[{pages}{more}]  {r['filename']}"
            )
        print()

    _dump("A. OCR CANDIDATES", keep)
    _dump("B. NEEDS INSPECTION", inspect)
    _dump("C. SKIP", skip)

    print(
        f"summary: OCR={len(keep)}  inspect={len(inspect)}  skip={len(skip)}   "
        f"total scanned pages to OCR = {sum(r['n_scanned'] for r in keep)}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
