"""
Tiny survey: from the image-modality diagnostic JSON, list every PDF
that is NOT already _OCR-suffixed but has one or more likely-scanned
pages. These are the OCR-first candidates.

Read-only. Just prints. No side effects.
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


def main() -> int:
    data = json.load(
        (_PROJECT_ROOT / "eval" / "results" / "image_modality" / "summary.json").open(
            encoding="utf-8"
        )
    )

    candidates = []
    for pdf in data["per_pdf"]:
        if "error" in pdf:
            continue
        fn = pdf["filename"]
        if "_OCR" in fn:
            continue
        n_scanned = pdf.get("n_scanned_pages", 0)
        if n_scanned <= 0:
            continue
        candidates.append(pdf)

    candidates.sort(key=lambda p: -p["n_scanned_pages"])

    header = f"{'scheme':<10} {'scanned':>7} {'pages':>6}  {'filename'}"
    print(header)
    print("-" * 110)
    for p in candidates:
        print(
            f"{p['scheme']:<10} {p['n_scanned_pages']:>7} {p['n_pages']:>6}  "
            f"{p['filename']}"
        )
    print()
    print(f"total non-OCR PDFs with scanned pages : {len(candidates)}")
    print(
        "total scanned pages across them        : "
        f"{sum(p['n_scanned_pages'] for p in candidates)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
