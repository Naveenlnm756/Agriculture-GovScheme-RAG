"""
Verification runner for `src/ingestion/loader.py`.

Loads the entire V1 corpus and prints the summary dict. Eyeball this
before we move on to chunking — it's the fastest way to spot a scheme
that lost documents, an OCR preference that didn't fire, or a silent
"loaded but empty" PDF that will hurt retrieval later.

Run from the project root:
    python scripts/run_loader_check.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Make `src` importable when this script is invoked as
# `python scripts/run_loader_check.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_corpus  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    corpus = load_corpus()
    summary = corpus.summary

    print()
    print("=" * 72)
    print("CORPUS LOAD SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print("-" * 72)
    print("HEADLINE COUNTS")
    print("-" * 72)
    print(f"loaded PDFs               : {summary['total_pdfs']}")
    print(f"loaded workflows          : {summary['total_workflows']}")
    print(f"total pages               : {summary['total_pages']}")
    print(f"total chars               : {summary['total_chars']:,}")
    print(f"OCR preferences applied   : {len(summary['ocr_preferences'])}")
    print(f"empty PDFs (<100 chars)   : {len(summary['empty_pdfs'])}")
    print(f"failures                  : {len(summary['failures'])}")


if __name__ == "__main__":
    main()
