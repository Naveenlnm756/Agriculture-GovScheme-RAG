"""
Verification runner for `src/ingestion/chunker.py`.

Loads the entire V1 corpus, chunks it, and prints the chunker summary.
Eyeball this before we move on to embedding — it is the fastest way to
spot a scheme that produced too few chunks, a surge of suspiciously
short chunks, or a PDF that flowed through as zero chunks (the accepted
V1 losses from scope.md §8).

Run from the project root:
    python scripts/run_chunker_check.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Make `src` importable when this script is invoked as
# `python scripts/run_chunker_check.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunker import chunk_corpus  # noqa: E402
from src.ingestion.loader import load_corpus  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    corpus = load_corpus()
    chunked = chunk_corpus(corpus)
    summary = chunked.summary

    print()
    print("=" * 72)
    print("CHUNKER SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print("-" * 72)
    print("HEADLINE COUNTS")
    print("-" * 72)
    print(f"total chunks                : {summary['total_chunks']}")
    print(f"  PDF chunks                : {summary['pdf_chunks']}")
    print(f"  workflow chunks           : {summary['workflow_chunks']}")
    print(f"avg chunk length (chars)    : {summary['avg_chunk_length']}")
    print(f"min chunk length (chars)    : {summary['min_chunk_length']}")
    print(f"max chunk length (chars)    : {summary['max_chunk_length']}")
    print(
        f"chunk_size / overlap        : "
        f"{summary['chunk_size_chars']} / {summary['chunk_overlap_chars']}"
    )
    print(
        f"suspicious chunks (< {summary['suspicious_chunk_threshold']} chars) : "
        f"{summary['suspicious_chunks_count']}"
    )
    print(f"PDFs that produced 0 chunks : {summary['zero_chunk_pdfs_count']}")


if __name__ == "__main__":
    main()
