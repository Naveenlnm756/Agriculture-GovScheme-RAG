"""
Verification runner for `src/retrieval/retriever.py`.

Runs a fixed set of five hardcoded queries against the baseline
semantic retriever and prints the top-K results in a readable table.
Eyeball this before we move on to any ablation intervention — it is
the fastest way to confirm the baseline is honest about what it finds
(and what it doesn't).

The five queries deliberately span the corpus:

  1. Straight PDF question, single scheme (PM-KISAN).
  2. Near-miss / no-good-answer probe (gold-loan LTV — the corpus
     has no gold-loan-specific docs, so the baseline should either
     return low-similarity KCC hits or admit it doesn't know).
  3. Multi-part PMFBY question (procedure + risk type).
  4. Cross-scheme-adjacent question (KCC + MISS interest subvention).
  5. AIF temporal / procedural question.

Run from the project root:
    python scripts/run_retriever_check.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import retrieve  # noqa: E402


QUERIES: list[str] = [
    "How do I apply for PM-KISAN?",
    "What is the LTV limit for gold loans?",
    "PMFBY claim procedure for crop loss due to hailstorm",
    "Kisan Credit Card interest subvention",
    "How long is the completion window for AIF projects?",
]


def _preview(text: str, n: int = 200) -> str:
    """First `n` chars of `text`, with newlines flattened for one-line printing."""
    snippet = text[:n].replace("\n", " ")
    if len(text) > n:
        snippet += "..."
    return snippet


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for qi, query in enumerate(QUERIES, start=1):
        print()
        print("=" * 88)
        print(f"QUERY {qi}: {query!r}")
        print("=" * 88)

        results = retrieve(query)

        if not results:
            print("  (no results)")
            continue

        for r in results:
            print()
            print(f"  [rank {r.rank}]  similarity={r.similarity_score:.4f}  "
                  f"scheme={r.scheme}  type={r.source_type}")
            print(f"    file: {r.source_filename}")
            loc_bits: list[str] = []
            if r.page_start is not None:
                loc_bits.append(f"pages {r.page_start}-{r.page_end}")
            if r.workflow_id is not None:
                loc_bits.append(f"workflow_id={r.workflow_id}")
            if r.is_ocr_source:
                loc_bits.append("OCR")
            if loc_bits:
                print(f"    loc : {'  '.join(loc_bits)}")
            print(f"    text: {_preview(r.text)}")


if __name__ == "__main__":
    main()
