"""
End-to-end verification runner for the Phase 2 baseline pipeline:
loader → chunker → embedder (already run) → retriever → generator.

Runs the same 5 diagnostic queries used by `run_retriever_check.py`
so the retrieval-to-answer story is directly comparable. For each
query it prints:

  - the query
  - the top-K retrieved sources (chunk_id + scheme + similarity)
  - the generated answer
  - every citation string parsed out of the answer
  - Groq usage (prompt tokens, completion tokens, latency, retries)

This is the Phase 2 closing smoke test. Eyeball it before we move
to Phase 3 (golden set) or Phase 5 (ablation interventions).

Run from the project root:
    python scripts/run_generator_check.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.generator import generate  # noqa: E402
from src.retrieval.retriever import retrieve  # noqa: E402


QUERIES: list[str] = [
    "How do I apply for PM-KISAN?",
    "What is the LTV limit for gold loans?",
    "PMFBY claim procedure for crop loss due to hailstorm",
    "Kisan Credit Card interest subvention",
    "How long is the completion window for AIF projects?",
]


# Match the two citation formats the system prompt tells the LLM to emit:
#   [Source: {scheme}/{filename}, page {n}]
#   [Source: {scheme}/{workflow_id}]
# Greedy on the inside because filenames and workflow_ids can carry
# spaces, underscores, dashes, and dots.
CITATION_RE = re.compile(r"\[Source:\s*[^\]]+\]")


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

        chunks = retrieve(query)

        print()
        print("-- retrieved sources --")
        if not chunks:
            print("  (no chunks returned)")
        for c in chunks:
            print(
                f"  [rank {c.rank}]  sim={c.similarity_score:.4f}  "
                f"scheme={c.scheme}  type={c.source_type}  "
                f"id={c.chunk_id}"
            )

        result = generate(query, chunks)

        print()
        print("-- generated answer --")
        print(result.answer)

        citations = CITATION_RE.findall(result.answer)
        print()
        print(f"-- citations parsed ({len(citations)}) --")
        for c in citations:
            print(f"  {c}")

        print()
        print("-- generation stats --")
        print(f"  model           : {result.model_used}")
        print(f"  prompt_tokens   : {result.prompt_tokens}")
        print(f"  completion_toks : {result.completion_tokens}")
        print(f"  reasoning_toks  : {result.reasoning_tokens}")
        # Answer tokens = completion tokens NOT spent on reasoning. This is
        # the number that actually landed in the visible answer.
        answer_toks = max(result.completion_tokens - result.reasoning_tokens, 0)
        print(f"  answer_toks     : {answer_toks}")
        print(f"  latency_ms      : {result.latency_ms}")
        print(f"  finish_reason   : {result.finish_reason}")
        print(f"  retries_taken   : {result.retries_taken}")


if __name__ == "__main__":
    main()
