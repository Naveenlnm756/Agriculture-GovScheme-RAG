"""
Fresh-embed runner for baseline v2 (structure-aware chunker, Phase 5 fix #1).

Same load → chunk → embed → upsert pipeline as `run_embedder_check.py`,
with `settings.use_structure_aware_chunking` flipped ON for this run
so pymupdf table preservation is active.

Kept as a SEPARATE script (rather than a flag on run_embedder_check.py)
so the v1 and v2 entry points are explicit and defensible:
  * `run_embedder_check.py` reproduces the v1 baseline chunking.
  * `run_embedder_v2.py`     produces the v2 structure-aware chunking.

The Chroma path is unchanged (`data/chroma_db/`). Before running this
script, delete `data/chroma_db/` so the fresh embed starts from an empty
collection — otherwise stale v1 chunk ids that no v2 chunk overwrites
would linger in the store and `collection_size_after` would exceed
`total_upserted`. The v1 anchor is preserved at `data/chroma_db_v1_locked/`.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from src.config import settings  # noqa: E402
from src.embeddings.embedder import embed_corpus  # noqa: E402
from src.ingestion.chunker import chunk_corpus  # noqa: E402
from src.ingestion.loader import load_corpus  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Flip ON only in this process — .env stays clean so accidental
    # future `run_embedder_check.py` invocations still produce v1.
    settings.use_structure_aware_chunking = True
    logging.getLogger(__name__).info(
        "use_structure_aware_chunking=%s   chunk_size=%d   chunk_overlap=%d",
        settings.use_structure_aware_chunking,
        settings.chunk_size,
        settings.chunk_overlap,
    )

    corpus = load_corpus()
    chunked = chunk_corpus(corpus)
    summary = embed_corpus(chunked)

    print()
    print("=" * 72)
    print("CHUNKER SUMMARY (structure-aware block only)")
    print("=" * 72)
    sa = chunked.summary.get("structure_aware")
    if sa is None:
        print("!! structure_aware block missing — flag not honoured?")
    else:
        # Print the counts inline and per-doc separately (per-doc is
        # long, keep it below the headline).
        headline = {
            k: v for k, v in sa.items()
            if k not in ("per_doc_table_counts", "fallback_docs")
        }
        print(json.dumps(headline, indent=2, ensure_ascii=False))
        print(f"\nfallback_docs (n={len(sa.get('fallback_docs', []))}):")
        for d in sa.get("fallback_docs", []):
            print(f"  {d['scheme']:<10} {d['filename']:<60} reason={d['reason']}")

    print()
    print("=" * 72)
    print("EMBEDDING SUMMARY")
    print("=" * 72)
    summary_dict = summary.model_dump()
    print(json.dumps(summary_dict, indent=2, ensure_ascii=False))

    print()
    print("-" * 72)
    print("HEADLINE COUNTS")
    print("-" * 72)
    print(f"model                       : {summary.model_name}")
    print(f"embedding dimension         : {summary.embedding_dimension}")
    print(f"collection                  : {summary.collection_name}")
    print(f"chunks processed            : {summary.total_chunks_processed}")
    print(f"chunks upserted             : {summary.total_upserted}")
    print(f"collection size after       : {summary.collection_size_after}")
    print(f"truncated (over max tokens) : {summary.truncated_chunk_count}")
    print(f"failures                    : {len(summary.failures)}")

    # Sanity: on a fresh collection, upserted == collection_size_after.
    # A divergence means either the pre-run wipe was skipped, or the
    # embedder hit an upsert failure that didn't get recorded.
    if summary.collection_size_after != summary.total_upserted:
        print(
            f"\nWARNING: collection_size_after ({summary.collection_size_after}) "
            f"!= total_upserted ({summary.total_upserted}). "
            f"Did you clear data/chroma_db/ before running?"
        )


if __name__ == "__main__":
    main()
