"""
Verification runner for `src/embeddings/embedder.py`.

Runs the full ingestion pipeline (loader → chunker → embedder), prints
the EmbeddingSummary, and then does a smoke test: issues one hardcoded
query against Chroma and shows the top-3 results. Eyeball this before
we move on to the retriever — it is the fastest way to confirm the
collection is populated, queryable, and returning plausibly relevant
chunks (not garbage or empty results).

Run from the project root:
    python scripts/run_embedder_check.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Make `src` importable when this script is invoked as
# `python scripts/run_embedder_check.py` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.embeddings.embedder import (  # noqa: E402
    _get_or_create_collection,
    _load_embedder,
    embed_corpus,
)
from src.ingestion.chunker import chunk_corpus  # noqa: E402
from src.ingestion.loader import load_corpus  # noqa: E402


# Hardcoded smoke-test query. Deliberately KCC/gold-loan-flavoured so the
# top hits should come from RBI / KCC-family documents (KCC prose is one
# of the densest, most obviously-relevant regions of the corpus). If the
# top-3 comes back scheme-random, the collection is not returning
# semantically meaningful nearest neighbours and something is wrong.
SMOKE_QUERY = "gold loan eligibility"
SMOKE_TOP_K = 3


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --- Full pipeline: load → chunk → embed → upsert ---
    corpus = load_corpus()
    chunked = chunk_corpus(corpus)
    summary = embed_corpus(chunked)

    # Serialise the summary through model_dump so we get clean JSON
    # (Path / pydantic objects → primitives) without hand-mapping fields.
    summary_dict = summary.model_dump()

    print()
    print("=" * 72)
    print("EMBEDDING SUMMARY")
    print("=" * 72)
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

    # --- Smoke test: run one query end-to-end against Chroma ---
    # We embed the query with the SAME model that produced the corpus
    # vectors — mismatched models here would be a silent, catastrophic
    # retrieval bug (nearest-neighbour distances would be meaningless).
    print()
    print("=" * 72)
    print(f"SMOKE QUERY: {SMOKE_QUERY!r} (top {SMOKE_TOP_K})")
    print("=" * 72)

    model = _load_embedder(settings)
    collection = _get_or_create_collection(settings)

    query_vec = model.encode(
        [SMOKE_QUERY], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_vec,
        n_results=SMOKE_TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not ids:
        print("NO RESULTS — collection appears empty or unreachable.")
        return

    for rank, (cid, doc, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        preview = (doc[:280] + "...") if len(doc) > 280 else doc
        preview = preview.replace("\n", " ")
        print()
        print(f"[{rank}] id={cid}  distance={dist:.4f}")
        print(f"     scheme={meta.get('scheme')}  "
              f"type={meta.get('source_type')}  "
              f"file={meta.get('source_filename')}")
        print(f"     text: {preview}")


if __name__ == "__main__":
    main()
