"""
Build the unified production Chroma collection `agri_schemes_prod`.

Combines three chunk sources under ONE collection so retrieval is
modality-agnostic:

  * Structure-aware text + table chunks (v2 chunker on all 96 PDFs,
    including the 22 newly-OCR'd _OCR siblings)
  * Image chunks (re-materialised from the vision cache — no fresh
    Gemini calls, entire ingestion should take seconds)

Idempotent: rebuild by deleting `data/chroma_prod/` first, then
running. Chunk ids are deterministic (chunker + image_chunker both
produce stable ids) so re-running against an existing collection
upserts in place.

Never modifies:
  * data/raw/ (owner-managed corpus)
  * data/chroma_db/ or data/chroma_db_v1_locked/ (ablation anchor)
  * data/vision_cache/ (build artefact, but read-only here)
  * data/chroma_prod_image_test/ (the B5/B6 test collection, kept
    around for E2E-test reproducibility)

CLI:
    --dry-run        chunk + count but skip Chroma upsert
    --skip-images    text + table only (useful for isolating the
                     re-embed cost from image work)
    --skip-text      images only (bootstrapping a prod collection
                     from an existing text-collection state)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import chromadb  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from src.config import settings  # noqa: E402
from src.embeddings.embedder import _chunk_to_chroma_record  # noqa: E402
from src.ingestion.chunker import chunk_corpus  # noqa: E402
from src.ingestion.image_chunker import make_image_chunks  # noqa: E402
from src.ingestion.image_extractor import extract_image_candidates  # noqa: E402
from src.ingestion.loader import load_corpus  # noqa: E402
from src.ingestion.models import Chunk  # noqa: E402
from src.vision.cache import VisionCache  # noqa: E402


logger = logging.getLogger(__name__)


PROD_COLLECTION = settings.production_collection_name
PROD_DIR = _PROJECT_ROOT / "data" / "chroma_prod"
BUILD_LOG = _PROJECT_ROOT / "eval" / "results" / "prod_collection_build_log.json"


def _get_prod_collection():
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(PROD_DIR))
    return client.get_or_create_collection(
        name=PROD_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _upsert_batches(collection, embedder, chunks: list[Chunk], batch_size: int) -> int:
    """Batch-embed and upsert. Returns the count actually upserted.
    Prints progress every 10 batches."""
    if not chunks:
        return 0
    n_upserted = 0
    n_batches = (len(chunks) + batch_size - 1) // batch_size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        records = [_chunk_to_chroma_record(c) for c in batch]
        ids = [r[0] for r in records]
        docs = [r[1] for r in records]
        metas = [r[2] for r in records]
        vecs = embedder.encode(
            docs, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()
        collection.upsert(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
        n_upserted += len(batch)
        bnum = (i // batch_size) + 1
        if bnum % 10 == 0 or bnum == n_batches:
            logger.info("  upserted %d/%d chunks (batch %d/%d)",
                        n_upserted, len(chunks), bnum, n_batches)
    return n_upserted


def _load_image_chunks_from_cache() -> tuple[list[Chunk], dict]:
    """
    Re-enumerate candidates + read the cache — no Gemini calls.

    We re-run the extractor (~3 min) so we get fresh per-page
    occurrence lists (the cache stores the VisionResult keyed by
    content-hash, not the occurrences). For every candidate whose
    result is cached, we materialise chunks.
    """
    candidates, diag = extract_image_candidates()
    cache = VisionCache()
    counts = {"TABLE": 0, "CHART": 0, "INFOGRAPHIC": 0, "OTHER": 0,
              "DECORATIVE": 0, "vision_failed": 0, "cache_miss": 0}
    chunks: list[Chunk] = []
    for cand in candidates:
        vr = cache.get(cand.content_hash)
        if vr is None:
            counts["cache_miss"] += 1
            continue
        key = "vision_failed" if vr.vision_failed else vr.classification
        counts[key] = counts.get(key, 0) + 1
        # Only useful + vision_failed produce chunks; OTHER / DECORATIVE
        # never do.
        chunks.extend(make_image_chunks(cand, vr))
    return chunks, {"candidate_diag": diag, "class_counts": counts,
                    "n_image_chunks": len(chunks)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="chunk + count but skip embed/upsert")
    ap.add_argument("--skip-images", action="store_true",
                    help="text + table only (no image chunks)")
    ap.add_argument("--skip-text", action="store_true",
                    help="images only (no text/table chunks)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    # Force the v2 chunker on for text side. Never write the change
    # back to .env; this is a per-process flip.
    settings.use_structure_aware_chunking = True

    summary: dict = {
        "collection_name": PROD_COLLECTION,
        "persist_dir": str(PROD_DIR),
        "dry_run": args.dry_run,
        "skip_images": args.skip_images,
        "skip_text": args.skip_text,
    }

    all_chunks: list[Chunk] = []

    # --- Text + table chunks ---
    if not args.skip_text:
        logger.info("Loading corpus (OCR-sibling preference applied)...")
        t0 = time.time()
        corpus = load_corpus()
        logger.info("Chunking with structure-aware chunker...")
        chunked = chunk_corpus(corpus)
        text_table_chunks = list(chunked.chunks)
        summary["text_table_chunks"] = len(text_table_chunks)
        summary["text_table_chunker_summary"] = {
            k: v for k, v in chunked.summary.items()
            if k in ("total_chunks", "pdf_chunks", "workflow_chunks",
                     "avg_chunk_length", "min_chunk_length", "max_chunk_length",
                     "zero_chunk_pdfs_count")
        }
        # Structure-aware sub-block (tables preserved etc.)
        sa = chunked.summary.get("structure_aware") or {}
        summary["text_table_chunker_summary"]["structure_aware"] = {
            k: sa.get(k) for k in
            ("enabled", "total_tables_detected", "total_text_windows",
             "table_chunks_over_1500_chars", "fallback_docs_count")
        }
        summary["text_table_elapsed_s"] = round(time.time() - t0, 1)
        logger.info("text+table: %d chunks in %.1fs",
                    len(text_table_chunks), summary["text_table_elapsed_s"])
        all_chunks.extend(text_table_chunks)

    # --- Image chunks (from cache, no fresh Gemini) ---
    if not args.skip_images:
        logger.info("Materialising image chunks from vision cache...")
        t0 = time.time()
        image_chunks, image_diag = _load_image_chunks_from_cache()
        summary["image_chunks"] = len(image_chunks)
        summary["image_diagnostic"] = image_diag
        summary["image_elapsed_s"] = round(time.time() - t0, 1)
        logger.info("image: %d chunks in %.1fs",
                    len(image_chunks), summary["image_elapsed_s"])
        all_chunks.extend(image_chunks)

    summary["total_chunks"] = len(all_chunks)

    # --- Chunk-id uniqueness sanity check ---
    ids = [c.chunk_id for c in all_chunks]
    unique_ids = set(ids)
    if len(unique_ids) != len(ids):
        # Which ids collide?
        from collections import Counter
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        summary["chunk_id_collisions"] = dupes[:10]
        summary["n_chunk_id_collisions"] = len(dupes)
        logger.error("chunk_id collisions: %d unique dupes; first 10: %s",
                     len(dupes), dupes[:10])

    # --- Upsert ---
    if not args.dry_run and all_chunks:
        logger.info("Loading embedder %s ...", settings.embedding_model)
        embedder = SentenceTransformer(settings.embedding_model)
        collection = _get_prod_collection()
        logger.info("Upserting %d chunks into collection %s (dir %s)...",
                    len(all_chunks), PROD_COLLECTION, PROD_DIR)
        t0 = time.time()
        upserted = _upsert_batches(collection, embedder, all_chunks,
                                   settings.embed_batch_size)
        summary["upserted"] = upserted
        summary["upsert_elapsed_s"] = round(time.time() - t0, 1)
        summary["collection_size_after"] = collection.count()
        logger.info("upserted %d chunks; collection size after = %d",
                    upserted, summary["collection_size_after"])
    else:
        summary["upserted"] = 0

    BUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
    BUILD_LOG.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print()
    print("=" * 72)
    print("PROD COLLECTION BUILD SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nfull log → {BUILD_LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
