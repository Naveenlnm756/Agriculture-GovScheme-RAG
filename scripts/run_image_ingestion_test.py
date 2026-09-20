"""
End-to-end runner: image candidates → vision → chunks → Chroma
(Deliverable 2 B5.6).

Pipeline (idempotent across runs — cached vision results and Chroma
upserts both dedup by content-hash / chunk_id):

  1. extract_image_candidates() → full unique-hash pool (no cap)
  2. For every candidate:
       a. Cache-hit → skip vision, reuse result
       b. Cache-miss → classify (stage 1)
          - DECORATIVE / OTHER    → cache result, skip stage 2, no chunks
          - TABLE / CHART / INFOGRAPHIC → stage 2 extraction
              - Empty / invalid response → retry once at
                `vision_fallback_dpi` (fewer pixels)
              - Still empty → mark vision_failed=True, still cache
  3. For every candidate that yields chunks (useful classification OR
     vision_failed), materialise one Chunk per per-page occurrence.
  4. Embed and upsert into the scratch collection `<production>_image_test`
     (kept SEPARATE from the ablation-anchor collection so an E2E test
     never contaminates the locked ablation table).

Between vision calls we sleep `vision_call_delay_s` seconds — matches
the benchmark spacing. On big cache-hit rates (re-runs) the sleep is
skipped so a resume is fast.

CLI:
    --limit N   process only the first N unique candidates (dev/test)
    --dry-run   run classify+extract but DO NOT embed/upsert into Chroma
                (useful for cost verification before spending the
                embedding pass)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import chromadb  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from src.config import settings  # noqa: E402
from src.ingestion.image_chunker import make_image_chunks  # noqa: E402
from src.ingestion.image_extractor import (  # noqa: E402
    ImageCandidate,
    extract_image_candidates,
    _STANDARD_MAX_PX,
    _FALLBACK_MAX_PX,
)
from src.ingestion.models import Chunk  # noqa: E402
from src.vision.adapter import (  # noqa: E402
    USEFUL_CLASSES,
    VisionResult,
    make_adapter,
)
from src.vision.cache import VisionCache  # noqa: E402


logger = logging.getLogger(__name__)


def _is_transport_error(err: str) -> bool:
    """True iff the error string looks like a temporary transport issue
    (429 / 5xx / network / quota) rather than a model-content failure.
    Transport errors are NOT cached — a re-run should retry them."""
    if not err:
        return False
    e = err.lower()
    return any(marker in e for marker in (
        "429", "resource_exhausted", "clienterror", "servererror",
        "500", "502", "503", "504", "quota", "rate", "timeout",
        "connection", "network",
    ))


TEST_COLLECTION_NAME = f"{settings.production_collection_name}_image_test"
TEST_CHROMA_DIR = _PROJECT_ROOT / "data" / "chroma_prod_image_test"
LOG_PATH = _PROJECT_ROOT / "eval" / "results" / "image_ingestion_test_log.json"


# --- Vision orchestration ---------------------------------------------------

def _classify_with_retry(
    adapter, cache: VisionCache, candidate: ImageCandidate,
) -> tuple[VisionResult, dict]:
    """
    Run stage 1 + (if useful) stage 2 with the two-attempt retry
    policy. Returns (result, telemetry).

    Attempt 1 renders at `_STANDARD_MAX_PX` (default). Attempt 2
    (only when attempt 1 yields empty/invalid extraction) re-renders
    at `_FALLBACK_MAX_PX` for a smaller payload — this dodged the
    empty-body failure in the micro-benchmark on 2 of 20 candidates.
    """
    telemetry = {"cache": "miss", "attempts": 0}

    # --- Cache check ---
    cached = cache.get(candidate.content_hash)
    if cached is not None:
        telemetry["cache"] = "hit"
        telemetry["attempts"] = cached.attempt
        return cached, telemetry

    # --- Stage 1: classify ---
    try:
        img_bytes = candidate.render_png(max_px=_STANDARD_MAX_PX)
    except Exception as e:
        # Pathological source image (unsupported pixmap format,
        # zero-byte payload, etc.). Never crash the whole run — mark
        # the candidate vision_failed so downstream still emits a
        # placeholder chunk pointing at the source page (rule 12.7).
        telemetry["attempts"] = 0
        result = VisionResult(
            classification="OTHER",
            vision_failed=True,
            error=f"render_png: {type(e).__name__}: {str(e)[:200]}",
        )
        cache.put(candidate.content_hash, result)
        return result, telemetry
    try:
        cls = adapter.classify(img_bytes)
    except Exception as e:
        telemetry["attempts"] = 1
        err = f"classify transport: {type(e).__name__}: {str(e)[:200]}"
        result = VisionResult(
            classification="OTHER",
            vision_failed=True,
            error=err,
        )
        # Transport-level failures (429 / 5xx / network) represent
        # temporary state, NOT the model refusing this specific image.
        # Do NOT cache them — a future run with fresh quota should
        # get a real classification. Content-level failures (empty
        # response body, bad JSON) still cache below in the extract
        # path because those repeat deterministically.
        if _is_transport_error(err):
            return result, telemetry
        cache.put(candidate.content_hash, result)
        return result, telemetry
    telemetry["attempts"] = 1

    # If not useful, cache and return — no extraction call.
    if cls not in USEFUL_CLASSES:
        result = VisionResult(classification=cls, attempt=1)
        cache.put(candidate.content_hash, result)
        return result, telemetry

    # --- Stage 2 attempt 1 ---
    result = adapter.extract(img_bytes, cls)
    if not result.vision_failed:
        # Store the actual classification returned by stage 1.
        result = _with_attempt(result, 1)
        cache.put(candidate.content_hash, result)
        return result, telemetry

    # --- Stage 2 attempt 2 at fallback resolution ---
    telemetry["attempts"] = 2
    smaller = candidate.render_png(max_px=_FALLBACK_MAX_PX)
    result2 = adapter.extract(smaller, cls)
    if not result2.vision_failed:
        result2 = _with_attempt(result2, 2)
        cache.put(candidate.content_hash, result2)
        return result2, telemetry

    # Both attempts failed. Materialise a vision_failed sentinel so
    # downstream code never has to guess (rule 12.7 — never silently
    # drop). Keep the original error for debugging.
    err = result2.error or result.error or "unknown vision failure"
    result = VisionResult(
        classification=cls,
        vision_failed=True,
        error=err,
        attempt=2,
    )
    # Same rule as stage-1: cache content-level failures (they repeat
    # deterministically) but NOT transport failures (fresh quota /
    # network on the next run may succeed).
    if not _is_transport_error(err):
        cache.put(candidate.content_hash, result)
    return result, telemetry


def _with_attempt(r: VisionResult, attempt: int) -> VisionResult:
    """Return a copy of `r` with `attempt` field overwritten."""
    from dataclasses import replace
    return replace(r, attempt=attempt)


# --- Embedder + Chroma helpers ----------------------------------------------

def _load_embedder() -> SentenceTransformer:
    logger.info("loading embedder: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def _get_test_collection():
    TEST_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(TEST_CHROMA_DIR))
    return client.get_or_create_collection(
        name=TEST_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _upsert_batch(collection, embedder, chunks: list[Chunk]) -> int:
    """Small local batcher — reuses embedder + chroma upsert like
    the production embedder but scoped to the image test collection."""
    if not chunks:
        return 0
    from src.embeddings.embedder import _chunk_to_chroma_record
    records = [_chunk_to_chroma_record(c) for c in chunks]
    ids = [r[0] for r in records]
    docs = [r[1] for r in records]
    metas = [r[2] for r in records]
    vecs = embedder.encode(docs, normalize_embeddings=True, convert_to_numpy=True).tolist()
    collection.upsert(ids=ids, documents=docs, embeddings=vecs, metadatas=metas)
    return len(chunks)


# --- Entry point ------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N unique candidates "
                         "(0 = no limit)")
    ap.add_argument("--dry-run", action="store_true",
                    help="run vision but skip Chroma upsert")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    # --- Enumerate candidates ---
    logger.info("Extracting image candidates from the corpus...")
    candidates, diag = extract_image_candidates()
    logger.info("candidate diag: %s", json.dumps(diag))
    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]
        logger.info("--limit %d applied → %d candidates", args.limit, len(candidates))

    cache = VisionCache()
    adapter = make_adapter()

    # --- Classify + extract each candidate ---
    counts = {
        "TABLE": 0, "CHART": 0, "INFOGRAPHIC": 0, "OTHER": 0,
        "DECORATIVE": 0, "vision_failed": 0,
    }
    n_cache_hits = 0
    n_vision_calls = 0
    per_candidate_log: list[dict] = []
    materialised: list[tuple[ImageCandidate, VisionResult]] = []

    t_vision_start = time.time()
    for i, cand in enumerate(candidates, start=1):
        result, tele = _classify_with_retry(adapter, cache, cand)

        if tele["cache"] == "hit":
            n_cache_hits += 1
        else:
            # Attempts is per-candidate stage-2 attempts; +1 for stage-1
            # only when we actually called classify (which is always
            # on cache miss unless stage-1 crashed).
            n_vision_calls += 1 + (tele["attempts"] if result.classification in USEFUL_CLASSES else 0)

        cls_key = "vision_failed" if result.vision_failed else result.classification
        counts[cls_key] = counts.get(cls_key, 0) + 1

        per_candidate_log.append({
            "content_hash": cand.content_hash,
            "cache": tele["cache"],
            "attempts": tele["attempts"],
            "classification": result.classification,
            "vision_failed": result.vision_failed,
            "n_occurrences": len(cand.occurrences),
            "rep": {
                "scheme": cand.representative.scheme,
                "filename": cand.representative.filename,
                "page": cand.representative.page,
            } if cand.representative else None,
            "error": result.error,
        })

        if result.vision_failed or result.classification in USEFUL_CLASSES:
            materialised.append((cand, result))

        if i % 10 == 0:
            logger.info(
                "  progress: %d/%d  cache_hits=%d  vision_calls=%d  "
                "table=%d chart=%d info=%d other=%d dec=%d failed=%d",
                i, len(candidates), n_cache_hits, n_vision_calls,
                counts["TABLE"], counts["CHART"], counts["INFOGRAPHIC"],
                counts["OTHER"], counts["DECORATIVE"], counts["vision_failed"],
            )
        # Respect rate limit ONLY on real calls, not cache hits.
        if tele["cache"] == "miss" and i < len(candidates):
            time.sleep(settings.vision_call_delay_s)
    t_vision_elapsed = time.time() - t_vision_start

    logger.info(
        "vision complete: %d cache hits, %d live calls, %.1fs elapsed",
        n_cache_hits, n_vision_calls, t_vision_elapsed,
    )

    # --- Materialise chunks ---
    all_chunks: list[Chunk] = []
    for cand, result in materialised:
        all_chunks.extend(make_image_chunks(cand, result))
    logger.info(
        "materialised %d chunks (from %d useful+failed candidates, "
        "avg %.1f occurrences/candidate)",
        len(all_chunks), len(materialised),
        (len(all_chunks) / len(materialised)) if materialised else 0.0,
    )

    # --- Upsert into scratch collection ---
    upserted = 0
    collection_size_after = 0
    if not args.dry_run and all_chunks:
        embedder = _load_embedder()
        collection = _get_test_collection()
        # Batch to avoid a giant single-call embed.
        batch_size = settings.embed_batch_size
        for start in range(0, len(all_chunks), batch_size):
            batch = all_chunks[start:start + batch_size]
            upserted += _upsert_batch(collection, embedder, batch)
        collection_size_after = collection.count()
        logger.info(
            "upserted %d image chunks; collection size after = %d",
            upserted, collection_size_after,
        )
    elif args.dry_run:
        logger.info("--dry-run: skipping embed/upsert")

    # --- Summary ---
    summary = {
        "candidate_diagnostics": diag,
        "processed_candidates": len(candidates),
        "cache_hits": n_cache_hits,
        "live_vision_calls": n_vision_calls,
        "vision_elapsed_s": round(t_vision_elapsed, 1),
        "counts_by_classification": counts,
        "n_useful_and_failed_candidates": len(materialised),
        "n_image_chunks": len(all_chunks),
        "chroma_upserted": upserted,
        "chroma_collection_size_after": collection_size_after,
        "chroma_collection_name": TEST_COLLECTION_NAME,
        "chroma_persist_dir": str(TEST_CHROMA_DIR),
        "model": settings.vision_model,
        "dry_run": args.dry_run,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        json.dumps({"summary": summary, "per_candidate": per_candidate_log},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print("=" * 72)
    print("IMAGE INGESTION SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print(f"\nfull log → {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
