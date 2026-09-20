"""
End-to-end retrieval test for the image ingestion pipeline (B5.7).

Runs a small set of queries hand-crafted to target CONTENT WE KNOW
lives in specific image chunks (identified by the vision micro-
benchmark). For each query:

  1. Embed with bge-small (same model used at ingestion time).
  2. Query the scratch collection `agri_schemes_prod_image_test`
     for top-K.
  3. Report:
       * whether ANY image chunk (source_type=image) landed in top-K
       * whether the highest-ranked image chunk contains the known
         answer tokens
       * for context: the top-1 by similarity, its source_type,
         scheme + page

Pass criterion: at least one image chunk retrieved and containing
answer-relevant tokens for each of the target queries. This is a
sanity gate, not a benchmark — the actual retrieval quality is
measured later in the extended golden-set eval.

The queries pick up known wins from the micro-benchmark:
  * NMEO-OilPalm p3 — state-wise oil-palm area map
  * NCCD p31 — sub-scheme reference table (NHM / HMNEH / NBM / NHB)
  * SMAM PIB p3 — SMAM progress infographic (Farm Machinery Banks etc.)

Read-only against Chroma.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from sentence_transformers import SentenceTransformer  # noqa: E402

from src.config import settings  # noqa: E402


TEST_COLLECTION_NAME = f"{settings.production_collection_name}_image_test"
TEST_CHROMA_DIR = _PROJECT_ROOT / "data" / "chroma_prod_image_test"
REPORT_PATH = _PROJECT_ROOT / "eval" / "results" / "image_e2e_retrieval_test_report.json"


# Each entry: (query, list of answer-relevance tokens — at least ONE must
# appear in the retrieved image chunk text for that query to pass).
# Tokens are lowercased before comparison so case doesn't matter.
QUERIES = [
    (
        "state-wise potential area for oil palm cultivation in India",
        ["oil palm", "andhra pradesh", "potential", "fruiting", "state"],
    ),
    (
        "how many Farm Machinery Banks have been established under SMAM",
        ["farm machinery", "machinery banks", "smam", "established"],
    ),
    (
        "which sub-schemes are under horticulture NHM HMNEH NBM NHB",
        ["nhm", "hmneh", "nbm", "nhb", "sub-scheme", "sub scheme"],
    ),
    (
        "cost norms and pattern of assistance for cold chain components",
        ["cost norm", "pattern of assistance", "cold chain", "norms"],
    ),
    (
        "SMAM Custom Hiring Centres established",
        ["custom hiring", "chc", "smam", "farm machinery"],
    ),
]

TOP_K = 5


def _run_query(collection, embedder, query: str, tokens: list[str]) -> dict:
    """Embed, retrieve, evaluate."""
    q_vec = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()
    resp = collection.query(
        query_embeddings=q_vec, n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    ids = resp.get("ids", [[]])[0]
    docs = resp.get("documents", [[]])[0]
    metas = resp.get("metadatas", [[]])[0]
    dists = resp.get("distances", [[]])[0]

    top_image_chunk = None
    top_image_rank = None
    n_image_in_topk = 0
    per_rank: list[dict] = []
    for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        st = meta.get("source_type", "?")
        is_img = st == "image"
        if is_img:
            n_image_in_topk += 1
            if top_image_chunk is None:
                top_image_chunk = (cid, doc, meta)
                top_image_rank = rank
        per_rank.append({
            "rank": rank,
            "id": cid,
            "distance": round(float(dist), 4),
            "source_type": st,
            "scheme": meta.get("scheme"),
            "filename": meta.get("source_filename"),
            "page": meta.get("page_start"),
            "vision_content_type": meta.get("vision_content_type"),
            "preview": (doc or "")[:200].replace("\n", " "),
        })

    # Token-hit check on the top image chunk
    hit_tokens: list[str] = []
    if top_image_chunk is not None:
        doc_lower = (top_image_chunk[1] or "").lower()
        hit_tokens = [t for t in tokens if t in doc_lower]

    passed = bool(top_image_chunk is not None and hit_tokens)

    return {
        "query": query,
        "answer_tokens": tokens,
        "n_image_chunks_in_top5": n_image_in_topk,
        "top_image_rank": top_image_rank,
        "top_image_token_hits": hit_tokens,
        "passed": passed,
        "per_rank": per_rank,
    }


def main() -> int:
    if not TEST_CHROMA_DIR.exists():
        print(f"missing: {TEST_CHROMA_DIR}", file=sys.stderr)
        print("run scripts/run_image_ingestion_test.py first", file=sys.stderr)
        return 2

    print(f"[e2e] embedder: {settings.embedding_model}")
    embedder = SentenceTransformer(settings.embedding_model)

    client = chromadb.PersistentClient(path=str(TEST_CHROMA_DIR))
    try:
        collection = client.get_collection(TEST_COLLECTION_NAME)
    except Exception as e:
        print(f"missing collection {TEST_COLLECTION_NAME}: {e}", file=sys.stderr)
        return 2
    size = collection.count()
    print(f"[e2e] collection {TEST_COLLECTION_NAME} size = {size}")
    if size == 0:
        print("collection empty — run the ingestion runner first", file=sys.stderr)
        return 2

    results: list[dict] = []
    n_pass = 0
    for query, tokens in QUERIES:
        print(f"\n--- query: {query}")
        r = _run_query(collection, embedder, query, tokens)
        results.append(r)
        n_pass += 1 if r["passed"] else 0
        status = "PASS" if r["passed"] else "FAIL"
        print(f"    {status}: n_image_in_top5={r['n_image_chunks_in_top5']}  "
              f"top_image_rank={r['top_image_rank']}  "
              f"token_hits={r['top_image_token_hits']}")
        for row in r["per_rank"]:
            marker = "★" if row["source_type"] == "image" else " "
            print(
                f"    {marker} r{row['rank']} d={row['distance']:.3f} "
                f"[{row['source_type']:<8}] {row['scheme']}/"
                f"{(row['filename'] or '')[:40]} p{row['page']}"
            )

    summary = {
        "collection_size": size,
        "n_queries": len(QUERIES),
        "n_pass": n_pass,
        "n_fail": len(QUERIES) - n_pass,
        "overall_pass": n_pass == len(QUERIES),
        "per_query": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print("=" * 72)
    print(f"E2E RETRIEVAL TEST: {n_pass}/{len(QUERIES)} passed")
    print("=" * 72)
    print(f"report → {REPORT_PATH}")
    return 0 if n_pass == len(QUERIES) else 1


if __name__ == "__main__":
    sys.exit(main())
