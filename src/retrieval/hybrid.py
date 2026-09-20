"""
Hybrid (semantic + BM25) retriever for the Agriculture Schemes RAG project.

Phase 5, fix #2 (CLAUDE.md §7). Runs the semantic retriever and the
BM25 retriever independently, fuses their ranked outputs with
Reciprocal Rank Fusion (RRF), and returns the top-K fused hits. The
two retrievers score on different scales (cosine similarity is a
bounded distance, BM25 is an unbounded relevance score), so we fuse
on RANKS, not raw scores — RRF is the standard fusion method for
exactly this reason.

Why this exists (interview framing to preserve in the code):

  Semantic retrieval misses exact-code lookups (scheme names,
  subsidy percentages, circular numbers). BM25 misses paraphrase and
  topical similarity. The failure modes are complementary, so fusing
  the two gives a strictly wider recall floor than either alone —
  provided the fusion doesn't itself introduce bias. RRF is popular
  precisely because it doesn't: it depends only on rank position, so
  a semantic score of 0.83 and a BM25 score of 12.4 don't need to be
  normalised into a common scale first (any normalisation choice is
  itself a knob you'd need to defend).

  The base RAG failure this fixes maps to CLAUDE.md §7 row #2 —
  "meaning-search misses exact codes." The reranker (Phase 5, fix #1)
  is orthogonal: it re-orders whatever pool it's given, but if the
  correct chunk was never in the semantic top-N in the first place,
  reranking can't bring it back. Hybrid is what enlarges the pool.

The fusion in one line:

    rrf_score(chunk) = sum over retrievers r of  1 / (k + rank_r(chunk))

where `rank_r(chunk)` is the chunk's rank in retriever r's top-N
(or +inf if the chunk wasn't in r's pool — contributing 0 to the
sum). `k` is a smoothing constant; the canonical TREC paper value is
60, and we use that.

Three decisions worth naming out loud so downstream edits don't
unpick them by accident:

  1. **The two retrievers run INDEPENDENTLY and are fused after the
     fact.** Neither retriever sees the other's scores; neither pool
     is filtered by the other. This is what makes the row a clean
     "semantic + BM25" measurement — a leaky abstraction (e.g. BM25
     filtered by a semantic threshold) would confound the ablation.

  2. **Pool sizes are held equal by default** (`hybrid_semantic_top_n
     == hybrid_bm25_top_n == 20`). Asymmetric pools would tilt the
     fusion toward one retriever without being visible in the fusion
     constant. If a later experiment shows an asymmetric pool
     materially helps, that becomes its own labelled variant.

  3. **The union — not just the intersection — is fused.** A chunk
     that appears in only one retriever's pool still gets a fused
     rank (its contribution from the missing retriever is 0). Using
     only the intersection would collapse recall on precisely the
     queries hybrid is meant to fix (a scheme code that only BM25
     finds, or a paraphrased query that only semantic finds).
"""

from __future__ import annotations

import logging

from src.config import settings
from src.ingestion.models import RetrievalResult
from src.retrieval.bm25_search import bm25_search
from src.retrieval.retriever import retrieve
from src.retrieval.rrf import rrf_fuse


logger = logging.getLogger(__name__)


# --- Public entry point -----------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int | None = None,
    config=settings,
) -> list[RetrievalResult]:
    """
    Fuse the semantic and BM25 pools for `query`, return top-K by RRF.

    Parameters
    ----------
    query
        Natural-language question. Passed verbatim to both retrievers;
        any query transformation must happen upstream (that would be
        Phase 5 fix #3 — a separate ablation row).
    top_k
        How many fused results to return. `None` uses
        `config.hybrid_top_k` (the count the generator sees). Callers
        wanting the full fused union (e.g. the smoke test's per-
        candidate audit table) can pass a large value.
    config
        Injectable so tests / notebooks can point at custom settings.

    Returns
    -------
    A list of `RetrievalResult`, best-first (highest rrf_score first,
    `rank` starting at 1), of length up to `top_k`. Each result carries
    the full provenance audit:

      * `similarity_score` + `semantic_rank`  — if the chunk was in
        the semantic top-N. `None` on both fields if it was BM25-only.
      * `bm25_score` + `bm25_rank`            — if the chunk was in
        the BM25 top-N. `None` on both fields if it was semantic-only.
      * `rrf_score`                           — always populated. It
        is the sum of the two `1/(k+rank)` contributions, zero-filled
        for the retriever that missed the chunk.
      * `rank`                                — the post-RRF-fusion
        1-indexed rank.

    Empty-input handling: if BOTH retrievers return zero candidates
    (e.g. an empty corpus at bm25 side + a query that returns nothing
    from Chroma), an empty list is returned. If exactly one returns
    zero, the other's results are ranked by RRF over a single-retriever
    pool — mathematically identical to that retriever's own ranking,
    which is the honest fallback.
    """
    if top_k is None:
        top_k = config.hybrid_top_k

    rrf_k = config.hybrid_rrf_k

    semantic_hits = retrieve(
        query, top_k=config.hybrid_semantic_top_n, config=config
    )
    bm25_hits = bm25_search(
        query, top_k=config.hybrid_bm25_top_n, config=config
    )

    # Lookup dicts by chunk_id — used AFTER fusion to decorate each
    # fused result with per-retriever provenance (semantic_rank,
    # bm25_rank, similarity_score, bm25_score). The fuser itself
    # doesn't need these: rrf_fuse operates purely on rank position
    # and doesn't know or care which retriever produced which list.
    semantic_by_id: dict[str, RetrievalResult] = {c.chunk_id: c for c in semantic_hits}
    bm25_by_id: dict[str, RetrievalResult] = {c.chunk_id: c for c in bm25_hits}

    # RRF fusion. Semantic is passed FIRST so first-occurrence-wins
    # (see rrf_fuse invariant) prefers the semantic record as the
    # source of hydrated fields (text, metadata). This preference is
    # stylistic — semantic_by_id[cid] and bm25_by_id[cid] read from
    # the same Chroma record for the same chunk_id, so their
    # metadata is identical — but preserving the same ordering
    # keeps this refactor bit-identical against the pre-extraction
    # hybrid_search behaviour.
    fused = rrf_fuse(
        ranked_lists=[semantic_hits, bm25_hits],
        top_k=top_k,
        rrf_k=rrf_k,
    )

    # Decoration pass. rrf_fuse sets `rank` and `rrf_score`; every
    # other field is copied from the first-occurrence source. That
    # means the semantic-side hit's `similarity_score` is preserved
    # (correct), but `semantic_rank`, `bm25_rank`, and `bm25_score`
    # are None on those records (semantic-side records don't know
    # about the BM25 pool). Overwrite all four provenance fields
    # explicitly here so hybrid_search's contract — every fused
    # result carries the FULL cross-retriever audit trail — holds
    # regardless of which pool's record was used as the hydration
    # source. This step also handles the symmetric case (BM25-only
    # chunk needing `similarity_score = None`).
    results: list[RetrievalResult] = []
    for hit in fused:
        cid = hit.chunk_id
        s_hit = semantic_by_id.get(cid)
        b_hit = bm25_by_id.get(cid)
        results.append(
            hit.model_copy(update={
                "semantic_rank": s_hit.rank if s_hit is not None else None,
                "bm25_rank": b_hit.rank if b_hit is not None else None,
                "similarity_score": (
                    s_hit.similarity_score if s_hit is not None else None
                ),
                "bm25_score": (
                    b_hit.bm25_score if b_hit is not None else None
                ),
            })
        )

    union_size = len(set(semantic_by_id.keys()) | set(bm25_by_id.keys()))
    overlap_size = len(set(semantic_by_id.keys()) & set(bm25_by_id.keys()))
    logger.info(
        "hybrid_search(query=%r, top_k=%d) → %d results "
        "(union=%d semantic=%d bm25=%d overlap=%d, top rrf=%.5f)",
        query, top_k, len(results),
        union_size, len(semantic_by_id), len(bm25_by_id), overlap_size,
        results[0].rrf_score if results else 0.0,
    )
    return results
