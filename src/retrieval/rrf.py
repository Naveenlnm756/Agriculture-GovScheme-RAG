"""
Reciprocal Rank Fusion primitive — the single RRF implementation for the project.

Used by two callers:
  * `src/retrieval/hybrid.py` fuses [semantic, BM25] into the hybrid row.
  * The `query_transform` pipeline mode in `eval/run_eval.py` fuses
    [original, rewrite_1, rewrite_2, ..., rewrite_N] into the
    query_transform row.

The primitive doesn't know what produced each ranked list. It fuses by
1-indexed rank position and returns fused results with `rank` and
`rrf_score` set. Per-list provenance (which retriever a chunk came
from, its per-list rank, its raw score) is the caller's job to
decorate afterwards — hybrid.py does this for its two-list case;
query_transform does not decorate at all because per-list-named
fields (`rewrite_0_rank`, `rewrite_1_rank`, ...) do not scale to
N sub-queries.

The fusion in one line:

    rrf_score(chunk) = sum over lists L containing chunk of  1 / (rrf_k + rank_L(chunk))

`rrf_k` smooths the rank contribution — the canonical TREC paper
value is 60, and callers pass their own (`hybrid_rrf_k=60`,
`query_transform_rrf_k=60`) so the two rows are independently
switchable per CLAUDE.md §5.

Chunks missing from a given list contribute 0 to the sum (they do
NOT contribute `1/(rrf_k + infinity)` — same math, but written
this way for clarity to a reader of the loop).

Determinism guarantees (important for eval reproducibility):
  * Tie-breaking on equal RRF scores is by ascending `chunk_id`
    (string compare). Same behaviour hybrid.py had before the
    extraction, so the equivalence check between old and new
    hybrid stays bit-identical.
  * Iteration order over the union of chunk_ids is not source-
    dependent (we sort the fused list at the end).
"""

from __future__ import annotations

import logging

from src.ingestion.models import RetrievalResult


logger = logging.getLogger(__name__)


def rrf_fuse(
    ranked_lists: list[list[RetrievalResult]],
    top_k: int,
    rrf_k: int,
) -> list[RetrievalResult]:
    """
    Fuse N ranked lists by Reciprocal Rank Fusion; return the top-K.

    Parameters
    ----------
    ranked_lists
        A list of ranked candidate lists. Each inner list is
        best-first with `RetrievalResult.rank` set to that list's
        local 1-indexed position. Lists may be empty; empty lists
        contribute nothing to the fusion.
    top_k
        How many fused results to return.
    rrf_k
        The RRF smoothing constant. Callers pass their own
        (`hybrid_rrf_k` or `query_transform_rrf_k`) — kept as a
        parameter, not a default, so the source of the value is
        visible at every call site.

    Returns
    -------
    A `list[RetrievalResult]` of length up to `top_k`, best-first,
    with two fields overwritten by this function:

      * `rank` — the post-fusion 1-indexed rank.
      * `rrf_score` — the fused score (sum of `1/(rrf_k + rank_L)`
        across every input list L that contained the chunk).

    Every OTHER field is copied verbatim from the first ranked list
    (in input order) that carried the chunk — see the invariant
    below. This function does NOT clear per-list provenance fields
    (`semantic_rank`, `bm25_rank`, `similarity_score`, `bm25_score`,
    `rerank_score`, `original_rank`); the caller either wants them
    preserved from the source list (query_transform: irrelevant,
    all sources are semantic) or overwrites them in a post-
    processing decoration step (hybrid_search does this).

    Invariant (LOAD-BEARING — do not violate silently)
    --------------------------------------------------
    For any given `chunk_id`, EVERY input list that carries that
    chunk_id must carry IDENTICAL hydrated fields (`text`, `scheme`,
    `source_filename`, `page_start`, ..., i.e. every field except
    `rank` and the per-retriever score/rank fields). The
    first-occurrence-wins policy for hydrated fields is only safe
    under this invariant.

    Both current callers satisfy it trivially: they read from the
    same Chroma collection via `retrieve()` or `bm25_search()`,
    which returns records hydrated identically for the same
    chunk_id. A future caller that mixes lists from DIFFERENT
    corpora, DIFFERENT chunk stores, or DIFFERENT versions of the
    same corpus WOULD violate the invariant — at which point this
    function's semantics are undefined and the correct fix is to
    surface the disagreement (a new argument, an assertion, or a
    dedicated multi-source fuser), NOT to silently prefer one
    version over another.

    Empty-input handling
    --------------------
    If `ranked_lists` is empty, or every inner list is empty, an
    empty list is returned. If exactly one list is non-empty, the
    result is that list's own ranking (mathematically identical
    because RRF over one input reduces to sorting by that input's
    ranks). No caller currently relies on this, but it is the
    honest fallback: a single-list fusion is that list.
    """
    if not ranked_lists:
        return []

    # 1) Compute per-chunk RRF contribution across every list.
    #    Also remember the FIRST occurrence of each chunk_id (in
    #    input-list order, then in that list's rank order) so we
    #    can hydrate the fused result from a stable source.
    rrf_by_id: dict[str, float] = {}
    first_source_by_id: dict[str, RetrievalResult] = {}

    for ranked_list in ranked_lists:
        for hit in ranked_list:
            cid = hit.chunk_id
            # RRF contribution from this list. Uses the hit's own
            # `rank` (not enumerate position) so callers can pass a
            # pre-sliced or pre-filtered list without our making
            # assumptions about their integer sequence.
            contrib = 1.0 / (rrf_k + hit.rank)
            rrf_by_id[cid] = rrf_by_id.get(cid, 0.0) + contrib
            # First-occurrence-wins for hydrated fields — see the
            # invariant in the docstring above.
            if cid not in first_source_by_id:
                first_source_by_id[cid] = hit

    if not rrf_by_id:
        return []

    # 2) Sort by descending RRF score with a deterministic tiebreak
    #    on ascending chunk_id. Matches hybrid.py's pre-extraction
    #    behaviour byte-for-byte, which is what the equivalence
    #    check relies on.
    ordered_ids = sorted(
        rrf_by_id.keys(),
        key=lambda cid: (-rrf_by_id[cid], cid),
    )

    # 3) Materialise the top-K by copying the first-source record
    #    and overwriting only `rank` and `rrf_score`.
    results: list[RetrievalResult] = []
    for new_rank, cid in enumerate(ordered_ids[:top_k], start=1):
        source = first_source_by_id[cid]
        results.append(
            source.model_copy(update={
                "rank": new_rank,
                "rrf_score": rrf_by_id[cid],
            })
        )

    logger.debug(
        "rrf_fuse(n_lists=%d, top_k=%d, rrf_k=%d) → %d results "
        "from union of %d chunks",
        len(ranked_lists), top_k, rrf_k, len(results), len(rrf_by_id),
    )
    return results
