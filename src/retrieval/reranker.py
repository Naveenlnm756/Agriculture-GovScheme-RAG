"""
Cross-encoder reranker for the Agriculture Schemes RAG project.

Phase 5, fix #1 (CLAUDE.md §7). Standalone downstream reordering step
that takes candidates from the semantic retriever and re-scores each
against the query using a cross-encoder, then returns the top-k by
cross-encoder score.

Why this exists (interview framing to preserve in the code):

  The base semantic retriever answers "which chunks are geometrically
  closest to the question in embedding space?" That is a fast,
  cheap-per-query similarity search. It is also approximate — the
  question and each candidate chunk get encoded INDEPENDENTLY, and
  their relevance is inferred from how close the two independent
  encodings land. For most questions that's good enough; for questions
  where the *interaction* between query terms and chunk terms matters
  (e.g. "for a rotavator under SMAM in 2018" — where the model needs
  to weigh three constraints jointly), the "nearest 5" out of the
  semantic index frequently isn't the same as the "best 5."

  A cross-encoder scores (query, chunk) as a JOINT input rather than
  two independent encodings. It's slower (O(n) per query, not O(1))
  but much better at "given this specific query, how relevant is this
  specific chunk?" — which is exactly the question a downstream
  generator needs answered accurately for the top-5 it will see.

  The pattern is: semantic retriever fetches a top-N *candidate pool*
  (config.reranker_top_n, default 20), the reranker re-scores all N
  and keeps the top-K (config.reranker_top_k, default 5). N=20 is a
  cheap-enough pool to score end-to-end on CPU in <500ms with
  ms-marco-MiniLM-L-6-v2; K=5 matches the baseline generator context
  size so the only variable this ablation row studies is the
  SELECTION of the 5, not their count.

Two decisions worth naming out loud so downstream edits don't unpick
them by accident:

  1. **The cross-encoder model is loaded exactly once per process** via
     a lazy module-level cache. CrossEncoder loads take 3-5s and hold
     a few hundred MB of weights; reloading per query would make the
     API and the eval harness unusable.

  2. **The reranker does NOT retrieve.** It takes candidates in (from
     the semantic retriever) and returns a reordered subset out. This
     mirrors the same discipline that keeps the generator separate from
     retrieval — each stage is measured independently by the ablation
     harness, and mashing them together would prevent isolating the
     reranker's contribution.

  3. **similarity_score is preserved through reranking.** The reranker
     writes `rerank_score` and updates `rank` + `original_rank`, but
     leaves `similarity_score` alone. That preserves the audit trail
     — a downstream reviewer can see both "what the semantic retriever
     thought" (similarity_score, original_rank) and "what the
     cross-encoder thought" (rerank_score, rank) on the same row.
"""

from __future__ import annotations

import logging
from threading import Lock

from sentence_transformers import CrossEncoder

from src.config import settings
from src.ingestion.models import RetrievalResult


logger = logging.getLogger(__name__)


# --- Lazy singleton ---------------------------------------------------------
# Module-level cache so the CrossEncoder is loaded exactly once per
# process. A Lock guards first-time construction against the (unlikely
# but real) case of two threads calling rerank() at once before either
# has warmed the cache.
_reranker: CrossEncoder | None = None
_lock = Lock()


def _load_reranker(config=settings) -> CrossEncoder:
    """
    Return a process-cached CrossEncoder.

    Model name is read from config (not hardcoded) so a future model
    swap is a single-point change. `cross-encoder/ms-marco-MiniLM-L-6-v2`
    is the industry-standard small reranker — well-known, easy to
    defend in an interview, cheap on CPU (~500ms for 20 pairs).
    """
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                logger.info(
                    "Loading cross-encoder reranker: %s", config.reranker_model
                )
                _reranker = CrossEncoder(config.reranker_model)
                logger.info("Reranker ready.")
    return _reranker


# --- Public entry point -----------------------------------------------------

def rerank(
    query: str,
    candidates: list[RetrievalResult],
    top_k: int | None = None,
    config=settings,
) -> list[RetrievalResult]:
    """
    Re-score `candidates` against `query` with a cross-encoder and
    return the top-k, best-first.

    Parameters
    ----------
    query
        Natural-language question. Passed to the cross-encoder as the
        first half of the (query, chunk) pair — never mutated, never
        preprocessed. Any query transformation must happen upstream.
    candidates
        The semantic retriever's output. Each candidate carries a
        pre-rerank `rank` and `similarity_score`; both are preserved
        (see below).
    top_k
        How many results to return. `None` uses `config.reranker_top_k`.
    config
        Injectable so tests / notebooks can point at custom settings.

    Returns
    -------
    A new list of `RetrievalResult`, best-first, of length
    `min(top_k, len(candidates))`. Each returned item is a COPY of the
    corresponding candidate with three fields updated:
      * `rerank_score`  — the cross-encoder score for (query, chunk).
                          Higher = more relevant.
      * `original_rank` — the pre-rerank rank the candidate had in the
                          semantic-retriever output. Preserves the
                          audit trail (`rank` becomes the new post-rerank
                          rank; `original_rank` records where the chunk
                          came from).
      * `rank`          — the new 1-indexed rank in the reranked list.

    `similarity_score` is left untouched so downstream code can see
    both scores side by side.

    Empty-candidate path: returns [] without loading the model. This
    matters because the eval harness calls rerank() once per question;
    a question that retrieves zero candidates (a possibility on a
    small collection) should not incur the 3-5s model load.
    """
    if not candidates:
        return []

    if top_k is None:
        top_k = config.reranker_top_k

    model = _load_reranker(config)

    # CrossEncoder.predict takes a list of (text_a, text_b) pairs and
    # returns a numpy array of relevance scores. Batching all pairs into
    # one predict() call is materially faster than one call per pair —
    # cross-encoders are transformer inferences and batching amortises
    # the per-call Python/PyTorch overhead. We do NOT need to sort or
    # slice inside the model call; we do it in pure Python below so the
    # transformation from "scored candidates" to "reranked list" stays
    # obvious to a reader.
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    # Score-annotate each candidate. `model.predict` returns numpy
    # floats; cast to Python float so downstream serialisation (JSON,
    # pydantic) doesn't get confused by numpy types.
    scored: list[tuple[float, RetrievalResult]] = []
    for cand, raw_score in zip(candidates, scores):
        scored.append((float(raw_score), cand))

    # Sort descending by rerank_score. Stable sort — for tied scores
    # the pre-rerank order is preserved, which is the honest fallback
    # (a tie means the cross-encoder couldn't distinguish them, so we
    # should keep the semantic retriever's ordering).
    scored.sort(key=lambda pair: pair[0], reverse=True)

    # Build the returned list. Each item is a COPY of the candidate
    # with rank / original_rank / rerank_score updated; we never mutate
    # the input list (the caller may still want to inspect it for
    # audit / smoke-test purposes).
    reranked: list[RetrievalResult] = []
    for new_rank, (score, cand) in enumerate(scored[:top_k], start=1):
        reranked.append(
            cand.model_copy(update={
                "rerank_score": score,
                "original_rank": cand.rank,
                "rank": new_rank,
            })
        )

    logger.info(
        "rerank(query=%r, n_candidates=%d, top_k=%d) → %d results "
        "(top score=%.4f, bottom kept score=%.4f)",
        query, len(candidates), top_k, len(reranked),
        reranked[0].rerank_score if reranked else 0.0,
        reranked[-1].rerank_score if reranked else 0.0,
    )
    return reranked
