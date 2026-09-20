"""
Production retrieval pipeline (C1).

One callable: `run_production_query(query, config)` →
`ProductionResult`. Composes existing modules — no new retrieval or
generation code lives here. Composition order matches the owner-locked
production plan (DECISIONS.md 2026-09-16, item 6):

    query
      → semantic top-N + BM25 top-N
      → RRF fusion  (via hybrid_search)
      → cross-encoder rerank the fused pool
      → grounded generation with citations

Guardrails wired in-line (see `src/production/guardrails.py`):
  * confidence-threshold check after rerank — refuse rather than
    let the generator hallucinate over marginal context.
  * chunk-injection hardening is delegated to the generator prompt
    (it already treats retrieved context as data; production-time
    we surface the sentinel-wrapped chunk ids so a reviewer can
    see the boundaries).

This function is thread-safe assuming the underlying retriever /
reranker singletons are (they lazily init global instances behind a
lock — see retriever.py / reranker.py). One process, many concurrent
requests: fine. Multi-process: each process has its own singleton
copy, no shared state.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.config import settings
from src.generation.generator import generate
from src.ingestion.models import GenerationResult, RetrievalResult
from src.production.guardrails import (
    ConfidenceCheckResult,
    check_retrieval_confidence,
)
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank


logger = logging.getLogger(__name__)


# --- Value objects ----------------------------------------------------------

@dataclass(frozen=True)
class Citation:
    """
    Shown to the user. One per retrieved chunk that survived reranking.
    Metadata only — no chunk text. Text is available in
    `ProductionResult.retrieved_chunks` for a debug / audit view.
    """

    chunk_id: str
    scheme: str
    source_filename: str
    source_type: str  # "pdf" | "workflow" | "image"
    page_start: int | None
    page_end: int | None
    is_ocr_source: bool
    parent_subfolder: str | None
    # Image-only bookkeeping. Populated on `source_type == "image"`.
    vision_content_type: str | None
    vision_failed: bool


@dataclass
class ProductionResult:
    """
    Full answer + audit trail. This is what the FastAPI endpoint
    serialises. `refused` distinguishes an honest refusal (low
    confidence, empty retrieval) from a generated answer — the
    front-end can present them differently.
    """

    query: str
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    retrieved_chunks: list[RetrievalResult]
    # Pipeline stage-level timing so the front-end / logs can show
    # where the latency budget went (retrieval-heavy vs generation-heavy
    # is a very different diagnosis).
    latency_hybrid_ms: int
    latency_rerank_ms: int
    latency_generation_ms: int
    latency_total_ms: int
    # Guardrail state — useful for observability. `confidence_check` is
    # always populated (even on happy path, so we can log the score).
    confidence_check: ConfidenceCheckResult
    # LLM accounting. Zeroed when we refused pre-generation.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    retries_taken: int = 0
    finish_reason: str = "not_called"
    llm_error: str | None = None


# --- Config knob for the confidence gate ------------------------------------

# Lives here rather than in `src/config.py` because it is production-only.
# Cross-encoder ms-marco-MiniLM scores range roughly [-11, +11]; the
# median useful chunk on the ablation golden set sits around +2 to +5.
# A threshold of -2 refuses truly weak matches without gating the
# borderline-useful ones. Owner tuning: expose as
# settings.production_confidence_threshold if we ever want to sweep it.
DEFAULT_MIN_RERANK_SCORE: float = -2.0


# --- Refusal template -------------------------------------------------------

REFUSAL_LOW_CONFIDENCE = (
    "The corpus does not contain information that clearly matches this "
    "question. I cannot answer without a strong source match — please "
    "rephrase or try a more specific query."
)
REFUSAL_EMPTY_RETRIEVAL = (
    "No relevant source content was retrieved for this question. "
    "The corpus may not cover this topic."
)


# --- Helpers ----------------------------------------------------------------

def _citation_from(rr: RetrievalResult) -> Citation:
    """Chunk metadata → the citation shape shown to the user."""
    return Citation(
        chunk_id=rr.chunk_id,
        scheme=rr.scheme,
        source_filename=rr.source_filename,
        source_type=rr.source_type,
        page_start=rr.page_start,
        page_end=rr.page_end,
        is_ocr_source=rr.is_ocr_source,
        parent_subfolder=rr.parent_subfolder,
        # These live on `Chunk` metadata upstream; RetrievalResult
        # currently doesn't surface them. Left None on this pass — a
        # follow-up can plumb them through if the UI wants richer
        # image-chunk annotations.
        vision_content_type=None,
        vision_failed=False,
    )


def _empty_generation_result() -> tuple[int, int, int, int, str, str | None]:
    """When we refuse before calling the LLM, zero out the accounting
    tuple that `run_production_query` unpacks."""
    return (0, 0, 0, 0, "not_called", None)


# --- Entry point ------------------------------------------------------------

def run_production_query(
    query: str,
    config=settings,
    min_rerank_score: float | None = None,
) -> ProductionResult:
    """
    Execute the full production pipeline for one query.

    Parameters
    ----------
    query
        The user's original question. Pre-validation (empty / length
        cap / prompt-injection) is the CALLER'S responsibility — this
        function assumes the query has already passed input guardrails.
    config
        Injected settings singleton. The API surface mutates
        `settings.chroma_persist_dir` + `chroma_collection_name` at
        startup so retriever / bm25 / reranker read from the prod
        collection.
    min_rerank_score
        Confidence-gate threshold. `None` → uses
        `DEFAULT_MIN_RERANK_SCORE`.

    Returns
    -------
    A `ProductionResult`. `refused=True` means the pipeline chose to
    refuse rather than generate; `answer` in that case is a fixed
    refusal template (see `REFUSAL_*` above). Retrieved chunks are
    STILL returned on refusal so the caller can show the user what
    the retriever surfaced — refusing silently would be worse UX.
    """
    threshold = (
        min_rerank_score if min_rerank_score is not None
        else DEFAULT_MIN_RERANK_SCORE
    )
    t0 = time.monotonic()

    # --- Retrieval: semantic + BM25 fused ---
    #
    # `hybrid_search` internally runs semantic top-N + BM25 top-N and
    # RRF-fuses them. We request enough fused candidates to give the
    # reranker headroom — `reranker_top_n` (default 20) is what the
    # reranker will re-score; the reranker then keeps `reranker_top_k`
    # (default 5). Both come from config, unchanged from ablation.
    t_hybrid_start = time.monotonic()
    fused = hybrid_search(query, top_k=config.reranker_top_n, config=config)
    latency_hybrid_ms = int((time.monotonic() - t_hybrid_start) * 1000)

    if not fused:
        # Corpus / index empty, or a query that neither retriever
        # matched. Refuse — no generation call.
        (pt, ct, rt, rets, fr, err) = _empty_generation_result()
        return ProductionResult(
            query=query,
            answer=REFUSAL_EMPTY_RETRIEVAL,
            refused=True,
            refusal_reason="empty retrieval",
            citations=[],
            retrieved_chunks=[],
            latency_hybrid_ms=latency_hybrid_ms,
            latency_rerank_ms=0,
            latency_generation_ms=0,
            latency_total_ms=int((time.monotonic() - t0) * 1000),
            confidence_check=ConfidenceCheckResult(
                passed=False, top_score=None, threshold=threshold,
                reason="empty retrieval",
            ),
            prompt_tokens=pt, completion_tokens=ct,
            reasoning_tokens=rt, retries_taken=rets,
            finish_reason=fr, llm_error=err,
        )

    # --- Rerank the fused pool ---
    t_rerank_start = time.monotonic()
    reranked = rerank(
        query, fused, top_k=config.reranker_top_k, config=config
    )
    latency_rerank_ms = int((time.monotonic() - t_rerank_start) * 1000)

    # --- Confidence gate ---
    conf = check_retrieval_confidence(reranked, threshold, score_attr="rerank_score")
    if not conf.passed:
        logger.info("refusing on low confidence: %s", conf.reason)
        return ProductionResult(
            query=query,
            answer=REFUSAL_LOW_CONFIDENCE,
            refused=True,
            refusal_reason=f"low confidence: {conf.reason}",
            citations=[_citation_from(rr) for rr in reranked],
            retrieved_chunks=reranked,
            latency_hybrid_ms=latency_hybrid_ms,
            latency_rerank_ms=latency_rerank_ms,
            latency_generation_ms=0,
            latency_total_ms=int((time.monotonic() - t0) * 1000),
            confidence_check=conf,
            prompt_tokens=0, completion_tokens=0,
            reasoning_tokens=0, retries_taken=0,
            finish_reason="refused_low_confidence", llm_error=None,
        )

    # --- Generation ---
    t_gen_start = time.monotonic()
    llm_error: str | None = None
    gen_result: GenerationResult | None = None
    try:
        gen_result = generate(query, reranked, config=config)
    except Exception as exc:  # pragma: no cover
        # LLM failure handling (production rule 12.7): never crash;
        # honestly report the failure to the caller.
        llm_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        logger.error("generation failed: %s", llm_error)
    latency_generation_ms = int((time.monotonic() - t_gen_start) * 1000)

    if gen_result is None:
        return ProductionResult(
            query=query,
            answer="Generation failed. Please try again shortly.",
            refused=True,
            refusal_reason=f"llm error: {llm_error}",
            citations=[_citation_from(rr) for rr in reranked],
            retrieved_chunks=reranked,
            latency_hybrid_ms=latency_hybrid_ms,
            latency_rerank_ms=latency_rerank_ms,
            latency_generation_ms=latency_generation_ms,
            latency_total_ms=int((time.monotonic() - t0) * 1000),
            confidence_check=conf,
            prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
            retries_taken=0, finish_reason="llm_error",
            llm_error=llm_error,
        )

    # --- Happy path ---
    return ProductionResult(
        query=query,
        answer=gen_result.answer,
        refused=False,
        refusal_reason=None,
        citations=[_citation_from(rr) for rr in reranked],
        retrieved_chunks=reranked,
        latency_hybrid_ms=latency_hybrid_ms,
        latency_rerank_ms=latency_rerank_ms,
        latency_generation_ms=latency_generation_ms,
        latency_total_ms=int((time.monotonic() - t0) * 1000),
        confidence_check=conf,
        prompt_tokens=gen_result.prompt_tokens,
        completion_tokens=gen_result.completion_tokens,
        reasoning_tokens=gen_result.reasoning_tokens,
        retries_taken=gen_result.retries_taken,
        finish_reason=gen_result.finish_reason,
        llm_error=None,
    )
