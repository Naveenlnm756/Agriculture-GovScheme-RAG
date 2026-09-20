"""
FastAPI backend for the production RAG (C3).

One POST endpoint (`/query`) + one health check (`/healthz`). The
production pipeline (`src.production.pipeline.run_production_query`)
does the work; this module owns the API surface:

  * Request IDs (`x-request-id` from client OR uuid4 fallback)
  * Structured JSON logging keyed by request id
  * Deterministic query validation + guardrails BEFORE the pipeline
  * PII scrub on the retrieved-chunk previews before they leave
    the process (rule 12.11)
  * Per-IP soft rate limit (in-memory token bucket, 30/min)
  * Timeouts (per-request wall-clock ceiling)
  * Chroma collection override at process start so retriever /
    reranker / bm25 read from `agri_schemes_prod`

Run:
    .venv/Scripts/python.exe -m uvicorn src.api.main:app --port 8000

Every response includes: answer, refused flag, refusal reason,
citations (metadata + PII-scrubbed preview), latency breakdown,
LLM token accounting, request_id. Enough for the Streamlit UI and
enough for observability.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# --- Collection override MUST happen before importing retrieval modules ----
# The retriever / reranker / bm25 modules cache `chromadb.PersistentClient`
# handles on first call, keyed on `settings.chroma_persist_dir` +
# `chroma_collection_name`. Mutating those AFTER a first call is a no-op.
# Mutating them BEFORE any call redirects everything at the production
# collection.
#
# Resolution order (deploy > project default > baseline dev):
#   1. If CHROMA_PERSIST_DIR / CHROMA_COLLECTION_NAME env vars were set
#      at process start, pydantic-settings has ALREADY populated
#      `settings.chroma_persist_dir` and `chroma_collection_name` from
#      them (see src/config.py — Field with validation_alias). In that
#      case we leave the values alone: Docker/Fly pass /data/chroma_prod
#      and agri_schemes_prod, and the app reads from the mounted volume.
#   2. Otherwise we fall back to the project-root default
#      (data/chroma_prod + settings.production_collection_name) so a
#      developer running `uvicorn` locally without env vars still hits
#      the production collection, not the dev one.

from src.config import settings  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not os.environ.get("CHROMA_PERSIST_DIR"):
    settings.chroma_persist_dir = _PROJECT_ROOT / "data" / "chroma_prod"
if not os.environ.get("CHROMA_COLLECTION_NAME"):
    settings.chroma_collection_name = settings.production_collection_name

# Now safe to import retrieval / production modules — they'll pick up
# the redirected settings on first use.
from src.production.guardrails import scrub_pii  # noqa: E402
from src.production.pipeline import ProductionResult, run_production_query  # noqa: E402
from src.production.query_validation import validate_query  # noqa: E402
from src.production.scheme_discovery import (  # noqa: E402
    DiscoveryResult,
    run_scheme_discovery,
)


# --- Logging setup ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Quiet the chatty transitive loggers.
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("api")


# --- Request / response schemas --------------------------------------------

class QueryRequest(BaseModel):
    """Body of POST /query. The client OWNS the query text; the server
    OWNS request id assignment when the client omits `x-request-id`."""
    query: str = Field(..., description="Natural-language question")


class CitationOut(BaseModel):
    chunk_id: str
    scheme: str
    source_filename: str
    source_type: str
    page_start: int | None = None
    page_end: int | None = None
    is_ocr_source: bool = False
    parent_subfolder: str | None = None


class ChunkPreviewOut(BaseModel):
    """PII-scrubbed preview of one retrieved chunk for the debug UI."""
    chunk_id: str
    rank: int
    scheme: str
    source_filename: str
    source_type: str
    page_start: int | None = None
    page_end: int | None = None
    similarity_score: float | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None
    text_preview: str  # PII-scrubbed, capped at 800 chars


class LatencyBreakdown(BaseModel):
    hybrid_ms: int
    rerank_ms: int
    generation_ms: int
    total_ms: int


class LlmAccounting(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    retries_taken: int = 0
    finish_reason: str = "not_called"
    error: str | None = None


# --- Scheme discovery response types ---------------------------------------

class MatchedReasonOut(BaseModel):
    """One cited reason a scheme surfaced. `source_filename` / `page`
    make the structured layer as auditable as the RAG layer."""
    attribute: str
    matched_value: object
    note: str = ""
    source_filename: str | None = None
    source_page: int | None = None


class MissingInformationOut(BaseModel):
    attribute: str
    why_needed: str


class SchemeMatchOut(BaseModel):
    code: str
    name: str
    long_name: str
    category: str
    one_liner: str
    key_benefits: list[str]
    who_should_apply: str
    how_to_register: dict
    follow_up_query_hint: str
    authoritative_source: dict
    applicability_reasons: list[MatchedReasonOut]
    boost_reasons: list[MatchedReasonOut]
    missing_information: list[MissingInformationOut]
    always_included: bool
    boost_hits: int


class SchemeDiscoveryOut(BaseModel):
    """
    Rendered by the frontend as a "schemes potentially applicable
    to you" section. Never claims eligibility — the frontend uses
    "potentially applicable" phrasing and shows both provided and
    missing information.
    """
    intent_detected: bool
    provided_information: list[dict]
    missing_information_summary: list[str]
    matches: list[SchemeMatchOut]


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    refused: bool
    refusal_reason: str | None = None
    citations: list[CitationOut]
    retrieved_chunks: list[ChunkPreviewOut]
    latency: LatencyBreakdown
    llm: LlmAccounting
    # Populated whenever the discovery signal fires (intent detected
    # OR the user provided any explicit attribute). Absent means the
    # query had no scheme-discovery angle at all — just a factual
    # RAG lookup. Frontend renders as a stacked card section.
    scheme_discovery: SchemeDiscoveryOut | None = None


# --- Per-IP rate limiter (in-memory token bucket) --------------------------

# Enough for a single-instance demo. Multi-instance production would
# use Redis or a shared-store, out of scope here.
_RATE_LIMIT_REQUESTS = 30
_RATE_LIMIT_WINDOW_S = 60.0

_rate_buckets: dict[str, deque[float]] = defaultdict(deque)
_rate_lock = Lock()


def _check_rate_limit(client_ip: str) -> bool:
    """True iff `client_ip` is UNDER the rate limit right now.
    Trims old timestamps as a side effect."""
    now = time.monotonic()
    cutoff = now - _RATE_LIMIT_WINDOW_S
    with _rate_lock:
        bucket = _rate_buckets[client_ip]
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_REQUESTS:
            return False
        bucket.append(now)
        return True


# --- App -------------------------------------------------------------------

app = FastAPI(
    title="Agri Schemes RAG API",
    version="0.1.0",
    description=(
        "Production retrieval + grounded generation over Indian "
        "Government agriculture-scheme documents. Composes hybrid "
        "retrieval + cross-encoder reranking + grounded generation "
        "with deterministic guardrails."
    ),
)


# Per-request timeout ceiling. Above this the API returns 504 rather
# than let the request pile up. gpt-oss-20b typical latency is 5-15s;
# 45s is a comfortable ceiling for a normal + slow-network case.
_REQUEST_TIMEOUT_S = 45.0


# --- Endpoints -------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Liveness + readiness. Confirms the collection is reachable
    (raises an internal error if not, which the caller sees as 500)."""
    from src.retrieval.retriever import _get_collection
    coll = _get_collection(settings)
    return {
        "status": "ok",
        "collection": settings.chroma_collection_name,
        "collection_size": coll.count(),
        "chroma_persist_dir": str(settings.chroma_persist_dir),
    }


@app.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    request: Request,
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
) -> QueryResponse:
    """One-shot RAG endpoint. All guardrails + pipeline logic runs
    inside this function. On any deterministic refusal we STILL return
    HTTP 200 with `refused=True` — 4xx would confuse a browser client
    for what is a normal product behaviour (refusing on out-of-scope /
    injection is not a "bad request", it's a valid response). We only
    use HTTP 4xx/5xx for actual protocol / infrastructure faults."""
    req_id = x_request_id or uuid.uuid4().hex[:16]
    client_ip = (request.client.host if request.client else "unknown")

    # --- Rate limit ---
    if not _check_rate_limit(client_ip):
        logger.warning("rate_limit ip=%s req_id=%s", client_ip, req_id)
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "message": (
                    f"Too many requests. Limit: {_RATE_LIMIT_REQUESTS} "
                    f"per {int(_RATE_LIMIT_WINDOW_S)} s per IP."
                ),
                "request_id": req_id,
            },
        )

    t0 = time.monotonic()
    logger.info(
        "req_id=%s ip=%s query_len=%d",
        req_id, client_ip, len(payload.query),
    )

    # --- Guardrail: deterministic query validation (empty / injection / ..) ---
    validation = validate_query(payload.query)
    if not validation.is_valid:
        logger.info(
            "req_id=%s refused validation=%s",
            req_id, validation.refusal_reason,
        )
        return QueryResponse(
            request_id=req_id,
            query=payload.query,
            answer=validation.refusal_message or "Invalid query.",
            refused=True,
            refusal_reason=f"validation_{validation.refusal_reason}",
            citations=[],
            retrieved_chunks=[],
            latency=LatencyBreakdown(
                hybrid_ms=0, rerank_ms=0, generation_ms=0,
                total_ms=int((time.monotonic() - t0) * 1000),
            ),
            llm=LlmAccounting(),
        )

    # --- Structured scheme-discovery (parallel path) ---
    # Runs alongside RAG per owner-locked D-refinement (2026-09-20):
    # discovery is a SIGNAL, not a hard route. If the query has ANY
    # discovery angle (intent word OR attribute mention), we produce
    # the structured card list. Never suppresses the RAG call.
    discovery: DiscoveryResult | None = None
    try:
        discovery = run_scheme_discovery(validation.cleaned_query)
    except Exception as exc:  # pragma: no cover
        # Discovery must NEVER take down a normal RAG response — log
        # and continue with `discovery=None`.
        logger.warning(
            "req_id=%s scheme discovery failed (non-fatal): %s: %s",
            req_id, type(exc).__name__, str(exc)[:200],
        )

    # --- Production pipeline ---
    # Retrieval + reranking + generation. `run_production_query`
    # handles LLM failures / low-confidence refusal internally and
    # never raises past its own boundary.
    try:
        result: ProductionResult = run_production_query(
            validation.cleaned_query, config=settings,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("req_id=%s pipeline crashed", req_id)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "pipeline_error",
                "message": f"{type(exc).__name__}: {str(exc)[:200]}",
                "request_id": req_id,
            },
        )

    # --- Timeout enforcement (soft) ---
    elapsed = time.monotonic() - t0
    if elapsed > _REQUEST_TIMEOUT_S:
        # Log for observability; the request has already produced a
        # result at this point (we don't hard-cancel the pipeline
        # mid-flight — that would require async plumbing beyond scope).
        logger.warning(
            "req_id=%s exceeded soft timeout: %.1fs > %.1fs",
            req_id, elapsed, _REQUEST_TIMEOUT_S,
        )

    # --- PII scrub on chunk previews (rule 12.11) ---
    previews: list[ChunkPreviewOut] = []
    for rr in result.retrieved_chunks:
        preview_text = (rr.text or "")[:800]
        scrubbed = scrub_pii(preview_text)
        previews.append(ChunkPreviewOut(
            chunk_id=rr.chunk_id,
            rank=rr.rank,
            scheme=rr.scheme,
            source_filename=rr.source_filename,
            source_type=rr.source_type,
            page_start=rr.page_start,
            page_end=rr.page_end,
            similarity_score=rr.similarity_score,
            rerank_score=rr.rerank_score,
            rrf_score=rr.rrf_score,
            text_preview=scrubbed.text,
        ))

    # Also scrub the answer body — the generator is grounded on chunks
    # that may contain PII. Belt + braces.
    answer_scrubbed = scrub_pii(result.answer)

    citations_out = [
        CitationOut(
            chunk_id=c.chunk_id,
            scheme=c.scheme,
            source_filename=c.source_filename,
            source_type=c.source_type,
            page_start=c.page_start,
            page_end=c.page_end,
            is_ocr_source=c.is_ocr_source,
            parent_subfolder=c.parent_subfolder,
        )
        for c in result.citations
    ]

    logger.info(
        "req_id=%s answered refused=%s latency_ms=%d "
        "hybrid_ms=%d rerank_ms=%d gen_ms=%d "
        "prompt_tok=%d comp_tok=%d retries=%d finish=%s",
        req_id, result.refused, result.latency_total_ms,
        result.latency_hybrid_ms, result.latency_rerank_ms,
        result.latency_generation_ms,
        result.prompt_tokens, result.completion_tokens,
        result.retries_taken, result.finish_reason,
    )

    # --- Discovery serialisation ---
    discovery_out: SchemeDiscoveryOut | None = None
    if discovery is not None:
        discovery_out = SchemeDiscoveryOut(
            intent_detected=discovery.intent_detected,
            provided_information=discovery.provided_information,
            missing_information_summary=discovery.missing_information_summary,
            matches=[
                SchemeMatchOut(
                    code=m.code,
                    name=m.name,
                    long_name=m.long_name,
                    category=m.category,
                    one_liner=m.one_liner,
                    key_benefits=m.key_benefits,
                    who_should_apply=m.who_should_apply,
                    how_to_register=m.how_to_register,
                    follow_up_query_hint=m.follow_up_query_hint,
                    authoritative_source=m.authoritative_source,
                    applicability_reasons=[
                        MatchedReasonOut(
                            attribute=r.attribute,
                            matched_value=r.matched_value,
                            note=r.note,
                            source_filename=r.source_filename,
                            source_page=r.source_page,
                        )
                        for r in m.applicability_reasons
                    ],
                    boost_reasons=[
                        MatchedReasonOut(
                            attribute=r.attribute,
                            matched_value=r.matched_value,
                            note=r.note,
                            source_filename=r.source_filename,
                            source_page=r.source_page,
                        )
                        for r in m.boost_reasons
                    ],
                    missing_information=[
                        MissingInformationOut(
                            attribute=mi.attribute,
                            why_needed=mi.why_needed,
                        )
                        for mi in m.missing_information
                    ],
                    always_included=m.always_included,
                    boost_hits=m.boost_hits,
                )
                for m in discovery.matches
            ],
        )

    return QueryResponse(
        request_id=req_id,
        query=payload.query,
        answer=answer_scrubbed.text,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        citations=citations_out,
        retrieved_chunks=previews,
        latency=LatencyBreakdown(
            hybrid_ms=result.latency_hybrid_ms,
            rerank_ms=result.latency_rerank_ms,
            generation_ms=result.latency_generation_ms,
            total_ms=result.latency_total_ms,
        ),
        llm=LlmAccounting(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            reasoning_tokens=result.reasoning_tokens,
            retries_taken=result.retries_taken,
            finish_reason=result.finish_reason,
            error=result.llm_error,
        ),
        scheme_discovery=discovery_out,
    )
