"""
Baseline semantic retriever for the Agriculture Schemes RAG project.

This is the DELIBERATE baseline: pure embedding-based nearest-neighbour
search over the single Chroma collection the embedder built. No BM25,
no cross-encoder rerank, no query rewriting, no CRAG. Those are the
Phase 5 ablation interventions (CLAUDE.md §7, fixes #2–#5) and must be
measurably better than this baseline before they earn a place in the
default pipeline.

Two decisions worth naming out loud so downstream edits don't unpick
them by accident:

  1. **The embedder is loaded exactly once per process** via a lazy
     module-level cache. SentenceTransformer loads take ~5s and hold
     a few hundred MB of weights; reloading on every query would make
     the API unusable. Same treatment for the Chroma collection handle
     — a PersistentClient opens files and reads the HNSW index; not
     something to redo per request.

  2. **We invert Chroma's distance to a similarity score** and expose
     that instead. Chroma returns cosine *distance* (lower = better);
     humans, LLMs, and downstream reranker code all expect a *score*
     where higher = better. Doing the flip once here means every caller
     gets a consistent, obvious ordering — no one has to remember which
     direction wins. See the module-level notes below for the exact
     conversion and why raw distance would be a footgun.
"""

from __future__ import annotations

import logging
from threading import Lock

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.ingestion.models import RetrievalResult


logger = logging.getLogger(__name__)


# --- Lazy singletons ---------------------------------------------------------
# Module-level caches so the embedder + Chroma handle are loaded exactly
# once per process. A Lock guards first-time construction against the
# (unlikely but real) case of two threads calling retrieve() at once
# before either has warmed the cache.
_embedder: SentenceTransformer | None = None
_collection: Collection | None = None
_lock = Lock()


def _load_embedder(config=settings) -> SentenceTransformer:
    """
    Return a process-cached SentenceTransformer.

    Must be the SAME model name as was used at ingest — the vectors in
    Chroma live in that model's embedding space and comparing them to
    vectors from a different model is nonsense (nearest-neighbour
    distances would be arbitrary noise). Reading the model name from
    config, not hardcoding it, is what enforces this discipline.
    """
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                logger.info("Loading embedding model: %s", config.embedding_model)
                _embedder = SentenceTransformer(config.embedding_model)
                logger.info(
                    "Embedder ready: dim=%d, max_seq_length=%d",
                    _embedder.get_sentence_embedding_dimension(),
                    _embedder.max_seq_length,
                )
    return _embedder


def _get_collection(config=settings) -> Collection:
    """
    Return a process-cached Chroma collection handle.

    Uses `get_collection` (not `get_or_create_collection`) on purpose:
    the retriever is a *reader*. If the collection doesn't exist yet
    it means the embedder was never run, and the correct behaviour is
    to fail loudly rather than silently create an empty collection and
    return zero results forever.
    """
    global _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                client = chromadb.PersistentClient(
                    path=str(config.chroma_persist_dir)
                )
                _collection = client.get_collection(
                    name=config.chroma_collection_name
                )
                logger.info(
                    "Chroma collection ready: name=%s size=%d",
                    config.chroma_collection_name,
                    _collection.count(),
                )
    return _collection


# --- Small helpers -----------------------------------------------------------

def _embed_query(query: str, embedder: SentenceTransformer) -> list[float]:
    """
    Embed a single query string.

    `normalize_embeddings=True` matches what the embedder used at ingest
    time — bge models are trained to be compared as unit vectors with
    cosine similarity, and mixing normalised corpus vectors with
    unnormalised query vectors would tilt every distance.
    """
    vec = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vec[0].tolist()


def _chroma_result_to_retrieval_result(
    chunk_id: str,
    document: str,
    metadata: dict,
    distance: float,
    rank: int,
) -> RetrievalResult:
    """
    Turn one row of Chroma's query response into a RetrievalResult.

    Metadata keys that were omitted at ingest time (because their value
    was None — see embedder._chunk_to_chroma_record) come back as
    absent keys; `.get(...)` cleanly restores them to None here.

    similarity_score = 1 - distance. Chroma's collection was created
    with `hnsw:space=cosine`, so distance is cosine distance in [0, 2].
    For unit-normalised text embeddings it sits in [0, 1] in practice.
    Higher similarity_score = more similar. See module docstring.
    """
    return RetrievalResult(
        chunk_id=chunk_id,
        text=document,
        scheme=metadata.get("scheme", ""),
        source_type=metadata.get("source_type", ""),
        source_filename=metadata.get("source_filename", ""),
        source_filepath=metadata.get("source_filepath", ""),
        is_ocr_source=bool(metadata.get("is_ocr_source", False)),
        parent_subfolder=metadata.get("parent_subfolder"),
        page_start=metadata.get("page_start"),
        page_end=metadata.get("page_end"),
        workflow_id=metadata.get("workflow_id"),
        similarity_score=1.0 - float(distance),
        rank=rank,
    )


# --- Public entry point ------------------------------------------------------

def retrieve(
    query: str,
    top_k: int | None = None,
    where: dict | None = None,
    config=settings,
) -> list[RetrievalResult]:
    """
    Return the top-K semantically nearest chunks for `query`.

    Parameters
    ----------
    query
        Natural-language question. Embedded with the ingest-time model
        and compared against every chunk vector in the collection.
    top_k
        How many results to return. `None` uses `config.retrieval_top_k`.
    where
        Optional Chroma metadata filter, forwarded verbatim. Not applied
        by default — this is a hook for callers that want scheme-scoped
        or type-scoped retrieval. Example: `where={"scheme": "PMFBY"}`
        restricts nearest-neighbour search to PMFBY chunks only.
    config
        Injectable so tests / notebooks can point at custom settings.

    Returns
    -------
    A list of `RetrievalResult`, ordered best-first (highest
    similarity_score first, rank starting at 1). May be shorter than
    `top_k` if the collection (or the filtered subset) has fewer chunks
    than requested — Chroma silently returns what it has, we surface it
    honestly.
    """
    if top_k is None:
        top_k = config.retrieval_top_k

    embedder = _load_embedder(config)
    collection = _get_collection(config)

    query_vec = _embed_query(query, embedder)

    # Chroma's query API: query_embeddings is a list of query vectors
    # (we send one), and every returned field is a list-of-lists shaped
    # [n_queries][n_results]. We index [0] throughout to pull out our
    # single query's results.
    raw = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=where,   # None is fine — Chroma treats it as "no filter"
        include=["documents", "metadatas", "distances"],
    )

    ids = raw.get("ids", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    results: list[RetrievalResult] = []
    for rank, (cid, doc, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        results.append(
            _chroma_result_to_retrieval_result(
                chunk_id=cid,
                document=doc,
                metadata=meta or {},
                distance=dist,
                rank=rank,
            )
        )

    logger.info(
        "retrieve(query=%r, top_k=%d, where=%s) → %d results",
        query, top_k, where, len(results),
    )
    return results
