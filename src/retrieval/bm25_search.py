"""
BM25 keyword retriever for the Agriculture Schemes RAG project.

Phase 5, fix #2 (CLAUDE.md §7). Standalone lexical-search index built
in-memory at first use from the same Chroma collection the semantic
retriever reads. It answers "which chunks contain the same tokens as
the question?" — a fundamentally different signal from the semantic
retriever's "which chunks are geometrically closest in embedding
space?" That is exactly why the two are fused downstream (see
`src/retrieval/hybrid.py`): they miss different things.

Why this exists (interview framing to preserve in the code):

  The semantic retriever is trained on paraphrase and topical
  similarity — a query and a chunk can score high together even if
  they share no literal terms. That is a huge win most of the time,
  but it has a well-known weakness: exact-code lookups (scheme names
  like "PMFBY", subsidy percentages like "50%", circular numbers like
  "RBI/2022-23/74") do not paraphrase. If the correct chunk contains
  the string "RBI/2022-23/74" and the question contains
  "RBI/2022-23/74", the semantic retriever can still miss it in
  favour of a chunk that is topically related but doesn't cite the
  circular.

  BM25 rewards exact term matches, weighted by IDF (rare terms count
  more) and normalised by document length (long chunks aren't
  automatically favoured just because they contain more words). It is
  the industry-standard sparse baseline — every major RAG paper
  reports it, and every reviewer expects to see it.

  The base RAG failure this fixes maps directly to CLAUDE.md §7 row
  #2: "Meaning-search misses exact codes." BM25 doesn't replace the
  semantic retriever — it complements it. Downstream fusion
  (`hybrid.py`) is what turns two complementary signals into one
  ordering.

Three decisions worth naming out loud so downstream edits don't
unpick them by accident:

  1. **The BM25 index is built ONCE per process** via a lazy
     module-level cache (mirror of retriever.py / reranker.py). The
     `collection.get()` call over ~thousands of chunks + the
     `BM25Okapi` construction take a couple of seconds; doing that
     per query would make the API and the eval harness unusable.

  2. **BM25 tokenises the RAW CHUNK TEXT only** — the exact same text
     that was embedded into Chroma. It is NOT enriched with metadata
     (scheme names, filenames, workflow_ids). Reason: the ablation
     row measures "what does adding BM25 buy me over pure semantic?"
     If BM25 were secretly a metadata search on top of a lexical
     search, we'd be conflating two interventions and could not
     attribute a metric movement to either alone. If metadata-boosted
     retrieval turns out to be useful, it becomes its own separate
     labelled variant — not a silent add-on to this row.

  3. **Tokenisation is deliberately simple**: lowercase + split on
     non-word characters (`re.findall(r"\\w+", text.lower())`). No
     stemming, no stopword list, no BPE, no lemmatisation. Reason:
     each of those is a knob that must be defended, and none of them
     are required to demonstrate the intended fix (exact-code recall).
     "PMFBY" tokenises as `["pmfby"]`; "PM-KISAN" as `["pm", "kisan"]`;
     "Rs. 6,000" as `["rs", "6", "000"]`. Punctuation and casing are
     lost — this is standard bag-of-terms treatment and matches the
     tokeniser most BM25 baselines in the literature use. If a later
     experiment shows term-normalisation would materially help, that
     becomes its own labelled variant with its own row.
"""

from __future__ import annotations

import logging
import re
from threading import Lock

import chromadb
from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.models import RetrievalResult


logger = logging.getLogger(__name__)


# --- Lazy singletons --------------------------------------------------------
# Module-level caches so the BM25 index + the chunk metadata dictionary are
# built exactly once per process. A Lock guards first-time construction so
# two threads calling bm25_search() before the cache is warm can't both
# trigger a rebuild.
_bm25: BM25Okapi | None = None
# Parallel to _bm25: ordered list of chunk_ids in the same order as the
# tokenised corpus the BM25 index was built over. BM25Okapi indexes by
# integer position; we need this to map back to a chunk_id.
_chunk_ids: list[str] | None = None
# chunk_id → (text, metadata dict) so `bm25_search()` can hydrate a
# `RetrievalResult` without going back to Chroma. Building this up-front
# from the same `.get()` call that populates the corpus is cheaper than
# a per-query Chroma lookup and keeps the index and the metadata in
# lockstep — an out-of-sync corpus and metadata dict would be a subtle,
# hard-to-diagnose bug.
_metadata: dict[str, dict] | None = None
_lock = Lock()


# --- Tokeniser --------------------------------------------------------------
# Compiled once at module import. `\w+` matches unicode word characters
# (letters, digits, underscore) — the standard bag-of-terms boundary.
# Lowercase applied before matching so casing does not fragment the
# vocabulary ("PMFBY" and "pmfby" both tokenise to `["pmfby"]`).
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """
    Bag-of-terms tokenisation used at both index time and query time.

    Same function applied to both sides so a query token and a chunk
    token that were originally spelled identically end up spelled
    identically after tokenisation. Any drift here (e.g. lowercasing
    the corpus but not the query) would silently break BM25 recall on
    exact-code lookups — the very thing this retriever exists to fix.
    """
    return _TOKEN_RE.findall(text.lower())


# --- Index construction -----------------------------------------------------

def _get_collection(config=settings) -> Collection:
    """
    Return a fresh handle to the same Chroma collection the semantic
    retriever reads. Deliberately does NOT share retriever.py's module
    cache — sharing would couple the two modules' first-use ordering,
    and Chroma collection handles are cheap.
    """
    client = chromadb.PersistentClient(path=str(config.chroma_persist_dir))
    return client.get_collection(name=config.chroma_collection_name)


def _build_index(config=settings) -> None:
    """
    Populate `_bm25`, `_chunk_ids`, `_metadata` from the full Chroma
    collection. Called once at first-use under the module lock.

    Reads every chunk out of Chroma via `collection.get()` (documents
    + metadatas + ids). No filtering — every chunk that the semantic
    retriever can see must also be indexable by BM25, otherwise the
    hybrid fusion is measuring "semantic vs a subset of BM25" instead
    of "semantic vs BM25 on the same corpus."
    """
    global _bm25, _chunk_ids, _metadata

    collection = _get_collection(config)
    logger.info(
        "Building BM25 index from Chroma collection: name=%s size=%d",
        config.chroma_collection_name,
        collection.count(),
    )

    raw = collection.get(include=["documents", "metadatas"])
    ids: list[str] = list(raw.get("ids") or [])
    documents: list[str] = list(raw.get("documents") or [])
    metadatas: list[dict] = list(raw.get("metadatas") or [])

    if not ids:
        raise RuntimeError(
            "Chroma collection is empty; cannot build BM25 index. "
            "Run the embedder first."
        )
    if not (len(ids) == len(documents) == len(metadatas)):
        # Not defensive paranoia — Chroma promises these lists are
        # aligned; if that promise ever breaks the whole hybrid pool is
        # scrambled and we want a loud failure at index-build time, not
        # a silently-wrong ordering at eval time.
        raise RuntimeError(
            f"Chroma returned misaligned lists: "
            f"ids={len(ids)} documents={len(documents)} metadatas={len(metadatas)}"
        )

    tokenised_corpus: list[list[str]] = [_tokenize(doc or "") for doc in documents]

    _bm25 = BM25Okapi(tokenised_corpus)
    _chunk_ids = ids
    _metadata = {
        cid: {"text": doc, "meta": (meta or {})}
        for cid, doc, meta in zip(ids, documents, metadatas)
    }

    logger.info(
        "BM25 index ready: n_chunks=%d, mean_tokens_per_chunk=%.1f",
        len(tokenised_corpus),
        (sum(len(t) for t in tokenised_corpus) / max(1, len(tokenised_corpus))),
    )


def _ensure_index(config=settings) -> None:
    """Lazy first-use build, protected by the module lock."""
    if _bm25 is None or _chunk_ids is None or _metadata is None:
        with _lock:
            if _bm25 is None or _chunk_ids is None or _metadata is None:
                _build_index(config)


# --- Result hydration -------------------------------------------------------

def _hydrate(
    chunk_id: str,
    bm25_score: float,
    bm25_rank: int,
) -> RetrievalResult:
    """
    Build a `RetrievalResult` for a BM25-only hit.

    `similarity_score` is deliberately left `None` (its default): this
    chunk was found by the LEXICAL retriever and there is no cosine
    similarity to report. Reporting 0.0 would falsely imply
    "orthogonal to query" instead of "we never asked." See
    `RetrievalResult` docstring for the hybrid field contract.

    `rank` is set to the BM25 rank so a caller that never fuses (e.g.
    a hypothetical pure-BM25 mode or a smoke test) still gets a
    coherent 1-indexed ordering. The hybrid fusion in `hybrid.py`
    overwrites `rank` with the post-RRF rank and preserves the BM25
    rank separately as `bm25_rank`.
    """
    assert _metadata is not None, "BM25 index accessed before build."
    entry = _metadata[chunk_id]
    text: str = entry["text"] or ""
    meta: dict = entry["meta"] or {}
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        scheme=meta.get("scheme", ""),
        source_type=meta.get("source_type", ""),
        source_filename=meta.get("source_filename", ""),
        source_filepath=meta.get("source_filepath", ""),
        is_ocr_source=bool(meta.get("is_ocr_source", False)),
        parent_subfolder=meta.get("parent_subfolder"),
        page_start=meta.get("page_start"),
        page_end=meta.get("page_end"),
        workflow_id=meta.get("workflow_id"),
        similarity_score=None,
        rank=bm25_rank,
        bm25_rank=bm25_rank,
        bm25_score=bm25_score,
    )


# --- Public entry point -----------------------------------------------------

def bm25_search(
    query: str,
    top_k: int | None = None,
    config=settings,
) -> list[RetrievalResult]:
    """
    Return the top-K BM25 hits for `query`.

    Parameters
    ----------
    query
        Natural-language question. Tokenised with the same function
        applied to the corpus at index time — see `_tokenize` for why
        that symmetry is load-bearing.
    top_k
        How many results to return. `None` uses
        `config.hybrid_bm25_top_n` (the BM25 pool size used by the
        hybrid retriever). Callers wanting a different pool depth
        (e.g. the smoke test asking for 50 to see the tail) can pass
        it explicitly.
    config
        Injectable so tests / notebooks can point at custom settings.

    Returns
    -------
    A list of `RetrievalResult`, best-first, of length
    `min(top_k, n_chunks_in_corpus)`. `similarity_score` is `None` on
    every result (see `_hydrate` for why). `bm25_score` and `bm25_rank`
    are populated. Empty-corpus / empty-query paths return `[]`.

    A note on empty queries. If the query contains no word characters
    (e.g. all punctuation) `_tokenize` returns `[]`. `BM25Okapi.get_scores`
    on an empty query returns a zero vector — every candidate scores
    0.0 and the "top-K" is meaningless. We short-circuit that to `[]`
    so the caller isn't misled by a plausible-looking-but-uniform
    ordering.
    """
    if top_k is None:
        top_k = config.hybrid_bm25_top_n

    _ensure_index(config)
    assert _bm25 is not None and _chunk_ids is not None

    tokens = _tokenize(query)
    if not tokens:
        logger.info(
            "bm25_search(query=%r, top_k=%d) → 0 results (empty tokenisation)",
            query, top_k,
        )
        return []

    # BM25Okapi.get_scores returns a numpy array of length n_chunks
    # aligned with the corpus order (== `_chunk_ids` order). We sort by
    # score descending and slice to top_k rather than using argpartition
    # so the top-K list is fully ordered — hybrid fusion needs the ranks,
    # not just the set.
    scores = _bm25.get_scores(tokens)

    # Pair each score with its chunk_id, sort descending, slice.
    # Python's sort is stable, so equal-score chunks come back in their
    # original corpus order — the honest fallback when BM25 can't
    # distinguish two documents.
    scored: list[tuple[float, str]] = list(zip(
        (float(s) for s in scores), _chunk_ids
    ))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    results: list[RetrievalResult] = []
    for rank, (score, cid) in enumerate(scored[:top_k], start=1):
        results.append(_hydrate(chunk_id=cid, bm25_score=score, bm25_rank=rank))

    logger.info(
        "bm25_search(query=%r, top_k=%d) → %d results "
        "(top score=%.4f, bottom kept score=%.4f)",
        query, top_k, len(results),
        results[0].bm25_score if results else 0.0,
        results[-1].bm25_score if results else 0.0,
    )
    return results
