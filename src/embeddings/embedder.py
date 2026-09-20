"""
Embedder + vector-store writer for the Agriculture Schemes RAG project.

Turns a ChunkedCorpus into vectors and lands them in a single ChromaDB
collection with the metadata a citation-grounded answer will need later.

Two decisions worth naming out loud so downstream edits don't unpick them
by accident:

  1. **One collection for the whole corpus, not one per scheme.** Some
     questions genuinely span schemes ("does the KCC limit differ by bank
     type in 2026?" pulls RBI + DA&FW; "PMFBY vs KCC eligibility for the
     same farmer" is cross-scheme by construction — see scope.md §1,
     §2, §4). If we split into per-scheme collections we would have to
     query all seven, merge the results, and renormalise scores by hand
     — and cross-scheme reranking would become a mess. Filtering by
     scheme happens at query time via `where={"scheme": "PMFBY"}` on
     the single collection instead. This is Chroma's intended pattern.

  2. **Every Chunk metadata field lands in Chroma's `metadatas`.** The
     raw text is what Chroma stores as `documents`; the *why-this-chunk-
     matters-for-citation* fields (scheme, source_filename, page_start
     etc.) live in metadatas so we can (a) show a faithful citation and
     (b) filter on them at retrieval time. Chroma requires metadata
     values to be scalar (str / int / float / bool). Paths get str()-ed;
     None values are omitted (Chroma treats "key absent" and "key = None"
     the same for `where`, and omitting keeps the summary honest —
     "this chunk has no workflow_id" is different from "this chunk has
     workflow_id = None").

The embedder does NOT retrieve. That is deliberate. Retrieval is the
next component — this file's only job is to make the collection exist
and be queryable.
"""

from __future__ import annotations

import logging
from typing import Iterable, Iterator

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.ingestion.models import (
    Chunk,
    ChunkedCorpus,
    EmbeddingFailure,
    EmbeddingSummary,
)


logger = logging.getLogger(__name__)


# --- Model + collection setup ------------------------------------------------

def _load_embedder(config=settings) -> SentenceTransformer:
    """
    Load the sentence-transformers model named in config.

    We resolve the model once and pass the loaded object through the rest
    of the pipeline. Loading a SentenceTransformer is not free (weights
    download + tokenizer init) — doing it per batch would be wasteful.
    """
    logger.info("Loading embedding model: %s", config.embedding_model)
    model = SentenceTransformer(config.embedding_model)
    logger.info(
        "Loaded embedder: dim=%d, max_seq_length=%d",
        model.get_sentence_embedding_dimension(),
        model.max_seq_length,
    )
    return model


def _get_or_create_collection(config=settings) -> Collection:
    """
    Return the single project collection, creating it on first run.

    Persistent client → the collection survives process restarts. If we
    used the in-memory client the DB would vanish every run and re-runs
    of the embedder would be non-idempotent by construction.

    `metadata={"hnsw:space": "cosine"}` is set at creation time. bge
    embeddings are trained with cosine similarity; using L2 would give
    subtly worse retrieval. Chroma bakes the distance metric into the
    HNSW index at creation, so this can't be changed later without
    dropping the collection.
    """
    config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.chroma_persist_dir))
    collection = client.get_or_create_collection(
        name=config.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "Chroma collection ready: name=%s path=%s current_count=%d",
        config.chroma_collection_name,
        config.chroma_persist_dir,
        collection.count(),
    )
    return collection


# --- Utilities ---------------------------------------------------------------

def _batch(iterable: Iterable, size: int) -> Iterator[list]:
    """
    Yield successive `size`-length lists from `iterable`.

    We don't rely on itertools.batched (Python 3.12+) — locked to 3.11
    per CLAUDE.md §12, so we roll it by hand. Small, obvious, no import
    of a fourth utility library.
    """
    if size <= 0:
        raise ValueError(f"batch size must be positive, got {size}")
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _chunk_to_chroma_record(
    chunk: Chunk,
) -> tuple[str, str, dict[str, str | int | float | bool]]:
    """
    Convert a Chunk into (id, document, metadata) for Chroma upsert.

    Embeddings are attached separately in embed_corpus (we batch the
    embed call for throughput). None-valued fields are omitted from
    metadata rather than passed through — Chroma's metadata type is
    Mapping[str, str|int|float|bool] and rejects None. Paths are
    coerced to str for the same reason.
    """
    metadata: dict[str, str | int | float | bool] = {
        "scheme": chunk.scheme,
        "source_type": chunk.source_type,
        "source_filename": chunk.source_filename,
        "source_filepath": str(chunk.source_filepath),
        "is_ocr_source": chunk.is_ocr_source,
    }
    if chunk.parent_subfolder is not None:
        metadata["parent_subfolder"] = chunk.parent_subfolder
    if chunk.page_start is not None:
        metadata["page_start"] = chunk.page_start
    if chunk.page_end is not None:
        metadata["page_end"] = chunk.page_end
    if chunk.workflow_id is not None:
        metadata["workflow_id"] = chunk.workflow_id
    # Image-source-only fields (see models.py Chunk docstring). Each
    # is added only when populated so Chroma's `where` filter behaves
    # sensibly: `where={"source_type": "image"}` selects image chunks
    # without pulling in text chunks that "happen to have image_xref=None".
    if chunk.image_xref is not None:
        metadata["image_xref"] = chunk.image_xref
    if chunk.content_hash is not None:
        metadata["content_hash"] = chunk.content_hash
    if chunk.vision_content_type is not None:
        metadata["vision_content_type"] = chunk.vision_content_type
    # `vision_failed` is a bool with a False default — emit only when
    # True so the majority of chunks stay lean. `where={"vision_failed":
    # True}` still finds the failures.
    if chunk.vision_failed:
        metadata["vision_failed"] = True

    return chunk.chunk_id, chunk.text, metadata


def _count_tokens_over_limit(
    model: SentenceTransformer, texts: list[str]
) -> list[bool]:
    """
    Return one bool per input text: True if it exceeds the model's
    max_seq_length in tokens (i.e. will be truncated at embed time).

    We tokenize with the model's own tokenizer so this is accurate
    rather than an "N chars ≈ N/4 tokens" guess. Called once per
    batch, so the overhead is small compared to the embed call itself.
    """
    tokenizer = model.tokenizer
    max_len = model.max_seq_length
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
    return [len(ids) > max_len for ids in encoded["input_ids"]]


# --- Entry point -------------------------------------------------------------

def embed_corpus(
    chunked_corpus: ChunkedCorpus, config=settings
) -> EmbeddingSummary:
    """
    Embed every chunk in `chunked_corpus` and upsert into Chroma.

    Idempotent by design: chunk_ids are deterministic (see chunker.py)
    and we use Chroma's `upsert`, so re-running with the same corpus
    overwrites in place instead of duplicating.
    """
    model = _load_embedder(config)
    collection = _get_or_create_collection(config)

    chunks = chunked_corpus.chunks
    total = len(chunks)
    logger.info(
        "Embedding %d chunks in batches of %d",
        total,
        config.embed_batch_size,
    )

    per_scheme_upserted: dict[str, int] = {}
    truncated_ids: list[str] = []
    failures: list[EmbeddingFailure] = []
    total_upserted = 0

    for batch_idx, batch in enumerate(_batch(chunks, config.embed_batch_size)):
        records = [_chunk_to_chroma_record(c) for c in batch]
        ids = [r[0] for r in records]
        documents = [r[1] for r in records]
        metadatas = [r[2] for r in records]

        # Truncation audit: flag any chunk that will silently lose its tail
        # at embed time. bge-small-en-v1.5 has a 512-token window, and a
        # single unlucky workflow chunk (e.g. the MIDH 12-step subsidy
        # claim) can go over. The truncation is not a bug — the leading
        # portion still embeds — but it should not be invisible.
        try:
            over = _count_tokens_over_limit(model, documents)
            for chunk_id, is_over in zip(ids, over):
                if is_over:
                    truncated_ids.append(chunk_id)
                    logger.warning(
                        "Chunk %s exceeds model max_seq_length (%d tokens); "
                        "sentence-transformers will truncate at embed time.",
                        chunk_id,
                        model.max_seq_length,
                    )
        except Exception as e:
            # Truncation counting is a diagnostic, not load-bearing. If
            # the tokenizer trips on one batch we log and continue —
            # the embedding call itself handles truncation internally.
            logger.warning("Token-length check failed for batch %d: %s", batch_idx, e)

        try:
            embeddings = model.encode(
                documents,
                batch_size=config.embed_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # bge models expect L2-normalised vectors
            ).tolist()
        except Exception as e:
            # A whole batch failing to embed is rare (usually OOM or a
            # tokenizer edge case). Record every id in the batch as a
            # failure and move on — refusing to write the rest of the
            # corpus over one bad batch would waste an hour of work.
            logger.error("Batch %d embed failed: %s", batch_idx, e)
            for chunk_id in ids:
                failures.append(
                    EmbeddingFailure(chunk_id=chunk_id, reason=f"embed error: {e}")
                )
            continue

        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as e:
            logger.error("Batch %d upsert failed: %s", batch_idx, e)
            for chunk_id in ids:
                failures.append(
                    EmbeddingFailure(chunk_id=chunk_id, reason=f"upsert error: {e}")
                )
            continue

        total_upserted += len(batch)
        for c in batch:
            per_scheme_upserted[c.scheme] = per_scheme_upserted.get(c.scheme, 0) + 1

        # Progress log every 10 batches — noisy enough to see life, quiet
        # enough not to drown the console on a 10k-chunk run.
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) * config.embed_batch_size >= total:
            logger.info(
                "Batch %d done: %d/%d chunks upserted",
                batch_idx + 1,
                total_upserted,
                total,
            )

    collection_size_after = collection.count()
    logger.info(
        "Embedding complete: upserted=%d, collection_size=%d, failures=%d, truncated=%d",
        total_upserted,
        collection_size_after,
        len(failures),
        len(truncated_ids),
    )

    return EmbeddingSummary(
        model_name=config.embedding_model,
        embedding_dimension=model.get_sentence_embedding_dimension(),
        collection_name=config.chroma_collection_name,
        total_chunks_processed=total,
        total_upserted=total_upserted,
        collection_size_after=collection_size_after,
        per_scheme_upserted=per_scheme_upserted,
        truncated_chunk_count=len(truncated_ids),
        truncated_chunks=truncated_ids,
        failures=failures,
    )
