"""
Image chunker (Deliverable 2 B5.5).

Takes a `VisionResult` + the list of per-page `ImageOccurrence`s for
that content-hash and materialises one `Chunk` PER PAGE OCCURRENCE.
All occurrences share the same rendered `text` (the [IMAGE]...[/IMAGE]
sentinel wrapping the extraction payload) — they differ only in
provenance metadata (scheme, filename, page, xref).

Why one chunk per occurrence, not one chunk per unique visual:
  * Each page is a distinct retrieval target with its own citation.
    A user asking about page 47 of the NFSM 2018 doc should retrieve
    a chunk whose metadata points at page 47, not "one of the pages
    this table appears on."
  * Chroma metadata is scalar. There's no natural way to store an
    array of (scheme, filename, page) tuples on a single chunk.
  * The vector duplicate cost is negligible (< 100 duplicate vectors
    across the corpus) and the retrieval-honesty win is large.

DECORATIVE and OTHER classifications do NOT reach this module — the
runner drops them before calling in. `vision_failed=True` DOES reach
this module (per production reliability rule 12.7 — "never silently
drop"). Those chunks are materialised with a placeholder text block
that says "vision extraction unavailable — refer to source page" so
retrieval can still surface the page's existence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.ingestion.image_extractor import ImageCandidate, ImageOccurrence
from src.ingestion.models import Chunk
from src.vision.adapter import VisionResult, USEFUL_CLASSES


# Sentinel tokens. Match the convention already established for tables
# in the structure-aware chunker (`[TABLE] ... [/TABLE]`) so a downstream
# reader sees a consistent markup language across modalities.
IMAGE_OPEN_MARKER = "[IMAGE]"
IMAGE_CLOSE_MARKER = "[/IMAGE]"


# Which extraction fields are worth rendering. Order matters for
# retrieval — put the highest-signal keys first so the leading portion
# of the embedded chunk (which bge-small sees before the 512-token cut)
# carries the tokens most likely to match a query.
_RENDER_KEY_ORDER = [
    "description",
    "key_information",
    "important_numbers",
    "monetary_values",
    "percentages",
    "dates",
    "labels",
    "relationships",
    "visible_text",
]

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _stem(filename: str) -> str:
    """`foo.pdf` → `foo`. Matches the convention in chunker.py."""
    return filename.rsplit(".", 1)[0]


def _chunk_id(occ: ImageOccurrence, content_hash: str) -> str:
    """
    Deterministic id:
      `{scheme}[_{parent_subfolder}]_{stem}_img_p{page:04d}_x{xref}_h{sha8}`

    Contains BOTH page and xref because a single page can carry
    multiple embedded images. content-hash prefix ensures that if the
    same PDF gets re-authored with different image bytes on the same
    page/xref, we don't silently overwrite the previous chunk.
    """
    parts = [occ.scheme]
    if occ.parent_subfolder:
        parts.append(occ.parent_subfolder)
    parts.append(_stem(occ.filename))
    parts.append(
        f"img_p{occ.page:04d}_x{occ.xref}_h{content_hash[:8]}"
    )
    # Ids feed Chroma primary keys — strip any characters that would
    # trip its regex on odd filenames.
    return _SAFE_ID_RE.sub("_", "_".join(parts))


def _render_chunk_text(result: VisionResult) -> str:
    """
    Build the sentinel-wrapped chunk body. Structured payload rendered
    as human-readable key: value lines rather than raw JSON, because
    bge-small embeds English prose better than JSON syntax noise
    (empirically: `"description": "X"` tokenises the quotes and colons
    as their own tokens, diluting the signal).
    """
    lines: list[str] = [IMAGE_OPEN_MARKER, f"classification: {result.classification}"]
    for key in _RENDER_KEY_ORDER:
        val = getattr(result, key)
        if isinstance(val, str):
            if val.strip():
                lines.append(f"{key}: {val.strip()}")
        elif isinstance(val, list):
            if val:
                # Bullet each item so bge-small sees a list, not a run-on.
                lines.append(f"{key}:")
                for item in val:
                    s = str(item).strip()
                    if s:
                        lines.append(f"  - {s}")
    lines.append(IMAGE_CLOSE_MARKER)
    return "\n".join(lines)


def _render_failed_text(occ: ImageOccurrence, content_hash: str) -> str:
    """
    Placeholder chunk for `vision_failed=True`. Deliberately terse so a
    reader retrieving this can tell immediately that the vision layer
    didn't work here — but the chunk STILL EXISTS, so retrieval can
    surface the page's presence in the corpus (per production rule
    12.7).
    """
    return (
        f"{IMAGE_OPEN_MARKER}\n"
        f"classification: vision_failed\n"
        f"description: Vision extraction was unavailable for this image. "
        f"Refer to the source page for details.\n"
        f"source: {occ.filename} page {occ.page}\n"
        f"content_hash: {content_hash}\n"
        f"{IMAGE_CLOSE_MARKER}"
    )


def make_image_chunks(
    candidate: ImageCandidate,
    result: VisionResult,
) -> list[Chunk]:
    """
    Emit one Chunk per occurrence. Returns [] when the classification
    is not TABLE / CHART / INFOGRAPHIC AND vision_failed is False
    (defensive — the runner should not call us in that case; we assert
    with an empty return rather than raise so a policy mismatch is
    visible but not fatal).
    """
    if not result.vision_failed and result.classification not in USEFUL_CLASSES:
        return []

    if result.vision_failed:
        # Different text per occurrence because the placeholder cites
        # the specific page — but the *shape* is identical.
        chunks: list[Chunk] = []
        for occ in candidate.occurrences:
            chunks.append(_build_chunk(candidate, occ, result,
                                       text=_render_failed_text(occ, candidate.content_hash)))
        return chunks

    # Happy path — one shared text body, one chunk per occurrence.
    body = _render_chunk_text(result)
    return [
        _build_chunk(candidate, occ, result, text=body)
        for occ in candidate.occurrences
    ]


def _build_chunk(
    candidate: ImageCandidate,
    occ: ImageOccurrence,
    result: VisionResult,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=_chunk_id(occ, candidate.content_hash),
        text=text,
        scheme=occ.scheme,
        source_type="image",
        source_filename=occ.filename,
        source_filepath=Path(occ.filepath),
        is_ocr_source=occ.is_ocr_source,
        parent_subfolder=occ.parent_subfolder,
        page_start=occ.page,
        page_end=occ.page,
        workflow_id=None,
        image_xref=occ.xref,
        content_hash=candidate.content_hash,
        vision_content_type=result.classification,
        vision_failed=result.vision_failed,
    )
