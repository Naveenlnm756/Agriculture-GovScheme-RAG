"""
Baseline chunker for the Agriculture Schemes RAG project.

This is the DELIBERATE baseline — fixed-size character-window chunking for
PDFs, one-chunk-per-workflow for workflow CSVs. It is not the final version.
Structure-aware chunking is a Phase 5 ablation improvement (CLAUDE.md §7,
fix #1); this baseline is what that improvement must measurably beat.

Two design decisions worth naming out loud so future edits don't unpick
them by accident:

  1. **PDFs and workflows are chunked separately.** PDFs are prose with
     no natural procedural unit, so we split them by character windows.
     Workflows are ordered step-lists — a coherent procedural unit — so
     each LoadedWorkflow becomes exactly ONE chunk. Splitting a workflow
     across chunks would separate step 5 from step 6 and destroy the
     thing that makes the workflow layer useful (scope.md §3).

  2. **PDF pages are joined into one document blob before windowing.**
     Chunking per-page would lose cross-page context — a rate defined on
     page 12 and its exception on page 13 would end up in different
     chunks. We join with `[PAGE {n}]` markers and remember each page's
     starting offset, so a chunk that spans pages 7–9 can honestly claim
     `page_start=7, page_end=9`.

Nothing here embeds, filters, deduplicates, or writes to disk. The
chunker's only job is to produce chunks.
"""

from __future__ import annotations

import logging
from statistics import mean

from src.config import settings
from src.ingestion.models import (
    Chunk,
    ChunkedCorpus,
    LoadedCorpus,
    LoadedPDFDocument,
    LoadedWorkflow,
)


logger = logging.getLogger(__name__)

# Injected between pages when joining a PDF into one blob. Kept visible on
# purpose: it shows up in chunk text, which means a downstream reader can
# see exactly which page a passage came from without extra machinery.
PAGE_MARKER_TEMPLATE = "\n\n[PAGE {n}]\n\n"

# Wrapper tokens for a table chunk. Kept as short, visible ASCII sentinels
# rather than markdown fences so:
#   * a retrieval reader can tell "this chunk is a table, not prose" at a
#     glance without inspecting formatting
#   * the tokens survive any downstream markdown-to-plaintext conversion
#     without collapsing into whitespace
# The trailing newline discipline matters — the closing `[/TABLE]` must be
# on its own line so a naive substring search for `[TABLE]` never finds
# `[/TABLE]` as a false positive.
TABLE_OPEN_MARKER = "[TABLE]"
TABLE_CLOSE_MARKER = "[/TABLE]"


# --- Small utilities ---------------------------------------------------------

def _pdf_chunk_id(pdf: LoadedPDFDocument, idx: int, kind: str = "c") -> str:
    """
    Deterministic chunk id: `{scheme}[_{parent_subfolder}]_{stem}_{kind}{idx:04d}`.

    `kind` is `"c"` for text windows (baseline and non-table text in
    structure-aware mode) and `"tbl"` for whole-table chunks emitted by
    the structure-aware pass. Keeping the split visible in the id means
    a reader browsing Chroma can tell a table chunk from a prose chunk
    without opening the payload — useful during ablation debugging.

    Determinism matters because chunk ids will become the primary key in
    Chroma. If we re-run ingestion on the same corpus, we want the same
    ids so that Chroma's upsert semantics work — otherwise a re-run would
    duplicate every chunk. Random ids would make the vector store append
    a fresh copy every time and slowly corrupt retrieval.
    """
    parts = [pdf.scheme]
    if pdf.parent_subfolder:
        parts.append(pdf.parent_subfolder)
    parts.append(_stem(pdf.filename))
    parts.append(f"{kind}{idx:04d}")
    return "_".join(parts)


def _workflow_chunk_id(wf: LoadedWorkflow) -> str:
    """Deterministic workflow chunk id: `{scheme}_wf_{workflow_id}`."""
    return f"{wf.scheme}_wf_{wf.workflow_id}"


def _stem(filename: str) -> str:
    """Filename without its extension. Kept as a helper to make ids readable."""
    # `.pdf.pdf` shows up in a handful of files (see scope.md §8 —
    # filename normalisation pending). rsplit once strips the last `.pdf`
    # and leaves any embedded dots alone, which is exactly what we want.
    return filename.rsplit(".", 1)[0]


# --- PDF page joining & page-range recovery ----------------------------------

def _join_pdf_pages_with_markers(
    pdf: LoadedPDFDocument,
) -> tuple[str, list[tuple[int, int]]]:
    """
    Join pages into one text blob and return (blob, page_offsets).

    `page_offsets` is a list of (page_num, offset_in_blob_where_page_starts),
    in ascending order of offset. It is what _extract_page_range_from_chunk
    uses to convert character offsets back into page numbers.
    """
    parts: list[str] = []
    page_offsets: list[tuple[int, int]] = []
    cursor = 0
    for page in pdf.pages:
        marker = PAGE_MARKER_TEMPLATE.format(n=page.page_num)
        parts.append(marker)
        cursor += len(marker)
        # After the marker we are at the first character of this page's
        # content. Record that as the page's start offset.
        page_offsets.append((page.page_num, cursor))
        parts.append(page.text)
        cursor += len(page.text)
    return "".join(parts), page_offsets


def _extract_page_range_from_chunk(
    chunk_start: int,
    chunk_end: int,
    page_offsets: list[tuple[int, int]],
) -> tuple[int, int]:
    """
    Given a chunk's [start, end) offset range in the joined blob, return
    the (page_start, page_end) it spans.

    Logic: page_offsets is sorted ascending by offset. The chunk's start
    page is the last page whose offset is <= chunk_start; the chunk's end
    page is the last page whose offset is < chunk_end (exclusive end).
    """
    start_page = page_offsets[0][0]
    end_page = page_offsets[0][0]
    for page_num, offset in page_offsets:
        if offset <= chunk_start:
            start_page = page_num
        if offset < chunk_end:
            end_page = page_num
        else:
            # page_offsets is sorted — no later page can start before chunk_end.
            break
    return (start_page, end_page)


# --- Character-window slicing -----------------------------------------------

def _window_text(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    """
    Slice `text` into windows of `size` chars with `overlap` between them.
    Returns a list of (start_offset, end_offset_exclusive, chunk_text).

    The last window can be shorter than `size` — we do NOT pad it. A short
    tail window is meaningful (it's the end of the document) and padding
    would inject fake content into retrieval.
    """
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(
            f"overlap must be in [0, size); got size={size} overlap={overlap}"
        )
    if not text:
        return []

    step = size - overlap
    windows: list[tuple[int, int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        windows.append((start, end, text[start:end]))
        if end >= n:
            break
        start += step
    return windows


# --- PDF & workflow chunking -------------------------------------------------

def _chunk_pdf(pdf: LoadedPDFDocument, config) -> list[Chunk]:
    """
    Chunk one PDF. Returns [] if the PDF has no extractable text at all
    (which is what happens on the ~25 non-priority scanned PDFs — see
    scope.md §8). Returns a single small chunk if the PDF has some text
    but less than `chunk_size` chars. Silence would hide V1 losses; visible
    small-or-zero-chunk documents keep them auditable.
    """
    if pdf.total_chars == 0:
        return []

    blob, page_offsets = _join_pdf_pages_with_markers(pdf)
    windows = _window_text(blob, config.chunk_size, config.chunk_overlap)

    chunks: list[Chunk] = []
    for idx, (start, end, chunk_text) in enumerate(windows):
        page_start, page_end = _extract_page_range_from_chunk(
            start, end, page_offsets
        )
        chunks.append(
            Chunk(
                chunk_id=_pdf_chunk_id(pdf, idx, kind="c"),
                text=chunk_text,
                scheme=pdf.scheme,
                source_type="pdf",
                source_filename=pdf.filename,
                source_filepath=pdf.filepath,
                is_ocr_source=pdf.is_ocr_version,
                parent_subfolder=pdf.parent_subfolder,
                page_start=page_start,
                page_end=page_end,
                workflow_id=None,
            )
        )
    return chunks


# --- Structure-aware PDF chunking (Phase 5, fix #1) -------------------------

def _render_table_as_pipes(rows: list[list]) -> str:
    """
    Render a table's `.extract()` output as pipe-delimited lines.

    Format matches lightweight markdown so a reader (human or LLM) sees
    the row/column structure without needing a markdown renderer:

        [TABLE]
        | Header A | Header B | Header C |
        | val 1    | val 2    | val 3    |
        [/TABLE]

    Cells that come back as None (empty cells in the source table)
    render as empty strings, not the literal token "None" — a naked
    "None" in a subsidy table would be an actively wrong retrieval
    result. Newlines inside cells are collapsed to spaces so the row
    stays on one line; a multi-line cell that breaks the pipe grid
    would confuse both a reader and any downstream table-aware parser.
    """
    lines: list[str] = [TABLE_OPEN_MARKER]
    data_row_count = 0
    for row in rows:
        if row is None:
            continue
        cells = [
            ("" if cell is None else str(cell)).replace("\n", " ").strip()
            for cell in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
        data_row_count += 1
        # After the first data row (assumed header), insert a markdown
        # separator so readers (human or LLM) can distinguish the header
        # row from data rows.  Costs nothing in token count and makes
        # the chunk self-describing for generation prompts.
        if data_row_count == 1 and len(rows) > 1:
            lines.append("| " + " | ".join("---" for _ in cells) + " |")
    lines.append(TABLE_CLOSE_MARKER)
    return "\n".join(lines)


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the (x, y) center of a bbox given as (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _center_inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    """
    True iff the center of `inner` sits inside `outer`.

    We use a center-in-bbox test rather than a full IoU because it is:
      * cheap (four float comparisons),
      * robust to small OCR-induced bbox jitter (a text block that
        barely spills past a table border on one side is still
        classified as part of the table),
      * symmetric-enough for our purposes (we only care about
        "is this block a table cell or prose?", not the exact overlap).
    """
    cx, cy = _bbox_center(inner)
    ox0, oy0, ox1, oy1 = outer
    return (ox0 <= cx <= ox1) and (oy0 <= cy <= oy1)


def _chunk_pdf_structure_aware(
    pdf: LoadedPDFDocument, config
) -> tuple[list[Chunk], dict]:
    """
    Structure-aware PDF chunker (Phase 5, fix #1).

    Strategy in words:
      1. Open the PDF with pymupdf. For each page, ask pymupdf for the
         set of detected tables and the set of text blocks (both with
         bounding boxes).
      2. Every detected table is extracted whole and emitted as ONE
         chunk. No size cap — the whole point of this fix is that a
         subsidy table sliced in half is worse than a table that is
         bigger than our usual window.
      3. On the same page, non-table text is every block whose bbox
         center falls OUTSIDE every table bbox. Non-table text is
         appended to a per-PDF buffer with a `[PAGE {n}]` marker.
      4. After walking every page, the non-table buffer is windowed
         with the same fixed 800/120 slider the baseline uses.

    Returns the list of Chunks plus a small dict of per-PDF diagnostics
    (table count, fallback-triggered flag, etc.) so the top-level
    summary can aggregate them.

    Falls back gracefully. If pymupdf refuses to open the file, or
    `find_tables()` raises on a page, we retreat to the baseline
    behaviour for the affected content and record it in `diagnostics`
    so the fallback is auditable rather than silent.
    """
    # Local import so a machine without pymupdf can still import the
    # baseline chunker path (unlikely, but keeps blast radius small).
    try:
        import pymupdf
    except ImportError:
        logger.warning(
            "pymupdf unavailable; falling back to baseline chunking for %s",
            pdf.filename,
        )
        return _chunk_pdf(pdf, config), {
            "tables_detected": 0,
            "text_windows": 0,
            "fallback": "pymupdf_import_failed",
        }

    diagnostics: dict = {
        "tables_detected": 0,
        "text_windows": 0,
        "fallback": None,
        "pages_with_find_tables_error": 0,
    }

    try:
        doc = pymupdf.open(str(pdf.filepath))
    except Exception as exc:
        logger.warning(
            "pymupdf could not open %s (%s); falling back to baseline chunking",
            pdf.filename,
            exc,
        )
        diagnostics["fallback"] = "pymupdf_open_failed"
        return _chunk_pdf(pdf, config), diagnostics

    chunks: list[Chunk] = []
    non_table_parts: list[str] = []
    non_table_page_offsets: list[tuple[int, int]] = []
    non_table_cursor = 0

    # We assign chunk ids by a running index across text windows AND a
    # separate running index across tables. Deterministic order:
    #   * text windows: order in which the windower yields them
    #   * tables: page-major order, then reading order within the page
    #     (pymupdf's own ordering of `tf.tables`, which follows y-first
    #     positioning — matches how a human reads the page)
    table_idx = 0

    for page_num_zero, page in enumerate(doc):
        page_num = page_num_zero + 1  # 1-indexed to match pypdf

        # --- Detect tables on this page ---
        table_bboxes: list[tuple[float, float, float, float]] = []
        try:
            tf = page.find_tables()
            page_tables = tf.tables if tf else []
        except Exception as exc:
            logger.warning(
                "find_tables failed on %s p%d (%s); treating page as pure text",
                pdf.filename,
                page_num,
                exc,
            )
            diagnostics["pages_with_find_tables_error"] += 1
            page_tables = []

        for t in page_tables:
            try:
                rows = t.extract()
            except Exception as exc:
                logger.warning(
                    "table.extract() failed on %s p%d (%s); skipping this table",
                    pdf.filename,
                    page_num,
                    exc,
                )
                continue
            if not rows:
                continue
            table_bboxes.append(tuple(t.bbox))
            rendered = _render_table_as_pipes(rows)
            chunks.append(
                Chunk(
                    chunk_id=_pdf_chunk_id(pdf, table_idx, kind="tbl"),
                    text=rendered,
                    scheme=pdf.scheme,
                    source_type="pdf",
                    source_filename=pdf.filename,
                    source_filepath=pdf.filepath,
                    is_ocr_source=pdf.is_ocr_version,
                    parent_subfolder=pdf.parent_subfolder,
                    page_start=page_num,
                    page_end=page_num,
                    workflow_id=None,
                )
            )
            table_idx += 1
            diagnostics["tables_detected"] += 1

        # --- Non-table text on this page ---
        # Blocks come back as (x0, y0, x1, y1, text, block_no, block_type).
        # `block_type == 0` is a text block; `1` is an image. We only care
        # about text blocks here.
        try:
            blocks = page.get_text("blocks")
        except Exception as exc:
            logger.warning(
                "get_text('blocks') failed on %s p%d (%s); using raw page text",
                pdf.filename,
                page_num,
                exc,
            )
            blocks = []
            raw_page_text = page.get_text() or ""
        else:
            raw_page_text = None  # Signals: reconstruct from blocks below.

        if raw_page_text is None:
            kept: list[str] = []
            for b in blocks:
                # Older pymupdf versions omit block_type; guard for both shapes.
                if len(b) >= 7 and b[6] != 0:
                    continue
                bbox = (b[0], b[1], b[2], b[3])
                text = b[4] or ""
                if not text.strip():
                    continue
                # Known limitation: if pymupdf reports an oversized table
                # bbox that covers real prose, that prose is excluded here
                # and lost (not in the table chunk, not in the text window).
                # Observed once: aif_sep2024 p10 (2 tables reported, 1 real).
                if any(_center_inside(bbox, tb) for tb in table_bboxes):
                    continue
                kept.append(text)
            page_text = "\n".join(kept).strip()
        else:
            page_text = raw_page_text.strip()

        # NOTE: if a page is 100% table (no non-table text survives the
        # center-in-bbox filter), that page is absent from
        # non_table_page_offsets. _extract_page_range_from_chunk will
        # attribute the surrounding text window to the nearest earlier
        # page that DID have text, so page_start/page_end can be
        # off-by-one on table-only pages. This is a known trade-off —
        # page ranges are provenance metadata, not retrieval keys, and
        # adding a zero-width sentinel offset for skipped pages would
        # complicate the offset math for no retrieval benefit.
        if page_text:
            marker = PAGE_MARKER_TEMPLATE.format(n=page_num)
            non_table_parts.append(marker)
            non_table_cursor += len(marker)
            non_table_page_offsets.append((page_num, non_table_cursor))
            non_table_parts.append(page_text)
            non_table_cursor += len(page_text)

    doc.close()

    # --- Window the non-table blob with the baseline slider ---
    blob = "".join(non_table_parts)
    if blob:
        windows = _window_text(blob, config.chunk_size, config.chunk_overlap)
        # A running text-window index scoped to this PDF. Tables use
        # their own `tbl` id namespace so the two do not collide.
        for idx, (start, end, chunk_text) in enumerate(windows):
            if non_table_page_offsets:
                page_start, page_end = _extract_page_range_from_chunk(
                    start, end, non_table_page_offsets
                )
            else:
                page_start = page_end = 1
            chunks.append(
                Chunk(
                    chunk_id=_pdf_chunk_id(pdf, idx, kind="c"),
                    text=chunk_text,
                    scheme=pdf.scheme,
                    source_type="pdf",
                    source_filename=pdf.filename,
                    source_filepath=pdf.filepath,
                    is_ocr_source=pdf.is_ocr_version,
                    parent_subfolder=pdf.parent_subfolder,
                    page_start=page_start,
                    page_end=page_end,
                    workflow_id=None,
                )
            )
            diagnostics["text_windows"] += 1

    return chunks, diagnostics


def _workflow_to_text(wf: LoadedWorkflow) -> str:
    """
    Flatten a workflow's step list into a single readable text blob.

    We include workflow-level metadata (title, source URL) at the top and
    then walk the steps in order. Extra columns beyond the standard
    workflow schema are appended per-step so nothing is silently dropped
    (a scheme's CSV occasionally carries a `notes` column, for example).
    """
    header = (
        f"Workflow: {wf.source_title or wf.workflow_id} ({wf.workflow_id})\n"
        f"Scheme: {wf.scheme}\n"
    )
    if wf.source_url:
        header += f"Source: {wf.source_url}\n"

    standard_keys = {
        "workflow_id",
        "step_no",
        "instruction",
        "input_required",
        "condition",
        "source_title",
        "official_url",
        "source_url",
    }

    step_blocks: list[str] = []
    for step in wf.steps:
        step_no = step.get("step_no", "")
        instruction = step.get("instruction", "")
        input_required = step.get("input_required", "")
        condition = step.get("condition", "")

        block = f"\nStep {step_no}: {instruction}"
        if input_required:
            block += f"\n  Input required: {input_required}"
        if condition:
            block += f"\n  Condition: {condition}"

        # Preserve any non-standard columns rather than dropping them.
        extras = {
            k: v for k, v in step.items()
            if k and k not in standard_keys and v
        }
        for k, v in extras.items():
            block += f"\n  {k}: {v}"
        step_blocks.append(block)

    return header + "".join(step_blocks)


def _chunk_workflow(wf: LoadedWorkflow) -> Chunk:
    """
    One workflow becomes exactly one chunk. See file-level docstring for
    the reason: workflows are procedural units and splitting them would
    break the layer that makes them useful.
    """
    return Chunk(
        chunk_id=_workflow_chunk_id(wf),
        text=_workflow_to_text(wf),
        scheme=wf.scheme,
        source_type="workflow",
        source_filename=wf.filename,
        source_filepath=wf.filepath,
        is_ocr_source=False,
        parent_subfolder=None,
        page_start=None,
        page_end=None,
        workflow_id=wf.workflow_id,
    )


# --- Entry point -------------------------------------------------------------

def chunk_corpus(loaded_corpus: LoadedCorpus, config=settings) -> ChunkedCorpus:
    """
    Chunk every PDF and every workflow in `loaded_corpus`.

    `config` is injectable so tests / notebooks can point at custom
    chunk_size / chunk_overlap without editing settings. All chunk-related
    knobs come from config; no magic numbers appear in this function.
    """
    chunks: list[Chunk] = []
    per_scheme_pdf_chunks: dict[str, int] = {}
    per_scheme_workflow_chunks: dict[str, int] = {}
    zero_chunk_pdfs: list[dict] = []

    # Structure-aware aggregate diagnostics — populated only when the
    # flag is on. Kept out of the summary dict entirely when off so a
    # baseline run's summary looks identical to what it looked like
    # before this feature landed.
    structure_aware_on = bool(
        getattr(config, "use_structure_aware_chunking", False)
    )
    total_tables = 0
    total_text_windows = 0
    fallback_docs: list[dict] = []
    per_doc_table_counts: list[dict] = []

    for pdf in loaded_corpus.pdfs:
        if structure_aware_on:
            pdf_chunks, diag = _chunk_pdf_structure_aware(pdf, config)
            total_tables += diag["tables_detected"]
            total_text_windows += diag["text_windows"]
            if diag["fallback"]:
                fallback_docs.append(
                    {
                        "scheme": pdf.scheme,
                        "filename": pdf.filename,
                        "reason": diag["fallback"],
                    }
                )
            per_doc_table_counts.append(
                {
                    "scheme": pdf.scheme,
                    "filename": pdf.filename,
                    "tables_detected": diag["tables_detected"],
                    "text_windows": diag["text_windows"],
                    "pages_with_find_tables_error": diag[
                        "pages_with_find_tables_error"
                    ],
                }
            )
        else:
            pdf_chunks = _chunk_pdf(pdf, config)
        chunks.extend(pdf_chunks)
        per_scheme_pdf_chunks[pdf.scheme] = (
            per_scheme_pdf_chunks.get(pdf.scheme, 0) + len(pdf_chunks)
        )
        if not pdf_chunks:
            # PDFs that produced no chunks at all — the ~25 non-priority
            # scanned PDFs flagged in scope.md §8. Kept visible in the
            # summary so V1 losses stay auditable.
            zero_chunk_pdfs.append(
                {
                    "scheme": pdf.scheme,
                    "filename": pdf.filename,
                    "total_chars": pdf.total_chars,
                }
            )

    for wf in loaded_corpus.workflows:
        chunks.append(_chunk_workflow(wf))
        per_scheme_workflow_chunks[wf.scheme] = (
            per_scheme_workflow_chunks.get(wf.scheme, 0) + 1
        )

    # --- Summary statistics ---
    pdf_chunk_count = sum(1 for c in chunks if c.source_type == "pdf")
    workflow_chunk_count = sum(1 for c in chunks if c.source_type == "workflow")

    lengths = [len(c.text) for c in chunks]
    avg_len = mean(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    # A chunk is "suspicious" if it is shorter than half the configured
    # chunk size. That surfaces two things worth watching:
    #   (a) file-end tail chunks in normal PDFs — usually fine
    #   (b) whole documents that came out much smaller than expected —
    #       often a sign the PDF was scanned and only a title-page
    #       fragment made it through OCR
    # The threshold isn't a bug filter; it's an eyeball flag for the
    # ablation phase, where an unexpected surge of these might indicate
    # a chunking regression.
    suspicious_threshold = config.chunk_size // 2
    suspicious_chunks = [
        {
            "chunk_id": c.chunk_id,
            "scheme": c.scheme,
            "source_type": c.source_type,
            "source_filename": c.source_filename,
            "length": len(c.text),
        }
        for c in chunks
        if len(c.text) < suspicious_threshold
    ]

    summary = {
        "total_chunks": len(chunks),
        "pdf_chunks": pdf_chunk_count,
        "workflow_chunks": workflow_chunk_count,
        "per_scheme_pdf_chunks": per_scheme_pdf_chunks,
        "per_scheme_workflow_chunks": per_scheme_workflow_chunks,
        "avg_chunk_length": round(avg_len, 1),
        "min_chunk_length": min_len,
        "max_chunk_length": max_len,
        "chunk_size_chars": config.chunk_size,
        "chunk_overlap_chars": config.chunk_overlap,
        "suspicious_chunk_threshold": suspicious_threshold,
        "suspicious_chunks_count": len(suspicious_chunks),
        "suspicious_chunks": suspicious_chunks,
        "zero_chunk_pdfs_count": len(zero_chunk_pdfs),
        "zero_chunk_pdfs": zero_chunk_pdfs,
    }

    if structure_aware_on:
        # Extra block appended only in structure-aware mode so a baseline
        # run's summary stays byte-identical to what it was pre-fix.
        # Tables always start with TABLE_OPEN_MARKER, so a cheap prefix
        # check separates them from text-window chunks without touching
        # ids or metadata.
        table_chunk_lengths = [
            len(c.text)
            for c in chunks
            if c.source_type == "pdf" and c.text.startswith(TABLE_OPEN_MARKER)
        ]
        summary["structure_aware"] = {
            "enabled": True,
            "total_tables_detected": total_tables,
            "total_text_windows": total_text_windows,
            "table_chunk_min_chars": min(table_chunk_lengths) if table_chunk_lengths else 0,
            "table_chunk_max_chars": max(table_chunk_lengths) if table_chunk_lengths else 0,
            "table_chunk_avg_chars": (
                round(mean(table_chunk_lengths), 1) if table_chunk_lengths else 0
            ),
            "table_chunks_over_1500_chars": sum(
                1 for n in table_chunk_lengths if n > 1500
            ),
            "fallback_docs_count": len(fallback_docs),
            "fallback_docs": fallback_docs,
            "per_doc_table_counts": per_doc_table_counts,
        }

    return ChunkedCorpus(chunks=chunks, summary=summary)
