"""
Image extractor for the vision ingestion branch (Deliverable 2 B5.4).

Walks the corpus, enumerates every embedded raster image, dedups by
(pdf_path, xref) within-doc and content-hash (SHA-256 of raw bytes)
across the whole corpus, and emits the FULL candidate pool above the
size floor. No candidate cap — the vision classifier decides usefulness,
not this module.

The image chunker downstream reads the emitted list twice:
  1. Once per unique content-hash to send to the vision adapter.
  2. Once per per-page occurrence to materialise one Chunk per page,
     all sharing the same vision-extracted text.

Trivial-decorative shortcut (safe to skip vision on): if a single
content-hash appears on >= `MIN_WATERMARK_PAGES` pages of the same
document, AND its max area is below `WATERMARK_MAX_AREA`, it's a
watermark or per-page branding element. Skipped BEFORE stage-1
classification so we don't burn API budget on obvious noise. Everything
else goes through stage-1.

Nothing in this module writes to Chroma or calls a vision model.
"""

from __future__ import annotations

import hashlib
import io
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from src.config import (
    CORPUS_SCOPE_PDF_SUBDIR,
    CORPUS_SCOPE_SCHEME_SLUGS,
    settings,
)
from src.ingestion.loader import _find_pdfs_for_scheme, _prefer_ocr_sibling


logger = logging.getLogger(__name__)


# --- Tuning constants (stated up-front) -------------------------------------

# Size floor. An image occurrence must cover at least this fraction of
# the page area to be a candidate. Below this and it's almost certainly
# a bullet / tick / logo fragment.
SIZE_FLOOR_AREA_RATIO: float = 0.05

# Watermark shortcut. Same content-hash appearing on many pages of the
# same document, none larger than 30% of a page = per-page branding.
# The 30% ceiling lets us catch even sizeable-looking watermarks
# (e.g. NCCD's cold-chain equipment banner ~30%) without accidentally
# skipping a real full-page figure that legitimately repeats.
MIN_WATERMARK_PAGES: int = 10
WATERMARK_MAX_AREA: float = 0.30

# When rendering the extracted image for the vision API, cap the longest
# side at these pixel counts. Attempt 1 uses `_STANDARD_MAX_PX`; the
# retry-on-empty path (runner-owned) uses `_FALLBACK_MAX_PX`. Both are
# well under Gemini's 20MB payload ceiling.
_STANDARD_MAX_PX: int = 1600
_FALLBACK_MAX_PX: int = 800


# --- Value objects ----------------------------------------------------------

@dataclass(frozen=True)
class ImageOccurrence:
    """One (scheme, PDF, page, xref) triple where an image occurs."""

    scheme: str
    filename: str
    filepath: str
    page: int
    xref: int
    area_ratio: float
    is_ocr_source: bool
    parent_subfolder: str | None


@dataclass
class ImageCandidate:
    """
    One unique visual across the corpus, plus every page-level
    occurrence of it. `content_hash` is the primary key.

    Rendering is lazy — the runner asks for bytes at attempt-appropriate
    resolution via `render_png(max_px=...)`. We store the raw source
    bytes so the resize is one PIL call, not a fresh pymupdf pass.
    """

    content_hash: str
    source_bytes: bytes  # raw image bytes as pymupdf returned them
    source_ext: str  # "png", "jpeg", "jp2" etc — from doc.extract_image
    occurrences: list[ImageOccurrence] = field(default_factory=list)
    # The occurrence used for "representative" fields (biggest area).
    representative: ImageOccurrence | None = None

    def render_png(self, max_px: int = _STANDARD_MAX_PX) -> bytes:
        """
        Return this image encoded as PNG with the longest side capped
        at `max_px`. Small originals pass through untouched (still
        re-encoded to PNG for uniform MIME handling downstream).

        Implementation note: we DO NOT feed `pix.samples` to
        `Image.frombytes` directly, because indexed / grayscale /
        oddly-strided source images produce a samples buffer whose
        length does not match `w*h*channels` for {RGB, RGBA} modes,
        and PIL raises `not enough image data`. Round-tripping
        through pymupdf's own PNG encoder guarantees a well-formed
        RGB/RGBA byte stream that PIL can then open and resize.
        """
        pix = pymupdf.Pixmap(self.source_bytes)
        # CMYK → RGB before PNG encoding.
        if pix.n - pix.alpha >= 4:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        w, h = pix.width, pix.height
        longest = max(w, h)
        if longest <= max_px:
            # Under the cap — encode as PNG straight from pymupdf.
            return pix.tobytes("png")

        # Over the cap — round-trip via PIL to resize.
        from PIL import Image  # deferred so a caller w/o Pillow can
                                # still import this module.
        base_png = pix.tobytes("png")
        img = Image.open(io.BytesIO(base_png))
        scale = max_px / longest
        new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


# --- PDF discovery reused from the loader -----------------------------------

def _discover_pdfs() -> list[tuple[str, Path]]:
    """
    Return (scheme, pdf_path) pairs after OCR-sibling preference.
    Reuses the loader's own helpers so the ingested set matches
    exactly what the text pipeline reads.
    """
    out: list[tuple[str, Path]] = []
    raw_root = settings.data_raw_dir
    for scheme in CORPUS_SCOPE_SCHEME_SLUGS:
        pdf_root = raw_root / scheme / CORPUS_SCOPE_PDF_SUBDIR
        raw = _find_pdfs_for_scheme(pdf_root)
        to_load, _pref = _prefer_ocr_sibling(raw)
        for p in to_load:
            out.append((scheme, p))
    return out


# --- Enumeration ------------------------------------------------------------

def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _enumerate_one_pdf(scheme: str, path: Path) -> dict[str, ImageCandidate]:
    """
    Walk one PDF. For each embedded image occurrence above the size
    floor, hash it and either merge into an existing ImageCandidate
    for that hash or create a new one. Returns a map keyed by content
    hash, scoped to this PDF (the caller merges across PDFs).
    """
    parent = path.parent
    # parent_subfolder mirrors the loader's logic — set only when the
    # PDF sits below an extra folder inside 01_RAW_PDFs/.
    pdf_root = (
        settings.data_raw_dir / scheme / CORPUS_SCOPE_PDF_SUBDIR
    )
    parent_subfolder = parent.name if parent != pdf_root else None
    is_ocr_source = path.stem.endswith("_OCR")

    per_hash: dict[str, ImageCandidate] = {}
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        logger.warning("skip %s: pymupdf.open failed: %s", path.name, exc)
        return per_hash

    try:
        # xref → (sha, source_bytes, source_ext) cache so we hash + read
        # each xref only once even when it's referenced from many pages.
        xref_cache: dict[int, tuple[str, bytes, str]] = {}

        for pnum_zero, page in enumerate(doc):
            page_num = pnum_zero + 1
            page_area = max(page.rect.width * page.rect.height, 1.0)
            try:
                infos = page.get_image_info(xrefs=True) or []
            except Exception:
                continue

            for info in infos:
                xref = info.get("xref") or 0
                bbox = info.get("bbox")
                if not bbox or not xref:
                    continue
                x0, y0, x1, y1 = bbox
                area_ratio = max(
                    (x1 - x0) * (y1 - y0) / page_area, 0.0
                )
                if area_ratio < SIZE_FLOOR_AREA_RATIO:
                    continue

                cached = xref_cache.get(xref)
                if cached is None:
                    try:
                        img = doc.extract_image(xref)
                        src_bytes = img.get("image") or b""
                        src_ext = img.get("ext") or "png"
                    except Exception as exc:
                        logger.debug(
                            "extract_image failed on %s xref %d: %s",
                            path.name, xref, exc,
                        )
                        src_bytes, src_ext = b"", "png"
                    sha = _hash_bytes(src_bytes) if src_bytes else "empty"
                    cached = (sha, src_bytes, src_ext)
                    xref_cache[xref] = cached

                sha, src_bytes, src_ext = cached
                if sha == "empty" or not src_bytes:
                    continue

                occ = ImageOccurrence(
                    scheme=scheme,
                    filename=path.name,
                    filepath=str(path),
                    page=page_num,
                    xref=xref,
                    area_ratio=round(area_ratio, 4),
                    is_ocr_source=is_ocr_source,
                    parent_subfolder=parent_subfolder,
                )
                cand = per_hash.get(sha)
                if cand is None:
                    cand = ImageCandidate(
                        content_hash=sha,
                        source_bytes=src_bytes,
                        source_ext=src_ext,
                        occurrences=[],
                        representative=None,
                    )
                    per_hash[sha] = cand
                cand.occurrences.append(occ)
                if cand.representative is None or occ.area_ratio > cand.representative.area_ratio:
                    cand.representative = occ
    finally:
        doc.close()

    return per_hash


def _is_watermark(candidate: ImageCandidate) -> bool:
    """
    True iff this content-hash looks like a per-page branding element.
    See MIN_WATERMARK_PAGES / WATERMARK_MAX_AREA at top of file for the
    knobs. The check is applied at the DOCUMENT level — a hash that
    appears on 20 pages of one document AND on 2 pages of another
    still fires (all its occurrences are per-page repeats).
    """
    if len(candidate.occurrences) < MIN_WATERMARK_PAGES:
        return False
    max_area = max(o.area_ratio for o in candidate.occurrences)
    return max_area <= WATERMARK_MAX_AREA


# --- Entry point ------------------------------------------------------------

def extract_image_candidates() -> tuple[list[ImageCandidate], dict]:
    """
    Enumerate every embedded raster image in the corpus. Return the
    full candidate list (post-watermark shortcut) plus a diagnostics
    dict for the runner to print.

    O(N) across the corpus, ~3-4 min end-to-end on this box for ~100
    PDFs. Not batchable — the extraction is I/O + pymupdf work.
    """
    pdfs = _discover_pdfs()
    logger.info("image_extractor: %d PDFs (OCR-sibling preferred)", len(pdfs))

    # Merge per-PDF hash maps into a corpus-wide map. Same content-hash
    # across two documents (rare here — dedup diagnostic found zero
    # cross-doc byte-hash matches — but the logic handles it) merges
    # their occurrence lists.
    all_candidates: dict[str, ImageCandidate] = {}
    for i, (scheme, path) in enumerate(pdfs, start=1):
        if i % 20 == 0:
            logger.info("  image_extractor: %d/%d", i, len(pdfs))
        per_hash = _enumerate_one_pdf(scheme, path)
        for sha, cand in per_hash.items():
            existing = all_candidates.get(sha)
            if existing is None:
                all_candidates[sha] = cand
            else:
                existing.occurrences.extend(cand.occurrences)
                if cand.representative and (
                    existing.representative is None
                    or cand.representative.area_ratio > existing.representative.area_ratio
                ):
                    existing.representative = cand.representative

    # Split the pool: watermarks skipped, everything else forwarded.
    kept: list[ImageCandidate] = []
    dropped_watermarks = 0
    dropped_wm_occurrences = 0
    for cand in all_candidates.values():
        if _is_watermark(cand):
            dropped_watermarks += 1
            dropped_wm_occurrences += len(cand.occurrences)
            continue
        kept.append(cand)

    total_occurrences_kept = sum(len(c.occurrences) for c in kept)
    diagnostics = {
        "n_pdfs": len(pdfs),
        "n_unique_candidates_all": len(all_candidates),
        "n_dropped_watermark_candidates": dropped_watermarks,
        "n_dropped_watermark_occurrences": dropped_wm_occurrences,
        "n_kept_candidates": len(kept),
        "n_kept_occurrences": total_occurrences_kept,
        "size_floor_area_ratio": SIZE_FLOOR_AREA_RATIO,
        "min_watermark_pages": MIN_WATERMARK_PAGES,
        "watermark_max_area": WATERMARK_MAX_AREA,
    }
    logger.info(
        "image_extractor: %d unique candidates kept (%d occurrences); "
        "%d watermarks dropped (%d occurrences)",
        len(kept), total_occurrences_kept,
        dropped_watermarks, dropped_wm_occurrences,
    )
    return kept, diagnostics
