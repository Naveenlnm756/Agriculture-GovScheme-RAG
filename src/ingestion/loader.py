"""
Corpus loader for the Agriculture Schemes RAG project.

Walks `data/raw/[SCHEME]/{01_RAW_PDFs,02_Workflows}/` for every scheme
listed in `CORPUS_SCOPE_SCHEME_SLUGS` and returns a normalized in-memory
representation, so downstream stages (chunker, embedder) work off a clean
data shape regardless of source file quirks.

Two ingestion rules live here, both from DECISIONS.md:

  1. **`_OCR.pdf` preference.** When both `X.pdf` and `X_OCR.pdf` exist
     in the same folder, load ONLY the `_OCR` sibling. The original is
     preserved on disk as an audit artefact but is not ingested.

  2. **`_index.csv` skip.** Workflow CSVs whose filename ends with
     `_index.csv` are summary indexes for humans (workflow_name,
     category, status) — not per-step workflows — and are ignored.

Individual file failures are recorded in the summary and never abort
the whole load. That way a single malformed PDF cannot silently drop
an entire scheme.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from pypdf import PdfReader

from src.config import (
    CORPUS_SCOPE_PDF_SUBDIR,
    CORPUS_SCOPE_SCHEME_SLUGS,
    CORPUS_SCOPE_WORKFLOW_SUBDIR,
    settings,
)
from src.ingestion.models import (
    LoadedCorpus,
    LoadedPDFDocument,
    LoadedWorkflow,
    PDFPage,
)


logger = logging.getLogger(__name__)

OCR_SUFFIX = "_OCR"
EMPTY_PDF_THRESHOLD_CHARS = 100


# --- PDF discovery -----------------------------------------------------------

def _find_pdfs_for_scheme(pdf_root: Path) -> list[Path]:
    """
    Return every PDF under `pdf_root`, recursively.

    Recursion is what makes PM_KISAN work — its PDFs live one level deeper
    (`01_RAW_PDFs/PM-KISAN/`, `01_RAW_PDFs/PM_KMY/`). For the six flat
    schemes, `rglob` returns the same set as `glob`, so recursion is a
    harmless no-op there.
    """
    if not pdf_root.exists():
        return []
    return sorted(pdf_root.rglob("*.pdf"))


def _prefer_ocr_sibling(
    pdf_paths: list[Path],
) -> tuple[list[Path], list[tuple[Path, Path]]]:
    """
    Apply the DECISIONS.md `_OCR.pdf` preference rule.

    For every `X_OCR.pdf` found, the corresponding `X.pdf` in the same
    directory is added to a skip set. The final load list is the input
    minus the skip set.

    Returns:
      to_load    — PDFs the loader will actually read
      preferred  — (original, ocr_sibling) pairs for the audit log
    """
    skip: set[Path] = set()
    preferred: list[tuple[Path, Path]] = []
    pdf_set = set(pdf_paths)

    for p in pdf_paths:
        if not p.stem.endswith(OCR_SUFFIX):
            continue
        original_stem = p.stem[: -len(OCR_SUFFIX)]
        original = p.with_name(original_stem + p.suffix)
        if original in pdf_set:
            skip.add(original)
            preferred.append((original, p))

    to_load = [p for p in pdf_paths if p not in skip]
    return to_load, preferred


# --- PDF loading -------------------------------------------------------------

def _load_pdf(scheme: str, pdf_path: Path, pdf_root: Path) -> LoadedPDFDocument:
    """
    Read one PDF and return a `LoadedPDFDocument`.

    A per-page `try/except` keeps a single mangled page from killing the
    whole document — the page is recorded with empty text and a warning
    is logged. A truly broken PDF (fails at `PdfReader(...)`) raises out
    and is caught by the caller, which records it as a failure.
    """
    reader = PdfReader(str(pdf_path))
    pages: list[PDFPage] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "page extract failed: scheme=%s file=%s page=%d err=%r",
                scheme, pdf_path.name, i, exc,
            )
            text = ""
        pages.append(PDFPage(page_num=i, text=text))

    parent = pdf_path.parent
    parent_subfolder = parent.name if parent != pdf_root else None

    total_chars = sum(len(p.text) for p in pages)

    return LoadedPDFDocument(
        scheme=scheme,
        filename=pdf_path.name,
        filepath=pdf_path,
        is_ocr_version=pdf_path.stem.endswith(OCR_SUFFIX),
        parent_subfolder=parent_subfolder,
        pages=pages,
        total_pages=len(pages),
        total_chars=total_chars,
    )


# --- Workflow CSV discovery + loading ---------------------------------------

def _find_workflows_for_scheme(workflow_root: Path) -> list[Path]:
    """
    Return per-step workflow CSVs under `workflow_root`.

    Skips `*_index.csv` per DECISIONS.md — those are summary indexes
    (workflow_name, category, status) meant for humans, not the loader.
    """
    if not workflow_root.exists():
        return []
    return [
        p for p in sorted(workflow_root.glob("*.csv"))
        if not p.name.endswith("_index.csv")
    ]


def _normalize_csv_value(v: object) -> str:
    """
    Coerce a `csv.DictReader` value to a stripped string.

    DictReader hands back three shapes:
      - `str`   — normal case; strip whitespace.
      - `None`  — missing value; return empty string.
      - `list`  — overflow: appears under the special `None` key when a
                  row has more fields than the header defines. Join the
                  list so the data is preserved rather than lost.
    """
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(str(x).strip() for x in v)
    return str(v).strip()


def _load_workflow_csv(scheme: str, csv_path: Path) -> LoadedWorkflow:
    """
    Parse one workflow CSV into a `LoadedWorkflow`.

    Uses `utf-8-sig` to strip Windows/Excel BOMs cleanly. Values are
    whitespace-trimmed. Workflow-level metadata is taken from the first
    row — the workflow-CSV schema (scope.md §3) repeats these columns
    identically across every row of one file.
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        # Row 1 is the header; data rows start at 2 (used only for warnings).
        for row_index, row in enumerate(reader, start=2):
            if any(isinstance(v, list) for v in row.values()):
                # `csv.DictReader` puts overflow fields into a list under the
                # `None` key when a data row has more fields than the header
                # (usually an un-quoted delimiter inside a source cell). We
                # preserve the data by joining, but flag it so the CSV can
                # be fixed at the source — the row's column alignment for
                # columns after the overflow point is not trustworthy.
                logger.warning(
                    "workflow CSV row has more fields than header "
                    "(data preserved but column alignment may be off): "
                    "file=%s row=%d",
                    csv_path.name, row_index,
                )
            rows.append(
                {
                    (k or "").strip(): _normalize_csv_value(v)
                    for k, v in row.items()
                }
            )

    first = rows[0] if rows else {}
    workflow_id = first.get("workflow_id") or csv_path.stem
    source_title = first.get("source_title", "")
    # CSV column is `official_url` (scope.md §3); the model exposes it as
    # `source_url` per the Phase 2 loader spec. Fall back to `source_url`
    # just in case a CSV uses that column name.
    source_url = first.get("official_url", "") or first.get("source_url", "")

    return LoadedWorkflow(
        scheme=scheme,
        workflow_id=workflow_id,
        filename=csv_path.name,
        filepath=csv_path,
        steps=rows,
        source_title=source_title,
        source_url=source_url,
        step_count=len(rows),
    )


# --- Entry point -------------------------------------------------------------

def load_corpus(config=settings) -> LoadedCorpus:
    """
    Walk `data/raw/` and load every scheme's PDFs + workflow CSVs.

    `config` defaults to the project settings singleton but is injectable
    so tests can point at a smaller corpus. All paths, subfolder names,
    and the scheme list come from config — nothing about the corpus
    layout is hardcoded in this function.
    """
    raw_root: Path = config.data_raw_dir

    pdfs: list[LoadedPDFDocument] = []
    workflows: list[LoadedWorkflow] = []
    failures: list[dict] = []
    ocr_preferences: list[dict] = []
    per_scheme_pdf_counts: dict[str, int] = {}
    per_scheme_workflow_counts: dict[str, int] = {}

    for scheme in CORPUS_SCOPE_SCHEME_SLUGS:
        scheme_dir = raw_root / scheme
        pdf_root = scheme_dir / CORPUS_SCOPE_PDF_SUBDIR
        workflow_root = scheme_dir / CORPUS_SCOPE_WORKFLOW_SUBDIR

        # --- PDFs ---
        raw_pdf_paths = _find_pdfs_for_scheme(pdf_root)
        to_load, preferred = _prefer_ocr_sibling(raw_pdf_paths)
        for original, ocr in preferred:
            ocr_preferences.append(
                {
                    "scheme": scheme,
                    "original": str(original.relative_to(raw_root)),
                    "preferred_ocr": str(ocr.relative_to(raw_root)),
                }
            )

        loaded_pdf_count = 0
        for pdf_path in to_load:
            try:
                doc = _load_pdf(scheme, pdf_path, pdf_root)
                pdfs.append(doc)
                loaded_pdf_count += 1
            except Exception as exc:
                logger.error(
                    "PDF load failed: scheme=%s file=%s err=%r",
                    scheme, pdf_path.name, exc,
                )
                failures.append(
                    {
                        "kind": "pdf",
                        "scheme": scheme,
                        "filepath": str(pdf_path.relative_to(raw_root)),
                        "error": repr(exc),
                    }
                )
        per_scheme_pdf_counts[scheme] = loaded_pdf_count

        # --- Workflow CSVs ---
        loaded_workflow_count = 0
        for csv_path in _find_workflows_for_scheme(workflow_root):
            try:
                wf = _load_workflow_csv(scheme, csv_path)
                workflows.append(wf)
                loaded_workflow_count += 1
            except Exception as exc:
                logger.error(
                    "workflow load failed: scheme=%s file=%s err=%r",
                    scheme, csv_path.name, exc,
                )
                failures.append(
                    {
                        "kind": "workflow",
                        "scheme": scheme,
                        "filepath": str(csv_path.relative_to(raw_root)),
                        "error": repr(exc),
                    }
                )
        per_scheme_workflow_counts[scheme] = loaded_workflow_count

    empty_pdfs = [
        {
            "scheme": d.scheme,
            "filepath": str(d.filepath.relative_to(raw_root)),
            "total_chars": d.total_chars,
        }
        for d in pdfs
        if d.total_chars < EMPTY_PDF_THRESHOLD_CHARS
    ]

    summary = {
        "per_scheme_pdf_counts": per_scheme_pdf_counts,
        "per_scheme_workflow_counts": per_scheme_workflow_counts,
        "total_pdfs": len(pdfs),
        "total_workflows": len(workflows),
        "total_pages": sum(d.total_pages for d in pdfs),
        "total_chars": sum(d.total_chars for d in pdfs),
        "empty_pdfs": empty_pdfs,
        "ocr_preferences": ocr_preferences,
        "failures": failures,
    }

    return LoadedCorpus(pdfs=pdfs, workflows=workflows, summary=summary)
