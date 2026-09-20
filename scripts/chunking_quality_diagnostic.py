"""
Chunking-quality diagnostic — compare baseline vs structure-aware chunking
on the same 5 test documents used in the table A/B experiment.

Loads only the 5 test PDFs directly (not the full corpus) for speed.

Reports:
  1. Chunk-count comparison (baseline vs structure-aware)
  2. Table-chunk inventory (page, chars, first-row preview)
  3. Content-preservation check (total chars baseline blob vs v2 sum)
  4. Suspicious table chunks (<50 chars or >3000 chars)

Run from the project root:
    python scripts/chunking_quality_diagnostic.py
"""

from __future__ import annotations

import logging
import sys
from io import StringIO
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.chunker import (  # noqa: E402
    TABLE_OPEN_MARKER,
    _chunk_pdf,
    _chunk_pdf_structure_aware,
    _join_pdf_pages_with_markers,
)
from src.ingestion.loader import _load_pdf  # noqa: E402

logger = logging.getLogger(__name__)

# The same 5 test documents from the table A/B experiment.
# Each tuple: (alias, relative_path, scheme)
TEST_DOCS: list[tuple[str, str, str]] = [
    ("midh_2025",    "data/raw/MIDH/01_RAW_PDFs/MIDH_Operational_Guideline_2025_OCR.pdf",                          "MIDH"),
    ("nhb_final",    "data/raw/MIDH/01_RAW_PDFs/FinalNHBOperationalGuideline_OCR.pdf",                             "MIDH"),
    ("aif_sep2024",  "data/raw/AIF/01_RAW_PDFs/AIF Revised Scheme Guidelines  September 2024_OCR.pdf",             "AIF"),
    ("pmfby_2020",   "data/raw/PMFBY/01_RAW_PDFs/Revamped Operational Guidelines_17th August 2020.pdf",            "PMFBY"),
    ("rbi_kcc_2026", "data/raw/KCC/01_RAW_PDFs/RBI_KCC_Directions_2026_Commercial_Banks.pdf",                      "KCC"),
]

OUT_PATH = PROJECT_ROOT / "scripts" / "chunking_quality_diagnostic_output.txt"


def _run_diagnostic(out: StringIO) -> None:
    out.write("=" * 100 + "\n")
    out.write("CHUNKING-QUALITY DIAGNOSTIC — baseline vs structure-aware\n")
    out.write(f"5 test documents from table A/B experiment\n")
    out.write(f"chunk_size={settings.chunk_size}  chunk_overlap={settings.chunk_overlap}\n")
    out.write("=" * 100 + "\n\n")

    # Aggregate counters
    total_baseline_chunks = 0
    total_v2_chunks = 0
    total_table_chunks = 0
    total_text_windows = 0
    all_table_chunk_sizes: list[int] = []
    content_flags: list[str] = []
    suspicious_flags: list[str] = []

    for alias, rel_path, scheme in TEST_DOCS:
        pdf_path = PROJECT_ROOT / rel_path
        if not pdf_path.exists():
            out.write(f"\n{'─'*80}\n[{alias}] *** NOT FOUND: {pdf_path}\n")
            continue

        # Load just this one PDF using the loader's internal function
        print(f"  Loading {alias} ...", flush=True)
        pdf_root = pdf_path.parent
        pdf = _load_pdf(scheme, pdf_path, pdf_root)

        out.write(f"\n{'─'*80}\n")
        out.write(f"[{alias}]  {pdf.filename}\n")
        out.write(f"  scheme={pdf.scheme}  pages={len(pdf.pages)}  total_chars={pdf.total_chars:,}\n")
        out.write(f"{'─'*80}\n")

        # --- Baseline chunking ---
        print(f"  Chunking baseline ...", flush=True)
        baseline_chunks = _chunk_pdf(pdf, settings)
        baseline_total_chars = sum(len(c.text) for c in baseline_chunks)

        # --- Structure-aware chunking ---
        print(f"  Chunking structure-aware ...", flush=True)
        v2_chunks, diag = _chunk_pdf_structure_aware(pdf, settings)
        v2_table_chunks = [c for c in v2_chunks if c.text.startswith(TABLE_OPEN_MARKER)]
        v2_text_chunks = [c for c in v2_chunks if not c.text.startswith(TABLE_OPEN_MARKER)]

        v2_total_chars = sum(len(c.text) for c in v2_chunks)
        v2_table_chars = sum(len(c.text) for c in v2_table_chunks)
        v2_text_chars = sum(len(c.text) for c in v2_text_chunks)

        total_baseline_chunks += len(baseline_chunks)
        total_v2_chunks += len(v2_chunks)
        total_table_chunks += len(v2_table_chunks)
        total_text_windows += len(v2_text_chunks)
        all_table_chunk_sizes.extend(len(c.text) for c in v2_table_chunks)

        # --- 1. Chunk counts ---
        out.write(f"\n  1. CHUNK COUNTS\n")
        out.write(f"     Baseline chunks:         {len(baseline_chunks):>5}\n")
        out.write(f"     Structure-aware chunks:   {len(v2_chunks):>5}  "
                  f"(table={len(v2_table_chunks)}, text={len(v2_text_chunks)})\n")
        delta = len(v2_chunks) - len(baseline_chunks)
        out.write(f"     Delta:                   {delta:>+5}\n")
        if diag.get("fallback"):
            out.write(f"     ⚠ FALLBACK: {diag['fallback']}\n")

        # --- 2. Table chunk inventory ---
        out.write(f"\n  2. TABLE CHUNKS ({len(v2_table_chunks)} total, "
                  f"{diag['tables_detected']} tables detected by pymupdf)\n")
        for i, tc in enumerate(v2_table_chunks):
            # Extract first content line (skip [TABLE] marker)
            lines = tc.text.split("\n")
            first_content = ""
            for line in lines:
                stripped = line.strip()
                if stripped and stripped != TABLE_OPEN_MARKER:
                    first_content = stripped[:80]
                    break
            out.write(f"     tbl[{i}]  page={tc.page_start}  chars={len(tc.text):>5}  "
                      f"first_row: {first_content}\n")

        # --- 3. Content preservation ---
        blob, _ = _join_pdf_pages_with_markers(pdf)
        blob_chars = len(blob)
        pct_diff = abs(v2_total_chars - baseline_total_chars) / max(baseline_total_chars, 1) * 100
        out.write(f"\n  3. CONTENT PRESERVATION\n")
        out.write(f"     Baseline blob chars:     {blob_chars:>8,}\n")
        out.write(f"     Baseline chunk chars:    {baseline_total_chars:>8,}\n")
        out.write(f"     V2 total chunk chars:    {v2_total_chars:>8,}  "
                  f"(table={v2_table_chars:,} + text={v2_text_chars:,})\n")
        out.write(f"     |Δ| vs baseline chunks:  {abs(v2_total_chars - baseline_total_chars):>8,} "
                  f"({pct_diff:.1f}%)\n")
        if pct_diff > 5:
            flag = f"[{alias}] Content delta {pct_diff:.1f}% > 5% threshold"
            content_flags.append(flag)
            out.write(f"     ⚠ FLAG: {flag}\n")

        # --- 4. Suspicious table chunks ---
        out.write(f"\n  4. SUSPICIOUS TABLE CHUNKS\n")
        has_suspicious = False
        for i, tc in enumerate(v2_table_chunks):
            tc_len = len(tc.text)
            if tc_len < 50:
                flag = f"[{alias}] tbl[{i}] page={tc.page_start} only {tc_len} chars — likely false positive"
                suspicious_flags.append(flag)
                out.write(f"     ⚠ TOO SHORT: {flag}\n")
                has_suspicious = True
            if tc_len > 3000:
                flag = f"[{alias}] tbl[{i}] page={tc.page_start} has {tc_len} chars — inspect for quality"
                suspicious_flags.append(flag)
                out.write(f"     ⚠ VERY LARGE: {flag}\n")
                has_suspicious = True
        if not has_suspicious:
            out.write(f"     (none)\n")

    # --- Aggregate summary ---
    out.write(f"\n\n{'='*100}\n")
    out.write(f"AGGREGATE SUMMARY\n")
    out.write(f"{'='*100}\n")
    out.write(f"  Total baseline chunks:        {total_baseline_chunks}\n")
    out.write(f"  Total structure-aware chunks:  {total_v2_chunks}  "
              f"(table={total_table_chunks}, text={total_text_windows})\n")
    out.write(f"  Delta:                         {total_v2_chunks - total_baseline_chunks:+d}\n")
    if all_table_chunk_sizes:
        out.write(f"  Table chunk sizes:  "
                  f"min={min(all_table_chunk_sizes)}  "
                  f"max={max(all_table_chunk_sizes)}  "
                  f"avg={mean(all_table_chunk_sizes):.0f}\n")
    else:
        out.write(f"  Table chunk sizes:  (no table chunks found)\n")

    if content_flags:
        out.write(f"\n  CONTENT PRESERVATION FLAGS ({len(content_flags)}):\n")
        for f in content_flags:
            out.write(f"    ⚠ {f}\n")
    else:
        out.write(f"\n  CONTENT PRESERVATION: ALL PASSED (Δ < 5%)\n")

    if suspicious_flags:
        out.write(f"\n  SUSPICIOUS TABLE FLAGS ({len(suspicious_flags)}):\n")
        for f in suspicious_flags:
            out.write(f"    ⚠ {f}\n")
    else:
        out.write(f"\n  SUSPICIOUS TABLES: NONE\n")

    out.write(f"\n{'='*100}\n")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print("Running chunking-quality diagnostic on 5 test docs...")
    buf = StringIO()
    _run_diagnostic(buf)
    output = buf.getvalue()

    # Write to file and stdout
    OUT_PATH.write_text(output, encoding="utf-8")
    print(output)
    print(f"\n[wrote] {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
