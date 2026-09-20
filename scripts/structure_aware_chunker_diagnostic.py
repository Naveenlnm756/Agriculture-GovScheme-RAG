"""
Structure-aware chunker diagnostic (Phase 5, fix #1).

Runs the new pymupdf-based chunker on the 5 A/B test documents (the
same set the table-detection F1=0.889 was measured on — see
scripts/table_ab_step2_score.py) and reports:

  * chunk-size distribution (min / avg / median / p90 / max)
  * count of table chunks and their size distribution
  * count of non-table text-window chunks (fixed-size fallbacks)
  * per-doc table-and-window breakdown
  * verification: every table chunk starts with `[TABLE]` and closes
    with `[/TABLE]` (i.e. no table was split by the windower)

Bounded to 5 documents on purpose: this is the "before we re-embed
the entire corpus, is the chunker doing what it should?" check. A
full-corpus diagnostic would drown the signal we care about here in
scheme-total noise.

Run from project root:
    python scripts/structure_aware_chunker_diagnostic.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

# Windows console defaults to cp1252, which cannot encode private-use
# glyphs (e.g.  — a Wingdings-style bullet OCR sometimes emits).
# Reconfigure stdout to utf-8 with replacement so a rogue glyph in a
# table header never aborts the whole diagnostic mid-run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from src.config import settings  # noqa: E402
from src.ingestion.chunker import (  # noqa: E402
    TABLE_CLOSE_MARKER,
    TABLE_OPEN_MARKER,
    _chunk_pdf_structure_aware,
)
from src.ingestion.loader import _load_pdf  # noqa: E402


# Same 5 docs as table_ab_step2_score.py so the F1 story and the
# chunking story sit on the same ground truth.
DOCS: list[tuple[str, str, str]] = [
    ("midh_2025", "MIDH",
     "data/raw/MIDH/01_RAW_PDFs/MIDH_Operational_Guideline_2025_OCR.pdf"),
    ("nhb_final", "MIDH",
     "data/raw/MIDH/01_RAW_PDFs/FinalNHBOperationalGuideline_OCR.pdf"),
    ("aif_sep2024", "AIF",
     "data/raw/AIF/01_RAW_PDFs/AIF Revised Scheme Guidelines  September 2024_OCR.pdf"),
    ("pmfby_2020", "PMFBY",
     "data/raw/PMFBY/01_RAW_PDFs/Revamped Operational Guidelines_17th August 2020.pdf"),
    ("rbi_kcc_2026", "KCC",
     "data/raw/KCC/01_RAW_PDFs/RBI_KCC_Directions_2026_Commercial_Banks.pdf"),
]


def _distribution(name: str, values: list[int]) -> str:
    """Compact percentile summary. Empty input renders as '(none)'."""
    if not values:
        return f"{name}: (none)"
    values_sorted = sorted(values)
    n = len(values_sorted)
    p50 = statistics.median(values_sorted)
    p90 = values_sorted[max(0, int(0.9 * n) - 1)]
    return (
        f"{name}: n={n}  min={min(values_sorted)}  "
        f"median={int(p50)}  p90={p90}  max={max(values_sorted)}  "
        f"mean={round(statistics.mean(values_sorted), 1)}"
    )


def main() -> int:
    # Structure-aware flag is per-run in-memory only; do not need to
    # touch .env. The chunker reads it via `getattr(config, ...)`.
    settings.use_structure_aware_chunking = True

    print("=" * 100)
    print("STRUCTURE-AWARE CHUNKER DIAGNOSTIC — 5 test docs")
    print(f"use_structure_aware_chunking = {settings.use_structure_aware_chunking}")
    print(f"chunk_size = {settings.chunk_size}   chunk_overlap = {settings.chunk_overlap}")
    print("=" * 100)

    all_table_lengths: list[int] = []
    all_text_lengths: list[int] = []
    total_tables = 0
    total_text_windows = 0
    split_tables_found: list[str] = []
    fallback_docs: list[tuple[str, str]] = []

    for name, scheme, rel_path in DOCS:
        path = _PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"\n[{name}] MISSING {path}")
            continue

        pdf = _load_pdf(scheme, path, path.parent)
        chunks, diag = _chunk_pdf_structure_aware(pdf, settings)

        table_chunks = [c for c in chunks if c.text.startswith(TABLE_OPEN_MARKER)]
        text_chunks = [c for c in chunks if not c.text.startswith(TABLE_OPEN_MARKER)]

        table_lens = [len(c.text) for c in table_chunks]
        text_lens = [len(c.text) for c in text_chunks]
        all_table_lengths.extend(table_lens)
        all_text_lengths.extend(text_lens)
        total_tables += len(table_chunks)
        total_text_windows += len(text_chunks)

        # Verify no table is split. A well-formed table chunk contains
        # exactly one [TABLE] opener and exactly one [/TABLE] closer,
        # and the closer sits at the end of the chunk. If the windower
        # ever mangled a table, we'd see an opener with no matching
        # closer (or vice versa) somewhere in the text-chunk pool.
        for c in table_chunks:
            opens = c.text.count(TABLE_OPEN_MARKER)
            closes = c.text.count(TABLE_CLOSE_MARKER)
            if opens != 1 or closes != 1 or not c.text.rstrip().endswith(TABLE_CLOSE_MARKER):
                split_tables_found.append(
                    f"{name}::{c.chunk_id} (opens={opens} closes={closes})"
                )
        # And check: no text chunk accidentally contains a table marker.
        # If it does, a table was split into the windower.
        for c in text_chunks:
            if TABLE_OPEN_MARKER in c.text or TABLE_CLOSE_MARKER in c.text:
                split_tables_found.append(
                    f"{name}::{c.chunk_id} (table marker leaked into text chunk)"
                )

        if diag.get("fallback"):
            fallback_docs.append((name, diag["fallback"]))

        print(f"\n{'-' * 100}\n[{name}]  scheme={scheme}  file={path.name}")
        print(f"  total chunks         : {len(chunks)}")
        print(f"  table chunks         : {len(table_chunks)}")
        print(f"  text-window chunks   : {len(text_chunks)}")
        print(f"  pages with find_tables error : "
              f"{diag.get('pages_with_find_tables_error', 0)}")
        if diag.get("fallback"):
            print(f"  FALLBACK             : {diag['fallback']}")
        if table_lens:
            print(f"  {_distribution('table chunk sizes (chars)', table_lens)}")
        if text_lens:
            print(f"  {_distribution('text  chunk sizes (chars)', text_lens)}")
        # Show a few example table first-lines so a reader can eyeball
        # that we're capturing real tabular content (not letter templates
        # or TOCs).
        if table_chunks:
            print("  first row of first 3 table chunks:")
            for c in table_chunks[:3]:
                lines = c.text.splitlines()
                # lines[0] = [TABLE], lines[1] = header row
                header = lines[1] if len(lines) > 1 else "(empty)"
                print(f"    p{c.page_start:>3}  {header[:90]}")

    print("\n" + "=" * 100)
    print("OVERALL (5 docs)")
    print("=" * 100)
    print(f"  total table chunks   : {total_tables}")
    print(f"  total text-window chunks : {total_text_windows}")
    print(f"  {_distribution('table chunk sizes (chars)', all_table_lengths)}")
    print(f"  {_distribution('text  chunk sizes (chars)', all_text_lengths)}")

    over_1500 = sum(1 for n in all_table_lengths if n > 1500)
    print(f"  table chunks over 1500 chars : {over_1500}  "
          "(these are the ones the baseline windower would have split; "
          "preserved whole here)")

    print()
    if split_tables_found:
        print("FAIL — table integrity violations found:")
        for e in split_tables_found:
            print(f"  * {e}")
    else:
        print("PASS — no table was split; every table chunk is bracketed by [TABLE]/[/TABLE].")

    if fallback_docs:
        print()
        print("FALLBACKS TRIGGERED:")
        for name, reason in fallback_docs:
            print(f"  {name}: {reason}")

    return 0 if not split_tables_found else 1


if __name__ == "__main__":
    sys.exit(main())
