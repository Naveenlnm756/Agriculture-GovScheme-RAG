"""
Retry the 6 PMFBY PDFs that ocrmypdf's Ghostscript rasterizer refused.

Strategy: `pikepdf.open(bad).save(clean)` first — this rewrites the PDF
into a canonical form, which fixes malformed xref / object stream issues
that trip Ghostscript. Then feed the repaired PDF to ocrmypdf.

The repaired file is written to a scratch dir (NOT into data/raw/), so
we never touch the original nor introduce a `_repaired.pdf` sibling.
ocrmypdf's output IS written to data/raw/ as the `_OCR.pdf` sibling of
the ORIGINAL filename — that's what the loader will pick up.

Idempotent: skip if the `_OCR.pdf` sibling already exists (i.e. the
first-pass succeeded or a previous retry did).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pikepdf

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

OCRMYPDF = _PROJECT_ROOT / ".venv" / "Scripts" / "ocrmypdf.exe"

# The 6 files that failed with Ghostscript rc=7 on the previous pass.
FAILURES = [
    "data/raw/PMFBY/01_RAW_PDFs/Cut off for Data Entry by Banks.pdf",
    "data/raw/PMFBY/01_RAW_PDFs/Letter_28oct2020_instructions_claims_premium.pdf",
    "data/raw/PMFBY/01_RAW_PDFs/Letter_28oct2020_recon_challan_statement.pdf",
    "data/raw/PMFBY/01_RAW_PDFs/Letter_28oct2020_subsidy_through_challan.pdf",
    "data/raw/PMFBY/01_RAW_PDFs/Online transmission of farmers' premium from the Banks to the Insurance Companies.pdf",
    "data/raw/PMFBY/01_RAW_PDFs/Reopening of NCIP for Rabi 19-20.pdf",
]

SCRATCH_DIR = _PROJECT_ROOT / "eval" / "results" / "image_modality" / "ocr_repair_scratch"
LOG_PATH = _PROJECT_ROOT / "eval" / "results" / "image_modality" / "ocr_repair_log.txt"


def _repair_pdf(src: Path, dst: Path) -> tuple[bool, str]:
    """Open with pikepdf and re-save. Fixes most malformed-xref / object
    stream issues by rewriting the PDF into a canonical form.

    Returns (ok, message)."""
    try:
        with pikepdf.open(src) as pdf:
            pdf.save(dst, linearize=True)
        return True, "repaired"
    except Exception as exc:
        return False, f"pikepdf.open/save failed: {exc}"


def _run_ocrmypdf(src: Path, dst: Path) -> tuple[bool, str, float]:
    if not OCRMYPDF.exists():
        return False, f"ocrmypdf not found at {OCRMYPDF}", 0.0
    t0 = time.time()
    try:
        result = subprocess.run(
            [
                str(OCRMYPDF),
                "--skip-text",
                "--language", "eng+hin",
                "--output-type", "pdf",
                "--quiet",
                str(src),
                str(dst),
            ],
            capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout", time.time() - t0
    except Exception as exc:
        return False, f"subprocess raised: {exc}", time.time() - t0
    elapsed = time.time() - t0
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-3:]
        return False, f"rc={result.returncode}: " + " | ".join(tail), elapsed
    return True, "ok", elapsed


def main() -> int:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    n_ok = n_skip = n_fail = 0

    for i, rel in enumerate(FAILURES, start=1):
        original = _PROJECT_ROOT / rel
        ocr_sibling = original.with_name(original.stem + "_OCR.pdf")

        if ocr_sibling.exists():
            msg = f"[{i}/{len(FAILURES)}] SKIP  {rel}  (sibling already exists)"
            print(msg); log_lines.append(msg); n_skip += 1
            continue

        # Step 1: pikepdf repair to scratch
        repaired = SCRATCH_DIR / (original.stem + "_repaired.pdf")
        print(f"[{i}/{len(FAILURES)}] REPAIR {rel}")
        ok, msg = _repair_pdf(original, repaired)
        if not ok:
            line = f"[{i}/{len(FAILURES)}] FAIL   pikepdf repair: {msg}"
            print("   " + line); log_lines.append(line); n_fail += 1
            continue

        # Step 2: ocrmypdf the repaired file → _OCR.pdf sibling of ORIGINAL
        print(f"[{i}/{len(FAILURES)}] OCR    {repaired.name}")
        ok, msg, elapsed = _run_ocrmypdf(repaired, ocr_sibling)
        if ok:
            line = (f"[{i}/{len(FAILURES)}] OK    "
                    f"{ocr_sibling.relative_to(_PROJECT_ROOT)}  "
                    f"elapsed={elapsed:.1f}s (via pikepdf repair)")
            n_ok += 1
        else:
            line = (f"[{i}/{len(FAILURES)}] FAIL  "
                    f"{ocr_sibling.name}  elapsed={elapsed:.1f}s  {msg}")
            n_fail += 1
            # Cleanup partial output
            if ocr_sibling.exists():
                try: ocr_sibling.unlink()
                except OSError: pass
        print("   " + line)
        log_lines.append(line)

    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print()
    print(f"Repair-and-retry complete. ok={n_ok}  skipped={n_skip}  failed={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
