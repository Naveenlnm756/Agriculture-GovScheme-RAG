"""
OCR-first pass: run ocrmypdf on the 22 non-OCR'd PDFs identified by the
image-modality diagnostic as having real scanned-content gaps.

Approach:
  * For each candidate, output a new file `<stem>_OCR.pdf` next to the
    original in data/raw/[SCHEME]/01_RAW_PDFs/. The loader's
    `_prefer_ocr_sibling` rule then picks up the _OCR file
    automatically on the next ingestion.
  * `--skip-text` so ocrmypdf does not re-OCR pages that already have a
    text layer (relevant for the mixed docs like PMFBY WINDS Manual
    where 4/134 pages are scanned).
  * `--language eng+hin` because tesseract has both packs installed
    and government letters occasionally contain Hindi headers /
    stamps (`सत्यमेव जयते`, seals).
  * IDEMPOTENT: skip a candidate if the `_OCR.pdf` sibling already
    exists on disk. Lets us re-run this script safely.
  * Individual failures do NOT abort the whole run — logged and
    counted, we keep going. A single broken PDF cannot deny the
    other 21 their OCR pass.

Read-only outside `data/raw/`. Writes one `_OCR.pdf` sibling per
success. Never modifies the original.

Run:
    .venv/Scripts/python.exe scripts/ocr_first_pass.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

DIAG_JSON = _PROJECT_ROOT / "eval" / "results" / "image_modality" / "summary.json"
OCRMYPDF = _PROJECT_ROOT / ".venv" / "Scripts" / "ocrmypdf.exe"

# Log path — audit trail per-doc: success / failure / skip + time.
LOG_PATH = _PROJECT_ROOT / "eval" / "results" / "image_modality" / "ocr_first_pass_log.txt"


def _select_candidates() -> list[dict]:
    """Reproduce the OCR-CANDIDATE bucket from _ocr_candidate_filter.py."""
    data = json.load(DIAG_JSON.open(encoding="utf-8"))
    out: list[dict] = []
    for pdf in data["per_pdf"]:
        if "error" in pdf:
            continue
        if "_OCR" in pdf["filename"]:
            continue
        n_scanned = pdf.get("n_scanned_pages", 0)
        n_pages = pdf["n_pages"]
        if n_scanned <= 0:
            continue
        frac = n_scanned / max(n_pages, 1)
        # Same rules as _ocr_candidate_filter.py, minus the "INSPECT"
        # branch (both INSPECT cases were confirmed decorative covers
        # in the visual audit).
        if frac >= 0.5:
            keep = True
        elif n_pages <= 30 and frac >= 0.1:
            keep = True
        elif n_pages <= 10 and n_scanned >= 2:
            keep = True
        elif n_scanned <= 2 and n_pages >= 50:
            keep = False  # decorative covers per visual audit
        else:
            keep = True
        if keep:
            out.append(pdf)
    return out


def _ocr_output_path(src: Path) -> Path:
    """`foo.pdf` → `foo_OCR.pdf` next to the original."""
    return src.with_name(src.stem + "_OCR.pdf")


def _run_ocrmypdf(src: Path, dst: Path) -> tuple[bool, str, float]:
    """
    Invoke ocrmypdf. Returns (ok, message, elapsed_s).

    We shell out to the console script rather than importing
    ocrmypdf.api directly so a hang inside tesseract can be watched
    via psutil / task manager, and so the invocation is exactly what
    a human would run from a terminal (defensible in an interview).
    """
    if not OCRMYPDF.exists():
        return False, f"ocrmypdf executable not found at {OCRMYPDF}", 0.0
    t0 = time.time()
    try:
        result = subprocess.run(
            [
                str(OCRMYPDF),
                "--skip-text",       # leave already-text pages alone
                "--language", "eng+hin",
                "--output-type", "pdf",
                "--quiet",
                str(src),
                str(dst),
            ],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min ceiling per file; scanned SMAM = 45 pages
        )
    except subprocess.TimeoutExpired:
        return False, "timeout after 1800s", time.time() - t0
    except Exception as exc:
        return False, f"subprocess raised: {exc}", time.time() - t0
    elapsed = time.time() - t0
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()[-3:]  # last 3 lines
        return False, f"rc={result.returncode}: " + " | ".join(stderr), elapsed
    return True, "ok", elapsed


def main() -> int:
    candidates = _select_candidates()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    print(f"OCR-first pass — {len(candidates)} candidate PDFs")
    print(f"log → {LOG_PATH}")
    print()

    n_ok = n_skip = n_fail = 0
    total_elapsed = 0.0

    for i, pdf in enumerate(candidates, start=1):
        src = Path(pdf["filepath"])
        dst = _ocr_output_path(src)
        rel_src = src.relative_to(_PROJECT_ROOT)

        if dst.exists():
            msg = f"[{i:>2}/{len(candidates)}] SKIP  {rel_src}  (sibling exists)"
            print(msg)
            log_lines.append(msg)
            n_skip += 1
            continue

        if not src.exists():
            msg = f"[{i:>2}/{len(candidates)}] FAIL  {rel_src}  source missing"
            print(msg)
            log_lines.append(msg)
            n_fail += 1
            continue

        print(f"[{i:>2}/{len(candidates)}] OCR   {rel_src} "
              f"({pdf['n_scanned_pages']}/{pdf['n_pages']} scanned)")
        ok, message, elapsed = _run_ocrmypdf(src, dst)
        total_elapsed += elapsed
        rel_dst = dst.relative_to(_PROJECT_ROOT)
        line = (
            f"[{i:>2}/{len(candidates)}] "
            f"{'OK  ' if ok else 'FAIL'}  {rel_dst}  "
            f"elapsed={elapsed:.1f}s  {message}"
        )
        print("   " + line)
        log_lines.append(line)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            # If ocrmypdf failed midway, remove any partial output so
            # a re-run doesn't skip the file thinking it's done.
            if dst.exists():
                try:
                    dst.unlink()
                except OSError:
                    pass

    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print()
    print("=" * 72)
    print(f"OCR pass complete. ok={n_ok}  skipped={n_skip}  failed={n_fail}   "
          f"total elapsed = {total_elapsed:.1f}s")
    print("=" * 72)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
