"""
Equivalence check for the hybrid.py refactor (Phase 5, fix #3 prep).

Runs hybrid_search on a fixed 5-question sample and compares its
output against a saved JSON snapshot. Used to prove that extracting
the RRF math into src/retrieval/rrf.py did NOT change hybrid_search's
observable output — because the hybrid ablation row is LOCKED and
its 78-question anchor summary must remain the same.

Usage
-----
    # Capture snapshot BEFORE refactoring hybrid.py:
    python scripts/hybrid_equivalence_check.py capture

    # Verify AFTER refactoring hybrid.py:
    python scripts/hybrid_equivalence_check.py verify

The capture-then-verify pattern mirrors
`eval/equivalence_check_baseline_after_refactor.py`, which was
written for the same purpose against the baseline retriever refactor.

Snapshot compares the full audit tuple (chunk_id, rank,
semantic_rank, bm25_rank, similarity_score, bm25_score, rrf_score)
for every returned result. Floats are compared to 12 decimal
places; the RRF math is deterministic addition of rationals so
this should be byte-identical, but a 12-dp tolerance guards
against IEEE-754 reordering if a future edit changes the summation
order.

Snapshot path: `eval/results/hybrid_equivalence_snapshot.json`.
Kept under eval/results/ alongside anchor summaries so all
verification artefacts live in one place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path when this script is run from
# `scripts/`. The eval harness uses the same shim.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.retrieval.hybrid import hybrid_search  # noqa: E402


SNAPSHOT_PATH = _PROJECT_ROOT / "eval" / "results" / "hybrid_equivalence_snapshot.json"

# Fixed 5-question sample — one per category (excluding out_of_scope,
# which has no meaningful retrieval to fuse). Question IDs are stable
# in golden_set.json v1.
SAMPLE_QUESTIONS: list[tuple[str, str]] = [
    ("Q001", "How do I apply for a Custom Hiring Centre (CHC) subsidy under SMAM through the DBT agri-mechanization portal?"),
    ("Q013", "What is the farmer's premium share for Kharif food-grain and oilseed crops under PMFBY?"),
    ("Q023", "What does 'notified area' mean under PMFBY?"),
    ("Q031", "Under PMFBY, if a farmer suffers a hailstorm loss to a standing crop, what is the claim procedure and timeline?"),
    ("Q043", "A farmer voluntarily surrendered PM-KISAN benefits. Can they still receive interest subvention under KCC?"),
]


def _run_hybrid_for_sample() -> dict:
    """Run hybrid_search on the fixed sample; serialise the audit tuple."""
    output: dict = {"hybrid_top_k": settings.hybrid_top_k, "results": {}}
    for qid, question in SAMPLE_QUESTIONS:
        hits = hybrid_search(query=question, top_k=settings.hybrid_top_k)
        output["results"][qid] = [
            {
                "chunk_id": h.chunk_id,
                "rank": h.rank,
                "semantic_rank": h.semantic_rank,
                "bm25_rank": h.bm25_rank,
                "similarity_score": h.similarity_score,
                "bm25_score": h.bm25_score,
                "rrf_score": h.rrf_score,
            }
            for h in hits
        ]
    return output


def capture() -> None:
    output = _run_hybrid_for_sample()
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    total_hits = sum(len(v) for v in output["results"].values())
    print(f"[capture] wrote {SNAPSHOT_PATH}")
    print(f"[capture] {len(output['results'])} questions, {total_hits} hits total")


def _fmt_row(row: dict) -> str:
    return (
        f"chunk_id={row['chunk_id']!r} "
        f"rank={row['rank']} "
        f"semantic_rank={row['semantic_rank']} "
        f"bm25_rank={row['bm25_rank']} "
        f"similarity_score={row['similarity_score']} "
        f"bm25_score={row['bm25_score']} "
        f"rrf_score={row['rrf_score']}"
    )


def _floats_equal(a, b) -> bool:
    """Compare with 12 dp tolerance; None == None; None != number."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-12


def verify() -> int:
    if not SNAPSHOT_PATH.exists():
        print(f"[verify] no snapshot at {SNAPSHOT_PATH}; run `capture` first.")
        return 2

    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        saved = json.load(f)
    live = _run_hybrid_for_sample()

    n_ok = 0
    n_fail = 0
    for qid, _ in SAMPLE_QUESTIONS:
        saved_rows = saved["results"][qid]
        live_rows = live["results"][qid]
        if len(saved_rows) != len(live_rows):
            print(f"[FAIL] {qid}: length mismatch (saved={len(saved_rows)}, live={len(live_rows)})")
            n_fail += 1
            continue
        row_fail = False
        for i, (s, l) in enumerate(zip(saved_rows, live_rows)):
            # Compare ints and strings for equality; floats with tolerance.
            same = (
                s["chunk_id"] == l["chunk_id"]
                and s["rank"] == l["rank"]
                and s["semantic_rank"] == l["semantic_rank"]
                and s["bm25_rank"] == l["bm25_rank"]
                and _floats_equal(s["similarity_score"], l["similarity_score"])
                and _floats_equal(s["bm25_score"], l["bm25_score"])
                and _floats_equal(s["rrf_score"], l["rrf_score"])
            )
            if not same:
                row_fail = True
                print(f"[FAIL] {qid} row {i}:")
                print(f"       saved:  {_fmt_row(s)}")
                print(f"       live :  {_fmt_row(l)}")
        if row_fail:
            n_fail += 1
        else:
            n_ok += 1
            print(f"[ ok ] {qid}: {len(live_rows)} rows identical")

    print()
    print(f"[verify] {n_ok}/{n_ok + n_fail} questions bit-identical")
    return 0 if n_fail == 0 else 1


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"capture", "verify"}:
        print("usage: python scripts/hybrid_equivalence_check.py {capture|verify}")
        return 2
    if sys.argv[1] == "capture":
        capture()
        return 0
    return verify()


if __name__ == "__main__":
    sys.exit(main())
