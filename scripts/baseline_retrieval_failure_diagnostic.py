"""
Diagnostic: on the baseline anchor row (n=78), for every question
where hit_rate_at_10 == 0 (i.e., no gold source was in the top-10
retrieved chunks), report whether the generator refused or attempted
an answer.

Purpose: answer the question "does baseline's grounding prompt
already handle low-confidence retrieval, or is there a real gap for
CRAG to fill?" A high refusal rate on hit@10==0 rows means the
existing grounding rule ("if passages don't contain enough info,
say so") is doing CRAG's job for free — CRAG would then add cost
without adding value.

Read-only diagnostic. Does not touch pipeline or generator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import OUT_OF_SCOPE_REFUSAL_PHRASES  # noqa: E402


BASELINE_RESULTS = _PROJECT_ROOT / "eval" / "results" / "20260913-104053_baseline_n78" / "results.jsonl"


def contains_refusal(answer: str) -> tuple[bool, list[str]]:
    if not answer:
        return False, []
    lower = answer.lower()
    hits = [p for p in OUT_OF_SCOPE_REFUSAL_PHRASES if p in lower]
    return (len(hits) > 0, hits)


def main() -> int:
    if not BASELINE_RESULTS.exists():
        print(f"Missing: {BASELINE_RESULTS}")
        return 2

    rows = [json.loads(line) for line in BASELINE_RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]

    failures = [r for r in rows if float(r.get("hit_rate_at_10") or 0.0) == 0.0]

    print("=" * 100)
    print(f"BASELINE (n=78) — questions where hit_rate_at_10 == 0")
    print(f"  Total: {len(failures)} of {len(rows)}")
    print("=" * 100)
    print()

    refused = []
    attempted = []

    for r in failures:
        qid = r["question_id"]
        cat = r["category"]
        q = r["question"]
        ans = r.get("generated_answer") or ""
        faith = r.get("faithfulness")
        did_refuse, hits = contains_refusal(ans)
        if did_refuse:
            refused.append((qid, cat, q, ans, hits, faith))
        else:
            attempted.append((qid, cat, q, ans, faith))

    # --- Refused rows ---
    print("-" * 100)
    print(f"REFUSED — {len(refused)} of {len(failures)} retrieval-failed questions")
    print("-" * 100)
    for qid, cat, q, ans, hits, faith in refused:
        print(f"\n[{qid} · {cat}]  faithfulness={faith}")
        print(f"  Q: {q}")
        # Show the refusal sentence(s), truncated
        print(f"  A ({len(ans)} chars, matched phrases {hits}):")
        # Print first ~2 lines of answer
        preview = ans.strip().split("\n\n")[0][:400]
        print(f"    {preview}")

    print()
    print("-" * 100)
    print(f"ATTEMPTED — {len(attempted)} of {len(failures)} retrieval-failed questions")
    print("-" * 100)
    for qid, cat, q, ans, faith in attempted:
        print(f"\n[{qid} · {cat}]  faithfulness={faith}")
        print(f"  Q: {q}")
        preview = ans.strip().split("\n\n")[0][:400]
        print(f"  A ({len(ans)} chars):")
        print(f"    {preview}")

    # --- Summary ---
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  refused           : {len(refused):>3} / {len(failures)}  "
          f"({100 * len(refused) / len(failures):.1f}%)")
    print(f"  attempted answer  : {len(attempted):>3} / {len(failures)}  "
          f"({100 * len(attempted) / len(failures):.1f}%)")

    attempted_faith = [f for _, _, _, _, f in attempted if f is not None]
    if attempted_faith:
        avg = sum(attempted_faith) / len(attempted_faith)
        n_zero = sum(1 for f in attempted_faith if f == 0.0)
        n_one = sum(1 for f in attempted_faith if f == 1.0)
        print(f"\n  attempted-answer faithfulness (n={len(attempted_faith)}):")
        print(f"    mean = {avg:.4f}")
        print(f"    == 0.0 : {n_zero} ({100 * n_zero / len(attempted_faith):.1f}%) — fully hallucinated")
        print(f"    == 1.0 : {n_one} ({100 * n_one / len(attempted_faith):.1f}%) — fully grounded (odd on hit@10=0)")
        print(f"    per row: {[round(f, 3) for f in attempted_faith]}")

    refused_faith = [f for _, _, _, _, _, f in refused if f is not None]
    if refused_faith:
        avg = sum(refused_faith) / len(refused_faith)
        print(f"\n  refused faithfulness (n={len(refused_faith)}, informational — RAGAS scores refusals ~0):")
        print(f"    mean = {avg:.4f}")

    # Bucket breakdown
    from collections import Counter
    ref_buckets = Counter(cat for _, cat, _, _, _, _ in refused)
    att_buckets = Counter(cat for _, cat, _, _, _ in attempted)
    all_buckets = set(list(ref_buckets.keys()) + list(att_buckets.keys()))
    print(f"\n  per-bucket breakdown (refused / attempted):")
    for b in sorted(all_buckets):
        print(f"    {b:<30}  {ref_buckets.get(b, 0):>2}  /  {att_buckets.get(b, 0):>2}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
