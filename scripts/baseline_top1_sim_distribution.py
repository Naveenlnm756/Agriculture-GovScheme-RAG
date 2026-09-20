"""
Diagnostic: distribution of TOP-1 similarity scores in the baseline
anchor run, grouped by whether the question was answered / refused
and by hit@10 outcome. Purpose: pick an interview-defensible top-1
similarity threshold for CRAG's confidence signal.

Groups reported:
  A. hit@10 == 0 AND refused (n=15)      — "true failure cases"
                                            (includes 8 out_of_scope)
  B. hit@10 == 0 AND attempted (n=7)      — "corpus-redundant successes"
                                            (mean faithfulness 0.976)
  C. hit@10 > 0                          (n=56, baseline answered
                                            using a gold-source chunk)

Read-only. No pipeline changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import OUT_OF_SCOPE_REFUSAL_PHRASES  # noqa: E402


BASELINE = _PROJECT_ROOT / "eval" / "results" / "20260913-104053_baseline_n78" / "results.jsonl"


def is_refusal(answer: str) -> bool:
    if not answer:
        return False
    lower = answer.lower()
    return any(p in lower for p in OUT_OF_SCOPE_REFUSAL_PHRASES)


def _pct(vs: list[float], q: float) -> float:
    """Percentile without numpy."""
    if not vs:
        return float("nan")
    xs = sorted(vs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def _report(name: str, sims: list[float]) -> None:
    if not sims:
        print(f"\n{name}: (empty)")
        return
    print(f"\n{name}  (n={len(sims)})")
    print(f"  min={min(sims):.4f}  p10={_pct(sims,0.10):.4f}  p25={_pct(sims,0.25):.4f}  "
          f"med={median(sims):.4f}  p75={_pct(sims,0.75):.4f}  p90={_pct(sims,0.90):.4f}  "
          f"max={max(sims):.4f}  mean={mean(sims):.4f}")
    print(f"  sorted: {[round(s, 4) for s in sorted(sims)]}")


def main() -> int:
    rows = [json.loads(line) for line in BASELINE.read_text(encoding="utf-8").splitlines() if line.strip()]

    group_A_refused_hit0 = []   # top1 sims: hit@10==0 and refused
    group_B_attempted_hit0 = [] # top1 sims: hit@10==0 and attempted
    group_C_hit_ge_1 = []       # top1 sims: hit@10>0

    detail_A = []
    detail_B = []

    for r in rows:
        sims = r.get("retrieved_similarity_scores") or []
        top1 = sims[0] if sims else None
        if top1 is None:
            continue
        hit10 = float(r.get("hit_rate_at_10") or 0.0)
        ans = r.get("generated_answer") or ""
        refused = is_refusal(ans)

        if hit10 == 0.0 and refused:
            group_A_refused_hit0.append(top1)
            detail_A.append((r["question_id"], r["category"], top1))
        elif hit10 == 0.0 and not refused:
            group_B_attempted_hit0.append(top1)
            detail_B.append((r["question_id"], r["category"], top1, r.get("faithfulness")))
        elif hit10 > 0.0:
            group_C_hit_ge_1.append(top1)

    print("=" * 100)
    print("BASELINE — top-1 similarity distribution by outcome group")
    print("=" * 100)

    _report("A: hit@10==0 AND refused (n=15 expected)", group_A_refused_hit0)
    _report("B: hit@10==0 AND attempted (n=7 expected)", group_B_attempted_hit0)
    _report("C: hit@10>0 (n=56 expected)", group_C_hit_ge_1)

    print("\n" + "-" * 100)
    print("Group A detail (refused, hit@10==0):")
    for qid, cat, top1 in sorted(detail_A, key=lambda x: x[2]):
        print(f"  {qid}  [{cat:<24}]  top1={top1:.4f}")
    print("\nGroup B detail (attempted, hit@10==0):")
    for qid, cat, top1, faith in sorted(detail_B, key=lambda x: x[2]):
        print(f"  {qid}  [{cat:<24}]  top1={top1:.4f}  faithfulness={faith}")

    # ---- Threshold candidates ----
    # For each candidate T, count how each group would be flagged (top1 < T):
    print("\n" + "=" * 100)
    print("Candidate thresholds — flag counts if TRIGGER = top1 < T")
    print("=" * 100)
    print(f"{'T':>6}   {'A_flag':>7}  {'B_flag':>7}  {'C_flag':>7}   {'A%':>6}  {'B%':>6}  {'C%':>6}")
    # Group A: we WANT to flag these (should trigger rewrite/refuse)
    # Group B: we do NOT want to flag these (corpus-redundant successes)
    # Group C: we do NOT want to flag these (correct-chunk-in-top-10)
    for T in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        a = sum(1 for s in group_A_refused_hit0 if s < T)
        b = sum(1 for s in group_B_attempted_hit0 if s < T)
        c = sum(1 for s in group_C_hit_ge_1 if s < T)
        print(f"  {T:.2f}  {a:>7}  {b:>7}  {c:>7}   "
              f"{100*a/max(len(group_A_refused_hit0),1):>5.1f}%  "
              f"{100*b/max(len(group_B_attempted_hit0),1):>5.1f}%  "
              f"{100*c/max(len(group_C_hit_ge_1),1):>5.1f}%")

    print("\nLegend:")
    print("  A_flag = true failures correctly flagged (want HIGH)")
    print("  B_flag = corpus-redundant successes false-flagged (want LOW)")
    print("  C_flag = hit@10>0 questions false-flagged (want LOW)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
