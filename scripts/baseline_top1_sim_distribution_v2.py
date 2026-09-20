"""
v2 of the top-1 similarity distribution diagnostic — extended to
include the malformed_query group (Q079-Q088).

Groups A, B, C are unchanged (computed from the saved baseline
anchor's per-question output on the original 78-question set):
  A. hit@10 == 0 AND refused (n=15)   — true failure cases
  B. hit@10 == 0 AND attempted (n=7)  — corpus-redundant successes
  C. hit@10 > 0 (n=56)                — baseline retrieval worked

New group D:
  D. malformed_query (n=10)           — user-style query shapes
     top-1 similarity computed FRESH via `retrieve()`
     (these questions were not in the baseline anchor run).

For each malformed row we ALSO check hit@5 on the clean expected_sources,
so we can label each row as "malformed + baseline retrieval succeeds"
vs "malformed + baseline retrieval fails" — CRAG's target rows.

Read-only for the pipeline; retrieves against the persisted Chroma
store, does not modify anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import OUT_OF_SCOPE_REFUSAL_PHRASES, settings  # noqa: E402
from src.retrieval.retriever import retrieve  # noqa: E402


BASELINE = _PROJECT_ROOT / "eval" / "results" / "20260913-104053_baseline_n78" / "results.jsonl"
GOLDEN = _PROJECT_ROOT / "eval" / "golden_set.json"


def is_refusal(answer: str) -> bool:
    if not answer:
        return False
    lower = answer.lower()
    return any(p in lower for p in OUT_OF_SCOPE_REFUSAL_PHRASES)


def _pct(vs: list[float], q: float) -> float:
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


def _hit_at_k(retrieved_ids: list[str], expected_sources: list[dict], k: int) -> int:
    """Same shape the eval harness uses — 1 if any expected source id in top-k."""
    top = set(retrieved_ids[:k])
    for src in expected_sources:
        if src.get("type") == "workflow":
            wfid = src.get("workflow_id")
            for cid in top:
                if wfid and wfid in cid:
                    return 1
        elif src.get("type") == "pdf":
            fname = src.get("source_filename")
            # For PDF gold sources we match against retrieved chunk filenames
            # via the chunk_id prefix. Loose match — same rule the eval uses.
            for cid in top:
                if fname and fname.replace(".pdf", "") in cid:
                    return 1
    return 0


def main() -> int:
    rows = [json.loads(line) for line in BASELINE.read_text(encoding="utf-8").splitlines() if line.strip()]
    gs = json.loads(GOLDEN.read_text(encoding="utf-8"))

    # Groups A / B / C from saved baseline
    group_A = []   # top1 sims: hit@10==0 AND refused
    group_B = []   # top1 sims: hit@10==0 AND attempted (faith ~ 1)
    group_C = []   # top1 sims: hit@10>0
    detail_A: list[tuple[str, str, float]] = []
    detail_B: list[tuple[str, str, float, float | None]] = []

    for r in rows:
        sims = r.get("retrieved_similarity_scores") or []
        top1 = sims[0] if sims else None
        if top1 is None:
            continue
        hit10 = float(r.get("hit_rate_at_10") or 0.0)
        ans = r.get("generated_answer") or ""
        refused = is_refusal(ans)
        if hit10 == 0.0 and refused:
            group_A.append(top1)
            detail_A.append((r["question_id"], r["category"], top1))
        elif hit10 == 0.0 and not refused:
            group_B.append(top1)
            detail_B.append((r["question_id"], r["category"], top1, r.get("faithfulness")))
        elif hit10 > 0.0:
            group_C.append(top1)

    # Group D: fresh retrieval on the 10 new malformed queries
    malformed = [q for q in gs["questions"] if q.get("category") == "malformed_query"]
    print("=" * 100)
    print(f"RETRIEVING FRESH on {len(malformed)} malformed queries — top-K={settings.retrieval_top_k}")
    print("=" * 100)

    group_D_sims: list[float] = []
    detail_D: list[tuple[str, str, float, int, str]] = []  # (qid, src_qid, top1, hit_at_5, question_text)

    for q in malformed:
        hits = retrieve(q["question"], top_k=settings.retrieval_top_k, config=settings)
        top1 = hits[0].similarity_score if hits else float("nan")
        retrieved_ids = [h.chunk_id for h in hits]
        h5 = _hit_at_k(retrieved_ids, q["expected_sources"], k=settings.retrieval_top_k)
        group_D_sims.append(top1)
        detail_D.append((q["question_id"], q.get("original_source_qid", ""), top1, h5, q["question"]))

    # Also retrieve on the clean SOURCE questions for a paired comparison
    print("\nRETRIEVING FRESH on the 10 CLEAN source questions for paired comparison")
    paired: list[tuple[str, str, str, str, float, int, float, int]] = []
    #        (mal_qid, src_qid, mal_q, src_q, mal_top1, mal_h5, src_top1, src_h5)
    by_qid_gs = {q["question_id"]: q for q in gs["questions"]}
    for q in malformed:
        src_qid = q["original_source_qid"]
        src = by_qid_gs.get(src_qid)
        if src is None:
            continue
        src_hits = retrieve(src["question"], top_k=settings.retrieval_top_k, config=settings)
        src_top1 = src_hits[0].similarity_score if src_hits else float("nan")
        src_ids = [h.chunk_id for h in src_hits]
        src_h5 = _hit_at_k(src_ids, q["expected_sources"], k=settings.retrieval_top_k)
        mal_top1 = next(top1 for (mq, _, top1, _, _) in detail_D if mq == q["question_id"])
        mal_h5 = next(h5 for (mq, _, _, h5, _) in detail_D if mq == q["question_id"])
        paired.append((q["question_id"], src_qid, q["question"], src["question"],
                       mal_top1, mal_h5, src_top1, src_h5))

    # ---------------- Report ----------------
    print("\n" + "=" * 100)
    print("BASELINE top-1 similarity distribution — 4 groups")
    print("=" * 100)

    _report("A: hit@10==0 AND refused", group_A)
    _report("B: hit@10==0 AND attempted", group_B)
    _report("C: hit@10>0", group_C)
    _report("D: malformed_query (fresh retrieval)", group_D_sims)

    print("\n" + "-" * 100)
    print("Group D detail (malformed_query) — sorted by top-1 sim:")
    print(f"  {'mal_qid':<8}{'src_qid':<8}{'top1':>7}  {'hit@5':>6}  question")
    for qid, src_qid, top1, h5, qtext in sorted(detail_D, key=lambda x: x[2]):
        print(f"  {qid:<8}{src_qid:<8}{top1:>7.4f}  {h5:>6}  {qtext}")

    print("\nPaired top-1 sim comparison (malformed vs its clean source):")
    print(f"  {'mal_qid':<8}{'src_qid':<8}{'mal_top1':>10}{'src_top1':>10}  {'delta':>8}   {'mal_h5':>6}{'src_h5':>6}")
    total_delta = 0.0
    for mal_qid, src_qid, _, _, mal_top1, mal_h5, src_top1, src_h5 in paired:
        delta = mal_top1 - src_top1
        total_delta += delta
        print(f"  {mal_qid:<8}{src_qid:<8}{mal_top1:>10.4f}{src_top1:>10.4f}  {delta:+.4f}   "
              f"{mal_h5:>6}{src_h5:>6}")
    if paired:
        print(f"  mean delta (malformed - clean): {total_delta/len(paired):+.4f}")

    # ---------------- Threshold sweep ----------------
    print("\n" + "=" * 100)
    print("Threshold sweep — flag counts if TRIGGER = top1 < T")
    print("=" * 100)
    print(f"{'T':>6}   {'A_flag':>7}  {'B_flag':>7}  {'C_flag':>7}  {'D_flag':>7}   "
          f"{'A%':>6}  {'B%':>6}  {'C%':>6}  {'D%':>6}")
    for T in [0.35, 0.40, 0.45, 0.50, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.72, 0.75]:
        a = sum(1 for s in group_A if s < T)
        b = sum(1 for s in group_B if s < T)
        c = sum(1 for s in group_C if s < T)
        d = sum(1 for s in group_D_sims if s < T)
        print(f"  {T:.3f}  {a:>7}  {b:>7}  {c:>7}  {d:>7}   "
              f"{100*a/max(len(group_A),1):>5.1f}% "
              f"{100*b/max(len(group_B),1):>5.1f}% "
              f"{100*c/max(len(group_C),1):>5.1f}% "
              f"{100*d/max(len(group_D_sims),1):>5.1f}%")

    print("\nLegend:")
    print("  A_flag = true failures correctly flagged (want HIGH)")
    print("  B_flag = corpus-redundant successes false-flagged (want LOW)")
    print("  C_flag = hit@10>0 questions false-flagged (want LOW)")
    print("  D_flag = malformed queries flagged for CRAG rewrite (want HIGH when malformed retrieval also fails)")

    # D_flag interpretation: we want D_flag to correspond to malformed
    # rows where baseline retrieval FAILED (hit@5=0). Let's break D_flag
    # down by hit@5 outcome at each candidate threshold.
    print("\n" + "-" * 100)
    print("D_flag broken down by malformed hit@5 outcome:")
    print(f"{'T':>6}   {'D_flag_h5=0':>12}{'D_flag_h5=1':>12}   {'D_miss_h5=0':>12}{'D_miss_h5=1':>12}")
    for T in [0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.72]:
        d_flag_h5_0 = sum(1 for (_, _, top1, h5, _) in detail_D if top1 < T and h5 == 0)
        d_flag_h5_1 = sum(1 for (_, _, top1, h5, _) in detail_D if top1 < T and h5 == 1)
        d_miss_h5_0 = sum(1 for (_, _, top1, h5, _) in detail_D if top1 >= T and h5 == 0)
        d_miss_h5_1 = sum(1 for (_, _, top1, h5, _) in detail_D if top1 >= T and h5 == 1)
        print(f"  {T:.3f}  {d_flag_h5_0:>12}{d_flag_h5_1:>12}   {d_miss_h5_0:>12}{d_miss_h5_1:>12}")
    print("\nInterpretation:")
    print("  D_flag_h5=0 (want HIGH)   = correctly-flagged malformed row where retrieval failed")
    print("  D_flag_h5=1 (want LOW)    = false-flagged malformed row where retrieval already succeeded")
    print("  D_miss_h5=0 (want LOW)    = missed malformed row where retrieval failed and needs CRAG")
    print("  D_miss_h5=1 (irrelevant)  = malformed row where retrieval succeeded and CRAG not needed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
