"""
Smoke test for the query_transform pipeline (Phase 5, fix #3).

Runs the multi-query expansion + RRF fusion pipeline on the same 5
diverse questions used by the hybrid equivalence check. For each
question, prints:

  * The actual rewrites the LLM produced (for eyeball inspection —
    watch for hallucinated scheme codes, invented dates, invented
    section numbers).
  * Baseline top-5 chunk_ids (pure semantic).
  * Query_transform top-5 chunk_ids (fused).
  * Diff: which chunk_ids were added, dropped, or reordered.

This is a CHEAP sanity gate before spending 78 questions of Groq
quota on a full ablation eval. If rewrites are trivial paraphrases,
or the fused top-5 is identical to baseline top-5 on every question,
the ablation row will be a null result — better to know now.

Does NOT compute RAGAS metrics; only calls the rewriter + retriever
+ RRF fuser. No judge calls, no generator calls. Cost per run:
5 × Groq rewriter calls, no generation, no judge.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.eval.groq_eval_client import EvalGroqClient  # noqa: E402
from src.retrieval.retriever import retrieve  # noqa: E402
from src.retrieval.rrf import rrf_fuse  # noqa: E402
from src.utils.key_rotator import KeyRotator  # noqa: E402


SAMPLE_QUESTIONS: list[tuple[str, str, str]] = [
    ("Q001", "simple_procedural",
     "How do I apply for a Custom Hiring Centre (CHC) subsidy under SMAM through the DBT agri-mechanization portal?"),
    ("Q013", "simple_factual",
     "What is the farmer's premium share for Kharif food-grain and oilseed crops under PMFBY?"),
    ("Q023", "definition",
     "What does 'notified area' mean under PMFBY?"),
    ("Q031", "multi_hop_scheme",
     "Under PMFBY, if a farmer suffers a hailstorm loss to a standing crop, what is the claim procedure and timeline?"),
    ("Q043", "cross_scheme",
     "A farmer voluntarily surrendered PM-KISAN benefits. Can they still receive interest subvention under KCC?"),
]


def _short(cid: str, width: int = 60) -> str:
    return cid if len(cid) <= width else cid[: width - 3] + "..."


def _run_one(eval_client: EvalGroqClient, qid: str, category: str, question: str) -> dict:
    print("=" * 100)
    print(f"{qid}  [{category}]")
    print(f"  Q: {question}")
    print()

    # --- 1. Rewrites ---
    t0 = time.perf_counter()
    rewrites = eval_client.rewrite_query(question)
    rewrite_ms = int((time.perf_counter() - t0) * 1000)
    print(f"  Rewrites ({len(rewrites)} of {settings.query_transform_n_rewrites} requested, {rewrite_ms}ms):")
    for i, r in enumerate(rewrites, 1):
        print(f"    {i}. {r}")
    if not rewrites:
        print("    (none — will fall back to original-only retrieval for this question)")
    print()

    # --- 2. Baseline top-5 (pure semantic on the original) ---
    baseline_hits = retrieve(question, top_k=settings.retrieval_top_k, config=settings)
    baseline_ids = [h.chunk_id for h in baseline_hits]
    print(f"  Baseline top-{len(baseline_ids)} (pure semantic on original):")
    for i, cid in enumerate(baseline_ids, 1):
        print(f"    {i}. {_short(cid)}")
    print()

    # --- 3. Query_transform top-5 (fused) ---
    queries: list[str] = []
    if settings.query_transform_include_original:
        queries.append(question)
    queries.extend(rewrites)
    if not queries:
        queries = [question]

    ranked_lists = [
        retrieve(q, top_k=settings.query_transform_semantic_top_n, config=settings)
        for q in queries
    ]
    fused = rrf_fuse(
        ranked_lists=ranked_lists,
        top_k=settings.query_transform_top_k,
        rrf_k=settings.query_transform_rrf_k,
    )
    fused_ids = [h.chunk_id for h in fused]
    print(f"  Query_transform top-{len(fused_ids)} (fused over {len(queries)} queries, rrf_k={settings.query_transform_rrf_k}):")
    for i, (h, cid) in enumerate(zip(fused, fused_ids), 1):
        print(f"    {i}. {_short(cid)}  (rrf_score={h.rrf_score:.5f})")
    print()

    # --- 4. Diff ---
    baseline_set = set(baseline_ids)
    fused_set = set(fused_ids)
    added = fused_set - baseline_set
    dropped = baseline_set - fused_set
    kept = baseline_set & fused_set
    reordered = [
        cid for cid in kept
        if baseline_ids.index(cid) != fused_ids.index(cid)
    ]

    print(f"  Diff (fused vs baseline):")
    print(f"    kept    : {len(kept)} chunks")
    print(f"    added   : {len(added)} chunks  {[_short(c, 40) for c in sorted(added)] if added else ''}")
    print(f"    dropped : {len(dropped)} chunks  {[_short(c, 40) for c in sorted(dropped)] if dropped else ''}")
    print(f"    reordered: {len(reordered)} kept chunks changed position")
    print()

    return {
        "qid": qid,
        "n_rewrites": len(rewrites),
        "rewrite_ms": rewrite_ms,
        "n_kept": len(kept),
        "n_added": len(added),
        "n_dropped": len(dropped),
        "n_reordered": len(reordered),
    }


def main() -> int:
    print("Query-transform smoke test — 5 diverse questions")
    print(f"Config: n_rewrites={settings.query_transform_n_rewrites}, "
          f"include_original={settings.query_transform_include_original}, "
          f"top_n={settings.query_transform_semantic_top_n}, "
          f"top_k={settings.query_transform_top_k}, "
          f"rrf_k={settings.query_transform_rrf_k}, "
          f"model={settings.llm_model}")
    print()

    rotator = KeyRotator(
        keys=settings.groq_api_keys,
        cooldown_s=settings.key_rotation_cooldown_s,
        simulate_exhaustion_after=settings.simulate_quota_exhaustion_after_n_calls,
    )
    eval_client = EvalGroqClient(rotator=rotator)

    stats = []
    for qid, category, question in SAMPLE_QUESTIONS:
        stats.append(_run_one(eval_client, qid, category, question))

    # --- Summary ---
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'qid':<6} {'nrw':>4} {'ms':>6} {'kept':>5} {'add':>4} {'drop':>4} {'reord':>6}")
    for s in stats:
        print(f"{s['qid']:<6} {s['n_rewrites']:>4} {s['rewrite_ms']:>6} "
              f"{s['n_kept']:>5} {s['n_added']:>4} {s['n_dropped']:>4} {s['n_reordered']:>6}")
    total_rewrites = sum(s["n_rewrites"] for s in stats)
    total_added = sum(s["n_added"] for s in stats)
    total_dropped = sum(s["n_dropped"] for s in stats)
    print()
    print(f"Total rewrites received: {total_rewrites} / {5 * settings.query_transform_n_rewrites} requested")
    print(f"Total chunks changed (added+dropped): {total_added + total_dropped} across {len(stats)} questions")
    if total_added + total_dropped == 0:
        print()
        print("!! WARNING: fused top-K identical to baseline top-K on every question.")
        print("   Either rewrites are trivial paraphrases or fusion isn't shifting the top-5.")
        print("   The ablation row will likely be a null result. Investigate before full eval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
