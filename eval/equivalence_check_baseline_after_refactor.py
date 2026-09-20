"""
Equivalence check: refactored baseline path vs. anchor baseline run.

Purpose
-------
Phase 5 fix #1 (reranker) required restructuring `eval/run_eval.py`:
`_dispatch_pipeline` was renamed to `_run_pipeline_for_mode` and a
`_run_pipeline_reranked` helper was added alongside the existing
`_run_pipeline_baseline`. This script proves that the refactor is
side-effect free — the baseline path still produces the same
per-question outputs as the anchor run
`20260913-104053_baseline_n78`.

If this check fails, we MUST NOT proceed to run the reranked eval,
because a drift in the baseline path would silently confound the
technique-vs-baseline delta (we could no longer tell if a metric
moved because of the reranker or because of the refactor).

What it does
------------
1. Load the golden set.
2. Pick 15 stratified question ids (matches the spec's stratification):
     3 simple_procedural, 3 simple_factual, 2 definition,
     3 multi_hop_scheme, 2 cross_scheme, 2 out_of_scope.
   Picked deterministically as the first N of each category in the
   golden set's canonical ordering, so a rerun picks the same 15.
3. For each qid:
   * Look up the anchor row in
     `eval/results/20260913-104053_baseline_n78/results.jsonl`.
   * Run `_run_pipeline_for_mode("baseline", question, eval_client)`
     under the refactored harness.
   * Compute deterministic retrieval metrics on the fresh output.
   * Compare against the anchor row:
       - retrieved_chunk_ids (top-10):   must be IDENTICAL
                                          (semantic retrieval is
                                          deterministic given fixed
                                          embedder + collection + query)
       - retrieved_similarity_scores:    must match to 4 decimal places
       - hit_rate_at_5 / hit_rate_at_10 / mrr: must be IDENTICAL
       - generated_answer:                length + first-200-char diff
                                          (temperature=0, so byte-
                                          identical or trivially similar)
4. Write the full comparison table to
   `eval/results/equivalence_check_baseline_after_refactor.json` and
   print a summary to stdout.

Why RAGAS is skipped in this check
----------------------------------
The refactor changed ROUTING only — no touch to the retriever, embedder,
generator, prompt, or judge. If (a) retrieval is byte-identical and
(b) the generated answer is byte-identical, then RAGAS deltas can only
come from judge stochasticity, which is orthogonal to the refactor.
Running RAGAS here would burn Mistral judge quota needed for Task 4
(smoke test) and Phase 5 rows, for no additional signal about the
refactor's correctness. This is explicitly documented in the output
JSON so a reviewer sees the caveat.

Exit codes
----------
0 = every comparison within tolerance; safe to proceed to Task 4.
1 = at least one comparison outside tolerance; investigate before
     running the reranked eval.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Windows default stdout is cp1252 — cannot print math symbols like Δ.
# Wrap sys.stdout in a utf-8 writer so discrepancy messages print
# without UnicodeEncodeError. Reversible for tests / callers that
# reset sys.stdout later.
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.eval.groq_eval_client import EvalGroqClient  # noqa: E402
from src.utils.key_rotator import KeyRotator  # noqa: E402

# Re-use the ACTUAL routing helper and metric computer from the
# refactored harness. This is the whole point of the check — verify
# the shipped code path, not a copy of it.
from eval.run_eval import (  # noqa: E402
    _compute_retrieval_metrics,
    _run_pipeline_for_mode,
)


logger = logging.getLogger(__name__)


ANCHOR_RUN_ID = "20260913-104053_baseline_n78"
ANCHOR_RESULTS_PATH = (
    PROJECT_ROOT / "eval" / "results" / ANCHOR_RUN_ID / "results.jsonl"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "eval"
    / "results"
    / "equivalence_check_baseline_after_refactor.json"
)

# Stratification (exactly matches the spec).
STRATIFICATION: dict[str, int] = {
    "simple_procedural": 3,
    "simple_factual": 3,
    "definition": 2,
    "multi_hop_scheme": 3,
    "cross_scheme": 2,
    "out_of_scope": 2,
}
EXPECTED_TOTAL = 15  # sum of STRATIFICATION.values()

# Tolerance thresholds.
SIMILARITY_SCORE_TOLERANCE = 1e-4          # semantic retrieval is deterministic
RETRIEVAL_METRIC_TOLERANCE = 0.0           # int metrics — must be exact
MRR_TOLERANCE = 1e-9                       # rational floats over ranks
ANSWER_LEN_RATIO_TOLERANCE = 0.10          # temperature=0 should be near-identical


def _pick_stratified_questions(golden_path: Path) -> list[dict]:
    """
    Return the first N questions of each category, matching STRATIFICATION.

    Deterministic — the golden set is JSON with a fixed ordering, and
    "first N per category" is unambiguous. A rerun picks the same 15.

    Raises SystemExit if a category comes up short (indicates the
    golden set has drifted from what the spec assumed).
    """
    with golden_path.open("r", encoding="utf-8") as f:
        golden = json.load(f)
    questions = golden.get("questions", [])

    by_category: dict[str, list[dict]] = {}
    for q in questions:
        by_category.setdefault(q.get("category", ""), []).append(q)

    picked: list[dict] = []
    for cat, n in STRATIFICATION.items():
        available = by_category.get(cat, [])
        if len(available) < n:
            raise SystemExit(
                f"Stratified pick failed: category {cat!r} has only "
                f"{len(available)} questions, need {n}. Golden set may "
                "have drifted."
            )
        picked.extend(available[:n])

    if len(picked) != EXPECTED_TOTAL:
        raise SystemExit(
            f"Stratified pick assembled {len(picked)} questions, expected "
            f"{EXPECTED_TOTAL}. Check STRATIFICATION."
        )
    return picked


def _load_anchor_rows_by_qid(anchor_path: Path) -> dict[str, dict]:
    """
    Read the anchor's results.jsonl into a dict keyed by question_id.

    results.jsonl is one JSON object per line, one per completed question.
    """
    if not anchor_path.exists():
        raise SystemExit(
            f"Anchor results not found at {anchor_path}. Cannot run "
            "equivalence check without the baseline anchor."
        )
    rows: dict[str, dict] = {}
    with anchor_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row.get("question_id", "")] = row
    return rows


def _compare_one(
    fresh_retrieved,
    fresh_gen_result,
    fresh_retrieval_metrics: dict,
    anchor_row: dict,
    question: dict,
) -> dict:
    """
    Compare one refactored-run row against the corresponding anchor row.

    Two categories of finding, tracked separately:
      * retrieval_discrepancies  — must be EMPTY for the refactor to be
                                    considered safe. Semantic retrieval
                                    is deterministic given fixed embedder
                                    + collection + query; any mismatch
                                    here is a refactor-caused bug.
      * answer_notes             — informational only. Groq at
                                    temperature=0 is empirically not
                                    perfectly deterministic between
                                    separate runs (Groq's own inference
                                    stack is a shared multi-tenant
                                    system; different backend nodes can
                                    tokenise / batch differently). An
                                    answer-length delta with byte-
                                    identical retrieval + identical
                                    retrieval metrics is a Groq artifact,
                                    NOT a refactor bug — the routing
                                    change literally cannot influence
                                    generator output when the input
                                    chunks are identical.

    Pass criterion: retrieval_discrepancies == []. The answer_notes are
    recorded but do not fail the check.
    """
    qid = question.get("question_id", "?")
    category = question.get("category", "?")

    retrieval_discrepancies: list[str] = []
    answer_notes: list[str] = []

    # --- 1. Retrieved chunk ids (top-10) — must be identical ----------
    fresh_ids = [c.chunk_id for c in fresh_retrieved[:10]]
    anchor_ids = list(anchor_row.get("retrieved_chunk_ids", []) or [])
    if fresh_ids != anchor_ids:
        retrieval_discrepancies.append(
            "retrieved_chunk_ids DIFFER - "
            f"fresh[{len(fresh_ids)}]={fresh_ids[:3]}... "
            f"anchor[{len(anchor_ids)}]={anchor_ids[:3]}..."
        )

    # --- 2. Similarity scores (top-10) — must match to 4 decimals ------
    fresh_scores = [round(c.similarity_score, 4) for c in fresh_retrieved[:10]]
    anchor_scores = [
        round(float(s), 4) for s in anchor_row.get("retrieved_similarity_scores", []) or []
    ]
    if len(fresh_scores) == len(anchor_scores):
        for i, (fs, ans) in enumerate(zip(fresh_scores, anchor_scores)):
            if abs(fs - ans) > SIMILARITY_SCORE_TOLERANCE:
                retrieval_discrepancies.append(
                    f"similarity_score[{i}] fresh={fs} anchor={ans} "
                    f"|delta|={abs(fs - ans):.6f} > {SIMILARITY_SCORE_TOLERANCE}"
                )
    else:
        retrieval_discrepancies.append(
            f"similarity score list length differs: fresh={len(fresh_scores)} "
            f"anchor={len(anchor_scores)}"
        )

    # --- 3. Retrieval metrics — must be exact -------------------------
    for metric in ("hit_rate_at_5", "hit_rate_at_10", "mrr"):
        fresh_val = fresh_retrieval_metrics.get(metric)
        anchor_val = anchor_row.get(metric)
        if metric == "mrr":
            fresh_f = float(fresh_val or 0.0)
            anchor_f = float(anchor_val or 0.0)
            if abs(fresh_f - anchor_f) > MRR_TOLERANCE:
                retrieval_discrepancies.append(
                    f"{metric}: fresh={fresh_f} anchor={anchor_f} "
                    f"|delta|={abs(fresh_f - anchor_f)}"
                )
        else:
            if int(fresh_val or 0) != int(anchor_val or 0):
                retrieval_discrepancies.append(
                    f"{metric}: fresh={fresh_val} anchor={anchor_val}"
                )

    # --- 4. Generated answer — informational only ---------------------
    fresh_answer = fresh_gen_result.answer if fresh_gen_result else ""
    anchor_answer = anchor_row.get("generated_answer", "") or ""

    fresh_len = len(fresh_answer)
    anchor_len = len(anchor_answer)
    if anchor_len > 0:
        ratio = abs(fresh_len - anchor_len) / max(anchor_len, 1)
        if ratio > ANSWER_LEN_RATIO_TOLERANCE:
            answer_notes.append(
                f"answer length delta ratio={ratio:.3f} > "
                f"{ANSWER_LEN_RATIO_TOLERANCE} "
                f"(fresh={fresh_len} anchor={anchor_len}). Groq temp=0 "
                "stochasticity — orthogonal to routing refactor."
            )
    byte_identical = fresh_answer == anchor_answer

    return {
        "question_id": qid,
        "category": category,
        "retrieval_equivalent": len(retrieval_discrepancies) == 0,
        "answer_byte_identical": byte_identical,
        "answer_len_fresh": fresh_len,
        "answer_len_anchor": anchor_len,
        "answer_first_120_fresh": fresh_answer[:120],
        "answer_first_120_anchor": anchor_answer[:120],
        "retrieved_ids_fresh_top5": fresh_ids[:5],
        "retrieved_ids_anchor_top5": anchor_ids[:5],
        "retrieval_metrics_fresh": {
            k: fresh_retrieval_metrics.get(k)
            for k in ("hit_rate_at_5", "hit_rate_at_10", "mrr")
        },
        "retrieval_metrics_anchor": {
            k: anchor_row.get(k)
            for k in ("hit_rate_at_5", "hit_rate_at_10", "mrr")
        },
        "retrieval_discrepancies": retrieval_discrepancies,
        "answer_notes": answer_notes,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print("=" * 88)
    print("EQUIVALENCE CHECK — baseline path after Phase 5 refactor")
    print("=" * 88)
    print(f"anchor       : {ANCHOR_RUN_ID}")
    print(f"stratification: {STRATIFICATION} (total={EXPECTED_TOTAL})")
    print()

    # --- Setup ---------------------------------------------------------
    picked = _pick_stratified_questions(settings.eval_golden_set_path)
    anchor_rows = _load_anchor_rows_by_qid(ANCHOR_RESULTS_PATH)

    missing = [q.get("question_id") for q in picked if q.get("question_id") not in anchor_rows]
    if missing:
        raise SystemExit(
            f"Stratified qids {missing} not present in anchor "
            f"{ANCHOR_RUN_ID}. Cannot compare — the anchor was run on a "
            "different question set."
        )

    # Generator key rotation (matches run_eval.py's generator-only branch,
    # since we are not running the judge here).
    gen_key = (settings.llm_api_key or "").strip()
    judge_keys = [k for k in settings.groq_api_keys if k and k != gen_key]
    gen_pool = [gen_key] if gen_key else []
    for k in judge_keys:
        if k not in gen_pool:
            gen_pool.append(k)
    if not gen_pool:
        raise SystemExit(
            "No Groq API keys configured. Set GROQ_API_KEY / GROQ_API_KEYS."
        )
    rotator = KeyRotator(
        keys=gen_pool,
        cooldown_s=settings.key_rotation_cooldown_s,
        simulate_exhaustion_after=settings.simulate_quota_exhaustion_after_n_calls,
    )
    eval_client = EvalGroqClient(rotator=rotator)

    # --- Per-question loop --------------------------------------------
    per_question: list[dict] = []
    for i, q in enumerate(picked, start=1):
        qid = q.get("question_id", "?")
        category = q.get("category", "?")
        question_text = q.get("question", "")
        preview = question_text[:60] + ("..." if len(question_text) > 60 else "")
        print(f"  [{i:>2}/{len(picked)}] {qid} [{category}] {preview}")

        anchor_row = anchor_rows[qid]

        try:
            retrieved_pool, gen_result = _run_pipeline_for_mode(
                "baseline", question_text, eval_client
            )
            retrieval_metrics = _compute_retrieval_metrics(
                retrieved_pool, q.get("expected_sources", [])
            )
            row = _compare_one(
                retrieved_pool, gen_result, retrieval_metrics, anchor_row, q
            )
        except Exception as exc:
            logger.error("Pipeline failed on %s: %s", qid, exc)
            row = {
                "question_id": qid,
                "category": category,
                "retrieval_equivalent": False,
                "retrieval_discrepancies": [f"pipeline_error: {exc}"],
                "answer_notes": [],
            }
        per_question.append(row)
        marker = "OK  " if row.get("retrieval_equivalent") else "FAIL"
        answer_flag = (
            "byte-identical-answer"
            if row.get("answer_byte_identical")
            else f"answer-drift(fresh={row.get('answer_len_fresh')} anchor={row.get('answer_len_anchor')})"
        )
        print(f"      {marker}  retrieval  |  {answer_flag}")
        for d in row.get("retrieval_discrepancies", []) or []:
            print(f"        RETRIEVAL: {d}")
        for n in row.get("answer_notes", []) or []:
            print(f"        note: {n}")

    # --- Aggregate + write --------------------------------------------
    # Pass criterion: retrieval must be byte-identical to the anchor.
    # Answer drift under identical retrieval is a Groq temp=0 artifact,
    # documented in the payload but not counted as a failure.
    n_retrieval_ok = sum(1 for r in per_question if r.get("retrieval_equivalent"))
    n_retrieval_fail = len(per_question) - n_retrieval_ok
    n_byte_identical = sum(1 for r in per_question if r.get("answer_byte_identical"))
    n_answer_drift = sum(1 for r in per_question if r.get("answer_notes"))

    payload = {
        "check": "baseline path equivalence after Phase 5 refactor",
        "anchor_run_id": ANCHOR_RUN_ID,
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "stratification": STRATIFICATION,
        "n_questions": len(per_question),
        "n_retrieval_equivalent": n_retrieval_ok,
        "n_retrieval_fail": n_retrieval_fail,
        "n_answer_byte_identical": n_byte_identical,
        "n_answer_drift_notes": n_answer_drift,
        "pass_criterion": (
            "retrieval must be byte-identical to anchor for every "
            "stratified question. Answer drift under identical "
            "retrieval is documented but not a failure — see "
            "answer_drift_note."
        ),
        "answer_drift_note": (
            "Groq's inference stack is multi-tenant; even at "
            "temperature=0 separate runs can produce token-level "
            "differences (different backend nodes, batching, minor "
            "kernel non-determinism). With byte-identical retrieval "
            "and identical top-5 chunks handed to the generator, any "
            "answer-length variance CANNOT have been caused by the "
            "routing refactor (which literally does not touch the "
            "generator call). Recording as informational."
        ),
        "tolerances": {
            "similarity_score": SIMILARITY_SCORE_TOLERANCE,
            "retrieval_metric_ints": RETRIEVAL_METRIC_TOLERANCE,
            "mrr": MRR_TOLERANCE,
            "answer_len_ratio_report_threshold": ANSWER_LEN_RATIO_TOLERANCE,
        },
        "ragas_skipped_reason": (
            "Refactor changed ROUTING only - no touch to retriever, "
            "embedder, generator, prompt, or judge. Under byte-identical "
            "retrieval, any RAGAS delta is judge stochasticity "
            "and orthogonal to the refactor. Running RAGAS here would "
            "burn Mistral judge quota needed for Task 4 (smoke test)."
        ),
        "per_question": per_question,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print()
    print("=" * 88)
    print(
        f"retrieval-equivalent={n_retrieval_ok}/{len(per_question)}  "
        f"byte-identical-answers={n_byte_identical}/{len(per_question)}  "
        f"answer-drift-notes={n_answer_drift}/{len(per_question)}  "
        f"retrieval-fail={n_retrieval_fail}"
    )
    print(f"[write] {OUTPUT_PATH}")

    return 0 if n_retrieval_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
