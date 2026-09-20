"""
Re-aggregate an existing eval run under the new out_of_scope
handling — no LLM calls, no judge calls, no re-execution.

Reads the on-disk `results.jsonl` for a completed run, applies the
deterministic `refusal_correct` metric to every out_of_scope row
(source of truth: `src/config.OUT_OF_SCOPE_REFUSAL_PHRASES`), then
re-aggregates using the SAME `_aggregate` / `_print_summary`
functions that live in `run_eval.py`. Result is written alongside
the original as `summary_v2.json` — the original `summary.json` and
`results.jsonl` are NEVER touched (provenance constraint).

Why a separate script (rather than "just re-run"):
  * Preserves the original LLM-call and judge outputs bit-for-bit;
    the new numbers are provably a re-aggregation, not a re-run.
  * Costs zero API tokens. On a 78-question baseline this is the
    difference between ~3 minutes of wall time and ~50 minutes.
  * Same aggregation code path as future runs, so a delta between
    baseline_v2 and Phase-5 config rows is a pure technique delta.

USAGE
-----
    python eval/reaggregate.py 20260913-104053_baseline_n78
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.models import EvalQuestionResult, EvalSummary  # noqa: E402

# Reuse the exact aggregation + print + refusal detector that
# `run_eval.py` uses on live runs. Any future change to metric
# handling ends up in one place.
from eval.run_eval import (  # noqa: E402
    _OUT_OF_SCOPE_CATEGORY,
    _aggregate,
    _compute_refusal_correct,
    _print_summary,
    _run_output_dir,
    _write_result,
)


def _load_results_rows(run_dir: Path) -> list[EvalQuestionResult]:
    """
    Load every completed question row from results.jsonl.

    Rows written before `refusal_correct` existed have no such field;
    pydantic falls back to the model's default of None on parse.
    """
    jsonl_path = run_dir / "results.jsonl"
    if not jsonl_path.exists():
        raise SystemExit(
            f"results.jsonl not found at {jsonl_path}. "
            "Nothing to re-aggregate."
        )
    rows: list[EvalQuestionResult] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(EvalQuestionResult(**json.loads(line)))
    return rows


def _load_original_summary(run_dir: Path) -> dict:
    """Return the parsed contents of the run's summary.json."""
    p = run_dir / "summary.json"
    if not p.exists():
        raise SystemExit(f"summary.json not found at {p}.")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_refusal_correct(rows: list[EvalQuestionResult]) -> int:
    """
    Populate `refusal_correct` on every out_of_scope row from its
    stored `generated_answer`. Returns the count of rows updated.

    Non-out_of_scope rows are left untouched (refusal_correct stays
    None on them — the metric doesn't apply).
    """
    n = 0
    for r in rows:
        if r.category == _OUT_OF_SCOPE_CATEGORY:
            r.refusal_correct = _compute_refusal_correct(r.generated_answer)
            n += 1
    return n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Re-aggregate an existing eval run under the new "
            "out_of_scope handling (refusal_correct metric + "
            "exclusion of out_of_scope from RAGAS averages). "
            "Writes summary_v2.json alongside the original summary.json."
        ),
    )
    p.add_argument(
        "run_id",
        type=str,
        help="Run id under eval/results/, e.g. 20260913-104053_baseline_n78",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = _run_output_dir(args.run_id)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    rows = _load_results_rows(run_dir)
    n_refusal = _apply_refusal_correct(rows)
    print(
        f"[reaggregate] loaded {len(rows)} rows from {run_dir.name}, "
        f"scored refusal_correct on {n_refusal} out_of_scope row(s)."
    )

    overall, per_category, failed_computations = _aggregate(rows)

    # Preserve everything from the original summary that depends on
    # runtime state (rotation history, token counters, key stats).
    # We are strictly re-aggregating metrics, not re-simulating the
    # run — the audit fields have to survive unchanged so the
    # provenance chain from summary_v2.json → results.jsonl →
    # original summary.json stays intact.
    orig = _load_original_summary(run_dir)
    orig_summary = orig.get("summary", {})
    rotation_history = orig.get("rotation_history", [])

    total_n = len(rows)
    ragas_n = sum(1 for r in rows if r.category != _OUT_OF_SCOPE_CATEGORY)
    refusal_n = sum(1 for r in rows if r.category == _OUT_OF_SCOPE_CATEGORY)

    # Recompute caveats. Same rule as run_eval.py: only fire per-metric
    # "None" caveats on RAGAS-eligible rows, and only when RAGAS wasn't
    # skipped in the original run.
    caveats: list[str] = []
    original_skipped_ragas = any(
        "skipped via --skip_ragas" in c
        for c in orig_summary.get("caveats", [])
    )
    if original_skipped_ragas:
        caveats.append(
            "RAGAS was skipped in the original run (--skip_ragas); "
            "RAGAS metrics remain None. refusal_correct was still "
            "computed deterministically."
        )
    else:
        for metric_name, n_failed in failed_computations.items():
            if n_failed > 0:
                caveats.append(
                    f"{metric_name}: {n_failed}/{ragas_n} in-scope questions "
                    f"returned None (judge failure / RAGAS internal error). "
                    f"Average is over the remaining {ragas_n - n_failed} "
                    f"successful computations; see per-bucket "
                    f"`{metric_name}_effective_n`."
                )
    caveats.append(
        "Re-aggregated by eval/reaggregate.py: out_of_scope questions "
        "scored with deterministic refusal_correct and excluded from "
        "RAGAS overall averages. Original results.jsonl and summary.json "
        "unchanged."
    )

    summary = EvalSummary(
        run_id=orig_summary.get("run_id", args.run_id),
        pipeline_mode=orig_summary.get("pipeline_mode", "baseline"),
        generation_model=orig_summary.get(
            "generation_model",
            f"{settings.llm_provider}/{settings.llm_model}",
        ),
        judge_model=orig_summary.get("judge_model", ""),
        judge_is_same_as_generation=orig_summary.get(
            "judge_is_same_as_generation", False
        ),
        n_questions=total_n,
        wall_time_s=orig_summary.get("wall_time_s", 0.0),
        total_n_questions=total_n,
        ragas_metrics_n_questions=ragas_n,
        refusal_metric_n_questions=refusal_n,
        overall=overall,
        per_category=per_category,
        total_llm_calls=orig_summary.get("total_llm_calls", 0),
        total_prompt_tokens=orig_summary.get("total_prompt_tokens", 0),
        total_completion_tokens=orig_summary.get("total_completion_tokens", 0),
        total_reasoning_tokens=orig_summary.get("total_reasoning_tokens", 0),
        retrieval_failure_count=orig_summary.get("retrieval_failure_count", 0),
        caveats=caveats,
        failed_computations=failed_computations,
        n_keys_configured=orig_summary.get("n_keys_configured", 0),
        key_rotations=orig_summary.get("key_rotations", 0),
        per_key_call_counts=orig_summary.get("per_key_call_counts", {}),
        cooldown_events=orig_summary.get("cooldown_events", 0),
        rate_limit_errors=orig_summary.get("rate_limit_errors", 0),
        mistral_stats=orig_summary.get("mistral_stats", {}),
    )

    out_path = run_dir / "summary_v2.json"
    _write_result(
        {
            "summary": summary.model_dump(),
            "rotation_history": rotation_history,
            "reaggregation_note": (
                "Recomputed by eval/reaggregate.py. Source rows: "
                "results.jsonl (unchanged). Original summary: summary.json "
                "(unchanged). This file is derived, not authoritative for "
                "the original run's audit trail."
            ),
        },
        out_path,
    )
    print(f"\n[write] {out_path}")
    _print_summary(summary)


if __name__ == "__main__":
    main()
