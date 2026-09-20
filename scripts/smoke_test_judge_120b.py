"""
One-off smoke test: verify Groq `openai/gpt-oss-120b` works as the RAGAS
judge under the newly-locked config.

What this script proves:
  1. `_build_judge_llm` picks the new `judge_provider="groq"` branch
     (the branch we just added at the top of the function).
  2. LangChain's ChatOpenAI, pointed at Groq's OpenAI-compatible
     endpoint with `reasoning_effort="low"` in `extra_body`, is
     reachable end-to-end from a real RAGAS call.
  3. All four RAGAS metrics (faithfulness, answer_relevancy,
     context_precision, context_recall) return real numbers on a
     hand-picked simple question (Q003 — PM-KISAN new registration),
     rather than crashing or returning NaN.

This is NOT a scored eval. It runs one question. Do not read the
numbers as "the eval baseline" — that requires the full golden set
via `eval/run_eval.py`.

Run from the project root:
    python scripts/smoke_test_judge_120b.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.generation.generator import generate  # noqa: E402
from src.retrieval.retriever import retrieve  # noqa: E402


GOLDEN_SET_PATH = PROJECT_ROOT / "eval" / "golden_set.json"
SMOKE_QUESTION_ID = "Q003"  # PM-KISAN new registration — simple procedural


def _load_golden_question(qid: str) -> dict:
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        payload = json.load(f)
    for q in payload["questions"]:
        if q["question_id"] == qid:
            return q
    raise KeyError(f"Question {qid} not found in golden set")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print()
    print("=" * 88)
    print("JUDGE SMOKE TEST — Groq openai/gpt-oss-120b")
    print("=" * 88)
    print(f"generator          : {settings.llm_model}")
    print(f"generator reasoning: {settings.generation_reasoning_effort}")
    print(f"generator max_toks : {settings.generation_max_tokens}")
    print(f"judge provider     : {settings.judge_provider}")
    print(f"judge model        : {settings.judge_model}")
    print(f"judge reasoning    : {settings.judge_reasoning_effort}")
    print(f"golden question    : {SMOKE_QUESTION_ID}")
    print()

    q = _load_golden_question(SMOKE_QUESTION_ID)
    question = q["question"]
    expected_answer = q["expected_answer"]
    print(f"Q: {question}")
    print()

    # --- Retrieve -----------------------------------------------------
    t0 = time.perf_counter()
    chunks = retrieve(question)
    t_retrieve = time.perf_counter() - t0

    print(f"-- retrieved {len(chunks)} chunks in {t_retrieve:.2f}s --")
    for c in chunks:
        print(
            f"  [rank {c.rank}]  sim={c.similarity_score:.4f}  "
            f"scheme={c.scheme}  type={c.source_type}  id={c.chunk_id}"
        )

    # --- Generate (qwen) ----------------------------------------------
    t0 = time.perf_counter()
    gen = generate(question, chunks)
    t_generate = time.perf_counter() - t0

    print()
    print(f"-- generated answer ({t_generate:.2f}s, "
          f"prompt={gen.prompt_tokens}, completion={gen.completion_tokens}, "
          f"finish={gen.finish_reason}) --")
    print(gen.answer)
    print()

    # --- Judge (gpt-oss-120b via new Groq branch) ---------------------
    # Import here so all the retriever/generator wiring is loaded first,
    # and any judge-init log lines land in the visible timing block.
    from eval.run_eval import (  # noqa: E402
        _build_judge_llm,
        _build_judge_embeddings,
        _compute_ragas_metrics,
    )

    print("-- building judge --")
    t0 = time.perf_counter()
    judge_llm, judge_label, judge_is_same = _build_judge_llm()
    judge_embeds = _build_judge_embeddings()
    t_judge_init = time.perf_counter() - t0
    print(f"  built in {t_judge_init:.2f}s: label={judge_label!r}  "
          f"same_as_generation={judge_is_same}")
    print()

    if judge_is_same:
        print("ERROR: judge fell through to same-model fallback. "
              "Groq branch did not initialise. Aborting smoke test.")
        sys.exit(2)

    contexts = [c.text for c in chunks]

    print("-- running RAGAS (4 metrics on 1 question) --")
    t0 = time.perf_counter()
    metrics = _compute_ragas_metrics(
        question=question,
        answer=gen.answer,
        contexts=contexts,
        expected_answer=expected_answer,
        judge_llm=judge_llm,
        judge_embeddings=judge_embeds,
    )
    t_metrics = time.perf_counter() - t0

    print(f"  finished in {t_metrics:.2f}s")
    print()
    print("-- RAGAS scores --")
    for metric_name, value in metrics.items():
        if value is None:
            print(f"  {metric_name:20s} : None  ⚠️  (RAGAS returned NaN or metric raised)")
        else:
            print(f"  {metric_name:20s} : {value:.4f}")

    print()
    print("-- verdict --")
    none_count = sum(1 for v in metrics.values() if v is None)
    if none_count == 0:
        print("  ✅ all 4 RAGAS metrics returned real numbers. Judge chain is wired.")
    elif none_count < 4:
        print(f"  ⚠️  {none_count}/4 metrics failed (None). Judge is reachable but"
              f" some metrics did not score. Investigate before committing to eval.")
    else:
        print("  ❌ all 4 metrics returned None. Judge is broken.")

    print()
    print(f"total wall time: retrieve={t_retrieve:.2f}s  generate={t_generate:.2f}s  "
          f"judge_init={t_judge_init:.2f}s  metrics={t_metrics:.2f}s")


if __name__ == "__main__":
    main()
