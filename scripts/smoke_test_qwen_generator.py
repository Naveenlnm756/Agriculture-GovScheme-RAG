"""
One-off smoke test: try qwen/qwen3.6-27b as generator, no config.py change.

Overrides settings.llm_model at runtime for the duration of this process
only. Retriever, system prompt, top_k, reasoning_effort, temperature, and
every other knob are left at their config.py defaults. Same 5 queries as
scripts/run_generator_check.py so the output is directly comparable.

Also dumps the raw completion_tokens_details from each Groq response so
we can see whether qwen exposes a reasoning channel.

Run from the project root:
    python scripts/smoke_test_qwen_generator.py
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.models import RetrievalResult  # noqa: E402
from src.retrieval.retriever import retrieve  # noqa: E402

# Override the LLM model BEFORE importing the generator's client cache
# uses it. The generator reads config.llm_model at call time, not import
# time, so mutation here is safe.
QWEN_MODEL = "qwen/qwen3.6-27b"
_original_model = settings.llm_model
settings.llm_model = QWEN_MODEL

# qwen/qwen3.6-27b on Groq rejects the "low" / "medium" / "high" scale
# used by openai/gpt-oss-*. It accepts only "none" or "default". We pick
# "none" as the closest analogue to gpt-oss-20b @ reasoning_effort="low"
# (near-zero reasoning) so the smoke test stays an apples-to-apples
# generator-swap comparison. All other config unchanged.
_original_effort = settings.generation_reasoning_effort
settings.generation_reasoning_effort = "none"

from src.generation.generator import (  # noqa: E402
    _build_system_prompt,
    _build_user_message,
    _call_groq_with_retry,
    _load_groq_client,
)


QUERIES: list[str] = [
    "How do I apply for PM-KISAN?",
    "What is the LTV limit for gold loans?",
    "PMFBY claim procedure for crop loss due to hailstorm",
    "Kisan Credit Card interest subvention",
    "How long is the completion window for AIF projects?",
]

CITATION_RE = re.compile(r"\[Source:\s*[^\]]+\]")


def _generate_with_raw(query: str, chunks: list[RetrievalResult]):
    """Same as generator.generate() but also returns the raw completion."""
    client = _load_groq_client(settings)
    system_prompt = _build_system_prompt(settings)
    user_message = _build_user_message(query, chunks)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    start_ns = time.perf_counter_ns()
    completion, retries = _call_groq_with_retry(client, messages, settings)
    latency_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
    return completion, retries, latency_ms


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings.generation_max_tokens=1200
    print(f"\n### Generator smoke test: model={settings.llm_model!r}, "
          f"reasoning_effort={settings.generation_reasoning_effort!r}, "
          f"temperature={settings.generation_temperature}, "
          f"max_tokens={settings.generation_max_tokens} ###\n")
    print(f"(Original config.py llm_model was {_original_model!r}, "
          f"reasoning_effort was {_original_effort!r}; unchanged on disk.)\n")

    for qi, query in enumerate(QUERIES, start=1):
        print()
        print("=" * 88)
        print(f"QUERY {qi}: {query!r}")
        print("=" * 88)

        chunks = retrieve(query)

        print()
        print("-- retrieved sources (top 5) --")
        if not chunks:
            print("  (no chunks returned)")
        for c in chunks:
            print(
                f"  [rank {c.rank}]  sim={c.similarity_score:.4f}  "
                f"scheme={c.scheme}  type={c.source_type}  "
                f"id={c.chunk_id}"
            )

        try:
            completion, retries, latency_ms = _generate_with_raw(query, chunks)
        except Exception as e:
            print()
            print("-- generation FAILED --")
            print(f"  {type(e).__name__}: {e}")
            continue

        choice = completion.choices[0]
        answer = choice.message.content or ""
        finish_reason = choice.finish_reason or ""

        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        ctd = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = 0
        if ctd is not None:
            reasoning_tokens = int(getattr(ctd, "reasoning_tokens", 0) or 0)

        print()
        print("-- generated answer --")
        print(answer if answer.strip() else "(EMPTY CONTENT)")

        citations = CITATION_RE.findall(answer)
        print()
        print(f"-- citations parsed ({len(citations)}) --")
        for c in citations:
            print(f"  {c}")

        print()
        print("-- generation stats --")
        print(f"  model              : {settings.llm_model}")
        print(f"  prompt_tokens      : {prompt_tokens}")
        print(f"  completion_tokens  : {completion_tokens}")
        print(f"  reasoning_tokens   : {reasoning_tokens}")
        answer_toks = max(completion_tokens - reasoning_tokens, 0)
        print(f"  answer_tokens      : {answer_toks}")
        print(f"  total_tokens       : {total_tokens}")
        print(f"  latency_ms         : {latency_ms}")
        print(f"  finish_reason      : {finish_reason}")
        print(f"  retries_taken      : {retries}")

        print()
        print("-- raw completion_tokens_details --")
        if ctd is None:
            print("  (usage.completion_tokens_details is None — no reasoning channel exposed)")
        else:
            try:
                print(f"  {ctd.model_dump() if hasattr(ctd, 'model_dump') else vars(ctd)}")
            except Exception:
                print(f"  repr: {ctd!r}")


if __name__ == "__main__":
    main()
