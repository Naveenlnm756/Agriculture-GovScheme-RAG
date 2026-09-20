"""
Phase 5, fix #1 — reranker end-to-end smoke test.

Runs the FIVE named smoke-test queries (same set as the Phase 2
generator smoke test) through the `reranked` pipeline mode and prints,
for each:

  * The semantic retriever's top-20 candidates (chunk_id, source,
    similarity_score) — pre-rerank state.
  * The cross-encoder's top-5 reranked selection (chunk_id, source,
    original_rank, new_rank, rerank_score) — post-rerank state.
  * A "movement" summary — which candidates were promoted into the
    top-5, and which top-5 semantic candidates were demoted out of
    the reranked selection. This is the whole point of showing this:
    the reranker's actual EFFECT on retrieval, not just its scores.
  * The generated answer with inline citations.

Deliberately does NOT run RAGAS or persist a summary in the ablation
table format — this is a correctness smoke test for the reranker
plumbing, not a metric-producing eval row. The full 78-question
reranked eval is a separate step and will not be run until this
smoke test is reviewed.

Output goes to stdout AND to
`eval/results/reranker_smoke_test.txt` (verbatim mirror) so the
review artefact is preserved on disk.

Usage
-----
    python scripts/smoke_test_reranker.py

That is the entire interface. No flags, no options. The five
queries are baked in because the point is a REPEATABLE smoke test
across Phase 5 iterations — comparing the reranker's behaviour on
Q1 today vs Q1 in a week matters, and comparing Q1 vs a
freshly-chosen "similar" query does not.
"""

from __future__ import annotations

import io
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Windows cp1252 stdout can't print math symbols / arrows we use in the
# movement summary. Wrap to utf-8 so the smoke output isn't garbled.
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from src.config import settings  # noqa: E402
from src.eval.groq_eval_client import EvalGroqClient  # noqa: E402
from src.retrieval.reranker import rerank  # noqa: E402
from src.retrieval.retriever import retrieve  # noqa: E402
from src.utils.key_rotator import KeyRotator  # noqa: E402


logger = logging.getLogger(__name__)


SMOKE_QUERIES: list[dict[str, str]] = [
    {
        "id": "S1",
        "query": "How do I apply for PM-KISAN?",
        "expected": "PM-KISAN new-registration workflow.",
    },
    {
        "id": "S2",
        "query": "What is the LTV limit for gold loans?",
        "expected": "OUT OF SCOPE — refusal expected.",
    },
    {
        "id": "S3",
        "query": "PMFBY claim procedure for crop loss due to hailstorm",
        "expected": "PMFBY localised-calamity claim procedure.",
    },
    {
        "id": "S4",
        "query": "Kisan Credit Card interest subvention",
        "expected": "MISS interest subvention on KCC (2%+3% PRI).",
    },
    {
        "id": "S5",
        "query": "How long is the completion window for AIF projects?",
        "expected": "AIF project completion window.",
    },
]


class Tee:
    """
    Duplicate stdout to a file so the review artefact is preserved
    without losing the on-screen output.

    Kept dead-simple — one write() forwards to both sinks, one flush()
    flushes both. `close()` closes the file only (never sys.stdout).
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, text: str) -> int:
        n = self._stdout.write(text)
        self._file.write(text)
        return n

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _short_source(chunk) -> str:
    """One-line source label for a chunk — scheme, filename or workflow, page."""
    if chunk.source_type == "workflow":
        return f"{chunk.scheme}/{chunk.workflow_id}"
    page = ""
    if chunk.page_start is not None:
        page = f" p.{chunk.page_start}"
        if chunk.page_end and chunk.page_end != chunk.page_start:
            page += f"-{chunk.page_end}"
    return f"{chunk.scheme}/{chunk.source_filename}{page}"


def _print_candidate_pool(candidates: list) -> None:
    """Print the semantic retriever's top-20 candidates as a numbered table."""
    print(f"  Semantic top-{len(candidates)} candidates (pre-rerank):")
    print(f"    {'rank':>4}  {'sim':>7}  {'chunk_id':<45}  source")
    print(f"    {'-'*4}  {'-'*7}  {'-'*45}  {'-'*40}")
    for c in candidates:
        print(
            f"    {c.rank:>4}  {c.similarity_score:>7.4f}  "
            f"{c.chunk_id[:45]:<45}  {_short_source(c)}"
        )


def _print_reranked_selection(reranked: list, candidates: list) -> None:
    """
    Print the reranker's top-5 with movement info.

    original_rank is the pre-rerank rank the chunk had in `candidates`;
    rank is the new post-rerank rank. Combining both makes the
    reranker's effect readable at a glance.
    """
    print(f"  Reranked top-{len(reranked)} (post-rerank):")
    print(
        f"    {'new':>3}  {'orig':>4}  {'move':>6}  "
        f"{'rerank':>8}  {'sim':>7}  {'chunk_id':<45}  source"
    )
    print(
        f"    {'-'*3}  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*7}  "
        f"{'-'*45}  {'-'*40}"
    )
    for c in reranked:
        orig = c.original_rank if c.original_rank is not None else "?"
        try:
            move_int = int(orig) - c.rank
            move_str = f"{move_int:+d}" if move_int != 0 else "0"
        except (TypeError, ValueError):
            move_str = "?"
        rerank_s = c.rerank_score if c.rerank_score is not None else float("nan")
        print(
            f"    {c.rank:>3}  {orig!s:>4}  {move_str:>6}  "
            f"{rerank_s:>8.4f}  {c.similarity_score:>7.4f}  "
            f"{c.chunk_id[:45]:<45}  {_short_source(c)}"
        )


def _print_movement_summary(reranked: list, candidates: list) -> None:
    """
    Summarise which candidates were promoted into and demoted out of the top-5.

    'Promoted' = present in reranked top-5 but was outside the semantic
    top-5 (original_rank > 5). This is the reranker's real "did anything
    change" signal — an identical top-5 means the cross-encoder didn't
    disagree with the embedder on this query.
    """
    baseline_top5_ids = {c.chunk_id for c in candidates[:5]}
    reranked_top5_ids = {c.chunk_id for c in reranked}

    promoted = [c for c in reranked if c.chunk_id not in baseline_top5_ids]
    demoted = [c for c in candidates[:5] if c.chunk_id not in reranked_top5_ids]

    print("  Movement:")
    if not promoted and not demoted:
        print("    (identical top-5 — reranker made no changes to the "
              "generator's context)")
        return
    for c in promoted:
        print(
            f"    PROMOTED IN : orig_rank={c.original_rank} -> new_rank={c.rank}  "
            f"{c.chunk_id[:45]}  ({_short_source(c)})"
        )
    for c in demoted:
        print(
            f"    DEMOTED OUT : orig_rank={c.rank}  "
            f"{c.chunk_id[:45]}  ({_short_source(c)})"
        )


def _run_one_query(smoke: dict, eval_client: EvalGroqClient) -> None:
    print()
    print("=" * 100)
    print(f"[{smoke['id']}] {smoke['query']}")
    print(f"      expected: {smoke['expected']}")
    print("=" * 100)

    # Semantic retrieval — top 20 candidates for the reranker.
    candidates = retrieve(
        smoke["query"], top_k=settings.reranker_top_n, config=settings
    )
    _print_candidate_pool(candidates)

    print()
    reranked = rerank(
        query=smoke["query"],
        candidates=candidates,
        top_k=settings.reranker_top_k,
        config=settings,
    )
    _print_reranked_selection(reranked, candidates)

    print()
    _print_movement_summary(reranked, candidates)

    # Generation — same eval_client path the harness would use.
    print()
    print("  Generating answer on the reranked top-5 ...")
    gen_result = eval_client.generate_answer(smoke["query"], reranked)

    print()
    print("  Generated answer:")
    print("  " + "-" * 96)
    for line in gen_result.answer.splitlines():
        print(f"  {line}")
    print("  " + "-" * 96)
    print(
        f"  [tokens] prompt={gen_result.prompt_tokens} "
        f"completion={gen_result.completion_tokens} "
        f"reasoning={gen_result.reasoning_tokens}  "
        f"latency={gen_result.latency_ms}ms  "
        f"finish={gen_result.finish_reason}  "
        f"retries={gen_result.retries_taken}"
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    output_path = (
        PROJECT_ROOT / "eval" / "results" / "reranker_smoke_test.txt"
    )
    tee = Tee(output_path)
    sys.stdout = tee

    try:
        print("=" * 100)
        print("RERANKER SMOKE TEST — Phase 5, fix #1")
        print("=" * 100)
        print(f"ran_at         : {datetime.now().isoformat(timespec='seconds')}")
        print(f"reranker_model : {settings.reranker_model}")
        print(f"reranker_top_n : {settings.reranker_top_n} "
              "(semantic candidates fed to the reranker)")
        print(f"reranker_top_k : {settings.reranker_top_k} "
              "(reranker output size)")
        print(f"generator      : {settings.llm_provider}/{settings.llm_model}")
        print(f"n_queries      : {len(SMOKE_QUERIES)}")

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

        for smoke in SMOKE_QUERIES:
            _run_one_query(smoke, eval_client)

        print()
        print("=" * 100)
        print("SMOKE TEST DONE")
        print("=" * 100)
        print(f"[write] {output_path}")
    finally:
        # Restore stdout so the "[write]" line prints normally and any
        # exception trace lands on the terminal rather than being lost.
        sys.stdout = tee._stdout
        tee.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
