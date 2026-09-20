"""
Phase 5, fix #2 — hybrid (semantic + BM25 → RRF) end-to-end smoke test.

Runs the FIVE named smoke-test queries (same set as the Phase 2
generator smoke test and the reranker smoke test) through the `hybrid`
pipeline mode and prints, for each:

  * The semantic retriever's top-N candidates (chunk_id, source,
    similarity_score) — pre-fusion state.
  * The BM25 retriever's top-N candidates (chunk_id, source,
    bm25_score) — pre-fusion state.
  * The RRF fusion's top-K selection (chunk_id, source,
    semantic_rank, bm25_rank, rrf_score) — post-fusion state.
  * A "movement" summary — which candidates were promoted into the
    top-K by fusion (i.e. surfaced from BM25-only or from deeper
    semantic ranks), and which top-K semantic candidates were
    demoted out of the fused selection. This is the whole point of
    showing this: the fusion's actual EFFECT on retrieval, not just
    its scores.
  * For queries S1 and S3, the FULL RRF calculation table over the
    union of both pools: rank_semantic, rank_bm25,
    semantic_contribution=1/(k+rank_semantic),
    bm25_contribution=1/(k+rank_bm25), rrf_score.
    This makes the fusion inspectable so a reviewer can verify RRF is
    doing what it says on the tin, not just producing plausible-looking
    rankings.
  * The generated answer with inline citations.

Deliberately does NOT run RAGAS or persist a summary in the ablation
table format — this is a correctness smoke test for the hybrid
plumbing, not a metric-producing eval row. The full 78-question
hybrid eval is a separate step and will not be run until this smoke
test is reviewed.

Output goes to stdout AND to
`eval/results/hybrid_smoke_test.txt` (verbatim mirror) so the review
artefact is preserved on disk.

Usage
-----
    python scripts/smoke_test_hybrid.py

That is the entire interface. No flags, no options. The five queries
are baked in because the point is a REPEATABLE smoke test across
Phase 5 iterations — comparing hybrid behaviour on Q1 today vs Q1 in
a week matters, and comparing Q1 vs a freshly-chosen "similar" query
does not.
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
from src.ingestion.models import RetrievalResult  # noqa: E402
from src.retrieval.bm25_search import bm25_search  # noqa: E402
from src.retrieval.hybrid import hybrid_search  # noqa: E402
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

# Queries whose full RRF-calculation table is dumped. Two is enough to
# make the fusion mechanically inspectable — one refusal-flavoured
# (S2 out-of-scope would be dull, so instead we pick S1 = clear PDF+workflow
# match) and one hard-hitting content query (S3 = PMFBY hailstorm, expected
# to force the fusion to work). Keeping the list short avoids drowning
# the smoke artefact in a 40-row table for every question.
QUERIES_WITH_FULL_RRF_TABLE: set[str] = {"S1", "S3"}


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


def _short_source(chunk: RetrievalResult) -> str:
    """One-line source label for a chunk — scheme, filename or workflow, page."""
    if chunk.source_type == "workflow":
        return f"{chunk.scheme}/{chunk.workflow_id}"
    page = ""
    if chunk.page_start is not None:
        page = f" p.{chunk.page_start}"
        if chunk.page_end and chunk.page_end != chunk.page_start:
            page += f"-{chunk.page_end}"
    return f"{chunk.scheme}/{chunk.source_filename}{page}"


def _print_semantic_pool(candidates: list[RetrievalResult]) -> None:
    """Print the semantic retriever's top-N candidates as a numbered table."""
    print(f"  Semantic top-{len(candidates)} candidates (pre-fusion):")
    print(f"    {'rank':>4}  {'sim':>7}  {'chunk_id':<45}  source")
    print(f"    {'-'*4}  {'-'*7}  {'-'*45}  {'-'*40}")
    for c in candidates:
        sim = c.similarity_score if c.similarity_score is not None else float("nan")
        print(
            f"    {c.rank:>4}  {sim:>7.4f}  "
            f"{c.chunk_id[:45]:<45}  {_short_source(c)}"
        )


def _print_bm25_pool(candidates: list[RetrievalResult]) -> None:
    """Print the BM25 retriever's top-N candidates as a numbered table."""
    print(f"  BM25 top-{len(candidates)} candidates (pre-fusion):")
    print(f"    {'rank':>4}  {'bm25':>8}  {'chunk_id':<45}  source")
    print(f"    {'-'*4}  {'-'*8}  {'-'*45}  {'-'*40}")
    for c in candidates:
        bm25 = c.bm25_score if c.bm25_score is not None else float("nan")
        print(
            f"    {c.rank:>4}  {bm25:>8.4f}  "
            f"{c.chunk_id[:45]:<45}  {_short_source(c)}"
        )


def _print_fused_selection(
    fused: list[RetrievalResult],
    semantic: list[RetrievalResult],
    bm25: list[RetrievalResult],
) -> None:
    """
    Print the hybrid top-K with movement info.

    `semantic_rank` and `bm25_rank` are the source-pool ranks; `rank`
    is the post-fusion position. Combining all three makes the
    fusion's effect readable at a glance.
    """
    print(f"  Hybrid RRF top-{len(fused)} (post-fusion):")
    print(
        f"    {'new':>3}  {'s_rk':>4}  {'b_rk':>4}  "
        f"{'rrf':>8}  {'sim':>7}  {'bm25':>8}  "
        f"{'chunk_id':<45}  source"
    )
    print(
        f"    {'-'*3}  {'-'*4}  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*8}  "
        f"{'-'*45}  {'-'*40}"
    )
    for c in fused:
        s_rk = c.semantic_rank if c.semantic_rank is not None else "—"
        b_rk = c.bm25_rank if c.bm25_rank is not None else "—"
        rrf = c.rrf_score if c.rrf_score is not None else float("nan")
        sim = c.similarity_score if c.similarity_score is not None else float("nan")
        bm25_s = c.bm25_score if c.bm25_score is not None else float("nan")
        print(
            f"    {c.rank:>3}  {s_rk!s:>4}  {b_rk!s:>4}  "
            f"{rrf:>8.5f}  {sim:>7.4f}  {bm25_s:>8.4f}  "
            f"{c.chunk_id[:45]:<45}  {_short_source(c)}"
        )


def _print_movement_summary(
    fused: list[RetrievalResult],
    semantic: list[RetrievalResult],
    bm25: list[RetrievalResult],
) -> None:
    """
    Summarise which candidates were promoted into and demoted out of the top-K.

    Baseline for comparison is the semantic top-K (what the baseline
    pipeline would have shown the generator). PROMOTED chunks are the
    ones fusion added; DEMOTED are the ones fusion dropped. This is
    the fusion's real "did anything change" signal — an identical
    top-K to semantic would mean BM25 contributed nothing here.
    """
    baseline_topk_ids = {c.chunk_id for c in semantic[: len(fused)]}
    fused_topk_ids = {c.chunk_id for c in fused}
    bm25_all_ids = {c.chunk_id for c in bm25}
    semantic_all_ids = {c.chunk_id for c in semantic}

    promoted = [c for c in fused if c.chunk_id not in baseline_topk_ids]
    demoted = [c for c in semantic[: len(fused)] if c.chunk_id not in fused_topk_ids]

    print("  Movement:")
    if not promoted and not demoted:
        print(
            "    (identical top-K to semantic — BM25 did not change the "
            "generator's context on this query)"
        )
        return
    for c in promoted:
        # Origin annotation — was this chunk in BM25 only, semantic only,
        # or both? Answering that in the smoke output is what lets a
        # reviewer see "hybrid pulled in a BM25-only chunk" vs "hybrid
        # promoted a deeper semantic chunk over a shallower one."
        origin_parts = []
        if c.chunk_id in semantic_all_ids:
            origin_parts.append(f"sem_rank={c.semantic_rank}")
        if c.chunk_id in bm25_all_ids:
            origin_parts.append(f"bm25_rank={c.bm25_rank}")
        origin = ", ".join(origin_parts) if origin_parts else "unknown"
        print(
            f"    PROMOTED IN : new_rank={c.rank}  ({origin})  "
            f"{c.chunk_id[:45]}  ({_short_source(c)})"
        )
    for c in demoted:
        # For demoted, `c` is a semantic hit — `bm25_rank`/`rrf_score`
        # are None on it. Report its original semantic rank; the fact
        # it fell out of the fused top-K IS the point.
        print(
            f"    DEMOTED OUT : sem_rank={c.rank}  "
            f"{c.chunk_id[:45]}  ({_short_source(c)})"
        )


def _print_full_rrf_table(
    query: str,
    semantic: list[RetrievalResult],
    bm25: list[RetrievalResult],
    config=settings,
) -> None:
    """
    Dump the full RRF calculation over the UNION of both pools.

    One row per unique chunk_id, showing:
      * rank in the semantic pool (or `—` if not present)
      * rank in the BM25 pool     (or `—` if not present)
      * 1/(k + rank_semantic)     (0 if not present)
      * 1/(k + rank_bm25)         (0 if not present)
      * total rrf_score           (sum of the two contributions)

    Sorted descending by rrf_score. This is the audit trail per the
    Phase 5 design constraint — a reviewer can verify by hand that
    RRF is doing the arithmetic the module claims, not just producing
    plausible-looking rankings.

    Independently re-derived here from the two pool lists so this
    printer is not just re-reading `hybrid.py`'s output — a bug in
    hybrid.py would surface as a mismatch between this table and
    the "Hybrid RRF top-K" table above.
    """
    k = config.hybrid_rrf_k
    sem_by_id = {c.chunk_id: c for c in semantic}
    bm_by_id = {c.chunk_id: c for c in bm25}
    union_ids = set(sem_by_id.keys()) | set(bm_by_id.keys())

    rows: list[tuple[float, str, str, str, float, float]] = []
    for cid in union_ids:
        s = sem_by_id.get(cid)
        b = bm_by_id.get(cid)
        s_rank_disp = str(s.rank) if s is not None else "—"
        b_rank_disp = str(b.rank) if b is not None else "—"
        s_contrib = (1.0 / (k + s.rank)) if s is not None else 0.0
        b_contrib = (1.0 / (k + b.rank)) if b is not None else 0.0
        total = s_contrib + b_contrib
        rows.append((total, cid, s_rank_disp, b_rank_disp, s_contrib, b_contrib))
    rows.sort(key=lambda t: (-t[0], t[1]))

    print(f"  Full RRF calculation over the union (k={k}):")
    print(
        f"    {'rank':>4}  {'s_rk':>4}  {'b_rk':>4}  "
        f"{'s_ctb':>8}  {'b_ctb':>8}  {'rrf_total':>10}  "
        f"{'chunk_id':<45}"
    )
    print(
        f"    {'-'*4}  {'-'*4}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*10}  "
        f"{'-'*45}"
    )
    for rank_pos, (total, cid, s_rk, b_rk, s_c, b_c) in enumerate(rows, start=1):
        print(
            f"    {rank_pos:>4}  {s_rk:>4}  {b_rk:>4}  "
            f"{s_c:>8.5f}  {b_c:>8.5f}  {total:>10.5f}  "
            f"{cid[:45]:<45}"
        )


def _run_one_query(smoke: dict, eval_client: EvalGroqClient) -> None:
    print()
    print("=" * 100)
    print(f"[{smoke['id']}] {smoke['query']}")
    print(f"      expected: {smoke['expected']}")
    print("=" * 100)

    # Independent semantic and BM25 pool runs so the smoke output shows
    # each retriever's raw view. hybrid_search() re-invokes them under
    # the hood — that redundancy is intentional in the smoke test so a
    # reader sees exactly what each pool contributed, and any drift
    # between "what the pool shows" and "what hybrid used" becomes
    # visible as a bug.
    semantic_pool = retrieve(
        smoke["query"], top_k=settings.hybrid_semantic_top_n, config=settings
    )
    _print_semantic_pool(semantic_pool)

    print()
    bm25_pool = bm25_search(
        smoke["query"], top_k=settings.hybrid_bm25_top_n, config=settings
    )
    _print_bm25_pool(bm25_pool)

    print()
    fused = hybrid_search(
        query=smoke["query"],
        top_k=settings.hybrid_top_k,
        config=settings,
    )
    _print_fused_selection(fused, semantic_pool, bm25_pool)

    print()
    _print_movement_summary(fused, semantic_pool, bm25_pool)

    if smoke["id"] in QUERIES_WITH_FULL_RRF_TABLE:
        print()
        _print_full_rrf_table(
            query=smoke["query"],
            semantic=semantic_pool,
            bm25=bm25_pool,
            config=settings,
        )

    # Generation — same eval_client path the harness would use.
    print()
    print("  Generating answer on the fused top-K ...")
    gen_result = eval_client.generate_answer(smoke["query"], fused)

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
        PROJECT_ROOT / "eval" / "results" / "hybrid_smoke_test.txt"
    )
    tee = Tee(output_path)
    sys.stdout = tee

    try:
        print("=" * 100)
        print("HYBRID SMOKE TEST — Phase 5, fix #2")
        print("=" * 100)
        print(f"ran_at              : {datetime.now().isoformat(timespec='seconds')}")
        print(f"embedding_model     : {settings.embedding_model}")
        print(f"hybrid_semantic_top_n: {settings.hybrid_semantic_top_n} "
              "(semantic pool size)")
        print(f"hybrid_bm25_top_n   : {settings.hybrid_bm25_top_n} "
              "(BM25 pool size)")
        print(f"hybrid_top_k        : {settings.hybrid_top_k} "
              "(fused output size)")
        print(f"hybrid_rrf_k        : {settings.hybrid_rrf_k} "
              "(RRF constant)")
        print(f"generator           : {settings.llm_provider}/{settings.llm_model}")
        print(f"n_queries           : {len(SMOKE_QUERIES)}")
        print(
            f"full RRF table for  : "
            f"{', '.join(sorted(QUERIES_WITH_FULL_RRF_TABLE))}"
        )

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
