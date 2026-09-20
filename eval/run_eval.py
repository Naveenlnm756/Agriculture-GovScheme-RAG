"""
Eval harness for the Agriculture Schemes RAG pipeline.

USAGE
-----
    # 20-question smoke run against the baseline pipeline (default):
    python eval/run_eval.py

    # Full 78-question anchor run for Phase 5 ablation:
    python eval/run_eval.py --full

    # Implemented modes (Phase 5, in ablation order):
    python eval/run_eval.py --pipeline_mode baseline         # implemented (Phase 4 anchor)
    python eval/run_eval.py --pipeline_mode reranked         # implemented (Phase 5 fix #1)
    python eval/run_eval.py --pipeline_mode hybrid           # implemented (Phase 5 fix #2)
    python eval/run_eval.py --pipeline_mode query_transform  # implemented (Phase 5 fix #3)
    python eval/run_eval.py --pipeline_mode crag             # NotImplementedError
    python eval/run_eval.py --pipeline_mode full             # implemented (headline row: multimodal + hybrid RRF + reranker)

    # Tune pacing (default 3s; raise if you hit 429s, lower if the run
    # is running way under the 8k TPM Groq cap):
    python eval/run_eval.py --sleep 3.0

WHAT IT DOES
------------
1. Loads the golden set and slices to --n_questions (default 20).
2. For each question, runs the configured pipeline (baseline only in
   Phase 4) and captures the answer + retrieval trace.
3. Computes retrieval-side metrics directly (no LLM): hit_rate@5,
   hit_rate@10, MRR. These are cheap and deterministic — they run
   even if RAGAS is broken.
4. Computes RAGAS metrics (faithfulness, answer_relevancy,
   context_precision, context_recall) using an INDEPENDENT judge
   model configured in src/config.py. Judge selection driven by
   `judge_provider`: "ollama" uses local Ollama (default); otherwise
   the cloud chain is Gemini `gemini-3.6-flash` → Mistral
   `mistral-medium-latest` → Groq `openai/gpt-oss-20b` (same-model
   caveat). RAGAS
   `evaluate()` is invoked with `RunConfig(timeout=180, max_retries=3,
   max_wait=30, max_workers=1)`. max_workers=1 serialises judge
   calls under Gemini's 5 RPM free-tier cap — parallel workers just
   burn the RPM budget faster and force every question into retry
   backoff.
5. Writes a per-question record + aggregated summary to
   eval/results/{timestamp}_{mode}_n{count}.json.

WHY THESE THREE DECISIONS HAVE BEEN MADE
----------------------------------------
- **Judge is separate config.** Same-model self-judging inflates
  faithfulness ~10 points in the RAGAS literature; keeping the judge
  independent removes the bias. Keeping the judge model constant across
  every row of an ablation table is the load-bearing rule from
  CLAUDE.md §11 — a judge change would mask whether a metric moved
  because of the technique or because of the judge changing its mind.

- **RAGAS is called one question at a time.** Batching would be
  faster but blows past the 8k TPM Groq cap on parallel judge calls,
  and it hides which question triggered a failure. One-at-a-time with
  a --sleep between calls is slow-but-defensible; we get real error
  attribution and stay under the cap.

- **Retrieval metrics compute independently of RAGAS.** hit_rate + MRR
  don't need a judge — they compare retrieved chunk ids to expected
  sources, which is deterministic. If the judge falls over halfway
  through a run, we still have the retrieval half of the story.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUT_OF_SCOPE_REFUSAL_PHRASES, settings  # noqa: E402
from src.eval.groq_eval_client import EvalGroqClient  # noqa: E402
from src.ingestion.models import (  # noqa: E402
    EvalQuestionResult,
    EvalSummary,
    RetrievalResult,
)
from src.retrieval.retriever import retrieve  # noqa: E402
from src.retrieval.reranker import rerank  # noqa: E402
from src.retrieval.hybrid import hybrid_search  # noqa: E402
from src.retrieval.rrf import rrf_fuse  # noqa: E402
from src.utils.key_rotator import KeyRotator  # noqa: E402


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Golden-set loading                                                    #
# --------------------------------------------------------------------- #

def _load_golden_set(path: Path, n_questions: int) -> list[dict]:
    """
    Load the golden set and slice to `n_questions`.

    `n_questions <= 0` returns the full set. Subsetting is deterministic
    (first N in file order) so a 20-question smoke run and a full
    78-question run compare like-for-like on the same first 20 rows.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Golden set not found at {path}. Phase 3 must land the "
            "hand-built set before eval can run."
        )
    with path.open("r", encoding="utf-8") as f:
        golden = json.load(f)
    questions: list[dict] = golden.get("questions", [])
    if n_questions and n_questions > 0:
        return questions[:n_questions]
    return questions


# --------------------------------------------------------------------- #
# Pipeline mode routing — inline in the harness (no pipeline.py)        #
# --------------------------------------------------------------------- #
#
# Every ablation row is one --pipeline_mode value. Each mode is an
# ISOLATED intervention on top of the baseline, NOT stacked on the
# previous mode. That is: `reranked` = semantic + reranker only, NOT
# semantic + reranker + hybrid + query_transform + ... . The `full` mode
# is the only place where every technique combines. Rationale: isolated
# ablation rows let each row answer "what does THIS one technique
# contribute?" — cumulative rows confound the contribution of
# overlapping techniques and are not standard practice in the ML
# literature. See DECISIONS.md (Phase 5 ablation methodology entry).
#
# Routing lives inline in this harness rather than being extracted to
# a `src/pipeline.py` orchestrator. Pragmatic choice: for the current
# ablation surface (a small handful of modes, each 2-4 lines of glue)
# a dedicated orchestrator module would add an indirection without
# saving code. When a mode grows past a handful of steps or a mode
# needs to be reused outside the eval harness (API server, notebook),
# the extraction becomes worth doing. Recorded in DECISIONS.md.

def _run_pipeline_baseline(
    question: str,
    eval_client: EvalGroqClient,
    config=settings,
) -> tuple[list[RetrievalResult], Any]:
    """
    Baseline pipeline: pure semantic retrieval → grounded generation.

    Retrieves `top_k_retrieval` candidates (default 20) as the candidate
    pool for the retrieval metrics — hit_rate@10 needs at least 10, and
    keeping a pool of 20 costs nothing extra. The generator is only
    passed the top `retrieval_top_k` (default 5) so its answer stays
    grounded on the same slim context the ablation is comparing.

    Generation is dispatched through `eval_client.generate_answer` (not
    the production `generate()` function) so the eval harness gets
    sequential key rotation on 429 without touching production.
    """
    retrieved_pool = retrieve(question, top_k=config.top_k_retrieval, config=config)
    top_for_generator = retrieved_pool[: config.retrieval_top_k]
    gen_result = eval_client.generate_answer(question, top_for_generator)
    return retrieved_pool, gen_result


def _run_pipeline_reranked(
    question: str,
    eval_client: EvalGroqClient,
    config=settings,
) -> tuple[list[RetrievalResult], Any]:
    """
    Reranked pipeline (Phase 5, fix #1): semantic → cross-encoder
    reranker → grounded generation.

    Semantic retriever fetches `reranker_top_n` (default 20) candidates
    for the reranker to re-score. Cross-encoder scores all N, returns
    the top `reranker_top_k` (default 5), and only those are handed to
    the generator. The generator's context depth (5) is intentionally
    the SAME as the baseline (5), so the only variable this ablation
    row studies is the SELECTION of the 5, not their count.

    NB on retrieval metrics: because this mode returns 5 reranked
    chunks (not a 20-deep pool), hit_rate_at_10 will degenerate to
    hit_rate_at_5 for reranked rows — only 5 items exist to inspect.
    That is the correct semantics for this ablation: the reranker's job
    IS "produce the top 5 the generator should see"; if the correct
    chunk is not in that top 5 then the reranker has failed regardless
    of what the semantic top-20 contained. The baseline anchor row
    retains the meaningful hit_rate_at_10 for comparison.

    Generation goes through `eval_client.generate_answer` on the same
    key-rotation path as the baseline — the only stage this function
    changes is between retrieval and generation.
    """
    candidates = retrieve(
        question, top_k=config.reranker_top_n, config=config
    )
    reranked_chunks = rerank(
        query=question,
        candidates=candidates,
        top_k=config.reranker_top_k,
        config=config,
    )
    gen_result = eval_client.generate_answer(question, reranked_chunks)
    return reranked_chunks, gen_result


def _run_pipeline_hybrid(
    question: str,
    eval_client: EvalGroqClient,
    config=settings,
) -> tuple[list[RetrievalResult], Any]:
    """
    Hybrid pipeline (Phase 5, fix #2): semantic + BM25 fusion → RRF →
    grounded generation.

    Semantic retriever fetches `hybrid_semantic_top_n` candidates,
    BM25 fetches `hybrid_bm25_top_n` candidates (both default 20).
    Their union is fused with Reciprocal Rank Fusion (constant
    `hybrid_rrf_k=60`, canonical value from the TREC paper) and the
    top `hybrid_top_k` (default 5) are handed to the generator. The
    generator's context depth (5) is intentionally the SAME as the
    baseline (5), so the only variable this ablation row studies is
    the SELECTION of the 5, not their count.

    Isolated methodology: the reranker is deliberately NOT invoked
    here. Stacking hybrid + reranker in the same row would confound
    which technique moved the metric; that combination is reserved
    for the `full` row at the end of the table.

    NB on retrieval metrics: because this mode returns 5 fused chunks
    (not a 20-deep pool), hit_rate_at_10 collapses to hit_rate_at_5
    for hybrid rows — only 5 items exist to inspect. That is the
    correct semantics for this ablation: the hybrid retriever's job
    IS "produce the top 5 the generator should see"; if the correct
    chunk is not in that top 5 then the fusion has failed regardless
    of what the deeper pools contained. The baseline anchor row
    retains the meaningful hit_rate_at_10 for comparison.

    Generation goes through `eval_client.generate_answer` on the same
    key-rotation path as the baseline and the reranked row — the
    only stage this function changes is between retrieval and
    generation.
    """
    fused_chunks = hybrid_search(
        query=question,
        top_k=config.hybrid_top_k,
        config=config,
    )
    gen_result = eval_client.generate_answer(question, fused_chunks)
    return fused_chunks, gen_result


def _run_pipeline_query_transform(
    question: str,
    eval_client: EvalGroqClient,
    config=settings,
) -> tuple[list[RetrievalResult], Any]:
    """
    Query-transform pipeline (Phase 5, fix #3): multi-query expansion →
    per-query semantic retrieval → RRF fusion → grounded generation.

    Strategy: multi-query expansion (see DECISIONS.md 2026-09-14 fix #3
    design decisions). The LLM rewrites the question into
    `config.query_transform_n_rewrites` (default 3) alternate
    phrasings. Each rewrite PLUS the ORIGINAL question is sent to the
    semantic retriever (top-N each; default 20). The N+1 ranked lists
    are fused via `rrf_fuse` with `config.query_transform_rrf_k`
    (default 60) into a single top-K (default 5) for the generator.

    Design choices encoded here (see DECISIONS.md for full reasoning):

    * Same LLM as generation. The rewriter uses `config.llm_model`
      (openai/gpt-oss-20b on Groq) via `eval_client.rewrite_query`.
      Consistent with CLAUDE.md §11 rule 2's spirit — every LLM call
      in the ablation runs on the committed model.

    * Retrieval-time fusion, not generation-time. One RRF fuse over
      N+1 ranked lists, one generator call. Generation-time fusion
      (N+1 generator calls plus a synthesizer) would 4× the Groq
      spend per question and introduce a new hallucination surface.

    * Original query stays in the fused set (per
      `query_transform_include_original=True`). Provides a strict
      retrieval floor: even if every rewrite is bad, the original's
      rank contributions are in the fusion, so RRF cannot rank below
      baseline retrieval on this row.

    * Isolated methodology. NO reranker, NO BM25 in this row —
      only the multi-query semantic retrieval and RRF. Stacking
      with hybrid or reranker here would confound which technique
      moved the metric. That stack is reserved for `full`.

    Defensive fallback. If the rewriter returns `[]` (malformed
    response, empty response, non-429 transport error — see
    `src.query.transform.rewrite_query`), we degrade this question
    to original-only retrieval instead of failing the whole run.
    The eval row records baseline-like retrieval for the affected
    question — the correct attribution for a failed rewrite. Only
    a pool-exhausted rotation (`RuntimeError` from the eval client)
    halts the whole run.

    NB on retrieval metrics: this mode returns `query_transform_top_k`
    fused chunks (default 5), so hit_rate_at_10 collapses to
    hit_rate_at_5 for this row — only 5 items exist to inspect.
    Same degeneracy as the reranked and hybrid rows. The baseline
    anchor row retains the meaningful hit_rate_at_10 for comparison.
    """
    # 1. Rewrite the question. Rotation is handled inside
    #    `eval_client.rewrite_query`; we get either a list of
    #    rewrites or `[]` on graceful degradation.
    rewrites = eval_client.rewrite_query(question)

    # 2. Build the set of queries to send to the semantic retriever.
    #    Original stays in the pool by default (see design entry). If
    #    the config flag is False, the fusion is over rewrites only —
    #    honest but removes the safety hedge.
    queries: list[str] = []
    if config.query_transform_include_original:
        queries.append(question)
    queries.extend(rewrites)

    # If neither the original nor any rewrite is available (impossible
    # under default config, but reachable if a future config sets both
    # include_original=False AND n_rewrites=0), fall back to a single
    # original-query retrieval so the row still produces something to
    # measure — a completely empty query list would return an empty
    # top-K and the eval would attribute the miss to the pipeline mode
    # rather than the misconfiguration.
    if not queries:
        logger.warning(
            "query_transform: no queries to fuse (include_original=%s, "
            "n_rewrites=%d); falling back to original-only retrieval.",
            config.query_transform_include_original,
            config.query_transform_n_rewrites,
        )
        queries = [question]

    # 3. Semantic retrieval per query. Each list is best-first with
    #    `rank` = 1..N locally, which is what rrf_fuse expects.
    ranked_lists: list[list[RetrievalResult]] = [
        retrieve(
            q,
            top_k=config.query_transform_semantic_top_n,
            config=config,
        )
        for q in queries
    ]

    # 4. Fuse. rrf_fuse handles first-occurrence-wins on hydrated
    #    fields — safe here because every list comes from the same
    #    Chroma store (invariant documented in rrf.py). Sets `rank`
    #    and `rrf_score` on each output; leaves per-list provenance
    #    (semantic_rank, bm25_rank, etc.) alone. We do NOT decorate
    #    afterwards: for the query_transform row there is no
    #    meaningful "which rewrite did this chunk come from" field to
    #    surface, and adding rewrite_N_rank fields to RetrievalResult
    #    would not scale to N sub-queries.
    fused_chunks = rrf_fuse(
        ranked_lists=ranked_lists,
        top_k=config.query_transform_top_k,
        rrf_k=config.query_transform_rrf_k,
    )

    # 5. Grounded generation. Same rotating path as every other row.
    gen_result = eval_client.generate_answer(question, fused_chunks)

    logger.info(
        "query_transform(question=%r) → %d rewrites received, "
        "fused %d ranked lists (%d queries), returned %d top-K chunks",
        question[:80],
        len(rewrites),
        len(ranked_lists),
        len(queries),
        len(fused_chunks),
    )
    return fused_chunks, gen_result


def _run_pipeline_full(
    question: str,
    eval_client: EvalGroqClient,
    config=settings,
) -> tuple[list[RetrievalResult], Any]:
    """
    Full pipeline (headline row): hybrid retrieval (semantic + BM25, RRF
    fused) → cross-encoder rerank → grounded generation.

    Same composition order as `src.production.pipeline.run_production_query`
    — this is the stack the shipped API runs. The single deviation from
    production is the generator dispatch: this function goes through
    `eval_client.generate_answer` (rotating key pool) instead of the
    production `generate()` singleton, so a 78-question run stays under
    the Groq 8k TPM cap. The pipeline UNDER STUDY (retrieval + rerank
    selection) is byte-identical.

    Reads from `agri_schemes_prod` — the multimodal collection that
    contains text + table + image chunks. Collection swap happens at
    `run_eval()` startup for this mode only, so all other rows still
    read from the frozen `agri_schemes_rag` ablation anchor.

    Retrieval-metric semantics match the other fusion rows: this mode
    returns `reranker_top_k` chunks (default 5), so hit_rate_at_10
    collapses to hit_rate_at_5 — only 5 items exist to inspect. Same
    degeneracy as `reranked` and `hybrid`. The baseline row retains
    the meaningful hit_rate_at_10 for cross-row comparison.
    """
    fused_pool = hybrid_search(
        query=question,
        top_k=config.reranker_top_n,
        config=config,
    )
    reranked_chunks = rerank(
        query=question,
        candidates=fused_pool,
        top_k=config.reranker_top_k,
        config=config,
    )
    gen_result = eval_client.generate_answer(question, reranked_chunks)
    return reranked_chunks, gen_result


def _run_pipeline_for_mode(
    pipeline_mode: str,
    question: str,
    eval_client: EvalGroqClient,
    config=settings,
) -> tuple[list[RetrievalResult], Any]:
    """
    Route to the right pipeline mode. Phase 5 modes are added one at a time.

    Currently implemented: `baseline`, `reranked`, `hybrid`,
    `query_transform`, `full`. `crag` remains a stub.

    Isolated methodology reminder: each mode applies ONE technique on
    top of the baseline, NOT stacked on previous techniques. Only
    `full` combines everything (headline row that mirrors production).
    """
    if pipeline_mode == "baseline":
        return _run_pipeline_baseline(question, eval_client, config)
    if pipeline_mode == "reranked":
        return _run_pipeline_reranked(question, eval_client, config)
    if pipeline_mode == "hybrid":
        return _run_pipeline_hybrid(question, eval_client, config)
    if pipeline_mode == "query_transform":
        return _run_pipeline_query_transform(question, eval_client, config)
    if pipeline_mode == "full":
        return _run_pipeline_full(question, eval_client, config)
    if pipeline_mode == "crag":
        raise NotImplementedError(
            f"pipeline_mode={pipeline_mode!r} is not yet implemented."
        )
    raise ValueError(f"Unknown pipeline_mode {pipeline_mode!r}")


# --------------------------------------------------------------------- #
# Retrieval metrics — deterministic, no LLM                             #
# --------------------------------------------------------------------- #

def _normalise_filename(name: str) -> str:
    """Collapse `.pdf.pdf` typo variants and lowercase. Mirrors inspect_golden_set.py."""
    n = (name or "").strip().lower()
    while n.endswith(".pdf.pdf"):
        n = n[:-4]
    return n


def _expected_source_keys(expected_sources: list[dict]) -> list[tuple[str, str]]:
    """
    Turn the golden set's expected_sources into (kind, id) keys.

    Kind is 'pdf' or 'workflow'; anything else (e.g. the legacy
    'external_scheme_fact' type or a missing type field) is dropped —
    those don't participate in the retrieval metrics because there's
    no chunk in the collection for retrieval to hit. `out_of_scope`
    refusal questions therefore have zero expected keys and their
    hit_rate/MRR are computed against zero targets (see below).
    """
    keys: list[tuple[str, str]] = []
    for src in expected_sources or []:
        stype = src.get("type")
        if stype == "pdf":
            fn = src.get("document") or src.get("source_filename") or ""
            if fn:
                keys.append(("pdf", _normalise_filename(fn)))
        elif stype == "workflow":
            wf = src.get("workflow_id", "")
            if wf:
                keys.append(("workflow", wf))
    return keys


def _retrieved_source_keys(retrieved: list[RetrievalResult]) -> list[tuple[str, str]]:
    """Same (kind, id) shape as `_expected_source_keys`, in retrieval rank order."""
    keys: list[tuple[str, str]] = []
    for r in retrieved:
        if r.source_type == "pdf":
            keys.append(("pdf", _normalise_filename(r.source_filename)))
        elif r.source_type == "workflow" and r.workflow_id:
            keys.append(("workflow", r.workflow_id))
    return keys


def _compute_retrieval_metrics(
    retrieved: list[RetrievalResult], expected_sources: list[dict]
) -> dict[str, float | int]:
    """
    hit_rate@5, hit_rate@10, MRR — no LLM, no dependencies.

    For questions with no expected sources (out_of_scope refusal
    questions), we return `hit_rate=0` and `mrr=0.0`. That looks like
    a retrieval failure at the aggregate level, but it's the honest
    number: the ideal retrieval for an out-of-scope question is
    "nothing plausibly relevant". Averaging in-scope hit rate is what
    tells you if retrieval is working; the out-of-scope block is
    scored separately by RAGAS on refusal quality.
    """
    exp_keys = set(_expected_source_keys(expected_sources))
    ret_keys = _retrieved_source_keys(retrieved)

    if not exp_keys:
        return {"hit_rate_at_5": 0, "hit_rate_at_10": 0, "mrr": 0.0}

    hit5 = int(any(k in exp_keys for k in ret_keys[:5]))
    hit10 = int(any(k in exp_keys for k in ret_keys[:10]))
    mrr = 0.0
    for rank, k in enumerate(ret_keys[:10], start=1):
        if k in exp_keys:
            mrr = 1.0 / rank
            break
    return {"hit_rate_at_5": hit5, "hit_rate_at_10": hit10, "mrr": mrr}


# --------------------------------------------------------------------- #
# Judge / RAGAS wiring                                                  #
# --------------------------------------------------------------------- #

def _build_judge_llm(
    eval_client: EvalGroqClient | None = None,
    config=settings,
    judge_rotator: "KeyRotator | None" = None,
) -> tuple[Any, str, bool, Any | None]:
    """
    Build the LLM RAGAS will use as its judge.

    Returns
    -------
    (llm_wrapper, judge_label, is_same_as_generation, mistral_rotator)

    `mistral_rotator` is the `RotatingChatMistralAI` instance when
    `judge_provider == "mistral"` and is `None` for every other
    branch. It is exposed so the eval loop can scrape
    `.stats()` at the end of the run and record Mistral rotation
    accounting into `EvalSummary.mistral_stats`.

    Judge selection order (driven by config.judge_provider):
      - "groq"    → Groq-hosted `config.judge_model`. Independent from
                    the qwen generator, so CLAUDE.md §11 rule 1 holds.
                    `reasoning_effort` passed via extra_body from
                    `config.judge_reasoning_effort`.
      - "mistral" → Mistral `config.judge_model` (currently
                    `mistral-small-2603`) wrapped in
                    RotatingChatMistralAI. Fails LOUD if no keys are
                    configured or all keys exhaust at runtime — no
                    cross-provider fallback (§11 rule 4).
      - "ollama"  → local Ollama first. Falls through to cloud fallback
                    chain only if Ollama init fails.
      - other     → cloud fallback chain (Gemini → Mistral-static →
                    Groq same-model).

    Cloud fallback chain (evaluated in order):
      1. Gemini `gemini-3.6-flash` via `langchain_google_genai`.
         Requires GEMINI_API_KEY. temperature/top_p from config.
      2. Mistral `mistral-medium-latest` via `langchain_mistralai`.
         Requires MISTRAL_API_KEY. Independent from both generator
         and primary judge.
      3. Groq same-model (`config.llm_model` — i.e. generator ==
         judge). Recorded as loud CAVEAT — violates CLAUDE.md §11
         rule 1 but lets the run complete rather than refusing to
         eval. NB: distinct from the `judge_provider == "groq"`
         branch above, which uses a DIFFERENT Groq model as an
         independent judge.

    The chain runs ONCE at judge-init time. Whichever judge
    initialises first is the judge for the whole eval run. Mid-run
    switching is deliberately not supported — a single ablation row
    must be scored under a single judge (CLAUDE.md §11 rule 4).
    """
    from ragas.llms import LangchainLLMWrapper

    # --- 0. Primary: Groq-hosted independent judge --------------------
    # Uses config.judge_model (NOT config.llm_model) so this stays
    # independent from the generator. Fallback #3 below uses
    # config.llm_model — that is the same-model caveat path.
    #
    # `reasoning_effort` is passed via extra_body inside
    # KeyRotatingChatOpenAI. When an `eval_client` (KeyRotator wrapper)
    # is supplied the judge shares the run-level rotator so per-key
    # call accounting covers BOTH the generator and the judge; if no
    # client is supplied (utility scripts calling _build_judge_llm
    # directly) we fall back to a plain ChatOpenAI so the smoke tests
    # for the old judge path keep working.
    if config.judge_provider == "groq":
        try:
            if eval_client is not None:
                # If the caller supplied a dedicated judge rotator (split-pool
                # mode), bind the judge to THAT pool so its rate-limit
                # pressure never touches the generator's key. Otherwise fall
                # back to the eval_client's rotator (shared-pool legacy).
                groq_judge = eval_client.build_rotating_judge_llm(
                    model=config.judge_model,
                    temperature=config.judge_temperature,
                    top_p=config.judge_top_p,
                    reasoning_effort=config.judge_reasoning_effort,
                    rotator=judge_rotator,
                )
                effective_rotator = judge_rotator or eval_client._rotator
                pool_desc = (
                    f"dedicated judge pool ({effective_rotator.n_keys()} keys)"
                    if judge_rotator is not None
                    else f"shared pool ({effective_rotator.n_keys()} keys)"
                )
                label = f"groq/{config.judge_model}"
                print(
                    f"[judge] using {label} "
                    f"(reasoning_effort={config.judge_reasoning_effort}, "
                    f"{pool_desc}, "
                    f"independent from generator {config.llm_model})"
                )
            else:
                from langchain_openai import ChatOpenAI
                groq_judge = ChatOpenAI(
                    model=config.judge_model,
                    api_key=config.llm_api_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=config.judge_temperature,
                    top_p=config.judge_top_p,
                    extra_body={
                        "reasoning_effort": config.judge_reasoning_effort,
                    },
                )
                label = f"groq/{config.judge_model}"
                print(
                    f"[judge] using {label} "
                    f"(reasoning_effort={config.judge_reasoning_effort}, "
                    f"single-key, independent from generator {config.llm_model})"
                )
            return LangchainLLMWrapper(groq_judge), label, False, None
        except ImportError as exc:
            logger.warning(
                "Groq judge requested but langchain-openai not installed: %s. "
                "Trying Ollama, then cloud fallback chain.",
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Groq judge init failed: %s. Trying Ollama, then cloud "
                "fallback chain.",
                exc,
            )

    # --- 0b. Primary: Mistral (multi-key, NO cross-provider fallback) --
    # When judge_provider == "mistral", the judge is `mistral-small-2603`
    # (CLAUDE.md §11) wrapped in RotatingChatMistralAI so a pool of
    # Mistral keys can be rotated on 429/401. This branch is deliberately
    # ISOLATED from the cloud fallback chain below: if construction fails
    # or all keys are exhausted at runtime, we FAIL LOUD rather than
    # silently switch to Gemini or Groq-same-model. Silently switching
    # the judge model mid-ablation invalidates §11 rule 4 (the metric
    # delta must be attributable to the technique, not to a judge swap).
    if config.judge_provider == "mistral":
        from src.eval.mistral_rotator import RotatingChatMistralAI
        keys = config.mistral_api_keys
        if not keys:
            raise RuntimeError(
                "judge_provider='mistral' but no Mistral API keys are "
                "configured. Set MISTRAL_API_KEYS (comma-separated, "
                "preferred) or the legacy MISTRAL_API_KEY in .env. No "
                "cross-provider fallback is attempted — CLAUDE.md §11 "
                "rule 4."
            )
        # `reasoning_effort` is threaded through model_kwargs. On
        # `mistral-small-2603` only "none" and "high" are legal values
        # (config.judge_reasoning_effort default = "none"). If a future
        # Mistral model rejects the field, ChatMistralAI will return
        # HTTP 400 and RotatingChatMistralAI's _is_rotate_error will
        # (correctly) treat it as non-rotate and propagate loudly.
        mistral_wrapper = RotatingChatMistralAI(
            keys=keys,
            model=config.judge_model,
            temperature=config.judge_temperature,
            top_p=config.judge_top_p,
            model_kwargs={
                "reasoning_effort": config.judge_reasoning_effort,
            },
        )
        label = f"mistral/{config.judge_model}"
        print(
            f"[judge] using {label} "
            f"(reasoning_effort={config.judge_reasoning_effort}, "
            f"temp={config.judge_temperature}, "
            f"{mistral_wrapper.n_keys()} key(s), "
            f"independent from generator {config.llm_model}, "
            f"NO cross-provider fallback)"
        )
        return LangchainLLMWrapper(mistral_wrapper), label, False, mistral_wrapper

    # --- 1. Local: Ollama (primary when judge_provider == "ollama") ---
    if config.judge_provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            class RAGASCompatibleChatOllama(ChatOllama):
                async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                    temperature = kwargs.pop("temperature", None)

                    if temperature is not None:
                        options = kwargs.get("options")

                        if options is None:
                             options = {}

                        options = dict(options)
                        options["temperature"] = temperature
                        kwargs["options"] = options

                    return await super()._agenerate(
                        messages,
                        stop=stop,
                        run_manager=run_manager,
                        **kwargs,
                    )
            ollama = RAGASCompatibleChatOllama(
                model=config.judge_model,
                base_url=config.ollama_base_url,
                temperature=None,
                top_p=None,
                reasoning=False,
            )
            label = f"ollama/{config.judge_model}"
            print(f"[judge] using {label} (LOCAL — no cloud dep, no RPM cap)")
            print(
                "[judge] NOTE: small local judges (≤7B) are noisier — "
                "expect ±0.10-0.15 RAGAS score variance between runs. "
                "Trade-off is free, offline, unlimited-throughput eval."
            )
            return LangchainLLMWrapper(ollama), label, False, None
        except ImportError as exc:
            logger.warning(
                "Ollama judge requested but langchain-ollama not installed: %s. "
                "Trying cloud fallback chain.",
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Ollama judge init failed: %s. Is `ollama serve` running "
                "and the model tag pulled? Trying cloud fallback chain.",
                exc,
            )

    # --- 1. Primary: Gemini ---
    if config.gemini_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Gemini's model name in config is the actual judge model
            # only when judge_provider=='gemini'. If the user set
            # judge_provider to something else but still has a Gemini
            # key, we still try Gemini as primary but use the
            # documented default judge model (gemini-3.6-flash) rather
            # than a Mistral/other model name that Gemini would 404 on.
            gemini_model = (
                config.judge_model
                if config.judge_provider == "gemini"
                else "gemini-3.6-flash"
            )
            gemini = ChatGoogleGenerativeAI(
                model=gemini_model,
                google_api_key=config.gemini_api_key,
                temperature=config.judge_temperature,
                top_p=config.judge_top_p,
            )
            label = f"gemini/{gemini_model}"
            print(f"[judge] using {label} (primary, independent from generation)")
            return LangchainLLMWrapper(gemini), label, False, None
        except ImportError as exc:
            logger.warning(
                "Gemini judge requested but langchain-google-genai not installed: %s. "
                "Trying Mistral fallback.",
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Gemini judge init failed: %s. Trying Mistral fallback.",
                exc,
            )

    # --- 2. Fallback #1: Mistral ---
    if config.mistral_api_key:
        try:
            from langchain_mistralai import ChatMistralAI
            mistral_model = (
                config.judge_model
                if config.judge_provider == "mistral"
                else "mistral-medium-latest"
            )
            mistral = ChatMistralAI(
                model=mistral_model,
                mistral_api_key=config.mistral_api_key,
                temperature=config.judge_temperature,
                top_p=config.judge_top_p,
            )
            label = f"mistral/{mistral_model}"
            print(f"[judge] using {label} (FALLBACK #1 — Gemini unavailable, "
                  "still independent from generation)")
            return LangchainLLMWrapper(mistral), label, False, None
        except ImportError as exc:
            logger.warning(
                "Mistral fallback requested but langchain-mistralai not installed: %s. "
                "Falling through to Groq same-model judge.",
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Mistral judge init failed: %s. Falling through to Groq same-model judge.",
                exc,
            )

    # --- 3. Fallback #2: Groq (same as generation — caveat) ---
    from langchain_openai import ChatOpenAI
    groq_judge = ChatOpenAI(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=config.judge_temperature,
        top_p=config.judge_top_p,
    )
    label = f"groq/{config.llm_model} (FALLBACK #2 — same as generation)"
    print(f"[judge] FALLBACK: {label}")
    print(
        "[judge] WARNING: judge == generation. Faithfulness / answer_relevancy "
        "will be inflated. This will be recorded as a caveat in the results file."
    )
    return LangchainLLMWrapper(groq_judge), label, True, None


def _build_judge_embeddings():
    """
    Embeddings for RAGAS answer_relevancy / context_precision.

    We reuse the SAME sentence-transformers model we used at ingest
    (bge-small-en-v1.5). That's fine — the embeddings judge doesn't
    have to be independent from the retriever the way the LLM judge
    has to be independent from the generator; it just needs to give
    a stable semantic-similarity signal. Using our existing embedder
    also means one less API dependency (no OpenAI embeddings key).
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings

    hf = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return LangchainEmbeddingsWrapper(hf)


def _compute_ragas_metrics(
    question: str,
    answer: str,
    contexts: list[str],
    expected_answer: str,
    judge_llm: Any,
    judge_embeddings: Any,
) -> dict[str, float | None]:
    """
    Score one question with RAGAS.

    Called one question at a time — see module docstring for why we
    don't batch. Each RAGAS metric internally does 1-2 judge LLM
    calls, so a single question is 4-8 LLM calls total; the outer
    loop pacing (--sleep) sits between whole questions, not between
    RAGAS's internal calls.

    Returns four floats keyed by metric name. Any metric that fails
    (RAGAS raises, judge returns garbage, etc.) is recorded as `None`
    rather than crashing the row — see EvalQuestionResult docstring.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_list([{
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts if contexts else [""],
        "reference": expected_answer,
    }])

    metrics_out: dict[str, float | None] = {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }

    # RunConfig parameters:
    #  timeout=180      per judge call — Gemini flash is typically <5s
    #                   but reasoning-heavy metrics on long contexts
    #                   can spike; 180s absorbs the tail.
    #  max_retries=3    on 429/5xx from the judge.
    #  max_wait=30      cap on exponential backoff between retries.
    #  max_workers=1    serialise judge calls. Gemini free tier for
    #                   `gemini-3.6-flash` is capped at 5 RPM; a burst
    #                   of 4 parallel judge calls exceeds this
    #                   instantly and forces every question into
    #                   retry backoff. Serial calls (max_workers=1)
    #                   pace themselves under the RPM ceiling and
    #                   are actually FASTER end-to-end than the
    #                   4-worker burst-then-backoff pattern.
    run_config = RunConfig(timeout=600, max_retries=3, max_wait=30, max_workers=1)

    try:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=False,
        )
        # RAGAS 0.2 returns a Result object with per-metric scores.
        # `.to_pandas()` gives us one row per question; column names
        # match the metric names.
        df = result.to_pandas()
        for metric_name in metrics_out:
            if metric_name in df.columns:
                val = df.iloc[0][metric_name]
                try:
                    fval = float(val)
                    # NaN check — RAGAS returns NaN for failures with
                    # raise_exceptions=False.
                    if fval == fval:  # NaN != NaN
                        metrics_out[metric_name] = fval
                except (TypeError, ValueError):
                    pass
    except Exception as exc:
        logger.warning("RAGAS evaluate() failed for this question: %s", exc)

    return metrics_out


# --------------------------------------------------------------------- #
# Result writing                                                        #
# --------------------------------------------------------------------- #

def _write_result(payload: dict, output_path: Path) -> None:
    """Serialise the full run to disk. Called once at end of run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


# --------------------------------------------------------------------- #
# Incremental save / resume plumbing                                    #
# --------------------------------------------------------------------- #
#
# Each run has its own directory:
#
#     eval/results/{run_id}/
#         results.jsonl    append-only, one JSON object per completed question
#         progress.json    completed_question_ids + last_updated + config_snapshot
#         summary.json     final aggregate; written once at successful end
#
# results.jsonl is never rewritten. progress.json is rewritten atomically
# after each question so a crash mid-question leaves a self-consistent
# picture: either the question is in BOTH files (completed) or in
# NEITHER (about to be redone on --resume).

def _run_output_dir(run_id: str, config=settings) -> Path:
    return config.eval_results_dir / run_id


def _append_result_jsonl(result: EvalQuestionResult, run_dir: Path) -> None:
    """
    Append one completed question record to results.jsonl.

    Flushes + fsyncs so a crash immediately after this call still leaves
    the row on disk. Ordering follows the golden-set iteration order.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "results.jsonl"
    line = json.dumps(result.model_dump(), ensure_ascii=False, default=str)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            import os
            os.fsync(f.fileno())
        except OSError:  # pragma: no cover — Windows may refuse fsync on some FS
            pass


def _write_progress(
    run_dir: Path,
    completed_ids: list[str],
    config_snapshot: dict,
) -> None:
    """
    Atomic rewrite of progress.json.

    Writes to a `.tmp` sibling then renames — a crash mid-write can't
    leave a half-written progress file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    tmp_path = run_dir / "progress.json.tmp"
    payload = {
        "completed_question_ids": completed_ids,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "config_snapshot": config_snapshot,
    }
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        try:
            import os
            os.fsync(f.fileno())
        except OSError:  # pragma: no cover
            pass
    tmp_path.replace(progress_path)


def _load_progress(run_dir: Path) -> dict | None:
    """Return the parsed progress.json or None if missing."""
    progress_path = run_dir / "progress.json"
    if not progress_path.exists():
        return None
    with progress_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_results_jsonl(run_dir: Path) -> list[dict]:
    """Load every previously-completed result row. Empty list if none."""
    jsonl_path = run_dir / "results.jsonl"
    if not jsonl_path.exists():
        return []
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _config_snapshot(config=settings) -> dict:
    """
    Minimal config fingerprint written into progress.json.

    Intentionally masks the Groq keys — records only how many are
    configured, not the strings themselves.
    """
    return {
        "generation_model": f"{config.llm_provider}/{config.llm_model}",
        "generation_reasoning_effort": config.generation_reasoning_effort,
        "generation_max_tokens": config.generation_max_tokens,
        "generation_temperature": config.generation_temperature,
        "judge_provider": config.judge_provider,
        "judge_model": config.judge_model,
        "judge_reasoning_effort": config.judge_reasoning_effort,
        "embedding_model": config.embedding_model,
        "retrieval_top_k": config.retrieval_top_k,
        "top_k_retrieval": config.top_k_retrieval,
        "n_keys_configured": len(config.groq_api_keys),
        "key_rotation_cooldown_s": config.key_rotation_cooldown_s,
        "simulate_quota_exhaustion_after_n_calls":
            config.simulate_quota_exhaustion_after_n_calls,
    }


_RAGAS_METRIC_NAMES: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

# The one category on which RAGAS faithfulness / answer_relevancy are
# the wrong tool — a refusal has no claims to verify and no
# substantive content for reverse-question generation. Kept as a
# constant here rather than a magic string so grep finds every place
# the special-case fires.
_OUT_OF_SCOPE_CATEGORY: str = "out_of_scope"


def _compute_refusal_correct(answer: str) -> float:
    """
    Deterministic refusal-quality score for one out_of_scope answer.

    Returns 1.0 if the answer contains any of the refusal phrases
    listed in `config.OUT_OF_SCOPE_REFUSAL_PHRASES` (case-insensitive
    substring match), else 0.0.

    Multi-phrase detection (rather than exact sentinel match) is
    intentional — small prompt-wording drift ("not enough information"
    vs "insufficient information") must not silently break the metric.
    The phrase list is centralised in `src/config.py` so a prompt
    change only touches one place.
    """
    lo = (answer or "").lower()
    for phrase in OUT_OF_SCOPE_REFUSAL_PHRASES:
        if phrase.lower() in lo:
            return 1.0
    return 0.0


def _aggregate(
    results: list[EvalQuestionResult],
) -> tuple[
    dict[str, float | int],
    dict[str, dict[str, float | int]],
    dict[str, int],
]:
    """
    Compute overall + per-category metric averages.

    out_of_scope handling (why this function looks split-brained):
      * RAGAS faithfulness / answer_relevancy return 0.0 mechanically
        on refusal answers — a refusal has no claims to verify and no
        substantive content to reverse-generate a hypothetical question
        from. Including out_of_scope rows in the RAGAS averages would
        drag the whole ablation table down for a NON-failure. So we
        exclude `out_of_scope` from every RAGAS metric average (both
        overall and per-category) and score refusals separately with
        the deterministic `refusal_correct` metric (see
        `_compute_refusal_correct`).
      * Retrieval metrics (hit_rate_at_5/10, mrr) remain averaged over
        ALL rows in the bucket. out_of_scope questions naturally score
        0 on those because the golden set marks them as having no
        expected sources — which is the correct behaviour to record
        (nothing to retrieve, nothing hit).
      * `failed_computations` counts Nones only over the RAGAS-eligible
        subset. Counting out_of_scope Nones would trigger a scary
        "8/78 questions returned None" caveat on every run, even
        though those Nones are by design.

    None handling (explicit, not silent):
      * Every unexpected per-question None on a RAGAS metric (i.e. on
        a NON-out_of_scope row) is logged at WARNING with the
        question_id + metric name.
      * Each bucket records a `{metric}_effective_n` alongside the
        averaged value, so the denominator used for that average is
        auditable — a judge that timed out on half the run cannot
        silently inflate the row by shrinking the denominator.
    """
    ragas_eligible_results = [
        r for r in results if r.category != _OUT_OF_SCOPE_CATEGORY
    ]

    for r in ragas_eligible_results:
        for metric_name in _RAGAS_METRIC_NAMES:
            if getattr(r, metric_name) is None:
                logger.warning(
                    "RAGAS metric %r is None for question_id=%s "
                    "(category=%s) — excluded from average, counted in "
                    "failed_computations.",
                    metric_name, r.question_id, r.category,
                )

    failed_computations: dict[str, int] = {
        m: sum(1 for r in ragas_eligible_results if getattr(r, m) is None)
        for m in _RAGAS_METRIC_NAMES
    }

    def _avg_with_effective_n(vals: list[float | None]) -> tuple[float, int]:
        clean = [v for v in vals if v is not None]
        avg = round(sum(clean) / len(clean), 4) if clean else 0.0
        return avg, len(clean)

    def _bucket(rs: list[EvalQuestionResult]) -> dict[str, float | int]:
        if not rs:
            return {}
        block: dict[str, float | int] = {
            "n": len(rs),
            "hit_rate_at_5": round(sum(r.hit_rate_at_5 for r in rs) / len(rs), 4),
            "hit_rate_at_10": round(sum(r.hit_rate_at_10 for r in rs) / len(rs), 4),
            "mrr": round(sum(r.mrr for r in rs) / len(rs), 4),
        }
        # RAGAS metrics: averaged over the RAGAS-eligible rows in this
        # bucket only. An out_of_scope-only bucket contributes no rows
        # here → all four metrics land at 0.0 with effective_n=0, which
        # is what `_print_summary` uses to render "n/a" in the table.
        ragas_rows = [r for r in rs if r.category != _OUT_OF_SCOPE_CATEGORY]
        for metric_name in _RAGAS_METRIC_NAMES:
            avg, eff_n = _avg_with_effective_n(
                [getattr(r, metric_name) for r in ragas_rows]
            )
            block[metric_name] = avg
            block[f"{metric_name}_effective_n"] = eff_n
        # refusal_correct: averaged over the out_of_scope rows in this
        # bucket only. The out_of_scope per-category block gets the
        # 8-question refusal average; other categories don't populate
        # this key at all (nothing to average). The overall block gets
        # it whenever the run includes any out_of_scope questions.
        refusal_rows = [
            r for r in rs if r.category == _OUT_OF_SCOPE_CATEGORY
        ]
        if refusal_rows:
            avg, eff_n = _avg_with_effective_n(
                [r.refusal_correct for r in refusal_rows]
            )
            block["refusal_correct"] = avg
            block["refusal_correct_effective_n"] = eff_n
        return block

    overall = _bucket(results)

    per_category: dict[str, dict[str, float | int]] = {}
    categories = sorted({r.category for r in results})
    for cat in categories:
        per_category[cat] = _bucket([r for r in results if r.category == cat])

    return overall, per_category, failed_computations


def _print_summary(summary: EvalSummary) -> None:
    """Print the ablation-ready table to stdout."""
    print()
    print("=" * 88)
    print(f"EVAL SUMMARY — {summary.pipeline_mode} — n={summary.n_questions}")
    print("=" * 88)
    print(f"generation : {summary.generation_model}")
    print(f"judge      : {summary.judge_model}")
    print(f"wall time  : {summary.wall_time_s:.1f}s")
    print(f"llm calls  : {summary.total_llm_calls} "
          f"(prompt={summary.total_prompt_tokens}, "
          f"completion={summary.total_completion_tokens}, "
          f"reasoning={summary.total_reasoning_tokens})")
    print(f"retrieval failures (hit@10==0) : {summary.retrieval_failure_count}")
    # Scope of the Groq accounting depends on whether the judge is Groq.
    # If mistral_stats is populated, the Groq counters cover the
    # generator only (§EvalSummary.mistral_stats docstring); otherwise
    # they may cover both generator and Groq judge pools.
    groq_scope = "generator only" if summary.mistral_stats else "generator+judge"
    print(f"groq keys ({groq_scope})      : {summary.n_keys_configured}")
    print(f"groq key rotations   : {summary.key_rotations}")
    print(f"groq cooldown events : {summary.cooldown_events}")
    print(f"groq rate-limit errs : {summary.rate_limit_errors}")
    print(f"groq per-key calls   : {summary.per_key_call_counts}")

    if summary.mistral_stats:
        ms = summary.mistral_stats
        print(
            f"mistral keys (judge) : {ms.get('n_keys_configured', 0)}"
        )
        print(f"mistral key rotations: {ms.get('key_rotations', 0)}")
        print(f"mistral rate-limit errs: {ms.get('rate_limit_errors', 0)}")
        # Rotation history is verbose — collapse to a count in the
        # printed table. The full list is preserved in summary.json.
        n_hist = len(ms.get('rotation_history', []))
        print(f"mistral rotation events logged: {n_hist}")
    if summary.caveats:
        print()
        print("CAVEATS:")
        for c in summary.caveats:
            print(f"  ! {c}")
    print()

    metric_cols = [
        "n", "hit_rate_at_5", "hit_rate_at_10", "mrr",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    ]
    header = f"{'category':<28} " + " ".join(f"{m:>16}" for m in metric_cols)
    print(header)
    print("-" * len(header))
    for cat, block in summary.per_category.items():
        # out_of_scope is the special row: RAGAS metrics don't apply,
        # so we render refusal_correct in the faithfulness column with
        # a '*' marker and dash the other RAGAS columns. A footer note
        # below explains the substitution.
        if cat == _OUT_OF_SCOPE_CATEGORY:
            cells = []
            for m in metric_cols:
                if m in ("n",):
                    cells.append(f"{int(block.get(m, 0)):>16}")
                elif m in ("hit_rate_at_5", "hit_rate_at_10", "mrr"):
                    cells.append(f"{block.get(m, 0):>16.4f}")
                elif m == "faithfulness":
                    refusal = float(block.get("refusal_correct", 0.0))
                    cells.append(f"{('*' + f'{refusal:.4f}'):>16}")
                else:
                    cells.append(f"{'n/a':>16}")
            print(f"{cat:<28} " + " ".join(cells))
            continue
        row = f"{cat:<28} " + " ".join(
            f"{block.get(m, 0):>16}" if isinstance(block.get(m, 0), int)
            else f"{block.get(m, 0):>16.4f}"
            for m in metric_cols
        )
        print(row)
    print("-" * len(header))
    row = f"{'OVERALL':<28} " + " ".join(
        f"{summary.overall.get(m, 0):>16}" if isinstance(summary.overall.get(m, 0), int)
        else f"{summary.overall.get(m, 0):>16.4f}"
        for m in metric_cols
    )
    print(row)

    # Refusal metric belongs to a different sample than the RAGAS
    # metrics (8 out_of_scope questions vs 70 in-scope), so it gets
    # its own line under OVERALL rather than a misleading extra column.
    if "refusal_correct" in summary.overall:
        oos_n = summary.refusal_metric_n_questions
        refusal = summary.overall.get("refusal_correct", 0.0)
        print(f"{'OOS refusal_correct':<28} " + " " * (16 * 3 + 3) +
              f"{'n=' + str(oos_n):>16} {float(refusal):>16.4f}")

    print()
    if any(cat == _OUT_OF_SCOPE_CATEGORY for cat in summary.per_category):
        print(
            "* out_of_scope uses refusal_correct metric (deterministic) "
            "instead of RAGAS faithfulness. Excluded from RAGAS overall "
            "averages."
        )
    # Sample-size provenance line — makes the 70-vs-78 split explicit
    # so a reader doesn't derive it themselves from per_category.
    print(
        f"[samples] total={summary.total_n_questions}  "
        f"RAGAS-avg-n={summary.ragas_metrics_n_questions}  "
        f"refusal-avg-n={summary.refusal_metric_n_questions}"
    )
    print()


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Agriculture Schemes RAG eval harness against the golden set."
    )
    p.add_argument(
        "--n_questions",
        type=int,
        default=20,
        help="Number of golden questions to run. Default: 20 (fast iteration).",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Shortcut for the full golden set. Overrides --n_questions.",
    )
    p.add_argument(
        "--pipeline_mode",
        type=str,
        default="baseline",
        choices=["baseline", "hybrid", "reranked", "query_transform", "crag", "full"],
        help="Which pipeline to evaluate. Only 'baseline' is implemented in Phase 4.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Seconds to sleep between questions. Default 3s (safe for Groq 8k TPM).",
    )
    p.add_argument(
        "--skip_ragas",
        action="store_true",
        help=(
            "Skip RAGAS metric computation — still runs pipeline + retrieval metrics. "
            "Useful when the judge model isn't set up yet or you're debugging the pipeline."
        ),
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help=(
            "Resume a partial run by id (e.g. 20260911-164500_baseline_n5). "
            "Loads progress.json from eval/results/{RUN_ID}/ and skips any "
            "question already present in completed_question_ids. If every "
            "question is already complete the summary is written and the "
            "run exits without re-executing anything."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete an existing run directory before starting. Only takes "
            "effect for NEW runs — meaningless with --resume. Refuses to "
            "run when the derived run_id already has data unless this flag "
            "is set explicitly."
        ),
    )
    return p.parse_args()


def run_eval(args: argparse.Namespace) -> EvalSummary:
    """The full eval run, top to bottom. Returns the summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Sentence-transformers and chromadb are chatty at INFO; quiet them.
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # --- Collection routing --------------------------------------------
    # Every Phase-5 ablation row reads from the frozen `agri_schemes_rag`
    # anchor collection (text-only). The `full` row is the exception: it
    # is the headline number that mirrors the shipped production stack,
    # so it must read from the multimodal `agri_schemes_prod` collection
    # (text + table + image chunks). Swap is done BEFORE any retriever
    # module is called — retriever / bm25 / reranker cache a chromadb
    # PersistentClient on first use and pin the collection name at that
    # point, so a later mutation would be a no-op (see retriever.py
    # comment). Kept SCOPED to `full` mode so the other ablation rows
    # stay reproducible against the anchor collection.
    if args.pipeline_mode == "full":
        prod_collection = settings.production_collection_name
        anchor_collection = settings.chroma_collection_name
        prod_persist = settings.production_persist_dir
        anchor_persist = settings.chroma_persist_dir
        print(
            f"[collection] pipeline_mode=full → switching collection "
            f"{anchor_collection!r} → {prod_collection!r} "
            f"(multimodal prod stack)"
        )
        print(
            f"[collection] persist dir "
            f"{anchor_persist} → {prod_persist}"
        )
        settings.chroma_collection_name = prod_collection
        settings.chroma_persist_dir = prod_persist

    n = 0 if args.full else args.n_questions
    questions = _load_golden_set(settings.eval_golden_set_path, n)

    # --- Resume vs new-run resolution ---------------------------------
    # Two paths converge here:
    #   * resuming a prior run  → load its progress.json + results.jsonl,
    #     use the SAME run_id, use the SAME n_questions the run started
    #     with (args.n_questions is ignored for correctness on resume).
    #   * new run               → fresh timestamped run_id; refuse if
    #     the derived directory already exists unless --overwrite.
    completed_ids: list[str] = []
    prior_results: list[EvalQuestionResult] = []

    if args.resume:
        run_id = args.resume
        run_dir = _run_output_dir(run_id)
        progress = _load_progress(run_dir)
        if progress is None:
            raise SystemExit(
                f"--resume {run_id}: no progress.json at {run_dir}. "
                "Nothing to resume."
            )
        completed_ids = list(progress.get("completed_question_ids", []))
        prior_rows = _load_results_jsonl(run_dir)
        # Reconstruct EvalQuestionResult from stored rows so the final
        # aggregate covers previously-completed questions.
        prior_results = [EvalQuestionResult(**row) for row in prior_rows]

        # Rebuild the question list from the snapshot's n_questions when
        # possible, so a caller that types `--n_questions 25` after a
        # `--n_questions 5` partial doesn't silently pick a different
        # slice. If snapshot missing, fall back to args.
        snap = progress.get("config_snapshot", {}) or {}
        # We stored config_snapshot but not questions_slice_size —
        # infer from completed count + remaining if all rows present.
        # Simplest: keep the CLI-supplied `n` and trust the caller.
        # (Documented in the resume log line below.)
        print(f"[resume] run_id={run_id}")
        print(f"[resume] completed: {len(completed_ids)}/{len(questions)}")
        if completed_ids:
            missing_from_jsonl = set(completed_ids) - {r.question_id for r in prior_results}
            if missing_from_jsonl:
                logger.warning(
                    "progress.json lists %d ids missing from results.jsonl: %s",
                    len(missing_from_jsonl), sorted(missing_from_jsonl),
                )
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{ts}_{args.pipeline_mode}_n{len(questions)}"
        run_dir = _run_output_dir(run_id)
        if run_dir.exists() and any(run_dir.iterdir()):
            if args.overwrite:
                # Explicit overwrite: nuke the directory contents so the
                # new run starts clean. Rare enough that we log loudly.
                import shutil
                logger.warning(
                    "--overwrite: deleting existing run directory %s", run_dir
                )
                shutil.rmtree(run_dir)
            else:
                raise SystemExit(
                    f"Run directory {run_dir} is not empty and neither "
                    "--resume nor --overwrite was supplied. Refusing to "
                    "clobber a partial run. Rerun with `--resume "
                    f"{run_id}` to continue, or `--overwrite` to discard."
                )

    # --- Set up rotators (SPLIT-POOL) + eval client -------------------
    # Pool-split rationale: the judge (`openai/gpt-oss-120b`) is ~20×
    # more token-heavy per question than the generator (small model).
    # When both share one key pool, a judge burst can burn every key's
    # per-minute TPM window and the next generator call 429s on a
    # "fresh" rotation. Splitting the pools isolates the two workloads
    # so generator throughput never depends on judge quota state.
    #
    # Layout:
    #   GROQ_API_KEY   (settings.llm_api_key)   → generator (1 key)
    #   GROQ_API_KEYS  (settings.groq_api_keys) → judge (N keys, disjoint)
    #
    # Falls back to a shared-pool single rotator if the plural list is
    # empty or collapses to the same single key as GROQ_API_KEY — that
    # matches the pre-split behaviour so a single-key .env still runs.
    gen_key = (settings.llm_api_key or "").strip()
    judge_keys_raw = settings.groq_api_keys
    # Drop the generator key from the judge pool if it appears there —
    # the whole point of the split is disjoint pools.
    judge_keys = [k for k in judge_keys_raw if k and k != gen_key]

    if not gen_key and not judge_keys:
        raise SystemExit(
            "No Groq API keys configured. Set GROQ_API_KEY (generator) "
            "and GROQ_API_KEYS (comma-separated, judge pool) in .env."
        )

    # The Groq judge pool is only meaningful when the RAGAS judge is
    # actually a Groq model. With judge_provider="mistral" (or ollama /
    # gemini) the "judge_keys" from GROQ_API_KEYS never get called — so
    # building a KeyRotator over them would report misleading zero-call
    # buckets in the run summary. Skip that branch entirely for
    # non-Groq judges.
    judge_is_groq = settings.judge_provider == "groq"

    if judge_is_groq and gen_key and judge_keys:
        gen_rotator = KeyRotator(
            keys=[gen_key],
            cooldown_s=settings.key_rotation_cooldown_s,
            simulate_exhaustion_after=settings.simulate_quota_exhaustion_after_n_calls,
        )
        judge_rotator = KeyRotator(
            keys=judge_keys,
            cooldown_s=settings.key_rotation_cooldown_s,
            simulate_exhaustion_after=settings.simulate_quota_exhaustion_after_n_calls,
        )
        pool_mode = "split"
        print(f"[keys] SPLIT-POOL: generator=1 key (GROQ_API_KEY), "
              f"judge={len(judge_keys)} keys (GROQ_API_KEYS, disjoint from generator)")
    elif not judge_is_groq:
        # Judge is on another provider — only build the generator pool.
        # GROQ_API_KEYS + GROQ_API_KEY are merged into a single generator
        # pool (dedup preserved order); "judge_keys" and the singular
        # gen_key are the same address space when judge is not Groq.
        gen_pool = [gen_key] if gen_key else []
        for k in judge_keys:
            if k not in gen_pool:
                gen_pool.append(k)
        if not gen_pool:
            raise SystemExit(
                "No Groq API keys configured. Set GROQ_API_KEY and/or "
                "GROQ_API_KEYS in .env (used by the qwen generator)."
            )
        gen_rotator = KeyRotator(
            keys=gen_pool,
            cooldown_s=settings.key_rotation_cooldown_s,
            simulate_exhaustion_after=settings.simulate_quota_exhaustion_after_n_calls,
        )
        judge_rotator = None
        pool_mode = "generator_only"
        print(f"[keys] GENERATOR-ONLY (judge is {settings.judge_provider}, "
              f"not Groq): generator pool = {len(gen_pool)} Groq key(s). "
              f"Judge keys are managed by the judge provider's own rotator.")
    else:
        # Judge is Groq but only one Groq pool is set — legacy shared-pool
        # fallback, generator and judge share the same rotator.
        shared_keys = [gen_key] if gen_key else judge_keys
        gen_rotator = KeyRotator(
            keys=shared_keys,
            cooldown_s=settings.key_rotation_cooldown_s,
            simulate_exhaustion_after=settings.simulate_quota_exhaustion_after_n_calls,
        )
        judge_rotator = gen_rotator
        pool_mode = "shared"
        print(f"[keys] SHARED-POOL fallback: {len(shared_keys)} key(s), "
              f"generator and judge share the same pool. "
              f"Set both GROQ_API_KEY and GROQ_API_KEYS to enable split-pool mode.")

    eval_client = EvalGroqClient(rotator=gen_rotator)

    print(f"[run] id={run_id}")
    print(f"[run] pipeline_mode={args.pipeline_mode}")
    print(f"[run] n_questions={len(questions)} "
          f"({'FULL' if args.full else 'subset'} of golden set)")
    print(f"[run] sleep_between_questions={args.sleep}s")
    print(f"[run] generation model = {settings.llm_provider} / {settings.llm_model}")
    if pool_mode == "generator_only":
        print(f"[run] groq keys — generator pool = {gen_rotator.n_keys()} "
              f"(mode={pool_mode}; judge is on {settings.judge_provider}, "
              f"managed separately)")
    else:
        print(f"[run] groq keys — generator pool = {gen_rotator.n_keys()}, "
              f"judge pool = {judge_rotator.n_keys()} (mode={pool_mode})")
    if settings.simulate_quota_exhaustion_after_n_calls is not None:
        print(f"[run] SIMULATED quota exhaustion after "
              f"{settings.simulate_quota_exhaustion_after_n_calls} calls per key")
    print()

    # --- Judge setup ---------------------------------------------------
    judge_llm: Any = None
    judge_embeddings: Any = None
    judge_label = "disabled"
    judge_is_same_as_generation = False
    caveats: list[str] = []

    # Populated by _build_judge_llm's Mistral branch; None for every
    # other judge provider. Held so we can scrape .stats() at end of
    # run and emit EvalSummary.mistral_stats.
    mistral_judge_rotator: Any | None = None

    if not args.skip_ragas:
        try:
            (
                judge_llm,
                judge_label,
                judge_is_same_as_generation,
                mistral_judge_rotator,
            ) = _build_judge_llm(
                eval_client=eval_client,
                judge_rotator=judge_rotator if pool_mode == "split" else None,
            )
            judge_embeddings = _build_judge_embeddings()
        except Exception as exc:
            logger.error(
                "Judge setup failed: %s. Continuing with retrieval metrics only.",
                exc,
            )
            traceback.print_exc()
            judge_llm = None
            judge_label = f"unavailable: {exc}"
            caveats.append(
                f"RAGAS judge unavailable ({exc}). RAGAS metrics are all None."
            )
    else:
        judge_label = "disabled (--skip_ragas)"
        caveats.append("RAGAS skipped via --skip_ragas; RAGAS metrics are all None.")

    if judge_is_same_as_generation:
        caveats.append(
            "Judge model == generation model (Groq fallback). Faithfulness "
            "and answer_relevancy are self-scored and likely inflated. "
            "This violates CLAUDE.md §11 rule 1 — the ablation table row "
            "produced by this run cannot be mixed with rows that used an "
            "independent Gemini judge."
        )

    # --- Filter out already-completed questions on resume -------------
    remaining_questions: list[dict] = []
    for q in questions:
        qid = q.get("question_id", "")
        if qid in completed_ids:
            continue
        remaining_questions.append(q)

    if args.resume and remaining_questions:
        # Log the resume point BEFORE running anything.
        next_qid = remaining_questions[0].get("question_id", "?")
        print(
            f"Resuming run_id={run_id}. "
            f"Completed: {len(completed_ids)}/{len(questions)}. "
            f"Continuing from {next_qid}."
        )
    elif args.resume and not remaining_questions:
        print(
            f"Resuming run_id={run_id}. "
            f"All {len(completed_ids)}/{len(questions)} questions already "
            f"completed — no work to do. Writing summary and exiting."
        )

    # --- Per-question loop --------------------------------------------
    results: list[EvalQuestionResult] = list(prior_results)
    llm_call_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    # Fold in prior audit-trail totals so the summary reflects the whole
    # run, not just the resumed slice.
    for r in prior_results:
        total_prompt_tokens += r.prompt_tokens
        total_completion_tokens += r.completion_tokens
        total_reasoning_tokens += r.reasoning_tokens

    t0 = time.perf_counter()
    snapshot = _config_snapshot()

    for i, q in enumerate(remaining_questions, start=len(completed_ids) + 1):
        qid = q.get("question_id", f"Q?{i}")
        category = q.get("category", "<unknown>")
        question = q.get("question", "")
        expected_answer = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])

        preview = question[:70] + ("..." if len(question) > 70 else "")
        print(f"  [{i:>2}/{len(questions)}] {qid} [{category}] {preview}")

        error_msg = ""
        try:
            retrieved_pool, gen_result = _run_pipeline_for_mode(
                args.pipeline_mode, question, eval_client
            )
        except NotImplementedError:
            raise
        except Exception as exc:
            logger.error("Pipeline failed on %s: %s", qid, exc)
            traceback.print_exc()
            error_msg = f"pipeline: {exc}"
            retrieved_pool = []
            gen_result = None

        llm_call_count += 1
        if gen_result is not None:
            total_prompt_tokens += gen_result.prompt_tokens
            total_completion_tokens += gen_result.completion_tokens
            total_reasoning_tokens += gen_result.reasoning_tokens

        retrieval_metrics = _compute_retrieval_metrics(retrieved_pool, expected_sources)

        ragas_metrics: dict[str, float | None] = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }
        if judge_llm is not None and gen_result is not None:
            contexts = [c.text for c in gen_result.retrieved_chunks]
            ragas_metrics = _compute_ragas_metrics(
                question=question,
                answer=gen_result.answer,
                contexts=contexts,
                expected_answer=expected_answer,
                judge_llm=judge_llm,
                judge_embeddings=judge_embeddings,
            )
            llm_call_count += 4

        # Deterministic refusal check — only meaningful on the
        # out_of_scope bucket, None everywhere else. Computed even when
        # RAGAS is skipped, so `--skip_ragas` runs still get the
        # refusal_correct headline number.
        refusal_correct: float | None = None
        if category == _OUT_OF_SCOPE_CATEGORY:
            refusal_correct = _compute_refusal_correct(
                gen_result.answer if gen_result else ""
            )

        row = EvalQuestionResult(
            question_id=qid,
            category=category,
            question=question,
            expected_answer=expected_answer,
            expected_sources=expected_sources,
            retrieved_chunk_ids=[c.chunk_id for c in retrieved_pool[:10]],
            retrieved_similarity_scores=[c.similarity_score for c in retrieved_pool[:10]],
            retrieved_source_filenames=[c.source_filename or None for c in retrieved_pool[:10]],
            retrieved_workflow_ids=[c.workflow_id for c in retrieved_pool[:10]],
            generated_answer=(gen_result.answer if gen_result else ""),
            faithfulness=ragas_metrics["faithfulness"],
            answer_relevancy=ragas_metrics["answer_relevancy"],
            context_precision=ragas_metrics["context_precision"],
            context_recall=ragas_metrics["context_recall"],
            refusal_correct=refusal_correct,
            hit_rate_at_5=int(retrieval_metrics["hit_rate_at_5"]),
            hit_rate_at_10=int(retrieval_metrics["hit_rate_at_10"]),
            mrr=float(retrieval_metrics["mrr"]),
            latency_ms=(gen_result.latency_ms if gen_result else 0),
            prompt_tokens=(gen_result.prompt_tokens if gen_result else 0),
            completion_tokens=(gen_result.completion_tokens if gen_result else 0),
            reasoning_tokens=(gen_result.reasoning_tokens if gen_result else 0),
            retries_taken=(gen_result.retries_taken if gen_result else 0),
            finish_reason=(gen_result.finish_reason if gen_result else ""),
            error=error_msg,
        )
        results.append(row)

        # --- INCREMENTAL PERSISTENCE (crash-safe) ---------------------
        # results.jsonl is append-only; progress.json is rewritten
        # atomically. Both are flushed+fsynced before the next question
        # starts. A crash between here and the next iteration leaves the
        # completed question fully persisted.
        _append_result_jsonl(row, run_dir)
        completed_ids.append(qid)
        _write_progress(run_dir, completed_ids, snapshot)

        # Pace between questions. Same rationale as before rework.
        if i < len(questions):
            time.sleep(args.sleep)

    wall_time = time.perf_counter() - t0

    # --- Aggregate + write final summary ------------------------------
    overall, per_category, failed_computations = _aggregate(results)
    retrieval_failures = sum(1 for r in results if r.hit_rate_at_10 == 0)

    # Surface metric-computation failures as a run-level caveat so the
    # ablation-table reader sees "this row's faithfulness average was
    # taken over only 34/40 questions" without having to grep the log.
    # Skipped when --skip_ragas is on — every metric is None *by design*
    # in that mode, and the single "RAGAS skipped" caveat already
    # explains it. Firing per-metric "judge failure / RAGAS internal
    # error" caveats on top of --skip_ragas would be actively
    # misleading.
    # Denominator here is the RAGAS-eligible subset (non-out_of_scope),
    # matching what `failed_computations` now counts.
    n_ragas_eligible = sum(
        1 for r in results if r.category != _OUT_OF_SCOPE_CATEGORY
    )
    if not args.skip_ragas:
        for metric_name, n_failed in failed_computations.items():
            if n_failed > 0:
                caveats.append(
                    f"{metric_name}: {n_failed}/{n_ragas_eligible} in-scope "
                    f"questions returned None (judge failure / RAGAS internal "
                    f"error). Average is over the remaining "
                    f"{n_ragas_eligible - n_failed} successful computations; "
                    f"see per-bucket `{metric_name}_effective_n`."
                )

    gen_stats = gen_rotator.stats()
    judge_stats = judge_rotator.stats() if pool_mode == "split" else None
    # Only populated when judge_provider == "mistral" — the wrapper
    # instance came back from _build_judge_llm's Mistral branch. All
    # other judges leave this as {}. Kept separate from the Groq
    # accounting (see EvalSummary.mistral_stats docstring).
    mistral_stats = (
        mistral_judge_rotator.stats() if mistral_judge_rotator is not None else {}
    )

    # Merge both pools' stats into the existing EvalSummary shape.
    # Per-key call counts get prefixed (`gen_key_0`, `judge_key_0`, ...)
    # so the split is auditable without a schema change. Aggregate
    # counters sum both pools' contributions.
    if judge_stats is not None:
        per_key_call_counts = {
            **{f"gen_{k}": v for k, v in gen_stats["per_key_call_counts"].items()},
            **{f"judge_{k}": v for k, v in judge_stats["per_key_call_counts"].items()},
        }
        n_keys_configured = gen_stats["n_keys_configured"] + judge_stats["n_keys_configured"]
        key_rotations = gen_stats["key_rotations"] + judge_stats["key_rotations"]
        cooldown_events = gen_stats["cooldown_events"] + judge_stats["cooldown_events"]
        rate_limit_errors = gen_stats["rate_limit_errors"] + judge_stats["rate_limit_errors"]
        rotation_history = (
            [{**e, "pool": "generator"} for e in gen_stats["rotation_history"]]
            + [{**e, "pool": "judge"} for e in judge_stats["rotation_history"]]
        )
        caveats.append(
            f"Split-pool mode active: generator pool has "
            f"{gen_stats['n_keys_configured']} key(s), judge pool has "
            f"{judge_stats['n_keys_configured']} key(s). per_key_call_counts "
            "keys are prefixed `gen_`/`judge_`."
        )
    else:
        per_key_call_counts = gen_stats["per_key_call_counts"]
        n_keys_configured = gen_stats["n_keys_configured"]
        key_rotations = gen_stats["key_rotations"]
        cooldown_events = gen_stats["cooldown_events"]
        rate_limit_errors = gen_stats["rate_limit_errors"]
        rotation_history = gen_stats["rotation_history"]

    total_n = len(results)
    ragas_n = sum(1 for r in results if r.category != _OUT_OF_SCOPE_CATEGORY)
    refusal_n = sum(1 for r in results if r.category == _OUT_OF_SCOPE_CATEGORY)

    summary = EvalSummary(
        run_id=run_id,
        pipeline_mode=args.pipeline_mode,
        generation_model=f"{settings.llm_provider}/{settings.llm_model}",
        judge_model=judge_label,
        judge_is_same_as_generation=judge_is_same_as_generation,
        n_questions=total_n,
        wall_time_s=round(wall_time, 2),
        total_n_questions=total_n,
        ragas_metrics_n_questions=ragas_n,
        refusal_metric_n_questions=refusal_n,
        overall=overall,
        per_category=per_category,
        total_llm_calls=llm_call_count,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_reasoning_tokens=total_reasoning_tokens,
        retrieval_failure_count=retrieval_failures,
        caveats=caveats,
        failed_computations=failed_computations,
        n_keys_configured=n_keys_configured,
        key_rotations=key_rotations,
        per_key_call_counts=per_key_call_counts,
        cooldown_events=cooldown_events,
        rate_limit_errors=rate_limit_errors,
        mistral_stats=mistral_stats,
    )

    summary_path = run_dir / "summary.json"
    _write_result(
        {
            "summary": summary.model_dump(),
            "rotation_history": rotation_history,
        },
        summary_path,
    )
    print(f"\n[write] {summary_path}")
    _print_summary(summary)
    return summary


def main() -> None:
    args = _parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
