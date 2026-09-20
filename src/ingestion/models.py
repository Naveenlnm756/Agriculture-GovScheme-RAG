"""
Pydantic models for the loaded corpus.

These are the shapes every downstream stage (chunker, embedder, retriever)
will consume. Keeping them in one file means the "what does a loaded
document look like" question has exactly one place to answer.

Design notes:
- Workflow steps are kept as a list of dicts rather than a typed row
  model, because per-scheme CSVs may carry extra columns and the loader
  should stay tolerant of that. The workflow-layer schema in scope.md §3
  gives the minimum expected columns.
- `filepath` is a `Path`, not a `str`, so callers can pass it straight
  to `open()` / `PdfReader` without conversion churn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class PDFPage(BaseModel):
    """One page of a loaded PDF: 1-based page number and its extracted text."""

    page_num: int
    text: str


class LoadedPDFDocument(BaseModel):
    """
    One PDF ingested by the loader.

    Produced only for files the loader actually reads — after the
    `_OCR.pdf` sibling preference rule has been applied, so if this
    document has an OCR sibling the *sibling* is what appears here
    (with `is_ocr_version=True`), not the scanned original.
    """

    scheme: str
    filename: str
    filepath: Path
    is_ocr_version: bool
    # Only set when the PDF sits below an extra folder inside 01_RAW_PDFs/
    # (PM_KISAN's PM-KISAN and PM_KMY subfolders). None for the six flat
    # schemes. Preserving it lets retrieval distinguish PM-KISAN prose
    # from PM-KMY prose without renaming files.
    parent_subfolder: str | None
    pages: list[PDFPage]
    total_pages: int
    total_chars: int


class LoadedWorkflow(BaseModel):
    """
    One workflow CSV ingested by the loader.

    Workflow-level metadata (`workflow_id`, `source_title`, `source_url`)
    is taken from the first row — the workflow CSV schema (scope.md §3,
    CLAUDE.md §7) is designed so these columns repeat identically across
    all rows within a single workflow file.
    """

    scheme: str
    workflow_id: str
    filename: str
    filepath: Path
    steps: list[dict[str, str]]
    source_title: str
    source_url: str
    step_count: int


class LoadedCorpus(BaseModel):
    """
    Everything the loader produces in one bag.

    `summary` is intentionally a plain dict rather than a nested model —
    its only consumer is the runner script (and the reader eyeballing
    output). Downstream typed code works from `pdfs` and `workflows`.
    """

    pdfs: list[LoadedPDFDocument]
    workflows: list[LoadedWorkflow]
    summary: dict


class Chunk(BaseModel):
    """
    One retrieval-ready chunk of text plus the metadata needed for a
    faithful citation later.

    Source variants (source_type):
      * `pdf`      — text or table content extracted from a PDF page.
                     page_start/page_end set; workflow_id + image_*
                     fields None.
      * `workflow` — one entire workflow CSV rendered as prose.
                     workflow_id set; page + image_* fields None.
      * `image`    — a Gemini-derived structured description of a
                     visual (table-as-image, infographic, chart)
                     embedded in a PDF. Wrapped in
                     `[IMAGE]...[/IMAGE]` sentinels. Populates
                     image-provenance fields (see below).

    The `source_type` field is authoritative — do not infer modality
    from which optional fields are populated.
    """

    chunk_id: str
    text: str
    scheme: str
    source_type: Literal["pdf", "workflow", "image"]
    source_filename: str
    source_filepath: Path
    # Carried from the loader (PDFs): True when the ingested version is
    # an _OCR sibling of a scanned original. Answer-time it lets a
    # citation say "this text came from an OCR pass, not the native PDF."
    # Always False for workflows. Always False for images (vision output
    # is not OCR).
    is_ocr_source: bool
    # Only set for PDFs that sit below an extra folder inside 01_RAW_PDFs/
    # (PM_KISAN's PM-KISAN / PM_KMY split). None otherwise.
    parent_subfolder: str | None = None
    # PDF- and image-sourced. Inclusive 1-based page range this chunk
    # comes from. For image chunks this is the single page the image
    # occurrence sits on (page_start == page_end).
    page_start: int | None = None
    page_end: int | None = None
    # Workflow-only. The workflow_id this chunk represents in full.
    workflow_id: str | None = None

    # --- Image-source-only fields (all None for pdf / workflow) --------
    # PDF-internal object number of the embedded raster image. Two
    # occurrences of the same visual on different pages share the same
    # xref within a document, which is how the within-doc dedup works.
    image_xref: int | None = None
    # SHA-256 (16-hex-char prefix) of the raw image bytes. The
    # cross-document / cross-run cache key. Two image chunks with the
    # same content_hash share their vision-extracted text; the metadata
    # differs only in page/xref/scheme.
    content_hash: str | None = None
    # What the vision classifier called this image on stage 1.
    # `TABLE`, `CHART`, `INFOGRAPHIC`, `OTHER`, `DECORATIVE`, or
    # `vision_failed` when both stage-1 attempts (normal + low-DPI
    # retry) returned empty/invalid. DECORATIVE chunks are NOT
    # emitted (dropped upstream), so this field never carries that
    # value on a materialised Chunk — kept in the union for schema
    # completeness so a reader isn't surprised by future policy.
    vision_content_type: str | None = None
    # True when the pipeline had to fall back after two vision
    # attempts failed. The chunk text is a placeholder pointing at
    # the PDF page; downstream retrieval treats it as a very-low-
    # priority citation candidate. Never silently dropped
    # (per production reliability rule 12.7).
    vision_failed: bool = False


class ChunkedCorpus(BaseModel):
    """
    Everything the chunker produces.

    Same shape as LoadedCorpus: typed list + free-form summary dict. The
    summary is what the runner script prints so the reader can eyeball
    the split for problems (a scheme with too few chunks, an unexpected
    surge of suspiciously short chunks, etc.).
    """

    chunks: list[Chunk]
    summary: dict


class GenerationResult(BaseModel):
    """
    One end-to-end grounded answer, plus the audit trail for it.

    `retrieved_chunks` is deliberately carried through — the whole
    honesty story of a RAG system is "here's the answer AND here's
    exactly which passages the model was allowed to look at." Without
    this field a low-faithfulness eval score becomes impossible to
    root-cause.
    """

    query: str
    answer: str
    retrieved_chunks: list["RetrievalResult"]
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    # Reasoning-model accounting: gpt-oss-20b on Groq emits reasoning
    # tokens through a channel separate from `answer` but counted in
    # `completion_tokens`. Surface it explicitly so a silent
    # reasoning-budget exhaustion is auditable.
    reasoning_tokens: int
    latency_ms: int
    finish_reason: str
    # How many 429-driven retries the Groq call needed. Zero on a happy
    # path; a persistently non-zero value across runs means we are
    # under-configured on `llm_call_delay_s` and burning API quota on
    # avoidable backoffs.
    retries_taken: int


class EmbeddingFailure(BaseModel):
    """One chunk the embedder could not process, with the reason why."""

    chunk_id: str
    reason: str


class RetrievalResult(BaseModel):
    """
    One retrieved chunk, shaped for downstream generation / display.

    Carries every metadata field a citation-grounded answer needs, plus
    a `similarity_score` (higher = better) and 1-indexed `rank`. The
    score is deliberately not the raw Chroma distance — see retriever.py
    for the reasoning.

    Reranker fields (Phase 5, fix #1):
      * `similarity_score` is ALWAYS the semantic-retriever score and is
        preserved through reranking, so downstream code can still see
        the pre-rerank confidence.
      * `rank` is the CURRENT rank (1-indexed). After reranking, `rank`
        is the new position in the reranked list; `original_rank` is
        the position the chunk had in the semantic-only result.
      * `rerank_score` is the cross-encoder relevance score
        (higher = better). None on baseline (non-reranked) results,
        which is how downstream code can tell a chunk went through the
        reranker.
      * `original_rank` is None on baseline results (there is no prior
        rank to preserve); populated by the reranker with the pre-rerank
        semantic-retriever rank.

    Hybrid fields (Phase 5, fix #2):
      * `similarity_score` is now `float | None`. It stays populated on
        baseline and reranked results (semantic always ran) and on hybrid
        results whose chunk was in the semantic top-N pool. It is `None`
        only on hybrid results whose chunk was surfaced by BM25 but was
        NOT in the semantic top-N pool — those chunks have no cosine
        similarity to report, and reporting 0.0 would falsely imply
        "orthogonal to query" instead of "we never asked."
      * `semantic_rank` is the chunk's 1-indexed rank in the semantic
        top-N pool. `None` if the chunk was surfaced by BM25 only.
      * `bm25_rank` is the chunk's 1-indexed rank in the BM25 top-N pool.
        `None` if the chunk was surfaced by semantic only.
      * `bm25_score` is the raw BM25 relevance score (higher = better).
        `None` when the chunk was outside the BM25 pool.
      * `rrf_score` is the fused Reciprocal Rank Fusion score
        (higher = better); populated on every hybrid result. `None` on
        baseline / reranked results (which never fused pools).
      * `rank` is the CURRENT rank. On a hybrid result, `rank` is the
        post-RRF-fusion rank; `semantic_rank` and `bm25_rank` preserve
        the source-pool audit trail.
    """

    chunk_id: str
    text: str
    scheme: str
    source_type: str
    source_filename: str
    source_filepath: str
    is_ocr_source: bool
    parent_subfolder: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    workflow_id: str | None = None
    similarity_score: float | None = None
    rank: int
    original_rank: int | None = None
    rerank_score: float | None = None
    semantic_rank: int | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None


class EvalQuestionResult(BaseModel):
    """
    One golden-set question's end-to-end eval record.

    Everything a reviewer or the ablation table needs to reason about a
    single (question, pipeline) run is here — the retrieved chunk ids and
    scores, the generated answer, both retrieval-side (hit_rate / MRR)
    and generation-side (RAGAS) metrics, and the audit fields (latency,
    tokens, retries, errors). Aggregation happens in `EvalSummary`; this
    model is the row-level truth.

    RAGAS metric fields are `float | None` because a judge failure on
    one metric (e.g. RAGAS raising on an empty context) must not
    poison the row — we record `None` and the aggregator averages over
    non-None values. Same reasoning for `error`: we don't crash the run
    on one bad row; we log the failure and keep going.
    """

    question_id: str
    category: str
    question: str
    expected_answer: str
    expected_sources: list[dict]

    retrieved_chunk_ids: list[str]
    # `float | None` per-entry: hybrid rows may carry `None` for chunks
    # that were surfaced by BM25 but were outside the semantic top-N
    # pool (see `RetrievalResult.similarity_score`). Baseline and
    # reranked rows never emit `None` here.
    retrieved_similarity_scores: list[float | None]
    retrieved_source_filenames: list[str | None]
    retrieved_workflow_ids: list[str | None]

    generated_answer: str

    # RAGAS metrics — None means the judge failed on that metric for
    # this row. See class docstring for why we don't crash. For
    # out_of_scope questions these are intentionally left None (RAGAS
    # can't score a refusal — see `refusal_correct` below).
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    # Deterministic refusal-quality metric — set on `out_of_scope`
    # questions ONLY, left None on all other categories.
    # 1.0 = generator refused correctly (any of the phrases in
    # config.OUT_OF_SCOPE_REFUSAL_PHRASES appears in the answer);
    # 0.0 = generator produced a substantive answer instead of
    # refusing. Deterministic — no LLM judge, no timeouts. This is
    # what replaces RAGAS faithfulness/answer_relevancy on the
    # out_of_scope bucket where those metrics are the wrong tool
    # (a refusal has no claims to verify and no substantive text
    # to reverse-generate a hypothetical question from).
    refusal_correct: float | None = None

    # Retrieval-side metrics computed directly (no LLM judge).
    hit_rate_at_5: int
    hit_rate_at_10: int
    mrr: float

    # Audit trail.
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    retries_taken: int
    finish_reason: str

    # Non-empty when something went sideways for this question but the
    # run was allowed to continue. Empty string on the happy path.
    error: str = ""


class EvalSummary(BaseModel):
    """
    Aggregated results across all questions in one eval run.

    The "ablation table row" data structure: one row per (config,
    generation model, judge model) triple. The `per_category` block is
    the axis Phase 5 techniques most need to move — reranking should
    help `multi_hop_scheme` and `late_content_diagnostic` more than
    `simple_factual`, and the ablation table won't show that if we
    only report overall averages.
    """

    run_id: str
    pipeline_mode: str
    generation_model: str
    judge_model: str
    judge_is_same_as_generation: bool

    n_questions: int
    wall_time_s: float

    # --- Sample-size provenance (out_of_scope split) --------------------
    # RAGAS metric averages are computed over NON-out_of_scope questions
    # only, because RAGAS mechanically scores refusals 0.0 (see
    # `EvalQuestionResult.refusal_correct` and DECISIONS.md). These three
    # fields make the sample split auditable at the summary level so a
    # reader doesn't have to derive it from `per_category["out_of_scope"].n`:
    #
    #   total_n_questions          = every question the run executed (e.g. 78)
    #   ragas_metrics_n_questions  = non-out_of_scope subset  (e.g. 70)
    #   refusal_metric_n_questions = out_of_scope subset      (e.g. 8)
    #
    # Retrieval metrics (hit_rate_at_5/10, mrr) remain averaged over
    # `total_n_questions` — out_of_scope naturally scores 0 there, which
    # is correct behaviour (nothing to retrieve).
    total_n_questions: int = 0
    ragas_metrics_n_questions: int = 0
    refusal_metric_n_questions: int = 0

    # Overall averages. RAGAS metrics are over `ragas_metrics_n_questions`;
    # `refusal_correct` (when present) is over `refusal_metric_n_questions`;
    # retrieval metrics are over `total_n_questions`.
    overall: dict[str, float | int]
    # Same shape as `overall`, keyed by golden-set category. The
    # `out_of_scope` block reports `refusal_correct` in place of RAGAS
    # metrics; other categories are unchanged.
    per_category: dict[str, dict[str, float | int]]

    # Cost/quota accounting for the whole run.
    total_llm_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_reasoning_tokens: int

    # How many questions retrieval outright failed on (hit_rate@10 == 0).
    # A useful sanity number: if this is high on the baseline, most of
    # the metric floor is "we couldn't find the passage", which
    # reranking alone won't fix — hybrid or query rewrite will.
    retrieval_failure_count: int

    # Caveats we want visible in the results file itself, not buried
    # in a log. Populated when e.g. the judge fell back to the same
    # provider as generation (violates CLAUDE.md §11 held-constant rule).
    caveats: list[str] = []

    # Per-metric count of questions where a RAGAS metric returned None
    # (judge timeout, RAGAS internal failure, empty-context path, etc.).
    # Averages in `overall` / `per_category` are computed over successful
    # rows only — the *effective* sample size for each averaged RAGAS
    # metric is exposed inside every bucket as `{metric}_effective_n`
    # (e.g. `faithfulness_effective_n`). Prevents silent denominator
    # shrinkage from making an ablation row look better than it is.
    failed_computations: dict[str, int] = {}

    # --- Groq key-rotation accounting (evaluation-only) -----------------
    # Populated from KeyRotator.stats() at the end of the run. Everything
    # here is MASKED — `per_key_call_counts` maps `key_0` / `key_1` / ...
    # to integer call counts, never raw key strings. See
    # src/utils/key_rotator.py for the source of these numbers.
    #
    # Scope depends on the configured judge:
    #   - judge_provider == "groq"   → these fields sum BOTH the generator
    #                                  and Groq judge pools (split-pool
    #                                  mode); per_key_call_counts keys are
    #                                  prefixed `gen_` / `judge_`.
    #   - judge_provider != "groq"   → these fields cover the Groq
    #                                  GENERATOR only (there is no Groq
    #                                  judge pool to sum). Mistral judge
    #                                  accounting lives in `mistral_stats`
    #                                  below.
    n_keys_configured: int = 0
    key_rotations: int = 0
    per_key_call_counts: dict[str, int] = {}
    cooldown_events: int = 0
    # Real Groq/OpenAI 429s that reached our layer (i.e. the underlying
    # SDK's own retries were exhausted). Distinct from `key_rotations`
    # because a simulated rotation also increments rotations but not
    # this counter — useful for telling real rate-limit pressure from
    # smoke-test simulation in results.jsonl.
    rate_limit_errors: int = 0

    # --- Mistral judge key-rotation accounting (evaluation-only) -------
    # Populated from RotatingChatMistralAI.stats() when
    # judge_provider == "mistral". Empty dict when the judge is on any
    # other provider. All identifiers are MASKED (`mistral_key_0`, ...);
    # raw keys never enter this field. Kept as a separate block from
    # the Groq accounting above rather than being merged into
    # `per_key_call_counts`, because merging would (a) conflate
    # per-minute-reset Groq TPM pressure with per-minute-reset Mistral
    # request-cap pressure — two independent quotas that only look the
    # same on paper — and (b) require every downstream ablation-table
    # reader to know which prefix belongs to which provider. Keeping
    # them separate makes each provider's rotation story defensible
    # on its own.
    mistral_stats: dict = {}


class EmbeddingSummary(BaseModel):
    """
    Everything the embedder produces.

    Not a typed list of vectors — the vectors live in Chroma, not in
    Python memory. This summary is the audit trail: what model was
    used, how many chunks landed in the collection, and any that
    failed. Print it after a run and you should be able to answer
    "is the vector store in a defensible state?" without opening Chroma.
    """

    model_name: str
    embedding_dimension: int
    collection_name: str
    total_chunks_processed: int
    total_upserted: int
    # Verified by asking Chroma for `collection.count()` after upsert.
    # If this diverges from `total_upserted` on a fresh DB, something
    # went wrong with id determinism (duplicate ids collapsing into one).
    collection_size_after: int
    per_scheme_upserted: dict[str, int]
    # Chunks whose char length exceeded the model's context window at
    # embed time. sentence-transformers silently truncates these; we
    # surface them so the truncation is auditable, not invisible.
    truncated_chunk_count: int
    truncated_chunks: list[str]  # chunk_ids only, to keep the summary small
    failures: list[EmbeddingFailure]
