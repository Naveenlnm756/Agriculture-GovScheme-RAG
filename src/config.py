"""
Central configuration for the Agriculture Schemes RAG project.

Every knob the pipeline can be tuned by — paths, model names, top-k values,
fusion weights, and the ablation feature flags — lives here. If you find a
magic number anywhere else in the codebase, move it into this file.

We use pydantic-settings so any of these values can be overridden by an
environment variable (via .env) without editing code. That is what makes
the same pipeline runnable under many configurations for the ablation
study: flip a flag in .env, rerun the eval, record the delta.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Paths -----------------------------------------------------------
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    # Chroma persistence lives under data/ so all build artefacts live in
    # one place. It is a build artefact, not source — .gitignore excludes it.
    # Overridable at deploy time via CHROMA_PERSIST_DIR env var (case-
    # insensitive per pydantic-settings config). In production deployment
    # this is set to `/data/chroma_prod` so the app reads from the
    # persistent /data volume, not from a path baked into the image.
    chroma_persist_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "chroma_db",
        validation_alias="CHROMA_PERSIST_DIR",
    )
    eval_results_dir: Path = PROJECT_ROOT / "eval" / "results"

    # SQLite database for chat / conversation persistence. Kept COMPLETELY
    # separate from the Chroma vector store (which holds corpus knowledge)
    # — this DB holds only user-authored conversations and their messages.
    # Overridable via AGRI_DB_PATH env var. In production deployment
    # this is `/data/agri_rag.db` so it lives on the persistent /data
    # volume next to the Chroma files.
    agri_db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "agri_rag.db",
        validation_alias="AGRI_DB_PATH",
    )

    # --- Vector store ----------------------------------------------------
    # Overridable via CHROMA_COLLECTION_NAME env var. Production
    # deployments set this to `agri_schemes_prod` so the running
    # container reads the production collection without the code path
    # caring which env it's in.
    chroma_collection_name: str = Field(
        default="agri_schemes_rag",
        validation_alias="CHROMA_COLLECTION_NAME",
    )
    # How many chunks to embed and upsert per batch. Embedding 10k+ chunks
    # in a single call would hold every vector in memory at once; batching
    # keeps peak RAM bounded and lets us log progress mid-run.
    embed_batch_size: int = 64

    # --- Models ----------------------------------------------------------
    # bge-small-en-v1.5: 384-dim, punches well above its weight on the
    # MTEB retrieval benchmark at the same vector size and compute cost as
    # all-MiniLM-L6-v2. Sensible default for a portfolio project.
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # The industry-standard small cross-encoder. Cheap, well-known,
    # easy to name and defend in an interview.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM used only for the final answer-generation step.
    # Provider, model, and API-key reference all live here so switching
    # providers or models is a single-point change (see project
    # generation-provider rules).
    #
    # Locked as of 2026-09-13 to `openai/gpt-oss-20b`. Reverted from
    # `qwen/qwen3.6-27b` (2026-09-11 → 2026-09-13) because qwen's Groq
    # free-tier OTPM cap of 1000 output-tokens-per-minute is enforced as
    # a PER-REQUEST ceiling, and our `generation_max_tokens=1200` request
    # was rejected on the first call with HTTP 429 (`Requested 1200 >
    # Limit 1000`). Key rotation cannot fix a per-request cap. Rather
    # than shrink `max_tokens` below the qwen smoke-test truncation
    # floor of 800 (PMFBY hailstorm query hit `finish_reason=length`),
    # we swap back to gpt-oss-20b, which uses the shared 8k TPM pool
    # and does not enforce a separate OTPM per-request squeeze.
    #
    # Reason this switch is safe RIGHT NOW: eval/results/ contains only
    # baseline rows (no use_hybrid / use_reranker rows yet), so the
    # ablation table hasn't started. Cost of the swap = redo the
    # baseline once. If the ablation had begun, the locked evaluation
    # rules would forbid this switch mid-table.
    #
    # gpt-oss-20b properties worth remembering:
    #   - Reasoning channel is REAL. Reasoning tokens are billed against
    #     the completion budget but do not appear in the visible answer.
    #     Set `generation_reasoning_effort="low"` (below) — "none" is
    #     rejected with HTTP 400 by gpt-oss-*.
    #   - Grounding/refusal behaviour was the baseline before qwen; the
    #     qwen switch was made for verbosity, not correctness. Reverting
    #     does not reintroduce a known grounding regression.
    # gpt-oss-120b remains reserved for the final headline eval only.
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-20b"
    # Previously: qwen/qwen3.6-27b (2026-09-11 → 2026-09-13, reverted per OTPM 429)
    # Previously reserved: openai/gpt-oss-120b (headline eval only)
    # Read from GROQ_API_KEY in .env. The alias is the only provider-specific
    # string in this file; swap it (and llm_provider / llm_model above) to
    # change providers — no other code touches the env var name.
    #
    # Kept for the PRODUCTION generator path (src/generation/generator.py),
    # which is deliberately single-key. Evaluation-time code uses
    # `groq_api_keys` below and rotates through them via KeyRotator.
    llm_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")

    # --- Evaluation-only: multi-key Groq rotation ------------------------
    # Comma-separated list of authorised Groq API keys, used sequentially
    # (never in parallel) by the eval harness to spread the 8k TPM cap
    # across several free-tier accounts. Order is preserved. One key
    # behaves exactly like the previous single-key config. If the plural
    # var is unset, we fall back to the single `llm_api_key` above so an
    # existing .env still works.
    #
    # Loaded as a raw string here and parsed into a list via the
    # `groq_api_keys_list` property below — pydantic-settings doesn't
    # split delimited env vars for us on this pydantic v1/v2 boundary.
    #
    # SECURITY: never log or serialise this field. Every downstream user
    # (KeyRotator, EvalGroqClient) treats keys as opaque handles and
    # exposes only masked identifiers (`key_0`, `key_1`, ...).
    groq_api_keys_raw: str = Field(
        default="", validation_alias="GROQ_API_KEYS"
    )

    # Seconds to sleep after every configured Groq key has been rotated
    # away from in a single cycle. On wake, the rotator resets to
    # `key_0` and zeroes per-key counters. Never invoked implicitly by
    # the rotator — the eval client calls `enter_cooldown()` explicitly
    # after catching `AllKeysExhaustedError`.
    #
    # Default is 60s — matches Groq's actual per-minute TPM reset
    # window. Was 3600s (1 hour) when the eval used Groq for both
    # generation AND judging, because the shared 8k-TPM budget really
    # could stay pinned for extended periods. Since the judge moved to
    # Mistral (2026-09-13, DECISIONS.md), generation has Groq's full
    # per-minute budget to itself, and a 429 is almost always a
    # single-minute TPM overrun that recovers within seconds. For
    # multi-key pools the 60s value is still correct — pool size just
    # multiplies effective throughput before the first cooldown fires.
    # Override via KEY_ROTATION_COOLDOWN_S in .env if a longer sleep is
    # needed (e.g. daily-token quota hit rather than per-minute TPM).
    key_rotation_cooldown_s: int = Field(
        default=60, validation_alias="KEY_ROTATION_COOLDOWN_S"
    )

    # Deterministic-test hook ONLY. When set to an integer N, the
    # KeyRotator treats the (N+1)-th call to the active key as a
    # simulated rate-limit condition (rotates instead of hitting the
    # network). Real Groq calls remain untouched — we do not attempt to
    # simulate the response payload itself, only the exhaustion signal
    # that triggers rotation. Leave None in production and for the
    # locked ablation table.
    simulate_quota_exhaustion_after_n_calls: int | None = Field(
        default=None, validation_alias="SIMULATE_QUOTA_EXHAUSTION_AFTER_N_CALLS"
    )

    @property
    def groq_api_keys(self) -> list[str]:
        """
        Parsed, ordered, whitespace-stripped list of Groq API keys.

        Falls back to the single `llm_api_key` when `GROQ_API_KEYS` is
        empty so a legacy single-key .env still works. Empty entries
        (double commas, trailing commas) are dropped. Duplicates are
        NOT deduplicated — the smoke tests deliberately seed duplicate
        entries so rotation mechanics can be exercised without needing
        genuinely distinct authorised keys.
        """
        raw = (self.groq_api_keys_raw or "").strip()
        if raw:
            parts = [p.strip() for p in raw.split(",")]
            keys = [p for p in parts if p]
            if keys:
                return keys
        single = (self.llm_api_key or "").strip()
        return [single] if single else []

    # --- Chunking --------------------------------------------------------
    # Baseline (fixed-size) values. When
    # `use_structure_aware_chunking=True` these still govern the
    # non-table windower — tables are always kept whole and never
    # touched by the size cap.
    chunk_size: int = 800          # characters
    chunk_overlap: int = 120

    # --- Vision (Deliverable 2 B5) ---------------------------------------
    # Provider + model for the image-ingestion vision pipeline. Owner-
    # locked spec: `gemini-2.5-flash-lite` — but Google's own API
    # returns 404 for new users on 2.5-flash-lite and points at
    # 3.5-flash-lite. The default here is the working replacement; the
    # 2.5-flash-lite deviation is documented in every response cache
    # entry and in DECISIONS.md.
    #
    # Swap is a single-point change: adjust `vision_provider` +
    # `vision_model` here and the adapter factory in
    # `src/vision/adapter.py` will resolve the concrete implementation.
    vision_provider: str = "gemini"
    vision_model: str = "gemini-3.5-flash-lite"
    # Retry policy on empty/invalid vision responses. See adapter for
    # semantics — attempt 1 renders at `vision_default_dpi`, attempt 2
    # renders at `vision_fallback_dpi` (lower). Beyond that, the chunk
    # is marked `vision_failed=True` (never silently dropped).
    vision_max_attempts: int = 2
    vision_default_dpi: int = 150
    vision_fallback_dpi: int = 96
    # Seconds to sleep between vision calls. gemini-3.5-flash-lite free
    # tier caps at 15 RPM; 5s keeps us at 12 RPM with headroom.
    vision_call_delay_s: float = 5.0
    # On-disk cache directory keyed by content-hash. One JSON sidecar
    # per unique visual — reused across ingestion runs so we never
    # re-call vision on the same image bytes.
    vision_cache_dir: Path = PROJECT_ROOT / "data" / "vision_cache"
    # Chroma collection for production. Image chunks land here alongside
    # text and table chunks. Kept separate from the ablation-anchor
    # `agri_schemes_rag` collection so re-running the ablation stays
    # reproducible (held-constant ablation rule).
    production_collection_name: str = "agri_schemes_prod"
    # On-disk persist dir for the production Chroma store. Kept separate
    # from `chroma_persist_dir` (which points at the ablation-anchor
    # store under `data/chroma_db/`) because the prod collection was
    # built by `scripts/build_prod_collection.py` into `data/chroma_prod/`.
    # In deployment this is overridden by `CHROMA_PERSIST_DIR=/data/chroma_prod`
    # via the API startup, which mutates `chroma_persist_dir` directly;
    # the eval harness reads THIS field so the same routing works
    # locally without needing an env-var swap.
    production_persist_dir: Path = PROJECT_ROOT / "data" / "chroma_prod"

    # Phase 5, fix #1 (ablation methodology). When True, PDFs are chunked
    # with the pymupdf-based structure-aware pass:
    #   - pymupdf.find_tables() detects tables on each page.
    #   - Every detected table is extracted whole into a single chunk
    #     with NO size cap (a table that renders to >1500 chars still
    #     lives in one chunk — the point of the fix is that a table
    #     split in half is worse than a table too big for the model's
    #     usual context slot).
    #   - Non-table text on each page is identified by taking the
    #     block-list from `page.get_text("blocks")` and dropping any
    #     block whose bbox center sits inside a table bbox.
    #   - Non-table text is concatenated across pages with the same
    #     `[PAGE {n}]` markers the baseline uses, then windowed with
    #     the same fixed 800/120 slider above.
    #
    # OFF by default so the baseline v1 chunking (locked ablation
    # anchor) is exactly reproducible. Flip to True for baseline v2
    # and every downstream Phase-5 row that follows.
    #
    # Design decisions worth pinning here:
    #   1. Heading detection is DELIBERATELY not implemented. The
    #      heading A/B (scripts/heading_ab_*) showed low precision on
    #      the corpus; it is documented as tested-and-skipped rather
    #      than latent code. If we ever want to revisit it, add a
    #      separate flag; do not overload this one.
    #   2. Chunk ids gain a small suffix change under this mode:
    #      text windows use `_c{idx:04d}` (unchanged), tables use
    #      `_tbl{idx:04d}` so a reader browsing Chroma can tell them
    #      apart without opening the chunk text.
    #   3. On PDFs that pymupdf refuses to open, or on pages where
    #      find_tables() raises, we fall back to the baseline behaviour
    #      for that document (whole-page text → windower). Failures are
    #      recorded in the chunker summary so a silent regression is
    #      auditable.
    use_structure_aware_chunking: bool = False

    # --- Retrieval -------------------------------------------------------
    # Baseline (pure semantic) retriever returns this many chunks. This
    # is the number the LLM sees when no rerank stage is active. Kept
    # separate from top_k_retrieval below (which is the *candidate pool*
    # size for hybrid + rerank) so that switching ablation flags on and
    # off doesn't quietly change how many chunks the generator receives.
    retrieval_top_k: int = 5
    top_k_retrieval: int = 20      # fetched from vector / hybrid search (candidate pool for rerank)
    top_k_rerank: int = 5          # kept after cross-encoder rerank
    hybrid_alpha: float = 0.5      # 1.0 = pure semantic, 0.0 = pure BM25

    # --- Reranker (Phase 5, fix #1) --------------------------------------
    # Dedicated knobs for the `reranked` pipeline mode. Deliberately kept
    # SEPARATE from `top_k_retrieval` / `retrieval_top_k` above (which
    # govern the baseline pipeline) so a change to the reranker's
    # candidate-pool size can never silently shift the baseline row.
    # Numerically identical to top_k_retrieval / top_k_rerank today, but
    # the two configs must be independently switchable so Phase 5
    # ablation rows stay isolated per project design rules (config in
    # one place, no magic numbers scattered).
    #
    # reranker_top_n : how many candidates the semantic retriever fetches
    #                  for the reranker to re-score. Larger = more chances
    #                  the correct chunk is in the pool, but slower
    #                  (cross-encoder is O(n) per query). 20 is the
    #                  interview-defensible default; the ms-marco-MiniLM
    #                  model handles 20 pairs in <500ms on CPU.
    # reranker_top_k : how many chunks the reranker returns to the
    #                  generator. Matches baseline `retrieval_top_k=5`
    #                  so the LLM sees the same context depth in both
    #                  baseline and reranked runs — the only variable
    #                  under study in the `reranked` row is the SELECTION
    #                  of those 5, not their count.
    reranker_top_n: int = 20
    reranker_top_k: int = 5

    # --- Hybrid retrieval (Phase 5, fix #2) ------------------------------
    # Dedicated knobs for the `hybrid` pipeline mode. Kept SEPARATE from
    # `top_k_retrieval` / `retrieval_top_k` (baseline) and
    # `reranker_top_n` / `reranker_top_k` (reranker) so a change to
    # hybrid pool sizes can never silently shift another ablation row.
    # Numerically identical to the reranker knobs today; they must stay
    # independently switchable per project design rules.
    #
    # hybrid_semantic_top_n : candidates fetched from the semantic
    #                         retriever for the fusion pool.
    # hybrid_bm25_top_n     : candidates fetched from BM25 for the fusion
    #                         pool. Same size as the semantic pool by
    #                         default — asymmetric pool sizes would tilt
    #                         the RRF fusion toward one retriever without
    #                         being visible in the fusion constant.
    # hybrid_top_k          : how many fused chunks the generator sees.
    #                         Matches baseline `retrieval_top_k=5` so the
    #                         LLM sees the same context depth across
    #                         baseline / reranked / hybrid — the only
    #                         variable this ablation row studies is the
    #                         SELECTION of the 5, not their count.
    # hybrid_rrf_k          : the Reciprocal Rank Fusion constant. The
    #                         canonical value from the original TREC RRF
    #                         paper is 60 — it makes the fusion mildly
    #                         insensitive to shallow-rank fluctuations
    #                         (differences among ranks 1-5 matter more
    #                         than differences among ranks 15-20).
    #                         Kept at the paper's default so the row is
    #                         defensible without tuning; a value chosen
    #                         by cross-validating on the golden set
    #                         would overfit the ablation.
    hybrid_semantic_top_n: int = 20
    hybrid_bm25_top_n: int = 20
    hybrid_top_k: int = 5
    hybrid_rrf_k: int = 60

    # --- Query transformation (Phase 5, fix #3) --------------------------
    # Dedicated knobs for the `query_transform` pipeline mode. Kept
    # SEPARATE from every other row's knobs (baseline / reranker /
    # hybrid) so a change to query-transform pool sizes or fusion
    # constant can never silently shift another ablation row. Same
    # discipline as the rest of the config (config in one place, no
    # magic numbers, ablation rows independently switchable).
    #
    # Strategy: multi-query expansion. The generator LLM (same model
    # as generation — see llm_model above) rewrites the user question
    # into N alternate phrasings; each rewrite plus the original is
    # sent to the semantic retriever; the N+1 ranked lists are fused
    # via RRF into a single top-K for the generator. See DECISIONS.md
    # (2026-09-14 fix #3 design decisions entry) for why multi-query
    # was chosen over sub-question decomposition and HyDE, and why
    # retrieval-time fusion was chosen over generation-time fusion.
    #
    # query_transform_n_rewrites : how many alternate phrasings the
    #                              LLM generates. 3 is the RAG-Fusion
    #                              / Query2Doc sweet spot; 5+ starts
    #                              producing near-duplicates that
    #                              don't retrieve new chunks. Kept as
    #                              a single knob so a future ablation
    #                              row (or a diagnostic) can flip N
    #                              without touching the prompt.
    # query_transform_include_original : whether the ORIGINAL query
    #                              contributes its own ranked list to
    #                              the fusion, alongside the rewrites.
    #                              Default True — provides a strict
    #                              retrieval floor: even if all N
    #                              rewrites are bad, the original's
    #                              rank votes are still in the pool,
    #                              so the fused top-K can never do
    #                              WORSE than baseline on retrieval.
    #                              Turning it off is the honest way
    #                              to measure the rewrites in
    #                              isolation, but that is not the
    #                              default because it removes the
    #                              safety hedge.
    # query_transform_semantic_top_n : how many candidates the
    #                              semantic retriever fetches PER
    #                              rewrite (and per original). Kept
    #                              identical to hybrid_semantic_top_n
    #                              so the per-query recall depth is
    #                              constant across the two fusion-
    #                              based rows; only the source of
    #                              the extra list(s) differs
    #                              (BM25 in hybrid, LLM-rewrites in
    #                              query_transform).
    # query_transform_top_k      : how many fused chunks the
    #                              generator sees. Matches baseline
    #                              retrieval_top_k=5 so the LLM sees
    #                              the same context depth across
    #                              every ablation row — the only
    #                              variable under study is the
    #                              SELECTION of the 5, not the count.
    # query_transform_rrf_k      : the Reciprocal Rank Fusion
    #                              constant applied when merging the
    #                              N+1 ranked lists. Kept at 60 (TREC
    #                              RRF paper default) for the same
    #                              reason hybrid_rrf_k = 60: any
    #                              value tuned on the golden set
    #                              would overfit the ablation.
    #                              Numerically identical to
    #                              hybrid_rrf_k today, but MUST stay
    #                              independently switchable — the two
    #                              rows fuse different signals.
    query_transform_n_rewrites: int = 3
    query_transform_include_original: bool = True
    query_transform_semantic_top_n: int = 20
    query_transform_top_k: int = 5
    query_transform_rrf_k: int = 60

    # Rewriter-call decoding parameters. Held constant across every
    # row of the ablation table for the same reason the generator's
    # decoding is (locked evaluation rules): a change here confounds
    # the technique-delta measurement.
    #
    # query_transform_max_tokens : output budget for the rewriter
    #                              call. Rewrites are short JSON
    #                              (three one-line paraphrases plus
    #                              structural tokens); 400 is a
    #                              generous ceiling that never
    #                              truncates a well-formed response
    #                              but caps a runaway one. Includes
    #                              gpt-oss-20b's invisible reasoning
    #                              channel — see llm_model docstring.
    # query_transform_temperature : 0.0. Deterministic rewrites so
    #                              the same golden question produces
    #                              the same rewrites run-to-run,
    #                              which is required for the
    #                              ablation row to be reproducible.
    # query_transform_reasoning_effort : "low" — matches
    #                              generation_reasoning_effort so
    #                              both LLM calls in this row use
    #                              the same reasoning depth. gpt-oss-*
    #                              rejects "none" with HTTP 400; if
    #                              the generator is ever swapped
    #                              back to qwen, this must be reset
    #                              to "none" or "default".
    query_transform_max_tokens: int = 400
    query_transform_temperature: float = 0.0
    query_transform_reasoning_effort: str = "low"

    # --- Ablation feature flags -----------------------------------------
    # Toggling these is exactly what the ablation study measures.
    # Baseline run = all false.
    use_hybrid: bool = False
    use_reranker: bool = False
    use_query_transform: bool = False
    use_crag: bool = False

    # --- Generation ------------------------------------------------------
    # 1200-token completion cap. NOTE for gpt-oss-20b: this budget is
    # SHARED between the invisible reasoning channel and the visible
    # answer (see llm_model docstring above). With
    # `generation_reasoning_effort="low"` reasoning is typically a few
    # hundred tokens, leaving ~700-900 for the visible answer — enough
    # for the multi-part procedural questions (e.g. PMFBY hailstorm)
    # that hit `finish_reason=length` at 800 under qwen. If eval shows
    # answers being truncated (finish_reason=length with a short
    # visible payload) the fix is to raise this cap, not to lower
    # reasoning_effort — reasoning_effort is held constant across the
    # ablation for the same reason the model is.
    generation_max_tokens: int = 1200
    generation_temperature: float = 0.0   # deterministic for eval

    # openai/gpt-oss-* accepts the "low" / "medium" / "high" scale for
    # reasoning_effort. Passing "none" here is rejected with HTTP 400 —
    # that scale is qwen's, not gpt-oss's. "low" is the deliberate
    # baseline choice: keeps the model "deliberately dumb" so Phase 5
    # interventions (CRAG, query decomposition) can demonstrate
    # measurable value against a minimal-reasoning floor rather than a
    # hidden self-CoT baseline.
    #
    # Value is HELD CONSTANT across every row of the ablation table
    # (locked evaluation rules). Changing it mid-table would confound the
    # technique-delta measurement — you could no longer tell whether a
    # metric moved because of the technique or because reasoning depth
    # changed. If we ever swap generator back to qwen, this must be
    # reset to "none" or "default" (or the request will 400).
    generation_reasoning_effort: str = "low"

    # --- Evaluation ------------------------------------------------------
    # Path to the hand-built golden question set (built in Phase 3).
    eval_golden_set_path: Path = PROJECT_ROOT / "eval" / "golden_set.json"

    # How many golden questions to actually run on each eval pass.
    # 0 = the entire set. Use a small number (~25) for fast iteration
    # during ablation; use 0 for the final headline run.
    # Overridable at the command line via `--size`.
    eval_subset_size: int = 25

    # --- LLM rate limiting ----------------------------------------------
    # Groq free tier caps openai/gpt-oss-20b at 8,000 tokens/min. Sleep
    # this many seconds between successive LLM calls to stay under it.
    # Applies to generation calls AND to RAGAS judge calls.
    llm_call_delay_s: float = 2.0

    # On HTTP 429, retry with exponential backoff.
    llm_max_retries: int = 5
    llm_backoff_base_s: float = 2.0   # first retry sleep; doubles each attempt

    # --- RAGAS judge model ----------------------------------------------
    # The judge is the LLM RAGAS uses to score faithfulness / answer
    # relevancy / context precision / context recall. It is kept as
    # SEPARATE config from the generation model for two reasons:
    #
    #  (a) Independence. Using the same model as both generator and
    #      judge introduces a self-consistency bias: the judge will
    #      systematically over-rate its own generation style,
    #      inflating faithfulness and answer_relevancy scores. RAGAS
    #      papers report this as ~10 point inflation on faithfulness.
    #      An independent judge grades an answer the same way a
    #      human reviewer with no stake in the generation would.
    #
    #  (b) Quota. Judge calls are 4-6x more frequent than generation
    #      calls (one per metric per question). Pointing the judge at
    #      a different provider (e.g. Gemini) preserves the Groq
    #      quota for generation and lets a 78-question eval actually
    #      complete on free tiers.
    #
    # HARD RULE (locked evaluation rules): the judge model MUST stay
    # constant across every row of a single ablation table. Never swap
    # mid-project — doing so invalidates the comparison (a metric
    # movement could be caused by the technique OR by the judge
    # changing its mind, and you can no longer tell which).
    #
    # Judge selection at runtime (implemented in eval/run_eval.py's
    # `_build_judge_llm`). Selection is driven by `judge_provider`:
    #   - "groq"   → Groq-hosted judge, model tag from `judge_model`
    #                (e.g. `openai/gpt-oss-120b`). Independent from the
    #                generator (qwen ≠ gpt-oss), so the judge-
    #                independence rule is satisfied. Uses
    #                `langchain_openai.ChatOpenAI` pointed at Groq's
    #                OpenAI-compatible endpoint, with
    #                `reasoning_effort="low"` passed via extra_body to
    #                keep judge calls fast and TPM-cheap. Kept as a
    #                fallback — not the current default.
    #   - "ollama" → local Ollama endpoint. No API key, no RPM cap,
    #                no cloud dependency. Model tag is `judge_model`
    #                (e.g. "qwen3:4b"), endpoint is `ollama_base_url`.
    #                Uses `langchain_ollama.ChatOllama`. Small models
    #                (≤7B) are noisier judges — RAGAS scores can shift
    #                ±0.15 between runs on the same question. Kept as
    #                an offline/no-quota escape hatch.
    #   - "gemini" → Gemini `gemini-3.6-flash` via `langchain_google_genai`.
    #                Cloud, rate-limited (5 RPM free tier for 3.6-flash),
    #                but a strong judge model. Requires GEMINI_API_KEY.
    #                Falls back to Mistral, then Groq if unavailable.
    #   - "mistral" → Mistral judge via `langchain_mistralai`, wrapped
    #                in `RotatingChatMistralAI` for multi-key rotation.
    #                Model tag comes from `judge_model` (currently
    #                `mistral-small-2603`, Mistral Small v26.03
    #                released 2026-03-16). Requires either
    #                MISTRAL_API_KEYS (plural, comma-separated,
    #                preferred) or the legacy singular MISTRAL_API_KEY.
    #                On 429/401 the wrapper rotates through the key
    #                pool sequentially; when every key has been
    #                burned in a cycle it raises
    #                AllMistralKeysExhaustedError and the eval run
    #                HALTS — there is NO cross-provider fallback,
    #                because silently switching judge models
    #                mid-ablation invalidates the locked-judge rule.
    #                `reasoning_effort` from `judge_reasoning_effort`
    #                is passed via model_kwargs.
    #   - anything else → Groq same-model fallback with loud caveat
    #                (violates the judge-independence rule).
    #
    # All independent judges (Ollama, Gemini, Mistral) share the same
    # temperature/top_p settings below.
    #
    # Once the first eval run under a chosen judge produces a scored
    # row, that judge is LOCKED for the whole technique-ablation
    # table (locked-judge rule). Switching judges mid-ablation
    # forces re-running every row.
    # CURRENT DEFAULT — Mistral Small v26.03 (`mistral-small-2603`,
    # released 2026-03-16) via Mistral's own API. Independent from the
    # generator (qwen ≠ mistral-small), satisfies the judge-independence
    # rule. Chosen
    # over Groq gpt-oss-120b to move judge quota off the Groq free tier
    # entirely, so the generator's 8k TPM cap is not shared with
    # RAGAS's ~4-6 calls per question. Requires MISTRAL_API_KEY in .env.
    judge_provider: str = "mistral"
    judge_model: str = "mistral-small-2603"

    # Reasoning effort for the judge model. Interpretation depends on
    # provider:
    #   - Groq gpt-oss-* judges accept "low" / "medium" / "high" (OpenAI
    #     scale). Older baseline: "low".
    #   - Mistral Small v26.03 (`mistral-small-2603`, current locked
    #     judge) accepts ONLY "none" or "high". "none" is chosen here
    #     because RAGAS judge prompts are short-form structured checks
    #     ("does claim X appear in context Y, yes/no") that do not
    #     benefit from chain-of-thought, and determinism matters more
    #     than depth for a judge locked across every ablation row
    #     (locked-judge rule). "high" would introduce stochastic
    #     scoring (Mistral pairs it with temp=0.7), raising the noise
    #     floor on technique-delta measurements.
    #   - Ollama / Gemini judges ignore this field.
    # Passed via model_kwargs to ChatMistralAI (surfaces as
    # `reasoning_effort` in the Mistral request body). If a future
    # Mistral model rejects the field, they'll return HTTP 400 and
    # the loud-fail path in RotatingChatMistralAI will surface it —
    # do not silently drop the parameter.
    judge_reasoning_effort: str = "none"

    # Local Ollama endpoint. Default matches `ollama serve` out of
    # the box on all platforms. Override via OLLAMA_BASE_URL in .env
    # if Ollama runs on a different host or port.
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )

    # Judge-model sampling parameters. Held constant across every row
    # of the ablation table for the same reason the model is: any
    # change in the judge's decoding is a confound on the metric.
    judge_temperature: float = 0.0
    judge_top_p: float = 1.0

    # Provider-specific keys. Both live independently in .env so the
    # Gemini→Mistral fallback chain has both keys available. If only
    # one is set, that provider is used and the other's branch is
    # skipped. If neither is set, run_eval.py falls through to Groq
    # same-model judge with a loud caveat.
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    mistral_api_key: str = Field(default="", validation_alias="MISTRAL_API_KEY")

    # --- Mistral judge: multi-key rotation --------------------------------
    # Comma-separated list of authorised Mistral API keys, used
    # sequentially by the RAGAS judge (never in parallel). Exactly
    # mirrors the Groq generator pattern (GROQ_API_KEYS above): order
    # is preserved, one key behaves identically to the legacy single-
    # key config, empty entries are dropped, duplicates are NOT
    # deduplicated (smoke-test hook).
    #
    # If MISTRAL_API_KEYS is unset, we fall back to the singular
    # MISTRAL_API_KEY above so an existing .env still works. If
    # neither is set and judge_provider == "mistral", judge
    # construction FAILS LOUD — no cross-provider fallback. That is
    # by design: silently switching to a different judge model
    # mid-ablation invalidates the locked-judge rule.
    #
    # SECURITY: never log or serialise this field. Downstream users
    # (RotatingChatMistralAI) expose only masked identifiers
    # (`mistral_key_0`, `mistral_key_1`, ...).
    mistral_api_keys_raw: str = Field(
        default="", validation_alias="MISTRAL_API_KEYS"
    )

    @property
    def mistral_api_keys(self) -> list[str]:
        """
        Parsed, ordered, whitespace-stripped list of Mistral API keys.

        Falls back to the singular `mistral_api_key` when
        `MISTRAL_API_KEYS` is empty so a legacy single-key .env
        still works. Empty entries (double commas, trailing
        commas) are dropped. Duplicates are preserved.
        """
        raw = (self.mistral_api_keys_raw or "").strip()
        if raw:
            parts = [p.strip() for p in raw.split(",")]
            keys = [p for p in parts if p]
            if keys:
                return keys
        single = (self.mistral_api_key or "").strip()
        return [single] if single else []


settings = Settings()


# ============================================================
# CORPUS_SCOPE — mirrored from scope.md v1.1 (locked after Phase 1
# collection). scope.md is the authoritative scope document; the
# values below are a machine-readable reflection of it, not a
# separate source of truth. If scope changes, edit scope.md first
# and then update this block to match.
#
# The corpus is curated by hand by the owner from official sources
# and lives under data/raw/[SCHEME]/{01_RAW_PDFs,02_Workflows}/.
# There is no scraper in this project — the loader in Phase 2 reads
# whatever is already on disk.
# ============================================================

# --- Schemes in scope (V1) ---------------------------------------
# Seven schemes, per scope.md §2. Folder slug is the on-disk name
# under data/raw/. Category is the regulatory-shape label used in
# scope.md §2 and §7 (knowledge-diversity argument).
CORPUS_SCOPE_SCHEMES: tuple[dict, ...] = (
    {
        "slug": "PM_KISAN",
        "name": "PM-KISAN",
        "category": "Income Support",
        "adjacent_included": ("PM-KMY",),
    },
    {
        "slug": "PMFBY",
        "name": "PMFBY",
        "category": "Crop Insurance",
        "adjacent_included": (),
    },
    {
        "slug": "KCC",
        "name": "KCC",
        "category": "Agricultural Credit",
        "adjacent_included": ("MISS", "AHDF-KCC"),
    },
    {
        "slug": "SMAM",
        "name": "SMAM",
        "category": "Farm Mechanization",
        "adjacent_included": (),
    },
    {
        "slug": "MIDH",
        "name": "MIDH",
        "category": "Horticulture Development",
        "adjacent_included": (),
    },
    {
        "slug": "NFSM",
        "name": "NFSM",
        "category": "Crop Productivity / Food Security",
        "adjacent_included": ("NMEO-Oil Palm",),
    },
    {
        "slug": "AIF",
        "name": "AIF",
        "category": "Agriculture Infrastructure Financing",
        "adjacent_included": (),
    },
)

# Convenience: just the slugs, in canonical order. Used to iterate
# data/raw/[SCHEME]/ folders in the loader.
CORPUS_SCOPE_SCHEME_SLUGS: tuple[str, ...] = tuple(
    s["slug"] for s in CORPUS_SCOPE_SCHEMES
)

# --- Per-scheme folder layout ------------------------------------
# Each scheme folder under data/raw/ has this fixed subfolder pattern.
# 01_RAW_PDFs/  — official PDF documents (guidelines, notifications, FAQs, amendments)
# 02_Workflows/ — CSVs extracted from scheme portals / SOPs
#                 (schema: workflow_id, step_no, instruction, input_required,
#                  condition, source_title, official_url)
CORPUS_SCOPE_PDF_SUBDIR: str = "01_RAW_PDFs"
CORPUS_SCOPE_WORKFLOW_SUBDIR: str = "02_Workflows"

# --- Corpus sources (metadata, per scope.md §6) ------------------
# These are recorded for reporting and README purposes. Not used
# as a filter — the corpus is what is physically on disk under
# data/raw/, which the owner has hand-curated.
CORPUS_SCOPE_PRIMARY_SOURCE: str = (
    "Department of Agriculture & Farmers Welfare (DA&FW), "
    "Ministry of Agriculture & Farmers Welfare, Government of India"
)
CORPUS_SCOPE_REGULATORY_SOURCE_KCC: str = "Reserve Bank of India (RBI)"
CORPUS_SCOPE_IMPLEMENTING_AGENCIES: tuple[str, ...] = (
    "NHB (National Horticulture Board) — MIDH",
    "State agriculture departments / DBT portals — NFSM (Maharashtra, Meghalaya, Tamil Nadu, Rajasthan)",
    "DBT Agriculture Mechanization portal — SMAM",
)
CORPUS_SCOPE_SUPPORTING_TECHNICAL_SOURCES: tuple[str, ...] = (
    "NCCD (National Committee on Cold-chain Development) — filed under MIDH, cross-referenced from AIF",
)


# ============================================================
# Out-of-scope refusal detection
# ============================================================
#
# Deterministic multi-phrase refusal detection used by the eval
# harness (eval/run_eval.py `refusal_correct` metric). Applied to
# `out_of_scope` questions only, where RAGAS faithfulness and
# answer_relevancy return 0.0 on refusal answers even when the
# generator refused correctly (verified against the 2026-09-13
# baseline: all 8 out_of_scope refusals were mechanically scored
# 0.0 by RAGAS). See DECISIONS.md.
#
# Matching is case-insensitive substring — any one phrase is
# sufficient. Multi-phrase check is deliberately used instead of
# an exact sentinel match so the metric survives small prompt
# wording changes (e.g. "not enough information" vs "insufficient
# information") without silently breaking. If the generator's
# system prompt ever grows a new refusal wording, add it here.
OUT_OF_SCOPE_REFUSAL_PHRASES: list[str] = [
    "not contain",
    "not available",
    "cannot answer",
    "no information",
    "does not include",
    "does not contain",
    "insufficient information",
    "unable to answer",
    "not enough information",
]


# ============================================================
# Query transformation prompt (Phase 5, fix #3)
# ============================================================
#
# Domain-primed, no-invention rewriter prompt used by
# src/query/transform.py to generate N alternate phrasings of a
# user question for multi-query retrieval. The prompt is kept
# HERE (not inside transform.py) for the same reason every other
# domain constant lives in config.py: one file to edit if the
# corpus changes, no magic strings scattered in modules.
#
# Design rules encoded in the prompt (see DECISIONS.md
# 2026-09-14 fix #3 entry):
#   1. Domain priming — the model is told the corpus is Indian
#      government agri-scheme documents so at least one rewrite
#      pulls toward corpus-side vocabulary (scheme codes, section
#      references, guideline titles) rather than user-side
#      phrasings ("what is X" / "define X" / "explain X").
#   2. No invention — the model is EXPLICITLY forbidden from
#      inserting facts, dates, subsidy rates, circular numbers,
#      or document IDs the user did NOT mention. Multi-query
#      rewrites exist to reword the question, not to add
#      information the retriever will then chase; a hallucinated
#      "section 4.2" in a rewrite would poison the fusion.
#   3. Structured output — the response MUST be a JSON object
#      with a single "rewrites" key holding a list of N strings,
#      so the parser is deterministic and short. Free-form
#      numbered lists break on model whims.
#   4. Preserve intent — every rewrite must be a valid rephrasing
#      of the SAME question, not a related question. Splitting
#      into sub-questions is a different technique (sub-question
#      decomposition) that we explicitly did NOT pick.
#
# The `{n}` and `{question}` placeholders are filled at call
# time by src/query/transform.py. Any other braces in this
# template must be escaped as `{{ }}` if we ever switch to
# str.format — currently we do simple substring replacement in
# transform.py to avoid the escaping requirement on the JSON
# example, so raw `{` is safe.
QUERY_TRANSFORM_PROMPT: str = """You are helping search a corpus of Indian government agricultural scheme \
documents (PM-KISAN, PMFBY, KCC, SMAM, MIDH, NFSM, AIF and their RBI/DA&FW \
guidelines, notifications, and circulars).

Your task: rewrite the user's question into {n} alternate phrasings, each of \
which asks the SAME question but uses different words. The goal is to help a \
retrieval system find the relevant paragraph even when the user's phrasing \
does not match the document's phrasing.

Guidelines for the rewrites:
- Use vocabulary that is likely to appear in scheme documents: \
scheme codes (SMAM, PMFBY, KCC, PM-KISAN, MIDH, NFSM, AIF), \
formal terms (e.g. "eligibility", "beneficiary", "subvention", \
"quantum of assistance"), and section/clause style phrasing where natural.
- At least one rewrite should be a corpus-side phrasing — how the guideline \
document itself would likely state the topic — not just a user-side paraphrase.
- Preserve the ORIGINAL intent exactly. Do NOT split the question into \
sub-questions. Do NOT change the scheme, the year, the entity, or the \
attribute being asked about.
- DO NOT invent facts, subsidy rates, dates, circular numbers, section \
numbers, document IDs, or any specifics the user did NOT include. Rewrites \
must only rephrase — never add information.
- If the user's question already uses precise document vocabulary, produce \
rewrites that stay close to it rather than paraphrasing away from it.

Respond with a JSON object of exactly this shape and nothing else:
{"rewrites": ["<rewrite 1>", "<rewrite 2>", ..., "<rewrite {n}>"]}

User question:
{question}
"""
