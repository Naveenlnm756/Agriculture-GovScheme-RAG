"""
Baseline grounded generator for the Agriculture Schemes RAG project.

Takes the RetrievalResult list produced by `src/retrieval/retriever.py`
and produces a natural-language answer with inline citations, using the
LLM locked in `src/config.py` (Groq gpt-oss-20b — see CLAUDE.md §11).

This is the DELIBERATE baseline. It does exactly two things beyond
"call an LLM": (1) it constrains the LLM to answer only from the
retrieved passages (grounding), and (2) it requires an inline citation
on every substantive claim. CRAG-style refusal logic, self-consistency
checks, per-claim faithfulness verification, and other quality
guardrails are Phase 5 ablation interventions (CLAUDE.md §7, fix #5)
and must beat this baseline on measured metrics.

Two decisions worth naming out loud so downstream edits don't unpick
them by accident:

  1. **The generator does NOT retrieve.** It takes retrieved chunks in
     and produces an answer out. Wiring the two together is the
     caller's job (a thin `pipeline.py` in Phase 5 will do it). The
     separation exists because retrieval and generation are measured
     independently by RAGAS (context precision/recall on retrieval;
     faithfulness on generation), and mashing them into one function
     would make ablation experiments messier — you couldn't hold one
     stage constant while varying the other.

  2. **Groq client is loaded exactly once per process** via a lazy
     module-level cache. Instantiating a Groq() object is cheap in
     wall-clock terms, but doing it per-call also re-reads the API key
     from the environment on every call, which is both wasteful and
     surprising (env changes during a run would silently take effect
     mid-run). One cached client makes the behaviour predictable.
"""

from __future__ import annotations

import logging
import time
from threading import Lock

from groq import APIStatusError, Groq, RateLimitError

from src.config import settings
from src.ingestion.models import GenerationResult, RetrievalResult


logger = logging.getLogger(__name__)


# --- Lazy singleton ---------------------------------------------------------

_client: Groq | None = None
_lock = Lock()


def _load_groq_client(config=settings) -> Groq:
    """
    Return a process-cached Groq client, constructed on first call.

    Reads the API key from `config.llm_api_key` (which is populated
    from the GROQ_API_KEY env var — see src/config.py). We deliberately
    do NOT touch `os.environ` here; the config layer is the single
    place that knows which env var names hold which credentials, so
    the provider swap in CLAUDE.md §11 stays a one-file change.
    """
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                if not config.llm_api_key:
                    raise RuntimeError(
                        "Groq API key is empty. Set GROQ_API_KEY in .env "
                        "(see .env.example). The generator cannot run "
                        "without it."
                    )
                logger.info(
                    "Initialising Groq client for model %s", config.llm_model
                )
                _client = Groq(api_key=config.llm_api_key)
    return _client


# --- Prompt construction ----------------------------------------------------

def _build_system_prompt(config=settings) -> str:
    """
    The grounding-and-citation contract for the LLM.

    Deliberately terse and rule-shaped rather than persona-shaped.
    Each rule earns its place in the file docstring's "what the
    baseline does" list — no persona-fluff, no "you are a helpful
    assistant" chatter, nothing that isn't load-bearing for either
    grounding or citation.
    """
    return (
        "You are answering questions about Government of India agriculture "
        "schemes using ONLY the passages provided to you below. Follow every "
        "rule exactly:\n"
        "\n"
        "1. Answer ONLY from the provided passages. Do not use any outside "
        "knowledge, even if you are confident it is correct. If the passages "
        "do not contain enough information to answer, say exactly: "
        "\"The retrieved sources do not contain enough information to answer "
        "this question.\" Do not guess, do not extrapolate, do not fill gaps "
        "from general knowledge.\n"
        "\n"
        "2. Cite your source for every substantive claim using inline "
        "citations in one of these two exact formats:\n"
        "   - For PDF passages: [Source: {scheme}/{source_filename}, page {page_start}]\n"
        "   - For workflow passages: [Source: {scheme}/{workflow_id}]\n"
        "   Use the scheme, filename, page, and workflow_id shown in each "
        "passage's header. Do not invent or reformat citations. Multiple "
        "citations per claim are allowed and encouraged when more than one "
        "passage supports the claim.\n"
        "\n"
        "3. Be concise. Do not restate the question. Do not add "
        "closing pleasantries. Do not add caveats the passages do not "
        "themselves make.\n"
        "\n"
        "4. If passages conflict (e.g. an older guideline vs a newer "
        "amendment), report both and cite both — do not silently prefer "
        "one over the other. The user needs to see the conflict.\n"
    )


def _chunk_header(chunk: RetrievalResult, idx: int) -> str:
    """
    One-line label above each chunk in the user message.

    Renders the exact fields the citation format needs so the LLM
    can copy them into a citation verbatim. Anything the LLM might
    have to guess (scheme spelling, filename) shows up here spelled
    correctly, which is the whole reason the citation rule is
    enforceable.
    """
    bits = [f"[Chunk {idx}]", f"scheme={chunk.scheme}"]
    if chunk.source_type == "pdf":
        bits.append(f"source_filename={chunk.source_filename}")
        if chunk.page_start is not None:
            bits.append(f"page_start={chunk.page_start}")
        if chunk.page_end is not None and chunk.page_end != chunk.page_start:
            bits.append(f"page_end={chunk.page_end}")
    elif chunk.source_type == "workflow":
        if chunk.workflow_id is not None:
            bits.append(f"workflow_id={chunk.workflow_id}")
    return "  ".join(bits)


def _build_user_message(
    query: str, chunks: list[RetrievalResult]
) -> str:
    """
    Assemble the three-section user message: chunks, separator, question.

    Passages come first (LLMs weight later tokens more heavily in some
    architectures, but instruction-following models trained with the
    "context then question" pattern in view do fine either way — and
    having the question at the end lets the LLM re-read it against the
    passages it just consumed). One blank line between chunks so a
    small model doesn't fuse them; a bold separator before the question
    so it's visually unmistakable.
    """
    if not chunks:
        chunks_block = "(no passages retrieved)"
    else:
        parts = []
        for idx, chunk in enumerate(chunks, start=1):
            parts.append(_chunk_header(chunk, idx))
            parts.append(chunk.text)
            parts.append("")  # blank line between chunks
        chunks_block = "\n".join(parts).rstrip()

    return (
        "RETRIEVED PASSAGES:\n"
        "\n"
        f"{chunks_block}\n"
        "\n"
        "---\n"
        "\n"
        f"QUESTION: {query}\n"
    )


# --- Groq call with retry ---------------------------------------------------

def _call_groq_with_retry(
    client: Groq, messages: list[dict], config=settings
) -> tuple[object, int]:
    """
    Call Groq chat.completions.create with exponential backoff on 429.

    Returns (completion, retries_taken). Raises after
    `config.llm_max_retries` retries are exhausted — the caller
    should surface that to the user, not swallow it (a silent failure
    would produce an empty-answer row in the eval and misattribute
    the metric drop to the retrieval stage).

    The `llm_call_delay_s` sleep at the top of every call is the
    baseline throttle for staying under Groq's 8k TPM cap during
    long eval runs — cheaper than eating a 429 every time.
    """
    delay_before_call = config.llm_call_delay_s
    if delay_before_call > 0:
        time.sleep(delay_before_call)

    # gpt-oss-20b is a reasoning model — see config.generation_reasoning_effort
    # for the full justification. We prefer the direct kwarg (cleaner, typed),
    # but if the installed Groq SDK version doesn't yet expose it, fall back
    # to `extra_body`, which the SDK forwards verbatim to the API. Both paths
    # produce identical wire calls.
    def _create(use_extra_body: bool):
        kwargs = dict(
            model=config.llm_model,
            messages=messages,
            max_tokens=config.generation_max_tokens,
            temperature=config.generation_temperature,
        )
        if use_extra_body:
            kwargs["extra_body"] = {
                "reasoning_effort": config.generation_reasoning_effort
            }
        else:
            kwargs["reasoning_effort"] = config.generation_reasoning_effort
        return client.chat.completions.create(**kwargs)

    last_exc: Exception | None = None
    for attempt in range(config.llm_max_retries + 1):
        try:
            try:
                completion = _create(use_extra_body=False)
            except TypeError:
                # SDK does not accept `reasoning_effort` as a direct kwarg
                # on this version — fall through to extra_body path.
                completion = _create(use_extra_body=True)
            return completion, attempt
        except RateLimitError as e:
            last_exc = e
            if attempt >= config.llm_max_retries:
                break
            # Exponential backoff: base, 2*base, 4*base, ...
            sleep_s = config.llm_backoff_base_s * (2 ** attempt)
            logger.warning(
                "Groq 429 (attempt %d/%d), backing off %.1fs",
                attempt + 1, config.llm_max_retries + 1, sleep_s,
            )
            time.sleep(sleep_s)
        except APIStatusError as e:
            # Some SDK versions surface 429 as APIStatusError with
            # status_code=429 rather than the dedicated RateLimitError.
            # Treat that path identically; anything else re-raises.
            if getattr(e, "status_code", None) == 429:
                last_exc = e
                if attempt >= config.llm_max_retries:
                    break
                sleep_s = config.llm_backoff_base_s * (2 ** attempt)
                logger.warning(
                    "Groq 429 via APIStatusError (attempt %d/%d), backing off %.1fs",
                    attempt + 1, config.llm_max_retries + 1, sleep_s,
                )
                time.sleep(sleep_s)
            else:
                raise

    raise RuntimeError(
        f"Groq generation failed after {config.llm_max_retries} retries "
        f"on 429 responses. Last error: {last_exc}"
    ) from last_exc


# --- Public entry point -----------------------------------------------------

def generate(
    query: str,
    retrieved_chunks: list[RetrievalResult],
    config=settings,
) -> GenerationResult:
    """
    Produce a grounded, cited answer for `query` using `retrieved_chunks`.

    The generator does NOT retrieve. Wiring retriever → generator is
    the caller's job.
    """
    client = _load_groq_client(config)
    system_prompt = _build_system_prompt(config)
    user_message = _build_user_message(query, retrieved_chunks)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    start_ns = time.perf_counter_ns()
    completion, retries = _call_groq_with_retry(client, messages, config)
    latency_ms = (time.perf_counter_ns() - start_ns) // 1_000_000

    choice = completion.choices[0]
    raw_content = choice.message.content or ""
    finish_reason = choice.finish_reason or ""

    usage = getattr(completion, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    # completion_tokens_details.reasoning_tokens exposes the reasoning-
    # channel usage on gpt-oss-* models. Older SDKs / non-reasoning models
    # may omit the field entirely — default to 0 so the guard below still
    # works and downstream code doesn't have to special-case None.
    reasoning_tokens = 0
    ctd = getattr(usage, "completion_tokens_details", None)
    if ctd is not None:
        reasoning_tokens = int(getattr(ctd, "reasoning_tokens", 0) or 0)

    # --- Empty-content guardrail ---
    # If the model exhausted its completion budget on reasoning tokens and
    # never emitted any visible content, the SDK returns an empty string
    # + finish_reason="length". Silently returning "" would mask the
    # failure downstream (an empty-answer eval row would misattribute the
    # miss to retrieval). Emit a visible sentinel + a WARNING log so the
    # failure is loud, not silent. See config.generation_reasoning_effort
    # for the primary mitigation (reasoning_effort="low").
    EMPTY_ANSWER_SENTINEL = (
        "The generator was interrupted before producing a visible answer. "
        "This should be investigated."
    )
    if not raw_content.strip() and finish_reason == "length":
        logger.warning(
            "Empty content with finish_reason=length: reasoning_tokens=%d, "
            "completion_tokens=%d, max_tokens=%d. Model likely exhausted "
            "completion budget on reasoning; returning sentinel.",
            reasoning_tokens, completion_tokens, config.generation_max_tokens,
        )
        answer = EMPTY_ANSWER_SENTINEL
    else:
        answer = raw_content

    logger.info(
        "generate(query=%r) → %d chars, prompt_tokens=%d, completion_tokens=%d, "
        "reasoning_tokens=%d, latency_ms=%d, retries=%d, finish=%s",
        query, len(answer), prompt_tokens, completion_tokens,
        reasoning_tokens, latency_ms, retries, finish_reason,
    )

    return GenerationResult(
        query=query,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        model_used=config.llm_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        latency_ms=int(latency_ms),
        finish_reason=finish_reason,
        retries_taken=retries,
    )
