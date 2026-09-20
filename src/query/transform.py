"""
Query transformation — multi-query expansion (Phase 5, fix #3).

Rewrites a user question into N alternate phrasings so a downstream
retriever can look for the same intent under different vocabulary.
The rewrites plus the ORIGINAL question are fused at retrieval time
via Reciprocal Rank Fusion (fusion happens in the pipeline layer,
not here) into a single top-K passage set for the generator.

Boundaries:
  * This module OWNS: prompt filling, LLM call kwargs specific to the
    rewriter (temperature, max_tokens, reasoning_effort, response_format),
    JSON parsing, shape validation, defensive [] fallback on parse
    failure.
  * This module does NOT own: retry-on-429, key rotation, retrieval,
    fusion. Rotation is the caller's job (wrap the call in
    `EvalGroqClient._call_with_rotation`); retrieval and fusion live
    in `src/retrieval/`.

Rate-limit exceptions (`RateLimitError`, 429 `APIStatusError`) are
RE-RAISED so the rotating caller can retry the whole thunk under a
fresh key. Every OTHER failure mode — malformed JSON, wrong shape,
empty response, transport error — returns `[]` so the pipeline can
gracefully degrade to original-only retrieval for that question. The
row will score no better than baseline on that question, which is
the correct behaviour for a failed intervention. It must not crash
the whole eval run.

See DECISIONS.md 2026-09-14 "fix #3 design decisions" entry for
why multi-query expansion was chosen over sub-question decomposition
and HyDE, and why retrieval-time fusion was chosen over
generation-time fusion.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from groq import APIStatusError, Groq, RateLimitError

from src.config import QUERY_TRANSFORM_PROMPT, settings


logger = logging.getLogger(__name__)


# --- Prompt filling ---------------------------------------------------------

def _fill_prompt(question: str, n: int) -> str:
    """
    Substitute {n} and {question} into the prompt template.

    Simple substring replacement — not str.format — so raw `{` and `}`
    inside the JSON-shape example in the template don't need to be
    escaped. `{n}` is also substituted inside the JSON example
    (`"<rewrite {n}>"` becomes e.g. `"<rewrite 3>"`), which is the
    desired behaviour: the model sees a concrete example.
    """
    return (
        QUERY_TRANSFORM_PROMPT
        .replace("{n}", str(n))
        .replace("{question}", question)
    )


# --- Response parsing -------------------------------------------------------

def _parse_rewrites(raw: str, expected_n: int) -> list[str]:
    """
    Parse the LLM's JSON response into a list of rewrite strings.

    Returns `[]` on ANY shape / parse failure. This is deliberately
    defensive: a rewriter that occasionally returns malformed JSON
    must degrade the affected question to original-only retrieval,
    not crash the whole eval. Every failure path logs a WARNING with
    the head of the raw response so a systematic parse failure (e.g.
    the model started emitting a numbered list because the JSON
    instruction slipped out of context) is visible in the eval log.

    Successful parse rules:
      * Response must be a JSON object.
      * Must contain a top-level "rewrites" key.
      * "rewrites" must be a list.
      * Each entry that survives is a non-empty string (str, stripped).
      * If the model returned FEWER than `expected_n` valid rewrites,
        we log at INFO and return what we have — a fusion with 2
        rewrites is still better than a fusion with 0. If it returned
        MORE, we truncate to `expected_n` so the caller's cost model
        (N+1 retrievals) is not silently violated.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "QueryRewriter: response was not valid JSON: %r",
            raw[:200],
        )
        return []

    if not isinstance(obj, dict) or "rewrites" not in obj:
        logger.warning(
            "QueryRewriter: response missing top-level 'rewrites' key: %r",
            raw[:200],
        )
        return []

    rewrites = obj["rewrites"]
    if not isinstance(rewrites, list):
        logger.warning(
            "QueryRewriter: 'rewrites' is not a list: %r",
            rewrites,
        )
        return []

    cleaned = [s.strip() for s in rewrites if isinstance(s, str) and s.strip()]
    if len(cleaned) < expected_n:
        logger.info(
            "QueryRewriter: got %d valid rewrites, expected %d — "
            "using what we have; the pipeline will fuse fewer lists.",
            len(cleaned),
            expected_n,
        )

    return cleaned[:expected_n]


# --- Groq call --------------------------------------------------------------

def _call_rewriter(client: Groq, messages: list[dict], config) -> Any:
    """
    Single Groq call for the rewriter, with the rewriter's own kwargs.

    Isolated from `rewrite_query` so the caller can also invoke this
    directly if it wants to bypass parsing (unused today, but keeps
    the two responsibilities cleanly separable).

    Uses `response_format={"type": "json_object"}` — Groq's structured-
    output mode for models that support it. This together with the
    prompt's explicit JSON-shape instruction is what makes the parser
    on the other side simple and deterministic. If a future model
    rejects the kwarg with HTTP 400, we surface the error rather than
    silently degrade — a rewriter that quietly stops enforcing JSON
    output would be a much subtler failure than a loud 400.

    The `reasoning_effort` handoff matches the pattern in
    generation/generator.py: prefer the direct kwarg, fall back to
    `extra_body` on older Groq SDK versions that don't accept it.
    """
    def _create(use_extra_body: bool):
        kwargs = dict(
            model=config.llm_model,
            messages=messages,
            max_tokens=config.query_transform_max_tokens,
            temperature=config.query_transform_temperature,
            response_format={"type": "json_object"},
        )
        if use_extra_body:
            kwargs["extra_body"] = {
                "reasoning_effort": config.query_transform_reasoning_effort
            }
        else:
            kwargs["reasoning_effort"] = config.query_transform_reasoning_effort
        return client.chat.completions.create(**kwargs)

    try:
        return _create(use_extra_body=False)
    except TypeError:
        # SDK does not accept `reasoning_effort` as a direct kwarg on
        # this version — forward via extra_body. Same fallback path
        # the production generator uses.
        return _create(use_extra_body=True)


# --- Public entry point -----------------------------------------------------

def rewrite_query(
    question: str,
    client: Groq,
    config=settings,
) -> list[str]:
    """
    Produce up to N alternate phrasings of `question` for multi-query
    retrieval.

    N is `config.query_transform_n_rewrites` (default 3).

    Returns a list of rewrite strings. On ANY LOGICAL failure —
    malformed JSON, wrong shape, empty response, unexpected content-
    type from the SDK — returns `[]`. The caller MUST interpret `[]`
    as "fall back to original-only retrieval for this question", not
    as a fatal error.

    On rate-limit failures (429), the underlying exceptions are
    RE-RAISED so a rotating caller can retry the whole thunk under a
    fresh key. If the caller is not wrapping us in rotation
    (e.g. a single-key production path), the exception propagates —
    which is the correct signal that the request couldn't complete,
    not something to silently mask by returning `[]`.

    Threading model. Same as the production generator: the caller
    supplies a `Groq` client; we make one synchronous call and
    return. No caching of the response, no shared state between
    calls, no thread safety concerns beyond whatever the client
    object provides.

    Pacing. We sleep `config.llm_call_delay_s` before the call, same
    as the generator. A query_transform row adds ONE extra LLM call
    per question; without this pacing the eval's effective TPM
    headroom would silently halve.
    """
    n = config.query_transform_n_rewrites
    if n <= 0:
        logger.info(
            "QueryRewriter: n_rewrites=%d, returning [] without a call.",
            n,
        )
        return []

    prompt = _fill_prompt(question, n)
    messages = [{"role": "user", "content": prompt}]

    if config.llm_call_delay_s > 0:
        time.sleep(config.llm_call_delay_s)

    start_ns = time.perf_counter_ns()
    try:
        completion = _call_rewriter(client, messages, config)
    except (RateLimitError, APIStatusError) as e:
        # Rate limits are the caller's problem. Re-raise so
        # EvalGroqClient._call_with_rotation can rotate keys and
        # retry the whole rewrite call under a fresh client. A
        # non-429 APIStatusError also re-raises: unexpected API
        # errors should surface, not silently degrade every rewrite.
        raise
    except Exception as e:
        # Any other transport error → defensive [] fallback. The
        # pipeline falls back to original-only retrieval for this
        # question; the eval row records the miss under baseline
        # semantics, which is the correct attribution.
        logger.warning(
            "QueryRewriter: Groq call raised %s: %s — returning [].",
            type(e).__name__,
            e,
        )
        return []
    latency_ms = (time.perf_counter_ns() - start_ns) // 1_000_000

    choice = completion.choices[0]
    raw_content = choice.message.content or ""
    finish_reason = choice.finish_reason or ""

    # Reasoning-budget-exhaustion path: gpt-oss-20b can consume the
    # entire completion budget on invisible reasoning tokens and
    # return empty visible content with finish_reason=length. Mirror
    # the production generator's WARNING + fallback so this failure
    # mode is loud in eval logs rather than a silent [] return.
    if not raw_content.strip():
        usage = getattr(completion, "usage", None)
        reasoning_tokens = 0
        completion_tokens = 0
        if usage is not None:
            completion_tokens = int(
                getattr(usage, "completion_tokens", 0) or 0
            )
            ctd = getattr(usage, "completion_tokens_details", None)
            if ctd is not None:
                reasoning_tokens = int(
                    getattr(ctd, "reasoning_tokens", 0) or 0
                )
        logger.warning(
            "QueryRewriter: empty visible content, finish_reason=%s, "
            "reasoning_tokens=%d, completion_tokens=%d, max_tokens=%d. "
            "Falling back to [] (original-only retrieval).",
            finish_reason,
            reasoning_tokens,
            completion_tokens,
            config.query_transform_max_tokens,
        )
        return []

    rewrites = _parse_rewrites(raw_content, expected_n=n)

    logger.info(
        "rewrite_query(question=%r) → %d rewrites (of %d requested), "
        "latency_ms=%d, finish=%s",
        question[:80],
        len(rewrites),
        n,
        latency_ms,
        finish_reason,
    )
    return rewrites
