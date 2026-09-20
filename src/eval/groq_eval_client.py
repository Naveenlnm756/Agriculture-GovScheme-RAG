"""
Evaluation-only Groq client with sequential key rotation.

Owns TWO surfaces on top of `KeyRotator`:

  * `EvalGroqClient.generate_answer(query, chunks)` — mirrors the
    production `src/generation/generator.generate()` signature and
    return type, but every Groq call is dispatched through
    `_call_with_rotation` so the eval harness can rotate keys on 429
    without touching the production generator.

  * `build_rotating_judge_llm(...)` — returns a LangChain ChatOpenAI
    subclass (`KeyRotatingChatOpenAI`) that rebuilds its underlying
    `openai` client after each rotation. This is what RAGAS's
    `LangchainLLMWrapper` receives as the judge.

Both surfaces share ONE `KeyRotator` instance so the run-level
`stats()` reflects generator-path AND judge-path traffic together.

Explicit non-goals:
  * No parallelism, no threading, no async rotation.
  * No mutation of an existing Groq client's api_key — rotation always
    constructs a fresh client bound to the new key.
  * Nothing here monkey-patches or otherwise touches the production
    generator at `src/generation/generator.py`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from groq import APIStatusError, Groq, RateLimitError

from src.config import settings
from src.generation.generator import _build_system_prompt, _build_user_message
from src.ingestion.models import GenerationResult, RetrievalResult
from src.utils.key_rotator import AllKeysExhaustedError, KeyRotator

logger = logging.getLogger(__name__)


def _default_max_retries(rotator: KeyRotator) -> int:
    # `>= 2 * n_keys` per Part 4 of the spec — enough to walk the whole
    # pool twice (once burn-through, one cooldown-reset walk) before we
    # give up. Clamped low so a misconfigured loop doesn't hammer Groq
    # forever.
    return max(6, rotator.n_keys() * 2)


class EvalGroqClient:
    """
    Serial Groq client with key rotation. One instance per eval run.

    Wraps a `KeyRotator`. Every call to `_call_with_rotation` builds a
    fresh `Groq(api_key=...)` client from the rotator's current key —
    we never mutate an existing client. On a rate-limit exception, the
    key is rotated and a new client is built for the retry.
    """

    def __init__(self, rotator: KeyRotator, config=settings) -> None:
        self._rotator = rotator
        self._config = config
        self._max_retries = _default_max_retries(rotator)

    # --- shared rotation loop ----------------------------------------

    def _call_with_rotation(self, thunk: Callable[[Groq], Any]) -> Any:
        """
        Run `thunk(groq_client)` with rotation on rate-limit failures.

        `thunk` receives a freshly-constructed `Groq` client bound to
        the currently-active key. Never cache the client outside a
        single thunk call — after rotation the previous client is
        stale.
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            # Simulated pre-emptive rotation for deterministic testing.
            if self._rotator.should_simulate_rate_limit():
                try:
                    self._rotator.rotate(reason="simulated_quota_exhaustion")
                except AllKeysExhaustedError:
                    self._rotator.enter_cooldown()
                continue

            client = Groq(api_key=self._rotator.current())
            try:
                self._rotator.record_call()
                return thunk(client)
            except (RateLimitError, APIStatusError) as e:
                if isinstance(e, APIStatusError) and getattr(e, "status_code", None) != 429:
                    # Non-429 API errors bubble out unchanged — do not rotate.
                    raise
                last_exc = e
                logger.warning(
                    "Groq call hit rate limit on %s (attempt %d/%d); "
                    "rotating keys.",
                    self._rotator.masked(), attempt + 1, self._max_retries,
                )
                try:
                    self._rotator.rotate(reason="rate_limit")
                except AllKeysExhaustedError:
                    self._rotator.enter_cooldown()
                continue
        raise RuntimeError(
            f"EvalGroqClient: exhausted {self._max_retries} rotation retries. "
            f"Last error: {last_exc!r}"
        )

    # --- generator surface -------------------------------------------

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievalResult],
    ) -> GenerationResult:
        """
        Mirror of `src.generation.generator.generate()` — same prompt
        contract, same GenerationResult shape — but Groq calls go
        through the rotator. Prompt-building helpers are imported from
        the production generator so the two paths cannot drift.
        """
        cfg = self._config
        system_prompt = _build_system_prompt(cfg)
        user_message = _build_user_message(query, retrieved_chunks)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Pacing between successive Groq calls — same behaviour the
        # production generator enforces via `llm_call_delay_s`. Kept
        # here so switching to the rotating path doesn't lose the
        # under-cap pacing.
        if cfg.llm_call_delay_s > 0:
            time.sleep(cfg.llm_call_delay_s)

        def _thunk(client: Groq):
            kwargs = dict(
                model=cfg.llm_model,
                messages=messages,
                max_tokens=cfg.generation_max_tokens,
                temperature=cfg.generation_temperature,
            )
            # qwen/qwen3.6-27b accepts reasoning_effort in ["none", "default"];
            # openai/gpt-oss-* accepts ["low", "medium", "high"]. Locked
            # combinations already agree with cfg.generation_reasoning_effort.
            try:
                return client.chat.completions.create(
                    **kwargs, reasoning_effort=cfg.generation_reasoning_effort,
                )
            except TypeError:
                # Older Groq SDK: no direct kwarg — forward via extra_body.
                return client.chat.completions.create(
                    **kwargs,
                    extra_body={
                        "reasoning_effort": cfg.generation_reasoning_effort
                    },
                )

        start_ns = time.perf_counter_ns()
        completion = self._call_with_rotation(_thunk)
        latency_ms = (time.perf_counter_ns() - start_ns) // 1_000_000

        choice = completion.choices[0]
        raw_content = choice.message.content or ""
        finish_reason = choice.finish_reason or ""

        usage = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        reasoning_tokens = 0
        ctd = getattr(usage, "completion_tokens_details", None)
        if ctd is not None:
            reasoning_tokens = int(getattr(ctd, "reasoning_tokens", 0) or 0)

        answer = raw_content
        if not raw_content.strip() and finish_reason == "length":
            # Same sentinel behaviour as production generator so eval
            # runs surface reasoning-budget exhaustion loudly rather
            # than counting the miss as a retrieval failure.
            answer = (
                "The generator was interrupted before producing a visible answer. "
                "This should be investigated."
            )
            logger.warning(
                "Empty content, finish_reason=length: reasoning_tokens=%d, "
                "completion_tokens=%d, max_tokens=%d",
                reasoning_tokens, completion_tokens, cfg.generation_max_tokens,
            )

        return GenerationResult(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            model_used=cfg.llm_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=int(latency_ms),
            finish_reason=finish_reason,
            # retries_taken here is not a 1:1 mirror of the production
            # counter (which only counts intra-call 429 retries on one
            # key). We keep it 0 — rotation events are tracked at the
            # run level in EvalSummary.
            retries_taken=0,
        )

    # --- query transformation surface -------------------------------
    #
    # Phase 5, fix #3. Wraps `src.query.transform.rewrite_query` in
    # the same rotation loop the generator uses so the eval harness
    # can rotate keys on 429 responses to the rewriter call without
    # touching the transform module. The public surface mirrors
    # `generate_answer` in shape (a bare method on the client) so
    # the eval pipeline reads uniformly: everything Groq-adjacent
    # goes through the same client instance and the same rotator.

    def rewrite_query(self, question: str) -> list[str]:
        """
        Produce up to `config.query_transform_n_rewrites` alternate
        phrasings of `question` via the LLM, with key rotation on 429.

        Returns a list of rewrites (possibly fewer than requested if
        the model returned an under-shape response — see
        `src.query.transform.rewrite_query`). Returns `[]` if the
        rewriter failed logically (malformed JSON, empty response,
        non-429 transport error). The caller MUST interpret `[]` as
        "fall back to original-only retrieval", not as a fatal error.

        Rate limits are handled here via `_call_with_rotation`: on a
        429, keys rotate and the whole rewriter call retries against
        a fresh client. If the entire key pool is exhausted (a real
        run-halting condition — the eval cannot proceed under a
        burned pool), `RuntimeError` propagates as it does for
        `generate_answer`.
        """
        from src.query.transform import rewrite_query as _rewrite_query

        def _thunk(client):
            return _rewrite_query(question, client, self._config)

        return self._call_with_rotation(_thunk)

    # --- judge surface -----------------------------------------------

    def build_rotating_judge_llm(
        self,
        *,
        model: str,
        temperature: float,
        top_p: float,
        reasoning_effort: str,
        base_url: str = "https://api.groq.com/openai/v1",
        rotator: KeyRotator | None = None,
    ) -> "KeyRotatingChatOpenAI":
        """
        Construct the LangChain ChatOpenAI-compatible judge.

        By default the judge shares this client's rotator (`self._rotator`)
        — historical behaviour, kept for callers that want a single shared
        Groq pool for generator and judge.

        When `rotator` is passed explicitly, the judge is bound to that
        SEPARATE rotator instead. This is the "pool-split" mode used by
        the eval harness: generator gets one key, judge gets a disjoint
        set. The judge's per-key rate-limit pressure never touches the
        generator's key, and vice versa.
        """
        judge_rotator = rotator if rotator is not None else self._rotator
        max_retries = (
            _default_max_retries(judge_rotator)
            if rotator is not None
            else self._max_retries
        )
        return KeyRotatingChatOpenAI(
            rotator=judge_rotator,
            max_retries=max_retries,
            model=model,
            temperature=temperature,
            top_p=top_p,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
        )


# ---------------------------------------------------------------------------
# LangChain judge with rotation
# ---------------------------------------------------------------------------
#
# Kept in this module so the whole eval-time Groq surface (generator +
# judge) sits together. LangChain's ChatOpenAI is a pydantic model, so
# private fields are attached via `PrivateAttr`. On rate-limit we rebuild
# the underlying `openai.OpenAI` / `openai.AsyncOpenAI` clients from the
# new key rather than mutating the existing ones — matches the "fresh
# client after rotation" rule from Part 4.

from pydantic import PrivateAttr  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from openai import AsyncOpenAI, OpenAI  # noqa: E402
from openai import APIStatusError as OpenAIAPIStatusError  # noqa: E402
from openai import RateLimitError as OpenAIRateLimitError  # noqa: E402


class KeyRotatingChatOpenAI(ChatOpenAI):
    """
    Drop-in ChatOpenAI subclass that rotates through a KeyRotator's
    pool on rate-limit. Sequential. No parallelism.

    Overrides `_generate` / `_agenerate` — both entry points RAGAS's
    LangchainLLMWrapper will end up calling depending on the metric.
    """

    _rotator: KeyRotator = PrivateAttr()
    _max_retries_kro: int = PrivateAttr(default=20)
    _base_url_str: str = PrivateAttr(default="")
    _reasoning_effort: str = PrivateAttr(default="low")

    def __init__(
        self,
        *,
        rotator: KeyRotator,
        max_retries: int,
        model: str,
        temperature: float,
        top_p: float,
        base_url: str,
        reasoning_effort: str,
        **extra: Any,
    ) -> None:
        # Build the base ChatOpenAI with the initial key.
        # `max_retries=0` on the LangChain wrapper disables its own retry
        # so 429s bubble up to us fast — but the underlying openai SDK
        # still applies its default retry budget (2), which is what
        # smoothed over the 429 storm in the earlier judge smoke test.
        super().__init__(
            model=model,
            api_key=rotator.current(),
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            max_retries=0,
            extra_body={"reasoning_effort": reasoning_effort},
            **extra,
        )
        self._rotator = rotator
        self._max_retries_kro = max_retries
        self._base_url_str = base_url
        self._reasoning_effort = reasoning_effort
        # Ensure the initial client honours our fresh-client + zero-retry
        # invariants (some ChatOpenAI versions ignore max_retries=0 on
        # the outer wrapper but not on the inner client).
        self._rebuild_underlying_clients()

    # --- underlying client management --------------------------------

    def _rebuild_underlying_clients(self) -> None:
        """
        Replace `self.client` / `self.async_client` (and root clients on
        newer LangChain) with fresh openai clients bound to the current
        rotator key. Called on init and after every rotation.
        """
        new_key = self._rotator.current()
        sync_root = OpenAI(
            api_key=new_key,
            base_url=self._base_url_str,
            max_retries=2,  # openai SDK default — smooth over transient 429s
        )
        async_root = AsyncOpenAI(
            api_key=new_key,
            base_url=self._base_url_str,
            max_retries=2,
        )
        self.client = sync_root.chat.completions
        self.async_client = async_root.chat.completions
        # LangChain 0.3+ also holds root client refs.
        try:
            object.__setattr__(self, "root_client", sync_root)
            object.__setattr__(self, "root_async_client", async_root)
        except Exception:  # pragma: no cover
            pass

    # --- rotation loop -----------------------------------------------

    def _rotate_and_rebuild(self, reason: str) -> None:
        try:
            self._rotator.rotate(reason=reason)
        except AllKeysExhaustedError:
            self._rotator.enter_cooldown()
        self._rebuild_underlying_clients()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self._max_retries_kro):
            if self._rotator.should_simulate_rate_limit():
                self._rotate_and_rebuild("simulated_quota_exhaustion")
                continue
            try:
                self._rotator.record_call()
                return super()._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            except (OpenAIRateLimitError, OpenAIAPIStatusError) as e:
                if isinstance(e, OpenAIAPIStatusError) and getattr(e, "status_code", None) != 429:
                    raise
                last_exc = e
                logger.warning(
                    "Judge call hit rate limit on %s (attempt %d/%d); rotating.",
                    self._rotator.masked(), attempt + 1, self._max_retries_kro,
                )
                self._rotate_and_rebuild("rate_limit")
                continue
        raise RuntimeError(
            f"KeyRotatingChatOpenAI: exhausted {self._max_retries_kro} rotation "
            f"retries. Last error: {last_exc!r}"
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self._max_retries_kro):
            if self._rotator.should_simulate_rate_limit():
                self._rotate_and_rebuild("simulated_quota_exhaustion")
                continue
            try:
                self._rotator.record_call()
                return await super()._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            except (OpenAIRateLimitError, OpenAIAPIStatusError) as e:
                if isinstance(e, OpenAIAPIStatusError) and getattr(e, "status_code", None) != 429:
                    raise
                last_exc = e
                logger.warning(
                    "Judge (async) call hit rate limit on %s (attempt %d/%d); rotating.",
                    self._rotator.masked(), attempt + 1, self._max_retries_kro,
                )
                self._rotate_and_rebuild("rate_limit")
                continue
        raise RuntimeError(
            f"KeyRotatingChatOpenAI: exhausted {self._max_retries_kro} rotation "
            f"retries (async). Last error: {last_exc!r}"
        )
