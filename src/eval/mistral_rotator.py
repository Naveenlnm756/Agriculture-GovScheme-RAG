"""
Multi-key rotation wrapper around `langchain_mistralai.ChatMistralAI`
for the RAGAS judge.

WHY THIS EXISTS
---------------
The RAGAS judge issues ~4 calls per golden question (one per metric).
A 78-question eval is ~312 judge calls, plus retries. On Mistral's
free tier a single key will occasionally hit the per-minute request
cap during a run, especially when several RAGAS metrics fire on the
same question in quick succession. When several authorised Mistral
keys are available, rotating through them sequentially spreads the
load and keeps the run moving without introducing minute-long
retry-backoff windows.

Mirrors the shape of `src/utils/key_rotator.py` (the Groq generator
rotator) but is Mistral-only:
  - Mistral-specific exception classification (429/401 on the
    Mistral endpoint via httpx).
  - No Groq / OpenAI imports.
  - **No cross-provider fallback.** When every configured Mistral
    key has been rotated away from in a cycle, this module raises
    `AllMistralKeysExhaustedError` and the eval halts. CLAUDE.md §11
    rule 4 forbids silently switching judge models mid-ablation-row
    (a metric movement between rows must be attributable to the
    technique under test, not to a swap in the judge).

Rotation is strictly sequential — no threads, no async races, no
random-key selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_mistralai import ChatMistralAI
from pydantic import Field, PrivateAttr

logger = logging.getLogger(__name__)


class AllMistralKeysExhaustedError(Exception):
    """
    Every configured Mistral API key has been rotated away from in
    the current cycle.

    Raised loud by `RotatingChatMistralAI` so the eval run halts
    rather than silently switching to a different judge model.
    Mid-ablation cross-provider fallback would invalidate the
    comparison table (CLAUDE.md §11 rule 4): a metric movement
    could then be caused by the technique OR by the judge changing.
    """


@dataclass
class MistralRotationEvent:
    ts: str
    from_masked: str
    to_masked: str
    reason: str


class RotatingChatMistralAI(BaseChatModel):
    """
    LangChain chat model that owns one `ChatMistralAI` per configured
    Mistral API key and rotates between them on 429 / 401 errors.

    Rotation is sequential and stateful (no threads, no random
    selection). On each rotation the burned key is marked; when every
    key has been burned in the current cycle, the wrapper raises
    `AllMistralKeysExhaustedError` — the caller does not get to
    retry, and the run stops.

    Model, temperature, top_p, and `model_kwargs` (which carries
    `reasoning_effort`) are held constant across every underlying
    client — CLAUDE.md §11 rule 4 forbids changing decoding within
    an ablation table. All keys must therefore behave identically
    apart from their auth header.

    Never log `keys`. All log lines and every field emitted from
    `stats()` use the masked form `mistral_key_N` where N is the
    position in the configured list.
    """

    keys: List[str] = Field(..., description="Mistral API keys, ordered")
    model: str = Field(default="mistral-small-2603")
    temperature: float = Field(default=0.0)
    top_p: float = Field(default=1.0)
    # `reasoning_effort`, `safe_prompt`, etc. Merged into the request
    # body by ChatMistralAI. Held identical across every underlying
    # per-key client — see class docstring.
    model_kwargs: dict = Field(default_factory=dict)

    _clients: List[ChatMistralAI] = PrivateAttr(default_factory=list)
    _current_idx: int = PrivateAttr(default=0)
    _burned: set = PrivateAttr(default_factory=set)
    _rotations: int = PrivateAttr(default=0)
    _rate_limit_errors: int = PrivateAttr(default=0)
    _history: List[MistralRotationEvent] = PrivateAttr(default_factory=list)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.keys:
            raise ValueError(
                "RotatingChatMistralAI requires at least one Mistral API "
                "key. Set MISTRAL_API_KEYS (comma-separated) or the legacy "
                "MISTRAL_API_KEY in .env."
            )
        # One ChatMistralAI per key. Built at init time so we don't pay
        # LangChain's per-request setup cost inside the retry loop, and
        # so a bad key surfaces its 401 immediately rather than mid-run.
        self._clients = [
            ChatMistralAI(
                model=self.model,
                mistral_api_key=k,
                temperature=self.temperature,
                top_p=self.top_p,
                model_kwargs=dict(self.model_kwargs),
            )
            for k in self.keys
        ]

    @property
    def _llm_type(self) -> str:
        return f"rotating-mistral-{self.model}"

    # --- masking / accessors -----------------------------------------

    def masked(self, idx: Optional[int] = None) -> str:
        i = self._current_idx if idx is None else idx
        return f"mistral_key_{i}"

    def n_keys(self) -> int:
        return len(self.keys)

    # --- error classification ----------------------------------------

    def _is_rotate_error(self, exc: Exception) -> bool:
        """
        Should this exception trigger rotation to the next key?

        True for HTTP 429 (rate limit) and 401 (auth — dead / revoked
        key). Everything else propagates: a 400 is a malformed request
        (rotating hides the bug), a 5xx is server-side (rotating won't
        help), a network error is environmental (retry belongs at a
        higher layer).
        """
        # httpx is what langchain-mistralai uses under the hood.
        try:
            import httpx
            if isinstance(exc, httpx.HTTPStatusError):
                code = exc.response.status_code
                return code in (429, 401)
        except ImportError:  # pragma: no cover
            pass

        # LangChain sometimes wraps transport errors as plain Exception
        # with the HTTP status embedded in the message. Best-effort
        # fallback — this is a heuristic, not a guarantee. If a real
        # error slips through and gets treated as rotate-worthy, the
        # cycle will still terminate at AllMistralKeysExhaustedError.
        msg = str(exc).lower()
        if "429" in msg or "rate limit" in msg or "too many requests" in msg:
            return True
        if "401" in msg or "unauthorized" in msg or "invalid api key" in msg:
            return True
        return False

    # --- rotation -----------------------------------------------------

    def _rotate(self, reason: str) -> None:
        prev = self._current_idx
        self._burned.add(prev)
        if reason == "rate_limit":
            self._rate_limit_errors += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if len(self._burned) >= len(self.keys):
            logger.warning(
                "[%s] Mistral keys exhausted (cycle burned): %s reason=%s",
                ts, sorted(f"mistral_key_{i}" for i in self._burned), reason,
            )
            raise AllMistralKeysExhaustedError(
                f"All {len(self.keys)} configured Mistral keys have been "
                f"rotated away from in this cycle (reason={reason}). No "
                f"cross-provider fallback — halting eval per CLAUDE.md "
                f"§11 rule 4."
            )

        nxt = (prev + 1) % len(self.keys)
        while nxt in self._burned:
            nxt = (nxt + 1) % len(self.keys)
        self._current_idx = nxt
        self._rotations += 1

        ev = MistralRotationEvent(
            ts=ts,
            from_masked=f"mistral_key_{prev}",
            to_masked=f"mistral_key_{nxt}",
            reason=reason,
        )
        self._history.append(ev)
        logger.info(
            "[%s] Mistral key rotation: %s -> %s reason=%s",
            ts, ev.from_masked, ev.to_masked, reason,
        )

    # --- LangChain chat-model interface ------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Sync path. RAGAS is async and mostly hits _agenerate below,
        # but the sync path is implemented for completeness and for
        # LangChain's `.invoke()` / smoke-test callers.
        while True:
            client = self._clients[self._current_idx]
            try:
                return client._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            except Exception as exc:
                if self._is_rotate_error(exc):
                    # _rotate() raises AllMistralKeysExhaustedError when
                    # every key has been burned — we do NOT catch that
                    # here; it propagates and stops the eval.
                    self._rotate(reason="rate_limit")
                    continue
                raise

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Async path — the one RAGAS actually hits.
        while True:
            client = self._clients[self._current_idx]
            try:
                return await client._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            except Exception as exc:
                if self._is_rotate_error(exc):
                    self._rotate(reason="rate_limit")
                    continue
                raise

    # --- stats for the eval summary ----------------------------------

    def stats(self) -> dict:
        """Masked snapshot for EvalSummary. Never contains raw keys."""
        return {
            "n_keys_configured": len(self.keys),
            "key_rotations": self._rotations,
            "rate_limit_errors": self._rate_limit_errors,
            "rotation_history": [
                {
                    "ts": e.ts,
                    "from": e.from_masked,
                    "to": e.to_masked,
                    "reason": e.reason,
                }
                for e in self._history
            ],
        }
