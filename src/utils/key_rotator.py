"""
Sequential key rotation for evaluation-time Groq calls.

WHY THIS EXISTS
---------------
Long evaluation runs (25/78 golden questions × 4 RAGAS judge calls per
question) burn through the 8k TPM cap on a single Groq free-tier key
and force minutes of retry-backoff. When several authorised keys are
available, rotating through them sequentially spreads the load and
keeps the run moving. This module owns ONLY the rotation state
(current index, per-key call counts, rotation history, cooldown /
exhaustion signalling). The actual Groq API calls, retries, and
exception mapping live in `src/eval/groq_eval_client.py` — see the
architecture note in `DECISIONS.md`.

Strictly sequential — no threads, no async, no random-key selection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class AllKeysExhaustedError(Exception):
    """
    Raised by `KeyRotator.rotate()` when every key in the current cycle
    has already been rotated away from. The caller (EvalGroqClient) is
    expected to either enter cooldown or surface the error — this class
    never sleeps on its own.
    """


@dataclass
class RotationEvent:
    ts: str
    from_masked: str
    to_masked: str
    reason: str


class KeyRotator:
    """
    Ordered pool of Groq API keys with per-key call accounting and
    rotation on rate-limit / quota-exhaustion signals.

    Actual keys are stored in memory only. Every log line and every
    field emitted from `stats()` uses the masked form `key_N` where
    N is the position in the configured list. Never log `current()`
    or the raw list contents.
    """

    def __init__(
        self,
        keys: list[str],
        cooldown_s: int,
        simulate_exhaustion_after: int | None = None,
    ) -> None:
        if not keys:
            raise ValueError(
                "KeyRotator requires at least one Groq API key. "
                "Set GROQ_API_KEYS in .env (comma-separated)."
            )
        self._keys: list[str] = list(keys)
        self._cooldown_s = int(cooldown_s)
        self._simulate_after = simulate_exhaustion_after
        self._current_index: int = 0
        self._per_key_calls: dict[int, int] = {i: 0 for i in range(len(self._keys))}
        # Keys we have rotated AWAY FROM in this cycle. Reset on cooldown.
        self._burned_in_cycle: set[int] = set()
        self._rotations: int = 0
        self._cooldown_events: int = 0
        self._rate_limit_errors: int = 0
        self._history: list[RotationEvent] = []

    # --- read-only accessors -----------------------------------------

    def current(self) -> str:
        """Active key. Never log this — use `masked()` instead."""
        return self._keys[self._current_index]

    def masked(self, index: int | None = None) -> str:
        idx = self._current_index if index is None else index
        return f"key_{idx}"

    def n_keys(self) -> int:
        return len(self._keys)

    # --- call accounting ---------------------------------------------

    def record_call(self) -> None:
        """Increment the attempt counter for the currently-active key."""
        self._per_key_calls[self._current_index] += 1

    def should_simulate_rate_limit(self) -> bool:
        """
        Simulation predicate for deterministic rotation testing.

        Interpretation: after N successful/attempted calls have been
        recorded against the active key, the next call is
        preemptively treated as a simulated 429. Concretely, this
        returns True when
            per_key_calls[current] >= SIMULATE_AFTER_N_CALLS
        and the caller should skip the API call and rotate instead.

        Returns False when the simulation flag is None (production).
        """
        if self._simulate_after is None:
            return False
        return self._per_key_calls[self._current_index] >= int(self._simulate_after)

    # --- error classification ----------------------------------------

    def on_error(self, exc: Exception) -> bool:
        """
        Should this exception trigger a key rotation?

        True only for rate-limit / 429-flavoured errors from either the
        Groq or OpenAI SDKs (LangChain's ChatOpenAI-over-Groq path).
        Everything else (auth, 400, network) returns False and the
        caller must propagate.
        """
        # Local imports keep the rotator import-light for scripts that
        # only need the accounting bits (e.g. tests).
        try:
            from groq import APIStatusError as GroqAPIStatusError
            from groq import RateLimitError as GroqRateLimitError
        except ImportError:  # pragma: no cover
            GroqRateLimitError = ()  # type: ignore
            GroqAPIStatusError = ()  # type: ignore
        try:
            from openai import APIStatusError as OpenAIAPIStatusError
            from openai import RateLimitError as OpenAIRateLimitError
        except ImportError:  # pragma: no cover
            OpenAIRateLimitError = ()  # type: ignore
            OpenAIAPIStatusError = ()  # type: ignore

        if isinstance(exc, (GroqRateLimitError, OpenAIRateLimitError)):
            return True
        if isinstance(exc, (GroqAPIStatusError, OpenAIAPIStatusError)):
            return getattr(exc, "status_code", None) == 429
        return False

    # --- rotation & cooldown -----------------------------------------

    def rotate(self, reason: str = "rate_limit") -> str:
        """
        Advance to the next un-burned key in cycle order.

        Marks the outgoing key as burned in the current cycle. If every
        key has now been burned, raises `AllKeysExhaustedError` and does
        NOT advance the pointer — the caller is expected to call
        `enter_cooldown()` explicitly rather than have the rotator
        silently sleep.
        """
        prev_idx = self._current_index
        self._burned_in_cycle.add(prev_idx)
        # Real 429 errors are counted whether the rotation succeeds or
        # signals exhaustion — the error happened either way. `key_rotations`
        # counts only *successful* rotations so it stays consistent with
        # `len(rotation_history)`.
        if reason == "rate_limit":
            self._rate_limit_errors += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if len(self._burned_in_cycle) >= len(self._keys):
            logger.warning(
                "[%s] Groq keys exhausted (cycle burned): %s reason=%s",
                ts, sorted(f"key_{i}" for i in self._burned_in_cycle), reason,
            )
            raise AllKeysExhaustedError(
                f"All {len(self._keys)} configured Groq keys have been rotated "
                f"away from in this cycle (reason={reason}). Caller should "
                f"invoke enter_cooldown() and retry."
            )

        # Walk to the next non-burned slot.
        next_idx = (prev_idx + 1) % len(self._keys)
        while next_idx in self._burned_in_cycle:
            next_idx = (next_idx + 1) % len(self._keys)
        self._current_index = next_idx

        self._rotations += 1
        ev = RotationEvent(
            ts=ts,
            from_masked=f"key_{prev_idx}",
            to_masked=f"key_{next_idx}",
            reason=reason,
        )
        self._history.append(ev)
        logger.info(
            "[%s] Groq key rotation: %s -> %s reason=%s",
            ts, ev.from_masked, ev.to_masked, reason,
        )
        return self.current()

    def enter_cooldown(self) -> None:
        """
        Explicit cooldown sleep after all keys were burned.

        Increments the cooldown-event counter, logs entry + exit, and
        after sleeping resets the burned-set, per-key counters, and
        pointer to `key_0`. Called ONLY from the eval client and only
        after `rotate()` has raised `AllKeysExhaustedError` — never
        implicitly.
        """
        self._cooldown_events += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.warning(
            "[%s] Groq all-keys-exhausted cooldown starting: sleeping %ds",
            ts, self._cooldown_s,
        )
        time.sleep(self._cooldown_s)
        self._burned_in_cycle.clear()
        self._per_key_calls = {i: 0 for i in range(len(self._keys))}
        self._current_index = 0
        logger.info(
            "[%s] cooldown complete. Reset to key_0.",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    # --- stats for the eval summary ----------------------------------

    def stats(self) -> dict:
        """Masked snapshot for EvalSummary. Never contains raw key strings."""
        return {
            "n_keys_configured": len(self._keys),
            "key_rotations": self._rotations,
            "per_key_call_counts": {
                f"key_{i}": self._per_key_calls[i] for i in range(len(self._keys))
            },
            "cooldown_events": self._cooldown_events,
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
