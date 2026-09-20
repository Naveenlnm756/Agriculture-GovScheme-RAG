"""
Deterministic query validation (C2).

Owner-locked (2026-09-16): production must not use an LLM to gate,
rewrite, or classify user queries. This module is the ONLY thing
allowed to reject a query before it enters the retrieval pipeline,
and its rules are:

  1. **Empty / whitespace-only.** Reject with a "please enter a
     question" template.
  2. **Non-UTF8 / control characters.** Reject with a "unusable
     input" template. Control chars can bleed into logs and
     downstream prompts; we drop them at the door.
  3. **Length ceiling.** Reject queries above `MAX_QUERY_CHARS`
     with a "too long" template. The ceiling is high enough
     (2000 chars) that every reasonable question passes, but
     bounded enough that a copy-paste jailbreak dump gets stopped.
  4. **Prompt-injection heuristics.** Delegated to
     `src.production.guardrails.detect_prompt_injection`.

Everything else (length lower-bound, punctuation, language) passes
through — the retrieval side already tolerates short / oddly-formed
queries (the ablation malformed_query bucket scored hit@5 = 0.90
without any rewriting).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.production.guardrails import (
    InjectionCheckResult,
    detect_prompt_injection,
)


logger = logging.getLogger(__name__)


# --- Tunables (stated up-front) ---------------------------------------------

# Hard cap on a query. 2000 chars is ~500 tokens — well above any
# reasonable natural question and below the point where a paste
# attack could smuggle a fake context / instruction block.
MAX_QUERY_CHARS: int = 2000

# Minimum characters to bother running through retrieval. Two chars
# is enough to reject "?" / "hi" / whitespace-only but low enough
# that legit acronym queries (KCC, PMFBY) pass. Tuned to the
# malformed_query golden-set bucket.
MIN_QUERY_CHARS: int = 2


# --- Regex for control-character strip --------------------------------------

# ASCII control chars 0x00-0x1F EXCEPT tab (0x09), newline (0x0A),
# and carriage return (0x0D) — those three are legal whitespace and
# preserved. Everything else (backspace, form-feed, escape, etc.) is
# suspicious in a chat / API surface.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


# --- Refusal templates -----------------------------------------------------

# Kept as module constants so a future edit updates every callsite
# at once. The FastAPI response type surfaces these to the frontend.

REFUSAL_EMPTY = (
    "Please enter a question about a Government of India agriculture "
    "scheme (PM-KISAN, PMFBY, KCC, SMAM, MIDH, NFSM, AIF)."
)
REFUSAL_TOO_LONG = (
    "Your question is too long (over {max_chars} characters). Please "
    "shorten it — most questions fit in a sentence or two."
)
REFUSAL_TOO_SHORT = (
    "Your question is too short. Please add a few more words about what "
    "you want to know."
)
REFUSAL_CONTROL_CHARS = (
    "Your input contained non-printable characters and was rejected. "
    "Please re-enter the question as plain text."
)
REFUSAL_INJECTION = (
    "Your query appears to contain prompt-override language and was "
    "rejected. Please rephrase as a plain question about the corpus."
)


# --- Value object -----------------------------------------------------------

@dataclass(frozen=True)
class QueryValidation:
    """
    Outcome of `validate_query`. On the happy path `is_valid=True`
    and `cleaned_query` carries the query with control chars stripped
    (usually the input verbatim). On rejection, `is_valid=False` plus
    a `refusal_reason` machine-readable tag and `refusal_message`
    ready to show the user.
    """

    is_valid: bool
    cleaned_query: str = ""
    refusal_reason: str | None = None
    refusal_message: str | None = None
    # Populated when the injection guardrail fires — surfaced for
    # observability (which pattern matched, what span). Never rendered
    # to the user; logged only.
    injection_check: InjectionCheckResult | None = None


# --- Entry point -----------------------------------------------------------

def validate_query(query: str) -> QueryValidation:
    """
    Deterministic checks against a user query. Returns a
    `QueryValidation` — inspect `is_valid` before dispatching to the
    retrieval pipeline.

    Order of checks matters. We reject on the most specific /
    cheapest fault first so the user gets a precise error and the
    log surface names the actual problem, not a generic "invalid".

    The check order:

      1. `query is None`  →  REFUSAL_EMPTY (nothing to work with).
      2. Strip control chars. If any were present, REFUSAL_CONTROL_CHARS.
      3. Strip / normalise whitespace. If empty afterwards, REFUSAL_EMPTY.
      4. Length > `MAX_QUERY_CHARS`  →  REFUSAL_TOO_LONG.
      5. Length < `MIN_QUERY_CHARS`  →  REFUSAL_TOO_SHORT.
      6. Prompt-injection heuristic  →  REFUSAL_INJECTION.
    """
    if query is None:
        return QueryValidation(
            is_valid=False,
            refusal_reason="empty",
            refusal_message=REFUSAL_EMPTY,
        )

    if _CONTROL_CHARS_RE.search(query):
        # Strip the control chars for logging safety, then reject.
        stripped = _CONTROL_CHARS_RE.sub("", query)
        logger.warning(
            "query rejected: control chars stripped=%d chars",
            len(query) - len(stripped),
        )
        return QueryValidation(
            is_valid=False,
            refusal_reason="control_chars",
            refusal_message=REFUSAL_CONTROL_CHARS,
        )

    cleaned = query.strip()

    if not cleaned:
        return QueryValidation(
            is_valid=False,
            refusal_reason="empty",
            refusal_message=REFUSAL_EMPTY,
        )

    if len(cleaned) > MAX_QUERY_CHARS:
        return QueryValidation(
            is_valid=False,
            refusal_reason="too_long",
            refusal_message=REFUSAL_TOO_LONG.format(max_chars=MAX_QUERY_CHARS),
        )

    if len(cleaned) < MIN_QUERY_CHARS:
        return QueryValidation(
            is_valid=False,
            refusal_reason="too_short",
            refusal_message=REFUSAL_TOO_SHORT,
        )

    # Prompt-injection check runs LAST because it's the most expensive
    # and the least common failure mode. If we've reached this point
    # the query passed every deterministic length + character check.
    injection = detect_prompt_injection(cleaned)
    if injection.is_injection:
        return QueryValidation(
            is_valid=False,
            refusal_reason="prompt_injection",
            refusal_message=REFUSAL_INJECTION,
            injection_check=injection,
        )

    return QueryValidation(
        is_valid=True,
        cleaned_query=cleaned,
    )
