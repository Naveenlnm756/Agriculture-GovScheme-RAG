"""
Provider-agnostic vision adapter (Deliverable 2 B5).

One interface, one Gemini implementation. Provider swap is a single
config change (`settings.vision_provider` / `settings.vision_model`);
no other code touches the concrete SDK.

Two-stage pipeline (owner-locked, DECISIONS.md 2026-09-16):

  Stage 1 — CLASSIFY. Cheap. Returns one of
    TABLE / CHART / INFOGRAPHIC / OTHER / DECORATIVE
  A visual classified as OTHER or DECORATIVE is dropped BEFORE stage 2
  runs. The micro-benchmark showed most vision-only wins concentrate in
  image-embedded tables / infographics / charts; the stage-1 gate is
  there to burn the extraction budget only on those.

  Stage 2 — EXTRACT. Structured retrieval-oriented JSON with:
    visible_text, key_information (list of specific facts),
    important_numbers, dates, percentages, monetary values,
    labels, relationships.
  This is deliberately NOT a generic caption. The extraction prompt
  says "extract, do not paraphrase" — captions are noise for retrieval.

Retry policy (locked, DECISIONS.md):
  * attempt 1 at `settings.vision_default_dpi` (150)
  * attempt 2 at `settings.vision_fallback_dpi` (96) — smaller payload
    dodges the empty-response failure mode observed on
    `difficult_19/20` in the benchmark
  * if attempt 2 also fails, return VisionResult(vision_failed=True).
    The caller writes a placeholder chunk pointing at the PDF page.
    NEVER silently dropped.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

from src.config import settings


logger = logging.getLogger(__name__)


# --- Value objects ----------------------------------------------------------

# Classification values worth their own type so a typo is caught early.
VisionClass = Literal["TABLE", "CHART", "INFOGRAPHIC", "OTHER", "DECORATIVE"]
USEFUL_CLASSES: frozenset[VisionClass] = frozenset({"TABLE", "CHART", "INFOGRAPHIC"})


@dataclass(frozen=True)
class VisionResult:
    """
    Everything the pipeline needs from one full stage-1 + stage-2 pass.

    Fields:
      * `classification`   — stage 1 verdict. Always populated.
      * `description`      — 1-2 sentence caption. Present on stage-2 runs.
      * `visible_text`     — every readable text token in the image,
                             concatenated into one string, structure
                             preserved with newlines. Empty for DECORATIVE.
      * `key_information`  — list of short facts (eligibility conditions,
                             workflow steps, thresholds, formulas). Empty
                             for DECORATIVE / OTHER (no stage-2 run).
      * `important_numbers`, `dates`, `percentages`, `monetary_values` —
                             owner-requested extraction dimensions.
      * `labels`           — column/row/legend labels (for tables & charts).
      * `relationships`    — free-form strings describing trends /
                             correlations / axes (for charts & diagrams).
      * `vision_failed`    — True when BOTH attempts returned empty /
                             invalid. Chunk-materialisation writes a
                             placeholder in that case.
      * `error`            — human-readable last-error string when
                             vision_failed is True. None on happy path.
      * `attempt`          — which attempt (1 or 2) actually succeeded.
                             2 means the low-DPI retry saved us.
    """

    classification: VisionClass
    description: str = ""
    visible_text: str = ""
    key_information: list[str] = field(default_factory=list)
    important_numbers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    percentages: list[str] = field(default_factory=list)
    monetary_values: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    vision_failed: bool = False
    error: str | None = None
    attempt: int = 1

    def as_extraction_dict(self) -> dict:
        """
        Serialise the stage-2 payload into the shape the image chunker
        renders inside `[IMAGE]...[/IMAGE]`. Kept as a method rather
        than a __dict__ dump so we control field ordering / omitted
        fields explicitly.
        """
        return {
            "classification": self.classification,
            "description": self.description,
            "visible_text": self.visible_text,
            "key_information": self.key_information,
            "important_numbers": self.important_numbers,
            "dates": self.dates,
            "percentages": self.percentages,
            "monetary_values": self.monetary_values,
            "labels": self.labels,
            "relationships": self.relationships,
            "vision_failed": self.vision_failed,
            "attempt": self.attempt,
            "error": self.error,
        }


# --- Adapter contract -------------------------------------------------------

class VisionAdapter(Protocol):
    """
    Minimum contract for a vision provider. Two methods — the pipeline
    calls `classify()` first, then `extract()` only when the classifier
    returned a useful class. Both accept raw image bytes and a MIME type
    hint. Implementations own their own retry / rate-limit policy so
    the pipeline stays modality-agnostic.

    Rendering (page vs image-only, DPI, size) is the ADAPTER'S concern:
    it receives the bytes ready-to-send. The extractor upstream is
    responsible for producing the right rendering.
    """

    def classify(self, image_bytes: bytes, mime_type: str = "image/png") -> VisionClass: ...

    def extract(
        self,
        image_bytes: bytes,
        classification: VisionClass,
        mime_type: str = "image/png",
    ) -> VisionResult: ...

    @property
    def model_name(self) -> str: ...


# --- Prompts ----------------------------------------------------------------

# Stage 1 is deliberately terse and picks ONE word. Verbose classifiers
# tend to hedge ("mostly decorative but also a small chart") which
# breaks the drop-early gate.
_CLASSIFY_PROMPT = """You are triaging an image from a Government of India \
agriculture-scheme document. Assign exactly one label:

  TABLE        — a data table with rows and columns of values
  CHART        — a bar chart, line chart, pie chart, or graph
  INFOGRAPHIC  — a stats visual with numbered callouts and icons
                 (typical of PIB press releases)
  OTHER        — a diagram, flowchart, photograph, or document
                 scan with meaningful content that does not fit
                 the three above
  DECORATIVE   — a logo, seal, watermark, cover art, header/footer
                 graphic, or any image with no retrieval value

Respond with EXACTLY ONE UPPERCASE WORD, no punctuation, no preamble."""


# Stage 2 says "extract, do not paraphrase" for the same reason RAG
# grounding-prompts say "answer only from context": we want retrieval-
# usable tokens, not the model's editorial voice. The JSON shape mirrors
# `VisionResult` — one missing/renamed field breaks the parser.
_EXTRACT_PROMPT = """You are extracting retrieval-oriented information from \
an image inside a Government of India agriculture-scheme document. The \
image has been pre-classified as: {classification}.

Return ONLY a JSON object with EXACTLY these keys (no code fences, no \
preamble):

  "description":      1-2 sentence caption of what the image contains.
  "visible_text":     every readable text token in the image, joined
                      as one string with newlines preserving structure.
                      OCR-style — do NOT summarise.
  "key_information":  array of short strings capturing specific facts a
                      reader would need (eligibility conditions, subsidy
                      rates, scheme names, workflow steps, deadlines,
                      thresholds, formulas). Empty array if none.
  "important_numbers": array of strings, every non-percentage numeric
                      value (with units where visible).
  "dates":            array of strings, every calendar date visible.
  "percentages":      array of strings, every '%' value.
  "monetary_values":  array of strings, every rupee amount (₹ / Rs.)
                      or other currency.
  "labels":           array of strings — column, row, axis, and legend
                      labels present in the image.
  "relationships":    array of strings — trends, axes, or correlations
                      the image expresses (charts and diagrams only).
                      Empty array for tables / infographics.

Rules:
  * EXTRACT — do not paraphrase, summarise, or invent.
  * If a field has no content, use an empty array (or empty string for
    the two string fields).
  * Do not repeat the classification value in the output.
  * Do not include Markdown, code fences, or commentary."""


# --- Gemini implementation --------------------------------------------------

class GeminiVisionAdapter:
    """
    Concrete adapter for Google's `gemini-3.5-flash-lite` and siblings.

    Owns its own retry / rate-limit policy so callers stay
    modality-agnostic. Rate spacing between calls is enforced at the
    caller level (`vision_call_delay_s`) — this class does NOT sleep
    between successive calls, only during 429 backoff.

    The `_client` is lazy so importing this module does not require
    the SDK to be installed (useful for tests that don't touch vision).
    """

    def __init__(self, config=settings):
        self._config = config
        self._client = None  # lazy init

    @property
    def model_name(self) -> str:
        return self._config.vision_model

    # --- SDK plumbing ------------------------------------------------------

    def _ensure_client(self):
        if self._client is None:
            from google import genai  # deferred
            if not self._config.gemini_api_key:
                raise RuntimeError(
                    "gemini_api_key is empty in settings — set GEMINI_API_KEY in .env"
                )
            self._client = genai.Client(api_key=self._config.gemini_api_key)
        return self._client

    def _call(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        want_json: bool,
    ) -> str:
        """
        One raw call. Retries only 429/5xx with exponential backoff.
        Content-level failures (empty response, JSON parse) are handled
        by the caller — the RE-RENDER-AT-LOWER-DPI retry lives one level
        up because it requires re-rendering, which the caller owns.
        """
        from google import genai
        from google.genai import errors, types

        client = self._ensure_client()
        cfg = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json" if want_json else "text/plain",
        )
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ]

        max_transport_retries = 3
        backoff_base = 4.0
        for attempt in range(1, max_transport_retries + 1):
            try:
                resp = client.models.generate_content(
                    model=self._config.vision_model,
                    contents=contents,
                    config=cfg,
                )
                return (resp.text or "").strip()
            except errors.ClientError as e:
                if e.code == 429 and attempt < max_transport_retries:
                    sleep_s = backoff_base * (2 ** (attempt - 1))
                    logger.warning("vision 429; sleep %.1fs", sleep_s)
                    time.sleep(sleep_s)
                    continue
                raise
            except errors.ServerError:
                if attempt < max_transport_retries:
                    sleep_s = backoff_base * (2 ** (attempt - 1))
                    logger.warning("vision 5xx; sleep %.1fs", sleep_s)
                    time.sleep(sleep_s)
                    continue
                raise
        return ""

    # --- Public methods ----------------------------------------------------

    def classify(self, image_bytes: bytes, mime_type: str = "image/png") -> VisionClass:
        """
        Return one of the five VisionClass values. On empty response or
        junk, defaults to `"OTHER"` (safer than DECORATIVE — misclassifying
        a decorative as OTHER wastes a stage-2 call but preserves recall;
        misclassifying a real chart as DECORATIVE loses it forever).
        """
        raw = self._call(image_bytes, mime_type, _CLASSIFY_PROMPT, want_json=False)
        token = raw.upper().strip().strip(".,!?\"'")
        # Some models occasionally wrap the token in a sentence.
        for c in ("TABLE", "CHART", "INFOGRAPHIC", "DECORATIVE", "OTHER"):
            if c in token:
                return c  # type: ignore[return-value]
        logger.warning("classify: unrecognised response %r — defaulting to OTHER", raw[:80])
        return "OTHER"

    def extract(
        self,
        image_bytes: bytes,
        classification: VisionClass,
        mime_type: str = "image/png",
    ) -> VisionResult:
        """
        Stage 2. Called only when caller has already gated on a useful
        classification. Return a VisionResult with vision_failed=True
        if both the transport succeeded and the JSON was still empty /
        malformed. The caller decides whether to re-render at lower
        DPI and retry.
        """
        prompt = _EXTRACT_PROMPT.format(classification=classification)
        try:
            raw = self._call(image_bytes, mime_type, prompt, want_json=True)
        except Exception as e:
            return VisionResult(
                classification=classification,
                vision_failed=True,
                error=f"transport: {type(e).__name__}: {str(e)[:200]}",
            )

        if not raw.strip():
            return VisionResult(
                classification=classification,
                vision_failed=True,
                error="empty response body",
            )

        # Strip code fences if the model wrapped despite instructions.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            return VisionResult(
                classification=classification,
                vision_failed=True,
                error=f"json parse: {e}",
            )

        # Coerce arrays; tolerate a JSON object that omits keys.
        def _arr(key: str) -> list[str]:
            v = data.get(key)
            if isinstance(v, list):
                return [str(x) for x in v if x is not None]
            if v is None or v == "":
                return []
            return [str(v)]

        return VisionResult(
            classification=classification,
            description=str(data.get("description") or ""),
            visible_text=str(data.get("visible_text") or ""),
            key_information=_arr("key_information"),
            important_numbers=_arr("important_numbers"),
            dates=_arr("dates"),
            percentages=_arr("percentages"),
            monetary_values=_arr("monetary_values"),
            labels=_arr("labels"),
            relationships=_arr("relationships"),
        )


# --- Factory ----------------------------------------------------------------

def make_adapter(config=settings) -> VisionAdapter:
    """
    Resolve `settings.vision_provider` to a concrete adapter. Single
    swap point — if we add an alternative vision provider later, it
    lands here and everything downstream picks up the change through
    this factory.
    """
    provider = (config.vision_provider or "").lower()
    if provider == "gemini":
        return GeminiVisionAdapter(config)
    raise ValueError(
        f"unknown vision_provider={provider!r}. Supported: 'gemini'."
    )
