"""
Content-hash-keyed on-disk cache for vision results (Deliverable 2 B5).

One JSON sidecar per unique visual. Idempotent: get(sha) returns a
fully-hydrated `VisionResult` when the file exists, `None` otherwise.
Two rules:

  1. **Cache key is the SHA-256 (16-hex-char prefix) of the raw image
     bytes.** Same image bytes → same cache file regardless of which
     PDF / page it lives on. This is the load-bearing dedup mechanism:
     the NCCD watermark that appears on 393 pages hits the vision API
     ONCE for its entire lifetime, then every subsequent occurrence
     reuses the cached description.

  2. **A DECORATIVE or OTHER result is a valid cache entry.** We still
     write those files so a later run doesn't re-classify the same
     watermark. The image chunker upstream is responsible for skipping
     DECORATIVE at chunk-materialisation time; the cache does not
     filter.

The cache is safe to delete and rebuild — every entry is derived from
the raw image bytes plus a vision call, both reproducible. Never store
anything here that the pipeline can't regenerate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import settings
from src.vision.adapter import USEFUL_CLASSES, VisionClass, VisionResult


logger = logging.getLogger(__name__)


class VisionCache:
    """
    Thin wrapper around a directory of JSON sidecars keyed by
    content-hash. No LRU, no eviction, no in-memory tier — this cache
    is a *build artefact* (like the Chroma dir), sized by unique image
    count (< 1000 for this corpus) not by traffic.
    """

    def __init__(self, cache_dir: Path | None = None):
        self._dir = Path(cache_dir or settings.vision_cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # --- I/O helpers -------------------------------------------------------

    def _path(self, content_hash: str) -> Path:
        # Basic guard: only [0-9a-f] chars, up to 64 chars. If the caller
        # ever passes something like a full SHA-1 or a random string we
        # want the mismatch to be visible, not to silently write to a
        # weirdly-named file.
        if not content_hash or not all(c in "0123456789abcdef" for c in content_hash):
            raise ValueError(f"invalid content_hash: {content_hash!r}")
        return self._dir / f"{content_hash}.json"

    # --- Public API --------------------------------------------------------

    def get(self, content_hash: str) -> VisionResult | None:
        """
        Return the cached VisionResult, or None on cache miss.
        Silently returns None on corrupt / unreadable files — the
        caller will then re-run vision and overwrite the bad entry.
        """
        p = self._path(content_hash)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cache file %s unreadable (%s); treating as miss", p.name, exc)
            return None

        cls_raw = data.get("classification")
        if cls_raw not in {"TABLE", "CHART", "INFOGRAPHIC", "OTHER", "DECORATIVE"}:
            logger.warning(
                "cache file %s has bad classification=%r; treating as miss",
                p.name, cls_raw,
            )
            return None

        return VisionResult(
            classification=cls_raw,  # type: ignore[arg-type]
            description=str(data.get("description") or ""),
            visible_text=str(data.get("visible_text") or ""),
            key_information=list(data.get("key_information") or []),
            important_numbers=list(data.get("important_numbers") or []),
            dates=list(data.get("dates") or []),
            percentages=list(data.get("percentages") or []),
            monetary_values=list(data.get("monetary_values") or []),
            labels=list(data.get("labels") or []),
            relationships=list(data.get("relationships") or []),
            vision_failed=bool(data.get("vision_failed") or False),
            error=data.get("error"),
            attempt=int(data.get("attempt") or 1),
        )

    def put(self, content_hash: str, result: VisionResult) -> None:
        """Write the sidecar. Atomic-ish (write-then-rename) so a
        concurrent reader never sees a truncated JSON. Concurrent
        writers aren't expected — the ingestion runner is single-
        process — but the rename discipline costs nothing."""
        p = self._path(content_hash)
        tmp = p.with_suffix(".json.tmp")
        payload = result.as_extraction_dict()
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def has(self, content_hash: str) -> bool:
        """Cheap existence check without deserialising."""
        return self._path(content_hash).exists()

    def is_useful_cached(self, content_hash: str) -> bool:
        """True iff cached AND classification is TABLE / CHART / INFOGRAPHIC.
        Useful for the runner to count "chunk-eligible" entries without
        materialising the full result."""
        r = self.get(content_hash)
        return r is not None and not r.vision_failed and r.classification in USEFUL_CLASSES
