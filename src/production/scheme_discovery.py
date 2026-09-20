"""
Structured scheme-discovery layer (Deliverable D — refactored to v2).

Scope.md §3's "Structured layer": deterministic filtering over farmer
attributes → list of schemes that are POTENTIALLY APPLICABLE. No LLM
on the query path. Eligibility JSON at
`data/scheme_eligibility.json` is authored + reviewed by the owner;
every rule carries its own source doc + page for auditability.

Refinements versus v1 (owner refinement 2026-09-20):

  1. **Per-rule provenance.** `applicability_rules`, `disqualifying_rules`,
     and `match_boost_if` each carry a `source: {filename, page, note}`.
     Reasons rendered to the user cite the doc + page — the structured
     layer is as citable as the RAG layer.
  2. **Intent detector is a SIGNAL, not a decision.** The runner is
     expected to also run RAG. Discovery fires whenever intent is
     detected OR the user provided ANY explicit attribute (so
     compound queries like "SC farmer with wheat in Punjab, what is
     the KCC interest rate?" surface schemes AND factual RAG).
  3. **Explicit vs unknown is first-class.** The extractor returns
     `"unknown"` for attributes the user didn't mention; the filter
     treats those as "we don't know, don't disqualify"; the response
     surfaces `provided_information` + `missing_information` so the
     user sees exactly what we assumed.
  4. **"Potentially applicable" language.** We never render "eligible"
     because we cannot confirm eligibility without every required
     attribute. `required_information` lists what we still need to
     know per scheme.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from src.config import settings


logger = logging.getLogger(__name__)


# --- Value objects ----------------------------------------------------------

@dataclass(frozen=True)
class FarmerAttributes:
    """
    Bag of attributes extracted from a single query. Every field
    defaults to `"unknown"` — a missing value NEVER filters a scheme
    out; only an explicit value that matches a `disqualifying_rules`
    entry can filter.
    """
    has_land: object = "unknown"       # True / False / "unknown"
    occupation: str = "unknown"        # "farmer" / "unknown"
    land_holding: str = "unknown"
    category: str = "unknown"
    gender: str = "unknown"
    region: str = "unknown"
    crop_group: str = "unknown"
    interests: frozenset[str] = field(default_factory=frozenset)

    def any_explicit(self) -> bool:
        """True iff the user gave us ANY attribute we could parse."""
        return (
            self.has_land != "unknown"
            or self.occupation != "unknown"
            or self.land_holding != "unknown"
            or self.category != "unknown"
            or self.gender != "unknown"
            or self.region != "unknown"
            or self.crop_group != "unknown"
            or bool(self.interests)
        )

    def provided_information(self) -> list[dict]:
        """Explicit attributes only — for surfacing back to the user."""
        out: list[dict] = []
        if self.has_land != "unknown":
            out.append({"attribute": "has_land", "value": self.has_land})
        if self.occupation != "unknown":
            out.append({"attribute": "occupation", "value": self.occupation})
        if self.land_holding != "unknown":
            out.append({"attribute": "land_holding", "value": self.land_holding})
        if self.category != "unknown":
            out.append({"attribute": "category", "value": self.category})
        if self.gender != "unknown":
            out.append({"attribute": "gender", "value": self.gender})
        if self.region != "unknown":
            out.append({"attribute": "region", "value": self.region})
        if self.crop_group != "unknown":
            out.append({"attribute": "crop_group", "value": self.crop_group})
        if self.interests:
            out.append({
                "attribute": "interests", "value": sorted(self.interests),
            })
        return out


@dataclass
class MatchedReason:
    """One human-readable reason a scheme surfaced, tied to a source."""
    attribute: str
    matched_value: object
    note: str
    source_filename: str | None
    source_page: int | None


@dataclass
class MissingInformation:
    """One attribute we would need to make an eligibility judgement."""
    attribute: str
    why_needed: str


@dataclass
class SchemeMatch:
    """
    One scheme that is POTENTIALLY APPLICABLE to the user based on
    what they told us.

    `applicability_reasons` cite the attributes + docs that produced
    the match. `boost_reasons` cite the soft signals that raised it in
    the ranking. `missing_information` lists what we still need to
    know to confirm eligibility.
    """
    code: str
    name: str
    long_name: str
    category: str
    one_liner: str
    key_benefits: list[str]
    who_should_apply: str
    how_to_register: dict
    follow_up_query_hint: str
    authoritative_source: dict
    applicability_reasons: list[MatchedReason]
    boost_reasons: list[MatchedReason]
    missing_information: list[MissingInformation]
    always_included: bool
    boost_hits: int


@dataclass
class DiscoveryResult:
    """The full structured-layer output for one query."""
    intent_detected: bool
    provided_information: list[dict]
    missing_information_summary: list[str]
    matches: list[SchemeMatch]


# --- Intent detection (SIGNAL, not gate) ------------------------------------

_INTENT_PATTERNS: tuple[str, ...] = (
    r"which\s+schemes?",
    r"what\s+schemes?",
    r"help\s+me\s+find\s+(?:a\s+)?scheme",
    r"scheme(?:s)?\s+(?:for\s+me|can\s+i|am\s+i\s+eligible)",
    r"i\s+can\s+register(?:\s+for)?",
    r"can\s+i\s+register(?:\s+for)?",
    r"register\s+as\s+(?:a\s+)?farmer",
    r"eligible\s+for\s+what",
    r"any\s+schemes?",
    r"list\s+(?:of\s+)?schemes?",
    r"suggest\s+(?:a\s+)?scheme",
    r"recommend\s+(?:a\s+)?scheme",
    r"what\s+can\s+i\s+get",
    r"am\s+i\s+eligible",
)

_INTENT_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INTENT_PATTERNS]


def detect_scheme_discover_intent(query: str) -> bool:
    """True iff the query looks like 'which scheme(s) can/should I ...'.
    The runner treats this as a routing signal — even a `False` result
    still runs discovery if the user provided attributes."""
    if not query:
        return False
    return any(rgx.search(query) for rgx in _INTENT_COMPILED)


# --- Attribute extraction ---------------------------------------------------

_LAND_BUCKET_KEYWORDS = {
    "marginal": "marginal",
    "small farmer": "small",
    "small holding": "small",
    "small scale": "small",
    "semi medium": "semi_medium",
    "semi-medium": "semi_medium",
    "medium farmer": "medium",
    "medium scale": "medium",
    "large farmer": "large",
    "large holding": "large",
    "big farmer": "large",
}

_NUMERIC_LAND_RE = re.compile(
    r"(?<!\d)(\d+(?:\.\d+)?)\s*(hectare[s]?|ha\b|acre[s]?|bigha[s]?)",
    re.IGNORECASE,
)


def _bucket_from_hectares(ha: float) -> str:
    """Indian Agri Census landholding classification."""
    if ha < 1.0:
        return "marginal"
    if ha < 2.0:
        return "small"
    if ha < 4.0:
        return "semi_medium"
    if ha < 10.0:
        return "medium"
    return "large"


def _extract_land_holding(q_lower: str) -> str:
    """Numeric mentions override named ones when both are present."""
    m = _NUMERIC_LAND_RE.search(q_lower)
    if m:
        num = float(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("acre"):
            ha = num / 2.47
        elif unit.startswith("bigha"):
            ha = num / 3.95
        else:
            ha = num
        return _bucket_from_hectares(ha)

    for kw, bucket in _LAND_BUCKET_KEYWORDS.items():
        if kw in q_lower:
            return bucket
    return "unknown"


_CATEGORY_PATTERNS = {
    "SC": [r"\bsc\b", r"scheduled\s+caste", r"dalit"],
    "ST": [r"\bst\b", r"scheduled\s+tribe", r"tribal", r"adivasi"],
    "OBC": [r"\bobc\b", r"other\s+backward"],
    "general": [r"\bgeneral\s+category\b"],
}


def _extract_category(query: str) -> str:
    for label, pats in _CATEGORY_PATTERNS.items():
        for pat in pats:
            if re.search(pat, query, re.IGNORECASE):
                return label
    return "unknown"


_GENDER_PATTERNS = {
    "woman": [r"\bwoman\b", r"\bwomen\b", r"\bfemale\b", r"\bmahila\b"],
    "man": [r"\bmale\s+farmer\b"],
}


def _extract_gender(query: str) -> str:
    for label, pats in _GENDER_PATTERNS.items():
        for pat in pats:
            if re.search(pat, query, re.IGNORECASE):
                return label
    return "unknown"


_NE_STATES = (
    "arunachal pradesh", "assam", "manipur", "meghalaya", "mizoram",
    "nagaland", "sikkim", "tripura", "north east", "north-east", "ne states",
)
_HIMALAYAN_STATES = (
    "jammu", "kashmir", "j&k", "ladakh", "uttarakhand", "himachal pradesh",
    "himalayan",
)


def _extract_region(query: str) -> str:
    q_lower = query.lower()
    for hint in _NE_STATES:
        if hint in q_lower:
            return "north_east"
    for hint in _HIMALAYAN_STATES:
        if hint in q_lower:
            return "himalayan"
    return "unknown"


_CROP_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("rice", "rice"), ("paddy", "rice"),
    ("wheat", "wheat"),
    ("pulse", "pulses"), ("dal", "pulses"), ("lentil", "pulses"),
    ("coarse cereal", "coarse_cereals"),
    ("millet", "nutri_cereals"), ("bajra", "nutri_cereals"),
    ("jowar", "nutri_cereals"), ("ragi", "nutri_cereals"),
    ("nutri cereal", "nutri_cereals"),
    ("oilseed", "oilseeds"), ("mustard", "oilseeds"),
    ("soybean", "oilseeds"), ("groundnut", "oilseeds"),
    ("sunflower", "oilseeds"),
    ("sugarcane", "sugarcane"),
    ("horticulture", "horticulture"),
    ("fruit", "fruit"), ("mango", "fruit"), ("banana", "fruit"),
    ("vegetable", "vegetable"), ("tomato", "vegetable"), ("onion", "vegetable"),
    ("flower", "flower"),
    ("spice", "spice"),
    ("medicinal", "medicinal"),
    ("cotton", "commercial"), ("tea", "commercial"), ("coffee", "commercial"),
)


def _extract_crop_group(query: str) -> str:
    q_lower = query.lower()
    for kw, group in _CROP_KEYWORDS:
        if kw in q_lower:
            return group
    return "unknown"


_INTEREST_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("insurance", "insurance"), ("insure", "insurance"),
    ("credit", "credit"), ("loan", "credit"), ("kcc", "credit"),
    ("income support", "income_support"), ("dbt", "income_support"),
    ("equipment", "equipment"),
    ("machinery", "equipment"), ("machin", "equipment"),
    ("mechanisation", "mechanization"), ("mechanization", "mechanization"),
    ("rotavator", "equipment"), ("tractor", "equipment"),
    ("custom hiring", "custom_hiring"), ("chc ", "custom_hiring"),
    ("cold storage", "cold_storage"), ("cold-storage", "cold_storage"),
    ("cold chain", "cold_storage"),
    ("post harvest", "post_harvest"), ("post-harvest", "post_harvest"),
    ("processing", "processing"),
    ("warehouse", "warehouse"),
    ("infrastructure", "infrastructure"),
    ("organic", "organic"),
    ("food security", "food_security"),
    ("horticulture", "horticulture"),
)


def _extract_interests(query: str) -> frozenset[str]:
    q_lower = query.lower()
    hits: set[str] = set()
    for kw, tag in _INTEREST_KEYWORDS:
        if kw in q_lower:
            hits.add(tag)
    return frozenset(hits)


def _extract_has_land_and_occupation(
    query: str, land_bucket: str,
) -> tuple[object, str]:
    q_lower = query.lower()
    has_land: object = "unknown"
    occupation = "unknown"
    if "farmer" in q_lower or "kisan" in q_lower or "farming" in q_lower:
        occupation = "farmer"
    if land_bucket != "unknown":
        has_land = True
        if occupation == "unknown":
            occupation = "farmer"
    if "landless" in q_lower or "no land" in q_lower:
        has_land = False
    return has_land, occupation


def extract_attributes(query: str) -> FarmerAttributes:
    """Deterministic extraction. Empty input → all fields "unknown"."""
    if not query:
        return FarmerAttributes()
    q_lower = query.lower()
    land_bucket = _extract_land_holding(q_lower)
    has_land, occupation = _extract_has_land_and_occupation(query, land_bucket)
    return FarmerAttributes(
        has_land=has_land,
        occupation=occupation,
        land_holding=land_bucket,
        category=_extract_category(query),
        gender=_extract_gender(query),
        region=_extract_region(query),
        crop_group=_extract_crop_group(query),
        interests=_extract_interests(query),
    )


# --- Scheme filter -----------------------------------------------------------

_SCHEMES_DATA: dict | None = None


def _load_schemes() -> list[dict]:
    """Read `data/scheme_eligibility.json` once. Restart to reload."""
    global _SCHEMES_DATA
    if _SCHEMES_DATA is None:
        path = Path(settings.data_raw_dir).parent / "scheme_eligibility.json"
        _SCHEMES_DATA = json.loads(path.read_text(encoding="utf-8"))
        n = len(_SCHEMES_DATA.get("schemes", []))
        logger.info("scheme_discovery: loaded %d schemes from %s", n, path)
    return _SCHEMES_DATA["schemes"]


def _attr_value(attrs: FarmerAttributes, name: str) -> object:
    return getattr(attrs, name, "unknown")


def _rule_matches(rule: dict, attrs: FarmerAttributes) -> tuple[bool, object]:
    """
    Evaluate one rule. Returns (matches, matched_value_for_reason).

    Semantics:
      * "unknown" in accepted values → matches ANY value INCLUDING
        "unknown" (permissive on missing info).
      * For `interests` (set attribute), matches when ANY interest is
        in accepted, OR when "unknown" is accepted AND the set is empty.
      * The reason value returned is the CONCRETE user value that
        matched (or "unknown" if permissive-match on missing data),
        for downstream reason-rendering.
    """
    name = rule.get("attribute")
    accepted = set(rule.get("values", []))
    user_val = _attr_value(attrs, name)

    if name == "interests":
        if isinstance(user_val, (set, frozenset)):
            overlap = user_val & accepted
            if overlap:
                return True, sorted(overlap)
            if "unknown" in accepted and not user_val:
                return True, "unknown"
            return False, None
        return False, None

    if user_val in accepted:
        return True, user_val
    return False, None


def _all_pass(rules: Iterable[dict], attrs: FarmerAttributes) -> bool:
    return all(_rule_matches(r, attrs)[0] for r in rules)


def _any_fires(rules: Iterable[dict], attrs: FarmerAttributes) -> bool:
    return any(_rule_matches(r, attrs)[0] for r in rules)


def _to_matched_reason(rule: dict, matched_value: object) -> MatchedReason:
    src = rule.get("source") or {}
    return MatchedReason(
        attribute=rule.get("attribute", "?"),
        matched_value=matched_value,
        note=src.get("note", ""),
        source_filename=src.get("filename"),
        source_page=src.get("page"),
    )


def _collect_applicability_reasons(
    scheme: dict, attrs: FarmerAttributes,
) -> list[MatchedReason]:
    """Reason list from `applicability_rules` that matched EXPLICIT
    user info. We hide "unknown"-matched rules from the user because
    they're not evidence FOR the scheme; they're only evidence that
    we couldn't disprove it."""
    reasons: list[MatchedReason] = []
    for rule in scheme.get("applicability_rules", []):
        ok, matched = _rule_matches(rule, attrs)
        if ok and matched != "unknown":
            reasons.append(_to_matched_reason(rule, matched))
    return reasons


def _collect_boost_reasons(
    scheme: dict, attrs: FarmerAttributes,
) -> list[MatchedReason]:
    reasons: list[MatchedReason] = []
    for rule in scheme.get("match_boost_if", []):
        ok, matched = _rule_matches(rule, attrs)
        if ok and matched != "unknown":
            reasons.append(_to_matched_reason(rule, matched))
    return reasons


def _missing_information_for(
    scheme: dict, attrs: FarmerAttributes,
) -> list[MissingInformation]:
    """List of `required_information` entries whose attribute is not
    explicitly provided. Interests as a set: missing iff empty."""
    out: list[MissingInformation] = []
    for req in scheme.get("required_information", []):
        name = req.get("attribute")
        user_val = _attr_value(attrs, name)
        is_missing = (
            (isinstance(user_val, (set, frozenset)) and not user_val)
            or user_val == "unknown"
        )
        if is_missing:
            out.append(MissingInformation(
                attribute=name,
                why_needed=req.get("why_needed", ""),
            ))
    return out


def find_matching_schemes(attrs: FarmerAttributes) -> list[SchemeMatch]:
    """
    Filter the eligibility JSON against extracted attributes.

    A scheme is INCLUDED when:
      * every `applicability_rules` entry matches (missing user info
        counts as a match if the rule lists "unknown"), AND
      * no `disqualifying_rules` entry fires against EXPLICIT user
        info (missing info never disqualifies), OR
      * the scheme is `always_include=true` AND the user gave no
        explicit attributes.

    Ranking: `boost_hits` descending; ties by JSON order (canonical).
    """
    matches: list[SchemeMatch] = []
    schemes = _load_schemes()
    any_explicit = attrs.any_explicit()

    for scheme in schemes:
        apply_rules = scheme.get("applicability_rules", [])
        disqual = scheme.get("disqualifying_rules", [])
        always = bool(scheme.get("always_include", False))

        if _any_fires(disqual, attrs):
            continue

        applies = _all_pass(apply_rules, attrs)
        always_included_now = always and not any_explicit

        if not applies and not always_included_now:
            continue

        app_reasons = _collect_applicability_reasons(scheme, attrs)
        boost_reasons = _collect_boost_reasons(scheme, attrs)
        missing = _missing_information_for(scheme, attrs)

        matches.append(SchemeMatch(
            code=scheme["code"],
            name=scheme["name"],
            long_name=scheme["long_name"],
            category=scheme["category"],
            one_liner=scheme["one_liner"],
            key_benefits=list(scheme.get("key_benefits", [])),
            who_should_apply=scheme.get("who_should_apply", ""),
            how_to_register=dict(scheme.get("how_to_register", {})),
            follow_up_query_hint=scheme.get("follow_up_query_hint", ""),
            authoritative_source=dict(scheme.get("authoritative_source", {})),
            applicability_reasons=app_reasons,
            boost_reasons=boost_reasons,
            missing_information=missing,
            always_included=always_included_now,
            boost_hits=len(boost_reasons),
        ))

    matches.sort(key=lambda m: (-m.boost_hits, m.code))
    return matches


# --- Runner-facing convenience ---------------------------------------------

def run_scheme_discovery(query: str) -> DiscoveryResult | None:
    """
    Public entrypoint. Returns a `DiscoveryResult` when discovery
    should fire (intent detected OR any explicit attribute), else
    `None`. The API surface calls this ALONGSIDE the RAG pipeline
    on every query, and stitches both results into one response.

    Language contract: the result never says "eligible". The API
    layer renders "potentially applicable" phrasing per the schema
    v2 rules.
    """
    if not query:
        return None

    attrs = extract_attributes(query)
    intent = detect_scheme_discover_intent(query)

    if not intent and not attrs.any_explicit():
        return None

    matches = find_matching_schemes(attrs)

    # Aggregate missing_information across all matches (dedup by
    # attribute; keep the first why_needed we see for each).
    seen: set[str] = set()
    summary: list[str] = []
    for m in matches:
        for mi in m.missing_information:
            if mi.attribute in seen:
                continue
            seen.add(mi.attribute)
            summary.append(f"{mi.attribute}: {mi.why_needed}")

    return DiscoveryResult(
        intent_detected=intent,
        provided_information=attrs.provided_information(),
        missing_information_summary=summary,
        matches=matches,
    )
