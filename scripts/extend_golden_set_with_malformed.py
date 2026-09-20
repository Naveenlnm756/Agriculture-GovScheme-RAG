"""
Extend eval/golden_set.json with 10 malformed_query entries derived
from existing clean questions.

Each new entry preserves the source's expected_answer and
expected_sources verbatim (the answer is the same; only the query
phrasing changes). New field: original_source_qid, so a diagnostic
can always tie a malformed row back to its clean baseline.

Design rules for the malformations (per owner-locked criteria):
  * Roman-transliterated Hindi ("kaise", "kya", "kitna"), NOT
    caricature ("please sir how apply pm kisan sir").
  * Code-switched English + Hindi mid-sentence, as real users type.
  * Telegraphic phrasing — missing articles/question words.
  * Abbreviations and lowercase throughout, no punctuation.
  * A REALISTIC MIX — pure English telegraphic, code-switched, and
    Hindi-heavy — so the malformed_query bucket reflects the range
    of real-user query shapes rather than a single style.

Categories sourced from (per owner's guidance):
  simple_procedural   : 5
  simple_factual      : 3
  multi_hop_scheme    : 2
Total: 10 new entries → golden set grows from 78 to 88.

Also updates:
  * `categories` block: adds `malformed_query` description.
  * `target_counts` block: adds `malformed_query: 10`.

Writes back to eval/golden_set.json. Makes a timestamped backup at
eval/golden_set.json.bak-<ts> before overwriting.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
GS_PATH = _PROJECT_ROOT / "eval" / "golden_set.json"


# --- Malformed queries — realistic user-style, not caricature ---
#
# Style annotation on each entry tells the reader which malformation
# pattern is exercised, so the diagnostic_purpose reads honestly.

MALFORMED: list[dict] = [
    # ---- simple_procedural (5) ----
    {
        "new_qid": "Q079",
        "source_qid": "Q003",
        "question": "pm kisan me naya registration kaise kare",
        "style": "Hindi-heavy transliteration",
    },
    {
        "new_qid": "Q080",
        "source_qid": "Q004",
        "question": "pmkisan ekyc process",
        "style": "English telegraphic, no question word",
    },
    {
        "new_qid": "Q081",
        "source_qid": "Q008",
        "question": "kcc online apply kaise",
        "style": "code-switched English + Hindi",
    },
    {
        "new_qid": "Q082",
        "source_qid": "Q005",
        "question": "pm kisan benefit chodna hai kya karu",
        "style": "Hindi-heavy, colloquial",
    },
    {
        "new_qid": "Q083",
        "source_qid": "Q012",
        "question": "smam single implement subsidy kaise milegi",
        "style": "code-switched, missing scheme spelling",
    },
    # ---- simple_factual (3) ----
    {
        "new_qid": "Q084",
        "source_qid": "Q013",
        "question": "pmfby kharif farmer premium kitna",
        "style": "code-switched telegraphic, no question word",
    },
    {
        "new_qid": "Q085",
        "source_qid": "Q018",
        "question": "pmfby hailstorm loss reporting time limit",
        "style": "English telegraphic keyword-style",
    },
    {
        "new_qid": "Q086",
        "source_qid": "Q017",
        "question": "miss interest subvention rate 2025 26",
        "style": "pure English telegraphic, keyword-only",
    },
    # ---- multi_hop_scheme (2) ----
    {
        "new_qid": "Q087",
        "source_qid": "Q031",
        "question": "pmfby olay crop damage claim procedure kitne din",
        "style": "heavy code-switch, colloquial 'olay' for hailstorm",
    },
    {
        "new_qid": "Q088",
        "source_qid": "Q032",
        "question": "kcc collateral free limit rbi 2026 vs 2017 farak",
        "style": "keyword-style comparison query, Hindi 'farak' for difference",
    },
]


def main() -> int:
    if not GS_PATH.exists():
        print(f"Missing: {GS_PATH}")
        return 2

    gs = json.loads(GS_PATH.read_text(encoding="utf-8"))

    by_qid = {q["question_id"]: q for q in gs["questions"]}

    # Backup
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = GS_PATH.with_name(f"golden_set.json.bak-{ts}")
    shutil.copy2(GS_PATH, backup)
    print(f"[backup] {backup}")

    # Build new entries
    new_entries: list[dict] = []
    for m in MALFORMED:
        src = by_qid.get(m["source_qid"])
        if src is None:
            print(f"[error] source_qid {m['source_qid']!r} not found in golden set")
            return 3
        new_entries.append({
            "question_id": m["new_qid"],
            "category": "malformed_query",
            "question": m["question"],
            "expected_answer": src["expected_answer"],
            "expected_sources": src["expected_sources"],
            "original_source_qid": m["source_qid"],
            "diagnostic_purpose": (
                f"Malformed-user-query variant of {m['source_qid']} "
                f"({src['category']}). Style: {m['style']}. "
                f"Tests retrieval robustness to real-user query shapes "
                f"— colloquial, telegraphic, code-switched — against "
                f"the same expected sources as the clean baseline."
            ),
        })

    # Extend questions list — keep existing 78 first, append the 10 new
    gs["questions"] = list(gs["questions"]) + new_entries

    # Update categories block
    if "categories" not in gs:
        gs["categories"] = {}
    gs["categories"]["malformed_query"] = (
        "Realistic user-style malformed variant of a clean question "
        "(same expected answer + sources). Tests retrieval robustness "
        "to Hindi transliteration, code-switching, telegraphic phrasing, "
        "and missing question words."
    )

    # Update target_counts
    if "target_counts" not in gs:
        gs["target_counts"] = {}
    gs["target_counts"]["malformed_query"] = len(new_entries)

    # Write back
    GS_PATH.write_text(
        json.dumps(gs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {GS_PATH}")
    print(f"[write] total questions now: {len(gs['questions'])}")
    print(f"[write] new: {[e['question_id'] for e in new_entries]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
