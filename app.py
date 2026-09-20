"""
Streamlit frontend for the Agri Schemes RAG — with persistent chat.

What this file owns:

  * Chat surface — `st.chat_message` / `st.chat_input` with sidebar
    conversation list, New Chat, rename, delete.
  * Persistence — every turn is written to SQLite via
    `src.chat.store.ChatStore`. Conversations survive page reloads,
    container restarts, and redeployments (the DB lives on the Fly
    volume, or under `./data/agri_rag.db` locally).
  * Rendering — scheme-discovery cards, citations, retrieved-chunks
    debug view. Same rich rendering as before; now scoped per-turn so
    reloading an old conversation shows exactly what the user saw
    originally (payload JSON is stored alongside each assistant
    message).

What this file deliberately does NOT do:

  * Talk to Chroma directly. All retrieval + generation goes through
    the FastAPI `/query` endpoint at AGRI_API_URL. Frontend / backend
    stay decoupled: the browser never sees the RAG pipeline.
  * Pass prior chat turns to the retriever. The RAG pipeline is
    stateless by design (owner decision D1, 2026-09-20). Each
    question triggers fresh retrieval. Loading an old conversation
    only re-renders past answers; asking a new question in that
    same conversation runs a brand-new retrieval pass.
  * Store secrets. Nothing sensitive is embedded in this file; the
    LLM key lives server-side in the FastAPI process's environment.

Run:
    streamlit run app.py

Env overrides:
    AGRI_API_URL   e.g. https://agri-schemes-rag.fly.dev
                   default: http://127.0.0.1:8000
    AGRI_DB_PATH   e.g. /data/agri_rag.db
                   default: ./data/agri_rag.db
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from src.chat.store import ChatStore, title_from_query
from src.chat.user_id import get_or_create_user_id


# --- Config ----------------------------------------------------------------

API_URL = os.environ.get("AGRI_API_URL", "http://127.0.0.1:8000")
API_TIMEOUT_S = 60.0

# SQLite path. Same env var name the FastAPI backend uses (harmless
# duplication — the two processes never touch the same DB, but
# consistent naming makes Fly.io config easier to reason about).
DB_PATH = Path(os.environ.get("AGRI_DB_PATH", "data/agri_rag.db"))


# --- Page config ------------------------------------------------------------

st.set_page_config(
    page_title="Agri Schemes Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Store singleton (survives Streamlit reruns) ---------------------------

@st.cache_resource
def _get_store() -> ChatStore:
    """`cache_resource` gives us one ChatStore instance per Streamlit
    server process. Every rerun of the script reuses the same handle
    (and therefore the same lock + connection factory)."""
    return ChatStore(DB_PATH)


store = _get_store()

# --- Per-browser anonymous identity ----------------------------------------
#
# Every store call is scoped to this id. It is minted the first time a
# browser hits the app and lives in the URL (?u=<hex>) so a refresh
# keeps the same identity. See src/chat/user_id.py for the full
# rationale; the short version is: no login, no cookies dependency,
# survives F5.
USER_ID = get_or_create_user_id()


# --- Constants (unchanged from prior UI) ------------------------------------

CATEGORY_STYLE: dict[str, tuple[str, str]] = {
    "Income Support": ("💰", "Income Support"),
    "Crop Insurance": ("🛡️", "Crop Insurance"),
    "Agricultural Credit": ("🏦", "Credit"),
    "Farm Mechanization": ("🚜", "Mechanization"),
    "Horticulture Development": ("🌱", "Horticulture"),
    "Crop Productivity / Food Security": ("🌾", "Food Security"),
    "Agriculture Infrastructure Financing": ("🏗️", "Infrastructure"),
}

SOURCE_TYPE_ICON: dict[str, str] = {
    "pdf": "📄",
    "workflow": "🧭",
    "image": "🖼️",
}

EXAMPLE_QUERIES: list[str] = [
    "What is PM-KISAN and who is eligible?",
    "Who is eligible for a Kisan Credit Card and what's the interest rate?",
    "I am a small farmer with 1 hectare of wheat — which schemes can I register for?",
    "What subsidy does SMAM provide for a rotavator?",
    "How many Farm Machinery Banks have been established under SMAM?",
    "I need tractor subsidy, crop insurance, and post-harvest infrastructure — what can I apply for?",
]


# --- Humanization helpers (unchanged) --------------------------------------

def _humanize_land_holding(v: str) -> str:
    return {
        "marginal": "🌱 marginal farmer (under 1 ha)",
        "small": "🚜 small farmer (1-2 ha)",
        "semi_medium": "🚜 semi-medium (2-4 ha)",
        "medium": "🚜 medium (4-10 ha)",
        "large": "🚜 large (over 10 ha)",
    }.get(v, f"land holding: {v}")


def _humanize_crop_group(v: str) -> str:
    return {
        "rice": "🌾 growing rice", "wheat": "🌾 growing wheat",
        "pulses": "🌰 growing pulses",
        "coarse_cereals": "🌾 growing coarse cereals",
        "nutri_cereals": "🌾 growing nutri-cereals",
        "oilseeds": "🌻 growing oilseeds", "sugarcane": "🎋 growing sugarcane",
        "food_grains": "🌾 growing food grains",
        "horticulture": "🌱 horticulture", "fruit": "🍎 growing fruit",
        "vegetable": "🥬 growing vegetables", "flower": "🌸 growing flowers",
        "spice": "🌶️ growing spices", "medicinal": "🌿 medicinal / aromatic",
        "commercial": "🌱 commercial crops",
    }.get(v, f"crop: {v}")


def _humanize_region(v: str) -> str:
    return {
        "north_east": "📍 North-East India",
        "himalayan": "📍 Himalayan region",
        "general": "📍 plains",
    }.get(v, f"region: {v}")


def _humanize_category(v: str) -> str:
    return {
        "SC": "🎯 SC category", "ST": "🎯 ST category",
        "OBC": "🎯 OBC category", "general": "🎯 general category",
    }.get(v, f"category: {v}")


def _humanize_gender(v: str) -> str:
    return {"woman": "👩 woman farmer", "man": "👨 male farmer"}.get(
        v, f"gender: {v}"
    )


def _humanize_interests(vs: list[str]) -> list[str]:
    icons = {
        "credit": "🏦 credit / loans", "insurance": "🛡️ crop insurance",
        "equipment": "🚜 farm equipment", "mechanization": "🚜 mechanization",
        "custom_hiring": "🚜 custom hiring", "income_support": "💰 income support",
        "horticulture": "🌱 horticulture", "cold_storage": "❄️ cold storage",
        "post_harvest": "📦 post-harvest", "processing": "🏭 processing",
        "warehouse": "🏪 warehouse", "infrastructure": "🏗️ infrastructure",
        "organic": "🌿 organic farming", "food_security": "🌾 food security",
    }
    return [icons.get(v, v) for v in vs]


def _humanize_provided(prov: list[dict]) -> list[str]:
    out: list[str] = []
    for p in prov:
        a = p.get("attribute")
        v = p.get("value")
        if a == "has_land":
            if v is True: out.append("✅ you have land")
            elif v is False: out.append("❌ landless")
        elif a == "occupation":
            if v == "farmer": out.append("👨‍🌾 you're a farmer")
        elif a == "land_holding": out.append(_humanize_land_holding(v))
        elif a == "category": out.append(_humanize_category(v))
        elif a == "gender": out.append(_humanize_gender(v))
        elif a == "region": out.append(_humanize_region(v))
        elif a == "crop_group": out.append(_humanize_crop_group(v))
        elif a == "interests" and isinstance(v, list):
            out.extend(_humanize_interests(v))
    return out


def _humanize_missing(items: list[str]) -> list[str]:
    friendly = []
    for s in items[:6]:
        if ":" in s:
            _, reason = s.split(":", 1)
            friendly.append(reason.strip())
        else:
            friendly.append(s)
    return friendly


# --- Session state ---------------------------------------------------------

# The active conversation id. `None` means "no conversation open yet";
# the first user message will lazily create one and stamp its id here.
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None
# Set by an example-chip / follow-up button to auto-submit on next rerun.
if "auto_run_query" not in st.session_state:
    st.session_state.auto_run_query = ""
# Rename mode toggle in the sidebar.
if "renaming_conv_id" not in st.session_state:
    st.session_state.renaming_conv_id = None


def _new_chat() -> None:
    """Sidebar "New Chat" callback. Leaves current_conv_id = None so
    the next message lazily creates a fresh conversation — avoids
    piling up empty rows if the user clicks New Chat repeatedly."""
    st.session_state.current_conv_id = None
    st.session_state.auto_run_query = ""


def _select_conv(conv_id: int) -> None:
    st.session_state.current_conv_id = conv_id
    st.session_state.auto_run_query = ""
    st.session_state.renaming_conv_id = None


def _set_query(text: str) -> None:
    """Example chip / follow-up button click. Runs on next rerun."""
    st.session_state.auto_run_query = text


# --- API call --------------------------------------------------------------

def _post_query(q: str) -> dict[str, Any]:
    """Talk to the FastAPI backend. Returns the parsed JSON on success,
    a synthetic refusal envelope on 429, and re-raises on other errors
    so the outer try/except in the render path can show an error."""
    resp = requests.post(
        f"{API_URL}/query", json={"query": q}, timeout=API_TIMEOUT_S,
    )
    if resp.status_code == 429:
        return {
            "query": q,
            "answer": (
                "Rate limit exceeded. Please wait a moment before asking again."
            ),
            "refused": True,
            "refusal_reason": "rate_limited",
        }
    resp.raise_for_status()
    return resp.json()


# --- Sidebar: conversation list + backend status ---------------------------

with st.sidebar:
    st.markdown("### 🌾 Agri Schemes Assistant")

    # Backend health probe. Runs on every rerun — cheap (one HTTP GET)
    # and gives the user immediate feedback that the backend is live.
    try:
        h = requests.get(f"{API_URL}/healthz", timeout=5).json()
        st.success(f"**{h['collection_size']:,}** chunks indexed")
        st.caption(f"Collection: `{h['collection']}`")
    except Exception as e:
        st.error("Backend not reachable")
        st.caption(f"`AGRI_API_URL` = `{API_URL}`\n\n{e}")

    st.divider()

    # --- New Chat ---
    st.button(
        "＋ New chat", type="primary", use_container_width=True,
        on_click=_new_chat,
    )

    # --- Conversation list ---
    # Scoped to this browser's id — a browser only ever sees rows it
    # owns. Legacy rows written before ownership existed have
    # user_id = NULL and never match any real UUID, so they stay
    # hidden without being deleted.
    convs = store.list_conversations(USER_ID)
    if convs:
        st.markdown("**Conversations**")
        for cv in convs:
            is_active = (cv.id == st.session_state.current_conv_id)
            if st.session_state.renaming_conv_id == cv.id:
                # Rename form — appears inline when the ✎ button was clicked.
                new_title = st.text_input(
                    "Rename", value=cv.title, key=f"rename_input_{cv.id}",
                    label_visibility="collapsed",
                )
                col_ok, col_cancel = st.columns(2)
                if col_ok.button("Save", key=f"rename_save_{cv.id}",
                                 use_container_width=True):
                    store.rename_conversation(USER_ID, cv.id, new_title)
                    st.session_state.renaming_conv_id = None
                    st.rerun()
                if col_cancel.button("Cancel", key=f"rename_cancel_{cv.id}",
                                     use_container_width=True):
                    st.session_state.renaming_conv_id = None
                    st.rerun()
            else:
                # Normal row: click title to open; ✎ rename; 🗑 delete.
                col_title, col_rename, col_del = st.columns([6, 1, 1])
                label = f"**{cv.title}**" if is_active else cv.title
                col_title.button(
                    label, key=f"conv_{cv.id}",
                    use_container_width=True,
                    on_click=_select_conv, args=(cv.id,),
                )
                if col_rename.button(
                    "✎", key=f"ren_{cv.id}",
                    help="Rename this conversation",
                ):
                    st.session_state.renaming_conv_id = cv.id
                    st.rerun()
                if col_del.button(
                    "🗑", key=f"del_{cv.id}",
                    help="Delete this conversation",
                ):
                    store.delete_conversation(USER_ID, cv.id)
                    if st.session_state.current_conv_id == cv.id:
                        st.session_state.current_conv_id = None
                    st.rerun()

    st.divider()

    with st.expander("Schemes covered"):
        st.markdown(
            "- 💰 **PM-KISAN** — direct income support\n"
            "- 🛡️ **PMFBY** — crop insurance\n"
            "- 🏦 **KCC** — Kisan Credit Card\n"
            "- 🚜 **SMAM** — farm mechanization subsidy\n"
            "- 🌱 **MIDH** — horticulture development\n"
            "- 🌾 **NFSM** (NFSNM) — food security\n"
            "- 🏗️ **AIF** — infrastructure financing\n"
        )

    with st.expander("How answers are grounded"):
        st.markdown(
            "- **Hybrid retrieval**: semantic + BM25, fused with RRF, "
            "reranked with a cross-encoder\n"
            "- **Grounded generation**: LLM only sees the top-5 chunks; "
            "every answer cites its sources\n"
            "- **Structured layer**: deterministic scheme discovery over "
            "hand-curated eligibility rules (no LLM), cited to doc + page\n"
            "- **Guardrails**: query validation, prompt-injection detection, "
            "confidence threshold, PII scrub, 30 req/min/IP"
        )


# --- Renderers -------------------------------------------------------------

def _render_scheme_discovery(discovery: dict) -> None:
    """Discovery card block. Called once per assistant message that has
    discovery matches — during a live turn AND when replaying history."""
    matches = discovery.get("matches") or []
    if not matches:
        return

    st.markdown("#### Schemes potentially applicable to you")
    st.caption(
        "Based on what you told me. Not a definitive eligibility "
        "decision — final eligibility depends on the conditions in each "
        "scheme card below."
    )

    prov_chips = _humanize_provided(discovery.get("provided_information") or [])
    miss_chips = _humanize_missing(discovery.get("missing_information_summary") or [])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**You told me**")
        if not prov_chips:
            st.markdown(
                "_Nothing specific — mention your **land size**, "
                "**category**, **crop**, or **interest** for a "
                "shorter list._"
            )
        else:
            for c in prov_chips:
                st.markdown(f"- {c}")
    with col2:
        st.markdown("**Still need to know**")
        if not miss_chips:
            st.markdown(
                "_Nothing extra — see each scheme card for its own "
                "required checks._"
            )
        else:
            for c in miss_chips:
                st.markdown(f"- {c}")

    st.markdown("")

    for m in matches:
        emoji, short_cat = CATEGORY_STYLE.get(
            m["category"], ("•", m["category"])
        )
        with st.container(border=True):
            header_left, header_right = st.columns([4, 1])
            with header_left:
                st.markdown(
                    f"### {emoji} {m['name']}  \n_{m['long_name']}_"
                )
                st.caption(f"**{short_cat}** — {m['one_liner']}")
            with header_right:
                if m["boost_hits"] > 0:
                    st.markdown(
                        f"<div style='text-align:right'>"
                        f"<span style='background:#FFE58F; padding:4px 10px; "
                        f"border-radius:12px; font-weight:600;'>"
                        f"🎯 strong match ×{m['boost_hits']}</span></div>",
                        unsafe_allow_html=True,
                    )
                elif m.get("always_included"):
                    st.markdown(
                        "<div style='text-align:right'>"
                        "<span style='color:#888; font-size:0.85em;'>"
                        "general scheme</span></div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("**Key benefits**")
            for b in m["key_benefits"][:4]:
                st.markdown(f"- {b}")

            hr = m.get("how_to_register") or {}
            if hr:
                st.markdown("**Where and how to apply**")
                if hr.get("portal"):
                    st.markdown(f"- 🔗 **Online:** `{hr['portal']}`")
                if hr.get("in_person"):
                    st.markdown(f"- 🏢 **In person:** {hr['in_person']}")

            fh = m.get("follow_up_query_hint")
            if fh:
                st.button(
                    f"💡 Ask: _{fh}_",
                    key=f"followup_{m['code']}_{fh[:40]}_{id(m)}",
                    on_click=_set_query, args=(fh,),
                    use_container_width=True,
                )

            with st.expander(
                "Why this scheme surfaced · what I still need"
            ):
                if m["applicability_reasons"] or m["boost_reasons"]:
                    st.markdown("**Why it surfaced**")
                    for r in m["applicability_reasons"]:
                        mv = r["matched_value"]
                        if isinstance(mv, list):
                            mv = ", ".join(str(x) for x in mv)
                        src = ""
                        if r.get("source_filename"):
                            src = (
                                f"  \n<small>source: "
                                f"`{r['source_filename']}` p{r['source_page']}"
                                f"</small>"
                            )
                        st.markdown(
                            f"- ✅ your `{r['attribute']}` = **{mv}**{src}",
                            unsafe_allow_html=True,
                        )
                    for r in m["boost_reasons"]:
                        mv = r["matched_value"]
                        if isinstance(mv, list):
                            mv = ", ".join(str(x) for x in mv)
                        src = ""
                        if r.get("source_filename"):
                            src = (
                                f"  \n<small>source: "
                                f"`{r['source_filename']}` p{r['source_page']}"
                                f"</small>"
                            )
                        st.markdown(
                            f"- 🎯 boost: `{r['attribute']}` = **{mv}**{src}",
                            unsafe_allow_html=True,
                        )
                if m["missing_information"]:
                    st.markdown("**To confirm eligibility, I still need**")
                    for mi in m["missing_information"]:
                        st.markdown(
                            f"- **`{mi['attribute']}`** — {mi['why_needed']}"
                        )


def _render_citations(citations: list) -> None:
    if not citations:
        return
    st.markdown("#### Sources")
    st.caption("The retrieval chunks the answer was grounded on.")
    for i, c in enumerate(citations, start=1):
        page_range = (
            f"pp. {c['page_start']}-{c['page_end']}"
            if c.get("page_end") and c.get("page_start")
            and c["page_start"] != c["page_end"]
            else f"p. {c.get('page_start') or '?'}"
        )
        ocr_badge = " · OCR" if c.get("is_ocr_source") else ""
        icon = SOURCE_TYPE_ICON.get(c.get("source_type", ""), "•")
        st.markdown(
            f"{i}. {icon} **{c['scheme']}** — "
            f"`{c['source_filename']}` · {page_range}{ocr_badge}"
        )


def _render_tech(data: dict) -> None:
    latency = data.get("latency") or {}
    llm = data.get("llm") or {}
    with st.expander("Technical details"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total latency", f"{latency.get('total_ms', 0)} ms")
        c2.metric(
            "Retrieval",
            f"{latency.get('hybrid_ms', 0) + latency.get('rerank_ms', 0)} ms",
        )
        c3.metric("Generation", f"{latency.get('generation_ms', 0)} ms")
        c4.metric(
            "Tokens (in/out)",
            f"{llm.get('prompt_tokens', 0)} / {llm.get('completion_tokens', 0)}",
        )
        if llm.get("error"):
            st.error(f"LLM error: {llm['error']}")
        st.caption(f"request_id: `{data.get('request_id', '?')}`")
        chunks = data.get("retrieved_chunks") or []
        if chunks:
            st.markdown("**Retrieved chunks** (PII-scrubbed, sorted by rank)")
            for chunk in chunks:
                hdr = (
                    f"r{chunk['rank']} · {chunk['source_type']} · "
                    f"{chunk['scheme']} · "
                    f"{(chunk['source_filename'] or '')[:40]} "
                    f"p{chunk.get('page_start') or '?'}"
                )
                score_bits = []
                for name, label in (
                    ("similarity_score", "sim"),
                    ("rerank_score", "rerank"),
                    ("rrf_score", "rrf"),
                ):
                    v = chunk.get(name)
                    if v is not None:
                        score_bits.append(f"{label}={v:.3f}")
                st.markdown(f"**{hdr}** — {'; '.join(score_bits)}")
                st.code(chunk.get("text_preview", "") or "(empty)", language=None)


def _render_assistant_body(data: dict) -> None:
    """Render one assistant message's body — answer / refusal, discovery
    cards, citations, tech expander. Used for both fresh turns and
    replayed history so the two paths look identical."""
    discovery = data.get("scheme_discovery") or {}
    disc_matches = discovery.get("matches") or []
    is_refused = data.get("refused", False)

    if is_refused and disc_matches:
        st.warning(
            "🤔 I couldn't find a single source that answers your question "
            "directly — but based on what you told me, here are the "
            "schemes that may be relevant to you."
        )
    elif is_refused:
        reason = data.get("refusal_reason") or "unspecified"
        if reason.startswith("validation_"):
            st.warning(f"**Query rejected** — {data.get('answer', '')}")
        elif "confidence" in reason:
            st.info(
                "🤔 **No strong source match.** "
                + data.get("answer", "")
                + "  \n_Try adding a scheme name (PM-KISAN, PMFBY, KCC, "
                "SMAM, MIDH, NFSM, AIF) or a specific attribute (crop, "
                "land size, category)._"
            )
        elif reason == "rate_limited":
            st.error(data.get("answer", ""))
        else:
            st.warning(data.get("answer", ""))
        st.caption(f"reason: `{reason}`")
    else:
        st.markdown(data.get("answer", ""))

    if disc_matches:
        _render_scheme_discovery(discovery)

    if not is_refused:
        _render_citations(data.get("citations") or [])

    _render_tech(data)


# --- Main pane -------------------------------------------------------------

st.title("Ask about Government of India agriculture schemes")
st.markdown(
    "Ask about **eligibility, benefits, subsidies, registration, or how "
    "to apply** for PM-KISAN, PMFBY, KCC, SMAM, MIDH, NFSM, or AIF."
)

# Replay the active conversation's message history (if any).
# get_messages is ownership-scoped: if the current session_state points
# at someone else's conversation id (stale URL, shared bookmark, etc.)
# the store returns [] rather than leaking cross-browser content.
active_id = st.session_state.current_conv_id
history_msgs = store.get_messages(USER_ID, active_id) if active_id else []

for msg in history_msgs:
    with st.chat_message(msg.role):
        if msg.role == "user":
            st.markdown(msg.content)
        else:
            payload = msg.payload
            if payload is not None:
                _render_assistant_body(payload)
            else:
                # Legacy or truncated payload — fall back to plain content.
                st.markdown(msg.content)


# Example chips before the first message of a fresh conversation.
if not history_msgs and not st.session_state.auto_run_query:
    st.markdown("**Try one of these:**")
    ncols = 3
    for i in range(0, len(EXAMPLE_QUERIES), ncols):
        cols = st.columns(ncols)
        for col, q in zip(cols, EXAMPLE_QUERIES[i:i + ncols]):
            col.button(
                q, key=f"example_{i}_{q[:24]}",
                use_container_width=True,
                on_click=_set_query, args=(q,),
            )


# Chat input pinned at the bottom.
typed = st.chat_input("Ask a question about a scheme…")

_query_to_run: str = ""
if typed and typed.strip():
    _query_to_run = typed.strip()
elif st.session_state.auto_run_query:
    _query_to_run = st.session_state.auto_run_query
    st.session_state.auto_run_query = ""  # consume before running


if _query_to_run:
    # Lazily create a conversation on the first message. Title comes
    # from the user's first query so the sidebar list is readable.
    # The row is stamped with USER_ID so it belongs to this browser.
    if st.session_state.current_conv_id is None:
        st.session_state.current_conv_id = store.create_conversation(
            USER_ID, title_from_query(_query_to_run)
        )
    active_id = st.session_state.current_conv_id

    # Persist the user turn immediately so it appears in history even
    # if the API call fails partway.
    store.add_message(USER_ID, active_id, "user", _query_to_run)

    # Render the just-added user message on this run (the store write
    # will be reflected on the next rerun, but we want the spinner-
    # blocking call to have the message visible in front of it).
    with st.chat_message("user"):
        st.markdown(_query_to_run)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + reasoning…"):
            try:
                data = _post_query(_query_to_run)
            except requests.RequestException as e:
                st.error(f"Backend error: {e}")
                data = {
                    "query": _query_to_run,
                    "answer": f"Backend unreachable: {e}",
                    "refused": True,
                    "refusal_reason": "backend_error",
                }

        _render_assistant_body(data)

    # Persist the assistant turn — content = the readable answer,
    # payload = the full response so a reload replays identically.
    store.add_message(
        USER_ID,
        active_id,
        "assistant",
        data.get("answer", ""),
        payload=data,
    )
    # Rerun so the sidebar re-sorts (this conversation just moved to
    # updated_at = now) and the example chips disappear.
    st.rerun()
