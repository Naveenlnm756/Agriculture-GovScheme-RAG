"""
Anonymous per-browser identity for the Streamlit frontend.

Why this module exists
----------------------
`st.session_state` is per-Streamlit-session and is cleared on browser
refresh (F5). If we scoped conversations only by session_state, every
refresh would look like a new user and history would disappear from the
sidebar (the rows would still exist in SQLite, invisible forever). We
need an id that survives reruns AND refreshes without a login.

Where the id lives
------------------
We put the id in the URL query string as ``?u=<32-hex>``. Streamlit
preserves query params across reruns and across browser refreshes, so
the same browser tab keeps the same id indefinitely. We also cache the
id in ``st.session_state`` so we do not re-parse / re-write the URL on
every rerun.

Explicitly rejected alternatives
--------------------------------
* **Cookies via a third-party component** (``streamlit-cookies-controller``
  et al.) — pulls in an unfamiliar JS-backed dependency, which
  CLAUDE.md §4 forbids ("only explainable, standard libraries").
* **``st.context.cookies``** — read-only in current Streamlit; we
  cannot *set* the cookie from Python.
* **``st.session_state`` alone** — requirement (2) forbids it, and it
  would be wiped on refresh anyway.

Trade-off (documented, not accidental)
--------------------------------------
Opening a brand-new tab with no ``?u=`` gets a new identity. Sharing
the URL shares the workspace. This is the standard anonymous-per-URL
model; no login is required, and users can bookmark their URL to
return to their chat history.
"""

from __future__ import annotations

import re
import uuid

import streamlit as st

# 128-bit UUID rendered as 32 lowercase hex characters. We validate
# incoming ids to this exact shape so a user cannot craft an arbitrary
# string in the URL (e.g. `?u=admin`) and start colliding with other
# people's ids or storing garbage in the DB.
_HEX32 = re.compile(r"^[0-9a-f]{32}$")

_SESSION_KEY = "browser_user_id"
_QUERY_KEY = "u"


def get_or_create_user_id() -> str:
    """Return this browser's stable user id, creating one if needed.

    Lookup order:

      1. ``st.session_state["browser_user_id"]`` — fast path on reruns.
      2. ``?u=<hex>`` in the URL — survives browser refreshes.
      3. Generate a fresh random UUID, write it into both (1) and the
         URL, and return it.

    A malformed value at any layer is discarded and a new id is minted;
    we never trust an unvalidated string as an identity.
    """
    cached = st.session_state.get(_SESSION_KEY)
    if isinstance(cached, str) and _HEX32.match(cached):
        return cached

    from_url = st.query_params.get(_QUERY_KEY)
    if isinstance(from_url, str) and _HEX32.match(from_url):
        st.session_state[_SESSION_KEY] = from_url
        return from_url

    fresh = uuid.uuid4().hex
    st.session_state[_SESSION_KEY] = fresh
    # Writing to st.query_params updates the URL in the browser bar so
    # a subsequent refresh sends the same ?u=... back to us.
    st.query_params[_QUERY_KEY] = fresh
    return fresh
