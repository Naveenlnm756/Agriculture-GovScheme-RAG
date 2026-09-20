#!/usr/bin/env bash
#
# Container entrypoint — runs as root just long enough to make /data
# writable by the runtime user, then drops privileges and execs the
# real startup (start.sh) as the `app` user.
#
# Why this exists:
#   Docker named volumes, Railway persistent volumes, and Fly.io
#   persistent volumes are all created owned by root:root on first
#   mount. If we ran the container as `app` from PID 1 (via `USER app`
#   in the Dockerfile), SQLite would fail on the first `open()` because
#   /data would be root-owned and app has no write bit. The classic
#   symptom is: `sqlite3.OperationalError: unable to open database file`
#   the first time ChatStore tries to create /data/agri_rag.db.
#
#   Trying to fix ownership from the host side (`chown` on a bind mount)
#   works locally but not on Railway / Fly — you don't get a shell on
#   the host, only inside the container. So we fix it inside the
#   container, at every startup. Idempotent — chown on an already-
#   correct dir is a no-op.
#
# What this does NOT do:
#   * Does not recursively chown /data — sub-mounts like a read-only
#     bind of ./data/chroma_prod would fail (chown-recursive on a ro
#     mount errors). We only chown the mount points we KNOW we own.
#   * Does not touch anything under /app — that was chowned at build
#     time in the Dockerfile.

set -euo pipefail

APP_USER="${APP_USER:-app}"
DATA_DIR="${DATA_DIR:-/data}"

# --- Ensure /data exists and is writable by app ------------------------
# On a fresh volume this creates the dir; on a mounted volume it's a
# no-op. chown is non-recursive on purpose (see comment above).
mkdir -p "$DATA_DIR"
chown "$APP_USER":"$APP_USER" "$DATA_DIR" 2>/dev/null || true
chmod 0755 "$DATA_DIR" 2>/dev/null || true

# --- Make the SQLite parent writable ----------------------------------
# AGRI_DB_PATH may point somewhere other than /data (e.g. /data/db/x).
# Whatever its parent is, that dir must be app-writable so SQLite can
# create the file + its WAL/SHM sidecars.
if [ -n "${AGRI_DB_PATH:-}" ]; then
    DB_PARENT="$(dirname "$AGRI_DB_PATH")"
    mkdir -p "$DB_PARENT"
    chown "$APP_USER":"$APP_USER" "$DB_PARENT" 2>/dev/null || true
fi

# --- Make Chroma readable ---------------------------------------------
# Chroma should be READABLE by app but never WRITTEN — we opened it via
# `client.get_collection` (not get_or_create), so the app can't modify
# the index by accident. If /data/chroma_prod is a read-only bind mount,
# `chmod` will fail silently (which is fine — the ro mount is already
# readable). If it's on a fresh volume, we grant a+rX so the app user
# can traverse into it.
if [ -d "$DATA_DIR/chroma_prod" ]; then
    chmod -R a+rX "$DATA_DIR/chroma_prod" 2>/dev/null || true
fi

# --- Drop root and exec the actual startup ----------------------------
# `runuser` ships with util-linux, which is already in python:3.11-slim.
# `exec` replaces this shell so start.sh becomes PID 1 — signals from
# Docker/Railway/Fly (SIGTERM on stop, SIGKILL after grace) reach it
# directly, which is what start.sh's trap logic needs to shut both
# uvicorn and streamlit down cleanly.
exec runuser -u "$APP_USER" -- "$@"
