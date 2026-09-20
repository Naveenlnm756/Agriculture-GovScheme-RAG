# syntax=docker/dockerfile:1.6
#
# Production image for the Agri Schemes RAG.
#
# Contents (in layer order, so cache reuse is maximised):
#   1. Debian slim + Python 3.11 (locked in CLAUDE.md §12)
#   2. requirements-prod.txt → pip install
#   3. Application source (src/, app.py, start.sh)
#   4. Scheme-eligibility JSON (baked into the image — code artefact,
#      not user data; must NOT live on the volume)
#
# The Chroma vector store and the SQLite chat DB do NOT live in the
# image. They live on the Fly.io persistent volume mounted at /data.
# See fly.toml. Env vars point the app at those paths.

FROM python:3.11-slim AS base

# System deps: curl for a lightweight healthcheck. No Tesseract — the
# runtime image never runs OCR (that happened at ingest time).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for the runtime process. Explicit UID 1000 is deliberate:
#   * On Linux hosts, bind-mounted files from the user's home dir are
#     typically owned by UID 1000 — matching UIDs avoids permission
#     errors on read-only bind mounts (Docker Desktop virtualises UIDs
#     on Windows/Mac, so this is Linux-specific but harmless elsewhere).
#   * The runtime container does NOT execute as `app` directly. It
#     executes as root long enough for docker-entrypoint.sh to fix
#     /data ownership on a fresh persistent volume (Docker named
#     volume, Railway volume, Fly volume — all create /data root-owned
#     on first mount). Entrypoint then execs `runuser -u app -- ...`
#     to drop privileges. See docker-entrypoint.sh for the full
#     reasoning.
RUN groupadd -g 1000 app && useradd -u 1000 -g app -d /app -s /bin/bash app

WORKDIR /app

# Python behaviour tweaks:
#   PYTHONDONTWRITEBYTECODE=1  no .pyc files scribbled into the image
#   PYTHONUNBUFFERED=1         stdout/stderr flush immediately so Fly logs
#                              see errors in real time instead of on crash
#   PIP_NO_CACHE_DIR=1         don't cache wheel downloads inside the image
#   HF_HOME / TRANSFORMERS_CACHE point the sentence-transformers download
#     cache at a writable path INSIDE the image so the SentenceTransformer
#     model gets baked in at build time (see the pre-download step below).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

# --- Layer 1: dependencies ---
# Copied BEFORE the app source so a source-only edit doesn't invalidate
# this (slow) layer's cache.
COPY requirements-prod.txt ./
RUN pip install --upgrade pip && pip install -r requirements-prod.txt

# --- Layer 2: pre-download embedder + reranker weights ---
# Fetching model weights at container-start time would add ~10-30s to
# cold-start AND require outbound Hugging Face access from the running
# container. Baking the weights into the image makes cold-starts fast
# AND lets the container run in restricted egress environments.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder;\
SentenceTransformer('BAAI/bge-small-en-v1.5');\
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2');\
print('models cached')\
"

# --- Layer 3: application code ---
# One COPY per source group so an edit to app.py doesn't invalidate the
# src/ layer, and vice versa.
COPY src ./src
COPY app.py start.sh docker-entrypoint.sh ./

# Scheme-eligibility rules ship WITH the image — they are authored code
# artefacts, reviewed alongside src/. They must NOT be on the volume
# (which is user-mutable). scheme_discovery.py reads
# `<data_raw_dir>.parent / "scheme_eligibility.json"`, so we place the
# file at /app/data/scheme_eligibility.json.
COPY data/scheme_eligibility.json /app/data/scheme_eligibility.json

# The application also needs data/raw and data/processed as PATHS to
# exist (config.py Field defaults); empty stubs are fine because the
# runtime never opens them.
RUN mkdir -p /app/data/raw /app/data/processed \
    && chown -R app:app /app \
    && chmod +x /app/start.sh /app/docker-entrypoint.sh

# NOTE: we deliberately do NOT `USER app` here. The container starts as
# root so docker-entrypoint.sh can chown /data (the persistent volume
# mount point, root-owned by default). Entrypoint then execs the real
# startup as `app` via runuser. See docker-entrypoint.sh for details.

# Streamlit's public port. FastAPI stays on 127.0.0.1:8000, unreachable
# from outside the container by design.
EXPOSE 8080

# Healthcheck hits Streamlit's built-in probe — good enough for the
# platform to know the container is serving. (Deeper /healthz on the
# FastAPI side is also available; we probe the outer layer here.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://127.0.0.1:8080/_stcore/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["/app/start.sh"]
