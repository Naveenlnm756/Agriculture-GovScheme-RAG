#!/usr/bin/env bash
#
# Container entrypoint. Boots FastAPI + Streamlit under one process
# tree and dies as a unit if either child dies (Fly.io will restart
# the whole container — no half-alive states).
#
# Layout:
#   - uvicorn (FastAPI):  127.0.0.1:8000  — private, loopback only
#   - streamlit          : 0.0.0.0:8080  — public entrypoint
#
# The Streamlit process talks to FastAPI via AGRI_API_URL, which
# defaults to http://127.0.0.1:8000 (see .env.example). Loopback means
# no CORS, no TLS between the two, no network hop.

set -euo pipefail

echo "[start.sh] launching FastAPI on 127.0.0.1:8000"
uvicorn src.api.main:app \
    --host 127.0.0.1 --port 8000 \
    --log-level info &
UVICORN_PID=$!

# Wait until the FastAPI /healthz is answering before starting
# Streamlit. That way the sidebar's health probe shows green on the
# first render instead of red-then-green after a rerun. Cap the wait
# so a wedged backend doesn't block us forever — Streamlit can start
# anyway and show its own "backend not reachable" state.
echo "[start.sh] waiting for backend to become healthy…"
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8000/healthz > /dev/null 2>&1; then
    echo "[start.sh] backend healthy"
    break
  fi
  sleep 1
done

echo "[start.sh] launching Streamlit on 0.0.0.0:8080"
streamlit run app.py \
    --server.port 8080 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection true \
    --browser.gatherUsageStats false &
STREAMLIT_PID=$!

# `wait -n` blocks until ANY child exits. When one dies we exit too
# (with that child's exit code) — Fly.io restarts the container.
# Simple, honest failure model.
wait -n "$UVICORN_PID" "$STREAMLIT_PID"
EXIT_CODE=$?
echo "[start.sh] child exited with code $EXIT_CODE — shutting down"
kill "$UVICORN_PID" "$STREAMLIT_PID" 2>/dev/null || true
exit $EXIT_CODE
