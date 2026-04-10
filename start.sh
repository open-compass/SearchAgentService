#!/bin/bash
# Start SearchAgentService
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8083}"
WORKERS="${WORKERS:-4}"
TIMEOUT_KEEP_ALIVE="${TIMEOUT_KEEP_ALIVE:-5}"

echo "Starting SearchAgentService on ${HOST}:${PORT} with ${WORKERS} workers (timeout_keep_alive=${TIMEOUT_KEEP_ALIVE}s)"

exec uvicorn service:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --timeout-keep-alive "$TIMEOUT_KEEP_ALIVE"
