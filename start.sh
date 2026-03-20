#!/bin/bash
# Start SearchAgentService
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8083}"
WORKERS="${WORKERS:-1}"

echo "Starting SearchAgentService on ${HOST}:${PORT} with ${WORKERS} workers"

exec uvicorn service:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
