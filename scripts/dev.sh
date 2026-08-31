#!/usr/bin/env bash
# Development mode: the API with auto-reload, and Vite with hot module replacement.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a

./.venv/bin/python -m ciws --reload --log-level DEBUG &
API=$!
trap 'kill $API 2>/dev/null || true' EXIT

cd apps/web && npm run dev
