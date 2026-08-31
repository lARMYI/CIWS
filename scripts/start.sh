#!/usr/bin/env bash
# Start CIWS and open it in your browser.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || { echo "Run ./scripts/setup.sh first."; exit 1; }
[ -f .env ] && set -a && . ./.env && set +a
exec ./.venv/bin/python -m ciws --open "$@"
