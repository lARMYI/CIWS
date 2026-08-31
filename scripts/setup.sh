#!/usr/bin/env bash
# One-time setup for macOS and Linux.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> CIWS setup"

command -v python3 >/dev/null || { echo "Python 3.10+ is required."; exit 1; }
command -v node    >/dev/null || { echo "Node 18+ is required (for the interface)."; exit 1; }

if [ ! -d .venv ]; then
  echo "--> creating the Python environment"
  python3 -m venv .venv
fi

echo "--> installing the server"
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -e "apps/server[ingest]"
./.venv/bin/pip install --quiet anthropic

echo "--> building the interface"
cd apps/web && npm install --silent && npm run build:fast && cd ../..

[ -f .env ] || { cp .env.example .env; echo "--> wrote .env (add your keys, or leave it empty)"; }

echo
echo "Done. Start it with:  ./scripts/start.sh"
