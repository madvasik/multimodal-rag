#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f .env ]]; then
  echo "No .env found; copying .env.example to .env (add MISTRAL_API_KEY and check paths)."
  cp .env.example .env
fi
docker compose up --build "$@"
