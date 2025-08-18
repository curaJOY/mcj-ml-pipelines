#!/usr/bin/env bash
set -euo pipefail

# Prodigy needs the license key at runtime; pass it as a secret/env (PRODIGY_KEY)
# Bind host/port from env so App Runner/ECS health checks work.
exec prodigy "${RECIPE}" "${DATASET}" "${MODEL}" "${DATA_FILE}" \
  --loader txt \
  --label "${LABELS}" \
  -H "${PRODIGY_HOST}" \
  -p "${PRODIGY_PORT}"
