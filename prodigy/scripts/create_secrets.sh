#!/usr/bin/env bash
set -euo pipefail
: "${PRODIGY_KEY_VALUE:?set PRODIGY_KEY_VALUE}"
# Optional Postgres values
: "${PGUSER:=}"; : "${PGPASSWORD:=}"; : "${PGHOST:=}"; : "${PGPORT:=}"; : "${PGDATABASE:=}"

aws secretsmanager create-secret --name prodigy/licenseKey --secret-string "$PRODIGY_KEY_VALUE" >/dev/null 2>&1 || \
aws secretsmanager put-secret-value --secret-id prodigy/licenseKey --secret-string "$PRODIGY_KEY_VALUE" >/dev/null

if [[ -n "$PGUSER$PGPASSWORD$PGHOST$PGPORT$PGDATABASE" ]]; then
  PG_JSON=$(jq -n --arg u "$PGUSER" --arg p "$PGPASSWORD" --arg h "$PGHOST" --arg P "$PGPORT" --arg d "$PGDATABASE" \
    '{PGUSER:$u, PGPASSWORD:$p, PGHOST:$h, PGPORT:$P, PGDATABASE:$d}')
  aws secretsmanager create-secret --name prodigy/pg --secret-string "$PG_JSON" >/dev/null 2>&1 || \
  aws secretsmanager put-secret-value --secret-id prodigy/pg --secret-string "$PG_JSON" >/dev/null
fi
