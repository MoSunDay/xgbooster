#!/usr/bin/env bash
# Shared helpers for the e2e suites. Sourced (never executed) by the stage
# scripts; every stage runs from the repository root.
#
# Environment hygiene first: XGBOOSTER_* variables leaked from the calling
# shell could silently change server behavior (token, rate limit, strict
# mode), so they are dropped here. Stage-specific settings are passed
# explicitly as srv_start env assignments instead.

# shellcheck shell=bash
set -euo pipefail

unset XGBOOSTER_ADMIN_TOKEN XGBOOSTER_RATE_LIMIT_RPS XGBOOSTER_RATE_BURST \
  XGBOOSTER_MAX_INFLIGHT XGBOOSTER_STRICT_VERSION XGBOOSTER_MODELS_DIR XGBOOSTER_LIB

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
BIN="$REPO_ROOT/infer/target/debug/xgbooster-infer"
LIB_PATH="$REPO_ROOT/infer/lib/libxgboost.so"
MODELS_DIR="$REPO_ROOT/models"

E2E_SRV_PID=""
E2E_SRV_LOG=""

log() { echo "== $* =="; }

fail() {
  echo "ERROR: $*" >&2
  if [ -f "${E2E_SRV_LOG:-}" ]; then
    echo "---- server log ($E2E_SRV_LOG, tail) ----" >&2
    tail -n 30 "$E2E_SRV_LOG" >&2
  fi
  exit 1
}

expect_code() { # expect_code WANT GOT DESC
  [ "$2" = "$1" ] || fail "$3: expected HTTP $1, got $2"
  echo "$3: HTTP $2"
}

expect_contains() { # expect_contains HAYSTACK NEEDLE DESC
  case "$1" in
    *"$2"*) ;;
    *) fail "$3: missing \"$2\"" ;;
  esac
  echo "$3: contains \"$2\""
}

# Assert the JSON error body {"error": "..."} contains a substring; the
# message is JSON-decoded first so quotes inside it need no escaping.
expect_error_contains() { # expect_error_contains BODY_FILE WANT DESC
  "$PY" - "$1" "$2" "$3" <<'PYEOF'
import json
import sys

body_file, want, desc = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    message = json.load(open(body_file))["error"]
except Exception as exc:
    raise SystemExit(f"{desc}: body is not a JSON error object: {exc}")
if want not in message:
    raise SystemExit(f"{desc}: error {message!r} missing {want!r}")
print(f"{desc}: error contains {want!r}")
PYEOF
}

http_code() { # http_code METHOD URL [JSON_BODY] -> status code on stdout
  local method="$1" url="$2" body="${3:-}"
  if [ -n "$body" ]; then
    curl -s -o /dev/null -w '%{http_code}' -X "$method" "$url" \
      -H 'Content-Type: application/json' -d "$body"
  else
    curl -s -o /dev/null -w '%{http_code}' -X "$method" "$url"
  fi
}

http_body() { # http_body OUT_FILE METHOD URL [JSON_BODY] -> status code
  local out="$1" method="$2" url="$3" body="${4:-}"
  if [ -n "$body" ]; then
    curl -s -o "$out" -w '%{http_code}' -X "$method" "$url" \
      -H 'Content-Type: application/json' -d "$body"
  else
    curl -s -o "$out" -w '%{http_code}' -X "$method" "$url"
  fi
}

# Numeric-aware version pick mirroring registry::pick_latest: a trailing
# "-<digits>" suffix compares numerically, otherwise the whole name
# compares lexicographically.
version_pick() { # version_pick max|min MODEL_VERSION_DIR
  "$PY" - "$1" "$2" <<'PYEOF'
import pathlib
import sys


def sort_key(name):
    base, sep, suffix = name.rpartition("-")
    if sep and suffix.isascii() and suffix.isdigit():
        return (base, int(suffix))
    return (name, -1)


mode, root = sys.argv[1], pathlib.Path(sys.argv[2])
names = [p.name for p in root.iterdir() if p.is_dir()]
if not names:
    raise SystemExit(f"no version directories under {root}")
pick = max if mode == "max" else min
print(pick(names, key=sort_key))
PYEOF
}

latest_version() { version_pick max "$1"; }
oldest_version() { version_pick min "$1"; }

version_count() { # version_count MODEL_VERSION_DIR -> number of version dirs
  find "$1" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d '[:space:]'
}

srv_wait_ready() { # srv_wait_ready URL PID LOG
  local url="$1" pid="$2" log="$3" waited=0
  until curl -sf "$url" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      cat "$log" >&2
      fail "server exited before becoming ready"
    fi
    if [ "$waited" -ge 30000 ]; then
      fail "server not ready after ${waited}ms: $url"
    fi
    sleep 0.2
    waited=$((waited + 200))
  done
}

# srv_start ADDR MODELS_DIR [ENV=VAL ...]: launch the debug server with the
# given env overrides, wait for readiness, track it for srv_stop/fail.
srv_start() {
  local addr="$1" models="$2"
  shift 2
  E2E_SRV_LOG="$(mktemp /tmp/e2e-server-XXXXXX.log)"
  env "$@" "$BIN" --models-dir "$models" --lib "$LIB_PATH" --addr "$addr" \
    >"$E2E_SRV_LOG" 2>&1 &
  E2E_SRV_PID=$!
  srv_wait_ready "http://$addr/models" "$E2E_SRV_PID" "$E2E_SRV_LOG"
}

srv_stop() {
  if [ -n "${E2E_SRV_PID:-}" ] && kill -0 "$E2E_SRV_PID" 2>/dev/null; then
    kill "$E2E_SRV_PID" 2>/dev/null || true
    wait "$E2E_SRV_PID" 2>/dev/null || true
  fi
  E2E_SRV_PID=""
}

# Run an inline python snippet (stdin) with argv forwarded, assert-style.
assert_py() {
  "$PY" - "$@"
}

# models/ is gitignored; on a fresh clone wait for the training venv to be
# provisioned and train a fallback artifact (mirrors tests/consistency).
ensure_models() {
  if [ -n "$(ls -A "$MODELS_DIR/risk_score" 2>/dev/null)" ]; then
    return
  fi
  log "models dir empty; waiting for xgboost in .venv"
  local waited=0
  until "$PY" -c "import xgboost" >/dev/null 2>&1; do
    if [ "$waited" -ge 900 ]; then
      fail "timed out waiting for .venv xgboost install"
    fi
    sleep 10
    waited=$((waited + 10))
  done
  log "training fallback model"
  PYTHONPATH="$REPO_ROOT/train" "$PY" -m xgbooster_train.train \
    --models-dir "$MODELS_DIR" --trials 0
}

[ -x "$PY" ] || fail "missing $PY (provision the training venv first)"
