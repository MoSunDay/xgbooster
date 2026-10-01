#!/usr/bin/env bash
# E2E orchestrator: build the debug server, then run the stage scripts in
# order. Each stage owns its servers and temp dirs; the trap below is a
# last-resort sweeper for aborts between stages.
#
# Stages:
#   1. http_contract.sh - /models + /predict request contract (read-only
#      over the repository models/, port 18091)
#   2. reload_gates.sh  - hot add/delete reload semantics, version gates,
#      rate-limit recovery (artifact churn in a mktemp copy, ports
#      18092-18094; never touches the repository models/)
set -euo pipefail
cd "$(dirname "$0")/../.."

# shellcheck source=tests/e2e/lib.sh
source tests/e2e/lib.sh

log "building debug server"
(cd "$REPO_ROOT/infer" && cargo build)

ensure_models

cleanup() {
  srv_stop
}
trap cleanup EXIT

log "stage 1/2: HTTP contract"
bash tests/e2e/http_contract.sh

log "stage 2/2: reload + gates"
bash tests/e2e/reload_gates.sh

log "E2E OK"
