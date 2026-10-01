#!/usr/bin/env bash
# Stage 1: full /models + /predict request-contract checks, read-only over
# the repository models/ directory. One server on 127.0.0.1:18091.
set -euo pipefail
cd "$(dirname "$0")/../.."

# shellcheck source=tests/e2e/lib.sh
source tests/e2e/lib.sh

ADDR=127.0.0.1:18091
BASE="http://$ADDR"
MODEL_DIR="$MODELS_DIR/risk_score"
OUT="$(mktemp /tmp/e2e-contract-XXXXXX.json)"
PINNED="$(mktemp /tmp/e2e-contract-pinned-XXXXXX.json)"

trap 'srv_stop; rm -f "$OUT" "$PINNED"' EXIT

# Fully valid feature set for the risk_score schema.
FEATURES_OK='{"amount":42.5,"channel":"web","hour":13,"is_new_user":0,"txn_count_24h":3,"card_age_days":900}'

req() { printf '{"model":"%s","features":%s}' "$1" "$FEATURES_OK"; }

latest="$(latest_version "$MODEL_DIR")"
oldest="$(oldest_version "$MODEL_DIR")"
count="$(version_count "$MODEL_DIR")"
log "fixtures: $count versions, latest=$latest, oldest=$oldest"

srv_start "$ADDR" "$MODELS_DIR"

echo "== GET /models structure =="
code=$(http_body "$OUT" GET "$BASE/models")
expect_code 200 "$code" "GET /models"
assert_py "$OUT" "$count" "$latest" "$oldest" <<'PYEOF'
import json
import sys

models = json.load(open(sys.argv[1]))["models"]
want_count = int(sys.argv[2])
latest, oldest = sys.argv[3], sys.argv[4]
assert len(models) == want_count, f"want {want_count} models, got {len(models)}"
keys = [m["key"] for m in models]
assert keys == sorted(keys), f"not sorted by key: {keys}"
assert f"risk_score@{latest}" in keys, f"latest missing: {keys}"
assert f"risk_score@{oldest}" in keys, f"oldest missing: {keys}"
assert all(m["name"] == "risk_score" and "metrics" in m for m in models)
print(f"models ok: {keys}")
PYEOF

echo "== version pinning and latest resolution =="
code=$(http_body "$OUT" POST "$BASE/predict" "$(req "risk_score@$oldest")")
expect_code 200 "$code" "predict pinned to oldest version"
assert_py "$OUT" "risk_score@$oldest" <<'PYEOF'
import json
import sys

body = json.load(open(sys.argv[1]))
assert body["model"] == sys.argv[2], f"want {sys.argv[2]}, got {body['model']}"
print("oldest pin served:", body["model"])
PYEOF

code=$(http_body "$PINNED" POST "$BASE/predict" "$(req "risk_score@$latest")")
expect_code 200 "$code" "predict pinned to latest version"
code=$(http_body "$OUT" POST "$BASE/predict" "$(req risk_score)")
expect_code 200 "$code" "predict without pin"
assert_py "$OUT" "$PINNED" "risk_score@$latest" <<'PYEOF'
import json
import sys

unpinned = json.load(open(sys.argv[1]))
pinned = json.load(open(sys.argv[2]))
want = sys.argv[3]
assert unpinned["model"] == want, f"latest resolution: want {want}, got {unpinned['model']}"
assert pinned["model"] == want
assert unpinned["score"] == pinned["score"], "same booster must give identical score"
assert isinstance(unpinned["latency_us"], int) and unpinned["latency_us"] >= 0
for absent in ("missing_features", "unknown_categories", "unexpected_fields"):
    assert absent not in unpinned, f"{absent} must be omitted for a clean request"
print(f"latest resolution + score identity ok: {unpinned['score']}")
PYEOF

echo "== malformed requests (400 branches) =="
code=$(http_body "$OUT" POST "$BASE/predict" '{"model":')
expect_code 400 "$code" "broken JSON -> 400"
expect_error_contains "$OUT" "invalid JSON body" "broken JSON message"

code=$(http_body "$OUT" POST "$BASE/predict" '[1, 2, 3]')
expect_code 400 "$code" "array body -> 400"
expect_error_contains "$OUT" "request body must be a JSON object" "array body message"

code=$(http_body "$OUT" POST "$BASE/predict" '{"model":"","features":{}}')
expect_code 400 "$code" "empty model -> 400"
expect_error_contains "$OUT" 'field "model" must be a non-empty string' "empty model message"

code=$(http_body "$OUT" POST "$BASE/predict" '{"model":7,"features":{}}')
expect_code 400 "$code" "non-string model -> 400"
expect_error_contains "$OUT" 'field "model" must be a non-empty string' "non-string model message"

code=$(http_body "$OUT" POST "$BASE/predict" '{"model":"risk_score","features":[1]}')
expect_code 400 "$code" "non-object features -> 400"
expect_error_contains "$OUT" 'field "features" must be a JSON object' "non-object features message"

code=$(http_body "$OUT" POST "$BASE/predict" "$(req 'risk_score@')")
expect_code 400 "$code" "empty version pin -> 400"
expect_error_contains "$OUT" 'invalid model reference "risk_score@"' "invalid reference message"

code=$(http_body "$OUT" POST "$BASE/predict" \
  '{"model":"risk_score","features":{"amount":"lots","channel":"web","hour":1,"is_new_user":0,"txn_count_24h":1,"card_age_days":1}}')
expect_code 400 "$code" "feature type mismatch -> 400"
expect_error_contains "$OUT" 'feature "amount" expects a JSON number, got a string' "type mismatch message"

echo "== unknown model references (404) =="
code=$(http_code POST "$BASE/predict" '{"model":"no_such_model","features":{}}')
expect_code 404 "$code" "unknown model -> 404"
code=$(http_code POST "$BASE/predict" "$(req 'risk_score@2099-01-01T0000')")
expect_code 404 "$code" "unknown pinned version -> 404"

echo "== extraction notes content and order =="
code=$(http_body "$OUT" POST "$BASE/predict" \
  '{"model":"risk_score","features":{"amount":10,"channel":"web"}}')
expect_code 200 "$code" "partial features -> 200"
assert_py "$OUT" "$MODEL_DIR/$latest/manifest.json" <<'PYEOF'
import json
import sys

body = json.load(open(sys.argv[1]))
manifest = json.load(open(sys.argv[2]))
provided = {"amount", "channel"}
want = [f["name"] for f in manifest["feature_schema"] if f["name"] not in provided]
assert body["missing_features"] == want, f"want schema order {want}, got {body['missing_features']}"
assert "unexpected_fields" not in body and "unknown_categories" not in body, body
print(f"missing_features follows schema order: {body['missing_features']}")
PYEOF

code=$(http_body "$OUT" POST "$BASE/predict" \
  '{"model":"risk_score","features":{"amount":10,"channel":"blockchain","hour":1,"is_new_user":0,"txn_count_24h":1,"card_age_days":1}}')
expect_code 200 "$code" "unknown category value -> 200"
assert_py "$OUT" <<'PYEOF'
import json
import sys

body = json.load(open(sys.argv[1]))
assert body["unknown_categories"] == ["channel=blockchain"], body
assert "missing_features" not in body and "unexpected_fields" not in body, body
print("unknown_categories reported: [channel=blockchain]")
PYEOF

code=$(http_body "$OUT" POST "$BASE/predict" \
  '{"model":"risk_score","features":{"amount":10,"channel":"web","hour":1,"is_new_user":0,"txn_count_24h":1,"card_age_days":1,"zeta_extra":1,"alpha_extra":2}}')
expect_code 200 "$code" "unexpected extra fields -> 200"
assert_py "$OUT" <<'PYEOF'
import json
import sys

body = json.load(open(sys.argv[1]))
assert body["unexpected_fields"] == ["alpha_extra", "zeta_extra"], body
assert "missing_features" not in body and "unknown_categories" not in body, body
print("unexpected_fields sorted alphabetically: [alpha_extra, zeta_extra]")
PYEOF

srv_stop
echo "stage http_contract: OK"
