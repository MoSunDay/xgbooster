#!/usr/bin/env bash
# Stage 2: hot-reload semantics and safety gates. All artifact churn happens
# inside a mktemp copy of models/ so the repository tree is never modified.
#   - hot add: train a new artifact, reload, verify @version + latest switch
#   - hot delete: remove it, reload, verify convergence + 404 fallback
#   - version gate: mismatched xgboost_version -> reload 500 keeps old
#     registry serving; strict mode refuses to boot without the field
#   - rate limit: 429 then token refill recovery
set -euo pipefail
cd "$(dirname "$0")/../.."

# shellcheck source=tests/e2e/lib.sh
source tests/e2e/lib.sh

ADDR=127.0.0.1:18092
AUX_ADDR=127.0.0.1:18093
RL_ADDR=127.0.0.1:18094
BASE="http://$ADDR"

WORK="$(mktemp -d /tmp/e2e-reload-XXXXXX)"
RESP="$WORK/resp.json"
trap 'srv_stop; rm -rf "$WORK"' EXIT

FEATURES_OK='{"amount":42.5,"channel":"web","hour":13,"is_new_user":0,"txn_count_24h":3,"card_age_days":900}'
req() { printf '{"model":"%s","features":%s}' "$1" "$FEATURES_OK"; }

WORK_MODELS="$WORK/models"
mkdir -p "$WORK_MODELS"
cp -r "$MODELS_DIR/risk_score" "$WORK_MODELS/risk_score"

BASELINE_LATEST="$(latest_version "$WORK_MODELS/risk_score")"
BASELINE_COUNT="$(version_count "$WORK_MODELS/risk_score")"
REPO_COUNT="$(version_count "$MODELS_DIR/risk_score")"
log "baseline: $BASELINE_COUNT versions, latest=$BASELINE_LATEST"

srv_start "$ADDR" "$WORK_MODELS"

echo "== baseline registry =="
code=$(http_body "$RESP" GET "$BASE/models")
expect_code 200 "$code" "GET /models baseline"
assert_py "$RESP" "$BASELINE_COUNT" <<'PYEOF'
import json
import sys

count = len(json.load(open(sys.argv[1]))["models"])
assert count == int(sys.argv[2]), f"want {sys.argv[2]}, got {count}"
print(f"baseline serves {count} model versions")
PYEOF

echo "== hot add: train a new artifact, then reload =="
PYTHONPATH="$REPO_ROOT/train" "$PY" -m xgbooster_train.train \
  --models-dir "$WORK_MODELS" --trials 0 --seed 7 >"$WORK/train.log" 2>&1 \
  || { cat "$WORK/train.log" >&2; fail "training into temp models dir failed"; }
NEW_VERSION="$(latest_version "$WORK_MODELS/risk_score")"
[ "$NEW_VERSION" != "$BASELINE_LATEST" ] || fail "training produced no new version"
dirs="$(version_count "$WORK_MODELS/risk_score")"
[ "$dirs" -eq $((BASELINE_COUNT + 1)) ] || fail "want $((BASELINE_COUNT + 1)) version dirs, got $dirs"
log "trained new artifact: $NEW_VERSION"

code=$(http_code POST "$BASE/predict" "$(req "risk_score@$NEW_VERSION")")
expect_code 404 "$code" "new version stays invisible before reload"

code=$(http_body "$RESP" POST "$BASE/admin/reload")
expect_code 200 "$code" "admin reload after hot add"
assert_py "$RESP" "$((BASELINE_COUNT + 1))" <<'PYEOF'
import json
import sys

body = json.load(open(sys.argv[1]))
want = {"status": "ok", "models": int(sys.argv[2])}
assert body == want, f"want {want}, got {body}"
print(f"reload swapped in {body['models']} versions")
PYEOF

echo "== hot-added model serves and matches its holdout =="
REQ="$WORK/req.json"
"$PY" - "$WORK_MODELS/risk_score/$NEW_VERSION" "$REQ" "$WORK/expected_score" <<'PYEOF'
import csv
import json
import pathlib
import sys

ver_dir = pathlib.Path(sys.argv[1])
req_path, score_path = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
manifest = json.loads((ver_dir / "manifest.json").read_text())
types = {f["name"]: f["type"] for f in manifest["feature_schema"]}
header, row = list(csv.reader((ver_dir / "holdout.csv").open(newline="")))[:2]
features = {
    name: raw if types[name] == "categorical" else float(raw)
    for name, raw in zip(header, row)
    if name in types
}
req_path.write_text(json.dumps({
    "model": manifest["name"] + "@" + manifest["version"],
    "features": features,
}))
score_col = next(i for i, name in reversed(list(enumerate(header))) if name not in types)
score_path.write_text(row[score_col])
PYEOF

code=$(http_body "$RESP" POST "$BASE/predict" "@$REQ")
expect_code 200 "$code" "predict new version by explicit pin"
assert_py "$RESP" "$WORK/expected_score" "risk_score@$NEW_VERSION" <<'PYEOF'
import json
import sys

body = json.load(open(sys.argv[1]))
expected = float(open(sys.argv[2]).read())
assert body["model"] == sys.argv[3], f"want {sys.argv[3]}, got {body['model']}"
diff = abs(body["score"] - expected)
assert diff < 1e-6, f"holdout mismatch: got {body['score']}, expected {expected}"
print(f"hot-added score matches holdout first row: diff={diff}")
PYEOF

code=$(http_body "$RESP" POST "$BASE/predict" "$(req risk_score)")
expect_code 200 "$code" "unpinned predict after hot add"
assert_py "$RESP" "risk_score@$NEW_VERSION" <<'PYEOF'
import json
import sys

body = json.load(open(sys.argv[1]))
assert body["model"] == sys.argv[2], f"latest did not switch: {body['model']}"
print("latest switched to:", body["model"])
PYEOF

echo "== hot delete: reload converges after artifact removal =="
rm -rf "$WORK_MODELS/risk_score/$NEW_VERSION"
code=$(http_body "$RESP" POST "$BASE/admin/reload")
expect_code 200 "$code" "admin reload after hot delete"
assert_py "$RESP" "$BASELINE_COUNT" <<'PYEOF'
import json
import sys

body = json.load(open(sys.argv[1]))
assert body == {"status": "ok", "models": int(sys.argv[2])}, body
print(f"registry back to {body['models']} versions")
PYEOF

code=$(http_code POST "$BASE/predict" "$(req "risk_score@$NEW_VERSION")")
expect_code 404 "$code" "deleted version unreachable after reload"
code=$(http_body "$RESP" POST "$BASE/predict" "$(req risk_score)")
expect_code 200 "$code" "unpinned predict after hot delete"
assert_py "$RESP" "risk_score@$BASELINE_LATEST" <<'PYEOF'
import json
import sys

body = json.load(open(sys.argv[1]))
assert body["model"] == sys.argv[2], f"latest did not revert: {body['model']}"
print("latest reverted to:", body["model"])
PYEOF

echo "== version gate: mismatch fails reload, old registry survives =="
MANIFEST="$WORK_MODELS/risk_score/$BASELINE_LATEST/manifest.json"
cp "$MANIFEST" "$MANIFEST.orig"
"$PY" - "$MANIFEST" <<'PYEOF'
import json
import sys

path = sys.argv[1]
manifest = json.load(open(path))
manifest["xgboost_version"] = "9.9.9"
with open(path, "w") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
PYEOF

code=$(http_body "$RESP" POST "$BASE/admin/reload")
expect_code 500 "$code" "reload with mismatched xgboost_version -> 500"
expect_error_contains "$RESP" "refusing to load" "mismatch refusal message"

code=$(http_body "$RESP" POST "$BASE/predict" "$(req risk_score)")
expect_code 200 "$code" "previous registry still serving after failed reload"
assert_py "$RESP" "risk_score@$BASELINE_LATEST" <<'PYEOF'
import json
import sys

body = json.load(open(sys.argv[1]))
assert body["model"] == sys.argv[2], body
print("failed reload kept old registry:", body["model"])
PYEOF

mv "$MANIFEST.orig" "$MANIFEST"
code=$(http_code POST "$BASE/admin/reload")
expect_code 200 "$code" "reload healthy again after manifest restore"

echo "== strict version gate at startup =="
STRICT_MODELS="$WORK/strict_models"
mkdir -p "$STRICT_MODELS/risk_score"
cp -r "$WORK_MODELS/risk_score/$BASELINE_LATEST" "$STRICT_MODELS/risk_score/"
"$PY" - "$STRICT_MODELS/risk_score/$BASELINE_LATEST/manifest.json" <<'PYEOF'
import json
import sys

path = sys.argv[1]
manifest = json.load(open(path))
del manifest["xgboost_version"]
with open(path, "w") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
PYEOF

if XGBOOSTER_STRICT_VERSION=1 "$BIN" --models-dir "$STRICT_MODELS" --lib "$LIB_PATH" \
  --addr "$AUX_ADDR" >"$WORK/strict.log" 2>&1; then
  fail "strict mode must refuse to start when xgboost_version is missing"
fi
expect_contains "$(cat "$WORK/strict.log")" "no xgboost_version" "strict refusal log"

srv_start "$AUX_ADDR" "$STRICT_MODELS"
expect_contains "$(cat "$E2E_SRV_LOG")" "no xgboost_version" "non-strict warn log"
code=$(http_code POST "http://$AUX_ADDR/predict" "$(req risk_score)")
expect_code 200 "$code" "non-strict mode serves the same artifact"
srv_stop

echo "== rate limit recovery after 429 =="
srv_start "$RL_ADDR" "$WORK_MODELS" XGBOOSTER_RATE_LIMIT_RPS=2 XGBOOSTER_RATE_BURST=1
code=$(http_code POST "http://$RL_ADDR/predict" "$(req risk_score)")
expect_code 200 "$code" "first predict consumes the single burst token"
code=$(http_code POST "http://$RL_ADDR/predict" "$(req risk_score)")
expect_code 429 "$code" "immediate second predict -> 429"
sleep 0.8 # at rps=2 a token refills within 0.5s
code=$(http_code POST "http://$RL_ADDR/predict" "$(req risk_score)")
expect_code 200 "$code" "predict succeeds after token refill"
srv_stop

echo "== repository models/ untouched =="
now="$(version_count "$MODELS_DIR/risk_score")"
[ "$now" = "$REPO_COUNT" ] || fail "repo models/ changed: $REPO_COUNT -> $now"
log "repo models/ unchanged ($REPO_COUNT versions)"

srv_stop
echo "stage reload_gates: OK"
