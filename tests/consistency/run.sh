#!/usr/bin/env bash
# Consistency + HTTP smoke test for the Rust inference side.
set -euo pipefail
cd "$(dirname "$0")/../.."

BIN=infer/target/debug/xgbooster-infer
ADDR=127.0.0.1:18099
MODELS_DIR=models

echo "== build + cargo consistency test =="
(cd infer && cargo build && cargo test --test consistency)

if [ -z "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
  echo "== models dir empty; waiting for xgboost in .venv =="
  waited=0
  until .venv/bin/python -c "import xgboost" >/dev/null 2>&1; do
    if [ "$waited" -ge 900 ]; then
      echo "ERROR: timed out waiting for .venv xgboost install" >&2
      exit 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
  echo "== training fallback model =="
  PYTHONPATH=train .venv/bin/python -m xgbooster_train.train --models-dir models --trials 0
fi

latest_version_of() { # latest_version_of MODEL_DIR -> newest version name
  .venv/bin/python - "$1" <<'PYEOF'
import pathlib
import sys


def sort_key(name):
    # Mirror the Rust registry pick_latest: trailing "-<digits>" suffix is
    # compared numerically, otherwise the full name lexicographically.
    base, sep, suffix = name.rpartition("-")
    if sep and suffix.isascii() and suffix.isdigit():
        return (base, int(suffix))
    return (name, -1)


names = [p.name for p in pathlib.Path(sys.argv[1]).iterdir() if p.is_dir()]
if not names:
    raise SystemExit("no model version directories found")
print(max(names, key=sort_key))
PYEOF
}

# build_request MODEL OUT_FILE: JSON /predict body from the first holdout row
# of the model's latest artifact (empty cells -> null, matching NaN training).
build_request() {
  local model="$1" out="$2"
  local artifact="$MODELS_DIR/$model/$(latest_version_of "$MODELS_DIR/$model")"
  .venv/bin/python - "$model" "$artifact/manifest.json" \
    "$artifact/holdout.csv" "$out" <<'PYEOF'
import csv
import json
import sys

model, manifest_path, holdout_path, out_path = sys.argv[1:5]
manifest = json.load(open(manifest_path))
with open(holdout_path, newline="") as fh:
    rows = list(csv.reader(fh))
header, data = rows[0], rows[1]
features = {}
for f in manifest["feature_schema"]:
    cell = data[header.index(f["name"])]
    if f["type"] == "categorical":
        features[f["name"]] = cell
    else:
        try:
            features[f["name"]] = float(cell)
        except ValueError:
            features[f["name"]] = None
request = {"model": model, "features": features}
json.dump(request, open(out_path, "w"))
print("request:", json.dumps(request))
PYEOF
}

echo "== starting server =="
SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
"$BIN" --models-dir "$MODELS_DIR" --lib infer/lib/libxgboost.so --addr "$ADDR" >/tmp/server.log 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 100); do
  if curl -sf "http://$ADDR/models" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server exited early"; cat /tmp/server.log; exit 1
  fi
  sleep 0.2
done

echo "== GET /models =="
curl -sf "http://$ADDR/models" -o /tmp/models.json
.venv/bin/python -m json.tool < /tmp/models.json
MODEL_NAMES=$(find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d \
  ! -name '.*' -printf '%f\n' | sort)
.venv/bin/python - "$MODELS_DIR" /tmp/models.json <<'PYEOF'
import json
import pathlib
import sys

models_dir = pathlib.Path(sys.argv[1])
listed = {m["name"] for m in json.load(open(sys.argv[2]))["models"]}
on_disk = {p.name for p in models_dir.iterdir() if p.is_dir()}
assert listed == on_disk, f"/models mismatch: {listed} vs {on_disk}"
print(f"/models lists every model: {sorted(listed)}")
PYEOF

echo "== POST /predict smoke per model =="
for MODEL in $MODEL_NAMES; do
  ARTIFACT="$MODELS_DIR/$MODEL/$(latest_version_of "$MODELS_DIR/$MODEL")"
  echo "-- model $MODEL ($ARTIFACT)"
  build_request "$MODEL" /tmp/predict_req.json
  curl -sf -X POST "http://$ADDR/predict" -H 'Content-Type: application/json' \
    -d @/tmp/predict_req.json -o /tmp/predict_resp.json
  cat /tmp/predict_resp.json; echo
  .venv/bin/python - "$ARTIFACT" /tmp/predict_resp.json <<'PYEOF'
import csv
import json
import sys

artifact = sys.argv[1]
manifest = json.load(open(f"{artifact}/manifest.json"))
names = {f["name"] for f in manifest["feature_schema"]}
with open(f"{artifact}/holdout.csv", newline="") as fh:
    rows = list(csv.reader(fh))
header, data = rows[0], rows[1]
others = [i for i, h in enumerate(header) if h not in names]
score_idx = None
for i in reversed(others):
    low = header[i].lower()
    if "score" in low or "pred" in low or "prob" in low:
        score_idx = i
        break
if score_idx is None:
    score_idx = others[-1]
expected = float(data[score_idx])
got = json.load(open(sys.argv[2]))["score"]
diff = abs(got - expected)
assert diff < 1e-6, f"smoke mismatch: got {got}, expected {expected}, diff {diff}"
print(f"smoke score ok: got={got} expected={expected} diff={diff}")
PYEOF
done

echo "== POST /admin/reload =="
curl -sf -X POST "http://$ADDR/admin/reload" | .venv/bin/python -m json.tool

expect_code() {
  local expect="$1" got="$2" desc="$3"
  if [ "$got" != "$expect" ]; then
    echo "ERROR: $desc: expected HTTP $expect, got $got" >&2
    exit 1
  fi
  echo "$desc: HTTP $got"
}

echo "== negative HTTP checks =="
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$ADDR/predict" \
  -H 'Content-Type: application/json' -d '{"model":"risk_score@","features":{}}')
expect_code 400 "$code" "invalid model reference -> 400"

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$ADDR/predict" \
  -H 'Content-Type: application/json' -d '{"model":"nope","features":{}}')
expect_code 404 "$code" "unknown model -> 404"

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$ADDR/predict" \
  -H 'Content-Type: application/json' -d '{"model":"risk_score","features":{"amount":"oops"}}')
expect_code 400 "$code" "bad features type -> 400"

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$ADDR/predict" \
  -H 'Content-Type: application/json' -d '{"model":"risk_score","features":{"hour":3}}')
expect_code 200 "$code" "missing feature still predicts -> 200"

curl -s -X POST "http://$ADDR/predict" -H 'Content-Type: application/json' \
  -d '{"model":"risk_score","features":{"hour":3}}' -o /tmp/predict_missing.json
.venv/bin/python - <<'PYEOF'
import json

resp = json.load(open("/tmp/predict_missing.json"))
assert "amount" in resp.get("missing_features", []), f"amount not reported missing: {resp}"
print("missing_features reporting ok:", resp["missing_features"])
PYEOF

echo "== admin auth =="
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

XGBOOSTER_ADMIN_TOKEN=test-token-123 \
  "$BIN" --models-dir "$MODELS_DIR" --lib infer/lib/libxgboost.so --addr "$ADDR" \
  >/tmp/server-auth.log 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 100); do
  if curl -sf "http://$ADDR/models" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: auth server exited early"; cat /tmp/server-auth.log; exit 1
  fi
  sleep 0.2
done

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$ADDR/admin/reload")
expect_code 401 "$code" "admin reload without token -> 401"

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$ADDR/admin/reload" \
  -H "Authorization: Bearer test-token-123")
expect_code 200 "$code" "admin reload with bearer token -> 200"

echo "== non-loopback startup guard =="
if "$BIN" --models-dir "$MODELS_DIR" --lib infer/lib/libxgboost.so \
  --addr 0.0.0.0:18097 >/tmp/server-guard.log 2>&1; then
  echo "ERROR: server must refuse non-loopback bind without token/rate limits" >&2
  exit 1
fi
grep -q "XGBOOSTER_ADMIN_TOKEN" /tmp/server-guard.log || {
  echo "ERROR: guard message missing env var names" >&2; cat /tmp/server-guard.log; exit 1
}
echo "non-loopback without guards refused to start: OK"

echo "== rate limit + strict version gate =="
ADDR2=127.0.0.1:18098
XGBOOSTER_RATE_LIMIT_RPS=0.001 XGBOOSTER_RATE_BURST=1 \
  XGBOOSTER_MAX_INFLIGHT=2 XGBOOSTER_STRICT_VERSION=1 \
  "$BIN" --models-dir "$MODELS_DIR" --lib infer/lib/libxgboost.so --addr "$ADDR2" \
  >/tmp/server-limited.log 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 100); do
  if curl -sf "http://$ADDR2/models" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: limited server exited early"; cat /tmp/server-limited.log; exit 1
  fi
  sleep 0.2
done

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$ADDR2/predict" \
  -H 'Content-Type: application/json' -d @/tmp/predict_req.json)
expect_code 200 "$code" "first predict under rate limit -> 200"

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://$ADDR2/predict" \
  -H 'Content-Type: application/json' -d @/tmp/predict_req.json)
expect_code 429 "$code" "second predict immediately -> 429 (rate limited)"

curl -s -D - -o /dev/null -X POST "http://$ADDR2/predict" \
  -H 'Content-Type: application/json' -d @/tmp/predict_req.json \
  | grep -qi '^retry-after: 1' || {
  echo "ERROR: 429 response missing Retry-After: 1 header" >&2; exit 1
}
echo "429 carries Retry-After: 1"

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
SERVER_PID=""

echo "SMOKE OK"
