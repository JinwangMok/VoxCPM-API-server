#!/usr/bin/env bash
# End-to-end check for a running VoxCPM API server.
#
# Usage:
#   ./tests/e2e.sh                       # defaults to http://localhost:9100
#   BASE_URL=http://dgx-a:9100 ./tests/e2e.sh
#
# Exits non-zero on any failure. Prints a short summary on success.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:9100}"
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
INPUT_TEXT="${INPUT_TEXT:-Hello from the VoxCPM API server. End to end verification is running.}"

pass() { printf "  \033[32mok\033[0m   %s\n" "$1"; }
fail() { printf "  \033[31mfail\033[0m %s\n" "$1" >&2; exit 1; }

echo "==> target: ${BASE_URL}"

echo "==> GET /health"
health=$(curl -fsS "${BASE_URL}/health") || fail "/health not reachable"
echo "    ${health}"
pass "/health responded"

echo "==> GET /v1/models"
curl -fsS "${BASE_URL}/v1/models" | grep -q '"voxcpm2"' \
  && pass "/v1/models lists voxcpm2" \
  || fail "/v1/models missing voxcpm2"

check_wav() {
  local path="$1"
  [[ -s "${path}" ]] || fail "empty output: ${path}"
  local header
  header=$(head -c 12 "${path}" | xxd -p)
  # "RIFF" .... "WAVE" => 52494646 ???????? 57415645
  [[ "${header:0:8}"  == "52494646" ]] || fail "missing RIFF header in ${path}"
  [[ "${header:16:8}" == "57415645" ]] || fail "missing WAVE marker in ${path}"
  local bytes
  bytes=$(wc -c < "${path}")
  (( bytes > 10000 )) || fail "wav too small (${bytes} bytes): ${path}"
  pass "wav valid (${bytes} bytes): ${path}"
}

for voice in alloy nova onyx; do
  out="${OUT_DIR}/${voice}.wav"
  echo "==> POST /v1/audio/speech voice=${voice}"
  curl -fsS -X POST "${BASE_URL}/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -o "${out}" \
    -d "$(printf '{"model":"voxcpm2","voice":"%s","input":%s,"response_format":"wav"}' \
          "${voice}" "$(printf '%s' "${INPUT_TEXT}" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')")"
  check_wav "${out}"
done

echo ""
echo "all checks passed. artifacts: ${OUT_DIR}"
