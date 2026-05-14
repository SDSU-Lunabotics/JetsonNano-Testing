#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/zed_ground_wall.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

url_uses_local_jetson_api() {
  local url="${1:-}"
  [[ "${url}" == http://127.0.0.1:8000/* || "${url}" == http://localhost:8000/* ]]
}

jetson_api_required=0
if url_uses_local_jetson_api "${CAMERA_HEARTBEAT_URL:-}" ||
   url_uses_local_jetson_api "${CAMERA_PUBLISH_URL:-}" ||
   url_uses_local_jetson_api "${MAP_PUBLISH_URL:-}"; then
  jetson_api_required=1
fi

if [[ "${AUTO_START_JETSON_API:-1}" == "1" && "${jetson_api_required}" == "1" ]]; then
  JETSON_API_BIND_HOST="${JETSON_API_BIND_HOST:-0.0.0.0}"
  JETSON_API_CHECK_HOST="${JETSON_API_CHECK_HOST:-127.0.0.1}"
  JETSON_API_PORT="${JETSON_API_PORT:-8000}"
  JETSON_API_LOG="${JETSON_API_LOG:-${REPO_ROOT}/jetson_api/jetson_api.log}"
  if [[ "${JETSON_API_LOG}" != /* ]]; then
    JETSON_API_LOG="${REPO_ROOT}/${JETSON_API_LOG}"
  fi

  if ! python3 - "${JETSON_API_CHECK_HOST}" "${JETSON_API_PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.25)
    sys.exit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
  then
    if [[ ! -d "${REPO_ROOT}/jetson_api" ]]; then
      echo "Warning: jetson_api folder not found under ${REPO_ROOT}; camera HTTP publishing may fail."
    elif ! python3 -c "import uvicorn" >/dev/null 2>&1; then
      echo "Warning: uvicorn is not installed for python3; camera HTTP publishing may fail."
      echo "Install with: python3 -m pip install -r ${REPO_ROOT}/jetson_api/requirements.txt"
    else
      echo "Starting Jetson API on ${JETSON_API_BIND_HOST}:${JETSON_API_PORT}..."
      (
        cd "${REPO_ROOT}/jetson_api"
        exec python3 -m uvicorn app.main:app --host "${JETSON_API_BIND_HOST}" --port "${JETSON_API_PORT}"
      ) >"${JETSON_API_LOG}" 2>&1 &
      JETSON_API_PID=$!

      for _ in {1..30}; do
        if python3 - "${JETSON_API_CHECK_HOST}" "${JETSON_API_PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.25)
    sys.exit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
        then
          echo "Jetson API started on ${JETSON_API_BIND_HOST}:${JETSON_API_PORT}."
          break
        fi

        if ! kill -0 "${JETSON_API_PID}" 2>/dev/null; then
          echo "Warning: Jetson API exited during startup. See ${JETSON_API_LOG}."
          break
        fi

        sleep 0.2
      done
    fi
  fi
fi

cmd=(python3 "${SCRIPT_DIR}/zed_ground_wall.py" --camera-only --manual-start)

if [[ "${DRIVE:-1}" == "1" ]]; then
  cmd+=(--drive)
  cmd+=(--roborio-ip "${ROBORIO_IP:-10.0.9.2}")
  cmd+=(--drive-speed "${DRIVE_SPEED:-1.0}")
  cmd+=(--main-rover-mode)
  if [[ "${HARD_DRIVE_FLIP:-1}" == "1" ]]; then cmd+=(--hard-drive-flip); fi
  if [[ "${DRIVE_HEADING_FLIP:-1}" == "1" ]]; then cmd+=(--drive-heading-flip); fi
  if [[ "${DS_JOYSTICK:-1}" == "1" ]]; then cmd+=(--ds-joystick); fi
  cmd+=(--ds-joystick-fwd-key "${DS_JOYSTICK_FWD_KEY:-DS/JoystickFwd}")
  cmd+=(--ds-joystick-turn-key "${DS_JOYSTICK_TURN_KEY:-DS/JoystickTurn}")
  cmd+=(--ds-joystick-scale "${DS_JOYSTICK_SCALE:-0.5}")
fi

if [[ "${NO_GUI:-0}" == "1" ]]; then cmd+=(--no-gui); fi

echo "Running camera-only mode: ${cmd[*]}"
exec "${cmd[@]}"
