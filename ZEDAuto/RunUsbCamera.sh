#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/usb_camera.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

url_uses_local_jetson_api() {
  local url="${1:-}"
  [[ "${url}" == http://127.0.0.1:8000/* || "${url}" == http://localhost:8000/* ]]
}

jetson_api_required=0
if url_uses_local_jetson_api "${USB_CAMERA_HEARTBEAT_URL:-}" ||
   url_uses_local_jetson_api "${USB_CAMERA_PUBLISH_URL:-}"; then
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
    echo "Starting Jetson API on ${JETSON_API_BIND_HOST}:${JETSON_API_PORT}..."
    (
      cd "${REPO_ROOT}/jetson_api"
      exec python3 -m uvicorn app.main:app --host "${JETSON_API_BIND_HOST}" --port "${JETSON_API_PORT}"
    ) >"${JETSON_API_LOG}" 2>&1 &
  fi
fi

if [[ "${AUTO_START_VNC:-0}" == "1" ]]; then
  VNC_DISPLAY="${VNC_DISPLAY:-:2}"
  VNC_XSTARTUP="${VNC_XSTARTUP:-$HOME/.vnc/xstartup}"
  VNC_LOCALHOST_NO="${VNC_LOCALHOST_NO:-1}"

  if command -v tigervncserver >/dev/null 2>&1; then
    if ! tigervncserver -list 2>/dev/null | grep -qE "^[[:space:]]*${VNC_DISPLAY}[[:space:]]"; then
      vnc_cmd=(tigervncserver "${VNC_DISPLAY}")
      if [[ "${VNC_LOCALHOST_NO}" == "1" ]]; then
        vnc_cmd+=(-localhost no)
      fi
      if [[ -x "${VNC_XSTARTUP}" ]]; then
        vnc_cmd+=(-xstartup "${VNC_XSTARTUP}")
      fi
      echo "Starting TigerVNC on ${VNC_DISPLAY}..."
      "${vnc_cmd[@]}" || echo "Warning: failed to start TigerVNC on ${VNC_DISPLAY}. Continuing launch."
    fi
  fi
fi

if [[ -n "${USB_CAMERA_DISPLAY:-}" ]]; then
  export DISPLAY="${USB_CAMERA_DISPLAY}"
fi

exec python3 "${SCRIPT_DIR}/usb_camera_viewer.py" "$@"
