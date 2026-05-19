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

cmd=(python3 "${SCRIPT_DIR}/zed_ground_wall.py" --manual-start)

if [[ "${REC_CAMERA_ONLY:-0}" == "1" ]]; then
  cmd+=(--camera-only)
fi

if [[ "${DRIVE:-1}" == "1" ]]; then
  cmd+=(--drive)
  cmd+=(--roborio-ip "${ROBORIO_IP:-10.0.9.2}")
  cmd+=(--drive-speed "${DRIVE_SPEED:-1.0}")
  cmd+=(--drive-forward-slew-per-sec "${DRIVE_FORWARD_SLEW_PER_SEC:-1.4}")
  cmd+=(--drive-turn-k "${DRIVE_TURN_K:-1.15}")
  cmd+=(--drive-max-turn-cmd "${DRIVE_MAX_TURN_CMD:-1.00}")
  cmd+=(--drive-slow-turn-deg "${DRIVE_SLOW_TURN_DEG:-20.0}")
  cmd+=(--drive-stop-turn-deg "${DRIVE_STOP_TURN_DEG:-40.0}")
  cmd+=(--drive-min-turn-forward-scale "${DRIVE_MIN_TURN_FORWARD_SCALE:-0.20}")
  cmd+=(--drive-min-arc-forward-scale "${DRIVE_MIN_ARC_FORWARD_SCALE:-0.32}")
  cmd+=(--drive-arc-turn-limit-deg "${DRIVE_ARC_TURN_LIMIT_DEG:-110.0}")
  cmd+=(--drive-goal-tol-m "${DRIVE_GOAL_TOL_M:-0.45}")
  cmd+=(--drive-heading-tol-deg "${DRIVE_HEADING_TOL_DEG:-16.0}")
  cmd+=(--backup-close-dist-m "${BACKUP_CLOSE_DIST_M:-0.45}")
  cmd+=(--backup-lane-half-width-m "${BACKUP_LANE_HALF_WIDTH_M:-0.35}")
  cmd+=(--backup-min-obstacle-points "${BACKUP_MIN_OBSTACLE_POINTS:-30}")
  cmd+=(--backup-critical-dist-m "${BACKUP_CRITICAL_DIST_M:-0.30}")
  cmd+=(--backup-critical-min-points "${BACKUP_CRITICAL_MIN_POINTS:-6}")
  cmd+=(--backup-speed "${BACKUP_SPEED:-0.35}")
  cmd+=(--backup-hold-sec "${BACKUP_HOLD_SEC:-0.40}")
  cmd+=(--drive-ready-pulse-sec "${DRIVE_READY_PULSE_SEC:-0.10}")
  cmd+=(--nt-health-period-sec "${NT_HEALTH_PERIOD_SEC:-1.0}")
  cmd+=(--nt-enable-heartbeat-sec "${NT_ENABLE_HEARTBEAT_SEC:-0.10}")
  cmd+=(--nt-command-ack-timeout-sec "${NT_COMMAND_ACK_TIMEOUT_SEC:-0.30}")
  cmd+=(--nt-forward-scale "${NT_FORWARD_SCALE:-1.0}")
  cmd+=(--nt-turn-scale "${NT_TURN_SCALE:-1.0}")
  if [[ "${MAIN_ROVER_MODE:-1}" == "1" ]]; then
    cmd+=(--main-rover-mode)
  else
    cmd+=(--no-main-rover-mode)
  fi
  if [[ "${MAIN_ROVER_DEBUG:-0}" == "1" ]]; then cmd+=(--main-rover-debug); fi
  if [[ "${DRIVE_DEBUG:-0}" == "1" ]]; then cmd+=(--drive-debug); fi
  if [[ "${NT_HEALTH_DEBUG:-0}" == "1" ]]; then cmd+=(--nt-health-debug); fi
  if [[ "${HARD_DRIVE_FLIP:-1}" == "1" ]]; then cmd+=(--hard-drive-flip); fi
  if [[ -n "${DRIVE_HEADING_FLIP:-}" ]]; then
    if [[ "${DRIVE_HEADING_FLIP}" == "1" ]]; then cmd+=(--drive-heading-flip); fi
  elif [[ "${CAMERA_MOUNT:-front}" != "rear" ]]; then
    cmd+=(--drive-heading-flip)
  fi
  if [[ "${DRIVER_PRIORITY_MODE:-1}" == "1" ]]; then
    cmd+=(--driver-priority-mode)
  else
    cmd+=(--no-driver-priority-mode)
  fi
  cmd+=(--driver-priority-threshold "${DRIVER_PRIORITY_THRESHOLD:-0.12}")
  cmd+=(--driver-priority-sample-stride "${DRIVER_PRIORITY_SAMPLE_STRIDE:-12}")
  if [[ "${DS_JOYSTICK:-1}" == "1" ]]; then cmd+=(--ds-joystick); fi
  cmd+=(--ds-joystick-fwd-key "${DS_JOYSTICK_FWD_KEY:-DS/JoystickFwd}")
  cmd+=(--ds-joystick-turn-key "${DS_JOYSTICK_TURN_KEY:-DS/JoystickTurn}")
  cmd+=(--ds-joystick-scale "${DS_JOYSTICK_SCALE:-0.5}")
fi

if [[ "${NO_GUI:-0}" == "1" ]]; then cmd+=(--no-gui); fi

echo "Running record mode: ${cmd[*]}"
exec "${cmd[@]}"
