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
      echo "Warning: jetson_api folder not found under ${REPO_ROOT}; camera/map HTTP publishing may fail."
    elif ! python3 -c "import uvicorn" >/dev/null 2>&1; then
      echo "Warning: uvicorn is not installed for python3; camera/map HTTP publishing may fail."
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

# Optional: auto-start TigerVNC before launching the app.
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
      if ! "${vnc_cmd[@]}"; then
        echo "Warning: failed to start TigerVNC on ${VNC_DISPLAY}. Continuing launch."
      fi
    else
      echo "TigerVNC already running on ${VNC_DISPLAY}."
    fi
  else
    echo "Warning: tigervncserver not found. Skipping VNC auto-start."
  fi
fi

cmd=(python3 "${SCRIPT_DIR}/zed_ground_wall.py")

# Core mapping/tracking
if [[ "${COMPLEX_MAPPING:-0}" == "1" ]]; then cmd+=(--complex); fi
if [[ "${TRACKING:-1}" == "1" ]]; then
  cmd+=(--tracking)
else
  cmd+=(--no-tracking)
fi
if [[ "${AREA_MEMORY_ENABLE:-1}" == "1" ]]; then cmd+=(--area-memory); fi
if [[ -n "${AREA_LOAD_PATH:-}" ]]; then cmd+=(--area-load-path "${AREA_LOAD_PATH}"); fi
if [[ -n "${AREA_SAVE_PATH:-}" ]]; then cmd+=(--area-save-path "${AREA_SAVE_PATH}"); fi
cmd+=(--area-save-every "${AREA_SAVE_EVERY:-30.0}")
cmd+=(--tracking-max-pose-jump-m "${TRACKING_MAX_POSE_JUMP_M:-0.80}")
cmd+=(--tracking-max-heading-jump-deg "${TRACKING_MAX_HEADING_JUMP_DEG:-55.0}")
cmd+=(--tracking-recover-stable-frames "${TRACKING_RECOVER_STABLE_FRAMES:-6}")
cmd+=(--recovery-save-every "${RECOVERY_SAVE_EVERY:-1.0}")
if [[ "${RECOVERY_LOAD:-1}" != "1" ]]; then cmd+=(--no-recovery-load); fi
if [[ "${RECOVERY_NT_MIRROR:-1}" != "1" ]]; then cmd+=(--no-recovery-nt-mirror); fi
cmd+=(--localize-turn-speed "${LOCALIZE_TURN_SPEED:-0.25}")
cmd+=(--localize-scan-sec "${LOCALIZE_SCAN_SEC:-8.0}")
cmd+=(--localize-max-sec "${LOCALIZE_MAX_SEC:-20.0}")
if [[ "${MAP_CENTER:-1}" == "1" ]]; then cmd+=(--map-center); fi
if [[ "${MAP_FOLLOW_ROVER:-1}" == "1" ]]; then cmd+=(--map-follow-rover); fi
cmd+=(--map-scale "${MAP_SCALE:-3}")
cmd+=(--free-decay "${FREE_DECAY:-1.0}")
cmd+=(--free-decay-unconfirmed "${FREE_DECAY_UNCONFIRMED:-0.995}")
cmd+=(--free-decay-confirmed "${FREE_DECAY_CONFIRMED:-1.0}")
cmd+=(--free-confirm-hits "${FREE_CONFIRM_HITS:-8}")
cmd+=(--free-confirm-ratio "${FREE_CONFIRM_RATIO:-1.2}")
cmd+=(--free-downgrade-factor "${FREE_DOWNGRADE_FACTOR:-0.6}")
cmd+=(--occ-decay "${OCC_DECAY:-0.98}")
cmd+=(--hole-decay "${HOLE_DECAY:-0.98}")
cmd+=(--obstacle-thresh-m "${OBSTACLE_THRESH_M:-0.05}")
cmd+=(--hole-thresh-m "${HOLE_THRESH_M:-0.05}")
cmd+=(--max-above-ground-m "${MAX_ABOVE_GROUND_M:-1.22}")
if [[ "${DISABLE_HOLES:-0}" == "1" ]]; then cmd+=(--disable-holes); fi
cmd+=(--path-avoid-occ-min "${PATH_AVOID_OCC_MIN:-5.0}")
cmd+=(--path-avoid-occ-ratio "${PATH_AVOID_OCC_RATIO:-1.8}")
cmd+=(--path-avoid-occ-advantage "${PATH_AVOID_OCC_ADVANTAGE:-3.0}")
cmd+=(--path-connectivity "${PATH_CONNECTIVITY:-8}")
cmd+=(--rover-size-m "${ROVER_SIZE_M:-0.30}")
cmd+=(--camera-mount "${CAMERA_MOUNT:-front}")
if [[ -n "${CAMERA_MOUNT_YAW_DEG:-}" ]]; then cmd+=(--camera-mount-yaw-deg "${CAMERA_MOUNT_YAW_DEG}"); fi
if [[ -n "${CAMERA_FORWARD_OFFSET_M:-}" ]]; then cmd+=(--camera-forward-offset-m "${CAMERA_FORWARD_OFFSET_M}"); fi
if [[ -n "${CAMERA_RIGHT_OFFSET_M:-}" ]]; then cmd+=(--camera-right-offset-m "${CAMERA_RIGHT_OFFSET_M}"); fi
if [[ "${CAMERA_SERVO_TRACK:-0}" == "1" ]]; then cmd+=(--camera-servo-track); fi
if [[ "${CAMERA_SERVO_INVERT:-0}" == "1" ]]; then cmd+=(--camera-servo-invert); fi
if [[ "${DISPLAY_HEADING_FLIP:-0}" == "1" ]]; then cmd+=(--display-heading-flip); fi
cmd+=(--camera-map-angle-deg "${CAMERA_MAP_ANGLE_DEG:-180.0}")
cmd+=(--camera-deposit-angle-deg "${CAMERA_DEPOSIT_ANGLE_DEG:-0.0}")
cmd+=(--camera-servo-map-tol-deg "${CAMERA_SERVO_MAP_TOL_DEG:-8.0}")
cmd+=(--start-clear-radius-m "${START_CLEAR_RADIUS_M:-0.35}")
cmd+=(--path-replan-sec "${PATH_REPLAN_SEC:-0.5}")
cmd+=(--path-soft-clearance-cells "${PATH_SOFT_CLEARANCE_CELLS:-8}")
if [[ "${PATH_RELAX_ON_FAIL:-1}" != "1" ]]; then cmd+=(--no-path-relax-on-fail); fi
if [[ "${ALLOW_DIRECT_NO_PATH:-0}" == "1" ]]; then cmd+=(--allow-direct-no-path); fi
cmd+=(--floor-update-sec "${FLOOR_UPDATE_SEC:-0.5}")
cmd+=(--floor-min-normal-y "${FLOOR_MIN_NORMAL_Y:-0.5}")
cmd+=(--plane-ema-alpha "${PLANE_EMA_ALPHA:-0.25}")
cmd+=(--plane-max-tilt-delta-deg "${PLANE_MAX_TILT_DELTA_DEG:-8.0}")
cmd+=(--plane-max-height-jump-m "${PLANE_MAX_HEIGHT_JUMP_M:-0.08}")
cmd+=(--plane-force-accept-rejects "${PLANE_FORCE_ACCEPT_REJECTS:-20}")
cmd+=(--sample-stride "${SAMPLE_STRIDE:-8}")
cmd+=(--min-range-z-m "${MIN_RANGE_Z_M:-0.25}")
cmd+=(--max-range-z-m "${MAX_RANGE_Z_M:-6.0}")
cmd+=(--map-save-path "${MAP_SAVE_PATH:-${SCRIPT_DIR}/zed_map.npz}")
cmd+=(--map-save-every "${MAP_SAVE_EVERY:-5.0}")
if [[ "${MAP_LOAD:-1}" == "1" ]]; then cmd+=(--map-load); fi
if [[ "${HEATMAP:-0}" == "1" ]]; then
  cmd+=(--heatmap)
  cmd+=(--heatmap-mode "${HEATMAP_MODE:-risk}")
  cmd+=(--heatmap-alpha "${HEATMAP_ALPHA:-0.35}")
  cmd+=(--heatmap-min-evidence "${HEATMAP_MIN_EVIDENCE:-1.0}")
  if [[ "${HEATMAP_WINDOW:-0}" == "1" ]]; then cmd+=(--heatmap-window); fi
fi

# Drive
if [[ "${DRIVE:-1}" == "1" ]]; then
  cmd+=(--drive)
  cmd+=(--roborio-ip "${ROBORIO_IP:-10.0.9.2}")
  cmd+=(--drive-speed "${DRIVE_SPEED:-0.7}")
  cmd+=(--drive-forward-slew-per-sec "${DRIVE_FORWARD_SLEW_PER_SEC:-1.4}")
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
  if [[ -n "${DRIVE_HEADING_FLIP:-}" ]]; then
    if [[ "${DRIVE_HEADING_FLIP}" == "1" ]]; then cmd+=(--drive-heading-flip); fi
  elif [[ "${CAMERA_MOUNT:-front}" != "rear" ]]; then
    cmd+=(--drive-heading-flip)
  fi
  if [[ "${HARD_DRIVE_FLIP:-0}" == "1" ]]; then cmd+=(--hard-drive-flip); fi
  if [[ "${DS_JOYSTICK:-1}" == "1" ]]; then cmd+=(--ds-joystick); fi
  cmd+=(--ds-joystick-fwd-key "${DS_JOYSTICK_FWD_KEY:-DS/JoystickFwd}")
  cmd+=(--ds-joystick-turn-key "${DS_JOYSTICK_TURN_KEY:-DS/JoystickTurn}")
  cmd+=(--ds-joystick-scale "${DS_JOYSTICK_SCALE:-0.5}")
  if [[ "${DRIVER_PRIORITY_MODE:-1}" == "1" ]]; then
    cmd+=(--driver-priority-mode)
  else
    cmd+=(--no-driver-priority-mode)
  fi
  cmd+=(--driver-priority-threshold "${DRIVER_PRIORITY_THRESHOLD:-0.12}")
  cmd+=(--driver-priority-sample-stride "${DRIVER_PRIORITY_SAMPLE_STRIDE:-12}")
fi

# Optional streaming
if [[ -n "${STREAM_IP:-}" ]]; then
  cmd+=(--stream-ip "${STREAM_IP}")
  cmd+=(--stream-port "${STREAM_PORT:-5600}")
  cmd+=(--stream-view "${STREAM_VIEW:-both}")
fi
if [[ -n "${MAP_COMMAND_FILE:-}" ]]; then
  cmd+=(--map-command-file "${MAP_COMMAND_FILE}")
fi
if [[ -n "${CAMERA_HEARTBEAT_URL:-}" ]]; then
  cmd+=(--camera-heartbeat-url "${CAMERA_HEARTBEAT_URL}")
  cmd+=(--camera-heartbeat-interval-ms "${CAMERA_HEARTBEAT_INTERVAL_MS:-1000}")
  cmd+=(--camera-heartbeat-timeout-ms "${CAMERA_HEARTBEAT_TIMEOUT_MS:-250}")
  cmd+=(--camera-heartbeat-source "${CAMERA_HEARTBEAT_SOURCE:-zed_ground_wall}")
fi
if [[ -n "${CAMERA_PUBLISH_URL:-}" ]]; then
  cmd+=(--camera-publish-url "${CAMERA_PUBLISH_URL}")
  cmd+=(--camera-publish-interval-ms "${CAMERA_PUBLISH_INTERVAL_MS:-120}")
  cmd+=(--camera-publish-jpeg-quality "${CAMERA_PUBLISH_JPEG_QUALITY:-75}")
  cmd+=(--camera-publish-timeout-ms "${CAMERA_PUBLISH_TIMEOUT_MS:-250}")
  cmd+=(--camera-publish-source "${CAMERA_PUBLISH_SOURCE:-zed_ground_wall}")
fi
if [[ -n "${MAP_PUBLISH_URL:-}" ]]; then
  cmd+=(--map-publish-url "${MAP_PUBLISH_URL}")
  cmd+=(--map-publish-interval-ms "${MAP_PUBLISH_INTERVAL_MS:-120}")
  cmd+=(--map-publish-jpeg-quality "${MAP_PUBLISH_JPEG_QUALITY:-70}")
  cmd+=(--map-publish-timeout-ms "${MAP_PUBLISH_TIMEOUT_MS:-250}")
  cmd+=(--map-publish-source "${MAP_PUBLISH_SOURCE:-zed_ground_wall}")
fi
if [[ "${NO_GUI:-0}" == "1" ]]; then cmd+=(--no-gui); fi
if [[ "${OVERLAY_RED_ONLY:-0}" == "1" ]]; then cmd+=(--overlay-red-only); fi
if [[ "${MANUAL_START:-0}" == "1" ]]; then cmd+=(--manual-start); fi

# Human detection
if [[ "${HUMAN_DETECT:-1}" == "1" ]]; then
  cmd+=(--human-detect)
  cmd+=(--human-od-confidence "${HUMAN_OD_CONFIDENCE:-40}")
  cmd+=(--human-od-every "${HUMAN_OD_EVERY:-1}")
  cmd+=(--human-stop-m "${HUMAN_STOP_M:-1.5}")
  cmd+=(--human-slow-m "${HUMAN_SLOW_M:-3.0}")
  cmd+=(--human-min-conf "${HUMAN_MIN_CONF:-0.40}")
fi

# Rock detection (custom YOLO model)
if [[ -n "${ROCK_MODEL:-}" ]]; then
  cmd+=(--rock-model "${ROCK_MODEL}")
  cmd+=(--rock-conf "${ROCK_CONF:-0.35}")
  cmd+=(--rock-every "${ROCK_EVERY:-5}")
  cmd+=(--rock-stamp "${ROCK_STAMP:-6.0}")
  cmd+=(--rock-classes "${ROCK_CLASSES:-rock,stone,boulder}")
fi
if [[ "${LANDMARK_MEMORY:-1}" == "1" ]]; then
  cmd+=(--landmark-memory)
else
  cmd+=(--no-landmark-memory)
fi
cmd+=(--landmark-classes "${LANDMARK_CLASSES:-backpack,rock,stone,boulder,obstacle}")
cmd+=(--landmark-path "${LANDMARK_PATH:-${SCRIPT_DIR}/zed_landmarks.json}")
cmd+=(--landmark-assoc-m "${LANDMARK_ASSOC_M:-0.45}")
cmd+=(--landmark-min-hits "${LANDMARK_MIN_HITS:-2}")
cmd+=(--landmark-save-every "${LANDMARK_SAVE_EVERY:-5.0}")
if [[ "${LANDMARK_RELOCALIZE:-1}" == "1" ]]; then
  cmd+=(--landmark-relocalize)
else
  cmd+=(--no-landmark-relocalize)
fi
cmd+=(--landmark-relocalize-max-offset-m "${LANDMARK_RELOCALIZE_MAX_OFFSET_M:-4.0}")
cmd+=(--landmark-relocalize-alpha "${LANDMARK_RELOCALIZE_ALPHA:-0.65}")

cd "${REPO_ROOT}"
echo "Running: ${cmd[*]} $*"
exec "${cmd[@]}" "$@"
