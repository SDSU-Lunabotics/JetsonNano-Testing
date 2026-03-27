#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/zed_ground_wall.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
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
if [[ "${TRACKING:-1}" == "1" ]]; then cmd+=(--tracking); fi
if [[ "${AREA_MEMORY_ENABLE:-1}" == "1" ]]; then cmd+=(--area-memory); fi
if [[ -n "${AREA_LOAD_PATH:-}" ]]; then cmd+=(--area-load-path "${AREA_LOAD_PATH}"); fi
if [[ -n "${AREA_SAVE_PATH:-}" ]]; then cmd+=(--area-save-path "${AREA_SAVE_PATH}"); fi
cmd+=(--area-save-every "${AREA_SAVE_EVERY:-30.0}")
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
cmd+=(--path-connectivity "${PATH_CONNECTIVITY:-8}")
cmd+=(--rover-size-m "${ROVER_SIZE_M:-0.30}")
cmd+=(--start-clear-radius-m "${START_CLEAR_RADIUS_M:-0.35}")
cmd+=(--path-replan-sec "${PATH_REPLAN_SEC:-0.5}")
cmd+=(--floor-update-sec "${FLOOR_UPDATE_SEC:-0.5}")
cmd+=(--floor-min-normal-y "${FLOOR_MIN_NORMAL_Y:-0.5}")
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
  cmd+=(--drive-ready-pulse-sec "${DRIVE_READY_PULSE_SEC:-0.10}")
  cmd+=(--nt-health-period-sec "${NT_HEALTH_PERIOD_SEC:-1.0}")
  cmd+=(--nt-enable-heartbeat-sec "${NT_ENABLE_HEARTBEAT_SEC:-0.10}")
  cmd+=(--nt-command-ack-timeout-sec "${NT_COMMAND_ACK_TIMEOUT_SEC:-0.30}")
  cmd+=(--nt-forward-scale "${NT_FORWARD_SCALE:-1.0}")
  cmd+=(--nt-turn-scale "${NT_TURN_SCALE:-1.0}")
  if [[ "${DRIVE_DEBUG:-0}" == "1" ]]; then cmd+=(--drive-debug); fi
  if [[ "${NT_HEALTH_DEBUG:-0}" == "1" ]]; then cmd+=(--nt-health-debug); fi
  if [[ "${DRIVE_HEADING_FLIP:-1}" == "1" ]]; then cmd+=(--drive-heading-flip); fi
fi

# Optional streaming
if [[ -n "${STREAM_IP:-}" ]]; then
  cmd+=(--stream-ip "${STREAM_IP}")
  cmd+=(--stream-port "${STREAM_PORT:-5600}")
  cmd+=(--stream-view "${STREAM_VIEW:-both}")
fi
if [[ "${NO_GUI:-0}" == "1" ]]; then cmd+=(--no-gui); fi

cd "${REPO_ROOT}"
echo "Running: ${cmd[*]} $*"
exec "${cmd[@]}" "$@"
