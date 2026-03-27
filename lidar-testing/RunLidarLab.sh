#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/lidar_lab.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

cmd=(python3 "${SCRIPT_DIR}/lidar_ground_wall.py")

default_sdk_cmd="python3 ${SCRIPT_DIR}/unitree_sdk_bridge.py"

cmd+=(--input-mode "${INPUT_MODE:-sdk}")
cmd+=(--sdk-cmd "${SDK_CMD:-${default_sdk_cmd}}")
cmd+=(--sdk-startup-timeout-sec "${SDK_STARTUP_TIMEOUT_SEC:-5.0}")
cmd+=(--topic "${LIDAR_TOPIC:-/utlidar/cloud}")
cmd+=(--queue-size "${QUEUE_SIZE:-5}")
cmd+=(--frame-stride "${FRAME_STRIDE:-1}")
cmd+=(--stride "${POINT_STRIDE:-2}")
cmd+=(--up-axis "${UP_AXIS:-z}")
cmd+=(--forward-axis "${FORWARD_AXIS:-x}")
cmd+=(--lateral-axis "${LATERAL_AXIS:-y}")

cmd+=(--min-range-m "${MIN_RANGE_M:-0.20}")
cmd+=(--max-range-m "${MAX_RANGE_M:-12.0}")
cmd+=(--min-forward-m "${MIN_FORWARD_M:--1.0}")
cmd+=(--max-forward-m "${MAX_FORWARD_M:-12.0}")
cmd+=(--max-abs-lateral-m "${MAX_ABS_LATERAL_M:-10.0}")

cmd+=(--ground-thresh-m "${GROUND_THRESH_M:-0.08}")
cmd+=(--obstacle-thresh-m "${OBSTACLE_THRESH_M:-0.08}")
cmd+=(--hole-thresh-m "${HOLE_THRESH_M:-0.10}")
cmd+=(--max-above-ground-m "${MAX_ABOVE_GROUND_M:-1.22}")
if [[ "${DISABLE_HOLES:-0}" == "1" ]]; then cmd+=(--disable-holes); fi

cmd+=(--plane-update-sec "${PLANE_UPDATE_SEC:-0.50}")
cmd+=(--plane-ransac-iters "${PLANE_RANSAC_ITERS:-120}")
cmd+=(--plane-min-normal-up "${PLANE_MIN_NORMAL_UP:-0.70}")
cmd+=(--plane-fit-min-range-m "${PLANE_FIT_MIN_RANGE_M:-0.25}")
cmd+=(--plane-fit-max-range-m "${PLANE_FIT_MAX_RANGE_M:-5.0}")
cmd+=(--plane-fit-max-abs-up-m "${PLANE_FIT_MAX_ABS_UP_M:-0.60}")

cmd+=(--map-width-m "${MAP_WIDTH_M:-20.0}")
cmd+=(--map-height-m "${MAP_HEIGHT_M:-20.0}")
cmd+=(--map-res-m "${MAP_RES_M:-0.05}")
cmd+=(--map-forward-min "${MAP_FORWARD_MIN:-0.0}")
cmd+=(--map-scale "${MAP_SCALE:-2}")
if [[ "${MAP_CENTER:-1}" == "1" ]]; then cmd+=(--map-center); fi

cmd+=(--map-save-path "${MAP_SAVE_PATH:-${SCRIPT_DIR}/lidar_map.npz}")
cmd+=(--map-save-every "${MAP_SAVE_EVERY:-5.0}")
if [[ "${MAP_LOAD:-0}" == "1" ]]; then cmd+=(--map-load); fi

cmd+=(--free-decay "${FREE_DECAY:-0.995}")
cmd+=(--occ-decay "${OCC_DECAY:-0.98}")
cmd+=(--hole-decay "${HOLE_DECAY:-0.98}")

cmd+=(--print-every-sec "${PRINT_EVERY_SEC:-1.0}")
if [[ "${NO_GUI:-0}" == "1" ]]; then cmd+=(--no-gui); fi

echo "Running: ${cmd[*]} $*"
exec "${cmd[@]}" "$@"
