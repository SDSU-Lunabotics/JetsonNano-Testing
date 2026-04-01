#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/perception_lab.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

cmd=(python3 "${SCRIPT_DIR}/terrain_ground_wall_simple.py")

if [[ "${TRACKING:-1}" == "1" ]]; then cmd+=(--tracking); fi
cmd+=(--floor-update-sec "${FLOOR_UPDATE_SEC:-0.5}")
cmd+=(--floor-min-normal-y "${FLOOR_MIN_NORMAL_Y:-0.5}")
cmd+=(--stride "${STRIDE:-8}")
cmd+=(--obstacle-thresh-m "${OBSTACLE_THRESH_M:-0.05}")
cmd+=(--hole-thresh-m "${HOLE_THRESH_M:-0.10}")
cmd+=(--max-above-ground-m "${MAX_ABOVE_GROUND_M:-1.22}")
cmd+=(--max-forward-m "${MAX_FORWARD_M:-6.0}")
cmd+=(--map-width-m "${MAP_WIDTH_M:-20.0}")
cmd+=(--map-height-m "${MAP_HEIGHT_M:-20.0}")
cmd+=(--map-res-m "${MAP_RES_M:-0.05}")
cmd+=(--map-scale "${TERRAIN_MAP_SCALE:-3}")
cmd+=(--map-decay "${TERRAIN_MAP_DECAY:-0.995}")
cmd+=(--free-decay "${TERRAIN_FREE_DECAY:-1.0}")
cmd+=(--occ-decay "${TERRAIN_OCC_DECAY:-1.0}")
cmd+=(--hole-decay "${TERRAIN_HOLE_DECAY:-1.0}")

if [[ "${MAP_CENTER:-1}" == "1" ]]; then cmd+=(--map-center); fi
if [[ "${TERRAIN_SHOW_HOLES:-1}" == "1" ]]; then cmd+=(--show-holes); fi

echo "Running: ${cmd[*]} $*"
exec "${cmd[@]}" "$@"

