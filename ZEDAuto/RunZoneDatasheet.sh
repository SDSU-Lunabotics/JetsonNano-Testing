#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/zone_datasheet.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

if [[ "${ENABLE_ZONE_DATASHEET:-0}" != "1" ]]; then
  echo "Zone datasheet logger is disabled (ENABLE_ZONE_DATASHEET=0)."
  echo "Set ENABLE_ZONE_DATASHEET=1 in ZEDAuto/zone_datasheet.env to run it."
  exit 0
fi

cmd=(python3 "${SCRIPT_DIR}/zed_zone_datasheet.py")

if [[ "${TRACKING:-1}" == "1" ]]; then cmd+=(--tracking); fi
cmd+=(--out-csv "${OUT_CSV:-${SCRIPT_DIR}/zone_datasheet.csv}")
cmd+=(--stride "${STRIDE:-8}")
cmd+=(--sample-every-sec "${SAMPLE_EVERY_SEC:-1.0}")
cmd+=(--obstacle-samples "${OBSTACLE_SAMPLES:-5}")
cmd+=(--hole-samples "${HOLE_SAMPLES:-3}")
cmd+=(--ground-samples "${GROUND_SAMPLES:-0}")
cmd+=(--obstacle-thresh-m "${OBSTACLE_THRESH_M:-0.05}")
cmd+=(--hole-thresh-m "${HOLE_THRESH_M:-0.10}")
cmd+=(--max-above-ground-m "${MAX_ABOVE_GROUND_M:-1.22}")
cmd+=(--max-forward-m "${MAX_FORWARD_M:-6.0}")
cmd+=(--floor-update-sec "${FLOOR_UPDATE_SEC:-0.5}")
cmd+=(--floor-min-normal-y "${FLOOR_MIN_NORMAL_Y:-0.5}")
cmd+=(--map-width-m "${MAP_WIDTH_M:-20.0}")
cmd+=(--map-height-m "${MAP_HEIGHT_M:-20.0}")
cmd+=(--map-res-m "${MAP_RES_M:-0.05}")
cmd+=(--map-z-min "${MAP_Z_MIN:-0.0}")
cmd+=(--seed "${RNG_SEED:-42}")
if [[ "${MAP_CENTER:-1}" == "1" ]]; then cmd+=(--map-center); fi

echo "Running: ${cmd[*]} $*"
exec "${cmd[@]}" "$@"
