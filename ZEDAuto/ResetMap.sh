#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/zed_ground_wall.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

MAP_PATH="${MAP_SAVE_PATH:-${SCRIPT_DIR}/zed_map.npz}"

if [[ -f "${MAP_PATH}" ]]; then
  rm -f "${MAP_PATH}"
  echo "Deleted saved map: ${MAP_PATH}"
else
  echo "No saved map found at: ${MAP_PATH}"
fi

echo "Next run will start fresh unless MAP_LOAD points to another file."
