#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/zed_ground_wall.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

# Resolve MAP_SAVE_PATH the same way RunAuto.sh does (cd to repo root first).
cd "${REPO_ROOT}"
MAP_PATH="${MAP_SAVE_PATH:-${SCRIPT_DIR}/zed_map.npz}"

if [[ -f "${MAP_PATH}" ]]; then
  rm -f "${MAP_PATH}"
  echo "Deleted saved map: ${MAP_PATH}"
else
  echo "No saved map found at: ${MAP_PATH}"
fi

echo "Next run will start fresh unless MAP_LOAD points to another file."
