#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fallback for debugging the lightweight legacy path only when explicitly requested.
if [[ "${REC_CAMERA_ONLY:-0}" == "1" ]]; then
  exec python3 "${SCRIPT_DIR}/zed_ground_wall.py" --manual-start --camera-only "$@"
fi

# Use the proven RunAuto launcher path, but force it into record-oriented startup:
# same main control stack, manual drive, and no mapping window/integration overhead.
MANUAL_START=1 \
NO_MAPPING_START=1 \
SKIP_DEPTH_PROCESSING=1 \
HUMAN_DETECT=0 \
LANDMARK_MEMORY=0 \
ROCK_MODEL="" \
MAP_PUBLISH_URL="" \
exec "${SCRIPT_DIR}/RunAuto.sh" "$@"
