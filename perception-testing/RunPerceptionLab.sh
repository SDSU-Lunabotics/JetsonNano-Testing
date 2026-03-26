#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/perception_lab.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

cmd=(python3 "${SCRIPT_DIR}/live_perception_lab.py")

if [[ "${TRACKING:-1}" == "1" ]]; then cmd+=(--tracking); fi
cmd+=(--floor-update-sec "${FLOOR_UPDATE_SEC:-0.5}")
cmd+=(--floor-min-normal-y "${FLOOR_MIN_NORMAL_Y:-0.5}")
cmd+=(--stride "${STRIDE:-8}")
cmd+=(--obstacle-thresh-m "${OBSTACLE_THRESH_M:-0.05}")
cmd+=(--hole-thresh-m "${HOLE_THRESH_M:-0.10}")
cmd+=(--max-above-ground-m "${MAX_ABOVE_GROUND_M:-1.22}")
cmd+=(--max-forward-m "${MAX_FORWARD_M:-6.0}")
cmd+=(--min-box-area-px "${MIN_BOX_AREA_PX:-1200}")
cmd+=(--ai-conf "${AI_CONF:-0.40}")
cmd+=(--ai-iou "${AI_IOU:-0.45}")
cmd+=(--ai-every "${AI_EVERY:-2}")
cmd+=(--ai-imgsz "${AI_IMGSZ:-640}")
cmd+=(--classes "${CLASSES:-rock,wall,person,cable,cone,other}")
cmd+=(--dataset-dir "${DATASET_DIR:-${SCRIPT_DIR}/dataset}")
cmd+=(--map-width-m "${MAP_WIDTH_M:-20.0}")
cmd+=(--map-height-m "${MAP_HEIGHT_M:-20.0}")
cmd+=(--map-res-m "${MAP_RES_M:-0.05}")
cmd+=(--semantic-point-stride "${SEMANTIC_POINT_STRIDE:-4}")
cmd+=(--semantic-decay "${SEMANTIC_DECAY:-1.0}")
cmd+=(--ground-band-m "${GROUND_BAND_M:-0.10}")

if [[ -n "${AI_DEVICE:-}" ]]; then cmd+=(--ai-device "${AI_DEVICE}"); fi
if [[ -n "${AI_MODEL_PATH:-}" ]]; then cmd+=(--ai-model "${AI_MODEL_PATH}"); fi
if [[ -n "${AI_LABELS_PATH:-}" ]]; then cmd+=(--ai-labels "${AI_LABELS_PATH}"); fi
if [[ "${ANNOTATION_MODE:-1}" == "1" ]]; then cmd+=(--annotation-mode); fi
if [[ "${SEMANTIC_MAP:-1}" == "1" ]]; then cmd+=(--semantic-map); fi
if [[ "${MAP_CENTER:-1}" == "1" ]]; then cmd+=(--map-center); fi

echo "Running: ${cmd[*]} $*"
exec "${cmd[@]}" "$@"
