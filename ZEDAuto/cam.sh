#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/zed_ground_wall.env"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

cmd=(python3 "${SCRIPT_DIR}/zed_ground_wall.py" --camera-only --manual-start)

if [[ "${DRIVE:-1}" == "1" ]]; then
  cmd+=(--drive)
  cmd+=(--roborio-ip "${ROBORIO_IP:-10.0.9.2}")
  cmd+=(--drive-speed "${DRIVE_SPEED:-1.0}")
  cmd+=(--main-rover-mode)
  if [[ "${HARD_DRIVE_FLIP:-1}" == "1" ]]; then cmd+=(--hard-drive-flip); fi
  if [[ "${DRIVE_HEADING_FLIP:-1}" == "1" ]]; then cmd+=(--drive-heading-flip); fi
  if [[ "${DS_JOYSTICK:-1}" == "1" ]]; then cmd+=(--ds-joystick); fi
  cmd+=(--ds-joystick-fwd-key "${DS_JOYSTICK_FWD_KEY:-DS/JoystickFwd}")
  cmd+=(--ds-joystick-turn-key "${DS_JOYSTICK_TURN_KEY:-DS/JoystickTurn}")
  cmd+=(--ds-joystick-scale "${DS_JOYSTICK_SCALE:-0.5}")
fi

if [[ "${NO_GUI:-0}" == "1" ]]; then cmd+=(--no-gui); fi

echo "Running camera-only mode: ${cmd[*]}"
exec "${cmd[@]}"
