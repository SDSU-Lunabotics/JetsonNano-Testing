#!/usr/bin/env bash
set -euo pipefail

echo "[SCRIPT] restart_jetson started at $(date)"

echo "JETSON_REBOOT_ALLOWED=${JETSON_REBOOT_ALLOWED:-0}"

if [ "${JETSON_REBOOT_ALLOWED:-0}" != "1" ]; then
  echo "Reboot disabled: set JETSON_REBOOT_ALLOWED=1 to enable."
  echo "[SCRIPT] restart_jetson finished"
  exit 0
fi

echo "Checking passwordless sudo access for reboot..."
if ! sudo -n true 2>/dev/null; then
  echo "Passwordless sudo required for reboot; configure sudoers for this script."
  exit 1
fi

echo "Scheduling Jetson reboot in 5 seconds..."
nohup bash -lc 'sleep 5 && sudo systemctl reboot' >/tmp/restart_jetson.log 2>&1 &

echo "Reboot command scheduled. Log: /tmp/restart_jetson.log"
echo "[SCRIPT] restart_jetson finished"