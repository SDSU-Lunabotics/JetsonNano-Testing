#!/usr/bin/env bash
set -euo pipefail

echo "[SCRIPT] restart_jetson started at $(date)"
echo "[SCRIPT] syncing filesystem before reboot"
sync
echo "[SCRIPT] rebooting Jetson now"
sudo reboot
