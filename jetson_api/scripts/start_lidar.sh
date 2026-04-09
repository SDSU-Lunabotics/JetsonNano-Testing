#!/usr/bin/env bash
set -e

echo "[SCRIPT] start_lidar started at $(date)"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIDAR_DIR="${ROOT_DIR}/lidar"
DEFAULT_BRIDGE_COMMAND="cd '${LIDAR_DIR}' && sudo ./lidar_bridge"
BRIDGE_BIN="${LIDAR_DIR}/lidar_bridge"

VIS_COMMAND="${LIDAR_VISUALIZATION_COMMAND:-${LIDAR_APF_COMMAND:-}}"
BRIDGE_COMMAND="${LIDAR_BRIDGE_COMMAND:-${DEFAULT_BRIDGE_COMMAND}}"

if [ ! -x "${BRIDGE_BIN}" ]; then
  echo "LiDAR bridge executable not found or not executable: ${BRIDGE_BIN}" >&2
  exit 1
fi

if [[ "${BRIDGE_COMMAND}" == *"sudo "* ]]; then
  if ! sudo -n true 2>/dev/null; then
    echo "sudo for lidar_bridge requires a password; configure passwordless sudo or update LIDAR_BRIDGE_COMMAND" >&2
    exit 1
  fi
fi

if [ -n "${VIS_COMMAND}" ]; then
  nohup bash -lc "${VIS_COMMAND}" >/tmp/jetson_lidar_visualization.log 2>&1 &
  echo "Started visualization/APF process with PID $!"
  sleep 2
else
  echo "Skipping visualization/APF startup; jetson_api will receive LiDAR points on TCP ${LIDAR_TCP_HOST:-127.0.0.1}:${LIDAR_DATA_PORT:-9876}"
fi

nohup bash -lc "${BRIDGE_COMMAND}" >/tmp/jetson_lidar_bridge.log 2>&1 &
BRIDGE_PID=$!
sleep 1
if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
  echo "LiDAR bridge process exited immediately. See /tmp/jetson_lidar_bridge.log" >&2
  exit 1
fi
echo "Started LiDAR bridge process with PID ${BRIDGE_PID}"

echo "[SCRIPT] start_lidar finished"
