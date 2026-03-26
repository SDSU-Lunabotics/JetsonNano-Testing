#!/usr/bin/env bash
set -e

echo "[SCRIPT] start_lidar started at $(date)"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIDAR_DIR="${ROOT_DIR}/lidar"
DEFAULT_VIS_COMMAND="cd '${LIDAR_DIR}' && python3 lidar_visualization.py"
DEFAULT_BRIDGE_COMMAND="cd '${LIDAR_DIR}' && sudo ./lidar_bridge"

VIS_COMMAND="${LIDAR_VISUALIZATION_COMMAND:-${LIDAR_APF_COMMAND:-${DEFAULT_VIS_COMMAND}}}"
BRIDGE_COMMAND="${LIDAR_BRIDGE_COMMAND:-${DEFAULT_BRIDGE_COMMAND}}"

nohup bash -lc "${VIS_COMMAND}" >/tmp/jetson_lidar_visualization.log 2>&1 &
echo "Started visualization/APF process with PID $!"
sleep 2
nohup bash -lc "${BRIDGE_COMMAND}" >/tmp/jetson_lidar_bridge.log 2>&1 &
echo "Started LiDAR bridge process with PID $!"

echo "[SCRIPT] start_lidar finished"
