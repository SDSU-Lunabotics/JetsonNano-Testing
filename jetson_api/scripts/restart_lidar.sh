#!/usr/bin/env bash
set -e

echo "[SCRIPT] restart_lidar started at $(date)"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIDAR_DIR="${ROOT_DIR}/lidar"
DEFAULT_VIS_COMMAND="cd '${LIDAR_DIR}' && python3 lidar_visualization.py"
DEFAULT_BRIDGE_COMMAND="cd '${LIDAR_DIR}' && sudo ./lidar_bridge"

VIS_COMMAND="${LIDAR_VISUALIZATION_COMMAND:-${LIDAR_APF_COMMAND:-${DEFAULT_VIS_COMMAND}}}"
BRIDGE_COMMAND="${LIDAR_BRIDGE_COMMAND:-${DEFAULT_BRIDGE_COMMAND}}"
CMD_HOST="${LIDAR_TCP_HOST:-127.0.0.1}"
CMD_PORT="${LIDAR_COMMAND_PORT:-9877}"

if [ -n "${LIDAR_SYSTEMD_SERVICE:-}" ]; then
  systemctl restart "${LIDAR_SYSTEMD_SERVICE}"
  echo "Restarted systemd service ${LIDAR_SYSTEMD_SERVICE}"
elif [ -n "${LIDAR_RESTART_COMMAND:-}" ]; then
  bash -lc "${LIDAR_RESTART_COMMAND}"
  echo "Ran LIDAR_RESTART_COMMAND"
else
  if python3 - <<PY
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect(("${CMD_HOST}", int("${CMD_PORT}")))
    sock.sendall(b"RESTART\n")
    response = sock.recv(256).decode("utf-8", errors="replace").strip()
    print(response)
    raise SystemExit(0 if response.startswith("OK") else 1)
except OSError:
    raise SystemExit(2)
finally:
    sock.close()
PY
  then
    echo "Restarted LiDAR through bridge command socket"
    echo "[SCRIPT] restart_lidar finished"
    exit 0
  fi

  pkill -f "lidar_visualization.py" || true
  pkill -f "lidar_apf.py" || true
  pkill -f "lidar_bridge" || true

  nohup bash -lc "${VIS_COMMAND}" >/tmp/jetson_lidar_visualization.log 2>&1 &
  echo "Started visualization/APF process with PID $!"
  sleep 2
  nohup bash -lc "${BRIDGE_COMMAND}" >/tmp/jetson_lidar_bridge.log 2>&1 &
  echo "Started LiDAR bridge process with PID $!"
fi

echo "[SCRIPT] restart_lidar finished"
