#!/usr/bin/env bash
set -e

echo "[SCRIPT] start_ros started at $(date)"

if [ -n "${ROS_SETUP_SCRIPT:-}" ] && [ -f "${ROS_SETUP_SCRIPT}" ]; then
  # shellcheck disable=SC1090
  source "${ROS_SETUP_SCRIPT}"
elif [ -f "/opt/ros/humble/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "/opt/ros/humble/setup.bash"
fi

if [ -n "${JETSON_WS_SETUP_SCRIPT:-}" ] && [ -f "${JETSON_WS_SETUP_SCRIPT}" ]; then
  # shellcheck disable=SC1090
  source "${JETSON_WS_SETUP_SCRIPT}"
fi

if [ -z "${ROS_START_COMMAND:-}" ]; then
  echo "ROS_START_COMMAND is not set. Example:"
  echo "  export ROS_START_COMMAND='ros2 launch sllidar_ros2 sllidar_a1_launch.py'"
  exit 1
fi

nohup bash -lc "${ROS_START_COMMAND}" >/tmp/jetson_ros_start.log 2>&1 &
echo "Started ROS command in background with PID $!"

echo "[SCRIPT] start_ros finished"
