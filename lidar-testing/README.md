# LiDAR Testing (Unitree L2)

Standalone ground/wall/hole detection + 2D occupancy map for Unitree L2.

This folder is independent from `ZEDAuto` and supports both:
- direct Unitree SDK input (default, no ROS2)
- ROS2 `PointCloud2` input (optional fallback)

## Files

- `lidar_ground_wall.py`: Main LiDAR mapping app.
- `lidar_map_utils.py`: Occupancy map logic (free/obstacle/hole evidence).
- `unitree_sdk_bridge.py`: SDK-to-JSON bridge used for direct mode.
- `RunLidarLab.sh`: Launcher that reads settings from `lidar_lab.env`.
- `lidar_lab.env`: Defaults for source mode, thresholds, and map tuning.

## Python Requirements

- `numpy`
- `opencv-python` (or `python3-opencv` on Jetson/Ubuntu)

## Quick Start (Direct SDK, No ROS2)

1. Make sure your Unitree Python SDK wrapper is installed on the Jetson.
2. Edit `lidar-testing/lidar_lab.env`:
   - `INPUT_MODE="sdk"`
   - optionally set `SDK_CMD` with your SDK module/class.
3. Run:

```bash
chmod +x ./lidar-testing/RunLidarLab.sh
./lidar-testing/RunLidarLab.sh
```

If your wrapper needs explicit module/class:

```bash
SDK_CMD="python3 ./lidar-testing/unitree_sdk_bridge.py --module unitree_lidar_sdk2 --class-name UnitreeLidar" \
./lidar-testing/RunLidarLab.sh
```

## Input Modes

- `INPUT_MODE="sdk"`: direct Unitree SDK bridge (recommended for your setup).
- `INPUT_MODE="ros2"`: consume ROS2 topic in `LIDAR_TOPIC`.
- `INPUT_MODE="auto"`: try SDK first, then ROS2 fallback.

## Connection Type (Ethernet vs USB/UART)

In `INPUT_MODE="sdk"`, this app reads LiDAR points through whatever transport your SDK command uses.

- Ethernet mode: pass `--ip` and `--port` to `unitree_sdk_bridge.py`.
- USB/UART mode: pass `--device` (example: `/dev/ttyUSB0`).

Examples:

```bash
# Ethernet
SDK_CMD="python3 ./lidar-testing/unitree_sdk_bridge.py --ip 192.168.123.10 --port 6100"
```

```bash
# USB / UART
SDK_CMD="python3 ./lidar-testing/unitree_sdk_bridge.py --device /dev/ttyUSB0"
```

You can place `SDK_CMD=...` directly in `lidar-testing/lidar_lab.env`.

## What You Get

- Ground detection (green)
- Wall/obstacle detection (red)
- Hole detection (blue, optional)
- Live 2D top-down map window
- Rover marker and forward arrow
- Console source/health logs

## Controls

- `q` or `Esc`: Quit
- `s`: Save map immediately

## Important Settings (`lidar_lab.env`)

- `INPUT_MODE`: `sdk`, `ros2`, or `auto`.
- `SDK_CMD`: command for direct SDK bridge process.
- `LIDAR_TOPIC`: ROS2 topic (used only in ros2/auto fallback).
- `UP_AXIS`, `FORWARD_AXIS`, `LATERAL_AXIS`: cloud axis mapping.
- `GROUND_THRESH_M`, `OBSTACLE_THRESH_M`, `HOLE_THRESH_M`: classification tolerance.
- `MAP_CENTER`: `1` keeps rover near middle of map.
- `MAP_SCALE`: display zoom only (does not change map accuracy).

## If SDK Bridge Does Not Connect

`unitree_sdk_bridge.py` tries common Unitree module/class/read-method names automatically.

If your SDK API is different, edit only this file:
- module import name
- class/factory constructor
- read method that returns point cloud

Output contract must stay one JSON line per frame:

```json
{"stamp": 1710000000.123, "frame_id": "unitree_l2", "points": [[x,y,z], [x,y,z]]}
```

## Optional ROS2 Path

Use only if you want ROS2 transport:

```bash
INPUT_MODE="ros2" LIDAR_TOPIC="/utlidar/cloud" ./lidar-testing/RunLidarLab.sh
```

Quick ROS2 check:

```bash
ros2 topic list
ros2 topic hz /utlidar/cloud
ros2 topic echo /utlidar/cloud --once
```

## Run Without GUI

```bash
./lidar-testing/RunLidarLab.sh --no-gui
```

This keeps logging and map accumulation active but disables OpenCV windows.
