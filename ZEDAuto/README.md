# ZEDAuto

Ground/wall segmentation, occupancy mapping, and optional RoboRIO drive output for a ZED camera.

## Main Entry Points
- `zed_ground_wall.py`: Main application.
- `RunAuto.sh`: One-command launcher that loads defaults from `zed_ground_wall.env`.
- `zed_ground_wall.env`: Default configuration for mapping, drive, heatmap, and optional streaming.

## Quick Start
1. Open a terminal in the repo root:
   ```bash
   cd /path/to/JetsonNano-Testing
   ```
2. Launch with defaults:
   ```bash
   ./ZEDAuto/RunAuto.sh
   ```
3. If the executable bit is missing, use:
   ```bash
   bash ZEDAuto/RunAuto.sh
   ```

## One-Time Overrides
Pass any `zed_ground_wall.py` flags directly to the launcher:

```bash
./ZEDAuto/RunAuto.sh --map-scale 4
./ZEDAuto/RunAuto.sh --no-gui
./ZEDAuto/RunAuto.sh --drive-debug
./ZEDAuto/RunAuto.sh --stream-ip 192.168.1.100 --stream-port 5600 --stream-view both
```

Show all runtime options:

```bash
python3 ZEDAuto/zed_ground_wall.py --help
```

## Legacy Manual Run Command
For older setups that used the previous folder layout, this was the direct command:

```bash
python3 ros2_configs/zed/zed_ground_wall.py \
  --tracking --map-center --map-scale 3 \
  --free-decay 1.0 --occ-decay 0.98 --hole-decay 0.98 \
  --drive --roborio-ip 10.0.9.2 --drive-speed 0.7 --drive-heading-flip
```

If this repository uses the current layout, run the same settings with:

```bash
python3 ZEDAuto/zed_ground_wall.py \
  --tracking --map-center --map-scale 3 \
  --free-decay 1.0 --occ-decay 0.98 --hole-decay 0.98 \
  --drive --roborio-ip 10.0.9.2 --drive-speed 0.7 --drive-heading-flip
```

Settings used in this legacy command:
- `--tracking`: Enable ZED positional tracking.
- `--map-center`: Center occupancy map around start position.
- `--map-scale 3`: Display map at 3x scale.
- `--free-decay 1.0`: Keep free-space evidence from fading.
- `--occ-decay 0.98`: Slowly decay obstacle evidence.
- `--hole-decay 0.98`: Slowly decay hole evidence.
- `--drive`: Enable RoboRIO command output via NetworkTables.
- `--roborio-ip 10.0.9.2`: RoboRIO target IP.
- `--drive-speed 0.7`: Forward speed command magnitude.
- `--drive-heading-flip`: Apply 180-degree heading flip for drive alignment.

Additional settings that can be used (with defaults):
- `--map-width-m 20.0`: Top-down map width in meters (X axis).
- `--map-height-m 20.0`: Top-down map height in meters (Z axis).
- `--map-res-m 0.05`: Map resolution in meters per cell.
- `--map-z-min 0.0`: Minimum Z (forward) bound for map.
- `--map-save-path zed_map.npz`: Path to save persistent map data.
- `--map-save-every 5.0`: Seconds between map saves (0 to disable).
- `--map-load`: Load existing map on startup if available.
- `--map-decay 0.995`: Map decay factor (1.0 = no decay).
- `--free-decay-unconfirmed`: Decay for low-confidence free cells.
- `--free-decay-confirmed 1.0`: Decay for confirmed free cells.
- `--free-confirm-hits 8.0`: Hits needed to mark a free cell as confirmed.
- `--free-confirm-ratio 1.2`: Free/obstacle ratio required for confirmed free.
- `--free-downgrade-factor 0.6`: Multiply free confidence when obstacle/hole appears.
- `--map-camera-size 3`: Camera marker size in cells.
- `--heatmap`: Enable heatmap view for map confidence/risk.
- `--heatmap-mode risk`: Heatmap mode (choices: risk, obstacle, hole, free, evidence).
- `--heatmap-alpha 0.35`: Heatmap overlay alpha (0-1).
- `--heatmap-min-evidence 1.0`: Min evidence needed before a heatmap cell is shown.
- `--heatmap-window`: Show heatmap in a separate window instead of overlaying on map.
- `--obstacle-thresh-m 0.05`: Obstacle height above ground (m).
- `--hole-thresh-m 0.05`: Hole depth below ground (m).
- `--disable-holes`: Disable hole detection (testing).
- `--path-avoid-occ-min 3.0`: Min obstacle count for path blocking.
- `--path-avoid-occ-ratio 1.5`: Min occupied/free ratio for blocking.
- `--path-connectivity 8`: A* grid connectivity (choices: 4, 8).
- `--path-replan-sec 0.5`: How often to retry path planning.
- `--block-unknown`: Treat unknown (black) cells as blocked.
- `--unknown-min-evidence 1.0`: Evidence threshold to mark a cell as known.
- `--start-clear-radius-m 0.35`: Clear blocked cells near rover start/blind spot.
- `--rover-size-m 0.305`: Rover footprint size (m, square).
- `--spatial-mapping`: Enable ZED SDK spatial mapping.
- `--spatial-res medium`: Spatial map resolution (low|medium|high).
- `--spatial-range medium`: Spatial map range (short|medium|long).
- `--spatial-save-path`: Optional path to save spatial mesh (.obj).
- `--spatial-save-every 10.0`: Seconds between spatial map saves.
- `--spatial-viewer`: Show live Open3D mesh viewer.
- `--spatial-filter none`: Mesh filter (none|low|medium|high).
- `--drive-turn-k 0.8`: Turn gain for heading error.
- `--drive-rate-hz 10.0`: Drive command rate (Hz).
- `--drive-goal-tol-m 0.3`: Goal tolerance (m).
- `--drive-heading-tol-deg 10.0`: Heading tolerance (deg).
- `--drive-ready-pulse-sec 0.10`: How long CommandReady stays high per command pulse.
- `--nt-health-debug`: Print NetworkTables session health plus incoming robot-published `Jetson/*` drive keys.
- `--nt-health-period-sec 1.0`: Seconds between NT health debug prints.
- `--floor-update-sec 0.5`: Seconds between floor-plane updates.
- `--floor-min-normal-y 0.5`: Reject floor planes with |normal.y| below this.
- `--stream-ip`: UDP target IP for GStreamer stream.
- `--stream-port 5600`: UDP port for GStreamer stream.
- `--stream-fps 15.0`: Stream FPS.
- `--stream-bitrate-kbps 2500`: Stream bitrate in kbps.
- `--stream-view both`: Which view to stream (camera, map, both).
- `--no-gui`: Disable local OpenCV windows.
- `--rviz`: Launch rviz2 on startup.
- `--rviz-config`: Path to an RViz2 config file.
- `--ros2`: Publish a PointCloud2 topic over ROS2.
- `--frame zed_camera`: Frame ID for ROS2 point cloud.

## Default Config File
Edit `ZEDAuto/zed_ground_wall.env` to set team/default launch values:
- Mapping and decay behavior.
- Heatmap behavior.
- RoboRIO IP and drive settings.
- Optional streaming target.
- Optional TigerVNC auto-start.

Then run the same single command:

```bash
./ZEDAuto/RunAuto.sh
```

## Runtime Controls
- Left click on the occupancy map: Set goal cell.
- Right click on the occupancy map: Emergency stop.
- `m`: Toggle manual drive mode.
- `w`, `a`, `s`, `d`: Manual drive (hold-to-move behavior).
- `x`: Zero manual drive command.
- `space`: Emergency stop.
- `q`: Quit.

## Optional Stream Receiver Example
If streaming is enabled (`--stream-ip ...`), a receiver can use:

```bash
gst-launch-1.0 -v udpsrc port=5600 caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

## Dependencies
Required:
- ZED SDK Python API (`pyzed.sl`)
- `numpy`
- `opencv-python` / OpenCV Python bindings

Optional features:
- `pynetworktables` for `--drive`
- ROS 2 Python libs (`rclpy`, `sensor_msgs_py`) for `--ros2`
- `open3d` for `--spatial-viewer`
- GStreamer + OpenCV GStreamer backend for UDP stream output

## Troubleshooting
- `Failed to import pyzed.sl`: install ZED SDK and Python bindings on the target machine.
- `OpenCV not found`: install OpenCV Python bindings.
- `NetworkTables not available`: install `pynetworktables` or disable drive mode.
- For communication debugging, set `NT_HEALTH_DEBUG=1` in `zed_ground_wall.env` and run `./ZEDAuto/RunAuto.sh`.
  You should see logs like `NT health connected=True ... peers=[...] ... rx_fwd=... rx_turn=...`.
- Stream pipeline fails: verify GStreamer plugins/codecs are installed and available to OpenCV.
