#!/usr/bin/env python3
"""
ZED 2i ground + wall segmentation (SDK Python).
This script classifies ground vs. non-ground points using a fitted plane.
It is safe to run without the camera connected (it will fail to open and exit).
"""

import sys
import time
import argparse
import os
import math
import json
import numpy as np

try:
    import pyzed.sl as sl
except Exception as exc:
    print("Failed to import pyzed.sl. Is the ZED SDK Python API installed?")
    print(f"Error: {exc}")
    sys.exit(1)

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

STATUS_PANEL_W = int(os.getenv("ZED_STATUS_PANEL_W", "820"))
STATUS_PANEL_H = int(os.getenv("ZED_STATUS_PANEL_H", "980"))
DEFAULT_CAMERA_MAP_ANGLE_DEG = 180.0
DEFAULT_CAMERA_DEPOSIT_ANGLE_DEG = 0.0

LEFT_KEYS = {81, 2424832, 65361, 63234}
UP_KEYS = {82, 2490368, 65362, 63232}
RIGHT_KEYS = {83, 2555904, 65363, 63235}
DOWN_KEYS = {84, 2621440, 65364, 63233}
PAGEUP_KEYS = {2162688, 65365, 63276}
PAGEDOWN_KEYS = {2228224, 65366, 63277}
HOME_KEYS = {2359296, 65360, 63273}
END_KEYS = {2293760, 65367, 63275}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import rviz_utils
import ros2_utils
import segmentation
import map_utils
import heatmap_utils
import zed_utils
import viewer_utils
import stream_utils
import auto_mining
import camera_status_client
import camera_publish_client
import map_publish_client
import calibration_profiles

try:
    from networktables import NetworkTables
    HAS_NT = True
except Exception:
    HAS_NT = False


def main():
    parser = argparse.ArgumentParser(description="ZED 2i ground + wall segmentation")
    parser.add_argument("--rviz", action="store_true", help="Launch rviz2 on startup")
    parser.add_argument("--rviz-config", default=None, help="Path to an RViz2 config file")
    parser.add_argument("--ros2", action="store_true", help="Publish a PointCloud2 topic over ROS2")
    parser.add_argument("--frame", default="zed_camera", help="Frame ID for ROS2 point cloud")
    parser.add_argument("--tracking", action="store_true", default=True, help="Enable ZED positional tracking (default: on)")
    parser.add_argument("--no-tracking", action="store_false", dest="tracking", help="Disable ZED positional tracking")
    parser.add_argument("--area-memory", action="store_true", help="Enable ZED area-memory relocalization")
    parser.add_argument("--area-load-path", default=None, help="Path to load ZED area memory (.area)")
    parser.add_argument("--area-save-path", default=None, help="Path to save ZED area memory (.area)")
    parser.add_argument("--area-save-every", type=float, default=30.0, help="Seconds between area-memory saves")
    parser.add_argument(
        "--tracking-max-pose-jump-m",
        type=float,
        default=0.80,
        help="Reject tracking pose jumps larger than this between frames. Set <=0 to disable.",
    )
    parser.add_argument(
        "--tracking-max-heading-jump-deg",
        type=float,
        default=55.0,
        help="Reject tracking heading jumps larger than this between accepted poses. Set <=0 to disable.",
    )
    parser.add_argument(
        "--tracking-recover-stable-frames",
        type=int,
        default=6,
        help="Require this many consecutive stable tracking frames before relocking after tracking loss.",
    )
    parser.add_argument(
        "--imu-heading-fallback",
        action="store_true",
        dest="imu_heading_fallback",
        help="Use ZED IMU orientation deltas as a display/jump-check fallback when tracking is weak.",
    )
    parser.add_argument(
        "--no-imu-heading-fallback",
        action="store_false",
        dest="imu_heading_fallback",
        help="Disable ZED IMU heading fallback.",
    )
    parser.set_defaults(imu_heading_fallback=True)
    parser.add_argument(
        "--imu-heading-max-mismatch-deg",
        type=float,
        default=45.0,
        help="Reject a tracking pose if its heading differs from the IMU delta estimate by more than this many degrees.",
    )
    parser.add_argument(
        "--navx-heading-aid",
        action="store_true",
        dest="navx_heading_aid",
        help="Use RoboRIO NavX yaw as a second heading reference for jump rejection and fallback display.",
    )
    parser.add_argument(
        "--no-navx-heading-aid",
        action="store_false",
        dest="navx_heading_aid",
        help="Disable RoboRIO NavX heading aid.",
    )
    parser.set_defaults(navx_heading_aid=True)
    parser.add_argument(
        "--navx-heading-max-mismatch-deg",
        type=float,
        default=35.0,
        help="Reject a tracking pose if its rover heading differs from the calibrated NavX estimate by more than this many degrees.",
    )
    parser.add_argument(
        "--localize-turn-speed",
        type=float,
        default=0.25,
        help="Turn command used during Localize Scan (0-1).",
    )
    parser.add_argument(
        "--localize-scan-sec",
        type=float,
        default=8.0,
        help="Seconds to rotate for a manual Localize Scan when tracking is already OK.",
    )
    parser.add_argument(
        "--localize-max-sec",
        type=float,
        default=20.0,
        help="Maximum seconds to keep Localize Scan active while tracking is lost.",
    )
    parser.add_argument("--map-width-m", type=float, default=20.0, help="Top-down map width in meters (X axis)")
    parser.add_argument("--map-height-m", type=float, default=20.0, help="Top-down map height in meters (Z axis)")
    parser.add_argument("--map-res-m", type=float, default=0.05, help="Map resolution in meters per cell")
    parser.add_argument("--map-z-min", type=float, default=0.0, help="Minimum Z (forward) bound for map")
    parser.add_argument("--map-scale", type=int, default=3, help="Upscale factor for map display window")
    parser.add_argument("--map-center", action="store_true", help="Center map on Z=0 (start camera in middle)")
    parser.add_argument(
        "--map-follow-rover",
        action="store_true",
        help="Keep rover centered in occupancy map view (toggle with 'c')",
    )
    parser.add_argument("--map-save-path", default="zed_map.npz", help="Path to save persistent map data")
    parser.add_argument("--map-save-every", type=float, default=5.0, help="Seconds between map saves (0 to disable)")
    parser.add_argument("--map-load", action="store_true", help="Load existing map on startup if available")
    parser.add_argument(
        "--recovery-checkpoint-path",
        default=os.path.join(SCRIPT_DIR, "zed_recovery_checkpoint.json"),
        help="Path to lightweight crash-recovery checkpoint JSON",
    )
    parser.add_argument(
        "--recovery-save-every",
        type=float,
        default=1.0,
        help="Seconds between pose/heading recovery checkpoint saves (0 to disable)",
    )
    parser.add_argument(
        "--no-recovery-load",
        action="store_false",
        dest="recovery_load",
        help="Disable loading the latest crash-recovery checkpoint on startup",
    )
    parser.add_argument(
        "--no-recovery-nt-mirror",
        action="store_false",
        dest="recovery_nt_mirror",
        help="Disable mirroring the latest lightweight recovery state to RoboRIO NetworkTables",
    )
    parser.set_defaults(recovery_load=True, recovery_nt_mirror=True)
    parser.add_argument("--map-decay", type=float, default=0.995, help="Map decay factor (1.0 = no decay)")
    parser.add_argument("--free-decay", type=float, default=None, help="Free-space decay (defaults to --map-decay)")
    parser.add_argument("--free-decay-unconfirmed", type=float, default=None, help="Decay for low-confidence free cells")
    parser.add_argument("--free-decay-confirmed", type=float, default=1.0, help="Decay for confirmed free cells")
    parser.add_argument("--free-confirm-hits", type=float, default=8.0, help="Hits needed to mark a free cell as confirmed")
    parser.add_argument("--free-confirm-ratio", type=float, default=1.2, help="Free/obstacle ratio required for confirmed free")
    parser.add_argument("--free-downgrade-factor", type=float, default=0.6, help="Multiply free confidence when obstacle/hole appears")
    parser.add_argument("--occ-decay", type=float, default=None, help="Obstacle decay (defaults to --map-decay)")
    parser.add_argument("--hole-decay", type=float, default=None, help="Hole decay (defaults to --map-decay)")
    parser.add_argument("--map-camera-size", type=int, default=3, help="Camera marker size in cells")
    parser.add_argument("--heatmap", action="store_true", help="Enable heatmap view for map confidence/risk")
    parser.add_argument(
        "--heatmap-mode",
        default="risk",
        choices=["risk", "obstacle", "hole", "free", "evidence"],
        help="Heatmap mode",
    )
    parser.add_argument("--heatmap-alpha", type=float, default=0.35, help="Heatmap overlay alpha (0-1)")
    parser.add_argument(
        "--heatmap-min-evidence",
        type=float,
        default=1.0,
        help="Min evidence needed before a heatmap cell is shown",
    )
    parser.add_argument(
        "--heatmap-window",
        action="store_true",
        help="Show heatmap in a separate window instead of overlaying on map",
    )
    parser.add_argument(
        "--complex",
        action="store_true",
        help="Use complex mapping (EMA plane smoothing, tilt/jump rejection, tracking-gated map). Default is simple mode.",
    )
    parser.add_argument("--obstacle-thresh-m", type=float, default=0.05, help="Obstacle height above ground (m)")
    parser.add_argument("--hole-thresh-m", type=float, default=0.05, help="Hole depth below ground (m)")
    parser.add_argument(
        "--max-above-ground-m",
        type=float,
        default=1.22,
        help="Ignore points above this height over floor plane (m). Set <=0 to disable.",
    )
    parser.add_argument("--disable-holes", action="store_true", help="Disable hole detection (testing)")
    parser.add_argument("--path-avoid-occ-min", type=float, default=3.0, help="Min obstacle count for path blocking")
    parser.add_argument("--path-avoid-occ-ratio", type=float, default=1.5, help="Min occupied/free ratio for blocking")
    parser.add_argument(
        "--path-avoid-occ-advantage",
        type=float,
        default=2.0,
        help="Min (occupied - free) evidence margin for path blocking",
    )
    parser.add_argument("--path-connectivity", type=int, default=8, choices=[4, 8], help="A* grid connectivity")
    parser.add_argument("--path-replan-sec", type=float, default=0.5, help="How often to retry path planning")
    parser.add_argument(
        "--path-soft-clearance-cells",
        type=int,
        default=8,
        help="Extra soft-cost clearance cells around inflated obstacles so A* prefers wider lanes",
    )
    parser.add_argument(
        "--no-path-relax-on-fail",
        action="store_false",
        dest="path_relax_on_fail",
        help="Disable staged relaxed/noise-filtered path retries when normal A* fails",
    )
    parser.set_defaults(path_relax_on_fail=True)
    parser.add_argument(
        "--path-max-search-sec",
        type=float,
        default=0.075,
        help="Maximum seconds allowed for a single A* path search",
    )
    parser.add_argument(
        "--allow-direct-no-path",
        action="store_true",
        help="Allow direct goal driving when A* has no path. Default is to stop and retry planning.",
    )
    parser.add_argument("--block-unknown", action="store_true", help="Treat unknown (black) cells as blocked")
    parser.add_argument("--unknown-min-evidence", type=float, default=1.0, help="Evidence threshold to mark a cell as known")
    parser.add_argument("--start-clear-radius-m", type=float, default=0.35, help="Clear blocked cells near rover start/blind spot")
    parser.add_argument("--rover-size-m", type=float, default=0.305, help="Rover footprint size (m, square)")
    parser.add_argument(
        "--camera-mount",
        choices=["front", "rear", "custom"],
        default="front",
        help="Camera mounting preset. rear assumes the ZED is on the rear looking backward.",
    )
    parser.add_argument(
        "--camera-mount-yaw-deg",
        type=float,
        default=None,
        help="Camera yaw relative to rover forward. 0=looks forward, 180=looks backward.",
    )
    parser.add_argument(
        "--camera-forward-offset-m",
        type=float,
        default=None,
        help="Camera position relative to rover center; positive is toward the dig/front side.",
    )
    parser.add_argument(
        "--camera-right-offset-m",
        type=float,
        default=0.0,
        help="Camera position relative to rover center; positive is rover right.",
    )
    parser.add_argument(
        "--camera-servo-track",
        action="store_true",
        help="Read camera servo angle from RoboRIO and use it as the live camera yaw relative to the rover.",
    )
    parser.add_argument(
        "--camera-map-angle-deg",
        type=float,
        default=180.0,
        help="Servo angle used for outward map/navigation view.",
    )
    parser.add_argument(
        "--camera-deposit-angle-deg",
        type=float,
        default=0.0,
        help="Servo angle used to look into the deposition bin/material.",
    )
    parser.add_argument(
        "--camera-servo-map-tol-deg",
        type=float,
        default=8.0,
        help="Allowed error from map-view servo angle before map integration pauses.",
    )
    parser.add_argument(
        "--camera-servo-invert",
        action="store_true",
        help="Invert RoboRIO servo angles so Jetson logical 0/180 map to reversed physical servo endpoints.",
    )
    parser.add_argument(
        "--display-heading-flip",
        action="store_true",
        help="Flip the yellow rover/map arrow by 180 degrees without changing drive commands.",
    )
    parser.add_argument("--spatial-mapping", action="store_true", help="Enable ZED SDK spatial mapping")
    parser.add_argument("--spatial-res", default="medium", help="Spatial map resolution: low|medium|high")
    parser.add_argument("--spatial-range", default="medium", help="Spatial map range: short|medium|long")
    parser.add_argument("--spatial-save-path", default=None, help="Optional path to save spatial mesh (.obj)")
    parser.add_argument("--spatial-save-every", type=float, default=10.0, help="Seconds between spatial map saves")
    parser.add_argument("--spatial-viewer", action="store_true", help="Show live Open3D mesh viewer")
    parser.add_argument("--spatial-filter", default="none", help="Mesh filter: none|low|medium|high")
    parser.add_argument("--drive", action="store_true", help="Enable RoboRIO driving commands")
    parser.add_argument("--roborio-ip", default="10.0.9.2", help="RoboRIO IP for NetworkTables")
    parser.add_argument("--drive-speed", type=float, default=1.0, help="Forward speed command (0-1)")
    parser.add_argument(
        "--drive-forward-slew-per-sec",
        type=float,
        default=1.4,
        help="Max change rate of forward command during auto path driving (command units per second)",
    )
    parser.add_argument("--drive-turn-k", type=float, default=1.15, help="Turn gain for heading error")
    parser.add_argument(
        "--drive-max-turn-cmd",
        type=float,
        default=1.00,
        help="Maximum absolute turn command while auto-driving (0-1)",
    )
    parser.add_argument(
        "--drive-slow-turn-deg",
        type=float,
        default=20.0,
        help="Begin reducing forward speed above this heading error (deg)",
    )
    parser.add_argument(
        "--drive-stop-turn-deg",
        type=float,
        default=40.0,
        help="Stop forward motion above this heading error (deg)",
    )
    parser.add_argument(
        "--drive-min-turn-forward-scale",
        type=float,
        default=0.20,
        help="Minimum forward-speed scale to keep while auto-turning toward the path (0-1).",
    )
    parser.add_argument(
        "--drive-turn-slew-per-sec",
        type=float,
        default=2.5,
        help="Max change rate of turn command (command units per second)",
    )
    parser.add_argument(
        "--drive-direct-lookahead-cells",
        type=int,
        default=24,
        help="When Direct Nav is enabled, target the farthest visible path cell up to this many cells ahead.",
    )
    parser.add_argument("--drive-rate-hz", type=float, default=10.0, help="Drive command rate (Hz)")
    parser.add_argument(
        "--driver-priority-mode",
        action="store_true",
        dest="driver_priority_mode",
        help="When DS/Xbox input is active, pause Jetson auto drive and reduce camera-processing load.",
    )
    parser.add_argument(
        "--no-driver-priority-mode",
        action="store_false",
        dest="driver_priority_mode",
        help="Disable DS/Xbox priority mode.",
    )
    parser.set_defaults(driver_priority_mode=True)
    parser.add_argument(
        "--driver-priority-threshold",
        type=float,
        default=0.12,
        help="Absolute DS joystick input needed before driver-priority mode engages.",
    )
    parser.add_argument(
        "--driver-priority-sample-stride",
        type=int,
        default=12,
        help="Minimum point-cloud stride while driver-priority mode is active.",
    )
    parser.add_argument(
        "--backup-close-dist-m",
        type=float,
        default=0.45,
        help="If obstacle points are this close in front of camera, command reverse (m). Set <=0 to disable.",
    )
    parser.add_argument(
        "--backup-lane-half-width-m",
        type=float,
        default=0.35,
        help="Half-width of forward safety lane for close-obstacle backup detection (m).",
    )
    parser.add_argument(
        "--backup-min-obstacle-points",
        type=int,
        default=30,
        help="Minimum close obstacle points in safety lane before backup triggers.",
    )
    parser.add_argument(
        "--backup-critical-dist-m",
        type=float,
        default=0.30,
        help="Critical forward distance (m). If enough obstacle points are inside this, backup triggers immediately.",
    )
    parser.add_argument(
        "--backup-critical-min-points",
        type=int,
        default=6,
        help="Minimum critical-distance obstacle points required for immediate backup trigger.",
    )
    parser.add_argument(
        "--backup-speed",
        type=float,
        default=0.35,
        help="Reverse command magnitude when close-obstacle backup triggers (0-1).",
    )
    parser.add_argument(
        "--backup-hold-sec",
        type=float,
        default=0.40,
        help="How long to continue backup once triggered (seconds).",
    )
    parser.add_argument("--drive-goal-tol-m", type=float, default=0.45, help="Goal tolerance (m)")
    parser.add_argument("--drive-heading-tol-deg", type=float, default=16.0, help="Heading tolerance (deg)")
    parser.add_argument("--drive-heading-flip", action="store_true", help="Flip heading by 180 degrees")
    parser.add_argument(
        "--hard-drive-flip",
        action="store_true",
        help="Invert the actual drivetrain forward/turn commands sent to the RoboRIO.",
    )
    parser.add_argument(
        "--steering-flip",
        action="store_true",
        help="Invert only the drivetrain turn command sent to the RoboRIO.",
    )
    parser.add_argument(
        "--main-rover-mode",
        action="store_true",
        dest="main_rover_mode",
        help="Enable main-rover controls on the RoboRIO (default: on)",
    )
    parser.add_argument(
        "--no-main-rover-mode",
        action="store_false",
        dest="main_rover_mode",
        help="Disable main-rover controls on the RoboRIO",
    )
    parser.set_defaults(main_rover_mode=True)
    parser.add_argument("--main-rover-debug", action="store_true", help="Enable Drive/MainRoverDebugMode on the RoboRIO")
    parser.add_argument(
        "--drive-ready-pulse-sec",
        type=float,
        default=0.10,
        help="How long CommandReady stays high per command pulse",
    )
    parser.add_argument("--drive-debug", action="store_true", help="Print outgoing NT drive commands")
    parser.add_argument(
        "--nt-health-debug",
        action="store_true",
        help="Print NetworkTables session health and robot-published Jetson drive keys",
    )
    parser.add_argument(
        "--nt-health-period-sec",
        type=float,
        default=1.0,
        help="Seconds between NetworkTables health debug prints",
    )
    parser.add_argument(
        "--nt-enable-heartbeat-sec",
        type=float,
        default=0.10,
        help="Seconds between automation-state heartbeat writes while driving",
    )
    parser.add_argument(
        "--nt-command-ack-timeout-sec",
        type=float,
        default=0.30,
        help="Seconds to wait before clearing a stuck CommandReady flag",
    )
    parser.add_argument(
        "--nt-forward-scale",
        type=float,
        default=1.0,
        help="Value written to Jetson/Speed while automation is enabled",
    )
    parser.add_argument(
        "--nt-turn-scale",
        type=float,
        default=1.0,
        help="Value written to Jetson/TurnSpeed while automation is enabled",
    )
    parser.add_argument("--floor-update-sec", type=float, default=0.5, help="Seconds between floor-plane updates")
    parser.add_argument("--floor-min-normal-y", type=float, default=0.5, help="Reject floor planes with |normal.y| below this")
    parser.add_argument(
        "--plane-ema-alpha",
        type=float,
        default=0.25,
        help="Smoothing factor for accepted floor-plane updates (0-1, higher=faster response)",
    )
    parser.add_argument(
        "--plane-max-tilt-delta-deg",
        type=float,
        default=8.0,
        help="Reject floor-plane updates that tilt more than this from prior plane (deg)",
    )
    parser.add_argument(
        "--plane-max-height-jump-m",
        type=float,
        default=0.08,
        help="Reject floor-plane updates with abrupt height offset jump (m)",
    )
    parser.add_argument(
        "--plane-force-accept-rejects",
        type=int,
        default=20,
        help="Force-accept next valid floor plane after this many consecutive jump rejections (0 disables)",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=8,
        help="Point-cloud downsample stride (higher = fewer points, lower CPU/noise)",
    )
    parser.add_argument(
        "--min-range-z-m",
        type=float,
        default=0.25,
        help="Ignore points closer than this forward distance (m) to reduce near-field depth noise",
    )
    parser.add_argument(
        "--max-range-z-m",
        type=float,
        default=6.0,
        help="Ignore points beyond this forward distance (m). Set <=0 to disable.",
    )
    parser.add_argument("--stream-ip", default=None, help="UDP target IP for GStreamer stream")
    parser.add_argument("--stream-port", type=int, default=5600, help="UDP port for GStreamer stream")
    parser.add_argument("--stream-fps", type=float, default=15.0, help="Stream FPS")
    parser.add_argument("--stream-bitrate-kbps", type=int, default=2500, help="Stream bitrate in kbps")
    parser.add_argument("--stream-view", default="both", choices=["camera", "map", "both"], help="Which view to stream")
    parser.add_argument("--map-command-file", default=os.path.join(SCRIPT_DIR, "zed_map_command.json"), help="Path to UI-issued map waypoint command file")
    parser.add_argument("--map-ui-state-file", default=os.path.join(SCRIPT_DIR, "zed_map_ui_state.json"), help="Path to published UI state JSON for remote map controls")
    parser.add_argument("--drive-calibration-file", default=os.path.join(SCRIPT_DIR, "zed_drive_calibration.json"), help="Path to saved drive-calibration JSON")
    parser.add_argument("--dig-profiles-path", default=os.path.join(SCRIPT_DIR, "zed_dig_profiles.json"), help="Path to recorded dig profile library JSON")
    parser.add_argument("--controller-macros-path", default=os.path.join(SCRIPT_DIR, "zed_controller_macros.json"), help="Path to recorded controller macro library JSON")
    parser.add_argument("--camera-heartbeat-url", default=None, help="HTTP endpoint that receives camera-owner heartbeats")
    parser.add_argument("--camera-heartbeat-interval-ms", type=int, default=1000, help="Interval between camera-owner heartbeats")
    parser.add_argument("--camera-heartbeat-timeout-ms", type=int, default=250, help="HTTP timeout for camera-owner heartbeats")
    parser.add_argument("--camera-heartbeat-source", default="zed_ground_wall", help="Source label attached to camera-owner heartbeats")
    parser.add_argument("--camera-publish-url", default=None, help="HTTP endpoint that receives JPEG camera view frames")
    parser.add_argument("--camera-publish-interval-ms", type=int, default=120, help="Min interval between camera frame publishes")
    parser.add_argument("--camera-publish-jpeg-quality", type=int, default=75, help="JPEG quality for published camera frames")
    parser.add_argument("--camera-publish-timeout-ms", type=int, default=250, help="HTTP timeout for published camera frames")
    parser.add_argument("--camera-publish-source", default="zed_ground_wall", help="Source label attached to published camera frames")
    parser.add_argument("--map-publish-url", default=None, help="HTTP endpoint that receives JPEG occupancy map frames")
    parser.add_argument("--map-publish-interval-ms", type=int, default=120, help="Min interval between map frame publishes")
    parser.add_argument("--map-publish-jpeg-quality", type=int, default=70, help="JPEG quality for published occupancy map")
    parser.add_argument("--map-publish-timeout-ms", type=int, default=250, help="HTTP timeout for published occupancy map")
    parser.add_argument("--map-publish-source", default="zed_ground_wall", help="Source label attached to published map frames")
    parser.add_argument("--manual-start", action="store_true", help="Start in keyboard manual drive mode")
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Run a lightweight camera/controller mode: skip tracking, depth, mapping, and AI detection.",
    )
    parser.add_argument(
        "--nt-timeout-sec",
        type=float,
        default=3.0,
        help="Stop driving if NetworkTables connection is lost for this many seconds (watchdog). 0 disables.",
    )
    parser.add_argument(
        "--ds-joystick",
        action="store_true",
        help="Read DS controller axes from NT (DS/JoystickFwd, DS/JoystickTurn) and mix into drive commands",
    )
    parser.add_argument(
        "--ds-joystick-fwd-key",
        default="DS/JoystickFwd",
        help="NT key the RoboRIO publishes for DS joystick forward axis (-1 to 1)",
    )
    parser.add_argument(
        "--ds-joystick-turn-key",
        default="DS/JoystickTurn",
        help="NT key the RoboRIO publishes for DS joystick turn axis (-1 to 1)",
    )
    parser.add_argument(
        "--ds-joystick-scale",
        type=float,
        default=0.5,
        help="Scale applied to DS joystick axes before mixing into auto drive (0-1)",
    )
    parser.add_argument("--no-gui", action="store_true", help="Disable local OpenCV windows")
    parser.add_argument(
        "--overlay-red-only",
        action="store_true",
        help="Show only red (obstacle) overlay on camera view, hide green ground coloring",
    )
    parser.add_argument("--human-detect", action="store_true", help="Enable ZED SDK human/person detection")
    parser.add_argument("--human-od-confidence", type=int, default=40, help="ZED OD confidence threshold (1-99)")
    parser.add_argument("--human-od-every", type=int, default=1, help="Run ZED OD every N frames")
    parser.add_argument("--human-stop-m", type=float, default=1.5, help="Person distance to trigger STOP (m)")
    parser.add_argument("--human-slow-m", type=float, default=3.0, help="Person distance to trigger SLOW (m)")
    parser.add_argument("--human-min-conf", type=float, default=0.40, help="Min person confidence for hazard state")
    # Rock detection (custom YOLO model)
    parser.add_argument("--rock-model", default="", help="Path to trained rock YOLO model (.pt). Leave empty to disable.")
    parser.add_argument("--rock-conf", type=float, default=0.35, help="Rock detection confidence threshold")
    parser.add_argument("--rock-every", type=int, default=5, help="Run rock detection every N frames")
    parser.add_argument("--rock-stamp", type=float, default=6.0, help="Obstacle evidence to stamp per detected rock cell")
    parser.add_argument("--rock-debug", action="store_true", help="Print custom YOLO rock detection decisions")
    parser.add_argument("--rock-snapshot-dir", default="", help="Directory to save annotated frames when rocks are detected")
    parser.add_argument("--rock-snapshot-cooldown", type=float, default=2.0, help="Minimum seconds between saved rock snapshots")
    parser.add_argument("--rock-classes", default="rock,stone,boulder", help="Comma-separated class names to treat as rocks")
    parser.add_argument(
        "--landmark-classes",
        default="backpack,rock,stone,boulder,obstacle",
        help="Comma-separated YOLO class names to save as persistent map landmarks.",
    )
    parser.add_argument(
        "--landmark-memory",
        action="store_true",
        default=True,
        help="Save static AI detections as map landmarks for relocalization awareness.",
    )
    parser.add_argument("--no-landmark-memory", action="store_false", dest="landmark_memory", help="Disable landmark memory")
    parser.add_argument("--landmark-path", default=os.path.join(SCRIPT_DIR, "zed_landmarks.json"), help="Path to static AI landmark memory JSON")
    parser.add_argument("--landmark-assoc-m", type=float, default=0.45, help="Merge detections into an existing landmark within this distance")
    parser.add_argument("--landmark-min-hits", type=int, default=2, help="Minimum repeated detections before drawing a landmark")
    parser.add_argument("--landmark-save-every", type=float, default=5.0, help="Seconds between landmark memory saves")
    parser.add_argument(
        "--landmark-relocalize",
        action="store_true",
        default=True,
        help="Use saved landmarks plus fallback heading to correct the map pose when tracking is lost.",
    )
    parser.add_argument(
        "--no-landmark-relocalize",
        action="store_false",
        dest="landmark_relocalize",
        help="Disable landmark-based pose correction while tracking is lost.",
    )
    parser.add_argument(
        "--landmark-relocalize-max-offset-m",
        type=float,
        default=4.0,
        help="Maximum XY correction allowed from a single landmark match while tracking is lost.",
    )
    parser.add_argument(
        "--landmark-relocalize-alpha",
        type=float,
        default=0.65,
        help="Smoothing factor (0-1) for landmark-based pose correction while tracking is lost.",
    )
    args = parser.parse_args()

    if args.camera_only:
        args.tracking = False
        args.area_memory = False
        args.human_detect = False
        args.rock_model = ""
        args.landmark_memory = False
        args.manual_start = True

    if args.rviz_config is None:
        args.rviz_config = os.path.join(os.path.dirname(__file__), "zed_pointcloud.rviz")
    rviz_utils.launch_rviz(args.rviz, args.rviz_config)

    node, pc_pub, pc_fields = ros2_utils.setup_ros2(args.ros2)

    try:
        zed = zed_utils.open_zed_camera(sl)
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)

    runtime = sl.RuntimeParameters()

    point_cloud = sl.Mat()
    image_left = sl.Mat()
    ground_plane = sl.Plane()
    tracking_reset = sl.Transform()
    tracking_enabled = False
    pose_warned = False
    tracking_pose_ok = False
    tracking_prev_ok = False
    tracking_loss_warned = False
    have_valid_tracking_pose = False
    tracking_recover_stable_count = 0
    imu_heading_warned = False
    imu_heading_enabled = False
    imu_sensors_data = None
    last_valid_imu_rotation = None
    navx_sign_score = 0.0
    navx_sign = 1.0
    navx_sign_locked = False
    navx_cal_last_yaw_deg = None
    navx_cal_last_rover_heading_deg = None
    last_valid_navx_yaw_deg = None
    pose = None
    if args.tracking:
        tracking_enabled, pose = zed_utils.enable_tracking(
            zed,
            sl,
            area_memory=args.area_memory,
            area_load_path=args.area_load_path,
        )
        tracking_pose_ok = not tracking_enabled
        tracking_prev_ok = tracking_pose_ok
    else:
        tracking_pose_ok = True
        tracking_prev_ok = True
    last_valid_R_world_cam = np.eye(3, dtype=np.float32)
    last_valid_t_world_cam = np.zeros(3, dtype=np.float32)
    imu_fallback_forward_world = None
    last_valid_rover_forward_world = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    last_valid_rover_right_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    map_origin_set = False
    map_origin_t = np.zeros(3, dtype=np.float32)
    spatial_enabled = False
    spatial_mesh = None
    last_spatial_save = time.time()
    last_area_save = time.time()
    if args.spatial_mapping:
        spatial_enabled, spatial_mesh = zed_utils.enable_spatial_mapping(
            zed,
            sl,
            resolution=args.spatial_res,
            mapping_range=args.spatial_range,
        )
    mesh_viewer = None
    if args.spatial_viewer:
        if not viewer_utils.HAS_OPEN3D:
            print("Open3D not available; mesh viewer disabled.")
        elif not args.spatial_save_path:
            print("Mesh viewer requires --spatial-save-path to load updates.")
        else:
            mesh_viewer = viewer_utils.MeshViewer()
            if not mesh_viewer.open():
                mesh_viewer = None

    if args.imu_heading_fallback and hasattr(sl, "SensorsData"):
        try:
            imu_sensors_data = sl.SensorsData()
            imu_heading_enabled = True
            print("IMU heading fallback enabled.")
        except Exception as exc:
            print(f"IMU heading fallback unavailable: {exc}")

    resizable_flags = 0
    fixed_flags = 0
    if not HAS_CV2:
        print("OpenCV not found. Install it for live visualization:")
        print("  sudo apt install -y python3-opencv")
    elif args.no_gui:
        print("GUI disabled (--no-gui): map/camera windows will not open.")
    elif not os.environ.get("DISPLAY"):
        args.no_gui = True
        print("GUI auto-disabled: DISPLAY is not set, running headless.")
    else:
        print("GUI enabled: opening camera/map windows.")
        try:
            gui_normal = getattr(cv2, "WINDOW_GUI_NORMAL", 0)
            resizable_flags = cv2.WINDOW_NORMAL | gui_normal
            fixed_flags = cv2.WINDOW_AUTOSIZE | gui_normal
            cv2.namedWindow("ZED Ground/Obstacle Segmentation", resizable_flags)
            if not args.camera_only:
                cv2.namedWindow("ZED Occupancy Map (XZ)", fixed_flags)
            cv2.namedWindow("ZED Drive Status", fixed_flags)
            cv2.resizeWindow("ZED Ground/Obstacle Segmentation", 1280, 720)
        except Exception:
            pass

    camera_heartbeat = None
    if args.camera_heartbeat_url:
        camera_heartbeat = camera_status_client.CameraStatusHeartbeat(
            args.camera_heartbeat_url,
            backend="zed",
            source=args.camera_heartbeat_source,
            interval_ms=args.camera_heartbeat_interval_ms,
            timeout_ms=args.camera_heartbeat_timeout_ms,
            streaming=bool(args.camera_publish_url),
        )
        print(f"Camera heartbeat enabled: {args.camera_heartbeat_url}")

    camera_publisher = None
    if args.camera_publish_url:
        if not HAS_CV2:
            print("Camera publisher requested, but OpenCV is unavailable; disabling camera publishing.")
        else:
            camera_publisher = camera_publish_client.HttpCameraPublisher(
                args.camera_publish_url,
                interval_ms=args.camera_publish_interval_ms,
                jpeg_quality=args.camera_publish_jpeg_quality,
                timeout_ms=args.camera_publish_timeout_ms,
                source=args.camera_publish_source,
            )
            print(f"Camera publish enabled: {args.camera_publish_url}")

    map_publisher = None
    if args.map_publish_url:
        if not HAS_CV2:
            print("Map publisher requested, but OpenCV is unavailable; disabling map publishing.")
        else:
            map_publisher = map_publish_client.HttpMapPublisher(
                args.map_publish_url,
                interval_ms=args.map_publish_interval_ms,
                jpeg_quality=args.map_publish_jpeg_quality,
                timeout_ms=args.map_publish_timeout_ms,
                source=args.map_publish_source,
            )
            print(f"Map publish enabled: {args.map_publish_url}")

    print("Running. Press Ctrl+C to exit.")
    mapping_mode = "complex" if args.complex else "simple"
    print(f"Mapping mode: {mapping_mode}")

    def _parse_class_name_set(csv_text):
        return {token.strip().lower() for token in str(csv_text or "").split(",") if token.strip()}

    # Rock detection via custom YOLO model
    rock_model = None
    rock_last_frame = -999999
    rock_last_snapshot_time = 0.0
    rock_class_names = _parse_class_name_set(args.rock_classes)
    landmark_class_names = _parse_class_name_set(args.landmark_classes)
    if args.rock_model:
        try:
            from ultralytics import YOLO as _YOLO
            rock_model = _YOLO(args.rock_model)
            print(
                "Rock detection model loaded: "
                f"{args.rock_model}  obstacle_classes={rock_class_names} "
                f"landmark_classes={landmark_class_names}"
            )
        except Exception as _rock_exc:
            print(f"[WARN] Could not load rock model '{args.rock_model}': {_rock_exc}")
            rock_model = None

    # Human detection via ZED SDK built-in object detection
    human_detect_available = False
    human_objects = None
    human_od_runtime = None
    human_last_frame = -999999
    human_hazard_state = "CLEAR"
    human_nearest_m = -1.0
    human_clear_hold = 12
    human_clear_countdown = 0
    if args.human_detect:
        if hasattr(sl, "ObjectDetectionParameters"):
            try:
                od_params = sl.ObjectDetectionParameters()
                if hasattr(od_params, "enable_tracking"):
                    od_params.enable_tracking = False
                model_enum = getattr(sl, "OBJECT_DETECTION_MODEL", None)
                if model_enum is not None and hasattr(od_params, "detection_model"):
                    for mn in ["MULTI_CLASS_BOX_MEDIUM", "MULTI_CLASS_BOX_FAST", "MULTI_CLASS_BOX"]:
                        if hasattr(model_enum, mn):
                            od_params.detection_model = getattr(model_enum, mn)
                            break
                err = zed.enable_object_detection(od_params)
                if err == sl.ERROR_CODE.SUCCESS:
                    human_objects = sl.Objects()
                    human_od_runtime = sl.ObjectDetectionRuntimeParameters()
                    if hasattr(human_od_runtime, "detection_confidence_threshold"):
                        human_od_runtime.detection_confidence_threshold = int(
                            max(1, min(99, args.human_od_confidence))
                        )
                    human_detect_available = True
                    print("Human detection enabled (ZED SDK object detection).")
                else:
                    print(f"Failed to enable ZED object detection: {err}")
            except Exception as exc:
                print(f"ZED object detection init error: {exc}")
        else:
            print("ZED SDK ObjectDetectionParameters not available; human detection disabled.")
    human_detect_enabled = bool(human_detect_available)
    rock_detect_enabled = False

    # Simple 2D occupancy map settings (XZ plane, Y up).
    # X: left/right, Z: forward. Units: meters.
    map_z_min = args.map_z_min
    if args.map_center:
        map_z_min = -args.map_height_m / 2.0

    occ_map = map_utils.OccupancyMap(
        map_res_m=args.map_res_m,
        map_width_m=args.map_width_m,
        map_height_m=args.map_height_m,
        map_z_min=map_z_min,
        decay=args.map_decay,
        free_decay=args.free_decay,
        free_decay_unconfirmed=args.free_decay_unconfirmed,
        free_decay_confirmed=args.free_decay_confirmed,
        free_confirm_hits=args.free_confirm_hits,
        free_confirm_ratio=args.free_confirm_ratio,
        free_downgrade_factor=args.free_downgrade_factor,
        occ_decay=args.occ_decay,
        hole_decay=args.hole_decay,
    )
    last_save = time.time()
    last_recovery_save = time.time()
    recovery_checkpoint = None
    recovery_pending_alignment = False
    recovery_alignment_offset_t = np.zeros(3, dtype=np.float32)
    recovery_alignment_yaw_deg = 0.0
    recovery_loaded_from_checkpoint = False
    recovery_jump_reject_count = 0

    def map_x_from_zed(x):
        # ZED +X is camera-right, while the occupancy map image mirrors X for display.
        return -float(x)

    def map_world_to_grid(x, z):
        return occ_map.world_to_grid(map_x_from_zed(x), float(z))

    def zed_x_from_map(x):
        return -float(x)

    def parse_start_frame_tag_layout(layout_text):
        layout = {}
        for raw_item in str(layout_text or "").split(";"):
            item = raw_item.strip()
            if not item:
                continue
            try:
                tag_str, coord_str = item.split(":", 1)
                u_str, v_str = coord_str.split(",", 1)
                layout[int(tag_str.strip())] = np.array(
                    [float(u_str.strip()), float(v_str.strip())],
                    dtype=np.float32,
                )
            except Exception:
                continue
        return layout

    def resolve_start_frame_aruco_dict(name):
        if (not HAS_CV2) or (not hasattr(cv2, "aruco")):
            return None
        raw_name = str(name or "").strip()
        if not raw_name:
            raw_name = "DICT_APRILTAG_25h9"
        candidates = [raw_name]
        lower_name = raw_name.lower()
        for attr in dir(cv2.aruco):
            if attr.lower() == lower_name:
                candidates.append(attr)
        for attr_name in candidates:
            if hasattr(cv2.aruco, attr_name):
                dict_id = getattr(cv2.aruco, attr_name)
                try:
                    return cv2.aruco.getPredefinedDictionary(dict_id)
                except Exception:
                    continue
        return None

    start_frame_tag_layout = parse_start_frame_tag_layout(
        os.getenv("START_FRAME_TAG_LAYOUT", "10:0.50,0.00;11:0.50,0.75;12:1.25,0.00")
    )
    start_frame_tag_dictionary = resolve_start_frame_aruco_dict(
        os.getenv("START_FRAME_TAG_DICT", "DICT_APRILTAG_25h9")
    )
    start_frame_tag_sample_radius_px = max(
        1,
        int(float(os.getenv("START_FRAME_TAG_SAMPLE_RADIUS_PX", "6"))),
    )
    start_frame_scan_duration_sec = max(
        0.25,
        float(os.getenv("START_FRAME_SCAN_DURATION_SEC", "0.9")),
    )
    start_frame_scan_min_samples = max(
        2,
        int(float(os.getenv("START_FRAME_SCAN_MIN_SAMPLES", "4"))),
    )

    def fit_rigid_transform_2d(src_pts, dst_pts):
        src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 2)
        dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 2)
        if src.shape[0] < 2 or dst.shape[0] != src.shape[0]:
            return None, None, None
        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst, axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        H = src_centered.T @ dst_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T
        t = dst_mean - (R @ src_mean)
        pred = (src @ R.T) + t
        err = np.linalg.norm(pred - dst, axis=1)
        return R.astype(np.float32), t.astype(np.float32), float(np.mean(err))

    def detect_start_frame_tags(image_bgr, cloud, R_world_cam, t_map):
        if start_frame_tag_dictionary is None:
            return [], "OpenCV ArUco/AprilTag support not available."
        if not start_frame_tag_layout:
            return [], "No start-frame tag layout configured."
        if image_bgr is None or cloud is None:
            return [], "Missing camera image or point cloud."
        if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
            return [], "Unexpected image format."

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        try:
            params = cv2.aruco.DetectorParameters()
        except Exception:
            params = cv2.aruco.DetectorParameters_create()
        try:
            detector = cv2.aruco.ArucoDetector(start_frame_tag_dictionary, params)
            corners, ids, _ = detector.detectMarkers(gray)
        except Exception:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, start_frame_tag_dictionary, parameters=params)

        if ids is None or len(ids) <= 0:
            return [], "No configured AprilTags detected."

        img_h, img_w = image_bgr.shape[:2]
        cld_h, cld_w = cloud.shape[:2]
        detections = []
        for det_corners, det_id_arr in zip(corners, ids):
            tag_id = int(det_id_arr[0])
            if tag_id not in start_frame_tag_layout:
                continue
            pts = np.asarray(det_corners, dtype=np.float32).reshape(-1, 2)
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            pc_c = int(round(cx * cld_w / max(1, img_w)))
            pc_r = int(round(cy * cld_h / max(1, img_h)))
            rad = start_frame_tag_sample_radius_px
            r0 = max(0, pc_r - rad)
            r1 = min(cld_h, pc_r + rad + 1)
            c0 = max(0, pc_c - rad)
            c1 = min(cld_w, pc_c + rad + 1)
            sample = cloud[r0:r1, c0:c1, :3].reshape(-1, 3)
            valid = sample[np.isfinite(sample).all(axis=1)]
            if valid.shape[0] <= 0:
                continue
            pt_cam = np.median(valid, axis=0).astype(np.float32)
            pt_world = (np.asarray(R_world_cam, dtype=np.float32).reshape(3, 3) @ pt_cam) + np.asarray(t_map, dtype=np.float32).reshape(3,)
            detections.append(
                {
                    "id": tag_id,
                    "local_uv": np.array(start_frame_tag_layout[tag_id], dtype=np.float32),
                    "map_xz": np.array([map_x_from_zed(float(pt_world[0])), float(pt_world[2])], dtype=np.float32),
                    "image_xy": (cx, cy),
                }
            )
        if len(detections) < 3:
            return detections, f"Detected only {len(detections)} configured tag(s); need 3."
        return detections, None

    def detect_start_frame_markers_2d(image_bgr):
        if start_frame_tag_dictionary is None or image_bgr is None:
            return []
        if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
            return []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        try:
            params = cv2.aruco.DetectorParameters()
        except Exception:
            params = cv2.aruco.DetectorParameters_create()
        try:
            detector = cv2.aruco.ArucoDetector(start_frame_tag_dictionary, params)
            corners, ids, _ = detector.detectMarkers(gray)
        except Exception:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, start_frame_tag_dictionary, parameters=params)

        if ids is None or len(ids) <= 0:
            return []

        detections = []
        for det_corners, det_id_arr in zip(corners, ids):
            tag_id = int(det_id_arr[0])
            if tag_id not in start_frame_tag_layout:
                continue
            pts = np.asarray(det_corners, dtype=np.float32).reshape(-1, 2)
            center = np.mean(pts, axis=0)
            detections.append(
                {
                    "id": tag_id,
                    "corners": pts,
                    "center": (float(center[0]), float(center[1])),
                }
            )
        detections.sort(key=lambda item: int(item["id"]))
        return detections

    def apply_start_frame_from_detection_set(detections, status_prefix="Start frame locked"):
        nonlocal start_frame_last_status, start_frame_last_ids, start_frame_last_error_m
        nonlocal start_frame_last_map_points
        start_frame_last_ids = [int(item["id"]) for item in detections]
        start_frame_last_map_points = [
            {
                "id": int(item["id"]),
                "map_x": float(item["map_xz"][0]),
                "map_z": float(item["map_xz"][1]),
            }
            for item in detections
        ]

        detections = sorted(detections, key=lambda item: int(item["id"]))[:3]
        local_pts = np.array([item["local_uv"] for item in detections], dtype=np.float32)
        map_pts = np.array([item["map_xz"] for item in detections], dtype=np.float32)
        R2, t2, fit_err = fit_rigid_transform_2d(local_pts, map_pts)
        if R2 is None:
            start_frame_last_error_m = None
            start_frame_last_status = "Start frame: transform fit failed."
            print(start_frame_last_status)
            return False

        start_w = float(_mining_cfg.get("starting_zone_width_m", 1.50))
        start_d = float(_mining_cfg.get("starting_zone_depth_m", 1.50))
        apply_start_to_excav = str(_mining_cfg.get("starting_zone_apply_to_excav", "1")).strip().lower() not in ("0", "false", "no", "off", "")

        def transform_local_poly(local_poly):
            local_arr = np.asarray(local_poly, dtype=np.float32).reshape(-1, 2)
            return (local_arr @ R2.T) + t2

        def transform_world_to_local(world_poly):
            world_arr = np.asarray(world_poly, dtype=np.float32).reshape(-1, 2)
            return (world_arr - t2) @ R2

        start_local = np.array(
            [[0.0, 0.0], [start_w, 0.0], [start_w, start_d], [0.0, start_d]],
            dtype=np.float32,
        )
        start_world = transform_local_poly(start_local)

        map_world = np.array(
            [
                [float(occ_map.x_min), float(occ_map.z_min)],
                [float(occ_map.x_max), float(occ_map.z_min)],
                [float(occ_map.x_max), float(occ_map.z_max)],
                [float(occ_map.x_min), float(occ_map.z_max)],
            ],
            dtype=np.float32,
        )
        map_local = transform_world_to_local(map_world)
        local_u_min = float(np.min(map_local[:, 0]))
        local_u_max = float(np.max(map_local[:, 0]))
        local_v_min = float(np.min(map_local[:, 1]))
        local_v_max = float(np.max(map_local[:, 1]))
        span_u = max(0.0, local_u_max - local_u_min)
        span_v = max(0.0, local_v_max - local_v_min)

        if span_u >= span_v:
            excav_axis = "u"
            split_value = start_w + max(0.0, (local_u_max - start_w) * 0.5)
            split_value = min(max(split_value, local_u_min), local_u_max)
            excav_local = np.array(
                [
                    [split_value, local_v_min],
                    [local_u_max, local_v_min],
                    [local_u_max, local_v_max],
                    [split_value, local_v_max],
                ],
                dtype=np.float32,
            )
        else:
            excav_axis = "v"
            split_value = start_d + max(0.0, (local_v_max - start_d) * 0.5)
            split_value = min(max(split_value, local_v_min), local_v_max)
            excav_local = np.array(
                [
                    [local_u_min, split_value],
                    [local_u_max, split_value],
                    [local_u_max, local_v_max],
                    [local_u_min, local_v_max],
                ],
                dtype=np.float32,
            )

        excav_world = transform_local_poly(excav_local)

        def world_poly_to_rc(poly_world):
            out = []
            for map_x, map_z in np.asarray(poly_world, dtype=np.float32):
                rc = occ_map.world_to_grid(float(map_x), float(map_z))
                if rc is None:
                    return None
                out.append(rc)
            return out

        start_rc = world_poly_to_rc(start_world)
        excav_rc = world_poly_to_rc(excav_world)
        if start_rc is None or (apply_start_to_excav and excav_rc is None):
            start_frame_last_error_m = None
            start_frame_last_status = "Start frame: transformed zones fell outside current map bounds."
            print(start_frame_last_status)
            return False

        mining.starting_corners_rc = list(start_rc)
        mining.starting_zone_preset_side = None
        mining.deposit_corners_rc = list(start_rc)
        mining.deposit_zone_preset_side = None
        if apply_start_to_excav:
            mining.excav_corners_rc = list(excav_rc)
            mining.preferred_start_rc = None
        mining._deposit_approach_rc = None
        mining.save_zones(occ_map)

        start_frame_last_error_m = float(fit_err)
        start_frame_last_status = (
            f"{status_prefix} from tags {start_frame_last_ids} "
            f"(fit {float(fit_err):.03f} m, deposit=start, excav={excav_axis}-half)."
        )
        print(start_frame_last_status)
        return True

    def apply_start_frame_from_tags(image_bgr, cloud, R_world_cam, t_map):
        detections, error_msg = detect_start_frame_tags(image_bgr, cloud, R_world_cam, t_map)
        if error_msg is not None:
            nonlocal start_frame_last_status, start_frame_last_error_m
            start_frame_last_error_m = None
            start_frame_last_status = f"Start frame: {error_msg}"
            print(start_frame_last_status)
            return False
        return apply_start_frame_from_detection_set(detections)

    camera_mount_yaw_deg = args.camera_mount_yaw_deg
    if camera_mount_yaw_deg is None:
        camera_mount_yaw_deg = 180.0 if args.camera_mount == "rear" else 0.0

    camera_forward_offset_m = args.camera_forward_offset_m
    if camera_forward_offset_m is None:
        # Default mount presets place the camera on the leading edge of the
        # rover footprint so planning starts from the rover body center, not
        # from the camera lens itself.
        camera_forward_offset_m = (
            -0.5 * float(args.rover_size_m)
            if args.camera_mount == "rear"
            else 0.5 * float(args.rover_size_m)
        )
    camera_right_offset_m = float(args.camera_right_offset_m)

    def angle_error_deg(angle_a, angle_b):
        err = float(angle_a) - float(angle_b)
        while err > 180.0:
            err -= 360.0
        while err < -180.0:
            err += 360.0
        return err

    def camera_view_flip_active():
        normal_error = (
            abs(angle_error_deg(args.camera_map_angle_deg, DEFAULT_CAMERA_MAP_ANGLE_DEG))
            + abs(angle_error_deg(args.camera_deposit_angle_deg, DEFAULT_CAMERA_DEPOSIT_ANGLE_DEG))
        )
        flipped_error = (
            abs(angle_error_deg(args.camera_map_angle_deg, DEFAULT_CAMERA_DEPOSIT_ANGLE_DEG))
            + abs(angle_error_deg(args.camera_deposit_angle_deg, DEFAULT_CAMERA_MAP_ANGLE_DEG))
        )
        return flipped_error < normal_error

    def servo_raw_to_logical(angle_deg):
        angle_deg = max(0.0, min(180.0, float(angle_deg)))
        if args.camera_servo_invert:
            return 180.0 - angle_deg
        return angle_deg

    def servo_logical_to_raw(angle_deg):
        angle_deg = max(0.0, min(180.0, float(angle_deg)))
        if args.camera_servo_invert:
            return 180.0 - angle_deg
        return angle_deg

    def wrap_angle_deg(angle_deg):
        angle = float(angle_deg)
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle

    def yaw_rotation_matrix_deg(angle_deg):
        yaw_rad = math.radians(float(angle_deg))
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        return np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float32,
        )

    def camera_mount_axes(yaw_deg):
        yaw_rad = math.radians(float(yaw_deg))
        robot_forward_cam = np.array(
            [math.sin(yaw_rad), 0.0, math.cos(yaw_rad)],
            dtype=np.float32,
        )
        robot_right_cam = np.array(
            [math.cos(yaw_rad), 0.0, -math.sin(yaw_rad)],
            dtype=np.float32,
        )
        close_obstacle_escape_sign = -1.0 if math.cos(yaw_rad) >= 0.0 else 1.0
        return robot_forward_cam, robot_right_cam, close_obstacle_escape_sign

    servo_angle_deg = float(args.camera_map_angle_deg if args.camera_servo_track else camera_mount_yaw_deg)
    servo_target_angle_deg = float(servo_angle_deg)
    servo_command_angle_deg = float(servo_angle_deg)
    servo_settled = True
    servo_turning = False
    servo_map_view = True
    servo_deposit_view = False
    servo_manual_override = False

    def current_camera_mount_yaw_deg():
        return float(servo_angle_deg if args.camera_servo_track else camera_mount_yaw_deg)

    def rover_pose_from_camera(R_world_cam, camera_pos_world, mount_yaw_deg=None):
        if mount_yaw_deg is None:
            mount_yaw_deg = current_camera_mount_yaw_deg()
        robot_forward_cam, robot_right_cam, _ = camera_mount_axes(mount_yaw_deg)
        rover_forward_world = (R_world_cam @ robot_forward_cam.reshape(3, 1)).reshape(3,)
        rover_right_world = (R_world_cam @ robot_right_cam.reshape(3, 1)).reshape(3,)
        rover_pos_world = (
            np.array(camera_pos_world, dtype=np.float32)
            - rover_forward_world * float(camera_forward_offset_m)
            - rover_right_world * float(camera_right_offset_m)
        )
        return rover_pos_world, rover_forward_world, rover_right_world

    def drive_forward_world_from_rover(forward_world):
        forward = np.array(forward_world, dtype=np.float32).reshape(3,)
        if args.drive_heading_flip:
            return -forward
        return forward

    def navigation_origin_world(rover_pos_world, rover_forward_world):
        if rover_pos_world is None or rover_forward_world is None:
            return None
        pos = np.array(rover_pos_world, dtype=np.float32).reshape(3,)
        forward = np.array(drive_forward_world_from_rover(rover_forward_world), dtype=np.float32).reshape(3,)
        return pos + forward * max(0.0, float(args.rover_size_m) * 0.5)

    def world_forward_from_rotation(R_world_cam):
        forward = (np.array(R_world_cam, dtype=np.float32) @ np.array([0.0, 0.0, 1.0], dtype=np.float32)).reshape(3,)
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return forward / norm

    def rotate_world_xz(vec_world, delta_deg):
        vec = np.array(vec_world, dtype=np.float32).reshape(3,)
        delta_rad = math.radians(float(delta_deg))
        c = math.cos(delta_rad)
        s = math.sin(delta_rad)
        x = c * float(vec[0]) - s * float(vec[2])
        z = s * float(vec[0]) + c * float(vec[2])
        rotated = np.array([x, float(vec[1]), z], dtype=np.float32)
        norm = float(np.linalg.norm(rotated[[0, 2]]))
        if norm <= 1e-6:
            return np.array(vec, dtype=np.float32)
        rotated[0] /= norm
        rotated[2] /= norm
        return rotated

    def heading_delta_deg(from_forward_world, to_forward_world):
        return wrap_angle_deg(
            rover_heading_deg_from_forward(to_forward_world)
            - rover_heading_deg_from_forward(from_forward_world)
        )

    def rover_heading_deg_from_forward(forward_world):
        forward = np.array(forward_world, dtype=np.float32).reshape(3,)
        return math.degrees(math.atan2(float(forward[2]), float(forward[0])))

    def camera_forward_from_rover_axes(rover_forward_world, rover_right_world, mount_yaw_deg=None):
        if mount_yaw_deg is None:
            mount_yaw_deg = current_camera_mount_yaw_deg()
        yaw_rad = math.radians(float(mount_yaw_deg))
        forward = (
            np.array(rover_forward_world, dtype=np.float32).reshape(3,) * math.cos(yaw_rad)
            - np.array(rover_right_world, dtype=np.float32).reshape(3,) * math.sin(yaw_rad)
        )
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return (forward / norm).astype(np.float32)

    def camera_right_from_rover_axes(rover_forward_world, rover_right_world, mount_yaw_deg=None):
        if mount_yaw_deg is None:
            mount_yaw_deg = current_camera_mount_yaw_deg()
        yaw_rad = math.radians(float(mount_yaw_deg))
        right = (
            np.array(rover_forward_world, dtype=np.float32).reshape(3,) * math.sin(yaw_rad)
            + np.array(rover_right_world, dtype=np.float32).reshape(3,) * math.cos(yaw_rad)
        )
        norm = float(np.linalg.norm(right))
        if norm <= 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return (right / norm).astype(np.float32)

    def display_forward_world(R_world_cam, rover_forward_world, tracking_ok=True, imu_forward_fallback=None):
        if (not tracking_ok) and imu_forward_fallback is not None:
            forward = np.array(imu_forward_fallback, dtype=np.float32).reshape(3,)
        else:
            forward = np.array(rover_forward_world, dtype=np.float32).reshape(3,)
        if args.display_heading_flip:
            forward = -forward
        return forward

    def camera_rotation_from_forward_world(forward_world):
        if forward_world is None:
            return None
        forward = np.array(forward_world, dtype=np.float32).reshape(3,)
        norm_xz = float(np.linalg.norm(forward[[0, 2]]))
        if norm_xz <= 1e-6:
            return None
        forward[0] /= norm_xz
        forward[2] /= norm_xz
        yaw_deg = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
        return yaw_rotation_matrix_deg(yaw_deg)

    def apply_recovery_alignment(R_world_cam, t_world_cam):
        if (
            np.linalg.norm(recovery_alignment_offset_t) <= 1e-6
            and abs(float(recovery_alignment_yaw_deg)) <= 1e-6
        ):
            return np.array(R_world_cam, dtype=np.float32), np.array(t_world_cam, dtype=np.float32).reshape(3,)
        align_R = yaw_rotation_matrix_deg(recovery_alignment_yaw_deg)
        raw_R = np.array(R_world_cam, dtype=np.float32).reshape(3, 3)
        raw_t = np.array(t_world_cam, dtype=np.float32).reshape(3,)
        aligned_R = (align_R @ raw_R).astype(np.float32)
        aligned_t = (align_R @ raw_t + recovery_alignment_offset_t).astype(np.float32)
        return aligned_R, aligned_t

    def angle_between_vec_deg(vec_a, vec_b):
        a = np.array(vec_a, dtype=np.float32).reshape(3,)
        b = np.array(vec_b, dtype=np.float32).reshape(3,)
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an <= 1e-6 or bn <= 1e-6:
            return 0.0
        dot = float(np.clip(np.dot(a / an, b / bn), -1.0, 1.0))
        return float(np.degrees(np.arccos(dot)))

    def estimate_world_rotation_from_imu(current_imu_rotation):
        if (
            current_imu_rotation is None
            or last_valid_imu_rotation is None
            or not have_valid_tracking_pose
        ):
            return None
        ref_imu_rot = np.array(last_valid_imu_rotation, dtype=np.float32).reshape(3, 3)
        cur_imu_rot = np.array(current_imu_rotation, dtype=np.float32).reshape(3, 3)
        delta_rot = cur_imu_rot @ ref_imu_rot.T
        return (delta_rot @ np.array(last_valid_R_world_cam, dtype=np.float32).reshape(3, 3)).astype(np.float32)

    def read_navx_yaw_deg():
        if (not args.navx_heading_aid) or (sd is None):
            return None
        try:
            return float(sd.getNumber("NavX/YawDeg", float("nan")))
        except Exception:
            return None

    def estimate_rover_axes_from_navx(navx_yaw_deg):
        if (
            navx_yaw_deg is None
            or not np.isfinite(navx_yaw_deg)
            or last_valid_navx_yaw_deg is None
            or not navx_sign_locked
            or not have_valid_tracking_pose
        ):
            return None, None
        delta_deg = navx_sign * wrap_angle_deg(float(navx_yaw_deg) - float(last_valid_navx_yaw_deg))
        est_forward = rotate_world_xz(last_valid_rover_forward_world, delta_deg)
        est_right = rotate_world_xz(last_valid_rover_right_world, delta_deg)
        return est_forward, est_right

    def update_navx_sign_calibration(navx_yaw_deg, rover_forward_world):
        nonlocal navx_sign_score, navx_sign, navx_sign_locked
        nonlocal navx_cal_last_yaw_deg, navx_cal_last_rover_heading_deg
        if navx_yaw_deg is None or (not np.isfinite(navx_yaw_deg)):
            return
        rover_heading_deg = rover_heading_deg_from_forward(rover_forward_world)
        if navx_cal_last_yaw_deg is not None and navx_cal_last_rover_heading_deg is not None:
            navx_delta = wrap_angle_deg(float(navx_yaw_deg) - float(navx_cal_last_yaw_deg))
            rover_delta = wrap_angle_deg(rover_heading_deg - float(navx_cal_last_rover_heading_deg))
            if abs(navx_delta) >= 1.0 and abs(rover_delta) >= 1.0:
                navx_sign_score += 1.0 if (navx_delta * rover_delta) >= 0.0 else -1.0
                navx_sign_score = max(-6.0, min(6.0, navx_sign_score))
                if (not navx_sign_locked) and abs(navx_sign_score) >= 3.0:
                    navx_sign = 1.0 if navx_sign_score >= 0.0 else -1.0
                    navx_sign_locked = True
                    print(
                        "NavX heading aid calibrated "
                        f"({'normal' if navx_sign > 0 else 'inverted'} yaw sign)."
                    )
        navx_cal_last_yaw_deg = float(navx_yaw_deg)
        navx_cal_last_rover_heading_deg = float(rover_heading_deg)

    print(
        "Camera mount: "
        f"{args.camera_mount} yaw={float(camera_mount_yaw_deg):+.1f}deg "
        f"forward_offset={float(camera_forward_offset_m):+.2f}m "
        f"right_offset={float(camera_right_offset_m):+.2f}m "
        f"servo_track={'on' if args.camera_servo_track else 'off'} "
        f"servo_invert={'on' if args.camera_servo_invert else 'off'} "
        f"heading_flip={'on' if args.drive_heading_flip else 'off'} "
        f"hard_drive_flip={'on' if args.hard_drive_flip else 'off'} "
        f"steering_flip={'on' if args.steering_flip else 'off'} "
        f"display_heading_flip={'on' if args.display_heading_flip else 'off'}"
    )

    if args.map_load and os.path.exists(args.map_save_path):
        try:
            ok, msg = occ_map.load(args.map_save_path)
            print(f"{msg} ({args.map_save_path})" if ok else msg)
        except Exception as exc:
            print(f"Failed to load map ({args.map_save_path}): {exc}")

    # --- Mining automation subsystem ---
    _mining_cfg = {
        "dig_duration":          float(os.getenv("MINING_DIG_DURATION",           "52.0")),
        "dig_speed":             float(os.getenv("MINING_DIG_SPEED",              "0.20")),
        "backup_duration":       float(os.getenv("MINING_BACKUP_DURATION",        "5.0")),
        "backup_speed":          float(os.getenv("MINING_BACKUP_SPEED",           "0.35")),
        "dig_cycles":            float(os.getenv("MINING_DIG_CYCLES",             "4")),
        "dig_pullup_duration":   float(os.getenv("MINING_DIG_PULLUP_DURATION",    "2.0")),
        "deposit_duration":      float(os.getenv("MINING_DEPOSIT_DURATION",       "14.0")),
        "deposit_backup_speed":  float(os.getenv("MINING_DEPOSIT_BACKUP_SPEED",   "0.35")),
        "deposit_approach_dist": float(os.getenv("MINING_DEPOSIT_APPROACH_DIST",  "1.0")),
        "deposit_boundary_inset_m": float(os.getenv(
            "MINING_DEPOSIT_BOUNDARY_INSET_M", "0.05"
        )),
        "continuous_runs":      os.getenv("MINING_CONTINUOUS_RUNS", "1"),
        "strip_pitch_m":         float(os.getenv("MINING_STRIP_PITCH",            "0.0")),
        "goal_tol_m":            float(os.getenv("MINING_GOAL_TOL_M",             "0.55")),
        "rover_size_m":          float(args.rover_size_m),
        "berm_left_center_x_m":  float(os.getenv("MINING_BERM_LEFT_CENTER_X_M",   "-6.80")),
        "berm_right_center_x_m": float(os.getenv("MINING_BERM_RIGHT_CENTER_X_M",  "6.80")),
        "berm_center_z_m":       float(os.getenv("MINING_BERM_CENTER_Z_M",        "3.57")),
        "berm_width_m":          float(os.getenv("MINING_BERM_WIDTH_M",           "1.50")),
        "berm_depth_m":          float(os.getenv("MINING_BERM_DEPTH_M",           "0.90")),
        "starting_zone_side":    os.getenv("MINING_STARTING_ZONE_SIDE",          "right"),
        "starting_zone_origin_x_m": float(os.getenv("MINING_STARTING_ZONE_ORIGIN_X_M", "0.0")),
        "starting_zone_origin_z_m": float(os.getenv("MINING_STARTING_ZONE_ORIGIN_Z_M", "0.0")),
        "starting_zone_width_m": float(os.getenv("MINING_STARTING_ZONE_WIDTH_M", "1.50")),
        "starting_zone_depth_m": float(os.getenv("MINING_STARTING_ZONE_DEPTH_M", "1.50")),
        "starting_zone_apply_to_excav": os.getenv("MINING_STARTING_ZONE_APPLY_TO_EXCAV", "1"),
        "zones_path":            os.getenv("MINING_ZONES_PATH",
                                           os.path.join(SCRIPT_DIR, "mining_zones.json")),
    }
    mining = auto_mining.MiningAutomation(_mining_cfg, occ_map)
    default_dig_duration_sec = float(_mining_cfg.get("dig_duration", 5.0))
    default_backup_duration_sec = float(_mining_cfg.get("backup_duration", 2.0))
    drive_calibration = calibration_profiles.DriveCalibrationManager(args.drive_calibration_file)
    dig_profiles = calibration_profiles.DigProfileLibrary(args.dig_profiles_path)
    controller_macros = calibration_profiles.ControllerMacroLibrary(args.controller_macros_path)
    dig_profile_playback_cmd = None
    controller_macro_playback_cmd = None
    start_frame_lock_requested = False
    start_frame_last_status = "Start frame: idle"
    start_frame_last_ids = []
    start_frame_last_error_m = None
    start_frame_last_map_points = []
    start_frame_auto_lock_enabled = str(os.getenv("START_FRAME_AUTO_LOCK", "1")).strip().lower() not in ("0", "false", "no", "off", "")
    start_frame_auto_retry_sec = max(0.25, float(os.getenv("START_FRAME_AUTO_RETRY_SEC", "1.0")))
    start_frame_locked_once = False
    start_frame_last_attempt_time = 0.0
    start_frame_scan_active = False
    start_frame_scan_started_at = 0.0
    start_frame_scan_samples = []
    if drive_calibration.last_saved_flip is not None:
        args.drive_heading_flip = bool(drive_calibration.last_saved_flip)
    if drive_calibration.last_saved_hard_drive_flip is not None:
        args.hard_drive_flip = bool(drive_calibration.last_saved_hard_drive_flip)
    if drive_calibration.last_saved_steering_flip is not None:
        args.steering_flip = bool(drive_calibration.last_saved_steering_flip)
    if drive_calibration.last_saved_display_heading_flip is not None:
        args.display_heading_flip = bool(drive_calibration.last_saved_display_heading_flip)
    if drive_calibration.last_saved_camera_map_angle_deg is not None:
        args.camera_map_angle_deg = float(drive_calibration.last_saved_camera_map_angle_deg)
    if drive_calibration.last_saved_camera_deposit_angle_deg is not None:
        args.camera_deposit_angle_deg = float(drive_calibration.last_saved_camera_deposit_angle_deg)

    emergency_stop = False
    sd = None
    if args.drive:
        if not HAS_NT:
            print("NetworkTables not available; disable --drive or install pynetworktables.")
        else:
            NetworkTables.initialize(server=args.roborio_ip)
            sd = NetworkTables.getTable("SmartDashboard")
            sd.putBoolean("Drive/UseMainRoverControls", bool(args.main_rover_mode))
            sd.putBoolean("Drive/MainRoverDebugMode", bool(args.main_rover_debug))
            sd.putBoolean("Drive/MainRoverEmergencyStop", bool(emergency_stop))
            sd.putBoolean("Drive/MainRoverHardFlip", bool(args.hard_drive_flip))
            sd.putString("Jetson/MiningState", mining.state.value)
            sd.putBoolean("Jetson/ExcavatorEnabled", False)
            sd.putBoolean("Jetson/ConveyorEnabled", False)
            sd.putBoolean("Jetson/ExcavatorLoweringSim", False)
            sd.putBoolean("Jetson/DoorActuatorsOpen", False)
            sd.putBoolean("Jetson/DoorActuatorsClose", False)
            sd.putNumber("Jetson/ServoCommandAngleDeg", float(servo_logical_to_raw(args.camera_map_angle_deg)))
            sd.putNumber("Jetson/ServoCommandSeq", 0.0)
            print(
                f"Drive enabled: NetworkTables to {args.roborio_ip} "
                f"main_rover_mode={'on' if args.main_rover_mode else 'off'}"
            )

    goal_cell = None
    path_cells = None
    last_path_cells = None
    last_start = None
    last_goal = None
    path_plan_mode = "none"
    mining_goal_active = False
    map_window_ready = False
    status_window_ready = False
    status_button_rects = {}
    status_section_jump_targets = {}
    reset_map_confirm = False
    status_scroll_y = 0
    status_scroll_max = 0
    status_scroll_drag_active = False
    status_scroll_drag_offset = 0
    status_view_drag_active = False
    status_view_drag_anchor_y = 0
    status_view_drag_anchor_scroll = 0
    last_status_panel_shape = None
    last_map_window_shape = None
    disable_holes = bool(args.disable_holes)
    whole_map_enabled = False
    smooth_map_enabled = True
    bidirectional_auto_enabled = True
    demo_auto_enabled = False
    show_all_dig_profiles = False
    last_drive_send = 0.0
    demo_rover_pos_map = None
    demo_rover_heading_rad = 0.0
    manual_fwd = 0.0
    manual_turn = 0.0
    manual_mode = True
    no_mapping_mode = False
    if manual_mode:
        print("Manual drive mode: ON (startup)")
    if no_mapping_mode:
        print("No Mapping mode: ON (startup, camera-only path)")
        if HAS_CV2 and (not args.no_gui):
            try:
                cv2.destroyWindow("ZED Occupancy Map (XZ)")
            except Exception:
                pass
    last_w_time = 0.0
    last_s_time = 0.0
    last_a_time = 0.0
    last_d_time = 0.0
    key_hold_timeout = 0.35
    nt_last_conn_log = 0.0
    nt_last_health_log = 0.0
    nt_health_seq = 0
    nt_command_seq = 0
    nt_ready_stuck_since = 0.0
    nt_last_auto_push = 0.0
    nt_ready_high = False
    nt_ready_clear_time = 0.0
    last_drive_debug_time = 0.0
    last_backup_log_time = 0.0
    backup_hold_until = 0.0
    nt_connected_cached = False
    last_nt_ok_time = time.time()   # watchdog: last time NT was confirmed connected
    nt_watchdog_tripped = False     # set True when watchdog fires, cleared on reconnect
    ds_joystick_fwd = 0.0           # DS controller forward axis from NT
    ds_joystick_turn = 0.0          # DS controller turn axis from NT
    driver_priority_active = False
    driver_priority_suppressed_until = 0.0
    status_cmd_enabled = False
    status_cmd_fwd = 0.0
    status_cmd_turn = 0.0
    status_cmd_duration = 0.0
    status_target_cell = None
    status_target_world = None
    camera_overlay_enabled = True
    rock_overlay_detections = []
    auto_digger_enabled = False
    test_excavation_left_extend_active = False
    test_excavation_right_extend_active = False
    test_excavation_dig_active = False
    test_excavation_lower_active = False
    test_excavation_lower_cycle_started_at = 0.0
    test_door_open_active = False
    test_door_close_active = False
    test_drive_forward_active = False
    test_drive_forward_until = 0.0
    excavation_pattern_test_active = False
    excavation_pattern_test_started_at = 0.0
    dig_profile_preview_active = False
    dig_profile_preview_started_at = 0.0
    dig_profile_preview_style = None
    dig_profile_preview_phase = None
    dig_profile_preview_name = None
    controller_macro_preview_active = False
    controller_macro_preview_started_at = 0.0
    controller_macro_preview_name = None
    controller_cycle_preview_active = False
    controller_cycle_phase = "forward"
    controller_cycle_phase_started_at = 0.0
    controller_cycle_preview_name = None
    controller_cycle_mechanism_hold_sec = 3.0

    def current_controller_macro_mechanism_state():
        if sd is not None:
            try:
                return {
                    "digger_on": bool(sd.getBoolean("Jetson/ExcavatorEnabled", bool(test_excavation_dig_active))),
                    "lower_on": bool(sd.getBoolean("Jetson/ExcavatorLoweringSim", bool(test_excavation_lower_active))),
                    "left_extend_on": bool(sd.getBoolean("Jetson/ExcavatorLeftExtend", bool(test_excavation_left_extend_active))),
                    "right_extend_on": bool(sd.getBoolean("Jetson/ExcavatorRightExtend", bool(test_excavation_right_extend_active))),
                    "door_open_on": bool(sd.getBoolean("Jetson/DoorActuatorsOpen", bool(test_door_open_active))),
                    "door_close_on": bool(sd.getBoolean("Jetson/DoorActuatorsClose", bool(test_door_close_active))),
                }
            except Exception:
                pass
        return {
            "digger_on": bool(test_excavation_dig_active),
            "lower_on": bool(test_excavation_lower_active),
            "left_extend_on": bool(test_excavation_left_extend_active),
            "right_extend_on": bool(test_excavation_right_extend_active),
            "door_open_on": bool(test_door_open_active),
            "door_close_on": bool(test_door_close_active),
        }
    low_latency_mode = bool(args.camera_only)
    low_latency_restore_state = {
        "camera_overlay_enabled": True,
        "human_detect_enabled": False,
        "rock_detect_enabled": False,
        "camera_publish_enabled": True,
        "map_publish_enabled": True,
    }
    digger_speed_scale = 1.0
    actuator_left_extension_pct = None
    actuator_right_extension_pct = None
    actuator_left_extension_inches = None
    actuator_right_extension_inches = None
    actuator_sync_fault = None
    actuator_left_counts = None
    actuator_right_counts = None
    actuator_left_inches = None
    actuator_right_inches = None
    actuator_tailgate_extension_pct = None
    actuator_tailgate_inches = None
    actuator_tailgate_counts = None
    actuator_tailgate_position_calibrated = None
    actuator_tailgate_state = None
    actuator_tailgate_moving = None
    actuator_tailgate_open = None
    actuator_tailgate_closed = None
    actuator_bottom_diff_counts = None
    actuator_bottom_position_calibrated = None
    direct_nav_enabled = False
    last_path_plan_time = 0.0
    last_auto_fwd_cmd = 0.0
    last_auto_turn_cmd = 0.0
    last_auto_turn_time = time.time()
    map_integration_ok = False
    camera_map_pause_reason = ""
    last_map_point_count = 0
    last_raw_point_count = 0
    last_in_range_point_count = 0
    last_ground_pct = 0.0
    last_obstacle_pct = 0.0
    last_hole_pct = 0.0
    last_depth_status = "init"
    range_filter_bypassed = False
    last_plane_update_time = 0.0
    plane_fail_count = 0
    plane_reject_count = 0
    no_points_count = 0
    a, b, c, d = 0.0, 1.0, 0.0, 0.0
    has_plane = False
    follow_rover_map = bool(args.map_follow_rover)
    map_red_only_view = False
    map_scale_live = max(1, int(args.map_scale))
    map_view_shift_r = 0
    map_view_shift_c = 0
    frame_idx = 0
    human_person_map_points = []  # list of (row, col) for map markers
    localization_scan_active = False
    localization_scan_started = 0.0
    localization_scan_started_lost = False
    localization_scan_reason = ""
    last_localization_log = 0.0
    localization_scan_autostart_blocked_until = 0.0
    landmark_memory = {"version": 1, "landmarks": []}
    landmark_pose_override_t_map = None
    landmark_pose_override_R_world_cam = None
    landmark_dirty = False
    last_landmark_save = time.time()
    last_landmark_relocalize_log = 0.0
    map_size_input_text = ""      # user-typed map size string e.g. "6x8" (feet)
    map_size_input_focused = False
    dig_name_input_text = ""
    dig_name_input_focused = False
    paint_safe_mode = False       # when True, map clicks paint cells as permanently safe
    erase_safe_mode = False      # when True, map clicks erase painted cells
    paint_obstacle_mode = False  # when True, map clicks paint cells as obstacles
    paint_brush_radius = 2       # radius in cells for all brush tools (1–15)
    lock_green_applied = False
    lock_green_locked_count = 0
    last_map_command_seq = 0
    last_map_ui_state_write = 0.0

    def _write_json_atomic(path, payload):
        if not path:
            return
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            pass

    def _json_vec3(value):
        arr = np.array(value, dtype=np.float32).reshape(3,)
        return [float(arr[0]), float(arr[1]), float(arr[2])]

    def _parse_vec3(payload, key, default=None):
        raw = payload.get(key)
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            return None if default is None else np.array(default, dtype=np.float32).reshape(3,)
        try:
            return np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float32)
        except Exception:
            return None if default is None else np.array(default, dtype=np.float32).reshape(3,)

    def recovery_checkpoint_payload(navx_yaw_deg=None):
        payload = {
            "version": 1,
            "timestamp_ms": int(time.time() * 1000),
            "tracking_enabled": bool(tracking_enabled),
            "have_valid_tracking_pose": bool(have_valid_tracking_pose),
            "camera_t_world": _json_vec3(last_valid_t_world_cam),
            "rover_forward_world": _json_vec3(last_valid_rover_forward_world),
            "rover_right_world": _json_vec3(last_valid_rover_right_world),
            "map_origin_t": _json_vec3(map_origin_t),
            "map_origin_set": bool(map_origin_set),
            "alignment_offset_t": _json_vec3(recovery_alignment_offset_t),
            "alignment_yaw_deg": float(recovery_alignment_yaw_deg),
            "navx_yaw_deg": None if navx_yaw_deg is None or (not np.isfinite(navx_yaw_deg)) else float(navx_yaw_deg),
            "navx_sign": float(navx_sign),
            "navx_sign_locked": bool(navx_sign_locked),
            "goal_cell": None if goal_cell is None else [int(goal_cell[0]), int(goal_cell[1])],
            "drive_heading_flip": bool(args.drive_heading_flip),
            "map_save_path": str(args.map_save_path),
        }
        rover_pos_world = (
            np.array(last_valid_t_world_cam, dtype=np.float32)
            - np.array(last_valid_rover_forward_world, dtype=np.float32) * float(camera_forward_offset_m)
            - np.array(last_valid_rover_right_world, dtype=np.float32) * float(camera_right_offset_m)
        )
        payload["rover_pos_world"] = _json_vec3(rover_pos_world)
        payload["heading_deg"] = float(rover_heading_deg_from_forward(last_valid_rover_forward_world))
        return payload

    def load_recovery_checkpoint_from_nt():
        if (sd is None) or (not args.recovery_nt_mirror):
            return None
        try:
            stamp_ms = int(sd.getNumber("Jetson/RecoveryTimestampMs", 0.0))
        except Exception:
            return None
        if stamp_ms <= 0:
            return None
        try:
            pose_x = float(sd.getNumber("Jetson/RecoveryPoseX", float("nan")))
            pose_y = float(sd.getNumber("Jetson/RecoveryPoseY", float("nan")))
            pose_z = float(sd.getNumber("Jetson/RecoveryPoseZ", float("nan")))
            fwd_x = float(sd.getNumber("Jetson/RecoveryForwardX", float("nan")))
            fwd_y = float(sd.getNumber("Jetson/RecoveryForwardY", 0.0))
            fwd_z = float(sd.getNumber("Jetson/RecoveryForwardZ", float("nan")))
            right_x = float(sd.getNumber("Jetson/RecoveryRightX", float("nan")))
            right_y = float(sd.getNumber("Jetson/RecoveryRightY", 0.0))
            right_z = float(sd.getNumber("Jetson/RecoveryRightZ", float("nan")))
            map_origin_x = float(sd.getNumber("Jetson/RecoveryMapOriginX", 0.0))
            map_origin_y = float(sd.getNumber("Jetson/RecoveryMapOriginY", 0.0))
            map_origin_z = float(sd.getNumber("Jetson/RecoveryMapOriginZ", 0.0))
            align_x = float(sd.getNumber("Jetson/RecoveryAlignOffsetX", 0.0))
            align_y = float(sd.getNumber("Jetson/RecoveryAlignOffsetY", 0.0))
            align_z = float(sd.getNumber("Jetson/RecoveryAlignOffsetZ", 0.0))
            align_yaw = float(sd.getNumber("Jetson/RecoveryAlignYawDeg", 0.0))
            map_origin_valid = bool(sd.getBoolean("Jetson/RecoveryMapOriginSet", False))
            navx_saved = float(sd.getNumber("Jetson/RecoveryNavXYawDeg", float("nan")))
            navx_sign_saved = float(sd.getNumber("Jetson/RecoveryNavXSign", 1.0))
            navx_sign_locked_saved = bool(sd.getBoolean("Jetson/RecoveryNavXSignLocked", False))
            has_goal = bool(sd.getBoolean("Jetson/RecoveryHasGoal", False))
            goal_r = int(sd.getNumber("Jetson/RecoveryGoalRow", -1.0))
            goal_c = int(sd.getNumber("Jetson/RecoveryGoalCol", -1.0))
        except Exception:
            return None
        vecs = [pose_x, pose_y, pose_z, fwd_x, fwd_z, right_x, right_z]
        if not all(np.isfinite(v) for v in vecs):
            return None
        return {
            "version": 1,
            "timestamp_ms": stamp_ms,
            "camera_t_world": [pose_x, pose_y, pose_z],
            "rover_forward_world": [fwd_x, fwd_y, fwd_z],
            "rover_right_world": [right_x, right_y, right_z],
            "map_origin_t": [map_origin_x, map_origin_y, map_origin_z],
            "map_origin_set": map_origin_valid,
            "alignment_offset_t": [align_x, align_y, align_z],
            "alignment_yaw_deg": align_yaw,
            "navx_yaw_deg": navx_saved if np.isfinite(navx_saved) else None,
            "navx_sign": navx_sign_saved,
            "navx_sign_locked": navx_sign_locked_saved,
            "goal_cell": [goal_r, goal_c] if has_goal and goal_r >= 0 and goal_c >= 0 else None,
        }

    def load_recovery_checkpoint():
        best = None
        if args.recovery_load and args.recovery_checkpoint_path and os.path.exists(args.recovery_checkpoint_path):
            try:
                with open(args.recovery_checkpoint_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    best = data
            except Exception as exc:
                print(f"Failed to load recovery checkpoint: {exc}")
        nt_data = load_recovery_checkpoint_from_nt()
        if nt_data is not None:
            nt_stamp = int(nt_data.get("timestamp_ms", 0) or 0)
            file_stamp = int(best.get("timestamp_ms", 0) or 0) if isinstance(best, dict) else 0
            if nt_stamp > file_stamp:
                best = nt_data
        return best if isinstance(best, dict) else None

    def publish_recovery_checkpoint_to_nt(payload):
        if (sd is None) or (not args.recovery_nt_mirror) or (not isinstance(payload, dict)):
            return
        try:
            cam_t = payload.get("camera_t_world") or [0.0, 0.0, 0.0]
            rover_fwd = payload.get("rover_forward_world") or [1.0, 0.0, 0.0]
            rover_right = payload.get("rover_right_world") or [0.0, 0.0, -1.0]
            map_origin_vec = payload.get("map_origin_t") or [0.0, 0.0, 0.0]
            align_vec = payload.get("alignment_offset_t") or [0.0, 0.0, 0.0]
            goal = payload.get("goal_cell")
            sd.putNumber("Jetson/RecoveryTimestampMs", float(payload.get("timestamp_ms", 0)))
            sd.putNumber("Jetson/RecoveryPoseX", float(cam_t[0]))
            sd.putNumber("Jetson/RecoveryPoseY", float(cam_t[1]))
            sd.putNumber("Jetson/RecoveryPoseZ", float(cam_t[2]))
            sd.putNumber("Jetson/RecoveryForwardX", float(rover_fwd[0]))
            sd.putNumber("Jetson/RecoveryForwardY", float(rover_fwd[1]))
            sd.putNumber("Jetson/RecoveryForwardZ", float(rover_fwd[2]))
            sd.putNumber("Jetson/RecoveryRightX", float(rover_right[0]))
            sd.putNumber("Jetson/RecoveryRightY", float(rover_right[1]))
            sd.putNumber("Jetson/RecoveryRightZ", float(rover_right[2]))
            sd.putNumber("Jetson/RecoveryMapOriginX", float(map_origin_vec[0]))
            sd.putNumber("Jetson/RecoveryMapOriginY", float(map_origin_vec[1]))
            sd.putNumber("Jetson/RecoveryMapOriginZ", float(map_origin_vec[2]))
            sd.putBoolean("Jetson/RecoveryMapOriginSet", bool(payload.get("map_origin_set", False)))
            sd.putNumber("Jetson/RecoveryAlignOffsetX", float(align_vec[0]))
            sd.putNumber("Jetson/RecoveryAlignOffsetY", float(align_vec[1]))
            sd.putNumber("Jetson/RecoveryAlignOffsetZ", float(align_vec[2]))
            sd.putNumber("Jetson/RecoveryAlignYawDeg", float(payload.get("alignment_yaw_deg", 0.0)))
            navx_saved = payload.get("navx_yaw_deg")
            sd.putNumber("Jetson/RecoveryNavXYawDeg", 0.0 if navx_saved is None else float(navx_saved))
            sd.putBoolean("Jetson/RecoveryHasNavX", navx_saved is not None)
            sd.putNumber("Jetson/RecoveryNavXSign", float(payload.get("navx_sign", 1.0)))
            sd.putBoolean("Jetson/RecoveryNavXSignLocked", bool(payload.get("navx_sign_locked", False)))
            sd.putBoolean("Jetson/RecoveryHasGoal", bool(goal is not None))
            sd.putNumber("Jetson/RecoveryGoalRow", -1.0 if goal is None else float(goal[0]))
            sd.putNumber("Jetson/RecoveryGoalCol", -1.0 if goal is None else float(goal[1]))
            sd.putNumber("Jetson/RecoveryHeadingDeg", float(payload.get("heading_deg", 0.0)))
        except Exception:
            pass

    recovery_checkpoint = load_recovery_checkpoint()
    if recovery_checkpoint is not None:
        ckpt_cam_t = _parse_vec3(recovery_checkpoint, "camera_t_world")
        ckpt_rover_fwd = _parse_vec3(recovery_checkpoint, "rover_forward_world")
        ckpt_rover_right = _parse_vec3(recovery_checkpoint, "rover_right_world")
        ckpt_map_origin = _parse_vec3(recovery_checkpoint, "map_origin_t", default=np.zeros(3, dtype=np.float32))
        ckpt_align_t = _parse_vec3(recovery_checkpoint, "alignment_offset_t", default=np.zeros(3, dtype=np.float32))
        if ckpt_cam_t is not None and ckpt_rover_fwd is not None and ckpt_rover_right is not None:
            last_valid_t_world_cam = np.array(ckpt_cam_t, dtype=np.float32).reshape(3,)
            last_valid_rover_forward_world = rotate_world_xz(ckpt_rover_fwd, 0.0)
            last_valid_rover_right_world = rotate_world_xz(ckpt_rover_right, 0.0)
            ckpt_cam_forward = camera_forward_from_rover_axes(
                last_valid_rover_forward_world,
                last_valid_rover_right_world,
                current_camera_mount_yaw_deg(),
            )
            ckpt_cam_right = camera_right_from_rover_axes(
                last_valid_rover_forward_world,
                last_valid_rover_right_world,
                current_camera_mount_yaw_deg(),
            )
            ckpt_cam_up = np.cross(ckpt_cam_forward, ckpt_cam_right)
            up_norm = float(np.linalg.norm(ckpt_cam_up))
            if up_norm > 1e-6:
                ckpt_cam_up = (ckpt_cam_up / up_norm).astype(np.float32)
            else:
                ckpt_cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            last_valid_R_world_cam = np.column_stack((ckpt_cam_right, ckpt_cam_up, ckpt_cam_forward)).astype(np.float32)
            recovery_alignment_offset_t = np.array(ckpt_align_t, dtype=np.float32).reshape(3,)
            recovery_alignment_yaw_deg = float(recovery_checkpoint.get("alignment_yaw_deg", 0.0) or 0.0)
            map_origin_t = np.array(ckpt_map_origin, dtype=np.float32).reshape(3,)
            map_origin_set = bool(recovery_checkpoint.get("map_origin_set", False))
            last_valid_navx_yaw_deg = recovery_checkpoint.get("navx_yaw_deg")
            navx_sign = float(recovery_checkpoint.get("navx_sign", navx_sign) or navx_sign)
            navx_sign_locked = bool(recovery_checkpoint.get("navx_sign_locked", navx_sign_locked))
            have_valid_tracking_pose = bool(recovery_checkpoint.get("have_valid_tracking_pose", True))
            recovery_pending_alignment = bool(tracking_enabled and have_valid_tracking_pose)
            recovery_loaded_from_checkpoint = True
            goal_payload = recovery_checkpoint.get("goal_cell")
            if isinstance(goal_payload, (list, tuple)) and len(goal_payload) == 2:
                try:
                    goal_candidate = (int(goal_payload[0]), int(goal_payload[1]))
                except Exception:
                    goal_candidate = None
                if goal_candidate is not None:
                    if 0 <= goal_candidate[0] < occ_map.grid_h and 0 <= goal_candidate[1] < occ_map.grid_w:
                        goal_cell = goal_candidate
            print(
                "Loaded recovery checkpoint: "
                f"pose=({last_valid_t_world_cam[0]:+.2f}, {last_valid_t_world_cam[2]:+.2f}) "
                f"heading={rover_heading_deg_from_forward(last_valid_rover_forward_world):+.1f}deg"
                + (" with saved goal." if goal_cell is not None else ".")
            )
            publish_recovery_checkpoint_to_nt(recovery_checkpoint)

    def save_recovery_checkpoint(force=False, navx_yaw_deg=None):
        nonlocal recovery_checkpoint, last_recovery_save
        if float(args.recovery_save_every) <= 0.0:
            return
        if not have_valid_tracking_pose:
            return
        now = time.time()
        if (not force) and (now - last_recovery_save) < float(args.recovery_save_every):
            return
        payload = recovery_checkpoint_payload(navx_yaw_deg=navx_yaw_deg)
        _write_json_atomic(args.recovery_checkpoint_path, payload)
        publish_recovery_checkpoint_to_nt(payload)
        recovery_checkpoint = payload
        last_recovery_save = now

    def load_landmark_memory():
        nonlocal landmark_memory
        if not args.landmark_memory or not args.landmark_path:
            return
        try:
            if os.path.exists(args.landmark_path):
                with open(args.landmark_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and isinstance(data.get("landmarks"), list):
                    landmark_memory = data
                    print(f"Loaded {len(landmark_memory['landmarks'])} AI landmark(s): {args.landmark_path}")
        except Exception as exc:
            print(f"Failed to load landmark memory: {exc}")

    def save_landmark_memory(force=False):
        nonlocal landmark_dirty, last_landmark_save
        if not args.landmark_memory or not args.landmark_path:
            return
        now = time.time()
        if (not force) and ((not landmark_dirty) or (now - last_landmark_save) < float(args.landmark_save_every)):
            return
        landmark_memory["version"] = 1
        landmark_memory["updated_ms"] = int(now * 1000)
        _write_json_atomic(args.landmark_path, landmark_memory)
        landmark_dirty = False
        last_landmark_save = now

    def record_static_landmark(label, map_x, map_z, confidence):
        """
        Save stable AI detections in map coordinates. People are intentionally
        excluded; this is for static objects like rocks/obstacles.
        """
        nonlocal landmark_dirty
        if not args.landmark_memory:
            return
        if not np.isfinite([map_x, map_z]).all():
            return
        label = str(label or "object").strip().lower()
        now_ms = int(time.time() * 1000)
        assoc_m = max(0.05, float(args.landmark_assoc_m))
        best = None
        best_dist = None
        for item in landmark_memory.get("landmarks", []):
            if item.get("label") != label:
                continue
            dist = math.hypot(float(item.get("x", 0.0)) - map_x, float(item.get("z", 0.0)) - map_z)
            if dist <= assoc_m and (best_dist is None or dist < best_dist):
                best = item
                best_dist = dist
        if best is None:
            best = {
                "id": f"{label}_{len(landmark_memory.get('landmarks', [])) + 1}",
                "label": label,
                "x": float(map_x),
                "z": float(map_z),
                "confidence": float(confidence),
                "hits": 1,
                "first_seen_ms": now_ms,
                "last_seen_ms": now_ms,
            }
            landmark_memory.setdefault("landmarks", []).append(best)
        else:
            hits = int(best.get("hits", 1)) + 1
            alpha = 1.0 / min(12.0, float(hits))
            best["x"] = float((1.0 - alpha) * float(best.get("x", map_x)) + alpha * map_x)
            best["z"] = float((1.0 - alpha) * float(best.get("z", map_z)) + alpha * map_z)
            best["confidence"] = float(max(float(best.get("confidence", 0.0)), float(confidence)))
            best["hits"] = hits
            best["last_seen_ms"] = now_ms
        landmark_dirty = True

    def iter_visible_landmarks():
        min_hits = max(1, int(args.landmark_min_hits))
        for item in landmark_memory.get("landmarks", []):
            if int(item.get("hits", 0)) < min_hits:
                continue
            rc = occ_map.world_to_grid(float(item.get("x", 0.0)), float(item.get("z", 0.0)))
            if rc is None:
                continue
            yield item, rc

    def draw_landmarks(frame):
        if not HAS_CV2 or frame is None:
            return
        for item, (row, col) in iter_visible_landmarks():
            display_cell = display_cell_for_map_cell(row, col, frame)
            if display_cell is None:
                continue
            dr, dc = display_cell
            color = (255, 0, 180)
            cv2.drawMarker(frame, (dc, dr), color, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
            label = str(item.get("label", "obj"))[:10]
            cv2.putText(
                frame,
                label,
                (min(frame.shape[1] - 1, dc + 5), max(10, dr - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                color,
                1,
                cv2.LINE_AA,
            )

    def try_landmark_relocalization(label, point_cam, base_t_map, fallback_forward_world=None):
        nonlocal landmark_pose_override_t_map, landmark_pose_override_R_world_cam, last_landmark_relocalize_log
        if (not args.landmark_memory) or (not args.landmark_relocalize):
            return False
        if label not in landmark_class_names:
            return False
        if point_cam is None or base_t_map is None:
            return False
        candidates = []
        min_hits = max(1, int(args.landmark_min_hits))
        for item in landmark_memory.get("landmarks", []):
            if str(item.get("label", "")).strip().lower() != label:
                continue
            if int(item.get("hits", 0)) < min_hits:
                continue
            candidates.append(item)
        if not candidates:
            return False

        est_R_world_cam = camera_rotation_from_forward_world(fallback_forward_world)
        if est_R_world_cam is None:
            est_R_world_cam = np.array(last_valid_R_world_cam, dtype=np.float32).reshape(3, 3)
        rel_world = (est_R_world_cam @ np.array(point_cam, dtype=np.float32).reshape(3, 1)).reshape(3,)

        base_t = np.array(base_t_map, dtype=np.float32).reshape(3,)
        max_offset_m = max(0.20, float(args.landmark_relocalize_max_offset_m))
        best_item = None
        best_candidate_t = None
        best_offset_m = None
        for item in candidates:
            candidate_t = np.array(base_t, dtype=np.float32)
            candidate_t[0] = -float(item.get("x", 0.0)) - float(rel_world[0])
            candidate_t[2] = float(item.get("z", 0.0)) - float(rel_world[2])
            offset_m = math.hypot(float(candidate_t[0] - base_t[0]), float(candidate_t[2] - base_t[2]))
            if offset_m > max_offset_m:
                continue
            if (
                best_offset_m is None
                or offset_m < best_offset_m
                or (
                    abs(offset_m - best_offset_m) <= 1e-6
                    and int(item.get("hits", 0)) > int(best_item.get("hits", 0))
                )
            ):
                best_item = item
                best_candidate_t = candidate_t
                best_offset_m = offset_m

        if best_item is None or best_candidate_t is None:
            return False

        alpha = max(0.0, min(1.0, float(args.landmark_relocalize_alpha)))
        corrected_t = np.array(base_t, dtype=np.float32)
        corrected_t[0] = float((1.0 - alpha) * base_t[0] + alpha * best_candidate_t[0])
        corrected_t[2] = float((1.0 - alpha) * base_t[2] + alpha * best_candidate_t[2])
        landmark_pose_override_t_map = corrected_t
        landmark_pose_override_R_world_cam = est_R_world_cam.astype(np.float32)
        now = time.time()
        if (now - last_landmark_relocalize_log) >= 0.75:
            print(
                "Landmark relocalization: "
                f"{label} -> {best_item.get('id', label)} "
                f"offset={best_offset_m:.2f}m "
                f"map=({map_x_from_zed(corrected_t[0]):+.2f}, {corrected_t[2]:+.2f})"
            )
            last_landmark_relocalize_log = now
        return True

    def start_localization_scan(reason="manual"):
        nonlocal localization_scan_active, localization_scan_started
        nonlocal localization_scan_started_lost, localization_scan_reason
        localization_scan_active = False
        localization_scan_started = 0.0
        localization_scan_started_lost = False
        localization_scan_reason = ""
        print(f"Localization scan is disabled; ignoring request ({reason}).")

    def stop_localization_scan(reason="done"):
        nonlocal localization_scan_active, localization_scan_reason
        localization_scan_active = False
        localization_scan_reason = ""

    def block_localization_autostart(seconds=5.0, source="manual"):
        nonlocal localization_scan_autostart_blocked_until
        seconds = max(0.0, float(seconds))
        if seconds <= 0.0:
            return
        localization_scan_autostart_blocked_until = max(
            localization_scan_autostart_blocked_until,
            time.time() + seconds,
        )
        print(f"Localize auto-start blocked for {seconds:.1f}s ({source}).")

    def set_manual_drive_mode(enabled, source="key"):
        nonlocal manual_mode, manual_fwd, manual_turn, emergency_stop
        enabled = bool(enabled)
        if localization_scan_active:
            block_localization_autostart(5.0, f"{source} manual override")
            stop_localization_scan(f"{source} manual override")
        manual_mode = enabled
        manual_fwd = 0.0
        manual_turn = 0.0
        emergency_stop = False
        if manual_mode:
            print("Manual drive mode: ON (localization canceled, auto paused)")
        else:
            print("Manual drive mode: OFF (auto resumed)")

    def set_no_mapping_mode(enabled, source="button"):
        nonlocal no_mapping_mode, map_window_ready
        enabled = bool(enabled)
        mining_running_now = mining.state in (
            auto_mining.MiningState.PLAN_SWEEP,
            auto_mining.MiningState.NAVIGATE_DIG,
            auto_mining.MiningState.DIGGING,
            auto_mining.MiningState.BACKUP,
            auto_mining.MiningState.NAVIGATE_DEPOSIT,
            auto_mining.MiningState.DEPOSITING,
        )
        if enabled and mining_running_now:
            print("No Mapping mode unavailable while Auto Run is active.")
            return
        if no_mapping_mode == enabled:
            return
        no_mapping_mode = enabled
        if not args.no_gui and HAS_CV2:
            try:
                if enabled:
                    cv2.destroyWindow("ZED Occupancy Map (XZ)")
                    map_window_ready = False
                else:
                    cv2.namedWindow("ZED Occupancy Map (XZ)", fixed_flags)
                    map_window_ready = False
            except Exception:
                pass
        state = "ON" if no_mapping_mode else "OFF"
        print(f"No Mapping mode: {state} via {source}.")
        publish_map_ui_state(force=True)

    def suppress_driver_priority(seconds, source="auto"):
        nonlocal driver_priority_suppressed_until, driver_priority_active
        seconds = max(0.0, float(seconds))
        if seconds <= 0.0:
            return
        driver_priority_suppressed_until = max(
            float(driver_priority_suppressed_until),
            time.time() + seconds,
        )
        driver_priority_active = False
        print(f"Driver-priority override suppressed for {seconds:.1f}s ({source}).")

    def update_localization_scan_state():
        return

    def draw_localization_banner(frame):
        if frame is None:
            return
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return

        locked = bool(start_frame_locked_once)
        waiting = bool(start_frame_auto_lock_enabled and (not start_frame_locked_once))
        if locked:
            headline = "START FRAME: LOCKED"
            color = (60, 215, 80)
        elif waiting:
            headline = "START FRAME: SEARCHING TAGS"
            color = (0, 215, 255)
        else:
            headline = "START FRAME: IDLE"
            color = (180, 180, 180)

        detail = str(start_frame_last_status or "").strip()
        if len(detail) > 84:
            detail = detail[:81] + "..."
        if start_frame_last_ids:
            id_text = ",".join(str(v) for v in start_frame_last_ids)
            detail = f"{detail}  IDs:{id_text}".strip()
        if start_frame_last_error_m is not None:
            detail = f"{detail}  fit:{float(start_frame_last_error_m):.03f}m".strip()

        pad = 10
        box_h = 54
        box_w = min(w - 2 * pad, max(340, int(0.56 * w)))
        x0 = pad
        y0 = pad
        x1 = x0 + box_w
        y1 = y0 + box_h
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            frame,
            headline,
            (x0 + 12, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
        if detail:
            cv2.putText(
                frame,
                detail,
                (x0 + 12, y0 + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

    load_landmark_memory()

    def mining_buttons_enabled():
        return mining.state not in (
            auto_mining.MiningState.PLAN_SWEEP,
            auto_mining.MiningState.NAVIGATE_DIG,
            auto_mining.MiningState.DIGGING,
            auto_mining.MiningState.BACKUP,
            auto_mining.MiningState.NAVIGATE_DEPOSIT,
            auto_mining.MiningState.DEPOSITING,
            auto_mining.MiningState.DRAW_EXCAV,
            auto_mining.MiningState.DRAW_DEPOSIT,
            auto_mining.MiningState.PICK_DIG_START,
        )

    def set_brush_tool(tool_name):
        nonlocal paint_safe_mode, erase_safe_mode, paint_obstacle_mode
        paint_safe_mode = tool_name == "paint_safe"
        erase_safe_mode = tool_name == "erase_safe"
        paint_obstacle_mode = tool_name == "paint_obstacle"

    def current_selected_tool():
        if dig_profiles.recording:
            return f"dig_record_{dig_profiles.recording_style}_{dig_profiles.recording_phase}"
        if drive_calibration.active:
            return "drive_calibration_mode"
        if paint_safe_mode:
            return "paint_safe"
        if erase_safe_mode:
            return "erase_safe"
        if paint_obstacle_mode:
            return "paint_obstacle"
        if mining.state == auto_mining.MiningState.DRAW_EXCAV:
            return "draw_excav_zone"
        if mining.state == auto_mining.MiningState.DRAW_DEPOSIT:
            return "draw_deposit_zone"
        if mining.state == auto_mining.MiningState.PICK_DIG_START:
            return "pick_dig_start"
        if mining.deposit_zone_preset_side == "left":
            return "set_berm_left"
        if mining.deposit_zone_preset_side == "right":
            return "set_berm_right"
        return None

    def clear_manual_paint():
        nonlocal lock_green_applied, lock_green_locked_count
        occ_map.painted_free[:] = False
        lock_green_applied = False
        lock_green_locked_count = 0
        print("Cleared all painted-safe cells")

    def set_main_rover_mode(enabled):
        args.main_rover_mode = bool(enabled)
        if sd is not None:
            sd.putBoolean("Drive/UseMainRoverControls", bool(args.main_rover_mode))
            sd.putBoolean("Drive/MainRoverDebugMode", bool(args.main_rover_debug))
            sd.putBoolean("Drive/MainRoverHardFlip", bool(args.hard_drive_flip))
        print(f"Main rover drive mode: {'ON' if args.main_rover_mode else 'OFF'}")

    def set_excavation_test_mode(mode_name, enabled, source="button"):
        nonlocal auto_digger_enabled
        nonlocal test_excavation_left_extend_active, test_excavation_right_extend_active
        nonlocal test_excavation_dig_active, test_excavation_lower_active
        nonlocal test_excavation_lower_cycle_started_at
        nonlocal test_door_open_active, test_door_close_active
        nonlocal excavation_pattern_test_active, excavation_pattern_test_started_at
        enabled = bool(enabled)
        if enabled and excavation_pattern_test_active:
            excavation_pattern_test_active = False
            excavation_pattern_test_started_at = 0.0
        if mode_name == "auto_digger":
            if auto_digger_enabled == enabled:
                return
            auto_digger_enabled = enabled
            print(f"Auto dig {'ENABLED' if enabled else 'DISABLED'} via {source}.")
        elif mode_name == "left_extend":
            if test_excavation_left_extend_active == enabled:
                return
            test_excavation_left_extend_active = enabled
            print(f"Left actuator extend {'ON' if enabled else 'OFF'} via {source}.")
        elif mode_name == "right_extend":
            if test_excavation_right_extend_active == enabled:
                return
            test_excavation_right_extend_active = enabled
            print(f"Right actuator extend {'ON' if enabled else 'OFF'} via {source}.")
        elif mode_name == "dig":
            if test_excavation_dig_active == enabled:
                return
            test_excavation_dig_active = enabled
            print(f"Excavation test digger {'ON' if enabled else 'OFF'} via {source}.")
        elif mode_name == "lower":
            if test_excavation_lower_active == enabled:
                return
            test_excavation_lower_active = enabled
            if enabled:
                test_excavation_lower_cycle_started_at = time.time()
                print(f"Excavation lower cycle STARTED via {source} (5s down, 5s up).")
            else:
                test_excavation_lower_cycle_started_at = 0.0
                print(f"Excavation lower cycle STOPPED via {source}.")
        elif mode_name == "door_open":
            if enabled:
                test_door_close_active = False
            if test_door_open_active == enabled:
                return
            test_door_open_active = enabled
            print(f"Door actuators open {'ON' if enabled else 'OFF'} via {source}.")
        elif mode_name == "door_close":
            if enabled:
                test_door_open_active = False
            if test_door_close_active == enabled:
                return
            test_door_close_active = enabled
            print(f"Door actuators close {'ON' if enabled else 'OFF'} via {source}.")
        else:
            return
        publish_map_ui_state(force=True)

    def stop_all_actuators(source="button"):
        nonlocal test_excavation_left_extend_active, test_excavation_right_extend_active
        nonlocal test_excavation_dig_active, test_excavation_lower_active
        nonlocal test_excavation_lower_cycle_started_at
        nonlocal test_door_open_active, test_door_close_active
        nonlocal excavation_pattern_test_active, excavation_pattern_test_started_at
        changed = (
            test_excavation_left_extend_active
            or test_excavation_right_extend_active
            or test_excavation_dig_active
            or test_excavation_lower_active
            or test_door_open_active
            or test_door_close_active
            or excavation_pattern_test_active
        )
        test_excavation_left_extend_active = False
        test_excavation_right_extend_active = False
        test_excavation_dig_active = False
        test_excavation_lower_active = False
        test_excavation_lower_cycle_started_at = 0.0
        test_door_open_active = False
        test_door_close_active = False
        excavation_pattern_test_active = False
        excavation_pattern_test_started_at = 0.0
        if changed:
            print(f"Actuator manual commands stopped via {source}.")
            if args.drive and sd is not None:
                reset_auto_drive_shape(time.time())
                send_nt_command(False, 0.0, 0.0, 0.1)
            publish_map_ui_state(force=True)

    def set_test_drive_forward(enabled, source="button"):
        nonlocal test_drive_forward_active, test_drive_forward_until
        nonlocal excavation_pattern_test_active, excavation_pattern_test_started_at
        if not args.drive or sd is None:
            print("Forward drive test unavailable because RoboRIO drive is not active.")
            return
        enabled = bool(enabled)
        if enabled:
            excavation_pattern_test_active = False
            excavation_pattern_test_started_at = 0.0
            test_drive_forward_active = True
            test_drive_forward_until = time.time() + 5.0
            reset_auto_drive_shape(time.time())
            print(f"Forward drive test STARTED via {source} (5s).")
        else:
            if not test_drive_forward_active:
                return
            test_drive_forward_active = False
            test_drive_forward_until = 0.0
            reset_auto_drive_shape(time.time())
            send_nt_command(False, 0.0, 0.0, 0.1)
            print(f"Forward drive test STOPPED via {source}.")
        publish_map_ui_state(force=True)

    def excavation_pattern_state(now):
        if not excavation_pattern_test_active or excavation_pattern_test_started_at <= 0.0:
            return None
        forward_sec = 5.0
        reverse_sec = 5.0
        pull_up_sec = 2.0
        cycles = 4
        final_retract_sec = max(0.0, (forward_sec - pull_up_sec) * cycles)
        total_cycle_sec = forward_sec + reverse_sec
        total_sec = total_cycle_sec * cycles + final_retract_sec
        elapsed = max(0.0, float(now) - float(excavation_pattern_test_started_at))
        slow_speed = max(0.10, min(1.0, float(args.drive_speed))) * 0.28
        if elapsed >= total_sec:
            return {
                "done": True,
                "label": "complete",
                "fwd": 0.0,
                "lower": False,
                "digger": False,
                "cycle_index": cycles,
                "cycles": cycles,
            }
        active_window = total_cycle_sec * cycles
        if elapsed < active_window:
            cycle_index = int(elapsed // total_cycle_sec)
            cycle_elapsed = elapsed - float(cycle_index * total_cycle_sec)
            if cycle_elapsed < forward_sec:
                return {
                    "done": False,
                    "label": f"Cycle {cycle_index + 1}/{cycles}: lower + forward",
                    "fwd": slow_speed,
                    "lower": True,
                    "digger": True,
                    "cycle_index": cycle_index + 1,
                    "cycles": cycles,
                }
            reverse_elapsed = cycle_elapsed - forward_sec
            if reverse_elapsed < pull_up_sec:
                label = f"Cycle {cycle_index + 1}/{cycles}: pull up + reverse"
            else:
                label = f"Cycle {cycle_index + 1}/{cycles}: reverse"
            return {
                "done": False,
                "label": label,
                "fwd": -slow_speed,
                "lower": False,
                "digger": True,
                "cycle_index": cycle_index + 1,
                "cycles": cycles,
            }
        retract_elapsed = elapsed - active_window
        return {
            "done": False,
            "label": f"Final retract {retract_elapsed:.1f}/{final_retract_sec:.1f}s",
            "fwd": 0.0,
            "lower": False,
            "digger": False,
            "cycle_index": cycles,
            "cycles": cycles,
        }

    def set_excavation_pattern_test(enabled, source="button"):
        nonlocal excavation_pattern_test_active, excavation_pattern_test_started_at
        nonlocal test_drive_forward_active, test_drive_forward_until
        nonlocal test_excavation_left_extend_active, test_excavation_right_extend_active
        nonlocal test_excavation_dig_active, test_excavation_lower_active
        nonlocal test_excavation_lower_cycle_started_at
        nonlocal test_door_open_active, test_door_close_active
        if not args.drive or sd is None:
            print("Excavation pattern test unavailable because RoboRIO drive is not active.")
            return
        enabled = bool(enabled)
        if enabled:
            test_drive_forward_active = False
            test_drive_forward_until = 0.0
            test_excavation_left_extend_active = False
            test_excavation_right_extend_active = False
            test_excavation_dig_active = False
            test_excavation_lower_active = False
            test_excavation_lower_cycle_started_at = 0.0
            test_door_open_active = False
            test_door_close_active = False
            excavation_pattern_test_active = True
            excavation_pattern_test_started_at = time.time()
            reset_auto_drive_shape(excavation_pattern_test_started_at)
            print(
                f"Excavation pattern test STARTED via {source} "
                "(4 cycles: 5s lower+forward, 5s reverse, final retract 12s)."
            )
        else:
            if not excavation_pattern_test_active:
                return
            excavation_pattern_test_active = False
            excavation_pattern_test_started_at = 0.0
            reset_auto_drive_shape(time.time())
            send_nt_command(False, 0.0, 0.0, 0.1)
            print(f"Excavation pattern test STOPPED via {source}.")
        publish_map_ui_state(force=True)

    def _read_first_nt_number(keys):
        if sd is None:
            return None
        for key in keys:
            try:
                value = float(sd.getNumber(key, float("nan")))
            except Exception:
                continue
            if np.isfinite(value):
                return value
        return None

    def _read_first_nt_boolean(keys):
        if sd is None:
            return None
        for key in keys:
            try:
                if hasattr(sd, "containsKey") and not sd.containsKey(key):
                    continue
                return bool(sd.getBoolean(key, False))
            except Exception:
                continue
        return None

    def refresh_actuator_feedback():
        nonlocal actuator_left_extension_pct, actuator_right_extension_pct
        nonlocal actuator_left_extension_inches, actuator_right_extension_inches
        nonlocal actuator_bottom_position_calibrated, actuator_sync_fault
        nonlocal actuator_left_counts, actuator_right_counts
        nonlocal actuator_left_inches, actuator_right_inches
        nonlocal actuator_tailgate_extension_pct, actuator_tailgate_inches
        nonlocal actuator_tailgate_counts, actuator_tailgate_position_calibrated
        nonlocal actuator_tailgate_state, actuator_tailgate_moving
        nonlocal actuator_tailgate_open, actuator_tailgate_closed
        nonlocal actuator_bottom_diff_counts, actuator_bottom_position_calibrated
        left_pct = _read_first_nt_number(
            (
                "Excav/BotLeftExtensionPct",
                "Jetson/ExcavatorLeftExtensionPct",
                "Jetson/LeftActuatorExtensionPct",
                "Excavator/LeftExtensionPct",
            )
        )
        right_pct = _read_first_nt_number(
            (
                "Excav/BotRightExtensionPct",
                "Jetson/ExcavatorRightExtensionPct",
                "Jetson/RightActuatorExtensionPct",
                "Excavator/RightExtensionPct",
            )
        )
        left_inches = _read_first_nt_number(
            (
                "Excav/BotLeftInches",
                "Jetson/ExcavatorLeftExtensionInches",
                "Jetson/ExcavatorLeftInches",
                "Jetson/LeftActuatorInches",
            )
        )
        right_inches = _read_first_nt_number(
            (
                "Excav/BotRightInches",
                "Jetson/ExcavatorRightExtensionInches",
                "Jetson/ExcavatorRightInches",
                "Jetson/RightActuatorInches",
            )
        )
        actuator_left_counts = _read_first_nt_number(
            (
                "Excav/BotLeftCounts",
                "Jetson/ExcavatorLeftCounts",
                "Jetson/LeftActuatorCounts",
            )
        )
        actuator_right_counts = _read_first_nt_number(
            (
                "Excav/BotRightCounts",
                "Jetson/ExcavatorRightCounts",
                "Jetson/RightActuatorCounts",
            )
        )
        actuator_left_inches = left_inches
        actuator_right_inches = right_inches
        actuator_bottom_diff_counts = _read_first_nt_number(
            (
                "Excav/BottomDiffCounts",
                "Jetson/ExcavatorBottomDiffCounts",
            )
        )
        actuator_left_extension_pct = None if left_pct is None else max(0.0, min(100.0, left_pct))
        actuator_right_extension_pct = None if right_pct is None else max(0.0, min(100.0, right_pct))
        actuator_left_extension_inches = left_inches
        actuator_right_extension_inches = right_inches
        actuator_bottom_position_calibrated = _read_first_nt_boolean(
            (
                "Excav/BottomPositionCalibrated",
                "Jetson/ExcavatorBottomPositionCalibrated",
            )
        )
        actuator_sync_fault = _read_first_nt_boolean(("Excav/SyncFault",))
        tailgate_pct = _read_first_nt_number(
            (
                "Deposit/TailgateExtensionPct",
                "Tailgate/ExtensionPct",
                "Jetson/TailgateExtensionPct",
                "Jetson/GateActuatorExtensionPct",
                "GateActuator/ExtensionPct",
            )
        )
        actuator_tailgate_extension_pct = None if tailgate_pct is None else max(0.0, min(100.0, tailgate_pct))
        actuator_tailgate_inches = _read_first_nt_number(
            (
                "Deposit/TailgateInches",
                "Tailgate/Inches",
                "Jetson/TailgateInches",
                "Jetson/GateActuatorInches",
                "GateActuator/Inches",
            )
        )
        actuator_tailgate_counts = _read_first_nt_number(
            (
                "Deposit/TailgateCounts",
                "Tailgate/Counts",
                "Jetson/TailgateCounts",
                "Jetson/GateActuatorCounts",
                "GateActuator/Counts",
            )
        )
        actuator_tailgate_position_calibrated = _read_first_nt_boolean(
            (
                "Deposit/TailgatePositionCalibrated",
                "Tailgate/PositionCalibrated",
                "Jetson/TailgatePositionCalibrated",
                "Jetson/GateActuatorPositionCalibrated",
                "GateActuator/PositionCalibrated",
            )
        )
        actuator_tailgate_moving = _read_first_nt_boolean(
            (
                "Deposit/TailgateMoving",
                "Tailgate/Moving",
                "Jetson/TailgateMoving",
                "GateActuator/Moving",
            )
        )
        actuator_tailgate_open = _read_first_nt_boolean(
            (
                "Deposit/TailgateOpen",
                "Tailgate/Open",
                "Jetson/TailgateOpen",
                "GateActuator/Open",
            )
        )
        actuator_tailgate_closed = _read_first_nt_boolean(
            (
                "Deposit/TailgateClosed",
                "Tailgate/Closed",
                "Jetson/TailgateClosed",
                "GateActuator/Closed",
            )
        )
        actuator_tailgate_state = None
        if sd is not None:
            for key in (
                "Deposit/TailgateState",
                "Tailgate/State",
                "Jetson/TailgateState",
                "Jetson/GateActuatorState",
                "GateActuator/State",
            ):
                try:
                    if hasattr(sd, "containsKey") and not sd.containsKey(key):
                        continue
                    text = str(sd.getString(key, "")).strip()
                except Exception:
                    continue
                if text:
                    actuator_tailgate_state = text
                    break

    def refresh_camera_servo_state():
        nonlocal servo_angle_deg, servo_target_angle_deg, servo_command_angle_deg
        nonlocal servo_settled, servo_turning, servo_map_view, servo_deposit_view
        nonlocal servo_manual_override
        if (not args.camera_servo_track) or (sd is None):
            servo_angle_deg = float(camera_mount_yaw_deg)
            servo_target_angle_deg = float(camera_mount_yaw_deg)
            servo_command_angle_deg = float(camera_mount_yaw_deg)
            servo_settled = True
            servo_turning = False
            servo_map_view = True
            servo_deposit_view = False
            servo_manual_override = False
            return

        raw_servo_angle_deg = float(sd.getNumber("Jetson/ServoAngleDeg", servo_logical_to_raw(servo_angle_deg)))
        raw_servo_target_angle_deg = float(sd.getNumber("Jetson/ServoTargetAngleDeg", servo_logical_to_raw(servo_target_angle_deg)))
        raw_servo_command_angle_deg = float(sd.getNumber("Jetson/ServoCommandAngleDeg", servo_logical_to_raw(servo_command_angle_deg)))
        servo_angle_deg = float(servo_raw_to_logical(raw_servo_angle_deg))
        servo_target_angle_deg = float(servo_raw_to_logical(raw_servo_target_angle_deg))
        servo_command_angle_deg = float(servo_raw_to_logical(raw_servo_command_angle_deg))
        servo_settled = bool(sd.getBoolean("Jetson/ServoSettled", True))
        servo_manual_override = bool(sd.getBoolean("Jetson/ServoManualOverride", False))
        servo_manual_moving = bool(sd.getBoolean("Jetson/ServoManualMoving", False))
        servo_turning = (not servo_settled) or servo_manual_moving
        servo_map_view = abs(angle_error_deg(servo_angle_deg, args.camera_map_angle_deg)) <= float(args.camera_servo_map_tol_deg)
        servo_deposit_view = abs(angle_error_deg(servo_angle_deg, args.camera_deposit_angle_deg)) <= float(args.camera_servo_map_tol_deg)

    def request_camera_servo_angle(angle_deg, reason="manual"):
        nonlocal servo_command_angle_deg
        if (not args.camera_servo_track) or (sd is None):
            return False
        angle_deg = max(0.0, min(180.0, float(angle_deg)))
        if abs(servo_command_angle_deg - angle_deg) < 0.5:
            return False
        raw_angle_deg = servo_logical_to_raw(angle_deg)
        next_seq = float(sd.getNumber("Jetson/ServoCommandSeq", 0.0)) + 1.0
        sd.putNumber("Jetson/ServoCommandAngleDeg", raw_angle_deg)
        sd.putNumber("Jetson/ServoCommandSeq", next_seq)
        servo_command_angle_deg = angle_deg
        print(f"Camera servo -> logical {angle_deg:.0f} deg / raw {raw_angle_deg:.0f} deg ({reason})")
        return True

    def request_camera_map_view(reason="manual"):
        return request_camera_servo_angle(args.camera_map_angle_deg, reason)

    def request_camera_deposit_view(reason="manual"):
        return request_camera_servo_angle(args.camera_deposit_angle_deg, reason)

    def toggle_camera_view():
        refresh_camera_servo_state()
        target_is_map = abs(angle_error_deg(servo_command_angle_deg, args.camera_map_angle_deg)) <= 2.0
        if servo_map_view or target_is_map:
            request_camera_deposit_view("button")
        else:
            request_camera_map_view("button")

    def set_camera_overlay_enabled(enabled, source="button"):
        nonlocal camera_overlay_enabled
        enabled = bool(enabled)
        if camera_overlay_enabled == enabled:
            return
        camera_overlay_enabled = enabled
        print(f"Camera overlay {'ENABLED' if enabled else 'DISABLED'} via {source}.")
        publish_map_ui_state(force=True)

    def set_human_detect_enabled(enabled, source="button"):
        nonlocal human_detect_enabled, human_hazard_state, human_nearest_m
        nonlocal human_clear_countdown, human_person_map_points
        enabled = bool(enabled)
        if enabled and not human_detect_available:
            print("Human detection is unavailable; start ZEDAuto with human detection enabled to use this toggle.")
            return
        if human_detect_enabled == enabled:
            return
        human_detect_enabled = enabled
        if not enabled:
            human_hazard_state = "CLEAR"
            human_nearest_m = -1.0
            human_clear_countdown = 0
            human_person_map_points = []
        print(f"Human detection {'ENABLED' if enabled else 'DISABLED'} via {source}.")
        publish_map_ui_state(force=True)

    def set_rock_detect_enabled(enabled, source="button"):
        nonlocal rock_detect_enabled, rock_overlay_detections
        enabled = bool(enabled)
        if enabled and rock_model is None:
            print("Rock YOLO is unavailable; load a rock model to use this toggle.")
            return
        if rock_detect_enabled == enabled:
            return
        rock_detect_enabled = enabled
        if not enabled:
            rock_overlay_detections = []
        print(f"Rock YOLO {'ENABLED' if enabled else 'DISABLED'} via {source}.")
        publish_map_ui_state(force=True)

    def set_low_latency_mode(enabled, source="button"):
        nonlocal low_latency_mode, low_latency_restore_state
        enabled = bool(enabled)
        if low_latency_mode == enabled:
            return
        if enabled:
            low_latency_restore_state = {
                "camera_overlay_enabled": bool(camera_overlay_enabled),
                "human_detect_enabled": bool(human_detect_enabled),
                "rock_detect_enabled": bool(rock_detect_enabled),
                "camera_publish_enabled": bool(camera_publisher is not None),
                "map_publish_enabled": bool(map_publisher is not None),
            }
            set_camera_overlay_enabled(False, source)
            set_human_detect_enabled(False, source)
            set_rock_detect_enabled(False, source)
            low_latency_mode = True
            print("Low latency mode ENABLED via %s. Heavy detection and publish paths reduced." % source)
        else:
            low_latency_mode = False
            if low_latency_restore_state.get("camera_overlay_enabled", False):
                set_camera_overlay_enabled(True, source)
            if low_latency_restore_state.get("human_detect_enabled", False):
                set_human_detect_enabled(True, source)
            if low_latency_restore_state.get("rock_detect_enabled", False):
                set_rock_detect_enabled(True, source)
            print("Low latency mode DISABLED via %s." % source)
        publish_map_ui_state(force=True)

    def save_calibration_settings(result_text):
        drive_calibration.save_runtime_settings(
            bool(args.drive_heading_flip),
            bool(args.hard_drive_flip),
            bool(args.steering_flip),
            bool(args.display_heading_flip),
            float(args.camera_map_angle_deg),
            float(args.camera_deposit_angle_deg),
            result_text,
        )

    def set_drive_heading_flip(enabled, source="button"):
        enabled = bool(enabled)
        if bool(args.drive_heading_flip) == enabled:
            return
        args.drive_heading_flip = enabled
        save_calibration_settings(
            f"Drive heading flip set {'ON' if args.drive_heading_flip else 'OFF'} via {source}."
        )
        print(f"Drive heading flip {'ENABLED' if enabled else 'DISABLED'} via {source}.")
        publish_map_ui_state(force=True)

    def set_hard_drive_flip(enabled, source="button"):
        enabled = bool(enabled)
        if bool(args.hard_drive_flip) == enabled:
            return
        args.hard_drive_flip = enabled
        if sd is not None:
            sd.putBoolean("Drive/MainRoverHardFlip", bool(args.hard_drive_flip))
        save_calibration_settings(
            f"Hard drive flip set {'ON' if args.hard_drive_flip else 'OFF'} via {source}."
        )
        print(f"Hard drive flip {'ENABLED' if enabled else 'DISABLED'} via {source}.")
        publish_map_ui_state(force=True)

    def set_steering_flip(enabled, source="button"):
        enabled = bool(enabled)
        if bool(args.steering_flip) == enabled:
            return
        args.steering_flip = enabled
        save_calibration_settings(
            f"Steering flip set {'ON' if args.steering_flip else 'OFF'} via {source}."
        )
        print(f"Steering flip {'ENABLED' if enabled else 'DISABLED'} via {source}.")
        publish_map_ui_state(force=True)

    def set_bidirectional_auto(enabled, source="button"):
        nonlocal bidirectional_auto_enabled
        enabled = bool(enabled)
        if bidirectional_auto_enabled == enabled:
            return
        bidirectional_auto_enabled = enabled
        print(f"Bidirectional auto {'ENABLED' if enabled else 'DISABLED'} via {source}.")
        publish_map_ui_state(force=True)

    def ensure_demo_rover_pose(seed_pos_map=None, seed_forward_world=None):
        nonlocal demo_rover_pos_map, demo_rover_heading_rad
        if demo_rover_pos_map is not None:
            return True
        if seed_pos_map is not None:
            demo_rover_pos_map = np.array(seed_pos_map, dtype=np.float32).reshape(3,)
        else:
            seed_rc = (
                mining.preferred_start_rc
                or (mining._poly_centroid(mining.excav_corners_rc) if mining.excav_corners_rc else None)
                or goal_cell
                or (occ_map.grid_h // 2, occ_map.grid_w // 2)
            )
            if seed_rc is None:
                return False
            seed_world = occ_map.grid_to_world(int(seed_rc[0]), int(seed_rc[1]))
            if seed_world is None:
                return False
            demo_rover_pos_map = np.array(
                [float(seed_world[0]), 0.0, float(seed_world[1])],
                dtype=np.float32,
            )
        if seed_forward_world is not None:
            fwd = np.array(seed_forward_world, dtype=np.float32).reshape(3,)
            demo_rover_heading_rad = math.atan2(float(fwd[2]), float(fwd[0]))
        else:
            demo_rover_heading_rad = 0.0
        return True

    def advance_demo_rover(fwd_cmd, turn_cmd, duration):
        nonlocal demo_rover_pos_map, demo_rover_heading_rad
        if (not demo_auto_enabled) or (not ensure_demo_rover_pose()):
            return
        dt = max(0.02, min(0.25, float(duration)))
        turn_rate_rad_per_sec = math.radians(135.0)
        demo_rover_heading_rad += float(turn_cmd) * turn_rate_rad_per_sec * dt
        speed_mps = 0.90 * max(-1.0, min(1.0, float(fwd_cmd)))
        demo_rover_pos_map[0] += math.cos(demo_rover_heading_rad) * speed_mps * dt
        demo_rover_pos_map[2] += math.sin(demo_rover_heading_rad) * speed_mps * dt

    def set_demo_auto(enabled, source="button", seed_pos_map=None, seed_forward_world=None):
        nonlocal demo_auto_enabled, demo_rover_pos_map, demo_rover_heading_rad
        enabled = bool(enabled)
        if demo_auto_enabled == enabled:
            return
        demo_auto_enabled = enabled
        if enabled:
            demo_rover_pos_map = None
            demo_rover_heading_rad = 0.0
            ensure_demo_rover_pose(seed_pos_map, seed_forward_world)
        else:
            demo_rover_pos_map = None
            demo_rover_heading_rad = 0.0
        print(f"Demo Auto {'ENABLED' if enabled else 'DISABLED'} via {source}.")
        publish_map_ui_state(force=True)

    def request_start_frame_lock(source="button"):
        nonlocal start_frame_lock_requested, start_frame_last_status
        nonlocal start_frame_locked_once, start_frame_scan_active
        nonlocal start_frame_scan_started_at, start_frame_scan_samples
        start_frame_scan_active = False
        start_frame_scan_started_at = 0.0
        start_frame_scan_samples = []
        start_frame_lock_requested = True
        start_frame_locked_once = False
        start_frame_last_status = f"Start frame: requested via {source}."
        print(start_frame_last_status)
        publish_map_ui_state(force=True)

    def request_start_frame_scan(source="button"):
        nonlocal start_frame_scan_active, start_frame_scan_started_at, start_frame_scan_samples
        nonlocal start_frame_lock_requested, start_frame_last_status, start_frame_locked_once
        start_frame_lock_requested = False
        start_frame_locked_once = False
        start_frame_scan_active = True
        start_frame_scan_started_at = time.time()
        start_frame_scan_samples = []
        start_frame_last_status = f"Start frame: scanning via {source}."
        print(start_frame_last_status)
        publish_map_ui_state(force=True)

    def set_show_all_dig_profiles(enabled, source="button"):
        nonlocal show_all_dig_profiles
        enabled = bool(enabled)
        if show_all_dig_profiles == enabled:
            return
        show_all_dig_profiles = enabled
        print(
            f"View-all dig recordings {'ENABLED' if enabled else 'DISABLED'} via {source}."
        )
        publish_map_ui_state(force=True)

    def set_drive_speed(value, source="slider"):
        value = max(0.10, min(1.00, float(value)))
        if abs(float(args.drive_speed) - value) <= 1e-6:
            return
        args.drive_speed = value
        print(f"Auto drive speed set to {args.drive_speed:.2f} via {source}.")
        publish_map_ui_state(force=True)

    def set_turn_speed(value, source="slider"):
        value = max(0.20, min(1.00, float(value)))
        new_turn_k = max(0.30, min(1.60, 0.15 + value))
        if (
            abs(float(args.drive_max_turn_cmd) - value) <= 1e-6
            and abs(float(args.drive_turn_k) - new_turn_k) <= 1e-6
        ):
            return
        args.drive_max_turn_cmd = value
        args.drive_turn_k = new_turn_k
        print(
            f"Auto turn speed set to {value:.2f} via {source} "
            f"(turn_k={args.drive_turn_k:.2f}, max_turn={args.drive_max_turn_cmd:.2f})."
        )
        publish_map_ui_state(force=True)

    def set_digger_speed(value, source="slider"):
        nonlocal digger_speed_scale
        value = max(0.10, min(1.00, float(value)))
        if abs(float(digger_speed_scale) - value) <= 1e-6:
            return
        digger_speed_scale = value
        print(f"Digger speed set to {digger_speed_scale:.2f} via {source}.")
        publish_map_ui_state(force=True)

    def set_display_heading_flip(enabled, source="button"):
        enabled = bool(enabled)
        if bool(args.display_heading_flip) == enabled:
            return
        args.display_heading_flip = enabled
        save_calibration_settings(
            f"Display heading arrow flip set {'ON' if args.display_heading_flip else 'OFF'} via {source}."
        )
        print(f"Display heading arrow flip {'ENABLED' if enabled else 'DISABLED'} via {source}.")
        publish_map_ui_state(force=True)

    def flip_camera_view_calibration(source="button"):
        refresh_camera_servo_state()
        target_is_map = abs(angle_error_deg(servo_command_angle_deg, args.camera_map_angle_deg)) <= 2.0
        was_map_view = bool(servo_map_view or target_is_map)
        args.camera_map_angle_deg, args.camera_deposit_angle_deg = (
            float(args.camera_deposit_angle_deg),
            float(args.camera_map_angle_deg),
        )
        result_text = (
            f"Camera map/deposit directions flipped via {source}. "
            f"Map={float(args.camera_map_angle_deg):.0f} Deposit={float(args.camera_deposit_angle_deg):.0f}."
        )
        save_calibration_settings(result_text)
        if args.camera_servo_track and sd is not None:
            if was_map_view:
                request_camera_map_view("camera flip")
            else:
                request_camera_deposit_view("camera flip")
        print(result_text)
        publish_map_ui_state(force=True)

    def sync_selected_dig_profile():
        dig_profile = dig_profiles.get_selected_profile(phase="dig")
        dig_duration_sec = dig_profiles.selected_duration_sec(phase="dig")
        retract_duration_sec = dig_profiles.selected_duration_sec(phase="retract")
        mining.cfg["dig_duration"] = float(dig_duration_sec if dig_duration_sec is not None else default_dig_duration_sec)
        mining.cfg["backup_duration"] = float(
            retract_duration_sec if retract_duration_sec is not None else default_backup_duration_sec
        )
        return dig_profile

    def set_drive_calibration_mode(enabled, source="button"):
        enabled = bool(enabled)
        drive_calibration.set_active(enabled)
        state = "ON" if drive_calibration.active else "OFF"
        print(f"Drive calibration mode: {state} via {source}.")
        if drive_calibration.last_result:
            print(drive_calibration.last_result)
        publish_map_ui_state(force=True)

    def cycle_dig_style(source="button"):
        style = dig_profiles.cycle_active_style()
        sync_selected_dig_profile()
        print(f"Active dig style: {style.upper()} via {source}.")
        publish_map_ui_state(force=True)

    def cycle_dig_phase(source="button"):
        phase = dig_profiles.cycle_active_phase()
        sync_selected_dig_profile()
        print(f"Active dig phase: {phase.upper()} via {source}.")
        publish_map_ui_state(force=True)

    def resolve_preview_dig_profile():
        profile = dig_profiles.get_selected_profile()
        if profile is None:
            profile = dig_profiles.get_cursor_profile()
        return profile

    def resolve_preview_controller_macro():
        macro = controller_macros.get_selected_macro()
        if macro is None:
            macro = controller_macros.get_cursor_macro()
        return macro

    def cycle_dig_profile_cursor(step, source="button"):
        profile = dig_profiles.cycle_cursor(step)
        if profile is None:
            print(f"No {dig_profiles.active_style} {dig_profiles.active_phase} profiles recorded yet.")
        else:
            print(
                f"Browsing {dig_profiles.active_style} {dig_profiles.active_phase} profile: "
                f"{profile['name']} ({float(profile.get('duration_sec', 0.0)):.2f}s)"
            )
        publish_map_ui_state(force=True)

    def use_browsed_dig_profile(source="button"):
        profile = dig_profiles.select_cursor()
        if profile is None:
            print(f"No {dig_profiles.active_style} {dig_profiles.active_phase} profile available to select.")
            return
        sync_selected_dig_profile()
        print(
            f"Selected {profile['style']} {profile.get('phase', 'dig')} profile: "
            f"{profile['name']} via {source}."
        )
        publish_map_ui_state(force=True)

    def delete_browsed_dig_profile(source="button"):
        profile = dig_profiles.delete_cursor()
        if profile is None:
            print(f"No {dig_profiles.active_style} {dig_profiles.active_phase} profile available to delete.")
            return
        sync_selected_dig_profile()
        print(
            f"Deleted {profile['style']} {profile.get('phase', 'dig')} profile: "
            f"{profile['name']} via {source}."
        )
        publish_map_ui_state(force=True)

    def stop_dig_profile_preview(source="button", completed=False):
        nonlocal dig_profile_preview_active, dig_profile_preview_started_at
        nonlocal dig_profile_preview_style, dig_profile_preview_phase, dig_profile_preview_name
        if not dig_profile_preview_active:
            return
        name = dig_profile_preview_name or "profile"
        dig_profile_preview_active = False
        dig_profile_preview_started_at = 0.0
        dig_profile_preview_style = None
        dig_profile_preview_phase = None
        dig_profile_preview_name = None
        reset_auto_drive_shape(time.time())
        send_nt_command(False, 0.0, 0.0, 0.1)
        if completed:
            print(f"Preview completed for {name}.")
        else:
            print(f"Preview stopped for {name} via {source}.")
        publish_map_ui_state(force=True)

    def start_dig_profile_preview(source="button"):
        nonlocal dig_profile_preview_active, dig_profile_preview_started_at
        nonlocal dig_profile_preview_style, dig_profile_preview_phase, dig_profile_preview_name
        if dig_profiles.recording:
            print("Stop recording before previewing a profile.")
            return
        if controller_macros.recording or controller_macro_preview_active:
            print("Stop controller recording/playback before previewing a dig record.")
            return
        if controller_cycle_preview_active:
            print("Stop controller cycle playback before previewing a dig record.")
            return
        profile = resolve_preview_dig_profile()
        if profile is None:
            print(f"No {dig_profiles.active_style} {dig_profiles.active_phase} profile available to preview.")
            return
        clear_navigation_goal()
        mining.abort()
        set_manual_drive_mode(False, f"{source} dig preview")
        dig_profile_preview_active = True
        dig_profile_preview_started_at = time.time()
        dig_profile_preview_style = str(profile.get("style", dig_profiles.active_style))
        dig_profile_preview_phase = str(profile.get("phase", dig_profiles.active_phase))
        dig_profile_preview_name = str(profile.get("name", "profile"))
        reset_auto_drive_shape(dig_profile_preview_started_at)
        print(
            f"Previewing {dig_profile_preview_style} {dig_profile_preview_phase} profile "
            f"{dig_profile_preview_name} via {source}."
        )
        publish_map_ui_state(force=True)

    def cycle_controller_macro_cursor(step, source="button"):
        macro = controller_macros.cycle_cursor(step)
        if macro is None:
            print("No controller macros recorded yet.")
        else:
            print(
                f"Browsing controller macro: "
                f"{macro['name']} ({float(macro.get('duration_sec', 0.0)):.2f}s)"
            )
        publish_map_ui_state(force=True)

    def use_browsed_controller_macro(source="button"):
        macro = controller_macros.select_cursor()
        if macro is None:
            print("No controller macro available to select.")
            return
        print(
            f"Selected controller macro: {macro['name']} "
            f"({float(macro.get('duration_sec', 0.0)):.2f}s) via {source}."
        )
        publish_map_ui_state(force=True)

    def stop_controller_macro_preview(source="button", completed=False):
        nonlocal controller_macro_preview_active, controller_macro_preview_started_at
        nonlocal controller_macro_preview_name
        if not controller_macro_preview_active:
            return
        name = controller_macro_preview_name or "controller macro"
        controller_macro_preview_active = False
        controller_macro_preview_started_at = 0.0
        controller_macro_preview_name = None
        reset_auto_drive_shape(time.time())
        send_nt_command(False, 0.0, 0.0, 0.1)
        if completed:
            print(f"Controller macro preview completed for {name}.")
        else:
            print(f"Controller macro preview stopped for {name} via {source}.")
        publish_map_ui_state(force=True)

    def stop_controller_cycle_preview(source="button", completed=False):
        nonlocal controller_cycle_preview_active, controller_cycle_phase, controller_cycle_phase_started_at
        nonlocal controller_cycle_preview_name
        if not controller_cycle_preview_active:
            return
        name = controller_cycle_preview_name or "controller cycle"
        controller_cycle_preview_active = False
        controller_cycle_phase = "forward"
        controller_cycle_phase_started_at = 0.0
        controller_cycle_preview_name = None
        reset_auto_drive_shape(time.time())
        send_nt_command(False, 0.0, 0.0, 0.1)
        if completed:
            print(f"Controller cycle completed for {name}.")
        else:
            print(f"Controller cycle stopped for {name} via {source}.")
        publish_map_ui_state(force=True)

    def start_controller_macro_preview(source="button"):
        nonlocal controller_macro_preview_active, controller_macro_preview_started_at
        nonlocal controller_macro_preview_name
        if controller_macros.recording:
            print("Stop controller recording before previewing a controller macro.")
            return
        if controller_cycle_preview_active:
            print("Stop controller cycle playback before previewing a controller macro.")
            return
        if dig_profiles.recording or dig_profile_preview_active:
            print("Stop dig recording/playback before previewing a controller macro.")
            return
        macro = resolve_preview_controller_macro()
        if macro is None:
            print("No controller macro available to preview.")
            return
        clear_navigation_goal()
        mining.abort()
        set_manual_drive_mode(False, f"{source} controller macro preview")
        controller_macro_preview_active = True
        controller_macro_preview_started_at = time.time()
        controller_macro_preview_name = str(macro.get("name", "controller_macro"))
        reset_auto_drive_shape(controller_macro_preview_started_at)
        print(
            f"Previewing controller macro {controller_macro_preview_name} via {source}."
        )
        publish_map_ui_state(force=True)

    def start_controller_cycle_preview(source="button"):
        nonlocal controller_cycle_preview_active, controller_cycle_phase, controller_cycle_phase_started_at
        nonlocal controller_cycle_preview_name
        if controller_macros.recording:
            print("Stop controller recording before starting a controller cycle.")
            return
        if dig_profiles.recording or dig_profile_preview_active:
            print("Stop dig recording/playback before starting a controller cycle.")
            return
        if controller_macro_preview_active:
            print("Stop normal controller playback before starting a controller cycle.")
            return
        macro = resolve_preview_controller_macro()
        if macro is None:
            print("No controller macro available to cycle.")
            return
        clear_navigation_goal()
        mining.abort()
        set_manual_drive_mode(False, f"{source} controller cycle")
        controller_cycle_preview_active = True
        controller_cycle_phase = "forward"
        controller_cycle_phase_started_at = time.time()
        controller_cycle_preview_name = str(macro.get("name", "controller_cycle"))
        reset_auto_drive_shape(controller_cycle_phase_started_at)
        print(
            f"Cycling controller macro {controller_cycle_preview_name} via {source} "
            f"(forward, then return without dig/deposit actions)."
        )
        publish_map_ui_state(force=True)

    def start_controller_recording(source="button"):
        nonlocal dig_name_input_focused
        name_base = str(dig_name_input_text or "").strip()
        if not name_base:
            dig_name_input_focused = True
            print("Enter a profile name in the status panel before recording a controller macro.")
            publish_map_ui_state(force=True)
            return
        if not any(ch.isalnum() for ch in name_base):
            dig_name_input_focused = True
            print("Controller macro name must include at least one letter or number.")
            publish_map_ui_state(force=True)
            return
        if controller_macro_preview_active:
            print("Stop controller macro preview before starting a new recording.")
            return
        if controller_cycle_preview_active:
            print("Stop controller cycle playback before starting a new recording.")
            return
        if dig_profiles.recording or dig_profile_preview_active:
            print("Stop dig recording/playback before starting a controller macro recording.")
            return
        if controller_macros.recording:
            print("Controller macro recording already active. Stop it first.")
            return
        clear_navigation_goal()
        mining.abort()
        set_manual_drive_mode(True, f"{source} controller macro recording")
        if not controller_macros.begin_recording(name_base=name_base):
            print("Failed to start controller macro recording.")
            return
        print(
            "Recording controller macro. Drive and use actuators/door controls, "
            "then stop recording to save it."
        )
        publish_map_ui_state(force=True)

    def stop_controller_recording(save=True, source="button"):
        macro = controller_macros.stop_recording(save=save)
        if macro is None:
            if save:
                print("Controller macro stopped without saving a usable recording.")
            else:
                print("Controller macro recording canceled.")
            publish_map_ui_state(force=True)
            return
        print(
            f"Saved controller macro {macro['name']} "
            f"({float(macro.get('duration_sec', 0.0)):.2f}s) via {source}."
        )
        publish_map_ui_state(force=True)

    def start_dig_recording(style, phase, source="button"):
        nonlocal dig_name_input_focused
        style = str(style).strip().lower()
        phase = str(phase).strip().lower()
        if style not in ("short", "long"):
            print(f"Ignoring dig recording request for unknown style '{style}'.")
            return
        if phase not in ("dig", "retract"):
            print(f"Ignoring dig recording request for unknown phase '{phase}'.")
            return
        name_base = str(dig_name_input_text or "").strip()
        if not name_base:
            dig_name_input_focused = True
            print("Enter a dig profile name in the status panel before recording.")
            publish_map_ui_state(force=True)
            return
        if not any(ch.isalnum() for ch in name_base):
            dig_name_input_focused = True
            print("Dig profile name must include at least one letter or number.")
            publish_map_ui_state(force=True)
            return
        if controller_macros.recording:
            print("Stop controller macro recording before starting a dig recording.")
            return
        if controller_cycle_preview_active:
            print("Stop controller cycle playback before starting a dig recording.")
            return
        if dig_profile_preview_active:
            print("Stop dig profile preview before starting a new recording.")
            return
        if dig_profiles.recording:
            print(
                "Dig recording already active "
                f"({dig_profiles.recording_style}/{dig_profiles.recording_phase}). Stop it first."
            )
            return
        dig_profiles.active_style = style
        dig_profiles.active_phase = phase
        if not dig_profiles.begin_recording(style, phase, name_base=name_base):
            print(f"Failed to start {style} {phase} recording.")
            return
        clear_navigation_goal()
        mining.abort()
        set_manual_drive_mode(True, f"{source} dig recording")
        print(
            f"Recording {style} {phase} profile. Use manual drive and digger controls, "
            "then stop recording to save it."
        )
        publish_map_ui_state(force=True)

    def start_active_dig_recording(source="button"):
        start_dig_recording(dig_profiles.active_style, dig_profiles.active_phase, source)

    def stop_dig_recording(save=True, source="button"):
        profile = dig_profiles.stop_recording(save=save)
        if profile is None:
            if save:
                print("Dig recording stopped without saving a usable profile.")
            else:
                print("Dig recording canceled.")
            publish_map_ui_state(force=True)
            return
        sync_selected_dig_profile()
        print(
            f"Saved {profile['style']} {profile.get('phase', 'dig')} profile {profile['name']} "
            f"({float(profile.get('duration_sec', 0.0)):.2f}s) via {source}."
        )
        publish_map_ui_state(force=True)

    sync_selected_dig_profile()

    def lock_green_zones_permanent():
        nonlocal last_save, lock_green_applied, lock_green_locked_count
        strong_occ = occ_map.obstacle_mask(min_occ_count=3.0, min_occ_ratio=2.0, min_occ_advantage=1.0)
        green_mask = (
            (occ_map.free_counts >= 1.0)
            & (occ_map.free_counts >= occ_map.occ_counts)
            & (occ_map.free_counts >= occ_map.hole_counts)
            & (~strong_occ)
        )
        count = int(np.count_nonzero(green_mask))
        if count <= 0:
            print("Lock Green: no green/safe cells to lock.")
            return
        lock_green_applied = True
        lock_green_locked_count = count
        occ_map.painted_free[green_mask] = True
        occ_map.occ_counts[green_mask] = 0.0
        occ_map.hole_counts[green_mask] = 0.0
        occ_map.free_counts[green_mask] = np.maximum(
            occ_map.free_counts[green_mask],
            float(occ_map.free_confirm_hits),
        )
        try:
            occ_map.save(args.map_save_path)
            last_save = time.time()
        except Exception as exc:
            print(f"Lock Green save failed: {exc}")
        print(f"Lock Green: {count} green/safe cells marked permanent.")

    def set_status_scroll(delta):
        nonlocal status_scroll_y
        status_scroll_y = max(0, min(int(status_scroll_max), int(status_scroll_y + delta)))

    def set_status_scroll_to(target_y):
        nonlocal status_scroll_y
        status_scroll_y = max(0, min(int(status_scroll_max), int(target_y)))

    def window_to_image_coords(window_name, x, y, frame_shape):
        if frame_shape is None:
            return int(x), int(y)
        img_h, img_w = int(frame_shape[0]), int(frame_shape[1])
        if img_h <= 0 or img_w <= 0:
            return int(x), int(y)
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
            if win_w > 1 and win_h > 1:
                scale = min(float(win_w) / float(img_w), float(win_h) / float(img_h))
                if scale > 0.0:
                    draw_w = float(img_w) * scale
                    draw_h = float(img_h) * scale
                    pad_x = max(0.0, (float(win_w) - draw_w) * 0.5)
                    pad_y = max(0.0, (float(win_h) - draw_h) * 0.5)
                    x = int(round((float(x) - pad_x) / scale))
                    y = int(round((float(y) - pad_y) / scale))
        except Exception:
            pass
        x = max(0, min(img_w - 1, int(x)))
        y = max(0, min(img_h - 1, int(y)))
        return x, y

    def reset_map_memory():
        nonlocal goal_cell, path_cells, last_path_cells, last_start, last_goal, last_path_plan_time
        nonlocal path_plan_mode
        nonlocal emergency_stop, reset_map_confirm, landmark_memory, landmark_dirty, last_save
        nonlocal lock_green_applied, lock_green_locked_count, mining_goal_active
        nonlocal landmark_pose_override_t_map, landmark_pose_override_R_world_cam
        occ_map.free_counts[:] = 0.0
        occ_map.occ_counts[:] = 0.0
        occ_map.hole_counts[:] = 0.0
        occ_map.painted_free[:] = False
        lock_green_applied = False
        lock_green_locked_count = 0
        goal_cell = None
        path_cells = None
        last_path_cells = None
        last_start = None
        last_goal = None
        last_path_plan_time = 0.0
        path_plan_mode = "none"
        mining_goal_active = False
        emergency_stop = True
        reset_map_confirm = False
        landmark_memory = {"version": 1, "landmarks": []}
        landmark_pose_override_t_map = None
        landmark_pose_override_R_world_cam = None
        landmark_dirty = True
        try:
            occ_map.save(args.map_save_path)
            last_save = time.time()
        except Exception as exc:
            print(f"Map reset save failed: {exc}")
        save_landmark_memory(force=True)
        print("Map reset: occupancy, painted cells, path, goal, and AI landmarks cleared.")

    def clear_navigation_goal():
        nonlocal goal_cell, path_cells, last_path_cells, last_start, last_goal, last_path_plan_time
        nonlocal path_plan_mode, mining_goal_active, status_target_cell, status_target_world
        goal_cell = None
        path_cells = None
        last_path_cells = None
        last_start = None
        last_goal = None
        last_path_plan_time = 0.0
        path_plan_mode = "none"
        mining_goal_active = False
        status_target_cell = None
        status_target_world = None

    def set_direct_nav_enabled(enabled, source="button"):
        nonlocal direct_nav_enabled
        enabled = bool(enabled)
        if direct_nav_enabled == enabled:
            return
        direct_nav_enabled = enabled
        state = "ENABLED" if direct_nav_enabled else "DISABLED"
        detail = (
            "straighter visible targets"
            if direct_nav_enabled
            else "normal short-lookahead path following"
        )
        print(f"Direct Nav {state} via {source} ({detail}).")
        publish_map_ui_state(force=True)

    def build_direct_nav_obstacle_mask(rover_rc):
        obs = occ_map.obstacle_mask(
            min_occ_count=args.path_avoid_occ_min,
            min_occ_ratio=args.path_avoid_occ_ratio,
            min_occ_advantage=args.path_avoid_occ_advantage,
        )
        if smooth_map_enabled and np.any(obs):
            kernel = np.ones((3, 3), np.uint8)
            obs = cv2.morphologyEx(obs.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        radius_cells = int(np.ceil((args.rover_size_m / 2.0) / occ_map.map_res_m))
        if radius_cells > 0 and np.any(obs):
            obs = map_utils.inflate_mask(obs, radius_cells)
        clear_cells = int(np.ceil(max(0.0, args.start_clear_radius_m) / occ_map.map_res_m))
        if clear_cells > 0:
            obs = map_utils.clear_mask_circle(obs, rover_rc, clear_cells)
        return obs

    def iter_grid_line_cells(start_rc, end_rc):
        r0, c0 = int(start_rc[0]), int(start_rc[1])
        r1, c1 = int(end_rc[0]), int(end_rc[1])
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dc - dr

        while True:
            yield r0, c0
            if r0 == r1 and c0 == c1:
                break
            e2 = err * 2
            if e2 > -dr:
                err -= dr
                c0 += sc
            if e2 < dc:
                err += dc
                r0 += sr

    def grid_segment_is_clear(start_rc, end_rc, obstacle_mask):
        if start_rc is None or end_rc is None or obstacle_mask is None:
            return False
        h, w = obstacle_mask.shape
        prev = None
        for rr, cc in iter_grid_line_cells(start_rc, end_rc):
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                return False
            if obstacle_mask[rr, cc]:
                return False
            if prev is not None:
                pr, pc = prev
                if abs(rr - pr) == 1 and abs(cc - pc) == 1:
                    if obstacle_mask[pr, cc] or obstacle_mask[rr, pc]:
                        return False
            prev = (rr, cc)
        return True

    def pick_drive_target(draw_path, rover_rc, goal_rc):
        if goal_rc is None:
            return None

        direct_obs = None
        if direct_nav_enabled and rover_rc is not None:
            direct_obs = build_direct_nav_obstacle_mask(rover_rc)
            if grid_segment_is_clear(rover_rc, goal_rc, direct_obs):
                return goal_rc

        if draw_path is not None and len(draw_path) > 0:
            wp_index = min(5, len(draw_path) - 1)
            if direct_nav_enabled and rover_rc is not None and len(draw_path) > 1:
                max_index = min(
                    len(draw_path) - 1,
                    max(wp_index, int(max(1, args.drive_direct_lookahead_cells))),
                )
                for idx in range(max_index, wp_index, -1):
                    if grid_segment_is_clear(rover_rc, draw_path[idx], direct_obs):
                        return draw_path[idx]
            return draw_path[wp_index]

        if direct_nav_enabled and rover_rc is not None and grid_segment_is_clear(rover_rc, goal_rc, direct_obs):
            return goal_rc

        return None

    def publish_map_ui_state(force=False):
        nonlocal last_map_ui_state_write
        refresh_camera_servo_state()
        refresh_actuator_feedback()
        now = time.time()
        if (not force) and (now - last_map_ui_state_write) < 0.20:
            return

        button_enabled = mining_buttons_enabled()
        selected_tool = current_selected_tool()
        dig_ui_state = dig_profiles.ui_state()
        drive_cal_state = drive_calibration.ui_state()
        mining_active = mining.state not in (
            auto_mining.MiningState.IDLE,
            auto_mining.MiningState.DRAW_EXCAV,
            auto_mining.MiningState.DRAW_DEPOSIT,
            auto_mining.MiningState.PICK_DIG_START,
            auto_mining.MiningState.DONE,
            auto_mining.MiningState.ABORTED,
        )
        payload = {
            "available": True,
            "source": "zed_ground_wall",
            "timestamp_ms": int(now * 1000),
            "mining_state": mining.state.value,
            "localization_scan_active": False,
            "landmark_count": int(len(landmark_memory.get("landmarks", []))),
            "selected_tool": selected_tool,
            "brush_radius": int(paint_brush_radius),
            "brush_radius_min": 1,
            "brush_radius_max": 15,
            "drive_speed": float(args.drive_speed),
            "drive_speed_min": 0.10,
            "drive_speed_max": 1.00,
            "drive_turn_speed": float(args.drive_max_turn_cmd),
            "drive_turn_speed_min": 0.20,
            "drive_turn_speed_max": 1.00,
            "bidirectional_auto": bool(bidirectional_auto_enabled),
            "demo_auto": bool(demo_auto_enabled),
            "drive_calibration": drive_cal_state,
            "dig_profiles": dig_ui_state,
            "actuators": {
                "left_extension_pct": actuator_left_extension_pct,
                "right_extension_pct": actuator_right_extension_pct,
                "left_extension_inches": actuator_left_extension_inches,
                "right_extension_inches": actuator_right_extension_inches,
                "tailgate_extension_pct": actuator_tailgate_extension_pct,
                "tailgate_inches": actuator_tailgate_inches,
                "tailgate_counts": actuator_tailgate_counts,
                "tailgate_position_calibrated": actuator_tailgate_position_calibrated,
                "tailgate_state": actuator_tailgate_state,
                "tailgate_moving": actuator_tailgate_moving,
                "tailgate_open": actuator_tailgate_open,
                "tailgate_closed": actuator_tailgate_closed,
                "bottom_position_calibrated": actuator_bottom_position_calibrated,
                "sync_fault": actuator_sync_fault,
                "left_extend_command": bool(test_excavation_left_extend_active),
                "right_extend_command": bool(test_excavation_right_extend_active),
                "dig_command": bool(test_excavation_dig_active),
                "lower_command": bool(test_excavation_lower_active),
                "door_open_command": bool(test_door_open_active),
                "door_close_command": bool(test_door_close_active),
                "pattern_test_active": bool(excavation_pattern_test_active),
            },
            "controls": [
                {
                    "id": "paint_obstacle",
                    "label": "Paint Obstacle",
                    "command": "paint_obstacle",
                    "active": bool(paint_obstacle_mode),
                    "enabled": True,
                },
                {
                    "id": "paint_safe",
                    "label": "Paint Safe",
                    "command": "paint_safe",
                    "active": bool(paint_safe_mode),
                    "enabled": True,
                },
                {
                    "id": "erase_safe",
                    "label": "Erase Safe",
                    "command": "erase_safe",
                    "active": bool(erase_safe_mode),
                    "enabled": True,
                },
                {
                    "id": "clear_all",
                    "label": "Clear All",
                    "command": "clear_all",
                    "active": False,
                    "enabled": True,
                },
                {
                    "id": "lock_green",
                    "label": "Green Locked" if lock_green_applied else "Lock Green",
                    "command": "lock_green",
                    "active": bool(lock_green_applied),
                    "enabled": True,
                },
                {
                    "id": "reset_map",
                    "label": "Reset Map",
                    "command": "reset_map",
                    "active": bool(reset_map_confirm),
                    "enabled": True,
                },
                {
                    "id": "reset_confirm",
                    "label": "Confirm Reset",
                    "command": "reset_confirm",
                    "active": False,
                    "enabled": bool(reset_map_confirm),
                },
                {
                    "id": "reset_cancel",
                    "label": "Cancel Reset",
                    "command": "reset_cancel",
                    "active": False,
                    "enabled": bool(reset_map_confirm),
                },
                {
                    "id": "auto_run",
                    "label": "Stop Auto Run" if mining_active else "Start Auto Run",
                    "command": "auto_run",
                    "active": bool(mining_active),
                    "enabled": True,
                },
                {
                    "id": "auto_digger",
                    "label": "Auto Dig",
                    "command": "auto_digger",
                    "active": bool(auto_digger_enabled),
                    "enabled": True,
                },
                {
                    "id": "camera_overlay",
                    "label": "Camera Overlay",
                    "command": "camera_overlay",
                    "active": bool(camera_overlay_enabled),
                    "enabled": True,
                },
                {
                    "id": "human_detect_toggle",
                    "label": "Human Detect",
                    "command": "human_detect_toggle",
                    "active": bool(human_detect_enabled),
                    "enabled": bool(human_detect_available),
                },
                {
                    "id": "rock_detect_toggle",
                    "label": "Rock YOLO",
                    "command": "rock_detect_toggle",
                    "active": bool(rock_detect_enabled),
                    "enabled": bool(rock_model is not None),
                },
                {
                    "id": "low_latency_mode",
                    "label": "Low Latency",
                    "command": "low_latency_mode",
                    "active": bool(low_latency_mode),
                    "enabled": True,
                },
                {
                    "id": "drive_heading_flip",
                    "label": "Flip Drive",
                    "command": "drive_heading_flip",
                    "active": bool(args.drive_heading_flip),
                    "enabled": True,
                },
                {
                    "id": "hard_drive_flip",
                    "label": "Hard Flip",
                    "command": "hard_drive_flip",
                    "active": bool(args.hard_drive_flip),
                    "enabled": True,
                },
                {
                    "id": "steering_flip",
                    "label": "Flip Steering",
                    "command": "steering_flip",
                    "active": bool(args.steering_flip),
                    "enabled": True,
                },
                {
                    "id": "bidirectional_auto",
                    "label": "Bidirectional Auto",
                    "command": "bidirectional_auto",
                    "active": bool(bidirectional_auto_enabled),
                    "enabled": True,
                },
                {
                    "id": "demo_auto",
                    "label": "Demo Auto",
                    "command": "demo_auto",
                    "active": bool(demo_auto_enabled),
                    "enabled": True,
                },
                {
                    "id": "lock_start_frame",
                    "label": "Lock Start Frame",
                    "command": "lock_start_frame",
                    "active": False,
                    "enabled": bool(tracking_enabled and start_frame_tag_dictionary is not None and len(start_frame_tag_layout) >= 3),
                },
                {
                    "id": "scan_start_frame",
                    "label": "Scan Start Frame",
                    "command": "scan_start_frame",
                    "active": bool(start_frame_scan_active),
                    "enabled": bool(tracking_enabled and start_frame_tag_dictionary is not None and len(start_frame_tag_layout) >= 3),
                },
                {
                    "id": "test_drive_forward",
                    "label": "Test Forward 5s",
                    "command": "test_drive_forward",
                    "active": bool(test_drive_forward_active),
                    "enabled": bool(args.drive and sd is not None),
                },
                {
                    "id": "camera_view_flip",
                    "label": "Flip Map/Depo: ON" if camera_view_flip_active() else "Flip Map/Depo",
                    "command": "camera_view_flip",
                    "active": bool(camera_view_flip_active()),
                    "enabled": True,
                },
                {
                    "id": "display_heading_flip",
                    "label": "Flip Arrow",
                    "command": "display_heading_flip",
                    "active": bool(args.display_heading_flip),
                    "enabled": True,
                },
                {
                    "id": "direct_nav",
                    "label": "Direct Nav",
                    "command": "direct_nav",
                    "active": bool(direct_nav_enabled),
                    "enabled": True,
                },
                {
                    "id": "drive_calibration_mode",
                    "label": "Drive Cal Mode",
                    "command": "drive_calibration_mode",
                    "active": bool(drive_calibration.active),
                    "enabled": True,
                },
                {
                    "id": "drive_calibration_cancel",
                    "label": "Cancel Cal",
                    "command": "drive_calibration_cancel",
                    "active": False,
                    "enabled": bool(drive_calibration.active or drive_calibration.target_cell is not None),
                },
                {
                    "id": "dig_style_cycle",
                    "label": f"Dig Style: {dig_profiles.active_style.title()}",
                    "command": "dig_style_cycle",
                    "active": False,
                    "enabled": True,
                },
                {
                    "id": "dig_phase_cycle",
                    "label": f"Dig Phase: {dig_profiles.active_phase.title()}",
                    "command": "dig_phase_cycle",
                    "active": False,
                    "enabled": True,
                },
                {
                    "id": "dig_record_active",
                    "label": f"Record {dig_profiles.active_phase.title()}",
                    "command": "dig_record_active",
                    "active": bool(
                        dig_profiles.recording
                        and dig_profiles.recording_style == dig_profiles.active_style
                        and dig_profiles.recording_phase == dig_profiles.active_phase
                    ),
                    "enabled": bool(not dig_profiles.recording),
                },
                {
                    "id": "dig_profile_preview",
                    "label": "Preview Profile",
                    "command": "dig_profile_preview",
                    "active": bool(dig_profile_preview_active),
                    "enabled": bool((not dig_profiles.recording) and (resolve_preview_dig_profile() is not None)),
                },
                {
                    "id": "dig_record_stop",
                    "label": "Stop Recording",
                    "command": "dig_record_stop",
                    "active": bool(dig_profiles.recording or dig_profile_preview_active),
                    "enabled": bool(dig_profiles.recording or dig_profile_preview_active),
                },
                {
                    "id": "dig_profile_prev",
                    "label": "Dig Prev",
                    "command": "dig_profile_prev",
                    "active": False,
                    "enabled": True,
                },
                {
                    "id": "dig_profile_next",
                    "label": "Dig Next",
                    "command": "dig_profile_next",
                    "active": False,
                    "enabled": True,
                },
                {
                    "id": "dig_profile_use",
                    "label": "Use Profile",
                    "command": "dig_profile_use",
                    "active": False,
                    "enabled": bool(dig_profiles.get_cursor_profile() is not None),
                },
                {
                    "id": "dig_profile_delete",
                    "label": "Delete Profile",
                    "command": "dig_profile_delete",
                    "active": False,
                    "enabled": bool(dig_profiles.get_cursor_profile() is not None),
                },
                {
                    "id": "controller_record",
                    "label": "Record Controller",
                    "command": "controller_record",
                    "active": bool(controller_macros.recording),
                    "enabled": bool(not controller_macros.recording),
                },
                {
                    "id": "controller_preview",
                    "label": "Play Controller",
                    "command": "controller_preview",
                    "active": bool(controller_macro_preview_active),
                    "enabled": bool((not controller_macros.recording) and (resolve_preview_controller_macro() is not None)),
                },
                {
                    "id": "controller_cycle",
                    "label": "Cycle Return",
                    "command": "controller_cycle",
                    "active": bool(controller_cycle_preview_active),
                    "enabled": bool((not controller_macros.recording) and (resolve_preview_controller_macro() is not None)),
                },
                {
                    "id": "controller_stop",
                    "label": "Stop Controller",
                    "command": "controller_stop",
                    "active": bool(controller_macros.recording or controller_macro_preview_active or controller_cycle_preview_active),
                    "enabled": bool(controller_macros.recording or controller_macro_preview_active or controller_cycle_preview_active),
                },
                {
                    "id": "controller_prev",
                    "label": "Controller Prev",
                    "command": "controller_prev",
                    "active": False,
                    "enabled": True,
                },
                {
                    "id": "controller_next",
                    "label": "Controller Next",
                    "command": "controller_next",
                    "active": False,
                    "enabled": True,
                },
                {
                    "id": "controller_use",
                    "label": "Use Controller",
                    "command": "controller_use",
                    "active": False,
                    "enabled": bool(controller_macros.get_cursor_macro() is not None),
                },
                {
                    "id": "test_excavation_dig",
                    "label": "Test Digger",
                    "command": "test_excavation_dig",
                    "active": bool(test_excavation_dig_active),
                    "enabled": True,
                },
                {
                    "id": "test_excavation_left_extend",
                    "label": "Left Extend",
                    "command": "test_excavation_left_extend",
                    "active": bool(test_excavation_left_extend_active),
                    "enabled": True,
                },
                {
                    "id": "test_excavation_right_extend",
                    "label": "Right Extend",
                    "command": "test_excavation_right_extend",
                    "active": bool(test_excavation_right_extend_active),
                    "enabled": True,
                },
                {
                    "id": "test_excavation_lower",
                    "label": "Lower Cycle",
                    "command": "test_excavation_lower",
                    "active": bool(test_excavation_lower_active),
                    "enabled": True,
                },
                {
                    "id": "test_excavation_pattern",
                    "label": "Excav Test x4",
                    "command": "test_excavation_pattern",
                    "active": bool(excavation_pattern_test_active),
                    "enabled": bool(args.drive and sd is not None),
                },
                {
                    "id": "door_open",
                    "label": "Open Door",
                    "command": "door_open",
                    "active": bool(test_door_open_active),
                    "enabled": True,
                },
                {
                    "id": "door_close",
                    "label": "Close Door",
                    "command": "door_close",
                    "active": bool(test_door_close_active),
                    "enabled": True,
                },
                {
                    "id": "stop_actuators",
                    "label": "Stop Actuators",
                    "command": "stop_actuators",
                    "active": False,
                    "enabled": True,
                },
                {
                    "id": "main_rover_mode",
                    "label": "Main Rover Mode",
                    "command": "main_rover_mode",
                    "active": bool(args.main_rover_mode),
                    "enabled": True,
                },
                {
                    "id": "camera_view",
                    "label": (
                        "Camera: Deposit"
                        if (servo_deposit_view or abs(angle_error_deg(servo_command_angle_deg, args.camera_deposit_angle_deg)) <= 2.0)
                        else "Camera: Map"
                    ),
                    "command": "camera_view",
                    "active": bool(servo_deposit_view),
                    "enabled": bool(args.camera_servo_track and sd is not None),
                },
                {
                    "id": "draw_excav_zone",
                    "label": "Draw Excav Zone",
                    "command": "draw_excav_zone",
                    "active": mining.state == auto_mining.MiningState.DRAW_EXCAV,
                    "enabled": bool(button_enabled),
                },
                {
                    "id": "draw_deposit_zone",
                    "label": "Draw Deposit Zone",
                    "command": "draw_deposit_zone",
                    "active": mining.state == auto_mining.MiningState.DRAW_DEPOSIT,
                    "enabled": bool(button_enabled),
                },
                {
                    "id": "set_starting_zone",
                    "label": "Set Starting Zone",
                    "command": "set_starting_zone",
                    "active": bool(mining.starting_corners_rc),
                    "enabled": bool(button_enabled),
                },
                {
                    "id": "set_berm_left",
                    "label": "Berm: Left",
                    "command": "set_berm_left",
                    "active": mining.deposit_zone_preset_side == "left",
                    "enabled": bool(button_enabled),
                },
                {
                    "id": "set_berm_right",
                    "label": "Berm: Right",
                    "command": "set_berm_right",
                    "active": mining.deposit_zone_preset_side == "right",
                    "enabled": bool(button_enabled),
                },
                {
                    "id": "pick_dig_start",
                    "label": "Pick Dig Start",
                    "command": "pick_dig_start",
                    "active": mining.state == auto_mining.MiningState.PICK_DIG_START
                              or mining.preferred_start_rc is not None,
                    "enabled": bool((not button_enabled and mining.state == auto_mining.MiningState.PICK_DIG_START)
                                    or (button_enabled and bool(mining.excav_corners_rc))),
                },
                {
                    "id": "brush_minus",
                    "label": "Brush -",
                    "command": "brush_minus",
                    "active": False,
                    "enabled": True,
                },
                {
                    "id": "brush_plus",
                    "label": "Brush +",
                    "command": "brush_plus",
                    "active": False,
                    "enabled": True,
                },
            ],
        }
        _write_json_atomic(args.map_ui_state_file, payload)
        last_map_ui_state_write = now

    def apply_map_view(frame, focus_cell):
        # Returns (frame_for_display, row_shift, col_shift) where:
        # display_row = source_row + row_shift, display_col = source_col + col_shift
        if frame is None:
            return frame, 0, 0
        if (not follow_rover_map) or (focus_cell is None):
            return frame, 0, 0

        h, w = frame.shape[:2]
        fr, fc = int(focus_cell[0]), int(focus_cell[1])
        shift_r = int((h // 2) - fr)
        shift_c = int((w // 2) - fc)
        out = np.zeros_like(frame)

        src_r0 = max(0, -shift_r)
        src_r1 = min(h, h - shift_r)
        src_c0 = max(0, -shift_c)
        src_c1 = min(w, w - shift_c)

        if src_r0 < src_r1 and src_c0 < src_c1:
            dst_r0 = src_r0 + shift_r
            dst_r1 = src_r1 + shift_r
            dst_c0 = src_c0 + shift_c
            dst_c1 = src_c1 + shift_c
            out[dst_r0:dst_r1, dst_c0:dst_c1] = frame[src_r0:src_r1, src_c0:src_c1]

        return out, shift_r, shift_c

    def display_cell_for_map_cell(row, col, frame):
        if frame is None:
            return None
        dr = int(row) + int(map_view_shift_r)
        dc = int(col) + int(map_view_shift_c)
        if dr < 0 or dr >= frame.shape[0] or dc < 0 or dc >= frame.shape[1]:
            return None
        return dr, dc

    def draw_live_detection_marker(frame, row, col, is_person):
        display_cell = display_cell_for_map_cell(row, col, frame)
        if display_cell is None:
            return
        dr, dc = display_cell
        color = (0, 0, 255) if is_person else (0, 200, 255)
        cv2.circle(frame, (dc, dr), 3, color, -1)
        if is_person:
            cv2.circle(frame, (dc, dr), 6, color, 1)

    def heading_vec_from_world(rover_pos_world, forward_world):
        if rover_pos_world is None or forward_world is None:
            return None
        pos = np.array(rover_pos_world, dtype=np.float32).reshape(3,)
        fwd = np.array(forward_world, dtype=np.float32).reshape(3,)
        fwd_xz_norm = float(np.linalg.norm(fwd[[0, 2]]))
        if fwd_xz_norm <= 1e-6:
            return None
        fwd = fwd / fwd_xz_norm
        start_rc = map_world_to_grid(float(pos[0]), float(pos[2]))
        if start_rc is None:
            return None
        step_candidates = (
            max(0.35, float(args.rover_size_m) * 0.75),
            max(0.50, float(args.rover_size_m)),
            max(0.75, float(args.rover_size_m) * 1.5),
        )
        for step_m in step_candidates:
            ahead = pos + fwd * float(step_m)
            end_rc = map_world_to_grid(float(ahead[0]), float(ahead[2]))
            if end_rc is None:
                continue
            dr = int(end_rc[0]) - int(start_rc[0])
            dc = int(end_rc[1]) - int(start_rc[1])
            if dr != 0 or dc != 0:
                return np.array([float(dr), float(dc)], dtype=np.float32)
        return None

    def draw_rover_overlay(frame, rover_cell, cam_cell=None, heading_vec_rc=None):
        if frame is None:
            return
        if cam_cell is not None:
            display_cam = display_cell_for_map_cell(cam_cell[0], cam_cell[1], frame)
            if display_cam is not None:
                r0, c0 = display_cam
                half = max(1, int(args.map_camera_size) // 2)
                r1 = max(0, r0 - half)
                r2 = min(frame.shape[0], r0 + half + 1)
                c1 = max(0, c0 - half)
                c2 = min(frame.shape[1], c0 + half + 1)
                frame[r1:r2, c1:c2, :] = (255, 0, 0)
        if rover_cell is None:
            return
        display_rover = display_cell_for_map_cell(rover_cell[0], rover_cell[1], frame)
        if display_rover is None:
            return

        r0, c0 = display_rover
        cv2.circle(frame, (c0, r0), max(2, int(args.map_camera_size)), (0, 180, 255), -1)
        rover_half_cells = max(1.0, float(args.rover_size_m) / (2.0 * float(occ_map.map_res_m)))
        front_edge_pts = None
        nose_pt = None
        if heading_vec_rc is None:
            rr = int(round(rover_half_cells))
            box_pts = np.array(
                [
                    [c0 - rr, r0 - rr],
                    [c0 + rr, r0 - rr],
                    [c0 + rr, r0 + rr],
                    [c0 - rr, r0 + rr],
                ],
                dtype=np.int32,
            )
        else:
            fwd_v = np.array(heading_vec_rc, dtype=np.float32).reshape(2,)
            fwd_norm = float(np.linalg.norm(fwd_v))
            if fwd_norm > 1e-6:
                fwd_v /= fwd_norm
            right_v = np.array([fwd_v[1], -fwd_v[0]], dtype=np.float32)
            center = np.array([float(r0), float(c0)], dtype=np.float32)
            p1 = center + fwd_v * rover_half_cells + right_v * rover_half_cells
            p2 = center + fwd_v * rover_half_cells - right_v * rover_half_cells
            p3 = center - fwd_v * rover_half_cells - right_v * rover_half_cells
            p4 = center - fwd_v * rover_half_cells + right_v * rover_half_cells
            front_edge_pts = (
                (int(round(p1[1])), int(round(p1[0]))),
                (int(round(p2[1])), int(round(p2[0]))),
            )
            nose = center + fwd_v * (rover_half_cells + max(2.0, float(args.map_camera_size) * 0.8))
            nose_pt = (int(round(nose[1])), int(round(nose[0])))
            box_pts = np.array(
                [
                    [int(round(p1[1])), int(round(p1[0]))],
                    [int(round(p2[1])), int(round(p2[0]))],
                    [int(round(p3[1])), int(round(p3[0]))],
                    [int(round(p4[1])), int(round(p4[0]))],
                ],
                dtype=np.int32,
            )
            size = max(18, int(args.map_camera_size) * 8)
            end_pt = (
                int(round(c0 + float(fwd_v[1]) * size)),
                int(round(r0 + float(fwd_v[0]) * size)),
            )
            cv2.arrowedLine(frame, (c0, r0), end_pt, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
        cv2.polylines(frame, [box_pts], True, (0, 220, 255), 1, cv2.LINE_AA)
        if front_edge_pts is not None:
            cv2.line(frame, front_edge_pts[0], front_edge_pts[1], (0, 255, 120), 2, cv2.LINE_AA)
        if nose_pt is not None:
            cv2.circle(frame, nose_pt, 3, (0, 255, 120), -1)

    def on_map_click(event, x, y, flags, param):
        nonlocal goal_cell, path_cells, last_path_cells, last_start, last_goal
        nonlocal emergency_stop, last_path_plan_time, map_view_shift_r, map_view_shift_c, map_scale_live
        nonlocal path_plan_mode, mining_goal_active
        if last_map_window_shape is not None:
            x, y = window_to_image_coords("ZED Occupancy Map (XZ)", x, y, last_map_window_shape)
        if event == cv2.EVENT_RBUTTONDOWN:
            emergency_stop = True
            print("EMERGENCY STOP")
            return
        is_left_down = event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE) and (flags & cv2.EVENT_FLAG_LBUTTON)
        if not is_left_down and event != cv2.EVENT_LBUTTONDOWN:
            return
        scale = max(1, int(map_scale_live))
        row = int(y / scale) - int(map_view_shift_r)
        col = int(x / scale) - int(map_view_shift_c)
        if row < 0 or row >= occ_map.grid_h or col < 0 or col >= occ_map.grid_w:
            return
        if event == cv2.EVENT_LBUTTONDOWN and mining.state in (
            auto_mining.MiningState.DRAW_EXCAV,
            auto_mining.MiningState.DRAW_DEPOSIT,
            auto_mining.MiningState.PICK_DIG_START,
        ):
            if mining.consume_click(row, col, occ_map):
                return
        # Helper — compute brush bounding box.
        def _brush(r, c):
            return (
                max(0, r - paint_brush_radius),
                min(occ_map.grid_h - 1, r + paint_brush_radius),
                max(0, c - paint_brush_radius),
                min(occ_map.grid_w - 1, c + paint_brush_radius),
            )
        # Paint safe mode — brush marks cells permanently free (cyan).
        if paint_safe_mode:
            r0, r1, c0, c1 = _brush(row, col)
            occ_map.painted_free[r0:r1+1, c0:c1+1] = True
            # Clear any existing obstacle/hole counts so it shows safe immediately.
            occ_map.occ_counts[r0:r1+1, c0:c1+1] = 0.0
            occ_map.hole_counts[r0:r1+1, c0:c1+1] = 0.0
            return
        # Erase safe mode — brush removes painted-free cells.
        if erase_safe_mode:
            r0, r1, c0, c1 = _brush(row, col)
            occ_map.painted_free[r0:r1+1, c0:c1+1] = False
            return
        # Paint obstacle mode — brush forces cells to be obstacles.
        if paint_obstacle_mode:
            r0, r1, c0, c1 = _brush(row, col)
            # Stamp strong obstacle evidence; also clear painted-free so it
            # doesn't silently override the obstacle.
            occ_map.painted_free[r0:r1+1, c0:c1+1] = False
            occ_map.occ_counts[r0:r1+1, c0:c1+1] = 20.0
            occ_map.free_counts[r0:r1+1, c0:c1+1] = 0.0
            occ_map.hole_counts[r0:r1+1, c0:c1+1] = 0.0
            return
        # Normal click — only on LBUTTONDOWN (not drag).
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if mining.consume_click(row, col, occ_map):
            return
        if drive_calibration.active:
            drive_calibration.set_target(row, col)
            print(
                f"Drive calibration target armed at row={row}, col={col}. "
                "The rover will drive there and evaluate heading direction."
            )
        goal_cell = (row, col)
        path_cells = None
        last_path_cells = None
        last_start = None
        last_goal = None
        last_path_plan_time = 0.0
        path_plan_mode = "none"
        mining_goal_active = False
        emergency_stop = False
        print(f"New goal set at row={row}, col={col}")
        if tracking_enabled and (not tracking_pose_ok):
            print("Goal queued, but tracking is lost. Rover will wait here until tracking recovers.")

    def on_status_click(event, x, y, flags, param):
        nonlocal disable_holes, whole_map_enabled, smooth_map_enabled, map_scale_live, map_size_input_focused, map_size_input_text
        nonlocal dig_name_input_focused, dig_name_input_text
        nonlocal paint_safe_mode, erase_safe_mode, paint_obstacle_mode, paint_brush_radius
        nonlocal reset_map_confirm, status_scroll_drag_active, status_scroll_drag_offset
        nonlocal status_view_drag_active, status_view_drag_anchor_y, status_view_drag_anchor_scroll
        nonlocal manual_mode, manual_fwd, manual_turn, emergency_stop
        nonlocal demo_rover_pos_map

        def scroll_from_thumb_top(track_rect, thumb_rect, thumb_top):
            if track_rect is None or thumb_rect is None:
                return
            track_y0, track_y1 = int(track_rect[1]), int(track_rect[3])
            thumb_h = max(1, int(thumb_rect[3] - thumb_rect[1]))
            max_thumb_top = max(track_y0, track_y1 - thumb_h)
            clamped_top = max(track_y0, min(max_thumb_top, int(thumb_top)))
            travel = max(1, max_thumb_top - track_y0)
            frac = 0.0 if status_scroll_max <= 0 else float(clamped_top - track_y0) / float(travel)
            set_status_scroll_to(int(round(frac * float(status_scroll_max))))

        if last_status_panel_shape is not None:
            x, y = window_to_image_coords("ZED Drive Status", x, y, last_status_panel_shape)
        if event == cv2.EVENT_LBUTTONUP:
            status_scroll_drag_active = False
            status_view_drag_active = False
            return
        if event == cv2.EVENT_MOUSEMOVE and not (flags & cv2.EVENT_FLAG_LBUTTON):
            status_scroll_drag_active = False
            status_view_drag_active = False
        if event == getattr(cv2, "EVENT_MOUSEWHEEL", -9999):
            try:
                wheel_delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                wheel_delta = 1 if flags > 0 else -1
            set_status_scroll(-80 if wheel_delta > 0 else 80)
            return
        is_drag = event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)
        track_rect = status_button_rects.get("scrollbar_track")
        thumb_rect = status_button_rects.get("scrollbar_thumb")
        if is_drag and status_scroll_drag_active and track_rect is not None and thumb_rect is not None:
            scroll_from_thumb_top(track_rect, thumb_rect, y - status_scroll_drag_offset)
            return
        if is_drag and status_view_drag_active:
            set_status_scroll_to(int(status_view_drag_anchor_scroll + (status_view_drag_anchor_y - y)))
            return
        if event != cv2.EVENT_LBUTTONDOWN and not is_drag:
            return

        if reset_map_confirm:
            rect = status_button_rects.get("reset_confirm")
            if rect is not None:
                x0, y0, x1, y1 = rect
                if x0 <= x <= x1 and y0 <= y <= y1:
                    reset_map_memory()
                    return
            rect = status_button_rects.get("reset_cancel")
            if rect is not None:
                x0, y0, x1, y1 = rect
                if x0 <= x <= x1 and y0 <= y <= y1:
                    reset_map_confirm = False
                    print("Map reset canceled.")
                    return
            return

        rect = status_button_rects.get("scroll_up")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_status_scroll(-90)
                return
        rect = status_button_rects.get("scroll_down")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_status_scroll(90)
                return
        if event == cv2.EVENT_LBUTTONDOWN and track_rect is not None and thumb_rect is not None:
            tx0, ty0, tx1, ty1 = track_rect
            if tx0 <= x <= tx1 and ty0 <= y <= ty1:
                hx0, hy0, hx1, hy1 = thumb_rect
                thumb_h = max(1, hy1 - hy0)
                if hx0 <= x <= hx1 and hy0 <= y <= hy1:
                    status_scroll_drag_active = True
                    status_scroll_drag_offset = y - hy0
                else:
                    status_scroll_drag_active = True
                    status_scroll_drag_offset = thumb_h // 2
                    scroll_from_thumb_top(track_rect, thumb_rect, y - status_scroll_drag_offset)
                return
        viewport_rect = status_button_rects.get("controls_viewport")
        for jump_name, target_y in status_section_jump_targets.items():
            rect = status_button_rects.get(jump_name)
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_status_scroll_to(int(target_y))
                return
        # Brush size slider supports drag.
        rect = status_button_rects.get("brush_slider")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                frac = (x - x0) / max(1, x1 - x0)
                paint_brush_radius = max(1, min(15, int(round(1 + frac * 14))))
                return
        rect = status_button_rects.get("drive_speed_slider")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                frac = (x - x0) / max(1, x1 - x0)
                set_drive_speed(0.10 + frac * 0.90, "slider")
                return
        rect = status_button_rects.get("turn_speed_slider")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                frac = (x - x0) / max(1, x1 - x0)
                set_turn_speed(0.20 + frac * 0.80, "slider")
                return
        rect = status_button_rects.get("digger_speed_slider")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                frac = (x - x0) / max(1, x1 - x0)
                set_digger_speed(0.10 + frac * 0.90, "slider")
                return
        rect = status_button_rects.get("manual_mode_toggle")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_manual_drive_mode(not manual_mode, "setup button")
                return
        rect = status_button_rects.get("no_mapping_mode")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_no_mapping_mode(not no_mapping_mode, "setup button")
                return
        rect = status_button_rects.get("setup_low_latency_mode")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_low_latency_mode(not low_latency_mode, "setup button")
                return
        if is_drag:
            return
        rect = status_button_rects.get("dig_name_input")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                dig_name_input_focused = True
                map_size_input_focused = False
                return
            else:
                dig_name_input_focused = False
        # Check if the map size input field was clicked
        rect = status_button_rects.get("map_size_input")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                map_size_input_focused = True
                dig_name_input_focused = False
                map_size_input_text = ""
                return
            else:
                map_size_input_focused = False
        mining_running = mining.state in (
            auto_mining.MiningState.PLAN_SWEEP,
            auto_mining.MiningState.NAVIGATE_DIG,
            auto_mining.MiningState.DIGGING,
            auto_mining.MiningState.BACKUP,
            auto_mining.MiningState.NAVIGATE_DEPOSIT,
            auto_mining.MiningState.DEPOSITING,
        )
        rect = status_button_rects.get("auto_run")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                if mining_running:
                    mining.abort()
                    clear_navigation_goal()
                    manual_mode = False
                    manual_fwd = 0.0
                    manual_turn = 0.0
                    print("Auto Run: ABORTED via button")
                else:
                    clear_navigation_goal()
                    emergency_stop = False
                    manual_mode = False
                    manual_fwd = 0.0
                    manual_turn = 0.0
                    set_no_mapping_mode(False, "auto run button")
                    suppress_driver_priority(1.25, "auto run button")
                    if demo_auto_enabled:
                        demo_rover_pos_map = None
                    mining.start_run()
                    print("Auto Run: START requested via button")
                return
        rect = status_button_rects.get("direct_nav")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_direct_nav_enabled(not direct_nav_enabled, "button")
                return
        rect = status_button_rects.get("auto_digger")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_excavation_test_mode("auto_digger", not auto_digger_enabled, "button")
                return
        rect = status_button_rects.get("test_excavation_dig")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_excavation_test_mode("dig", not test_excavation_dig_active, "button")
                return
        rect = status_button_rects.get("test_excavation_left_extend")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_excavation_test_mode("left_extend", not test_excavation_left_extend_active, "button")
                return
        rect = status_button_rects.get("test_excavation_right_extend")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_excavation_test_mode("right_extend", not test_excavation_right_extend_active, "button")
                return
        rect = status_button_rects.get("test_excavation_lower")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_excavation_test_mode("lower", not test_excavation_lower_active, "button")
                return
        rect = status_button_rects.get("test_excavation_pattern")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_excavation_pattern_test(not excavation_pattern_test_active, "button")
                return
        rect = status_button_rects.get("door_open")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_excavation_test_mode("door_open", not test_door_open_active, "button")
                return
        rect = status_button_rects.get("door_close")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_excavation_test_mode("door_close", not test_door_close_active, "button")
                return
        rect = status_button_rects.get("stop_actuators")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                stop_all_actuators("button")
                return
        rect = status_button_rects.get("reset_map")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                reset_map_confirm = True
                print("Confirm map reset in the status panel.")
                return
        rect = status_button_rects.get("lock_green")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                lock_green_zones_permanent()
                return
        rect = status_button_rects.get("main_rover_mode")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_main_rover_mode(not args.main_rover_mode)
                return
        rect = status_button_rects.get("camera_view")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                toggle_camera_view()
                return
        rect = status_button_rects.get("camera_overlay")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_camera_overlay_enabled(not camera_overlay_enabled, "button")
                return
        rect = status_button_rects.get("human_detect_toggle")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_human_detect_enabled(not human_detect_enabled, "button")
                return
        rect = status_button_rects.get("rock_detect_toggle")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_rock_detect_enabled(not rock_detect_enabled, "button")
                return
        rect = status_button_rects.get("low_latency_mode")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_low_latency_mode(not low_latency_mode, "button")
                return
        rect = status_button_rects.get("drive_heading_flip")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_drive_heading_flip(not args.drive_heading_flip, "button")
                return
        rect = status_button_rects.get("hard_drive_flip")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_hard_drive_flip(not args.hard_drive_flip, "button")
                return
        rect = status_button_rects.get("steering_flip")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_steering_flip(not args.steering_flip, "button")
                return
        rect = status_button_rects.get("bidirectional_auto")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_bidirectional_auto(not bidirectional_auto_enabled, "button")
                return
        rect = status_button_rects.get("demo_auto")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_demo_auto(not demo_auto_enabled, "button")
                return
        rect = status_button_rects.get("lock_start_frame")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                request_start_frame_lock("button")
                return
        rect = status_button_rects.get("scan_start_frame")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                request_start_frame_scan("button")
                return
        rect = status_button_rects.get("test_drive_forward")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_test_drive_forward(not test_drive_forward_active, "button")
                return
        rect = status_button_rects.get("camera_view_flip")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                flip_camera_view_calibration("button")
                return
        rect = status_button_rects.get("display_heading_flip")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_display_heading_flip(not args.display_heading_flip, "button")
                return
        rect = status_button_rects.get("drive_calibration_mode")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_drive_calibration_mode(not drive_calibration.active, "button")
                return
        rect = status_button_rects.get("drive_calibration_cancel")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                drive_calibration.clear_target("Calibration target cleared.")
                if drive_calibration.active:
                    clear_navigation_goal()
                print(drive_calibration.last_result)
                publish_map_ui_state(force=True)
                return
        rect = status_button_rects.get("dig_style_cycle")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                cycle_dig_style("button")
                return
        rect = status_button_rects.get("dig_phase_cycle")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                cycle_dig_phase("button")
                return
        rect = status_button_rects.get("dig_record_active")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                start_active_dig_recording("button")
                return
        rect = status_button_rects.get("dig_profile_preview")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                if dig_profile_preview_active:
                    stop_dig_profile_preview("button")
                else:
                    start_dig_profile_preview("button")
                return
        rect = status_button_rects.get("dig_record_stop")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                if dig_profiles.recording:
                    stop_dig_recording(True, "button")
                else:
                    stop_dig_profile_preview("button")
                return
        rect = status_button_rects.get("dig_profile_prev")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                cycle_dig_profile_cursor(-1, "button")
                return
        rect = status_button_rects.get("dig_profile_next")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                cycle_dig_profile_cursor(1, "button")
                return
        rect = status_button_rects.get("dig_profile_use")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                use_browsed_dig_profile("button")
                return
        rect = status_button_rects.get("dig_profile_delete")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                delete_browsed_dig_profile("button")
                return
        rect = status_button_rects.get("dig_profiles_view_all")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_show_all_dig_profiles(not show_all_dig_profiles, "button")
                return
        rect = status_button_rects.get("controller_record")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                start_controller_recording("button")
                return
        rect = status_button_rects.get("controller_preview")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                if controller_macro_preview_active:
                    stop_controller_macro_preview("button")
                else:
                    start_controller_macro_preview("button")
                return
        rect = status_button_rects.get("controller_cycle")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                if controller_cycle_preview_active:
                    stop_controller_cycle_preview("button")
                else:
                    start_controller_cycle_preview("button")
                return
        rect = status_button_rects.get("controller_stop")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                if controller_macros.recording:
                    stop_controller_recording(True, "button")
                elif controller_cycle_preview_active:
                    stop_controller_cycle_preview("button")
                else:
                    stop_controller_macro_preview("button")
                return
        rect = status_button_rects.get("controller_prev")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                cycle_controller_macro_cursor(-1, "button")
                return
        rect = status_button_rects.get("controller_next")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                cycle_controller_macro_cursor(1, "button")
                return
        rect = status_button_rects.get("controller_use")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                use_browsed_controller_macro("button")
                return
        rect = status_button_rects.get("excav")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_brush_tool(None)
                mining.start_draw_excavation()
                return
        rect = status_button_rects.get("deposit")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_brush_tool(None)
                mining.start_draw_deposit()
                return
        rect = status_button_rects.get("starting_zone")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_brush_tool(None)
                mining.set_starting_zone_preset(occ_map)
                return
        rect = status_button_rects.get("set_berm_left")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_brush_tool(None)
                mining.set_deposit_zone_preset("left", occ_map)
                return
        rect = status_button_rects.get("set_berm_right")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_brush_tool(None)
                mining.set_deposit_zone_preset("right", occ_map)
                return
        rect = status_button_rects.get("pick_dig_start")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                if not mining_running and mining.excav_corners_rc:
                    set_brush_tool(None)
                    mining.start_pick_dig_start()
                return
        button_enabled = mining_buttons_enabled()
        if not button_enabled:
            return
        rect = status_button_rects.get("whole")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                whole_map_enabled = not whole_map_enabled
                print(f"Whole-map mode {'ENABLED' if whole_map_enabled else 'DISABLED'}")
                return
        rect = status_button_rects.get("holes")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                disable_holes = not disable_holes
                print(f"Hole detection {'DISABLED' if disable_holes else 'ENABLED'}")
                return
        rect = status_button_rects.get("paint_safe")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                next_active = not paint_safe_mode
                set_brush_tool("paint_safe" if next_active else None)
                print(f"Paint Safe mode {'ON — click/drag map to lock cells safe' if paint_safe_mode else 'OFF'}")
                return
        rect = status_button_rects.get("erase_safe")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                next_active = not erase_safe_mode
                set_brush_tool("erase_safe" if next_active else None)
                print(f"Erase Safe mode {'ON — click/drag map to remove painted cells' if erase_safe_mode else 'OFF'}")
                return
        rect = status_button_rects.get("paint_obstacle")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                next_active = not paint_obstacle_mode
                set_brush_tool("paint_obstacle" if next_active else None)
                print(f"Paint Obstacle mode {'ON — click/drag map to force obstacle cells' if paint_obstacle_mode else 'OFF'}")
                return
        rect = status_button_rects.get("smooth_map")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                smooth_map_enabled = not smooth_map_enabled
                print(f"Smooth Map {'ENABLED — noise pixels filtered' if smooth_map_enabled else 'DISABLED'}")
                return
        rect = status_button_rects.get("clear_paint")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                clear_manual_paint()
                return
        rect = status_button_rects.get("brush_minus")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                paint_brush_radius = max(1, paint_brush_radius - 1)
                print(f"Brush radius: {paint_brush_radius} cells")
                return
        rect = status_button_rects.get("brush_plus")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                paint_brush_radius = min(15, paint_brush_radius + 1)
                print(f"Brush radius: {paint_brush_radius} cells")
                return
        if viewport_rect is not None:
            vx0, vy0, vx1, vy1 = viewport_rect
            if vx0 <= x <= vx1 and vy0 <= y <= vy1:
                status_view_drag_active = True
                status_view_drag_anchor_y = int(y)
                status_view_drag_anchor_scroll = int(status_scroll_y)
                return
    def process_external_map_command():
        nonlocal last_map_command_seq, reset_map_confirm, paint_brush_radius
        nonlocal demo_rover_pos_map
        try:
            if not args.map_command_file or (not os.path.exists(args.map_command_file)):
                return
            with open(args.map_command_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            return

        seq = int(payload.get("seq", 0) or 0)
        if seq <= last_map_command_seq:
            return

        cmd_type = str(payload.get("type", "") or "")
        if cmd_type == "set_goal_click":
            display_x = payload.get("display_x")
            display_y = payload.get("display_y")
            if display_x is None or display_y is None:
                last_map_command_seq = seq
                return

            on_map_click(cv2.EVENT_LBUTTONDOWN, int(display_x), int(display_y), 0, None)
            source = payload.get("source")
            if source:
                print(f"Processed external waypoint from {source} (x={display_x}, y={display_y})")
            last_map_command_seq = seq
            return

        if cmd_type != "ui_action":
            last_map_command_seq = seq
            return

        action = str(payload.get("action", "") or "").strip()
        if action == "auto_run":
            mining_running = mining.state in (
                auto_mining.MiningState.PLAN_SWEEP,
                auto_mining.MiningState.NAVIGATE_DIG,
                auto_mining.MiningState.DIGGING,
                auto_mining.MiningState.BACKUP,
                auto_mining.MiningState.NAVIGATE_DEPOSIT,
                auto_mining.MiningState.DEPOSITING,
            )
            clear_navigation_goal()
            manual_mode = False
            manual_fwd = 0.0
            manual_turn = 0.0
            if mining_running:
                mining.abort()
                print("Auto Run: ABORTED via external command")
            else:
                emergency_stop = False
                set_no_mapping_mode(False, "auto run external command")
                suppress_driver_priority(1.25, "auto run external command")
                if demo_auto_enabled:
                    demo_rover_pos_map = None
                mining.start_run()
                print("Auto Run: START requested via external command")
        elif action == "paint_safe":
            set_brush_tool(None if paint_safe_mode else "paint_safe")
            print(f"Paint Safe mode {'ON — click/drag map to lock cells safe' if paint_safe_mode else 'OFF'}")
        elif action == "erase_safe":
            set_brush_tool(None if erase_safe_mode else "erase_safe")
            print(f"Erase Safe mode {'ON — click/drag map to remove painted cells' if erase_safe_mode else 'OFF'}")
        elif action == "paint_obstacle":
            set_brush_tool(None if paint_obstacle_mode else "paint_obstacle")
            print(f"Paint Obstacle mode {'ON — click/drag map to force obstacle cells' if paint_obstacle_mode else 'OFF'}")
        elif action == "clear_all":
            clear_manual_paint()
        elif action == "lock_green":
            lock_green_zones_permanent()
        elif action == "reset_map":
            reset_map_confirm = True
            print("Confirm map reset in the status panel.")
        elif action == "reset_confirm":
            if reset_map_confirm:
                reset_map_memory()
        elif action == "reset_cancel":
            if reset_map_confirm:
                reset_map_confirm = False
                print("Map reset canceled.")
        elif action == "localize_scan":
            print("Localization scan has been removed; ignoring external command.")
        elif action == "direct_nav":
            set_direct_nav_enabled(not direct_nav_enabled, "external command")
        elif action == "auto_digger":
            set_excavation_test_mode("auto_digger", not auto_digger_enabled, "external command")
        elif action == "test_excavation_dig":
            set_excavation_test_mode("dig", not test_excavation_dig_active, "external command")
        elif action == "test_excavation_left_extend":
            set_excavation_test_mode("left_extend", not test_excavation_left_extend_active, "external command")
        elif action == "test_excavation_right_extend":
            set_excavation_test_mode("right_extend", not test_excavation_right_extend_active, "external command")
        elif action == "test_excavation_lower":
            set_excavation_test_mode("lower", not test_excavation_lower_active, "external command")
        elif action == "test_excavation_pattern":
            set_excavation_pattern_test(not excavation_pattern_test_active, "external command")
        elif action == "door_open":
            set_excavation_test_mode("door_open", not test_door_open_active, "external command")
        elif action == "door_close":
            set_excavation_test_mode("door_close", not test_door_close_active, "external command")
        elif action == "stop_actuators":
            stop_all_actuators("external command")
        elif action == "main_rover_mode":
            set_main_rover_mode(not args.main_rover_mode)
        elif action == "set_control_mode":
            requested_mode = str(payload.get("mode", "") or "").strip().lower()
            if requested_mode == "manual":
                set_manual_drive_mode(True, "external command")
            elif requested_mode == "autonomy":
                set_manual_drive_mode(False, "external command")
            else:
                print(f"Ignoring unsupported control mode command: {requested_mode}")
        elif action == "camera_view":
            toggle_camera_view()
        elif action == "camera_overlay":
            set_camera_overlay_enabled(not camera_overlay_enabled, "external command")
        elif action == "human_detect_toggle":
            set_human_detect_enabled(not human_detect_enabled, "external command")
        elif action == "rock_detect_toggle":
            set_rock_detect_enabled(not rock_detect_enabled, "external command")
        elif action == "low_latency_mode":
            set_low_latency_mode(not low_latency_mode, "external command")
        elif action == "drive_heading_flip":
            set_drive_heading_flip(not args.drive_heading_flip, "external command")
        elif action == "hard_drive_flip":
            set_hard_drive_flip(not args.hard_drive_flip, "external command")
        elif action == "steering_flip":
            set_steering_flip(not args.steering_flip, "external command")
        elif action == "bidirectional_auto":
            set_bidirectional_auto(not bidirectional_auto_enabled, "external command")
        elif action == "demo_auto":
            set_demo_auto(not demo_auto_enabled, "external command")
        elif action == "lock_start_frame":
            request_start_frame_lock("external command")
        elif action == "scan_start_frame":
            request_start_frame_scan("external command")
        elif action == "test_drive_forward":
            set_test_drive_forward(not test_drive_forward_active, "external command")
        elif action == "camera_view_flip":
            flip_camera_view_calibration("external command")
        elif action == "display_heading_flip":
            set_display_heading_flip(not args.display_heading_flip, "external command")
        elif action == "drive_calibration_mode":
            set_drive_calibration_mode(not drive_calibration.active, "external command")
        elif action == "drive_calibration_cancel":
            drive_calibration.clear_target("Calibration target cleared.")
            if drive_calibration.active:
                clear_navigation_goal()
            print(drive_calibration.last_result)
        elif action == "dig_style_cycle":
            cycle_dig_style("external command")
        elif action == "dig_phase_cycle":
            cycle_dig_phase("external command")
        elif action == "dig_record_active":
            start_active_dig_recording("external command")
        elif action == "dig_profile_preview":
            if dig_profile_preview_active:
                stop_dig_profile_preview("external command")
            else:
                start_dig_profile_preview("external command")
        elif action == "dig_record_dig":
            start_dig_recording(dig_profiles.active_style, "dig", "external command")
        elif action == "dig_record_retract":
            start_dig_recording(dig_profiles.active_style, "retract", "external command")
        elif action == "dig_record_short":
            start_dig_recording("short", "dig", "external command")
        elif action == "dig_record_long":
            start_dig_recording("long", "dig", "external command")
        elif action == "dig_record_stop":
            if dig_profiles.recording:
                stop_dig_recording(True, "external command")
            else:
                stop_dig_profile_preview("external command")
        elif action == "dig_profile_prev":
            cycle_dig_profile_cursor(-1, "external command")
        elif action == "dig_profile_next":
            cycle_dig_profile_cursor(1, "external command")
        elif action == "dig_profile_use":
            use_browsed_dig_profile("external command")
        elif action == "dig_profile_delete":
            delete_browsed_dig_profile("external command")
        elif action == "controller_record":
            start_controller_recording("external command")
        elif action == "controller_preview":
            if controller_macro_preview_active:
                stop_controller_macro_preview("external command")
            else:
                start_controller_macro_preview("external command")
        elif action == "controller_cycle":
            if controller_cycle_preview_active:
                stop_controller_cycle_preview("external command")
            else:
                start_controller_cycle_preview("external command")
        elif action == "controller_stop":
            if controller_macros.recording:
                stop_controller_recording(True, "external command")
            elif controller_cycle_preview_active:
                stop_controller_cycle_preview("external command")
            else:
                stop_controller_macro_preview("external command")
        elif action == "controller_prev":
            cycle_controller_macro_cursor(-1, "external command")
        elif action == "controller_next":
            cycle_controller_macro_cursor(1, "external command")
        elif action == "controller_use":
            use_browsed_controller_macro("external command")
        elif action == "dig_profiles_view_all":
            set_show_all_dig_profiles(not show_all_dig_profiles, "external command")
        elif action == "draw_excav_zone":
            if mining_buttons_enabled():
                set_brush_tool(None)
                mining.start_draw_excavation()
        elif action == "draw_deposit_zone":
            if mining_buttons_enabled():
                set_brush_tool(None)
                mining.start_draw_deposit()
        elif action == "set_starting_zone":
            if mining_buttons_enabled():
                set_brush_tool(None)
                mining.set_starting_zone_preset(occ_map)
        elif action == "set_berm_left":
            if mining_buttons_enabled():
                set_brush_tool(None)
                mining.set_deposit_zone_preset("left", occ_map)
        elif action == "set_berm_right":
            if mining_buttons_enabled():
                set_brush_tool(None)
                mining.set_deposit_zone_preset("right", occ_map)
        elif action == "pick_dig_start":
            if mining_buttons_enabled() and mining.excav_corners_rc:
                set_brush_tool(None)
                mining.start_pick_dig_start()
                mining.handle_key(ord("d"))
        elif action == "brush_minus":
            paint_brush_radius = max(1, paint_brush_radius - 1)
            print(f"Brush radius: {paint_brush_radius} cells")
        elif action == "brush_plus":
            paint_brush_radius = min(15, paint_brush_radius + 1)
            print(f"Brush radius: {paint_brush_radius} cells")
        elif action == "set_brush_radius":
            try:
                value = int(payload.get("value", paint_brush_radius))
            except Exception:
                value = paint_brush_radius
            paint_brush_radius = max(1, min(15, value))
            print(f"Brush radius: {paint_brush_radius} cells")
        elif action == "set_drive_speed":
            try:
                value = float(payload.get("value", args.drive_speed))
            except Exception:
                value = float(args.drive_speed)
            set_drive_speed(value, "external command")
        elif action == "set_turn_speed":
            try:
                value = float(payload.get("value", args.drive_max_turn_cmd))
            except Exception:
                value = float(args.drive_max_turn_cmd)
            set_turn_speed(value, "external command")

        last_map_command_seq = seq

    def send_nt_command(enabled, fwd, turn, duration):
        nonlocal nt_command_seq, nt_ready_stuck_since, nt_last_auto_push
        nonlocal nt_ready_high, nt_ready_clear_time, last_drive_debug_time
        nonlocal status_cmd_enabled, status_cmd_fwd, status_cmd_turn, status_cmd_duration
        nonlocal test_excavation_lower_active, test_excavation_lower_cycle_started_at
        if sd is None:
            return
        now = time.time()
        fwd = float(fwd)
        turn = float(turn)
        if args.hard_drive_flip:
            fwd = -fwd
            turn = -turn
        if args.steering_flip:
            turn = -turn
        enabled = bool(enabled)
        status_cmd_enabled = bool(enabled)
        status_cmd_fwd = float(fwd)
        status_cmd_turn = float(turn)
        status_cmd_duration = float(duration)

        def push_automation_state(force=False):
            nonlocal nt_last_auto_push
            nonlocal test_excavation_lower_active, test_excavation_lower_cycle_started_at
            pattern_state = excavation_pattern_state(now)
            if pattern_state is not None and pattern_state.get("done"):
                pattern_state = None
            auto_excavation_pattern = None
            if enabled and mining.state == auto_mining.MiningState.DIGGING:
                auto_excavation_pattern = mining.excavation_pattern_command(now)
                if auto_excavation_pattern is not None and bool(auto_excavation_pattern.get("done")):
                    auto_excavation_pattern = None
            if (not force) and (now - nt_last_auto_push) < max(0.02, float(args.nt_enable_heartbeat_sec)):
                return
            lower_cycle_elapsed = 0.0
            lower_cycle_active = bool(test_excavation_lower_active)
            if lower_cycle_active:
                if test_excavation_lower_cycle_started_at <= 0.0:
                    test_excavation_lower_cycle_started_at = now
                lower_cycle_elapsed = max(0.0, now - float(test_excavation_lower_cycle_started_at))
                if lower_cycle_elapsed >= 10.0:
                    test_excavation_lower_active = False
                    test_excavation_lower_cycle_started_at = 0.0
                    lower_cycle_active = False
                    lower_cycle_elapsed = 0.0
                    print("Excavation lower cycle completed.")
                    publish_map_ui_state(force=True)
            mining_state_value = mining.state.value
            auto_dig_active = auto_digger_enabled and enabled and mining.state == auto_mining.MiningState.DIGGING
            playback_cmd = controller_macro_playback_cmd if controller_macro_playback_cmd is not None else (
                dig_profile_playback_cmd if (
                enabled
                and dig_profile_preview_active
                ) else None
            )
            excavator_enabled = test_excavation_dig_active or (
                bool(playback_cmd.get("digger_on")) if playback_cmd is not None else auto_dig_active
            )
            if pattern_state is not None and bool(pattern_state.get("digger")):
                excavator_enabled = True
            if auto_excavation_pattern is not None and playback_cmd is None:
                excavator_enabled = excavator_enabled or bool(auto_excavation_pattern.get("digger"))
            if excavator_enabled:
                digger_pwm_period_sec = 0.25
                digger_phase = (now % digger_pwm_period_sec) / digger_pwm_period_sec
                excavator_enabled = bool(digger_phase < max(0.10, min(1.00, float(digger_speed_scale))))
            excavator_lower_requested = (lower_cycle_active and lower_cycle_elapsed < 5.0) or (
                bool(playback_cmd.get("lower_on")) if playback_cmd is not None else auto_dig_active
            )
            if pattern_state is not None:
                excavator_lower_requested = excavator_lower_requested or bool(pattern_state.get("lower"))
            if auto_excavation_pattern is not None and playback_cmd is None:
                excavator_lower_requested = excavator_lower_requested or bool(auto_excavation_pattern.get("lower"))
            conveyor_enabled = enabled and mining.state == auto_mining.MiningState.DEPOSITING
            if playback_cmd is not None:
                left_extend_enabled = test_excavation_left_extend_active or bool(playback_cmd.get("left_extend_on", False))
                right_extend_enabled = test_excavation_right_extend_active or bool(playback_cmd.get("right_extend_on", False))
            else:
                left_extend_enabled = test_excavation_left_extend_active
                right_extend_enabled = test_excavation_right_extend_active
            if auto_excavation_pattern is not None and playback_cmd is None:
                left_extend_enabled = left_extend_enabled or bool(auto_excavation_pattern.get("left_extend"))
                right_extend_enabled = right_extend_enabled or bool(auto_excavation_pattern.get("right_extend"))
            door_open_enabled = bool(test_door_open_active)
            door_close_enabled = bool(test_door_close_active)
            if playback_cmd is not None:
                door_open_enabled = door_open_enabled or bool(playback_cmd.get("door_open_on", False))
                door_close_enabled = door_close_enabled or bool(playback_cmd.get("door_close_on", False))
            if enabled and not (door_open_enabled or door_close_enabled):
                if mining.state == auto_mining.MiningState.DEPOSITING:
                    door_open_enabled = True
                elif mining.state in (
                    auto_mining.MiningState.NAVIGATE_DIG,
                    auto_mining.MiningState.DIGGING,
                    auto_mining.MiningState.BACKUP,
                    auto_mining.MiningState.NAVIGATE_DEPOSIT,
                ):
                    door_close_enabled = True
            sd.putBoolean("Drive/UseMainRoverControls", bool(args.main_rover_mode))
            sd.putBoolean("Drive/MainRoverDebugMode", bool(args.main_rover_debug))
            sd.putBoolean("Drive/MainRoverEmergencyStop", False)
            sd.putBoolean("Drive/MainRoverHardFlip", bool(args.hard_drive_flip))
            mechanism_request_active = bool(
                test_excavation_dig_active
                or lower_cycle_active
                or test_excavation_left_extend_active
                or test_excavation_right_extend_active
                or door_open_enabled
                or door_close_enabled
                or pattern_state is not None
                or auto_excavation_pattern is not None
            )
            automation_request_active = bool(enabled or mechanism_request_active)
            sd.putBoolean("Jetson/AutomationEnabled", automation_request_active)
            sd.putString("Jetson/MiningState", mining_state_value)
            sd.putBoolean("Jetson/ExcavatorEnabled", bool(excavator_enabled))
            sd.putBoolean("Jetson/ConveyorEnabled", bool(conveyor_enabled))
            sd.putBoolean("Jetson/ExcavatorLoweringSim", bool(excavator_lower_requested))
            sd.putBoolean("Jetson/ExcavatorLeftExtend", bool(left_extend_enabled))
            sd.putBoolean("Jetson/ExcavatorRightExtend", bool(right_extend_enabled))
            sd.putBoolean("Jetson/DoorActuatorsOpen", bool(door_open_enabled and not door_close_enabled))
            sd.putBoolean("Jetson/DoorActuatorsClose", bool(door_close_enabled and not door_open_enabled))
            # Robot-side code may scale command by these keys.
            if automation_request_active:
                sd.putNumber("Jetson/Speed", float(args.nt_forward_scale))
                sd.putNumber("Jetson/TurnSpeed", float(args.nt_turn_scale))
            else:
                sd.putNumber("Jetson/Speed", 0.0)
                sd.putNumber("Jetson/TurnSpeed", 0.0)
            nt_last_auto_push = now

        push_automation_state(force=not enabled)
        if not enabled:
            # Clear string-based legacy command channels when auto-drive is off.
            sd.putString("Jetson/Command", "")
            sd.putBoolean("Jetson/CommandReady", False)
            sd.putNumber("Jetson/CommandForward", 0.0)
            sd.putNumber("Jetson/CommandTurn", 0.0)
            nt_ready_high = False
            nt_ready_stuck_since = 0.0
            return

        if demo_auto_enabled:
            advance_demo_rover(fwd, turn, duration)
            sd.putString("Jetson/Command", "")
            sd.putBoolean("Jetson/CommandReady", False)
            sd.putNumber("Jetson/CommandForward", 0.0)
            sd.putNumber("Jetson/CommandTurn", 0.0)
            nt_ready_high = False
            nt_ready_stuck_since = 0.0
            return

        # Avoid stomping in-flight commands if robot has not consumed CommandReady yet.
        remote_ready = sd.getBoolean("Jetson/CommandReady", False)
        if remote_ready and not nt_ready_high:
            if nt_ready_stuck_since <= 0.0:
                nt_ready_stuck_since = now
            if (now - nt_ready_stuck_since) >= max(0.05, float(args.nt_command_ack_timeout_sec)):
                if args.drive_debug:
                    print("Warning: stale CommandReady detected; clearing flag.")
                sd.putBoolean("Jetson/CommandReady", False)
                nt_ready_stuck_since = 0.0
            else:
                return
        else:
            nt_ready_stuck_since = 0.0

        nt_command_seq += 1
        sd.putNumber("Jetson/CommandForward", float(fwd))
        sd.putNumber("Jetson/CommandTurn", float(turn))
        sd.putNumber("Jetson/CommandDuration", float(duration))
        sd.putNumber("Jetson/CommandSeq", float(nt_command_seq))

        # Pulse CommandReady high, then clear shortly after.
        # Keep it near configured value, but ensure a short low interval exists each cycle.
        cmd_period = max(0.02, float(duration))
        pulse_sec = max(0.01, float(args.drive_ready_pulse_sec))
        pulse_sec = min(pulse_sec, max(0.01, cmd_period - 0.01))
        if not nt_ready_high:
            sd.putBoolean("Jetson/CommandReady", True)
            nt_ready_high = True
            nt_ready_clear_time = now + pulse_sec
        if args.drive_debug:
            if (now - last_drive_debug_time) >= 0.2:
                print(
                    f"NT cmd seq={nt_command_seq:.0f} enabled={bool(enabled)} "
                    f"fwd={float(fwd):+.2f} turn={float(turn):+.2f} "
                    f"dur={float(duration):.2f} pulse={pulse_sec:.2f}"
                )
                last_drive_debug_time = now

    def reset_auto_drive_shape(now=None):
        nonlocal last_auto_fwd_cmd, last_auto_turn_cmd, last_auto_turn_time
        if now is None:
            now = time.time()
        last_auto_fwd_cmd = 0.0
        last_auto_turn_cmd = 0.0
        last_auto_turn_time = float(now)

    def apply_auto_drive_shape(fwd_target, turn_target, now):
        nonlocal last_auto_fwd_cmd, last_auto_turn_cmd, last_auto_turn_time
        dt_cmd = max(1e-3, float(now) - float(last_auto_turn_time))
        max_fwd_step = max(0.0, float(args.drive_forward_slew_per_sec)) * dt_cmd
        max_turn_step = max(0.0, float(args.drive_turn_slew_per_sec)) * dt_cmd
        fwd_target = max(-1.0, min(1.0, float(fwd_target)))
        turn_target = max(-1.0, min(1.0, float(turn_target)))

        delta_fwd = fwd_target - last_auto_fwd_cmd
        if delta_fwd > max_fwd_step:
            fwd = last_auto_fwd_cmd + max_fwd_step
        elif delta_fwd < -max_fwd_step:
            fwd = last_auto_fwd_cmd - max_fwd_step
        else:
            fwd = fwd_target

        delta_turn = turn_target - last_auto_turn_cmd
        if delta_turn > max_turn_step:
            turn = last_auto_turn_cmd + max_turn_step
        elif delta_turn < -max_turn_step:
            turn = last_auto_turn_cmd - max_turn_step
        else:
            turn = turn_target

        last_auto_fwd_cmd = float(fwd)
        last_auto_turn_cmd = float(turn)
        last_auto_turn_time = float(now)
        return float(fwd), float(turn)

    def mix_ds_drive(fwd, turn):
        if not args.ds_joystick:
            return float(fwd), float(turn)
        if args.main_rover_mode:
            return float(fwd), float(turn)
        mixed_fwd = max(-1.0, min(1.0, float(fwd) + float(ds_joystick_fwd)))
        mixed_turn = max(-1.0, min(1.0, float(turn) + float(ds_joystick_turn)))
        return mixed_fwd, mixed_turn

    def refresh_ds_joystick_state():
        nonlocal ds_joystick_fwd, ds_joystick_turn, driver_priority_active
        if args.ds_joystick and sd is not None:
            _ds_scale = max(0.0, min(1.0, float(args.ds_joystick_scale)))
            ds_joystick_fwd = float(sd.getNumber(args.ds_joystick_fwd_key, 0.0)) * _ds_scale
            ds_joystick_turn = float(sd.getNumber(args.ds_joystick_turn_key, 0.0)) * _ds_scale
        else:
            ds_joystick_fwd = 0.0
            ds_joystick_turn = 0.0
        threshold = max(0.0, float(args.driver_priority_threshold))
        driver_priority_active = bool(
            args.driver_priority_mode
            and (abs(ds_joystick_fwd) >= threshold or abs(ds_joystick_turn) >= threshold)
        )
        if time.time() < float(driver_priority_suppressed_until):
            driver_priority_active = False
        return ds_joystick_fwd, ds_joystick_turn

    def nt_connections_summary():
        if not HAS_NT:
            return "nt-disabled"
        try:
            conns = NetworkTables.getConnections()
        except Exception:
            return "unavailable"
        if not conns:
            return "none"
        parts = []
        for conn in conns[:3]:
            remote_id = getattr(conn, "remote_id", "?")
            remote_ip = getattr(conn, "remote_ip", "?")
            parts.append(f"{remote_id}@{remote_ip}")
        if len(conns) > 3:
            parts.append(f"+{len(conns) - 3} more")
        return ", ".join(parts)

    def render_status_panel(cam_cell):
        nonlocal status_scroll_y, status_scroll_max
        panel_h = STATUS_PANEL_H
        panel_w = STATUS_PANEL_W
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (24, 24, 24)
        status_button_rects.clear()
        status_section_jump_targets.clear()

        def put_line(text, y, color=(235, 235, 235), scale=0.55):
            cv2.putText(
                panel,
                text,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                1,
                cv2.LINE_AA,
            )

        def draw_axis(label, value, y):
            value = max(-1.0, min(1.0, float(value)))
            put_line(f"{label}: {value:+.2f}", y - 10)
            x0, x1 = 170, panel_w - 20
            cx = (x0 + x1) // 2
            cv2.line(panel, (x0, y), (x1, y), (90, 90, 90), 1)
            cv2.line(panel, (cx, y - 9), (cx, y + 9), (140, 140, 140), 1)
            half = (x1 - x0) // 2
            vx = int(cx + value * half)
            color = (0, 220, 0) if abs(value) <= 0.05 else ((0, 220, 255) if value > 0 else (255, 180, 0))
            cv2.rectangle(panel, (min(cx, vx), y - 7), (max(cx, vx), y + 7), color, -1)

        def draw_button(rect, label, enabled):
            x0, y0, x1, y1 = rect
            fill = (70, 130, 220) if enabled else (50, 50, 50)
            border = (200, 200, 200) if enabled else (120, 120, 120)
            cv2.rectangle(panel, (x0, y0), (x1, y1), fill, -1)
            cv2.rectangle(panel, (x0, y0), (x1, y1), border, 1)
            text_color = (255, 255, 255) if enabled else (180, 180, 180)
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            text_x = x0 + (x1 - x0 - text_size[0]) // 2
            text_y = y0 + (y1 - y0 + text_size[1]) // 2
            cv2.putText(panel, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)

        if not args.drive:
            mode_label = "DRIVE OFF"
            mode_color = (180, 180, 180)
        elif emergency_stop:
            mode_label = "STOPPED"
            mode_color = (0, 80, 255)
        elif tracking_enabled and not tracking_pose_ok:
            mode_label = "TRACK LOST"
            mode_color = (0, 140, 255)
        elif manual_mode:
            mode_label = "MANUAL"
            mode_color = (0, 220, 255)
        elif goal_cell is not None:
            mode_label = "AUTO"
            mode_color = (0, 220, 0)
        else:
            mode_label = "IDLE"
            mode_color = (180, 180, 180)

        put_line("ZED DRIVE STATUS", 30, (255, 255, 255), 0.72)
        put_line(
            f"Mode: {mode_label} | Direct Nav: {'ON' if direct_nav_enabled else 'OFF'}",
            62,
            mode_color,
            0.56,
        )
        put_line(f"E-stop: {'ON' if emergency_stop else 'OFF'}", 88, (0, 80, 255) if emergency_stop else (170, 255, 170))
        _nt_status_color = (170, 255, 170) if nt_connected_cached else (0, 60, 255)
        _nt_watchdog_txt = " [WATCHDOG TRIPPED]" if nt_watchdog_tripped else ""
        put_line(f"NT: {'OK' if nt_connected_cached else 'LOST'}{_nt_watchdog_txt}", 114, _nt_status_color)
        put_line(
            f"Main rover mode: {'ON' if args.main_rover_mode else 'OFF'} (u)",
            132,
            (255, 220, 170) if args.main_rover_mode else (190, 190, 190),
            0.45,
        )
        if tracking_enabled:
            track_txt = "OK" if tracking_pose_ok else "LOST"
            if args.area_memory:
                area_txt = "LOCKED" if tracking_pose_ok else "SEARCH"
            else:
                area_txt = "OFF"
        else:
            track_txt = "OFF"
            area_txt = "OFF"
        track_color = (170, 255, 170) if tracking_pose_ok else (0, 140, 255)
        navx_status_txt = "OFF"
        navx_status_color = (190, 190, 190)
        if args.navx_heading_aid:
            if navx_yaw_deg is None or (not np.isfinite(navx_yaw_deg)):
                navx_status_txt = "NO DATA"
                navx_status_color = (0, 140, 255)
            else:
                navx_fallback_active = bool(
                    (not tracking_pose_ok)
                    and heading_fallback_forward_world is not None
                    and navx_estimated_rover_forward_world is not None
                )
                if navx_fallback_active:
                    navx_state = "FALLBACK"
                    navx_status_color = (255, 210, 120)
                elif navx_sign_locked:
                    navx_state = "LOCKED"
                    navx_status_color = (170, 255, 170)
                else:
                    navx_state = "CAL"
                    navx_status_color = (0, 220, 255)
                navx_status_txt = f"{navx_state} {float(navx_yaw_deg):+.1f}deg"
        put_line(
            f"Tracking: {track_txt} | AreaMem: {area_txt} | NavX: {navx_status_txt}",
            150,
            navx_status_color if args.navx_heading_aid else track_color,
            0.50,
        )
        map_state = "ACTIVE" if map_integration_ok else "PAUSED"
        map_view_mode = "RED-ONLY" if map_red_only_view else "NORMAL"
        map_mode = "COMPLEX" if args.complex else "SIMPLE"
        map_state_color = (170, 255, 170) if map_integration_ok else (0, 180, 255)
        put_line(
            f"Map: {map_state} {map_mode} | Follow: {'ON' if follow_rover_map else 'OFF'} | View: {map_view_mode} | Pts: {last_map_point_count}",
            168,
            map_state_color,
            0.46,
        )
        put_line(
            f"Depth: {last_depth_status} | Raw {last_raw_point_count} | In-range {last_in_range_point_count}"
            + (" | RANGE BYPASS" if range_filter_bypassed else ""),
            186,
            (180, 240, 255) if last_raw_point_count > 0 else (0, 180, 255),
            0.42,
        )
        if not map_integration_ok:
            pause_detail = camera_map_pause_reason if camera_map_pause_reason else "TRACKING LOST OR PAUSED"
            put_line(f"Map pause reason: {pause_detail}", 204, (0, 200, 255), 0.44)
            servo_info_y = 222
        else:
            put_line(
                f"Ground {last_ground_pct:4.1f}% | Obst {last_obstacle_pct:4.1f}% | Holes {last_hole_pct:4.1f}%",
                204,
                (180, 240, 255),
                0.44,
            )
            servo_info_y = 222
        landmark_status = (
            f"AI landmarks: {len(landmark_memory.get('landmarks', []))} saved"
            f" | Human: {'ON' if human_detect_enabled else 'OFF'}"
            f" | Rock YOLO: {'ON' if rock_detect_enabled else 'OFF'}"
        )
        if tracking_enabled and (not tracking_pose_ok) and landmark_pose_override_t_map is not None:
            landmark_status += " | pose hold: landmark"
        put_line(
            landmark_status,
            servo_info_y,
            (190, 190, 190),
            0.45,
        )
        servo_state_txt = "OFF"
        servo_state_color = (190, 190, 190)
        if args.camera_servo_track:
            if servo_turning:
                servo_state_txt = "TURNING"
                servo_state_color = (0, 200, 255)
            elif servo_deposit_view:
                servo_state_txt = "DEPOSIT"
                servo_state_color = (255, 200, 120)
            else:
                servo_state_txt = "MAP"
                servo_state_color = (170, 255, 170)
        put_line(
            f"Camera servo: {servo_state_txt} | angle {servo_angle_deg:.0f} | target {servo_command_angle_deg:.0f}",
            servo_info_y + 18,
            servo_state_color,
            0.46,
        )

        excav_set = bool(mining.excav_corners_rc)
        deposit_set = bool(mining.deposit_corners_rc)
        starting_set = bool(mining.starting_corners_rc)
        put_line(f"Excavation zone: {'SET' if excav_set else 'unset'}", servo_info_y + 42, (170, 255, 170) if excav_set else (190, 190, 190))
        put_line(f"Deposit zone: {'SET' if deposit_set else 'unset'}", servo_info_y + 66, (170, 255, 170) if deposit_set else (190, 190, 190))
        put_line(f"Starting zone: {'SET' if starting_set else 'unset'}", servo_info_y + 90, (170, 255, 170) if starting_set else (190, 190, 190))
        put_line("Click a button below, then define 4 corners on the map.", servo_info_y + 114, (210, 210, 210), 0.48)

        if goal_cell is None:
            put_line("Goal cell: none", servo_info_y + 136, (190, 190, 190))
            put_line("Goal world: none", servo_info_y + 160, (190, 190, 190))
        else:
            goal_world = occ_map.grid_to_world(goal_cell[0], goal_cell[1])
            put_line(f"Goal cell: r={goal_cell[0]} c={goal_cell[1]}", servo_info_y + 136, (220, 240, 255))
            if goal_world is None:
                put_line("Goal world: unavailable", servo_info_y + 160, (190, 190, 190))
            else:
                put_line(f"Goal world: x={goal_world[0]:+.2f} z={goal_world[1]:+.2f}", servo_info_y + 160, (220, 240, 255))

        if status_target_world is None:
            put_line("Active target: none", servo_info_y + 150, (190, 190, 190))
        else:
            tc = status_target_cell
            if tc is None:
                put_line(
                    f"Active target: x={status_target_world[0]:+.2f} z={status_target_world[1]:+.2f}",
                    servo_info_y + 150,
                    (255, 235, 170),
                )
            else:
                put_line(
                    f"Active target: r={tc[0]} c={tc[1]} x={status_target_world[0]:+.2f} z={status_target_world[1]:+.2f}",
                    servo_info_y + 150,
                    (255, 235, 170),
                )

        if cam_cell is None:
            put_line("Robot cell: unavailable", servo_info_y + 174, (190, 190, 190))
        else:
            put_line(f"Robot cell: r={cam_cell[0]} c={cam_cell[1]}", servo_info_y + 174, (180, 255, 220))

        put_line(f"Map zoom: x{map_scale_live}", servo_info_y + 198, (220, 240, 255))
        put_line(
            f"Last command: {'ENABLED' if status_cmd_enabled else 'DISABLED'} dur={status_cmd_duration:.2f}s",
            servo_info_y + 222,
            (190, 255, 190) if status_cmd_enabled else (190, 190, 190),
        )
        draw_axis("Forward", status_cmd_fwd, servo_info_y + 246)
        draw_axis("Turn", status_cmd_turn, servo_info_y + 268)

        # --- Rover size input field (placed between axis bars and the zone buttons) ---
        cur_rover_ft = args.rover_size_m / 0.3048
        put_line(
            "Rover size (ft, square) — click field, type e.g. 2.5, press Enter",
            servo_info_y + 272,
            (170, 200, 230),
            0.44,
        )
        input_rect = (16, servo_info_y + 282, panel_w - 16, servo_info_y + 322)
        status_button_rects["map_size_input"] = input_rect
        border_color = (100, 220, 255) if map_size_input_focused else (120, 120, 120)
        cv2.rectangle(panel, (input_rect[0], input_rect[1]), (input_rect[2], input_rect[3]), (40, 40, 40), -1)
        cv2.rectangle(panel, (input_rect[0], input_rect[1]), (input_rect[2], input_rect[3]), border_color, 1)
        display_text = map_size_input_text if map_size_input_text else f"{cur_rover_ft:.2f}"
        cursor = "|" if map_size_input_focused else ""
        cv2.putText(
            panel,
            display_text + cursor,
            (input_rect[0] + 8, input_rect[1] + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        put_line(
            f"Current rover size: {cur_rover_ft:.2f} ft  ({args.rover_size_m:.3f} m)",
            servo_info_y + 336,
            (200, 240, 255),
            0.48,
        )
        refresh_actuator_feedback()
        left_pct_text = "n/a" if actuator_left_extension_pct is None else f"{actuator_left_extension_pct:.0f}%"
        right_pct_text = "n/a" if actuator_right_extension_pct is None else f"{actuator_right_extension_pct:.0f}%"
        left_in_text = "n/a" if actuator_left_extension_inches is None else f"{actuator_left_extension_inches:.2f}in"
        right_in_text = "n/a" if actuator_right_extension_inches is None else f"{actuator_right_extension_inches:.2f}in"
        tailgate_pct_text = "n/a" if actuator_tailgate_extension_pct is None else f"{actuator_tailgate_extension_pct:.0f}%"
        tailgate_in_text = "n/a" if actuator_tailgate_inches is None else f"{actuator_tailgate_inches:.2f}in"
        cal_text = "CAL" if actuator_bottom_position_calibrated else "UNCAL"
        if actuator_bottom_position_calibrated is None:
            cal_text = "CAL?"
        sync_text = "SYNC FAULT" if actuator_sync_fault else "SYNC OK"
        if actuator_sync_fault is None:
            sync_text = "SYNC?"
        left_counts_text = "n/a" if actuator_left_counts is None else f"{int(round(actuator_left_counts))}"
        right_counts_text = "n/a" if actuator_right_counts is None else f"{int(round(actuator_right_counts))}"
        hall_state_text = (
            "Hall sensors: OK"
            if (actuator_left_counts is not None and actuator_right_counts is not None)
            else "Hall sensors: no data"
        )
        put_line(
            f"Actuator extension: L {left_in_text} ({left_pct_text}) | R {right_in_text} ({right_pct_text}) | {cal_text} | {sync_text}",
            servo_info_y + 360,
            (0, 120, 255) if actuator_sync_fault else (200, 240, 255),
            0.40,
        )
        put_line(
            f"{hall_state_text} | counts L {left_counts_text} | R {right_counts_text} | TG {tailgate_in_text} {tailgate_pct_text}",
            servo_info_y + 378,
            (170, 255, 170) if (actuator_left_counts is not None and actuator_right_counts is not None) else (190, 190, 190),
            0.38,
        )

        def draw_pct_bar(x0, y0, width, height, label, value, active):
            cv2.putText(panel, label, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 240), 1, cv2.LINE_AA)
            cv2.rectangle(panel, (x0, y0), (x0 + width, y0 + height), (55, 55, 55), -1)
            cv2.rectangle(panel, (x0, y0), (x0 + width, y0 + height), (120, 120, 120), 1)
            if value is not None:
                fill_w = int(round((max(0.0, min(100.0, float(value))) / 100.0) * max(1, width - 2)))
                fill_color = (0, 180, 110) if active else (0, 140, 220)
                cv2.rectangle(panel, (x0 + 1, y0 + 1), (x0 + 1 + fill_w, y0 + height - 1), fill_color, -1)
                val_text = f"{value:.0f}%"
            else:
                val_text = "n/a"
            cv2.putText(
                panel,
                val_text,
                (x0 + width + 8, y0 + height - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        pct_bar_w = max(180, min(260, panel_w - 220))
        draw_pct_bar(16, servo_info_y + 398, pct_bar_w, 18, "Left actuator", actuator_left_extension_pct, test_excavation_left_extend_active)
        draw_pct_bar(16, servo_info_y + 426, pct_bar_w, 18, "Right actuator", actuator_right_extension_pct, test_excavation_right_extend_active)
        put_line(
            "Dig profile name — click field, type a name, recording uses style+phase automatically",
            servo_info_y + 466,
            (170, 200, 230),
            0.44,
        )
        dig_name_rect = (16, servo_info_y + 476, panel_w - 16, servo_info_y + 516)
        status_button_rects["dig_name_input"] = dig_name_rect
        dig_name_border = (100, 220, 255) if dig_name_input_focused else (120, 120, 120)
        cv2.rectangle(panel, (dig_name_rect[0], dig_name_rect[1]), (dig_name_rect[2], dig_name_rect[3]), (40, 40, 40), -1)
        cv2.rectangle(panel, (dig_name_rect[0], dig_name_rect[1]), (dig_name_rect[2], dig_name_rect[3]), dig_name_border, 1)
        dig_name_display = dig_name_input_text if dig_name_input_text else "example: trench_v2"
        dig_name_cursor = "|" if dig_name_input_focused else ""
        dig_name_color = (255, 255, 255) if dig_name_input_text else (180, 180, 180)
        cv2.putText(
            panel,
            dig_name_display + dig_name_cursor,
            (dig_name_rect[0] + 8, dig_name_rect[1] + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            dig_name_color,
            1,
            cv2.LINE_AA,
        )

        jump_bar_top = dig_name_rect[3] + 16
        jump_bar_h = 40
        jump_gap = 10
        jump_buttons = [
            ("jump_setup", "Setup"),
            ("jump_map_tools", "Map"),
            ("jump_zones_camera", "Zones"),
            ("jump_calibration", "Calibration"),
            ("jump_actuators", "Actuators"),
            ("jump_dig_profiles", "Dig"),
        ]
        jump_btn_w = int((panel_w - 32 - (len(jump_buttons) - 1) * jump_gap) / len(jump_buttons))

        def draw_nav_button(rect, label, fill, border):
            x0, y0, x1b, y1b = rect
            cv2.rectangle(panel, (x0, y0), (x1b, y1b), fill, -1)
            cv2.rectangle(panel, (x0, y0), (x1b, y1b), border, 1)
            tsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.putText(
                panel,
                label,
                (x0 + (x1b - x0 - tsz[0]) // 2, y0 + (y1b - y0 + tsz[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        put_line("Quick Access", jump_bar_top - 6, (170, 200, 230), 0.44)
        for idx, (jump_name, jump_label) in enumerate(jump_buttons):
            x0 = 16 + idx * (jump_btn_w + jump_gap)
            rect = (x0, jump_bar_top, x0 + jump_btn_w, jump_bar_top + jump_bar_h)
            status_button_rects[jump_name] = rect
            draw_nav_button(rect, jump_label, (46, 72, 112), (145, 195, 255))

        controls_top = jump_bar_top + jump_bar_h + 18
        controls_bottom = panel_h - 20
        controls_h = max(1, controls_bottom - controls_top)
        button_h = 46
        card_gap = 14
        card_x0 = 10
        scrollbar_margin = 34
        scrollbar_w = 24
        card_x1 = panel_w - scrollbar_margin - scrollbar_w - 8
        card_inner = 14
        grid_gap = 12
        button_w = max(160, int((card_x1 - card_x0 - 2 * card_inner - 2 * grid_gap) / 3))
        controls = np.zeros((2600, panel_w, 3), dtype=np.uint8)
        controls[:] = (24, 24, 28)

        def tint(color, mix=0.35, base=(36, 36, 42)):
            return tuple(int(base[i] * (1.0 - mix) + int(color[i]) * mix) for i in range(3))

        def put_control_line(text, y, color=(235, 235, 235), scale=0.48, x=22):
            cv2.putText(controls, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

        def draw_control_button(rect, label, enabled, active=False, active_color=(70, 130, 220), active_border=(200, 200, 200)):
            x0, y0, x1b, y1b = rect
            fill = active_color if active else ((66, 104, 164) if enabled else (48, 48, 54))
            border = active_border if active else ((220, 225, 230) if enabled else (120, 120, 126))
            text_color = (255, 255, 255) if enabled or active else (180, 180, 180)
            cv2.rectangle(controls, (x0, y0), (x1b, y1b), fill, -1)
            cv2.rectangle(controls, (x0, y0), (x1b, y1b), border, 2 if active else 1)
            tsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.putText(
                controls,
                label,
                (x0 + (x1b - x0 - tsz[0]) // 2, y0 + (y1b - y0 + tsz[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                text_color,
                1,
                cv2.LINE_AA,
            )

        def _screen_rect(rect):
            x0, y0, x1b, y1b = rect
            sy0 = y0 - status_scroll_y + controls_top
            sy1 = y1b - status_scroll_y + controls_top
            if sy1 < controls_top or sy0 > controls_bottom:
                return None
            return (x0, max(controls_top, sy0), x1b, min(controls_bottom, sy1))

        def _register_button(name, rect):
            srect = _screen_rect(rect)
            if srect is not None:
                status_button_rects[name] = srect

        def section_frame(y0, height, title, subtitle, accent, key):
            x0 = card_x0
            x1b = card_x1
            cv2.rectangle(controls, (x0, y0), (x1b, y0 + height), (30, 30, 36), -1)
            cv2.rectangle(controls, (x0, y0), (x1b, y0 + height), tint(accent, 0.90), 1)
            cv2.rectangle(controls, (x0, y0), (x1b, y0 + 38), tint(accent, 0.45), -1)
            cv2.putText(controls, title, (x0 + 14, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
            if subtitle:
                cv2.putText(controls, subtitle, (x0 + 14, y0 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 210, 230), 1, cv2.LINE_AA)
            section_offsets[key] = y0
            return y0 + 72

        def grid_rect(body_y, row_idx, col_idx, span=1):
            x0 = card_x0 + card_inner + col_idx * (button_w + grid_gap)
            width = span * button_w + (span - 1) * grid_gap
            y0 = body_y + row_idx * (button_h + 10)
            return (x0, y0, x0 + width, y0 + button_h)

        button_enabled = mining_buttons_enabled()
        mining_running = mining.state in (
            auto_mining.MiningState.PLAN_SWEEP,
            auto_mining.MiningState.NAVIGATE_DIG,
            auto_mining.MiningState.DIGGING,
            auto_mining.MiningState.BACKUP,
            auto_mining.MiningState.NAVIGATE_DEPOSIT,
            auto_mining.MiningState.DEPOSITING,
        )
        excav_drawing = mining.state == auto_mining.MiningState.DRAW_EXCAV
        deposit_drawing = mining.state == auto_mining.MiningState.DRAW_DEPOSIT
        picking_dig_start = mining.state == auto_mining.MiningState.PICK_DIG_START
        zone_buttons_enabled = not mining_running
        _mining_active = mining.state not in (
            auto_mining.MiningState.IDLE,
            auto_mining.MiningState.DRAW_EXCAV,
            auto_mining.MiningState.DRAW_DEPOSIT,
            auto_mining.MiningState.PICK_DIG_START,
            auto_mining.MiningState.DONE,
            auto_mining.MiningState.ABORTED,
        )

        section_offsets = {}
        cursor_y = 12

        setup_section_h = 72 + (button_h + 10) + 18
        setup_body_y = section_frame(
            cursor_y,
            setup_section_h,
            "Setup",
            "Startup and low-latency camera-only controls for manual operation.",
            (170, 210, 255),
            "setup",
        )
        manual_mode_rect = grid_rect(setup_body_y, 0, 0)
        no_mapping_rect = grid_rect(setup_body_y, 0, 1)
        setup_low_latency_rect = grid_rect(setup_body_y, 0, 2)
        draw_control_button(
            manual_mode_rect,
            "Manual: ON" if manual_mode else "Manual",
            True,
            manual_mode,
            (0, 130, 210),
            (120, 220, 255),
        )
        draw_control_button(
            no_mapping_rect,
            "No Mapping: ON" if no_mapping_mode else "No Mapping",
            True,
            no_mapping_mode,
            (0, 150, 90),
            (110, 255, 180),
        )
        draw_control_button(
            setup_low_latency_rect,
            "Low Latency: ON" if low_latency_mode else "Low Latency",
            True,
            low_latency_mode,
            (80, 60, 180),
            (180, 160, 255),
        )
        cursor_y += setup_section_h + card_gap

        map_section_h = 72 + 5 * (button_h + 10) + 84
        map_body_y = section_frame(
            cursor_y,
            map_section_h,
            "Map Tools",
            "Run, navigation, paint tools, and persistent map controls.",
            (88, 170, 255),
            "map_tools",
        )
        auto_run_rect = grid_rect(map_body_y, 0, 0, span=2)
        direct_nav_rect = grid_rect(map_body_y, 0, 2)
        whole_rect = grid_rect(map_body_y, 1, 0)
        smooth_rect = grid_rect(map_body_y, 1, 1)
        holes_rect = grid_rect(map_body_y, 1, 2)
        reset_map_rect = grid_rect(map_body_y, 2, 0)
        obstacle_rect = grid_rect(map_body_y, 2, 1)
        paint_rect = grid_rect(map_body_y, 2, 2)
        erase_rect = grid_rect(map_body_y, 3, 0)
        clear_paint_rect = grid_rect(map_body_y, 3, 1)
        lock_green_rect = grid_rect(map_body_y, 3, 2)
        main_rover_rect = grid_rect(map_body_y, 4, 0, span=3)
        slider_y = map_body_y + 5 * (button_h + 10) + 16
        btn_sm = 36
        brush_minus_rect = (card_x0 + card_inner, slider_y + 6, card_x0 + card_inner + btn_sm, slider_y + 6 + btn_sm)
        brush_plus_rect = (card_x1 - card_inner - btn_sm, slider_y + 6, card_x1 - card_inner, slider_y + 6 + btn_sm)
        slider_x0 = brush_minus_rect[2] + 10
        slider_x1 = brush_plus_rect[0] - 10
        brush_slider_rect = (slider_x0, slider_y, slider_x1, slider_y + 48)
        auto_run_label = "Stop Auto Run" if _mining_active else "Start Auto Run"
        draw_control_button(auto_run_rect, auto_run_label, True, _mining_active, (0, 140, 40), (60, 240, 100))
        draw_control_button(
            direct_nav_rect,
            "Direct Nav: ON" if direct_nav_enabled else "Direct Nav",
            True,
            direct_nav_enabled,
            (0, 150, 90),
            (100, 255, 180),
        )
        draw_control_button(whole_rect, "Whole Map", button_enabled)
        draw_control_button(
            smooth_rect,
            "Smooth Map: ON" if smooth_map_enabled else "Smooth Map",
            button_enabled,
            smooth_map_enabled,
            (0, 160, 160),
            (80, 220, 220),
        )
        draw_control_button(holes_rect, "Disable Holes", button_enabled)
        draw_control_button(reset_map_rect, "Reset Map", True, reset_map_confirm, (0, 70, 200), (80, 160, 255))
        draw_control_button(
            obstacle_rect,
            "Paint Obstacle: ON" if paint_obstacle_mode else "Paint Obstacle",
            True,
            paint_obstacle_mode,
            (0, 0, 200),
            (80, 80, 255),
        )
        draw_control_button(
            paint_rect,
            "Paint Safe: ON" if paint_safe_mode else "Paint Safe",
            True,
            paint_safe_mode,
            (0, 180, 80),
            (80, 255, 140),
        )
        draw_control_button(
            erase_rect,
            "Erase: ON" if erase_safe_mode else "Erase Safe",
            True,
            erase_safe_mode,
            (0, 80, 200),
            (80, 140, 255),
        )
        draw_control_button(clear_paint_rect, "Clear Paint", True)
        lock_label = "Green Locked" if lock_green_applied else "Lock Green"
        draw_control_button(lock_green_rect, lock_label, True, lock_green_applied, (0, 160, 80), (80, 255, 140))
        draw_control_button(
            main_rover_rect,
            "Main Rover: ON" if args.main_rover_mode else "Main Rover",
            True,
            args.main_rover_mode,
            (0, 120, 200),
            (80, 220, 255),
        )
        put_control_line("Brush size", slider_y - 2, (170, 200, 230), 0.44, x=card_x0 + card_inner)
        cv2.rectangle(controls, (slider_x0, slider_y + 18), (slider_x1, slider_y + 34), (60, 60, 60), -1)
        cv2.rectangle(controls, (slider_x0, slider_y + 18), (slider_x1, slider_y + 34), (120, 120, 120), 1)
        frac = (paint_brush_radius - 1) / 14.0
        knob_x = int(slider_x0 + frac * (slider_x1 - slider_x0))
        cv2.circle(controls, (knob_x, slider_y + 26), 11, (100, 200, 255), -1)
        cv2.circle(controls, (knob_x, slider_y + 26), 11, (200, 240, 255), 1)
        put_control_line(f"Brush: {paint_brush_radius} cells", slider_y + 52, (220, 240, 255), 0.44, x=card_x0 + card_inner)
        for rect, lbl in ((brush_minus_rect, "-"), (brush_plus_rect, "+")):
            x0b, y0b, x1b, y1b = rect
            cv2.rectangle(controls, (x0b, y0b), (x1b, y1b), (70, 130, 220), -1)
            cv2.rectangle(controls, (x0b, y0b), (x1b, y1b), (200, 200, 200), 1)
            tsz, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(
                controls,
                lbl,
                (x0b + (x1b - x0b - tsz[0]) // 2, y0b + (y1b - y0b + tsz[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cursor_y += map_section_h + card_gap

        zones_section_h = 72 + 4 * (button_h + 10) + 20
        zones_body_y = section_frame(
            cursor_y,
            zones_section_h,
            "Zones & Camera",
            "Excavation/deposit selection and camera access.",
            (255, 188, 92),
            "zones_camera",
        )
        excav_rect = grid_rect(zones_body_y, 0, 0)
        deposit_rect = grid_rect(zones_body_y, 0, 1)
        starting_zone_rect = grid_rect(zones_body_y, 0, 2)
        pick_dig_start_rect = grid_rect(zones_body_y, 1, 0)
        berm_left_rect = grid_rect(zones_body_y, 1, 1)
        berm_right_rect = grid_rect(zones_body_y, 1, 2)
        camera_view_rect = grid_rect(zones_body_y, 2, 0)
        camera_overlay_rect = grid_rect(zones_body_y, 2, 1)
        auto_digger_rect = grid_rect(zones_body_y, 2, 2)
        human_detect_rect = grid_rect(zones_body_y, 3, 0)
        rock_detect_rect = grid_rect(zones_body_y, 3, 1)
        excav_label = "Drawing Excav..." if excav_drawing else ("Excav Zone Set" if excav_set else "Draw Excav Zone")
        deposit_label = "Drawing Deposit..." if deposit_drawing else ("Deposit Zone Set" if deposit_set else "Draw Deposit Zone")
        starting_label = "Starting Zone Set" if starting_set else "Set Starting Zone"
        draw_control_button(excav_rect, excav_label, zone_buttons_enabled, excav_drawing or excav_set, (0, 120, 220), (80, 200, 255))
        draw_control_button(deposit_rect, deposit_label, zone_buttons_enabled, deposit_drawing or deposit_set, (180, 150, 0), (255, 230, 80))
        draw_control_button(starting_zone_rect, starting_label, zone_buttons_enabled, starting_set, (0, 150, 70), (100, 255, 160))
        pick_label = "Picking Start..." if picking_dig_start else ("Dig Start Set" if mining.preferred_start_rc is not None else "Pick Dig Start")
        draw_control_button(
            pick_dig_start_rect,
            pick_label,
            zone_buttons_enabled and excav_set,
            picking_dig_start or mining.preferred_start_rc is not None,
            (0, 170, 70),
            (100, 255, 160),
        )
        draw_control_button(
            berm_left_rect,
            "Berm: Left",
            zone_buttons_enabled,
            mining.deposit_zone_preset_side == "left",
            (170, 120, 0),
            (255, 220, 120),
        )
        draw_control_button(
            berm_right_rect,
            "Berm: Right",
            zone_buttons_enabled,
            mining.deposit_zone_preset_side == "right",
            (170, 120, 0),
            (255, 220, 120),
        )
        camera_label = f"Camera: Deposit {args.camera_deposit_angle_deg:.0f}" if (
            servo_deposit_view
            or abs(angle_error_deg(servo_command_angle_deg, args.camera_deposit_angle_deg)) <= 2.0
        ) else f"Camera: Map {args.camera_map_angle_deg:.0f}"
        camera_active = servo_deposit_view or (servo_turning and abs(angle_error_deg(servo_command_angle_deg, args.camera_deposit_angle_deg)) <= 2.0)
        draw_control_button(
            camera_view_rect,
            camera_label,
            bool(args.camera_servo_track and sd is not None),
            camera_active,
            (150, 90, 0),
            (255, 220, 120),
        )
        draw_control_button(
            camera_overlay_rect,
            "Cam Overlay: ON" if camera_overlay_enabled else "Cam Overlay",
            True,
            camera_overlay_enabled,
            (0, 120, 180),
            (120, 220, 255),
        )
        draw_control_button(
            auto_digger_rect,
            "Auto Dig: ON" if auto_digger_enabled else "Auto Dig",
            True,
            auto_digger_enabled,
            (0, 140, 60),
            (110, 255, 150),
        )
        draw_control_button(
            human_detect_rect,
            "Human Detect: ON" if human_detect_enabled else "Human Detect",
            bool(human_detect_available),
            human_detect_enabled,
            (130, 50, 180),
            (230, 170, 255),
        )
        draw_control_button(
            rock_detect_rect,
            "Rock YOLO: ON" if rock_detect_enabled else "Rock YOLO",
            bool(rock_model is not None),
            rock_detect_enabled,
            (180, 60, 20),
            (255, 180, 120),
        )
        low_latency_rect = grid_rect(zones_body_y, 3, 2)
        draw_control_button(
            low_latency_rect,
            "Low Latency: ON" if low_latency_mode else "Low Latency",
            True,
            low_latency_mode,
            (80, 60, 180),
            (180, 160, 255),
        )
        cursor_y += zones_section_h + card_gap

        cal_section_h = 72 + 5 * (button_h + 10) + 272
        cal_body_y = section_frame(
            cursor_y,
            cal_section_h,
            "Calibration & Drive",
            "Drive flip, arrow flip, and heading calibration tools.",
            (118, 182, 255),
            "calibration",
        )
        drive_calibration_mode_rect = grid_rect(cal_body_y, 0, 0)
        drive_calibration_cancel_rect = grid_rect(cal_body_y, 0, 1)
        drive_heading_flip_rect = grid_rect(cal_body_y, 0, 2)
        display_heading_flip_rect = grid_rect(cal_body_y, 1, 0)
        hard_drive_flip_rect = grid_rect(cal_body_y, 1, 1)
        camera_view_flip_rect = grid_rect(cal_body_y, 1, 2)
        steering_flip_rect = grid_rect(cal_body_y, 2, 0)
        test_drive_forward_rect = grid_rect(cal_body_y, 2, 1)
        bidirectional_auto_rect = grid_rect(cal_body_y, 2, 2)
        demo_auto_rect = grid_rect(cal_body_y, 3, 0)
        lock_start_frame_rect = grid_rect(cal_body_y, 3, 1)
        scan_start_frame_rect = grid_rect(cal_body_y, 3, 2)
        cal_slider_y = cal_body_y + 5 * (button_h + 10) + 90
        cal_slider_x0 = card_x0 + card_inner + 8
        cal_slider_x1 = card_x1 - card_inner - 8
        drive_speed_slider_rect = (cal_slider_x0, cal_slider_y, cal_slider_x1, cal_slider_y + 44)
        turn_slider_y = cal_slider_y + 80
        turn_speed_slider_rect = (cal_slider_x0, turn_slider_y, cal_slider_x1, turn_slider_y + 44)
        draw_control_button(
            drive_calibration_mode_rect,
            "Drive Cal: ON" if drive_calibration.active else "Drive Cal",
            True,
            drive_calibration.active,
            (0, 130, 200),
            (90, 220, 255),
        )
        draw_control_button(
            drive_calibration_cancel_rect,
            "Cancel Cal",
            bool(drive_calibration.active or drive_calibration.target_cell is not None),
            False,
        )
        draw_control_button(
            drive_heading_flip_rect,
            "Flip Drive: ON" if args.drive_heading_flip else "Flip Drive",
            True,
            bool(args.drive_heading_flip),
            (180, 80, 0),
            (255, 190, 110),
        )
        draw_control_button(
            display_heading_flip_rect,
            "Flip Arrow: ON" if args.display_heading_flip else "Flip Arrow",
            True,
            bool(args.display_heading_flip),
            (140, 80, 180),
            (220, 150, 255),
        )
        draw_control_button(
            hard_drive_flip_rect,
            "Hard Flip: ON" if args.hard_drive_flip else "Hard Flip",
            True,
            bool(args.hard_drive_flip),
            (160, 40, 40),
            (255, 120, 120),
        )
        draw_control_button(
            camera_view_flip_rect,
            "Flip Map/Depo: ON" if camera_view_flip_active() else "Flip Map/Depo",
            True,
            bool(camera_view_flip_active()),
            (100, 90, 20),
            (220, 210, 120),
        )
        draw_control_button(
            steering_flip_rect,
            "Flip Steering: ON" if args.steering_flip else "Flip Steering",
            True,
            bool(args.steering_flip),
            (40, 120, 160),
            (120, 220, 255),
        )
        draw_control_button(
            test_drive_forward_rect,
            "Test Forward: ON" if test_drive_forward_active else "Test Forward 5s",
            bool(args.drive and sd is not None),
            bool(test_drive_forward_active),
            (20, 130, 60),
            (120, 255, 170),
        )
        draw_control_button(
            bidirectional_auto_rect,
            "Bidirectional: ON" if bidirectional_auto_enabled else "Bidirectional Auto",
            True,
            bool(bidirectional_auto_enabled),
            (90, 90, 20),
            (255, 235, 120),
        )
        draw_control_button(
            demo_auto_rect,
            "Demo Auto: ON" if demo_auto_enabled else "Demo Auto",
            True,
            bool(demo_auto_enabled),
            (80, 50, 150),
            (185, 140, 255),
        )
        draw_control_button(
            lock_start_frame_rect,
            "Lock Start Frame",
            bool(tracking_enabled and start_frame_tag_dictionary is not None and len(start_frame_tag_layout) >= 3),
            False,
            (0, 130, 120),
            (120, 255, 235),
        )
        draw_control_button(
            scan_start_frame_rect,
            "Scan Start: ON" if start_frame_scan_active else "Scan Start Frame",
            bool(tracking_enabled and start_frame_tag_dictionary is not None and len(start_frame_tag_layout) >= 3),
            bool(start_frame_scan_active),
            (80, 90, 180),
            (160, 190, 255),
        )
        put_control_line(
            f"Calibration status: {'ACTIVE' if drive_calibration.active else 'IDLE'}",
            cal_body_y + 5 * (button_h + 10) + 14,
            (180, 220, 255),
            0.44,
            x=card_x0 + card_inner,
        )
        put_control_line(
            drive_calibration.last_result[:88],
            cal_body_y + 5 * (button_h + 10) + 38,
            (210, 230, 255),
            0.40,
            x=card_x0 + card_inner,
        )
        put_control_line(
            start_frame_last_status[:88],
            cal_body_y + 5 * (button_h + 10) + 62,
            (180, 255, 235),
            0.40,
            x=card_x0 + card_inner,
        )
        put_control_line(
            f"Auto speed: {float(args.drive_speed):.2f}",
            cal_slider_y - 2,
            (180, 220, 255),
            0.42,
            x=card_x0 + card_inner,
        )
        cv2.rectangle(controls, (cal_slider_x0, cal_slider_y + 18), (cal_slider_x1, cal_slider_y + 34), (60, 60, 60), -1)
        cv2.rectangle(controls, (cal_slider_x0, cal_slider_y + 18), (cal_slider_x1, cal_slider_y + 34), (120, 120, 120), 1)
        drive_speed_frac = (max(0.10, min(1.00, float(args.drive_speed))) - 0.10) / 0.90
        drive_speed_knob_x = int(cal_slider_x0 + drive_speed_frac * (cal_slider_x1 - cal_slider_x0))
        cv2.circle(controls, (drive_speed_knob_x, cal_slider_y + 26), 11, (80, 210, 255), -1)
        cv2.circle(controls, (drive_speed_knob_x, cal_slider_y + 26), 11, (220, 245, 255), 1)
        put_control_line(
            "Slower",
            cal_slider_y + 52,
            (180, 180, 180),
            0.36,
            x=cal_slider_x0,
        )
        put_control_line(
            "Faster",
            cal_slider_y + 52,
            (180, 180, 180),
            0.36,
            x=max(cal_slider_x0, cal_slider_x1 - 44),
        )
        put_control_line(
            f"Auto turn: {float(args.drive_max_turn_cmd):.2f}",
            turn_slider_y - 2,
            (255, 220, 150),
            0.42,
            x=card_x0 + card_inner,
        )
        cv2.rectangle(controls, (cal_slider_x0, turn_slider_y + 18), (cal_slider_x1, turn_slider_y + 34), (60, 60, 60), -1)
        cv2.rectangle(controls, (cal_slider_x0, turn_slider_y + 18), (cal_slider_x1, turn_slider_y + 34), (120, 120, 120), 1)
        turn_speed_frac = (max(0.20, min(1.00, float(args.drive_max_turn_cmd))) - 0.20) / 0.80
        turn_speed_knob_x = int(cal_slider_x0 + turn_speed_frac * (cal_slider_x1 - cal_slider_x0))
        cv2.circle(controls, (turn_speed_knob_x, turn_slider_y + 26), 11, (255, 170, 70), -1)
        cv2.circle(controls, (turn_speed_knob_x, turn_slider_y + 26), 11, (255, 235, 180), 1)
        put_control_line(
            "Gentler",
            turn_slider_y + 52,
            (180, 180, 180),
            0.36,
            x=cal_slider_x0,
        )
        put_control_line(
            "Sharper",
            turn_slider_y + 52,
            (180, 180, 180),
            0.36,
            x=max(cal_slider_x0, cal_slider_x1 - 44),
        )
        cursor_y += cal_section_h + card_gap

        actuators_section_h = 72 + 4 * (button_h + 10) + 118
        actuators_body_y = section_frame(
            cursor_y,
            actuators_section_h,
            "Actuators & Manual Test",
            "Manual excavator and door outputs for bench checks and recovery.",
            (208, 148, 255),
            "actuators",
        )
        test_excavation_lower_rect = grid_rect(actuators_body_y, 0, 0)
        stop_actuators_rect = grid_rect(actuators_body_y, 0, 1, span=2)
        test_excavation_left_extend_rect = grid_rect(actuators_body_y, 1, 0)
        test_excavation_right_extend_rect = grid_rect(actuators_body_y, 1, 1)
        test_excavation_dig_rect = grid_rect(actuators_body_y, 1, 2)
        door_open_rect = grid_rect(actuators_body_y, 2, 0)
        door_close_rect = grid_rect(actuators_body_y, 2, 1)
        test_excavation_pattern_rect = grid_rect(actuators_body_y, 2, 2)
        digger_slider_y = actuators_body_y + 3 * (button_h + 10) + 10
        digger_slider_x0 = card_x0 + card_inner
        digger_slider_x1 = card_x1 - card_inner
        digger_speed_slider_rect = (digger_slider_x0, digger_slider_y, digger_slider_x1, digger_slider_y + 44)
        draw_control_button(
            test_excavation_lower_rect,
            "Lower Cycle: ON" if test_excavation_lower_active else "Lower Cycle",
            True,
            test_excavation_lower_active,
            (140, 70, 140),
            (235, 150, 235),
        )
        draw_control_button(
            stop_actuators_rect,
            "Stop Actuators",
            True,
            False,
            (120, 60, 0),
            (255, 180, 120),
        )
        draw_control_button(
            test_excavation_left_extend_rect,
            "Left Extend: ON" if test_excavation_left_extend_active else "Left Extend",
            True,
            test_excavation_left_extend_active,
            (0, 115, 215),
            (120, 205, 255),
        )
        draw_control_button(
            test_excavation_right_extend_rect,
            "Right Extend: ON" if test_excavation_right_extend_active else "Right Extend",
            True,
            test_excavation_right_extend_active,
            (215, 120, 0),
            (255, 205, 120),
        )
        draw_control_button(
            test_excavation_dig_rect,
            "Test Digger: ON" if test_excavation_dig_active else "Test Digger",
            True,
            test_excavation_dig_active,
            (0, 90, 200),
            (80, 170, 255),
        )
        draw_control_button(
            door_open_rect,
            "Open Door: ON" if test_door_open_active else "Open Door",
            True,
            test_door_open_active,
            (0, 120, 60),
            (130, 255, 170),
        )
        draw_control_button(
            door_close_rect,
            "Close Door: ON" if test_door_close_active else "Close Door",
            True,
            test_door_close_active,
            (170, 90, 0),
            (255, 200, 120),
        )
        draw_control_button(
            test_excavation_pattern_rect,
            "Excav Test: ON" if excavation_pattern_test_active else "Excav Test x4",
            bool(args.drive and sd is not None),
            excavation_pattern_test_active,
            (90, 40, 160),
            (205, 150, 255),
        )
        put_control_line(
            f"Digger speed: {float(digger_speed_scale):.2f}",
            digger_slider_y - 2,
            (255, 220, 150),
            0.42,
            x=card_x0 + card_inner,
        )
        cv2.rectangle(controls, (digger_slider_x0, digger_slider_y + 18), (digger_slider_x1, digger_slider_y + 34), (60, 60, 60), -1)
        cv2.rectangle(controls, (digger_slider_x0, digger_slider_y + 18), (digger_slider_x1, digger_slider_y + 34), (120, 120, 120), 1)
        digger_speed_frac = (max(0.10, min(1.00, float(digger_speed_scale))) - 0.10) / 0.90
        digger_speed_knob_x = int(digger_slider_x0 + digger_speed_frac * (digger_slider_x1 - digger_slider_x0))
        cv2.circle(controls, (digger_speed_knob_x, digger_slider_y + 26), 11, (255, 210, 70), -1)
        cv2.circle(controls, (digger_speed_knob_x, digger_slider_y + 26), 11, (255, 245, 180), 1)
        put_control_line(
            "Slower",
            digger_slider_y + 52,
            (180, 180, 180),
            0.36,
            x=digger_slider_x0,
        )
        put_control_line(
            "Faster",
            digger_slider_y + 52,
            (180, 180, 180),
            0.36,
            x=max(digger_slider_x0, digger_slider_x1 - 44),
        )
        if excavation_pattern_test_active:
            _pattern_state = excavation_pattern_state(time.time())
            door_mode_text = (
                f"Excav test: {_pattern_state.get('label', 'running')}"
                if _pattern_state is not None
                else "Excav test: running"
            )
        else:
            door_mode_text = "Manual door override active" if (test_door_open_active or test_door_close_active) else "Door auto: closes for dig, opens for deposit"
        put_control_line(
            door_mode_text,
            digger_slider_y + 76,
            (220, 232, 255),
            0.41,
            x=card_x0 + card_inner,
        )
        hall_counts_color = (
            (170, 255, 170)
            if (actuator_left_counts is not None and actuator_right_counts is not None)
            else (190, 190, 190)
        )
        hall_counts_text = (
            f"Hall counts: L {int(round(actuator_left_counts)) if actuator_left_counts is not None else 'n/a'}"
            f" | R {int(round(actuator_right_counts)) if actuator_right_counts is not None else 'n/a'}"
        )
        hall_inches_text = (
            f"Travel: L {actuator_left_inches:.2f} in | R {actuator_right_inches:.2f} in"
            if (actuator_left_inches is not None and actuator_right_inches is not None)
            else "Travel: L n/a | R n/a"
        )
        hall_meta_parts = []
        if actuator_bottom_diff_counts is not None:
            hall_meta_parts.append(f"diff {int(round(actuator_bottom_diff_counts))}")
        if actuator_bottom_position_calibrated is not None:
            hall_meta_parts.append(
                "home set" if actuator_bottom_position_calibrated else "home unset"
            )
        if actuator_tailgate_counts is not None:
            hall_meta_parts.append(f"tailgate {int(round(actuator_tailgate_counts))} ct")
        elif actuator_tailgate_inches is not None:
            hall_meta_parts.append(f"tailgate {actuator_tailgate_inches:.2f} in")
        hall_meta_text = "Hall feedback: " + (" | ".join(hall_meta_parts) if hall_meta_parts else "status unavailable")
        put_control_line(
            hall_counts_text,
            actuators_body_y + 3 * (button_h + 10) + 30,
            hall_counts_color,
            0.41,
            x=card_x0 + card_inner,
        )
        put_control_line(
            hall_inches_text,
            actuators_body_y + 3 * (button_h + 10) + 50,
            (220, 232, 255),
            0.41,
            x=card_x0 + card_inner,
        )
        put_control_line(
            hall_meta_text,
            actuators_body_y + 3 * (button_h + 10) + 70,
            (220, 232, 255),
            0.41,
            x=card_x0 + card_inner,
        )
        cursor_y += actuators_section_h + card_gap

        current_dig_profiles = dig_profiles.list_profiles(
            dig_profiles.active_style, dig_profiles.active_phase
        )
        visible_dig_rows = (
            len(current_dig_profiles)
            if show_all_dig_profiles
            else min(4, len(current_dig_profiles))
        )
        visible_dig_rows = max(4, visible_dig_rows)
        dig_section_h = 72 + 8 * (button_h + 10) + 190 + visible_dig_rows * 20
        dig_body_y = section_frame(
            cursor_y,
            dig_section_h,
            "Dig Recording & Profiles",
            "Record, browse, and select dig/retract routines by style.",
            (122, 220, 160),
            "dig_profiles",
        )
        dig_style_cycle_rect = grid_rect(dig_body_y, 0, 0)
        dig_phase_cycle_rect = grid_rect(dig_body_y, 0, 1)
        dig_profile_use_rect = grid_rect(dig_body_y, 0, 2)
        dig_record_active_rect = grid_rect(dig_body_y, 1, 0)
        dig_profile_preview_rect = grid_rect(dig_body_y, 1, 1)
        dig_record_stop_rect = grid_rect(dig_body_y, 1, 2)
        dig_profile_prev_rect = grid_rect(dig_body_y, 2, 0)
        dig_profile_next_rect = grid_rect(dig_body_y, 2, 1)
        dig_profile_delete_rect = grid_rect(dig_body_y, 2, 2)
        dig_profiles_view_all_rect = grid_rect(dig_body_y, 3, 0)
        controller_record_rect = grid_rect(dig_body_y, 4, 0)
        controller_preview_rect = grid_rect(dig_body_y, 4, 1)
        controller_stop_rect = grid_rect(dig_body_y, 4, 2)
        controller_prev_rect = grid_rect(dig_body_y, 5, 0)
        controller_next_rect = grid_rect(dig_body_y, 5, 1)
        controller_use_rect = grid_rect(dig_body_y, 5, 2)
        controller_cycle_rect = grid_rect(dig_body_y, 6, 0)
        draw_control_button(dig_style_cycle_rect, f"Dig Style: {dig_profiles.active_style.title()}", True, False)
        draw_control_button(dig_phase_cycle_rect, f"Phase: {dig_profiles.active_phase.title()}", True, False)
        draw_control_button(
            dig_profile_use_rect,
            "Use Dig Record",
            bool(dig_profiles.get_cursor_profile() is not None),
            False,
        )
        draw_control_button(
            dig_record_active_rect,
            (
                f"Record {dig_profiles.active_phase.title()}: ON"
                if (
                    dig_profiles.recording
                    and dig_profiles.recording_style == dig_profiles.active_style
                    and dig_profiles.recording_phase == dig_profiles.active_phase
                )
                else f"Record {dig_profiles.active_phase.title()}"
            ),
            bool(not dig_profiles.recording),
            bool(
                dig_profiles.recording
                and dig_profiles.recording_style == dig_profiles.active_style
                and dig_profiles.recording_phase == dig_profiles.active_phase
            ),
            (0, 120, 80),
            (120, 255, 180),
        )
        draw_control_button(
            dig_profile_preview_rect,
            "Test Dig Record: ON" if dig_profile_preview_active else "Test Dig Record",
            bool((not dig_profiles.recording) and (resolve_preview_dig_profile() is not None)),
            bool(dig_profile_preview_active),
            (0, 110, 170),
            (120, 220, 255),
        )
        draw_control_button(
            dig_record_stop_rect,
            "Stop Record/Preview",
            bool(dig_profiles.recording or dig_profile_preview_active),
            bool(dig_profiles.recording or dig_profile_preview_active),
            (170, 70, 0),
            (255, 180, 120),
        )
        draw_control_button(dig_profile_prev_rect, "Dig Prev", True)
        draw_control_button(dig_profile_next_rect, "Dig Next", True)
        draw_control_button(
            dig_profile_delete_rect,
            "Delete Profile",
            bool(dig_profiles.get_cursor_profile() is not None),
            False,
        )
        draw_control_button(
            dig_profiles_view_all_rect,
            "View All Digs: ON" if show_all_dig_profiles else "View All Digs",
            True,
            bool(show_all_dig_profiles),
            (70, 100, 140),
            (150, 220, 255),
        )
        draw_control_button(
            controller_record_rect,
            "Record Controller: ON" if controller_macros.recording else "Record Controller",
            bool(not controller_macros.recording),
            bool(controller_macros.recording),
            (80, 70, 170),
            (180, 150, 255),
        )
        draw_control_button(
            controller_preview_rect,
            "Play Controller: ON" if controller_macro_preview_active else "Play Controller",
            bool((not controller_macros.recording) and (resolve_preview_controller_macro() is not None)),
            bool(controller_macro_preview_active),
            (0, 110, 170),
            (120, 220, 255),
        )
        draw_control_button(
            controller_cycle_rect,
            (
                f"Cycle Return: {controller_cycle_phase.title()}"
                if controller_cycle_preview_active else "Cycle Return"
            ),
            bool((not controller_macros.recording) and (resolve_preview_controller_macro() is not None)),
            bool(controller_cycle_preview_active),
            (90, 110, 10),
            (210, 255, 140),
        )
        draw_control_button(
            controller_stop_rect,
            "Stop Controller",
            bool(controller_macros.recording or controller_macro_preview_active or controller_cycle_preview_active),
            bool(controller_macros.recording or controller_macro_preview_active or controller_cycle_preview_active),
            (170, 70, 0),
            (255, 180, 120),
        )
        draw_control_button(controller_prev_rect, "Ctrl Prev", True)
        draw_control_button(controller_next_rect, "Ctrl Next", True)
        draw_control_button(
            controller_use_rect,
            "Use Controller",
            bool(controller_macros.get_cursor_macro() is not None),
            False,
        )

        selected_short_dig = dig_profiles.selected.get("short", {}).get("dig") or "none"
        selected_short_retract = dig_profiles.selected.get("short", {}).get("retract") or "none"
        selected_long_dig = dig_profiles.selected.get("long", {}).get("dig") or "none"
        selected_long_retract = dig_profiles.selected.get("long", {}).get("retract") or "none"
        cursor_profile = dig_profiles.get_cursor_profile()
        cursor_name = cursor_profile["name"] if cursor_profile is not None else "none"
        cursor_duration = float(cursor_profile.get("duration_sec", 0.0)) if cursor_profile is not None else 0.0
        recording_text = (
            f"Recording: {dig_profiles.recording_style.upper()} {dig_profiles.recording_phase.upper()}"
            if dig_profiles.recording and dig_profiles.recording_style and dig_profiles.recording_phase
            else (
                f"Preview: {str(dig_profile_preview_name or 'OFF')[:22]}"
                if dig_profile_preview_active
                else "Recording: OFF"
            )
        )
        summary_y = dig_body_y + 7 * (button_h + 10) + 34
        put_control_line(
            f"Active {dig_profiles.active_style.upper()} {dig_profiles.active_phase.upper()} | {recording_text}",
            summary_y,
            (180, 255, 200),
            0.42,
            x=card_x0 + card_inner,
        )
        active_selected_name = (
            dig_profiles.selected.get(dig_profiles.active_style, {}).get(dig_profiles.active_phase) or "none"
        )
        put_control_line(
            f"Using now: {active_selected_name[:44]}",
            summary_y + 24,
            (255, 255, 190),
            0.42,
            x=card_x0 + card_inner,
        )
        put_control_line(f"Short dig: {selected_short_dig[:42]}", summary_y + 48, (235, 235, 235), 0.40, x=card_x0 + card_inner)
        put_control_line(f"Short retract: {selected_short_retract[:38]}", summary_y + 70, (235, 235, 235), 0.40, x=card_x0 + card_inner)
        put_control_line(f"Long dig: {selected_long_dig[:43]}", summary_y + 92, (235, 235, 235), 0.40, x=card_x0 + card_inner)
        put_control_line(f"Long retract: {selected_long_retract[:39]}", summary_y + 114, (235, 235, 235), 0.40, x=card_x0 + card_inner)
        put_control_line(
            f"Browse {dig_profiles.active_style}/{dig_profiles.active_phase}: {cursor_name[:26]} ({cursor_duration:.2f}s)",
            summary_y + 138,
            (255, 230, 190),
            0.40,
            x=card_x0 + card_inner,
        )
        controller_cursor_macro = controller_macros.get_cursor_macro()
        controller_cursor_name = controller_cursor_macro["name"] if controller_cursor_macro is not None else "none"
        controller_cursor_duration = float(controller_cursor_macro.get("duration_sec", 0.0)) if controller_cursor_macro is not None else 0.0
        controller_selected_macro = controller_macros.get_selected_macro()
        controller_selected_name = controller_selected_macro["name"] if controller_selected_macro is not None else "none"
        put_control_line(
            f"Controller selected: {controller_selected_name[:30]}",
            summary_y + 162,
            (220, 210, 255),
            0.40,
            x=card_x0 + card_inner,
        )
        put_control_line(
            f"Browse controller: {controller_cursor_name[:24]} ({controller_cursor_duration:.2f}s)",
            summary_y + 184,
            (220, 210, 255),
            0.40,
            x=card_x0 + card_inner,
        )
        put_control_line(
            (
                f"Controller cycle: {controller_cycle_phase.upper()} {controller_cycle_preview_name[:20]}"
                if controller_cycle_preview_active and controller_cycle_preview_name
                else f"Controller cycle: OFF | +{controller_cycle_mechanism_hold_sec:.1f}s lower/extend hold"
            ),
            summary_y + 206,
            (220, 255, 200),
            0.40,
            x=card_x0 + card_inner,
        )
        visible_profiles = (
            current_dig_profiles
            if show_all_dig_profiles
            else current_dig_profiles[:4]
        )
        put_control_line(
            f"{dig_profiles.active_style.title()} {dig_profiles.active_phase.title()} recordings ({len(current_dig_profiles)} total):",
            summary_y + 230,
            (170, 210, 255),
            0.40,
            x=card_x0 + card_inner,
        )
        for idx, profile in enumerate(visible_profiles):
            marker = "* " if profile["name"] == dig_profiles.selected.get(dig_profiles.active_style, {}).get(dig_profiles.active_phase) else "  "
            cursor_marker = ">" if profile["name"] == dig_profiles.cursor.get(dig_profiles.active_style, {}).get(dig_profiles.active_phase) else " "
            line_y = summary_y + 252 + idx * 20
            put_control_line(
                f"{cursor_marker}{marker}{profile['name'][:34]} ({float(profile.get('duration_sec', 0.0)):.2f}s)",
                line_y,
                (230, 230, 230),
                0.38,
                x=card_x0 + card_inner,
            )
        cursor_y += dig_section_h + 8

        status_section_jump_targets["jump_setup"] = int(section_offsets.get("setup", 0))
        status_section_jump_targets["jump_map_tools"] = int(section_offsets.get("map_tools", 0))
        status_section_jump_targets["jump_zones_camera"] = int(section_offsets.get("zones_camera", 0))
        status_section_jump_targets["jump_calibration"] = int(section_offsets.get("calibration", 0))
        status_section_jump_targets["jump_actuators"] = int(section_offsets.get("actuators", 0))
        status_section_jump_targets["jump_dig_profiles"] = int(section_offsets.get("dig_profiles", 0))
        content_h = max(controls_h + 1, cursor_y)

        status_scroll_max = max(0, content_h - controls_h)
        status_scroll_y = max(0, min(status_scroll_y, status_scroll_max))
        panel[controls_top:controls_bottom, :] = controls[status_scroll_y:status_scroll_y + controls_h, :]
        cv2.rectangle(panel, (0, controls_top), (panel_w - 1, controls_bottom), (90, 90, 90), 1)
        status_button_rects["controls_viewport"] = (0, controls_top, panel_w - 1, controls_bottom)

        for name, rect in (
            ("manual_mode_toggle", manual_mode_rect),
            ("no_mapping_mode", no_mapping_rect),
            ("setup_low_latency_mode", setup_low_latency_rect),
            ("auto_run", auto_run_rect),
            ("auto_digger", auto_digger_rect),
            ("test_excavation_left_extend", test_excavation_left_extend_rect),
            ("test_excavation_right_extend", test_excavation_right_extend_rect),
            ("test_excavation_dig", test_excavation_dig_rect),
            ("test_excavation_lower", test_excavation_lower_rect),
            ("test_excavation_pattern", test_excavation_pattern_rect),
            ("direct_nav", direct_nav_rect),
            ("excav", excav_rect),
            ("deposit", deposit_rect),
            ("starting_zone", starting_zone_rect),
            ("set_berm_left", berm_left_rect),
            ("set_berm_right", berm_right_rect),
            ("whole", whole_rect),
            ("paint_obstacle", obstacle_rect),
            ("paint_safe", paint_rect),
            ("erase_safe", erase_rect),
            ("smooth_map", smooth_rect),
            ("holes", holes_rect),
            ("clear_paint", clear_paint_rect),
            ("reset_map", reset_map_rect),
            ("lock_green", lock_green_rect),
            ("pick_dig_start", pick_dig_start_rect),
            ("main_rover_mode", main_rover_rect),
            ("camera_view", camera_view_rect),
            ("camera_overlay", camera_overlay_rect),
            ("human_detect_toggle", human_detect_rect),
            ("rock_detect_toggle", rock_detect_rect),
            ("low_latency_mode", low_latency_rect),
            ("drive_heading_flip", drive_heading_flip_rect),
            ("hard_drive_flip", hard_drive_flip_rect),
            ("steering_flip", steering_flip_rect),
            ("test_drive_forward", test_drive_forward_rect),
            ("bidirectional_auto", bidirectional_auto_rect),
            ("demo_auto", demo_auto_rect),
            ("lock_start_frame", lock_start_frame_rect),
            ("scan_start_frame", scan_start_frame_rect),
            ("drive_speed_slider", drive_speed_slider_rect),
            ("turn_speed_slider", turn_speed_slider_rect),
            ("digger_speed_slider", digger_speed_slider_rect),
            ("camera_view_flip", camera_view_flip_rect),
            ("display_heading_flip", display_heading_flip_rect),
            ("drive_calibration_mode", drive_calibration_mode_rect),
            ("drive_calibration_cancel", drive_calibration_cancel_rect),
            ("dig_style_cycle", dig_style_cycle_rect),
            ("dig_phase_cycle", dig_phase_cycle_rect),
            ("dig_record_active", dig_record_active_rect),
            ("dig_profile_preview", dig_profile_preview_rect),
            ("dig_record_stop", dig_record_stop_rect),
            ("dig_profile_prev", dig_profile_prev_rect),
            ("dig_profile_next", dig_profile_next_rect),
            ("dig_profile_use", dig_profile_use_rect),
            ("dig_profile_delete", dig_profile_delete_rect),
            ("dig_profiles_view_all", dig_profiles_view_all_rect),
            ("controller_record", controller_record_rect),
            ("controller_preview", controller_preview_rect),
            ("controller_cycle", controller_cycle_rect),
            ("controller_stop", controller_stop_rect),
            ("controller_prev", controller_prev_rect),
            ("controller_next", controller_next_rect),
            ("controller_use", controller_use_rect),
            ("door_open", door_open_rect),
            ("door_close", door_close_rect),
            ("stop_actuators", stop_actuators_rect),
            ("brush_minus", brush_minus_rect),
            ("brush_plus", brush_plus_rect),
            ("brush_slider", brush_slider_rect),
        ):
            _register_button(name, rect)

        scrollbar_track_rect = (
            panel_w - scrollbar_margin - scrollbar_w,
            controls_top + 6,
            panel_w - scrollbar_margin,
            controls_bottom - 6,
        )
        track_h = max(1, scrollbar_track_rect[3] - scrollbar_track_rect[1])
        if status_scroll_max > 0:
            thumb_h = max(52, int(round(track_h * float(controls_h) / float(content_h))))
            thumb_h = min(track_h, thumb_h)
            thumb_travel = max(1, track_h - thumb_h)
            thumb_y0 = scrollbar_track_rect[1] + int(round((float(status_scroll_y) / float(status_scroll_max)) * thumb_travel))
        else:
            thumb_h = track_h
            thumb_y0 = scrollbar_track_rect[1]
        scrollbar_thumb_rect = (
            scrollbar_track_rect[0] + 2,
            thumb_y0,
            scrollbar_track_rect[2] - 2,
            thumb_y0 + thumb_h,
        )
        status_button_rects["scrollbar_track"] = scrollbar_track_rect
        status_button_rects["scrollbar_thumb"] = scrollbar_thumb_rect
        cv2.rectangle(
            panel,
            (scrollbar_track_rect[0], scrollbar_track_rect[1]),
            (scrollbar_track_rect[2], scrollbar_track_rect[3]),
            (34, 34, 38),
            -1,
        )
        cv2.rectangle(
            panel,
            (scrollbar_track_rect[0], scrollbar_track_rect[1]),
            (scrollbar_track_rect[2], scrollbar_track_rect[3]),
            (160, 160, 170),
            2,
        )
        thumb_fill = (0, 180, 255) if status_scroll_max > 0 else (72, 72, 78)
        thumb_border = (255, 255, 255) if status_scroll_max > 0 else (120, 120, 126)
        cv2.rectangle(
            panel,
            (scrollbar_thumb_rect[0], scrollbar_thumb_rect[1]),
            (scrollbar_thumb_rect[2], scrollbar_thumb_rect[3]),
            thumb_fill,
            -1,
        )
        cv2.rectangle(
            panel,
            (scrollbar_thumb_rect[0], scrollbar_thumb_rect[1]),
            (scrollbar_thumb_rect[2], scrollbar_thumb_rect[3]),
            thumb_border,
            2,
        )
        cv2.putText(
            panel,
            "SCROLL",
            (scrollbar_track_rect[0] - 12, max(controls_top - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (170, 220, 255),
            1,
            cv2.LINE_AA,
        )

        scroll_btn_x1 = scrollbar_track_rect[0] - 8
        scroll_btn_x0 = scroll_btn_x1 - 28
        scroll_up_rect = (scroll_btn_x0, controls_top + 6, scroll_btn_x1, controls_top + 34)
        scroll_down_rect = (scroll_btn_x0, controls_bottom - 34, scroll_btn_x1, controls_bottom - 6)
        status_button_rects["scroll_up"] = scroll_up_rect
        status_button_rects["scroll_down"] = scroll_down_rect
        for rect, lbl, enabled in (
            (scroll_up_rect, "^", status_scroll_y > 0),
            (scroll_down_rect, "v", status_scroll_y < status_scroll_max),
        ):
            x0b, y0b, x1b, y1b = rect
            fill = (70, 130, 220) if enabled else (50, 50, 50)
            cv2.rectangle(panel, (x0b, y0b), (x1b, y1b), fill, -1)
            cv2.rectangle(panel, (x0b, y0b), (x1b, y1b), (200, 200, 200), 1)
            tsz, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            cv2.putText(panel, lbl, (x0b + (x1b - x0b - tsz[0]) // 2, y0b + (y1b - y0b + tsz[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        put_line(
            (
                f"Controls scroll: {status_scroll_y}/{status_scroll_max} | "
                "Wheel, drag panel, drag scrollbar, Up/Down, PgUp/PgDn, j/k, 1-5 sections"
            ),
            controls_top - 8,
            (170, 200, 230),
            0.42,
        )

        if reset_map_confirm:
            overlay = panel.copy()
            overlay[:] = (0, 0, 0)
            panel[:] = cv2.addWeighted(panel, 0.35, overlay, 0.65, 0)
            box = (70, 230, panel_w - 70, 430)
            cv2.rectangle(panel, (box[0], box[1]), (box[2], box[3]), (36, 36, 36), -1)
            cv2.rectangle(panel, (box[0], box[1]), (box[2], box[3]), (210, 210, 210), 1)
            cv2.putText(panel, "Confirm Map Reset", (box[0] + 22, box[1] + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, "This clears map evidence, painted cells,", (box[0] + 22, box[1] + 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(panel, "current goal/path, and saved AI landmarks.", (box[0] + 22, box[1] + 106),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
            cancel_rect = (box[0] + 22, box[3] - 58, box[0] + 190, box[3] - 18)
            confirm_rect = (box[2] - 210, box[3] - 58, box[2] - 22, box[3] - 18)
            status_button_rects["reset_cancel"] = cancel_rect
            status_button_rects["reset_confirm"] = confirm_rect
            for rect, label, fill in (
                (cancel_rect, "Cancel", (70, 130, 220)),
                (confirm_rect, "Reset Map", (0, 70, 220)),
            ):
                x0b, y0b, x1b, y1b = rect
                cv2.rectangle(panel, (x0b, y0b), (x1b, y1b), fill, -1)
                cv2.rectangle(panel, (x0b, y0b), (x1b, y1b), (230, 230, 230), 1)
                tsz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.putText(panel, label,
                            (x0b + (x1b - x0b - tsz[0]) // 2, y0b + (y1b - y0b + tsz[1]) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return panel

    publish_map_ui_state(force=True)

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            frame_idx += 1
            imu_rotation = None
            imu_estimated_R_world_cam = None
            imu_fallback_forward_world = None
            imu_heading_available = False
            navx_yaw_deg = read_navx_yaw_deg()
            navx_estimated_rover_forward_world = None
            navx_estimated_rover_right_world = None
            heading_fallback_forward_world = None
            if imu_heading_enabled:
                imu_rotation, imu_heading_warned, imu_heading_available = zed_utils.get_imu_rotation_with_status(
                    zed,
                    sl,
                    imu_sensors_data,
                    imu_heading_warned,
                )
            # Update camera pose (world frame) if tracking is enabled.
            if tracking_enabled:
                raw_R_world_cam, raw_t_world_cam, pose_warned, tracking_pose_ok = zed_utils.get_world_transform_with_status(
                    zed, sl, pose, pose_warned
                )
                if imu_heading_available:
                    imu_estimated_R_world_cam = estimate_world_rotation_from_imu(imu_rotation)
                if tracking_pose_ok and recovery_pending_alignment:
                    predicted_forward = np.array(last_valid_rover_forward_world, dtype=np.float32).reshape(3,)
                    if (
                        navx_yaw_deg is not None
                        and np.isfinite(navx_yaw_deg)
                        and last_valid_navx_yaw_deg is not None
                        and np.isfinite(last_valid_navx_yaw_deg)
                        and navx_sign_locked
                    ):
                        predicted_forward = rotate_world_xz(
                            last_valid_rover_forward_world,
                            navx_sign * wrap_angle_deg(float(navx_yaw_deg) - float(last_valid_navx_yaw_deg)),
                        )
                    raw_rover_pos_world, raw_rover_forward_world, _raw_rover_right_world = rover_pose_from_camera(
                        raw_R_world_cam,
                        raw_t_world_cam,
                        current_camera_mount_yaw_deg(),
                    )
                    recovery_alignment_yaw_deg = heading_delta_deg(
                        raw_rover_forward_world,
                        predicted_forward,
                    )
                    align_R = yaw_rotation_matrix_deg(recovery_alignment_yaw_deg)
                    desired_t_world = np.array(last_valid_t_world_cam, dtype=np.float32).reshape(3,)
                    recovery_alignment_offset_t = (
                        desired_t_world - (align_R @ np.array(raw_t_world_cam, dtype=np.float32).reshape(3,))
                    ).astype(np.float32)
                    recovery_pending_alignment = False
                    print(
                        "Recovery alignment locked: "
                        f"yaw={recovery_alignment_yaw_deg:+.1f}deg "
                        f"offset=({recovery_alignment_offset_t[0]:+.2f}, {recovery_alignment_offset_t[2]:+.2f})"
                    )
                R_world_cam, t_world_cam = apply_recovery_alignment(raw_R_world_cam, raw_t_world_cam)
                candidate_rover_pos_world, candidate_rover_forward_world, candidate_rover_right_world = rover_pose_from_camera(
                    R_world_cam,
                    t_world_cam,
                    current_camera_mount_yaw_deg(),
                )
                navx_estimated_rover_forward_world, navx_estimated_rover_right_world = estimate_rover_axes_from_navx(navx_yaw_deg)
                if tracking_pose_ok and have_valid_tracking_pose:
                    pose_jump_m = float(
                        np.linalg.norm(np.array(t_world_cam, dtype=np.float32) - last_valid_t_world_cam)
                    )
                    heading_jump_deg = angle_between_vec_deg(
                        world_forward_from_rotation(R_world_cam),
                        world_forward_from_rotation(last_valid_R_world_cam),
                    )
                    jump_reason = None
                    if (
                        float(args.tracking_max_pose_jump_m) > 0.0
                        and pose_jump_m > float(args.tracking_max_pose_jump_m)
                    ):
                        jump_reason = (
                            f"pose jump {pose_jump_m:.2f}m > {float(args.tracking_max_pose_jump_m):.2f}m"
                        )
                    elif (
                        float(args.tracking_max_heading_jump_deg) > 0.0
                        and heading_jump_deg > float(args.tracking_max_heading_jump_deg)
                    ):
                        jump_reason = (
                            f"heading jump {heading_jump_deg:.1f}deg > "
                            f"{float(args.tracking_max_heading_jump_deg):.1f}deg"
                        )
                    elif (
                        imu_estimated_R_world_cam is not None
                        and float(args.imu_heading_max_mismatch_deg) > 0.0
                    ):
                        imu_heading_mismatch_deg = angle_between_vec_deg(
                            world_forward_from_rotation(R_world_cam),
                            world_forward_from_rotation(imu_estimated_R_world_cam),
                        )
                        if imu_heading_mismatch_deg > float(args.imu_heading_max_mismatch_deg):
                            jump_reason = (
                                f"IMU mismatch {imu_heading_mismatch_deg:.1f}deg > "
                                f"{float(args.imu_heading_max_mismatch_deg):.1f}deg"
                            )
                    elif (
                        navx_estimated_rover_forward_world is not None
                        and float(args.navx_heading_max_mismatch_deg) > 0.0
                    ):
                        navx_heading_mismatch_deg = angle_between_vec_deg(
                            candidate_rover_forward_world,
                            navx_estimated_rover_forward_world,
                        )
                        if navx_heading_mismatch_deg > float(args.navx_heading_max_mismatch_deg):
                            jump_reason = (
                                f"NavX mismatch {navx_heading_mismatch_deg:.1f}deg > "
                                f"{float(args.navx_heading_max_mismatch_deg):.1f}deg"
                            )
                    if jump_reason is not None:
                        if recovery_loaded_from_checkpoint or recovery_pending_alignment:
                            recovery_jump_reject_count += 1
                            if recovery_jump_reject_count >= 5:
                                print(
                                    "Startup recovery pose kept being rejected; "
                                    "clearing saved recovery alignment and waiting for fresh tracking."
                                )
                                have_valid_tracking_pose = False
                                tracking_prev_ok = False
                                tracking_loss_warned = False
                                tracking_recover_stable_count = 0
                                recovery_pending_alignment = False
                                recovery_loaded_from_checkpoint = False
                                recovery_alignment_offset_t = np.zeros(3, dtype=np.float32)
                                recovery_alignment_yaw_deg = 0.0
                                last_valid_R_world_cam = np.eye(3, dtype=np.float32)
                                last_valid_t_world_cam = np.zeros(3, dtype=np.float32)
                                last_valid_rover_forward_world = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                                last_valid_rover_right_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                                last_valid_imu_rotation = None
                                last_valid_navx_yaw_deg = None
                                map_origin_set = False
                                map_origin_t = np.zeros(3, dtype=np.float32)
                                goal_cell = None
                                path_cells = None
                                last_path_cells = None
                                last_start = None
                                last_goal = None
                                last_path_plan_time = 0.0
                                jump_reason = None
                        else:
                            recovery_jump_reject_count = 0
                    else:
                        recovery_jump_reject_count = 0
                    if jump_reason is not None:
                        tracking_pose_ok = False
                        tracking_recover_stable_count = 0
                        R_world_cam = last_valid_R_world_cam
                        t_world_cam = last_valid_t_world_cam
                        print(f"Tracking jump rejected ({jump_reason}); holding last pose.")
                if tracking_pose_ok and have_valid_tracking_pose and not tracking_prev_ok:
                    tracking_recover_stable_count += 1
                    recover_needed = max(1, int(args.tracking_recover_stable_frames))
                    if tracking_recover_stable_count < recover_needed:
                        tracking_pose_ok = False
                        R_world_cam = last_valid_R_world_cam
                        t_world_cam = last_valid_t_world_cam
                        if tracking_recover_stable_count == 1:
                            print(
                                "Tracking candidate recovered; waiting for "
                                f"{recover_needed} stable frame(s) before relocking."
                            )
                    else:
                        print("Tracking recovered: relocalized/locked.")
                elif not tracking_pose_ok:
                    tracking_recover_stable_count = 0
                if tracking_pose_ok:
                    last_valid_R_world_cam = R_world_cam
                    last_valid_t_world_cam = t_world_cam
                    last_valid_rover_forward_world = np.array(candidate_rover_forward_world, dtype=np.float32).reshape(3,)
                    last_valid_rover_right_world = np.array(candidate_rover_right_world, dtype=np.float32).reshape(3,)
                    if imu_heading_available:
                        last_valid_imu_rotation = np.array(imu_rotation, dtype=np.float32).reshape(3, 3)
                    if navx_yaw_deg is not None and np.isfinite(navx_yaw_deg):
                        last_valid_navx_yaw_deg = float(navx_yaw_deg)
                        update_navx_sign_calibration(navx_yaw_deg, candidate_rover_forward_world)
                    have_valid_tracking_pose = True
                    tracking_loss_warned = False
                    landmark_pose_override_t_map = None
                    landmark_pose_override_R_world_cam = None
                    if not args.complex and not map_origin_set:
                        map_origin_t = np.array(t_world_cam, dtype=np.float32)
                        map_origin_set = True
                        print(
                            "Map origin anchored at "
                            f"x={map_origin_t[0]:+.2f}, y={map_origin_t[1]:+.2f}, z={map_origin_t[2]:+.2f}"
                        )
                else:
                    # Hold last known pose and pause map integration until tracking recovers.
                    if imu_estimated_R_world_cam is not None:
                        imu_fallback_forward_world = world_forward_from_rotation(imu_estimated_R_world_cam)
                    if navx_estimated_rover_forward_world is not None and navx_estimated_rover_right_world is not None:
                        heading_fallback_forward_world = camera_forward_from_rover_axes(
                            navx_estimated_rover_forward_world,
                            navx_estimated_rover_right_world,
                            current_camera_mount_yaw_deg(),
                        )
                    elif imu_fallback_forward_world is not None:
                        heading_fallback_forward_world = imu_fallback_forward_world
                    if have_valid_tracking_pose:
                        R_world_cam = last_valid_R_world_cam
                        t_world_cam = last_valid_t_world_cam
                    if not tracking_loss_warned:
                        if heading_fallback_forward_world is not None and navx_estimated_rover_forward_world is not None:
                            print("Tracking lost: holding last pose, using NavX heading fallback for display, and pausing map integration.")
                        elif heading_fallback_forward_world is not None:
                            print("Tracking lost: holding last pose, using IMU heading fallback for display, and pausing map integration.")
                        else:
                            print("Tracking lost: holding last pose and pausing map integration.")
                        tracking_loss_warned = True
                tracking_prev_ok = tracking_pose_ok
            else:
                R_world_cam = np.eye(3, dtype=np.float32)
                t_world_cam = np.zeros(3, dtype=np.float32)
                tracking_pose_ok = True
                tracking_prev_ok = True
            update_localization_scan_state()
            refresh_camera_servo_state()
            refresh_ds_joystick_state()

            if args.camera_only:
                zed.retrieve_image(image_left, sl.VIEW.LEFT)

                if sd is not None:
                    now = time.time()
                    if (now - last_drive_send) >= (1.0 / max(1.0, args.drive_rate_hz)):
                        last_drive_send = now
                        refresh_ds_joystick_state()
                        controller_macro_playback_cmd = None
                        if controller_macros.recording:
                            record_fwd, record_turn = (
                                mix_ds_drive(manual_fwd, manual_turn)
                                if manual_mode else (float(ds_joystick_fwd), float(ds_joystick_turn))
                            )
                            mechanism_state = current_controller_macro_mechanism_state()
                            if abs(record_fwd) < 0.05:
                                record_fwd = 0.0
                            if abs(record_turn) < 0.05:
                                record_turn = 0.0
                            controller_macros.capture_sample(
                                now,
                                record_fwd,
                                record_turn,
                                mechanism_state["digger_on"],
                                mechanism_state["lower_on"],
                                mechanism_state["left_extend_on"],
                                mechanism_state["right_extend_on"],
                                mechanism_state["door_open_on"],
                                mechanism_state["door_close_on"],
                            )
                        if emergency_stop:
                            reset_auto_drive_shape(now)
                            send_nt_command(False, 0.0, 0.0, 0.1)
                        elif excavation_pattern_test_active:
                            pattern_state = excavation_pattern_state(now)
                            if pattern_state is None or pattern_state.get("done"):
                                excavation_pattern_test_active = False
                                excavation_pattern_test_started_at = 0.0
                                reset_auto_drive_shape(now)
                                send_nt_command(False, 0.0, 0.0, 0.1)
                                print("Excavation pattern test completed.")
                                publish_map_ui_state(force=True)
                            else:
                                reset_auto_drive_shape(now)
                                send_nt_command(
                                    True,
                                    float(pattern_state.get("fwd", 0.0)),
                                    0.0,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                        elif test_drive_forward_active:
                            if now >= test_drive_forward_until:
                                test_drive_forward_active = False
                                test_drive_forward_until = 0.0
                                reset_auto_drive_shape(now)
                                send_nt_command(False, 0.0, 0.0, 0.1)
                                print("Forward drive test completed.")
                                publish_map_ui_state(force=True)
                            else:
                                reset_auto_drive_shape(now)
                                send_nt_command(
                                    True,
                                    max(0.0, min(1.0, float(args.drive_speed))) * 0.45,
                                    0.0,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                        elif controller_cycle_preview_active:
                            cycle_macro = resolve_preview_controller_macro()
                            elapsed_preview = now - float(controller_cycle_phase_started_at)
                            controller_macro_playback_cmd = controller_macros.playback_sample_for_macro(
                                cycle_macro,
                                elapsed_preview,
                                mode=("return" if controller_cycle_phase == "return" else "forward"),
                                mechanism_hold_sec=(
                                    float(controller_cycle_mechanism_hold_sec)
                                    if controller_cycle_phase == "forward" else 0.0
                                ),
                                suppress_mechanisms=bool(controller_cycle_phase == "return"),
                            )
                            macro_duration = 0.0
                            if cycle_macro is not None:
                                macro_duration = float(cycle_macro.get("duration_sec", 0.0))
                            if controller_macro_playback_cmd is None or elapsed_preview > max(0.05, macro_duration):
                                if controller_cycle_phase == "forward":
                                    controller_cycle_phase = "return"
                                    controller_cycle_phase_started_at = now
                                    print("Controller cycle: returning to recorded start point.")
                                    publish_map_ui_state(force=True)
                                else:
                                    controller_cycle_phase = "forward"
                                    controller_cycle_phase_started_at = now
                                    print("Controller cycle: restarting recorded pass.")
                                    publish_map_ui_state(force=True)
                                controller_macro_playback_cmd = None
                            else:
                                reset_auto_drive_shape(now)
                                preview_fwd, preview_turn = mix_ds_drive(
                                    controller_macro_playback_cmd["fwd"],
                                    controller_macro_playback_cmd["turn"],
                                )
                                send_nt_command(
                                    True,
                                    preview_fwd,
                                    preview_turn,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                        elif controller_macro_preview_active:
                            elapsed_preview = now - float(controller_macro_preview_started_at)
                            preview_macro = resolve_preview_controller_macro()
                            controller_macro_playback_cmd = controller_macros.playback_sample_for_macro(preview_macro, elapsed_preview)
                            macro_duration = 0.0
                            if preview_macro is not None:
                                macro_duration = float(preview_macro.get("duration_sec", 0.0))
                            if controller_macro_playback_cmd is None or elapsed_preview > max(0.05, macro_duration):
                                stop_controller_macro_preview("auto", completed=True)
                            else:
                                reset_auto_drive_shape(now)
                                preview_fwd, preview_turn = mix_ds_drive(
                                    controller_macro_playback_cmd["fwd"],
                                    controller_macro_playback_cmd["turn"],
                                )
                                send_nt_command(
                                    True,
                                    preview_fwd,
                                    preview_turn,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                        elif manual_mode:
                            reset_auto_drive_shape(now)
                            _man_fwd, _man_turn = mix_ds_drive(manual_fwd, manual_turn)
                            send_nt_command(
                                True,
                                _man_fwd,
                                _man_turn,
                                1.0 / max(1.0, args.drive_rate_hz),
                            )
                        else:
                            reset_auto_drive_shape(now)
                            send_nt_command(False, 0.0, 0.0, 0.1)

                if HAS_CV2:
                    img = image_left.get_data()
                    vis = None
                    if img is not None:
                        if img.ndim == 2:
                            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        elif img.ndim == 3:
                            if img.shape[2] == 4:
                                vis = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                            elif img.shape[2] == 1:
                                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            elif img.shape[2] >= 3:
                                vis = img[:, :, :3].copy()
                    if vis is not None:
                        cv2.putText(
                            vis,
                            "CAMERA ONLY",
                            (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 220, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        if not args.no_gui:
                            cv2.imshow("ZED Ground/Obstacle Segmentation", vis)
                    process_external_map_command()
                    publish_map_ui_state()
                    if not args.no_gui:
                        status_panel = render_status_panel(None)
                        last_status_panel_shape = status_panel.shape[:2]
                        cv2.imshow("ZED Drive Status", status_panel)
                        if not status_window_ready:
                            cv2.setMouseCallback("ZED Drive Status", on_status_click)
                            status_window_ready = True
                        raw_key = cv2.waitKeyEx(1)
                        key = (raw_key & 0xFF) if raw_key >= 0 else -1
                        if key == ord("q"):
                            break
                        if key == ord("m"):
                            set_manual_drive_mode(not manual_mode, "key")
                        if key == ord("u"):
                            set_main_rover_mode(not args.main_rover_mode)
                        if raw_key in UP_KEYS:
                            set_status_scroll(-80)
                        if raw_key in DOWN_KEYS:
                            set_status_scroll(80)
                        if raw_key in PAGEUP_KEYS or key == ord("k"):
                            set_status_scroll(-260)
                        if raw_key in PAGEDOWN_KEYS or key == ord("j"):
                            set_status_scroll(260)
                        if raw_key in HOME_KEYS:
                            set_status_scroll_to(0)
                        if raw_key in END_KEYS:
                            set_status_scroll_to(status_scroll_max)
                        if key == ord("1"):
                            set_status_scroll_to(status_section_jump_targets.get("jump_setup", 0))
                        if key == ord("2"):
                            set_status_scroll_to(status_section_jump_targets.get("jump_map_tools", 0))
                        if key == ord("3"):
                            set_status_scroll_to(status_section_jump_targets.get("jump_zones_camera", 0))
                        if key == ord("4"):
                            set_status_scroll_to(status_section_jump_targets.get("jump_calibration", 0))
                        if key == ord("5"):
                            set_status_scroll_to(status_section_jump_targets.get("jump_actuators", 0))
                        if key == ord("6"):
                            set_status_scroll_to(status_section_jump_targets.get("jump_dig_profiles", 0))
                        now_key = time.time()
                        if key == ord("w"):
                            manual_fwd = max(0.0, min(1.0, args.drive_speed))
                            last_w_time = now_key
                        if key == ord("s"):
                            manual_fwd = -max(0.0, min(1.0, args.drive_speed))
                            last_s_time = now_key
                        if key == ord("a"):
                            manual_turn = max(0.0, min(1.0, args.drive_speed))
                            last_a_time = now_key
                        if key == ord("d"):
                            manual_turn = -max(0.0, min(1.0, args.drive_speed))
                            last_d_time = now_key
                        if key == ord("x"):
                            manual_fwd = 0.0
                            manual_turn = 0.0
                        if key == ord(" "):
                            emergency_stop = True
                            manual_fwd = 0.0
                            manual_turn = 0.0
                        if manual_mode:
                            if now_key - last_w_time > key_hold_timeout and now_key - last_s_time > key_hold_timeout:
                                manual_fwd = 0.0
                            if now_key - last_a_time > key_hold_timeout and now_key - last_d_time > key_hold_timeout:
                                manual_turn = 0.0
                    else:
                        time.sleep(0.01)
                    continue

            # Retrieve point cloud
            depth_status = zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            last_depth_status = str(depth_status)
            # Fit/update floor plane on an interval to reduce SDK load.
            now = time.time()
            should_update_plane = (not has_plane) or ((now - last_plane_update_time) >= args.floor_update_sec)
            if should_update_plane:
                status = zed.find_floor_plane(ground_plane, tracking_reset)
                last_plane_update_time = now
                if status == sl.ERROR_CODE.SUCCESS:
                    a0, b0, c0, d0 = segmentation.plane_params(ground_plane)
                    a0, b0, c0, d0 = segmentation.canonical_plane(a0, b0, c0, d0)
                    if abs(float(b0)) >= float(args.floor_min_normal_y):
                        if args.complex:
                            # Complex mode: EMA smoothing + tilt/jump rejection
                            accept_plane = True
                            if has_plane:
                                prev_n = np.array([a, b, c], dtype=np.float32)
                                new_n = np.array([a0, b0, c0], dtype=np.float32)
                                dot = float(np.clip(np.dot(prev_n, new_n), -1.0, 1.0))
                                tilt_deg = float(np.degrees(np.arccos(dot)))
                                d_jump = abs(float(d0) - float(d))
                                if (
                                    tilt_deg > float(args.plane_max_tilt_delta_deg)
                                    or d_jump > float(args.plane_max_height_jump_m)
                                ):
                                    accept_plane = False
                                    plane_reject_count += 1
                                    if plane_reject_count % 10 == 1:
                                        print(
                                            "Rejected floor plane jump: "
                                            f"tilt_delta={tilt_deg:.2f}deg d_jump={d_jump:.3f}m"
                                        )
                                    if (
                                        int(args.plane_force_accept_rejects) > 0
                                        and plane_reject_count >= int(args.plane_force_accept_rejects)
                                    ):
                                        print(
                                            "Too many plane rejections; force-accepting new plane "
                                            "to recover live mapping."
                                        )
                                        accept_plane = True
                            if accept_plane:
                                alpha = max(0.0, min(1.0, float(args.plane_ema_alpha)))
                                if not has_plane or alpha >= 1.0:
                                    a, b, c, d = a0, b0, c0, d0
                                else:
                                    blend_n = (1.0 - alpha) * np.array([a, b, c], dtype=np.float32) + (
                                        alpha * np.array([a0, b0, c0], dtype=np.float32)
                                    )
                                    n_norm = float(np.linalg.norm(blend_n))
                                    if n_norm <= 1e-6:
                                        blend_n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                                        n_norm = 1.0
                                    blend_n /= n_norm
                                    a, b, c = float(blend_n[0]), float(blend_n[1]), float(blend_n[2])
                                    d = float((1.0 - alpha) * float(d) + alpha * float(d0))
                                has_plane = True
                                plane_fail_count = 0
                                plane_reject_count = 0
                        else:
                            # Simple mode: accept plane directly, no smoothing
                            a, b, c, d = a0, b0, c0, d0
                            has_plane = True
                            plane_fail_count = 0
                            plane_reject_count = 0
                    else:
                        plane_fail_count += 1
                        if plane_fail_count % 10 == 1:
                            print(f"Rejected floor plane with weak normal.y={b0:.3f}")
                else:
                    plane_fail_count += 1
                    if plane_fail_count % 10 == 1:
                        print(f"find_floor_plane failed: {status}")

            if not has_plane:
                # No valid plane yet; wait for a stable floor estimate.
                continue

            # Sample a downscaled cloud to compute a quick summary
            cloud = point_cloud.get_data()
            if cloud is None:
                last_raw_point_count = 0
                last_in_range_point_count = 0
                range_filter_bypassed = False
                last_depth_status = "no-cloud"
                continue
            # Downsample for speed
            stride = max(1, int(args.sample_stride))
            if driver_priority_active:
                stride = max(stride, int(max(1, args.driver_priority_sample_stride)))
            xyz_all = cloud[::stride, ::stride, :3].reshape(-1, 3)
            # Filter invalid points
            finite_mask = np.isfinite(xyz_all).all(axis=1)
            last_raw_point_count = int(np.count_nonzero(finite_mask))
            mask = finite_mask.copy()
            if float(args.min_range_z_m) > 0.0:
                mask &= xyz_all[:, 2] >= float(args.min_range_z_m)
            if float(args.max_range_z_m) > 0.0:
                mask &= xyz_all[:, 2] <= float(args.max_range_z_m)
            last_in_range_point_count = int(np.count_nonzero(mask))
            range_filter_bypassed = False
            if last_in_range_point_count == 0 and last_raw_point_count > 0:
                xyz = xyz_all[finite_mask]
                range_filter_bypassed = True
                if no_points_count % 30 == 0:
                    print(
                        "All finite depth points failed Z-range filtering; "
                        "temporarily bypassing min/max Z filter for mapping diagnostics."
                    )
            else:
                xyz = xyz_all[mask]

            if xyz.size == 0:
                no_points_count += 1
                if no_points_count % 30 == 1:
                    print(
                        "No valid depth points after filtering; "
                        "check depth validity, min/max Z limits, and lighting/texture."
                    )
                dist = np.empty((0,), dtype=np.float32)
                ground_mask = np.zeros((0,), dtype=bool)
                obstacle_mask = np.zeros((0,), dtype=bool)
                hole_mask = np.zeros((0,), dtype=bool)
                ground_pct = 0.0
                obstacle_pct = 0.0
                hole_pct = 0.0
                if last_raw_point_count <= 0:
                    last_depth_status = f"{depth_status}|all-invalid"
                elif range_filter_bypassed:
                    last_depth_status = f"{depth_status}|bypass-empty"
                else:
                    last_depth_status = f"{depth_status}|range-empty"
            else:
                no_points_count = 0
                if range_filter_bypassed:
                    last_depth_status = f"{depth_status}|range-bypassed"
                else:
                    last_depth_status = f"{depth_status}|ok"
                # Distance to plane (signed)
                dist, ground_mask, obstacle_mask = segmentation.classify_points(
                    xyz, a, b, c, d, ground_thresh=args.obstacle_thresh_m
                )
                # Use an explicit signed band for ground classification:
                #   - above lower bound (hole threshold)
                #   - below upper bound (obstacle threshold)
                # This avoids "unknown gaps" between hole and ground and keeps sky/ceiling
                # from being marked ground when max-above filtering is active.
                ground_mask = (dist >= -float(args.hole_thresh_m)) & (dist <= float(args.obstacle_thresh_m))
                obstacle_mask = dist > float(args.obstacle_thresh_m)
                if args.max_above_ground_m > 0.0:
                    keep_mask = dist <= float(args.max_above_ground_m)
                    if np.any(keep_mask):
                        xyz = xyz[keep_mask]
                        dist = dist[keep_mask]
                        ground_mask = ground_mask[keep_mask]
                        obstacle_mask = obstacle_mask[keep_mask]
                    else:
                        xyz = np.empty((0, 3), dtype=np.float32)
                        dist = np.empty((0,), dtype=np.float32)
                        ground_mask = np.zeros((0,), dtype=bool)
                        obstacle_mask = np.zeros((0,), dtype=bool)
                if disable_holes:
                    hole_mask = np.zeros(dist.shape, dtype=bool)
                else:
                    hole_mask = dist < -args.hole_thresh_m
                if xyz.shape[0] > 0:
                    ground_pct = 100.0 * np.count_nonzero(ground_mask) / xyz.shape[0]
                    obstacle_pct = 100.0 * np.count_nonzero(obstacle_mask) / xyz.shape[0]
                    hole_pct = 100.0 * np.count_nonzero(hole_mask) / xyz.shape[0]
                else:
                    ground_pct = 0.0
                    obstacle_pct = 0.0
                    hole_pct = 0.0
                last_map_point_count = int(xyz.shape[0])
                last_ground_pct = float(ground_pct)
                last_obstacle_pct = float(obstacle_pct)
                last_hole_pct = float(hole_pct)

            close_obstacle_detected = False
            close_obstacle_min_z = None
            if (
                xyz.shape[0] > 0
                and float(args.backup_close_dist_m) > 0.0
                and (int(args.backup_min_obstacle_points) > 0 or int(args.backup_critical_min_points) > 0)
            ):
                lane_half = max(0.05, float(args.backup_lane_half_width_m))
                in_lane = (
                    obstacle_mask
                    & (xyz[:, 2] > 0.0)
                    & (np.abs(xyz[:, 0]) <= lane_half)
                )
                close_mask = in_lane & (xyz[:, 2] <= float(args.backup_close_dist_m))
                close_count = int(np.count_nonzero(close_mask))
                critical_count = 0
                critical_mask = None
                if float(args.backup_critical_dist_m) > 0.0:
                    critical_mask = in_lane & (xyz[:, 2] <= float(args.backup_critical_dist_m))
                    critical_count = int(np.count_nonzero(critical_mask))
                critical_trigger = (
                    float(args.backup_critical_dist_m) > 0.0
                    and int(args.backup_critical_min_points) > 0
                    and critical_count >= int(args.backup_critical_min_points)
                )
                close_trigger = (
                    int(args.backup_min_obstacle_points) > 0
                    and close_count >= int(args.backup_min_obstacle_points)
                )
                if critical_trigger or close_trigger:
                    close_obstacle_detected = True
                    trigger_mask = close_mask
                    if critical_trigger and critical_mask is not None and np.any(critical_mask):
                        trigger_mask = critical_mask
                    if np.any(trigger_mask):
                        close_obstacle_min_z = float(np.min(xyz[trigger_mask, 2]))

            print(
                f"Ground {ground_pct:5.1f}% | Obstacles {obstacle_pct:5.1f}% "
                f"| Holes {hole_pct:5.1f}% | Points {xyz.shape[0]}"
            )

            # Publish point cloud to ROS2 (optional)
            if xyz.shape[0] > 0:
                ros2_utils.publish_pointcloud(node, pc_pub, pc_fields, xyz, args.frame)

            # Build a simple 2D top-down occupancy map (XZ) from ground/obstacle points.
            if HAS_CV2:
                map_vis = None
                heatmap_vis = None
                cam_row_col = None
                rover_row_col = None
                drive_origin_pos_map = None
                drive_origin_row_col = None
                rover_heading_vec_rc = None
                camera_map_pause_reason = ""
                current_mount_yaw_deg = current_camera_mount_yaw_deg()
                _, _, close_obstacle_escape_sign = camera_mount_axes(current_mount_yaw_deg)
                # Never integrate new points while tracking is lost or a pose
                # jump was rejected; otherwise one bad pose can drag the map.
                map_integration_ok = (not tracking_enabled) or tracking_pose_ok
                if args.camera_servo_track and servo_turning:
                    map_integration_ok = False
                    camera_map_pause_reason = "CAMERA TURNING"
                # Compute map-local translation for simple mode
                if not args.complex and map_origin_set:
                    t_map = np.array(t_world_cam, dtype=np.float32) - map_origin_t
                else:
                    t_map = np.array(t_world_cam, dtype=np.float32)
                if tracking_enabled and (not tracking_pose_ok) and landmark_pose_override_t_map is not None:
                    t_map = np.array(landmark_pose_override_t_map, dtype=np.float32).reshape(3,)
                    if landmark_pose_override_R_world_cam is not None:
                        R_world_cam = np.array(landmark_pose_override_R_world_cam, dtype=np.float32).reshape(3, 3)
                    if not camera_map_pause_reason:
                        camera_map_pause_reason = "TRACKING LOST - LANDMARK HOLD"
                rover_pos_map, rover_forward_world, rover_right_world = rover_pose_from_camera(
                    R_world_cam,
                    t_map,
                    current_mount_yaw_deg,
                )
                actual_rover_pos_map = np.array(rover_pos_map, dtype=np.float32).reshape(3,)
                actual_rover_forward_world = np.array(rover_forward_world, dtype=np.float32).reshape(3,)
                cam_row_col = map_world_to_grid(t_map[0], t_map[2])
                rover_row_col = map_world_to_grid(rover_pos_map[0], rover_pos_map[2])
                drive_origin_pos_map = navigation_origin_world(rover_pos_map, rover_forward_world)
                if drive_origin_pos_map is not None:
                    drive_origin_row_col = map_world_to_grid(drive_origin_pos_map[0], drive_origin_pos_map[2])
                mining_running_now = mining.state in (
                    auto_mining.MiningState.PLAN_SWEEP,
                    auto_mining.MiningState.NAVIGATE_DIG,
                    auto_mining.MiningState.DIGGING,
                    auto_mining.MiningState.BACKUP,
                    auto_mining.MiningState.NAVIGATE_DEPOSIT,
                    auto_mining.MiningState.DEPOSITING,
                )
                if start_frame_scan_active:
                    if tracking_enabled and (not tracking_pose_ok):
                        start_frame_scan_active = False
                        start_frame_scan_started_at = 0.0
                        start_frame_scan_samples = []
                        start_frame_last_status = "Start frame: scan canceled, tracking not locked."
                        start_frame_last_error_m = None
                        print(start_frame_last_status)
                        publish_map_ui_state(force=True)
                    else:
                        try:
                            _img_raw = image_left.get_data()
                            if _img_raw is not None:
                                if _img_raw.ndim == 3 and _img_raw.shape[2] == 4:
                                    _img_bgr = cv2.cvtColor(_img_raw, cv2.COLOR_BGRA2BGR)
                                elif _img_raw.ndim == 3 and _img_raw.shape[2] >= 3:
                                    _img_bgr = _img_raw[:, :, :3].copy()
                                else:
                                    _img_bgr = None
                            else:
                                _img_bgr = None
                            if _img_bgr is not None:
                                _scan_detections, _scan_error = detect_start_frame_tags(_img_bgr, cloud, R_world_cam, t_map)
                                if _scan_error is None:
                                    start_frame_scan_samples.append(_scan_detections)
                                    start_frame_last_ids = [int(item["id"]) for item in _scan_detections]
                                    start_frame_last_map_points = [
                                        {
                                            "id": int(item["id"]),
                                            "map_x": float(item["map_xz"][0]),
                                            "map_z": float(item["map_xz"][1]),
                                        }
                                        for item in _scan_detections
                                    ]
                                    start_frame_last_status = (
                                        f"Start frame: scanning {len(start_frame_scan_samples)}/"
                                        f"{start_frame_scan_min_samples} samples."
                                    )
                        except Exception as exc:
                            start_frame_scan_active = False
                            start_frame_scan_started_at = 0.0
                            start_frame_scan_samples = []
                            start_frame_last_status = f"Start frame: scan failed ({exc})."
                            start_frame_last_error_m = None
                            print(start_frame_last_status)
                            publish_map_ui_state(force=True)
                        if start_frame_scan_active:
                            elapsed_scan = now - float(start_frame_scan_started_at)
                            if (
                                len(start_frame_scan_samples) >= int(start_frame_scan_min_samples)
                                and elapsed_scan >= float(start_frame_scan_duration_sec)
                            ):
                                tag_groups = {}
                                for sample in start_frame_scan_samples:
                                    for item in sample:
                                        tag_groups.setdefault(int(item["id"]), []).append(item)
                                averaged_detections = []
                                for tag_id in sorted(tag_groups.keys()):
                                    items = tag_groups[tag_id]
                                    if len(items) < int(start_frame_scan_min_samples):
                                        continue
                                    local_uv = np.array(items[0]["local_uv"], dtype=np.float32)
                                    avg_map = np.mean(
                                        np.array([it["map_xz"] for it in items], dtype=np.float32),
                                        axis=0,
                                    )
                                    averaged_detections.append(
                                        {
                                            "id": int(tag_id),
                                            "local_uv": local_uv,
                                            "map_xz": np.array(avg_map, dtype=np.float32),
                                        }
                                    )
                                start_frame_scan_active = False
                                start_frame_scan_started_at = 0.0
                                start_frame_scan_samples = []
                                if len(averaged_detections) >= 3:
                                    if apply_start_frame_from_detection_set(
                                        averaged_detections[:3],
                                        status_prefix="Start frame scan locked",
                                    ):
                                        start_frame_locked_once = True
                                else:
                                    start_frame_last_status = "Start frame: scan needs 3 stable tags."
                                    start_frame_last_error_m = None
                                    print(start_frame_last_status)
                                publish_map_ui_state(force=True)

                should_try_start_frame_lock = bool(start_frame_lock_requested)
                if (
                    (not should_try_start_frame_lock)
                    and start_frame_auto_lock_enabled
                    and (not start_frame_locked_once)
                    and (not start_frame_scan_active)
                    and (not mining_running_now)
                    and (now - float(start_frame_last_attempt_time)) >= float(start_frame_auto_retry_sec)
                ):
                    should_try_start_frame_lock = True
                if should_try_start_frame_lock:
                    start_frame_lock_requested = False
                    start_frame_last_attempt_time = float(now)
                    if tracking_enabled and (not tracking_pose_ok):
                        if not start_frame_locked_once:
                            start_frame_last_status = "Start frame: tracking is not locked."
                            start_frame_last_error_m = None
                            print(start_frame_last_status)
                    else:
                        try:
                            _img_raw = image_left.get_data()
                            if _img_raw is not None:
                                if _img_raw.ndim == 3 and _img_raw.shape[2] == 4:
                                    _img_bgr = cv2.cvtColor(_img_raw, cv2.COLOR_BGRA2BGR)
                                elif _img_raw.ndim == 3 and _img_raw.shape[2] >= 3:
                                    _img_bgr = _img_raw[:, :, :3].copy()
                                else:
                                    _img_bgr = None
                            else:
                                _img_bgr = None
                            if _img_bgr is None:
                                if not start_frame_locked_once:
                                    start_frame_last_status = "Start frame: no usable camera frame."
                                    start_frame_last_error_m = None
                                    print(start_frame_last_status)
                            else:
                                if apply_start_frame_from_tags(_img_bgr, cloud, R_world_cam, t_map):
                                    start_frame_locked_once = True
                        except Exception as exc:
                            start_frame_last_status = f"Start frame: lock failed ({exc})."
                            start_frame_last_error_m = None
                            print(start_frame_last_status)
                    publish_map_ui_state(force=True)
                if xyz.size > 0:
                    if map_integration_ok and (not no_mapping_mode):
                        # Transform to world frame if tracking is enabled.
                        xyz_world = (R_world_cam @ xyz.T).T + t_map
                        x = -xyz_world[:, 0]
                        z = xyz_world[:, 2]
                        occ_map.update(x, z, ground_mask, obstacle_mask, hole_mask)

                    # Object detection: persist configured landmarks, stamp configured obstacles,
                    # and use saved landmarks to correct the held map pose when tracking is lost.
                    if ((not driver_priority_active)
                            and rock_detect_enabled
                            and (rock_model is not None)
                            and (frame_idx - rock_last_frame) >= max(1, args.rock_every)):
                        rock_last_frame = frame_idx
                        try:
                            _img_raw = image_left.get_data()
                            if _img_raw is not None:
                                if _img_raw.ndim == 3 and _img_raw.shape[2] == 4:
                                    _img_bgr = cv2.cvtColor(_img_raw, cv2.COLOR_BGRA2BGR)
                                elif _img_raw.ndim == 3 and _img_raw.shape[2] == 3:
                                    _img_bgr = _img_raw
                                else:
                                    _img_bgr = None
                                if _img_bgr is not None:
                                    _results = rock_model.predict(
                                        source=_img_bgr,
                                        conf=args.rock_conf,
                                        verbose=False,
                                    )[0]
                                    _names = _results.names if hasattr(_results, "names") else {}
                                    _boxes = list(_results.boxes or [])
                                    if args.rock_debug:
                                        print(
                                            "[RockDebug] "
                                            f"frame={frame_idx} boxes={len(_boxes)} "
                                            f"map_ok={map_integration_ok} "
                                            f"tracking_ok={tracking_pose_ok} "
                                            f"driver_priority={driver_priority_active}"
                                        )
                                    _ih, _iw = _img_bgr.shape[:2]
                                    _cld_h, _cld_w = cloud.shape[:2]
                                    _det_R_world_cam = np.array(R_world_cam, dtype=np.float32).reshape(3, 3)
                                    _det_t_map = np.array(t_map, dtype=np.float32).reshape(3,)
                                    _landmark_pose_changed = False
                                    for _det in _boxes:
                                        _lbl = str(_names.get(int(_det.cls[0]), "")).strip().lower()
                                        if not _lbl:
                                            if args.rock_debug:
                                                print("[RockDebug] skip unlabeled detection")
                                            continue
                                        _conf = float(_det.conf[0]) if hasattr(_det, "conf") else float(args.rock_conf)
                                        _x1, _y1, _x2, _y2 = _det.xyxy[0].tolist()
                                        if _lbl == "rock":
                                            _now_overlay = time.time()
                                            rock_overlay_detections = [
                                                _item
                                                for _item in rock_overlay_detections
                                                if (_now_overlay - float(_item.get("time", 0.0))) <= 1.0
                                            ]
                                            rock_overlay_detections.append(
                                                {
                                                    "time": _now_overlay,
                                                    "label": _lbl,
                                                    "conf": _conf,
                                                    "box": (_x1, _y1, _x2, _y2),
                                                    "size": (_iw, _ih),
                                                }
                                            )
                                            rock_overlay_detections = rock_overlay_detections[-12:]
                                        if _lbl == "rock" and args.rock_snapshot_dir:
                                            _now_snapshot = time.time()
                                            if (_now_snapshot - rock_last_snapshot_time) >= max(
                                                0.0,
                                                float(args.rock_snapshot_cooldown),
                                            ):
                                                try:
                                                    os.makedirs(args.rock_snapshot_dir, exist_ok=True)
                                                    _snap = _img_bgr.copy()
                                                    _sx1 = max(0, min(_iw - 1, int(_x1)))
                                                    _sy1 = max(0, min(_ih - 1, int(_y1)))
                                                    _sx2 = max(0, min(_iw - 1, int(_x2)))
                                                    _sy2 = max(0, min(_ih - 1, int(_y2)))
                                                    cv2.rectangle(_snap, (_sx1, _sy1), (_sx2, _sy2), (0, 0, 255), 2)
                                                    cv2.putText(
                                                        _snap,
                                                        f"rock {_conf:.2f}",
                                                        (_sx1, max(18, _sy1 - 8)),
                                                        cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.7,
                                                        (0, 0, 255),
                                                        2,
                                                        cv2.LINE_AA,
                                                    )
                                                    _stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(_now_snapshot))
                                                    _snap_path = os.path.join(
                                                        args.rock_snapshot_dir,
                                                        f"rock_{_stamp}_frame{frame_idx}_conf{_conf:.2f}.jpg",
                                                    )
                                                    if cv2.imwrite(_snap_path, _snap):
                                                        rock_last_snapshot_time = _now_snapshot
                                                        print(f"[RockSnapshot] saved {_snap_path}")
                                                    elif args.rock_debug:
                                                        print(f"[RockSnapshot] failed to save {_snap_path}")
                                                except Exception as _snap_exc:
                                                    if args.rock_debug:
                                                        print(f"[RockSnapshot] failed: {_snap_exc}")
                                        _cx = int((_x1 + _x2) / 2)
                                        _cy = int((_y1 + _y2) / 2)
                                        _pc_c = int(_cx * _cld_w / max(1, _iw))
                                        _pc_r = int(_cy * _cld_h / max(1, _ih))
                                        _pc_r = max(0, min(_cld_h - 1, _pc_r))
                                        _pc_c = max(0, min(_cld_w - 1, _pc_c))
                                        _pt = cloud[_pc_r, _pc_c, :3]
                                        if not np.isfinite(_pt).all():
                                            if args.rock_debug:
                                                print(
                                                    "[RockDebug] "
                                                    f"skip label={_lbl} conf={_conf:.2f} "
                                                    f"center=({_cx},{_cy}) reason=invalid_depth"
                                                )
                                            continue

                                        _is_landmark = _lbl in landmark_class_names
                                        _is_obstacle = _lbl in rock_class_names
                                        if (
                                            _is_landmark
                                            and tracking_enabled
                                            and (not tracking_pose_ok)
                                            and try_landmark_relocalization(
                                                _lbl,
                                                _pt,
                                                _det_t_map,
                                                fallback_forward_world=heading_fallback_forward_world,
                                            )
                                        ):
                                            _landmark_pose_changed = True
                                            _det_t_map = np.array(landmark_pose_override_t_map, dtype=np.float32).reshape(3,)
                                            _det_R_world_cam = np.array(
                                                landmark_pose_override_R_world_cam,
                                                dtype=np.float32,
                                            ).reshape(3, 3)

                                        _pt_w = (_det_R_world_cam @ _pt.astype(np.float32)) + _det_t_map
                                        _rc = map_world_to_grid(_pt_w[0], _pt_w[2])
                                        if _rc is None:
                                            if args.rock_debug:
                                                print(
                                                    "[RockDebug] "
                                                    f"skip label={_lbl} conf={_conf:.2f} "
                                                    f"point=({_pt_w[0]:+.2f},{_pt_w[2]:+.2f}) reason=off_map"
                                                )
                                            continue
                                        _rr, _cc = _rc

                                        if _is_obstacle and map_integration_ok:
                                            _r0 = max(0, _rr - 1); _r1 = min(occ_map.grid_h - 1, _rr + 1)
                                            _c0 = max(0, _cc - 1); _c1 = min(occ_map.grid_w - 1, _cc + 1)
                                            occ_map.occ_counts[_r0:_r1+1, _c0:_c1+1] += float(args.rock_stamp)
                                            occ_map.free_counts[_r0:_r1+1, _c0:_c1+1] = 0.0
                                            if args.rock_debug:
                                                print(
                                                    "[RockDebug] "
                                                    f"stamp label={_lbl} conf={_conf:.2f} "
                                                    f"grid=({_rr},{_cc}) point=({_pt_w[0]:+.2f},{_pt_w[2]:+.2f})"
                                                )
                                        elif args.rock_debug:
                                            reason = "not_in_rock_classes"
                                            if _is_obstacle and not map_integration_ok:
                                                reason = "map_paused"
                                            print(
                                                "[RockDebug] "
                                                f"no_stamp label={_lbl} conf={_conf:.2f} "
                                                f"grid=({_rr},{_cc}) reason={reason}"
                                            )

                                        if _is_landmark and (
                                            ((not tracking_enabled) or tracking_pose_ok)
                                            or (landmark_pose_override_t_map is not None)
                                        ):
                                            record_static_landmark(
                                                _lbl,
                                                map_x_from_zed(_pt_w[0]),
                                                float(_pt_w[2]),
                                                _conf,
                                            )

                                    if _landmark_pose_changed:
                                        t_map = np.array(landmark_pose_override_t_map, dtype=np.float32).reshape(3,)
                                        R_world_cam = np.array(
                                            landmark_pose_override_R_world_cam,
                                            dtype=np.float32,
                                        ).reshape(3, 3)
                                        rover_pos_map, rover_forward_world, rover_right_world = rover_pose_from_camera(
                                            R_world_cam,
                                            t_map,
                                            current_mount_yaw_deg,
                                        )
                                        cam_row_col = map_world_to_grid(t_map[0], t_map[2])
                                        rover_row_col = map_world_to_grid(rover_pos_map[0], rover_pos_map[2])
                                        drive_origin_pos_map = navigation_origin_world(rover_pos_map, rover_forward_world)
                                        if drive_origin_pos_map is not None:
                                            drive_origin_row_col = map_world_to_grid(
                                                drive_origin_pos_map[0],
                                                drive_origin_pos_map[2],
                                            )
                        except Exception:
                            pass  # never crash the main loop on detection errors
                    map_vis = None
                    if not no_mapping_mode:
                        map_vis = occ_map.render(whole_mode=whole_map_enabled)
                        # Smooth mode: remove isolated red-dot noise from display.
                        if smooth_map_enabled and map_vis is not None:
                            kernel = np.ones((3, 3), np.uint8)
                            red_ch = map_vis[:, :, 2]
                            red_mask = (red_ch > 100).astype(np.uint8)
                            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
                            map_vis[:, :, 2] = np.where(red_mask, red_ch, 0)
                            map_vis[:, :, 0] = np.where(red_mask, map_vis[:, :, 0], 0)
                            map_vis[:, :, 1] = np.where(red_mask, map_vis[:, :, 1], 0)
                    if map_vis is not None and start_frame_last_map_points:
                        tag_pts = []
                        for item in start_frame_last_map_points:
                            rc = occ_map.world_to_grid(float(item["map_x"]), float(item["map_z"]))
                            if rc is None:
                                continue
                            rr, cc = int(rc[0]), int(rc[1])
                            tag_pts.append((rr, cc, int(item["id"])))
                        if len(tag_pts) >= 2:
                            poly_pts = np.array([[cc, rr] for rr, cc, _ in tag_pts], dtype=np.int32)
                            cv2.polylines(map_vis, [poly_pts], False, (255, 120, 255), 1, cv2.LINE_AA)
                        for rr, cc, tag_id in tag_pts:
                            cv2.circle(map_vis, (cc, rr), 3, (255, 120, 255), -1)
                            cv2.circle(map_vis, (cc, rr), 5, (255, 255, 255), 1)
                            cv2.putText(
                                map_vis,
                                str(tag_id),
                                (cc + 6, max(10, rr - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.32,
                                (255, 200, 255),
                                1,
                                cv2.LINE_AA,
                            )
                    if demo_auto_enabled and ensure_demo_rover_pose(actual_rover_pos_map, actual_rover_forward_world):
                        rover_pos_map = np.array(demo_rover_pos_map, dtype=np.float32).reshape(3,)
                        rover_forward_world = np.array(
                            [
                                math.cos(demo_rover_heading_rad),
                                0.0,
                                math.sin(demo_rover_heading_rad),
                            ],
                            dtype=np.float32,
                        )
                        rover_right_world = np.array(
                            [
                                -math.sin(demo_rover_heading_rad),
                                0.0,
                                math.cos(demo_rover_heading_rad),
                            ],
                            dtype=np.float32,
                        )
                        cam_row_col = map_world_to_grid(rover_pos_map[0], rover_pos_map[2])
                        rover_row_col = map_world_to_grid(rover_pos_map[0], rover_pos_map[2])
                        drive_origin_pos_map = navigation_origin_world(rover_pos_map, rover_forward_world)
                        if drive_origin_pos_map is not None:
                            drive_origin_row_col = map_world_to_grid(
                                drive_origin_pos_map[0], drive_origin_pos_map[2]
                            )
                    # Draw camera position marker (blue square).
                    # Mining tick: may override goal_cell or supply a direct drive command.
                    _mine_goal, _mine_drive, _mine_status = mining.tick(
                        rover_row_col, occ_map, time.time()
                    )
                    if _mine_goal is not None and _mine_goal != goal_cell:
                        goal_cell = _mine_goal
                        path_cells = None
                        path_plan_mode = "none"
                        last_path_plan_time = 0.0
                        mining_goal_active = True
                    elif mining_goal_active and mining.state in (
                        auto_mining.MiningState.IDLE,
                        auto_mining.MiningState.DONE,
                        auto_mining.MiningState.ABORTED,
                    ):
                        clear_navigation_goal()
                    if rover_row_col is not None:
                        display_forward = display_forward_world(
                            R_world_cam,
                            rover_forward_world,
                            tracking_ok=tracking_pose_ok,
                            imu_forward_fallback=heading_fallback_forward_world,
                        )
                        rover_heading_vec_rc = heading_vec_from_world(rover_pos_map, display_forward)
                    if (not map_integration_ok) and HAS_CV2:
                        pause_text = "TRACKING LOST - MAP PAUSED"
                        pause_color = (0, 140, 255)
                        if camera_map_pause_reason:
                            pause_text = f"{camera_map_pause_reason} - MAP PAUSED"
                            pause_color = (0, 200, 255)
                        cv2.putText(
                            map_vis,
                            pause_text,
                            (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            pause_color,
                            1,
                            cv2.LINE_AA,
                        )

                    # Compute/update path to goal (avoid red obstacles only).
                    if goal_cell is not None and drive_origin_row_col is not None:
                        now = time.time()
                        should_replan = (
                            drive_origin_row_col != last_start
                            or goal_cell != last_goal
                            or path_cells is None
                            or (now - last_path_plan_time) >= args.path_replan_sec
                        )
                        if should_replan:
                            base_obs = occ_map.obstacle_mask(
                                min_occ_count=args.path_avoid_occ_min,
                                min_occ_ratio=args.path_avoid_occ_ratio,
                                min_occ_advantage=args.path_avoid_occ_advantage,
                            )
                            # Smooth mode: remove isolated noise pixels (morphological opening)
                            if smooth_map_enabled and np.any(base_obs):
                                kernel = np.ones((3, 3), np.uint8)
                                base_obs = cv2.morphologyEx(
                                    base_obs.astype(np.uint8), cv2.MORPH_OPEN, kernel
                                ).astype(bool)
                            radius_cells = int(np.ceil((args.rover_size_m / 2.0) / occ_map.map_res_m))

                            def _try_plan(obs_src, inflate_radius, soft_clearance, search_sec):
                                obs_try = obs_src.copy()
                                if inflate_radius > 0 and np.any(obs_try):
                                    obs_try = map_utils.inflate_mask(obs_try, inflate_radius)

                                path_cost = np.zeros(obs_try.shape, dtype=np.float32)
                                if args.block_unknown:
                                    known = occ_map.known_mask(min_evidence=args.unknown_min_evidence)
                                    unknown = np.logical_not(known)
                                    if np.any(unknown):
                                        path_cost[unknown] += 0.75

                                if np.any(obs_try) and soft_clearance > 0:
                                    for ring in range(soft_clearance, 0, -1):
                                        near = map_utils.inflate_mask(obs_try, ring)
                                        cost = 0.12 * float(soft_clearance - ring + 1)
                                        path_cost[near] = np.maximum(path_cost[near], cost)

                                path_cost += np.minimum(3.0, occ_map.occ_counts).astype(np.float32) * 0.05

                                clear_cells = int(np.ceil(max(0.0, args.start_clear_radius_m) / occ_map.map_res_m))
                                if clear_cells > 0:
                                    obs_try = map_utils.clear_mask_circle(obs_try, drive_origin_row_col, clear_cells)
                                    keep_cost = map_utils.clear_mask_circle(
                                        np.ones(obs_try.shape, dtype=bool), drive_origin_row_col, clear_cells
                                    )
                                    path_cost[~keep_cost] = 0.0

                                return map_utils.astar_path(
                                    drive_origin_row_col,
                                    goal_cell,
                                    obs_try,
                                    connectivity=args.path_connectivity,
                                    traversal_cost_map=path_cost,
                                    max_search_sec=search_sec,
                                )

                            attempts = [
                                (
                                    "normal",
                                    base_obs,
                                    max(0, radius_cells),
                                    max(1, int(args.path_soft_clearance_cells)),
                                    float(args.path_max_search_sec),
                                ),
                            ]
                            if args.path_relax_on_fail:
                                relaxed_radius = max(0, radius_cells // 2)
                                relaxed_clearance = max(1, int(args.path_soft_clearance_cells) // 2)
                                attempts.append((
                                    "relaxed",
                                    base_obs,
                                    relaxed_radius,
                                    relaxed_clearance,
                                    float(args.path_max_search_sec) * 1.5,
                                ))
                                noise_obs = base_obs
                                if np.any(noise_obs):
                                    kernel = np.ones((3, 3), np.uint8)
                                    noise_obs = cv2.morphologyEx(
                                        noise_obs.astype(np.uint8), cv2.MORPH_OPEN, kernel
                                    ).astype(bool)
                                attempts.append((
                                    "orange-noise-point",
                                    noise_obs,
                                    0,
                                    1,
                                    float(args.path_max_search_sec) * 2.0,
                                ))

                            path_cells = None
                            path_plan_mode = "none"
                            for mode_name, obs_try, inflate_radius, soft_clearance, search_sec in attempts:
                                candidate = _try_plan(obs_try, inflate_radius, soft_clearance, search_sec)
                                if candidate:
                                    path_cells = candidate
                                    path_plan_mode = mode_name
                                    if mode_name != "normal":
                                        print(f"Path fallback: using {mode_name} path.")
                                    break
                            if path_cells:
                                last_path_cells = path_cells
                                stuck_escape_counter = 0
                            else:
                                # Do not keep stale path to an old goal.
                                last_path_cells = None
                                print("No path to selected goal yet; retrying...")
                                if 'stuck_escape_counter' not in locals():
                                    stuck_escape_counter = 0
                                stuck_escape_counter += 1
                                if stuck_escape_counter >= 3:
                                    print("Auto escape: backing up and turning to escape red spot.")
                                    backup_hold_until = max(backup_hold_until, time.time() + 0.5)
                                    stuck_escape_counter = 0
                            last_start = drive_origin_row_col
                            last_goal = goal_cell
                            last_path_plan_time = now

                    # Draw path if available.
                    draw_path = path_cells if path_cells else last_path_cells
                    if draw_path:
                        pts = np.array([[c, r] for r, c in draw_path], dtype=np.int32)
                        if pts.shape[0] >= 2:
                            if path_plan_mode == "normal":
                                path_color = (255, 255, 0)
                            elif path_plan_mode == "relaxed":
                                path_color = (0, 190, 255)
                            else:
                                path_color = (0, 120, 255)
                            cv2.polylines(map_vis, [pts], False, path_color, 2 if path_plan_mode != "normal" else 1)
                            if path_plan_mode != "normal":
                                cv2.putText(
                                    map_vis,
                                    f"PATH:{path_plan_mode}",
                                    (8, 36),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.42,
                                    path_color,
                                    1,
                                    cv2.LINE_AA,
                                )
                    # Draw goal marker.
                    if goal_cell is not None:
                        gr, gc = goal_cell
                        if 0 <= gr < occ_map.grid_h and 0 <= gc < occ_map.grid_w:
                            cv2.circle(map_vis, (gc, gr), 2, (0, 255, 255), -1)
                    if drive_calibration.target_cell is not None:
                        cal_r, cal_c = drive_calibration.target_cell
                        if 0 <= cal_r < occ_map.grid_h and 0 <= cal_c < occ_map.grid_w:
                            cv2.circle(map_vis, (cal_c, cal_r), 8, (255, 200, 0), 1)
                            cv2.putText(
                                map_vis,
                                "CAL",
                                (cal_c + 6, max(12, cal_r - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.40,
                                (255, 200, 0),
                                1,
                                cv2.LINE_AA,
                            )
                    if drive_calibration.active and goal_cell is not None and rover_pos_map is not None:
                        goal_world_for_cal = occ_map.grid_to_world(goal_cell[0], goal_cell[1])
                        rover_xz_for_cal = (map_x_from_zed(rover_pos_map[0]), float(rover_pos_map[2]))
                        if goal_world_for_cal is not None:
                            cal_event = drive_calibration.update(
                                rover_xz_for_cal,
                                goal_world_for_cal,
                                bool(args.drive_heading_flip),
                            )
                            if cal_event is not None:
                                if cal_event.get("message"):
                                    print(str(cal_event["message"]))
                                if "apply_drive_heading_flip" in cal_event:
                                    set_drive_heading_flip(
                                        bool(cal_event["apply_drive_heading_flip"]),
                                        "drive calibration",
                                    )
                                if cal_event.get("clear_goal"):
                                    clear_navigation_goal()

                    # Mining zone / dig-point overlay (drawn before apply_map_view so
                    # it scrolls correctly with the follow-rover map view).
                    mining.render_overlay(map_vis, occ_map)

                    if args.heatmap:
                        heatmap_vis = heatmap_utils.render_heatmap(
                            occ_map,
                            mode=args.heatmap_mode,
                            min_evidence=args.heatmap_min_evidence,
                        )
                        if args.heatmap_window:
                            # Keep normal map colors untouched when showing separate heatmap window.
                            pass
                        else:
                            map_vis = heatmap_utils.blend_with_map(
                                map_vis,
                                heatmap_vis,
                                alpha=args.heatmap_alpha,
                            )
                            cv2.putText(
                                map_vis,
                                f"HEAT:{args.heatmap_mode}",
                                (8, 16),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )

                    # Optional display-only map recentering around rover.
                    map_vis, map_view_shift_r, map_view_shift_c = apply_map_view(map_vis, rover_row_col)
                    draw_rover_overlay(map_vis, rover_row_col, cam_row_col, rover_heading_vec_rc)
                    mining.render_status_banner(map_vis)
                    draw_localization_banner(map_vis)
                    if map_red_only_view:
                        cv2.putText(
                            map_vis,
                            "RED-ONLY VIEW (press 'v' to toggle off)",
                            (8, max(24, map_vis.shape[0] - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 120, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    if args.heatmap and args.heatmap_window and heatmap_vis is not None:
                        heatmap_vis, _, _ = apply_map_view(heatmap_vis, rover_row_col)

                    # Drive output to RoboRIO (optional).
                    if sd is not None:
                        now = time.time()
                        # Clear ready pulse to create edges like working test scripts.
                        if nt_ready_high and now >= nt_ready_clear_time:
                            sd.putBoolean("Jetson/CommandReady", False)
                            nt_ready_high = False
                        if args.nt_health_debug and (now - nt_last_health_log) >= max(0.2, args.nt_health_period_sec):
                            nt_last_health_log = now
                            nt_health_seq += 1
                            connected = NetworkTables.isConnected()
                            nt_connected_cached = bool(connected)
                            if connected:
                                last_nt_ok_time = now
                                nt_watchdog_tripped = False
                            sd.putNumber("Jetson/NTClientSeq", float(nt_health_seq))
                            sd.putNumber("Jetson/NTClientUnix", float(now))
                            sd.putString("Jetson/NTClientName", "zed_ground_wall.py")
                            drive_forward_in = sd.getNumber("Jetson/DriveForward", float("nan"))
                            drive_turn_in = sd.getNumber("Jetson/DriveTurn", float("nan"))
                            speed_in = sd.getNumber("Jetson/Speed", float("nan"))
                            turn_speed_in = sd.getNumber("Jetson/TurnSpeed", float("nan"))
                            ack_seq = sd.getNumber("Jetson/NTServerAckSeq", -1.0)
                            print(
                                f"NT health connected={connected} target={args.roborio_ip} "
                                f"peers=[{nt_connections_summary()}] tx_seq={nt_health_seq} ack_seq={ack_seq:.0f} "
                                f"rx_fwd={drive_forward_in:+.2f} rx_turn={drive_turn_in:+.2f} "
                                f"rx_speed={speed_in:+.2f} rx_turn_speed={turn_speed_in:+.2f}"
                            )
                        elif (now - nt_last_conn_log) >= 2.0:
                            # Periodic lightweight connection status when health debug is off.
                            connected = NetworkTables.isConnected()
                            nt_connected_cached = bool(connected)
                            if connected:
                                last_nt_ok_time = now
                                nt_watchdog_tripped = False
                            print(f"NT connected={connected} target={args.roborio_ip}")
                            nt_last_conn_log = now
                        if (now - last_drive_send) >= (1.0 / max(1.0, args.drive_rate_hz)):
                            last_drive_send = now
                            status_target_cell = None
                            status_target_world = None
                            dig_profile_playback_cmd = None
                            controller_macro_playback_cmd = None
                            refresh_ds_joystick_state()
                            if controller_macros.recording:
                                record_fwd = 0.0
                                record_turn = 0.0
                                telemetry_fwd = float("nan")
                                telemetry_turn = float("nan")
                                mechanism_state = current_controller_macro_mechanism_state()
                                if sd is not None:
                                    telemetry_fwd = float(sd.getNumber("Jetson/DriveForward", float("nan")))
                                    telemetry_turn = float(sd.getNumber("Jetson/DriveTurn", float("nan")))
                                if math.isfinite(telemetry_fwd) and math.isfinite(telemetry_turn):
                                    # Match the known-good mini-rover recorder: capture what the
                                    # rover is actually executing, not just the controller intent.
                                    record_fwd = telemetry_fwd
                                    record_turn = telemetry_turn
                                elif manual_mode:
                                    record_fwd, record_turn = mix_ds_drive(manual_fwd, manual_turn)
                                elif args.ds_joystick:
                                    # Fallback if live drive telemetry is unavailable.
                                    record_fwd = float(ds_joystick_fwd)
                                    record_turn = float(ds_joystick_turn)
                                if abs(record_fwd) < 0.05:
                                    record_fwd = 0.0
                                if abs(record_turn) < 0.05:
                                    record_turn = 0.0
                                controller_macros.capture_sample(
                                    now,
                                    record_fwd,
                                    record_turn,
                                    mechanism_state["digger_on"],
                                    mechanism_state["lower_on"],
                                    mechanism_state["left_extend_on"],
                                    mechanism_state["right_extend_on"],
                                    mechanism_state["door_open_on"],
                                    mechanism_state["door_close_on"],
                                )
                            # Watchdog: NT telemetry lost — stop immediately.
                            _nt_timeout = float(args.nt_timeout_sec)
                            if _nt_timeout > 0 and (now - last_nt_ok_time) > _nt_timeout:
                                if not nt_watchdog_tripped:
                                    nt_watchdog_tripped = True
                                    print(f"[WATCHDOG] NT telemetry lost for >{_nt_timeout:.1f}s — stopping rover!")
                                send_nt_command(False, 0.0, 0.0, 0.1)
                                reset_auto_drive_shape(now)
                                continue
                            if driver_priority_active and not controller_macro_preview_active:
                                # Let the RoboRIO/Xbox path own the drivetrain while the driver is actively commanding it.
                                send_nt_command(False, 0.0, 0.0, 0.1)
                                reset_auto_drive_shape(now)
                                continue
                            # Human STOP: stamp person cells as temporary obstacles so A* reroutes.
                            if human_hazard_state == "STOP" and human_person_map_points:
                                for _hr, _hc, _hisp in human_person_map_points:
                                    if _hisp and 0 <= _hr < occ_map.grid_h and 0 <= _hc < occ_map.grid_w:
                                        r0 = max(0, _hr - 1); r1 = min(occ_map.grid_h - 1, _hr + 1)
                                        c0 = max(0, _hc - 1); c1 = min(occ_map.grid_w - 1, _hc + 1)
                                        occ_map.occ_counts[r0:r1+1, c0:c1+1] += 4.0
                                        occ_map.free_counts[r0:r1+1, c0:c1+1] = 0.0
                                path_cells = None          # force replan around person
                                last_path_plan_time = 0.0
                            if close_obstacle_detected:
                                backup_hold_until = max(
                                    backup_hold_until,
                                    now + max(0.05, float(args.backup_hold_sec)),
                                )
                            if emergency_stop:
                                reset_auto_drive_shape(now)
                                send_nt_command(False, 0.0, 0.0, 0.1)
                            elif human_hazard_state == "STOP":
                                # Person too close — hold still while A* replans around them.
                                reset_auto_drive_shape(now)
                                send_nt_command(False, 0.0, 0.0, 0.1)
                            elif now < backup_hold_until:
                                reset_auto_drive_shape(now)
                                escape_fwd = close_obstacle_escape_sign * max(0.0, min(1.0, float(args.backup_speed)))
                                escape_label = "driving forward" if escape_fwd > 0.0 else "backing up"
                                if (now - last_backup_log_time) >= 0.5:
                                    if close_obstacle_min_z is not None:
                                        print(f"Close obstacle {close_obstacle_min_z:.2f}m in camera lane: {escape_label}.")
                                    else:
                                        print(f"Close obstacle in camera lane: {escape_label}.")
                                    last_backup_log_time = now
                                send_nt_command(
                                    True,
                                    escape_fwd,
                                    0.0,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                            elif excavation_pattern_test_active:
                                pattern_state = excavation_pattern_state(now)
                                if pattern_state is None or pattern_state.get("done"):
                                    excavation_pattern_test_active = False
                                    excavation_pattern_test_started_at = 0.0
                                    reset_auto_drive_shape(now)
                                    send_nt_command(False, 0.0, 0.0, 0.1)
                                    print("Excavation pattern test completed.")
                                    publish_map_ui_state(force=True)
                                    continue
                                reset_auto_drive_shape(now)
                                send_nt_command(
                                    True,
                                    float(pattern_state.get("fwd", 0.0)),
                                    0.0,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                            elif test_drive_forward_active:
                                if now >= test_drive_forward_until:
                                    test_drive_forward_active = False
                                    test_drive_forward_until = 0.0
                                    reset_auto_drive_shape(now)
                                    send_nt_command(False, 0.0, 0.0, 0.1)
                                    print("Forward drive test completed.")
                                    publish_map_ui_state(force=True)
                                    continue
                                reset_auto_drive_shape(now)
                                send_nt_command(
                                    True,
                                    max(0.0, min(1.0, float(args.drive_speed))) * 0.45,
                                    0.0,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                            elif controller_cycle_preview_active:
                                cycle_macro = resolve_preview_controller_macro()
                                elapsed_preview = now - float(controller_cycle_phase_started_at)
                                controller_macro_playback_cmd = controller_macros.playback_sample_for_macro(
                                    cycle_macro,
                                    elapsed_preview,
                                    mode=("return" if controller_cycle_phase == "return" else "forward"),
                                    mechanism_hold_sec=(
                                        float(controller_cycle_mechanism_hold_sec)
                                        if controller_cycle_phase == "forward" else 0.0
                                    ),
                                    suppress_mechanisms=bool(controller_cycle_phase == "return"),
                                )
                                macro_duration = 0.0
                                if cycle_macro is not None:
                                    macro_duration = float(cycle_macro.get("duration_sec", 0.0))
                                if controller_macro_playback_cmd is None or elapsed_preview > max(0.05, macro_duration):
                                    if controller_cycle_phase == "forward":
                                        controller_cycle_phase = "return"
                                        controller_cycle_phase_started_at = now
                                        print("Controller cycle: returning to recorded start point.")
                                        publish_map_ui_state(force=True)
                                    else:
                                        controller_cycle_phase = "forward"
                                        controller_cycle_phase_started_at = now
                                        print("Controller cycle: restarting recorded pass.")
                                        publish_map_ui_state(force=True)
                                    controller_macro_playback_cmd = None
                                    continue
                                reset_auto_drive_shape(now)
                                preview_fwd, preview_turn = mix_ds_drive(
                                    controller_macro_playback_cmd["fwd"],
                                    controller_macro_playback_cmd["turn"],
                                )
                                send_nt_command(
                                    True,
                                    preview_fwd,
                                    preview_turn,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                            elif controller_macro_preview_active:
                                elapsed_preview = now - float(controller_macro_preview_started_at)
                                preview_macro = resolve_preview_controller_macro()
                                controller_macro_playback_cmd = controller_macros.playback_sample_for_macro(preview_macro, elapsed_preview)
                                macro_duration = 0.0
                                if preview_macro is not None:
                                    macro_duration = float(preview_macro.get("duration_sec", 0.0))
                                if controller_macro_playback_cmd is None or elapsed_preview > max(0.05, macro_duration):
                                    stop_controller_macro_preview("auto", completed=True)
                                    continue
                                reset_auto_drive_shape(now)
                                preview_fwd, preview_turn = mix_ds_drive(
                                    controller_macro_playback_cmd["fwd"],
                                    controller_macro_playback_cmd["turn"],
                                )
                                send_nt_command(
                                    True,
                                    preview_fwd,
                                    preview_turn,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                            elif manual_mode:
                                reset_auto_drive_shape(now)
                                # Driver Station joystick blends with ZED keyboard manual commands.
                                _man_fwd, _man_turn = mix_ds_drive(manual_fwd, manual_turn)
                                send_nt_command(
                                    True,
                                    _man_fwd,
                                    _man_turn,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                                if dig_profiles.recording:
                                    dig_profiles.capture_sample(
                                        now,
                                        _man_fwd,
                                        _man_turn,
                                        test_excavation_dig_active,
                                        test_excavation_lower_active,
                                        test_excavation_left_extend_active,
                                        test_excavation_right_extend_active,
                                    )
                            elif tracking_enabled and (not tracking_pose_ok):
                                reset_auto_drive_shape(now)
                                # Keep robot safe while localization is uncertain.
                                send_nt_command(False, 0.0, 0.0, 0.1)
                            elif drive_origin_row_col is None or drive_origin_pos_map is None:
                                reset_auto_drive_shape(now)
                                send_nt_command(False, 0.0, 0.0, 0.1)
                            elif dig_profile_preview_active:
                                elapsed_preview = now - float(dig_profile_preview_started_at)
                                dig_profile_playback_cmd = dig_profiles.playback_sample(
                                    elapsed_preview,
                                    style=dig_profile_preview_style,
                                    phase=dig_profile_preview_phase,
                                )
                                profile_duration = 0.0
                                if dig_profile_playback_cmd is not None:
                                    profile_duration = float(dig_profile_playback_cmd.get("duration_sec", 0.0))
                                if dig_profile_playback_cmd is None or elapsed_preview > max(0.05, profile_duration):
                                    stop_dig_profile_preview("auto", completed=True)
                                    continue
                                reset_auto_drive_shape(now)
                                preview_fwd, preview_turn = mix_ds_drive(
                                    dig_profile_playback_cmd["fwd"],
                                    dig_profile_playback_cmd["turn"],
                                )
                                send_nt_command(
                                    True,
                                    preview_fwd,
                                    preview_turn,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                            elif _mine_drive is not None:
                                reset_auto_drive_shape(now)
                                # Mining automation has direct drive control
                                # (DIGGING creep, BACKUP reverse, DEPOSITING reverse).
                                dig_profile_playback_cmd = None
                                if dig_profile_playback_cmd is not None:
                                    mine_fwd, mine_turn = mix_ds_drive(
                                        dig_profile_playback_cmd["fwd"],
                                        dig_profile_playback_cmd["turn"],
                                    )
                                else:
                                    mine_fwd, mine_turn = mix_ds_drive(_mine_drive[0], _mine_drive[1])
                                send_nt_command(
                                    True,
                                    mine_fwd,
                                    mine_turn,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                            else:
                                reverse_path_drive = (
                                    bidirectional_auto_enabled
                                    and mining.state == auto_mining.MiningState.NAVIGATE_DEPOSIT
                                )
                                target_rc = pick_drive_target(draw_path, drive_origin_row_col, goal_cell)
                                if target_rc is not None:
                                    target_world = occ_map.grid_to_world(target_rc[0], target_rc[1])
                                    if target_world is None:
                                        send_nt_command(False, 0.0, 0.0, 0.1)
                                        continue
                                    tx, tz = target_world
                                    status_target_cell = target_rc
                                    status_target_world = (float(tx), float(tz))
                                elif goal_cell is not None and args.allow_direct_no_path:
                                    # Fallback: drive directly toward clicked goal if path is not ready yet.
                                    goal_world_fallback = occ_map.grid_to_world(goal_cell[0], goal_cell[1])
                                    if goal_world_fallback is None:
                                        send_nt_command(False, 0.0, 0.0, 0.1)
                                        continue
                                    tx, tz = goal_world_fallback
                                    status_target_cell = goal_cell
                                    status_target_world = (float(tx), float(tz))
                                elif goal_cell is not None:
                                    reset_auto_drive_shape(now)
                                    send_nt_command(False, 0.0, 0.0, 0.1)
                                    continue
                                else:
                                    reset_auto_drive_shape(now)
                                    send_nt_command(False, 0.0, 0.0, 0.1)
                                    continue
                                # Auto-drive uses the robot/ZED physical X axis.
                                # The map view is mirrored for display, so convert map targets back.
                                cx, cz = float(drive_origin_pos_map[0]), float(drive_origin_pos_map[2])
                                tx_drive = zed_x_from_map(tx)
                                dx = tx_drive - cx
                                dz = tz - cz

                                # Let mining own the final arrival handoff for excavation/deposit
                                # targets so auto-drive does not stop early and strand the state
                                # machine in NAV_DIG/NAV_DEPOSIT.
                                mining_arrival_owned = mining.state in (
                                    auto_mining.MiningState.NAVIGATE_DIG,
                                    auto_mining.MiningState.NAVIGATE_DEPOSIT,
                                )
                                if not mining_arrival_owned:
                                    goal_world = occ_map.grid_to_world(goal_cell[0], goal_cell[1])
                                    if goal_world is not None:
                                        gx, gz = goal_world
                                        gx_drive = zed_x_from_map(gx)
                                        if math.hypot(gx_drive - cx, gz - cz) <= args.drive_goal_tol_m:
                                            reset_auto_drive_shape(now)
                                            send_nt_command(False, 0.0, 0.0, 0.1)
                                            continue

                                # Heading error in the physical X/Z frame, not the mirrored display frame.
                                forward = drive_forward_world_from_rover(rover_forward_world)
                                heading = math.atan2(float(forward[2]), float(forward[0]))
                                target = math.atan2(dz, dx)  # dz = target_z - curr_z, dx = target_x - curr_x
                                if bidirectional_auto_enabled and mining.state != auto_mining.MiningState.NAVIGATE_DIG:
                                    forward_err = target - heading
                                    while forward_err > math.pi:
                                        forward_err -= 2 * math.pi
                                    while forward_err < -math.pi:
                                        forward_err += 2 * math.pi
                                    reverse_target = target + math.pi
                                    reverse_err = reverse_target - heading
                                    while reverse_err > math.pi:
                                        reverse_err -= 2 * math.pi
                                    while reverse_err < -math.pi:
                                        reverse_err += 2 * math.pi
                                    if abs(reverse_err) + math.radians(12.0) < abs(forward_err):
                                        reverse_path_drive = True
                                if reverse_path_drive:
                                    # For deposition, point the rover nose away from the waypoint
                                    # so negative forward backs the rear toward the deposit area.
                                    target += math.pi
                                err = target - heading
                                # Wrap to [-pi, pi].
                                while err > math.pi:
                                    err -= 2 * math.pi
                                while err < -math.pi:
                                    err += 2 * math.pi

                                tol = math.radians(max(0.0, args.drive_heading_tol_deg))
                                err_abs = abs(err)
                                max_turn_cmd = max(0.0, min(1.0, float(args.drive_max_turn_cmd)))

                                if err_abs <= tol:
                                    turn_target = 0.0
                                else:
                                    turn_target = max(-max_turn_cmd, min(max_turn_cmd, args.drive_turn_k * err))

                                # Reduce forward speed as heading error grows to avoid cutting sharp arcs.
                                slow_turn_rad = math.radians(max(0.0, float(args.drive_slow_turn_deg)))
                                stop_turn_rad = math.radians(max(0.0, float(args.drive_stop_turn_deg)))
                                if stop_turn_rad < slow_turn_rad:
                                    stop_turn_rad = slow_turn_rad

                                min_turn_forward_scale = max(
                                    0.0, min(1.0, float(args.drive_min_turn_forward_scale))
                                )
                                if stop_turn_rad <= 1e-6:
                                    turn_scale = min_turn_forward_scale if err_abs > 0.0 else 1.0
                                elif err_abs >= stop_turn_rad:
                                    turn_scale = min_turn_forward_scale
                                elif err_abs <= slow_turn_rad:
                                    turn_scale = 1.0
                                else:
                                    turn_scale = (stop_turn_rad - err_abs) / max(1e-6, (stop_turn_rad - slow_turn_rad))
                                    turn_scale = max(min_turn_forward_scale, turn_scale)

                                align_scale = max(0.0, math.cos(err))
                                fwd_mag = max(0.0, min(1.0, args.drive_speed)) * align_scale * max(0.0, min(1.0, turn_scale))
                                fwd_target = -fwd_mag if reverse_path_drive else fwd_mag

                                # Driver Station joystick can nudge auto commands.
                                fwd_target, turn_target = mix_ds_drive(fwd_target, turn_target)
                                fwd, turn = apply_auto_drive_shape(fwd_target, turn_target, now)

                                send_nt_command(
                                    True,
                                    fwd,
                                    turn,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                    # Periodically save persistent map to disk.
                    if args.map_save_every > 0 and (time.time() - last_save) >= args.map_save_every:
                        occ_map.save(args.map_save_path)
                        last_save = time.time()
                    save_recovery_checkpoint(navx_yaw_deg=navx_yaw_deg)
                    save_landmark_memory()
                    # Periodically save ZED area memory for startup relocalization.
                    if (
                        tracking_enabled
                        and args.area_save_path
                        and args.area_save_every > 0
                        and tracking_pose_ok
                        and (time.time() - last_area_save) >= args.area_save_every
                    ):
                        if zed_utils.save_area_memory(zed, sl, args.area_save_path):
                            last_area_save = time.time()
                    # Periodically update and save spatial map (mesh) if enabled.
                    if spatial_enabled and args.spatial_save_path and args.spatial_save_every > 0:
                        if (time.time() - last_spatial_save) >= args.spatial_save_every:
                            ok = zed_utils.update_spatial_map(
                                zed, sl, spatial_mesh, args.spatial_save_path, mesh_filter=args.spatial_filter
                            )
                            if ok:
                                last_spatial_save = time.time()
                                if mesh_viewer is not None:
                                    mesh_viewer.update_from_path(args.spatial_save_path)
                    if mesh_viewer is not None:
                        mesh_viewer.poll()
                else:
                    map_vis = None
                    if not no_mapping_mode:
                        map_vis = occ_map.render(whole_mode=whole_map_enabled)
                    if args.heatmap and map_vis is not None:
                        heatmap_vis = heatmap_utils.render_heatmap(
                            occ_map,
                            mode=args.heatmap_mode,
                            min_evidence=args.heatmap_min_evidence,
                        )
                        if not args.heatmap_window:
                            map_vis = heatmap_utils.blend_with_map(
                                map_vis,
                                heatmap_vis,
                                alpha=args.heatmap_alpha,
                            )
                    map_vis, map_view_shift_r, map_view_shift_c = apply_map_view(map_vis, rover_row_col)
                    draw_rover_overlay(map_vis, rover_row_col, cam_row_col, rover_heading_vec_rc)
                    mining.render_status_banner(map_vis)
                    draw_localization_banner(map_vis)
                    if map_red_only_view:
                        cv2.putText(
                            map_vis,
                            "RED-ONLY VIEW (press 'v' to toggle off)",
                            (8, max(24, map_vis.shape[0] - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 120, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    if args.heatmap and args.heatmap_window and heatmap_vis is not None:
                        heatmap_vis, _, _ = apply_map_view(heatmap_vis, rover_row_col)

            # Live visualization (optional)
            if HAS_CV2:
                img = image_left.get_data()
                if img is not None:
                    # Normalize to BGR (3 channels) across SDK image formats.
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif img.ndim == 3:
                        if img.shape[2] == 4:
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        elif img.shape[2] == 1:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        elif img.shape[2] > 3:
                            img = img[:, :, :3]
                    else:
                        img = None
                if img is not None:
                    # Build a ground/obstacle mask at the same stride
                    xyz_full = cloud[::stride, ::stride, :3]
                    valid = np.isfinite(xyz_full).all(axis=2)
                    dist_num = (a * xyz_full[:, :, 0] + b * xyz_full[:, :, 1] + c * xyz_full[:, :, 2] + d)
                    denom = np.sqrt(a * a + b * b + c * c)
                    dist_full = np.full_like(dist_num, np.nan, dtype=np.float32)
                    if np.any(valid):
                        dist_full[valid] = (dist_num[valid] / denom).astype(np.float32)

                    vis_thresh = float(args.obstacle_thresh_m)
                    obstacle = (dist_full > vis_thresh) & valid
                    if args.max_above_ground_m > 0.0:
                        max_above = float(args.max_above_ground_m)
                        obstacle = obstacle & (dist_full <= max_above)
                    else:
                        max_above = None
                    ground = (
                        valid
                        & (dist_full >= -float(args.hole_thresh_m))
                        & (dist_full <= vis_thresh)
                    )
                    if max_above is not None:
                        ground = ground & (dist_full <= max_above)

                    # Resize masks to full resolution
                    h, w, _ = img.shape
                    gh, gw = ground.shape
                    ground_full = cv2.resize(ground.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    obstacle_full = cv2.resize(obstacle.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

                    if camera_overlay_enabled:
                        overlay = img.copy()
                        # Green for ground (skip if red-only mode)
                        if not args.overlay_red_only:
                            overlay[ground_full == 1] = (0, 200, 0)
                        # Red for obstacles/walls
                        overlay[obstacle_full == 1] = (0, 0, 255)
                        # Blend
                        vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
                    else:
                        vis = img.copy()

                    # Custom YOLO rock overlay. Detection runs every N frames, so keep
                    # recent boxes on screen briefly between inference passes.
                    now_overlay = time.time()
                    rock_overlay_detections = [
                        item
                        for item in rock_overlay_detections
                        if (now_overlay - float(item.get("time", 0.0))) <= 1.0
                    ]
                    for item in rock_overlay_detections:
                        box = item.get("box")
                        src_size = item.get("size", (w, h))
                        if not box:
                            continue
                        src_w, src_h = src_size
                        if src_w <= 0 or src_h <= 0:
                            continue
                        scale_x = float(w) / float(src_w)
                        scale_y = float(h) / float(src_h)
                        x1 = max(0, min(w - 1, int(float(box[0]) * scale_x)))
                        y1 = max(0, min(h - 1, int(float(box[1]) * scale_y)))
                        x2 = max(0, min(w - 1, int(float(box[2]) * scale_x)))
                        y2 = max(0, min(h - 1, int(float(box[3]) * scale_y)))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        label = str(item.get("label", "rock"))
                        conf = float(item.get("conf", 0.0))
                        color = (0, 255, 255)
                        text = f"{label} {conf:.0%}"
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
                        label_y0 = max(0, y1 - text_size[1] - 8)
                        label_y1 = max(text_size[1] + 8, y1)
                        label_x1 = min(w - 1, x1 + text_size[0] + 8)
                        cv2.rectangle(vis, (x1, label_y0), (label_x1, label_y1), color, -1)
                        cv2.putText(
                            vis,
                            text,
                            (x1 + 4, label_y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )

                    start_frame_marker_detections = detect_start_frame_markers_2d(img)
                    if start_frame_marker_detections:
                        marker_centers = []
                        for item in start_frame_marker_detections:
                            pts = np.asarray(item["corners"], dtype=np.int32).reshape(-1, 1, 2)
                            center_xy = (
                                int(round(float(item["center"][0]))),
                                int(round(float(item["center"][1]))),
                            )
                            marker_centers.append(center_xy)
                            cv2.polylines(vis, [pts], True, (255, 0, 255), 2, cv2.LINE_AA)
                            cv2.circle(vis, center_xy, 4, (255, 255, 255), -1)
                            cv2.putText(
                                vis,
                                f"TAG {int(item['id'])}",
                                (center_xy[0] + 8, max(18, center_xy[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.56,
                                (255, 0, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        if len(marker_centers) >= 2:
                            cv2.polylines(
                                vis,
                                [np.asarray(marker_centers, dtype=np.int32).reshape(-1, 1, 2)],
                                False,
                                (255, 160, 255),
                                1,
                                cv2.LINE_AA,
                            )

                    if start_frame_locked_once:
                        sf_text = "SF LOCKED"
                        sf_color = (60, 215, 80)
                    elif start_frame_auto_lock_enabled:
                        sf_text = "SF SEARCHING"
                        sf_color = (0, 215, 255)
                    else:
                        sf_text = "SF IDLE"
                        sf_color = (180, 180, 180)
                    if start_frame_marker_detections:
                        sf_text += " " + ",".join(str(int(item["id"])) for item in start_frame_marker_detections)
                    cv2.putText(
                        vis,
                        sf_text,
                        (10, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        sf_color,
                        2,
                        cv2.LINE_AA,
                    )

                    # Human detection overlay
                    human_person_map_points = []
                    if (not driver_priority_active) and human_detect_enabled and human_detect_available and human_objects is not None and human_od_runtime is not None:
                        if (frame_idx - human_last_frame) >= max(1, args.human_od_every):
                            human_last_frame = frame_idx
                            try:
                                od_err = zed.retrieve_objects(human_objects, human_od_runtime)
                                if od_err == sl.ERROR_CODE.SUCCESS:
                                    obj_list = getattr(human_objects, "object_list", [])
                                    nearest_m = None
                                    for obj in obj_list:
                                        bb = getattr(obj, "bounding_box_2d", None)
                                        if bb is None:
                                            continue
                                        raw_label = getattr(obj, "sublabel", "") or getattr(obj, "label", "")
                                        label = str(raw_label).strip().lower()
                                        conf = float(getattr(obj, "confidence", 0.0)) / 100.0
                                        pts = np.array(bb, dtype=np.float32).reshape(-1, 2)
                                        if pts.shape[0] == 0:
                                            continue
                                        x1 = int(np.floor(np.min(pts[:, 0])))
                                        y1 = int(np.floor(np.min(pts[:, 1])))
                                        x2 = int(np.ceil(np.max(pts[:, 0])))
                                        y2 = int(np.ceil(np.max(pts[:, 1])))
                                        if x2 <= x1 or y2 <= y1:
                                            continue
                                        is_person = label in {"person", "people", "human", "pedestrian"}
                                        # Draw box on camera view
                                        box_color = (0, 255, 120) if is_person else (0, 200, 255)
                                        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
                                        cv2.putText(
                                            vis,
                                            f"{label} {conf:.0%}",
                                            (x1, max(16, y1 - 6)),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.52,
                                            box_color,
                                            1,
                                            cv2.LINE_AA,
                                        )
                                        # Get 3D position from point cloud center of box
                                        cx_px = max(0, min(w - 1, (x1 + x2) // 2))
                                        cy_px = max(0, min(h - 1, (y1 + y2) // 2))
                                        p3 = cloud[cy_px, cx_px, :3]
                                        if np.isfinite(p3).all():
                                            pw = (R_world_cam @ p3.reshape(3, 1)).reshape(3,) + t_map
                                            rc = map_world_to_grid(pw[0], pw[2])
                                            if rc is not None:
                                                human_person_map_points.append((rc[0], rc[1], is_person))
                                            if is_person and conf >= float(args.human_min_conf):
                                                dist = float(np.linalg.norm(p3))
                                                if nearest_m is None or dist < nearest_m:
                                                    nearest_m = dist
                                    # Update hazard state
                                    if nearest_m is not None:
                                        human_nearest_m = nearest_m
                                        if nearest_m <= float(args.human_stop_m):
                                            human_hazard_state = "STOP"
                                        elif nearest_m <= float(args.human_slow_m):
                                            human_hazard_state = "SLOW"
                                        else:
                                            human_hazard_state = "CLEAR"
                                        human_clear_countdown = human_clear_hold
                                    else:
                                        if human_clear_countdown > 0:
                                            human_clear_countdown -= 1
                                        else:
                                            human_hazard_state = "CLEAR"
                                            human_nearest_m = -1.0
                            except Exception as exc:
                                if frame_idx % 60 == 1:
                                    print(f"Human detect error: {exc}")

                    # Show hazard state on camera
                    if (not driver_priority_active) and human_detect_enabled and human_detect_available and human_hazard_state != "CLEAR":
                        hz_color = (0, 220, 255) if human_hazard_state == "SLOW" else (0, 0, 255)
                        dist_txt = f"{human_nearest_m:.1f}m" if human_nearest_m > 0 else "--"
                        cv2.putText(
                            vis,
                            f"HUMAN: {human_hazard_state} ({dist_txt})",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            hz_color,
                            2,
                            cv2.LINE_AA,
                        )

                    if camera_publisher is not None and not driver_priority_active and (not low_latency_mode or args.camera_only):
                        camera_publisher.push_frame(vis)

                    if not args.no_gui:
                        cv2.imshow("ZED Ground/Obstacle Segmentation", vis)
                display_map_vis = None
                if map_vis is not None:
                    display_map_vis = map_vis.copy()
                    draw_landmarks(display_map_vis)
                    for pr, pc_col, is_p in human_person_map_points:
                        draw_live_detection_marker(display_map_vis, pr, pc_col, is_p)
                    if args.map_scale > 1:
                        display_map_vis = cv2.resize(
                            display_map_vis,
                            (occ_map.grid_w * args.map_scale, occ_map.grid_h * args.map_scale),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    if map_publisher is not None and not low_latency_mode:
                        map_publisher.push_frame(display_map_vis)
                    process_external_map_command()
                    publish_map_ui_state()

                    if not args.no_gui:
                        # Always show the map (even if the image frame is missing)
                        if map_vis is not None:
                            map_window_vis = map_vis.copy()
                        # Draw detected persons on the map
                        draw_landmarks(map_window_vis)
                        for pr, pc_col, is_p in human_person_map_points:
                            draw_live_detection_marker(map_window_vis, pr, pc_col, is_p)
                        if map_red_only_view:
                            # Red-only map mode for easier obstacle inspection.
                            map_window_vis[:, :, 0] = 0
                            map_window_vis[:, :, 1] = 0
                        if map_scale_live > 1:
                            map_window_vis = cv2.resize(
                                map_window_vis,
                                (occ_map.grid_w * map_scale_live, occ_map.grid_h * map_scale_live),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        last_map_window_shape = map_window_vis.shape[:2]
                        cv2.imshow("ZED Occupancy Map (XZ)", map_window_vis)
                        if not map_window_ready:
                            cv2.setMouseCallback("ZED Occupancy Map (XZ)", on_map_click)
                            map_window_ready = True
                    if args.heatmap and args.heatmap_window and heatmap_vis is not None:
                        heatmap_show = heatmap_vis
                        if map_scale_live > 1:
                            heatmap_show = cv2.resize(
                                heatmap_show,
                                (occ_map.grid_w * map_scale_live, occ_map.grid_h * map_scale_live),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        cv2.imshow("ZED Heatmap (XZ)", heatmap_show)
                elif no_mapping_mode:
                    process_external_map_command()
                    publish_map_ui_state()
                if not args.no_gui:
                    status_panel = render_status_panel(rover_row_col)
                    last_status_panel_shape = status_panel.shape[:2]
                    cv2.imshow("ZED Drive Status", status_panel)
                    if not status_window_ready:
                        cv2.setMouseCallback("ZED Drive Status", on_status_click)
                        status_window_ready = True
                    raw_key = cv2.waitKeyEx(1)
                    key = (raw_key & 0xFF) if raw_key >= 0 else -1
                    # Route keys to focused text input fields first.
                    if dig_name_input_focused:
                        if key == 13:  # Enter
                            dig_name_input_focused = False
                            dig_name_input_text = dig_name_input_text.strip()
                            if dig_name_input_text:
                                print(f"Dig profile name set: {dig_name_input_text}")
                        elif key == 8 or key == 127:  # Backspace/Delete
                            dig_name_input_text = dig_name_input_text[:-1]
                        elif key == 27:  # Escape
                            dig_name_input_focused = False
                        elif key >= 0:
                            ch = chr(key)
                            if ch.isalnum() or ch in " _-.":
                                dig_name_input_text += ch
                    elif map_size_input_focused:
                        if key == 13:  # Enter — apply the new rover size
                            map_size_input_focused = False
                            raw = map_size_input_text.strip().lower().replace(" ", "")
                            map_size_input_text = ""
                            try:
                                size_ft = float(raw)
                                new_size_m = round(size_ft * 0.3048, 4)
                                if new_size_m <= 0:
                                    raise ValueError("must be positive")
                                args.rover_size_m = new_size_m
                                mining.cfg["rover_size_m"] = new_size_m
                                print(f"Rover size updated to {size_ft:.2f} ft ({new_size_m:.3f} m)")
                            except Exception as _e:
                                print(f"Rover size parse error: {_e}. Enter a single number in feet, e.g. '2' or '1.5'.")
                        elif key == 8 or key == 127:  # Backspace/Delete
                            map_size_input_text = map_size_input_text[:-1]
                        elif key == 27:  # Escape — cancel
                            map_size_input_focused = False
                            map_size_input_text = ""
                        elif key >= 0:
                            ch = chr(key)
                            if ch in "0123456789.":
                                map_size_input_text += ch
                    else:
                        # Mining keys: r=run, t=abort. Zone boxes are button-only.
                        if key == ord("r"):
                            clear_navigation_goal()
                            emergency_stop = False
                            manual_mode = False
                            manual_fwd = 0.0
                            manual_turn = 0.0
                            mining.handle_key(key)
                        elif key == ord("t"):
                            mining.handle_key(key)
                            clear_navigation_goal()
                            manual_mode = False
                            manual_fwd = 0.0
                            manual_turn = 0.0
                    if key == ord("q"):
                        break
                    if key == ord("m"):
                        set_manual_drive_mode(not manual_mode, "key")
                    if key == ord("c"):
                        follow_rover_map = not follow_rover_map
                        state = "ON" if follow_rover_map else "OFF"
                        print(f"Map follow mode: {state}")
                    if key == ord("v"):
                        map_red_only_view = not map_red_only_view
                        state = "ON" if map_red_only_view else "OFF"
                        print(f"Map red-only mode: {state}")
                    if key == ord("u"):
                        set_main_rover_mode(not args.main_rover_mode)
                    if raw_key in UP_KEYS:
                        set_status_scroll(-80)
                    if raw_key in DOWN_KEYS:
                        set_status_scroll(80)
                    if raw_key in PAGEUP_KEYS or key == ord("k"):
                        set_status_scroll(-260)
                    if raw_key in PAGEDOWN_KEYS or key == ord("j"):
                        set_status_scroll(260)
                    if raw_key in HOME_KEYS:
                        set_status_scroll_to(0)
                    if raw_key in END_KEYS:
                        set_status_scroll_to(status_scroll_max)
                    if key == ord("1"):
                        set_status_scroll_to(status_section_jump_targets.get("jump_setup", 0))
                    if key == ord("2"):
                        set_status_scroll_to(status_section_jump_targets.get("jump_map_tools", 0))
                    if key == ord("3"):
                        set_status_scroll_to(status_section_jump_targets.get("jump_zones_camera", 0))
                    if key == ord("4"):
                        set_status_scroll_to(status_section_jump_targets.get("jump_calibration", 0))
                    if key == ord("5"):
                        set_status_scroll_to(status_section_jump_targets.get("jump_actuators", 0))
                    if key == ord("6"):
                        set_status_scroll_to(status_section_jump_targets.get("jump_dig_profiles", 0))
                    if key == ord("o"):
                        map_scale_live = min(12, map_scale_live + 1)
                        print(f"Map zoom: x{map_scale_live}")
                    if key == ord("p"):
                        map_scale_live = max(1, map_scale_live - 1)
                        print(f"Map zoom: x{map_scale_live}")
                    if key == ord(" "):
                        emergency_stop = True
                        manual_fwd = 0.0
                        manual_turn = 0.0
                    now = time.time()
                    if key == ord("w"):
                        manual_fwd = max(0.0, min(1.0, args.drive_speed))
                        last_w_time = now
                    if key == ord("s"):
                        manual_fwd = -max(0.0, min(1.0, args.drive_speed))
                        last_s_time = now
                    if key == ord("a"):
                        manual_turn = max(0.0, min(1.0, args.drive_speed))
                        last_a_time = now
                    if key == ord("d"):
                        manual_turn = -max(0.0, min(1.0, args.drive_speed))
                        last_d_time = now
                    if key == ord("x"):
                        manual_fwd = 0.0
                        manual_turn = 0.0
                    # Hold-to-move: decay to 0 if key not pressed recently.
                    if manual_mode:
                        if now - last_w_time > key_hold_timeout and now - last_s_time > key_hold_timeout:
                            manual_fwd = 0.0
                        if now - last_a_time > key_hold_timeout and now - last_d_time > key_hold_timeout:
                            manual_turn = 0.0
                else:
                    time.sleep(0.01)
                    publish_map_ui_state()

    if human_detect_available:
        try:
            zed.disable_object_detection()
        except Exception:
            pass
    if spatial_enabled:
        zed_utils.disable_spatial_mapping(zed)
    save_recovery_checkpoint(force=True)
    save_landmark_memory(force=True)
    if tracking_enabled and args.area_save_path:
        zed_utils.save_area_memory(zed, sl, args.area_save_path)
    if mesh_viewer is not None:
        mesh_viewer.close()
    _write_json_atomic(
        args.map_ui_state_file,
        {
            "available": False,
            "source": "zed_ground_wall",
            "timestamp_ms": int(time.time() * 1000),
            "mining_state": mining.state.value,
            "selected_tool": current_selected_tool(),
            "brush_radius": int(paint_brush_radius),
            "drive_calibration": drive_calibration.ui_state(),
            "dig_profiles": dig_profiles.ui_state(),
            "controls": [],
        },
    )
    if camera_heartbeat is not None:
        camera_heartbeat.stop()
    if camera_publisher is not None:
        camera_publisher.stop()
    if map_publisher is not None:
        map_publisher.stop()
    ros2_utils.shutdown_ros2(node)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
