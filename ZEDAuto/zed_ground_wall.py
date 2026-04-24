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
    parser.add_argument("--spatial-mapping", action="store_true", help="Enable ZED SDK spatial mapping")
    parser.add_argument("--spatial-res", default="medium", help="Spatial map resolution: low|medium|high")
    parser.add_argument("--spatial-range", default="medium", help="Spatial map range: short|medium|long")
    parser.add_argument("--spatial-save-path", default=None, help="Optional path to save spatial mesh (.obj)")
    parser.add_argument("--spatial-save-every", type=float, default=10.0, help="Seconds between spatial map saves")
    parser.add_argument("--spatial-viewer", action="store_true", help="Show live Open3D mesh viewer")
    parser.add_argument("--spatial-filter", default="none", help="Mesh filter: none|low|medium|high")
    parser.add_argument("--drive", action="store_true", help="Enable RoboRIO driving commands")
    parser.add_argument("--roborio-ip", default="10.0.9.2", help="RoboRIO IP for NetworkTables")
    parser.add_argument("--drive-speed", type=float, default=0.7, help="Forward speed command (0-1)")
    parser.add_argument("--drive-turn-k", type=float, default=0.8, help="Turn gain for heading error")
    parser.add_argument(
        "--drive-max-turn-cmd",
        type=float,
        default=0.60,
        help="Maximum absolute turn command while auto-driving (0-1)",
    )
    parser.add_argument(
        "--drive-slow-turn-deg",
        type=float,
        default=12.0,
        help="Begin reducing forward speed above this heading error (deg)",
    )
    parser.add_argument(
        "--drive-stop-turn-deg",
        type=float,
        default=22.0,
        help="Stop forward motion above this heading error (deg)",
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
    parser.add_argument("--drive-goal-tol-m", type=float, default=0.3, help="Goal tolerance (m)")
    parser.add_argument("--drive-heading-tol-deg", type=float, default=10.0, help="Heading tolerance (deg)")
    parser.add_argument("--drive-heading-flip", action="store_true", help="Flip heading by 180 degrees")
    parser.add_argument("--main-rover-mode", action="store_true", help="Enable main-rover controls on the RoboRIO")
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
    parser.add_argument("--rock-classes", default="rock,stone,boulder", help="Comma-separated class names to treat as rocks")
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
    args = parser.parse_args()

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

    # Rock detection via custom YOLO model
    rock_model = None
    rock_last_frame = -999999
    rock_class_names = set(n.strip().lower() for n in args.rock_classes.split(",") if n.strip())
    if args.rock_model:
        try:
            from ultralytics import YOLO as _YOLO
            rock_model = _YOLO(args.rock_model)
            print(f"Rock detection model loaded: {args.rock_model}  classes={rock_class_names}")
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

    def map_x_from_zed(x):
        # ZED +X is camera-right, while the occupancy map image mirrors X for display.
        return -float(x)

    def map_world_to_grid(x, z):
        return occ_map.world_to_grid(map_x_from_zed(x), float(z))

    def zed_x_from_map(x):
        return -float(x)

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
        if args.drive_heading_flip:
            rover_forward_world = -rover_forward_world
            rover_right_world = -rover_right_world
        rover_pos_world = (
            np.array(camera_pos_world, dtype=np.float32)
            - rover_forward_world * float(camera_forward_offset_m)
            - rover_right_world * float(camera_right_offset_m)
        )
        return rover_pos_world, rover_forward_world, rover_right_world

    def world_forward_from_rotation(R_world_cam):
        forward = (np.array(R_world_cam, dtype=np.float32) @ np.array([0.0, 0.0, 1.0], dtype=np.float32)).reshape(3,)
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return forward / norm

    def angle_between_vec_deg(vec_a, vec_b):
        a = np.array(vec_a, dtype=np.float32).reshape(3,)
        b = np.array(vec_b, dtype=np.float32).reshape(3,)
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an <= 1e-6 or bn <= 1e-6:
            return 0.0
        dot = float(np.clip(np.dot(a / an, b / bn), -1.0, 1.0))
        return float(np.degrees(np.arccos(dot)))

    print(
        "Camera mount: "
        f"{args.camera_mount} yaw={float(camera_mount_yaw_deg):+.1f}deg "
        f"forward_offset={float(camera_forward_offset_m):+.2f}m "
        f"right_offset={float(camera_right_offset_m):+.2f}m "
        f"servo_track={'on' if args.camera_servo_track else 'off'} "
        f"heading_flip={'on' if args.drive_heading_flip else 'off'}"
    )

    if args.map_load and os.path.exists(args.map_save_path):
        try:
            ok, msg = occ_map.load(args.map_save_path)
            print(f"{msg} ({args.map_save_path})" if ok else msg)
        except Exception as exc:
            print(f"Failed to load map ({args.map_save_path}): {exc}")

    # --- Mining automation subsystem ---
    _mining_cfg = {
        "dig_duration":          float(os.getenv("MINING_DIG_DURATION",           "5.0")),
        "dig_speed":             float(os.getenv("MINING_DIG_SPEED",              "0.20")),
        "backup_duration":       float(os.getenv("MINING_BACKUP_DURATION",        "2.0")),
        "backup_speed":          float(os.getenv("MINING_BACKUP_SPEED",           "0.35")),
        "deposit_duration":      float(os.getenv("MINING_DEPOSIT_DURATION",       "5.0")),
        "deposit_backup_speed":  float(os.getenv("MINING_DEPOSIT_BACKUP_SPEED",   "0.35")),
        "deposit_approach_dist": float(os.getenv("MINING_DEPOSIT_APPROACH_DIST",  "1.0")),
        "deposit_boundary_inset_m": float(os.getenv(
            "MINING_DEPOSIT_BOUNDARY_INSET_M", "0.05"
        )),
        "continuous_runs":      os.getenv("MINING_CONTINUOUS_RUNS", "1"),
        "strip_pitch_m":         float(os.getenv("MINING_STRIP_PITCH",            "0.0")),
        "goal_tol_m":            float(os.getenv("MINING_GOAL_TOL_M",             "0.4")),
        "rover_size_m":          float(args.rover_size_m),
        "zones_path":            os.getenv("MINING_ZONES_PATH",
                                           os.path.join(SCRIPT_DIR, "mining_zones.json")),
    }
    mining = auto_mining.MiningAutomation(_mining_cfg, occ_map)

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
            sd.putString("Jetson/MiningState", mining.state.value)
            sd.putBoolean("Jetson/ExcavatorEnabled", False)
            sd.putBoolean("Jetson/ConveyorEnabled", False)
            sd.putNumber("Jetson/ServoCommandAngleDeg", float(args.camera_map_angle_deg))
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
    reset_map_confirm = False
    status_scroll_y = 0
    status_scroll_max = 0
    disable_holes = bool(args.disable_holes)
    whole_map_enabled = False
    smooth_map_enabled = False
    last_drive_send = 0.0
    manual_fwd = 0.0
    manual_turn = 0.0
    manual_mode = bool(args.manual_start)
    if manual_mode:
        print("Manual drive mode: ON (startup)")
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
    status_cmd_enabled = False
    status_cmd_fwd = 0.0
    status_cmd_turn = 0.0
    status_cmd_duration = 0.0
    status_target_cell = None
    status_target_world = None
    direct_nav_enabled = False
    last_path_plan_time = 0.0
    last_auto_turn_cmd = 0.0
    last_auto_turn_time = time.time()
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
    landmark_memory = {"version": 1, "landmarks": []}
    landmark_dirty = False
    last_landmark_save = time.time()
    map_size_input_text = ""      # user-typed map size string e.g. "6x8" (feet)
    map_size_input_focused = False
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

    def start_localization_scan(reason="manual"):
        nonlocal localization_scan_active, localization_scan_started
        nonlocal localization_scan_started_lost, localization_scan_reason
        nonlocal emergency_stop, manual_mode, manual_fwd, manual_turn
        localization_scan_active = True
        localization_scan_started = time.time()
        localization_scan_started_lost = bool(tracking_enabled and not tracking_pose_ok)
        localization_scan_reason = reason
        emergency_stop = False
        manual_mode = False
        manual_fwd = 0.0
        manual_turn = 0.0
        print(f"Localize Scan started ({reason}). Rotate/look for known area-memory features and AI landmarks.")

    def stop_localization_scan(reason="done"):
        nonlocal localization_scan_active, localization_scan_reason
        if localization_scan_active:
            print(f"Localize Scan stopped ({reason}).")
        localization_scan_active = False
        localization_scan_reason = ""

    def update_localization_scan_state():
        if not localization_scan_active:
            return
        elapsed = time.time() - localization_scan_started
        if tracking_enabled and localization_scan_started_lost and tracking_pose_ok:
            stop_localization_scan("tracking relocked")
        elif tracking_pose_ok and elapsed >= max(0.5, float(args.localize_scan_sec)):
            stop_localization_scan("scan complete")
        elif (not tracking_pose_ok) and elapsed >= max(1.0, float(args.localize_max_sec)):
            stop_localization_scan("timeout")

    def draw_localization_banner(frame):
        if not HAS_CV2 or frame is None or not localization_scan_active:
            return
        h, w = frame.shape[:2]
        y0, y1 = 24, min(h, 48)
        if y1 <= y0:
            return
        cv2.rectangle(frame, (0, y0), (w, y1), (0, 70, 180), -1)
        if tracking_enabled:
            lock = "LOCKED" if tracking_pose_ok else "SEARCHING"
        else:
            lock = "TRACKING OFF"
        count = len(list(iter_visible_landmarks()))
        text = f"LOCALIZE: {lock} | landmarks {count} | L toggles"
        cv2.putText(frame, text, (6, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)

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
        print(f"Main rover drive mode: {'ON' if args.main_rover_mode else 'OFF'}")

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

        servo_angle_deg = float(sd.getNumber("Jetson/ServoAngleDeg", servo_angle_deg))
        servo_target_angle_deg = float(sd.getNumber("Jetson/ServoTargetAngleDeg", servo_target_angle_deg))
        servo_command_angle_deg = float(sd.getNumber("Jetson/ServoCommandAngleDeg", servo_command_angle_deg))
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
        next_seq = float(sd.getNumber("Jetson/ServoCommandSeq", 0.0)) + 1.0
        sd.putNumber("Jetson/ServoCommandAngleDeg", angle_deg)
        sd.putNumber("Jetson/ServoCommandSeq", next_seq)
        servo_command_angle_deg = angle_deg
        print(f"Camera servo -> {angle_deg:.0f} deg ({reason})")
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

    def reset_map_memory():
        nonlocal goal_cell, path_cells, last_path_cells, last_start, last_goal, last_path_plan_time
        nonlocal path_plan_mode
        nonlocal emergency_stop, reset_map_confirm, landmark_memory, landmark_dirty, last_save
        nonlocal lock_green_applied, lock_green_locked_count, mining_goal_active
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
        now = time.time()
        if (not force) and (now - last_map_ui_state_write) < 0.20:
            return

        button_enabled = mining_buttons_enabled()
        selected_tool = current_selected_tool()
        payload = {
            "available": True,
            "source": "zed_ground_wall",
            "timestamp_ms": int(now * 1000),
            "mining_state": mining.state.value,
            "localization_scan_active": bool(localization_scan_active),
            "landmark_count": int(len(landmark_memory.get("landmarks", []))),
            "selected_tool": selected_tool,
            "brush_radius": int(paint_brush_radius),
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
                    "id": "localize_scan",
                    "label": "Localize Scan",
                    "command": "localize_scan",
                    "active": bool(localization_scan_active),
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
                    "id": "pick_dig_start",
                    "label": "Pick Dig Start",
                    "command": "pick_dig_start",
                    "active": mining.state == auto_mining.MiningState.PICK_DIG_START
                              or mining.preferred_start_rc is not None,
                    "enabled": bool((not button_enabled and mining.state == auto_mining.MiningState.PICK_DIG_START)
                                    or (button_enabled and bool(mining.excav_corners_rc))),
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

    def on_map_click(event, x, y, flags, param):
        nonlocal goal_cell, path_cells, last_path_cells, last_start, last_goal
        nonlocal emergency_stop, last_path_plan_time, map_view_shift_r, map_view_shift_c, map_scale_live
        nonlocal path_plan_mode, mining_goal_active
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

    def on_status_click(event, x, y, flags, param):
        nonlocal disable_holes, whole_map_enabled, smooth_map_enabled, map_scale_live, map_size_input_focused, map_size_input_text
        nonlocal paint_safe_mode, erase_safe_mode, paint_obstacle_mode, paint_brush_radius
        nonlocal reset_map_confirm
        nonlocal manual_mode, manual_fwd, manual_turn, emergency_stop
        if event == getattr(cv2, "EVENT_MOUSEWHEEL", -9999):
            try:
                wheel_delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                wheel_delta = 1 if flags > 0 else -1
            set_status_scroll(-80 if wheel_delta > 0 else 80)
            return
        is_drag = event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)
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
        # Brush size slider supports drag.
        rect = status_button_rects.get("brush_slider")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                frac = (x - x0) / max(1, x1 - x0)
                paint_brush_radius = max(1, min(15, int(round(1 + frac * 14))))
                return
        if is_drag:
            return
        # Check if the map size input field was clicked
        rect = status_button_rects.get("map_size_input")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                map_size_input_focused = True
                map_size_input_text = ""
                return
            else:
                map_size_input_focused = False
        rect = status_button_rects.get("zoom_in")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                map_scale_live = min(12, map_scale_live + 1)
                print(f"Map zoom: x{map_scale_live}")
                return
        rect = status_button_rects.get("zoom_out")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                map_scale_live = max(1, map_scale_live - 1)
                print(f"Map zoom: x{map_scale_live}")
                return
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
                    mining.start_run()
                    print("Auto Run: START requested via button")
                return
        rect = status_button_rects.get("localize_scan")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                if localization_scan_active:
                    stop_localization_scan("button")
                else:
                    start_localization_scan("button")
                return
        rect = status_button_rects.get("direct_nav")
        if rect is not None:
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                set_direct_nav_enabled(not direct_nav_enabled, "button")
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
    def process_external_map_command():
        nonlocal last_map_command_seq, reset_map_confirm
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
        if action == "paint_safe":
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
        elif action == "localize_scan":
            if localization_scan_active:
                stop_localization_scan("external command")
            else:
                start_localization_scan("external command")
        elif action == "direct_nav":
            set_direct_nav_enabled(not direct_nav_enabled, "external command")
        elif action == "main_rover_mode":
            set_main_rover_mode(not args.main_rover_mode)
        elif action == "camera_view":
            toggle_camera_view()
        elif action == "draw_excav_zone":
            if mining_buttons_enabled():
                set_brush_tool(None)
                mining.start_draw_excavation()
        elif action == "draw_deposit_zone":
            if mining_buttons_enabled():
                set_brush_tool(None)
                mining.start_draw_deposit()
        elif action == "pick_dig_start":
            if mining_buttons_enabled() and mining.excav_corners_rc:
                set_brush_tool(None)
                mining.start_pick_dig_start()

        last_map_command_seq = seq

    def send_nt_command(enabled, fwd, turn, duration):
        nonlocal nt_command_seq, nt_ready_stuck_since, nt_last_auto_push
        nonlocal nt_ready_high, nt_ready_clear_time, last_drive_debug_time
        nonlocal status_cmd_enabled, status_cmd_fwd, status_cmd_turn, status_cmd_duration
        if sd is None:
            return
        now = time.time()
        enabled = bool(enabled)
        status_cmd_enabled = bool(enabled)
        status_cmd_fwd = float(fwd)
        status_cmd_turn = float(turn)
        status_cmd_duration = float(duration)

        def push_automation_state(force=False):
            nonlocal nt_last_auto_push
            if (not force) and (now - nt_last_auto_push) < max(0.02, float(args.nt_enable_heartbeat_sec)):
                return
            mining_state_value = mining.state.value
            excavator_enabled = enabled and mining.state == auto_mining.MiningState.DIGGING
            conveyor_enabled = enabled and mining.state == auto_mining.MiningState.DEPOSITING
            sd.putBoolean("Drive/UseMainRoverControls", bool(args.main_rover_mode))
            sd.putBoolean("Drive/MainRoverDebugMode", bool(args.main_rover_debug))
            sd.putBoolean("Drive/MainRoverEmergencyStop", False)
            sd.putBoolean("Jetson/AutomationEnabled", enabled)
            sd.putString("Jetson/MiningState", mining_state_value)
            sd.putBoolean("Jetson/ExcavatorEnabled", bool(excavator_enabled))
            sd.putBoolean("Jetson/ConveyorEnabled", bool(conveyor_enabled))
            # Robot-side code may scale command by these keys.
            if enabled:
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

    def mix_ds_drive(fwd, turn):
        if not args.ds_joystick:
            return float(fwd), float(turn)
        mixed_fwd = max(-1.0, min(1.0, float(fwd) + float(ds_joystick_fwd)))
        mixed_turn = max(-1.0, min(1.0, float(turn) + float(ds_joystick_turn)))
        return mixed_fwd, mixed_turn

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
        panel_h = 680
        panel_w = 620
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (24, 24, 24)
        status_button_rects.clear()

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
        elif localization_scan_active:
            mode_label = "LOCALIZE"
            mode_color = (0, 180, 255)
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
        put_line(
            f"Tracking: {track_txt} | AreaMem: {area_txt} | Follow: {'ON' if follow_rover_map else 'OFF'} (c)",
            150,
            track_color,
            0.52,
        )
        put_line(
            f"AI landmarks: {len(landmark_memory.get('landmarks', []))} saved | Localize Scan: {'ON' if localization_scan_active else 'OFF'} (l)",
            168,
            (255, 220, 170) if localization_scan_active else (190, 190, 190),
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
            186,
            servo_state_color,
            0.46,
        )

        excav_set = bool(mining.excav_corners_rc)
        deposit_set = bool(mining.deposit_corners_rc)
        put_line(f"Excavation zone: {'SET' if excav_set else 'unset'}", 210, (170, 255, 170) if excav_set else (190, 190, 190))
        put_line(f"Deposit zone: {'SET' if deposit_set else 'unset'}", 234, (170, 255, 170) if deposit_set else (190, 190, 190))
        put_line("Click a button below, then define 4 corners on the map.", 258, (210, 210, 210), 0.48)

        if goal_cell is None:
            put_line("Goal cell: none", 280, (190, 190, 190))
            put_line("Goal world: none", 304, (190, 190, 190))
        else:
            goal_world = occ_map.grid_to_world(goal_cell[0], goal_cell[1])
            put_line(f"Goal cell: r={goal_cell[0]} c={goal_cell[1]}", 280, (220, 240, 255))
            if goal_world is None:
                put_line("Goal world: unavailable", 304, (190, 190, 190))
            else:
                put_line(f"Goal world: x={goal_world[0]:+.2f} z={goal_world[1]:+.2f}", 304, (220, 240, 255))

        if status_target_world is None:
            put_line("Active target: none", 318, (190, 190, 190))
        else:
            tc = status_target_cell
            if tc is None:
                put_line(
                    f"Active target: x={status_target_world[0]:+.2f} z={status_target_world[1]:+.2f}",
                    318,
                    (255, 235, 170),
                )
            else:
                put_line(
                    f"Active target: r={tc[0]} c={tc[1]} x={status_target_world[0]:+.2f} z={status_target_world[1]:+.2f}",
                    318,
                    (255, 235, 170),
                )

        if cam_cell is None:
            put_line("Robot cell: unavailable", 342, (190, 190, 190))
        else:
            put_line(f"Robot cell: r={cam_cell[0]} c={cam_cell[1]}", 342, (180, 255, 220))

        put_line(f"Map zoom: x{map_scale_live}", 366, (220, 240, 255))
        put_line(
            f"Last command: {'ENABLED' if status_cmd_enabled else 'DISABLED'} dur={status_cmd_duration:.2f}s",
            390,
            (190, 255, 190) if status_cmd_enabled else (190, 190, 190),
        )
        draw_axis("Forward", status_cmd_fwd, 414)
        draw_axis("Turn", status_cmd_turn, 436)

        # --- Rover size input field (placed between axis bars and the zone buttons) ---
        cur_rover_ft = args.rover_size_m / 0.3048
        put_line(
            "Rover size (ft, square) — click field, type e.g. 2.5, press Enter",
            440,
            (170, 200, 230),
            0.44,
        )
        input_rect = (16, 450, panel_w - 16, 490)
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
            504,
            (200, 240, 255),
            0.48,
        )

        controls_top = 522
        controls_bottom = panel_h - 16
        controls_h = max(1, controls_bottom - controls_top)
        button_h = 42
        button_w = 160
        gap = 20
        x1 = 16 + button_w + gap
        x2 = 16 + 2 * (button_w + gap)
        row0 = 36
        row1 = row0 + button_h + 10
        row2 = row1 + button_h + 10
        row3 = row2 + button_h + 10
        row4 = row3 + button_h + 10
        row5 = row4 + button_h + 10
        row6 = row5 + button_h + 10
        row7 = row6 + button_h + 10
        slider_y = row7 + button_h + 20
        content_h = slider_y + 72
        controls = np.zeros((content_h, panel_w, 3), dtype=np.uint8)
        controls[:] = (28, 28, 28)

        def put_control_line(text, y, color=(235, 235, 235), scale=0.48):
            cv2.putText(controls, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

        def draw_control_button(rect, label, enabled, active=False, active_color=(70, 130, 220), active_border=(200, 200, 200)):
            x0, y0, x1b, y1b = rect
            fill = active_color if active else ((70, 130, 220) if enabled else (50, 50, 50))
            border = active_border if active else ((200, 200, 200) if enabled else (120, 120, 120))
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

        auto_run_rect = (16, row0, 16 + 2 * button_w + gap, row0 + button_h)
        localize_rect = (x2, row0, x2 + button_w, row0 + button_h)
        excav_rect = (16, row1, 16 + button_w, row1 + button_h)
        deposit_rect = (x1, row1, x1 + button_w, row1 + button_h)
        whole_rect = (x2, row1, x2 + button_w, row1 + button_h)
        obstacle_rect = (16, row2, 16 + button_w, row2 + button_h)
        paint_rect = (x1, row2, x1 + button_w, row2 + button_h)
        erase_rect = (x2, row2, x2 + button_w, row2 + button_h)
        smooth_rect = (16, row3, 16 + button_w, row3 + button_h)
        holes_rect = (x1, row3, x1 + button_w, row3 + button_h)
        clear_paint_rect = (x2, row3, x2 + button_w, row3 + button_h)
        reset_map_rect = (16, row4, 16 + button_w, row4 + button_h)
        lock_green_rect = (x1, row4, x1 + button_w, row4 + button_h)
        pick_dig_start_rect = (x2, row4, x2 + button_w, row4 + button_h)
        zoom_in_rect = (16, row5, 16 + button_w, row5 + button_h)
        zoom_out_rect = (x1, row5, x1 + button_w, row5 + button_h)
        main_rover_rect = (x2, row5, x2 + button_w, row5 + button_h)
        camera_view_rect = (16, row6, panel_w - 16, row6 + button_h)
        direct_nav_rect = (16, row7, panel_w - 16, row7 + button_h)

        auto_run_label = "Stop Auto Run" if _mining_active else "Start Auto Run"
        draw_control_button(auto_run_rect, auto_run_label, True, _mining_active, (0, 140, 40), (60, 240, 100))
        draw_control_button(localize_rect, "Localize: ON" if localization_scan_active else "Localize Scan",
                            True, localization_scan_active, (0, 120, 200), (80, 220, 255))
        excav_label = "Drawing Excav..." if excav_drawing else ("Excav Zone Set" if excav_set else "Draw Excav Zone")
        deposit_label = "Drawing Deposit..." if deposit_drawing else ("Deposit Zone Set" if deposit_set else "Draw Deposit Zone")
        draw_control_button(excav_rect, excav_label, zone_buttons_enabled, excav_drawing or excav_set, (0, 120, 220), (80, 200, 255))
        draw_control_button(deposit_rect, deposit_label, zone_buttons_enabled, deposit_drawing or deposit_set, (180, 150, 0), (255, 230, 80))
        draw_control_button(whole_rect, "Whole Map", button_enabled)
        draw_control_button(obstacle_rect, "Paint Obstacle: ON" if paint_obstacle_mode else "Paint Obstacle",
                            True, paint_obstacle_mode, (0, 0, 200), (80, 80, 255))
        draw_control_button(paint_rect, "Paint Safe: ON" if paint_safe_mode else "Paint Safe",
                            True, paint_safe_mode, (0, 180, 80), (80, 255, 140))
        draw_control_button(erase_rect, "Erase: ON" if erase_safe_mode else "Erase Safe",
                            True, erase_safe_mode, (0, 80, 200), (80, 140, 255))
        draw_control_button(smooth_rect, "Smooth Map: ON" if smooth_map_enabled else "Smooth Map",
                            button_enabled, smooth_map_enabled, (0, 160, 160), (80, 220, 220))
        draw_control_button(holes_rect, "Disable Holes", button_enabled)
        draw_control_button(clear_paint_rect, "Clear Paint", True)
        draw_control_button(reset_map_rect, "Reset Map", True, reset_map_confirm, (0, 70, 200), (80, 160, 255))
        lock_label = "Green Locked" if lock_green_applied else "Lock Green"
        draw_control_button(lock_green_rect, lock_label, True, lock_green_applied, (0, 160, 80), (80, 255, 140))
        pick_label = "Picking Start..." if picking_dig_start else (
            "Dig Start Set" if mining.preferred_start_rc is not None else "Pick Dig Start"
        )
        draw_control_button(
            pick_dig_start_rect,
            pick_label,
            zone_buttons_enabled and excav_set,
            picking_dig_start or mining.preferred_start_rc is not None,
            (0, 170, 70),
            (100, 255, 160),
        )
        draw_control_button(zoom_in_rect, "+ Zoom", True)
        draw_control_button(zoom_out_rect, "- Zoom", True)
        draw_control_button(
            main_rover_rect,
            "Main Rover: ON" if args.main_rover_mode else "Main Rover",
            True,
            args.main_rover_mode,
            (0, 120, 200),
            (80, 220, 255),
        )
        camera_label = "Camera: Deposit 0" if (
            servo_deposit_view
            or abs(angle_error_deg(servo_command_angle_deg, args.camera_deposit_angle_deg)) <= 2.0
        ) else "Camera: Map 180"
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
            direct_nav_rect,
            "Direct Nav: ON" if direct_nav_enabled else "Direct Nav",
            True,
            direct_nav_enabled,
            (0, 150, 90),
            (100, 255, 180),
        )

        btn_sm = 36
        brush_minus_rect = (16, slider_y + 4, 16 + btn_sm, slider_y + 4 + btn_sm)
        brush_plus_rect = (panel_w - 16 - btn_sm, slider_y + 4, panel_w - 16, slider_y + 4 + btn_sm)
        slider_x0 = brush_minus_rect[2] + 8
        slider_x1 = brush_plus_rect[0] - 8
        brush_slider_rect = (slider_x0, slider_y, slider_x1, slider_y + 44)
        put_control_line("Brush size:", slider_y - 8, (170, 200, 230), 0.44)
        cv2.rectangle(controls, (slider_x0, slider_y + 14), (slider_x1, slider_y + 30), (60, 60, 60), -1)
        cv2.rectangle(controls, (slider_x0, slider_y + 14), (slider_x1, slider_y + 30), (120, 120, 120), 1)
        frac = (paint_brush_radius - 1) / 14.0
        knob_x = int(slider_x0 + frac * (slider_x1 - slider_x0))
        cv2.circle(controls, (knob_x, slider_y + 22), 11, (100, 200, 255), -1)
        cv2.circle(controls, (knob_x, slider_y + 22), 11, (200, 240, 255), 1)
        put_control_line(f"Brush: {paint_brush_radius}", slider_y + 28, (220, 240, 255), 0.46)
        for rect, lbl in ((brush_minus_rect, "-"), (brush_plus_rect, "+")):
            x0b, y0b, x1b, y1b = rect
            cv2.rectangle(controls, (x0b, y0b), (x1b, y1b), (70, 130, 220), -1)
            cv2.rectangle(controls, (x0b, y0b), (x1b, y1b), (200, 200, 200), 1)
            tsz, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(controls, lbl, (x0b + (x1b - x0b - tsz[0]) // 2, y0b + (y1b - y0b + tsz[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        status_scroll_max = max(0, content_h - controls_h)
        status_scroll_y = max(0, min(status_scroll_y, status_scroll_max))
        panel[controls_top:controls_bottom, :] = controls[status_scroll_y:status_scroll_y + controls_h, :]
        cv2.rectangle(panel, (0, controls_top), (panel_w - 1, controls_bottom), (90, 90, 90), 1)

        for name, rect in (
            ("auto_run", auto_run_rect),
            ("localize_scan", localize_rect),
            ("direct_nav", direct_nav_rect),
            ("excav", excav_rect),
            ("deposit", deposit_rect),
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
            ("zoom_in", zoom_in_rect),
            ("zoom_out", zoom_out_rect),
            ("main_rover_mode", main_rover_rect),
            ("camera_view", camera_view_rect),
            ("brush_minus", brush_minus_rect),
            ("brush_plus", brush_plus_rect),
            ("brush_slider", brush_slider_rect),
        ):
            _register_button(name, rect)

        scroll_up_rect = (panel_w - 42, controls_top + 6, panel_w - 10, controls_top + 34)
        scroll_down_rect = (panel_w - 42, controls_bottom - 34, panel_w - 10, controls_bottom - 6)
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
            f"Controls scroll: {status_scroll_y}/{status_scroll_max} | Wheel or ^/v buttons",
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
            # Update camera pose (world frame) if tracking is enabled.
            if tracking_enabled:
                R_world_cam, t_world_cam, pose_warned, tracking_pose_ok = zed_utils.get_world_transform_with_status(
                    zed, sl, pose, pose_warned
                )
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
                    if jump_reason is not None:
                        tracking_pose_ok = False
                        tracking_recover_stable_count = 0
                        R_world_cam = last_valid_R_world_cam
                        t_world_cam = last_valid_t_world_cam
                        if not localization_scan_active:
                            start_localization_scan(jump_reason)
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
                    have_valid_tracking_pose = True
                    tracking_loss_warned = False
                    if not args.complex and not map_origin_set:
                        map_origin_t = np.array(t_world_cam, dtype=np.float32)
                        map_origin_set = True
                        print(
                            "Map origin anchored at "
                            f"x={map_origin_t[0]:+.2f}, y={map_origin_t[1]:+.2f}, z={map_origin_t[2]:+.2f}"
                        )
                else:
                    # Hold last known pose and pause map integration until tracking recovers.
                    if have_valid_tracking_pose:
                        R_world_cam = last_valid_R_world_cam
                        t_world_cam = last_valid_t_world_cam
                    if not tracking_loss_warned:
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
            auto_camera_map_required = localization_scan_active or goal_cell is not None or mining.state in (
                auto_mining.MiningState.PLAN_SWEEP,
                auto_mining.MiningState.NAVIGATE_DIG,
                auto_mining.MiningState.DIGGING,
                auto_mining.MiningState.BACKUP,
                auto_mining.MiningState.NAVIGATE_DEPOSIT,
                auto_mining.MiningState.DEPOSITING,
            )
            if auto_camera_map_required:
                request_camera_map_view("auto navigation")
                refresh_camera_servo_state()

            # Retrieve point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
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
                continue
            # Downsample for speed
            stride = max(1, int(args.sample_stride))
            xyz = cloud[::stride, ::stride, :3].reshape(-1, 3)
            # Filter invalid points
            mask = np.isfinite(xyz).all(axis=1)
            if float(args.min_range_z_m) > 0.0:
                mask &= xyz[:, 2] >= float(args.min_range_z_m)
            if float(args.max_range_z_m) > 0.0:
                mask &= xyz[:, 2] <= float(args.max_range_z_m)
            xyz = xyz[mask]

            if xyz.size == 0:
                no_points_count += 1
                if no_points_count % 30 == 1:
                    print(
                        "No valid depth points after filtering; "
                        "consider lowering --min-range-z-m or disabling range limits."
                    )
                dist = np.empty((0,), dtype=np.float32)
                ground_mask = np.zeros((0,), dtype=bool)
                obstacle_mask = np.zeros((0,), dtype=bool)
                hole_mask = np.zeros((0,), dtype=bool)
                ground_pct = 0.0
                obstacle_pct = 0.0
                hole_pct = 0.0
            else:
                no_points_count = 0
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
                camera_map_pause_reason = ""
                current_mount_yaw_deg = current_camera_mount_yaw_deg()
                _, _, close_obstacle_escape_sign = camera_mount_axes(current_mount_yaw_deg)
                # Never integrate new points while tracking is lost or a pose
                # jump was rejected; otherwise one bad pose can drag the map.
                map_integration_ok = (not tracking_enabled) or tracking_pose_ok
                if args.camera_servo_track and (servo_turning or not servo_map_view):
                    map_integration_ok = False
                    if servo_turning:
                        camera_map_pause_reason = "CAMERA TURNING"
                    else:
                        camera_map_pause_reason = "CAMERA DEPOSIT VIEW"
                # Compute map-local translation for simple mode
                if not args.complex and map_origin_set:
                    t_map = np.array(t_world_cam, dtype=np.float32) - map_origin_t
                else:
                    t_map = np.array(t_world_cam, dtype=np.float32)
                rover_pos_map, rover_forward_world, rover_right_world = rover_pose_from_camera(
                    R_world_cam,
                    t_map,
                    current_mount_yaw_deg,
                )
                cam_row_col = map_world_to_grid(t_map[0], t_map[2])
                rover_row_col = map_world_to_grid(rover_pos_map[0], rover_pos_map[2])
                if xyz.size > 0:
                    if map_integration_ok:
                        # Transform to world frame if tracking is enabled.
                        xyz_world = (R_world_cam @ xyz.T).T + t_map
                        x = -xyz_world[:, 0]
                        z = xyz_world[:, 2]
                        occ_map.update(x, z, ground_mask, obstacle_mask, hole_mask)

                        # Object detection: persist static objects, keep people as live-only markers.
                        if (rock_model is not None
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
                                        _ih, _iw = _img_bgr.shape[:2]
                                        _cld_h, _cld_w = cloud.shape[:2]
                                        for _det in (_results.boxes or []):
                                            _lbl = str(_names.get(int(_det.cls[0]), "")).lower()
                                            _conf = float(_det.conf[0]) if hasattr(_det, "conf") else float(args.rock_conf)
                                            _x1, _y1, _x2, _y2 = _det.xyxy[0].tolist()
                                            # Centre pixel of bounding box
                                            _cx = int((_x1 + _x2) / 2)
                                            _cy = int((_y1 + _y2) / 2)
                                            # Map pixel → point cloud index
                                            _pc_c = int(_cx * _cld_w / max(1, _iw))
                                            _pc_r = int(_cy * _cld_h / max(1, _ih))
                                            _pc_r = max(0, min(_cld_h - 1, _pc_r))
                                            _pc_c = max(0, min(_cld_w - 1, _pc_c))
                                            _pt = cloud[_pc_r, _pc_c, :3]
                                            if not np.isfinite(_pt).all():
                                                continue
                                            _pt_w = (R_world_cam @ _pt.astype(np.float32)) + t_map
                                            _rc = map_world_to_grid(_pt_w[0], _pt_w[2])
                                            if _rc is None:
                                                continue
                                            _rr, _cc = _rc
                                            if _lbl in rock_class_names:
                                                # Static object: persist on map
                                                _r0 = max(0, _rr - 1); _r1 = min(occ_map.grid_h - 1, _rr + 1)
                                                _c0 = max(0, _cc - 1); _c1 = min(occ_map.grid_w - 1, _cc + 1)
                                                occ_map.occ_counts[_r0:_r1+1, _c0:_c1+1] += float(args.rock_stamp)
                                                occ_map.free_counts[_r0:_r1+1, _c0:_c1+1] = 0.0
                                                if (not tracking_enabled) or tracking_pose_ok:
                                                    record_static_landmark(
                                                        _lbl,
                                                        map_x_from_zed(_pt_w[0]),
                                                        float(_pt_w[2]),
                                                        _conf,
                                                    )
                                            elif _lbl in {"person", "people", "human", "pedestrian"}:
                                                # Dynamic object: show as live marker only (handled elsewhere)
                                                pass
                            except Exception as _rock_err:
                                pass  # never crash the main loop on detection errors
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
                    if cam_row_col is not None:
                        r0, c0 = cam_row_col
                        half = max(1, int(args.map_camera_size) // 2)
                        r1 = max(0, r0 - half)
                        r2 = min(occ_map.grid_h, r0 + half + 1)
                        c1 = max(0, c0 - half)
                        c2 = min(occ_map.grid_w, c0 + half + 1)
                        map_vis[r1:r2, c1:c2, :] = (255, 0, 0)
                    if rover_row_col is not None:
                        r0, c0 = rover_row_col
                        cv2.circle(map_vis, (c0, r0), max(2, int(args.map_camera_size)), (0, 180, 255), -1)
                        heading_ang = None
                        if tracking_enabled:
                            forward = rover_forward_world
                            fx, fz = map_x_from_zed(forward[0]), float(forward[2])
                            heading_ang = np.arctan2(fz, fx)
                        # Draw rover footprint (orange outline), scaled by rover size.
                        rover_half_cells = max(1.0, float(args.rover_size_m) / (2.0 * float(occ_map.map_res_m)))
                        if heading_ang is None:
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
                            center = np.array([float(r0), float(c0)], dtype=np.float32)
                            fwd_v = np.array(
                                [-np.sin(heading_ang), np.cos(heading_ang)],
                                dtype=np.float32,
                            )
                            right_v = np.array(
                                [np.cos(heading_ang), np.sin(heading_ang)],
                                dtype=np.float32,
                            )
                            p1 = center + fwd_v * rover_half_cells + right_v * rover_half_cells
                            p2 = center + fwd_v * rover_half_cells - right_v * rover_half_cells
                            p3 = center - fwd_v * rover_half_cells - right_v * rover_half_cells
                            p4 = center - fwd_v * rover_half_cells + right_v * rover_half_cells
                            box_pts = np.array(
                                [
                                    [int(round(p1[1])), int(round(p1[0]))],
                                    [int(round(p2[1])), int(round(p2[0]))],
                                    [int(round(p3[1])), int(round(p3[0]))],
                                    [int(round(p4[1])), int(round(p4[0]))],
                                ],
                                dtype=np.int32,
                            )
                        cv2.polylines(map_vis, [box_pts], True, (0, 220, 255), 1, cv2.LINE_AA)
                        # Draw heading arrow (yellow) if tracking is enabled.
                        if tracking_enabled and HAS_CV2:
                            ang = heading_ang
                            size = max(8, int(args.map_camera_size) * 4)
                            start_pt = (int(c0), int(r0))
                            end_pt = (int(c0 + np.cos(ang) * size), int(r0 - np.sin(ang) * size))
                            cv2.arrowedLine(map_vis, start_pt, end_pt, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
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
                    if goal_cell is not None and rover_row_col is not None:
                        now = time.time()
                        should_replan = (
                            rover_row_col != last_start
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
                                    obs_try = map_utils.clear_mask_circle(obs_try, rover_row_col, clear_cells)
                                    keep_cost = map_utils.clear_mask_circle(
                                        np.ones(obs_try.shape, dtype=bool), rover_row_col, clear_cells
                                    )
                                    path_cost[~keep_cost] = 0.0

                                return map_utils.astar_path(
                                    rover_row_col,
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
                            last_start = rover_row_col
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
                    mining.render_status_banner(map_vis)
                    draw_localization_banner(map_vis)
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
                            # Read DS joystick axes from NT if enabled.
                            if args.ds_joystick and sd is not None:
                                _ds_scale = max(0.0, min(1.0, float(args.ds_joystick_scale)))
                                ds_joystick_fwd  = float(sd.getNumber(args.ds_joystick_fwd_key,  0.0)) * _ds_scale
                                ds_joystick_turn = float(sd.getNumber(args.ds_joystick_turn_key, 0.0)) * _ds_scale
                            else:
                                ds_joystick_fwd  = 0.0
                                ds_joystick_turn = 0.0
                            # Watchdog: NT telemetry lost — stop immediately.
                            _nt_timeout = float(args.nt_timeout_sec)
                            if _nt_timeout > 0 and (now - last_nt_ok_time) > _nt_timeout:
                                if not nt_watchdog_tripped:
                                    nt_watchdog_tripped = True
                                    print(f"[WATCHDOG] NT telemetry lost for >{_nt_timeout:.1f}s — stopping rover!")
                                send_nt_command(False, 0.0, 0.0, 0.1)
                                last_auto_turn_cmd = 0.0
                                last_auto_turn_time = now
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
                                last_auto_turn_cmd = 0.0
                                last_auto_turn_time = now
                                send_nt_command(False, 0.0, 0.0, 0.1)
                            elif human_hazard_state == "STOP":
                                # Person too close — hold still while A* replans around them.
                                last_auto_turn_cmd = 0.0
                                last_auto_turn_time = now
                                send_nt_command(False, 0.0, 0.0, 0.1)
                            elif now < backup_hold_until:
                                last_auto_turn_cmd = 0.0
                                last_auto_turn_time = now
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
                            elif localization_scan_active:
                                last_auto_turn_cmd = 0.0
                                last_auto_turn_time = now
                                turn_speed = max(-1.0, min(1.0, float(args.localize_turn_speed)))
                                if abs(turn_speed) < 0.05:
                                    turn_speed = 0.05
                                if (now - last_localization_log) >= 1.0:
                                    last_localization_log = now
                                    state_txt = "tracking OK" if tracking_pose_ok else "tracking lost"
                                    print(f"Localize Scan rotating in place ({state_txt}).")
                                send_nt_command(
                                    True,
                                    0.0,
                                    turn_speed,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                            elif manual_mode:
                                last_auto_turn_cmd = 0.0
                                last_auto_turn_time = now
                                # Driver Station joystick blends with ZED keyboard manual commands.
                                _man_fwd, _man_turn = mix_ds_drive(manual_fwd, manual_turn)
                                send_nt_command(
                                    True,
                                    _man_fwd,
                                    _man_turn,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                            elif tracking_enabled and (not tracking_pose_ok):
                                last_auto_turn_cmd = 0.0
                                last_auto_turn_time = now
                                # Keep robot safe while localization is uncertain.
                                send_nt_command(False, 0.0, 0.0, 0.1)
                            elif rover_row_col is None:
                                last_auto_turn_cmd = 0.0
                                last_auto_turn_time = now
                                send_nt_command(False, 0.0, 0.0, 0.1)
                            elif _mine_drive is not None:
                                last_auto_turn_cmd = 0.0
                                last_auto_turn_time = now
                                # Mining automation has direct drive control
                                # (DIGGING creep, BACKUP reverse, DEPOSITING reverse).
                                mine_fwd, mine_turn = mix_ds_drive(_mine_drive[0], _mine_drive[1])
                                send_nt_command(
                                    True,
                                    mine_fwd,
                                    mine_turn,
                                    1.0 / max(1.0, args.drive_rate_hz),
                                )
                            else:
                                reverse_path_drive = mining.state == auto_mining.MiningState.NAVIGATE_DEPOSIT
                                target_rc = pick_drive_target(draw_path, rover_row_col, goal_cell)
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
                                    last_auto_turn_cmd = 0.0
                                    last_auto_turn_time = now
                                    send_nt_command(False, 0.0, 0.0, 0.1)
                                    continue
                                else:
                                    last_auto_turn_cmd = 0.0
                                    last_auto_turn_time = now
                                    send_nt_command(False, 0.0, 0.0, 0.1)
                                    continue
                                # Auto-drive uses the robot/ZED physical X axis.
                                # The map view is mirrored for display, so convert map targets back.
                                cx, cz = float(rover_pos_map[0]), float(rover_pos_map[2])
                                tx_drive = zed_x_from_map(tx)
                                dx = tx_drive - cx
                                dz = tz - cz

                                # Stop if close enough to goal.
                                goal_world = occ_map.grid_to_world(goal_cell[0], goal_cell[1])
                                if goal_world is not None:
                                    gx, gz = goal_world
                                    gx_drive = zed_x_from_map(gx)
                                    if math.hypot(gx_drive - cx, gz - cz) <= args.drive_goal_tol_m:
                                        last_auto_turn_cmd = 0.0
                                        last_auto_turn_time = now
                                        send_nt_command(False, 0.0, 0.0, 0.1)
                                        continue

                                # Heading error in the physical X/Z frame, not the mirrored display frame.
                                forward = rover_forward_world
                                heading = math.atan2(float(forward[2]), float(forward[0]))
                                target = math.atan2(dz, dx)  # dz = target_z - curr_z, dx = target_x - curr_x
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

                                dt_turn = max(1e-3, now - last_auto_turn_time)
                                max_turn_step = max(0.0, float(args.drive_turn_slew_per_sec)) * dt_turn
                                delta_turn = turn_target - last_auto_turn_cmd
                                if delta_turn > max_turn_step:
                                    turn = last_auto_turn_cmd + max_turn_step
                                elif delta_turn < -max_turn_step:
                                    turn = last_auto_turn_cmd - max_turn_step
                                else:
                                    turn = turn_target
                                last_auto_turn_cmd = turn
                                last_auto_turn_time = now

                                # Reduce forward speed as heading error grows to avoid cutting sharp arcs.
                                slow_turn_rad = math.radians(max(0.0, float(args.drive_slow_turn_deg)))
                                stop_turn_rad = math.radians(max(0.0, float(args.drive_stop_turn_deg)))
                                if stop_turn_rad < slow_turn_rad:
                                    stop_turn_rad = slow_turn_rad

                                if stop_turn_rad <= 1e-6:
                                    turn_scale = 0.0 if err_abs > 0.0 else 1.0
                                elif err_abs >= stop_turn_rad:
                                    turn_scale = 0.0
                                elif err_abs <= slow_turn_rad:
                                    turn_scale = 1.0
                                else:
                                    turn_scale = (stop_turn_rad - err_abs) / max(1e-6, (stop_turn_rad - slow_turn_rad))

                                align_scale = max(0.0, math.cos(err))
                                fwd_mag = max(0.0, min(1.0, args.drive_speed)) * align_scale * max(0.0, min(1.0, turn_scale))
                                fwd = -fwd_mag if reverse_path_drive else fwd_mag

                                # Driver Station joystick can nudge auto commands.
                                fwd, turn = mix_ds_drive(fwd, turn)

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
                    map_vis = occ_map.render(whole_mode=whole_map_enabled)
                    if args.heatmap:
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
                    mining.render_status_banner(map_vis)
                    draw_localization_banner(map_vis)
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

                    overlay = img.copy()
                    # Green for ground (skip if red-only mode)
                    if not args.overlay_red_only:
                        overlay[ground_full == 1] = (0, 200, 0)
                    # Red for obstacles/walls
                    overlay[obstacle_full == 1] = (0, 0, 255)
                    # Blend
                    vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

                    # Human detection overlay
                    human_person_map_points = []
                    if human_detect_available and human_objects is not None and human_od_runtime is not None:
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
                    if human_detect_available and human_hazard_state != "CLEAR":
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

                    if camera_publisher is not None:
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
                    if map_publisher is not None:
                        map_publisher.push_frame(display_map_vis)
                    process_external_map_command()
                    publish_map_ui_state()

                if not args.no_gui:
                    # Always show the map (even if the image frame is missing)
                    if map_vis is not None:
                        # Draw detected persons on the map
                        draw_landmarks(map_vis)
                        for pr, pc_col, is_p in human_person_map_points:
                            draw_live_detection_marker(map_vis, pr, pc_col, is_p)
                        if map_red_only_view:
                            # Red-only map mode for easier obstacle inspection.
                            map_vis[:, :, 0] = 0
                            map_vis[:, :, 1] = 0
                        if map_scale_live > 1:
                            map_vis = cv2.resize(
                                map_vis,
                                (occ_map.grid_w * map_scale_live, occ_map.grid_h * map_scale_live),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        cv2.imshow("ZED Occupancy Map (XZ)", map_vis)
                    if display_map_vis is not None:
                        cv2.imshow("ZED Occupancy Map (XZ)", display_map_vis)
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
                if not args.no_gui:
                    status_panel = render_status_panel(rover_row_col)
                    cv2.imshow("ZED Drive Status", status_panel)
                    if not status_window_ready:
                        cv2.setMouseCallback("ZED Drive Status", on_status_click)
                        status_window_ready = True
                    key = cv2.waitKey(1) & 0xFF
                    # If map size input field is focused, route keys to it.
                    if map_size_input_focused:
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
                        elif key != 255:
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
                        manual_mode = not manual_mode
                        if manual_mode:
                            # Entering manual mode pauses auto navigation but keeps the last goal.
                            emergency_stop = False
                            manual_fwd = 0.0
                            manual_turn = 0.0
                            print("Manual drive mode: ON (auto paused)")
                        else:
                            print("Manual drive mode: OFF (auto resumed)")
                    if key == ord("c"):
                        follow_rover_map = not follow_rover_map
                        state = "ON" if follow_rover_map else "OFF"
                        print(f"Map follow mode: {state}")
                    if key == ord("v"):
                        map_red_only_view = not map_red_only_view
                        state = "ON" if map_red_only_view else "OFF"
                        print(f"Map red-only mode: {state}")
                    if key == ord("l"):
                        if localization_scan_active:
                            stop_localization_scan("key")
                        else:
                            start_localization_scan("key")
                    if key == ord("u"):
                        set_main_rover_mode(not args.main_rover_mode)
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
